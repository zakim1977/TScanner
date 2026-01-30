"""
ETF Money Flow Scoring Module

Combines volume-based indicators (OBV, MFI, CMF), price extension analysis,
and optional institutional data to determine money flow phase for ETFs/Stocks.

Used by the quality model to replace whale features for ETF/Stock markets.
"""

import pandas as pd
import numpy as np
from typing import Dict

from .money_flow import (
    calculate_money_flow,
    detect_accumulation_zone,
    detect_distribution_zone
)
from .indicators import calculate_ema


def calculate_etf_flow_score(
    df: pd.DataFrame,
    ticker: str = None,
    include_institutional: bool = False
) -> Dict:
    """
    Calculate ETF/Stock money flow score combining volume indicators,
    price extension, and optionally institutional data.

    Args:
        df: OHLCV DataFrame (needs 50+ candles for reliable signals)
        ticker: Symbol for institutional API lookup (optional)
        include_institutional: If True, fetch institutional data via API (live only)

    Returns:
        Dict with flow_score (-100 to +100), flow_phase, action, and components
    """
    default = {
        'flow_score': 0,
        'flow_phase': 'NEUTRAL',
        'obv_trend': 0,
        'mfi_value': 50.0,
        'cmf_value': 0.0,
        'price_extension_ema50': 0.0,
        'price_extension_ema200': 0.0,
        'in_accumulation_zone': False,
        'in_distribution_zone': False,
        'institutional_score': 0.0,
        'options_sentiment_score': 0.0,
        'action': 'HOLD',
    }

    if df is None or len(df) < 50:
        return default

    try:
        # Normalize column names — accept lowercase ('close') or capitalized ('Close')
        df = df.copy()
        df.columns = [c.lower().capitalize() if c.lower() in ('open', 'high', 'low', 'close', 'volume') else c for c in df.columns]

        if 'Close' not in df.columns or 'Volume' not in df.columns:
            return default

        score = 0.0

        # ─── Volume-based indicators ───
        mf = calculate_money_flow(df)

        obv_rising = mf.get('obv_rising', False)
        mfi_val = mf.get('mfi', 50.0)
        cmf_val = mf.get('cmf', 0.0)

        # OBV trend: ±20
        if obv_rising:
            score += 20
        else:
            score -= 20

        # MFI: map 0-100 to -20..+20 (50 = neutral)
        mfi_component = (mfi_val - 50) / 50 * 20
        score += mfi_component

        # CMF: map -1..+1 to -20..+20
        cmf_component = cmf_val * 20
        score += cmf_component

        # ─── Accumulation / Distribution zones ───
        acc_zone = detect_accumulation_zone(df)
        dist_zone = detect_distribution_zone(df)

        in_acc = acc_zone.get('in_accumulation', False)
        in_dist = dist_zone.get('in_distribution', False)

        if in_acc:
            score += 15
        if in_dist:
            score -= 15

        # ─── Price extension from EMAs ───
        close = df['Close'].values
        current_price = close[-1]

        ema50 = calculate_ema(df['Close'], 50)
        ema50_val = ema50.iloc[-1] if len(ema50) >= 50 else current_price

        ext_ema50 = ((current_price - ema50_val) / ema50_val) * 100 if ema50_val > 0 else 0

        if len(df) >= 200:
            ema200 = calculate_ema(df['Close'], 200)
            ema200_val = ema200.iloc[-1]
            ext_ema200 = ((current_price - ema200_val) / ema200_val) * 100 if ema200_val > 0 else 0
        else:
            ext_ema200 = ext_ema50 * 0.5  # rough estimate if not enough data

        # ─── Institutional data (live only) ───
        inst_score = 0.0
        options_score = 0.0

        if include_institutional and ticker:
            try:
                from .stock_institutional import get_stock_institutional_analysis
                inst_data = get_stock_institutional_analysis(ticker)

                # Institutional composite score: -100 to +100, scale to ±25
                raw_inst = inst_data.get('score', 0)
                inst_score = (raw_inst / 100) * 25
                score += inst_score

                # Options sentiment
                options = inst_data.get('options', {})
                pcr = options.get('put_call_ratio', 1.0)
                if pcr > 0:
                    # PCR < 0.7 = bullish, > 1.3 = bearish
                    options_score = max(-25, min(25, (1.0 - pcr) * 25))

            except Exception:
                pass  # Graceful degradation — institutional data is optional

        # ─── Clamp score ───
        flow_score = max(-100, min(100, score))

        # ─── Determine phase ───
        is_extended = ext_ema50 > 10 or ext_ema200 > 20
        is_very_extended = ext_ema200 > 20

        if is_extended:
            flow_phase = 'EXTENDED'
        elif flow_score >= 40:
            flow_phase = 'ACCUMULATING'
        elif flow_score <= -40:
            flow_phase = 'DISTRIBUTING'
        else:
            flow_phase = 'NEUTRAL'

        # ─── Action recommendation ───
        if flow_phase == 'EXTENDED' and is_very_extended:
            action = 'TRIM_15_20'
        elif flow_phase == 'EXTENDED' or flow_phase == 'DISTRIBUTING':
            action = 'TRIM_5_10'
        elif flow_phase == 'ACCUMULATING':
            action = 'ACCUMULATE_MORE'
        else:
            action = 'HOLD'

        return {
            'flow_score': round(flow_score, 1),
            'flow_phase': flow_phase,
            'obv_trend': 1 if obv_rising else 0,
            'mfi_value': round(mfi_val, 1),
            'cmf_value': round(cmf_val, 4),
            'price_extension_ema50': round(ext_ema50, 2),
            'price_extension_ema200': round(ext_ema200, 2),
            'in_accumulation_zone': in_acc,
            'in_distribution_zone': in_dist,
            'institutional_score': round(inst_score, 1),
            'options_sentiment_score': round(options_score, 1),
            'action': action,
        }

    except Exception as e:
        print(f"[ETF_FLOW] Error calculating flow score: {e}")
        return default
