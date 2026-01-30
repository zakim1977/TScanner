"""
Unified Analyzer
================
ONE function that calculates EVERYTHING once.
"""

import pandas as pd
from typing import Optional
import sys
import os

from core.models import (
    Result, WhaleData, ExplosionData, MLData, SMCData, FVGData,
    HTFData, MoneyFlow, TradeSetup, CombinedLearning, RulesDecision,
    Direction, Trade, Position, Confidence, T
)


def analyze(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market_type: str = "Crypto",
    trading_mode: str = "Day Trade",
    engine_mode: str = "hybrid",
    fetch_whale_api: bool = True,
) -> Result:
    """
    THE SINGLE ANALYSIS FUNCTION
    
    Call this ONCE. Use the Result everywhere.
    """
    
    result = Result(
        symbol=symbol,
        timeframe=timeframe,
        market_type=market_type,
        trading_mode=trading_mode,
        engine_mode=engine_mode,
    )
    
    if df is None or len(df) < 50:
        result.error = f"Need 50+ rows, got {len(df) if df is not None else 0}"
        return result
    
    try:
        result.df = df
        result.price = float(df['Close'].iloc[-1])
        
        # STEP 1: Whale Data
        result.whale = _get_whale_data(symbol, market_type, fetch_whale_api)
        
        # STEP 2: SMC - Order Blocks, FVG, Structure
        result.smc = _get_smc_data(df, result.price)
        
        # STEP 3: Explosion - ONCE!
        result.explosion = _get_explosion_data(df, result.whale)
        
        # STEP 4: Money Flow
        result.money_flow = _get_money_flow(df)
        
        # STEP 5: Position in Range
        result.position, result.position_pct = _get_position(
            result.price, result.smc.swing_low, result.smc.swing_high
        )
        
        # STEP 6: TA Score
        result.ta_score = _get_ta_score(df)
        
        # STEP 7: HTF Data
        if market_type == "Crypto":
            result.htf = _get_htf_data(symbol, timeframe, result.price)
        
        # STEP 8: MASTER_RULES Decision
        if engine_mode in ["rules", "hybrid"]:
            result.rules = _get_rules_decision(result)
        
        # STEP 9: ML Prediction
        if engine_mode in ["ml", "hybrid"]:
            result.ml = _get_ml_prediction(result)
        
        # STEP 10: Final Decision
        _make_final_decision(result)
        
        # STEP 11: Trade Setup
        result.setup = _get_trade_setup(df, result)
        
        # STEP 12: Combined Learning
        result.learning = _get_combined_learning(result)
        
        return result
        
    except Exception as e:
        import traceback
        result.error = f"{type(e).__name__}: {str(e)}"
        print(f"Analysis error: {traceback.format_exc()}")
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_whale_data(symbol: str, market_type: str, fetch_api: bool) -> WhaleData:
    """Get whale data"""
    whale = WhaleData()
    
    if market_type != "Crypto" or not fetch_api:
        return whale
    
    try:
        from core.whale_institutional import get_whale_analysis
        data = get_whale_analysis(symbol)
        
        if data:
            whale.oi_change_24h = data.get('open_interest', {}).get('change_24h', 0)
            whale.oi_change_1h = data.get('open_interest', {}).get('change_1h', 0)
            whale.funding_rate = data.get('funding', {}).get('rate', 0)
            whale.whale_pct = data.get('top_trader_ls', {}).get('long_pct', 50)
            whale.retail_pct = data.get('retail_ls', {}).get('long_pct', 50)
            whale.price_change_24h = data.get('price_change_24h', 0)
    except Exception as e:
        print(f"Whale data error: {e}")
    
    return whale


def _get_smc_data(df: pd.DataFrame, price: float) -> SMCData:
    """Get SMC data - OBs, FVG, Structure"""
    smc = SMCData()
    smc.swing_high = price * 1.05
    smc.swing_low = price * 0.95
    
    try:
        from core.smc_detector import detect_smc, detect_all_order_blocks
        
        result = detect_smc(df)
        if result:
            ob = result.get('order_blocks', {})
            smc.bullish_ob_top = ob.get('bullish_ob_top', 0)
            smc.bullish_ob_bottom = ob.get('bullish_ob_bottom', 0)
            smc.bearish_ob_top = ob.get('bearish_ob_top', 0)
            smc.bearish_ob_bottom = ob.get('bearish_ob_bottom', 0)
            smc.at_bullish_ob = ob.get('at_bullish_ob', False)
            smc.at_bearish_ob = ob.get('at_bearish_ob', False)
            smc.near_bullish_ob = ob.get('near_bullish_ob', False)
            smc.near_bearish_ob = ob.get('near_bearish_ob', False)
            
            fvg_data = result.get('fvg', {})
            smc.fvg = FVGData(
                bullish_fvg=fvg_data.get('bullish_fvg', False),
                bearish_fvg=fvg_data.get('bearish_fvg', False),
                bullish_fvg_top=fvg_data.get('bullish_fvg_high', 0),
                bullish_fvg_bottom=fvg_data.get('bullish_fvg_low', 0),
                bearish_fvg_top=fvg_data.get('bearish_fvg_high', 0),
                bearish_fvg_bottom=fvg_data.get('bearish_fvg_low', 0),
                at_bullish_fvg=fvg_data.get('at_bullish_fvg', False),
                at_bearish_fvg=fvg_data.get('at_bearish_fvg', False),
            )
            
            struct = result.get('structure', {})
            smc.swing_high = struct.get('last_swing_high', price * 1.05) or price * 1.05
            smc.swing_low = struct.get('last_swing_low', price * 0.95) or price * 0.95
            smc.structure = struct.get('structure', 'NEUTRAL')
            smc.bias = struct.get('bias', 'Neutral')
            
            liq = result.get('liquidity_sweep', {})
            smc.liquidity_swept_high = liq.get('swept_high', False)
            smc.liquidity_swept_low = liq.get('swept_low', False)
        
        all_obs = detect_all_order_blocks(df, price, lookback=50, max_obs=5)
        if all_obs:
            smc.bullish_obs = all_obs.get('bullish_obs', [])
            smc.bearish_obs = all_obs.get('bearish_obs', [])
            
    except Exception as e:
        print(f"SMC data error: {e}")
    
    return smc


def _get_explosion_data(df: pd.DataFrame, whale: WhaleData) -> ExplosionData:
    """Get explosion data - CALCULATED ONCE"""
    exp = ExplosionData()
    
    try:
        from core.explosion_detector import calculate_explosion_readiness, detect_market_state
        
        whale_dict = {
            'whale_long_pct': whale.whale_pct,
            'whale_short_pct': 100 - whale.whale_pct,
            'retail_pct': whale.retail_pct
        }
        
        result = calculate_explosion_readiness(df, whale_dict, whale.oi_change_24h)
        state = detect_market_state(df, whale_dict, whale.oi_change_24h)
        
        exp.score = result.get('explosion_score', 0)
        exp.ready = result.get('explosion_ready', False)
        exp.direction = result.get('direction')
        exp.state = state.get('state', 'UNKNOWN')
        exp.entry_valid = state.get('entry_valid', False)
        exp.signals = result.get('signals', [])
        
        squeeze = result.get('squeeze', {})
        exp.squeeze_pct = squeeze.get('squeeze_percentile', 50)
        exp.bb_squeeze = squeeze.get('bb_squeeze', 0)
        
    except Exception as e:
        print(f"Explosion data error: {e}")
    
    return exp


def _get_money_flow(df: pd.DataFrame) -> MoneyFlow:
    """Get money flow"""
    mf = MoneyFlow()
    
    try:
        from core.money_flow import calculate_money_flow
        
        result = calculate_money_flow(df)
        if result:
            mf.phase = result.get('flow_status', 'CONSOLIDATION')
            mf.is_accumulating = result.get('is_accumulating', False)
            mf.is_distributing = result.get('is_distributing', False)
            mf.flow_score = result.get('flow_score', 50)
    except Exception as e:
        print(f"Money flow error: {e}")
    
    return mf


def _get_position(price: float, swing_low: float, swing_high: float) -> tuple:
    """Calculate position in range"""
    if swing_high <= swing_low:
        return Position.MIDDLE, 50.0
    
    pct = ((price - swing_low) / (swing_high - swing_low)) * 100
    pct = max(0, min(100, pct))
    
    if pct <= T.EARLY_MAX:
        return Position.EARLY, pct
    elif pct >= T.LATE_MIN:
        return Position.LATE, pct
    return Position.MIDDLE, pct


def _get_ta_score(df: pd.DataFrame) -> int:
    """Get TA score"""
    try:
        from core.indicators import calculate_indicators
        result = calculate_indicators(df)
        if result and 'ta_score' in result:
            return int(result['ta_score'])
    except:
        pass
    
    try:
        close = float(df['Close'].iloc[-1])
        ema20 = df['Close'].rolling(20).mean().iloc[-1]
        ema50 = df['Close'].rolling(50).mean().iloc[-1]
        
        score = 50
        if close > ema20: score += 15
        if close > ema50: score += 15
        if ema20 > ema50: score += 10
        return min(100, max(0, score))
    except:
        return 50


def _get_htf_data(symbol: str, timeframe: str, price: float) -> HTFData:
    """Get higher timeframe data"""
    htf = HTFData()
    
    htf_map = {
        "1m": "15m", "5m": "1h", "15m": "4h",
        "1h": "4h", "4h": "1d", "1d": "1w"
    }
    htf.timeframe = htf_map.get(timeframe, "4h")
    
    try:
        from core.data_fetcher import fetch_binance_klines
        from core.smc_detector import detect_all_order_blocks
        from core.money_flow import calculate_money_flow
        
        htf_df = fetch_binance_klines(symbol, htf.timeframe, 100)
        
        if htf_df is not None and len(htf_df) >= 20:
            close = float(htf_df['Close'].iloc[-1])
            ema20 = htf_df['Close'].rolling(20).mean().iloc[-1]
            ema50 = htf_df['Close'].rolling(50).mean().iloc[-1] if len(htf_df) >= 50 else ema20
            
            if close > ema20 and ema20 > ema50:
                htf.structure = "BULLISH"
            elif close < ema20 and ema20 < ema50:
                htf.structure = "BEARISH"
            elif close > ema20:
                htf.structure = "LEAN_BULLISH"
            elif close < ema20:
                htf.structure = "LEAN_BEARISH"
            
            obs = detect_all_order_blocks(htf_df, price, lookback=50)
            if obs:
                htf.bullish_obs = obs.get('bullish_obs', [])
                htf.bearish_obs = obs.get('bearish_obs', [])
                htf.swing_high = obs.get('htf_swing_high', price * 1.1)
                htf.swing_low = obs.get('htf_swing_low', price * 0.9)
            
            mf = calculate_money_flow(htf_df)
            if mf:
                htf.money_flow_phase = mf.get('flow_status', 'CONSOLIDATION')
                
    except Exception as e:
        print(f"HTF data error: {e}")
    
    return htf


def _get_rules_decision(result: Result) -> Optional[RulesDecision]:
    """Get decision from MASTER_RULES"""
    try:
        from core.MASTER_RULES import get_trade_decision
        
        decision = get_trade_decision(
            whale_pct=result.whale.whale_pct,
            retail_pct=result.whale.retail_pct,
            oi_change=result.whale.oi_change_24h,
            price_change=result.whale.price_change_24h,
            position_pct=result.position_pct,
            ta_score=result.ta_score,
            money_flow_phase=result.money_flow.phase,
            at_bullish_ob=result.smc.at_bullish_ob,
            at_bearish_ob=result.smc.at_bearish_ob,
            near_bullish_ob=result.smc.near_bullish_ob,
            near_bearish_ob=result.smc.near_bearish_ob,
            at_support=result.smc.at_support,
            at_resistance=result.smc.at_resistance,
        )
        
        return RulesDecision(
            action=decision.action,
            trade_direction=decision.trade_direction,
            confidence=decision.confidence,
            direction_score=decision.direction_score,
            squeeze_score=decision.squeeze_score,
            entry_score=decision.entry_score,
            total_score=decision.total_score,
            direction_label=decision.direction_label,
            squeeze_label=decision.squeeze_label,
            position_label=decision.position_label,
            main_reason=decision.main_reason,
            warnings=decision.warnings,
            is_valid_long=decision.is_valid_long,
            is_valid_short=decision.is_valid_short,
            whale_story=decision.whale_story,
            oi_story=decision.oi_story,
            position_story=decision.position_story,
            conclusion=decision.conclusion,
            conclusion_action=decision.conclusion_action,
        )
    except Exception as e:
        print(f"Rules decision error: {e}")
        return None


def _get_ml_prediction(result: Result) -> Optional[MLData]:
    """Get ML prediction"""
    try:
        from ml.ml_engine import get_ml_prediction, is_ml_available
        
        if not is_ml_available():
            return None
        
        pred = get_ml_prediction(
            whale_pct=result.whale.whale_pct,
            retail_pct=result.whale.retail_pct,
            oi_change=result.whale.oi_change_24h,
            price_change_24h=result.whale.price_change_24h,
            position_pct=result.position_pct,
            current_price=result.price,
            swing_high=result.smc.swing_high,
            swing_low=result.smc.swing_low,
            ta_score=result.ta_score,
            trend=result.smc.structure,
            money_flow_phase=result.money_flow.phase,
        )
        
        if pred:
            return MLData(
                direction=pred.direction,
                confidence=pred.confidence if hasattr(pred, 'confidence') else 0,
                top_factors=pred.top_factors if hasattr(pred, 'top_factors') else [],
            )
    except Exception as e:
        print(f"ML prediction error: {e}")
    
    return None


def _make_final_decision(result: Result) -> None:
    """Make final trade decision"""
    whale = result.whale
    rules = result.rules
    ml = result.ml
    htf = result.htf
    exp = result.explosion
    
    if rules:
        if rules.trade_direction == "LONG":
            result.trade = Trade.LONG
        elif rules.trade_direction == "SHORT":
            result.trade = Trade.SHORT
        else:
            result.trade = Trade.WAIT
        result.action = rules.action
        result.reason = rules.main_reason
        result.warnings = rules.warnings.copy()
    
    if result.engine_mode == "hybrid" and ml:
        rules_bullish = whale.direction.is_bullish
        rules_bearish = whale.direction.is_bearish
        ml_long = ml.direction == "LONG"
        ml_short = ml.direction == "SHORT"
        
        if ml_long and rules_bullish:
            result.trade = Trade.LONG
            result.action = "STRONG_LONG" if ml.confidence > 70 else "LONG_SETUP"
            result.reason = f"ML + Rules aligned LONG"
        elif ml_short and rules_bearish:
            result.trade = Trade.SHORT
            result.action = "STRONG_SHORT" if ml.confidence > 70 else "SHORT_SETUP"
            result.reason = f"ML + Rules aligned SHORT"
        elif result.engine_mode == "ml":
            if ml_long:
                result.trade = Trade.LONG
            elif ml_short:
                result.trade = Trade.SHORT
            else:
                result.trade = Trade.WAIT
    
    if htf.timeframe:
        if htf.is_bullish and result.trade == Trade.SHORT:
            result.warnings.append(f"⚠️ HTF ({htf.timeframe}) is BULLISH - counter-trend short")
        elif htf.is_bearish and result.trade == Trade.LONG:
            result.warnings.append(f"⚠️ HTF ({htf.timeframe}) is BEARISH - counter-trend long")
    
    if exp.ready and exp.direction:
        result.setup_type = f"EXPLOSION_{exp.direction}"
        if exp.direction == "LONG" and result.trade != Trade.SHORT:
            result.trade = Trade.LONG
            result.action = "EXPLOSION_LONG"
            result.reason = f"Explosion ready: {exp.score}/100"
        elif exp.direction == "SHORT" and result.trade != Trade.LONG:
            result.trade = Trade.SHORT
            result.action = "EXPLOSION_SHORT"
            result.reason = f"Explosion ready: {exp.score}/100"
    
    if whale.is_trap and result.trade == Trade.LONG:
        result.trade = Trade.WAIT
        result.action = "AVOID_TRAP"
        result.reason = f"Retail trap: R:{whale.retail_pct:.0f}% > W:{whale.whale_pct:.0f}%"


def _get_trade_setup(df: pd.DataFrame, result: Result) -> TradeSetup:
    """Get trade setup with levels"""
    setup = TradeSetup(
        direction=result.trade.value,
        entry=result.price,
    )
    
    if result.trade == Trade.WAIT:
        return setup
    
    try:
        from core.signal_generator import SignalGeneratorV2
        
        signal = SignalGeneratorV2.generate_signal(
            df=df,
            symbol=result.symbol,
            timeframe=result.timeframe,
            force_direction=result.trade.value
        )
        
        if signal:
            setup.entry = signal.entry
            setup.stop_loss = signal.stop_loss
            setup.tp1 = signal.tp1
            setup.tp2 = signal.tp2
            setup.tp3 = signal.tp3
            setup.entry_reason = signal.entry_reason if hasattr(signal, 'entry_reason') else ""
            setup.sl_reason = signal.sl_reason if hasattr(signal, 'sl_reason') else ""
            setup.strategy = signal.strategy if hasattr(signal, 'strategy') else ""
            
            if setup.stop_loss > 0:
                risk = abs(setup.entry - setup.stop_loss)
                if risk > 0:
                    setup.rr1 = abs(setup.tp1 - setup.entry) / risk
                    setup.rr2 = abs(setup.tp2 - setup.entry) / risk
                    setup.rr3 = abs(setup.tp3 - setup.entry) / risk
            
            return setup
    except Exception as e:
        print(f"Signal generator error: {e}")
    
    try:
        atr = _calculate_atr(df)
        
        if result.trade == Trade.LONG:
            setup.stop_loss = result.price - (atr * 1.5)
            setup.tp1 = result.price + (atr * 1.5)
            setup.tp2 = result.price + (atr * 2.5)
            setup.tp3 = result.price + (atr * 4.0)
        elif result.trade == Trade.SHORT:
            setup.stop_loss = result.price + (atr * 1.5)
            setup.tp1 = result.price - (atr * 1.5)
            setup.tp2 = result.price - (atr * 2.5)
            setup.tp3 = result.price - (atr * 4.0)
        
        risk = abs(setup.entry - setup.stop_loss)
        if risk > 0:
            setup.rr1 = abs(setup.tp1 - setup.entry) / risk
            setup.rr2 = abs(setup.tp2 - setup.entry) / risk
            setup.rr3 = abs(setup.tp3 - setup.entry) / risk
    except:
        pass
    
    return setup


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR"""
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    return float(tr.rolling(period).mean().iloc[-1])


def _get_combined_learning(result: Result) -> CombinedLearning:
    """Get combined learning"""
    learning = CombinedLearning()
    
    try:
        from core.unified_scoring import generate_combined_learning
        
        cl = generate_combined_learning(
            signal_name=result.action,
            direction=result.direction_label,
            whale_pct=result.whale.whale_pct,
            retail_pct=result.whale.retail_pct,
            oi_change=result.whale.oi_change_24h,
            price_change=result.whale.price_change_24h,
            money_flow_phase=result.money_flow.phase,
            structure_type=result.smc.structure,
            position_pct=result.position_pct,
            ta_score=result.ta_score,
            at_bullish_ob=result.smc.at_bullish_ob,
            at_bearish_ob=result.smc.at_bearish_ob,
            near_bullish_ob=result.smc.near_bullish_ob,
            near_bearish_ob=result.smc.near_bearish_ob,
            final_verdict=result.trade.value,
            final_verdict_reason=result.reason,
            ml_prediction=result.ml.direction if result.ml else None,
            engine_mode=result.engine_mode,
            explosion_data={
                'score': result.explosion.score,
                'ready': result.explosion.ready,
                'direction': result.explosion.direction,
                'state': result.explosion.state,
                'entry_valid': result.explosion.entry_valid,
                'squeeze_pct': result.explosion.squeeze_pct,
            }
        )
        
        if cl:
            learning.conclusion = cl.get('conclusion', '')
            learning.conclusion_action = cl.get('conclusion_action', 'WAIT')
            learning.stories = cl.get('stories', [])
            
            learning.layer1_direction = cl.get('direction', 'NEUTRAL')
            learning.layer1_score = cl.get('direction_score', 15)
            learning.layer1_confidence = cl.get('confidence', 'LOW')
            
            learning.layer2_squeeze = cl.get('squeeze_label', 'NONE')
            learning.layer2_score = cl.get('squeeze_score', 0)
            
            learning.layer3_entry = cl.get('entry_timing', 'WAIT')
            learning.layer3_score = cl.get('entry_score', 10)
            
            learning.is_squeeze = cl.get('is_squeeze', False)
            learning.has_conflict = cl.get('has_conflict', False)
            learning.conflicts = cl.get('conflicts', [])
            
    except Exception as e:
        print(f"Combined learning error: {e}")
        
        learning.conclusion = result.rules.conclusion if result.rules else result.summary
        learning.conclusion_action = result.trade.value
        
        learning.layer1_direction = result.direction_label
        learning.layer1_score = result.direction_score
        learning.layer1_confidence = result.confidence.value
        
        learning.layer2_squeeze = result.squeeze_label
        learning.layer2_score = result.squeeze_score
        
        learning.layer3_entry = "NOW" if result.timing_score >= 20 else "SOON" if result.timing_score >= 12 else "WAIT"
        learning.layer3_score = result.timing_score
        
        stories = []
        stories.append(("🐋 Whale Positioning", f"Whales {result.whale.whale_pct:.0f}% vs Retail {result.whale.retail_pct:.0f}%"))
        if result.whale.oi_change_24h != 0:
            stories.append(("📊 Open Interest", f"OI {result.whale.oi_change_24h:+.1f}% 24h"))
        if result.explosion.score >= 50:
            stories.append(("💥 Explosion", f"Score {result.explosion.score}/100 - {result.explosion.state}"))
        stories.append(("📍 Position", f"{result.position.value} ({result.position_pct:.0f}% in range)"))
        if result.ml:
            stories.append(("🤖 ML Prediction", f"{result.ml.direction} ({result.ml.confidence:.0f}%)"))
        learning.stories = stories
    
    return learning
