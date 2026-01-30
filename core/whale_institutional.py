"""
🐋 WHALE & INSTITUTIONAL DATA MODULE
=====================================
Fetches real-time whale/institutional data from Binance Futures API
Provides clear interpretation of what the data means

FREE DATA FROM BINANCE:
- Open Interest (OI)
- Funding Rate
- Long/Short Ratio (Retail)
- Long/Short Ratio (Top Traders = Whales)
- Taker Buy/Sell Volume
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_SPOT_BASE = "https://api.binance.com/api/v3"
TIMEOUT = 10

# Disable SSL warnings for corporate environments
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class WhaleSignal(Enum):
    STRONG_ACCUMULATION = "🟢 STRONG ACCUMULATION"
    ACCUMULATION = "🟢 ACCUMULATION"
    WEAK_ACCUMULATION = "🟡 WEAK ACCUMULATION"
    NEUTRAL = "⚪ NEUTRAL"
    WEAK_DISTRIBUTION = "🟠 WEAK DISTRIBUTION"
    DISTRIBUTION = "🔴 DISTRIBUTION"
    STRONG_DISTRIBUTION = "🔴 STRONG DISTRIBUTION"


@dataclass
class InstitutionalData:
    """Complete institutional data snapshot"""
    symbol: str
    timestamp: datetime
    
    # Open Interest
    open_interest: float
    oi_change_24h: float
    oi_trend: str  # "RISING", "FALLING", "STABLE"
    
    # Funding Rate
    funding_rate: float
    funding_status: str  # "EXTREME_BULLISH", "BULLISH", "NEUTRAL", etc.
    
    # Long/Short Ratios
    retail_long_pct: float
    retail_short_pct: float
    top_trader_long_pct: float
    top_trader_short_pct: float
    
    # Taker Volume
    taker_buy_ratio: float
    taker_status: str  # "AGGRESSIVE_BUYING", "AGGRESSIVE_SELLING", "BALANCED"
    
    # Overall verdict
    verdict: str
    confidence: int
    signals: list
    
    # Human-readable explanation
    explanation: str


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def interpret_oi_price(oi_change: float, price_change: float) -> Dict:
    """
    Interpret Open Interest + Price relationship
    
    This is CRUCIAL for understanding whale behavior:
    - OI ↑ + Price ↑ = New LONGS entering (bullish continuation)
    - OI ↑ + Price ↓ = New SHORTS entering (bearish continuation)
    - OI ↓ + Price ↑ = SHORT COVERING (weak rally, may reverse)
    - OI ↓ + Price ↓ = LONG LIQUIDATION (weak dump, may reverse)
    """
    if oi_change > 3:  # OI rising significantly
        if price_change > 1:
            return {
                'signal': 'NEW_LONGS',
                'emoji': '🟢',
                'interpretation': 'New LONGS entering - Bullish continuation expected',
                'action': 'LONG bias - Follow the trend',
                'strength': 'STRONG' if oi_change > 8 else 'MODERATE'
            }
        elif price_change < -1:
            return {
                'signal': 'NEW_SHORTS',
                'emoji': '🔴',
                'interpretation': 'New SHORTS entering - Bearish continuation expected',
                'action': 'SHORT bias - Follow the trend',
                'strength': 'STRONG' if oi_change > 8 else 'MODERATE'
            }
        else:
            return {
                'signal': 'BUILDING_POSITIONS',
                'emoji': '🟡',
                'interpretation': 'Positions building but direction unclear',
                'action': 'WAIT for breakout direction',
                'strength': 'MODERATE'
            }
    
    elif oi_change < -3:  # OI falling significantly
        if price_change > 1:
            return {
                'signal': 'SHORT_COVERING',
                'emoji': '🟠',
                'interpretation': 'SHORT COVERING - Rally may be weak',
                'action': 'CAUTION on longs - Not new buying, just shorts closing',
                'strength': 'WEAK'
            }
        elif price_change < -1:
            return {
                'signal': 'LONG_LIQUIDATION',
                'emoji': '🟠',
                'interpretation': 'LONG LIQUIDATION - Dump may be ending',
                'action': 'WATCH for reversal - Forced selling, not new shorts',
                'strength': 'WEAK'
            }
        else:
            return {
                'signal': 'CLOSING_POSITIONS',
                'emoji': '⚪',
                'interpretation': 'Traders closing positions - Reduced interest',
                'action': 'WAIT - Low conviction environment',
                'strength': 'LOW'
            }
    
    else:  # OI stable
        return {
            'signal': 'STABLE',
            'emoji': '⚪',
            'interpretation': 'OI stable - No significant position changes',
            'action': 'Follow price action',
            'strength': 'NEUTRAL'
        }


def interpret_funding(funding_rate: float) -> Dict:
    """
    Interpret Funding Rate
    
    Funding is a contrarian indicator:
    - Very high funding (>0.1%) = Longs overleveraged → Dump likely
    - Very low funding (<-0.1%) = Shorts overleveraged → Pump likely
    """
    if funding_rate > 0.001:  # > 0.1%
        return {
            'signal': 'EXTREME_BULLISH',
            'emoji': '⚠️',
            'interpretation': f'Funding EXTREME ({funding_rate*100:.3f}%) - Longs overleveraged',
            'action': '🔴 CONTRARIAN SHORT - Dump likely, longs will get liquidated',
            'is_contrarian': True
        }
    elif funding_rate > 0.0005:  # > 0.05%
        return {
            'signal': 'BULLISH',
            'emoji': '🟢',
            'interpretation': f'Funding positive ({funding_rate*100:.3f}%) - Market leaning bullish',
            'action': 'Slight long bias, but watch for overleveraging',
            'is_contrarian': False
        }
    elif funding_rate < -0.001:  # < -0.1%
        return {
            'signal': 'EXTREME_BEARISH',
            'emoji': '⚠️',
            'interpretation': f'Funding NEGATIVE ({funding_rate*100:.3f}%) - Shorts overleveraged',
            'action': '🟢 CONTRARIAN LONG - Pump likely, shorts will get squeezed',
            'is_contrarian': True
        }
    elif funding_rate < -0.0005:
        return {
            'signal': 'BEARISH',
            'emoji': '🔴',
            'interpretation': f'Funding negative ({funding_rate*100:.3f}%) - Market leaning bearish',
            'action': 'Slight short bias',
            'is_contrarian': False
        }
    else:
        return {
            'signal': 'NEUTRAL',
            'emoji': '⚪',
            'interpretation': f'Funding neutral ({funding_rate*100:.3f}%)',
            'action': 'No funding edge - Follow other signals',
            'is_contrarian': False
        }


def interpret_long_short(retail_long: float, top_trader_long: float) -> Dict:
    """
    Interpret Long/Short Ratios
    
    KEY INSIGHT: Fade retail, follow top traders
    """
    # Determine if there's a divergence
    retail_bias = 'LONG' if retail_long > 55 else 'SHORT' if retail_long < 45 else 'NEUTRAL'
    whale_bias = 'LONG' if top_trader_long > 55 else 'SHORT' if top_trader_long < 45 else 'NEUTRAL'
    
    if retail_long > 65 and top_trader_long < 50:
        # Retail very long, whales not → FADE RETAIL
        return {
            'signal': 'FADE_RETAIL_LONGS',
            'emoji': '🔴',
            'interpretation': f'Retail {retail_long:.0f}% LONG but Top Traders {top_trader_long:.0f}% long',
            'action': '🔴 FADE RETAIL - Go SHORT, whales are not with retail',
            'retail_bias': retail_bias,
            'whale_bias': whale_bias,
            'edge': 'HIGH'
        }
    elif retail_long < 35 and top_trader_long > 50:
        # Retail very short, whales long → FADE RETAIL
        return {
            'signal': 'FADE_RETAIL_SHORTS',
            'emoji': '🟢',
            'interpretation': f'Retail {100-retail_long:.0f}% SHORT but Top Traders {top_trader_long:.0f}% long',
            'action': '🟢 FADE RETAIL - Go LONG, whales are buying',
            'retail_bias': retail_bias,
            'whale_bias': whale_bias,
            'edge': 'HIGH'
        }
    elif top_trader_long > 60:
        # Whales clearly long
        return {
            'signal': 'FOLLOW_WHALES_LONG',
            'emoji': '🟢',
            'interpretation': f'Top Traders {top_trader_long:.0f}% LONG - Smart money bullish',
            'action': '🟢 FOLLOW WHALES - Go LONG',
            'retail_bias': retail_bias,
            'whale_bias': whale_bias,
            'edge': 'MEDIUM'
        }
    elif top_trader_long < 40:
        # Whales clearly short
        return {
            'signal': 'FOLLOW_WHALES_SHORT',
            'emoji': '🔴',
            'interpretation': f'Top Traders {100-top_trader_long:.0f}% SHORT - Smart money bearish',
            'action': '🔴 FOLLOW WHALES - Go SHORT',
            'retail_bias': retail_bias,
            'whale_bias': whale_bias,
            'edge': 'MEDIUM'
        }
    else:
        return {
            'signal': 'NO_CLEAR_EDGE',
            'emoji': '⚪',
            'interpretation': f'Retail {retail_long:.0f}% long, Top Traders {top_trader_long:.0f}% long',
            'action': 'No clear positioning edge',
            'retail_bias': retail_bias,
            'whale_bias': whale_bias,
            'edge': 'LOW'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# API FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_open_interest(symbol: str) -> Dict:
    """Fetch current and historical Open Interest"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        # Current OI
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        response = requests.get(url, params={'symbol': symbol}, timeout=TIMEOUT, verify=False)
        
        if response.status_code != 200:
            return {'current': 0, 'change_24h': 0, 'trend': 'UNAVAILABLE', 'error': f'API returned {response.status_code}'}
        
        current_oi = float(response.json().get('openInterest', 0))
        
        # Historical OI (24h)
        url_hist = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
        response_hist = requests.get(url_hist, params={'symbol': symbol, 'period': '1h', 'limit': 24}, 
                                     timeout=TIMEOUT, verify=False)
        hist_data = response_hist.json() if response_hist.status_code == 200 else []
        
        if hist_data and len(hist_data) > 0:
            oi_24h_ago = float(hist_data[0].get('sumOpenInterest', current_oi))
            oi_change = ((current_oi - oi_24h_ago) / oi_24h_ago * 100) if oi_24h_ago > 0 else 0
        else:
            oi_change = 0
        
        return {
            'current': current_oi,
            'change_24h': oi_change,
            'trend': 'RISING' if oi_change > 5 else 'FALLING' if oi_change < -5 else 'STABLE',
            'available': True
        }
    except Exception as e:
        return {'current': 0, 'change_24h': 0, 'trend': 'UNAVAILABLE', 'error': str(e), 'available': False}


def fetch_funding_rate(symbol: str) -> Dict:
    """Fetch current funding rate"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        response = requests.get(url, params={'symbol': symbol, 'limit': 1}, timeout=TIMEOUT, verify=False)
        data = response.json()
        
        if data and len(data) > 0:
            return {
                'rate': float(data[-1].get('fundingRate', 0)),
                'time': data[-1].get('fundingTime', 0)
            }
        return {'rate': 0, 'time': 0}
    except Exception as e:
        return {'rate': 0, 'time': 0, 'error': str(e)}


def fetch_long_short_ratio(symbol: str, period: str = '1h') -> Dict:
    """Fetch retail long/short ratio"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        url = f"{BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio"
        response = requests.get(url, params={'symbol': symbol, 'period': period, 'limit': 1}, 
                               timeout=TIMEOUT, verify=False)
        data = response.json()
        
        if data and len(data) > 0:
            return {
                'long_ratio': float(data[-1].get('longAccount', 0.5)) * 100,
                'short_ratio': float(data[-1].get('shortAccount', 0.5)) * 100
            }
        return {'long_ratio': 50, 'short_ratio': 50}
    except Exception as e:
        return {'long_ratio': 50, 'short_ratio': 50, 'error': str(e)}


def fetch_top_trader_ratio(symbol: str, period: str = '1h') -> Dict:
    """Fetch TOP TRADER (whale) long/short ratio - THIS IS THE ALPHA"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        url = f"{BINANCE_FUTURES_BASE}/futures/data/topLongShortPositionRatio"
        response = requests.get(url, params={'symbol': symbol, 'period': period, 'limit': 1}, 
                               timeout=TIMEOUT, verify=False)
        data = response.json()
        
        if data and len(data) > 0:
            return {
                'long_ratio': float(data[-1].get('longAccount', 0.5)) * 100,
                'short_ratio': float(data[-1].get('shortAccount', 0.5)) * 100
            }
        return {'long_ratio': 50, 'short_ratio': 50}
    except Exception as e:
        return {'long_ratio': 50, 'short_ratio': 50, 'error': str(e)}


def fetch_taker_buy_sell(symbol: str, period: str = '1h') -> Dict:
    """Fetch taker buy/sell volume ratio"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        url = f"{BINANCE_FUTURES_BASE}/futures/data/takerlongshortRatio"
        response = requests.get(url, params={'symbol': symbol, 'period': period, 'limit': 10}, 
                               timeout=TIMEOUT, verify=False)
        data = response.json()
        
        if data and len(data) > 0:
            avg_ratio = sum(float(d.get('buySellRatio', 1)) for d in data) / len(data)
            latest = float(data[-1].get('buySellRatio', 1))
            return {
                'ratio': latest,
                'avg_ratio': avg_ratio,
                'status': 'AGGRESSIVE_BUYING' if latest > 1.2 else 'AGGRESSIVE_SELLING' if latest < 0.8 else 'BALANCED'
            }
        return {'ratio': 1, 'avg_ratio': 1, 'status': 'UNKNOWN'}
    except Exception as e:
        return {'ratio': 1, 'avg_ratio': 1, 'status': 'ERROR', 'error': str(e)}


def fetch_price_change(symbol: str) -> float:
    """Fetch 24h price change percentage"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        symbol = symbol + 'USDT'
    
    try:
        url = f"{BINANCE_SPOT_BASE}/ticker/24hr"
        response = requests.get(url, params={'symbol': symbol}, timeout=TIMEOUT, verify=False)
        data = response.json()
        return float(data.get('priceChangePercent', 0))
    except:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_institutional_analysis(symbol: str) -> Dict:
    """
    Get complete institutional analysis with human-readable explanations
    
    Returns everything needed to understand what whales are doing
    """
    # Fetch all data
    oi_data = fetch_open_interest(symbol)
    funding_data = fetch_funding_rate(symbol)
    retail_ls = fetch_long_short_ratio(symbol)
    top_trader_ls = fetch_top_trader_ratio(symbol)
    taker_data = fetch_taker_buy_sell(symbol)
    price_change = fetch_price_change(symbol)
    
    # Interpret data
    oi_interp = interpret_oi_price(oi_data.get('change_24h', 0), price_change)
    funding_interp = interpret_funding(funding_data.get('rate', 0))
    ls_interp = interpret_long_short(
        retail_ls.get('long_ratio', 50), 
        top_trader_ls.get('long_ratio', 50)
    )
    
    # Calculate scores
    bullish_score = 0
    bearish_score = 0
    signals = []
    
    # OI + Price interpretation
    if oi_interp['signal'] == 'NEW_LONGS':
        bullish_score += 30
        signals.append(f"📈 {oi_interp['interpretation']}")
    elif oi_interp['signal'] == 'NEW_SHORTS':
        bearish_score += 30
        signals.append(f"📉 {oi_interp['interpretation']}")
    elif oi_interp['signal'] in ['SHORT_COVERING', 'LONG_LIQUIDATION']:
        signals.append(f"⚠️ {oi_interp['interpretation']}")
    
    # Funding interpretation (contrarian)
    if funding_interp['is_contrarian']:
        if funding_interp['signal'] == 'EXTREME_BULLISH':
            bearish_score += 25
            signals.append(f"💰 {funding_interp['interpretation']}")
        elif funding_interp['signal'] == 'EXTREME_BEARISH':
            bullish_score += 25
            signals.append(f"💰 {funding_interp['interpretation']}")
    
    # Long/Short interpretation
    if ls_interp['edge'] == 'HIGH':
        if 'LONG' in ls_interp['action']:
            bullish_score += 35
        else:
            bearish_score += 35
        signals.append(f"🐋 {ls_interp['interpretation']}")
    elif ls_interp['edge'] == 'MEDIUM':
        if 'LONG' in ls_interp['action']:
            bullish_score += 20
        else:
            bearish_score += 20
        signals.append(f"🐋 {ls_interp['interpretation']}")
    
    # Taker volume
    if taker_data['status'] == 'AGGRESSIVE_BUYING':
        bullish_score += 20
        signals.append(f"🔥 Aggressive BUYING (taker ratio: {taker_data['ratio']:.2f})")
    elif taker_data['status'] == 'AGGRESSIVE_SELLING':
        bearish_score += 20
        signals.append(f"🔥 Aggressive SELLING (taker ratio: {taker_data['ratio']:.2f})")
    
    # Determine verdict
    total = bullish_score + bearish_score
    if total == 0:
        verdict = "NEUTRAL"
        confidence = 30
    elif bullish_score > bearish_score * 1.5:
        verdict = "BULLISH"
        confidence = min(85, 50 + bullish_score // 2)
    elif bearish_score > bullish_score * 1.5:
        verdict = "BEARISH"
        confidence = min(85, 50 + bearish_score // 2)
    else:
        verdict = "MIXED"
        confidence = 40
    
    # Build explanation
    explanation_parts = [
        f"📊 **OI Analysis**: {oi_interp['interpretation']}",
        f"   → OI Change: {oi_data.get('change_24h', 0):+.1f}% | Price Change: {price_change:+.1f}%",
        f"   → Action: {oi_interp['action']}",
        "",
        f"💰 **Funding Rate**: {funding_interp['interpretation']}",
        f"   → Action: {funding_interp['action']}",
        "",
        f"🐋 **Whale Positioning**:",
        f"   → Retail: {retail_ls.get('long_ratio', 50):.0f}% long / {retail_ls.get('short_ratio', 50):.0f}% short",
        f"   → Top Traders: {top_trader_ls.get('long_ratio', 50):.0f}% long / {top_trader_ls.get('short_ratio', 50):.0f}% short",
        f"   → Action: {ls_interp['action']}",
        "",
        f"🔥 **Taker Activity**: {taker_data['status']} (ratio: {taker_data.get('ratio', 1):.2f})"
    ]
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        
        # Raw data
        'open_interest': {
            'current': oi_data.get('current', 0),
            'change_24h': oi_data.get('change_24h', 0),
            'trend': oi_data.get('trend', 'UNKNOWN')
        },
        'funding': {
            'rate': funding_data.get('rate', 0),
            'rate_pct': funding_data.get('rate', 0) * 100,
            'status': funding_interp['signal']
        },
        'retail_ls': {
            'long_pct': retail_ls.get('long_ratio', 50),
            'short_pct': retail_ls.get('short_ratio', 50)
        },
        'top_trader_ls': {
            'long_pct': top_trader_ls.get('long_ratio', 50),
            'short_pct': top_trader_ls.get('short_ratio', 50)
        },
        'taker': {
            'ratio': taker_data.get('ratio', 1),
            'status': taker_data['status']
        },
        'price_change_24h': price_change,
        
        # Interpretations
        'oi_interpretation': oi_interp,
        'funding_interpretation': funding_interp,
        'ls_interpretation': ls_interp,
        
        # Overall
        'verdict': verdict,
        'confidence': confidence,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'signals': signals,
        'explanation': '\n'.join(explanation_parts)
    }


def format_institutional_html(data: Dict) -> str:
    """Format institutional data as HTML for display"""
    
    verdict_color = {
        'BULLISH': '#00d4aa',
        'BEARISH': '#ff4444',
        'NEUTRAL': '#888888',
        'MIXED': '#ffcc00'
    }.get(data['verdict'], '#888888')
    
    html = f"""
    <div style='background: #1a1a2e; border-radius: 12px; padding: 15px; margin: 10px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 1.2em; font-weight: bold; color: #00d4ff;'>🐋 Institutional Analysis</span>
            <span style='background: {verdict_color}22; color: {verdict_color}; padding: 5px 15px; 
                         border-radius: 20px; font-weight: bold;'>
                {data['verdict']} ({data['confidence']}%)
            </span>
        </div>
        
        <!-- OI + Price -->
        <div style='background: #252540; padding: 10px; border-radius: 8px; margin: 8px 0;'>
            <div style='color: #aaa; font-size: 0.85em;'>📊 Open Interest + Price</div>
            <div style='color: #fff; margin-top: 5px;'>
                OI: <strong style='color: {"#00d4aa" if data["open_interest"]["change_24h"] > 0 else "#ff4444"};'>
                    {data['open_interest']['change_24h']:+.1f}%
                </strong> | 
                Price: <strong style='color: {"#00d4aa" if data["price_change_24h"] > 0 else "#ff4444"};'>
                    {data['price_change_24h']:+.1f}%
                </strong>
            </div>
            <div style='color: {data["oi_interpretation"]["emoji"] == "🟢" and "#00d4aa" or data["oi_interpretation"]["emoji"] == "🔴" and "#ff4444" or "#ffcc00"}; 
                        margin-top: 5px; font-size: 0.9em;'>
                {data['oi_interpretation']['emoji']} {data['oi_interpretation']['interpretation']}
            </div>
        </div>
        
        <!-- Funding -->
        <div style='background: #252540; padding: 10px; border-radius: 8px; margin: 8px 0;'>
            <div style='color: #aaa; font-size: 0.85em;'>💰 Funding Rate</div>
            <div style='color: #fff; margin-top: 5px;'>
                Rate: <strong>{data['funding']['rate_pct']:.4f}%</strong>
            </div>
            <div style='color: #ffcc00; margin-top: 5px; font-size: 0.9em;'>
                {data['funding_interpretation']['emoji']} {data['funding_interpretation']['action']}
            </div>
        </div>
        
        <!-- Long/Short -->
        <div style='background: #252540; padding: 10px; border-radius: 8px; margin: 8px 0;'>
            <div style='color: #aaa; font-size: 0.85em;'>🐋 Whale vs Retail Positioning</div>
            <div style='display: flex; justify-content: space-between; margin-top: 8px;'>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>🐑 Retail</div>
                    <div style='color: #00d4aa;'>{data['retail_ls']['long_pct']:.0f}% Long</div>
                    <div style='color: #ff4444;'>{data['retail_ls']['short_pct']:.0f}% Short</div>
                </div>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>🐋 Top Traders</div>
                    <div style='color: #00d4aa;'>{data['top_trader_ls']['long_pct']:.0f}% Long</div>
                    <div style='color: #ff4444;'>{data['top_trader_ls']['short_pct']:.0f}% Short</div>
                </div>
            </div>
            <div style='color: {"#00d4aa" if "LONG" in data["ls_interpretation"]["action"] else "#ff4444" if "SHORT" in data["ls_interpretation"]["action"] else "#888"}; 
                        margin-top: 8px; font-size: 0.9em; text-align: center;'>
                {data['ls_interpretation']['action']}
            </div>
        </div>
        
        <!-- Signals -->
        <div style='margin-top: 10px;'>
            <div style='color: #aaa; font-size: 0.85em; margin-bottom: 5px;'>📋 Key Signals:</div>
    """
    
    for signal in data['signals']:
        html += f"<div style='color: #ccc; font-size: 0.9em; margin: 3px 0;'>• {signal}</div>"
    
    html += """
        </div>
    </div>
    """
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SUMMARY FOR TRADE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

def get_whale_summary(symbol: str) -> Dict:
    """Get quick whale summary for Trade Monitor display"""
    try:
        data = get_institutional_analysis(symbol)
        return {
            'verdict': data['verdict'],
            'confidence': data['confidence'],
            'top_signal': data['signals'][0] if data['signals'] else 'No clear signal',
            'whale_long_pct': data['top_trader_ls']['long_pct'],
            'retail_long_pct': data['retail_ls']['long_pct'],
            'oi_change': data['open_interest']['change_24h'],
            'funding_pct': data['funding']['rate_pct'],
            'quick_action': data['ls_interpretation']['action']
        }
    except Exception as e:
        return {
            'verdict': 'ERROR',
            'confidence': 0,
            'top_signal': f'Failed to fetch: {str(e)[:30]}',
            'whale_long_pct': 50,
            'retail_long_pct': 50,
            'oi_change': 0,
            'funding_pct': 0,
            'quick_action': 'Data unavailable'
        }


if __name__ == "__main__":
    # Test with BTC
    print("Testing with BTCUSDT...")
    result = get_institutional_analysis("BTCUSDT")
    print(f"\nVerdict: {result['verdict']} ({result['confidence']}%)")
    print(f"\nSignals:")
    for sig in result['signals']:
        print(f"  {sig}")
    print(f"\nExplanation:\n{result['explanation']}")
