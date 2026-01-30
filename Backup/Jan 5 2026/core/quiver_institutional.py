"""
Quiver Quant Integration for Stock Institutional Data
======================================================

Provides:
- Congress Trading (Nancy Pelosi, etc.)
- Insider Trading (CEO buys/sells)
- Short Interest
- Lobbying Data

Free tier: 100 API calls/day
Pro ($30/mo): 1000 API calls/day

Get your API key at: https://www.quiverquant.com/
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

QUIVER_BASE_URL = "https://api.quiverquant.com/beta"

# Cache to avoid hitting rate limits
_cache = {}
_cache_ttl = 3600  # 1 hour cache


def _get_cached(key: str) -> Optional[Dict]:
    """Get cached data if not expired"""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return data
    return None


def _set_cache(key: str, data: Dict):
    """Cache data with timestamp"""
    _cache[key] = (data, time.time())


# ═══════════════════════════════════════════════════════════════════════════════
# CONGRESS TRADING
# "Follow the smart money" - Congress members often trade ahead of legislation
# ═══════════════════════════════════════════════════════════════════════════════

def get_congress_trades(api_key: str, symbol: str = None, days: int = 30) -> Dict:
    """
    Get recent Congress trading activity
    
    Args:
        api_key: Quiver Quant API key
        symbol: Optional - filter by stock symbol (e.g., 'AAPL')
        days: Look back period
        
    Returns:
        Dict with congress trading data and sentiment
    """
    cache_key = f"congress_{symbol}_{days}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        if symbol:
            url = f"{QUIVER_BASE_URL}/historical/congresstrading/{symbol}"
        else:
            url = f"{QUIVER_BASE_URL}/live/congresstrading"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            trades = response.json()
            
            # Filter by date
            cutoff = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade in trades[:100]:  # Limit to avoid processing too many
                try:
                    trade_date = datetime.strptime(trade.get('TransactionDate', ''), '%Y-%m-%d')
                    if trade_date >= cutoff:
                        recent_trades.append(trade)
                except:
                    continue
            
            # Analyze sentiment
            buys = [t for t in recent_trades if 'Purchase' in t.get('Transaction', '')]
            sells = [t for t in recent_trades if 'Sale' in t.get('Transaction', '')]
            
            # Notable traders (known for good returns)
            notable_traders = ['Nancy Pelosi', 'Dan Crenshaw', 'Tommy Tuberville', 'Josh Gottheimer']
            notable_buys = [t for t in buys if t.get('Representative', '') in notable_traders]
            
            # Calculate sentiment score
            if len(recent_trades) == 0:
                sentiment_score = 50
                sentiment = "NEUTRAL"
            else:
                buy_ratio = len(buys) / len(recent_trades) if recent_trades else 0.5
                sentiment_score = int(buy_ratio * 100)
                
                if sentiment_score >= 70:
                    sentiment = "BULLISH"
                elif sentiment_score <= 30:
                    sentiment = "BEARISH"
                else:
                    sentiment = "NEUTRAL"
            
            result = {
                'total_trades': len(recent_trades),
                'buys': len(buys),
                'sells': len(sells),
                'buy_ratio': len(buys) / len(recent_trades) if recent_trades else 0.5,
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'notable_buys': len(notable_buys),
                'recent_trades': recent_trades[:10],  # Last 10 trades
                'notable_traders_active': [t.get('Representative') for t in notable_buys],
                'interpretation': _interpret_congress(sentiment_score, notable_buys, symbol)
            }
            
            _set_cache(cache_key, result)
            return result
            
        elif response.status_code == 429:
            return {'error': 'Rate limit exceeded', 'sentiment': 'UNKNOWN', 'sentiment_score': 50}
        else:
            return {'error': f'API error: {response.status_code}', 'sentiment': 'UNKNOWN', 'sentiment_score': 50}
            
    except Exception as e:
        return {'error': str(e), 'sentiment': 'UNKNOWN', 'sentiment_score': 50}


def _interpret_congress(score: int, notable_buys: List, symbol: str) -> str:
    """Generate human-readable interpretation of congress trading"""
    if notable_buys:
        traders = list(set([t.get('Representative', 'Unknown') for t in notable_buys[:3]]))
        return f"🏛️ Notable congress members ({', '.join(traders)}) buying {symbol or 'stocks'} - historically a bullish signal"
    elif score >= 70:
        return f"🏛️ Congress members net BUYING - they often know about upcoming legislation"
    elif score <= 30:
        return f"🏛️ Congress members net SELLING - could signal upcoming negative news"
    else:
        return "🏛️ Mixed congress activity - no clear directional bias"


# ═══════════════════════════════════════════════════════════════════════════════
# INSIDER TRADING
# CEO/CFO buys are often the strongest signal
# ═══════════════════════════════════════════════════════════════════════════════

def get_insider_trades(api_key: str, symbol: str = None, days: int = 30) -> Dict:
    """
    Get insider trading activity (CEO, CFO, Directors)
    
    CEO buying their own stock = Very bullish (they know the company best)
    CEO selling = Often just diversification, less meaningful
    """
    cache_key = f"insider_{symbol}_{days}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        if symbol:
            url = f"{QUIVER_BASE_URL}/historical/insiders/{symbol}"
        else:
            url = f"{QUIVER_BASE_URL}/live/insiders"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            trades = response.json()
            
            # Filter by date
            cutoff = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade in trades[:100]:
                try:
                    trade_date = datetime.strptime(trade.get('Date', ''), '%Y-%m-%d')
                    if trade_date >= cutoff:
                        recent_trades.append(trade)
                except:
                    continue
            
            # Categorize by type and importance
            buys = [t for t in recent_trades if t.get('AcquiredDisposed', '') == 'A']
            sells = [t for t in recent_trades if t.get('AcquiredDisposed', '') == 'D']
            
            # C-suite transactions (most important)
            c_suite_titles = ['CEO', 'CFO', 'COO', 'CTO', 'President', 'Chairman']
            c_suite_buys = [t for t in buys if any(title in t.get('Title', '') for title in c_suite_titles)]
            c_suite_sells = [t for t in sells if any(title in t.get('Title', '') for title in c_suite_titles)]
            
            # Calculate total values
            total_buy_value = sum(float(t.get('Value', 0) or 0) for t in buys)
            total_sell_value = sum(float(t.get('Value', 0) or 0) for t in sells)
            
            # Sentiment scoring (C-suite buys weighted heavily)
            base_score = 50
            
            if c_suite_buys:
                base_score += 25  # Strong bullish signal
            if len(buys) > len(sells):
                base_score += 10
            if total_buy_value > total_sell_value:
                base_score += 10
            if c_suite_sells and not c_suite_buys:
                base_score -= 15  # C-suite selling without buying
            
            sentiment_score = max(0, min(100, base_score))
            
            if sentiment_score >= 70:
                sentiment = "BULLISH"
            elif sentiment_score <= 35:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            result = {
                'total_trades': len(recent_trades),
                'buys': len(buys),
                'sells': len(sells),
                'c_suite_buys': len(c_suite_buys),
                'c_suite_sells': len(c_suite_sells),
                'total_buy_value': total_buy_value,
                'total_sell_value': total_sell_value,
                'net_value': total_buy_value - total_sell_value,
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'recent_trades': recent_trades[:10],
                'interpretation': _interpret_insider(sentiment_score, c_suite_buys, c_suite_sells, symbol)
            }
            
            _set_cache(cache_key, result)
            return result
            
        elif response.status_code == 429:
            return {'error': 'Rate limit exceeded', 'sentiment': 'UNKNOWN', 'sentiment_score': 50}
        else:
            return {'error': f'API error: {response.status_code}', 'sentiment': 'UNKNOWN', 'sentiment_score': 50}
            
    except Exception as e:
        return {'error': str(e), 'sentiment': 'UNKNOWN', 'sentiment_score': 50}


def _interpret_insider(score: int, c_buys: List, c_sells: List, symbol: str) -> str:
    """Generate interpretation of insider trading"""
    sym = symbol or "this stock"
    
    if c_buys:
        titles = list(set([t.get('Title', 'Executive')[:20] for t in c_buys[:3]]))
        return f"👔 C-Suite BUYING {sym} ({', '.join(titles)}) - Strongest bullish signal!"
    elif score >= 70:
        return f"👔 Insiders net buying {sym} - They believe in the company"
    elif c_sells and not c_buys:
        return f"👔 C-Suite selling {sym} - Could be diversification OR concern"
    elif score <= 35:
        return f"👔 Heavy insider selling in {sym} - Proceed with caution"
    else:
        return f"👔 Mixed insider activity in {sym}"


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT INTEREST
# High short interest = potential squeeze OR justified bearishness
# ═══════════════════════════════════════════════════════════════════════════════

def get_short_interest(api_key: str, symbol: str) -> Dict:
    """
    Get short interest data
    
    High short interest (>20%) = Either:
    - Squeeze potential (if fundamentals improving)
    - Smart money bearish (if fundamentals weak)
    """
    cache_key = f"short_{symbol}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{QUIVER_BASE_URL}/historical/shortsqueeze/{symbol}"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data:
                latest = data[0] if isinstance(data, list) else data
                
                short_pct = float(latest.get('Short%Float', 0) or latest.get('ShortPercentFloat', 0) or 0)
                days_to_cover = float(latest.get('DaystoCover', 0) or 0)
                
                # Squeeze potential scoring
                if short_pct >= 30:
                    squeeze_potential = "HIGH"
                    squeeze_score = 85
                elif short_pct >= 20:
                    squeeze_potential = "MODERATE"
                    squeeze_score = 65
                elif short_pct >= 10:
                    squeeze_potential = "LOW"
                    squeeze_score = 45
                else:
                    squeeze_potential = "MINIMAL"
                    squeeze_score = 25
                
                # Sentiment (contrarian view)
                # High short = contrarian bullish IF other signals align
                result = {
                    'short_percent': short_pct,
                    'days_to_cover': days_to_cover,
                    'squeeze_potential': squeeze_potential,
                    'squeeze_score': squeeze_score,
                    'interpretation': _interpret_short(short_pct, days_to_cover, symbol),
                    'contrarian_signal': "BULLISH" if short_pct > 20 else "NEUTRAL"
                }
                
                _set_cache(cache_key, result)
                return result
            
            return {'short_percent': 0, 'squeeze_potential': 'UNKNOWN', 'squeeze_score': 50}
            
        elif response.status_code == 429:
            return {'error': 'Rate limit exceeded', 'squeeze_potential': 'UNKNOWN'}
        else:
            return {'error': f'API error: {response.status_code}', 'squeeze_potential': 'UNKNOWN'}
            
    except Exception as e:
        return {'error': str(e), 'squeeze_potential': 'UNKNOWN'}


def _interpret_short(short_pct: float, days_cover: float, symbol: str) -> str:
    """Interpret short interest data"""
    if short_pct >= 30:
        return f"🩳 EXTREME short interest ({short_pct:.1f}%) in {symbol} - High squeeze potential if catalyst appears!"
    elif short_pct >= 20:
        return f"🩳 High short interest ({short_pct:.1f}%) - Shorts could get squeezed on good news"
    elif short_pct >= 10:
        return f"🩳 Moderate shorts ({short_pct:.1f}%) - Some bearish sentiment"
    else:
        return f"🩳 Low short interest ({short_pct:.1f}%) - Not a significant factor"


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED INSTITUTIONAL ANALYSIS FOR STOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_institutional_analysis(api_key: str, symbol: str) -> Dict:
    """
    Get comprehensive institutional analysis for a stock
    Combines: Congress + Insider + Short Interest
    
    Returns combined sentiment and actionable insights
    """
    if not api_key:
        return {
            'available': False,
            'error': 'No API key provided',
            'message': 'Get free API key at quiverquant.com'
        }
    
    # Fetch all data
    congress = get_congress_trades(api_key, symbol, days=60)
    insider = get_insider_trades(api_key, symbol, days=60)
    short = get_short_interest(api_key, symbol)
    
    # Calculate combined institutional score
    scores = []
    factors = []
    
    # Congress factor (weight: 25%)
    if 'sentiment_score' in congress and 'error' not in congress:
        scores.append(('congress', congress['sentiment_score'], 0.25))
        if congress.get('notable_buys', 0) > 0:
            factors.append(('🏛️ Notable Congress buying', +15))
        elif congress.get('sentiment') == 'BULLISH':
            factors.append(('🏛️ Congress net buyers', +10))
        elif congress.get('sentiment') == 'BEARISH':
            factors.append(('🏛️ Congress net sellers', -10))
    
    # Insider factor (weight: 40% - most important)
    if 'sentiment_score' in insider and 'error' not in insider:
        scores.append(('insider', insider['sentiment_score'], 0.40))
        if insider.get('c_suite_buys', 0) > 0:
            factors.append(('👔 C-Suite BUYING', +20))
        elif insider.get('sentiment') == 'BULLISH':
            factors.append(('👔 Insiders net buying', +10))
        elif insider.get('c_suite_sells', 0) > 0 and insider.get('c_suite_buys', 0) == 0:
            factors.append(('👔 C-Suite selling', -15))
    
    # Short interest factor (weight: 35%)
    if 'squeeze_score' in short and 'error' not in short:
        # For short interest, high = contrarian bullish potential
        scores.append(('short', short['squeeze_score'], 0.35))
        if short.get('squeeze_potential') == 'HIGH':
            factors.append(('🩳 High squeeze potential', +15))
        elif short.get('squeeze_potential') == 'MODERATE':
            factors.append(('🩳 Moderate short interest', +5))
    
    # Calculate weighted average
    if scores:
        weighted_sum = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)
        combined_score = int(weighted_sum / total_weight) if total_weight > 0 else 50
    else:
        combined_score = 50
    
    # Determine overall sentiment
    if combined_score >= 70:
        overall_sentiment = "BULLISH"
        sentiment_color = "#00ff88"
        action_hint = "Institutional backing - consider full position"
    elif combined_score >= 55:
        overall_sentiment = "LEAN BULLISH"
        sentiment_color = "#00d4aa"
        action_hint = "Moderate institutional interest"
    elif combined_score >= 45:
        overall_sentiment = "NEUTRAL"
        sentiment_color = "#ffcc00"
        action_hint = "Mixed institutional signals"
    elif combined_score >= 35:
        overall_sentiment = "LEAN BEARISH"
        sentiment_color = "#ff9500"
        action_hint = "Some institutional caution"
    else:
        overall_sentiment = "BEARISH"
        sentiment_color = "#ff4444"
        action_hint = "Institutional selling pressure"
    
    return {
        'available': True,
        'symbol': symbol,
        'combined_score': combined_score,
        'overall_sentiment': overall_sentiment,
        'sentiment_color': sentiment_color,
        'action_hint': action_hint,
        'factors': factors,
        'congress': congress,
        'insider': insider,
        'short_interest': short,
        'data_sources': ['Congress Trading', 'Insider Trading', 'Short Interest'],
        'last_updated': datetime.now().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_stock_institutional_html(data: Dict) -> str:
    """Format institutional data as HTML for display"""
    if not data.get('available'):
        return f"""
        <div style='background: #2a2a3a; border-left: 4px solid #666; padding: 15px; border-radius: 8px;'>
            <div style='color: #888;'>📊 Stock Institutional Data</div>
            <div style='color: #aaa; margin-top: 8px;'>
                {data.get('message', 'Configure API key in settings')}
            </div>
            <div style='color: #00d4ff; margin-top: 8px;'>
                <a href='https://www.quiverquant.com/' target='_blank' style='color: #00d4ff;'>
                    Get free API key →
                </a>
            </div>
        </div>
        """
    
    score = data.get('combined_score', 50)
    sentiment = data.get('overall_sentiment', 'NEUTRAL')
    color = data.get('sentiment_color', '#888')
    
    # Build factors HTML
    factors_html = ""
    for factor, impact in data.get('factors', []):
        impact_color = "#00d4aa" if impact > 0 else "#ff4444" if impact < 0 else "#888"
        impact_str = f"+{impact}" if impact > 0 else str(impact)
        factors_html += f"""
        <div style='display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #333;'>
            <span style='color: #ccc;'>{factor}</span>
            <span style='color: {impact_color}; font-weight: bold;'>{impact_str}</span>
        </div>
        """
    
    # Congress summary
    congress = data.get('congress', {})
    congress_html = ""
    if congress and 'error' not in congress:
        c_sentiment = congress.get('sentiment', 'NEUTRAL')
        c_color = "#00d4aa" if c_sentiment == "BULLISH" else "#ff4444" if c_sentiment == "BEARISH" else "#888"
        congress_html = f"""
        <div style='margin-top: 10px; padding: 10px; background: #1a1a2e; border-radius: 6px;'>
            <div style='color: #888; font-size: 0.85em;'>🏛️ Congress Trading</div>
            <div style='color: {c_color}; font-weight: bold;'>{c_sentiment}</div>
            <div style='color: #666; font-size: 0.8em;'>
                {congress.get('buys', 0)} buys / {congress.get('sells', 0)} sells (30d)
            </div>
        </div>
        """
    
    # Insider summary
    insider = data.get('insider', {})
    insider_html = ""
    if insider and 'error' not in insider:
        i_sentiment = insider.get('sentiment', 'NEUTRAL')
        i_color = "#00d4aa" if i_sentiment == "BULLISH" else "#ff4444" if i_sentiment == "BEARISH" else "#888"
        c_suite = insider.get('c_suite_buys', 0)
        insider_html = f"""
        <div style='margin-top: 10px; padding: 10px; background: #1a1a2e; border-radius: 6px;'>
            <div style='color: #888; font-size: 0.85em;'>👔 Insider Trading</div>
            <div style='color: {i_color}; font-weight: bold;'>{i_sentiment}</div>
            <div style='color: #666; font-size: 0.8em;'>
                {insider.get('buys', 0)} buys / {insider.get('sells', 0)} sells | C-Suite buys: {c_suite}
            </div>
        </div>
        """
    
    # Short interest summary
    short = data.get('short_interest', {})
    short_html = ""
    if short and 'error' not in short:
        squeeze = short.get('squeeze_potential', 'UNKNOWN')
        s_color = "#ff00ff" if squeeze == "HIGH" else "#ffcc00" if squeeze == "MODERATE" else "#888"
        short_html = f"""
        <div style='margin-top: 10px; padding: 10px; background: #1a1a2e; border-radius: 6px;'>
            <div style='color: #888; font-size: 0.85em;'>🩳 Short Interest</div>
            <div style='color: {s_color}; font-weight: bold;'>{short.get('short_percent', 0):.1f}% Short</div>
            <div style='color: #666; font-size: 0.8em;'>
                Squeeze Potential: {squeeze} | Days to Cover: {short.get('days_to_cover', 0):.1f}
            </div>
        </div>
        """
    
    return f"""
    <div style='background: linear-gradient(135deg, #0d1b2a, #1b263b); border: 2px solid {color}; 
                border-radius: 12px; padding: 20px; margin: 15px 0;'>
        <div style='text-align: center; margin-bottom: 15px;'>
            <span style='color: {color}; font-size: 1.3em; font-weight: bold;'>
                📊 STOCK INSTITUTIONAL SCORE: {score}/100 ({sentiment})
            </span>
        </div>
        
        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; margin-bottom: 15px;'>
            <div style='color: #888; font-size: 0.9em; margin-bottom: 8px;'>Key Factors:</div>
            {factors_html if factors_html else "<div style='color: #666;'>No significant factors detected</div>"}
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
            {congress_html}
            {insider_html}
            {short_html}
        </div>
        
        <div style='text-align: center; margin-top: 15px; padding-top: 12px; border-top: 1px solid #333;'>
            <span style='background: {color}33; color: {color}; padding: 6px 16px; border-radius: 6px;'>
                💡 {data.get('action_hint', '')}
            </span>
        </div>
    </div>
    """
