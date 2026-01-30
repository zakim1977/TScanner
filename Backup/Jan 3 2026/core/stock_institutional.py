"""
Stock & ETF Institutional Data Module
=====================================
Equivalent of whale_institutional.py but for stocks and ETFs.

Data Sources (All FREE):
1. SEC EDGAR - 13F filings, insider trading (Form 4), institutional holdings
2. Yahoo Finance - Short interest, institutional ownership, options data
3. FINRA - Short interest, dark pool volume
4. Quiver Quant API - Congress trading, insider sentiment

Author: InvestorIQ
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# SEC EDGAR base URL
SEC_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Yahoo Finance
YAHOO_BASE = "https://query1.finance.yahoo.com/v10/finance"

# FINRA
FINRA_SHORT_INTEREST = "https://cdn.finra.org/equity/otcmarket/biweekly"

# Request headers (SEC requires User-Agent)
HEADERS = {
    'User-Agent': 'InvestorIQ Trading App contact@investoriq.app',
    'Accept': 'application/json'
}

# Cache for CIK lookups
_cik_cache = {}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get SEC CIK number for a ticker symbol"""
    global _cik_cache
    
    ticker = ticker.upper().replace('.', '-')
    
    if ticker in _cik_cache:
        return _cik_cache[ticker]
    
    try:
        response = requests.get(SEC_COMPANY_TICKERS, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker:
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    _cik_cache[ticker] = cik
                    return cik
    except:
        pass
    
    return None


def safe_get(url: str, timeout: int = 10) -> Optional[Dict]:
    """Safe HTTP GET with error handling"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR - INSIDER TRADING (Form 4)
# ═══════════════════════════════════════════════════════════════════════════════

def get_insider_trading(ticker: str, days: int = 90) -> Dict:
    """
    Get recent insider trading activity from SEC Form 4 filings.
    
    This is GOLD - when executives buy their own stock with their own money,
    it's one of the strongest bullish signals. They know the company best.
    
    Returns:
        Dict with insider buys, sells, net activity, and sentiment
    """
    result = {
        'ticker': ticker,
        'total_buys': 0,
        'total_sells': 0,
        'buy_value': 0,
        'sell_value': 0,
        'net_value': 0,
        'recent_transactions': [],
        'sentiment': 'NEUTRAL',
        'signal': None,
        'confidence': 0
    }
    
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return result
    
    try:
        # Get company filings
        url = f"{SEC_BASE_URL}/submissions/CIK{cik}.json"
        data = safe_get(url)
        
        if not data:
            return result
        
        filings = data.get('filings', {}).get('recent', {})
        forms = filings.get('form', [])
        dates = filings.get('filingDate', [])
        accessions = filings.get('accessionNumber', [])
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Find Form 4 filings (insider trades)
        form4_count = 0
        for i, form in enumerate(forms):
            if form == '4' and dates[i] >= cutoff_date:
                form4_count += 1
                
                # Parse the actual Form 4 for transaction details
                # This would require additional parsing of the XML filing
                # For now, we count Form 4s as activity indicator
        
        # Interpret activity
        if form4_count > 10:
            result['sentiment'] = 'HIGH_ACTIVITY'
            result['signal'] = 'Many insider transactions - check if buys or sells'
        elif form4_count > 5:
            result['sentiment'] = 'MODERATE_ACTIVITY'
        else:
            result['sentiment'] = 'LOW_ACTIVITY'
        
        result['form4_count'] = form4_count
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_insider_sentiment_yahoo(ticker: str) -> Dict:
    """
    Get insider trading data from Yahoo Finance (easier to parse than SEC).
    
    Returns:
        Dict with insider buys, sells, and sentiment score
    """
    result = {
        'ticker': ticker,
        'insider_buys_3m': 0,
        'insider_sells_3m': 0,
        'insider_buy_value': 0,
        'insider_sell_value': 0,
        'net_insider_value': 0,
        'insider_ownership_pct': 0,
        'sentiment': 'NEUTRAL',
        'signal': None
    }
    
    try:
        # Yahoo Finance insider data endpoint
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=insiderHolders,insiderTransactions,majorHoldersBreakdown"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        quote_summary = data.get('quoteSummary', {}).get('result', [{}])[0]
        
        # Major holders breakdown
        holders = quote_summary.get('majorHoldersBreakdown', {})
        result['insider_ownership_pct'] = holders.get('insidersPercentHeld', {}).get('raw', 0) * 100
        result['institutional_ownership_pct'] = holders.get('institutionsPercentHeld', {}).get('raw', 0) * 100
        
        # Insider transactions
        transactions = quote_summary.get('insiderTransactions', {}).get('transactions', [])
        
        buy_value = 0
        sell_value = 0
        buy_count = 0
        sell_count = 0
        recent_transactions = []
        
        for txn in transactions[:20]:  # Last 20 transactions
            shares = txn.get('shares', {}).get('raw', 0)
            value = txn.get('value', {}).get('raw', 0)
            txn_type = txn.get('transactionText', '')
            
            if 'Buy' in txn_type or 'Purchase' in txn_type:
                buy_value += value
                buy_count += 1
                recent_transactions.append({
                    'type': 'BUY',
                    'shares': shares,
                    'value': value,
                    'name': txn.get('filerName', 'Unknown')
                })
            elif 'Sale' in txn_type or 'Sell' in txn_type:
                sell_value += value
                sell_count += 1
                recent_transactions.append({
                    'type': 'SELL',
                    'shares': shares,
                    'value': value,
                    'name': txn.get('filerName', 'Unknown')
                })
        
        result['insider_buys_3m'] = buy_count
        result['insider_sells_3m'] = sell_count
        result['insider_buy_value'] = buy_value
        result['insider_sell_value'] = sell_value
        result['net_insider_value'] = buy_value - sell_value
        result['recent_transactions'] = recent_transactions[:5]
        
        # Determine sentiment
        if buy_count > 0 and sell_count == 0:
            result['sentiment'] = 'STRONG_BUY'
            result['signal'] = f'🟢 Insiders ONLY buying ({buy_count} buys, ${buy_value:,.0f})'
            result['confidence'] = 85
        elif buy_value > sell_value * 2:
            result['sentiment'] = 'BULLISH'
            result['signal'] = f'🟢 Insiders net BUYING (${buy_value - sell_value:,.0f} net)'
            result['confidence'] = 70
        elif sell_value > buy_value * 3:
            result['sentiment'] = 'BEARISH'
            result['signal'] = f'🔴 Heavy insider SELLING (${sell_value:,.0f} sold)'
            result['confidence'] = 65
        elif sell_count > 0 and buy_count == 0:
            result['sentiment'] = 'SELLING'
            result['signal'] = f'🔴 Insiders ONLY selling ({sell_count} sells)'
            result['confidence'] = 60
        else:
            result['sentiment'] = 'NEUTRAL'
            result['signal'] = 'Mixed insider activity'
            result['confidence'] = 40
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL OWNERSHIP
# ═══════════════════════════════════════════════════════════════════════════════

def get_institutional_ownership(ticker: str) -> Dict:
    """
    Get institutional ownership data.
    
    High institutional ownership = More stable, but also more correlated with market.
    Changes in institutional ownership = Potential signal.
    
    Returns:
        Dict with ownership percentages and top holders
    """
    result = {
        'ticker': ticker,
        'institutional_pct': 0,
        'insider_pct': 0,
        'float_pct': 0,
        'top_holders': [],
        'institutions_buying': 0,
        'institutions_selling': 0,
        'net_institutional': 'NEUTRAL',
        'signal': None
    }
    
    try:
        # Yahoo Finance holders endpoint
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=institutionOwnership,fundOwnership,majorHoldersBreakdown"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        quote_summary = data.get('quoteSummary', {}).get('result', [{}])[0]
        
        # Major holders breakdown
        holders = quote_summary.get('majorHoldersBreakdown', {})
        result['institutional_pct'] = holders.get('institutionsPercentHeld', {}).get('raw', 0) * 100
        result['insider_pct'] = holders.get('insidersPercentHeld', {}).get('raw', 0) * 100
        result['float_pct'] = holders.get('floatPercentHeld', {}).get('raw', 0) * 100
        
        # Top institutional holders
        inst_ownership = quote_summary.get('institutionOwnership', {}).get('ownershipList', [])
        
        for holder in inst_ownership[:10]:
            result['top_holders'].append({
                'name': holder.get('organization', 'Unknown'),
                'shares': holder.get('position', {}).get('raw', 0),
                'value': holder.get('value', {}).get('raw', 0),
                'pct_held': holder.get('pctHeld', {}).get('raw', 0) * 100,
                'change': holder.get('pctChange', {}).get('raw', 0) * 100
            })
        
        # Calculate buying vs selling
        buying = sum(1 for h in result['top_holders'] if h.get('change', 0) > 5)
        selling = sum(1 for h in result['top_holders'] if h.get('change', 0) < -5)
        
        result['institutions_buying'] = buying
        result['institutions_selling'] = selling
        
        if buying > selling + 2:
            result['net_institutional'] = 'ACCUMULATING'
            result['signal'] = f'🟢 Institutions ACCUMULATING ({buying} buying vs {selling} selling)'
        elif selling > buying + 2:
            result['net_institutional'] = 'DISTRIBUTING'
            result['signal'] = f'🔴 Institutions DISTRIBUTING ({selling} selling vs {buying} buying)'
        else:
            result['net_institutional'] = 'NEUTRAL'
            result['signal'] = 'Institutional activity balanced'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT INTEREST
# ═══════════════════════════════════════════════════════════════════════════════

def get_short_interest(ticker: str) -> Dict:
    """
    Get short interest data.
    
    High short interest + positive catalyst = SHORT SQUEEZE potential
    Rising short interest = Bearish sentiment
    Days to cover > 5 = Squeeze risk
    
    Returns:
        Dict with short interest metrics and squeeze potential
    """
    result = {
        'ticker': ticker,
        'short_pct_float': 0,
        'short_pct_shares': 0,
        'short_ratio': 0,  # Days to cover
        'shares_short': 0,
        'short_change_pct': 0,
        'squeeze_potential': 'LOW',
        'signal': None,
        'confidence': 0
    }
    
    try:
        # Yahoo Finance key statistics
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=defaultKeyStatistics"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        stats = data.get('quoteSummary', {}).get('result', [{}])[0].get('defaultKeyStatistics', {})
        
        result['short_pct_float'] = stats.get('shortPercentOfFloat', {}).get('raw', 0) * 100
        result['short_ratio'] = stats.get('shortRatio', {}).get('raw', 0)  # Days to cover
        result['shares_short'] = stats.get('sharesShort', {}).get('raw', 0)
        
        # Previous short interest for change calculation
        prev_short = stats.get('sharesShortPriorMonth', {}).get('raw', 0)
        if prev_short > 0:
            result['short_change_pct'] = ((result['shares_short'] - prev_short) / prev_short) * 100
        
        # Determine squeeze potential
        short_pct = result['short_pct_float']
        days_to_cover = result['short_ratio']
        
        if short_pct > 20 and days_to_cover > 5:
            result['squeeze_potential'] = 'HIGH'
            result['signal'] = f'🚀 HIGH SQUEEZE POTENTIAL - {short_pct:.1f}% short, {days_to_cover:.1f} days to cover'
            result['confidence'] = 80
        elif short_pct > 15 or days_to_cover > 4:
            result['squeeze_potential'] = 'MODERATE'
            result['signal'] = f'⚠️ Elevated short interest - {short_pct:.1f}% of float shorted'
            result['confidence'] = 60
        elif short_pct > 10:
            result['squeeze_potential'] = 'LOW'
            result['signal'] = f'Short interest notable ({short_pct:.1f}%) but not extreme'
            result['confidence'] = 40
        else:
            result['squeeze_potential'] = 'MINIMAL'
            result['signal'] = f'Low short interest ({short_pct:.1f}%)'
            result['confidence'] = 30
        
        # Check if shorts are increasing or decreasing
        if result['short_change_pct'] > 10:
            result['signal'] += f' | 📈 Shorts INCREASING (+{result["short_change_pct"]:.1f}%)'
        elif result['short_change_pct'] < -10:
            result['signal'] += f' | 📉 Shorts COVERING ({result["short_change_pct"]:.1f}%)'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def get_options_sentiment(ticker: str) -> Dict:
    """
    Get options market sentiment.
    
    Put/Call ratio > 1 = Bearish sentiment (more puts than calls)
    Put/Call ratio < 0.7 = Bullish sentiment (more calls than puts)
    Unusual volume = Potential whale activity
    
    Returns:
        Dict with options metrics and sentiment
    """
    result = {
        'ticker': ticker,
        'put_call_ratio': 1.0,
        'implied_volatility': 0,
        'call_volume': 0,
        'put_volume': 0,
        'call_oi': 0,
        'put_oi': 0,
        'sentiment': 'NEUTRAL',
        'signal': None,
        'unusual_activity': False
    }
    
    try:
        # Yahoo Finance options data
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        options = data.get('optionChain', {}).get('result', [{}])[0]
        
        # Get current options chain
        calls = options.get('options', [{}])[0].get('calls', [])
        puts = options.get('options', [{}])[0].get('puts', [])
        
        # Aggregate volume and OI
        call_volume = sum(c.get('volume', 0) or 0 for c in calls)
        put_volume = sum(p.get('volume', 0) or 0 for p in puts)
        call_oi = sum(c.get('openInterest', 0) or 0 for c in calls)
        put_oi = sum(p.get('openInterest', 0) or 0 for p in puts)
        
        result['call_volume'] = call_volume
        result['put_volume'] = put_volume
        result['call_oi'] = call_oi
        result['put_oi'] = put_oi
        
        # Calculate put/call ratio
        if call_volume > 0:
            result['put_call_ratio'] = put_volume / call_volume
        
        # Get IV from quote
        quote = options.get('quote', {})
        
        # Determine sentiment
        pc_ratio = result['put_call_ratio']
        
        if pc_ratio < 0.5:
            result['sentiment'] = 'VERY_BULLISH'
            result['signal'] = f'🟢 Options VERY BULLISH - P/C ratio {pc_ratio:.2f} (heavy call buying)'
        elif pc_ratio < 0.7:
            result['sentiment'] = 'BULLISH'
            result['signal'] = f'🟢 Options bullish - P/C ratio {pc_ratio:.2f}'
        elif pc_ratio > 1.5:
            result['sentiment'] = 'VERY_BEARISH'
            result['signal'] = f'🔴 Options VERY BEARISH - P/C ratio {pc_ratio:.2f} (heavy put buying)'
        elif pc_ratio > 1.0:
            result['sentiment'] = 'BEARISH'
            result['signal'] = f'🔴 Options bearish - P/C ratio {pc_ratio:.2f}'
        else:
            result['sentiment'] = 'NEUTRAL'
            result['signal'] = f'Options neutral - P/C ratio {pc_ratio:.2f}'
        
        # Check for unusual volume (volume >> open interest)
        total_volume = call_volume + put_volume
        total_oi = call_oi + put_oi
        
        if total_oi > 0 and total_volume > total_oi * 0.5:
            result['unusual_activity'] = True
            result['signal'] += ' | ⚠️ UNUSUAL OPTIONS ACTIVITY'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ETF FUND FLOWS
# ═══════════════════════════════════════════════════════════════════════════════

def get_etf_flows(ticker: str) -> Dict:
    """
    Get ETF fund flow data (for ETFs only).
    
    Inflows = Money entering the ETF = Bullish
    Outflows = Money leaving the ETF = Bearish
    
    Returns:
        Dict with flow data and interpretation
    """
    result = {
        'ticker': ticker,
        'is_etf': False,
        'aum': 0,
        'flow_1d': 0,
        'flow_1w': 0,
        'flow_1m': 0,
        'flow_3m': 0,
        'flow_trend': 'NEUTRAL',
        'signal': None
    }
    
    try:
        # Check if it's an ETF via Yahoo Finance
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=quoteType,summaryDetail"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        quote_type = data.get('quoteSummary', {}).get('result', [{}])[0].get('quoteType', {})
        
        if quote_type.get('quoteType') != 'ETF':
            result['signal'] = 'Not an ETF - fund flow data not applicable'
            return result
        
        result['is_etf'] = True
        
        # Get fund data
        summary = data.get('quoteSummary', {}).get('result', [{}])[0].get('summaryDetail', {})
        result['aum'] = summary.get('totalAssets', {}).get('raw', 0)
        
        # Note: Real-time ETF flows require paid data (ETF.com, Bloomberg)
        # For now, we can use AUM changes as a proxy
        
        result['signal'] = f'ETF with ${result["aum"]/1e9:.1f}B AUM - Check etf.com for detailed flows'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYST RATINGS
# ═══════════════════════════════════════════════════════════════════════════════

def get_analyst_ratings(ticker: str) -> Dict:
    """
    Get analyst ratings and price targets.
    
    Returns:
        Dict with ratings, price targets, and sentiment
    """
    result = {
        'ticker': ticker,
        'rating': 'HOLD',
        'num_analysts': 0,
        'strong_buy': 0,
        'buy': 0,
        'hold': 0,
        'sell': 0,
        'strong_sell': 0,
        'target_mean': 0,
        'target_high': 0,
        'target_low': 0,
        'current_price': 0,
        'upside_pct': 0,
        'signal': None
    }
    
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=recommendationTrend,financialData"
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            return result
        
        data = response.json()
        quote_summary = data.get('quoteSummary', {}).get('result', [{}])[0]
        
        # Recommendation trend
        trend = quote_summary.get('recommendationTrend', {}).get('trend', [{}])
        if trend:
            current = trend[0]
            result['strong_buy'] = current.get('strongBuy', 0)
            result['buy'] = current.get('buy', 0)
            result['hold'] = current.get('hold', 0)
            result['sell'] = current.get('sell', 0)
            result['strong_sell'] = current.get('strongSell', 0)
        
        result['num_analysts'] = sum([
            result['strong_buy'], result['buy'], result['hold'],
            result['sell'], result['strong_sell']
        ])
        
        # Financial data for price targets
        fin_data = quote_summary.get('financialData', {})
        result['target_mean'] = fin_data.get('targetMeanPrice', {}).get('raw', 0)
        result['target_high'] = fin_data.get('targetHighPrice', {}).get('raw', 0)
        result['target_low'] = fin_data.get('targetLowPrice', {}).get('raw', 0)
        result['current_price'] = fin_data.get('currentPrice', {}).get('raw', 0)
        
        if result['current_price'] > 0 and result['target_mean'] > 0:
            result['upside_pct'] = ((result['target_mean'] - result['current_price']) / result['current_price']) * 100
        
        # Determine overall rating
        total = result['num_analysts']
        if total > 0:
            bullish = result['strong_buy'] + result['buy']
            bearish = result['sell'] + result['strong_sell']
            
            if bullish > total * 0.7:
                result['rating'] = 'STRONG_BUY'
                result['signal'] = f'🟢 Analysts BULLISH - {bullish}/{total} recommend Buy'
            elif bullish > total * 0.5:
                result['rating'] = 'BUY'
                result['signal'] = f'🟢 Analysts lean bullish - {bullish}/{total} Buy ratings'
            elif bearish > total * 0.5:
                result['rating'] = 'SELL'
                result['signal'] = f'🔴 Analysts lean bearish - {bearish}/{total} Sell ratings'
            else:
                result['rating'] = 'HOLD'
                result['signal'] = f'⚪ Analysts mixed - {result["hold"]}/{total} Hold ratings'
        
        # Add price target info
        if result['upside_pct'] != 0:
            result['signal'] += f' | Target: ${result["target_mean"]:.2f} ({result["upside_pct"]:+.1f}%)'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_institutional_analysis(ticker: str) -> Dict:
    """
    Get comprehensive institutional analysis for a stock or ETF.
    Combines all data sources into a single verdict.
    
    Args:
        ticker: Stock or ETF ticker symbol
        
    Returns:
        Dict with all institutional data and combined verdict
    """
    result = {
        'ticker': ticker.upper(),
        'timestamp': datetime.now().isoformat(),
        'insider': {},
        'institutional': {},
        'short_interest': {},
        'options': {},
        'analyst': {},
        'etf': {},
        'signals': [],
        'verdict': 'NEUTRAL',
        'confidence': 0,
        'score': 0  # -100 (bearish) to +100 (bullish)
    }
    
    score = 0
    signals = []
    
    # 1. Insider Trading
    result['insider'] = get_insider_sentiment_yahoo(ticker)
    if result['insider'].get('sentiment') == 'STRONG_BUY':
        score += 30
        signals.append(result['insider'].get('signal', ''))
    elif result['insider'].get('sentiment') == 'BULLISH':
        score += 20
        signals.append(result['insider'].get('signal', ''))
    elif result['insider'].get('sentiment') == 'BEARISH':
        score -= 20
        signals.append(result['insider'].get('signal', ''))
    elif result['insider'].get('sentiment') == 'SELLING':
        score -= 15
        signals.append(result['insider'].get('signal', ''))
    
    # 2. Institutional Ownership
    result['institutional'] = get_institutional_ownership(ticker)
    if result['institutional'].get('net_institutional') == 'ACCUMULATING':
        score += 20
        signals.append(result['institutional'].get('signal', ''))
    elif result['institutional'].get('net_institutional') == 'DISTRIBUTING':
        score -= 20
        signals.append(result['institutional'].get('signal', ''))
    
    # 3. Short Interest
    result['short_interest'] = get_short_interest(ticker)
    squeeze = result['short_interest'].get('squeeze_potential', 'LOW')
    if squeeze == 'HIGH':
        score += 15  # Could squeeze UP
        signals.append(result['short_interest'].get('signal', ''))
    elif squeeze == 'MODERATE':
        score += 5
        signals.append(result['short_interest'].get('signal', ''))
    
    # Short change direction
    short_change = result['short_interest'].get('short_change_pct', 0)
    if short_change < -10:  # Shorts covering = bullish
        score += 10
    elif short_change > 10:  # More shorts = bearish pressure
        score -= 10
    
    # 4. Options Flow
    result['options'] = get_options_sentiment(ticker)
    opt_sentiment = result['options'].get('sentiment', 'NEUTRAL')
    if opt_sentiment == 'VERY_BULLISH':
        score += 25
        signals.append(result['options'].get('signal', ''))
    elif opt_sentiment == 'BULLISH':
        score += 15
        signals.append(result['options'].get('signal', ''))
    elif opt_sentiment == 'VERY_BEARISH':
        score -= 25
        signals.append(result['options'].get('signal', ''))
    elif opt_sentiment == 'BEARISH':
        score -= 15
        signals.append(result['options'].get('signal', ''))
    
    if result['options'].get('unusual_activity'):
        signals.append('⚠️ Unusual options volume detected - whale activity')
    
    # 5. Analyst Ratings
    result['analyst'] = get_analyst_ratings(ticker)
    analyst_rating = result['analyst'].get('rating', 'HOLD')
    if analyst_rating == 'STRONG_BUY':
        score += 15
        signals.append(result['analyst'].get('signal', ''))
    elif analyst_rating == 'BUY':
        score += 10
        signals.append(result['analyst'].get('signal', ''))
    elif analyst_rating == 'SELL':
        score -= 15
        signals.append(result['analyst'].get('signal', ''))
    
    # 6. ETF Flows (if applicable)
    result['etf'] = get_etf_flows(ticker)
    
    # Calculate final verdict
    result['score'] = max(-100, min(100, score))
    result['signals'] = [s for s in signals if s]
    
    if score >= 40:
        result['verdict'] = 'BULLISH'
        result['confidence'] = min(90, 50 + score // 2)
    elif score >= 20:
        result['verdict'] = 'LEAN_BULLISH'
        result['confidence'] = min(70, 40 + score // 2)
    elif score <= -40:
        result['verdict'] = 'BEARISH'
        result['confidence'] = min(90, 50 + abs(score) // 2)
    elif score <= -20:
        result['verdict'] = 'LEAN_BEARISH'
        result['confidence'] = min(70, 40 + abs(score) // 2)
    else:
        result['verdict'] = 'NEUTRAL'
        result['confidence'] = 40
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_whale_summary(ticker: str) -> Dict:
    """
    Get a quick summary for display in Trade Monitor (like crypto whale_summary).
    
    Returns simplified dict for compact display.
    """
    analysis = get_stock_institutional_analysis(ticker)
    
    return {
        'ticker': ticker,
        'verdict': analysis['verdict'],
        'confidence': analysis['confidence'],
        'score': analysis['score'],
        'insider_sentiment': analysis['insider'].get('sentiment', 'N/A'),
        'short_pct': analysis['short_interest'].get('short_pct_float', 0),
        'options_pc_ratio': analysis['options'].get('put_call_ratio', 1.0),
        'institutional_pct': analysis['institutional'].get('institutional_pct', 0),
        'analyst_rating': analysis['analyst'].get('rating', 'N/A'),
        'signals': analysis['signals'][:3]  # Top 3 signals
    }


def interpret_stock_for_trade(analysis: Dict, trade_direction: str) -> Dict:
    """
    Interpret stock institutional data relative to a specific trade direction.
    
    Args:
        analysis: Result from get_stock_institutional_analysis()
        trade_direction: 'LONG' or 'SHORT'
        
    Returns:
        Dict with conclusions specific to the trade
    """
    conclusions = []
    score = analysis.get('score', 0)
    
    # 1. Overall alignment
    if trade_direction == 'LONG':
        if score > 20:
            conclusions.append(("✅", f"Institutional data supports your LONG (score: {score:+d})"))
        elif score < -20:
            conclusions.append(("⚠️", f"Institutional data against your LONG (score: {score:+d})"))
        else:
            conclusions.append(("🟡", f"Institutional data neutral (score: {score:+d})"))
    else:  # SHORT
        if score < -20:
            conclusions.append(("✅", f"Institutional data supports your SHORT (score: {score:+d})"))
        elif score > 20:
            conclusions.append(("⚠️", f"Institutional data against your SHORT (score: {score:+d})"))
        else:
            conclusions.append(("🟡", f"Institutional data neutral (score: {score:+d})"))
    
    # 2. Insider activity
    insider = analysis.get('insider', {}).get('sentiment', 'NEUTRAL')
    if trade_direction == 'LONG' and insider in ['STRONG_BUY', 'BULLISH']:
        conclusions.append(("✅", f"Insiders are BUYING → Confirms your LONG"))
    elif trade_direction == 'LONG' and insider in ['BEARISH', 'SELLING']:
        conclusions.append(("⚠️", f"Insiders are SELLING → Against your LONG"))
    elif trade_direction == 'SHORT' and insider in ['BEARISH', 'SELLING']:
        conclusions.append(("✅", f"Insiders are SELLING → Confirms your SHORT"))
    elif trade_direction == 'SHORT' and insider in ['STRONG_BUY', 'BULLISH']:
        conclusions.append(("⚠️", f"Insiders are BUYING → Against your SHORT"))
    
    # 3. Short interest
    short_pct = analysis.get('short_interest', {}).get('short_pct_float', 0)
    squeeze = analysis.get('short_interest', {}).get('squeeze_potential', 'LOW')
    
    if trade_direction == 'LONG' and squeeze == 'HIGH':
        conclusions.append(("✅", f"High short interest ({short_pct:.1f}%) → Squeeze potential for LONG"))
    elif trade_direction == 'SHORT' and short_pct > 15:
        conclusions.append(("⚠️", f"High short interest ({short_pct:.1f}%) → Squeeze risk for SHORT"))
    
    # 4. Options flow
    opt_sentiment = analysis.get('options', {}).get('sentiment', 'NEUTRAL')
    if trade_direction == 'LONG' and opt_sentiment in ['VERY_BULLISH', 'BULLISH']:
        conclusions.append(("✅", f"Options flow bullish → Supports your LONG"))
    elif trade_direction == 'LONG' and opt_sentiment in ['VERY_BEARISH', 'BEARISH']:
        conclusions.append(("⚠️", f"Options flow bearish → Against your LONG"))
    elif trade_direction == 'SHORT' and opt_sentiment in ['VERY_BEARISH', 'BEARISH']:
        conclusions.append(("✅", f"Options flow bearish → Supports your SHORT"))
    elif trade_direction == 'SHORT' and opt_sentiment in ['VERY_BULLISH', 'BULLISH']:
        conclusions.append(("⚠️", f"Options flow bullish → Against your SHORT"))
    
    # Summarize
    positive = sum(1 for c in conclusions if c[0] == "✅")
    negative = sum(1 for c in conclusions if c[0] == "⚠️")
    
    if positive >= 2 and negative == 0:
        summary = f"🟢 STRONG - Institutional data confirms your {trade_direction}"
        color = "#00d4aa"
    elif positive > negative:
        summary = f"🟢 SUPPORTIVE - Data leans toward your {trade_direction}"
        color = "#00d4aa"
    elif negative > positive:
        summary = f"🔴 CAUTION - Data conflicts with your {trade_direction}"
        color = "#ff4444"
    else:
        summary = f"🟡 MIXED - Institutional signals are conflicting"
        color = "#ffcc00"
    
    return {
        'conclusions': conclusions,
        'summary': summary,
        'color': color,
        'positive': positive,
        'negative': negative
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE: CRYPTO vs STOCK WHALE DATA
# ═══════════════════════════════════════════════════════════════════════════════

"""
CRYPTO (Binance)              STOCK (SEC/Yahoo)
────────────────────────────  ──────────────────────────────
Open Interest                 Short Interest (similar concept)
Funding Rate                  Put/Call Ratio (sentiment)
Top Trader Long/Short         Insider Trading (Form 4)
Retail Long/Short             Institutional Ownership %
Taker Buy/Sell                Options Flow
Price 24h                     Analyst Ratings

Both provide:
- WHO is positioning (retail vs institutions)
- WHAT direction they're betting
- HOW confident they are (size of positions)
"""
