"""
COIN RISK DETECTOR
==================

Unified module for detecting high-risk coins:
- 🎰 Meme Coins (DOGE, SHIB, PEPE, etc.)
- 🆕 New Listings (< 90 days old)
- 🤖 ML Reliability Assessment (based on data history)
- ⚠️ Combined Risk Assessment

USAGE:
    from coin_risk_detector import (
        check_coin_risks,
        display_coin_warnings,
        get_risk_badges_html,
        get_ml_reliability,
    )
    
    # Get all risks for a symbol
    risks = check_coin_risks('PEPEUSDT', whale_pct=70, price_change=25)
    
    # Display warnings in Streamlit
    display_coin_warnings('PEPEUSDT', risks)
    
    # Get ML reliability for weighting
    ml_info = get_ml_reliability('PEPEUSDT')
    if ml_info['trust_level'] == 'UNRELIABLE':
        # Disable or heavily discount ML signals
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# New listing thresholds (days)
NEW_LISTING_THRESHOLDS = {
    'very_new': 14,      # < 14 days = VERY HIGH RISK
    'new': 30,           # < 30 days = HIGH RISK  
    'recent': 90,        # < 90 days = CAUTION
}

# ML Reliability thresholds based on coin age (days)
ML_RELIABILITY_THRESHOLDS = {
    'unreliable': 30,    # < 30 days = ML predictions UNRELIABLE
    'limited': 90,       # 30-90 days = LIMITED reliability
    'acceptable': 180,   # 90-180 days = ACCEPTABLE but flag it
    # > 180 days = RELIABLE (full trust)
}

# ML weight multipliers based on reliability
ML_WEIGHT_MULTIPLIERS = {
    'UNRELIABLE': 0.0,   # Don't use ML at all
    'LIMITED': 0.3,      # Heavily discount ML
    'ACCEPTABLE': 0.7,   # Slight discount
    'RELIABLE': 1.0,     # Full weight
}

# Known meme coins (base names without USDT/BTC suffix)
MEME_COINS = {
    # Classic Memes
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME',
    'ELON', 'BABYDOGE', 'SHIBADOGE', 'DOGELON',
    
    # Cat memes
    'CAT', 'POPCAT', 'MEW', 'MOG',
    
    # Frog memes  
    'PEPE', 'PEPE2', 'PEPECOIN', 'BRETT',
    
    # Dog variants
    'DOGE2', 'SHIB2', 'FLOKI2', 'CHEEMS', 'MYRO',
    
    # Other memes
    'WOJAK', 'CHAD', 'GIGACHAD', 'BITCOIN', 'PEOPLE',
    'LUNC', 'USTC',  # Terra collapse coins
    'BOME', 'SLERF', 'SILLY',  # Solana memes
    
    # AI/Hype memes
    'TURBO', 'AIDOGE', 'AICODE',
    
    # Food/Random
    'SUSHI', 'BURGER', 'CAKE',  # Could be DeFi but often traded as memes
    'APE',  # NFT related
    
    # New 2024 memes
    'NEIRO', 'GOAT', 'PNUT', 'ACT', 'HIPPO',
}

# Meme coin patterns (regex)
MEME_PATTERNS = [
    r'.*DOGE.*',      # Any DOGE variant
    r'.*SHIB.*',      # Any SHIB variant
    r'.*PEPE.*',      # Any PEPE variant
    r'.*FLOKI.*',     # Any FLOKI variant
    r'.*INU$',        # Ends with INU (Shiba Inu style)
    r'.*MOON$',       # Ends with MOON
    r'.*ELON.*',      # Elon related
    r'.*SAFE.*',      # SafeMoon style
    r'.*BABY.*',      # Baby tokens
    r'.*MINI.*',      # Mini tokens
    r'.*KING$',       # King tokens
    r'.*CAT$',        # Cat tokens
    r'.*DOG$',        # Dog tokens
]

# Cache for API calls
_listing_cache: Dict[str, datetime] = {}
_cache_ttl = 3600  # 1 hour


# ═══════════════════════════════════════════════════════════════════════════════
# MEME COIN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_base_symbol(symbol: str) -> str:
    """Extract base coin name from trading pair (DOGEUSDT -> DOGE)"""
    # Remove common quote currencies
    for suffix in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB', 'TUSD', 'FDUSD']:
        if symbol.endswith(suffix):
            return symbol[:-len(suffix)]
    return symbol


def is_meme_coin(symbol: str) -> Dict:
    """
    Check if a symbol is a meme coin.
    
    Returns:
        {
            'is_meme': bool,
            'reason': str,
            'match_type': 'exact' | 'pattern' | None
        }
    """
    base = extract_base_symbol(symbol.upper())
    
    # Check exact match
    if base in MEME_COINS:
        return {
            'is_meme': True,
            'reason': f'{base} is a known meme coin',
            'match_type': 'exact'
        }
    
    # Check patterns
    for pattern in MEME_PATTERNS:
        if re.match(pattern, base, re.IGNORECASE):
            return {
                'is_meme': True,
                'reason': f'{base} matches meme coin pattern',
                'match_type': 'pattern'
            }
    
    return {
        'is_meme': False,
        'reason': '',
        'match_type': None
    }


def get_meme_coin_warnings(symbol: str, whale_pct: float = 50, price_change: float = 0) -> List[str]:
    """
    Get contextual warnings for meme coins.
    
    Args:
        symbol: Trading pair
        whale_pct: Current whale positioning %
        price_change: 24h price change %
        
    Returns:
        List of warning strings
    """
    meme_check = is_meme_coin(symbol)
    warnings = []
    
    if not meme_check['is_meme']:
        return warnings
    
    base = extract_base_symbol(symbol)
    
    # Base warnings
    warnings.append(f"🎰 {base} is a MEME COIN - extreme volatility!")
    warnings.append("⚠️ Meme coins can pump/dump 50%+ in hours")
    warnings.append("⚠️ Whale manipulation is common")
    
    # Contextual warnings
    if price_change > 30:
        warnings.append(f"🚀 +{price_change:.1f}% pump - FOMO DANGER! May dump hard")
    elif price_change > 15:
        warnings.append(f"📈 +{price_change:.1f}% rise - Late entry risk")
    
    if price_change < -20:
        warnings.append(f"📉 {price_change:.1f}% dump - Could be dead cat bounce")
    elif price_change < -10:
        warnings.append(f"📉 {price_change:.1f}% drop - May continue lower")
    
    if whale_pct > 75:
        warnings.append(f"🐋 {whale_pct:.0f}% whale positioning - Exit liquidity trap?")
    elif whale_pct < 35:
        warnings.append(f"🐋 Only {whale_pct:.0f}% whale interest - Pure retail speculation")
    
    # Divergence warning
    if whale_pct < 45 and price_change > 20:
        warnings.append("⚠️ Whales NOT buying this pump - DANGER!")
    
    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# NEW LISTING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_listing_date(symbol: str) -> Optional[datetime]:
    """
    Get the first available trade date for a symbol on Binance.
    Uses the earliest 1d kline as a proxy for listing date.
    
    Args:
        symbol: Trading pair (e.g., 'ZKPUSDT')
        
    Returns:
        datetime of first trade, or None if not found
    """
    global _listing_cache
    
    # Check cache first
    if symbol in _listing_cache:
        return _listing_cache[symbol]
    
    try:
        # Get earliest available kline (1d timeframe for efficiency)
        response = requests.get(
            'https://api.binance.com/api/v3/klines',
            params={
                'symbol': symbol,
                'interval': '1d',
                'limit': 1,
                'startTime': 0,  # From the beginning
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                # First element [0] is the open time in milliseconds
                first_trade_ms = data[0][0]
                first_trade = datetime.fromtimestamp(first_trade_ms / 1000)
                
                # Cache the result
                _listing_cache[symbol] = first_trade
                return first_trade
                
    except Exception as e:
        print(f"[COIN-RISK] Error checking listing date for {symbol}: {e}")
    
    return None


def get_coin_age_days(symbol: str) -> Optional[int]:
    """Get the age of a coin in days since listing."""
    listing_date = get_listing_date(symbol)
    if listing_date:
        age = datetime.now() - listing_date
        return age.days
    return None


def check_new_listing(symbol: str) -> Dict:
    """
    Check if a symbol is a new listing and return risk assessment.
    
    Returns:
        {
            'is_new_listing': bool,
            'age_days': int or None,
            'risk_level': 'VERY_HIGH' | 'HIGH' | 'CAUTION' | 'OK',
            'reason': str,
            'warnings': list[str],
            'listing_date': datetime or None,
        }
    """
    result = {
        'is_new_listing': False,
        'age_days': None,
        'risk_level': 'OK',
        'reason': '',
        'warnings': [],
        'listing_date': None,
    }
    
    listing_date = get_listing_date(symbol)
    if not listing_date:
        return result
    
    result['listing_date'] = listing_date
    age_days = (datetime.now() - listing_date).days
    result['age_days'] = age_days
    
    # Determine risk level
    if age_days < NEW_LISTING_THRESHOLDS['very_new']:
        result['is_new_listing'] = True
        result['risk_level'] = 'VERY_HIGH'
        result['reason'] = f"Listed only {age_days} days ago"
        result['warnings'] = [
            f"🆕 VERY NEW LISTING ({age_days} days old)",
            "⚠️ Extremely high volatility expected",
            "⚠️ Easy to manipulate - low liquidity",
            "⚠️ Consider AVOIDING or use tiny position",
        ]
        
    elif age_days < NEW_LISTING_THRESHOLDS['new']:
        result['is_new_listing'] = True
        result['risk_level'] = 'HIGH'
        result['reason'] = f"Listed only {age_days} days ago"
        result['warnings'] = [
            f"🆕 NEW LISTING ({age_days} days old)",
            "⚠️ High volatility - limited price history",
            "⚠️ Use smaller position size",
        ]
        
    elif age_days < NEW_LISTING_THRESHOLDS['recent']:
        result['is_new_listing'] = True
        result['risk_level'] = 'CAUTION'
        result['reason'] = f"Listed {age_days} days ago"
        result['warnings'] = [
            f"📅 Recent listing ({age_days} days old)",
            "⚠️ Limited historical data for patterns",
        ]
    
    return result


def get_new_listing_warnings(symbol: str, whale_pct: float = 50, price_change: float = 0) -> List[str]:
    """Get contextual warnings for new listings."""
    check = check_new_listing(symbol)
    warnings = check['warnings'].copy()
    
    if check['is_new_listing']:
        age = check['age_days']
        
        # Additional contextual warnings
        if price_change > 20:
            warnings.append(f"🚀 +{price_change:.1f}% in 24h on new listing = EXTREME FOMO risk")
        
        if price_change < -15:
            warnings.append(f"📉 {price_change:.1f}% dump on new listing = Could go lower")
        
        if whale_pct > 70 and age < 30:
            warnings.append(f"🐋 {whale_pct:.0f}% whale positioning on new coin = Potential trap")
        
        if whale_pct < 40 and age < 30:
            warnings.append(f"🐋 Only {whale_pct:.0f}% whale interest = Retail-driven pump risk")
    
    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# ML RELIABILITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_ml_reliability(symbol: str) -> Dict:
    """
    Assess ML prediction reliability based on coin age.
    
    Thresholds:
    - < 30 days: UNRELIABLE - Not enough data for ML training
    - 30-90 days: LIMITED - Some data but patterns not established
    - 90-180 days: ACCEPTABLE - Reasonable data, some caution
    - > 180 days: RELIABLE - Sufficient historical data
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        
    Returns:
        {
            'age_days': int or None,
            'trust_level': 'UNRELIABLE' | 'LIMITED' | 'ACCEPTABLE' | 'RELIABLE',
            'weight_multiplier': float (0.0 - 1.0),
            'reason': str,
            'warning': str or None,
            'color': str (hex color for display),
        }
    """
    age_days = get_coin_age_days(symbol)
    
    # Default for unknown age - assume unreliable
    if age_days is None:
        return {
            'age_days': None,
            'trust_level': 'UNRELIABLE',
            'weight_multiplier': ML_WEIGHT_MULTIPLIERS['UNRELIABLE'],
            'reason': 'Unable to determine coin age',
            'warning': '🤖 ML UNRELIABLE: Cannot verify coin history',
            'color': '#ff4444',
        }
    
    # Determine trust level
    if age_days < ML_RELIABILITY_THRESHOLDS['unreliable']:
        return {
            'age_days': age_days,
            'trust_level': 'UNRELIABLE',
            'weight_multiplier': ML_WEIGHT_MULTIPLIERS['UNRELIABLE'],
            'reason': f'Only {age_days} days of data (need 30+)',
            'warning': f'🤖 ML UNRELIABLE: Only {age_days}d of data - ML predictions DISABLED',
            'color': '#ff4444',
        }
    
    elif age_days < ML_RELIABILITY_THRESHOLDS['limited']:
        return {
            'age_days': age_days,
            'trust_level': 'LIMITED',
            'weight_multiplier': ML_WEIGHT_MULTIPLIERS['LIMITED'],
            'reason': f'{age_days} days of data (limited patterns)',
            'warning': f'🤖 ML LIMITED: {age_days}d of data - ML confidence reduced to 30%',
            'color': '#ff8800',
        }
    
    elif age_days < ML_RELIABILITY_THRESHOLDS['acceptable']:
        return {
            'age_days': age_days,
            'trust_level': 'ACCEPTABLE',
            'weight_multiplier': ML_WEIGHT_MULTIPLIERS['ACCEPTABLE'],
            'reason': f'{age_days} days of data (acceptable)',
            'warning': f'🤖 ML ACCEPTABLE: {age_days}d of data - ML confidence at 70%',
            'color': '#ffcc00',
        }
    
    else:  # > 180 days
        return {
            'age_days': age_days,
            'trust_level': 'RELIABLE',
            'weight_multiplier': ML_WEIGHT_MULTIPLIERS['RELIABLE'],
            'reason': f'{age_days} days of data (reliable)',
            'warning': None,  # No warning needed for reliable
            'color': '#00cc00',
        }


def get_ml_reliability_warnings(symbol: str) -> List[str]:
    """
    Get warnings related to ML reliability.
    
    Args:
        symbol: Trading pair
        
    Returns:
        List of warning strings
    """
    ml_info = get_ml_reliability(symbol)
    warnings = []
    
    if ml_info['trust_level'] == 'UNRELIABLE':
        warnings.append(f"🤖 ML predictions DISABLED - only {ml_info['age_days'] or '?'} days of data")
        warnings.append("⚠️ Not enough historical data for pattern recognition")
        warnings.append("📊 Rely on WHALE DATA and STRUCTURE only")
        
    elif ml_info['trust_level'] == 'LIMITED':
        warnings.append(f"🤖 ML confidence REDUCED - only {ml_info['age_days']} days of data")
        warnings.append("⚠️ Limited historical patterns - ML weight at 30%")
        warnings.append("📊 Give more weight to whale positioning")
        
    elif ml_info['trust_level'] == 'ACCEPTABLE':
        warnings.append(f"🤖 ML moderately reliable - {ml_info['age_days']} days of data")
        warnings.append("📊 ML predictions functional but use caution")
    
    return warnings


def get_adjusted_ml_score(raw_ml_score: float, symbol: str) -> Tuple[float, str]:
    """
    Adjust ML score based on coin age reliability.
    
    Args:
        raw_ml_score: Original ML prediction score
        symbol: Trading pair
        
    Returns:
        (adjusted_score, explanation)
    """
    ml_info = get_ml_reliability(symbol)
    multiplier = ml_info['weight_multiplier']
    
    if multiplier == 0:
        return 0, f"ML disabled ({ml_info['reason']})"
    elif multiplier < 1:
        adjusted = raw_ml_score * multiplier
        return adjusted, f"ML reduced: {raw_ml_score:.1f} × {multiplier:.0%} = {adjusted:.1f} ({ml_info['reason']})"
    else:
        return raw_ml_score, "ML at full weight"


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def check_coin_risks(symbol: str, whale_pct: float = 50, price_change: float = 0) -> Dict:
    """
    Comprehensive risk check for a coin - combines meme + new listing + ML reliability.
    
    Args:
        symbol: Trading pair (e.g., 'PEPEUSDT')
        whale_pct: Current whale positioning %
        price_change: 24h price change %
        
    Returns:
        {
            'symbol': str,
            'base_coin': str,
            
            # Meme coin info
            'is_meme_coin': bool,
            'meme_reason': str,
            
            # New listing info
            'is_new_listing': bool,
            'listing_age_days': int or None,
            'listing_risk': str,
            'listing_date': datetime or None,
            
            # ML Reliability info
            'ml_trust_level': 'UNRELIABLE' | 'LIMITED' | 'ACCEPTABLE' | 'RELIABLE',
            'ml_weight_multiplier': float,
            'ml_warning': str or None,
            
            # Combined assessment
            'overall_risk': 'EXTREME' | 'HIGH' | 'MEDIUM' | 'LOW',
            'risk_score': int (0-100, higher = riskier),
            'warnings': list[str],
            'recommendations': list[str],
        }
    """
    base = extract_base_symbol(symbol)
    
    # Check meme status
    meme_check = is_meme_coin(symbol)
    meme_warnings = get_meme_coin_warnings(symbol, whale_pct, price_change) if meme_check['is_meme'] else []
    
    # Check new listing status
    listing_check = check_new_listing(symbol)
    listing_warnings = get_new_listing_warnings(symbol, whale_pct, price_change) if listing_check['is_new_listing'] else []
    
    # Check ML reliability
    ml_info = get_ml_reliability(symbol)
    ml_warnings = get_ml_reliability_warnings(symbol)
    
    # Calculate risk score (0-100)
    risk_score = 0
    
    # Meme coin adds risk
    if meme_check['is_meme']:
        risk_score += 40
        if meme_check['match_type'] == 'exact':
            risk_score += 10  # Known meme coins are riskier
    
    # New listing adds risk
    if listing_check['is_new_listing']:
        if listing_check['risk_level'] == 'VERY_HIGH':
            risk_score += 40
        elif listing_check['risk_level'] == 'HIGH':
            risk_score += 25
        elif listing_check['risk_level'] == 'CAUTION':
            risk_score += 10
    
    # ML unreliability adds risk (can't trust signals)
    if ml_info['trust_level'] == 'UNRELIABLE':
        risk_score += 15
    elif ml_info['trust_level'] == 'LIMITED':
        risk_score += 8
    
    # Price volatility adds risk
    if abs(price_change) > 30:
        risk_score += 15
    elif abs(price_change) > 15:
        risk_score += 8
    
    # Low whale interest on pumping coin = risky
    if whale_pct < 40 and price_change > 15:
        risk_score += 10
    
    # Determine overall risk level
    if risk_score >= 70:
        overall_risk = 'EXTREME'
    elif risk_score >= 45:
        overall_risk = 'HIGH'
    elif risk_score >= 20:
        overall_risk = 'MEDIUM'
    else:
        overall_risk = 'LOW'
    
    # Generate recommendations
    recommendations = []
    
    if overall_risk == 'EXTREME':
        recommendations.append("🚫 Consider AVOIDING this trade")
        recommendations.append("💰 If trading: MAX 1% of portfolio")
        recommendations.append("🎯 Set tight stop loss (1-2%)")
    elif overall_risk == 'HIGH':
        recommendations.append("⚠️ HIGH RISK - Trade with extreme caution")
        recommendations.append("💰 Reduce position size by 50%")
        recommendations.append("🎯 Use tighter stop loss than normal")
    elif overall_risk == 'MEDIUM':
        recommendations.append("📊 MODERATE RISK - Be cautious")
        recommendations.append("💰 Consider smaller position")
    
    # Add ML-specific recommendations
    if ml_info['trust_level'] == 'UNRELIABLE':
        recommendations.append("🤖 IGNORE ML predictions - use whale data + structure only")
    elif ml_info['trust_level'] == 'LIMITED':
        recommendations.append("🤖 ML signals have reduced weight (30%) - prioritize whale data")
    
    # Combine all warnings (deduplicate)
    all_warnings = list(dict.fromkeys(meme_warnings + listing_warnings + ml_warnings))
    
    return {
        'symbol': symbol,
        'base_coin': base,
        
        # Meme info
        'is_meme_coin': meme_check['is_meme'],
        'meme_reason': meme_check['reason'],
        
        # Listing info
        'is_new_listing': listing_check['is_new_listing'],
        'listing_age_days': listing_check['age_days'],
        'listing_risk': listing_check['risk_level'],
        'listing_date': listing_check['listing_date'],
        
        # ML Reliability info
        'ml_trust_level': ml_info['trust_level'],
        'ml_weight_multiplier': ml_info['weight_multiplier'],
        'ml_warning': ml_info['warning'],
        'ml_color': ml_info['color'],
        
        # Combined
        'overall_risk': overall_risk,
        'risk_score': min(100, risk_score),
        'warnings': all_warnings,
        'recommendations': recommendations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS (HTML for Streamlit)
# ═══════════════════════════════════════════════════════════════════════════════

def get_ml_reliability_badge_html(risks: Dict) -> str:
    """
    Get HTML badge for ML reliability.
    
    Args:
        risks: Output from check_coin_risks()
        
    Returns:
        HTML string with ML reliability badge
    """
    trust_level = risks.get('ml_trust_level', 'RELIABLE')
    
    if trust_level == 'RELIABLE':
        return ""  # No badge needed for reliable
    
    color = risks.get('ml_color', '#888888')
    multiplier = risks.get('ml_weight_multiplier', 1.0)
    
    # Different badge text based on level
    if trust_level == 'UNRELIABLE':
        badge_text = "🤖 ML OFF"
    elif trust_level == 'LIMITED':
        badge_text = f"🤖 ML {int(multiplier*100)}%"
    else:  # ACCEPTABLE
        badge_text = f"🤖 ML {int(multiplier*100)}%"
    
    return f"""
        <span style='background: {color}22; color: {color}; padding: 2px 8px; 
                    border-radius: 4px; font-size: 0.8em; font-weight: bold;
                    border: 1px solid {color}; margin-right: 5px;'>
            {badge_text}
        </span>
    """


def get_risk_badges_html(risks: Dict) -> str:
    """
    Get HTML badges for coin risks (meme + new listing + ML reliability).
    
    Args:
        risks: Output from check_coin_risks()
        
    Returns:
        HTML string with badges
    """
    badges = []
    
    # Meme coin badge
    if risks['is_meme_coin']:
        badges.append("""
            <span style='background: #ff44ff22; color: #ff44ff; padding: 2px 8px; 
                        border-radius: 4px; font-size: 0.8em; font-weight: bold;
                        border: 1px solid #ff44ff; margin-right: 5px;'>
                🎰 MEME
            </span>
        """)
    
    # New listing badge
    if risks['is_new_listing']:
        age = risks['listing_age_days']
        risk = risks['listing_risk']
        
        if risk == 'VERY_HIGH':
            color = '#ff4444'
        elif risk == 'HIGH':
            color = '#ff8800'
        else:
            color = '#ffcc00'
        
        badges.append(f"""
            <span style='background: {color}22; color: {color}; padding: 2px 8px; 
                        border-radius: 4px; font-size: 0.8em; font-weight: bold;
                        border: 1px solid {color}; margin-right: 5px;'>
                🆕 {age}d OLD
            </span>
        """)
    
    # ML Reliability badge (only show if not reliable)
    ml_badge = get_ml_reliability_badge_html(risks)
    if ml_badge:
        badges.append(ml_badge)
    
    # Overall risk badge
    risk = risks['overall_risk']
    score = risks['risk_score']
    
    risk_colors = {
        'EXTREME': '#ff0000',
        'HIGH': '#ff4444',
        'MEDIUM': '#ffcc00',
        'LOW': '#00cc00',
    }
    color = risk_colors.get(risk, '#888888')
    
    if risk in ['EXTREME', 'HIGH']:
        badges.append(f"""
            <span style='background: {color}22; color: {color}; padding: 2px 8px; 
                        border-radius: 4px; font-size: 0.8em; font-weight: bold;
                        border: 1px solid {color}; margin-right: 5px;'>
                ⚠️ {risk} RISK
            </span>
        """)
    
    return ''.join(badges)


def get_ml_reliability_html(risks: Dict) -> str:
    """
    Get dedicated HTML section for ML reliability warning.
    Use this for prominent display in analysis views.
    
    Args:
        risks: Output from check_coin_risks()
        
    Returns:
        HTML string for ML reliability section
    """
    trust_level = risks.get('ml_trust_level', 'RELIABLE')
    
    if trust_level == 'RELIABLE':
        return ""  # No display needed
    
    color = risks.get('ml_color', '#888888')
    warning = risks.get('ml_warning', '')
    multiplier = risks.get('ml_weight_multiplier', 1.0)
    age = risks.get('listing_age_days', '?')
    
    # Icon based on severity
    if trust_level == 'UNRELIABLE':
        icon = '🚫'
        title = 'ML PREDICTIONS DISABLED'
        extra = "Not enough historical data for reliable ML predictions. Trade using WHALE DATA and MARKET STRUCTURE only."
    elif trust_level == 'LIMITED':
        icon = '⚠️'
        title = 'ML RELIABILITY LIMITED'
        extra = f"ML predictions have reduced confidence ({int(multiplier*100)}% weight). Prioritize whale positioning data."
    else:  # ACCEPTABLE
        icon = '📊'
        title = 'ML RELIABILITY ACCEPTABLE'
        extra = f"ML predictions functional but with moderate confidence ({int(multiplier*100)}% weight)."
    
    return f"""
    <div style='background: {color}15; border: 1px solid {color}; 
                padding: 12px 15px; border-radius: 8px; margin: 10px 0;'>
        <div style='color: {color}; font-weight: bold; font-size: 1.0em; margin-bottom: 6px;'>
            {icon} {title}
        </div>
        <div style='color: #ccc; font-size: 0.9em;'>
            <b>Coin Age:</b> {age} days | <b>ML Weight:</b> {int(multiplier*100)}%<br>
            {extra}
        </div>
    </div>
    """


def get_risk_warning_html(risks: Dict) -> str:
    """
    Get full HTML warning box for coin risks.
    
    Args:
        risks: Output from check_coin_risks()
        
    Returns:
        HTML string for warning box
    """
    if risks['overall_risk'] == 'LOW':
        return ""
    
    # Determine colors
    risk = risks['overall_risk']
    if risk == 'EXTREME':
        color = '#ff0000'
        icon = '🚨'
        title = 'EXTREME RISK COIN'
    elif risk == 'HIGH':
        color = '#ff4444'
        icon = '⚠️'
        title = 'HIGH RISK COIN'
    else:
        color = '#ffcc00'
        icon = '📊'
        title = 'ELEVATED RISK'
    
    # Build content
    content_parts = []
    
    # Risk factors
    factors = []
    if risks['is_meme_coin']:
        factors.append(f"🎰 Meme Coin ({risks['meme_reason']})")
    if risks['is_new_listing']:
        factors.append(f"🆕 New Listing ({risks['listing_age_days']} days old)")
    if risks['ml_trust_level'] in ['UNRELIABLE', 'LIMITED']:
        factors.append(f"🤖 ML {risks['ml_trust_level']} ({risks['listing_age_days'] or '?'}d data)")
    
    if factors:
        content_parts.append("<b>Risk Factors:</b><br>" + "<br>".join(factors))
    
    # Warnings
    if risks['warnings']:
        warnings_html = "<br>".join([f"• {w}" for w in risks['warnings'][:5]])
        content_parts.append(f"<b>Warnings:</b><br>{warnings_html}")
    
    # Recommendations
    if risks['recommendations']:
        recs_html = "<br>".join([f"• {r}" for r in risks['recommendations']])
        content_parts.append(f"<b>Recommendations:</b><br>{recs_html}")
    
    content = "<br><br>".join(content_parts)
    
    return f"""
    <div style='background: {color}22; border-left: 4px solid {color}; 
                padding: 15px; border-radius: 8px; margin: 10px 0;'>
        <div style='color: {color}; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;'>
            {icon} {title} - Risk Score: {risks['risk_score']}/100
        </div>
        <div style='color: #ccc; font-size: 0.95em; line-height: 1.6;'>
            {content}
        </div>
    </div>
    """


def display_coin_warnings(symbol: str, risks: Dict = None):
    """
    Display Streamlit warning for coin risks.
    Call this in your analysis display section.
    
    Args:
        symbol: Trading pair
        risks: Output from check_coin_risks() (will fetch if not provided)
    """
    try:
        import streamlit as st
    except ImportError:
        print(f"[COIN-RISK] Streamlit not available, skipping display")
        return
    
    if risks is None:
        risks = check_coin_risks(symbol)
    
    # Always show ML reliability warning if not reliable
    ml_html = get_ml_reliability_html(risks)
    if ml_html:
        st.markdown(ml_html, unsafe_allow_html=True)
    
    # Show general risk warning if not low
    if risks['overall_risk'] != 'LOW':
        html = get_risk_warning_html(risks)
        if html:
            st.markdown(html, unsafe_allow_html=True)


def display_ml_reliability_warning(symbol: str, risks: Dict = None):
    """
    Display ONLY the ML reliability warning.
    Use this when you want to show ML warning separately from other risks.
    
    Args:
        symbol: Trading pair
        risks: Output from check_coin_risks() (will fetch if not provided)
    """
    try:
        import streamlit as st
    except ImportError:
        print(f"[COIN-RISK] Streamlit not available, skipping display")
        return
    
    if risks is None:
        risks = check_coin_risks(symbol)
    
    ml_html = get_ml_reliability_html(risks)
    if ml_html:
        st.markdown(ml_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def add_coin_risks_to_data_quality(data_quality: Dict, symbol: str, 
                                    whale_pct: float = 50, price_change: float = 0) -> Dict:
    """
    Add coin risk info to existing data_quality dict.
    Use this in analyze_symbol_full() function.
    
    Args:
        data_quality: Existing data quality dict
        symbol: Trading pair
        whale_pct: Whale positioning %
        price_change: 24h price change %
        
    Returns:
        Updated data_quality dict
    """
    risks = check_coin_risks(symbol, whale_pct, price_change)
    
    # Add to data_quality
    data_quality['is_meme_coin'] = risks['is_meme_coin']
    data_quality['meme_reason'] = risks['meme_reason']
    data_quality['is_new_listing'] = risks['is_new_listing']
    data_quality['listing_age_days'] = risks['listing_age_days']
    data_quality['listing_risk'] = risks['listing_risk']
    data_quality['coin_risk_score'] = risks['risk_score']
    data_quality['coin_risk_level'] = risks['overall_risk']
    
    # Add ML reliability info
    data_quality['ml_trust_level'] = risks['ml_trust_level']
    data_quality['ml_weight_multiplier'] = risks['ml_weight_multiplier']
    data_quality['ml_warning'] = risks['ml_warning']
    
    # Add warnings
    if 'warnings' not in data_quality:
        data_quality['warnings'] = []
    data_quality['warnings'].extend(risks['warnings'])
    
    return data_quality


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK CHECKS (for inline use)
# ═══════════════════════════════════════════════════════════════════════════════

def is_high_risk_coin(symbol: str) -> bool:
    """Quick check if coin is high risk (meme OR new listing OR ML unreliable)."""
    meme = is_meme_coin(symbol)
    listing = check_new_listing(symbol)
    ml = get_ml_reliability(symbol)
    return meme['is_meme'] or listing['is_new_listing'] or ml['trust_level'] in ['UNRELIABLE', 'LIMITED']


def is_ml_reliable(symbol: str) -> bool:
    """Quick check if ML predictions are reliable for this coin."""
    ml = get_ml_reliability(symbol)
    return ml['trust_level'] == 'RELIABLE'


def get_ml_weight(symbol: str) -> float:
    """Get ML weight multiplier for this coin (0.0 - 1.0)."""
    ml = get_ml_reliability(symbol)
    return ml['weight_multiplier']


def get_coin_risk_emoji(symbol: str) -> str:
    """Get single emoji representing coin risk."""
    risks = check_coin_risks(symbol)
    
    if risks['overall_risk'] == 'EXTREME':
        return '🚨'
    elif risks['overall_risk'] == 'HIGH':
        return '⚠️'
    elif risks['is_meme_coin']:
        return '🎰'
    elif risks['is_new_listing']:
        return '🆕'
    elif risks['ml_trust_level'] in ['UNRELIABLE', 'LIMITED']:
        return '🤖'
    return ''


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("COIN RISK DETECTOR TEST (with ML Reliability)")
    print("=" * 70)
    
    # Test symbols
    test_cases = [
        ('BTCUSDT', 55, 2),       # Normal, old coin
        ('ETHUSDT', 60, -1),      # Normal, old coin
        ('DOGEUSDT', 45, 15),     # Meme, old coin
        ('PEPEUSDT', 70, 30),     # Meme + possibly newer
        ('SHIBUSDT', 35, -10),    # Meme
        ('ZKPUSDT', 58, 5),       # Possibly new
        ('WIFUSDT', 72, 25),      # Meme + possibly newer
    ]
    
    for symbol, whale_pct, price_change in test_cases:
        print(f"\n{'─' * 70}")
        print(f"📊 {symbol} | Whale: {whale_pct}% | Price 24h: {price_change:+.1f}%")
        print(f"{'─' * 70}")
        
        # Check meme
        meme = is_meme_coin(symbol)
        if meme['is_meme']:
            print(f"   🎰 MEME COIN: {meme['reason']}")
        
        # Check ML reliability
        ml = get_ml_reliability(symbol)
        print(f"   🤖 ML Trust: {ml['trust_level']} (weight: {ml['weight_multiplier']:.0%})")
        if ml['warning']:
            print(f"      └─ {ml['warning']}")
        
        # Check all risks
        risks = check_coin_risks(symbol, whale_pct, price_change)
        
        print(f"   Risk Score: {risks['risk_score']}/100")
        print(f"   Overall Risk: {risks['overall_risk']}")
        
        if risks['is_new_listing']:
            print(f"   🆕 New Listing: {risks['listing_age_days']} days old ({risks['listing_risk']})")
        
        if risks['warnings']:
            print(f"   Warnings:")
            for w in risks['warnings'][:4]:
                print(f"      • {w}")
        
        if risks['recommendations']:
            print(f"   Recommendations:")
            for r in risks['recommendations'][:3]:
                print(f"      • {r}")
    
    print("\n" + "=" * 70)
    print("ML RELIABILITY LEVELS")
    print("=" * 70)
    print("""
    | Coin Age    | Trust Level  | ML Weight | Action                    |
    |-------------|--------------|-----------|---------------------------|
    | < 30 days   | UNRELIABLE   | 0%        | ML disabled, whale only   |
    | 30-90 days  | LIMITED      | 30%       | ML reduced, whale priority|
    | 90-180 days | ACCEPTABLE   | 70%       | Normal with caution       |
    | > 180 days  | RELIABLE     | 100%      | Full trust                |
    """)
    
    print("\n" + "=" * 70)
    print("BADGES TEST")
    print("=" * 70)
    
    for symbol, whale_pct, price_change in [('PEPEUSDT', 70, 30), ('BTCUSDT', 55, 2)]:
        risks = check_coin_risks(symbol, whale_pct, price_change)
        badges = get_risk_badges_html(risks)
        ml_level = risks['ml_trust_level']
        print(f"\n{symbol}: ML={ml_level} | {'[Has badges]' if badges else '[No badges]'}")