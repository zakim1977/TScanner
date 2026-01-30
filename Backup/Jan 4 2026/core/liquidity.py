"""
Liquidity Scoring Module
=========================

Evaluates market liquidity to:
1. Warn about low liquidity coins
2. Penalize score for illiquid markets
3. Suggest appropriate position sizes

Liquidity affects EXECUTION, not prediction accuracy.
ML learns patterns; liquidity determines if you can trade them profitably.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY THRESHOLDS (in USD)
# ═══════════════════════════════════════════════════════════════════════════════

LIQUIDITY_TIERS = {
    'EXCELLENT': {
        'min_volume_24h': 500_000_000,  # $500M+
        'score_bonus': 5,
        'position_pct': 0.001,  # 0.1% of 24h volume max position
        'label': '🟢 Excellent',
        'description': 'Institutional-grade liquidity'
    },
    'HIGH': {
        'min_volume_24h': 100_000_000,  # $100M+
        'score_bonus': 0,
        'position_pct': 0.0005,  # 0.05% of volume
        'label': '🟢 High',
        'description': 'Very liquid, minimal slippage'
    },
    'MEDIUM': {
        'min_volume_24h': 20_000_000,  # $20M+
        'score_bonus': 0,
        'position_pct': 0.0003,  # 0.03% of volume
        'label': '🟡 Medium',
        'description': 'Adequate liquidity for most trades'
    },
    'LOW': {
        'min_volume_24h': 5_000_000,  # $5M+
        'score_penalty': -8,
        'position_pct': 0.0001,  # 0.01% of volume
        'label': '🟠 Low',
        'description': 'May experience slippage on larger orders'
    },
    'VERY_LOW': {
        'min_volume_24h': 1_000_000,  # $1M+
        'score_penalty': -15,
        'position_pct': 0.00005,  # 0.005% of volume
        'label': '🔴 Very Low',
        'description': 'High slippage risk, reduce position size'
    },
    'ILLIQUID': {
        'min_volume_24h': 0,
        'score_penalty': -25,
        'position_pct': 0.00001,  # 0.001% of volume
        'label': '⛔ Illiquid',
        'description': 'Avoid or use minimal size'
    }
}

# Mode-specific minimum volume requirements
MODE_MIN_VOLUME = {
    'Scalp': 50_000_000,      # $50M min for scalping (need fast execution)
    'Day Trade': 20_000_000,  # $20M min for day trading
    'Swing': 5_000_000,       # $5M min for swing
    'Investment': 1_000_000,  # $1M min for investment
}


@dataclass
class LiquidityScore:
    """Liquidity assessment result"""
    volume_24h: float
    tier: str
    label: str
    description: str
    score_adjustment: int
    max_position_usd: float
    max_position_pct: float
    meets_mode_requirement: bool
    mode_min_volume: float
    warning: Optional[str] = None


def get_liquidity_tier(volume_24h: float) -> Tuple[str, Dict]:
    """Get liquidity tier based on 24h volume"""
    for tier_name, tier_config in LIQUIDITY_TIERS.items():
        if volume_24h >= tier_config['min_volume_24h']:
            return tier_name, tier_config
    return 'ILLIQUID', LIQUIDITY_TIERS['ILLIQUID']


def calculate_liquidity_score(
    volume_24h: float,
    trading_mode: str = 'Day Trade',
    position_size_usd: float = 1000
) -> LiquidityScore:
    """
    Calculate liquidity score for a given volume.
    
    Args:
        volume_24h: 24-hour trading volume in USD
        trading_mode: Trading mode (Scalp, Day Trade, Swing, Investment)
        position_size_usd: Intended position size in USD
    
    Returns:
        LiquidityScore with tier, warnings, and position suggestions
    """
    tier_name, tier_config = get_liquidity_tier(volume_24h)
    
    # Get score adjustment
    score_adj = tier_config.get('score_bonus', 0) or tier_config.get('score_penalty', 0)
    
    # Calculate max position based on volume
    position_pct = tier_config['position_pct']
    max_position = volume_24h * position_pct
    
    # Check mode requirement
    mode_min = MODE_MIN_VOLUME.get(trading_mode, 10_000_000)
    meets_requirement = volume_24h >= mode_min
    
    # Generate warning if needed
    warning = None
    if not meets_requirement:
        warning = f"⚠️ Volume ${volume_24h/1e6:.1f}M below {trading_mode} minimum ${mode_min/1e6:.0f}M"
    elif tier_name in ['LOW', 'VERY_LOW', 'ILLIQUID']:
        warning = f"⚠️ Low liquidity - consider reducing position size"
    
    # Additional penalty if position size exceeds recommended max
    if position_size_usd > max_position:
        overage_ratio = position_size_usd / max_position
        if overage_ratio > 2:
            score_adj -= 5  # Extra penalty for oversized position
            warning = f"⚠️ Position ${position_size_usd:,.0f} exceeds recommended max ${max_position:,.0f}"
    
    return LiquidityScore(
        volume_24h=volume_24h,
        tier=tier_name,
        label=tier_config['label'],
        description=tier_config['description'],
        score_adjustment=score_adj,
        max_position_usd=max_position,
        max_position_pct=position_pct * 100,
        meets_mode_requirement=meets_requirement,
        mode_min_volume=mode_min,
        warning=warning
    )


def fetch_binance_volume(symbol: str) -> Optional[float]:
    """
    Fetch 24h volume for a symbol from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
    
    Returns:
        24h volume in USD or None if failed
    """
    try:
        # Try futures first (more accurate for our use case)
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return float(data.get('quoteVolume', 0))
        
        # Fallback to spot
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return float(data.get('quoteVolume', 0))
            
    except Exception as e:
        print(f"⚠️ Failed to fetch volume for {symbol}: {e}")
    
    return None


def fetch_all_volumes() -> Dict[str, float]:
    """
    Fetch 24h volumes for all futures pairs.
    
    Returns:
        Dict of {symbol: volume_usd}
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                item['symbol']: float(item.get('quoteVolume', 0))
                for item in data
                if item['symbol'].endswith('USDT')
            }
    except Exception as e:
        print(f"⚠️ Failed to fetch volumes: {e}")
    
    return {}


def format_volume(volume: float) -> str:
    """Format volume for display"""
    if volume >= 1_000_000_000:
        return f"${volume/1e9:.1f}B"
    elif volume >= 1_000_000:
        return f"${volume/1e6:.1f}M"
    elif volume >= 1_000:
        return f"${volume/1e3:.1f}K"
    else:
        return f"${volume:.0f}"


def get_liquidity_html(liquidity: LiquidityScore) -> str:
    """Generate HTML for liquidity display"""
    tier_colors = {
        'EXCELLENT': '#00ff88',
        'HIGH': '#00d4aa',
        'MEDIUM': '#ffcc00',
        'LOW': '#ff9900',
        'VERY_LOW': '#ff6600',
        'ILLIQUID': '#ff4444'
    }
    
    color = tier_colors.get(liquidity.tier, '#888')
    
    html = f"""
    <div style='background: #1a1a2e; padding: 10px; border-radius: 6px; border-left: 3px solid {color};'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #888;'>Liquidity:</span>
            <span style='color: {color}; font-weight: bold;'>{liquidity.label}</span>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 4px;'>
            <span style='color: #888; font-size: 0.85em;'>24h Volume:</span>
            <span style='color: #fff;'>{format_volume(liquidity.volume_24h)}</span>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 4px;'>
            <span style='color: #888; font-size: 0.85em;'>Max Position:</span>
            <span style='color: #ccc;'>{format_volume(liquidity.max_position_usd)}</span>
        </div>
    """
    
    if liquidity.warning:
        html += f"""
        <div style='color: #ffaa00; font-size: 0.8em; margin-top: 8px;'>{liquidity.warning}</div>
        """
    
    if liquidity.score_adjustment != 0:
        adj_color = '#ff4444' if liquidity.score_adjustment < 0 else '#00ff88'
        html += f"""
        <div style='color: {adj_color}; font-size: 0.8em; margin-top: 4px;'>
            Score adjustment: {liquidity.score_adjustment:+d} pts
        </div>
        """
    
    html += "</div>"
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test with different volumes
    test_volumes = [
        ("BTCUSDT", 30_000_000_000),   # $30B - Excellent
        ("ETHUSDT", 15_000_000_000),   # $15B - Excellent
        ("SUIUSDT", 800_000_000),      # $800M - Excellent
        ("ARUSDT", 50_000_000),        # $50M - Medium
        ("KITEUSDT", 5_000_000),       # $5M - Low
        ("RANDOMUSDT", 500_000),       # $500K - Very Low
    ]
    
    print("=" * 60)
    print("LIQUIDITY SCORING TEST")
    print("=" * 60)
    
    for symbol, volume in test_volumes:
        score = calculate_liquidity_score(volume, 'Day Trade', 5000)
        print(f"\n{symbol}:")
        print(f"  Volume: {format_volume(volume)}")
        print(f"  Tier: {score.label}")
        print(f"  Score Adj: {score.score_adjustment:+d}")
        print(f"  Max Position: {format_volume(score.max_position_usd)}")
        if score.warning:
            print(f"  Warning: {score.warning}")
