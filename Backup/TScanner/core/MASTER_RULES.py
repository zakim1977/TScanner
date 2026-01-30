"""
═══════════════════════════════════════════════════════════════════════════════
                    INVESTORIQ MASTER TRADING RULES
                    ================================
                    
    THIS IS THE SINGLE SOURCE OF TRUTH FOR ALL TRADING DECISIONS
    
    NO OTHER FILE should make trading decisions. Everything calls this.
    
    ONE FUNCTION: get_trade_decision()
    ONE SCORE: total_score (0-100)
    ONE ANSWER: action + direction + reasoning
    
═══════════════════════════════════════════════════════════════════════════════

CORE PHILOSOPHY:
----------------
1. Whale positioning is the LEADING indicator (they move first)
2. Retail positioning shows who will get trapped
3. OI + Price confirms the flow of money
4. Position in range determines entry quality
5. ALL factors must align for a valid trade

GOLDEN RULES:
-------------
Rule 1: Retail > Whale on LONG → NEVER GO LONG (trap incoming)
Rule 2: Whale > Retail on LONG → Squeeze potential (favor long)
Rule 3: OI falling + Price falling → Longs exiting (bearish)
Rule 4: OI rising + Price rising → New longs entering (bullish)
Rule 5: Everyone bullish at highs → Distribution (short setup)
Rule 6: Everyone bearish at lows → Capitulation (long setup)
Rule 7: MIDDLE position = wait for better entry
Rule 8: LATE position on LONG = tighten stops or skip

SCORING (0-100):
----------------
- Direction (Whale conviction): 0-40 points
- Squeeze (Whale vs Retail divergence): 0-30 points  
- Entry (Position + TA): 0-30 points

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS - Clear naming for all states
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    LEAN_LONG = "LEAN_LONG"
    NEUTRAL = "NEUTRAL"
    LEAN_SHORT = "LEAN_SHORT"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"


class Action(Enum):
    STRONG_LONG = "STRONG_LONG"       # High conviction long
    LONG_SETUP = "LONG_SETUP"         # Good long setup
    BUILDING_LONG = "BUILDING_LONG"   # Long forming
    MONITOR_LONG = "MONITOR_LONG"     # Watch for long
    CAUTION_LONG = "CAUTION_LONG"     # Long risky - retail trap
    WAIT = "WAIT"                     # No trade
    CAUTION_SHORT = "CAUTION_SHORT"   # Short risky
    MONITOR_SHORT = "MONITOR_SHORT"   # Watch for short
    BUILDING_SHORT = "BUILDING_SHORT" # Short forming
    SHORT_SETUP = "SHORT_SETUP"       # Good short setup
    STRONG_SHORT = "STRONG_SHORT"     # High conviction short


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Position(Enum):
    EARLY = "EARLY"     # 0-35% - Near lows (good for longs)
    MIDDLE = "MIDDLE"   # 35-65% - Mid range (wait)
    LATE = "LATE"       # 65-100% - Near highs (good for shorts)


class OISignal(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"   # OI↑ Price↑ = New longs
    BULLISH = "BULLISH"                  # OI↑ Price↓ = Accumulation
    BEARISH = "BEARISH"                  # OI↓ Price↑ = Shorts closing
    STRONG_BEARISH = "STRONG_BEARISH"   # OI↓ Price↓ = Longs exiting
    NEUTRAL = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS - All numbers in ONE place
# ═══════════════════════════════════════════════════════════════════════════════

class T:
    """All thresholds - NEVER use magic numbers elsewhere"""
    
    # Whale/Retail Positioning
    EXTREME_BULLISH = 75
    STRONG_BULLISH = 70
    BULLISH = 65
    LEAN_BULLISH = 60
    NEUTRAL = 50
    LEAN_BEARISH = 40
    BEARISH = 35
    STRONG_BEARISH = 30
    EXTREME_BEARISH = 25
    
    # Position in Range
    EARLY_MAX = 35      # 0-35% = EARLY
    LATE_MIN = 65       # 65-100% = LATE
    
    # OI Thresholds
    OI_STRONG = 3.0     # Strong OI change
    OI_MODERATE = 1.0   # Moderate OI change
    
    # Price Thresholds
    PRICE_SIGNIFICANT = 1.0  # Significant price move


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeDecision:
    """Complete trading decision with all context - THE ONLY OUTPUT NEEDED"""
    
    # Primary outputs
    action: str              # What to do: STRONG_LONG, WAIT, etc.
    trade_direction: str     # LONG, SHORT, or WAIT
    confidence: str          # HIGH, MEDIUM, LOW
    
    # Scores (0-100 scale)
    direction_score: int     # 0-40 points (whale conviction)
    squeeze_score: int       # 0-30 points (divergence/trap)
    entry_score: int         # 0-30 points (position + TA)
    total_score: int         # 0-100 points (THE score)
    
    # Labels for display
    direction_label: str     # BULLISH, LEAN_BULLISH, etc.
    squeeze_label: str       # SQUEEZE, SLIGHT_EDGE, TRAP, etc.
    position_label: str      # EARLY, MIDDLE, LATE
    oi_signal: str           # What OI+Price tells us
    
    # Explanations
    main_reason: str         # Primary reason for decision
    warnings: List[str]      # Any warnings/cautions
    story: str               # Full narrative explanation
    
    # Validity flags
    is_valid_long: bool      # Can we go long?
    is_valid_short: bool     # Can we go short?
    invalidation_reason: str # Why invalid (if applicable)
    
    # Stories for Combined Learning
    oi_story: str = ""           # What OI is telling us
    whale_story: str = ""        # What whales are doing  
    position_story: str = ""     # Entry timing analysis
    market_story: str = ""       # Market conditions (Fear/Greed, BTC, Liquidity)
    historical_story: str = ""   # Historical performance of similar setups
    conclusion: str = ""         # Final conclusion text
    conclusion_action: str = ""  # Action text (LONG NOW, WAIT, etc.)
    
    # Market Conditions
    fear_greed: int = 50         # Fear & Greed index used
    market_modifier: int = 0     # Score adjustment from market conditions
    
    # Historical Validation
    historical_win_rate: float = None   # Win rate of similar setups
    historical_sample_size: int = 0     # Number of similar setups found
    historical_modifier: int = 0        # Score adjustment from history


# ═══════════════════════════════════════════════════════════════════════════════
# OI + PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_oi_price(oi_change: float, price_change: float) -> Tuple[str, str, int]:
    """
    Analyze OI and Price combination.
    
    Returns: (signal, explanation, score_modifier)
    
    ╔═══════════════╦═══════════════╦════════════════════════════════════════╗
    ║ OI Change     ║ Price Change  ║ Meaning                                ║
    ╠═══════════════╬═══════════════╬════════════════════════════════════════╣
    ║ ↑ Rising      ║ ↑ Rising      ║ New LONGS entering (BULLISH)           ║
    ║ ↑ Rising      ║ ↓ Falling     ║ New SHORTS entering (BEARISH)          ║
    ║ ↓ Falling     ║ ↑ Rising      ║ SHORTS closing - CAUTION (temporary?)  ║
    ║ ↓ Falling     ║ ↓ Falling     ║ LONGS exiting/liquidating (BEARISH)    ║
    ╚═══════════════╩═══════════════╩════════════════════════════════════════╝
    """
    
    oi_rising = oi_change > T.OI_MODERATE
    oi_falling = oi_change < -T.OI_MODERATE
    price_rising = price_change > T.PRICE_SIGNIFICANT
    price_falling = price_change < -T.PRICE_SIGNIFICANT
    
    if oi_rising and price_rising:
        return "STRONG_BULLISH", "New longs entering - Money flowing IN with price UP", 4
    
    elif oi_rising and price_falling:
        return "BEARISH", "New shorts entering - Betting against price", -2
    
    elif oi_falling and price_rising:
        # CRITICAL: This is SHORT COVERING, not real buying!
        # Rally may be temporary - no new longs entering
        return "SHORT_COVERING", "Shorts covering - NO new longs! Rally may fade", -2
    
    elif oi_falling and price_falling:
        return "STRONG_BEARISH", "Longs exiting - Money flowing OUT with price DOWN", -4
    
    else:
        return "NEUTRAL", "No significant OI/Price signal", 0


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION IN RANGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_position(position_pct: float) -> Tuple[str, int]:
    """
    Determine position in range and score modifier.
    
    EARLY (0-35%): Good for longs, bad for shorts
    MIDDLE (35-65%): Wait for better entry
    LATE (65-100%): Good for shorts, bad for longs
    """
    
    if position_pct <= T.EARLY_MAX:
        return "EARLY", 8  # Bonus for good entry
    elif position_pct >= T.LATE_MIN:
        return "LATE", -4  # Penalty for late entry on longs
    else:
        return "MIDDLE", 0  # No bonus/penalty


# ═══════════════════════════════════════════════════════════════════════════════
# WHALE vs RETAIL ANALYSIS (THE CORE LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_whale_retail(whale_pct: float, retail_pct: float) -> dict:
    """
    THE MOST IMPORTANT FUNCTION
    
    Analyzes whale vs retail positioning to determine:
    1. Direction (bullish/bearish)
    2. Conviction (high/medium/low)
    3. Squeeze potential
    4. Trap risk
    """
    
    divergence = whale_pct - retail_pct
    
    result = {
        'whale_pct': whale_pct,
        'retail_pct': retail_pct,
        'divergence': divergence,
        'direction': 'NEUTRAL',
        'confidence': 'LOW',
        'direction_score': 15,
        'squeeze_type': 'NONE',
        'squeeze_score': 0,
        'is_trap': False,
        'trap_type': None,
        'warnings': [],
        'reason': ''
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 1: CHECK FOR RETAIL TRAP (Retail > Whale)
    # This is checked FIRST because it invalidates long setups
    # ANY amount of retail > whale is a warning!
    # ═══════════════════════════════════════════════════════════════════════
    
    if divergence < 0:  # Retail more bullish than whales by ANY amount
        result['is_trap'] = True
        result['trap_type'] = 'LONG_TRAP'
        result['warnings'].append(f"⚠️ RETAIL TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%")
        
        if divergence <= -15:
            result['squeeze_type'] = 'EXTREME_TRAP'
            result['squeeze_score'] = -15
            result['reason'] = f"🚨 EXTREME TRAP: Retail overleveraged by {abs(divergence):.0f}%"
        elif divergence <= -10:
            result['squeeze_type'] = 'HIGH_TRAP'
            result['squeeze_score'] = -10
            result['reason'] = f"⚠️ HIGH TRAP RISK: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        elif divergence <= -5:
            result['squeeze_type'] = 'MODERATE_TRAP'
            result['squeeze_score'] = -6
            result['reason'] = f"⚠️ TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        else:
            result['squeeze_type'] = 'SLIGHT_TRAP'
            result['squeeze_score'] = -3
            result['reason'] = f"⚠️ Caution: Retail ({retail_pct:.0f}%) slightly > Whales ({whale_pct:.0f}%)"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 2: CHECK FOR SQUEEZE (Whale > Retail)
    # Whales positioned, retail not = squeeze incoming
    # ═══════════════════════════════════════════════════════════════════════
    
    elif divergence >= 15:
        result['squeeze_type'] = 'HIGH_SQUEEZE'
        result['squeeze_score'] = 20
        result['reason'] = f"🔥 SQUEEZE: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
        
    elif divergence >= 10:
        result['squeeze_type'] = 'MODERATE_SQUEEZE'
        result['squeeze_score'] = 14
        result['reason'] = f"Squeeze potential: {divergence:.0f}% divergence"
        
    elif divergence >= 5:
        result['squeeze_type'] = 'SLIGHT_EDGE'
        result['squeeze_score'] = 8
        result['reason'] = f"Slight edge: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
    
    else:
        result['squeeze_type'] = 'NO_EDGE'
        result['squeeze_score'] = 0
        result['reason'] = "No significant whale/retail divergence"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 3: WHALE CONVICTION (Direction)
    # ═══════════════════════════════════════════════════════════════════════
    
    if whale_pct >= T.EXTREME_BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 40
        
    elif whale_pct >= T.STRONG_BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 36
        
    elif whale_pct >= T.BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 32
        
    elif whale_pct >= T.LEAN_BULLISH:
        result['direction'] = 'LEAN_BULLISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 28
        
    elif whale_pct <= T.EXTREME_BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 40
        
    elif whale_pct <= T.STRONG_BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 36
        
    elif whale_pct <= T.BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 32
        
    elif whale_pct <= T.LEAN_BEARISH:
        result['direction'] = 'LEAN_BEARISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 28
        
    else:
        result['direction'] = 'NEUTRAL'
        result['confidence'] = 'LOW'
        result['direction_score'] = 15
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def check_special_scenarios(
    whale_pct: float, 
    retail_pct: float, 
    position: str,
    oi_signal: str
) -> Optional[dict]:
    """
    Check for special market conditions:
    - CAPITULATION: Everyone bearish at lows → Reversal long
    - EUPHORIA: Everyone bullish at highs → Reversal short
    """
    
    # CAPITULATION: Both whale and retail bearish at EARLY position
    if whale_pct <= T.BEARISH and retail_pct <= T.BEARISH and position == "EARLY":
        return {
            'scenario': 'CAPITULATION',
            'action': 'LONG NOW',
            'trade_direction': 'LONG',
            'confidence': 'HIGH',
            'score_bonus': 15,
            'reason': 'Everyone gave up at lows - reversal likely'
        }
    
    # EUPHORIA: Both whale and retail bullish at LATE position
    if whale_pct >= T.BULLISH and retail_pct >= T.BULLISH and position == "LATE":
        return {
            'scenario': 'EUPHORIA',
            'action': 'SHORT NOW',
            'trade_direction': 'SHORT',
            'confidence': 'HIGH',
            'score_bonus': 15,
            'reason': 'Everyone bullish at highs - distribution likely'
        }
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DECISION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_trade_decision(
    whale_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    price_change: float = 0,
    position_pct: float = 50,
    ta_score: float = 50,
    money_flow_phase: str = "UNKNOWN",
    # NEW: Market Conditions
    fear_greed: int = 50,           # Fear & Greed index (0-100)
    btc_change_24h: float = 0,      # BTC 24h change %
    is_options_expiry: bool = False, # Is today options expiry?
    is_holiday: bool = False,        # Holiday/weekend thin liquidity?
    # NEW: Historical Validation (input from database lookup)
    historical_win_rate: float = None,    # Win rate from similar setups (0-100)
    historical_sample_size: int = 0,      # Number of similar historical setups
    # NEW: SMC Level Proximity (from Pulse Flow / SMC detector)
    at_bullish_ob: bool = False,     # Price is AT a bullish Order Block
    at_bearish_ob: bool = False,     # Price is AT a bearish Order Block
    near_bullish_ob: bool = False,   # Price is NEAR a bullish Order Block (within 2%)
    near_bearish_ob: bool = False,   # Price is NEAR a bearish Order Block (within 2%)
    at_support: bool = False,        # Price is at key support level
    at_resistance: bool = False      # Price is at key resistance level
) -> TradeDecision:
    """
    ═══════════════════════════════════════════════════════════════════════════
    THE MASTER DECISION FUNCTION
    ═══════════════════════════════════════════════════════════════════════════
    
    This is the ONLY function that should be called for trading decisions.
    ALL other code paths should use this.
    
    Args:
        whale_pct: Whale long percentage (0-100)
        retail_pct: Retail long percentage (0-100)
        oi_change: Open Interest change % (24h)
        price_change: Price change % (24h)
        position_pct: Position in range (0=low, 100=high)
        ta_score: Technical analysis score (0-100)
        money_flow_phase: ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
        fear_greed: Fear & Greed index (0-100, lower = more fear)
        btc_change_24h: BTC 24h price change %
        is_options_expiry: True if major options expiry today
        is_holiday: True if holiday/weekend (thin liquidity)
        historical_win_rate: Win rate from similar historical setups (0-100)
        historical_sample_size: Number of similar setups in database
        at_bullish_ob: Price is AT a bullish Order Block (best long entry)
        at_bearish_ob: Price is AT a bearish Order Block (best short entry)
        near_bullish_ob: Price is NEAR bullish OB within 2% (good long entry)
        near_bearish_ob: Price is NEAR bearish OB within 2% (good short entry)
        at_support: Price is at key support level
        at_resistance: Price is at key resistance level
    
    Returns:
        TradeDecision with complete analysis
    
    MARKET CONDITIONS LOGIC:
    - EXTREME FEAR + Whales Buying = STRONG BUY (smart money accumulating)
    - EXTREME FEAR + Whales Selling = WAIT (capitulation not over)
    - EXTREME GREED + Whales Selling = STRONG SHORT (distribution)
    - EXTREME GREED + Whales Buying = CAUTION (late/FOMO trap)
    - Options Expiry = Reduce confidence (expect volatility)
    - Holiday/Weekend = Reduce confidence (thin liquidity)
    
    HISTORICAL VALIDATION LOGIC:
    - Win rate >= 70% with 15+ samples = +12 pts (proven setup!)
    - Win rate >= 60% with 15+ samples = +8 pts
    - Win rate <= 40% with 15+ samples = -15 pts (AVOID!)
    - Less than 5 samples = no adjustment (insufficient data)
    
    SMC LEVEL LOGIC (NEW!):
    - LATE position + at_bullish_ob = OVERRIDE late penalty for longs (good entry!)
    - EARLY position + at_bearish_ob = OVERRIDE early penalty for shorts (good entry!)
    - Being at OB/support/resistance trumps raw position percentage
    """
    
    warnings = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Analyze Whale vs Retail
    # ═══════════════════════════════════════════════════════════════════════
    
    wr_analysis = analyze_whale_retail(whale_pct, retail_pct)
    warnings.extend(wr_analysis['warnings'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Analyze OI + Price
    # ═══════════════════════════════════════════════════════════════════════
    
    oi_signal, oi_explanation, oi_modifier = analyze_oi_price(oi_change, price_change)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Get Position in Range
    # ═══════════════════════════════════════════════════════════════════════
    
    position_label, position_modifier = get_position(position_pct)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3.5: Analyze Market Conditions (NEW!)
    # Fear/Greed + BTC trend + Options expiry + Liquidity
    # ═══════════════════════════════════════════════════════════════════════
    
    market_modifier = 0
    market_warnings = []
    market_story_parts = []
    
    # Determine whale direction for fear/greed logic
    whale_direction = 'BULLISH' if whale_pct >= 60 else ('BEARISH' if whale_pct <= 40 else 'NEUTRAL')
    
    # === FEAR & GREED ANALYSIS ===
    # This is the Warren Buffett logic: "Be greedy when others are fearful"
    
    if fear_greed <= 25:  # EXTREME FEAR
        if whale_direction == 'BULLISH':
            # Smart money buying while retail panics = GOLDEN OPPORTUNITY
            market_modifier += 10
            market_story_parts.append(f"🟢 EXTREME FEAR ({fear_greed}) + Whales BUYING = Smart money accumulation!")
        elif whale_direction == 'BEARISH':
            # Everyone selling including whales = capitulation not over
            market_modifier -= 10
            market_warnings.append(f"⚠️ EXTREME FEAR + Whales also selling = Capitulation")
            market_story_parts.append(f"🔴 EXTREME FEAR ({fear_greed}) + Whales SELLING = Wait for capitulation to end")
        else:
            market_story_parts.append(f"😱 EXTREME FEAR ({fear_greed}) - Watch for whale accumulation")
            
    elif fear_greed <= 45:  # FEAR
        if whale_direction == 'BULLISH':
            market_modifier += 5
            market_story_parts.append(f"🟢 FEAR ({fear_greed}) + Whales bullish = Accumulation zone")
        elif whale_direction == 'BEARISH':
            market_modifier -= 5
            market_story_parts.append(f"⚠️ FEAR ({fear_greed}) + Whales bearish = More downside likely")
            
    elif fear_greed >= 75:  # EXTREME GREED
        if whale_direction == 'BEARISH':
            # Smart money selling while retail is greedy = TOP SIGNAL
            market_modifier += 5  # Good for shorts
            market_story_parts.append(f"🔴 EXTREME GREED ({fear_greed}) + Whales SELLING = Distribution/Top signal!")
        elif whale_direction == 'BULLISH':
            # Even whales bullish at extreme greed = FOMO trap risk
            market_modifier -= 10
            market_warnings.append(f"⚠️ EXTREME GREED + Whales bullish = Late stage, FOMO trap risk")
            market_story_parts.append(f"⚠️ EXTREME GREED ({fear_greed}) - Even whales bullish could be FOMO trap")
        else:
            market_modifier -= 5
            market_story_parts.append(f"🤑 EXTREME GREED ({fear_greed}) - Market likely overextended")
            
    elif fear_greed >= 55:  # GREED
        if whale_direction == 'BEARISH':
            market_modifier += 3  # Slight edge for shorts
            market_story_parts.append(f"📊 GREED ({fear_greed}) + Whales bearish = Distribution phase")
        elif whale_direction == 'BULLISH':
            market_modifier -= 3
            market_warnings.append(f"⚠️ GREED ({fear_greed}) + Whales bullish = Late stage entry")
            market_story_parts.append(f"⚠️ GREED ({fear_greed}) - Be careful chasing longs")
    else:
        # NEUTRAL (45-55)
        market_story_parts.append(f"😐 Neutral sentiment ({fear_greed})")
    
    # === BTC TREND CHECK ===
    # Altcoins follow BTC - if BTC dumps, everything dumps
    
    if btc_change_24h <= -5:
        market_modifier -= 10
        market_warnings.append(f"🔴 BTC DUMP: {btc_change_24h:+.1f}% - Altcoins will follow!")
        market_story_parts.append(f"🔴 BTC crashing {btc_change_24h:+.1f}% - Risk off for all alts")
    elif btc_change_24h <= -3:
        market_modifier -= 5
        market_warnings.append(f"⚠️ BTC down {btc_change_24h:+.1f}%")
        market_story_parts.append(f"⚠️ BTC weak {btc_change_24h:+.1f}% - Alts may follow")
    elif btc_change_24h >= 5:
        market_modifier += 5
        market_story_parts.append(f"🟢 BTC pumping {btc_change_24h:+.1f}% - Risk on environment")
    elif btc_change_24h >= 3:
        market_modifier += 3
        market_story_parts.append(f"🟢 BTC strong {btc_change_24h:+.1f}%")
    
    # === OPTIONS EXPIRY CHECK ===
    if is_options_expiry:
        market_modifier -= 10
        market_warnings.append("⚠️ MAJOR OPTIONS EXPIRY - Expect HIGH volatility!")
        market_story_parts.append("📅 Options expiry day - Expect wild swings, consider waiting")
    
    # === LIQUIDITY CHECK ===
    if is_holiday:
        market_modifier -= 10
        market_warnings.append("⚠️ HOLIDAY/WEEKEND - Thin liquidity, wild swings possible!")
        market_story_parts.append("💧 Thin liquidity - Prices can move on low volume")
    
    # Build final market story
    market_story = " | ".join(market_story_parts) if market_story_parts else "Market conditions neutral"
    
    # Add market warnings to main warnings
    warnings.extend(market_warnings)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3.6: Historical Validation Adjustment (NEW!)
    # Uses actual historical performance of similar setups
    # ═══════════════════════════════════════════════════════════════════════
    
    historical_modifier = 0
    historical_story = ""
    
    if historical_win_rate is not None and historical_sample_size >= 5:
        # We have enough historical data to trust
        
        if historical_sample_size >= 15:
            # HIGH confidence in historical data
            if historical_win_rate >= 70:
                historical_modifier = +12
                historical_story = f"🟢 PROVEN SETUP: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples) - Historically excellent!"
            elif historical_win_rate >= 60:
                historical_modifier = +8
                historical_story = f"🟢 Good history: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            elif historical_win_rate >= 50:
                historical_modifier = +3
                historical_story = f"📊 Decent history: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            elif historical_win_rate >= 40:
                historical_modifier = -5
                historical_story = f"⚠️ Below average: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            else:
                historical_modifier = -15
                historical_story = f"🔴 POOR HISTORY: Only {historical_win_rate:.0f}% win rate ({historical_sample_size} samples) - AVOID THIS SETUP!"
                warnings.append(f"🔴 Historical win rate only {historical_win_rate:.0f}% - consider skipping")
                
        elif historical_sample_size >= 10:
            # MEDIUM confidence
            if historical_win_rate >= 70:
                historical_modifier = +8
                historical_story = f"🟢 Good track record: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate >= 55:
                historical_modifier = +4
                historical_story = f"📊 Positive history: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate <= 40:
                historical_modifier = -10
                historical_story = f"🔴 Poor history: {historical_win_rate:.0f}% ({historical_sample_size} samples) - Caution!"
                warnings.append(f"⚠️ Similar setups only {historical_win_rate:.0f}% win rate")
            else:
                historical_story = f"📊 Mixed results: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
                
        else:  # 5-9 samples
            # LOW confidence - smaller adjustments
            if historical_win_rate >= 75:
                historical_modifier = +5
                historical_story = f"📊 Limited data positive: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate <= 35:
                historical_modifier = -8
                historical_story = f"⚠️ Limited data negative: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            else:
                historical_story = f"📊 Limited data: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
    
    elif historical_sample_size > 0 and historical_sample_size < 5:
        historical_story = f"📊 Insufficient history: Only {historical_sample_size} similar setups found"
    else:
        historical_story = "📊 No historical data for this exact setup"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Check Special Scenarios
    # ═══════════════════════════════════════════════════════════════════════
    
    special = check_special_scenarios(whale_pct, retail_pct, position_label, oi_signal)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Calculate Entry Score (SMC-AWARE!)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Determine if price is at a key SMC level
    at_bullish_level = at_bullish_ob or near_bullish_ob or at_support
    at_bearish_level = at_bearish_ob or near_bearish_ob or at_resistance
    
    # TA contributes up to 15 points
    ta_contribution = int((ta_score / 100) * 15)
    
    # Position contributes up to 15 points - BUT SMC levels can OVERRIDE!
    # 
    # KEY INSIGHT: Position % alone doesn't tell the full story.
    # Being LATE (86%) but AT a bullish Order Block is actually a GOOD entry!
    # The OB provides support that raw position % doesn't account for.
    #
    smc_override = False
    smc_override_reason = ""
    
    if position_label == "EARLY":
        position_contribution = 15
    elif position_label == "MIDDLE":
        position_contribution = 8
    else:  # LATE
        position_contribution = 3  # Default late penalty
        
        # SMC OVERRIDE: If LATE but at bullish OB/support → Good entry for longs!
        if at_bullish_level:
            position_contribution = 12  # Upgrade from 3 to 12
            smc_override = True
            if at_bullish_ob:
                smc_override_reason = "LATE but AT bullish OB - good entry!"
            elif near_bullish_ob:
                smc_override_reason = "LATE but NEAR bullish OB - decent entry"
            else:
                smc_override_reason = "LATE but AT support level"
    
    # For shorts: EARLY position at bearish OB is actually good
    # (We'll check direction later and apply this bonus)
    short_entry_bonus = 0
    if position_label == "EARLY" and at_bearish_level:
        short_entry_bonus = 8  # Bonus for shorts when at bearish OB at lows
        if at_bearish_ob:
            smc_override_reason = "EARLY but AT bearish OB - good short entry!"
        elif near_bearish_ob:
            smc_override_reason = "EARLY but NEAR bearish OB - decent short"
    
    entry_score = ta_contribution + position_contribution
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Calculate Total Score
    # ═══════════════════════════════════════════════════════════════════════
    
    direction_score = wr_analysis['direction_score']
    squeeze_score = max(0, wr_analysis['squeeze_score'])  # Don't go negative for total
    
    # Include market_modifier in total score
    total_score = direction_score + squeeze_score + entry_score + oi_modifier + market_modifier + historical_modifier
    total_score = max(0, min(100, total_score))
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: DETERMINE FINAL ACTION
    # This is where ALL the rules come together
    # ═══════════════════════════════════════════════════════════════════════
    
    direction = wr_analysis['direction']
    confidence = wr_analysis['confidence']
    is_trap = wr_analysis['is_trap']
    squeeze_type = wr_analysis['squeeze_type']
    
    # Default values
    action = "WAIT"
    trade_direction = "WAIT"
    is_valid_long = True
    is_valid_short = True
    invalidation_reason = ""
    
    # --- SPECIAL SCENARIO OVERRIDE ---
    if special:
        action = special['action']
        trade_direction = special['trade_direction']
        total_score += special['score_bonus']
        main_reason = special['reason']
    
    # --- TRAP DETECTION (HIGHEST PRIORITY) ---
    elif is_trap:
        # Retail > Whale = NEVER go long
        is_valid_long = False
        invalidation_reason = f"Retail ({retail_pct:.0f}%) > Whale ({whale_pct:.0f}%) - trap risk"
        
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            action = "CAUTION_LONG"
            trade_direction = "WAIT"
            main_reason = f"⚠️ RETAIL TRAP: {invalidation_reason}"
        else:
            action = "WAIT"
            trade_direction = "WAIT"
            main_reason = f"No clear direction + trap risk"
    
    # --- SHORT COVERING CHECK (OI↓ + Price↑) ---
    # This is a CRITICAL pattern: Price rising but money LEAVING
    # Shorts are covering, but NO NEW LONGS entering = temporary rally
    elif oi_change < -2 and price_change > 3 and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"⚠️ SHORT COVERING: OI {oi_change:+.1f}% but Price {price_change:+.1f}%")
        action = "SHORT_COVERING"
        trade_direction = "WAIT"  # Don't chase short covering rallies!
        is_valid_long = False
        invalidation_reason = "Short covering rally - no new longs entering"
        main_reason = f"Short covering rally (OI↓{oi_change:.1f}% + Price↑{price_change:.1f}%) - Wait for OI to rise"
        # Significantly reduce score - this is not a sustainable rally
        direction_score = max(15, direction_score - 15)
        squeeze_score = max(5, squeeze_score - 10)
    
    # --- MONEY FLOW PHASE CHECK ---
    # DISTRIBUTION / PROFIT TAKING = Smart money selling into strength
    # This is a WARNING even with bullish whale positioning!
    elif money_flow_phase.upper() in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING'] and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"⚠️ PROFIT TAKING: Smart money may be selling into strength")
        # Don't change direction to WAIT, but reduce conviction
        action = "CAUTION_LONG"
        # Reduce scores - this is a late-stage signal
        direction_score = max(20, direction_score - 8)
        squeeze_score = max(10, squeeze_score - 5)
        entry_score = max(5, entry_score - 5)
        main_reason = f"Bullish but PROFIT TAKING - Smart money may be distributing"
        # If also in LATE position, this is very risky
        if position_label == "LATE":
            trade_direction = "WAIT"
            is_valid_long = False
            invalidation_reason = "Profit taking at highs - distribution likely"
            main_reason = f"⚠️ PROFIT TAKING at LATE position ({position_pct:.0f}%) - High risk!"
    
    # --- MARKDOWN PHASE CHECK ---
    # MARKDOWN = Already trending down, be extra cautious on longs
    elif money_flow_phase.upper() == 'MARKDOWN' and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"⚠️ MARKDOWN phase: Trend is down, longs are counter-trend")
        action = "CAUTION_LONG"
        direction_score = max(15, direction_score - 10)
        main_reason = f"Counter-trend long in MARKDOWN phase - high risk"
    
    # --- OI CONFLICT CHECK (OI↓ + Price↓) ---
    elif oi_signal == "STRONG_BEARISH" and direction in ['BULLISH', 'LEAN_BULLISH']:
        # OI falling + Price falling = Longs exiting
        # Even if whales are long, money is leaving
        warnings.append(f"⚠️ OI conflict: Longs exiting (OI {oi_change:+.1f}%, Price {price_change:+.1f}%)")
        action = "CAUTION_LONG"
        trade_direction = "WAIT"
        is_valid_long = False
        invalidation_reason = "Money leaving (OI↓ + Price↓)"
        main_reason = f"OI conflict - wait for OI to stabilize"
    
    # --- POSITION CHECK FOR LONGS ---
    # LATE position = DON'T CHASE! Wait for pullback.
    # EXCEPTION: If at bullish OB/support, LATE is actually a GOOD entry!
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and position_label == "LATE" and not at_bullish_level:
        warnings.append(f"⚠️ LATE entry ({position_pct:.0f}%) - Wait for pullback!")
        
        # Even with squeeze, LATE is risky - wait for dip
        if confidence == 'HIGH' and squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
            action = "WAIT_FOR_DIP"
            trade_direction = "WAIT"  # NOT LONG! Wait for better entry
            main_reason = f"Squeeze potential but LATE ({position_pct:.0f}%) - wait for pullback"
            # Reduce score for late entry
            entry_score = max(0, entry_score - 10)
        else:
            action = "WAIT_FOR_DIP"
            trade_direction = "WAIT"  # NOT LONG! Don't chase
            main_reason = f"Bullish but LATE ({position_pct:.0f}%) - Don't chase! Wait for dip"
            entry_score = max(0, entry_score - 8)
    
    # --- LATE BUT AT BULLISH OB/SUPPORT (SMC OVERRIDE!) ---
    # Position is LATE but price is at a key support level - this IS a good entry!
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and position_label == "LATE" and at_bullish_level:
        # Don't penalize - the OB/support provides a valid entry
        if at_bullish_ob:
            action = "LONG_SETUP"
            trade_direction = "LONG"
            main_reason = f"🎯 LATE but AT bullish OB - valid entry with defined risk!"
            warnings.append(f"✅ Position {position_pct:.0f}% but at bullish Order Block")
        elif near_bullish_ob:
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"LATE but NEAR bullish OB - decent entry"
            warnings.append(f"📍 Position {position_pct:.0f}% but near bullish Order Block")
        else:  # at_support
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"LATE but AT support level - entry with structure"
            warnings.append(f"📍 Position {position_pct:.0f}% but at key support")
    
    # --- HIGH CONFIDENCE BULLISH ---
    elif direction == 'BULLISH' and confidence == 'HIGH' and not is_trap:
        if squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
            action = "STRONG_LONG"
            trade_direction = "LONG"
            main_reason = f"🔥 High conviction + Squeeze: {wr_analysis['reason']}"
        elif position_label == "EARLY":
            action = "LONG_SETUP"
            trade_direction = "LONG"
            main_reason = f"Strong setup: Whale {whale_pct:.0f}% + EARLY position"
        else:
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"Building long: Whale conviction {whale_pct:.0f}%"
    
    # --- MEDIUM CONFIDENCE BULLISH ---
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and not is_trap:
        if position_label == "EARLY":
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish ({whale_pct:.0f}%) + good entry"
        elif position_label == "MIDDLE":
            action = "MONITOR_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish but MID range - wait for dip"
        else:
            action = "MONITOR_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish but LATE - wait for pullback"
    
    # --- HIGH CONFIDENCE BEARISH ---
    elif direction == 'BEARISH' and confidence == 'HIGH':
        if position_label == "LATE":
            action = "STRONG_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Strong short: Whale {whale_pct:.0f}% bearish at highs"
        elif position_label == "MIDDLE":
            action = "SHORT_SETUP"
            trade_direction = "SHORT"
            main_reason = f"Short setup: Whale conviction"
        else:
            # EARLY position = near lows = BAD for shorts
            action = "WAIT_FOR_BOUNCE"
            trade_direction = "WAIT"  # NOT SHORT! Don't short the lows
            main_reason = f"Bearish but at LOWS ({position_pct:.0f}%) - Don't short here! Wait for bounce"
            warnings.append(f"⚠️ EARLY position ({position_pct:.0f}%) - bad entry for shorts")
    
    # --- MEDIUM CONFIDENCE BEARISH ---
    elif direction in ['BEARISH', 'LEAN_BEARISH']:
        if position_label == "LATE":
            action = "BUILDING_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Lean bearish ({whale_pct:.0f}%) at highs"
        elif position_label == "MIDDLE":
            action = "MONITOR_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Lean bearish ({whale_pct:.0f}%)"
        else:
            # EARLY = near lows = bad for shorts
            action = "WAIT_FOR_BOUNCE"
            trade_direction = "WAIT"
            main_reason = f"Bearish but near LOWS - don't short bottom"
            warnings.append(f"⚠️ Near lows ({position_pct:.0f}%) - wait for bounce")
    
    # --- NEUTRAL ---
    else:
        action = "WAIT"
        trade_direction = "WAIT"
        main_reason = f"No clear edge - Whale {whale_pct:.0f}%, Retail {retail_pct:.0f}%"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECALCULATE TOTAL SCORE (after all adjustments)
    # Must include ALL modifiers: oi, market, historical
    # ═══════════════════════════════════════════════════════════════════════
    total_score = direction_score + squeeze_score + entry_score + oi_modifier + market_modifier + historical_modifier
    total_score = max(0, min(100, total_score))
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Build Stories for Combined Learning
    # ═══════════════════════════════════════════════════════════════════════
    
    story_parts = []
    
    # OI Story
    if oi_signal == "STRONG_BULLISH":
        oi_story = f"📈 OI up {oi_change:+.1f}% + Price up {price_change:+.1f}% → New LONGS entering. Fresh buying pressure!"
    elif oi_signal == "STRONG_BEARISH":
        oi_story = f"📉 OI down {oi_change:+.1f}% + Price down {price_change:+.1f}% → Longs EXITING. Selling pressure or liquidations."
    elif oi_signal == "SHORT_COVERING":
        oi_story = f"⚠️ SHORT COVERING: OI {oi_change:+.1f}% + Price {price_change:+.1f}% → Shorts closing but NO NEW LONGS! Rally may be temporary."
    elif oi_signal == "BULLISH":
        oi_story = f"📊 OI down {oi_change:+.1f}% + Price up {price_change:+.1f}% → Shorts CLOSING. Some squeeze potential."
    elif oi_signal == "BEARISH":
        oi_story = f"📊 OI up {oi_change:+.1f}% + Price down {price_change:+.1f}% → New SHORTS entering. Bearish pressure."
    else:
        # Neutral but still informative
        if oi_change < -2:
            oi_story = f"📉 OI falling {oi_change:+.1f}% (positions closing) | Price {price_change:+.1f}%"
        elif oi_change > 2:
            oi_story = f"📈 OI rising {oi_change:+.1f}% (new positions) | Price {price_change:+.1f}%"
        else:
            oi_story = f"📊 OI {oi_change:+.1f}% | Price {price_change:+.1f}% → No significant flow signal."
    
    # Whale Story
    divergence = whale_pct - retail_pct
    if is_trap:
        whale_story = f"⚠️ TRAP WARNING: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%! Retail overleveraged - whales may dump on them."
    elif divergence >= 15:
        whale_story = f"🐋 SQUEEZE SETUP: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% → {divergence:.0f}% divergence! Retail shorts will be liquidated."
    elif divergence >= 5:
        whale_story = f"🐋 Whales {whale_pct:.0f}% LONG vs Retail {retail_pct:.0f}% → Slight edge to bulls."
    elif divergence <= -10:
        whale_story = f"⚠️ Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% → Retail more bullish. Caution on longs!"
    else:
        whale_story = f"🐋 Whales {whale_pct:.0f}% | Retail {retail_pct:.0f}% → No significant divergence."
    
    # Position Story - based on underlying direction, not trade_direction (which may be WAIT)
    # NOW WITH SMC CONTEXT!
    if position_label == "EARLY":
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            position_story = f"🎯 EARLY entry ({position_pct:.0f}%) → Near lows. Excellent R:R for longs!"
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            if at_bearish_level:
                position_story = f"🎯 EARLY ({position_pct:.0f}%) but AT bearish OB → Valid short entry with structure!"
            else:
                position_story = f"❌ EARLY ({position_pct:.0f}%) → Near LOWS! BAD for shorts - don't short the bottom!"
        else:
            position_story = f"📍 Position: {position_pct:.0f}% of range → Near lows."
    elif position_label == "LATE":
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            # SMC OVERRIDE for position story
            if at_bullish_ob:
                position_story = f"🎯 Position {position_pct:.0f}% but **AT BULLISH OB** → Valid entry! OB provides support."
            elif near_bullish_ob:
                position_story = f"📍 Position {position_pct:.0f}% but **NEAR BULLISH OB** → Decent entry with nearby support."
            elif at_support:
                position_story = f"📍 Position {position_pct:.0f}% but **AT SUPPORT** → Entry with structure."
            else:
                position_story = f"❌ LATE entry ({position_pct:.0f}%) → Near HIGHS! DON'T CHASE - wait for pullback!"
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            position_story = f"🎯 Good entry for shorts ({position_pct:.0f}%) → Near highs."
        else:
            position_story = f"📍 Position: {position_pct:.0f}% of range → Near highs."
    else:
        position_story = f"📍 MID-RANGE ({position_pct:.0f}%) → Wait for price to reach support/resistance."
    
    # Add money flow phase to position story if significant
    mf_upper = money_flow_phase.upper() if money_flow_phase else 'UNKNOWN'
    if mf_upper in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING']:
        position_story += f"\n\n⚠️ **MONEY FLOW: PROFIT TAKING** - Smart money may be selling into this rally!"
    elif mf_upper == 'MARKDOWN':
        position_story += f"\n\n⚠️ **MONEY FLOW: MARKDOWN** - Trend is down, be cautious with longs."
    elif mf_upper == 'ACCUMULATION':
        position_story += f"\n\n✅ **MONEY FLOW: ACCUMULATION** - Smart money building positions."
    elif mf_upper == 'MARKUP':
        position_story += f"\n\n✅ **MONEY FLOW: MARKUP** - Strong uptrend with volume."
    
    # Build summary story
    story_parts.append(f"🐋 Whales: {whale_pct:.0f}% long")
    story_parts.append(f"👥 Retail: {retail_pct:.0f}% long")
    
    if divergence > 10:
        story_parts.append(f"📊 Divergence: +{divergence:.0f}% (whales more bullish)")
    elif divergence < -5:
        story_parts.append(f"📊 Divergence: {divergence:.0f}% (⚠️ retail more bullish)")
    
    story_parts.append(f"📈 OI Signal: {oi_signal} ({oi_explanation})")
    story_parts.append(f"📍 Position: {position_label} ({position_pct:.0f}%)")
    
    story = " | ".join(story_parts)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: Build Conclusion (clean, no duplication)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Create short reason for conclusion
    divergence = whale_pct - retail_pct
    
    if is_trap:
        short_reason = f"Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        conclusion = f"⚠️ RETAIL TRAP DETECTED - {short_reason}"
        conclusion_action = "WAIT - Don't fight the trap"
    elif special:
        conclusion = f"🎯 {special['scenario']} - {special['reason']}"
        conclusion_action = special['action']
    elif action == "WAIT_FOR_DIP":
        # Bullish but LATE - don't chase!
        conclusion = f"⏳ BULLISH BUT LATE ({position_pct:.0f}%) - Don't chase!"
        conclusion_action = "WAIT FOR DIP"
    elif action == "WAIT_FOR_BOUNCE":
        # Bearish but EARLY - don't short lows!
        conclusion = f"⏳ BEARISH BUT AT LOWS ({position_pct:.0f}%) - Don't short!"
        conclusion_action = "WAIT FOR BOUNCE"
    elif action == "SHORT_COVERING":
        # OI falling + Price rising = shorts covering, not new longs
        conclusion = f"⚠️ SHORT COVERING RALLY - OI {oi_change:+.1f}% + Price {price_change:+.1f}%"
        conclusion_action = "WAIT - Rally may be temporary"
    elif action == "CAUTION_LONG" and money_flow_phase.upper() in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING']:
        # Money flow showing profit taking - caution on longs
        mf_upper = money_flow_phase.upper()
        if position_label == "LATE":
            conclusion = f"⚠️ PROFIT TAKING at HIGHS ({position_pct:.0f}%) - High risk!"
            conclusion_action = "WAIT - Distribution likely"
        else:
            conclusion = f"⚠️ PROFIT TAKING detected - W:{whale_pct:.0f}% but smart money selling"
            conclusion_action = "CAUTION - Wait for confirmation"
    elif trade_direction == "LONG":
        if confidence == "HIGH":
            if squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
                conclusion = f"🟢 STRONG LONG (Squeeze) - W:{whale_pct:.0f}% vs R:{retail_pct:.0f}%"
            else:
                conclusion = f"🟢 STRONG LONG - Whales {whale_pct:.0f}% bullish"
            conclusion_action = "LONG NOW" if position_label == "EARLY" else "LONG (watch entry)"
        else:
            conclusion = f"🟢 LONG SETUP - Whales {whale_pct:.0f}% lean bullish"
            conclusion_action = "LONG on dip" if position_label != "EARLY" else "LONG"
    elif trade_direction == "SHORT":
        if confidence == "HIGH":
            conclusion = f"🔴 STRONG SHORT - Whales {whale_pct:.0f}% bearish"
            conclusion_action = "SHORT NOW" if position_label == "LATE" else "SHORT (watch entry)"
        else:
            conclusion = f"🔴 SHORT SETUP - Whales {whale_pct:.0f}% lean bearish"
            conclusion_action = "SHORT on rally" if position_label != "LATE" else "SHORT"
    else:
        conclusion = f"⏳ NO CLEAR EDGE - W:{whale_pct:.0f}% R:{retail_pct:.0f}%"
        conclusion_action = "WAIT for setup"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: Return Complete Decision
    # ═══════════════════════════════════════════════════════════════════════
    
    return TradeDecision(
        action=action,
        trade_direction=trade_direction,
        confidence=confidence,
        
        direction_score=direction_score,
        squeeze_score=squeeze_score,
        entry_score=entry_score,
        total_score=total_score,
        
        direction_label=direction,
        squeeze_label=squeeze_type,
        position_label=position_label,
        oi_signal=oi_signal,
        
        main_reason=main_reason,
        warnings=warnings,
        story=story,
        
        is_valid_long=is_valid_long,
        is_valid_short=is_valid_short,
        invalidation_reason=invalidation_reason,
        
        # Stories for Combined Learning
        oi_story=oi_story,
        whale_story=whale_story,
        position_story=position_story,
        market_story=market_story,
        historical_story=historical_story,  # NEW: Historical validation
        conclusion=conclusion,
        conclusion_action=conclusion_action,
        
        # Market Conditions
        fear_greed=fear_greed,
        market_modifier=market_modifier,
        
        # Historical Validation
        historical_win_rate=historical_win_rate,
        historical_sample_size=historical_sample_size,
        historical_modifier=historical_modifier
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_long(whale_pct: float, retail_pct: float, oi_change: float = 0, price_change: float = 0) -> Tuple[bool, str]:
    """
    Quick check if a LONG trade is valid.
    
    Returns: (is_valid, reason)
    """
    
    divergence = whale_pct - retail_pct
    
    # Rule 1: Retail > Whale = NO LONG
    if divergence < -5:
        return False, f"Retail ({retail_pct:.0f}%) > Whale ({whale_pct:.0f}%) - trap risk"
    
    # Rule 2: OI falling + Price falling = NO LONG
    if oi_change < -2 and price_change < -1:
        return False, f"Longs exiting (OI {oi_change:+.1f}%, Price {price_change:+.1f}%)"
    
    # Rule 3: Whales not bullish = WEAK LONG
    if whale_pct < 55:
        return False, f"Whales not bullish ({whale_pct:.0f}%)"
    
    return True, "Valid long setup"


def is_valid_short(whale_pct: float, retail_pct: float, position_pct: float = 50) -> Tuple[bool, str]:
    """
    Quick check if a SHORT trade is valid.
    
    Returns: (is_valid, reason)
    """
    
    # Rule 1: Whales not bearish = NO SHORT
    if whale_pct > 45:
        return False, f"Whales not bearish ({whale_pct:.0f}%)"
    
    # Rule 2: At lows = risky short (potential capitulation)
    if position_pct < 30:
        return False, f"Near lows ({position_pct:.0f}%) - potential capitulation"
    
    return True, "Valid short setup"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DECISION FROM MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

def decide_from_context(ctx) -> TradeDecision:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    THE UNIFIED DECISION FUNCTION
    ═══════════════════════════════════════════════════════════════════════════════
    
    Takes a MarketContext object and returns the complete trading decision.
    This is the ONLY function that should be called for scoring.
    
    Args:
        ctx: MarketContext object with all gathered metrics
        
    Returns:
        TradeDecision with complete analysis, scores, and stories
    """
    
    return get_trade_decision(
        whale_pct=ctx.whale_pct,
        retail_pct=ctx.retail_pct,
        oi_change=ctx.oi_change_24h,
        price_change=ctx.price_change_24h,
        position_pct=ctx.position_in_range,
        ta_score=ctx.ta_score,
        money_flow_phase=ctx.money_flow_phase
    )


def get_unified_result(ctx, decision: TradeDecision) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    GET UNIFIED RESULT FOR DISPLAY
    ═══════════════════════════════════════════════════════════════════════════════
    
    Takes MarketContext and TradeDecision, returns everything needed for display.
    This replaces calculate_predictive_score output.
    
    Returns dict with:
        - All PredictiveScore-compatible fields
        - Combined learning data
        - Display-ready formatted strings
    """
    
    # Map direction label to display format
    direction_display = {
        'BULLISH': 'BULLISH',
        'LEAN_BULLISH': 'LEAN_BULLISH',
        'BEARISH': 'BEARISH',
        'LEAN_BEARISH': 'LEAN_BEARISH',
        'NEUTRAL': 'NEUTRAL',
    }.get(decision.direction_label, 'NEUTRAL')
    
    # Map squeeze label to display format
    squeeze_display = {
        'HIGH_SQUEEZE': 'HIGH',
        'MODERATE_SQUEEZE': 'MEDIUM',
        'SLIGHT_EDGE': 'LOW',
        'NO_EDGE': 'NONE',
        'EXTREME_TRAP': 'CONFLICT',
        'HIGH_TRAP': 'CONFLICT',
        'MODERATE_TRAP': 'CONFLICT',
        'SLIGHT_TRAP': 'CONFLICT',
    }.get(decision.squeeze_label, 'NONE')
    
    # Map entry timing
    if decision.trade_direction == 'WAIT':
        entry_timing = 'WAIT'
    elif decision.position_label == 'EARLY' and decision.trade_direction == 'LONG':
        entry_timing = 'NOW'
    elif decision.position_label == 'LATE' and decision.trade_direction == 'SHORT':
        entry_timing = 'NOW'
    elif decision.position_label == 'MIDDLE':
        entry_timing = 'SOON'
    else:
        entry_timing = 'WAIT'
    
    # Determine confidence for direction
    if decision.direction_score >= 36:
        direction_confidence = 'HIGH'
    elif decision.direction_score >= 28:
        direction_confidence = 'MEDIUM'
    else:
        direction_confidence = 'LOW' if decision.direction_score >= 15 else 'NONE'
    
    # Build final action string
    if decision.trade_direction == 'LONG':
        if decision.total_score >= 80:
            final_action = '🚀 STRONG BUY'
        elif decision.total_score >= 65:
            final_action = '✅ BUY'
        elif decision.total_score >= 50:
            final_action = '📊 CAUTIOUS BUY'
        else:
            final_action = '⏳ WAIT FOR DIP'
    elif decision.trade_direction == 'SHORT':
        if decision.total_score >= 80:
            final_action = '🔴 STRONG SELL'
        elif decision.total_score >= 65:
            final_action = '🔴 SELL'
        elif decision.total_score >= 50:
            final_action = '📊 CAUTIOUS SELL'
        else:
            final_action = '⏳ WAIT FOR RALLY'
    else:
        if 'TRAP' in decision.action:
            final_action = f'⚠️ {decision.action}'
        elif 'WAIT_FOR' in decision.action:
            final_action = f'⏳ {decision.action.replace("_", " ")}'
        else:
            final_action = '⏳ WAIT'
    
    # Build summary
    if decision.trade_direction == 'WAIT':
        final_summary = decision.conclusion
    else:
        final_summary = f"{decision.conclusion}"
    
    # Reason strings
    direction_reason = f"Whales {ctx.whale_pct:.0f}% | OI: {ctx.oi_change_24h:+.1f}%"
    squeeze_reason = f"Divergence: {ctx.divergence:+.0f}%"
    timing_reason = f"TA: {ctx.ta_score} | Position: {decision.position_label} ({ctx.position_in_range:.0f}%)"
    
    return {
        # PredictiveScore-compatible fields
        'direction': direction_display,
        'direction_confidence': direction_confidence,
        'direction_score': decision.direction_score,
        'direction_reason': direction_reason,
        
        'squeeze_potential': squeeze_display,
        'divergence_pct': ctx.divergence,
        'squeeze_score': decision.squeeze_score,
        'squeeze_reason': squeeze_reason,
        
        'entry_timing': entry_timing,
        'ta_score': ctx.ta_score,
        'timing_score': decision.entry_score,
        'timing_reason': timing_reason,
        'move_position': decision.position_label,
        'move_position_pct': ctx.position_in_range,
        
        'final_score': decision.total_score,
        'final_action': final_action,
        'final_summary': final_summary,
        
        'trade_mode': ctx.trading_mode,
        'timeframe': ctx.timeframe,
        'trade_direction': decision.trade_direction,
        
        # Additional fields
        'action': decision.action,
        'confidence': decision.confidence,
        'is_valid_long': decision.is_valid_long,
        'is_valid_short': decision.is_valid_short,
        'warnings': decision.warnings,
        
        # Combined Learning stories
        'oi_story': decision.oi_story,
        'whale_story': decision.whale_story,
        'position_story': decision.position_story,
        'market_story': decision.market_story,
        'historical_story': decision.historical_story,  # NEW: Historical validation
        'conclusion': decision.conclusion,
        'conclusion_action': decision.conclusion_action,
        
        # Market Conditions data
        'fear_greed': decision.fear_greed,
        'market_modifier': decision.market_modifier,
        
        # Historical Validation data
        'historical_win_rate': decision.historical_win_rate,
        'historical_sample_size': decision.historical_sample_size,
        'historical_modifier': decision.historical_modifier,
        
        # Raw decision for reference
        'master_decision': decision,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Run all test scenarios"""
    
    print("=" * 80)
    print("MASTER RULES TEST SUITE")
    print("=" * 80)
    
    test_cases = [
        # (whale, retail, oi, price, position, expected_direction, expected_valid_long, description)
        (62, 65, -3.0, -0.4, 47, "WAIT", False, "ACHUSDT - Retail > Whale trap"),
        (63, 70, -1.2, -3.0, 8, "WAIT", False, "PORTAL - Retail trap at lows"),
        (75, 55, 5.0, 2.0, 20, "LONG", True, "Strong squeeze setup"),
        (73, 65, 1.2, 1.1, 70, "SHORT", True, "ETH - Both bullish at highs = EUPHORIA"),
        (30, 30, -3.0, -5.0, 10, "LONG", True, "Capitulation at lows"),
        (70, 70, 2.0, 3.0, 85, "SHORT", True, "Euphoria at highs"),
        (50, 50, 0, 0, 50, "WAIT", True, "Neutral - no edge"),
        (68, 50, 3.0, 2.0, 35, "LONG", True, "Good setup - whale conviction"),
        (58, 62, 2.0, 1.0, 30, "WAIT", False, "Slight retail > whale = trap"),
        (72, 58, 3.0, 2.0, 25, "LONG", True, "Clear whale edge"),
    ]
    
    for whale, retail, oi, price, pos, exp_dir, exp_valid, desc in test_cases:
        result = get_trade_decision(
            whale_pct=whale,
            retail_pct=retail,
            oi_change=oi,
            price_change=price,
            position_pct=pos
        )
        
        # Check if result matches expectation
        dir_match = "✅" if result.trade_direction == exp_dir else "❌"
        valid_match = "✅" if result.is_valid_long == exp_valid else "❌"
        
        print(f"\n{desc}")
        print(f"  Input: W:{whale}% R:{retail}% OI:{oi:+.1f}% P:{price:+.1f}% Pos:{pos}%")
        print(f"  Output: {result.action} | {result.trade_direction} | Score:{result.total_score}")
        print(f"  Valid Long: {result.is_valid_long}")
        print(f"  Direction: {dir_match} (expected {exp_dir}, got {result.trade_direction})")
        print(f"  Valid: {valid_match} (expected {exp_valid}, got {result.is_valid_long})")
        print(f"  Reason: {result.main_reason}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")


if __name__ == "__main__":
    run_tests()