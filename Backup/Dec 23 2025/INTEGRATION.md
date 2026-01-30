# InvestorIQ Integration Guide: Smart Trade Management

## New Modules Created

### 1. `core/trade_manager.py`
**Smart Level Finding using ta library**

Features:
- Uses `ta` library (BollingerBands, EMA, ATR, VWAP, Donchian) for structural levels
- Finds Order Blocks, FVG, Swing Highs/Lows
- Calculates SL based on structure priority: OB > FVG > Swing > S/R > EMA > VWAP > ATR fallback
- Calculates TPs based on opposing structural levels
- NO arbitrary breakeven - only structural trailing (Option C)
- MAE/MFE tracking for future vectorbt optimization
- `EnhancedTrade` dataclass with full tracking

### 2. `core/trading_card.py`
**Actionable Trade Display Components**

Features:
- `render_trading_card()` - Full trading card at top of analysis
- `render_mini_trading_card()` - Compact card for scanner results
- `render_monitor_card_enhanced()` - Live trade monitoring with MAE/MFE
- `render_stats_summary()` - System performance summary
- SL type badges showing quality (ORDER BLOCK > FVG > SWING > ATR FALLBACK)
- Dollar amounts for position size

---

## Integration Steps

### Step 1: Add Imports to app.py

Add these imports after the existing imports (around line 70):

```python
# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Smart Trade Management (ta library based)
# ═══════════════════════════════════════════════════════════════════════════════

from core.trade_manager import (
    calculate_smart_levels as calc_smart_levels_ta,
    SmartLevelFinder, SmartTradeCalculator, SmartTradeLevels,
    EnhancedTrade, create_enhanced_trade, update_trade_mae_mfe,
    find_structural_sl_update
)

from core.trading_card import (
    render_trading_card, render_mini_trading_card,
    render_monitor_card_enhanced, render_stats_summary,
    get_grade_color, get_sl_type_badge, format_price as card_format_price
)
```

### Step 2: Add Trading Card to Single Analysis

Find the Single Analysis section (around line 3146) and add this code after the grade is determined:

```python
# After determining grade and direction, calculate smart levels
smart_levels = calc_smart_levels_ta(df, direction, max_risk_pct=5.0)

# Render Trading Card at TOP (before the narrative)
render_trading_card(
    symbol=symbol,
    grade=grade,
    confidence=master_score,
    direction=direction,
    levels=smart_levels,
    position_size=2500,
    auto_added=is_auto_added  # True if A/A+ and added to monitor
)
```

### Step 3: Enhance Trade Monitor with MAE/MFE

In the Trade Monitor section (around line 1944), you can add enhanced tracking:

```python
# When updating trade prices, also update MAE/MFE
from core.trade_manager import update_trade_mae_mfe

# In the trade monitoring loop:
enhanced_trade = update_trade_mae_mfe(trade, current_price)
# Now trade has highest_profit_pct (MFE) and worst_drawdown_pct (MAE)
```

### Step 4: Add Structural SL Trailing (Option C)

After TP1 is hit, check for structural level to trail SL to:

```python
from core.trade_manager import find_structural_sl_update

# After TP1 hit:
if tp1_hit and not sl_already_moved:
    new_sl_info = find_structural_sl_update(df, current_sl, entry_price, direction)
    if new_sl_info:
        # Show user the new structural level available
        st.info(f"🔄 New structural SL at {new_sl_info['new_sl']:.4f} ({new_sl_info['reason']})")
        # User can click button to move SL
```

---

## Key Concepts

### SL Type Priority (Best to Worst)
1. **ORDER BLOCK** (🏦) - Institutional entry zone, best protection
2. **FVG** (📊) - Fair Value Gap, structural imbalance
3. **SWING** (📈) - Swing high/low, classic structure
4. **SUPPORT/RESISTANCE** (🛡️) - Donchian channel levels
5. **VWAP** (📊) - Volume weighted average price
6. **EMA** (📉) - Moving averages
7. **ATR FALLBACK** (⚠️) - Last resort, most likely to get hunted

### MAE/MFE Explained
- **MAE (Maximum Adverse Excursion)** - Worst drawdown during trade
- **MFE (Maximum Favorable Excursion)** - Best profit during trade

These metrics help optimize:
- SL placement (if MAE often exceeds SL, maybe SL too tight)
- TP placement (if MFE often exceeds TP3, maybe targets too conservative)

### Grade System
- **A+** (85+) - Exceptional, auto-monitor
- **A** (70-84) - Strong, auto-monitor  
- **B** (55-69) - Good
- **C** (40-54) - Moderate
- **D** (<40) - Weak

---

## Example Usage

```python
# In Single Analysis after getting df and determining direction:

# 1. Calculate smart levels
levels = calc_smart_levels_ta(df, direction="LONG", max_risk_pct=5.0)

# 2. Levels contains:
#    - entry, stop_loss, tp1, tp2, tp3
#    - sl_type, sl_reason (e.g., "below_ob", "Below Bullish Order Block")
#    - tp1_rr, tp2_rr, tp3_rr (risk:reward ratios)
#    - confidence (0-100 based on level quality)

# 3. Display trading card
render_trading_card(
    symbol="BTCUSDT",
    grade="A",
    confidence=75,
    direction="LONG",
    levels=levels,
    position_size=2500
)

# 4. The card shows:
#    - Entry price
#    - SL with type badge (🏦 ORDER BLOCK, 📊 FVG, etc.)
#    - TPs with R:R ratios
#    - Dollar amounts for $2,500 position
```

---

## Files Created

```
core/
├── trade_manager.py     # Smart level finding, MAE/MFE tracking
└── trading_card.py      # UI components for trade display
```

Both modules integrate with existing InvestorIQ infrastructure and don't require changes to existing files until you're ready to add the features.
