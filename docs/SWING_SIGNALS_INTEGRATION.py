"""
ML SIGNALS INTEGRATION GUIDE - ALL MODES
=========================================

This shows how to add ML signals to your UI for ALL trading modes:
- Scalp: continuation, fakeout (DIRECTIONAL), vol_expansion
- DayTrade: continuation, fakeout (DIRECTIONAL), vol_expansion  
- Swing: trend_holds, reversal (DIRECTIONAL), drawdown
- Investment: accumulation, distribution, reversal (DIRECTIONAL), large_drawdown

ALL REVERSAL/FAKEOUT SIGNALS ARE NOW DIRECTIONAL!
- reversal_to_bull = Was going DOWN, will reverse UP
- reversal_to_bear = Was going UP, will reverse DOWN
- fakeout_to_bull = Faking DOWN, will trap shorts and go UP
- fakeout_to_bear = Faking UP, will trap longs and go DOWN

STEP 1: The module is at /ml/ml_signals_display.py

STEP 2: Add import at top of app.py (around line 80-100 with other imports):
"""

# ADD THIS IMPORT (around line 80-100 in app.py):
# ════════════════════════════════════════════════════════════════════════════
# from ml.ml_signals_display import get_ml_signals, render_ml_signals_ui, get_ml_signals_summary
# ════════════════════════════════════════════════════════════════════════════


"""
STEP 3: Add the ML Signals display section.

Find this line in app.py (around line 7954):
    st.markdown("### 🐋 Whale & Institutional Analysis")

ADD THE FOLLOWING CODE **BEFORE** that line (around line 7950):
"""

# ════════════════════════════════════════════════════════════════════════════
# 🤖 ML SIGNALS - ADD THIS BLOCK BEFORE WHALE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
"""
                st.markdown("---")
                st.markdown("### 🤖 ML Signals")
                
                try:
                    from ml.ml_signals_display import get_ml_signals, render_ml_signals_ui
                    
                    # Determine mode from timeframe
                    mode_map = {
                        '1m': 'scalp', '5m': 'scalp',
                        '15m': 'daytrade', '1h': 'daytrade',
                        '4h': 'swing', '1d': 'swing',
                        '1w': 'investment'
                    }
                    ml_mode = mode_map.get(timeframe, 'daytrade')
                    
                    # Determine market type
                    market_type = 'crypto' if is_crypto else 'stock'
                    
                    # Get ML signals
                    ml_signals = get_ml_signals(df, mode=ml_mode, whale_data=whale_data, market_type=market_type)
                    
                    if ml_signals:
                        render_ml_signals_ui(ml_signals, show_details=True)
                    else:
                        st.info(f"ML model not trained for {ml_mode} mode. Train in ML Training tab.")
                        
                except Exception as e:
                    st.caption(f"ML Signals unavailable: {e}")
"""
# ════════════════════════════════════════════════════════════════════════════


"""
STEP 4: For Scanner view, add a compact summary.

Find the Scanner signal card area and add:
"""

# In Scanner, add this to show ML summary in each card:
"""
                        # ML Signal Summary (compact)
                        try:
                            from ml.ml_signals_display import get_ml_signals, get_ml_signals_summary
                            
                            mode_map = {'1m': 'scalp', '5m': 'scalp', '15m': 'daytrade', '1h': 'daytrade', '4h': 'swing', '1d': 'swing', '1w': 'investment'}
                            ml_mode = mode_map.get(timeframe, 'daytrade')
                            market_type = 'crypto' if is_crypto else 'stock'
                            
                            ml_sig = get_ml_signals(df, mode=ml_mode, whale_data=whale_data, market_type=market_type)
                            if ml_sig:
                                ml_summary = get_ml_signals_summary(ml_sig)
                                st.caption(ml_summary)
                        except:
                            pass
"""


"""
WHAT EACH SIGNAL MEANS - QUICK REFERENCE:
=========================================

SCALP / DAYTRADE SIGNALS:
┌──────────────────────┬─────────────────────────────────────────────────────────┐
│ Signal               │ What It Means                                           │
├──────────────────────┼─────────────────────────────────────────────────────────┤
│ continuation_bull    │ Price will continue UP (0.3-0.5%)                       │
│ continuation_bear    │ Price will continue DOWN (0.3-0.5%)                     │
│ fakeout_to_bull ⭐   │ BEAR TRAP - Faking down, will reverse UP                │
│ fakeout_to_bear ⭐   │ BULL TRAP - Faking up, will reverse DOWN                │
│ vol_expansion        │ Volatility about to increase                            │
└──────────────────────┴─────────────────────────────────────────────────────────┘

SWING SIGNALS:
┌──────────────────────┬─────────────────────────────────────────────────────────┐
│ Signal               │ What It Means                                           │
├──────────────────────┼─────────────────────────────────────────────────────────┤
│ trend_holds_bull     │ Bullish trend will continue (2%+ up)                    │
│ trend_holds_bear     │ Bearish trend will continue (2%+ down)                  │
│ reversal_to_bull ⭐  │ Was going DOWN, will REVERSE UP (97.8% F1!)             │
│ reversal_to_bear ⭐  │ Was going UP, will REVERSE DOWN (97.8% F1!)             │
│ drawdown             │ 3%+ adverse move coming (risk warning)                  │
└──────────────────────┴─────────────────────────────────────────────────────────┘

INVESTMENT SIGNALS:
┌──────────────────────┬─────────────────────────────────────────────────────────┐
│ Signal               │ What It Means                                           │
├──────────────────────┼─────────────────────────────────────────────────────────┤
│ accumulation         │ Smart money buying - expect 3%+ rise                    │
│ distribution         │ Smart money selling - expect 3%+ drop                   │
│ reversal_to_bull ⭐  │ Major bottom forming, will reverse UP                   │
│ reversal_to_bear ⭐  │ Major top forming, will reverse DOWN                    │
│ large_drawdown       │ 7%+ drawdown risk                                       │
└──────────────────────┴─────────────────────────────────────────────────────────┘

HOW TO USE:
===========

1. REVERSAL_TO_BULL > 60% → GO LONG (bottom forming, reversal UP)
2. REVERSAL_TO_BEAR > 60% → GO SHORT or EXIT LONG (top forming, reversal DOWN)
3. FAKEOUT_TO_BULL > 60% → GO LONG (bear trap, shorts getting rekt)
4. FAKEOUT_TO_BEAR > 60% → GO SHORT (bull trap, longs getting rekt)
5. DRAWDOWN > 60% → Reduce position size, tighten stop

PRIORITY ORDER:
===============
1. Check DIRECTIONAL REVERSAL first (tells you WHICH WAY)
2. Check FAKEOUT for trap warnings (tells you WHICH WAY trap reverses)
3. Check DRAWDOWN for risk management
4. Use trend_holds/continuation for confirmation

MARKET TYPE:
============
- market_type='crypto' uses crypto-trained models with whale/OI features
- market_type='stock' uses stock-trained models with institutional features
"""
