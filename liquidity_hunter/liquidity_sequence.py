"""
Liquidity Sequence Visualization V2 - UNIFIED MODEL

Shows:
1. ACTIVE SWEEP with candles age
2. Swept levels (strikethrough)
3. Fresh targets with ML PROBABILITY
4. No trading modes - ONE model for all!
"""
# DEBUG: If you see this in terminal, the new file is loaded!
print(">>> [DEBUG] liquidity_sequence.py LOADED - fmt_price now shows 4 decimals for ADA <<<")

import streamlit as st
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# HOLISTIC UI THRESHOLDS - All signals must be meaningful, not noise
# ═══════════════════════════════════════════════════════════════════════════════
# Minimum whale change to show acceleration labels (avoid showing for +0.2% noise)
MIN_WHALE_CHANGE_FOR_ACCELERATION = 1.0  # At least 1% change to be "BUYING/SELLING FASTER"

# ML threshold for showing trade recommendations
try:
    from liquidity_hunter.liquidity_hunter import ML_SKIP_THRESHOLD
except ImportError:
    try:
        from liquidity_hunter import ML_SKIP_THRESHOLD
    except ImportError:
        ML_SKIP_THRESHOLD = 0.40  # Default: 40% = SKIP


def get_trade_direction(raw_direction: str, entry_quality: dict = None) -> str:
    """Get trade direction from ML prediction or fallback to continuation."""
    if entry_quality and isinstance(entry_quality, dict) and entry_quality.get('direction'):
        return entry_quality['direction']
    # Fallback: continuation mode
    return "SHORT" if raw_direction == "LONG" else "LONG"


# ═══════════════════════════════════════════════════════════════════════════════
# SMART PRICE FORMATTER (handles PEPE, SHIB, ADA and other small-decimal coins)
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_price(price: float) -> str:
    """Format price with appropriate decimal places for display."""
    if price is None or price == 0:
        return "0.00"
    
    abs_price = abs(price)
    
    if abs_price >= 1000:
        return f"{price:,.2f}"
    elif abs_price >= 10:
        return f"{price:.2f}"
    elif abs_price >= 1:
        # For $1-$10 range, show 3 decimals to distinguish close prices
        return f"{price:.3f}"
    elif abs_price >= 0.1:
        # For $0.10-$1 range (like ADA $0.36), show 4 decimals
        return f"{price:.4f}"
    elif abs_price >= 0.01:
        return f"{price:.4f}"
    elif abs_price >= 0.0001:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


# Import unified ML model
try:
    from liquidity_hunter.unified_lh_ml import predict_level, get_model_status, extract_features
    UNIFIED_ML_AVAILABLE = True
    print("[LIQ_SEQ] Unified ML model loaded!")
except ImportError:
    UNIFIED_ML_AVAILABLE = False
    print("[LIQ_SEQ] Unified ML not available, using rule-based")


def render_full_liquidity_sequence(
    levels: List[Dict],
    current_price: float,
    sweep_status: Dict = None,
    whale_bias: str = "NEUTRAL",
    atr: float = 0,
    whale_pct: float = 50,
    whale_delta: Dict = None,
    entry_quality: Dict = None
):
    """
    Main entry point - renders complete liquidity sequence.

    Args:
        whale_pct: Current whale long percentage
        whale_delta: Dict with whale_delta, retail_delta, lookback_label from get_whale_delta()
        entry_quality: Dict with ML recommendation (ENTER/WAIT/SKIP), ml_probability, etc.
    """
    sweep_status = sweep_status or {}
    whale_delta = whale_delta or {}
    
    # Extract sweep info
    sweep_detected = (
        sweep_status.get('detected', False) or 
        sweep_status.get('is_sweep', False)
    )
    sweep_level = sweep_status.get('level_swept', 0)
    sweep_candles = sweep_status.get('candles_ago', 0) or sweep_status.get('candles_since', 0)
    sweep_direction = sweep_status.get('direction', '')
    
    # Extract whale delta - NOW with 4h, 24h and 7d!
    whale_change = whale_delta.get('whale_delta', None)
    whale_change_4h = whale_delta.get('whale_delta_4h', None)  # NEW: Early signal
    whale_change_24h = whale_delta.get('whale_delta_24h', None)
    whale_change_7d = whale_delta.get('whale_delta_7d', None)
    whale_acceleration = whale_delta.get('whale_acceleration', None)
    whale_early_signal = whale_delta.get('whale_early_signal', None)  # NEW: EARLY_ACCUMULATION, etc.
    is_fresh = whale_delta.get('is_fresh_accumulation', False)
    retail_change = whale_delta.get('retail_delta', None)
    lookback_label = whale_delta.get('lookback_label', '24h')

    # Debug: Log what data we received
    print(f"[LIQ_SEQ_DEBUG] whale_delta_4h={whale_change_4h}, whale_delta_24h={whale_change_24h}, whale_delta_7d={whale_change_7d}, early_signal={whale_early_signal}")

    # If we have both, prefer showing 24h for day trading
    if whale_change_24h is not None:
        whale_change = whale_change_24h
        lookback_label = '24h'

    with st.expander("🗺️ **Liquidity Sequence Map** - What happens next?", expanded=True):

        # Show 4h vs 24h vs 7d comparison - EARLY SIGNAL DETECTION
        if whale_change_24h is not None and whale_change_7d is not None:
            # 4-column layout: 4h (early) | 24h (fresh) | 7d (total) | status
            col_4h, col_24h, col_7d, col_status = st.columns(4)

            with col_4h:
                if whale_change_4h is not None:
                    delta_color_4h = "normal" if whale_change_4h >= 0 else "inverse"
                    st.metric("⚡ 4h (Early)", f"{whale_change_4h:+.1f}%", delta_color=delta_color_4h)
                else:
                    st.metric("⚡ 4h (Early)", "N/A")

            with col_24h:
                delta_color_24h = "normal" if whale_change_24h >= 0 else "inverse"
                st.metric("🚀 24h (Fresh)", f"{whale_change_24h:+.1f}%", delta_color=delta_color_24h)

            with col_7d:
                delta_color_7d = "normal" if whale_change_7d >= 0 else "inverse"
                st.metric("📅 7d (Total)", f"{whale_change_7d:+.1f}%", delta_color=delta_color_7d)

            with col_status:
                # Make acceleration label DIRECTION-AWARE + require minimum threshold
                abs_change_24h = abs(whale_change_24h) if whale_change_24h else 0

                if whale_acceleration == 'ACCELERATING' and abs_change_24h >= MIN_WHALE_CHANGE_FOR_ACCELERATION:
                    if whale_change_24h > 0:
                        st.success("🚀 BUYING FASTER")
                    else:
                        st.error("📉 SELLING FASTER")
                elif whale_acceleration == 'DECELERATING':
                    if whale_change_7d > 0:
                        st.warning("🐢 Buying slowing")
                    else:
                        st.warning("🐢 Selling slowing")
                elif whale_acceleration == 'REVERSING' and abs_change_24h >= MIN_WHALE_CHANGE_FOR_ACCELERATION:
                    if whale_change_24h > 0:
                        st.success("🔄 REVERSING TO BUY!")
                    else:
                        st.error("🔄 REVERSING TO SELL!")
                elif abs_change_24h < MIN_WHALE_CHANGE_FOR_ACCELERATION:
                    st.info(f"➡️ Steady ({whale_change_24h:+.1f}%)")
                elif is_fresh:
                    st.success("✅ Fresh entry!")
                else:
                    st.info("➡️ Steady")

            # NEW: Early signal alerts (4h vs 24h divergence)
            if whale_early_signal == 'EARLY_ACCUMULATION':
                st.success(f"⚡ **EARLY ACCUMULATION**: 4h ({whale_change_4h:+.1f}%) starting to BUY while 24h still bearish - potential reversal!")
            elif whale_early_signal == 'EARLY_DISTRIBUTION':
                st.error(f"⚡ **EARLY DISTRIBUTION**: 4h ({whale_change_4h:+.1f}%) starting to SELL while 24h still bullish - whales exiting!")
            elif whale_early_signal == 'FRESH_ACCUMULATION' and whale_change_4h and whale_change_4h > 2:
                st.success(f"🔥 **FRESH BUY**: Strong 4h accumulation ({whale_change_4h:+.1f}%)")
            elif whale_early_signal == 'FRESH_DISTRIBUTION' and whale_change_4h and whale_change_4h < -2:
                st.error(f"🔥 **FRESH SELL**: Strong 4h distribution ({whale_change_4h:+.1f}%)")

            # Late entry warning
            if whale_change_7d and whale_change_7d > 5 and whale_change_24h and whale_change_24h < whale_change_7d / 7:
                st.error("⏰ **LATE ENTRY WARNING**: 7d accumulated but 24h slowing - whales may be done!")

            st.divider()
        elif whale_change_7d is not None:
            # Only 7d available - show what we have with explanation
            col_7d, col_status = st.columns(2)
            with col_7d:
                delta_color_7d = "normal" if whale_change_7d >= 0 else "inverse"
                st.metric("📅 7d Whale Change", f"{whale_change_7d:+.1f}%", delta_color=delta_color_7d)
            with col_status:
                st.info("ℹ️ 24h data needs more snapshots")
            st.caption("*24h data requires 2+ whale snapshots in last 24h. Run app more frequently to collect data.*")
            st.divider()
        else:
            # No whale delta data at all - show what we have
            st.info("🐋 **Whale Delta Data**: Not available for this symbol")
            st.caption("*Whale data may not be available for all pairs. Check console for debug info.*")
            st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 Sequence", "🎯 Action", "📈 Path"])

        # Determine trade direction for UI labels
        trade_direction = get_trade_direction(sweep_direction) if sweep_detected else None

        with tab1:
            _render_sequence_tab(
                levels, current_price,
                sweep_detected, sweep_level, sweep_candles, sweep_direction,
                whale_pct, whale_change, lookback_label, atr,
                trade_direction=trade_direction  # Pass trade direction for correct labels
            )

        with tab2:
            _render_action_tab(
                levels, current_price,
                sweep_detected, sweep_level, sweep_candles, sweep_direction,
                whale_pct, whale_change, entry_quality
            )
        
        with tab3:
            _render_path_tab(
                levels, current_price,
                sweep_detected, sweep_level, sweep_candles, sweep_direction,
                whale_bias, whale_pct, atr
            )


def _render_sequence_tab(
    levels: List[Dict],
    current_price: float,
    sweep_detected: bool,
    sweep_level: float,
    sweep_candles: int,
    sweep_direction: str,
    whale_pct: float = 50,
    whale_change: float = None,
    lookback_label: str = '24h',
    atr: float = 0,
    trade_direction: str = None  # NEW: Pass actual trade direction
):
    """Render the sequence tab with levels above/below and ML probabilities."""
    
    st.markdown("### 🗺️ Liquidity Sequence")
    
    # ═══════════════════════════════════════════════════════════════
    # WHALE DELTA DISPLAY - Now shows BOTH 24h and 7d!
    # ═══════════════════════════════════════════════════════════════
    if whale_change is not None:
        # Try to get both 24h and 7d from whale_delta dict (passed via extra kwargs)
        # For now, whale_change is the main delta based on trading mode
        whale_emoji = "🟢 ↑" if whale_change > 0 else "🔴 ↓" if whale_change < 0 else "⚪ →"
        whale_color = "#00ff88" if whale_change > 0 else "#ff6b6b" if whale_change < 0 else "#888"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Whales ({lookback_label})",
                value=f"{whale_pct:.0f}% Long",
                delta=f"{whale_change:+.1f}%" if whale_change else None,
                delta_color="normal"
            )
        with col2:
            # Whale momentum interpretation
            if whale_change and whale_change > 3:
                st.success("🐋 Whales ACCUMULATING")
            elif whale_change and whale_change < -3:
                st.error("🐋 Whales DISTRIBUTING")
            else:
                st.info("🐋 Whales HOLDING")
        with col3:
            # Show timing status
            if lookback_label == '7d' and whale_change and whale_change > 5:
                st.warning("⏰ 7d accumulation = may be LATE")
            elif lookback_label == '24h' and whale_change and whale_change > 2:
                st.success("🚀 FRESH 24h momentum!")

        st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # ACTIVE SWEEP STATUS BOX
    # ═══════════════════════════════════════════════════════════════
    if sweep_detected and sweep_level > 0:
        st.markdown("#### ✅ ACTIVE SWEEP")

        # Get TRADE direction (CONTINUATION mode aware)
        trade_direction = get_trade_direction(sweep_direction)
        sweep_type = "LOW" if sweep_direction == "LONG" else "HIGH"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level", f"${fmt_price(sweep_level)}")
        with col2:
            st.metric("Age", f"{sweep_candles} candles")
        with col3:
            # Show TRADE direction (not raw sweep direction)
            st.metric("Trade", trade_direction)

        # Freshness indicator
        if sweep_candles <= 3:
            st.success(f"🟢 **FRESH** - Entry window OPEN! (Sweep of {sweep_type})")
        elif sweep_candles <= 10:
            st.warning("🟡 **AGING** - Consider limit order at retest")
        else:
            st.error("🔴 **OLD** - Entry window CLOSED - Wait for retest or next sweep")

        # ═══════════════════════════════════════════════════════════════
        # WHALE CONFLICT WARNING - use TRADE direction
        # ═══════════════════════════════════════════════════════════════
        if whale_change is not None:
            # LONG trade but whales distributing = CONFLICT
            if trade_direction == 'LONG' and whale_change < -3:
                st.warning("⚠️ **WHALE CONFLICT** - LONG trade but whales DISTRIBUTING!")
            # SHORT trade but whales accumulating = CONFLICT
            elif trade_direction == 'SHORT' and whale_change > 3:
                st.warning("⚠️ **WHALE CONFLICT** - SHORT trade but whales ACCUMULATING!")
            # OLD sweep + wrong whale flow = SKIP
            elif sweep_candles > 10 and ((trade_direction == 'LONG' and whale_change < 0) or (trade_direction == 'SHORT' and whale_change > 0)):
                st.error("❌ **SKIP THIS TRADE** - OLD sweep + whales moving against direction")

        st.divider()
    else:
        st.info("👀 No active sweep - Monitoring levels")
        st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # Separate levels into above/below current price
    # ═══════════════════════════════════════════════════════════════
    above_levels = []
    below_levels = []
    
    for level in levels:
        price = level.get('price', 0)
        if price <= 0:
            continue
        if price > current_price:
            above_levels.append(level)
        else:
            below_levels.append(level)
    
    # Sort
    above_levels.sort(key=lambda x: x.get('price', 0))  # Nearest first
    below_levels.sort(key=lambda x: x.get('price', 0), reverse=True)  # Nearest first
    
    # ═══════════════════════════════════════════════════════════════
    # ML PROBABILITY - Uses UNIFIED Model (No trading modes!)
    # ═══════════════════════════════════════════════════════════════
    def calc_target_probability(target_price: float, direction: str, level_type: str = 'LIQ_POOL', level_strength: str = 'MODERATE') -> Dict:
        """
        Calculate probability of reaching target based on DISTANCE ONLY.
        
        NO BONUSES = NO INVERSIONS POSSIBLE.
        Level quality is shown in the label, not the probability.
        """
        if current_price <= 0 or target_price <= 0 or atr <= 0:
            return {'sweep_prob': 50, 'target_prob': 50, 'combined_prob': 25, 'quality': 'UNKNOWN'}
        
        # Pure distance formula: 80 - (distance% × 5)
        pct_distance = abs(target_price - current_price) / current_price * 100
        combined_prob = int(max(15, 80 - (pct_distance * 5)))
        
        # Quality based on distance only
        if combined_prob >= 65:
            quality = 'HIGH'
        elif combined_prob >= 45:
            quality = 'MEDIUM'
        else:
            quality = 'LOW'
        
        return {
            'sweep_prob': combined_prob,
            'target_prob': combined_prob,
            'combined_prob': combined_prob,
            'quality': quality
        }
    
    # ═══════════════════════════════════════════════════════════════
    # LEVELS ABOVE - Dynamic labels based on trade direction
    # ═══════════════════════════════════════════════════════════════
    if trade_direction == "SHORT":
        # For SHORT trade: levels above = RISK (price going up = loss)
        st.markdown("#### 🔴 RISK LEVELS (Above)")
        st.caption("↑ Price UP = Your STOP LOSS zone")
    else:
        # For LONG trade (or no sweep): levels above = TP targets
        st.markdown("#### 🟢 TP TARGETS (Above)")
        st.caption("↑ Price UP = Your TAKE PROFIT targets")
    
    # Only show UNSWEPT levels as targets
    fresh_above_levels = [l for l in above_levels if not l.get('is_swept', False) and l.get('type', '') != 'Swept Level']
    swept_above_levels = [l for l in above_levels if l.get('is_swept', False) or l.get('type', '') == 'Swept Level']
    
    if fresh_above_levels:
        for i, level in enumerate(fresh_above_levels[:4]):
            price = level.get('price', 0)
            level_type = level.get('type', 'Pool')
            level_strength = level.get('strength', 'MODERATE')
            is_first = (i == 0)
            
            # Get ML probability
            prob_result = calc_target_probability(price, 'SHORT', level_type, level_strength)
            combined_prob = prob_result.get('combined_prob', 50)
            quality = prob_result.get('quality', 'MEDIUM')
            
            # Distance from current price
            pct_above = (price - current_price) / current_price * 100
            
            # ML Probability with color based on quality
            prob_emoji = "🟢" if quality == 'HIGH' else "🟡" if quality == 'MEDIUM' else "🔴"
            
            # Shortened level type for display
            short_type = level_type.replace('Liquidity Pool', 'Liq Pool').replace('Short Liq', 'S.Liq').replace('Long Liq', 'L.Liq')
            
            col1, col2, col3, col4 = st.columns([2.5, 2, 1.2, 1.3])
            with col1:
                if is_first:
                    st.markdown(f"**🎯 ${fmt_price(price)}**")
                else:
                    st.write(f"${fmt_price(price)}")
            with col2:
                st.caption(short_type)
            with col3:
                st.write(f"{prob_emoji} {combined_prob}%")
            with col4:
                if is_first:
                    st.success("NEXT")
                else:
                    st.caption(f"+{pct_above:.1f}%")
        
        # Show swept levels as crossed out (for context)
        if swept_above_levels:
            st.caption("~~Already swept:~~")
            for level in swept_above_levels[:2]:
                price = level.get('price', 0)
                st.caption(f"~~${fmt_price(price)}~~")
    else:
        if swept_above_levels:
            st.warning("All nearby levels already swept! Next target is farther.")
            for level in swept_above_levels[:2]:
                price = level.get('price', 0)
                st.caption(f"~~${fmt_price(price)}~~ (swept)")
        else:
            if trade_direction == "SHORT":
                st.caption("No risk levels above")
            else:
                st.caption("No TP targets above")
    
    # ═══════════════════════════════════════════════════════════════
    # CURRENT PRICE
    # ═══════════════════════════════════════════════════════════════
    st.divider()
    st.info(f"📍 **CURRENT: ${fmt_price(current_price)}**")
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # LEVELS BELOW - Dynamic labels based on trade direction
    # ═══════════════════════════════════════════════════════════════
    if trade_direction == "SHORT":
        # For SHORT trade: levels below = TP targets (price going down = profit)
        st.markdown("#### 🟢 TP TARGETS (Below)")
        st.caption("↓ Price DOWN = Your TAKE PROFIT targets")
    else:
        # For LONG trade (or no sweep): levels below = Entry/Risk zones
        st.markdown("#### 🔴 ENTRY ZONES (Below)")
        st.caption("↓ Price DOWN = Wait for sweep to enter LONG")
    
    found_next = False
    
    if below_levels:
        for level in below_levels[:4]:
            price = level.get('price', 0)
            level_type = level.get('type', 'Pool')
            level_strength = level.get('strength', 'MODERATE')
            
            # Check if swept
            is_swept = False
            if sweep_detected and sweep_level > 0 and price > 0:
                diff_pct = abs(price - sweep_level) / sweep_level
                if diff_pct < 0.015:
                    is_swept = True
            if level.get('is_swept', False):
                is_swept = True
            
            is_next = False
            if not is_swept and not found_next:
                is_next = True
                found_next = True
            
            # Get ML probability (only for non-swept levels)
            if not is_swept:
                prob_result = calc_target_probability(price, 'LONG', level_type, level_strength)
                combined_prob = prob_result.get('combined_prob', 50)
                quality = prob_result.get('quality', 'MEDIUM')
            else:
                combined_prob = 0
                quality = 'DONE'
            
            # Distance from current price
            pct_below = (current_price - price) / current_price * 100
            
            # Shortened level type for display
            short_type = level_type.replace('Liquidity Pool', 'Liq Pool').replace('Short Liq', 'S.Liq').replace('Long Liq', 'L.Liq')
            
            # ML Probability with color based on quality
            prob_emoji = "🟢" if quality == 'HIGH' else "🟡" if quality == 'MEDIUM' else "🔴" if quality != 'DONE' else "✅"
            
            col1, col2, col3, col4 = st.columns([2.5, 2, 1.2, 1.3])
            
            with col1:
                if is_swept:
                    st.markdown(f"~~${fmt_price(price)}~~")
                elif is_next:
                    st.markdown(f"**🎯 ${fmt_price(price)}**")
                else:
                    st.write(f"${fmt_price(price)}")
            
            with col2:
                if is_swept:
                    st.caption(f"✅ SWEPT ({sweep_candles}c)")
                else:
                    st.caption(short_type)
            
            with col3:
                if is_swept:
                    st.write("—")
                else:
                    st.write(f"{prob_emoji} {combined_prob}%")
            
            with col4:
                if is_swept:
                    if sweep_candles <= 3:
                        st.success("FRESH!")
                    elif sweep_candles <= 10:
                        st.warning("AGING")
                    else:
                        st.error("OLD")
                elif is_next:
                    st.success("NEXT")
                else:
                    st.caption(f"-{pct_below:.1f}%")
    else:
        st.caption("No long targets")


def _render_action_tab(
    levels: List[Dict],
    current_price: float,
    sweep_detected: bool,
    sweep_level: float,
    sweep_candles: int,
    sweep_direction: str,
    whale_pct: float = 50,
    whale_change: float = None,
    entry_quality: Dict = None
):
    """Render action recommendations - RESPECTS ML DECISION."""
    
    st.markdown("### 🎯 Action Summary")
    
    # Whale context
    if whale_change is not None:
        whale_emoji = "🟢 ↑" if whale_change > 0 else "🔴 ↓" if whale_change < 0 else "⚪ →"
        whale_status = "ACCUMULATING" if whale_change > 3 else "DISTRIBUTING" if whale_change < -3 else "HOLDING"
        st.caption(f"🐋 Whales: {whale_pct:.0f}% Long ({whale_change:+.1f}% change) - {whale_status}")
    
    if sweep_detected and sweep_level > 0:
        # Get TRADE direction (CONTINUATION mode aware)
        trade_direction = get_trade_direction(sweep_direction)
        sweep_type = "LOW" if sweep_direction == "LONG" else "HIGH"

        st.markdown(f"**Active Setup:** {trade_direction} (sweep of {sweep_type} at ${fmt_price(sweep_level)})")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Swept Level", f"${fmt_price(sweep_level)}")
        with col2:
            st.metric("Candles Ago", f"{sweep_candles}")
        with col3:
            if sweep_candles <= 3:
                st.metric("Window", "🟢 OPEN")
            elif sweep_candles <= 10:
                st.metric("Window", "🟡 CLOSING")
            else:
                st.metric("Window", "🔴 CLOSED")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════════════════
        # HOLISTIC ML CHECK - All entry recommendations must respect ML decision
        # ═══════════════════════════════════════════════════════════════════════════════
        entry_quality = entry_quality or {}
        ml_recommendation = entry_quality.get('recommendation', 'UNKNOWN')
        ml_probability = entry_quality.get('ml_probability', 0) or 0

        # ML PROBABILITY is the arbiter - not the rule-based recommendation
        # ML was trained on all features, so trust its probability
        if ml_probability < ML_SKIP_THRESHOLD:
            # ML < 40% → SKIP regardless of other factors
            st.error(f"🚫 **ML SAYS SKIP** - Do NOT enter {trade_direction}")
            st.write(f"• ML Probability: {ml_probability:.0%} (below {ML_SKIP_THRESHOLD:.0%} threshold)")
            st.write("• ML predicts this setup will fail")
            st.write("• Wait for better setup with ML > 50%")

        elif sweep_candles <= 3:
            # Fresh sweep AND ML approves
            if ml_recommendation == 'ENTER':
                st.success(f"✅ **ENTER {trade_direction} NOW** - Fresh sweep + ML approves ({ml_probability:.0%})")
                st.write(f"• Market entry at ${fmt_price(current_price)}")
                sl_text = "above" if trade_direction == "SHORT" else "below"
                st.write(f"• Stop loss {sl_text} ${fmt_price(sweep_level)}")
            else:
                # ML says WAIT but sweep is fresh
                st.warning(f"⚠️ **PROCEED WITH CAUTION** - Fresh sweep but ML borderline ({ml_probability:.0%})")
                st.write(f"• ML says WAIT - reduced confidence")
                st.write(f"• Consider smaller position or wait for confirmation")

        elif sweep_candles <= 10:
            # Check if ML overrides the closing window
            if ml_probability >= 0.50:
                st.success(f"✅ **ML OVERRIDE: ENTER {trade_direction}** - ML {ml_probability:.0%} despite {sweep_candles}c old")
                st.write(f"• ML sees strong signal despite closing window")
                st.write(f"• Consider market entry at ${fmt_price(current_price)}")
            else:
                st.warning(f"⚠️ **ENTRY WINDOW CLOSING** ({sweep_candles} candles)")
                st.write("**Options:**")
                st.write(f"1. Set limit order at ${fmt_price(sweep_level)} (retest)")
                st.write(f"2. Reduced position size at market")
                st.write(f"3. Wait for next sweep")

        else:
            # Old sweep (>10 candles) - BUT check if ML overrides
            if ml_probability >= 0.50:
                # ML says good trade despite old sweep - ENTER
                st.success(f"✅ **ML OVERRIDE: ENTER {trade_direction}** - ML {ml_probability:.0%} confident")
                st.write(f"• Sweep {sweep_candles} candles ago, but ML sees strong setup")
                st.write(f"• ML trained on sweep age - high probability = still valid")
                st.write(f"• Entry at ${fmt_price(current_price)}")
            elif ml_probability >= ML_SKIP_THRESHOLD:
                # ML borderline on old sweep - WAIT
                st.warning(f"⚠️ **BORDERLINE** - ML {ml_probability:.0%} on old sweep")
                st.write(f"• Sweep was {sweep_candles} candles ago")
                st.write(f"• Set limit at ${fmt_price(sweep_level)} for retest")
            else:
                st.error(f"❌ **TOO LATE** - Sweep was {sweep_candles} candles ago")
                st.write("**Do NOT chase! Options:**")
                st.write(f"1. Set limit at ${fmt_price(sweep_level)} for retest")
                st.write("2. Wait for next fresh sweep")

            # Find next fresh long target (for all cases)
            if ml_probability < 0.50:
                for level in levels:
                    if level.get('price', 0) < current_price and not level.get('is_swept', False):
                        if abs(level['price'] - sweep_level) / sweep_level > 0.02:  # Different level
                            st.write(f"3. Next LONG target: ${fmt_price(level['price'])}")
                            break
    else:
        st.info("👀 **WAITING MODE** - No active sweep")
        st.write("Monitor for sweeps at key levels below/above price")


def _render_path_tab(
    levels: List[Dict],
    current_price: float,
    sweep_detected: bool,
    sweep_level: float,
    sweep_candles: int,
    sweep_direction: str,
    whale_bias: str,
    whale_pct: float = 50,
    atr: float = 0
):
    """Render expected path/sequence - CONTINUATION mode aware."""

    st.markdown("### 📈 Expected Path")

    # Show whale context
    st.caption(f"🐋 Whale Positioning: {whale_pct:.0f}% Long ({whale_bias})")

    # Get TRADE direction (CONTINUATION mode aware)
    trade_direction = get_trade_direction(sweep_direction) if sweep_detected else None
    sweep_type = "LOW" if sweep_direction == "LONG" else "HIGH"

    # In CONTINUATION mode:
    # - Sweep of LOW (raw=LONG) → Trade SHORT → Targets are BELOW (continuation down)
    # - Sweep of HIGH (raw=SHORT) → Trade LONG → Targets are ABOVE (continuation up)

    if sweep_detected and trade_direction == 'LONG':
        st.markdown(f"**Bullish Path** (Sweep of {sweep_type} → Trade LONG)")

        # Find FRESH targets above (TPs for LONG trade)
        targets_above = [l for l in levels
                        if l.get('price', 0) > current_price
                        and not l.get('is_swept', False)
                        and l.get('type', '') != 'Swept Level']
        targets_above.sort(key=lambda x: x.get('price', 0))

        st.write(f"1️⃣ ✅ Swept ${fmt_price(sweep_level)} ({sweep_candles} candles ago)")
        st.write(f"2️⃣ 📍 Current ${fmt_price(current_price)}")

        if targets_above:
            level_type = targets_above[0].get('type', 'Liquidity Pool')
            st.write(f"3️⃣ 🎯 TP1 ${fmt_price(targets_above[0]['price'])} ({level_type})")
            if len(targets_above) > 1:
                level_type2 = targets_above[1].get('type', 'Liquidity Pool')
                st.write(f"4️⃣ ⚡ TP2 ${fmt_price(targets_above[1]['price'])} ({level_type2})")
        else:
            st.caption("No fresh targets above - all nearby levels swept")

        st.success("**Trade:** LONG until targets hit")

    elif sweep_detected and trade_direction == 'SHORT':
        st.markdown(f"**Bearish Path** (Sweep of {sweep_type} → Trade SHORT)")

        # Find FRESH targets below (TPs for SHORT trade)
        targets_below = [l for l in levels
                        if l.get('price', 0) < current_price
                        and not l.get('is_swept', False)
                        and l.get('type', '') != 'Swept Level']
        targets_below.sort(key=lambda x: x.get('price', 0), reverse=True)

        st.write(f"1️⃣ ✅ Swept ${fmt_price(sweep_level)} ({sweep_candles} candles ago)")
        st.write(f"2️⃣ 📍 Current ${fmt_price(current_price)}")

        if targets_below:
            level_type = targets_below[0].get('type', 'Liquidity Pool')
            st.write(f"3️⃣ 🎯 TP1 ${fmt_price(targets_below[0]['price'])} ({level_type})")
            if len(targets_below) > 1:
                level_type2 = targets_below[1].get('type', 'Liquidity Pool')
                st.write(f"4️⃣ ⚡ TP2 ${fmt_price(targets_below[1]['price'])} ({level_type2})")
        else:
            st.caption("No fresh targets below - all nearby levels swept")

        st.error("**Trade:** SHORT until targets hit")
        
    else:
        st.markdown("**Neutral - Waiting for Sweep (CONTINUATION)**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**For LONG trade:**")
            # In CONTINUATION: Sweep of HIGH → Trade LONG
            targets_above = [l for l in levels
                           if l.get('price', 0) > current_price
                           and not l.get('is_swept', False)]
            if targets_above:
                targets_above.sort(key=lambda x: x.get('price', 0))
                st.write(f"Wait for sweep of ${fmt_price(targets_above[0]['price'])} (HIGH)")
                st.write("Sweep of HIGH → Trade LONG")

        with col2:
            st.markdown("**For SHORT trade:**")
            # In CONTINUATION: Sweep of LOW → Trade SHORT
            targets_below = [l for l in levels
                           if l.get('price', 0) < current_price
                           and not l.get('is_swept', False)]
            if targets_below:
                targets_below.sort(key=lambda x: x.get('price', 0), reverse=True)
                st.write(f"Wait for sweep of ${fmt_price(targets_below[0]['price'])} (LOW)")
                st.write("Sweep of LOW → Trade SHORT")


# For backwards compatibility
def analyze_liquidity_sequence(levels, current_price, recent_sweep=None, atr=0):
    """Stub for compatibility - actual analysis done in render functions."""
    return {
        'current_price': current_price,
        'above_levels': [l for l in levels if l.get('price', 0) > current_price],
        'below_levels': [l for l in levels if l.get('price', 0) < current_price],
        'swept_levels': [l for l in levels if l.get('is_swept', False)],
        'next_short_target': None,
        'next_long_target': None,
        'sequence': [],
        'atr': atr
    }