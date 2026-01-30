"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      LIQUIDITY HUNTER UI COMPONENTS                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

# ML-DRIVEN DIRECTION: No hardcoded strategy mode
# Direction comes from entry_quality['direction'] set by ML


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
    """
    Format price with appropriate decimal places for display.
    
    - Large prices (>=1000): 1,234.56
    - Medium prices (10-1000): 12.34
    - Small-medium (1-10): 1.234 (3 decimals)
    - Small prices (0.1-1): 0.3604 (4 decimals for ADA etc)
    - Smaller prices (0.01-0.1): 0.0123 (4 decimals)
    - Tiny prices (<0.01): 0.00001234 (8 decimals for PEPE, SHIB)
    """
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
        # Very small prices like PEPE, SHIB
        return f"{price:.8f}"


def render_liquidity_sequence(analysis: dict):
    """
    Render a clear liquidity sequence diagram showing:
    - SWEPT levels (already taken - greyed out)
    - FRESH levels (available targets)
    - Current price position
    - Next targets with clear arrows

    Uses CONTINUATION mode trade directions!
    """
    current_price = analysis.get('current_price', 0)
    sweep = analysis.get('sweep', {})
    approaching = analysis.get('approaching', [])
    liquidity_levels = analysis.get('liquidity_levels', {})

    # Determine if we have an active sweep
    sweep_active = sweep.get('is_sweep', False)
    sweep_level = sweep.get('level_swept', 0)
    raw_sweep_direction = sweep.get('direction', '')
    candles_since = sweep.get('candles_since', 0)

    # Get TRADE direction for display (CONTINUATION mode aware)
    # In CONTINUATION: Sweep of LOW → SHORT, Sweep of HIGH → LONG
    # Levels ABOVE current price: Sweep of HIGH → trade direction
    # Levels BELOW current price: Sweep of LOW → trade direction
    dir_for_above = get_trade_direction("SHORT")  # Sweep HIGH = SHORT raw, trade = LONG in CONTINUATION
    dir_for_below = get_trade_direction("LONG")   # Sweep LOW = LONG raw, trade = SHORT in CONTINUATION

    # Build list of all levels with status
    levels_above = []  # Levels above current price
    levels_below = []  # Levels below current price

    # Add highs (above current price)
    for h in liquidity_levels.get('highs', []):
        if h['price'] > current_price:
            levels_above.append({
                'price': h['price'],
                'type': 'LIQUIDITY_POOL',
                'strength': h.get('strength', 'MODERATE'),
                'swept': False,
                'is_target': True,
                'direction': dir_for_above  # CONTINUATION aware
            })

    # Add equal highs
    for eh in liquidity_levels.get('equal_highs', []):
        if eh['price'] > current_price:
            levels_above.append({
                'price': eh['price'],
                'type': 'EQUAL_HIGH',
                'strength': 'MAJOR',
                'swept': False,
                'is_target': True,
                'direction': dir_for_above  # CONTINUATION aware
            })

    # Add lows (below current price)
    for l in liquidity_levels.get('lows', []):
        if l['price'] < current_price:
            # Check if this level was swept
            was_swept = False
            if sweep_active and raw_sweep_direction == 'LONG':  # LONG raw = sweep of LOW
                if abs(l['price'] - sweep_level) / sweep_level < 0.005:  # Within 0.5%
                    was_swept = True
            
            levels_below.append({
                'price': l['price'],
                'type': 'LIQUIDITY_POOL',
                'strength': l.get('strength', 'MODERATE'),
                'swept': was_swept,
                'candles_since': candles_since if was_swept else 0,
                'is_target': not was_swept,
                'direction': dir_for_below  # CONTINUATION aware
            })

    # Add equal lows
    for el in liquidity_levels.get('equal_lows', []):
        if el['price'] < current_price:
            was_swept = False
            if sweep_active and raw_sweep_direction == 'LONG':  # LONG raw = sweep of LOW
                if abs(el['price'] - sweep_level) / sweep_level < 0.005:
                    was_swept = True

            levels_below.append({
                'price': el['price'],
                'type': 'EQUAL_LOW',
                'strength': 'MAJOR',
                'swept': was_swept,
                'candles_since': candles_since if was_swept else 0,
                'is_target': not was_swept,
                'direction': dir_for_below  # CONTINUATION aware
            })
    
    # Sort levels
    levels_above.sort(key=lambda x: x['price'])  # Ascending (closest first)
    levels_below.sort(key=lambda x: x['price'], reverse=True)  # Descending (closest first)
    
    # Find NEXT targets (first FRESH level in each direction)
    next_long_target = next((l for l in levels_below if not l['swept']), None)
    next_short_target = levels_above[0] if levels_above else None
    
    # Render the sequence
    st.markdown("#### 📍 Liquidity Sequence")
    
    # Current status
    if sweep_active:
        status_emoji = "✅" if candles_since <= 3 else "⏰"
        status_text = f"SWEEP ACTIVE at ${sweep_level:,.2f} ({candles_since} candles ago)"
        if candles_since <= 3:
            st.success(f"{status_emoji} **{status_text}** - Entry window OPEN!")
        elif candles_since <= 10:
            st.warning(f"{status_emoji} **{status_text}** - Entry window closing...")
        else:
            st.info(f"{status_emoji} **{status_text}** - Sweep is old, wait for retest or next sweep")
    
    # Create two columns for the visualization
    col_left, col_right = st.columns([2, 1])

    # Get labels based on CONTINUATION mode
    # In CONTINUATION: Above = LONG trades (sweep HIGH → LONG), Below = SHORT trades (sweep LOW → SHORT)
    above_label = f"**🟢 {dir_for_above} Territory** (Above Price)" if dir_for_above == "LONG" else f"**🔴 {dir_for_above} Territory** (Above Price)"
    below_label = f"**🔴 {dir_for_below} Territory** (Below Price)" if dir_for_below == "SHORT" else f"**🟢 {dir_for_below} Territory** (Below Price)"
    above_emoji = "🟢" if dir_for_above == "LONG" else "🔴"
    below_emoji = "🔴" if dir_for_below == "SHORT" else "🟢"

    with col_left:
        # Territory Above
        st.markdown(above_label)

        if levels_above:
            for i, level in enumerate(levels_above[:4]):
                strength_emoji = "⭐" if level['strength'] == 'MAJOR' else "🔹" if level['strength'] == 'STRONG' else "▫️"
                is_next = (next_short_target and level['price'] == next_short_target['price'])

                if is_next:
                    st.markdown(f"**→ ${level['price']:,.2f}** {strength_emoji} {level['type']} ({level['strength']}) ← **NEXT {dir_for_above} TARGET**")
                else:
                    st.markdown(f"   ${level['price']:,.2f} {strength_emoji} {level['type']} ({level['strength']})")
        else:
            st.caption(f"No {dir_for_above.lower()} liquidity pools detected above")

        st.markdown("---")

        # Current Price
        st.markdown(f"### 📍 **${current_price:,.2f}** ← YOU ARE HERE")

        st.markdown("---")

        # Territory Below
        st.markdown(below_label)

        if levels_below:
            for i, level in enumerate(levels_below[:4]):
                strength_emoji = "⭐" if level['strength'] == 'MAJOR' else "🔹" if level['strength'] == 'STRONG' else "▫️"
                is_next = (next_long_target and level['price'] == next_long_target['price'])

                if level['swept']:
                    st.markdown(f"   ~~${level['price']:,.2f}~~ ✅ **SWEPT** ({level.get('candles_since', 0)} candles ago) - Liquidity TAKEN")
                elif is_next:
                    st.markdown(f"**→ ${level['price']:,.2f}** {strength_emoji} {level['type']} ({level['strength']}) ← **NEXT {dir_for_below} TARGET**")
                else:
                    st.markdown(f"   ${level['price']:,.2f} {strength_emoji} {level['type']} ({level['strength']})")
        else:
            st.caption(f"No {dir_for_below.lower()} liquidity pools detected below")
    
    with col_right:
        # Sequence Flow Diagram - uses TRADE direction (CONTINUATION aware)
        st.markdown("**📊 Sequence Flow**")

        # Get trade direction from raw sweep direction
        trade_dir = get_trade_direction(raw_sweep_direction) if raw_sweep_direction else None
        # TP direction is opposite of trade direction
        tp_dir = "SHORT" if trade_dir == "LONG" else "LONG" if trade_dir == "SHORT" else None

        if sweep_active and trade_dir:
            st.markdown(f"""
            ```
            ┌─────────────────┐
            │ 1. SWEEP DONE ✓ │
            │  ({trade_dir} entry)  │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 2. NOW: Hold    │
            │   or Trail SL   │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 3. NEXT: {tp_dir}  │
            │   Liq = TP      │
            └─────────────────┘
            ```
            """)
        else:
            st.markdown("""
            ```
            ┌─────────────────┐
            │ WAITING...      │
            │ No active sweep │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ Watch for:      │
            │ • Price → Level │
            │ • Wick through  │
            │ • Close back    │
            └─────────────────┘
            ```
            """)
        
        # Next Actions
        st.markdown("**🎯 Next Actions:**")
        
        if sweep_active:
            if candles_since <= 3:
                st.success("✅ Entry window OPEN")
                st.caption("Market or limit entry valid")
            elif candles_since <= 10:
                st.warning("⏰ Consider limit at retest")
                st.caption(f"Retest level: ~${sweep_level:,.2f}")
            else:
                st.info("⏳ Sweep is old")
                st.caption("Wait for retest or next sweep")
            
            # Next TP target (based on TRADE direction)
            # In CONTINUATION: LONG trade TPs at levels above, SHORT trade TPs at levels below
            if trade_dir == 'LONG' and next_short_target:
                st.markdown(f"**TP Target:** ${next_short_target['price']:,.2f}")
            elif trade_dir == 'SHORT' and next_long_target:
                st.markdown(f"**TP Target:** ${next_long_target['price']:,.2f}")
        else:
            if next_long_target:
                st.markdown(f"**Next {dir_for_below} sweep:** ${next_long_target['price']:,.2f}")
            if next_short_target:
                st.markdown(f"**Next {dir_for_above} sweep:** ${next_short_target['price']:,.2f}")


def render_next_targets_improved(analysis: dict):
    """
    Render improved Next Targets section that clearly distinguishes:
    - SWEPT levels (crossed out, with timing)
    - FRESH levels (highlighted as targets)
    Uses CONTINUATION mode trade directions!
    """
    sweep = analysis.get('sweep', {})
    approaching = analysis.get('approaching', [])
    current_price = analysis.get('current_price', 0)

    sweep_active = sweep.get('is_sweep', False)
    sweep_level = sweep.get('level_swept', 0)
    candles_since = sweep.get('candles_since', 0)
    raw_direction = sweep.get('direction', '')
    # Get TRADE direction (CONTINUATION aware)
    trade_direction = get_trade_direction(raw_direction) if raw_direction else ''

    st.markdown("#### 🎯 Next Targets")

    # Show swept level first (if active)
    if sweep_active:
        freshness = "🟢 FRESH" if candles_since <= 3 else "🟡 AGING" if candles_since <= 10 else "⚪ OLD"

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown(f"~~${sweep_level:,.2f}~~ - {candles_since} candles ago")
        with col2:
            st.markdown(f"✅ **SWEPT**")
        with col3:
            st.markdown(freshness)
        with col4:
            # Show TRADE direction, not raw sweep direction
            dir_emoji = "🟢" if trade_direction == 'LONG' else "🔴"
            st.markdown(f"{dir_emoji} → {trade_direction}")

    # Show approaching levels (FRESH)
    if approaching:
        for app in approaching[:3]:
            level_price = app.get('level', 0)
            distance = app.get('distance_atr', 0)
            raw_app_direction = app.get('direction_after_sweep', 'LONG')
            # Get TRADE direction (CONTINUATION aware)
            app_trade_direction = get_trade_direction(raw_app_direction)

            # Skip if this is the swept level
            if sweep_active and abs(level_price - sweep_level) / sweep_level < 0.005:
                continue

            status = "🔥 IMMINENT" if distance < 0.5 else "⚡ APPROACHING" if distance < 1.0 else "👀 WATCHING"

            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.markdown(f"**${level_price:,.2f}**")
            with col2:
                st.markdown(status)
            with col3:
                st.markdown(f"{distance:.1f} ATR")
            with col4:
                dir_emoji = "🟢" if app_trade_direction == 'LONG' else "🔴"
                st.markdown(f"{dir_emoji} → {app_trade_direction}")
    else:
        if not sweep_active:
            st.caption("No imminent targets. Monitoring liquidity levels...")


def render_liquidity_header():
    """Render the Liquidity Hunter header"""
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #0a192f 100%); 
                    border-radius: 15px; padding: 20px; margin-bottom: 20px;
                    border: 2px solid #00d4aa;'>
            <div style='display: flex; align-items: center; gap: 15px;'>
                <div style='font-size: 2.5em;'>🎯</div>
                <div>
                    <div style='color: #00d4aa; font-size: 1.8em; font-weight: bold;'>
                        LIQUIDITY HUNTER
                    </div>
                    <div style='color: #888; font-size: 0.95em;'>
                        Follow the whale footprints • Enter at sweeps • Exit into liquidations
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_liquidity_map(analysis: dict):
    """Render the visual liquidity map"""
    current_price = analysis.get('current_price', 0)
    liquidity_levels = analysis.get('liquidity_levels', {})
    liq_zones = analysis.get('liquidation_zones', {})

    # Determine current trade direction from sweep status
    sweep_status = analysis.get('sweep_status', {})
    sweep_detected = sweep_status.get('detected', False) or sweep_status.get('is_sweep', False)
    sweep_direction = sweep_status.get('direction', '')
    trade_direction = get_trade_direction(sweep_direction) if sweep_detected and sweep_direction else None

    # Build levels for display
    all_levels = []

    # Add short liquidation zones (above price)
    # For SHORT trade: levels above = RISK (SL zone)
    # For LONG trade: levels above = TP TARGET
    short_liq_label = '◄── SL ZONE' if trade_direction == 'SHORT' else '◄── TP TARGET'
    short_liq_color = '#ff6b6b' if trade_direction == 'SHORT' else '#00ff88'
    for liq in liq_zones.get('short_liquidations', [])[:3]:
        all_levels.append({
            'price': liq['price'], 'type': 'SHORT_LIQ',
            'label': f"Short Liq ({liq['leverage']}x)", 'color': short_liq_color, 'arrow': short_liq_label
        })
    
    # Add liquidity highs
    # In CONTINUATION mode: Sweep HIGH → LONG (continue up)
    # In REVERSAL mode: Sweep HIGH → SHORT (reverse down)
    high_trade_dir = get_trade_direction("SHORT")  # Raw "SHORT" = sweep of HIGH
    high_color = '#00ff88' if high_trade_dir == 'LONG' else '#ff6b6b'
    for h in liquidity_levels.get('highs', [])[:3]:
        if h['price'] > current_price:
            all_levels.append({
                'price': h['price'], 'type': 'LIQUIDITY_HIGH',
                'label': f"Liquidity Pool ({h.get('strength', 'MODERATE')})",
                'color': high_color, 'arrow': f'◄── SWEEP FOR {high_trade_dir}'
            })
    
    # Add equal highs
    for eh in liquidity_levels.get('equal_highs', []):
        if eh['price'] > current_price:
            all_levels.append({
                'price': eh['price'], 'type': 'EQUAL_HIGH',
                'label': f"Equal High ⭐", 'color': '#ff9500', 'arrow': '◄── MAJOR'
            })
    
    # Current price marker
    all_levels.append({
        'price': current_price, 'type': 'CURRENT',
        'label': '◄ CURRENT', 'color': '#3b82f6', 'arrow': ''
    })
    
    # Add liquidity lows
    # In CONTINUATION mode: Sweep LOW → SHORT (continue down)
    # In REVERSAL mode: Sweep LOW → LONG (reverse up)
    low_trade_dir = get_trade_direction("LONG")  # Raw "LONG" = sweep of LOW
    low_color = '#00ff88' if low_trade_dir == 'LONG' else '#ff6b6b'
    for l in liquidity_levels.get('lows', [])[:3]:
        if l['price'] < current_price:
            all_levels.append({
                'price': l['price'], 'type': 'LIQUIDITY_LOW',
                'label': f"Liquidity Pool ({l.get('strength', 'MODERATE')})",
                'color': low_color, 'arrow': f'◄── SWEEP FOR {low_trade_dir}'
            })
    
    # Add equal lows
    for el in liquidity_levels.get('equal_lows', []):
        if el['price'] < current_price:
            all_levels.append({
                'price': el['price'], 'type': 'EQUAL_LOW',
                'label': f"Equal Low ⭐", 'color': '#ffcc00', 'arrow': '◄── MAJOR'
            })
    
    # Add long liquidation zones (below price)
    # For SHORT trade: levels below = TP TARGET (price going down = profit)
    # For LONG trade: levels below = RISK (SL zone)
    long_liq_label = '◄── TP TARGET' if trade_direction == 'SHORT' else '◄── RISK'
    long_liq_color = '#00ff88' if trade_direction == 'SHORT' else '#ff6b6b'
    for liq in liq_zones.get('long_liquidations', [])[:3]:
        all_levels.append({
            'price': liq['price'], 'type': 'LONG_LIQ',
            'label': f"Long Liq ({liq['leverage']}x)", 'color': long_liq_color, 'arrow': long_liq_label
        })
    
    # Sort by price descending
    all_levels.sort(key=lambda x: x['price'], reverse=True)
    
    # Render
    st.markdown("#### 📊 LIQUIDITY MAP")
    
    for level in all_levels:
        if level['type'] == 'CURRENT':
            st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center; 
                padding: 10px; margin: 4px 0; background: #1a1a2e; border-radius: 8px; border: 2px solid #3b82f6;'>
                <span style='color: #fff; font-weight: bold;'>${fmt_price(level['price'])}</span>
                <span style='color: #3b82f6; font-weight: bold;'>{level['label']}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center; 
                padding: 8px; margin: 3px 0; background: #0d0d1a; border-radius: 6px;'>
                <span style='color: #ccc;'>${fmt_price(level['price'])}</span>
                <span style='color: {level["color"]};'>{level['label']} {level['arrow']}</span>
            </div>""", unsafe_allow_html=True)


def render_sweep_status(sweep_status: dict):
    """Render sweep detection status with CONTINUATION mode trade direction"""
    if not sweep_status.get('detected'):
        st.markdown("""<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border: 1px solid #444;'>
            <div style='color: #00d4aa; font-weight: bold;'>⏳ No Active Sweep</div>
            <div style='color: #888;'>Waiting for price to sweep a liquidity level...</div>
        </div>""", unsafe_allow_html=True)
        return

    raw_direction = sweep_status.get('direction', 'LONG')
    # Get TRADE direction (flipped in CONTINUATION mode)
    trade_direction = get_trade_direction(raw_direction)
    dir_color = '#00ff88' if trade_direction == 'LONG' else '#ff6b6b'
    confidence = sweep_status.get('confidence', 0)
    level_swept = sweep_status.get('level_swept', 0)
    volume_confirmed = sweep_status.get('volume_confirmed', False)
    candles_ago = sweep_status.get('candles_ago', 0)

    # Clarify what was swept vs trade direction
    sweep_type = "LOW" if raw_direction == "LONG" else "HIGH"
    mode_note = "(ML-driven)"

    st.markdown(f"""<div style='background: #0a1a0a; border-radius: 12px; padding: 20px; border: 2px solid {dir_color};'>
        <div style='color: {dir_color}; font-weight: bold; font-size: 1.2em;'>🎯 TRADE {trade_direction} {mode_note}</div>
        <div style='margin-top: 5px; color: #888; font-size: 0.9em;'>
            Sweep of {sweep_type} at ${fmt_price(level_swept)}
        </div>
        <div style='margin-top: 10px;'>
            <span style='color: #888;'>Confidence:</span>
            <span style='color: {dir_color}; font-weight: bold;'> {confidence}%</span>
        </div>
        <div style='margin-top: 5px;'>
            <span style='color: #888;'>Age:</span>
            <span style='color: #fff;'> {candles_ago} candles ago</span>
        </div>
        <div style='margin-top: 5px;'>
            <span style='color: #888;'>Volume Spike:</span>
            <span style='color: {"#00ff88" if volume_confirmed else "#888"};'>
                {"✓ Yes" if volume_confirmed else "No"}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)


def render_approaching_levels(approaching: dict, atr: float):
    """Render approaching liquidity levels as NEXT TARGETS (CONTINUATION aware)"""
    if not approaching.get('has_nearby'):
        st.markdown("""<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border: 1px solid #444;'>
            <div style='color: #888;'>📏 No liquidity targets within range</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("#### 🎯 NEXT TARGETS")

    # Show next targets clearly
    next_long = approaching.get('next_long_target')
    next_short = approaching.get('next_short_target')

    for level in approaching.get('levels', [])[:4]:
        proximity = level.get('proximity', 'APPROACHING')
        prox_config = {
            'IMMINENT': ('#ff6b6b', '🔥 IMMINENT'),
            'CLOSE': ('#ffcc00', '⚡ CLOSE'),
            'APPROACHING': ('#888', '👀 APPROACHING')
        }
        prox_color, prox_text = prox_config.get(proximity, ('#888', '👀'))

        # Get TRADE direction (CONTINUATION aware)
        raw_direction = level.get('direction_after_sweep', 'LONG')
        trade_direction = get_trade_direction(raw_direction)
        dir_color = '#00ff88' if trade_direction == 'LONG' else '#ff6b6b'

        # Check if this is the NEXT target for this direction (use raw for matching)
        is_next = False
        if raw_direction == 'LONG' and next_long and abs(level.get('price', 0) - next_long.get('price', 0)) < 1:
            is_next = True
        elif raw_direction == 'SHORT' and next_short and abs(level.get('price', 0) - next_short.get('price', 0)) < 1:
            is_next = True

        # Show when level formed
        formed_ago = level.get('formed_ago', '')
        formed_text = f" • {formed_ago}" if formed_ago else ""

        border_style = f"2px solid {dir_color}" if is_next else "1px solid #333"

        st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center;
            padding: 12px; margin: 4px 0; background: #0d0d1a; border-radius: 8px; border: {border_style};'>
            <div>
                <span style='color: #fff; font-weight: bold;'>${level.get('price', 0):,.2f}</span>
                <span style='color: #666; font-size: 0.8em;'>{formed_text}</span>
            </div>
            <span style='color: {prox_color};'>{prox_text}</span>
            <span style='color: #888;'>{level.get('distance_atr', 0):.1f} ATR</span>
            <span style='color: {dir_color}; font-weight: bold;'>→ {trade_direction}</span>
        </div>""", unsafe_allow_html=True)


def render_trade_plan(trade_plan: dict):
    """Render the generated trade plan with ROI%, R:R, and Entry Quality."""
    status = trade_plan.get('status', 'NO_SETUP')

    if status == 'SWEEP_ENTRY':
        direction = trade_plan.get('direction', 'LONG')
        dir_color = '#00ff88' if 'LONG' in direction else '#ff6b6b'
        entry = trade_plan.get('entry', 0)
        stop_loss = trade_plan.get('stop_loss', 0)
        take_profits = trade_plan.get('take_profits', [])
        confidence = trade_plan.get('confidence', 0)

        # ML Quality
        ml_quality = trade_plan.get('ml_quality', '')
        ml_probability = trade_plan.get('ml_probability', 0)
        ml_prediction = trade_plan.get('ml_prediction', {})

        # Entry Quality - ML ONLY
        entry_quality = trade_plan.get('entry_quality', {})
        eq_window = entry_quality.get('entry_window', 'NO_SWEEP') if entry_quality else 'NO_SWEEP'
        eq_candles_left = entry_quality.get('candles_until_close', 0) if entry_quality else 0
        eq_recommendation = entry_quality.get('recommendation', 'UNKNOWN') if entry_quality else 'UNKNOWN'
        eq_warnings = entry_quality.get('warnings', []) if entry_quality else []
        eq_ml_prob = entry_quality.get('ml_probability', 0) if entry_quality else 0

        # ML-ONLY badge (replaces grade badge)
        ml_badge = ""
        if eq_ml_prob and eq_ml_prob > 0:
            if eq_ml_prob >= 0.60:
                ml_color = '#00ff88'
                ml_label = 'STRONG'
            elif eq_ml_prob >= 0.50:
                ml_color = '#88ff00'
                ml_label = 'GOOD'
            elif eq_ml_prob >= 0.40:
                ml_color = '#ffaa00'
                ml_label = 'WEAK'
            else:
                ml_color = '#ff4444'
                ml_label = 'SKIP'
            ml_badge = f"<span style='background: {ml_color}22; padding: 3px 8px; border-radius: 10px; border: 1px solid {ml_color}; margin-left: 10px;'><span style='color: {ml_color};'>🤖 {eq_ml_prob:.0%} {ml_label}</span></span>"

        # Recommendation badge
        eq_badge = ""
        rec_config = {
            'ENTER': ('✅', '#00ff88'),
            'WAIT': ('⏳', '#ffcc00'),
            'SKIP': ('🚫', '#ff4444'),
            'TRIM': ('✂️', '#ff8800'),
        }
        rec_emoji, rec_color = rec_config.get(eq_recommendation, ('❓', '#888'))
        eq_badge = f"""<span style='background: {rec_color}22; padding: 3px 8px; border-radius: 10px; border: 1px solid {rec_color}; margin-left: 5px;'>
            <span style='color: {rec_color}; font-weight: bold;'>{rec_emoji} {eq_recommendation}</span>
        </span>"""

        # Calculate SL %
        if entry > 0:
            if 'LONG' in direction:
                sl_pct = ((entry - stop_loss) / entry) * 100
            else:
                sl_pct = ((stop_loss - entry) / entry) * 100
        else:
            sl_pct = 0

        st.markdown(f"""<div style='background: #0a1a0a; border-radius: 12px; padding: 20px; border: 2px solid {dir_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='color: {dir_color}; font-weight: bold; font-size: 1.3em;'>🎯 TRADE PLAN - {direction}{ml_badge}{eq_badge}</div>
                <div style='background: {dir_color}22; padding: 5px 12px; border-radius: 20px; border: 1px solid {dir_color};'>
                    <span style='color: {dir_color};'>{confidence}% Confidence</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Entry Window Warning (NEW)
        if eq_window == 'OPEN' and eq_candles_left > 0:
            st.success(f"✅ **Entry Window OPEN** - {eq_candles_left} candles left for optimal entry")
        elif eq_window == 'CLOSING' and eq_candles_left > 0:
            st.warning(f"⏳ **Entry Window CLOSING** - {eq_candles_left} candles left, consider limit order")
        elif eq_window == 'CLOSED':
            st.error("❌ **Entry Window CLOSED** - Wait for retest or next sweep")

        # ENHANCED: Show Recommendation Box
        if eq_recommendation == 'ENTER':
            st.markdown("""<div style='background: #00ff8822; padding: 12px 16px; border-radius: 8px;
                border-left: 4px solid #00ff88; margin: 10px 0;'>
                <span style='color: #00ff88; font-weight: bold; font-size: 1.1em;'>✅ ENTER - Setup confirmed with reversal signals</span>
            </div>""", unsafe_allow_html=True)
        elif eq_recommendation == 'WAIT':
            st.markdown("""<div style='background: #ffcc0022; padding: 12px 16px; border-radius: 8px;
                border-left: 4px solid #ffcc00; margin: 10px 0;'>
                <span style='color: #ffcc00; font-weight: bold; font-size: 1.1em;'>⏳ WAIT - Setup decent but needs more confirmation</span>
            </div>""", unsafe_allow_html=True)
        elif eq_recommendation == 'SKIP':
            st.markdown("""<div style='background: #ff444422; padding: 12px 16px; border-radius: 8px;
                border-left: 4px solid #ff4444; margin: 10px 0;'>
                <span style='color: #ff4444; font-weight: bold; font-size: 1.1em;'>🚫 SKIP - Setup too weak or reversal failed</span>
            </div>""", unsafe_allow_html=True)

        # ETF/Stock: Show primary action (ACCUMULATE / TRIM / HOLD)
        etf_action = entry_quality.get('etf_action') if entry_quality else None
        etf_flow = entry_quality.get('etf_flow') if entry_quality else None
        if etf_action:
            action_config = {
                'ACCUMULATE': ('💰', 'ACCUMULATE', '#00ff88', 'Money flowing in — add to position on this dip'),
                'TRIM_5_10': ('📉', 'TRIM 5-10%', '#ff8800', 'Money flowing out or price extended — reduce exposure'),
                'TRIM_15_20': ('📉', 'TRIM 15-20%', '#ff4444', 'Price very extended — take significant profits'),
                'HOLD': ('⏸️', 'HOLD', '#888888', 'Flow neutral — wait for clearer signal'),
            }
            a_emoji, a_label, a_color, a_desc = action_config.get(etf_action, ('❓', etf_action, '#888', ''))
            st.markdown(f"""<div style='background: {a_color}15; padding: 16px 20px; border-radius: 10px;
                border: 2px solid {a_color}; margin: 12px 0;'>
                <div style='color: {a_color}; font-weight: bold; font-size: 1.4em;'>{a_emoji} {a_label}</div>
                <div style='color: #ccc; margin-top: 6px;'>{a_desc}</div>
            </div>""", unsafe_allow_html=True)
        # ENHANCED: Show Warnings (collapsed by default - expand if needed)
        if eq_warnings:
            with st.expander("📋 Entry Quality Signals & Warnings", expanded=False):
                for w in eq_warnings:
                    if w.startswith('✅') or w.startswith('🎯'):
                        st.success(w)
                    elif w.startswith('⏳'):
                        st.warning(w)
                    elif w.startswith('🚨') or w.startswith('🚫') or w.startswith('⚠️'):
                        st.error(w)
                    else:
                        st.info(w)
        
        # ML Factors (if available)
        if ml_prediction and ml_prediction.get('factors'):
            factors = ml_prediction['factors']
            positive = factors.get('positive', [])
            negative = factors.get('negative', [])
            
            factors_html = ""
            if positive:
                factors_html += f"<span style='color: #00ff88;'>✓ {', '.join(positive)}</span>"
            if negative:
                if factors_html:
                    factors_html += " | "
                factors_html += f"<span style='color: #ff6b6b;'>✗ {', '.join(negative)}</span>"
            
            if factors_html:
                st.markdown(f"""<div style='background: #0d0d1a; padding: 8px 15px; border-radius: 6px; margin-top: 5px;'>
                    <span style='color: #888; font-size: 0.85em;'>ML Factors: </span>{factors_html}
                </div>""", unsafe_allow_html=True)
        
        # Entry and Stop Loss
        st.markdown(f"""<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;'>
            <div style='background: #1a1a2e; padding: 15px; border-radius: 8px; border-left: 3px solid #3b82f6;'>
                <div style='color: #888; font-size: 0.85em;'>ENTRY</div>
                <div style='color: #3b82f6; font-weight: bold; font-size: 1.3em;'>${fmt_price(entry)}</div>
            </div>
            <div style='background: #1a1a2e; padding: 15px; border-radius: 8px; border-left: 3px solid #ff6b6b;'>
                <div style='color: #888; font-size: 0.85em;'>STOP LOSS</div>
                <div style='color: #ff6b6b; font-weight: bold; font-size: 1.3em;'>${fmt_price(stop_loss)}</div>
                <div style='color: #ff6b6b; font-size: 0.9em;'>-{sl_pct:.2f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Take Profits with ROI% and R:R
        if take_profits:
            st.markdown("<div style='margin-top: 15px; color: #888; font-size: 0.9em;'>TAKE PROFITS</div>", unsafe_allow_html=True)
            
            for tp in take_profits:
                tp_price = tp.get('price', 0)
                tp_rr = tp.get('rr', 0)
                tp_level = tp.get('level', 1)
                tp_reason = tp.get('reason', '')
                
                # Use roi from trade plan or calculate
                tp_roi = tp.get('roi', 0)
                if tp_roi == 0 and entry > 0:
                    if 'LONG' in direction:
                        tp_roi = ((tp_price - entry) / entry) * 100
                    else:
                        tp_roi = ((entry - tp_price) / entry) * 100
                
                # Color based on R:R
                rr_color = '#00ff88' if tp_rr >= 2 else '#ffcc00' if tp_rr >= 1 else '#ff6b6b'
                
                st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center;
                    padding: 12px; margin: 5px 0; background: #0d0d1a; border-radius: 8px; border-left: 3px solid #00d4aa;'>
                    <div>
                        <span style='color: #00d4aa; font-weight: bold;'>TP{tp_level}</span>
                        <span style='color: #fff; font-weight: bold; margin-left: 10px;'>${fmt_price(tp_price)}</span>
                        <span style='color: #888; margin-left: 10px; font-size: 0.85em;'>{tp_reason}</span>
                    </div>
                    <div style='display: flex; gap: 15px;'>
                        <span style='color: #00ff88; font-weight: bold;'>+{tp_roi:.2f}%</span>
                        <span style='color: {rr_color}; font-weight: bold;'>{tp_rr:.1f}R</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ENHANCED: Show detailed Entry Quality breakdown (ML components)
        if entry_quality and eq_ml_prob and eq_ml_prob > 0:
            with st.expander("📊 ML Quality Details", expanded=False):
                render_entry_quality_breakdown(entry_quality)

    elif status == 'WAITING_FOR_SWEEP':
        # Get the sweep level (the level we're watching)
        sweep_level = trade_plan.get('sweep_level', 0)
        direction = trade_plan.get('direction', 'LONG')
        proximity = trade_plan.get('proximity', 'APPROACHING')
        confidence = trade_plan.get('confidence', 50)
        
        # Get estimated levels
        est_entry = trade_plan.get('est_entry', sweep_level)
        est_sl = trade_plan.get('est_sl', 0)
        est_sl_pct = trade_plan.get('est_sl_pct', 0)
        est_tps = trade_plan.get('est_tps', [])
        
        dir_color = '#00ff88' if 'LONG' in direction else '#ff6b6b'
        prox_color = '#ff6b6b' if proximity == 'IMMINENT' else '#ffcc00' if proximity == 'CLOSE' else '#888'
        
        # Header
        st.markdown(f"""<div style='background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #3b82f6;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='color: #3b82f6; font-weight: bold; font-size: 1.1em;'>⏳ WAITING FOR SWEEP</div>
                <div style='display: flex; gap: 10px;'>
                    <span style='background: {prox_color}22; padding: 5px 12px; border-radius: 20px; border: 1px solid {prox_color}; color: {prox_color};'>{proximity}</span>
                    <span style='background: {dir_color}22; padding: 5px 12px; border-radius: 20px; border: 1px solid {dir_color}; color: {dir_color};'>{confidence}%</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Estimated Entry and Stop Loss
        st.markdown(f"""<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;'>
            <div style='background: #0d0d1a; padding: 15px; border-radius: 8px; border-left: 3px solid #3b82f6;'>
                <div style='color: #888; font-size: 0.85em;'>EST. ENTRY (at sweep level)</div>
                <div style='color: #3b82f6; font-weight: bold; font-size: 1.3em;'>${fmt_price(est_entry)}</div>
            </div>
            <div style='background: #0d0d1a; padding: 15px; border-radius: 8px; border-left: 3px solid #ff6b6b;'>
                <div style='color: #888; font-size: 0.85em;'>EST. STOP LOSS</div>
                <div style='color: #ff6b6b; font-weight: bold; font-size: 1.3em;'>${fmt_price(est_sl)}</div>
                <div style='color: #ff6b6b; font-size: 0.9em;'>-{est_sl_pct:.2f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Estimated Take Profits with ROI% and R:R
        if est_tps:
            st.markdown("<div style='margin-top: 15px; color: #888; font-size: 0.9em;'>ESTIMATED TAKE PROFITS</div>", unsafe_allow_html=True)
            
            for tp in est_tps:
                tp_price = tp.get('price', 0)
                tp_rr = tp.get('rr', 0)
                tp_roi = tp.get('roi', 0)
                tp_level = tp.get('level', 1)
                tp_reason = tp.get('reason', '')
                
                # Color based on R:R
                rr_color = '#00ff88' if tp_rr >= 2 else '#ffcc00' if tp_rr >= 1 else '#ff6b6b'
                
                st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center;
                    padding: 12px; margin: 5px 0; background: #0d0d1a; border-radius: 8px; border-left: 3px solid #00d4aa;'>
                    <div>
                        <span style='color: #00d4aa; font-weight: bold;'>TP{tp_level}</span>
                        <span style='color: #fff; font-weight: bold; margin-left: 10px;'>${fmt_price(tp_price)}</span>
                        <span style='color: #888; margin-left: 10px; font-size: 0.85em;'>{tp_reason}</span>
                    </div>
                    <div style='display: flex; gap: 15px;'>
                        <span style='color: #00ff88; font-weight: bold;'>+{tp_roi:.2f}%</span>
                        <span style='color: {rr_color}; font-weight: bold;'>{tp_rr:.1f}R</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        
        # Determine sweep direction for instructions
        is_long = 'LONG' in direction
        sweep_word = "below" if is_long else "above"
        close_word = "above" if is_long else "below"
        bg_color = "#1a2a1a" if is_long else "#2a1a1a"
        buy_sell = "BUY" if is_long else "SELL"
        long_short = "LONG" if is_long else "SHORT"
        
        # Entry Options Header
        st.markdown(f"""<div style='margin-top: 15px; background: #0d1117; padding: 15px; border-radius: 8px; border: 1px solid {dir_color};'>
            <div style='color: {dir_color}; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>📍 HOW TO ENTER - {direction}</div>
        </div>""", unsafe_allow_html=True)
        
        # Option 1
        st.markdown(f"""<div style='background: {bg_color}; padding: 12px; border-radius: 8px; margin-top: 5px; border-left: 3px solid {dir_color};'>
            <div style='color: {dir_color}; font-weight: bold;'>OPTION 1: LIMIT ORDER (Aggressive)</div>
            <div style='color: #ccc; margin-top: 5px;'>
                ✅ Place limit {buy_sell} at ${fmt_price(sweep_level)}
            </div>
            <div style='color: #ccc;'>
                ✅ SL at ${fmt_price(est_sl)} (-{est_sl_pct:.2f}%)
            </div>
            <div style='color: #ffcc00;'>
                ⚠️ Risk: Price may sweep further before reversing
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Option 2
        st.markdown(f"""<div style='background: #1a1a2a; padding: 12px; border-radius: 8px; margin-top: 5px; border-left: 3px solid #3b82f6;'>
            <div style='color: #3b82f6; font-weight: bold;'>OPTION 2: WAIT FOR REJECTION (Safer)</div>
            <div style='color: #ccc; margin-top: 5px;'>
                1️⃣ Wait for price to sweep {sweep_word} ${sweep_level:,.2f}
            </div>
            <div style='color: #ccc;'>
                2️⃣ Wait for candle to CLOSE back {close_word} the level
            </div>
            <div style='color: #ccc;'>
                3️⃣ Enter {long_short} on next candle
            </div>
            <div style='color: #00ff88;'>
                ✅ Higher win rate, might miss some moves
            </div>
        </div>""", unsafe_allow_html=True)
    
    elif status == 'ML_SKIP':
        # ML says SKIP - show clear warning, don't show entry/SL/TP
        entry_quality = trade_plan.get('entry_quality', {})
        ml_probability = entry_quality.get('ml_probability', 0) if entry_quality else 0
        eq_warnings = entry_quality.get('warnings', []) if entry_quality else []

        # Dynamic message based on actual ML probability
        if ml_probability < 0.40:
            prob_msg = f"ML Probability: {ml_probability:.0%} (below 40% threshold)"
            reason_msg = "The ML model detected this setup is likely to fail. Do NOT enter."
        else:
            prob_msg = f"ML Probability: {ml_probability:.0%}"
            reason_msg = "Setup filtered due to other factors. Check warnings below."

        st.markdown(f"""<div style='background: #2a1a1a; border-radius: 12px; padding: 20px; border: 2px solid #ff6b6b;'>
            <div style='color: #ff6b6b; font-weight: bold; font-size: 1.3em;'>🚫 ML SAYS SKIP - NO TRADE</div>
            <div style='color: #ff8888; margin-top: 10px; font-size: 1.1em;'>
                {prob_msg}
            </div>
            <div style='color: #888; margin-top: 10px;'>
                {reason_msg}
            </div>
        </div>""", unsafe_allow_html=True)

        # Show warnings if any
        if eq_warnings:
            st.markdown("#### ⚠️ Why ML Says Skip:")
            for warning in eq_warnings[:5]:  # Show top 5 warnings
                st.markdown(f"- {warning}")

    else:
        st.markdown("""<div style='background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #444;'>
            <div style='color: #888; font-weight: bold;'>📋 NO ACTIVE SETUP</div>
            <div style='color: #666; margin-top: 8px;'>Wait for price to sweep a liquidity level and reject.</div>
            <div style='color: #666; margin-top: 5px; font-size: 0.9em;'>Best setups come AFTER sweeps, not before.</div>
        </div>""", unsafe_allow_html=True)


def render_whale_positioning(liquidation_data: dict, whale_delta: dict = None):
    """Render whale vs retail positioning with change over time"""
    if not liquidation_data.get('success'):
        st.markdown("""<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border: 1px solid #444;'>
            <div style='color: #888;'>🐋 Whale data unavailable</div>
        </div>""", unsafe_allow_html=True)
        return
    
    whale_long = liquidation_data.get('whale_long', 50)
    whale_short = liquidation_data.get('whale_short', 50)
    retail_long = liquidation_data.get('retail_long', 50)
    funding = liquidation_data.get('funding_rate', 0)
    
    whale_bias = "BULLISH" if whale_long > 55 else ("BEARISH" if whale_short > 55 else "NEUTRAL")
    whale_color = "#00ff88" if whale_long > 55 else ("#ff6b6b" if whale_short > 55 else "#888")
    
    st.markdown("#### 🐋 WHALE POSITIONING")
    
    # Main positioning
    st.markdown(f"""<div style='background: #1a1a2e; border-radius: 10px; padding: 15px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
            <span style='color: #888;'>Whales:</span>
            <span style='color: {whale_color}; font-weight: bold;'>{whale_bias} ({whale_long:.0f}% Long)</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
            <span style='color: #888;'>Retail:</span>
            <span style='color: #ccc;'>{retail_long:.0f}% Long</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #888;'>Funding:</span>
            <span style='color: {"#00ff88" if funding < 0 else "#ff6b6b"};'>{funding:.4f}%</span>
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Delta section (separate markdown call)
    if whale_delta and whale_delta.get('data_available'):
        w_delta = whale_delta.get('whale_delta', 0) or 0
        r_delta = whale_delta.get('retail_delta', 0) or 0
        lookback = whale_delta.get('lookback_label', '24h')
        
        w_delta_color = '#00ff88' if w_delta > 0 else '#ff6b6b' if w_delta < 0 else '#888'
        r_delta_color = '#00ff88' if r_delta > 0 else '#ff6b6b' if r_delta < 0 else '#888'
        w_arrow = '↑' if w_delta > 0 else '↓' if w_delta < 0 else '→'
        r_arrow = '↑' if r_delta > 0 else '↓' if r_delta < 0 else '→'
        
        st.markdown(f"""<div style='margin-top: 10px; padding: 10px; background: #0d0d1a; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.9em; margin-bottom: 8px;'>Change ({lookback})</div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: #888;'>Whales:</span>
                <span style='color: {w_delta_color}; font-weight: bold;'>{w_arrow} {w_delta:+.1f}%</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span style='color: #888;'>Retail:</span>
                <span style='color: {r_delta_color};'>{r_arrow} {r_delta:+.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY QUALITY UI COMPONENTS (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def render_entry_quality_badge(entry_quality: dict) -> str:
    """
    Render entry quality as a colored badge HTML string.

    ML-ONLY version - Shows ML probability and recommendation.

    Badge format: [ML %] [RECOMMENDATION] [ENTRY WINDOW]

    Args:
        entry_quality: Dict from calculate_entry_quality()

    Returns:
        HTML string for st.markdown
    """
    if entry_quality is None:
        return ""

    entry_window = entry_quality.get('entry_window', 'NO_SWEEP')
    candles_until = entry_quality.get('candles_until_close', 0)
    recommendation = entry_quality.get('recommendation', 'UNKNOWN')
    ml_prob = entry_quality.get('ml_probability', 0) or 0

    # ML probability determines color and emoji
    if ml_prob >= 0.60:
        ml_color = '#00ff88'
        ml_label = 'STRONG'
    elif ml_prob >= 0.50:
        ml_color = '#88ff00'
        ml_label = 'GOOD'
    elif ml_prob >= 0.40:
        ml_color = '#ffaa00'
        ml_label = 'WEAK'
    else:
        ml_color = '#ff4444'
        ml_label = 'SKIP'

    # Recommendation badge (derived from ML)
    rec_config = {
        'ENTER': {'emoji': '✅', 'color': '#00ff88', 'text': 'ENTER'},
        'WAIT': {'emoji': '⏳', 'color': '#ffcc00', 'text': 'WAIT'},
        'SKIP': {'emoji': '🚫', 'color': '#ff4444', 'text': 'SKIP'},
        'NO_SETUP': {'emoji': '⚪', 'color': '#888888', 'text': 'NO SETUP'},
        'UNKNOWN': {'emoji': '❓', 'color': '#888888', 'text': '?'}
    }
    rec = rec_config.get(recommendation, rec_config['UNKNOWN'])

    # Entry window indicator
    window_html = ""
    if entry_window == 'OPEN':
        window_html = f"<span style='color: #00ff88; margin-left: 8px;'>🟢 OPEN ({candles_until}c)</span>"
    elif entry_window == 'CLOSING':
        window_html = f"<span style='color: #ffcc00; margin-left: 8px;'>🟡 CLOSING ({candles_until}c)</span>"
    elif entry_window == 'CLOSED':
        window_html = f"<span style='color: #888; margin-left: 8px;'>⚫ OLD</span>"

    return f"""
    <div style='display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap;'>
        <span style='background: {ml_color}22; padding: 4px 12px; border-radius: 12px; border: 1px solid {ml_color};'>
            <span style='color: {ml_color}; font-weight: bold;'>🤖 {ml_prob:.0%}</span>
            <span style='color: #aaa; margin-left: 4px; font-size: 0.85em;'>{ml_label}</span>
        </span>
        <span style='background: {rec["color"]}22; padding: 4px 10px; border-radius: 8px; border: 1px solid {rec["color"]};'>
            <span style='color: {rec["color"]}; font-weight: bold;'>{rec["emoji"]} {rec["text"]}</span>
        </span>
        {window_html}
    </div>
    """


def render_ml_features_debug(ml_features: dict):
    """
    Render ALL 32 ML features in a debug panel.
    This shows exactly what the ML model receives for prediction.
    """
    if not ml_features:
        st.caption("No ML features available")
        return

    # Group features by category for clarity
    st.markdown("##### 📊 Raw Input Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Setup Info**")
        st.code(f"""Direction: {ml_features.get('direction', 'N/A')}
Level Type: {ml_features.get('level_type', 'N/A')}
Level Price: ${ml_features.get('level_price', 0):,.4f}
Current Price: ${ml_features.get('current_price', 0):,.4f}
ATR: {ml_features.get('atr', 0):.4f}""")

    with col2:
        st.markdown("**Distance & Momentum**")
        st.code(f"""Distance ATR: {ml_features.get('distance_atr', 0):.2f}
Momentum: {ml_features.get('momentum', 0):.3f}
Volume Ratio: {ml_features.get('volume_ratio', 0):.2f}
Volatility: {ml_features.get('volatility', 0):.2f}""")

    with col3:
        st.markdown("**Level Quality**")
        st.code(f"""Level Type Score: {ml_features.get('level_type_score', 0):.2f}
Level Distance Score: {ml_features.get('level_distance_score', 0):.2f}
Level Quality: {ml_features.get('level_quality', 0):.2f}
Level Tradeable: {ml_features.get('level_tradeable', 0)}
Is Long: {ml_features.get('is_long', 0)}""")

    st.markdown("##### 🐋 Whale Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Whale Position**")
        whale_pct = ml_features.get('whale_pct', 50)
        whale_delta = ml_features.get('whale_delta', 0)
        whale_color = '#00ff88' if whale_pct >= 55 else '#ff6b6b' if whale_pct <= 45 else '#888'
        st.code(f"""Whale %: {whale_pct:.1f}%
Whale Delta: {whale_delta:+.2f}%
Quality: {ml_features.get('whale_quality', 0):.2f}
Aligned: {ml_features.get('whale_aligned', 0)}
Strong: {ml_features.get('whale_strong', 0)}
Trap: {ml_features.get('whale_trap', 0)}""")

    with col2:
        st.markdown("**Whale Behavior**")
        st.code(f"""Accumulating: {ml_features.get('whale_accumulating', 0)}
Distributing: {ml_features.get('whale_distributing', 0)}
24h Delta: {ml_features.get('whale_delta_24h', 0):+.2f}%
7d Delta: {ml_features.get('whale_delta_7d', 0):+.2f}%
Daily Avg 7d: {ml_features.get('whale_daily_avg_7d', 0):+.2f}%""")

    with col3:
        st.markdown("**Whale Acceleration**")
        st.code(f"""Accel Ratio: {ml_features.get('whale_acceleration_ratio', 0):.2f}
Accelerating: {ml_features.get('whale_accel_accelerating', 0)}
Decelerating: {ml_features.get('whale_accel_decelerating', 0)}
Reversing: {ml_features.get('whale_accel_reversing', 0)}
Steady: {ml_features.get('whale_accel_steady', 0)}
Is Fresh: {ml_features.get('whale_is_fresh', 0)}
Late Entry: {ml_features.get('whale_is_late_entry', 0)}""")

    st.markdown("##### 📈 Price Action Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Candle Analysis**")
        st.code(f"""PA Candle Score: {ml_features.get('pa_candle_score', 0)}
PA Structure Score: {ml_features.get('pa_structure_score', 0)}
PA Total Score: {ml_features.get('pa_total_score', 0)}""")

    with col2:
        st.markdown("**Structure**")
        st.code(f"""Has Order Block: {ml_features.get('pa_has_order_block', 0)}
Has FVG: {ml_features.get('pa_has_fvg', 0)}
PA Volume Score: {ml_features.get('pa_volume_score', 0)}
PA Momentum Score: {ml_features.get('pa_momentum_score', 0)}""")

    # Summary: Key features that matter most
    st.markdown("##### ⚡ Key Decision Factors")
    distance = ml_features.get('distance_atr', 0)
    whale_aligned = ml_features.get('whale_aligned', 0)
    whale_trap = ml_features.get('whale_trap', 0)
    is_fresh = ml_features.get('whale_is_fresh', 0)
    late_entry = ml_features.get('whale_is_late_entry', 0)

    issues = []
    positives = []

    if distance > 2.0:
        issues.append(f"⚠️ Distance {distance:.1f} ATR (>2 = late entry)")
    elif distance <= 1.0:
        positives.append(f"✅ Close to level ({distance:.1f} ATR)")

    if whale_trap:
        issues.append("⚠️ WHALE TRAP detected")
    if whale_aligned:
        positives.append("✅ Whale aligned with direction")

    if late_entry:
        issues.append("⚠️ Late entry signal")
    if is_fresh:
        positives.append("✅ Fresh accumulation/distribution")

    if ml_features.get('whale_distributing') and ml_features.get('is_long'):
        issues.append("⚠️ Going LONG while whales distribute")
    if ml_features.get('whale_accumulating') and not ml_features.get('is_long'):
        issues.append("⚠️ Going SHORT while whales accumulate")

    col1, col2 = st.columns(2)
    with col1:
        if positives:
            for p in positives:
                st.success(p)
        else:
            st.caption("No strong positive signals")
    with col2:
        if issues:
            for i in issues:
                st.warning(i)
        else:
            st.caption("No major issues detected")


def render_entry_quality_breakdown(entry_quality: dict):
    """
    Render detailed breakdown of entry quality components using Streamlit.

    ENHANCED to show 6 components in 2 rows:
    Row 1: Freshness, Reversal Candle, Follow-Through
    Row 2: Momentum, Whale, Volume

    Also shows warnings and structure break bonus.
    """
    if entry_quality is None:
        st.caption("No entry quality data")
        return

    components = entry_quality.get('components', {})
    warnings = entry_quality.get('warnings', [])
    recommendation = entry_quality.get('recommendation', 'UNKNOWN')

    st.markdown("#### 📊 Entry Quality Breakdown (ENHANCED)")

    # Show recommendation prominently
    rec_colors = {
        'ENTER': ('#00ff88', '✅ ENTER - Setup confirmed'),
        'WAIT': ('#ffcc00', '⏳ WAIT - Needs more confirmation'),
        'SKIP': ('#ff4444', '🚫 SKIP - Setup too weak or failed'),
        'NO_SETUP': ('#888888', '⚪ NO SETUP'),
    }
    rec_color, rec_text = rec_colors.get(recommendation, ('#888888', f'❓ {recommendation}'))
    st.markdown(f"<div style='background: {rec_color}22; padding: 8px 16px; border-radius: 8px; "
                f"border-left: 4px solid {rec_color}; margin-bottom: 16px;'>"
                f"<span style='color: {rec_color}; font-weight: bold; font-size: 1.1em;'>{rec_text}</span></div>",
                unsafe_allow_html=True)

    # ROW 1: Freshness, Reversal Candle, Follow-Through
    col1, col2, col3 = st.columns(3)

    with col1:
        fresh = components.get('freshness', {})
        score = fresh.get('score', 0)
        max_score = fresh.get('max', 20)
        label = fresh.get('label', 'N/A')
        candles = fresh.get('candles_ago', 0)

        color = '#00ff88' if score >= 15 else '#ffcc00' if score >= 8 else '#ff4444'
        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>⏱️ FRESHNESS</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>{label}</div>
            <div style='color: #666; font-size: 0.8em;'>{candles} candles ago</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        rev = components.get('reversal_candle', {})
        score = rev.get('score', 0)
        max_score = rev.get('max', 25)
        has_rejection = rev.get('has_rejection', False)
        favorable_close = rev.get('favorable_close', False)
        is_engulfing = rev.get('is_engulfing', False)
        strength = rev.get('strength', 'WEAK')

        color = '#00ff88' if score >= 18 else '#ffcc00' if score >= 10 else '#ff4444'
        indicators = []
        if has_rejection:
            indicators.append('✓ Rejection')
        if favorable_close:
            indicators.append('✓ Close')
        if is_engulfing:
            indicators.append('✓ Engulf')
        ind_text = ', '.join(indicators) if indicators else '✗ Weak candle'

        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>🕯️ REVERSAL CANDLE</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>{strength}</div>
            <div style='color: #666; font-size: 0.75em;'>{ind_text}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        follow = components.get('follow_through', {})
        score = follow.get('score', 0)
        max_score = follow.get('max', 15)
        has_follow = follow.get('has_follow_through', False)
        price_held = follow.get('price_held', False)
        cont_candles = follow.get('continuation_candles', 0)
        strength = follow.get('strength', 'NONE')

        color = '#00ff88' if score >= 10 else '#ffcc00' if score >= 5 else '#ff4444'
        status_icon = '✓' if price_held else '✗'
        follow_icon = '✓' if has_follow else '✗'

        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>📈 FOLLOW-THROUGH</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>{strength}</div>
            <div style='color: #666; font-size: 0.75em;'>{status_icon} Held | {follow_icon} Cont. ({cont_candles})</div>
        </div>""", unsafe_allow_html=True)

    st.write("")  # Spacer

    # ROW 2: Momentum, Whale, Volume
    col4, col5, col6 = st.columns(3)

    with col4:
        mom = components.get('momentum', {})
        score = mom.get('score', 0)
        max_score = mom.get('max', 15)
        rsi = mom.get('rsi_value', 50)
        rsi_extreme = mom.get('rsi_extreme', False)
        has_div = mom.get('has_divergence', False)
        aligned = mom.get('momentum_aligned', False)

        color = '#00ff88' if score >= 10 else '#ffcc00' if score >= 5 else '#ff4444'
        indicators = []
        if rsi_extreme:
            indicators.append('✓ RSI Extreme')
        if has_div:
            indicators.append('✓ Divergence')
        if aligned:
            indicators.append('✓ Aligned')
        ind_text = ', '.join(indicators) if indicators else '✗ Neutral'

        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>📊 MOMENTUM</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>RSI: {rsi:.0f}</div>
            <div style='color: #666; font-size: 0.75em;'>{ind_text}</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        whale = components.get('whale', {})
        score = whale.get('score', 0)
        max_score = whale.get('max', 15)
        pct = whale.get('pct', 50)
        delta = whale.get('delta', 0)
        behavior = whale.get('behavior', 'HOLDING')
        aligned = whale.get('aligned', False)

        color = '#00ff88' if score >= 12 else '#ffcc00' if score >= 6 else '#ff4444'
        align_emoji = '✓' if aligned else '✗'

        # Behavior indicator - THE KEY!
        if behavior == 'ACCUMULATING':
            behavior_color = '#00ff88'
            behavior_emoji = '📈'
        elif behavior == 'DISTRIBUTING':
            behavior_color = '#ff4444'
            behavior_emoji = '📉'
        else:
            behavior_color = '#888'
            behavior_emoji = '➡️'

        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>🐋 WHALE</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>{pct:.0f}% <span style='color: {behavior_color};'>({delta:+.1f}%)</span></div>
            <div style='color: {behavior_color}; font-size: 0.8em; font-weight: bold;'>{behavior_emoji} {behavior}</div>
        </div>""", unsafe_allow_html=True)

    with col6:
        vol = components.get('volume', {})
        score = vol.get('score', 0)
        max_score = vol.get('max', 10)
        ratio = vol.get('ratio', 1.0)

        color = '#00ff88' if score >= 7 else '#ffcc00' if score >= 4 else '#ff4444'
        st.markdown(f"""<div style='text-align: center; background: #1a1a2e; padding: 12px; border-radius: 8px;'>
            <div style='color: #888; font-size: 0.8em;'>📊 VOLUME</div>
            <div style='color: {color}; font-size: 1.5em; font-weight: bold;'>{score}/{max_score}</div>
            <div style='color: #ccc; font-size: 0.9em;'>{ratio:.1f}x avg</div>
        </div>""", unsafe_allow_html=True)

    # Structure Break Bonus (if present)
    struct = components.get('structure_break', {})
    if struct.get('broken', False):
        st.markdown(f"""<div style='background: #00ff8822; padding: 8px 16px; border-radius: 8px;
            border: 1px solid #00ff88; margin-top: 12px; text-align: center;'>
            <span style='color: #00ff88; font-weight: bold;'>🎯 +5 BONUS: Structure Break Confirmed</span>
            <span style='color: #ccc; margin-left: 8px;'>({struct.get("type", "?")} at ${struct.get("level", 0):,.2f})</span>
        </div>""", unsafe_allow_html=True)

    # TREND ALIGNMENT (Critical indicator!)
    trend = components.get('trend', {})
    trend_name = trend.get('trend', 'UNKNOWN')
    trend_aligned = trend.get('aligned', False)
    trend_score = trend.get('score', 0)

    if trend_name != 'UNKNOWN':
        if trend_aligned:
            trend_color = '#00ff88'
            trend_emoji = '✅'
            trend_label = f"WITH {trend_name}"
        elif trend_score <= -10:
            trend_color = '#ff4444'
            trend_emoji = '🚨'
            trend_label = f"AGAINST {trend_name} (HIGH RISK!)"
        else:
            trend_color = '#ffcc00'
            trend_emoji = '⚠️'
            trend_label = f"{trend_name} (Caution)"

        price_vs_ema = trend.get('price_vs_ema200', 0)
        ema_text = f"Price is {abs(price_vs_ema):.1f}% {'above' if price_vs_ema > 0 else 'below'} EMA200"

        st.markdown(f"""<div style='background: {trend_color}22; padding: 12px 16px; border-radius: 8px;
            border: 1px solid {trend_color}; margin-top: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: {trend_color}; font-weight: bold; font-size: 1.1em;'>{trend_emoji} TREND: {trend_label}</span>
                <span style='color: {trend_color}; font-weight: bold;'>{trend_score:+d} pts</span>
            </div>
            <div style='color: #888; font-size: 0.85em; margin-top: 4px;'>{ema_text}</div>
        </div>""", unsafe_allow_html=True)

    # ML QUALITY MODEL PREDICTION (if available)
    ml_quality = components.get('ml_quality', {})
    if ml_quality.get('available', False):
        ml_prob = ml_quality.get('probability', 0.5)
        ml_decision = ml_quality.get('decision', 'UNKNOWN')
        ml_win_rate = ml_quality.get('expected_win_rate', 0.5)
        ml_roi = ml_quality.get('expected_roi', 0)
        ml_features = ml_quality.get('features', {})

        # Color based on decision
        if ml_decision == 'STRONG_YES':
            ml_color = '#00ff88'
            ml_emoji = '🔥'
            ml_label = 'STRONG YES'
        elif ml_decision == 'YES':
            ml_color = '#88ff00'
            ml_emoji = '✅'
            ml_label = 'YES'
        elif ml_decision == 'MAYBE':
            ml_color = '#ffaa00'
            ml_emoji = '⚠️'
            ml_label = 'MAYBE'
        else:
            ml_color = '#ff4444'
            ml_emoji = '❌'
            ml_label = 'NO'

        st.markdown(f"""<div style='background: {ml_color}22; padding: 12px 16px; border-radius: 8px;
            border: 1px solid {ml_color}; margin-top: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: {ml_color}; font-weight: bold; font-size: 1.1em;'>
                    🤖 ML QUALITY: {ml_emoji} {ml_label}
                </span>
                <span style='color: {ml_color}; font-weight: bold; font-size: 1.2em;'>
                    {ml_prob:.0%}
                </span>
            </div>
            <div style='color: #888; font-size: 0.85em; margin-top: 4px;'>
                Expected Win Rate: {ml_win_rate:.1%} | ROI/Trade: {ml_roi:+.1f}%
            </div>
            <div style='color: #666; font-size: 0.75em; margin-top: 2px;'>
                CONTINUATION mode | 99% win rate at &gt;60% confidence
            </div>
        </div>""", unsafe_allow_html=True)

        # ML FEATURES DEBUG - Expandable section showing all 32 features
        if ml_features:
            with st.expander("🔬 ML Features Debug (32 features fed to model)", expanded=False):
                render_ml_features_debug(ml_features)

    # Warnings
    if warnings:
        st.markdown("#### ⚠️ Signals & Warnings")
        for w in warnings:
            if w.startswith('✅') or w.startswith('🎯'):
                st.success(w)
            elif w.startswith('⏳'):
                st.warning(w)
            elif w.startswith('🚨') or w.startswith('🚫'):
                st.error(w)
            else:
                st.info(w)


def render_scanner_quality_summary(results: list):
    """
    Render a summary of scanner results by ML probability.
    ML-ONLY version - shows ML probability distribution and recommendations.
    """
    if not results:
        return

    # Count by ML probability range and recommendation
    ml_counts = {'STRONG': 0, 'GOOD': 0, 'WEAK': 0, 'SKIP': 0}  # >60%, 50-60%, 40-50%, <40%
    rec_counts = {'ENTER': 0, 'WAIT': 0, 'SKIP': 0}
    window_counts = {'OPEN': 0, 'CLOSING': 0, 'CLOSED': 0, 'NO_SWEEP': 0}
    active_sweeps = 0

    for result in results:
        if result.get('status') == 'SWEEP_ACTIVE':
            active_sweeps += 1
            ml_prob = result.get('ml_probability', 0) or 0
            rec = result.get('entry_recommendation', 'SKIP')
            window = result.get('entry_window', 'NO_SWEEP')

            # Categorize by ML probability
            if ml_prob >= 0.60:
                ml_counts['STRONG'] += 1
            elif ml_prob >= 0.50:
                ml_counts['GOOD'] += 1
            elif ml_prob >= 0.40:
                ml_counts['WEAK'] += 1
            else:
                ml_counts['SKIP'] += 1

            rec_counts[rec] = rec_counts.get(rec, 0) + 1
            window_counts[window] = window_counts.get(window, 0) + 1

    if active_sweeps == 0:
        st.info("No active sweeps in results")
        return

    st.markdown("#### 🤖 ML Quality Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🟢 ENTER", rec_counts.get('ENTER', 0), help="ML ≥50% - Take trade")
    with col2:
        st.metric("🟡 WAIT", rec_counts.get('WAIT', 0), help="ML 40-50% - Borderline")
    with col3:
        st.metric("🔴 SKIP", rec_counts.get('SKIP', 0), help="ML <40% - Avoid")
    with col4:
        fresh = window_counts.get('OPEN', 0) + window_counts.get('CLOSING', 0)
        st.metric("🔥 Fresh", fresh, help="Entry window OPEN or CLOSING")
    with col5:
        strong = ml_counts.get('STRONG', 0)
        st.metric("💎 Strong", strong, help="ML ≥60% - High confidence")

    # Show ML breakdown
    col6, col7, col8 = st.columns(3)
    with col6:
        st.caption(f"💎 ML≥60%: {ml_counts.get('STRONG', 0)} | 🟢 ML≥50%: {ml_counts.get('GOOD', 0)}")
    with col7:
        st.caption(f"🟡 ML 40-50%: {ml_counts.get('WEAK', 0)} | 🔴 ML<40%: {ml_counts.get('SKIP', 0)}")
    with col8:
        st.caption(f"⏱️ Open: {window_counts.get('OPEN', 0)} | ⏳ Closing: {window_counts.get('CLOSING', 0)}")


def render_scanner_results(results: list, show_quality_badge: bool = True):
    """
    Render scanner results with direction indicator and entry quality badge.

    ENHANCED: Now shows entry quality score, grade, and entry window status.
    """
    if not results:
        st.info("No results to display")
        return

    for result in results:
        status = result.get('status', 'MONITORING')
        symbol = result.get('symbol', '')
        current_price = result.get('current_price', 0)
        whale_pct = result.get('whale_pct', 50)
        whale_delta = result.get('whale_delta', 0)
        whale_conflict = result.get('whale_conflict', '')

        # Entry quality (NEW)
        entry_quality = result.get('entry_quality')
        entry_quality_score = result.get('entry_quality_score', 0)
        entry_quality_grade = result.get('entry_quality_grade', 'N/A')
        entry_window = result.get('entry_window', 'NO_SWEEP')

        # Get TRADE direction (CONTINUATION mode aware!)
        # trade_direction is the actual direction we trade, not the raw sweep direction
        sweep = result.get('sweep', {})
        direction = result.get('trade_direction', '')  # Use CONTINUATION-aware direction
        level = sweep.get('level_swept', 0) if sweep else 0
        candles_ago = sweep.get('candles_ago', 0) if sweep else 0

        # DEBUG: Log what we're getting
        raw_dir = sweep.get('direction', '') if sweep else ''
        print(f"[UI_DIR_DEBUG] {symbol}: trade_direction='{direction}', raw='{raw_dir}'")

        # Fallback to raw direction if trade_direction not present
        if not direction:
            if sweep:
                direction = sweep.get('direction', '')
                print(f"[UI_DIR_DEBUG] {symbol}: FALLBACK to raw direction '{direction}'")
            elif result.get('approaching'):
                direction = result['approaching'].get('direction_after_sweep', '')
                print(f"[UI_DIR_DEBUG] {symbol}: FALLBACK to approaching direction '{direction}'")

        # Status emoji
        status_emoji = {
            'SWEEP_ACTIVE': '🎯',
            'IMMINENT': '🔥',
            'APPROACHING': '⚡',
            'MONITORING': '👀'
        }.get(status, '👀')

        # Direction emoji - SHOW WARNING if whale conflict
        if whale_conflict:
            dir_emoji = '⚠️'
        else:
            dir_emoji = '🟢' if direction == 'LONG' else '🔴' if direction == 'SHORT' else ''

        # ML-ONLY: Get ML probability and recommendation
        ml_probability = 0
        ml_recommendation = 'SKIP'
        if entry_quality:
            ml_probability = entry_quality.get('ml_probability', 0) or 0
            ml_recommendation = entry_quality.get('recommendation', 'SKIP')

        # ML probability badge (replaces grade)
        if ml_probability >= 0.60:
            ml_emoji = '💎'  # Strong
            ml_color = 'green'
        elif ml_probability >= 0.50:
            ml_emoji = '🟢'  # Good
            ml_color = 'green'
        elif ml_probability >= 0.40:
            ml_emoji = '🟡'  # Weak
            ml_color = 'orange'
        else:
            ml_emoji = '🔴'  # Skip
            ml_color = 'red'

        quality_badge = f"{ml_emoji}{ml_probability:.0%}" if ml_probability > 0 else ""

        # Entry window indicator
        window_text = ""
        if entry_quality and entry_window != 'NO_SWEEP':
            candles_left = entry_quality.get('candles_until_close', 0)
            if entry_window == 'OPEN':
                window_text = f"⏱️{candles_left}c"
            elif entry_window == 'CLOSING':
                window_text = f"⏳{candles_left}c"
            elif entry_window == 'CLOSED':
                window_text = "⚫"  # Changed from ❌ since ML can override

        # Level text
        level_text = f"${level:,.2f}" if level > 0 else ""

        # Whale delta indicator
        if whale_delta and whale_delta > 3:
            whale_status = "↑"
        elif whale_delta and whale_delta < -3:
            whale_status = "↓"
        else:
            whale_status = ""

        # Use columns for clean layout (6 columns now for entry quality)
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1.8, 1.8, 1.5, 1.2, 1.2])

        with col1:
            st.write(status_emoji)
        with col2:
            st.write(f"**{symbol}**")
            if level_text:
                st.caption(level_text)
        with col3:
            st.write(f"${current_price:,.2f}")
            st.caption(status)
        with col4:
            st.write(f"{dir_emoji} {direction}")
            if whale_conflict:
                st.caption("⚠️ Conflict")
            elif candles_ago > 0:
                st.caption(f"{candles_ago}c ago")
        with col5:
            # ML-ONLY: Show ML probability and recommendation
            if quality_badge and show_quality_badge:
                st.write(quality_badge)
                # Show recommendation below ML %
                rec_emoji = {'ENTER': '✅', 'WAIT': '⏳', 'SKIP': '🚫'}.get(ml_recommendation, '❓')
                st.caption(f"{rec_emoji}{ml_recommendation}")
            elif window_text:
                st.write(window_text)
        with col6:
            # ETF/Stock: show flow phase instead of whale %
            _etf_flow = entry_quality.get('etf_flow') if entry_quality else None
            if _etf_flow:
                phase = _etf_flow.get('flow_phase', 'NEUTRAL')
                phase_map = {
                    'ACCUMULATING': ('🟢', 'ACC'),
                    'DISTRIBUTING': ('🟠', 'DIST'),
                    'EXTENDED': ('🔴', 'EXT'),
                    'NEUTRAL': ('⚪', 'NEU'),
                }
                p_emoji, p_label = phase_map.get(phase, ('⚪', 'NEU'))
                st.write(f"{p_emoji}{p_label}")
                action = _etf_flow.get('action', '')
                if action:
                    st.caption(action.replace('_', ' ').title())
            else:
                whale_display = f"🐋{whale_pct:.0f}%{whale_status}"
                st.write(whale_display)

        st.divider()


def render_liquidity_education():
    """Render educational content"""
    with st.expander("📚 How Liquidity Hunting Works"):
        st.markdown("""
        <div style='color: #fff;'>
            <div style='color: #00d4aa; font-weight: bold; margin-bottom: 10px;'>🎯 The Whale Playbook</div>
            <div style='color: #ccc; margin-bottom: 15px;'>
                Big players need liquidity (your stop losses) to fill their orders.
            </div>
            <div style='color: #3b82f6; font-weight: bold; margin-bottom: 8px;'>The Sweep Cycle:</div>
            <div style='color: #ccc; margin-left: 15px;'>
                1️⃣ Find where stops cluster (swing lows/highs)<br>
                2️⃣ Price pushed through to trigger stops<br>
                3️⃣ Whales fill orders against triggered stops<br>
                4️⃣ Price reverses sharply (the "sweep")<br>
                5️⃣ Real direction revealed
            </div>
            <div style='color: #ffcc00; font-weight: bold; margin-top: 15px;'>⭐ Equal Lows/Highs = Maximum Liquidity</div>
            <div style='color: #ccc;'>Multiple touches = more stops = higher probability sweep target.</div>
        </div>
        """, unsafe_allow_html=True)


def render_scanner_controls():
    """
    Render scanner control panel with filters including entry quality.
    Returns: dict with all filter settings
    """
    # ═══════════════════════════════════════════════════════════════════════
    # QUICK PRESET BUTTONS - Most common filter combinations
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("#### 🚀 Quick Filters")

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    with preset_col1:
        long_enter_only = st.button("🟢 LONG + ENTER", help="Show only LONG trades with ENTER recommendation", use_container_width=True)
    with preset_col2:
        short_enter_only = st.button("🔴 SHORT + ENTER", help="Show only SHORT trades with ENTER recommendation", use_container_width=True)
    with preset_col3:
        all_enter = st.button("🎯 All ENTER", help="Show all ENTER recommendations (LONG and SHORT)", use_container_width=True)
    with preset_col4:
        show_all = st.button("📊 Show All", help="Show all results without filtering", use_container_width=True)

    # Store preset in session state
    if long_enter_only:
        st.session_state['scanner_preset'] = 'LONG_ENTER'
    elif short_enter_only:
        st.session_state['scanner_preset'] = 'SHORT_ENTER'
    elif all_enter:
        st.session_state['scanner_preset'] = 'ALL_ENTER'
    elif show_all:
        st.session_state['scanner_preset'] = 'SHOW_ALL'

    # Get current preset (default to LONG_ENTER for user's preference)
    current_preset = st.session_state.get('scanner_preset', 'LONG_ENTER')

    # Show current preset
    preset_labels = {
        'LONG_ENTER': '🟢 LONG + ENTER (recommended)',
        'SHORT_ENTER': '🔴 SHORT + ENTER',
        'ALL_ENTER': '🎯 All ENTER trades',
        'SHOW_ALL': '📊 Showing all results'
    }
    st.info(f"**Active Filter:** {preset_labels.get(current_preset, 'Custom')}")

    st.markdown("---")
    st.markdown("#### ⚙️ Scanner Settings")

    col1, col2 = st.columns(2)

    with col1:
        scan_count = st.select_slider(
            "Symbols to Scan",
            options=[10, 20, 50, 100, 150, 200],
            value=50,
            help="More symbols = longer scan time (200 coins available)"
        )

    with col2:
        # Direction filter - preset-aware
        if current_preset == 'LONG_ENTER':
            st.success("🟢 **Direction:** LONG ONLY (from preset)")
            direction_filter = "LONG ONLY"
        elif current_preset == 'SHORT_ENTER':
            st.error("🔴 **Direction:** SHORT ONLY (from preset)")
            direction_filter = "SHORT ONLY"
        elif current_preset in ['ALL_ENTER', 'SHOW_ALL']:
            st.info("📊 **Direction:** ALL (from preset)")
            direction_filter = "ALL"
        else:
            direction_filter = st.selectbox(
                "Direction Filter",
                options=["ALL", "LONG ONLY", "SHORT ONLY"],
                index=0,
                help="CONTINUATION mode: LONG=sweep of highs, SHORT=sweep of lows"
            )

    # Status filter
    col3, col4 = st.columns(2)

    with col3:
        status_filter = st.multiselect(
            "Status Filter",
            options=["SWEEP_ACTIVE", "IMMINENT", "APPROACHING", "MONITORING"],
            default=["SWEEP_ACTIVE", "IMMINENT", "APPROACHING"],
            help="Show only selected statuses"
        )

    with col4:
        min_whale_pct = st.slider(
            "Min Whale %",
            min_value=40,
            max_value=80,
            value=50,
            help="Only show symbols with whale % above this"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY QUALITY FILTERS (NEW)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("#### 🎯 Entry Quality Filters")

    col5, col6 = st.columns(2)

    with col5:
        # Fresh Only toggle - most important filter!
        fresh_only = st.checkbox(
            "🔥 **Fresh Sweeps Only**",
            value=False,
            help="Only show sweeps with entry window OPEN or CLOSING (B grade or better, < 10 candles old)"
        )

    with col6:
        # Max candles age filter
        max_candles_ago = st.slider(
            "Max Sweep Age (candles)",
            min_value=3,
            max_value=50,
            value=10 if fresh_only else 25,
            help="Only show sweeps younger than this many candles"
        )

    # Entry window filter (ML can override CLOSED if probability is high)
    if fresh_only:
        entry_windows = ['OPEN', 'CLOSING']
        quality_grades = ['A', 'B', 'C', 'D']  # Keep for backwards compatibility but not used
        st.info("🔥 Fresh Only: OPEN & CLOSING windows")
    else:
        entry_windows = st.multiselect(
            "Entry Window",
            options=['OPEN', 'CLOSING', 'CLOSED', 'NO_SWEEP'],
            default=['OPEN', 'CLOSING', 'CLOSED'],  # Include CLOSED since ML can override
            help="OPEN=0-3c, CLOSING=4-10c, CLOSED=11+c (ML can override if high probability)"
        )
        quality_grades = ['A', 'B', 'C', 'D']  # Keep for backwards compatibility

    # ═══════════════════════════════════════════════════════════════════════
    # TRADE QUALITY FILTERS (REDESIGNED)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("#### 🎯 Trade Quality Filters")

    col9, col10 = st.columns(2)

    with col9:
        # RECOMMENDATION FILTER - preset-aware defaults
        # When preset is active, show what will actually be applied
        if current_preset in ['LONG_ENTER', 'SHORT_ENTER', 'ALL_ENTER']:
            st.info("📋 **Recommendation:** ENTER only (from preset)")
            recommendation_filter = ['ENTER']  # Preset overrides
        elif current_preset == 'SHOW_ALL':
            st.info("📋 **Recommendation:** All (from preset)")
            recommendation_filter = ['ENTER', 'WAIT', 'SKIP']
        else:
            recommendation_filter = st.multiselect(
                "📋 Recommendation",
                options=['ENTER', 'WAIT', 'SKIP'],
                default=['ENTER'],
                help="ENTER=Good setup, WAIT=Needs confirmation, SKIP=Avoid"
            )

    with col10:
        # Level type filter - prefer EQUAL/DOUBLE
        level_type_filter = st.multiselect(
            "📊 Level Types",
            options=['EQUAL', 'DOUBLE', 'SWING', 'ANY'],
            default=['EQUAL', 'DOUBLE', 'SWING'],  # Include all but show warning for SWING
            help="EQUAL=Best (3+ tests), DOUBLE=Good (2 tests), SWING=Weak (1 test)"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # ML QUALITY FILTERS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("#### 🤖 ML Quality Filter")

    col11, col12 = st.columns(2)

    with col11:
        min_ml_probability = st.slider(
            "Min ML Probability",
            min_value=0,
            max_value=70,
            value=50,
            step=5,
            format="%d%%",
            help="CONTINUATION mode: 50%=89% win, 60%=99% win"
        )

    with col12:
        ml_decisions = st.multiselect(
            "ML Decisions",
            options=['STRONG_YES', 'YES', 'MAYBE', 'NO', 'UNKNOWN'],
            default=['STRONG_YES', 'YES'],
            help="STRONG_YES (>60%=99% win), YES (50-60%=89% win)"
        )

    # Show expected performance based on filters
    st.markdown("---")
    if 'ENTER' in recommendation_filter and min_ml_probability >= 50:
        if min_ml_probability >= 60:
            st.success(f"🔥 **BEST QUALITY**: ENTER + ML≥60% = **99% win rate** (CONTINUATION)")
        else:
            st.success(f"✅ **HIGH QUALITY**: ENTER + ML≥50% = **89% win rate** (CONTINUATION)")
    elif 'ENTER' in recommendation_filter:
        st.info(f"📊 ENTER filter active - 90% base win rate with CONTINUATION")
    elif min_ml_probability >= 50:
        st.info(f"📊 ML filter active - 89%+ win rate with CONTINUATION")
    else:
        st.warning(f"⚠️ No filters - showing all sweeps (90% base win rate)")

    # ═══════════════════════════════════════════════════════════════════════
    # APPLY PRESET OVERRIDES
    # ═══════════════════════════════════════════════════════════════════════
    final_direction = direction_filter
    final_recommendation = recommendation_filter

    if current_preset == 'LONG_ENTER':
        final_direction = 'LONG ONLY'
        final_recommendation = ['ENTER']
    elif current_preset == 'SHORT_ENTER':
        final_direction = 'SHORT ONLY'
        final_recommendation = ['ENTER']
    elif current_preset == 'ALL_ENTER':
        final_direction = 'ALL'
        final_recommendation = ['ENTER']
    elif current_preset == 'SHOW_ALL':
        final_direction = 'ALL'
        final_recommendation = ['ENTER', 'WAIT', 'SKIP']

    return {
        'scan_count': scan_count,
        'direction_filter': final_direction,
        'status_filter': status_filter,
        'min_whale_pct': min_whale_pct,
        # Entry quality filters
        'fresh_only': fresh_only,
        'max_candles_ago': max_candles_ago,
        'quality_grades': quality_grades,
        'entry_windows': entry_windows,
        # Trade quality filters (preset-aware)
        'recommendation_filter': final_recommendation,
        'level_type_filter': level_type_filter,
        # ML quality filters
        'min_ml_probability': min_ml_probability / 100,  # Convert to 0-1
        'ml_decisions': ml_decisions,
        # Preset info
        'active_preset': current_preset
    }


def filter_scanner_results(results: list, filters: dict) -> list:
    """
    Apply filters to scanner results including entry quality and ML filters.

    Filters:
    - direction_filter: ALL, LONG ONLY, SHORT ONLY
    - status_filter: SWEEP_ACTIVE, IMMINENT, APPROACHING, MONITORING
    - min_whale_pct: Minimum whale %
    - fresh_only: Only B+ grade with OPEN/CLOSING window
    - max_candles_ago: Maximum sweep age in candles
    - quality_grades: List of acceptable grades [A, B, C, D]
    - entry_windows: List of acceptable windows [OPEN, CLOSING, CLOSED, NO_SWEEP]
    - min_ml_probability: Minimum ML probability (0-1)
    - ml_decisions: List of acceptable ML decisions [STRONG_YES, YES, MAYBE, NO]
    """
    filtered = []

    direction_filter = filters.get('direction_filter', 'ALL')
    status_filter = filters.get('status_filter', [])
    min_whale_pct = filters.get('min_whale_pct', 50)

    # Entry quality filters
    fresh_only = filters.get('fresh_only', False)
    max_candles_ago = filters.get('max_candles_ago', 50)
    quality_grades = filters.get('quality_grades', ['A', 'B', 'C', 'D'])
    entry_windows = filters.get('entry_windows', ['OPEN', 'CLOSING'])  # Default excludes CLOSED

    # Trade quality filters (NEW - critical!)
    recommendation_filter = filters.get('recommendation_filter', ['ENTER'])  # Default ENTER only
    level_type_filter = filters.get('level_type_filter', ['EQUAL', 'DOUBLE', 'SWING'])

    # ML quality filters
    min_ml_probability = filters.get('min_ml_probability', 0)
    ml_decisions = filters.get('ml_decisions', ['STRONG_YES', 'YES', 'MAYBE', 'NO', 'UNKNOWN'])

    for result in results:
        # Direction filter - USE trade_direction (CONTINUATION mode aware!)
        # trade_direction is the ACTUAL trade direction after applying mode flip
        # In CONTINUATION mode: Sweep LOW → SHORT, Sweep HIGH → LONG
        trade_direction = result.get('trade_direction')

        # Fallback to raw direction if trade_direction not present (backwards compat)
        if trade_direction is None:
            sweep = result.get('sweep', {})
            approaching = result.get('approaching', {})
            if sweep:
                trade_direction = sweep.get('direction')
            elif approaching:
                trade_direction = approaching.get('direction_after_sweep')

        if direction_filter == "LONG ONLY" and trade_direction != "LONG":
            continue
        if direction_filter == "SHORT ONLY" and trade_direction != "SHORT":
            continue

        # Status filter
        if status_filter and result.get('status') not in status_filter:
            continue

        # Whale % filter
        if result.get('whale_pct', 50) < min_whale_pct:
            continue

        # ═══════════════════════════════════════════════════════════════════
        # ENTRY QUALITY FILTERS - ML ONLY (simplified)
        # ═══════════════════════════════════════════════════════════════════

        # Get sweep age (candles_ago)
        candles_ago = 0
        if sweep:
            candles_ago = sweep.get('candles_ago', 0)

        # Max candles ago filter (for active sweeps)
        if result.get('status') == 'SWEEP_ACTIVE' and candles_ago > max_candles_ago:
            continue

        entry_window = result.get('entry_window', 'NO_SWEEP')

        # For active sweeps, apply window filter (ML can override CLOSED if high prob)
        if result.get('status') == 'SWEEP_ACTIVE':
            if fresh_only:
                # Fresh only = must have open or closing window
                if entry_window not in ['OPEN', 'CLOSING']:
                    continue
            else:
                # Entry window filter
                if entry_window not in entry_windows:
                    continue

        # ═══════════════════════════════════════════════════════════════════
        # RECOMMENDATION FILTER (Critical - filters SKIP/WAIT/ENTER)
        # ═══════════════════════════════════════════════════════════════════
        if result.get('status') == 'SWEEP_ACTIVE':
            recommendation = result.get('entry_recommendation', 'UNKNOWN')
            if recommendation_filter and recommendation not in recommendation_filter:
                continue

        # ═══════════════════════════════════════════════════════════════════
        # LEVEL TYPE FILTER (EQUAL > DOUBLE > SWING)
        # ═══════════════════════════════════════════════════════════════════
        if result.get('status') == 'SWEEP_ACTIVE' and 'ANY' not in level_type_filter:
            level_type = result.get('level_type', 'SWING')
            if level_type not in level_type_filter:
                continue

        # ═══════════════════════════════════════════════════════════════════
        # ML QUALITY FILTERS
        # ═══════════════════════════════════════════════════════════════════
        if result.get('status') == 'SWEEP_ACTIVE':
            ml_probability = result.get('ml_probability')
            ml_decision = result.get('ml_decision', 'UNKNOWN')

            # ML probability filter (if set > 0)
            if min_ml_probability > 0 and ml_probability is not None:
                if ml_probability < min_ml_probability:
                    continue

            # ML decision filter
            if ml_decisions and ml_decision not in ml_decisions:
                continue

        filtered.append(result)

    return filtered


def render_ml_training_panel():
    """Render ML training and stats panel"""
    st.markdown("#### 🤖 ML Model Training")
    
    # Try to import ML module
    try:
        from .liquidity_hunter_ml import get_training_stats, get_predictor
    except ImportError:
        try:
            from liquidity_hunter_ml import get_training_stats, get_predictor
        except ImportError:
            st.warning("ML module not available")
            return
    
    # Get training stats
    stats = get_training_stats()
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col2:
        st.metric("Wins", stats.get('wins', 0), delta=None)
    with col3:
        st.metric("Losses", stats.get('losses', 0), delta=None)
    with col4:
        win_rate = stats.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}" if win_rate > 0 else "N/A")
    
    # Training status
    predictor = get_predictor()
    model_status = "✅ Model Loaded" if predictor.model is not None else "⚠️ No Model (Rule-based)"
    
    st.markdown(f"""<div style='background: #1a1a2e; padding: 15px; border-radius: 8px; margin: 10px 0;'>
        <div style='color: #fff;'>Status: {model_status}</div>
        <div style='color: #888; font-size: 0.9em;'>Pending trades: {stats.get('pending', 0)}</div>
    </div>""", unsafe_allow_html=True)
    
    # Training button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Train Model", use_container_width=True):
            if stats.get('wins', 0) + stats.get('losses', 0) < 50:
                st.warning(f"Need at least 50 completed trades. Have {stats.get('wins', 0) + stats.get('losses', 0)}")
            else:
                with st.spinner("Training model..."):
                    metrics = predictor.train()
                    if 'error' in metrics:
                        st.error(metrics['error'])
                    else:
                        st.success(f"Model trained! F1: {metrics.get('f1_score', 0):.3f}")
                        st.json(metrics)
    
    with col2:
        if st.button("📊 View Feature Importance", use_container_width=True):
            if predictor.model is not None:
                try:
                    importance = dict(zip(
                        predictor.feature_names[:len(predictor.model.feature_importances_)],
                        predictor.model.feature_importances_
                    ))
                    # Sort by importance
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                    
                    st.markdown("**Top Features:**")
                    for feat, imp in list(importance.items())[:10]:
                        bar_width = int(imp * 300)
                        st.markdown(f"""<div style='display: flex; align-items: center; margin: 3px 0;'>
                            <span style='color: #888; width: 150px;'>{feat}</span>
                            <div style='background: #3b82f6; height: 12px; width: {bar_width}px; border-radius: 3px;'></div>
                            <span style='color: #fff; margin-left: 10px;'>{imp:.3f}</span>
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("No model trained yet")


def render_ml_prediction_badge(ml_prediction: dict):
    """Render a compact ML prediction badge"""
    if not ml_prediction:
        return ""
    
    quality = ml_prediction.get('prediction', 'UNKNOWN')
    probability = ml_prediction.get('probability', 0)
    
    colors = {
        'HIGH_QUALITY': '#00ff88',
        'MEDIUM_QUALITY': '#ffcc00',
        'LOW_QUALITY': '#ff6b6b'
    }
    icons = {
        'HIGH_QUALITY': '🔥',
        'MEDIUM_QUALITY': '⚡',
        'LOW_QUALITY': '⚠️'
    }
    
    color = colors.get(quality, '#888')
    icon = icons.get(quality, '❓')
    
    return f"""<span style='background: {color}22; padding: 3px 8px; border-radius: 10px;
        border: 1px solid {color}; font-size: 0.85em;'>
        <span style='color: {color};'>{icon} {probability:.0%}</span>
    </span>"""


# ═══════════════════════════════════════════════════════════════════════════════
# ETF MONEY FLOW UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_etf_flow_badge(etf_flow: dict):
    """Render a color-coded flow phase badge."""
    if not etf_flow:
        return

    phase = etf_flow.get('flow_phase', 'NEUTRAL')
    flow_score = etf_flow.get('flow_score', 0)
    action = etf_flow.get('action', 'HOLD')

    phase_config = {
        'ACCUMULATING': {'color': '#00C853', 'icon': '💰', 'label': 'MONEY FLOWING IN'},
        'NEUTRAL': {'color': '#888888', 'icon': '⚖️', 'label': 'NEUTRAL FLOW'},
        'DISTRIBUTING': {'color': '#FF9800', 'icon': '💸', 'label': 'MONEY FLOWING OUT'},
        'EXTENDED': {'color': '#FF1744', 'icon': '🔴', 'label': 'PRICE EXTENDED'},
    }

    config = phase_config.get(phase, phase_config['NEUTRAL'])

    action_labels = {
        'ACCUMULATE_MORE': '📈 ACCUMULATE MORE',
        'HOLD': '⏸️ HOLD',
        'TRIM_5_10': '✂️ TRIM 5-10%',
        'TRIM_15_20': '✂️ TRIM 15-20%',
    }
    action_label = action_labels.get(action, action)

    st.markdown(f"""
    <div style='background: {config["color"]}15; border: 2px solid {config["color"]}; border-radius: 12px; padding: 12px 16px; margin: 8px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 1.3em;'>{config["icon"]}</span>
                <span style='color: {config["color"]}; font-weight: bold; font-size: 1.1em; margin-left: 8px;'>{config["label"]}</span>
                <span style='color: #aaa; margin-left: 12px;'>Flow Score: {flow_score:+.0f}</span>
            </div>
            <div style='background: {config["color"]}30; padding: 6px 14px; border-radius: 8px;'>
                <span style='color: {config["color"]}; font-weight: bold;'>{action_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_etf_flow_panel(etf_flow: dict):
    """Render full ETF money flow analysis panel."""
    if not etf_flow:
        return

    st.markdown("### 💰 Money Flow Analysis")

    # Flow badge
    render_etf_flow_badge(etf_flow)

    # Component breakdown
    cols = st.columns(4)

    with cols[0]:
        obv = "Rising ↑" if etf_flow.get('obv_trend', 0) else "Falling ↓"
        obv_color = "🟢" if etf_flow.get('obv_trend', 0) else "🔴"
        st.metric("OBV Trend", f"{obv_color} {obv}")

    with cols[1]:
        mfi = etf_flow.get('mfi_value', 50)
        mfi_label = "Inflow" if mfi > 55 else ("Outflow" if mfi < 45 else "Neutral")
        st.metric("MFI", f"{mfi:.0f} ({mfi_label})")

    with cols[2]:
        cmf = etf_flow.get('cmf_value', 0)
        cmf_label = "Buying" if cmf > 0.05 else ("Selling" if cmf < -0.05 else "Flat")
        st.metric("CMF", f"{cmf:.3f} ({cmf_label})")

    with cols[3]:
        ext50 = etf_flow.get('price_extension_ema50', 0)
        ext200 = etf_flow.get('price_extension_ema200', 0)
        st.metric("Price vs EMA200", f"{ext200:+.1f}%", delta=f"vs EMA50: {ext50:+.1f}%")

    # Zones
    zone_cols = st.columns(2)
    with zone_cols[0]:
        if etf_flow.get('in_accumulation_zone'):
            st.success("📊 In Accumulation Zone (ranging + OBV rising + higher lows)")
        elif etf_flow.get('in_distribution_zone'):
            st.warning("📊 In Distribution Zone (ranging + OBV falling + lower highs)")

    with zone_cols[1]:
        inst = etf_flow.get('institutional_score', 0)
        if inst != 0:
            label = "Bullish" if inst > 0 else "Bearish"
            st.metric("Institutional Signal", f"{label} ({inst:+.0f})")
        else:
            st.caption("Institutional data: not available")