"""
═══════════════════════════════════════════════════════════════════════════════
ENHANCED TRADE MONITOR SECTION
═══════════════════════════════════════════════════════════════════════════════

This file contains the enhanced Trade Monitor code to be integrated into app.py
It adds:
1. Trading Mode + Timeframe display for each trade
2. Whale/Institutional data integration
3. Clear OI + Price interpretation
4. Better organization and visual feedback

INTEGRATION INSTRUCTIONS:
1. Add import at top of app.py:
   from core.whale_institutional import get_institutional_analysis, get_whale_summary, format_institutional_html

2. Replace the Trade Monitor section (starting at "elif app_mode == '📈 Trade Monitor':") 
   with this enhanced version
"""

# ═══════════════════════════════════════════════════════════════════════════════
# START OF ENHANCED TRADE MONITOR CODE
# Copy from here and replace the existing Trade Monitor section
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_MONITOR_CODE = '''
elif app_mode == "📈 Trade Monitor":
    st.markdown("## 📈 Trade Monitor")
    st.markdown("<p style='color: #888;'>Live tracking with whale data, alerts & recommendations</p>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # IMPORT WHALE MODULE (add at top of app.py imports)
    # from core.whale_institutional import get_institutional_analysis, get_whale_summary, format_institutional_html
    # ═══════════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚠️ CLOUD PERSISTENCE WARNING
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style='background: #2a2a1a; border-left: 4px solid #ffcc00; padding: 10px 15px; 
                border-radius: 8px; margin-bottom: 15px;'>
        <span style='color: #ffcc00;'>⚠️ <strong>Cloud Warning:</strong></span>
        <span style='color: #ccc;'> On Streamlit Cloud, trades reset on refresh. 
        Use <strong>Export</strong> to save your trades to your computer, 
        then <strong>Import</strong> to restore them.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔧 MANAGEMENT BUTTONS + EXPORT/IMPORT
    # ═══════════════════════════════════════════════════════════════════
    
    mgmt_col1, mgmt_col2, mgmt_col3, mgmt_col4 = st.columns([1, 1, 1, 2])
    
    with mgmt_col1:
        if st.button("🔄 Reload", help="Force reload trades from saved file"):
            st.session_state.active_trades = get_active_trades()
            st.success(f"Reloaded {len(st.session_state.active_trades)} trades")
            st.rerun()
    
    with mgmt_col2:
        if st.session_state.active_trades:
            export_data = export_trades_json()
            st.download_button(
                label="📥 Export",
                data=export_data,
                file_name=f"investoriq_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Download your trades to your computer"
            )
        else:
            st.button("📥 Export", disabled=True, help="No trades to export")
    
    with mgmt_col3:
        if st.button("🗑️ Clear All", help="Remove all active trades"):
            if st.session_state.active_trades:
                all_trades = load_trade_history()
                closed_only = [t for t in all_trades if t.get('status') == 'closed']
                save_trade_history(closed_only)
                st.session_state.active_trades = []
                st.success("Cleared all active trades")
                st.rerun()
    
    with mgmt_col4:
        # Show whale data toggle
        show_whale_data = st.checkbox("🐋 Show Whale Data", value=True, 
                                       help="Fetch institutional positioning data")
    
    # Import section (keep existing code)
    with st.expander("📤 **Import Trades from File**", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload your exported trades JSON file",
            type=['json'],
            help="Select a previously exported trades file"
        )
        
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                imported_trades = json.loads(content)
                st.success(f"Found {len(imported_trades)} trades in file")
                
                col_import1, col_import2 = st.columns(2)
                with col_import1:
                    if st.button("✅ Import & Replace All", type="primary"):
                        if import_trades_json(content):
                            st.session_state.active_trades = get_active_trades()
                            st.success(f"Imported {len(st.session_state.active_trades)} active trades!")
                            st.rerun()
                with col_import2:
                    if st.button("➕ Add to Existing"):
                        existing = load_trade_history()
                        existing_symbols = {t.get('symbol') for t in existing if t.get('status') == 'active'}
                        new_trades = [t for t in imported_trades 
                                     if t.get('symbol') not in existing_symbols or t.get('status') != 'active']
                        combined = existing + new_trades
                        save_trade_history(combined)
                        st.session_state.active_trades = get_active_trades()
                        st.success(f"Added {len(new_trades)} new trades!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 📊 OI + PRICE INTERPRETATION GUIDE
    # ═══════════════════════════════════════════════════════════════════
    
    with st.expander("📊 **Understanding Whale Data (OI + Price)**", expanded=False):
        st.markdown("""
        ### 🐋 How to Read Institutional Data
        
        #### 📊 Open Interest (OI) + Price Relationship
        
        | OI Change | Price Change | What It Means | Action |
        |-----------|--------------|---------------|--------|
        | 📈 OI UP | 📈 Price UP | **New LONGS entering** | ✅ Bullish continuation - Follow trend |
        | 📈 OI UP | 📉 Price DOWN | **New SHORTS entering** | ❌ Bearish continuation - Follow trend |
        | 📉 OI DOWN | 📈 Price UP | **Short covering** (forced buying) | ⚠️ Weak rally - May reverse |
        | 📉 OI DOWN | 📉 Price DOWN | **Long liquidation** (forced selling) | ⚠️ Weak dump - May reverse |
        
        #### 💰 Funding Rate (Contrarian!)
        
        | Funding | What It Means | Action |
        |---------|---------------|--------|
        | **> 0.1%** | Longs overleveraged | 🔴 Contrarian SHORT - Dump coming |
        | **< -0.1%** | Shorts overleveraged | 🟢 Contrarian LONG - Pump coming |
        | **-0.01% to 0.01%** | Balanced | Follow other signals |
        
        #### 🐑 Retail vs 🐋 Top Traders
        
        | Situation | Action |
        |-----------|--------|
        | Retail 70% LONG + Whales SHORT | 🔴 **FADE RETAIL** - Go short |
        | Retail 70% SHORT + Whales LONG | 🟢 **FADE RETAIL** - Go long |
        | Both aligned | Follow the direction |
        
        **KEY INSIGHT**: Retail is usually WRONG at extremes. Top Traders (whales) have better information.
        """)
    
    if not st.session_state.active_trades:
        st.info("No active trades. Go to Scanner to find setups!")
    else:
        # ═══════════════════════════════════════════════════════════════════
        # 📊 SUMMARY BY TRADING MODE
        # ═══════════════════════════════════════════════════════════════════
        
        # Group trades by mode
        trades_by_mode = {}
        for trade in st.session_state.active_trades:
            mode = trade.get('mode_name') or trade.get('mode', 'Unknown')
            if mode not in trades_by_mode:
                trades_by_mode[mode] = []
            trades_by_mode[mode].append(trade)
        
        # Show mode breakdown
        st.markdown("### 📋 Trades by Mode")
        mode_cols = st.columns(len(trades_by_mode) if len(trades_by_mode) <= 5 else 5)
        
        mode_colors = {
            'Scalp': '#ff9500',
            'scalp': '#ff9500',
            'Day Trade': '#00d4ff',
            'day_trade': '#00d4ff',
            'Swing': '#00d4aa',
            'swing': '#00d4aa',
            'Position': '#9d4edd',
            'position': '#9d4edd',
            'Investment': '#ffd700',
            'investment': '#ffd700'
        }
        
        for i, (mode, trades) in enumerate(trades_by_mode.items()):
            color = mode_colors.get(mode, '#888888')
            with mode_cols[i % len(mode_cols)]:
                st.markdown(f"""
                <div style='background: {color}22; border: 1px solid {color}; border-radius: 8px; 
                            padding: 10px; text-align: center;'>
                    <div style='color: {color}; font-weight: bold; font-size: 1.1em;'>{mode}</div>
                    <div style='color: #fff; font-size: 1.5em;'>{len(trades)}</div>
                    <div style='color: #888; font-size: 0.8em;'>trades</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # SUMMARY METRICS
        # ═══════════════════════════════════════════════════════════════════
        
        total_pnl = 0
        winning = 0
        losing = 0
        in_danger = 0
        
        for trade in st.session_state.active_trades:
            price = get_current_price(trade['symbol'])
            if price > 0:
                entry = trade['entry']
                pnl = ((price - entry) / entry) * 100 if trade['direction'] == 'LONG' else ((entry - price) / entry) * 100
                total_pnl += pnl
                
                if pnl > 0.1:
                    winning += 1
                elif pnl < -0.1:
                    losing += 1
                
                # Check danger zone
                sl = trade['stop_loss']
                if trade['direction'] == 'LONG':
                    dist_to_sl = ((price - sl) / price) * 100
                else:
                    dist_to_sl = ((sl - price) / price) * 100
                
                if dist_to_sl < 1.5:
                    in_danger += 1
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📊 Trades", len(st.session_state.active_trades))
        m2.metric("🟢 Winning", winning)
        m3.metric("🔴 Losing", losing)
        m4.metric("🚨 In Danger", in_danger, delta=None if in_danger == 0 else "CLOSE!")
        m5.metric("💰 Total P&L", f"{total_pnl:+.2f}%")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # INDIVIDUAL TRADES WITH MODE/TIMEFRAME + WHALE DATA
        # ═══════════════════════════════════════════════════════════════════
        
        for i, trade in enumerate(st.session_state.active_trades):
            price = get_current_price(trade['symbol'])
            
            if price > 0:
                entry = trade['entry']
                pnl = ((price - entry) / entry) * 100 if trade['direction'] == 'LONG' else ((entry - price) / entry) * 100
                
                # Get mode and timeframe
                mode_name = trade.get('mode_name') or trade.get('mode', 'Unknown')
                timeframe = trade.get('timeframe', 'N/A')
                grade = trade.get('grade', 'N/A')
                
                # Mode color
                mode_color = mode_colors.get(mode_name, '#888888')
                
                # TP/SL status
                tp1_hit = price >= trade['tp1'] if trade['direction'] == 'LONG' else price <= trade['tp1']
                tp2_hit = price >= trade['tp2'] if trade['direction'] == 'LONG' else price <= trade['tp2']
                tp3_hit = price >= trade['tp3'] if trade['direction'] == 'LONG' else price <= trade['tp3']
                sl_hit = price <= trade['stop_loss'] if trade['direction'] == 'LONG' else price >= trade['stop_loss']
                
                # Distance calculations
                dist_to_tp1 = ((trade['tp1'] - price) / price * 100) if trade['direction'] == 'LONG' else ((price - trade['tp1']) / price * 100)
                dist_to_sl = ((price - trade['stop_loss']) / price * 100) if trade['direction'] == 'LONG' else ((trade['stop_loss'] - price) / price * 100)
                
                # Status
                if sl_hit:
                    status = "🔴 STOPPED OUT"
                    emoji = "🔴"
                    status_color = "#ff4444"
                elif tp3_hit:
                    status = "🏆 TP3 HIT"
                    emoji = "🏆"
                    status_color = "#00ff88"
                elif tp2_hit:
                    status = "🎯 TP2 HIT"
                    emoji = "🎯"
                    status_color = "#00d4aa"
                elif tp1_hit:
                    status = "✅ TP1 HIT"
                    emoji = "✅"
                    status_color = "#00d4aa"
                elif pnl > 5:
                    status = "🟢 STRONG PROFIT"
                    emoji = "🟢"
                    status_color = "#00d4aa"
                elif pnl > 0:
                    status = "🟢 IN PROFIT"
                    emoji = "🟢"
                    status_color = "#00d4aa"
                elif dist_to_sl < 1.5:
                    status = "🚨 DANGER ZONE"
                    emoji = "🚨"
                    status_color = "#ff0000"
                elif pnl > -3:
                    status = "🟡 ACTIVE"
                    emoji = "🟡"
                    status_color = "#ffcc00"
                else:
                    status = "🟠 DRAWDOWN"
                    emoji = "🟠"
                    status_color = "#ff9500"
                
                # Expander header with MODE + TIMEFRAME
                header = f"{emoji} {trade['symbol']} | {pnl:+.2f}% | {status}"
                
                should_expand = bool(pnl < -5 or tp1_hit or dist_to_sl < 2)
                with st.expander(header, expanded=should_expand):
                    
                    # ═══════════════════════════════════════════════════════════
                    # MODE + TIMEFRAME BADGE (NEW!)
                    # ═══════════════════════════════════════════════════════════
                    
                    st.markdown(f"""
                    <div style='display: flex; gap: 10px; margin-bottom: 15px;'>
                        <span style='background: {mode_color}; color: #000; padding: 5px 15px; 
                                     border-radius: 20px; font-weight: bold; font-size: 0.9em;'>
                            {mode_name}
                        </span>
                        <span style='background: #333; color: #fff; padding: 5px 15px; 
                                     border-radius: 20px; font-size: 0.9em;'>
                            ⏱️ {timeframe}
                        </span>
                        <span style='background: #1a1a2e; color: #00d4ff; padding: 5px 15px; 
                                     border-radius: 20px; font-size: 0.9em;'>
                            Grade: {grade}
                        </span>
                        <span style='background: #1a1a2e; color: #888; padding: 5px 15px; 
                                     border-radius: 20px; font-size: 0.85em;'>
                            📅 {trade.get('created_at', 'N/A')[:10]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🐋 WHALE DATA SECTION (NEW!)
                    # ═══════════════════════════════════════════════════════════
                    
                    if show_whale_data:
                        try:
                            from core.whale_institutional import get_whale_summary, format_institutional_html, get_institutional_analysis
                            
                            with st.spinner(f"Fetching whale data for {trade['symbol']}..."):
                                whale_data = get_institutional_analysis(trade['symbol'])
                            
                            if whale_data.get('open_interest', {}).get('available', False) or whale_data.get('signals'):
                                # Show whale summary
                                verdict = whale_data.get('verdict', 'UNKNOWN')
                                confidence = whale_data.get('confidence', 0)
                                
                                verdict_colors = {
                                    'BULLISH': '#00d4aa',
                                    'BEARISH': '#ff4444',
                                    'NEUTRAL': '#888888',
                                    'MIXED': '#ffcc00'
                                }
                                v_color = verdict_colors.get(verdict, '#888888')
                                
                                # Whale summary bar
                                st.markdown(f"""
                                <div style='background: #1a1a2e; border-radius: 10px; padding: 12px; margin: 10px 0;
                                            border-left: 4px solid {v_color};'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <span style='color: #00d4ff; font-weight: bold;'>🐋 Whale Verdict</span>
                                        <span style='background: {v_color}33; color: {v_color}; padding: 4px 12px; 
                                                     border-radius: 15px; font-weight: bold;'>
                                            {verdict} ({confidence}%)
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Key metrics
                                whale_cols = st.columns(4)
                                
                                with whale_cols[0]:
                                    oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                                    oi_color = "#00d4aa" if oi_change > 3 else "#ff4444" if oi_change < -3 else "#888"
                                    st.markdown(f"""
                                    <div style='background: #252540; padding: 8px; border-radius: 6px; text-align: center;'>
                                        <div style='color: #888; font-size: 0.8em;'>OI Change</div>
                                        <div style='color: {oi_color}; font-size: 1.2em; font-weight: bold;'>{oi_change:+.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with whale_cols[1]:
                                    funding = whale_data.get('funding', {}).get('rate_pct', 0)
                                    fund_color = "#ff4444" if abs(funding) > 0.05 else "#888"
                                    st.markdown(f"""
                                    <div style='background: #252540; padding: 8px; border-radius: 6px; text-align: center;'>
                                        <div style='color: #888; font-size: 0.8em;'>Funding</div>
                                        <div style='color: {fund_color}; font-size: 1.2em; font-weight: bold;'>{funding:.4f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with whale_cols[2]:
                                    whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                                    whale_color = "#00d4aa" if whale_long > 55 else "#ff4444" if whale_long < 45 else "#888"
                                    st.markdown(f"""
                                    <div style='background: #252540; padding: 8px; border-radius: 6px; text-align: center;'>
                                        <div style='color: #888; font-size: 0.8em;'>🐋 Whales Long</div>
                                        <div style='color: {whale_color}; font-size: 1.2em; font-weight: bold;'>{whale_long:.0f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with whale_cols[3]:
                                    retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                                    # Fade retail - opposite color
                                    retail_color = "#ff4444" if retail_long > 65 else "#00d4aa" if retail_long < 35 else "#888"
                                    st.markdown(f"""
                                    <div style='background: #252540; padding: 8px; border-radius: 6px; text-align: center;'>
                                        <div style='color: #888; font-size: 0.8em;'>🐑 Retail Long</div>
                                        <div style='color: {retail_color}; font-size: 1.2em; font-weight: bold;'>{retail_long:.0f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # OI + Price Interpretation
                                oi_interp = whale_data.get('oi_interpretation', {})
                                if oi_interp:
                                    interp_color = "#00d4aa" if oi_interp.get('emoji') == '🟢' else "#ff4444" if oi_interp.get('emoji') == '🔴' else "#ffcc00"
                                    st.markdown(f"""
                                    <div style='background: #1a2a2a; border-left: 3px solid {interp_color}; 
                                                padding: 10px; border-radius: 6px; margin: 8px 0;'>
                                        <div style='color: {interp_color}; font-weight: bold;'>
                                            {oi_interp.get('emoji', '📊')} {oi_interp.get('interpretation', 'Analyzing...')}
                                        </div>
                                        <div style='color: #888; font-size: 0.9em; margin-top: 5px;'>
                                            Action: {oi_interp.get('action', 'Wait for clearer signals')}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Signals
                                if whale_data.get('signals'):
                                    with st.expander("📋 All Whale Signals", expanded=False):
                                        for sig in whale_data['signals']:
                                            st.write(sig)
                            else:
                                st.info("🐋 Whale data unavailable for this pair (Futures API may be restricted)")
                        
                        except Exception as e:
                            st.caption(f"⚠️ Could not fetch whale data: {str(e)[:50]}")
                    
                    st.markdown("---")
                    
                    # ═══════════════════════════════════════════════════════════
                    # POSITION DETAILS
                    # ═══════════════════════════════════════════════════════════
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown("### 💵 Position")
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 8px;'>
                            <div><span style='color: #888;'>Current:</span> <strong style='color: #00d4ff;'>{fmt_price(price)}</strong></div>
                            <div><span style='color: #888;'>Entry:</span> <strong>{fmt_price(trade['entry'])}</strong></div>
                            <div><span style='color: #888;'>P&L:</span> <strong style='color: {status_color};'>{pnl:+.2f}%</strong></div>
                            <div><span style='color: #888;'>Direction:</span> <strong>{trade['direction']}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown("### 🎯 Targets")
                        st.markdown(f"""
                        <div style='background: #1a2a1a; padding: 10px; border-radius: 8px;'>
                            <div>{'✅' if tp1_hit else '⏳'} <strong>TP1:</strong> {fmt_price(trade['tp1'])} 
                                <span style='color: #00d4aa;'>(+{calc_roi(trade['tp1'], trade['entry']):.1f}%)</span>
                            </div>
                            <div>{'✅' if tp2_hit else '⏳'} <strong>TP2:</strong> {fmt_price(trade['tp2'])} 
                                <span style='color: #00d4aa;'>(+{calc_roi(trade['tp2'], trade['entry']):.1f}%)</span>
                            </div>
                            <div>{'✅' if tp3_hit else '⏳'} <strong>TP3:</strong> {fmt_price(trade['tp3'])} 
                                <span style='color: #00d4aa;'>(+{calc_roi(trade['tp3'], trade['entry']):.1f}%)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c3:
                        st.markdown("### ⚠️ Risk")
                        sl_color = "#ff0000" if dist_to_sl < 1.5 else "#ff4444" if dist_to_sl < 3 else "#ffcc00" if dist_to_sl < 5 else "#00d4aa"
                        sl_status = "🚨 CRITICAL!" if dist_to_sl < 1.5 else "⚠️ Close" if dist_to_sl < 3 else "🟡 Watch" if dist_to_sl < 5 else "🟢 Safe"
                        
                        st.markdown(f"""
                        <div style='background: #2a1a1a; padding: 10px; border-radius: 8px; border: 1px solid {sl_color};'>
                            <div><strong>Stop Loss:</strong> {fmt_price(trade['stop_loss'])}</div>
                            <div style='color: {sl_color}; font-size: 1.2em; font-weight: bold; margin-top: 5px;'>
                                {dist_to_sl:.2f}% to SL
                            </div>
                            <div style='color: {sl_color}; margin-top: 5px;'>{sl_status}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if dist_to_sl < 2:
                            st.markdown(f"""
                            <div style='background: #ff444422; border: 1px solid #ff4444; border-radius: 6px; 
                                        padding: 8px; margin-top: 8px; text-align: center;'>
                                <span style='color: #ff4444; font-weight: bold;'>⚡ Close on Binance NOW!</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # RECOMMENDATION + ACTIONS
                    # ═══════════════════════════════════════════════════════════
                    
                    st.markdown("---")
                    
                    rec_col, action_col = st.columns([2, 1])
                    
                    with rec_col:
                        st.markdown("### 💡 Recommendation")
                        
                        if sl_hit:
                            st.error("**Trade closed.** Stop loss was hit.")
                        elif tp3_hit:
                            st.success("**🏆 WINNER!** Full target reached. Close remaining position!")
                        elif tp2_hit:
                            st.success("**Take Profit:** TP2 hit! Close 50-75% of remaining. Trail stop on rest.")
                        elif tp1_hit:
                            st.success("**Partial Profit:** TP1 hit! Take 33% profit. Move stop to breakeven.")
                        elif dist_to_sl < 1.5:
                            st.error("**🚨 DANGER!** Close this trade NOW on Binance to limit losses!")
                        elif pnl > 5:
                            st.info("**Strong profit.** Consider trailing your stop loss to lock in gains.")
                        elif pnl > 0:
                            st.info("**In profit.** Hold with original stop. Let the trade develop.")
                        else:
                            st.warning(f"**Drawdown:** {pnl:.1f}% | SL is {dist_to_sl:.1f}% away. Monitor closely.")
                    
                    with action_col:
                        st.markdown("### 🔧 Actions")
                        
                        if st.button(f"🗑️ Remove Trade", key=f"remove_{i}_{trade['symbol']}"):
                            delete_trade_by_symbol(trade['symbol'])
                            st.session_state.active_trades = get_active_trades()
                            st.success(f"Removed {trade['symbol']}")
                            st.rerun()
'''
