"""
UI Renderers
============
Functions that take a Result and render HTML.
"""

from core.models import Result, Trade, Position, Direction


def render_header(r: Result) -> str:
    """Main signal header"""
    
    if r.trade == Trade.LONG:
        bg = "linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%)"
        border = "#00ff88"
        action_color = "#00ff88"
    elif r.trade == Trade.SHORT:
        bg = "linear-gradient(135deg, #2e1a1a 0%, #1a1a2e 100%)"
        border = "#ff6b6b"
        action_color = "#ff6b6b"
    else:
        bg = "#1a1a2e"
        border = "#ffcc00"
        action_color = "#ffcc00"
    
    htf_badge = ""
    if r.htf.timeframe:
        htf_color = "#00ff88" if r.htf.is_bullish else "#ff6b6b" if r.htf.is_bearish else "#888"
        htf_badge = f"<span style='background: #252540; color: {htf_color}; padding: 2px 8px; border-radius: 4px; margin-left: 10px; font-size: 0.8em;'>HTF: {r.htf.structure}</span>"
    
    exp_badge = ""
    if r.explosion.score >= 30:
        exp_color = "#00ff88" if r.explosion.ready else "#ffcc00" if r.explosion.score >= 50 else "#888"
        exp_badge = f"<span style='color: {exp_color}; margin-left: 10px;'>💥 Explosion: {r.explosion.score}/100</span>"
    
    return f"""
    <div style='background: {bg}; border: 2px solid {border}; border-radius: 12px; padding: 20px; margin-bottom: 15px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 1.8em; font-weight: bold; color: {action_color};'>
                    {r.action_emoji} {r.symbol}
                </span>
                <span style='color: #888; margin-left: 15px;'>{r.timeframe} | {r.market_type}</span>
                {htf_badge}
            </div>
            <div style='text-align: right;'>
                <span style='font-size: 2.2em; font-weight: bold; color: {action_color};'>{r.total_score}/100</span>
            </div>
        </div>
        <div style='margin-top: 10px; color: #aaa;'>
            {r.summary}{exp_badge}
        </div>
        <div style='margin-top: 10px; display: flex; gap: 20px; color: #666; font-size: 0.9em;'>
            <span>Direction: {r.direction_score}/40</span>
            <span>Squeeze: {r.squeeze_score}/30</span>
            <span>Entry: {r.timing_score}/30</span>
        </div>
    </div>
    """


def render_ml_rules_box(r: Result) -> str:
    """ML vs Rules comparison"""
    if not r.ml:
        return ""
    
    ml_color = r.ml.color
    rules_color = r.direction.color
    
    aligned = r.ml_rules_aligned
    align_badge = "✅ ALIGNED" if aligned else "⚠️ CONFLICT"
    align_color = "#00ff88" if aligned else "#ff6b6b"
    
    factors_html = ""
    if r.ml.top_factors:
        factors_html = "<div style='color: #666; font-size: 0.75em; margin-top: 5px;'>Top factors: " + ", ".join(r.ml.top_factors[:3]) + "</div>"
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 15px 0;'>
        <div style='text-align: center; margin-bottom: 10px;'>
            <span style='background: {"#1a2e1a" if aligned else "#2e1a1a"}; color: {align_color}; padding: 4px 12px; border-radius: 4px; font-weight: bold;'>
                {align_badge}
            </span>
            <span style='color: #888; margin-left: 10px;'>ML vs Rules</span>
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
            <div style='background: #252540; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid {ml_color};'>
                <div style='color: #888; font-size: 0.85em;'>🤖 ML Says</div>
                <div style='color: {ml_color}; font-size: 1.5em; font-weight: bold; margin: 8px 0;'>{r.ml.direction}</div>
                <div style='color: #aaa; font-size: 0.85em;'>{r.ml.confidence:.0f}% confidence</div>
                {factors_html}
            </div>
            <div style='background: #252540; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid {rules_color};'>
                <div style='color: #888; font-size: 0.85em;'>📊 Rules Say</div>
                <div style='color: {rules_color}; font-size: 1.5em; font-weight: bold; margin: 8px 0;'>{r.direction_label}</div>
                <div style='color: #aaa; font-size: 0.85em;'>{r.confidence.value} confidence</div>
            </div>
        </div>
    </div>
    """


def render_explosion_box(r: Result) -> str:
    """Explosion readiness"""
    if r.explosion.score < 30:
        return ""
    
    exp = r.explosion
    exp_color = "#00ff88" if exp.ready else "#ffcc00" if exp.score >= 50 else "#888"
    state_emoji = "🚀" if exp.state == "IGNITION" else "🎯" if exp.state == "LIQUIDITY_CLEAR" else "⚡" if exp.state == "COMPRESSION" else "📊"
    
    dir_html = ""
    if exp.direction:
        dir_color = "#00ff88" if exp.direction == "LONG" else "#ff6b6b"
        dir_html = f"<span style='color: {dir_color}; margin-left: 10px;'>Direction: {exp.direction} {'📈' if exp.direction == 'LONG' else '📉'}</span>"
    
    signals_html = ""
    if exp.signals:
        signals_html = "<div style='color: #888; font-size: 0.8em; margin-top: 8px;'>" + " | ".join(exp.signals[:4]) + "</div>"
    
    return f"""
    <div style='background: linear-gradient(135deg, #1a2a1a 0%, #1a1a2e 100%); padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid {exp_color};'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: {exp_color}; font-weight: bold;'>{state_emoji} {exp.state}</span>
            <span style='color: {exp_color}; font-size: 1.2em; font-weight: bold;'>Explosion: {exp.score}/100</span>
        </div>
        <div style='color: #aaa; font-size: 0.85em; margin-top: 5px;'>
            BB Squeeze: {exp.squeeze_pct:.0f}% | 
            {"✅ Entry Valid" if exp.entry_valid else "⏳ Wait for trigger"}
            {dir_html}
        </div>
        {signals_html}
    </div>
    """


def render_layers(r: Result) -> str:
    """3 layer boxes"""
    
    dir_color = r.direction.color
    dir_html = f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {dir_color}; min-height: 160px;'>
        <div style='color: #888; font-size: 0.85em;'>📈 LAYER 1: Direction</div>
        <div style='color: {dir_color}; font-size: 1.4em; font-weight: bold; margin: 8px 0;'>{r.direction_label}</div>
        <div style='color: #aaa; font-size: 0.85em;'>{r.confidence.value} confidence</div>
        <div style='color: {dir_color}; font-size: 1.8em; font-weight: bold; margin-top: 8px;'>{r.direction_score}/40</div>
        <div style='color: #666; font-size: 0.75em; margin-top: 8px; border-top: 1px solid #333; padding-top: 8px;'>
            Whales {r.whale.whale_pct:.0f}% long
        </div>
    </div>
    """
    
    sq_color = "#00ff88" if r.squeeze_label == "HIGH" else "#ffcc00" if r.squeeze_label == "MEDIUM" else "#ff6b6b" if r.squeeze_label == "CONFLICT" else "#888"
    sq_html = f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {sq_color}; min-height: 160px;'>
        <div style='color: #888; font-size: 0.85em;'>🔥 LAYER 2: Squeeze</div>
        <div style='color: {sq_color}; font-size: 1.4em; font-weight: bold; margin: 8px 0;'>{r.squeeze_label}</div>
        <div style='color: #aaa; font-size: 0.85em;'>W:{r.whale.whale_pct:.0f}% vs R:{r.whale.retail_pct:.0f}%</div>
        <div style='color: {sq_color}; font-size: 1.8em; font-weight: bold; margin-top: 8px;'>{r.squeeze_score}/30</div>
        <div style='color: #666; font-size: 0.75em; margin-top: 8px; border-top: 1px solid #333; padding-top: 8px;'>
            Divergence: {r.whale.divergence:+.0f}%
        </div>
    </div>
    """
    
    pos_color = r.position.color
    timing_color = "#00ff88" if r.timing_score >= 20 else "#ffcc00" if r.timing_score >= 12 else "#ff6b6b"
    entry_label = "NOW" if r.timing_score >= 20 else "SOON" if r.timing_score >= 12 else "WAIT"
    
    entry_html = f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {timing_color}; min-height: 160px;'>
        <div style='color: #888; font-size: 0.85em;'>⏰ LAYER 3: Entry</div>
        <div style='color: {timing_color}; font-size: 1.4em; font-weight: bold; margin: 8px 0;'>{entry_label}</div>
        <div style='color: #aaa; font-size: 0.85em;'>TA: {r.ta_score} | Pos: <span style='color: {pos_color};'>{r.position.value}</span></div>
        <div style='color: {timing_color}; font-size: 1.8em; font-weight: bold; margin-top: 8px;'>{r.timing_score}/30</div>
        <div style='color: #666; font-size: 0.75em; margin-top: 8px; border-top: 1px solid #333; padding-top: 8px;'>
            Position: {r.position_pct:.0f}% in range
        </div>
    </div>
    """
    
    return f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;'>
        {dir_html}
        {sq_html}
        {entry_html}
    </div>
    """


def render_smc_levels(r: Result) -> str:
    """SMC Order Blocks and FVG"""
    smc = r.smc
    
    ob_html = ""
    if smc.has_bullish_ob or smc.has_bearish_ob:
        ob_items = []
        if smc.has_bullish_ob:
            at_badge = " <span style='color: #00ff88;'>📍 AT OB</span>" if smc.at_bullish_ob else " <span style='color: #ffcc00;'>📍 NEAR</span>" if smc.near_bullish_ob else ""
            ob_items.append(f"<div style='color: #00ff88;'>🟢 Bullish OB: ${smc.bullish_ob_bottom:,.4f} - ${smc.bullish_ob_top:,.4f}{at_badge}</div>")
        if smc.has_bearish_ob:
            at_badge = " <span style='color: #ff6b6b;'>📍 AT OB</span>" if smc.at_bearish_ob else " <span style='color: #ffcc00;'>📍 NEAR</span>" if smc.near_bearish_ob else ""
            ob_items.append(f"<div style='color: #ff6b6b;'>🔴 Bearish OB: ${smc.bearish_ob_bottom:,.4f} - ${smc.bearish_ob_top:,.4f}{at_badge}</div>")
        ob_html = "<div style='margin: 10px 0;'>" + "".join(ob_items) + "</div>"
    
    fvg_html = ""
    fvg = smc.fvg
    if fvg.bullish_fvg or fvg.bearish_fvg:
        fvg_items = []
        if fvg.bullish_fvg:
            at_badge = " <span style='color: #00ff88;'>📍 IN FVG</span>" if fvg.at_bullish_fvg else ""
            fvg_items.append(f"<div style='color: #00d4aa;'>⬜ Bullish FVG: ${fvg.bullish_fvg_bottom:,.4f} - ${fvg.bullish_fvg_top:,.4f}{at_badge}</div>")
        if fvg.bearish_fvg:
            at_badge = " <span style='color: #ff6b6b;'>📍 IN FVG</span>" if fvg.at_bearish_fvg else ""
            fvg_items.append(f"<div style='color: #ff9966;'>⬜ Bearish FVG: ${fvg.bearish_fvg_bottom:,.4f} - ${fvg.bearish_fvg_top:,.4f}{at_badge}</div>")
        fvg_html = "<div style='margin: 10px 0;'>" + "".join(fvg_items) + "</div>"
    
    struct_color = "#00ff88" if "bullish" in smc.structure.lower() else "#ff6b6b" if "bearish" in smc.structure.lower() else "#888"
    
    if not ob_html and not fvg_html:
        return ""
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 15px 0;'>
        <div style='color: #888; font-weight: bold; margin-bottom: 10px;'>📊 Smart Money Concepts</div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
            <span style='color: #888;'>Structure: <span style='color: {struct_color};'>{smc.structure}</span></span>
            <span style='color: #888;'>Swing: ${smc.swing_low:,.4f} - ${smc.swing_high:,.4f}</span>
        </div>
        {ob_html}
        {fvg_html}
    </div>
    """


def render_trade_setup(r: Result) -> str:
    """Trade setup with levels"""
    if r.trade == Trade.WAIT:
        return ""
    
    setup = r.setup
    tp_color = "#00ff88"
    sl_color = "#ff6b6b"
    entry_color = "#00d4ff"
    
    def rr_badge(rr: float) -> str:
        if rr >= 2: return f"<span style='color: #00ff88;'>({rr:.1f}R) 🔥</span>"
        elif rr >= 1: return f"<span style='color: #00ff88;'>({rr:.1f}R)</span>"
        elif rr >= 0.5: return f"<span style='color: #ffcc00;'>({rr:.1f}R)</span>"
        return f"<span style='color: #ff6b6b;'>({rr:.1f}R)</span>"
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 15px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <span style='color: #888; font-weight: bold;'>📊 Trade Setup</span>
            <span style='color: {r.action_color}; font-weight: bold;'>{r.trade.value}</span>
        </div>
        <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;'>
            <div style='background: #252540; padding: 12px; border-radius: 6px; text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>Entry</div>
                <div style='color: {entry_color}; font-weight: bold; font-size: 0.95em;'>${setup.entry:,.4f}</div>
            </div>
            <div style='background: #252540; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid {sl_color};'>
                <div style='color: #888; font-size: 0.75em;'>Stop Loss</div>
                <div style='color: {sl_color}; font-weight: bold; font-size: 0.95em;'>${setup.stop_loss:,.4f}</div>
                <div style='color: #666; font-size: 0.7em;'>Risk: {setup.risk_pct:.2f}%</div>
            </div>
            <div style='background: #252540; padding: 12px; border-radius: 6px; text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>TP1 {rr_badge(setup.rr1)}</div>
                <div style='color: {tp_color}; font-weight: bold; font-size: 0.95em;'>${setup.tp1:,.4f}</div>
            </div>
            <div style='background: #252540; padding: 12px; border-radius: 6px; text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>TP2 {rr_badge(setup.rr2)}</div>
                <div style='color: {tp_color}; font-weight: bold; font-size: 0.95em;'>${setup.tp2:,.4f}</div>
            </div>
            <div style='background: #252540; padding: 12px; border-radius: 6px; text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>TP3 {rr_badge(setup.rr3)}</div>
                <div style='color: {tp_color}; font-weight: bold; font-size: 0.95em;'>${setup.tp3:,.4f}</div>
            </div>
        </div>
    </div>
    """


def render_combined_learning(r: Result) -> str:
    """Combined learning stories"""
    learning = r.learning
    
    stories_html = ""
    for title, content in learning.stories:
        stories_html += f"""
        <div style='background: #252540; padding: 10px; border-radius: 6px; margin: 5px 0;'>
            <div style='color: #888; font-size: 0.8em;'>{title}</div>
            <div style='color: #fff; font-size: 0.9em;'>{content}</div>
        </div>
        """
    
    concl_color = "#00ff88" if "LONG" in learning.conclusion_action else "#ff6b6b" if "SHORT" in learning.conclusion_action else "#ffcc00"
    
    conflicts_html = ""
    if learning.conflicts:
        conflicts_html = "<div style='background: #2e2a1a; padding: 10px; border-radius: 6px; margin-top: 10px; border: 1px solid #ffcc00;'>"
        conflicts_html += "<div style='color: #ffcc00; font-weight: bold;'>⚠️ Conflicts</div>"
        for c in learning.conflicts:
            conflicts_html += f"<div style='color: #aaa; font-size: 0.85em;'>• {c}</div>"
        conflicts_html += "</div>"
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 15px 0;'>
        <div style='color: #888; font-weight: bold; margin-bottom: 10px;'>🎓 Combined Learning</div>
        {stories_html}
        <div style='background: linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%); padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid {concl_color};'>
            <div style='color: {concl_color}; font-weight: bold; font-size: 1.1em;'>{learning.conclusion}</div>
            <div style='color: #aaa; font-size: 0.9em; margin-top: 5px;'>Action: {learning.conclusion_action}</div>
        </div>
        {conflicts_html}
    </div>
    """


def render_raw_data(r: Result) -> str:
    """Raw data boxes"""
    return f"""
    <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 15px;'>
        <div style='background: #1a1a2e; padding: 12px; border-radius: 8px; text-align: center;'>
            <div style='color: #888; font-size: 0.75em;'>OI 24h</div>
            <div style='color: {"#00ff88" if r.whale.oi_change_24h > 0 else "#ff6b6b" if r.whale.oi_change_24h < 0 else "#888"}; font-size: 1.2em; font-weight: bold;'>{r.whale.oi_change_24h:+.1f}%</div>
        </div>
        <div style='background: #1a1a2e; padding: 12px; border-radius: 8px; text-align: center;'>
            <div style='color: #888; font-size: 0.75em;'>Price 24h</div>
            <div style='color: {"#00ff88" if r.whale.price_change_24h > 0 else "#ff6b6b" if r.whale.price_change_24h < 0 else "#888"}; font-size: 1.2em; font-weight: bold;'>{r.whale.price_change_24h:+.1f}%</div>
        </div>
        <div style='background: #1a1a2e; padding: 12px; border-radius: 8px; text-align: center;'>
            <div style='color: #888; font-size: 0.75em;'>🐋 Whales</div>
            <div style='color: #00d4ff; font-size: 1.2em; font-weight: bold;'>{r.whale.whale_pct:.0f}%</div>
        </div>
        <div style='background: #1a1a2e; padding: 12px; border-radius: 8px; text-align: center;'>
            <div style='color: #888; font-size: 0.75em;'>👥 Retail</div>
            <div style='color: #ff9966; font-size: 1.2em; font-weight: bold;'>{r.whale.retail_pct:.0f}%</div>
        </div>
        <div style='background: #1a1a2e; padding: 12px; border-radius: 8px; text-align: center;'>
            <div style='color: #888; font-size: 0.75em;'>TA Score</div>
            <div style='color: #888; font-size: 1.2em; font-weight: bold;'>{r.ta_score}</div>
        </div>
    </div>
    """


def render_warnings(r: Result) -> str:
    """Warnings box"""
    if not r.warnings:
        return ""
    
    warnings_html = "".join([f"<div style='color: #ffcc00; margin: 4px 0;'>⚠️ {w}</div>" for w in r.warnings])
    
    return f"""
    <div style='background: #2e2a1a; border-radius: 8px; padding: 12px; margin: 10px 0; border: 1px solid #ffcc00;'>
        <div style='color: #ffcc00; font-weight: bold; margin-bottom: 5px;'>⚠️ Warnings</div>
        {warnings_html}
    </div>
    """


def render_scanner_row(r: Result) -> str:
    """Scanner result row"""
    action_color = r.action_color
    grade = "🔥" if r.total_score >= 80 else "✅" if r.total_score >= 65 else "⚡" if r.total_score >= 50 else "⏳"
    
    exp_badge = ""
    if r.explosion.score >= 50:
        exp_color = "#00ff88" if r.explosion.ready else "#ffcc00"
        exp_badge = f"<span style='color: {exp_color}; margin-left: 8px;'>💥{r.explosion.score}</span>"
    
    ml_badge = ""
    if r.ml:
        ml_badge = f"<span style='color: {r.ml.color}; margin-left: 8px;'>🤖{r.ml.direction}</span>"
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 4px solid {action_color};'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-weight: bold; color: white;'>{grade} {r.symbol}</span>
                <span style='color: #888; margin-left: 10px;'>{r.timeframe}</span>
                {exp_badge}{ml_badge}
            </div>
            <div>
                <span style='color: {action_color}; font-weight: bold;'>{r.action_emoji} {r.action}</span>
                <span style='color: {action_color}; margin-left: 10px; font-size: 1.2em; font-weight: bold;'>{r.total_score}/100</span>
            </div>
        </div>
        <div style='color: #aaa; font-size: 0.85em; margin-top: 5px;'>
            {r.summary} | W:{r.whale.whale_pct:.0f}% R:{r.whale.retail_pct:.0f}%
        </div>
    </div>
    """


def render_trade_monitor_card(r: Result) -> str:
    """Trade monitor card"""
    setup = r.setup
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid {r.action_color};'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 1.3em; font-weight: bold; color: {r.action_color};'>{r.action_emoji} {r.symbol}</span>
                <span style='color: #888; margin-left: 10px;'>{r.timeframe}</span>
            </div>
            <span style='color: {r.action_color}; font-weight: bold;'>{r.trade.value}</span>
        </div>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px;'>
            <div style='text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>Entry</div>
                <div style='color: #00d4ff; font-weight: bold;'>${setup.entry:,.4f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>Stop Loss</div>
                <div style='color: #ff6b6b; font-weight: bold;'>${setup.stop_loss:,.4f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>TP1 ({setup.rr1:.1f}R)</div>
                <div style='color: #00ff88; font-weight: bold;'>${setup.tp1:,.4f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #888; font-size: 0.75em;'>Score</div>
                <div style='color: {r.action_color}; font-weight: bold;'>{r.total_score}/100</div>
            </div>
        </div>
    </div>
    """
