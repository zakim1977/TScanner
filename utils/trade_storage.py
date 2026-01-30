"""
Trade Storage Module
JSON-based persistence for trade history

NOTE: On Streamlit Cloud, files don't persist between sessions.
Use Export/Import feature to save your trades to your computer.
"""

import json
import os
import base64
from datetime import datetime
from typing import List, Dict, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS - Relative to script location for consistency
# ═══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_FILE = os.path.join(_SCRIPT_DIR, "trade_history.json")
SETTINGS_FILE = os.path.join(_SCRIPT_DIR, "user_settings.json")

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS PERSISTENCE (API Keys, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def load_user_settings() -> Dict:
    """Load user settings (API keys, preferences) from file"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}


def save_user_settings(settings: Dict) -> bool:
    """Save user settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def get_saved_api_key(key_name: str) -> str:
    """Get a saved API key by name"""
    settings = load_user_settings()
    return settings.get(key_name, '')


def save_api_key(key_name: str, key_value: str) -> bool:
    """Save an API key"""
    settings = load_user_settings()
    settings[key_name] = key_value
    return save_user_settings(settings)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD/SAVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_trade_history() -> List[Dict]:
    """Load trade history from JSON file"""
    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading trades: {e}")
        return []


def save_trade_history(trades: List[Dict]) -> bool:
    """Save trade history to JSON file"""
    try:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving trades: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT/IMPORT FUNCTIONS (For Streamlit Cloud persistence)
# ═══════════════════════════════════════════════════════════════════════════════

def export_trades_json() -> str:
    """Export all trades as JSON string for download"""
    trades = load_trade_history()
    return json.dumps(trades, indent=2, default=str)


def import_trades_json(json_string: str) -> bool:
    """Import trades from JSON string"""
    try:
        trades = json.loads(json_string)
        if isinstance(trades, list):
            return save_trade_history(trades)
        return False
    except Exception as e:
        print(f"Error importing trades: {e}")
        return False


def get_download_link(data: str, filename: str) -> str:
    """Generate a download link for data"""
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:application/json;base64,{b64}" download="{filename}" style="background: #00d4ff; color: black; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: bold;">📥 Download {filename}</a>'


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def add_trade(trade: Dict) -> bool:
    """Add a new trade to history"""
    trades = load_trade_history()
    
    # Check for duplicate symbol (prevent adding same coin twice)
    if any(t.get('symbol') == trade.get('symbol') and t.get('status') == 'active' for t in trades):
        return False  # Already exists
    
    # Add metadata if not present
    if 'id' not in trade:
        trade['id'] = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trade.get('symbol', 'UNK')}"
    if 'created_at' not in trade:
        trade['created_at'] = datetime.now().isoformat()
    if 'status' not in trade:
        trade['status'] = 'active'
    
    trades.append(trade)
    return save_trade_history(trades)


def update_trade(trade_id: str, updates: Dict) -> bool:
    """Update an existing trade"""
    trades = load_trade_history()
    
    for i, trade in enumerate(trades):
        if trade.get('id') == trade_id:
            trades[i].update(updates)
            trades[i]['updated_at'] = datetime.now().isoformat()
            return save_trade_history(trades)
    
    return False


def close_trade(trade_id: str, exit_price: float, outcome: str) -> bool:
    """Close a trade with exit price and outcome"""
    trades = load_trade_history()
    
    for i, trade in enumerate(trades):
        if trade.get('id') == trade_id:
            entry = trade.get('entry', 0)
            
            # Calculate P&L
            if trade.get('direction', 'LONG') == 'LONG':
                pnl_pct = ((exit_price - entry) / entry) * 100 if entry > 0 else 0
            else:
                pnl_pct = ((entry - exit_price) / entry) * 100 if entry > 0 else 0
            
            trades[i]['status'] = 'closed'
            trades[i]['exit_price'] = exit_price
            trades[i]['exit_time'] = datetime.now().isoformat()
            trades[i]['outcome'] = outcome
            trades[i]['pnl_pct'] = pnl_pct
            
            return save_trade_history(trades)
    
    return False


def delete_trade(trade_id: str) -> bool:
    """Delete a trade from history"""
    trades = load_trade_history()
    
    trades = [t for t in trades if t.get('id') != trade_id]
    return save_trade_history(trades)


def delete_trade_by_symbol(symbol: str) -> bool:
    """Delete an active trade by symbol"""
    trades = load_trade_history()
    
    # Only remove active trades with this symbol
    trades = [t for t in trades if not (t.get('symbol') == symbol and t.get('status') == 'active')]
    return save_trade_history(trades)


def sync_active_trades(active_trades: List[Dict]) -> bool:
    """
    Sync session state active trades back to file.
    Preserves all CLOSED trades (WIN, LOSS, PARTIAL, closed).
    """
    trades = load_trade_history()
    
    # Get all closed trades from history (keep them for Performance tracking)
    closed = [t for t in trades if t.get('closed', False) or t.get('status') in ['closed', 'WIN', 'LOSS', 'PARTIAL']]
    
    # Filter active_trades to only include NON-closed trades
    truly_active = [t for t in active_trades if not t.get('closed', False) and t.get('status') not in ['closed', 'WIN', 'LOSS', 'PARTIAL']]
    
    # Combine: closed history + currently active
    all_trades = closed + truly_active
    
    return save_trade_history(all_trades)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_trades() -> List[Dict]:
    """Get all active trades (not closed)"""
    trades = load_trade_history()
    # A trade is active if it's not closed and status is not a closed status
    return [t for t in trades if not t.get('closed', False) and t.get('status') not in ['closed', 'WIN', 'LOSS', 'PARTIAL']]


def get_closed_trades() -> List[Dict]:
    """Get all closed trades (TP3 hit, SL hit, or manually closed)"""
    trades = load_trade_history()
    # A trade is closed if: closed=True, or status in [closed, WIN, LOSS, PARTIAL]
    return [t for t in trades if t.get('closed', False) or t.get('status') in ['closed', 'WIN', 'LOSS', 'PARTIAL']]


def get_trade_by_symbol(symbol: str) -> Optional[Dict]:
    """Get active trade for a symbol"""
    trades = get_active_trades()
    for trade in trades:
        if trade.get('symbol') == symbol:
            return trade
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_statistics(closed_trades: List[Dict] = None) -> Dict:
    """
    Calculate professional trading statistics.
    
    Key Metrics:
    - TP1 Hit Rate: Measures trade SELECTION quality
    - Win Rate: Traditional win/loss
    - Avg R-Multiple: Risk-adjusted returns
    - Profit Factor: Gross profit / Gross loss
    
    Args:
        closed_trades: Optional list of closed trades. If not provided, loads from file.
    """
    if closed_trades is None:
        closed_trades = get_closed_trades()
    
    if not closed_trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'partials': 0,
            'win_rate': 0,
            'tp1_hit_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_r_multiple': 0,
            'profit_factor': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'expectancy': 0
        }
    
    # Get P&L from either final_pnl (auto-close) or pnl_pct (manual close)
    def get_pnl(t):
        return t.get('final_pnl', t.get('blended_pnl', t.get('pnl_pct', 0)))
    
    def get_r_multiple(t):
        return t.get('r_multiple', 0)
    
    # Categorize trades by outcome
    full_wins = [t for t in closed_trades if t.get('outcome_type') == 'FULL_WIN' or t.get('status') == 'WIN']
    partial_wins = [t for t in closed_trades if t.get('status') == 'PARTIAL' or t.get('outcome_type') in ['PARTIAL_WIN_TP2', 'PARTIAL_WIN_TP1', 'BREAKEVEN']]
    losses = [t for t in closed_trades if t.get('status') == 'LOSS' or t.get('outcome_type') == 'FULL_LOSS']
    
    # TP1 hit rate - measures trade SELECTION quality
    tp1_hits = [t for t in closed_trades if t.get('tp1_hit', False)]
    tp1_hit_rate = (len(tp1_hits) / len(closed_trades)) * 100 if closed_trades else 0
    
    # Traditional metrics
    wins = full_wins + partial_wins  # Any trade that hit TP1 is considered successful
    
    # Calculate AVERAGE P&L (not sum!) - this is per-trade average return
    pnl_values = [get_pnl(t) for t in closed_trades]
    total_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0  # AVERAGE!
    
    win_pnls = [get_pnl(t) for t in wins if get_pnl(t) > 0]
    loss_pnls = [abs(get_pnl(t)) for t in losses]
    
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    
    # R-Multiple analysis
    r_multiples = [get_r_multiple(t) for t in closed_trades if get_r_multiple(t) != 0]
    avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0
    
    # Profit Factor
    total_wins = sum(win_pnls)
    total_losses = sum(loss_pnls)
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
    
    # Expectancy (expected value per trade)
    win_rate_decimal = len(wins) / len(closed_trades) if closed_trades else 0
    loss_rate_decimal = 1 - win_rate_decimal
    expectancy = (win_rate_decimal * avg_win) - (loss_rate_decimal * avg_loss)
    
    all_pnls = [get_pnl(t) for t in closed_trades]
    
    return {
        'total_trades': len(closed_trades),
        'wins': len(full_wins),
        'partials': len(partial_wins),
        'losses': len(losses),
        'win_rate': (len(wins) / len(closed_trades)) * 100 if closed_trades else 0,
        'tp1_hit_rate': tp1_hit_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_r_multiple': avg_r_multiple,
        'profit_factor': profit_factor,
        'best_trade': max(all_pnls) if all_pnls else 0,
        'worst_trade': min(all_pnls) if all_pnls else 0,
        'expectancy': expectancy
    }


def export_trades_csv(filepath: str) -> bool:
    """Export trades to CSV file"""
    try:
        import csv
        
        trades = load_trade_history()
        
        if not trades:
            return False
        
        # Get all keys
        keys = set()
        for trade in trades:
            keys.update(trade.keys())
        
        keys = sorted(list(keys))
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trades)
        
        return True
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ML PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_ml_performance_stats() -> Dict:
    """
    Calculate performance statistics grouped by ML alignment.
    
    Returns dict with:
    - aligned: Stats when ML and Rules agreed
    - conflict: Stats when ML and Rules disagreed
    - ml_wait_ignored: Stats when ML said WAIT but trade was taken
    - duration_analysis: Time to TP analysis
    """
    trades = load_trade_history()
    
    stats = {
        'aligned': {'count': 0, 'tp1_hits': 0, 'sl_hits': 0, 'total_duration_mins': 0, 'duration_count': 0},
        'conflict': {'count': 0, 'tp1_hits': 0, 'sl_hits': 0, 'total_duration_mins': 0, 'duration_count': 0},
        'ml_wait_ignored': {'count': 0, 'tp1_hits': 0, 'sl_hits': 0, 'total_duration_mins': 0, 'duration_count': 0},
        'no_ml_data': {'count': 0, 'tp1_hits': 0, 'sl_hits': 0},
        'slow_trades': [],  # Trades that took much longer than expected
        'by_liquidity': {},  # Stats by liquidity tier
    }
    
    for trade in trades:
        # Skip non-closed or very recent trades
        if trade.get('status') != 'closed' and not trade.get('tp1_hit') and not trade.get('sl_hit'):
            continue
        
        ml_direction = trade.get('ml_direction')
        ml_confidence = trade.get('ml_confidence', 0)
        has_conflict = trade.get('has_ml_conflict', False)
        ml_wait = ml_direction in ['WAIT', 'NEUTRAL'] if ml_direction else False
        
        tp1_hit = trade.get('tp1_hit', False)
        sl_hit = trade.get('sl_hit', False)
        
        # Calculate duration to TP1 if available
        duration_mins = None
        if tp1_hit and trade.get('tp1_hit_time') and trade.get('added_at'):
            try:
                added = datetime.fromisoformat(trade['added_at'].replace('Z', '+00:00'))
                tp1_time = datetime.fromisoformat(trade['tp1_hit_time'].replace('Z', '+00:00'))
                duration_mins = (tp1_time - added).total_seconds() / 60
            except:
                pass
        
        # Categorize trade
        if ml_direction is None:
            category = 'no_ml_data'
        elif ml_wait and (tp1_hit or sl_hit):
            category = 'ml_wait_ignored'  # ML said wait but trade was taken
        elif has_conflict:
            category = 'conflict'
        else:
            category = 'aligned'
        
        # Update stats
        stats[category]['count'] += 1
        if tp1_hit:
            stats[category]['tp1_hits'] += 1
        if sl_hit:
            stats[category]['sl_hits'] += 1
        
        if duration_mins is not None and category != 'no_ml_data':
            stats[category]['total_duration_mins'] += duration_mins
            stats[category]['duration_count'] += 1
            
            # Check if trade was slow (compare to expected ETA)
            expected_candles = trade.get('ml_eta_candles')
            timeframe = trade.get('timeframe', '15m')
            
            # Convert candles to expected minutes
            tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            tf_mins = tf_minutes.get(timeframe, 15)
            expected_mins = (expected_candles or 10) * tf_mins
            
            if duration_mins > expected_mins * 3:  # 3x slower than expected
                stats['slow_trades'].append({
                    'symbol': trade.get('symbol'),
                    'expected_mins': expected_mins,
                    'actual_mins': duration_mins,
                    'ratio': duration_mins / expected_mins if expected_mins > 0 else 0,
                    'ml_direction': ml_direction,
                    'ml_confidence': ml_confidence,
                    'liquidity_tier': trade.get('liquidity_tier'),
                })
        
        # Track by liquidity tier
        liq_tier = trade.get('liquidity_tier', 'UNKNOWN')
        if liq_tier not in stats['by_liquidity']:
            stats['by_liquidity'][liq_tier] = {'count': 0, 'tp1_hits': 0, 'sl_hits': 0}
        stats['by_liquidity'][liq_tier]['count'] += 1
        if tp1_hit:
            stats['by_liquidity'][liq_tier]['tp1_hits'] += 1
        if sl_hit:
            stats['by_liquidity'][liq_tier]['sl_hits'] += 1
    
    # Calculate win rates and averages
    for cat in ['aligned', 'conflict', 'ml_wait_ignored', 'no_ml_data']:
        cat_stats = stats[cat]
        total = cat_stats['count']
        if total > 0:
            cat_stats['win_rate'] = (cat_stats['tp1_hits'] / total) * 100
            cat_stats['loss_rate'] = (cat_stats['sl_hits'] / total) * 100
        else:
            cat_stats['win_rate'] = 0
            cat_stats['loss_rate'] = 0
        
        if cat_stats.get('duration_count', 0) > 0:
            cat_stats['avg_duration_mins'] = cat_stats['total_duration_mins'] / cat_stats['duration_count']
        else:
            cat_stats['avg_duration_mins'] = 0
    
    # Calculate liquidity win rates
    for tier, tier_stats in stats['by_liquidity'].items():
        total = tier_stats['count']
        if total > 0:
            tier_stats['win_rate'] = (tier_stats['tp1_hits'] / total) * 100
        else:
            tier_stats['win_rate'] = 0
    
    return stats


def get_trade_duration_warning(trade: Dict) -> Optional[str]:
    """
    Check if a trade is taking longer than expected.
    Returns warning message if trade is slow, None otherwise.
    """
    if not trade.get('added_at'):
        return None
    
    try:
        added = datetime.fromisoformat(trade['added_at'].replace('Z', '+00:00'))
        now = datetime.now(added.tzinfo) if added.tzinfo else datetime.now()
        duration_mins = (now - added).total_seconds() / 60
        
        # Get expected duration
        expected_candles = trade.get('ml_eta_candles', 10)
        timeframe = trade.get('timeframe', '15m')
        tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        tf_mins = tf_minutes.get(timeframe, 15)
        expected_mins = expected_candles * tf_mins
        
        if duration_mins > expected_mins * 2:
            ratio = duration_mins / expected_mins
            hours = duration_mins / 60
            expected_hours = expected_mins / 60
            
            if trade.get('has_ml_conflict'):
                return f"⚠️ {hours:.1f}hrs elapsed (expected {expected_hours:.1f}hrs) - ML conflict may explain delay"
            else:
                return f"⏰ {hours:.1f}hrs elapsed (expected {expected_hours:.1f}hrs) - {ratio:.1f}x slower"
    except:
        pass
    
    return None

