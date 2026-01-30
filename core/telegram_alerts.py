"""
Telegram Alert System for InvestorIQ
=====================================

Sends alerts when watchlist conditions are met:
- VWAP bounce (FRESH) aligned with direction
- ML + Rules aligned
- Move position EARLY
- Explosion score high

Setup:
1. Create a bot via @BotFather on Telegram
2. Get your chat ID via @userinfobot
3. Enter bot token and chat ID in settings
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

# Alert settings file
ALERT_SETTINGS_FILE = "alert_settings.json"


def load_alert_settings() -> dict:
    """Load alert settings from file"""
    defaults = {
        'telegram_enabled': False,
        'telegram_bot_token': '',
        'telegram_chat_id': '',
        'refresh_interval_minutes': 15,
        'alert_conditions': {
            'require_vwap_fresh': True,
            'require_ml_aligned': True,
            'require_early_position': True,
            'min_score': 70,
            'min_explosion': 60,
        },
        'quiet_hours': {
            'enabled': False,
            'start': '22:00',
            'end': '08:00'
        }
    }
    
    try:
        if os.path.exists(ALERT_SETTINGS_FILE):
            with open(ALERT_SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                # Merge with defaults
                for key, value in defaults.items():
                    if key not in saved:
                        saved[key] = value
                return saved
    except Exception:
        pass
    
    return defaults


def save_alert_settings(settings: dict):
    """Save alert settings to file"""
    try:
        with open(ALERT_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving alert settings: {e}")


def send_telegram_message(bot_token: str, chat_id: str, message: str, parse_mode: str = 'HTML') -> bool:
    """
    Send a message via Telegram bot.
    
    Args:
        bot_token: Telegram bot token from BotFather
        chat_id: Your Telegram chat ID
        message: Message to send (supports HTML formatting)
        parse_mode: 'HTML' or 'Markdown'
    
    Returns:
        True if sent successfully, False otherwise
    """
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def test_telegram_connection(bot_token: str, chat_id: str) -> tuple:
    """
    Test Telegram connection with a test message.
    
    Returns:
        (success: bool, message: str)
    """
    test_msg = """
🧪 <b>InvestorIQ Alert Test</b>

✅ Connection successful!
Your alerts will appear here when conditions are met.

<i>Powered by InvestorIQ</i>
"""
    
    success = send_telegram_message(bot_token, chat_id, test_msg)
    
    if success:
        return True, "✅ Test message sent! Check your Telegram."
    else:
        return False, "❌ Failed to send. Check your bot token and chat ID."


def format_alert_message(
    symbol: str,
    direction: str,
    score: int,
    timeframe: str,
    mode: str,
    vwap_info: dict = None,
    ml_info: dict = None,
    whale_pct: float = 50,
    position: str = 'MIDDLE',
    explosion_score: int = 0,
    entry_price: float = 0,
    tp1: float = 0,
    sl: float = 0,
) -> str:
    """
    Format a beautiful alert message for Telegram.
    """
    # Direction emoji
    dir_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⏳"
    
    # Score color indicator
    if score >= 80:
        score_indicator = "🔥🔥🔥"
    elif score >= 70:
        score_indicator = "🔥🔥"
    elif score >= 60:
        score_indicator = "🔥"
    else:
        score_indicator = ""
    
    # Build message
    msg_parts = []
    msg_parts.append(f"🚨 <b>ALERT: {symbol}</b> {dir_emoji}")
    msg_parts.append("")
    msg_parts.append(f"📊 <b>Score:</b> {score}/100 {score_indicator}")
    msg_parts.append(f"⏱️ <b>Mode:</b> {mode} ({timeframe})")
    msg_parts.append(f"🐋 <b>Whales:</b> {whale_pct:.0f}% {direction}")
    msg_parts.append(f"📍 <b>Position:</b> {position}")
    
    # VWAP info
    if vwap_info:
        vwap_age = vwap_info.get('signal_age', '')
        vwap_type = vwap_info.get('bounce_type', '')
        time_ago = vwap_info.get('time_ago', '')
        
        if vwap_type == 'BULLISH_BOUNCE':
            vwap_emoji = "🟢"
        elif vwap_type == 'BEARISH_BOUNCE':
            vwap_emoji = "🔴"
        else:
            vwap_emoji = "🟡"
        
        age_emoji = "🆕" if vwap_age == 'FRESH' else "⏰" if vwap_age == 'STALE' else ""
        msg_parts.append(f"📈 <b>VWAP:</b> {vwap_emoji} {vwap_age} {age_emoji} ({time_ago})")
    
    # ML alignment
    if ml_info:
        ml_dir = ml_info.get('direction', '')
        ml_conf = ml_info.get('confidence', 0)
        aligned = ml_info.get('aligned', False)
        align_emoji = "✅" if aligned else "⚠️"
        msg_parts.append(f"🤖 <b>ML:</b> {ml_dir} ({ml_conf:.0f}%) {align_emoji}")
    
    # Explosion
    if explosion_score > 0:
        exp_emoji = "🚀" if explosion_score >= 70 else "⚡" if explosion_score >= 50 else "📊"
        msg_parts.append(f"{exp_emoji} <b>Explosion:</b> {explosion_score}/100")
    
    # Levels
    msg_parts.append("")
    msg_parts.append("<b>📍 Levels:</b>")
    if entry_price:
        msg_parts.append(f"  Entry: ${entry_price:,.6f}" if entry_price < 1 else f"  Entry: ${entry_price:,.2f}")
    if tp1:
        msg_parts.append(f"  TP1: ${tp1:,.6f}" if tp1 < 1 else f"  TP1: ${tp1:,.2f}")
    if sl:
        msg_parts.append(f"  SL: ${sl:,.6f}" if sl < 1 else f"  SL: ${sl:,.2f}")
    
    # Footer
    msg_parts.append("")
    msg_parts.append(f"<i>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")
    
    return "\n".join(msg_parts)


def check_alert_conditions(
    analysis: dict,
    settings: dict,
) -> tuple:
    """
    Check if analysis meets alert conditions.
    
    Returns:
        (should_alert: bool, reasons: list)
    """
    conditions = settings.get('alert_conditions', {})
    reasons = []
    failures = []
    
    # Get analysis data
    score = analysis.get('score', 0)
    direction = analysis.get('trade_direction', 'WAIT')
    vwap = analysis.get('vwap_bounce', {})
    ml_aligned = not analysis.get('has_ml_conflict', True)
    position = analysis.get('move_position', 'MIDDLE')
    explosion = analysis.get('explosion', {}).get('score', 0) if analysis.get('explosion') else 0
    
    # Check minimum score
    min_score = conditions.get('min_score', 70)
    if score >= min_score:
        reasons.append(f"Score {score} >= {min_score}")
    else:
        failures.append(f"Score {score} < {min_score}")
    
    # Check VWAP fresh
    if conditions.get('require_vwap_fresh', True):
        vwap_age = vwap.get('signal_age', '')
        vwap_type = vwap.get('bounce_type', '')
        
        # Check if VWAP bounce aligns with direction
        vwap_aligned = (
            (direction == 'LONG' and vwap_type == 'BULLISH_BOUNCE') or
            (direction == 'SHORT' and vwap_type == 'BEARISH_BOUNCE')
        )
        
        if vwap_age == 'FRESH' and vwap_aligned:
            reasons.append(f"VWAP FRESH + aligned")
        elif vwap_age in ['FRESH', 'RECENT'] and vwap_aligned:
            reasons.append(f"VWAP {vwap_age} + aligned")
        else:
            failures.append(f"VWAP not fresh/aligned")
    
    # Check ML alignment
    if conditions.get('require_ml_aligned', True):
        if ml_aligned:
            reasons.append("ML + Rules aligned")
        else:
            failures.append("ML conflict")
    
    # Check early position
    if conditions.get('require_early_position', True):
        if position in ['EARLY', 'IDEAL']:
            reasons.append(f"Position: {position}")
        else:
            failures.append(f"Position: {position} (not early)")
    
    # Check explosion
    min_explosion = conditions.get('min_explosion', 0)
    if min_explosion > 0:
        if explosion >= min_explosion:
            reasons.append(f"Explosion {explosion} >= {min_explosion}")
        else:
            failures.append(f"Explosion {explosion} < {min_explosion}")
    
    # Direction must not be WAIT
    if direction == 'WAIT':
        failures.append("Direction is WAIT")
    
    # All conditions must pass
    should_alert = len(failures) == 0 and len(reasons) > 0
    
    return should_alert, reasons, failures


class WatchlistMonitor:
    """
    Monitors watchlist items and sends alerts when conditions are met.
    """
    
    def __init__(self):
        self.settings = load_alert_settings()
        self.last_alerts = {}  # symbol -> last alert timestamp
        self.alert_cooldown_minutes = 30  # Don't re-alert same symbol within this time
    
    def reload_settings(self):
        """Reload settings from file"""
        self.settings = load_alert_settings()
    
    def should_send_alert(self, symbol: str) -> bool:
        """Check if we should send alert (cooldown check)"""
        if symbol in self.last_alerts:
            last_time = self.last_alerts[symbol]
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            if elapsed < self.alert_cooldown_minutes:
                return False
        return True
    
    def mark_alerted(self, symbol: str):
        """Mark symbol as alerted"""
        self.last_alerts[symbol] = datetime.now()
    
    def send_alert(self, symbol: str, analysis: dict, signal: dict = None) -> bool:
        """
        Send alert if conditions met and Telegram configured.
        
        Returns True if alert was sent.
        """
        if not self.settings.get('telegram_enabled', False):
            return False
        
        bot_token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')
        
        if not bot_token or not chat_id:
            return False
        
        # Check cooldown
        if not self.should_send_alert(symbol):
            return False
        
        # Check conditions
        should_alert, reasons, failures = check_alert_conditions(analysis, self.settings)
        
        if not should_alert:
            return False
        
        # Build and send message
        vwap_info = analysis.get('vwap_bounce')
        ml_prediction = analysis.get('ml_prediction')
        
        ml_info = None
        if ml_prediction:
            ml_info = {
                'direction': ml_prediction.direction if hasattr(ml_prediction, 'direction') else '',
                'confidence': ml_prediction.confidence if hasattr(ml_prediction, 'confidence') else 0,
                'aligned': not analysis.get('has_ml_conflict', True)
            }
        
        message = format_alert_message(
            symbol=symbol,
            direction=analysis.get('trade_direction', 'WAIT'),
            score=analysis.get('score', 0),
            timeframe=analysis.get('timeframe', '15m'),
            mode=analysis.get('mode_name', 'Day Trade'),
            vwap_info=vwap_info,
            ml_info=ml_info,
            whale_pct=analysis.get('whale_pct', 50),
            position=analysis.get('move_position', 'MIDDLE'),
            explosion_score=analysis.get('explosion', {}).get('score', 0) if analysis.get('explosion') else 0,
            entry_price=signal.get('entry', 0) if signal else 0,
            tp1=signal.get('tp1', 0) if signal else 0,
            sl=signal.get('stop_loss', 0) if signal else 0,
        )
        
        success = send_telegram_message(bot_token, chat_id, message)
        
        if success:
            self.mark_alerted(symbol)
        
        return success


# Global monitor instance
_monitor = None

def get_monitor() -> WatchlistMonitor:
    """Get or create the global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = WatchlistMonitor()
    return _monitor
