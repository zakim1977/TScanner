"""
ERROR TRACKER - Replace silent passes with proper error tracking
================================================================

PRINCIPLE: Never hide errors. Track them, log them, show them.

This module provides:
1. Error collection during analysis
2. Warning display for UI
3. Debug logging for troubleshooting
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Set up logging to file
logging.basicConfig(
    filename='investoriq_errors.log',
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('InvestorIQ')


@dataclass
class AnalysisError:
    """Single error during analysis"""
    step: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    severity: str = "warning"  # warning, error, critical
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.step}: {self.message}"


@dataclass
class ErrorTracker:
    """Collect errors during analysis - REPLACES SILENT PASSES"""
    errors: List[AnalysisError] = field(default_factory=list)
    symbol: str = ""
    timeframe: str = ""
    
    def add(self, step: str, error: Exception, severity: str = "warning"):
        """Add an error - NEVER SILENTLY PASS"""
        err = AnalysisError(
            step=step,
            error_type=type(error).__name__,
            message=str(error)[:200],
            severity=severity
        )
        self.errors.append(err)
        
        # Log to file
        log_msg = f"{self.symbol}|{self.timeframe}|{step}|{type(error).__name__}|{str(error)[:100]}"
        if severity == "critical":
            logger.error(log_msg)
        elif severity == "error":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def add_warning(self, step: str, message: str):
        """Add a warning (not from exception)"""
        err = AnalysisError(
            step=step,
            error_type="Warning",
            message=message,
            severity="warning"
        )
        self.errors.append(err)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_critical(self) -> bool:
        return any(e.severity == "critical" for e in self.errors)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "critical")
    
    def get_summary(self) -> str:
        """Get summary for UI display"""
        if not self.errors:
            return ""
        
        critical = self.critical_count
        total = self.error_count
        
        if critical > 0:
            return f"⚠️ {critical} critical, {total - critical} warnings"
        else:
            return f"⚠️ {total} warnings"
    
    def get_warnings_html(self) -> str:
        """Get HTML for displaying warnings in UI"""
        if not self.errors:
            return ""
        
        parts = ["<div style='background: #2a1a1a; border: 1px solid #ff4444; border-radius: 8px; padding: 10px; margin: 10px 0;'>"]
        parts.append("<div style='color: #ff4444; font-weight: bold; margin-bottom: 8px;'>⚠️ Analysis Warnings</div>")
        
        for err in self.errors[:5]:  # Show max 5
            color = "#ff4444" if err.severity == "critical" else "#ffcc00" if err.severity == "error" else "#888"
            parts.append(f"<div style='color: {color}; font-size: 0.85em; margin: 4px 0;'>• {err.step}: {err.message[:80]}</div>")
        
        if len(self.errors) > 5:
            parts.append(f"<div style='color: #888; font-size: 0.8em;'>... and {len(self.errors) - 5} more</div>")
        
        parts.append("</div>")
        return "".join(parts)
    
    def to_dict(self) -> Dict:
        """Convert to dict for storage"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'error_count': self.error_count,
            'critical_count': self.critical_count,
            'errors': [
                {
                    'step': e.step,
                    'type': e.error_type,
                    'message': e.message,
                    'severity': e.severity,
                    'timestamp': e.timestamp
                }
                for e in self.errors
            ]
        }


# Global tracker for current analysis (reset before each analysis)
_current_tracker: Optional[ErrorTracker] = None


def start_tracking(symbol: str, timeframe: str) -> ErrorTracker:
    """Start tracking errors for a new analysis"""
    global _current_tracker
    _current_tracker = ErrorTracker(symbol=symbol, timeframe=timeframe)
    return _current_tracker


def track_error(step: str, error: Exception, severity: str = "warning"):
    """Track an error in the current analysis"""
    global _current_tracker
    if _current_tracker:
        _current_tracker.add(step, error, severity)
    else:
        # Fallback: just log
        logger.warning(f"UNTRACKED|{step}|{type(error).__name__}|{str(error)[:100]}")


def track_warning(step: str, message: str):
    """Track a warning (not from exception)"""
    global _current_tracker
    if _current_tracker:
        _current_tracker.add_warning(step, message)


def get_tracker() -> Optional[ErrorTracker]:
    """Get current tracker"""
    return _current_tracker


def get_error_summary() -> str:
    """Get error summary for current analysis"""
    if _current_tracker:
        return _current_tracker.get_summary()
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE EXECUTION HELPERS - Use these instead of try/except pass
# ═══════════════════════════════════════════════════════════════════════════════

def safe_execute(step: str, func, *args, default=None, severity="warning", **kwargs):
    """
    Execute function safely with error tracking.
    
    REPLACES:
        try:
            result = some_function()
        except:
            pass  # DANGEROUS!
    
    WITH:
        result = safe_execute("step_name", some_function, default=None)
    
    Returns default if function fails, but TRACKS the error.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        track_error(step, e, severity)
        return default


def safe_get(data: dict, key: str, default=None, step: str = None, required: bool = False):
    """
    Safely get value from dict with tracking.
    
    If required=True and value is missing, tracks as warning.
    """
    value = data.get(key) if isinstance(data, dict) else None
    
    if value is None and required and step:
        track_warning(step, f"Missing required field: {key}")
    
    return value if value is not None else default


# ═══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION WITH TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def validate_whale_pct(whale_dict: dict, step: str = "whale_data") -> tuple[float, bool]:
    """
    Validate whale percentage with error tracking.
    
    Returns: (value, is_real)
    """
    if not whale_dict:
        track_warning(step, "No whale data available")
        return 50.0, False
    
    top_trader = whale_dict.get('top_trader_ls', {})
    if not isinstance(top_trader, dict):
        track_warning(step, "Invalid whale data format")
        return 50.0, False
    
    value = top_trader.get('long_pct')
    if value is None:
        track_warning(step, "Whale long_pct not in data")
        return 50.0, False
    
    return float(value), True


def validate_explosion(explosion_dict: dict, step: str = "explosion") -> tuple[int, bool]:
    """Validate explosion score with tracking"""
    if not explosion_dict:
        track_warning(step, "No explosion data")
        return 0, False
    
    score = explosion_dict.get('score')
    if score is None:
        track_warning(step, "Explosion score not calculated")
        return 0, False
    
    return int(score), True


def validate_signal(signal, step: str = "signal") -> bool:
    """Check if signal is valid"""
    if signal is None:
        track_warning(step, "Signal is None")
        return False
    
    if not getattr(signal, 'is_valid', True):
        track_warning(step, "Signal marked as invalid")
        return False
    
    if not hasattr(signal, 'entry') or not signal.entry:
        track_warning(step, "Signal has no entry price")
        return False
    
    return True
