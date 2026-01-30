"""
DATA VALIDATOR - Ensures data quality for trading decisions
============================================================

PRINCIPLE: Never hide missing data behind defaults.
- Track what data is REAL vs FALLBACK
- Show warnings when critical data is missing
- Fail loudly for trading-critical data

This module provides:
1. DataQuality class to track data source reliability
2. Validation functions for critical trading inputs
3. Warning flags for UI display
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class DataSource(Enum):
    """Source of data - for transparency"""
    REAL_API = "real_api"           # From actual API call
    CACHED = "cached"               # From cache (may be stale)
    CALCULATED = "calculated"       # Derived from other data
    DEFAULT = "default"             # Hardcoded default - DANGEROUS!
    MISSING = "missing"             # No data available
    ERROR = "error"                 # API/calculation error


@dataclass
class DataQuality:
    """Track quality and source of data"""
    value: Any
    source: DataSource
    is_reliable: bool
    warning: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    
    @property
    def is_real(self) -> bool:
        """Is this real data (not a default)?"""
        return self.source in [DataSource.REAL_API, DataSource.CACHED, DataSource.CALCULATED]
    
    @property
    def is_usable(self) -> bool:
        """Can this data be used for trading decisions?"""
        return self.source != DataSource.MISSING and self.source != DataSource.ERROR
    
    def __repr__(self):
        status = "✅" if self.is_real else "⚠️" if self.is_usable else "❌"
        return f"{status} {self.value} ({self.source.value})"


@dataclass 
class AnalysisDataQuality:
    """Track data quality for entire analysis"""
    whale_pct: DataQuality = None
    retail_pct: DataQuality = None
    explosion_score: DataQuality = None
    ml_prediction: DataQuality = None
    vwap_data: DataQuality = None
    ta_score: DataQuality = None
    
    # Overall quality
    critical_data_missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_tradeable(self) -> bool:
        """Is there enough reliable data to make trading decisions?"""
        # At minimum need whale data OR explosion score
        has_whale = self.whale_pct and self.whale_pct.is_real
        has_explosion = self.explosion_score and self.explosion_score.is_real
        return has_whale or has_explosion
    
    @property
    def reliability_pct(self) -> int:
        """Percentage of data that is real (not defaults)"""
        fields = [self.whale_pct, self.retail_pct, self.explosion_score, 
                  self.ml_prediction, self.vwap_data, self.ta_score]
        real_count = sum(1 for f in fields if f and f.is_real)
        total = sum(1 for f in fields if f is not None)
        return int(real_count / total * 100) if total > 0 else 0
    
    def get_quality_display(self) -> str:
        """Get display string for UI"""
        pct = self.reliability_pct
        if pct >= 80:
            return f"✅ {pct}% Real Data"
        elif pct >= 50:
            return f"⚠️ {pct}% Real Data"
        else:
            return f"❌ {pct}% Real Data - CAUTION!"


def validate_whale_data(whale_dict: Optional[dict], source: str = "unknown") -> DataQuality:
    """
    Validate whale data and track its source.
    
    Returns DataQuality with proper source tracking.
    """
    if whale_dict is None:
        return DataQuality(
            value=50,
            source=DataSource.MISSING,
            is_reliable=False,
            warning="No whale data available",
            error="whale_dict is None"
        )
    
    top_trader = whale_dict.get('top_trader_ls', {})
    
    if not isinstance(top_trader, dict):
        return DataQuality(
            value=50,
            source=DataSource.DEFAULT,
            is_reliable=False,
            warning="Invalid whale data format"
        )
    
    whale_pct = top_trader.get('long_pct')
    
    if whale_pct is None:
        return DataQuality(
            value=50,
            source=DataSource.DEFAULT,
            is_reliable=False,
            warning="Whale percentage not in data"
        )
    
    # Real data!
    return DataQuality(
        value=whale_pct,
        source=DataSource.REAL_API if source == "api" else DataSource.CACHED,
        is_reliable=True
    )


def validate_explosion_score(explosion_dict: Optional[dict]) -> DataQuality:
    """Validate explosion/compression score"""
    if explosion_dict is None:
        return DataQuality(
            value=0,
            source=DataSource.MISSING,
            is_reliable=False,
            warning="No explosion data"
        )
    
    score = explosion_dict.get('score')
    
    if score is None:
        return DataQuality(
            value=0,
            source=DataSource.DEFAULT,
            is_reliable=False,
            warning="Explosion score not calculated"
        )
    
    return DataQuality(
        value=score,
        source=DataSource.CALCULATED,
        is_reliable=True
    )


def validate_vwap_data(vwap_dict: Optional[dict]) -> DataQuality:
    """Validate VWAP bounce data"""
    if vwap_dict is None:
        return DataQuality(
            value=None,
            source=DataSource.MISSING,
            is_reliable=False,
            warning="No VWAP data",
            error="vwap_dict is None"
        )
    
    # Check for error flag
    if vwap_dict.get('error'):
        return DataQuality(
            value=vwap_dict,
            source=DataSource.ERROR,
            is_reliable=False,
            error=vwap_dict.get('error')
        )
    
    # Must have at least distance_pct and position
    has_distance = vwap_dict.get('distance_pct') is not None
    has_position = vwap_dict.get('position') is not None
    
    if not has_distance and not has_position:
        return DataQuality(
            value=vwap_dict,
            source=DataSource.DEFAULT,
            is_reliable=False,
            warning="VWAP data incomplete"
        )
    
    return DataQuality(
        value=vwap_dict,
        source=DataSource.CALCULATED,
        is_reliable=True
    )


def validate_ml_prediction(ml_pred) -> DataQuality:
    """Validate ML prediction"""
    if ml_pred is None:
        return DataQuality(
            value=None,
            source=DataSource.MISSING,
            is_reliable=False,
            warning="No ML prediction"
        )
    
    if not hasattr(ml_pred, 'direction') or not hasattr(ml_pred, 'confidence'):
        return DataQuality(
            value=ml_pred,
            source=DataSource.ERROR,
            is_reliable=False,
            error="Invalid ML prediction object"
        )
    
    if ml_pred.direction == 'UNKNOWN':
        return DataQuality(
            value=ml_pred,
            source=DataSource.DEFAULT,
            is_reliable=False,
            warning="ML prediction is UNKNOWN"
        )
    
    return DataQuality(
        value=ml_pred,
        source=DataSource.CALCULATED,
        is_reliable=True
    )


def create_data_quality_report(analysis: dict) -> AnalysisDataQuality:
    """
    Create a comprehensive data quality report for an analysis.
    
    This should be called after analyze_symbol_full() to assess
    the reliability of the data before making trading decisions.
    """
    report = AnalysisDataQuality()
    
    # Validate each critical data source
    report.whale_pct = validate_whale_data(analysis.get('whale'), "api")
    
    # Retail
    retail_dict = analysis.get('whale', {})
    retail_data = retail_dict.get('retail_ls', {}) if retail_dict else {}
    retail_pct = retail_data.get('long_pct') if isinstance(retail_data, dict) else None
    if retail_pct is not None:
        report.retail_pct = DataQuality(retail_pct, DataSource.REAL_API, True)
    else:
        report.retail_pct = DataQuality(50, DataSource.DEFAULT, False, "No retail data")
    
    report.explosion_score = validate_explosion_score(analysis.get('explosion'))
    report.vwap_data = validate_vwap_data(analysis.get('vwap_bounce'))
    report.ml_prediction = validate_ml_prediction(analysis.get('ml_prediction'))
    
    # TA Score
    ta = analysis.get('confidence_scores', {}).get('ta_score')
    if ta is not None:
        report.ta_score = DataQuality(ta, DataSource.CALCULATED, True)
    else:
        report.ta_score = DataQuality(50, DataSource.DEFAULT, False, "No TA score")
    
    # Collect warnings
    for field_name in ['whale_pct', 'retail_pct', 'explosion_score', 'vwap_data', 'ml_prediction', 'ta_score']:
        field = getattr(report, field_name)
        if field and field.warning:
            report.warnings.append(f"{field_name}: {field.warning}")
        if field and not field.is_real:
            report.critical_data_missing.append(field_name)
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE GETTERS - Use these instead of .get() with defaults
# ═══════════════════════════════════════════════════════════════════════════════

def safe_get_whale_pct(whale_dict: dict, require_real: bool = True) -> tuple[float, bool, str]:
    """
    Safely get whale percentage with data quality flag.
    
    Returns: (value, is_real_data, warning_message)
    
    Usage:
        whale_pct, is_real, warning = safe_get_whale_pct(whale)
        if not is_real and require_real:
            # Handle missing data - don't proceed with trade
    """
    quality = validate_whale_data(whale_dict)
    return quality.value, quality.is_real, quality.warning or ""


def safe_get_explosion(explosion_dict: dict) -> tuple[int, bool, str]:
    """Safely get explosion score"""
    quality = validate_explosion_score(explosion_dict)
    return quality.value, quality.is_real, quality.warning or ""


def safe_get_vwap(vwap_dict: dict) -> tuple[dict, bool, str]:
    """Safely get VWAP data"""
    quality = validate_vwap_data(vwap_dict)
    return quality.value, quality.is_real, quality.warning or quality.error or ""
