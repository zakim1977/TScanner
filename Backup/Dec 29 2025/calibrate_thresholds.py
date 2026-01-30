#!/usr/bin/env python
"""
Threshold Calibration Tool
===========================
Analyzes historical data to find ACTUAL win rates for different indicator combinations.

Instead of guessing thresholds, we let the DATA tell us:
- Which patterns have >65% win rate → TRUST
- Which patterns have <40% win rate → AVOID
- Optimal threshold values based on outcomes

Usage:
    python calibrate_thresholds.py --db data/whale_history.db
    python calibrate_thresholds.py --db data/whale_history.db --export calibrated_rules.json
"""

import sqlite3
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os


@dataclass
class PatternResult:
    """Result of a pattern analysis"""
    pattern_name: str
    conditions: Dict
    total_samples: int
    wins: int
    losses: int
    win_rate: float
    avg_gain_on_win: float
    avg_loss_on_loss: float
    expected_value: float  # (win_rate * avg_gain) - ((1-win_rate) * avg_loss)
    recommendation: str  # STRONG_BUY, BUY, NEUTRAL, AVOID, STRONG_AVOID
    

class ThresholdCalibrator:
    """
    Analyzes historical whale/price data to find optimal trading thresholds.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.data = []
        self.results = {}
        
    def load_data(self) -> int:
        """Load all historical data with price outcomes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all snapshots ordered by symbol and time
        cursor.execute("""
            SELECT 
                symbol,
                timestamp,
                whale_long_pct,
                retail_long_pct,
                oi_change_24h,
                price,
                price_change_24h,
                funding_rate,
                position_in_range
            FROM whale_snapshots 
            WHERE whale_long_pct IS NOT NULL 
            AND price IS NOT NULL
            ORDER BY symbol, timestamp
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        self.data = []
        for row in rows:
            self.data.append({
                'symbol': row[0],
                'timestamp': row[1],
                'whale_pct': row[2] or 50,
                'retail_pct': row[3] or 50,
                'oi_change': row[4] or 0,
                'price': row[5],
                'price_change': row[6] or 0,
                'funding_rate': row[7] or 0,
                'position_pct': row[8] or 50
            })
        
        print(f"Loaded {len(self.data)} historical snapshots")
        return len(self.data)
    
    def calculate_future_returns(self, lookahead_periods: int = 6) -> List[Dict]:
        """
        For each snapshot, calculate what price did in the next N periods.
        lookahead_periods=6 means 6 x 4h = 24h lookahead
        """
        enriched = []
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for d in self.data:
            by_symbol[d['symbol']].append(d)
        
        for symbol, snapshots in by_symbol.items():
            # Sort by timestamp
            snapshots.sort(key=lambda x: x['timestamp'])
            
            for i, snap in enumerate(snapshots):
                if i + lookahead_periods < len(snapshots):
                    future = snapshots[i + lookahead_periods]
                    
                    # Calculate return
                    if snap['price'] > 0:
                        future_return = ((future['price'] - snap['price']) / snap['price']) * 100
                    else:
                        future_return = 0
                    
                    snap_enriched = snap.copy()
                    snap_enriched['future_return'] = future_return
                    snap_enriched['future_price'] = future['price']
                    enriched.append(snap_enriched)
        
        print(f"Calculated future returns for {len(enriched)} snapshots")
        return enriched
    
    def analyze_pattern(
        self, 
        data: List[Dict],
        pattern_name: str,
        conditions: Dict,
        direction: str = "LONG"  # LONG or SHORT
    ) -> Optional[PatternResult]:
        """
        Analyze a specific pattern's win rate.
        
        conditions example:
        {
            'whale_pct': ('>=', 65),
            'retail_pct': ('<=', 45),
            'oi_change': ('<', -2),
            'price_change': ('>', 1.5),
            'position_pct': ('<=', 40)
        }
        """
        
        # Filter data matching conditions
        matches = []
        for d in data:
            match = True
            for field, (op, value) in conditions.items():
                actual = d.get(field, 0)
                if op == '>=' and not (actual >= value):
                    match = False
                elif op == '<=' and not (actual <= value):
                    match = False
                elif op == '>' and not (actual > value):
                    match = False
                elif op == '<' and not (actual < value):
                    match = False
                elif op == '==' and not (actual == value):
                    match = False
                elif op == 'between':
                    low, high = value
                    if not (low <= actual <= high):
                        match = False
            
            if match:
                matches.append(d)
        
        if len(matches) < 20:
            # Not enough samples for statistical significance
            return None
        
        # Calculate wins/losses
        # For LONG: win = future_return > 0
        # For SHORT: win = future_return < 0
        wins = 0
        losses = 0
        gains = []
        loss_amounts = []
        
        for m in matches:
            ret = m['future_return']
            
            if direction == "LONG":
                if ret > 0.5:  # Min 0.5% to count as win (covers fees)
                    wins += 1
                    gains.append(ret)
                elif ret < -0.5:
                    losses += 1
                    loss_amounts.append(abs(ret))
            else:  # SHORT
                if ret < -0.5:
                    wins += 1
                    gains.append(abs(ret))
                elif ret > 0.5:
                    losses += 1
                    loss_amounts.append(ret)
        
        total = wins + losses
        if total == 0:
            return None
        
        win_rate = (wins / total) * 100
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0
        
        # Expected value per trade
        ev = (win_rate/100 * avg_gain) - ((1 - win_rate/100) * avg_loss)
        
        # Recommendation
        if win_rate >= 70 and ev > 1:
            rec = "STRONG_BUY"
        elif win_rate >= 60 and ev > 0.5:
            rec = "BUY"
        elif win_rate >= 50 and ev > 0:
            rec = "NEUTRAL"
        elif win_rate >= 40:
            rec = "AVOID"
        else:
            rec = "STRONG_AVOID"
        
        return PatternResult(
            pattern_name=pattern_name,
            conditions=conditions,
            total_samples=len(matches),
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 1),
            avg_gain_on_win=round(avg_gain, 2),
            avg_loss_on_loss=round(avg_loss, 2),
            expected_value=round(ev, 2),
            recommendation=rec
        )
    
    def calibrate_all_patterns(self) -> Dict[str, PatternResult]:
        """
        Test ALL important patterns and find their actual win rates.
        """
        
        # Load and prepare data
        self.load_data()
        data = self.calculate_future_returns(lookahead_periods=6)  # 24h lookahead
        
        results = {}
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 1: SHORT COVERING (OI down + Price up)
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: SHORT COVERING PATTERNS")
        print("="*60)
        
        # Test different OI/Price thresholds
        for oi_thresh in [-1, -2, -3, -4, -5]:
            for price_thresh in [1, 1.5, 2, 2.5, 3, 4, 5]:
                name = f"SHORT_COVERING_OI{oi_thresh}_P{price_thresh}"
                result = self.analyze_pattern(
                    data, name,
                    conditions={
                        'oi_change': ('<', oi_thresh),
                        'price_change': ('>', price_thresh)
                    },
                    direction="LONG"
                )
                if result:
                    results[name] = result
                    print(f"  {name}: {result.win_rate:.1f}% win ({result.total_samples} samples) → {result.recommendation}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 2: WHALE SQUEEZE (Whale bullish, Retail bearish)
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: WHALE SQUEEZE PATTERNS")
        print("="*60)
        
        for whale_thresh in [60, 65, 70, 75]:
            for retail_thresh in [35, 40, 45, 50]:
                if whale_thresh > retail_thresh + 10:  # Must have divergence
                    name = f"WHALE_SQUEEZE_W{whale_thresh}_R{retail_thresh}"
                    result = self.analyze_pattern(
                        data, name,
                        conditions={
                            'whale_pct': ('>=', whale_thresh),
                            'retail_pct': ('<=', retail_thresh)
                        },
                        direction="LONG"
                    )
                    if result:
                        results[name] = result
                        print(f"  {name}: {result.win_rate:.1f}% win ({result.total_samples} samples) → {result.recommendation}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 3: RETAIL TRAP (Retail more bullish than Whale)
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: RETAIL TRAP PATTERNS")
        print("="*60)
        
        for whale_thresh in [40, 45, 50]:
            for retail_thresh in [55, 60, 65, 70]:
                if retail_thresh > whale_thresh:
                    name = f"RETAIL_TRAP_W{whale_thresh}_R{retail_thresh}"
                    result = self.analyze_pattern(
                        data, name,
                        conditions={
                            'whale_pct': ('<=', whale_thresh),
                            'retail_pct': ('>=', retail_thresh)
                        },
                        direction="LONG"  # Testing if LONG is bad here
                    )
                    if result:
                        results[name] = result
                        status = "CONFIRMS TRAP" if result.win_rate < 45 else "WEAK TRAP"
                        print(f"  {name}: {result.win_rate:.1f}% win for LONGS ({result.total_samples} samples) → {status}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 4: POSITION-BASED (Entry timing)
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: POSITION-BASED PATTERNS")
        print("="*60)
        
        # Early position (near lows) - should be good for longs
        for pos_thresh in [20, 30, 40]:
            name = f"EARLY_POSITION_{pos_thresh}"
            result = self.analyze_pattern(
                data, name,
                conditions={
                    'position_pct': ('<=', pos_thresh),
                    'whale_pct': ('>=', 55)  # At least slightly bullish whales
                },
                direction="LONG"
            )
            if result:
                results[name] = result
                print(f"  {name} + Whale>55%: {result.win_rate:.1f}% win ({result.total_samples} samples)")
        
        # Late position (near highs) - should be bad for longs
        for pos_thresh in [60, 70, 80]:
            name = f"LATE_POSITION_{pos_thresh}"
            result = self.analyze_pattern(
                data, name,
                conditions={
                    'position_pct': ('>=', pos_thresh),
                    'whale_pct': ('>=', 55)
                },
                direction="LONG"
            )
            if result:
                results[name] = result
                print(f"  {name} + Whale>55%: {result.win_rate:.1f}% win ({result.total_samples} samples)")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 5: OI CONFIRMATION (OI rising with price)
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: OI CONFIRMATION PATTERNS")
        print("="*60)
        
        for oi_thresh in [2, 3, 5, 8]:
            for price_thresh in [1, 2, 3]:
                name = f"OI_CONFIRM_OI{oi_thresh}_P{price_thresh}"
                result = self.analyze_pattern(
                    data, name,
                    conditions={
                        'oi_change': ('>=', oi_thresh),
                        'price_change': ('>=', price_thresh)
                    },
                    direction="LONG"
                )
                if result:
                    results[name] = result
                    print(f"  {name}: {result.win_rate:.1f}% win ({result.total_samples} samples) → {result.recommendation}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 6: COMBINED BEST SETUPS
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: COMBINED BEST SETUPS")
        print("="*60)
        
        # Whale squeeze + Early position
        result = self.analyze_pattern(
            data, "SQUEEZE_EARLY",
            conditions={
                'whale_pct': ('>=', 65),
                'retail_pct': ('<=', 45),
                'position_pct': ('<=', 40)
            },
            direction="LONG"
        )
        if result:
            results["SQUEEZE_EARLY"] = result
            print(f"  SQUEEZE_EARLY (W>=65, R<=45, Pos<=40): {result.win_rate:.1f}% ({result.total_samples} samples)")
        
        # Whale squeeze + OI rising
        result = self.analyze_pattern(
            data, "SQUEEZE_OI_RISING",
            conditions={
                'whale_pct': ('>=', 65),
                'retail_pct': ('<=', 45),
                'oi_change': ('>=', 2)
            },
            direction="LONG"
        )
        if result:
            results["SQUEEZE_OI_RISING"] = result
            print(f"  SQUEEZE_OI_RISING (W>=65, R<=45, OI>=2): {result.win_rate:.1f}% ({result.total_samples} samples)")
        
        # Whale squeeze + SHORT COVERING (the case you found!)
        result = self.analyze_pattern(
            data, "SQUEEZE_SHORT_COVERING",
            conditions={
                'whale_pct': ('>=', 65),
                'retail_pct': ('<=', 45),
                'oi_change': ('<', -2),
                'price_change': ('>', 1.5)
            },
            direction="LONG"
        )
        if result:
            results["SQUEEZE_SHORT_COVERING"] = result
            print(f"  SQUEEZE + SHORT_COVERING: {result.win_rate:.1f}% ({result.total_samples} samples) ⚠️ KEY INSIGHT!")
        
        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 7: FUNDING RATE EXTREMES
        # ═══════════════════════════════════════════════════════════════════════
        
        print("\n" + "="*60)
        print("ANALYZING: FUNDING RATE PATTERNS")
        print("="*60)
        
        # Very negative funding (everyone short) + whale bullish
        result = self.analyze_pattern(
            data, "NEGATIVE_FUNDING_WHALE_BULLISH",
            conditions={
                'funding_rate': ('<', -0.01),
                'whale_pct': ('>=', 60)
            },
            direction="LONG"
        )
        if result:
            results["NEGATIVE_FUNDING_WHALE_BULLISH"] = result
            print(f"  NEG_FUNDING + WHALE_BULLISH: {result.win_rate:.1f}% ({result.total_samples} samples)")
        
        # Very positive funding (everyone long) + whale bearish = dump incoming
        result = self.analyze_pattern(
            data, "POSITIVE_FUNDING_WHALE_BEARISH",
            conditions={
                'funding_rate': ('>', 0.02),
                'whale_pct': ('<=', 45)
            },
            direction="SHORT"
        )
        if result:
            results["POSITIVE_FUNDING_WHALE_BEARISH"] = result
            print(f"  POS_FUNDING + WHALE_BEARISH (SHORT): {result.win_rate:.1f}% ({result.total_samples} samples)")
        
        self.results = results
        return results
    
    def get_optimal_thresholds(self) -> Dict:
        """
        Extract optimal thresholds based on calibration results.
        """
        
        optimal = {
            'short_covering': {
                'oi_threshold': -2,
                'price_threshold': 3,
                'action': 'AVOID',
                'win_rate': 0,
                'samples': 0
            },
            'whale_squeeze': {
                'whale_min': 65,
                'retail_max': 45,
                'action': 'BUY',
                'win_rate': 0,
                'samples': 0
            },
            'retail_trap': {
                'whale_max': 45,
                'retail_min': 55,
                'action': 'AVOID_LONG',
                'win_rate': 0,
                'samples': 0
            }
        }
        
        # Find best SHORT COVERING threshold
        best_sc = None
        for name, result in self.results.items():
            if name.startswith('SHORT_COVERING_') and result.total_samples >= 30:
                if best_sc is None or result.total_samples > best_sc.total_samples:
                    best_sc = result
                    # Parse thresholds from name
                    parts = name.replace('SHORT_COVERING_OI', '').split('_P')
                    optimal['short_covering']['oi_threshold'] = int(parts[0])
                    optimal['short_covering']['price_threshold'] = float(parts[1])
                    optimal['short_covering']['win_rate'] = result.win_rate
                    optimal['short_covering']['samples'] = result.total_samples
                    optimal['short_covering']['action'] = result.recommendation
        
        # Find best WHALE SQUEEZE threshold
        best_ws = None
        for name, result in self.results.items():
            if name.startswith('WHALE_SQUEEZE_') and result.total_samples >= 30:
                if best_ws is None or result.win_rate > best_ws.win_rate:
                    best_ws = result
                    parts = name.replace('WHALE_SQUEEZE_W', '').split('_R')
                    optimal['whale_squeeze']['whale_min'] = int(parts[0])
                    optimal['whale_squeeze']['retail_max'] = int(parts[1])
                    optimal['whale_squeeze']['win_rate'] = result.win_rate
                    optimal['whale_squeeze']['samples'] = result.total_samples
                    optimal['whale_squeeze']['action'] = result.recommendation
        
        return optimal
    
    def generate_report(self) -> str:
        """Generate human-readable calibration report"""
        
        report = []
        report.append("=" * 70)
        report.append("THRESHOLD CALIBRATION REPORT")
        report.append("Based on historical data analysis")
        report.append("=" * 70)
        report.append("")
        
        # Group by recommendation
        strong_buy = []
        buy = []
        neutral = []
        avoid = []
        strong_avoid = []
        
        for name, result in self.results.items():
            if result.recommendation == "STRONG_BUY":
                strong_buy.append(result)
            elif result.recommendation == "BUY":
                buy.append(result)
            elif result.recommendation == "NEUTRAL":
                neutral.append(result)
            elif result.recommendation == "AVOID":
                avoid.append(result)
            else:
                strong_avoid.append(result)
        
        report.append("🟢 STRONG BUY PATTERNS (Win Rate >= 70%)")
        report.append("-" * 50)
        for r in sorted(strong_buy, key=lambda x: -x.win_rate):
            report.append(f"  {r.pattern_name}")
            report.append(f"    Win Rate: {r.win_rate}% | Samples: {r.total_samples}")
            report.append(f"    Avg Win: +{r.avg_gain_on_win}% | Avg Loss: -{r.avg_loss_on_loss}%")
            report.append(f"    Expected Value: {r.expected_value}% per trade")
            report.append("")
        
        report.append("🟡 BUY PATTERNS (Win Rate 60-70%)")
        report.append("-" * 50)
        for r in sorted(buy, key=lambda x: -x.win_rate):
            report.append(f"  {r.pattern_name}")
            report.append(f"    Win Rate: {r.win_rate}% | Samples: {r.total_samples}")
            report.append("")
        
        report.append("⚪ NEUTRAL PATTERNS (Win Rate 50-60%)")
        report.append("-" * 50)
        for r in sorted(neutral, key=lambda x: -x.win_rate):
            report.append(f"  {r.pattern_name}: {r.win_rate}% ({r.total_samples} samples)")
        report.append("")
        
        report.append("🔴 AVOID PATTERNS (Win Rate 40-50%)")
        report.append("-" * 50)
        for r in sorted(avoid, key=lambda x: x.win_rate):
            report.append(f"  {r.pattern_name}: {r.win_rate}% ({r.total_samples} samples)")
        report.append("")
        
        report.append("⛔ STRONG AVOID PATTERNS (Win Rate < 40%)")
        report.append("-" * 50)
        for r in sorted(strong_avoid, key=lambda x: x.win_rate):
            report.append(f"  {r.pattern_name}: {r.win_rate}% ({r.total_samples} samples)")
            report.append(f"    CONDITIONS TO AVOID: {r.conditions}")
        report.append("")
        
        # Optimal thresholds
        optimal = self.get_optimal_thresholds()
        report.append("=" * 70)
        report.append("RECOMMENDED THRESHOLD UPDATES")
        report.append("=" * 70)
        report.append("")
        report.append(f"SHORT_COVERING:")
        report.append(f"  OI Threshold: {optimal['short_covering']['oi_threshold']}%")
        report.append(f"  Price Threshold: {optimal['short_covering']['price_threshold']}%")
        report.append(f"  Action: {optimal['short_covering']['action']}")
        report.append(f"  Confidence: {optimal['short_covering']['samples']} samples")
        report.append("")
        report.append(f"WHALE_SQUEEZE:")
        report.append(f"  Min Whale: {optimal['whale_squeeze']['whale_min']}%")
        report.append(f"  Max Retail: {optimal['whale_squeeze']['retail_max']}%")
        report.append(f"  Win Rate: {optimal['whale_squeeze']['win_rate']}%")
        report.append("")
        
        return "\n".join(report)
    
    def export_calibrated_rules(self, output_path: str):
        """Export calibrated thresholds as JSON for MASTER_RULES"""
        
        output = {
            'calibration_date': str(os.popen('date').read().strip()),
            'total_samples_analyzed': len(self.data),
            'patterns': {},
            'optimal_thresholds': self.get_optimal_thresholds()
        }
        
        for name, result in self.results.items():
            output['patterns'][name] = asdict(result)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nExported calibrated rules to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate trading thresholds from historical data")
    parser.add_argument("--db", default="data/whale_history.db", help="Path to whale history database")
    parser.add_argument("--export", help="Export calibrated rules to JSON file")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        print("   Run batch_import.py first to import historical data")
        return
    
    calibrator = ThresholdCalibrator(args.db)
    results = calibrator.calibrate_all_patterns()
    
    if args.report or True:  # Always show report
        print("\n")
        print(calibrator.generate_report())
    
    if args.export:
        calibrator.export_calibrated_rules(args.export)
    else:
        # Default export
        calibrator.export_calibrated_rules("calibrated_thresholds.json")


if __name__ == "__main__":
    main()
