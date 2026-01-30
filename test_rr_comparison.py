"""
Test different R:R ratios to find optimal setting.
Run: python test_rr_comparison.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

print("=" * 70)
print("R:R RATIO COMPARISON TEST")
print("=" * 70)

# Test with 2 coins, 90 days
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TEST_DAYS = 90

print(f"\nTesting with {len(TEST_SYMBOLS)} coins, {TEST_DAYS} days...")

# Fetch data
print("\n[1/3] Fetching data...")
from core.data_fetcher import fetch_klines_parallel

klines_data = fetch_klines_parallel(
    symbols=TEST_SYMBOLS,
    interval='4h',
    limit=min(TEST_DAYS * 6, 500),
    progress_callback=lambda c, t, s: print(f"  Fetching {s}... {c}/{t}")
)

# Load whale history
print("\n[2/3] Loading whale history...")
import os
import sqlite3

whale_history = {}
whale_db = "data/whale_history.db"

if os.path.exists(whale_db):
    conn = sqlite3.connect(whale_db)
    df_whale = pd.read_sql(
        "SELECT symbol, timestamp, whale_long_pct FROM whale_snapshots ORDER BY timestamp",
        conn
    )
    conn.close()
    for symbol in df_whale['symbol'].unique():
        sym_data = df_whale[df_whale['symbol'] == symbol]
        whale_history[symbol] = sym_data.to_dict('records')
    print(f"  ✅ Loaded whale data for {len(whale_history)} symbols")

# Test different R:R ratios
print("\n[3/3] Testing different R:R ratios...")
print("=" * 70)

# Import and modify the quality_model module
import liquidity_hunter.quality_model as qm

# Store original values
original_rr = qm.TARGET_RR_RATIO
original_stop = qm.STOP_ATR_MULTIPLIER

# R:R ratios to test
RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]

results = []

for rr in RR_RATIOS:
    print(f"\n{'─' * 70}")
    print(f"Testing R:R = {rr}:1 (TP = {rr}x SL distance)")
    print(f"{'─' * 70}")

    # Update the module constants
    qm.TARGET_RR_RATIO = rr

    # Regenerate samples with new R:R
    all_samples = []
    for symbol, df in klines_data.items():
        if df is None or len(df) < 100:
            continue
        sym_whale = whale_history.get(symbol, [])
        samples = qm.generate_quality_samples(df, symbol, sym_whale)
        all_samples.extend(samples)

    if not all_samples:
        print("  No samples generated!")
        continue

    # Calculate win rate
    df_samples = pd.DataFrame(all_samples)
    win_rate = df_samples['won'].mean() if 'won' in df_samples.columns else 0
    total_samples = len(df_samples)
    wins = df_samples['won'].sum() if 'won' in df_samples.columns else 0

    # Break-even calculation
    break_even = 1 / (rr + 1)

    # Expected value per trade (at 1% risk)
    # EV = (Win% × R:R) - (Loss% × 1)
    ev_per_trade = (win_rate * rr) - ((1 - win_rate) * 1)

    # Is it profitable?
    profitable = win_rate > break_even

    print(f"  Samples: {total_samples}")
    print(f"  Wins: {wins} ({win_rate:.1%})")
    print(f"  Break-even: {break_even:.1%}")
    print(f"  EV per trade: {ev_per_trade:+.2f}R {'✅' if profitable else '❌'}")

    if profitable:
        # Monthly projection (assume 50 trades/month)
        monthly_roi = ev_per_trade * 50  # 50 trades × EV × 1% risk
        print(f"  Monthly ROI (50 trades, 1% risk): {monthly_roi:+.1f}%")

    results.append({
        'rr': rr,
        'win_rate': win_rate,
        'break_even': break_even,
        'ev_per_trade': ev_per_trade,
        'profitable': profitable,
        'samples': total_samples,
        'wins': wins
    })

# Restore original values
qm.TARGET_RR_RATIO = original_rr
qm.STOP_ATR_MULTIPLIER = original_stop

# Summary
print("\n" + "=" * 70)
print("SUMMARY: R:R COMPARISON")
print("=" * 70)
print(f"\n{'R:R':<8} {'Win Rate':<12} {'Break-Even':<12} {'EV/Trade':<12} {'Status':<10}")
print("─" * 60)

best_rr = None
best_ev = -999

for r in results:
    status = "✅ PROFIT" if r['profitable'] else "❌ LOSS"
    print(f"{r['rr']}:1    {r['win_rate']:<12.1%} {r['break_even']:<12.1%} {r['ev_per_trade']:+.3f}R      {status}")

    if r['ev_per_trade'] > best_ev:
        best_ev = r['ev_per_trade']
        best_rr = r['rr']

print("─" * 60)
print(f"\n🏆 Best R:R Ratio: {best_rr}:1 (EV = {best_ev:+.3f}R per trade)")

# Recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if best_rr == 1.0:
    print("\n✅ 1:1 R:R is optimal - high win rate compensates for lower reward")
elif best_rr <= 1.5:
    print(f"\n✅ {best_rr}:1 R:R is optimal - good balance of win rate and reward")
else:
    print(f"\n✅ {best_rr}:1 R:R is optimal - lower win rate but higher reward per win")

print("\nNote: Test with full 200 coins for more reliable results!")
print("=" * 70)
