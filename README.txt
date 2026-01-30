═══════════════════════════════════════════════════════════════════════════════
INVESTORIQ UPDATE PACKAGE - BUILD 124 (Session Timing + ML Tracking)
═══════════════════════════════════════════════════════════════════════════════

INSTALLATION:
Copy files to their respective folders, restart Streamlit

FILE PLACEMENT:
├── app.py                  → Root folder
├── core/
│   ├── liquidity.py        → core/ (NEW!)
│   ├── unified_scoring.py  → core/
│   └── trade_manager.py    → core/
├── ml/
│   └── ml_engine.py        → ml/
└── utils/
    ├── formatters.py       → utils/
    └── trade_storage.py    → utils/ (UPDATED!)

═══════════════════════════════════════════════════════════════════════════════
NEW IN BUILD 124: SESSION TIMING WARNING
═══════════════════════════════════════════════════════════════════════════════

When ETA extends beyond current session, you'll now see:

┌─────────────────────────────────────────────────────────────────────────────┐
│ ⏰ Session Timing Note                                                      │
│    ETA ~5.2h but ASIA ends in ~3.0h                                         │
│    Trade may close during LONDON session • WR is based on ENTRY session     │
└─────────────────────────────────────────────────────────────────────────────┘

EXPLANATION:
- The 69% WR measures: "When you ENTER during Asia, 69% hit TP1"
- It doesn't matter when the trade CLOSES
- But knowing the trade will cross sessions is useful context
- Different sessions have different volatility

═══════════════════════════════════════════════════════════════════════════════
OTHER FEATURES (from Build 123):
═══════════════════════════════════════════════════════════════════════════════

1. ML DATA STORED IN TRADES:
   - ml_direction, ml_confidence, ml_eta_candles
   - rules_direction, has_ml_conflict
   - liquidity_tier, liquidity_volume_24h

2. TP HIT TIME TRACKING:
   - tp1_hit_time, tp2_hit_time, tp3_hit_time, sl_hit_time

3. TRADE MONITOR WARNINGS:
   - ML vs Rules conflict warning
   - Duration warning (if 2x+ slower than expected)
   - Liquidity tier warning

4. PERFORMANCE TAB - ML ANALYTICS:
   - Win rates by ML alignment
   - Average time to TP by category
   - Slow trades list
   - Performance by liquidity tier

═══════════════════════════════════════════════════════════════════════════════
