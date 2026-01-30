# InvestorIQ Technical Analysis Guide
## Comprehensive Documentation for Single Asset Analysis

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Core Analysis Framework](#2-core-analysis-framework)
3. [The Three Pillars of Analysis](#3-the-three-pillars-of-analysis)
4. [Layer 1: Direction Score (Whale Analysis)](#4-layer-1-direction-score)
5. [Layer 2: Squeeze Potential](#5-layer-2-squeeze-potential)
6. [Layer 3: Entry Timing](#6-layer-3-entry-timing)
7. [Flow Story Analysis](#7-flow-story-analysis)
8. [Explosion & Energy Detection](#8-explosion--energy-detection)
9. [ML + Rules Hybrid System](#9-ml--rules-hybrid-system)
10. [Setup Types & Trade Classification](#10-setup-types--trade-classification)
11. [Risk Management](#11-risk-management)
12. [Complete Example Walkthrough](#12-complete-example-walkthrough)
13. [Quick Reference Tables](#13-quick-reference-tables)

---

# 1. Introduction

InvestorIQ is a **predictive trading analysis system** that combines institutional whale positioning data, technical analysis, and machine learning to identify high-probability trade setups **before** they happen.

## Key Philosophy

> **"We position BEFORE the move, not after confirmation."**

Traditional technical analysis waits for price confirmation (breakouts, candlestick patterns). By that time, smart money has already positioned. InvestorIQ detects WHERE smart money is positioning and helps you get in early.

## What Makes This Different

| Traditional TA | InvestorIQ |
|----------------|------------|
| Waits for breakout | Positions before breakout |
| Follows price | Follows smart money |
| Reactive | Predictive |
| Single indicator | Holistic story analysis |
| Fixed rules | Context-aware rules |

---

# 2. Core Analysis Framework

## The Holistic Story Approach

InvestorIQ doesn't check individual thresholds in isolation. Instead, it asks:

> **"What is the COMPLETE PICTURE telling us?"**

Every trade decision combines THREE stories:
1. **Divergence Story** - Where are whales vs retail positioned?
2. **Flow Story** - Is money entering or exiting?
3. **Position Story** - Are we early or late to the move?

Only when all stories **align** do we get high-conviction setups.

## Decision Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA INPUTS                          │
│  Whale %, Retail %, OI Change, Price Change, Position %    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STORY INTERPRETATION                           │
│  Divergence Story → Flow Story → Position Story            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HOLISTIC VERDICT                               │
│  Stories Align? → IDEAL Setup                              │
│  Stories Conflict? → WAIT                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL OUTPUT                                   │
│  Direction + Confidence + Entry Levels                     │
└─────────────────────────────────────────────────────────────┘
```

---

# 3. The Three Pillars of Analysis

## Overview

The Combined Learning section displays three key metrics that form the foundation of every trade decision:

| Pillar | What It Measures | Max Score |
|--------|------------------|-----------|
| **Direction** | Whale conviction & bias | 40 |
| **Squeeze** | Whale vs Retail divergence | 30 |
| **Entry** | Position timing + TA | 30 |

**Total possible score: 100 points**

## Score Interpretation

| Total Score | Signal Strength | Action |
|-------------|-----------------|--------|
| 80-100 | IDEAL | Strong entry, full position |
| 65-79 | STRONG | Good entry, standard position |
| 50-64 | MODERATE | Use limit orders, reduced size |
| 35-49 | WEAK | Wait for better setup |
| 0-34 | AVOID | No trade, conflicting signals |

---

# 4. Layer 1: Direction Score

## What It Measures

The Direction score (out of 40) measures **whale conviction** - how strongly institutional traders are positioned in one direction.

## Calculation

```python
Whale % >= 70%  → 40 points (VERY STRONG BULLISH)
Whale % >= 60%  → 32-36 points (STRONG BULLISH)
Whale % >= 55%  → 26-31 points (LEAN BULLISH)
Whale % 45-55%  → 18-25 points (NEUTRAL - no edge)
Whale % <= 45%  → 26-31 points (LEAN BEARISH)
Whale % <= 40%  → 32-36 points (STRONG BEARISH)
Whale % <= 30%  → 40 points (VERY STRONG BEARISH)
```

## Display Format

```
🐋 LAYER 1: Direction
━━━━━━━━━━━━━━━━━━━━━
BULLISH
HIGH confidence

36/40

🐋 Whales 72% bullish (vs Retail 50%)
```

## Interpretation Examples

### Example 1: Strong Bullish Signal
```
Whale %: 72%
Retail %: 50%
Score: 36/40
Interpretation: "Whales strongly positioned LONG while retail is neutral"
```

### Example 2: Neutral (No Edge)
```
Whale %: 52%
Retail %: 48%
Score: 22/40
Interpretation: "Both whales and retail similarly positioned - no informational advantage"
```

### Example 3: Contrarian Short
```
Whale %: 35%
Retail %: 68%
Score: 34/40 (BEARISH)
Interpretation: "Whales positioned SHORT while retail is heavily LONG - retail trap likely"
```

---

# 5. Layer 2: Squeeze Potential

## What It Measures

The Squeeze score (out of 30) measures the **divergence between whale and retail positioning**. Large divergences indicate potential for explosive moves as one side gets liquidated.

## The Squeeze Concept

When whales are positioned opposite to retail, retail's positions will eventually be **liquidated** (squeezed). This creates momentum in the whale's direction.

```
SQUEEZE = |Whale % - Retail %|

If Whale > Retail → Retail shorts will be squeezed (LONG)
If Whale < Retail → Retail longs will be squeezed (SHORT)
```

## Score Calculation

| Divergence | Label | Score | Meaning |
|------------|-------|-------|---------|
| ≥ +20% | WHALES WAY AHEAD | 28-30 | Massive squeeze potential LONG |
| +15% to +19% | HIGH DIVERGENCE | 24-27 | Strong squeeze potential LONG |
| +8% to +14% | MODERATE | 18-23 | Some edge LONG |
| -7% to +7% | NO EDGE | 10-17 | No clear advantage |
| -8% to -14% | MODERATE | 18-23 | Some edge SHORT |
| -15% to -19% | HIGH DIVERGENCE | 24-27 | Strong squeeze potential SHORT |
| ≤ -20% | RETAIL TRAPPED | 28-30 | Massive squeeze potential SHORT |

## Display Format

```
🔥 LAYER 2: Squeeze
━━━━━━━━━━━━━━━━━━━━━
HIGH
W:72% vs R:50%

20/30

Divergence: +22%
```

## Real Examples

### Example 1: Ideal Long Squeeze Setup
```
Whale %: 72%
Retail %: 50%
Divergence: +22%
Score: 26/30
Interpretation: "Whales way ahead - retail shorts will be liquidated"
```

### Example 2: Retail Trap (Short Setup)
```
Whale %: 38%
Retail %: 75%
Divergence: -37%
Score: 30/30
Interpretation: "Retail massively long, whales short - retail longs will be liquidated"
```

---

# 6. Layer 3: Entry Timing

## What It Measures

The Entry score (out of 30) combines:
1. **Position in Range** (0-100%) - Where price is relative to recent swing high/low
2. **TA Score** - Technical analysis confirmation

## Position Zones

```
Position 0-30%   = EARLY (near support/lows) - BEST for LONGS
Position 30-50%  = MIDDLE (neutral zone) - Use limit orders
Position 50-70%  = LATE (approaching resistance) - Risky for longs
Position 70-100% = CHASING (near highs) - DON'T CHASE
```

## Visual Representation

```
SWING LOW ────────────────────────────────────── SWING HIGH
     │                                                │
     │ EARLY        MIDDLE         LATE      CHASING │
     │ (0-30%)     (30-50%)      (50-70%)   (70-100%)│
     │  ✅ BUY      👀 LIMIT      ⚠️ CAREFUL  ❌ DON'T │
     │                                                │
     └────────────────────────────────────────────────┘
```

## Score Calculation

```python
# Position component (max 15 points for longs)
if position <= 20%:  position_score = 15  # Perfect EARLY entry
elif position <= 35%: position_score = 12  # Good EARLY
elif position <= 50%: position_score = 8   # MIDDLE
elif position <= 70%: position_score = 4   # LATE
else: position_score = 0                   # CHASING

# TA component (max 15 points)
if ta_score >= 70: ta_component = 15  # Strong TA
elif ta_score >= 55: ta_component = 10  # Moderate TA
elif ta_score >= 40: ta_component = 5   # Weak TA
else: ta_component = 0                   # Poor TA

ENTRY_SCORE = position_score + ta_component
```

## Display Format

```
📍 LAYER 3: Entry (TA + Position)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOW
TA: 55 | Pos: EARLY

23/30

Good entry: EARLY (9%) + TA 55
```

## The LATE Override Rule

**Important**: Even if position is LATE (70%+), if price is AT a Bullish Order Block (demand zone), the system allows the trade:

```python
if position >= 70% AND at_bullish_ob:
    # Override LATE restriction - OB provides structural support
    allow_trade = True
    note = "At Bullish OB - structure overrides late position"
```

---

# 7. Flow Story Analysis

## The Four Flow States

The Flow Story combines **Open Interest change** and **Price change** to determine what type of money is flowing:

| OI Change | Price Change | Flow State | Meaning |
|-----------|--------------|------------|---------|
| ↑ Rising | ↑ Rising | NEW LONGS | Fresh buying, bullish continuation |
| ↑ Rising | ↓ Falling | NEW SHORTS* | Fresh selling, bearish continuation |
| ↓ Falling | ↑ Rising | SHORT COVERING | Fake rally - shorts exiting, not new buyers |
| ↓ Falling | ↓ Falling | LONGS EXITING | Liquidation, bearish |

### *Critical Context: Whale Accumulation

When OI rises but price falls, the traditional interpretation is "new shorts entering." However, **if whales are heavily long (65%+)**, this is actually **WHALE ACCUMULATION** - smart money buying the dip!

```python
if oi_up and price_down:
    if whale_pct >= 65:
        # Whales are accumulating, not shorts entering!
        flow = "WHALE_ACCUMULATION"
        direction = "BULLISH"
    else:
        flow = "NEW_SHORTS"
        direction = "BEARISH"
```

## Flow Story Display

```
📈 Open Interest
━━━━━━━━━━━━━━━━━━━━
OI +9.7% + Price -1.7% → WHALE ACCUMULATION (buying dip!)
```

## Practical Examples

### Example 1: Healthy Bullish Move
```
OI: +5.2%
Price: +3.1%
Flow: NEW_LONGS
Interpretation: "Fresh money entering long positions - healthy trend"
```

### Example 2: Fake Rally (Don't Chase!)
```
OI: -8.3%
Price: +4.5%
Flow: SHORT_COVERING
Interpretation: "Price up but OI down = shorts exiting, NOT new buyers. Rally will fade."
```

### Example 3: Whale Accumulation (Buy the Dip)
```
OI: +9.7%
Price: -1.7%
Whale %: 72%
Flow: WHALE_ACCUMULATION
Interpretation: "Whales at 72% long, OI rising = whales adding to longs on dip"
```

---

# 8. Explosion & Energy Detection

## Bollinger Band Squeeze

The system measures how "tight" Bollinger Bands are compared to recent history. Tight bands = compressed energy = explosion imminent.

### BB Squeeze Percentile

```
BB Squeeze 80-100%  = ⚡ LOADED - Explosion imminent!
BB Squeeze 60-79%   = ⚡ BUILDING - Compression forming
BB Squeeze 40-59%   = 📊 NORMAL - Nothing special
BB Squeeze 0-39%    = 📊 RELAXED - Bands wide, no squeeze
```

### Display Format

```
⚡ Market State: COMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                    Explosion Score: 55/100
```

## Explosion Score Components

The Explosion Score (0-100) combines multiple factors:

| Component | Max Points | Description |
|-----------|------------|-------------|
| BB Squeeze | 30 | How tight are Bollinger Bands? |
| OI Building | 25 | Is Open Interest increasing? |
| Whale Alignment | 25 | Are whales positioned for the move? |
| Volume Pattern | 10 | Is volume contracting (pre-explosion)? |
| ATR Pattern | 10 | Is ATR decreasing (calm before storm)? |

## Explosion States

| State | Description | Action |
|-------|-------------|--------|
| **COMPRESSION** | Energy loading, squeeze forming | Prepare to enter |
| **IGNITION** | Explosion started, bands expanding | Prime entry window |
| **LIQUIDITY_CLEAR** | Stops swept, reversal zone | Counter-trade opportunity |
| **DISTRIBUTION** | Whales selling into strength | Exit/avoid longs |
| **ACCUMULATION** | Whales buying on weakness | Look for long entries |

---

# 9. ML + Rules Hybrid System

## How the Hybrid Works

InvestorIQ uses a **hybrid approach** combining:
1. **Rules Engine** - Holistic story analysis (divergence, flow, position)
2. **ML Engine** - Pattern recognition from historical data

```
┌─────────────────┐     ┌─────────────────┐
│   RULES ENGINE  │     │   ML ENGINE     │
│   (Holistic)    │     │   (Heuristic)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   BLEND & VALIDATE    │
         │                       │
         │ If ALIGNED → Boost    │
         │ If CONFLICT → Caution │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   FINAL VERDICT       │
         └───────────────────────┘
```

## ML Confidence Threshold

```python
ML_CONFIDENCE_THRESHOLD = 65%

if ml_confidence >= 65:
    # ML is confident - consider its input
    trust_ml = True
else:
    # ML is uncertain (coin flip) - ignore it
    trust_ml = False
```

## Alignment States

| State | Meaning | Effect |
|-------|---------|--------|
| ✅ ALIGNED | ML and Rules agree | +10 points confidence boost |
| ⚠️ CONFLICT | ML and Rules disagree | Warning, but Rules win for IDEAL setups |
| N/A | ML unavailable | Use Rules only |

## The Rules Always Win for IDEAL Setups

When the holistic analysis shows an **IDEAL setup** (divergence + flow + position all aligned), ML cannot override it:

```python
if setup_type == "IDEAL_LONG":
    if ml_says == "SHORT":
        # ML disagreement noted but DOES NOT block trade
        note = "ML disagrees but holistic analysis is IDEAL"
        final_direction = "LONG"  # Rules win!
```

**Why?** Because for IDEAL setups, the whale/flow/position story is so clear that ML's historical pattern matching shouldn't override it.

---

# 10. Setup Types & Trade Classification

## Setup Type Hierarchy

| Setup Type | Score Range | Description |
|------------|-------------|-------------|
| 🎯 IDEAL_LONG | 75-95 | Perfect long: Whales ahead + New longs + Early position |
| 🎯 IDEAL_SHORT | 75-95 | Perfect short: Retail trapped + Longs exiting + Late position |
| 🚀 PREDICTIVE_LONG | 60-74 | Good long but one element weak |
| 🚀 PREDICTIVE_SHORT | 60-74 | Good short but one element weak |
| 🎯 RETAIL_TRAP_LONG | 65-75 | Retail shorting against whales |
| 🎯 RETAIL_TRAP_SHORT | 65-75 | Retail longing against whales |
| ⚠️ WEAK_LONG | 50-59 | Marginal long, use small size |
| ⚠️ WEAK_SHORT | 50-59 | Marginal short, use small size |
| 🟡 DIVERGENCE_CONFLICT | 40-49 | Mixed signals - WAIT |
| 🟡 NO_EDGE | 30-39 | No whale edge - WAIT |
| 🚨 WHALE_EXIT | 20-30 | Whales exiting - AVOID |
| ❌ DONT_CHASE | 0-30 | Too late - AVOID |

## IDEAL Setup Criteria (Long)

ALL must be true:
- [x] Whale % ≥ 65% (strong long conviction)
- [x] Divergence ≥ +15% (whales way ahead of retail)
- [x] Flow = NEW_LONGS or WHALE_ACCUMULATION (money entering)
- [x] Position ≤ 35% (EARLY entry)
- [x] No whale exit warning (OI not collapsing)

## IDEAL Setup Criteria (Short)

ALL must be true:
- [x] Whale % ≤ 35% (strong short conviction)
- [x] Divergence ≤ -15% (retail trapped long)
- [x] Flow = NEW_SHORTS or LONGS_EXITING (money exiting longs)
- [x] Position ≥ 65% (LATE = good for shorts)
- [x] No short covering (OI not collapsing while price rises)

---

# 11. Risk Management

## Position Sizing Based on Setup Type

| Setup Type | Suggested Position Size |
|------------|-------------------------|
| IDEAL | 100% of intended size |
| PREDICTIVE | 75% of intended size |
| RETAIL_TRAP | 75% of intended size |
| WEAK | 50% of intended size |
| CONFLICT/WAIT | 0% - No trade |

## Stop Loss Placement

Stops are placed at **chart structure**, not arbitrary percentages:

```
LONG Trade:
  SL = Below most recent Swing Low OR Bullish Order Block

SHORT Trade:
  SL = Above most recent Swing High OR Bearish Order Block
```

## Take Profit Strategy

Three-tier profit taking:

| Level | Distance | Allocation | Purpose |
|-------|----------|------------|---------|
| TP1 | 1-2 ATR | 33% | Lock in profit |
| TP2 | 2-3 ATR | 33% | Capture trend |
| TP3 | 3-5 ATR | 34% | Runner for explosions |

### TP3 Runner Conditions

TP3 (the runner) only applies when:
- [x] BB width expanding (volatility increasing)
- [x] OI holding or rising (money staying in)
- [x] Price above VWAP/range midpoint (strength)

If any condition fails, TP3 is capped at TP2.

## Risk:Reward Filtering

Trades are filtered by minimum R:R at TP1:

| TP1 R:R | Color | Action |
|---------|-------|--------|
| ≥ 1.0 | 🟢 Green | Good trade |
| 0.5 - 0.99 | 🟡 Yellow | Marginal, use limit |
| < 0.5 | 🔴 Red | Skip trade |

---

# 12. Complete Example Walkthrough

## Example: SAPIENUSDT Long Setup

### Raw Data
```
Symbol: SAPIENUSDT
Timeframe: 15m
Current Price: $0.4523

Whale Data:
  - Whale Long %: 72%
  - Retail Long %: 50%
  - Divergence: +22%

Open Interest:
  - OI 24h Change: +9.7%
  - Price 24h Change: -1.7%

Position:
  - Position in Range: 7%
  - Swing High: $0.52
  - Swing Low: $0.44

Technical:
  - TA Score: 55
  - BB Squeeze: 99%
  - Explosion Score: 55/100
```

### Step 1: Divergence Story
```
Whale %: 72%
Retail %: 50%
Divergence: 72 - 50 = +22%

Since divergence >= +15%:
  Divergence Story = "WHALES_WAY_AHEAD"
  Direction Edge = "STRONG_BULLISH"
  Description = "Whales way ahead of retail - squeeze potential HIGH"
```

### Step 2: Flow Story
```
OI: +9.7% (UP)
Price: -1.7% (DOWN)
Whale %: 72% (heavily long)

Traditional: OI↑ + Price↓ = "NEW_SHORTS"
But Whale >= 65%, so:
  Flow Story = "WHALE_ACCUMULATION"
  Flow Direction = "BULLISH"
  Description = "Whales buying the dip!"
```

### Step 3: Position Story
```
Position: 7%

Since position <= 30%:
  Position Zone = "EARLY"
  Description = "Near lows - excellent entry for longs"
```

### Step 4: Holistic Verdict
```
Divergence: STRONG_BULLISH ✅
Flow: BULLISH (Accumulation) ✅
Position: EARLY ✅

ALL THREE ALIGNED!

Setup Type = "IDEAL_LONG"
Base Score = 75
```

### Step 5: Energy/Explosion Check
```
BB Squeeze: 99% → LOADED! (+5 points)
Explosion Score: 55 → Building (+3 points)
Energy Loaded: YES

Updated Score: 75 + 5 + 3 = 83
```

### Step 6: ML Check
```
ML Direction: LONG
ML Confidence: 90%
Confidence >= 65% → ML is trusted

ML agrees with Rules → ALIGNED (+10 points)

Final Score: 83 + 10 = 93 (capped at 95)
```

### Final Output
```
═══════════════════════════════════════════════════════════════
🎯 SAPIENUSDT: STRONG BUY (93% predictive score)
═══════════════════════════════════════════════════════════════

Direction: 36/40 | Squeeze: 26/30 | Entry: 23/30 | Explosion: 55/100

UNIFIED ANALYSIS:
🐋 Whales strongly bullish (72%) | Retail lagging at 50% = HIGH squeeze potential |
📈 WHALE ACCUMULATION (buying dip!) | ✅ EARLY entry (7%) | ⚡ BB LOADED (99%) |
💰 Accumulation phase | ✅ ML confirms

Setup Type: 🎯 IDEAL LONG

Entry: $0.4523
Stop Loss: $0.4380 (below swing low)
TP1: $0.4750 (+5.0%, R:R 1.6:1) - Take 33%
TP2: $0.5000 (+10.5%, R:R 3.3:1) - Take 33%
TP3: $0.5200 (+15.0%, R:R 4.7:1) - Runner

Risk: 3.2% | Position Size: 100% (IDEAL setup)
═══════════════════════════════════════════════════════════════
```

---

# 13. Quick Reference Tables

## Whale Positioning Guide

| Whale % | Interpretation | Trade Bias |
|---------|----------------|------------|
| 75%+ | Extremely bullish | STRONG LONG |
| 65-74% | Very bullish | LONG |
| 55-64% | Lean bullish | LEAN LONG |
| 45-54% | Neutral | NO TRADE |
| 35-44% | Lean bearish | LEAN SHORT |
| 25-34% | Very bearish | SHORT |
| <25% | Extremely bearish | STRONG SHORT |

## Divergence Guide

| Divergence | Meaning | Setup |
|------------|---------|-------|
| ≥ +20% | Massive whale lead | IDEAL LONG |
| +15% to +19% | Strong whale lead | PREDICTIVE LONG |
| +8% to +14% | Moderate lead | LEAN LONG |
| -7% to +7% | No edge | NO TRADE |
| -8% to -14% | Moderate retail lead | LEAN SHORT |
| -15% to -19% | Strong retail lead | PREDICTIVE SHORT |
| ≤ -20% | Retail massively trapped | IDEAL SHORT |

## Flow State Guide

| Flow State | What Happened | Action |
|------------|---------------|--------|
| NEW_LONGS | Fresh buying | Go LONG |
| WHALE_ACCUMULATION | Smart money buying dip | Go LONG |
| NEW_SHORTS | Fresh selling | Go SHORT |
| SHORT_COVERING | Fake rally | DON'T CHASE |
| LONGS_EXITING | Capitulation | Go SHORT |
| WHALE_EXIT | Smart money leaving | AVOID |

## Position Zone Guide

| Position % | Zone | For LONGS | For SHORTS |
|------------|------|-----------|------------|
| 0-30% | EARLY | ✅ Perfect | ❌ Too early |
| 30-50% | MIDDLE | 👀 Use limit | 👀 Use limit |
| 50-70% | LATE | ⚠️ Risky | ✅ Good |
| 70-100% | CHASING | ❌ Don't chase | ✅ Perfect |

## BB Squeeze Guide

| Squeeze % | State | Meaning |
|-----------|-------|---------|
| 80-100% | LOADED | Explosion imminent |
| 60-79% | BUILDING | Compression forming |
| 40-59% | NORMAL | Nothing special |
| 0-39% | RELAXED | Wide bands, no squeeze |

## Setup Type Actions

| Setup | Score | Action | Size |
|-------|-------|--------|------|
| 🎯 IDEAL | 75-95 | ENTER NOW | 100% |
| 🚀 PREDICTIVE | 60-74 | Enter with limit | 75% |
| 🎯 RETAIL_TRAP | 65-75 | Enter with limit | 75% |
| ⚠️ WEAK | 50-59 | Small position | 50% |
| 🟡 CONFLICT | 40-49 | WAIT | 0% |
| 🟡 NO_EDGE | 30-39 | WAIT | 0% |
| 🚨 AVOID | 0-30 | NO TRADE | 0% |

---

# Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ATR** | Average True Range - measure of volatility |
| **BB** | Bollinger Bands - volatility indicator |
| **Divergence** | Difference between whale and retail positioning |
| **Flow Story** | Interpretation of OI + Price movement together |
| **HTF** | Higher Time Frame (e.g., 4H for 15m trades) |
| **OB** | Order Block - institutional supply/demand zone |
| **OI** | Open Interest - total open positions |
| **R:R** | Risk:Reward ratio |
| **Squeeze** | Compression followed by explosive move |
| **TA** | Technical Analysis |
| **TP** | Take Profit level |
| **SL** | Stop Loss level |
| **Whale** | Institutional/large trader |

---

# Appendix B: Troubleshooting

## "Why did I get WAIT when ML says LONG?"

The holistic story analysis found a conflict. Common reasons:
1. Flow story is bearish (shorts covering or longs exiting)
2. Position is too late (chasing)
3. No divergence edge (whales and retail similarly positioned)

## "Why is ML showing 55% confidence?"

Low ML confidence means the model has seen mixed outcomes for similar setups. It's essentially saying "I don't know." The system ignores ML when confidence is below 65%.

## "Why didn't explosion boost my score?"

Explosion boosts only apply when:
1. BB Squeeze is actually loaded (80%+)
2. Explosion score is significant (60%+)
3. Direction aligns with whale positioning

## "Why is my R:R showing red?"

The distance to TP1 divided by distance to SL is less than 0.5:1. This is a poor risk:reward trade. Either:
1. Wait for price to come to a better entry
2. Use a tighter stop (if structure allows)
3. Skip the trade

---

*Document Version: BUILD 72*
*Last Updated: January 2025*
*InvestorIQ by Mohammed*
