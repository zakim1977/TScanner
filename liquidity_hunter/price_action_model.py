"""
Price Action Model - Pure Structure-Based Direction Prediction

This model predicts market direction based on price action structure alone,
WITHOUT relying on sweep detection. The idea is:

1. Train on pure price structure (lower highs, higher lows, BOS, impulse/correction)
2. Use sweeps only as ENTRY TRIGGERS, not for ML training
3. Cleaner signal without the noise of failed sweeps

Flow:
  Price Action Model: "Structure is BEARISH" → Wait for sweep of HIGH → SKIP LONG
  Price Action Model: "Structure is BULLISH" → Wait for sweep of LOW → ENTER LONG

This separates:
  - WHAT to trade (price action model decides direction)
  - WHEN to trade (sweep detection times the entry)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import random

# Model path
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PA_MODEL_PATH = os.path.join(MODEL_DIR, 'price_action_model.pkl')

# Target: Predict if price will go UP or DOWN in next N candles
FORWARD_CANDLES = 10  # Look ahead 10 candles (40 hours on 4h timeframe)
MIN_MOVE_ATR = 1.0    # Minimum move of 1 ATR to count as directional


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION - Pure Price Action
# ═══════════════════════════════════════════════════════════════════════════════

def extract_price_action_features(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    """
    Extract pure price action features from OHLCV data.
    No sweep detection, no whale data - just structure.
    """
    if df is None or len(df) < lookback + 10:
        return None

    try:
        df.columns = [c.lower() for c in df.columns]

        current_price = float(df['close'].iloc[-1])
        current_high = float(df['high'].iloc[-1])
        current_low = float(df['low'].iloc[-1])

        # ATR calculation
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        if atr <= 0:
            return None

        # ═══════════════════════════════════════════════════════════════
        # STRUCTURE FEATURES
        # ═══════════════════════════════════════════════════════════════

        # Get recent highs and lows for structure analysis
        recent_df = df.iloc[-lookback:]
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        closes = recent_df['close'].values

        # Find swing points (simplified)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(recent_df) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))

        # Structure analysis
        structure_bullish = 0  # Higher highs, higher lows
        structure_bearish = 0  # Lower highs, lower lows
        structure_higher_high = 0
        structure_higher_low = 0
        structure_lower_high = 0
        structure_lower_low = 0

        if len(swing_highs) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1]:
                structure_higher_high = 1
            else:
                structure_lower_high = 1

        if len(swing_lows) >= 2:
            if swing_lows[-1][1] > swing_lows[-2][1]:
                structure_higher_low = 1
            else:
                structure_lower_low = 1

        # Bullish structure: HH + HL
        if structure_higher_high and structure_higher_low:
            structure_bullish = 1
        # Bearish structure: LH + LL
        if structure_lower_high and structure_lower_low:
            structure_bearish = 1

        # ═══════════════════════════════════════════════════════════════
        # TREND FEATURES
        # ═══════════════════════════════════════════════════════════════

        # Moving averages
        ma_10 = df['close'].rolling(10).mean().iloc[-1]
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20

        # Price position relative to MAs
        above_ma10 = 1 if current_price > ma_10 else 0
        above_ma20 = 1 if current_price > ma_20 else 0
        above_ma50 = 1 if current_price > ma_50 else 0

        # MA alignment (bullish: price > 10 > 20 > 50)
        ma_bullish_aligned = 1 if (current_price > ma_10 > ma_20 > ma_50) else 0
        ma_bearish_aligned = 1 if (current_price < ma_10 < ma_20 < ma_50) else 0

        # Trend strength: distance from MA20
        trend_strength = (current_price - ma_20) / atr if atr > 0 else 0

        # ═══════════════════════════════════════════════════════════════
        # MOMENTUM FEATURES
        # ═══════════════════════════════════════════════════════════════

        # Price change over different periods
        price_change_5 = (current_price - closes[-5]) / atr if len(closes) >= 5 else 0
        price_change_10 = (current_price - closes[-10]) / atr if len(closes) >= 10 else 0
        price_change_20 = (current_price - closes[0]) / atr

        # RSI (simplified)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        rsi_oversold = 1 if rsi_value < 30 else 0
        rsi_overbought = 1 if rsi_value > 70 else 0

        # ═══════════════════════════════════════════════════════════════
        # IMPULSE vs CORRECTION
        # ═══════════════════════════════════════════════════════════════

        # Candle size analysis
        candle_ranges = (recent_df['high'] - recent_df['low']).values
        avg_range = candle_ranges.mean()

        # Recent candles expanding or contracting?
        first_half_avg = candle_ranges[:len(candle_ranges)//2].mean()
        second_half_avg = candle_ranges[len(candle_ranges)//2:].mean()

        is_expanding = 1 if second_half_avg > first_half_avg * 1.2 else 0
        is_contracting = 1 if second_half_avg < first_half_avg * 0.8 else 0

        # Impulse: large directional candles
        # Correction: small choppy candles
        recent_direction = 1 if closes[-1] > closes[-5] else -1
        candle_direction_alignment = sum(1 for i in range(-5, 0) if (closes[i] - closes[i-1]) * recent_direction > 0)

        is_impulse = 1 if is_expanding and candle_direction_alignment >= 4 else 0
        is_correction = 1 if is_contracting and candle_direction_alignment <= 2 else 0

        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY REGIME
        # ═══════════════════════════════════════════════════════════════

        atr_20 = tr.rolling(20).mean().iloc[-1]
        atr_50 = tr.rolling(50).mean().iloc[-1] if len(df) >= 50 else atr_20
        volatility_ratio = atr_20 / atr_50 if atr_50 > 0 else 1.0

        high_volatility = 1 if volatility_ratio > 1.3 else 0
        low_volatility = 1 if volatility_ratio < 0.7 else 0

        # ═══════════════════════════════════════════════════════════════
        # VOLUME FEATURES (if available)
        # ═══════════════════════════════════════════════════════════════

        volume_ratio = 1.0
        volume_increasing = 0
        if 'volume' in df.columns:
            avg_vol = df['volume'].iloc[-20:].mean()
            recent_vol = df['volume'].iloc[-5:].mean()
            volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            volume_increasing = 1 if volume_ratio > 1.2 else 0

        return {
            # Structure
            'structure_bullish': structure_bullish,
            'structure_bearish': structure_bearish,
            'structure_higher_high': structure_higher_high,
            'structure_higher_low': structure_higher_low,
            'structure_lower_high': structure_lower_high,
            'structure_lower_low': structure_lower_low,

            # Trend
            'above_ma10': above_ma10,
            'above_ma20': above_ma20,
            'above_ma50': above_ma50,
            'ma_bullish_aligned': ma_bullish_aligned,
            'ma_bearish_aligned': ma_bearish_aligned,
            'trend_strength': round(trend_strength, 4),

            # Momentum
            'price_change_5': round(price_change_5, 4),
            'price_change_10': round(price_change_10, 4),
            'price_change_20': round(price_change_20, 4),
            'rsi_value': round(rsi_value, 2),
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,

            # Impulse/Correction
            'is_expanding': is_expanding,
            'is_contracting': is_contracting,
            'is_impulse': is_impulse,
            'is_correction': is_correction,

            # Volatility
            'volatility_ratio': round(volatility_ratio, 4),
            'high_volatility': high_volatility,
            'low_volatility': low_volatility,

            # Volume
            'volume_ratio': round(volume_ratio, 4),
            'volume_increasing': volume_increasing,

            # Raw values for context
            'atr': round(atr, 6),
            'current_price': round(current_price, 6),
        }

    except Exception as e:
        print(f"[PA_MODEL] Feature extraction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LABELING - What happened next?
# ═══════════════════════════════════════════════════════════════════════════════

def label_direction(df: pd.DataFrame, idx: int, atr: float, forward_candles: int = 10) -> Optional[Dict]:
    """
    Label what direction price moved after this point.

    Returns:
        direction: 'LONG' if price went up significantly, 'SHORT' if down
        magnitude: How far price moved (in ATR)
    """
    if idx + forward_candles >= len(df):
        return None

    try:
        entry_price = float(df['close'].iloc[idx])
        future_df = df.iloc[idx+1:idx+1+forward_candles]

        max_high = future_df['high'].max()
        min_low = future_df['low'].min()
        final_close = float(df['close'].iloc[idx + forward_candles])

        up_move = (max_high - entry_price) / atr
        down_move = (entry_price - min_low) / atr
        net_move = (final_close - entry_price) / atr

        # Determine direction based on which move was larger
        if up_move > down_move and up_move >= MIN_MOVE_ATR:
            direction = 'LONG'
            won = 1
        elif down_move > up_move and down_move >= MIN_MOVE_ATR:
            direction = 'SHORT'
            won = 1
        else:
            # No clear direction - could label as 'NEUTRAL' or skip
            direction = 'LONG' if net_move > 0 else 'SHORT'
            won = 0  # Not a clear winner

        return {
            'direction': direction,
            'won': won,
            'up_move_atr': round(up_move, 4),
            'down_move_atr': round(down_move, 4),
            'net_move_atr': round(net_move, 4),
        }

    except Exception as e:
        print(f"[PA_MODEL] Labeling error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pa_samples(df: pd.DataFrame, symbol: str, forward_candles: int = 10) -> List[Dict]:
    """
    Generate price action samples for training.
    One sample per candle (with spacing to avoid overlap).
    """
    samples = []

    if df is None or len(df) < 100:
        return samples

    df.columns = [c.lower() for c in df.columns]

    # Calculate ATR once
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()

    # Sample every 5 candles to avoid overlap
    sample_spacing = 5

    for i in range(50, len(df) - forward_candles - 1, sample_spacing):
        atr = float(atr_series.iloc[i])
        if atr <= 0:
            continue

        # Get historical data up to this point
        hist_df = df.iloc[max(0, i-50):i+1]

        # Extract features
        features = extract_price_action_features(hist_df)
        if features is None:
            continue

        # Label outcome
        outcome = label_direction(df, i, atr, forward_candles)
        if outcome is None:
            continue

        sample = {
            'symbol': symbol,
            'sample_idx': i,
            **features,
            **outcome
        }
        samples.append(sample)

    print(f"[PA_MODEL] {symbol}: Generated {len(samples)} price action samples")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PriceActionModel:
    """
    Pure price action model for direction prediction.
    """

    FEATURE_COLUMNS = [
        # Structure
        'structure_bullish',
        'structure_bearish',
        'structure_higher_high',
        'structure_higher_low',
        'structure_lower_high',
        'structure_lower_low',

        # Trend
        'above_ma10',
        'above_ma20',
        'above_ma50',
        'ma_bullish_aligned',
        'ma_bearish_aligned',
        'trend_strength',

        # Momentum
        'price_change_5',
        'price_change_10',
        'price_change_20',
        'rsi_value',
        'rsi_oversold',
        'rsi_overbought',

        # Impulse/Correction
        'is_expanding',
        'is_contracting',
        'is_impulse',
        'is_correction',

        # Volatility
        'volatility_ratio',
        'high_volatility',
        'low_volatility',

        # Volume
        'volume_ratio',
        'volume_increasing',
    ]

    def __init__(self, model_path: str = None):
        self.model = None
        self.metrics = {}
        self.is_trained = False
        self.model_path = model_path or PA_MODEL_PATH

    def train(self, samples: List[Dict]) -> Dict:
        """Train the price action model with comprehensive metrics."""
        if len(samples) < 100:
            return {'error': f'Need at least 100 samples, got {len(samples)}'}

        TARGET_RR = 2.0  # Risk:Reward ratio for ROI calculations

        print(f"\n{'='*60}")
        print(f"[PA_MODEL] Training on {len(samples)} samples")
        print(f"[PA_MODEL] Strategy: Pure Price Action @ {TARGET_RR}:1 R:R")
        print(f"{'='*60}")

        # Convert to DataFrame
        df = pd.DataFrame(samples)

        # Show direction distribution with win rates
        long_samples = df[df['direction'] == 'LONG']
        short_samples = df[df['direction'] == 'SHORT']
        long_wr = long_samples['won'].mean() * 100 if len(long_samples) > 0 else 0
        short_wr = short_samples['won'].mean() * 100 if len(short_samples) > 0 else 0

        print(f"[PA_MODEL] Direction split: {len(long_samples)} LONG ({long_wr:.1f}% win), {len(short_samples)} SHORT ({short_wr:.1f}% win)")
        print(f"[PA_MODEL] Base win rate: {df['won'].mean()*100:.1f}%")

        # Prepare features
        available_features = [f for f in self.FEATURE_COLUMNS if f in df.columns]
        X = df[available_features].fillna(0)
        y = df['won'].astype(int)  # 1 = WIN, 0 = LOSS (predict trade outcome, not direction)

        print(f"[PA_MODEL] Features: {len(available_features)}")

        # Time-based split (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_df = df.iloc[split_idx:].copy()

        print(f"[PA_MODEL] ⏰ TIME-BASED SPLIT: Train on older {len(X_train)} samples, Test on newer {len(X_test)} samples")

        # Train multiple models and compare
        try:
            from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            }

            print(f"\n{'─'*60}")
            print("MODEL COMPARISON (5-Fold Cross-Validation)")
            print(f"{'─'*60}")
            print(f"{'Model':<25} {'CV Acc':<10} {'CV F1':<10} {'Std':<8}")
            print(f"{'─'*60}")

            best_model = None
            best_cv_score = 0
            best_name = ''

            for name, model in models.items():
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

                if cv_scores.mean() > best_cv_score:
                    best_cv_score = cv_scores.mean()
                    best_model = model
                    best_name = name

                print(f"{name:<25} {cv_scores.mean():.1%}     {cv_f1.mean():.1%}     {cv_scores.std():.1%}")

            print(f"{'─'*60}")
            print(f"🏆 Best Model: {best_name}")
            print(f"{'─'*60}")

            # Train best model on full training set
            self.model = best_model
            self.model.fit(X_train, y_train)

            # Evaluate
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            test_df['prob_win'] = y_prob

            # Calculate win rates at different confidence levels
            base_wr = test_df['won'].mean()

            high_conf = test_df[test_df['prob_win'] > 0.5]
            high_conf_wr = high_conf['won'].mean() if len(high_conf) > 0 else 0

            very_high_conf = test_df[test_df['prob_win'] > 0.6]
            very_high_wr = very_high_conf['won'].mean() if len(very_high_conf) > 0 else 0

            # ROI calculations at 2:1 R:R
            # EV per trade = (win_rate * reward) - (loss_rate * risk) = win_rate * 2 - (1 - win_rate) * 1
            base_roi = (base_wr * TARGET_RR - (1 - base_wr)) * 100
            high_roi = (high_conf_wr * TARGET_RR - (1 - high_conf_wr)) * 100 if len(high_conf) > 0 else 0
            very_high_roi = (very_high_wr * TARGET_RR - (1 - very_high_wr)) * 100 if len(very_high_conf) > 0 else 0

            print(f"\n{'─'*60}")
            print(f"WIN RATES & ROI @ {TARGET_RR}:1 R:R")
            print(f"{'─'*60}")
            print(f"Base Win Rate:      {base_wr:.1%} ({len(test_df)} test) → ROI: {base_roi:+.1f}%")
            print(f"High Conf (>50%):   {high_conf_wr:.1%} ({len(high_conf)} test) → ROI: {high_roi:+.1f}%")
            print(f"Very High (>60%):   {very_high_wr:.1%} ({len(very_high_conf)} test) → ROI: {very_high_roi:+.1f}%")

            # LONG vs SHORT breakdown
            print(f"\n{'─'*60}")
            print(f"📊 LONG vs SHORT BREAKDOWN (Test Set at >50% confidence)")
            print(f"{'─'*60}")

            for direction in ['LONG', 'SHORT']:
                dir_df = high_conf[high_conf['direction'] == direction]
                if len(dir_df) > 0:
                    dir_wr = dir_df['won'].mean()
                    dir_roi = (dir_wr * TARGET_RR - (1 - dir_wr)) * 100
                    emoji = '🟢' if direction == 'LONG' else '🔴'
                    print(f"{emoji} {direction}: {dir_wr:.1%} win rate ({len(dir_df)} trades) → ROI: {dir_roi:+.1f}%")

            # Feature importance
            importance = sorted(zip(available_features, self.model.feature_importances_),
                              key=lambda x: x[1], reverse=True)
            print(f"\n[PA_MODEL] Top 10 Features:")
            for feat, imp in importance[:10]:
                bar = '█' * int(imp * 50)
                print(f"  {feat}: {imp:.4f} {bar}")

            # Trade frequency projection
            training_days = len(samples) / (len(df['symbol'].unique()) if 'symbol' in df.columns else 1) / 6
            very_high_per_month = len(very_high_conf) / len(test_df) * 30 * 6 if len(test_df) > 0 else 0
            monthly_roi = very_high_roi * min(very_high_per_month / 100, 0.1)  # Cap at 10 trades/month

            print(f"\nMonthly ROI Projection (at {TARGET_RR}:1 R:R, 1% risk per trade):")
            print(f"  Very High trades/month: ~{very_high_per_month:.1f}")
            print(f"  EV per Very High trade: {very_high_roi/100:.2f}% of account")
            print(f"  Monthly (Very High): +{monthly_roi:.1f}%")

            self.metrics = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'samples': len(samples),
                'total_samples': len(samples),
                'features': len(available_features),
                'base_win_rate': base_wr,
                'high_conf_win_rate': high_conf_wr,
                'very_high_win_rate': very_high_wr,
                'high_conf_trades': len(high_conf),
                'very_high_trades': len(very_high_conf),
                'base_roi': base_roi,
                'high_conf_roi': high_roi,
                'very_high_roi': very_high_roi,
                'monthly_roi_very_high': monthly_roi,
                'very_high_per_month': very_high_per_month,
                'best_model': best_name,
                'top_features': importance[:15],
                'trained_at': datetime.now().isoformat(),
            }

            self.is_trained = True
            self._save()

            return self.metrics

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def predict(self, features: Dict) -> Dict:
        """Predict direction from features."""
        if not self.is_trained or self.model is None:
            return {'error': 'Model not trained'}

        try:
            # Prepare feature vector
            X = pd.DataFrame([features])[self.FEATURE_COLUMNS].fillna(0)

            prob = self.model.predict_proba(X)[0]
            prob_long = prob[1]
            prob_short = prob[0]

            if prob_long > 0.6:
                direction = 'LONG'
                confidence = 'HIGH'
            elif prob_short > 0.6:
                direction = 'SHORT'
                confidence = 'HIGH'
            elif prob_long > 0.55:
                direction = 'LONG'
                confidence = 'MEDIUM'
            elif prob_short > 0.55:
                direction = 'SHORT'
                confidence = 'MEDIUM'
            else:
                direction = 'NEUTRAL'
                confidence = 'LOW'

            return {
                'direction': direction,
                'confidence': confidence,
                'prob_long': round(prob_long, 4),
                'prob_short': round(prob_short, 4),
            }

        except Exception as e:
            return {'error': str(e)}

    def _save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics,
                'feature_columns': self.FEATURE_COLUMNS,
            }, f)
        print(f"[PA_MODEL] Saved to {self.model_path}")

    def load(self) -> bool:
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            return False

        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.metrics = data.get('metrics', {})
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[PA_MODEL] Load error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION (called from UI)
# ═══════════════════════════════════════════════════════════════════════════════

def train_price_action_model(symbols: List[str], days: int = 365) -> Dict:
    """
    Train the price action model on given symbols.

    Args:
        symbols: List of symbols to train on
        days: Number of days of data to use
    """
    from core.data_fetcher import fetch_binance_klines

    all_samples = []

    for symbol in symbols:
        print(f"\n[PA_MODEL] Fetching {symbol}...")

        # Fetch 4h data
        limit = days * 6  # 6 candles per day for 4h
        df = fetch_binance_klines(symbol, '4h', limit=limit)

        if df is None or len(df) < 100:
            print(f"[PA_MODEL] {symbol}: Insufficient data")
            continue

        samples = generate_pa_samples(df, symbol, forward_candles=FORWARD_CANDLES)
        all_samples.extend(samples)

    print(f"\n[PA_MODEL] Total samples: {len(all_samples)}")

    if len(all_samples) < 100:
        return {'error': 'Insufficient samples'}

    model = PriceActionModel()
    return model.train(all_samples)


def get_price_action_model() -> PriceActionModel:
    """Get the price action model (load if exists)."""
    model = PriceActionModel()
    model.load()
    return model


def get_pa_model_status() -> Dict:
    """Get price action model status."""
    model = get_price_action_model()

    if not model.is_trained:
        return {
            'trained': False,
            'message': 'Model not trained yet'
        }

    return {
        'trained': True,
        'metrics': model.metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_price_action_prediction(df: pd.DataFrame) -> Dict:
    """
    Get price action prediction for current market state.

    Usage:
        prediction = get_price_action_prediction(df)
        if prediction['direction'] == 'LONG' and prediction['confidence'] == 'HIGH':
            # Wait for sweep of LOW to enter LONG
        elif prediction['direction'] == 'SHORT' and prediction['confidence'] == 'HIGH':
            # Wait for sweep of HIGH to enter SHORT
    """
    model = get_price_action_model()

    if not model.is_trained:
        return {
            'direction': 'NEUTRAL',
            'confidence': 'NONE',
            'error': 'Model not trained'
        }

    features = extract_price_action_features(df)
    if features is None:
        return {
            'direction': 'NEUTRAL',
            'confidence': 'NONE',
            'error': 'Feature extraction failed'
        }

    return model.predict(features)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Quick test
    print("Testing Price Action Model...")

    # Would need to import data fetcher and run training
    # train_price_action_model(['BTCUSDT', 'ETHUSDT'], days=180)
