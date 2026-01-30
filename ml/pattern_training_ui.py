"""
Deep Learning Pattern Detection Training UI
============================================
Streamlit UI for training CNN/LSTM pattern detection models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pattern detection with error handling
HAS_TORCH = False
TORCH_ERROR = None

try:
    from .deep_pattern_detection import (
        DeepPatternTrainer, RuleBasedPatternDetector, CombinedPredictor,
        PatternType, PATTERN_LABELS, PATTERN_DIRECTION, MODE_PATTERNS,
        MODE_SEQUENCE_LENGTH, MARKET_VOLATILITY, HAS_TORCH, TORCH_ERROR
    )
except ImportError:
    try:
        from deep_pattern_detection import (
            DeepPatternTrainer, RuleBasedPatternDetector, CombinedPredictor,
            PatternType, PATTERN_LABELS, PATTERN_DIRECTION, MODE_PATTERNS,
            MODE_SEQUENCE_LENGTH, MARKET_VOLATILITY, HAS_TORCH, TORCH_ERROR
        )
    except Exception as e:
        TORCH_ERROR = f"Could not import pattern detection: {e}"
        # Create minimal fallbacks
        class PatternType:
            pass
        PATTERN_LABELS = []
        PATTERN_DIRECTION = {}
        MODE_PATTERNS = {}
        MODE_SEQUENCE_LENGTH = {'scalp': 50, 'daytrade': 100, 'swing': 150, 'investment': 200}
        MARKET_VOLATILITY = {'crypto': 1.5, 'stocks': 1.0, 'etfs': 0.7}
        RuleBasedPatternDetector = None
        DeepPatternTrainer = None
        CombinedPredictor = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'patterns')

DEFAULT_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
                  'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'LTCUSDT']

DEFAULT_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 
                  'V', 'UNH', 'JNJ', 'WMT', 'PG', 'MA', 'HD']

DEFAULT_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'VWO', 'EEM', 'GLD',
                'SLV', 'TLT', 'XLF', 'XLK', 'XLE']


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (reuse from training_ui)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_crypto_data(symbols, timeframe, days, progress_bar):
    """Fetch crypto data from Binance"""
    try:
        from core.data_fetcher import fetch_binance_klines
    except ImportError:
        st.error("Could not import data_fetcher")
        return pd.DataFrame()
    
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
            limit = min(int((days * 1440) / tf_minutes.get(timeframe, 60)), 1000)
            df = fetch_binance_klines(symbol, timeframe, limit)
            if df is not None and len(df) > 50:
                df['symbol'] = symbol
                all_data.append(df)
        except:
            pass
        progress_bar.progress((i + 1) / len(symbols) * 0.3)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def fetch_stock_data(symbols, timeframe, days, progress_bar):
    """Fetch stock data"""
    try:
        from core.data_fetcher import fetch_stock_data as fetch_stock
    except ImportError:
        st.error("Could not import stock data fetcher")
        return pd.DataFrame()
    
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d', '1w': '1wk'}
            df = fetch_stock(symbol, tf_map.get(timeframe, '1d'), days * 10)
            if df is not None and len(df) > 50:
                df['symbol'] = symbol
                all_data.append(df)
        except:
            pass
        progress_bar.progress((i + 1) / len(symbols) * 0.3)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def get_pattern_model_status():
    """Get status of pattern detection models with accuracy"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    status = {}
    
    for mode in ['scalp', 'daytrade', 'swing', 'investment']:
        for market in ['crypto', 'stock', 'etf']:
            key = f"{mode}_{market}"
            
            # Try multiple naming patterns
            possible_paths = [
                os.path.join(MODEL_DIR, f'pattern_{key}.pkl'),
                os.path.join(MODEL_DIR, f'pattern_{mode}_{market}s.pkl'),  # pattern_swing_stocks.pkl
                os.path.join(MODEL_DIR, f'pattern_{mode}_{market}.pkl'),
            ]
            
            model_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path:
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                # Try to get accuracy from saved model
                accuracy = 0
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        bundle = pickle.load(f)
                        if isinstance(bundle, dict):
                            accuracy = bundle.get('accuracy', 0) or bundle.get('val_accuracy', 0) or bundle.get('best_val_acc', 0) or 0
                            if accuracy and accuracy < 1:
                                accuracy *= 100
                except:
                    pass
                
                status[key] = {
                    'trained': True,
                    'date': mtime.strftime('%Y-%m-%d'),
                    'accuracy': accuracy,
                    'size_kb': os.path.getsize(model_path) / 1024
                }
            else:
                status[key] = {'trained': False, 'date': None, 'accuracy': 0, 'size_kb': 0}
    
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_pattern_model(mode, market, model_type, symbols, timeframe, days):
    """
    Train a pattern detection model.
    
    Returns:
        bool: True if training succeeded, False if failed
    """
    
    st.subheader(f"🧠 Training {mode.upper()} Pattern Model ({market})")
    st.write(f"Model: {model_type.upper()} | Symbols: {len(symbols)} | Timeframe: {timeframe}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check PyTorch
    if not HAS_TORCH:
        st.error("❌ PyTorch not installed!")
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", language="bash")
        st.write("For GPU support:")
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", language="bash")
        return False  # FAILED
    
    # Fetch data
    status_text.write("📡 Fetching historical data...")
    
    if market == 'crypto':
        df = fetch_crypto_data(symbols, timeframe, days, progress_bar)
    else:
        df = fetch_stock_data(symbols, timeframe, days, progress_bar)
    
    if df.empty:
        st.error("No data fetched. Check symbols and connection.")
        return False  # FAILED
    
    st.success(f"✅ Fetched {len(df):,} candles from {df['symbol'].nunique()} symbols")
    
    # Generate training samples
    status_text.write("🔍 Detecting patterns in historical data (rule-based labeling)...")
    progress_bar.progress(0.35)
    
    trainer = DeepPatternTrainer(mode=mode, market=market, model_type=model_type)
    
    all_samples = []
    all_labels = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.reset_index(drop=True)
        
        samples, labels = trainer.generate_training_data(symbol_df)
        all_samples.extend(samples)
        all_labels.extend(labels)
    
    if len(all_samples) < 100:
        st.error(f"Not enough pattern samples. Found {len(all_samples)}, need 100+")
        st.info("Try: More symbols, longer timeframe, or more days of data")
        return False  # FAILED
    
    st.success(f"✅ Generated {len(all_samples):,} training samples")
    
    # Show label distribution
    label_counts = pd.Series(all_labels).value_counts()
    with st.expander("📊 Pattern Distribution", expanded=True):
        st.dataframe(label_counts.to_frame('Count'), use_container_width=True)
        
        # Check for rare patterns
        rare_patterns = label_counts[label_counts < 5].index.tolist()
        if rare_patterns:
            st.warning(f"⚠️ Rare patterns with <5 samples: {rare_patterns}. These may be filtered during training.")
        
        if label_counts.min() < 2:
            st.warning("⚠️ Some patterns have <2 samples. Try: More days (365+), more symbols (20+)")
    
    # Train model
    status_text.write(f"🧠 Training {model_type.upper()} model...")
    progress_bar.progress(0.5)
    
    # Create detailed progress container
    epoch_status = st.empty()
    
    def progress_callback(progress, message):
        progress_bar.progress(0.5 + progress * 0.45)
        status_text.write(f"🔄 {message}")
        epoch_status.info(f"🧠 Deep Learning: {message}")
    
    try:
        import torch
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"🖥️ Training on: {device}")
        
        # Training parameters based on model type
        epochs = {'cnn': 30, 'lstm': 40, 'hybrid': 50}.get(model_type, 40)
        
        metrics = trainer.train(
            samples=all_samples,
            labels=all_labels,
            epochs=epochs,
            batch_size=32,
            lr=0.001,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(1.0)
        status_text.write("✅ Training complete!")
        
        # Show results
        st.subheader("📊 Training Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", f"{metrics['n_samples']:,}")
        col2.metric("Classes", metrics['n_classes'])
        col3.metric("Best Val Acc", f"{metrics['best_val_acc']:.1%}")
        col4.metric("Final Val Acc", f"{metrics['final_val_acc']:.1%}")
        
        # Training history
        with st.expander("📈 Training History"):
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
            
            epochs_range = list(range(1, len(metrics['history']['train_loss']) + 1))
            
            fig.add_trace(
                go.Scatter(x=epochs_range, y=metrics['history']['train_loss'], name='Train Loss'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=metrics['history']['val_loss'], name='Val Loss'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=metrics['history']['val_acc'], name='Val Accuracy'),
                row=1, col=2
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Classes detected
        with st.expander("🏷️ Pattern Classes"):
            st.write(metrics['classes'])
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'pattern_{mode}_{market}.pkl')
        trainer.save(model_path, accuracy=metrics['best_val_acc'])
        st.success(f"💾 Model saved to: {model_path}")
        
        # Performance interpretation
        st.divider()
        acc = metrics['best_val_acc']
        if acc >= 0.7:
            st.success(f"✅ **Excellent!** {acc:.1%} accuracy - Model is reliable for pattern detection")
        elif acc >= 0.55:
            st.warning(f"⚠️ **Good** {acc:.1%} accuracy - Useful but use with traditional ML confirmation")
        else:
            st.error(f"❌ **Poor** {acc:.1%} accuracy - Need more diverse training data")
        
        return True  # SUCCESS
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"Training failed: {error_msg}")
        
        # Give specific advice based on error type
        if "least populated classes" in error_msg or "too few" in error_msg:
            st.warning("💡 **Solution:** Not enough samples per pattern type.")
            st.info("""
            **Try these settings:**
            - Days of History: **365** (max)
            - Symbols: **20+** symbols
            - Timeframe: **15m** or **1h** (more candles = more patterns)
            """)
        elif "Not enough training data" in error_msg:
            st.warning("💡 **Solution:** Need more diverse data to detect patterns.")
            st.info("Increase days to 365 and use more symbols.")
        
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        
        return False  # FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# TEST PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_detection(mode, market):
    """Test pattern detection on current data"""
    
    st.subheader(f"🔍 Test Pattern Detection ({mode}/{market})")
    
    # Load model
    model_path = os.path.join(MODEL_DIR, f'pattern_{mode}_{market}.pkl')
    
    if not os.path.exists(model_path):
        st.warning(f"No trained model found for {mode}/{market}. Train first!")
        
        # Offer rule-based detection
        st.info("Using rule-based detection as fallback...")
        use_rules = True
        detector = RuleBasedPatternDetector(market=market)
    else:
        use_rules = False
        trainer = DeepPatternTrainer(mode=mode, market=market)
        trainer.load(model_path)
        detector = trainer
    
    # Symbol input
    if market == 'crypto':
        symbol = st.selectbox("Select Symbol", DEFAULT_CRYPTO)
    else:
        symbol = st.selectbox("Select Symbol", DEFAULT_STOCKS if market == 'stocks' else DEFAULT_ETFS)
    
    # Fetch data
    if st.button("🔍 Detect Patterns"):
        with st.spinner("Fetching data..."):
            try:
                if market == 'crypto':
                    from core.data_fetcher import fetch_binance_klines
                    df = fetch_binance_klines(symbol, '1h', 200)
                else:
                    from core.data_fetcher import fetch_stock_data as fetch_stock
                    df = fetch_stock(symbol, '1d', 200)
                
                if df is None or len(df) < 50:
                    st.error("Could not fetch enough data")
                    return
                
                df['symbol'] = symbol
                
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return
        
        # Detect patterns
        with st.spinner("Detecting patterns..."):
            if use_rules:
                patterns = detector.detect_all_patterns(df)
                result = patterns[0] if patterns else None
            else:
                result = detector.predict(df)
        
        if result:
            # Display result
            col1, col2, col3 = st.columns(3)
            
            direction_color = {
                'BULLISH': '🟢',
                'BEARISH': '🔴',
                'NEUTRAL': '⚪'
            }
            
            col1.metric("Pattern", result.pattern.value.replace('_', ' ').title())
            col2.metric("Direction", f"{direction_color.get(result.direction, '⚪')} {result.direction}")
            col3.metric("Confidence", f"{result.confidence:.1f}%")
            
            if result.target_pct != 0:
                st.write(f"**Target:** {result.target_pct:+.1f}% | **Stop:** {result.stop_pct:+.1f}%")
            
            # Show chart with pattern
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index if 'DateTime' not in df.columns else df['DateTime'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                )])
                
                # Add pattern annotation
                fig.add_annotation(
                    x=df.index[-1] if 'DateTime' not in df.columns else df['DateTime'].iloc[-1],
                    y=df['High'].iloc[-1],
                    text=f"{result.pattern.value}",
                    showarrow=True,
                    arrowhead=1
                )
                
                fig.update_layout(
                    title=f"{symbol} - {result.pattern.value.replace('_', ' ').title()}",
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except:
                pass
        else:
            st.info("No pattern detected")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_pattern_training_page():
    """Main page for pattern detection training"""
    
    st.title("🎯 Deep Learning Pattern Detection")
    
    st.write("""
    **Train neural networks to detect chart patterns like:**
    - **Reversal:** Double Top (M), Double Bottom (W), Head & Shoulders
    - **Continuation:** Bull/Bear Flags, Triangles, Wedges
    - **Breakout:** Cup & Handle, Ascending/Descending Triangles
    """)
    
    # PyTorch status
    if HAS_TORCH:
        import torch
        col1, col2 = st.columns(2)
        col1.success("✅ PyTorch Installed")
        col2.info(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    else:
        st.error("❌ PyTorch not installed!")
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", language="bash")
        st.write("For GPU:")
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", language="bash")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏋️ Train Model", "🔍 Test Detection", "📊 Model Status"])
    
    with tab1:
        st.subheader("Train Pattern Detection Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.selectbox(
                "Trading Mode",
                ['scalp', 'daytrade', 'swing', 'investment'],
                format_func=lambda x: {
                    'scalp': '⚡ Scalp (Flags, Pennants)',
                    'daytrade': '📊 Day Trade (All patterns)',
                    'swing': '📈 Swing (H&S, Wedges)',
                    'investment': '🏦 Investment (Major patterns)'
                }.get(x, x)
            )
            
            market = st.selectbox(
                "Market",
                ['crypto', 'stocks', 'etfs'],
                format_func=lambda x: {'crypto': '₿ Crypto', 'stocks': '📈 Stocks', 'etfs': '📊 ETFs'}.get(x, x)
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Architecture",
                ['hybrid', 'cnn', 'lstm'],
                format_func=lambda x: {
                    'hybrid': '🔀 Hybrid (CNN + LSTM) - Best',
                    'cnn': '🖼️ CNN (Image-based)',
                    'lstm': '📊 LSTM (Sequence-based)'
                }.get(x, x)
            )
            
            timeframe = st.selectbox(
                "Timeframe",
                ['15m', '1h', '4h', '1d'],
                index=1
            )
        
        # Symbols
        if market == 'crypto':
            default_symbols = DEFAULT_CRYPTO
        elif market == 'stocks':
            default_symbols = DEFAULT_STOCKS
        else:
            default_symbols = DEFAULT_ETFS
        
        symbols = st.multiselect(
            "Symbols",
            default_symbols,
            default=default_symbols[:10]
        )
        
        days = st.slider("Training Days", 30, 365, 180)
        
        # Show patterns for this mode
        with st.expander("🏷️ Patterns for this mode"):
            patterns = MODE_PATTERNS.get(mode, [])
            for p in patterns:
                dir = PATTERN_DIRECTION.get(p, "?")
                emoji = '🟢' if dir == 'BULLISH' else '🔴' if dir == 'BEARISH' else '⚪'
                st.write(f"{emoji} {p.value.replace('_', ' ').title()}")
        
        # Train button
        if st.button("🚀 Train Pattern Model", type="primary", disabled=not HAS_TORCH):
            train_pattern_model(mode, market, model_type, symbols, timeframe, days)
    
    with tab2:
        st.subheader("Test Pattern Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            test_mode = st.selectbox("Mode", ['scalp', 'daytrade', 'swing', 'investment'], key='test_mode')
        with col2:
            test_market = st.selectbox("Market", ['crypto', 'stocks', 'etfs'], key='test_market')
        
        test_pattern_detection(test_mode, test_market)
    
    with tab3:
        st.subheader("Model Status")
        
        status = get_pattern_model_status()
        
        # Create table
        status_data = []
        for key, info in status.items():
            mode, market = key.rsplit('_', 1)
            status_data.append({
                'Mode': mode.upper(),
                'Market': market.upper(),
                'Trained': '✅' if info['trained'] else '❌',
                'Date': info['date'] or '-',
                'Size (KB)': f"{info['size_kb']:.1f}" if info['size_kb'] > 0 else '-'
            })
        
        df_status = pd.DataFrame(status_data)
        st.dataframe(df_status, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    st.set_page_config(page_title="Pattern Detection Training", layout="wide")
    render_pattern_training_page()