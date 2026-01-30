"""
ML Training UI Page
====================

Provides UI for training ML models directly from the app.
Users can train mode-specific models with a button click.
"""

import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, Optional
import threading
import time

# Path to models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def get_model_status() -> Dict:
    """Get status of all trained models."""
    modes = ['scalp', 'daytrade', 'swing', 'investment']
    status = {}
    
    # Crypto models
    for mode in modes:
        meta_path = os.path.join(MODEL_DIR, f'metadata_{mode}.json')
        dir_model_path = os.path.join(MODEL_DIR, f'direction_model_{mode}.pkl')
        
        if os.path.exists(meta_path) and os.path.exists(dir_model_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            status[mode] = {
                'trained': True,
                'trained_at': metadata.get('trained_at', 'Unknown'),
                'n_samples': metadata.get('metrics', {}).get('n_samples', 0),
                'f1_score': metadata.get('metrics', {}).get('direction_f1', 0),
                'eta_mae': metadata.get('metrics', {}).get('eta_mae', 0),
                'sl_mae': metadata.get('metrics', {}).get('sl_mae', 0),
            }
        else:
            status[mode] = {'trained': False}
    
    # Stock/ETF models
    for asset_type in ['stock', 'etf']:
        for mode in ['daytrade', 'swing', 'investment']:
            key = f"{mode}_{asset_type}"
            meta_path = os.path.join(MODEL_DIR, f'metadata_{key}.json')
            dir_model_path = os.path.join(MODEL_DIR, f'direction_model_{key}.pkl')
            
            if os.path.exists(meta_path) and os.path.exists(dir_model_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                status[key] = {
                    'trained': True,
                    'trained_at': metadata.get('trained_at', 'Unknown'),
                    'n_samples': metadata.get('metrics', {}).get('n_samples', 0),
                    'f1_score': metadata.get('metrics', {}).get('direction_f1', 0),
                    'eta_mae': metadata.get('metrics', {}).get('eta_mae', 0),
                    'sl_mae': metadata.get('metrics', {}).get('sl_mae', 0),
                }
            else:
                status[key] = {'trained': False}
    
    # Also check general model
    general_meta = os.path.join(MODEL_DIR, 'training_metadata.json')
    if os.path.exists(general_meta):
        with open(general_meta, 'r') as f:
            metadata = json.load(f)
        status['general'] = {
            'trained': True,
            'trained_at': metadata.get('trained_at', 'Unknown'),
            'n_samples': metadata.get('n_samples', 0),
            'best_model': metadata.get('best_model', 'Unknown'),
            'f1_score': metadata.get('best_f1', 0),
        }
    else:
        status['general'] = {'trained': False}
    
    return status


def render_training_page():
    """Render the ML Training page."""
    
    st.title("🤖 ML Model Training")
    st.markdown("Train machine learning models on historical market data")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CURRENT STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("📊 Current Model Status")
    
    status = get_model_status()
    
    # Mode-specific models
    cols = st.columns(4)
    mode_labels = {
        'scalp': ('⚡ Scalp', '1m/5m'),
        'daytrade': ('📈 Day Trade', '15m/1h'),
        'swing': ('🌊 Swing', '4h/1d'),
        'investment': ('💎 Investment', '1d/1w')
    }
    
    for i, (mode, (label, timeframes)) in enumerate(mode_labels.items()):
        with cols[i]:
            mode_status = status.get(mode, {})
            if mode_status.get('trained'):
                st.success(f"**{label}**")
                st.caption(f"Timeframes: {timeframes}")
                st.metric("F1 Score", f"{mode_status['f1_score']:.1%}")
                st.caption(f"Samples: {mode_status['n_samples']:,}")
                
                # Parse and format date
                trained_at = mode_status.get('trained_at', '')
                if trained_at:
                    try:
                        dt = datetime.fromisoformat(trained_at)
                        st.caption(f"Trained: {dt.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Trained: {trained_at[:10]}")
            else:
                st.error(f"**{label}**")
                st.caption(f"Timeframes: {timeframes}")
                st.markdown("❌ Not trained")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days = st.slider(
            "📅 Days of History",
            min_value=30,
            max_value=730,  # 2 years max
            value=90,
            step=30,
            help="More days = more training data. 90 days for Scalp/DayTrade, 365+ for Investment"
        )
        
        # Show recommendation based on days
        if days < 60:
            st.caption("⚡ Quick training - good for Scalp")
        elif days <= 180:
            st.caption("📊 Standard - good for DayTrade/Swing")
        else:
            st.caption("📈 Extended - recommended for Investment mode")
    
    with col2:
        symbols_option = st.selectbox(
            "🪙 Symbols",
            options=[
                "🎯 Auto: Top 30 by Volume (Recommended)",
                "🚀 Auto: Top 50 by Volume",
                "📋 Default (10 symbols)",
                "📋 Extended (20 symbols)",
                "⚡ Quick: BTC & ETH only"
            ],
            help="Auto mode fetches currently most active futures pairs - best for scanning different coins!"
        )
    
    # Symbol lists
    DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                       'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']
    
    EXTENDED_SYMBOLS = DEFAULT_SYMBOLS + ['DOTUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 
                                          'APTUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 
                                          'FILUSDT', 'INJUSDT']
    
    BTC_ETH_ONLY = ['BTCUSDT', 'ETHUSDT']
    
    # Determine symbols based on selection
    use_dynamic = False
    dynamic_count = 30
    
    if "Auto: Top 30" in symbols_option:
        use_dynamic = True
        dynamic_count = 30
        symbols = None  # Will fetch dynamically
        st.caption("📊 Will fetch top 30 futures pairs by 24h volume")
    elif "Auto: Top 50" in symbols_option:
        use_dynamic = True
        dynamic_count = 50
        symbols = None
        st.caption("📊 Will fetch top 50 futures pairs by 24h volume")
    elif "Default" in symbols_option:
        symbols = DEFAULT_SYMBOLS
        st.caption(f"Will train on: {', '.join(symbols[:5])}...")
    elif "Extended" in symbols_option:
        symbols = EXTENDED_SYMBOLS
        st.caption(f"Will train on: {', '.join(symbols[:5])}...")
    else:
        symbols = BTC_ETH_ONLY
        st.caption("⚡ Quick training on BTC & ETH only")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING BUTTONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("🚀 Train Models")
    
    # Initialize session state for training status
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'training_log' not in st.session_state:
        st.session_state.training_log = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Train Individual Mode:**")
        
        mode_cols = st.columns(2)
        
        with mode_cols[0]:
            if st.button("⚡ Train Scalp", use_container_width=True, 
                        disabled=st.session_state.training_in_progress):
                train_mode('scalp', symbols, days, use_dynamic, dynamic_count)
            
            if st.button("🌊 Train Swing", use_container_width=True,
                        disabled=st.session_state.training_in_progress):
                train_mode('swing', symbols, days, use_dynamic, dynamic_count)
        
        with mode_cols[1]:
            if st.button("📈 Train DayTrade", use_container_width=True,
                        disabled=st.session_state.training_in_progress):
                train_mode('daytrade', symbols, days, use_dynamic, dynamic_count)
            
            if st.button("💎 Train Investment", use_container_width=True,
                        disabled=st.session_state.training_in_progress):
                train_mode('investment', symbols, days, use_dynamic, dynamic_count)
    
    with col2:
        st.markdown("**Train All at Once:**")
        
        if st.button("🎯 TRAIN ALL MODES", use_container_width=True, type="primary",
                    disabled=st.session_state.training_in_progress):
            train_all_modes(symbols, days, use_dynamic, dynamic_count)
        
        st.caption("⏱️ Estimated time: 5-15 minutes depending on settings")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING LOG
    # ═══════════════════════════════════════════════════════════════════════════
    
    if st.session_state.training_log:
        st.divider()
        st.subheader("📝 Training Log")
        
        log_container = st.container()
        with log_container:
            for log_entry in st.session_state.training_log[-20:]:  # Last 20 entries
                st.text(log_entry)
        
        if st.button("🗑️ Clear Log"):
            st.session_state.training_log = []
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STOCK/ETF TRAINING
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.divider()
    st.subheader("📈 Stock & ETF Training")
    st.caption("Train ML models on stocks and ETFs")
    
    # Stock/ETF Model Status - Now includes DayTrade!
    stock_labels = {
        'daytrade_stock': ('📊 DayTrade (Stocks)', '15m/1h data'),
        'swing_stock': ('🌊 Swing (Stocks)', '4h/1d data'),
        'investment_stock': ('💎 Investment (Stocks)', '1d/1w data'),
    }
    etf_labels = {
        'daytrade_etf': ('📊 DayTrade (ETFs)', '15m/1h data'),
        'swing_etf': ('🌊 Swing (ETFs)', '4h/1d data'),
        'investment_etf': ('💎 Investment (ETFs)', '1d/1w data'),
    }
    
    status = get_model_status()
    
    # Stock models
    st.markdown("**🏢 Stock Models:**")
    stock_cols = st.columns(3)
    for i, (mode_key, (label, desc)) in enumerate(stock_labels.items()):
        with stock_cols[i]:
            mode_status = status.get(mode_key, {})
            if mode_status.get('trained'):
                st.success(f"**{label}**")
                st.caption(desc)
                st.metric("F1", f"{mode_status['f1_score']:.1%}")
            else:
                st.warning(f"**{label}**")
                st.caption(desc)
                st.markdown("❌ Not trained")
    
    # ETF models
    st.markdown("**📈 ETF Models:**")
    etf_cols = st.columns(3)
    for i, (mode_key, (label, desc)) in enumerate(etf_labels.items()):
        with etf_cols[i]:
            mode_status = status.get(mode_key, {})
            if mode_status.get('trained'):
                st.success(f"**{label}**")
                st.caption(desc)
                st.metric("F1", f"{mode_status['f1_score']:.1%}")
            else:
                st.warning(f"**{label}**")
                st.caption(desc)
                st.markdown("❌ Not trained")
    
    st.markdown("---")
    
    col_stock, col_etf = st.columns(2)
    
    with col_stock:
        st.markdown("**📊 Train on Stocks:**")
        if st.button("🏢 Train Stocks (All Modes)", use_container_width=True,
                    disabled=st.session_state.training_in_progress):
            train_stocks_ui('stock', modes=['daytrade', 'swing', 'investment'])
        st.caption("DayTrade + Swing + Investment")
    
    with col_etf:
        st.markdown("**📊 Train on ETFs:**")
        if st.button("📈 Train ETFs (All Modes)", use_container_width=True,
                    disabled=st.session_state.training_in_progress):
            train_stocks_ui('etf', modes=['daytrade', 'swing', 'investment'])
        st.caption("DayTrade + Swing + Investment")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIPS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.divider()
    
    with st.expander("💡 Training Tips"):
        st.markdown("""
        ### 🎯 Why Dynamic Symbols (Recommended)?
        Since you scan for explosive setups across many coins, the ML needs to work on ANY coin.
        
        **Auto: Top 30/50 by Volume** fetches the currently most active futures pairs:
        - Always trains on what's actually being traded
        - Captures trending coins automatically  
        - Mix of majors, alts, memecoins, new listings
        - Better generalization to ANY coin you scan!
        
        ### When to Retrain:
        - **Weekly** - Market conditions change, models should adapt
        - **After major market events** - Crashes, rallies, regime changes
        - **If win rate drops** - Model may be stale
        
        ### Best Practices:
        - **90 days** is a good balance of recency and sample size
        - **Auto: Top 30** for daily retraining (faster)
        - **Auto: Top 50** for weekly retraining (more diverse)
        
        ### What Each Mode Learns:
        | Mode | Patterns | Hold Time |
        |------|----------|-----------|
        | Scalp | Quick momentum, spikes | Minutes |
        | DayTrade | Intraday trends, breakouts | Hours |
        | Swing | Multi-day moves, accumulation | Days |
        | Investment | Major trends, macro cycles | Weeks |
        
        ### 🧠 How ML Works on New Coins:
        The ML doesn't learn "BTCUSDT goes up" - it learns:
        - "When whale_pct=72% + BB_squeeze=90% → Usually LONG"
        - "When retail > whale → Retail trap risk"
        
        These **patterns transfer to ANY coin** - even ones not in training!
        """)


def train_mode(mode: str, symbols: list, days: int, use_dynamic: bool = False, dynamic_count: int = 30):
    """Train a single mode with progress bar."""
    st.session_state.training_in_progress = True
    st.session_state.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {mode.upper()} training...")
    
    try:
        # Progress bar
        progress_bar = st.progress(0, text=f"Initializing {mode.upper()} training...")
        status_text = st.empty()
        
        # Step 1: Import
        progress_bar.progress(5, text="Loading ML modules...")
        from ml.mode_specific_trainer import train_all_modes as train_modes, MODE_CONFIG, get_diverse_training_symbols
        
        # Step 2: Get symbols (dynamic or static)
        if use_dynamic:
            progress_bar.progress(10, text=f"Fetching top {dynamic_count} futures pairs by volume...")
            status_text.info(f"📊 Fetching currently most active futures pairs...")
            training_symbols = get_diverse_training_symbols(dynamic_count)
            status_text.success(f"✅ Got {len(training_symbols)} symbols: {', '.join(training_symbols[:5])}...")
        else:
            training_symbols = symbols
        
        # Step 3: Show data fetching progress
        progress_bar.progress(20, text=f"Fetching historical data for {len(training_symbols)} symbols...")
        status_text.info(f"📊 Downloading {days} days of {MODE_CONFIG[mode]['timeframes']} data...")
        
        # Step 4: Train
        progress_bar.progress(40, text="Training ML models... This takes a few minutes.")
        status_text.info("🤖 Training direction classifier, ETA predictor, and SL optimizer...")
        
        result = train_modes(symbols=training_symbols, days=days, modes=[mode])
        
        # Step 5: Complete
        progress_bar.progress(100, text="✅ Complete!")
        
        if mode in result and isinstance(result[mode], dict) and 'direction_f1' in result[mode]:
            f1 = result[mode]['direction_f1']
            samples = result[mode]['n_samples']
            st.session_state.training_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {mode.upper()} complete! F1={f1:.1%}, Samples={samples:,}"
            )
            status_text.success(f"✅ {mode.upper()} trained! F1 Score: {f1:.1%} | Samples: {samples:,}")
        else:
            st.session_state.training_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {mode.upper()} failed"
            )
            status_text.error(f"❌ Training failed for {mode}")
        
        time.sleep(2)  # Show result briefly
    
    except Exception as e:
        st.session_state.training_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}"
        )
        st.error(f"❌ Error: {str(e)}")
    
    finally:
        st.session_state.training_in_progress = False
        st.rerun()


def train_all_modes(symbols: list, days: int, use_dynamic: bool = False, dynamic_count: int = 30):
    """Train all modes with progress bar."""
    st.session_state.training_in_progress = True
    st.session_state.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ALL MODES training...")
    
    modes = ['scalp', 'daytrade', 'swing', 'investment']
    mode_labels = {'scalp': '⚡ Scalp', 'daytrade': '📈 DayTrade', 'swing': '🌊 Swing', 'investment': '💎 Investment'}
    
    try:
        # Progress tracking
        progress_bar = st.progress(0, text="Initializing training...")
        status_text = st.empty()
        results_container = st.container()
        
        # Import
        progress_bar.progress(3, text="Loading ML modules...")
        from ml.mode_specific_trainer import train_all_modes as train_modes_func, MODE_CONFIG, get_diverse_training_symbols
        
        # Get symbols (dynamic or static) - ONCE for all modes
        if use_dynamic:
            progress_bar.progress(5, text=f"Fetching top {dynamic_count} futures pairs by volume...")
            status_text.info(f"📊 Fetching currently most active futures pairs...")
            training_symbols = get_diverse_training_symbols(dynamic_count)
            with results_container:
                st.success(f"📊 Training on {len(training_symbols)} symbols: {', '.join(training_symbols[:8])}...")
            st.session_state.training_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Fetched {len(training_symbols)} symbols dynamically"
            )
        else:
            training_symbols = symbols
        
        results = {}
        total_modes = len(modes)
        
        for i, mode in enumerate(modes):
            # Calculate progress
            base_progress = int(((i) / total_modes) * 85) + 10  # 10-95%
            
            # Update status
            progress_bar.progress(base_progress, text=f"Training {mode_labels[mode]}... ({i+1}/{total_modes})")
            status_text.info(f"📊 {mode_labels[mode]}: Fetching {MODE_CONFIG[mode]['timeframes']} data and training...")
            
            st.session_state.training_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Training {mode.upper()}..."
            )
            
            # Train this mode
            result = train_modes_func(symbols=training_symbols, days=days, modes=[mode])
            
            # Store result
            if mode in result:
                results[mode] = result[mode]
                
                if isinstance(result[mode], dict) and 'direction_f1' in result[mode]:
                    f1 = result[mode]['direction_f1']
                    samples = result[mode]['n_samples']
                    st.session_state.training_log.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {mode.upper()}: F1={f1:.1%}, Samples={samples:,}"
                    )
                    with results_container:
                        st.success(f"✅ {mode_labels[mode]}: F1={f1:.1%} | {samples:,} samples")
                else:
                    st.session_state.training_log.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {mode.upper()}: Failed"
                    )
                    with results_container:
                        st.error(f"❌ {mode_labels[mode]}: Failed")
        
        # Complete
        progress_bar.progress(100, text="✅ All training complete!")
        
        success_count = sum(1 for mode, metrics in results.items() 
                          if isinstance(metrics, dict) and 'direction_f1' in metrics)
        
        if success_count > 0:
            status_text.success(f"🎉 Training complete! {success_count}/{total_modes} modes trained successfully.")
        else:
            status_text.error("❌ Training failed for all modes")
        
        time.sleep(3)  # Show results briefly
    
    except Exception as e:
        st.session_state.training_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}"
        )
        st.error(f"❌ Error: {str(e)}")
    
    finally:
        st.session_state.training_in_progress = False
        st.rerun()


def train_stocks_ui(asset_type: str = 'stock', modes: list = None):
    """Train stock/ETF models with progress bar."""
    st.session_state.training_in_progress = True
    st.session_state.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {asset_type.upper()} training...")
    
    if modes is None:
        modes = ['daytrade', 'swing', 'investment']
    mode_labels = {'daytrade': '📊 DayTrade', 'swing': '🌊 Swing', 'investment': '💎 Investment'}
    
    try:
        progress_bar = st.progress(0, text=f"Initializing {asset_type.upper()} training...")
        status_text = st.empty()
        results_container = st.container()
        
        # Import
        progress_bar.progress(5, text="Loading ML modules...")
        from ml.stock_trainer import train_stocks, DEFAULT_STOCKS, DEFAULT_ETFS
        
        symbols = DEFAULT_STOCKS if asset_type == 'stock' else DEFAULT_ETFS
        
        with results_container:
            st.info(f"📊 Training on {len(symbols)} {asset_type}s: {', '.join(symbols[:5])}...")
        
        results = {}
        total_modes = len(modes)
        
        for i, mode in enumerate(modes):
            base_progress = int(((i) / total_modes) * 85) + 10
            
            progress_bar.progress(base_progress, text=f"Training {mode_labels[mode]} ({asset_type})... ({i+1}/{total_modes})")
            status_text.info(f"📊 {mode_labels[mode]}: Fetching data and training...")
            
            st.session_state.training_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Training {mode.upper()} ({asset_type})..."
            )
            
            # Train this mode
            result = train_stocks(symbols=symbols, modes=[mode], asset_type=asset_type)
            
            if mode in result:
                results[mode] = result[mode]
                
                if isinstance(result[mode], dict) and 'direction_f1' in result[mode]:
                    f1 = result[mode]['direction_f1']
                    samples = result[mode]['n_samples']
                    st.session_state.training_log.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {mode.upper()} ({asset_type}): F1={f1:.1%}"
                    )
                    with results_container:
                        st.success(f"✅ {mode_labels[mode]} ({asset_type}): F1={f1:.1%} | {samples:,} samples")
                else:
                    st.session_state.training_log.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {mode.upper()} ({asset_type}): Failed"
                    )
                    with results_container:
                        st.error(f"❌ {mode_labels[mode]} ({asset_type}): Failed")
        
        progress_bar.progress(100, text="✅ Training complete!")
        
        success_count = sum(1 for mode, metrics in results.items() 
                          if isinstance(metrics, dict) and 'direction_f1' in metrics)
        
        if success_count > 0:
            status_text.success(f"🎉 {asset_type.upper()} training complete! {success_count}/{total_modes} modes trained.")
        else:
            status_text.error(f"❌ Training failed for {asset_type}")
        
        time.sleep(3)
    
    except Exception as e:
        st.session_state.training_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}"
        )
        st.error(f"❌ Error: {str(e)}")
    
    finally:
        st.session_state.training_in_progress = False
        st.rerun()


def render_training_sidebar():
    """Render compact training status in sidebar."""
    
    status = get_model_status()
    
    # Count trained models
    trained_count = sum(1 for mode in ['scalp', 'daytrade', 'swing', 'investment'] 
                       if status.get(mode, {}).get('trained', False))
    
    if trained_count == 4:
        st.sidebar.success(f"🤖 ML: {trained_count}/4 modes trained")
    elif trained_count > 0:
        st.sidebar.warning(f"🤖 ML: {trained_count}/4 modes trained")
    else:
        st.sidebar.error("🤖 ML: Not trained")
    
    # Quick train button
    if st.sidebar.button("🎯 Train ML Models", use_container_width=True):
        st.session_state.page = 'training'
        st.rerun()


# For direct testing
if __name__ == "__main__":
    st.set_page_config(page_title="ML Training", layout="wide")
    render_training_page()
