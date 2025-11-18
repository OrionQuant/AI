"""
Example usage script demonstrating the full pipeline.

This script shows how to:
1. Download data
2. Build features
3. Train a model
4. Run backtest

Run this to verify your installation works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.binance_downloader import BinanceDataDownloader
from src.features.feature_builder import FeatureBuilder
from src.models.lstm_signal_net import LSTMSignalNet
from src.training.train import train_model
from src.backtest.backtester import Backtester, Signal

print("="*60)
print("OrionQuant - Example Usage")
print("="*60)

# Step 1: Download data (or use synthetic for testing)
print("\n[1/4] Downloading data...")
try:
    downloader = BinanceDataDownloader(data_dir="data/raw")
    # Try to download a small sample (last 30 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"Downloading BTC 5m data from {start_date} to {end_date}...")
    df = downloader.download_historical(
        symbol="BTCUSDT",
        timeframe="5m",
        start_date=start_date,
        end_date=end_date,
        save=True
    )
    
    if len(df) == 0:
        print("Warning: No data downloaded. Using synthetic data for demo.")
        # Generate synthetic data
        dates = pd.date_range(start_date, end_date, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
        df = pd.DataFrame({
            'timestamp': dates[:len(prices)],
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.rand(len(prices)) * 1000
        })
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Using synthetic data for demo...")
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(len(dates)) * 1000
    })

print(f"✓ Loaded {len(df)} candles")

# Step 2: Build features
print("\n[2/4] Building features...")
builder = FeatureBuilder(seq_len=128, normalize=True)
df_features = builder.build_features(df)
df_labeled = builder.create_labels(
    df_features,
    horizon=1,
    buy_threshold=0.005,
    sell_threshold=-0.005
)

# Create sequences
X, y_cls, y_reg = builder.create_sequences(df_labeled)
print(f"✓ Created {len(X)} sequences of shape {X.shape}")

# Split into train/val (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train_cls, y_val_cls = y_cls[:split_idx], y_cls[split_idx:]
y_train_reg, y_val_reg = y_reg[:split_idx], y_reg[split_idx:]

print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")

# Step 3: Train model (quick training for demo)
print("\n[3/4] Training model (quick demo - 5 epochs)...")
config = {
    'hidden_dim': 128,  # Smaller for quick demo
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'epochs': 5,  # Just 5 epochs for demo
    'lambda_reg': 0.1,
    'early_stop_patience': 10,
    'seed': 42,
    'use_amp': False,  # Disable AMP for compatibility
    'min_lr': 1e-6
}

try:
    model, history = train_model(
        X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg,
        config, LSTMSignalNet,
        save_dir="models"
    )
    print("✓ Model training complete")
except Exception as e:
    print(f"Error during training: {e}")
    print("This might be due to missing CUDA or other dependencies.")
    print("Check requirements.txt and install missing packages.")
    sys.exit(1)

# Step 4: Run backtest
print("\n[4/4] Running backtest...")
# Get predictions on validation set
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

signals = []
confidences = []

with torch.no_grad():
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    logits, pred_return = model(X_val_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_signal = torch.argmax(probs, dim=1)
    confidence = torch.max(probs, dim=1)[0]
    
    signals = pred_signal.cpu().numpy()
    confidences = confidence.cpu().numpy()

# Align with dataframe
test_df = df_labeled.iloc[split_idx + 128:].reset_index(drop=True)  # Account for seq_len
signals_full = np.concatenate([np.full(len(df_labeled) - len(test_df), Signal.HOLD.value), signals])
confidences_full = np.concatenate([np.ones(len(df_labeled) - len(test_df)), confidences])

# Trim to match test_df length
if len(signals_full) > len(test_df):
    signals_full = signals_full[-len(test_df):]
    confidences_full = confidences_full[-len(test_df):]

backtester = Backtester(
    initial_capital=10000.0,
    taker_fee=0.001,
    slippage_pct=0.0005,
    stop_loss_pct=0.02,
    take_profit_pct=0.05
)

results = backtester.run_backtest(test_df, signals_full, confidences_full)

print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
print(f"Number of Trades: {results['num_trades']}")
print(f"Win Rate: {results['win_rate']*100:.2f}%")
print(f"Profit Factor: {results['profit_factor']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print("="*60)

print("\n✓ Example pipeline complete!")
print("\nNext steps:")
print("1. Download more historical data (2+ years recommended)")
print("2. Tune hyperparameters in config/train_config.json")
print("3. Train for more epochs")
print("4. Experiment with different model architectures")
print("5. Run walk-forward validation for robust evaluation")

