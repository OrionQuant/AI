# OrionQuant - BTC Trading Model Framework

A production-ready framework for building, training, validating, and deploying BTC trading signal models. This repository implements a complete pipeline from data collection to backtesting with realistic market conditions.

## 🎯 Features

- **Data Collection**: Automated Binance data downloader with multi-timeframe support
- **Feature Engineering**: Comprehensive technical indicators and normalized features
- **Model Architectures**: LSTM and CNN-LSTM models for sequence prediction
- **Multi-task Learning**: Simultaneous classification (BUY/HOLD/SELL) and regression (expected return)
- **Walk-Forward Validation**: Time-series cross-validation to prevent lookahead bias
- **Realistic Backtesting**: Includes slippage, fees, stop-loss, and take-profit
- **Performance Metrics**: Sharpe ratio, profit factor, max drawdown, win rate

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Binance API key (optional, for live data updates)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd OrionQuant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: If `ta-lib` installation fails, install it separately:
```bash
# macOS
brew install ta-lib
pip install ta-lib

# Linux
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

### 2. Download Data

```bash
# Download historical BTC data (default: 5m timeframe, from 2022-01-01)
python -c "from src.data.binance_downloader import BinanceDataDownloader; \
           d = BinanceDataDownloader(); \
           d.download_historical('BTCUSDT', '5m', '2022-01-01')"
```

Or use the main training script which downloads automatically:
```bash
python train_main.py --download
```

### 3. Train Model

```bash
# Train with default config
python train_main.py

# Train with backtest
python train_main.py --backtest

# Train CNN-LSTM model
python train_main.py --model-type CNN-LSTM --backtest

# Use custom config
python train_main.py --config config/my_config.json
```

### 4. Configuration

Edit `config/train_config.json` to customize:

- **Model architecture**: hidden_dim, num_layers, dropout
- **Training hyperparameters**: lr, batch_size, epochs, lambda_reg
- **Data splits**: train/val/test date ranges
- **Feature thresholds**: buy_threshold, sell_threshold
- **Backtest parameters**: fees, slippage, stop-loss, take-profit

## 📁 Project Structure

```
OrionQuant/
├── src/
│   ├── data/
│   │   └── binance_downloader.py    # Data collection from Binance
│   ├── features/
│   │   └── feature_builder.py       # Feature engineering pipeline
│   ├── models/
│   │   └── lstm_signal_net.py       # Model architectures
│   ├── training/
│   │   ├── train.py                 # Training utilities
│   │   └── walk_forward.py          # Walk-forward validation
│   ├── backtest/
│   │   └── backtester.py            # Backtesting engine
│   └── utils/
├── config/
│   └── train_config.json            # Training configuration
├── data/
│   └── raw/                          # Downloaded market data (parquet)
├── models/                           # Saved model checkpoints
├── train_main.py                     # Main training script
├── requirements.txt                  # Python dependencies
└── README.md
```

## 🔧 Usage Examples

### Download Data for Multiple Timeframes

```python
from src.data.binance_downloader import BinanceDataDownloader

downloader = BinanceDataDownloader()

# Download 5-minute data
df_5m = downloader.download_historical('BTCUSDT', '5m', '2022-01-01')

# Download 1-hour data
df_1h = downloader.download_historical('BTCUSDT', '1h', '2022-01-01')

# Update existing data
df_5m = downloader.update_data('BTCUSDT', '5m')
```

### Build Features and Create Sequences

```python
from src.features.feature_builder import FeatureBuilder

builder = FeatureBuilder(seq_len=128, normalize=True)

# Build features
df_features = builder.build_features(df)

# Create labels (BUY if return > 0.5%, SELL if < -0.5%)
df_labeled = builder.create_labels(df_features, horizon=1, 
                                   buy_threshold=0.005, sell_threshold=-0.005)

# Create sequences for training
X, y_cls, y_reg = builder.create_sequences(df_labeled)
```

### Train Model

```python
from src.models.lstm_signal_net import LSTMSignalNet
from src.training.train import train_model

config = {
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.2,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'batch_size': 128,
    'epochs': 100,
    'lambda_reg': 0.1,
    'early_stop_patience': 10,
    'seed': 42
}

model, history = train_model(
    X_train, y_train_cls, y_train_reg,
    X_val, y_val_cls, y_val_reg,
    config, LSTMSignalNet
)
```

### Run Backtest

```python
from src.backtest.backtester import Backtester, Signal
import numpy as np

# Initialize backtester
backtester = Backtester(
    initial_capital=10000.0,
    taker_fee=0.001,      # 0.1% taker fee
    slippage_pct=0.0005,  # 0.05% slippage
    stop_loss_pct=0.02,  # 2% stop loss
    take_profit_pct=0.05 # 5% take profit
)

# Generate signals (0=BUY, 1=HOLD, 2=SELL)
signals = model.predict(df_test)  # Your prediction logic

# Run backtest
results = backtester.run_backtest(df_test, signals)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

## 📊 Model Architecture

### LSTM Signal Network

- **Input**: Sequences of normalized features (seq_len × n_features)
- **LSTM Layers**: Multi-layer bidirectional LSTM
- **Outputs**:
  - Classification: BUY (0), HOLD (1), SELL (2)
  - Regression: Expected return

### Loss Function

```
Loss = CrossEntropy(classification) + λ * MSE(regression)
```

Default λ = 0.1 (adjust based on scale of returns).

## 🎓 Training Best Practices

1. **Walk-Forward Validation**: Always use time-based splits, never shuffle across time
2. **Feature Normalization**: Use RobustScaler to handle outliers
3. **Early Stopping**: Monitor validation loss to prevent overfitting
4. **Hyperparameter Tuning**: Use Optuna for systematic search
5. **Ensemble Models**: Combine multiple models for robustness
6. **Backtest First**: Always backtest before live trading

## 🔍 Validation & Metrics

### Classification Metrics
- Precision, Recall, F1-score per class (BUY/HOLD/SELL)
- Overall accuracy

### Regression Metrics
- MSE, MAE
- Correlation between predicted and actual returns

### Trading Metrics
- **Total Return**: Cumulative P&L
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## ⚠️ Risk Management

The backtester includes:
- **Stop Loss**: Automatic exit at configured loss threshold
- **Take Profit**: Automatic exit at profit target
- **Position Sizing**: Configurable max position size
- **Slippage**: Realistic execution prices
- **Fees**: Maker/taker fee simulation

**Important**: Always test on paper trading/testnet before live deployment!

## 🚧 Roadmap

- [ ] Transformer-based model architecture
- [ ] Optuna hyperparameter optimization script
- [ ] Real-time inference service
- [ ] Integration with Binance API for live trading
- [ ] Advanced orderbook features
- [ ] Multi-asset support
- [ ] Reinforcement learning layer

## 📝 Configuration Reference

### Model Config
```json
{
  "model": {
    "type": "LSTM",
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.2
  }
}
```

### Training Config
```json
{
  "training": {
    "lr": 3e-4,
    "batch_size": 128,
    "epochs": 100,
    "lambda_reg": 0.1,
    "early_stop_patience": 10
  }
}
```

### Data Config
```json
{
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "5m",
    "seq_len": 128,
    "horizon": 1,
    "buy_threshold": 0.005,
    "sell_threshold": -0.005
  }
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `seq_len` or `hidden_dim`
- Use gradient accumulation

### Data Download Fails
- Check internet connection
- Verify Binance API is accessible
- Try reducing date range

### Poor Model Performance
- Increase training data
- Tune buy/sell thresholds
- Try different model architectures
- Check for data leakage

## 📄 License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results. Always test thoroughly on paper trading before using real capital.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for quantitative trading**

