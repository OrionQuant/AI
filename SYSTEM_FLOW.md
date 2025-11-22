# OrionQuant System Flow Documentation

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORIONQUANT TRADING SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Complete System Flow

### 1. Data Collection Flow

```
┌─────────────┐
│   Binance   │
│     API     │
└──────┬──────┘
       │
       │ Download Historical Data
       │ (OHLCV: Open, High, Low, Close, Volume)
       ▼
┌─────────────────────┐
│ BinanceDataDownloader│
│  - Download 5m data  │
│  - Store as Parquet  │
│  - Auto-update       │
└──────┬──────────────┘
       │
       │ Raw Market Data
       ▼
┌─────────────────────┐
│   data/raw/         │
│ BTCUSDT_5m_*.parquet│
└─────────────────────┘
```

**Process:**
1. Connects to Binance API
2. Downloads historical candlestick data (5-minute intervals)
3. Stores data in efficient Parquet format
4. Automatically updates with latest candles
5. Data includes: timestamp, open, high, low, close, volume

---

### 2. Feature Engineering Flow

```
┌─────────────────────┐
│   Raw OHLCV Data    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              FeatureBuilder Pipeline                     │
├─────────────────────────────────────────────────────────┤
│ 1. Price Features:                                       │
│    - Log returns (1, 2, 3, 5 periods)                  │
│    - Percentage changes                                 │
│                                                          │
│ 2. Technical Indicators:                                 │
│    - EMAs (8, 21, 50, 200)                             │
│    - RSI (14 period)                                    │
│    - MACD (12, 26, 9)                                   │
│    - ATR (14 period)                                    │
│    - Bollinger Bands (20, 2σ)                          │
│                                                          │
│ 3. Volume Features:                                      │
│    - Volume moving averages                             │
│    - Volume ratios                                      │
│    - On-Balance Volume (OBV)                           │
│    - Volume-Price Trend (VPT)                           │
│                                                          │
│ 4. Volatility Features:                                 │
│    - Rolling volatility (5, 20 periods)                │
│    - High-Low range                                     │
└──────┬──────────────────────────────────────────────────┘
       │
       │ 38 Features Created
       ▼
┌─────────────────────┐
│  Feature DataFrame   │
│  (Normalized)        │
└──────┬──────────────┘
       │
       │ Create Sequences
       │ (128 timesteps)
       ▼
┌─────────────────────┐
│  Sequence Data       │
│  Shape: (N, 128, 37) │
└─────────────────────┘
```

**Output:** 38 engineered features per timestep, normalized and ready for model input

---

### 3. Label Creation Flow

```
┌─────────────────────┐
│  Feature DataFrame   │
└──────┬──────────────┘
       │
       │ Calculate Future Returns
       ▼
┌─────────────────────────────────────────┐
│         Label Creation                  │
├─────────────────────────────────────────┤
│ Future Return = (Price[t+1] - Price[t]) │
│                    / Price[t]           │
│                                         │
│ Classification Labels:                  │
│ - BUY  (0): Return > 0.5%               │
│ - HOLD (1): -0.5% ≤ Return ≤ 0.5%       │
│ - SELL (2): Return < -0.5%              │
│                                         │
│ Regression Label:                       │
│ - Actual future return value            │
└──────┬──────────────────────────────────┘
       │
       │ Labeled Data
       ▼
┌─────────────────────┐
│  Training Dataset   │
└─────────────────────┘
```

---

### 4. Model Training Flow

```
┌─────────────────────┐
│  Labeled Sequences   │
└──────┬──────────────┘
       │
       │ Time-Based Split
       │ (70% train, 15% val, 15% test)
       ▼
┌─────────────────────────────────────────────────────────┐
│              Model Architecture                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────┐              │
│  │   TransformerSignalNet               │              │
│  │                                       │              │
│  │  1. Input Projection (37 → 256)      │              │
│  │  2. Positional Encoding              │              │
│  │  3. Transformer Blocks (6 layers)    │              │
│  │     - Multi-Head Attention (8 heads) │              │
│  │     - Feed-Forward Networks          │              │
│  │     - Layer Normalization            │              │
│  │     - Residual Connections           │              │
│  │  4. Global Pooling                   │              │
│  │  5. Classification Head (3 classes) │              │
│  │  6. Regression Head (1 output)      │              │
│  └──────────────────────────────────────┘              │
│                                                          │
│  Parameters: 4.1M                                        │
└──────┬──────────────────────────────────────────────────┘
       │
       │ Training Process
       ▼
┌─────────────────────────────────────────────────────────┐
│              Training Loop                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  For each epoch:                                        │
│    1. Forward pass through model                        │
│    2. Calculate loss:                                   │
│       Loss = CrossEntropy(classification) +             │
│              λ * MSE(regression)                        │
│    3. Backpropagation                                   │
│    4. Update weights (AdamW optimizer)                  │
│    5. Validate on validation set                        │
│    6. Save best model (lowest val loss)                │
│                                                          │
│  Early Stopping: Stop if no improvement for 10 epochs   │
└──────┬──────────────────────────────────────────────────┘
       │
       │ Trained Model
       ▼
┌─────────────────────┐
│ models/best_model.pth│
│  (48 MB checkpoint)  │
└─────────────────────┘
```

**Training Details:**
- Multi-task learning: Classification + Regression
- Loss function: `Loss = CE_loss + 0.1 * MSE_loss`
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)
- Batch size: 128
- Mixed precision training (if GPU available)

---

### 5. Real-Time Prediction Flow (Web UI)

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
│              http://localhost:5000                      │
└──────────────────────┬─────────────────────────────────┘
                        │
                        │ HTTP Request
                        │ GET /api/predict
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Flask Web Server (app.py)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Receive prediction request                          │
│  2. Load/Update market data from Binance                │
│  3. Build features (same pipeline as training)         │
│  4. Create sequence (last 128 timesteps)                │
│  5. Load trained Transformer model                      │
│  6. Run inference:                                      │
│     - Forward pass through model                        │
│     - Get classification probabilities                 │
│     - Get expected return prediction                    │
│  7. Return JSON response                                │
└──────┬──────────────────────────────────────────────────┘
       │
       │ JSON Response
       │ {
       │   "signal": "BUY/HOLD/SELL",
       │   "confidence": 0.95,
       │   "probabilities": {...},
       │   "expected_return": 0.0023,
       │   "current_price": 84299.82
       │ }
       ▼
┌─────────────────────────────────────────────────────────┐
│              Frontend (JavaScript)                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Display signal with color coding                    │
│  2. Show confidence percentage                          │
│  3. Display probability bars                            │
│  4. Update price chart (Chart.js)                       │
│  5. Auto-refresh every 30 seconds                       │
└─────────────────────────────────────────────────────────┘
```

---

### 6. Backtesting Flow

```
┌─────────────────────┐
│  Historical Data     │
│  (Test Period)      │
└──────┬──────────────┘
       │
       │ Generate Predictions
       ▼
┌─────────────────────────────────────────────────────────┐
│              Model Inference                            │
│  - Process each timestep                                │
│  - Generate BUY/HOLD/SELL signals                       │
│  - Get confidence scores                                │
└──────┬──────────────────────────────────────────────────┘
       │
       │ Signals + Prices
       ▼
┌─────────────────────────────────────────────────────────┐
│              Backtester Engine                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  For each signal:                                       │
│    1. Check current position                            │
│    2. Execute trade if signal changes                   │
│    3. Apply realistic costs:                            │
│       - Slippage (0.05%)                                │
│       - Trading fees (0.1% taker)                       │
│    4. Risk management:                                  │
│       - Stop loss (2%)                                   │
│       - Take profit (5%)                                │
│    5. Update portfolio equity                           │
│                                                          │
└──────┬──────────────────────────────────────────────────┘
       │
       │ Performance Metrics
       ▼
┌─────────────────────────────────────────────────────────┐
│              Results                                    │
├─────────────────────────────────────────────────────────┤
│  - Total Return: $X,XXX (XX%)                           │
│  - Sharpe Ratio: X.XX                                   │
│  - Win Rate: XX%                                        │
│  - Profit Factor: X.XX                                  │
│  - Max Drawdown: XX%                                    │
│  - Number of Trades: XXX                                │
│  - Equity Curve (time series)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete End-to-End Flow

```
┌──────────────┐
│  1. DATA     │  Binance API → Download → Store Parquet
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  2. FEATURES │  OHLCV → Technical Indicators → 38 Features
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  3. LABELS   │  Future Returns → BUY/HOLD/SELL + Return Value
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  4. TRAIN    │  Sequences → Transformer Model → Trained Model
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  5. DEPLOY   │  Load Model → Flask Server → Web UI
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  6. PREDICT  │  Live Data → Features → Model → Signal
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  7. DISPLAY  │  Signal → Dashboard → User
└──────────────┘
```

---

## 📱 Web UI Components Flow

```
┌─────────────────────────────────────────────────────────┐
│                    DASHBOARD LAYOUT                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Header: Status Indicator + Refresh Button       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  Prediction Card │  │   Price Chart             │  │
│  │  - Signal Icon   │  │   - Interactive Chart.js  │  │
│  │  - Confidence    │  │   - Multiple timeframes   │  │
│  │  - Probabilities │  │   - Real-time updates     │  │
│  │  - Expected Ret  │  └──────────────────────────┘  │
│  └──────────────────┘                                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Backtest Section                                 │  │
│  │  - Date range selector                            │  │
│  │  - Run backtest button                            │  │
│  │  - Performance metrics                            │  │
│  │  - Equity curve chart                             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Key Components

### Backend (Python/Flask)
- **app.py**: Main Flask application
- **BinanceDataDownloader**: Data collection
- **FeatureBuilder**: Feature engineering pipeline
- **TransformerSignalNet**: Model architecture
- **Backtester**: Realistic backtesting engine

### Frontend (HTML/CSS/JavaScript)
- **index.html**: Dashboard layout
- **style.css**: Modern dark theme styling
- **app.js**: Real-time updates and API calls
- **Chart.js**: Interactive price charts

### Data Flow
1. **Raw Data**: Parquet files in `data/raw/`
2. **Features**: 38 engineered features per timestep
3. **Sequences**: 128-timestep windows
4. **Model**: Trained Transformer (4.1M parameters)
5. **Predictions**: Real-time signals with confidence

---

## 🚀 Usage Flow

### Training a Model
```bash
1. Download data: python train_main.py --download
2. Train model: python train_main.py --model-type Transformer --backtest
3. Model saved to: models/best_model.pth
```

### Running Web UI
```bash
1. Start server: python app.py
2. Open browser: http://localhost:5000
3. View predictions: Auto-updates every 30 seconds
4. Run backtest: Select date range → Click "Run Backtest"
```

### Making Predictions
```
User Request → Flask API → Load Data → Build Features → 
Model Inference → Return Signal → Display in UI
```

---

## 📈 Model Architecture Details

### Transformer Signal Network
- **Input**: (batch_size, 128, 37) - sequences of 128 timesteps with 37 features
- **Processing**:
  - Input projection: 37 → 256 dimensions
  - Positional encoding for temporal awareness
  - 6 transformer blocks with multi-head attention
  - Global pooling (last timestep + mean)
- **Output**:
  - Classification: 3 classes (BUY/HOLD/SELL) with probabilities
  - Regression: Expected return value

### Why Transformer?
- **Attention Mechanism**: Captures long-range dependencies
- **Parallel Processing**: Efficient training
- **Temporal Patterns**: Understands market trends over time
- **Multi-head Attention**: Captures different aspects of price movements

---

## 🔐 Data Security & Best Practices

1. **No API Keys Required**: Uses public Binance API for market data
2. **Local Storage**: All data stored locally in `data/raw/`
3. **Model Checkpoints**: Saved locally in `models/`
4. **No Live Trading**: System is for analysis only
5. **Paper Trading First**: Always backtest before considering live trading

---

## 📊 Performance Metrics Explained

- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Profit Factor**: Gross profit / Gross loss (>1 means profitable)
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Total Return**: Cumulative profit/loss percentage

---

This system provides a complete pipeline from data collection to live predictions with a modern web interface!

