# 🚀 Quick Start Flow Guide

## Simple Step-by-Step Flow

### Step 1: Install & Setup
```bash
pip install -r requirements.txt
```

### Step 2: Train Model
```bash
python train_main.py --model-type Transformer --backtest
```

**What happens:**
1. ✅ Downloads BTC data from Binance
2. ✅ Builds 38 technical features
3. ✅ Creates BUY/HOLD/SELL labels
4. ✅ Trains Transformer model (4.1M parameters)
5. ✅ Runs backtest
6. ✅ Saves model to `models/best_model.pth`

### Step 3: Start Web Server
```bash
python app.py
```

**What happens:**
1. ✅ Loads trained model
2. ✅ Starts Flask server on port 5000
3. ✅ Ready for predictions

### Step 4: Open Dashboard
```
Open browser: http://localhost:5000
```

**What you see:**
- 📊 Real-time BTC price
- 🎯 Trading signal (BUY/HOLD/SELL)
- 📈 Confidence percentage
- 💹 Expected return
- 📉 Interactive price chart
- 🧪 Backtest results

---

## 🔄 Real-Time Prediction Flow

```
Every 30 seconds:
  ↓
1. Fetch latest BTC data from Binance
  ↓
2. Build 38 technical features
  ↓
3. Create sequence (last 128 timesteps)
  ↓
4. Run through Transformer model
  ↓
5. Get prediction:
   - Signal: BUY/HOLD/SELL
   - Confidence: 0-100%
   - Expected Return: %
  ↓
6. Update dashboard automatically
```

---

## 📋 File Structure Flow

```
OrionQuant/
├── data/raw/              ← Raw market data (Parquet)
├── models/                ← Trained models (.pth)
├── src/
│   ├── data/              ← Data downloader
│   ├── features/          ← Feature engineering
│   ├── models/            ← Model architectures
│   ├── training/          ← Training scripts
│   └── backtest/          ← Backtesting engine
├── templates/             ← HTML dashboard
├── static/                ← CSS & JavaScript
├── app.py                 ← Flask web server
└── train_main.py          ← Training script
```

---

## 🎯 Decision Flow

```
Market Data
    ↓
Technical Analysis (38 features)
    ↓
Transformer Model
    ↓
Prediction:
    ├─→ BUY (if expected return > 0.5%)
    ├─→ HOLD (if -0.5% ≤ return ≤ 0.5%)
    └─→ SELL (if expected return < -0.5%)
    ↓
Confidence Score (0-100%)
    ↓
Display on Dashboard
```

---

## 🔧 Troubleshooting Flow

```
Problem? → Check:
    ↓
1. Is model trained? → Check models/best_model.pth exists
    ↓
2. Is data available? → Check data/raw/ has .parquet files
    ↓
3. Is server running? → Check http://localhost:5000/api/health
    ↓
4. Are dependencies installed? → Run pip install -r requirements.txt
```

---

## 📊 Data Flow Summary

```
Binance API
    ↓
Download OHLCV Data
    ↓
Store as Parquet
    ↓
Feature Engineering (38 features)
    ↓
Create Sequences (128 timesteps)
    ↓
Train Model
    ↓
Save Checkpoint
    ↓
Load for Predictions
    ↓
Real-time Inference
    ↓
Display Results
```

---

This is the complete flow of how OrionQuant works from data to predictions!

