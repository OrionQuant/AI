# OrionQuant Web UI Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model (Optional)

If you don't have a trained model yet, train one first:

```bash
# Train with default LSTM model
python train_main.py --backtest

# Train with Transformer model (more powerful)
python train_main.py --model-type Transformer --backtest

# Train with Attention-LSTM model
python train_main.py --model-type Attention-LSTM --backtest
```

### 3. Start the Web Server

```bash
python app.py
```

The web interface will be available at: **http://localhost:5000**

## 📊 Features

### Real-Time Predictions
- Get live trading signals (BUY/HOLD/SELL)
- View prediction confidence and probabilities
- See expected returns
- Auto-refresh every 30 seconds

### Price Charts
- Interactive price charts with Chart.js
- Multiple timeframe support (5m, 15m, 1h, 4h, 1d)
- Real-time price updates

### Backtesting
- Run backtests on historical data
- View comprehensive performance metrics:
  - Total Return
  - Sharpe Ratio
  - Win Rate
  - Profit Factor
  - Max Drawdown
  - Number of Trades
- Visualize equity curve

## 🎨 UI Features

- **Modern Dark Theme**: Beautiful dark UI optimized for trading
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-Time Updates**: Auto-refreshes predictions and charts
- **Interactive Charts**: Zoom, pan, and explore data
- **Status Indicators**: See model connection status

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```

### Get Prediction
```
POST /api/predict
Body: {
  "symbol": "BTCUSDT",
  "timeframe": "5m"
}
```

### Run Backtest
```
POST /api/backtest
Body: {
  "symbol": "BTCUSDT",
  "timeframe": "5m",
  "start_date": "2023-12-01",
  "end_date": "2024-01-01"
}
```

### Get Latest Data
```
GET /api/data/latest?symbol=BTCUSDT&timeframe=5m&limit=100
```

### Load Model
```
POST /api/model/load
Body: {
  "model_path": "models/best_model.pth",
  "model_arch": "Transformer"
}
```

## 🎯 Model Architectures

The web UI supports all model architectures:

1. **LSTM** - Standard LSTM network
2. **CNN-LSTM** - Hybrid CNN-LSTM model
3. **Transformer** - Transformer-based model with multi-head attention (most powerful)
4. **Attention-LSTM** - LSTM with attention mechanism

## 📝 Configuration

The app automatically loads models from `models/best_model.pth`. To load a different model, use the API endpoint or modify `app.py`.

## 🐛 Troubleshooting

### Model Not Loading
- Ensure you have a trained model at `models/best_model.pth`
- Check that the model architecture matches (LSTM, Transformer, etc.)
- Verify the model checkpoint contains the correct state dict

### No Data Available
- Check your internet connection
- Verify Binance API is accessible
- Try downloading data manually first

### Port Already in Use
- Change the port in `app.py`: `app.run(port=5001)`

## 🚀 Deployment

For production deployment:

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set `debug=False` in `app.py`
3. Use environment variables for configuration
4. Set up proper logging
5. Use HTTPS with SSL certificates

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📚 Next Steps

- Add user authentication
- Implement model comparison
- Add more visualization options
- Integrate live trading (with caution!)
- Add alert system for signals

