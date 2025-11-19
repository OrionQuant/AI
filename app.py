"""
Flask web application for OrionQuant trading model.

Provides REST API and web interface for:
- Model inference (real-time predictions)
- Training management
- Backtest visualization
- Performance metrics
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.binance_downloader import BinanceDataDownloader
from src.features.feature_builder import FeatureBuilder
from src.models.lstm_signal_net import LSTMSignalNet, CNNLSTMSignalNet
from src.models.transformer_signal_net import TransformerSignalNet, AttentionLSTMNet
from src.backtest.backtester import Backtester, Signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global model and data handlers
model = None
model_type = None
feature_builder = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_downloader = BinanceDataDownloader(data_dir="data/raw")


def load_model(model_path: str = "models/best_model.pth", model_arch: str = "LSTM"):
    """Load trained model."""
    global model, model_type, feature_builder
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Determine input dimension (default to 50 if not in config)
        input_dim = config.get('input_dim', 50)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.2)
        
        # Select model architecture
        # Try to infer from checkpoint if model_arch not specified
        if model_arch == "Transformer" or config.get('type') == 'Transformer':
            model = TransformerSignalNet(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_heads=8,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        elif model_arch == "Attention-LSTM" or config.get('type') == 'Attention-LSTM':
            model = AttentionLSTMNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        elif model_arch == "CNN-LSTM" or config.get('type') == 'CNN-LSTM':
            model = CNNLSTMSignalNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        else:  # Default to LSTM
            model = LSTMSignalNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=config.get('bidirectional', False)
            ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_type = model_arch
        
        # Initialize feature builder
        seq_len = config.get('seq_len', 128)
        normalize = config.get('normalize', True)
        feature_builder = FeatureBuilder(seq_len=seq_len, normalize=normalize)
        
        logger.info(f"Model loaded successfully: {model_arch}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type,
        'device': str(device)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get prediction for current market data."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '5m')
        
        # Download latest data
        df = data_downloader.load_data(symbol, timeframe)
        if len(df) == 0:
            df = data_downloader.download_historical(symbol, timeframe, 
                                                     start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 400
        
        # Build features
        df_features = feature_builder.build_features(df)
        
        # Get last sequence
        if len(df_features) < feature_builder.seq_len:
            return jsonify({'error': 'Insufficient data'}), 400
        
        # Create sequence
        X, _, _ = feature_builder.create_sequences(df_features.iloc[-feature_builder.seq_len:])
        
        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X[-1:]).unsqueeze(0).to(device)
            logits, pred_return = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            signal = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            expected_return = pred_return.item()
        
        signal_names = ['BUY', 'HOLD', 'SELL']
        
        return jsonify({
            'signal': signal_names[signal],
            'signal_id': signal,
            'confidence': float(confidence),
            'probabilities': {
                'BUY': float(probs[0][0].item()),
                'HOLD': float(probs[0][1].item()),
                'SELL': float(probs[0][2].item())
            },
            'expected_return': float(expected_return),
            'current_price': float(df.iloc[-1]['close']),
            'timestamp': df.iloc[-1]['timestamp'].isoformat() if 'timestamp' in df.columns else datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest on historical data."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '5m')
        start_date = data.get('start_date', '2023-12-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Download data
        df = data_downloader.download_historical(symbol, timeframe, start_date=start_date)
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 400
        
        # Filter by date range
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        if len(df) == 0:
            return jsonify({'error': 'No data in date range'}), 400
        
        # Build features and create sequences
        df_features = feature_builder.build_features(df)
        df_labeled = feature_builder.create_labels(df_features, horizon=1, buy_threshold=0.005, sell_threshold=-0.005)
        X, y_cls, y_reg = feature_builder.create_sequences(df_labeled)
        
        # Get predictions
        signals = []
        confidences = []
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y_cls),
            torch.FloatTensor(y_reg)
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(device)
                logits, pred_return = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                pred_signal = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                signals.extend(pred_signal.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        signals = np.array(signals)
        confidences = np.array(confidences)
        
        # Pad signals to match dataframe length
        pad_signals = np.full(len(df_labeled) - len(signals), Signal.HOLD.value)
        signals_full = np.concatenate([pad_signals, signals])
        confidences_full = np.concatenate([np.ones(len(pad_signals)), confidences])
        
        # Run backtest
        backtester = Backtester(
            initial_capital=10000.0,
            taker_fee=0.001,
            maker_fee=0.0005,
            slippage_pct=0.0005,
            max_position_size=1.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )
        
        results = backtester.run_backtest(df_labeled, signals_full, confidences_full)
        
        # Format results for JSON
        return jsonify({
            'total_return': float(results['total_return']),
            'total_return_pct': float(results['total_return_pct']),
            'num_trades': int(results['num_trades']),
            'win_rate': float(results['win_rate']),
            'profit_factor': float(results['profit_factor']),
            'sharpe_ratio': float(results['sharpe_ratio']),
            'max_drawdown': float(results['max_drawdown']),
            'avg_trade_return': float(results['avg_trade_return']),
            'equity_curve': [float(x) for x in results['equity_curve']],
            'timestamps': [str(ts) for ts in results['timestamps']] if 'timestamps' in results else []
        })
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/latest', methods=['GET'])
def get_latest_data():
    """Get latest market data."""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '5m')
        limit = int(request.args.get('limit', 100))
        
        df = data_downloader.load_data(symbol, timeframe)
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 400
        
        # Get last N rows
        df = df.tail(limit)
        
        # Convert to JSON
        data = []
        for _, row in df.iterrows():
            data.append({
                'timestamp': row['timestamp'].isoformat() if 'timestamp' in row else None,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model_endpoint():
    """Load a model."""
    data = request.json
    model_path = data.get('model_path', 'models/best_model.pth')
    model_arch = data.get('model_arch', 'LSTM')
    
    success = load_model(model_path, model_arch)
    
    if success:
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load model'}), 400


if __name__ == '__main__':
    # Try to load default model
    load_model()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

