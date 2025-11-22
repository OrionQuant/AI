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
# Ensure data directory exists
os.makedirs("data/raw", exist_ok=True)
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
        
        # Determine input dimension from state dict if not in config
        # Check the actual input_projection weight shape
        state_dict = checkpoint['model_state_dict']
        if 'input_projection.weight' in state_dict:
            # input_projection.weight shape is (d_model, input_dim)
            input_dim = state_dict['input_projection.weight'].shape[1]
            logger.info(f"Detected input_dim from checkpoint: {input_dim}")
        elif 'lstm.weight_ih_l0' in state_dict:
            # For LSTM: weight_ih_l0 shape is (4*hidden_dim, input_dim)
            input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
            logger.info(f"Detected input_dim from LSTM checkpoint: {input_dim}")
        else:
            input_dim = config.get('input_dim', 50)
            logger.warning(f"Could not detect input_dim, using default: {input_dim}")
        
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.2)
        
        # Select model architecture
        # Check state dict keys to determine actual model type
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        is_transformer = any('transformer_blocks' in k or 'pos_encoding' in k or 'input_projection' in k for k in state_dict_keys)
        is_attention_lstm = any('attention' in k and 'lstm' in k.lower() for k in state_dict_keys) and not is_transformer
        is_cnn_lstm = any('conv' in k.lower() for k in state_dict_keys) and not is_transformer
        
        # Try to infer from checkpoint if model_arch not specified
        if model_arch == "Transformer" or config.get('type') == 'Transformer' or is_transformer:
            model = TransformerSignalNet(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_heads=8,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        elif model_arch == "Attention-LSTM" or config.get('type') == 'Attention-LSTM' or is_attention_lstm:
            model = AttentionLSTMNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        elif model_arch == "CNN-LSTM" or config.get('type') == 'CNN-LSTM' or is_cnn_lstm:
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
        
        # Set model_type based on what was actually loaded
        if is_transformer:
            model_type = "Transformer"
        elif is_attention_lstm:
            model_type = "Attention-LSTM"
        elif is_cnn_lstm:
            model_type = "CNN-LSTM"
        else:
            model_type = "LSTM"
        
        # Initialize feature builder
        seq_len = config.get('seq_len', 128)
        normalize = config.get('normalize', True)
        feature_builder = FeatureBuilder(seq_len=seq_len, normalize=normalize)
        
        logger.info(f"Model loaded successfully: {model_type}")
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
    # Check if we have recent data
    try:
        df = data_downloader.load_data('BTCUSDT', '5m')
        has_recent_data = False
        data_age_minutes = None
        if len(df) > 0 and 'timestamp' in df.columns:
            last_timestamp = pd.to_datetime(df.iloc[-1]['timestamp'])
            data_age_minutes = (datetime.now() - last_timestamp).total_seconds() / 60
            has_recent_data = data_age_minutes < 10  # Consider "live" if data is less than 10 minutes old
    except:
        has_recent_data = False
        data_age_minutes = None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type,
        'device': str(device),
        'live_data': has_recent_data,
        'data_age_minutes': round(data_age_minutes, 1) if data_age_minutes is not None else None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get prediction for current market data."""
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '5m')
        
        # Try to load existing data first (so we can show price even without model)
        df = data_downloader.load_data(symbol, timeframe)
        
        # Always update with latest data for live predictions
        try:
            logger.info(f"Updating data for {symbol} {timeframe}...")
            df_updated = data_downloader.update_data(symbol, timeframe)
            if len(df_updated) > 0:
                df = df_updated
                latest_ts = df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else None
                if latest_ts:
                    from datetime import datetime
                    age_minutes = (datetime.now() - pd.to_datetime(latest_ts)).total_seconds() / 60
                    age_hours = age_minutes / 60
                    if age_minutes < 10:
                        status = "🟢 Fresh"
                    elif age_minutes < 60:
                        status = "🟡 Recent"
                    else:
                        status = "🔴 Stale"
                    logger.info(f"Data updated: {len(df)} candles, latest: {latest_ts}, age: {age_minutes:.1f} min ({age_hours:.1f} hours) {status}")
        except Exception as e:
            logger.warning(f"Could not update data: {e}, using cached data")
        
        # If no data found, download it automatically
        if len(df) == 0:
            logger.info(f"No data found for {symbol} {timeframe}, downloading...")
            try:
                # Download last 60 days of data to ensure we have enough
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                df = data_downloader.download_historical(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    start_date=start_date,
                    save=True
                )
                if len(df) == 0:
                    return jsonify({
                        'error': f'Failed to download data for {symbol} {timeframe}. Please check your internet connection and try again.'
                    }), 400
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                return jsonify({
                    'error': f'Failed to download data: {str(e)}. Please check your internet connection.'
                }), 500
        
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 400
        
        # Get current price - ALWAYS use live price from ticker for real-time data
        current_price = None
        current_timestamp = None
        is_live = False
        
        # Get live price from ticker (most up-to-date, real-time)
        try:
            live_ticker = data_downloader.get_live_price(symbol)
            if live_ticker:
                current_price = live_ticker['price']
                current_timestamp = live_ticker['timestamp'].isoformat()
                is_live = True
                logger.info(f"✅ Using LIVE market price: ${current_price:,.2f} at {live_ticker['timestamp']}")
        except Exception as e:
            logger.warning(f"Could not get live price: {e}")
        
        # Fallback to last candle if live price not available
        if current_price is None and len(df) > 0:
            current_price = float(df.iloc[-1]['close'])
            current_timestamp = df.iloc[-1]['timestamp'].isoformat() if 'timestamp' in df.columns else datetime.now().isoformat()
            is_live = False
            logger.info(f"Using last candle price: ${current_price:,.2f}")
        elif current_price is None:
            current_price = 0.0
            current_timestamp = datetime.now().isoformat()
            is_live = False
        
        # Check if model is loaded
        if model is None or feature_builder is None:
            return jsonify({
                'error': 'Model not loaded. Please train a model first using: python train_main.py --model-type Transformer --backtest',
                'current_price': current_price,
                'timestamp': current_timestamp
            }), 400
        
        # Build features
        df_features = feature_builder.build_features(df)
        
        # Get last sequence - need enough data for sequence length
        if len(df_features) < feature_builder.seq_len:
            return jsonify({
                'error': f'Insufficient data. Need at least {feature_builder.seq_len} samples, got {len(df_features)}',
                'current_price': current_price,
                'timestamp': current_timestamp
            }), 400
        
        # Create sequences - need to pass full dataframe
        # The create_sequences function expects a full dataframe and creates sequences internally
        try:
            sequences = feature_builder.create_sequences(df_features)
            if len(sequences) == 3:
                X, _, _ = sequences
            else:
                X = sequences[0] if len(sequences) > 0 else np.array([])
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return jsonify({
                'error': f'Error creating sequences: {str(e)}',
                'current_price': current_price,
                'timestamp': current_timestamp
            }), 500
        
        if len(X) == 0:
            return jsonify({
                'error': 'Could not create sequences from data. Need more historical data.',
                'current_price': current_price,
                'timestamp': current_timestamp
            }), 400
        
        # Get the last sequence
        X_last = X[-1:]  # Take only the last sequence (shape: (1, seq_len, n_features))
        
        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X_last).to(device)  # Already has batch dimension
            logits, pred_return = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions - use the actual model prediction
            max_prob, predicted_class = torch.max(probs, dim=1)
            confidence = max_prob.item()
            predicted_class = predicted_class.item()
            
            # Get probabilities for each class
            buy_prob = probs[0][0].item()
            hold_prob = probs[0][1].item()
            sell_prob = probs[0][2].item()
            
            # Get expected return from regression head
            expected_return = pred_return[0].item()
            
            # Smart prediction strategy to handle class imbalance:
            # Since the model was trained on imbalanced data (99.5% HOLD), we need to
            # use a more sophisticated approach that considers both probabilities AND expected return
            
            signal = predicted_class
            
            # Strategy: Use expected return as a tie-breaker when probabilities are close
            # If expected return is strong and BUY/SELL probability is reasonable, override HOLD
            
            # Calculate relative probabilities (how much better is BUY/SELL vs HOLD)
            buy_vs_hold = buy_prob / (hold_prob + 1e-8)
            sell_vs_hold = sell_prob / (hold_prob + 1e-8)
            
            # More aggressive override for imbalanced models:
            # Since expected return is usually small, use very lenient thresholds
            if predicted_class == 1:  # HOLD predicted
                # Override if expected return suggests action (even small amounts)
                if expected_return > 0.0005:  # >0.05% return (very lenient)
                    if buy_prob > 0.001:  # At least 0.1% BUY probability
                        signal = 0  # Override to BUY
                        logger.info(f"🟢 Overriding HOLD → BUY: return={expected_return:.4f}, buy_prob={buy_prob:.3f}")
                elif expected_return < -0.0005:  # <-0.05% return (very lenient)
                    if sell_prob > 0.001:  # At least 0.1% SELL probability
                        signal = 2  # Override to SELL
                        logger.info(f"🔴 Overriding HOLD → SELL: return={expected_return:.4f}, sell_prob={sell_prob:.3f}")
            
            # Log the probabilities for debugging
            logger.info(f"Prediction: signal={signal}, probs=[BUY={buy_prob:.3f}, HOLD={hold_prob:.3f}, SELL={sell_prob:.3f}], expected_return={expected_return:.4f}")
        
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
            'current_price': current_price,  # Live price from ticker
            'timestamp': current_timestamp,  # Live timestamp
            'is_live': is_live  # Flag indicating if this is live data
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
        try:
            df = data_downloader.download_historical(
                symbol=symbol, 
                timeframe=timeframe, 
                start_date=start_date,
                save=True
            )
            if len(df) == 0:
                return jsonify({
                    'error': f'Failed to download data for {symbol} {timeframe}. Please check your internet connection.'
                }), 400
        except Exception as e:
            logger.error(f"Error downloading data for backtest: {e}")
            return jsonify({
                'error': f'Failed to download data: {str(e)}'
            }), 500
        
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
    """Get latest market data. Works even without a model."""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '5m')
        limit = int(request.args.get('limit', 100))
        
        # Try to load existing data
        df = data_downloader.load_data(symbol, timeframe)
        
        # If no data found, download it automatically
        if len(df) == 0:
            logger.info(f"No data found for {symbol} {timeframe}, downloading...")
            try:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                df = data_downloader.download_historical(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    save=True
                )
                logger.info(f"Downloaded {len(df)} candles for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                return jsonify({
                    'error': f'Failed to download data: {str(e)}. Please check your internet connection and Binance API availability.'
                }), 500
        
        if len(df) == 0:
            return jsonify({'error': 'No data available. Please check your internet connection.'}), 400
        
        # Get last N rows
        df = df.tail(limit)
        
        # Try to get live price and add it as the most recent data point
        try:
            live_ticker = data_downloader.get_live_price(symbol)
            if live_ticker:
                # Add live price as the latest data point
                live_data_point = {
                    'timestamp': live_ticker['timestamp'].isoformat(),
                    'open': live_ticker['price'],
                    'high': live_ticker['price'],
                    'low': live_ticker['price'],
                    'close': live_ticker['price'],
                    'volume': 0.0  # Volume not available from ticker
                }
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
                # Add live price as the last point
                data.append(live_data_point)
                return jsonify({'data': data, 'has_live_data': True})
        except Exception as e:
            logger.debug(f"Could not add live price to chart: {e}")
        
        # Convert to JSON (fallback without live data)
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
        
        return jsonify({'data': data, 'has_live_data': False})
    
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
    # Try to load default model - auto-detect type
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        try:
            logger.info("Loading model from checkpoint...")
            # Try Transformer first (most likely since we just trained one)
            success = load_model(model_path, 'Transformer')
            if not success:
                logger.warning("Failed to load as Transformer, trying auto-detect...")
                # Auto-detect will happen inside load_model
                load_model(model_path, 'LSTM')  # Will auto-detect inside
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
    else:
        logger.info("No model found. Train a model first using: python train_main.py --model-type Transformer --backtest")
    
    logger.info(f"Final status - Model loaded: {model is not None}, Model type: {model_type}, Feature builder: {feature_builder is not None}")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

