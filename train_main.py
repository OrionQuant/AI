"""
Main training script for BTC trading signal model.

Orchestrates data loading, feature engineering, training, and backtesting.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.binance_downloader import BinanceDataDownloader
from src.features.feature_builder import FeatureBuilder
from src.models.lstm_signal_net import LSTMSignalNet, CNNLSTMSignalNet
from src.training.train import train_model
from src.training.walk_forward import create_time_based_splits
from src.backtest.backtester import Backtester, Signal
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def download_data(config: dict, force_download: bool = False) -> pd.DataFrame:
    """Download or load historical data."""
    downloader = BinanceDataDownloader(data_dir="data/raw")
    
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
    # Try to load existing data
    if not force_download:
        df = downloader.load_data(symbol, timeframe)
        if len(df) > 0:
            logger.info(f"Loaded existing data: {len(df)} candles")
            # Update with latest data
            df = downloader.update_data(symbol, timeframe)
            return df
    
    # Download fresh data
    logger.info("Downloading historical data...")
    start_date = config['data'].get('train_start', '2022-01-01')
    df = downloader.download_historical(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )
    
    return df


def prepare_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Prepare data for training: build features, create labels, split.
    
    Returns:
        (X_train, y_train_cls, y_train_reg, X_val, y_val_cls, y_val_reg,
         X_test, y_test_cls, y_test_reg, feature_builder)
    """
    # Build features
    logger.info("Building features...")
    feature_builder = FeatureBuilder(
        seq_len=config['data']['seq_len'],
        normalize=config['features']['normalize']
    )
    
    df_features = feature_builder.build_features(df)
    
    # Create labels
    logger.info("Creating labels...")
    df_labeled = feature_builder.create_labels(
        df_features,
        horizon=config['data']['horizon'],
        buy_threshold=config['data']['buy_threshold'],
        sell_threshold=config['data']['sell_threshold']
    )
    
    # Time-based splits
    logger.info("Creating train/val/test splits...")
    splits = create_time_based_splits(
        df_labeled,
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end']
    )
    
    # Create sequences
    logger.info("Creating sequences...")
    X_train, y_train_cls, y_train_reg = feature_builder.create_sequences(splits['train_df'])
    X_val, y_val_cls, y_val_reg = feature_builder.create_sequences(splits['val_df'])
    X_test, y_test_cls, y_test_reg = feature_builder.create_sequences(splits['test_df'])
    
    logger.info(f"Train: {len(X_train)} sequences")
    logger.info(f"Val: {len(X_val)} sequences")
    logger.info(f"Test: {len(X_test)} sequences")
    
    return (X_train, y_train_cls, y_train_reg,
            X_val, y_val_cls, y_val_reg,
            X_test, y_test_cls, y_test_reg,
            feature_builder)


def run_backtest(model, test_df: pd.DataFrame, feature_builder: FeatureBuilder, 
                config: dict, device: torch.device):
    """Run backtest on test set."""
    logger.info("Running backtest...")
    
    # Create test sequences
    X_test, y_test_cls, y_test_reg = feature_builder.create_sequences(test_df)
    
    # Get predictions
    model.eval()
    signals = []
    confidences = []
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test_cls),
        torch.FloatTensor(y_test_reg)
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
    
    # Align signals with test_df (account for sequence length)
    # Pad with HOLD signals at the beginning
    pad_signals = np.full(len(test_df) - len(signals), Signal.HOLD.value)
    signals_full = np.concatenate([pad_signals, signals])
    confidences_full = np.concatenate([np.ones(len(pad_signals)), confidences])
    
    # Run backtest
    backtester = Backtester(
        initial_capital=config['backtest']['initial_capital'],
        taker_fee=config['backtest']['taker_fee'],
        maker_fee=config['backtest']['maker_fee'],
        slippage_pct=config['backtest']['slippage_pct'],
        max_position_size=config['backtest']['max_position_size'],
        stop_loss_pct=config['backtest']['stop_loss_pct'],
        take_profit_pct=config['backtest']['take_profit_pct']
    )
    
    results = backtester.run_backtest(test_df, signals_full, confidences_full)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Average Trade Return: ${results['avg_trade_return']:.2f}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train BTC trading signal model')
    parser.add_argument('--config', type=str, default='config/train_config.json',
                       help='Path to config file')
    parser.add_argument('--download', action='store_true',
                       help='Force download fresh data')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest after training')
    parser.add_argument('--model-type', type=str, choices=['LSTM', 'CNN-LSTM'],
                       default='LSTM', help='Model architecture')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Download/load data
    df = download_data(config, force_download=args.download)
    
    if len(df) == 0:
        logger.error("No data available. Please download data first.")
        return
    
    # Prepare data
    (X_train, y_train_cls, y_train_reg,
     X_val, y_val_cls, y_val_reg,
     X_test, y_test_cls, y_test_reg,
     feature_builder) = prepare_data(df, config)
    
    # Select model
    model_class = LSTMSignalNet if args.model_type == 'LSTM' else CNNLSTMSignalNet
    
    # Training config
    train_config = {
        **config['model'],
        **config['training'],
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout']
    }
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg,
        train_config,
        model_class,
        save_dir="models"
    )
    
    # Run backtest if requested
    if args.backtest:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Get test dataframe
        splits = create_time_based_splits(
            feature_builder.create_labels(
                feature_builder.build_features(df),
                horizon=config['data']['horizon'],
                buy_threshold=config['data']['buy_threshold'],
                sell_threshold=config['data']['sell_threshold']
            ),
            train_start=config['data']['train_start'],
            train_end=config['data']['train_end'],
            val_start=config['data']['val_start'],
            val_end=config['data']['val_end'],
            test_start=config['data']['test_start'],
            test_end=config['data']['test_end']
        )
        
        run_backtest(model, splits['test_df'], feature_builder, config, device)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

