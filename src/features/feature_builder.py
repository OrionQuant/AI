"""
Feature engineering pipeline for BTC trading signals.

Creates technical indicators, normalized features, and sequence windows
for model training.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds features from raw OHLCV data for sequence models.
    
    Features include:
    - Normalized OHLCV
    - Returns (log returns at multiple horizons)
    - Technical indicators (EMA, RSI, MACD, ATR, Bollinger Bands)
    - Volume features
    - Volatility measures
    """
    
    def __init__(self, seq_len: int = 128, normalize: bool = True):
        """
        Initialize feature builder.
        
        Args:
            seq_len: Length of input sequences
            normalize: Whether to normalize features
        """
        self.seq_len = seq_len
        self.normalize = normalize
        self.scaler = RobustScaler() if normalize else None
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from raw OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic price features
        df = self._add_returns(df)
        df = self._add_technical_indicators(df)
        df = self._add_volume_features(df)
        df = self._add_volatility_features(df)
        
        # Remove rows with NaN (from indicator calculations)
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Built {len(df.columns)} features, {len(df)} samples")
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns at multiple horizons."""
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_2'] = np.log(df['close'] / df['close'].shift(2))
        df['log_return_3'] = np.log(df['close'] / df['close'].shift(3))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price change percentages
        df['pct_change_1'] = df['close'].pct_change(1)
        df['pct_change_5'] = df['close'].pct_change(5)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # EMAs
        df['ema_8'] = close.ewm(span=8, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()
        
        # EMA ratios
        df['ema_8_21_ratio'] = df['ema_8'] / df['ema_21']
        df['ema_21_50_ratio'] = df['ema_21'] / df['ema_50']
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(close, period=14)
        
        # MACD
        macd, signal = self._calculate_macd(close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(high, low, close, period=14)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(close, period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_mid
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Price position relative to high/low
        df['high_low_ratio'] = (close - low) / (high - low + 1e-8)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        
        # Volume-price trend
        df['vpt'] = (df['volume'] * df['pct_change_1']).cumsum()
        
        # On-balance volume
        df['obv'] = (np.sign(df['pct_change_1']) * df['volume']).cumsum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures."""
        # Rolling volatility (std of returns)
        df['volatility_5'] = df['log_return_1'].rolling(window=5).std()
        df['volatility_20'] = df['log_return_1'].rolling(window=20).std()
        
        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(window=20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower, ma
    
    def create_sequences(self, df: pd.DataFrame, 
                        feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for model training.
        
        Args:
            df: DataFrame with features
            feature_cols: List of column names to use as features (default: all numeric except timestamp)
            
        Returns:
            X: Sequences of shape (n_samples, seq_len, n_features)
            y_cls: Classification labels (BUY=0, HOLD=1, SELL=2)
            y_reg: Regression labels (future returns)
        """
        if feature_cols is None:
            # Auto-select numeric columns (exclude timestamp and target columns)
            exclude_cols = ['timestamp', 'label_cls', 'label_reg', 'future_return']
            feature_cols = [c for c in df.columns 
                          if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        # Extract feature matrix
        X_raw = df[feature_cols].values
        
        # Normalize if enabled
        if self.normalize and self.scaler is not None:
            X_raw = self.scaler.fit_transform(X_raw)
        
        # Create sequences
        X, y_cls, y_reg = [], [], []
        
        for i in range(self.seq_len, len(df)):
            # Input sequence
            seq = X_raw[i - self.seq_len:i]
            X.append(seq)
            
            # Labels (assumes labels are pre-computed in df)
            if 'label_cls' in df.columns:
                y_cls.append(df.iloc[i]['label_cls'])
            else:
                y_cls.append(1)  # Default HOLD
            
            if 'label_reg' in df.columns:
                y_reg.append(df.iloc[i]['label_reg'])
            else:
                y_reg.append(0.0)
        
        X = np.array(X, dtype=np.float32)
        y_cls = np.array(y_cls, dtype=np.int64)
        y_reg = np.array(y_reg, dtype=np.float32)
        
        logger.info(f"Created {len(X)} sequences of shape {X.shape}")
        return X, y_cls, y_reg
    
    def create_labels(self, df: pd.DataFrame, 
                     horizon: int = 1,
                     buy_threshold: float = 0.005,
                     sell_threshold: float = -0.005) -> pd.DataFrame:
        """
        Create classification and regression labels from future returns.
        
        Args:
            df: DataFrame with close prices
            horizon: Number of steps ahead to predict
            buy_threshold: Minimum return for BUY signal (default: 0.5%)
            sell_threshold: Maximum return for SELL signal (default: -0.5%)
            
        Returns:
            DataFrame with added label_cls and label_reg columns
        """
        df = df.copy()
        
        # Calculate future return
        df['future_return'] = (df['close'].shift(-horizon) - df['close']) / df['close']
        
        # Classification labels: BUY=0, HOLD=1, SELL=2
        df['label_cls'] = 1  # Default HOLD
        df.loc[df['future_return'] > buy_threshold, 'label_cls'] = 0  # BUY
        df.loc[df['future_return'] < sell_threshold, 'label_cls'] = 2  # SELL
        
        # Regression label (future return)
        df['label_reg'] = df['future_return']
        
        # Remove rows where future return is NaN (end of dataset)
        df = df.dropna(subset=['future_return']).reset_index(drop=True)
        
        logger.info(f"Label distribution: BUY={sum(df['label_cls']==0)}, "
                   f"HOLD={sum(df['label_cls']==1)}, SELL={sum(df['label_cls']==2)}")
        
        return df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../../')
    
    from src.data.binance_downloader import BinanceDataDownloader
    
    # Load sample data
    downloader = BinanceDataDownloader()
    df = downloader.load_data("BTCUSDT", "5m")
    
    if len(df) > 0:
        # Build features
        builder = FeatureBuilder(seq_len=128)
        df_features = builder.build_features(df)
        
        # Create labels
        df_labeled = builder.create_labels(df_features, horizon=1, buy_threshold=0.005, sell_threshold=-0.005)
        
        # Create sequences
        X, y_cls, y_reg = builder.create_sequences(df_labeled)
        
        print(f"\nSequences shape: {X.shape}")
        print(f"Labels shape: {y_cls.shape}, {y_reg.shape}")
        print(f"Label distribution: {np.bincount(y_cls)}")

