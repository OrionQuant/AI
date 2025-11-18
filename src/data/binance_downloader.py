"""
Binance data downloader for historical and live market data.

Supports multiple timeframes, efficient storage, and incremental updates.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataDownloader:
    """
    Downloads and manages historical BTC market data from Binance.
    
    Features:
    - Multi-timeframe support (1m, 5m, 15m, 1h, 4h, 1d)
    - Efficient storage in parquet format
    - Incremental updates
    - Error handling and retry logic
    """
    
    # Binance timeframe mapping
    TIMEFRAMES = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 data_dir: str = "data/raw"):
        """
        Initialize Binance client and data directory.
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            data_dir: Directory to store downloaded data
        """
        self.client = Client(api_key=api_key, api_secret=api_secret)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_historical(self, 
                           symbol: str = "BTCUSDT",
                           timeframe: str = "5m",
                           start_date: str = "2022-01-01",
                           end_date: Optional[str] = None,
                           save: bool = True) -> pd.DataFrame:
        """
        Download historical kline/candlestick data.
        
        Args:
            symbol: Trading pair (default: BTCUSDT)
            timeframe: One of '1m', '5m', '15m', '1h', '4h', '1d'
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            save: Whether to save to parquet file
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(self.TIMEFRAMES.keys())}")
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        interval_ms = self._get_interval_ms(timeframe)
        batch_size_ms = 1000 * interval_ms  # Binance limit: 1000 candles per request
        
        while current_ts < end_ts:
            try:
                # Request batch
                end_batch_ts = min(current_ts + batch_size_ms, end_ts)
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=self.TIMEFRAMES[timeframe],
                    start_str=str(current_ts),
                    end_str=str(end_batch_ts)
                )
                
                if not klines:
                    break
                
                # Parse klines
                for k in klines:
                    all_data.append({
                        'timestamp': pd.to_datetime(k[0], unit='ms'),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'quote_volume': float(k[7]),
                        'trades': int(k[8]),
                    })
                
                current_ts = end_batch_ts
                logger.info(f"Downloaded {len(all_data)} candles...")
                
                # Rate limiting
                time.sleep(0.1)
                
            except BinanceAPIException as e:
                logger.error(f"Binance API error: {e}")
                if e.code == -1003:  # Rate limit
                    logger.info("Rate limited, waiting 60s...")
                    time.sleep(60)
                else:
                    raise
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        logger.info(f"Downloaded {len(df)} candles total")
        
        if save and len(df) > 0:
            self._save_data(df, symbol, timeframe, start_date, end_date)
        
        return df
    
    def _get_interval_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        mapping = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return mapping[timeframe]
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                   start_date: str, end_date: str):
        """Save DataFrame to parquet file."""
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, symbol: str = "BTCUSDT", timeframe: str = "5m") -> pd.DataFrame:
        """
        Load data from parquet files matching symbol and timeframe.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            
        Returns:
            Combined DataFrame sorted by timestamp
        """
        import glob
        
        pattern = os.path.join(self.data_dir, f"{symbol}_{timeframe}_*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No data files found for {symbol} {timeframe}")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        combined = combined.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} candles from {len(files)} files")
        return combined
    
    def update_data(self, symbol: str = "BTCUSDT", timeframe: str = "5m") -> pd.DataFrame:
        """
        Incrementally update existing data with latest candles.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            
        Returns:
            Updated DataFrame
        """
        # Load existing data
        df_existing = self.load_data(symbol, timeframe)
        
        if len(df_existing) == 0:
            logger.info("No existing data found, downloading from scratch")
            return self.download_historical(symbol, timeframe, 
                                          start_date=(datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d"))
        
        # Get last timestamp
        last_ts = df_existing['timestamp'].max()
        start_date = (pd.to_datetime(last_ts) + timedelta(minutes=1)).strftime("%Y-%m-%d")
        
        logger.info(f"Updating data from {start_date}")
        df_new = self.download_historical(symbol, timeframe, start_date=start_date, save=False)
        
        if len(df_new) > 0:
            # Combine and deduplicate
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            df_combined = df_combined.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # Save updated data
            start_date_all = df_combined['timestamp'].min().strftime("%Y-%m-%d")
            end_date_all = df_combined['timestamp'].max().strftime("%Y-%m-%d")
            self._save_data(df_combined, symbol, timeframe, start_date_all, end_date_all)
            
            return df_combined
        
        return df_existing


if __name__ == "__main__":
    # Example usage
    downloader = BinanceDataDownloader()
    
    # Download 2 years of 5m data
    df = downloader.download_historical(
        symbol="BTCUSDT",
        timeframe="5m",
        start_date="2022-01-01"
    )
    
    print(f"\nDownloaded {len(df)} candles")
    print(df.head())
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")

