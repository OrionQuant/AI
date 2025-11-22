"""Update live market data from Binance."""
from src.data.binance_downloader import BinanceDataDownloader
from datetime import datetime
import pandas as pd

print("Updating live market data...")
downloader = BinanceDataDownloader(data_dir="data/raw")

# Update BTC data
symbol = "BTCUSDT"
timeframe = "5m"

print(f"\nUpdating {symbol} {timeframe} data...")
df = downloader.update_data(symbol, timeframe)

if len(df) > 0:
    latest_ts = df.iloc[-1]['timestamp']
    now = datetime.now()
    age_minutes = (now - pd.to_datetime(latest_ts)).total_seconds() / 60
    
    print(f"\n✅ Data updated successfully!")
    print(f"   Total candles: {len(df):,}")
    print(f"   Latest timestamp: {latest_ts}")
    print(f"   Data age: {age_minutes:.1f} minutes ({age_minutes/60:.1f} hours)")
    print(f"   Status: {'🟢 Fresh' if age_minutes < 10 else '🟡 Recent' if age_minutes < 60 else '🔴 Stale'}")
else:
    print("❌ Failed to update data")

