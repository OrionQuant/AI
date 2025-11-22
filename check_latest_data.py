"""Check the latest available data from Binance."""
from binance.client import Client
import pandas as pd
from datetime import datetime

print("Checking latest data from Binance...")
client = Client()

# Get latest candles
klines = client.get_klines(
    symbol='BTCUSDT',
    interval=Client.KLINE_INTERVAL_5MINUTE,
    limit=10
)

print(f"\nLatest 10 candles from Binance API:")
print(f"{'Timestamp':<25} {'Age (min)':<12} {'Price':<15} {'Status'}")
print("-" * 70)

now = datetime.now()
for k in klines:
    ts = pd.to_datetime(int(k[0]), unit='ms')
    age_minutes = (now - ts).total_seconds() / 60
    price = float(k[4])
    
    if age_minutes < 10:
        status = "🟢 Very Fresh"
    elif age_minutes < 60:
        status = "🟡 Recent"
    else:
        status = "🔴 Old"
    
    print(f"{ts}  {age_minutes:>6.1f} min    ${price:>12,.2f}  {status}")

print(f"\nCurrent time: {now}")
print(f"Latest candle from Binance: {pd.to_datetime(int(klines[-1][0]), unit='ms')}")
latest_age = (now - pd.to_datetime(int(klines[-1][0]), unit='ms')).total_seconds() / 60
print(f"Latest candle age: {latest_age:.1f} minutes ({latest_age/60:.1f} hours)")

