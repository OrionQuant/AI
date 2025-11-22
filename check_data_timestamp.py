import pandas as pd
import glob
from datetime import datetime

# Check data files
files = glob.glob('data/raw/BTCUSDT_5m_*.parquet')
print(f"Found {len(files)} data files")

if files:
    df = pd.concat([pd.read_parquet(f) for f in files[:2]])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print('\n=== Data Timestamp Analysis ===')
    print(f'Total records: {len(df)}')
    print(f'\nOldest timestamp: {df.iloc[0]["timestamp"]}')
    print(f'Newest timestamp: {df.iloc[-1]["timestamp"]}')
    
    # Check timestamp type
    print(f'\nTimestamp type: {type(df.iloc[-1]["timestamp"])}')
    
    # Convert to datetime if needed
    if isinstance(df.iloc[-1]['timestamp'], str):
        last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    else:
        last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    
    now = datetime.now()
    age_minutes = (now - last_ts).total_seconds() / 60
    age_hours = age_minutes / 60
    age_days = age_hours / 24
    
    print(f'\nCurrent time: {now}')
    print(f'Last data timestamp: {last_ts}')
    print(f'\nData age:')
    print(f'  {age_minutes:.1f} minutes')
    print(f'  {age_hours:.1f} hours')
    print(f'  {age_days:.2f} days')
    
    print(f'\nIs data fresh? (< 10 minutes): {age_minutes < 10}')
    print(f'Is data recent? (< 1 hour): {age_minutes < 60}')
    
    # Check timezone
    print(f'\nTimezone info:')
    print(f'  Last timestamp timezone: {last_ts.tz}')
    print(f'  Current time timezone: {now.tzinfo}')

