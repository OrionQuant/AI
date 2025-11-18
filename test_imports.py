"""
Test script to verify all modules can be imported and basic functionality works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("Testing OrionQuant Module Imports")
print("="*60)

errors = []

# Test 1: Check if we can import standard libraries
print("\n[1] Testing standard library imports...")
try:
    import numpy as np
    import pandas as pd
    print("[OK] numpy, pandas OK")
except ImportError as e:
    errors.append(f"Standard libraries: {e}")
    print(f"[ERROR] Missing: {e}")

# Test 2: Test data downloader (structure only, no API calls)
print("\n[2] Testing data downloader module...")
try:
    from src.data.binance_downloader import BinanceDataDownloader
    print("[OK] BinanceDataDownloader imported")
    
    # Check class structure
    assert hasattr(BinanceDataDownloader, 'download_historical')
    assert hasattr(BinanceDataDownloader, 'load_data')
    print("[OK] Class methods exist")
except Exception as e:
    errors.append(f"Data downloader: {e}")
    print(f"[ERROR] Error: {e}")

# Test 3: Test feature builder
print("\n[3] Testing feature builder module...")
try:
    from src.features.feature_builder import FeatureBuilder
    print("[OK] FeatureBuilder imported")
    
    # Check class structure
    assert hasattr(FeatureBuilder, 'build_features')
    assert hasattr(FeatureBuilder, 'create_sequences')
    assert hasattr(FeatureBuilder, 'create_labels')
    print("[OK] Class methods exist")
except Exception as e:
    errors.append(f"Feature builder: {e}")
    print(f"[ERROR] Error: {e}")

# Test 4: Test model architectures
print("\n[4] Testing model architectures...")
try:
    from src.models.lstm_signal_net import LSTMSignalNet, CNNLSTMSignalNet
    print("[OK] Model classes imported")
    
    # Check class structure
    assert hasattr(LSTMSignalNet, 'forward')
    assert hasattr(LSTMSignalNet, 'predict_signal')
    print("[OK] Model methods exist")
except Exception as e:
    errors.append(f"Models: {e}")
    print(f"[ERROR] Error: {e}")

# Test 5: Test training module
print("\n[5] Testing training module...")
try:
    from src.training.train import train_model, SeqDataset, set_seed
    print("[OK] Training functions imported")
except Exception as e:
    errors.append(f"Training: {e}")
    print(f"[ERROR] Error: {e}")

# Test 6: Test walk-forward validation
print("\n[6] Testing walk-forward validation...")
try:
    from src.training.walk_forward import WalkForwardValidator, create_time_based_splits
    print("[OK] Walk-forward functions imported")
except Exception as e:
    errors.append(f"Walk-forward: {e}")
    print(f"[ERROR] Error: {e}")

# Test 7: Test backtester
print("\n[7] Testing backtester...")
try:
    from src.backtest.backtester import Backtester, Signal, Trade
    print("[OK] Backtester classes imported")
    
    # Check Signal enum
    assert Signal.BUY.value == 0
    assert Signal.HOLD.value == 1
    assert Signal.SELL.value == 2
    print("[OK] Signal enum values correct")
except Exception as e:
    errors.append(f"Backtester: {e}")
    print(f"[ERROR] Error: {e}")

# Test 8: Test feature builder with dummy data (if pandas available)
print("\n[8] Testing feature builder with dummy data...")
try:
    import pandas as pd
    import numpy as np
    
    # Create dummy OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000
    })
    
    builder = FeatureBuilder(seq_len=32, normalize=False)  # Small seq_len for test
    df_features = builder.build_features(df)
    print(f"[OK] Built features: {len(df_features.columns)} columns")
    
    df_labeled = builder.create_labels(df_features, horizon=1)
    print(f"[OK] Created labels: {df_labeled['label_cls'].value_counts().to_dict()}")
    
except Exception as e:
    errors.append(f"Feature builder test: {e}")
    print(f"[ERROR] Error: {e}")

# Test 9: Test backtester with dummy data
print("\n[9] Testing backtester with dummy data...")
try:
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.random.randn(100).cumsum() * 10
    })
    
    signals = np.random.randint(0, 3, len(df))
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run_backtest(df, signals)
    
    print(f"[OK] Backtest completed: {results['num_trades']} trades")
    print(f"  Total return: ${results['total_return']:.2f}")
    
except Exception as e:
    errors.append(f"Backtester test: {e}")
    print(f"[ERROR] Error: {e}")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

if len(errors) == 0:
    print("[SUCCESS] All tests passed! Code structure is correct.")
    print("\nNote: To run full training, install dependencies:")
    print("  pip install -r requirements.txt")
else:
    print(f"[FAILED] Found {len(errors)} error(s):")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")

print("="*60)

