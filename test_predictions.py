"""Test model predictions to see probability distribution."""
import requests
import json

print("Testing model predictions...\n")

# Make multiple predictions to see the distribution
for i in range(5):
    try:
        r = requests.post('http://localhost:5000/api/predict', 
                         json={'symbol': 'BTCUSDT', 'timeframe': '5m'})
        if r.status_code == 200:
            data = r.json()
            print(f"Prediction {i+1}:")
            print(f"  Signal: {data['signal']}")
            print(f"  Confidence: {data['confidence']*100:.1f}%")
            print(f"  Probabilities:")
            print(f"    BUY:  {data['probabilities']['BUY']*100:6.2f}%")
            print(f"    HOLD: {data['probabilities']['HOLD']*100:6.2f}%")
            print(f"    SELL: {data['probabilities']['SELL']*100:6.2f}%")
            print(f"  Expected Return: {data['expected_return']*100:.2f}%")
            print()
    except Exception as e:
        print(f"Error: {e}")
        break

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("If HOLD is always > 99%, the model was trained on imbalanced data.")
print("Solution: Retrain the model with:")
print("  1. Class weights (already fixed in code)")
print("  2. Lower thresholds (0.2% instead of 0.5%)")
print("  3. Run: python train_main.py --model-type Transformer --backtest")

