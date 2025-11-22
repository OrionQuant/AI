import requests
import json

try:
    # Test health
    r = requests.get('http://localhost:5000/api/health')
    health = r.json()
    print("✅ Server Status:")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Model type: {health.get('model_type', 'N/A')}")
    print(f"   Device: {health['device']}")
    
    # Test prediction
    print("\n✅ Testing Prediction:")
    r = requests.post('http://localhost:5000/api/predict', 
                     json={'symbol': 'BTCUSDT', 'timeframe': '5m'})
    if r.status_code == 200:
        data = r.json()
        print(f"   Signal: {data.get('signal', 'N/A')}")
        print(f"   Confidence: {data.get('confidence', 0)*100:.1f}%")
        print(f"   Current Price: ${data.get('current_price', 0):,.2f}")
        print(f"   Expected Return: {data.get('expected_return', 0)*100:.2f}%")
        print("\n🎉 Everything is working!")
    else:
        print(f"   Error: {r.status_code}")
        print(f"   {r.text}")
except Exception as e:
    print(f"❌ Error: {e}")

