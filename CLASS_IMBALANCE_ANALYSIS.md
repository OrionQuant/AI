# Why the Model Predicts HOLD Too Often

## 🔍 Problem Identified

### Current Label Distribution:
- **BUY** (>0.5% return): 97 samples (0.3%)
- **HOLD** (-0.5% to 0.5%): 35,256 samples (99.5%)
- **SELL** (<-0.5% return): 73 samples (0.2%)

**Problem:** HOLD is **363x more common** than BUY!

### Why This Happens:
1. **5-minute timeframe is too short** - Most price movements are small
2. **0.5% threshold is too strict** - Very few 5-minute candles move >0.5%
3. **No class balancing** - Model learns to always predict HOLD (99.5% accuracy!)

### Current Behavior:
- Model sees 99.5% HOLD examples
- Learns: "Always predict HOLD = 99.5% accuracy"
- This is "correct" but useless for trading!

---

## ✅ Solutions

### Solution 1: Adjust Thresholds (Easiest)
Make thresholds smaller to get more BUY/SELL signals:
- Current: BUY > 0.5%, SELL < -0.5%
- Better: BUY > 0.2%, SELL < -0.2%
- Or: BUY > 0.15%, SELL < -0.15%

### Solution 2: Add Class Weights (Recommended)
Weight the loss function to penalize HOLD predictions more:
- Give BUY/SELL 100x more weight than HOLD
- Forces model to learn actual patterns

### Solution 3: Use Confidence Thresholds
Instead of always using argmax, use confidence:
- Only predict BUY if confidence > 70%
- Only predict SELL if confidence > 70%
- Otherwise HOLD

### Solution 4: Focal Loss
Use focal loss which focuses on hard examples

---

## 🛠️ Implementation

See the fixes in the code - I'll implement Solution 1 + 2 + 3 combined!

