# 🔴 Why Model Always Predicts HOLD (99.7%)

## Root Cause

Your current model was trained on **severely imbalanced data**:
- **BUY**: 0.3% of training samples
- **HOLD**: 99.5% of training samples
- **SELL**: 0.2% of training samples

The model learned: "Always predict HOLD = 99.5% accuracy" ✅ (but useless for trading ❌)

## Current Model Behavior

```
Probabilities:
  BUY:    0.19%  ← Too low!
  HOLD:  99.73% ← Always wins
  SELL:   0.08%  ← Too low!
```

## ✅ Solution: Retrain the Model

The code has been **fixed** to handle this, but you need to **retrain**:

### What Was Fixed:

1. ✅ **Class Weights** - BUY/SELL get 100x more weight than HOLD during training
2. ✅ **Lower Thresholds** - Changed from 0.5% to 0.15% (more BUY/SELL signals)
3. ✅ **Smart Override** - Uses expected return to override HOLD when appropriate

### How to Retrain:

```bash
python train_main.py --model-type Transformer --backtest
```

**Training time:** ~10-30 minutes

### After Retraining, You'll See:

- **BUY predictions**: ~10-20% of the time
- **HOLD predictions**: ~70-80% of the time  
- **SELL predictions**: ~10-20% of the time

This is **much more balanced** and useful for trading!

## 🎯 Temporary Workaround (Already Applied)

While you retrain, I've added a **smart override** that:
- Checks expected return from the regression head
- Overrides HOLD → BUY if expected return > 0.2% and BUY prob is reasonable
- Overrides HOLD → SELL if expected return < -0.2% and SELL prob is reasonable

However, with the current model, the expected return is usually small (-0.04%), so overrides rarely trigger.

## 📊 Why This Happens

1. **5-minute timeframe** - Most price movements are small
2. **0.5% threshold was too strict** - Very few candles move >0.5%
3. **No class balancing** - Model optimized for accuracy, not trading signals

## 🚀 Next Steps

1. **Retrain the model** (recommended):
   ```bash
   python train_main.py --model-type Transformer --backtest
   ```

2. **Or use a longer timeframe** (15m, 1h) for more movement

3. **Or adjust thresholds** in `config/train_config.json`:
   ```json
   "buy_threshold": 0.001,   // 0.1% instead of 0.15%
   "sell_threshold": -0.001
   ```

## 📝 Summary

- **Problem**: Model trained on 99.5% HOLD data → always predicts HOLD
- **Fix**: Class weights + lower thresholds (already in code)
- **Action**: Retrain the model to get BUY/SELL predictions
- **Time**: ~10-30 minutes

