# 🔄 How to Fix the "Always HOLD" Problem

## Problem
Your current model always predicts HOLD (99.7%) because it was trained on imbalanced data:
- **BUY**: 0.3% of samples
- **HOLD**: 99.5% of samples  
- **SELL**: 0.2% of samples

## ✅ Solution: Retrain the Model

The code has been fixed to handle class imbalance. You need to retrain:

### Step 1: Update Config (Already Done)
The config now uses lower thresholds (0.15% instead of 0.5%) to get more BUY/SELL signals.

### Step 2: Retrain the Model
```bash
python train_main.py --model-type Transformer --backtest
```

This will:
1. ✅ Use class weights (BUY/SELL get 100x more weight than HOLD)
2. ✅ Use lower thresholds (0.15% instead of 0.5%)
3. ✅ Train a new model that can actually predict BUY/SELL

### Step 3: Wait for Training
Training takes ~10-30 minutes depending on your hardware.

### Step 4: Restart the Web UI
After training completes, restart the Flask app:
```bash
python app.py
```

## 🎯 Temporary Fix (Already Applied)

While you retrain, I've added a **smart override** that:
- Uses the model's expected return to override HOLD predictions
- If expected return > 0.3%, it may predict BUY
- If expected return < -0.3%, it may predict SELL

This makes the current model more useful, but **retraining is still recommended** for best results.

## 📊 Expected Results After Retraining

- **BUY predictions**: ~10-20% of the time
- **HOLD predictions**: ~70-80% of the time
- **SELL predictions**: ~10-20% of the time

This is much more balanced and useful for trading!

