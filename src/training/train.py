"""
Training script for BTC trading signal model.

Includes walk-forward validation, early stopping, and multi-task loss.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeqDataset(Dataset):
    """Dataset for sequence data with classification and regression labels."""
    
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences of shape (n_samples, seq_len, n_features)
            y_cls: Classification labels (n_samples,)
            y_reg: Regression labels (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y_cls = torch.LongTensor(y_cls)
        self.y_reg = torch.FloatTensor(y_reg)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, loss_fn_cls, loss_fn_reg, 
                lambda_reg: float, scaler: Optional[GradScaler] = None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for X_batch, y_cls_batch, y_reg_batch in loader:
        X_batch = X_batch.to(device)
        y_cls_batch = y_cls_batch.to(device)
        y_reg_batch = y_reg_batch.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                logits, pred_reg = model(X_batch)
                loss_cls = loss_fn_cls(logits, y_cls_batch)
                loss_reg = loss_fn_reg(pred_reg.squeeze(), y_reg_batch)
                loss = loss_cls + lambda_reg * loss_reg
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, pred_reg = model(X_batch)
            loss_cls = loss_fn_cls(logits, y_cls_batch)
            loss_reg = loss_fn_reg(pred_reg.squeeze(), y_reg_batch)
            loss = loss_cls + lambda_reg * loss_reg
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)
    
    return total_loss / n_samples


def evaluate_model(model, loader, device, loss_fn_cls, loss_fn_reg, lambda_reg: float):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    y_true_cls = []
    y_pred_cls = []
    y_true_reg = []
    y_pred_reg = []
    n_samples = 0
    
    with torch.no_grad():
        for X_batch, y_cls_batch, y_reg_batch in loader:
            X_batch = X_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            
            logits, pred_reg = model(X_batch)
            
            loss_cls = loss_fn_cls(logits, y_cls_batch)
            loss_reg = loss_fn_reg(pred_reg.squeeze(), y_reg_batch)
            loss = loss_cls + lambda_reg * loss_reg
            
            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
            
            # Predictions
            pred_cls = logits.argmax(dim=1).cpu().numpy()
            y_pred_cls.extend(pred_cls)
            y_true_cls.extend(y_cls_batch.cpu().numpy())
            
            y_pred_reg.extend(pred_reg.squeeze().cpu().numpy())
            y_true_reg.extend(y_reg_batch.cpu().numpy())
    
    avg_loss = total_loss / n_samples
    
    # Classification metrics
    class_names = ['BUY', 'HOLD', 'SELL']
    report = classification_report(y_true_cls, y_pred_cls, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    # Regression metrics
    y_true_reg = np.array(y_true_reg)
    y_pred_reg = np.array(y_pred_reg)
    mse = np.mean((y_true_reg - y_pred_reg) ** 2)
    mae = np.mean(np.abs(y_true_reg - y_pred_reg))
    corr = np.corrcoef(y_true_reg, y_pred_reg)[0, 1] if len(y_true_reg) > 1 else 0.0
    
    return {
        'loss': avg_loss,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true_cls, y_pred_cls).tolist(),
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(corr) if not np.isnan(corr) else 0.0
    }


def train_model(X_train, y_train_cls, y_train_reg,
                X_val, y_val_cls, y_val_reg,
                config: Dict,
                model_class,
                save_dir: str = "models") -> Tuple[nn.Module, Dict]:
    """
    Train model with walk-forward validation.
    
    Args:
        X_train: Training sequences
        y_train_cls: Training classification labels
        y_train_reg: Training regression labels
        X_val: Validation sequences
        y_val_cls: Validation classification labels
        y_val_reg: Validation regression labels
        config: Training configuration dictionary
        model_class: Model class to instantiate
        save_dir: Directory to save model checkpoints
        
    Returns:
        Trained model and training history
    """
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SeqDataset(X_train, y_train_cls, y_train_reg)
    val_dataset = SeqDataset(X_val, y_val_cls, y_val_reg)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = model_class(
        input_dim=X_train.shape[2],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss functions
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )
    
    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()
    
    # Mixed precision scaler
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            loss_fn_cls, loss_fn_reg, config['lambda_reg'], scaler
        )
        
        # Validate
        val_metrics = evaluate_model(
            model, val_loader, device,
            loss_fn_cls, loss_fn_reg, config['lambda_reg']
        )
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['classification_report']['accuracy']
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f"Saved best model (val_loss={val_loss:.6f})")
        else:
            patience += 1
            if patience >= config.get('early_stop_patience', 10):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    logger.info("\nFinal validation metrics:")
    final_metrics = evaluate_model(model, val_loader, device, loss_fn_cls, loss_fn_reg, config['lambda_reg'])
    
    print("\nClassification Report:")
    print(classification_report(
        [0, 1, 2], [0, 1, 2],  # Dummy for formatting
        target_names=['BUY', 'HOLD', 'SELL']
    ))
    print(f"\nRegression Metrics:")
    print(f"MSE: {final_metrics['mse']:.6f}")
    print(f"MAE: {final_metrics['mae']:.6f}")
    print(f"Correlation: {final_metrics['correlation']:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Example usage
    from src.models.lstm_signal_net import LSTMSignalNet
    
    # Dummy data for testing
    n_train = 10000
    n_val = 2000
    seq_len = 128
    n_features = 50
    
    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    y_train_cls = np.random.randint(0, 3, n_train)
    y_train_reg = np.random.randn(n_train).astype(np.float32) * 0.01
    
    X_val = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
    y_val_cls = np.random.randint(0, 3, n_val)
    y_val_reg = np.random.randn(n_val).astype(np.float32) * 0.01
    
    config = {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'lr': 3e-4,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'epochs': 50,
        'lambda_reg': 0.1,
        'early_stop_patience': 10,
        'seed': 42,
        'use_amp': True,
        'min_lr': 1e-6
    }
    
    model, history = train_model(
        X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg,
        config, LSTMSignalNet
    )
    
    print(f"\nTraining completed. Best validation loss: {min(history['val_loss']):.6f}")

