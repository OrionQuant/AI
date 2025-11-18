"""
LSTM-based signal prediction model for BTC trading.

Multi-task learning: classification (BUY/HOLD/SELL) + regression (expected return).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LSTMSignalNet(nn.Module):
    """
    LSTM network for predicting trading signals and expected returns.
    
    Architecture:
    - Multi-layer LSTM for sequence processing
    - Classification head for signal prediction (BUY/HOLD/SELL)
    - Regression head for expected return prediction
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 num_classes: int = 3,
                 bidirectional: bool = False):
        """
        Initialize LSTM signal network.
        
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of classification classes (default: 3 for BUY/HOLD/SELL)
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMSignalNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head (BUY/HOLD/SELL)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Regression head (expected return)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            pred_return: Predicted return of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_dim)
        last_output = self.dropout(last_output)
        
        # Classification and regression heads
        logits = self.classifier(last_output)
        pred_return = self.regressor(last_output)
        
        return logits, pred_return
    
    def predict_signal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict signal with probabilities and expected return.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            signal: Predicted class (0=BUY, 1=HOLD, 2=SELL)
            probs: Class probabilities
            expected_return: Predicted return
        """
        self.eval()
        with torch.no_grad():
            logits, pred_return = self.forward(x)
            probs = F.softmax(logits, dim=1)
            signal = torch.argmax(probs, dim=1)
        return signal, probs, pred_return


class CNNLSTMSignalNet(nn.Module):
    """
    CNN-LSTM hybrid model for signal prediction.
    
    Uses 1D convolutions to extract local patterns, then LSTM for temporal dependencies.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 num_classes: int = 3,
                 conv_filters: int = 64):
        """
        Initialize CNN-LSTM network.
        
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of classification classes
            conv_filters: Number of convolutional filters
        """
        super(CNNLSTMSignalNet, self).__init__()
        
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, conv_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.conv_dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            logits: Classification logits
            pred_return: Predicted return
        """
        # Reshape for convolution: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.conv_dropout(x)
        x = F.relu(self.conv2(x))
        x = self.conv_dropout(x)
        
        # Reshape back: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Heads
        logits = self.classifier(last_output)
        pred_return = self.regressor(last_output)
        
        return logits, pred_return


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = LSTMSignalNet(
        input_dim=50,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        num_classes=3
    ).to(device)
    
    # Test forward pass
    batch_size = 32
    seq_len = 128
    x = torch.randn(batch_size, seq_len, 50).to(device)
    
    logits, pred_return = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted return shape: {pred_return.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

