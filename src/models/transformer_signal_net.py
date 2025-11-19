"""
Transformer-based signal prediction model for BTC trading.

Enhanced architecture with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Residual connections
- Multi-task learning (classification + regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerSignalNet(nn.Module):
    """
    Transformer-based network for trading signal prediction.
    
    Architecture:
    - Input projection
    - Positional encoding
    - Stack of transformer blocks
    - Classification and regression heads
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 num_classes: int = 3,
                 max_seq_len: int = 512):
        """
        Initialize Transformer signal network.
        
        Args:
            input_dim: Number of input features per timestep
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            num_classes: Number of classification classes
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling (use last timestep + mean pooling)
        self.pooling = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
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
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global pooling: concatenate last timestep and mean
        last_timestep = x[:, -1, :]  # (batch_size, d_model)
        mean_pool = x.mean(dim=1)  # (batch_size, d_model)
        pooled = torch.cat([last_timestep, mean_pool], dim=1)  # (batch_size, 2*d_model)
        pooled = self.pooling(pooled)  # (batch_size, d_model)
        
        # Classification and regression heads
        logits = self.classifier(pooled)
        pred_return = self.regressor(pooled)
        
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


class AttentionLSTMNet(nn.Module):
    """
    Hybrid LSTM with attention mechanism.
    
    Combines LSTM for temporal modeling with attention for feature importance.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 num_classes: int = 3,
                 bidirectional: bool = True,
                 attention_dim: int = 128):
        """
        Initialize Attention-LSTM network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of classes
            bidirectional: Whether to use bidirectional LSTM
            attention_dim: Attention dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_output_dim)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        attended = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, lstm_output_dim)
        attended = self.dropout(attended)
        
        # Classification and regression
        logits = self.classifier(attended)
        pred_return = self.regressor(attended)
        
        return logits, pred_return


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 32
    seq_len = 128
    input_dim = 50
    
    # Test Transformer
    print("Testing TransformerSignalNet...")
    transformer_model = TransformerSignalNet(
        input_dim=input_dim,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    logits, pred_return = transformer_model(x)
    print(f"Transformer - Input: {x.shape}, Logits: {logits.shape}, Return: {pred_return.shape}")
    print(f"Transformer parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # Test Attention-LSTM
    print("\nTesting AttentionLSTMNet...")
    attn_lstm_model = AttentionLSTMNet(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    logits, pred_return = attn_lstm_model(x)
    print(f"Attention-LSTM - Input: {x.shape}, Logits: {logits.shape}, Return: {pred_return.shape}")
    print(f"Attention-LSTM parameters: {sum(p.numel() for p in attn_lstm_model.parameters()):,}")

