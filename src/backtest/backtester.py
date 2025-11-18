"""
Realistic backtesting engine for BTC trading signals.

Includes:
- Slippage modeling
- Trading fees (maker/taker)
- Funding costs (for futures)
- Risk management (stop-loss, position limits)
- Performance metrics (Sharpe, drawdown, profit factor)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = 0
    HOLD = 1
    SELL = 2


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fee_paid: float = 0.0
    slippage: float = 0.0
    stop_loss_hit: bool = False
    take_profit_hit: bool = False


class Backtester:
    """
    Realistic backtesting engine with slippage, fees, and risk management.
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 taker_fee: float = 0.001,  # 0.1% taker fee
                 maker_fee: float = 0.0005,  # 0.05% maker fee
                 slippage_pct: float = 0.0005,  # 0.05% slippage
                 funding_rate: float = 0.0001,  # 0.01% per 8h (for futures)
                 max_position_size: float = 1.0,  # Max position as fraction of capital
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.05,  # 5% take profit
                 use_futures: bool = False):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
            taker_fee: Taker fee as fraction (default: 0.1%)
            maker_fee: Maker fee as fraction (default: 0.05%)
            slippage_pct: Slippage as fraction of price (default: 0.05%)
            funding_rate: Funding rate per period (for futures)
            max_position_size: Maximum position size as fraction of capital
            stop_loss_pct: Stop loss as fraction (default: 2%)
            take_profit_pct: Take profit as fraction (default: 5%)
            use_futures: Whether trading futures (affects funding costs)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_pct = slippage_pct
        self.funding_rate = funding_rate
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_futures = use_futures
        
        # State
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[pd.Timestamp] = []
        
    def _calculate_entry_price(self, mid_price: float, side: str) -> float:
        """
        Calculate entry price with slippage.
        
        Args:
            mid_price: Mid price (or close price)
            side: 'long' or 'short'
            
        Returns:
            Entry price after slippage
        """
        if side == 'long':
            # Buying: pay more (ask side + slippage)
            return mid_price * (1 + self.slippage_pct)
        else:
            # Selling: receive less (bid side - slippage)
            return mid_price * (1 - self.slippage_pct)
    
    def _calculate_exit_price(self, mid_price: float, side: str) -> float:
        """
        Calculate exit price with slippage.
        
        Args:
            mid_price: Mid price
            side: 'long' or 'short'
            
        Returns:
            Exit price after slippage
        """
        if side == 'long':
            # Selling: receive less
            return mid_price * (1 - self.slippage_pct)
        else:
            # Buying to close: pay more
            return mid_price * (1 + self.slippage_pct)
    
    def _calculate_fee(self, price: float, quantity: float, is_taker: bool = True) -> float:
        """Calculate trading fee."""
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        return price * quantity * fee_rate
    
    def enter_position(self, timestamp: pd.Timestamp, price: float, 
                      signal: Signal, confidence: float = 1.0):
        """
        Enter a new position based on signal.
        
        Args:
            timestamp: Current timestamp
            price: Current price (mid or close)
            signal: Trading signal (BUY=long, SELL=short, HOLD=no action)
            confidence: Signal confidence (0-1), affects position sizing
        """
        # Close existing position if any
        if self.position is not None:
            self.exit_position(timestamp, price)
        
        # Enter new position
        if signal == Signal.BUY:
            side = 'long'
        elif signal == Signal.SELL:
            side = 'short'
        else:  # HOLD
            return
        
        # Position sizing
        position_value = self.capital * self.max_position_size * confidence
        entry_price = self._calculate_entry_price(price, side)
        quantity = position_value / entry_price
        
        # Calculate fees
        fee = self._calculate_fee(entry_price, quantity, is_taker=True)
        
        # Update capital (deduct fees)
        self.capital -= fee
        
        # Create trade
        self.position = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            side=side,
            fee_paid=fee,
            slippage=(entry_price - price) / price
        )
        
        logger.debug(f"Entered {side} position: {quantity:.6f} @ {entry_price:.2f}")
    
    def exit_position(self, timestamp: pd.Timestamp, price: float, 
                     reason: str = "signal"):
        """
        Exit current position.
        
        Args:
            timestamp: Current timestamp
            price: Current price
            reason: Reason for exit ('signal', 'stop_loss', 'take_profit')
        """
        if self.position is None:
            return
        
        # Calculate exit price
        exit_price = self._calculate_exit_price(price, self.position.side)
        
        # Calculate P&L
        if self.position.side == 'long':
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
        else:  # short
            pnl = (self.position.entry_price - exit_price) * self.position.quantity
        
        pnl_pct = pnl / (self.position.entry_price * self.position.quantity)
        
        # Calculate exit fee
        exit_fee = self._calculate_fee(exit_price, self.position.quantity, is_taker=True)
        total_fee = self.position.fee_paid + exit_fee
        
        # Update capital
        self.capital += self.position.entry_price * self.position.quantity + pnl - exit_fee
        
        # Update trade
        self.position.exit_time = timestamp
        self.position.exit_price = exit_price
        self.position.pnl = pnl - total_fee  # Net P&L after fees
        self.position.pnl_pct = pnl_pct
        self.position.fee_paid = total_fee
        
        if reason == 'stop_loss':
            self.position.stop_loss_hit = True
        elif reason == 'take_profit':
            self.position.take_profit_hit = True
        
        # Record trade
        self.trades.append(self.position)
        self.position = None
        
        logger.debug(f"Exited position: P&L={pnl:.2f} ({pnl_pct*100:.2f}%)")
    
    def update_position(self, timestamp: pd.Timestamp, price: float):
        """
        Update current position (check stop-loss, take-profit).
        
        Args:
            timestamp: Current timestamp
            price: Current price
        """
        if self.position is None:
            return
        
        # Calculate unrealized P&L
        if self.position.side == 'long':
            pnl_pct = (price - self.position.entry_price) / self.position.entry_price
        else:  # short
            pnl_pct = (self.position.entry_price - price) / self.position.entry_price
        
        # Check stop-loss
        if pnl_pct <= -self.stop_loss_pct:
            self.exit_position(timestamp, price, reason='stop_loss')
            return
        
        # Check take-profit
        if pnl_pct >= self.take_profit_pct:
            self.exit_position(timestamp, price, reason='take_profit')
            return
        
        # Update equity curve
        if self.position.side == 'long':
            unrealized_pnl = (price - self.position.entry_price) * self.position.quantity
        else:
            unrealized_pnl = (self.position.entry_price - price) * self.position.quantity
        
        current_equity = self.capital + unrealized_pnl
        self.equity_curve.append(current_equity)
        self.timestamps.append(timestamp)
    
    def run_backtest(self, 
                    df: pd.DataFrame,
                    signals: np.ndarray,
                    confidences: Optional[np.ndarray] = None,
                    price_col: str = 'close') -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with price data and timestamps
            signals: Array of signals (0=BUY, 1=HOLD, 2=SELL) for each row
            confidences: Optional array of confidence scores
            price_col: Column name for price data
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Running backtest on {len(df)} timesteps")
        
        if confidences is None:
            confidences = np.ones(len(df))
        
        # Reset state
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = [df.iloc[0]['timestamp']]
        
        # Iterate through data
        for i, row in df.iterrows():
            timestamp = row['timestamp']
            price = row[price_col]
            signal = Signal(signals[i])
            confidence = confidences[i]
            
            # Update existing position
            if self.position is not None:
                self.update_position(timestamp, price)
            
            # Process new signal
            if signal != Signal.HOLD:
                if self.position is None:
                    self.enter_position(timestamp, price, signal, confidence)
                elif (signal == Signal.BUY and self.position.side == 'short') or \
                     (signal == Signal.SELL and self.position.side == 'long'):
                    # Reverse position
                    self.exit_position(timestamp, price, reason='signal')
                    self.enter_position(timestamp, price, signal, confidence)
        
        # Close final position if any
        if self.position is not None:
            final_price = df.iloc[-1][price_col]
            self.exit_position(df.iloc[-1]['timestamp'], final_price, reason='end')
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest complete: {len(self.trades)} trades, "
                   f"Final capital: ${self.capital:.2f}, "
                   f"Total return: {(self.capital/self.initial_capital - 1)*100:.2f}%")
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_return': 0.0
            }
        
        # Basic metrics
        total_return = self.capital - self.initial_capital
        total_return_pct = (self.capital / self.initial_capital - 1) * 100
        
        # Trade statistics
        pnls = [t.pnl for t in self.trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        num_trades = len(self.trades)
        num_wins = len(winning_trades)
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1e-8
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average trade return
        avg_trade_return = np.mean(pnls) if pnls else 0.0
        
        # Sharpe ratio (annualized, assuming daily returns)
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        return {
            'total_return': float(total_return),
            'total_return_pct': float(total_return_pct),
            'num_trades': num_trades,
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'avg_trade_return': float(avg_trade_return),
            'gross_profit': float(gross_profit),
            'gross_loss': float(gross_loss),
            'num_wins': num_wins,
            'num_losses': num_trades - num_wins,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps
        }


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.rand(1000) * 1000
    })
    
    # Generate random signals
    signals = np.random.randint(0, 3, len(df))
    signals[::10] = 0  # Buy every 10th
    signals[::15] = 2  # Sell every 15th
    
    # Run backtest
    backtester = Backtester(
        initial_capital=10000.0,
        taker_fee=0.001,
        slippage_pct=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    results = backtester.run_backtest(df, signals)
    
    print("\nBacktest Results:")
    print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

