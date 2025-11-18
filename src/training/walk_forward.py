"""
Walk-forward validation for time series models.

Implements rolling window cross-validation to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Splits data into contiguous windows: train -> validate -> test,
    then slides forward in time.
    """
    
    def __init__(self,
                 train_window: int,
                 val_window: int,
                 test_window: int,
                 step_size: int = None):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window: Number of samples in training window
            val_window: Number of samples in validation window
            test_window: Number of samples in test window
            step_size: Step size for sliding (default: test_window)
        """
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size or test_window
    
    def split(self, data: np.ndarray, timestamps: np.ndarray = None) -> List[Dict]:
        """
        Generate walk-forward splits.
        
        Args:
            data: Full dataset (can be X, y, or DataFrame)
            timestamps: Optional timestamps for each sample
            
        Returns:
            List of dictionaries with 'train', 'val', 'test' indices
        """
        n_samples = len(data)
        splits = []
        
        start = 0
        while start + self.train_window + self.val_window + self.test_window <= n_samples:
            train_end = start + self.train_window
            val_end = train_end + self.val_window
            test_end = val_end + self.test_window
            
            split = {
                'train': (start, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end),
                'train_indices': np.arange(start, train_end),
                'val_indices': np.arange(train_end, val_end),
                'test_indices': np.arange(val_end, test_end)
            }
            
            if timestamps is not None:
                split['train_start'] = timestamps[start]
                split['train_end'] = timestamps[train_end - 1]
                split['val_start'] = timestamps[train_end]
                split['val_end'] = timestamps[val_end - 1]
                split['test_start'] = timestamps[val_end]
                split['test_end'] = timestamps[test_end - 1]
            
            splits.append(split)
            
            # Slide forward
            start += self.step_size
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def split_dataframe(self, df: pd.DataFrame, 
                       date_col: str = 'timestamp') -> List[Dict]:
        """
        Generate walk-forward splits from DataFrame using dates.
        
        Args:
            df: DataFrame with timestamp column
            date_col: Name of timestamp column
            
        Returns:
            List of split dictionaries
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        timestamps = df[date_col].values
        
        splits = self.split(df, timestamps)
        
        # Convert to DataFrame slices
        for split in splits:
            split['train_df'] = df.iloc[split['train_indices']]
            split['val_df'] = df.iloc[split['val_indices']]
            split['test_df'] = df.iloc[split['test_indices']]
        
        return splits


def create_time_based_splits(df: pd.DataFrame,
                             train_start: str,
                             train_end: str,
                             val_start: str,
                             val_end: str,
                             test_start: str,
                             test_end: str,
                             date_col: str = 'timestamp') -> Dict:
    """
    Create simple time-based train/val/test splits.
    
    Args:
        df: DataFrame with data
        train_start: Start date for training (YYYY-MM-DD)
        train_end: End date for training
        val_start: Start date for validation
        val_end: End date for validation
        test_start: Start date for testing
        test_end: End date for testing
        date_col: Name of timestamp column
        
    Returns:
        Dictionary with train_df, val_df, test_df
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_df = df[(df[date_col] >= train_start) & (df[date_col] < train_end)]
    val_df = df[(df[date_col] >= val_start) & (df[date_col] < val_end)]
    test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]
    
    logger.info(f"Train: {len(train_df)} samples ({train_start} to {train_end})")
    logger.info(f"Val: {len(val_df)} samples ({val_start} to {val_end})")
    logger.info(f"Test: {len(test_df)} samples ({test_start} to {test_end})")
    
    return {
        'train_df': train_df.reset_index(drop=True),
        'val_df': val_df.reset_index(drop=True),
        'test_df': test_df.reset_index(drop=True)
    }


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2022-01-01', periods=10000, freq='5min')
    data = np.random.randn(len(dates), 50)
    
    df = pd.DataFrame(data)
    df['timestamp'] = dates
    
    # Walk-forward validation
    validator = WalkForwardValidator(
        train_window=5000,
        val_window=1000,
        test_window=1000,
        step_size=1000
    )
    
    splits = validator.split_dataframe(df)
    
    print(f"\nGenerated {len(splits)} splits")
    print(f"\nFirst split:")
    print(f"  Train: {splits[0]['train_start']} to {splits[0]['train_end']}")
    print(f"  Val: {splits[0]['val_start']} to {splits[0]['val_end']}")
    print(f"  Test: {splits[0]['test_start']} to {splits[0]['test_end']}")

