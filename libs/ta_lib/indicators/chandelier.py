import pandas as pd
import numpy as np
import logging

def add_chandelier_exit(df: pd.DataFrame, period: int = 22, multiplier: float = 3.0, use_close: bool = True) -> pd.DataFrame:
    """Calculate Chandelier Exit indicator to match PineScript implementation."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Calculating Chandelier Exit (period={period}, multiplier={multiplier}, use_close={use_close})")
    
    try:
        df = df.copy()
        
        # Calculate ATR (matches ta.atr(length))
        df['tr'] = True_Range(df)
        df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
        atr = multiplier * df['atr']
        
        # Calculate highest/lowest (matches ta.highest/ta.lowest)
        if use_close:
            highest = df['Close'].rolling(window=period).max()
            lowest = df['Close'].rolling(window=period).min()
        else:
            highest = df['High'].rolling(window=period).max()
            lowest = df['Low'].rolling(window=period).min()
        
        # Initialize base stops
        df['ce_long_stop'] = highest - atr
        df['ce_short_stop'] = lowest + atr
        
        # Get previous values with proper initialization
        df['prev_close'] = df['Close'].shift(1)
        df['prev_long_stop'] = df['ce_long_stop'].shift(1).fillna(df['ce_long_stop'])
        df['prev_short_stop'] = df['ce_short_stop'].shift(1).fillna(df['ce_short_stop'])
        
        # Create masks for trend conditions
        uptrend_mask = df['prev_close'] > df['prev_long_stop']
        downtrend_mask = df['prev_close'] < df['prev_short_stop']
        
        # Update stops using cumulative max/min to maintain trailing nature
        df.loc[uptrend_mask, 'ce_long_stop'] = df.loc[uptrend_mask, ['ce_long_stop', 'prev_long_stop']].max(axis=1)
        df.loc[downtrend_mask, 'ce_short_stop'] = df.loc[downtrend_mask, ['ce_short_stop', 'prev_short_stop']].min(axis=1)
        
        # Calculate direction
        df['ce_direction'] = np.where(
            df['Close'] > df['prev_short_stop'], 1,
            np.where(df['Close'] < df['prev_long_stop'], -1, np.nan)
        )
        df['ce_direction'] = df['ce_direction'].ffill().fillna(1)
        
        # Calculate signals
        df['ce_signal'] = np.where(
            (df['ce_direction'] == 1) & (df['ce_direction'].shift(1) == -1), 1,
            np.where((df['ce_direction'] == -1) & (df['ce_direction'].shift(1) == 1), -1, 0)
        )
        
        # Clean up intermediate columns
        df.drop(['tr', 'atr', 'prev_close', 'prev_long_stop', 'prev_short_stop'], axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating Chandelier Exit: {str(e)}")
        raise

def True_Range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    # Fixed: Use `.max(axis=1)` for row-wise max
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range
