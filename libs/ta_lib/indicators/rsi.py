import pandas as pd
import numpy as np

def add_rsi(df: pd.DataFrame, period: int = 14, slope_period: int = 1) -> pd.DataFrame:
    """
    Calculate RSI and optional slope/divergence indicators.
    
    Args:
        df: DataFrame with 'Close' column
        period: Number of periods for RSI calculation
        slope_period: Number of periods for RSI slope calculation (0 to disable)
    
    Returns:
        DataFrame with additional columns:
        rsi: Relative Strength Index
        rsi_slope: Rate of change in RSI (if slope_period > 0)
        rsi_div: Divergence between price and RSI slopes (if slope_period > 0)
    """
    df = df.copy()
    
    # Calculate price differences
    df['dif'] = df.Close.diff()
    
    # Separate gains and losses
    df['win'] = np.where(df['dif'] > 0, df['dif'], 0)
    df['loss'] = np.where(df['dif'] < 0, abs(df['dif']), 0)
    
    # Calculate exponential moving averages
    df['ema_win'] = df.win.ewm(alpha=1/period).mean()
    df['ema_loss'] = df.loss.ewm(alpha=1/period).mean()
    
    # Calculate RS and RSI
    df['rs'] = df.ema_win / df.ema_loss
    # Handle case where ema_loss is 0 (all gains, no losses)
    df['rsi'] = np.where(df['ema_loss'] == 0, 100, 100 - (100 / (1 + df.rs)))
    
    if slope_period > 0:
        # Calculate RSI slope and divergence
        df['rsi_slope'] = (df.rsi / df.rsi.shift(slope_period) - 1) * 100
        price_slope = (df.Close / df.Close.shift(slope_period) - 1) * 100
        df['rsi_div'] = df.rsi_slope * price_slope
    
    # Clean up intermediate columns
    df.drop(['dif', 'win', 'loss', 'ema_win', 'ema_loss', 'rs'], axis=1, inplace=True)
    
    return df 