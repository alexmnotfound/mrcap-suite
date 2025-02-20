import pandas as pd
import numpy as np

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify candlestick patterns in OHLC data.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added 'candle_pattern' column containing pattern names
    """
    df = df.copy()
    
    # Calculate basic candlestick properties
    df['body'] = df['Close'] - df['Open']
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_size'] = abs(df['body'])
    df['total_size'] = df['High'] - df['Low']
    
    # Initialize pattern column
    df['candle_pattern'] = 'Normal'
    
    # Doji (very small body)
    doji_mask = df['body_size'] <= 0.1 * df['total_size']
    df.loc[doji_mask, 'candle_pattern'] = 'Doji'
    
    # Hammer and Hanging Man (small body, long lower shadow, small upper shadow)
    hammer_mask = (
        (df['lower_shadow'] > 2 * df['body_size']) & 
        (df['upper_shadow'] <= 0.1 * df['total_size']) &
        (df['body_size'] <= 0.3 * df['total_size'])
    )
    df.loc[hammer_mask & (df['body'] > 0), 'candle_pattern'] = 'Hammer'
    df.loc[hammer_mask & (df['body'] < 0), 'candle_pattern'] = 'Hanging Man'
    
    # Shooting Star and Inverted Hammer (small body, long upper shadow, small lower shadow)
    star_mask = (
        (df['upper_shadow'] > 2 * df['body_size']) & 
        (df['lower_shadow'] <= 0.1 * df['total_size']) &
        (df['body_size'] <= 0.3 * df['total_size'])
    )
    df.loc[star_mask & (df['body'] < 0), 'candle_pattern'] = 'Shooting Star'
    df.loc[star_mask & (df['body'] > 0), 'candle_pattern'] = 'Inverted Hammer'
    
    # Marubozu (long body, very small shadows)
    marubozu_mask = (
        (df['body_size'] >= 0.9 * df['total_size']) &
        (df['upper_shadow'] <= 0.1 * df['total_size']) &
        (df['lower_shadow'] <= 0.1 * df['total_size'])
    )
    df.loc[marubozu_mask & (df['body'] > 0), 'candle_pattern'] = 'Bullish Marubozu'
    df.loc[marubozu_mask & (df['body'] < 0), 'candle_pattern'] = 'Bearish Marubozu'
    
    # Cleanup intermediate columns
    df.drop(['body', 'upper_shadow', 'lower_shadow', 'body_size', 'total_size'], axis=1, inplace=True)
    
    return df 