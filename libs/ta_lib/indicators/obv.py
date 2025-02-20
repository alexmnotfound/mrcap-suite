import pandas as pd
import numpy as np

def add_obv(df: pd.DataFrame, ma_type: str = None, ma_length: int = 14, bb_mult: float = 2.0) -> pd.DataFrame:
    """
    Calculate On Balance Volume (OBV) with optional moving averages and Bollinger Bands.
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        ma_type: Type of Moving Average ('SMA', 'EMA', 'SMMA', 'WMA', 'VWMA', 'SMA + BB', None)
        ma_length: Length for moving average calculations
        bb_mult: Bollinger Bands multiplier
        
    Returns:
        DataFrame with added columns:
        - obv: On Balance Volume
        - obv_ma: Moving average of OBV (if ma_type specified)
        - obv_bb_upper: Upper Bollinger Band (if ma_type = 'SMA + BB')
        - obv_bb_lower: Lower Bollinger Band (if ma_type = 'SMA + BB')
    """
    df = df.copy()
    
    # Calculate price change direction
    price_change = df['Close'].diff()
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    
    # Calculate OBV
    df['obv'] = (direction * df['Volume']).cumsum()
    
    if ma_type:
        # Calculate moving average
        if ma_type in ['SMA', 'SMA + BB']:
            df['obv_ma'] = df['obv'].rolling(window=ma_length).mean()
        elif ma_type == 'EMA':
            df['obv_ma'] = df['obv'].ewm(span=ma_length, adjust=False).mean()
        elif ma_type == 'SMMA':  # Also known as RMA
            df['obv_ma'] = df['obv'].ewm(alpha=1/ma_length, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, ma_length + 1)
            df['obv_ma'] = df['obv'].rolling(window=ma_length).apply(
                lambda x: np.sum(weights * x) / weights.sum()
            )
        elif ma_type == 'VWMA':
            df['obv_ma'] = (df['obv'] * df['Volume']).rolling(window=ma_length).sum() / \
                          df['Volume'].rolling(window=ma_length).sum()
        
        # Calculate Bollinger Bands if requested
        if ma_type == 'SMA + BB':
            std = df['obv'].rolling(window=ma_length).std()
            df['obv_bb_upper'] = df['obv_ma'] + (std * bb_mult)
            df['obv_bb_lower'] = df['obv_ma'] - (std * bb_mult)
    
    return df 