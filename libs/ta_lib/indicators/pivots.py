import pandas as pd

def add_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pivot points based on previous period's High, Low, and Close.
    
    Args:
        df: DataFrame with columns ['High', 'Low', 'Close']
    
    Returns:
        DataFrame with additional pivot point columns:
        PP: Pivot Point
        R1-R5: Resistance levels
        S1-S5: Support levels
    """
    df = df.copy()
    
    # Calculate Pivot Point
    df['PP'] = (df.High.shift(1) + df.Low.shift(1) + df.Close.shift(1)) / 3
    
    # Calculate Resistance Levels
    df['R1'] = (2 * df.PP) - df.Low.shift(1)
    df['R2'] = df.PP + (df.High.shift(1) - df.Low.shift(1))
    df['R3'] = df.High.shift(1) + (2 * (df.PP - df.Low.shift(1)))
    df['R4'] = df.PP * 3 + (df.High.shift(1) - 3 * df.Low.shift(1))
    df['R5'] = df.PP * 4 + (df.High.shift(1) - 4 * df.Low.shift(1))
    
    # Calculate Support Levels
    df['S1'] = (2 * df.PP) - df.High.shift(1)
    df['S2'] = df.PP - (df.High.shift(1) - df.Low.shift(1))
    df['S3'] = df.Low.shift(1) - (2 * (df.High.shift(1) - df.PP))
    df['S4'] = df.PP * 3 - (3 * df.High.shift(1) - df.Low.shift(1))
    df['S5'] = df.PP * 4 - (4 * df.High.shift(1) - df.Low.shift(1))
    
    return df 