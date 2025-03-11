import pandas as pd
import logging

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
    logger = logging.getLogger(__name__)
    logger.debug("Calculating pivot points...")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    
    df = df.copy()
    
    try:
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
        
        logger.debug("Pivot points calculated successfully")
        logger.debug(f"Pivot columns: {[col for col in df.columns if col in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']]}")
        
        # Log a sample of the calculated values
        if not df.empty:
            sample_ts = df.index[-1]  # Get last timestamp
            logger.debug(f"Sample pivot values for {sample_ts}:")
            for col in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']:
                if col in df.columns:
                    logger.debug(f"  {col}: {df.loc[sample_ts, col]:.2f}")
        
    except Exception as e:
        logger.error(f"Error calculating pivot points: {str(e)}")
        raise
        
    return df 