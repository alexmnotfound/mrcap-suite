import pandas as pd
import numpy as np
import logging

def RMA(series: pd.Series, period: int) -> pd.Series:
    """Calculate the RMA (Wilder's Moving Average) equivalent to PineScript."""
    alpha = 1 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def True_Range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range

def add_chandelier_exit(df: pd.DataFrame, period: int = 22, multiplier: float = 3.0, use_close: bool = True) -> pd.DataFrame:
    """Calculate Chandelier Exit indicator matching PineScript implementation closely."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Calculating Chandelier Exit (period={period}, multiplier={multiplier}, use_close={use_close})")

    try:
        df = df.copy()

        # Calculate ATR using RMA to match PineScript exactly
        df['tr'] = True_Range(df)
        df['atr'] = RMA(df['tr'], period)
        atr = multiplier * df['atr']

        # Calculate highest/lowest (matching PineScript)
        if use_close:
            highest = df['Close'].rolling(window=period).max()
            lowest = df['Close'].rolling(window=period).min()
        else:
            highest = df['High'].rolling(window=period).max()
            lowest = df['Low'].rolling(window=period).min()

        # Initial stops
        df['long_stop'] = highest - atr
        df['short_stop'] = lowest + atr

        # Initialize previous stops
        df['long_stop_prev'] = df['long_stop'].shift(1)
        df['short_stop_prev'] = df['short_stop'].shift(1)
        df['prev_close'] = df['Close'].shift(1)

        # PineScript's nz() initialization
        df['long_stop_prev'].fillna(df['long_stop'], inplace=True)
        df['short_stop_prev'].fillna(df['short_stop'], inplace=True)

        # Update stops based on previous close
        df['long_stop'] = np.where(
            df['prev_close'] > df['long_stop_prev'],
            np.maximum(df['long_stop'], df['long_stop_prev']),
            df['long_stop']
        )

        df['short_stop'] = np.where(
            df['prev_close'] < df['short_stop_prev'],
            np.minimum(df['short_stop'], df['short_stop_prev']),
            df['short_stop']
        )

        # Direction calculation
        df['ce_direction'] = np.where(
            df['Close'] > df['short_stop_prev'], 1,
            np.where(df['Close'] < df['long_stop_prev'], -1, np.nan)
        )

        # Forward-fill direction to match PineScript var behavior
        df['ce_direction'] = df['ce_direction'].ffill().fillna(1)

        # Signals calculation
        df['ce_signal'] = np.where(
            (df['ce_direction'] == 1) & (df['ce_direction'].shift(1) == -1), 1,
            np.where((df['ce_direction'] == -1) & (df['ce_direction'].shift(1) == 1), -1, 0)
        )

        # Final stops - store both values without np.where
        df['ce_long_stop'] = df['long_stop']
        df['ce_short_stop'] = df['short_stop']

        # Cleanup - remove these from drop list since we need them
        df.drop(['prev_close', 'long_stop_prev', 'short_stop_prev'], axis=1, inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error calculating Chandelier Exit: {str(e)}")
        raise