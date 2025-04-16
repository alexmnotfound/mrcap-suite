from datetime import datetime, timedelta, timezone
import pandas as pd
from .db import db_cursor

def get_market_data(
    ticker: str,
    timeframe: str,
    start_time: datetime = None,
    end_time: datetime = None,
    include_indicators: bool = True
) -> pd.DataFrame:
    """
    Get market data with indicators from database.
    
    Args:
        ticker: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe in Binance format (e.g., '1h', '4h', '1d', '1M')
        start_time: Start datetime (default: 7 days ago)
        end_time: End datetime (default: now)
        include_indicators: Whether to include indicator data
    
    Returns:
        DataFrame with OHLC data and indicators
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time - timedelta(days=7)
        
    # Ensure timestamps are timezone-aware and in UTC
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
        
    # Validate timeframe format
    if timeframe not in ['1h', '4h', '1d', '1M']:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: 1h, 4h, 1d, 1M")
    
    with db_cursor() as cursor:
        # Get OHLC data with explicit UTC conversion
        cursor.execute("""
        SELECT 
            timestamp AT TIME ZONE 'UTC' as timestamp,
            open, high, low, close, volume, candle_pattern
        FROM ohlc_data
        WHERE ticker = %s 
        AND timeframe = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """, (ticker, timeframe, start_time, end_time))
        
        df = pd.DataFrame(cursor.fetchall(), 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_pattern'])
        
        if df.empty:
            return df
            
        # Ensure timestamp index is timezone-aware
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(timezone.utc)
        df.set_index('timestamp', inplace=True)
        
        if include_indicators:
            # Get EMAs with UTC conversion
            cursor.execute("""
            SELECT 
                timestamp AT TIME ZONE 'UTC' as timestamp,
                period, value
            FROM emas
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp, period
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, period, value in cursor.fetchall():
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                df.loc[timestamp, f'ema_{period}'] = value
            
            # Get RSI with UTC conversion
            cursor.execute("""
            SELECT 
                timestamp AT TIME ZONE 'UTC' as timestamp,
                value, slope, divergence
            FROM rsi
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, value, slope, div in cursor.fetchall():
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                df.loc[timestamp, 'rsi'] = value
                df.loc[timestamp, 'rsi_slope'] = slope
                df.loc[timestamp, 'rsi_div'] = div
            
            # Get Chandelier Exit with UTC conversion
            cursor.execute("""
            SELECT 
                timestamp AT TIME ZONE 'UTC' as timestamp,
                long_stop, short_stop, direction, signal
            FROM chandelier_exit
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, long_stop, short_stop, direction, signal in cursor.fetchall():
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                df.loc[timestamp, 'ce_long_stop'] = long_stop
                df.loc[timestamp, 'ce_short_stop'] = short_stop
                df.loc[timestamp, 'ce_direction'] = direction
                df.loc[timestamp, 'ce_signal'] = signal
            
            # Get OBV with UTC conversion
            cursor.execute("""
            SELECT 
                timestamp AT TIME ZONE 'UTC' as timestamp,
                value, ma_value, bb_upper, bb_lower
            FROM obv
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, value, ma, bb_upper, bb_lower in cursor.fetchall():
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                df.loc[timestamp, 'obv'] = value
                if ma is not None:
                    df.loc[timestamp, 'obv_ma'] = ma
                if bb_upper is not None:
                    df.loc[timestamp, 'obv_bb_upper'] = bb_upper
                if bb_lower is not None:
                    df.loc[timestamp, 'obv_bb_lower'] = bb_lower
    
    return df 