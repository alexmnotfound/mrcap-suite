from datetime import datetime, timedelta
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
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(days=7)
        
    # Validate timeframe format
    if timeframe not in ['1h', '4h', '1d', '1M']:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: 1h, 4h, 1d, 1M")
    
    with db_cursor() as cursor:
        # Get OHLC data
        cursor.execute("""
        SELECT timestamp, open, high, low, close, volume, candle_pattern
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
            
        df.set_index('timestamp', inplace=True)
        
        if include_indicators:
            # Get EMAs
            cursor.execute("""
            SELECT timestamp, period, value
            FROM emas
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp, period
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, period, value in cursor.fetchall():
                df.loc[timestamp, f'ema_{period}'] = value
            
            # Get RSI
            cursor.execute("""
            SELECT timestamp, value, slope, divergence
            FROM rsi
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, value, slope, div in cursor.fetchall():
                df.loc[timestamp, 'rsi'] = value
                df.loc[timestamp, 'rsi_slope'] = slope
                df.loc[timestamp, 'rsi_div'] = div
            
            # Get monthly pivots - apply previous month's pivots to current month's candles
            cursor.execute("""
            WITH monthly_data AS (
                SELECT timestamp, level, value,
                       DATE_TRUNC('month', timestamp) as pivot_month
                FROM pivots 
                WHERE ticker = %s 
                AND timeframe = '1M'
                AND DATE_TRUNC('month', timestamp) 
                    BETWEEN DATE_TRUNC('month', %s::timestamp - INTERVAL '1 month') 
                    AND DATE_TRUNC('month', %s::timestamp - INTERVAL '1 month')
            )
            SELECT d.timestamp, md.level, md.value
            FROM generate_series(%s::timestamp, %s::timestamp, '1 hour') as d(timestamp)
            LEFT JOIN monthly_data md 
            ON DATE_TRUNC('month', d.timestamp) = DATE_TRUNC('month', md.pivot_month + INTERVAL '1 month')
            ORDER BY d.timestamp, md.level;
            """, (ticker, start_time, end_time, start_time, end_time))
            
            for timestamp, level, value in cursor.fetchall():
                if value is not None:
                    df.loc[timestamp, f'M_{level}'] = value
            
            # Get Chandelier Exit
            cursor.execute("""
            SELECT timestamp, long_stop, short_stop, direction, signal
            FROM chandelier_exit
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, long_stop, short_stop, direction, signal in cursor.fetchall():
                df.loc[timestamp, 'ce_long_stop'] = long_stop
                df.loc[timestamp, 'ce_short_stop'] = short_stop
                df.loc[timestamp, 'ce_direction'] = direction
                df.loc[timestamp, 'ce_signal'] = signal
            
            # Get OBV
            cursor.execute("""
            SELECT timestamp, value, ma_value, bb_upper, bb_lower
            FROM obv
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            for timestamp, value, ma, bb_upper, bb_lower in cursor.fetchall():
                df.loc[timestamp, 'obv'] = value
                if ma is not None:
                    df.loc[timestamp, 'obv_ma'] = ma
                if bb_upper is not None:
                    df.loc[timestamp, 'obv_bb_upper'] = bb_upper
                if bb_lower is not None:
                    df.loc[timestamp, 'obv_bb_lower'] = bb_lower
    
    return df 