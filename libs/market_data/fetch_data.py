from datetime import datetime, timedelta
import pandas as pd
from .db import db_cursor

def fetch_market_data(
    ticker: str,
    timeframe: str,
    start_time: datetime = None,
    end_time: datetime = None,
    include_indicators: bool = True
) -> pd.DataFrame:
    """
    Fetch market data with indicators from database.
    
    Args:
        ticker: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1H', '4H', '1D')
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
        
    # Map timeframe to database format
    timeframe_map = {'1H': '1H', '4H': '4H', '1D': '1D'}
    db_timeframe = timeframe_map.get(timeframe)
    if not db_timeframe:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: 1H, 4H, 1D")
    
    with db_cursor() as cursor:
        # Fetch OHLC data
        cursor.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlc_data
        WHERE ticker = %s 
        AND timeframe = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """, (ticker, db_timeframe, start_time, end_time))
        
        df = pd.DataFrame(cursor.fetchall(), 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if df.empty:
            return df
            
        df.set_index('timestamp', inplace=True)
        
        if include_indicators:
            # Fetch EMAs
            cursor.execute("""
            SELECT timestamp, period, value
            FROM emas
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp, period
            """, (ticker, db_timeframe, start_time, end_time))
            
            for timestamp, period, value in cursor.fetchall():
                df.loc[timestamp, f'ema_{period}'] = value
            
            # Fetch RSI
            cursor.execute("""
            SELECT timestamp, value, slope, divergence
            FROM rsi
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, db_timeframe, start_time, end_time))
            
            for timestamp, value, slope, div in cursor.fetchall():
                df.loc[timestamp, 'rsi'] = value
                df.loc[timestamp, 'rsi_slope'] = slope
                df.loc[timestamp, 'rsi_div'] = div
            
            # Fetch monthly pivots - get previous month's data for current month's pivots
            cursor.execute("""
            WITH monthly_data AS (
                SELECT timestamp, level, value,
                       DATE_TRUNC('month', timestamp + INTERVAL '1 month') as apply_month
                FROM pivots 
                WHERE ticker = %s 
                AND timeframe = '1M'
                AND DATE_TRUNC('month', timestamp + INTERVAL '1 month') 
                    BETWEEN DATE_TRUNC('month', %s::timestamp) 
                    AND DATE_TRUNC('month', %s::timestamp)
            )
            SELECT d.timestamp, md.level, md.value
            FROM generate_series(%s::timestamp, %s::timestamp, '1 hour') as d(timestamp)
            LEFT JOIN monthly_data md 
            ON DATE_TRUNC('month', d.timestamp) = md.apply_month
            ORDER BY d.timestamp, md.level;
            """, (ticker, start_time, end_time, start_time, end_time))
            
            for timestamp, level, value in cursor.fetchall():
                if value is not None:
                    df.loc[timestamp, f'M_{level}'] = value
            
            # Fetch Chandelier Exit
            cursor.execute("""
            SELECT timestamp, long_stop, short_stop, direction, signal
            FROM chandelier_exit
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, db_timeframe, start_time, end_time))
            
            for timestamp, long_stop, short_stop, direction, signal in cursor.fetchall():
                df.loc[timestamp, 'ce_long_stop'] = long_stop
                df.loc[timestamp, 'ce_short_stop'] = short_stop
                df.loc[timestamp, 'ce_direction'] = direction
                df.loc[timestamp, 'ce_signal'] = signal
            
            # Fetch OBV
            cursor.execute("""
            SELECT timestamp, value, ma_value, bb_upper, bb_lower
            FROM obv
            WHERE ticker = %s 
            AND timeframe = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """, (ticker, db_timeframe, start_time, end_time))
            
            for timestamp, value, ma, bb_upper, bb_lower in cursor.fetchall():
                df.loc[timestamp, 'obv'] = value
                if ma is not None:
                    df.loc[timestamp, 'obv_ma'] = ma
                if bb_upper is not None:
                    df.loc[timestamp, 'obv_bb_upper'] = bb_upper
                if bb_lower is not None:
                    df.loc[timestamp, 'obv_bb_lower'] = bb_lower
    
    return df 