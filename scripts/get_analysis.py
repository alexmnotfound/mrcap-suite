"""
Script to get and analyze market data with indicators.
Run with: python -m scripts.get_analysis [options]

Examples:
    # Get last candle analysis
    python -m scripts.get_analysis

    # Get specific timestamp
    python -m scripts.get_analysis --timestamp "2025-03-13 10:00:00"

    # Get all candles for a specific date
    python -m scripts.get_analysis --date 2025-03-13

    # Get candles for a date range
    python -m scripts.get_analysis --start 2025-03-01 --end 2025-03-13

    # Get specific ticker and timeframe
    python -m scripts.get_analysis --ticker ETHUSDT --timeframe 4h

    # Get analysis with debug info
    python -m scripts.get_analysis --debug

Available Tickers:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    (and other major cryptocurrency pairs)

Available Timeframes:
    - 1h  (1 hour candles)
    - 4h  (4 hour candles)
    - 1d  (daily candles)

Options:
    --ticker     Trading pair to analyze (default: BTCUSDT)
    --timeframe  Candle timeframe (default: 1h)
    --timestamp  Specific timestamp to analyze (YYYY-MM-DD HH:MM:SS)
    --date       Specific date to analyze (YYYY-MM-DD)
    --start      Start date for period (YYYY-MM-DD)
    --end        End date for period (YYYY-MM-DD)
    --debug      Enable debug logging

Output:
    Creates a CSV file with the following columns:
    - timestamp: Candle timestamp
    - open, high, low, close, volume: Basic candle data
    - pattern: Detected candlestick pattern
    - rsi: RSI indicator
    - ema_*: Various exponential moving averages
    - M_*: Monthly pivot points
    - ce_long_stop, ce_short_stop, ce_direction, ce_signal: Chandelier Exit values
"""

import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging
import os
import numpy as np
from libs.market_data.get_data import get_market_data
from libs.ta_lib.indicators.chandelier import add_chandelier_exit
from libs.ta_lib.indicators.rsi import add_rsi
from libs.ta_lib.indicators.obv import add_obv
from libs.ta_lib.indicators.ema import add_emas
from libs.ta_lib.indicators.candles import add_candle_patterns

def get_candle_data(ticker: str, timeframe: str, timestamp: datetime = None, date: datetime = None, 
                    start_date: datetime = None, end_date: datetime = None) -> list:
    """Get candle data from the database."""
    with db_cursor() as cursor:
        if timestamp:
            # Get specific timestamp
            cursor.execute("""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                candle_pattern
            FROM ohlc_data 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            """, (ticker, timeframe, timestamp))
        elif date:
            # Get all candles for the specified date
            start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
            
            cursor.execute("""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                candle_pattern
            FROM ohlc_data 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp >= %s
            AND timestamp < %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
        elif start_date and end_date:
            # Get candles for date range
            start_time = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Debug: Log the query parameters
            logging.debug(f"Fetching candles from {start_time} to {end_time}")
            
            cursor.execute("""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                candle_pattern
            FROM ohlc_data 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp >= %s
            AND timestamp <= %s
            ORDER BY timestamp
            """, (ticker, timeframe, start_time, end_time))
            
            # Debug: Log the number of results
            results = cursor.fetchall()
            logging.debug(f"Found {len(results)} candles in the date range")
            
            # Debug: Log the first and last timestamps if we have results
            if results:
                logging.debug(f"First candle: {results[0][0]}")
                logging.debug(f"Last candle: {results[-1][0]}")
            
            candles = []
            for result in results:
                candle_data = {
                    'timestamp': result[0],
                    'open': float(result[1]),
                    'high': float(result[2]),
                    'low': float(result[3]),
                    'close': float(result[4]),
                    'volume': float(result[5]),
                    'pattern': result[6],
                    # Initialize indicator columns with None
                    'rsi': None
                }
                
                # Get monthly pivots for the current month
                cursor.execute("""
                WITH monthly_data AS (
                    SELECT timestamp, level, value,
                           DATE_TRUNC('month', timestamp) as pivot_month
                    FROM pivots 
                    WHERE ticker = %s 
                    AND timeframe = '1M'
                    AND DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', %s::timestamp - INTERVAL '1 month')
                )
                SELECT level, value
                FROM monthly_data
                ORDER BY level;
                """, (ticker, candle_data['timestamp']))
                
                for level, value in cursor.fetchall():
                    candle_data[f'M_{level}'] = float(value)
                    
                # Get EMAs
                cursor.execute("""
                SELECT period, value 
                FROM emas 
                WHERE ticker = %s 
                AND timeframe = %s 
                AND timestamp = %s
                ORDER BY period
                """, (ticker, timeframe, candle_data['timestamp']))
                
                for period, value in cursor.fetchall():
                    candle_data[f'ema_{period}'] = float(value)
                
                # Get RSI
                cursor.execute("""
                SELECT value
                FROM rsi 
                WHERE ticker = %s 
                AND timeframe = %s 
                AND timestamp = %s
                """, (ticker, timeframe, candle_data['timestamp']))
                
                rsi = cursor.fetchone()
                if rsi:
                    candle_data['rsi'] = float(rsi[0]) if rsi[0] is not None else None
                
                # Get Chandelier Exit
                cursor.execute("""
                SELECT long_stop, short_stop, direction, signal
                FROM chandelier_exit 
                WHERE ticker = %s 
                AND timeframe = %s 
                AND timestamp = %s
                """, (ticker, timeframe, candle_data['timestamp']))
                
                ce = cursor.fetchone()
                if ce:
                    candle_data['ce_long_stop'] = float(ce[0]) if ce[0] is not None else None
                    candle_data['ce_short_stop'] = float(ce[1]) if ce[1] is not None else None
                    candle_data['ce_direction'] = int(ce[2]) if ce[2] is not None else None
                    candle_data['ce_signal'] = int(ce[3]) if ce[3] is not None else None
                    
                candles.append(candle_data)
            
            return candles
        else:
            # Get last candle
            cursor.execute("""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                candle_pattern
            FROM ohlc_data 
            WHERE ticker = %s 
            AND timeframe = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
            """, (ticker, timeframe))
        
        results = cursor.fetchall()
        if not results:
            return None
            
        candles = []
        for result in results:
            candle_data = {
                'timestamp': result[0],
                'open': float(result[1]),
                'high': float(result[2]),
                'low': float(result[3]),
                'close': float(result[4]),
                'volume': float(result[5]),
                'pattern': result[6],
                # Initialize indicator columns with None
                'rsi': None
            }
            
            # Get monthly pivots for the current month
            cursor.execute("""
            WITH monthly_data AS (
                SELECT timestamp, level, value,
                       DATE_TRUNC('month', timestamp) as pivot_month
                FROM pivots 
                WHERE ticker = %s 
                AND timeframe = '1M'
                AND DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', %s::timestamp - INTERVAL '1 month')
            )
            SELECT level, value
            FROM monthly_data
            ORDER BY level;
            """, (ticker, candle_data['timestamp']))
            
            for level, value in cursor.fetchall():
                candle_data[f'M_{level}'] = float(value)
                
            # Get EMAs
            cursor.execute("""
            SELECT period, value 
            FROM emas 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            ORDER BY period
            """, (ticker, timeframe, candle_data['timestamp']))
            
            for period, value in cursor.fetchall():
                candle_data[f'ema_{period}'] = float(value)
            
            # Get RSI
            cursor.execute("""
            SELECT value
            FROM rsi 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            """, (ticker, timeframe, candle_data['timestamp']))
            
            rsi = cursor.fetchone()
            if rsi:
                candle_data['rsi'] = float(rsi[0]) if rsi[0] is not None else None
            
            # Get Chandelier Exit
            cursor.execute("""
            SELECT long_stop, short_stop, direction, signal
            FROM chandelier_exit 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            """, (ticker, timeframe, candle_data['timestamp']))
            
            ce = cursor.fetchone()
            if ce:
                candle_data['ce_long_stop'] = float(ce[0]) if ce[0] is not None else None
                candle_data['ce_short_stop'] = float(ce[1]) if ce[1] is not None else None
                candle_data['ce_direction'] = int(ce[2]) if ce[2] is not None else None
                candle_data['ce_signal'] = int(ce[3]) if ce[3] is not None else None
                
            candles.append(candle_data)
            
        return candles

def create_dataframe(candles: list) -> pd.DataFrame:
    """Create a DataFrame from candle data."""
    # Debug: Log the number of candles being processed
    logging.debug(f"Creating DataFrame from {len(candles)} candles")
    
    # Create DataFrame from list of candles
    df = pd.DataFrame(candles)
    
    # Debug: Log DataFrame info
    logging.debug(f"DataFrame shape: {df.shape}")
    logging.debug(f"DataFrame columns: {df.columns.tolist()}")
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # For debugging Chandelier Exit, keep all values
    if 'ce_direction' in df.columns:
        # Map direction values but keep original for debugging
        df['ce_direction_text'] = df['ce_direction'].map({
            1: 'LONG',
            -1: 'SHORT'
        })
        
        # Keep both stops visible
        def round_stop(value):
            if pd.isna(value):
                return value
            if value >= 100:
                return round(value, 2)
            return round(value, 3)
            
        df['ce_long_stop'] = df['ce_long_stop'].apply(round_stop)
        df['ce_short_stop'] = df['ce_short_stop'].apply(round_stop)
        
        # Add signal column if present
        if 'ce_signal' in df.columns:
            df['ce_signal'] = df['ce_signal'].map({
                1: 'BUY',
                -1: 'SELL',
                0: '-'
            })
    
    # Round EMA values
    ema_cols = [col for col in df.columns if col.startswith('ema_')]
    for col in ema_cols:
        df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else x)
    
    # Round pivot values
    pivot_cols = [col for col in df.columns if col.startswith('M_')]
    for col in pivot_cols:
        df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else x)
    
    # Define all possible columns
    all_columns = [
        'open', 'high', 'low', 'close', 'volume', 'pattern',
        'ce_direction', 'ce_direction_text', 'ce_long_stop', 'ce_short_stop', 'ce_signal',  # Debug columns
        'rsi'
    ]
    
    # Add EMA columns
    ema_cols = sorted([col for col in df.columns if col.startswith('ema_')],
                     key=lambda x: int(x.split('_')[1]))
    all_columns.extend(ema_cols)
    
    # Add pivot columns
    pivot_cols = sorted([col for col in df.columns if col.startswith('M_')])
    all_columns.extend(pivot_cols)
    
    # Debug: Log the columns we're using
    logging.debug(f"All columns to include: {all_columns}")
    
    # Create a new DataFrame with all columns, filling missing values with None
    new_df = pd.DataFrame(index=df.index)
    for col in all_columns:
        new_df[col] = df.get(col, None)
    
    # Debug: Log final DataFrame info
    logging.debug(f"Final DataFrame shape: {new_df.shape}")
    logging.debug(f"Final DataFrame columns: {new_df.columns.tolist()}")
    
    return new_df

def get_analysis(ticker: str, timeframe: str, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
    """
    Get market data with all indicators calculated.
    
    Args:
        ticker: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe in Binance format (e.g., '1h', '4h', '1d', '1M')
        start_time: Start datetime (default: 7 days ago)
        end_time: End datetime (default: now)
    
    Returns:
        DataFrame with OHLC data and all indicators
    """
    # Ensure timestamps are timezone-aware and in UTC
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time - pd.Timedelta(days=7)
        
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    # Get base market data
    df = get_market_data(ticker, timeframe, start_time, end_time, include_indicators=False)
    
    if df.empty:
        return df
    
    # Calculate indicators
    df = add_emas(df, periods=[9, 21, 50, 200])
    df = add_rsi(df, period=14)
    df = add_obv(df, ma_period=20, bb_period=20, bb_std=2)
    df = add_chandelier_exit(df, period=22, atr_mult=3)
    df = add_candle_patterns(df)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze market data for a specific timestamp or period')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Ticker to analyze')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to analyze')
    parser.add_argument('--timestamp', type=str, help='Specific timestamp (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date for period (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for period (YYYY-MM-DD)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    # Validate inputs
    if args.ticker not in TICKERS:
        logger.error(f"Invalid ticker: {args.ticker}")
        return
    
    # Ensure timeframe is uppercase for validation
    timeframe = args.timeframe
    if timeframe not in TIMEFRAMES.values():
        logger.error(f"Invalid timeframe: {timeframe}")
        return
    
    try:
        # Parse timestamp or date if provided
        timestamp = None
        date = None
        start_date = None
        end_date = None
        
        if args.timestamp:
            try:
                timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                logger.debug(f"Analyzing specific timestamp: {timestamp}")
            except ValueError:
                logger.error("Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS")
                return
        elif args.date:
            try:
                date = datetime.strptime(args.date, '%Y-%m-%d')
                date = date.replace(tzinfo=timezone.utc)
                logger.debug(f"Analyzing specific date: {date}")
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD")
                return
        elif args.start:
            try:
                start_date = datetime.strptime(args.start, '%Y-%m-%d')
                start_date = start_date.replace(tzinfo=timezone.utc)
                if args.end:
                    end_date = datetime.strptime(args.end, '%Y-%m-%d')
                    end_date = end_date.replace(tzinfo=timezone.utc)
                    if end_date < start_date:
                        logger.error("End date must be after start date")
                        return
                    logger.debug(f"Analyzing date range: {start_date} to {end_date}")
                else:
                    end_date = datetime.now(timezone.utc)
                    logger.debug(f"Analyzing from {start_date} to now")
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD")
                return
        
        # Get candle data
        logger.debug(f"Fetching candle data for {args.ticker} {timeframe}")
        candles = get_candle_data(args.ticker, timeframe, timestamp, date, start_date, end_date)
        
        if not candles:
            logger.error(f"No data found for {args.ticker} {timeframe}")
            
            # Check if we have any data at all
            with db_cursor() as cursor:
                cursor.execute("""
                SELECT COUNT(*) FROM ohlc_data 
                WHERE ticker = %s
                """, (args.ticker,))
                count = cursor.fetchone()[0]
                logger.debug(f"Total records for {args.ticker}: {count}")
                
                cursor.execute("""
                SELECT DISTINCT timeframe FROM ohlc_data 
                WHERE ticker = %s
                """, (args.ticker,))
                timeframes = cursor.fetchall()
                logger.debug(f"Available timeframes: {[t[0] for t in timeframes]}")
                
                # Debug: Check data in the requested date range
                if start_date and end_date:
                    cursor.execute("""
                    SELECT COUNT(*) FROM ohlc_data 
                    WHERE ticker = %s 
                    AND timeframe = %s
                    AND timestamp >= %s
                    AND timestamp <= %s
                    """, (args.ticker, timeframe, start_date, end_date))
                    range_count = cursor.fetchone()[0]
                    logger.debug(f"Records in date range: {range_count}")
                    
                    # Get min and max timestamps
                    cursor.execute("""
                    SELECT MIN(timestamp), MAX(timestamp) FROM ohlc_data 
                    WHERE ticker = %s 
                    AND timeframe = %s
                    """, (args.ticker, timeframe))
                    min_ts, max_ts = cursor.fetchone()
                    logger.debug(f"Data range: {min_ts} to {max_ts}")
            return
        
        # Create DataFrame
        df = create_dataframe(candles)
        
        # Generate output filename based on parameters
        output_file = f"analysis_{args.ticker}_{timeframe}"
        if timestamp:
            output_file += f"_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        elif date:
            output_file += f"_{date.strftime('%Y%m%d')}"
        elif start_date and end_date:
            output_file += f"_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        output_file += ".csv"
        
        # Save to CSV
        df.to_csv(output_file)
        logger.info(f"Analysis saved to {output_file}")
        
        # Print DataFrame
        print("\nAnalysis DataFrame:")
        print(df)
        # print(df[['open', 'close', 'ce_long_stop', 'ce_short_stop', 'ce_signal', 'ce_direction_text']])
        
    except Exception as e:
        logger.error(f"Error analyzing market data: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 