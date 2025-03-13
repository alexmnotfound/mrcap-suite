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
    --debug      Enable debug logging

Output:
    Creates a CSV file with the following columns:
    - timestamp: Candle timestamp
    - open, high, low, close, volume: Basic candle data
    - pattern: Detected candlestick pattern
    - rsi, rsi_slope, rsi_div: RSI indicators
    - ema_*: Various exponential moving averages
    - M_*: Monthly pivot points
"""

import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging
import os

def get_candle_data(ticker: str, timeframe: str, timestamp: datetime = None, date: datetime = None) -> list:
    """Get candle data from the database."""
    with db_cursor() as cursor:
        if date:
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
        elif timestamp:
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
                'pattern': result[6]
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
            SELECT value, slope, divergence
            FROM rsi 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            """, (ticker, timeframe, candle_data['timestamp']))
            
            rsi = cursor.fetchone()
            if rsi:
                candle_data['rsi'] = float(rsi[0])
                candle_data['rsi_slope'] = float(rsi[1]) if rsi[1] is not None else None
                candle_data['rsi_div'] = float(rsi[2]) if rsi[2] is not None else None
                
            candles.append(candle_data)
            
        return candles

def create_dataframe(candles: list) -> pd.DataFrame:
    """Create a DataFrame from candle data."""
    # Create DataFrame from list of candles
    df = pd.DataFrame(candles)
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Reorder columns for better readability
    columns = [
        'open', 'high', 'low', 'close', 'volume', 'pattern',
        'rsi', 'rsi_slope', 'rsi_div'
    ]
    
    # Add EMA columns
    ema_cols = sorted([col for col in df.columns if col.startswith('ema_')],
                     key=lambda x: int(x.split('_')[1]))
    columns.extend(ema_cols)
    
    # Add pivot columns
    pivot_cols = sorted([col for col in df.columns if col.startswith('M_')])
    columns.extend(pivot_cols)
    
    # Reorder columns
    df = df[columns]
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze market data for a specific timestamp')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Ticker to analyze')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to analyze')
    parser.add_argument('--timestamp', type=str, help='Specific timestamp (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD)')
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
        if args.timestamp:
            try:
                timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.error("Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS")
                return
        elif args.date:
            try:
                date = datetime.strptime(args.date, '%Y-%m-%d')
                date = date.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD")
                return
        
        # Get candle data
        logger.debug(f"Fetching candle data for {args.ticker} {timeframe}")
        candles = get_candle_data(args.ticker, timeframe, timestamp, date)
        
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
            return
        
        # Create DataFrame
        df = create_dataframe(candles)
        
        # Generate output filename based on parameters
        output_file = f"analysis_{args.ticker}_{timeframe}"
        if timestamp:
            output_file += f"_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        elif date:
            output_file += f"_{date.strftime('%Y%m%d')}"
        output_file += ".csv"
        
        # Save to CSV
        df.to_csv(output_file)
        logger.info(f"Analysis saved to {output_file}")
        
        # Print DataFrame
        print("\nAnalysis DataFrame:")
        print(df)
        
    except Exception as e:
        logger.error(f"Error analyzing market data: {e}")

if __name__ == "__main__":
    main() 