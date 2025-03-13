"""
Script to fetch the last available candle for a ticker/timeframe.
Run with: python -m scripts.get_last_candle [options]

Examples:
    # Show last candle for default ticker/timeframe
    python -m scripts.get_last_candle

    # Show last candle for specific ticker
    python -m scripts.get_last_candle --ticker BTCUSDT

    # Show last candle for specific timeframe
    python -m scripts.get_last_candle --timeframe 1h

    # Show last candle with debug info
    python -m scripts.get_last_candle --debug

Available Tickers:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    (and other major cryptocurrency pairs)

Available Timeframes:
    - 1h  (1 hour candles)
    - 4H  (4 hour candles)
    - 1D  (daily candles)

Options:
    --ticker     Trading pair to analyze (default: BTCUSDT)
    --timeframe  Candle timeframe (default: 1h)
    --debug      Enable debug logging

Output Information:
    Displays for the last candle:
    - Timestamp
    - Basic Data: Open, High, Low, Close, Volume
    - Pattern: If a candlestick pattern is detected
    - EMAs: If available, shows all EMA periods
    - RSI: Value and Slope if available
    
    If no data is found, shows:
    - Total records count for the ticker
    - Available timeframes for the ticker
"""

import argparse
from datetime import datetime, timezone
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging

def get_last_candle(ticker: str, timeframe: str) -> dict:
    """Get the last available candle from the database."""
    with db_cursor() as cursor:
        # Get OHLC data
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
        
        result = cursor.fetchone()
        if not result:
            return None
            
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
            
        return candle_data

def main():
    parser = argparse.ArgumentParser(description='Fetch last available candle')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Ticker to fetch')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to fetch')
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
        # Get last candle
        logger.debug(f"Fetching last candle for {args.ticker} {timeframe}")
        candle = get_last_candle(args.ticker, timeframe)
        if not candle:
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
        
        # Print candle info
        print(f"\nLast candle for {args.ticker} {timeframe}:")
        print(f"Time: {candle['timestamp']}")
        print(f"Open:   {candle['open']:.2f}")
        print(f"High:   {candle['high']:.2f}")
        print(f"Low:    {candle['low']:.2f}")
        print(f"Close:  {candle['close']:.2f}")
        print(f"Volume: {candle['volume']:.8f}")
        if candle['pattern']:
            print(f"Pattern: {candle['pattern']}")
        
        # Print monthly pivots if available
        pivot_cols = [col for col in candle.keys() if col.startswith('M_')]
        if pivot_cols:
            print("\nMonthly Pivot Points:")
            for col in sorted(pivot_cols):
                level = col.replace('M_', '')
                print(f"  {level}: {candle[col]:.2f}")
        
        # Also get indicator values
        with db_cursor() as cursor:
            # Get EMAs
            cursor.execute("""
            SELECT period, value 
            FROM emas 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            ORDER BY period
            """, (args.ticker, timeframe, candle['timestamp']))
            
            emas = cursor.fetchall()
            if emas:
                print("\nMoving Averages:")
                for period, value in emas:
                    print(f"  EMA-{period}: {float(value):.8f}")
            
            # Get RSI
            cursor.execute("""
            SELECT value, slope
            FROM rsi 
            WHERE ticker = %s 
            AND timeframe = %s 
            AND timestamp = %s
            """, (args.ticker, timeframe, candle['timestamp']))
            
            rsi = cursor.fetchone()
            if rsi:
                print("\nRSI:")
                print(f"  Value: {float(rsi[0]):.8f}")
                if rsi[1]:
                    print(f"  Slope: {float(rsi[1]):.8f}")
    except Exception as e:
        logger.error(f"Error fetching last candle: {e}")

if __name__ == "__main__":
    main() 