"""
Script to fetch the last available candle for a ticker/timeframe.
Run with: python -m scripts.fetch_last_candle [options]

Examples:
    # Show last candle for default ticker/timeframe
    python -m scripts.fetch_last_candle

    # Show last candle for specific ticker
    python -m scripts.fetch_last_candle --ticker BTCUSDT

    # Show last candle for specific timeframe
    python -m scripts.fetch_last_candle --timeframe 1H
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
            
        return {
            'timestamp': result[0],
            'open': float(result[1]),
            'high': float(result[2]),
            'low': float(result[3]),
            'close': float(result[4]),
            'volume': float(result[5]),
            'pattern': result[6]
        }

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
    
    if args.timeframe not in TIMEFRAMES.values():
        logger.error(f"Invalid timeframe: {args.timeframe}")
        return
    
    try:
        # Get last candle
        logger.debug(f"Fetching last candle for {args.ticker} {args.timeframe}")
        candle = get_last_candle(args.ticker, args.timeframe)
        if not candle:
            logger.error(f"No data found for {args.ticker} {args.timeframe}")
            
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
        print(f"\nLast candle for {args.ticker} {args.timeframe}:")
        print(f"Time: {candle['timestamp']}")
        print(f"Open:   {candle['open']:.2f}")
        print(f"High:   {candle['high']:.2f}")
        print(f"Low:    {candle['low']:.2f}")
        print(f"Close:  {candle['close']:.2f}")
        print(f"Volume: {candle['volume']:.8f}")
        if candle['pattern']:
            print(f"Pattern: {candle['pattern']}")
        
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
            """, (args.ticker, args.timeframe, candle['timestamp']))
            
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
            """, (args.ticker, args.timeframe, candle['timestamp']))
            
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