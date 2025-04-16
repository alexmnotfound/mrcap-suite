"""
Script to fetch historical market data and save to database.
Run with: python -m scripts.fetch_market_data [options]

Examples:
    # Initialize database and fetch last 7 days of data
    python -m scripts.fetch_market_data --init-db

    # Fetch specific date range
    python -m scripts.fetch_market_data --start 2025-02-01 --end 2025-02-21

    # Fetch specific ticker and timeframe
    python -m scripts.fetch_market_data --ticker BTCUSDT --timeframe 1h --days 7

    # Fetch all tickers with debug info
    python -m scripts.fetch_market_data --days 30 --debug

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
    --ticker     Trading pair to fetch (default: all configured tickers)
    --timeframe  Candle timeframe (default: all configured timeframes)
    --start      Start date (YYYY-MM-DD)
    --end        End date (YYYY-MM-DD)
    --days       Number of days to fetch (default: 7)
    --init-db    Initialize database before fetching
    --debug      Enable debug logging

Fetching Process:
    1. Database Initialization (if --init-db):
       - Creates required tables if they don't exist
       - Sets up indexes for optimal query performance
    
    2. Data Collection:
       - Fetches OHLCV data from Binance API
       - Includes extra historical data for indicator calculations
       - Validates and processes timestamps
       - Handles rate limiting automatically
    
    3. Data Storage:
       - Saves raw candle data to database
       - Updates technical indicators
       - Calculates candlestick patterns
    
    4. Historical Requirements:
       - Base requirement: 200 candles for indicators
       - Additional buffer: 50 candles
       - Automatically fetches required historical data

Note: This script requires an active internet connection and
      access to the Binance API. Some requests may take time
      due to API rate limits and data volume.
"""

import argparse
from datetime import datetime, timezone, timedelta
from libs.market_data.fetch_ohlc import (
    TICKERS, TIMEFRAMES, fetch_binance_ohlc, save_to_db, update_indicators
)
from libs.market_data.init_db import init_db
from libs.utils.logging import setup_logging
import logging
import pandas as pd

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.astimezone(timezone.utc)

def get_required_history_length(timeframe: str) -> int:
    """Get the number of historical candles needed for indicator calculations."""
    BASE_REQUIREMENT = 200  # Longest EMA period
    BUFFER = 50  # Extra candles for safety
    return BASE_REQUIREMENT + BUFFER

def get_interval_timedelta(interval: str) -> timedelta:
    """Convert interval string to timedelta."""
    unit = interval[-1].lower()
    number = int(interval[:-1])
    
    if unit == 'm':
        return timedelta(minutes=number)
    elif unit == 'h':
        return timedelta(hours=number)
    elif unit == 'd':
        return timedelta(days=number)
    elif unit == 'w':
        return timedelta(weeks=number)
    elif unit == 'M':
        return timedelta(days=30 * number)  # Approximate
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

def fetch_and_save_data(
    ticker: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    logger: logging.Logger
) -> list:
    """
    Fetch and save market data for the specified period.
    Returns list of timestamps for which data was saved.
    """
    try:
        # Convert timeframe to lowercase (except for monthly)
        timeframe = timeframe.lower() if timeframe != '1M' else timeframe
        
        # Validate inputs
        if ticker not in TICKERS:
            raise ValueError(f"Invalid ticker: {ticker}")
        
        if timeframe not in TIMEFRAMES.values():
            raise ValueError(f"Invalid timeframe: {timeframe}")
            
        logger.debug(f"Fetching {ticker} {timeframe} data from {start_time} to {end_time}")
        
        # Calculate start time for historical data needed for indicators
        required_candles = get_required_history_length(timeframe)
        historical_start = start_time - get_interval_timedelta(timeframe) * required_candles
        
        # Fetch data including historical data needed for indicators
        data = fetch_binance_ohlc(ticker, timeframe, start_time=historical_start, end_time=end_time)
        if not data:
            logger.debug("No data returned from Binance")
            return []
            
        # Convert data to DataFrame for indicator calculations
        df_data = []
        timestamps = []
        for candle in data:
            ts = datetime.fromtimestamp(candle[0] / 1000, timezone.utc)
            if ts >= start_time:  # Only include requested timestamps
                timestamps.append(ts)
            df_data.append({
                'timestamp': ts,
                'Open': float(candle[1]),
                'High': float(candle[2]),
                'Low': float(candle[3]),
                'Close': float(candle[4]),
                'Volume': float(candle[5])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Save OHLC data
        logger.debug(f"Saving {len(data)} candles to database")
        save_to_db(data, ticker, timeframe)
        
        if timestamps:
            # Update indicators using the full dataset
            logger.debug(f"Calculating indicators with {len(df)} candles of data")
            update_indicators(ticker, timeframe, timestamps, logger, df)
            
        return timestamps
        
    except Exception as e:
        logger.error(f"Error fetching/saving market data: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and store market data')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Ticker symbol (default: all configured tickers)')
    parser.add_argument('--timeframe', type=str, default=None,
                       help='Timeframe (default: all configured timeframes)')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to fetch (default: 7)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--init-db', action='store_true',
                       help='Initialize database before fetching data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    try:
        # Initialize database if requested
        if args.init_db:
            logger.info("Initializing database...")
            init_db()
            logger.info("Database initialization complete")
        
        # Determine date range
        end_time = parse_date(args.end) if args.end else datetime.now().astimezone(timezone.utc)
        if args.start:
            start_time = parse_date(args.start)
        else:
            start_time = end_time - timedelta(days=args.days)
        
        # If no timeframe specified, fetch all timeframes
        timeframes = [args.timeframe] if args.timeframe else list(TIMEFRAMES.values())
        
        for timeframe in timeframes:
            logger.info(f"Fetching {args.ticker} {timeframe} data...")
            new_timestamps = fetch_and_save_data(
                ticker=args.ticker,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                logger=logger
            )
            if new_timestamps:
                logger.info(f"Successfully saved {len(new_timestamps)} candles for {timeframe}")
            else:
                logger.info(f"No new data to save for {timeframe}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()