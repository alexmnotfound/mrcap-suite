"""
Script to fetch historical market data and save to database.
Run with: python -m scripts.fetch_market_data [options]

Examples:
    # Fetch last 7 days of data
    python -m scripts.fetch_market_data

    # Fetch specific date range
    python -m scripts.fetch_market_data --start 2025-02-01 --end 2025-02-21

    # Fetch specific ticker and timeframe
    python -m scripts.fetch_market_data --ticker BTCUSDT --timeframe 1h --days 7
"""

import argparse
from datetime import datetime, timezone, timedelta
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES, fetch_ohlc, save_to_db
from libs.utils.logging import setup_logging
import logging

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.replace(tzinfo=timezone.utc)

def main():
    parser = argparse.ArgumentParser(description='Fetch historical market data')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Ticker to fetch')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to fetch')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=7, help='Days to look back')
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
    
    # Convert timeframe to lowercase (except for monthly)
    timeframe = args.timeframe.lower() if args.timeframe != '1M' else args.timeframe
    
    # Determine date range
    end_time = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start_time = parse_date(args.start)
    else:
        start_time = end_time - timedelta(days=args.days)
    
    logger.info(f"Fetching {args.ticker} {timeframe} data from {start_time} to {end_time}")
    
    try:
        # Fetch data from Binance
        data = fetch_ohlc(args.ticker, timeframe, start_time=start_time, end_time=end_time)
        if not data:
            logger.warning("No data returned from Binance")
            return
            
        # Save to database
        logger.info(f"Saving {len(data)} candles to database")
        save_to_db(data, args.ticker, timeframe)
        logger.info("Data saved successfully")
        
    except Exception as e:
        logger.error(f"Error fetching/saving market data: {e}")

if __name__ == "__main__":
    main()