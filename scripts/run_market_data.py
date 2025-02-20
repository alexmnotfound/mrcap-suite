"""
Market data fetching script.
Run with: python -m scripts.run_market_data [options]

Examples:
    # Default: fetch today's data
    python -m scripts.run_market_data

    # Fetch specific date range
    python -m scripts.run_market_data --start 2025-02-01 --end 2025-02-18

    # Fetch from start date until today
    python -m scripts.run_market_data --start 2025-02-01

    # Initialize DB and fetch specific period
    python -m scripts.run_market_data --init-db --start 2025-01-01 --ticker BTCUSDT --timeframes 1H

"""

import argparse
from datetime import datetime, timedelta, timezone
from libs.market_data import init_db, update_ohlc
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES
from libs.utils.logging import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch market data from Binance')
    parser.add_argument('--init-db', action='store_true',
                       help='Initialize the database before fetching data')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD) (default: today)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD) (default: now)')
    parser.add_argument('--tickers', nargs='+', default=TICKERS,
                       help=f'List of tickers to fetch (default: {" ".join(TICKERS)})')
    parser.add_argument('--timeframes', nargs='+', default=list(TIMEFRAMES.values()),
                       help=f'List of timeframes to fetch (default: {" ".join(TIMEFRAMES.values())})')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime with UTC timezone."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.replace(tzinfo=timezone.utc)

def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    logger.info("Starting market data collection")
    
    if args.init_db:
        logger.info("Initializing database...")
        init_db()
    
    # Parse dates
    end_time = datetime.now(timezone.utc)
    start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if args.start:
        start_time = parse_date(args.start)
    if args.end:
        end_time = parse_date(args.end) + timedelta(days=1)  # Include end date
    
    logger.info(f"Fetching data from {start_time.date()} to {end_time.date()}")
    logger.info(f"Tickers: {', '.join(args.tickers)}")
    logger.info(f"Timeframes: {', '.join(args.timeframes)}")
    
    try:
        update_ohlc(start_time=start_time, end_time=end_time, 
                   tickers=args.tickers, timeframes=args.timeframes)
        logger.info("Data collection completed successfully!")
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()