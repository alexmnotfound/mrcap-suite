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
        
        # Fetch data from Binance
        data = fetch_ohlc(ticker, timeframe, start_time=start_time, end_time=end_time)
        if not data:
            logger.debug("No data returned from Binance")
            return []
            
        # Save to database
        logger.debug(f"Saving {len(data)} candles to database")
        save_to_db(data, ticker, timeframe)
        
        # Return timestamps of saved data
        return [datetime.fromtimestamp(candle[0] / 1000, timezone.utc) for candle in data]
        
    except Exception as e:
        logger.error(f"Error fetching/saving market data: {e}")
        raise

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
    
    # Determine date range
    end_time = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start_time = parse_date(args.start)
    else:
        start_time = end_time - timedelta(days=args.days)
    
    try:
        new_timestamps = fetch_and_save_data(
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_time=start_time,
            end_time=end_time,
            logger=logger
        )
        if new_timestamps:
            logger.info(f"Successfully saved {len(new_timestamps)} candles")
        else:
            logger.info("No new data to save")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()