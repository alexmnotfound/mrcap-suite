"""
Script to update missing OHLC data and indicators for all tickers and timeframes.
Run with: python -m scripts.update_ohlc [options]

Examples:
    # Update all tickers and timeframes
    python -m scripts.update_ohlc

    # Update specific ticker
    python -m scripts.update_ohlc --ticker BTCUSDT

    # Update specific timeframe
    python -m scripts.update_ohlc --timeframe 1H

    # Only update OHLC data (skip indicators)
    python -m scripts.update_ohlc --skip-indicators

    # Debug mode 
    python -m scripts.update_ohlc --debug
"""

import argparse
from datetime import datetime, timezone, timedelta
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES, update_indicators
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging
from .fetch_market_data import fetch_and_save_data

def get_last_candle(ticker: str, timeframe: str) -> datetime:
    """Get the timestamp of the last available candle."""
    with db_cursor() as cursor:
        cursor.execute("""
        SELECT timestamp AT TIME ZONE 'UTC'  -- Ensure we get UTC timestamp
        FROM ohlc_data 
        WHERE ticker = %s 
        AND timeframe = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """, (ticker, timeframe))
        
        result = cursor.fetchone()
        if result and result[0]:
            # Ensure returned timestamp is timezone-aware
            ts = result[0]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        return None

def update_ticker_timeframe(
    ticker: str, 
    timeframe: str, 
    skip_indicators: bool,
    logger: logging.Logger
) -> list:
    """
    Update OHLC data and indicators for a specific ticker and timeframe.
    Returns list of new timestamps that were added.
    """
    # Get last candle timestamp
    last_timestamp = get_last_candle(ticker, timeframe)
    end_time = datetime.now(timezone.utc)
    
    if last_timestamp is None:
        logger.info(f"No data found for {ticker} {timeframe}, fetching last 7 days")
        start_time = end_time - timedelta(days=7)
    else:
        logger.info(f"Last candle for {ticker} {timeframe}: {last_timestamp}")
        # Ensure last_timestamp is timezone-aware
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
        # Start from the last candle to ensure it's updated if incomplete
        start_time = last_timestamp
        
    if start_time >= end_time:
        logger.info(f"Data is up to date for {ticker} {timeframe}")
        return []
        
    # Fetch and save new data
    new_timestamps = fetch_and_save_data(
        ticker=ticker,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        logger=logger
    )
    
    if new_timestamps and not skip_indicators:
        try:
            logger.info(f"Calculating indicators for {ticker} {timeframe}")
            update_indicators(ticker, timeframe, new_timestamps, logger)
        except Exception as e:
            logger.error(f"Error updating indicators for {ticker} {timeframe}: {str(e)}")
            
    return new_timestamps

def main():
    parser = argparse.ArgumentParser(description='Update missing OHLC data and indicators')
    parser.add_argument('--ticker', type=str, help='Specific ticker to update')
    parser.add_argument('--timeframe', type=str, help='Specific timeframe to update')
    parser.add_argument('--skip-indicators', action='store_true', help='Skip indicator updates')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    # Filter tickers and timeframes if specified
    tickers = [args.ticker] if args.ticker else TICKERS
    timeframes = [args.timeframe] if args.timeframe else list(TIMEFRAMES.values())
    
    logger.info("Starting data update check")
    logger.info(f"Checking tickers: {', '.join(tickers)}")
    logger.info(f"Checking timeframes: {', '.join(timeframes)}")
    
    for ticker in tickers:
        for timeframe in timeframes:
            try:
                new_timestamps = update_ticker_timeframe(
                    ticker, 
                    timeframe, 
                    args.skip_indicators,
                    logger
                )
                if new_timestamps:
                    logger.info(f"Updated {ticker} {timeframe} with {len(new_timestamps)} candles")
            except Exception as e:
                logger.error(f"Error updating {ticker} {timeframe}: {str(e)}")
                continue
    
    logger.info("Data update completed")

if __name__ == "__main__":
    main() 