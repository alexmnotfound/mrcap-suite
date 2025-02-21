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
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES, fetch_ohlc, update_indicators, save_to_db
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging

def get_last_candle(ticker: str, timeframe: str) -> datetime:
    """Get the timestamp of the last available candle."""
    with db_cursor() as cursor:
        cursor.execute("""
        SELECT timestamp 
        FROM ohlc_data 
        WHERE ticker = %s 
        AND timeframe = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """, (ticker, timeframe))
        
        result = cursor.fetchone()
        return result[0] if result else None

def update_ohlc_data(ticker: str, timeframe: str, logger: logging.Logger) -> list:
    """
    Update OHLC data for a specific ticker and timeframe.
    Returns list of new timestamps that were added.
    """
    # Get last candle timestamp
    last_timestamp = get_last_candle(ticker, timeframe)
    
    if last_timestamp is None:
        logger.info(f"No data found for {ticker} {timeframe}, fetching all available data")
        start_time = None
    else:
        logger.info(f"Last candle for {ticker} {timeframe}: {last_timestamp}")
        # Start from the last candle to ensure it's updated if incomplete
        start_time = last_timestamp
    
    # Fetch and update OHLC data
    data = fetch_ohlc(
        ticker=ticker,
        interval=timeframe,
        start_time=start_time,
        end_time=datetime.now(timezone.utc)
    )
    
    if data:
        logger.info(f"Updated {ticker} {timeframe} with {len(data)} candles")
        return [datetime.fromtimestamp(candle[0] / 1000, timezone.utc) for candle in data]
    else:
        logger.info(f"No new data for {ticker} {timeframe}")
        return []

def update_ohlc(start_time=None, end_time=None, tickers=None, timeframes=None):
    """
    Update missing OHLC data for specified tickers and timeframes.
    Only fetches data newer than the last candle in database.
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if tickers is None:
        tickers = TICKERS
    if timeframes is None:
        timeframes = TIMEFRAMES.values()
    
    # Create reverse mapping for timeframes
    timeframe_map = {v: k for k, v in TIMEFRAMES.items()}
    
    for ticker in tickers:
        logger.info(f"Checking updates for {ticker}")
        for interval in timeframes:
            timeframe = timeframe_map.get(interval)
            if timeframe is None:
                logger.warning(f"Unknown timeframe {interval}, skipping...")
                continue
                
            try:
                # Get last candle timestamp
                with db_cursor() as cursor:
                    cursor.execute("""
                    SELECT timestamp 
                    FROM ohlc_data 
                    WHERE ticker = %s 
                    AND timeframe = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                    """, (ticker, timeframe))
                    
                    result = cursor.fetchone()
                    if result:
                        start_time = result[0]
                        logger.info(f"Last candle for {ticker} {timeframe}: {start_time}")
                    else:
                        logger.info(f"No data found for {ticker} {timeframe}, fetching all available data")
                        start_time = None
                
                # Only fetch if we need new data
                if start_time is None or start_time < end_time:
                    data = fetch_ohlc(ticker, interval, start_time=start_time, end_time=end_time)
                    if data:  # Only save if we got data back
                        logger.info(f"Updated {ticker} {timeframe} with {len(data)} candles")
                        save_to_db(data, ticker, timeframe)
                        
            except Exception as e:
                logger.error(f"Error processing {ticker} {interval}: {str(e)}")
                raise
    
    logger.info("Data update completed")

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
    
    # First phase: Update OHLC data
    updated_data = {}  # Store which timestamps were updated for each ticker/timeframe
    
    for ticker in tickers:
        updated_data[ticker] = {}
        for timeframe in timeframes:
            try:
                new_timestamps = update_ohlc_data(ticker, timeframe, logger)
                if new_timestamps:
                    updated_data[ticker][timeframe] = new_timestamps
            except Exception as e:
                logger.error(f"Error updating OHLC for {ticker} {timeframe}: {str(e)}")
                continue
    
    # Second phase: Update indicators if needed
    if not args.skip_indicators and any(updated_data.values()):
        logger.info("Updating indicators for new data...")
        for ticker in updated_data:
            for timeframe, timestamps in updated_data[ticker].items():
                if timestamps:
                    try:
                        logger.info(f"Calculating indicators for {ticker} {timeframe}")
                        update_indicators(ticker, timeframe, timestamps, logger)
                    except Exception as e:
                        logger.error(f"Error updating indicators for {ticker} {timeframe}: {str(e)}")
                        continue
    
    logger.info("Data update completed")

if __name__ == "__main__":
    main() 