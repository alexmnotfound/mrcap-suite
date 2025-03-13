"""
Script to update missing OHLC data and indicators for all tickers and timeframes.
Run with: python -m scripts.update_ohlc [options]

Examples:
    # Update all tickers and timeframes
    python -m scripts.update_ohlc

    # Update specific ticker
    python -m scripts.update_ohlc --ticker BTCUSDT

    # Update specific timeframe
    python -m scripts.update_ohlc --timeframe 1h

    # Only update OHLC data (skip indicators)
    python -m scripts.update_ohlc --skip-indicators

    # Debug mode for detailed logging
    python -m scripts.update_ohlc --debug

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
    --ticker           Trading pair to update (default: all tickers)
    --timeframe       Candle timeframe to update (default: all timeframes)
    --skip-indicators Skip indicator calculations (faster)
    --debug           Enable debug logging

Update Process:
    1. For each ticker/timeframe combination:
       - Checks last available candle in database
       - If no data exists, fetches last 7 days
       - If data exists, fetches only missing candles
    
    2. Indicator Updates (unless --skip-indicators):
       - EMAs (multiple periods)
       - RSI with slope
       - Chandelier Exit
       - Pivot Points
       - On Balance Volume (OBV)
       
    3. Historical Requirements:
       - Fetches extra historical candles for accurate indicator calculation
       - Base requirement: 200 candles
       - Additional buffer: 50 candles
"""

import argparse
from datetime import datetime, timezone, timedelta
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES, update_indicators
from libs.market_data.db import db_cursor
from libs.utils.logging import setup_logging
import logging
from .fetch_market_data import fetch_and_save_data
import pandas as pd

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

def get_required_history_length(timeframe: str) -> int:
    """
    Get the number of historical candles needed for indicator calculations.
    This should be the maximum lookback period needed by any indicator.
    """
    # Base requirements for different indicators:
    # - EMA200: 200 periods
    # - RSI: 14 periods
    # - Chandelier Exit: 22 periods
    # - Pivots: 2 periods before and after
    # - OBV: No specific requirement, but uses moving averages
    
    # Add some buffer to ensure enough data
    BASE_REQUIREMENT = 200  # Longest EMA period
    BUFFER = 50  # Extra candles for safety
    
    return BASE_REQUIREMENT + BUFFER

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
    
    # Fetch only new data
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
            
            # Get required historical data from database
            required_candles = get_required_history_length(timeframe)
            historical_start = min(new_timestamps) - get_interval_timedelta(timeframe) * required_candles
            
            with db_cursor() as cursor:
                # Fetch all needed data from database
                cursor.execute("""
                SELECT 
                    timestamp AT TIME ZONE 'UTC' as timestamp,
                    CAST(open AS FLOAT) as open,
                    CAST(high AS FLOAT) as high,
                    CAST(low AS FLOAT) as low,
                    CAST(close AS FLOAT) as close,
                    CAST(volume AS FLOAT) as volume
                FROM ohlc_data 
                WHERE ticker = %s 
                AND timeframe = %s
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp ASC
                """, (ticker, timeframe, historical_start, max(new_timestamps)))
                
                data = cursor.fetchall()
                if not data:
                    logger.error("No historical data found for indicator calculation")
                    return new_timestamps
                
                # Create DataFrame with all needed data
                df = pd.DataFrame(data, 
                    columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df.set_index('timestamp', inplace=True)
                
                logger.debug(f"Calculating indicators with {len(df)} candles of data")
                
                # Update indicators only for new timestamps
                update_indicators(ticker, timeframe, new_timestamps, logger, df)
                
        except Exception as e:
            logger.error(f"Error updating indicators for {ticker} {timeframe}: {str(e)}")
            
    return new_timestamps

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