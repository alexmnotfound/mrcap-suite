import requests
from datetime import datetime, timedelta, timezone
from libs.market_data.db import db_cursor
from libs.ta_lib.indicators import add_ema, add_rsi, add_pivots, add_chandelier_exit, add_obv, add_candlestick_patterns
import pandas as pd
import logging
import numpy as np
import time
from libs.market_data.config import APIConfig 

logger = logging.getLogger(__name__)

TICKERS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1M': '1M'
}
BINANCE_API = f"{APIConfig.BASE_URL}/api/v3/klines"
PERIODS = [11, 22, 50, 200]

def get_ticker(cursor, ticker: str):
    """Get ticker from database, create if doesn't exist."""
    cursor.execute("SELECT ticker FROM tickers WHERE ticker = %s", (ticker,))
    result = cursor.fetchone()
    
    if not result:
        # If ticker doesn't exist, create it
        cursor.execute(
            "INSERT INTO tickers (ticker, name) VALUES (%s, %s)",
            (ticker, ticker)  # Using ticker as name for now
        )
    return ticker

def fetch_ohlc(ticker: str, interval: str, start_time: datetime = None, end_time: datetime = None):
    """
    Fetch OHLC data from Binance API.
    
    Args:
        ticker: Trading pair symbol
        interval: Timeframe interval
        start_time: Optional start datetime
        end_time: Optional end datetime
        
    Returns:
        List of candle data
        
    Raises:
        RequestException: If API request fails
        ValueError: If invalid parameters or response format
    """
    params = {
        "symbol": ticker,  
        "interval": interval,
        "limit": 1000
    }
    
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)

    all_candles = []
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            response = requests.get(BINANCE_API, params=params, timeout=10)
            response.raise_for_status()  # Raise error for bad status codes
            
            candles = response.json()
            
            if not isinstance(candles, list):
                raise ValueError(f"Unexpected API response format: {response.text[:200]}")
            
            if not candles:
                break
                
            # Validate candle data format
            for candle in candles:
                if not isinstance(candle, list) or len(candle) < 6:
                    raise ValueError(f"Invalid candle data format: {candle}")
                
                # Validate numeric values
                try:
                    timestamp = int(candle[0])
                    open_price = float(candle[1])
                    high = float(candle[2])
                    low = float(candle[3])
                    close = float(candle[4])
                    volume = float(candle[5])
                    
                    # Basic sanity checks
                    if high < low or high < 0 or low < 0 or volume < 0:
                        raise ValueError(f"Invalid price/volume values in candle: {candle}")
                        
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid numeric values in candle: {candle}") from e
            
            # Remove the last candle if it's incomplete
            if not end_time and len(candles) > 0:
                current_time = datetime.now(timezone.utc).timestamp() * 1000
                last_candle_time = candles[-1][0]
                if last_candle_time + get_interval_ms(interval) > current_time:
                    candles = candles[:-1]
            
            all_candles.extend(candles)
            
            # If we got less than 1000 candles, we've reached the end
            if len(candles) < 1000:
                break
                
            # Update startTime for next iteration
            params["startTime"] = candles[-1][0] + 1
            
            # Reset retry counter on successful request
            retry_count = 0
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            else:
                logger.error("No response received from server")
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Failed to fetch data after {max_retries} retries")
                raise
            
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(f"Request failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
            time.sleep(wait_time)
            continue
            
        except ValueError as e:
            logger.error(f"Invalid data received for {ticker} {interval}")
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker} {interval}")
            raise

    return all_candles

def get_interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    multipliers = {
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000
    }
    
    unit = interval[-1]
    number = int(interval[:-1])
    
    return number * multipliers[unit.lower()]

def save_emas(cursor, ticker: str, timeframe: str, df: pd.DataFrame, save_timestamps: list):
    """Calculate and save EMAs for different periods."""
    for period in PERIODS:
        ema_values = add_ema(df['Close'].values, period=period)
        ema_series = pd.Series(ema_values, index=df.index)
        
        # Only save values for specified timestamps
        for timestamp in save_timestamps:
            if pd.isna(ema_series[timestamp]):
                continue
                
            cursor.execute("""
            INSERT INTO emas (ticker, timeframe, timestamp, period, value)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, timeframe, timestamp, period) 
            DO UPDATE SET value = EXCLUDED.value;
            """, (ticker, timeframe, timestamp, period, float(ema_series[timestamp])))

def save_rsi(cursor, ticker: str, timeframe: str, df: pd.DataFrame, save_timestamps: list, period: int = 14):
    """Calculate and save RSI values."""
    # Calculate RSI for the full dataset
    df_rsi = add_rsi(df, period=period)
    
    # Save RSI values only for specified timestamps
    for timestamp in save_timestamps:
        if timestamp not in df_rsi.index:
            continue
            
        row = df_rsi.loc[timestamp]
        
        # Handle potential NaN or infinite values
        rsi_value = row.get('rsi', None)
        rsi_slope = row.get('rsi_slope', None)
        rsi_div = row.get('rsi_div', None)
        
        # Convert to None if value is NaN or infinite
        if pd.isna(rsi_value) or np.isinf(rsi_value):
            continue
            
        if pd.isna(rsi_slope) or np.isinf(rsi_slope):
            rsi_slope = None
        if pd.isna(rsi_div) or np.isinf(rsi_div):
            rsi_div = None
            
        query = """
        INSERT INTO rsi (ticker, timeframe, timestamp, period, value, slope, divergence)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, timeframe, timestamp, period) DO UPDATE 
        SET value = EXCLUDED.value,
            slope = EXCLUDED.slope,
            divergence = EXCLUDED.divergence;
        """
        cursor.execute(query, (
            ticker,
            timeframe,
            timestamp,
            period,
            float(rsi_value),
            float(rsi_slope) if rsi_slope is not None else None,
            float(rsi_div) if rsi_div is not None else None
        ))

def save_pivots(cursor, ticker: str, timeframe: str, df: pd.DataFrame, save_timestamps: list):
    """Calculate and save pivot points."""
    # Calculate pivots for the full dataset
    df_pivots = add_pivots(df)
    
    # Save pivot values only for specified timestamps
    for timestamp in save_timestamps:
        if timestamp not in df_pivots.index:
            continue
            
        row = df_pivots.loc[timestamp]
        
        # Skip if no pivot values
        if pd.isna(row.get('pivot')):
            continue
            
        cursor.execute("""
        INSERT INTO pivots (
            ticker, timeframe, timestamp,
            pivot, r1, r2, r3, s1, s2, s3
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE SET
            pivot = EXCLUDED.pivot,
            r1 = EXCLUDED.r1,
            r2 = EXCLUDED.r2,
            r3 = EXCLUDED.r3,
            s1 = EXCLUDED.s1,
            s2 = EXCLUDED.s2,
            s3 = EXCLUDED.s3;
        """, (
            ticker, timeframe, timestamp,
            float(row['pivot']),
            float(row['r1']),
            float(row['r2']),
            float(row['r3']),
            float(row['s1']),
            float(row['s2']),
            float(row['s3'])
        ))

def save_chandelier_exit(cursor, ticker: str, timeframe: str, df: pd.DataFrame, 
                        period: int = 22, multiplier: float = 3.0):
    """Calculate and save Chandelier Exit values."""
    logger.debug(f"Calculating Chandelier Exit for {ticker} {timeframe}")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    
    try:
        # Calculate Chandelier Exit
        df_ce = add_chandelier_exit(df, period=period, multiplier=multiplier)
        
        # Save values
        for timestamp, row in df_ce.iterrows():
            if pd.isna(row['ce_long_stop']):
                logger.debug(f"Skipping row with NaN values at {timestamp}")
                continue
                
            # Convert boolean signal to BUY/SELL
            signal = 'BUY' if row['ce_signal'] else 'SELL'
                
            cursor.execute("""
            INSERT INTO chandelier_exit (
                ticker, timeframe, timestamp, period, multiplier,
                long_stop, short_stop, direction, signal
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE SET
                period = EXCLUDED.period,
                multiplier = EXCLUDED.multiplier,
                long_stop = EXCLUDED.long_stop,
                short_stop = EXCLUDED.short_stop,
                direction = EXCLUDED.direction,
                signal = EXCLUDED.signal;
            """, (
                ticker, timeframe, timestamp, period, multiplier,
                float(row['ce_long_stop']),
                float(row['ce_short_stop']),
                int(row['ce_direction']),
                signal
            ))
            
    except Exception as e:
        logger.error(f"Failed to calculate/save Chandelier Exit for {ticker} {timeframe}: {str(e)}")
        raise

def save_obv(cursor, ticker: str, timeframe: str, df: pd.DataFrame, save_timestamps: list):
    """Calculate and save OBV values."""
    # Calculate OBV for the full dataset
    df_obv = add_obv(df, ma_type='SMA + BB')
    
    # Save OBV values only for specified timestamps
    for timestamp in save_timestamps:
        if timestamp not in df_obv.index:
            continue
            
        row = df_obv.loc[timestamp]
        if pd.isna(row['obv']):
            continue
            
        cursor.execute("""
        INSERT INTO obv (ticker, timeframe, timestamp, value, ma_value, bb_upper, bb_lower)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE 
        SET value = EXCLUDED.value,
            ma_value = EXCLUDED.ma_value,
            bb_upper = EXCLUDED.bb_upper,
            bb_lower = EXCLUDED.bb_lower;
        """, (
            ticker,
            timeframe,
            timestamp,
            float(row['obv']),
            float(row['obv_ma']) if 'obv_ma' in row and not pd.isna(row['obv_ma']) else None,
            float(row['obv_bb_upper']) if 'obv_bb_upper' in row and not pd.isna(row['obv_bb_upper']) else None,
            float(row['obv_bb_lower']) if 'obv_bb_lower' in row and not pd.isna(row['obv_bb_lower']) else None
        ))

def save_to_db(data, ticker, timeframe):
    """Insert OHLC data and calculate indicators."""
    try:
        with db_cursor() as cursor:
            timeframe = timeframe.lower() if timeframe != '1M' else timeframe
            
            timestamps = []
            df_ohlc = pd.DataFrame()
            
            for candle in data:
                try:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000, timezone.utc)
                    open_price, high, low, close, volume = map(float, candle[1:6])
                    timestamps.append(timestamp)
                    df_ohlc.loc[timestamp, ['Open', 'High', 'Low', 'Close', 'Volume']] = [open_price, high, low, close, volume]
                    
                    # Save OHLC data
                    cursor.execute("""
                    INSERT INTO ohlc_data (ticker, timeframe, open, high, low, close, volume, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
                    """, (ticker, timeframe, open_price, high, low, close, volume, timestamp))
                    
                except Exception as e:
                    logger.error(f"Error saving candle to database: {e}")
                    raise
            
            # Calculate and save indicators
            save_emas(cursor, ticker, timeframe, df_ohlc, timestamps)
            save_rsi(cursor, ticker, timeframe, df_ohlc, timestamps)
            save_pivots(cursor, ticker, timeframe, df_ohlc, timestamps)
            save_chandelier_exit(cursor, ticker, timeframe, df_ohlc, period=22)
            save_obv(cursor, ticker, timeframe, df_ohlc, timestamps)
            
    except Exception as e:
        logger.error(f"Database error while saving data: {str(e)}")
        raise

def update_ohlc(start_time=None, end_time=None, tickers=None, timeframes=None):
    """
    Fetch and update OHLC data for specified tickers and timeframes.
    
    Args:
        start_time: Start datetime (default: today)
        end_time: End datetime (default: now)
        tickers: List of tickers to fetch (default: TICKERS)
        timeframes: List of timeframes to fetch (default: TIMEFRAMES.values())
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    if tickers is None:
        tickers = TICKERS
    if timeframes is None:
        timeframes = TIMEFRAMES.values()
    
    # Create reverse mapping for timeframes
    timeframe_map = {v: k for k, v in TIMEFRAMES.items()}
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        for interval in timeframes:
            timeframe = timeframe_map.get(interval)
            if timeframe is None:
                logger.warning(f"Unknown timeframe {interval}, skipping...")
                continue
                
            try:
                data = fetch_ohlc(ticker, interval, start_time=start_time, end_time=end_time)
                if data:  # Only save if we got data back
                    save_to_db(data, ticker, timeframe)
            except Exception as e:
                logger.error(f"Error processing {ticker} {interval}: {str(e)}")
                raise
    
    logger.info("Data update completed successfully")

def get_required_history(timeframe: str) -> int:
    """Calculate required number of candles based on timeframe and longest indicator."""
    # Base number of candles needed (e.g., EMA-200 needs 200 candles)
    BASE_CANDLES = 200

    # Add buffer for calculations
    BUFFER = 50
    
    return BASE_CANDLES + BUFFER

def update_indicators(ticker: str, timeframe: str, new_timestamps: list, logger: logging.Logger):
    """Update indicators for specified timestamps using sufficient historical data."""
    try:
        # Convert timeframe to lowercase (except for monthly)
        timeframe = timeframe.lower() if timeframe != '1M' else timeframe
        
        with db_cursor() as cursor:
            # Get required historical data
            required_candles = get_required_history(timeframe)
            
            logger.debug(f"Fetching {required_candles} candles for indicator calculations")
            
            # Ensure timestamps have UTC timezone
            new_timestamps = [
                ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                for ts in new_timestamps
            ]
            
            newest_timestamp = max(new_timestamps)
            
            # Fetch historical data including new timestamps
            cursor.execute("""
            WITH ordered_data AS (
                SELECT 
                    timestamp AT TIME ZONE 'UTC' as timestamp,
                    CAST(open AS FLOAT) as open,
                    CAST(high AS FLOAT) as high,
                    CAST(low AS FLOAT) as low,
                    CAST(close AS FLOAT) as close,
                    CAST(volume AS FLOAT) as volume,
                    ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
                FROM ohlc_data
                WHERE ticker = %s 
                AND timeframe = %s
                AND timestamp <= %s
            )
            SELECT timestamp, open, high, low, close, volume
            FROM ordered_data
            WHERE rn <= %s
            ORDER BY timestamp ASC;
            """, (ticker, timeframe, newest_timestamp, required_candles))
            
            # Create DataFrame with all needed data
            data = cursor.fetchall()
            if not data:
                logger.warning(f"No data found for indicator calculation")
                return
                
            # Use capitalized column names and convert to float
            df = pd.DataFrame(data, 
                columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Calculating indicators with {len(df)} candles of data")
            
            try:
                # Calculate candlestick patterns first
                patterns_df = add_candlestick_patterns(df)
                for ts in new_timestamps:
                    if ts in patterns_df.index:
                        pattern = patterns_df.loc[ts, 'candle_pattern']
                        cursor.execute("""
                        UPDATE ohlc_data 
                        SET candle_pattern = %s
                        WHERE ticker = %s 
                        AND timeframe = %s 
                        AND timestamp = %s
                        """, (pattern, ticker, timeframe, ts))

                # Calculate other indicators
                save_emas(cursor, ticker, timeframe, df, new_timestamps)
                save_rsi(cursor, ticker, timeframe, df, new_timestamps)
                save_pivots(cursor, ticker, timeframe, df, new_timestamps)
                save_chandelier_exit(cursor, ticker, timeframe, df, period=22)
                save_obv(cursor, ticker, timeframe, df, new_timestamps)
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                raise
            
    except Exception as e:
        logger.error(f"Error updating indicators: {str(e)}")
        raise
