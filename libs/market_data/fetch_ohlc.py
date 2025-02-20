import requests
from datetime import datetime, timedelta, timezone
from libs.market_data.db import db_cursor
from libs.ta_lib.indicators import add_ema, add_rsi, add_pivots, add_chandelier_exit, add_obv, add_candlestick_patterns
import pandas as pd
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

TICKERS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = {
    "1D": "1d", 
    "4H": "4h", 
    "1H": "1h",
    "1M": "1M"  # Add monthly timeframe
}
BINANCE_API = "https://api.binance.com/api/v3/klines"
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

def save_emas(cursor, ticker: str, timeframe: str, timestamps: list, closes: list):
    """Calculate and save EMAs for different periods."""
    for period in PERIODS:
        ema_values = add_ema(closes, period=period)
        
        for idx, timestamp in enumerate(timestamps):
            query = """
            INSERT INTO emas (ticker, timeframe, timestamp, period, value)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, timeframe, timestamp, period) DO UPDATE 
            SET value = EXCLUDED.value;
            """
            cursor.execute(query, (
                ticker,
                timeframe, 
                timestamp,
                period,
                float(ema_values[idx])
            ))

def save_rsi(cursor, ticker: str, timeframe: str, timestamps: list, closes: list, period: int = 14):
    """Calculate and save RSI values."""
    # Create DataFrame for RSI calculation
    df = pd.DataFrame({'Close': closes}, index=timestamps)
    df = add_rsi(df, period=period)
    
    # Save RSI values
    for timestamp, row in df.iterrows():
        if pd.isna(row['rsi']):
            continue
            
        # Handle potential NaN or infinite values
        rsi_value = row['rsi']
        rsi_slope = row.get('rsi_slope', None)
        rsi_div = row.get('rsi_div', None)
        
        # Convert to None if value is NaN or infinite
        if pd.isna(rsi_value) or np.isinf(rsi_value):
            rsi_value = None
        if pd.isna(rsi_slope) or np.isinf(rsi_slope):
            rsi_slope = None
        if pd.isna(rsi_div) or np.isinf(rsi_div):
            rsi_div = None
            
        # Skip if main RSI value is None
        if rsi_value is None:
            continue
            
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

def save_pivots(cursor, ticker: str, timeframe: str, timestamps: list, df_ohlc: pd.DataFrame):
    """Calculate and save Pivot Points."""
    df = add_pivots(df_ohlc)
    
    # List of all pivot levels
    pivot_levels = ['PP', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']
    
    for timestamp, row in df.iterrows():
        for level in pivot_levels:
            if pd.isna(row[level]):
                continue
                
            query = """
            INSERT INTO pivots (ticker, timeframe, timestamp, level, value)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, timeframe, timestamp, level) DO UPDATE 
            SET value = EXCLUDED.value;
            """
            cursor.execute(query, (
                ticker,
                timeframe,
                timestamp,
                level,
                float(row[level])
            ))

def save_chandelier_exit(cursor, ticker: str, timeframe: str, df_ohlc: pd.DataFrame, 
                        period: int = 22, multiplier: float = 3.0):
    """Calculate and save Chandelier Exit values."""
    logger.debug(f"Calculating Chandelier Exit for {ticker} {timeframe}")
    logger.debug(f"Input DataFrame shape: {df_ohlc.shape}")
    
    try:
        # Validate input DataFrame
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df_ohlc.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df_ohlc.empty:
            logger.warning(f"Empty DataFrame for {ticker} {timeframe}, skipping Chandelier Exit")
            return
            
        print(df_ohlc.head())
        # Calculate Chandelier Exit
        df = add_chandelier_exit(df_ohlc, period=period, multiplier=multiplier)
        
        # Verify calculation results
        ce_cols = ['ce_long_stop', 'ce_short_stop', 'ce_direction', 'ce_signal']
        missing_ce_cols = [col for col in ce_cols if col not in df.columns]
        if missing_ce_cols:
            raise ValueError(f"Chandelier Exit calculation failed, missing columns: {missing_ce_cols}")
        
        logger.debug(f"Calculated Chandelier Exit values: {len(df)} rows")
        
        # Save values
        rows_saved = 0
        for timestamp, row in df.iterrows():
            if pd.isna(row['ce_long_stop']):
                logger.debug(f"Skipping row with NaN values at {timestamp}")
                continue
                
            try:
                query = """
                INSERT INTO chandelier_exit 
                (ticker, timeframe, timestamp, period, multiplier, long_stop, short_stop, direction, signal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, timeframe, timestamp, period, multiplier) DO UPDATE 
                SET long_stop = EXCLUDED.long_stop,
                    short_stop = EXCLUDED.short_stop,
                    direction = EXCLUDED.direction,
                    signal = EXCLUDED.signal;
                """
                cursor.execute(query, (
                    ticker,
                    timeframe,
                    timestamp,
                    period,
                    multiplier,
                    float(row['ce_long_stop']),
                    float(row['ce_short_stop']),
                    int(row['ce_direction']),
                    int(row['ce_signal'])
                ))
                rows_saved += 1
                
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting Chandelier Exit values at {timestamp}: {e}")
                logger.debug(f"Row values: {row[ce_cols]}")
                raise
            except Exception as e:
                logger.error(f"Database error saving Chandelier Exit at {timestamp}: {e}")
                raise
        
        logger.info(f"Saved {rows_saved} Chandelier Exit values for {ticker} {timeframe}")
        
    except Exception as e:
        logger.error(f"Failed to calculate/save Chandelier Exit for {ticker} {timeframe}: {e}")
        raise

def save_obv(cursor, ticker: str, timeframe: str, df_ohlc: pd.DataFrame):
    """Calculate and save OBV values."""
    df = add_obv(df_ohlc, ma_type='SMA + BB')
    
    for timestamp, row in df.iterrows():
        if pd.isna(row['obv']):
            continue
            
        query = """
        INSERT INTO obv (ticker, timeframe, timestamp, value, ma_value, bb_upper, bb_lower)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE 
        SET value = EXCLUDED.value,
            ma_value = EXCLUDED.ma_value,
            bb_upper = EXCLUDED.bb_upper,
            bb_lower = EXCLUDED.bb_lower;
        """
        cursor.execute(query, (
            ticker,
            timeframe,
            timestamp,
            float(row['obv']),
            float(row['obv_ma']) if 'obv_ma' in row else None,
            float(row['obv_bb_upper']) if 'obv_bb_upper' in row else None,
            float(row['obv_bb_lower']) if 'obv_bb_lower' in row else None
        ))

def save_to_db(data, ticker, timeframe):
    """Insert OHLC data and calculate indicators."""
    try:
        with db_cursor() as cursor:
            ticker = get_ticker(cursor, ticker)
            
            timestamps = []
            closes = []
            df_ohlc = pd.DataFrame()
            
            for candle in data:
                try:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000, timezone.utc)
                    open_price, high, low, close, volume = map(float, candle[1:6])
                    
                    timestamps.append(timestamp)
                    closes.append(close)
                    
                    # Add to OHLC DataFrame for pivot calculations
                    df_ohlc.loc[timestamp, ['Open', 'High', 'Low', 'Close', 'Volume']] = [open_price, high, low, close, volume]
                    
                    query = """
                    INSERT INTO ohlc_data (ticker, timeframe, open, high, low, close, volume, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, timeframe, timestamp) DO NOTHING;
                    """
                    cursor.execute(query, (ticker, timeframe, open_price, high, low, close, volume, timestamp))
                    
                    # Save candlestick pattern
                    pattern = add_candlestick_patterns(df_ohlc).loc[timestamp, 'candle_pattern']
                    cursor.execute("""
                    UPDATE ohlc_data 
                    SET candle_pattern = %s
                    WHERE ticker = %s 
                    AND timeframe = %s 
                    AND timestamp = %s
                    """, (pattern, ticker, timeframe, timestamp))
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing candle data: {candle}")
                    raise ValueError(f"Invalid candle data format") from e
                except Exception as e:
                    logger.error(f"Error saving candle to database: {e}")
                    raise
            
            try:
                # Calculate and save all indicators
                save_emas(cursor, ticker, timeframe, timestamps, closes)
                save_rsi(cursor, ticker, timeframe, timestamps, closes)
                save_pivots(cursor, ticker, timeframe, timestamps, df_ohlc)
                save_chandelier_exit(cursor, ticker, timeframe, df_ohlc)
                save_obv(cursor, ticker, timeframe, df_ohlc)
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                raise
                
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
