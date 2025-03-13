import requests
from datetime import datetime, timedelta, timezone
from libs.market_data.db import db_cursor, batch_insert
from libs.ta_lib.indicators import add_ema, add_rsi, add_pivots, add_chandelier_exit, add_obv, add_candlestick_patterns
import pandas as pd
import logging
import numpy as np
import time
from libs.market_data.config import APIConfig 
from typing import List, Optional
from .config import market_config

logger = logging.getLogger(__name__)

# Use the config values
BINANCE_API = f"{APIConfig.BASE_URL}/api/v3/klines"
TICKERS = market_config.TICKERS
TIMEFRAMES = market_config.TIMEFRAMES
PERIODS = market_config.EMA_PERIODS

def get_ticker(cursor, ticker: str):
    """Get ticker from database, create if doesn't exist."""
    try:
        cursor.execute("SELECT ticker FROM tickers WHERE ticker = %s", (ticker,))
        result = cursor.fetchone()
        
        if not result:
            logger.debug(f"Creating new ticker entry for {ticker}")
            cursor.execute(
                "INSERT INTO tickers (ticker, name) VALUES (%s, %s)",
                (ticker, ticker)  # Using ticker as name for now    
            )
            logger.info(f"Created new ticker: {ticker}")
        else:
            logger.debug(f"Found existing ticker: {ticker}")
        return ticker
    except Exception as e:
        logger.error(f"Error getting/creating ticker {ticker}: {str(e)}")
        raise

def fetch_binance_ohlc(ticker: str, interval: str, start_time: datetime = None, end_time: datetime = None):
    """
    Fetch OHLC data from Binance API.
    
    Args:
        ticker: Trading pair symbol
        interval: Timeframe interval (in Binance format: 1h, 4h, 1d, 1M)
        start_time: Optional start datetime
        end_time: Optional end datetime
        
    Returns:
        List of candle data
        
    Raises:
        RequestException: If API request fails
        ValueError: If invalid parameters or response format
    """
    # Validate interval format
    if interval not in TIMEFRAMES.values():
        raise ValueError(f"Invalid interval: {interval}. Must be one of: {', '.join(TIMEFRAMES.values())}")
    
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

def save_to_db(data: list, ticker: str, timeframe: str):
    """Insert OHLC data using batch insert."""
    try:
        logger.debug(f"Preparing to save {len(data)} candles for {ticker} {timeframe}")
        
        # Validate timeframe
        if timeframe not in TIMEFRAMES.values():
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Ensure ticker exists in database
        with db_cursor() as cursor:
            get_ticker(cursor, ticker)
        
        # Prepare OHLC data for batch insert
        ohlc_values = []
        timestamps = []
        df_ohlc = pd.DataFrame()
        
        for candle in data:
            timestamp = datetime.fromtimestamp(candle[0] / 1000, timezone.utc)
            open_price, high, low, close, volume = map(float, candle[1:6])
            
            timestamps.append(timestamp)
            df_ohlc.loc[timestamp, ['Open', 'High', 'Low', 'Close', 'Volume']] = [
                open_price, high, low, close, volume
            ]
            
            ohlc_values.append((
                ticker, timeframe, timestamp,
                open_price, high, low, close, volume
            ))
        
        # Batch insert OHLC data
        logger.debug("Saving OHLC data...")
        batch_insert(
            table='ohlc_data',
            columns=['ticker', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
            values=ohlc_values,
            conflict_action="(ticker, timeframe, timestamp) DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume"
        )
        
        # Calculate and batch save candlestick patterns
        logger.debug("Calculating candlestick patterns...")
        patterns_df = add_candlestick_patterns(df_ohlc)
        pattern_data = []
        for ts in timestamps:
            if ts in patterns_df.index:
                pattern = patterns_df.loc[ts, 'candle_pattern']
                if pattern != 'Normal':  # Only save non-normal patterns
                    pattern_data.append((ticker, timeframe, ts, pattern))
        
        if pattern_data:
            save_patterns_batch(ticker, timeframe, pattern_data)
        
        logger.info(f"Successfully saved {len(data)} candles data for {ticker} {timeframe}")
        return timestamps
        
    except Exception as e:
        logger.error(f"Database error while saving data: {str(e)}")
        raise

def save_ema_batch(ticker: str, timeframe: str, ema_data: List[tuple]):
    """Batch save EMA values."""
    try:
        batch_insert(
            table='emas',
            columns=['ticker', 'timeframe', 'timestamp', 'period', 'value'],
            values=ema_data,
            conflict_action="(ticker, timeframe, timestamp, period) DO UPDATE SET value = EXCLUDED.value"
        )
        logger.debug(f"Saved {len(ema_data)} EMA values")
    except Exception as e:
        logger.error(f"Error saving EMA values: {str(e)}")
        raise

def save_rsi_batch(ticker: str, timeframe: str, rsi_data: List[tuple]):
    """Batch save RSI values."""
    try:
        batch_insert(
            table='rsi',
            columns=['ticker', 'timeframe', 'timestamp', 'period', 'value', 'slope', 'divergence'],
            values=rsi_data,
            conflict_action="(ticker, timeframe, timestamp, period) DO UPDATE SET value = EXCLUDED.value, slope = EXCLUDED.slope, divergence = EXCLUDED.divergence"
        )
        logger.debug(f"Saved {len(rsi_data)} RSI values")
    except Exception as e:
        logger.error(f"Error saving RSI values: {str(e)}")
        raise

def save_pivots_batch(ticker: str, timeframe: str, pivot_data: List[tuple]):
    """Batch save pivot point values."""
    try:
        batch_insert(
            table='pivots',
            columns=['ticker', 'timeframe', 'timestamp', 'level', 'value'],
            values=pivot_data,
            conflict_action="(ticker, timeframe, timestamp, level) DO UPDATE SET value = EXCLUDED.value"
        )
        logger.debug(f"Saved {len(pivot_data)} pivot points")
    except Exception as e:
        logger.error(f"Error saving pivot points: {str(e)}")
        raise

def save_chandelier_batch(ticker: str, timeframe: str, ce_data: List[tuple]):
    """Batch save Chandelier Exit values."""
    try:
        batch_insert(
            table='chandelier_exit',
            columns=['ticker', 'timeframe', 'timestamp', 'period', 'multiplier',
                    'long_stop', 'short_stop', 'direction', 'signal'],
            values=ce_data,
            conflict_action="(ticker, timeframe, timestamp, period, multiplier) DO UPDATE SET long_stop = EXCLUDED.long_stop, short_stop = EXCLUDED.short_stop, direction = EXCLUDED.direction, signal = EXCLUDED.signal"
        )
        logger.debug(f"Saved {len(ce_data)} Chandelier Exit values")
    except Exception as e:
        logger.error(f"Error saving Chandelier Exit values: {str(e)}")
        raise

def save_obv_batch(ticker: str, timeframe: str, obv_data: List[tuple]):
    """Batch save OBV values."""
    try:
        batch_insert(
            table='obv',
            columns=['ticker', 'timeframe', 'timestamp', 'value', 
                    'ma_value', 'bb_upper', 'bb_lower'],
            values=obv_data,
            conflict_action="(ticker, timeframe, timestamp) DO UPDATE SET value = EXCLUDED.value, ma_value = EXCLUDED.ma_value, bb_upper = EXCLUDED.bb_upper, bb_lower = EXCLUDED.bb_lower"
        )
        logger.debug(f"Saved {len(obv_data)} OBV values")
    except Exception as e:
        logger.error(f"Error saving OBV values: {str(e)}")
        raise

def save_patterns_batch(ticker: str, timeframe: str, pattern_data: List[tuple]):
    """Batch save candlestick patterns."""
    try:
        with db_cursor() as cursor:
            for ticker, timeframe, timestamp, pattern in pattern_data:
                cursor.execute("""
                UPDATE ohlc_data 
                SET candle_pattern = %s
                WHERE ticker = %s AND timeframe = %s AND timestamp = %s
                """, (pattern, ticker, timeframe, timestamp))
        logger.debug(f"Saved {len(pattern_data)} candlestick patterns")
    except Exception as e:
        logger.error(f"Error saving candlestick patterns: {str(e)}")
        raise

def update_indicators(ticker: str, timeframe: str, timestamps: list, logger: logging.Logger, df: pd.DataFrame) -> None:
    """Update all indicators using batch operations."""
    try:
        logger.debug(f"Updating indicators for {ticker} {timeframe}")
        logger.debug(f"Processing {len(timestamps)} timestamps")
        
        # Calculate EMAs
        logger.debug("Calculating EMAs...")
        ema_data = []
        for period in PERIODS:
            ema_values = add_ema(df['Close'].values, period=period)
            ema_series = pd.Series(ema_values, index=df.index)
            
            for ts in timestamps:
                if ts in ema_series.index and not pd.isna(ema_series[ts]):
                    ema_data.append((ticker, timeframe, ts, period, float(ema_series[ts])))
        
        if ema_data:
            logger.debug(f"Saving {len(ema_data)} EMA values")
            save_ema_batch(ticker, timeframe, ema_data)
        
        # Calculate RSI
        logger.debug("Calculating RSI...")
        df_rsi = add_rsi(df)
        rsi_data = []
        for ts in timestamps:
            if ts in df_rsi.index:
                row = df_rsi.loc[ts]
                if not pd.isna(row['rsi']):
                    rsi_data.append((
                        ticker, timeframe, ts, 14,  # Default RSI period
                        float(row['rsi']),
                        float(row['rsi_slope']) if 'rsi_slope' in row and not pd.isna(row['rsi_slope']) else None,
                        float(row['rsi_div']) if 'rsi_div' in row and not pd.isna(row['rsi_div']) else None
                    ))
        
        if rsi_data:
            logger.debug(f"Saving {len(rsi_data)} RSI values")
            save_rsi_batch(ticker, timeframe, rsi_data)
            
        # Calculate Pivot Points
        logger.debug("Calculating Pivot Points...")
        df_pivots = add_pivots(df)
        pivot_data = []
        for ts in timestamps:
            if ts in df_pivots.index:
                row = df_pivots.loc[ts]
                for level in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']:
                    if level in row and not pd.isna(row[level]):
                        pivot_data.append((
                            ticker, timeframe, ts, level, float(row[level])
                        ))
        
        if pivot_data:
            logger.debug(f"Saving {len(pivot_data)} pivot points")
            save_pivots_batch(ticker, timeframe, pivot_data)
            
        # Calculate Chandelier Exit
        logger.debug("Calculating Chandelier Exit...")
        df_ce = add_chandelier_exit(df)
        ce_data = []
        for ts in timestamps:
            if ts in df_ce.index:
                row = df_ce.loc[ts]
                if not pd.isna(row['ce_long_stop']):
                    ce_data.append((
                        ticker, timeframe, ts,
                        22,  # Default period
                        3.0,  # Default multiplier
                        float(row['ce_long_stop']),
                        float(row['ce_short_stop']),
                        int(row['ce_direction']),
                        int(row['ce_signal'])
                    ))
        
        if ce_data:
            logger.debug(f"Saving {len(ce_data)} Chandelier Exit values")
            save_chandelier_batch(ticker, timeframe, ce_data)
            
        # Calculate OBV
        logger.debug("Calculating OBV...")
        df_obv = add_obv(df, ma_type='SMA + BB')
        obv_data = []
        for ts in timestamps:
            if ts in df_obv.index:
                row = df_obv.loc[ts]
                if not pd.isna(row['obv']):
                    obv_data.append((
                        ticker, timeframe, ts,
                        float(row['obv']),
                        float(row['obv_ma']) if 'obv_ma' in row and not pd.isna(row['obv_ma']) else None,
                        float(row['obv_bb_upper']) if 'obv_bb_upper' in row and not pd.isna(row['obv_bb_upper']) else None,
                        float(row['obv_bb_lower']) if 'obv_bb_lower' in row and not pd.isna(row['obv_bb_lower']) else None
                    ))
        
        if obv_data:
            logger.debug(f"Saving {len(obv_data)} OBV values")
            save_obv_batch(ticker, timeframe, obv_data)
            
        logger.info(f"Successfully updated all indicators for {ticker} {timeframe}")
            
    except Exception as e:
        logger.error(f"Failed to update indicators: {str(e)}")
        raise

    
def update_ohlc(ticker: str = None, timeframe: str = None, days: int = 7) -> None:
    """
    Update OHLC data for specified ticker and timeframe.
    
    Args:
        ticker: Optional ticker symbol (default: update all tickers)
        timeframe: Optional timeframe (default: update all timeframes)
        days: Number of days to look back (default: 7)
    """
    try:
        logger.info("Starting OHLC update")
        
        # Filter tickers and timeframes if specified
        tickers = [ticker] if ticker else TICKERS
        timeframes = [timeframe] if timeframe else list(TIMEFRAMES.values())
        
        logger.debug(f"Updating tickers: {', '.join(tickers)}")
        logger.debug(f"Updating timeframes: {', '.join(timeframes)}")
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        with db_cursor() as cursor:
            for ticker in tickers:
                for tf in timeframes:
                    try:
                        # Get ticker record or create if doesn't exist
                        get_ticker(cursor, ticker)
                        
                        # Fetch and save OHLC data
                        logger.debug(f"Fetching {ticker} {tf} data from {start_time} to {end_time}")
                        candles = fetch_binance_ohlc(ticker, tf, start_time, end_time)
                        
                        if candles:
                            # Save OHLC data and get timestamps for indicator updates
                            timestamps = save_to_db(candles, ticker, tf)
                            
                            if timestamps:
                                # Get enough historical data for indicator calculations
                                history_start = min(timestamps) - timedelta(days=30)  # Get extra history for accuracy
                                historical_data = fetch_binance_ohlc(ticker, tf, history_start, end_time)
                                
                                if historical_data:
                                    # Convert to DataFrame for indicator calculations
                                    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                                    df.set_index('timestamp', inplace=True)
                                    
                                    # Update indicators
                                    update_indicators(ticker, tf, timestamps, logger, df)
                                    
                            logger.info(f"Successfully updated {ticker} {tf}")
                        else:
                            logger.info(f"No new data for {ticker} {tf}")
                            
                    except Exception as e:
                        logger.error(f"Error updating {ticker} {tf}: {str(e)}")
                        continue
                        
        logger.info("OHLC update completed")
        
    except Exception as e:
        logger.error(f"Failed to update OHLC data: {str(e)}")
        raise
