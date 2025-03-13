"""
Script to display market data for a specific day or date range.
Run with: python -m scripts.get_market_data [options]

Examples:
    # Show today's BTC data
    python -m scripts.get_market_data

    # Show specific date
    python -m scripts.get_market_data --date 2025-02-16 --ticker BTCUSDT --timeframe 1h
    
    # Show date range
    python -m scripts.get_market_data --start 2025-02-15 --end 2025-02-18 --timeframe 1D

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
    --date       Single date to show (YYYY-MM-DD) (default: today)
    --start      Start date for range (YYYY-MM-DD)
    --end        End date for range (YYYY-MM-DD)
    --ticker     Trading pair to analyze (default: BTCUSDT)
    --timeframe  Candle timeframe (default: 1D)

Output Information:
    For each candle, displays:
    - Basic Data: Open, High, Low, Close, Volume
    - Pattern: If a candlestick pattern is detected
    - EMAs: Various exponential moving averages
    - RSI: Value, Slope, and Divergence
    - Monthly Pivot Points: PP, R1-R5, S1-S5
    - Chandelier Exit: Long/Short stops and signals
    - OBV (On Balance Volume): Value, MA, Bollinger Bands
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
from libs.market_data.get_data import get_market_data
from libs.utils.logging import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Get market data for a specific day or date range')
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--date', type=str,
                          help='Single date to show (YYYY-MM-DD) (default: today)')
    date_group.add_argument('--start', type=str,
                          help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD) (default: today if start provided)')
    parser.add_argument('--ticker', type=str, default='BTCUSDT',
                       help='Ticker symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1D',
                       help='Timeframe (1h/4H/1D) (default: 1D)')
    return parser.parse_args()

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    return datetime.strptime(date_str, '%Y-%m-%d')

def format_value(value):
    """Format numeric values for display."""
    if pd.isna(value):
        return '-'
    if isinstance(value, (int, float)):
        if abs(value) > 1000:
            return f"{value:,.2f}"
        return f"{value:.2f}"
    return str(value)

def print_candle_data(timestamp, row):
    """Print data for a single candle."""
    print(f"\nTime: {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"Open:   {format_value(row['open'])}")
    print(f"High:   {format_value(row['high'])}")
    print(f"Low:    {format_value(row['low'])}")
    print(f"Close:  {format_value(row['close'])}")
    print(f"Volume: {format_value(row['volume'])}")
    if 'candle_pattern' in row:
        print(f"Pattern: {row['candle_pattern']}")
    
    # EMAs
    ema_cols = [col for col in row.index if col.startswith('ema_')]
    if ema_cols:
        print("\nEMAs:")
        for col in ema_cols:
            period = col.split('_')[1]
            print(f"EMA-{period}: {format_value(row[col])}")
    
    # RSI
    if 'rsi' in row:
        print("\nRSI:")
        print(f"Value:      {format_value(row['rsi'])}")
        print(f"Slope:      {format_value(row['rsi_slope'])}")
        print(f"Divergence: {format_value(row['rsi_div'])}")
    
    # Pivot Points
    pivot_cols = {
        #'Daily': ['PP', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5'],
        'Monthly': ['M_PP', 'M_R1', 'M_R2', 'M_R3', 'M_R4', 'M_R5', 'M_S1', 'M_S2', 'M_S3', 'M_S4', 'M_S5']
    }
    
    for period, levels in pivot_cols.items():
        if any(col in row.index for col in levels):
            print(f"\n{period} Pivot Points:")
            for col in levels:
                if col in row.index:
                    print(f"{col.replace('M_', '')}: {format_value(row[col])}")
    
    # Chandelier Exit
    if 'ce_long_stop' in row:
        print("\nChandelier Exit:")
        print(f"Long Stop:  {format_value(row['ce_long_stop'])}")
        print(f"Short Stop: {format_value(row['ce_short_stop'])}")
        print(f"Direction:  {int(row['ce_direction'])}")
        print(f"Signal:     {int(row['ce_signal'])}")
    
    # OBV
    if 'obv' in row:
        print("\nOn Balance Volume:")
        print(f"OBV:    {format_value(row['obv'])}")
        if 'obv_ma' in row:
            print(f"MA:     {format_value(row['obv_ma'])}")
        if 'obv_bb_upper' in row and 'obv_bb_lower' in row:
            print(f"BB Upper: {format_value(row['obv_bb_upper'])}")
            print(f"BB Lower: {format_value(row['obv_bb_lower'])}")
    
    print("-" * 80)

def validate_dates(start_time: datetime, end_time: datetime) -> tuple[datetime, datetime]:
    """
    Validate and adjust date range.
    
    Args:
        start_time: Requested start datetime
        end_time: Requested end datetime
        
    Returns:
        Tuple of validated (start_time, end_time)
    """
    now = datetime.now()
    
    # Don't allow future dates
    if start_time.date() > now.date():
        raise ValueError(f"Start date {start_time.date()} is in the future")
    
    if end_time.date() > now.date():
        end_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Ensure start is before end
    if start_time >= end_time:
        raise ValueError(f"Start date {start_time.date()} must be before end date {end_time.date()}")
    
    return start_time, end_time

def main():
    args = parse_args()
    logger = setup_logging()
    
    try:
        # Handle date arguments
        if args.date:
            start_time = parse_date(args.date)
            end_time = start_time + timedelta(days=1)
        elif args.start:
            start_time = parse_date(args.start)
            if args.end:
                end_time = parse_date(args.end) + timedelta(days=1)
            else:
                end_time = datetime.now()
        else:
            # Default to today
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
        
        # Ensure start time is at beginning of day
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Validate dates
        start_time, end_time = validate_dates(start_time, end_time)
        
        df = get_market_data(
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            logger.info(f"No data found for {args.ticker} between {start_time.date()} and {(end_time - timedelta(days=1)).date()}")
            return
        
        date_range = f"from {start_time.date()} to {(end_time - timedelta(days=1)).date()}"
        if start_time.date() == (end_time - timedelta(days=1)).date():
            date_range = f"for {start_time.date()}"
            
        print(f"\n=== {args.ticker} {args.timeframe} Data {date_range} ===")
        print("=" * 80)
        
        if args.timeframe in ['1h', '4H']:
            # For intraday timeframes, show all candles
            for timestamp, row in df.iterrows():
                print_candle_data(timestamp, row)
        else:
            # For daily timeframe, show each day's candle
            for timestamp, row in df.iterrows():
                print_candle_data(timestamp, row)
        
    except ValueError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 