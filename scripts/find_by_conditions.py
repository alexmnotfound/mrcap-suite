"""
Script to find market data matching specific conditions.
Run with: python -m scripts.find_by_conditions [options]

Examples:
    # Find all Doji patterns in the last 7 days
    python -m scripts.find_by_conditions --pattern 'Doji' --days 7

    # Find Hammer patterns with RSI conditions
    python -m scripts.find_by_conditions --pattern 'Hammer' --rsi-min 30 --rsi-max 70

    # Find patterns in a specific date range
    python -m scripts.find_by_conditions --pattern 'Shooting Star' --start 2024-01-01 --end 2024-01-31

    # Find patterns with minimum volume
    python -m scripts.find_by_conditions --pattern 'Bullish Marubozu' --volume-min 1000

    # Find patterns for a specific timeframe
    python -m scripts.find_by_conditions --pattern 'Doji' --timeframe '4H'

Available Patterns:
    - Doji
    - Hammer
    - Hanging Man
    - Shooting Star
    - Inverted Hammer
    - Bullish Marubozu
    - Bearish Marubozu

Conditions:
    --rsi-min        Minimum RSI value
    --rsi-max        Maximum RSI value
    --volume-min     Minimum volume
    --timeframe      Candle timeframe (1H/4H/1D)
    --days          Number of days to look back
    --start         Start date (YYYY-MM-DD)
    --end           End date (YYYY-MM-DD)
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
from libs.market_data.fetch_data import fetch_market_data
from libs.utils.logging import setup_logging
import logging

def find_patterns(df: pd.DataFrame, pattern_type: str = None, rsi_min: float = None, 
                 rsi_max: float = None, volume_min: float = None) -> pd.DataFrame:
    """Find candlestick patterns in the data."""
    # Filter by pattern type if specified
    if pattern_type:
        matches = df[df['candle_pattern'] == pattern_type].copy()
    else:
        matches = df[df['candle_pattern'] != 'Normal'].copy()
    
    # Apply additional filters if present in the data
    if 'rsi' in df.columns and rsi_min is not None:
        matches = matches[matches['rsi'] >= rsi_min]
    if 'rsi' in df.columns and rsi_max is not None:
        matches = matches[matches['rsi'] <= rsi_max]
    if volume_min is not None:
        matches = matches[matches['volume'] >= volume_min]
    
    return matches

def format_value(value):
    """Format numeric values for display."""
    if pd.isna(value):
        return '-'
    if isinstance(value, (int, float)):
        if abs(value) > 1000:
            return f"{value:,.2f}"
        return f"{value:.2f}"
    return str(value)

def print_pattern_match(timestamp, row):
    """Print a single pattern match in a readable format."""
    print(f"\n{'='*80}")
    print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M')} - Pattern: {row['candle_pattern']}")
    print(f"{'-'*80}")
    
    # Price data
    print("Price Data:")
    print(f"  Open:   {format_value(row['open'])}")
    print(f"  High:   {format_value(row['high'])}")
    print(f"  Low:    {format_value(row['low'])}")
    print(f"  Close:  {format_value(row['close'])}")
    print(f"  Volume: {format_value(row['volume'])}")
    
    # Indicators
    if 'rsi' in row:
        print("\nRSI:")
        print(f"  Value: {format_value(row['rsi'])}")
        if 'rsi_slope' in row:
            print(f"  Slope: {format_value(row['rsi_slope'])}")
    
    # EMAs if present
    ema_cols = [col for col in row.index if col.startswith('ema_')]
    if ema_cols:
        print("\nMoving Averages:")
        for col in sorted(ema_cols, key=lambda x: int(x.split('_')[1])):
            period = col.split('_')[1]
            print(f"  EMA-{period}: {format_value(row[col])}")

def main():
    parser = argparse.ArgumentParser(description='Find candlestick patterns with conditions')
    parser.add_argument('--pattern', type=str, help='Candlestick pattern to search for')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1H', help='Timeframe')
    
    # Date range options (mutually exclusive with --days)
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--days', type=int, default=30, help='Days to look back')
    date_group.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), requires --start')
    
    # Filter options
    parser.add_argument('--rsi-min', type=float, help='Minimum RSI value')
    parser.add_argument('--rsi-max', type=float, help='Maximum RSI value')
    parser.add_argument('--volume-min', type=float, help='Minimum volume')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Handle date range
        end_time = datetime.now()
        if args.start:
            start_time = datetime.strptime(args.start, '%Y-%m-%d')
            if args.end:
                end_time = datetime.strptime(args.end, '%Y-%m-%d')
        else:
            start_time = end_time - timedelta(days=args.days)
        
        df = fetch_market_data(
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_time=start_time,
            end_time=end_time,
            include_indicators=True
        )
        
        # Find patterns with filters
        matches = find_patterns(
            df, 
            pattern_type=args.pattern,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max,
            volume_min=args.volume_min
        )
        
        if matches.empty:
            logger.info("No matching patterns found")
            return
        
        # Print summary header
        print(f"\nFound {len(matches)} {args.pattern} patterns for {args.ticker}")
        print(f"Time range: {matches.index[0].strftime('%Y-%m-%d')} to {matches.index[-1].strftime('%Y-%m-%d')}")
        
        # Print each pattern match
        for timestamp, row in matches.iterrows():
            print_pattern_match(timestamp, row)
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("Summary Statistics:")
        print(f"{'-'*80}")
        if 'rsi' in matches.columns:
            print(f"RSI:")
            print(f"  Average: {matches['rsi'].mean():.2f}")
            print(f"  Range:   {matches['rsi'].min():.2f} - {matches['rsi'].max():.2f}")
        
        if 'volume' in matches.columns:
            print(f"\nVolume:")
            print(f"  Average: {format_value(matches['volume'].mean())}")
            print(f"  Range:   {format_value(matches['volume'].min())} - {format_value(matches['volume'].max())}")
        
    except Exception as e:
        logger.error(f"Error finding patterns: {str(e)}")
        raise

if __name__ == "__main__":
    main() 