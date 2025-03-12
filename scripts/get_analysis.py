"""
Script to get and analyze market data with indicators.
Run with: python -m scripts.get_analysis [options]

Examples:
    # Get last 7 days of BTC/USDT hourly data
    python -m scripts.get_analysis --ticker BTCUSDT --timeframe 1H --days 7

    # Get 30 days of ETH/USDT 4-hour data with debug info
    python -m scripts.get_analysis --ticker ETHUSDT --timeframe 4H --days 30 --debug

    # Save analysis to custom file
    python -m scripts.get_analysis --ticker BTCUSDT --output btc_analysis.csv

Available Tickers:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    (and other major cryptocurrency pairs)

Available Timeframes:
    - 1H  (1 hour)
    - 4H  (4 hours)
    - 1D  (1 day)

Options:
    --ticker     Trading pair to analyze (default: BTCUSDT)
    --timeframe  Candle timeframe (default: 1H)
    --days       Number of days to look back (default: 7)
    --output     Output CSV file path (default: data.csv)
    --debug      Enable debug logging
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
from libs.market_data.get_data import get_market_data
from libs.market_data.fetch_ohlc import TICKERS, TIMEFRAMES
from libs.utils.logging import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and analyze market data')
    parser.add_argument('--ticker', type=str, default='BTCUSDT',
                       help='Ticker symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1H',
                       help='Timeframe (default: 1H)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to fetch (default: 7)')
    parser.add_argument('--output', type=str, default='data.csv',
                       help='Output file (default: data.csv)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    logger.info(f"Fetching {args.days} days of data for {args.ticker} {args.timeframe}")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)
    
    try:
        # Fetch data
        df = get_market_data(
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save to file
        df.to_csv(args.output)
        logger.info(f"Data saved to {args.output}")
        
        # Display summary
        logger.info("\nData Summary:")
        logger.info(f"Timeframe: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Number of records: {len(df)}")
        logger.info("\nColumns:")
        for col in df.columns:
            logger.info(f"- {col}")
        
    except Exception as e:
        logger.error(f"Error getting data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 