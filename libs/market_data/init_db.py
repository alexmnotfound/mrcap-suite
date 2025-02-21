from .db import db_cursor
import logging

logger = logging.getLogger(__name__)

def create_tables():
    """Create necessary database tables if they don't exist."""
    with db_cursor() as cursor:
        # Create tickers table with ticker as primary key
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create OHLC data table using ticker directly
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlc_data (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC(18, 8) NOT NULL,
            high NUMERIC(18, 8) NOT NULL,
            low NUMERIC(18, 8) NOT NULL,
            close NUMERIC(18, 8) NOT NULL,
            volume NUMERIC(18, 8) NOT NULL,
            PRIMARY KEY(ticker, timeframe, timestamp),
            CONSTRAINT ohlc_data_timeframe_check 
                CHECK (timeframe IN ('1H', '4H', '1D', '1M'))
        );
        """)

        # Create indexes for better query performance
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlc_timestamp 
        ON ohlc_data (timestamp);
        """)

        # EMAs table using ticker directly
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emas (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            period INTEGER NOT NULL,
            value NUMERIC(18, 8) NOT NULL,
            PRIMARY KEY(ticker, timeframe, timestamp, period)
        );

        CREATE INDEX IF NOT EXISTS idx_emas_ticker_timeframe_period 
        ON emas (ticker, timeframe, period);
        """)

        # Create RSI table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rsi (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            period INTEGER NOT NULL,
            value NUMERIC(18, 8) NOT NULL,
            slope NUMERIC(18, 8) NULL,
            divergence NUMERIC(18, 8) NULL,
            PRIMARY KEY(ticker, timeframe, timestamp, period)
        );
        
        CREATE INDEX IF NOT EXISTS idx_rsi_ticker_timeframe_period 
        ON rsi (ticker, timeframe, period);
        """)

        # Create Pivot Points table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pivots (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            level TEXT NOT NULL,
            value NUMERIC(18, 8) NOT NULL,
            PRIMARY KEY(ticker, timeframe, timestamp, level)
        );
        
        CREATE INDEX IF NOT EXISTS idx_pivots_ticker_timeframe 
        ON pivots (ticker, timeframe);
        """)

        # Create Chandelier Exit table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chandelier_exit (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            period INTEGER NOT NULL,
            multiplier NUMERIC(8, 2) NOT NULL,
            long_stop NUMERIC(18, 8) NOT NULL,
            short_stop NUMERIC(18, 8) NOT NULL,
            direction INTEGER NOT NULL,
            signal INTEGER NOT NULL,
            PRIMARY KEY(ticker, timeframe, timestamp, period, multiplier)
        );

        CREATE INDEX IF NOT EXISTS idx_chandelier_ticker_timeframe 
        ON chandelier_exit (ticker, timeframe);
        """)

        # Create OBV table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS obv (
            ticker TEXT REFERENCES tickers(ticker) ON DELETE CASCADE,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value NUMERIC(18, 8) NOT NULL,
            ma_value NUMERIC(18, 8),
            bb_upper NUMERIC(18, 8),
            bb_lower NUMERIC(18, 8),
            PRIMARY KEY(ticker, timeframe, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_obv_ticker_timeframe 
        ON obv (ticker, timeframe);
        """)

        # Add pattern column to OHLC data table
        cursor.execute("""
        ALTER TABLE ohlc_data 
        ADD COLUMN IF NOT EXISTS candle_pattern TEXT;
        """)

def truncate_tables():
    """Truncate all tables in the database."""
    with db_cursor() as cursor:
        # Disable foreign key checks while truncating
        cursor.execute("""
        DO $$ 
        BEGIN
            -- Truncate all tables in a single transaction
            TRUNCATE TABLE 
                ohlc_data,
                emas,
                rsi,
                pivots,
                chandelier_exit,
                obv,
                tickers
            CASCADE;
        END $$;
        """)
        logger.info("All tables truncated successfully")

def init_db():
    """Initialize the database with required tables and indexes."""
    try:
        create_tables()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_db() 