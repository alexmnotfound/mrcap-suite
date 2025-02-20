import psycopg2
from contextlib import contextmanager
from .config import DatabaseConfig

db_config = DatabaseConfig()

def get_db_connection():
    """Returns a database connection."""
    return psycopg2.connect(db_config.connection_string)

@contextmanager
def db_cursor():
    """Context manager for database operations."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
