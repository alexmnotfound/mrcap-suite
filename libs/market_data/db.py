import psycopg2
from contextlib import contextmanager
from .config import DatabaseConfig
from typing import List, Tuple
import psycopg2.extras

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

def batch_insert(table: str, columns: List[str], values: List[Tuple], 
                conflict_action: str = None) -> int:
    """
    Perform efficient batch insert using executemany.
    
    Args:
        table: Table name
        columns: List of column names
        values: List of value tuples
        conflict_action: Optional ON CONFLICT action
        
    Returns:
        Number of rows inserted/updated
    """
    if not values:
        return 0
        
    placeholders = ','.join(['%s'] * len(columns))
    query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    
    if conflict_action:
        query += f" ON CONFLICT {conflict_action}"
    
    with db_cursor() as cursor:
        try:
            # Use psycopg2.extras.execute_batch for better performance
            psycopg2.extras.execute_batch(
                cursor,
                query,
                values,
                page_size=1000  # Adjust batch size based on your needs
            )
            return len(values)
        except Exception as e:
            raise Exception(f"Batch insert failed: {str(e)}")
