"""
db_connect.py - PostgreSQL Database Connection Handler
Manages all database connections to the xauusd PostgreSQL database.
"""

import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# Load environment variables from .env file
load_dotenv()


class DatabaseConnection:
    """
    Handles PostgreSQL database connections and queries.
    All data for the ML engine comes directly from this connection.
    """

    def __init__(self):
        """Initialize database connection parameters from environment variables."""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'XAUUSD')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', '')
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None
        self.engine = None  # SQLAlchemy engine for pandas

    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database.
        Returns True if connection successful, False otherwise.
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Create SQLAlchemy engine for pandas compatibility
            # URL-encode password to handle special characters
            encoded_password = quote_plus(self.password)
            db_url = f"postgresql+psycopg2://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(db_url)
            
            # Use ASCII to avoid Windows cp1252 console encoding errors.
            print(f"[OK] Connected to PostgreSQL database: {self.database}")
            return True
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close database connection and cursor."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("[OK] Database connection closed")

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries.

        Args:
            query: SQL query string
            params: Optional tuple of query parameters

        Returns:
            List of dictionaries containing query results
        """
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            print(f"✗ Query execution failed: {e}")
            return []

    def fetch_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame.
        This is the primary method for pulling training data.

        Args:
            query: SQL query string
            params: Optional tuple of query parameters

        Returns:
            pandas DataFrame with query results
        """
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            return df
        except Exception as e:
            print(f"✗ Failed to fetch DataFrame: {e}")
            return pd.DataFrame()

    def get_available_timeframes(self) -> List[str]:
        """Get list of all available timeframes in the database."""
        query = "SELECT DISTINCT timeframe FROM xauusd_ohlcv WHERE 1=1 ORDER BY timeframe"
        results = self.execute_query(query)
        return [row['timeframe'] for row in results]

    def get_date_range(self, timeframe: str = None) -> Dict[str, str]:
        """
        Get the date range of available data.

        Args:
            timeframe: Optional timeframe filter

        Returns:
            Dictionary with 'min_date' and 'max_date'
        """
        if timeframe:
            query = """
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM xauusd_ohlcv
                WHERE timeframe = %s
            """
            results = self.execute_query(query, (timeframe,))
        else:
            query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM xauusd_ohlcv WHERE 1=1"
            results = self.execute_query(query)

        if results:
            return {
                'min_date': str(results[0]['min_date']),
                'max_date': str(results[0]['max_date'])
            }
        return {'min_date': None, 'max_date': None}

    def get_record_count(self, timeframe: str = None) -> int:
        """Get total number of records, optionally filtered by timeframe."""
        if timeframe:
            query = "SELECT COUNT(*) as count FROM xauusd_ohlcv WHERE timeframe = %s"
            results = self.execute_query(query, (timeframe,))
        else:
            query = "SELECT COUNT(*) as count FROM xauusd_ohlcv WHERE 1=1"
            results = self.execute_query(query)

        return results[0]['count'] if results else 0

    def fetch_ohlcv_data(
        self,
        timeframe: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from the database with optional filters.

        Args:
            timeframe: Filter by specific timeframe (e.g., '1min', '1H', '1D')
            start_date: Filter by start date (YYYY-MM-DD format)
            end_date: Filter by end date (YYYY-MM-DD format)
            limit: Maximum number of records to return

        Returns:
            pandas DataFrame with OHLCV data
        """
        query_parts = ["SELECT * FROM xauusd_ohlcv WHERE 1=1"]
        params = []

        if timeframe:
            query_parts.append("AND timeframe = %s")
            params.append(timeframe)

        if start_date:
            query_parts.append("AND date >= %s")
            params.append(start_date)

        if end_date:
            query_parts.append("AND date <= %s")
            params.append(end_date)

        query_parts.append("ORDER BY timestamp ASC")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)
        return self.fetch_dataframe(query, tuple(params) if params else None)

    def get_direction_counts_by_day(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get count of buy/sell candles per day for a given timeframe.

        Args:
            timeframe: Timeframe to analyze
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with date, buy_count, sell_count columns
        """
        query_parts = ["""
            SELECT
                date,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count,
                COUNT(*) as total_count
            FROM xauusd_ohlcv
            WHERE timeframe = %s
        """]
        params = [timeframe]

        if start_date:
            query_parts.append("AND date >= %s")
            params.append(start_date)

        if end_date:
            query_parts.append("AND date <= %s")
            params.append(end_date)

        query_parts.append("GROUP BY date ORDER BY date")
        query = " ".join(query_parts)

        return self.fetch_dataframe(query, tuple(params))

    def get_direction_counts_by_hour(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Get count of buy/sell candles per hour."""
        query_parts = ["""
            SELECT
                hour,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count,
                COUNT(*) as total_count
            FROM xauusd_ohlcv
            WHERE timeframe = %s
        """]
        params = [timeframe]

        if start_date:
            query_parts.append("AND date >= %s")
            params.append(start_date)

        if end_date:
            query_parts.append("AND date <= %s")
            params.append(end_date)

        query_parts.append("GROUP BY hour ORDER BY hour")
        query = " ".join(query_parts)

        return self.fetch_dataframe(query, tuple(params))

    def get_direction_counts_by_day_of_week(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Get count of buy/sell candles per day of week."""
        query_parts = ["""
            SELECT
                day_of_week,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count,
                COUNT(*) as total_count
            FROM xauusd_ohlcv
            WHERE timeframe = %s
        """]
        params = [timeframe]

        if start_date:
            query_parts.append("AND date >= %s")
            params.append(start_date)

        if end_date:
            query_parts.append("AND date <= %s")
            params.append(end_date)

        query_parts.append("GROUP BY day_of_week ORDER BY day_of_week")
        query = " ".join(query_parts)

        return self.fetch_dataframe(query, tuple(params))

    def get_direction_counts_by_month(
        self,
        timeframe: str,
        year: int = None
    ) -> pd.DataFrame:
        """Get count of buy/sell candles per month."""
        query_parts = ["""
            SELECT
                year,
                month,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count,
                COUNT(*) as total_count
            FROM xauusd_ohlcv
            WHERE timeframe = %s
        """]
        params = [timeframe]

        if year:
            query_parts.append("AND year = %s")
            params.append(year)

        query_parts.append("GROUP BY year, month ORDER BY year, month")
        query = " ".join(query_parts)

        return self.fetch_dataframe(query, tuple(params))


# Singleton instance for easy import
db = DatabaseConnection()


def get_connection() -> DatabaseConnection:
    """Get the database connection singleton."""
    return db


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")

    if db.connect():
        print(f"\nAvailable timeframes: {db.get_available_timeframes()}")
        print(f"Date range: {db.get_date_range()}")
        print(f"Total records: {db.get_record_count()}")

        # Test fetching some data
        sample_data = db.fetch_ohlcv_data(timeframe='1min', limit=5)
        if not sample_data.empty:
            print(f"\nSample data (5 rows):")
            print(sample_data.head())

        db.disconnect()
    else:
        print("Failed to connect to database. Check your .env configuration.")
