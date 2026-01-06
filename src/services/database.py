"""
Database Manager Module.

This module contains the DatabaseManager class for handling all SQLite
database operations for the California housing prediction project.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATABASE_PATH

class DatabaseManager:
    """
    Manages SQLite database operations for housing data.

    This class provides methods for:
    - Creating and managing database connection
    - Creating tables with appropriate schemas
    - CRUD operations (Create, Read, Update, Delete)
    - Filtering with WHERE clauses
    - Aggregations with GROUP BY
    - Joining tables with INNER JOIN

    Attributes:
        db_path (str or Path): Path to SQLite database file
        conn (sqlite3.Connection): Database connection object
    """

    def __init__(self, db_path: Union[str, Path] = None):
        """
        Initialize database manager.

        Args:
            db_path (str or Path, optional): Path to database file.
                If None, uses DATABASE_PATH from config.
        """
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.conn: Optional[sqlite3.Connection] = None

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Establish connection
        self.create_connection()

    def create_connection(self) -> sqlite3.Connection:
        """
        Create database connection.

        Returns:
            sqlite3.Connection: Active database connection

        Raises:
            Exception: If connection fails
        """
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            print(f"Database connection established: {self.db_path}")
            return self.conn

        except sqlite3.Error as e:
            raise Exception(f"Failed to connect to database: {str(e)}")

    def close_connection(self):
        """Close database connection if it exists."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def create_tables(self):
        """
        Create housing and district_summary tables if they don't exist.

        The housing table stores all individual housing records.
        The district_summary table stores aggregated statistics by income category.

        Raises:
            Exception: If table creation fails
        """
        try:
            cursor = self.conn.cursor()

            # Create housing table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS housing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    longitude REAL NOT NULL,
                    latitude REAL NOT NULL,
                    housing_median_age REAL NOT NULL,
                    total_rooms REAL NOT NULL,
                    total_bedrooms REAL NOT NULL,
                    population REAL NOT NULL,
                    households REAL NOT NULL,
                    median_income REAL NOT NULL,
                    median_house_value REAL NOT NULL,
                    rooms_per_household REAL,
                    bedrooms_per_room REAL,
                    population_per_household REAL,
                    income_category TEXT,
                    age_category TEXT
                )
            ''')

            # Create district_summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS district_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    income_category TEXT NOT NULL UNIQUE,
                    avg_house_value REAL,
                    avg_rooms REAL,
                    avg_age REAL,
                    district_count INTEGER
                )
            ''')

            self.conn.commit()
            print("Tables created successfully")

        except sqlite3.Error as e:
            raise Exception(f"Failed to create tables: {str(e)}")

    def insert_data(self, data: pd.DataFrame, table_name: str = 'housing') -> int:
        """
        Insert DataFrame into database table.

        Args:
            data (pd.DataFrame): Data to insert
            table_name (str): Target table name (default: 'housing')

        Returns:
            int: Number of rows inserted

        Raises:
            ValueError: If data is None or empty
            Exception: If insertion fails
        """
        if data is None or len(data) == 0:
            raise ValueError("No data to insert")

        try:
            # Use pandas to_sql for efficient insertion
            rows_before = self.get_table_count(table_name)

            data.to_sql(table_name, self.conn, if_exists='replace', index=False)

            rows_after = self.get_table_count(table_name)
            rows_inserted = rows_after - rows_before

            print(f"Inserted {rows_after} rows into {table_name} table")
            return rows_after

        except Exception as e:
            raise Exception(f"Failed to insert data: {str(e)}")

    def fetch_all(self, table_name: str = 'housing') -> pd.DataFrame:
        """
        Fetch all records from table.

        Args:
            table_name (str): Table to query (default: 'housing')

        Returns:
            pd.DataFrame: All records from the table

        Raises:
            Exception: If query fails
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.conn)
            print(f"Fetched {len(df)} rows from {table_name}")
            return df

        except Exception as e:
            raise Exception(f"Failed to fetch data: {str(e)}")

    def filter_by_income(self, min_income: float, max_income: float) -> pd.DataFrame:
        """
        Filter housing records by income range (WHERE clause demonstration).

        Args:
            min_income (float): Minimum median income
            max_income (float): Maximum median income

        Returns:
            pd.DataFrame: Filtered records

        Raises:
            Exception: If query fails
        """
        try:
            query = """
                SELECT *
                FROM housing
                WHERE median_income >= ? AND median_income <= ?
            """
            df = pd.read_sql_query(query, self.conn, params=(min_income, max_income))
            print(f"Filtered by income ({min_income} to {max_income}): {len(df)} rows")
            return df

        except Exception as e:
            raise Exception(f"Failed to filter by income: {str(e)}")

    def filter_by_location(self, min_lat: float, max_lat: float,
                          min_lon: float, max_lon: float) -> pd.DataFrame:
        """
        Filter records by geographic boundaries (WHERE with AND demonstration).

        Args:
            min_lat (float): Minimum latitude
            max_lat (float): Maximum latitude
            min_lon (float): Minimum longitude
            max_lon (float): Maximum longitude

        Returns:
            pd.DataFrame: Filtered records

        Raises:
            Exception: If query fails
        """
        try:
            query = """
                SELECT *
                FROM housing
                WHERE latitude >= ? AND latitude <= ?
                  AND longitude >= ? AND longitude <= ?
            """
            params = (min_lat, max_lat, min_lon, max_lon)
            df = pd.read_sql_query(query, self.conn, params=params)
            print(f"Filtered by location: {len(df)} rows")
            return df

        except Exception as e:
            raise Exception(f"Failed to filter by location: {str(e)}")

    def aggregate_by_income_category(self) -> pd.DataFrame:
        """
        Aggregate statistics grouped by income category (GROUP BY demonstration).

        Calculates average house value, average rooms per household,
        average housing age, and count of districts for each income category.

        Returns:
            pd.DataFrame: Aggregated statistics with columns:
                - income_category
                - avg_house_value
                - avg_rooms_per_household
                - avg_age
                - count_districts

        Raises:
            Exception: If query fails
        """
        try:
            query = """
                SELECT
                    income_category,
                    AVG(median_house_value) as avg_house_value,
                    AVG(rooms_per_household) as avg_rooms_per_household,
                    AVG(housing_median_age) as avg_age,
                    COUNT(*) as count_districts
                FROM housing
                WHERE income_category IS NOT NULL
                GROUP BY income_category
                ORDER BY avg_house_value DESC
            """
            df = pd.read_sql_query(query, self.conn)
            print(f"Aggregated by income category: {len(df)} categories")
            return df

        except Exception as e:
            raise Exception(f"Failed to aggregate by income category: {str(e)}")

    def populate_district_summary(self) -> int:
        """
        Populate district_summary table with aggregated data from housing table.

        This method performs a GROUP BY aggregation on the housing table
        and inserts the results into the district_summary table.

        Returns:
            int: Number of rows inserted into district_summary

        Raises:
            Exception: If operation fails
        """
        try:
            # Get aggregated data
            agg_data = self.aggregate_by_income_category()

            # Rename columns to match district_summary schema
            agg_data = agg_data.rename(columns={
                'avg_rooms_per_household': 'avg_rooms',
                'count_districts': 'district_count'
            })

            # Select only the columns we need
            agg_data = agg_data[[
                'income_category', 'avg_house_value',
                'avg_rooms', 'avg_age', 'district_count'
            ]]

            # Insert into district_summary table
            agg_data.to_sql('district_summary', self.conn, if_exists='replace', index=False)

            print(f"Populated district_summary table with {len(agg_data)} rows")
            return len(agg_data)

        except Exception as e:
            raise Exception(f"Failed to populate district_summary: {str(e)}")

    def join_housing_with_summary(self, limit: int = 100) -> pd.DataFrame:
        """
        Join housing table with district_summary table (INNER JOIN demonstration).

        This shows individual housing records alongside their district averages,
        demonstrating SQL JOIN capability.

        Args:
            limit (int): Maximum number of rows to return (default: 100)

        Returns:
            pd.DataFrame: Joined data showing individual records with district averages

        Raises:
            Exception: If query fails
        """
        try:
            query = """
                SELECT
                    h.ROWID as id,
                    h.longitude,
                    h.latitude,
                    h.median_income,
                    h.median_house_value,
                    h.income_category,
                    h.rooms_per_household,
                    ds.avg_house_value as district_avg_value,
                    ds.avg_rooms as district_avg_rooms,
                    ds.district_count
                FROM housing h
                INNER JOIN district_summary ds
                    ON h.income_category = ds.income_category
                LIMIT ?
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            print(f"Joined housing with district_summary: {len(df)} rows")
            return df

        except Exception as e:
            raise Exception(f"Failed to join tables: {str(e)}")

    def update_income_category(self, housing_id: int, new_category: str) -> bool:
        """
        Update income category for specific record (UPDATE demonstration).

        Args:
            housing_id (int): Record ID to update
            new_category (str): New income category value

        Returns:
            bool: True if successful

        Raises:
            Exception: If update fails
        """
        try:
            cursor = self.conn.cursor()
            query = """
                UPDATE housing
                SET income_category = ?
                WHERE id = ?
            """
            cursor.execute(query, (new_category, housing_id))
            self.conn.commit()

            if cursor.rowcount > 0:
                print(f"Updated record {housing_id} with new category: {new_category}")
                return True
            else:
                print(f"No record found with id: {housing_id}")
                return False

        except sqlite3.Error as e:
            raise Exception(f"Failed to update record: {str(e)}")

    def delete_by_id(self, housing_id: int) -> bool:
        """
        Delete record by ID (DELETE demonstration).

        Args:
            housing_id (int): Record ID to delete

        Returns:
            bool: True if successful

        Raises:
            Exception: If deletion fails
        """
        try:
            cursor = self.conn.cursor()
            query = "DELETE FROM housing WHERE id = ?"
            cursor.execute(query, (housing_id,))
            self.conn.commit()

            if cursor.rowcount > 0:
                print(f"Deleted record with id: {housing_id}")
                return True
            else:
                print(f"No record found with id: {housing_id}")
                return False

        except sqlite3.Error as e:
            raise Exception(f"Failed to delete record: {str(e)}")

    def execute_custom_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute custom SQL query.

        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters for parameterized queries

        Returns:
            pd.DataFrame: Query results

        Raises:
            Exception: If query fails
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            print(f"Custom query executed: {len(df)} rows returned")
            return df

        except Exception as e:
            raise Exception(f"Failed to execute custom query: {str(e)}")

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            dict: Contains:
                - housing_count: Number of records in housing table
                - summary_count: Number of records in district_summary table
                - database_size: Size of database file in MB
                - tables: List of all tables in the database
        """
        try:
            cursor = self.conn.cursor()

            # Get table counts
            stats = {
                'housing_count': self.get_table_count('housing'),
                'summary_count': self.get_table_count('district_summary'),
                'database_size': self.db_path.stat().st_size / 1024**2,  # MB
            }

            # Get list of all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            stats['tables'] = [row[0] for row in cursor.fetchall()]

            return stats

        except Exception as e:
            raise Exception(f"Failed to get statistics: {str(e)}")

    def get_table_count(self, table_name: str) -> int:
        """
        Get count of rows in a table.

        Args:
            table_name (str): Name of the table

        Returns:
            int: Number of rows in the table
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            return count

        except sqlite3.Error:
            # Table might not exist yet
            return 0

    def verify_connection(self) -> bool:
        """
        Verify that database connection is active and working.

        Returns:
            bool: True if connection is valid

        Raises:
            Exception: If connection is not valid
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            if result[0] == 1:
                print("Database connection verified")
                return True
            else:
                raise Exception("Connection verification failed")

        except sqlite3.Error as e:
            raise Exception(f"Database connection error: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close_connection()
