"""
Housing Data Processor Module.

This module contains the HousingDataProcessor class for loading, cleaning,
and preprocessing California housing data.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from config import (
    DATA_DIR,
    INTERIM_DATA_PATH,
    MISSING_VALUE_STRATEGY,
    OUTLIER_METHOD,
    OUTLIER_THRESHOLD,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from features import apply_all_features


class HousingDataProcessor:
    """
    Handles loading, cleaning, and initial processing of California housing data.

    This class provides methods for:
    - Loading data from sklearn datasets
    - Checking and handling missing values
    - Removing outliers
    - Applying feature engineering
    - Saving and loading processed data

    Attributes:
        data (pd.DataFrame): Raw housing data
        cleaned_data (pd.DataFrame): Data after cleaning
        feature_engineered_data (pd.DataFrame): Data with engineered features
    """

    def __init__(self):
        """Initialize the data processor with empty data attributes."""
        self.data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.feature_engineered_data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load California housing data from sklearn.datasets.

        The data is loaded with feature names and stored as a DataFrame.
        It includes 8 features and 1 target variable.

        Returns:
            pd.DataFrame: Raw housing data with columns:
                - longitude, latitude, housing_median_age
                - total_rooms, total_bedrooms, population
                - households, median_income
                - median_house_value (target)

        Raises:
            Exception: If data loading fails
        """
        try:
            # Try to load from sklearn cache or download with alternative method
            housing = fetch_california_housing(as_frame=True, download_if_missing=True)

            # Combine features and target into single DataFrame
            self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
            self.data['median_house_value'] = housing.target * 100000  # Convert to actual dollars

            # Rename columns to match our naming convention
            self.data.columns = [
                'median_income', 'housing_median_age', 'avg_rooms', 'avg_bedrooms',
                'population', 'avg_occupancy', 'latitude', 'longitude', 'median_house_value'
            ]

            # Convert average values back to totals (multiply by number of households)
            # Note: sklearn dataset provides averages, but we need totals for feature engineering
            # We'll derive households from population and avg_occupancy
            self.data['households'] = (self.data['population'] / self.data['avg_occupancy']).round()
            self.data['total_rooms'] = (self.data['avg_rooms'] * self.data['households']).round()
            self.data['total_bedrooms'] = (self.data['avg_bedrooms'] * self.data['households']).round()

            # Drop the average columns
            self.data = self.data.drop(columns=['avg_rooms', 'avg_bedrooms', 'avg_occupancy'])

            # Reorder columns for clarity
            self.data = self.data[[
                'longitude', 'latitude', 'housing_median_age',
                'total_rooms', 'total_bedrooms', 'population', 'households',
                'median_income', 'median_house_value'
            ]]

            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

            return self.data

        except Exception as e:
            # Fallback: try loading without downloading
            print(f"⚠ Warning: {str(e)}")
            print("Attempting alternative data loading method...")
            try:
                # Try loading from local sklearn cache
                import os

                from sklearn.datasets import fetch_california_housing
                os.environ['SCIKIT_LEARN_DATA'] = str(DATA_DIR / 'sklearn_cache')
                housing = fetch_california_housing(as_frame=True, download_if_missing=False)

                self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
                self.data['median_house_value'] = housing.target * 100000

                # Process as before
                self.data.columns = [
                    'median_income', 'housing_median_age', 'avg_rooms', 'avg_bedrooms',
                    'population', 'avg_occupancy', 'latitude', 'longitude', 'median_house_value'
                ]
                self.data['households'] = (self.data['population'] / self.data['avg_occupancy']).round()
                self.data['total_rooms'] = (self.data['avg_rooms'] * self.data['households']).round()
                self.data['total_bedrooms'] = (self.data['avg_bedrooms'] * self.data['households']).round()
                self.data = self.data.drop(columns=['avg_rooms', 'avg_bedrooms', 'avg_occupancy'])
                self.data = self.data[[
                    'longitude', 'latitude', 'housing_median_age',
                    'total_rooms', 'total_bedrooms', 'population', 'households',
                    'median_income', 'median_house_value'
                ]]
                return self.data
            except:
                # Last resort: download CSV directly
                print("Downloading from alternative source...")
                try:
                    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
                    self.data = pd.read_csv(url)
                    # CSV already has correct column names and values in dollars
                    print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
                    return self.data
                except:
                    raise Exception(f"Failed to load housing data. Download manually from:\n"
                                  f"https://www.kaggle.com/datasets/camnugent/california-housing-prices\n"
                                  f"Save to: {RAW_DATA_PATH}")

    def get_data_info(self) -> dict:
        """
        Get summary statistics about the dataset.

        Returns:
            dict: Contains:
                - shape: tuple (rows, columns)
                - columns: list of column names
                - dtypes: dictionary of column data types
                - missing_values: count of missing values per column
                - memory_usage: memory usage in MB
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }

    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.

        Returns:
            pd.Series: Count of missing values per column, sorted in descending order

        Raises:
            ValueError: If no data is loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            print("No missing values found!")
        else:
            print(f"Missing values found in {len(missing)} columns:")
            print(missing)

        return missing

    def handle_missing_values(self, strategy: str = None, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.

        Args:
            strategy (str, optional): Imputation strategy:
                - 'median': Fill with median (default for numeric)
                - 'mean': Fill with mean
                - 'mode': Fill with most frequent value
                - 'drop': Drop rows with missing values
                If None, uses MISSING_VALUE_STRATEGY from config
            data (pd.DataFrame, optional): Data to process. If None, uses self.data

        Returns:
            pd.DataFrame: Data with missing values handled

        Raises:
            ValueError: If no data is loaded or strategy is invalid
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if strategy is None:
            strategy = MISSING_VALUE_STRATEGY

        data = data.copy()

        if strategy == 'drop':
            data = data.dropna()
            print(f"Dropped rows with missing values. New shape: {data.shape}")

        elif strategy in ['median', 'mean', 'mode']:
            for column in data.columns:
                if data[column].isnull().sum() > 0:
                    if strategy == 'median':
                        fill_value = data[column].median()
                    elif strategy == 'mean':
                        fill_value = data[column].mean()
                    else:  # mode
                        fill_value = data[column].mode()[0]

                    data[column].fillna(fill_value, inplace=True)
                    print(f"Filled {column} missing values with {strategy}: {fill_value:.2f}")

        else:
            raise ValueError(f"Invalid strategy: {strategy}. Use 'median', 'mean', 'mode', or 'drop'")

        self.cleaned_data = data
        return data

    def remove_outliers(self, method: str = None, threshold: float = None,
                       columns: list = None, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Remove outliers from specified columns.

        Args:
            method (str, optional): Method for outlier detection:
                - 'iqr': Interquartile Range method (default)
                - 'zscore': Z-score method
                If None, uses OUTLIER_METHOD from config
            threshold (float, optional): Threshold for outlier detection:
                - For IQR: multiplier (default 1.5)
                - For Z-score: number of standard deviations (default 3.0)
                If None, uses OUTLIER_THRESHOLD from config
            columns (list, optional): Columns to check for outliers.
                If None, checks all numeric columns except target
            data (pd.DataFrame, optional): Data to process. If None, uses self.cleaned_data or self.data

        Returns:
            pd.DataFrame: Data with outliers removed

        Raises:
            ValueError: If no data is loaded or method is invalid
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if method is None:
            method = OUTLIER_METHOD
        if threshold is None:
            threshold = OUTLIER_THRESHOLD

        data = data.copy()
        original_shape = data.shape[0]

        # Select numeric columns (exclude target variable)
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'median_house_value' in columns:
                columns.remove('median_house_value')

        if method == 'iqr':
            for column in columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                data = data[outliers_mask]

        elif method == 'zscore':
            for column in columns:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                data = data[z_scores < threshold]

        else:
            raise ValueError(f"Invalid method: {method}. Use 'iqr' or 'zscore'")

        removed = original_shape - data.shape[0]
        print(f"Removed {removed} outliers ({removed/original_shape*100:.2f}%). New shape: {data.shape}")

        self.cleaned_data = data
        return data

    def apply_feature_engineering(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply feature engineering functions from features.py.

        This creates derived features like:
        - rooms_per_household
        - bedrooms_per_room
        - population_per_household
        - income_category
        - age_category

        Args:
            data (pd.DataFrame, optional): Data to process. If None, uses self.cleaned_data or self.data

        Returns:
            pd.DataFrame: Data with engineered features

        Raises:
            ValueError: If no data is loaded
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        data = apply_all_features(data, include_log=False)

        print(f"Feature engineering completed. New columns: {data.shape[1]}")
        print("Engineered features:", [col for col in data.columns if col not in [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'
        ]])

        self.feature_engineered_data = data
        return data

    def save_data(self, data: pd.DataFrame, stage: str) -> str:
        """
        Save data to appropriate directory based on processing stage.

        Args:
            data (pd.DataFrame): Data to save
            stage (str): Processing stage - 'raw', 'interim', or 'processed'

        Returns:
            str: Path where data was saved

        Raises:
            ValueError: If stage is invalid or data is None
        """
        if data is None:
            raise ValueError("No data to save")

        # Determine file path based on stage
        if stage == 'raw':
            filepath = RAW_DATA_PATH
        elif stage == 'interim':
            filepath = INTERIM_DATA_PATH
        elif stage == 'processed':
            filepath = PROCESSED_DATA_PATH
        else:
            raise ValueError(f"Invalid stage: {stage}. Use 'raw', 'interim', or 'processed'")

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = filepath.stem
        filepath_with_timestamp = filepath.parent / f"{base_name}_{timestamp}.csv"

        # Save both with and without timestamp
        data.to_csv(filepath, index=False)
        data.to_csv(filepath_with_timestamp, index=False)

        print(f"Data saved to: {filepath}")
        print(f"Backup saved to: {filepath_with_timestamp}")

        return str(filepath)

    def load_from_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath (str or Path): Path to CSV file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file cannot be read
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}: {data.shape[0]} rows, {data.shape[1]} columns")
            return data

        except Exception as e:
            raise Exception(f"Failed to load data from {filepath}: {str(e)}")

    def run_pipeline(self, save_intermediate: bool = True) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.

        This method executes all steps in sequence:
        1. Load raw data
        2. Handle missing values
        3. Remove outliers
        4. Apply feature engineering
        5. Save processed data

        Args:
            save_intermediate (bool): Whether to save intermediate results.
                Default is True.

        Returns:
            pd.DataFrame: Fully processed data ready for modeling

        Raises:
            Exception: If any step in the pipeline fails
        """
        print("="*50)
        print("Starting Data Processing Pipeline")
        print("="*50)

        # Step 1: Load data
        print("\n[Step 1/5] Loading data...")
        self.load_data()
        if save_intermediate:
            self.save_data(self.data, 'raw')

        # Step 2: Handle missing values
        print("\n[Step 2/5] Handling missing values...")
        self.check_missing_values()
        self.handle_missing_values()

        # Step 3: Remove outliers
        print("\n[Step 3/5] Removing outliers...")
        self.remove_outliers()
        if save_intermediate:
            self.save_data(self.cleaned_data, 'interim')

        # Step 4: Feature engineering
        print("\n[Step 4/5] Applying feature engineering...")
        self.apply_feature_engineering()

        # Step 5: Save processed data
        print("\n[Step 5/5] Saving processed data...")
        self.save_data(self.feature_engineered_data, 'processed')

        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        print("="*50)
        print(f"Final dataset shape: {self.feature_engineered_data.shape}")
        print(f"Columns: {list(self.feature_engineered_data.columns)}")

        return self.feature_engineered_data
