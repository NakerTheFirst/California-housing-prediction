"""
Feature engineering functions for California Housing dataset.

This module contains functions for creating derived features from the raw housing data.
These features help improve model performance by capturing relationships between variables.
"""

import pandas as pd
import numpy as np
from typing import Optional
from config import INCOME_CATEGORIES, AGE_CATEGORIES


def create_rooms_per_household(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rooms per household ratio.

    This feature indicates the average number of rooms per household,
    which can be a proxy for house size and quality.

    Formula: total_rooms / households

    Args:
        data (pd.DataFrame): DataFrame containing 'total_rooms' and 'households' columns

    Returns:
        pd.DataFrame: DataFrame with added 'rooms_per_household' column

    Raises:
        KeyError: If required columns are not present
    """
    data = data.copy()

    # Handle division by zero
    data['rooms_per_household'] = data['total_rooms'] / data['households'].replace(0, np.nan)

    # Fill any resulting NaN values with median
    median_value = data['rooms_per_household'].median()
    data['rooms_per_household'].fillna(median_value, inplace=True)

    return data


def create_bedrooms_per_room(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate bedrooms per room ratio.

    This feature helps understand the layout of the house.
    A lower ratio suggests more living space relative to bedrooms.

    Formula: total_bedrooms / total_rooms

    Args:
        data (pd.DataFrame): DataFrame containing 'total_bedrooms' and 'total_rooms' columns

    Returns:
        pd.DataFrame: DataFrame with added 'bedrooms_per_room' column

    Raises:
        KeyError: If required columns are not present
    """
    data = data.copy()

    # Handle division by zero
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms'].replace(0, np.nan)

    # Fill any resulting NaN values with median
    median_value = data['bedrooms_per_room'].median()
    data['bedrooms_per_room'].fillna(median_value, inplace=True)

    # Cap at 1.0 (can't have more bedrooms than rooms)
    data['bedrooms_per_room'] = data['bedrooms_per_room'].clip(upper=1.0)

    return data


def create_population_per_household(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate population per household.

    This feature indicates the average household size,
    which can correlate with house values and neighborhood characteristics.

    Formula: population / households

    Args:
        data (pd.DataFrame): DataFrame containing 'population' and 'households' columns

    Returns:
        pd.DataFrame: DataFrame with added 'population_per_household' column

    Raises:
        KeyError: If required columns are not present
    """
    data = data.copy()

    # Handle division by zero
    data['population_per_household'] = data['population'] / data['households'].replace(0, np.nan)

    # Fill any resulting NaN values with median
    median_value = data['population_per_household'].median()
    data['population_per_household'].fillna(median_value, inplace=True)

    return data


def create_income_categories(data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize median income into bins.

    Income categories help capture non-linear relationships between
    income and house prices.

    Categories:
    - 'low': < 2.5
    - 'medium': 2.5 - 4.5
    - 'high': 4.5 - 6.0
    - 'very_high': > 6.0

    Args:
        data (pd.DataFrame): DataFrame containing 'median_income' column

    Returns:
        pd.DataFrame: DataFrame with added 'income_category' column

    Raises:
        KeyError: If 'median_income' column is not present
    """
    data = data.copy()

    def categorize_income(income: float) -> str:
        """Helper function to categorize a single income value."""
        for category, (min_val, max_val) in INCOME_CATEGORIES.items():
            if min_val <= income < max_val:
                return category
        return 'very_high'  # Default for values above highest threshold

    data['income_category'] = data['median_income'].apply(categorize_income)

    return data


def create_age_categories(data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize housing median age into bins.

    Age categories help capture the relationship between
    housing age and property values.

    Categories:
    - 'new': < 10 years
    - 'medium': 10-30 years
    - 'old': > 30 years

    Args:
        data (pd.DataFrame): DataFrame containing 'housing_median_age' column

    Returns:
        pd.DataFrame: DataFrame with added 'age_category' column

    Raises:
        KeyError: If 'housing_median_age' column is not present
    """
    data = data.copy()

    def categorize_age(age: float) -> str:
        """Helper function to categorize a single age value."""
        for category, (min_val, max_val) in AGE_CATEGORIES.items():
            if min_val <= age < max_val:
                return category
        return 'old'  # Default for values above highest threshold

    data['age_category'] = data['housing_median_age'].apply(categorize_age)

    return data


def create_log_features(data: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Create log-transformed features for specified columns.

    Log transformation helps normalize skewed distributions and
    can improve model performance for features with wide ranges.

    Args:
        data (pd.DataFrame): Input DataFrame
        columns (list, optional): Columns to transform.
            Defaults to ['total_rooms', 'total_bedrooms', 'population', 'households']

    Returns:
        pd.DataFrame: DataFrame with added log-transformed columns (prefixed with 'log_')

    Raises:
        KeyError: If specified columns are not present
    """
    data = data.copy()

    if columns is None:
        columns = ['total_rooms', 'total_bedrooms', 'population', 'households']

    for column in columns:
        if column in data.columns:
            # Add 1 to avoid log(0) issues
            data[f'log_{column}'] = np.log1p(data[column])

    return data


def apply_all_features(data: pd.DataFrame, include_log: bool = False) -> pd.DataFrame:
    """
    Apply all feature engineering functions to the data.

    This is the main function that should be called to create all
    engineered features at once.

    Args:
        data (pd.DataFrame): Raw housing data
        include_log (bool): Whether to include log-transformed features.
            Default is False to keep model simple.

    Returns:
        pd.DataFrame: DataFrame with all engineered features added

    Raises:
        KeyError: If required columns are not present in the input data
    """
    data = data.copy()

    # Create ratio features
    data = create_rooms_per_household(data)
    data = create_bedrooms_per_room(data)
    data = create_population_per_household(data)

    # Create categorical features
    data = create_income_categories(data)
    data = create_age_categories(data)

    # Optionally create log features
    if include_log:
        data = create_log_features(data)

    return data


def validate_features(data: pd.DataFrame) -> dict:
    """
    Validate that all expected features are present and have valid values.

    Args:
        data (pd.DataFrame): DataFrame to validate

    Returns:
        dict: Validation results with keys:
            - 'valid' (bool): Overall validity
            - 'missing_features' (list): List of missing expected features
            - 'null_counts' (dict): Count of null values per feature
            - 'invalid_ranges' (dict): Features with values outside expected ranges
    """
    expected_features = [
        'rooms_per_household',
        'bedrooms_per_room',
        'population_per_household',
        'income_category',
        'age_category'
    ]

    results = {
        'valid': True,
        'missing_features': [],
        'null_counts': {},
        'invalid_ranges': {}
    }

    # Check for missing features
    for feature in expected_features:
        if feature not in data.columns:
            results['missing_features'].append(feature)
            results['valid'] = False

    # Check for null values
    for feature in expected_features:
        if feature in data.columns:
            null_count = data[feature].isnull().sum()
            if null_count > 0:
                results['null_counts'][feature] = null_count
                results['valid'] = False

    # Check for invalid ranges
    if 'bedrooms_per_room' in data.columns:
        invalid_count = ((data['bedrooms_per_room'] < 0) |
                        (data['bedrooms_per_room'] > 1.0)).sum()
        if invalid_count > 0:
            results['invalid_ranges']['bedrooms_per_room'] = invalid_count

    return results
