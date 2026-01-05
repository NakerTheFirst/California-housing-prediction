"""
Configuration settings for California Housing prediction project.

This module contains all configuration constants, paths, and hyperparameters
used throughout the project. All other modules should import from this file
to ensure consistency.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# Data paths
RAW_DATA_PATH = DATA_DIR / 'raw' / 'housing_raw.csv'
INTERIM_DATA_PATH = DATA_DIR / 'interim' / 'housing_cleaned.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'housing_processed.csv'

# Database configuration
DATABASE_PATH = DATA_DIR / 'housing.db'

# Model configuration
MODEL_PATH = MODELS_DIR / 'linear_regression_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
METRICS_PATH = MODELS_DIR / 'evaluation_metrics.json'

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# Feature engineering parameters
INCOME_CATEGORIES = {
    'low': (0, 2.5),
    'medium': (2.5, 4.5),
    'high': (4.5, 6.0),
    'very_high': (6.0, float('inf'))
}

AGE_CATEGORIES = {
    'new': (0, 10),
    'medium': (10, 30),
    'old': (30, float('inf'))
}

# Number of location clusters for geographic feature engineering
N_LOCATION_CLUSTERS = 5

# Visualization settings
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Feature columns for modeling (excluding target)
FEATURE_COLUMNS = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'rooms_per_household',
    'bedrooms_per_room',
    'population_per_household'
]

# Target column
TARGET_COLUMN = 'median_house_value'

# All original columns from sklearn dataset
ORIGINAL_COLUMNS = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# Outlier removal settings
OUTLIER_METHOD = 'iqr'  # Options: 'iqr' or 'zscore'
OUTLIER_THRESHOLD = 1.5  # IQR multiplier (1.5) or z-score threshold (3.0)

# Missing value handling
MISSING_VALUE_STRATEGY = 'median'  # Options: 'median', 'mean', 'mode', 'drop'

# Streamlit configuration
STREAMLIT_TITLE = "California Housing Price Prediction"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_STATE = "expanded"

# Data validation thresholds
MIN_LONGITUDE = -124.35
MAX_LONGITUDE = -114.31
MIN_LATITUDE = 32.54
MAX_LATITUDE = 41.95
MIN_HOUSING_AGE = 1
MAX_HOUSING_AGE = 52
MIN_INCOME = 0.5
MAX_INCOME = 15.0

# Expected model performance targets
TARGET_RMSE = 70000  # Maximum acceptable RMSE
TARGET_R2 = 0.60     # Minimum acceptable R²
TARGET_MAE = 50000   # Maximum acceptable MAE

# Ensure directories exist
def ensure_directories_exist():
    """
    Create all necessary directories if they don't exist.
    Should be called at application startup.
    """
    directories = [
        DATA_DIR / 'raw',
        DATA_DIR / 'interim',
        DATA_DIR / 'processed',
        DATA_DIR / 'external',
        MODELS_DIR,
        REPORTS_DIR / 'figures',
        NOTEBOOKS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on module import
ensure_directories_exist()
