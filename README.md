# California Housing Price Prediction

End-to-end machine learning pipeline for predicting median house prices in California districts using Linear Regression, SQL integration, and an interactive Streamlit dashboard.

## Overview

1. Dataset: California Housing Dataset (20,640 districts with 8 features)
2. Algorithm: Linear Regression with feature engineering
3. Tech Stack: Python, scikit-learn, SQLite, Streamlit, pandas, seaborn

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/WSL (.venv\Scripts\activate for Windows)
pip install -r requirements.txt

# Run complete pipeline (data processing → database → visualizations → model training)
python pipeline.py

# Launch interactive dashboard
streamlit run app.py
```

## Project Structure

```
├── src/
│   ├── config.py                 # Configuration constants and paths
│   ├── dataset.py                # HousingDataProcessor: data loading & cleaning
│   ├── features.py               # Feature engineering functions
│   ├── plots.py                  # EDAAnalyser: 10 visualization types
│   ├── modeling/
│   │   ├── train.py              # PricePredictionModel: model training
│   │   └── predict.py            # PredictionInterface: inference
│   └── services/
│       └── database.py           # DatabaseManager: SQLite operations
├── app.py                        # Streamlit dashboard (main interface)
├── pipeline.py                   # Full pipeline orchestrator
├── data/
│   ├── raw/                      # Original dataset
│   ├── interim/                  # Cleaned data
│   ├── processed/                # Feature-engineered data
│   └── housing.db                # SQLite database
├── models/                       # Trained model artifacts (.pkl files)
├── reports/figures/              # Generated visualizations (12 PNG files)
└── notebooks/                    # Jupyter notebooks for EDA
```

## Core Components

### 1. HousingDataProcessor (`src/dataset.py`)
Data loading, cleaning, and preprocessing
- Loads California Housing dataset from sklearn
- Handles missing values (median/mean/mode imputation)
- Removes outliers (IQR or z-score methods)
- Applies feature engineering
- Saves data at each pipeline stage

### 2. DatabaseManager (`src/services/database.py`)
SQLite database operations demonstrating SQL concepts
- **Tables:** `housing` (20,390 rows), `district_summary` (4 rows)
- **WHERE filtering:** Income and location-based queries
- **GROUP BY aggregation:** Statistics by income category
- **INNER JOIN:** Merges housing data with district summaries
- CRUD operations with Python API

### 3. EDAAnalyser (`src/plots.py`)
Generates 10 visualization types
1. Histogram (price distribution)
2. Boxplot (price by income category)
3. Scatter (geographic coordinates)
4. Correlation heatmap (14×14 matrix)
5. Pairplot (key feature relationships)
6. Bar chart (mean price by category)
7. Violin plot (income distribution)
8. Line chart (price trends by age)
9. Density plot (multiple features)
10. Geographic scatter (California map)

### 4. PricePredictionModel (`src/modeling/train.py`)
Linear regression training and evaluation
- Uses 11 features (8 original + 3 engineered ratios)
- StandardScaler normalization
- 80/20 train-test split
- Evaluation metrics: RMSE, R², MAE
- Saves model, scaler, and metrics as .pkl files

### 5. PredictionInterface (`src/modeling/predict.py`)
Simplified prediction API
- Single and batch predictions
- Automatic feature engineering
- Input validation (geographic bounds, positive values)
- Returns predictions with detailed metadata

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Data Processing                                         │
├─────────────────────────────────────────────────────────────────┤
│ sklearn.datasets → Load → Clean → Remove Outliers →             │
│ Feature Engineering → data/processed/housing_processed.csv      │
│ (20,640 rows → ~20,390 rows after outlier removal)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Database Operations                                     │
├─────────────────────────────────────────────────────────────────┤
│ Create SQLite DB → Create Tables → Insert Data →                │
│ Populate Aggregated Summary → data/housing.db (1.7 MB)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Exploratory Data Analysis                               │
├─────────────────────────────────────────────────────────────────┤
│ Generate 10 Visualizations → reports/figures/*.png              │
│ Correlation Analysis → Summary Statistics                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Model Training                                          │
├─────────────────────────────────────────────────────────────────┤
│ Prepare Features (11 columns) → Split Data (80/20) →            │
│ StandardScaler → LinearRegression.fit() →                       │
│ models/{model.pkl, scaler.pkl, metrics.json}                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Model Evaluation & Visualization                        │
├─────────────────────────────────────────────────────────────────┤
│ Calculate Metrics (RMSE, R², MAE) →                             │
│ Plot Predictions vs Actual → Plot Residuals →                   │
│ reports/figures/model_*.png                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Database Structure

### Table: `housing`
Primary data table with 20,390 rows and 14 columns

**Original Features (8):**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of houses in district
- `total_rooms`, `total_bedrooms` - Total counts in district
- `population`, `households` - Population statistics
- `median_income` - Median household income (×$10,000)
- `median_house_value` - Target variable (price in dollars)

**Engineered Features (5):**
- `rooms_per_household` = total_rooms / households
- `bedrooms_per_room` = total_bedrooms / total_rooms
- `population_per_household` = population / households
- `income_category` - Categorical: low/medium/high/very_high
- `age_category` - Categorical: new/medium/old

### Table: `district_summary`
Aggregated statistics by income category (4 rows)

**Columns:**
- `income_category` - Category identifier (UNIQUE)
- `avg_house_value` - Average price in category
- `avg_rooms` - Average rooms per household
- `avg_age` - Average housing age
- `district_count` - Number of districts in category

### SQL Operations Demonstrated

**WHERE Filtering:**
```sql
SELECT * FROM housing
WHERE median_income >= ? AND median_income <= ?
```

**GROUP BY Aggregation:**
```sql
SELECT income_category,
       AVG(median_house_value) as avg_price,
       COUNT(*) as count
FROM housing
GROUP BY income_category
ORDER BY avg_price DESC
```

**INNER JOIN:**
```sql
SELECT h.*, ds.avg_house_value, ds.district_count
FROM housing h
INNER JOIN district_summary ds
ON h.income_category = ds.income_category
```

## Streamlit Dashboard

Interactive 4-page application at `http://localhost:8501`:

1. **Home** - Project overview and dataset statistics
2. **Data Exploration** - Interactive data table with filters, SQL query demonstrations
3. **Visualizations** - Gallery of 10 EDA plots + 2 model performance plots
4. **Price Prediction** - Interactive prediction interface with input sliders

Features:
- Real-time SQL query execution
- Dynamic data filtering
- Model performance metrics
- Single-house price predictions
- Engineered feature display

## Results

### Model Performance
- **RMSE:** ~$68,000-72,000 (typical error in predictions)
- **R²:** ~0.58-0.62 (model explains 58-62% of variance)
- **MAE:** ~$48,000-53,000 (average absolute error)

### Key Findings
- **Median income** is the strongest predictor of house prices
- **Geographic location** (latitude/longitude) significantly impacts prices
- **Engineered features** (rooms per household, population density) improve model performance
- Linear regression provides interpretable baseline but has moderate accuracy due to:
  - Non-linear relationships in housing data
  - Geographic clustering effects
  - Presence of outliers in price distribution

### Generated Artifacts
- **12 visualizations** in `reports/figures/`
- **SQLite database** (1.7 MB) with 20,390 records
- **3 model files** (model, scaler, metrics)
- **Interactive dashboard** for exploration and predictions

## Configuration

Key settings in `src/config.py`:
- `TEST_SIZE = 0.2` (80/20 train-test split)
- `RANDOM_STATE = 42` (reproducibility)
- `OUTLIER_THRESHOLD = 1.5` (IQR multiplier)
- `MISSING_VALUE_STRATEGY = 'median'` (imputation method)

## Dependencies

Core libraries (see `requirements.txt`):
- pandas 2.1.0 - Data manipulation
- numpy 1.25.0 - Numerical operations
- scikit-learn 1.3.0 - ML algorithms
- streamlit 1.40.0 - Web dashboard
- seaborn 0.12.2 - Visualizations
- matplotlib 3.7.2 - Plotting

## Project Workflow

### Development
```bash
# Explore data in Jupyter
jupyter notebook notebooks/

# Run individual components
python src/modeling/train.py      # Train model only
python src/modeling/predict.py    # Run predictions

# Verify database
python -c "from src.services.database import DatabaseManager; DatabaseManager().verify_connection()"
```

### Production
```bash
python pipeline.py       # Execute complete ETL + training pipeline
streamlit run app.py     # Launch web interface
```

## Course Requirements Met

- ✅ Data wrangling with pandas/numpy
- ✅ SQL integration (WHERE, GROUP BY, INNER JOIN)
- ✅ 10 distinct visualization types
- ✅ Object-oriented programming (5 classes)
- ✅ Interactive Streamlit dashboard
- ✅ Linear regression with evaluation metrics

## License

Educational project for Data Science Master's programme.
