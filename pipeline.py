"""
Main Data Pipeline for California Housing Prediction Project.

This script orchestrates the entire workflow:
1. Load raw data from sklearn
2. Clean and preprocess data
3. Engineer features
4. Save to database
5. Train model
6. Generate visualizations

Run this script BEFORE launching the Streamlit dashboard.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset import HousingDataProcessor
from src.services.database import DatabaseManager
from src.modeling.train import PricePredictionModel
from src.plots import EDAAnalyser
from src.config import PROCESSED_DATA_PATH


def main():
    """Execute the complete data pipeline."""

    print("="*70)
    print("CALIFORNIA HOUSING PREDICTION - DATA PIPELINE")
    print("="*70)

    # Step 1: Load and process data
    print("\n[STEP 1/5] Loading and processing data...")
    print("-" * 70)

    processor = HousingDataProcessor()

    # Load raw data
    print("Loading California housing data from sklearn...")
    raw_data = processor.load_data()
    print(f"✓ Loaded {raw_data.shape[0]:,} rows, {raw_data.shape[1]} columns")

    # Save raw data
    processor.save_data(raw_data, 'raw')

    # Check for missing values
    print("\nChecking for missing values...")
    missing = processor.check_missing_values()

    # Handle missing values if any exist
    if len(missing) > 0:
        print("Handling missing values...")
        cleaned_data = processor.handle_missing_values(strategy='median')
    else:
        cleaned_data = raw_data.copy()

    # Save interim data
    processor.save_data(cleaned_data, 'interim')
    print(f"✓ Cleaned data saved")

    # Remove outliers
    print("\nRemoving outliers...")
    cleaned_data = processor.remove_outliers(method='iqr', threshold=1.5)

    # Apply feature engineering
    print("\nApplying feature engineering...")
    processed_data = processor.apply_feature_engineering(cleaned_data)
    print(f"✓ Features engineered: {processed_data.shape[1]} total columns")

    # Save processed data
    processor.save_data(processed_data, 'processed')
    print(f"✓ Processed data saved to {PROCESSED_DATA_PATH}")

    # Step 2: Load data into database
    print("\n[STEP 2/5] Loading data into SQLite database...")
    print("-" * 70)

    db = DatabaseManager()
    db.create_connection()

    # Create housing table
    print("Creating database tables...")
    db.create_tables()

    # Insert data
    print("Inserting data into database...")
    rows_inserted = db.insert_data(processed_data, table_name='housing')
    print(f"✓ Inserted {rows_inserted:,} rows into database")

    # Verify data
    count = db.get_table_count('housing')
    print(f"✓ Database verification: {count:,} records")

    db.close_connection()

    # Step 3: Generate visualizations
    print("\n[STEP 3/5] Generating exploratory visualizations...")
    print("-" * 70)

    eda = EDAAnalyser(processed_data)

    try:
        print("Generating all visualizations...")
        plots = eda.generate_all_plots()
        print(f"✓ Generated {len(plots)} visualizations")
        print("✓ Visualizations saved to reports/figures/")
    except Exception as e:
        print(f"⚠ Warning: Some visualizations failed: {str(e)}")
        print("  (This is non-critical, continuing with pipeline...)")

    # Step 4: Train model
    print("\n[STEP 4/5] Training prediction model...")
    print("-" * 70)

    model = PricePredictionModel()

    # Prepare features
    print("Preparing features and target variable...")
    X, y = model.prepare_features(processed_data)
    print(f"✓ Prepared {X.shape[0]:,} samples with {X.shape[1]} features")

    # Split data
    print("Splitting into training and test sets...")
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    print(f"✓ Training set: {X_train.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")

    # Train model
    print("\nTraining Linear Regression model...")
    model.train(X_train, y_train)
    print("✓ Model trained successfully")

    # Step 5: Evaluate model
    print("\n[STEP 5/5] Evaluating model performance...")
    print("-" * 70)

    metrics = model.evaluate(X_test, y_test)

    print(f"\nModel Performance Metrics:")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Save model
    print("\nSaving trained model...")
    model.save_model()
    print("✓ Model saved successfully")

    # Generate prediction plots
    try:
        print("\nGenerating prediction visualizations...")
        model.plot_predictions_vs_actual(y_test)
        model.plot_residuals(y_test)
        print("✓ Prediction plots saved")
    except Exception as e:
        print(f"⚠ Warning: Prediction plots failed: {str(e)}")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nData Summary:")
    print(f"  Total records: {processed_data.shape[0]:,}")
    print(f"  Features: {processed_data.shape[1] - 1}")
    print(f"  Target: median_house_value")

    print("\nNext Steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Open your browser to view the dashboard")
    print("  3. Explore the data and make predictions!")

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
