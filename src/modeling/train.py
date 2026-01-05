"""
Model Training Module.

This module contains the PricePredictionModel class for training and evaluating
a linear regression model for California housing price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.linear_regression import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.config import (
    MODEL_PATH, SCALER_PATH, METRICS_PATH,
    FEATURE_COLUMNS, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE,
    FIGURES_DIR
)


class PricePredictionModel:
    """
    Linear regression model for house price prediction.

    This class handles:
    - Feature preparation and scaling
    - Train/test splitting
    - Model training with Linear Regression
    - Model evaluation (RMSE, R², MAE)
    - Model persistence (saving/loading)
    - Feature importance analysis
    - Prediction visualization

    Attributes:
        model (LinearRegression): Scikit-learn linear regression model
        scaler (StandardScaler): Feature scaler
        feature_columns (list): List of feature column names
        metrics (dict): Model evaluation metrics
    """

    def __init__(self):
        """Initialize the prediction model."""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns: Optional[list] = None
        self.metrics: Dict = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def prepare_features(self, data: pd.DataFrame,
                        target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Separates the dataset into features (X) and target (y).
        Uses only numeric features specified in FEATURE_COLUMNS.

        Args:
            data (pd.DataFrame): Full dataset
            target_column (str, optional): Name of target column.
                If None, uses TARGET_COLUMN from config.

        Returns:
            tuple: (X, y) where X is features DataFrame, y is target Series

        Raises:
            ValueError: If data is None or required columns are missing
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")

        if target_column is None:
            target_column = TARGET_COLUMN

        # Determine feature columns
        if self.feature_columns is None:
            # Use FEATURE_COLUMNS from config, but only if they exist in data
            self.feature_columns = [col for col in FEATURE_COLUMNS if col in data.columns]

        # Validate that we have features
        if len(self.feature_columns) == 0:
            raise ValueError("No valid feature columns found")

        # Validate that target exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Extract features and target
        X = data[self.feature_columns].copy()
        y = data[target_column].copy()

        # Check for missing values
        if X.isnull().any().any():
            print("Warning: Missing values found in features. Filling with median...")
            X = X.fillna(X.median())

        if y.isnull().any():
            print("Warning: Missing values found in target. Dropping rows...")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]

        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Feature columns: {self.feature_columns}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = None,
                   random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float, optional): Proportion for test set.
                If None, uses TEST_SIZE from config.
            random_state (int, optional): Random seed.
                If None, uses RANDOM_STATE from config.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If X or y is None
        """
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if test_size is None:
            test_size = TEST_SIZE
        if random_state is None:
            random_state = RANDOM_STATE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"Data split: Train={len(X_train)}, Test={len(X_test)} (test_size={test_size})")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame = None, y_train: pd.Series = None) -> None:
        """
        Train the linear regression model.

        Features are scaled using StandardScaler before training.

        Args:
            X_train (pd.DataFrame, optional): Training features.
                If None, uses self.X_train
            y_train (pd.Series, optional): Training target.
                If None, uses self.y_train

        Raises:
            ValueError: If training data is not available
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("Training data not available. Call split_data() first.")

        print("Training model...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Calculate training score
        train_score = self.model.score(X_train_scaled, y_train)

        print(f"Model trained successfully!")
        print(f"Training R² score: {train_score:.4f}")

    def evaluate(self, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """
        Evaluate model performance.

        Calculates RMSE, R², and MAE metrics.

        Args:
            X_test (pd.DataFrame, optional): Test features.
                If None, uses self.X_test
            y_test (pd.Series, optional): Test target.
                If None, uses self.y_test

        Returns:
            dict: Contains:
                - rmse: Root Mean Squared Error
                - r2: R² score
                - mae: Mean Absolute Error
                - predictions: Array of predictions

        Raises:
            ValueError: If test data is not available or model is not trained
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        if X_test is None or y_test is None:
            raise ValueError("Test data not available. Call split_data() first.")

        print("Evaluating model...")

        # Scale features
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        self.metrics = {
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae),
            'predictions': y_pred
        }

        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print("="*50)

        return self.metrics

    def save_model(self, model_path: Path = None, scaler_path: Path = None,
                  metrics_path: Path = None) -> Tuple[str, str, str]:
        """
        Save trained model, scaler, and metrics to pickle files.

        Args:
            model_path (Path, optional): Path to save model.
                If None, uses MODEL_PATH from config.
            scaler_path (Path, optional): Path to save scaler.
                If None, uses SCALER_PATH from config.
            metrics_path (Path, optional): Path to save metrics.
                If None, uses METRICS_PATH from config.

        Returns:
            tuple: (model_path, scaler_path, metrics_path) where files were saved

        Raises:
            Exception: If saving fails
        """
        if model_path is None:
            model_path = MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_PATH
        if metrics_path is None:
            metrics_path = METRICS_PATH

        try:
            # Ensure directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save feature columns and metrics
            save_data = {
                'feature_columns': self.feature_columns,
                'metrics': {k: v for k, v in self.metrics.items() if k != 'predictions'}
            }

            with open(metrics_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"\nModel saved to: {model_path}")
            print(f"Scaler saved to: {scaler_path}")
            print(f"Metrics saved to: {metrics_path}")

            return str(model_path), str(scaler_path), str(metrics_path)

        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, model_path: Path = None, scaler_path: Path = None,
                  metrics_path: Path = None) -> None:
        """
        Load trained model, scaler, and metrics from pickle files.

        Args:
            model_path (Path, optional): Path to model file
            scaler_path (Path, optional): Path to scaler file
            metrics_path (Path, optional): Path to metrics file

        Raises:
            FileNotFoundError: If files don't exist
            Exception: If loading fails
        """
        if model_path is None:
            model_path = MODEL_PATH
        if scaler_path is None:
            scaler_path = SCALER_PATH
        if metrics_path is None:
            metrics_path = METRICS_PATH

        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load feature columns and metrics
            with open(metrics_path, 'r') as f:
                load_data = json.load(f)
                self.feature_columns = load_data['feature_columns']
                self.metrics = load_data['metrics']

            print(f"Model loaded from: {model_path}")
            print(f"Scaler loaded from: {scaler_path}")
            print(f"Metrics loaded from: {metrics_path}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature coefficients as importance measure.

        For linear regression, coefficients indicate the change in target
        for a one-unit change in the feature (when other features are held constant).

        Returns:
            pd.DataFrame: Features with coefficients sorted by absolute value

        Raises:
            ValueError: If model is not trained
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get coefficients
        coefficients = self.model.coef_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # Sort by absolute value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        print("\nFeature Importance (Linear Regression Coefficients):")
        print("="*50)
        print(importance_df[['feature', 'coefficient']])
        print("="*50)

        return importance_df

    def plot_predictions_vs_actual(self, y_test: pd.Series = None,
                                   y_pred: np.ndarray = None,
                                   save: bool = True) -> None:
        """
        Plot predicted vs actual values.

        Shows how well predictions match actual values.
        Perfect predictions would lie on the diagonal line.

        Args:
            y_test (pd.Series, optional): Actual values
            y_pred (np.ndarray, optional): Predicted values
            save (bool): Whether to save figure

        Raises:
            ValueError: If data is not available
        """
        if y_test is None:
            y_test = self.y_test
        if y_pred is None:
            y_pred = self.metrics.get('predictions')

        if y_test is None or y_pred is None:
            raise ValueError("Test data not available. Call evaluate() first.")

        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')

        # Diagonal line (perfect predictions)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual House Value ($)')
        plt.ylabel('Predicted House Value ($)')
        plt.title('Predicted vs Actual House Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            filename = FIGURES_DIR / 'model_predictions_vs_actual.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def plot_residuals(self, y_test: pd.Series = None,
                      y_pred: np.ndarray = None,
                      save: bool = True) -> None:
        """
        Plot residuals distribution.

        Residuals are the differences between actual and predicted values.
        A good model should have residuals centered around zero.

        Args:
            y_test (pd.Series, optional): Actual values
            y_pred (np.ndarray, optional): Predicted values
            save (bool): Whether to save figure

        Raises:
            ValueError: If data is not available
        """
        if y_test is None:
            y_test = self.y_test
        if y_pred is None:
            y_pred = self.metrics.get('predictions')

        if y_test is None or y_pred is None:
            raise ValueError("Test data not available. Call evaluate() first.")

        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of residuals
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[0].set_xlabel('Residuals ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Residuals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot of residuals
        axes[1].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[1].set_xlabel('Predicted House Value ($)')
        axes[1].set_ylabel('Residuals ($)')
        axes[1].set_title('Residuals vs Predicted Values')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = FIGURES_DIR / 'model_residuals.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.close()

    def run_full_training_pipeline(self, data: pd.DataFrame,
                                   save_model: bool = True) -> Dict:
        """
        Run the complete training pipeline.

        This method executes all steps in sequence:
        1. Prepare features
        2. Split data
        3. Train model
        4. Evaluate performance
        5. Generate visualizations
        6. Save model (optional)

        Args:
            data (pd.DataFrame): Processed housing data
            save_model (bool): Whether to save the trained model

        Returns:
            dict: Evaluation metrics

        Raises:
            Exception: If any step fails
        """
        print("="*50)
        print("Starting Model Training Pipeline")
        print("="*50)

        # Step 1: Prepare features
        print("\n[Step 1/6] Preparing features...")
        X, y = self.prepare_features(data)

        # Step 2: Split data
        print("\n[Step 2/6] Splitting data...")
        self.split_data(X, y)

        # Step 3: Train model
        print("\n[Step 3/6] Training model...")
        self.train()

        # Step 4: Evaluate
        print("\n[Step 4/6] Evaluating model...")
        metrics = self.evaluate()

        # Step 5: Visualizations
        print("\n[Step 5/6] Generating visualizations...")
        self.plot_predictions_vs_actual()
        self.plot_residuals()
        self.get_feature_importance()

        # Step 6: Save model
        if save_model:
            print("\n[Step 6/6] Saving model...")
            self.save_model()

        print("\n" + "="*50)
        print("Training Pipeline Completed Successfully!")
        print("="*50)

        return metrics
