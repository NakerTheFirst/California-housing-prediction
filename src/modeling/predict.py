"""
Prediction Interface Module.

This module contains the PredictionInterface class for making predictions
with a trained model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union

from src.modeling.train import PricePredictionModel
from src.config import MODEL_PATH, SCALER_PATH, METRICS_PATH
from src.features import (
    create_rooms_per_household,
    create_bedrooms_per_room,
    create_population_per_household,
    create_income_categories
)


class PredictionInterface:
    """
    Interface for making predictions with trained model.

    This class provides a simplified interface for:
    - Loading trained model
    - Validating input features
    - Making single predictions
    - Making batch predictions
    - Automatic feature engineering for new data

    Attributes:
        model (PricePredictionModel): Loaded trained model
        feature_columns (list): Required feature columns
    """

    def __init__(self, model_path: Path = None, scaler_path: Path = None,
                 metrics_path: Path = None):
        """
        Initialize prediction interface.

        Loads the trained model, scaler, and feature configuration.

        Args:
            model_path (Path, optional): Path to trained model file
            scaler_path (Path, optional): Path to scaler file
            metrics_path (Path, optional): Path to metrics file

        Raises:
            FileNotFoundError: If model files don't exist
        """
        self.model = PricePredictionModel()

        # Load model
        try:
            self.model.load_model(model_path, scaler_path, metrics_path)
            self.feature_columns = self.model.feature_columns
            print("Prediction interface initialized successfully")
            print(f"Required features: {self.feature_columns}")

        except Exception as e:
            raise Exception(f"Failed to initialize prediction interface: {str(e)}")

    def validate_features(self, features: Dict) -> bool:
        """
        Validate that all required features are present.

        Args:
            features (dict): Dictionary of feature values

        Returns:
            bool: True if valid

        Raises:
            ValueError: If required features are missing or invalid
        """
        # Check for base features needed for feature engineering
        base_features = [
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income'
        ]

        missing_features = []
        for feature in base_features:
            if feature not in features:
                missing_features.append(feature)

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Validate data types and ranges
        try:
            # Longitude should be negative (California)
            if features['longitude'] > 0:
                raise ValueError("Longitude should be negative for California")

            # Latitude should be between 32 and 42 (California)
            if not (32 <= features['latitude'] <= 42):
                raise ValueError("Latitude should be between 32 and 42 for California")

            # Positive values for counts
            for feature in ['total_rooms', 'total_bedrooms', 'population', 'households']:
                if features[feature] <= 0:
                    raise ValueError(f"{feature} must be positive")

            # Age should be reasonable
            if not (0 <= features['housing_median_age'] <= 100):
                raise ValueError("Housing median age should be between 0 and 100")

            # Income should be positive
            if features['median_income'] <= 0:
                raise ValueError("Median income must be positive")

        except KeyError as e:
            raise ValueError(f"Missing feature: {e}")

        return True

    def engineer_features(self, features: Dict) -> Dict:
        """
        Apply feature engineering to input features.

        Creates the derived features needed by the model.

        Args:
            features (dict): Base feature values

        Returns:
            dict: Features with engineered features added
        """
        # Create a DataFrame from the dict
        df = pd.DataFrame([features])

        # Apply feature engineering
        df = create_rooms_per_household(df)
        df = create_bedrooms_per_room(df)
        df = create_population_per_household(df)
        df = create_income_categories(df)

        # Convert back to dict
        features_with_engineering = df.iloc[0].to_dict()

        return features_with_engineering

    def predict_single(self, features: Dict) -> float:
        """
        Predict price for single house.

        Args:
            features (dict): Dictionary of feature values with keys:
                - longitude, latitude, housing_median_age
                - total_rooms, total_bedrooms, population
                - households, median_income

        Returns:
            float: Predicted house price in dollars

        Raises:
            ValueError: If features are invalid
        """
        # Validate input
        self.validate_features(features)

        # Apply feature engineering
        features_engineered = self.engineer_features(features)

        # Extract only the features needed by the model
        feature_values = [features_engineered[col] for col in self.feature_columns]

        # Create DataFrame with single row
        X = pd.DataFrame([feature_values], columns=self.feature_columns)

        # Scale features
        X_scaled = self.model.scaler.transform(X)

        # Make prediction
        prediction = self.model.model.predict(X_scaled)[0]

        return float(prediction)

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices for multiple houses.

        Args:
            features_df (pd.DataFrame): DataFrame of features with columns:
                - longitude, latitude, housing_median_age
                - total_rooms, total_bedrooms, population
                - households, median_income

        Returns:
            np.ndarray: Array of predicted prices

        Raises:
            ValueError: If features are invalid
        """
        if features_df is None or len(features_df) == 0:
            raise ValueError("Features DataFrame cannot be None or empty")

        # Validate base features exist
        base_features = [
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income'
        ]

        missing = [col for col in base_features if col not in features_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Apply feature engineering
        df = features_df.copy()
        df = create_rooms_per_household(df)
        df = create_bedrooms_per_room(df)
        df = create_population_per_household(df)
        df = create_income_categories(df)

        # Extract features needed by model
        X = df[self.feature_columns]

        # Scale features
        X_scaled = self.model.scaler.transform(X)

        # Make predictions
        predictions = self.model.model.predict(X_scaled)

        return predictions

    def get_model_metrics(self) -> Dict:
        """
        Get model performance metrics.

        Returns:
            dict: Model evaluation metrics (RMSE, R², MAE)
        """
        return self.model.metrics

    def predict_with_details(self, features: Dict) -> Dict:
        """
        Predict price and return detailed information.

        Args:
            features (dict): Base feature values

        Returns:
            dict: Contains:
                - prediction: Predicted price
                - input_features: Original features
                - engineered_features: Derived features
                - feature_values: Values used for prediction
                - model_metrics: Model performance metrics
        """
        # Validate and engineer features
        self.validate_features(features)
        features_engineered = self.engineer_features(features)

        # Make prediction
        prediction = self.predict_single(features)

        # Prepare detailed result
        result = {
            'prediction': prediction,
            'prediction_formatted': f"${prediction:,.2f}",
            'input_features': features,
            'engineered_features': {
                'rooms_per_household': features_engineered.get('rooms_per_household'),
                'bedrooms_per_room': features_engineered.get('bedrooms_per_room'),
                'population_per_household': features_engineered.get('population_per_household'),
                'income_category': features_engineered.get('income_category')
            },
            'model_metrics': self.get_model_metrics()
        }

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            pd.DataFrame: Features with their coefficients
        """
        return self.model.get_feature_importance()


def create_sample_prediction():
    """
    Create a sample prediction for testing.

    Returns:
        dict: Sample prediction result
    """
    # Sample house features (median values from California housing dataset)
    sample_features = {
        'longitude': -122.23,
        'latitude': 37.88,
        'housing_median_age': 28.0,
        'total_rooms': 2000.0,
        'total_bedrooms': 400.0,
        'population': 1200.0,
        'households': 400.0,
        'median_income': 3.5
    }

    print("Creating sample prediction...")
    print("\nSample input features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")

    try:
        predictor = PredictionInterface()
        result = predictor.predict_with_details(sample_features)

        print("\nPrediction result:")
        print(f"  Predicted house value: {result['prediction_formatted']}")
        print("\nEngineered features:")
        for key, value in result['engineered_features'].items():
            print(f"  {key}: {value}")

        return result

    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the prediction interface
    create_sample_prediction()
