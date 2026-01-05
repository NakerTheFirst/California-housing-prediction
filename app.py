"""
California Housing Price Prediction - Streamlit Dashboard.

This is the main entry point for the interactive web application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import (
    STREAMLIT_TITLE, STREAMLIT_LAYOUT, STREAMLIT_SIDEBAR_STATE,
    PROCESSED_DATA_PATH, FIGURES_DIR, DATABASE_PATH,
    MIN_LONGITUDE, MAX_LONGITUDE, MIN_LATITUDE, MAX_LATITUDE,
    MIN_HOUSING_AGE, MAX_HOUSING_AGE, MIN_INCOME, MAX_INCOME
)
from src.services.database import DatabaseManager
from src.modeling.predict import PredictionInterface


# Page configuration
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state=STREAMLIT_SIDEBAR_STATE,
    page_icon="🏠"
)


# Caching functions for performance
@st.cache_data
def load_processed_data():
    """Load processed housing data."""
    try:
        if PROCESSED_DATA_PATH.exists():
            data = pd.read_csv(PROCESSED_DATA_PATH)
            return data
        else:
            st.error(f"Processed data not found at {PROCESSED_DATA_PATH}")
            st.info("Please run the data processing pipeline first.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def load_trained_model():
    """Load trained prediction model."""
    try:
        predictor = PredictionInterface()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please train the model first by running src/modeling/train.py")
        return None


@st.cache_resource
def get_database_connection():
    """Get database connection."""
    try:
        db = DatabaseManager()
        return db
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None


def show_home():
    """Display home page with project overview."""
    st.title("🏠 " + STREAMLIT_TITLE)

    st.markdown("""
    ## Welcome to the California Housing Price Prediction System

    This interactive dashboard provides comprehensive analysis and price prediction
    for California housing data using machine learning.

    ### Project Overview

    **Objective**: Predict median house prices in California districts based on
    demographic and housing characteristics.

    **Dataset**: California Housing Dataset from scikit-learn
    - **20,640** housing districts
    - **8** original features + engineered features
    - Target variable: Median house value

    ### Key Features

    📊 **Data Exploration**: Interactive data tables and SQL query demonstrations

    📈 **Visualizations**: 10 different chart types for exploratory data analysis

    🎯 **Price Prediction**: Real-time house price predictions using Linear Regression

    📉 **Model Performance**: Evaluation metrics and performance visualizations
    """)

    # Load data for statistics
    data = load_processed_data()

    if data is not None:
        st.markdown("---")
        st.subheader("📋 Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(data):,}")

        with col2:
            st.metric("Features", f"{len(data.columns)-1}")

        with col3:
            mean_price = data['median_house_value'].mean()
            st.metric("Average Price", f"${mean_price:,.0f}")

        with col4:
            median_price = data['median_house_value'].median()
            st.metric("Median Price", f"${median_price:,.0f}")

        # Show feature list
        st.markdown("### Available Features")
        features = [col for col in data.columns if col != 'median_house_value']
        st.write(", ".join(features))

        st.markdown("---")
        st.info("👈 Use the sidebar to navigate to different sections of the dashboard")


def show_data_exploration():
    """Display data exploration tools and SQL demonstrations."""
    st.title("📊 Data Exploration")

    data = load_processed_data()

    if data is None:
        return

    # Interactive data table
    st.subheader("🔍 Interactive Data Table")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        income_range = st.slider(
            "Filter by Median Income",
            float(data['median_income'].min()),
            float(data['median_income'].max()),
            (float(data['median_income'].min()), float(data['median_income'].max()))
        )

    with col2:
        price_range = st.slider(
            "Filter by House Value",
            float(data['median_house_value'].min()),
            float(data['median_house_value'].max()),
            (float(data['median_house_value'].min()), float(data['median_house_value'].max()))
        )

    # Apply filters
    filtered_data = data[
        (data['median_income'] >= income_range[0]) &
        (data['median_income'] <= income_range[1]) &
        (data['median_house_value'] >= price_range[0]) &
        (data['median_house_value'] <= price_range[1])
    ]

    st.write(f"Showing {len(filtered_data):,} of {len(data):,} records")
    st.dataframe(filtered_data.head(100), use_container_width=True)

    st.markdown("---")

    # Summary statistics
    st.subheader("📈 Summary Statistics")

    if st.checkbox("Show detailed statistics"):
        st.write(filtered_data.describe())

    # Missing values check
    st.subheader("🔍 Data Quality")
    missing_values = filtered_data.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.warning("⚠️ Missing values detected:")
        st.write(missing_values[missing_values > 0])

    st.markdown("---")

    # SQL Demonstrations
    st.subheader("💾 SQL Query Demonstrations")

    db = get_database_connection()

    if db is None:
        st.warning("Database not available. Please run the data processing pipeline first.")
        return

    # Check if database has data
    try:
        housing_count = db.get_table_count('housing')
        if housing_count == 0:
            st.warning("Database is empty. Please populate it by running the data processing pipeline.")
            return
    except:
        st.warning("Database tables not created yet.")
        return

    tab1, tab2, tab3 = st.tabs(["WHERE Filtering", "GROUP BY Aggregation", "INNER JOIN"])

    with tab1:
        st.markdown("#### WHERE Clause Example")
        st.code("""
SELECT *
FROM housing
WHERE median_income >= ? AND median_income <= ?
        """, language="sql")

        col1, col2 = st.columns(2)
        with col1:
            min_income = st.number_input("Min Income", value=3.0, step=0.5)
        with col2:
            max_income = st.number_input("Max Income", value=6.0, step=0.5)

        if st.button("Run WHERE Query"):
            try:
                result = db.filter_by_income(min_income, max_income)
                st.write(f"Found {len(result)} records")
                st.dataframe(result.head(20), use_container_width="strech")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("#### GROUP BY Aggregation Example")
        st.code("""
SELECT
    income_category,
    AVG(median_house_value) as avg_house_value,
    AVG(rooms_per_household) as avg_rooms,
    COUNT(*) as count_districts
FROM housing
GROUP BY income_category
ORDER BY avg_house_value DESC
        """, language="sql")

        if st.button("Run GROUP BY Query"):
            try:
                result = db.aggregate_by_income_category()
                st.dataframe(result, use_container_width="strech")

                # Visualize results
                st.bar_chart(result.set_index('income_category')['avg_house_value'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.markdown("#### INNER JOIN Example")
        st.code("""
SELECT
    h.id, h.longitude, h.latitude,
    h.median_income, h.median_house_value,
    h.income_category,
    ds.avg_house_value as district_avg_value,
    ds.district_count
FROM housing h
INNER JOIN district_summary ds
    ON h.income_category = ds.income_category
LIMIT ?
        """, language="sql")

        limit = st.slider("Number of records to fetch", 10, 500, 100)

        if st.button("Run JOIN Query"):
            try:
                result = db.join_housing_with_summary(limit=limit)
                st.write(f"Joined {len(result)} records")
                st.dataframe(result, width="stretch")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Database statistics
    st.markdown("---")
    st.subheader("📊 Database Statistics")

    try:
        stats = db.get_statistics()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Housing Records", f"{stats['housing_count']:,}")
        with col2:
            st.metric("Summary Records", f"{stats['summary_count']:,}")
        with col3:
            st.metric("Database Size", f"{stats['database_size']:.2f} MB")

    except Exception as e:
        st.error(f"Error getting database statistics: {str(e)}")


def show_visualizations():
    """Display all visualizations."""
    st.title("📈 Visualizations Gallery")

    st.markdown("""
    This section displays 10 different types of visualizations for exploratory data analysis.
    All charts are generated from the processed California housing dataset.
    """)

    # Check if visualizations exist
    if not FIGURES_DIR.exists():
        st.error(f"Figures directory not found: {FIGURES_DIR}")
        st.info("Please run the EDA pipeline to generate visualizations.")
        return

    # Get all PNG files from figures directory
    figure_files = sorted(list(FIGURES_DIR.glob("*.png")))

    if len(figure_files) == 0:
        st.warning("No visualizations found. Please generate them first using EDAAnalyser.")
        st.info("Run the data processing and EDA pipeline to create visualizations.")
        return

    st.success(f"Found {len(figure_files)} visualizations")

    # Display visualizations in tabs
    viz_tabs = st.tabs([
        "Overview", "Distributions", "Relationships",
        "Correlations", "Comparisons", "Geographic"
    ])

    with viz_tabs[0]:
        st.subheader("📊 All Visualizations")
        st.markdown("Browse through all generated visualizations below:")

        # Display all figures in a grid
        for i, fig_path in enumerate(figure_files, 1):
            st.markdown(f"### {i}. {fig_path.stem.replace('_', ' ').title()}")
            st.image(str(fig_path), width="stretch")
            st.markdown("---")

    with viz_tabs[1]:
        st.subheader("📊 Distribution Plots")
        # Show histogram, boxplot, violin
        for pattern in ['histogram', 'boxplot', 'violin', 'density']:
            matching = [f for f in figure_files if pattern in f.name.lower()]
            for fig_path in matching:
                st.image(str(fig_path), width="stretch")

    with viz_tabs[2]:
        st.subheader("🔗 Relationship Plots")
        # Show scatter, pairplot, line
        for pattern in ['scatter', 'pairplot', 'line']:
            matching = [f for f in figure_files if pattern in f.name.lower()]
            for fig_path in matching:
                st.image(str(fig_path), width="stretch")

    with viz_tabs[3]:
        st.subheader("🔥 Correlation Analysis")
        # Show heatmap
        matching = [f for f in figure_files if 'heatmap' in f.name.lower() or 'correlation' in f.name.lower()]
        for fig_path in matching:
            st.image(str(fig_path), width="stretch")

    with viz_tabs[4]:
        st.subheader("📊 Comparison Charts")
        # Show bar charts
        matching = [f for f in figure_files if 'bar' in f.name.lower()]
        for fig_path in matching:
            st.image(str(fig_path), width="stretch")

    with viz_tabs[5]:
        st.subheader("🗺️ Geographic Visualization")
        # Show geographic scatter
        matching = [f for f in figure_files if 'geographic' in f.name.lower()]
        for fig_path in matching:
            st.image(str(fig_path), width="stretch")

    # Model performance plots
    st.markdown("---")
    st.subheader("🎯 Model Performance Visualizations")

    model_plots = [f for f in figure_files if 'model' in f.name.lower() or 'predictions' in f.name.lower() or 'residuals' in f.name.lower()]
    if len(model_plots) > 0:
        for fig_path in model_plots:
            st.image(str(fig_path), width="stretch")
    else:
        st.info("Model performance plots will appear here after training the model.")


def show_prediction_interface():
    """Display interactive prediction form."""
    st.title("🎯 House Price Prediction")

    st.markdown("""
    Enter the characteristics of a house to predict its median value.
    The model uses Linear Regression trained on California housing data.
    """)

    # Load model
    predictor = load_trained_model()

    if predictor is None:
        return

    # Display model metrics
    with st.expander("📊 Model Performance Metrics"):
        try:
            metrics = predictor.get_model_metrics()
            col1, col2, col3 = st.columns(3)

            with col1:
                rmse = metrics.get('rmse', 0)
                st.metric("RMSE", f"${rmse:,.0f}")

            with col2:
                r2 = metrics.get('r2', 0)
                st.metric("R² Score", f"{r2:.4f}")

            with col3:
                mae = metrics.get('mae', 0)
                st.metric("MAE", f"${mae:,.0f}")

            st.info("""
            **RMSE**: Root Mean Squared Error - Average prediction error in dollars.
            **R²**: Coefficient of determination - Proportion of variance explained (0-1).
            **MAE**: Mean Absolute Error - Average absolute prediction error.
            """)

        except Exception as e:
            st.warning(f"Could not load model metrics: {str(e)}")

    st.markdown("---")

    # Input form
    st.subheader("🏠 Enter House Characteristics")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Geographic Location**")
            longitude = st.slider(
                "Longitude",
                MIN_LONGITUDE, MAX_LONGITUDE,
                -122.0,
                step=0.1
            )

            latitude = st.slider(
                "Latitude",
                MIN_LATITUDE, MAX_LATITUDE,
                37.5,
                step=0.1
            )

            st.markdown("**House Details**")
            housing_age = st.number_input(
                "Housing Median Age (years)",
                min_value=MIN_HOUSING_AGE,
                max_value=MAX_HOUSING_AGE,
                value=28,
                step=1
            )

            median_income = st.slider(
                "Median Income (tens of thousands)",
                MIN_INCOME, MAX_INCOME,
                3.5,
                step=0.1
            )

        with col2:
            st.markdown("**Property Characteristics**")
            total_rooms = st.number_input(
                "Total Rooms",
                min_value=1,
                max_value=50000,
                value=2000,
                step=100
            )

            total_bedrooms = st.number_input(
                "Total Bedrooms",
                min_value=1,
                max_value=10000,
                value=400,
                step=50
            )

            population = st.number_input(
                "Population",
                min_value=1,
                max_value=40000,
                value=1200,
                step=100
            )

            households = st.number_input(
                "Households",
                min_value=1,
                max_value=10000,
                value=400,
                step=50
            )

        submitted = st.form_submit_button("🔮 Predict House Price", width="stretch")

    if submitted:
        # Create features dictionary
        features = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income
        }

        # Make prediction
        try:
            with st.spinner("Making prediction..."):
                result = predictor.predict_with_details(features)

            st.success("✅ Prediction completed!")

            # Display prediction
            st.markdown("---")
            st.subheader("💰 Predicted House Value")

            prediction = result['prediction']
            st.markdown(f"## ${prediction:,.2f}")

            # Calculate comparison with average
            data = load_processed_data()
            if data is not None:
                avg_price = data['median_house_value'].mean()
                diff = prediction - avg_price
                diff_pct = (diff / avg_price) * 100

                if diff > 0:
                    st.success(f"📈 {diff_pct:+.1f}% above average (${avg_price:,.0f})")
                else:
                    st.info(f"📉 {diff_pct:+.1f}% below average (${avg_price:,.0f})")

            # Show engineered features
            with st.expander("🔧 Auto-Calculated Features"):
                eng_features = result['engineered_features']
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Rooms per Household",
                             f"{eng_features['rooms_per_household']:.2f}")
                    st.metric("Bedrooms per Room",
                             f"{eng_features['bedrooms_per_room']:.2f}")

                with col2:
                    st.metric("Population per Household",
                             f"{eng_features['population_per_household']:.2f}")
                    st.metric("Income Category",
                             eng_features['income_category'])

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your input values and try again.")


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("🏠 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Exploration", "📈 Visualizations", "🎯 Price Prediction"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About

    **California Housing Price Prediction**

    A complete end-to-end ML pipeline with:
    - Data processing & cleaning
    - Feature engineering
    - SQL integration
    - 10 visualization types
    - Linear regression modeling
    - Interactive predictions

    Built with Python, scikit-learn, and Streamlit
    """)

    # Route to appropriate page
    if "Home" in page:
        show_home()
    elif "Data Exploration" in page:
        show_data_exploration()
    elif "Visualizations" in page:
        show_visualizations()
    elif "Price Prediction" in page:
        show_prediction_interface()


if __name__ == "__main__":
    main()
