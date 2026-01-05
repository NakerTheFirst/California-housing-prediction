# How to Run the Project

## TL;DR - Quick Commands

```bash
# 1. Activate virtual environment
source .venv/bin/activate          # Linux/Mac/WSL
# OR
.venv\Scripts\activate             # Windows

# 2. Run the data pipeline (REQUIRED - first time only)
python pipeline.py

# 3. Launch the dashboard
streamlit run app.py
```

---

## Detailed Steps

### First Time Setup

1. **Activate your virtual environment:**
   ```bash
   source .venv/bin/activate  # Linux/Mac/WSL
   ```
   or
   ```bash
   .venv\Scripts\activate  # Windows
   ```

2. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data pipeline:**
   ```bash
   python pipeline.py
   ```

   This will:
   - Load California housing data from sklearn
   - Clean and preprocess the data
   - Create engineered features
   - Save data to SQLite database (`data/housing.db`)
   - Train the Linear Regression model
   - Save the trained model to `models/`
   - Generate visualizations in `reports/figures/`
   - Show model performance metrics

   **Expected runtime:** 30-60 seconds

   **Output files created:**
   ```
   data/
   ├── raw/housing_raw.csv
   ├── interim/housing_cleaned.csv
   ├── processed/housing_processed.csv
   └── housing.db

   models/
   ├── linear_regression_model.pkl
   ├── scaler.pkl
   └── evaluation_metrics.json

   reports/figures/
   ├── histogram_*.png
   ├── correlation_heatmap.png
   ├── scatter_*.png
   └── ... (more visualizations)
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

   The dashboard will open automatically at `http://localhost:8501`

---

## Using the Dashboard

Once the Streamlit app is running, you can:

1. **Home Tab**: View project overview and model metrics
2. **Data Explorer**: Browse the housing dataset
3. **Visualizations**: View all generated plots
4. **Make Predictions**: Input features and predict house prices
5. **SQL Queries**: Run custom queries on the database

---

## When to Re-run the Pipeline

You need to run `python pipeline.py` again if you:

- Modify feature engineering logic in `src/features.py`
- Change model hyperparameters in `src/config.py`
- Delete or corrupt files in `data/` or `models/`
- Want to regenerate visualizations
- Want to retrain the model with different settings

---

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'pandas'`
**Solution:**
```bash
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError` when running `streamlit run app.py`
**Solution:**
```bash
python pipeline.py  # Run pipeline first!
```

**Problem:** Dashboard shows empty data
**Solution:**
```bash
# Delete existing data and rerun pipeline
rm -rf data/*.db models/*.pkl  # Linux/Mac
del data\*.db models\*.pkl     # Windows
python pipeline.py
```

**Problem:** Port already in use
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

---

## Project Workflow Diagram

```
[1] pipeline.py
    ↓
    ├─→ Load data from sklearn
    ├─→ Clean & engineer features
    ├─→ Save to CSV files (data/)
    ├─→ Import to SQLite (data/housing.db)
    ├─→ Train model
    ├─→ Save model (models/)
    └─→ Generate plots (reports/figures/)

[2] app.py (Streamlit)
    ↓
    ├─→ Read from data/housing.db
    ├─→ Load model from models/
    ├─→ Display visualizations
    └─→ Accept user inputs for predictions
```

---

## Development Workflow

```bash
# Daily workflow
source .venv/bin/activate
streamlit run app.py

# If you modify code
python pipeline.py        # Regenerate data/models
streamlit run app.py      # View updated results

# Explore data interactively
jupyter notebook notebooks/
```

---

## Summary

**First time:**
1. `python pipeline.py` ← Initialize everything
2. `streamlit run app.py` ← View dashboard

**Every other time:**
1. `streamlit run app.py` ← Just launch the dashboard

**Re-initialize only if needed:**
1. `python pipeline.py` ← Reprocess data/retrain model
