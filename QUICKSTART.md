# Quick Start Guide

## Getting Started with California Housing Prediction

This guide will help you set up and run the project from scratch.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

---

## Step-by-Step Setup

### 1. Clone the Repository (if not already done)

```bash
cd California-housing-prediction
```

### 2. Create and Activate Virtual Environment

**Linux/Mac/WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualizations)
- streamlit (web dashboard)
- jupyter (notebooks)
- python-dotenv (configuration)

### 4. Configure Environment Variables

```bash
# Linux/Mac/WSL
cp .env.example .env

# Windows Command Prompt
copy .env.example .env
```

Edit `.env` if you need to customize any settings (optional).

### 5. Run the Data Pipeline

**This is the most important step!** Run this BEFORE launching the dashboard:

```bash
python pipeline.py
```

This script will:
1. ✓ Load California housing data from sklearn
2. ✓ Clean and preprocess the data
3. ✓ Engineer features (rooms per household, etc.)
4. ✓ Save data to SQLite database
5. ✓ Train the Linear Regression model
6. ✓ Evaluate model performance
7. ✓ Generate visualizations
8. ✓ Save everything for the dashboard

**Expected output:**
- `data/raw/` - Original dataset
- `data/interim/` - Cleaned dataset
- `data/processed/` - Feature-engineered dataset
- `data/housing.db` - SQLite database
- `models/` - Trained model and metrics
- `reports/figures/` - Visualizations

**Time:** ~30-60 seconds depending on your machine

### 6. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Using the Dashboard

Once the dashboard is running, you can:

- **View Data Statistics**: See summary statistics and data distribution
- **Explore Visualizations**: View correlation heatmaps, geographic plots, etc.
- **Make Predictions**: Input housing features and get price predictions
- **Query Database**: Run SQL queries on the housing data
- **Analyze Model Performance**: View RMSE, R², MAE metrics

---

## Jupyter Notebooks (Optional)

To explore the data interactively:

```bash
jupyter notebook notebooks/
```

Open `1.0-exploratory-data-analysis.ipynb` to see the EDA process.

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Verify you're in the project root directory

### Pipeline Fails

If `pipeline.py` fails:
- Check that all directories exist (they're auto-created)
- Ensure you have write permissions
- Check available disk space

### Streamlit Won't Start

If the dashboard doesn't launch:
- Ensure `pipeline.py` ran successfully first
- Check that `data/housing.db` and `models/` exist
- Try: `streamlit run app.py --server.port 8502` (different port)

### Database Issues

If you see database errors:
- Delete `data/housing.db` and run `pipeline.py` again
- Check file permissions on `data/` directory

---

## Project Structure

```
├── pipeline.py          # ⭐ Main data pipeline (RUN THIS FIRST)
├── app.py               # Streamlit dashboard
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
│
├── data/
│   ├── raw/            # Original data
│   ├── interim/        # Cleaned data
│   ├── processed/      # Feature-engineered data
│   └── housing.db      # SQLite database
│
├── models/             # Trained models
├── reports/figures/    # Visualizations
├── notebooks/          # Jupyter notebooks
│
└── src/
    ├── config.py       # Configuration
    ├── dataset.py      # Data processing
    ├── features.py     # Feature engineering
    ├── plots.py        # Visualizations
    ├── modeling/       # ML models
    └── services/       # Database operations
```

---

## Complete Workflow

```bash
# 1. Setup (one time)
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env

# 2. Run pipeline (one time, or when you want to retrain)
python pipeline.py

# 3. Launch dashboard (every time you want to use it)
streamlit run app.py

# 4. Explore notebooks (optional)
jupyter notebook notebooks/
```

---

## Next Steps

- Modify hyperparameters in `src/config.py`
- Add custom features in `src/features.py`
- Train different models in `src/modeling/train.py`
- Customize dashboard in `app.py`

---

## Need Help?

- Check the main `README.md` for project overview
- Review code documentation in each module
- Examine `src/config.py` for configuration options
