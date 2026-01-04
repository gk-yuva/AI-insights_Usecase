# ML Portfolio Recommender — README

This workspace contains an experimental Streamlit app `dashboard_ml.py` that demonstrates a simple ML-enhanced portfolio recommender.

Quick start

1. Activate your virtual environment (Windows PowerShell):

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

2. Install dependencies (if not already installed):

```powershell
F:\AI Insights Dashboard\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional: install lightgbm for faster models
F:\AI Insights Dashboard\.venv\Scripts\python.exe -m pip install lightgbm
```

3. Run the ML dashboard:

```powershell
cd "F:\AI Insights Dashboard"
F:\AI Insights Dashboard\.venv\Scripts\python.exe -m streamlit run dashboard_ml.py
```

What the app does

- Loads IID (investor profile) JSON uploaded via sidebar
- Loads historical price CSVs (one file per asset) uploaded via sidebar
- Optionally loads India VIX from `C:\Users\USER\trading\india_vix_history.csv` if present
- Simulates random portfolios to create training data, trains regressors (LightGBM if installed, otherwise RandomForest)
- Runs a constrained optimizer (differential evolution) to find a portfolio allocation that balances PQS and IFS
- Supports uploading a symbol->sector mapping CSV (columns `symbol,sector`) to enforce sector exposure penalties

Notes & limitations

- This is experimental: validate outputs before using for live decisions.
- Training time depends on the number of simulations and asset count.
- The optimizer uses a penalty approach for sector constraints; a hard-constrained optimization or MOEA algorithm can be added later.

Files added/modified

- `dashboard_ml.py` — ML-enhanced copy of `dashboard.py`
- `README_ML.md` — this file
- `SECTOR_CONSTRAINTS_REMINDER.md` — reminder to add sector constraints later

Next steps

- Add stricter sector-exposure enforcement in the optimizer (hard constraints)
- Persist trained models and reuse instead of re-training each session
- Add pre-built asset-universe loader for Nifty50+Next50
- Collect user action/outcome data for ML model improvement

