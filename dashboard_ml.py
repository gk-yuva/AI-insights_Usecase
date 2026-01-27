"""
ML-enhanced Portfolio Recommender Dashboard (copy of dashboard.py + ML)

Features:
- Loads IID (investor profile) and historical price data
- Simulates random portfolios from the asset universe and trains LightGBM/RandomForest regressors
  to predict portfolio metrics (return, volatility, drawdown, VaR)
- Uses a differential evolution optimizer to find a portfolio that maximizes a blend of PQS and IFS
- Respects provided portfolio constraints

Notes:
- Training can be slow depending on asset count and simulation size. Models are cached.
- If LightGBM is not installed, falls back to RandomForestRegressor (scikit-learn).
- Attempts to load VIX from C:\\Users\\USER\\trading\\india_vix_history.csv if present.

Run with: `streamlit run dashboard_ml.py`
"""

import streamlit as st
from streamlit import tabs
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict

from portfolio_quality import PortfolioQualityScorer
from investor_fit import InvestorFitScorer
diagnostics_msgs = []
# ML imports (optional)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from scipy.optimize import differential_evolution, minimize
import yfinance as yf
from pathlib import Path

st.set_page_config(page_title="ML Portfolio Recommender", layout="wide")

# ------------------ Helper functions ------------------

def load_vix(path=r"C:\Users\USER\trading\india_vix_history.csv") -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=True)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def annualize_returns(returns, periods_per_year=252):
    mean = returns.mean() * periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)
    return mean, vol


def max_drawdown(returns_series: pd.Series) -> float:
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(abs(drawdown.min()) * 100)


def var_historical(returns_series: pd.Series, level=0.05) -> float:
    return float(abs(np.percentile(returns_series, level*100)) * 100)


@st.cache_data(show_spinner=False)
def load_price_data(paths: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Load price data CSVs provided by user. Returns dict of symbol -> price series (adj close).
    Expects CSV with Date and Adj Close or Close columns."""
    data = {}
    if not paths:
        return data
    for p in paths:
        if os.path.exists(p):
            try:
                ext = os.path.splitext(p)[1].lower()
                if ext in ['.csv']:
                    df = pd.read_csv(p, parse_dates=[0], index_col=0)
                elif ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(p, parse_dates=[0], index_col=0)
                else:
                    # unsupported
                    continue
                # Try common column names
                col = None
                for c in ['Adj Close', 'Adj_Close', 'AdjClose', 'Close', 'close']:
                    if c in df.columns:
                        col = c
                        break
                if col is None:
                    # try first numeric column
                    numeric_cols = df.select_dtypes(include=[float, int]).columns
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                    else:
                        continue
                series = df[col].dropna()
                name = os.path.splitext(os.path.basename(p))[0]
                data[name] = series
            except Exception:
                continue
    return data


def compute_portfolio_metrics(weights: np.ndarray, price_series: Dict[str, pd.Series]) -> Dict:
    # weights aligned with price_series order
    symbols = list(price_series.keys())
    
    # Check if we have data
    if not symbols or len(symbols) == 0:
        return {
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0
        }
    
    prices = pd.concat([price_series[symbol].rename(symbol) for symbol in symbols], axis=1).dropna()
    
    # Debug: show data alignment info
    st.write(f"**Debug: Price data alignment**")
    st.write(f"  - Assets: {len(symbols)}")
    for sym in symbols:
        st.write(f"  - {sym}: {len(price_series[sym])} dates ({price_series[sym].index[0].date()} to {price_series[sym].index[-1].date()})")
    st.write(f"  - After dropna: {len(prices)} overlapping dates")
    if len(prices) > 0:
        st.write(f"  - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # If dropna leaves too few dates, try forward/backward fill
    if len(prices) < 50 and len(symbols) > 1:
        st.write(f"  ⚠️ Only {len(prices)} overlapping dates, trying fill methods...")
        prices_filled = pd.concat([price_series[symbol].rename(symbol) for symbol in symbols], axis=1)
        
        # Forward fill then backward fill to handle missing data
        prices_filled = prices_filled.fillna(method='ffill').fillna(method='bfill')
        prices_filled = prices_filled.dropna()  # Remove any remaining NaN rows
        
        if len(prices_filled) >= 50:
            st.write(f"  ✓ After filling: {len(prices_filled)} dates available")
            prices = prices_filled
        else:
            st.write(f"  ✗ Still only {len(prices_filled)} dates after filling")
    
    # Check if we have enough data after dropna
    if len(prices) < 50:
        return {
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0
        }
    
    rets = prices.pct_change().dropna()
    weights = np.array(weights)
    port_ret = (rets @ weights)
    ann_ret, ann_vol = annualize_returns(port_ret)
    sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(port_ret)
    var95 = var_historical(port_ret, 0.05)
    var99 = var_historical(port_ret, 0.01)
    return {
        'expected_return': ann_ret * 100,
        'expected_volatility': ann_vol * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'var_95': var95,
        'var_99': var99
    }


def simulate_random_portfolios(price_series: Dict[str, pd.Series], n: int = 2000, min_stocks=5, max_stocks=25) -> Dict:
    """Simulate random portfolios and compute metrics for training data."""
    symbols = list(price_series.keys())
    m = len(symbols)
    X = []
    y = []

    # If no assets available, return empty training set
    if m == 0:
        return {'X': np.empty((0, 0)), 'y': pd.DataFrame(), 'symbols': symbols}

    # Clamp min/max to available assets
    min_k = max(1, min(min_stocks, m))
    max_k = max(1, min(max_stocks, m))
    
    # Debug: show first few computed metrics
    debug_shown = False

    for i in range(n):
        # choose a portfolio size k in [min_k, max_k]
        if min_k == max_k:
            k = min_k
        else:
            k = np.random.randint(min_k, max_k + 1)

        # sample k distinct assets
        picks = np.random.choice(m, size=k, replace=False)
        raw = np.random.random(k)
        w = np.zeros(m)
        w[picks] = raw / raw.sum()
        metrics = compute_portfolio_metrics(w, price_series)
        
        # Show first computed metric to diagnose
        if not debug_shown and i == 0:
            st.write(f"  🔍 First portfolio metrics: {metrics}")
            debug_shown = True
        
        # Also show a few more examples
        if not debug_shown and i in [10, 50, 100]:
            st.write(f"  🔍 Portfolio {i} metrics: {metrics}")
            if i == 100:
                debug_shown = True
        
        X.append(w)
        y.append(metrics)
    # Convert
    X = np.array(X)
    y_df = pd.DataFrame(y)
    return {'X': X, 'y': y_df, 'symbols': symbols}


class SimpleRegressors:
    """Holds regressors for each predicted metric."""
    def __init__(self):
        self.models = {}
        self.features = None

    def fit(self, X: np.ndarray, y: pd.DataFrame):
        # X shape (n_samples, n_assets)
        # Fit one model per column in y
        st.write(f"🔧 Training ML models on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Validate training data
        if X.shape[0] < 10:
            st.error(f"❌ Insufficient training samples: {X.shape[0]}")
            return
            
        for col in y.columns:
            y_col = y[col].values
            
            # Check for variance in target variable
            if np.std(y_col) < 1e-6:
                st.warning(f"⚠️ No variance in {col}: min={y_col.min():.4f}, max={y_col.max():.4f}")
            
            # Prefer scikit-learn RandomForest if available, else fall back to LightGBM
            if SKLEARN_AVAILABLE:
                try:
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                    model.fit(X, y_col)

                    # Validate model learned something
                    train_pred = model.predict(X[:10])
                    st.write(f"  ✓ {col}: RandomForest trained, sample predictions: {train_pred[:3]}")
                except Exception as e:
                    st.error(f"❌ RandomForest training failed for {col}: {e}")
                    model = None
            elif LGB_AVAILABLE:
                try:
                    dtrain = lgb.Dataset(X, label=y_col)
                    params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'num_leaves': 31, 'learning_rate': 0.05}
                    model = lgb.train(params, dtrain, num_boost_round=200)

                    # Validate model learned something
                    train_pred = model.predict(X[:10])
                    st.write(f"  ✓ {col}: LightGBM trained, sample predictions: {train_pred[:3]}")
                except Exception as e:
                    st.error(f"❌ LightGBM training failed for {col}: {e}")
                    model = None
            else:
                model = None
            self.models[col] = model
        self.features = X.shape[1]

    def predict(self, w: np.ndarray) -> Dict:
        X = np.atleast_2d(w)
        out = {}
        for k, m in self.models.items():
            if m is None:
                out[k] = 0.0
            else:
                if LGB_AVAILABLE:
                    out[k] = float(m.predict(X)[0])
                else:
                    out[k] = float(m.predict(X)[0])
        return out


# ------------------ Streamlit UI ------------------

st.title("ML Portfolio Recommender")

with st.sidebar:
    st.header("Data Inputs")
    iid_file = st.file_uploader("Upload IID (investor profile JSON)", type=['json'])
    price_files = st.file_uploader("Upload price CSVs for candidate assets (one per file)", type=['csv'], accept_multiple_files=True)
    vix_auto = st.checkbox("Load India VIX from C:\\Users\\USER\\trading\\india_vix_history.csv (if present)", value=True)
    if vix_auto:
        vix_df = load_vix()
        if vix_df.empty:
            st.info("VIX file not found at default path; you can upload manually")
            vix_upload = st.file_uploader("Upload india_vix_history.csv", type=['csv'])
            if vix_upload:
                vix_df = pd.read_csv(vix_upload)
    else:
        vix_df = pd.DataFrame()

    st.markdown("---")
    st.header("Model & Simulation")
    n_sim = st.number_input("Simulation count for training (lower = faster)", min_value=100, max_value=10000, value=2000, step=100)
    train_button = st.button("Train ML Models")
    st.markdown("---")
    st.header("Sector Mapping")
    sector_file = st.file_uploader("Upload symbol->sector mapping CSV (columns: symbol,sector)", type=['csv'])
    st.markdown("---")
    st.header("Prebuilt Asset Loader")
    st.write("Load Nifty50 + Next50 asset universe from workspace or download by tickers.")
    load_prebuilt = st.button("Load Nifty50 + Next50 (local/data or tickers files)")

    st.markdown("---")
    st.header("Optimization")
    #blend = st.slider("PQS vs IFS blend (alpha for PQS)", 0.0, 1.0, 0.5)
    objective_choice = st.selectbox("Investment Objective", ['Moderate Growth', 'Conservative Income', 'Aggressive Growth', 'Balanced'])
    run_opt = st.button("Recommend Portfolio")
    st.markdown("**Optimization Criteria:** Maximization of Sharpe Ratio")

# Load IID
if iid_file:
    import json
    iid = json.load(iid_file)
    st.write("Loaded IID")
else:
    # Try to load IID saved by `dashboard.py`
    iid = None
    try:
        iid_path = Path('data') / 'latest_iid.json'
        if iid_path.exists():
            import json
            with open(iid_path, 'r') as f:
                iid = json.load(f)
            st.write("Loaded IID from data/latest_iid.json")
    except Exception:
        iid = None
# If IID loaded from disk, show key summary in sidebar and set as session
if iid is not None:
    try:
        st.sidebar.success("IID loaded from data/latest_iid.json")
        ip = iid.get('investor_profile') if isinstance(iid, dict) else iid.get('investor_profile', {})
        # if pandas read returned DataFrame/Series, convert
        if not isinstance(ip, dict):
            try:
                ip = dict(ip)
            except Exception:
                ip = {}
        st.sidebar.write("**Investor:**", ip.get('age_band', ''), ip.get('employment_type', ''))
        st.sidebar.write("**Industry:**", ip.get('industry', ''))
        # Add IID tab
        tab_names = ["Dashboard", "IID"]
        tabs_obj = st.tabs(tab_names)
        with tabs_obj[1]:
            st.subheader("Investor Profile (IID)")
            tab_names = ["Dashboard", "IID"]
            tabs_obj = st.tabs(tab_names)
            with tabs_obj[1]:
                st.subheader("Investor Profile (IID)")
                if not isinstance(iid, dict):
                    try:
                        iid_dict = iid.to_dict()
                    except Exception:
                        iid_dict = {"ERROR": "Could not convert IID to dict"}
                    st.json(iid_dict)
                else:
                    st.json(iid)
    except Exception:
        pass

# Load price data
price_series = {}
data_source = "none"

if price_files:
    paths = []
    for f in price_files:
        # save to temp file
        tmp = os.path.join(".", f.name)
        with open(tmp, 'wb') as fh:
            fh.write(f.getbuffer())
        paths.append(tmp)
    price_series = load_price_data(paths)
    data_source = "uploaded_csv"
    st.success(f"✅ Using uploaded CSV files for training and recommendations")
    st.write(f"Loaded {len(price_series)} assets from uploaded CSVs")
    if price_series:
        st.write(f"  📊 Symbols: {', '.join(list(price_series.keys())[:20])}")
        # Show data quality
        for sym, series in list(price_series.items())[:3]:
            st.write(f"  - {sym}: {len(series)} days, last price: {series.iloc[-1]:.2f}")
else:
    st.info("Please upload price CSVs for candidate assets (Nifty50 + Next50).")
    # Also try to load ZNifty zip files present in workspace root
    try:
        base = Path('data')
        # extraction handled in load_prebuilt_universe; attempt to call it directly
        loaded = load_prebuilt_universe()
        if loaded:
            price_series.update(loaded)
            data_source = "zip_files"
            st.success(f"Loaded {len(loaded)} assets from prebuilt zip/universe (fallback)")
            st.write(f"  📊 Symbols: {', '.join(list(loaded.keys())[:20])}")
            # Show data quality
            for sym, series in list(loaded.items())[:3]:
                st.write(f"  - {sym}: {len(series)} days, last price: {series.iloc[-1]:.2f}")
    except Exception:
        pass

# Prebuilt loader: look for data/nifty50 and data/next50 or tickers files
def load_prebuilt_universe(base_dir: str = "data") -> Dict[str, pd.Series]:
    base = Path(base_dir)
    series = {}
    # If zip files for Nifty are present in workspace root, extract them to data folders
    try:
        import zipfile
        for zname, sub in [("ZNifty50.zip", "nifty50"), ("ZNiftyNext50.zip", "next50")]:
            zpath = Path(zname)
            out_folder = base / sub
            if zpath.exists():
                try:
                    out_folder.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zpath, 'r') as z:
                        z.extractall(path=out_folder)
                except Exception:
                    pass
    except Exception:
        pass
    # try folders
    for sub in ["nifty50", "next50"]:
        folder = base / sub
        if folder.exists() and folder.is_dir():
            for f in folder.glob("*.csv"):
                try:
                    df = pd.read_csv(f, parse_dates=[0], index_col=0)
                    col = None
                    for c in ['Adj Close', 'Adj_Close', 'AdjClose', 'Close', 'close']:
                        if c in df.columns:
                            col = c
                            break
                    if col is None:
                        continue
                    series[f.stem] = df[col].dropna()
                except Exception:
                    continue
    # if none found, try tickers files
    if not series:
        for tf in ["nifty50_tickers.txt", "next50_tickers.txt"]:
            p = Path(tf)
            if p.exists():
                with open(p, 'r') as fh:
                    ticks = [l.strip() for l in fh if l.strip()]
                # download last 5 years
                if ticks:
                    data = yf.download(ticks, period='5y', progress=False, threads=True)
                    # data may be multi-column; prefer 'Adj Close'
                    if ('Adj Close' in data.columns) or ('AdjClose' in data.columns):
                        adj = data['Adj Close'] if 'Adj Close' in data.columns else data['AdjClose']
                    else:
                        adj = data['Close']
                    for col in adj.columns:
                        series[col] = adj[col].dropna()
                    # save to data folder for future
                    out_folder = base / ("nifty50" if 'nifty50' in tf else 'next50')
                    out_folder.mkdir(parents=True, exist_ok=True)
                    for col in adj.columns:
                        out_csv = out_folder / f"{col}.csv"
                        adj[col].to_csv(out_csv)
    return series
    for sub in ["nifty50", "next50"]:
        folder = base / sub
        if folder.exists() and folder.is_dir():
            for f in folder.glob("*.csv"):
                try:
                    df = pd.read_csv(f, parse_dates=[0], index_col=0)
                    col = None
                    for c in ['Adj Close', 'Adj_Close', 'AdjClose', 'Close', 'close']:
                        if c in df.columns:
                            col = c
                            break
                    if col is None:
                        continue
                    series[f.stem] = df[col].dropna()
                except Exception:
                    continue
    # if none found, try tickers files
    if not series:
        for tf in ["nifty50_tickers.txt", "next50_tickers.txt"]:
            p = Path(tf)
            if p.exists():
                with open(p, 'r') as fh:
                    ticks = [l.strip() for l in fh if l.strip()]
                # download last 5 years
                if ticks:
                    data = yf.download(ticks, period='5y', progress=False, threads=True)
                    # data may be multi-column; prefer 'Adj Close'
                    if ('Adj Close' in data.columns) or ('AdjClose' in data.columns):
                        adj = data['Adj Close'] if 'Adj Close' in data.columns else data['AdjClose']
                    else:
                        adj = data['Close']
                    for col in adj.columns:
                        series[col] = adj[col].dropna()
                    # save to data folder for future
                    out_folder = base / ("nifty50" if 'nifty50' in tf else 'next50')
                    out_folder.mkdir(parents=True, exist_ok=True)
                    for col in adj.columns:
                        out_csv = out_folder / f"{col}.csv"
                        adj[col].to_csv(out_csv)
    return series


def extract_symbol_files_from_zips(symbols: List[str], zip_names: List[str] = ["ZNifty50.zip", "ZNiftyNext50.zip"], out_base: str = "data") -> List[str]:
    """Search the provided zip files for per-symbol xlsx/csv files and extract matching files.
    Returns list of extracted file paths."""
    extracted = []
    base = Path(out_base)
    try:
        import zipfile
        for zname in zip_names:
            zpath = Path(zname)
            if not zpath.exists():
                continue
            with zipfile.ZipFile(zpath, 'r') as zf:
                namelist = zf.namelist()
                for sym in symbols:
                    # look for files that match symbol (case-insensitive) and end with .xlsx/.xls/.csv
                    matches = [n for n in namelist if sym.lower() in Path(n).stem.lower() and Path(n).suffix.lower() in ['.xlsx', '.xls', '.csv']]
                    for m in matches:
                        # determine output folder by zip name
                        sub = 'nifty50' if 'Nifty50' in zname or 'nifty50' in zname.lower() else 'next50'
                        out_folder = base / sub
                        out_folder.mkdir(parents=True, exist_ok=True)
                        out_path = out_folder / Path(m).name
                        try:
                            with zf.open(m) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted.append(str(out_path))
                        except Exception:
                            continue
    except Exception:
        pass
    return extracted

if load_prebuilt:
    st.info("Attempting to load prebuilt universe from data/ or tickers files...")
    loaded = load_prebuilt_universe()
    if loaded:
        price_series.update(loaded)
        st.success(f"Loaded {len(loaded)} assets from prebuilt universe")
    else:
        st.warning("No prebuilt data found. Provide data/nifty50 or tickers files (nifty50_tickers.txt).")

# Display VIX basic info if loaded
if not vix_df.empty:
    st.sidebar.write(f"VIX records: {len(vix_df)}")

# ML model holder
if 'ml_models' not in st.session_state:
    st.session_state['ml_models'] = None

if train_button:
    if not price_series:
        st.error("No price series loaded. Upload CSVs first.")
    else:
        # Check if we have existing models with different data source
        existing_models = st.session_state.get('ml_models')
        existing_source = existing_models.get('data_source') if existing_models else None
        if existing_source and existing_source != data_source:
            st.warning(f"⚠️ Data source changed from {existing_source} to {data_source}. Retraining required.")
        
        st.info("Simulating portfolios and training models. This may take a few minutes.")
        
        # Show what we're training on
        st.write(f"**Training with {len(price_series)} assets:** {', '.join(list(price_series.keys())[:10])}")
        
        # Check if price series have enough data
        min_len = min([len(v) for v in price_series.values()])
        st.write(f"**Min price history length:** {min_len} days")
        
        if min_len < 100:
            st.warning("Price data has less than 100 days - training may be unreliable")
        
        sim = simulate_random_portfolios(price_series, n=int(n_sim))
        
        # Validate simulation results
        if sim['X'].shape[0] == 0:
            st.error("Simulation produced no training data. Check price series.")
        else:
            st.write(f"**Generated {sim['X'].shape[0]} training samples with {sim['X'].shape[1]} assets**")
            st.write(f"**Sample metrics ranges:**")
            for col in sim['y'].columns:
                st.write(f"  - {col}: [{sim['y'][col].min():.2f}, {sim['y'][col].max():.2f}]")
            
            modeler = SimpleRegressors()
            modeler.fit(sim['X'], sim['y'])
            
            # Test prediction on equal weights
            test_w = np.ones(len(sim['symbols'])) / len(sim['symbols'])
            test_pred = modeler.predict(test_w)
            st.write(f"**Test prediction (equal weights):** {test_pred}")
            
            st.session_state['ml_models'] = {'modeler': modeler, 'symbols': sim['symbols'], 'data_source': data_source}
            st.success(f"Training completed and models cached using {data_source.replace('_', ' ')}.")

# If no uploaded price files, attempt to extract per-symbol files from ZNifty zips using latest portfolio
if not price_series:
    try:
        portfolio_path = Path('data') / 'latest_portfolio.json'
        if portfolio_path.exists():
            import json
            pf = json.loads(portfolio_path.read_text())
            # expect pf to be a list of holdings with 'Instrument' or 'symbol' key
            symbols = []
            for item in pf:
                if isinstance(item, dict):
                    sym = item.get('Instrument') or item.get('instrument') or item.get('symbol') or item.get('Symbol')
                    if sym:
                        symbols.append(str(sym))
            symbols = list({s for s in symbols if s})
            if symbols:
                st.sidebar.write(f"Detected {len(symbols)} portfolio symbols from latest_portfolio.json")
                extracted = extract_symbol_files_from_zips(symbols)
                if extracted:
                    # load extracted files
                    loaded = load_price_data(extracted)
                    if loaded:
                        price_series.update(loaded)
                        st.sidebar.success(f"Extracted and loaded {len(loaded)} symbol files from zips")
    except Exception:
        pass

# Parse sector mapping if uploaded
sector_map = {}
if sector_file:
    try:
        dfsec = pd.read_csv(sector_file)
        if 'symbol' in dfsec.columns and 'sector' in dfsec.columns:
            sector_map = dict(zip(dfsec['symbol'].astype(str), dfsec['sector'].astype(str)))
            st.sidebar.write(f"Loaded sector map for {len(sector_map)} symbols")
        else:
            st.sidebar.warning("Sector CSV must contain 'symbol' and 'sector' columns")
    except Exception as e:
        st.sidebar.warning("Failed to read sector mapping CSV")

# Helper to compute PQS and IFS using existing modules
pqs_scorer = PortfolioQualityScorer()
ifs_scorer = InvestorFitScorer()


def score_from_pred(pred: Dict, investor_indices: Dict) -> Dict:
    # Build metrics dict expected by PortfolioQualityScorer and InvestorFitScorer
    try:
        metrics = {
            'annual_return': pred.get('expected_return', 0.0),
            'volatility': pred.get('expected_volatility', 0.0),
            'sharpe_ratio': pred.get('sharpe_ratio', 0.0),
            'max_drawdown': pred.get('max_drawdown', 0.0),
            'var_95': pred.get('var_95', 0.0),
            'var_99': pred.get('var_99', 0.0)
        }
        pqs = pqs_scorer.calculate_pqs(metrics)
        # pqs may be a dict with 'pqs_score' or a numeric value; handle both
        if isinstance(pqs, dict):
            pqs_val = pqs.get('pqs_score', 0.0)
        else:
            try:
                pqs_val = float(pqs)
            except Exception:
                pqs_val = 0.0

        ifs_result = ifs_scorer.calculate_ifs(investor_indices, metrics)
        if isinstance(ifs_result, dict):
            ifs_val = ifs_result.get('ifs_score', 0.0)
        else:
            try:
                ifs_val = float(ifs_result)
            except Exception:
                ifs_val = 0.0

        return {'pqs': float(pqs_val), 'ifs': float(ifs_val)}
    except Exception:
        # Fallback if scoring fails
        return {'pqs': 0.0, 'ifs': 0.0}


# Objective for optimizer

def objective_fn(weights, modeler, symbols, investor_indices, alpha=0.5, sector_map=None, constraints=None):
    # weights is a flattened array summing to 1 (we will normalize inside)
    w = np.array(weights)
    if w.sum() == 0:
        return 1e6
    w = np.maximum(0, w)
    w = w / w.sum()

    try:
        pred = modeler.predict(w)
        sharpe = pred.get('sharpe_ratio', 0.0)
    except Exception:
        # If prediction fails, return high penalty
        return 1e6

    # We want to maximize Sharpe Ratio -> minimize negative Sharpe Ratio
    val = -sharpe

    # Sector exposure penalty (unchanged)
    penalty = 0.0
    try:
        max_sector = constraints.get('max_sector_exposure', 0.30) if constraints else 0.30
        if sector_map and len(sector_map) > 0:
            # compute exposures per sector
            sector_exposures = {}
            for sym, wgt in zip(symbols, w):
                sec = sector_map.get(sym)
                if sec:
                    sector_exposures[sec] = sector_exposures.get(sec, 0.0) + float(wgt)
            # apply penalty if any sector exceeds
            for sec, exp in sector_exposures.items():
                if exp > max_sector:
                    penalty += (exp - max_sector) * 100.0  # scaled penalty
    except Exception:
        penalty = 0.0

    return float(val + penalty)


def recommend_portfolio(modeler, symbols, investor_indices, constraints: Dict, alpha=0.5, sector_map=None):
    n = len(symbols)
    bounds = [(0.0, constraints.get('max_position_size', 0.2)) for _ in range(n)]

    # Use SLSQP optimizer instead of differential_evolution to avoid parallelization issues
    # Start with equal weights
    x0 = np.ones(n) / n
    
    # Constraint: weights sum to 1
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    # Debug: show initial objective
    initial_obj = objective_fn(x0, modeler, symbols, investor_indices, alpha, sector_map, constraints)
    st.write(f"**Debug: Initial equal weights objective: {initial_obj:.4f}**")
    
    result = minimize(
        lambda x: objective_fn(x, modeler, symbols, investor_indices, alpha, sector_map=sector_map, constraints=constraints),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
    )
    
    st.write(f"**Debug: Optimization result - success: {result.success}, fun: {result.fun:.4f}, nfev: {result.nfev}**")
    st.write(f"**Debug: Optimized weights: {result.x}**")
    
    # If optimization didn't improve, try random restart
    if not result.success or abs(result.fun - initial_obj) < 1e-4:
        st.write("**Debug: Optimization didn't improve, trying random restarts...**")
        # Try a few random starts
        best_result = result
        for i in range(5):
            x_rand = np.random.random(n)
            x_rand = x_rand / x_rand.sum()
            rand_obj_initial = objective_fn(x_rand, modeler, symbols, investor_indices, alpha, sector_map, constraints)
            st.write(f"**Debug: Random start {i+1} initial objective: {rand_obj_initial:.4f}**")
            
            res_rand = minimize(
                lambda x: objective_fn(x, modeler, symbols, investor_indices, alpha, sector_map=sector_map, constraints=constraints),
                x_rand,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )
            st.write(f"**Debug: Random start {i+1} result - success: {res_rand.success}, fun: {res_rand.fun:.4f}**")
            if res_rand.fun < best_result.fun:
                best_result = res_rand
                st.write(f"**Debug: New best found!**")
        result = best_result
    
    st.write(f"**Debug: Final result - fun: {result.fun:.4f}, weights: {result.x}**")
    
    w = np.maximum(0, result.x)
    w = w / w.sum()
    # enforce min stocks by zeroing low weights if needed
    min_stocks = constraints.get('min_stocks', 5)
    if (w > 1e-3).sum() < min_stocks:
        # keep top-k weights
        idx = np.argsort(-w)
        mask = np.zeros_like(w)
        mask[idx[:min_stocks]] = 1
        w = w * mask
        w = w / w.sum()
    return dict(zip(symbols, w)), result.fun


# Default constraints from user
DEFAULT_CONSTRAINTS = {
  "min_stocks": 5,
  "max_stocks": 25,
  "max_position_size": 0.20,
  "max_sector_exposure": 0.30,
  "rebalancing_frequency": "quarterly",
  "conservative": {"max_equity": 0.40, "min_debt": 0.50, "max_volatility": 0.15},
  "moderate": {"max_equity": 0.70, "min_debt": 0.20, "max_volatility": 0.25},
  "aggressive": {"max_equity": 1.0, "min_debt": 0.0, "max_volatility": 0.40}
}

# Run optimization when requested
if run_opt:
    if st.session_state.get('ml_models') is None:
        st.error("No ML models available. Please train models first.")
    elif iid is None:
        st.error("Please upload IID (investor profile JSON) first.")
    else:
        modeler = st.session_state['ml_models']['modeler']
        symbols = st.session_state['ml_models']['symbols']
        data_source = st.session_state['ml_models'].get('data_source', 'unknown')
        st.info(f"🔄 Generating recommendation using {data_source.replace('_', ' ')} data with {len(symbols)} assets: {', '.join(symbols)}")
        
        # Prepare investor indices from IID: attempt to map to expected fields
        inv = {}
        try:
            inv['risk_capacity_index'] = float(iid.get('risk_capacity_index', 50))
            inv['risk_tolerance_index'] = float(iid.get('risk_tolerance_index', 50))
            inv['behavioral_fragility_index'] = float(iid.get('behavioral_fragility_index', 50))
            inv['time_horizon_strength'] = float(iid.get('time_horizon_strength', 50))
        except Exception:
            inv = {'risk_capacity_index':50,'risk_tolerance_index':50,'behavioral_fragility_index':50,'time_horizon_strength':50}

        # Debug: Test objective function with different weight combinations
        st.write("**Debug: Testing objective function with different weights**")
        test_weights = [
            np.ones(len(symbols)) / len(symbols),  # equal weights
            np.array([0.5] + [0.5/(len(symbols)-1)] * (len(symbols)-1)),  # concentrated in first asset
            np.random.random(len(symbols)),  # random weights
        ]
        test_weights[2] = test_weights[2] / test_weights[2].sum()  # normalize random
        
        for i, w in enumerate(test_weights):
            try:
                pred = modeler.predict(w)
                scores = score_from_pred(pred, inv)
                obj_val = objective_fn(w, modeler, symbols, inv, alpha=0.5, sector_map=sector_map, constraints=DEFAULT_CONSTRAINTS)
                st.write(f"  Test {i+1}: weights={w[:3]}..., pred={pred}, scores={scores}, objective={obj_val:.4f}")
            except Exception as e:
                st.write(f"  Test {i+1}: ERROR - {e}")
        
        st.info("Running optimizer. This may take a minute.")
        
        # Debug: show initial equal-weight prediction
        x0 = np.ones(len(symbols)) / len(symbols)
        try:
            pred_init = modeler.predict(x0)
            sc_init = score_from_pred(pred_init, inv)
            st.write(f"**Initial equal weights:** PQS={sc_init['pqs']:.1f}, IFS={sc_init['ifs']:.1f}")
        except Exception as e:
            st.warning(f"Could not compute initial prediction: {e}")
        
        alloc, score = recommend_portfolio(modeler, symbols, inv, DEFAULT_CONSTRAINTS, alpha=0.5, sector_map=sector_map)
        st.success("Recommendation completed.")
        st.subheader("Recommended Allocations")
        alloc_df = pd.DataFrame(list(alloc.items()), columns=['symbol','weight'])
        st.dataframe(alloc_df.sort_values('weight', ascending=False).reset_index(drop=True))
        
        # Show predicted metrics and scores - use symbols order from training
        w_vec = np.array([alloc[s] for s in symbols])
        pred_metrics = modeler.predict(w_vec)
        sc = score_from_pred(pred_metrics, inv)
        st.metric("Predicted PQS", f"{sc['pqs']:.1f}/100")
        st.metric("Predicted IFS", f"{sc['ifs']:.1f}/100")
        st.json({'predicted_metrics': pred_metrics})
        st.write(f"**Optimizer score (negated):** {score:.4f}")
with st.expander("Errors and Warnings", expanded=False):
    if diagnostics_msgs:
        for msg in diagnostics_msgs:
            if msg.startswith("❌"):
                st.error(msg)
            else:
                st.warning(msg)
    else:
        st.success("No errors or warnings detected.")

st.write("---")
st.write("This is an experimental ML-based recommender. Review results and validate before using.")

