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
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict

from portfolio_quality import PortfolioQualityScorer
from investor_fit import InvestorFitScorer

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
    prices = pd.concat([price_series[symbol].rename(symbol) for symbol in symbols], axis=1).dropna()
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
        for col in y.columns:
            y_col = y[col].values
            if LGB_AVAILABLE:
                dtrain = lgb.Dataset(X, label=y_col)
                params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
                model = lgb.train(params, dtrain, num_boost_round=100)
            elif SKLEARN_AVAILABLE:
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X, y_col)
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
    blend = st.slider("PQS vs IFS blend (alpha for PQS)", 0.0, 1.0, 0.5)
    objective_choice = st.selectbox("Investment Objective", ['Moderate Growth', 'Conservative Income', 'Aggressive Growth', 'Balanced'])
    run_opt = st.button("Recommend Portfolio")

# Load IID
if iid_file:
    iid = pd.read_json(iid_file)
    st.write("Loaded IID")
else:
    # Try to load IID saved by `dashboard.py`
    iid = None
    try:
        iid_path = Path('data') / 'latest_iid.json'
        if iid_path.exists():
            iid = pd.read_json(iid_path)
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
    except Exception:
        pass

# Load price data
price_series = {}
if price_files:
    paths = []
    for f in price_files:
        # save to temp file
        tmp = os.path.join(".", f.name)
        with open(tmp, 'wb') as fh:
            fh.write(f.getbuffer())
        paths.append(tmp)
    price_series = load_price_data(paths)
    st.write(f"Loaded {len(price_series)} assets from uploaded CSVs")
else:
    st.info("Please upload price CSVs for candidate assets (Nifty50 + Next50).")
    # Also try to load ZNifty zip files present in workspace root
    try:
        base = Path('data')
        # extraction handled in load_prebuilt_universe; attempt to call it directly
        loaded = load_prebuilt_universe()
        if loaded:
            price_series.update(loaded)
            st.success(f"Loaded {len(loaded)} assets from prebuilt zip/universe")
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
        st.info("Simulating portfolios and training models. This may take a few minutes.")
        sim = simulate_random_portfolios(price_series, n=int(n_sim))
        modeler = SimpleRegressors()
        modeler.fit(sim['X'], sim['y'])
        st.session_state['ml_models'] = {'modeler': modeler, 'symbols': sim['symbols']}
        st.success("Training completed and models cached.")

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
    metrics = {
        'annual_return': pred.get('expected_return', 0.0),
        'volatility': pred.get('expected_volatility', 0.0),
        'sharpe_ratio': pred.get('sharpe_ratio', 0.0),
        'max_drawdown': pred.get('max_drawdown', 0.0),
        'var_95': pred.get('var_95', 0.0),
        'var_99': pred.get('var_99', 0.0)
    }
    pqs = pqs_scorer.calculate_pqs(metrics)
    ifs = ifs_scorer.calculate_ifs(investor_indices, metrics)['ifs_score']
    return {'pqs': pqs, 'ifs': ifs}


# Objective for optimizer

def objective_fn(weights, modeler, symbols, investor_indices, alpha=0.5, sector_map=None, constraints=None):
    # weights is a flattened array summing to 1 (we will normalize inside)
    w = np.array(weights)
    if w.sum() == 0:
        return 1e6
    w = np.maximum(0, w)
    w = w / w.sum()
    pred = modeler.predict(w)
    scores = score_from_pred(pred, investor_indices)
    # we want to maximize alpha*pqs + (1-alpha)*ifs -> minimize negative
    val = -(alpha * scores['pqs'] + (1 - alpha) * scores['ifs'])
    # Sector exposure penalty
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
    
    result = minimize(
        lambda x: objective_fn(x, modeler, symbols, investor_indices, alpha, sector_map=sector_map, constraints=constraints),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-6}
    )
    
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
        # Prepare investor indices from IID: attempt to map to expected fields
        inv = {}
        try:
            inv['risk_capacity_index'] = float(iid.get('risk_capacity_index', 50))
            inv['risk_tolerance_index'] = float(iid.get('risk_tolerance_index', 50))
            inv['behavioral_fragility_index'] = float(iid.get('behavioral_fragility_index', 50))
            inv['time_horizon_strength'] = float(iid.get('time_horizon_strength', 50))
        except Exception:
            inv = {'risk_capacity_index':50,'risk_tolerance_index':50,'behavioral_fragility_index':50,'time_horizon_strength':50}

        st.info("Running optimizer. This may take a minute.")
        alloc, score = recommend_portfolio(modeler, symbols, inv, DEFAULT_CONSTRAINTS, alpha=blend, sector_map=sector_map)
        st.success("Recommendation completed.")
        st.subheader("Recommended Allocations")
        alloc_df = pd.DataFrame(list(alloc.items()), columns=['symbol','weight'])
        st.dataframe(alloc_df.sort_values('weight', ascending=False).reset_index(drop=True))
        # Show predicted metrics and scores
        w_vec = alloc_df['weight'].values
        # align weights to symbols order
        pred_metrics = modeler.predict(w_vec)
        sc = score_from_pred(pred_metrics, inv)
        st.metric("Predicted PQS", f"{sc['pqs']:.1f}/100")
        st.metric("Predicted IFS", f"{sc['ifs']:.1f}/100")
        st.json({'predicted_metrics': pred_metrics})


st.write("---")
st.write("This is an experimental ML-based recommender. Review results and validate before using.")

