import runpy
import sys
import types
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create a dummy streamlit module to allow importing dashboard_ml without an interactive UI
class DummySidebar:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def header(self, *a, **k):
        pass
    def write(self, *a, **k):
        pass
    def success(self, *a, **k):
        pass
    def warning(self, *a, **k):
        pass

class DummySt:
    def __init__(self):
        self.sidebar = DummySidebar()
    def set_page_config(self, **k):
        pass
    def cache_data(self, show_spinner=False):
        def deco(f):
            return f
        return deco
    def title(self, *a, **k):
        pass
    def write(self, *a, **k):
        # print some debug lines
        print(*a)
    def header(self, *a, **k):
        pass
    def file_uploader(self, *a, **k):
        return None
    def checkbox(self, *a, **k):
        return False
    def number_input(self, *a, **k):
        return k.get('value', 2000)
    def button(self, *a, **k):
        return False
    def slider(self, *a, **k):
        # return provided default if present
        if len(a) >= 4:
            return a[3]
        return k.get('value', 0)
    def selectbox(self, *a, **k):
        # return first option if provided
        if len(a) >= 2 and isinstance(a[1], (list, tuple)) and len(a[1]) > 0:
            return a[1][0]
        return None
    def markdown(self, *a, **k):
        pass
    def info(self, *a, **k):
        print('INFO:', *a)
    def success(self, *a, **k):
        print('SUCCESS:', *a)
    def warning(self, *a, **k):
        print('WARNING:', *a)
    def error(self, *a, **k):
        print('ERROR:', *a)
    def sidebar(self):
        return self
    def metric(self, *a, **k):
        print('METRIC', a)
    def dataframe(self, df):
        print(df.head())
    def json(self, *a, **k):
        print('JSON', a)

# Insert dummy streamlit into sys.modules
sys.modules['streamlit'] = types.ModuleType('streamlit')
_dummy = DummySt()
# copy methods to module
for name in dir(_dummy):
    if name.startswith('_'):
        continue
    attr = getattr(_dummy, name)
    try:
        setattr(sys.modules['streamlit'], name, attr)
    except Exception:
        pass
# ensure attributes used as properties exist
sys.modules['streamlit'].sidebar = DummySidebar()
sys.modules['streamlit'].cache_data = _dummy.cache_data
sys.modules['streamlit'].set_page_config = _dummy.set_page_config
sys.modules['streamlit'].write = _dummy.write
sys.modules['streamlit'].title = _dummy.title
sys.modules['streamlit'].file_uploader = _dummy.file_uploader
sys.modules['streamlit'].checkbox = _dummy.checkbox
sys.modules['streamlit'].number_input = _dummy.number_input
sys.modules['streamlit'].button = _dummy.button
sys.modules['streamlit'].info = _dummy.info
sys.modules['streamlit'].success = _dummy.success
sys.modules['streamlit'].warning = _dummy.warning
sys.modules['streamlit'].error = _dummy.error
sys.modules['streamlit'].metric = _dummy.metric
sys.modules['streamlit'].dataframe = _dummy.dataframe
sys.modules['streamlit'].json = _dummy.json
sys.modules['streamlit'].session_state = {}

# Now run the dashboard module in a controlled namespace
globs = runpy.run_path('dashboard_ml.py')

# Build synthetic price_series for 6 assets with 300 days of prices
n_assets = 6
days = 300
dates = pd.date_range(end=datetime.today(), periods=days, freq='B')
price_series = {}
np.random.seed(42)
for i in range(n_assets):
    # simulate geometric Brownian motion
    returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
    price = 100 * np.exp(np.cumsum(returns))
    s = pd.Series(price, index=dates)
    price_series[f"ASSET{i+1}"] = s

# Call simulate_random_portfolios and train SimpleRegressors
sim = globs['simulate_random_portfolios'](price_series, n=500, min_stocks=2, max_stocks=5)
print('\nSim X shape:', sim['X'].shape, 'y shape:', sim['y'].shape)

modeler_cls = globs['SimpleRegressors']
modeler = modeler_cls()
modeler.fit(sim['X'], sim['y'])

# Predict on equal weights
symbols = sim['symbols']
w = np.ones(len(symbols)) / len(symbols)
pred = modeler.predict(w)
print('\nPrediction for equal weights:', pred)

# Compute objective for equal and a concentrated weight
inv = {'risk_capacity_index':50,'risk_tolerance_index':50,'behavioral_fragility_index':50,'time_horizon_strength':50}
from functools import partial
obj_fn = globs['objective_fn']
obj_equal = obj_fn(w, modeler, symbols, inv, alpha=0.5, sector_map={}, constraints=globs['DEFAULT_CONSTRAINTS'])
print('Objective (equal):', obj_equal)

w_conc = np.zeros(len(symbols))
w_conc[0] = 1.0
obj_conc = obj_fn(w_conc, modeler, symbols, inv, alpha=0.5, sector_map={}, constraints=globs['DEFAULT_CONSTRAINTS'])
print('Objective (concentrated):', obj_conc)

print('\nSmoke test completed.')
