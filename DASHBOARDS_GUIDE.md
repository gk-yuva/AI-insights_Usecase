# Dashboard Suite Guide

## 🎯 Two Dashboard System

This project now includes **two separate Streamlit dashboards** for different purposes:

---

## 📊 Dashboard 1: Portfolio Health Dashboard (Main)
**Port**: `8501`  
**File**: `dashboard.py`

### Purpose
Comprehensive analysis of your existing portfolio with 6 tabs:
- **📊 Overview**: Portfolio metrics, holdings, health score
- **🎯 Two-Dimensional Analysis**: Portfolio Quality Score (PQS) vs Investor Fit Score (IFS)
- **📈 Portfolio Quality**: Detailed quality metrics and risk analysis
- **👤 Investor Fit**: Investor profile analysis and alignment
- **💼 Portfolio Details**: Complete portfolio data download
- **🧮 Optimized Allocation**: Weight optimization using Sharpe ratio maximization

### How to Use
```bash
# In terminal, run:
streamlit run dashboard.py --server.port 8501

# Then open browser to:
http://localhost:8501
```

### Required Inputs
1. Portfolio Excel file (with columns: Instrument, Qty, Avg cost, LTP, Cur val, P&L, etc.)
2. Investor Information Document (IID JSON) - your risk profile and preferences

---

## 🎯 Dashboard 2: Asset Recommendations (New)
**Port**: `8502`  
**File**: `asset_recommendations_dashboard.py`

### Purpose
Lightweight dashboard focused solely on asset recommendations:
- ✅ Recommends which **Nifty50 assets to ADD** to your portfolio
- ❌ Identifies which assets to **DROP** (underperformers)
- 📈 Shows expected impact on portfolio
- 📋 Provides implementation roadmap in 3 phases

**Scope**: Considers only **Nifty50 stocks** (50 major Indian companies)

### How to Use
```bash
# In terminal, run:
streamlit run asset_recommendations_dashboard.py --server.port 8502

# Then open browser to:
http://localhost:8502
```

### Required Inputs
1. Portfolio Excel file (same format as Dashboard 1)
2. Select investment objective (Conservative/Moderate/Aggressive)
3. Click "Analyze & Recommend"

### Faster than Main Dashboard
- ✅ Focuses only on Nifty50 (50 stocks instead of 100+)
- ✅ Loads in seconds
- ✅ No complex optimization required
- ✅ Standalone - doesn't depend on other modules

---

## 🚀 Quick Start

### Run Both Dashboards
Open two terminals and run simultaneously:

**Terminal 1** - Main Dashboard:
```bash
cd f:\AI Insights Dashboard
python -m streamlit run dashboard.py --server.port 8501
```

**Terminal 2** - Asset Recommendations:
```bash
cd f:\AI Insights Dashboard
python -m streamlit run asset_recommendations_dashboard.py --server.port 8502
```

Then access:
- Main Dashboard: http://localhost:8501
- Recommendations: http://localhost:8502

---

## 📁 File Structure

```
F:\AI Insights Dashboard\
├── dashboard.py                           # Main portfolio health dashboard
├── asset_recommendations_dashboard.py     # Asset recommendations dashboard ⭐ NEW
├── portfolio_optimizer.py                 # Optimization engine
├── portfolio_analyzer.py                  # Analysis orchestrator
├── data_fetcher.py                        # Data retrieval
├── portfolio_metrics.py                   # Metrics calculation
│
└── [Supporting files...]
```

---

## 🎯 Which Dashboard Should I Use?

### Use Dashboard 1 (Main) if you want to:
- Analyze your complete portfolio health
- See detailed risk metrics (VaR, Sharpe, etc.)
- Compare against benchmarks
- Understand investor profile fit
- See weight optimization recommendations

### Use Dashboard 2 (Recommendations) if you want to:
- Get quick asset recommendations
- Find which Nifty50 stocks to add/drop
- See expected impact on portfolio
- Get a 3-phase implementation plan
- Faster analysis (seconds instead of minutes)

---

## 💡 Typical Workflow

1. **Start with Dashboard 1** (portfolio.py)
   - Upload your portfolio
   - Complete investor profile
   - See comprehensive analysis

2. **Then use Dashboard 2** (asset_recommendations.py)
   - Upload same portfolio
   - Get Nifty50 recommendations
   - Review suggested additions/removals

3. **Make decisions**
   - Compare recommendations from both
   - Validate with financial advisor
   - Execute changes gradually

---

## 🔧 Requirements

- Python 3.10+
- Streamlit 1.25+
- pandas, numpy, yfinance
- scipy

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Sample Portfolio Format

Your Excel file should have columns:
| Instrument | Qty | Avg cost | LTP | Cur val | P&L | Net chg |
|-----------|-----|----------|-----|---------|-----|---------|
| GOLD1 | 100 | 5000 | 6300 | 630000 | 130000 | 26% |
| INFY | 50 | 1500 | 1750 | 87500 | 12500 | 16.7% |

---

## ❓ Troubleshooting

### White screen on localhost:8502
- Check that port 8502 is not in use
- Restart the Streamlit process
- Clear browser cache

### "Error loading portfolio"
- Ensure Excel file has required columns
- Check file format (.xlsx or .xls)
- Verify data types (quantities should be numbers)

### Slow performance on Dashboard 1
- Use Dashboard 2 instead (faster)
- Reduce number of holdings
- Check internet connection (for data fetch)

---

## 📝 Version History

- **v1.1.0** (Jan 18, 2026): Added separate Asset Recommendations Dashboard
- **v1.0.0** (Dec 24, 2025): Initial release with main portfolio health dashboard

---

**Last Updated**: January 18, 2026  
**Status**: ✅ Both dashboards operational
