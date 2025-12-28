# Portfolio Health Analysis System

## 📊 Project Overview

A comprehensive portfolio health classification system that analyzes investment portfolios based on quantitative metrics and investment objectives. The system fetches real-time data from multiple sources, calculates risk and performance metrics, and provides actionable health assessments.

**Built**: December 2024  
**Status**: ✅ Fully Operational

---

## 🎯 Business Problem Solved

**Challenge**: Classify the health of a user's investment portfolio based on their objectives and market performance.

**Requirements**:
- Calculate 1-year returns
- Compute Value at Risk (VaR)
- Assess alignment with investment objectives
- Compare performance against dynamic benchmarks (sector-specific)
- Provide actionable health classification

**Solution**: A hybrid rule-based + quantitative system that:
1. Fetches historical data for stocks and mutual funds
2. Calculates comprehensive risk and performance metrics
3. Compares against sector-appropriate benchmarks
4. Scores objective alignment
5. Generates overall health classification with recommendations

---

## ✨ Features Implemented

### 1. **Multi-Source Data Integration**
- **Indian Stocks**: yfinance (NSE/BSE data)
- **Mutual Funds**: MFAPI (free Indian MF API)
- **Benchmarks**: Sector-specific indices (Nifty 50, Metal, Gold, etc.)
- **Upstox Integration**: Framework ready (requires subscription for historical data)

### 2. **Performance Metrics**
- ✅ **1-Year Returns** (Annualized)
- ✅ **Sharpe Ratio** (Risk-adjusted returns)
- ✅ **Sortino Ratio** (Downside risk-adjusted returns)
- ✅ **Jensen's Alpha** (Benchmark-relative performance)
- ✅ **Beta** (Market correlation)

### 3. **Risk Metrics**
- ✅ **VaR (95%)** - Value at Risk at 95% confidence
- ✅ **VaR (99%)** - Value at Risk at 99% confidence
- ✅ **Maximum Drawdown** - Largest peak-to-trough decline
- ✅ **Volatility** (Annualized standard deviation)

### 4. **Benchmark Analysis**
- Dynamic benchmark selection based on portfolio composition
- Sector-specific benchmarks:
  - Equity funds → Nifty 50
  - Metal stocks → Nifty Metal
  - Gold → Gold futures
  - Technology → Nifty IT
  - Banking → Bank Nifty
- Outperformance/underperformance calculation

### 5. **Objective Alignment Scoring**
- Pre-defined objectives:
  - **Conservative Income**: Low risk, stable returns
  - **Moderate Growth**: Balanced risk-return
  - **Aggressive Growth**: High risk, high returns
- Individual component scores (0-100):
  - Return score
  - VaR score
  - Sharpe ratio score
  - Volatility score
  - Sortino ratio score
- Overall alignment score with categorization

### 6. **Portfolio Health Classification**
- **3-Tier Health Status**:
  - ✅ **Healthy** (Score: 80-100)
  - ⚠️ **Warning** (Score: 60-79)
  - ❌ **At Risk** (Score: <60)
- Component breakdown:
  - Performance score
  - Risk management score
  - Objective alignment score
- Actionable recommendations

---

## 🏗️ System Architecture

### Core Modules

#### 1. **data_fetcher.py**
Handles all data retrieval operations:
- Fetches stock data from yfinance
- Fetches mutual fund NAV from MFAPI
- Upstox API integration (ready for activation)
- Benchmark data retrieval
- Timezone normalization

#### 2. **portfolio_metrics.py**
Calculates performance and risk metrics:
- Annual returns
- Sharpe and Sortino ratios
- VaR (parametric method)
- Maximum drawdown
- Jensen's alpha and beta
- Risk-free rate handling

#### 3. **benchmark_analyzer.py**
Manages benchmark comparison:
- Sector-to-benchmark mapping
- Portfolio benchmark determination
- Performance comparison
- Outperformance calculation

#### 4. **objective_alignment.py**
Assesses portfolio-objective fit:
- Objective definitions and thresholds
- Component scoring algorithms
- Overall alignment calculation
- Recommendation generation

#### 5. **portfolio_health.py**
Generates health classification:
- Component score calculation
- Overall health score
- Status determination (Healthy/Warning/At Risk)
- Issue identification
- Recommendation synthesis

#### 6. **portfolio_analyzer.py**
Main orchestrator:
- Loads portfolio data
- Coordinates all modules
- Generates comprehensive reports
- Handles error cases

#### 7. **config.py**
Configuration management:
- Environment variable loading
- API credential validation
- Security configuration

---

## 📁 File Structure

```
F:\AI Insights Dashboard\
│
├── portfolio_analyzer.py          # Main analysis script
├── data_fetcher.py                 # Data retrieval module
├── portfolio_metrics.py            # Metrics calculation
├── benchmark_analyzer.py           # Benchmark comparison
├── objective_alignment.py          # Objective scoring
├── portfolio_health.py             # Health classification
├── config.py                       # Configuration manager
│
├── generate_upstox_token.py        # Upstox token generator
├── test_upstox.py                  # Upstox connection test
├── test_historical.py              # Historical data test
├── test_historical_simple.py       # Simplified test
│
├── .env                            # API credentials (secured)
├── .env.template                   # Credential template
├── .gitignore                      # Git ignore rules
│
├── UPSTOX_SETUP.md                # Upstox setup guide
├── UPSTOX_STATUS.md               # Integration status
└── README.md                       # This file
```

---

## 📊 Sample Analysis Output

### Portfolio Analyzed
- **Holdings**: 3 instruments
- **Total Value**: ₹213,369.43
- **Composition**:
  - GOLD1: ₹60,842.70 (28.5%) - Precious Metals
  - NATIONALUM: ₹24,116.42 (11.3%) - Mining
  - Motilal Oswal Large & Midcap Fund: ₹128,410.31 (60.2%) - Equity Fund

### Performance Metrics
- **Annual Return**: 27.53%
- **Sharpe Ratio**: 1.571
- **Sortino Ratio**: 2.104
- **Jensen's Alpha**: 18.19%
- **Beta**: 0.811

### Risk Metrics
- **VaR (95%)**: 18.35%
- **VaR (99%)**: 28.61%
- **Max Drawdown**: 8.87%
- **Volatility**: 11.93%

### Benchmark Performance
- **Primary Benchmark**: Nifty 50 (^NSEI)
- **Portfolio Return**: 27.53%
- **Benchmark Return**: 7.38%
- **Outperformance**: +20.15% 📈

### Objective Alignment
- **Objective**: Moderate Growth
- **Alignment Score**: 95.6/100 (Excellent)
- **Component Scores**:
  - Return: 100/100
  - VaR: 78/100
  - Sharpe: 100/100
  - Volatility: 100/100
  - Sortino: 100/100

### Portfolio Health
- **Status**: ✅ Healthy
- **Overall Score**: 95.8/100
- **Component Breakdown**:
  - Performance: 96.7/100
  - Risk Management: 95.0/100
  - Objective Alignment: 95.6/100

---

## 🚀 How to Use

### 1. Setup Environment

```bash
# Install dependencies
pip install pandas numpy yfinance scipy requests python-dotenv upstox-python-sdk openpyxl

# Or use the virtual environment
.venv\Scripts\activate
```

### 2. Configure API Credentials (Optional)

For Upstox integration:
```bash
copy .env.template .env
# Edit .env with your credentials
```

### 3. Prepare Portfolio Data

Create an Excel file with columns:
- `Instrument` - Stock/fund name
- `Qty.` - Quantity held
- `Avg. cost` - Average purchase price
- `LTP` - Last traded price
- `Invested` - Total investment
- `Cur. val` - Current value
- `P&L` - Profit/Loss
- `Net chg.` - Net change %
- `Day chg.` - Day change %
- `Asset Class` - Equity/Commodity/etc.
- `Sector` - Sector classification

### 4. Run Analysis

```bash
python portfolio_analyzer.py
```

### 5. Review Output

The system generates a comprehensive console report with:
- Portfolio summary
- Historical data status
- All calculated metrics
- Benchmark comparison
- Objective alignment
- Health classification
- Actionable recommendations

---

## 🔧 Configuration Options

### Investment Objectives

Modify in `objective_alignment.py`:

```python
OBJECTIVES = {
    'Conservative Income': {...},
    'Moderate Growth': {...},
    'Aggressive Growth': {...}
}
```

### Benchmark Mappings

Update in `data_fetcher.py`:

```python
sector_benchmarks = {
    'Precious Metals': 'GC=F',
    'Mining': '^CNXMETAL',
    'Technology': '^CNXIT',
    # Add custom sectors
}
```

### Risk-Free Rate

Modify in `data_fetcher.py`:

```python
def get_risk_free_rate(self) -> float:
    return 0.065  # 6.5% annual (Indian govt bonds)
```

---

## 🔐 Security Features

- ✅ API credentials stored in `.env` (not committed to git)
- ✅ `.gitignore` configured to exclude sensitive files
- ✅ Template file (`.env.template`) for easy setup
- ✅ Credential validation before API calls
- ✅ Secure token generation workflow

---

## 🌐 Data Sources

### 1. **yfinance**
- Free, no API key required
- NSE/BSE stock data
- Index data
- Commodity futures
- Limitations: 15-minute delay on some data

### 2. **MFAPI (api.mfapi.in)**
- Free Indian mutual fund API
- Daily NAV updates (3x daily)
- Historical data (5+ years)
- 10,000+ schemes
- No authentication required

### 3. **Upstox API (Optional)**
- Real-time Indian market data
- Requires API credentials
- Historical data needs subscription
- Framework ready for activation

---

## 📈 Metrics Explained

### **Sharpe Ratio**
- Measures risk-adjusted returns
- Higher is better (>1 is good, >2 is excellent)
- Formula: `(Return - Risk_Free_Rate) / Volatility`

### **Sortino Ratio**
- Similar to Sharpe but penalizes only downside volatility
- Better for asymmetric return distributions
- Higher values indicate better downside risk management

### **Jensen's Alpha**
- Excess return over expected return (CAPM)
- Positive alpha = outperformance
- Measures manager skill vs. market exposure

### **Value at Risk (VaR)**
- Maximum expected loss at given confidence level
- VaR(95%) = loss not exceeded 95% of the time
- Used for risk budgeting and limits

### **Beta**
- Portfolio sensitivity to market movements
- β < 1: Less volatile than market
- β = 1: Moves with market
- β > 1: More volatile than market

---

## 🎓 Technical Approach

### Why Hybrid System (Not Pure ML)?

**Decision Rationale**:
1. **Domain Logic**: Portfolio health involves business rules (VaR thresholds, objective definitions) that are better expressed as rules than learned
2. **Data Requirements**: ML models need large training datasets; most users have limited portfolio history
3. **Interpretability**: Stakeholders need to understand why a portfolio is classified as "At Risk"
4. **Regulatory**: Financial decisions often require auditable, explainable logic
5. **Accuracy**: For well-defined metrics (VaR, Sharpe), mathematical formulas are more accurate than approximations

**ML Enhancement Opportunities** (Future):
- Anomaly detection in portfolio behavior
- Predictive models for future health trends
- Pattern recognition across similar portfolios
- Optimization recommendations

---

## 🐛 Troubleshooting

### Issue: No data fetched for stocks
**Solution**: Check ticker symbols are correct for NSE (.NS suffix)

### Issue: Mutual fund not found
**Solution**: Use full, exact fund name or check MFAPI search results

### Issue: Upstox historical data fails
**Solution**: Upstox requires paid subscription - system falls back to yfinance automatically

### Issue: Timezone errors
**Solution**: Already handled - all data normalized to timezone-naive

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Web dashboard (Streamlit/Dash)
- [ ] PDF report generation
- [ ] Email alerts for health changes
- [ ] Multi-portfolio comparison
- [ ] Rebalancing recommendations
- [ ] Tax optimization insights
- [ ] Monte Carlo simulations
- [ ] Stress testing scenarios

### ML Integration Opportunities
- [ ] Portfolio clustering
- [ ] Anomaly detection
- [ ] Predictive health forecasting
- [ ] Automated objective detection
- [ ] Sentiment analysis integration

---

## 📝 Notes

### Design Decisions

1. **No Database**: Uses Excel input for simplicity; can be extended to database storage
2. **Console Output**: Easy to debug; can be wrapped in API/web interface
3. **Modular Design**: Each component is independent and testable
4. **Fallback Strategy**: yfinance as backup ensures system always works
5. **Parametric VaR**: Faster than historical/Monte Carlo; sufficient for daily risk monitoring

### Known Limitations

1. **Mutual Fund Data**: MFAPI provides NAV only (no intraday)
2. **Upstox Historical**: Requires subscription (not included in free tier)
3. **Benchmark Mapping**: Manual sector mapping (could be automated)
4. **Single Currency**: Currently supports INR only

---

## 👨‍💻 Technical Stack

- **Language**: Python 3.12
- **Data Processing**: pandas, numpy
- **Financial Data**: yfinance, requests (MFAPI)
- **Statistics**: scipy
- **API Integration**: upstox-python-sdk
- **Configuration**: python-dotenv
- **Excel Handling**: openpyxl

---

## 📄 License

This is a proprietary portfolio analysis system built for specific use cases.

---

## 🙏 Acknowledgments

- **yfinance** - Yahoo Finance data wrapper
- **MFAPI** - Free mutual fund API by @SkyRocknRoll
- **Upstox** - Indian broker API platform

---

## 📧 Support

For questions or issues:
1. Check `UPSTOX_SETUP.md` for Upstox configuration
2. Check `UPSTOX_STATUS.md` for integration status
3. Review error messages in console output
4. Verify Excel data format matches requirements

---

**Last Updated**: December 24, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
