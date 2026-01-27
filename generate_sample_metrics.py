"""
Asset Metrics Generator - Sample/Demo Version
Creates sample CSV files showing the structure for asset-level inputs
Shows format and structure that will be filled with real data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Stock lists
NIFTY50 = [
    'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV',
    'BPCL', 'BHARTIARTL', 'BOSCHIND', 'BRITANNIA', 'CIPLA',
    'COALINDIA', 'COLPAL', 'DIVISLAB', 'DRREDDY', 'EICHERMOT',
    'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK',
    'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
    'IPCALAB', 'INDIGO', 'INFY', 'ITC', 'JSWSTEEL',
    'KOTAKBANK', 'LT', 'LTTS', 'M&M', 'MARUTI',
    'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE',
    'SBICARD', 'SBILIFE', 'SBIN', 'SUNPHARMA', 'TATAMOTORS',
    'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
    'UPL', 'VBL', 'WIPRO'
]

NIFTY_NEXT50 = [
    'ABB', 'AARTIIND', 'ABCAPITAL', 'ABSL', 'ADANIENT',
    'ADANIGREEN', 'ADANIPOWER', 'AMARAJABAT', 'AMBUJACEM', 'APOLLOHOSP',
    'APOLLOTYRE', 'ASHOKLEY', 'AUROPHARMA', 'AUBANK', 'AUTOCLAD',
    'AUTOIND', 'AXISMF', 'BAJAJTRIPUR', 'BASF', 'BATAINDIA',
    'BEL', 'BERGEPAINT', 'BHARATRAS', 'BHEL', 'BLS',
    'BRIGADE', 'CADILAHC', 'CAMS', 'CARBORUNM', 'CASTROL',
    'CGCONSTRUCT', 'CGPOWER', 'CHAMBLFERT', 'CHEMPLAST', 'CHOLAFIN',
    'CHROMIND', 'COLDRAKE', 'COALGO', 'CUMMINSIND', 'DCBBANK',
    'DEEPAKNTR', 'DIALBRANDS', 'ELGIEQUIP', 'EMAMILTD', 'EXIDEIND'
]


def generate_sample_metrics():
    """Generate sample metrics CSV files"""
    
    print("=" * 80)
    print("📊 ASSET METRICS SAMPLE DATA GENERATOR")
    print("=" * 80)
    print("\nGenerating sample CSV files with structure for asset-level inputs...")
    print("Note: These are SAMPLE values. Real data will be calculated from price feeds.\n")
    
    # Generate sample data for Nifty50
    print("🔄 Generating Nifty50 sample metrics...")
    nifty50_data = []
    
    np.random.seed(42)
    for symbol in NIFTY50:
        metrics = {
            'symbol': symbol,
            'data_points': 252,  # 1 year of trading days
            'last_price': round(np.random.uniform(100, 5000), 2),
            'last_date': '2026-01-22',
            'returns_5d_ma': round(np.random.uniform(-2, 2), 3),
            'returns_20d_ma': round(np.random.uniform(-1.5, 1.5), 3),
            'returns_60d_ma': round(np.random.uniform(-1, 1), 3),
            'volatility_30d': round(np.random.uniform(10, 35), 2),
            'volatility_90d': round(np.random.uniform(12, 30), 2),
            'sharpe_ratio': round(np.random.uniform(0, 2), 3),
            'sortino_ratio': round(np.random.uniform(0, 2.5), 3),
            'calmar_ratio': round(np.random.uniform(0, 3), 3),
            'max_drawdown_90d': round(np.random.uniform(5, 25), 2),
            'skewness': round(np.random.uniform(-1, 1), 3),
            'kurtosis': round(np.random.uniform(-1, 3), 3),
            'beta': round(np.random.uniform(0.5, 1.5), 3),
        }
        nifty50_data.append(metrics)
    
    df_nifty50 = pd.DataFrame(nifty50_data).sort_values('sharpe_ratio', ascending=False)
    
    # Generate sample data for Nifty Next50
    print("🔄 Generating Nifty Next50 sample metrics...")
    nifty_next50_data = []
    
    for symbol in NIFTY_NEXT50:
        metrics = {
            'symbol': symbol,
            'data_points': 252,
            'last_price': round(np.random.uniform(50, 3000), 2),
            'last_date': '2026-01-22',
            'returns_5d_ma': round(np.random.uniform(-2.5, 2.5), 3),
            'returns_20d_ma': round(np.random.uniform(-2, 2), 3),
            'returns_60d_ma': round(np.random.uniform(-1.5, 1.5), 3),
            'volatility_30d': round(np.random.uniform(12, 40), 2),
            'volatility_90d': round(np.random.uniform(14, 35), 2),
            'sharpe_ratio': round(np.random.uniform(-0.5, 1.8), 3),
            'sortino_ratio': round(np.random.uniform(-0.5, 2.2), 3),
            'calmar_ratio': round(np.random.uniform(-1, 2.5), 3),
            'max_drawdown_90d': round(np.random.uniform(8, 30), 2),
            'skewness': round(np.random.uniform(-1.5, 1.5), 3),
            'kurtosis': round(np.random.uniform(-1, 4), 3),
            'beta': round(np.random.uniform(0.3, 1.8), 3),
        }
        nifty_next50_data.append(metrics)
    
    df_nifty_next50 = pd.DataFrame(nifty_next50_data).sort_values('sharpe_ratio', ascending=False)
    
    # Save to CSV files
    base_dir = r'f:\AI Insights Dashboard\Asset returns'
    
    nifty50_path = os.path.join(base_dir, 'nifty50', 'Nifty50_metrics.csv')
    nifty_next50_path = os.path.join(base_dir, 'nifty_next_50', 'Nifty_Next50_metrics.csv')
    
    os.makedirs(os.path.dirname(nifty50_path), exist_ok=True)
    os.makedirs(os.path.dirname(nifty_next50_path), exist_ok=True)
    
    df_nifty50.to_csv(nifty50_path, index=False)
    df_nifty_next50.to_csv(nifty_next50_path, index=False)
    
    print(f"\n✅ Sample metrics files created!")
    print(f"   📁 {nifty50_path}")
    print(f"   📁 {nifty_next50_path}")
    
    # Display sample data
    print("\n" + "=" * 80)
    print("📊 NIFTY50 - TOP 10 BY SHARPE RATIO")
    print("=" * 80)
    print(df_nifty50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("📊 NIFTY NEXT50 - TOP 10 BY SHARPE RATIO")
    print("=" * 80)
    print(df_nifty_next50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("📋 COLUMN DESCRIPTIONS (9 Asset-Level Metrics)")
    print("=" * 80)
    descriptions = {
        'symbol': 'Stock symbol (without .NS)',
        'data_points': 'Number of trading days in dataset',
        'last_price': 'Latest stock price',
        'last_date': 'Date of last data point',
        'returns_5d_ma': 'Average daily return (5-day moving average) %',
        'returns_20d_ma': 'Average daily return (20-day moving average) %',
        'returns_60d_ma': 'Average daily return (60-day moving average) %',
        'volatility_30d': 'Annualized volatility (30-day rolling) %',
        'volatility_90d': 'Annualized volatility (90-day rolling) %',
        'sharpe_ratio': 'Risk-adjusted return metric (higher = better)',
        'sortino_ratio': 'Downside risk-adjusted return (higher = better)',
        'calmar_ratio': 'Return/Max Drawdown ratio (higher = better)',
        'max_drawdown_90d': 'Maximum drawdown in 90-day period %',
        'skewness': 'Return distribution skewness (negative = left tail)',
        'kurtosis': 'Return distribution tail risk (higher = more extremes)',
        'beta': 'Systematic risk vs market (1.0 = market, >1 = more volatile)',
    }
    
    for col, desc in descriptions.items():
        print(f"  • {col:20s} : {desc}")
    
    print("\n" + "=" * 80)
    print("✅ NEXT STEP: Production Data Integration")
    print("=" * 80)
    print("""
To fill these CSV files with REAL data:

Option 1 (Recommended - Using Upstox API):
  - Current system already has Upstox integration
  - Use data_fetcher.py to fetch historical prices
  - Calculate metrics from actual OHLCV data

Option 2 (Alternative - Using Yahoo Finance):
  - Use yfinance library to download 1-year data
  - May have rate limiting issues for 100+ stocks
  - Good for testing/prototyping

Option 3 (Manual - Using CSV imports):
  - Download data from NSE website or Screeners
  - Upload to Asset returns/nifty50/ and nifty_next_50/ folders
  - Run calculation script to compute metrics

Current sample files can be:
  ✓ Used to understand the expected format
  ✓ Tested in the ML pipeline for structure validation
  ✓ Replaced with real data once collection setup is complete
""")
    
    print("=" * 80)


if __name__ == "__main__":
    generate_sample_metrics()
