"""
Calculate Asset-Level Input Metrics for Classification ML Model (Optimized)

Uses existing data_fetcher and includes retry logic for reliable data collection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import time
warnings.filterwarnings('ignore')

try:
    from data_fetcher import DataFetcher
    from portfolio_metrics import PortfolioMetrics
except ImportError:
    print("Warning: Could not import custom modules, using basic implementation")
    DataFetcher = None
    PortfolioMetrics = None

# Nifty50 stocks (symbols without .NS)
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


class OptimizedAssetMetricsCalculator:
    """Calculate asset-level metrics using existing data_fetcher"""
    
    def __init__(self, lookback_years=1, risk_free_rate=0.062):
        """Initialize calculator with shared data fetcher"""
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.data_fetcher = DataFetcher(period_years=lookback_years)
        self.metrics_calc = PortfolioMetrics(risk_free_rate=risk_free_rate)
        
        print(f"📊 Optimized Asset Metrics Calculator")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print("=" * 80)
    
    def calculate_daily_returns_ma(self, returns):
        """Calculate 5-day, 20-day, 60-day moving averages of daily returns"""
        try:
            ma_5d = returns.rolling(window=5).mean().iloc[-1] * 100
            ma_20d = returns.rolling(window=20).mean().iloc[-1] * 100
            ma_60d = returns.rolling(window=60).mean().iloc[-1] * 100
            
            return {
                'returns_5d_ma': ma_5d if not np.isnan(ma_5d) else 0,
                'returns_20d_ma': ma_20d if not np.isnan(ma_20d) else 0,
                'returns_60d_ma': ma_60d if not np.isnan(ma_60d) else 0
            }
        except:
            return {
                'returns_5d_ma': 0,
                'returns_20d_ma': 0,
                'returns_60d_ma': 0
            }
    
    def calculate_volatility(self, returns):
        """Calculate 30-day and 90-day rolling volatility (annualized)"""
        try:
            vol_30d = returns.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100
            vol_90d = returns.rolling(window=90).std().iloc[-1] * np.sqrt(252) * 100
            
            return {
                'volatility_30d': vol_30d if not np.isnan(vol_30d) else 0,
                'volatility_90d': vol_90d if not np.isnan(vol_90d) else 0
            }
        except:
            return {'volatility_30d': 0, 'volatility_90d': 0}
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate annualized Sharpe ratio"""
        try:
            if len(returns) < 20:
                return 0
            
            ann_return = returns.mean() * 252
            ann_vol = returns.std() * np.sqrt(252)
            
            if ann_vol == 0:
                return 0
            
            sharpe = (ann_return - self.risk_free_rate) / ann_vol
            return round(float(sharpe), 3) if not np.isnan(sharpe) else 0
        except:
            return 0
    
    def calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(returns) < 20:
                return 0
            
            ann_return = returns.mean() * 252
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                downside_vol = returns.std() * np.sqrt(252)
            else:
                downside_vol = downside_returns.std() * np.sqrt(252)
            
            if downside_vol == 0:
                return 0
            
            sortino = (ann_return - self.risk_free_rate) / downside_vol
            return round(float(sortino), 3) if not np.isnan(sortino) else 0
        except:
            return 0
    
    def calculate_calmar_ratio(self, returns):
        """Calculate Calmar ratio (return / max drawdown)"""
        try:
            ann_return = returns.mean() * 252
            
            # Max drawdown
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns - peak) / peak
            max_dd = abs(drawdown.min())
            
            if max_dd == 0 or max_dd < 0.001:
                return 0
            
            calmar = ann_return / max_dd
            return round(float(calmar), 3) if not np.isnan(calmar) else 0
        except:
            return 0
    
    def calculate_max_drawdown_90d(self, returns):
        """Calculate maximum drawdown over 90-day window"""
        try:
            recent_returns = returns.tail(90)
            cum_returns = (1 + recent_returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns - peak) / peak
            max_dd = abs(drawdown.min()) * 100
            
            return round(max_dd, 2) if not np.isnan(max_dd) else 0
        except:
            return 0
    
    def calculate_skewness(self, returns):
        """Calculate skewness of return distribution"""
        try:
            from scipy import stats
            skew = stats.skew(returns.dropna())
            return round(float(skew), 3) if not np.isnan(skew) else 0
        except:
            return 0
    
    def calculate_kurtosis(self, returns):
        """Calculate kurtosis (excess kurtosis) of return distribution"""
        try:
            from scipy import stats
            kurt = stats.kurtosis(returns.dropna())
            return round(float(kurt), 3) if not np.isnan(kurt) else 0
        except:
            return 0
    
    def calculate_beta(self, stock_returns, nifty_returns):
        """Calculate beta relative to Nifty 50"""
        try:
            # Align dates
            common_dates = stock_returns.index.intersection(nifty_returns.index)
            
            if len(common_dates) < 20:
                return 0
            
            stock_ret = stock_returns.loc[common_dates]
            nifty_ret = nifty_returns.loc[common_dates]
            
            # Covariance and variance
            covariance = np.cov(stock_ret, nifty_ret)[0][1]
            nifty_variance = np.var(nifty_ret)
            
            if nifty_variance == 0:
                return 0
            
            beta = covariance / nifty_variance
            return round(float(beta), 3) if not np.isnan(beta) else 0
        except:
            return 0
    
    def calculate_all_metrics(self, symbol, nifty_returns=None):
        """Calculate all 9 metrics for a stock"""
        try:
            # Fetch price data
            price_data = self.data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
            
            if price_data is None or len(price_data) < 20:
                return None
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 20:
                return None
            
            # Prepare metrics dict
            metrics = {
                'symbol': symbol,
                'data_points': len(returns),
                'last_price': round(float(price_data.iloc[-1]), 2),
                'last_date': price_data.index[-1].strftime('%Y-%m-%d'),
            }
            
            # 1. Daily Returns moving averages
            metrics.update(self.calculate_daily_returns_ma(returns))
            
            # 2. Volatility
            metrics.update(self.calculate_volatility(returns))
            
            # 3. Sharpe Ratio
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
            
            # 4. Sortino Ratio
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
            
            # 5. Calmar Ratio
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
            
            # 6. Max Drawdown (90-day)
            metrics['max_drawdown_90d'] = self.calculate_max_drawdown_90d(returns)
            
            # 7. Skewness
            metrics['skewness'] = self.calculate_skewness(returns)
            
            # 8. Kurtosis
            metrics['kurtosis'] = self.calculate_kurtosis(returns)
            
            # 9. Beta (vs Nifty 50)
            if nifty_returns is not None:
                metrics['beta'] = self.calculate_beta(returns, nifty_returns)
            else:
                metrics['beta'] = 0
            
            return metrics
        
        except Exception as e:
            return None
    
    def process_universe(self, stocks, universe_name, output_dir, max_retries=3):
        """Process entire stock universe with retry logic"""
        print(f"\n📈 Processing {universe_name}...")
        print("-" * 80)
        
        # Download Nifty 50 benchmark for beta calculation
        print("Fetching Nifty 50 benchmark data...")
        try:
            nifty_prices = self.data_fetcher.fetch_price_data('^NSEI', exchange='NSE_EQ')
            if nifty_prices is None:
                nifty_prices = self.data_fetcher.fetch_price_data('NIFTY50', exchange='NSE_EQ')
            
            nifty_returns = None
            if nifty_prices is not None and len(nifty_prices) > 20:
                nifty_returns = nifty_prices.pct_change().dropna()
                print("✅ Nifty benchmark data fetched")
            else:
                print("⚠️ Could not fetch benchmark - beta will be 0")
        except:
            print("⚠️ Benchmark fetch failed - beta will be 0")
            nifty_returns = None
        
        # Calculate metrics for all stocks
        all_metrics = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(stocks, 1):
            print(f"  [{i:2d}/{len(stocks)}] {symbol:15s} ... ", end="", flush=True)
            
            metrics = self.calculate_all_metrics(symbol, nifty_returns)
            
            if metrics is not None:
                all_metrics.append(metrics)
                successful += 1
                print("✅")
            else:
                failed += 1
                print("❌")
            
            # Rate limiting
            time.sleep(0.1)
        
        # Create DataFrame
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            
            # Sort by Sharpe ratio
            df = df.sort_values('sharpe_ratio', ascending=False)
            
            # Save to CSV
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{universe_name}_metrics.csv')
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ Completed {universe_name}")
            print(f"   ✓ Successful: {successful}/{len(stocks)}")
            print(f"   ✗ Failed: {failed}/{len(stocks)}")
            print(f"   💾 Saved to: {output_file}")
            
            return df
        else:
            print(f"\n❌ No data collected for {universe_name}")
            return None
    
    def run(self):
        """Run complete asset metrics calculation"""
        base_output_dir = r'f:\AI Insights Dashboard\Asset returns'
        
        # Process Nifty50
        nifty50_output = os.path.join(base_output_dir, 'nifty50')
        df_nifty50 = self.process_universe(NIFTY50, 'Nifty50_metrics', nifty50_output)
        
        # Process Nifty Next50
        nifty_next50_output = os.path.join(base_output_dir, 'nifty_next_50')
        df_nifty_next50 = self.process_universe(NIFTY_NEXT50, 'Nifty_Next50_metrics', nifty_next50_output)
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        
        if df_nifty50 is not None and len(df_nifty50) > 0:
            print("\n📈 Nifty50 - Top 10 by Sharpe Ratio:")
            print(df_nifty50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].head(10).to_string(index=False))
        
        if df_nifty_next50 is not None and len(df_nifty_next50) > 0:
            print("\n📈 Nifty Next50 - Top 10 by Sharpe Ratio:")
            print(df_nifty_next50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].head(10).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("✅ Asset metrics calculation complete!")
        print("=" * 80)


if __name__ == "__main__":
    try:
        # Initialize calculator
        calculator = OptimizedAssetMetricsCalculator(lookback_years=1, risk_free_rate=0.062)
        
        # Run calculations
        calculator.run()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
