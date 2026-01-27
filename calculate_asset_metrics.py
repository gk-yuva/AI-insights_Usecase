"""
Calculate Asset-Level Input Metrics for Classification ML Model

This script calculates 9 key metrics for all Nifty50 and Nifty Next50 stocks:
1. Daily Returns (5-day, 20-day, 60-day averages)
2. Volatility (30-day rolling, 90-day rolling)
3. Sharpe Ratio (annualized)
4. Sortino Ratio (downside-adjusted return)
5. Calmar Ratio (return/max drawdown)
6. Maximum Drawdown (90-day period)
7. Skewness (return distribution skew)
8. Kurtosis (return distribution tail risk)
9. Beta (relative to Nifty 50 benchmark)

Output: CSV files saved to Asset returns/nifty50/ and Asset returns/nifty_next_50/
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Nifty50 and Nifty Next50 stocks
NIFTY50 = [
    'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS',
    'BPCL.NS', 'BHARTIARTL.NS', 'BOSCHIND.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'COALINDIA.NS', 'COLPAL.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
    'GAIL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS',
    'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
    'IPCALAB.NS', 'INDIGO.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS',
    'KOTAKBANK.NS', 'LT.NS', 'LTTS.NS', 'M&M.NS', 'MARUTI.NS',
    'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',
    'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS',
    'TATAPOWER.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS',
    'UPL.NS', 'VBL.NS', 'WIPRO.NS'
]

NIFTY_NEXT50 = [
    'ABB.NS', 'AARTIIND.NS', 'ABCAPITAL.NS', 'ABSL.NS', 'ADANIENT.NS',
    'ADANIGREEN.NS', 'ADANIPOWER.NS', 'AMARAJABAT.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS',
    'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'AUROPHARMA.NS', 'AUBANK.NS', 'AUTOCLAD.NS',
    'AUTOIND.NS', 'AXISMF.NS', 'BAJAJTRIPUR.NS', 'BASF.NS', 'BATAINDIA.NS',
    'BEL.NS', 'BERGEPAINT.NS', 'BHARATRAS.NS', 'BHEL.NS', 'BLS.NS',
    'BRIGADE.NS', 'CADILAHC.NS', 'CAMS.NS', 'CARBORUNM.NS', 'CASTROL.NS',
    'CGCONSTRUCT.NS', 'CGPOWER.NS', 'CHAMBLFERT.NS', 'CHEMPLAST.NS', 'CHOLAFIN.NS',
    'CHROMIND.NS', 'COLDRAKE.NS', 'COALGO.NS', 'CUMMINSIND.NS', 'DCBBANK.NS',
    'DEEPAKNTR.NS', 'DIALBRANDS.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'EXIDEIND.NS'
]

class AssetMetricsCalculator:
    """Calculate asset-level metrics for ML model inputs"""
    
    def __init__(self, lookback_years=1, risk_free_rate=0.062):
        """
        Initialize calculator
        
        Args:
            lookback_years: Number of years of historical data (default: 1 year)
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations (default: 6.2%)
        """
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * lookback_years)
        
        print(f"📊 Asset Metrics Calculator")
        print(f"Data Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print("=" * 80)
    
    def download_price_data(self, symbol):
        """Download historical price data for a stock"""
        try:
            data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            if data.empty or len(data) < 20:
                return None
            return data['Close']
        except Exception as e:
            print(f"  ⚠️ Error downloading {symbol}: {str(e)}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def calculate_daily_returns_ma(self, returns):
        """Calculate 5-day, 20-day, 60-day moving averages of daily returns"""
        ma_5d = returns.rolling(window=5).mean().iloc[-1]
        ma_20d = returns.rolling(window=20).mean().iloc[-1]
        ma_60d = returns.rolling(window=60).mean().iloc[-1]
        
        return {
            'returns_5d_ma': ma_5d if not np.isnan(ma_5d) else 0,
            'returns_20d_ma': ma_20d if not np.isnan(ma_20d) else 0,
            'returns_60d_ma': ma_60d if not np.isnan(ma_60d) else 0
        }
    
    def calculate_volatility(self, returns):
        """Calculate 30-day and 90-day rolling volatility (annualized)"""
        vol_30d = returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)
        vol_90d = returns.rolling(window=90).std().iloc[-1] * np.sqrt(252)
        
        return {
            'volatility_30d': vol_30d * 100 if not np.isnan(vol_30d) else 0,
            'volatility_90d': vol_90d * 100 if not np.isnan(vol_90d) else 0
        }
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 20:
            return 0
        
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        
        if ann_vol == 0:
            return 0
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        return sharpe if not np.isnan(sharpe) else 0
    
    def calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 20:
            return 0
        
        ann_return = returns.mean() * 252
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            downside_vol = returns.std() * np.sqrt(252)
        else:
            downside_vol = downside_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return 0
        
        sortino = (ann_return - self.risk_free_rate) / downside_vol
        return sortino if not np.isnan(sortino) else 0
    
    def calculate_calmar_ratio(self, returns, prices):
        """Calculate Calmar ratio (return / max drawdown)"""
        ann_return = returns.mean() * 252
        
        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = abs(drawdown.min())
        
        if max_dd == 0:
            return 0
        
        calmar = ann_return / max_dd
        return calmar if not np.isnan(calmar) else 0
    
    def calculate_max_drawdown_90d(self, returns):
        """Calculate maximum drawdown over 90-day window"""
        recent_returns = returns.tail(90)
        cum_returns = (1 + recent_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = abs(drawdown.min()) * 100
        
        return max_dd if not np.isnan(max_dd) else 0
    
    def calculate_skewness(self, returns):
        """Calculate skewness of return distribution"""
        from scipy import stats
        skew = stats.skew(returns.dropna())
        return skew if not np.isnan(skew) else 0
    
    def calculate_kurtosis(self, returns):
        """Calculate kurtosis (excess kurtosis) of return distribution"""
        from scipy import stats
        kurt = stats.kurtosis(returns.dropna())
        return kurt if not np.isnan(kurt) else 0
    
    def calculate_beta(self, stock_returns, nifty_returns):
        """Calculate beta relative to Nifty 50"""
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
        return beta if not np.isnan(beta) else 0
    
    def calculate_all_metrics(self, symbol, nifty_returns=None):
        """Calculate all 9 metrics for a stock"""
        print(f"Calculating metrics for {symbol}...", end="")
        
        # Download price data
        prices = self.download_price_data(symbol)
        if prices is None:
            print(" ❌ Failed to download data")
            return None
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        if len(returns) < 20:
            print(" ❌ Insufficient data")
            return None
        
        try:
            # Calculate all metrics
            metrics = {
                'symbol': symbol.replace('.NS', ''),
                'data_points': len(returns),
                'last_price': prices.iloc[-1],
                'last_date': prices.index[-1].strftime('%Y-%m-%d'),
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
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, prices)
            
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
            
            print(" ✅")
            return metrics
        
        except Exception as e:
            print(f" ❌ Error calculating metrics: {str(e)}")
            return None
    
    def process_universe(self, stocks, universe_name, output_dir):
        """Process entire stock universe"""
        print(f"\n📈 Processing {universe_name}...")
        print("-" * 80)
        
        # Download Nifty 50 benchmark for beta calculation
        print("Downloading Nifty 50 benchmark data...")
        nifty_prices = self.download_price_data('^NSEBANK50')  # Alternative: '^NSEI' for broader index
        if nifty_prices is None:
            nifty_prices = self.download_price_data('^NSEI')  # Try broader index
        
        nifty_returns = None
        if nifty_prices is not None:
            nifty_returns = self.calculate_returns(nifty_prices)
            print("✅ Nifty benchmark data downloaded")
        else:
            print("⚠️ Could not download benchmark - beta will be 0")
        
        # Calculate metrics for all stocks
        all_metrics = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(stocks, 1):
            metrics = self.calculate_all_metrics(symbol, nifty_returns)
            
            if metrics is not None:
                all_metrics.append(metrics)
                successful += 1
            else:
                failed += 1
            
            if i % 10 == 0:
                print(f"Progress: {i}/{len(stocks)} stocks processed...")
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'{universe_name}_metrics.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ Completed {universe_name}")
        print(f"   Successful: {successful}/{len(stocks)}")
        print(f"   Failed: {failed}/{len(stocks)}")
        print(f"   Saved to: {output_file}")
        
        return df
    
    def run(self):
        """Run complete asset metrics calculation"""
        base_output_dir = r'f:\AI Insights Dashboard\Asset returns'
        
        # Process Nifty50
        nifty50_output = os.path.join(base_output_dir, 'nifty50')
        df_nifty50 = self.process_universe(NIFTY50, 'Nifty50', nifty50_output)
        
        # Process Nifty Next50
        nifty_next50_output = os.path.join(base_output_dir, 'nifty_next_50')
        df_nifty_next50 = self.process_universe(NIFTY_NEXT50, 'Nifty Next50', nifty_next50_output)
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        
        print("\nNifty50 Metrics Summary:")
        print(df_nifty50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].sort_values('sharpe_ratio', ascending=False).head(10))
        
        print("\nNifty Next50 Metrics Summary:")
        print(df_nifty_next50[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta', 'max_drawdown_90d']].sort_values('sharpe_ratio', ascending=False).head(10))
        
        print("\n✅ Asset metrics calculation complete!")
        print(f"Output files:")
        print(f"  - {os.path.join(nifty50_output, 'Nifty50_metrics.csv')}")
        print(f"  - {os.path.join(nifty_next50_output, 'Nifty Next50_metrics.csv')}")


if __name__ == "__main__":
    try:
        # Initialize calculator
        calculator = AssetMetricsCalculator(lookback_years=1, risk_free_rate=0.062)
        
        # Run calculations
        calculator.run()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
