"""
Calculate Asset-Level Input Metrics - Standalone Version
Uses yfinance for reliable data collection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Nifty50 stocks with .NS suffix for yfinance
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


class StandaloneAssetMetricsCalculator:
    """Calculate asset metrics using yfinance"""
    
    def __init__(self, lookback_years=1, risk_free_rate=0.062):
        """Initialize"""
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * lookback_years)
        
        print(f"📊 Standalone Asset Metrics Calculator")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print("=" * 80)
    
    def fetch_data(self, symbol, max_retries=3):
        """Fetch price data with retry logic"""
        import yfinance as yf
        
        for attempt in range(max_retries):
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, quiet=True)
                if not data.empty and len(data) > 20:
                    return data['Close']
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                continue
        
        return None
    
    def calculate_metrics(self, symbol, nifty_returns=None):
        """Calculate all metrics"""
        symbol_clean = symbol.replace('.NS', '')
        
        try:
            # Fetch data
            prices = self.fetch_data(symbol)
            if prices is None or len(prices) < 20:
                return None
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            if len(returns) < 20:
                return None
            
            # Initialize metrics
            metrics = {
                'symbol': symbol_clean,
                'data_points': len(returns),
                'last_price': round(float(prices.iloc[-1]), 2),
                'last_date': prices.index[-1].strftime('%Y-%m-%d'),
            }
            
            # 1. Daily Returns MA
            try:
                ma_5d = returns.rolling(5).mean().iloc[-1] * 100
                ma_20d = returns.rolling(20).mean().iloc[-1] * 100
                ma_60d = returns.rolling(60).mean().iloc[-1] * 100
                metrics['returns_5d_ma'] = round(float(ma_5d), 3) if not np.isnan(ma_5d) else 0
                metrics['returns_20d_ma'] = round(float(ma_20d), 3) if not np.isnan(ma_20d) else 0
                metrics['returns_60d_ma'] = round(float(ma_60d), 3) if not np.isnan(ma_60d) else 0
            except:
                metrics['returns_5d_ma'] = 0
                metrics['returns_20d_ma'] = 0
                metrics['returns_60d_ma'] = 0
            
            # 2. Volatility
            try:
                vol_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
                vol_90d = returns.rolling(90).std().iloc[-1] * np.sqrt(252) * 100
                metrics['volatility_30d'] = round(float(vol_30d), 2) if not np.isnan(vol_30d) else 0
                metrics['volatility_90d'] = round(float(vol_90d), 2) if not np.isnan(vol_90d) else 0
            except:
                metrics['volatility_30d'] = 0
                metrics['volatility_90d'] = 0
            
            # 3. Sharpe Ratio
            try:
                ann_return = returns.mean() * 252
                ann_vol = returns.std() * np.sqrt(252)
                sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
                metrics['sharpe_ratio'] = round(float(sharpe), 3) if not np.isnan(sharpe) else 0
            except:
                metrics['sharpe_ratio'] = 0
            
            # 4. Sortino Ratio
            try:
                ann_return = returns.mean() * 252
                downside_ret = returns[returns < 0]
                downside_vol = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else returns.std() * np.sqrt(252)
                sortino = (ann_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
                metrics['sortino_ratio'] = round(float(sortino), 3) if not np.isnan(sortino) else 0
            except:
                metrics['sortino_ratio'] = 0
            
            # 5. Calmar Ratio
            try:
                ann_return = returns.mean() * 252
                cum_ret = (1 + returns).cumprod()
                peak = cum_ret.cummax()
                dd = (cum_ret - peak) / peak
                max_dd = abs(dd.min())
                calmar = ann_return / max_dd if max_dd > 0.001 else 0
                metrics['calmar_ratio'] = round(float(calmar), 3) if not np.isnan(calmar) else 0
            except:
                metrics['calmar_ratio'] = 0
            
            # 6. Max Drawdown (90-day)
            try:
                recent_ret = returns.tail(90)
                cum_ret = (1 + recent_ret).cumprod()
                peak = cum_ret.cummax()
                dd = (cum_ret - peak) / peak
                max_dd = abs(dd.min()) * 100
                metrics['max_drawdown_90d'] = round(float(max_dd), 2) if not np.isnan(max_dd) else 0
            except:
                metrics['max_drawdown_90d'] = 0
            
            # 7. Skewness
            try:
                from scipy.stats import skew
                skewness = skew(returns.dropna())
                metrics['skewness'] = round(float(skewness), 3) if not np.isnan(skewness) else 0
            except:
                metrics['skewness'] = 0
            
            # 8. Kurtosis
            try:
                from scipy.stats import kurtosis
                kurt = kurtosis(returns.dropna())
                metrics['kurtosis'] = round(float(kurt), 3) if not np.isnan(kurt) else 0
            except:
                metrics['kurtosis'] = 0
            
            # 9. Beta
            try:
                if nifty_returns is not None:
                    common_dates = returns.index.intersection(nifty_returns.index)
                    if len(common_dates) > 20:
                        stock_ret = returns.loc[common_dates].values
                        nifty_ret = nifty_returns.loc[common_dates].values
                        covariance = np.cov(stock_ret, nifty_ret)[0, 1]
                        nifty_var = np.var(nifty_ret)
                        beta = covariance / nifty_var if nifty_var > 0 else 0
                        metrics['beta'] = round(float(beta), 3) if not np.isnan(beta) else 0
                    else:
                        metrics['beta'] = 0
                else:
                    metrics['beta'] = 0
            except:
                metrics['beta'] = 0
            
            return metrics
        
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def process_universe(self, stocks, name, output_dir):
        """Process universe"""
        print(f"\n📈 Processing {name}...")
        print("-" * 80)
        
        # Get Nifty benchmark
        print("Fetching Nifty 50 benchmark...")
        nifty_prices = self.fetch_data('^NSEI')
        nifty_returns = nifty_prices.pct_change().dropna() if nifty_prices is not None else None
        if nifty_returns is not None:
            print("✅ Benchmark data fetched")
        else:
            print("⚠️ Benchmark failed")
        
        # Process stocks
        all_metrics = []
        success = 0
        failed = 0
        
        for i, symbol in enumerate(stocks, 1):
            print(f"  [{i:2d}/{len(stocks)}] {symbol:15s} ", end="", flush=True)
            
            metrics = self.calculate_metrics(symbol, nifty_returns)
            
            if metrics is not None:
                all_metrics.append(metrics)
                success += 1
                print("✅")
            else:
                failed += 1
                print("❌")
            
            time.sleep(0.1)
        
        # Save results
        if all_metrics:
            df = pd.DataFrame(all_metrics).sort_values('sharpe_ratio', ascending=False)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{name}_metrics.csv')
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ {name}: {success} success, {failed} failed")
            print(f"   Saved: {output_file}\n")
            
            # Top 10
            print(f"Top 10 {name} by Sharpe Ratio:")
            print(df[['symbol', 'sharpe_ratio', 'volatility_30d', 'beta']].head(10).to_string(index=False))
            print()
            
            return df
        
        return None
    
    def run(self):
        """Run"""
        base_dir = r'f:\AI Insights Dashboard\Asset returns'
        
        df_n50 = self.process_universe(NIFTY50, 'Nifty50_metrics', os.path.join(base_dir, 'nifty50'))
        df_n50n = self.process_universe(NIFTY_NEXT50, 'Nifty_Next50_metrics', os.path.join(base_dir, 'nifty_next_50'))
        
        print("=" * 80)
        print("✅ Complete!")
        print("=" * 80)


if __name__ == "__main__":
    import yfinance as yf
    
    calc = StandaloneAssetMetricsCalculator(lookback_years=1, risk_free_rate=0.062)
    calc.run()
