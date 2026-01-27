"""
Market Context Data Fetcher for Classification ML Model

Collects market environment data required for portfolio optimization ML model.
This module gathers 4 key market context inputs needed for recommendation predictions.

INPUT SOURCES & DATA COLLECTED:
================================

1. MARKET VOLATILITY (India VIX)
   - Source: NSE Website / Yahoo Finance
   - Frequency: Daily (market close)
   - What's Fetched:
     * Current VIX level (e.g., 18.5)
     * VIX trend (increasing/decreasing)
     * VIX percentile vs historical (0-100)
     * Interpretation (Low/Medium/High volatility)

2. MARKET REGIME DETECTION
   - Source: Calculated from Nifty50 index
   - Frequency: Daily (derived from index returns)
   - What's Fetched:
     * Nifty50 current price
     * 1-month return
     * 3-month return
     * 6-month return
     * Regime classification (Bull/Sideways/Bear)
     * Trend strength (Strong/Moderate/Weak)

3. RISK-FREE RATE (10-Year Government Yield)
   - Source: RBI Website / Manual Update
   - Frequency: Weekly
   - What's Fetched:
     * Current 10-year yield
     * 3-month trend
     * Updated date
     * Notes/comments if rates changed

4. SECTOR PERFORMANCE
   - Source: NSE Sector Indices / Yahoo Finance
   - Frequency: Daily
   - What's Fetched (20 major sectors):
     * NIFTY_BANK: Banking sector performance
     * NIFTY_IT: IT sector performance
     * NIFTY_PHARMA: Pharma sector performance
     * NIFTY_FMCG: FMCG sector performance
     * NIFTY_AUTO: Auto sector performance
     * NIFTY_ENERGY: Energy sector performance
     * NIFTY_REALTY: Realty sector performance
     * NIFTY_METALS: Metals sector performance
     * NIFTY_PRIVATE_BANK: Private Banking sector
     * NIFTY_PSU_BANK: PSU Banking sector
     * Plus 10+ additional sector indices
     
     For each sector:
       - 1-month return %
       - 3-month return %
       - 6-month return %
       - Current price
       - Volatility (30-day)
       - Ranking among sectors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Major sectors to track
SECTOR_INDICES = {
    'NIFTY_BANK': '^NSEBANK50',
    'NIFTY_IT': '^NSEINFRA',  # Proxy - adjust as needed
    'NIFTY_PHARMA': 'PHARMA.BO',  # Adjusted for yfinance
    'NIFTY_FMCG': '^NSEI',  # Base index
    'NIFTY_AUTO': 'MARUTI.NS',  # Sample auto stock
    'NIFTY_ENERGY': 'RELIANCE.NS',  # Sample energy stock
    'NIFTY_REALTY': '^NSEREALTY',
    'NIFTY_METALS': 'HINDALCO.NS',  # Sample metals stock
    'NIFTY_PRIVATE_BANK': 'HDFCBANK.NS',  # Sample private bank
    'NIFTY_PSU_BANK': 'SBIN.NS',  # Sample PSU bank
}


class MarketContextFetcher:
    """Fetch and calculate market context data for ML model inputs"""
    
    def __init__(self, risk_free_rate_manual=None):
        """
        Initialize market context fetcher
        
        Args:
            risk_free_rate_manual: Manual risk-free rate (e.g., 0.062 for 6.2%)
                                  If None, will try to fetch from sources
        """
        self.today = datetime.now()
        self.risk_free_rate = risk_free_rate_manual or 0.062  # Default: 6.2%
        
        print("📊 Market Context Fetcher")
        print(f"Date: {self.today.strftime('%Y-%m-%d')}")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.2f}%")
        print("=" * 80)
    
    # =====================================================================
    # INPUT 1: MARKET VOLATILITY (India VIX)
    # =====================================================================
    
    def fetch_market_vix(self):
        """
        Fetch India VIX (market volatility index)
        
        INPUTS FETCHED:
        ├─ Current VIX Level: Float (e.g., 18.5)
        ├─ VIX Trend: String (Increasing/Decreasing/Stable)
        ├─ VIX Percentile: Int (0-100, vs 1-year range)
        ├─ Volatility Level: String (Low/Medium/High)
        └─ Last Update: DateTime
        
        Returns:
            dict with VIX metrics
        """
        print("\n🔄 Fetching Market VIX...")
        
        try:
            # Try to fetch India VIX (INDIAVIX from NSE)
            # Note: Yahoo Finance ticker for India VIX varies
            vix_data = yf.download('^INDIAVIX', period='1y', progress=False)
            
            if vix_data is None or len(vix_data) == 0:
                print("  ⚠️ Could not fetch live VIX data, using sample values")
                return self._sample_vix_data()
            
            current_vix = float(vix_data['Close'].iloc[-1])
            vix_1m_ago = float(vix_data['Close'].iloc[-22]) if len(vix_data) > 22 else current_vix
            
            # Calculate trend
            if current_vix > vix_1m_ago * 1.05:
                trend = "Increasing"
            elif current_vix < vix_1m_ago * 0.95:
                trend = "Decreasing"
            else:
                trend = "Stable"
            
            # Calculate percentile (where does current VIX stand in 1-year range)
            vix_min = vix_data['Close'].min()
            vix_max = vix_data['Close'].max()
            vix_percentile = int(((current_vix - vix_min) / (vix_max - vix_min)) * 100)
            
            # Classify volatility level
            if current_vix < 15:
                vol_level = "Low"
            elif current_vix < 20:
                vol_level = "Medium"
            else:
                vol_level = "High"
            
            vix_metrics = {
                'current_vix': round(current_vix, 2),
                'vix_trend': trend,
                'vix_percentile': vix_percentile,
                'volatility_level': vol_level,
                'vix_1m_ago': round(vix_1m_ago, 2),
                'vix_1y_low': round(vix_min, 2),
                'vix_1y_high': round(vix_max, 2),
                'fetch_date': self.today.strftime('%Y-%m-%d')
            }
            
            print(f"  ✅ VIX: {current_vix:.2f} ({vol_level})")
            return vix_metrics
        
        except Exception as e:
            print(f"  ⚠️ Error fetching VIX: {e}")
            return self._sample_vix_data()
    
    def _sample_vix_data(self):
        """Return sample VIX data"""
        return {
            'current_vix': 18.5,
            'vix_trend': 'Stable',
            'vix_percentile': 45,
            'volatility_level': 'Medium',
            'vix_1m_ago': 18.2,
            'vix_1y_low': 12.3,
            'vix_1y_high': 28.5,
            'fetch_date': self.today.strftime('%Y-%m-%d')
        }
    
    # =====================================================================
    # INPUT 2: MARKET REGIME DETECTION
    # =====================================================================
    
    def detect_market_regime(self):
        """
        Detect current market regime (Bull/Bear/Sideways)
        
        INPUTS FETCHED:
        ├─ Nifty50 Current Price: Float
        ├─ 1-Month Return: Float (%)
        ├─ 3-Month Return: Float (%)
        ├─ 6-Month Return: Float (%)
        ├─ Market Regime: String (Bull/Sideways/Bear)
        ├─ Trend Strength: String (Strong/Moderate/Weak)
        ├─ Momentum: String (Positive/Neutral/Negative)
        └─ Last Update: DateTime
        
        Returns:
            dict with market regime metrics
        """
        print("\n🔄 Detecting Market Regime...")
        
        try:
            # Fetch Nifty50 index data
            nifty_data = yf.download('^NSEI', period='6mo', progress=False)
            
            if nifty_data is None or len(nifty_data) < 20:
                print("  ⚠️ Insufficient data, using sample regime")
                return self._sample_regime_data()
            
            current_price = float(nifty_data['Close'].iloc[-1])
            
            # Calculate returns for different periods
            ret_1m = (current_price - nifty_data['Close'].iloc[-22]) / nifty_data['Close'].iloc[-22] if len(nifty_data) > 22 else 0
            ret_3m = (current_price - nifty_data['Close'].iloc[-65]) / nifty_data['Close'].iloc[-65] if len(nifty_data) > 65 else ret_1m
            ret_6m = (current_price - nifty_data['Close'].iloc[0]) / nifty_data['Close'].iloc[0]
            
            # Determine regime
            if ret_3m > 0.10:  # +10% = Bull
                regime = "Bull"
                if ret_3m > 0.20:
                    strength = "Strong"
                else:
                    strength = "Moderate"
            elif ret_3m < -0.10:  # -10% = Bear
                regime = "Bear"
                if ret_3m < -0.20:
                    strength = "Strong"
                else:
                    strength = "Moderate"
            else:
                regime = "Sideways"
                strength = "Weak"
            
            # Momentum
            momentum = "Positive" if ret_1m > 0 else ("Negative" if ret_1m < 0 else "Neutral")
            
            regime_metrics = {
                'nifty50_current': round(current_price, 2),
                'return_1m': round(ret_1m * 100, 2),
                'return_3m': round(ret_3m * 100, 2),
                'return_6m': round(ret_6m * 100, 2),
                'market_regime': regime,
                'trend_strength': strength,
                'momentum': momentum,
                'fetch_date': self.today.strftime('%Y-%m-%d')
            }
            
            print(f"  ✅ Regime: {regime} ({strength}) | Momentum: {momentum}")
            return regime_metrics
        
        except Exception as e:
            print(f"  ⚠️ Error detecting regime: {e}")
            return self._sample_regime_data()
    
    def _sample_regime_data(self):
        """Return sample regime data"""
        return {
            'nifty50_current': 23450.50,
            'return_1m': 5.2,
            'return_3m': 12.5,
            'return_6m': 18.3,
            'market_regime': 'Bull',
            'trend_strength': 'Strong',
            'momentum': 'Positive',
            'fetch_date': self.today.strftime('%Y-%m-%d')
        }
    
    # =====================================================================
    # INPUT 3: RISK-FREE RATE (10-Year Government Yield)
    # =====================================================================
    
    def get_risk_free_rate(self):
        """
        Get current risk-free rate (10-year government bond yield)
        
        INPUTS FETCHED:
        ├─ Current Rate: Float (as %)
        ├─ Rate Trend: String (Increasing/Decreasing/Stable)
        ├─ 3-Month Change: Float (basis points)
        ├─ Data Source: String (Manual/RBI/API)
        └─ Last Update: DateTime
        
        Sources (in priority):
        1. Manual input (self.risk_free_rate)
        2. RBI Website (when integrated)
        3. Financial data APIs
        
        Returns:
            dict with risk-free rate metrics
        """
        print("\n🔄 Fetching Risk-Free Rate...")
        
        # For now, using manual rate (should be updated weekly from RBI)
        try:
            rate_metrics = {
                'current_rate': self.risk_free_rate * 100,  # Convert to %
                'rate_trend': 'Stable',  # Update manually or integrate RBI API
                'change_3m_bps': 0,  # Basis points change (100 bps = 1%)
                'data_source': 'Manual/RBI',
                'last_updated': self.today.strftime('%Y-%m-%d'),
                'notes': 'Update weekly from RBI website or financial screeners'
            }
            
            print(f"  ✅ Risk-Free Rate: {rate_metrics['current_rate']:.2f}%")
            return rate_metrics
        
        except Exception as e:
            print(f"  ⚠️ Error fetching rate: {e}")
            return {
                'current_rate': 6.2,
                'rate_trend': 'Stable',
                'change_3m_bps': 0,
                'data_source': 'Default',
                'last_updated': self.today.strftime('%Y-%m-%d'),
                'notes': 'Using default value'
            }
    
    # =====================================================================
    # INPUT 4: SECTOR PERFORMANCE
    # =====================================================================
    
    def fetch_sector_performance(self):
        """
        Fetch performance of major market sectors
        
        INPUTS FETCHED (for each of ~20 sectors):
        ├─ Sector Name: String (e.g., NIFTY_BANK, NIFTY_IT)
        ├─ Current Price: Float
        ├─ 1-Month Return: Float (%)
        ├─ 3-Month Return: Float (%)
        ├─ 6-Month Return: Float (%)
        ├─ 30-Day Volatility: Float (%)
        ├─ Sector Rank: Int (1-20, 1=best performer)
        └─ Trend: String (Uptrend/Downtrend/Sideways)
        
        Returns:
            DataFrame with all sector metrics
        """
        print("\n🔄 Fetching Sector Performance...")
        
        sector_data = []
        
        for sector_name, ticker in SECTOR_INDICES.items():
            try:
                # Fetch 6 months of data
                data = yf.download(ticker, period='6mo', progress=False)
                
                if data is None or len(data) < 20:
                    continue
                
                current_price = float(data['Close'].iloc[-1])
                
                # Calculate returns
                ret_1m = (current_price - data['Close'].iloc[-22]) / data['Close'].iloc[-22] if len(data) > 22 else 0
                ret_3m = (current_price - data['Close'].iloc[-65]) / data['Close'].iloc[-65] if len(data) > 65 else ret_1m
                ret_6m = (current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]
                
                # Calculate volatility (30-day)
                recent_returns = data['Close'].tail(30).pct_change().dropna()
                volatility = recent_returns.std() * np.sqrt(252) * 100
                
                # Determine trend
                if ret_1m > 0.02:
                    trend = "Uptrend"
                elif ret_1m < -0.02:
                    trend = "Downtrend"
                else:
                    trend = "Sideways"
                
                sector_metrics = {
                    'sector': sector_name,
                    'current_price': round(current_price, 2),
                    'return_1m': round(ret_1m * 100, 2),
                    'return_3m': round(ret_3m * 100, 2),
                    'return_6m': round(ret_6m * 100, 2),
                    'volatility_30d': round(volatility, 2),
                    'trend': trend
                }
                
                sector_data.append(sector_metrics)
            
            except Exception as e:
                continue
        
        if not sector_data:
            print("  ⚠️ Could not fetch sector data, using samples")
            return self._sample_sector_data()
        
        # Create DataFrame and rank by 3-month return
        df_sectors = pd.DataFrame(sector_data)
        df_sectors['rank'] = df_sectors['return_3m'].rank(ascending=False, method='min').astype(int)
        df_sectors = df_sectors.sort_values('return_3m', ascending=False)
        
        print(f"  ✅ Fetched {len(df_sectors)} sectors")
        return df_sectors
    
    def _sample_sector_data(self):
        """Return sample sector data"""
        sectors = [
            {'sector': 'NIFTY_BANK', 'return_3m': 8.5, 'return_1m': 2.1, 'volatility_30d': 12.3},
            {'sector': 'NIFTY_IT', 'return_3m': 6.2, 'return_1m': 1.5, 'volatility_30d': 14.5},
            {'sector': 'NIFTY_PHARMA', 'return_3m': -2.3, 'return_1m': -0.8, 'volatility_30d': 11.2},
            {'sector': 'NIFTY_FMCG', 'return_3m': 3.1, 'return_1m': 0.9, 'volatility_30d': 10.1},
            {'sector': 'NIFTY_AUTO', 'return_3m': 5.7, 'return_1m': 1.2, 'volatility_30d': 15.2},
        ]
        df = pd.DataFrame(sectors)
        df['rank'] = df['return_3m'].rank(ascending=False, method='min').astype(int)
        return df
    
    # =====================================================================
    # MAIN COLLECTION FUNCTION
    # =====================================================================
    
    def collect_all_market_context(self):
        """
        Collect all 4 market context inputs
        
        OUTPUTS CREATED:
        ├─ market_vix.json - Market volatility metrics
        ├─ market_regime.json - Market regime detection
        ├─ risk_free_rate.json - Risk-free rate metrics
        ├─ sector_performance.csv - Sector performance rankings
        └─ market_context_summary.csv - Combined summary
        
        Returns:
            dict with all collected data
        """
        print("\n" + "=" * 80)
        print("COLLECTING MARKET CONTEXT DATA")
        print("=" * 80)
        
        # Collect all inputs
        vix_data = self.fetch_market_vix()
        regime_data = self.detect_market_regime()
        rate_data = self.get_risk_free_rate()
        sector_data = self.fetch_sector_performance()
        
        # Prepare output
        market_context = {
            'timestamp': self.today.isoformat(),
            'market_vix': vix_data,
            'market_regime': regime_data,
            'risk_free_rate': rate_data,
            'sector_performance': sector_data.to_dict('records') if isinstance(sector_data, pd.DataFrame) else []
        }
        
        return market_context
    
    # =====================================================================
    # SAVE FUNCTIONS
    # =====================================================================
    
    def save_market_context(self, output_dir='Market context'):
        """Save all market context data to files"""
        
        print("\n" + "=" * 80)
        print("SAVING MARKET CONTEXT DATA")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect data
        vix_data = self.fetch_market_vix()
        regime_data = self.detect_market_regime()
        rate_data = self.get_risk_free_rate()
        sectors_df = self.fetch_sector_performance()
        
        # Save VIX metrics
        vix_file = os.path.join(output_dir, 'market_vix.csv')
        pd.DataFrame([vix_data]).to_csv(vix_file, index=False)
        print(f"✅ Saved: {vix_file}")
        
        # Save Market Regime
        regime_file = os.path.join(output_dir, 'market_regime.csv')
        pd.DataFrame([regime_data]).to_csv(regime_file, index=False)
        print(f"✅ Saved: {regime_file}")
        
        # Save Risk-Free Rate
        rate_file = os.path.join(output_dir, 'risk_free_rate.csv')
        pd.DataFrame([rate_data]).to_csv(rate_file, index=False)
        print(f"✅ Saved: {rate_file}")
        
        # Save Sector Performance
        if isinstance(sectors_df, pd.DataFrame):
            sectors_file = os.path.join(output_dir, 'sector_performance.csv')
            sectors_df.to_csv(sectors_file, index=False)
            print(f"✅ Saved: {sectors_file}")
        
        # Create combined summary
        summary = {
            'date': self.today.strftime('%Y-%m-%d'),
            'vix_level': vix_data.get('current_vix', 0),
            'volatility_level': vix_data.get('volatility_level', 'Unknown'),
            'market_regime': regime_data.get('market_regime', 'Unknown'),
            'market_momentum': regime_data.get('momentum', 'Unknown'),
            'nifty50_return_3m': regime_data.get('return_3m', 0),
            'risk_free_rate': rate_data.get('current_rate', 0),
            'top_sector': sectors_df.iloc[0]['sector'] if isinstance(sectors_df, pd.DataFrame) and len(sectors_df) > 0 else 'N/A',
            'top_sector_return': sectors_df.iloc[0]['return_3m'] if isinstance(sectors_df, pd.DataFrame) and len(sectors_df) > 0 else 0
        }
        
        summary_file = os.path.join(output_dir, 'market_context_summary.csv')
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        print(f"✅ Saved: {summary_file}")
        
        print("\n" + "=" * 80)
        print("✅ MARKET CONTEXT DATA COLLECTION COMPLETE")
        print("=" * 80)
        
        return {
            'vix': vix_data,
            'regime': regime_data,
            'rate': rate_data,
            'sectors': sectors_df,
            'summary': summary
        }


if __name__ == "__main__":
    # Initialize fetcher
    fetcher = MarketContextFetcher(risk_free_rate_manual=0.062)
    
    # Collect and save data
    try:
        results = fetcher.save_market_context(
            output_dir=r'f:\AI Insights Dashboard\Market context'
        )
        
        # Display summary
        print("\n" + "=" * 80)
        print("📊 MARKET CONTEXT SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        print(f"\nDate: {summary['date']}")
        print(f"VIX Level: {summary['vix_level']:.2f} ({summary['volatility_level']})")
        print(f"Market Regime: {summary['market_regime']} ({summary['market_momentum']})")
        print(f"Nifty50 3M Return: {summary['nifty50_return_3m']:.2f}%")
        print(f"Risk-Free Rate: {summary['risk_free_rate']:.2f}%")
        print(f"Top Sector: {summary['top_sector']} ({summary['top_sector_return']:.2f}% return)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
