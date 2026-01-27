"""
Feature Consolidation Module
Combines all 4 input types (Asset, Market, Portfolio, Investor) into unified feature matrix
for ML model training
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureConsolidator:
    """
    Consolidates features from 4 sources:
    1. Asset-Level Inputs (9 metrics × 103 stocks)
    2. Market Context Inputs (4 components)
    3. Portfolio-Level Inputs (holdings, metrics, concentration)
    4. Investor Profile Inputs (risk indices, time horizon)
    """
    
    def __init__(self, base_path: str = "f:\\AI Insights Dashboard"):
        """
        Initialize consolidator with paths to all data sources
        
        Args:
            base_path: Base path to workspace
        """
        self.base_path = Path(base_path)
        self.asset_returns_path = self.base_path / "Asset returns"
        self.market_context_path = self.base_path / "Market context"
        self.data_path = self.base_path / "data"
        
        # Load all data
        self.asset_metrics_nifty50 = self._load_asset_metrics("nifty50")
        self.asset_metrics_next50 = self._load_asset_metrics("nifty_next_50")
        self.market_context = self._load_market_context()
        self.portfolio_data = self._load_portfolio_data()
        self.investor_profile = self._load_investor_profile()
        
    def _load_asset_metrics(self, fund_name: str) -> pd.DataFrame:
        """Load asset metrics CSV for given fund (nifty50 or nifty_next_50)"""
        csv_path = self.asset_returns_path / fund_name / f"{fund_name.replace('_', ' ').title()}_metrics.csv"
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            # Use sample data if file doesn't exist
            print(f"⚠️  {csv_path} not found, creating sample data...")
            return self._create_sample_asset_metrics(fund_name)
    
    def _create_sample_asset_metrics(self, fund_name: str) -> pd.DataFrame:
        """Create sample asset metrics if real data not available"""
        if fund_name == "nifty50":
            stocks = ['INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK',
                     'AXISBANK', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'TATASTEEL', 'JSWSTEEL',
                     'HINDALCO', 'GRASIM', 'WIPRO', 'TECHM', 'LT', 'POWERGRID']
            num_stocks = 18
        else:
            stocks = ['ABB', 'AARTIIND', 'ABCAPITAL', 'APOLLOHOSP', 'ASHOKLEY',
                     'AUROPHARMA', 'AUBANK', 'BERGEPAINT', 'BHARATRAS', 'BHEL']
            num_stocks = 10
        
        data = {
            'symbol': stocks,
            'data_points': [252] * num_stocks,
            'last_price': np.random.uniform(50, 500, num_stocks),
            'returns_5d_ma': np.random.uniform(-2, 3, num_stocks),
            'returns_20d_ma': np.random.uniform(-1, 2, num_stocks),
            'returns_60d_ma': np.random.uniform(5, 25, num_stocks),
            'volatility_30d': np.random.uniform(8, 25, num_stocks),
            'volatility_90d': np.random.uniform(8, 25, num_stocks),
            'sharpe_ratio': np.random.uniform(0.3, 1.5, num_stocks),
            'sortino_ratio': np.random.uniform(0.4, 2.0, num_stocks),
            'calmar_ratio': np.random.uniform(0.2, 1.5, num_stocks),
            'max_drawdown_90d': -np.random.uniform(5, 30, num_stocks),
            'skewness': np.random.uniform(-1, 1, num_stocks),
            'kurtosis': np.random.uniform(2, 6, num_stocks),
            'beta': np.random.uniform(0.7, 1.5, num_stocks),
        }
        return pd.DataFrame(data)
    
    def _load_market_context(self) -> Dict:
        """Load all market context components"""
        market_data = {}
        
        # VIX metrics
        vix_path = self.market_context_path / "market_vix.csv"
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path)
            market_data['vix'] = {
                'current_vix': vix_df['current_vix'].iloc[0] if len(vix_df) > 0 else 18.5,
                'volatility_level': vix_df['volatility_level'].iloc[0] if len(vix_df) > 0 else 'Medium',
                'vix_percentile': vix_df['vix_percentile'].iloc[0] if len(vix_df) > 0 else 45,
            }
        else:
            market_data['vix'] = {'current_vix': 18.5, 'volatility_level': 'Medium', 'vix_percentile': 45}
        
        # Market regime
        regime_path = self.market_context_path / "market_regime.csv"
        if regime_path.exists():
            regime_df = pd.read_csv(regime_path)
            market_data['regime'] = {
                'nifty50_level': regime_df['nifty50_current'].iloc[0] if len(regime_df) > 0 else 23450,
                'return_1m': regime_df['return_1m'].iloc[0] if len(regime_df) > 0 else 5.2,
                'return_3m': regime_df['return_3m'].iloc[0] if len(regime_df) > 0 else 12.5,
                'market_regime': regime_df['market_regime'].iloc[0] if len(regime_df) > 0 else 'Bull',
            }
        else:
            market_data['regime'] = {'nifty50_level': 23450, 'return_1m': 5.2, 'return_3m': 12.5, 'market_regime': 'Bull'}
        
        # Risk-free rate
        rfr_path = self.market_context_path / "risk_free_rate.csv"
        if rfr_path.exists():
            rfr_df = pd.read_csv(rfr_path)
            market_data['rfr'] = {
                'current_rate': rfr_df['current_rate'].iloc[0] if len(rfr_df) > 0 else 6.2,
            }
        else:
            market_data['rfr'] = {'current_rate': 6.2}
        
        # Sector performance
        sector_path = self.market_context_path / "sector_performance.csv"
        if sector_path.exists():
            sector_df = pd.read_csv(sector_path)
            market_data['sectors'] = sector_df.to_dict('records') if len(sector_df) > 0 else []
        else:
            market_data['sectors'] = []
        
        return market_data
    
    def _load_portfolio_data(self) -> Dict:
        """Load portfolio data and compute metrics"""
        portfolio_path = self.data_path / "latest_portfolio.json"
        
        if portfolio_path.exists():
            with open(portfolio_path, 'r') as f:
                portfolio_list = json.load(f)
        else:
            portfolio_list = []
        
        if not portfolio_list:
            return {
                'num_holdings': 0,
                'portfolio_value': 0,
                'cash_position': 0.0,
                'sector_concentration_top3': 0,
                'asset_class_concentration': {},
                'avg_weight': 0,
                'portfolio_volatility': 0,
            }
        
        portfolio_df = pd.DataFrame(portfolio_list)
        total_value = portfolio_df['Cur. val'].sum()
        
        # Compute concentration metrics
        portfolio_df['weight'] = portfolio_df['Cur. val'] / total_value
        
        # Top 3 sectors concentration
        sector_concentration = portfolio_df.groupby('Sector')['weight'].sum().nlargest(3).sum()
        
        # Asset class breakdown
        asset_class_weights = portfolio_df.groupby('Asset Class')['weight'].sum().to_dict()
        
        # Average weight per holding
        avg_weight = portfolio_df['weight'].mean()
        
        # Estimate portfolio volatility (proxy: weighted average of stock volatilities)
        portfolio_volatility = np.random.uniform(10, 20)  # Placeholder
        
        return {
            'num_holdings': len(portfolio_df),
            'portfolio_value': total_value,
            'cash_position': 0.0,
            'sector_concentration_top3': sector_concentration,
            'asset_class_equity_pct': asset_class_weights.get('Equity', 0),
            'asset_class_commodity_pct': asset_class_weights.get('Commodity', 0),
            'avg_weight': avg_weight,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe': np.random.uniform(0.5, 1.5),
            'portfolio_max_drawdown': np.random.uniform(-20, -5),
        }
    
    def _load_investor_profile(self) -> Dict:
        """Load investor profile indices"""
        profile_path = self.data_path / "latest_investor_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                profile = json.load(f)
                return {
                    'risk_capacity_index': profile.get('indices', {}).get('risk_capacity_index', 50),
                    'risk_tolerance_index': profile.get('indices', {}).get('risk_tolerance_index', 50),
                    'behavioral_fragility_index': profile.get('indices', {}).get('behavioral_fragility_index', 50),
                    'time_horizon_strength': profile.get('indices', {}).get('time_horizon_strength', 50),
                    'effective_risk_tolerance': profile.get('effective_risk_tolerance', 40),
                    'time_horizon_years': profile.get('time_horizon_years', 20),
                    'age_band': profile.get('age_band', '30-35'),
                    'employment_type': profile.get('employment_type', 'salaried'),
                }
        else:
            return {
                'risk_capacity_index': 50,
                'risk_tolerance_index': 50,
                'behavioral_fragility_index': 50,
                'time_horizon_strength': 50,
                'effective_risk_tolerance': 40,
                'time_horizon_years': 20,
                'age_band': '30-35',
                'employment_type': 'salaried',
            }
    
    def _extract_sector_features(self, sectors_data: list) -> Dict:
        """Extract sector performance features"""
        if not sectors_data:
            return {
                'top_sector_return': 0,
                'bottom_sector_return': 0,
                'sector_return_dispersion': 0,
            }
        
        sector_df = pd.DataFrame(sectors_data)
        
        if 'return_3m' in sector_df.columns:
            returns = sector_df['return_3m'].values
            return {
                'top_sector_return': float(returns.max()),
                'bottom_sector_return': float(returns.min()),
                'sector_return_dispersion': float(returns.std()),
            }
        return {}
    
    def consolidate_features(self) -> pd.DataFrame:
        """
        Consolidate all 4 input types into unified feature matrix
        
        Returns:
            DataFrame with consolidated features for ML model
        """
        print("\n" + "="*80)
        print("CONSOLIDATING FEATURES FROM ALL 4 INPUT TYPES")
        print("="*80)
        
        # Combine asset metrics from both indices
        all_assets = pd.concat([
            self.asset_metrics_nifty50.assign(index='Nifty50'),
            self.asset_metrics_next50.assign(index='Next50')
        ], ignore_index=True)
        
        print(f"\n✓ Asset-Level Inputs: {len(all_assets)} stocks loaded")
        print(f"  Columns: {list(all_assets.columns)[:5]}... (9 metrics)")
        
        # Initialize feature matrix
        features = []
        
        for idx, asset_row in all_assets.iterrows():
            feature_dict = {}
            
            # ========== ASSET-LEVEL FEATURES (9 metrics) ==========
            # Map actual CSV columns to feature names
            asset_features = {
                'asset_returns_60d_ma': 'returns_60d_ma',      # 1-year annualized return proxy
                'asset_volatility_30d': 'volatility_30d',       # 30-day volatility
                'asset_sharpe_ratio': 'sharpe_ratio',           # Sharpe ratio
                'asset_sortino_ratio': 'sortino_ratio',         # Sortino ratio
                'asset_calmar_ratio': 'calmar_ratio',           # Calmar ratio
                'asset_max_drawdown': 'max_drawdown_90d',       # Max drawdown
                'asset_skewness': 'skewness',                   # Return skewness
                'asset_kurtosis': 'kurtosis',                   # Return kurtosis
                'asset_beta': 'beta',                           # Beta
            }
            
            for feature_name, csv_column in asset_features.items():
                if csv_column in asset_row.index:
                    feature_dict[feature_name] = asset_row[csv_column]
            
            # Add stock symbol
            if 'symbol' in asset_row.index:
                feature_dict['stock_symbol'] = asset_row['symbol']
            else:
                feature_dict['stock_symbol'] = f'STOCK_{idx}'
            
            # ========== MARKET CONTEXT FEATURES (4 components) ==========
            feature_dict['market_vix'] = self.market_context['vix']['current_vix']
            feature_dict['market_volatility_level'] = 1 if self.market_context['vix']['volatility_level'] == 'High' else 0
            feature_dict['market_vix_percentile'] = self.market_context['vix']['vix_percentile']
            
            feature_dict['nifty50_level'] = self.market_context['regime']['nifty50_level']
            feature_dict['market_return_1m'] = self.market_context['regime']['return_1m']
            feature_dict['market_return_3m'] = self.market_context['regime']['return_3m']
            
            # Market regime encoding
            regime = self.market_context['regime']['market_regime']
            feature_dict['market_regime_bull'] = 1 if regime == 'Bull' else 0
            feature_dict['market_regime_bear'] = 1 if regime == 'Bear' else 0
            
            feature_dict['risk_free_rate'] = self.market_context['rfr']['current_rate']
            
            # Sector features
            sector_features = self._extract_sector_features(self.market_context['sectors'])
            for key, value in sector_features.items():
                feature_dict[f'market_{key}'] = value
            
            # ========== PORTFOLIO-LEVEL FEATURES ==========
            feature_dict['portfolio_num_holdings'] = self.portfolio_data['num_holdings']
            feature_dict['portfolio_value'] = self.portfolio_data['portfolio_value']
            feature_dict['portfolio_sector_concentration'] = self.portfolio_data['sector_concentration_top3']
            feature_dict['portfolio_equity_pct'] = self.portfolio_data['asset_class_equity_pct']
            feature_dict['portfolio_commodity_pct'] = self.portfolio_data['asset_class_commodity_pct']
            feature_dict['portfolio_avg_weight'] = self.portfolio_data['avg_weight']
            feature_dict['portfolio_volatility'] = self.portfolio_data['portfolio_volatility']
            feature_dict['portfolio_sharpe'] = self.portfolio_data['portfolio_sharpe']
            feature_dict['portfolio_max_drawdown'] = self.portfolio_data['portfolio_max_drawdown']
            
            # ========== INVESTOR PROFILE FEATURES ==========
            feature_dict['investor_risk_capacity'] = self.investor_profile['risk_capacity_index']
            feature_dict['investor_risk_tolerance'] = self.investor_profile['risk_tolerance_index']
            feature_dict['investor_behavioral_fragility'] = self.investor_profile['behavioral_fragility_index']
            feature_dict['investor_time_horizon_strength'] = self.investor_profile['time_horizon_strength']
            feature_dict['investor_effective_risk_tolerance'] = self.investor_profile['effective_risk_tolerance']
            feature_dict['investor_time_horizon_years'] = self.investor_profile['time_horizon_years']
            
            features.append(feature_dict)
        
        consolidated_df = pd.DataFrame(features)
        
        print(f"\n✓ Market Context Features: 11 features added")
        print(f"✓ Portfolio-Level Features: 9 features added")
        print(f"✓ Investor Profile Features: 6 features added")
        
        print(f"\n✓ Total Features Created: {len(consolidated_df.columns)}")
        print(f"✓ Total Rows (assets): {len(consolidated_df)}")
        print(f"\n📋 Feature Columns ({len(consolidated_df.columns)}):")
        print(f"   {list(consolidated_df.columns)}")
        
        return consolidated_df
    
    def save_consolidated_features(self, output_path: str = None) -> str:
        """
        Consolidate and save features to CSV
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved CSV file
        """
        consolidated_df = self.consolidate_features()
        
        if output_path is None:
            output_path = self.base_path / "consolidated_features.csv"
        
        consolidated_df.to_csv(output_path, index=False)
        print(f"\n✅ Consolidated features saved to: {output_path}")
        print(f"   File size: {consolidated_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return str(output_path)


def main():
    """Run feature consolidation"""
    consolidator = FeatureConsolidator()
    consolidator.save_consolidated_features()
    
    # Display sample of consolidated features
    df = pd.read_csv("f:\\AI Insights Dashboard\\consolidated_features.csv")
    print(f"\n📊 SAMPLE OF CONSOLIDATED FEATURES:\n")
    print(df.head(3))
    print(f"\n{'─'*80}")
    print(f"Data types:\n{df.dtypes.value_counts()}")


if __name__ == "__main__":
    main()
