"""
Feature Extraction Pipeline

Converts raw asset, market, portfolio, and investor inputs into 37-dimensional
feature vectors for ML model inference.

Used by portfolio_optimizer.py to prepare data for ML-based scoring.

Phase 6: Model Deployment & Integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts 37 ML features from portfolio and investor data
    
    Feature Breakdown (37 total):
    - Asset Features (9): returns, volatility, Sharpe, Sortino, Calmar, max_dd, skew, kurt, beta
    - Market Features (11): VIX, regime, returns, rate, sector performance [9]
    - Portfolio Features (9): holdings, value, concentration, cash, Sharpe, vol, max_dd, etc.
    - Investor Features (6): risk_capacity, tolerance, fragility, time_horizon, effective_risk, years
    """
    
    def __init__(self, historical_data: Optional[Dict] = None):
        """
        Initialize feature extractor
        
        Args:
            historical_data: Optional dict with pre-calculated statistics
        """
        self.historical_data = historical_data or {}
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> list:
        """Get all 36 feature names in exact training order"""
        return [
            # Investor Features (6)
            'investor_risk_capacity',
            'investor_risk_tolerance',
            'investor_behavioral_fragility',
            'investor_time_horizon_strength',
            'investor_effective_risk_tolerance',
            'investor_time_horizon_years',
            
            # Asset Features (9)
            'asset_returns_60d_ma',
            'asset_volatility_30d',
            'asset_sharpe_ratio',
            'asset_sortino_ratio',
            'asset_calmar_ratio',
            'asset_max_drawdown',
            'asset_skewness',
            'asset_kurtosis',
            'asset_beta',
            
            # Market Features (10)
            'market_vix',
            'market_volatility_regime',
            'market_returns_1m',
            'market_returns_3m',
            'market_risk_free_rate',
            'market_sector_perf_financials',
            'market_sector_perf_it',
            'market_sector_perf_pharma',
            'market_sector_perf_energy',
            'market_sector_perf_consumer',
            
            # Portfolio Features (9)
            'portfolio_num_holdings',
            'portfolio_value',
            'portfolio_concentration_index',
            'portfolio_cash_pct',
            'portfolio_sharpe_ratio',
            'portfolio_volatility',
            'portfolio_max_drawdown',
            'portfolio_rebalance_frequency',
            'portfolio_time_since_rebalance'
        ]
    
    # ================== ASSET FEATURES (9) ==================
    
    def extract_asset_features(self, asset_data: Dict) -> np.ndarray:
        """
        Extract 9 asset-level features
        
        Args:
            asset_data: Dict with keys:
                - historical_returns: list of daily returns
                - current_price: current price
                - benchmark_returns: benchmark returns (for beta calculation)
        
        Returns:
            [9] array of asset features
        """
        returns = np.array(asset_data.get('historical_returns', []))
        if len(returns) == 0:
            return np.zeros(9)
        
        # 1. Returns 60-day MA
        returns_60d_ma = np.mean(returns[-60:]) if len(returns) >= 60 else np.mean(returns)
        
        # 2. Volatility 30-day
        volatility_30d = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)
        
        # 3. Sharpe Ratio (assuming 0.05 risk-free rate annual)
        rf_daily = 0.05 / 252
        excess_returns = returns - rf_daily
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # 4. Sortino Ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        # 5. Calmar Ratio
        cumulative_returns = np.cumprod(1 + returns) - 1
        max_drawdown = np.min(cumulative_returns)
        total_return = cumulative_returns[-1]
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # 6. Max Drawdown
        max_dd = max_drawdown
        
        # 7. Skewness
        skewness = (np.mean((returns - np.mean(returns))**3)) / (np.std(returns)**3) if np.std(returns) > 0 else 0
        
        # 8. Kurtosis
        kurtosis = (np.mean((returns - np.mean(returns))**4)) / (np.std(returns)**4) - 3 if np.std(returns) > 0 else 0
        
        # 9. Beta (vs benchmark)
        benchmark_returns = np.array(asset_data.get('benchmark_returns', returns))
        if len(benchmark_returns) == len(returns):
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            beta = 1.0
        
        return np.array([
            returns_60d_ma,
            volatility_30d,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_dd,
            skewness,
            kurtosis,
            beta
        ])
    
    # ================== MARKET FEATURES (11) ==================
    
    def extract_market_features(self, market_data: Dict) -> np.ndarray:
        """
        Extract 10 market context features
        
        Args:
            market_data: Dict with keys:
                - vix: Current VIX index value
                - volatility_regime: 'bull' (0) or 'bear' (1)
                - returns_1m: 1-month market returns
                - returns_3m: 3-month market returns
                - risk_free_rate: Current risk-free rate (annual %)
                - sector_performance: Dict of sector returns
        
        Returns:
            [10] array of market features
        """
        vix = float(market_data.get('vix', 15.0))
        
        # Volatility regime (0=bull, 1=bear)
        regime = 0.0 if market_data.get('volatility_regime', 'bull') == 'bull' else 1.0
        
        returns_1m = float(market_data.get('returns_1m', 0.0))
        returns_3m = float(market_data.get('returns_3m', 0.0))
        
        # Risk-free rate (convert to decimal)
        rf_rate = float(market_data.get('risk_free_rate', 5.5)) / 100
        
        # Sector performance (5 sectors - top used in training)
        sector_perf = market_data.get('sector_performance', {})
        sectors = [
            sector_perf.get('financials', 0.0),
            sector_perf.get('it', 0.0),
            sector_perf.get('pharma', 0.0),
            sector_perf.get('energy', 0.0),
            sector_perf.get('consumer', 0.0)
        ]
        
        return np.array([
            vix,
            regime,
            returns_1m,
            returns_3m,
            rf_rate,
        ] + sectors)
    
    # ================== PORTFOLIO FEATURES (9) ==================
    
    def extract_portfolio_features(self, portfolio_data: Dict) -> np.ndarray:
        """
        Extract 9 portfolio-level features
        
        Args:
            portfolio_data: Dict with keys:
                - holdings: List of holdings with weights
                - total_value: Total portfolio value
                - historical_returns: Portfolio returns history
                - last_rebalance: Datetime of last rebalance
        
        Returns:
            [9] array of portfolio features
        """
        holdings = portfolio_data.get('holdings', [])
        num_holdings = float(len(holdings))
        
        portfolio_value = float(portfolio_data.get('total_value', 100000))
        
        # Concentration index (Herfindahl)
        weights = np.array([h.get('weight', 1/max(num_holdings, 1)) for h in holdings])
        concentration_index = float(np.sum(weights**2))
        
        # Cash percentage
        cash_pct = float(portfolio_data.get('cash_pct', 0.05))
        
        # Portfolio returns statistics
        port_returns = np.array(portfolio_data.get('historical_returns', []))
        if len(port_returns) > 0:
            rf_daily = 0.05 / 252
            excess = port_returns - rf_daily
            portfolio_sharpe = np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0
            portfolio_volatility = np.std(port_returns)
            cumulative = np.cumprod(1 + port_returns) - 1
            portfolio_max_dd = np.min(cumulative)
        else:
            portfolio_sharpe = 0.0
            portfolio_volatility = 0.0
            portfolio_max_dd = 0.0
        
        # Rebalance frequency (trades per year)
        rebalance_freq = float(portfolio_data.get('rebalance_frequency', 4))
        
        # Time since last rebalance (days)
        last_rebalance = portfolio_data.get('last_rebalance')
        if last_rebalance:
            days_since = (datetime.now() - last_rebalance).days
            time_since_rebalance = float(min(days_since, 180))  # Cap at 180 days
        else:
            time_since_rebalance = 90.0
        
        return np.array([
            num_holdings,
            portfolio_value,
            concentration_index,
            cash_pct,
            portfolio_sharpe,
            portfolio_volatility,
            portfolio_max_dd,
            rebalance_freq,
            time_since_rebalance
        ])
    
    # ================== INVESTOR FEATURES (6) ==================
    
    def extract_investor_features(self, investor_data: Dict) -> np.ndarray:
        """
        Extract 6 investor profile features
        
        Args:
            investor_data: Dict with keys (all 0-1 scaled):
                - risk_capacity: Financial capacity to take risk
                - risk_tolerance: Psychological comfort with volatility
                - behavioral_fragility: Tendency to panic sell
                - time_horizon_strength: Ability to stay invested
                - effective_risk_tolerance: Combined metric
                - time_horizon_years: Planning period
        
        Returns:
            [6] array of investor features (all normalized to [0,1] or [0, max])
        """
        risk_capacity = float(investor_data.get('risk_capacity', 0.5))
        risk_tolerance = float(investor_data.get('risk_tolerance', 0.5))
        behavioral_fragility = float(investor_data.get('behavioral_fragility', 0.3))
        time_horizon_strength = float(investor_data.get('time_horizon_strength', 0.6))
        effective_risk_tolerance = float(investor_data.get('effective_risk_tolerance', 0.5))
        time_horizon_years = float(investor_data.get('time_horizon_years', 20.0)) / 50.0  # Normalize to [0, 1]
        
        return np.array([
            risk_capacity,
            risk_tolerance,
            behavioral_fragility,
            time_horizon_strength,
            effective_risk_tolerance,
            time_horizon_years
        ])
    
    # ================== CONSOLIDATION ==================
    
    def extract_all_features(
        self,
        asset_data: Dict,
        market_data: Dict,
        portfolio_data: Dict,
        investor_data: Dict
    ) -> np.ndarray:
        """
        Extract all 36 features from complete input data
        
        Args:
            asset_data: Asset-specific data
            market_data: Market context data
            portfolio_data: Portfolio-level data
            investor_data: Investor profile data
        
        Returns:
            [36] feature vector ready for ML model
        """
        investor_features = self.extract_investor_features(investor_data)
        asset_features = self.extract_asset_features(asset_data)
        market_features = self.extract_market_features(market_data)
        portfolio_features = self.extract_portfolio_features(portfolio_data)
        
        # EXACT ORDER MATCHING TRAINING DATA
        all_features = np.concatenate([
            investor_features,     # 6
            asset_features,        # 9
            market_features,       # 10
            portfolio_features     # 9
        ])
        
        return all_features
    
    def get_feature_names(self) -> list:
        """Get all 37 feature names"""
        return self.feature_names
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate feature vector before passing to model
        
        Args:
            features: [37] feature vector
        
        Returns:
            True if valid, False otherwise
        """
        if len(features) != 37:
            logger.warning(f"Expected 37 features, got {len(features)}")
            return False
        
        if np.isnan(features).any():
            logger.warning("Feature vector contains NaN values")
            return False
        
        if np.isinf(features).any():
            logger.warning("Feature vector contains infinite values")
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Sample data
    asset_data = {
        'historical_returns': np.random.normal(0.001, 0.02, 252),
        'benchmark_returns': np.random.normal(0.0005, 0.015, 252)
    }
    
    market_data = {
        'vix': 18.5,
        'volatility_regime': 'bull',
        'returns_1m': 0.03,
        'returns_3m': 0.05,
        'risk_free_rate': 5.5,
        'sector_performance': {
            'financials': 0.02,
            'it': 0.04,
            'pharma': 0.01,
            'energy': -0.01,
            'consumer': 0.02,
            'industrial': 0.03,
            'utilities': 0.005,
            'materials': -0.005,
            'telecom': 0.01
        }
    }
    
    portfolio_data = {
        'holdings': [
            {'symbol': 'INFY', 'weight': 0.25},
            {'symbol': 'TCS', 'weight': 0.25},
            {'symbol': 'HDFC', 'weight': 0.25},
            {'symbol': 'ICICI', 'weight': 0.25}
        ],
        'total_value': 1000000,
        'cash_pct': 0.05,
        'historical_returns': np.random.normal(0.0008, 0.018, 252),
        'rebalance_frequency': 4,
        'last_rebalance': datetime.now() - timedelta(days=90)
    }
    
    investor_data = {
        'risk_capacity': 0.7,
        'risk_tolerance': 0.65,
        'behavioral_fragility': 0.2,
        'time_horizon_strength': 0.8,
        'effective_risk_tolerance': 0.7,
        'time_horizon_years': 20
    }
    
    # Extract features
    features = extractor.extract_all_features(
        asset_data,
        market_data,
        portfolio_data,
        investor_data
    )
    
    print("\n" + "="*60)
    print("EXTRACTED FEATURES")
    print("="*60)
    print(f"Total Features: {len(features)}")
    print(f"Valid: {extractor.validate_features(features)}")
    print(f"Shape: {features.shape}")
    print(f"\nFirst 10 features:")
    for i, (name, value) in enumerate(zip(extractor.get_feature_names()[:10], features[:10])):
        print(f"  {i+1:2}. {name:40} = {value:.6f}")
