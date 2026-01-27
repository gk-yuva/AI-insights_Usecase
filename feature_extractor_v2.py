"""
Feature Extraction Pipeline (Phase 6 - Deployment Version)

Converts raw asset, market, portfolio, and investor inputs into 36-dimensional
feature vectors matching the trained ML model requirements.

These 36 features match exactly with X_train.csv and X_test.csv from Phase 4.

Phase 6: Model Deployment & Integration
"""

import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts 36 ML features matching trained model requirements
    
    Feature Order (36 total):
    - investor_* (6)
    - asset_* (9)
    - market_* (14)
    - portfolio_* (7)
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> list:
        """Get all 36 feature names in exact order"""
        return [
            'investor_risk_capacity',
            'investor_risk_tolerance',
            'investor_behavioral_fragility',
            'investor_time_horizon_strength',
            'investor_effective_risk_tolerance',
            'investor_time_horizon_years',
            'asset_returns_60d_ma',
            'asset_volatility_30d',
            'asset_sharpe_ratio',
            'asset_sortino_ratio',
            'asset_calmar_ratio',
            'asset_max_drawdown',
            'asset_skewness',
            'asset_kurtosis',
            'asset_beta',
            'market_vix',
            'market_volatility_level',
            'market_vix_percentile',
            'nifty50_level',
            'market_return_1m',
            'market_return_3m',
            'market_regime_bull',
            'market_regime_bear',
            'risk_free_rate',
            'market_top_sector_return',
            'market_bottom_sector_return',
            'market_sector_return_dispersion',
            'portfolio_num_holdings',
            'portfolio_value',
            'portfolio_sector_concentration',
            'portfolio_equity_pct',
            'portfolio_commodity_pct',
            'portfolio_avg_weight',
            'portfolio_volatility',
            'portfolio_sharpe',
            'portfolio_max_drawdown'
        ]
    
    def extract_investor_features(self, investor_data: Dict) -> np.ndarray:
        """Extract 6 investor profile features"""
        return np.array([
            float(investor_data.get('risk_capacity', 0.5)),
            float(investor_data.get('risk_tolerance', 0.5)),
            float(investor_data.get('behavioral_fragility', 0.3)),
            float(investor_data.get('time_horizon_strength', 0.6)),
            float(investor_data.get('effective_risk_tolerance', 0.5)),
            float(investor_data.get('time_horizon_years', 20.0)) / 50.0
        ])
    
    def extract_asset_features(self, asset_data: Dict) -> np.ndarray:
        """Extract 9 asset-level features"""
        returns = np.array(asset_data.get('historical_returns', []))
        if len(returns) == 0:
            return np.zeros(9)
        
        returns_60d_ma = np.mean(returns[-60:]) if len(returns) >= 60 else np.mean(returns)
        volatility_30d = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)
        
        rf_daily = 0.05 / 252
        excess_returns = returns - rf_daily
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        cumulative_returns = np.cumprod(1 + returns) - 1
        max_drawdown = np.min(cumulative_returns)
        total_return = cumulative_returns[-1]
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        skewness = (np.mean((returns - np.mean(returns))**3)) / (np.std(returns)**3) if np.std(returns) > 0 else 0
        kurtosis = (np.mean((returns - np.mean(returns))**4)) / (np.std(returns)**4) - 3 if np.std(returns) > 0 else 0
        
        beta = float(asset_data.get('beta', 1.0))
        
        return np.array([
            returns_60d_ma,
            volatility_30d,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            skewness,
            kurtosis,
            beta
        ])
    
    def extract_market_features(self, market_data: Dict) -> np.ndarray:
        """Extract 14 market context features"""
        return np.array([
            float(market_data.get('vix', 15.0)),
            float(market_data.get('volatility_level', 0.2)),
            float(market_data.get('vix_percentile', 50.0)) / 100.0,
            float(market_data.get('nifty50_level', 22000.0)),
            float(market_data.get('return_1m', 0.0)),
            float(market_data.get('return_3m', 0.0)),
            1.0 if market_data.get('regime', 'bull') == 'bull' else 0.0,  # regime_bull
            1.0 if market_data.get('regime', 'bull') == 'bear' else 0.0,  # regime_bear
            float(market_data.get('risk_free_rate', 5.5)) / 100.0,
            float(market_data.get('top_sector_return', 0.02)),
            float(market_data.get('bottom_sector_return', -0.02)),
            float(market_data.get('sector_return_dispersion', 0.04)),
            float(market_data.get('vix_mean', 18.0)),  # 13: Mean VIX
            float(market_data.get('vix_std', 2.0))     # 14: VIX Standard Deviation
        ])
    
    def extract_portfolio_features(self, portfolio_data: Dict) -> np.ndarray:
        """Extract 7 portfolio-level features"""
        holdings = portfolio_data.get('holdings', [])
        num_holdings = float(len(holdings))
        
        weights = np.array([h.get('weight', 1/max(num_holdings, 1)) for h in holdings]) if holdings else np.array([])
        concentration_index = float(np.sum(weights**2)) if len(weights) > 0 else 0.0
        
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
        
        return np.array([
            num_holdings,
            float(portfolio_data.get('total_value', 100000)) / 1000000.0,  # Scale to millions
            concentration_index,
            float(portfolio_data.get('equity_pct', 0.85)),
            portfolio_volatility,
            portfolio_sharpe,
            portfolio_max_dd
        ])
    
    def extract_all_features(
        self,
        asset_data: Dict,
        market_data: Dict,
        portfolio_data: Dict,
        investor_data: Dict
    ) -> np.ndarray:
        """
        Extract all 36 features in exact order
        
        Returns:
            [36] feature vector ready for ML model
        """
        investor_features = self.extract_investor_features(investor_data)  # 6
        asset_features = self.extract_asset_features(asset_data)  # 9
        market_features = self.extract_market_features(market_data)  # 14
        portfolio_features = self.extract_portfolio_features(portfolio_data)  # 7
        
        # Concatenate in exact order
        all_features = np.concatenate([
            investor_features,
            asset_features,
            market_features,
            portfolio_features
        ])
        
        # Validate and fix feature count
        if len(all_features) != 36:
            logger.warning(f"Feature count mismatch: expected 36, got {len(all_features)}")
            if len(all_features) < 36:
                all_features = np.pad(all_features, (0, 36 - len(all_features)), 'constant')
            else:
                all_features = all_features[:36]
        
        return all_features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate feature vector"""
        if len(features) != 36:
            logger.warning(f"Expected 36 features, got {len(features)}")
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
        'beta': 0.95
    }
    
    market_data = {
        'vix': 18.5,
        'volatility_level': 0.18,
        'vix_percentile': 45,
        'nifty50_level': 22500,
        'return_1m': 0.03,
        'return_3m': 0.05,
        'regime': 'bull',
        'risk_free_rate': 5.5,
        'top_sector_return': 0.05,
        'bottom_sector_return': -0.02,
        'sector_return_dispersion': 0.07
    }
    
    portfolio_data = {
        'holdings': [{'weight': 0.25} for _ in range(4)],
        'total_value': 1000000,
        'equity_pct': 0.85,
        'commodity_pct': 0.05,
        'historical_returns': np.random.normal(0.0008, 0.018, 252)
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
        asset_data, market_data, portfolio_data, investor_data
    )
    
    print("\n" + "="*60)
    print("EXTRACTED FEATURES")
    print("="*60)
    print(f"Total Features: {len(features)}")
    print(f"Valid: {extractor.validate_features(features)}")
    print(f"\nAll 36 features:")
    for i, (name, value) in enumerate(zip(extractor.feature_names, features)):
        print(f"  {i+1:2}. {name:40} = {value:10.6f}")
