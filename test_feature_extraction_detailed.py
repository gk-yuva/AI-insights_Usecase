"""
Detailed debug script to trace feature extraction errors
"""
import numpy as np
import pandas as pd
from datetime import datetime
from feature_extractor_v2 import FeatureExtractor

def test_feature_extraction():
    """Test feature extraction in detail"""
    
    print("=" * 70)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    print(f"\nFeature names ({len(extractor.feature_names)}):")
    for i, name in enumerate(extractor.feature_names):
        print(f"  {i+1}: {name}")
    
    # Create test data
    np.random.seed(42)
    days = 252
    returns = np.random.normal(0.0005, 0.02, days)
    
    asset_data = {
        'historical_returns': returns,
        'beta': 1.1,
        'volatility': np.std(returns),
        'sharpe_ratio': 1.5
    }
    
    market_data = {
        'vix': 18.5,
        'volatility_level': 0.15,
        'vix_percentile': 50,
        'nifty50_level': 22500,
        'return_1m': 0.02,
        'return_3m': 0.05,
        'regime': 'bull',
        'risk_free_rate': 5.5,
        'top_sector_return': 0.05,
        'bottom_sector_return': -0.02,
        'sector_return_dispersion': 0.07,
        'vix_mean': 18.5,
        'vix_std': 2.0
    }
    
    portfolio_data = {
        'holdings': [{'weight': 0.02}] * 50,
        'total_value': 1000000,
        'equity_pct': 0.85,
        'commodity_pct': 0.05,
        'bond_pct': 0.10,
        'cash_pct': 0.00,
        'historical_returns': returns[-252:],
        'concentration': 0.3,
        'diversification': 0.7
    }
    
    investor_data = {
        'risk_capacity': 0.7,
        'risk_tolerance': 0.65,
        'behavioral_fragility': 0.2,
        'time_horizon_strength': 0.8,
        'effective_risk_tolerance': 0.7,
        'time_horizon_years': 20,
        'age': 45,
        'income_stability': 0.8
    }
    
    print("\n" + "=" * 70)
    print("EXTRACTING INDIVIDUAL FEATURE GROUPS")
    print("=" * 70)
    
    # Test each extraction method
    print("\n1. Investor Features:")
    try:
        investor_features = extractor.extract_investor_features(investor_data)
        print(f"   ✓ Count: {len(investor_features)} (expected 6)")
        print(f"   Values: {investor_features}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    print("\n2. Asset Features:")
    try:
        asset_features = extractor.extract_asset_features(asset_data)
        print(f"   ✓ Count: {len(asset_features)} (expected 9)")
        print(f"   Values: {asset_features}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    print("\n3. Market Features:")
    try:
        market_features = extractor.extract_market_features(market_data)
        print(f"   ✓ Count: {len(market_features)} (expected 14)")
        print(f"   Values: {market_features}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    print("\n4. Portfolio Features:")
    try:
        portfolio_features = extractor.extract_portfolio_features(portfolio_data)
        print(f"   ✓ Count: {len(portfolio_features)} (expected 7)")
        print(f"   Values: {portfolio_features}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test concatenation
    print("\n" + "=" * 70)
    print("CONCATENATING ALL FEATURES")
    print("=" * 70)
    
    total = len(investor_features) + len(asset_features) + len(market_features) + len(portfolio_features)
    print(f"\nTotal features: {total} (expected 36)")
    print(f"  Investor: {len(investor_features)}")
    print(f"  Asset: {len(asset_features)}")
    print(f"  Market: {len(market_features)}")
    print(f"  Portfolio: {len(portfolio_features)}")
    
    # Test full extraction
    print("\n" + "=" * 70)
    print("TESTING FULL EXTRACTION")
    print("=" * 70)
    
    try:
        all_features = extractor.extract_all_features(
            asset_data, market_data, portfolio_data, investor_data
        )
        print(f"✓ SUCCESS!")
        print(f"  Total features: {len(all_features)}")
        print(f"  Features: {all_features}")
        return True
    except Exception as e:
        print(f"✗ FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_extraction()
    exit(0 if success else 1)
