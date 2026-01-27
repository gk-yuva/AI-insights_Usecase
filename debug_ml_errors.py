"""
Debug script to identify errors in ML model analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime
from ml_optimizer_wrapper import MLPortfolioOptimizer
from feature_extractor_v2 import FeatureExtractor
from portfolio_optimizer import PortfolioOptimizer

def debug_ml_analysis():
    """Debug the ML analysis to identify errors"""
    
    print("=" * 70)
    print("DEBUGGING ML MODEL ANALYSIS")
    print("=" * 70)
    
    try:
        # Initialize components
        print("\n1. Initializing ML components...")
        ml_optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
        print("   ✓ ML Optimizer loaded")
        
        extractor = FeatureExtractor()
        print("   ✓ Feature Extractor loaded")
        
        nifty50 = PortfolioOptimizer.NIFTY50
        print(f"   ✓ Nifty50 list: {len(nifty50)} symbols")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return
    
    # Test with first 5 symbols
    print(f"\n2. Testing feature extraction for first 5 symbols...")
    print("-" * 70)
    
    errors = []
    successes = 0
    
    for idx, symbol in enumerate(nifty50[:5]):
        try:
            print(f"\n   Testing {symbol}...")
            
            # Generate synthetic price data
            np.random.seed(hash(symbol) % 2**32)
            days = 252
            returns = np.random.normal(0.0005, 0.02, days)
            price_data = pd.Series(
                100 * np.exp(np.cumsum(returns)),
                index=pd.date_range(end=datetime.now(), periods=days, freq='D')
            )
            
            returns_series = price_data.pct_change().dropna()
            print(f"      - Returns shape: {returns_series.shape}")
            
            # Create asset data
            asset_data = {
                'historical_returns': returns_series.values,
                'beta': 1.0 + np.random.normal(0, 0.3),
                'volatility': returns_series.std(),
                'sharpe_ratio': returns_series.mean() / (returns_series.std() + 0.001) * np.sqrt(252)
            }
            print(f"      - Asset data created")
            
            # Create market data
            market_data = {
                'vix': 18.5 + np.random.normal(0, 2),
                'volatility_level': 0.15,
                'vix_percentile': 50,
                'nifty50_level': 22500,
                'return_1m': np.random.uniform(-0.05, 0.05),
                'return_3m': np.random.uniform(-0.1, 0.1),
                'regime': 'bull' if np.random.random() > 0.3 else 'bear',
                'risk_free_rate': 5.5,
                'top_sector_return': np.random.uniform(0.02, 0.08),
                'bottom_sector_return': np.random.uniform(-0.05, 0.02),
                'sector_return_dispersion': np.random.uniform(0.05, 0.1)
            }
            print(f"      - Market data created")
            
            # Create portfolio data
            portfolio_data = {
                'holdings': [{'weight': 0.02}] * 50,
                'total_value': 1000000,
                'equity_pct': 0.85,
                'commodity_pct': 0.05,
                'bond_pct': 0.10,
                'cash_pct': 0.00,
                'historical_returns': returns_series.values[-252:] if len(returns_series) >= 252 else returns_series.values,
                'concentration': np.random.uniform(0.1, 0.5),
                'diversification': np.random.uniform(0.5, 0.9)
            }
            print(f"      - Portfolio data created")
            
            # Create investor data
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
            print(f"      - Investor data created")
            
            # Extract features
            print(f"      - Extracting features...")
            features = extractor.extract_all_features(
                asset_data, market_data, portfolio_data, investor_data
            )
            
            if features is None:
                print(f"      ✗ Features is None!")
                errors.append(f"{symbol}: Features returned None")
                continue
            
            print(f"      - Features shape: {len(features)}")
            
            if len(features) != 36:
                print(f"      ! WARNING: Expected 36 features, got {len(features)}")
                # Pad if needed
                if len(features) < 36:
                    features = np.pad(features, (0, 36 - len(features)), 'constant')
                else:
                    features = features[:36]
                print(f"      - Adjusted to 36 features")
            
            # Get prediction
            print(f"      - Getting ML prediction...")
            prediction = ml_optimizer.predict_recommendation_success(features)
            
            print(f"      ✓ SUCCESS!")
            print(f"         Probability: {prediction.get('success_probability', 0):.4f}")
            print(f"         Recommendation: {prediction.get('recommendation', 'N/A')}")
            print(f"         Score: {prediction.get('score', 0):.2f}")
            
            successes += 1
            
        except Exception as e:
            error_msg = f"{symbol}: {str(e)[:80]}"
            errors.append(error_msg)
            print(f"      ✗ ERROR: {error_msg}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {successes} successes, {len(errors)} errors")
    print("=" * 70)
    
    if errors:
        print("\nERROR DETAILS:")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    debug_ml_analysis()
