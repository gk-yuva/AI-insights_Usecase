"""
Phase 6 Integration Tests

Validates that ML model integrates correctly with portfolio_optimizer.py
and produces valid recommendations.

Tests:
- ML wrapper loads model correctly
- Feature extraction produces valid vectors
- Model predictions are in valid range [0,1]
- A/B testing framework logs data correctly
- Monitoring detects issues appropriately
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_optimizer_wrapper import MLPortfolioOptimizer
from feature_extractor_v2 import FeatureExtractor
from ab_testing import ABTestingFramework, RecommendationMethod
from model_monitoring import ModelMonitor


def test_ml_wrapper_loading():
    """Test 1: ML wrapper loads model correctly"""
    print("\n" + "="*70)
    print("TEST 1: ML Wrapper Model Loading")
    print("="*70)
    
    try:
        optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
        print("[PASS] Model loaded successfully")
        print(f"   Type: {type(optimizer.model).__name__}")
        print(f"   Features: {len(optimizer.feature_names)}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return False


def test_feature_extraction():
    """Test 2: Feature extraction produces valid 36-dim vectors"""
    print("\n" + "="*70)
    print("TEST 2: Feature Extraction")
    print("="*70)
    
    try:
        extractor = FeatureExtractor()
        
        # Create sample data
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
        
        # Validate
        assert len(features) == 36, f"Expected 36 features, got {len(features)}"
        assert not np.isnan(features).any(), "Features contain NaN"
        assert not np.isinf(features).any(), "Features contain infinity"
        assert extractor.validate_features(features), "Feature validation failed"
        
        print("[PASS] Feature extraction successful")
        print(f"   Features: {len(features)} dimensions")
        print(f"   Sample values: {features[:5]}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Feature extraction failed: {e}")
        return False


def test_model_prediction():
    """Test 3: Model produces valid predictions"""
    print("\n" + "="*70)
    print("TEST 3: Model Predictions")
    print("="*70)
    
    try:
        optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
        extractor = FeatureExtractor()
        
        # Create realistic sample data
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
        
        # Get prediction
        result = optimizer.predict_recommendation_success(features)
        
        # Validate
        assert 0 <= result['success_probability'] <= 1, f"Invalid probability: {result['success_probability']}"
        assert result['recommendation'] in ['ADD', 'REMOVE', 'HOLD'], f"Invalid recommendation: {result['recommendation']}"
        assert 0 <= result['score'] <= 100, f"Invalid score: {result['score']}"
        
        print("[PASS] Model prediction successful")
        print(f"   Probability: {result['success_probability']:.4f}")
        print(f"   Recommendation: {result['recommendation']}")
        print(f"   Score: {result['score']:.1f}/100")
        return True
        
    except Exception as e:
        print(f"[FAIL] Model prediction failed: {e}")
        return False


def test_ab_testing_framework():
    """Test 4: A/B testing framework logs and analyzes correctly"""
    print("\n" + "="*70)
    print("TEST 4: A/B Testing Framework")
    print("="*70)
    
    try:
        ab_test = ABTestingFramework(ml_ratio=0.5, test_id='phase6_integration_test')
        
        # Simulate recommendations
        for i in range(50):
            method = ab_test.get_method()
            score = np.random.random()
            
            rec_id = ab_test.log_recommendation(
                asset_symbol=f"ASSET_{i}",
                method=method,
                score=score,
                recommendation='BUY' if score > 0.5 else 'SELL'
            )
            
            # Log outcome
            succeeded = np.random.random() < 0.65
            ab_test.log_outcome(rec_id, succeeded, np.random.normal(0.1, 0.05))
        
        # Analyze
        analysis = ab_test.analyze_performance(min_outcomes=20)
        
        assert 'ml_performance' in analysis, "Missing ML performance data"
        assert 'rule_performance' in analysis, "Missing rule-based performance data"
        
        print("[PASS] A/B testing framework successful")
        print(f"   ML outcomes: {analysis['ml_outcomes']}")
        print(f"   Rule outcomes: {analysis['rule_outcomes']}")
        if 'ml_performance' in analysis:
            print(f"   ML success: {analysis['ml_performance']['success_rate']:.1%}")
        return True
        
    except Exception as e:
        print(f"[FAIL] A/B testing framework failed: {e}")
        return False


def test_monitoring():
    """Test 5: Model monitoring detects issues appropriately"""
    print("\n" + "="*70)
    print("TEST 5: Model Monitoring")
    print("="*70)
    
    try:
        monitor = ModelMonitor(baseline_auc=0.9853, baseline_f1=0.9375)
        
        # Set baseline distributions
        monitor.set_baseline_distributions(
            feature_means=np.ones(36) * 0.5,
            feature_stds=np.ones(36) * 0.2,
            prediction_mean=0.65,
            prediction_std=0.15
        )
        
        # Log predictions
        for i in range(100):
            features = np.random.normal(0.5, 0.2, 36)
            pred = np.random.beta(6, 3)
            
            monitor.log_prediction(
                features=features,
                prediction_probability=pred,
                confidence=pred,
                latency_ms=np.random.normal(50, 10),
                actual_label=1 if pred > 0.5 else 0
            )
        
        # Check metrics
        data_drift = monitor.check_data_drift()
        perf = monitor.check_performance_degradation()
        latency = monitor.check_inference_latency()
        
        assert 'drifted' in data_drift, "Data drift check missing"
        if 'recent_auc' in perf:
            assert 0 <= perf['recent_auc'] <= 1, f"Invalid AUC: {perf['recent_auc']}"
        assert 'mean_latency_ms' in latency, "Latency check missing"
        
        print("[PASS] Monitoring successful")
        print(f"   Data drift: {data_drift.get('drifted', 'unknown')}")
        if 'recent_auc' in perf:
            print(f"   Recent AUC: {perf['recent_auc']:.4f}")
        print(f"   Mean latency: {latency.get('mean_latency_ms', 0):.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Monitoring failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("PHASE 6 INTEGRATION TESTS")
    print("="*70)
    
    results = {
        'ML Wrapper Loading': test_ml_wrapper_loading(),
        'Feature Extraction': test_feature_extraction(),
        'Model Prediction': test_model_prediction(),
        'A/B Testing': test_ab_testing_framework(),
        'Monitoring': test_monitoring(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "[PASS]" if passed_flag else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL INTEGRATION TESTS PASSED - PHASE 6 READY FOR DEPLOYMENT")
        return 0
    else:
        print(f"\n[WARN] {total - passed} test(s) failed - review errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
