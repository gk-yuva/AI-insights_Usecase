# Phase 6 Quickstart Guide

## Overview

Phase 6 deploys the trained ML model to production with integrated monitoring and A/B testing framework. This guide shows how to use each component.

---

## Component 1: ML Optimizer Wrapper

**Purpose:** Load model and make predictions

### Basic Usage

```python
from ml_optimizer_wrapper import MLPortfolioOptimizer
from feature_extractor_v2 import FeatureExtractor
import numpy as np

# Initialize
optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
extractor = FeatureExtractor()

# Prepare data
asset_data = {
    'historical_returns': np.random.normal(0.001, 0.02, 252),
    'beta': 0.95
}

market_data = {
    'vix': 18.5,
    'volatility_level': 0.18,
    'regime': 'bull',
    # ... other market metrics
}

portfolio_data = {
    'holdings': [{'weight': 0.25} for _ in range(4)],
    'total_value': 1000000,
    # ... other portfolio metrics
}

investor_data = {
    'risk_capacity': 0.7,
    'risk_tolerance': 0.65,
    # ... other investor metrics
}

# Extract features (36 dimensions)
features = extractor.extract_all_features(
    asset_data, market_data, portfolio_data, investor_data
)

# Get prediction
result = optimizer.predict_recommendation_success(features)
print(f"Add probability: {result['success_probability']:.2%}")
print(f"Recommendation: {result['recommendation']}")
print(f"Score: {result['score']:.0f}/100")
```

### Batch Predictions

```python
# Predict for multiple assets
features_list = [features1, features2, features3]  # List of 36-dim vectors
predictions = optimizer.batch_predict(features_list)

for i, pred in enumerate(predictions):
    print(f"Asset {i}: {pred['recommendation']} ({pred['success_probability']:.2%})")
```

### Model Explainability

```python
# Get feature importance
importance = optimizer.get_feature_importance()
for feature, score in importance[:5]:  # Top 5 features
    print(f"{feature}: {score:.4f}")

# Get prediction explanation
explanation = optimizer.explain_prediction(features)
print(explanation)
```

---

## Component 2: Feature Extraction

**Purpose:** Convert raw data into ML-ready 36-dimensional vectors

### Supported Features

**Investor Features (6):**
- `risk_capacity` - Investor's financial capacity to take risk
- `risk_tolerance` - Psychological willingness to take risk
- `behavioral_fragility` - Tendency to panic sell
- `time_horizon_strength` - Alignment with investment timeline
- `effective_risk_tolerance` - Combined capacity + tolerance
- `time_horizon_years` - Total investment years

**Asset Features (9):**
- `returns_60d_ma` - 60-day moving average of returns
- `volatility_30d` - 30-day rolling volatility
- `sharpe_ratio` - Risk-adjusted return metric
- `sortino_ratio` - Downside risk-adjusted return
- `calmar_ratio` - Return to max drawdown ratio
- `max_drawdown` - Largest peak-to-trough decline
- `skewness` - Distribution asymmetry
- `kurtosis` - Distribution tail weight
- `beta` - Systematic risk vs market

**Market Features (14):**
- `vix` - Volatility Index level
- `volatility_level` - Current market volatility
- `vix_percentile` - VIX position in historical range
- `nifty50_level` - Index level
- `return_1m` - 1-month market return
- `return_3m` - 3-month market return
- `regime_bull` - Indicator for bull market
- `regime_bear` - Indicator for bear market
- `risk_free_rate` - RBI repo rate
- `top_sector_return` - Best performing sector
- `bottom_sector_return` - Worst performing sector
- `sector_return_dispersion` - Spread of sector returns

**Portfolio Features (7):**
- `num_holdings` - Number of securities
- `value` - Total portfolio value
- `sector_concentration` - HHI of sector weights
- `equity_pct` - Equity allocation %
- `commodity_pct` - Commodity allocation %
- `avg_weight` - Average asset weight
- `volatility` - Portfolio volatility

### Usage Example

```python
from feature_extractor_v2 import FeatureExtractor
import numpy as np

extractor = FeatureExtractor()

# Prepare data (must include all required fields)
asset_data = {
    'historical_returns': np.array([...]),  # 252-day returns
    'volatility_30d': 0.18,
    'sharpe_ratio': 1.2,
    'sortino_ratio': 1.8,
    'calmar_ratio': 2.0,
    'max_drawdown': -0.35,
    'skewness': 0.2,
    'kurtosis': 3.5,
    'beta': 0.95
}

market_data = {
    'vix': 18.5,
    'volatility_level': 0.18,
    'vix_percentile': 45,
    'nifty50_level': 22500,
    'return_1m': 0.03,
    'return_3m': 0.05,
    'regime': 'bull',  # or 'bear'
    'risk_free_rate': 5.5,
    'top_sector_return': 0.05,
    'bottom_sector_return': -0.02,
    'sector_return_dispersion': 0.07
}

portfolio_data = {
    'holdings': [
        {'weight': 0.25, 'returns': np.array([...])},
        {'weight': 0.25, 'returns': np.array([...])},
        {'weight': 0.25, 'returns': np.array([...])},
        {'weight': 0.25, 'returns': np.array([...])}
    ],
    'total_value': 1000000,
    'sector_weights': {'Bank': 0.30, 'Tech': 0.25, ...},
    'equity_pct': 0.85,
    'commodity_pct': 0.05,
    'historical_returns': np.array([...])  # Portfolio returns
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
if extractor.validate_features(features):
    print(f"Features ready: {len(features)} dimensions")
else:
    print("Feature validation failed")
```

---

## Component 3: A/B Testing Framework

**Purpose:** Compare ML vs rule-based recommendations during gradual rollout

### Setup

```python
from ab_testing import ABTestingFramework

# Initialize (20% ML, 80% rule-based)
ab_test = ABTestingFramework(
    ml_ratio=0.20,  # Start with 20% ML
    test_id='phase6_production'
)
```

### Log Recommendations

```python
# Get which method to use
method = ab_test.get_method()  # Returns 'ML' or 'RULE_BASED'

if method == 'ML':
    prediction = optimizer.predict_recommendation_success(features)
    score = prediction['success_probability']
    recommendation = prediction['recommendation']
else:
    # Use existing rule-based logic
    score = rule_based_score(asset)
    recommendation = rule_based_recommendation(score)

# Log the recommendation
rec_id = ab_test.log_recommendation(
    asset_symbol='INFY',
    method=method,
    score=score,
    recommendation=recommendation
)
```

### Log Outcomes

```python
# After recommendation plays out (e.g., after 4 weeks)
actual_return = 0.05  # 5% actual return
succeeded = actual_return > 0.02  # Success threshold

ab_test.log_outcome(rec_id, succeeded, actual_return)
```

### Analyze Performance

```python
# Compare methods after collecting outcomes
analysis = ab_test.analyze_performance(min_outcomes=20)

print(f"ML Success Rate: {analysis['ml_performance']['success_rate']:.1%}")
print(f"Rule Success Rate: {analysis['rule_performance']['success_rate']:.1%}")
print(f"ML Avg Return: {analysis['ml_performance']['avg_return']:.2%}")
print(f"Rule Avg Return: {analysis['rule_performance']['avg_return']:.2%}")
```

### Gradual Rollout

```python
# After Week 2, increase to 50% ML
ab_test.update_ml_ratio(0.50)

# After Week 4, increase to 100% ML
ab_test.update_ml_ratio(1.00)
```

---

## Component 4: Production Monitoring

**Purpose:** Track model performance and detect when retraining is needed

### Setup

```python
from model_monitoring import ModelMonitor
import numpy as np

# Initialize with baseline metrics from training
monitor = ModelMonitor(
    baseline_auc=0.9853,
    baseline_f1=0.9375
)

# Set baseline feature distributions from training data
monitor.set_baseline_distributions(
    feature_means=np.array([...]),  # Mean of each 36 features
    feature_stds=np.array([...]),   # Std of each 36 features
    prediction_mean=0.65,            # Mean of prediction probabilities
    prediction_std=0.15              # Std of prediction probabilities
)
```

### Log Predictions

```python
# Log each prediction made in production
monitor.log_prediction(
    features=features,                     # 36-dim feature vector
    prediction_probability=0.72,           # Model output [0,1]
    confidence=0.72,                       # Confidence score
    latency_ms=45,                         # Prediction time
    actual_label=1                         # Actual outcome (0 or 1)
)
```

### Check for Issues

```python
# 1. Data drift - are inputs changing?
drift_status = monitor.check_data_drift()
if drift_status['drifted']:
    print(f"Alert: Data drift detected in {drift_status['drifted_features']} features")

# 2. Performance degradation - is model accuracy declining?
perf_status = monitor.check_performance_degradation()
if 'recent_auc' in perf_status:
    print(f"Recent AUC: {perf_status['recent_auc']:.4f} (baseline: 0.9853)")

# 3. Prediction drift - are outputs changing?
pred_drift = monitor.check_prediction_drift()
if pred_drift['drifted']:
    print(f"Alert: Prediction distribution shifted")

# 4. Latency - is inference slowing down?
latency = monitor.check_inference_latency()
print(f"Mean latency: {latency['mean_latency_ms']:.1f}ms (target: <200ms)")

# 5. Should retrain?
if monitor.should_trigger_retraining():
    print("ALERT: Start retraining pipeline")
```

### Example Monitoring Loop

```python
import time
from datetime import datetime

while True:
    # Get today's predictions
    for rec_id, features, prediction, actual_label in get_todays_predictions():
        monitor.log_prediction(
            features=features,
            prediction_probability=prediction,
            confidence=prediction,
            latency_ms=50,
            actual_label=actual_label
        )
    
    # Run checks every hour
    if datetime.now().minute == 0:
        checks = {
            'data_drift': monitor.check_data_drift(),
            'performance': monitor.check_performance_degradation(),
            'latency': monitor.check_inference_latency(),
        }
        
        # Alert on issues
        for check_name, result in checks.items():
            if result.get('alert', False):
                send_alert(f"Model monitoring: {check_name} issue detected")
        
        # Trigger retraining if needed
        if monitor.should_trigger_retraining():
            start_retraining_pipeline()
    
    time.sleep(300)  # Check every 5 minutes
```

---

## Integration Timeline

### Week 1: 20% ML Rollout
```
- 80% recommendations use rule-based logic
- 20% use ML model
- Monitor for issues, collect A/B test data
```

### Week 2-3: 50% ML Rollout
```
- 50% ML, 50% rule-based
- Compare performance metrics
- Ensure ML is performing better
```

### Week 4: 100% ML Rollout
```
- Full production deployment
- Continue monitoring
- Setup automated retraining
```

---

## Troubleshooting

### Issue: Feature dimension mismatch
**Solution:** Ensure all 36 features are extracted in correct order
```python
assert len(features) == 36
```

### Issue: Model not loading
**Solution:** Verify model file exists and is valid JSON
```python
import os
assert os.path.exists('trained_xgboost_model.json')
```

### Issue: Predictions all ADD/REMOVE
**Solution:** Check feature distributions match training data
```python
# Compare test features vs training distribution
print(f"Feature mean: {np.mean(features):.3f}")
print(f"Expected: ~0.5")
```

### Issue: Monitoring shows constant drift alerts
**Solution:** Adjust drift threshold or retrain baseline
```python
# More lenient threshold (p-value < 0.01 instead of < 0.05)
result = monitor.check_data_drift()
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Model ROC-AUC** | >0.95 | 0.9853 ✓ |
| **Prediction Latency** | <200ms | ~50ms ✓ |
| **Feature Extraction** | <100ms | Included in latency |
| **A/B Test Min Size** | >20 outcomes | Variable |
| **Data Drift Alert** | p-value < 0.05 | Configurable |
| **Performance Degrade** | <95% baseline | 95% threshold |

---

## Quick Reference

```python
# Complete end-to-end example
from ml_optimizer_wrapper import MLPortfolioOptimizer
from feature_extractor_v2 import FeatureExtractor
from ab_testing import ABTestingFramework
from model_monitoring import ModelMonitor

# 1. Initialize components
optimizer = MLPortfolioOptimizer('trained_xgboost_model.json')
extractor = FeatureExtractor()
ab_test = ABTestingFramework(ml_ratio=0.20)
monitor = ModelMonitor(baseline_auc=0.9853, baseline_f1=0.9375)

# 2. Get recommendation
method = ab_test.get_method()
features = extractor.extract_all_features(asset_data, market_data, 
                                          portfolio_data, investor_data)
prediction = optimizer.predict_recommendation_success(features)

# 3. Log recommendation
rec_id = ab_test.log_recommendation(
    asset_symbol='INFY',
    method=method,
    score=prediction['success_probability'],
    recommendation=prediction['recommendation']
)

# 4. Monitor
monitor.log_prediction(features, prediction['success_probability'], 
                      latency_ms=45, actual_label=1)

# 5. Check health
if monitor.should_trigger_retraining():
    print("Start retraining")
```

---

**Document Version:** 1.0  
**Created:** January 2026  
**Last Updated:** Phase 6 Deployment
