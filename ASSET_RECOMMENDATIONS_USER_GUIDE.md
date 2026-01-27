# Asset Recommendations Dashboard - User Guide

## Overview

The Asset Recommendations dashboard now features **two recommendation tabs**:

1. **📊 Rule-Based Recommendations** - Traditional heuristic-based recommendations
2. **🤖 ML-Based Recommendations** - AI-powered recommendations using Phase 6 ML model

---

## How to Use

### Step 1: Load Your Portfolio

1. Open the Asset Recommendations dashboard
2. In the **sidebar**, upload your portfolio Excel file
3. The dashboard displays: "✅ Portfolio loaded: X holdings"

### Step 2: Run Analysis

1. Click the blue **"🔍 Analyze & Recommend"** button in the sidebar
2. Wait for the analysis to complete (shows progress: 20% → 40% → 60% → 80% → 100%)
3. After completion, **two tabs will appear**

### Step 3: View Results

#### Tab 1: 📊 Rule-Based Recommendations

**Shows:**
- **Metrics**: Current holdings, assets to add, assets to drop
- **Portfolio Comparison**: Chart showing portfolio performance with recommended changes
- **Assets to Drop**: Table of underperforming holdings
- **New Portfolio**: Proposed composition with new assets highlighted
- **Implementation Roadmap**: Phased approach over weeks
- **Expected Impact**: Sharpe ratio improvement, volatility reduction, return changes

**Best for:** Traditional analysis, conservative approach, familiar metrics

#### Tab 2: 🤖 ML-Based Recommendations

**Shows:**
- **ML Model Analysis**: AI-powered evaluation of all Nifty50 stocks
- **Metrics**: Total analyzed, recommended for ADD, flagged for review
- **Top Recommendations**: 5 highest-scoring assets to add
- **Assets to Review**: 5 lowest-scoring assets to consider removing
- **Model Details**: 
  - Model Type: XGBoost
  - ROC-AUC: 0.9853 (excellent accuracy)
  - F1-Score: 0.9375
  - Features Used: 36 dimensions
  - Inference Latency: ~50ms

**Best for:** Data-driven approach, ML confidence scores, deeper analysis

---

## Understanding the Recommendations

### Rule-Based Tab

**Scoring Logic:**
- Return alignment (25 points)
- Volatility threshold (20 points)
- Sharpe ratio (20 points)
- Maximum drawdown (15 points)
- Diversification (20 points)

**Recommendation Types:**
- ✓ **Hold** - Asset performing adequately
- ➕ **Add** - High-quality asset to add
- ❌ **Remove** - Underperforming asset to drop

---

### ML-Based Tab

**ML Scoring (0-100):**
- **70-100**: Strong recommendation (✓ ADD)
- **40-70**: Neutral/Hold
- **0-40**: Review/Consider removing (❌ REVIEW)

**ML Confidence (0-1):**
- Probability that adding this asset will improve portfolio
- Higher = more confident recommendation
- Based on historical pattern matching

**Features Used:**
- **Investor Profile** (6): Risk capacity, tolerance, time horizon
- **Asset Metrics** (9): Returns, volatility, Sharpe ratio, drawdown, skewness
- **Market Data** (14): VIX, regime, sector returns, risk-free rate
- **Portfolio** (7): Concentration, weights, diversification, volatility

---

## Comparing the Two Approaches

| Aspect | Rule-Based | ML-Based |
|--------|-----------|----------|
| **Method** | Heuristic rules | XGBoost model |
| **Training** | Expert-defined thresholds | 63 historical portfolios |
| **Accuracy** | Moderate | 98.53% ROC-AUC |
| **Transparency** | Easy to understand | Feature importance available |
| **Adaptability** | Fixed rules | Learns from data |
| **Speed** | Fast | ~50ms per asset |
| **Data Used** | Current metrics | Historical + current |

---

## Decision Making

### Use Rule-Based When:
- You prefer transparent, rule-based decisions
- You want to understand exactly why an asset was recommended
- You need quick recommendations
- You prefer conservative approach

### Use ML-Based When:
- You want AI-driven insights
- You trust machine learning models
- You want confidence-weighted decisions
- You need data-driven recommendations

### Best Practice:
✓ **Compare both tabs** and look for consensus  
✓ If both recommend the same asset → High confidence  
✓ If they disagree → Do additional research  
✓ Always validate with your financial advisor

---

## Example Scenarios

### Scenario 1: Both Tabs Agree
```
Rule-Based: ADD INFY (Score: 85/100)
ML-Based:   ADD INFY (Score: 82/100, Confidence: 92%)

Decision: STRONG BUY SIGNAL - High agreement between methods
```

### Scenario 2: Strong Disagreement
```
Rule-Based: ADD RELIANCE (Score: 75/100)
ML-Based:   REMOVE RELIANCE (Score: 35/100, Confidence: 87%)

Decision: Do additional research - methods disagree
```

### Scenario 3: ML Confident, Rule-Based Neutral
```
Rule-Based: HOLD TCS (Score: 60/100)
ML-Based:   ADD TCS (Score: 89/100, Confidence: 95%)

Decision: ML sees strong signals Rule-based method misses
```

---

## Tips & Best Practices

### ✓ DO:
- Compare recommendations from both tabs
- Review the rationale for each recommendation
- Check the implementation roadmap before making changes
- Monitor your portfolio for 4 weeks after changes
- Start with smaller allocations to recommended assets
- Use the confidence scores as a guide

### ✗ DON'T:
- Follow recommendations blindly without research
- Make drastic portfolio changes all at once
- Ignore your financial advisor's input
- Use only one approach (diversify decision methods)
- Expect 100% accuracy (both methods are probabilistic)

---

## Understanding the Metrics

### Portfolio Metrics

**Sharpe Ratio Improvement:**
- Higher is better (more risk-adjusted return)
- Example: +0.25 means 0.25 additional return per unit of risk
- Typical target: >1.0 for good portfolios

**Volatility Reduction:**
- Lower is better (less risk)
- Example: -2.5% means 2.5% less daily fluctuation
- Typical target: Stay consistent with risk profile

**Raw Return Change:**
- Expected absolute return change
- Example: +3.2% means 3.2% higher annual returns
- Note: Higher returns usually come with higher risk

---

## Troubleshooting

### Issue: "No investor profile found"
**Solution:** 
1. Go to Main Dashboard (http://localhost:8501)
2. Fill in Investor Profile form
3. Click "💾 Save & Continue"
4. Refresh this page

### Issue: "Error during analysis"
**Solution:**
1. Check internet connection (for market data)
2. Verify portfolio file format (Excel .xlsx or .xls)
3. Ensure portfolio has 'Instrument' and 'Cur. val' columns
4. Try uploading file again

### Issue: "ML modules not available"
**Solution:**
1. Ensure Phase 6 components are installed
2. Check that these files exist:
   - ml_optimizer_wrapper.py
   - feature_extractor_v2.py
   - ab_testing.py
   - trained_xgboost_model.json
3. Restart dashboard

---

## Performance Notes

- **Analysis Time**: Typically 30-60 seconds
- **ML Processing**: ~50ms per asset (on 50 assets ≈ 2.5 seconds)
- **Data Freshness**: Updated daily (overnight)
- **Accuracy**: Model validated on 78.9% accuracy

---

## Next Steps

After reviewing recommendations:

1. **Research** - Look deeper into recommended assets
2. **Validate** - Check with your financial advisor
3. **Plan** - Use the Implementation Roadmap provided
4. **Execute** - Make changes gradually over phases
5. **Monitor** - Track results for 4 weeks
6. **Review** - Come back and analyze again after 1 month

---

## Contact & Support

For questions or issues:

1. Check [PHASE_6_QUICKSTART.md](PHASE_6_QUICKSTART.md) for technical details
2. Review [ML_UseCase.md](ML_UseCase.md) for model information
3. Refer to [ASSET_RECOMMENDATIONS_TAB_ENHANCEMENT.md](ASSET_RECOMMENDATIONS_TAB_ENHANCEMENT.md) for changes

---

**Version:** 2.0 (Tab-based interface)  
**Last Updated:** January 2026  
**Model Version:** Phase 6 - XGBoost (ROC-AUC 0.9853)
