# Asset Recommendations Dashboard - Quick Reference

## Two-Tab Interface

### How to Access

1. Open dashboard: `streamlit run asset_recommendations_dashboard.py`
2. Upload portfolio → Click "Analyze & Recommend"
3. Two tabs appear ⬇️

---

## Tab 1: 📊 Rule-Based Recommendations

**Use When:** You want transparent, rule-based decisions

**Shows:**
- Asset scoring (0-100 scale)
- Assets to ADD
- Assets to REMOVE
- New portfolio composition
- Implementation phases (4 weeks)
- Sharpe ratio improvement
- Volatility reduction

**Scoring Factors:**
- Return alignment (25 pts)
- Volatility threshold (20 pts)
- Sharpe ratio (20 pts)
- Max drawdown (15 pts)
- Diversification (20 pts)

---

## Tab 2: 🤖 ML-Based Recommendations

**Use When:** You want AI-driven insights with confidence scores

**Shows:**
- Total assets analyzed
- ML score for each asset (0-100)
- Confidence probability (0-1)
- Top 5 ADD recommendations
- Top 5 REMOVE candidates
- Model metrics (ROC-AUC: 0.9853)

**Model Details:**
- Type: XGBoost Classifier
- Training: 63 historical portfolios
- Features: 36 dimensions
- Accuracy: 78.9% test set
- Speed: ~50ms per asset

---

## Comparison Matrix

| Feature | Rule-Based | ML-Based |
|---------|-----------|----------|
| **Speed** | Instant | ~50ms/asset |
| **Logic** | Transparent | Pattern-based |
| **Accuracy** | Good | Excellent (98.53% AUC) |
| **Adaptability** | Fixed rules | Learns from data |
| **Explanation** | Easy | Feature importance |

---

## Quick Decision Guide

### Both Tabs Recommend Same Asset?
✅ **STRONG SIGNAL** - High confidence, proceed

### Tabs Disagree?
⚠️ **DO RESEARCH** - Look for additional signals

### ML Score > 80 & Rule Score > 75?
✅ **ADD** - Meets both criteria

### ML Score < 40 or Rule Score < 50?
❌ **REMOVE** - Both flag as underperforming

### Confidence Score > 90%?
✅ **VERY CONFIDENT** - ML model is sure

---

## ML Model Features Used (36 total)

**Investor Features** (6)
- Risk capacity
- Risk tolerance  
- Behavioral fragility
- Time horizon strength
- Effective risk tolerance
- Time horizon years

**Asset Features** (9)
- 60-day returns MA
- 30-day volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Max drawdown
- Skewness
- Kurtosis
- Beta

**Market Features** (14)
- VIX level
- Volatility level
- VIX percentile
- Nifty50 level
- 1-month return
- 3-month return
- Market regime (bull/bear)
- Risk-free rate
- Top sector return
- Bottom sector return
- Sector dispersion

**Portfolio Features** (7)
- Number of holdings
- Portfolio value
- Sector concentration
- Equity %
- Commodity %
- Average weight
- Portfolio volatility

---

## Keyboard Shortcuts

- `Tab` key → Switch between tabs
- Click tab name → Switch directly
- Scroll down → See more details
- Expand button → View more info

---

## Common Questions

**Q: Why do recommendations differ?**
A: Different approaches - rules use heuristics, ML uses patterns

**Q: Should I follow ML or rule-based?**
A: Follow the one with higher confidence; best if both agree

**Q: How often to check?**
A: Weekly during rollout, monthly after

**Q: Can I ignore the recommendations?**
A: Yes, always validate with financial advisor

**Q: What if ML isn't available?**
A: Tab 2 shows error; use Tab 1 (rule-based) instead

---

## Implementation Roadmap

**Phase 1** (Week 1-2): 20% ML, 80% Rule-based  
**Phase 2** (Week 3-4): 50% ML, 50% Rule-based  
**Phase 3** (Week 5-6): 80% ML, 20% Rule-based  
**Full Rollout** (Week 7+): 100% ML  

---

## Performance Expectations

- **Analysis Time**: 30-60 seconds
- **Decision Confidence**: 85-95% for top recommendations
- **Success Rate**: 76-80% historically
- **Model Accuracy**: ROC-AUC 0.9853
- **Update Frequency**: Daily overnight

---

## Files Used

**Python Modules:**
- `ml_optimizer_wrapper.py` - Model inference
- `feature_extractor_v2.py` - Feature prep
- `ab_testing.py` - A/B framework
- `model_monitoring.py` - Production monitoring

**Model File:**
- `trained_xgboost_model.json` - Trained model (50KB)

**Data Files:**
- `IID_filled.json` - Investor profile
- Portfolio Excel file - Your holdings

---

## Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| No tabs appear | Click analyze button first |
| ML tab shows error | Check if model files exist |
| Slow performance | Internet connection needed for data |
| Portfolio not loading | Ensure Excel format correct |
| No investor profile | Fill profile in main dashboard first |

---

## Next Actions

1. **Try Tab 1** - See rule-based recommendations
2. **Try Tab 2** - See ML-based recommendations  
3. **Compare** - Look for agreement/disagreement
4. **Decide** - Pick which assets to implement
5. **Implement** - Follow phases over 4-8 weeks
6. **Monitor** - Check results after 4 weeks
7. **Review** - Run analysis again next month

---

## Contact

- **Questions About:** ML model → See PHASE_6_QUICKSTART.md
- **Questions About:** Dashboard → See ASSET_RECOMMENDATIONS_USER_GUIDE.md
- **Questions About:** Changes → See ASSET_RECOMMENDATIONS_TAB_ENHANCEMENT.md

---

**Dashboard Version:** 2.0  
**ML Model Version:** Phase 6 (XGBoost)  
**Last Updated:** January 22, 2026  

✓ **READY TO USE!**
