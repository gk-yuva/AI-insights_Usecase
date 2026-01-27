# 📋 Investor Information Document (IID) Workflow

## Overview
The portfolio dashboard system uses your **Investor Information Document (IID)** - a comprehensive profile of your investment goals, risk tolerance, and financial situation. This profile is essential for both dashboards.

---

## ✅ Correct Workflow

### Step 1: Start Main Dashboard (Port 8501)
```bash
cd f:\AI Insights Dashboard
python -m streamlit run dashboard.py --server.port 8501
```

### Step 2: Fill Investor Profile Form
1. Open browser: **http://localhost:8501**
2. You'll see the **"📝 Investor Information"** form
3. Fill in all sections:
   - **Personal Profile**: Age, Employment, City, Industry
   - **Investment Goals**: Primary goal, Target corpus, Liquidity timeline
   - **Risk Profile**: Income stability, Emergency fund, Drawdown tolerance
   - **Cash Flow**: Monthly investment, EMI, SIP step-up
   - **Family & Dependents**: Number and duration
   - **Behavioral Factors**: Decision style, Abandonment triggers

### Step 3: Save Profile
1. Click **"💾 Save & Continue"** button
2. You'll see: **"✅ Profile saved successfully!"**
3. The profile is saved to: `IID_filled.json`

### Step 4: Upload Portfolio & Run Analysis
1. Still on Dashboard, click **"🔍 Run Analysis"** in sidebar
2. Upload your portfolio Excel file
3. Dashboard analyzes your portfolio based on YOUR SAVED PROFILE
4. View comprehensive analysis across 6 tabs

### Step 5 (Optional): Use Asset Recommendations Dashboard
1. Start second dashboard on port 8502:
   ```bash
   python -m streamlit run asset_recommendations_dashboard.py --server.port 8502
   ```
2. Open: **http://localhost:8502**
3. The dashboard automatically loads your saved IID profile
4. Upload portfolio file
5. Get Nifty50 recommendations

---

## 📊 Where IID Data is Used

### Main Dashboard (dashboard.py)
- **Investor Fit Score (IFS)**: Calculates how well portfolio matches YOUR profile
- **Two-Dimensional Analysis**: Compares against your risk tolerance
- **Portfolio Quality Score (PQS)**: Adjusted for your goals
- **Recommendations**: Tailored to your specific situation

### Asset Recommendations Dashboard (asset_recommendations_dashboard.py)
- **Asset Scoring**: Weights assets based on your profile
- **Recommendations**: Finds Nifty50 assets matching your objectives
- **Impact Calculation**: Estimates improvement relative to your goals

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Skipping IID Form
**Problem**: Starting analysis without filling investor profile  
**Result**: Dashboard uses empty profile, recommendations are generic  
**Solution**: Always fill and save IID form FIRST (Step 2 & 3)

### ❌ Mistake 2: Only Using Asset Recommendations Dashboard
**Problem**: Not having completed IID form on Main Dashboard  
**Result**: "No investor profile found" warning  
**Solution**: Complete Main Dashboard workflow FIRST, then use recommendations

### ❌ Mistake 3: Using Outdated IID
**Problem**: Portfolio situation changed but IID not updated  
**Result**: Recommendations based on old risk profile  
**Solution**: Update IID form when your situation changes (income, goals, etc.)

### ❌ Mistake 4: Wrong File Format
**Problem**: Portfolio file with wrong column names  
**Result**: "Error loading portfolio"  
**Solution**: Use standard Excel format with columns: Instrument, Qty, Avg cost, LTP, Cur val, P&L

---

## 📁 IID File Structure

When you save your profile, it creates `IID_filled.json`:

```json
{
  "schema_version": "1.0",
  "investor_id": "user_20260118",
  "investor_profile": {
    "age_band": "30-35",
    "employment_type": "salaried",
    "industry": "Technology",
    "city_tier": "tier_1"
  },
  "investment_motivation": {
    "primary_goals": ["wealth_creation"],
    "goal_priority_rank": {
      "primary": "wealth_creation",
      "secondary": "inflation_beating"
    }
  },
  "time_horizon": {
    "earliest_liquidity_year": 2035,
    "comfortable_exit_year": 2045,
    "horizon_type": "flexible"
  },
  "target_corpus": {
    "amount": 10000000,
    "confidence_level": "approximate",
    "inflation_adjusted": true
  },
  "risk_profile": {
    "risk_capacity": {...},
    "risk_tolerance": {...},
    "risk_behavior_history": {...}
  },
  "cash_flow_and_investment_style": {...},
  "tax_profile": {...},
  "family_and_dependents": {...},
  "behavioral_triggers": {...},
  "meta": {
    "profile_created_at": "2026-01-18T10:30:00",
    "last_updated_at": "2026-01-18T10:30:00",
    "data_source": "user_declared"
  }
}
```

---

## 🔄 Complete Dashboard Flow

```
┌─────────────────────────────────────────────────────────┐
│ Start Main Dashboard (Port 8501)                        │
│ http://localhost:8501                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Fill Investor Profile Form                              │
│ - Personal info, goals, risk profile, cash flow         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Click "💾 Save & Continue"                              │
│ → Creates IID_filled.json                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Upload Portfolio & Run Analysis                         │
│ → Dashboard uses your saved IID profile                 │
│ → Shows comprehensive 6-tab analysis                    │
└─────────────────────────────────────────────────────────┘
                          ↓ (Optional)
┌─────────────────────────────────────────────────────────┐
│ Start Asset Recommendations (Port 8502)                 │
│ http://localhost:8502                                   │
│ → Automatically loads your IID profile                  │
│ → Upload portfolio & get Nifty50 recommendations        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Reference

| Step | Action | File/Port | Required |
|------|--------|-----------|----------|
| 1 | Start Main Dashboard | Port 8501 | ✅ YES |
| 2 | Fill Investor Profile | Browser | ✅ YES |
| 3 | Save Profile | IID_filled.json | ✅ YES |
| 4 | Upload Portfolio | Portfolio.xlsx | ✅ YES |
| 5 | Run Analysis | Dashboard | ✅ YES |
| 6 | View Results | 6 tabs | ✅ YES |
| 7 | Start Recommendations (Optional) | Port 8502 | ❌ OPTIONAL |
| 8 | Get Asset Recommendations | Browser | ❌ OPTIONAL |

---

## 📞 Troubleshooting

### Q: "No investor profile found" on Asset Recommendations
**A**: You haven't filled the IID form on Main Dashboard yet. Complete Step 2-3 first.

### Q: IID form not showing on Main Dashboard
**A**: The form appears on first load. If it's not showing, check you haven't already saved an IID. Click "🔍 Run Analysis" to proceed.

### Q: Changes to IID not reflecting in recommendations
**A**: 
1. Edit IID on Main Dashboard
2. Re-save it
3. Refresh Asset Recommendations Dashboard

### Q: Saved IID file missing
**A**: Check `IID_filled.json` exists in `f:\AI Insights Dashboard\` folder. If missing, fill and save the form again.

---

## 🔄 When to Update Your IID

Update your investor profile when:
- ✏️ Age band changes
- ✏️ Job changes (employment type, industry)
- ✏️ Income changes significantly
- ✏️ Investment goals change
- ✏️ Life situation changes (marriage, kids, dependents)
- ✏️ Risk tolerance changes (experienced market crash, etc.)
- ✏️ Time horizon changes (retirement plans change)

---

**Version**: 1.0  
**Last Updated**: January 18, 2026  
**Status**: ✅ Recommended Workflow
