# 🎯 START HERE - Dashboard Setup Guide

## ⚠️ IMPORTANT: IID Data Flow

Your **Investor Information Document (IID)** is the foundation of all analysis in this system.

### The Problem We Solved
The Asset Recommendations Dashboard was trying to run WITHOUT your investor profile data.

### The Solution
**Always start with the Main Dashboard to fill your IID first!**

---

## 🚀 Quick Start (5 Steps)

### Step 1: Start Main Dashboard
```bash
cd f:\AI Insights Dashboard
python -m streamlit run dashboard.py --server.port 8501
```

**Main Dashboard is NOW RUNNING on: http://localhost:8501**

### Step 2: Fill Your Investor Profile
Open **http://localhost:8501** in browser

You'll see a form with sections:
- 👤 Personal Profile (Age, Job, City, Industry)
- 🎯 Investment Goals (Primary goal, Target amount, Timeline)
- ⚠️ Risk Profile (Income stability, Emergency fund, Drawdown tolerance)
- 💰 Cash Flow (Monthly investment, EMI, SIP step-up)
- 👨‍👩‍👧‍👦 Family & Dependents
- 🧠 Behavioral Factors

**Fill all fields carefully** - these determine your entire investment strategy.

### Step 3: Save Your Profile
Click **"💾 Save & Continue"** button

You'll see: **"✅ Profile saved successfully!"**

This creates: `IID_filled.json` (your investor profile)

### Step 4: Upload Portfolio & Run Analysis
- Upload your portfolio Excel file
- Click **"🔍 Run Analysis"**
- Browse 6 analysis tabs with your personalized insights

### Step 5 (Optional): Use Asset Recommendations
After saving IID on Main Dashboard:
```bash
python -m streamlit run asset_recommendations_dashboard.py --server.port 8502
```

Open **http://localhost:8502**
- Dashboard auto-loads your IID profile
- Upload portfolio
- Get Nifty50 recommendations

---

## 📊 Dashboard Comparison

| Feature | Main Dashboard | Asset Recommendations |
|---------|---|---|
| **Port** | 8501 | 8502 |
| **Requires IID Form** | ✅ YES (fill here) | ✅ YES (reads from Main) |
| **Asset Universe** | All major stocks | Nifty50 only |
| **Analysis Type** | Comprehensive (6 tabs) | Quick recommendations |
| **Time to Analysis** | ~2-3 minutes | ~30-60 seconds |
| **Use For** | Portfolio health check | Find new assets to add |
| **When to Use** | First, or when portfolio changes | After Main Dashboard |

---

## 📁 Where IID Data Goes

```
You Fill Form (Main Dashboard)
        ↓
     ✅ Save
        ↓
IID_filled.json (created in workspace)
        ↓
   Asset Recommendations Dashboard reads this
        ↓
✅ Personalized recommendations generated
```

---

## ⚡ Two Dashboards, One System

### Dashboard 1: Main Portfolio Health (Port 8501)
**When to use**: First time, or when portfolio changes
**Purpose**: Comprehensive 6-tab analysis
**Key Metrics**: PQS, IFS, Risk, Quality, Recommendations

**How to run**:
```bash
python -m streamlit run dashboard.py --server.port 8501
```

### Dashboard 2: Asset Recommendations (Port 8502)
**When to use**: After saving IID on Main Dashboard
**Purpose**: Quick Nifty50 asset recommendations
**Key Output**: Add/Drop lists, 3-phase roadmap

**How to run**:
```bash
python -m streamlit run asset_recommendations_dashboard.py --server.port 8502
```

---

## ✅ Verification Checklist

Before accessing Asset Recommendations Dashboard:

- [ ] Main Dashboard running on port 8501
- [ ] IID form completed and saved
- [ ] See "✅ Profile saved successfully!" message
- [ ] `IID_filled.json` file exists in workspace folder
- [ ] Portfolio uploaded to Main Dashboard
- [ ] Analysis completed on Main Dashboard

If all ✅, then start Asset Recommendations Dashboard!

---

## 🆘 Common Issues

### Issue: "No investor profile found" on port 8502
**Solution**: 
1. Go back to Main Dashboard (8501)
2. Complete the IID form
3. Click "💾 Save & Continue"
4. Refresh Asset Recommendations page (8502)

### Issue: IID form not visible on port 8501
**Solution**:
- Form appears on first load
- If not showing, you've already saved an IID
- Click "🔍 Run Analysis" to proceed
- To edit, scroll to sidebar to see edit option

### Issue: Portfolio upload fails
**Solution**: 
- Use Excel format (.xlsx or .xls)
- Ensure columns: Instrument, Qty, Avg cost, LTP, Cur val, P&L
- File size < 5MB

### Issue: Analysis takes too long
**Solution**:
- Nifty50 analysis (8502) is faster than Main Dashboard (8501)
- Both need internet for stock data fetch
- First run may be slower (data caching)

---

## 📚 Learn More

- **IID Workflow**: Read [IID_WORKFLOW.md](IID_WORKFLOW.md)
- **Data Flow**: Read [IID_DATA_FLOW.md](IID_DATA_FLOW.md)
- **Dashboard Guide**: Read [DASHBOARDS_GUIDE.md](DASHBOARDS_GUIDE.md)
- **Full Docs**: Read [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

---

## 🔑 Key Concepts

### IID (Investor Information Document)
Your complete investment profile. Created by filling the form on Main Dashboard.
- Contains: Age, goals, risk tolerance, cash flow, family situation
- Used by: Both dashboards for analysis
- Format: JSON file (`IID_filled.json`)

### PQS (Portfolio Quality Score)
How good is your portfolio? (0-100 scale)
- High PQS = High quality holdings, good metrics
- Low PQS = Consider improving quality

### IFS (Investor Fit Score)
Is your portfolio right for YOU? (0-100 scale)
- High IFS = Your portfolio matches your goals & risk tolerance
- Low IFS = Portfolio may not align with your situation

### Nifty50
India's 50 largest, most liquid companies
- Used by Asset Recommendations Dashboard
- Includes: TCS, Reliance, HDFC, ICICI, Infy, etc.

---

## 🎓 First Time User Flow

```
1. Start Main Dashboard (port 8501)
   ↓
2. See "📝 Investor Information" form
   ↓
3. Fill all form fields (takes ~5-10 minutes)
   ↓
4. Click "💾 Save & Continue"
   ↓
5. Upload portfolio Excel
   ↓
6. Click "🔍 Run Analysis"
   ↓
7. Explore 6 tabs of analysis
   ↓
8. (Optional) Start Asset Recommendations (port 8502)
   ↓
9. Get Nifty50 recommendations
```

---

## 🔗 Access Points

**Main Dashboard**
- URL: http://localhost:8501
- Purpose: IID form + Comprehensive analysis
- Required: IID form must be filled here first

**Asset Recommendations**
- URL: http://localhost:8502
- Purpose: Nifty50 recommendations
- Required: IID data from Main Dashboard

---

## 📞 Troubleshooting

1. **Dashboard won't load**: Check port not in use, try different port
2. **Form not showing**: Scroll down in sidebar, may be below fold
3. **Analysis errors**: Check internet connection for stock data
4. **IID not saved**: Ensure "💾 Save & Continue" clicked, not just "Run Analysis"
5. **Asset Recommendations empty**: Ensure Main Dashboard IID saved first

---

## ✨ Tips for Best Results

1. **Be honest on IID form**: Your answers determine analysis accuracy
2. **Use current data**: Portfolio should be up-to-date
3. **Update IID when situation changes**: Job change, goals change, etc.
4. **Check both dashboards**: Main for health, Recommendations for growth
5. **Print results**: Screenshot analysis for records

---

**Status**: ✅ Main Dashboard running on http://localhost:8501  
**Next**: Open browser and fill your IID form!

---

*Version 1.0 - January 18, 2026*
