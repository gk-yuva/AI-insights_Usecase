# Two-Dimensional Portfolio Analysis System

## 🎯 Revolutionary Approach: Portfolio Quality vs. Investor Fit

### The Problem with Traditional Portfolio Analysis

Most portfolio analysis tools answer only ONE question:
> **"Is this a good portfolio?"**

But they miss the critical second dimension:
> **"Is this portfolio right for THIS investor?"**

---

## ✨ Our Solution: Two Orthogonal Scores

We compute TWO independent scores that together provide complete insight:

| Dimension | Question Answered | What It Measures |
|-----------|-------------------|------------------|
| **Portfolio Quality Score (PQS)** | Is this portfolio efficient? | Risk-adjusted performance in isolation |
| **Investor Fit Score (IFS)** | Is this portfolio right for YOU? | Alignment with YOUR capacity and behavior |

### Why This Matters

```
Traditional App:  Portfolio Score: 85/100 ✅
                 "Great portfolio!"

Our App:         PQS: 85/100  ✅  IFS: 35/100 ❌
                "Great portfolio, TERRIBLE for you!"
```

This is your **competitive differentiation**.

---

## 📊 Dimension 1: Portfolio Quality Score (PQS)

### What It Measures
Portfolio efficiency regardless of who owns it.

### Components (Weighted)

| Metric | Weight | What It Shows |
|--------|--------|---------------|
| **Sharpe Ratio** | 30% | Risk-adjusted returns |
| **Sortino Ratio** | 25% | Downside risk-adjusted returns |
| **Jensen's Alpha** | 20% | Skill vs. market |
| **Annual Returns** | 15% | Absolute performance |
| **Volatility** | 10% | Risk level (lower is better) |

### Normalization

Each metric is normalized to 0-100 using threshold values:

```python
Sharpe Ratio:
  Bad (0-30):       < 0.5
  Average (30-70):  0.5 - 1.2
  Excellent (70-100): > 1.2
```

### PQS Categories

- **80-100**: Excellent - World-class portfolio
- **65-79**: Good - Above average performance  
- **50-64**: Average - Acceptable returns
- **35-49**: Below Average - Needs improvement
- **0-34**: Poor - Significant issues

---

## 🎯 Dimension 2: Investor Fit Score (IFS)

### What It Measures
How well the portfolio matches the investor's **capacity** and **behavior**.

### The Core Innovation

IFS is calculated using **4 Investor Indices** extracted from the IID:

#### 1. Risk Capacity Index (RCI) - 0-100
**Ability to bear risk financially**

Derived from:
- Income stability (30%)
- Emergency fund months (25%)
- Liability burden (25%)
- Portfolio concentration (20%)

```
Higher RCI = Greater financial capacity for risk
```

#### 2. Risk Tolerance Index (RTI) - 0-100
**Willingness to accept risk psychologically**

Derived from:
- Drawdown response (40%): invest_more/hold/reduce/exit
- Sleep-loss threshold (35%): % drawdown causing stress
- Loss vs gain preference (25%): loss_averse vs gain_seeking

```
Higher RTI = Greater psychological comfort with risk
```

#### 3. Behavioral Fragility Index (BFI) - 0-100
**Likelihood of abandoning plan**

Derived from:
- Past exit behavior (40%): exited during crash? re-entered?
- Abandonment triggers (30%): count of panic triggers
- Decision autonomy (30%): rules_based/guided/full_control

```
Higher BFI = Higher risk of panic selling
```

#### 4. Time Horizon Strength (THS) - 0-100
**Ability to stay invested long-term**

Derived from:
- Horizon duration (40%): years until exit
- Horizon flexibility (35%): fixed/flexible/aspirational
- Dependency timeline (25%): when dependents need money

```
Higher THS = Better suited for long-term investing
```

---

## 🧮 IFS Calculation Logic

### Step 1: Calculate Portfolio Risk Indices

From portfolio metrics, calculate:

1. **Portfolio Risk Index (PRI)** - 0-100
   - Volatility (35%)
   - VaR 95% (30%)
   - Max Drawdown (25%)
   - Beta (10%)

2. **Portfolio Drawdown Severity (PDS)** - 0-100
   - Max Drawdown (40%)
   - VaR 99% (35%)
   - Downside deviation (25%)

### Step 2: Apply Mismatch Rules

```python
IFS = 100
      − w1 × max(0, PRI − RCI)           # Capacity mismatch
      − w2 × max(0, PDS − Effective_RTI)  # Tolerance mismatch
      − w3 × (BFI × PRI / 100)            # Fragility amplifier

where:
  w1 = 0.4  (structural mismatch weight)
  w2 = 0.4  (behavioral mismatch weight)
  w3 = 0.2  (fragility weight)
  
  Effective_RTI = RTI × (1 − BFI/100)
```

### The Three Mismatch Rules

#### Rule 1: Capacity ≥ Portfolio Risk
```
If PRI > RCI → Structural mismatch
Portfolio risk exceeds financial capacity
```

#### Rule 2: Tolerance ≥ Drawdown Severity  
```
If PDS > Effective_RTI → Behavioral mismatch
Drawdowns will cause psychological pain
```

#### Rule 3: Fragility Amplifies Risk
```
High BFI reduces effective tolerance
Even moderate risk becomes dangerous
```

---

## 🎭 The Four Quadrants

### Quadrant Analysis Matrix

```
         │
   Poor  │  POOR PORTFOLIO      │  IDEAL ZONE
   Fit   │  GOOD FIT           │  EXCELLENT FIT
         │  (Fix portfolio)     │  (Perfect!)
    IFS  ├─────────────────────┼────────────────────
         │                     │
   Good  │  GOOD PORTFOLIO     │  GOOD PORTFOLIO
   Fit   │  POOR FIT           │  GOOD FIT
         │  (This is you!)     │  (Aligned)
         │                     │
         └─────────────────────┴────────────────────
               Low PQS               High PQS
```

### Quadrant Interpretations

#### ✅ Top-Right: IDEAL ZONE (PQS ≥65, IFS ≥65)
- **Meaning**: Excellent portfolio that matches you perfectly
- **Action**: Maintain and monitor

#### ⚠️ Top-Left: POOR PORTFOLIO, GOOD FIT (PQS <65, IFS ≥65)
- **Meaning**: Portfolio matches your profile but performance is weak
- **Action**: Improve portfolio quality (rebalance, better stocks)

#### ⚠️ Bottom-Right: GOOD PORTFOLIO, POOR FIT (PQS ≥65, IFS <65)
- **Meaning**: Strong portfolio but misaligned with YOUR profile
- **Action**: Reduce risk OR build capacity
- **Key Insight**: "Not bad portfolio, just not YOUR portfolio"

#### ❌ Bottom-Left: NEEDS IMPROVEMENT (PQS <65, IFS <65)
- **Meaning**: Both quality and fit are problematic
- **Action**: Major restructuring needed

---

## ⭐ Plan Survival Probability (PSP)

### The Killer Metric

> **"What's the probability you'll stick to this plan?"**

### Calculation

```python
PSP = Base_Fit + Fragility_Impact + Horizon_Boost

where:
  Base_Fit = IFS score
  Fragility_Impact = −BFI × 0.5
  Horizon_Boost = (THS − 50) × 0.3
```

### Why PSP Matters More Than CAGR

Traditional advisors focus on:
- Expected returns
- CAGR projections
- Optimal allocations

But miss the critical question:
- **Will the investor actually stay invested?**

PSP directly addresses behavioral risk.

---

## 🎯 Actionable Diagnostics

### Instead of Generic Advice

❌ **Traditional**: "Your portfolio is aggressive"  
❌ **Traditional**: "Consider reducing risk"

### We Provide Specific, Data-Driven Insights

✅ **Our Diagnostics**:
```
❌ CRITICAL: Your portfolio drawdown potential (−28%) 
   exceeds your sleep-loss threshold (−20%)

❌ WARNING: Your income volatility + high equity exposure 
   increases abandonment risk by 40%

✅ INSIGHT: Returns are strong (27%), but behavior risk 
   (BFI: 77/100) is very high
```

### Targeted Recommendations

Based on mismatch type:

**If Capacity Mismatch (PRI > RCI)**:
- Reduce equity allocation by X%
- Add stable income assets
- Build emergency fund first

**If Tolerance Mismatch (PDS > RTI)**:
- Lower exposure to high-volatility stocks
- Add downside protection (puts, stop-losses)
- Increase diversification

**If High Fragility (BFI > 60)**:
- Implement SIP instead of lump sum
- Set up auto-rebalancing
- Add rule-based sell discipline
- Consider robo-advisory

---

## 💡 Example Use Cases

### Case 1: High Performer, Poor Fit

```
Investor: 30-year-old, salaried, first-time investor
Portfolio: Aggressive small-cap fund
```

**Analysis**:
- PQS: 82/100 (Excellent returns, Sharpe 1.8)
- IFS: 38/100 (Poor fit)

**Why Poor Fit?**:
- RCI: 45 (low emergency fund, no liabilities buffer)
- RTI: 35 (exits during drawdown historically)
- BFI: 75 (high fragility)
- PRI: 85 (very risky portfolio)

**Diagnosis**:
> "This portfolio has delivered 32% returns, but there's a 78% 
> chance you'll exit during the next 20% correction."

**Recommendation**:
> Move to balanced fund OR build 12-month emergency fund first

---

### Case 2: Mediocre Portfolio, Perfect Fit

```
Investor: 55-year-old, retired, pension income
Portfolio: Conservative debt+equity fund
```

**Analysis**:
- PQS: 48/100 (Below average returns, Sharpe 0.6)
- IFS: 88/100 (Excellent fit)

**Why Good Fit?**:
- RCI: 90 (strong pension, no liabilities)
- RTI: 85 (comfortable with 15% drawdowns)
- BFI: 25 (disciplined, rules-based)
- PRI: 35 (low-risk portfolio)

**Diagnosis**:
> "Returns are modest (9%), but this portfolio perfectly matches 
> your income needs and risk comfort."

**Recommendation**:
> Maintain allocation; focus on income stability over growth

---

## 🔮 Strategic Advantage

### Why This Approach Wins

#### 1. **Behavioral Focus**
- Traditional apps optimize for returns
- We optimize for **staying invested**
- PSP is your north star, not CAGR

#### 2. **Human-Centric**
- Portfolio analytics exist
- **Investor analytics** are our innovation
- We profile the human, not just the portfolio

#### 3. **Explainable**
- Clear two-dimensional visualization
- Specific diagnostics, not vague warnings
- Actionable recommendations

#### 4. **Adaptive**
- IID captures changing life situations
- Re-run analysis when life changes
- Dynamic fit vs static profiling

---

## 📊 Technical Implementation

### Data Requirements

1. **Portfolio Data** (existing):
   - Holdings, quantities, prices
   - Historical performance
   - Risk metrics

2. **IID Document** (new):
   - Investor profile
   - Financial situation
   - Behavioral history
   - Time horizon
   - Goals and triggers

### Workflow

```
1. Load Portfolio → Calculate Metrics
2. Load IID → Extract Indices (RCI, RTI, BFI, THS)
3. Calculate PQS (portfolio in isolation)
4. Calculate IFS (portfolio × investor)
5. Calculate PSP (plan survival)
6. Generate Quadrant Analysis
7. Provide Diagnostics & Recommendations
```

---

## 🚀 Implementation Status

### ✅ Completed Modules

1. **investor_profile.py** - IID parser and index calculator
2. **portfolio_quality.py** - PQS scoring engine
3. **investor_fit.py** - IFS calculation and PSP
4. **portfolio_analyzer.py** - Main orchestrator (updated)

### ✅ Working Features

- [x] IID JSON parsing
- [x] Four investor indices (RCI, RTI, BFI, THS)
- [x] Portfolio Quality Score (PQS)
- [x] Portfolio risk indices (PRI, PDS)
- [x] Investor Fit Score (IFS)
- [x] Mismatch detection and penalties
- [x] Plan Survival Probability (PSP)
- [x] Quadrant analysis
- [x] Specific diagnostics
- [x] Targeted recommendations

---

## 📈 Sample Output

```
┌──────────────────────────────────────────────────────┐
│  PORTFOLIO QUALITY SCORE  │  INVESTOR FIT SCORE      │
│         70.0/100           │       36.7/100          │
│          Good              │      Poor Fit           │
└──────────────────────────────────────────────────────┘

📍 Portfolio Positioning:
   ⚠️ GOOD PORTFOLIO, POOR FIT
   Strong portfolio performance, but misaligned with your profile

⭐ Plan Survival Probability: 6.6/100
   Low probability - high risk of plan abandonment
   Risk Level: Very high abandonment risk

💡 Strategic Recommendations:
   1. IMPROVE INVESTOR FIT:
      - Reduce portfolio risk to match your financial capacity
      - Lower exposure to high-volatility assets
      - Add behavioral guardrails (SIPs, auto-rebalancing)
```

---

## 🎓 Key Takeaways

### For Investors

1. **Portfolio quality ≠ Portfolio suitability**
2. **Your behavior matters more than market behavior**
3. **Staying invested > Picking winners**

### For the Product

1. **Unique positioning** - No competitor does this
2. **Behavioral edge** - Focus on plan adherence
3. **Clear value** - Prevent costly mistakes (panic selling)

---

**Built**: December 2024  
**Status**: ✅ Fully Operational  
**Innovation**: Two-dimensional analysis (PQS × IFS)
