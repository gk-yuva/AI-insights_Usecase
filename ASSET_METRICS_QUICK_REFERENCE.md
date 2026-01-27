# 9 Asset-Level Metrics - Quick Reference Guide

## What Are They?

Nine metrics that describe each stock's **financial performance and risk characteristics**. These are inputs to the Classification ML model that predicts whether adding/dropping an asset will improve portfolio performance.

---

## The 9 Metrics at a Glance

### 1️⃣ **Returns (3 metrics)**
- **5-day MA**: Short-term momentum
- **20-day MA**: Medium-term trend  
- **60-day MA**: Longer-term trend

| Metric | Range | Example |
|--------|-------|---------|
| returns_5d_ma | -2% to +2% | 0.35% |
| returns_20d_ma | -1.5% to +1.5% | 0.12% |
| returns_60d_ma | -1% to +1% | 0.08% |

**Interpretation**: Positive values = stock moving up, Negative = stock moving down

---

### 2️⃣ **Volatility (2 metrics)**
- **30-day**: Recent volatility
- **90-day**: Medium-term volatility

| Metric | Range | Example |
|--------|-------|---------|
| volatility_30d | 10% to 40% | 18.5% |
| volatility_90d | 12% to 35% | 15.2% |

**Interpretation**: Higher = More risky, Lower = More stable

---

### 3️⃣ **Risk-Adjusted Returns (3 metrics)**

#### **Sharpe Ratio** = Risk-adjusted return
- Accounts for: Return earned per unit of risk
- Higher is better
- Range: 0 to 3

#### **Sortino Ratio** = Downside risk-adjusted return
- Like Sharpe but only penalizes downside volatility
- Ignores "good" volatility (upside moves)
- Higher is better
- Range: 0 to 3

#### **Calmar Ratio** = Return / Drawdown
- How much return vs how much risk
- Higher is better
- Range: 0 to 5

| Stock | Sharpe | Sortino | Calmar |
|-------|--------|---------|--------|
| HINDUNILVR | 1.95 | 1.29 | 0.97 |
| HDFC | 1.88 | 2.39 | 2.75 |
| RELIANCE | 1.73 | 0.11 | 0.08 |

**Interpretation**: HDFC is best risk-adjusted performer

---

### 4️⃣ **Max Drawdown (1 metric)**
- **90-day Drawdown**: Worst loss in last 90 days
- Range: 5% to 30%

| Stock | Max_DD_90d | Meaning |
|-------|-----------|---------|
| HDFC | 12.4% | Lost at most 12.4% |
| HINDUNILVR | 20.9% | Lost at most 20.9% |

**Interpretation**: Higher = More downside risk, Lower = Less downside risk

---

### 5️⃣ **Distribution Shape (2 metrics)**

#### **Skewness** = Asymmetry of returns
- Positive = More upside surprises (good)
- Negative = More downside surprises (bad)
- Range: -2 to +2

#### **Kurtosis** = Tail risk
- Higher = More extreme events (bad)
- Lower = More normal distribution (good)
- Range: -1 to +4

| Stock | Skewness | Kurtosis | Meaning |
|-------|----------|----------|---------|
| HINDUNILVR | -0.46 | 0.76 | Left tail risk, moderate extremes |
| HDFC | -0.97 | 2.71 | Strong left tail, high extremes |

**Interpretation**: More negative skew = More downside risk, High kurtosis = More extreme events

---

### 6️⃣ **Beta (1 metric)**
- **Systematic Risk**: Volatility vs market
- β = 1.0 → Moves with market
- β < 1.0 → Less volatile (defensive)
- β > 1.0 → More volatile (aggressive)
- Range: 0.3 to 1.8

| Stock | Beta | Classification |
|-------|------|-----------------|
| HINDUNILVR | 0.58 | Defensive ← |
| HDFC | 0.93 | Near market ↔ |
| JSWSTEEL | 1.31 | Aggressive → |

**Interpretation**: 0.58 = 58% of market volatility (half as risky as market)

---

## How to Use These 9 Metrics

### For Asset Selection
```
Good Asset Profile:
✓ High Sharpe ratio (>1.5)
✓ Lower volatility (<20%)
✓ Lower drawdown (<15%)
✓ Positive/low skewness
```

### For Portfolio Building
```
Balanced Portfolio:
✓ Mix of defensive (β < 0.8) and aggressive (β > 1.2)
✓ Average volatility in 15-20% range
✓ Combined Sharpe > 1.0
```

### For Risk Management
```
Risk Monitoring:
✓ Watch for increasing drawdown
✓ Monitor skewness for tail risk
✓ Track volatility changes
```

---

## Data Location

📁 **Nifty50**: `Asset returns/nifty50/Nifty50_metrics.csv`  
📁 **Nifty Next50**: `Asset returns/nifty_next_50/Nifty_Next50_metrics.csv`

---

## Example Usage

### Find defensive stocks (low risk)
```
Filter: beta < 0.8 AND volatility_30d < 15
Result: Safest stocks for conservative portfolios
```

### Find growth stocks (high return potential)
```
Filter: sharpe_ratio > 1.5 AND returns_60d_ma > 0.1
Result: Best risk-adjusted performers
```

### Find crisis-resistant stocks
```
Filter: skewness > 0 AND kurtosis < 1
Result: Stocks with less downside risk
```

---

## Column Names for Reference

```
Asset Metrics:
- returns_5d_ma
- returns_20d_ma
- returns_60d_ma
- volatility_30d
- volatility_90d
- sharpe_ratio
- sortino_ratio
- calmar_ratio
- max_drawdown_90d
- skewness
- kurtosis
- beta
```

---

## Key Takeaways

| Metric | What It Measures | Higher = | Use For |
|--------|-----------------|----------|---------|
| Sharpe | Risk-adj return | Better | Asset selection |
| Volatility | Price swings | Worse | Risk assessment |
| Beta | Systematic risk | More risky | Diversification |
| Drawdown | Worst loss | Worse | Risk limit check |
| Skewness | Tail direction | Better if > 0 | Tail risk |
| Kurtosis | Extreme events | Worse | Extreme risk |

---

✅ These 9 metrics will be combined with portfolio and market context to train the ML classification model!

*Next: Portfolio-level inputs (current holdings, portfolio metrics)*
