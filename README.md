# Crude Oil Price Forecasting

Forecasting global crude oil prices using Time Series (SARIMAX) and
Machine Learning models (Random Forest, XGBoost) on 192 monthly
observations spanning 2006–2022.

---

## Project Overview

This project investigates whether macroeconomic indicators (CPI and
exchange rates from India, USA, Germany, China, and South Korea) can
improve crude oil price forecasts. Three models are compared: SARIMAX,
Random Forest, and XGBoost — with engineered lag and rolling features.

Key finding: macroeconomic variables were statistically insignificant
(p > 0.05) in the SARIMAX model, and lag-based features dominated
feature importance in ML models — confirming that oil prices are
primarily driven by their own past values.

---

## Dataset

- **Period:** January 2006 – December 2022 (192 monthly observations)
- **Target variable:** Global Crude Oil Price (USD)
- **Features:** CPI and Exchange Rates from 5 countries
  - India, USA, Germany, China, South Korea

---

## Models Used

| Model | Type | Key Details |
|---|---|---|
| SARIMAX | Time Series | ARIMA(1,1,0) with exogenous macro variables |
| Random Forest | ML | 300 estimators, lag + rolling features |
| XGBoost | ML | 500 estimators, learning rate 0.03, max depth 3 |

---

## Feature Engineering

- Lag features: lag_1, lag_3, lag_6, lag_12
- Rolling averages: 3-month, 6-month, 12-month
- Rolling std: 3-month
- Seasonality: month_sin, month_cos (cyclical encoding)

---

## Results Summary

### Test Set Performance (80/20 time-based split)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| SARIMAX | 33.78 | 28.01 | -1.51 |
| Random Forest | 43.96 | 38.86 | -3.25 |
| **XGBoost** | **11.04** | **8.72** | **0.70** |

### XGBoost — 5-Fold Time-Series Cross-Validation

| Fold | RMSE | MAE | R² |
|---|---|---|---|
| 1 | 33.47 | 31.00 | -5.19 |
| 2 | 11.29 | 8.83 | 0.79 |
| 3 | 16.71 | 14.42 | -1.14 |
| 4 | 10.03 | 7.09 | 0.53 |
| 5 | 9.60 | 7.16 | 0.58 |
| **Mean** | **16.22 ± 10.05** | **13.70 ± 10.12** | **-0.89 ± 2.53** |

> **Note:** Fold 1 covers the 2020 COVID-19 period — a structural break
> that severely impacted model performance. Folds 2, 4, and 5 show
> strong R² values (0.53–0.79), reflecting genuine predictive ability
> under normal market conditions.

---

## Key Findings

- **Macroeconomic indicators were statistically insignificant** (p > 0.05)
  in SARIMAX — CPI and exchange rates did not improve oil price forecasts
- **Strong autoregressive behavior confirmed** — oil prices are primarily
  driven by their own past values (significant AR(1) term)
- **XGBoost achieved the best performance** — RMSE: 11.04, MAE: 8.72,
  R²: 0.70 on the test set
- **SARIMAX and Random Forest underperformed** (negative R²) due to
  inability to handle the high volatility in the 2019–2022 test period
- **Lag-based features dominated** both Gini and permutation importance
  rankings, outperforming all macroeconomic variables
- **COVID-19 (2020) acted as a structural break** — fold-level CV
  results show dramatically worse performance when this period is in
  the test fold

---

## How to Run

1. Clone this repository
```bash
git clone https://github.com/ts-safiya/crude-oil-price-forecasting
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn xgboost joblib
```

3. Open the notebook
```bash
jupyter notebook crudeoil.ipynb
```

---

## Tools & Libraries

Python, Pandas, NumPy, Statsmodels, pmdarima, Scikit-learn, XGBoost,
Matplotlib, Seaborn

---

## Author

**Safiya T S** — Machine Learning Engineer | MSc Statistics, AMU
[LinkedIn](https://linkedin.com/in/safiya-t-s) | [GitHub](https://github.com/ts-safiya)
