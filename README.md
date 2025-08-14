# Bayesian Time Series Forecasting for Stock Prices

**ARIMA vs. Prophet (with uncertainty) on daily equity data**

Goal: Forecast a stock’s **daily closing price** and its **uncertainty band**; compare simple baselines to **ARIMA** and **Prophet** on an out-of-sample year.

---

## Tools Used
- Python
- pandas, numpy
- matplotlib
- scikit-learn (MAE/RMSE)
- statsmodels (ADF, ACF/PACF)
- pmdarima (auto_arima)
- prophet (Meta’s Prophet)

---

## Project Steps
1. **Loaded and cleaned data** (`all_stocks_5yr.csv` from Kaggle), standardized columns (`ticker`, `date`, `close`, …).
2. **Exploratory look:** price plot, log-returns, ADF test, ACF/PACF.
3. **Time split:** last **252 trading days** as **test**; earlier data as **train**.
4. **Baselines:**  
   - **Naïve** (next = last close)  
   - **MA-20** (next = 20-day mean)
5. **ARIMA (log-price):** `auto_arima` with **drift** and **weekday seasonality (m=5)**; generated prediction intervals.
6. **Interval calibration:** swept `alpha` and chose a value to target **~80% empirical coverage** on the test window.
7. **Prophet (log-price):** no weekly/yearly seasonality, **changepoint_prior_scale=0.2**; predicted on **business-day** dates; back-transformed to price.
8. **Evaluation:** MAE, RMSE, and **coverage** (% of actuals inside the band).
9. **Plots + comparison table** for a selected ticker (e.g., `AAPL`).

---

## Results (Example: AAPL, test horizon ≈252 trading days)

| Model                           | MAE   | RMSE  | n   | Interval_Coverage_% |
|---------------------------------|-------|-------|-----|---------------------|
| **ARIMA (drift+weekly, α=0.50)**| 12.52 | 14.18 | 252 | 78.6                |
| **Prophet (log, cps=0.2, 80%)** | 16.47 | 17.22 | 252 | 82.9                |
| Naïve                           | 24.10 | 27.05 | 252 | —                   |
| MA(20)                          | 32.93 | 35.15 | 252 | —                   |

**Takeaways**
- **ARIMA** delivered the best point accuracy and near-target ~80% coverage after alpha tuning.  
- **Prophet (log)** produced reasonable accuracy and slightly **over-covered** (~83%), with the expected asymmetric (log-normal) band.  
- Baselines underperformed → modeling adds value.  
- Long horizons widen uncertainty for both models; shorter horizons (60–120d) tighten bands.

> *Numbers will vary by ticker and horizon; the example above uses AAPL.*

---

## File Structure
- `bayesian_ts_forecasting.ipynb` : full working notebook (data → models → metrics → plots)
- `data/all_stocks_5yr.csv` : Kaggle S&P 500 OHLCV (place the CSV here)
- `outputs/` : saved figures and CSVs (optional)
- `README.md` : project summary
- `requirements.txt` : minimal dependencies

---

## How to Run

```bash
# 1) (Optional) create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt
# If Prophet fails to build, try:
# pip install --no-build-isolation prophet

# 3) Launch Jupyter
jupyter notebook bayesian_ts_forecasting.ipynb
