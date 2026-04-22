# Rubric Alignment

## Data Collection & Cleaning

- Independent dataset: Yahoo Finance daily data for AAPL, MSFT, GOOGL,
  AMZN, META, NVDA
- Evidence: `data/aligned_prices.csv`, notebook preprocessing cells,
  report methods section

## EDA & Visualizations

- Core visuals: normalized prices, rolling state metrics, stress/regime
  shading, lead-lag heatmap, transition heatmap, forecast vs actual
- Evidence: `figures/` and `portfolio_regime_stress_lab.ipynb`

## Modeling or Statistical Analysis

- Regime construction: 3-cluster rolling-state segmentation
- Forecasting model: ARIMA(2, 0, 1)
- Metrics: RMSE 0.0269, MAE 0.0167,
  AIC -6592.87, BIC -6568.17

## Insights & Interpretation

- Stress regime has materially higher realized volatility and lower breadth
- Transition/dwell analytics add interpretation beyond raw charts
- Evidence: `portfolio_regime_stress_lab_report.pdf`, `supporting_material/speaker_notes.md`

## Code Quality

- Backend engine: `backend/research_lab/engine.py`
- Packaging utility: `backend/research_lab/package_submission.py`
- Reproducible exported run: `dfc04a9e8d78431e`
