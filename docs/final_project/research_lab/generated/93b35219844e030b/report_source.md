# Portfolio Regime & Stress Propagation Lab

## Dataset
- Universe: AAPL, MSFT, GOOGL, AMZN, META, NVDA
- Window: 2021-04-18 to 2026-04-18
- Frequency: Daily
- Portfolio construction: Equal-weight synthetic portfolio

## Methods
- Rolling time-series features: realized volatility, pairwise correlation, breadth, dispersion, lead-lag concentration
- Stress Propagation Score: standardized blend of five state variables
- Regime detection: 3-cluster KMeans on rolling state features
- Forecast model: ARIMA(1, 0, 1) on 20-day realized volatility

## Key Findings
- Stress Contagion carried an average realized vol of 33.63% versus 32.25% in Calm Expansion.
- The Stress Propagation Score was higher in Stress Contagion (+1.28) than in Calm Expansion (-0.10).
- Before stress-regime entries, the score rose in 53.8% of observed transitions.
- The strongest lead-lag relation was MSFT leading NVDA by 1 day(s) (corr=+0.08).
- ARIMA(1, 0, 1) produced RMSE 0.0168 versus baseline 0.0168 (-0.0001 improvement).

## Exported Visuals
- normalized_prices: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\normalized_prices.png
- rolling_state: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\rolling_state.png
- stress_regimes: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\stress_regimes.png
- lead_lag_heatmap: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\lead_lag_heatmap.png
- transition_heatmap: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\transition_heatmap.png
- forecast_actual: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\93b35219844e030b\charts\forecast_actual.png
