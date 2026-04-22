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
- Forecast model: ARIMA(2, 0, 1) on 20-day realized volatility

## Key Findings
- Stress Contagion carried an average realized vol of 45.90% versus 7.65% in Calm Expansion.
- The Stress Propagation Score was higher in Stress Contagion (+1.49) than in Calm Expansion (-0.93).
- Before stress-regime entries, the score rose in 35.7% of observed transitions.
- The strongest lead-lag relation was GOOGL leading AAPL by 3 day(s) (corr=+0.12).
- ARIMA(2, 0, 1) closely matched the naive volatility baseline (RMSE 0.0269 versus 0.0271).

## Exported Visuals
- normalized_prices: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\normalized_prices.png
- rolling_state: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\rolling_state.png
- stress_regimes: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\stress_regimes.png
- lead_lag_heatmap: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\lead_lag_heatmap.png
- transition_heatmap: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\transition_heatmap.png
- forecast_actual: C:\VSCodeFolders\PortfolioMangement_V2\docs\final_project\research_lab\generated\dfc04a9e8d78431e\charts\forecast_actual.png
