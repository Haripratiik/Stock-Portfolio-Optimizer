# Portfolio Regime & Stress Propagation Lab Submission

This folder packages the three mandatory deliverables from the Quant
Mentorship final project brief into one submission-ready location.

## Included Deliverables

1. `portfolio_regime_stress_lab.ipynb`
2. `portfolio_regime_stress_lab_report.pdf`
3. `portfolio_regime_stress_lab_slide.pdf`

## Selected Research Run

- Run ID: `dfc04a9e8d78431e`
- Universe: AAPL, MSFT, GOOGL, AMZN, META, NVDA
- Window: 2021-04-18 to 2026-04-18
- Portfolio method: equal_weight
- Forecast model: ARIMA(2, 0, 1)
- Forecast RMSE: 0.0269
- Baseline RMSE: 0.0271

## Strongest Takeaways

- Stress Contagion carried an average realized vol of 45.90% versus 7.65% in Calm Expansion.
- The Stress Propagation Score was higher in Stress Contagion (+1.49) than in Calm Expansion (-0.93).
- Before stress-regime entries, the score rose in 35.7% of observed transitions.

## Folder Guide

- `figures/`: charts referenced in the report and slide
- `tables/`: CSV summaries for regimes, forecast metrics, and transitions
- `data/`: exported run outputs backing the notebook/report/slide
- `supporting_material/`: checklist, rubric alignment, and speaker notes
