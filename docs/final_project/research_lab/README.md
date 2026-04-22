# Quant Final Project Deliverables

This folder contains the reusable deliverable scaffold for the
Portfolio Regime & Stress Propagation Lab.

## Core Files

- `portfolio_regime_stress_lab.ipynb`
  - Jupyter notebook scaffold for the class submission.
- `generated/<research_run_id>/`
  - Per-run research artifacts exported by `backend/research_lab/engine.py`.
  - Includes:
    - `aligned_prices.csv`
    - `portfolio_features.csv`
    - `metrics.json`
    - `insights.md`
    - `report_source.md`
    - `summary_report.pdf`
    - `slide_source.md`
    - `presentation_slide.pdf`
    - `charts/*.png`

## Recommended Submission Workflow

1. Queue and complete a `research_lab` run from the app.
2. Use the newest folder under `generated/`.
3. Open `portfolio_regime_stress_lab.ipynb` and bind its narrative to that run's outputs.
4. Submit:
   - the notebook,
   - the generated report PDF,
   - the generated one-slide PDF.
