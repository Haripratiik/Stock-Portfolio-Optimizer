# Research Lab Subsystem

This package contains the portfolio regime and stress propagation research workflow.

## Purpose

It turns the existing trading platform into a research-capable module for:

- aligned portfolio time-series analysis,
- interpretable regime segmentation,
- stress propagation diagnostics,
- ARIMA volatility forecasting,
- class-ready deliverable generation,
- and submission-folder packaging via `package_submission.py`.

## Main Entry Point

- `engine.py`
  - `ResearchLab.run(...)`

## Design Notes

- The app queues research jobs through `run_commands.type = "research_lab"`.
- The LocalAgent executes the run and persists:
  - `research_runs`
  - `research_charts`
- The same backend run exports notebook/report/slide assets under:
  - `docs/final_project/research_lab/generated/<run_id>/`
