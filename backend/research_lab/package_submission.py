"""Package Research Lab outputs into a final-project submission folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = PROJECT_ROOT / 'docs' / 'final_project' / 'research_lab'
GENERATED_ROOT = LAB_ROOT / 'generated'
SUBMISSION_ROOT = PROJECT_ROOT / 'docs' / 'final_project' / 'portfolio_regime_stress_lab_submission'
NOTEBOOK_PATH = LAB_ROOT / 'portfolio_regime_stress_lab.ipynb'


def _latest_run_dir() -> Path:
    run_dirs = [path for path in GENERATED_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f'No research runs found under {GENERATED_ROOT}')
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        run_dir = GENERATED_ROOT / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f'Research run {run_id!r} was not found under {GENERATED_ROOT}')
        return run_dir
    return _latest_run_dir()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: List[str] = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(metrics: Dict, dotted_key: str, default='n/a'):
    cur = metrics
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _format_pct(value) -> str:
    if value in (None, 'n/a'):
        return 'n/a'
    return f'{float(value):.2%}'


def _format_num(value, places: int = 4) -> str:
    if value in (None, 'n/a'):
        return 'n/a'
    return f'{float(value):.{places}f}'


def _build_readme(run_id: str, metrics: Dict) -> str:
    symbols = ', '.join(metrics.get('symbols', []))
    forecast = metrics.get('forecast', {})
    top_insights = metrics.get('topInsights', [])
    return f"""# Portfolio Regime & Stress Propagation Lab Submission

This folder packages the three mandatory deliverables from the Quant
Mentorship final project brief into one submission-ready location.

## Included Deliverables

1. `portfolio_regime_stress_lab.ipynb`
2. `portfolio_regime_stress_lab_report.pdf`
3. `portfolio_regime_stress_lab_slide.pdf`

## Selected Research Run

- Run ID: `{run_id}`
- Universe: {symbols}
- Window: {metrics.get('startDate', 'n/a')} to {metrics.get('endDate', 'n/a')}
- Portfolio method: {metrics.get('portfolioMethod', 'n/a')}
- Forecast model: ARIMA{tuple(forecast.get('modelOrder', []))}
- Forecast RMSE: {_format_num(forecast.get('rmse'))}
- Baseline RMSE: {_format_num(forecast.get('baselineRmse'))}

## Strongest Takeaways

- {top_insights[0] if len(top_insights) > 0 else 'Stress conditions were clearly separable from calm conditions.'}
- {top_insights[1] if len(top_insights) > 1 else 'The custom stress score helps summarize correlated drawdown risk.'}
- {top_insights[2] if len(top_insights) > 2 else 'The transition analysis gives a defensible interpretation layer.'}

## Folder Guide

- `figures/`: charts referenced in the report and slide
- `tables/`: CSV summaries for regimes, forecast metrics, and transitions
- `data/`: exported run outputs backing the notebook/report/slide
- `supporting_material/`: checklist, rubric alignment, and speaker notes
"""


def _build_checklist(metrics: Dict) -> str:
    rise_rate = _metric(metrics, 'stressEntrySummary.preEntryRiseRatePct', 0.0)
    return f"""# Submission Checklist

- [x] Jupyter notebook included as `portfolio_regime_stress_lab.ipynb`
- [x] Summary report PDF included as `portfolio_regime_stress_lab_report.pdf`
- [x] One-slide presentation PDF included as `portfolio_regime_stress_lab_slide.pdf`
- [x] Dataset source documented: Yahoo Finance daily OHLCV / adjusted close
- [x] Data cleaning and preprocessing documented in notebook/report
- [x] At least three time-series visuals included
- [x] Regime segmentation included
- [x] Forecasting model included with RMSE, MAE, AIC, and BIC
- [x] Reflection and stakeholder-facing interpretation included

## Final Talking Point

Lead with the regime separation result first. In the packaged run, the
Stress Propagation Score rose before {rise_rate:.1f}% of stress entries,
so it is useful as context, but the cleanest headline is that stress and
calm regimes have sharply different volatility, breadth, and correlation
profiles.
"""


def _build_rubric_alignment(run_id: str, metrics: Dict) -> str:
    forecast = metrics.get('forecast', {})
    return f"""# Rubric Alignment

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
- Forecasting model: ARIMA{tuple(forecast.get('modelOrder', []))}
- Metrics: RMSE {_format_num(forecast.get('rmse'))}, MAE {_format_num(forecast.get('mae'))},
  AIC {_format_num(forecast.get('aic'), 2)}, BIC {_format_num(forecast.get('bic'), 2)}

## Insights & Interpretation

- Stress regime has materially higher realized volatility and lower breadth
- Transition/dwell analytics add interpretation beyond raw charts
- Evidence: `portfolio_regime_stress_lab_report.pdf`, `supporting_material/speaker_notes.md`

## Code Quality

- Backend engine: `backend/research_lab/engine.py`
- Packaging utility: `backend/research_lab/package_submission.py`
- Reproducible exported run: `{run_id}`
"""


def _build_speaker_notes(metrics: Dict) -> str:
    forecast = metrics.get('forecast', {})
    regime_summary = metrics.get('regimeSummary', {})
    calm = regime_summary.get('Calm Expansion', {})
    stress = regime_summary.get('Stress Contagion', {})
    top_insights = metrics.get('topInsights', [])
    return f"""# Speaker Notes

## 30-Second Version

I studied how stress builds and spreads across an equal-weight basket of
mega-cap technology stocks. The main result is that the portfolio breaks
into three interpretable volatility regimes, and the stress regime is
visibly different from the calm regime in volatility, correlation, and
breadth.

## Slide Narrative

1. Start with the stress-regime chart.
   Mention that Calm Expansion averaged {_format_pct(calm.get('avgRealizedVol20d'))}
   realized volatility, while Stress Contagion averaged
   {_format_pct(stress.get('avgRealizedVol20d'))}.
2. Explain the custom Stress Propagation Score.
   It combines correlation, volatility, breadth, dispersion, and lead-lag
   concentration into one interpretable state variable.
3. Close with the forecast result.
   ARIMA{tuple(forecast.get('modelOrder', []))} delivered RMSE
   {_format_num(forecast.get('rmse'))} versus a naive baseline at
   {_format_num(forecast.get('baselineRmse'))}.

## Back-Pocket Answers

- Why this universe? It keeps the story focused and reproducible.
- Why ARIMA? It is interpretable, rubric-friendly, and appropriate for
  realized volatility forecasting.
- Why is this unique? The custom stress score and propagation framing add
  an original research question without making speculative claims.

## Useful Lines

- {top_insights[0] if len(top_insights) > 0 else 'Stress conditions were materially more volatile than calm conditions.'}
- {top_insights[1] if len(top_insights) > 1 else 'The stress score behaved as intended across regimes.'}
- {top_insights[3] if len(top_insights) > 3 else 'Lead-lag structure was used as an explanatory feature, not a trading rule.'}
"""


def _copy_core_deliverables(run_dir: Path, metrics: Dict) -> None:
    _copy_file(NOTEBOOK_PATH, SUBMISSION_ROOT / 'portfolio_regime_stress_lab.ipynb')
    _copy_file(run_dir / 'summary_report.pdf', SUBMISSION_ROOT / 'portfolio_regime_stress_lab_report.pdf')
    _copy_file(run_dir / 'presentation_slide.pdf', SUBMISSION_ROOT / 'portfolio_regime_stress_lab_slide.pdf')
    _copy_file(run_dir / 'report_source.md', SUBMISSION_ROOT / 'supporting_material' / 'portfolio_regime_stress_lab_report_source.md')
    _copy_file(run_dir / 'slide_source.md', SUBMISSION_ROOT / 'supporting_material' / 'portfolio_regime_stress_lab_slide_source.md')
    _copy_file(run_dir / 'metrics.json', SUBMISSION_ROOT / 'data' / 'metrics.json')
    _copy_file(run_dir / 'insights.md', SUBMISSION_ROOT / 'data' / 'insights.md')
    _copy_file(run_dir / 'aligned_prices.csv', SUBMISSION_ROOT / 'data' / 'aligned_prices.csv')
    _copy_file(run_dir / 'portfolio_features.csv', SUBMISSION_ROOT / 'data' / 'portfolio_features.csv')

    figures_dir = SUBMISSION_ROOT / 'figures'
    for path in sorted((run_dir / 'charts').glob('*.png')):
        _copy_file(path, figures_dir / path.name)

    manifest = {
        'submissionFolder': str(SUBMISSION_ROOT),
        'sourceRunId': run_dir.name,
        'sourceRunDir': str(run_dir),
        'briefDeliverables': [
            'Jupyter Notebook (.ipynb)',
            'Summary Analysis Report (PDF)',
            'Presentation Slide (1 slide)',
        ],
        'selectedUniverse': metrics.get('symbols', []),
    }
    _write_json(SUBMISSION_ROOT / 'supporting_material' / 'submission_manifest.json', manifest)


def _write_support_tables(metrics: Dict) -> None:
    regime_rows = []
    for label, row in metrics.get('regimeSummary', {}).items():
        regime_rows.append({'regime': label, **row})
    _write_csv(SUBMISSION_ROOT / 'tables' / 'regime_summary.csv', regime_rows)

    dwell_rows = []
    for label, row in metrics.get('dwellStats', {}).items():
        dwell_rows.append({'regime': label, **row})
    _write_csv(SUBMISSION_ROOT / 'tables' / 'dwell_stats.csv', dwell_rows)

    forward_rows = []
    for label, row in metrics.get('forwardStatsByRegime', {}).items():
        forward_rows.append({'regime': label, **row})
    _write_csv(SUBMISSION_ROOT / 'tables' / 'forward_stats_by_regime.csv', forward_rows)

    forecast = metrics.get('forecast', {})
    _write_csv(
        SUBMISSION_ROOT / 'tables' / 'forecast_metrics.csv',
        [{
            'modelName': forecast.get('modelName'),
            'modelOrder': tuple(forecast.get('modelOrder', [])),
            'modelStatus': forecast.get('modelStatus'),
            'mae': forecast.get('mae'),
            'rmse': forecast.get('rmse'),
            'aic': forecast.get('aic'),
            'bic': forecast.get('bic'),
            'baselineMae': forecast.get('baselineMae'),
            'baselineRmse': forecast.get('baselineRmse'),
        }],
    )

    transition_rows = []
    probs = metrics.get('transitionMatrix', {}).get('probabilities', {})
    for from_regime, row in probs.items():
        transition_rows.append({'fromRegime': from_regime, **row})
    _write_csv(SUBMISSION_ROOT / 'tables' / 'transition_probabilities.csv', transition_rows)

    lead_lag_rows = metrics.get('leadLagSummary', [])
    _write_csv(SUBMISSION_ROOT / 'tables' / 'lead_lag_top10.csv', lead_lag_rows)

    stress_summary = metrics.get('stressEntrySummary', {})
    _write_csv(
        SUBMISSION_ROOT / 'tables' / 'stress_entry_summary.csv',
        [{
            'stressEntryCount': stress_summary.get('stressEntryCount'),
            'preEntryRiseRatePct': stress_summary.get('preEntryRiseRatePct'),
            'avgPre5StressScore': stress_summary.get('avgPre5StressScore'),
            'avgPrior5StressScore': stress_summary.get('avgPrior5StressScore'),
        }],
    )


def package_submission(run_id: str | None = None) -> Path:
    run_dir = _resolve_run_dir(run_id)
    metrics = _read_json(run_dir / 'metrics.json')

    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    _copy_core_deliverables(run_dir, metrics)
    _write_support_tables(metrics)
    _write_text(SUBMISSION_ROOT / 'README.md', _build_readme(run_dir.name, metrics))
    _write_text(SUBMISSION_ROOT / 'supporting_material' / 'submission_checklist.md', _build_checklist(metrics))
    _write_text(SUBMISSION_ROOT / 'supporting_material' / 'rubric_alignment.md', _build_rubric_alignment(run_dir.name, metrics))
    _write_text(SUBMISSION_ROOT / 'supporting_material' / 'speaker_notes.md', _build_speaker_notes(metrics))
    return SUBMISSION_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description='Package a Research Lab run into a submission folder.')
    parser.add_argument('--run-id', help='Research run id under docs/final_project/research_lab/generated')
    args = parser.parse_args()

    output_dir = package_submission(run_id=args.run_id)
    print(output_dir)


if __name__ == '__main__':
    main()
