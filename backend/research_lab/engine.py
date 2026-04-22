"""
ResearchLab - Portfolio Regime & Stress Propagation Study
=========================================================

Builds a reproducible, class-ready time-series research workflow around a
concentrated six-stock tech portfolio.

Capabilities
------------
1. Fetch and align adjusted daily OHLCV data
2. Compute portfolio state features:
   - realized volatility
   - pairwise correlation
   - breadth above 50DMA
   - cross-sectional dispersion
   - lead-lag concentration
3. Build a custom Stress Propagation Score
4. Detect interpretable portfolio regimes with KMeans clustering
5. Run expanding-window ARIMA walk-forward forecasting on realized vol
6. Export charts, CSVs, markdown, PDF report, and slide artifacts
7. Persist run metadata + charts to Firestore for the frontend
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except Exception:
    ARIMA = None
    _HAS_STATSMODELS = False


DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
DEFAULT_START_DATE = '2021-04-18'
DEFAULT_FREQUENCY = '1d'
REGIME_LABELS = (
    'Calm Expansion',
    'Rotation / Transition',
    'Stress Contagion',
)
REGIME_COLORS = {
    'Calm Expansion': '#2ea043',
    'Rotation / Transition': '#d29922',
    'Stress Contagion': '#f85149',
}


def _now() -> datetime:
    return datetime.now()


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _series_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-12 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    mean = float(series.mean())
    return (series - mean) / std


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - pred)))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return {
            str(idx): {str(col): _jsonable(v) for col, v in row.items()}
            for idx, row in value.iterrows()
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class ForecastResult:
    model_name: str
    model_order: Tuple[int, int, int]
    model_status: str
    mae: float
    rmse: float
    aic: Optional[float]
    bic: Optional[float]
    baseline_mae: float
    baseline_rmse: float
    series: pd.DataFrame


class ResearchLab:
    """End-to-end research workflow for the portfolio stress study."""

    def __init__(
        self,
        dbClient=None,
        logFn=None,
        artifactsRoot: Optional[str] = None,
    ):
        self.db = dbClient
        self._log = logFn or print
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.projectRoot = base_dir
        self.docsRoot = _ensure_dir(os.path.join(base_dir, 'docs', 'final_project', 'research_lab'))
        self.generatedRoot = _ensure_dir(
            artifactsRoot or os.path.join(self.docsRoot, 'generated')
        )
        self.cacheRoot = _ensure_dir(os.path.join(base_dir, '.cache', 'yfinance'))
        try:
            yf.set_tz_cache_location(self.cacheRoot)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: Optional[List[str]] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        frequency: str = DEFAULT_FREQUENCY,
        portfolioMethod: str = 'equal_weight',
        persist: bool = True,
        useSyntheticData: bool = False,
        verbose: bool = True,
    ) -> Dict:
        if frequency != '1d':
            raise ValueError('ResearchLab currently supports daily data only.')

        symbols = self._normalise_symbols(symbols or DEFAULT_SYMBOLS)
        startDate = startDate or DEFAULT_START_DATE
        endDate = endDate or _now().strftime('%Y-%m-%d')
        run_id = self._build_run_id(symbols, startDate, endDate, portfolioMethod, useSyntheticData)

        if verbose:
            self._log(f"[ResearchLab] Starting run {run_id} for {symbols}")

        if useSyntheticData:
            prices = self._build_synthetic_prices(symbols, startDate, endDate)
        else:
            prices = self._fetch_adjusted_close_prices(symbols, startDate, endDate)

        analysis = self._analyze(prices, symbols)
        artifact_paths, chart_payloads = self._export_artifacts(
            run_id=run_id,
            symbols=symbols,
            startDate=startDate,
            endDate=endDate,
            portfolioMethod=portfolioMethod,
            analysis=analysis,
        )

        run_doc = {
            'researchRunId': run_id,
            'timestamp': _now().isoformat(),
            'status': 'completed',
            'symbols': symbols,
            'startDate': startDate,
            'endDate': endDate,
            'frequency': frequency,
            'portfolioMethod': portfolioMethod,
            'regimeMethod': 'kmeans_3_cluster_rolling_state',
            'forecastTarget': 'portfolio_realized_vol_20d',
            'forecastModel': analysis['forecast'].model_name,
            'forecastModelStatus': analysis['forecast'].model_status,
            'forecastModelOrder': list(analysis['forecast'].model_order),
            'mae': analysis['forecast'].mae,
            'rmse': analysis['forecast'].rmse,
            'aic': analysis['forecast'].aic,
            'bic': analysis['forecast'].bic,
            'baselineMae': analysis['forecast'].baseline_mae,
            'baselineRmse': analysis['forecast'].baseline_rmse,
            'dataStart': analysis['dataStart'],
            'dataEnd': analysis['dataEnd'],
            'numObservations': analysis['numObservations'],
            'regimeSummary': analysis['regimeSummary'],
            'transitionMatrix': analysis['transitionMatrix'],
            'dwellStats': analysis['dwellStats'],
            'forwardStatsByRegime': analysis['forwardStatsByRegime'],
            'stressEntrySummary': analysis['stressEntrySummary'],
            'leadLagSummary': analysis['leadLagSummary'],
            'topInsights': analysis['topInsights'],
            'artifactPaths': artifact_paths,
        }

        if persist and self.db is not None:
            self._persist_run(run_doc, chart_payloads)

        result = {
            'message': 'Research lab completed',
            'researchRunId': run_id,
            'mae': analysis['forecast'].mae,
            'rmse': analysis['forecast'].rmse,
            'baselineMae': analysis['forecast'].baseline_mae,
            'baselineRmse': analysis['forecast'].baseline_rmse,
            'numCharts': len(chart_payloads),
            'artifactDir': artifact_paths['runDir'],
            'forecastModelStatus': analysis['forecast'].model_status,
        }
        if verbose:
            self._log(
                f"[ResearchLab] Completed run {run_id} - "
                f"MAE {result['mae']:.4f}, RMSE {result['rmse']:.4f}"
            )
        return result

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _normalise_symbols(self, symbols: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for sym in symbols:
            s = (sym or '').strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        if len(cleaned) < 2:
            raise ValueError('Need at least two symbols for portfolio research.')
        return cleaned[:6]

    def _build_run_id(
        self,
        symbols: List[str],
        startDate: str,
        endDate: str,
        portfolioMethod: str,
        useSyntheticData: bool,
    ) -> str:
        raw = f"{symbols}|{startDate}|{endDate}|{portfolioMethod}|{useSyntheticData}|{_now().isoformat()}"
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    def _fetch_adjusted_close_prices(
        self,
        symbols: List[str],
        startDate: str,
        endDate: str,
    ) -> pd.DataFrame:
        aligned = {}
        for sym in symbols:
            self._log(f"[ResearchLab] Fetching {sym} ({startDate} -> {endDate})")
            try:
                df = yf.Ticker(sym).history(
                    start=startDate,
                    end=endDate,
                    interval='1d',
                    auto_adjust=True,
                )
            except Exception as exc:
                raise ValueError(f'Failed to fetch {sym} from Yahoo Finance: {exc}') from exc
            if df is None or len(df) == 0:
                raise ValueError(f'No data returned for {sym}')
            df = df.rename(columns=str.lower)
            close = df.get('close')
            if close is None:
                raise ValueError(f'Close series missing for {sym}')
            close.index = pd.to_datetime(close.index).tz_localize(None)
            aligned[sym] = close.astype(float)

        prices = pd.DataFrame(aligned).sort_index().dropna(how='any')
        if len(prices) < 140:
            raise ValueError('Aligned dataset is too short for regime + ARIMA analysis.')
        return prices

    def _build_synthetic_prices(
        self,
        symbols: List[str],
        startDate: str,
        endDate: str,
    ) -> pd.DataFrame:
        idx = pd.date_range(start=startDate, end=endDate, freq='B')
        if len(idx) < 260:
            idx = pd.date_range(end=_now(), periods=320, freq='B')

        rng = np.random.default_rng(42)
        n = len(idx)
        base_returns = np.zeros((n, len(symbols)))
        symbol_bias = np.linspace(-0.0005, 0.0005, len(symbols))

        # Three stylized regimes keep the offline path deterministic while
        # still producing a narrative that mirrors the live research setup.
        for i in range(n):
            if i < n * 0.4:
                mu, sigma, corr = 0.0007, 0.009, 0.15
                dispersion_scale = 0.35
                jump = 0.0
            elif i < n * 0.75:
                mu, sigma, corr = 0.0001, 0.015, 0.35
                dispersion_scale = 1.00
                jump = 0.0
            else:
                mu, sigma, corr = -0.0011, 0.030, 0.82
                dispersion_scale = 0.25
                jump = rng.normal(-0.035, 0.010) if rng.random() < 0.10 else 0.0

            common_shock = np.sqrt(corr) * rng.normal(0.0, sigma)
            idio_shocks = rng.normal(
                loc=symbol_bias * dispersion_scale,
                scale=sigma * np.sqrt(max(1e-6, 1.0 - corr)),
                size=len(symbols),
            )
            base_returns[i] = mu + jump + common_shock + idio_shocks

        prices = pd.DataFrame(index=idx)
        for j, sym in enumerate(symbols):
            prices[sym] = 100 * np.exp(np.cumsum(base_returns[:, j]))
        return prices

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def _analyze(self, prices: pd.DataFrame, symbols: List[str]) -> Dict:
        log_returns = np.log(prices).diff().dropna()
        portfolio_returns = log_returns.mean(axis=1)
        portfolio_level = np.exp(portfolio_returns.cumsum())
        portfolio_drawdown = portfolio_level / portfolio_level.cummax() - 1

        breadth = (prices > prices.rolling(50).mean()).sum(axis=1) / len(symbols)
        dispersion = log_returns.std(axis=1)
        dispersion_20d = dispersion.rolling(20).mean()
        realized_vol_20d = portfolio_returns.rolling(20).std() * np.sqrt(252)
        avg_pairwise_corr_20d = self._rolling_average_pairwise_corr(log_returns, window=20)
        lead_lag_concentration = self._rolling_lead_lag_concentration(
            log_returns, window=20, max_lag=3
        )
        lead_lag_summary, lead_lag_matrix = self._detect_global_lead_lag(log_returns, max_lag=5)

        features = pd.DataFrame(index=log_returns.index)
        features['portfolio_log_return'] = portfolio_returns
        features['portfolio_level'] = portfolio_level.reindex(features.index)
        features['portfolio_drawdown'] = portfolio_drawdown.reindex(features.index)
        features['portfolio_realized_vol_20d'] = realized_vol_20d
        features['avg_pairwise_corr_20d'] = avg_pairwise_corr_20d
        features['breadth_above_50dma'] = breadth.reindex(features.index)
        features['cross_sectional_dispersion_20d'] = dispersion_20d
        features['portfolio_return_20d'] = portfolio_returns.rolling(20).sum()
        features['lead_lag_concentration'] = lead_lag_concentration

        stress_raw = (
            _series_zscore(features['avg_pairwise_corr_20d'].bfill())
            + _series_zscore(features['portfolio_realized_vol_20d'].bfill())
            + _series_zscore((1.0 - features['breadth_above_50dma']).bfill())
            + _series_zscore(features['cross_sectional_dispersion_20d'].bfill())
            + _series_zscore(features['lead_lag_concentration'].bfill())
        ) / 5.0
        features['stress_propagation_score'] = _series_zscore(stress_raw.fillna(0.0))

        forward_vol, forward_dd = self._forward_20d_stats(portfolio_returns)
        features['forward_vol_20d'] = forward_vol.reindex(features.index)
        features['forward_drawdown_20d'] = forward_dd.reindex(features.index)

        clustered = self._fit_regimes(features)
        features['regime_label'] = clustered['regime_series'].reindex(features.index)

        forecast = self._walk_forward_forecast(features['portfolio_realized_vol_20d'])
        transition_matrix = self._transition_matrix(features['regime_label'])
        dwell_stats = self._dwell_stats(features['regime_label'])
        forward_stats = self._forward_stats_by_regime(features)
        stress_entry_summary = self._stress_entry_summary(features)
        regime_summary = self._regime_summary(features)

        top_insights = self._build_top_insights(
            features=features,
            regime_summary=regime_summary,
            stress_entry_summary=stress_entry_summary,
            lead_lag_summary=lead_lag_summary,
            forecast=forecast,
        )

        return {
            'prices': prices,
            'features': features,
            'leadLagSummary': lead_lag_summary,
            'leadLagMatrix': lead_lag_matrix,
            'forecast': forecast,
            'transitionMatrix': transition_matrix,
            'dwellStats': dwell_stats,
            'forwardStatsByRegime': forward_stats,
            'stressEntrySummary': stress_entry_summary,
            'regimeSummary': regime_summary,
            'topInsights': top_insights,
            'dataStart': prices.index[0].date().isoformat(),
            'dataEnd': prices.index[-1].date().isoformat(),
            'numObservations': int(len(prices)),
        }

    def _rolling_average_pairwise_corr(self, returns: pd.DataFrame, window: int = 20) -> pd.Series:
        values = []
        for end in range(len(returns)):
            if end + 1 < window:
                values.append(np.nan)
                continue
            corr = returns.iloc[end + 1 - window:end + 1].corr()
            arr = corr.to_numpy()
            mask = ~np.eye(arr.shape[0], dtype=bool)
            values.append(float(np.nanmean(arr[mask])))
        return pd.Series(values, index=returns.index)

    def _detect_global_lead_lag(
        self,
        returns: pd.DataFrame,
        max_lag: int = 5,
    ) -> Tuple[List[Dict], pd.DataFrame]:
        symbols = list(returns.columns)
        matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)
        relations = []
        for leader in symbols:
            for follower in symbols:
                if leader == follower:
                    continue
                best_corr = 0.0
                best_lag = 1
                for lag in range(1, max_lag + 1):
                    corr = returns[leader].shift(lag).corr(returns[follower])
                    if pd.isna(corr):
                        continue
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                matrix.loc[leader, follower] = float(best_corr)
                relations.append({
                    'leader': leader,
                    'follower': follower,
                    'lagDays': int(best_lag),
                    'correlation': float(best_corr),
                    'absCorrelation': float(abs(best_corr)),
                    'confidence': float(min(1.0, abs(best_corr) / 0.5)),
                })
        relations.sort(key=lambda x: x['absCorrelation'], reverse=True)
        return relations[:10], matrix

    def _rolling_lead_lag_concentration(
        self,
        returns: pd.DataFrame,
        window: int = 20,
        max_lag: int = 3,
    ) -> pd.Series:
        symbols = list(returns.columns)
        values = []
        for end in range(len(returns)):
            if end + 1 < window + max_lag:
                values.append(np.nan)
                continue
            window_df = returns.iloc[end + 1 - (window + max_lag):end + 1]
            pair_strengths = []
            for leader in symbols:
                for follower in symbols:
                    if leader == follower:
                        continue
                    best_corr = 0.0
                    for lag in range(1, max_lag + 1):
                        corr = window_df[leader].shift(lag).corr(window_df[follower])
                        if pd.isna(corr):
                            continue
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                    pair_strengths.append(abs(best_corr))
            values.append(float(np.mean(pair_strengths)) if pair_strengths else np.nan)
        return pd.Series(values, index=returns.index)

    def _fit_regimes(self, features: pd.DataFrame) -> Dict:
        cols = [
            'portfolio_return_20d',
            'portfolio_realized_vol_20d',
            'avg_pairwise_corr_20d',
            'breadth_above_50dma',
            'cross_sectional_dispersion_20d',
            'stress_propagation_score',
        ]
        frame = features[cols].dropna().copy()
        if len(frame) < 90:
            raise ValueError('Not enough rows to fit portfolio regimes.')

        scaler = StandardScaler()
        scaled = scaler.fit_transform(frame)
        km = KMeans(n_clusters=3, n_init=25, random_state=42)
        cluster_ids = km.fit_predict(scaled)

        labeled = frame.copy()
        labeled['cluster'] = cluster_ids

        cluster_centers = labeled.groupby('cluster').agg({
            'portfolio_return_20d': 'mean',
            'portfolio_realized_vol_20d': 'mean',
            'avg_pairwise_corr_20d': 'mean',
            'breadth_above_50dma': 'mean',
            'cross_sectional_dispersion_20d': 'mean',
            'stress_propagation_score': 'mean',
        })

        severity_components = pd.DataFrame(index=cluster_centers.index)
        severity_components['stress'] = _series_zscore(cluster_centers['stress_propagation_score'])
        severity_components['vol'] = _series_zscore(cluster_centers['portfolio_realized_vol_20d'])
        severity_components['corr'] = _series_zscore(cluster_centers['avg_pairwise_corr_20d'])
        severity_components['negative_breadth'] = _series_zscore(
            1.0 - cluster_centers['breadth_above_50dma']
        )
        severity_components['dispersion'] = _series_zscore(
            cluster_centers['cross_sectional_dispersion_20d']
        )
        severity_components['negative_return'] = _series_zscore(
            -cluster_centers['portfolio_return_20d']
        )

        severity_score = (
            severity_components['stress']
            + severity_components['vol']
            + severity_components['corr']
            + severity_components['negative_breadth']
            + 0.75 * severity_components['dispersion']
            + 0.50 * severity_components['negative_return']
        )

        ordered_clusters = list(severity_score.sort_values().index)
        calm_cluster = ordered_clusters[0]
        rotation_cluster = ordered_clusters[1]
        stress_cluster = ordered_clusters[2]

        cluster_to_label = {
            calm_cluster: REGIME_LABELS[0],
            rotation_cluster: REGIME_LABELS[1],
            stress_cluster: REGIME_LABELS[2],
        }
        regime_series = pd.Series(cluster_ids, index=frame.index).map(cluster_to_label)
        return {'regime_series': regime_series}

    def _forward_20d_stats(self, portfolio_returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        idx = portfolio_returns.index
        forward_vol = pd.Series(np.nan, index=idx)
        forward_dd = pd.Series(np.nan, index=idx)
        values = portfolio_returns.to_numpy()

        for i in range(len(values) - 20):
            window = values[i + 1:i + 21]
            forward_vol.iloc[i] = float(np.std(window, ddof=0) * np.sqrt(252))
            cum = np.exp(np.cumsum(window))
            drawdown = cum / np.maximum.accumulate(cum) - 1.0
            forward_dd.iloc[i] = float(drawdown.min())
        return forward_vol, forward_dd

    def _transition_matrix(self, regime_series: pd.Series) -> Dict:
        series = regime_series.dropna()
        labels = list(REGIME_LABELS)
        counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
        for prev, nxt in zip(series.iloc[:-1], series.iloc[1:]):
            counts.loc[prev, nxt] += 1
        probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return {
            'counts': _jsonable(counts.round(0)),
            'probabilities': _jsonable(probs.round(4)),
        }

    def _dwell_stats(self, regime_series: pd.Series) -> Dict:
        series = regime_series.dropna()
        stats = {}
        if series.empty:
            return stats

        current = series.iloc[0]
        length = 1
        segments = {label: [] for label in REGIME_LABELS}
        for value in series.iloc[1:]:
            if value == current:
                length += 1
            else:
                segments[current].append(length)
                current = value
                length = 1
        segments[current].append(length)

        for label, lengths in segments.items():
            if lengths:
                stats[label] = {
                    'episodes': int(len(lengths)),
                    'avgDwellDays': float(np.mean(lengths)),
                    'maxDwellDays': int(np.max(lengths)),
                }
            else:
                stats[label] = {
                    'episodes': 0,
                    'avgDwellDays': 0.0,
                    'maxDwellDays': 0,
                }
        return stats

    def _forward_stats_by_regime(self, features: pd.DataFrame) -> Dict:
        frame = features.dropna(subset=['regime_label', 'forward_vol_20d', 'forward_drawdown_20d'])
        stats = {}
        for label in REGIME_LABELS:
            sub = frame[frame['regime_label'] == label]
            stats[label] = {
                'count': int(len(sub)),
                'avgForwardVol20d': _safe_float(sub['forward_vol_20d'].mean()),
                'medianForwardVol20d': _safe_float(sub['forward_vol_20d'].median()),
                'avgForwardDrawdown20d': _safe_float(sub['forward_drawdown_20d'].mean()),
                'worstForwardDrawdown20d': _safe_float(sub['forward_drawdown_20d'].min()),
            }
        return stats

    def _stress_entry_summary(self, features: pd.DataFrame) -> Dict:
        regime = features['regime_label']
        score = features['stress_propagation_score']
        entries = features.index[
            (regime == 'Stress Contagion') &
            (regime.shift(1) != 'Stress Contagion')
        ]

        comparisons = []
        for ts in entries:
            loc = features.index.get_loc(ts)
            if loc < 10:
                continue
            pre5 = score.iloc[max(0, loc - 5):loc].mean()
            prev10 = score.iloc[max(0, loc - 10):max(0, loc - 5)].mean()
            if pd.isna(pre5) or pd.isna(prev10):
                continue
            comparisons.append({
                'entryDate': ts.date().isoformat(),
                'pre5Mean': float(pre5),
                'prev5Mean': float(prev10),
                'isRising': bool(pre5 > prev10),
            })

        rise_rate = float(np.mean([c['isRising'] for c in comparisons]) * 100.0) if comparisons else 0.0
        avg_pre = float(np.mean([c['pre5Mean'] for c in comparisons])) if comparisons else 0.0
        avg_prev = float(np.mean([c['prev5Mean'] for c in comparisons])) if comparisons else 0.0

        return {
            'stressEntryCount': int(len(comparisons)),
            'preEntryRiseRatePct': rise_rate,
            'avgPre5StressScore': avg_pre,
            'avgPrior5StressScore': avg_prev,
            'entries': comparisons[:20],
        }

    def _regime_summary(self, features: pd.DataFrame) -> Dict:
        frame = features.dropna(subset=['regime_label'])
        total = max(len(frame), 1)
        summary = {}
        for label in REGIME_LABELS:
            sub = frame[frame['regime_label'] == label]
            summary[label] = {
                'count': int(len(sub)),
                'pctOfSample': float(len(sub) / total * 100.0),
                'avgStressScore': _safe_float(sub['stress_propagation_score'].mean()),
                'avgRealizedVol20d': _safe_float(sub['portfolio_realized_vol_20d'].mean()),
                'avgCorr20d': _safe_float(sub['avg_pairwise_corr_20d'].mean()),
                'avgBreadth': _safe_float(sub['breadth_above_50dma'].mean()),
            }
        return summary

    def _walk_forward_forecast(self, vol_series: pd.Series) -> ForecastResult:
        series = vol_series.dropna().astype(float)
        if len(series) < 140:
            empty = pd.DataFrame(columns=['actual', 'forecast', 'baseline'])
            return ForecastResult(
                model_name='ARIMA',
                model_order=(1, 0, 1),
                model_status='insufficient_data',
                mae=0.0,
                rmse=0.0,
                aic=None,
                bic=None,
                baseline_mae=0.0,
                baseline_rmse=0.0,
                series=empty,
            )

        test_size = min(252, max(90, len(series) // 3))
        train_end = len(series) - test_size
        test_index = series.index[train_end:]
        actual = series.iloc[train_end:].to_numpy()
        baseline = series.shift(1).reindex(test_index).to_numpy()

        if not _HAS_STATSMODELS:
            forecast_df = pd.DataFrame({
                'actual': actual,
                'forecast': baseline,
                'baseline': baseline,
            }, index=test_index).dropna()
            actual_arr = forecast_df['actual'].to_numpy()
            pred_arr = forecast_df['forecast'].to_numpy()
            return ForecastResult(
                model_name='ARIMA',
                model_order=(1, 0, 1),
                model_status='statsmodels_unavailable_baseline_used',
                mae=_mae(actual_arr, pred_arr),
                rmse=_rmse(actual_arr, pred_arr),
                aic=None,
                bic=None,
                baseline_mae=_mae(actual_arr, forecast_df['baseline'].to_numpy()),
                baseline_rmse=_rmse(actual_arr, forecast_df['baseline'].to_numpy()),
                series=forecast_df,
            )

        candidate_orders = [(1, 0, 0), (1, 0, 1), (2, 0, 1), (1, 1, 1)]
        selection_train = series.iloc[:train_end]
        best_order = (1, 0, 1)
        best_fit = None
        best_aic = float('inf')

        for order in candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit = ARIMA(selection_train, order=order).fit()
                if fit.aic < best_aic:
                    best_order = order
                    best_aic = float(fit.aic)
                    best_fit = fit
            except Exception:
                continue

        if best_fit is None:
            forecast_df = pd.DataFrame({
                'actual': actual,
                'forecast': baseline,
                'baseline': baseline,
            }, index=test_index).dropna()
            actual_arr = forecast_df['actual'].to_numpy()
            pred_arr = forecast_df['forecast'].to_numpy()
            return ForecastResult(
                model_name='ARIMA',
                model_order=best_order,
                model_status='arima_fit_failed_baseline_used',
                mae=_mae(actual_arr, pred_arr),
                rmse=_rmse(actual_arr, pred_arr),
                aic=None,
                bic=None,
                baseline_mae=_mae(actual_arr, forecast_df['baseline'].to_numpy()),
                baseline_rmse=_rmse(actual_arr, forecast_df['baseline'].to_numpy()),
                series=forecast_df,
            )

        preds = []
        pred_times = []
        for end in range(train_end, len(series)):
            train = series.iloc[:end]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit = ARIMA(train, order=best_order).fit()
                    next_pred = float(fit.forecast(steps=1).iloc[0])
            except Exception:
                next_pred = float(train.iloc[-1])
            preds.append(next_pred)
            pred_times.append(series.index[end])

        forecast_df = pd.DataFrame({
            'actual': series.loc[pred_times].to_numpy(),
            'forecast': np.asarray(preds, dtype=float),
            'baseline': series.shift(1).reindex(pred_times).to_numpy(),
        }, index=pred_times).dropna()
        actual_arr = forecast_df['actual'].to_numpy()
        pred_arr = forecast_df['forecast'].to_numpy()
        baseline_arr = forecast_df['baseline'].to_numpy()
        return ForecastResult(
            model_name='ARIMA',
            model_order=best_order,
            model_status='ok',
            mae=_mae(actual_arr, pred_arr),
            rmse=_rmse(actual_arr, pred_arr),
            aic=float(best_fit.aic) if best_fit is not None else None,
            bic=float(best_fit.bic) if best_fit is not None else None,
            baseline_mae=_mae(actual_arr, baseline_arr),
            baseline_rmse=_rmse(actual_arr, baseline_arr),
            series=forecast_df,
        )

    def _build_top_insights(
        self,
        features: pd.DataFrame,
        regime_summary: Dict,
        stress_entry_summary: Dict,
        lead_lag_summary: List[Dict],
        forecast: ForecastResult,
    ) -> List[str]:
        stress = regime_summary.get('Stress Contagion', {})
        calm = regime_summary.get('Calm Expansion', {})
        best_relation = lead_lag_summary[0] if lead_lag_summary else None
        improvement = forecast.baseline_rmse - forecast.rmse

        insights = [
            (
                f"Stress Contagion carried an average realized vol of "
                f"{stress.get('avgRealizedVol20d', 0):.2%} versus "
                f"{calm.get('avgRealizedVol20d', 0):.2%} in Calm Expansion."
            ),
            (
                f"The Stress Propagation Score was higher in Stress Contagion "
                f"({stress.get('avgStressScore', 0):+.2f}) than in Calm Expansion "
                f"({calm.get('avgStressScore', 0):+.2f})."
            ),
            (
                f"Before stress-regime entries, the score rose in "
                f"{stress_entry_summary.get('preEntryRiseRatePct', 0):.1f}% of observed transitions."
            ),
        ]

        if best_relation:
            insights.append(
                f"The strongest lead-lag relation was {best_relation['leader']} leading "
                f"{best_relation['follower']} by {best_relation['lagDays']} day(s) "
                f"(corr={best_relation['correlation']:+.2f})."
            )

        if forecast.model_status == 'ok':
            if abs(improvement) < 0.001:
                insights.append(
                    f"ARIMA{forecast.model_order} closely matched the naive volatility baseline "
                    f"(RMSE {forecast.rmse:.4f} versus {forecast.baseline_rmse:.4f})."
                )
            elif improvement > 0:
                insights.append(
                    f"ARIMA{forecast.model_order} improved on the naive volatility baseline "
                    f"by {improvement:.4f} RMSE "
                    f"({forecast.rmse:.4f} versus {forecast.baseline_rmse:.4f})."
                )
            else:
                insights.append(
                    f"ARIMA{forecast.model_order} trailed the naive volatility baseline "
                    f"by {abs(improvement):.4f} RMSE "
                    f"({forecast.rmse:.4f} versus {forecast.baseline_rmse:.4f})."
                )
        else:
            insights.append(
                f"ARIMA forecasting fell back to baseline mode ({forecast.model_status}); "
                f"install statsmodels to enable the full model path."
            )
        return insights

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_artifacts(
        self,
        run_id: str,
        symbols: List[str],
        startDate: str,
        endDate: str,
        portfolioMethod: str,
        analysis: Dict,
    ) -> Tuple[Dict, List[Dict]]:
        run_dir = _ensure_dir(os.path.join(self.generatedRoot, run_id))
        charts_dir = _ensure_dir(os.path.join(run_dir, 'charts'))

        aligned_prices_path = os.path.abspath(os.path.join(run_dir, 'aligned_prices.csv'))
        features_path = os.path.abspath(os.path.join(run_dir, 'portfolio_features.csv'))
        metrics_path = os.path.abspath(os.path.join(run_dir, 'metrics.json'))
        insights_path = os.path.abspath(os.path.join(run_dir, 'insights.md'))
        report_md_path = os.path.abspath(os.path.join(run_dir, 'report_source.md'))
        report_pdf_path = os.path.abspath(os.path.join(run_dir, 'summary_report.pdf'))
        slide_md_path = os.path.abspath(os.path.join(run_dir, 'slide_source.md'))
        slide_pdf_path = os.path.abspath(os.path.join(run_dir, 'presentation_slide.pdf'))
        notebook_path = os.path.abspath(os.path.join(self.docsRoot, 'portfolio_regime_stress_lab.ipynb'))

        analysis['prices'].to_csv(aligned_prices_path, index_label='date')
        analysis['features'].to_csv(features_path, index_label='date')

        metric_payload = {
            'symbols': symbols,
            'startDate': startDate,
            'endDate': endDate,
            'portfolioMethod': portfolioMethod,
            'regimeSummary': analysis['regimeSummary'],
            'transitionMatrix': analysis['transitionMatrix'],
            'dwellStats': analysis['dwellStats'],
            'forwardStatsByRegime': analysis['forwardStatsByRegime'],
            'stressEntrySummary': analysis['stressEntrySummary'],
            'leadLagSummary': analysis['leadLagSummary'],
            'forecast': {
                'modelName': analysis['forecast'].model_name,
                'modelOrder': list(analysis['forecast'].model_order),
                'modelStatus': analysis['forecast'].model_status,
                'mae': analysis['forecast'].mae,
                'rmse': analysis['forecast'].rmse,
                'aic': analysis['forecast'].aic,
                'bic': analysis['forecast'].bic,
                'baselineMae': analysis['forecast'].baseline_mae,
                'baselineRmse': analysis['forecast'].baseline_rmse,
            },
            'topInsights': analysis['topInsights'],
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(_jsonable(metric_payload), f, indent=2)

        insights_md = self._build_insights_markdown(analysis['topInsights'])
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write(insights_md)

        chart_specs = [
            ('normalized_prices', 'Normalized Price Paths', self._plot_normalized_prices),
            ('rolling_state', 'Rolling State Metrics', self._plot_state_small_multiples),
            ('stress_regimes', 'Stress Propagation Score by Regime', self._plot_stress_regimes),
            ('lead_lag_heatmap', 'Lead-Lag Heatmap', self._plot_lead_lag_heatmap),
            ('transition_heatmap', 'Regime Transition Heatmap', self._plot_transition_heatmap),
            ('forecast_actual', 'Forecast vs Actual Realized Volatility', self._plot_forecast_vs_actual),
        ]

        chart_payloads = []
        chart_paths = {}
        for chart_type, label, fn in chart_specs:
            chart_path = os.path.abspath(os.path.join(charts_dir, f'{chart_type}.png'))
            fn(analysis, chart_path)
            chart_paths[chart_type] = chart_path
            chart_payloads.append({
                'researchRunId': run_id,
                'chartType': chart_type,
                'label': label,
                'imageData': self._image_to_data_uri(chart_path),
                'createdAt': _now().isoformat(),
            })

        report_md = self._build_report_source(
            symbols=symbols,
            startDate=startDate,
            endDate=endDate,
            analysis=analysis,
            artifact_paths=chart_paths,
        )
        with open(report_md_path, 'w', encoding='utf-8') as f:
            f.write(report_md)

        slide_md = self._build_slide_source(analysis)
        with open(slide_md_path, 'w', encoding='utf-8') as f:
            f.write(slide_md)

        self._build_report_pdf(report_pdf_path, analysis, chart_paths, symbols, startDate, endDate)
        self._build_slide_pdf(slide_pdf_path, analysis, chart_paths, symbols)

        artifact_paths = {
            'runDir': os.path.abspath(run_dir),
            'alignedPricesCsv': aligned_prices_path,
            'portfolioFeaturesCsv': features_path,
            'metricsJson': metrics_path,
            'insightsMarkdown': insights_path,
            'reportSource': report_md_path,
            'reportPdf': report_pdf_path,
            'slideSource': slide_md_path,
            'slidePdf': slide_pdf_path,
            'notebook': notebook_path,
            'charts': chart_paths,
        }
        return artifact_paths, chart_payloads

    def _build_insights_markdown(self, insights: List[str]) -> str:
        lines = ['# Top Insights', '']
        for item in insights:
            lines.append(f'- {item}')
        lines.append('')
        return '\n'.join(lines)

    def _build_report_source(
        self,
        symbols: List[str],
        startDate: str,
        endDate: str,
        analysis: Dict,
        artifact_paths: Dict[str, str],
    ) -> str:
        forecast = analysis['forecast']
        lines = [
            '# Portfolio Regime & Stress Propagation Lab',
            '',
            '## Dataset',
            f'- Universe: {", ".join(symbols)}',
            f'- Window: {startDate} to {endDate}',
            '- Frequency: Daily',
            '- Portfolio construction: Equal-weight synthetic portfolio',
            '',
            '## Methods',
            '- Rolling time-series features: realized volatility, pairwise correlation, breadth, dispersion, lead-lag concentration',
            '- Stress Propagation Score: standardized blend of five state variables',
            '- Regime detection: 3-cluster KMeans on rolling state features',
            f'- Forecast model: ARIMA{forecast.model_order} on 20-day realized volatility',
            '',
            '## Key Findings',
        ]
        lines.extend(f'- {insight}' for insight in analysis['topInsights'])
        lines.extend([
            '',
            '## Exported Visuals',
        ])
        for label, path in artifact_paths.items():
            lines.append(f'- {label}: {path}')
        lines.append('')
        return '\n'.join(lines)

    def _build_slide_source(self, analysis: Dict) -> str:
        best_story = 'Stress Propagation Score' if (
            analysis['stressEntrySummary'].get('preEntryRiseRatePct', 0.0) >= 55.0
        ) else 'ARIMA Volatility Forecast'
        lines = [
            '# Final Project Slide',
            '',
            f'## Hero Result: {best_story}',
            '',
            '### Three Talking Points',
        ]
        lines.extend(f'- {insight}' for insight in analysis['topInsights'][:3])
        lines.append('')
        return '\n'.join(lines)

    def _build_report_pdf(
        self,
        output_path: str,
        analysis: Dict,
        chart_paths: Dict[str, str],
        symbols: List[str],
        startDate: str,
        endDate: str,
    ):
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            fig.text(0.05, 0.93, 'Portfolio Regime & Stress Propagation Lab', fontsize=22, fontweight='bold')
            fig.text(0.05, 0.89, f'Universe: {", ".join(symbols)}', fontsize=12)
            fig.text(0.05, 0.86, f'Window: {startDate} to {endDate}', fontsize=12)
            fig.text(0.05, 0.83, 'Method: equal-weight daily portfolio, regime clustering, ARIMA vol forecast', fontsize=12)

            bullet_font = 10.5
            bullet_spacing = 1.28
            line_step = (bullet_font / 72.0) * bullet_spacing / fig.get_figheight()
            y = 0.75
            for insight in analysis['topInsights']:
                wrapped = textwrap.fill(
                    insight,
                    width=96,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                fig.text(0.07, y, u'\u2022 ' + wrapped,
                         fontsize=bullet_font, va='top', linespacing=bullet_spacing)
                line_count = wrapped.count('\n') + 1
                y -= max(0.058, line_count * line_step + 0.020)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5))
            axes = [
                fig.add_axes([0.06, 0.52, 0.40, 0.34]),
                fig.add_axes([0.54, 0.52, 0.40, 0.34]),
                fig.add_axes([0.06, 0.08, 0.40, 0.34]),
                fig.add_axes([0.54, 0.08, 0.40, 0.34]),
            ]
            selected = [
                chart_paths['normalized_prices'],
                chart_paths['stress_regimes'],
                chart_paths['transition_heatmap'],
                chart_paths['forecast_actual'],
            ]
            titles = [
                'Normalized Price Paths',
                'Stress Propagation Score',
                'Regime Transition Heatmap',
                'Forecast vs Actual Vol',
            ]
            for ax, path, title in zip(axes, selected, titles):
                ax.imshow(mpimg.imread(path))
                ax.set_title(title, fontsize=11)
                ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _build_slide_pdf(
        self,
        output_path: str,
        analysis: Dict,
        chart_paths: Dict[str, str],
        symbols: List[str],
    ):
        spotlight = (
            chart_paths['stress_regimes']
            if analysis['stressEntrySummary'].get('preEntryRiseRatePct', 0.0) >= 55.0
            else chart_paths['forecast_actual']
        )
        title = (
            'Stress Score Rose Ahead of Stress Regimes'
            if spotlight == chart_paths['stress_regimes']
            else 'ARIMA Tracked Portfolio Volatility Regimes'
        )

        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(13.333, 7.5))
            fig.patch.set_facecolor('white')
            fig.text(0.05, 0.92, title, fontsize=24, fontweight='bold')
            fig.text(0.05, 0.88, f'Universe: {", ".join(symbols)}', fontsize=12)

            ax = fig.add_axes([0.05, 0.14, 0.58, 0.66])
            ax.imshow(mpimg.imread(spotlight))
            ax.axis('off')

            bullet_font = 10.5
            bullet_spacing = 1.24
            line_step = (bullet_font / 72.0) * bullet_spacing / fig.get_figheight()

            y = 0.73
            fig.text(0.68, 0.80, 'Key Takeaways', fontsize=16, fontweight='bold')
            for insight in analysis['topInsights'][:3]:
                wrapped = textwrap.fill(
                    insight,
                    width=40,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                fig.text(0.68, y, u'\u2022 ' + wrapped,
                         fontsize=bullet_font, va='top', linespacing=bullet_spacing)
                line_count = wrapped.count('\n') + 1
                y -= max(0.115, line_count * line_step + 0.040)
            fig.text(0.68, 0.15, 'Daily equal-weight portfolio\nRegime clustering + ARIMA forecasting', fontsize=11)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_normalized_prices(self, analysis: Dict, output_path: str):
        prices = analysis['prices']
        normalized = prices / prices.iloc[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in normalized.columns:
            ax.plot(normalized.index, normalized[col], lw=1.8, label=col)
        ax.set_title('Normalized Price Paths')
        ax.set_ylabel('Index (Start = 1.0)')
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_state_small_multiples(self, analysis: Dict, output_path: str):
        features = analysis['features']
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        plots = [
            ('portfolio_realized_vol_20d', 'Portfolio Realized Vol (20d)', '#58a6ff'),
            ('avg_pairwise_corr_20d', 'Average Pairwise Correlation (20d)', '#bc8cff'),
            ('breadth_above_50dma', 'Breadth Above 50DMA', '#3fb950'),
            ('cross_sectional_dispersion_20d', 'Cross-Sectional Dispersion (20d)', '#f0883e'),
        ]
        for ax, (col, title, color) in zip(axes, plots):
            ax.plot(features.index, features[col], color=color, lw=1.7)
            ax.set_title(title, loc='left', fontsize=11)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_stress_regimes(self, analysis: Dict, output_path: str):
        features = analysis['features'].dropna(subset=['stress_propagation_score'])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(features.index, features['stress_propagation_score'], color='#f85149', lw=2.0, label='Stress Propagation Score')

        regime_series = features['regime_label'].ffill()
        start = None
        current = None
        for idx, label in regime_series.items():
            if current is None:
                current = label
                start = idx
                continue
            if label != current:
                ax.axvspan(start, idx, color=REGIME_COLORS.get(current, '#8b949e'), alpha=0.08)
                start = idx
                current = label
        if current is not None:
            ax.axvspan(start, regime_series.index[-1], color=REGIME_COLORS.get(current, '#8b949e'), alpha=0.08)

        ax.axhline(0.0, color='#8b949e', lw=1.0, alpha=0.6)
        ax.set_title('Stress Propagation Score with Regime Shading')
        ax.set_ylabel('Standardized Score')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_lead_lag_heatmap(self, analysis: Dict, output_path: str):
        matrix = analysis['leadLagMatrix']
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(matrix.to_numpy(), cmap='coolwarm', vmin=-0.4, vmax=0.4)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(matrix.index)
        ax.set_title('Lead-Lag Heatmap (Leader on Y, Follower on X)')
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                if i == j:
                    continue
                ax.text(j, i, f"{matrix.iloc[i, j]:+.2f}", ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_transition_heatmap(self, analysis: Dict, output_path: str):
        probs = pd.DataFrame(analysis['transitionMatrix']['probabilities']).T
        probs = probs.reindex(index=REGIME_LABELS, columns=REGIME_LABELS).fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(probs.to_numpy(), cmap='Blues', vmin=0.0, vmax=max(0.5, float(probs.to_numpy().max())))
        ax.set_xticks(range(len(REGIME_LABELS)))
        ax.set_yticks(range(len(REGIME_LABELS)))
        ax.set_xticklabels(REGIME_LABELS, rotation=25, ha='right')
        ax.set_yticklabels(REGIME_LABELS)
        ax.set_title('Regime Transition Probability Heatmap')
        for i in range(len(REGIME_LABELS)):
            for j in range(len(REGIME_LABELS)):
                ax.text(j, i, f"{probs.iloc[i, j]:.2f}", ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_forecast_vs_actual(self, analysis: Dict, output_path: str):
        forecast_df = analysis['forecast'].series
        fig, ax = plt.subplots(figsize=(12, 6))
        if forecast_df.empty:
            ax.text(0.5, 0.5, 'Forecast output unavailable', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot(forecast_df.index, forecast_df['actual'], label='Actual', color='#58a6ff', lw=1.8)
            ax.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='#f0883e', lw=1.5)
            ax.plot(forecast_df.index, forecast_df['baseline'], label='Naive Baseline', color='#8b949e', lw=1.2, ls='--')
            ax.legend(frameon=False)
        ax.set_title('Forecast vs Actual Portfolio Realized Volatility')
        ax.set_ylabel('Annualized Volatility')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _image_to_data_uri(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')
        return f"data:image/png;base64,{encoded}"

    # ------------------------------------------------------------------
    # Firestore persistence
    # ------------------------------------------------------------------

    def _persist_run(self, run_doc: Dict, chart_payloads: List[Dict]):
        self.db.collection('research_runs').document(run_doc['researchRunId']).set(_jsonable(run_doc))
        charts_col = self.db.collection('research_charts')
        for payload in chart_payloads:
            charts_col.add(_jsonable(payload))


if __name__ == '__main__':
    lab = ResearchLab(dbClient=None)
    result = lab.run(persist=False, useSyntheticData=True, verbose=True)
    print(json.dumps(result, indent=2))
