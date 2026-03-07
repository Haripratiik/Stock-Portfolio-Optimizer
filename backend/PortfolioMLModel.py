"""
PortfolioMLModel — Portfolio-Level Cross-Stock ML Model

Trained AFTER all per-stock StockMLModels, this model learns how stocks
within the portfolio interact and produces portfolio-wide signals:

Capabilities:
  1. Cross-stock correlation tracking   — rolling & regime-aware
  2. Lead-lag relationship discovery     — does stock A predict stock B?
  3. Portfolio risk-regime classification— low/normal/high/crisis
  4. Dynamic allocation recommendations — shift capital toward best opportunities
  5. Hedging signals                    — reduce exposure when systemic risk rises
  6. Mean-reversion detection           — buy laggards when corr is high
  7. Diversification monitoring         — warn when portfolio is over-concentrated

Architecture:
  - GradientBoosting classifiers for regime + hedge signals
  - GradientBoosting regressor for expected portfolio return
  - Correlation + lead-lag analysis via rolling statistics
  - Rule-based hedging overlay on top of ML signals
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import warnings

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from StockMLModel import StockMLModel, StockPrediction, TradingSignal, MarketRegime


# =============================================================================
# Enums & data classes
# =============================================================================

class RiskRegime(Enum):
    LOW_RISK  = 'LOW_RISK'
    NORMAL    = 'NORMAL'
    HIGH_RISK = 'HIGH_RISK'
    CRISIS    = 'CRISIS'


class HedgeAction(Enum):
    NONE        = 'NONE'           # full exposure
    REDUCE_10   = 'REDUCE_10'      # trim 10 % of equity exposure
    REDUCE_25   = 'REDUCE_25'      # trim 25 %
    REDUCE_50   = 'REDUCE_50'      # trim 50 %
    FULL_HEDGE  = 'FULL_HEDGE'     # go to cash / full hedge


@dataclass
class LeadLagRelation:
    """A detected lead-lag relationship between two stocks."""
    leader: str
    follower: str
    lagPeriods: int                 # how many periods the follower lags
    correlation: float             # correlation at the optimal lag
    confidence: float              # 0-1 reliability of the relationship


@dataclass
class AllocationAdjustment:
    """Suggested allocation change for a single stock."""
    symbol: str
    currentAllocation: float       # current fraction
    suggestedAllocation: float     # recommended fraction
    reason: str                    # human-readable justification


@dataclass
class PortfolioSignal:
    """
    Comprehensive portfolio-level signal produced each period.
    """
    timestamp: Optional[pd.Timestamp] = None
    riskRegime: RiskRegime = RiskRegime.NORMAL
    hedgeAction: HedgeAction = HedgeAction.NONE
    expectedPortfolioReturn: float = 0.0       # predicted forward return (%)
    portfolioVolatility: float = 0.0           # annualised vol estimate
    avgCrossCorrelation: float = 0.0           # mean pairwise correlation
    allocationAdjustments: List[AllocationAdjustment] = field(default_factory=list)
    leadLagSignals: List[LeadLagRelation] = field(default_factory=list)
    meanReversionOpportunities: List[str] = field(default_factory=list)
    confidence: float = 0.0                    # overall confidence 0-1


@dataclass
class PortfolioModelMetrics:
    """Training / evaluation metrics for the portfolio model."""
    regimeAccuracy: float = 0.0
    hedgeAccuracy: float = 0.0
    returnRMSE: float = 0.0
    trainSamples: int = 0
    testSamples: int = 0
    topFeatures: List[Tuple[str, float]] = field(default_factory=list)


# =============================================================================
# PortfolioMLModel
# =============================================================================

class PortfolioMLModel:
    """
    Portfolio-level ML model that operates on top of per-stock models.

    Lifecycle::

        pm = PortfolioMLModel(
            stockModels={'AAPL': model_aapl, 'GOOGL': model_googl, ...},
            allocations={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3},
        )
        pm.train(stockDataDict, forwardPeriods=10)
        signal = pm.predict(latestStockPredictions)
    """

    def __init__(self,
                 stockModels: Dict[str, StockMLModel],
                 allocations: Dict[str, float],
                 totalFund: float = 100_000):
        self.stockModels = stockModels
        self.allocations = allocations
        self.totalFund = totalFund
        self.symbols = sorted(stockModels.keys())

        # Models (trained later)
        self._regimeModel:  Optional[GradientBoostingClassifier] = None
        self._hedgeModel:   Optional[GradientBoostingClassifier] = None
        self._returnModel:  Optional[GradientBoostingRegressor]  = None
        self._scaler:       Optional[StandardScaler] = None

        self._featureNames: List[str] = []
        self._trained: bool = False
        self._metrics: Optional[PortfolioModelMetrics] = None

        # Cached analytics
        self._correlationMatrix: Optional[pd.DataFrame] = None
        self._leadLagRelations: List[LeadLagRelation] = []

        # Sentiment data: {symbol → date-indexed pd.Series in [-1, +1]}
        self._sentimentDataDict: Dict[str, pd.Series] = {}

        # Strategy features from StrategyEngine (injected externally)
        self._strategyFeatures: Dict[str, np.ndarray] = {}  # {sym: features}
        self._crossStockRuleFeatures: Optional[pd.DataFrame] = None

        # Stock metadata for sector-aware features
        self._stockMetadata: Dict[str, dict] = {}  # {sym: {sector, industry, ...}}

    # =====================================================================
    # Strategy / metadata injection
    # =====================================================================

    def setStrategyFeatures(self, strategyFeatures: Dict[str, np.ndarray]):
        """
        Inject per-symbol strategy signal features from StrategyEngine.
        ``strategyFeatures`` = {symbol: np.array of strategy signals}.
        """
        self._strategyFeatures = strategyFeatures

    def setCrossStockRuleFeatures(self, ruleFeatures: pd.DataFrame):
        """
        Inject cross-stock rule features DataFrame from
        ``StrategyEngine.featuriseCrossStockRules()``.
        """
        self._crossStockRuleFeatures = ruleFeatures

    def setStockMetadata(self, metaDict: Dict[str, dict]):
        """
        Inject stock metadata (sector, industry, etc.) for sector-relative
        features.  ``metaDict`` = {symbol: {sector, industry, ...}}.
        """
        self._stockMetadata = metaDict

    # =====================================================================
    # Public API
    # =====================================================================

    def train(self,
              stockDataDict: Dict[str, pd.DataFrame],
              forwardPeriods: int = 10,
              corrWindow: int = 60,
              testSize: float = 0.2,
              sentimentDataDict: Optional[Dict[str, pd.Series]] = None,
              verbose: bool = True) -> PortfolioModelMetrics:
        """
        Train the portfolio model on aligned multi-stock historical data.

        Args:
            stockDataDict:     {symbol: OHLCV DataFrame} — all should cover
                               the same date range (daily recommended).
            forwardPeriods:    Horizon for portfolio-return labels.
            corrWindow:        Rolling window for correlation features.
            testSize:          Held-out fraction.
            sentimentDataDict: Optional {symbol: pd.Series} of daily
                               sentiment scores from SentimentAnalyzer.
                               When provided, cross-stock sentiment features
                               are added to the portfolio feature matrix.
            verbose:           Print progress.
        """
        if sentimentDataDict:
            self._sentimentDataDict = sentimentDataDict
        if verbose:
            print(f"\n  [PortfolioMLModel] Training across "
                  f"{list(stockDataDict.keys())} ...")

        # ---- 1. Align all stock close prices to a common date index ----
        pricesDf, returnsDf = self._alignStockData(stockDataDict)

        # Auto-adapt corrWindow when data is limited so we don't lose
        # too many rows to NaN from rolling windows.
        availableRows = len(pricesDf) if pricesDf is not None else 0
        if availableRows > 0:
            # Need: corrWindow + forwardPeriods + 30 rows minimum
            # If tight, shrink corrWindow
            maxUsable = availableRows - forwardPeriods - 30
            if maxUsable < corrWindow and maxUsable >= 20:
                if verbose:
                    print(f"    Adapting corrWindow: {corrWindow} → {maxUsable} "
                          f"(only {availableRows} aligned rows)")
                corrWindow = maxUsable

        if pricesDf is None or len(pricesDf) < corrWindow + forwardPeriods + 20:
            if verbose:
                print(f"    Insufficient aligned data for portfolio model "
                      f"({availableRows} rows, need {corrWindow + forwardPeriods + 20})")
            return PortfolioModelMetrics()

        # ---- 2. Compute cross-stock analytics ---------------------------
        self._correlationMatrix = returnsDf.rolling(corrWindow).corr()
        self._leadLagRelations = self._detectLeadLag(returnsDf, maxLag=5)
        if verbose and self._leadLagRelations:
            print(f"    Detected {len(self._leadLagRelations)} lead-lag relations")
            for ll in self._leadLagRelations[:3]:
                print(f"      {ll.leader} → {ll.follower} (lag={ll.lagPeriods}, "
                      f"r={ll.correlation:.3f})")

        # ---- 3. Build feature matrix ------------------------------------
        X, yRegime, yHedge, yReturn, names = self._buildDataset(
            pricesDf, returnsDf, corrWindow, forwardPeriods,
            stockDataDict, self._sentimentDataDict
        )
        self._featureNames = names

        if len(X) < 35:
            if verbose:
                print(f"    Not enough aligned rows to train portfolio model ({len(X)} < 35)")
            return PortfolioModelMetrics()

        # ---- 4. Train models --------------------------------------------
        metrics = self._fitModels(X, yRegime, yHedge, yReturn, testSize, verbose)
        self._trained = True
        self._metrics = metrics
        return metrics

    def predict(self,
                stockPredictions: Dict[str, StockPrediction],
                stockDataDict: Dict[str, pd.DataFrame],
                corrWindow: int = 60
                ) -> PortfolioSignal:
        """
        Generate a portfolio-level signal using stock-level predictions
        and the latest market data.
        """
        if not self._trained:
            return PortfolioSignal()

        # Build a single-row feature vector from current state
        featureRow = self._buildCurrentFeatures(
            stockPredictions, stockDataDict, corrWindow
        )
        if featureRow is None:
            return PortfolioSignal()

        rowScaled = self._scaler.transform(featureRow.reshape(1, -1))

        # Regime
        regimeIdx = int(self._regimeModel.predict(rowScaled)[0])
        regimeMap = {0: RiskRegime.LOW_RISK, 1: RiskRegime.NORMAL,
                     2: RiskRegime.HIGH_RISK, 3: RiskRegime.CRISIS}
        regime = regimeMap.get(regimeIdx, RiskRegime.NORMAL)

        # Hedge
        hedgeIdx = int(self._hedgeModel.predict(rowScaled)[0])
        hedgeMap = {0: HedgeAction.NONE, 1: HedgeAction.REDUCE_10,
                    2: HedgeAction.REDUCE_25, 3: HedgeAction.REDUCE_50,
                    4: HedgeAction.FULL_HEDGE}
        hedge = hedgeMap.get(hedgeIdx, HedgeAction.NONE)

        # Expected portfolio return
        expReturn = float(self._returnModel.predict(rowScaled)[0])

        # Allocation adjustments (rule-based on stock predictions + regime)
        adjustments = self._computeAllocationAdjustments(
            stockPredictions, regime
        )

        # Mean-reversion opportunities
        meanRevOps = self._findMeanReversionOpportunities(
            stockPredictions, stockDataDict, corrWindow
        )

        # Compute portfolio volatility estimate
        portVol = self._estimatePortfolioVolatility(stockDataDict, corrWindow)

        # Average cross-correlation
        avgCorr = self._averageCrossCorrelation(stockDataDict, corrWindow)

        # Overall confidence
        regimeProba = self._regimeModel.predict_proba(rowScaled)[0]
        confidence = float(np.max(regimeProba))

        return PortfolioSignal(
            riskRegime=regime,
            hedgeAction=hedge,
            expectedPortfolioReturn=expReturn,
            portfolioVolatility=portVol,
            avgCrossCorrelation=avgCorr,
            allocationAdjustments=adjustments,
            leadLagSignals=self._leadLagRelations,
            meanReversionOpportunities=meanRevOps,
            confidence=confidence,
        )

    def getCorrelationMatrix(self,
                             stockDataDict: Dict[str, pd.DataFrame],
                             window: int = 60) -> pd.DataFrame:
        """Return the most recent rolling correlation matrix."""
        _, returnsDf = self._alignStockData(stockDataDict)
        if returnsDf is None or len(returnsDf) < window:
            return pd.DataFrame()
        return returnsDf.tail(window).corr()

    def getLeadLagRelations(self) -> List[LeadLagRelation]:
        return self._leadLagRelations

    def evaluate(self) -> Optional[PortfolioModelMetrics]:
        return self._metrics

    def getFeatureImportances(self, topN: int = 15) -> List[Tuple[str, float]]:
        if self._regimeModel is None:
            return []
        imp = self._regimeModel.feature_importances_
        ranked = sorted(zip(self._featureNames, imp), key=lambda x: x[1], reverse=True)
        return ranked[:topN]

    @property
    def isTrained(self) -> bool:
        return self._trained

    # =====================================================================
    # INTERNAL — data alignment
    # =====================================================================

    def _alignStockData(self, stockDataDict: Dict[str, pd.DataFrame]):
        """
        Align all stocks to a common date index (inner join on dates).
        Returns (pricesDf, returnsDf) where each column is a stock symbol.
        """
        closeFrames = {}
        for symbol, df in stockDataDict.items():
            if df is not None and 'close' in df.columns and len(df) > 0:
                closeFrames[symbol] = df['close']

        if len(closeFrames) < 2:
            return None, None

        pricesDf = pd.DataFrame(closeFrames).dropna()
        returnsDf = pricesDf.pct_change().dropna()
        return pricesDf, returnsDf

    # =====================================================================
    # INTERNAL — cross-stock analytics
    # =====================================================================

    def _detectLeadLag(self, returnsDf: pd.DataFrame,
                       maxLag: int = 5, minCorr: float = 0.15
                       ) -> List[LeadLagRelation]:
        """
        Detect lead-lag relationships by computing cross-correlations
        at various lags.

        For each pair (A, B), check if A's returns at time t correlate
        with B's returns at time t+lag.  If the correlation exceeds
        *minCorr*, record the relationship.
        """
        relations = []
        symbols = list(returnsDf.columns)

        for s1, s2 in combinations(symbols, 2):
            r1 = returnsDf[s1].values
            r2 = returnsDf[s2].values
            bestCorr = 0.0
            bestLag = 0
            bestLeader = s1

            for lag in range(1, maxLag + 1):
                # s1 leads s2
                if lag < len(r1):
                    c12 = np.corrcoef(r1[:-lag], r2[lag:])[0, 1]
                    if abs(c12) > abs(bestCorr):
                        bestCorr = c12
                        bestLag = lag
                        bestLeader = s1

                # s2 leads s1
                if lag < len(r2):
                    c21 = np.corrcoef(r2[:-lag], r1[lag:])[0, 1]
                    if abs(c21) > abs(bestCorr):
                        bestCorr = c21
                        bestLag = lag
                        bestLeader = s2

            if abs(bestCorr) >= minCorr:
                follower = s2 if bestLeader == s1 else s1
                confidence = min(1.0, abs(bestCorr) / 0.5)
                relations.append(LeadLagRelation(
                    leader=bestLeader, follower=follower,
                    lagPeriods=bestLag, correlation=bestCorr,
                    confidence=confidence,
                ))

        relations.sort(key=lambda x: abs(x.correlation), reverse=True)
        return relations

    # =====================================================================
    # INTERNAL — feature engineering
    # =====================================================================

    def _buildDataset(self, pricesDf, returnsDf, corrWindow,
                      forwardPeriods, stockDataDict,
                      sentimentDataDict: Optional[Dict[str, pd.Series]] = None):
        """
        Build the portfolio-level feature matrix + labels.

        Features per row (period t):
          - Per-stock: ret_1, ret_5, ret_10, vol_20, rsi_proxy
          - Cross-stock: pairwise rolling corr, avg corr, max corr, min corr
          - Portfolio: weighted return, rolling portfolio vol, drawdown
          - Lead-lag signals: predicted moves from leader stocks
          - Regime proxies: VIX-like (average vol), trend direction

        Labels:
          - yRegime:  risk regime classification (0-3)
          - yHedge:   hedge action classification (0-4)
          - yReturn:  forward portfolio return (regression)
        """
        symbols = list(pricesDf.columns)
        nSym = len(symbols)

        feats = pd.DataFrame(index=pricesDf.index)

        # ---- Per-stock features -----------------------------------------
        for sym in symbols:
            c = pricesDf[sym].values.astype(float)
            r = returnsDf[sym].reindex(pricesDf.index).values

            feats[f'{sym}_ret_1']  = pd.Series(c, index=pricesDf.index).pct_change(1)
            feats[f'{sym}_ret_5']  = pd.Series(c, index=pricesDf.index).pct_change(5)
            feats[f'{sym}_ret_10'] = pd.Series(c, index=pricesDf.index).pct_change(10)
            feats[f'{sym}_vol_20'] = pd.Series(r, index=pricesDf.index).rolling(20).std()

            # RSI proxy (14-period)
            deltas = np.diff(c, prepend=c[0])
            gains = pd.Series(np.where(deltas > 0, deltas, 0), index=pricesDf.index)
            losses = pd.Series(np.where(deltas < 0, -deltas, 0), index=pricesDf.index)
            avgGain = gains.rolling(14).mean()
            avgLoss = losses.rolling(14).mean()
            rsi = 100 - 100 / (1 + avgGain / (avgLoss + 1e-9))
            feats[f'{sym}_rsi'] = rsi

        # ---- Pairwise rolling correlations ------------------------------
        corrSeries = {}
        for s1, s2 in combinations(symbols, 2):
            r1 = returnsDf[s1].reindex(pricesDf.index)
            r2 = returnsDf[s2].reindex(pricesDf.index)
            corrSeries[f'corr_{s1}_{s2}'] = r1.rolling(corrWindow).corr(r2)

        for name, series in corrSeries.items():
            feats[name] = series

        # Cross-correlation aggregates
        if corrSeries:
            corrDf = pd.DataFrame(corrSeries)
            feats['avg_corr'] = corrDf.mean(axis=1)
            feats['max_corr'] = corrDf.max(axis=1)
            feats['min_corr'] = corrDf.min(axis=1)
            feats['corr_spread'] = feats['max_corr'] - feats['min_corr']

        # ---- Portfolio-level features -----------------------------------
        # Weighted portfolio return
        portRet = sum(
            returnsDf[sym].reindex(pricesDf.index) * self.allocations.get(sym, 0)
            for sym in symbols
        )
        feats['port_ret_1'] = portRet
        feats['port_ret_5'] = portRet.rolling(5).sum()
        feats['port_ret_10'] = portRet.rolling(10).sum()
        feats['port_vol_20'] = portRet.rolling(20).std()

        # Portfolio cumulative value & drawdown
        portCum = (1 + portRet).cumprod()
        portPeak = portCum.cummax()
        feats['port_drawdown'] = (portCum - portPeak) / (portPeak + 1e-9)

        # Average cross-stock volatility (market-stress proxy)
        avgStockVol = sum(
            feats.get(f'{sym}_vol_20', pd.Series(0, index=pricesDf.index))
            for sym in symbols
        ) / nSym
        feats['avg_stock_vol'] = avgStockVol

        # Dispersion: std of per-stock returns (low = herding, high = idiosyncratic)
        stockRetCols = [f'{sym}_ret_1' for sym in symbols]
        feats['return_dispersion'] = feats[stockRetCols].std(axis=1)

        # ---- Relative strength (stock vs portfolio mean) ----------------
        # Captures which stocks are outperforming/underperforming the
        # portfolio — critical for rotation and rebalancing signals.
        portMeanRet = returnsDf.reindex(pricesDf.index).mean(axis=1)
        for sym in symbols:
            symRet = returnsDf[sym].reindex(pricesDf.index)
            feats[f'{sym}_rel_strength_5'] = symRet.rolling(5).sum() - portMeanRet.rolling(5).sum()
            feats[f'{sym}_rel_strength_10'] = symRet.rolling(10).sum() - portMeanRet.rolling(10).sum()
            feats[f'{sym}_rel_strength_20'] = symRet.rolling(20).sum() - portMeanRet.rolling(20).sum()

        # ---- Momentum rank changes (rotation detection) -----------------
        # Track which stocks are rising/falling in rank by recent return.
        # Large rank changes hint at sector rotation.
        retLookback5 = returnsDf.reindex(pricesDf.index).rolling(5).sum()
        retLookback20 = returnsDf.reindex(pricesDf.index).rolling(20).sum()
        rank5  = retLookback5.rank(axis=1, pct=True)
        rank20 = retLookback20.rank(axis=1, pct=True)
        for sym in symbols:
            feats[f'{sym}_rank_5'] = rank5[sym]
            feats[f'{sym}_rank_chg'] = rank5[sym] - rank20[sym]

        # ---- Pair spread z-scores (cointegration proxy) -----------------
        # For each pair, price-ratio z-score flags mean-reversion setups.
        for s1, s2 in combinations(symbols, 2):
            priceRatio = pricesDf[s1] / (pricesDf[s2] + 1e-9)
            ratioMean = priceRatio.rolling(corrWindow).mean()
            ratioStd = priceRatio.rolling(corrWindow).std()
            feats[f'spread_z_{s1}_{s2}'] = (priceRatio - ratioMean) / (ratioStd + 1e-9)

        # ---- Cross-stock volume divergence ------------------------------
        # If one stock's volume spikes while another's drops, it signals
        # capital rotation between names.
        volFrames = {}
        for sym, df in stockDataDict.items():
            if df is not None and 'volume' in df.columns:
                vSeries = df['volume'].reindex(pricesDf.index)
                volFrames[sym] = vSeries

        if len(volFrames) >= 2:
            volDf = pd.DataFrame(volFrames).ffill().fillna(0)
            # Normalise each stock's volume to its own 20-day mean
            volNorm = volDf / (volDf.rolling(20).mean() + 1e-9)
            for s1, s2 in combinations(symbols, 2):
                if s1 in volNorm.columns and s2 in volNorm.columns:
                    feats[f'vol_div_{s1}_{s2}'] = volNorm[s1] - volNorm[s2]
            feats['avg_vol_norm'] = volNorm.mean(axis=1)

        # ---- Lead-lag signals -------------------------------------------
        for ll in self._leadLagRelations:
            leaderRet = returnsDf.get(ll.leader)
            if leaderRet is not None:
                feats[f'leadlag_{ll.leader}_{ll.follower}_lag{ll.lagPeriods}'] = \
                    leaderRet.reindex(pricesDf.index).shift(ll.lagPeriods)

        # ---- Sentiment features -----------------------------------------
        # Cross-stock sentiment scores from SentimentAnalyzer (all three layers).
        # Flags portfolio-wide bullish / bearish news environment and
        # captures sentiment divergence between individual stocks.
        if sentimentDataDict:
            sentCols = []
            for sym in symbols:
                sentSeries = sentimentDataDict.get(sym)
                if sentSeries is not None and len(sentSeries) > 0:
                    # Normalise to date and align to the portfolio date index
                    s = sentSeries.copy()
                    # Strip timezone so dtypes match the (usually tz-naive) price index
                    if hasattr(s.index, 'tz') and s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    if hasattr(s.index, 'normalize'):
                        s.index = s.index.normalize()
                    pIdx = pricesDf.index.normalize() if hasattr(pricesDf.index, 'normalize') else pricesDf.index
                    if hasattr(pIdx, 'tz') and pIdx.tz is not None:
                        pIdx = pIdx.tz_localize(None)
                    aligned = s.reindex(pIdx, method='ffill').fillna(0.0)
                    aligned.index = pricesDf.index
                    feats[f'{sym}_sentiment'] = aligned.values
                    sentCols.append(f'{sym}_sentiment')

            if sentCols:
                sentDf = feats[sentCols]
                feats['avg_sentiment']          = sentDf.mean(axis=1)
                feats['sentiment_dispersion']   = sentDf.std(axis=1).fillna(0)
                feats['sentiment_ma5']          = feats['avg_sentiment'].rolling(5, min_periods=1).mean()
                # Rolling correlation: avg sentiment vs portfolio return
                portRetForCorr = sum(
                    returnsDf[sym].reindex(pricesDf.index).fillna(0)
                    * self.allocations.get(sym, 0)
                    for sym in symbols
                )
                feats['sentiment_port_corr'] = (
                    feats['avg_sentiment']
                    .rolling(20, min_periods=5)
                    .corr(portRetForCorr)
                    .fillna(0)
                )

        # ---- Sector-aware features (from stock metadata) ----------------
        if self._stockMetadata:
            sectorMap = {}
            for sym in symbols:
                meta = self._stockMetadata.get(sym, {})
                sect = meta.get('sector', 'Unknown') if isinstance(meta, dict) else 'Unknown'
                sectorMap[sym] = sect

            # Same-sector pair indicator
            for s1, s2 in combinations(symbols, 2):
                sameSector = 1.0 if sectorMap.get(s1) == sectorMap.get(s2) else 0.0
                feats[f'same_sector_{s1}_{s2}'] = sameSector

            # Per-sector mean return
            sectorReturns: Dict[str, pd.Series] = {}
            for sym in symbols:
                sect = sectorMap.get(sym, 'Unknown')
                r = returnsDf[sym].reindex(pricesDf.index)
                sectorReturns.setdefault(sect, []).append(r)
            for sect, seriesList in sectorReturns.items():
                if seriesList:
                    sectorMean = pd.concat(seriesList, axis=1).mean(axis=1)
                    feats[f'sector_ret_{sect[:8]}'] = sectorMean.rolling(5).sum()

            # Supply chain relationship features
            for s1, s2 in combinations(symbols, 2):
                m1 = self._stockMetadata.get(s1, {})
                m2 = self._stockMetadata.get(s2, {})
                supplyUp1 = m1.get('supplyChainUp', []) or []
                supplyDown1 = m1.get('supplyChainDown', []) or []
                supplyUp2 = m2.get('supplyChainUp', []) or []
                supplyDown2 = m2.get('supplyChainDown', []) or []

                # s1 is supplier of s2 (s1 in s2's upstream) or s2 is customer of s1
                isSupplyChain = (
                    s1 in supplyUp2 or s2 in supplyDown1
                    or s2 in supplyUp1 or s1 in supplyDown2
                )
                feats[f'supply_chain_{s1}_{s2}'] = 1.0 if isSupplyChain else 0.0

            # Portfolio type features
            autoCount = sum(1 for sym in symbols
                          if self._stockMetadata.get(sym, {}).get('portfolioType') == 'automatic')
            feats['auto_stock_ratio'] = autoCount / max(len(symbols), 1)

            for sym in symbols:
                m = self._stockMetadata.get(sym, {})
                feats[f'{sym}_is_auto'] = 1.0 if m.get('portfolioType') == 'automatic' else 0.0
                feats[f'{sym}_auto_score'] = float(m.get('autoScore', 0.0) or 0.0)
                connType = m.get('connectionType', '')
                feats[f'{sym}_conn_supplier']   = 1.0 if connType == 'supplier' else 0.0
                feats[f'{sym}_conn_competitor']  = 1.0 if connType == 'competitor' else 0.0
                feats[f'{sym}_conn_customer']    = 1.0 if connType == 'customer' else 0.0

            # Connected-to-parent correlation features
            for s1, s2 in combinations(symbols, 2):
                m1 = self._stockMetadata.get(s1, {})
                m2 = self._stockMetadata.get(s2, {})
                parentOf1 = m1.get('autoAddedFrom', []) or []
                parentOf2 = m2.get('autoAddedFrom', []) or []
                isParentChild = s2 in parentOf1 or s1 in parentOf2
                feats[f'parent_child_{s1}_{s2}'] = 1.0 if isParentChild else 0.0

        # ---- Strategy features (from StrategyEngine) --------------------
        if self._strategyFeatures:
            for sym in symbols:
                sfeat = self._strategyFeatures.get(sym)
                if sfeat is not None:
                    for si, sv in enumerate(sfeat):
                        feats[f'{sym}_strat_{si}'] = float(sv)

        # ---- Cross-stock rule features ----------------------------------
        if self._crossStockRuleFeatures is not None and len(self._crossStockRuleFeatures) > 0:
            ruleFeat = self._crossStockRuleFeatures.reindex(pricesDf.index).ffill().fillna(0)
            for col in ruleFeat.columns:
                feats[col] = ruleFeat[col]

        # ---- Drop NaN rows ----------------------------------------------
        feats.dropna(inplace=True)
        if len(feats) < forwardPeriods + 30:
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # ---- Labels (forward portfolio return) --------------------------
        portRetAligned = portRet.reindex(feats.index)
        forwardReturn = portRetAligned.rolling(forwardPeriods).sum().shift(-forwardPeriods)
        forwardVol = portRetAligned.rolling(forwardPeriods).std().shift(-forwardPeriods)

        # Regime labels (based on forward conditions)
        # 0=LOW_RISK, 1=NORMAL, 2=HIGH_RISK, 3=CRISIS
        regimeLabels = pd.Series(1, index=feats.index)  # default NORMAL

        for i in range(len(feats)):
            fwd_r = forwardReturn.iloc[i] if i < len(forwardReturn) else np.nan
            fwd_v = forwardVol.iloc[i] if i < len(forwardVol) else np.nan
            if pd.isna(fwd_r) or pd.isna(fwd_v):
                continue
            if fwd_v < 0.008 and fwd_r > -0.01:
                regimeLabels.iloc[i] = 0   # LOW_RISK
            elif fwd_r < -0.05:
                regimeLabels.iloc[i] = 3   # CRISIS
            elif fwd_v > 0.02 or fwd_r < -0.02:
                regimeLabels.iloc[i] = 2   # HIGH_RISK
            else:
                regimeLabels.iloc[i] = 1   # NORMAL

        # Hedge labels (derived from regime + forward drawdown)
        hedgeLabels = pd.Series(0, index=feats.index)   # default NONE
        for i in range(len(feats)):
            regime = regimeLabels.iloc[i]
            fwd_r = forwardReturn.iloc[i] if i < len(forwardReturn) else np.nan
            if pd.isna(fwd_r):
                continue
            if regime == 3:
                hedgeLabels.iloc[i] = 4    # FULL_HEDGE
            elif regime == 2 and fwd_r < -0.03:
                hedgeLabels.iloc[i] = 3    # REDUCE_50
            elif regime == 2:
                hedgeLabels.iloc[i] = 2    # REDUCE_25
            elif fwd_r < -0.01:
                hedgeLabels.iloc[i] = 1    # REDUCE_10

        # Trim to rows where forward return is available
        valid = ~forwardReturn.isna()
        validIdx = valid[valid].index
        feats = feats.loc[feats.index.isin(validIdx)]
        regimeLabels = regimeLabels.loc[feats.index]
        hedgeLabels = hedgeLabels.loc[feats.index]
        forwardReturn = forwardReturn.loc[feats.index]

        X = feats.values
        yRegime = regimeLabels.values.astype(int)
        yHedge = hedgeLabels.values.astype(int)
        yReturn = forwardReturn.values.astype(float)
        names = list(feats.columns)

        return X, yRegime, yHedge, yReturn, names

    def _buildCurrentFeatures(self,
                              stockPredictions: Dict[str, StockPrediction],
                              stockDataDict: Dict[str, pd.DataFrame],
                              corrWindow: int) -> Optional[np.ndarray]:
        """
        Build a single feature row for the current moment (latest data).
        """
        pricesDf, returnsDf = self._alignStockData(stockDataDict)
        if pricesDf is None or len(pricesDf) < corrWindow + 10:
            return None

        symbols = list(pricesDf.columns)
        feats = {}

        for sym in symbols:
            c = pricesDf[sym].values.astype(float)
            r = returnsDf[sym].values

            feats[f'{sym}_ret_1']  = (c[-1] - c[-2]) / (c[-2] + 1e-9) if len(c) >= 2 else 0
            feats[f'{sym}_ret_5']  = (c[-1] - c[-6]) / (c[-6] + 1e-9) if len(c) >= 6 else 0
            feats[f'{sym}_ret_10'] = (c[-1] - c[-11]) / (c[-11] + 1e-9) if len(c) >= 11 else 0
            feats[f'{sym}_vol_20'] = float(np.std(r[-20:])) if len(r) >= 20 else 0

            # RSI proxy
            deltas = np.diff(c[-15:]) if len(c) >= 15 else np.array([0])
            gains = np.mean(np.where(deltas > 0, deltas, 0))
            losses = np.mean(np.where(deltas < 0, -deltas, 0))
            feats[f'{sym}_rsi'] = 100 - 100 / (1 + gains / (losses + 1e-9))

        # Pairwise correlations
        for s1, s2 in combinations(symbols, 2):
            r1 = returnsDf[s1].values[-corrWindow:]
            r2 = returnsDf[s2].values[-corrWindow:]
            if len(r1) >= corrWindow and len(r2) >= corrWindow:
                feats[f'corr_{s1}_{s2}'] = float(np.corrcoef(r1, r2)[0, 1])
            else:
                feats[f'corr_{s1}_{s2}'] = 0.0

        corrVals = [v for k, v in feats.items() if k.startswith('corr_')]
        feats['avg_corr'] = float(np.mean(corrVals)) if corrVals else 0
        feats['max_corr'] = float(np.max(corrVals)) if corrVals else 0
        feats['min_corr'] = float(np.min(corrVals)) if corrVals else 0
        feats['corr_spread'] = feats['max_corr'] - feats['min_corr']

        # Portfolio-level
        portReturns = returnsDf.tail(20)
        portRet = sum(portReturns[sym] * self.allocations.get(sym, 0)
                      for sym in symbols if sym in portReturns.columns)
        feats['port_ret_1'] = float(portRet.iloc[-1]) if len(portRet) > 0 else 0
        feats['port_ret_5'] = float(portRet.tail(5).sum()) if len(portRet) >= 5 else 0
        feats['port_ret_10'] = float(portRet.tail(10).sum()) if len(portRet) >= 10 else 0
        feats['port_vol_20'] = float(portRet.std()) if len(portRet) >= 10 else 0

        portCum = (1 + portRet).cumprod()
        portPeak = portCum.cummax()
        dd = float((portCum.iloc[-1] - portPeak.iloc[-1]) / (portPeak.iloc[-1] + 1e-9)) if len(portCum) > 0 else 0
        feats['port_drawdown'] = dd

        feats['avg_stock_vol'] = float(np.mean([feats.get(f'{s}_vol_20', 0) for s in symbols]))
        feats['return_dispersion'] = float(np.std([feats.get(f'{s}_ret_1', 0) for s in symbols]))

        # ---- Relative strength ------------------------------------------
        meanRet5 = float(np.mean([float(returnsDf[s].tail(5).sum()) for s in symbols]))
        meanRet10 = float(np.mean([float(returnsDf[s].tail(10).sum()) for s in symbols]))
        meanRet20 = float(np.mean([float(returnsDf[s].tail(20).sum()) for s in symbols]))
        for sym in symbols:
            feats[f'{sym}_rel_strength_5'] = float(returnsDf[sym].tail(5).sum()) - meanRet5
            feats[f'{sym}_rel_strength_10'] = float(returnsDf[sym].tail(10).sum()) - meanRet10
            feats[f'{sym}_rel_strength_20'] = float(returnsDf[sym].tail(20).sum()) - meanRet20

        # ---- Momentum rank changes -------------------------------------
        ret5_vals = {s: float(returnsDf[s].tail(5).sum()) for s in symbols}
        ret20_vals = {s: float(returnsDf[s].tail(20).sum()) for s in symbols}
        sorted5 = sorted(ret5_vals.items(), key=lambda x: x[1])
        sorted20 = sorted(ret20_vals.items(), key=lambda x: x[1])
        rank5 = {s: i / max(len(symbols) - 1, 1) for i, (s, _) in enumerate(sorted5)}
        rank20 = {s: i / max(len(symbols) - 1, 1) for i, (s, _) in enumerate(sorted20)}
        for sym in symbols:
            feats[f'{sym}_rank_5'] = rank5.get(sym, 0.5)
            feats[f'{sym}_rank_chg'] = rank5.get(sym, 0.5) - rank20.get(sym, 0.5)

        # ---- Pair spread z-scores (cointegration proxy) -----------------
        for s1, s2 in combinations(symbols, 2):
            p1 = pricesDf[s1].values[-corrWindow:]
            p2 = pricesDf[s2].values[-corrWindow:]
            if len(p1) >= corrWindow and len(p2) >= corrWindow:
                ratio = p1 / (p2 + 1e-9)
                ratioMean = float(np.mean(ratio))
                ratioStd = float(np.std(ratio))
                feats[f'spread_z_{s1}_{s2}'] = (ratio[-1] - ratioMean) / (ratioStd + 1e-9)
            else:
                feats[f'spread_z_{s1}_{s2}'] = 0.0

        # ---- Cross-stock volume divergence ------------------------------
        volNormVals = {}
        for sym in symbols:
            df = stockDataDict.get(sym)
            if df is not None and 'volume' in df.columns:
                vols = df['volume'].reindex(pricesDf.index).values
                if len(vols) >= 20:
                    mean20 = float(np.mean(vols[-20:]))
                    volNormVals[sym] = vols[-1] / (mean20 + 1e-9)
                else:
                    volNormVals[sym] = 1.0
            else:
                volNormVals[sym] = 1.0

        for s1, s2 in combinations(symbols, 2):
            feats[f'vol_div_{s1}_{s2}'] = volNormVals.get(s1, 1.0) - volNormVals.get(s2, 1.0)
        feats['avg_vol_norm'] = float(np.mean(list(volNormVals.values()))) if volNormVals else 1.0

        # Lead-lag
        for ll in self._leadLagRelations:
            leaderRet = returnsDf.get(ll.leader)
            if leaderRet is not None and len(leaderRet) > ll.lagPeriods:
                feats[f'leadlag_{ll.leader}_{ll.follower}_lag{ll.lagPeriods}'] = \
                    float(leaderRet.iloc[-(ll.lagPeriods + 1)])
            else:
                feats[f'leadlag_{ll.leader}_{ll.follower}_lag{ll.lagPeriods}'] = 0.0

        # ---- Sentiment features (current-day lookup) --------------------
        if self._sentimentDataDict:
            today = pricesDf.index[-1]
            todayDate = today.normalize() if hasattr(today, 'normalize') else today
            # Strip timezone so lookups work against tz-naive price indices
            if hasattr(todayDate, 'tz') and todayDate.tz is not None:
                todayDate = todayDate.tz_localize(None)
            sentScores = {}
            for sym in symbols:
                sentSeries = self._sentimentDataDict.get(sym)
                if sentSeries is not None and len(sentSeries) > 0:
                    s = sentSeries.copy()
                    if hasattr(s.index, 'tz') and s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    if hasattr(s.index, 'normalize'):
                        s.index = s.index.normalize()
                    loc = s.index.get_indexer([todayDate], method='ffill')[0]
                    score = float(s.iloc[loc]) if loc >= 0 else 0.0
                else:
                    score = 0.0
                feats[f'{sym}_sentiment'] = score
                sentScores[sym] = score

            if sentScores:
                vals = list(sentScores.values())
                feats['avg_sentiment']        = float(np.mean(vals))
                feats['sentiment_dispersion'] = float(np.std(vals))
                feats['sentiment_ma5']        = feats['avg_sentiment']  # current-day proxy
                # Compute sentiment-vs-portfolio-return correlation from
                # the trailing 20 days (mirrors the rolling window used in
                # _buildDataset) so the feature distribution is consistent.
                try:
                    avgSentSeries = pd.Series(dtype=float)
                    for sym in symbols:
                        sentSeries = self._sentimentDataDict.get(sym)
                        if sentSeries is not None and len(sentSeries) > 0:
                            s = sentSeries.copy()
                            if hasattr(s.index, 'tz') and s.index.tz is not None:
                                s.index = s.index.tz_localize(None)
                            if hasattr(s.index, 'normalize'):
                                s.index = s.index.normalize()
                            pIdx = pricesDf.index.normalize() if hasattr(pricesDf.index, 'normalize') else pricesDf.index
                            if hasattr(pIdx, 'tz') and pIdx.tz is not None:
                                pIdx = pIdx.tz_localize(None)
                            aligned = s.reindex(pIdx, method='ffill').fillna(0.0)
                            aligned.index = pricesDf.index
                            avgSentSeries = aligned if avgSentSeries.empty else avgSentSeries + aligned
                    if not avgSentSeries.empty:
                        avgSentSeries = avgSentSeries / max(len(sentScores), 1)
                        portRetFull = sum(
                            returnsDf[sym].reindex(pricesDf.index).fillna(0)
                            * self.allocations.get(sym, 0)
                            for sym in symbols
                        )
                        tail20Sent = avgSentSeries.tail(20)
                        tail20Ret  = portRetFull.reindex(tail20Sent.index)
                        if len(tail20Sent) >= 5:
                            feats['sentiment_port_corr'] = float(
                                tail20Sent.corr(tail20Ret)
                            )
                            if np.isnan(feats['sentiment_port_corr']):
                                feats['sentiment_port_corr'] = 0.0
                        else:
                            feats['sentiment_port_corr'] = 0.0
                    else:
                        feats['sentiment_port_corr'] = 0.0
                except Exception:
                    feats['sentiment_port_corr'] = 0.0

        # ---- Sector-aware features (mirror _buildDataset) ---------------
        if self._stockMetadata:
            sectorMap = {}
            for sym in symbols:
                meta = self._stockMetadata.get(sym, {})
                sect = meta.get('sector', 'Unknown') if isinstance(meta, dict) else 'Unknown'
                sectorMap[sym] = sect

            for s1, s2 in combinations(symbols, 2):
                sameSector = 1.0 if sectorMap.get(s1) == sectorMap.get(s2) else 0.0
                feats[f'same_sector_{s1}_{s2}'] = sameSector

            sectorReturns: Dict[str, List[float]] = {}
            for sym in symbols:
                sect = sectorMap.get(sym, 'Unknown')
                r5 = float(returnsDf[sym].tail(5).sum()) if sym in returnsDf.columns else 0.0
                sectorReturns.setdefault(sect, []).append(r5)
            for sect, vals in sectorReturns.items():
                feats[f'sector_ret_{sect[:8]}'] = float(np.mean(vals))

            # Supply chain relationship features (mirror _buildDataset)
            for s1, s2 in combinations(symbols, 2):
                m1 = self._stockMetadata.get(s1, {})
                m2 = self._stockMetadata.get(s2, {})
                supplyUp1 = m1.get('supplyChainUp', []) or []
                supplyDown1 = m1.get('supplyChainDown', []) or []
                supplyUp2 = m2.get('supplyChainUp', []) or []
                supplyDown2 = m2.get('supplyChainDown', []) or []
                isSupplyChain = (
                    s1 in supplyUp2 or s2 in supplyDown1
                    or s2 in supplyUp1 or s1 in supplyDown2
                )
                feats[f'supply_chain_{s1}_{s2}'] = 1.0 if isSupplyChain else 0.0

            autoCount = sum(1 for sym in symbols
                          if self._stockMetadata.get(sym, {}).get('portfolioType') == 'automatic')
            feats['auto_stock_ratio'] = autoCount / max(len(symbols), 1)

            for sym in symbols:
                m = self._stockMetadata.get(sym, {})
                feats[f'{sym}_is_auto'] = 1.0 if m.get('portfolioType') == 'automatic' else 0.0
                feats[f'{sym}_auto_score'] = float(m.get('autoScore', 0.0) or 0.0)
                connType = m.get('connectionType', '')
                feats[f'{sym}_conn_supplier']   = 1.0 if connType == 'supplier' else 0.0
                feats[f'{sym}_conn_competitor']  = 1.0 if connType == 'competitor' else 0.0
                feats[f'{sym}_conn_customer']    = 1.0 if connType == 'customer' else 0.0

            for s1, s2 in combinations(symbols, 2):
                m1 = self._stockMetadata.get(s1, {})
                m2 = self._stockMetadata.get(s2, {})
                parentOf1 = m1.get('autoAddedFrom', []) or []
                parentOf2 = m2.get('autoAddedFrom', []) or []
                isParentChild = s2 in parentOf1 or s1 in parentOf2
                feats[f'parent_child_{s1}_{s2}'] = 1.0 if isParentChild else 0.0

        # ---- Strategy features ------------------------------------------
        if self._strategyFeatures:
            for sym in symbols:
                sfeat = self._strategyFeatures.get(sym)
                if sfeat is not None:
                    for si, sv in enumerate(sfeat):
                        feats[f'{sym}_strat_{si}'] = float(sv)

        # ---- Cross-stock rule features ----------------------------------
        if self._crossStockRuleFeatures is not None and len(self._crossStockRuleFeatures) > 0:
            lastIdx = self._crossStockRuleFeatures.index[-1]
            for col in self._crossStockRuleFeatures.columns:
                feats[col] = float(self._crossStockRuleFeatures.loc[lastIdx, col])

        # Guarantee feature order matches training
        row = np.array([feats.get(name, 0.0) for name in self._featureNames])
        return row

    # =====================================================================
    # INTERNAL — model fitting
    # =====================================================================

    def _fitModels(self, X, yRegime, yHedge, yReturn, testSize, verbose):
        Xtr, Xte, yRTr, yRTe, yHTr, yHTe, yRetTr, yRetTe = train_test_split(
            X, yRegime, yHedge, yReturn, test_size=testSize, shuffle=False
        )

        self._scaler = StandardScaler()
        XtrS = self._scaler.fit_transform(Xtr)
        XteS = self._scaler.transform(Xte)

        # --- Regime classifier ---
        self._regimeModel = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        self._regimeModel.fit(XtrS, yRTr)
        regAcc = accuracy_score(yRTe, self._regimeModel.predict(XteS))

        # --- Hedge classifier ---
        self._hedgeModel = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        self._hedgeModel.fit(XtrS, yHTr)
        hedgeAcc = accuracy_score(yHTe, self._hedgeModel.predict(XteS))

        # --- Portfolio return regressor ---
        self._returnModel = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        self._returnModel.fit(XtrS, yRetTr)
        retPred = self._returnModel.predict(XteS)
        retRMSE = float(np.sqrt(np.mean((retPred - yRetTe) ** 2)))

        topFeat = sorted(
            zip(self._featureNames, self._regimeModel.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:10]

        metrics = PortfolioModelMetrics(
            regimeAccuracy=regAcc,
            hedgeAccuracy=hedgeAcc,
            returnRMSE=retRMSE,
            trainSamples=len(Xtr),
            testSamples=len(Xte),
            topFeatures=topFeat,
        )

        if verbose:
            print(f"    Regime accuracy  : {regAcc * 100:.1f}%")
            print(f"    Hedge accuracy   : {hedgeAcc * 100:.1f}%")
            print(f"    Return RMSE      : {retRMSE:.5f}")
            print(f"    Train/test       : {len(Xtr)}/{len(Xte)}")
            print(f"    Top features     : "
                  + ", ".join(f"{n}={v:.3f}" for n, v in topFeat[:5]))

        return metrics

    # =====================================================================
    # INTERNAL — allocation & hedging logic
    # =====================================================================

    def _computeAllocationAdjustments(
            self,
            stockPredictions: Dict[str, StockPrediction],
            regime: RiskRegime
    ) -> List[AllocationAdjustment]:
        """
        Combine per-stock ML signals with the current risk regime to
        produce allocation adjustments.

        Rules:
          - In CRISIS / HIGH_RISK: reduce allocation to stocks with SELL signals
          - Shift capital toward stocks with BUY + high confidence
          - Manual stocks: minimum 5% allocation (diversification floor)
          - Connected/auto stocks: can go to 0% (phase-out if underperforming)
          - Connected stocks get capped at lower maximum (20% vs 60% for manual)
          - Phase-in: new connected stocks start at low allocation, ramp up
          - Phase-out: connected stocks with persistent SELL signals get removed
        """
        adjustments = []
        signalScores: Dict[str, float] = {}

        for symbol in self.symbols:
            pred = stockPredictions.get(symbol, StockPrediction())
            meta = self._stockMetadata.get(symbol, {}) if self._stockMetadata else {}
            isAuto = meta.get('portfolioType') == 'automatic'

            if pred.signal == TradingSignal.BUY:
                score = 0.5 + 0.5 * pred.confidence
            elif pred.signal == TradingSignal.SELL:
                score = -(0.5 + 0.5 * pred.confidence)
            else:
                score = 0.0

            if regime in (RiskRegime.CRISIS, RiskRegime.HIGH_RISK):
                score -= 0.3
                if isAuto:
                    score -= 0.2  # extra penalty: shed auto stocks first in crisis

            autoScore = float(meta.get('autoScore', 0.5) or 0.5)
            if isAuto:
                score *= (0.5 + 0.5 * autoScore)

            signalScores[symbol] = score

        MANUAL_MIN = 0.05
        MANUAL_MAX = 0.60
        AUTO_MIN   = 0.0
        AUTO_MAX   = 0.20
        AUTO_PHASE_IN = 0.03  # max initial allocation for a brand-new auto stock

        for symbol in self.symbols:
            meta = self._stockMetadata.get(symbol, {}) if self._stockMetadata else {}
            isAuto = meta.get('portfolioType') == 'automatic'
            currentAlloc = self.allocations.get(symbol, 0)
            score = signalScores[symbol]

            if isAuto:
                minA, maxA = AUTO_MIN, AUTO_MAX
                # Phase-in: if stock has very low current allocation, ramp gently
                if currentAlloc < AUTO_PHASE_IN and score > 0:
                    shift = min(score * 0.02, AUTO_PHASE_IN - currentAlloc)
                else:
                    shift = score * 0.04
                # Phase-out: persistent sell → drop to zero
                if score < -0.3:
                    shift = -currentAlloc  # zero it out
            else:
                minA, maxA = MANUAL_MIN, MANUAL_MAX
                shift = score * 0.05

            suggested = currentAlloc + shift
            suggested = max(minA, min(maxA, suggested))

            if abs(suggested - currentAlloc) > 0.005:
                reason = self._allocationReason(symbol, stockPredictions.get(symbol), regime, shift)
                if isAuto and suggested < 0.005:
                    reason += ' → phase-out (remove from auto portfolio)'
                adjustments.append(AllocationAdjustment(
                    symbol=symbol,
                    currentAllocation=currentAlloc,
                    suggestedAllocation=round(suggested, 4),
                    reason=reason,
                ))

        # Normalise suggested allocations to sum to 1.0
        if adjustments:
            totalSuggested = sum(a.suggestedAllocation for a in adjustments)
            adjustedSymbols = {a.symbol for a in adjustments}
            for sym in self.symbols:
                if sym not in adjustedSymbols:
                    totalSuggested += self.allocations.get(sym, 0)
            if totalSuggested > 0:
                for adj in adjustments:
                    adj.suggestedAllocation = round(
                        adj.suggestedAllocation / totalSuggested, 4
                    )

        return adjustments

    @staticmethod
    def _allocationReason(symbol, pred, regime, shift):
        parts = []
        if pred and pred.signal != TradingSignal.HOLD:
            parts.append(f"{pred.signal.value} signal (conf={pred.confidence:.0%})")
        if regime in (RiskRegime.CRISIS, RiskRegime.HIGH_RISK):
            parts.append(f"risk regime={regime.value}")
        direction = "increase" if shift > 0 else "decrease"
        parts.append(f"{direction} by {abs(shift)*100:.1f}%")
        return f"{symbol}: " + ", ".join(parts) if parts else ""

    def _findMeanReversionOpportunities(
            self,
            stockPredictions: Dict[str, StockPrediction],
            stockDataDict: Dict[str, pd.DataFrame],
            corrWindow: int
    ) -> List[str]:
        """
        If two stocks are highly correlated but currently diverging
        (one up, one down), flag a mean-reversion opportunity.
        """
        opps = []
        pricesDf, returnsDf = self._alignStockData(stockDataDict)
        if returnsDf is None or len(returnsDf) < corrWindow:
            return opps

        recentCorr = returnsDf.tail(corrWindow).corr()
        symbols = list(returnsDf.columns)

        for s1, s2 in combinations(symbols, 2):
            corr = recentCorr.loc[s1, s2] if s1 in recentCorr.index and s2 in recentCorr.columns else 0
            if corr < 0.4:
                continue

            # Check recent divergence
            ret1_5 = float(returnsDf[s1].tail(5).sum())
            ret2_5 = float(returnsDf[s2].tail(5).sum())

            if abs(ret1_5 - ret2_5) > 0.03:  # 3% divergence in 5 days
                laggard = s1 if ret1_5 < ret2_5 else s2
                leader = s2 if laggard == s1 else s1
                opps.append(
                    f"Mean-reversion: {laggard} lagging {leader} "
                    f"(corr={corr:.2f}, divergence={abs(ret1_5-ret2_5)*100:.1f}%)"
                )

        return opps

    def _estimatePortfolioVolatility(self, stockDataDict, corrWindow):
        """Annualised portfolio volatility estimate."""
        pricesDf, returnsDf = self._alignStockData(stockDataDict)
        if returnsDf is None or len(returnsDf) < corrWindow:
            return 0.0

        portRet = sum(
            returnsDf[sym] * self.allocations.get(sym, 0)
            for sym in returnsDf.columns
        )
        dailyVol = float(portRet.tail(corrWindow).std())
        return dailyVol * np.sqrt(252)  # annualise

    def _averageCrossCorrelation(self, stockDataDict, corrWindow):
        """Average pairwise correlation over recent window."""
        pricesDf, returnsDf = self._alignStockData(stockDataDict)
        if returnsDf is None or len(returnsDf) < corrWindow:
            return 0.0

        corrMat = returnsDf.tail(corrWindow).corr()
        n = len(corrMat)
        if n < 2:
            return 0.0

        # Average of off-diagonal elements
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        return float(corrMat.values[mask].mean())
