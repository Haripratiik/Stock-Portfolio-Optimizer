"""
StockMLModel — Per-Stock ML Trading Model

Combines refined pattern information with technical indicators to create
a unified ML model that learns optimal trading decisions for a specific stock.

Training data sources:
  1. Historical OHLCV data with computed technical indicators
  2. Pattern match signals from the refined PatternBank
  3. Trade outcomes from the Backtester  (real historical)
  4. Trade outcomes from the MCMCSimulator (simulated paths)

Capabilities:
  - Signal generation   : BUY / SELL / HOLD per period with confidence
  - Return prediction   : Expected forward return (regression)
  - Market regime detect: Trending, mean-reverting, volatile
  - Pattern effectiveness: Which patterns work best in which conditions
  - Position sizing     : Suggested fraction of available capital

The model is GENERALISABLE — the same class is instantiated once per stock,
but the architecture, feature engineering, and training loop are identical
regardless of ticker.
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import warnings

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GeneticAlgorithm import (
    PatternChromosome, PatternGene, PatternBank, CandleType, StockDataFetcher
)
from Backtester import BacktestResult
from MCMCSimulator import MonteCarloResults


# =============================================================================
# Enums & data classes
# =============================================================================

class TradingSignal(Enum):
    BUY  = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


class MarketRegime(Enum):
    TRENDING_UP   = 'TRENDING_UP'
    TRENDING_DOWN = 'TRENDING_DOWN'
    MEAN_REVERTING = 'MEAN_REVERTING'
    VOLATILE      = 'VOLATILE'
    LOW_VOL       = 'LOW_VOL'


@dataclass
class StockPrediction:
    """A single-period prediction from the StockMLModel."""
    timestamp: Optional[pd.Timestamp] = None
    signal: TradingSignal = TradingSignal.HOLD
    confidence: float = 0.0          # 0-1 (from ML model)
    expectedReturn: float = 0.0      # predicted forward return (%)
    positionSize: float = 0.0        # suggested capital fraction (0-1)
    regime: MarketRegime = MarketRegime.LOW_VOL
    triggeringPatterns: List[int] = field(default_factory=list)
    patternConfidence: float = 0.0   # 0-1 (from pattern matching, independent of ML)
    patternSignal: TradingSignal = TradingSignal.HOLD  # signal from patterns alone
    # ── Sentiment (Layer 1+2+3 ensemble from SentimentAnalysis) ────────────
    sentimentScore: float = 0.0          # raw ensemble score -1 → +1
    sentimentSignal: TradingSignal = TradingSignal.HOLD
    sentimentConfidence: float = 0.0     # ensemble confidence 0–1


@dataclass
class ModelMetrics:
    """Training / evaluation metrics."""
    directionAccuracy: float = 0.0
    returnRMSE: float = 0.0
    trainSamples: int = 0
    testSamples: int = 0
    topFeatures: List[Tuple[str, float]] = field(default_factory=list)


# =============================================================================
# StockMLModel
# =============================================================================

class StockMLModel:
    """
    Per-stock ML trading model.

    Typical lifecycle::

        model = StockMLModel(symbol='AAPL')
        model.train(df, patternBank, forwardPeriods=5)
        model.augmentFromBacktest(backtestResults)   # optional
        predictions = model.predictBatch(latestDf)
    """

    # — construction -------------------------------------------------------

    def __init__(self, symbol: str, forgiveness: float = 0.05):
        self.symbol = symbol
        self.forgiveness = forgiveness

        # Models (trained later)
        self._directionModel: Optional[GradientBoostingClassifier] = None
        self._returnModel: Optional[GradientBoostingRegressor] = None
        self._scaler: Optional[StandardScaler] = None

        # Book-keeping
        self._featureNames: List[str] = []
        self._patterns: List[PatternChromosome] = []
        self._trained: bool = False
        self._metrics: Optional[ModelMetrics] = None
        self._originalMetrics: Optional[ModelMetrics] = None  # pre-augmentation accuracy

        # Accumulated training data (for incremental augmentation)
        self._X_accum: Optional[np.ndarray] = None
        self._yDir_accum: Optional[np.ndarray] = None
        self._yRet_accum: Optional[np.ndarray] = None
        self._weights_accum: Optional[np.ndarray] = None

        # Feature cache to avoid recomputing pattern similarity
        self._featureCache: Optional[pd.DataFrame] = None
        self._featureCacheKey: Optional[tuple] = None

        # Sentiment data (date-indexed Series with scores in [-1, +1])
        self._sentimentSeries: Optional[pd.Series] = None

        # Strategy features from StrategyEngine (injected externally)
        self._strategyFeatures: Optional[np.ndarray] = None
        self._stockMeta: Optional[dict] = None

        # Earnings dates (fetched once at train/predict time for feature engineering)
        self._earningsDates: Optional[List] = None

    # — strategy/metadata injection ----------------------------------------

    def setStrategyFeatures(self, features: np.ndarray):
        """
        Inject strategy signal features computed by StrategyEngine.
        ``features`` is a 1-D array of per-strategy signal strengths.
        These are appended to every row of the feature matrix.
        """
        self._strategyFeatures = features
        self._featureCache = None
        self._featureCacheKey = None

    def setStockMetadata(self, meta: dict):
        """
        Inject stock metadata (sector, industry, etc.) for sector-relative
        features.  Expected keys: 'sector', 'industry', etc.
        """
        self._stockMeta = meta

    def loadEarningsDates(self):
        """
        Fetch historical and upcoming earnings dates for this symbol and
        cache them as a list of tz-naive pd.Timestamps.  Called once during
        train/predict so the feature pipeline has earnings context.
        """
        if self._earningsDates is not None:
            return
        try:
            import yfinance as yf
            tk = yf.Ticker(self.symbol)
            cal = tk.get_earnings_dates(limit=40)
            if cal is not None and not cal.empty:
                self._earningsDates = [
                    pd.Timestamp(d).normalize().tz_localize(None)
                    for d in cal.index
                ]
            else:
                self._earningsDates = []
        except Exception:
            self._earningsDates = []

    # — public API ----------------------------------------------------------

    def train(self,
              df: pd.DataFrame,
              patternBank: PatternBank,
              forwardPeriods: int = 5,
              testSize: float = 0.2,
              sentimentData: Optional[pd.Series] = None,
              verbose: bool = True) -> ModelMetrics:
        """
        Train both direction-classifier and return-regressor on historical
        OHLCV data augmented with pattern-match signals and (optionally)
        sentiment scores.

        Args:
            df:              OHLCV DataFrame (with 'open','high','low','close','volume')
            patternBank:     Refined pattern bank for this stock
            forwardPeriods:  Prediction horizon (candles ahead)
            testSize:        Held-out fraction for evaluation
            sentimentData:   Optional date-indexed pd.Series of daily sentiment
                             scores in [-1, +1] from SentimentAnalyzer.
                             When provided, four sentiment features are added
                             to the ML feature matrix.
            verbose:         Print progress

        Returns:
            ModelMetrics with accuracy, RMSE, top features.
        """
        self._patterns = list(patternBank.patterns) if patternBank else []
        self._sentimentSeries = sentimentData
        self._sentimentNormalized = None
        self._featureCache = None
        self._featureCacheKey = None

        # Pre-fetch earnings dates so _computeFeatures can use them
        self.loadEarningsDates()

        if verbose:
            print(f"\n  [StockMLModel] Training for {self.symbol} "
                  f"({len(df)} rows, {len(self._patterns)} patterns) ...")

        # ---- Feature matrix + earnings-aware sample weights ---------------
        X, yDir, yRet, names, weights = self._buildDataset(df, forwardPeriods)
        self._featureNames = names

        if len(X) < 60:
            if verbose:
                print("    Not enough data to train (need ≥60 rows)")
            return ModelMetrics()

        self._X_accum = X
        self._yDir_accum = yDir
        self._yRet_accum = yRet
        self._weights_accum = weights

        # ---- Split & scale -----------------------------------------------
        metrics = self._fitModels(X, yDir, yRet, weights, testSize, verbose)
        # Only mark as trained if models were actually created
        self._trained = (self._scaler is not None and self._directionModel is not None)
        self._metrics = metrics
        self._originalMetrics = metrics  # snapshot before augmentation inflates it
        return metrics

    def augmentFromBacktest(self,
                           backtestResults: Dict[str, BacktestResult],
                           df: pd.DataFrame,
                           forwardPeriods: int = 5,
                           verbose: bool = True):
        """
        Augment training data with labelled trade rows from the Backtester,
        then re-fit the models.  Trade rows receive 2× weight so the model
        pays extra attention to periods where trades actually occurred.
        """
        if not self._trained or self._X_accum is None:
            return

        tradeTimestamps = set()
        tradeOutcomes: Dict[pd.Timestamp, float] = {}
        for interval, btResult in backtestResults.items():
            for trade in btResult.trades:
                ts = trade['timestamp']
                tradeTimestamps.add(ts)
                tradeOutcomes[ts] = trade['returnPct']

        if not tradeTimestamps:
            return

        # Build features only for trade rows, then up-weight them
        Xnew, yDirNew, yRetNew, _, _wNew = self._buildDataset(df, forwardPeriods)
        if len(Xnew) == 0:
            return

        # Assign higher weight to rows whose timestamp matched a trade
        newWeights = np.ones(len(Xnew))
        for i, ts in enumerate(df.index[:-forwardPeriods]):
            if i < len(newWeights) and ts in tradeTimestamps:
                newWeights[i] = 3.0  # 3× importance for trade periods

        self._X_accum = Xnew
        self._yDir_accum = yDirNew
        self._yRet_accum = yRetNew
        self._weights_accum = newWeights

        self._fitModels(Xnew, yDirNew, yRetNew, newWeights, testSize=0.2,
                        verbose=verbose)

        if verbose:
            print(f"    Augmented with {len(tradeTimestamps)} trade rows (3× weight)")

    def augmentFromMC(self,
                      mcResults: Dict[str, MonteCarloResults],
                      selfImprovementRounds: int = 2,
                      verbose: bool = True):
        """
        Self-improvement loop: augment ML training data with MC trade
        outcomes, then iteratively test predictions against MC data and
        upweight mistakes.

        Process (repeated for *selfImprovementRounds*):
          1.  Extract trade-level data from MC simulation paths.
          2.  Build synthetic feature rows from each trade's entry/exit.
          3.  Run the current model's predictions on these synthetic rows.
          4.  Compare predictions to actual MC outcomes (did trade succeed?).
          5.  Upweight rows where the model was WRONG (mistake emphasis).
          6.  Re-train on combined real + synthetic data.

        This forces the model to learn from its mistakes on synthetic
        market scenarios it has never seen before, reducing overfitting
        to the limited historical training set.
        """
        if not self._trained or self._X_accum is None:
            return

        # -- Collect MC trade data into synthetic feature rows --
        mcTradeRows: List[np.ndarray] = []
        mcDirLabels: List[int] = []
        mcRetLabels: List[float] = []

        for interval, mc in mcResults.items():
            if not mc.paths:
                continue
            for path in mc.paths:
                if not path.trades:
                    continue
                for trade in path.trades:
                    entryPrice = trade.get('entryPrice', 0)
                    exitPrice = trade.get('exitPrice', 0)
                    if entryPrice <= 0:
                        continue

                    priceMoveRaw = (exitPrice - entryPrice) / entryPrice
                    successful = trade.get('successful', False)

                    row = np.zeros(len(self._featureNames))
                    for fIdx, fName in enumerate(self._featureNames):
                        if fName == 'ret_1':
                            row[fIdx] = priceMoveRaw
                        elif fName == 'ret_3':
                            row[fIdx] = priceMoveRaw * 0.8
                        elif fName == 'ret_5':
                            row[fIdx] = priceMoveRaw * 0.6
                        elif fName == 'is_bullish':
                            row[fIdx] = 1.0 if priceMoveRaw > 0 else 0.0
                        elif fName == 'body_ratio':
                            row[fIdx] = min(1.0, abs(priceMoveRaw) * 20)
                        elif fName == 'vol_ratio':
                            row[fIdx] = 1.0
                        elif fName == 'rsi_14':
                            row[fIdx] = 60.0 if priceMoveRaw > 0 else 40.0
                        elif fName.startswith('vol_'):
                            row[fIdx] = abs(priceMoveRaw) * 5
                        elif fName == 'boll_pos':
                            row[fIdx] = 0.7 if priceMoveRaw > 0 else 0.3

                    mcTradeRows.append(row)
                    mcDirLabels.append(1 if successful else 0)
                    mcRetLabels.append(trade.get('returnPct', priceMoveRaw * 100))

        if not mcTradeRows:
            if verbose:
                print("    MC augmentation: no trade data available")
            return

        # Cap MC rows at 3x the real data size
        maxMCRows = len(self._X_accum) * 3
        if len(mcTradeRows) > maxMCRows:
            indices = np.random.choice(len(mcTradeRows), maxMCRows, replace=False)
            mcTradeRows = [mcTradeRows[i] for i in indices]
            mcDirLabels = [mcDirLabels[i] for i in indices]
            mcRetLabels = [mcRetLabels[i] for i in indices]

        Xmc = np.array(mcTradeRows)
        yDirMC = np.array(mcDirLabels)
        yRetMC = np.array(mcRetLabels)

        if verbose:
            mcWinPct = np.mean(yDirMC) * 100 if len(yDirMC) > 0 else 0
            print(f"    MC trades collected: {len(Xmc)} | MC win rate: {mcWinPct:.1f}%")

        # -- Self-improvement loop --
        for roundNum in range(1, selfImprovementRounds + 1):
            # Test current model on MC synthetic rows
            if self._scaler is not None and self._directionModel is not None:
                XmcScaled = self._scaler.transform(Xmc)
                predDir = self._directionModel.predict(XmcScaled)

                # Find mistakes: where model prediction disagrees with MC outcome
                mistakes = (predDir != yDirMC)
                numMistakes = int(np.sum(mistakes))
                numCorrect = len(mistakes) - numMistakes
                mistakeRate = numMistakes / max(len(mistakes), 1) * 100

                # Assign weights: mistakes get 2x, correct predictions get 0.5x
                wMC = np.where(mistakes, 2.0, 0.5)

                if verbose:
                    print(f"    Round {roundNum}/{selfImprovementRounds}: "
                          f"{numCorrect} correct, {numMistakes} mistakes "
                          f"({mistakeRate:.1f}% error) on MC data")
            else:
                wMC = np.full(len(Xmc), 0.5)

            # Combine real training data + MC synthetic data
            Xcombined = np.vstack([self._X_accum, Xmc])
            yDirCombined = np.concatenate([self._yDir_accum, yDirMC])
            yRetCombined = np.concatenate([self._yRet_accum, yRetMC])
            wCombined = np.concatenate([self._weights_accum, wMC])

            # Re-fit models on combined dataset
            metrics = self._fitModels(Xcombined, yDirCombined, yRetCombined,
                                      wCombined, testSize=0.2, verbose=False)
            self._trained = (self._scaler is not None and self._directionModel is not None)
            self._metrics = metrics  # keep metrics current (quality gate uses these)

            if verbose and metrics and metrics.directionAccuracy > 0:
                total = len(self._X_accum) + len(Xmc)
                print(f"    Post-round-{roundNum} accuracy: "
                      f"{metrics.directionAccuracy*100:.1f}% "
                      f"(on {total} total rows)")

        if verbose:
            print(f"    MC self-improvement complete ({selfImprovementRounds} rounds)")

    def augmentFromPhase2(
        self,
        phase2Results: 'MonteCarloResults',
        verbose: bool = True,
    ):
        """
        Learn from Phase 2 MC results (full-pipeline simulation).

        Phase 2 paths include ML confidence, sentiment, and TradingDecider
        blending — so mistakes here indicate the WHOLE system is wrong,
        not just the pattern.  This makes the correction signal much more
        valuable than Phase 1 augmentation.

        Process:
          1. Extract trades that had ML signals (hasMLSignal=True).
          2. Build synthetic feature rows from trade entry/exit data.
          3. Where the system was wrong (lost money), upweight 3×.
          4. Where the system was right, upweight 0.5×.
          5. Retrain on combined real + Phase 2 data.

        Parameters
        ----------
        phase2Results : MonteCarloResults
            Results from MCMCSimulator.simulateFullPipeline().
        verbose : bool
            Print progress.
        """
        if not self._trained or self._X_accum is None:
            return

        p2Rows: List[np.ndarray] = []
        p2DirLabels: List[int] = []
        p2RetLabels: List[float] = []
        p2Weights: List[float] = []

        for path in phase2Results.paths:
            if not path.trades:
                continue
            for trade in path.trades:
                entryPrice = trade.get('entryPrice', 0)
                if entryPrice <= 0:
                    continue

                # Only learn from trades where ML was involved
                hasML = trade.get('hasMLSignal', False)
                returnPct = trade.get('returnPct', 0.0)
                successful = trade.get('successful', False)
                mlConf = trade.get('mlConfidence', 0.0)

                row = np.zeros(len(self._featureNames))
                priceMoveRaw = returnPct / 100.0
                for fIdx, fName in enumerate(self._featureNames):
                    if fName == 'ret_1':
                        row[fIdx] = priceMoveRaw
                    elif fName == 'ret_3':
                        row[fIdx] = priceMoveRaw * 0.8
                    elif fName == 'ret_5':
                        row[fIdx] = priceMoveRaw * 0.6
                    elif fName == 'is_bullish':
                        row[fIdx] = 1.0 if priceMoveRaw > 0 else 0.0
                    elif fName == 'body_ratio':
                        row[fIdx] = min(1.0, abs(priceMoveRaw) * 20)
                    elif fName == 'vol_ratio':
                        row[fIdx] = 1.0
                    elif fName == 'rsi_14':
                        row[fIdx] = 60.0 if priceMoveRaw > 0 else 40.0
                    elif fName.startswith('vol_'):
                        row[fIdx] = abs(priceMoveRaw) * 5
                    elif fName == 'boll_pos':
                        row[fIdx] = 0.7 if priceMoveRaw > 0 else 0.3
                    elif 'sentiment' in fName:
                        # Inject mild sentiment signal aligned with outcome
                        if successful:
                            row[fIdx] = 0.2 if priceMoveRaw > 0 else -0.2
                        else:
                            row[fIdx] = -0.1 if priceMoveRaw > 0 else 0.1

                p2Rows.append(row)
                p2DirLabels.append(1 if successful else 0)
                p2RetLabels.append(returnPct)

                # Weight: ML-involved mistakes get 3×, correct get 0.5×
                # Non-ML trades get lower weight (1×) since they're
                # pattern-only and already covered by Phase 1
                if hasML:
                    p2Weights.append(3.0 if not successful else 0.5)
                else:
                    p2Weights.append(1.5 if not successful else 0.3)

        if not p2Rows:
            if verbose:
                print("    Phase 2 augmentation: no trade data")
            return

        # Cap at 2× real data size
        maxRows = len(self._X_accum) * 2
        if len(p2Rows) > maxRows:
            indices = np.random.choice(len(p2Rows), maxRows, replace=False)
            p2Rows = [p2Rows[i] for i in indices]
            p2DirLabels = [p2DirLabels[i] for i in indices]
            p2RetLabels = [p2RetLabels[i] for i in indices]
            p2Weights = [p2Weights[i] for i in indices]

        Xp2 = np.array(p2Rows)
        yDirP2 = np.array(p2DirLabels)
        yRetP2 = np.array(p2RetLabels)
        wP2 = np.array(p2Weights)

        if verbose:
            winPct = np.mean(yDirP2) * 100 if len(yDirP2) > 0 else 0
            mlTrades = sum(1 for t in phase2Results.paths
                          for tr in t.trades if tr.get('hasMLSignal'))
            print(f"    Phase 2 trades: {len(Xp2)} "
                  f"({mlTrades} ML-involved) | Win rate: {winPct:.1f}%")

        # Combine real + Phase 2 data and retrain
        Xcombined = np.vstack([self._X_accum, Xp2])
        yDirCombined = np.concatenate([self._yDir_accum, yDirP2])
        yRetCombined = np.concatenate([self._yRet_accum, yRetP2])
        wCombined = np.concatenate([self._weights_accum, wP2])

        metrics = self._fitModels(
            Xcombined, yDirCombined, yRetCombined,
            wCombined, testSize=0.2, verbose=False,
        )
        self._trained = (self._scaler is not None
                         and self._directionModel is not None)
        self._metrics = metrics

        if verbose and metrics and metrics.directionAccuracy > 0:
            print(f"    Post-Phase2 accuracy: "
                  f"{metrics.directionAccuracy*100:.1f}% "
                  f"(on {len(Xcombined)} total rows)")

    def predict(self, df: pd.DataFrame, idx: int = -1) -> StockPrediction:
        """
        Generate a prediction for a single period in *df*.

        Args:
            df:  Full OHLCV DataFrame (needs enough history for indicators)
            idx: Row index to predict at (-1 = latest)

        Returns:
            StockPrediction with signal, confidence, expected return, regime.
        """
        if not self._trained:
            return StockPrediction()

        features = self._computeFeatures(df)
        if features is None or len(features) == 0:
            return StockPrediction()

        row = features.iloc[idx].values.reshape(1, -1)
        rowScaled = self._scaler.transform(row)

        # Direction
        dirProba = self._directionModel.predict_proba(rowScaled)[0]
        upProb = dirProba[1] if len(dirProba) > 1 else 0.5

        # Return
        expectedRet = float(self._returnModel.predict(rowScaled)[0])

        # Regime
        regime = self._detectRegime(df, idx)

        # Signal decision
        signal, confidence, posSize = self._decideSignal(
            upProb, expectedRet, regime
        )

        # Which patterns triggered at this point?
        triggeringPats = self._whichPatternsTriggered(df, idx)
        patternConf, patternSig = self._computePatternSignal(df, idx)

        # Sentiment signal for this period
        sentScore, sentSig, sentConf = self._computeSentimentSignal(df, idx)

        pred = StockPrediction(
            timestamp=df.index[idx] if idx < len(df) else None,
            signal=signal,
            confidence=confidence,
            expectedReturn=expectedRet,
            positionSize=posSize,
            regime=regime,
            triggeringPatterns=triggeringPats,
            patternConfidence=patternConf,
            patternSignal=patternSig,
            sentimentScore=sentScore,
            sentimentSignal=sentSig,
            sentimentConfidence=sentConf,
        )
        return pred

    def predictBatch(self, df: pd.DataFrame) -> List[StockPrediction]:
        """Generate predictions for every valid row in *df*."""
        if not self._trained:
            return []

        # Ensure earnings dates are loaded so feature pipeline has them
        self.loadEarningsDates()

        features = self._computeFeatures(df)
        if features is None or len(features) == 0:
            return []

        Xsf = self._scaler.transform(features.values)
        dirProba = self._directionModel.predict_proba(Xsf)
        retPred = self._returnModel.predict(Xsf)

        predictions = []
        for i in range(len(features)):
            upProb = dirProba[i][1] if len(dirProba[i]) > 1 else 0.5
            expectedRet = float(retPred[i])
            regime = self._detectRegime(df, i)
            signal, confidence, posSize = self._decideSignal(
                upProb, expectedRet, regime
            )

            # Pattern and sentiment signals — same logic as single predict()
            patternConf, patternSig = self._computePatternSignal(df, i)
            sentScore, sentSig, sentConf = self._computeSentimentSignal(df, i)

            pred = StockPrediction(
                timestamp=features.index[i],
                signal=signal,
                confidence=confidence,
                expectedReturn=expectedRet,
                positionSize=posSize,
                regime=regime,
                patternConfidence=patternConf,
                patternSignal=patternSig,
                sentimentScore=sentScore,
                sentimentSignal=sentSig,
                sentimentConfidence=sentConf,
            )
            predictions.append(pred)
        return predictions

    def evaluate(self) -> Optional[ModelMetrics]:
        """Return the metrics from the most recent training run."""
        return self._metrics

    def getFeatureImportances(self, topN: int = 15) -> List[Tuple[str, float]]:
        """Top-N feature importances from the direction model."""
        if self._directionModel is None:
            return []
        importances = self._directionModel.feature_importances_
        ranked = sorted(
            zip(self._featureNames, importances),
            key=lambda x: x[1], reverse=True
        )
        return ranked[:topN]

    @property
    def isTrained(self) -> bool:
        return self._trained

    # =====================================================================
    # INTERNAL — feature engineering
    # =====================================================================

    def _computeFeatures(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build the full feature matrix for *df*.  Returns a DataFrame with
        one row per valid period (rows that need look-back are dropped).
        Uses caching to avoid recomputing when called on the same data.
        """
        if len(df) < 30:
            return None

        # Cache key: (len, first timestamp, last timestamp, has_sentiment)
        hasSent = self._sentimentSeries is not None and len(self._sentimentSeries) > 0
        cacheKey = (len(df), df.index[0], df.index[-1], hasSent)
        if self._featureCacheKey == cacheKey and self._featureCache is not None:
            return self._featureCache.copy()

        feat = pd.DataFrame(index=df.index)

        closes = df['close'].values.astype(float)
        opens  = df['open'].values.astype(float)
        highs  = df['high'].values.astype(float)
        lows   = df['low'].values.astype(float)
        vols   = df['volume'].values.astype(float)

        # ---- 1. Returns at multiple horizons ----------------------------
        for h in [1, 3, 5, 10, 20]:
            feat[f'ret_{h}'] = pd.Series(closes, index=df.index).pct_change(h)

        # ---- 2. Volatility (rolling std of 1-period returns) ------------
        ret1 = pd.Series(closes, index=df.index).pct_change()
        for w in [5, 10, 20]:
            feat[f'vol_{w}'] = ret1.rolling(w).std()

        # ---- 3. RSI (14-period) -----------------------------------------
        feat['rsi_14'] = self._rsi(closes, 14).values

        # ---- 4. Moving-average crossovers (binary) ----------------------
        sma5  = pd.Series(closes, index=df.index).rolling(5).mean()
        sma20 = pd.Series(closes, index=df.index).rolling(20).mean()
        sma50 = pd.Series(closes, index=df.index).rolling(50).mean()
        feat['sma5_above_20']  = (sma5 > sma20).astype(float)
        feat['sma20_above_50'] = (sma20 > sma50).astype(float)
        feat['sma5_dist']  = (closes - sma5.values)  / (sma5.values  + 1e-9)
        feat['sma20_dist'] = (closes - sma20.values) / (sma20.values + 1e-9)

        # ---- 5. Bollinger band position ---------------------------------
        bb_mid  = sma20
        bb_std  = ret1.rolling(20).std() * closes
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = bb_upper - bb_lower
        feat['boll_pos'] = np.where(bb_width > 0,
                                    (closes - bb_lower.values) / (bb_width.values + 1e-9),
                                    0.5)

        # ---- 6. Volume ratio (current / rolling mean) -------------------
        avgVol20 = pd.Series(vols, index=df.index).rolling(20).mean()
        feat['vol_ratio'] = vols / (avgVol20.values + 1e-9)

        # ---- 7. Candle features -----------------------------------------
        totalRange = highs - lows
        body = np.abs(closes - opens)
        feat['body_ratio'] = np.where(totalRange > 0, body / totalRange, 0)
        feat['upper_wick'] = np.where(
            totalRange > 0,
            (highs - np.maximum(opens, closes)) / totalRange, 0
        )
        feat['lower_wick'] = np.where(
            totalRange > 0,
            (np.minimum(opens, closes) - lows) / totalRange, 0
        )
        feat['is_bullish'] = (closes > opens).astype(float)

        # ---- 8. Sentiment features (from SentimentAnalyzer) ---------------
        # Aligned daily sentiment scores are merged onto the intraday/daily
        # OHLCV index.  Features capture score level, trend, and momentum.
        if self._sentimentSeries is not None and len(self._sentimentSeries) > 0:
            # Normalise the sentiment index to date (handles DatetimeIndex)
            sentIdx = self._sentimentSeries.copy()
            # Strip timezone so dtypes match the (usually tz-naive) OHLCV index
            if hasattr(sentIdx.index, 'tz') and sentIdx.index.tz is not None:
                sentIdx.index = sentIdx.index.tz_localize(None)
            if hasattr(sentIdx.index, 'normalize'):
                sentIdx.index = sentIdx.index.normalize()  # strip time component
            # Reindex onto the df date (forward-fill gaps, zero-fill leading NaN)
            dfDates = df.index.normalize() if hasattr(df.index, 'normalize') else df.index
            if hasattr(dfDates, 'tz') and dfDates.tz is not None:
                dfDates = dfDates.tz_localize(None)
            sentAligned = sentIdx.reindex(dfDates, method='ffill').fillna(0.0)
            sentAligned.index = df.index

            feat['sentiment_score']    = sentAligned.values
            feat['sentiment_ma5']      = sentAligned.rolling(5, min_periods=1).mean().values
            feat['sentiment_change']   = sentAligned.diff(1).fillna(0.0).values
            feat['sentiment_strength'] = sentAligned.abs().rolling(3, min_periods=1).mean().values
            # ---- Raw sentiment as continuous feature for ML to learn thresholds ---
            feat['sentiment_signed_sq'] = np.sign(sentAligned.values) * sentAligned.values**2
            feat['sentiment_regime']    = (sentAligned.rolling(10, min_periods=1).mean() > 0).astype(float).values

        # ---- 9. Strategy features (from StrategyEngine) -------------------
        if self._strategyFeatures is not None and len(self._strategyFeatures) > 0:
            for si, sv in enumerate(self._strategyFeatures):
                feat[f'strategy_{si}'] = float(sv)  # broadcast scalar to all rows

        # ---- 10. Connected stock metadata features -------------------------
        if self._stockMeta:
            pType = self._stockMeta.get('portfolioType', 'manual')
            feat['is_auto_stock'] = 1.0 if pType == 'automatic' else 0.0

            connType = self._stockMeta.get('connectionType', '')
            for ct in ['supplier', 'customer', 'competitor', 'sector_peer', 'related']:
                feat[f'conn_{ct}'] = 1.0 if connType == ct else 0.0

            feat['auto_score'] = float(self._stockMeta.get('autoScore', 0.0) or 0.0)

            parentCorr = float(self._stockMeta.get('parentCorrelation', 0.0) or 0.0)
            feat['parent_correlation'] = parentCorr

            parentPerf = float(self._stockMeta.get('parentRecentPerf', 0.0) or 0.0)
            feat['parent_recent_perf'] = parentPerf

        # ---- 11. Earnings proximity features --------------------------------
        # These give the model explicit awareness of earnings risk:
        #   earnings_days_away: trading days until next earnings (capped at 30, 0 if past)
        #   earnings_window_pre: 1 in the 5 days BEFORE earnings, else 0
        #   earnings_window_post: 1 in the 5 days AFTER earnings, else 0
        #   earnings_vol_regime: rolling 5-day realised vol spike around earnings
        if self._earningsDates:
            earningsDaysAway  = np.zeros(len(df), dtype=float)
            earningsWindowPre = np.zeros(len(df), dtype=float)
            earningsWindowPost = np.zeros(len(df), dtype=float)

            dfDates = df.index.normalize() if hasattr(df.index, 'normalize') else df.index
            if hasattr(dfDates, 'tz') and dfDates.tz is not None:
                dfDates = dfDates.tz_localize(None)

            for i, barDate in enumerate(dfDates):
                barDate = pd.Timestamp(barDate)
                # Find the nearest upcoming earnings
                future = [e for e in self._earningsDates if e >= barDate]
                past   = [e for e in self._earningsDates if e < barDate]

                if future:
                    daysToNext = (min(future) - barDate).days
                    earningsDaysAway[i] = min(daysToNext, 30) / 30.0  # normalise 0-1
                    if daysToNext <= 5:
                        earningsWindowPre[i] = 1.0
                else:
                    earningsDaysAway[i] = 1.0  # no known upcoming earnings → far away

                if past:
                    daysSinceLast = (barDate - max(past)).days
                    if daysSinceLast <= 5:
                        earningsWindowPost[i] = 1.0

            feat['earnings_days_away']  = earningsDaysAway
            feat['earnings_window_pre'] = earningsWindowPre
            feat['earnings_window_post'] = earningsWindowPost
            feat['earnings_vol_pre'] = feat['vol_5'].fillna(0) * earningsWindowPre
            feat['earnings_vol_post'] = feat['vol_5'].fillna(0) * earningsWindowPost

        # Drop rows with NaN (from look-back windows)
        feat.dropna(inplace=True)

        # Store in cache
        self._featureCacheKey = cacheKey
        self._featureCache = feat.copy()

        return feat

    def _buildDataset(self, df: pd.DataFrame, forwardPeriods: int):
        """Build (X, yDirection, yReturn, featureNames, sampleWeights) from
        historical df.  Samples near earnings are up-weighted so the model
        learns to handle earnings volatility rather than being dominated by
        quiet periods."""
        features = self._computeFeatures(df)
        if features is None or len(features) < forwardPeriods + 30:
            return np.array([]), np.array([]), np.array([]), [], np.array([])

        closes = df['close'].reindex(features.index).values.astype(float)

        forwardReturns = np.full(len(closes), np.nan)
        for i in range(len(closes) - forwardPeriods):
            forwardReturns[i] = (closes[i + forwardPeriods] - closes[i]) / (closes[i] + 1e-9) * 100

        valid = ~np.isnan(forwardReturns)
        X = features.values[valid]
        yRet = forwardReturns[valid]
        yDir = (yRet > 0).astype(int)

        # Earnings-aware sample weights: up-weight rows near earnings 2×
        # so the model sees earnings-period patterns as equally important
        # as quiet-period patterns despite them being less frequent.
        weights = np.ones(len(yDir))
        if 'earnings_window_pre' in features.columns or 'earnings_window_post' in features.columns:
            preFeat  = features.columns.get_loc('earnings_window_pre') if 'earnings_window_pre' in features.columns else None
            postFeat = features.columns.get_loc('earnings_window_post') if 'earnings_window_post' in features.columns else None
            for idx in range(len(weights)):
                near = False
                if preFeat  is not None and X[idx, preFeat]  > 0.5:
                    near = True
                if postFeat is not None and X[idx, postFeat] > 0.5:
                    near = True
                if near:
                    weights[idx] = 2.0

        return X, yDir, yRet, list(features.columns), weights

    # =====================================================================
    # INTERNAL — model fitting
    # =====================================================================

    def _fitModels(self, X, yDir, yRet, weights, testSize, verbose):
        """Train direction + return models, compute metrics."""
        if len(np.unique(yDir)) < 2:
            if verbose:
                print("    Only one class present — skipping direction model")
            return ModelMetrics()

        Xtr, Xte, yDirTr, yDirTe, yRetTr, yRetTe, wTr, wTe = train_test_split(
            X, yDir, yRet, weights, test_size=testSize, shuffle=False
        )

        self._scaler = StandardScaler()
        XtrS = self._scaler.fit_transform(Xtr)
        XteS = self._scaler.transform(Xte)

        # Direction classifier
        self._directionModel = GradientBoostingClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        self._directionModel.fit(XtrS, yDirTr, sample_weight=wTr)
        dirAcc = accuracy_score(yDirTe, self._directionModel.predict(XteS))

        # Return regressor
        self._returnModel = GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        self._returnModel.fit(XtrS, yRetTr, sample_weight=wTr)
        retPred = self._returnModel.predict(XteS)
        retRMSE = float(np.sqrt(np.mean((retPred - yRetTe) ** 2)))

        # Feature importances (direction)
        importances = self._directionModel.feature_importances_
        topFeat = sorted(
            zip(self._featureNames, importances),
            key=lambda x: x[1], reverse=True
        )[:10]

        metrics = ModelMetrics(
            directionAccuracy=dirAcc,
            returnRMSE=retRMSE,
            trainSamples=len(Xtr),
            testSamples=len(Xte),
            topFeatures=topFeat,
        )

        if verbose:
            print(f"    Direction accuracy : {dirAcc * 100:.1f}%")
            print(f"    Return RMSE        : {retRMSE:.4f}%")
            print(f"    Train/test samples : {len(Xtr)}/{len(Xte)}")
            print(f"    Top features       : "
                  + ", ".join(f"{n}={v:.3f}" for n, v in topFeat[:5]))

        return metrics

    # =====================================================================
    # INTERNAL — signal decision logic
    # =====================================================================

    def _decideSignal(self, upProb: float, expectedReturn: float,
                      regime: MarketRegime
                      ) -> Tuple[TradingSignal, float, float]:
        """
        Convert raw model outputs into a trading signal.

        Asymmetric thresholds: markets have a long-term upward bias, so
        BUY requires less evidence than SELL (shorting).  Regime-aware
        adjustments further bias toward staying invested in uptrends.

        Rules:
          - BUY  : P(up) > buyThresh AND expectedReturn > buyRetThresh
          - SELL : P(up) < sellThresh AND expectedReturn < -sellRetThresh
          - HOLD : otherwise (maintain current position)
        """
        # ── Base thresholds (BALANCED: profit from ups AND downs) ──
        buyThresh     = 0.55    # require solid evidence to go long
        sellThresh    = 0.45    # require solid evidence to go short
        buyRetThresh  = 0.2     # need meaningful expected return (+0.2%)
        sellRetThresh = 0.2     # symmetric threshold for shorts

        # ── Regime adjustments ──
        if regime == MarketRegime.TRENDING_UP:
            buyThresh     = 0.52   # easy to buy in uptrend
            sellThresh    = 0.44   # still allow shorts when model is bearish
            buyRetThresh  = 0.1    # low bar for longs in uptrend
            sellRetThresh = 0.3    # moderate conviction to short uptrend (was 0.6 — too strict)
        elif regime == MarketRegime.TRENDING_DOWN:
            buyThresh     = 0.60   # harder to buy in downtrend
            sellThresh    = 0.42   # easy to short in downtrend
            buyRetThresh  = 0.4    # need conviction to go long in downtrend
            sellRetThresh = 0.1    # very easy to trigger shorts in downtrend
        elif regime == MarketRegime.VOLATILE:
            buyThresh     = 0.58   # stricter in high volatility
            sellThresh    = 0.42   # symmetric
            buyRetThresh  = 0.3
            sellRetThresh = 0.3

        if upProb > buyThresh and expectedReturn > buyRetThresh:
            confidence = min(1.0, (upProb - 0.5) * 4.0)   # 4.0 vs 3.5: more leverage
            confidence *= self._qualityFactor()
            # Smooth power curve — no fixed floor.  conf^0.6:
            #   0.05→0.16 | 0.30→0.55 | 0.60→0.74 | 1.0→1.00
            posSize = confidence ** 0.6
            if confidence < 0.05:
                return TradingSignal.HOLD, 0.0, 0.0
            return TradingSignal.BUY, confidence, posSize

        if upProb < sellThresh and expectedReturn < -sellRetThresh:
            confidence = min(1.0, (0.5 - upProb) * 4.0)   # 4.0 vs 3.5: more leverage
            confidence *= self._qualityFactor()
            posSize = confidence ** 0.6
            if confidence < 0.05:
                return TradingSignal.HOLD, 0.0, 0.0
            return TradingSignal.SELL, confidence, posSize

        return TradingSignal.HOLD, 0.0, 0.0

    def _qualityFactor(self) -> float:
        """Scale confidence by model accuracy.

        Maps direction accuracy to a 0–1 multiplier:
          acc ≤ 35%  → 0.0  (no trades — model is worse than chance)
          acc  = 42% → 0.35
          acc  = 50% → 0.75
          acc ≥ 55% → 1.0  (full confidence)

        This prevents truly poor models from generating trades while
        still allowing imperfect models to contribute at reduced weight.
        """
        # Use ORIGINAL (pre-augmentation) accuracy.  MC augmentation
        # inflates accuracy with easy-to-classify synthetic data, which
        # would let bad models pass the quality gate.
        m = self._originalMetrics or self._metrics
        if m is None:
            return 1.0   # no metrics yet (shouldn't happen post-training)
        acc = m.directionAccuracy
        if acc <= 0.35:
            return 0.0
        return min(1.0, (acc - 0.35) / 0.20)

    def _detectRegime(self, df: pd.DataFrame, idx: int) -> MarketRegime:
        """Simple regime detection based on recent volatility & trend."""
        lookback = 20
        start = max(0, idx - lookback)
        if start >= idx or idx >= len(df):
            return MarketRegime.LOW_VOL

        closes = df['close'].values[start:idx + 1].astype(float)
        if len(closes) < 5:
            return MarketRegime.LOW_VOL

        returns = np.diff(closes) / (closes[:-1] + 1e-9)
        vol = float(np.std(returns))
        trend = float(np.mean(returns))

        if vol > 0.025:
            return MarketRegime.VOLATILE
        if trend > 0.0005:   # ~12.6% annualized → uptrend
            return MarketRegime.TRENDING_UP
        if trend < -0.0005:  # ~-12.6% annualized → downtrend
            return MarketRegime.TRENDING_DOWN
        if vol < 0.008:
            return MarketRegime.LOW_VOL
        return MarketRegime.MEAN_REVERTING

    # =====================================================================
    # INTERNAL — pattern matching helpers
    # =====================================================================

    def _patternSimilarity(self, df: pd.DataFrame,
                           pattern: PatternChromosome,
                           avgVolume: float) -> pd.Series:
        """
        Compute a soft pattern-match similarity score (0-1) at every row.

        For each row *t*, we look back over the pattern's historical genes
        and compute how closely the recent candles match.  The score is the
        average per-gene similarity across price, volume, candle-type, and
        body-ratio dimensions.

        Vectorized implementation using numpy array operations.
        """
        pLen = len(pattern.genes)
        histLen = pLen - 1  # last gene is the prediction gene
        scores = np.zeros(len(df))

        closes = df['close'].values.astype(float)
        opens  = df['open'].values.astype(float)
        highs  = df['high'].values.astype(float)
        lows   = df['low'].values.astype(float)
        vols   = df['volume'].values.astype(float)

        n = len(df)
        forgiveness = self.forgiveness

        # Pre-extract gene attributes into arrays
        genePriceExp = np.array([pattern.genes[g].expectedPriceChangePct for g in range(histLen)])
        geneVolumeExp = np.array([pattern.genes[g].expectedVolumePct for g in range(histLen)])
        geneCandleType = np.array([
            ['BULLISH', 'BEARISH', 'DOJI'].index(pattern.genes[g].candleType.value)
            for g in range(histLen)
        ])
        geneMinBody = np.array([pattern.genes[g].minBodyPct for g in range(histLen)])

        # For each gene offset, compute similarity across ALL valid rows at once
        # Each gene g corresponds to looking at position (t - histLen + 1 + g)
        # relative to baseline at position (t - histLen)
        
        # Collect per-gene similarity arrays (each of shape [n])
        geneSims = np.zeros((histLen, n))

        for g in range(histLen):
            # For row t, baseline is at t - histLen, gene candle is at t - histLen + 1 + g
            # Valid range: t from histLen to n-1
            # baseline indices: histLen - histLen = 0 to n-1 - histLen
            # candle indices: 0 + 1 + g = 1+g to n-1 - histLen + 1 + g

            baselineIndices = np.arange(0, n - histLen)
            candleIndices = baselineIndices + 1 + g
            tIndices = baselineIndices + histLen  # corresponding t values

            baselinePrices = closes[baselineIndices]
            # Skip zeros
            validMask = baselinePrices != 0

            # Price similarity
            actualPctChg = np.where(validMask,
                                     (closes[candleIndices] - baselinePrices) / (baselinePrices + 1e-15),
                                     0)
            priceDiff = np.abs(actualPctChg - genePriceExp[g])
            priceSim = np.maximum(0, 1.0 - priceDiff / (forgiveness + 1e-9))

            # Volume similarity
            volRatio = vols[candleIndices] / (avgVolume + 1e-9)
            volDiff = np.abs(volRatio - geneVolumeExp[g])
            volSim = np.maximum(0, 1.0 - volDiff / 1.0)

            # Candle type similarity
            bodyDir = closes[candleIndices] - opens[candleIndices]
            ct = geneCandleType[g]
            if ct == 0:  # BULLISH
                candleSim = np.where(bodyDir > 0, 1.0, 0.2)
            elif ct == 1:  # BEARISH
                candleSim = np.where(bodyDir < 0, 1.0, 0.2)
            else:  # DOJI
                tr = highs[candleIndices] - lows[candleIndices]
                candleSim = np.where((tr > 0) & (np.abs(bodyDir) / (tr + 1e-15) < 0.15), 1.0, 0.2)

            # Body ratio similarity
            tr = highs[candleIndices] - lows[candleIndices]
            bodyPct = np.where(tr > 0, np.abs(bodyDir) / tr, 0)
            bodySim = np.maximum(0, 1.0 - np.abs(bodyPct - geneMinBody[g]))

            # Weighted average
            geneSim = 0.4 * priceSim + 0.25 * volSim + 0.2 * candleSim + 0.15 * bodySim
            geneSim = np.where(validMask, geneSim, 0)

            # Place into the correct t positions
            geneSims[g, tIndices] = geneSim

        # Average across genes for each t
        # Only valid for t >= histLen
        scores[histLen:] = geneSims[:, histLen:].mean(axis=0)

        return pd.Series(scores, index=df.index)

    def _whichPatternsTriggered(self, df: pd.DataFrame, idx: int,
                                threshold: float = 0.7) -> List[int]:
        """Return indices of patterns with similarity ≥ threshold at *idx*."""
        avgVol = float(df['volume'].mean())
        triggered = []
        for pIdx, pattern in enumerate(self._patterns):
            sims = self._patternSimilarity(df, pattern, avgVol)
            if idx < len(sims) and sims.iloc[idx] >= threshold:
                triggered.append(pIdx)
        return triggered

    def _computePatternSignal(self, df: pd.DataFrame, idx: int,
                             threshold: float = 0.7) -> Tuple[float, TradingSignal]:
        """
        Compute independent pattern signal by analyzing which patterns trigger
        and their prediction genes (bullish/bearish).
        
        Returns:
            (confidence, signal) where confidence is 0-1 and signal is BUY/SELL/HOLD
        """
        if not self._patterns:
            return 0.0, TradingSignal.HOLD
            
        avgVol = float(df['volume'].mean())
        bullishScore = 0.0
        bearishScore = 0.0
        totalWeight = 0.0
        
        for pattern in self._patterns:
            # Check if pattern triggers at this index
            sims = self._patternSimilarity(df, pattern, avgVol)
            if idx >= len(sims):
                continue
                
            similarity = sims.iloc[idx]
            if similarity < threshold:
                continue
            
            # Pattern triggered! Check its prediction gene
            predGene = pattern.genes[-1]  # Last gene is the prediction
            expectedChange = predGene.expectedPriceChangePct
            
            # Weight by pattern fitness * similarity (patterns with higher MC fitness matter more)
            weight = pattern.fitness * similarity
            
            if expectedChange > 0:
                bullishScore += weight
            else:
                bearishScore += abs(weight)
            
            totalWeight += weight
        
        if totalWeight == 0:
            return 0.0, TradingSignal.HOLD
        
        # Normalize scores
        bullishScore /= totalWeight
        bearishScore /= totalWeight
        
        # Compute net score and confidence
        netScore = bullishScore - bearishScore  # -1 to +1
        confidence = abs(netScore)  # 0 to 1
        
        # Determine signal (require at least 20% confidence to trade)
        if netScore > 0.20:
            return confidence, TradingSignal.BUY
        elif netScore < -0.20:
            return confidence, TradingSignal.SELL
        else:
            return confidence, TradingSignal.HOLD

    # =====================================================================
    # INTERNAL — technical indicator helpers
    # =====================================================================

    def _computeSentimentSignal(
        self, df: pd.DataFrame, idx: int
    ) -> Tuple[float, TradingSignal, float]:
        """
        Look up the sentiment score for the date at *idx* from the stored
        ``_sentimentSeries`` and convert it to a (score, signal, confidence).

        **ML-learned thresholds**: The raw sentiment score is passed as a
        continuous feature to the ML model, which learns the optimal
        buy/sell thresholds from the combination of news data and actual
        price movements.  The signal returned here is an *initial hint*
        using adaptive thresholds calibrated from the sentiment
        distribution, NOT a hardcoded value.

        Returns
        -------
        (sentimentScore, TradingSignal, confidence)
          sentimentScore  : raw ensemble score in [-1, +1]
          TradingSignal   : BUY / SELL / HOLD based on adaptive threshold
          confidence      : abs(score) clipped to [0, 1]
        """
        if self._sentimentSeries is None or len(self._sentimentSeries) == 0:
            return 0.0, TradingSignal.HOLD, 0.0

        # Lazily build a normalised (date-only index) copy once per series
        if not hasattr(self, '_sentimentNormalized') or self._sentimentNormalized is None:
            sentIdx = self._sentimentSeries.copy()
            if hasattr(sentIdx.index, 'tz') and sentIdx.index.tz is not None:
                sentIdx.index = sentIdx.index.tz_localize(None)
            if hasattr(sentIdx.index, 'normalize'):
                sentIdx.index = sentIdx.index.normalize()
            self._sentimentNormalized = sentIdx
            # Calibrate adaptive thresholds from sentiment distribution
            self._calibrateSentimentThresholds(sentIdx)

        try:
            ts = df.index[idx]
            date = ts.normalize() if hasattr(ts, 'normalize') else ts
            if hasattr(date, 'tz') and date.tz is not None:
                date = date.tz_localize(None)
            loc = self._sentimentNormalized.index.get_indexer([date], method='ffill')[0]
            if loc < 0:
                loc = 0
            score = float(self._sentimentNormalized.iloc[loc])
        except Exception:
            return 0.0, TradingSignal.HOLD, 0.0

        confidence = min(1.0, abs(score))

        # Adaptive thresholds: use calibrated values from distribution
        buyThresh = getattr(self, '_sentBuyThreshold', 0.10)
        sellThresh = getattr(self, '_sentSellThreshold', -0.10)

        if score > buyThresh:
            signal = TradingSignal.BUY
        elif score < sellThresh:
            signal = TradingSignal.SELL
        else:
            signal = TradingSignal.HOLD
            confidence *= 0.3  # low confidence in the dead zone, but not zero
        return score, signal, confidence

    def _calibrateSentimentThresholds(self, sentSeries: pd.Series):
        """
        Calibrate buy/sell thresholds from the sentiment score distribution.
        Instead of fixed 0.15/-0.15, use the 65th and 35th percentiles.
        This adapts to news-sparse or news-heavy environments.
        """
        try:
            vals = sentSeries.dropna().values
            if len(vals) < 5:
                self._sentBuyThreshold = 0.10
                self._sentSellThreshold = -0.10
                return
            p65 = float(np.percentile(vals, 65))
            p35 = float(np.percentile(vals, 35))
            # Clamp to reasonable range
            self._sentBuyThreshold = float(np.clip(p65, 0.05, 0.30))
            self._sentSellThreshold = float(np.clip(p35, -0.30, -0.05))
        except Exception:
            self._sentBuyThreshold = 0.10
            self._sentSellThreshold = -0.10

    def updateSentimentSeries(self, series: Optional[pd.Series]):
        """Replace the stored sentiment series (e.g. for backtesting).

        Invalidates the feature cache and the lazy-normalised copy so that
        subsequent ``predictBatch`` / ``predict`` calls use the new data.
        """
        self._sentimentSeries = series
        self._sentimentNormalized = None
        self._featureCache = None
        self._featureCacheKey = None

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avgGain = pd.Series(gains).rolling(period, min_periods=period).mean()
        avgLoss = pd.Series(losses).rolling(period, min_periods=period).mean()

        rs = avgGain / (avgLoss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        return rsi
