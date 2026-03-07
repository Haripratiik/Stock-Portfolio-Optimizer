"""
PortfolioTester — Unified Pipeline Orchestrator

Orchestrates the complete pattern-based trading pipeline:
  1. GA Discovery      → top 10 patterns per timeframe (multi-run)
  2. MC Validation     → composite-score ranking, keeps top 5 per timeframe
  2b. MC Refinement    → iterative improvement via random + ML mutations
  3. Full MC Sim       → fan charts on refined set
  4. Backtesting       → refined patterns on real historical data
  5. Visualisation     → MC fan charts + per-stock + portfolio graphs

One-trade-per-period rule:
  Patterns are sorted by their MC composite score. When multiple patterns
  trigger on the same period window, only the highest-ranked pattern trades.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import io
import base64
import time as _time_module
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import os

# Ensure backend directory is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Load .env at module level so keys are available regardless of how
#    this module is run (directly, imported by LocalAgent, test scripts, etc.)
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
try:
    from dotenv import load_dotenv as _load_dotenv_module
    _load_dotenv_module(dotenv_path=_ENV_PATH, override=True)
except ImportError:
    pass  # python-dotenv not installed — keys must be set in the system environment

from GeneticAlgorithm import (
    GeneticAlgorithmPatternFinder,
    PatternBank,
    PatternChromosome,
    PatternGene,
    CandleType,
    StockDataFetcher,
)
from Backtester import Backtester, BacktestResult
from MCMCSimulator import MCMCSimulator, MonteCarloResults
from PatternRefiner import PatternRefiner
from StockMLModel import StockMLModel, StockPrediction, TradingSignal, ModelMetrics
from PortfolioMLModel import PortfolioMLModel, PortfolioSignal, PortfolioModelMetrics
from TradingDecider import TradingDecider, FinalTradeDecision
from DynamicAllocator import DynamicAllocator, AllocationResult, DynamicReplayResult
from SentimentAnalysis import SentimentAnalyzer, SyntheticSentimentGenerator, MCSyntheticHeadlineGenerator
from PersistenceManager import PersistenceManager, RunResult, StoredPattern, StoredStrategy
from StrategyEngine import StrategyEngine
from ConnectedStockManager import ConnectedStockManager


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StockPipelineResult:
    """Stores the full pipeline output for a single stock."""
    symbol: str
    allocation: float                                   # fraction of total fund
    portfolioType: str = 'manual'                       # 'manual' | 'automatic'
    rawPatternBank: PatternBank = None                  # GA output (≤10 per TF)
    refinedPatternBank: PatternBank = None              # MC-filtered (≤5 per TF)
    mcResults: Dict[str, MonteCarloResults] = None      # interval → MC results
    backtestResults: Dict[str, BacktestResult] = None   # interval → backtest
    patternRankings: Dict[str, List[Tuple[PatternChromosome, float]]] = None
    stockMLModel: Optional[StockMLModel] = None         # per-stock ML model
    stockPredictions: Optional[List[StockPrediction]] = None  # latest predictions

    def __post_init__(self):
        if self.mcResults is None:
            self.mcResults = {}
        if self.backtestResults is None:
            self.backtestResults = {}
        if self.patternRankings is None:
            self.patternRankings = {}


# =============================================================================
# PortfolioTester
# =============================================================================

class PortfolioTester:
    """
    End-to-end pipeline: GA → MC validation → Backtest → Graphs.

    Usage::

        tester = PortfolioTester(
            totalFund=100_000,
            stocks={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3},
        )
        tester.run(verbose=True)
    """

    def __init__(self,
                 totalFund: float = 100_000,
                 stocks: Dict[str, float] = None,
                 # GA settings
                 intervals: List[Tuple[str, str]] = None,
                 patternLengths: List[int] = None,
                 populationSize: int = 500,
                 numGenerations: int = 50,
                 numRunsPerConfig: int = 3,
                 maxPatternsPerInterval: int = 10,
                 gaEarlyStopGenerations: int = 20,
                 gaElitismCount: int = 15,
                 gaMutationRate: float = 0.15,
                 gaCrossoverRate: float = 0.70,
                 gaMinImprovement: float = 0.01,
                 # MC settings
                 mcNumSimulations: int = 1000,
                 mcSimulationPeriods: int = 252,
                 mcMethod: str = 'bootstrap',
                 mcCalibrationPeriod: str = '2y',
                 mcRankSimulations: int = 200,
                 mcTopN: int = 5,
                 mcTargets: List[float] = None,
                 mcRuinThreshold: float = 0.50,
                 # MC Refinement settings
                 mcRefineIterations: int = 0,
                 mcRefineSimsPerCandidate: int = 150,
                 mcRefineRandomMutants: int = 6,
                 mcRefineMLMutants: int = 4,
                 mcRefineMutationStrength: float = 0.3,
                 # Backtest settings (5y backtest, 3y train = robust across regimes)
                 mlTrainPeriodDays: float = 365 * 3,
                 backtestPeriodDays: float = 365 * 5,
                 forgiveness: float = 0.05,
                 useStopLoss: bool = True,
                 useWalkForward: bool = True,
                 useEarningsBlackout: bool = True,
                 useRegimeDetection: bool = True,
                 useCorrelationAdjustment: bool = True,
                 # ML Model settings
                 mlForwardPeriods: int = 5,
                 mlPortfolioForwardPeriods: int = 10,
                 mlCorrWindow: int = 60,
                 mlEnabled: bool = True,
                 # Trading Decider settings
                 deciderPatternWeight: float = 0.80,
                 deciderPortfolioWeight: float = 0.6,
                 deciderMinConfidence: float = 0.30,
                 # Dynamic Fund Allocation settings
                 ifaMinAllocation: float = 0.0,
                 ifaMaxSlotAllocation: float = 0.40,
                 ifaMaxStockAllocation: float = 0.60,
                 ifaShadowThreshold: float = -0.3,
                 ifaRestoreThreshold: float = 3.0,
                 ifaRestoreAllocation: float = 0.03,
                 ifaSmoothingFactor: float = 0.35,
                 ifaUseML: bool = True,
                 ifaMLBlendWeight: float = 0.4,
                 ifaEvalWindowDays: int = 5,
                 ifaMinEvalPeriodDays: int = 10,
                 # Sentiment Analysis settings
                 sentimentEnabled: bool = True,
                 sentimentOpenAIKey: Optional[str] = None,
                 sentimentNewsAPIKey: Optional[str] = None,
                 sentimentFinnhubKey: Optional[str] = None,
                 sentimentAlphaVantageKey: Optional[str] = None,
                 sentimentDecayHalfLife: float = 3.0,
                 sentimentCacheDir: Optional[str] = None):
        """
        Args:
            totalFund:              Total capital ($)
            stocks:                 {symbol: allocation_fraction} — must sum to 1.0
            intervals:              List of (interval, period) tuples for GA
            patternLengths:         Pattern lengths to discover
            populationSize:         GA population size
            numGenerations:         GA generations per run
            numRunsPerConfig:       GA runs per interval×length combo
            maxPatternsPerInterval: Max patterns GA keeps per timeframe (default 10)
            mcNumSimulations:       Full MC simulations per interval (fan chart)
            mcSimulationPeriods:    Periods per MC path (252 ≈ 1 trading year)
            mcMethod:               'bootstrap' or 'gbm'
            mcCalibrationPeriod:    Historical window for MC calibration ('2y', '1y')
            mcRankSimulations:      Lighter MC sims used for per-pattern ranking
            mcTopN:                 Patterns to keep after MC ranking per TF
            mcTargets:              Return targets for probability calc
            mcRuinThreshold:        Loss fraction considered ruin (0.5 = 50%)
            mcRefineIterations:     Refinement loops (0 = disabled)
            mcRefineSimsPerCandidate: MC paths per candidate during refinement
            mcRefineRandomMutants:  Random mutations per pattern per iteration
            mcRefineMLMutants:      ML-guided mutations per pattern per iteration
            mcRefineMutationStrength: Magnitude of random perturbation (0-1)
            mlTrainPeriodDays:      How many days of historical data to TRAIN ML models on
            backtestPeriodDays:     How many days of historical data to BACKTEST on (test dataset)
            forgiveness:            Pattern matching tolerance (±%)
            mlForwardPeriods:       Prediction horizon for per-stock ML model
            mlPortfolioForwardPeriods: Prediction horizon for portfolio ML model
            mlCorrWindow:           Rolling correlation window for portfolio model
            mlEnabled:              Whether to train ML models (stock + portfolio)
            deciderPatternWeight:   Pattern signal weight (0=ignore, 1=equal weight)
            deciderPortfolioWeight: Portfolio model influence (0=ignore, 1=equal weight, 2=double)
            deciderMinConfidence:   Min blended confidence to trigger a trade
            ifaMinAllocation:       Hard floor per slot (0.0 allows ghost mode)
            ifaMaxSlotAllocation:   Hard ceiling per slot (concentration limit)
            ifaMaxStockAllocation:  Hard ceiling per stock across all intervals
            ifaShadowThreshold:     Score below which a slot enters ghost mode
            ifaRestoreThreshold:    Rolling return (%) to restore a ghost slot
            ifaRestoreAllocation:   Initial allocation for a restored slot
            ifaSmoothingFactor:     Allocation change dampener (0=instant, 1=no change)
            ifaUseML:               Whether to use ML-based allocation alongside rules
            ifaMLBlendWeight:       ML blend weight (0=pure rule, 1=pure ML)
            ifaSmoothingFactor:     Allocation change dampener (0=instant, 1=no change)
            ifaEvalWindowDays:      Days between allocation re-evaluation checkpoints
            ifaMinEvalPeriodDays:   Minimum days before first reallocation
            sentimentEnabled:       Whether to run sentiment analysis pipeline
            sentimentOpenAIKey:     OpenAI key for Layer 3 (or OPENAI_API_KEY env var)
            sentimentNewsAPIKey:    NewsAPI key (or NEWSAPI_KEY env var)
            sentimentFinnhubKey:    Finnhub key (or FINNHUB_KEY env var)
            sentimentAlphaVantageKey: Alpha Vantage key (or ALPHAVANTAGE_KEY env var)
            sentimentDecayHalfLife: Sentiment decay half-life in days
            sentimentCacheDir:      Directory to cache scored headlines
        """
        # --- Portfolio ---
        self.totalFund = totalFund
        if stocks is None:
            stocks = {'AAPL': 0.50, 'GOOGL': 0.50}
        totalAlloc = sum(stocks.values())
        if not (0.99 <= totalAlloc <= 1.01):
            raise ValueError(f"Stock allocations must sum to 1.0, got {totalAlloc}")
        normFactor = 1.0 / totalAlloc
        self.stocks = {s: a * normFactor for s, a in stocks.items()}

        # --- GA ---
        # Only include intervals the trading bot can actually execute.
        # 30m was phased out — the bot only supports 1d and 1h, so backtesting
        # on 30m produces results that can never be traded and corrupts P&L
        # summaries by mixing un-tradeable interval returns with real ones.
        self.intervals = intervals or [('1d', '5y'), ('1h', '730d')]
        self.patternLengths = patternLengths or [3, 4, 5, 6, 7, 8, 9, 10]
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.numRunsPerConfig = numRunsPerConfig
        self.maxPatternsPerInterval = maxPatternsPerInterval
        self.gaEarlyStopGenerations = gaEarlyStopGenerations
        self.gaElitismCount = gaElitismCount
        self.gaMutationRate = gaMutationRate
        self.gaCrossoverRate = gaCrossoverRate
        self.gaMinImprovement = gaMinImprovement

        # --- MC ---
        self.mcNumSimulations = mcNumSimulations
        self.mcSimulationPeriods = mcSimulationPeriods
        self.mcMethod = mcMethod
        self.mcCalibrationPeriod = mcCalibrationPeriod
        self.mcRankSimulations = mcRankSimulations
        self.mcTopN = mcTopN
        self.mcTargets = mcTargets or [5, 10, 20, 50, 100]
        self.mcRuinThreshold = mcRuinThreshold

        # --- MC Refinement ---
        self.mcRefineIterations = mcRefineIterations
        self.mcRefineSimsPerCandidate = mcRefineSimsPerCandidate
        self.mcRefineRandomMutants = mcRefineRandomMutants
        self.mcRefineMLMutants = mcRefineMLMutants
        self.mcRefineMutationStrength = mcRefineMutationStrength

        # --- Backtest & Train/Test Split ---
        self.gaDiscoveryPeriodDays = mlTrainPeriodDays  # GA uses same period as ML training (avoid look-ahead bias)
        self.mlTrainPeriodDays = mlTrainPeriodDays
        self.backtestPeriodDays = backtestPeriodDays
        self.forgiveness = forgiveness
        self.useStopLoss = useStopLoss
        self.useWalkForward = useWalkForward
        self.useEarningsBlackout = useEarningsBlackout

        # --- ML Models ---
        self.mlForwardPeriods = mlForwardPeriods
        self.mlPortfolioForwardPeriods = mlPortfolioForwardPeriods
        self.mlCorrWindow = mlCorrWindow
        self.mlEnabled = mlEnabled

        # --- Sentiment ---
        self.sentimentEnabled = sentimentEnabled and mlEnabled
        # Fall back to env var so callers that don't pass the key explicitly still work
        self.sentimentOpenAIKey = sentimentOpenAIKey or os.environ.get('OPENAI_API_KEY') or None
        self.sentimentNewsAPIKey = sentimentNewsAPIKey or os.environ.get('NEWSAPI_KEY') or None
        self.sentimentFinnhubKey = sentimentFinnhubKey
        self.sentimentAlphaVantageKey = sentimentAlphaVantageKey
        self.sentimentDecayHalfLife = sentimentDecayHalfLife
        self.sentimentCacheDir = sentimentCacheDir

        # --- Trading Decider ---
        self.tradingDecider = TradingDecider(
            patternWeight=deciderPatternWeight,
            portfolioWeight=deciderPortfolioWeight,
            minConfidence=deciderMinConfidence,
            useRegimeDetection=useRegimeDetection,
            useCorrelationAdjustment=useCorrelationAdjustment,
        )

        # --- Dynamic Fund Allocation ---
        self.fundAllocator = DynamicAllocator(
            useDynamicReallocation=True,
            minSlotAllocation=ifaMinAllocation,
            maxSlotAllocation=ifaMaxSlotAllocation,
            maxStockAllocation=ifaMaxStockAllocation,
            ghostThreshold=ifaShadowThreshold,
            restoreThreshold=ifaRestoreThreshold,
            restoreAllocation=ifaRestoreAllocation,
            smoothingFactor=ifaSmoothingFactor,
            useML=ifaUseML,
            mlBlendWeight=ifaMLBlendWeight,
            evalWindowDays=ifaEvalWindowDays,
            minEvalPeriodDays=ifaMinEvalPeriodDays,
        )

        # --- Results storage ---
        self.results: Dict[str, StockPipelineResult] = {}
        # ML models per (symbol, interval) pair for multi-timeframe trading
        self.stockMLModels: Dict[Tuple[str, str], StockMLModel] = {}
        self.portfolioMLModel: Optional[PortfolioMLModel] = None
        self.portfolioSignal: Optional[PortfolioSignal] = None
        self.allocationResult: Optional[AllocationResult] = None
        self.tradeDecisions: Dict[str, FinalTradeDecision] = {}
        self.benchmarks: Dict[str, Dict] = {}  # symbol/SPY → {returnPct, series}
        # Sentiment: {symbol → pd.Series} covering training + backtest period
        self.sentimentData: Dict[str, pd.Series] = {}

        # --- Persistence & Strategy Engine ---
        self.persistence = PersistenceManager()
        self.strategyEngine = StrategyEngine(
            persistence=self.persistence,
            openAIKey=sentimentOpenAIKey or os.environ.get('OPENAI_API_KEY', ''),
        )
        self.connectedStockManager = ConnectedStockManager(self.persistence)

        # --- Chart buffer (filled during Step 9, uploaded in Step 10) ---
        self._pendingCharts: list = []

    # =====================================================================
    # Full pipeline
    # =====================================================================

    def run(self, verbose: bool = True, stop_check=None):
        """
        Execute the complete pipeline for every stock.
        stop_check: optional callable returning True to abort mid-run.

        Steps:
          1.  GA pattern discovery   (top 10 per timeframe, multi-run)
          2.  MC per-pattern ranking (composite score → top 5)
          2a. Pattern Memory — merge fresh patterns with stored best from DB
          2b. MC iterative refinement (random + ML mutation loops)
          3.  Full MC simulation     (fan charts on refined set)
          3b. Sentiment analysis     (historical news + synthetic fallback)
          4.  Train per-stock ML models (on training data + sentiment)
          5.  Train portfolio ML model (on training data + sentiment)
          6.  Trading Decider — reconcile stock + portfolio + sentiment signals
          7.  ML-based backtesting with Decider (on test data, incl. shadow)
          8.  Intelligent Fund Allocation — rebalance based on performance
          9.  Visualisation (fan charts, per-stock & portfolio graphs)
        """
        _stop = stop_check if callable(stop_check) else (lambda: False)
        self._printBanner(verbose)

        if _stop():
            if verbose:
                print("\n  [STOP] Pipeline aborted.")
            return

        # --- Load previous best patterns from persistence (if any) ---
        if verbose:
            summary = self.persistence.getSummary()
            print(f"\n  [Persistence] DB: {summary.get('totalRuns', 0)} prior runs, "
                  f"{summary.get('totalPatterns', 0)} stored patterns, "
                  f"{summary.get('totalStrategies', 0)} strategies")

        # --- Ensure stock metadata is populated ---
        self.strategyEngine.ensureStockMetadata(
            list(self.stocks.keys()), verbose=verbose
        )

        # Pre-compute training and backtest date ranges (used by ALL steps)
        from datetime import datetime, timedelta
        self._trainEndDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')
        self._trainStartDate = (datetime.now() - timedelta(days=self.mlTrainPeriodDays + self.backtestPeriodDays)).strftime('%Y-%m-%d')

        # ── Per-stock pipeline (GA → MC → Refine) — parallelised ────────
        _stageStart = _time_module.time()

        def _runStockStages(symbol: str, allocation: float) -> StockPipelineResult:
            if _stop():
                raise InterruptedError("Stop requested")
            stockFund = self.totalFund * allocation
            result = StockPipelineResult(symbol=symbol, allocation=allocation)

            if verbose:
                print(f"\n{'#' * 80}")
                print(f"# PIPELINE — {symbol}  (allocation {allocation*100:.1f}%, "
                      f"${stockFund:,.2f})")
                print(f"{'#' * 80}")

            rawBank = self._discoverPatterns(symbol, verbose)
            if _stop():
                raise InterruptedError("Stop requested")
            result.rawPatternBank = rawBank

            refinedBank, rankings = self._mcRankAndFilter(
                rawBank, symbol, stockFund, verbose
            )
            if _stop():
                raise InterruptedError("Stop requested")
            result.refinedPatternBank = refinedBank
            result.patternRankings = rankings

            refinedBank, rankings = self._mergeWithStoredPatterns(
                refinedBank, rankings, symbol, verbose
            )
            result.refinedPatternBank = refinedBank
            result.patternRankings = rankings

            if self.mcRefineIterations > 0:
                if _stop():
                    raise InterruptedError("Stop requested")
                refinedBank = self._mcRefinePatterns(
                    refinedBank, symbol, stockFund, verbose
                )
                result.refinedPatternBank = refinedBank

            if _stop():
                raise InterruptedError("Stop requested")
            mcResults = self._runFullMC(refinedBank, symbol, stockFund, verbose)
            result.mcResults = mcResults
            return result

        maxWorkers = min(4, len(self.stocks))
        if maxWorkers > 1:
            if verbose:
                print(f"\n  [Parallel] Running GA+MC for {len(self.stocks)} stocks "
                      f"using {maxWorkers} workers ...")
            with ThreadPoolExecutor(max_workers=maxWorkers) as pool:
                futMap = {
                    pool.submit(_runStockStages, sym, alloc): sym
                    for sym, alloc in self.stocks.items()
                }
                for fut in as_completed(futMap):
                    sym = futMap[fut]
                    try:
                        self.results[sym] = fut.result()
                    except InterruptedError:
                        if verbose:
                            print(f"\n  [STOP] Pipeline aborted during {sym} — cancelling remaining.")
                        for f in futMap:
                            if f != fut and not f.done():
                                f.cancel()
                        return
                    except Exception as e:
                        print(f"  [ERROR] {sym} pipeline failed: {e}")
                        self.results[sym] = StockPipelineResult(
                            symbol=sym, allocation=self.stocks[sym]
                        )
        else:
            for sym, alloc in self.stocks.items():
                if _stop():
                    if verbose:
                        print("\n  [STOP] Pipeline aborted before GA+MC.")
                    return
                try:
                    self.results[sym] = _runStockStages(sym, alloc)
                except InterruptedError:
                    if verbose:
                        print(f"\n  [STOP] Pipeline aborted during {sym}.")
                    return

        if verbose:
            _elapsed = _time_module.time() - _stageStart
            print(f"\n  [Timing] GA+MC stages completed in {_elapsed:.1f}s")

        if _stop():
            if verbose:
                print("\n  [STOP] Pipeline aborted after GA+MC.")
            return

        # ── Timed stage runner ────────────────────────────────────────────
        _pipelineTimings: Dict[str, float] = {}

        def _timedStage(name: str, fn, *args, **kwargs):
            if _stop():
                return None
            t0 = _time_module.time()
            result = fn(*args, **kwargs)
            elapsed = _time_module.time() - t0
            _pipelineTimings[name] = elapsed
            if verbose:
                print(f"\n  [Timing] {name}: {elapsed:.1f}s")
            return result

        # ---- Step 3b: Sentiment Analysis (historical + synthetic) ----
        if self.sentimentEnabled and not _stop():
            _timedStage('3b Sentiment', self._fetchHistoricalSentiment, verbose)

        # ---- Step 3c: Strategy Engine (generate & store strategies) ----
        if not _stop():
            _timedStage('3c Strategies', self._runStrategyEngine, verbose)

        # ---- Step 4: Train per-stock ML models (on training data) ----
        if self.mlEnabled and not _stop():
            _timedStage('4 ML Training', self._trainStockMLModels, verbose)

        # ---- Step 4b: Augment ML models with MC simulation data ----
        if self.mlEnabled and not _stop():
            _timedStage('4b ML+MC Augment', self._augmentMLFromMC, verbose)

        # ---- Step 5: Train portfolio-level ML model (on training data) ----
        if self.mlEnabled and len(self.stocks) >= 2 and not _stop():
            _timedStage('5 Portfolio ML', self._trainPortfolioMLModel, verbose)

        # ---- Step 5b: MC Phase 2 — Full-pipeline simulation ----
        if self.mlEnabled and not _stop():
            _timedStage('5b MC Phase 2', self._runMCPhase2, verbose)

        # ---- Step 6: Trading Decider — reconcile stock + portfolio signals ----
        if self.mlEnabled and not _stop():
            _timedStage('6 Decider', self._runTradingDecider, verbose)

        # ---- Step 7: ML-Based Backtesting with Decider (on test data) ----
        if self.mlEnabled and not _stop():
            _timedStage('7 ML Backtest', self._mlBacktest, verbose)

        # ---- Step 8a: Connected Stocks Evaluation (BEFORE allocation) ----
        if self.mlEnabled and not _stop():
            _timedStage('8a Connected', self._evaluateConnectedStocks, verbose)

        # ---- Step 8b: Dynamic Fund Allocation (main + connected) ----
        if self.mlEnabled and not _stop():
            _timedStage('8b Allocation', self._dynamicReallocation, verbose)

        # ---- Step 9: Benchmarks & Visualisation ----
        if not _stop():
            _timedStage('9 Visualization', lambda v: (
            self._computeBenchmarks(v),
            self._generateFanCharts(v),
            self._generateBacktestGraphs(v),
            self._generatePortfolioGraph(v),
        ), verbose)

        # ---- Summary ----
        if not _stop():
            self._printSummary()

        # ── Print timing summary ─────────────────────────────────────────
        if verbose and _pipelineTimings:
            totalTime = sum(_pipelineTimings.values())
            print(f"\n{'=' * 70}")
            print(f"PIPELINE TIMING SUMMARY  (total: {totalTime/60:.1f} min)")
            print(f"{'=' * 70}")
            for stage, t in sorted(_pipelineTimings.items(), key=lambda x: -x[1]):
                pct = t / totalTime * 100 if totalTime > 0 else 0
                bar = '█' * int(pct / 2)
                print(f"  {stage:<25s} {t:>7.1f}s  ({pct:4.1f}%)  {bar}")
            print()

        # ---- Step 10: Persistence — save run result & patterns ----
        if not _stop():
            self._saveRunToPersistence(verbose)

    # =====================================================================
    # Incremental Update — Lightweight refresh for live trading
    # =====================================================================

    def runIncremental(self, verbose: bool = True, stop_check=None):
        """
        Lightweight pipeline refresh for live trading.  Skips expensive
        GA discovery and MC refinement — instead loads stored patterns
        from the database and only re-runs:

          1. Fetch latest price data
          2. Update sentiment (fetch + score new headlines)
          3. Retrain per-stock ML models on recent data + stored patterns
          4. Retrain portfolio ML model
          5. Run Trading Decider for fresh signals
          6. Quick backtest on recent data only
          7. Evaluate connected stocks

        Designed to run every 1-2 weeks for daily timeframes, or weekly
        for hourly timeframes — much cheaper than a full pipeline run.
        stop_check: optional callable returning True to abort mid-run.
        """
        _stop = stop_check if callable(stop_check) else (lambda: False)
        if verbose:
            print(f"\n{'=' * 80}")
            print("INCREMENTAL UPDATE — Lightweight Pipeline Refresh")
            print(f"{'=' * 80}")
            print(f"Stocks: {list(self.stocks.keys())}")
            print(f"Mode: Reuse stored patterns, retrain ML with new data")

        from datetime import datetime, timedelta

        # Shorter date ranges for incremental updates
        self._trainEndDate = (datetime.now() - timedelta(days=min(self.backtestPeriodDays, 180))).strftime('%Y-%m-%d')
        self._trainStartDate = (datetime.now() - timedelta(days=min(self.mlTrainPeriodDays, 365) + min(self.backtestPeriodDays, 180))).strftime('%Y-%m-%d')

        if _stop():
            if verbose:
                print("\n  [STOP] Incremental run aborted.")
            return

        # --- Step 0: Include existing connected stocks in portfolio ---
        autoStocks = self.persistence.getAutoPortfolioStocks()
        if autoStocks:
            nAuto = len(autoStocks)
            autoTotalFrac = min(0.25, 0.05 * nAuto)
            perAutoFrac = autoTotalFrac / nAuto
            manualTotal = sum(self.stocks.values())
            scaleFactor = (1.0 - autoTotalFrac) / manualTotal if manualTotal > 0 else 1.0
            for k in list(self.stocks.keys()):
                self.stocks[k] *= scaleFactor
            for sym in autoStocks:
                if sym not in self.stocks:
                    self.stocks[sym] = perAutoFrac
            if verbose:
                print(f"\n  Included {len(autoStocks)} connected stocks: "
                      f"{list(autoStocks.keys())}")

        # --- Ensure stock metadata ---
        self.strategyEngine.ensureStockMetadata(
            list(self.stocks.keys()), verbose=verbose
        )

        # --- Step 1: Load stored patterns (no GA discovery) ---
        for symbol, allocation in self.stocks.items():
            if _stop():
                if verbose:
                    print("\n  [STOP] Incremental run aborted.")
                return
            result = StockPipelineResult(symbol=symbol, allocation=allocation)

            if verbose:
                print(f"\n{'#' * 60}")
                print(f"# {symbol} — Loading stored patterns")
                print(f"{'#' * 60}")

            storedPatterns = self.persistence.loadAllActivePatterns(symbol, topN=50)
            if storedPatterns:
                from GeneticAlgorithm import PatternBank, PatternChromosome
                patterns = []
                for sp in storedPatterns:
                    try:
                        import json as _json
                        from GeneticAlgorithm import PatternGene, CandleType
                        genes = _json.loads(sp.genesJson)
                        geneObjs = [PatternGene(
                            expectedPriceChangePct=g['expectedPriceChangePct'],
                            expectedVolumePct=g.get('expectedVolumePct', 1.0),
                            candleType=CandleType(g.get('candleType', 'BULLISH')),
                            minBodyPct=g.get('minBodyPct', 0.3),
                        ) for g in genes]
                        chrom = PatternChromosome(
                            genes=geneObjs, fitness=sp.fitness,
                            interval=sp.interval, symbol=symbol,
                        )
                        patterns.append(chrom)
                    except Exception:
                        continue
                result.refinedPatternBank = PatternBank(symbol=symbol, patterns=patterns)
                if verbose:
                    print(f"  Loaded {len(patterns)} stored patterns")
            else:
                from GeneticAlgorithm import PatternBank
                result.refinedPatternBank = PatternBank(symbol=symbol, patterns=[])
                if verbose:
                    print("  No stored patterns found — ML models will rely on technical indicators only")

            self.results[symbol] = result

        # --- Step 2: Sentiment update (fetch fresh headlines) ---
        if _stop():
            if verbose:
                print("\n  [STOP] Incremental run aborted.")
            return
        if self.sentimentEnabled:
            self._fetchHistoricalSentiment(verbose)

        if _stop():
            return
        # --- Step 3: Reload strategies (no regeneration) ---
        self._runStrategyEngine(verbose)

        if _stop():
            return
        # --- Step 4: Retrain per-stock ML models with new data ---
        if self.mlEnabled:
            self._trainStockMLModels(verbose)

        if _stop():
            return
        # --- Step 5: Retrain portfolio ML model ---
        if self.mlEnabled and len(self.stocks) >= 2:
            self._trainPortfolioMLModel(verbose)

        if _stop():
            return
        # --- Step 6: Trading Decider ---
        if self.mlEnabled:
            self._runTradingDecider(verbose)

        if _stop():
            return
        # --- Step 7: Quick backtest (shorter period) ---
        if self.mlEnabled:
            self._mlBacktest(verbose)

        if _stop():
            return
        # --- Step 7b: Walk-forward validation ---
        if self.mlEnabled and self.useWalkForward:
            self._walkForwardValidation(verbose)

        if _stop():
            return
        # --- Step 8a: Connected stocks (before allocation) ---
        if self.mlEnabled:
            self._evaluateConnectedStocks(verbose)

        if _stop():
            return
        # --- Step 8b: Dynamic allocation (main + connected) ---
        if self.mlEnabled:
            self._dynamicReallocation(verbose)

        # --- Summary + persist ---
        self._printSummary()
        self._saveRunToPersistence(verbose)

        if verbose:
            print(f"\n{'=' * 80}")
            print("INCREMENTAL UPDATE COMPLETE")
            print(f"{'=' * 80}")

    # =====================================================================
    # Add-Stock Pipeline — Full GA for new stocks, reuse patterns for rest
    # =====================================================================

    def runForNewStocks(self, newSymbols: List[str],
                        verbose: bool = True, stop_check=None):
        """
        Targeted pipeline for adding new stocks to an existing portfolio.

        - **New stocks** (in ``newSymbols``): full GA pattern discovery, MC
          ranking/refinement, and MC simulation — identical to ``run()``.
        - **Existing stocks**: load stored patterns from the database —
          identical to ``runIncremental()``.
        - **All stocks together**: sentiment, ML training, portfolio ML,
          trading decider, backtesting, connected-stock evaluation,
          dynamic allocation, and persistence.

        This avoids re-running the expensive GA+MC stages for stocks that
        already have patterns while still giving new stocks a proper
        foundation.
        """
        _stop = stop_check if callable(stop_check) else (lambda: False)

        if verbose:
            print(f"\n{'=' * 80}")
            print("ADD-STOCK PIPELINE — Full GA for new, stored patterns for existing")
            print(f"{'=' * 80}")
            print(f"All stocks : {list(self.stocks.keys())}")
            print(f"New stocks : {newSymbols}")
            existingSyms = [s for s in self.stocks if s not in newSymbols]
            print(f"Existing   : {existingSyms}")

        from datetime import datetime, timedelta
        import time as _time_module

        self._trainEndDate = (
            datetime.now() - timedelta(days=self.backtestPeriodDays)
        ).strftime('%Y-%m-%d')
        self._trainStartDate = (
            datetime.now() - timedelta(
                days=self.mlTrainPeriodDays + self.backtestPeriodDays
            )
        ).strftime('%Y-%m-%d')

        if _stop():
            return

        # --- Ensure stock metadata ---
        self.strategyEngine.ensureStockMetadata(
            list(self.stocks.keys()), verbose=verbose
        )

        if verbose:
            summary = self.persistence.getSummary()
            print(f"\n  [Persistence] DB: {summary.get('totalRuns', 0)} prior "
                  f"runs, {summary.get('totalPatterns', 0)} stored patterns")

        # ── Per-stock stage: full GA for new, load patterns for existing ──
        _stageStart = _time_module.time()

        def _fullStockStages(symbol: str, allocation: float) -> StockPipelineResult:
            """Full GA + MC pipeline for a single NEW stock."""
            if _stop():
                raise InterruptedError("Stop requested")
            stockFund = self.totalFund * allocation
            result = StockPipelineResult(symbol=symbol, allocation=allocation)

            if verbose:
                print(f"\n{'#' * 80}")
                print(f"# NEW STOCK — {symbol}  (allocation {allocation*100:.1f}%, "
                      f"${stockFund:,.2f})")
                print(f"{'#' * 80}")

            rawBank = self._discoverPatterns(symbol, verbose)
            if _stop():
                raise InterruptedError("Stop requested")
            result.rawPatternBank = rawBank

            refinedBank, rankings = self._mcRankAndFilter(
                rawBank, symbol, stockFund, verbose,
            )
            if _stop():
                raise InterruptedError("Stop requested")
            result.refinedPatternBank = refinedBank
            result.patternRankings = rankings

            refinedBank, rankings = self._mergeWithStoredPatterns(
                refinedBank, rankings, symbol, verbose,
            )
            result.refinedPatternBank = refinedBank
            result.patternRankings = rankings

            if self.mcRefineIterations > 0:
                if _stop():
                    raise InterruptedError("Stop requested")
                refinedBank = self._mcRefinePatterns(
                    refinedBank, symbol, stockFund, verbose,
                )
                result.refinedPatternBank = refinedBank

            if _stop():
                raise InterruptedError("Stop requested")
            mcResults = self._runFullMC(refinedBank, symbol, stockFund, verbose)
            result.mcResults = mcResults
            return result

        def _loadStoredPatterns(symbol: str, allocation: float) -> StockPipelineResult:
            """Load stored patterns for an EXISTING stock (no GA)."""
            result = StockPipelineResult(symbol=symbol, allocation=allocation)
            if verbose:
                print(f"\n{'#' * 60}")
                print(f"# EXISTING — {symbol} — loading stored patterns")
                print(f"{'#' * 60}")

            storedPatterns = self.persistence.loadAllActivePatterns(symbol, topN=50)
            if storedPatterns:
                from GeneticAlgorithm import PatternBank, PatternChromosome
                patterns = []
                for sp in storedPatterns:
                    try:
                        import json as _json
                        from GeneticAlgorithm import PatternGene, CandleType
                        genes = _json.loads(sp.genesJson)
                        geneObjs = [PatternGene(
                            expectedPriceChangePct=g['expectedPriceChangePct'],
                            expectedVolumePct=g.get('expectedVolumePct', 1.0),
                            candleType=CandleType(g.get('candleType', 'BULLISH')),
                            minBodyPct=g.get('minBodyPct', 0.3),
                        ) for g in genes]
                        chrom = PatternChromosome(
                            genes=geneObjs, fitness=sp.fitness,
                            interval=sp.interval, symbol=symbol,
                        )
                        patterns.append(chrom)
                    except Exception:
                        continue
                result.refinedPatternBank = PatternBank(
                    symbol=symbol, patterns=patterns,
                )
                if verbose:
                    print(f"  Loaded {len(patterns)} stored patterns")
            else:
                from GeneticAlgorithm import PatternBank
                result.refinedPatternBank = PatternBank(
                    symbol=symbol, patterns=[],
                )
                if verbose:
                    print("  No stored patterns — ML will use technical "
                          "indicators only")
            return result

        # Run full pipeline for new stocks (parallelised)
        newSet = set(newSymbols)
        newStockItems = [(s, a) for s, a in self.stocks.items() if s in newSet]
        existingItems = [(s, a) for s, a in self.stocks.items() if s not in newSet]

        maxWorkers = min(4, max(len(newStockItems), 1))
        if len(newStockItems) > 1:
            if verbose:
                print(f"\n  [Parallel] Running GA+MC for {len(newStockItems)} "
                      f"new stocks using {maxWorkers} workers ...")
            with ThreadPoolExecutor(max_workers=maxWorkers) as pool:
                futMap = {
                    pool.submit(_fullStockStages, sym, alloc): sym
                    for sym, alloc in newStockItems
                }
                for fut in as_completed(futMap):
                    sym = futMap[fut]
                    try:
                        self.results[sym] = fut.result()
                    except InterruptedError:
                        if verbose:
                            print(f"\n  [STOP] Aborted during {sym}.")
                        return
                    except Exception as e:
                        print(f"  [ERROR] {sym} pipeline failed: {e}")
                        self.results[sym] = StockPipelineResult(
                            symbol=sym, allocation=self.stocks[sym],
                        )
        elif newStockItems:
            sym, alloc = newStockItems[0]
            try:
                self.results[sym] = _fullStockStages(sym, alloc)
            except InterruptedError:
                if verbose:
                    print(f"\n  [STOP] Aborted during {sym}.")
                return

        if _stop():
            return

        # Load stored patterns for existing stocks
        for sym, alloc in existingItems:
            if _stop():
                return
            self.results[sym] = _loadStoredPatterns(sym, alloc)

        if verbose:
            _elapsed = _time_module.time() - _stageStart
            print(f"\n  [Timing] Per-stock stages completed in {_elapsed:.1f}s "
                  f"({len(newStockItems)} new + {len(existingItems)} existing)")

        if _stop():
            return

        # ── Shared stages for ALL stocks ──────────────────────────────────
        _pipelineTimings: Dict[str, float] = {}

        def _timedStage(name: str, fn, *args, **kwargs):
            if _stop():
                return None
            t0 = _time_module.time()
            result = fn(*args, **kwargs)
            elapsed = _time_module.time() - t0
            _pipelineTimings[name] = elapsed
            if verbose:
                print(f"\n  [Timing] {name}: {elapsed:.1f}s")
            return result

        if self.sentimentEnabled and not _stop():
            _timedStage('Sentiment', self._fetchHistoricalSentiment, verbose)

        if not _stop():
            _timedStage('Strategies', self._runStrategyEngine, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('ML Training', self._trainStockMLModels, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('ML+MC Augment', self._augmentMLFromMC, verbose)

        if self.mlEnabled and len(self.stocks) >= 2 and not _stop():
            _timedStage('Portfolio ML', self._trainPortfolioMLModel, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('MC Phase 2', self._runMCPhase2, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('Decider', self._runTradingDecider, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('ML Backtest', self._mlBacktest, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('Connected Stocks', self._evaluateConnectedStocks, verbose)

        if self.mlEnabled and not _stop():
            _timedStage('Allocation', self._dynamicReallocation, verbose)

        if not _stop():
            self._printSummary()

        if verbose and _pipelineTimings:
            totalTime = sum(_pipelineTimings.values())
            print(f"\n{'=' * 70}")
            print(f"ADD-STOCK TIMING SUMMARY  (total: {totalTime/60:.1f} min)")
            print(f"{'=' * 70}")
            for stage, t in sorted(_pipelineTimings.items(), key=lambda x: -x[1]):
                pctTime = t / totalTime * 100 if totalTime > 0 else 0
                bar = '█' * int(pctTime / 2)
                print(f"  {stage:<25s} {t:>7.1f}s  ({pctTime:4.1f}%)  {bar}")
            print()

        if not _stop():
            self._saveRunToPersistence(verbose)

        if verbose:
            print(f"\n{'=' * 80}")
            print("ADD-STOCK PIPELINE COMPLETE")
            print(f"{'=' * 80}")

    # =====================================================================
    # Step 1 — GA Pattern Discovery
    # =====================================================================

    def _discoverPatterns(self, symbol: str, verbose: bool) -> PatternBank:
        """Run the Genetic Algorithm to discover up to maxPatternsPerInterval
        patterns per timeframe, with multi-run for robustness.
        
        CRITICAL: GA discovers patterns on TRAINING data only (chronologically
        BEFORE the backtest period) to avoid look-ahead bias.
        """
        from datetime import datetime, timedelta
        
        # Calculate training date range (chronologically BEFORE backtest)
        # Example: if now=Feb 2026, backtestPeriods=5y, gaDiscoveryPeriods=5y:
        #   trainEndDate = Feb 2021 (backtest starts here)
        #   trainStartDate = Feb 2016 (5 years before trainEndDate)
        trainEndDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')
        trainStartDate = (datetime.now() - timedelta(days=self.backtestPeriodDays + self.gaDiscoveryPeriodDays)).strftime('%Y-%m-%d')
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 1: GA PATTERN DISCOVERY — {symbol}")
            print(f"{'=' * 70}")
            print(f"⚠ TRAINING DATA ONLY (out-of-sample): {trainStartDate} → {trainEndDate}")
            print(f"  (Backtest will run on: {trainEndDate} → {datetime.now().strftime('%Y-%m-%d')})")
            print(f"Intervals: {[i[0] for i in self.intervals]}")
            print(f"Pattern Lengths: {self.patternLengths}")
            print(f"Population: {self.populationSize}, Generations: {self.numGenerations}")
            print(f"Runs per config: {self.numRunsPerConfig}")
            print(f"Max patterns per interval: {self.maxPatternsPerInterval}")

        # Convert relative intervals to absolute date ranges
        # Original: [('1d', '5y'), ('1h', '730d'), ...]
        # New:      [('1d', trainStartDate, trainEndDate), ...]
        #
        # ADAPTIVE PER-INTERVAL TRAINING:
        # Instead of skipping intervals when trainStartDate exceeds
        # Yahoo's limit, we SHORTEN the training window for sub-daily
        # intervals to the maximum that Yahoo can serve.  This ensures
        # 1h (and potentially 30m) still contribute patterns and trades.
        # Only skip if the resulting training window is too short (<30 days)
        # or if even the backtest end date exceeds Yahoo's limit.
        YAHOO_INTERVAL_LIMITS = {
            '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60,
            '60m': 730, '1h': 730, '1d': 36500, '1wk': 36500, '1mo': 36500,
        }
        YAHOO_SAFETY_MARGIN = 7  # days buffer — Yahoo rejects requests at exact boundary
        MIN_TRAINING_DAYS = 30  # Minimum useful training window
        intervalsWithDates = []
        trainStartDt = datetime.strptime(trainStartDate, '%Y-%m-%d')
        trainEndDt = datetime.strptime(trainEndDate, '%Y-%m-%d')

        for interval, _ in self.intervals:  # Ignore the period, use our date range
            maxDays = YAHOO_INTERVAL_LIMITS.get(interval, 730)
            daysFromToday = (datetime.now() - trainStartDt).days
            intTrainStart = trainStartDate

            if daysFromToday > maxDays:
                # Shorten training window: earliest date Yahoo can serve (with safety margin)
                earliestFeasible = datetime.now() - timedelta(days=maxDays - YAHOO_SAFETY_MARGIN)
                # Training must end BEFORE backtest starts (trainEndDt)
                if earliestFeasible >= trainEndDt:
                    if verbose:
                        print(f"  ⚠ SKIPPING {interval}: Yahoo {maxDays}d limit "
                              f"cannot reach the training period ending {trainEndDate}.")
                    continue
                intTrainStart = earliestFeasible.strftime('%Y-%m-%d')
                shortenedDays = (trainEndDt - earliestFeasible).days
                if shortenedDays < MIN_TRAINING_DAYS:
                    if verbose:
                        print(f"  ⚠ SKIPPING {interval}: Only {shortenedDays}d of "
                              f"training data available (need ≥{MIN_TRAINING_DAYS}d).")
                    continue
                if verbose:
                    print(f"  ℹ {interval}: Shortened training to {intTrainStart} → "
                          f"{trainEndDate} ({shortenedDays}d) to fit Yahoo {maxDays}d limit.")

            intervalsWithDates.append((interval, intTrainStart, trainEndDate))

        if not intervalsWithDates:
            if verbose:
                print("  ⚠ No intervals are feasible for the training window!")
            return PatternBank(symbol=symbol, patterns=[])

        ga = GeneticAlgorithmPatternFinder(
            populationSize=self.populationSize,
            patternLength=5,       # overridden inside discoverPatternBank
            generations=self.numGenerations,
            forgivenessPct=self.forgiveness,
            earlyStopGenerations=self.gaEarlyStopGenerations,
            elitismCount=self.gaElitismCount,
            mutationRate=self.gaMutationRate,
            crossoverRate=self.gaCrossoverRate,
            minImprovementThreshold=self.gaMinImprovement,
        )

        patternBank = ga.discoverPatternBank(
            symbol=symbol,
            intervals=intervalsWithDates,  # Now with absolute dates
            patternLengths=self.patternLengths,
            numRunsPerConfig=self.numRunsPerConfig,
            maxPatternsPerInterval=self.maxPatternsPerInterval,
            verbose=verbose,
        )

        if verbose:
            print(f"\nGA finished — {len(patternBank.patterns)} total patterns")
            print(patternBank.summary())

        return patternBank

    # =====================================================================
    # Step 2 — MC Per-Pattern Ranking + Filtering
    # =====================================================================

    def _mcRankAndFilter(self, patternBank: PatternBank, symbol: str,
                         stockFund: float, verbose: bool
                         ) -> Tuple[PatternBank, Dict[str, list]]:
        """
        For each interval in the bank, rank patterns by MC composite score
        and keep only the top N.

        Returns:
            (refinedPatternBank, {interval: [(pattern, score), ...]})
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 2: MC PATTERN RANKING — {symbol}")
            print(f"{'=' * 70}")
            print(f"Ranking sims per pattern: {self.mcRankSimulations}")
            print(f"Keeping top {self.mcTopN} per interval")

        simulator = MCMCSimulator(
            initialFund=stockFund,
            forgiveness=self.forgiveness,
            numSimulations=self.mcRankSimulations,
            simulationPeriods=self.mcSimulationPeriods,
            method=self.mcMethod,
        )

        intervals = sorted(set(
            p.interval for p in patternBank.patterns if p.interval
        ))

        allRankings: Dict[str, list] = {}
        refinedBank = PatternBank(symbol=symbol, patterns=[])

        for interval in intervals:
            ranked = simulator.rankPatterns(
                patternBank=patternBank,
                symbol=symbol,
                interval=interval,
                calibrationPeriod=self.mcCalibrationPeriod,
                calibrationStart=self._trainStartDate,
                calibrationEnd=self._trainEndDate,
                numSimulations=self.mcRankSimulations,
                numPeriods=self.mcSimulationPeriods,
                topN=self.mcTopN,
                verbose=verbose,
            )
            allRankings[interval] = ranked

            # Add surviving patterns to the refined bank (sorted by score)
            for pattern, score in ranked:
                refinedBank.patterns.append(pattern)

        if verbose:
            print(f"\nRefined PatternBank: {len(refinedBank.patterns)} patterns")
            for interval in intervals:
                n = len([p for p in refinedBank.patterns if p.interval == interval])
                print(f"  {interval}: {n} patterns")

        return refinedBank, allRankings

    # =====================================================================
    # Step 2a — Pattern Memory (merge fresh with stored best)
    # =====================================================================

    def _mergeWithStoredPatterns(
        self, freshBank: PatternBank,
        rankings: Dict[str, list],
        symbol: str, verbose: bool
    ) -> Tuple[PatternBank, Dict[str, list]]:
        """
        PATTERN MEMORY — Compare fresh GA/MC patterns against stored best
        patterns from prior runs.  Per interval, keep whichever set has a
        higher average MC composite score.  This prevents regression: if a
        run produces worse patterns for a stock, the pipeline falls back to
        the proven historical patterns.

        Also imports stored patterns for intervals that the current run did
        not cover (e.g., from prior runs with different timeframes).

        Returns:
            (mergedPatternBank, updatedRankings)
        """
        if verbose:
            print(f"\n  {'─' * 60}")
            print(f"  STEP 2a: PATTERN MEMORY — {symbol}")
            print(f"  {'─' * 60}")

        freshIntervals = sorted(set(
            p.interval for p in freshBank.patterns if p.interval
        ))

        mergedPatterns: List[PatternChromosome] = []
        mergedRankings: Dict[str, list] = {}
        memoryUsed = False

        for interval in freshIntervals:
            # --- Fresh patterns + their MC scores ---
            freshForInterval = [
                p for p in freshBank.patterns if p.interval == interval
            ]
            freshScores = rankings.get(interval, [])
            avgFreshScore = (
                float(np.mean([score for _, score in freshScores]))
                if freshScores else 0.0
            )

            # --- Stored best from DB ---
            try:
                storedPatterns = self.persistence.loadBestPatterns(
                    symbol, interval, topN=25
                )
            except Exception as e:
                if verbose:
                    print(f"    {interval}: ⚠ DB load failed ({e}), using fresh")
                storedPatterns = []

            avgStoredScore = (
                float(np.mean([sp.mcCompositeScore for sp in storedPatterns]))
                if storedPatterns else 0.0
            )

            if storedPatterns and avgStoredScore > avgFreshScore:
                # ✱ DB patterns are better — reconstruct chromosomes
                if verbose:
                    print(f"    {interval}: ★ MEMORY  (stored avg {avgStoredScore:.4f} "
                          f"> fresh avg {avgFreshScore:.4f}, "
                          f"{len(storedPatterns)} stored patterns)")
                converted = []
                convertedRankings = []
                for sp in storedPatterns:
                    chrom = self._storedPatternToChromosome(sp)
                    converted.append(chrom)
                    convertedRankings.append((chrom, sp.mcCompositeScore))
                mergedPatterns.extend(converted)
                mergedRankings[interval] = convertedRankings
                memoryUsed = True
            else:
                # Fresh patterns are equal or better — keep them
                if verbose:
                    if storedPatterns:
                        print(f"    {interval}: ✓ FRESH   (fresh avg {avgFreshScore:.4f} "
                              f">= stored avg {avgStoredScore:.4f})")
                    else:
                        print(f"    {interval}: ✓ FRESH   (no stored patterns in DB)")
                mergedPatterns.extend(freshForInterval)
                mergedRankings[interval] = freshScores

        # --- Import stored patterns for intervals NOT in this run ---
        try:
            allStored = self.persistence.loadAllActivePatterns(symbol, topN=200)
        except Exception:
            allStored = []

        extraIntervals = sorted(set(
            sp.interval for sp in allStored
            if sp.interval and sp.interval not in freshIntervals
        ))

        for interval in extraIntervals:
            storedForInterval = [
                sp for sp in allStored if sp.interval == interval
            ]
            if storedForInterval:
                if verbose:
                    print(f"    {interval}: ⊕ IMPORT  ({len(storedForInterval)} stored "
                          f"patterns from prior runs)")
                converted = []
                convertedRankings = []
                for sp in storedForInterval:
                    chrom = self._storedPatternToChromosome(sp)
                    converted.append(chrom)
                    convertedRankings.append((chrom, sp.mcCompositeScore))
                mergedPatterns.extend(converted)
                mergedRankings[interval] = convertedRankings
                memoryUsed = True

        if verbose:
            if memoryUsed:
                print(f"    → Merged bank: {len(mergedPatterns)} patterns "
                      f"(memory contributed to final set)")
            else:
                print(f"    → All fresh patterns kept ({len(mergedPatterns)} total)")

        mergedBank = PatternBank(symbol=symbol, patterns=mergedPatterns)
        return mergedBank, mergedRankings

    def _storedPatternToChromosome(self, sp: StoredPattern) -> PatternChromosome:
        """Convert a StoredPattern (from DB) back into a live PatternChromosome."""
        genesData = json.loads(sp.genesJson)
        genes = [
            PatternGene(
                expectedPriceChangePct=g.get('expectedPriceChangePct', 0.0),
                expectedVolumePct=g.get('expectedVolumePct', 1.0),
                candleType=CandleType(g.get('candleType', 'BULLISH')),
                minBodyPct=g.get('minBodyPct', 0.0),
            )
            for g in genesData
        ]
        return PatternChromosome(
            genes=genes,
            fitness=sp.fitness,
            interval=sp.interval,
            symbol=sp.symbol,
            totalMatches=0,
            correctPredictions=0,
        )

    # =====================================================================
    # Step 2b — MC Iterative Refinement (Random + ML Mutations)
    # =====================================================================

    def _mcRefinePatterns(self, patternBank: PatternBank, symbol: str,
                          stockFund: float, verbose: bool) -> PatternBank:
        """
        Iteratively improve patterns by running MC simulations, scoring
        each pattern, generating random + ML-guided mutations, and keeping
        the best-performing variants.

        Uses the PatternRefiner engine which:
          - Random: perturbs 1-3 gene values stochastically (exploration)
          - ML: trains a GradientBoosting model on MC trade data, uses
            feature importances and partial dependence to find gene values
            that maximise success probability (exploitation)

        Returns a new PatternBank with the refined patterns.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 2b: MC ITERATIVE REFINEMENT — {symbol}")
            print(f"{'=' * 70}")
            print(f"Iterations: {self.mcRefineIterations}")
            print(f"Sims/candidate: {self.mcRefineSimsPerCandidate}")
            print(f"Random mutants/pattern: {self.mcRefineRandomMutants}")
            print(f"ML mutants/pattern: {self.mcRefineMLMutants}")
            print(f"Mutation strength: {self.mcRefineMutationStrength}")

        refiner = PatternRefiner(
            initialFund=stockFund,
            forgiveness=self.forgiveness,
            mcMethod=self.mcMethod,
            simsPerCandidate=self.mcRefineSimsPerCandidate,
            simPeriods=self.mcSimulationPeriods,
            randomMutantsPerPattern=self.mcRefineRandomMutants,
            mlMutantsPerPattern=self.mcRefineMLMutants,
            mutationStrength=self.mcRefineMutationStrength,
        )

        # Group patterns by interval and refine each group
        intervals = sorted(set(
            p.interval for p in patternBank.patterns if p.interval
        ))

        refinedBank = PatternBank(symbol=symbol, patterns=[])

        for interval in intervals:
            patternsForInterval = [
                p for p in patternBank.patterns if p.interval == interval
            ]
            if not patternsForInterval:
                continue

            if verbose:
                print(f"\n  Refining {len(patternsForInterval)} patterns "
                      f"for {interval}...")

            # Determine calibration period based on interval
            calibPeriod = self.mcCalibrationPeriod

            ranked = refiner.refinePatterns(
                patterns=patternsForInterval,
                symbol=symbol,
                interval=interval,
                calibrationPeriod=calibPeriod,
                calibrationStart=self._trainStartDate,
                calibrationEnd=self._trainEndDate,
                iterations=self.mcRefineIterations,
                verbose=verbose,
            )

            # Add refined patterns to the new bank
            for pattern, score in ranked:
                refinedBank.patterns.append(pattern)

        if verbose:
            print(f"\n  Refinement complete: {len(refinedBank.patterns)} "
                  f"patterns in refined bank")

        return refinedBank

    # =====================================================================
    # Step 3 — Full MC Simulation
    # =====================================================================

    def _runFullMC(self, patternBank: PatternBank, symbol: str,
                   stockFund: float, verbose: bool
                   ) -> Dict[str, MonteCarloResults]:
        """Run the full Monte Carlo simulation on the refined pattern set."""
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 3: FULL MC SIMULATION — {symbol}")
            print(f"{'=' * 70}")
            print(f"Simulations: {self.mcNumSimulations}")
            print(f"Periods: {self.mcSimulationPeriods}")
            print(f"Method: {self.mcMethod}")

        simulator = MCMCSimulator(
            initialFund=stockFund,
            forgiveness=self.forgiveness,
            numSimulations=self.mcNumSimulations,
            simulationPeriods=self.mcSimulationPeriods,
            method=self.mcMethod,
        )

        allResults = simulator.simulateAll(
            patternBank=patternBank,
            symbol=symbol,
            calibrationPeriod=self.mcCalibrationPeriod,
            calibrationStart=self._trainStartDate,
            calibrationEnd=self._trainEndDate,
            targets=self.mcTargets,
            ruinThreshold=self.mcRuinThreshold,
            verbose=verbose,
        )

        return allResults

    # =====================================================================
    # Step 3b — Historical Sentiment Analysis
    # =====================================================================

    def _fetchHistoricalSentiment(self, verbose: bool):
        """
        Fetch (or synthesise) daily sentiment series for every stock in
        the portfolio, covering both the **training** and **backtest**
        periods so that downstream ML training and backtesting can both
        consume sentiment features.

        Results are stored in ``self.sentimentData``  — a dict mapping
        each symbol to a date-indexed ``pd.Series`` of scores in [-1, +1].
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 3b: HISTORICAL SENTIMENT ANALYSIS")
            print(f"{'=' * 70}")

        from datetime import datetime as _dt, timedelta as _td

        # Cover the full date range: from the training start to today
        # so both ML training and backtest periods have sentiment data.
        startDate = self._trainStartDate           # earliest date for ML training
        endDate = _dt.now().strftime('%Y-%m-%d')    # up to today for backtest

        if verbose:
            print(f"  Date range: {startDate} → {endDate}")
            print(f"  Stocks:     {list(self.stocks.keys())}")

        # Optionally pre-fetch daily OHLCV for synthetic fallback so the
        # SentimentAnalyzer doesn't re-download inside its own loop.
        fetcher = StockDataFetcher()
        priceDataDict: Dict[str, pd.DataFrame] = {}
        for symbol in self.stocks:
            df = fetcher.fetchData(symbol, interval='1d',
                                   start=startDate, end=endDate)
            if df is not None and len(df) > 0:
                priceDataDict[symbol] = df

        # Build the analyser (uses the same cache dir / decay settings)
        analyzer = SentimentAnalyzer(
            openAIKey=self.sentimentOpenAIKey,
            sentimentDecayHalfLifeDays=self.sentimentDecayHalfLife,
            cacheDir=self.sentimentCacheDir,
        )

        self.sentimentData = analyzer.fetchHistorical(
            symbols=list(self.stocks.keys()),
            startDate=startDate,
            endDate=endDate,
            priceDataDict=priceDataDict,
            newsAPIKey=self.sentimentNewsAPIKey,
            finnhubKey=self.sentimentFinnhubKey,
            alphaVantageKey=self.sentimentAlphaVantageKey,
            verbose=verbose,
        )

        if verbose:
            totalDays = sum(len(s) for s in self.sentimentData.values())
            print(f"\n  Sentiment data ready: {len(self.sentimentData)} symbols, "
                  f"{totalDays} total day-scores")

    # =====================================================================
    # Step 4 — Backtesting
    # =====================================================================

    def _backtest(self, patternBank: PatternBank, symbol: str,
                  stockFund: float, verbose: bool
                  ) -> Dict[str, BacktestResult]:
        """Backtest the MC-refined patterns on real historical data."""
        endDate = datetime.now()
        startDate = endDate - timedelta(days=self.backtestPeriodDays)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 4: BACKTESTING — {symbol}")
            print(f"{'=' * 70}")
            print(f"Period: {startDate.strftime('%Y-%m-%d')} to {endDate.strftime('%Y-%m-%d')}")
            print(f"Fund: ${stockFund:,.2f}")

        backtester = Backtester(
            initialFund=stockFund,
            forgiveness=self.forgiveness,
        )

        results = backtester.backtestPatternBank(
            patternBank=patternBank,
            symbol=symbol,
            startDate=startDate.strftime('%Y-%m-%d'),
            endDate=endDate.strftime('%Y-%m-%d'),
            topN=self.mcTopN,
            verbose=verbose,
        )

        return results

    # =====================================================================
    # Step 3c — Strategy Engine
    # =====================================================================

    def _runStrategyEngine(self, verbose: bool):
        """
        Generate cross-stock strategies using the StrategyEngine.
        Strategies are stored in the database for persistence across runs.
        Strategy features are injected into ML models.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 3c: STRATEGY ENGINE — Cross-Stock Strategies")
            print(f"{'=' * 70}")

        symbols = list(self.stocks.keys())

        # Gather stock data for statistical analysis (parallel)
        stockDataDict: Dict[str, pd.DataFrame] = {}
        def _fetchSym(sym):
            try:
                df = StockDataFetcher().fetchData(sym, interval='1d', period='2y')
                return sym, df if (df is not None and len(df) > 30) else None
            except Exception:
                return sym, None
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
            for sym, df in pool.map(_fetchSym, symbols):
                if df is not None:
                    stockDataDict[sym] = df

        if len(stockDataDict) < 2:
            if verbose:
                print("  Not enough stock data for strategy generation")
            return

        # Generate / load strategies
        strategies = self.strategyEngine.ensureStrategies(
            symbols=symbols,
            stockDataDict=stockDataDict,
            verbose=verbose,
        )

        if not strategies:
            if verbose:
                print("  No strategies generated")
            return

        # Featurise strategies for ML injection
        stratFeatures = self.strategyEngine.featuriseStrategies(
            strategies, stockDataDict, symbols
        )
        crossRuleFeats = self.strategyEngine.featuriseCrossStockRules(
            symbols, stockDataDict
        )

        # Store for later injection into ML models
        self._strategyFeatures = stratFeatures
        self._crossStockRuleFeatures = crossRuleFeats
        self._strategies = strategies

        if verbose:
            print(f"  Strategies: {len(strategies)}, "
                  f"Rule features: {len(crossRuleFeats.columns) if len(crossRuleFeats) > 0 else 0}")

    # =====================================================================
    # Step 10 — Save Run to Persistence
    # =====================================================================

    def _saveRunToPersistence(self, verbose: bool):
        """
        Save the current run's results, patterns, and performance to the
        SQLite database.  If patterns are better than stored ones, they
        replace the worst in a ranked manner.
        """
        import json
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"STEP 10: SAVING RUN TO PERSISTENCE")
            print(f"{'=' * 70}")

        # Compute REAL portfolio-level metrics using dollar P/L (same logic as _printSummary)
        totalProfit = 0.0
        totalTrades = 0
        totalWins   = 0
        totalPatterns = 0

        for sym, res in self.results.items():
            stockFund = self.totalFund * res.allocation
            if res.backtestResults:
                for interval, bt in res.backtestResults.items():
                    if interval.startswith('WalkForward_'):
                        continue
                    intervalProfit = sum(
                        float(t.get('dollarPnL', 0.0)) for t in bt.trades
                    )
                    totalProfit += intervalProfit
                    totalTrades += bt.totalTrades
                    totalWins   += bt.successfulTrades
            if res.refinedPatternBank:
                totalPatterns += len(res.refinedPatternBank.patterns)

        totalReturnPct = (totalProfit / self.totalFund * 100) if self.totalFund > 0 else 0.0

        # Alpha vs Buy-and-Hold (weighted)
        alphaVsBuyHold = 0.0
        if self.benchmarks:
            weightedBH = sum(
                self.stocks.get(sym, 0) * self.benchmarks[sym]['returnPct']
                for sym in self.stocks if sym in self.benchmarks
            )
            alphaVsBuyHold = totalReturnPct - weightedBH

        # Alpha vs S&P 500
        alphaVsSP500 = 0.0
        if self.benchmarks and '^GSPC' in self.benchmarks:
            alphaVsSP500 = totalReturnPct - self.benchmarks['^GSPC']['returnPct']

        # Sharpe ratio (annualised, from trade returns)
        sharpeRatio = 0.0
        if totalTrades >= 5:
            allTradeReturns = []
            for sym, res in self.results.items():
                if res.backtestResults:
                    for interval, bt in res.backtestResults.items():
                        if interval.startswith('WalkForward_'):
                            continue
                        for t in bt.trades:
                            pnl = float(t.get('dollarPnL', 0.0))
                            allTradeReturns.append(pnl / self.totalFund)
            if allTradeReturns:
                import numpy as _np
                trArr = _np.array(allTradeReturns)
                meanR = trArr.mean()
                stdR  = trArr.std()
                if stdR > 1e-9:
                    sharpeRatio = (meanR / stdR) * (252 ** 0.5)  # annualised

        # Win rate
        winRate = (totalWins / totalTrades * 100) if totalTrades > 0 else 0.0

        # Per-stock breakdown
        # Denominator: use max(notional, effective capital). When all slots are replayed,
        # notional = sum(bt.initialBalance) = effective capital. When slots are skipped
        # (trade count mismatch), trades keep unscaled P/L and bt.initialBalance stays
        # at NOTIONAL_FUND — using notional prevents inflating % when profit is unscaled.
        perStockResults = {}
        for sym, res in self.results.items():
            stockProfit = 0.0
            stockTrades = 0
            stockWins   = 0
            stockNotional = 0.0
            if res.backtestResults:
                for interval, bt in res.backtestResults.items():
                    if interval.startswith('WalkForward_'):
                        continue
                    stockProfit += sum(float(t.get('dollarPnL', 0.0)) for t in bt.trades)
                    if bt.trades and bt.initialBalance > 0:
                        stockNotional += bt.initialBalance
                    stockTrades += bt.totalTrades
                    stockWins   += bt.successfulTrades
            effectiveCapital = max(self.totalFund * res.allocation, 1.0)
            denom = max(stockNotional, effectiveCapital) if stockNotional > 0 else effectiveCapital
            meta = self.persistence.getStockMetadata(sym)
            pType = (meta.portfolioType if meta and meta.portfolioType else 'manual')
            perStockResults[sym] = {
                'returnPct': round((stockProfit / denom * 100) if denom > 0 else 0, 2),
                'profit':    round(stockProfit, 2),
                'trades':    stockTrades,
                'winRate':   round((stockWins / stockTrades * 100) if stockTrades > 0 else 0, 1),
                'allocation': round(res.allocation * 100, 1),
                'portfolioType': pType,
            }

        # Save run result
        import uuid as _uuid
        from datetime import datetime as _dtNow
        runResult = RunResult(
            runId=str(_uuid.uuid4())[:12],
            timestamp=_dtNow.now().isoformat(),
            symbols=list(self.stocks.keys()),
            totalReturnPct=totalReturnPct,
            totalFund=self.totalFund,
            alphaVsBuyHold=alphaVsBuyHold,
            alphaVsSP500=alphaVsSP500,
            sharpeRatio=round(sharpeRatio, 4),
            winRate=round(winRate, 2),
            numTrades=totalTrades,
            numPatterns=totalPatterns,
            perStockResults=perStockResults,
            configJson=json.dumps({
                'populationSize': self.populationSize,
                'numGenerations': self.numGenerations,
                'forgiveness': self.forgiveness,
                'mcMethod': self.mcMethod,
                'mlEnabled': self.mlEnabled,
            }),
        )
        self.persistence.saveRunResult(runResult)

        # Persist individual backtest trades for frontend trade-history page
        self._saveBacktestTrades(runResult.runId, verbose=verbose)

        # Persist live allocation (slotAllocations, ghostSlots, stockAllocations)
        # so TradingBot uses the same intelligent fund allocation as backtesting
        if self.allocationResult:
            slotPerfFlat = None
            if self.allocationResult.slotPerformances:
                slotPerfFlat = {}
                for k, p in self.allocationResult.slotPerformances.items():
                    slotPerfFlat[k] = {
                        'totalTrades': getattr(p, 'totalTrades', 0),
                        'recentTrades': getattr(p, 'recentTrades', 0),
                        'recentReturnPct': getattr(p, 'recentReturnPct', 0),
                        'recentWinRate': getattr(p, 'recentWinRate', 0),
                        'ruleScore': getattr(p, 'ruleScore', 0) if hasattr(p, 'ruleScore') else 0,
                        'sharpeRatio': getattr(p, 'sharpeRatio', 0),
                        'recentSharpe': getattr(p, 'recentSharpe', 0),
                        'isGhost': getattr(p, 'isGhost', False),
                    }
            self.persistence.saveLiveAllocation(
                slotAllocations=dict(self.allocationResult.slotAllocations),
                ghostSlots=list(self.allocationResult.ghostSlots),
                stockAllocations=dict(self.allocationResult.stockAllocations),
                runId=runResult.runId,
                slotPerformances=slotPerfFlat,
            )
            if verbose:
                print(f"  [Persistence] Live allocation saved: {len(self.allocationResult.slotAllocations)} slots, "
                      f"{len(self.allocationResult.ghostSlots)} ghost")

        # Upload charts to Firestore run_charts collection
        self._uploadChartsToFirestore(runResult.runId, verbose=verbose)

        # Save patterns with ranked replacement (grouped by interval)
        for sym, res in self.results.items():
            if res.refinedPatternBank:
                # Build a lookup of MC scores from rankings
                mcScoreLookup = {}  # (interval, fitness) → (compositeScore, sharpe, winRate, returnPct)
                if res.patternRankings:
                    for interval, ranked in res.patternRankings.items():
                        for pattern, score in ranked:
                            mcScoreLookup[(interval, id(pattern))] = score

                # Also pull per-interval MC results for aggregated stats
                mcResultLookup = res.mcResults or {}

                from collections import defaultdict
                patternsByInterval: Dict[str, list] = defaultdict(list)
                for p in res.refinedPatternBank.patterns:
                    interval = p.interval or 'unknown'
                    # Get MC composite score from ranking
                    mcScore = mcScoreLookup.get((interval, id(p)), 0.0)

                    # Get MC per-interval stats
                    mcRes = mcResultLookup.get(interval, None)
                    mcSharpe  = mcRes.avgSharpeRatio        if mcRes else 0.0
                    mcWinRate = mcRes.probabilityOfProfit / 100.0 if mcRes else 0.0
                    mcReturn  = mcRes.expectedReturnPct / 100.0   if mcRes else 0.0

                    sp = StoredPattern(
                        symbol=sym,
                        interval=interval,
                        genesJson=json.dumps([{
                            'expectedPriceChangePct': g.expectedPriceChangePct,
                            'expectedVolumePct': g.expectedVolumePct,
                            'candleType': g.candleType.value,
                            'minBodyPct': g.minBodyPct,
                        } for g in p.genes]),
                        fitness=p.fitness,
                        accuracy=p.getAccuracy(),
                        mcCompositeScore=mcScore,
                        mcSharpe=mcSharpe,
                        mcWinRate=mcWinRate,
                        mcReturn=mcReturn,
                    )
                    patternsByInterval[interval].append(sp)
                for interval, storedPatterns in patternsByInterval.items():
                    self.persistence.updatePatternRanksIfBetter(
                        newPatterns=storedPatterns,
                        symbol=sym,
                        interval=interval,
                    )

        if verbose:
            summary = self.persistence.getSummary()
            print(f"  Saved: run #{summary.get('totalRuns', 0)}, "
                  f"{totalPatterns} patterns, return={totalReturnPct:+.2f}%, "
                  f"alpha={alphaVsBuyHold:+.2f}%, sharpe={sharpeRatio:.2f}, "
                  f"trades={totalTrades}")

    # =====================================================================
    # Step 5 — Train Per-Stock ML Models
    # =====================================================================

    def _trainStockMLModels(self, verbose: bool):
        """
        Train a StockMLModel for each stock PER INTERVAL using its refined patterns,
        historical data, and backtest results.

        Each model learns:
          - Technical indicators → forward return / direction prediction
          - Which refined patterns work in which market conditions
          - Signal generation: BUY / SELL / HOLD with confidence
        
        Training uses OLDER historical data (before the backtest period) to avoid overfitting.
        Separate models per interval ensure timeframe alignment (30m/1h/1d patterns match data).
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 4: TRAINING PER-STOCK ML MODELS (MULTI-TIMEFRAME)")
            print(f"{'=' * 70}")

        from datetime import datetime, timedelta
        fetcher = StockDataFetcher()
        
        # Calculate training period dates (chronologically BEFORE backtest period)
        # Training: from (trainPeriod + backtestPeriod) days ago → backtestPeriod days ago
        trainEndDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')
        trainStartDate = (datetime.now() - timedelta(days=self.mlTrainPeriodDays + self.backtestPeriodDays)).strftime('%Y-%m-%d')

        # Pre-compute per-interval training dates (adaptive, same logic as GA)
        YAHOO_INTERVAL_LIMITS = {
            '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60,
            '60m': 730, '1h': 730, '1d': 36500, '1wk': 36500, '1mo': 36500,
        }
        YAHOO_SAFETY_MARGIN = 7  # days buffer — Yahoo rejects requests at exact boundary
        MIN_TRAINING_DAYS = 30
        trainStartDt = datetime.strptime(trainStartDate, '%Y-%m-%d')
        trainEndDt = datetime.strptime(trainEndDate, '%Y-%m-%d')

        # Maps interval → (start, end) dates for training
        feasibleIntervals: dict = {}
        for interval, _ in self.intervals:
            maxDays = YAHOO_INTERVAL_LIMITS.get(interval, 730)
            daysFromToday = (datetime.now() - trainStartDt).days
            intTrainStart = trainStartDate

            if daysFromToday > maxDays:
                earliestFeasible = datetime.now() - timedelta(days=maxDays - YAHOO_SAFETY_MARGIN)
                if earliestFeasible >= trainEndDt:
                    if verbose:
                        print(f"  ⚠ SKIPPING {interval} ML training: Yahoo {maxDays}d "
                              f"limit cannot reach training period ending {trainEndDate}.")
                    continue
                intTrainStart = earliestFeasible.strftime('%Y-%m-%d')
                shortenedDays = (trainEndDt - earliestFeasible).days
                if shortenedDays < MIN_TRAINING_DAYS:
                    if verbose:
                        print(f"  ⚠ SKIPPING {interval} ML training: Only {shortenedDays}d "
                              f"available (need ≥{MIN_TRAINING_DAYS}d).")
                    continue
                if verbose:
                    print(f"  ℹ {interval} ML: Shortened training to {intTrainStart} → "
                          f"{trainEndDate} ({shortenedDays}d) to fit Yahoo {maxDays}d limit.")
            feasibleIntervals[interval] = (intTrainStart, trainEndDate)

        # Pre-fetch per-symbol metadata once (Firestore, not parallelised)
        metaCache: Dict[str, object] = {}
        for symbol in self.results:
            metaCache[symbol] = self.persistence.getStockMetadata(symbol)

        # Build a flat task list: one entry per (symbol, interval) that is feasible
        mlTasks = []
        for symbol, result in self.results.items():
            patternBank = result.refinedPatternBank
            if patternBank is None or len(patternBank.patterns) == 0:
                patternBank = PatternBank(symbol=symbol, patterns=[])
            stratFeats = (getattr(self, '_strategyFeatures', {}) or {}).get(symbol)
            meta = metaCache.get(symbol)
            sentData = self.sentimentData.get(symbol)
            for interval, _ in self.intervals:
                if interval not in feasibleIntervals:
                    continue
                intStart, intEnd = feasibleIntervals[interval]
                fwdPeriods = min(self.mlForwardPeriods, 5) if interval == '1d' else self.mlForwardPeriods
                mlTasks.append((symbol, result, interval, intStart, intEnd,
                                 patternBank, stratFeats, meta, sentData, fwdPeriods))

        def _trainOneModel(task):
            (symbol, result, interval, intStart, intEnd,
             patternBank, stratFeats, meta, sentData, fwdPeriods) = task
            localFetcher = StockDataFetcher()
            df = localFetcher.fetchData(symbol, interval=interval, start=intStart, end=intEnd)
            if df is None or len(df) < 60:
                return symbol, interval, None, None
            model = StockMLModel(symbol=symbol, forgiveness=self.forgiveness)
            if stratFeats is not None:
                model.setStrategyFeatures(stratFeats)
            if meta:
                model.setStockMetadata({
                    'sector': meta.sector,
                    'industry': meta.industry,
                    'marketCapBucket': meta.marketCapBucket,
                    'supplyChainUp': meta.supplyChainUp,
                    'supplyChainDown': meta.supplyChainDown,
                    'relatedTickers': meta.relatedTickers,
                    'sectorPeers': meta.sectorPeers,
                    'competesWidth': meta.competesWidth,
                    'portfolioType': meta.portfolioType or 'manual',
                    'autoScore': meta.autoScore or 0.0,
                    'autoAddedFrom': meta.autoAddedFrom or [],
                    'connectionType': (meta.autoAddedReason or '').split(' ')[0].lower()
                                      if meta.portfolioType == 'automatic' else '',
                })
            metrics = model.train(df, patternBank, forwardPeriods=fwdPeriods,
                                  sentimentData=sentData, verbose=False)
            return symbol, interval, model, metrics

        mlWorkers = min(6, len(mlTasks))
        if verbose:
            print(f"\n  [Parallel] Training {len(mlTasks)} ML models "
                  f"({mlWorkers} workers) ...")

        if mlWorkers > 1 and len(mlTasks) > 1:
            with ThreadPoolExecutor(max_workers=mlWorkers) as pool:
                for sym, interval, model, metrics in pool.map(_trainOneModel, mlTasks):
                    if model is None:
                        continue
                    self.stockMLModels[(sym, interval)] = model
                    if verbose and metrics and metrics.directionAccuracy > 0:
                        print(f"    {sym}/{interval}: {metrics.directionAccuracy*100:.1f}% accuracy")
        else:
            for task in mlTasks:
                sym, interval, model, metrics = _trainOneModel(task)
                if model is None:
                    continue
                self.stockMLModels[(sym, interval)] = model
                if verbose and metrics and metrics.directionAccuracy > 0:
                    print(f"    {sym}/{interval}: {metrics.directionAccuracy*100:.1f}% accuracy")

        # Store the daily model as the default and generate predictions
        for symbol, result in self.results.items():
            if (symbol, '1d') in self.stockMLModels:
                dailyModel = self.stockMLModels[(symbol, '1d')]
                if dailyModel.isTrained:
                    result.stockMLModel = dailyModel
                    localFetcher = StockDataFetcher()
                    df_daily = localFetcher.fetchData(symbol, interval='1d',
                                                      start=trainStartDate, end=trainEndDate)
                    if df_daily is not None and len(df_daily) > 0:
                        result.stockPredictions = dailyModel.predictBatch(df_daily)

    # =====================================================================
    # Step 4b — Augment ML Models with MC Simulation Data
    # =====================================================================

    def _augmentMLFromMC(self, verbose: bool):
        """
        Feed MC simulation trade data back into trained StockMLModels so
        they learn from synthetic outcomes (pattern robustness signal).

        Only augments models that are already trained and have MC results.
        Synthetic rows receive 0.5× weight to avoid overwhelming real data.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 4b: AUGMENTING ML MODELS WITH MC DATA")
            print(f"{'=' * 70}")

        augmented = 0
        for symbol, result in self.results.items():
            if not result.mcResults:
                if verbose:
                    print(f"  {symbol}: No MC results, skipping")
                continue

            for interval, _ in self.intervals:
                model = self.stockMLModels.get((symbol, interval))
                if model is None or not model.isTrained:
                    continue

                # Filter MC results relevant to this interval
                intervalMC = {}
                for key, mcResult in result.mcResults.items():
                    if mcResult.interval == interval or key == interval:
                        intervalMC[key] = mcResult

                if not intervalMC:
                    # Fall back to all MC results if interval keys don't match
                    intervalMC = result.mcResults

                if verbose:
                    print(f"  {symbol}/{interval}: Augmenting from "
                          f"{len(intervalMC)} MC result set(s)...")

                model.augmentFromMC(intervalMC, verbose=verbose)
                augmented += 1

        if verbose:
            print(f"\n  Augmented {augmented} model(s) with MC data")

    # =====================================================================
    # Step 5b — MC Phase 2: Full-Pipeline Simulation
    # =====================================================================

    def _runMCPhase2(self, verbose: bool):
        """
        Monte Carlo Phase 2: run MC simulations using the FULL trading
        pipeline — patterns + ML models + synthetic headlines + sentiment +
        Trading Decider.

        Unlike Phase 1 (patterns only), this measures what the complete
        system would produce on synthetic data.  Results are fed back into
        the ML models for self-improvement.

        Steps per stock/interval:
          1. Build a MCSyntheticHeadlineGenerator with the real SentimentAnalyzer
          2. Run MCMCSimulator.simulateFullPipeline() with all components
          3. Feed Phase 2 results into StockMLModel.augmentFromPhase2()
          4. Store Phase 2 results alongside Phase 1 results
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 5b: MC PHASE 2 — FULL PIPELINE SIMULATION")
            print(f"{'=' * 70}")

        # Number of Phase 2 simulations (lighter than Phase 1 since each
        # path is more expensive: ML inference + headline generation)
        P2_NUM_SIMS = max(50, self.mcNumSimulations // 4)
        P2_PERIODS = self.mcSimulationPeriods

        if verbose:
            print(f"  Simulations: {P2_NUM_SIMS} per interval "
                  f"(Phase 1 was {self.mcNumSimulations})")

        # Build headline generator with the real sentiment pipeline
        sentimentAnalyzer = None
        if self.sentimentEnabled:
            sentimentAnalyzer = SentimentAnalyzer(
                openAIKey=self.sentimentOpenAIKey,
                sentimentDecayHalfLifeDays=self.sentimentDecayHalfLife,
                cacheDir=self.sentimentCacheDir,
            )
        headlineGen = MCSyntheticHeadlineGenerator(
            headlineProbability=0.40,
            largeMoveBoost=0.25,
            sentimentAnalyzer=sentimentAnalyzer,
        )

        # Build list of (symbol, interval) tasks
        tasks = []
        for symbol, result in self.results.items():
            if not result.refinedPatternBank or len(result.refinedPatternBank.patterns) == 0:
                if verbose:
                    print(f"\n  {symbol}: No refined patterns — skipping Phase 2")
                continue
            stockFund = self.totalFund * result.allocation
            for interval, _ in self.intervals:
                mlModel = self.stockMLModels.get((symbol, interval))
                if mlModel is None or not mlModel.isTrained:
                    continue
                tasks.append((symbol, result, interval, stockFund, mlModel))

        if not tasks:
            if verbose:
                print("\n  No Phase 2 tasks to run.")
            return

        def _runOnePhase2(args):
            symbol, result, interval, stockFund, mlModel = args
            simulator = MCMCSimulator(
                initialFund=stockFund,
                forgiveness=self.forgiveness,
                numSimulations=P2_NUM_SIMS,
                simulationPeriods=P2_PERIODS,
                method=self.mcMethod,
            )
            phase2Results = simulator.simulateFullPipeline(
                patternBank=result.refinedPatternBank,
                symbol=symbol,
                interval=interval,
                calibrationStart=self._trainStartDate,
                calibrationEnd=self._trainEndDate,
                mlModel=mlModel,
                headlineGenerator=headlineGen,
                tradingDecider=self.tradingDecider,
                portfolioSignal=self.portfolioSignal,
                numSimulations=P2_NUM_SIMS,
                numPeriods=P2_PERIODS,
                targets=self.mcTargets,
                ruinThreshold=self.mcRuinThreshold,
                verbose=False,  # Avoid interleaved output; summary below
            )
            return (symbol, interval, result, mlModel, phase2Results)

        phase2Improved = 0
        maxWorkers = min(6, len(tasks))

        if maxWorkers > 1 and len(tasks) > 1:
            if verbose:
                print(f"  [Parallel] Running Phase 2 for {len(tasks)} stock/interval(s) "
                      f"with {maxWorkers} workers ...")
            with ThreadPoolExecutor(max_workers=maxWorkers) as pool:
                for fut in as_completed({pool.submit(_runOnePhase2, t): t for t in tasks}):
                    try:
                        symbol, interval, result, mlModel, phase2Results = fut.result()
                        p2Key = f'{interval}_phase2'
                        result.mcResults[p2Key] = phase2Results
                        if phase2Results.paths and len(phase2Results.paths) > 0:
                            mlModel.augmentFromPhase2(phase2Results, verbose=False)
                            phase2Improved += 1
                        if verbose:
                            print(f"    Done: {symbol}/{interval}")
                    except Exception as e:
                        print(f"  [ERROR] Phase 2 failed for a task: {e}")
        else:
            for t in tasks:
                symbol, interval, result, mlModel, phase2Results = _runOnePhase2(t)
                p2Key = f'{interval}_phase2'
                result.mcResults[p2Key] = phase2Results
                if phase2Results.paths and len(phase2Results.paths) > 0:
                    mlModel.augmentFromPhase2(phase2Results, verbose=verbose)
                    phase2Improved += 1

        if verbose:
            print(f"\n  Phase 2 complete: {phase2Improved} model(s) improved "
                  f"from full-pipeline simulation")

    # =====================================================================
    # Step 6 — Trading Decider (reconcile stock + portfolio signals)
    # =====================================================================

    def _runTradingDecider(self, verbose: bool):
        """
        Use the TradingDecider to blend per-stock ML signals with the
        portfolio ML signal and produce one final BUY/SELL/HOLD decision
        per stock.  This step runs BEFORE backtesting so the backtest
        can use the decider logic for every candle.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 6: TRADING DECIDER — Signal Reconciliation")
            print(f"{'=' * 70}")

        # Gather latest stock predictions
        latestPredictions: Dict[str, StockPrediction] = {}
        for symbol, result in self.results.items():
            if result.stockPredictions:
                latestPredictions[symbol] = result.stockPredictions[-1]
            else:
                latestPredictions[symbol] = StockPrediction()

        self.tradeDecisions = self.tradingDecider.decide(
            stockPredictions=latestPredictions,
            portfolioSignal=self.portfolioSignal,
            currentAllocations=self.stocks,
            verbose=verbose,
        )

    # =====================================================================
    # Step 7 — ML-Based Backtesting with Decider
    # =====================================================================

    # Notional fund used for ALL backtests (normalises returns across slots)
    NOTIONAL_FUND = 10_000.0

    # Slippage per side in basis points.  Models spread + market impact so
    # backtested returns are closer to real-world achievable performance.
    # 5 bps ≈ 0.05 % per fill; total round-trip cost ≈ 10 bps.
    SLIPPAGE_BPS = 5.0
    STOP_LOSS_PCT = 5.0          # per-trade stop-loss % (when enabled)
    USE_STOP_LOSS = True         # per-trade stop-loss (keep enabled)
    PORTFOLIO_DRAWDOWN_LIMIT = 15.0  # portfolio circuit breaker %

    def _mlBacktest(self, verbose: bool):
        """
        Run ML-based backtesting for each stock using trained ML models
        AND the TradingDecider to reconcile stock vs. portfolio signals.

        IMPORTANT: Every (stock, interval) slot is backtested with a
        FIXED notional fund ($10 k) regardless of allocation.  This
        normalises returns so the DynamicAllocator can scale P/L by
        each slot's dynamically changing allocation weight later.

        ALL stocks are backtested so performance can be tracked for
        potential re-allocation (ghost mode).
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 7: ML-BASED BACKTESTING (with Trading Decider)")
            print(f"{'=' * 70}")

        from datetime import datetime, timedelta
        endDate = datetime.now().strftime('%Y-%m-%d')
        startDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')

        if verbose:
            print(f"  Backtest (test) period: {startDate} to {endDate} "
                  f"({self.backtestPeriodDays:.0f} days)")
            print(f"  Notional fund per slot: ${self.NOTIONAL_FUND:,.0f} "
                  f"(returns will be scaled by dynamic allocation)")

        # Initialise empty backtestResults for all stocks
        for result in self.results.values():
            result.backtestResults = {}

        # Build flat task list: one per (symbol, interval) that has a trained model
        btTasks = []
        for symbol, result in self.results.items():
            allocation = self.stocks.get(symbol, 0.0)
            for interval, _ in self.intervals:
                model = self.stockMLModels.get((symbol, interval))
                if not model or not model.isTrained:
                    continue
                btTasks.append((symbol, result, interval, allocation,
                                 model, self.sentimentData.get(symbol)))

        def _runOneBacktest(task):
            symbol, result, interval, allocation, stockMLModel, sentSeries = task
            backtester = Backtester(initialFund=self.NOTIONAL_FUND,
                                    forgiveness=self.forgiveness,
                                    slippageBps=self.SLIPPAGE_BPS,
                                    stopLossPct=self.STOP_LOSS_PCT,
                                    portfolioDrawdownLimit=self.PORTFOLIO_DRAWDOWN_LIMIT,
                                    useStopLoss=self.useStopLoss)
            btResult = backtester.backtestWithDecider(
                mlModel=stockMLModel,
                symbol=symbol,
                startDate=startDate,
                endDate=endDate,
                tradingDecider=self.tradingDecider,
                portfolioSignal=self.portfolioSignal,
                allocation=allocation,
                interval=interval,
                holdPeriods=self.mlForwardPeriods,
                sentimentSeries=sentSeries,
                verbose=False,
                earningsBlackout=self.useEarningsBlackout,
            )
            return symbol, interval, btResult

        btWorkers = min(6, len(btTasks))
        if verbose:
            print(f"\n  [Parallel] Backtesting {len(btTasks)} slot(s) "
                  f"({btWorkers} workers) ...")

        if btWorkers > 1 and len(btTasks) > 1:
            with ThreadPoolExecutor(max_workers=btWorkers) as pool:
                for sym, interval, btResult in pool.map(_runOneBacktest, btTasks):
                    self.results[sym].backtestResults[f'ML_{interval}'] = btResult
                    if verbose:
                        pnl = btResult.getCompoundPnL()
                        returnPct = btResult.getCompoundReturnPct()
                        print(f"    {sym}/{interval}: {btResult.totalTrades} trades, "
                              f"${pnl:,.2f} ({returnPct:+.2f}%)")
        else:
            for task in btTasks:
                sym, interval, btResult = _runOneBacktest(task)
                self.results[sym].backtestResults[f'ML_{interval}'] = btResult
                if verbose:
                    pnl = btResult.getCompoundPnL()
                    returnPct = btResult.getCompoundReturnPct()
                    print(f"    {sym}/{interval}: {btResult.totalTrades} trades, "
                          f"${pnl:,.2f} ({returnPct:+.2f}%)")

    # =====================================================================
    # Step 7b — Walk-Forward Validation (anchored expanding window)
    # =====================================================================

    def _walkForwardValidation(self, verbose: bool):
        """
        Rolling walk-forward backtest:
          Window 1: train on [T-3Y .. T-2Y],  test on [T-2Y .. T-1Y]
          Window 2: train on [T-3Y .. T-1Y],  test on [T-1Y .. T]
          ...

        Each window re-trains the ML model from scratch on the expanding
        training set and backtests on the next unseen test period.  The
        aggregate across all windows gives a far more reliable performance
        estimate than a single in-sample backtest.

        Results are saved as `walkForward_<interval>` entries in
        `self.results[symbol].backtestResults`.
        """
        from datetime import datetime, timedelta

        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 7b: WALK-FORWARD VALIDATION")
            print(f"{'=' * 70}")

        totalDays = self.mlTrainPeriodDays + self.backtestPeriodDays
        windowTestDays = max(int(self.backtestPeriodDays), 90)
        numWindows = max(2, int(totalDays / windowTestDays) - 1)

        now = datetime.now()
        anchorStart = now - timedelta(days=totalDays)

        if verbose:
            print(f"  Total period: {totalDays} days")
            print(f"  Test window: {windowTestDays} days, {numWindows} windows")

        allWindowResults: Dict[str, List] = {}

        for wIdx in range(numWindows):
            trainEnd = anchorStart + timedelta(days=windowTestDays * (wIdx + 1))
            testStart = trainEnd
            testEnd = trainEnd + timedelta(days=windowTestDays)
            if testEnd > now:
                testEnd = now

            trainStartStr = anchorStart.strftime('%Y-%m-%d')
            trainEndStr = trainEnd.strftime('%Y-%m-%d')
            testStartStr = testStart.strftime('%Y-%m-%d')
            testEndStr = testEnd.strftime('%Y-%m-%d')

            if verbose:
                print(f"\n  ── Window {wIdx+1}/{numWindows}: "
                      f"train [{trainStartStr} → {trainEndStr}], "
                      f"test [{testStartStr} → {testEndStr}]")

            fetcher = StockDataFetcher()

            for symbol in list(self.results.keys()):
                allocation = self.stocks.get(symbol, 0.0)
                for interval, _ in self.intervals:
                    existingModel = self.stockMLModels.get((symbol, interval))
                    if existingModel is None:
                        continue

                    try:
                        wfModel = StockMLModel(symbol, forgiveness=self.forgiveness)
                        period = f'{int((trainEnd - anchorStart).days + 30)}d'
                        trainDf = fetcher.fetchData(symbol, interval=interval, period=period)
                        if trainDf is None or len(trainDf) < 50:
                            continue

                        mask = (trainDf.index >= pd.Timestamp(trainStartStr)) & \
                               (trainDf.index <= pd.Timestamp(trainEndStr))
                        trainSlice = trainDf[mask]
                        if len(trainSlice) < 50:
                            continue

                        patternBank = PatternBank()
                        if hasattr(existingModel, '_patternBank') and existingModel._patternBank:
                            patternBank = existingModel._patternBank

                        wfModel.train(
                            trainSlice,
                            forwardPeriods=self.mlForwardPeriods,
                            patternBank=patternBank,
                        )

                        if not wfModel.isTrained:
                            continue

                        backtester = Backtester(
                            initialFund=self.NOTIONAL_FUND,
                            forgiveness=self.forgiveness,
                            slippageBps=self.SLIPPAGE_BPS,
                            stopLossPct=self.STOP_LOSS_PCT,
                            portfolioDrawdownLimit=self.PORTFOLIO_DRAWDOWN_LIMIT,
                            useStopLoss=self.useStopLoss,
                        )
                        btResult = backtester.backtestWithDecider(
                            mlModel=wfModel,
                            symbol=symbol,
                            startDate=testStartStr,
                            endDate=testEndStr,
                            tradingDecider=self.tradingDecider,
                            portfolioSignal=self.portfolioSignal,
                            allocation=allocation,
                            interval=interval,
                            holdPeriods=self.mlForwardPeriods,
                            verbose=False,
                            earningsBlackout=self.useEarningsBlackout,
                        )

                        key = f"{symbol}/{interval}"
                        if key not in allWindowResults:
                            allWindowResults[key] = []
                        allWindowResults[key].append(btResult)

                        if verbose:
                            pnl = btResult.getCompoundPnL()
                            returnPct = btResult.getCompoundReturnPct()
                            print(f"    {symbol}/{interval}: "
                                  f"{btResult.totalTrades} trades, "
                                  f"${pnl:,.2f} ({returnPct:+.2f}%)")

                    except Exception as e:
                        if verbose:
                            print(f"    {symbol}/{interval}: FAILED — {e}")

        # Aggregate: create a combined BacktestResult per slot
        for key, windowResults in allWindowResults.items():
            symbol, interval = key.split('/')
            combined = BacktestResult(initialBalance=self.NOTIONAL_FUND)
            for wr in windowResults:
                for t in wr.trades:
                    rpct = t.get('returnPct', 0)
                    combined.addTrade(
                        isSuccessful=t.get('dollarPnL', 0) > 0,
                        returnPct=abs(rpct) if rpct else 0,
                        timestamp=t.get('timestamp', t.get('entryTime')),
                        patternId=-2,
                        entryPrice=t.get('entryPrice', 0),
                        exitPrice=t.get('exitPrice', 0),
                        fundAllocation=t.get('fundAllocation', 0),
                        dollarPnL=t.get('dollarPnL', 0),
                        confidence=t.get('confidence', 0),
                        exitTimestamp=t.get('exitTimestamp'),
                    )

            if symbol in self.results:
                self.results[symbol].backtestResults[f'WalkForward_{interval}'] = combined

            if verbose:
                pnl = combined.getCompoundPnL()
                returnPct = combined.getCompoundReturnPct()
                print(f"\n  ▸ {key} aggregate: {combined.totalTrades} trades, "
                      f"${pnl:,.2f} ({returnPct:+.2f}%)")

    # =====================================================================
    # Step 8 — Dynamic Fund Allocation (Continuous Throughout Backtest)
    # =====================================================================

    def _dynamicReallocation(self, verbose: bool):
        """
        Replay ALL backtest trades chronologically with dynamic allocation.

        Every slot was backtested with a fixed $10k notional fund (Step 7).
        This method:
          1. Collects all trades from all (stock, interval) slots
          2. Passes them to DynamicAllocator.replayWithDynamicAllocation()
          3. The allocator walks through time in evaluation windows:
             - After each window, evaluates every slot's rolling performance
             - Adjusts allocation weights (rule + ML blend)
             - Scales each trade's P/L by (totalFund × slotAlloc) / notionalFund
          4. Writes the dynamically-scaled P/L back into BacktestResult
             objects so downstream code (summary, visualization) uses them

        Ghost slots (0 %) still produce trades that are tracked for
        recovery detection but have $0 real P/L.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 8b: DYNAMIC FUND ALLOCATION (Main + Connected)")
            print(f"{'=' * 70}")

        # --- 1. Collect all slot trades ---
        #     ONLY include symbols that are in self.stocks (have a valid
        #     allocation).  Any orphaned results are ignored.
        allSlotTrades: Dict[str, list] = {}
        for symbol, result in self.results.items():
            if symbol not in self.stocks:
                if verbose:
                    print(f"  [Alloc] Skipping orphaned result for {symbol} "
                          f"(not in self.stocks)")
                continue
            for intervalKey, btResult in result.backtestResults.items():
                if intervalKey.startswith('WalkForward_'):
                    continue
                interval = intervalKey.replace('ML_', '')
                slotStr = f"{symbol}/{interval}"
                if btResult.trades:
                    allSlotTrades[slotStr] = list(btResult.trades)

        if not allSlotTrades:
            if verbose:
                print("  No backtest trades to replay")
            return

        # --- 2. Compute initial per-slot allocations (equal split) ---
        #     Only allocate to slots that actually produced trades.
        initialAllocations: Dict[str, float] = {}
        for symbol, alloc in self.stocks.items():
            if symbol not in self.results:
                continue
            intervals = [
                intervalKey.replace('ML_', '')
                for intervalKey in self.results[symbol].backtestResults
                if not intervalKey.startswith('WalkForward_')
                and f"{symbol}/{intervalKey.replace('ML_', '')}" in allSlotTrades
            ]
            nInt = max(len(intervals), 1)
            for interval in intervals:
                slotStr = f"{symbol}/{interval}"
                initialAllocations[slotStr] = alloc / nInt

        # Normalise to sum = 1.0
        totalAlloc = sum(initialAllocations.values())
        if totalAlloc > 0 and abs(totalAlloc - 1.0) > 0.01:
            initialAllocations = {k: v / totalAlloc
                                  for k, v in initialAllocations.items()}

        if verbose:
            print(f"  Initial allocations:")
            for s in sorted(initialAllocations):
                print(f"    {s}: {initialAllocations[s]*100:.1f}%")

        # --- 3. Run dynamic replay ---
        replayResult = self.fundAllocator.replayWithDynamicAllocation(
            allSlotTrades=allSlotTrades,
            totalFund=self.totalFund,
            initialAllocations=initialAllocations,
            notionalFund=self.NOTIONAL_FUND,
            verbose=verbose,
        )

        # --- 4. Write replayed P/L back into BacktestResult objects ---
        skippedSlots = []
        for symbol, result in self.results.items():
            for intervalKey, btResult in result.backtestResults.items():
                interval = intervalKey.replace('ML_', '')
                slotStr = f"{symbol}/{interval}"

                replayedTrades = replayResult.replayedTrades.get(slotStr, [])
                initSlotAlloc = initialAllocations.get(slotStr, 0.0)
                initSlotFund = self.totalFund * initSlotAlloc
                scaleFactor = (initSlotFund / self.NOTIONAL_FUND) if self.NOTIONAL_FUND > 0 else 0.0

                if len(replayedTrades) != len(btResult.trades):
                    if btResult.trades:
                        skippedSlots.append(slotStr)
                        if verbose:
                            print(f"  [Alloc] CRITICAL: trade count mismatch for "
                                  f"{slotStr} (replayed={len(replayedTrades)}, "
                                  f"original={len(btResult.trades)}) — applying proportional scaling")
                        # Fallback: scale raw P/L so per-stock return uses allocated capital
                        for t in btResult.trades:
                            raw = t.get('dollarPnL', 0.0)
                            t['dollarPnL'] = raw * scaleFactor
                        btResult.initialBalance = initSlotFund
                        totalScaled = sum(t.get('dollarPnL', 0) for t in btResult.trades)
                        btResult.totalReturnPct = (totalScaled / initSlotFund * 100) if initSlotFund > 0 else 0.0
                        btResult.successfulTrades = sum(1 for t in btResult.trades if t.get('dollarPnL', 0) > 0)
                        btResult.unsuccessfulTrades = len(btResult.trades) - btResult.successfulTrades
                    continue

                runningBalance = initSlotFund
                prevSlotAlloc = initSlotAlloc
                for i, rTrade in enumerate(replayedTrades):
                    btResult.trades[i]['dollarPnL'] = rTrade['dollarPnL']
                    btResult.trades[i]['fundAllocation'] = rTrade['fundAllocation']
                    curSlotAlloc = rTrade.get('slotAllocation', prevSlotAlloc)
                    btResult.trades[i]['slotAllocation'] = curSlotAlloc

                    if abs(curSlotAlloc - prevSlotAlloc) > 1e-6:
                        newSlotCapital = self.totalFund * curSlotAlloc
                        capitalFlow = newSlotCapital - (self.totalFund * prevSlotAlloc)
                        runningBalance += capitalFlow
                        prevSlotAlloc = curSlotAlloc

                    runningBalance += rTrade['dollarPnL']
                    btResult.trades[i]['balanceAfter'] = runningBalance

                btResult.initialBalance = initSlotFund
                btResult.finalBalance = runningBalance

                scaledWins = sum(
                    1 for t in btResult.trades if t.get('dollarPnL', 0) > 0
                )
                scaledLosses = sum(
                    1 for t in btResult.trades if t.get('dollarPnL', 0) <= 0
                )
                btResult.successfulTrades = scaledWins
                btResult.unsuccessfulTrades = scaledLosses
                totalScaledPnL = sum(
                    t.get('dollarPnL', 0) for t in btResult.trades
                )
                btResult.totalReturnPct = (
                    (totalScaledPnL / initSlotFund * 100)
                    if initSlotFund > 0 else 0.0
                )

        if skippedSlots and verbose:
            print(f"  [Alloc] ⚠ {len(skippedSlots)} slot(s) had trade count "
                  f"mismatches: {skippedSlots}")

        # --- 5. Build AllocationResult for downstream (summary, charts) ---
        self.allocationResult = AllocationResult(
            slotAllocations=replayResult.finalAllocations,
            stockAllocations={},
            slotPerformances=replayResult.finalPerformances,
            ghostSlots=[s for s, a in replayResult.finalAllocations.items()
                        if a < 0.001],
            restoredSlots=[],
            method=replayResult.method,
            previousAllocations=dict(self.stocks),
            allocationHistory=replayResult.allocationHistory,
        )

        # Aggregate per-stock allocations
        for slotStr, alloc in replayResult.finalAllocations.items():
            sym = slotStr.split('/')[0]
            self.allocationResult.stockAllocations[sym] = (
                self.allocationResult.stockAllocations.get(sym, 0.0) + alloc
            )
        self.allocationResult.newAllocations = dict(
            self.allocationResult.stockAllocations
        )

        # Update per-stock allocations
        newStockAllocs = self.allocationResult.stockAllocations
        if newStockAllocs:
            self.stocks = newStockAllocs
            # Sync res.allocation so _saveRunToPersistence uses the post-reallocation
            # fraction when computing stockFund (avoids mismatch with scaled dollarPnL).
            for sym, res in self.results.items():
                if sym in newStockAllocs:
                    res.allocation = newStockAllocs[sym]

    # =====================================================================
    # Step 5 — Train Portfolio-Level ML Model
    # =====================================================================

    def _trainPortfolioMLModel(self, verbose: bool):
        """
        Train the cross-stock portfolio ML model.

        Requires all per-stock models to be trained first.  Learns:
          - Cross-stock correlations and lead-lag relationships
          - Portfolio risk regime classification
          - Hedging signals (when to reduce exposure)
          - Dynamic allocation recommendations
          - Mean-reversion opportunities between correlated stocks
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 5: TRAINING PORTFOLIO ML MODEL")
            print(f"{'=' * 70}")

        if len(self.stockMLModels) < 2:
            if verbose:
                print("    Need ≥2 trained stock models for portfolio model")
            return

        from datetime import datetime, timedelta
        
        # Calculate training period dates (same as stock ML training)
        trainEndDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')
        trainStartDate = (datetime.now() - timedelta(days=self.mlTrainPeriodDays + self.backtestPeriodDays)).strftime('%Y-%m-%d')

        # Extract daily models for portfolio training (portfolio analysis uses daily timeframe)
        # Fall back to hourly models if daily didn't train
        dailyModels: Dict[str, StockMLModel] = {}
        for (symbol, interval), model in self.stockMLModels.items():
            if interval == '1d' and model.isTrained:
                dailyModels[symbol] = model
        # If a stock has no 1d model, use 1h as fallback
        for (symbol, interval), model in self.stockMLModels.items():
            if symbol not in dailyModels and interval == '1h' and model.isTrained:
                dailyModels[symbol] = model
                if verbose:
                    print(f"    {symbol}: Using 1h model as fallback (no trained 1d model)")

        if len(dailyModels) < 2:
            if verbose:
                print("    Need ≥2 trained daily stock models for portfolio model")
            return

        # Fetch aligned daily data for all stocks (training period)
        fetcher = StockDataFetcher()
        stockDataDict: Dict[str, pd.DataFrame] = {}
        for symbol in dailyModels:
            df = fetcher.fetchData(symbol, interval='1d', start=trainStartDate, end=trainEndDate)
            if df is not None and len(df) > 40:
                stockDataDict[symbol] = df

        if len(stockDataDict) < 2:
            if verbose:
                print("    Insufficient data for portfolio model")
            return

        self.portfolioMLModel = PortfolioMLModel(
            stockModels=dailyModels,  # Use daily models only
            allocations=self.stocks,
            totalFund=self.totalFund,
        )

        # Inject strategy features and metadata
        if hasattr(self, '_strategyFeatures'):
            self.portfolioMLModel.setStrategyFeatures(self._strategyFeatures)
        if hasattr(self, '_crossStockRuleFeatures') and self._crossStockRuleFeatures is not None:
            self.portfolioMLModel.setCrossStockRuleFeatures(self._crossStockRuleFeatures)
        # Inject full stock metadata (sector, supply chain, relationships, auto-stock info)
        metaDict = {}
        for sym in self.stocks:
            meta = self.persistence.getStockMetadata(sym)
            if meta:
                metaDict[sym] = {
                    'sector': meta.sector,
                    'industry': meta.industry,
                    'marketCapBucket': meta.marketCapBucket,
                    'supplyChainUp': meta.supplyChainUp,
                    'supplyChainDown': meta.supplyChainDown,
                    'relatedTickers': meta.relatedTickers,
                    'sectorPeers': meta.sectorPeers,
                    'competesWidth': meta.competesWidth,
                    'portfolioType': meta.portfolioType or 'manual',
                    'autoScore': meta.autoScore or 0.0,
                    'autoAddedFrom': meta.autoAddedFrom or [],
                    'autoAddedReason': meta.autoAddedReason or '',
                    'connectionType': (meta.autoAddedReason or '').split(' ')[0].lower()
                                      if meta.portfolioType == 'automatic' else '',
                }
        if metaDict:
            self.portfolioMLModel.setStockMetadata(metaDict)

        metrics = self.portfolioMLModel.train(
            stockDataDict=stockDataDict,
            forwardPeriods=self.mlPortfolioForwardPeriods,
            corrWindow=self.mlCorrWindow,
            sentimentDataDict=self.sentimentData if self.sentimentData else None,
            verbose=verbose,
        )

        # Generate current portfolio signal
        if self.portfolioMLModel.isTrained:
            latestPredictions = {}
            for symbol, result in self.results.items():
                if result.stockPredictions:
                    latestPredictions[symbol] = result.stockPredictions[-1]
                else:
                    latestPredictions[symbol] = StockPrediction()

            self.portfolioSignal = self.portfolioMLModel.predict(
                stockPredictions=latestPredictions,
                stockDataDict=stockDataDict,
                corrWindow=self.mlCorrWindow,
            )

            if verbose:
                sig = self.portfolioSignal
                print(f"\n    Portfolio Signal:")
                print(f"      Risk Regime       : {sig.riskRegime.value}")
                print(f"      Hedge Action      : {sig.hedgeAction.value}")
                print(f"      Expected Return   : {sig.expectedPortfolioReturn:+.4f}")
                print(f"      Portfolio Vol (ann): {sig.portfolioVolatility:.4f}")
                print(f"      Avg Cross-Corr    : {sig.avgCrossCorrelation:.3f}")
                print(f"      Confidence        : {sig.confidence:.2f}")

                if sig.allocationAdjustments:
                    print(f"      Allocation Adjustments:")
                    for adj in sig.allocationAdjustments:
                        print(f"        {adj.symbol}: {adj.currentAllocation*100:.1f}% "
                              f"→ {adj.suggestedAllocation*100:.1f}% "
                              f"({adj.reason})")

                if sig.meanReversionOpportunities:
                    print(f"      Mean-Reversion Opportunities:")
                    for opp in sig.meanReversionOpportunities:
                        print(f"        {opp}")

                if sig.leadLagSignals:
                    print(f"      Lead-Lag Relations:")
                    for ll in sig.leadLagSignals[:3]:
                        print(f"        {ll.leader} → {ll.follower} "
                              f"(lag={ll.lagPeriods}, r={ll.correlation:.3f})")

    # =====================================================================
    # Step 8b — Connected Stocks Evaluation
    # =====================================================================

    def _evaluateConnectedStocks(self, verbose: bool):
        """
        Evaluate connected/supply-chain stocks, update the automatic
        portfolio, then run the full pipeline for any connected stocks
        that were added or kept.

        Phase 1 — Discovery & scoring:
          1. Discover candidates from manual stock metadata
          2. Score using price correlation and lead-lag signals
          3. Add high-scoring candidates / remove underperformers

        Phase 2 — Full pipeline for connected stocks:
          4. Load stored patterns (skip GA — too expensive for auto stocks)
          5. Run MC simulations on stored patterns
          6. Train per-stock ML models (with connected stock features)
          7. Backtest with Trading Decider
          8. Include in dynamic fund allocation (capped)
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 8b: CONNECTED STOCKS EVALUATION")
            print(f"{'=' * 70}")

        try:
            # ── Phase 1: Discovery & update ──────────────────────────────
            stockDataDict: Dict[str, pd.DataFrame] = {}
            fetcher = StockDataFetcher()
            for symbol in self.stocks:
                try:
                    df = fetcher.fetchData(symbol, interval='1d', period='1y')
                    if df is not None and len(df) > 0:
                        stockDataDict[symbol] = df
                except Exception:
                    pass

            candidates = self.connectedStockManager.discoverCandidates(verbose)
            candidateSyms = {c.symbol for c in candidates}
            autoStocks = self.persistence.getAutoPortfolioStocks()
            needData = (candidateSyms | set(autoStocks.keys())) - set(stockDataDict.keys())

            if needData:
                if verbose:
                    print(f"  Fetching price data for {len(needData)} connected stock candidates...")
                for sym in needData:
                    try:
                        df = fetcher.fetchData(sym, interval='1d', period='1y')
                        if df is not None and len(df) > 0:
                            stockDataDict[sym] = df
                    except Exception:
                        pass

            runResults = {}
            for sym, result in self.results.items():
                if result.backtestResults:
                    totalRet = 0.0
                    totalTrades = 0
                    for interval, btRes in result.backtestResults.items():
                        totalRet += getattr(btRes, 'totalReturn', 0)
                        totalTrades += getattr(btRes, 'totalTrades', 0)
                    runResults[sym] = {
                        'returnPct': totalRet,
                        'trades': totalTrades,
                    }

            actions = self.connectedStockManager.updateAutoPortfolio(
                stockDataDict, runResults, verbose
            )

            added = [s for s, a in actions.items() if a == 'added']
            removed = [s for s, a in actions.items() if a == 'removed']
            kept = [s for s, a in actions.items() if a == 'kept']

            if verbose:
                print(f"\n  Connected stocks: {len(added)} added, "
                      f"{len(removed)} removed, {len(kept)} kept")

            # ── Phase 2: Full pipeline for active connected stocks ───────
            activeAutoSymbols = added + kept
            if not activeAutoSymbols:
                if verbose:
                    print("  No active connected stocks — skipping pipeline")
                return

            if verbose:
                print(f"\n  Running pipeline for {len(activeAutoSymbols)} "
                      f"connected stocks: {activeAutoSymbols}")

            autoMeta = self.persistence.getAutoPortfolioStocks()
            nAuto = len(activeAutoSymbols)
            # Reserve up to 25% of total fund for connected stocks
            autoTotalFraction = min(0.25, 0.05 * nAuto)
            perAutoFraction = autoTotalFraction / nAuto

            from datetime import datetime, timedelta

            trainEndDate = getattr(self, '_trainEndDate',
                                   (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d'))
            trainStartDate = getattr(self, '_trainStartDate',
                                     (datetime.now() - timedelta(days=self.mlTrainPeriodDays + self.backtestPeriodDays)).strftime('%Y-%m-%d'))
            btEndDate = datetime.now().strftime('%Y-%m-%d')
            btStartDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')

            # Fetch sentiment for connected stocks that don't already have it
            missingsentiment = [s for s in activeAutoSymbols
                                if s not in self.sentimentData]
            if missingsentiment and self.sentimentEnabled:
                try:
                    if verbose:
                        print(f"\n  Fetching sentiment for {len(missingsentiment)} "
                              f"connected stocks...")
                    from SentimentAnalysis import SentimentAnalyzer
                    _sa = SentimentAnalyzer(
                        openAIKey=self.sentimentOpenAIKey,
                        sentimentDecayHalfLifeDays=self.sentimentDecayHalfLife,
                        cacheDir=self.sentimentCacheDir,
                    )
                    autoSentiment = _sa.fetchHistorical(
                        symbols=missingsentiment,
                        startDate=getattr(self, '_trainStartDate', btStartDate),
                        endDate=btEndDate,
                        priceDataDict=stockDataDict,
                        newsAPIKey=self.sentimentNewsAPIKey,
                        finnhubKey=self.sentimentFinnhubKey,
                        alphaVantageKey=self.sentimentAlphaVantageKey,
                        verbose=False,
                        maxOpenAIPerSymbol=2,
                    )
                    self.sentimentData.update(autoSentiment)
                    if verbose:
                        n = sum(len(s) for s in autoSentiment.values())
                        print(f"    Got {n} day-scores for {len(autoSentiment)} symbols")
                except Exception as e:
                    if verbose:
                        print(f"    Sentiment fetch failed (non-fatal): {e}")

            for sym in activeAutoSymbols:
              try:
                if verbose:
                    print(f"\n    --- {sym} (connected stock) ---")

                result = StockPipelineResult(symbol=sym, allocation=perAutoFraction,
                                            portfolioType='automatic')

                # Step A: Load stored patterns (no GA for connected stocks)
                storedPatterns = self.persistence.loadAllActivePatterns(sym, topN=50)
                if storedPatterns:
                    patterns = []
                    for sp in storedPatterns:
                        try:
                            import json as _json
                            genes = _json.loads(sp.genesJson)
                            geneObjs = [PatternGene(
                                expectedPriceChangePct=g['expectedPriceChangePct'],
                                expectedVolumePct=g.get('expectedVolumePct', 1.0),
                                candleType=CandleType(g.get('candleType', 'BULLISH')),
                                minBodyPct=g.get('minBodyPct', 0.3),
                            ) for g in genes]
                            chrom = PatternChromosome(
                                genes=geneObjs, fitness=sp.fitness,
                                interval=sp.interval, symbol=sym,
                            )
                            patterns.append(chrom)
                        except Exception:
                            continue
                    result.refinedPatternBank = PatternBank(symbol=sym, patterns=patterns)
                    if verbose:
                        print(f"      Loaded {len(patterns)} stored patterns")
                else:
                    result.refinedPatternBank = PatternBank(symbol=sym, patterns=[])
                    if verbose:
                        print(f"      No stored patterns — ML will use technical indicators only")

                # Step B: MC simulation on stored patterns (lighter: 200 sims)
                if result.refinedPatternBank and len(result.refinedPatternBank.patterns) > 0:
                    stockFund = self.totalFund * perAutoFraction
                    simulator = MCMCSimulator(
                        initialFund=stockFund,
                        forgiveness=self.forgiveness,
                        numSimulations=min(200, self.mcNumSimulations),
                        simulationPeriods=self.mcSimulationPeriods,
                        method=self.mcMethod,
                    )
                    for interval, _ in self.intervals:
                        intPatterns = [p for p in result.refinedPatternBank.patterns
                                       if p.interval == interval]
                        if not intPatterns:
                            continue
                        try:
                            intBank = PatternBank(symbol=sym, patterns=intPatterns)
                            mcRes = simulator.simulate(
                                patternBank=intBank, symbol=sym, interval=interval,
                                calibrationPeriod=self.mcCalibrationPeriod,
                                targets=self.mcTargets,
                                ruinThreshold=self.mcRuinThreshold,
                                verbose=False,
                            )
                            result.mcResults[interval] = mcRes
                        except Exception:
                            pass
                    if verbose and result.mcResults:
                        print(f"      MC simulation: {len(result.mcResults)} interval(s)")

                # Step C: Train per-stock ML model (with connected stock metadata)
                meta = autoMeta.get(sym)
                metaDict = {}
                if meta:
                    metaDict = {
                        'sector': meta.sector,
                        'industry': meta.industry,
                        'marketCapBucket': meta.marketCapBucket,
                        'supplyChainUp': meta.supplyChainUp,
                        'supplyChainDown': meta.supplyChainDown,
                        'relatedTickers': meta.relatedTickers,
                        'sectorPeers': meta.sectorPeers,
                        'competesWidth': meta.competesWidth,
                        'portfolioType': meta.portfolioType or 'automatic',
                        'connectionType': meta.autoAddedReason.split(' ')[0].lower() if meta.autoAddedReason else '',
                        'autoScore': meta.autoScore or 0.0,
                        'autoAddedFrom': meta.autoAddedFrom or [],
                    }
                    # Compute parent correlation from cached price data
                    parentSyms = meta.autoAddedFrom or []
                    parentCorrs = []
                    parentPerfs = []
                    symDf = stockDataDict.get(sym)
                    if symDf is not None and len(symDf) > 30:
                        symRet = symDf['close'].pct_change().dropna()
                        for ps in parentSyms:
                            parentDf = stockDataDict.get(ps)
                            if parentDf is not None and len(parentDf) > 30:
                                parentRet = parentDf['close'].pct_change().dropna()
                                common = symRet.index.intersection(parentRet.index)
                                if len(common) > 20:
                                    c = float(np.corrcoef(
                                        symRet.loc[common].values[-60:],
                                        parentRet.loc[common].values[-60:]
                                    )[0, 1])
                                    parentCorrs.append(abs(c))
                                    parentPerfs.append(float(parentRet.tail(20).sum()))
                    metaDict['parentCorrelation'] = float(np.mean(parentCorrs)) if parentCorrs else 0.0
                    metaDict['parentRecentPerf'] = float(np.mean(parentPerfs)) if parentPerfs else 0.0

                for interval, _ in self.intervals:
                    try:
                        df = fetcher.fetchData(sym, interval=interval,
                                               start=trainStartDate, end=trainEndDate)
                        if df is None or len(df) < 60:
                            continue

                        model = StockMLModel(symbol=sym, forgiveness=self.forgiveness)
                        if metaDict:
                            model.setStockMetadata(metaDict)

                        fwdPeriods = min(self.mlForwardPeriods, 5) if interval == '1d' else self.mlForwardPeriods
                        patternBank = result.refinedPatternBank or PatternBank(symbol=sym, patterns=[])

                        metrics = model.train(
                            df, patternBank,
                            forwardPeriods=fwdPeriods,
                            sentimentData=self.sentimentData.get(sym),
                            verbose=False,
                        )

                        self.stockMLModels[(sym, interval)] = model
                        if verbose and metrics and metrics.directionAccuracy > 0:
                            hasSent = 'sent' if self.sentimentData.get(sym) is not None else 'no-sent'
                            nPat = len(patternBank.patterns) if patternBank else 0
                            print(f"      {interval} ML: {metrics.directionAccuracy * 100:.1f}% accuracy "
                                  f"({nPat} patterns, {hasSent}, {len(df)} bars)")

                        if interval == '1d' and model.isTrained:
                            result.stockMLModel = model
                            df_pred = fetcher.fetchData(sym, interval='1d',
                                                        start=trainStartDate, end=trainEndDate)
                            if df_pred is not None and len(df_pred) > 0:
                                result.stockPredictions = model.predictBatch(df_pred)

                    except Exception:
                        continue

                # Step D: Backtest with Trading Decider
                for interval, _ in self.intervals:
                    mlModel = self.stockMLModels.get((sym, interval))
                    if not mlModel or not mlModel.isTrained:
                        continue
                    try:
                        backtester = Backtester(
                            initialFund=self.NOTIONAL_FUND,
                            forgiveness=self.forgiveness,
                            slippageBps=self.SLIPPAGE_BPS,
                            stopLossPct=self.STOP_LOSS_PCT,
                            portfolioDrawdownLimit=self.PORTFOLIO_DRAWDOWN_LIMIT,
                            useStopLoss=self.useStopLoss,
                        )
                        btResult = backtester.backtestWithDecider(
                            mlModel=mlModel, symbol=sym,
                            startDate=btStartDate, endDate=btEndDate,
                            tradingDecider=self.tradingDecider,
                            portfolioSignal=self.portfolioSignal,
                            allocation=perAutoFraction,
                            interval=interval,
                            holdPeriods=self.mlForwardPeriods,
                            sentimentSeries=self.sentimentData.get(sym),
                            verbose=False,
                            earningsBlackout=self.useEarningsBlackout,
                        )
                        result.backtestResults[f'ML_{interval}'] = btResult

                        if verbose:
                            pnl = btResult.getCompoundPnL()
                            trades = btResult.totalTrades
                            returnPct = btResult.getCompoundReturnPct()
                            print(f"      {interval} backtest: {trades} trades, "
                                  f"${pnl:,.2f} ({returnPct:+.2f}%)")
                    except Exception:
                        continue

                # Store result
                self.results[sym] = result
                self.stocks[sym] = perAutoFraction

              except Exception as e:
                if verbose:
                    print(f"      {sym} failed: {e}")

            # ── Phase 2b: Update ML scores from actual pipeline results ──
            if verbose:
                print(f"\n  Updating ML scores from pipeline results...")
            for sym in activeAutoSymbols:
                res = self.results.get(sym)
                if not res or not res.backtestResults:
                    continue

                totalTrades = 0
                totalWins = 0
                totalReturn = 0.0
                mlConfidence = 0.0

                for intKey, bt in res.backtestResults.items():
                    totalTrades += bt.totalTrades
                    totalWins += bt.successfulTrades
                    totalReturn += bt.getCompoundReturnPct() if hasattr(bt, 'getCompoundReturnPct') else 0

                # ML model confidence (if we have predictions)
                if res.stockPredictions and len(res.stockPredictions) > 0:
                    lastPred = res.stockPredictions[-1]
                    mlConfidence = lastPred.confidence

                winRate = (totalWins / totalTrades) if totalTrades > 0 else 0.0
                returnScore = min(1.0, max(0.0, (totalReturn + 10) / 30))
                winRateScore = winRate
                confidenceScore = mlConfidence
                tradeScore = min(1.0, totalTrades / 20.0)

                # Real ML-backed score: blend of actual performance
                realScore = (
                    0.30 * returnScore
                    + 0.25 * winRateScore
                    + 0.25 * confidenceScore
                    + 0.20 * tradeScore
                )

                reason = (f"ML evaluation: {totalReturn:+.1f}% return, "
                          f"{winRate * 100:.0f}% win rate, "
                          f"{mlConfidence * 100:.0f}% ML confidence, "
                          f"{totalTrades} trades")

                self.persistence.updateAutoStockScore(sym, realScore, reason)

                if verbose:
                    print(f"    {sym}: score {realScore:.2f} ({reason})")

            # ── Phase 3: Determine phase-out via portfolio ML ────────────
            if self.portfolioMLModel and self.portfolioMLModel.isTrained:
                latestPredictions = {}
                for sym, result in self.results.items():
                    if result.stockPredictions:
                        latestPredictions[sym] = result.stockPredictions[-1]
                    else:
                        latestPredictions[sym] = StockPrediction()

                adjustments = self.portfolioMLModel._computeAllocationAdjustments(
                    latestPredictions,
                    self.portfolioSignal.riskRegime if self.portfolioSignal else PortfolioSignal().riskRegime,
                )

                for adj in adjustments:
                    if adj.suggestedAllocation < 0.005 and adj.symbol in activeAutoSymbols:
                        if verbose:
                            print(f"\n    ML recommends phasing out {adj.symbol}: "
                                  f"{adj.reason}")
                        self.connectedStockManager.pm.removeAutoStock(adj.symbol)
                        if adj.symbol in self.stocks:
                            del self.stocks[adj.symbol]
                        if adj.symbol in self.results:
                            del self.results[adj.symbol]

            # Re-normalise manual stock allocations (auto stocks don't affect normalisation)
            manualTotal = sum(v for k, v in self.stocks.items()
                             if k not in activeAutoSymbols)
            autoTotal = sum(v for k, v in self.stocks.items()
                           if k in activeAutoSymbols)
            if manualTotal > 0 and autoTotal > 0:
                scaleFactor = (1.0 - autoTotal) / manualTotal
                for k in list(self.stocks.keys()):
                    if k not in activeAutoSymbols:
                        self.stocks[k] *= scaleFactor

            if verbose:
                print(f"\n  Final allocations with connected stocks:")
                for sym, alloc in sorted(self.stocks.items()):
                    tag = ' (auto)' if sym in activeAutoSymbols else ''
                    print(f"    {sym}: {alloc * 100:.1f}%{tag}")

        except Exception as e:
            if verbose:
                print(f"  Connected stocks evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # =====================================================================
    # Benchmark Comparison — Buy-and-Hold vs Strategy
    # =====================================================================

    def _computeBenchmarks(self, verbose: bool):
        """
        Fetch buy-and-hold returns for each stock and the S&P 500 over the
        same period used by the backtester, so results can be compared
        directly to a passive strategy.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("BENCHMARK COMPARISON — Buy & Hold")
            print(f"{'=' * 70}")

        endDate = datetime.now().strftime('%Y-%m-%d')
        startDate = (datetime.now() - timedelta(days=self.backtestPeriodDays)).strftime('%Y-%m-%d')

        fetcher = StockDataFetcher()

        # Fetch for each portfolio stock + S&P 500 (parallel)
        tickers = list(self.stocks.keys()) + ['^GSPC']
        def _fetchBenchmark(ticker):
            try:
                return ticker, StockDataFetcher().fetchData(
                    ticker, interval='1d', start=startDate, end=endDate)
            except Exception:
                return ticker, None
        with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as pool:
            fetchedTickers = list(pool.map(_fetchBenchmark, tickers))

        for ticker, df in fetchedTickers:
            if df is None or df.empty or len(df) < 2:
                if verbose:
                    print(f"  {ticker}: insufficient data for benchmark")
                continue

            firstClose = df['close'].iloc[0]
            lastClose = df['close'].iloc[-1]
            returnPct = ((lastClose - firstClose) / firstClose) * 100

            normSeries = df['close'] / firstClose
            self.benchmarks[ticker] = {
                'returnPct': returnPct,
                'firstClose': firstClose,
                'lastClose': lastClose,
                'series': normSeries,
                'dates': df.index,
                'startDate': startDate,
                'endDate': endDate,
            }

            if verbose:
                label = 'S&P 500' if ticker == '^GSPC' else ticker
                print(f"  {label}: {returnPct:+.2f}% "
                      f"(${firstClose:,.2f} → ${lastClose:,.2f})")

        if verbose:
            # Compute weighted buy-and-hold for the portfolio
            weightedReturn = sum(
                self.stocks.get(sym, 0) * self.benchmarks[sym]['returnPct']
                for sym in self.stocks if sym in self.benchmarks
            )
            print(f"\n  Weighted Buy & Hold (portfolio mix): {weightedReturn:+.2f}%")
            if '^GSPC' in self.benchmarks:
                print(f"  S&P 500 Buy & Hold:                  "
                      f"{self.benchmarks['^GSPC']['returnPct']:+.2f}%")

    # =====================================================================
    # Step 9a — Fan Chart Visualisation  (separate image)
    # =====================================================================

    def _fig_to_b64(self, fig, dpi: int = 80, quality: int = 80, **_kwargs) -> str:
        """
        Render a matplotlib figure to a base64 data-URI.

        Strategy: save as PNG first (always works with Agg), then attempt
        to re-compress to JPEG via Pillow for a 3-5× size reduction.
        Firestore has a 1 MB per-document limit, so JPEG is important for
        multi-interval figures.  Falls back to PNG if Pillow is absent.
        """
        # Step 1: render to PNG via Agg (no quality param — always safe)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)

        # Step 2: re-compress to JPEG via Pillow (if available)
        try:
            from PIL import Image as _PILImage
            img = _PILImage.open(buf)
            jpgBuf = io.BytesIO()
            img.convert('RGB').save(jpgBuf, format='JPEG',
                                    quality=quality, optimize=True)
            jpgBuf.seek(0)
            encoded = base64.b64encode(jpgBuf.read()).decode('utf-8')
            return 'data:image/jpeg;base64,' + encoded
        except Exception:
            pass

        # Step 3: fallback — return PNG as-is
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return 'data:image/png;base64,' + encoded

    def _generateFanCharts(self, verbose: bool):
        """Generate standalone MC fan chart images (one per stock) and stage for Firestore upload."""
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 9a: GENERATING FAN CHARTS")
            print(f"{'=' * 70}")

        for symbol, result in self.results.items():
            if not result.mcResults:
                continue

            validResults = {k: v for k, v in result.mcResults.items() if v and v.paths}
            if not validResults:
                continue

            numIntervals = len(validResults)
            stockFund = self.totalFund * result.allocation

            fig, axes = plt.subplots(numIntervals, 1,
                                     figsize=(16, 9 * numIntervals))
            if numIntervals == 1:
                axes = [axes]

            fig.patch.set_facecolor('#0d1117')
            fig.suptitle(f'Monte Carlo Fan Chart — {symbol}  |  Fund: ${stockFund:,.0f}',
                         fontsize=16, fontweight='bold', y=0.99, color='#e6edf3')

            simulator = MCMCSimulator(
                initialFund=stockFund,
                forgiveness=self.forgiveness,
                numSimulations=self.mcNumSimulations,
                simulationPeriods=self.mcSimulationPeriods,
                method=self.mcMethod,
            )

            for idx, (interval, mcResult) in enumerate(sorted(validResults.items())):
                simulator._plotBalancePaths(axes[idx], mcResult, interval)
                axes[idx].set_facecolor('#161b22')
                axes[idx].tick_params(colors='#8b949e')
                axes[idx].xaxis.label.set_color('#8b949e')
                axes[idx].yaxis.label.set_color('#8b949e')
                axes[idx].title.set_color('#e6edf3')
                for spine in axes[idx].spines.values():
                    spine.set_edgecolor('#30363d')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.subplots_adjust(hspace=0.25)

            imgData = self._fig_to_b64(fig, dpi=80, quality=80)
            plt.close(fig)

            self._pendingCharts.append({
                'type': 'fan_chart',
                'symbol': symbol,
                'label': f'MC Fan Chart — {symbol}',
                'data': imgData,
            })
            if verbose:
                print(f"  Staged fan chart for {symbol}")

    # =====================================================================
    # Step 9b — Backtest Performance Graphs  (separate image)
    # =====================================================================

    def _generateBacktestGraphs(self, verbose: bool):
        """
        Generate backtest performance graphs (one image per stock).

        Layout per stock:
          Row 1: Cumulative P/L over time (all intervals combined)
          Row 2: Per-interval cumulative curves side-by-side
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 9b: GENERATING BACKTEST GRAPHS")
            print(f"{'=' * 70}")

        for symbol, result in self.results.items():
            if not result.backtestResults:
                continue

            stockFund = self.totalFund * result.allocation
            intervals = sorted(result.backtestResults.keys())
            numIntervals = len(intervals)

            fig, axes = plt.subplots(
                2, max(numIntervals, 1),
                figsize=(8 * max(numIntervals, 1), 14),
            )

            # Ensure axes is 2-D
            if numIntervals == 1:
                axes = axes.reshape(-1, 1)
            elif numIntervals == 0:
                plt.close(fig)
                continue

            fig.suptitle(
                f'Backtest Performance — {symbol}  |  '
                f'Fund: ${stockFund:,.0f}  |  Allocation: {result.allocation*100:.0f}%',
                fontsize=16, fontweight='bold', y=0.99,
            )

            # ---- Row 0: Combined cumulative curve (span all cols) ----
            axCombined = fig.add_subplot(2, 1, 1)
            self._plotCombinedBacktest(axCombined, result, stockFund)
            # Hide the individual top-row axes created by subplots
            for c in range(numIntervals):
                axes[0, c].set_visible(False)

            # ---- Row 1: Per-interval cumulative curves ----
            for col, interval in enumerate(intervals):
                self._plotIntervalBacktest(
                    axes[1, col], result.backtestResults[interval],
                    interval, stockFund,
                )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(hspace=0.35, wspace=0.3)

            imgData = self._fig_to_b64(fig, dpi=80, quality=80)
            plt.close(fig)

            self._pendingCharts.append({
                'type': 'backtest',
                'symbol': symbol,
                'label': f'Backtest Performance — {symbol}',
                'data': imgData,
            })
            if verbose:
                print(f"  Staged backtest chart for {symbol}")

    # ---- plot helpers ----------------------------------------------------

    def _plotCombinedBacktest(self, ax, result: StockPipelineResult,
                              stockFund: float):
        """Cumulative P/L across all intervals for one stock."""
        allTrades = []
        for interval, btResult in result.backtestResults.items():
            if interval.startswith('WalkForward_'):
                continue
            for trade in btResult.trades:
                allTrades.append({
                    'timestamp': trade.get('exitTimestamp', trade['timestamp']),
                    'returnPct': trade['returnPct'],
                    'fundAllocation': trade.get('fundAllocation', 0),
                    'dollarPnL': trade.get('dollarPnL', None),
                    'interval': interval,
                })

        if not allTrades:
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(f'{result.symbol} — Combined Backtest', fontweight='bold')
            return

        allTrades.sort(key=lambda t: t['timestamp'])

        timestamps = [allTrades[0]['timestamp']]
        values = [stockFund]
        cumProfit = 0.0

        for trade in allTrades:
            profit = trade.get('dollarPnL', trade['fundAllocation'] * (trade['returnPct'] / 100))
            cumProfit += profit
            timestamps.append(trade['timestamp'])
            values.append(stockFund + cumProfit)

        ax.plot(timestamps, values, lw=2.5, color='#2E7D32', alpha=0.9)
        ax.fill_between(timestamps, stockFund, values, alpha=0.15, color='#4CAF50')
        ax.axhline(stockFund, color='black', ls='--', lw=1, alpha=0.4,
                   label=f'Initial: ${stockFund:,.0f}')

        finalVal = stockFund + cumProfit
        retPct = (cumProfit / stockFund) * 100 if stockFund > 0 else 0
        totalTrades = len(allTrades)
        totalWins = sum(1 for t in allTrades if t['returnPct'] > 0)
        winRate = (totalWins / totalTrades * 100) if totalTrades > 0 else 0

        summaryText = (f'Trades: {totalTrades} | Win Rate: {winRate:.1f}% | '
                       rf'P/L: \${cumProfit:,.2f} ({retPct:+.2f}%)')
        ax.text(0.5, 0.95, summaryText, transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.annotate(rf'\${finalVal:,.0f}\n({retPct:+.2f}%)',
                    xy=(timestamps[-1], finalVal),
                    ha='right', va='bottom' if retPct >= 0 else 'top',
                    fontsize=9, fontweight='bold',
                    color='green' if retPct >= 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              alpha=0.8, edgecolor='gray'))

        ax.set_title(f'{result.symbol} — Combined Backtest Performance',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(r'Portfolio Value (\$)', fontsize=10)
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='best', fontsize=8)
        ax.ticklabel_format(style='plain', axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    def _plotIntervalBacktest(self, ax, btResult: BacktestResult,
                              interval: str, stockFund: float):
        """Cumulative P/L for a single interval."""
        if not btResult.trades:
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{interval}', fontsize=11, fontweight='bold')
            return

        sortedTrades = sorted(btResult.trades, key=lambda t: t['timestamp'])

        timestamps = [sortedTrades[0]['timestamp']]
        values = [0.0]
        cumProfit = 0.0

        for trade in sortedTrades:
            profit = trade.get('dollarPnL', trade.get('fundAllocation', 0) * (trade['returnPct'] / 100))
            cumProfit += profit
            timestamps.append(trade['timestamp'])
            values.append(cumProfit)

        color = '#1f77b4' if cumProfit >= 0 else '#d62728'
        ax.plot(timestamps, values, lw=2, color=color, alpha=0.85,
                marker='o', markersize=3)
        ax.fill_between(timestamps, 0, values, alpha=0.12, color=color)
        ax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.4)

        winRate = btResult.getSuccessRate()
        ax.set_title(f'{interval} — {btResult.totalTrades} trades, '
                     rf'{winRate:.0f}% WR, P/L=\${cumProfit:,.0f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel(r'Cumulative P/L (\$)', fontsize=9)
        ax.grid(True, alpha=0.3, ls='--')
        ax.ticklabel_format(style='plain', axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # =====================================================================
    # Step 9c — Portfolio-Level Performance Graph (all stocks combined)
    # =====================================================================

    def _generatePortfolioGraph(self, verbose: bool):
        """
        Generate a portfolio-level performance graph that combines
        backtest results across ALL stocks into one cumulative view,
        plus a per-slot allocation breakdown.

        Layout:
          Top-left:     Overall portfolio cumulative value over time
          Top-right:    Per-slot allocation breakdown (pie chart)
          Bottom:       Per-stock cumulative P/L stacked on one axis
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("STEP 9c: GENERATING PORTFOLIO PERFORMANCE GRAPH")
            print(f"{'=' * 70}")

        # Collect ALL trades across ALL stocks
        allTrades = []
        perStockTrades: Dict[str, list] = {}

        for symbol, result in self.results.items():
            if not result.backtestResults:
                continue
            stockFund = self.totalFund * result.allocation
            perStockTrades[symbol] = []
            for interval, btResult in result.backtestResults.items():
                if interval.startswith('WalkForward_'):
                    continue
                for trade in btResult.trades:
                    tradeEntry = {
                        'timestamp': trade.get('exitTimestamp', trade['timestamp']),
                        'entryTimestamp': trade['timestamp'],
                        'returnPct': trade['returnPct'],
                        'fundAllocation': trade.get('fundAllocation', 0),
                        'dollarPnL': trade.get('dollarPnL', None),
                        'interval': interval,
                        'symbol': symbol,
                        'stockFund': stockFund,
                    }
                    allTrades.append(tradeEntry)
                    perStockTrades[symbol].append(tradeEntry)

        if not allTrades:
            if verbose:
                print("  No trades to plot")
            return

        # Determine layout: 3-panel if we have allocation history, 2-panel otherwise
        hasAllocHistory = (self.allocationResult is not None
                           and bool(self.allocationResult.allocationHistory)
                           and len(self.allocationResult.allocationHistory) > 1)

        if hasAllocHistory:
            fig = plt.figure(figsize=(18, 20))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 1],
                                  hspace=0.35, wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])       # top-left: portfolio value
            axAlloc = fig.add_subplot(gs[0, 1])   # top-right: allocation evolution
            axPie = fig.add_subplot(gs[1, 0])     # mid-left: final alloc pie
            axEmpty = fig.add_subplot(gs[1, 1])   # mid-right: placeholder
            axEmpty.axis('off')
            ax2 = fig.add_subplot(gs[2, :])       # bottom: per-stock P/L
        elif (self.allocationResult is not None
              and bool(self.allocationResult.slotAllocations)):
            fig = plt.figure(figsize=(18, 16))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
            axPie = fig.add_subplot(gs[0, 1])
            axAlloc = None
            ax2 = fig.add_subplot(gs[1, :])
        else:
            fig, axes = plt.subplots(2, 1, figsize=(16, 14))
            ax = axes[0]
            ax2 = axes[1]
            axPie = None
            axAlloc = None

        hasSlotAllocs = (self.allocationResult is not None
                         and bool(self.allocationResult.slotAllocations))

        # ==== Top-left: Overall Portfolio Cumulative Value ====
        allTrades.sort(key=lambda t: t['timestamp'])

        timestamps = [allTrades[0]['timestamp']]
        values = [self.totalFund]
        cumProfit = 0.0

        for trade in allTrades:
            profit = trade.get('dollarPnL', trade['fundAllocation'] * (trade['returnPct'] / 100))
            cumProfit += profit
            timestamps.append(trade['timestamp'])
            values.append(self.totalFund + cumProfit)

        ax.plot(timestamps, values, lw=3, color='#1565C0', alpha=0.9,
                label='Portfolio Value')
        ax.fill_between(timestamps, self.totalFund, values, alpha=0.12,
                        color='#42A5F5')
        ax.axhline(self.totalFund, color='black', ls='--', lw=1, alpha=0.4,
                   label=f'Initial: ${self.totalFund:,.0f}')

        # ---- Benchmark overlays: Weighted Buy & Hold + S&P 500 ----
        if self.benchmarks:
            weightedSeries = None
            for sym, alloc in self.stocks.items():
                if sym in self.benchmarks:
                    bm = self.benchmarks[sym]
                    stockValue = self.totalFund * alloc * bm['series']
                    if weightedSeries is None:
                        weightedSeries = stockValue.copy()
                    else:
                        combined = pd.concat([weightedSeries, stockValue], axis=1)
                        combined.columns = ['a', 'b']
                        combined = combined.dropna()
                        weightedSeries = combined['a'] + combined['b']

            if weightedSeries is not None:
                wRetPct = ((weightedSeries.iloc[-1] - self.totalFund) / self.totalFund) * 100
                ax.plot(weightedSeries.index, weightedSeries.values,
                        lw=2, color='#FF9800', ls='--', alpha=0.8,
                        label=f'Buy & Hold: {wRetPct:+.1f}%')

            if '^GSPC' in self.benchmarks:
                spBm = self.benchmarks['^GSPC']
                spValues = self.totalFund * spBm['series']
                spRetPct = spBm['returnPct']
                ax.plot(spBm['dates'], spValues.values,
                        lw=2, color='#9C27B0', ls='-.', alpha=0.7,
                        label=f'S&P 500: {spRetPct:+.1f}%')

        # Annotations
        finalValue = self.totalFund + cumProfit
        totalReturn = (cumProfit / self.totalFund * 100) if self.totalFund > 0 else 0
        totalTrades = len(allTrades)
        totalWins = sum(1 for t in allTrades if t['returnPct'] > 0)
        winRate = (totalWins / totalTrades * 100) if totalTrades > 0 else 0

        summaryText = (
            f'Trades: {totalTrades}  |  WR: {winRate:.1f}%  |  '
            rf'P/L: \${cumProfit:,.0f} ({totalReturn:+.1f}%)'
        )
        ax.text(0.5, 0.95, summaryText, transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.annotate(rf'\${finalValue:,.0f}',
                    xy=(timestamps[-1], finalValue),
                    ha='right', va='bottom' if totalReturn >= 0 else 'top',
                    fontsize=10, fontweight='bold',
                    color='green' if totalReturn >= 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              alpha=0.85, edgecolor='gray'))

        ax.set_title('Portfolio Value Over Time',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(r'Portfolio Value (\$)', fontsize=10)
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='best', fontsize=8)
        ax.ticklabel_format(style='plain', axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right',
                 fontsize=7)

        # ==== Top-right: Allocation Pie Chart ====
        if hasSlotAllocs:
            slotAllocs = self.allocationResult.slotAllocations
            # Filter out ghost slots (0% alloc) for the pie chart
            activeSlots = {k: v for k, v in slotAllocs.items() if v > 0.001}
            ghostSlots = {k: v for k, v in slotAllocs.items() if v <= 0.001}

            if activeSlots:
                colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                labels = []
                sizes = []
                pieColours = []
                slotIdx = 0
                for slotStr in sorted(activeSlots.keys()):
                    alloc = activeSlots[slotStr]
                    labels.append(f'{slotStr}\n{alloc*100:.1f}%')
                    sizes.append(alloc)
                    pieColours.append(colours[slotIdx % len(colours)])
                    slotIdx += 1

                wedges, texts, autotexts = axPie.pie(
                    sizes, labels=labels, colors=pieColours,
                    autopct='${:,.0f}'.format,
                    pctdistance=0.65,
                    startangle=90,
                    textprops={'fontsize': 8},
                )
                # Fix autopct to show dollar amounts instead of %
                for i, autotext in enumerate(autotexts):
                    dollarAmt = self.totalFund * sizes[i]
                    autotext.set_text(f'${dollarAmt:,.0f}')
                    autotext.set_fontsize(7)

                axPie.set_title(f'Fund Allocation by Slot ({self.allocationResult.method})',
                               fontsize=12, fontweight='bold')

                # Add ghost slots as text below
                if ghostSlots:
                    ghostText = 'Ghost: ' + ', '.join(sorted(ghostSlots.keys()))
                    axPie.text(0.5, -0.1, ghostText, transform=axPie.transAxes,
                               ha='center', va='top', fontsize=8, color='gray',
                               style='italic')

        # ==== Allocation Evolution Over Time (stacked area) ====
        if hasAllocHistory and axAlloc is not None:
            allocHistory = self.allocationResult.allocationHistory
            allocColours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Extract timestamps and per-slot allocation series
            allocTs = [entry['timestamp'] for entry in allocHistory]
            allSlots = sorted(set().union(
                *(entry['allocations'].keys() for entry in allocHistory)
            ))

            slotSeries = {}
            for slotStr in allSlots:
                slotSeries[slotStr] = [
                    entry['allocations'].get(slotStr, 0.0) * 100
                    for entry in allocHistory
                ]

            # Stacked area chart
            prevBottom = np.zeros(len(allocTs))
            for sIdx, slotStr in enumerate(allSlots):
                vals = np.array(slotSeries[slotStr])
                colour = allocColours[sIdx % len(allocColours)]
                axAlloc.fill_between(allocTs, prevBottom, prevBottom + vals,
                                     alpha=0.6, color=colour, label=slotStr,
                                     linewidth=0.5)
                axAlloc.plot(allocTs, prevBottom + vals, lw=0.8, color=colour,
                             alpha=0.8)
                prevBottom = prevBottom + vals

            axAlloc.set_title('Allocation Evolution Over Time',
                              fontsize=12, fontweight='bold')
            axAlloc.set_xlabel('Date', fontsize=10)
            axAlloc.set_ylabel('Allocation (%)', fontsize=10)
            axAlloc.set_ylim(0, 105)
            axAlloc.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                           fontsize=7, framealpha=0.8)
            axAlloc.grid(True, alpha=0.3, ls='--')
            plt.setp(axAlloc.xaxis.get_majorticklabels(), rotation=30,
                     ha='right', fontsize=7)

            # Add checkpoint markers
            for entry in allocHistory[1:]:  # skip initial
                axAlloc.axvline(entry['timestamp'], color='gray', ls=':',
                                lw=0.5, alpha=0.5)

        # ==== Bottom: Per-Stock Cumulative P/L (overlaid) ====
        colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for sIdx, (symbol, trades) in enumerate(perStockTrades.items()):
            if not trades:
                continue
            trades.sort(key=lambda t: t['timestamp'])
            stockFund = trades[0]['stockFund']

            ts = [trades[0]['timestamp']]
            cum = [0.0]
            cumP = 0.0
            for trade in trades:
                profit = trade.get('dollarPnL', trade['fundAllocation'] * (trade['returnPct'] / 100))
                cumP += profit
                ts.append(trade['timestamp'])
                cum.append(cumP)

            retPct = (cumP / stockFund * 100) if stockFund > 0 else 0
            colour = colours[sIdx % len(colours)]
            ax2.plot(ts, cum, lw=2.2, color=colour, alpha=0.85,
                     label=f'{symbol}: ${cumP:,.0f} ({retPct:+.1f}%)',
                     marker='o', markersize=2)

        ax2.axhline(0, color='black', ls='--', lw=0.8, alpha=0.4)
        ax2.set_title('Per-Stock Cumulative P/L', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Cumulative P/L ($)', fontsize=11)
        ax2.grid(True, alpha=0.3, ls='--')
        ax2.legend(loc='best', fontsize=9)
        ax2.ticklabel_format(style='plain', axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right',
                 fontsize=8)

        # Suptitle
        allocText = ' | '.join(
            f'{sym}: {alloc*100:.0f}%' for sym, alloc in self.stocks.items()
        )
        fig.suptitle(
            f'Portfolio Overview  |  Total Fund: ${self.totalFund:,.0f}  |  '
            f'Allocations: {allocText}',
            fontsize=14, fontweight='bold', y=0.99,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        imgData = self._fig_to_b64(fig, dpi=80, quality=80)
        plt.close(fig)

        self._pendingCharts.append({
            'type': 'portfolio',
            'symbol': '',
            'label': 'Portfolio Performance Overview',
            'data': imgData,
        })
        if verbose:
            print("  Staged portfolio performance chart")

    # =====================================================================
    # Chart upload helper
    # =====================================================================

    def _saveBacktestTrades(self, runId: str, verbose: bool = True):
        """
        Persist individual backtest trades to the 'backtest_trades' Firestore
        collection.  One document per (symbol, interval) pair, containing the
        full ordered trade list so the frontend can render equity curves and
        per-trade drill-downs.

        Document ID: {runId}_{symbol}_{interval}  (deterministic → idempotent)

        All trades MUST have been processed by the dynamic allocator before
        this method is called.  The allocator stamps each trade with:
          - dollarPnL   : scaled P&L (not notional)
          - fundAllocation : real capital deployed for that trade
          - slotAllocation : the slot's allocation fraction at trade time

        The BacktestResult.initialBalance is set by the allocator write-back
        to ``totalFund × initialSlotAlloc``.  This method uses it directly
        so that the equity curve and return % are exact.
        """
        import datetime as _btdt
        col = self.persistence.db.collection('backtest_trades')
        saved = 0
        for sym, res in self.results.items():
            if not res.backtestResults:
                continue

            for interval, bt in res.backtestResults.items():
                if not bt.trades or interval.startswith('WalkForward_'):
                    continue

                # Use the initialBalance set by the dynamic allocator
                # write-back (= totalFund × initialSlotAlloc for this slot).
                # Fall back to per-interval fraction if allocator didn't set it.
                initBal = bt.initialBalance
                if initBal <= 0:
                    nInt = max(sum(1 for _b in res.backtestResults.values() if _b.trades), 1)
                    initBal = max(self.totalFund * res.allocation / nInt, 1.0)
                    if verbose:
                        print(f"  [Trades] WARNING: {sym}/{interval} missing "
                              f"initialBalance — using fallback ${initBal:,.0f}")

                serialized = []
                for t in bt.trades:
                    ts = t['timestamp']
                    ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                    exitTs = t.get('exitTimestamp', ts)
                    exitTs_str = exitTs.isoformat() if hasattr(exitTs, 'isoformat') else str(exitTs)
                    pnl = float(t.get('dollarPnL', 0.0))
                    alloc = float(t.get('fundAllocation', 0.0))
                    bal = float(t.get('balanceAfter', 0.0))

                    tradeDoc = {
                        'ts':    ts_str,
                        'exitTs': exitTs_str,
                        'ok':    bool(t['successful']),
                        'ret':   round(float(t['returnPct']), 4),
                        'entry': round(float(t['entryPrice']), 4),
                        'exit':  round(float(t['exitPrice']), 4),
                        'pnl':   round(pnl, 2),
                        'bal':   round(bal, 2),
                        'alloc': round(alloc, 2),
                        'conf':  round(float(t.get('confidence', t.get('size', 0.0))), 3),
                    }
                    if t.get('signalBreakdown'):
                        tradeDoc['signals'] = t['signalBreakdown']
                    if t.get('exitReason'):
                        tradeDoc['exitReason'] = t['exitReason']
                    serialized.append(tradeDoc)
                finalBal = bt.finalBalance if bt.finalBalance else initBal
                totalPnL = finalBal - initBal
                compoundReturn = (totalPnL / initBal * 100) if initBal > 0 else 0.0
                doc_id = f"{runId}_{sym}_{interval}"
                try:
                    col.document(doc_id).set({
                        'runId':         runId,
                        'symbol':        sym,
                        'interval':      interval,
                        'portfolioType': res.portfolioType,
                        'trades':        serialized,
                        'stats': {
                            'totalTrades':    bt.totalTrades,
                            'winRate':        round(bt.getSuccessRate(), 2),
                            'totalReturnPct': round(compoundReturn, 4),
                            'initialBalance': round(initBal, 2),
                            'finalBalance':   round(finalBal, 2),
                        },
                        'isWalkForward': interval.startswith('WalkForward_'),
                        'createdAt': datetime.now().isoformat(),
                    })
                    saved += 1
                except Exception as e:
                    if verbose:
                        print(f"  [Trades] Warning: failed to save trades for {sym}/{interval}: {e}")
        if verbose:
            print(f"  [Trades] Saved {saved} backtest-trade documents for run {runId}")

    def _uploadChartsToFirestore(self, runId: str, verbose: bool = True):
        """
        Upload all staged chart images (base64 data-URI) to the 'run_charts'
        Firestore collection.  Each chart becomes one document so the
        frontend can fetch them individually without loading a huge run doc.
        """
        if not self._pendingCharts:
            return
        import datetime as _dt
        col = self.persistence.db.collection('run_charts')
        uploaded = 0
        for chart in self._pendingCharts:
            try:
                col.add({
                    'runId':     runId,
                    'chartType': chart['type'],
                    'symbol':    chart.get('symbol', ''),
                    'label':     chart['label'],
                    'imageData': chart['data'],
                    'createdAt': datetime.now().isoformat(),
                })
                uploaded += 1
            except Exception as e:
                print(f"  [Charts] Warning: failed to upload '{chart['label']}': {e}")
        if verbose:
            print(f"  [Charts] Uploaded {uploaded}/{len(self._pendingCharts)} chart(s) for run {runId}")
        self._pendingCharts = []

    # =====================================================================
    # Summary
    # =====================================================================

    def _printBanner(self, verbose: bool):
        if not verbose:
            return
        print("\n" + "#" * 80)
        print("#" + " " * 78 + "#")
        print("#   PORTFOLIO TESTER — GA → MC → BACKTEST PIPELINE" + " " * 29 + "#")
        print("#" + " " * 78 + "#")
        print("#" * 80)
        print(f"\nTotal Fund:   ${self.totalFund:,.2f}")
        print(f"Stocks:       {self.stocks}")
        print(f"Intervals:    {[i[0] for i in self.intervals]}")
        print(f"Pattern Lens: {self.patternLengths}")
        print(f"GA runs/cfg:  {self.numRunsPerConfig}")
        print(f"GA max/TF:    {self.maxPatternsPerInterval} → MC keeps {self.mcTopN}")
        print(f"MC sims:      {self.mcNumSimulations} (rank: {self.mcRankSimulations})")
        print(f"MC method:    {self.mcMethod}")
        if self.mcRefineIterations > 0:
            print(f"MC refine:    {self.mcRefineIterations} iterations, "
                  f"{self.mcRefineRandomMutants} random + "
                  f"{self.mcRefineMLMutants} ML mutants/pattern, "
                  f"strength={self.mcRefineMutationStrength}")
        else:
            print(f"MC refine:    disabled (0 iterations)")
        print(f"Backtest:     last {self.backtestPeriodDays:.0f} days")
        if self.mlEnabled:
            print(f"ML models:    enabled (stock fwd={self.mlForwardPeriods}, "
                  f"portfolio fwd={self.mlPortfolioForwardPeriods}, "
                  f"corr window={self.mlCorrWindow})")
            print(f"Dyn Alloc:    eval every {self.fundAllocator.evalWindowDays}d, "
                  f"min eval {self.fundAllocator.minEvalPeriodDays}d, "
                  f"ML={'on' if self.fundAllocator.useML else 'off'}")
        else:
            print(f"ML models:    disabled")
        print()

    def _printSummary(self):
        """Print a consolidated summary of the entire pipeline."""
        print("\n" + "=" * 80)
        print("PORTFOLIO TESTER — FINAL SUMMARY")
        print("=" * 80)
        print(f"Total Fund: ${self.totalFund:,.2f}")
        print(f"Stocks: {list(self.stocks.keys())}")
        print("=" * 80)

        totalProfit = 0.0
        totalTrades = 0
        totalWins = 0

        for symbol, result in self.results.items():
            stockFund = self.totalFund * result.allocation
            print(f"\n--- {symbol} (Alloc {result.allocation*100:.0f}%, "
                  f"${stockFund:,.0f}) ---")

            # GA summary
            if result.rawPatternBank:
                print(f"  GA Raw Patterns:    {len(result.rawPatternBank.patterns)}")
            if result.refinedPatternBank:
                print(f"  MC Refined Patterns: {len(result.refinedPatternBank.patterns)}")

            # MC summary per interval
            for interval, mc in sorted(result.mcResults.items()):
                print(f"  MC {interval}: "
                      f"E[R]={mc.expectedReturnPct:+.2f}%, "
                      f"Sharpe={mc.avgSharpeRatio:.3f}, "
                      f"P(Profit)={mc.probabilityOfProfit:.1f}%")

            # Backtest summary per interval
            stockProfit = 0.0
            stockTrades = 0
            stockWins = 0
            stockInitFund = 0.0
            for interval, bt in sorted(result.backtestResults.items()):
                intervalProfit = sum(float(t.get('dollarPnL', 0.0)) for t in bt.trades)
                stockProfit += intervalProfit
                if bt.trades and bt.initialBalance > 0:
                    stockInitFund += bt.initialBalance
                stockTrades += bt.totalTrades
                stockWins += bt.successfulTrades
                wr = bt.getSuccessRate()
                print(f"  BT {interval}: "
                      f"{bt.totalTrades} trades, WR={wr:.1f}%, "
                      f"P/L=${intervalProfit:,.2f}")

            if stockInitFund <= 0:
                stockInitFund = stockFund
            stockReturn = (stockProfit / stockInitFund * 100) if stockInitFund > 0 else 0
            print(f"  STOCK TOTAL: {stockTrades} trades, "
                  f"P/L=${stockProfit:,.2f} ({stockReturn:+.2f}%)")

            # ML model summary
            if result.stockMLModel and result.stockMLModel.isTrained:
                mlMetrics = result.stockMLModel.evaluate()
                if mlMetrics:
                    print(f"  ML Model: direction_acc={mlMetrics.directionAccuracy*100:.1f}%, "
                          f"return_RMSE={mlMetrics.returnRMSE:.4f}")
                if result.stockPredictions:
                    latest = result.stockPredictions[-1]
                    print(f"  ML Latest Signal: {latest.signal.value} "
                          f"(conf={latest.confidence:.0%}, "
                          f"E[R]={latest.expectedReturn:+.2f}%, "
                          f"regime={latest.regime.value})")

            totalProfit += stockProfit
            totalTrades += stockTrades
            totalWins += stockWins

        # ML Portfolio summary
        if self.portfolioMLModel and self.portfolioMLModel.isTrained:
            print("\n" + "-" * 80)
            print("PORTFOLIO ML MODEL")
            print("-" * 80)
            pMetrics = self.portfolioMLModel.evaluate()
            if pMetrics:
                print(f"  Regime Accuracy : {pMetrics.regimeAccuracy*100:.1f}%")
                print(f"  Hedge Accuracy  : {pMetrics.hedgeAccuracy*100:.1f}%")
                print(f"  Return RMSE     : {pMetrics.returnRMSE:.5f}")
            if self.portfolioSignal:
                sig = self.portfolioSignal
                print(f"  Risk Regime     : {sig.riskRegime.value}")
                print(f"  Hedge Action    : {sig.hedgeAction.value}")
                print(f"  Expected Return : {sig.expectedPortfolioReturn:+.4f}")
                print(f"  Portfolio Vol   : {sig.portfolioVolatility:.4f} (annualised)")
                print(f"  Avg Correlation : {sig.avgCrossCorrelation:.3f}")
                if sig.allocationAdjustments:
                    print(f"  Allocation Adjustments:")
                    for adj in sig.allocationAdjustments:
                        print(f"    {adj.symbol}: {adj.currentAllocation*100:.1f}% "
                              f"→ {adj.suggestedAllocation*100:.1f}%")

        # Dynamic Allocation Slot Breakdown
        if self.allocationResult and self.allocationResult.slotAllocations:
            print("\n" + "-" * 80)
            print("DYNAMIC ALLOCATION — Per-Slot Breakdown (Final State)")
            print("-" * 80)
            print(f"  Method: {self.allocationResult.method}")
            nCheckpoints = max(0, len(self.allocationResult.allocationHistory) - 1)
            if nCheckpoints > 0:
                print(f"  Evaluation checkpoints: {nCheckpoints}")
            print(f"  {'Slot':<14} {'Alloc':>7} {'Fund':>10} {'Score':>7} {'Status':<10}")
            print(f"  {'─' * 52}")
            for slotStr in sorted(self.allocationResult.slotAllocations):
                alloc = self.allocationResult.slotAllocations[slotStr]
                fund = self.totalFund * alloc
                perf = self.allocationResult.slotPerformances.get(slotStr)
                score = perf.ruleScore if perf else 0.0
                if slotStr in self.allocationResult.ghostSlots:
                    status = "GHOST"
                elif slotStr in self.allocationResult.restoredSlots:
                    status = "RESTORED"
                else:
                    status = "ACTIVE"
                print(f"  {slotStr:<14} {alloc*100:>6.1f}% ${fund:>9,.0f} {score:>+6.3f} {status:<10}")

        # Overall
        print("\n" + "=" * 80)
        totalReturn = (totalProfit / self.totalFund * 100) if self.totalFund > 0 else 0
        winRate = (totalWins / totalTrades * 100) if totalTrades > 0 else 0
        finalValue = self.totalFund + totalProfit
        print(f"PORTFOLIO TOTAL")
        print(f"  Initial Value: ${self.totalFund:,.2f}")
        print(f"  Final Value:   ${finalValue:,.2f}")
        print(f"  Total Trades:  {totalTrades}")
        print(f"  Win Rate:      {winRate:.2f}%")
        print(f"  Total P/L:     ${totalProfit:,.2f}")
        print(f"  Total Return:  {totalReturn:+.2f}%")

        # ---- Benchmark Comparison ----
        if self.benchmarks:
            print("\n" + "-" * 80)
            print("BENCHMARK COMPARISON  (same backtest period)")
            print("-" * 80)

            # Per-stock buy-and-hold
            for symbol, result in self.results.items():
                if symbol in self.benchmarks:
                    bm = self.benchmarks[symbol]
                    stockFund = self.totalFund * result.allocation
                    bhProfit = stockFund * bm['returnPct'] / 100
                    # Actual strategy profit for this stock
                    stratProfit = sum(
                        t.get('dollarPnL', t.get('fundAllocation', 0) * (t['returnPct'] / 100))
                        for intv, bt in result.backtestResults.items()
                        for t in bt.trades
                    )
                    stratReturn = (stratProfit / stockFund * 100) if stockFund > 0 else 0
                    diff = stratReturn - bm['returnPct']
                    emoji = '+' if diff >= 0 else ''
                    print(f"  {symbol:6s}  Strategy: {stratReturn:+8.2f}%  |  "
                          f"Buy & Hold: {bm['returnPct']:+8.2f}%  |  "
                          f"Alpha: {emoji}{diff:.2f}%")

            # Weighted buy-and-hold
            weightedBH = sum(
                self.stocks.get(sym, 0) * self.benchmarks[sym]['returnPct']
                for sym in self.stocks if sym in self.benchmarks
            )
            weightedBHProfit = self.totalFund * weightedBH / 100
            diff = totalReturn - weightedBH
            print(f"\n  {'PORTFOLIO':6s}  Strategy: {totalReturn:+8.2f}%  |  "
                  f"Buy & Hold: {weightedBH:+8.2f}%  |  "
                  f"Alpha: {'+' if diff >= 0 else ''}{diff:.2f}%")

            if '^GSPC' in self.benchmarks:
                spReturn = self.benchmarks['^GSPC']['returnPct']
                diff = totalReturn - spReturn
                print(f"  {'':6s}  Strategy: {totalReturn:+8.2f}%  |  "
                      f"S&P 500:    {spReturn:+8.2f}%  |  "
                      f"Alpha: {'+' if diff >= 0 else ''}{diff:.2f}%")

        print("=" * 80)

        # Graph file output suppressed
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#
#   ALL CONFIGURABLE SETTINGS
#
#   Every tuneable parameter for the entire pipeline lives here.
#   Change whatever you need, then run:   python PortfolioTester.py
#
# ═══════════════════════════════════════════════════════════════════════════════


# ── Portfolio ────────────────────────────────────────────────────────────────
#    How much money you start with and how it's split across stocks.

TOTAL_FUND  = 100_000                   # Total starting capital in dollars
STOCKS      = {                         # Ticker → fraction of TOTAL_FUND (must sum to 1.0)
    'AAPL':   0.40,                     #   40 % allocated to Apple
    'GOOGL':  0.30,                     #   30 % allocated to Alphabet
    'MSFT':   0.30,                     #   30 % allocated to Microsoft
}


# ── Genetic Algorithm (Step 1 — Pattern Discovery) ──────────────────────────
#    Controls how the GA searches for candlestick patterns in historical data.

# CRITICAL: GA must use TRAINING data only (before backtest period) to avoid look-ahead bias.
# The periods below are IGNORED - we now use absolute date ranges calculated in _discoverPatterns().
# Kept here for reference only.
# MULTI-TIMEFRAME: ML models train separately per interval to match pattern timeframes.
INTERVALS = [                           # (interval, history period) pairs - period values are overridden with calculated dates
    ('1d',  '2y'),                      #   daily candles  — swing trading & macro trends
    ('1h',  '1mo'),                     #   hourly candles — intraday momentum
    # ('30m', '1mo'),                   #   30-min candles — phased out (too short for quant fund to profit from)
]
PATTERN_LENGTHS           = [3, 4, 5, 6, 7, 8, 10]  # Full sweep of pattern lengths
POPULATION_SIZE           = 2000        # Large population for thorough search-space exploration
NUM_GENERATIONS           = 150         # Max generations (early stop triggers much sooner in practice)
NUM_RUNS_PER_CONFIG       = 3           # Three independent runs per config for robustness
MAX_PATTERNS_PER_INTERVAL = 25          # Keep top 25 per interval

# GA quality parameters — not configurable before, now fully exposed
GA_EARLY_STOP_GENERATIONS = 25          # Stop evolving if no improvement for this many consecutive gens
GA_ELITISM_COUNT          = 25          # Top N chromosomes preserved unchanged each generation
GA_MUTATION_RATE          = 0.20        # Gene mutation probability (higher = more diversity)
GA_CROSSOVER_RATE         = 0.75        # Chromosome crossover probability
GA_MIN_IMPROVEMENT        = 0.001       # Minimum fitness delta to count as real progress (patient)


# ── Monte Carlo Simulation (Step 2 — Ranking  &  Step 3 — Full Sim) ────────
#    MC validates patterns by simulating thousands of synthetic price paths.

MC_RANK_SIMULATIONS   = 2000            # Thorough ranking simulations per pattern
MC_TOP_N              = 12              # Keep top 12 patterns per interval after ranking
MC_NUM_SIMULATIONS    = 2000            # Full MC simulations per pattern
MC_SIMULATION_PERIODS = 252             # Full trading year of simulation per path
MC_METHOD             = 'bootstrap'     # Price-path generator: 'bootstrap' (resample history) or 'gbm' (geometric Brownian motion)
MC_CALIBRATION_PERIOD = '1y'            # How much real history to calibrate the MC model from
MC_TARGETS            = [5, 10, 20, 50, 100]  # Return-% thresholds for P(≥X%) probability report
MC_RUIN_THRESHOLD     = 0.50            # Fraction of capital lost to count as "ruin" (0.50 = 50 %)


# ── MC Refinement (Step 2b — Iterative Mutation Loop) ───────────────────────
#    After MC ranking, patterns are improved via random + ML-guided mutations.
#    Set MC_REFINE_ITERATIONS = 0 to skip this step entirely.

MC_REFINE_ITERATIONS         = 3        # Three refinement passes for pattern improvement
MC_REFINE_SIMS_PER_CANDIDATE = 500      # Reliable scoring per candidate mutation
MC_REFINE_RANDOM_MUTANTS     = 6        # Random mutations per pattern per iteration
MC_REFINE_ML_MUTANTS         = 5        # ML-guided mutations per pattern per iteration
MC_REFINE_MUTATION_STRENGTH  = 0.25     # How aggressively genes are perturbed (0.0 = none, 1.0 = heavy)


# ── Backtesting (Step 7 — ML-Based Trading with Decider) ──────────────────
#    ML models execute trades based on pattern features + technical indicators.
#    The TradingDecider reconciles stock-level and portfolio-level signals.
#    Shadow stocks (0% alloc) still get simulated trades for recovery detection.

# ── Train/Test Split (CRITICAL for avoiding look-ahead bias) ──
# Timeline (with today = Feb 2026):
#   1. GA discovers patterns on TRAIN data:  Aug 2018 → Aug 2021  (3y discovery)
#   2. ML trains on same TRAIN data:         Aug 2018 → Aug 2021  (3y training)
#   3. Backtest runs on out-of-sample TEST:  Aug 2021 → Feb 2026  (5y out-of-sample)
# Patterns NEVER see the test period → true out-of-sample validation.
# 5y backtest = multiple regimes (bull, bear, COVID, inflation). 3y train = enough bars for ML.
GA_DISCOVERY_PERIOD_DAYS = 365 * 3      # 3 years training data (~750 daily bars)
ML_TRAIN_PERIOD_DAYS = 365 * 3          # Match GA discovery window for consistent features
BACKTEST_PERIOD_DAYS = 365 * 5          # 5 years out-of-sample backtest (robust across regimes)
FORGIVENESS          = 0.07             # Slightly relaxed tolerance (0.07 = ±7%) — 5% was too strict for daily bars
USE_STOP_LOSS        = True             # Per-trade stop-loss (keep enabled)
USE_WALK_FORWARD     = True             # Walk-forward validation (Step 7b in incremental)
USE_EARNINGS_BLACKOUT  = True           # Earnings proximity sizing/conf boost (helps avoid weak signals near events)
USE_REGIME_DETECTION   = True           # Regime-based size/conf overrides (reduces size in HIGH_RISK/CRISIS)
USE_CORRELATION_ADJUSTMENT = True       # Correlation-aware position scaling (avoids over-concentration)


# ── ML Models (Step 4 & 5 — Stock Model  &  Portfolio Model) ───────────────
#    ML models learn from patterns (as features) + technical indicators.
#    When ML_ENABLED = True, ML models make ALL trading decisions.
#    When ML_ENABLED = False, the system only runs steps 1-3 (no trading).

ML_ENABLED                   = True     # Master switch: enables ML training & ML-based trading
ML_FORWARD_PERIODS           = 8        # Prediction horizon (5–20 bars optimal for swing; 8 = responsive)
ML_PORTFOLIO_FORWARD_PERIODS = 8        # Portfolio prediction horizon (match stock model)
ML_CORR_WINDOW               = 60       # Rolling window for cross-stock correlation


# ── Trading Decider (Step 6 — Signal Reconciliation) ───────────────────────
#    Blends THREE independent signals: patterns (GA) + ML (OHLCV) + portfolio.
#    Patterns and ML are now completely separate - they help each other by voting!

DECIDER_PATTERN_WEIGHT   = 0.70         # Pattern weight (hedge funds: balance patterns with diversification)
DECIDER_PORTFOLIO_WEIGHT = 0.45         # Portfolio influence (higher = more diversification, less correlated bets)
DECIDER_MIN_CONFIDENCE   = 0.25         # Min confidence to trade (0.25 allows more signals; allocator kills bad slots)


# ── Dynamic Fund Allocation (Step 8 — Continuous Rebalancing) ───────────────
#    Allocates capital at the (stock, interval) level CONTINUOUSLY throughout
#    the backtest, not just once at the end.  At each evaluation checkpoint,
#    slot performance is re-evaluated and allocations are adjusted.
#    Uses rule-based scoring + optional ML model to learn allocation patterns.
#    Ghost slots (0% alloc) still run simulated trades for recovery detection.

IFA_MIN_ALLOCATION      = 0.0           # Hard floor per slot (0.0 = allow ghost mode)
IFA_MAX_SLOT_ALLOC      = 0.40          # Hard ceiling per slot (concentration limit)
IFA_MAX_STOCK_ALLOC     = 0.60          # Hard ceiling per stock across all intervals
IFA_GHOST_THRESHOLD     = -0.35         # Per-slot score below this → ghost (slightly more forgiving before cut)
IFA_RESTORE_THRESHOLD   = 2.5           # Rolling return (%) to restore ghost (faster recovery)
IFA_RESTORE_ALLOCATION  = 0.04          # Initial allocation (4%) when restoring (slightly larger probe)
IFA_SMOOTHING_FACTOR    = 0.28          # Dampener (lower = faster adaptation to regime changes)
IFA_USE_ML              = True          # Enable ML allocation model alongside rule-based
IFA_ML_BLEND_WEIGHT     = 0.4           # ML blend weight (0=pure rule, 1=pure ML)
IFA_EVAL_WINDOW_DAYS    = 4             # Days between reallocation checkpoints (more responsive)
IFA_MIN_EVAL_PERIOD_DAYS = 8            # Minimum days before first reallocation (adapt sooner)


# ═════════════════════════════════════════════════════════════════════════════
#
#  QUICK-TEST OVERRIDES  (activated with:  python PortfolioTester.py --quick)
#
#  Every step still runs, but parameters are dialled way down so the full
#  pipeline completes in ~3-8 minutes instead of hours.  Useful for
#  verifying that nothing crashes before committing to a production run.
#
# ═════════════════════════════════════════════════════════════════════════════

_QUICK_OVERRIDES = {
    # Lighter GA/MC for fast validation; trading params match production
    'totalFund':                10_000,
    'stocks':                   {'AAPL': 0.50, 'MSFT': 0.50},
    'intervals':                [('1d', '1y')],
    'patternLengths':           [3, 5],
    'populationSize':           200,
    'numGenerations':           20,
    'numRunsPerConfig':         1,
    'maxPatternsPerInterval':   5,
    'gaEarlyStopGenerations':   8,
    'gaElitismCount':           5,
    'mcRankSimulations':        100,
    'mcTopN':                   3,
    'mcNumSimulations':         100,
    'mcSimulationPeriods':      60,
    'mcTargets':                [5, 10, 20],
    'mcRefineIterations':       1,
    'mcRefineSimsPerCandidate': 50,
    'mcRefineRandomMutants':    2,
    'mcRefineMLMutants':        2,
    'mlTrainPeriodDays':        365,
    'backtestPeriodDays':       180,
    'mlForwardPeriods':         ML_FORWARD_PERIODS,
    'mlPortfolioForwardPeriods': ML_PORTFOLIO_FORWARD_PERIODS,
    'mlCorrWindow':             ML_CORR_WINDOW,
    'deciderPatternWeight':     DECIDER_PATTERN_WEIGHT,
    'deciderPortfolioWeight':   DECIDER_PORTFOLIO_WEIGHT,
    'deciderMinConfidence':     DECIDER_MIN_CONFIDENCE,
    'ifaEvalWindowDays':        IFA_EVAL_WINDOW_DAYS,
    'ifaMinEvalPeriodDays':     IFA_MIN_EVAL_PERIOD_DAYS,
    'ifaSmoothingFactor':       IFA_SMOOTHING_FACTOR,
}


# =============================================================================
# Run the pipeline
# =============================================================================

if __name__ == "__main__":

    import sys as _sys

    # ── Detect --quick flag ────────────────────────────────────────────────────
    _quickMode = '--quick' in _sys.argv

    # ── Load API keys from .env (project root, one level above this file) ──────
    import os as _os
    import pathlib as _pathlib
    _env_path = _pathlib.Path(__file__).parent.parent / '.env'
    try:
        from dotenv import load_dotenv as _load_dotenv
        _loaded = _load_dotenv(dotenv_path=_env_path, override=True)  # override=True so .env always wins
        if _loaded:
            print(f"[Config] Loaded secrets from {_env_path}")
        else:
            print(f"[Config] No .env found at {_env_path} — using environment variables only")
    except ImportError:
        print("[Config] python-dotenv not installed; reading keys from environment variables only.")
        print("         Install with: pip install python-dotenv")

    _OPENAI_KEY        = _os.environ.get('OPENAI_API_KEY')    or None
    _NEWSAPI_KEY       = _os.environ.get('NEWSAPI_KEY')       or None
    _FINNHUB_KEY       = _os.environ.get('FINNHUB_KEY')       or None
    _ALPHAVANTAGE_KEY  = _os.environ.get('ALPHAVANTAGE_KEY')  or None

    # Report which sources are active so the user knows what's running
    _active = [k for k, v in [
        ('OpenAI/Layer3', _OPENAI_KEY),
        ('Finnhub',       _FINNHUB_KEY),
        ('NewsAPI',       _NEWSAPI_KEY),
        ('AlphaVantage',  _ALPHAVANTAGE_KEY),
    ] if v]
    print(f"[Sentiment] Active API sources: {', '.join(_active) if _active else 'none — synthetic fallback only'}")

    # ── Build the full config dict, then overlay quick overrides if needed ─────
    _cfg = dict(
        # Portfolio
        totalFund=TOTAL_FUND,
        stocks=STOCKS,
        # GA
        intervals=INTERVALS,
        patternLengths=PATTERN_LENGTHS,
        populationSize=POPULATION_SIZE,
        numGenerations=NUM_GENERATIONS,
        numRunsPerConfig=NUM_RUNS_PER_CONFIG,
        maxPatternsPerInterval=MAX_PATTERNS_PER_INTERVAL,
        gaEarlyStopGenerations=GA_EARLY_STOP_GENERATIONS,
        gaElitismCount=GA_ELITISM_COUNT,
        gaMutationRate=GA_MUTATION_RATE,
        gaCrossoverRate=GA_CROSSOVER_RATE,
        gaMinImprovement=GA_MIN_IMPROVEMENT,
        # MC
        mcNumSimulations=MC_NUM_SIMULATIONS,
        mcSimulationPeriods=MC_SIMULATION_PERIODS,
        mcMethod=MC_METHOD,
        mcCalibrationPeriod=MC_CALIBRATION_PERIOD,
        mcRankSimulations=MC_RANK_SIMULATIONS,
        mcTopN=MC_TOP_N,
        mcTargets=MC_TARGETS,
        mcRuinThreshold=MC_RUIN_THRESHOLD,
        # MC Refinement
        mcRefineIterations=MC_REFINE_ITERATIONS,
        mcRefineSimsPerCandidate=MC_REFINE_SIMS_PER_CANDIDATE,
        mcRefineRandomMutants=MC_REFINE_RANDOM_MUTANTS,
        mcRefineMLMutants=MC_REFINE_ML_MUTANTS,
        mcRefineMutationStrength=MC_REFINE_MUTATION_STRENGTH,
        # Backtest
        mlTrainPeriodDays=ML_TRAIN_PERIOD_DAYS,
        backtestPeriodDays=BACKTEST_PERIOD_DAYS,
        forgiveness=FORGIVENESS,
        useStopLoss=_os.environ.get('USE_STOP_LOSS', str(USE_STOP_LOSS)).lower() in ('true', '1', 'yes'),
        useWalkForward=_os.environ.get('USE_WALK_FORWARD', str(USE_WALK_FORWARD)).lower() in ('true', '1', 'yes'),
        useEarningsBlackout=_os.environ.get('USE_EARNINGS_BLACKOUT', str(USE_EARNINGS_BLACKOUT)).lower() in ('true', '1', 'yes'),
        useRegimeDetection=_os.environ.get('USE_REGIME_DETECTION', str(USE_REGIME_DETECTION)).lower() in ('true', '1', 'yes'),
        useCorrelationAdjustment=_os.environ.get('USE_CORRELATION_ADJUSTMENT', str(USE_CORRELATION_ADJUSTMENT)).lower() in ('true', '1', 'yes'),
        # ML Models
        mlForwardPeriods=ML_FORWARD_PERIODS,
        mlPortfolioForwardPeriods=ML_PORTFOLIO_FORWARD_PERIODS,
        mlCorrWindow=ML_CORR_WINDOW,
        mlEnabled=ML_ENABLED,
        # Trading Decider
        deciderPatternWeight=DECIDER_PATTERN_WEIGHT,
        deciderPortfolioWeight=DECIDER_PORTFOLIO_WEIGHT,
        deciderMinConfidence=DECIDER_MIN_CONFIDENCE,
        # Dynamic Fund Allocation
        ifaMinAllocation=IFA_MIN_ALLOCATION,
        ifaMaxSlotAllocation=IFA_MAX_SLOT_ALLOC,
        ifaMaxStockAllocation=IFA_MAX_STOCK_ALLOC,
        ifaShadowThreshold=IFA_GHOST_THRESHOLD,
        ifaRestoreThreshold=IFA_RESTORE_THRESHOLD,
        ifaRestoreAllocation=IFA_RESTORE_ALLOCATION,
        ifaSmoothingFactor=IFA_SMOOTHING_FACTOR,
        ifaUseML=IFA_USE_ML,
        ifaMLBlendWeight=IFA_ML_BLEND_WEIGHT,
        ifaEvalWindowDays=IFA_EVAL_WINDOW_DAYS,
        ifaMinEvalPeriodDays=IFA_MIN_EVAL_PERIOD_DAYS,
        # Sentiment API keys (loaded from .env above)
        sentimentOpenAIKey=_OPENAI_KEY,
        sentimentNewsAPIKey=_NEWSAPI_KEY,
        sentimentFinnhubKey=_FINNHUB_KEY,
        sentimentAlphaVantageKey=_ALPHAVANTAGE_KEY,
    )

    if _quickMode:
        _cfg.update(_QUICK_OVERRIDES)
        print("\n[MODE] *** QUICK TEST *** — lightweight params for fast validation")
        print(f"       Stocks: {_cfg['stocks']}, Pop: {_cfg['populationSize']}, "
              f"Gens: {_cfg['numGenerations']}, MC: {_cfg['mcNumSimulations']} sims")
        print(f"       Expected runtime: ~3-8 minutes\n")
    else:
        print(f"\n[MODE] PRODUCTION — full pipeline for next trading day")
        print(f"       Stocks: {list(STOCKS.keys())}, Pop: {POPULATION_SIZE}, "
              f"Gens: {NUM_GENERATIONS}, MC: {MC_NUM_SIMULATIONS} sims")
        print(f"       Run: python backend/PortfolioTester.py  (no --quick)\n")

    tester = PortfolioTester(**_cfg)
    tester.run(verbose=True)
