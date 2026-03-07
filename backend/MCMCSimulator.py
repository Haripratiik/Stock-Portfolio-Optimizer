"""
Monte Carlo & Markov Chain Monte Carlo Simulator for Pattern-Based
Trading Strategies
=====================================================================

Generates thousands of synthetic but model-consistent future price paths,
then tests the discovered patterns (from the Genetic Algorithm) on each
path to produce a full distribution of possible outcomes.

Risk metrics computed:
- Expected return, median return, standard deviation
- VaR (Value at Risk) at 95% and 99% confidence
- CVaR (Conditional VaR / Expected Shortfall)
- Maximum drawdown distribution
- Probability of profit, probability of ruin
- Probability of hitting user-defined return targets
- Sharpe ratio distribution
- Win rate distribution

Supports FOUR path generation methods:
- GBM (Geometric Brownian Motion): Standard parametric model
- Bootstrap: Non-parametric, samples from actual historical returns
             (preserves fat tails)
- MCMC (Metropolis-Hastings): Bayesian posterior sampling of model
       parameters — each path uses parameters drawn from the posterior
       distribution, capturing *parameter uncertainty* on top of the
       usual stochastic uncertainty.
- Regime-switching: Two-state (bull/bear) Hidden Markov Model calibrated
       from historical data.  Each simulation step first samples the
       current regime from the Markov chain, then draws a return from
       that regime's distribution.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os
from joblib import Parallel, delayed

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GeneticAlgorithm import (
    PatternBank, PatternChromosome, PatternGene,
    StockDataFetcher, CandleType
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimulationPath:
    """Results from a single Monte Carlo simulation path."""
    pathId: int
    trades: List[dict] = field(default_factory=list)
    totalReturnPct: float = 0.0
    maxDrawdownPct: float = 0.0
    numTrades: int = 0
    successfulTrades: int = 0
    unsuccessfulTrades: int = 0
    finalBalance: float = 0.0
    peakBalance: float = 0.0
    sharpeRatio: float = 0.0
    balanceTimeSeries: List[float] = field(default_factory=list)  # Balance at each period

    def getWinRate(self) -> float:
        """Win rate as a percentage."""
        if self.numTrades == 0:
            return 0.0
        return (self.successfulTrades / self.numTrades) * 100


@dataclass
class MonteCarloResults:
    """Aggregated results from all Monte Carlo simulation paths."""
    symbol: str
    interval: str
    numSimulations: int
    simulationPeriods: int
    initialFund: float
    paths: List[SimulationPath] = field(default_factory=list)

    # --- Return distribution ---
    expectedReturnPct: float = 0.0
    medianReturnPct: float = 0.0
    stdReturnPct: float = 0.0
    bestReturnPct: float = 0.0
    worstReturnPct: float = 0.0

    # --- VaR ---
    var95: float = 0.0
    var99: float = 0.0

    # --- CVaR (Expected Shortfall) ---
    cvar95: float = 0.0
    cvar99: float = 0.0

    # --- Drawdown ---
    avgMaxDrawdown: float = 0.0
    worstMaxDrawdown: float = 0.0
    medianMaxDrawdown: float = 0.0

    # --- Probabilities ---
    probabilityOfProfit: float = 0.0
    probabilityOfRuin: float = 0.0
    ruinThreshold: float = 0.5  # 50% loss = ruin by default
    targetProbabilities: Dict[float, float] = field(default_factory=dict)

    # --- Trade stats ---
    avgTradesPerPath: float = 0.0
    avgWinRate: float = 0.0

    # --- Sharpe ---
    avgSharpeRatio: float = 0.0

    def computeMetrics(self, targets: List[float] = None, ruinThreshold: float = 0.5):
        """Compute all aggregate metrics from the completed simulation paths."""
        if not self.paths:
            return

        self.ruinThreshold = ruinThreshold
        returns = np.array([p.totalReturnPct for p in self.paths])
        drawdowns = np.array([p.maxDrawdownPct for p in self.paths])
        winRates = np.array([p.getWinRate() for p in self.paths])
        tradeCounts = np.array([p.numTrades for p in self.paths])
        sharpes = np.array([p.sharpeRatio for p in self.paths])

        # Basic stats
        self.expectedReturnPct = float(np.mean(returns))
        self.medianReturnPct = float(np.median(returns))
        self.stdReturnPct = float(np.std(returns))
        self.bestReturnPct = float(np.max(returns))
        self.worstReturnPct = float(np.min(returns))

        # VaR — 5th and 1st percentile of return distribution
        self.var95 = float(np.percentile(returns, 5))
        self.var99 = float(np.percentile(returns, 1))

        # CVaR — mean of the tail beyond VaR
        tail95 = returns[returns <= self.var95]
        self.cvar95 = float(np.mean(tail95)) if len(tail95) > 0 else self.var95
        tail99 = returns[returns <= self.var99]
        self.cvar99 = float(np.mean(tail99)) if len(tail99) > 0 else self.var99

        # Drawdowns
        self.avgMaxDrawdown = float(np.mean(drawdowns))
        self.worstMaxDrawdown = float(np.max(drawdowns))
        self.medianMaxDrawdown = float(np.median(drawdowns))

        # Probabilities
        self.probabilityOfProfit = float(np.mean(returns > 0) * 100)
        self.probabilityOfRuin = float(np.mean(returns <= -(ruinThreshold * 100)) * 100)

        # Target probabilities
        if targets is None:
            targets = [5, 10, 20, 50, 100]
        self.targetProbabilities = {}
        for target in targets:
            self.targetProbabilities[target] = float(np.mean(returns >= target) * 100)

        # Trade statistics
        self.avgTradesPerPath = float(np.mean(tradeCounts))
        self.avgWinRate = float(np.mean(winRates))

        # Sharpe
        validSharpes = sharpes[np.isfinite(sharpes)]
        self.avgSharpeRatio = float(np.mean(validSharpes)) if len(validSharpes) > 0 else 0.0

    def summary(self) -> str:
        """Generate a formatted summary string."""
        lines = [
            f"\n{'=' * 70}",
            f"MONTE CARLO RESULTS — {self.symbol} ({self.interval})",
            f"{'=' * 70}",
            f"Simulations: {self.numSimulations:,} paths x {self.simulationPeriods} periods",
            f"Initial Fund: ${self.initialFund:,.2f}",
            f"",
            f"--- Return Distribution ---",
            f"  Expected Return: {self.expectedReturnPct:+.2f}%",
            f"  Median Return:   {self.medianReturnPct:+.2f}%",
            f"  Std Deviation:   {self.stdReturnPct:.2f}%",
            f"  Best Case:       {self.bestReturnPct:+.2f}%",
            f"  Worst Case:      {self.worstReturnPct:+.2f}%",
            f"",
            f"--- Risk Metrics ---",
            f"  VaR  (95%):  {self.var95:+.2f}%  (${self.initialFund * self.var95 / 100:>+,.2f})",
            f"  VaR  (99%):  {self.var99:+.2f}%  (${self.initialFund * self.var99 / 100:>+,.2f})",
            f"  CVaR (95%):  {self.cvar95:+.2f}%  (${self.initialFund * self.cvar95 / 100:>+,.2f})",
            f"  CVaR (99%):  {self.cvar99:+.2f}%  (${self.initialFund * self.cvar99 / 100:>+,.2f})",
            f"",
            f"--- Drawdown ---",
            f"  Avg Max Drawdown:    {self.avgMaxDrawdown:.2f}%",
            f"  Median Max Drawdown: {self.medianMaxDrawdown:.2f}%",
            f"  Worst Max Drawdown:  {self.worstMaxDrawdown:.2f}%",
            f"",
            f"--- Probabilities ---",
            f"  Probability of Profit:  {self.probabilityOfProfit:.1f}%",
            f"  Probability of Ruin ({self.ruinThreshold*100:.0f}% loss): {self.probabilityOfRuin:.1f}%",
        ]

        if self.targetProbabilities:
            lines.append(f"  Target Probabilities:")
            for target, prob in sorted(self.targetProbabilities.items()):
                lines.append(f"    >= {target:+.0f}% return: {prob:.1f}%")

        lines.extend([
            f"",
            f"--- Sharpe ---",
            f"  Avg Sharpe Ratio: {self.avgSharpeRatio:.3f}",
            f"",
            f"--- Trade Statistics ---",
            f"  Avg Trades per Path: {self.avgTradesPerPath:.1f}",
            f"  Avg Win Rate:        {self.avgWinRate:.1f}%",
        ])

        if self.avgTradesPerPath < 1.0:
            lines.append(f"  ⚠ WARNING: Very few trades per path — patterns may be "
                         f"too specific for synthetic data (fan chart will be flat).")

        lines.append(f"{'=' * 70}")

        return "\n".join(lines)


# =============================================================================
# Monte Carlo Simulator
# =============================================================================

class MCMCSimulator:
    """
    Monte Carlo & MCMC simulator for pattern-based trading strategies.

    Generates thousands of synthetic future price paths calibrated from
    historical data, then tests the discovered patterns on each path
    to produce a distribution of possible outcomes.

    Supports four path-generation methods:
      - 'gbm'        Geometric Brownian Motion (parametric)
      - 'bootstrap'   Resample actual historical returns (non-parametric)
      - 'mcmc'        Bayesian posterior sampling via Metropolis-Hastings
      - 'regime'      Two-state (bull/bear) regime-switching Hidden Markov Model
    """

    def __init__(self,
                 initialFund: float = 10000.0,
                 forgiveness: float = 0.05,
                 numSimulations: int = 1000,
                 simulationPeriods: int = 252,
                 method: str = 'bootstrap',
                 randomSeed: Optional[int] = None,
                 mcForgivenessMultiplier: float = 3.0):
        """
        Args:
            initialFund:              Starting capital for each simulated path
            forgiveness:              Pattern matching tolerance (±%)
            numSimulations:           Number of simulation paths to generate
            simulationPeriods:        Number of future periods to simulate per path
            method:                   Price path method ('gbm', 'bootstrap',
                                      'mcmc', or 'regime')
            randomSeed:               Seed for reproducibility (None = not set)
            mcForgivenessMultiplier:  How much wider pattern matching should be
                                      in MC mode vs backtest (default 3× →
                                      0.05 forgiveness becomes 0.15 on synthetic data).
        """
        self.initialFund = initialFund
        self.forgiveness = forgiveness
        self.mcForgiveness = forgiveness * mcForgivenessMultiplier
        self.numSimulations = numSimulations
        self.simulationPeriods = simulationPeriods
        self.method = method.lower()
        self.fetcher = StockDataFetcher()

        # MCMC posterior samples cache (filled lazily)
        self._posteriorSamples: Optional[np.ndarray] = None
        # Regime-switching parameters cache
        self._regimeParams: Optional[dict] = None

        if randomSeed is not None:
            np.random.seed(randomSeed)

    # -------------------------------------------------------------------------
    # Model calibration
    # -------------------------------------------------------------------------

    def _calibrateFromHistory(self, df: pd.DataFrame) -> dict:
        """
        Extract statistical properties from historical OHLCV data for
        path generation:
          - Return distribution (mu, sigma)
          - Open-gap distribution
          - High / Low extension distributions
          - Volume ratio distribution
          - Body-ratio distribution
          - Bullish candle fraction

        Returns:
            dict of calibration parameters
        """
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        # Close-to-close simple returns
        returns = np.diff(closes) / closes[:-1]

        # Log returns (for GBM drift/vol)
        logReturns = np.log(closes[1:] / closes[:-1])

        # Open gap: (open_t - close_{t-1}) / close_{t-1}
        openGaps = (opens[1:] - closes[:-1]) / closes[:-1]

        # High extension above max(open, close)
        maxOC = np.maximum(opens, closes)
        highExtensions = (highs - maxOC) / closes
        highExtensions = highExtensions[highExtensions > 0]

        # Low extension below min(open, close)
        minOC = np.minimum(opens, closes)
        lowExtensions = (minOC - lows) / closes
        lowExtensions = lowExtensions[lowExtensions > 0]

        # Body ratio: |close - open| / (high - low)
        totalRange = highs - lows
        bodySize = np.abs(closes - opens)
        validRange = totalRange > 0
        bodyRatios = bodySize[validRange] / totalRange[validRange]

        # Volume
        avgVolume = float(np.mean(volumes))
        volumeRatios = volumes / avgVolume if avgVolume > 0 else np.ones_like(volumes)

        # Bullish fraction
        bullishPct = float(np.mean(closes > opens))

        return {
            'mu': float(np.mean(logReturns)),
            'sigma': float(np.std(logReturns)),
            'returns': returns,
            'logReturns': logReturns,
            'openGaps': openGaps,
            'highExtensions': highExtensions if len(highExtensions) > 0 else np.array([0.005]),
            'lowExtensions': lowExtensions if len(lowExtensions) > 0 else np.array([0.005]),
            'bodyRatios': bodyRatios if len(bodyRatios) > 0 else np.array([0.5]),
            'volumeRatios': volumeRatios,
            'avgVolume': avgVolume,
            'lastClose': float(closes[-1]),
            'lastVolume': float(volumes[-1]),
            'bullishPct': bullishPct,
        }

    # -------------------------------------------------------------------------
    # Price-path generators
    # -------------------------------------------------------------------------

    def _generatePricePath(self, calibration: dict, numPeriods: int) -> pd.DataFrame:
        """Dispatch to the configured path generation method."""
        if self.method == 'gbm':
            return self._generateGBMPath(calibration, numPeriods)
        elif self.method == 'mcmc':
            return self._generateMCMCPath(calibration, numPeriods)
        elif self.method == 'regime':
            return self._generateRegimeSwitchingPath(calibration, numPeriods)
        return self._generateBootstrapPath(calibration, numPeriods)

    def _generatePricePathArrays(self, calibration: dict, numPeriods: int):
        """
        Fast path generation returning raw numpy arrays instead of DataFrame.
        Avoids DataFrame overhead for inner MC simulation loops.
        Returns (closes, opens, highs, lows, volumes, dates).
        """
        cal = calibration
        lastClose = cal['lastClose']

        if self.method == 'mcmc':
            # Sample (mu, sigma) from posterior
            mu, sigma = self._drawPosteriorSample(cal)
            randomLogReturns = np.random.normal(mu, sigma, numPeriods)
            closes = lastClose * np.exp(np.cumsum(randomLogReturns))
        elif self.method == 'regime':
            closes = self._generateRegimeClosesArray(cal, numPeriods, lastClose)
        elif self.method == 'gbm':
            randomLogReturns = np.random.normal(cal['mu'], cal['sigma'], numPeriods)
            closes = lastClose * np.exp(np.cumsum(randomLogReturns))
        else:
            sampledReturns = np.random.choice(cal['returns'], size=numPeriods, replace=True)
            closes = lastClose * np.cumprod(1 + sampledReturns)

        prevCloses = np.empty(numPeriods)
        prevCloses[0] = lastClose
        prevCloses[1:] = closes[:-1]

        openGaps = np.random.choice(cal['openGaps'], size=numPeriods)
        opens = prevCloses * (1 + openGaps)

        highExts = np.random.choice(cal['highExtensions'], size=numPeriods)
        lowExts = np.random.choice(cal['lowExtensions'], size=numPeriods)

        maxOC = np.maximum(opens, closes)
        minOC = np.minimum(opens, closes)

        highs = maxOC + np.abs(closes) * highExts
        lows = minOC - np.abs(closes) * lowExts

        highs = np.maximum(highs, maxOC)
        lows = np.minimum(lows, minOC)
        lows = np.maximum(lows, 0.01)

        volumes = cal['avgVolume'] * np.random.choice(cal['volumeRatios'], size=numPeriods)

        baseDate = datetime.now()
        dates = pd.date_range(baseDate, periods=numPeriods, freq='D')

        return closes, opens, highs, lows, volumes, dates

    def _generateGBMPath(self, cal: dict, numPeriods: int) -> pd.DataFrame:
        """
        Geometric Brownian Motion:  dS = mu*S*dt + sigma*S*dW

        Generates a full OHLCV path by sampling close-to-close log-returns
        from N(mu, sigma) and then layering on realistic open-gaps,
        high/low wicks, and volume drawn from the historical distribution.

        Fully vectorized — no Python loop over periods.
        """
        mu = cal['mu']
        sigma = cal['sigma']
        lastClose = cal['lastClose']

        randomLogReturns = np.random.normal(mu, sigma, numPeriods)

        # Vectorized close prices via cumulative product
        closes = lastClose * np.exp(np.cumsum(randomLogReturns))

        # Previous close array (lastClose, then closes[:-1])
        prevCloses = np.empty(numPeriods)
        prevCloses[0] = lastClose
        prevCloses[1:] = closes[:-1]

        # Open (small gap from previous close)
        openGaps = np.random.choice(cal['openGaps'], size=numPeriods)
        opens = prevCloses * (1 + openGaps)

        # High / Low extensions
        highExts = np.random.choice(cal['highExtensions'], size=numPeriods)
        lowExts = np.random.choice(cal['lowExtensions'], size=numPeriods)

        maxOC = np.maximum(opens, closes)
        minOC = np.minimum(opens, closes)

        highs = maxOC + np.abs(closes) * highExts
        lows = minOC - np.abs(closes) * lowExts

        highs = np.maximum(highs, maxOC)
        lows = np.minimum(lows, minOC)
        lows = np.maximum(lows, 0.01)  # never negative

        # Volume
        volumes = cal['avgVolume'] * np.random.choice(cal['volumeRatios'], size=numPeriods)

        baseDate = datetime.now()
        dates = pd.date_range(baseDate, periods=numPeriods, freq='D')

        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes,
        }, index=dates)

    def _generateBootstrapPath(self, cal: dict, numPeriods: int) -> pd.DataFrame:
        """
        Bootstrap path: resample actual historical simple returns with
        replacement.  Preserves fat tails, skew, and kurtosis of real data
        while generating new, plausible paths.

        Fully vectorized — no Python loop over periods.
        """
        historicalReturns = cal['returns']
        lastClose = cal['lastClose']

        sampledReturns = np.random.choice(historicalReturns, size=numPeriods, replace=True)

        # Vectorized close prices via cumulative product
        closes = lastClose * np.cumprod(1 + sampledReturns)

        # Previous close array (lastClose, then closes[:-1])
        prevCloses = np.empty(numPeriods)
        prevCloses[0] = lastClose
        prevCloses[1:] = closes[:-1]

        # Open (small gap from previous close)
        openGaps = np.random.choice(cal['openGaps'], size=numPeriods)
        opens = prevCloses * (1 + openGaps)

        # High / Low extensions
        highExts = np.random.choice(cal['highExtensions'], size=numPeriods)
        lowExts = np.random.choice(cal['lowExtensions'], size=numPeriods)

        maxOC = np.maximum(opens, closes)
        minOC = np.minimum(opens, closes)

        highs = maxOC + np.abs(closes) * highExts
        lows = minOC - np.abs(closes) * lowExts

        highs = np.maximum(highs, maxOC)
        lows = np.minimum(lows, minOC)
        lows = np.maximum(lows, 0.01)

        volumes = cal['avgVolume'] * np.random.choice(cal['volumeRatios'], size=numPeriods)

        baseDate = datetime.now()
        dates = pd.date_range(baseDate, periods=numPeriods, freq='D')

        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes,
        }, index=dates)

    # -------------------------------------------------------------------------
    # MCMC — Metropolis-Hastings posterior sampling
    # -------------------------------------------------------------------------

    def _runMetropolisHastings(
        self, cal: dict, nSamples: int = 5000, burnIn: int = 1000,
    ) -> np.ndarray:
        """
        Metropolis-Hastings MCMC to sample from the posterior distribution
        of (mu, sigma) given observed log-returns.

        Likelihood:  logReturns ~ N(mu, sigma^2)
        Priors:      mu     ~ N(sample_mu, 0.02)         (weakly informative)
                     sigma  ~ HalfCauchy(scale=0.05)      (allows fat tails)

        Returns:
            np.ndarray of shape (nSamples - burnIn, 2) — each row is (mu, sigma)
        """
        logR = cal['logReturns']
        n = len(logR)
        sumLogR = float(np.sum(logR))
        sumLogR2 = float(np.sum(logR ** 2))

        # Initial values = MLE estimates
        mu_curr = float(cal['mu'])
        sigma_curr = max(float(cal['sigma']), 1e-6)

        # Proposal step sizes (tuned for acceptance ~0.25-0.40)
        mu_step = sigma_curr * 0.1
        sigma_step = sigma_curr * 0.05

        def logPosterior(mu: float, sigma: float) -> float:
            """Log of unnormalised posterior."""
            if sigma <= 0:
                return -np.inf
            # Log-likelihood (Normal)
            ll = -n * np.log(sigma) - 0.5 * (sumLogR2 - 2 * mu * sumLogR + n * mu**2) / (sigma**2)
            # Log-prior: mu ~ N(cal['mu'], 0.02)
            lp_mu = -0.5 * ((mu - cal['mu']) / 0.02)**2
            # Log-prior: sigma ~ HalfCauchy(0.05)
            lp_sigma = -np.log(1 + (sigma / 0.05)**2)
            return ll + lp_mu + lp_sigma

        samples = np.empty((nSamples, 2))
        logP_curr = logPosterior(mu_curr, sigma_curr)
        accepted = 0

        for i in range(nSamples):
            # Propose
            mu_prop = mu_curr + np.random.normal(0, mu_step)
            sigma_prop = sigma_curr + np.random.normal(0, sigma_step)

            logP_prop = logPosterior(mu_prop, sigma_prop)

            # Accept / reject
            if np.log(np.random.uniform()) < (logP_prop - logP_curr):
                mu_curr = mu_prop
                sigma_curr = sigma_prop
                logP_curr = logP_prop
                accepted += 1

            samples[i] = [mu_curr, sigma_curr]

        return samples[burnIn:]

    def _ensurePosteriorSamples(self, cal: dict):
        """Cache posterior samples so MH only runs once per calibration."""
        if self._posteriorSamples is None:
            self._posteriorSamples = self._runMetropolisHastings(cal)

    def _drawPosteriorSample(self, cal: dict) -> Tuple[float, float]:
        """Draw a single (mu, sigma) from the cached posterior."""
        self._ensurePosteriorSamples(cal)
        idx = np.random.randint(len(self._posteriorSamples))
        mu, sigma = self._posteriorSamples[idx]
        return float(mu), max(float(sigma), 1e-8)

    def _generateMCMCPath(self, cal: dict, numPeriods: int) -> pd.DataFrame:
        """
        MCMC path: draw (mu, sigma) from the Bayesian posterior, then
        generate a GBM path using those parameters.

        Each simulation path uses *different* parameter values, capturing
        both stochastic uncertainty AND parameter uncertainty.
        """
        mu, sigma = self._drawPosteriorSample(cal)
        lastClose = cal['lastClose']

        randomLogReturns = np.random.normal(mu, sigma, numPeriods)
        closes = lastClose * np.exp(np.cumsum(randomLogReturns))

        prevCloses = np.empty(numPeriods)
        prevCloses[0] = lastClose
        prevCloses[1:] = closes[:-1]

        openGaps = np.random.choice(cal['openGaps'], size=numPeriods)
        opens = prevCloses * (1 + openGaps)

        highExts = np.random.choice(cal['highExtensions'], size=numPeriods)
        lowExts = np.random.choice(cal['lowExtensions'], size=numPeriods)

        maxOC = np.maximum(opens, closes)
        minOC = np.minimum(opens, closes)
        highs = maxOC + np.abs(closes) * highExts
        lows = minOC - np.abs(closes) * lowExts
        highs = np.maximum(highs, maxOC)
        lows = np.minimum(lows, minOC)
        lows = np.maximum(lows, 0.01)

        volumes = cal['avgVolume'] * np.random.choice(cal['volumeRatios'], size=numPeriods)

        baseDate = datetime.now()
        dates = pd.date_range(baseDate, periods=numPeriods, freq='D')

        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes,
        }, index=dates)

    # -------------------------------------------------------------------------
    # Regime-switching path generation (2-state HMM)
    # -------------------------------------------------------------------------

    def _calibrateRegimes(self, cal: dict) -> dict:
        """
        Fit a simple 2-state (bull/bear) Hidden Markov Model to the
        observed log-returns using EM (Baum-Welch style, simplified).

        Returns dict with:
          mu_bull, sigma_bull   – bull regime parameters
          mu_bear, sigma_bear   – bear regime parameters
          P_bb, P_bu            – transition probs from bull
          P_ub, P_uu            – transition probs from bear (u = ursine)
          pi_bull               – steady-state probability of bull regime
        """
        if self._regimeParams is not None:
            return self._regimeParams

        logR = cal['logReturns']
        n = len(logR)

        # --- Simple k-means-style regime assignment ---
        median_r = float(np.median(logR))

        bull_mask = logR >= median_r
        bear_mask = ~bull_mask

        mu_bull = float(np.mean(logR[bull_mask])) if bull_mask.any() else cal['mu']
        sigma_bull = float(np.std(logR[bull_mask])) if bull_mask.any() else cal['sigma']
        mu_bear = float(np.mean(logR[bear_mask])) if bear_mask.any() else cal['mu']
        sigma_bear = float(np.std(logR[bear_mask])) if bear_mask.any() else cal['sigma']

        sigma_bull = max(sigma_bull, 1e-6)
        sigma_bear = max(sigma_bear, 1e-6)

        # --- EM refinement (5 iterations) ---
        for _ in range(5):
            # E-step: compute responsibilities
            from scipy.stats import norm as _norm
            ll_bull = _norm.logpdf(logR, mu_bull, sigma_bull)
            ll_bear = _norm.logpdf(logR, mu_bear, sigma_bear)
            max_ll = np.maximum(ll_bull, ll_bear)
            resp_bull = np.exp(ll_bull - max_ll) / (
                np.exp(ll_bull - max_ll) + np.exp(ll_bear - max_ll) + 1e-300
            )

            # M-step
            w_bull = resp_bull.sum()
            w_bear = n - w_bull
            if w_bull > 1:
                mu_bull = float(np.sum(resp_bull * logR) / w_bull)
                sigma_bull = max(
                    float(np.sqrt(np.sum(resp_bull * (logR - mu_bull)**2) / w_bull)),
                    1e-6,
                )
            if w_bear > 1:
                mu_bear = float(np.sum((1 - resp_bull) * logR) / w_bear)
                sigma_bear = max(
                    float(np.sqrt(np.sum((1 - resp_bull) * (logR - mu_bear)**2) / w_bear)),
                    1e-6,
                )

        # Ensure bull has higher mu
        if mu_bear > mu_bull:
            mu_bull, mu_bear = mu_bear, mu_bull
            sigma_bull, sigma_bear = sigma_bear, sigma_bull
            resp_bull = 1 - resp_bull

        # --- Transition probabilities from regime assignments ---
        regime = (resp_bull > 0.5).astype(int)  # 1=bull, 0=bear
        transitions = {'bb': 0, 'bu': 0, 'ub': 0, 'uu': 0}
        for i in range(len(regime) - 1):
            fr = regime[i]
            to = regime[i + 1]
            if fr == 1 and to == 1:
                transitions['bb'] += 1
            elif fr == 1 and to == 0:
                transitions['bu'] += 1
            elif fr == 0 and to == 1:
                transitions['ub'] += 1
            else:
                transitions['uu'] += 1

        total_from_bull = transitions['bb'] + transitions['bu'] + 1e-9
        total_from_bear = transitions['ub'] + transitions['uu'] + 1e-9

        P_bb = transitions['bb'] / total_from_bull
        P_bu = transitions['bu'] / total_from_bull
        P_ub = transitions['ub'] / total_from_bear
        P_uu = transitions['uu'] / total_from_bear

        pi_bull = P_ub / (P_bu + P_ub + 1e-9)

        self._regimeParams = {
            'mu_bull': mu_bull, 'sigma_bull': sigma_bull,
            'mu_bear': mu_bear, 'sigma_bear': sigma_bear,
            'P_bb': P_bb, 'P_bu': P_bu,
            'P_ub': P_ub, 'P_uu': P_uu,
            'pi_bull': pi_bull,
        }
        return self._regimeParams

    def _generateRegimeClosesArray(
        self, cal: dict, numPeriods: int, lastClose: float,
    ) -> np.ndarray:
        """
        Generate close prices using 2-state regime-switching model.
        Returns np.ndarray of close prices.
        """
        rp = self._calibrateRegimes(cal)

        # Start in a random regime weighted by steady-state probability
        inBull = np.random.random() < rp['pi_bull']

        logReturns = np.empty(numPeriods)
        for t in range(numPeriods):
            if inBull:
                logReturns[t] = np.random.normal(rp['mu_bull'], rp['sigma_bull'])
                inBull = np.random.random() < rp['P_bb']
            else:
                logReturns[t] = np.random.normal(rp['mu_bear'], rp['sigma_bear'])
                inBull = np.random.random() < rp['P_ub']

        return lastClose * np.exp(np.cumsum(logReturns))

    def _generateRegimeSwitchingPath(
        self, cal: dict, numPeriods: int,
    ) -> pd.DataFrame:
        """
        Regime-switching path: two-state (bull/bear) Hidden Markov Model.
        """
        lastClose = cal['lastClose']
        closes = self._generateRegimeClosesArray(cal, numPeriods, lastClose)

        prevCloses = np.empty(numPeriods)
        prevCloses[0] = lastClose
        prevCloses[1:] = closes[:-1]

        openGaps = np.random.choice(cal['openGaps'], size=numPeriods)
        opens = prevCloses * (1 + openGaps)

        highExts = np.random.choice(cal['highExtensions'], size=numPeriods)
        lowExts = np.random.choice(cal['lowExtensions'], size=numPeriods)

        maxOC = np.maximum(opens, closes)
        minOC = np.minimum(opens, closes)
        highs = maxOC + np.abs(closes) * highExts
        lows = minOC - np.abs(closes) * lowExts
        highs = np.maximum(highs, maxOC)
        lows = np.minimum(lows, minOC)
        lows = np.maximum(lows, 0.01)

        volumes = cal['avgVolume'] * np.random.choice(cal['volumeRatios'], size=numPeriods)

        baseDate = datetime.now()
        dates = pd.date_range(baseDate, periods=numPeriods, freq='D')

        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes,
        }, index=dates)

    # -------------------------------------------------------------------------
    # Pattern matching (mirrors Backtester._checkGeneMatchPython)
    # -------------------------------------------------------------------------

    def _checkGeneMatch(self, gene: PatternGene, row: pd.Series,
                        baselinePrice: float, avgVolume: float) -> bool:
        """
        Check whether a single pattern gene matches an OHLCV data row.
        Uses the same tolerance logic as Backtester.
        """
        # Normalized price change from baseline
        actualNormalizedPct = (row['close'] - baselinePrice) / baselinePrice

        # Price check
        priceDiff = abs(actualNormalizedPct - gene.expectedPriceChangePct)
        if priceDiff > self.forgiveness:
            return False

        # Volume check (+-50% tolerance)
        volumeRatio = row['volume'] / avgVolume if avgVolume > 0 else 1.0
        volumeDiff = abs(volumeRatio - gene.expectedVolumePct)
        if volumeDiff > 0.5:
            return False

        # Candle type check
        bodyDirection = row['close'] - row['open']
        candleTypeInt = ['BULLISH', 'BEARISH', 'DOJI'].index(gene.candleType.value)

        if candleTypeInt == 0 and bodyDirection < -0.0001:   # expected BULLISH
            return False
        elif candleTypeInt == 1 and bodyDirection > 0.0001:  # expected BEARISH
            return False
        elif candleTypeInt == 2:                              # expected DOJI
            totalRange = row['high'] - row['low']
            if totalRange > 0 and abs(bodyDirection) > totalRange * 0.15:
                return False

        # Body percentage check
        if gene.minBodyPct > 0.1:
            totalRange = row['high'] - row['low']
            if totalRange > 0:
                bodyPct = abs(bodyDirection) / totalRange
                if bodyPct < gene.minBodyPct:
                    return False

        return True

    def _findAllPatternMatches(self, closes: np.ndarray, opens: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray,
                                volumes: np.ndarray, avgVolume: float,
                                patterns: List[PatternChromosome]) -> list:
        """
        Vectorized pattern matching across all patterns on OHLCV arrays.
        Returns list of (predictionIdx, patternIdx, baselineIdx) tuples.

        Uses numpy array operations instead of per-candle Python loops.
        """
        forgiveness = self.mcForgiveness  # wider tolerance for synthetic MC data
        candidates = []

        for patternIdx, pattern in enumerate(patterns):
            patternLength = len(pattern.genes)
            historicalLength = patternLength - 1
            historicalGenes = pattern.genes[:-1]

            # Pre-extract gene data into arrays for vectorized comparison
            genePriceExp = np.array([g.expectedPriceChangePct for g in historicalGenes])
            geneVolumeExp = np.array([g.expectedVolumePct for g in historicalGenes])
            geneCandleType = np.array([['BULLISH', 'BEARISH', 'DOJI'].index(g.candleType.value) for g in historicalGenes])
            geneMinBody = np.array([g.minBodyPct for g in historicalGenes])

            dataLen = len(closes)
            nGenes = historicalLength

            # For each potential starting position, check all genes at once
            for i in range(historicalLength, dataLen - 1):
                baselineIdx = i - historicalLength
                baselinePrice = closes[baselineIdx]
                if baselinePrice == 0:
                    continue

                # Indices for the historical candles of this pattern window
                indices = np.arange(baselineIdx + 1, baselineIdx + 1 + historicalLength)

                # --- SOFT MATCHING: count how many genes pass each check ---
                # On bootstrapped (synthetic) data, requiring ALL genes to pass
                # every check simultaneously is too strict.  Instead, we require
                # that ≥ ceil(nGenes/2) genes pass EACH check (majority vote).
                # This lets patterns trigger enough to produce meaningful MC
                # statistics while still filtering random noise.

                minPassCount = max(1, (nGenes + 1) // 2)  # majority: ceil(n/2)

                # PRICE check — most important signal
                actualPctChg = (closes[indices] - baselinePrice) / baselinePrice
                priceDiffs = np.abs(actualPctChg - genePriceExp)
                pricePass = np.sum(priceDiffs <= forgiveness)
                if pricePass < minPassCount:
                    continue

                # VOLUME check — relaxed tolerance (2.0 for MC vs 0.5 for real)
                volRatios = volumes[indices] / (avgVolume + 1e-9)
                volDiffs = np.abs(volRatios - geneVolumeExp)
                volPass = np.sum(volDiffs <= 2.0)
                if volPass < minPassCount:
                    continue

                # CANDLE TYPE check — soft: majority of genes must match direction
                bodyDirs = closes[indices] - opens[indices]
                bullishFail = (geneCandleType == 0) & (bodyDirs < -0.0001)
                bearishFail = (geneCandleType == 1) & (bodyDirs > 0.0001)
                dojiMask = geneCandleType == 2
                candleTypeFails = bullishFail | bearishFail
                if np.any(dojiMask):
                    tRange = highs[indices] - lows[indices]
                    dojiCheck = dojiMask & (tRange > 0) & (np.abs(bodyDirs) > tRange * 0.15)
                    candleTypeFails = candleTypeFails | dojiCheck
                candlePass = nGenes - np.sum(candleTypeFails)
                if candlePass < minPassCount:
                    continue

                # BODY PCT check — skip for MC (too strict on bootstrap data)
                # Body percentage adds minimal signal vs price+direction checks

                # Enough genes matched — record this candidate
                candidates.append((i + 1, patternIdx, baselineIdx))

        return candidates

    def _checkGeneMatchArrays(self, gene: PatternGene, idx: int,
                               closes: np.ndarray, opens: np.ndarray,
                               highs: np.ndarray, lows: np.ndarray,
                               volumes: np.ndarray,
                               baselinePrice: float, avgVolume: float) -> bool:
        """
        Check a single gene match using pre-extracted numpy arrays
        (avoids pd.Series/iloc overhead).
        """
        actualNormalizedPct = (closes[idx] - baselinePrice) / baselinePrice

        priceDiff = abs(actualNormalizedPct - gene.expectedPriceChangePct)
        if priceDiff > self.forgiveness:
            return False

        volumeRatio = volumes[idx] / avgVolume if avgVolume > 0 else 1.0
        volumeDiff = abs(volumeRatio - gene.expectedVolumePct)
        if volumeDiff > 0.5:
            return False

        bodyDirection = closes[idx] - opens[idx]
        candleTypeInt = ['BULLISH', 'BEARISH', 'DOJI'].index(gene.candleType.value)

        if candleTypeInt == 0 and bodyDirection < -0.0001:
            return False
        elif candleTypeInt == 1 and bodyDirection > 0.0001:
            return False
        elif candleTypeInt == 2:
            totalRange = highs[idx] - lows[idx]
            if totalRange > 0 and abs(bodyDirection) > totalRange * 0.15:
                return False

        if gene.minBodyPct > 0.1:
            totalRange = highs[idx] - lows[idx]
            if totalRange > 0:
                bodyPct = abs(bodyDirection) / totalRange
                if bodyPct < gene.minBodyPct:
                    return False

        return True

    # -------------------------------------------------------------------------
    # Single-path simulation
    # -------------------------------------------------------------------------

    def _simulateSinglePath(self, pathId: int, patterns: List[PatternChromosome],
                            calibration: dict, numPeriods: int) -> SimulationPath:
        """
        Generate one synthetic price path, apply all patterns, record trades,
        and compute per-path metrics (P/L, drawdown, Sharpe).
        """
        # Generate synthetic OHLCV data as raw arrays (skip DataFrame overhead)
        closesArr, opensArr, highsArr, lowsArr, volumesArr, dates = \
            self._generatePricePathArrays(calibration, numPeriods)
        avgVolume = float(volumesArr.mean())

        # Allocate fund per pattern proportional to accuracy
        totalAccuracy = sum(p.getAccuracy() for p in patterns)
        if totalAccuracy == 0:
            patternFunds = {idx: self.initialFund / len(patterns)
                           for idx in range(len(patterns))}
        else:
            patternFunds = {
                idx: self.initialFund * (p.getAccuracy() / totalAccuracy)
                for idx, p in enumerate(patterns)
            }

        path = SimulationPath(
            pathId=pathId,
            finalBalance=self.initialFund,
            peakBalance=self.initialFund,
        )
        
        # Track balance at each period for fan chart visualization
        periodBalances = np.full(numPeriods, self.initialFund, dtype=float)
        currentBalance = self.initialFund
        
        # --- Collect ALL candidate matches using vectorized matching ---
        rawCandidates = self._findAllPatternMatches(
            closesArr, opensArr, highsArr, lowsArr, volumesArr,
            avgVolume, patterns
        )

        # Sort by (predictionIdx, patternIdx) so earlier periods are processed
        # first and, within the same period window, the highest-priority
        # (lowest index) pattern wins.
        rawCandidates.sort(key=lambda c: (c[0], c[1]))

        # Use a numpy boolean array for occupancy instead of set operations
        occupied = np.zeros(numPeriods + 2, dtype=bool)

        for predictionIdx, patternIdx, baselineIdx in rawCandidates:
            pattern = patterns[patternIdx]
            patternLength = len(pattern.genes)

            # Check occupancy using array slice instead of set intersection
            occStart = baselineIdx
            occEnd = min(predictionIdx + 1, len(occupied))
            if np.any(occupied[occStart:occEnd]):
                continue

            predictionGene = pattern.genes[-1]
            baselinePrice = closesArr[baselineIdx]

            # Check prediction gene match using arrays directly
            predictionMatches = self._checkGeneMatchArrays(
                predictionGene, predictionIdx, closesArr, opensArr,
                highsArr, lowsArr, volumesArr, baselinePrice, avgVolume
            )

            entryIdx = predictionIdx - 1
            entryPrice = closesArr[entryIdx]
            exitPrice = closesArr[predictionIdx]
            returnPct = ((exitPrice - entryPrice) / entryPrice) * 100

            fundForPattern = patternFunds[patternIdx]

            if predictionMatches:
                profit = fundForPattern * (abs(returnPct) / 100)
                path.successfulTrades += 1
                path.totalReturnPct += abs(returnPct)
            else:
                profit = -(fundForPattern * (abs(returnPct) / 100))
                path.unsuccessfulTrades += 1
                path.totalReturnPct -= abs(returnPct)

            currentBalance += profit
            path.finalBalance = currentBalance
            path.peakBalance = max(path.peakBalance, currentBalance)

            # Update period balances from this point forward
            periodBalances[predictionIdx:] = currentBalance

            path.trades.append({
                'timestamp': dates[entryIdx],
                'patternId': patternIdx,
                'entryPrice': entryPrice,
                'exitPrice': exitPrice,
                'returnPct': returnPct if predictionMatches else -abs(returnPct),
                'successful': predictionMatches,
                'fundAllocation': fundForPattern,
                'profit': profit,
                'balance': currentBalance,
            })

            path.numTrades += 1
            occupied[occStart:occEnd] = True
        
        # Store the period-by-period balance time series
        path.balanceTimeSeries = periodBalances.tolist()

        # --- Compute per-path max drawdown ---
        if path.trades:
            balances = [self.initialFund] + [t['balance'] for t in path.trades]
            peak = self.initialFund
            maxDrawdown = 0.0
            for bal in balances:
                peak = max(peak, bal)
                dd = ((peak - bal) / peak) * 100 if peak > 0 else 0
                maxDrawdown = max(maxDrawdown, dd)
            path.maxDrawdownPct = maxDrawdown

        # Use compound return (same as Phase 2) so stats are comparable
        path.totalReturnPct = (
            (path.finalBalance - self.initialFund) / self.initialFund * 100
            if self.initialFund > 0 else 0.0
        )

        # --- Compute per-path Sharpe ratio ---
        if path.trades and len(path.trades) > 2:
            tradeReturns = np.array([t['returnPct'] for t in path.trades])
            meanR = np.mean(tradeReturns)
            stdR = np.std(tradeReturns)
            if stdR > 0.01:  # minimum floor to avoid division-by-near-zero
                raw = meanR / stdR
                path.sharpeRatio = float(np.clip(raw, -10.0, 10.0))  # reasonable range
            else:
                path.sharpeRatio = 0.0
        else:
            path.sharpeRatio = 0.0

        return path

    # -------------------------------------------------------------------------
    # Phase 2: Full-pipeline MC simulation (patterns + ML + sentiment)
    # -------------------------------------------------------------------------

    def simulateFullPipeline(
        self,
        patternBank: PatternBank,
        symbol: str,
        interval: str = '1d',
        calibration: Optional[dict] = None,
        calibrationPeriod: str = '2y',
        calibrationStart: Optional[str] = None,
        calibrationEnd: Optional[str] = None,
        mlModel = None,
        headlineGenerator = None,
        tradingDecider = None,
        portfolioSignal = None,
        numSimulations: int = 200,
        numPeriods: int = 252,
        targets: List[float] = None,
        ruinThreshold: float = 0.5,
        verbose: bool = True,
    ) -> 'MonteCarloResults':
        """
        Phase 2 Monte Carlo: simulate the FULL trading pipeline on
        synthetic price paths.

        Unlike Phase 1 (patterns only), each simulated path runs:
          1. Pattern matching  (same as Phase 1)
          2. Synthetic headline generation → sentiment scoring
          3. ML model prediction on the synthetic OHLCV data
          4. TradingDecider blending all signals
          5. Trade execution based on the final blended decision

        This produces expected returns that incorporate ALL system
        components, giving a realistic estimate of live performance.

        Parameters
        ----------
        patternBank : PatternBank
            Refined patterns from GA + Phase 1 MC.
        symbol : str
            Stock ticker.
        interval : str
            Timeframe ('1d', '1h', etc.).
        calibration : dict, optional
            Pre-computed calibration (from Phase 1).
        calibrationPeriod : str
            History period for calibration (if calibration is None).
        calibrationStart, calibrationEnd : str, optional
            Explicit date range for calibration data.
        mlModel : StockMLModel, optional
            Trained per-stock ML model.  If None, ML signals are skipped.
        headlineGenerator : MCSyntheticHeadlineGenerator, optional
            Generates synthetic headlines for sentiment scoring.
        tradingDecider : TradingDecider, optional
            Blends pattern + ML + sentiment + portfolio signals.
        portfolioSignal : PortfolioSignal, optional
            Portfolio-level signal for decision blending.
        numSimulations : int
            Number of simulation paths.
        numPeriods : int
            Periods per path.
        targets : list
            Return-% thresholds for probability reporting.
        ruinThreshold : float
            Capital loss fraction considered ruin.
        verbose : bool
            Print progress.

        Returns
        -------
        MonteCarloResults
            Aggregated Phase 2 results with full-pipeline metrics.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"MC PHASE 2: FULL PIPELINE SIMULATION — {symbol} ({interval})")
            print(f"{'=' * 70}")
            components = ['Patterns']
            if mlModel is not None and mlModel.isTrained:
                components.append('ML Model')
            if headlineGenerator is not None:
                components.append('Synthetic Headlines')
            if tradingDecider is not None:
                components.append('Trading Decider')
            if portfolioSignal is not None:
                components.append('Portfolio Signal')
            print(f"  Components: {' + '.join(components)}")
            print(f"  Simulations: {numSimulations}, Periods: {numPeriods}")

        # Get patterns for this interval
        patterns = patternBank.getPatternsByInterval(interval)
        if not patterns:
            if verbose:
                print(f"  No patterns for {interval}")
            return MonteCarloResults(
                symbol=symbol, interval=interval,
                numSimulations=0, simulationPeriods=0,
                initialFund=self.initialFund,
            )

        # Calibrate if needed
        if calibration is None:
            self._posteriorSamples = None
            self._regimeParams = None
            if calibrationStart and calibrationEnd:
                df = self.fetcher.fetchData(
                    symbol, interval=interval,
                    start=calibrationStart, end=calibrationEnd,
                )
            else:
                df = self.fetcher.fetchData(
                    symbol, interval=interval, period=calibrationPeriod,
                )
            if df is None or len(df) < 30:
                if verbose:
                    print("  Insufficient calibration data")
                return MonteCarloResults(
                    symbol=symbol, interval=interval,
                    numSimulations=0, simulationPeriods=0,
                    initialFund=self.initialFund,
                )
            calibration = self._calibrateFromHistory(df)

        WARMUP = 60  # Extra periods for ML indicator warm-up

        mcResults = MonteCarloResults(
            symbol=symbol, interval=interval,
            numSimulations=numSimulations,
            simulationPeriods=numPeriods,
            initialFund=self.initialFund,
        )

        hasML = mlModel is not None and mlModel.isTrained
        hasSent = headlineGenerator is not None
        hasDecider = tradingDecider is not None

        for pathId in range(numSimulations):
            path = self._simulateFullPipelinePath(
                pathId=pathId,
                patterns=patterns,
                calibration=calibration,
                numPeriods=numPeriods,
                warmupPeriods=WARMUP,
                mlModel=mlModel if hasML else None,
                headlineGenerator=headlineGenerator if hasSent else None,
                tradingDecider=tradingDecider if hasDecider else None,
                portfolioSignal=portfolioSignal,
                symbol=symbol,
            )
            mcResults.paths.append(path)

            if verbose and (pathId + 1) % max(numSimulations // 5, 1) == 0:
                avgRet = np.mean([p.totalReturnPct for p in mcResults.paths])
                avgTrades = np.mean([p.numTrades for p in mcResults.paths])
                pct = (pathId + 1) / numSimulations * 100
                print(f"  [{pct:.0f}%] {pathId + 1}/{numSimulations} | "
                      f"Avg Return: {avgRet:+.2f}% | Avg Trades: {avgTrades:.1f}")

        mcResults.computeMetrics(targets=targets, ruinThreshold=ruinThreshold)

        if verbose:
            print(f"\n  Phase 2 Summary: E[R]={mcResults.expectedReturnPct:+.2f}%, "
                  f"Sharpe={mcResults.avgSharpeRatio:.3f}, "
                  f"P(Profit)={mcResults.probabilityOfProfit:.1f}%, "
                  f"Avg Trades={mcResults.avgTradesPerPath:.1f}")

        return mcResults

    def _simulateFullPipelinePath(
        self,
        pathId: int,
        patterns: List[PatternChromosome],
        calibration: dict,
        numPeriods: int,
        warmupPeriods: int,
        mlModel,
        headlineGenerator,
        tradingDecider,
        portfolioSignal,
        symbol: str,
    ) -> SimulationPath:
        """
        Single Phase 2 path: generates synthetic data and trades using
        the full pipeline (patterns + ML + sentiment + decider).
        """
        totalPeriods = numPeriods + warmupPeriods

        # Generate full synthetic OHLCV DataFrame (including warmup)
        syntheticDF = self._generatePricePath(calibration, totalPeriods)

        closesArr = syntheticDF['close'].values
        opensArr = syntheticDF['open'].values
        highsArr = syntheticDF['high'].values
        lowsArr = syntheticDF['low'].values
        volumesArr = syntheticDF['volume'].values
        avgVolume = float(volumesArr.mean())

        # ---- Generate synthetic sentiment for the path ----
        sentimentSeries = None
        if headlineGenerator is not None:
            sentimentSeries = headlineGenerator.generateForPath(syntheticDF, symbol)

        # ---- Run ML predictions on synthetic data ----
        mlPredictions = None
        if mlModel is not None and mlModel.isTrained:
            try:
                # Temporarily inject synthetic sentiment into the model
                origSentiment = mlModel._sentimentSeries
                if sentimentSeries is not None and len(sentimentSeries) > 0:
                    mlModel._sentimentSeries = sentimentSeries
                # Clear feature cache so it recomputes for synthetic data
                mlModel._featureCacheKey = None
                mlModel._featureCache = None

                mlPredictions = mlModel.predictBatch(syntheticDF)

                # Restore original sentiment
                mlModel._sentimentSeries = origSentiment
                mlModel._featureCacheKey = None
                mlModel._featureCache = None
            except Exception:
                mlPredictions = None

        # ---- Pattern matching on synthetic data ----
        rawCandidates = self._findAllPatternMatches(
            closesArr, opensArr, highsArr, lowsArr, volumesArr,
            avgVolume, patterns,
        )
        rawCandidates.sort(key=lambda c: (c[0], c[1]))

        # Build a set of pattern-triggered periods (after warmup)
        patternSignals = {}  # idx → (patternIdx, baselineIdx)
        for predIdx, patIdx, baseIdx in rawCandidates:
            if predIdx not in patternSignals:
                patternSignals[predIdx] = (patIdx, baseIdx)

        # ---- Trade simulation ----
        path = SimulationPath(
            pathId=pathId,
            finalBalance=self.initialFund,
            peakBalance=self.initialFund,
        )
        periodBalances = np.full(numPeriods, self.initialFund, dtype=float)
        currentBalance = self.initialFund

        # Fund allocation per pattern
        totalAccuracy = sum(p.getAccuracy() for p in patterns) or 1.0
        baseFundPerTrade = self.initialFund / max(len(patterns), 3)

        occupied = np.zeros(totalPeriods + 2, dtype=bool)

        # Walk through trading periods (skip warmup)
        for i in range(warmupPeriods, totalPeriods - 1):
            periodIdx = i - warmupPeriods  # index into output period array

            if occupied[i]:
                if periodIdx < numPeriods:
                    periodBalances[periodIdx] = currentBalance
                continue

            # ---- Collect signals from all sources ----
            hasPatternSignal = i in patternSignals
            mlPred = None
            if mlPredictions is not None:
                # mlPredictions indices correspond to rows after feature
                # computation (which drops some rows for look-back).  Find
                # the prediction closest to this period.
                predTS = syntheticDF.index[i]
                for mp in mlPredictions:
                    if mp.timestamp == predTS:
                        mlPred = mp
                        break
                if mlPred is None and periodIdx < len(mlPredictions):
                    mlPred = mlPredictions[min(periodIdx, len(mlPredictions) - 1)]

            # ---- Decide whether to trade ----
            shouldTrade = False
            tradeDirection = 0  # +1 = long, -1 = short

            if hasDecider := (tradingDecider is not None and mlPred is not None):
                # Use TradingDecider to blend signals
                from StockMLModel import TradingSignal
                finalDecision = tradingDecider.decideForBacktest(
                    stockPred=mlPred,
                    symbol=symbol,
                    portfolioSignal=portfolioSignal,
                    allocation=1.0,
                )
                if hasattr(finalDecision, 'signal'):
                    if finalDecision.signal == TradingSignal.BUY:
                        shouldTrade = True
                        tradeDirection = 1
                    elif finalDecision.signal == TradingSignal.SELL:
                        shouldTrade = True
                        tradeDirection = -1
            elif mlPred is not None:
                # Use ML signal directly (no decider available)
                from StockMLModel import TradingSignal
                if mlPred.signal == TradingSignal.BUY and mlPred.confidence > 0.1:
                    shouldTrade = True
                    tradeDirection = 1
                elif mlPred.signal == TradingSignal.SELL and mlPred.confidence > 0.1:
                    shouldTrade = True
                    tradeDirection = -1
            elif hasPatternSignal:
                # Fallback to pattern-only signal
                shouldTrade = True
                patIdx, baseIdx = patternSignals[i]
                predGene = patterns[patIdx].genes[-1]
                tradeDirection = 1 if predGene.expectedPriceChangePct > 0 else -1

            if not shouldTrade:
                if periodIdx < numPeriods:
                    periodBalances[periodIdx] = currentBalance
                continue

            # ---- Execute trade ----
            entryPrice = closesArr[i]
            exitIdx = min(i + 1, totalPeriods - 1)
            exitPrice = closesArr[exitIdx]
            actualReturn = (exitPrice - entryPrice) / (entryPrice + 1e-9)
            signedReturn = actualReturn * tradeDirection  # positive if correct direction

            fundForTrade = baseFundPerTrade
            if mlPred is not None:
                fundForTrade *= max(0.3, min(1.5, mlPred.positionSize))

            profit = fundForTrade * signedReturn

            currentBalance += profit
            path.finalBalance = currentBalance
            path.peakBalance = max(path.peakBalance, currentBalance)

            isWin = signedReturn > 0
            if isWin:
                path.successfulTrades += 1
            else:
                path.unsuccessfulTrades += 1
            path.numTrades += 1
            path.totalReturnPct += signedReturn * 100

            # Mark occupied
            occupied[i:exitIdx + 1] = True

            if periodIdx < numPeriods:
                periodBalances[periodIdx:] = currentBalance

            path.trades.append({
                'timestamp': syntheticDF.index[i],
                'patternId': patternSignals.get(i, (-1, -1))[0],
                'entryPrice': float(entryPrice),
                'exitPrice': float(exitPrice),
                'returnPct': float(signedReturn * 100),
                'successful': isWin,
                'fundAllocation': float(fundForTrade),
                'profit': float(profit),
                'balance': float(currentBalance),
                'hasMLSignal': mlPred is not None,
                'hasSentiment': sentimentSeries is not None,
                'mlConfidence': float(mlPred.confidence) if mlPred else 0.0,
            })

        path.balanceTimeSeries = periodBalances.tolist()

        # ---- Compute per-path metrics (drawdown, Sharpe) ----
        if path.trades:
            balances = [self.initialFund] + [t['balance'] for t in path.trades]
            peak = self.initialFund
            maxDD = 0.0
            for bal in balances:
                peak = max(peak, bal)
                dd = ((peak - bal) / peak) * 100 if peak > 0 else 0
                maxDD = max(maxDD, dd)
            path.maxDrawdownPct = maxDD

        if path.trades and len(path.trades) > 2:
            tradeReturns = np.array([t['returnPct'] for t in path.trades])
            meanR = np.mean(tradeReturns)
            stdR = np.std(tradeReturns)
            if stdR > 0.01:
                raw = meanR / stdR
                path.sharpeRatio = float(np.clip(raw, -10.0, 10.0))

        path.totalReturnPct = (
            (path.finalBalance - self.initialFund) / self.initialFund * 100
        )

        return path

    # -------------------------------------------------------------------------
    # Per-pattern ranking via Monte Carlo
    # -------------------------------------------------------------------------

    def rankPatterns(self,
                    patternBank: PatternBank,
                    symbol: str,
                    interval: str = '1d',
                    calibrationPeriod: str = '2y',
                    calibrationStart: Optional[str] = None,
                    calibrationEnd: Optional[str] = None,
                    numSimulations: int = 200,
                    numPeriods: int = 252,
                    topN: int = 5,
                    minAvgTradesPerPath: float = 1.0,
                    verbose: bool = True) -> List[Tuple[PatternChromosome, float]]:
        """
        Rank individual patterns by running a Monte Carlo simulation for each
        pattern in isolation and computing a composite score.

        Patterns that produce fewer than *minAvgTradesPerPath* trades per
        simulated path receive a heavy penalty so that overly specific
        (overfit) patterns are ranked below more generalizable ones.

        Composite score = 0.4 * normSharpe + 0.3 * normWinRate + 0.3 * normReturn

        Args:
            patternBank:       PatternBank from the Genetic Algorithm
            symbol:            Stock ticker
            interval:          Timeframe to rank ('1d', '1h', '30m', ...)
            calibrationPeriod: Historical period for model calibration
            numSimulations:    Simulations per pattern (lower than full MC for speed)
            numPeriods:        Periods per simulation path
            topN:              How many patterns to return
            verbose:           Print progress

        Returns:
            List of (PatternChromosome, compositeScore) sorted best-first,
            length min(topN, available patterns).
        """
        patterns = patternBank.getPatternsByInterval(interval)
        if not patterns:
            if verbose:
                print(f"  No patterns for interval {interval}")
            return []

        if verbose:
            print(f"\n  Ranking {len(patterns)} patterns for {interval} via MC ({numSimulations} sims each)...")

        # Fetch & calibrate once for all patterns
        if calibrationStart and calibrationEnd:
            df = self.fetcher.fetchData(symbol, interval=interval,
                                        start=calibrationStart, end=calibrationEnd)
        else:
            df = self.fetcher.fetchData(symbol, interval=interval, period=calibrationPeriod)
        if df is None or len(df) < 30:
            if verbose:
                print(f"  Insufficient data for calibration")
            return [(p, 0.0) for p in patterns[:topN]]

        calibration = self._calibrateFromHistory(df)

        # Run MC for each pattern individually (parallelized across patterns)
        patternScores: List[Tuple[PatternChromosome, float, float, float]] = []

        def evaluate_pattern(pIdx, pattern):
            """Evaluate a single pattern with multiple MC simulations (sequential sims)."""
            pathReturns = []
            pathWinRates = []
            pathSharpes = []
            pathTradeCounts = []
            
            # Run simulations sequentially to avoid nested parallelization deadlock
            for simIdx in range(numSimulations):
                path = self._simulateSinglePath(
                    pathId=simIdx,
                    patterns=[pattern],
                    calibration=calibration,
                    numPeriods=numPeriods,
                )
                pathReturns.append(path.totalReturnPct)
                pathWinRates.append(path.getWinRate())
                pathSharpes.append(path.sharpeRatio)
                pathTradeCounts.append(path.numTrades)
            
            avgReturn = float(np.mean(pathReturns))
            avgWinRate = float(np.mean(pathWinRates))
            avgSharpe = float(np.mean([s for s in pathSharpes if np.isfinite(s)]))
            avgTrades = float(np.mean(pathTradeCounts))
            if not np.isfinite(avgSharpe):
                avgSharpe = 0.0
            
            return (pattern, avgSharpe, avgWinRate, avgReturn, avgTrades, pIdx)

        # Evaluate all patterns in parallel (but sims are sequential within each pattern)
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(evaluate_pattern)(pIdx, pattern)
            for pIdx, pattern in enumerate(patterns)
        )
        
        # Sort by original index to match sequential output
        results.sort(key=lambda r: r[5])
        
        for pattern, avgSharpe, avgWinRate, avgReturn, avgTrades, pIdx in results:
            patternScores.append((pattern, avgSharpe, avgWinRate, avgReturn, avgTrades))
            if verbose:
                tradeWarn = ' ⚠ OVERFIT' if avgTrades < minAvgTradesPerPath else ''
                print(f"    Pattern {pIdx+1}/{len(patterns)}: "
                      f"Sharpe={avgSharpe:.3f}, WinRate={avgWinRate:.1f}%, "
                      f"Return={avgReturn:+.2f}%, AvgTrades={avgTrades:.1f}{tradeWarn}")

        # --- Normalise each metric to [0, 1] across the candidate set ---
        sharpes = np.array([s[1] for s in patternScores])
        winRates = np.array([s[2] for s in patternScores])
        returns = np.array([s[3] for s in patternScores])
        avgTrades = np.array([s[4] for s in patternScores])

        def _minMaxNorm(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            if rng == 0:
                return np.ones_like(arr) * 0.5
            return (arr - arr.min()) / rng

        normSharpe = _minMaxNorm(sharpes)
        normWin = _minMaxNorm(winRates)
        normRet = _minMaxNorm(returns)

        compositeScores = 0.4 * normSharpe + 0.3 * normWin + 0.3 * normRet

        # --- Generalization penalty: penalize patterns that rarely trigger ---
        # Patterns with fewer avg trades than the threshold are likely overfit
        # (too specific to match even synthetic data → will be flat in production)
        for i in range(len(compositeScores)):
            avgT = avgTrades[i]
            if avgT < minAvgTradesPerPath:
                # Proportional penalty: 0 trades → 90% penalty, half-threshold → 45%
                penalty = 1.0 - (avgT / minAvgTradesPerPath) * 0.9
                compositeScores[i] *= (1.0 - penalty)
                if verbose:
                    print(f"    Pattern {i+1}: generalization penalty "
                          f"({avgT:.1f} avg trades < {minAvgTradesPerPath:.1f} threshold) "
                          f"→ score reduced by {penalty*100:.0f}%")

        # Pair pattern with composite score and sort descending
        ranked = [(patternScores[i][0], float(compositeScores[i]))
                  for i in range(len(patternScores))]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"\n  Top {min(topN, len(ranked))} patterns by composite score:")
            for i, (p, score) in enumerate(ranked[:topN]):
                print(f"    #{i+1}: score={score:.3f}, fitness={p.fitness:.2f}, "
                      f"accuracy={p.getAccuracy()*100:.1f}%")

        return ranked[:topN]

    # -------------------------------------------------------------------------
    # Public simulation API
    # -------------------------------------------------------------------------

    def simulate(self,
                 patternBank: PatternBank,
                 symbol: str,
                 interval: str = '1d',
                 calibrationPeriod: str = '2y',
                 calibrationStart: Optional[str] = None,
                 calibrationEnd: Optional[str] = None,
                 targets: List[float] = None,
                 ruinThreshold: float = 0.5,
                 verbose: bool = True) -> MonteCarloResults:
        """
        Run a full Monte Carlo simulation for one interval of a stock.

        Args:
            patternBank:        PatternBank from the Genetic Algorithm
            symbol:             Stock ticker (e.g. 'AAPL')
            interval:           Timeframe to simulate ('1d', '1h', '30m', ...)
            calibrationPeriod:  Historical period for model calibration ('2y', '1y', ...)
                                Only used when calibrationStart/End are not provided.
            calibrationStart:   Explicit start date (YYYY-MM-DD) for calibration data.
            calibrationEnd:     Explicit end date (YYYY-MM-DD) for calibration data.
            targets:            List of return-% targets for probability calc
            ruinThreshold:      Fraction of capital loss considered ruin (0.5 = 50%)
            verbose:            Print progress
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"MONTE CARLO SIMULATION — {symbol} ({interval})")
            print(f"{'=' * 70}")
            print(f"Method: {self.method.upper()}")
            if self.method == 'mcmc':
                print(f"  (Bayesian posterior sampling via Metropolis-Hastings)")
            elif self.method == 'regime':
                print(f"  (2-state bull/bear regime-switching HMM)")
            print(f"Simulations: {self.numSimulations:,}")
            print(f"Periods per path: {self.simulationPeriods}")
            print(f"Initial Fund: ${self.initialFund:,.2f}")

        # Get patterns for this interval
        patterns = patternBank.getPatternsByInterval(interval)
        if not patterns:
            if verbose:
                print(f"No patterns found for interval {interval}")
            return MonteCarloResults(
                symbol=symbol, interval=interval,
                numSimulations=0, simulationPeriods=0,
                initialFund=self.initialFund,
            )

        # Reset MCMC / regime caches for this new calibration
        self._posteriorSamples = None
        self._regimeParams = None

        if verbose:
            print(f"Patterns: {len(patterns)}")
            for i, p in enumerate(patterns):
                print(f"  Pattern {i+1}: length={len(p.genes)}, "
                      f"accuracy={p.getAccuracy()*100:.1f}%, "
                      f"fitness={p.fitness:.2f}")

        # Fetch historical data for calibration
        if calibrationStart and calibrationEnd:
            calLabel = f"{calibrationStart} to {calibrationEnd}"
            if verbose:
                print(f"\nCalibrating from training-period data ({calLabel})...")
            df = self.fetcher.fetchData(symbol, interval=interval,
                                        start=calibrationStart, end=calibrationEnd)
        else:
            calLabel = calibrationPeriod
            if verbose:
                print(f"\nCalibrating from historical data ({calLabel})...")
            df = self.fetcher.fetchData(symbol, interval=interval, period=calibrationPeriod)

        if df is None or len(df) < 30:
            if verbose:
                n = len(df) if df is not None else 0
                print(f"Insufficient historical data for calibration ({n} candles)")
            return MonteCarloResults(
                symbol=symbol, interval=interval,
                numSimulations=0, simulationPeriods=0,
                initialFund=self.initialFund,
            )

        if verbose:
            print(f"Historical candles: {len(df)}")

        # Calibrate model
        calibration = self._calibrateFromHistory(df)

        if verbose:
            print(f"Calibration: mu={calibration['mu']*100:.4f}%/period, "
                  f"sigma={calibration['sigma']*100:.4f}%/period")
            print(f"Last close: ${calibration['lastClose']:,.2f}")
            print(f"\nRunning {self.numSimulations:,} simulations...")

        # Run simulations
        mcResults = MonteCarloResults(
            symbol=symbol,
            interval=interval,
            numSimulations=self.numSimulations,
            simulationPeriods=self.simulationPeriods,
            initialFund=self.initialFund,
        )

        # Run simulations in parallel (using all CPU cores)
        if verbose:
            print(f"  Parallelizing across all CPU cores...")
        
        paths = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._simulateSinglePath)(i, patterns, calibration, self.simulationPeriods)
            for i in range(self.numSimulations)
        )
        
        mcResults.paths.extend(paths)
        
        if verbose:
            avgReturn = np.mean([p.totalReturnPct for p in mcResults.paths])
            avgTrades = np.mean([p.numTrades for p in mcResults.paths])
            print(f"  [100.0%] {self.numSimulations:,}/{self.numSimulations:,} | "
                  f"Avg Return: {avgReturn:+.2f}% | Avg Trades: {avgTrades:.1f}")

        # Compute aggregate metrics
        mcResults.computeMetrics(targets=targets, ruinThreshold=ruinThreshold)

        if verbose:
            print(mcResults.summary())

        return mcResults

    def simulateAll(self,
                    patternBank: PatternBank,
                    symbol: str,
                    calibrationPeriod: str = '2y',
                    calibrationStart: Optional[str] = None,
                    calibrationEnd: Optional[str] = None,
                    targets: List[float] = None,
                    ruinThreshold: float = 0.5,
                    verbose: bool = True) -> Dict[str, MonteCarloResults]:
        """
        Run Monte Carlo simulation across every interval present in the
        pattern bank.

        Returns:
            Dictionary mapping interval -> MonteCarloResults
        """
        intervals = sorted(set(p.interval for p in patternBank.patterns if p.interval))

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"FULL MONTE CARLO SIMULATION — {symbol}")
            print(f"Intervals: {intervals}")
            print(f"{'=' * 70}")

        from concurrent.futures import ThreadPoolExecutor

        def _simOneInterval(interval):
            return interval, self.simulate(
                patternBank=patternBank,
                symbol=symbol,
                interval=interval,
                calibrationPeriod=calibrationPeriod,
                calibrationStart=calibrationStart,
                calibrationEnd=calibrationEnd,
                targets=targets,
                ruinThreshold=ruinThreshold,
                verbose=verbose,
            )

        allResults = {}
        if len(intervals) > 1:
            with ThreadPoolExecutor(max_workers=len(intervals)) as pool:
                for interval, result in pool.map(_simOneInterval, intervals):
                    allResults[interval] = result
        else:
            for interval in intervals:
                _, result = _simOneInterval(interval)
                allResults[interval] = result

        return allResults

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------

    def generateReport(self,
                       results: Dict[str, MonteCarloResults],
                       savePath: Optional[str] = None,
                       showPlot: bool = True):
        """
        Create a multi-panel figure for each interval:
          Row 1: P/L distribution histogram with VaR / CVaR lines
          Row 2: Cumulative balance paths (spaghetti + percentile bands)
          Row 3: Max-drawdown distribution
          Row 4: Summary statistics table
        """
        validResults = {k: v for k, v in results.items() if v.paths}

        if not validResults:
            print("No simulation results to visualise.")
            return

        numIntervals = len(validResults)

        fig, axes = plt.subplots(4, numIntervals,
                                 figsize=(8 * numIntervals, 24))

        # Handle single-interval case (axes is 1-D)
        if numIntervals == 1:
            axes = axes.reshape(-1, 1)

        symbol = list(validResults.values())[0].symbol
        fig.suptitle(f"Monte Carlo Simulation Report — {symbol}",
                     fontsize=16, fontweight='bold', y=0.98)

        for colIdx, (interval, mcResult) in enumerate(sorted(validResults.items())):
            self._plotReturnDistribution(axes[0, colIdx], mcResult, interval)
            self._plotBalancePaths(axes[1, colIdx], mcResult, interval)
            self._plotDrawdownDistribution(axes[2, colIdx], mcResult, interval)
            self._plotSummaryTable(axes[3, colIdx], mcResult, interval)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
            print(f"\nReport saved to: {savePath}")

        if showPlot:
            plt.show()
    
    def generateFanChart(self,
                        results: Dict[str, MonteCarloResults],
                        savePath: Optional[str] = None,
                        showPlot: bool = True):
        """
        Create a standalone fan chart visualization showing all Monte Carlo paths
        spreading from the initial fund value.
        
        This creates a separate figure from the main report, focusing purely on
        the path evolution visualization.
        
        Args:
            results: Dictionary of MonteCarloResults by interval
            savePath: Optional path to save the figure
            showPlot: Whether to display the plot
        """
        validResults = {k: v for k, v in results.items() if v.paths}

        if not validResults:
            print("No simulation results to visualise.")
            return

        numIntervals = len(validResults)
        
        # Create figure — tall and wide for maximum impact
        fig, axes = plt.subplots(numIntervals, 1,
                                 figsize=(16, 9 * numIntervals))
        
        # Handle single-interval case (axes is not array)
        if numIntervals == 1:
            axes = [axes]

        symbol = list(validResults.values())[0].symbol
        fig.suptitle(f"Monte Carlo Fan Chart — {symbol}",
                     fontsize=20, fontweight='bold', y=0.99)

        for idx, (interval, mcResult) in enumerate(sorted(validResults.items())):
            self._plotBalancePaths(axes[idx], mcResult, interval)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.subplots_adjust(hspace=0.25)

        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
            print(f"\nFan chart saved to: {savePath}")

        if showPlot:
            plt.show()

    # --- individual plot helpers ------------------------------------------

    def _plotReturnDistribution(self, ax, mc: MonteCarloResults, interval: str):
        """Histogram of total return % across all paths, with VaR/CVaR lines."""
        returns = [p.totalReturnPct for p in mc.paths]

        n, bins, patches = ax.hist(
            returns, bins=60, alpha=0.7, color='#1f77b4',
            edgecolor='black', linewidth=0.3, density=True,
        )

        # Colour the loss side red
        for patch, edge in zip(patches, bins):
            if edge < 0:
                patch.set_facecolor('#d62728')
                patch.set_alpha(0.7)

        # VaR / CVaR lines
        ax.axvline(mc.var95, color='orange', lw=2, ls='--',
                   label=f'VaR 95%: {mc.var95:+.1f}%')
        ax.axvline(mc.var99, color='red', lw=2, ls='--',
                   label=f'VaR 99%: {mc.var99:+.1f}%')
        ax.axvline(mc.cvar95, color='orange', lw=1.5, ls=':',
                   label=f'CVaR 95%: {mc.cvar95:+.1f}%')

        # Mean / median
        ax.axvline(mc.expectedReturnPct, color='green', lw=2, ls='-',
                   label=f'Mean: {mc.expectedReturnPct:+.1f}%')
        ax.axvline(mc.medianReturnPct, color='lime', lw=1.5, ls='-.',
                   label=f'Median: {mc.medianReturnPct:+.1f}%')

        # Zero line
        ax.axvline(0, color='black', lw=1, alpha=0.3)

        ax.set_title(f'{interval} — Return Distribution', fontsize=11, fontweight='bold')
        ax.set_xlabel('Total Return (%)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

    def _plotBalancePaths(self, ax, mc: MonteCarloResults, interval: str):
        """Monte Carlo fan chart showing all simulated balance paths over time."""
        if not mc.paths or len(mc.paths) == 0:
            ax.text(0.5, 0.5, 'No Simulation Paths', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{interval} — Balance Paths', fontsize=11, fontweight='bold')
            return
        
        # Collect all period-by-period balances
        allBalances = []
        for path in mc.paths:
            if path.balanceTimeSeries and len(path.balanceTimeSeries) > 0:
                allBalances.append(np.array(path.balanceTimeSeries))
        
        if not allBalances:
            ax.text(0.5, 0.5, 'No Balance Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{interval} — Balance Paths', fontsize=11, fontweight='bold')
            return
        
        # Convert to numpy array for easier manipulation
        balanceMatrix = np.array(allBalances)
        numPeriods = balanceMatrix.shape[1]
        numPaths = len(allBalances)
        periods = np.arange(numPeriods)
        
        # --- Detect flat simulations (no trades triggered) ---
        finalBalances = balanceMatrix[:, -1]
        balanceSpread = np.std(finalBalances)
        avgTrades = np.mean([p.numTrades for p in mc.paths]) if mc.paths else 0

        if balanceSpread < 0.01 * mc.initialFund and avgTrades < 0.5:
            # All paths are essentially flat — no pattern matches in synthetic data
            ax.axhline(mc.initialFund, color='gray', lw=2, ls='-', alpha=0.6)
            ax.text(0.5, 0.55,
                    f'FLAT: Avg {avgTrades:.1f} trades/path\n'
                    'Patterns are too specific to match synthetic data.\n'
                    'This may indicate overfitting or tight forgiveness.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#c0392b', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdecea',
                              edgecolor='#e74c3c', alpha=0.9))
            ax.set_facecolor('#f8f9fa')
            ax.set_title(f'{interval} — Monte Carlo Fan Chart  '
                         f'({numPaths:,} sims, FLAT — no trades)',
                         fontsize=12, fontweight='bold', pad=10,
                         color='#c0392b')
            ax.set_xlabel('Simulation Period', fontsize=10)
            ax.set_ylabel('Portfolio Balance ($)', fontsize=10)
            margin = mc.initialFund * 0.05
            ax.set_ylim(mc.initialFund - margin, mc.initialFund + margin)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'${x:,.0f}'))
            ax.grid(True, alpha=0.25, linestyle='-', color='#cccccc')
            return

        # --- Sort paths by final balance so colors map from worst to best ---
        sortOrder = np.argsort(finalBalances)
        
        # Color map: red (worst) → yellow → green (best)
        cmap = plt.cm.RdYlGn
        normalizedRank = np.linspace(0, 1, numPaths)
        pathColors = cmap(normalizedRank)
        
        # Decide how many to draw
        maxDraw = min(numPaths, 800)
        if numPaths > maxDraw:
            drawIdx = np.sort(np.random.choice(numPaths, maxDraw, replace=False))
        else:
            drawIdx = np.arange(numPaths)
        
        # Plot each path, colored by rank
        for rank, pathIdx in enumerate(sortOrder):
            if pathIdx not in drawIdx:
                continue
            # Rank-based color
            color = pathColors[rank]
            ax.plot(periods, balanceMatrix[pathIdx], alpha=0.18, lw=0.5,
                    color=color, zorder=1)
        
        # --- Percentile shading bands ---
        p5  = np.percentile(balanceMatrix, 5,  axis=0)
        p10 = np.percentile(balanceMatrix, 10, axis=0)
        p25 = np.percentile(balanceMatrix, 25, axis=0)
        p50 = np.percentile(balanceMatrix, 50, axis=0)
        p75 = np.percentile(balanceMatrix, 75, axis=0)
        p90 = np.percentile(balanceMatrix, 90, axis=0)
        p95 = np.percentile(balanceMatrix, 95, axis=0)
        
        ax.fill_between(periods, p5, p95, alpha=0.06, color='steelblue', zorder=2)
        ax.fill_between(periods, p10, p90, alpha=0.08, color='steelblue', zorder=2)
        ax.fill_between(periods, p25, p75, alpha=0.10, color='steelblue', zorder=2)
        
        # --- Key percentile lines ---
        ax.plot(periods, p50, color='white', lw=3.0, zorder=97, alpha=0.7)  # white glow
        ax.plot(periods, p50, color='black', lw=2.0, zorder=98,
                label=f'Median: ${p50[-1]:,.0f}')
        ax.plot(periods, p5, color='#c0392b', lw=1.4, ls='--', zorder=96, alpha=0.8,
                label=f'5th pctl: ${p5[-1]:,.0f}')
        ax.plot(periods, p95, color='#27ae60', lw=1.4, ls='--', zorder=96, alpha=0.8,
                label=f'95th pctl: ${p95[-1]:,.0f}')
        ax.plot(periods, p25, color='#e67e22', lw=1.0, ls=':', zorder=95, alpha=0.5,
                label=f'25th pctl: ${p25[-1]:,.0f}')
        ax.plot(periods, p75, color='#2980b9', lw=1.0, ls=':', zorder=95, alpha=0.5,
                label=f'75th pctl: ${p75[-1]:,.0f}')
        
        # --- Initial fund reference line ---
        ax.axhline(mc.initialFund, color='white', lw=2, alpha=0.6, zorder=93)
        ax.axhline(mc.initialFund, color='dimgray', lw=1.2, ls='-', alpha=0.6,
                   zorder=94, label=f'Initial: ${mc.initialFund:,.0f}')
        
        # --- Annotate final values on right edge ---
        rightX = numPeriods - 1
        fontSize = 7
        ax.annotate(f'${p95[-1]:,.0f}', xy=(rightX, p95[-1]),
                    fontsize=fontSize, color='#27ae60', fontweight='bold',
                    va='bottom', ha='left',
                    xytext=(5, 2), textcoords='offset points')
        ax.annotate(f'${p50[-1]:,.0f}', xy=(rightX, p50[-1]),
                    fontsize=fontSize, color='black', fontweight='bold',
                    va='center', ha='left',
                    xytext=(5, 0), textcoords='offset points')
        ax.annotate(f'${p5[-1]:,.0f}', xy=(rightX, p5[-1]),
                    fontsize=fontSize, color='#c0392b', fontweight='bold',
                    va='top', ha='left',
                    xytext=(5, -2), textcoords='offset points')
        
        # --- Styling ---
        ax.set_facecolor('#f8f9fa')
        ax.set_title(f'{interval} — Monte Carlo Fan Chart  ({numPaths:,} simulations)',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Simulation Period', fontsize=10)
        ax.set_ylabel('Portfolio Balance ($)', fontsize=10)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85,
                  fancybox=True, shadow=True, ncol=2)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'${x:,.0f}'))
        ax.grid(True, alpha=0.25, linestyle='-', color='#cccccc')
        
        # Subtle frame
        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_linewidth(0.5)

    def _plotDrawdownDistribution(self, ax, mc: MonteCarloResults, interval: str):
        """Histogram of max-drawdown % across all paths."""
        drawdowns = [p.maxDrawdownPct for p in mc.paths]

        if all(d == 0 for d in drawdowns):
            ax.text(0.5, 0.5, 'No Drawdowns Recorded', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{interval} — Max Drawdown', fontsize=11, fontweight='bold')
            return

        ax.hist(drawdowns, bins=50, alpha=0.7, color='#d62728',
                edgecolor='black', linewidth=0.3, density=True)

        avg = np.mean(drawdowns)
        median = np.median(drawdowns)
        p95 = np.percentile(drawdowns, 95)

        ax.axvline(avg, color='blue', lw=2, ls='-', label=f'Mean: {avg:.1f}%')
        ax.axvline(median, color='cyan', lw=1.5, ls='-.', label=f'Median: {median:.1f}%')
        ax.axvline(p95, color='darkred', lw=2, ls='--', label=f'95th pctl: {p95:.1f}%')

        ax.set_title(f'{interval} — Max Drawdown Distribution',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Max Drawdown (%)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

    def _plotSummaryTable(self, ax, mc: MonteCarloResults, interval: str):
        """Render key statistics as a styled table."""
        ax.axis('off')

        rows = [
            ['Expected Return',  f'{mc.expectedReturnPct:+.2f}%'],
            ['Median Return',    f'{mc.medianReturnPct:+.2f}%'],
            ['Std Deviation',    f'{mc.stdReturnPct:.2f}%'],
            ['Best Case',        f'{mc.bestReturnPct:+.2f}%'],
            ['Worst Case',       f'{mc.worstReturnPct:+.2f}%'],
            ['', ''],
            ['VaR (95%)',        f'{mc.var95:+.2f}%'],
            ['VaR (99%)',        f'{mc.var99:+.2f}%'],
            ['CVaR (95%)',       f'{mc.cvar95:+.2f}%'],
            ['CVaR (99%)',       f'{mc.cvar99:+.2f}%'],
            ['', ''],
            ['Avg Max Drawdown', f'{mc.avgMaxDrawdown:.2f}%'],
            ['Worst Drawdown',   f'{mc.worstMaxDrawdown:.2f}%'],
            ['', ''],
            ['P(Profit)',        f'{mc.probabilityOfProfit:.1f}%'],
            ['P(Ruin)',          f'{mc.probabilityOfRuin:.1f}%'],
            ['Avg Sharpe',       f'{mc.avgSharpeRatio:.3f}'],
            ['Avg Trades/Path',  f'{mc.avgTradesPerPath:.1f}'],
            ['Avg Win Rate',     f'{mc.avgWinRate:.1f}%'],
        ]

        # Target probabilities
        if mc.targetProbabilities:
            rows.append(['', ''])
            for target, prob in sorted(mc.targetProbabilities.items()):
                rows.append([f'P(>= {target:+.0f}%)', f'{prob:.1f}%'])

        table = ax.table(cellText=rows,
                         colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor('#2c3e50')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Alternate row colours, collapse blank separator rows
        for i in range(1, len(rows) + 1):
            if i <= len(rows):
                for j in range(2):
                    if rows[i - 1][0] == '':
                        table[(i, j)].set_height(0.015)
                        table[(i, j)].set_facecolor('white')
                        table[(i, j)].set_edgecolor('white')
                    else:
                        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

        ax.set_title(f'{interval} — Summary Statistics',
                     fontsize=11, fontweight='bold', pad=10)


# =============================================================================
# Example / standalone usage
# =============================================================================

if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithmPatternFinder

    print("=" * 70)
    print("MONTE CARLO SIMULATION FOR PATTERN-BASED TRADING")
    print("=" * 70)

    # ---- Configuration ----
    SYMBOL = 'AAPL'
    INITIAL_FUND = 10000
    NUM_SIMULATIONS = 1000
    SIMULATION_PERIODS = 252       # ~1 trading year
    METHOD = 'bootstrap'           # 'gbm' or 'bootstrap'
    CALIBRATION_PERIOD = '2y'
    RUIN_THRESHOLD = 0.50          # 50% loss = ruin
    TARGETS = [5, 10, 20, 50, 100] # Return targets for probability calc

    # GA Settings (for pattern discovery)
    INTERVALS = [('1d', '5y'), ('1h', '730d')]
    PATTERN_LENGTHS = [3, 4, 5, 6, 7]
    POPULATION_SIZE = 200
    NUM_GENERATIONS = 50
    NUM_RUNS = 2

    # ---- Discover Patterns ----
    print(f"\nDiscovering patterns for {SYMBOL}...")
    ga = GeneticAlgorithmPatternFinder(
        populationSize=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        forgivenessPct=0.05,
        earlyStopGenerations=15,
    )

    patternBank = ga.discoverPatternBank(
        symbol=SYMBOL,
        intervals=INTERVALS,
        patternLengths=PATTERN_LENGTHS,
        numRunsPerConfig=NUM_RUNS,
        verbose=True,
    )

    print(f"\nPattern bank: {len(patternBank.patterns)} patterns")
    print(patternBank.summary())

    # ---- Run Monte Carlo Simulation ----
    simulator = MCMCSimulator(
        initialFund=INITIAL_FUND,
        forgiveness=0.05,
        numSimulations=NUM_SIMULATIONS,
        simulationPeriods=SIMULATION_PERIODS,
        method=METHOD,
    )

    allResults = simulator.simulateAll(
        patternBank=patternBank,
        symbol=SYMBOL,
        calibrationPeriod=CALIBRATION_PERIOD,
        targets=TARGETS,
        ruinThreshold=RUIN_THRESHOLD,
        verbose=True,
    )

    # ---- Generate Report ----
    print("\n📊 Generating full Monte Carlo report...")
    simulator.generateReport(
        results=allResults,
        savePath='monte_carlo_report.png',
        showPlot=True,
    )
    
    print("\n📈 Generating standalone fan chart...")
    simulator.generateFanChart(
        results=allResults,
        savePath='monte_carlo_fan_chart.png',
        showPlot=True,
    )

    print("\n✅ Monte Carlo simulation complete!")
    print("   - Full report: monte_carlo_report.png")
    print("   - Fan chart: monte_carlo_fan_chart.png")
