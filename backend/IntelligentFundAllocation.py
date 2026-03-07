"""
IntelligentFundAllocation  —  Dynamic Capital Allocation by Performance
=========================================================================

Static 40 / 30 / 30 splits don't cut it.  If one stock bleeds money for
six months straight, why keep feeding it capital?

This module analyses per-stock **and** per-timeframe backtesting results
and reallocates the total fund accordingly:

  1.  Score every stock (+timeframe) by risk-adjusted return.
  2.  Shrink allocation for chronic losers (can reach 0 %).
  3.  Boost allocation for consistent winners.
  4.  **Shadow mode**: even at 0 % allocation, keep running simulated
      (zero-capital) trades so we can detect when a stock turns around
      and deserves capital again.
  5.  Periodically re-evaluate: if a previously-zero stock shows a
      positive trend over a rolling window, gradually bring it back.

The module is designed to be called inside the PortfolioTester pipeline
AFTER backtesting and BEFORE the next trading cycle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from Backtester import BacktestResult


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class TimeframePerformance:
    """Performance record for one stock in one timeframe."""
    interval: str          # e.g. '1d', '1h', '30m'
    totalTrades: int = 0
    winRate: float = 0.0   # 0-100
    totalReturnPct: float = 0.0
    avgReturnPct: float = 0.0
    sharpeRatio: float = 0.0
    profitFactor: float = 0.0  # gross profit / gross loss (>1 = profitable)

    @property
    def score(self) -> float:
        """Composite score combining return, risk-adjusted return, and consistency."""
        if self.totalTrades == 0:
            return 0.0
        # Weighted formula:
        #   50 % Sharpe, 25 % total return, 25 % profit factor
        return (
            0.50 * max(-2.0, min(3.0, self.sharpeRatio))
            + 0.25 * np.clip(self.totalReturnPct / 100.0, -1.0, 2.0)
            + 0.25 * np.clip(self.profitFactor - 1.0, -1.0, 2.0)
        )


@dataclass
class StockPerformanceSummary:
    """Aggregated performance across all timeframes for a single stock."""
    symbol: str
    currentAllocation: float                       # current fraction (0-1)
    timeframes: List[TimeframePerformance] = field(default_factory=list)
    overallScore: float = 0.0                      # calculated composite
    suggestedAllocation: float = 0.0               # new fraction (0-1)
    isShadow: bool = False                         # True if allocation dropped to 0
    rollingReturnPct: float = 0.0                  # recent rolling-window return

    @property
    def bestTimeframe(self) -> Optional[TimeframePerformance]:
        """Return the timeframe with the highest composite score."""
        if not self.timeframes:
            return None
        return max(self.timeframes, key=lambda t: t.score)

    @property
    def worstTimeframe(self) -> Optional[TimeframePerformance]:
        if not self.timeframes:
            return None
        return min(self.timeframes, key=lambda t: t.score)


@dataclass
class AllocationResult:
    """The output of the allocation engine for the entire portfolio."""
    stockSummaries: Dict[str, StockPerformanceSummary] = field(default_factory=dict)
    previousAllocations: Dict[str, float] = field(default_factory=dict)
    newAllocations: Dict[str, float] = field(default_factory=dict)
    shadowStocks: List[str] = field(default_factory=list)   # stocks at 0 % (still tracked)
    restoredStocks: List[str] = field(default_factory=list)  # stocks brought back from 0 %


# =====================================================================
# Intelligent Fund Allocation Engine
# =====================================================================

class IntelligentFundAllocation:
    """
    Dynamically adjust per-stock capital allocation based on backtested
    performance.

    Parameters
    ----------
    minAllocation : float
        Hard floor per stock.  Set to 0.0 to allow full removal.
    maxAllocation : float
        Hard ceiling per stock (concentration limit).
    shadowThreshold : float
        If a stock's overall score drops below this, its allocation is
        zeroed out and it enters shadow mode.  Default -0.3.
    restoreThreshold : float
        A shadow stock is restored when its rolling return exceeds this
        value (%).  Default 5.0 (i.e. 5 % rolling return).
    restoreAllocation : float
        The initial allocation given to a restored stock.  Default 0.05
        (5 %).
    smoothingFactor : float
        Limits how fast allocations change (0-1).
        New = old × smooth + suggested × (1 - smooth).
        0.0 = instant switch, 0.5 = half-life per cycle.  Default 0.4.
    rollingWindowTrades : int
        How many recent trades to use for the rolling return calculation
        (for shadow-stock recovery detection).  Default 30.
    """

    def __init__(
        self,
        minAllocation: float = 0.0,
        maxAllocation: float = 0.60,
        shadowThreshold: float = -0.3,
        restoreThreshold: float = 5.0,
        restoreAllocation: float = 0.05,
        smoothingFactor: float = 0.4,
        rollingWindowTrades: int = 30,
    ):
        self.minAllocation = minAllocation
        self.maxAllocation = maxAllocation
        self.shadowThreshold = shadowThreshold
        self.restoreThreshold = restoreThreshold
        self.restoreAllocation = restoreAllocation
        self.smoothingFactor = smoothingFactor
        self.rollingWindowTrades = rollingWindowTrades

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def allocate(
        self,
        currentAllocations: Dict[str, float],
        backtestResults: Dict[str, Dict[str, 'BacktestResult']],
        verbose: bool = False,
    ) -> AllocationResult:
        """
        Compute new allocations for every stock.

        Parameters
        ----------
        currentAllocations : dict
            {symbol: fraction} — the allocations entering this cycle.
        backtestResults : dict
            {symbol: {interval: BacktestResult}} — full backtest data
            for every stock and timeframe.  Shadow stocks MUST still
            have results here (from zero-capital simulated trades).
        verbose : bool
            Print the allocation table.

        Returns
        -------
        AllocationResult
        """
        result = AllocationResult()
        result.previousAllocations = dict(currentAllocations)

        # --- 1. Build per-stock performance summaries ---
        summaries: Dict[str, StockPerformanceSummary] = {}
        for symbol, alloc in currentAllocations.items():
            tfResults = backtestResults.get(symbol, {})
            summary = self._buildSummary(symbol, alloc, tfResults)
            summaries[symbol] = summary

        # --- 2. Score each stock ---
        for summary in summaries.values():
            summary.overallScore = self._computeOverallScore(summary)
            summary.isShadow = (
                summary.overallScore < self.shadowThreshold
                and summary.currentAllocation <= self.minAllocation + 0.001
            )

        # --- 3. Check shadow-stock recovery ---
        restoredStocks = []
        for symbol, summary in summaries.items():
            if summary.currentAllocation < 0.001:
                # It's currently at zero — check if it deserves restoration
                if summary.rollingReturnPct >= self.restoreThreshold:
                    restoredStocks.append(symbol)
                    summary.isShadow = False

        # --- 4. Compute raw suggested allocations ---
        rawAllocations = self._scoreToAllocation(summaries, restoredStocks)

        # --- 5. Apply smoothing ---
        smoothedAllocations = self._smooth(currentAllocations, rawAllocations)

        # --- 6. Enforce floor / ceiling and normalise ---
        finalAllocations = self._clampAndNormalise(smoothedAllocations)

        # --- 7. Identify shadow stocks ---
        shadowStocks = [
            sym for sym, alloc in finalAllocations.items()
            if alloc < 0.001
        ]

        # --- 8. Package result ---
        for symbol, summary in summaries.items():
            summary.suggestedAllocation = finalAllocations.get(symbol, 0.0)
            summary.isShadow = symbol in shadowStocks

        result.stockSummaries = summaries
        result.newAllocations = finalAllocations
        result.shadowStocks = shadowStocks
        result.restoredStocks = restoredStocks

        if verbose:
            self._printReport(result)

        return result

    # -----------------------------------------------------------------
    # Internal — performance analysis
    # -----------------------------------------------------------------

    def _buildSummary(
        self,
        symbol: str,
        currentAllocation: float,
        tfResults: Dict[str, 'BacktestResult'],
    ) -> StockPerformanceSummary:
        """Build a StockPerformanceSummary from raw backtest results."""
        summary = StockPerformanceSummary(
            symbol=symbol,
            currentAllocation=currentAllocation,
        )

        allReturns: List[float] = []

        for interval, btResult in tfResults.items():
            if btResult.totalTrades == 0:
                continue

            # Per-trade returns
            tradeReturns = [t['returnPct'] for t in btResult.trades]
            grossProfit = sum(r for r in tradeReturns if r > 0) or 0.0001
            grossLoss = abs(sum(r for r in tradeReturns if r < 0)) or 0.0001

            mean_r = np.mean(tradeReturns) if tradeReturns else 0.0
            std_r = np.std(tradeReturns) if len(tradeReturns) > 1 else 1.0
            sharpe = float(mean_r / (std_r + 1e-9))

            tfPerf = TimeframePerformance(
                interval=interval,
                totalTrades=btResult.totalTrades,
                winRate=btResult.getSuccessRate(),
                totalReturnPct=btResult.totalReturnPct,
                avgReturnPct=float(mean_r),
                sharpeRatio=sharpe,
                profitFactor=grossProfit / grossLoss,
            )
            summary.timeframes.append(tfPerf)
            allReturns.extend(tradeReturns)

        # Rolling return for recovery detection (most recent N trades)
        if allReturns:
            recentReturns = allReturns[-self.rollingWindowTrades:]
            summary.rollingReturnPct = float(sum(recentReturns))

        return summary

    def _computeOverallScore(self, summary: StockPerformanceSummary) -> float:
        """
        Weighted average of timeframe scores.  Timeframes with more
        trades get more weight (they're more statistically reliable).
        """
        if not summary.timeframes:
            return 0.0

        totalTrades = sum(tf.totalTrades for tf in summary.timeframes)
        if totalTrades == 0:
            return 0.0

        weightedScore = 0.0
        for tf in summary.timeframes:
            weight = tf.totalTrades / totalTrades
            weightedScore += tf.score * weight

        return weightedScore

    # -----------------------------------------------------------------
    # Internal — allocation computation
    # -----------------------------------------------------------------

    def _scoreToAllocation(
        self,
        summaries: Dict[str, StockPerformanceSummary],
        restoredStocks: List[str],
    ) -> Dict[str, float]:
        """
        Convert overall scores into raw allocation fractions.

        Approach:
          - Stocks with score <= shadowThreshold → 0 %
          - Stocks with positive scores get proportionally more capital
          - Stocks with near-zero scores get a baseline minimum
          - Restored stocks get ``restoreAllocation`` as a starting point
        """
        allocations: Dict[str, float] = {}

        # Separate into active and zero-out pools
        activeScores: Dict[str, float] = {}
        for symbol, summary in summaries.items():
            score = summary.overallScore
            if symbol in restoredStocks:
                # Give a fixed starting allocation for restored stocks
                allocations[symbol] = self.restoreAllocation
            elif score < self.shadowThreshold:
                allocations[symbol] = 0.0
            else:
                # Shift score so that 0 maps to a small positive value
                adjustedScore = max(0.01, score - self.shadowThreshold)
                activeScores[symbol] = adjustedScore

        # Remaining capital (after restored stock allocations)
        reservedCapital = sum(
            alloc for sym, alloc in allocations.items()
            if alloc > 0
        )
        availableCapital = max(0.0, 1.0 - reservedCapital)

        # Proportional allocation among active stocks
        totalActiveScore = sum(activeScores.values())
        if totalActiveScore > 0 and availableCapital > 0:
            for symbol, adjScore in activeScores.items():
                allocations[symbol] = (adjScore / totalActiveScore) * availableCapital
        else:
            # Fallback: equal weight among active stocks
            if activeScores:
                equalShare = availableCapital / len(activeScores)
                for symbol in activeScores:
                    allocations[symbol] = equalShare

        return allocations

    def _smooth(
        self,
        current: Dict[str, float],
        suggested: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Exponential smoothing to prevent violent allocation swings
        between cycles.
        """
        smoothed: Dict[str, float] = {}
        s = self.smoothingFactor
        for symbol in set(current) | set(suggested):
            old = current.get(symbol, 0.0)
            new = suggested.get(symbol, 0.0)
            # If dropping to zero (shadow), skip smoothing
            if new < 0.001:
                smoothed[symbol] = 0.0
            else:
                smoothed[symbol] = old * s + new * (1 - s)
        return smoothed

    def _clampAndNormalise(
        self, allocations: Dict[str, float]
    ) -> Dict[str, float]:
        """Enforce min/max and normalise to sum to 1.0."""
        clamped: Dict[str, float] = {}
        for symbol, alloc in allocations.items():
            if alloc < 0.001:
                clamped[symbol] = 0.0  # true zero for shadow stocks
            else:
                clamped[symbol] = max(self.minAllocation, min(self.maxAllocation, alloc))

        totalActive = sum(v for v in clamped.values() if v > 0)
        if totalActive > 0:
            for symbol in clamped:
                if clamped[symbol] > 0:
                    clamped[symbol] = round(clamped[symbol] / totalActive, 4)

        return clamped

    # -----------------------------------------------------------------
    # Internal — reporting
    # -----------------------------------------------------------------

    def _printReport(self, result: AllocationResult):
        print(f"\n  {'─' * 80}")
        print("  INTELLIGENT FUND ALLOCATION — Performance-Based Rebalancing")
        print(f"  {'─' * 80}")

        header = (f"  {'Symbol':<8} {'OldAlloc':>9} {'Score':>7} "
                  f"{'BestTF':<8} {'WinRate':>8} {'TotalRet':>9} "
                  f"{'NewAlloc':>9} {'Status':<10}")
        print(header)
        print(f"  {'─' * len(header)}")

        for sym in sorted(result.stockSummaries.keys()):
            s = result.stockSummaries[sym]
            bestTF = s.bestTimeframe
            tfStr = bestTF.interval if bestTF else '--'
            winStr = f"{bestTF.winRate:.0f}%" if bestTF else '--'
            retStr = f"{bestTF.totalReturnPct:+.1f}%" if bestTF else '--'

            oldAlloc = result.previousAllocations.get(sym, 0.0)
            newAlloc = result.newAllocations.get(sym, 0.0)

            if sym in result.restoredStocks:
                status = 'RESTORED'
            elif sym in result.shadowStocks:
                status = 'SHADOW'
            else:
                change = newAlloc - oldAlloc
                if abs(change) < 0.005:
                    status = '—'
                elif change > 0:
                    status = f'+{change*100:.1f}%'
                else:
                    status = f'{change*100:.1f}%'

            print(f"  {sym:<8} {oldAlloc*100:>8.1f}% {s.overallScore:>+6.3f} "
                  f"{tfStr:<8} {winStr:>8} {retStr:>9} "
                  f"{newAlloc*100:>8.1f}% {status:<10}")

        if result.shadowStocks:
            print(f"\n  Shadow stocks (0% alloc, simulated trades continue): "
                  f"{', '.join(result.shadowStocks)}")
        if result.restoredStocks:
            print(f"  Restored stocks (back from shadow): "
                  f"{', '.join(result.restoredStocks)}")
        print()
