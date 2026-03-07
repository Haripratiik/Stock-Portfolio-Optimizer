"""
DynamicAllocator  —  Continuous Per-Stock-Per-Timeframe Fund Allocation
=========================================================================

Allocates capital at the (stock, interval) level, dynamically THROUGHOUT
the backtest — not just once at the end.  The allocator:

  1. Receives all backtest trades (run with a normalised notional fund)
  2. Replays them chronologically in evaluation windows
  3. After each window, re-evaluates every slot's rolling performance
  4. Adjusts allocation weights for the next window
  5. Scales each trade's dollar P/L by the slot's current allocation

Key features
------------
  - **Performance-based**:  bad slots get reduced, good slots get increased
  - **Minimum evaluation period**:  no changes until enough data accumulates
  - **Ghost trades**:  0 % slots still run so recovery can be detected
  - **ML-trained**:  GradientBoosting learns optimal allocation from rolling
    performance history (predicts forward return → higher → more capital)
  - **Dual rule + ML** approach with configurable blending
  - **Smoothing** prevents violent swings between windows
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from Backtester import BacktestResult

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class SlotKey:
    """Unique identifier for a (stock, interval) allocation slot."""
    symbol: str
    interval: str

    def __hash__(self):
        return hash((self.symbol, self.interval))

    def __eq__(self, other):
        if not isinstance(other, SlotKey):
            return False
        return self.symbol == other.symbol and self.interval == other.interval

    def __repr__(self):
        return f"{self.symbol}/{self.interval}"


@dataclass
class SlotPerformance:
    """Rolling performance snapshot for one (stock, interval) slot."""
    key: SlotKey
    totalTrades: int = 0
    recentTrades: int = 0            # trades in last rolling window
    winRate: float = 0.0             # 0-100
    recentWinRate: float = 0.0       # 0-100, last N trades
    totalReturnPct: float = 0.0
    recentReturnPct: float = 0.0     # sum of last N trade returns
    avgReturnPct: float = 0.0
    sharpeRatio: float = 0.0
    recentSharpe: float = 0.0
    maxDrawdownPct: float = 0.0
    profitFactor: float = 0.0
    consecutiveLosses: int = 0
    consecutiveWins: int = 0
    isGhost: bool = False            # True if allocation == 0
    currentAllocation: float = 0.0   # current fraction of total fund

    @property
    def ruleScore(self) -> float:
        """
        Rule-based composite score.  Higher = better.
        Range roughly [-3, +5].

          40 % recent Sharpe (capped)
          25 % recent return
          20 % recent win rate (centred on 50 %)
          10 % profit factor bonus
           5 % streak penalty / bonus
        """
        if self.totalTrades == 0:
            return 0.0

        s = 0.0
        s += 0.40 * np.clip(self.recentSharpe, -2.0, 3.0)
        s += 0.25 * np.clip(self.recentReturnPct / 100.0, -1.0, 2.0)
        s += 0.20 * np.clip((self.recentWinRate - 50.0) / 50.0, -1.0, 1.0)
        s += 0.10 * np.clip(self.profitFactor - 1.0, -1.0, 2.0)
        if self.consecutiveLosses >= 5:
            s -= 0.05 * min(self.consecutiveLosses, 10) / 10.0
        if self.consecutiveWins >= 3:
            s += 0.05 * min(self.consecutiveWins, 10) / 10.0

        return float(s)


@dataclass
class AllocationResult:
    """Output of the allocation engine."""
    slotAllocations: Dict[str, float] = field(default_factory=dict)
    # key = "SYMBOL/interval" → fraction of total fund
    stockAllocations: Dict[str, float] = field(default_factory=dict)
    # key = "SYMBOL" → fraction of total fund
    slotPerformances: Dict[str, SlotPerformance] = field(default_factory=dict)
    ghostSlots: List[str] = field(default_factory=list)
    restoredSlots: List[str] = field(default_factory=list)
    method: str = "rule"
    previousAllocations: Dict[str, float] = field(default_factory=dict)
    newAllocations: Dict[str, float] = field(default_factory=dict)
    shadowStocks: List[str] = field(default_factory=list)
    # Dynamic allocation history (one entry per checkpoint)
    allocationHistory: List[Dict] = field(default_factory=list)


@dataclass
class DynamicReplayResult:
    """Result of replaying trades with dynamic allocation weights."""
    totalPnL: float = 0.0
    totalReturnPct: float = 0.0
    slotPnL: Dict[str, float] = field(default_factory=dict)
    allocationHistory: List[Dict] = field(default_factory=list)
    # [{timestamp, allocations: {slotStr: frac}, method: str}]
    replayedTrades: Dict[str, List[dict]] = field(default_factory=dict)
    # {slotStr: [trade dicts with dynamicPnL]}
    windowSummaries: List[Dict] = field(default_factory=list)
    finalAllocations: Dict[str, float] = field(default_factory=dict)
    finalPerformances: Dict[str, SlotPerformance] = field(default_factory=dict)
    method: str = "rule"


# =====================================================================
# Dynamic Allocator Engine
# =====================================================================

class DynamicAllocator:
    """
    Per-slot fund allocator with continuous rebalancing during replay.

    All backtests are run with a fixed notional fund ($10 k).  Then
    ``replayWithDynamicAllocation`` walks through every trade
    chronologically, scaling dollar P/L by the slot's current allocation
    fraction and re-evaluating at regular checkpoints.

    Parameters
    ----------
    minSlotAllocation   : Hard floor per slot (0 = allow ghost).
    maxSlotAllocation   : Hard ceiling per slot (concentration limit).
    maxStockAllocation  : Hard ceiling per STOCK across all intervals.
    ghostThreshold      : Rule-score below which a slot is ghosted.
    restoreThreshold    : Recent return % a ghost needs for restoration.
    restoreAllocation   : Starting allocation for a restored slot.
    smoothingFactor     : EMA dampen (0 = instant, 1 = no change).
    rollingWindow       : Recent-N trades for "recent" metrics.
    useML               : Use ML model alongside rule-based scoring.
    mlBlendWeight       : Blend weight (0 = pure rule, 1 = pure ML).
    evalWindowDays      : Calendar days between evaluation checkpoints.
    minEvalPeriodDays   : Minimum days before the first reallocation.
    useDynamicReallocation : If False, keep initial allocations fixed (no rebalancing).
    """

    def __init__(
        self,
        useDynamicReallocation: bool = True,
        minSlotAllocation: float = 0.0,
        maxSlotAllocation: float = 0.40,
        maxStockAllocation: float = 0.60,
        ghostThreshold: float = -0.3,
        restoreThreshold: float = 3.0,
        restoreAllocation: float = 0.03,
        smoothingFactor: float = 0.35,
        rollingWindow: int = 15,
        useML: bool = True,
        mlBlendWeight: float = 0.4,
        evalWindowDays: int = 5,
        minEvalPeriodDays: int = 10,
    ):
        self.useDynamicReallocation = useDynamicReallocation
        self.minSlotAllocation = minSlotAllocation
        self.maxSlotAllocation = maxSlotAllocation
        self.maxStockAllocation = maxStockAllocation
        self.ghostThreshold = ghostThreshold
        self.restoreThreshold = restoreThreshold
        self.restoreAllocation = restoreAllocation
        self.smoothingFactor = smoothingFactor
        self.rollingWindow = rollingWindow
        self.useML = useML and _HAS_SKLEARN
        self.mlBlendWeight = mlBlendWeight
        self.evalWindowDays = evalWindowDays
        self.minEvalPeriodDays = minEvalPeriodDays

        # ML model internals
        self._mlModel: Optional[GradientBoostingRegressor] = None
        self._mlScaler: Optional[StandardScaler] = None
        self._mlTrained: bool = False
        self._trainingHistory: List[Dict] = []
        self._featureNames: List[str] = []

    # =================================================================
    # Public API — Dynamic Replay (main entry point)
    # =================================================================

    def replayWithDynamicAllocation(
        self,
        allSlotTrades: Dict[str, List[dict]],
        totalFund: float,
        initialAllocations: Dict[str, float],
        notionalFund: float = 10_000.0,
        verbose: bool = False,
    ) -> DynamicReplayResult:
        """
        Replay trades chronologically with dynamic allocation weights.

        Every slot must have been backtested with the SAME ``notionalFund``.
        The dollar P/L of each trade is scaled by::

            scaledPnL = originalPnL × (totalFund × slotAllocation) / notionalFund

        Evaluation checkpoints happen every ``evalWindowDays`` calendar
        days, starting only after ``minEvalPeriodDays`` of data.

        Parameters
        ----------
        allSlotTrades : dict
            ``{slotStr: [trade dicts]}`` where ``slotStr = "SYMBOL/interval"``.
        totalFund : float
            Total portfolio capital.
        initialAllocations : dict
            ``{slotStr: fraction}`` — starting allocation (sums ≈ 1.0).
        notionalFund : float
            Fixed fund each backtest used (default $10,000).
        verbose : bool
            Print progress.

        Returns
        -------
        DynamicReplayResult
        """
        result = DynamicReplayResult()
        if not allSlotTrades:
            return result

        allSlotStrings = sorted(allSlotTrades.keys())

        # --- 1. Flatten trades with slot tag and sort by timestamp --------
        taggedTrades: List[dict] = []
        for slotStr, trades in allSlotTrades.items():
            for trade in trades:
                taggedTrades.append({**trade, '_slot': slotStr})

        if not taggedTrades:
            return result

        taggedTrades.sort(key=lambda t: self._asTimestamp(t['timestamp']))

        # --- 2. Compute evaluation checkpoints ---------------------------
        firstTs = self._asTimestamp(taggedTrades[0]['timestamp'])
        lastTs = self._asTimestamp(taggedTrades[-1]['timestamp'])

        firstCheckpoint = firstTs + pd.Timedelta(days=self.minEvalPeriodDays)
        checkpoints: List[pd.Timestamp] = []
        cp = firstCheckpoint
        while cp <= lastTs:
            checkpoints.append(cp)
            cp += pd.Timedelta(days=self.evalWindowDays)

        if verbose:
            print(f"\n  Dynamic allocation replay:")
            print(f"    Period       : {firstTs.strftime('%Y-%m-%d')} → "
                  f"{lastTs.strftime('%Y-%m-%d')}")
            print(f"    Checkpoints  : {len(checkpoints)} "
                  f"(first after {self.minEvalPeriodDays}d, "
                  f"then every {self.evalWindowDays}d)")
            print(f"    Slots        : {', '.join(allSlotStrings)}")
            print(f"    Total trades : {len(taggedTrades)}")

        # --- 3. Initialise state -----------------------------------------
        currentAllocations = dict(initialAllocations)

        # Running trade lists per slot (for performance evaluation)
        slotTradeHistory: Dict[str, List[dict]] = {s: [] for s in allSlotStrings}
        # Cumulative signed return per slot (for ML targets)
        slotCumReturn: Dict[str, float] = {s: 0.0 for s in allSlotStrings}

        # Replayed trades per slot
        replayedPerSlot: Dict[str, List[dict]] = {s: [] for s in allSlotStrings}

        totalPnL = 0.0
        slotPnL: Dict[str, float] = {s: 0.0 for s in allSlotStrings}

        # Allocation history
        allocationHistory: List[Dict] = [{
            'timestamp': firstTs,
            'allocations': dict(currentAllocations),
            'method': 'initial',
        }]

        # ML training data: pending features from previous checkpoint
        prevCheckpointFeatures: Dict[str, List[float]] = {}
        prevCheckpointCumReturns: Dict[str, float] = {}

        checkpointIdx = 0

        # --- 4. Replay chronologically -----------------------------------
        for trade in taggedTrades:
            slotStr = trade['_slot']
            tradeTs = self._asTimestamp(trade['timestamp'])

            # 4a. Check if we hit a checkpoint BEFORE processing this trade
            while (checkpointIdx < len(checkpoints)
                   and tradeTs >= checkpoints[checkpointIdx]):
                cpTs = checkpoints[checkpointIdx]

                # Evaluate all slot performances from history so far
                slotPerfs = self._evaluateSlotPerformances(
                    slotTradeHistory, currentAllocations, allSlotStrings,
                )

                # Collect ML training pair: prev features → forward return
                if prevCheckpointFeatures:
                    for s in allSlotStrings:
                        if s in prevCheckpointFeatures:
                            forwardReturn = (slotCumReturn[s]
                                             - prevCheckpointCumReturns.get(s, 0.0))
                            self._trainingHistory.append({
                                'features': prevCheckpointFeatures[s],
                                'target': forwardReturn,
                                'slot': s,
                            })

                # Save current features as "pending" for next ML target
                prevCheckpointFeatures = {}
                prevCheckpointCumReturns = dict(slotCumReturn)
                for s in allSlotStrings:
                    perf = slotPerfs.get(s)
                    if perf and perf.totalTrades > 0:
                        prevCheckpointFeatures[s] = self._getFeatureVector(perf)

                # Compute new allocations (skip if static mode)
                if self.useDynamicReallocation:
                    newAllocations = self._computeNewAllocations(
                        slotPerfs, currentAllocations, allSlotStrings, verbose=False,
                    )
                else:
                    newAllocations = dict(currentAllocations)

                # Record
                allocationHistory.append({
                    'timestamp': cpTs,
                    'allocations': dict(newAllocations),
                    'method': ('static' if not self.useDynamicReallocation
                               else ('blended' if self.useML and self._mlTrained
                                     else 'rule')),
                })

                # Window summary (only meaningful when dynamic)
                changes = {}
                if self.useDynamicReallocation:
                    for s in allSlotStrings:
                        oldA = currentAllocations.get(s, 0.0)
                        newA = newAllocations.get(s, 0.0)
                        if abs(newA - oldA) > 0.003:
                            changes[s] = {'old': oldA, 'new': newA,
                                          'delta': newA - oldA}
                result.windowSummaries.append({
                    'checkpoint': cpTs,
                    'method': allocationHistory[-1]['method'],
                    'changes': changes,
                })

                if verbose and changes:
                    print(f"\n    Checkpoint {cpTs.strftime('%Y-%m-%d')} "
                          f"({allocationHistory[-1]['method']}):")
                    for s, ch in sorted(changes.items()):
                        arrow = '\u2191' if ch['delta'] > 0 else '\u2193'
                        print(f"      {s}: {ch['old']*100:.1f}% \u2192 "
                              f"{ch['new']*100:.1f}% {arrow}")

                currentAllocations = newAllocations
                checkpointIdx += 1

            # 4b. Record raw trade in slot history
            slotTradeHistory[slotStr].append(trade)
            slotCumReturn[slotStr] += self._signedReturn(trade)

            # 4c. Compute scaled P/L
            slotAlloc = currentAllocations.get(slotStr, 0.0)
            isGhost = slotAlloc < 0.001

            originalPnL = trade.get('dollarPnL', 0.0)
            if isGhost:
                scaledPnL = 0.0
            else:
                scaleFactor = (totalFund * slotAlloc) / notionalFund
                scaledPnL = originalPnL * scaleFactor

            totalPnL += scaledPnL
            slotPnL[slotStr] += scaledPnL

            # 4d. Store replayed trade
            replayedTrade = dict(trade)
            replayedTrade['dollarPnL'] = scaledPnL
            replayedTrade['originalPnL'] = originalPnL
            replayedTrade['slotAllocation'] = slotAlloc
            replayedTrade['isGhost'] = isGhost
            if isGhost:
                replayedTrade['fundAllocation'] = 0.0
            else:
                origFundAlloc = trade.get('fundAllocation', 0.0)
                replayedTrade['fundAllocation'] = origFundAlloc * scaleFactor
            # Remove internal tag
            replayedTrade.pop('_slot', None)
            replayedPerSlot[slotStr].append(replayedTrade)

        # --- 5. Final results ---------------------------------------------
        result.totalPnL = totalPnL
        result.totalReturnPct = ((totalPnL / totalFund * 100)
                                 if totalFund > 0 else 0.0)
        result.slotPnL = slotPnL
        result.allocationHistory = allocationHistory
        result.replayedTrades = replayedPerSlot
        result.finalAllocations = currentAllocations
        result.method = ('static'
                         if not self.useDynamicReallocation
                         else ('dynamic (rule + ML)'
                               if self.useML and self._mlTrained
                               else 'dynamic (rule)'))

        # Final performance snapshot
        result.finalPerformances = self._evaluateSlotPerformances(
            slotTradeHistory, currentAllocations, allSlotStrings,
        )

        if verbose:
            self._printReplaySummary(result, allSlotStrings)

        return result

    # =================================================================
    # Internal — Allocation pipeline (called at each checkpoint)
    # =================================================================

    def _computeNewAllocations(
        self,
        slotPerfs: Dict[str, SlotPerformance],
        currentAllocations: Dict[str, float],
        allSlotStrings: List[str],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Full allocation pipeline: rule → ML → blend → smooth → clamp."""

        # 1. Rule-based
        ruleAllocs = self._ruleBasedAllocation(slotPerfs)

        # 2. ML-based (if enough history)
        mlAllocs = None
        usedML = False
        if self.useML and len(self._trainingHistory) >= 10:
            self._trainMLModel(verbose=verbose)
            if self._mlTrained:
                mlAllocs = self._mlPredict(slotPerfs, verbose=verbose)
                usedML = True

        # 3. Blend
        if usedML and mlAllocs:
            blended = self._blendAllocations(ruleAllocs, mlAllocs)
        else:
            blended = ruleAllocs

        # 4. Smooth against current allocations
        smoothed = self._smooth(currentAllocations, blended)

        # 5. Clamp & normalise
        final = self._clampAndNormalise(smoothed)

        return final

    # =================================================================
    # Internal — Performance evaluation from trade lists
    # =================================================================

    @staticmethod
    def _asTimestamp(ts) -> pd.Timestamp:
        """Convert any timestamp type to pandas Timestamp."""
        if isinstance(ts, pd.Timestamp):
            return ts
        if isinstance(ts, datetime):
            return pd.Timestamp(ts)
        if isinstance(ts, str):
            return pd.Timestamp(ts)
        return pd.Timestamp(ts)

    @staticmethod
    def _signedReturn(trade: dict) -> float:
        """Get signed return % from a trade dict.

        Uses dollarPnL sign as the authority (returnPct may be stored
        as absolute in some code paths).
        """
        pnl = trade.get('dollarPnL', 0.0)
        rpct = abs(trade.get('returnPct', 0.0))
        return rpct if pnl >= 0 else -rpct

    def _evaluateSlotPerformances(
        self,
        slotTradeHistory: Dict[str, List[dict]],
        currentAllocations: Dict[str, float],
        allSlotStrings: List[str],
    ) -> Dict[str, SlotPerformance]:
        """Build SlotPerformance for each slot from running trade history."""
        perfs: Dict[str, SlotPerformance] = {}

        for slotStr in allSlotStrings:
            parts = slotStr.split('/')
            symbol = parts[0]
            interval = parts[1] if len(parts) > 1 else ''
            key = SlotKey(symbol, interval)

            trades = slotTradeHistory.get(slotStr, [])
            alloc = currentAllocations.get(slotStr, 0.0)

            perf = SlotPerformance(key=key, currentAllocation=alloc)
            perf.isGhost = alloc < 0.001

            if not trades:
                perfs[slotStr] = perf
                continue

            returns = [self._signedReturn(t) for t in trades]

            perf.totalTrades = len(returns)
            perf.winRate = (100.0 * sum(1 for r in returns if r > 0)
                            / len(returns))
            perf.totalReturnPct = sum(returns)
            perf.avgReturnPct = float(np.mean(returns))

            # Sharpe
            if len(returns) > 1:
                std = float(np.std(returns))
                perf.sharpeRatio = float(np.mean(returns) / (std + 1e-9))

            # Recent metrics
            recent = returns[-self.rollingWindow:]
            perf.recentTrades = len(recent)
            perf.recentWinRate = (100.0 * sum(1 for r in recent if r > 0)
                                  / len(recent))
            perf.recentReturnPct = sum(recent)
            if len(recent) > 1:
                rStd = float(np.std(recent))
                perf.recentSharpe = float(np.mean(recent) / (rStd + 1e-9))
            else:
                perf.recentSharpe = perf.sharpeRatio

            # Profit factor
            grossProfit = sum(r for r in returns if r > 0) or 0.0001
            grossLoss = abs(sum(r for r in returns if r < 0)) or 0.0001
            perf.profitFactor = grossProfit / grossLoss

            # Max drawdown
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = peak - cumulative
            perf.maxDrawdownPct = (float(np.max(drawdowns))
                                   if len(drawdowns) > 0 else 0.0)

            # Consecutive streaks
            consLoss = consWin = maxConsLoss = maxConsWin = 0
            for r in returns:
                if r <= 0:
                    consLoss += 1; consWin = 0
                else:
                    consWin += 1; consLoss = 0
                maxConsLoss = max(maxConsLoss, consLoss)
                maxConsWin = max(maxConsWin, consWin)
            perf.consecutiveLosses = maxConsLoss
            perf.consecutiveWins = maxConsWin

            perfs[slotStr] = perf

        return perfs

    # =================================================================
    # Internal — Rule-based allocation
    # =================================================================

    def _ruleBasedAllocation(
        self,
        slotPerfs: Dict[str, SlotPerformance],
    ) -> Dict[str, float]:
        """
        Score each slot → proportional allocation.

        Ghost: score < ghostThreshold → 0 %.
        Restore: ghost + recent return > restoreThreshold → small alloc.
        Active: proportional to adjusted score.
        """
        allocations: Dict[str, float] = {}
        activeScores: Dict[str, float] = {}

        for slotStr, perf in slotPerfs.items():
            score = perf.ruleScore

            # Ghost restoration check
            if perf.currentAllocation < 0.001:
                if (perf.recentReturnPct >= self.restoreThreshold
                        and perf.recentTrades >= 3):
                    allocations[slotStr] = self.restoreAllocation
                else:
                    allocations[slotStr] = 0.0
                continue

            # Active slot — ghost if bad enough
            if score < self.ghostThreshold and perf.totalTrades >= 5:
                allocations[slotStr] = 0.0
            else:
                adjScore = max(0.01, score - self.ghostThreshold)
                activeScores[slotStr] = adjScore

        # Distribute capital proportionally
        reservedCapital = sum(a for a in allocations.values() if a > 0)
        availableCapital = max(0.0, 1.0 - reservedCapital)

        totalScore = sum(activeScores.values())
        if totalScore > 0 and availableCapital > 0:
            for slotStr, adjScore in activeScores.items():
                allocations[slotStr] = (adjScore / totalScore) * availableCapital
        elif activeScores:
            equalShare = availableCapital / len(activeScores)
            for slotStr in activeScores:
                allocations[slotStr] = equalShare

        return allocations

    # =================================================================
    # Internal — ML allocation model
    # =================================================================

    def _getFeatureVector(self, perf: SlotPerformance) -> List[float]:
        """14-feature vector from a slot performance snapshot."""
        return [
            perf.totalTrades,
            perf.recentTrades,
            perf.winRate / 100.0,
            perf.recentWinRate / 100.0,
            perf.totalReturnPct,
            perf.recentReturnPct,
            perf.avgReturnPct,
            perf.sharpeRatio,
            perf.recentSharpe,
            perf.maxDrawdownPct,
            perf.profitFactor,
            perf.consecutiveLosses,
            perf.consecutiveWins,
            perf.currentAllocation,
        ]

    def _getFeatureNames(self) -> List[str]:
        return [
            'totalTrades', 'recentTrades', 'winRate', 'recentWinRate',
            'totalReturnPct', 'recentReturnPct', 'avgReturnPct',
            'sharpeRatio', 'recentSharpe', 'maxDrawdownPct',
            'profitFactor', 'consecutiveLosses', 'consecutiveWins',
            'currentAllocation',
        ]

    def _trainMLModel(self, verbose: bool = False):
        """Train GBR to predict forward return from performance features."""
        if not _HAS_SKLEARN or len(self._trainingHistory) < 10:
            return

        X = np.array([h['features'] for h in self._trainingHistory])
        y = np.array([h['target'] for h in self._trainingHistory])

        # Weight recent observations more heavily
        n = len(X)
        weights = np.linspace(0.5, 1.5, n)

        self._mlScaler = StandardScaler()
        X_scaled = self._mlScaler.fit_transform(X)

        self._mlModel = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._mlModel.fit(X_scaled, y, sample_weight=weights)
        self._mlTrained = True
        self._featureNames = self._getFeatureNames()

        if verbose:
            importances = self._mlModel.feature_importances_
            topIdx = np.argsort(importances)[::-1][:5]
            featStr = ', '.join(
                f"{self._featureNames[i]}={importances[i]:.3f}"
                for i in topIdx
            )
            print(f"      ML allocator trained on {n} snapshots | "
                  f"Top features: {featStr}")

    def _mlPredict(
        self,
        slotPerfs: Dict[str, SlotPerformance],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Predict forward return for each slot → convert to allocation.

        Higher predicted forward return → higher allocation weight.
        """
        if not self._mlTrained or self._mlModel is None:
            return {}

        predictions: Dict[str, float] = {}
        for slotStr, perf in slotPerfs.items():
            features = np.array([self._getFeatureVector(perf)])
            features_scaled = self._mlScaler.transform(features)
            pred = float(self._mlModel.predict(features_scaled)[0])
            predictions[slotStr] = pred

        # Convert predicted returns to allocation weights:
        # shift so minimum maps to a small positive value, then normalise
        minPred = min(predictions.values()) if predictions else 0.0
        shifted = {k: max(0.001, v - minPred + 0.01)
                   for k, v in predictions.items()}
        total = sum(shifted.values())
        if total > 0:
            return {k: v / total for k, v in shifted.items()}
        # Fallback: equal
        n = max(len(shifted), 1)
        return {k: 1.0 / n for k in shifted}

    # =================================================================
    # Internal — Blending
    # =================================================================

    def _blendAllocations(
        self,
        ruleAllocs: Dict[str, float],
        mlAllocs: Dict[str, float],
    ) -> Dict[str, float]:
        """Blend rule-based and ML-based allocations."""
        w = self.mlBlendWeight
        blended: Dict[str, float] = {}

        allSlots = set(ruleAllocs) | set(mlAllocs)
        for slotStr in allSlots:
            rVal = ruleAllocs.get(slotStr, 0.0)
            mVal = mlAllocs.get(slotStr, 0.0)

            # Rule ghosts (0%) cannot be overridden by ML
            if rVal < 0.001:
                blended[slotStr] = 0.0
            else:
                blended[slotStr] = rVal * (1.0 - w) + mVal * w

        return blended

    # =================================================================
    # Internal — Smoothing and constraints
    # =================================================================

    def _smooth(
        self,
        current: Dict[str, float],
        suggested: Dict[str, float],
    ) -> Dict[str, float]:
        """EMA smoothing between current and suggested allocations."""
        s = self.smoothingFactor
        smoothed: Dict[str, float] = {}
        allSlots = set(current) | set(suggested)
        for slotStr in allSlots:
            old = current.get(slotStr, 0.0)
            new = suggested.get(slotStr, 0.0)
            if new < 0.001:
                smoothed[slotStr] = 0.0   # ghost immediately
            else:
                smoothed[slotStr] = old * s + new * (1 - s)
        return smoothed

    def _clampAndNormalise(
        self,
        allocations: Dict[str, float],
    ) -> Dict[str, float]:
        """Enforce per-slot max, per-stock max, and normalize to sum = 1."""
        # Step 1: Per-slot ceiling
        clamped: Dict[str, float] = {}
        for slotStr, alloc in allocations.items():
            if alloc < 0.001:
                clamped[slotStr] = 0.0
            else:
                clamped[slotStr] = min(alloc, self.maxSlotAllocation)

        # Step 2: Per-stock ceiling
        stockTotals: Dict[str, float] = {}
        for slotStr, alloc in clamped.items():
            symbol = slotStr.split('/')[0]
            stockTotals[symbol] = stockTotals.get(symbol, 0.0) + alloc

        for symbol, total in stockTotals.items():
            if total > self.maxStockAllocation:
                scale = self.maxStockAllocation / total
                for slotStr in clamped:
                    if slotStr.startswith(f"{symbol}/"):
                        clamped[slotStr] *= scale

        # Step 3: Normalise to sum = 1.0
        totalActive = sum(v for v in clamped.values() if v > 0)
        if totalActive > 0:
            for slotStr in clamped:
                if clamped[slotStr] > 0:
                    clamped[slotStr] = round(clamped[slotStr] / totalActive, 4)

        return clamped

    # =================================================================
    # One-shot API (backward-compat / post-backtest)
    # =================================================================

    def allocate(
        self,
        currentAllocations: Dict[str, float],
        backtestResults: Dict[str, Dict[str, 'BacktestResult']],
        verbose: bool = False,
    ) -> AllocationResult:
        """
        One-shot allocation from backtest results (kept for
        backward compatibility; the main pipeline uses
        ``replayWithDynamicAllocation`` instead).
        """
        result = AllocationResult()
        result.previousAllocations = dict(currentAllocations)

        slotPerfs: Dict[str, SlotPerformance] = {}
        for symbol, alloc in currentAllocations.items():
            tfResults = backtestResults.get(symbol, {})
            if not tfResults:
                continue
            nIntervals = max(len(tfResults), 1)
            perSlotAlloc = alloc / nIntervals
            for intervalKey, btResult in tfResults.items():
                interval = intervalKey.replace('ML_', '')
                slotStr = f"{symbol}/{interval}"
                perf = self._buildSlotPerformanceBT(
                    SlotKey(symbol, interval), btResult,
                    currentAllocation=perSlotAlloc,
                )
                slotPerfs[slotStr] = perf

        if not slotPerfs:
            result.newAllocations = dict(currentAllocations)
            result.stockAllocations = dict(currentAllocations)
            return result

        ruleAllocs = self._ruleBasedAllocation(slotPerfs)
        final = self._clampAndNormalise(ruleAllocs)

        stockAllocs: Dict[str, float] = {}
        for slotStr, alloc in final.items():
            symbol = slotStr.split('/')[0]
            stockAllocs[symbol] = stockAllocs.get(symbol, 0.0) + alloc

        result.slotAllocations = final
        result.stockAllocations = stockAllocs
        result.slotPerformances = slotPerfs
        result.newAllocations = stockAllocs
        return result

    # =================================================================
    # Internal — BacktestResult → SlotPerformance (for one-shot API)
    # =================================================================

    def _buildSlotPerformanceBT(
        self,
        key: SlotKey,
        btResult: 'BacktestResult',
        currentAllocation: float = 0.0,
    ) -> SlotPerformance:
        """Build SlotPerformance from a BacktestResult object."""
        perf = SlotPerformance(key=key, currentAllocation=currentAllocation)
        if btResult.totalTrades == 0:
            return perf

        returns = []
        for t in btResult.trades:
            pnl = t.get('dollarPnL', 0.0)
            rpct = abs(t.get('returnPct', 0.0))
            returns.append(rpct if pnl >= 0 else -rpct)

        perf.totalTrades = len(returns)
        perf.winRate = (100.0 * sum(1 for r in returns if r > 0)
                        / len(returns)) if returns else 0.0
        perf.totalReturnPct = sum(returns)
        perf.avgReturnPct = float(np.mean(returns)) if returns else 0.0

        if len(returns) > 1:
            std = float(np.std(returns))
            perf.sharpeRatio = float(np.mean(returns) / (std + 1e-9))

        recent = returns[-self.rollingWindow:]
        perf.recentTrades = len(recent)
        perf.recentWinRate = (100.0 * sum(1 for r in recent if r > 0)
                              / len(recent)) if recent else 0.0
        perf.recentReturnPct = sum(recent)
        if len(recent) > 1:
            rStd = float(np.std(recent))
            perf.recentSharpe = float(np.mean(recent) / (rStd + 1e-9))
        else:
            perf.recentSharpe = perf.sharpeRatio

        grossProfit = sum(r for r in returns if r > 0) or 0.0001
        grossLoss = abs(sum(r for r in returns if r < 0)) or 0.0001
        perf.profitFactor = grossProfit / grossLoss

        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        perf.maxDrawdownPct = (float(np.max(drawdowns))
                               if len(drawdowns) > 0 else 0.0)

        consLoss = consWin = maxConsLoss = maxConsWin = 0
        for r in returns:
            if r <= 0:
                consLoss += 1; consWin = 0
            else:
                consWin += 1; consLoss = 0
            maxConsLoss = max(maxConsLoss, consLoss)
            maxConsWin = max(maxConsWin, consWin)
        perf.consecutiveLosses = maxConsLoss
        perf.consecutiveWins = maxConsWin

        return perf

    # =================================================================
    # Internal — Reporting
    # =================================================================

    def _printReplaySummary(
        self,
        result: DynamicReplayResult,
        allSlotStrings: List[str],
    ):
        """Print a summary table after the full replay."""
        print(f"\n    {'=' * 80}")
        print(f"    DYNAMIC REPLAY SUMMARY  ({result.method})")
        print(f"    {'=' * 80}")

        nCheckpoints = len(result.allocationHistory) - 1  # minus initial

        print(f"    Checkpoints evaluated    : {nCheckpoints}")
        print(f"    Total P/L               : ${result.totalPnL:,.2f} "
              f"({result.totalReturnPct:+.2f}%)")

        header = (
            f"    {'Slot':<14} {'Trades':>7} {'PnL':>11} "
            f"{'Score':>7} {'WR':>6} {'Sharpe':>7} "
            f"{'Init%':>7} {'Final%':>7} {'Status':<10}"
        )
        print(f"\n{header}")
        print(f"    {'─' * 76}")

        initAllocs = result.allocationHistory[0]['allocations']
        finalAllocs = result.finalAllocations

        for slotStr in sorted(allSlotStrings):
            perf = result.finalPerformances.get(slotStr)
            pnl = result.slotPnL.get(slotStr, 0.0)
            initA = initAllocs.get(slotStr, 0.0)
            finalA = finalAllocs.get(slotStr, 0.0)

            trades = perf.totalTrades if perf else 0
            score = perf.ruleScore if perf else 0.0
            wr = perf.recentWinRate if perf else 0.0
            sharpe = perf.recentSharpe if perf else 0.0

            if finalA < 0.001:
                status = 'GHOST'
            elif abs(finalA - initA) < 0.005:
                status = '\u2014'
            elif finalA > initA:
                status = f'+{(finalA - initA)*100:.1f}%'
            else:
                status = f'{(finalA - initA)*100:.1f}%'

            print(
                f"    {slotStr:<14} {trades:>7d} ${pnl:>10,.2f} "
                f"{score:>+6.3f} {wr:>5.0f}% {sharpe:>+6.2f} "
                f"{initA*100:>6.1f}% {finalA*100:>6.1f}% {status:<10}"
            )

        # Per-stock totals
        print(f"\n    Per-stock totals:")
        stockPnL: Dict[str, float] = {}
        stockFinalA: Dict[str, float] = {}
        for slotStr in allSlotStrings:
            sym = slotStr.split('/')[0]
            stockPnL[sym] = stockPnL.get(sym, 0.0) + result.slotPnL.get(slotStr, 0.0)
            stockFinalA[sym] = stockFinalA.get(sym, 0.0) + finalAllocs.get(slotStr, 0.0)

        for sym in sorted(stockPnL):
            print(f"      {sym}: ${stockPnL[sym]:,.2f}  "
                  f"(alloc: {stockFinalA[sym]*100:.1f}%)")

        print()
