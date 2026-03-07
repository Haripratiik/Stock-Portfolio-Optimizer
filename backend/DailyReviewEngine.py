"""
DailyReviewEngine — End-of-Day Performance Summary & Self-Improvement
======================================================================

Runs after each daily (1d) trading cycle to:

  1. Aggregate per-slot and portfolio-level metrics for the day
  2. Evaluate each slot's recent track record (win rate, streaks, Sharpe)
  3. Evaluate live pattern accuracy vs MC-expected performance
  4. Auto-adjust allocations conservatively:
     - Phase out (ghost) sustained losers
     - Reduce allocation for borderline underperformers
     - Boost allocation for consistent outperformers
  5. Supersede patterns whose live accuracy has collapsed
  6. Report on ghost slots (are any recovering?)
  7. Persist the review to Firestore and send via AlertManager

All auto-adjustments are logged with reasons so the user can audit or
revert.  The engine never touches core safety parameters (stopLossPct,
minConfidence, etc.) — only slot allocations and pattern active status.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from firebase_admin import firestore

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None

logger = logging.getLogger(__name__)


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()


# ── Thresholds (conservative) ────────────────────────────────────────

PHASE_OUT_CONSEC_LOSSES   = 7
PHASE_OUT_WIN_RATE        = 0.25
PHASE_OUT_MIN_TRADES      = 15

REDUCE_CONSEC_LOSSES      = 4
REDUCE_WIN_RATE           = 0.35
REDUCE_MIN_TRADES         = 10

BOOST_CONSEC_WINS         = 5
BOOST_WIN_RATE            = 0.65
BOOST_MIN_TRADES          = 10
BOOST_FACTOR              = 1.25

PATTERN_SUPERSEDE_MIN_TRIGGERS = 10
PATTERN_SUPERSEDE_MAX_ACCURACY = 0.20

RECENT_TRADE_WINDOW       = 30   # look at last N trades for slot eval
GHOST_NEAR_RESTORE_PCT    = 0.75  # flag ghost as "near restore" at 75% of threshold


class DailyReviewEngine:
    """End-of-day performance reviewer and self-improvement engine."""

    def __init__(
        self,
        db,
        persistence,
        alertManager=None,
        logger: Optional[Callable] = None,
        restoreThreshold: float = 3.0,
        maxSlotAllocation: float = 0.20,
    ):
        self.db = db
        self.persistence = persistence
        self.alertManager = alertManager
        self._log = logger or (lambda msg: None)
        self.restoreThreshold = restoreThreshold
        self.maxSlotAllocation = maxSlotAllocation

    # ══════════════════════════════════════════════════════════════════
    # Public entry point
    # ══════════════════════════════════════════════════════════════════

    def runDailyReview(self) -> Optional[Dict]:
        """Run the full daily review pipeline. Returns the report dict."""
        try:
            self._log("[DailyReview] Starting end-of-day performance review...")

            liveAlloc = self.persistence.loadLiveAllocation()
            if not liveAlloc:
                self._log("[DailyReview] No live allocation found — skipping review.")
                return None

            slotAllocs = liveAlloc.get('slotAllocations', {})
            ghostSlots = set(liveAlloc.get('ghostSlots', []))
            slotPerfs = liveAlloc.get('slotPerformances', {})

            dayMetrics = self._aggregateDayMetrics()
            slotEvals = self._evaluateSlots(slotAllocs, ghostSlots, slotPerfs)
            patternEvals = self._evaluatePatterns(slotAllocs)
            actions, allocChanges = self._autoAdjustAllocations(
                slotEvals, slotAllocs, ghostSlots,
            )
            patternActions = self._autoSupersedeBadPatterns(patternEvals)
            ghostWatch = self._buildGhostWatch(ghostSlots, slotEvals)

            if allocChanges:
                self._applyAllocationChanges(liveAlloc, allocChanges, actions)

            report = self._buildSummaryReport(
                dayMetrics=dayMetrics,
                slotEvals=slotEvals,
                actions=actions,
                patternActions=patternActions,
                ghostWatch=ghostWatch,
                allocChanges=allocChanges,
            )

            today = _now().strftime('%Y-%m-%d')
            self.persistence.saveDailyReview(today, report)

            if self.alertManager:
                self.alertManager.notifyDailyReview(report)

            self._log(
                f"[DailyReview] Complete — "
                f"{len(actions)} actions, "
                f"{len(patternActions)} pattern changes, "
                f"{len(ghostWatch)} ghost slots watched"
            )
            return report

        except Exception as e:
            self._log(f"[DailyReview] Error: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Aggregate today's metrics from trade_cycles
    # ══════════════════════════════════════════════════════════════════

    def _aggregateDayMetrics(self) -> Dict[str, Any]:
        """Pull today's cycle results from trade_cycles collection."""
        today = _now().strftime('%Y-%m-%d')
        metrics: Dict[str, Any] = {
            'date': today,
            'dayPL': 0.0,
            'dayPLPct': 0.0,
            'equity': 0.0,
            'cash': 0.0,
            'numOrders': 0,
            'numFilled': 0,
            'cycles': 0,
        }
        if not self.db:
            return metrics

        try:
            docs = (
                self.db.collection('trade_cycles')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(10)
                .stream()
            )
            for doc in docs:
                d = doc.to_dict()
                ts = d.get('timestamp', '')
                if not ts.startswith(today):
                    continue
                metrics['cycles'] += 1
                metrics['dayPL'] += d.get('dayPL', 0)
                metrics['dayPLPct'] += d.get('dayPLPct', 0)
                metrics['numOrders'] += d.get('numOrders', 0)
                metrics['numFilled'] += d.get('numFilled', 0)
                if d.get('equity', 0) > 0:
                    metrics['equity'] = d['equity']
                if d.get('cash', 0) > 0:
                    metrics['cash'] = d['cash']
        except Exception as e:
            self._log(f"[DailyReview] Metrics aggregation error: {e}")

        return metrics

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Evaluate each slot
    # ══════════════════════════════════════════════════════════════════

    def _evaluateSlots(
        self,
        slotAllocs: Dict[str, float],
        ghostSlots: set,
        slotPerfs: Dict[str, dict],
    ) -> List[Dict[str, Any]]:
        """Compute per-slot metrics from live_slot_trades history."""
        evals: List[Dict[str, Any]] = []

        for slotStr, alloc in slotAllocs.items():
            isGhost = slotStr in ghostSlots or alloc < 0.001
            trades = self.persistence.loadLiveSlotTrades(
                slotStr, limit=RECENT_TRADE_WINDOW,
            )

            totalTrades = len(trades)
            wins = sum(1 for t in trades if t.get('returnPct', 0) > 0)
            winRate = (wins / totalTrades) if totalTrades > 0 else 0.0

            consecLosses = 0
            consecWins = 0
            for t in trades:
                if t.get('returnPct', 0) <= 0:
                    consecLosses += 1
                else:
                    break
            for t in trades:
                if t.get('returnPct', 0) > 0:
                    consecWins += 1
                else:
                    break

            returns = [t.get('returnPct', 0) for t in trades]
            recentReturn = sum(returns)
            avgReturn = (recentReturn / totalTrades) if totalTrades > 0 else 0.0

            recentSharpe = 0.0
            if totalTrades >= 3:
                arr = np.array(returns)
                std = arr.std()
                if std > 0:
                    recentSharpe = float(arr.mean() / std)

            pipePerf = slotPerfs.get(slotStr, {})

            evals.append({
                'slot': slotStr,
                'allocation': round(alloc * 100, 2),
                'isGhost': isGhost,
                'totalTrades': totalTrades,
                'winRate': round(winRate, 4),
                'consecutiveLosses': consecLosses,
                'consecutiveWins': consecWins,
                'recentReturn': round(recentReturn, 4),
                'avgReturn': round(avgReturn, 4),
                'recentSharpe': round(recentSharpe, 4),
                'pipelineRuleScore': pipePerf.get('ruleScore', 0),
            })

        evals.sort(key=lambda e: e['recentReturn'], reverse=True)
        return evals

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Evaluate pattern live accuracy
    # ══════════════════════════════════════════════════════════════════

    def _evaluatePatterns(
        self,
        slotAllocs: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Check live trigger/hit counts for patterns of active slots."""
        patternEvals: List[Dict[str, Any]] = []

        symbols = set()
        for slotStr in slotAllocs:
            parts = slotStr.split('/')
            if parts:
                symbols.add(parts[0])

        for sym in symbols:
            try:
                patterns = self.persistence.loadAllActivePatterns(sym, topN=50)
            except Exception:
                continue

            for p in patterns:
                triggers, hits = self.persistence.getPatternLiveAccuracy(
                    p.patternId,
                )
                if triggers < 3:
                    continue

                liveAccuracy = hits / triggers if triggers > 0 else 0.0
                patternEvals.append({
                    'patternId': p.patternId,
                    'symbol': p.symbol,
                    'interval': p.interval,
                    'mcCompositeScore': p.mcCompositeScore,
                    'liveTriggers': triggers,
                    'liveHits': hits,
                    'liveAccuracy': round(liveAccuracy, 4),
                })

        return patternEvals

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Auto-adjust allocations (conservative)
    # ══════════════════════════════════════════════════════════════════

    def _autoAdjustAllocations(
        self,
        slotEvals: List[Dict],
        slotAllocs: Dict[str, float],
        ghostSlots: set,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Apply conservative allocation adjustments.
        Returns (actions_list, {slot: new_allocation}).
        """
        actions: List[Dict] = []
        changes: Dict[str, float] = {}

        for ev in slotEvals:
            slot = ev['slot']
            currentAlloc = slotAllocs.get(slot, 0)

            if ev['isGhost']:
                continue

            # Phase out: sustained losers
            if (ev['consecutiveLosses'] >= PHASE_OUT_CONSEC_LOSSES or
                    (ev['winRate'] < PHASE_OUT_WIN_RATE
                     and ev['totalTrades'] >= PHASE_OUT_MIN_TRADES)):
                changes[slot] = 0.0
                reason = (
                    f"{ev['consecutiveLosses']} consecutive losses"
                    if ev['consecutiveLosses'] >= PHASE_OUT_CONSEC_LOSSES
                    else f"win rate {ev['winRate']:.0%} over {ev['totalTrades']} trades"
                )
                actions.append({
                    'slot': slot,
                    'action': 'phased_out',
                    'reason': reason,
                    'previousAllocation': round(currentAlloc * 100, 2),
                })
                self._log(
                    f"[DailyReview] PHASE OUT {slot}: {reason}"
                )
                continue

            # Reduce: borderline underperformers
            if (ev['consecutiveLosses'] >= REDUCE_CONSEC_LOSSES or
                    (ev['winRate'] < REDUCE_WIN_RATE
                     and ev['totalTrades'] >= REDUCE_MIN_TRADES)):
                newAlloc = max(0.01, currentAlloc * 0.5)
                if newAlloc < currentAlloc:
                    changes[slot] = newAlloc
                    reason = (
                        f"{ev['consecutiveLosses']} consecutive losses"
                        if ev['consecutiveLosses'] >= REDUCE_CONSEC_LOSSES
                        else f"win rate {ev['winRate']:.0%} over {ev['totalTrades']} trades"
                    )
                    actions.append({
                        'slot': slot,
                        'action': 'reduced',
                        'reason': reason,
                        'previousAllocation': round(currentAlloc * 100, 2),
                        'newAllocation': round(newAlloc * 100, 2),
                    })
                    self._log(
                        f"[DailyReview] REDUCE {slot}: {reason} "
                        f"({currentAlloc*100:.1f}% -> {newAlloc*100:.1f}%)"
                    )
                continue

            # Boost: consistent winners
            if (ev['consecutiveWins'] >= BOOST_CONSEC_WINS
                    and ev['winRate'] >= BOOST_WIN_RATE
                    and ev['totalTrades'] >= BOOST_MIN_TRADES):
                newAlloc = min(
                    self.maxSlotAllocation,
                    currentAlloc * BOOST_FACTOR,
                )
                if newAlloc > currentAlloc:
                    changes[slot] = newAlloc
                    actions.append({
                        'slot': slot,
                        'action': 'boosted',
                        'reason': (
                            f"win rate {ev['winRate']:.0%}, "
                            f"{ev['consecutiveWins']} consecutive wins"
                        ),
                        'previousAllocation': round(currentAlloc * 100, 2),
                        'newAllocation': round(newAlloc * 100, 2),
                    })
                    self._log(
                        f"[DailyReview] BOOST {slot}: "
                        f"({currentAlloc*100:.1f}% -> {newAlloc*100:.1f}%)"
                    )

        return actions, changes

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Auto-supersede bad patterns
    # ══════════════════════════════════════════════════════════════════

    def _autoSupersedeBadPatterns(
        self,
        patternEvals: List[Dict],
    ) -> List[Dict]:
        """Supersede patterns whose live accuracy has collapsed."""
        patternActions: List[Dict] = []

        for pev in patternEvals:
            if (pev['liveTriggers'] >= PATTERN_SUPERSEDE_MIN_TRIGGERS
                    and pev['liveAccuracy'] < PATTERN_SUPERSEDE_MAX_ACCURACY):
                try:
                    self.persistence.supersedePattern(
                        pev['patternId'],
                        reason='daily_review_live_accuracy',
                    )
                    patternActions.append({
                        'patternId': pev['patternId'],
                        'symbol': pev['symbol'],
                        'interval': pev['interval'],
                        'action': 'superseded',
                        'liveAccuracy': pev['liveAccuracy'],
                        'triggers': pev['liveTriggers'],
                    })
                    self._log(
                        f"[DailyReview] SUPERSEDE pattern {pev['patternId']} "
                        f"({pev['symbol']}/{pev['interval']}): "
                        f"live accuracy {pev['liveAccuracy']:.0%} "
                        f"over {pev['liveTriggers']} triggers"
                    )
                except Exception as e:
                    self._log(
                        f"[DailyReview] Failed to supersede {pev['patternId']}: {e}"
                    )

        return patternActions

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Ghost watch
    # ══════════════════════════════════════════════════════════════════

    def _buildGhostWatch(
        self,
        ghostSlots: set,
        slotEvals: List[Dict],
    ) -> List[Dict]:
        """Report on ghost slots — highlight those nearing restore threshold."""
        ghostWatch: List[Dict] = []

        evalLookup = {e['slot']: e for e in slotEvals}

        for slotStr in ghostSlots:
            ev = evalLookup.get(slotStr, {})
            recentReturn = ev.get('recentReturn', 0)
            recentWinRate = ev.get('winRate', 0)
            totalTrades = ev.get('totalTrades', 0)
            nearRestore = (
                recentReturn >= self.restoreThreshold * GHOST_NEAR_RESTORE_PCT
                and totalTrades >= 3
            )

            ghostWatch.append({
                'slot': slotStr,
                'recentReturn': round(recentReturn, 2),
                'recentWinRate': round(recentWinRate, 4),
                'recentTrades': totalTrades,
                'restoreThreshold': self.restoreThreshold,
                'nearRestore': nearRestore,
            })

        ghostWatch.sort(key=lambda g: g['recentReturn'], reverse=True)
        return ghostWatch

    # ══════════════════════════════════════════════════════════════════
    # Apply allocation changes to live_allocation
    # ══════════════════════════════════════════════════════════════════

    def _applyAllocationChanges(
        self,
        liveAlloc: Dict,
        changes: Dict[str, float],
        actions: List[Dict],
    ) -> None:
        """Write updated allocations back to Firestore."""
        slotAllocs = dict(liveAlloc.get('slotAllocations', {}))
        ghostSlots = list(liveAlloc.get('ghostSlots', []))
        stockAllocs = dict(liveAlloc.get('stockAllocations', {}))

        for slot, newAlloc in changes.items():
            slotAllocs[slot] = newAlloc
            if newAlloc < 0.001:
                if slot not in ghostSlots:
                    ghostSlots.append(slot)
            else:
                if slot in ghostSlots:
                    ghostSlots.remove(slot)

        # Rebuild stock-level allocations from slot-level
        newStockAllocs: Dict[str, float] = {}
        for slot, alloc in slotAllocs.items():
            sym = slot.split('/')[0]
            newStockAllocs[sym] = newStockAllocs.get(sym, 0) + alloc
        for sym in stockAllocs:
            if sym not in newStockAllocs:
                newStockAllocs[sym] = 0.0

        self.persistence.saveLiveAllocation(
            slotAllocations=slotAllocs,
            ghostSlots=ghostSlots,
            stockAllocations=newStockAllocs,
            runId=liveAlloc.get('runId', ''),
        )

    # ══════════════════════════════════════════════════════════════════
    # Build summary report
    # ══════════════════════════════════════════════════════════════════

    def _buildSummaryReport(
        self,
        dayMetrics: Dict,
        slotEvals: List[Dict],
        actions: List[Dict],
        patternActions: List[Dict],
        ghostWatch: List[Dict],
        allocChanges: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assemble the structured daily review document."""
        activeSlots = [e for e in slotEvals if not e['isGhost']]
        ghostSlotEvals = [e for e in slotEvals if e['isGhost']]

        # Top / bottom performers (active only)
        topPerformers = activeSlots[:3] if activeSlots else []
        bottomPerformers = activeSlots[-3:] if activeSlots else []

        suggestions = self._generateSuggestions(
            slotEvals, actions, patternActions, ghostWatch,
        )

        return {
            'date': dayMetrics.get('date', _now().strftime('%Y-%m-%d')),
            'timestamp': _now().isoformat(),
            'portfolioPL': round(dayMetrics.get('dayPL', 0), 2),
            'portfolioPLPct': round(dayMetrics.get('dayPLPct', 0), 2),
            'equity': round(dayMetrics.get('equity', 0), 2),
            'cash': round(dayMetrics.get('cash', 0), 2),
            'numOrders': dayMetrics.get('numOrders', 0),
            'numFilled': dayMetrics.get('numFilled', 0),
            'activeSlots': len(activeSlots),
            'ghostSlotCount': len(ghostSlotEvals),
            'slotBreakdown': slotEvals,
            'topPerformers': [
                {'slot': s['slot'], 'recentReturn': s['recentReturn']}
                for s in topPerformers
            ],
            'bottomPerformers': [
                {'slot': s['slot'], 'recentReturn': s['recentReturn']}
                for s in bottomPerformers
            ],
            'ghostWatch': ghostWatch,
            'actionsTaken': actions,
            'patternActions': patternActions,
            'suggestions': suggestions,
        }

    # ══════════════════════════════════════════════════════════════════
    # Generate human-readable suggestions
    # ══════════════════════════════════════════════════════════════════

    def _generateSuggestions(
        self,
        slotEvals: List[Dict],
        actions: List[Dict],
        patternActions: List[Dict],
        ghostWatch: List[Dict],
    ) -> List[str]:
        """Produce actionable suggestions based on the day's data."""
        suggestions: List[str] = []

        # Highlight recovering ghosts
        for g in ghostWatch:
            if g['nearRestore']:
                suggestions.append(
                    f"{g['slot']} is recovering "
                    f"({g['recentReturn']:+.1f}% recent return, "
                    f"threshold {g['restoreThreshold']}%) — may auto-restore soon"
                )

        # Warn about declining active slots (negative Sharpe, still active)
        for ev in slotEvals:
            if (not ev['isGhost']
                    and ev['recentSharpe'] < -0.5
                    and ev['totalTrades'] >= 5):
                suggestions.append(
                    f"{ev['slot']} has negative Sharpe "
                    f"({ev['recentSharpe']:.2f}) — monitor closely"
                )

        # Suggest pipeline re-run if many patterns were superseded
        superseded_symbols = set(p['symbol'] for p in patternActions)
        for sym in superseded_symbols:
            count = sum(1 for p in patternActions if p['symbol'] == sym)
            if count >= 2:
                suggestions.append(
                    f"Multiple patterns superseded for {sym} — "
                    f"consider re-running the pipeline to generate fresh patterns"
                )

        # If no actions and no suggestions, note the all-clear
        if not actions and not patternActions and not suggestions:
            suggestions.append(
                "All slots performing within normal parameters — no changes needed"
            )

        return suggestions
