"""
TradingDecider  —  Reconciles Stock-Level and Portfolio-Level ML Signals
=========================================================================

When the per-stock ML model says "BUY AAPL" but the portfolio model says
"SELL AAPL for hedging", which one should we listen to?

This module resolves conflicting signals using a confidence-weighted
approach that considers:

  1. Each stock ML model's signal + confidence
  2. The portfolio ML model's allocation adjustments + risk regime
  3. The current hedge action (risk-off reduces position sizing)
  4. A configurable weight that controls how much influence the
     portfolio-level model has vs. per-stock models

Output:  a ``FinalTradeDecision`` per stock with a single resolved
         signal (BUY / SELL / HOLD), final confidence, and position size.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── Local imports (design-time only) ───────────────────────────────────
from StockMLModel import StockPrediction, TradingSignal, MarketRegime
from PortfolioMLModel import (
    PortfolioSignal, AllocationAdjustment, HedgeAction, RiskRegime
)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FinalTradeDecision:
    """The reconciled trade decision for a single stock."""
    symbol: str
    signal: TradingSignal = TradingSignal.HOLD
    confidence: float = 0.0          # 0-1  (blended confidence)
    positionSize: float = 0.0        # fraction of that stock's allocation (0-1)
    stockSignal: TradingSignal = TradingSignal.HOLD    # raw stock-model signal
    stockConfidence: float = 0.0
    portfolioSignal: TradingSignal = TradingSignal.HOLD  # what portfolio model wanted
    portfolioConfidence: float = 0.0
    sentimentSignal: TradingSignal = TradingSignal.HOLD  # from SentimentAnalyzer
    sentimentConfidence: float = 0.0
    patternSignal: TradingSignal = TradingSignal.HOLD    # from GA pattern matching
    patternConfidence: float = 0.0
    reason: str = ''                 # human-readable explanation

    def signalBreakdown(self) -> Dict:
        """Return a JSON-serialisable dict of all component signals for logging."""
        return {
            'ml': {'signal': self.stockSignal.value, 'confidence': round(self.stockConfidence, 3)},
            'portfolio': {'signal': self.portfolioSignal.value, 'confidence': round(self.portfolioConfidence, 3)},
            'sentiment': {'signal': self.sentimentSignal.value, 'confidence': round(self.sentimentConfidence, 3)},
            'pattern': {'signal': self.patternSignal.value, 'confidence': round(self.patternConfidence, 3)},
            'blended': {'signal': self.signal.value, 'confidence': round(self.confidence, 3)},
            'positionSize': round(self.positionSize, 4),
        }


# =====================================================================
# Trading Decider
# =====================================================================

class TradingDecider:
    """
    Blends per-stock ML signals with portfolio-level signals to produce
    one final decision per stock.

    Parameters
    ----------
    portfolioWeight : float
        How much weight the portfolio model gets relative to the stock
        model.  0.0 = ignore portfolio model entirely,
                1.0 = equal weight, 2.0 = portfolio model twice as
                influential as per-stock model.  Default 0.6.
    minConfidence : float
        Minimum blended confidence required to trade. Below this the
        decision is forced to HOLD.  Default 0.25.
    hedgeMultipliers : dict
        Position-size multipliers for each ``HedgeAction`` value.
        E.g. ``REDUCE_25`` → 0.75 means keep only 75 % of normal size.
    """

    def __init__(
        self,
        portfolioWeight: float = 0.30,
        patternWeight: float = 0.80,
        sentimentWeight: float = 0.50,
        minConfidence: float = 0.25,
        hedgeMultipliers: Optional[Dict[HedgeAction, float]] = None,
        useRegimeDetection: bool = True,
        useCorrelationAdjustment: bool = True,
    ):
        self.portfolioWeight = portfolioWeight
        self.patternWeight = patternWeight      # weight for GA pattern signal
        self.sentimentWeight = sentimentWeight  # weight for sentiment ensemble signal
        self.minConfidence = minConfidence
        self.hedgeMultipliers: Dict[HedgeAction, float] = hedgeMultipliers or {
            HedgeAction.NONE:       1.00,
            HedgeAction.REDUCE_10:  0.95,
            HedgeAction.REDUCE_25:  0.85,
            HedgeAction.REDUCE_50:  0.70,
            HedgeAction.FULL_HEDGE: 0.30,
        }
        self.useRegimeDetection = useRegimeDetection
        self.useCorrelationAdjustment = useCorrelationAdjustment
        # Regime-aware overrides: (positionSize multiplier, minConfidence override)
        self._regimeOverrides: Dict[RiskRegime, Tuple[float, float]] = {
            RiskRegime.LOW_RISK:  (1.00, self.minConfidence),
            RiskRegime.NORMAL:    (1.00, self.minConfidence),
            RiskRegime.HIGH_RISK: (0.50, max(self.minConfidence, 0.35)),
            RiskRegime.CRISIS:    (0.15, max(self.minConfidence, 0.55)),
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def decide(
        self,
        stockPredictions: Dict[str, StockPrediction],
        portfolioSignal: Optional[PortfolioSignal],
        currentAllocations: Dict[str, float],
        verbose: bool = False,
    ) -> Dict[str, FinalTradeDecision]:
        """
        Produce one ``FinalTradeDecision`` per symbol.

        Parameters
        ----------
        stockPredictions : dict
            {symbol: StockPrediction} — latest prediction from each
            per-stock ML model.
        portfolioSignal : PortfolioSignal or None
            The most recent portfolio-level signal.  If ``None`` the
            decider falls back to stock-only signals.
        currentAllocations : dict
            {symbol: fraction} — how capital is currently split.
        verbose : bool
            Print the reasoning table.

        Returns
        -------
        dict[str, FinalTradeDecision]
        """
        decisions: Dict[str, FinalTradeDecision] = {}

        portSignals = self._extractPortfolioSignals(portfolioSignal)
        hedgeMult = self._hedgeMultiplier(portfolioSignal)

        # Regime-aware adjustments (reverted when useRegimeDetection=False)
        if self.useRegimeDetection and portfolioSignal:
            regime = portfolioSignal.riskRegime
            regimeSizeMult, regimeMinConf = self._regimeOverrides.get(
                regime, (1.0, self.minConfidence))
        else:
            regimeSizeMult, regimeMinConf = 1.0, self.minConfidence

        for symbol, stockPred in stockPredictions.items():
            decision = self._resolveOneStock(
                symbol=symbol,
                stockPred=stockPred,
                portSignal=portSignals.get(symbol),
                portConfidence=(portfolioSignal.confidence
                                if portfolioSignal else 0.0),
                hedgeMult=hedgeMult,
                allocation=currentAllocations.get(symbol, 0.0),
                regimeSizeMult=regimeSizeMult,
                regimeMinConf=regimeMinConf,
            )
            decisions[symbol] = decision

        # Correlation-aware position sizing: reduce positions when correlated
        # stocks all signal the same direction simultaneously.
        if portfolioSignal and portfolioSignal.avgCrossCorrelation > 0:
            decisions = self._applyCorrelationAdjustment(
                decisions, portfolioSignal.avgCrossCorrelation)

        if verbose:
            self._printDecisions(decisions, portfolioSignal)

        return decisions

    def decideForBacktest(
        self,
        stockPred: StockPrediction,
        symbol: str,
        portfolioSignal: Optional[PortfolioSignal],
        allocation: float,
    ) -> FinalTradeDecision:
        """
        Lightweight version of ``decide()`` for use inside the backtester
        loop — processes a single stock / single candle.
        """
        portSignals = self._extractPortfolioSignals(portfolioSignal)
        hedgeMult = self._hedgeMultiplier(portfolioSignal)

        if self.useRegimeDetection and portfolioSignal:
            regime = portfolioSignal.riskRegime
            regimeSizeMult, regimeMinConf = self._regimeOverrides.get(
                regime, (1.0, self.minConfidence))
        else:
            regimeSizeMult, regimeMinConf = 1.0, self.minConfidence

        return self._resolveOneStock(
            symbol=symbol,
            stockPred=stockPred,
            portSignal=portSignals.get(symbol),
            portConfidence=(portfolioSignal.confidence
                            if portfolioSignal else 0.0),
            hedgeMult=hedgeMult,
            allocation=allocation,
            regimeSizeMult=regimeSizeMult,
            regimeMinConf=regimeMinConf,
        )

    # -----------------------------------------------------------------
    # Internal — resolution logic
    # -----------------------------------------------------------------

    def _resolveOneStock(
        self,
        symbol: str,
        stockPred: StockPrediction,
        portSignal: Optional[Tuple[TradingSignal, float]],
        portConfidence: float,
        hedgeMult: float,
        allocation: float,
        regimeSizeMult: float = 1.0,
        regimeMinConf: Optional[float] = None,
    ) -> FinalTradeDecision:
        """
        Core logic: blend FOUR independent signals using confidence-weighted scoring:
          1. Pattern signal  (from GA-discovered patterns)
          2. ML signal       (from OHLCV/technical indicators)
          3. Portfolio signal (from portfolio-level model)
          4. Sentiment signal (from three-layer SentimentAnalyzer ensemble)

        Scoring model
        -------------
        Each signal direction gets a numeric score:
          BUY  → +1
          SELL → -1
          HOLD →  0

        Weighted score =
            patternScore   × patternConf   × patternWeight
          + mlScore        × mlConf        × 1.0
          + portScore      × portConf      × portfolioWeight
          + sentimentScore × sentimentConf × sentimentWeight

        The magnitude of the weighted score determines the blended
        confidence, and the sign determines BUY vs SELL.

        ADAPTIVE WEIGHTING: When a signal source has zero confidence
        (e.g. no patterns match out-of-sample), its weight is removed
        from the denominator so it does not dilute the remaining signals.
        """
        # --- Pattern contribution (independent signal from GA patterns) ---
        patternScore = self._signalToScore(stockPred.patternSignal)
        patternConf = stockPred.patternConfidence

        # --- Sentiment contribution (three-layer ensemble signal) ---
        sentimentScore = self._signalToScore(stockPred.sentimentSignal)
        sentimentConf  = stockPred.sentimentConfidence

        # --- ML model contribution (trained on OHLCV/technical data) ---
        stockScore = self._signalToScore(stockPred.signal)
        stockConf = stockPred.confidence
        stockWeight = 1.0  # ML model always has weight 1.0

        # --- Portfolio model contribution ---
        if portSignal is not None:
            portSig, portSigConf = portSignal
            portScore = self._signalToScore(portSig)
            # Use the per-symbol confidence from the allocation adjustment,
            # multiplied by the overall portfolio confidence.
            effectivePortConf = portSigConf * portConfidence
        else:
            portSig = TradingSignal.HOLD
            portScore = 0.0
            effectivePortConf = 0.0

        # --- ADAPTIVE weighted blend ---
        # Only include signal sources that are actually contributing (non-zero
        # confidence).  This prevents a permanently-HOLD signal channel
        # from dragging down ML confidence during out-of-sample backtesting.
        numerator = 0.0
        totalWeight = 0.0

        # Pattern channel — only counted when patterns actually trigger
        if patternConf > 0.0:
            numerator += patternScore * patternConf * self.patternWeight
            totalWeight += self.patternWeight

        # ML channel — always present when model is trained
        if stockConf > 0.0:
            numerator += stockScore * stockConf * stockWeight
            totalWeight += stockWeight

        # Portfolio channel — only when portfolio signal is provided
        if effectivePortConf > 0.0:
            numerator += portScore * effectivePortConf * self.portfolioWeight
            totalWeight += self.portfolioWeight

        # Sentiment channel — only when sentiment engine has run
        if sentimentConf > 0.0:
            numerator += sentimentScore * sentimentConf * self.sentimentWeight
            totalWeight += self.sentimentWeight

        if totalWeight > 0:
            blendedScore = numerator / totalWeight
        else:
            blendedScore = 0.0

        # Convert blended score back to signal
        blendedConf = min(1.0, abs(blendedScore))
        _effectiveMinConf = regimeMinConf if regimeMinConf is not None else self.minConfidence
        if blendedScore > 0 and blendedConf >= _effectiveMinConf:
            finalSignal = TradingSignal.BUY
        elif blendedScore < 0 and blendedConf >= _effectiveMinConf:
            finalSignal = TradingSignal.SELL
        else:
            finalSignal = TradingSignal.HOLD
            blendedConf = 0.0

        # --- Position sizing (intelligent: confidence curve × signal agreement × hedge) ---
        if finalSignal == TradingSignal.HOLD:
            posSize = 0.0
        else:
            # --- Agreement factor -------------------------------------------
            # Count how many *active* signal channels agree with the final
            # direction.  Active = confidence > 0 (i.e. actually fired).
            # 4/4 agree → multiplier ~1.00, 1/4 agree → ~0.40.
            finalDir = 1.0 if finalSignal == TradingSignal.BUY else -1.0
            agreeing, active = 0, 0
            for _score, _conf in [
                (patternScore,   patternConf),
                (stockScore,     stockConf),
                (portScore,      effectivePortConf),
                (sentimentScore, sentimentConf),
            ]:
                if _conf > 0.0:
                    active += 1
                    if _score * finalDir > 0:
                        agreeing += 1
            # Soft scaling: lone agreeing signal → 0.40, unanimous → 1.00
            agreementMult = (0.40 + 0.60 * (agreeing / active)) if active > 0 else 1.0

            # --- Confidence curve ------------------------------------------
            # sqrt removes the old 70 %-floor and gives smooth scaling:
            #   conf=0.25 → 0.50 | conf=0.50 → 0.71 | conf=1.0 → 1.00
            confCurve = blendedConf ** 0.5

            basePosSize = confCurve * agreementMult
            posSize = basePosSize * hedgeMult * regimeSizeMult

        # --- Build reason string ---
        reason = self._buildReason(
            symbol, stockPred.signal, stockConf,
            portSig, effectivePortConf, finalSignal, blendedConf,
            hedgeMult, patternConf, stockPred.patternSignal,
            sentimentConf, stockPred.sentimentSignal,
        )

        return FinalTradeDecision(
            symbol=symbol,
            signal=finalSignal,
            confidence=round(blendedConf, 4),
            positionSize=round(posSize, 4),
            stockSignal=stockPred.signal,
            stockConfidence=round(stockConf, 4),
            portfolioSignal=portSig,
            portfolioConfidence=round(effectivePortConf, 4),
            sentimentSignal=stockPred.sentimentSignal,
            sentimentConfidence=round(sentimentConf, 4),
            patternSignal=stockPred.patternSignal,
            patternConfidence=round(patternConf, 4),
            reason=reason,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _applyCorrelationAdjustment(
        self,
        decisions: Dict[str, FinalTradeDecision],
        avgCorrelation: float,
    ) -> Dict[str, FinalTradeDecision]:
        """
        When multiple correlated stocks all signal the same direction,
        reduce each one's positionSize proportionally.

        If 3 stocks all BUY at 0.90 correlation, the effective diversification
        is ~1 bet, not 3.  We scale each position by 1 / sqrt(N_same_direction)
        weighted by the average cross-correlation.
        """
        buys = [s for s, d in decisions.items() if d.signal == TradingSignal.BUY]
        sells = [s for s, d in decisions.items() if d.signal == TradingSignal.SELL]

        for group in [buys, sells]:
            if len(group) <= 1:
                continue
            n = len(group)
            corrFactor = min(1.0, avgCorrelation)
            scale = 1.0 / (1.0 + corrFactor * (n - 1) ** 0.5)
            for sym in group:
                d = decisions[sym]
                object.__setattr__(d, 'positionSize', round(d.positionSize * scale, 4))

        return decisions

    @staticmethod
    def _signalToScore(signal: TradingSignal) -> float:
        if signal == TradingSignal.BUY:
            return 1.0
        if signal == TradingSignal.SELL:
            return -1.0
        return 0.0

    def _hedgeMultiplier(self, portSig: Optional[PortfolioSignal]) -> float:
        if portSig is None:
            return 1.0
        return self.hedgeMultipliers.get(portSig.hedgeAction, 1.0)

    def _extractPortfolioSignals(
        self, portSig: Optional[PortfolioSignal]
    ) -> Dict[str, Tuple[TradingSignal, float]]:
        """
        Convert the portfolio model's allocation adjustments into a
        per-symbol (signal, confidence) tuple.

        Logic:
          - If suggestedAllocation > currentAllocation → BUY signal
          - If suggestedAllocation < currentAllocation → SELL signal
          - Confidence = magnitude of the relative shift
        """
        result: Dict[str, Tuple[TradingSignal, float]] = {}
        if portSig is None:
            return result

        for adj in portSig.allocationAdjustments:
            diff = adj.suggestedAllocation - adj.currentAllocation
            if abs(diff) < 0.005:
                result[adj.symbol] = (TradingSignal.HOLD, 0.0)
            elif diff > 0:
                # Portfolio says increase allocation → BUY
                conf = min(1.0, abs(diff) / 0.10)  # 10 % shift = full confidence
                result[adj.symbol] = (TradingSignal.BUY, conf)
            else:
                # Portfolio says decrease allocation → SELL
                conf = min(1.0, abs(diff) / 0.10)
                result[adj.symbol] = (TradingSignal.SELL, conf)

        return result

    @staticmethod
    def _buildReason(
        symbol: str,
        stockSig: TradingSignal, stockConf: float,
        portSig: TradingSignal, portConf: float,
        finalSig: TradingSignal, finalConf: float,
        hedgeMult: float,
        patternConf: float = 0.0,
        patternSig: TradingSignal = TradingSignal.HOLD,
        sentimentConf: float = 0.0,
        sentimentSig: TradingSignal = TradingSignal.HOLD,
    ) -> str:
        parts = []
        parts.append(f"Pat={patternSig.value}({patternConf:.0%})")
        parts.append(f"ML={stockSig.value}({stockConf:.0%})")
        parts.append(f"Port={portSig.value}({portConf:.0%})")
        parts.append(f"Sent={sentimentSig.value}({sentimentConf:.0%})")

        # Check for conflicts between the four signals
        signals = {patternSig, stockSig, portSig, sentimentSig}
        if TradingSignal.BUY in signals and TradingSignal.SELL in signals:
            parts.append("CONFLICT")

        parts.append(f"→ {finalSig.value}({finalConf:.0%})")
        if hedgeMult < 1.0:
            parts.append(f"hedge={hedgeMult:.0%}")
        return ' | '.join(parts)

    @staticmethod
    def _printDecisions(
        decisions: Dict[str, FinalTradeDecision],
        portfolioSignal: Optional[PortfolioSignal],
    ):
        print(f"\n  {'─' * 70}")
        print("  TRADING DECIDER — Signal Reconciliation")
        print(f"  {'─' * 70}")
        if portfolioSignal:
            print(f"  Risk Regime : {portfolioSignal.riskRegime.value}")
            print(f"  Hedge Action: {portfolioSignal.hedgeAction.value}")
            print(f"  Portfolio Conf: {portfolioSignal.confidence:.0%}")
        print()
        header = f"  {'Symbol':<8} {'StockSig':<10} {'StockConf':>9} " \
                 f"{'PortSig':<10} {'PortConf':>9} " \
                 f"{'SentSig':<10} {'SentConf':>9} " \
                 f"{'FinalSig':<10} {'FinalConf':>9} {'PosSize':>8}  Reason"
        print(header)
        print(f"  {'─' * len(header)}")
        for sym, d in sorted(decisions.items()):
            print(f"  {sym:<8} {d.stockSignal.value:<10} "
                  f"{d.stockConfidence:>8.0%}  "
                  f"{d.portfolioSignal.value:<10} "
                  f"{d.portfolioConfidence:>8.0%}  "
                  f"{d.sentimentSignal.value:<10} "
                  f"{d.sentimentConfidence:>8.0%}  "
                  f"{d.signal.value:<10} "
                  f"{d.confidence:>8.0%} "
                  f"{d.positionSize:>7.0%}  "
                  f"{d.reason}")
        print()
