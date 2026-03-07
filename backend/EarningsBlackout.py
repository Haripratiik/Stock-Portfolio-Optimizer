"""
Earnings Proximity Module
==========================
Earnings reports are high-volatility events — the stock can surge or
crash significantly.  Rather than blocking trades, the system adjusts
two parameters based on how close the next earnings date is:

  1. Position size multiplier — trade smaller when outcome uncertainty is higher
  2. Min-confidence boost — only enter if the model has strong conviction;
     weak signals near earnings are much less reliable

Position-size tiers (days to next earnings):
  >10 days    → 1.00 (full size, no adjustment)
  6-10 days   → 0.80 (minor caution)
  3-5 days    → 0.50 (trade smaller — pre-earnings tension)
  1-2 days    → 0.30 (very high conviction trades only)
  0 days      → 0.25 (earnings day — small but allowed)
  1-5 AFTER   → 0.60 (post-earnings volatility, momentum often continues)

Min-confidence boosts near earnings:
  >10 days   → +0.00 (use normal threshold)
  3-10 days  → +0.10 (require 10% more confidence than normal)
  0-2 days   → +0.20 (require 20% higher conviction on earnings day)

There is NO hard block.  A correctly predicted earnings move is one of
the best opportunities in systematic trading.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


_lxml_warned = False


class EarningsBlackoutChecker:
    """
    Caches earnings dates per symbol and returns per-bar position-size
    multipliers and confidence boosts based on proximity to earnings.
    """

    def __init__(self, blackoutDays: int = 0):
        """
        Args:
            blackoutDays: Kept for API compatibility but no longer enforces
                          a hard block.  Set to 0 (no block).
        """
        self.blackoutDays = blackoutDays
        self._cache: Dict[str, List[pd.Timestamp]] = {}

    def loadEarningsDates(self, symbol: str) -> List[pd.Timestamp]:
        """Fetch earnings dates from yfinance (cached per symbol)."""
        global _lxml_warned
        if symbol in self._cache:
            return self._cache[symbol]

        dates: List[pd.Timestamp] = []
        try:
            import yfinance as yf
            tk = yf.Ticker(symbol)
            cal = tk.get_earnings_dates(limit=40)
            if cal is not None and not cal.empty:
                dates = [
                    pd.Timestamp(d).normalize().tz_localize(None)
                    for d in cal.index
                ]
        except Exception as e:
            if "lxml" in str(e).lower() and not _lxml_warned:
                _lxml_warned = True
                logger.warning(
                    "[EarningsProximity] Install lxml for earnings dates: pip install lxml"
                )

        self._cache[symbol] = dates
        return dates

    def earningsProximity(
        self,
        symbol: str,
        referenceDate: Optional[pd.Timestamp] = None,
        earningsDatesList: Optional[List[pd.Timestamp]] = None,
    ) -> Tuple[float, float]:
        """
        Return (positionSizeMult, minConfidenceBoost) for `referenceDate`.

        positionSizeMult  : multiply the normal positionSize by this (0.25–1.0)
        minConfidenceBoost: add this to the normal minConfidence threshold (0.0–0.20)

        Args:
            symbol:           Ticker symbol.
            referenceDate:    Bar date to evaluate. Defaults to now (ET).
            earningsDatesList: Pre-loaded list for backtest use.
        """
        if referenceDate is None:
            referenceDate = pd.Timestamp.now(tz='America/New_York')
        refDate = pd.Timestamp(referenceDate).normalize().tz_localize(None)

        dates = (earningsDatesList if earningsDatesList is not None
                 else self.loadEarningsDates(symbol))
        if not dates:
            return 1.0, 0.0

        minDaysAhead: Optional[int] = None
        minDaysAfter: Optional[int] = None
        for ed in dates:
            edNorm = pd.Timestamp(ed).normalize().tz_localize(None)
            delta = (edNorm - refDate).days
            if delta >= 0:
                if minDaysAhead is None or delta < minDaysAhead:
                    minDaysAhead = delta
            else:
                daysSince = abs(delta)
                if minDaysAfter is None or daysSince < minDaysAfter:
                    minDaysAfter = daysSince

        # --- PRE-EARNINGS tiers ------------------------------------------
        if minDaysAhead == 0:
            # Earnings day itself — allowed but very small, high conviction
            return 0.25, 0.20
        if minDaysAhead is not None and minDaysAhead <= 2:
            # Day before earnings — significant catalyst imminent
            return 0.30, 0.20
        if minDaysAhead is not None and minDaysAhead <= 5:
            # 3-5 days out — pre-earnings positioning still volatile
            return 0.50, 0.10
        if minDaysAhead is not None and minDaysAhead <= 10:
            # 6-10 days out — slight caution
            return 0.80, 0.10

        # --- POST-EARNINGS tiers -----------------------------------------
        if minDaysAfter is not None and minDaysAfter <= 5:
            # Post-earnings momentum can be strong — slightly larger size
            # but still elevated vol
            return 0.60, 0.05

        return 1.0, 0.0

    def earningsProximitySizeMult(
        self,
        symbol: str,
        referenceDate: Optional[pd.Timestamp] = None,
        earningsDatesList: Optional[List[pd.Timestamp]] = None,
    ) -> float:
        """Convenience wrapper — returns only the position-size multiplier."""
        mult, _ = self.earningsProximity(symbol, referenceDate, earningsDatesList)
        return mult

    def isBlackedOut(
        self,
        symbol: str,
        referenceDate: Optional[pd.Timestamp] = None,
    ) -> bool:
        """No hard block.  Always returns False (kept for API compatibility)."""
        return False

    def isBlackedOutForBacktest(
        self,
        symbol: str,
        tradeDate: pd.Timestamp,
        earningsDatesList: Optional[List[pd.Timestamp]] = None,
    ) -> bool:
        """No hard block.  Always returns False (kept for API compatibility)."""
        return False

    def getBlackedOutSymbols(
        self,
        symbols: List[str],
        referenceDate: Optional[pd.Timestamp] = None,
    ) -> Set[str]:
        """No hard block.  Always returns empty set."""
        return set()

    def getSizingMultipliers(
        self,
        symbols: List[str],
        referenceDate: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """Return {symbol: positionSizeMultiplier} for all given symbols."""
        return {s: self.earningsProximitySizeMult(s, referenceDate)
                for s in symbols}

    def getProximityParams(
        self,
        symbols: List[str],
        referenceDate: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Return {symbol: (sizeMult, confBoost)} for all given symbols."""
        return {s: self.earningsProximity(s, referenceDate) for s in symbols}
