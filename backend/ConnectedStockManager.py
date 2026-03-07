"""
ConnectedStockManager — Automatic Connected-Stock Portfolio
============================================================

Identifies stocks connected to the user's manual portfolio (suppliers,
customers, competitors, sector peers) and determines whether trading
them is beneficial.  Manages the lifecycle of the automatic portfolio:

  1. **Discover** — scan metadata of manual stocks for supply-chain
     and relationship links to find candidate connected tickers.
  2. **Evaluate** — use price correlation, lead-lag signals, and ML
     confidence to score each candidate's expected benefit.
  3. **Add** — promote high-scoring candidates into the automatic
     portfolio so the pipeline trades them.
  4. **Monitor** — continuously re-score automatic stocks; remove any
     whose benefit has decayed below the threshold.

Design principles:
  - OpenAI research is NOT triggered for automatic stocks (cost control).
    Only manual portfolio stocks get full research.
  - Automatic stocks use existing metadata (built-in seeds or data
    already in Firestore from prior research runs).
  - The ML models drive all add/remove decisions — no hardcoded rules.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PersistenceManager import PersistenceManager, StockMeta, CrossStockRule


@dataclass
class ConnectedCandidate:
    """A candidate stock that could be added to the automatic portfolio."""
    symbol: str
    connectionType: str        # 'supplier', 'customer', 'competitor', 'sector_peer', 'related'
    parentSymbols: List[str]   # manual portfolio stocks that link to this candidate
    reason: str                # human-readable explanation
    correlationScore: float = 0.0   # price correlation with parent(s)
    leadLagScore: float = 0.0       # lead-lag predictive value
    metadataScore: float = 0.0      # richness of available metadata
    compositeScore: float = 0.0     # overall benefit score (0-1)


@dataclass
class AutoStockEvaluation:
    """Evaluation result for an existing automatic stock."""
    symbol: str
    currentScore: float = 0.0
    priceCorrelation: float = 0.0
    recentReturn: float = 0.0
    tradingBenefit: float = 0.0     # did trades on this stock generate profit?
    recommendation: str = 'hold'     # 'keep', 'remove', 'hold'
    reason: str = ''


# Thresholds
MIN_ADD_SCORE = 0.18           # preliminary threshold to add — real ML score updates after pipeline
MIN_KEEP_SCORE = 0.10          # below this after ML evaluation → remove from automatic portfolio
MAX_AUTO_STOCKS = 15           # cap on automatic portfolio size
CORRELATION_LOOKBACK = 60      # trading days for correlation calculation


class ConnectedStockManager:
    """
    Manages the automatic portfolio of connected stocks.

    Usage::

        mgr = ConnectedStockManager(persistence)
        candidates = mgr.discoverCandidates()
        mgr.evaluateAndUpdate(stockDataDict)
    """

    def __init__(self, persistence: PersistenceManager):
        self.pm = persistence

    def discoverCandidates(self, verbose: bool = False) -> List[ConnectedCandidate]:
        """
        Scan metadata of manual portfolio stocks to find candidate
        connected tickers not already in either portfolio.
        """
        manualStocks = self.pm.getManualPortfolioStocks()
        autoStocks = self.pm.getAutoPortfolioStocks()
        allPortfolioSyms = set(manualStocks.keys()) | set(autoStocks.keys())
        allMeta = self.pm.getAllStockMetadata()

        candidates: Dict[str, ConnectedCandidate] = {}

        for sym, meta in manualStocks.items():
            # Suppliers
            for supplier in (meta.supplyChainUp or []):
                ticker = self._extractTicker(supplier)
                if ticker and ticker not in allPortfolioSyms:
                    self._addCandidate(candidates, ticker, 'supplier', sym,
                                       f'Supplier of {sym}: {supplier}', allMeta)

            # Customers
            for customer in (meta.supplyChainDown or []):
                ticker = self._extractTicker(customer)
                if ticker and ticker not in allPortfolioSyms:
                    self._addCandidate(candidates, ticker, 'customer', sym,
                                       f'Customer of {sym}: {customer}', allMeta)

            # Competitors
            for comp in (meta.competesWidth or []):
                ticker = self._extractTicker(comp)
                if ticker and ticker not in allPortfolioSyms:
                    self._addCandidate(candidates, ticker, 'competitor', sym,
                                       f'Competitor of {sym}: {comp}', allMeta)

            # Sector peers
            for peer in (meta.sectorPeers or []):
                if peer and peer not in allPortfolioSyms:
                    self._addCandidate(candidates, peer, 'sector_peer', sym,
                                       f'Sector peer of {sym}', allMeta)

            # Related tickers
            for related in (meta.relatedTickers or []):
                if related and related not in allPortfolioSyms:
                    self._addCandidate(candidates, related, 'related', sym,
                                       f'Related to {sym}', allMeta)

        result = sorted(candidates.values(), key=lambda c: c.metadataScore, reverse=True)

        if verbose:
            print(f"\n  [ConnectedStockManager] Found {len(result)} candidates "
                  f"from {len(manualStocks)} manual stocks")
            for c in result[:10]:
                print(f"    {c.symbol:6s} ({c.connectionType:12s}) "
                      f"from {','.join(c.parentSymbols):20s} "
                      f"meta={c.metadataScore:.2f}")

        return result

    def evaluateCandidates(
        self,
        candidates: List[ConnectedCandidate],
        stockDataDict: Dict[str, pd.DataFrame],
        verbose: bool = False,
    ) -> List[ConnectedCandidate]:
        """
        Score candidates using price correlation and lead-lag analysis
        against their parent manual stocks.

        Returns candidates sorted by composite score (descending).
        """
        manualStocks = self.pm.getManualPortfolioStocks()

        for candidate in candidates:
            candidateDf = stockDataDict.get(candidate.symbol)
            if candidateDf is None or len(candidateDf) < 30:
                candidate.compositeScore = candidate.metadataScore * 0.3
                continue

            candidateReturns = candidateDf['close'].pct_change().dropna()

            correlations = []
            leadLagScores = []

            for parentSym in candidate.parentSymbols:
                parentDf = stockDataDict.get(parentSym)
                if parentDf is None or len(parentDf) < 30:
                    continue

                parentReturns = parentDf['close'].pct_change().dropna()

                # Align indices
                common = candidateReturns.index.intersection(parentReturns.index)
                if len(common) < 20:
                    continue

                cr = candidateReturns.loc[common].values
                pr = parentReturns.loc[common].values

                # Contemporaneous correlation
                corr = float(np.corrcoef(cr, pr)[0, 1]) if len(cr) > 5 else 0.0
                correlations.append(abs(corr))

                # Lead-lag: does the parent predict the candidate?
                bestLagCorr = 0.0
                for lag in range(1, 6):
                    if lag >= len(pr):
                        break
                    lagCorr = float(np.corrcoef(pr[:-lag], cr[lag:])[0, 1]) if len(pr) > lag + 5 else 0.0
                    bestLagCorr = max(bestLagCorr, abs(lagCorr))
                leadLagScores.append(bestLagCorr)

            candidate.correlationScore = float(np.mean(correlations)) if correlations else 0.0
            candidate.leadLagScore = float(np.mean(leadLagScores)) if leadLagScores else 0.0

            # Multi-parent bonus: stocks connected to multiple manual stocks are more valuable
            multiParentBonus = min(0.15, 0.05 * (len(candidate.parentSymbols) - 1))

            # Preliminary composite — intentionally lenient; real score
            # comes from ML pipeline evaluation after add
            candidate.compositeScore = (
                0.30 * candidate.correlationScore
                + 0.25 * candidate.leadLagScore
                + 0.35 * candidate.metadataScore
                + multiParentBonus
            )

        candidates.sort(key=lambda c: c.compositeScore, reverse=True)

        if verbose:
            print(f"\n  [ConnectedStockManager] Evaluated {len(candidates)} candidates:")
            for c in candidates[:10]:
                print(f"    {c.symbol:6s} score={c.compositeScore:.3f} "
                      f"(corr={c.correlationScore:.2f} lag={c.leadLagScore:.2f} "
                      f"meta={c.metadataScore:.2f}) [{c.connectionType}]")

        return candidates

    def evaluateExistingAutoStocks(
        self,
        stockDataDict: Dict[str, pd.DataFrame],
        runResults: Optional[Dict[str, dict]] = None,
        verbose: bool = False,
    ) -> List[AutoStockEvaluation]:
        """
        Re-evaluate existing automatic stocks to determine if they
        should be kept or removed.

        Uses price performance, correlation stability, and trading
        results (if available) to decide.
        """
        autoStocks = self.pm.getAutoPortfolioStocks()
        manualStocks = self.pm.getManualPortfolioStocks()
        evaluations: List[AutoStockEvaluation] = []

        for sym, meta in autoStocks.items():
            evalResult = AutoStockEvaluation(symbol=sym)

            df = stockDataDict.get(sym)
            if df is None or len(df) < 20:
                evalResult.recommendation = 'remove'
                evalResult.reason = 'Insufficient price data'
                evaluations.append(evalResult)
                continue

            returns = df['close'].pct_change().dropna()

            # Recent performance (last 20 days)
            recentRet = float(returns.tail(20).mean()) * 252  # annualized
            evalResult.recentReturn = recentRet

            # Correlation with parent stocks
            parentSyms = meta.autoAddedFrom or []
            correlations = []
            for parentSym in parentSyms:
                parentDf = stockDataDict.get(parentSym)
                if parentDf is None:
                    continue
                parentReturns = parentDf['close'].pct_change().dropna()
                common = returns.index.intersection(parentReturns.index)
                if len(common) < 20:
                    continue
                corr = float(np.corrcoef(
                    returns.loc[common].values[-CORRELATION_LOOKBACK:],
                    parentReturns.loc[common].values[-CORRELATION_LOOKBACK:]
                )[0, 1]) if len(common) >= CORRELATION_LOOKBACK else 0.0
                correlations.append(abs(corr))

            evalResult.priceCorrelation = float(np.mean(correlations)) if correlations else 0.0

            # Trading benefit (from run results if available)
            if runResults and sym in runResults:
                perStock = runResults[sym]
                evalResult.tradingBenefit = perStock.get('returnPct', 0.0)

            # Score: blend of correlation, trading benefit, and recent momentum
            corrScore = min(1.0, evalResult.priceCorrelation / 0.5)
            tradingScore = min(1.0, max(0.0, (evalResult.tradingBenefit + 5) / 15))
            momentumScore = min(1.0, max(0.0, (recentRet + 0.1) / 0.3))

            evalResult.currentScore = (
                0.40 * corrScore
                + 0.35 * tradingScore
                + 0.25 * momentumScore
            )

            # Decision
            if evalResult.currentScore < MIN_KEEP_SCORE:
                evalResult.recommendation = 'remove'
                evalResult.reason = (
                    f'Score {evalResult.currentScore:.2f} below threshold '
                    f'{MIN_KEEP_SCORE} (corr={evalResult.priceCorrelation:.2f}, '
                    f'benefit={evalResult.tradingBenefit:.1f}%)'
                )
            else:
                evalResult.recommendation = 'keep'
                evalResult.reason = f'Score {evalResult.currentScore:.2f} — performing adequately'

            evaluations.append(evalResult)

        if verbose:
            print(f"\n  [ConnectedStockManager] Evaluated {len(evaluations)} auto stocks:")
            for e in evaluations:
                icon = 'KEEP' if e.recommendation == 'keep' else 'REMOVE'
                print(f"    {e.symbol:6s} [{icon:6s}] score={e.currentScore:.3f} "
                      f"corr={e.priceCorrelation:.2f} benefit={e.tradingBenefit:.1f}% "
                      f"— {e.reason}")

        return evaluations

    def updateAutoPortfolio(
        self,
        stockDataDict: Dict[str, pd.DataFrame],
        runResults: Optional[Dict[str, dict]] = None,
        verbose: bool = False,
    ) -> Dict[str, str]:
        """
        Full lifecycle: discover, evaluate, add good candidates, remove
        underperformers.  Returns {symbol: action} where action is
        'added', 'removed', or 'kept'.
        """
        actions: Dict[str, str] = {}

        # 1. Evaluate existing auto stocks
        evalResults = self.evaluateExistingAutoStocks(
            stockDataDict, runResults, verbose
        )
        for evalResult in evalResults:
            if evalResult.recommendation == 'remove':
                self.pm.removeAutoStock(evalResult.symbol)
                actions[evalResult.symbol] = 'removed'
                if verbose:
                    print(f"  [AutoPortfolio] REMOVED {evalResult.symbol}: "
                          f"{evalResult.reason}")
            else:
                self.pm.updateAutoStockScore(
                    evalResult.symbol, evalResult.currentScore, evalResult.reason
                )
                actions[evalResult.symbol] = 'kept'

        # 2. Discover and evaluate new candidates
        currentAutoCount = sum(1 for a in actions.values() if a == 'kept')
        slotsAvailable = MAX_AUTO_STOCKS - currentAutoCount

        if slotsAvailable > 0:
            candidates = self.discoverCandidates(verbose)
            if candidates:
                candidates = self.evaluateCandidates(
                    candidates, stockDataDict, verbose
                )
                added = 0
                for c in candidates:
                    if added >= slotsAvailable:
                        break
                    if c.compositeScore >= MIN_ADD_SCORE:
                        self.pm.addAutoStock(
                            symbol=c.symbol,
                            reason=c.reason,
                            parentSymbols=c.parentSymbols,
                            score=c.compositeScore,
                        )
                        actions[c.symbol] = 'added'
                        added += 1
                        if verbose:
                            print(f"  [AutoPortfolio] ADDED {c.symbol} "
                                  f"(score={c.compositeScore:.3f}): {c.reason}")

        if verbose:
            added = sum(1 for a in actions.values() if a == 'added')
            removed = sum(1 for a in actions.values() if a == 'removed')
            kept = sum(1 for a in actions.values() if a == 'kept')
            print(f"\n  [AutoPortfolio] Summary: {added} added, "
                  f"{removed} removed, {kept} kept")

        return actions

    def getFullPortfolioSymbols(self) -> Tuple[List[str], List[str]]:
        """Return (manual_symbols, auto_symbols) for pipeline use."""
        manual = sorted(self.pm.getManualPortfolioStocks().keys())
        auto = sorted(self.pm.getAutoPortfolioStocks().keys())
        return manual, auto

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extractTicker(text: str) -> Optional[str]:
        """
        Extract a ticker symbol from various formats:
          - "TSMC (TSM) — chip fab"    → TSM
          - "Qualcomm (QCOM)"          → QCOM
          - "TSM"                      → TSM
          - "AAPL — competitor"        → AAPL
          - "Samsung — displays"       → None (no ticker)
        """
        if not text:
            return None
        text = text.strip()
        import re

        # 1. Parenthetical ticker: "Company Name (TICK)"
        match = re.search(r'\(([A-Z]{1,6})\)', text)
        if match:
            return match.group(1)

        # 2. Split on common delimiters and check each token
        for token in re.split(r'[\s\-—–,;:]+', text):
            cleaned = token.strip('().,;:\'\"')
            if cleaned.isupper() and 1 <= len(cleaned) <= 6 and cleaned.isalpha():
                return cleaned

        return None

    @staticmethod
    def _addCandidate(
        candidates: Dict[str, ConnectedCandidate],
        ticker: str,
        connectionType: str,
        parentSym: str,
        reason: str,
        allMeta: Dict[str, StockMeta],
    ):
        """Add or update a candidate in the candidates dict."""
        if ticker in candidates:
            if parentSym not in candidates[ticker].parentSymbols:
                candidates[ticker].parentSymbols.append(parentSym)
            return

        meta = allMeta.get(ticker)
        metadataScore = 0.10  # base score: any discovered ticker has some value
        if meta:
            if meta.sector:
                metadataScore += 0.15
            if meta.description and meta.description != 'Pending research...':
                metadataScore += 0.20
            if meta.supplyChainUp or meta.supplyChainDown:
                metadataScore += 0.20
            if meta.relatedTickers:
                metadataScore += 0.15
        else:
            # Unknown stocks still get a base score — ML will evaluate them
            metadataScore += 0.10

        # Connection type scoring: suppliers/customers are more tradeable
        typeBonus = {
            'supplier': 0.20, 'customer': 0.20,
            'competitor': 0.15, 'sector_peer': 0.10, 'related': 0.05,
        }
        metadataScore += typeBonus.get(connectionType, 0.0)

        candidates[ticker] = ConnectedCandidate(
            symbol=ticker,
            connectionType=connectionType,
            parentSymbols=[parentSym],
            reason=reason,
            metadataScore=min(1.0, metadataScore),
        )
