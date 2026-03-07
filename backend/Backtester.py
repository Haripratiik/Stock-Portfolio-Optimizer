"""
Backtester for Genetic Algorithm Pattern Discovery

This module backtests patterns discovered by the genetic algorithm against historical data.
It simulates trading based on pattern matches and tracks success/failure rates and returns.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GeneticAlgorithm import (
    PatternChromosome, PatternBank, PatternGene, StockDataFetcher, CandleType
)

# Import for ML-based backtesting (with forward reference handling)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from StockMLModel import StockMLModel, TradingSignal
    from TradingDecider import TradingDecider, FinalTradeDecision
    from PortfolioMLModel import PortfolioSignal

from EarningsBlackout import EarningsBlackoutChecker


class BacktestResult:
    """Stores results from a backtest run with compound balance tracking."""
    
    def __init__(self, initialBalance: float = 0.0):
        self.totalTrades = 0
        self.successfulTrades = 0
        self.unsuccessfulTrades = 0
        self.totalReturnPct = 0.0
        self.trades = []  # List of individual trade details
        # Compound balance tracking
        self.initialBalance = initialBalance    # starting capital
        self.finalBalance = initialBalance      # updated after every trade
        
    def addTrade(self, isSuccessful: bool, returnPct: float, timestamp: pd.Timestamp, 
                 patternId: int, entryPrice: float, exitPrice: float,
                 fundAllocation: float = 0.0, dollarPnL: float = None,
                 confidence: float = 0.0, signalBreakdown: Optional[Dict] = None,
                 exitReason: str = '', exitTimestamp: pd.Timestamp = None):
        """Add a trade result to the backtest.
        
        Args:
            dollarPnL: Actual dollar profit/loss for this trade (compound-aware).
                       When provided, it updates the running compound balance.
                       When None, dollar P/L is estimated as fundAllocation * returnPct / 100.
            confidence: The blended confidence that determined position sizing.
            signalBreakdown: Dict of component signals that triggered this trade.
            exitReason: Why the trade was closed (signal_reversal, stop_loss, max_hold).
            exitTimestamp: When the trade was closed (for equity curve plotting).
        """
        self.totalTrades += 1
        signedReturnPct = returnPct if isSuccessful else -abs(returnPct)
        
        if isSuccessful:
            self.successfulTrades += 1
            self.totalReturnPct += abs(returnPct)
        else:
            self.unsuccessfulTrades += 1
            self.totalReturnPct -= abs(returnPct)
        
        if dollarPnL is None:
            dollarPnL = fundAllocation * (signedReturnPct / 100)
        
        self.finalBalance += dollarPnL
        
        self.trades.append({
            'timestamp': timestamp,
            'exitTimestamp': exitTimestamp if exitTimestamp is not None else timestamp,
            'successful': isSuccessful,
            'returnPct': signedReturnPct,
            'patternId': patternId,
            'entryPrice': entryPrice,
            'exitPrice': exitPrice,
            'fundAllocation': fundAllocation,
            'dollarPnL': dollarPnL,
            'balanceAfter': self.finalBalance,
            'confidence': confidence,
            'signalBreakdown': signalBreakdown or {},
            'exitReason': exitReason,
        })
    
    def getSuccessRate(self) -> float:
        """Calculate the percentage of successful trades"""
        if self.totalTrades == 0:
            return 0.0
        return (self.successfulTrades / self.totalTrades) * 100
    
    def getAverageReturn(self) -> float:
        """Calculate average return per trade"""
        if self.totalTrades == 0:
            return 0.0
        return self.totalReturnPct / self.totalTrades
    
    def getCompoundReturnPct(self) -> float:
        """Compound return from initial to final balance (with reinvestment)."""
        if self.initialBalance > 0:
            return (self.finalBalance - self.initialBalance) / self.initialBalance * 100
        return self.totalReturnPct  # fallback for legacy callers

    def getCompoundPnL(self) -> float:
        """Total dollar profit/loss (compound)."""
        return self.finalBalance - self.initialBalance

    def getFinalBalance(self, initialFund: float = None) -> float:
        """Calculate final balance after all trades."""
        if initialFund is None and self.initialBalance > 0:
            return self.finalBalance
        if initialFund is not None:
            return initialFund * (1 + self.totalReturnPct / 100)
        return 0.0
    
    def __str__(self) -> str:
        return f"""Backtest Results:
{'=' * 50}
Total Trades: {self.totalTrades}
Successful: {self.successfulTrades} ({self.getSuccessRate():.2f}%)
Unsuccessful: {self.unsuccessfulTrades} ({100 - self.getSuccessRate():.2f}%)
Total Return: {self.totalReturnPct:.2f}%
Average Return per Trade: {self.getAverageReturn():.2f}%
"""


class Backtester:
    """
    Backtests patterns discovered by the genetic algorithm.
    
    The backtester:
    1. Loads patterns from GA output
    2. Splits initial fund equally across all patterns
    3. For each pattern, searches historical data for matches
    4. When finding the first N-1 genes matching:
       - Enters a trade at current close price
       - Checks if the Nth gene (prediction) matches the next period
       - If matches → successful trade (profit)
       - If doesn't match → unsuccessful trade (loss)
    """
    
    def __init__(self, initialFund: float = 10000.0, forgiveness: float = 0.05,
                 slippageBps: float = 5.0, stopLossPct: float = 5.0,
                 portfolioDrawdownLimit: float = 15.0, useStopLoss: bool = True):
        """
        Initialize the backtester.
        
        Args:
            initialFund: Starting capital for backtesting
            forgiveness: Tolerance for pattern matching (e.g., 0.05 = ±5%)
            slippageBps: Slippage per side in basis points (default 5 = 0.05%).
                         Applied to both entry and exit prices to model
                         realistic execution costs (spread + market impact).
            stopLossPct: Per-trade stop-loss threshold (%). If a position is
                         down more than this from its entry, it is force-closed.
                         Ignored when useStopLoss is False.
            portfolioDrawdownLimit: Portfolio-level circuit breaker (%).
                         If cumulative loss from peak exceeds this, all new
                         trades are blocked for the rest of the backtest.
            useStopLoss: If False, disable per-trade stop-loss (exit only on
                         signal reversal or max hold). Restores pre-stop-loss
                         behaviour that often performed better in backtests.
        """
        self.initialFund = initialFund
        self.forgiveness = forgiveness
        self.slippageBps = slippageBps
        self.stopLossPct = stopLossPct
        self.portfolioDrawdownLimit = portfolioDrawdownLimit
        self.useStopLoss = useStopLoss
        self.fetcher = StockDataFetcher()
    
    def _checkGeneMatchPython(self, gene: PatternGene, row: pd.Series, 
                              baselinePrice: float, avgVolume: float) -> bool:
        """
        Python wrapper for gene matching logic (non-Numba version for easier debugging).
        
        Args:
            gene: Expected pattern gene
            row: Actual data row
            baselinePrice: Baseline price for normalization
            avgVolume: Average volume for normalization
        
        Returns:
            True if the gene matches within forgiveness
        """
        # Calculate actual normalized price change from baseline
        actualNormalizedPct = (row['close'] - baselinePrice) / baselinePrice
        
        # Calculate volume ratio
        volumeRatio = row['volume'] / avgVolume if avgVolume > 0 else 1.0
        
        # Price movement check with forgiveness
        priceDiff = abs(actualNormalizedPct - gene.expectedPriceChangePct)
        if priceDiff > self.forgiveness:
            return False
        
        # Volume check with more forgiveness (±50% tolerance)
        volumeDiff = abs(volumeRatio - gene.expectedVolumePct)
        if volumeDiff > 0.5:
            return False
        
        # Candle type check
        bodyDirection = row['close'] - row['open']
        candleTypeInt = ['BULLISH', 'BEARISH', 'DOJI'].index(gene.candleType.value)
        
        if candleTypeInt == 0 and bodyDirection < -0.0001:  # BULLISH
            return False
        elif candleTypeInt == 1 and bodyDirection > 0.0001:  # BEARISH
            return False
        elif candleTypeInt == 2:  # DOJI
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
    
    def backtest(self, 
                 patterns: List[PatternChromosome],
                 symbol: str,
                 startDate: str,
                 endDate: str,
                 interval: str = '1d',
                 verbose: bool = True) -> BacktestResult:
        """
        Backtest a list of patterns on historical data.
        Prevents overlapping trades - once a time window is used, patterns cannot retrade it.
        
        Args:
            patterns: List of PatternChromosome objects to test
            symbol: Stock symbol (e.g., 'AAPL')
            startDate: Start date for backtesting (YYYY-MM-DD)
            endDate: End date for backtesting (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '30m', etc.)
            verbose: Print progress information
        
        Returns:
            BacktestResult containing trade statistics
        """
        if not patterns:
            if verbose:
                print("No patterns to backtest")
            return BacktestResult()
        
        # Fetch historical data
        if verbose:
            print(f"Fetching data for {symbol} from {startDate} to {endDate} ({interval})...")
        
        df = self.fetcher.fetchData(symbol, interval=interval, start=startDate, end=endDate)
        
        if df is None or len(df) == 0:
            if verbose:
                print(f"Failed to fetch data for {symbol}")
            return BacktestResult()
        
        if verbose:
            print(f"Data fetched: {len(df)} candles")
        
        # Calculate average volume for normalization
        avgVolume = df['volume'].mean()
        
        # Calculate fund allocation per pattern based on accuracies
        totalAccuracy = sum(p.getAccuracy() for p in patterns)
        if totalAccuracy == 0:
            # If no patterns have accuracy, fall back to equal distribution
            patternFunds = {idx: self.initialFund / len(patterns) for idx in range(len(patterns))}
        else:
            # Allocate proportionally to accuracy
            patternFunds = {
                idx: self.initialFund * (pattern.getAccuracy() / totalAccuracy)
                for idx, pattern in enumerate(patterns)
            }
        
        if verbose:
            print("\nFund allocation by pattern accuracy:")
            for idx, pattern in enumerate(patterns):
                accuracy = pattern.getAccuracy()
                fund = patternFunds[idx]
                print(f"  Pattern {idx+1}: Accuracy={accuracy*100:.1f}%, Fund=${fund:,.2f}")
        
        result = BacktestResult()
        
        # --- Collect ALL candidate matches across all patterns ---
        # Patterns are assumed pre-sorted by priority (index 0 = highest rank).
        # One-trade-per-period: if multiple patterns trigger on overlapping
        # windows, only the highest-ranked (lowest patternIdx) executes.

        candidates = []   # (predictionIdx, patternIdx, baselineIdx, occupiedRange)

        # Extract numpy arrays once for vectorized matching
        closesArr = df['close'].values.astype(float)
        opensArr = df['open'].values.astype(float)
        highsArr = df['high'].values.astype(float)
        lowsArr = df['low'].values.astype(float)
        volumesArr = df['volume'].values.astype(float)

        for patternIdx, pattern in enumerate(patterns):
            patternLength = len(pattern.genes)
            historicalLength = patternLength - 1
            historicalGenes = pattern.genes[:-1]

            if verbose:
                print(f"\nScanning pattern {patternIdx + 1}/{len(patterns)} (length={patternLength})...")

            # Pre-extract gene data into arrays for vectorized comparison
            genePriceExp = np.array([g.expectedPriceChangePct for g in historicalGenes])
            geneVolumeExp = np.array([g.expectedVolumePct for g in historicalGenes])
            geneCandleType = np.array([['BULLISH', 'BEARISH', 'DOJI'].index(g.candleType.value) for g in historicalGenes])
            geneMinBody = np.array([g.minBodyPct for g in historicalGenes])

            for i in range(historicalLength, len(df) - 1):
                baselineIdx = i - historicalLength
                baselinePrice = closesArr[baselineIdx]
                if baselinePrice == 0:
                    continue

                # Indices for the historical candles of this pattern window
                indices = np.arange(baselineIdx + 1, baselineIdx + 1 + historicalLength)

                # VECTORIZED price check
                actualPctChg = (closesArr[indices] - baselinePrice) / baselinePrice
                priceDiffs = np.abs(actualPctChg - genePriceExp)
                if np.any(priceDiffs > self.forgiveness):
                    continue

                # VECTORIZED volume check
                volRatios = volumesArr[indices] / (avgVolume + 1e-9)
                volDiffs = np.abs(volRatios - geneVolumeExp)
                if np.any(volDiffs > 0.5):
                    continue

                # VECTORIZED candle type check
                bodyDirs = closesArr[indices] - opensArr[indices]
                bullishFail = (geneCandleType == 0) & (bodyDirs < -0.0001)
                if np.any(bullishFail):
                    continue
                bearishFail = (geneCandleType == 1) & (bodyDirs > 0.0001)
                if np.any(bearishFail):
                    continue
                dojiMask = geneCandleType == 2
                if np.any(dojiMask):
                    tRange = highsArr[indices] - lowsArr[indices]
                    dojiCheck = dojiMask & (tRange > 0) & (np.abs(bodyDirs) > tRange * 0.15)
                    if np.any(dojiCheck):
                        continue

                # VECTORIZED body pct check
                bodyCheckMask = geneMinBody > 0.1
                if np.any(bodyCheckMask):
                    tRange = highsArr[indices] - lowsArr[indices]
                    bodyPcts = np.where(tRange > 0, np.abs(bodyDirs) / tRange, 0)
                    bodyFail = bodyCheckMask & (tRange > 0) & (bodyPcts < geneMinBody)
                    if np.any(bodyFail):
                        continue

                occupiedRange = set(range(baselineIdx, i + 2))
                candidates.append((i + 1, patternIdx, baselineIdx, occupiedRange))

        # Sort by (predictionIdx, patternIdx) so earlier periods first,
        # and within collisions the highest-priority pattern wins.
        candidates.sort(key=lambda c: (c[0], c[1]))

        # Track occupied time windows to prevent overlapping trades
        occupiedIndices = set()
        tradeCountByPattern = {idx: 0 for idx in range(len(patterns))}
        overlapCountByPattern = {idx: 0 for idx in range(len(patterns))}

        for predictionIdx, patternIdx, baselineIdx, occupiedRange in candidates:
            if occupiedRange & occupiedIndices:
                overlapCountByPattern[patternIdx] += 1
                continue

            pattern = patterns[patternIdx]
            predictionGene = pattern.genes[-1]
            baselinePrice = closesArr[baselineIdx]

            # Use numpy arrays directly instead of pd.Series for prediction check
            actualNormalizedPct = (closesArr[predictionIdx] - baselinePrice) / baselinePrice
            priceDiff = abs(actualNormalizedPct - predictionGene.expectedPriceChangePct)
            predictionMatches = priceDiff <= self.forgiveness

            if predictionMatches:
                volumeRatio = volumesArr[predictionIdx] / (avgVolume + 1e-9)
                volumeDiff = abs(volumeRatio - predictionGene.expectedVolumePct)
                if volumeDiff > 0.5:
                    predictionMatches = False

            if predictionMatches:
                bodyDirection = closesArr[predictionIdx] - opensArr[predictionIdx]
                candleTypeInt = ['BULLISH', 'BEARISH', 'DOJI'].index(predictionGene.candleType.value)
                if candleTypeInt == 0 and bodyDirection < -0.0001:
                    predictionMatches = False
                elif candleTypeInt == 1 and bodyDirection > 0.0001:
                    predictionMatches = False
                elif candleTypeInt == 2:
                    totalRange = highsArr[predictionIdx] - lowsArr[predictionIdx]
                    if totalRange > 0 and abs(bodyDirection) > totalRange * 0.15:
                        predictionMatches = False

            if predictionMatches and predictionGene.minBodyPct > 0.1:
                bodyDirection = closesArr[predictionIdx] - opensArr[predictionIdx]
                totalRange = highsArr[predictionIdx] - lowsArr[predictionIdx]
                if totalRange > 0:
                    bodyPct = abs(bodyDirection) / totalRange
                    if bodyPct < predictionGene.minBodyPct:
                        predictionMatches = False

            entryIdx = predictionIdx - 1
            entryPrice = closesArr[entryIdx]
            exitPrice = closesArr[predictionIdx]
            returnPct = ((exitPrice - entryPrice) / entryPrice) * 100

            result.addTrade(
                isSuccessful=predictionMatches,
                returnPct=returnPct,
                timestamp=df.index[entryIdx],
                patternId=patternIdx,
                entryPrice=entryPrice,
                exitPrice=exitPrice,
                fundAllocation=patternFunds[patternIdx],
                exitTimestamp=df.index[predictionIdx],
            )

            occupiedIndices.update(occupiedRange)
            tradeCountByPattern[patternIdx] += 1

        if verbose:
            for patternIdx in range(len(patterns)):
                print(f"  Pattern {patternIdx+1}: {tradeCountByPattern[patternIdx]} trades, "
                      f"{overlapCountByPattern[patternIdx]} overlaps skipped")
        
        return result
    
    def backtestPatternBank(self,
                           patternBank: PatternBank,
                           symbol: str,
                           startDate: str,
                           endDate: str,
                           topN: int = 10,
                           verbose: bool = True) -> Dict[str, BacktestResult]:
        """
        Backtest the best patterns from a PatternBank across different intervals.
        Splits initial fund equally across timeframes (not across all patterns).
        
        Args:
            patternBank: PatternBank containing discovered patterns
            symbol: Stock symbol to backtest
            startDate: Start date for backtesting
            endDate: End date for backtesting
            topN: Number of top patterns to test per interval
            verbose: Print progress information
        
        Returns:
            Dictionary mapping interval to BacktestResult
        """
        results = {}
        
        # Get unique intervals in the pattern bank
        intervals = set()
        for pattern in patternBank.patterns:
            interval = pattern.interval if pattern.interval else '1d'
            intervals.add(interval)
        
        if verbose:
            print(f"\nBacktesting Pattern Bank for {patternBank.symbol}")
            print(f"Intervals found: {intervals}")
            print("=" * 60)
        
        # Split fund equally across timeframes
        fundPerInterval = self.initialFund / len(intervals) if intervals else self.initialFund
        
        for interval in intervals:
            # Get best patterns for this interval
            intervalPatterns = patternBank.getPatternsByInterval(interval)[:topN]
            
            if not intervalPatterns:
                continue
            
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Interval: {interval}")
                print(f"Testing {len(intervalPatterns)} patterns")
                print(f"Fund for this interval: ${fundPerInterval:,.2f}")
                print('=' * 60)
            
            # Create a temporary backtester with this interval's fund
            intervalBacktester = Backtester(
                initialFund=fundPerInterval,
                forgiveness=self.forgiveness
            )
            
            result = intervalBacktester.backtest(
                patterns=intervalPatterns,
                symbol=symbol,
                startDate=startDate,
                endDate=endDate,
                interval=interval,
                verbose=verbose
            )
            
            results[interval] = result
            
            if verbose:
                print(f"\n{result}")
        
        return results
    
    def backtestMLModel(self,
                       mlModel: 'StockMLModel',
                       symbol: str,
                       startDate: str,
                       endDate: str,
                       interval: str = '1d',
                       holdPeriods: int = 5,
                       verbose: bool = True) -> BacktestResult:
        """
        Backtest a trained ML model on historical data.
        
        Unlike pattern-based backtesting, the ML model:
          - Generates BUY/SELL/HOLD signals at every candle
          - Can enter both long (BUY) and short (SELL) positions
          - Dynamically sizes positions based on confidence
          - Holds for a fixed number of periods (or until next signal)
        
        Args:
            mlModel:      Trained StockMLModel instance
            symbol:       Stock ticker
            startDate:    Backtest start date (YYYY-MM-DD)
            endDate:      Backtest end date (YYYY-MM-DD)
            interval:     Data interval ('1d', '1h', '30m')
            holdPeriods:  How many periods to hold each position
            verbose:      Print progress
        
        Returns:
            BacktestResult with ML-based trades
        """
        if not mlModel.isTrained:
            if verbose:
                print("ML model is not trained. Cannot backtest.")
            return BacktestResult()
        
        if verbose:
            print(f"ML Backtesting {symbol} from {startDate} to {endDate} ({interval})...")
        
        # Fetch historical data (need extra history for indicator calculation)
        df = self.fetcher.fetchData(symbol, interval=interval, start=startDate, end=endDate)
        if df is None or len(df) == 0:
            if verbose:
                print(f"Failed to fetch data for {symbol}")
            return BacktestResult()
        
        if verbose:
            print(f"Data fetched: {len(df)} candles")
        
        # Generate predictions for all candles
        predictions = mlModel.predictBatch(df)
        if not predictions:
            if verbose:
                print("ML model produced no predictions")
            return BacktestResult()
        
        if verbose:
            print(f"ML predictions generated: {len(predictions)}")
        
        result = BacktestResult()
        
        # Track open position
        openPosition = None  # {'type': 'LONG'/'SHORT', 'entryIdx': int, 'entryPrice': float, 'size': float, 'confidence': float}
        
        for i, pred in enumerate(predictions):
            # Skip if we don't have enough data ahead
            if i + holdPeriods >= len(df):
                break
            
            # Close existing position if hold period expired
            if openPosition and i >= openPosition['entryIdx'] + holdPeriods:
                exitPrice = df.iloc[i]['close']
                entryPrice = openPosition['entryPrice']
                posSize = openPosition['size']
                
                if openPosition['type'] == 'LONG':
                    returnPct = ((exitPrice - entryPrice) / entryPrice) * 100 * posSize
                else:  # SHORT
                    returnPct = ((entryPrice - exitPrice) / entryPrice) * 100 * posSize
                
                isSuccessful = returnPct > 0
                
                result.addTrade(
                    isSuccessful=isSuccessful,
                    returnPct=abs(returnPct),
                    timestamp=df.index[openPosition['entryIdx']],
                    patternId=-1,
                    entryPrice=entryPrice,
                    exitPrice=exitPrice,
                    fundAllocation=self.initialFund * posSize,
                    exitTimestamp=df.index[i],
                )
                
                openPosition = None
            
            # Open new position if no position open and signal is not HOLD
            if openPosition is None:
                from StockMLModel import TradingSignal
                
                if pred.signal == TradingSignal.BUY and pred.confidence > 0.3:
                    openPosition = {
                        'type': 'LONG',
                        'entryIdx': i,
                        'entryPrice': df.iloc[i]['close'],
                        'size': pred.positionSize,  # 0.3 - 1.0 based on confidence
                        'confidence': pred.confidence
                    }
                
                elif pred.signal == TradingSignal.SELL and pred.confidence > 0.3:
                    openPosition = {
                        'type': 'SHORT',
                        'entryIdx': i,
                        'entryPrice': df.iloc[i]['close'],
                        'size': pred.positionSize,
                        'confidence': pred.confidence
                    }
        
        # Close any remaining open position at the end
        if openPosition:
            exitPrice = df.iloc[len(predictions) - 1]['close']
            entryPrice = openPosition['entryPrice']
            posSize = openPosition['size']
            
            if openPosition['type'] == 'LONG':
                returnPct = ((exitPrice - entryPrice) / entryPrice) * 100 * posSize
            else:
                returnPct = ((entryPrice - exitPrice) / entryPrice) * 100 * posSize
            
            isSuccessful = returnPct > 0
            
            result.addTrade(
                isSuccessful=isSuccessful,
                returnPct=abs(returnPct),
                timestamp=df.index[openPosition['entryIdx']],
                patternId=-1,
                entryPrice=entryPrice,
                exitPrice=exitPrice,
                fundAllocation=self.initialFund * posSize,
                exitTimestamp=df.index[len(predictions) - 1],
            )
        
        if verbose:
            print(f"\nML Backtest completed:")
            print(f"  Total Trades: {result.totalTrades}")
            print(f"  Success Rate: {result.getSuccessRate():.1f}%")
            print(f"  Total Return: {result.totalReturnPct:+.2f}%")
            if result.totalTrades > 0:
                longTrades = sum(1 for t in result.trades if t.get('patternId') == -1)
                print(f"  ML-generated trades: {longTrades}")
        
        return result

    def backtestWithDecider(self,
                            mlModel: 'StockMLModel',
                            symbol: str,
                            startDate: str,
                            endDate: str,
                            tradingDecider: 'TradingDecider',
                            portfolioSignal: Optional['PortfolioSignal'] = None,
                            allocation: float = 1.0,
                            interval: str = '1d',
                            holdPeriods: int = 5,
                            sentimentSeries: Optional['pd.Series'] = None,
                            verbose: bool = True,
                            earningsBlackout: bool = True) -> BacktestResult:
        """
        Backtest an ML model using the TradingDecider to reconcile stock-
        level and portfolio-level signals before entering trades.

        Uses **signal-based position management**: positions are held as
        long as the signal agrees (or is HOLD).  Exits occur only on
        signal reversal (LONG→SELL or SHORT→BUY) or after a max-hold
        safety period.  This keeps the system invested during sustained
        trends rather than creating cash drag with fixed hold periods.

        Also supports **shadow mode**: even at 0% allocation the trades
        are simulated so performance can be tracked. Shadow trades use
        a notional $10,000 fund for P/L calculation.

        Args:
            mlModel:          Trained StockMLModel instance
            symbol:           Stock ticker
            startDate:        Backtest start date (YYYY-MM-DD)
            endDate:          Backtest end date (YYYY-MM-DD)
            tradingDecider:   TradingDecider that blends stock + portfolio signals
            portfolioSignal:  Latest PortfolioSignal (optional)
            allocation:       Stock's capital fraction (0 = shadow mode)
            interval:         Data interval ('1d', '1h', '30m')
            holdPeriods:      How many periods to hold each position
            sentimentSeries:  Optional date-indexed pd.Series of daily sentiment
                              scores in [-1,+1].  When provided, the model's
                              sentiment features are updated before prediction
                              so the backtest captures sentiment signal.
            verbose:          Print progress

        Returns:
            BacktestResult with decider-reconciled trades
        """
        if not mlModel.isTrained:
            if verbose:
                print("ML model is not trained. Cannot backtest.")
            return BacktestResult()

        isShadow = allocation < 0.001
        modeLabel = "SHADOW" if isShadow else "LIVE"

        if verbose:
            print(f"  [{modeLabel}] ML+Decider Backtesting {symbol} "
                  f"from {startDate} to {endDate} ({interval})")

        df = self.fetcher.fetchData(symbol, interval=interval,
                                    start=startDate, end=endDate)
        if df is None or len(df) == 0:
            if verbose:
                print(f"    Failed to fetch data for {symbol}")
            return BacktestResult()

        if verbose:
            print(f"    Data fetched: {len(df)} candles")

        # Update sentiment data on the model so predictions include
        # sentiment features for the backtest period.
        if sentimentSeries is not None and hasattr(mlModel, 'updateSentimentSeries'):
            mlModel.updateSentimentSeries(sentimentSeries)

        predictions = mlModel.predictBatch(df)
        if not predictions:
            if verbose:
                print("    ML model produced no predictions")
            return BacktestResult()

        if verbose:
            print(f"    Predictions generated: {len(predictions)}")

        from StockMLModel import TradingSignal

        # Map each prediction's timestamp to its integer position in df
        # so that price look-ups are aligned with predictions.
        predDfIndices = []
        for pred in predictions:
            ts = pred.timestamp
            if ts in df.index:
                predDfIndices.append(df.index.get_loc(ts))
            else:
                # Fall back to nearest position
                predDfIndices.append(df.index.get_indexer([ts], method='nearest')[0])

        notionalFund = self.initialFund if not isShadow else 10000.0
        result = BacktestResult(initialBalance=notionalFund)
        openPosition = None
        currentBalance = notionalFund  # tracks compound growth (reporting only)
        lastExitDfIdx = -999  # tracks when the last exit happened (for cooldown)
        lastExitWasStopLoss = False

        # Scale slippage by interval — intraday has higher execution friction.
        _intervalMultipliers = {'30m': 3.0, '15m': 4.0, '5m': 5.0,
                                '1h': 2.0, '1d': 1.0, '1wk': 1.0}
        slipMult = _intervalMultipliers.get(interval, 1.0)
        slipFrac = (self.slippageBps * slipMult) / 10_000

        # Scale stop-loss % by interval.
        # For intraday bars (1h, 30m, etc.) the stop % must be tighter because
        # normal intraday moves are much smaller than daily moves.
        # For daily bars we use close-based stops (see below) so the full 5%
        # is appropriate — a stock closing -5% from entry on the same day is
        # a meaningful signal to exit.
        # Targets ~2.5-3x ATR for each interval, calibrated for large-cap
        # tech stocks (AAPL/GOOGL/MSFT).  Too tight → false stop-outs that
        # convert winning trades into realised losses.  Too loose → the stop
        # never fires and doesn't protect capital.
        #
        #   interval | eff. stop | typical ATR | ratio (stop/ATR)
        #   ---------|-----------|-------------|------------------
        #   5m       |   0.75%   |   ~0.25%    |  3.0x
        #   15m      |   1.00%   |   ~0.35%    |  2.9x
        #   30m      |   1.50%   |   ~0.50%    |  3.0x
        #   1h       |   4.50%   |   ~1.00%    |  4.5x  (widened from 3% - tech stocks swing 2-4%/hr)
        #   1d       |   5.00%   |   ~1.80%    |  2.8x  (close-based)
        #   1wk      |  10.00%   |   ~3.50%    |  2.9x
        _stopLossMultipliers = {
            '5m':  0.15,  # 5% × 0.15 = 0.75%
            '15m': 0.20,  # 5% × 0.20 = 1.00%
            '30m': 0.30,  # 5% × 0.30 = 1.50%
            '1h':  0.90,  # 5% × 0.90 = 4.50% (wider for hourly - reduces false stop-outs)
            '1d':  1.00,  # 5% — checked against CLOSE, not intraday low
            '1wk': 2.00,  # 10% for weekly bars
        }
        effectiveStopLossPct = self.stopLossPct * _stopLossMultipliers.get(interval, 1.0)

        # For daily, weekly, AND 1h bars we check the stop against the CLOSING price.
        # A stock can dip mid-bar and close higher — using low/high stops you out
        # on noise that reversed. For 1h: typical tech swings 2–4%/hr; close-based
        # reduces false stop-outs. For 5m/15m/30m we keep low/high (shorter bars).
        _useClosePriceForStop = interval in ('1d', '1wk', '1h')

        # Cooldown: after closing a position, wait before re-entering.
        # Prevents the system from being perpetually invested (unrealistic)
        # and ensures each trade is an independent decision.
        cooldownBars = max(holdPeriods, 3)
        stopLossCooldownBars = int(cooldownBars * 1.5)  # 1.5× (was 2×) — less delay to re-enter

        # Signal-based position management:
        #  - Enter on BUY/SELL signal
        #  - Hold through HOLD signals and same-direction signals
        #  - Exit only on signal REVERSAL or max hold period (safety)
        # This keeps the system invested during sustained trends instead
        # of repeatedly entering/exiting with fixed-hold periods.
        # 
        # POSITION SIZING: Uses the FIXED notional fund (not compounding balance)
        # to prevent exponential growth artifacts from high-frequency trading.
        # P/L is still tracked cumulatively for reporting.
        #
        # EXECUTION MODEL: entries and exits use the NEXT bar's open price
        # (not the decision bar's close) to eliminate look-ahead bias.
        # Slippage is applied on both sides to model realistic execution.
        maxHoldPeriods = max(holdPeriods * 5, 252)  # safety valve (≥1 year)

        # Earnings blackout: pre-load dates once to avoid per-bar fetches
        _earningsChecker = None
        _earningsDates: List = []
        if earningsBlackout:
            try:
                _earningsChecker = EarningsBlackoutChecker()
                _earningsDates = _earningsChecker.loadEarningsDates(symbol)
            except Exception:
                _earningsDates = []

        # Portfolio circuit breaker state
        peakBalance = notionalFund
        circuitBroken = False

        for i, pred in enumerate(predictions):
            dfIdx = predDfIndices[i]

            if dfIdx + 2 >= len(df):
                break

            # Portfolio circuit breaker: stop all NEW trades if drawdown
            # from peak exceeds the limit.  Existing positions can still exit.
            peakBalance = max(peakBalance, currentBalance)
            drawdownPct = ((peakBalance - currentBalance) / peakBalance * 100
                           if peakBalance > 0 else 0.0)
            if drawdownPct >= self.portfolioDrawdownLimit:
                if not circuitBroken:
                    circuitBroken = True
                    if verbose:
                        print(f"    ⚠ CIRCUIT BREAKER: {drawdownPct:.1f}% drawdown "
                              f"from peak (limit {self.portfolioDrawdownLimit}%) — "
                              f"no new trades")

            decision = tradingDecider.decideForBacktest(
                stockPred=pred,
                symbol=symbol,
                portfolioSignal=portfolioSignal,
                allocation=allocation,
            )

            # ── Manage existing position ──
            if openPosition is not None:
                holdDuration = dfIdx - openPosition['entryDfIdx']
                shouldExit = False
                stopLossTriggered = False

                # STOP-LOSS CHECK
                # Daily/weekly bars: use the CLOSING price.
                #   A stock can drop -10% intraday and close +5% — the intraday
                #   low is irrelevant for end-of-day stops.  Only exit if the
                #   position is still underwater at the bar's close.
                # Intraday bars (1h, 30m, etc.): use LOW/HIGH.
                #   Each bar represents a discrete trading window; if the price
                #   touched the stop within that window, you would have been filled.
                entryPrice = openPosition['entryPrice']
                if _useClosePriceForStop:
                    checkPrice = float(df.iloc[dfIdx]['close'])
                else:
                    checkPrice = (float(df.iloc[dfIdx]['low'])
                                  if openPosition['type'] == 'LONG'
                                  else float(df.iloc[dfIdx]['high']))

                if openPosition['type'] == 'LONG':
                    lossPct = (entryPrice - checkPrice) / entryPrice * 100
                else:
                    lossPct = (checkPrice - entryPrice) / entryPrice * 100

                if self.useStopLoss and lossPct >= effectiveStopLossPct:
                    shouldExit = True
                    stopLossTriggered = True

                # Minimum hold period (does NOT override stop-loss)
                pastMinHold = holdDuration >= holdPeriods

                if not shouldExit and pastMinHold:
                    if (openPosition['type'] == 'LONG'
                            and decision.signal == TradingSignal.SELL
                            and decision.confidence > 0.0):
                        shouldExit = True
                    elif (openPosition['type'] == 'SHORT'
                          and decision.signal == TradingSignal.BUY
                          and decision.confidence > 0.0):
                        shouldExit = True
                if holdDuration >= maxHoldPeriods:
                    shouldExit = True

                if shouldExit:
                    entryPrice = openPosition['entryPrice']
                    amountDeployed = openPosition['amountDeployed']

                    if stopLossTriggered:
                        # Stop-loss exit price — depends on the stop mechanism used:
                        #
                        # DAILY/WEEKLY (close-based stop):
                        #   The decision is made at the CLOSE of the stop bar.
                        #   We exit at the NEXT bar's open (same as a normal signal
                        #   exit), since that's when orders execute.  The close
                        #   IS the "confirmed" stop price; no gap assumption needed.
                        #
                        # INTRADAY (low/high-based stop):
                        #   If the bar OPENS past the stop level (gap), fill at open.
                        #   If hit during the bar, fill at the theoretical stop price.
                        if _useClosePriceForStop:
                            # Exit at next bar's open (same as signal-based exit)
                            rawExit = df.iloc[dfIdx + 1]['open']
                            if openPosition['type'] == 'LONG':
                                exitPrice = rawExit * (1 - slipFrac)
                            else:
                                exitPrice = rawExit * (1 + slipFrac)
                        else:
                            barOpen = float(df.iloc[dfIdx]['open'])
                            if openPosition['type'] == 'LONG':
                                stopPrice = entryPrice * (1 - effectiveStopLossPct / 100)
                                if barOpen <= stopPrice:
                                    exitPrice = barOpen * (1 - slipFrac)
                                else:
                                    exitPrice = stopPrice * (1 - slipFrac)
                            else:
                                stopPrice = entryPrice * (1 + effectiveStopLossPct / 100)
                                if barOpen >= stopPrice:
                                    exitPrice = barOpen * (1 + slipFrac)
                                else:
                                    exitPrice = stopPrice * (1 + slipFrac)
                    else:
                        rawExit = df.iloc[dfIdx + 1]['open']
                        if openPosition['type'] == 'LONG':
                            exitPrice = rawExit * (1 - slipFrac)
                        else:
                            exitPrice = rawExit * (1 + slipFrac)

                    if openPosition['type'] == 'LONG':
                        priceReturnPct = ((exitPrice - entryPrice) / entryPrice) * 100
                        dollarPnL = amountDeployed * (exitPrice - entryPrice) / entryPrice
                    else:
                        priceReturnPct = ((entryPrice - exitPrice) / entryPrice) * 100
                        dollarPnL = amountDeployed * (entryPrice - exitPrice) / entryPrice

                    currentBalance += dollarPnL

                    isSuccessful = dollarPnL > 0
                    _exitReason = ('stop_loss' if stopLossTriggered
                                   else 'max_hold' if holdDuration >= maxHoldPeriods
                                   else 'signal_reversal')
                    if stopLossTriggered and not _useClosePriceForStop:
                        _exitDfIdx = dfIdx
                    else:
                        _exitDfIdx = min(dfIdx + 1, len(df) - 1)
                    result.addTrade(
                        isSuccessful=isSuccessful,
                        returnPct=abs(priceReturnPct),
                        timestamp=df.index[openPosition['entryDfIdx']],
                        patternId=-2,
                        entryPrice=entryPrice,
                        exitPrice=exitPrice,
                        fundAllocation=amountDeployed,
                        dollarPnL=dollarPnL,
                        confidence=openPosition.get('confidence', 0.0),
                        signalBreakdown=openPosition.get('signalBreakdown'),
                        exitReason=_exitReason,
                        exitTimestamp=df.index[_exitDfIdx],
                    )
                    lastExitDfIdx = dfIdx
                    lastExitWasStopLoss = stopLossTriggered
                    openPosition = None

            # ── Open new position if flat, cooldown elapsed, and no circuit break ──
            # Earnings proximity: adjust position size and minimum confidence.
            # No hard block — earnings moves are opportunities if the model
            # has high conviction.  Weak signals near earnings are filtered out.
            _earningsSizeMult = 1.0
            _earningsConfBoost = 0.0
            _inBlackout = False  # never blocked; kept for legacy compatibility
            if _earningsChecker and openPosition is None:
                barDate = df.index[dfIdx]
                _earningsSizeMult, _earningsConfBoost = _earningsChecker.earningsProximity(
                    symbol, barDate, earningsDatesList=_earningsDates or None)

            # Minimum confidence gate (hedge fund practice: allow smaller positions at lower conf).
            # 0.18 base allows more signals; allocator + regime logic handle risk.
            _entryMinConf = max(0.18, _earningsConfBoost)

            _activeCooldown = stopLossCooldownBars if lastExitWasStopLoss else cooldownBars
            if (openPosition is None
                    and (dfIdx - lastExitDfIdx) >= _activeCooldown
                    and not circuitBroken):
                if (decision.signal == TradingSignal.BUY
                        and decision.confidence >= _entryMinConf
                        and decision.positionSize > 0.03):
                    rawEntry = df.iloc[dfIdx + 1]['open']
                    entryPrice = rawEntry * (1 + slipFrac)
                    effSize = decision.positionSize * _earningsSizeMult
                    amountDeployed = notionalFund * effSize
                    openPosition = {
                        'type': 'LONG',
                        'entryDfIdx': dfIdx + 1,
                        'entryPrice': entryPrice,
                        'size': effSize,
                        'amountDeployed': amountDeployed,
                        'confidence': decision.confidence,
                        'signalBreakdown': decision.signalBreakdown(),
                    }
                elif (decision.signal == TradingSignal.SELL
                      and decision.confidence >= _entryMinConf
                      and decision.positionSize > 0.05):
                    rawEntry = df.iloc[dfIdx + 1]['open']
                    entryPrice = rawEntry * (1 - slipFrac)
                    effSize = decision.positionSize * _earningsSizeMult
                    amountDeployed = notionalFund * effSize
                    openPosition = {
                        'type': 'SHORT',
                        'entryDfIdx': dfIdx + 1,
                        'entryPrice': entryPrice,
                        'size': effSize,
                        'amountDeployed': amountDeployed,
                        'confidence': decision.confidence,
                        'signalBreakdown': decision.signalBreakdown(),
                    }

        # Close remaining open position at end of backtest
        # (use last bar's close as best available — no "next" bar exists)
        if openPosition:
            exitDfIdx = min(predDfIndices[-1], len(df) - 1)
            rawExit = df.iloc[exitDfIdx]['close']
            entryPrice = openPosition['entryPrice']
            amountDeployed = openPosition['amountDeployed']

            if openPosition['type'] == 'LONG':
                exitPrice = rawExit * (1 - slipFrac)
                priceReturnPct = ((exitPrice - entryPrice) / entryPrice) * 100
                dollarPnL = amountDeployed * (exitPrice - entryPrice) / entryPrice
            else:
                exitPrice = rawExit * (1 + slipFrac)
                priceReturnPct = ((entryPrice - exitPrice) / entryPrice) * 100
                dollarPnL = amountDeployed * (entryPrice - exitPrice) / entryPrice

            currentBalance += dollarPnL
            isSuccessful = dollarPnL > 0
            result.addTrade(
                isSuccessful=isSuccessful,
                returnPct=abs(priceReturnPct),
                timestamp=df.index[openPosition['entryDfIdx']],
                patternId=-2,
                entryPrice=entryPrice,
                exitPrice=exitPrice,
                fundAllocation=amountDeployed,
                dollarPnL=dollarPnL,
                confidence=openPosition.get('confidence', 0.0),
                signalBreakdown=openPosition.get('signalBreakdown'),
                exitReason='end_of_backtest',
                exitTimestamp=df.index[exitDfIdx],
            )

        if verbose:
            compoundRet = result.getCompoundReturnPct()
            print(f"    [{modeLabel}] Backtest complete — "
                  f"{result.totalTrades} trades, "
                  f"WR {result.getSuccessRate():.1f}%, "
                  f"Ret {compoundRet:+.2f}%"
                  f" (${result.initialBalance:,.0f} → ${result.finalBalance:,.0f})")

        return result

    def getDetailedTradeHistory(self, result: BacktestResult) -> pd.DataFrame:
        """
        Convert trade results to a pandas DataFrame for analysis.
        
        Args:
            result: BacktestResult object
        
        Returns:
            DataFrame with one row per trade
        """
        if not result.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(result.trades)
        df['cumulativeReturn'] = df['returnPct'].cumsum()
        return df


# Example usage
if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithmPatternFinder
    
    print("=" * 70)
    print("GENETIC ALGORITHM PATTERN BACKTESTER")
    print("=" * 70)
    
    print("\n🔍 Discovering pattern bank for AAPL (3 runs per config)...")
    print("(This may take a few minutes...)")
    
    ga = GeneticAlgorithmPatternFinder(
        populationSize=150,
        generations=50,
        forgivenessPct=0.05,
        earlyStopGenerations=10
    )
    
    patternBank = ga.discoverPatternBank(
        symbol='AAPL',
        intervals=None,  # Use defaults: 1d, 1h, 30m
        patternLengths=[3, 5, 7],  # Test a subset of lengths
        numRunsPerConfig=3,  # 3 runs per config for robustness
        verbose=False
    )
    
    print(f"\n✅ Pattern bank created with {len(patternBank.patterns)} patterns")
    
    # Backtest the pattern bank
    print("\n💰 Backtesting pattern bank...")
    
    # Use recent dates for backtesting (last 2 years to today)
    from datetime import datetime, timedelta
    today = datetime.now()
    startDate = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years ago
    endDate = today.strftime('%Y-%m-%d')  # Today
    
    print(f"   Backtest period: {startDate} to {endDate}")
    
    backtester = Backtester(initialFund=10000.0, forgiveness=0.05)
    results = backtester.backtestPatternBank(
        patternBank=patternBank,
        symbol='AAPL',
        startDate=startDate,
        endDate=endDate,
        topN=5,
        verbose=True
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY OF ALL INTERVALS")
    print("=" * 70)
    
    totalTrades = 0
    totalReturn = 0.0
    
    for interval, result in results.items():
        totalTrades += result.totalTrades
        totalReturn += result.totalReturnPct
        
        print(f"\n{interval}:")
        print(f"  Trades: {result.totalTrades}")
        print(f"  Success Rate: {result.getSuccessRate():.2f}%")
        print(f"  Average Return: {result.getAverageReturn():.2f}%")
        print(f"  Total Return: {result.totalReturnPct:.2f}%")
    
    print(f"\n{'=' * 70}")
    print(f"Overall Total Trades: {totalTrades}")
    print(f"Overall Total Return: {totalReturn:.2f}%")
    if totalTrades > 0:
        print(f"Overall Average Return: {totalReturn / totalTrades:.2f}%")
    print(f"{'=' * 70}")
