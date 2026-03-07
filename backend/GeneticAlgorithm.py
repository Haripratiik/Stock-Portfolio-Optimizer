"""
Genetic Algorithm for Pattern Discovery in Stock Data

This module implements a genetic algorithm that discovers profitable patterns
in stock OHLCV (Open, High, Low, Close, Volume) data.

Patterns are represented as sequences of conditions on:
- Price movements (up/down/flat)
- Volume changes (increase/decrease/flat)
- Candlestick properties (body size, wick size, etc.)
- Multi-timeframe features
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime, timedelta
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')


class PriceMove(Enum):
    """Price movement direction"""
    UP = "UP"           # Price increased
    DOWN = "DOWN"       # Price decreased
    FLAT = "FLAT"       # Price relatively unchanged


class VolumeChange(Enum):
    """Volume change level"""
    HIGH = "HIGH"       # Volume significantly above average
    NORMAL = "NORMAL"   # Volume near average
    LOW = "LOW"         # Volume significantly below average


class CandleType(Enum):
    """Candlestick pattern types"""
    BULLISH = "BULLISH"     # Close > Open
    BEARISH = "BEARISH"     # Close < Open
    DOJI = "DOJI"           # Close ≈ Open


# Numba-optimized functions for performance with normalization and forgiveness
@jit(nopython=True, cache=True)
def checkGeneMatch(expectedPriceChangePct: float, expectedVolumePct: float, 
                   candleType: int, minBodyPct: float,
                   actualNormalizedPct: float, volumeRatio: float,
                   openPrice: float, highPrice: float, lowPrice: float, 
                   closePrice: float, forgivenessPct: float) -> bool:
    """
    Numba-optimized gene matching with normalization and forgiveness.
    CandleType encoded as: 0=BULLISH, 1=BEARISH, 2=DOJI
    
    Args:
        expectedPriceChangePct: Expected % change from baseline
        expectedVolumePct: Expected volume as % of average
        candleType: Candle type (0/1/2)
        minBodyPct: Minimum body percentage
        actualNormalizedPct: Actual % change from baseline (normalized)
        volumeRatio: Actual volume / average volume
        forgivenessPct: Tolerance for price matching (e.g., 0.02 = ±2%)
    """
    # Price movement check with forgiveness
    priceDiff = abs(actualNormalizedPct - expectedPriceChangePct)
    if priceDiff > forgivenessPct:
        return False
    
    # Volume check with more forgiveness (±50% tolerance)
    volumeDiff = abs(volumeRatio - expectedVolumePct)
    if volumeDiff > 0.5:
        return False
    
    # Candle type check (more lenient - only check if strong pattern expected)
    bodyDirection = closePrice - openPrice
    if candleType == 0 and bodyDirection < -0.0001:  # BULLISH - allow small red candles
        return False
    elif candleType == 1 and bodyDirection > 0.0001:  # BEARISH - allow small green candles
        return False
    elif candleType == 2:  # DOJI
        totalRange = highPrice - lowPrice
        if totalRange > 0 and abs(bodyDirection) > totalRange * 0.15:  # More lenient
            return False
    
    # Body percentage check - only if significant
    if minBodyPct > 0.1:  # Only enforce if gene requires substantial body
        totalRange = highPrice - lowPrice
        if totalRange > 0:
            bodyPct = abs(bodyDirection) / totalRange
            if bodyPct < minBodyPct:
                return False
    
    return True


@jit(nopython=True, cache=True)
def findMatches(geneData: np.ndarray, patternLength: int,
                openArr: np.ndarray, highArr: np.ndarray, 
                lowArr: np.ndarray, closeArr: np.ndarray, 
                volumeArr: np.ndarray, avgVolume: float,
                forgivenessPct: float) -> np.ndarray:
    """
    Numba-optimized pattern matching with normalization.
    Pattern structure: First (patternLength-1) genes are historical, last gene is prediction.
    geneData is a 2D array where each row is [expectedPriceChangePct, expectedVolumePct, candleType, minBodyPct]
    
    For a pattern of length 5:
    - Genes 0-3 represent the historical pattern (what happened in the past)
    - Gene 4 is the prediction (what we expect to happen next)
    
    Uses pre-allocated array with counter instead of reflected list append
    for better Numba performance.
    """
    dataLen = len(closeArr)
    # Pre-allocate matches array (worst case: every position matches)
    maxMatches = dataLen
    matchBuffer = np.empty(maxMatches, dtype=np.int32)
    matchCount = 0
    
    # We need patternLength points to match pattern, plus 1 more to verify prediction
    historicalLength = patternLength - 1  # Number of historical steps
    
    for i in range(historicalLength, dataLen - 1):  # -1 to have room for prediction
        match = True
        
        # Get baseline price (price BEFORE the pattern window starts)
        baselineIdx = i - historicalLength
        if baselineIdx <= 0:
            continue
        
        baselinePrice = closeArr[baselineIdx]
        if baselinePrice == 0:
            continue
        
        # Check each historical gene in pattern (exclude last gene which is prediction)
        for j in range(historicalLength):
            idx = baselineIdx + j + 1  # +1 because baseline is before the pattern
            
            gene = geneData[j]
            
            # Calculate normalized price change from baseline
            actualNormalizedPct = (closeArr[idx] - baselinePrice) / baselinePrice
            volumeRatio = volumeArr[idx] / avgVolume if avgVolume > 0 else 1.0
            
            isMatch = checkGeneMatch(
                gene[0],  # expectedPriceChangePct
                gene[1],  # expectedVolumePct
                int(gene[2]),  # candleType
                gene[3],  # minBodyPct
                actualNormalizedPct,
                volumeRatio,
                openArr[idx], highArr[idx], lowArr[idx], closeArr[idx],
                forgivenessPct
            )
            
            if not isMatch:
                match = False
                break
        
        if match:
            matchBuffer[matchCount] = i
            matchCount += 1
    
    return matchBuffer[:matchCount].copy()


@jit(nopython=True, cache=True)
def calculateFitness(matches: np.ndarray, predictionGene: np.ndarray,
                     closeArr: np.ndarray, volumeArr: np.ndarray,
                     openArr: np.ndarray, highArr: np.ndarray, lowArr: np.ndarray,
                     avgVolume: float, patternLength: int, forgivenessPct: float) -> Tuple[float, int, int]:
    """
    Numba-optimized fitness calculation with prediction checking.
    predictionGene: [expectedPriceChangePct, expectedVolumePct, candleType, minBodyPct]
    
    The prediction gene represents what we expect to happen AFTER the pattern matches.
    Returns: (fitness, totalMatches, correctPredictions)
    """
    totalMatches = len(matches)
    if totalMatches == 0:
        return (0.0, 0, 0)
    
    correctPredictions = 0
    totalReturn = 0.0
    historicalLength = patternLength - 1
    
    for matchIdx in matches:
        # Get the baseline price for normalization (same as in findMatches)
        baselineIdx = matchIdx - historicalLength
        if baselineIdx <= 0 or matchIdx + 1 >= len(closeArr):
            continue
            
        baselinePrice = closeArr[baselineIdx]
        if baselinePrice == 0:
            continue
        
        # Actual next period data
        nextIdx = matchIdx + 1
        actualNormalizedPct = (closeArr[nextIdx] - baselinePrice) / baselinePrice
        volumeRatio = volumeArr[nextIdx] / avgVolume if avgVolume > 0 else 1.0
        
        # Check if prediction matches with forgiveness
        isPredictionCorrect = checkGeneMatch(
            predictionGene[0],  # expectedPriceChangePct
            predictionGene[1],  # expectedVolumePct
            int(predictionGene[2]),  # candleType
            predictionGene[3],  # minBodyPct
            actualNormalizedPct,
            volumeRatio,
            openArr[nextIdx], highArr[nextIdx], lowArr[nextIdx], closeArr[nextIdx],
            forgivenessPct
        )
        
        if isPredictionCorrect:
            correctPredictions += 1
            # Calculate return from current to next
            returnPct = (closeArr[nextIdx] - closeArr[matchIdx]) / closeArr[matchIdx]
            totalReturn += abs(returnPct)
    
    if len(matches) == 0:
        return (0.0, 0, 0)
    
    accuracy = correctPredictions / len(matches)
    avgReturn = totalReturn / len(matches)
    
    # Logarithmic frequency scaling: rewards frequency with diminishing returns
    # log1p(x) = log(1+x) to handle small values gracefully
    # Scale by 0.5 to keep bonus reasonable: 10 matches ≈ 1.2x, 50 matches ≈ 2.0x, 100 matches ≈ 2.3x
    frequencyBonus = np.log1p(len(matches)) * 0.5
    
    # Fitness heavily weights accuracy, with bonuses for frequency and profitability
    fitness = (accuracy * 100.0) * (1.0 + frequencyBonus) * (1.0 + avgReturn * 100.0)
    return (fitness, len(matches), correctPredictions)


@jit(nopython=True, parallel=True, cache=True)
def evaluateAllChromosomes(populationGenes: np.ndarray, 
                           patternLength: int, openArr: np.ndarray, highArr: np.ndarray,
                           lowArr: np.ndarray, closeArr: np.ndarray, 
                           volumeArr: np.ndarray, avgVolume: float,
                           forgivenessPct: float):
    """
    Parallel evaluation of entire population using Numba.
    Returns tuple of (fitnessValues, totalMatchesArray, correctPredictionsArray).
    Pattern structure: First (patternLength-1) genes are historical, last gene is prediction.
    """
    popSize = len(populationGenes)
    fitnessValues = np.zeros(popSize, dtype=np.float64)
    totalMatchesArray = np.zeros(popSize, dtype=np.int32)
    correctPredictionsArray = np.zeros(popSize, dtype=np.int32)
    
    for i in prange(popSize):
        geneData = populationGenes[i]
        
        # Split into historical genes and prediction gene
        historicalGenes = geneData[:patternLength-1]
        predictionGene = geneData[patternLength-1]
        
        # Find matches using only historical genes
        matches = findMatches(historicalGenes, patternLength, openArr, highArr, 
                             lowArr, closeArr, volumeArr, avgVolume, forgivenessPct)
        
        # Calculate fitness based on how well the prediction gene matches what actually happened
        fitness, totalMatches, correctPreds = calculateFitness(matches, predictionGene, closeArr, volumeArr,
                                           openArr, highArr, lowArr, avgVolume, 
                                           patternLength, forgivenessPct)
        fitnessValues[i] = fitness
        totalMatchesArray[i] = totalMatches
        correctPredictionsArray[i] = correctPreds
    
    return (fitnessValues, totalMatchesArray, correctPredictionsArray)


@dataclass
class PatternGene:
    """
    A single gene in the pattern chromosome with normalized percentage changes.
    All price changes are expressed as % from a baseline (first point in pattern).
    """
    expectedPriceChangePct: float   # Expected % change from baseline (e.g., 0.05 = +5%)
    expectedVolumePct: float        # Expected volume as % of average (e.g., 1.5 = 150% of avg)
    candleType: CandleType          # Bullish/Bearish/Doji
    minBodyPct: float               # Minimum body size as % of total range (0.0-1.0)


@dataclass
class PatternChromosome:
    """
    A chromosome representing a complete pattern.
    Structure: First N-1 genes are historical pattern, last gene is the prediction.
    
    For example, with patternLength=5:
    - genes[0-3]: Historical pattern (what happened in the past 4 periods)
    - genes[4]: Prediction (what we expect to happen next)
    """
    genes: List[PatternGene]
    fitness: float = 0.0
    interval: Optional[str] = None  # Timeframe this pattern was found on (e.g., "1h", "1d")
    symbol: Optional[str] = None    # Stock symbol
    discoveredDate: Optional[datetime] = None  # When pattern was discovered
    totalMatches: int = 0  # How many times the pattern was found in historical data
    correctPredictions: int = 0  # How many times the prediction was correct
    totalPossiblePositions: int = 0  # Total possible positions in data (dataLen - patternLength)
    
    def __len__(self):
        return len(self.genes)
    
    def getAccuracy(self) -> float:
        """Calculate pattern accuracy as correctPredictions / totalMatches"""
        if self.totalMatches == 0:
            return 0.0
        return self.correctPredictions / self.totalMatches
    
    def getOverallAccuracy(self) -> float:
        """Calculate overall accuracy as correctPredictions / totalPossiblePositions"""
        if self.totalPossiblePositions == 0:
            return 0.0
        return self.correctPredictions / self.totalPossiblePositions
    
    def __repr__(self) -> str:
        """String representation showing key metrics"""
        accuracy = self.getAccuracy()
        overallAccuracy = self.getOverallAccuracy()
        return (f"PatternChromosome(length={len(self.genes)}, fitness={self.fitness:.2f}, "
                f"accuracy={accuracy*100:.1f}% ({self.correctPredictions}/{self.totalMatches}), "
                f"overall={overallAccuracy*100:.2f}% ({self.correctPredictions}/{self.totalPossiblePositions}), "
                f"interval={self.interval}, symbol={self.symbol})")

    # --- mutation / cloning helpers used by MC refinement ---

    def deepCopy(self) -> 'PatternChromosome':
        """Return an independent deep copy of this chromosome."""
        newGenes = [
            PatternGene(
                expectedPriceChangePct=g.expectedPriceChangePct,
                expectedVolumePct=g.expectedVolumePct,
                candleType=g.candleType,
                minBodyPct=g.minBodyPct,
            )
            for g in self.genes
        ]
        return PatternChromosome(
            genes=newGenes,
            fitness=self.fitness,
            interval=self.interval,
            symbol=self.symbol,
            discoveredDate=self.discoveredDate,
            totalMatches=self.totalMatches,
            correctPredictions=self.correctPredictions,
            totalPossiblePositions=self.totalPossiblePositions,
        )

    def mutateRandom(self, strength: float = 0.3) -> 'PatternChromosome':
        """
        Create a randomly mutated copy of this pattern.

        Args:
            strength: Controls mutation magnitude (0.0-1.0).
                      0.1 = gentle tweaks, 0.5 = moderate, 1.0 = aggressive.

        The mutation randomly perturbs 1-3 genes:
          - expectedPriceChangePct  ±(strength × 3)%
          - expectedVolumePct      ±(strength × 0.4)
          - candleType             small chance of flip
          - minBodyPct             ±(strength × 0.15)
        """
        child = self.deepCopy()
        numGenesToMutate = random.randint(1, max(1, min(3, len(child.genes))))
        indicesToMutate = random.sample(range(len(child.genes)), numGenesToMutate)

        for idx in indicesToMutate:
            gene = child.genes[idx]

            # Price change
            gene.expectedPriceChangePct += random.gauss(0, strength * 0.03)

            # Volume
            gene.expectedVolumePct += random.gauss(0, strength * 0.4)
            gene.expectedVolumePct = max(0.1, gene.expectedVolumePct)

            # Candle type flip (small probability)
            if random.random() < 0.15 * strength:
                options = [CandleType.BULLISH, CandleType.BEARISH, CandleType.DOJI]
                options.remove(gene.candleType)
                gene.candleType = random.choice(options)

            # Body percentage
            gene.minBodyPct += random.gauss(0, strength * 0.15)
            gene.minBodyPct = max(0.0, min(1.0, gene.minBodyPct))

        # Reset fitness — must be re-evaluated
        child.fitness = 0.0
        child.totalMatches = 0
        child.correctPredictions = 0
        return child

    def mutateGene(self, geneIndex: int, **kwargs) -> 'PatternChromosome':
        """
        Create a copy with one specific gene's attributes replaced.

        Args:
            geneIndex: Which gene to modify (0-based).
            **kwargs:  Any of expectedPriceChangePct, expectedVolumePct,
                       candleType, minBodyPct.

        Example::

            child = parent.mutateGene(2, candleType=CandleType.BEARISH,
                                       expectedVolumePct=1.8)
        """
        child = self.deepCopy()
        gene = child.genes[geneIndex]

        for attr, value in kwargs.items():
            if hasattr(gene, attr):
                setattr(gene, attr, value)

        child.fitness = 0.0
        child.totalMatches = 0
        child.correctPredictions = 0
        return child


@dataclass
class PatternBank:
    """
    A collection of patterns discovered for a stock across different timeframes and lengths.
    Automatically filters out similar/duplicate patterns.
    """
    symbol: str
    patterns: List[PatternChromosome]
    createdDate: datetime = None
    similarityThreshold: float = 0.15  # Patterns within 15% similarity are considered duplicates
    
    def __post_init__(self):
        if self.createdDate is None:
            self.createdDate = datetime.now()
    
    def _calculatePatternSimilarity(self, p1: PatternChromosome, p2: PatternChromosome) -> float:
        """Calculate similarity between two patterns (0.0 = different, 1.0 = identical)"""
        # Must be same length and interval to be considered similar
        if len(p1.genes) != len(p2.genes):
            return 0.0
        if p1.interval != p2.interval:
            return 0.0
        
        # Compare each gene
        totalDiff = 0.0
        numGenes = len(p1.genes)
        
        for g1, g2 in zip(p1.genes, p2.genes):
            # Price difference (most important)
            priceDiff = abs(g1.expectedPriceChangePct - g2.expectedPriceChangePct)
            totalDiff += priceDiff * 2.0  # Weight price heavily
            
            # Volume difference
            volumeDiff = abs(g1.expectedVolumePct - g2.expectedVolumePct)
            totalDiff += volumeDiff * 0.5  # Weight volume less
            
            # Candle type difference (binary)
            if g1.candleType != g2.candleType:
                totalDiff += 0.3
        
        # Normalize by number of genes
        avgDiff = totalDiff / numGenes
        
        # Convert to similarity (lower diff = higher similarity)
        # Using exponential decay: similarity = e^(-diff)
        similarity = np.exp(-avgDiff * 2.0)  # Factor of 2 makes it more sensitive
        
        return similarity
    
    def addPattern(self, pattern: PatternChromosome, maxPatternsPerInterval: int = 5):
        """Add a pattern to the bank, keeping top 5 per timeframe (interval)"""
        if not pattern or pattern.fitness <= 0:
            return False
        
        # Get all patterns for the same interval (timeframe)
        sameInterval = [p for p in self.patterns if p.interval == pattern.interval]
        
        # Check similarity with existing patterns of same interval
        for existingPattern in sameInterval:
            # Only check similarity if same length
            if len(existingPattern.genes) == len(pattern.genes):
                similarity = self._calculatePatternSimilarity(pattern, existingPattern)
                
                # If very similar, keep only the better one
                if similarity > (1.0 - self.similarityThreshold):
                    if pattern.fitness > existingPattern.fitness:
                        # Replace the existing pattern with the better one
                        self.patterns.remove(existingPattern)
                        self.patterns.append(pattern)
                        return True
                    else:
                        # Keep existing pattern, discard new one
                        return False
        
        # Check if we have too many patterns for this interval
        if len(sameInterval) >= maxPatternsPerInterval:
            # Only add if better than the worst one
            worstPattern = min(sameInterval, key=lambda p: p.fitness)
            if pattern.fitness > worstPattern.fitness:
                self.patterns.remove(worstPattern)
                self.patterns.append(pattern)
                return True
            else:
                return False
        
        # No similar pattern found and room available, add it
        self.patterns.append(pattern)
        return True
    
    def getBestPatterns(self, n: int = 10) -> List[PatternChromosome]:
        """Get top N patterns by fitness"""
        return sorted(self.patterns, key=lambda x: x.fitness, reverse=True)[:n]
    
    def getPatternsByInterval(self, interval: str) -> List[PatternChromosome]:
        """Get all patterns for a specific interval"""
        return [p for p in self.patterns if p.interval == interval]
    
    def getPatternsByLength(self, length: int) -> List[PatternChromosome]:
        """Get all patterns of a specific length"""
        return [p for p in self.patterns if len(p) == length]
    
    def summary(self) -> str:
        """Get summary of pattern bank"""
        if not self.patterns:
            return f"PatternBank for {self.symbol}: No patterns"
        
        intervals = set(p.interval for p in self.patterns if p.interval)
        lengths = set(len(p) for p in self.patterns)
        avgFitness = sum(p.fitness for p in self.patterns) / len(self.patterns)
        bestFitness = max(p.fitness for p in self.patterns)
        
        return (f"PatternBank for {self.symbol}:\n"
                f"  Total Patterns: {len(self.patterns)}\n"
                f"  Intervals: {sorted(intervals)}\n"
                f"  Pattern Lengths: {sorted(lengths)}\n"
                f"  Avg Fitness: {avgFitness:.2f}\n"
                f"  Best Fitness: {bestFitness:.2f}")
    
    def saveToFile(self, filepath: str):
        """Save pattern bank to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def loadFromFile(filepath: str) -> 'PatternBank':
        """Load pattern bank from file"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Known delisted or invalid symbols — skip fetch to avoid 404 noise
_SKIP_SYMBOLS = frozenset({'IPG', 'TIKTOK'})

class StockDataFetcher:
    """
    Handles fetching and caching stock data from various sources.
    
    Note: For more frequent updates and minute-level data, consider:
    - Alpha Vantage (premium required for minute data)
    - Polygon.io (excellent for real-time and historical minute data)
    - IEX Cloud
    - Interactive Brokers API
    - Alpaca API (free for US stocks)
    """
    
    def __init__(self, cacheMinutes: int = 60):
        self.cache = {}
        self.cacheDuration = timedelta(minutes=cacheMinutes)
        
        # Yahoo Finance data availability limits (approximate)
        self.intervalLimits = {
            '1m': 7,        # 7 days
            '2m': 60,       # 60 days
            '5m': 60,       # 60 days
            '15m': 60,      # 60 days
            '30m': 60,      # 60 days
            '60m': 730,     # ~2 years
            '1h': 730,      # ~2 years
            '1d': 36500,    # ~100 years (essentially unlimited)
            '1wk': 36500,   # ~100 years
            '1mo': 36500    # ~100 years
        }
    
    def _adjustDateRange(self, interval: str, start: str, end: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Adjust date range to fit within Yahoo Finance limits for the given interval.
        For intraday data, Yahoo is very restrictive and often requires using periods instead.
        
        Args:
            interval: Data interval (e.g., '1h', '1d')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            Adjusted (start, end) tuple, or (None, None) to use period instead
        """
        from datetime import datetime, timedelta
        
        # Get max days for this interval
        maxDays = self.intervalLimits.get(interval, 730)  # Default to 2 years
        
        # Yahoo Finance limits are from TODAY, not from arbitrary dates
        today = datetime.now()
        
        # Parse requested dates
        endDate = datetime.strptime(end, '%Y-%m-%d') if end else today
        startDate = datetime.strptime(start, '%Y-%m-%d')
        
        # Check if end date is in the future - if so, use today
        if endDate > today:
            endDate = today
            end = today.strftime('%Y-%m-%d')
        
        # For ALL intraday data, Yahoo is unreliable with start/end dates
        # BUT we must still respect the requested date range to avoid
        # look-ahead bias. Try start/end first; only fallback if it fails.
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
            # Check if the range is within Yahoo's limits
            daysFromToday = (today - startDate).days
            if daysFromToday > maxDays:
                print(f"⚠ Using period instead of date range for {interval} data (Yahoo limitation)")
                return None, None
            # Try to use start/end dates directly (Yahoo often supports this)
            return start, end
        
        # For daily and above, we can use date ranges more reliably
        # Calculate the earliest allowed start date (from end date backwards)
        earliestStart = endDate - timedelta(days=maxDays)
        
        # If requested start is too old, adjust it
        if startDate < earliestStart:
            adjustedStart = earliestStart
            requestedDays = (endDate - startDate).days
            print(f"⚠ Requested {requestedDays} days of {interval} data, but limit is {maxDays} days.")
            print(f"  Adjusted to: {adjustedStart.strftime('%Y-%m-%d')} to {end}")
            return adjustedStart.strftime('%Y-%m-%d'), end
        
        return start, end
    
    def fetchData(self, 
                   symbol: str, 
                   interval: str = "1h",  # 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
                   period: str = "1mo",   # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
                   start: Optional[str] = None,
                   end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol with automatic date range adjustment.
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m requires period <= 7d for yfinance)
            period: Time period to fetch (used if start/end not provided or invalid)
            start: Start date (YYYY-MM-DD) - will be auto-adjusted if too old
            end: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        if symbol.upper() in _SKIP_SYMBOLS:
            return pd.DataFrame()
        
        originalStart = start
        originalEnd = end
        
        # Auto-adjust date range if provided
        if start and end:
            adjustedStart, adjustedEnd = self._adjustDateRange(interval, start, end)
            # If dates are invalid (returned None), fall back to period
            if adjustedStart is None:
                # Calculate appropriate period based on interval limits
                maxDays = self.intervalLimits.get(interval, 730)
                # Use the maximum available period for this interval
                useStartEnd = False
                period = self._getAdjustedPeriod(interval, f"{maxDays}d")
            else:
                start = adjustedStart
                end = adjustedEnd
                useStartEnd = True
        else:
            useStartEnd = False
        
        cacheKey = f"{symbol}_{interval}_{period}_{start}_{end}_{useStartEnd}"
        
        # Check cache
        if cacheKey in self.cache:
            cachedData, cachedTime = self.cache[cacheKey]
            if datetime.now() - cachedTime < self.cacheDuration:
                return cachedData.copy()
        
        # Fetch new data
        ticker = yf.Ticker(symbol)
        
        try:
            if useStartEnd and start and end:
                df = ticker.history(interval=interval, start=start, end=end)
            else:
                # Use period instead - calculate appropriate period based on interval
                adjustedPeriod = self._getAdjustedPeriod(interval, period)
                df = ticker.history(interval=interval, period=adjustedPeriod)
        except Exception as e:
            err = str(e)
            if "404" in err or "Not Found" in err or "delisted" in err.lower():
                pass  # Skip noisy logs for delisted/invalid symbols
            else:
                print(f"  Could not fetch {symbol}: {err[:80]}")
            return pd.DataFrame()
        
        # Check if data is empty
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Normalize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Cache the data
        self.cache[cacheKey] = (df, datetime.now())
        
        return df.copy()
    
    def _getAdjustedPeriod(self, interval: str, requestedPeriod: str) -> str:
        """
        Get an appropriate period string that fits within interval limits.
        
        Args:
            interval: Data interval
            requestedPeriod: Requested period (e.g., '1mo', '60d', '2y')
        
        Returns:
            Adjusted period string that fits limits
        """
        maxDays = self.intervalLimits.get(interval, 730)
        
        # Map period to days (approximate)
        periodToDays = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, 'ytd': 365, 'max': 36500
        }
        
        # Map days back to best period
        daysToPeriod = [
            (1, '1d'), (5, '5d'), (30, '1mo'), (60, '60d'), 
            (90, '3mo'), (180, '6mo'), (365, '1y'), (730, '2y'), (1825, '5y')
        ]
        
        # Handle "XXXd" or "Ny" format
        if requestedPeriod.endswith('d') and requestedPeriod[:-1].isdigit():
            requestedDays = int(requestedPeriod[:-1])
        elif requestedPeriod.endswith('y') and requestedPeriod[:-1].isdigit():
            requestedDays = int(requestedPeriod[:-1]) * 365
        else:
            requestedDays = periodToDays.get(requestedPeriod, 30)
        
        if requestedDays > maxDays:
            # Find the largest period that fits
            for days, period in reversed(daysToPeriod):
                if days <= maxDays:
                    print(f"⚠ Adjusted period from {requestedPeriod} to {period} for {interval} data")
                    return period
            # If nothing fits, use the max as a string
            return f"{maxDays}d"
        
        # If requestedDays is exactly one of our standard days, return the period
        for days, period in daysToPeriod:
            if days == requestedDays:
                return period
        
        # Otherwise return as days
        return f"{requestedDays}d"
    
    def fetchMultipleTimeframes(self, 
                                  symbol: str, 
                                  timeframes: List[str] = ["1h", "1d"]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        return {tf: self.fetchData(symbol, interval=tf, period="3mo") for tf in timeframes}


class GeneticAlgorithmPatternFinder:
    """
    Genetic Algorithm for discovering patterns in stock data.
    
    The algorithm evolves a population of pattern chromosomes to find
    sequences that reliably predict future price movements.
    """
    
    def __init__(self,
                 populationSize: int = 200,
                 patternLength: int = 5,
                 generations: int = 100,
                 mutationRate: float = 0.15,
                 crossoverRate: float = 0.7,
                 elitismCount: int = 10,
                 forgivenessPct: float = 0.05,
                 earlyStopGenerations: int = 15,
                 minImprovementThreshold: float = 0.01):
        """
        Initialize the genetic algorithm.
        
        Args:
            populationSize: Number of patterns in population
            patternLength: Total pattern length (N-1 historical + 1 prediction)
            generations: Number of generations to evolve
            mutationRate: Probability of mutation
            crossoverRate: Probability of crossover
            elitismCount: Number of best patterns to preserve
            forgivenessPct: Tolerance for pattern matching (e.g., 0.05 = ±5%)
            earlyStopGenerations: Stop if no improvement for this many generations
            minImprovementThreshold: Minimum fitness improvement to count as progress
        """
        self.populationSize = populationSize
        self.patternLength = patternLength
        self.generations = generations
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.elitismCount = elitismCount
        self.forgivenessPct = forgivenessPct
        self.earlyStopGenerations = earlyStopGenerations
        self.minImprovementThreshold = minImprovementThreshold
        
        self.population: List[PatternChromosome] = []
        self.bestPattern: Optional[PatternChromosome] = None
        self.dataFetcher = StockDataFetcher()
        
        # Cache for numpy arrays
        self.dataArrays = None
        self.avgVolume = 0.0
    
    def encodeGene(self, gene: PatternGene) -> np.ndarray:
        """Convert a PatternGene to numpy array for Numba"""
        candleTypeMap = {CandleType.BULLISH: 0, CandleType.BEARISH: 1, CandleType.DOJI: 2}
        
        return np.array([
            gene.expectedPriceChangePct,
            gene.expectedVolumePct,
            candleTypeMap[gene.candleType],
            gene.minBodyPct
        ], dtype=np.float64)
    
    def encodeChromosome(self, chromosome: PatternChromosome) -> np.ndarray:
        """Convert a PatternChromosome to numpy array for Numba"""
        geneData = np.array([self.encodeGene(gene) for gene in chromosome.genes])
        return geneData
    
    def prepareDataArrays(self, data: pd.DataFrame):
        """Convert DataFrame to numpy arrays for Numba"""
        self.dataArrays = {
            'open': data['open'].values.astype(np.float64),
            'high': data['high'].values.astype(np.float64),
            'low': data['low'].values.astype(np.float64),
            'close': data['close'].values.astype(np.float64),
            'volume': data['volume'].values.astype(np.float64)
        }
        self.avgVolume = float(data['volume'].mean())
    
    def createRandomGene(self) -> PatternGene:
        """Create a random pattern gene with normalized percentage values"""
        return PatternGene(
            expectedPriceChangePct=random.uniform(-0.05, 0.05),  # -5% to +5% change (more realistic for hourly)
            expectedVolumePct=random.uniform(0.7, 1.5),  # 70% to 150% of average volume
            candleType=random.choice(list(CandleType)),
            minBodyPct=random.uniform(0.0, 0.3)  # Lower body requirements
        )
    
    def createRandomChromosome(self) -> PatternChromosome:
        """
        Create a random pattern chromosome.
        Structure: (patternLength-1) historical genes + 1 prediction gene
        """
        genes = [self.createRandomGene() for _ in range(self.patternLength)]
        return PatternChromosome(genes=genes)
    
    def initializePopulation(self):
        """Create initial random population"""
        self.population = [self.createRandomChromosome() 
                          for _ in range(self.populationSize)]
    
    def findPatternMatches(self, 
                            pattern: PatternChromosome, 
                            data: pd.DataFrame) -> List[int]:
        """
        Find all occurrences of a pattern in the data using Numba.
        Only uses historical genes (first patternLength-1 genes).
        
        Returns:
            List of indices where pattern matches
        """
        # Prepare data arrays if needed
        if self.dataArrays is None:
            self.prepareDataArrays(data)
        
        # Encode pattern
        geneData = self.encodeChromosome(pattern)
        
        # Use Numba-optimized matching (findMatches handles the historical split)
        matches = findMatches(
            geneData[:self.patternLength-1],  # Only historical genes
            self.patternLength,
            self.dataArrays['open'],
            self.dataArrays['high'],
            self.dataArrays['low'],
            self.dataArrays['close'],
            self.dataArrays['volume'],
            self.avgVolume,
            self.forgivenessPct
        )
        
        return matches.tolist()
    
    def evaluatePopulation(self, data: pd.DataFrame):
        """Evaluate fitness for entire population using parallel Numba"""
        # Prepare data arrays if not already done
        if self.dataArrays is None:
            self.prepareDataArrays(data)
        
        # Encode all chromosomes into numpy arrays
        populationGenes = []
        
        for chromosome in self.population:
            geneData = self.encodeChromosome(chromosome)
            populationGenes.append(geneData)
        
        populationGenesArray = np.array(populationGenes, dtype=np.float64)
        
        # Parallel fitness evaluation using Numba
        fitnessValues, totalMatchesArray, correctPredictionsArray = evaluateAllChromosomes(
            populationGenesArray,
            self.patternLength,
            self.dataArrays['open'],
            self.dataArrays['high'],
            self.dataArrays['low'],
            self.dataArrays['close'],
            self.dataArrays['volume'],
            self.avgVolume,
            self.forgivenessPct
        )
        
        # Assign fitness values and accuracy metrics back to chromosomes
        dataLen = len(self.dataArrays['close'])
        totalPossiblePositions = max(0, dataLen - self.patternLength)
        
        for i, chromosome in enumerate(self.population):
            chromosome.fitness = float(fitnessValues[i])
            chromosome.totalMatches = int(totalMatchesArray[i])
            chromosome.correctPredictions = int(correctPredictionsArray[i])
            chromosome.totalPossiblePositions = totalPossiblePositions
        
        # Update best pattern
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.bestPattern is None or self.population[0].fitness > self.bestPattern.fitness:
            self.bestPattern = self.population[0]
    
    def selectParent(self) -> PatternChromosome:
        """Tournament selection"""
        tournamentSize = 5
        tournament = random.sample(self.population, tournamentSize)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, 
                  parent1: PatternChromosome, 
                  parent2: PatternChromosome) -> Tuple[PatternChromosome, PatternChromosome]:
        """Single-point crossover"""
        if random.random() > self.crossoverRate:
            return parent1, parent2
        
        point = random.randint(1, len(parent1.genes) - 1)
        
        child1Genes = parent1.genes[:point] + parent2.genes[point:]
        child2Genes = parent2.genes[:point] + parent1.genes[point:]
        
        child1 = PatternChromosome(genes=child1Genes)
        child2 = PatternChromosome(genes=child2Genes)
        
        return child1, child2
    
    def mutate(self, chromosome: PatternChromosome):
        """Mutate a chromosome"""
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutationRate:
                # Mutate a random attribute of the gene
                gene = chromosome.genes[i]
                mutationType = random.randint(0, 3)
                
                if mutationType == 0:
                    # Mutate expected price change
                    gene.expectedPriceChangePct += random.uniform(-0.02, 0.02)
                    gene.expectedPriceChangePct = np.clip(gene.expectedPriceChangePct, -0.10, 0.10)
                elif mutationType == 1:
                    # Mutate expected volume
                    gene.expectedVolumePct += random.uniform(-0.2, 0.2)
                    gene.expectedVolumePct = np.clip(gene.expectedVolumePct, 0.5, 2.0)
                elif mutationType == 2:
                    # Mutate candle type
                    gene.candleType = random.choice(list(CandleType))
                else:
                    # Mutate body percentage
                    gene.minBodyPct = random.uniform(0.0, 0.3)
    
    def evolveGeneration(self, data: pd.DataFrame):
        """Create next generation"""
        newPopulation = []
        
        # Elitism: keep best patterns
        newPopulation.extend(self.population[:self.elitismCount])
        
        # Create rest of population through crossover and mutation
        while len(newPopulation) < self.populationSize:
            parent1 = self.selectParent()
            parent2 = self.selectParent()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            newPopulation.append(child1)
            if len(newPopulation) < self.populationSize:
                newPopulation.append(child2)
        
        self.population = newPopulation
    
    def run(self, 
            symbol: str, 
            interval: str = "1h",
            period: str = "3mo",
            start: Optional[str] = None,
            end: Optional[str] = None,
            numRuns: int = 3,
            verbose: bool = True) -> PatternChromosome:
        """
        Run the genetic algorithm to find patterns.
        Runs multiple times and keeps the best result for robustness.
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            period: Time period for data (used if start/end not provided)
            start: Start date (YYYY-MM-DD) for data fetch (overrides period)
            end: End date (YYYY-MM-DD) for data fetch (overrides period)
            numRuns: Number of GA runs (keeps best result)
            verbose: Print progress
        
        Returns:
            Best pattern found across all runs
        """
        # Fetch data once for all runs
        if verbose:
            if start and end:
                print(f"Fetching data for {symbol} ({interval} interval, {start} to {end})...")
            else:
                print(f"Fetching data for {symbol} ({interval} interval, {period})...")
        
        # Fetch using either date range or period
        if start and end:
            data = self.dataFetcher.fetchData(symbol, interval=interval, start=start, end=end)
        else:
            data = self.dataFetcher.fetchData(symbol, interval=interval, period=period)
        
        if len(data) < self.patternLength + 10:
            raise ValueError(f"Insufficient data: need at least {self.patternLength + 10} points")
        
        if verbose:
            print(f"Data loaded: {len(data)} data points")
            if numRuns > 1:
                print(f"\nRunning GA {numRuns} times to find best pattern...")
        
        # Run GA multiple times and keep the best
        bestOverallPattern = None
        bestOverallFitness = -1.0
        
        for runNum in range(numRuns):
            if verbose and numRuns > 1:
                print(f"\n{'='*60}")
                print(f"Run {runNum + 1}/{numRuns}")
                print(f"{'='*60}")
            
            if verbose:
                print(f"Initializing population of {self.populationSize} patterns...")
        
            # Initialize population
            self.initializePopulation()
            
            # Evolution loop with early stopping
            if verbose:
                print(f"Evolving for up to {self.generations} generations...")
                print(f"Early stopping: {self.earlyStopGenerations} gens without improvement >{self.minImprovementThreshold:.3f}")
        
            generationsWithoutImprovement = 0
            lastBestFitness = 0.0
            stoppedEarly = False
            
            for generation in range(self.generations):
                self.evaluatePopulation(data)
                
                currentBestFitness = self.population[0].fitness
                
                # Check for improvement
                improvement = currentBestFitness - lastBestFitness
                if improvement > self.minImprovementThreshold:
                    generationsWithoutImprovement = 0
                    lastBestFitness = currentBestFitness
                else:
                    generationsWithoutImprovement += 1
                
                if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                    avgFitness = sum(c.fitness for c in self.population) / len(self.population)
                    print(f"Generation {generation}: Best={currentBestFitness:.2f}, Avg={avgFitness:.2f}, "
                          f"NoImprove={generationsWithoutImprovement}")
                
                # Early stopping check
                if generationsWithoutImprovement >= self.earlyStopGenerations:
                    if verbose:
                        print(f"\n⚠ Early stopping at generation {generation}: "
                              f"No improvement for {self.earlyStopGenerations} generations")
                    stoppedEarly = True
                    break
                
                if generation < self.generations - 1:
                    self.evolveGeneration(data)
            
            # Final evaluation
            self.evaluatePopulation(data)
            
            # Set metadata on best pattern
            if self.bestPattern:
                self.bestPattern.symbol = symbol
                self.bestPattern.interval = interval
                self.bestPattern.discoveredDate = datetime.now()
            
            # Update best overall
            if self.bestPattern and self.bestPattern.fitness > bestOverallFitness:
                bestOverallPattern = self.bestPattern
                bestOverallFitness = self.bestPattern.fitness
            
            if verbose:
                if not stoppedEarly:
                    print(f"\n✓ Completed all {self.generations} generations")
                print(f"Run {runNum + 1} best fitness: {self.bestPattern.fitness:.2f}")
        
        # Final summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best pattern across {numRuns} runs: fitness={bestOverallFitness:.2f}")
            print(f"{'='*60}")
            print(f"\n=== Best Pattern Found ===")
            self.printPattern(bestOverallPattern, data)
        
        return bestOverallPattern
    
    def discoverPatternBank(self,
                           symbol: str,
                           intervals: List[str] = None,
                           patternLengths: List[int] = None,
                           numRunsPerConfig: int = 3,
                           maxPatternsPerInterval: int = 5,
                           verbose: bool = True) -> PatternBank:
        """
        Discover multiple patterns across different timeframes and pattern lengths.
        Each configuration runs multiple times for robustness.
        
        Args:
            symbol: Stock ticker symbol
            intervals: List of intervals to test (default: ["1d", "1h"])
            patternLengths: List of pattern lengths to test (default: [3, 4, 5, 6])
            numRunsPerConfig: Number of GA runs per configuration (keeps best)
            verbose: Print progress
        
        Returns:
            PatternBank containing all discovered patterns
        """
        if intervals is None:
            # Different intervals with appropriate periods based on Yahoo Finance limits
            intervals = [
                ("1d", "2y"),   # Daily data: 2 years
                ("1h", "1mo"),  # Hourly data: 1 month (Yahoo limit)
                ("30m", "1mo"), # 30-min data: 1 month
            ]
        else:
            # If user provides tuples, handle both 2-element (interval, period) and 3-element (interval, start, end)
            processed = []
            for item in intervals:
                if isinstance(item, (tuple, list)):
                    if len(item) == 2:
                        # (interval, period) format
                        processed.append((item[0], item[1], None, None))
                    elif len(item) == 3:
                        # (interval, start_date, end_date) format
                        processed.append((item[0], None, item[1], item[2]))
                    else:
                        raise ValueError(f"Interval tuple must have 2 or 3 elements, got {len(item)}")
                else:
                    # Just interval string
                    processed.append((item, self._getDefaultPeriod(item), None, None))
            intervals = processed
        
        if patternLengths is None:
            patternLengths = [3, 4, 5, 6, 7, 8, 9, 10]
        
        bank = PatternBank(symbol=symbol, patterns=[])
        
        totalRuns = len(intervals) * len(patternLengths)
        currentRun = 0
        
        if verbose:
            print(f"="*60)
            print(f"Creating Pattern Bank for {symbol}")
            print(f"Intervals: {[i[0] for i in intervals]}")
            print(f"Pattern Lengths: {patternLengths}")
            print(f"Total Runs: {totalRuns}")
            print(f"="*60)
        
        for interval_data in intervals:
            interval, period, start, end = interval_data  # Unpack 4-element tuple
            for patternLength in patternLengths:
                currentRun += 1
                
                periodStr = f"{start} to {end}" if start and end else period
                if verbose:
                    print(f"\n[{currentRun}/{totalRuns}] Running GA: {interval} interval, length={patternLength}, period={periodStr}")
                
                try:
                    # Create new GA instance with this pattern length
                    ga = GeneticAlgorithmPatternFinder(
                        populationSize=self.populationSize,
                        patternLength=patternLength,
                        generations=self.generations,
                        mutationRate=self.mutationRate,
                        crossoverRate=self.crossoverRate,
                        elitismCount=self.elitismCount,
                        forgivenessPct=self.forgivenessPct,
                        earlyStopGenerations=self.earlyStopGenerations,
                        minImprovementThreshold=self.minImprovementThreshold
                    )
                    
                    # Run the algorithm multiple times for robustness
                    # Pass either period or start/end dates depending on what was provided
                    if start and end:
                        bestPattern = ga.run(
                            symbol=symbol,
                            interval=interval,
                            start=start,
                            end=end,
                            numRuns=numRunsPerConfig,
                            verbose=False  # Suppress detailed output
                        )
                    else:
                        bestPattern = ga.run(
                            symbol=symbol,
                            interval=interval,
                            period=period,
                            numRuns=numRunsPerConfig,
                            verbose=False  # Suppress detailed output
                        )
                    
                    if bestPattern and bestPattern.fitness > 0:
                        bank.addPattern(bestPattern, maxPatternsPerInterval=maxPatternsPerInterval)
                        if verbose:
                            print(f"  ✓ Found pattern with fitness {bestPattern.fitness:.2f}")
                    else:
                        if verbose:
                            print(f"  ✗ No viable pattern found")
                
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Error: {str(e)}")
                    continue
        
        if verbose:
            print(f"\n" + "="*60)
            print(bank.summary())
            print("="*60)
        
        return bank
    
    def _getDefaultPeriod(self, interval: str) -> str:
        """Get default period based on interval"""
        if interval in ["1m", "2m", "5m"]:
            return "7d"   # Minute data limited to 7 days
        elif interval in ["15m", "30m", "1h"]:
            return "1mo"  # Intraday data limited to ~60 days
        elif interval == "1d":
            return "2y"   # Daily data can go back years
        elif interval == "1wk":
            return "5y"   # Weekly data
        elif interval == "1mo":
            return "10y"  # Monthly data
        else:
            return "1y"   # Default
    
    def printPattern(self, pattern: PatternChromosome, data: pd.DataFrame):
        """Print pattern details in human-readable format"""
        accuracy = pattern.getAccuracy()
        overallAccuracy = pattern.getOverallAccuracy()
        print(f"Fitness: {pattern.fitness:.2f}")
        print(f"Pattern Accuracy: {accuracy*100:.1f}% ({pattern.correctPredictions}/{pattern.totalMatches} matches)")
        print(f"Overall Accuracy: {overallAccuracy*100:.2f}% ({pattern.correctPredictions}/{pattern.totalPossiblePositions} possible)")
        print(f"Forgiveness: ±{self.forgivenessPct:.1%}")
        print(f"\nPattern Structure: {self.patternLength-1} historical steps + 1 prediction")
        print(f"\nHistorical Pattern (steps 1-{self.patternLength-1}):")
        
        for i, gene in enumerate(pattern.genes[:self.patternLength-1], 1):
            print(f"  Step {i}:")
            print(f"    Expected Price Change: {gene.expectedPriceChangePct:+.2%} from baseline")
            print(f"    Expected Volume: {gene.expectedVolumePct:.1f}x avg")
            print(f"    Candle: {gene.candleType.value}")
            print(f"    Min Body %: {gene.minBodyPct:.1%}")
        
        print(f"\nPrediction (step {self.patternLength}):")
        predGene = pattern.genes[-1]
        print(f"  Expected Price Change: {predGene.expectedPriceChangePct:+.2%} from baseline")
        print(f"  Expected Volume: {predGene.expectedVolumePct:.1f}x avg")
        print(f"  Candle: {predGene.candleType.value}")
        print(f"  Min Body %: {predGene.minBodyPct:.1%}")
        
        # Show matches
        matches = self.findPatternMatches(pattern, data)
        print(f"\nMatches found: {len(matches)}")
        
        if len(matches) > 0:
            print(f"Match dates: {[data.index[m].strftime('%Y-%m-%d %H:%M') for m in matches[:5]]}")


if __name__ == "__main__":
    # Example 1: Discover a single pattern with multi-run optimization
    print("="*60)
    print("EXAMPLE 1: Single Pattern Discovery (Multi-Run)")
    print("="*60)
    
    ga = GeneticAlgorithmPatternFinder(
        populationSize=200,  # Larger population for better exploration
        patternLength=5,  # 4 historical steps + 1 prediction
        generations=100,  # Max generations per run
        mutationRate=0.15,
        crossoverRate=0.7,
        forgivenessPct=0.05,  # ±5% tolerance for pattern matching
        earlyStopGenerations=15,  # More patient stopping
        minImprovementThreshold=0.01  # Minimum 0.01 fitness improvement
    )
    
    bestPattern = ga.run(
        symbol="AAPL",
        interval="1d",
        period="1y",  # 1 year of daily data
        numRuns=3,  # Run 3 times, keep best
        verbose=True
    )
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Pattern Bank Creation (Multi-Run)")
    print("="*60)
    
    # Create a pattern bank with multiple configurations
    # Each config runs 3 times for robustness!
    ga = GeneticAlgorithmPatternFinder(
        populationSize=150,  # Slightly smaller for speed
        generations=50,  # Faster per run
        forgivenessPct=0.05,
        earlyStopGenerations=10,
        minImprovementThreshold=0.01
    )
    
    # Discover patterns across different timeframes and lengths
    patternBank = ga.discoverPatternBank(
        symbol="AAPL",
        intervals=None,  # Use defaults: daily, hourly, 30-min
        patternLengths=[3, 5, 7, 9],  # Test various pattern lengths (3-10 available)
        numRunsPerConfig=3,  # Run each config 3 times
        verbose=True
    )
    
    # Show best patterns
    print("\n" + "="*60)
    print("Top 5 Patterns in Bank:")
    print("="*60)
    for i, pattern in enumerate(patternBank.getBestPatterns(5), 1):
        print(f"{i}. Interval: {pattern.interval}, Length: {len(pattern)}, "
              f"Fitness: {pattern.fitness:.2f}")
    
    # Save pattern bank
    # patternBank.saveToFile("aapl_patterns.pkl")
    # print("\nPattern bank saved to aapl_patterns.pkl")
    
    print("\n" + "="*60)
    print("Pattern discovery complete!")
    print("="*60)
