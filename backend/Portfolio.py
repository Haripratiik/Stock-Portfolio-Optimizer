"""
Portfolio Management System with Pattern-Based Trading

This module implements a portfolio manager that:
1. Discovers patterns for multiple stocks using genetic algorithms
2. Backtests patterns with proper fund allocation per stock
3. Generates comprehensive performance visualizations
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from GeneticAlgorithm import (
    GeneticAlgorithmPatternFinder, 
    PatternBank, 
    PatternChromosome,
    StockDataFetcher
)
from Backtester import Backtester, BacktestResult


@dataclass
class PortfolioStock:
    """Represents a stock in the portfolio"""
    symbol: str
    allocation: float  # Percentage of total fund (0.0 to 1.0)
    patternBank: PatternBank = None
    backtestResults: Dict[str, BacktestResult] = None  # interval -> BacktestResult
    
    def __post_init__(self):
        if self.backtestResults is None:
            self.backtestResults = {}


class Portfolio:
    """
    Portfolio manager that discovers patterns and backtests across multiple stocks.
    """
    
    def __init__(self, totalFund: float, stocks: Dict[str, float]):
        """
        Initialize portfolio.
        
        Args:
            totalFund: Total capital available for trading
            stocks: Dictionary mapping stock symbol to allocation percentage (must sum to 1.0)
        
        Example:
            portfolio = Portfolio(
                totalFund=100000,
                stocks={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
            )
        """
        self.totalFund = totalFund
        self.stocks: Dict[str, PortfolioStock] = {}
        
        # Validate allocations sum to 1.0
        totalAllocation = sum(stocks.values())
        if not (0.99 <= totalAllocation <= 1.01):  # Allow small floating point error
            raise ValueError(f"Stock allocations must sum to 1.0, got {totalAllocation}")
        
        # Normalize allocations to exactly 1.0
        normFactor = 1.0 / totalAllocation
        for symbol, allocation in stocks.items():
            self.stocks[symbol] = PortfolioStock(
                symbol=symbol,
                allocation=allocation * normFactor
            )
    
    def discoverPatterns(self,
                        intervals: List[Tuple[str, str]] = None,
                        patternLengths: List[int] = None,
                        populationSize: int = 200,
                        numGenerations: int = 100,
                        numRuns: int = 3,
                        verbose: bool = True):
        """
        Discover patterns for all stocks in the portfolio.
        
        Args:
            intervals: List of (interval, period) tuples for pattern discovery
            patternLengths: List of pattern lengths to discover
            populationSize: GA population size
            numGenerations: GA generations per run
            numRuns: Number of GA runs per configuration (keeps best)
            verbose: Print progress
        """
        if intervals is None:
            intervals = [('1d', '2y'), ('1h', '730d'), ('30m', '60d')]
        
        if patternLengths is None:
            patternLengths = [3, 4, 5, 6, 7, 8]
        
        if verbose:
            print("=" * 80)
            print("PORTFOLIO PATTERN DISCOVERY")
            print("=" * 80)
            print(f"Total Fund: ${self.totalFund:,.2f}")
            print(f"Stocks: {list(self.stocks.keys())}")
            print(f"Intervals: {[i[0] for i in intervals]}")
            print(f"Pattern Lengths: {patternLengths}")
            print("=" * 80)
        
        for symbol, stockInfo in self.stocks.items():
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Discovering patterns for {symbol} (Allocation: {stockInfo.allocation*100:.1f}%)")
                print(f"{'=' * 80}")
            
            # Discover pattern bank for this stock
            ga = GeneticAlgorithmPatternFinder(
                populationSize=populationSize,
                patternLength=5,  # Will be overridden for each config in discoverPatternBank
                generations=numGenerations
            )
            
            patternBank = ga.discoverPatternBank(
                symbol=symbol,
                intervals=intervals,
                patternLengths=patternLengths,
                numRunsPerConfig=numRuns,
                verbose=verbose
            )
            
            stockInfo.patternBank = patternBank
            
            if verbose:
                print(f"\n✓ Discovered {len(patternBank.patterns)} patterns for {symbol}")
                for interval in set(p.interval for p in patternBank.patterns):
                    intervalPatterns = [p for p in patternBank.patterns if p.interval == interval]
                    print(f"  - {interval}: {len(intervalPatterns)} patterns")
    
    def backtest(self,
                startDate: str,
                endDate: str,
                topPatternsPerInterval: int = 5,
                verbose: bool = True) -> Dict[str, Dict[str, BacktestResult]]:
        """
        Backtest all discovered patterns with proper fund allocation.
        
        Args:
            startDate: Backtest start date (YYYY-MM-DD)
            endDate: Backtest end date (YYYY-MM-DD)
            topPatternsPerInterval: Number of top patterns to test per interval
            verbose: Print progress
        
        Returns:
            Dictionary: {symbol: {interval: BacktestResult}}
        """
        if verbose:
            print("\n" + "=" * 80)
            print("PORTFOLIO BACKTESTING")
            print("=" * 80)
            print(f"Period: {startDate} to {endDate}")
            print("=" * 80)
        
        results = {}
        
        for symbol, stockInfo in self.stocks.items():
            if stockInfo.patternBank is None:
                if verbose:
                    print(f"\n⚠ No patterns discovered for {symbol}, skipping...")
                continue
            
            # Calculate fund for this stock
            stockFund = self.totalFund * stockInfo.allocation
            
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Backtesting {symbol}")
                print(f"Allocated Fund: ${stockFund:,.2f} ({stockInfo.allocation*100:.1f}% of total)")
                print(f"{'=' * 80}")
            
            # Create backtester with this stock's fund
            backtester = Backtester(
                initialFund=stockFund,
                forgiveness=0.05
            )
            
            # Backtest pattern bank
            stockResults = backtester.backtestPatternBank(
                patternBank=stockInfo.patternBank,
                symbol=symbol,
                startDate=startDate,
                endDate=endDate,
                topN=topPatternsPerInterval,
                verbose=verbose
            )
            
            stockInfo.backtestResults = stockResults
            results[symbol] = stockResults
        
        return results
    
    def generatePerformanceReport(self,
                                 startDate: str,
                                 endDate: str,
                                 savePath: str = None,
                                 showPlot: bool = True):
        """
        Generate comprehensive performance graphs.
        
        Creates three sets of graphs:
        1. Total portfolio performance over time
        2. Individual stock performance comparison
        3. Per-timeframe performance for each stock
        
        Args:
            startDate: Start date for visualization
            endDate: End date for visualization
            savePath: Path to save figure (optional)
            showPlot: Whether to display the plot
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.35, top=0.96, bottom=0.05, left=0.05, right=0.98)
        
        # 1. Total Portfolio Performance (top row, spanning all columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plotTotalPortfolioPerformance(ax1, startDate, endDate)
        
        # 2. Individual Stock Performance (second row)
        ax2 = fig.add_subplot(gs[1, :])
        self._plotStockComparison(ax2)
        
        # 3. Per-Timeframe Performance for Each Stock (remaining rows)
        stockSymbols = list(self.stocks.keys())
        numStocks = len(stockSymbols)
        
        # Arrange stock charts in a grid
        for idx, symbol in enumerate(stockSymbols):
            row = 2 + (idx // 3)
            col = idx % 3
            
            if row < 4:  # Limit to available space
                ax = fig.add_subplot(gs[row, col])
                self._plotStockTimeframePerformance(ax, symbol)
        
        # Overall title
        fig.suptitle(
            f'Portfolio Performance Analysis | Fund: ${self.totalFund:,.0f} | Period: {startDate} to {endDate}',
            fontsize=18,
            fontweight='bold',
            y=0.99
        )
        
        # Save if path provided
        if savePath:
            plt.savefig(savePath, dpi=300, bbox_inches='tight')
            print(f"\n✓ Performance report saved to: {savePath}")
        
        # Show plot
        if showPlot:
            plt.show()
        
        return fig
    
    def _plotTotalPortfolioPerformance(self, ax, startDate: str, endDate: str):
        """Plot total portfolio value over time (day-by-day cumulative)"""
        # Collect all trades from all stocks/intervals
        allTrades = []
        
        for symbol, stockInfo in self.stocks.items():
            for interval, result in stockInfo.backtestResults.items():
                for trade in result.trades:
                    allTrades.append({
                        'timestamp': trade['timestamp'],
                        'returnPct': trade['returnPct'],
                        'fundAllocation': trade.get('fundAllocation', 0)
                    })
        
        if not allTrades:
            # No trades, just show flat line
            ax.plot([startDate, endDate], [self.totalFund, self.totalFund], 
                   linewidth=2, color='gray', linestyle='--')
            ax.text(0.5, 0.5, 'No Trades', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, color='gray')
        else:
            # Sort trades by timestamp
            allTrades.sort(key=lambda x: x['timestamp'])
            
            # Build cumulative performance curve
            timestamps = [pd.to_datetime(startDate)]
            portfolioValues = [self.totalFund]
            cumulativeProfit = 0
            
            for trade in allTrades:
                profit = trade['fundAllocation'] * (trade['returnPct'] / 100)
                cumulativeProfit += profit
                timestamps.append(trade['timestamp'])
                portfolioValues.append(self.totalFund + cumulativeProfit)
            
            # Add final point at endDate
            timestamps.append(pd.to_datetime(endDate))
            portfolioValues.append(self.totalFund + cumulativeProfit)
            
            # Plot the curve (line connecting trade points)
            ax.plot(timestamps, portfolioValues, linewidth=2.5, color='#2E7D32', 
                   alpha=0.9)
            ax.fill_between(timestamps, self.totalFund, portfolioValues, 
                           alpha=0.2, color='#4CAF50')
            
            # Calculate metrics
            totalTrades = len(allTrades)
            finalValue = self.totalFund + cumulativeProfit
            portfolioReturn = (cumulativeProfit / self.totalFund) * 100
            
            # Add annotations
            ax.text(timestamps[0], portfolioValues[0], f'${self.totalFund:,.0f}', 
                   ha='left', va='bottom', fontsize=9, fontweight='bold')
            
            # Position final value annotation to avoid overlap
            finalX = timestamps[-1]
            finalY = finalValue
            offset = (finalValue - self.totalFund) * 0.1  # Offset based on gain/loss
            ax.annotate(f'${finalValue:,.0f}\n({portfolioReturn:+.2f}%)', 
                       xy=(finalX, finalY),
                       xytext=(finalX, finalY + offset),
                       ha='right', va='bottom' if portfolioReturn >= 0 else 'top', 
                       fontsize=9, fontweight='bold', 
                       color='green' if portfolioReturn >= 0 else 'red',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Summary text
            summaryText = f'Trades: {totalTrades} | P/L: ${cumulativeProfit:,.2f} | Return: {portfolioReturn:.2f}%'
            ax.text(0.5, 0.95, summaryText, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_title('Total Portfolio Performance Over Time', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=self.totalFund, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial')
        ax.legend(loc='best', fontsize=8)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Rotate and format date labels to prevent overlap
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    
    def _plotStockComparison(self, ax):
        """Plot performance comparison across stocks"""
        stockLabels = []
        returns = []
        winRates = []
        allocations = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (symbol, stockInfo) in enumerate(self.stocks.items()):
            stockLabels.append(symbol)
            allocations.append(stockInfo.allocation)
            stockFund = self.totalFund * stockInfo.allocation
            
            # Calculate stock-level metrics
            stockProfit = 0
            stockTrades = 0
            stockWins = 0
            
            for interval, result in stockInfo.backtestResults.items():
                # Sum up actual profits from trades (using their specific fund allocations)
                for trade in result.trades:
                    profit = trade.get('fundAllocation', 0) * (trade['returnPct'] / 100)
                    stockProfit += profit
                stockTrades += result.totalTrades
                stockWins += result.successfulTrades
            
            stockReturn = (stockProfit / stockFund) * 100 if stockFund > 0 else 0
            winRate = (stockWins / stockTrades * 100) if stockTrades > 0 else 0
            
            returns.append(stockReturn)
            winRates.append(winRate)
        
        # Create side-by-side bars
        x = np.arange(len(stockLabels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, returns, width, label='Return (%)', 
                      color=[colors[i % len(colors)] for i in range(len(returns))], alpha=0.8)
        bars2 = ax.bar(x + width/2, winRates, width, label='Win Rate (%)', 
                      color=[colors[i % len(colors)] for i in range(len(winRates))], alpha=0.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add allocation percentages below x-axis
        ax_ylim = ax.get_ylim()
        y_below = ax_ylim[0] - (ax_ylim[1] - ax_ylim[0]) * 0.12
        for i, alloc in enumerate(allocations):
            ax.text(i, y_below, f'Alloc:\n{alloc*100:.0f}%', 
                   ha='center', va='top', fontsize=7, style='italic', color='gray')
        
        ax.set_title('Stock Performance Comparison', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Stock Symbol', fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stockLabels, fontweight='bold', fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylim(bottom=y_below - 5)  # Make room for allocation text
    
    def _plotStockTimeframePerformance(self, ax, symbol: str):
        """Plot per-timeframe performance for a single stock (cumulative over time)"""
        stockInfo = self.stocks[symbol]
        
        if not stockInfo.backtestResults:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{symbol} - Timeframe Performance', fontsize=11, fontweight='bold')
            return
        
        # Plot cumulative performance curve for each timeframe
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (interval, result) in enumerate(sorted(stockInfo.backtestResults.items())):
            if not result.trades:
                continue
            
            # Sort trades by timestamp
            sortedTrades = sorted(result.trades, key=lambda x: x['timestamp'])
            
            # Build cumulative curve
            timestamps = [sortedTrades[0]['timestamp']]
            values = [0]  # Start at 0 profit
            cumulativeProfit = 0
            
            for trade in sortedTrades:
                profit = trade.get('fundAllocation', 0) * (trade['returnPct'] / 100)
                cumulativeProfit += profit
                timestamps.append(trade['timestamp'])
                values.append(cumulativeProfit)
            
            # Plot line for this interval
            color = colors[idx % len(colors)]
            winRate = (result.successfulTrades / result.totalTrades * 100) if result.totalTrades > 0 else 0
            label = f"{interval}: ${cumulativeProfit:,.0f} ({result.totalTrades} trades, {winRate:.0f}% WR)"
            ax.plot(timestamps, values, linewidth=2, color=color, alpha=0.8, 
                   marker='o', markersize=4, label=label)
        
        ax.set_title(f'{symbol} - Timeframe Performance Over Time', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Date', fontsize=9, fontweight='bold')
        ax.set_ylabel('Cumulative P/L ($)', fontsize=9, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=7, framealpha=0.9)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Rotate and format date labels to prevent overlap
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    
    def printSummary(self):
        """Print detailed portfolio summary"""
        print("\n" + "=" * 80)
        print("PORTFOLIO SUMMARY")
        print("=" * 80)
        print(f"Total Fund: ${self.totalFund:,.2f}")
        print(f"Number of Stocks: {len(self.stocks)}")
        print("=" * 80)
        
        totalProfit = 0
        totalTrades = 0
        totalWins = 0
        
        for symbol, stockInfo in self.stocks.items():
            stockFund = self.totalFund * stockInfo.allocation
            print(f"\n{symbol}:")
            print(f"  Allocation: {stockInfo.allocation*100:.1f}% (${stockFund:,.2f})")
            
            if stockInfo.patternBank:
                print(f"  Patterns Discovered: {len(stockInfo.patternBank.patterns)}")
                
                # Group by interval
                intervals = set(p.interval for p in stockInfo.patternBank.patterns)
                for interval in sorted(intervals):
                    intervalPatterns = [p for p in stockInfo.patternBank.patterns 
                                      if p.interval == interval]
                    avgPatternAccuracy = np.mean([p.getAccuracy() for p in intervalPatterns])
                    avgOverallAccuracy = np.mean([p.getOverallAccuracy() for p in intervalPatterns])
                    avgMatches = np.mean([p.totalMatches for p in intervalPatterns])
                    avgPossible = np.mean([p.totalPossiblePositions for p in intervalPatterns])
                    print(f"    {interval}: {len(intervalPatterns)} patterns")
                    print(f"      Pattern Accuracy: {avgPatternAccuracy*100:.1f}% (avg {avgMatches:.0f} matches)")
                    print(f"      Overall Accuracy: {avgOverallAccuracy*100:.2f}% (of {avgPossible:.0f} possible)")
            
            if stockInfo.backtestResults:
                # Sum up actual profits from trades (using their specific fund allocations)
                stockProfit = 0
                for result in stockInfo.backtestResults.values():
                    for trade in result.trades:
                        profit = trade.get('fundAllocation', 0) * (trade['returnPct'] / 100)
                        stockProfit += profit
                stockTrades = sum(r.totalTrades for r in stockInfo.backtestResults.values())
                stockWins = sum(r.successfulTrades for r in stockInfo.backtestResults.values())
                stockWinRate = (stockWins / stockTrades * 100) if stockTrades > 0 else 0
                stockReturn = (stockProfit / stockFund * 100) if stockFund > 0 else 0
                
                print(f"  Backtest Results:")
                print(f"    Total Trades: {stockTrades}")
                print(f"    Win Rate: {stockWinRate:.2f}%")
                print(f"    Total P/L: ${stockProfit:,.2f}")
                print(f"    Return: {stockReturn:+.2f}%")
                
                totalProfit += stockProfit
                totalTrades += stockTrades
                totalWins += stockWins
        
        # Overall portfolio metrics
        print("\n" + "=" * 80)
        print("OVERALL PORTFOLIO PERFORMANCE")
        print("=" * 80)
        portfolioReturn = (totalProfit / self.totalFund * 100) if self.totalFund > 0 else 0
        portfolioWinRate = (totalWins / totalTrades * 100) if totalTrades > 0 else 0
        finalValue = self.totalFund + totalProfit
        
        print(f"Initial Value: ${self.totalFund:,.2f}")
        print(f"Final Value: ${finalValue:,.2f}")
        print(f"Total Trades: {totalTrades}")
        print(f"Win Rate: {portfolioWinRate:.2f}%")
        print(f"Total P/L: ${totalProfit:,.2f}")
        print(f"Total Return: {portfolioReturn:+.2f}%")
        print("=" * 80)


# =============================================================================
#
#   PORTFOLIO CONFIGURATION
#
#   Edit the values below to define your portfolio.  These are imported
#   by PortfolioTester.py — you only need to change them here.
#
# =============================================================================

# ── Total Capital ────────────────────────────────────────────────────────────
TOTAL_FUND = 100000                    # Total investment capital ($)

# ── Stocks & Allocations ────────────────────────────────────────────────────
#    Symbol  →  fraction of TOTAL_FUND   (must sum to 1.0)
STOCKS = {
    'AAPL':   0.40,                      # 40 %
    'GOOGL':  0.30,                      # 30 %
    'MSFT':   0.30,                      # 30 %
}


# =============================================================================
# Standalone run  (optional — the main entry point is PortfolioTester.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO PATTERN TRADING SYSTEM")
    print("=" * 80)

    # ---- Algorithm / backtest settings (only used for standalone run) ----
    INTERVALS = [('1d', '5y'), ('1h', '730d'), ('30m', '60d')]
    PATTERN_LENGTHS = [3, 4, 5, 6, 7, 8, 9, 10]
    POPULATION_SIZE = 500
    NUM_GENERATIONS = 50
    NUM_RUNS = 2
    BACKTEST_PERIOD_DAYS = 730 * 2.5
    TOP_PATTERNS_PER_INTERVAL = 5

    # Create portfolio
    portfolio = Portfolio(
        totalFund=TOTAL_FUND,
        stocks=STOCKS
    )

    print(f"\nPortfolio Created:")
    print(f"  Total Fund: ${portfolio.totalFund:,.2f}")
    print(f"  Stocks: {list(portfolio.stocks.keys())}")

    # Discover patterns
    print("\nDiscovering patterns...")
    portfolio.discoverPatterns(
        intervals=INTERVALS,
        patternLengths=PATTERN_LENGTHS,
        populationSize=POPULATION_SIZE,
        numGenerations=NUM_GENERATIONS,
        numRuns=NUM_RUNS,
        verbose=True
    )

    # Backtest
    print("\nBacktesting patterns...")
    endDate = datetime.now()
    startDate = endDate - timedelta(days=BACKTEST_PERIOD_DAYS)

    portfolio.backtest(
        startDate=startDate.strftime('%Y-%m-%d'),
        endDate=endDate.strftime('%Y-%m-%d'),
        topPatternsPerInterval=TOP_PATTERNS_PER_INTERVAL,
        verbose=True
    )

    portfolio.printSummary()

    print("\nGenerating performance visualizations...")
    portfolio.generatePerformanceReport(
        startDate=startDate.strftime('%Y-%m-%d'),
        endDate=endDate.strftime('%Y-%m-%d'),
        savePath='portfolio_performance.png',
        showPlot=True
    )

    print("\nPortfolio analysis complete!")
