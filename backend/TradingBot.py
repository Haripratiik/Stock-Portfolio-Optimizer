"""
TradingBot — Live Signal-to-Order Execution Engine
====================================================

Orchestrates one full trading cycle:

  1. Load portfolio stocks (manual + connected)
  2. Fetch latest OHLCV for each symbol
  3. Retrieve (or train) per-stock ML models & portfolio model
  4. Generate StockPredictions → PortfolioSignal → TradingDecider
  5. Convert FinalTradeDecisions into sized orders (OrderBuilder)
  6. Execute orders through BrokerClient (paper / live / dry-run)
  7. Log everything to StockOrderBook (Firestore audit trail)
  8. Enforce safety limits (kill switch, daily loss cap, position caps)

Usage:
  bot = TradingBot(dbClient=db, logFn=print, brokerMode='paper')
  bot.runCycle()
"""

from __future__ import annotations

import datetime
import math
import os
import uuid
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    """Current time in US/Eastern (falls back to local if zoneinfo unavailable)."""
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Local imports ────────────────────────────────────────────────────
from BrokerClient import (
    BrokerClient, AlpacaBroker, DryRunBroker,
    OrderResult, Position, AccountInfo,
)
from StockOrderBook import StockOrderBook, OrderEntry
from StockMLModel import StockMLModel, StockPrediction, TradingSignal
from PortfolioMLModel import PortfolioMLModel, PortfolioSignal
from TradingDecider import TradingDecider, FinalTradeDecision
from GeneticAlgorithm import StockDataFetcher, PatternBank
from PersistenceManager import PersistenceManager, StockMeta
from EarningsBlackout import EarningsBlackoutChecker
from AlertManager import AlertManager


# =====================================================================
# Configuration
# =====================================================================

# Period per interval (Yahoo Finance limits: 1h ~60d, 1d ~6mo+)
INTERVAL_PERIODS = {'1h': '60d', '1d': '6mo'}

@dataclass
class TradingConfig:
    """Tunable safety and sizing parameters."""
    maxPctPerStock: float = 0.25       # no single stock > 25% of portfolio
    maxDollarPerOrder: float = 25_000  # hard cap per order
    minOrderDollars: float = 50        # skip tiny orders
    minConfidence: float = 0.25        # ignore signals below this (aligned with pipeline)
    dailyLossLimitPct: float = 5.0     # halt if day loss exceeds this %
    stopLossPct: float = 5.0           # per-position stop-loss (close if down >X%)
    useStopLoss: bool = True           # if False, disable per-trade stop-loss (respects USE_STOP_LOSS env)
    useEarningsBlackout: bool = True   # earnings proximity sizing/conf boost (helps avoid weak signals)
    portfolioDrawdownPct: float = 15.0 # portfolio circuit breaker (% from peak equity)
    roundShares: bool = True           # round to whole shares
    allowShorts: bool = False          # if True, SELL with no position opens a short (requires margin)
    # Ghost/restore (matches DynamicAllocator)
    restoreThreshold: float = 3.0      # recent return % needed to phase ghost back in
    restoreAllocation: float = 0.03    # initial allocation when restoring


# =====================================================================
# OrderBuilder — converts decisions to sized orders
# =====================================================================

class OrderBuilder:
    """
    Translates FinalTradeDecision + allocation into concrete order params.

    Sizing formula (BUY):
        dollarAmount = totalFund × allocation × positionSize × confidence
        qty = floor(dollarAmount / currentPrice)

    For SELL:
        If held, sell/close long (or partial based on positionSize).
        If no position and allowShorts, open a short (sell to open).
    """

    def __init__(self, config: TradingConfig):
        self.cfg = config

    def build(
        self,
        decision: FinalTradeDecision,
        allocation: float,
        totalFund: float,
        currentPrice: float,
        currentQty: int,
        positionSide: str = 'long',  # 'long' | 'short' — for SELL, determines close vs add
    ) -> Optional[Dict]:
        """
        Return dict with {symbol, side, qty, orderType, limitPrice, reason}
        or None if no action needed.
        """
        if decision.signal == TradingSignal.HOLD:
            return None

        if decision.confidence < self.cfg.minConfidence:
            return None

        symbol = decision.symbol

        if decision.signal == TradingSignal.BUY:
            return self._buildBuy(
                symbol, decision, allocation, totalFund, currentPrice
            )
        elif decision.signal == TradingSignal.SELL:
            return self._buildSell(
                symbol, decision, allocation, totalFund, currentPrice, currentQty
            )
        return None

    def _buildBuy(
        self,
        symbol: str,
        decision: FinalTradeDecision,
        allocation: float,
        totalFund: float,
        currentPrice: float,
    ) -> Optional[Dict]:
        targetDollar = totalFund * allocation * decision.positionSize

        # Cap by max % per stock
        maxDollar = totalFund * self.cfg.maxPctPerStock
        targetDollar = min(targetDollar, maxDollar, self.cfg.maxDollarPerOrder)

        if targetDollar < self.cfg.minOrderDollars or currentPrice <= 0:
            return None

        qty = targetDollar / currentPrice
        if self.cfg.roundShares:
            qty = math.floor(qty)
        if qty <= 0:
            return None

        return {
            'symbol': symbol,
            'side': 'buy',
            'qty': qty,
            'orderType': 'market',
            'limitPrice': None,
            'reason': decision.reason,
            'confidence': decision.confidence,
            'allocation': allocation,
            'positionSizePct': decision.positionSize,
        }

    def _buildSell(
        self,
        symbol: str,
        decision: FinalTradeDecision,
        allocation: float,
        totalFund: float,
        currentPrice: float,
        currentQty: int,
        positionSide: str = 'long',
    ) -> Optional[Dict]:
        if currentQty > 0 and positionSide.lower() == 'long':
            # Close/reduce long position
            sellQty = max(1, math.floor(currentQty * decision.positionSize))
            sellQty = min(sellQty, currentQty)
            return {
                'symbol': symbol,
                'side': 'sell',
                'qty': sellQty,
                'orderType': 'market',
                'limitPrice': None,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'allocation': 0.0,
                'positionSizePct': decision.positionSize,
            }
        if currentQty > 0 and positionSide.lower() == 'short':
            # Add to short — size like new short
            return self._buildShort(symbol, decision, allocation, totalFund, currentPrice)

        # No position — open short if allowed
        if not getattr(self.cfg, 'allowShorts', False):
            return None
        return self._buildShort(symbol, decision, allocation, totalFund, currentPrice)

    def _buildShort(
        self,
        symbol: str,
        decision: FinalTradeDecision,
        allocation: float,
        totalFund: float,
        currentPrice: float,
    ) -> Optional[Dict]:
        """Open a short position — same sizing as BUY but side='sell'."""
        targetDollar = totalFund * allocation * decision.positionSize
        maxDollar = totalFund * self.cfg.maxPctPerStock
        targetDollar = min(targetDollar, maxDollar, self.cfg.maxDollarPerOrder)
        if targetDollar < self.cfg.minOrderDollars or currentPrice <= 0:
            return None
        qty = targetDollar / currentPrice
        if self.cfg.roundShares:
            qty = math.floor(qty)
        if qty <= 0:
            return None
        return {
            'symbol': symbol,
            'side': 'sell',
            'qty': qty,
            'orderType': 'market',
            'limitPrice': None,
            'reason': decision.reason,
            'confidence': decision.confidence,
            'allocation': allocation,
            'positionSizePct': decision.positionSize,
        }


# =====================================================================
# TradingBot
# =====================================================================

class TradingBot:
    """
    Full trading cycle engine.

    Parameters
    ----------
    dbClient : Firestore client
    logFn : callable for status logging
    brokerMode : 'paper' | 'live' | 'dry_run'
    config : TradingConfig overrides
    """

    def __init__(
        self,
        dbClient=None,
        logFn=None,
        brokerMode: str = 'paper',
        config: Optional[TradingConfig] = None,
    ):
        self.db = dbClient
        self._log = logFn or print
        self.config = config or TradingConfig()

        self.persistence = PersistenceManager()
        self.fetcher = StockDataFetcher()
        self.decider = TradingDecider()
        self.orderBuilder = OrderBuilder(self.config)
        self.orderBook = StockOrderBook(dbClient=dbClient, logFn=self._log)
        self.alertManager = AlertManager(dbClient=dbClient)

        self._brokerMode = brokerMode
        self.broker: BrokerClient = self._initBroker(brokerMode)
        self._cycleId = ''

    # ─── Broker init ─────────────────────────────────────────────────

    def _initBroker(self, mode: str) -> BrokerClient:
        if mode == 'dry_run':
            self._log("[TradingBot] Using DryRun broker (no real orders)")
            totalFund = self._loadTotalFund()
            return DryRunBroker(startingCash=totalFund)

        try:
            broker = AlpacaBroker()
            self._log(f"[TradingBot] Alpaca broker connected ({broker.mode} mode)")
            return broker
        except (ImportError, ValueError) as e:
            self._log(
                f"[TradingBot] Alpaca unavailable ({e}), falling back to DryRun"
            )
            totalFund = self._loadTotalFund()
            return DryRunBroker(startingCash=totalFund)

    # ─── Main cycle ──────────────────────────────────────────────────

    def runCycle(self, interval: str = '1d') -> Dict:
        """
        Execute one complete trading cycle for the given interval.
        interval: '1d' (daily) or '1h' (hourly). Uses pipeline allocations
        and patterns trained by the backtesting engine.
        """
        if interval not in ('1d', '1h'):
            interval = '1d'
        self._currentInterval = interval
        self._cycleId = f"cycle-{interval}-{_now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._log(f"\n{'='*60}")
        self._log(f"[TradingBot] Starting cycle {self._cycleId} [interval={interval}]")
        self._log(f"[TradingBot] Broker mode: {self.broker.mode}")
        self._log(f"{'='*60}\n")

        # 0. Market hours check — only trade during US market hours
        if not self._isMarketOpen():
            nowEt = _now()
            self._log(
                f"[TradingBot] SKIPPED — market closed. "
                f"Current ET: {nowEt.strftime('%A %Y-%m-%d %H:%M %Z')}. "
                f"Window: {'9:00 AM–5:00 PM ET' if interval == '1d' else '9:30 AM–4:00 PM ET'} Mon–Fri."
            )
            return {'status': 'skipped', 'reason': 'market_closed'}

        # 0a. Safety: check kill switch
        if not self._checkKillSwitch():
            self._log("[TradingBot] HALTED — trading is disabled in app_config")
            return {'status': 'halted', 'reason': 'kill_switch'}

        # 0b. Load existing order book
        self.orderBook.loadFromFirestore()

        # 1. Load account state
        account = self.broker.get_account()
        configFund = self._loadTotalFund()
        if self._brokerMode in ('paper', 'live'):
            totalFund = account.equity
        else:
            totalFund = configFund or account.equity
        self._log(
            f"[TradingBot] Account: equity=${account.equity:,.2f}, "
            f"cash=${account.cash:,.2f}, fund=${totalFund:,.2f} "
            f"(source={'broker' if totalFund == account.equity else 'config'})"
        )

        # 2. Check daily loss limit
        if not self._checkDailyLossLimit(account):
            return {'status': 'halted', 'reason': 'daily_loss_limit'}

        # 3. Load portfolio stocks
        allStocks = self.persistence.getPortfolioStocks()
        if not allStocks:
            self._log("[TradingBot] No portfolio stocks found")
            return {'status': 'skipped', 'reason': 'no_stocks'}
        symbols = list(allStocks.keys())
        self._log(f"[TradingBot] Portfolio: {len(symbols)} stocks — {symbols}")

        # 4. Fetch latest OHLCV for this interval
        period = INTERVAL_PERIODS.get(interval, '6mo')
        stockData = self._fetchAllData(symbols, interval=interval, period=period)
        if not stockData:
            self._log("[TradingBot] No data fetched — aborting cycle")
            return {'status': 'error', 'reason': 'no_data'}

        # 5. Load live allocation (same intelligent allocation as backtesting)
        liveAlloc = self._loadLiveAllocation()
        ghostSlots = set(liveAlloc.get('ghostSlots', [])) if liveAlloc else set()
        slotAllocs = liveAlloc.get('slotAllocations', {}) if liveAlloc else {}
        stockAllocsFromPipeline = liveAlloc.get('stockAllocations', {}) if liveAlloc else {}

        # 6. Build ML models (trained on pipeline patterns), get predictions
        stockPredictions, portfolioSignal, allocations = (
            self._getSignals(
                stockData, allStocks, totalFund,
                interval=interval,
                stockAllocsFromPipeline=stockAllocsFromPipeline,
                slotAllocs=slotAllocs,
            )
        )

        # 7. Run TradingDecider
        decisions = self.decider.decide(
            stockPredictions=stockPredictions,
            portfolioSignal=portfolioSignal,
            currentAllocations=allocations,
            verbose=True,
        )
        self._log(f"\n[TradingBot] Decisions for {len(decisions)} stocks [{interval}]:")
        for sym, dec in decisions.items():
            slotStr = f"{sym}/{interval}"
            ghostTag = " [GHOST]" if slotStr in ghostSlots else ""
            self._log(
                f"  {sym}: {dec.signal.value} "
                f"(conf={dec.confidence:.2f}, size={dec.positionSize:.2f}) "
                f"— {dec.reason[:80]}{ghostTag}"
            )

        # 8. Get current broker positions for sell sizing (uppercase keys for reliable lookup)
        brokerPositions = {(p.symbol or '').upper(): p for p in self.broker.get_positions() if (p.symbol or '').strip()}

        # 8a. STOP-LOSS CHECK: close any position that has breached the per-stock stop
        self._checkStopLosses(brokerPositions, interval=interval)

        # 8b. PORTFOLIO CIRCUIT BREAKER: check if peak drawdown exceeded
        if self._checkPortfolioCircuitBreaker(account):
            return {'status': 'halted', 'reason': 'portfolio_drawdown'}

        # Refresh positions after any stop-loss closures
        brokerPositions = {(p.symbol or '').upper(): p for p in self.broker.get_positions() if (p.symbol or '').strip()}

        # 9. Build and execute orders (skip real orders for ghost slots; do shadow trades)
        results = self._executeOrders(
            decisions, allocations, totalFund, stockData, brokerPositions,
            ghostSlots=ghostSlots,
            interval=interval,
        )

        # 10. Process shadow trades for ghost slots (track for restore/phase-back)
        self._processShadowTrades(
            decisions, stockData, ghostSlots, interval=interval,
        )

        # 11. Check restore — phase ghost slots back in if performance recovered
        self._checkRestoreAndUpdateAllocation(ghostSlots, liveAlloc)

        # 12. Daily summary — use Alpaca's dayPL/dayPLPct as source of truth when available
        endAccount = self.broker.get_account()
        useBrokerPnL = self._brokerMode in ('paper', 'live')
        summary = self.orderBook.recordDailySummary(
            startEquity=account.equity,
            endEquity=endAccount.equity,
            brokerDayPL=endAccount.dayPL if useBrokerPnL else None,
            brokerDayPLPct=endAccount.dayPLPct if useBrokerPnL else None,
        )

        # 13. Persist cycle result + live account snapshot to Firestore
        cycleResult = {
            'cycleId': self._cycleId,
            'interval': interval,
            'timestamp': _now().isoformat(),
            'brokerMode': self.broker.mode,
            'numStocks': len(symbols),
            'numDecisions': len(decisions),
            'numOrders': results['numOrders'],
            'numFilled': results['numFilled'],
            'numRejected': results['numRejected'],
            'orderSummary': results['orderSummary'],
            'dayPL': summary.dayPL,
            'dayPLPct': summary.dayPLPct,
            'equity': endAccount.equity,
            'cash': endAccount.cash,
        }
        self._persistBrokerAccountSnapshot(endAccount)
        self._persistCycleResult(cycleResult)

        # 14. Save daily trade summary for next day (1d only, for website display)
        if interval == '1d':
            self._saveDailyTradeSummary(
                decisions=decisions,
                allocations=allocations,
                results=results,
                stockData=stockData,
                ghostSlots=ghostSlots,
            )

        # 15. Daily performance review & self-improvement (1d only)
        if interval == '1d':
            try:
                from DailyReviewEngine import DailyReviewEngine
                reviewer = DailyReviewEngine(
                    db=self.db,
                    persistence=self.persistence,
                    alertManager=self.alertManager,
                    logger=self._log,
                )
                reviewer.runDailyReview()
            except Exception as e:
                self._log(f"[TradingBot] Daily review failed (non-fatal): {e}")

        self._log(f"\n[TradingBot] Cycle complete — "
                  f"{results['numFilled']} filled, "
                  f"{results['numRejected']} rejected, "
                  f"day P&L: ${summary.dayPL:+,.2f} ({summary.dayPLPct:+.2f}%)")

        self.alertManager.notifyPnLThreshold(summary.dayPLPct, endAccount.equity)

        return {'status': 'completed', **cycleResult}

    # ─── Signal generation ───────────────────────────────────────────

    def _getSignals(
        self,
        stockData: Dict[str, pd.DataFrame],
        allStocks: Dict[str, StockMeta],
        totalFund: float,
        interval: str = '1d',
        stockAllocsFromPipeline: Optional[Dict[str, float]] = None,
        slotAllocs: Optional[Dict[str, float]] = None,
    ) -> Tuple[
        Dict[str, StockPrediction],
        Optional[PortfolioSignal],
        Dict[str, float],
    ]:
        """
        Build StockMLModels using patterns from pipeline (trained by backtesting).
        Uses live allocation from pipeline when available.
        """
        stockModels: Dict[str, StockMLModel] = {}
        stockPredictions: Dict[str, StockPrediction] = {}

        # Load patterns for this interval from pipeline (trained by backtesting)
        patternBank = self._loadLatestPatternBank(interval=interval)

        for sym, df in stockData.items():
            if df is None or len(df) < 30:
                self._log(f"[TradingBot] Skipping {sym} — insufficient data ({len(df) if df is not None else 0} rows)")
                continue

            model = StockMLModel(symbol=sym)
            try:
                model.train(df, patternBank, verbose=False)
                pred = model.predict(df, idx=-1)
                stockModels[sym] = model
                stockPredictions[sym] = pred
                self._log(
                    f"[TradingBot] {sym}: signal={pred.signal.value}, "
                    f"conf={pred.confidence:.3f}, er={pred.expectedReturn:+.3f}"
                )
            except Exception as e:
                self._log(f"[TradingBot] {sym} model error: {e}")

        # Use live allocation from pipeline (intelligent fund allocation) or fallback to equal-weight
        # interval is already the parameter passed to _getSignals
        if stockAllocsFromPipeline and slotAllocs:
            allocations = {}
            for s in stockModels:
                slotStr = f"{s}/{interval}"
                if slotStr in slotAllocs and slotAllocs[slotStr] > 0.001:
                    allocations[s] = slotAllocs[slotStr]
                elif s in stockAllocsFromPipeline and stockAllocsFromPipeline[s] > 0.001:
                    allocations[s] = stockAllocsFromPipeline[s]
                else:
                    allocations[s] = 0.0
            totalAlloc = sum(allocations.values())
            if totalAlloc > 0:
                allocations = {k: v / totalAlloc for k, v in allocations.items()}
            else:
                nSymbols = max(1, len(stockModels))
                allocations = {s: 1.0 / nSymbols for s in stockModels}
            self._log(f"[TradingBot] Using pipeline allocation: {allocations}")
        else:
            nSymbols = max(1, len(stockModels))
            allocations = {s: 1.0 / nSymbols for s in stockModels}

        portfolioSignal: Optional[PortfolioSignal] = None
        if len(stockModels) >= 2:
            try:
                portModel = PortfolioMLModel(
                    stockModels=stockModels,
                    allocations=allocations,
                    totalFund=totalFund,
                )
                portModel.train(
                    stockDataDict=stockData,
                    verbose=False,
                )
                portfolioSignal = portModel.predict(
                    stockPredictions=stockPredictions,
                    stockDataDict=stockData,
                )
                # Apply allocation adjustments
                for adj in (portfolioSignal.allocationAdjustments or []):
                    if adj.symbol in allocations:
                        allocations[adj.symbol] = adj.suggestedAllocation
                self._log(
                    f"[TradingBot] Portfolio signal: regime={portfolioSignal.riskRegime.value}, "
                    f"hedge={portfolioSignal.hedgeAction.value}, "
                    f"conf={portfolioSignal.confidence:.3f}"
                )
            except Exception as e:
                self._log(f"[TradingBot] Portfolio model error: {e}")

        return stockPredictions, portfolioSignal, allocations

    # ─── Order execution ─────────────────────────────────────────────

    def _executeOrders(
        self,
        decisions: Dict[str, FinalTradeDecision],
        allocations: Dict[str, float],
        totalFund: float,
        stockData: Dict[str, pd.DataFrame],
        brokerPositions: Dict[str, Position],
        ghostSlots: Optional[set] = None,
        interval: str = '1d',
    ) -> Dict:
        numOrders = 0
        numFilled = 0
        numRejected = 0
        orderSummary: List[Dict] = []
        ghostSlots = ghostSlots or set()

        # Earnings proximity (reverted when useEarningsBlackout=False)
        _earningsParams: Dict[str, Tuple[float, float]] = {}
        if getattr(self.config, 'useEarningsBlackout', False):
            _earningsChecker = EarningsBlackoutChecker()
            _earningsParams = _earningsChecker.getProximityParams(list(decisions.keys()))

        for sym, decision in decisions.items():
            slotStr = f"{sym}/{interval}"
            if slotStr in ghostSlots:
                self._log(f"[TradingBot] {sym} GHOST — skipping real order, shadow trade will track")
                continue

            earningsSizeMult, earningsConfBoost = _earningsParams.get(sym, (1.0, 0.0))

            # Near earnings: require higher confidence (only when useEarningsBlackout)
            if (getattr(self.config, 'useEarningsBlackout', False)
                    and earningsConfBoost > 0 and decision.signal != TradingSignal.HOLD):
                requiredConf = self.config.minConfidence + earningsConfBoost
                if decision.confidence < requiredConf:
                    self._log(
                        f"[EARNINGS] {sym}: skipping {decision.signal.value} — "
                        f"confidence {decision.confidence:.2f} below earnings "
                        f"threshold {requiredConf:.2f}")
                    continue

            # Apply earnings position-size reduction (only when useEarningsBlackout)
            if (getattr(self.config, 'useEarningsBlackout', False)
                    and earningsSizeMult < 1.0 and decision.signal != TradingSignal.HOLD):
                original = decision.positionSize
                object.__setattr__(decision, 'positionSize',
                                   round(decision.positionSize * earningsSizeMult, 4))
                self._log(f"[EARNINGS] {sym}: sizing {original:.2f}→{decision.positionSize:.2f} "
                          f"(mult={earningsSizeMult:.2f}, confBoost=+{earningsConfBoost:.2f})")

            df = stockData.get(sym)
            currentPrice = float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0.0
            # Position lookup: broker first (uppercase keys), then OrderBook fallback
            sym_upper = (sym or '').upper()
            brokerPos = brokerPositions.get(sym_upper) if sym_upper else None
            bookPos = self.orderBook.getPosition(sym) or self.orderBook.getPosition(sym_upper)
            brokerQty = int(brokerPos.qty) if brokerPos is not None and brokerPos.qty is not None else 0
            bookQty = int(bookPos.qty) if bookPos is not None and bookPos.qty is not None else 0
            # Use broker qty; fallback to OrderBook; for SELL use max of both to handle sync lag
            if decision.signal == TradingSignal.SELL and (brokerQty > 0 or bookQty > 0):
                currentQty = max(brokerQty, bookQty)
            elif brokerPos is not None:
                currentQty = brokerQty
            elif bookPos is not None and bookPos.qty > 0:
                currentQty = bookQty
            else:
                currentQty = 0

            if decision.signal == TradingSignal.SELL and currentQty <= 0 and not getattr(self.config, 'allowShorts', False):
                self._log(
                    f"[TradingBot] {sym} SELL skipped — no position "
                    f"(broker qty={brokerQty}, orderBook qty={bookQty}). "
                    f"Set allowShorts=True to open shorts when bearish."
                )

            pos_side = 'long'
            if brokerPos is not None and hasattr(brokerPos, 'side'):
                pos_side = str(getattr(brokerPos, 'side', 'long')).lower()
            elif bookPos is not None and hasattr(bookPos, 'side'):
                pos_side = str(getattr(bookPos, 'side', 'long')).lower()

            order = self.orderBuilder.build(
                decision=decision,
                allocation=allocations.get(sym, 0.0),
                totalFund=totalFund,
                currentPrice=currentPrice,
                currentQty=currentQty,
                positionSide=pos_side,
            )

            if order is None:
                continue

            # Enforce position cap
            if order['side'] == 'buy':
                bp = brokerPositions.get(sym_upper) if sym_upper else None
                existingValue = float(bp.marketValue) if bp and bp.marketValue is not None else 0.0
                orderValue = order['qty'] * currentPrice
                if (existingValue + orderValue) / totalFund > self.config.maxPctPerStock:
                    self._log(
                        f"[TradingBot] {sym} BUY capped — would exceed "
                        f"{self.config.maxPctPerStock*100:.0f}% position limit"
                    )
                    maxNewDollar = (totalFund * self.config.maxPctPerStock) - existingValue
                    if maxNewDollar < self.config.minOrderDollars:
                        continue
                    order['qty'] = max(1, math.floor(maxNewDollar / currentPrice))

            # Submit to broker
            self._log(
                f"[TradingBot] Submitting: {order['side'].upper()} "
                f"{order['qty']} {sym} ({order['orderType']})"
            )
            result: OrderResult = self.broker.place_order(
                symbol=sym,
                side=order['side'],
                qty=order['qty'],
                order_type=order['orderType'],
                limit_price=order.get('limitPrice'),
            )

            numOrders += 1
            _ok = ('filled', 'submitted', 'dry_run', 'accepted',
                   'new', 'pending_new', 'partially_filled', 'held')
            if result.status in _ok:
                numFilled += 1
            else:
                numRejected += 1
                if result.status == 'rejected' and result.message:
                    self._log(f"[TradingBot] REJECTED {sym} {order['side']}: {result.message}")

            entry = OrderEntry(
                orderId=result.orderId,
                symbol=sym,
                side=order['side'],
                qty=order['qty'],
                orderType=order['orderType'],
                limitPrice=order.get('limitPrice'),
                filledPrice=result.filledPrice,
                filledAt=result.filledAt,
                status=result.status,
                brokerMode=self.broker.mode,
                signal=decision.signal.value,
                confidence=decision.confidence,
                allocation=allocations.get(sym, 0.0),
                positionSizePct=decision.positionSize,
                reason=decision.reason[:200],
                rejectionMessage=result.message if result.status == 'rejected' else '',
                totalCost=(result.filledPrice or currentPrice) * order['qty'],
                runCycleId=self._cycleId,
                signalBreakdown=decision.signalBreakdown(),
            )
            self.orderBook.recordOrder(entry)

            self.alertManager.notifyTrade(
                symbol=sym, side=order['side'], qty=order['qty'],
                price=result.filledPrice or currentPrice,
                status=result.status, confidence=decision.confidence)

            orderSummary.append({
                'symbol': sym,
                'side': order['side'],
                'qty': order['qty'],
                'price': result.filledPrice or currentPrice,
                'status': result.status,
                'message': result.message,
            })

        return {
            'numOrders': numOrders,
            'numFilled': numFilled,
            'numRejected': numRejected,
            'orderSummary': orderSummary,
        }

    # ─── Stop-loss & circuit breaker ──────────────────────────────────

    # Interval-aware stop-loss thresholds (mirrors Backtester scaling).
    # For live trading the bot checks the broker's real-time current price at
    # each cycle — this naturally aligns with the close-based daily approach
    # since the daily cycle runs at market close.
    # Targets ~2.5-3x ATR for AAPL/GOOGL/MSFT at each interval.
    _STOP_LOSS_INTERVAL_MULT = {
        '5m':  0.15,  # 0.75%
        '15m': 0.20,  # 1.00%
        '30m': 0.30,  # 1.50%
        '1h':  0.90,  # 4.50%  (widened to match Backtester, reduce false stop-outs)
        '1d':  1.00,  # 5.00%  (close-based, ~2.8x daily ATR)
        '1wk': 2.00,  # 10.00%
    }

    def _checkStopLosses(
        self,
        positions: Dict[str, 'Position'],
        interval: str = '1d',
    ):
        """Close any position whose unrealised loss exceeds the interval-scaled stop-loss %."""
        if not getattr(self.config, 'useStopLoss', True):
            return
        mult = self._STOP_LOSS_INTERVAL_MULT.get(interval, 1.0)
        effectiveStop = self.config.stopLossPct * mult

        for sym, pos in positions.items():
            if float(pos.qty) == 0:
                continue
            avgEntry = float(pos.avgEntryPrice) if pos.avgEntryPrice else 0.0
            currentPrice = float(pos.currentPrice) if pos.currentPrice else 0.0
            if avgEntry <= 0 or currentPrice <= 0:
                continue

            # Use pos.side ('long'/'short') — Alpaca always returns positive qty
            # regardless of direction, so qty > 0 does NOT correctly detect shorts.
            isLong = str(pos.side).lower() != 'short'
            if isLong:
                lossPct = (avgEntry - currentPrice) / avgEntry * 100
            else:
                lossPct = (currentPrice - avgEntry) / avgEntry * 100

            # Positive lossPct = position is underwater
            if lossPct <= 0:
                continue

            if lossPct >= effectiveStop:
                self._log(
                    f"[STOP-LOSS] {sym}: {'LONG' if isLong else 'SHORT'} "
                    f"down {lossPct:.1f}% vs threshold {effectiveStop:.1f}% "
                    f"(entry=${avgEntry:.2f}, now=${currentPrice:.2f}) "
                    f"— closing position"
                )
                result = self.broker.close_position(sym)
                self._log(f"[STOP-LOSS] {sym}: {result.status} — {result.message}")
                self.alertManager.notifyStopLoss(
                    sym, lossPct, avgEntry, currentPrice)

    def _checkPortfolioCircuitBreaker(self, account) -> bool:
        """Return True (halt) if portfolio drawdown from peak exceeds limit."""
        try:
            doc = self.db.collection('app_config').document('trading').get()
            peakEquity = doc.to_dict().get('peakEquity', account.equity) if doc.exists else account.equity
        except Exception:
            peakEquity = account.equity

        peakEquity = max(peakEquity, account.equity)
        try:
            self.db.collection('app_config').document('trading').set(
                {'peakEquity': peakEquity}, merge=True)
        except Exception:
            pass

        if peakEquity <= 0:
            return False
        drawdownPct = (peakEquity - account.equity) / peakEquity * 100
        if drawdownPct >= self.config.portfolioDrawdownPct:
            self._log(
                f"[CIRCUIT BREAKER] Portfolio drawdown {drawdownPct:.1f}% "
                f"from peak ${peakEquity:,.0f} exceeds "
                f"{self.config.portfolioDrawdownPct}% limit — HALTING"
            )
            self.alertManager.notifyCircuitBreaker(
                drawdownPct, peakEquity, account.equity)
            return True
        return False

    # ─── Data fetching ───────────────────────────────────────────────

    def _fetchAllData(
        self, symbols: List[str], interval: str = '1d', period: str = '6mo'
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for sym in symbols:
            try:
                df = self.fetcher.fetchData(
                    sym,
                    interval=interval,
                    period=period,
                )
                if df is not None and len(df) > 0:
                    result[sym] = df
                    self._log(f"[TradingBot] Fetched {len(df)} bars for {sym}")
                else:
                    self._log(f"[TradingBot] No data for {sym}")
            except Exception as e:
                self._log(f"[TradingBot] Fetch error for {sym}: {e}")
        return result

    # ─── Pattern bank loading ────────────────────────────────────────

    def _loadLatestPatternBank(self, interval: str = '1d') -> PatternBank:
        """Load active patterns for the given interval (trained by pipeline/backtesting)."""
        allPatterns = []
        try:
            from GeneticAlgorithm import PatternChromosome as CandlestickPattern
            import json

            allStocks = self.persistence.getPortfolioStocks()
            for sym in allStocks:
                stored = self.persistence.loadBestPatterns(sym, interval, topN=25)
                for sp in stored:
                    try:
                        genes = json.loads(sp.genesJson) if isinstance(sp.genesJson, str) else sp.genesJson
                        p = CandlestickPattern(length=sp.patternLength)
                        p.genes = genes
                        p.fitness = sp.fitness
                        p.interval = interval
                        allPatterns.append(p)
                    except Exception:
                        continue
            self._log(f"[TradingBot] Loaded {len(allPatterns)} {interval} patterns from {len(allStocks)} symbols")
        except Exception as e:
            self._log(f"[TradingBot] Pattern loading error: {e}")
        return PatternBank(symbol='_portfolio', patterns=allPatterns)

    # ─── Market hours ─────────────────────────────────────────────────

    # US market holidays (NYSE observed). Update annually.
    _US_HOLIDAYS_2026 = {
        (1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (7, 3),
        (9, 7), (11, 26), (12, 25),
    }

    def _isMarketOpen(self) -> bool:
        """True if current Eastern time is within NYSE regular trading hours.

        Regular session: 9:30 AM – 4:00 PM ET, Monday–Friday, excluding
        US market holidays.  The 1d cycle gets a wider window (9:00 AM – 5:00 PM)
        to allow for pre/post-market analysis.
        """
        now = _now()
        # Weekend check (Monday=0 .. Sunday=6)
        if now.weekday() >= 5:
            return False
        # Holiday check
        if (now.month, now.day) in self._US_HOLIDAYS_2026:
            return False
        t = now.hour * 60 + now.minute  # minutes since midnight
        if self._currentInterval == '1d':
            # Wider window for daily cycles (pre-market prep through post-close)
            return 9 * 60 <= t <= 17 * 60  # 9:00 AM – 5:00 PM
        else:
            # Standard market hours for intraday
            return 9 * 60 + 30 <= t <= 16 * 60  # 9:30 AM – 4:00 PM

    # ─── Safety checks ───────────────────────────────────────────────

    def _checkKillSwitch(self) -> bool:
        """Return True if trading is enabled, False to halt."""
        if not self.db:
            return True
        try:
            doc = self.db.collection('app_config').document('trading').get()
            if doc.exists:
                data = doc.to_dict() or {}
                return bool(data.get('enabled', False))
            # If no config doc exists, default to disabled for safety
            return False
        except Exception:
            return False

    def _checkDailyLossLimit(self, account: AccountInfo) -> bool:
        """Return True if within daily loss limit, False to halt."""
        if account.dayPLPct < -self.config.dailyLossLimitPct:
            self._log(
                f"[TradingBot] HALTED — daily loss {account.dayPLPct:.2f}% "
                f"exceeds limit of -{self.config.dailyLossLimitPct:.1f}%"
            )
            return False
        return True

    # ─── Helpers ─────────────────────────────────────────────────────

    def _loadTotalFund(self) -> float:
        if not self.db:
            return 100_000
        try:
            doc = self.db.collection('app_config').document('portfolio').get()
            if doc.exists:
                val = (doc.to_dict() or {}).get('totalFund', 100_000)
                return float(val) if val else 100_000
        except Exception:
            pass
        return 100_000

    def _loadLiveAllocation(self) -> Optional[Dict]:
        """Load allocation from latest pipeline run (same as backtesting)."""
        try:
            return self.persistence.loadLiveAllocation()
        except Exception as e:
            self._log(f"[TradingBot] Could not load live allocation: {e}")
            return None

    def _processShadowTrades(
        self,
        decisions: Dict[str, FinalTradeDecision],
        stockData: Dict[str, pd.DataFrame],
        ghostSlots: set,
        interval: str = '1d',
    ) -> None:
        """
        For ghost slots: track hypothetical trades (no real orders).
        Open shadow position on BUY, close on SELL — record P/L for restore logic.
        """
        if not self.db or not ghostSlots:
            return

        shadowPositions = self.persistence.loadShadowPositions()
        now = _now().isoformat()

        for sym, decision in decisions.items():
            slotStr = f"{sym}/{interval}"
            if slotStr not in ghostSlots:
                continue

            df = stockData.get(sym)
            currentPrice = float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0.0
            if currentPrice <= 0:
                continue

            pos = shadowPositions.get(slotStr)

            if decision.signal == TradingSignal.BUY and pos is None:
                shadowPositions[slotStr] = {
                    'side': 'buy',
                    'entryPrice': currentPrice,
                    'entryTime': now,
                    'symbol': sym,
                }
                self._log(f"[TradingBot] GHOST shadow BUY {sym} @ ${currentPrice:.2f} (tracking)")

            elif decision.signal == TradingSignal.SELL and pos is not None:
                entryPrice = pos.get('entryPrice', currentPrice)
                returnPct = ((currentPrice - entryPrice) / entryPrice * 100) if entryPrice > 0 else 0
                dollarPnL = 0  # shadow = no real $, but we track return %
                trade = {
                    'timestamp': now,
                    'returnPct': returnPct,
                    'dollarPnL': dollarPnL,
                    'entryPrice': entryPrice,
                    'exitPrice': currentPrice,
                    'isGhost': True,
                }
                self.persistence.appendLiveSlotTrade(slotStr, trade)
                del shadowPositions[slotStr]
                self._log(
                    f"[TradingBot] GHOST shadow SELL {sym} @ ${currentPrice:.2f} "
                    f"— return {returnPct:+.2f}% (logged for restore)"
                )

        self.persistence.saveShadowPositions(shadowPositions)

    def _checkRestoreAndUpdateAllocation(
        self, ghostSlots: set, liveAlloc: Optional[Dict]
    ) -> None:
        """
        Check if any ghost slot has recovered (recent return >= restoreThreshold).
        If so, phase it back in by updating live_allocation.
        """
        if not liveAlloc or not ghostSlots or not self.db:
            return

        slotAllocs = dict(liveAlloc.get('slotAllocations', {}))
        restored: List[str] = []

        for slotStr in list(ghostSlots):
            trades = self.persistence.loadLiveSlotTrades(slotStr, limit=100)
            if len(trades) < 3:
                continue

            returns = []
            for t in trades:
                r = t.get('returnPct', 0)
                pnl = t.get('dollarPnL', 0)
                if pnl != 0:
                    r = abs(r) if pnl >= 0 else -abs(r)
                returns.append(r)

            recentReturnPct = sum(returns[-15:]) if len(returns) >= 15 else sum(returns)
            recentTrades = min(15, len(returns))

            if recentReturnPct >= self.config.restoreThreshold and recentTrades >= 3:
                slotAllocs[slotStr] = self.config.restoreAllocation
                restored.append(slotStr)
                self._log(
                    f"[TradingBot] RESTORE: {slotStr} phased back in "
                    f"(recent return {recentReturnPct:.2f}% >= {self.config.restoreThreshold}%)"
                )

        if restored:
            # Re-normalise allocations
            total = sum(slotAllocs.values())
            if total > 0:
                slotAllocs = {k: v / total for k, v in slotAllocs.items()}
            newGhostSlots = [s for s in liveAlloc.get('ghostSlots', []) if s not in restored]
            stockAllocs = {}
            for s, a in slotAllocs.items():
                sym = s.split('/')[0]
                stockAllocs[sym] = stockAllocs.get(sym, 0) + a
            self.persistence.saveLiveAllocation(
                slotAllocations=slotAllocs,
                ghostSlots=newGhostSlots,
                stockAllocations=stockAllocs,
                runId=liveAlloc.get('runId', ''),
            )

    def _saveDailyTradeSummary(
        self,
        decisions: Dict[str, FinalTradeDecision],
        allocations: Dict[str, float],
        results: Dict,
        stockData: Dict[str, pd.DataFrame],
        ghostSlots: set,
    ) -> None:
        """Save daily trade summary for next day (website display)."""
        if not self.db:
            return
        trades = []
        for sym, dec in decisions.items():
            slotStr = f"{sym}/1d"
            df = stockData.get(sym)
            price = float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0
            ghost = slotStr in ghostSlots
            # Find executed order for this symbol if any
            executed = next(
                (o for o in results.get('orderSummary', []) if o.get('symbol') == sym),
                None,
            )
            trades.append({
                'symbol': sym,
                'signal': dec.signal.value,
                'confidence': round(dec.confidence, 3),
                'allocation': round(allocations.get(sym, 0) * 100, 1),
                'reason': (dec.reason or '')[:150],
                'price': round(price, 2),
                'ghost': ghost,
                'executed': executed is not None,
                'qty': executed.get('qty', 0) if executed else 0,
                'status': executed.get('status', 'skipped') if executed else ('ghost' if ghost else 'hold'),
            })
        self.persistence.saveDailyTradeSummary({
            'interval': '1d',
            'cycleId': self._cycleId,
            'timestamp': _now().isoformat(),
            'trades': trades,
            'numOrders': results.get('numOrders', 0),
            'numFilled': results.get('numFilled', 0),
        })

    def _persistCycleResult(self, result: Dict) -> None:
        if not self.db:
            return
        try:
            self.db.collection('trade_cycles').document(
                result['cycleId']
            ).set(result)
        except Exception as e:
            self._log(f"[TradingBot] Failed to persist cycle result: {e}")

    def _persistBrokerAccountSnapshot(self, account) -> None:
        """Write live portfolio value from Alpaca to Firestore for frontend display."""
        if not self.db or self._brokerMode == 'dry_run':
            return
        try:
            self.db.collection('broker_account').document('live').set({
                'equity': account.equity,
                'cash': account.cash,
                'buyingPower': account.buyingPower,
                'portfolioValue': account.portfolioValue,
                'dayPL': account.dayPL,
                'dayPLPct': account.dayPLPct,
                'updatedAt': _now().isoformat(),
                'brokerMode': self.broker.mode,
            }, merge=True)
        except Exception as e:
            self._log(f"[TradingBot] Failed to persist broker snapshot: {e}")
