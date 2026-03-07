"""
StockOrderBook — Trade Log & Position Tracker
===============================================

Tracks every order placed by the TradingBot and maintains a running
view of positions, P&L, and execution history.  All data is persisted
to Firestore's `trade_log` and `trade_positions` collections so the
frontend can display a live audit trail.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    """Current time in US/Eastern (falls back to local if zoneinfo unavailable)."""
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

# ── Firebase ─────────────────────────────────────────────────────────
try:
    from firebase_admin import firestore as _fs
except ImportError:
    _fs = None


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class OrderEntry:
    """One order in the book."""
    orderId: str = ''
    symbol: str = ''
    side: str = ''              # 'buy' | 'sell'
    qty: int = 0
    orderType: str = 'market'   # 'market' | 'limit'
    limitPrice: Optional[float] = None
    filledPrice: Optional[float] = None
    filledAt: Optional[str] = None
    status: str = ''            # 'submitted' | 'filled' | 'rejected' | 'dry_run'
    brokerMode: str = ''        # 'paper' | 'live' | 'dry_run'
    signal: str = ''            # 'BUY' | 'SELL' | 'HOLD'
    confidence: float = 0.0
    allocation: float = 0.0
    positionSizePct: float = 0.0
    reason: str = ''
    rejectionMessage: str = ''  # Alpaca rejection reason when status='rejected'
    createdAt: str = ''
    totalCost: float = 0.0
    runCycleId: str = ''
    signalBreakdown: Optional[Dict] = None


@dataclass
class PositionEntry:
    """Tracked position for a symbol."""
    symbol: str = ''
    qty: int = 0
    avgEntryPrice: float = 0.0
    currentPrice: float = 0.0
    marketValue: float = 0.0
    unrealizedPL: float = 0.0
    unrealizedPLPct: float = 0.0
    totalInvested: float = 0.0
    totalRealized: float = 0.0
    numBuys: int = 0
    numSells: int = 0
    lastUpdated: str = ''


@dataclass
class DailySummary:
    """End-of-day summary for safety monitoring."""
    date: str = ''
    startEquity: float = 0.0
    endEquity: float = 0.0
    dayPL: float = 0.0
    dayPLPct: float = 0.0
    numOrders: int = 0
    numFilled: int = 0
    numRejected: int = 0


# =====================================================================
# StockOrderBook
# =====================================================================

class StockOrderBook:
    """
    Maintains the full trading audit trail in Firestore.

    Collections used:
      - trade_log:       Individual order records (append-only)
      - trade_positions: Per-symbol running position snapshots
      - trade_daily:     Daily P&L summaries for risk monitoring
    """

    _COL_LOG       = 'trade_log'
    _COL_POSITIONS = 'trade_positions'
    _COL_DAILY     = 'trade_daily'

    def __init__(self, dbClient=None, logFn=None):
        self.db = dbClient
        self._log = logFn or print
        self._orders: List[OrderEntry] = []
        self._positions: Dict[str, PositionEntry] = {}

    # ─── Record a new order ──────────────────────────────────────────

    def recordOrder(self, order: OrderEntry) -> None:
        if not order.createdAt:
            order.createdAt = _now().isoformat()
        self._orders.append(order)
        self._updateLocalPosition(order)
        self._persistOrder(order)
        self._log(
            f"[OrderBook] {order.side.upper()} {order.qty} {order.symbol} "
            f"@ ${order.filledPrice or 0:.2f} — {order.status} ({order.brokerMode})"
        )

    # ─── Position tracking ───────────────────────────────────────────

    def _updateLocalPosition(self, order: OrderEntry) -> None:
        """Update in-memory position based on a filled order."""
        if order.status not in ('filled', 'dry_run'):
            return

        sym = order.symbol
        pos = self._positions.get(sym, PositionEntry(symbol=sym))
        price = order.filledPrice or 0.0

        if order.side == 'buy':
            totalQty = pos.qty + order.qty
            if totalQty > 0:
                pos.avgEntryPrice = (
                    (pos.avgEntryPrice * pos.qty + price * order.qty) / totalQty
                )
            pos.qty = totalQty
            pos.totalInvested += price * order.qty
            pos.numBuys += 1
        elif order.side == 'sell':
            realized = (price - pos.avgEntryPrice) * order.qty
            pos.totalRealized += realized
            pos.qty = max(0, pos.qty - order.qty)
            pos.numSells += 1

        pos.currentPrice = price
        pos.marketValue = pos.qty * price
        if pos.avgEntryPrice > 0 and pos.qty > 0:
            pos.unrealizedPL = (price - pos.avgEntryPrice) * pos.qty
            pos.unrealizedPLPct = (
                (price - pos.avgEntryPrice) / pos.avgEntryPrice * 100
            )
        else:
            pos.unrealizedPL = 0
            pos.unrealizedPLPct = 0
        pos.lastUpdated = _now().isoformat()

        self._positions[sym] = pos
        self._persistPosition(pos)

    def getPosition(self, symbol: str) -> Optional[PositionEntry]:
        return self._positions.get(symbol)

    def getAllPositions(self) -> Dict[str, PositionEntry]:
        return dict(self._positions)

    def getOpenPositions(self) -> Dict[str, PositionEntry]:
        return {s: p for s, p in self._positions.items() if p.qty > 0}

    # ─── Daily summary ───────────────────────────────────────────────

    def recordDailySummary(
        self,
        startEquity: float,
        endEquity: float,
        *,
        brokerDayPL: Optional[float] = None,
        brokerDayPLPct: Optional[float] = None,
    ) -> DailySummary:
        """Record daily summary. When brokerDayPL/brokerDayPLPct are provided (from Alpaca),
        use them as the source of truth for P&L instead of endEquity - startEquity."""
        today = datetime.date.today().isoformat()
        todayOrders = [o for o in self._orders if o.createdAt.startswith(today)]
        if brokerDayPL is not None and brokerDayPLPct is not None:
            dayPL, dayPLPct = brokerDayPL, brokerDayPLPct
        else:
            dayPL = endEquity - startEquity
            dayPLPct = (
                (endEquity - startEquity) / startEquity * 100
                if startEquity > 0 else 0
            )
        summary = DailySummary(
            date=today,
            startEquity=startEquity,
            endEquity=endEquity,
            dayPL=dayPL,
            dayPLPct=dayPLPct,
            numOrders=len(todayOrders),
            numFilled=sum(1 for o in todayOrders if o.status in ('filled', 'dry_run')),
            numRejected=sum(1 for o in todayOrders if o.status == 'rejected'),
        )
        self._persistDailySummary(summary)
        return summary

    # ─── Query recent orders ─────────────────────────────────────────

    def getRecentOrders(self, limit: int = 20) -> List[OrderEntry]:
        return list(reversed(self._orders[-limit:]))

    def getOrdersForCycle(self, cycleId: str) -> List[OrderEntry]:
        return [o for o in self._orders if o.runCycleId == cycleId]

    def getTodayOrders(self) -> List[OrderEntry]:
        today = datetime.date.today().isoformat()
        return [o for o in self._orders if o.createdAt.startswith(today)]

    # ─── Firestore persistence ───────────────────────────────────────

    def _orderToDict(self, order: OrderEntry) -> dict:
        """Convert OrderEntry to Firestore-safe dict."""
        d = asdict(order)
        # Ensure signalBreakdown is a plain dict (Firestore-safe)
        sb = d.get('signalBreakdown')
        if sb is not None and not isinstance(sb, dict):
            d['signalBreakdown'] = dict(sb) if hasattr(sb, 'items') else None
        return d

    def _persistOrder(self, order: OrderEntry) -> None:
        if not self.db:
            return
        try:
            docId = order.orderId or (
                f"{order.symbol}-{order.side}-"
                f"{_now().strftime('%Y%m%d%H%M%S%f')}"
            )
            # Ensure unique docId for rejected orders (orderId often empty)
            if not order.orderId and order.status == 'rejected':
                docId = f"{order.symbol}-{order.side}-rej-{_now().strftime('%Y%m%d%H%M%S%f')}"
            self.db.collection(self._COL_LOG).document(docId).set(
                self._orderToDict(order)
            )
        except Exception as e:
            self._log(f"[OrderBook] Failed to persist order: {e}")

    def _persistPosition(self, pos: PositionEntry) -> None:
        if not self.db:
            return
        try:
            self.db.collection(self._COL_POSITIONS).document(pos.symbol).set(
                asdict(pos), merge=True
            )
        except Exception as e:
            self._log(f"[OrderBook] Failed to persist position: {e}")

    def _persistDailySummary(self, summary: DailySummary) -> None:
        if not self.db:
            return
        try:
            self.db.collection(self._COL_DAILY).document(summary.date).set(
                asdict(summary)
            )
        except Exception as e:
            self._log(f"[OrderBook] Failed to persist daily summary: {e}")

    # ─── Load from Firestore on startup ──────────────────────────────

    def loadFromFirestore(self) -> None:
        """Load existing positions and recent orders from Firestore."""
        if not self.db:
            return
        try:
            docs = self.db.collection(self._COL_POSITIONS).stream()
            for doc in docs:
                d = doc.to_dict()
                pos = PositionEntry(**{
                    k: d.get(k, getattr(PositionEntry(), k))
                    for k in PositionEntry.__dataclass_fields__
                })
                if pos.qty > 0:
                    self._positions[pos.symbol] = pos
            self._log(
                f"[OrderBook] Loaded {len(self._positions)} open positions from Firestore"
            )
        except Exception as e:
            self._log(f"[OrderBook] Failed to load positions: {e}")

        try:
            docs = (
                self.db.collection(self._COL_LOG)
                .order_by('createdAt', direction=_fs.Query.DESCENDING)
                .limit(100)
                .stream()
            )
            for doc in docs:
                d = doc.to_dict()
                entry = OrderEntry(**{
                    k: d.get(k, getattr(OrderEntry(), k))
                    for k in OrderEntry.__dataclass_fields__
                })
                self._orders.append(entry)
            self._orders.reverse()
            self._log(
                f"[OrderBook] Loaded {len(self._orders)} recent orders from Firestore"
            )
        except Exception as e:
            self._log(f"[OrderBook] Failed to load orders: {e}")
