"""
BrokerClient — Abstract Broker + Alpaca Implementation
=======================================================

Provides a unified interface for placing trades, checking positions, and
querying account info.  Two concrete implementations:

  - **AlpacaBroker** — Real broker (paper or live, controlled by base URL)
  - **DryRunBroker** — Logs orders locally, never touches a broker

API keys are read from environment variables — never hardcoded.
"""

from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

# ── Load .env ────────────────────────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
except ImportError:
    pass


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class Position:
    symbol: str
    qty: float
    side: str             # 'long' or 'short'
    avgEntryPrice: float
    currentPrice: float
    marketValue: float
    unrealizedPL: float
    unrealizedPLPct: float


@dataclass
class OrderResult:
    orderId: str
    symbol: str
    side: str             # 'buy' or 'sell'
    qty: float
    orderType: str        # 'market', 'limit'
    limitPrice: Optional[float]
    status: str           # 'submitted', 'filled', 'rejected', 'dry_run'
    filledPrice: Optional[float]
    filledAt: Optional[str]
    message: str = ''


@dataclass
class BrokerOrder:
    orderId: str
    symbol: str
    side: str             # 'buy' or 'sell'
    qty: float
    orderType: str        # 'market', 'limit'
    filledQty: float
    filledAvgPrice: Optional[float]
    status: str           # 'new', 'partially_filled', 'filled', 'canceled', etc.
    createdAt: Optional[str] = None
    limitPrice: Optional[float] = None


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buyingPower: float
    portfolioValue: float
    dayPL: float
    dayPLPct: float


# =====================================================================
# Abstract base
# =====================================================================

class BrokerClient(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
    ) -> OrderResult:
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        ...

    @abstractmethod
    def cancel_all_orders(self) -> int:
        ...

    @abstractmethod
    def cancel_order(self, orderId: str) -> bool:
        """Cancel a single order by ID.  Returns True if successfully canceled."""
        ...

    @abstractmethod
    def get_orders(self, status: str = 'open') -> List[BrokerOrder]:
        """Return orders from the broker.  status: 'open', 'closed', 'all'."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return 'paper', 'live', or 'dry_run'."""
        ...


# =====================================================================
# Alpaca implementation (alpaca-py SDK)
# =====================================================================

class AlpacaBroker(BrokerClient):
    """
    Alpaca broker integration using the alpaca-py SDK.

    Mode is determined by ALPACA_BASE_URL or paper=True:
      - Paper: https://paper-api.alpaca.markets  (default)
      - Live:  https://api.alpaca.markets

    Required env vars:
      - ALPACA_API_KEY
      - ALPACA_SECRET_KEY
      - ALPACA_BASE_URL  (optional, paper if URL contains 'paper')

    Install: pip install alpaca-py
    """

    PAPER_URL = 'https://paper-api.alpaca.markets'
    LIVE_URL  = 'https://api.alpaca.markets'

    def __init__(
        self,
        apiKey: Optional[str] = None,
        secretKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
    ):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError(
                'alpaca-py is required. Install with: pip install alpaca-py'
            )

        self._apiKey   = apiKey or os.environ.get('ALPACA_API_KEY', '')
        self._secret   = secretKey or os.environ.get('ALPACA_SECRET_KEY', '')
        self._baseUrl  = baseUrl or os.environ.get('ALPACA_BASE_URL', self.PAPER_URL)
        self._paper    = 'paper' in str(self._baseUrl).lower()

        if not self._apiKey or not self._secret:
            raise ValueError(
                'Alpaca credentials missing. Set ALPACA_API_KEY and '
                'ALPACA_SECRET_KEY in .env or pass them directly.'
            )

        self._client = TradingClient(
            self._apiKey, self._secret, paper=self._paper
        )
        self._mode = 'paper' if self._paper else 'live'

    @property
    def mode(self) -> str:
        return self._mode

    def get_account(self) -> AccountInfo:
        a = self._client.get_account()
        eq = float(a.equity or 0)
        last_eq = float(getattr(a, 'last_equity', eq) or eq)
        return AccountInfo(
            equity=eq,
            cash=float(a.cash or 0),
            buyingPower=float(a.buying_power or 0),
            portfolioValue=float(getattr(a, 'portfolio_value', eq) or eq),
            dayPL=eq - last_eq,
            dayPLPct=((eq - last_eq) / last_eq * 100) if last_eq > 0 else 0,
        )

    def get_positions(self) -> List[Position]:
        positions = self._client.get_all_positions()
        return [self._mapPos(p) for p in positions]

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = self._client.get_open_position(symbol)
            return self._mapPos(p)
        except Exception:
            return None

    def get_cash(self) -> float:
        return float(self._client.get_account().cash or 0)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
    ) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif = TimeInForce.DAY if time_in_force.lower() == 'day' else TimeInForce.GTC
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        qty_int = int(qty)

        if order_type == 'limit' and limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty_int,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty_int,
                side=order_side,
                time_in_force=tif,
            )

        try:
            o = self._client.submit_order(order_data=req)
            filled = getattr(o, 'filled_avg_price', None) or getattr(o, 'average_fill_price', None)
            raw_status = getattr(o, 'status', None)
            status = getattr(raw_status, 'value', str(raw_status or 'submitted')).lower()
            return OrderResult(
                orderId=str(o.id) if o.id else '',
                symbol=symbol,
                side=side,
                qty=qty_int,
                orderType=order_type,
                limitPrice=limit_price,
                status=status,
                filledPrice=float(filled) if filled is not None else None,
                filledAt=str(o.filled_at) if getattr(o, 'filled_at', None) else None,
                message='',
            )
        except Exception as e:
            return OrderResult(
                orderId='',
                symbol=symbol,
                side=side,
                qty=qty_int,
                orderType=order_type,
                limitPrice=limit_price,
                status='rejected',
                filledPrice=None,
                filledAt=None,
                message=str(e),
            )

    def close_position(self, symbol: str) -> OrderResult:
        try:
            self._client.close_position(symbol)
            return OrderResult(
                orderId='', symbol=symbol, side='sell', qty=0,
                orderType='market', limitPrice=None,
                status='submitted', filledPrice=None, filledAt=None,
                message=f'Close position for {symbol}',
            )
        except Exception as e:
            return OrderResult(
                orderId='', symbol=symbol, side='sell', qty=0,
                orderType='market', limitPrice=None,
                status='rejected', filledPrice=None, filledAt=None,
                message=str(e),
            )

    def cancel_all_orders(self) -> int:
        cancelled = self._client.cancel_orders()
        return len(cancelled) if cancelled else 0

    def cancel_order(self, orderId: str) -> bool:
        try:
            self._client.cancel_order_by_id(orderId)
            return True
        except Exception:
            return False

    def get_orders(self, status: str = 'open') -> List[BrokerOrder]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        statusMap = {
            'open': QueryOrderStatus.OPEN,
            'closed': QueryOrderStatus.CLOSED,
            'all': QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(status=statusMap.get(status, QueryOrderStatus.OPEN))
        orders = self._client.get_orders(filter=req)
        result = []
        for o in orders:
            side_val = getattr(o, 'side', 'buy')
            side = getattr(side_val, 'value', str(side_val)).lower()
            otype_val = getattr(o, 'order_type', 'market')
            otype = getattr(otype_val, 'value', str(otype_val)).lower()
            status_val = getattr(o, 'status', 'new')
            ostat = getattr(status_val, 'value', str(status_val)).lower()
            result.append(BrokerOrder(
                orderId=str(o.id) if o.id else '',
                symbol=getattr(o, 'symbol', ''),
                side=side,
                qty=float(getattr(o, 'qty', 0) or 0),
                orderType=otype,
                filledQty=float(getattr(o, 'filled_qty', 0) or 0),
                filledAvgPrice=float(o.filled_avg_price) if getattr(o, 'filled_avg_price', None) else None,
                status=ostat,
                createdAt=str(o.created_at) if getattr(o, 'created_at', None) else None,
                limitPrice=float(o.limit_price) if getattr(o, 'limit_price', None) else None,
            ))
        return result

    @staticmethod
    def _mapPos(p) -> Position:
        sym = getattr(p, 'symbol', '') or getattr(p, 'symbol_id', '') or ''
        qty_raw = getattr(p, 'qty', 0) or getattr(p, 'quantity', 0) or 0
        qty = float(qty_raw) if qty_raw else 0.0
        side_val = getattr(p, 'side', 'long')
        side = getattr(side_val, 'value', str(side_val)).lower()
        avg = float(getattr(p, 'avg_entry_price', 0) or getattr(p, 'average_entry_price', 0))
        cur = float(getattr(p, 'current_price', 0) or getattr(p, 'market_value', 0) / qty if qty else 0)
        mv = float(getattr(p, 'market_value', 0) or qty * cur)
        upl = float(getattr(p, 'unrealized_pl', 0) or 0)
        plpc = float(getattr(p, 'unrealized_plpc', 0) or 0) * 100
        return Position(
            symbol=sym,
            qty=qty,
            side=side,
            avgEntryPrice=avg,
            currentPrice=cur,
            marketValue=mv,
            unrealizedPL=upl,
            unrealizedPLPct=plpc,
        )


# =====================================================================
# Dry-run (no broker) — logs orders to a list
# =====================================================================

class DryRunBroker(BrokerClient):
    """
    Simulated broker that keeps positions in memory.  Useful for
    testing the TradingBot without any broker account.
    """

    def __init__(self, startingCash: float = 100_000):
        self._cash = startingCash
        self._startingCash = startingCash
        self._positions: Dict[str, Position] = {}
        self._orderLog: List[OrderResult] = []

    @property
    def mode(self) -> str:
        return 'dry_run'

    def get_account(self) -> AccountInfo:
        posValue = sum(p.marketValue for p in self._positions.values())
        equity = self._cash + posValue
        return AccountInfo(
            equity=equity,
            cash=self._cash,
            buyingPower=self._cash,
            portfolioValue=equity,
            dayPL=equity - self._startingCash,
            dayPLPct=((equity - self._startingCash) / self._startingCash * 100)
                     if self._startingCash > 0 else 0,
        )

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_cash(self) -> float:
        return self._cash

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
    ) -> OrderResult:
        qty = int(qty)
        if qty <= 0:
            return OrderResult(
                orderId='', symbol=symbol, side=side, qty=0,
                orderType=order_type, limitPrice=limit_price,
                status='rejected', filledPrice=None, filledAt=None,
                message='Quantity must be > 0',
            )

        price = limit_price or self._estimatePrice(symbol)
        now = _now().isoformat()

        if side == 'buy':
            cost = price * qty
            if cost > self._cash:
                return OrderResult(
                    orderId='', symbol=symbol, side=side, qty=qty,
                    orderType=order_type, limitPrice=limit_price,
                    status='rejected', filledPrice=None, filledAt=None,
                    message=f'Insufficient cash: need ${cost:,.0f}, have ${self._cash:,.0f}',
                )
            self._cash -= cost
            existing = self._positions.get(symbol)
            if existing and str(getattr(existing, 'side', 'long')).lower() == 'short':
                # Close/reduce short
                closeQty = min(qty, existing.qty)
                existing.qty -= closeQty
                if existing.qty < 0.01:
                    del self._positions[symbol]
                else:
                    existing.currentPrice = price
                    existing.marketValue = existing.qty * price
                if qty > closeQty:
                    # Excess buy opens long
                    extra = qty - closeQty
                    self._positions[symbol] = Position(
                        symbol=symbol, qty=extra, side='long',
                        avgEntryPrice=price, currentPrice=price,
                        marketValue=extra * price, unrealizedPL=0, unrealizedPLPct=0,
                    )
            elif existing:
                # Add to long
                totalQty = existing.qty + qty
                avgPrice = (existing.avgEntryPrice * existing.qty + price * qty) / totalQty
                existing.qty = totalQty
                existing.avgEntryPrice = avgPrice
                existing.currentPrice = price
                existing.marketValue = totalQty * price
                existing.unrealizedPL = 0
            else:
                self._positions[symbol] = Position(
                    symbol=symbol, qty=qty, side='long',
                    avgEntryPrice=price, currentPrice=price,
                    marketValue=qty * price, unrealizedPL=0, unrealizedPLPct=0,
                )

        elif side == 'sell':
            existing = self._positions.get(symbol)
            is_long = existing and str(getattr(existing, 'side', 'long')).lower() == 'long'
            # Close/reduce long
            if is_long and existing.qty >= qty:
                proceeds = price * qty
                self._cash += proceeds
                existing.qty -= qty
                if existing.qty < 0.01:
                    del self._positions[symbol]
                else:
                    existing.marketValue = existing.qty * price
            # Open or add to short (no position, or existing short)
            elif not existing or str(getattr(existing, 'side', 'long')).lower() == 'short':
                proceeds = price * qty
                self._cash += proceeds
                if existing and str(getattr(existing, 'side', 'long')).lower() == 'short':
                    totalQty = existing.qty + qty
                    avgPrice = (existing.avgEntryPrice * existing.qty + price * qty) / totalQty
                    existing.qty = totalQty
                    existing.avgEntryPrice = avgPrice
                    existing.currentPrice = price
                    existing.marketValue = totalQty * price
                    existing.unrealizedPL = 0
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol, qty=qty, side='short',
                        avgEntryPrice=price, currentPrice=price,
                        marketValue=qty * price, unrealizedPL=0, unrealizedPLPct=0,
                    )
            else:
                return OrderResult(
                    orderId='', symbol=symbol, side=side, qty=qty,
                    orderType=order_type, limitPrice=limit_price,
                    status='rejected', filledPrice=None, filledAt=None,
                    message=f'Insufficient long shares for {symbol} (have {existing.qty})',
                )

        result = OrderResult(
            orderId=f'DRY-{len(self._orderLog)+1:04d}',
            symbol=symbol, side=side, qty=qty,
            orderType=order_type, limitPrice=limit_price,
            status='filled', filledPrice=price, filledAt=now,
            message=f'[dry_run] {side.upper()} {qty} {symbol} @ ${price:.2f}',
        )
        self._orderLog.append(result)
        return result

    def close_position(self, symbol: str) -> OrderResult:
        pos = self._positions.get(symbol)
        if not pos:
            return OrderResult(
                orderId='', symbol=symbol, side='sell', qty=0,
                orderType='market', limitPrice=None,
                status='rejected', filledPrice=None, filledAt=None,
                message=f'No position in {symbol}',
            )
        return self.place_order(symbol, 'sell', pos.qty)

    def cancel_all_orders(self) -> int:
        return 0

    def cancel_order(self, orderId: str) -> bool:
        return False

    def get_orders(self, status: str = 'open') -> List[BrokerOrder]:
        return []

    def _estimatePrice(self, symbol: str) -> float:
        try:
            from GeneticAlgorithm import StockDataFetcher
            df = StockDataFetcher().fetchData(symbol, interval='1d', period='5d')
            if df is not None and len(df) > 0:
                return float(df['close'].iloc[-1])
        except Exception:
            pass
        return 100.0  # fallback
