"""
PersistenceManager - Firebase Firestore Cloud Persistence
=========================================================

Gives the pipeline *memory* across runs **and** across devices.  Every
time the system executes, discovered patterns, ML strategies, stock
metadata, and performance metrics are persisted to Google Cloud
Firestore.  On the next run the system loads the best-known patterns
and strategies and only replaces them if the new run produces better
results.

Setup
-----
1. Create a Firebase project at https://console.firebase.google.com
2. Go to Project Settings → Service Accounts → Generate New Private Key
3. Save the JSON file as ``<project_root>/firebase_service_account.json``
   (or set the env var ``FIREBASE_SERVICE_ACCOUNT_PATH``)
4. ``pip install firebase-admin``

Firestore Collections
---------------------
runs               Run-level metadata (timestamp, config hash, total
                   return, alpha vs benchmark, Sharpe ratio).
patterns           Individual GA patterns with MC composite scores,
                   ranked per (symbol, interval).
strategies         AI-generated cross-stock strategies (hedging, sector
                   rotation, pairs, etc.).
stock_metadata     Static per-stock info (sector, industry, market cap
                   bucket, related tickers, description).
cross_stock_rules  Learned cross-stock relationships.

Usage::

    pm = PersistenceManager()              # auto-connects to Firestore
    pm.saveRunResult(runMeta)              # persist a completed run
    best = pm.loadBestPatterns('AAPL','1d')  # ranked patterns
    pm.savePatterns(patternList, runId)    # bulk upsert
    pm.saveStrategies(strategyList)        # from StrategyEngine
    meta = pm.getStockMetadata('AAPL')     # sector, industry, etc.
"""

from __future__ import annotations

import json
import os
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

# Firebase imports — installed with:  pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore


# ═══════════════════════════════════════════════════════════════════════
# Data classes  (unchanged — same interface as the old SQLite version)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Metadata for one complete pipeline execution."""
    runId: str = ''
    timestamp: str = ''
    configHash: str = ''
    totalReturnPct: float = 0.0
    totalFund: float = 0.0           # fund size ($) used for this run
    alphaVsBuyHold: float = 0.0
    alphaVsSP500: float = 0.0
    sharpeRatio: float = 0.0
    winRate: float = 0.0
    numTrades: int = 0
    numPatterns: int = 0
    symbols: List[str] = field(default_factory=list)
    perStockResults: dict = field(default_factory=dict)  # {sym: {returnPct, trades, winRate, profit}}
    configJson: str = '{}'


@dataclass
class StoredPattern:
    """A single GA pattern serialised for database storage."""
    patternId: str = ''
    runId: str = ''
    symbol: str = ''
    interval: str = ''
    patternLength: int = 0
    genesJson: str = '[]'
    fitness: float = 0.0
    accuracy: float = 0.0
    mcCompositeScore: float = 0.0
    mcSharpe: float = 0.0
    mcWinRate: float = 0.0
    mcReturn: float = 0.0
    rank: int = 0                    # 1 = best for this (symbol, interval)
    createdAt: str = ''
    supersededBy: str = ''           # patternId of the replacement (empty = still active)


@dataclass
class StockMeta:
    """Static metadata about a stock."""
    symbol: str = ''
    sector: str = ''
    industry: str = ''
    marketCapBucket: str = ''        # 'mega', 'large', 'mid', 'small'
    description: str = ''
    relatedTickers: List[str] = field(default_factory=list)
    sectorPeers: List[str] = field(default_factory=list)
    supplyChainUp: List[str] = field(default_factory=list)    # suppliers
    supplyChainDown: List[str] = field(default_factory=list)  # customers
    competesWidth: List[str] = field(default_factory=list)
    updatedAt: str = ''
    inPortfolio: bool = False
    portfolioType: str = ''          # 'manual' or 'automatic'
    autoAddedReason: str = ''        # why the stock was auto-added
    autoAddedFrom: List[str] = field(default_factory=list)  # parent stocks that triggered auto-add
    autoAddedAt: str = ''
    autoScore: float = 0.0           # ML confidence that trading this stock is beneficial


@dataclass
class CrossStockRule:
    """A learned relationship between stocks."""
    ruleId: str = ''
    ruleType: str = ''          # 'sector_correlation', 'lead_lag', 'hedge_pair', 'pairs_trade', 'supply_chain'
    sourceSymbols: List[str] = field(default_factory=list)
    targetSymbols: List[str] = field(default_factory=list)
    conditionJson: str = '{}'   # {"direction": "up", "threshold": 0.02, "timeframe": "5d"}
    actionJson: str = '{}'      # {"signal": "BUY", "confidence": 0.7, "reason": "..."}
    confidence: float = 0.0
    hitRate: float = 0.0        # historical accuracy of this rule
    sampleSize: int = 0
    source: str = ''            # 'openai', 'statistical', 'manual'
    createdAt: str = ''


@dataclass
class StoredStrategy:
    """An AI-generated or statistically-discovered trading strategy."""
    strategyId: str = ''
    name: str = ''
    description: str = ''
    strategyType: str = ''       # 'hedge', 'pairs_trade', 'sector_rotation', 'momentum', 'mean_reversion'
    conditionsJson: str = '[]'   # list of condition dicts
    actionsJson: str = '[]'      # list of action dicts
    symbols: List[str] = field(default_factory=list)
    confidence: float = 0.0
    backtestReturn: float = 0.0
    backtestSharpe: float = 0.0
    source: str = ''             # 'openai', 'ml_discovered', 'rule_based'
    active: bool = True
    createdAt: str = ''
    updatedAt: str = ''


# ═══════════════════════════════════════════════════════════════════════
# Built-in stock metadata (seeded — avoids API call for common stocks)
# ═══════════════════════════════════════════════════════════════════════

_BUILTIN_STOCK_METADATA: Dict[str, Dict] = {
    'AAPL': {
        'sector': 'Technology', 'industry': 'Consumer Electronics',
        'marketCapBucket': 'mega', 'description': 'Apple Inc. — iPhone, Mac, Services, Wearables.',
        'relatedTickers': ['MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
        'sectorPeers': ['MSFT', 'GOOGL', 'META', 'AMZN', 'CRM', 'ORCL'],
        'supplyChainUp': ['TSM', 'QCOM', 'AVGO', 'MU', 'SWKS', 'LRCX'],
        'supplyChainDown': ['T', 'VZ', 'TMUS', 'BBY'],
        'competesWidth': ['MSFT', 'GOOGL', 'SAMSUNG', 'DELL', 'HPQ'],
    },
    'MSFT': {
        'sector': 'Technology', 'industry': 'Software — Infrastructure',
        'marketCapBucket': 'mega', 'description': 'Microsoft Corp. — Azure, Office 365, Windows, Gaming.',
        'relatedTickers': ['AAPL', 'GOOGL', 'AMZN', 'META', 'CRM'],
        'sectorPeers': ['AAPL', 'GOOGL', 'CRM', 'ORCL', 'SAP', 'ADBE'],
        'supplyChainUp': ['NVDA', 'AMD', 'INTC', 'TSM'],
        'supplyChainDown': ['DELL', 'HPQ', 'CSCO'],
        'competesWidth': ['GOOGL', 'AMZN', 'CRM', 'ORCL', 'SAP'],
    },
    'GOOGL': {
        'sector': 'Technology', 'industry': 'Internet Content & Information',
        'marketCapBucket': 'mega', 'description': 'Alphabet Inc. — Google Search, YouTube, Cloud, Waymo.',
        'relatedTickers': ['AAPL', 'MSFT', 'META', 'AMZN', 'NVDA'],
        'sectorPeers': ['META', 'AMZN', 'SNAP', 'PINS', 'TTD'],
        'supplyChainUp': ['NVDA', 'AMD', 'TSM', 'INTC'],
        'supplyChainDown': ['WPP', 'OMC'],
        'competesWidth': ['META', 'AMZN', 'MSFT', 'AAPL'],
    },
    'AMZN': {
        'sector': 'Technology', 'industry': 'Internet Retail / Cloud',
        'marketCapBucket': 'mega', 'description': 'Amazon.com Inc. — E-commerce, AWS, Advertising.',
        'relatedTickers': ['MSFT', 'GOOGL', 'META', 'SHOP', 'WMT'],
        'sectorPeers': ['MSFT', 'GOOGL', 'CRM', 'SHOP'],
        'supplyChainUp': ['NVDA', 'AMD', 'INTC'],
        'supplyChainDown': ['UPS', 'FDX', 'SHOP'],
        'competesWidth': ['MSFT', 'GOOGL', 'WMT', 'SHOP'],
    },
    'META': {
        'sector': 'Technology', 'industry': 'Internet Content & Information',
        'marketCapBucket': 'mega', 'description': 'Meta Platforms Inc. — Facebook, Instagram, WhatsApp, Reality Labs.',
        'relatedTickers': ['GOOGL', 'SNAP', 'PINS', 'TTD', 'MSFT'],
        'sectorPeers': ['GOOGL', 'SNAP', 'PINS', 'TTD'],
        'supplyChainUp': ['NVDA', 'AMD', 'TSM'],
        'supplyChainDown': ['WPP', 'OMC'],
        'competesWidth': ['GOOGL', 'SNAP', 'TIKTOK', 'PINS'],
    },
    'NVDA': {
        'sector': 'Technology', 'industry': 'Semiconductors',
        'marketCapBucket': 'mega', 'description': 'NVIDIA Corp. — GPUs, AI accelerators, data-center.',
        'relatedTickers': ['AMD', 'INTC', 'TSM', 'AVGO', 'MU'],
        'sectorPeers': ['AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU'],
        'supplyChainUp': ['TSM', 'ASML', 'LRCX', 'AMAT'],
        'supplyChainDown': ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'],
        'competesWidth': ['AMD', 'INTC', 'AVGO'],
    },
    'TSLA': {
        'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers',
        'marketCapBucket': 'mega', 'description': 'Tesla Inc. — EVs, Energy Storage, Autonomous Driving.',
        'relatedTickers': ['RIVN', 'LCID', 'NIO', 'F', 'GM'],
        'sectorPeers': ['F', 'GM', 'TM', 'RIVN', 'LCID'],
        'supplyChainUp': ['PANASONIC', 'ALB', 'LTHM', 'SQM'],
        'supplyChainDown': [],
        'competesWidth': ['F', 'GM', 'TM', 'RIVN', 'BYD'],
    },
    'JPM': {
        'sector': 'Financial Services', 'industry': 'Banks — Diversified',
        'marketCapBucket': 'mega', 'description': 'JPMorgan Chase & Co. — Investment banking, retail banking.',
        'relatedTickers': ['BAC', 'GS', 'MS', 'WFC', 'C'],
        'sectorPeers': ['BAC', 'GS', 'MS', 'WFC', 'C'],
        'supplyChainUp': [],
        'supplyChainDown': [],
        'competesWidth': ['BAC', 'GS', 'MS', 'WFC'],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# PersistenceManager  (Firestore Implementation)
# ═══════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """
    Cloud Firestore persistence layer for the portfolio pipeline.

    On first instantiation it initialises the Firebase Admin SDK using
    a service-account JSON file.  The default location is
    ``<project_root>/firebase_service_account.json``; override via the
    ``FIREBASE_SERVICE_ACCOUNT_PATH`` environment variable.

    Parameters
    ----------
    serviceAccountPath : str, optional
        Explicit path to the Firebase service-account JSON key.
        If ``None``, checks env var ``FIREBASE_SERVICE_ACCOUNT_PATH``,
        then falls back to ``<project_root>/firebase_service_account.json``.
    projectId : str, optional
        Explicit GCP project ID.  Usually auto-detected from the
        service-account JSON.
    """

    # Collection names (change these if you want a different Firestore layout)
    _COL_RUNS            = 'runs'
    _COL_PATTERNS        = 'patterns'
    _COL_STRATEGIES      = 'strategies'
    _COL_STOCK_META      = 'stock_metadata'
    _COL_CROSS_RULES     = 'cross_stock_rules'

    def __init__(self, serviceAccountPath: Optional[str] = None,
                 projectId: Optional[str] = None):

        # ── Resolve service-account path ──────────────────────────────
        if serviceAccountPath is None:
            serviceAccountPath = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
        if serviceAccountPath is None:
            baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            serviceAccountPath = os.path.join(baseDir, 'protfoliomanagerv2-firebase-adminsdk-fbsvc-9fbdf566d1.json')

        if not os.path.isfile(serviceAccountPath):
            raise FileNotFoundError(
                f"Firebase service-account key not found at: {serviceAccountPath}\n"
                f"Download it from Firebase Console → Project Settings → Service Accounts → "
                f"Generate New Private Key.\n"
                f"Save the JSON file to your project root as 'firebase_service_account.json' "
                f"or set the FIREBASE_SERVICE_ACCOUNT_PATH environment variable."
            )

        # ── Initialise Firebase (only once per process) ───────────────
        if not firebase_admin._apps:
            cred = credentials.Certificate(serviceAccountPath)
            opts = {}
            if projectId:
                opts['projectId'] = projectId
            firebase_admin.initialize_app(cred, opts)

        self.db = firestore.client()
        self._seedStockMetadata()

    # ──────────────────────────────────────────────────────────────────
    # Seed built-in stock metadata
    # ──────────────────────────────────────────────────────────────────

    def _seedStockMetadata(self):
        """Populate stock_metadata from built-in data if not already present."""
        now = _now().isoformat()
        col = self.db.collection(self._COL_STOCK_META)
        for sym, info in _BUILTIN_STOCK_METADATA.items():
            doc = col.document(sym).get()
            if not doc.exists:
                col.document(sym).set({
                    'symbol': sym,
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'marketCapBucket': info.get('marketCapBucket', ''),
                    'description': info.get('description', ''),
                    'relatedTickers': info.get('relatedTickers', []),
                    'sectorPeers': info.get('sectorPeers', []),
                    'supplyChainUp': info.get('supplyChainUp', []),
                    'supplyChainDown': info.get('supplyChainDown', []),
                    'competesWidth': info.get('competesWidth', []),
                    'updatedAt': now,
                })

    # ──────────────────────────────────────────────────────────────────
    # Run management
    # ──────────────────────────────────────────────────────────────────

    def generateRunId(self) -> str:
        now = _now()
        raw = f"{now.isoformat()}-{os.getpid()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def saveRunResult(self, result: RunResult):
        """Persist a completed pipeline run."""
        runId = result.runId or self.generateRunId()
        doc = {
            'runId': runId,
            'timestamp': result.timestamp or _now().isoformat(),
            'configHash': result.configHash,
            'totalReturnPct': result.totalReturnPct,
            'totalFund': result.totalFund,
            'alphaVsBuyHold': result.alphaVsBuyHold,
            'alphaVsSP500': result.alphaVsSP500,
            'sharpeRatio': result.sharpeRatio,
            'winRate': result.winRate,
            'numTrades': result.numTrades,
            'numPatterns': result.numPatterns,
            'symbols': result.symbols,
            'perStockResults': result.perStockResults,
            'configJson': result.configJson,
        }
        self.db.collection(self._COL_RUNS).document(runId).set(doc)

    # ──────────────────────────────────────────────────────────────────
    # Live allocation (for TradingBot — from DynamicAllocator)
    # ──────────────────────────────────────────────────────────────────

    def saveLiveAllocation(
        self,
        slotAllocations: Dict[str, float],
        ghostSlots: List[str],
        stockAllocations: Dict[str, float],
        runId: str = '',
        slotPerformances: Optional[Dict[str, dict]] = None,
    ):
        """
        Persist the allocation result from the pipeline so the TradingBot
        can use the same intelligent fund allocation (including ghost slots).
        """
        doc = {
            'slotAllocations': slotAllocations,
            'ghostSlots': ghostSlots,
            'stockAllocations': stockAllocations,
            'runId': runId,
            'timestamp': _now().isoformat(),
        }
        if slotPerformances:
            # Flatten SlotPerformance for JSON (ruleScore, recentReturnPct, etc.)
            flat = {}
            for k, v in slotPerformances.items():
                if hasattr(v, '__dict__'):
                    flat[k] = {f: getattr(v, f, None) for f in
                               ('totalTrades', 'recentTrades', 'recentReturnPct', 'recentWinRate',
                                'ruleScore', 'sharpeRatio', 'recentSharpe', 'isGhost')}
                elif isinstance(v, dict):
                    flat[k] = v
            doc['slotPerformances'] = flat
        self.db.collection('app_config').document('live_allocation').set(doc, merge=True)

    def loadLiveAllocation(self) -> Optional[Dict]:
        """
        Load the last saved allocation from the pipeline. Returns dict with
        slotAllocations, ghostSlots, stockAllocations, runId, timestamp.
        """
        doc = self.db.collection('app_config').document('live_allocation').get()
        if not doc.exists:
            return None
        return doc.to_dict()

    def appendLiveSlotTrade(self, slotStr: str, trade: dict):
        """Append a live or shadow trade to the slot history (for restore logic)."""
        col = self.db.collection('live_slot_trades')
        col.add({
            'slotStr': slotStr,
            **trade,
            'createdAt': _now().isoformat(),
        })

    def loadLiveSlotTrades(self, slotStr: str, limit: int = 500) -> List[dict]:
        """Load recent trades for a slot (for computing SlotPerformance)."""
        docs = (
            self.db.collection('live_slot_trades')
            .where('slotStr', '==', slotStr)
            .limit(limit * 2)  # fetch extra, sort client-side
            .stream()
        )
        trades = [d.to_dict() for d in docs]
        trades.sort(key=lambda t: t.get('createdAt', ''), reverse=True)
        return trades[:limit]

    def saveShadowPositions(self, positions: Dict[str, dict]):
        """Persist open shadow positions for ghost slots."""
        self.db.collection('app_config').document('shadow_positions').set(
            {'positions': positions, 'updatedAt': _now().isoformat()},
            merge=True,
        )

    def loadShadowPositions(self) -> Dict[str, dict]:
        """Load open shadow positions."""
        doc = self.db.collection('app_config').document('shadow_positions').get()
        if not doc.exists:
            return {}
        data = doc.to_dict() or {}
        return data.get('positions', {})

    def saveDailyTradeSummary(self, summary: dict):
        """Save daily trade summary for next day display (website)."""
        self.db.collection('app_config').document('daily_trade_summary').set(
            {**summary, 'updatedAt': _now().isoformat()},
            merge=True,
        )

    def loadDailyTradeSummary(self) -> Optional[dict]:
        """Load the last saved daily trade summary."""
        doc = self.db.collection('app_config').document('daily_trade_summary').get()
        if not doc.exists:
            return None
        return doc.to_dict()

    # ──────────────────────────────────────────────────────────────────
    # Daily reviews (DailyReviewEngine)
    # ──────────────────────────────────────────────────────────────────

    _COL_DAILY_REVIEWS = 'daily_reviews'

    def saveDailyReview(self, date: str, report: dict):
        """Save a daily performance review keyed by date (YYYY-MM-DD)."""
        self.db.collection(self._COL_DAILY_REVIEWS).document(date).set(
            {**report, 'updatedAt': _now().isoformat()},
        )

    def loadDailyReview(self, date: str) -> Optional[dict]:
        """Load a specific day's review."""
        doc = self.db.collection(self._COL_DAILY_REVIEWS).document(date).get()
        if not doc.exists:
            return None
        return doc.to_dict()

    def loadRecentDailyReviews(self, limit: int = 7) -> List[dict]:
        """Load the N most recent daily reviews, newest first."""
        docs = (
            self.db.collection(self._COL_DAILY_REVIEWS)
            .order_by('date', direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs]

    # ──────────────────────────────────────────────────────────────────
    # Pattern live accuracy tracking
    # ──────────────────────────────────────────────────────────────────

    def trackPatternTrigger(self, patternId: str, hit: bool):
        """Increment trigger count (and hit count if hit=True) on a pattern."""
        ref = self.db.collection(self._COL_PATTERNS).document(patternId)
        ref.update({
            'liveTriggers': firestore.Increment(1),
            **(
                {'liveHits': firestore.Increment(1)} if hit else {}
            ),
            'liveLastTriggered': _now().isoformat(),
        })

    def getPatternLiveAccuracy(self, patternId: str) -> Tuple[int, int]:
        """Return (triggers, hits) for a pattern's live trading accuracy."""
        doc = self.db.collection(self._COL_PATTERNS).document(patternId).get()
        if not doc.exists:
            return (0, 0)
        d = doc.to_dict()
        return (d.get('liveTriggers', 0), d.get('liveHits', 0))

    def supersedePattern(self, patternId: str, reason: str = ''):
        """Mark a pattern as superseded (inactive)."""
        ref = self.db.collection(self._COL_PATTERNS).document(patternId)
        ref.update({
            'supersededBy': reason or 'daily_review',
            'supersededAt': _now().isoformat(),
        })

    def loadBestRun(self) -> Optional[RunResult]:
        """Load the run with the highest alpha vs buy-and-hold."""
        docs = (self.db.collection(self._COL_RUNS)
                .order_by('alphaVsBuyHold', direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream())
        for doc in docs:
            return self._docToRunResult(doc.to_dict())
        return None

    def loadAllRuns(self, limit: int = 50) -> List[RunResult]:
        docs = (self.db.collection(self._COL_RUNS)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
        return [self._docToRunResult(d.to_dict()) for d in docs]

    def _docToRunResult(self, d: dict) -> RunResult:
        return RunResult(
            runId=d.get('runId', ''),
            timestamp=d.get('timestamp', ''),
            configHash=d.get('configHash', ''),
            totalReturnPct=d.get('totalReturnPct', 0),
            totalFund=d.get('totalFund', 0),
            alphaVsBuyHold=d.get('alphaVsBuyHold', 0),
            alphaVsSP500=d.get('alphaVsSP500', 0),
            sharpeRatio=d.get('sharpeRatio', 0),
            winRate=d.get('winRate', 0),
            numTrades=d.get('numTrades', 0),
            numPatterns=d.get('numPatterns', 0),
            symbols=d.get('symbols', []),
            perStockResults=d.get('perStockResults', {}),
            configJson=d.get('configJson', '{}'),
        )

    # ──────────────────────────────────────────────────────────────────
    # Pattern management
    # ──────────────────────────────────────────────────────────────────

    def savePattern(self, pattern: StoredPattern):
        """Save or update a single pattern."""
        patId = pattern.patternId or self._generatePatternId(pattern)
        pattern.patternId = patId
        doc = {
            'patternId': patId,
            'runId': pattern.runId,
            'symbol': pattern.symbol,
            'interval': pattern.interval,
            'patternLength': pattern.patternLength,
            'genesJson': pattern.genesJson,
            'fitness': pattern.fitness,
            'accuracy': pattern.accuracy,
            'mcCompositeScore': pattern.mcCompositeScore,
            'mcSharpe': pattern.mcSharpe,
            'mcWinRate': pattern.mcWinRate,
            'mcReturn': pattern.mcReturn,
            'rank': pattern.rank,
            'createdAt': pattern.createdAt or _now().isoformat(),
            'supersededBy': pattern.supersededBy,
        }
        self.db.collection(self._COL_PATTERNS).document(patId).set(doc)

    def _generatePatternId(self, pattern: StoredPattern) -> str:
        """Deterministic ID from genes hash + symbol + interval."""
        raw = f"{pattern.symbol}_{pattern.interval}_{pattern.genesJson}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def savePatterns(self, patterns: List[StoredPattern]):
        """Bulk save patterns using batched writes."""
        batch = self.db.batch()
        col = self.db.collection(self._COL_PATTERNS)
        for i, p in enumerate(patterns):
            patId = p.patternId or self._generatePatternId(p)
            p.patternId = patId
            doc = {
                'patternId': patId,
                'runId': p.runId,
                'symbol': p.symbol,
                'interval': p.interval,
                'patternLength': p.patternLength,
                'genesJson': p.genesJson,
                'fitness': p.fitness,
                'accuracy': p.accuracy,
                'mcCompositeScore': p.mcCompositeScore,
                'mcSharpe': p.mcSharpe,
                'mcWinRate': p.mcWinRate,
                'mcReturn': p.mcReturn,
                'rank': p.rank,
                'createdAt': p.createdAt or _now().isoformat(),
                'supersededBy': p.supersededBy,
            }
            batch.set(col.document(patId), doc)
            # Firestore batch limit is 500 writes
            if (i + 1) % 450 == 0:
                batch.commit()
                batch = self.db.batch()
        batch.commit()

    def loadBestPatterns(self, symbol: str, interval: str,
                         topN: int = 25) -> List[StoredPattern]:
        """Load the top-N active patterns for (symbol, interval) ranked by MC score.
        Falls back to client-side sort if the composite index is missing."""
        try:
            docs = (self.db.collection(self._COL_PATTERNS)
                    .where('symbol', '==', symbol)
                    .where('interval', '==', interval)
                    .where('supersededBy', '==', '')
                    .order_by('mcCompositeScore', direction=firestore.Query.DESCENDING)
                    .limit(topN)
                    .stream())
            return [self._docToPattern(d.to_dict()) for d in docs]
        except Exception:
            docs = (self.db.collection(self._COL_PATTERNS)
                    .where('symbol', '==', symbol)
                    .where('interval', '==', interval)
                    .where('supersededBy', '==', '')
                    .stream())
            patterns = [self._docToPattern(d.to_dict()) for d in docs]
            patterns.sort(key=lambda p: p.mcCompositeScore, reverse=True)
            return patterns[:topN]

    def loadAllActivePatterns(self, symbol: str,
                               topN: int = 100) -> List[StoredPattern]:
        """All active patterns for a symbol across all intervals.
        Falls back to client-side sort if the composite index is missing."""
        try:
            docs = (self.db.collection(self._COL_PATTERNS)
                    .where('symbol', '==', symbol)
                    .where('supersededBy', '==', '')
                    .order_by('mcCompositeScore', direction=firestore.Query.DESCENDING)
                    .limit(topN)
                    .stream())
            return [self._docToPattern(d.to_dict()) for d in docs]
        except Exception:
            # Composite index missing — fall back to simpler query + client sort
            docs = (self.db.collection(self._COL_PATTERNS)
                    .where('symbol', '==', symbol)
                    .where('supersededBy', '==', '')
                    .stream())
            patterns = [self._docToPattern(d.to_dict()) for d in docs]
            patterns.sort(key=lambda p: p.mcCompositeScore, reverse=True)
            return patterns[:topN]

    def updatePatternRanksIfBetter(self, newPatterns: List[StoredPattern],
                                    symbol: str, interval: str):
        """
        Compare new patterns against stored ones.  If a new pattern's
        MC composite score exceeds the worst stored active pattern,
        insert it and mark the worst as superseded.

        Implements ranked replacement: better patterns push out weaker
        ones automatically.
        """
        existing = self.loadBestPatterns(symbol, interval, topN=999)

        for newP in newPatterns:
            # Ensure pattern has an ID
            if not newP.patternId:
                newP.patternId = self._generatePatternId(newP)

            # Check if this exact pattern (by genes hash) already exists
            genesHash = hashlib.md5(newP.genesJson.encode()).hexdigest()[:12]
            dupeFound = False
            for ep in existing:
                epHash = hashlib.md5(ep.genesJson.encode()).hexdigest()[:12]
                if epHash == genesHash:
                    # Same genes — update if better score
                    if newP.mcCompositeScore > ep.mcCompositeScore:
                        ep.supersededBy = newP.patternId
                        self.savePattern(ep)
                        self.savePattern(newP)
                    dupeFound = True
                    break

            if not dupeFound:
                # Find worst existing pattern
                if existing and len(existing) >= 25:
                    worst = min(existing, key=lambda p: p.mcCompositeScore)
                    if newP.mcCompositeScore > worst.mcCompositeScore:
                        worst.supersededBy = newP.patternId
                        self.savePattern(worst)
                self.savePattern(newP)

    def getPatternCount(self, symbol: Optional[str] = None) -> int:
        query = self.db.collection(self._COL_PATTERNS).where('supersededBy', '==', '')
        if symbol:
            query = query.where('symbol', '==', symbol)
        # Use aggregation if available, else count docs
        try:
            result = query.count().get()
            return result[0][0].value
        except Exception:
            # Fallback: stream and count
            return sum(1 for _ in query.stream())

    def _docToPattern(self, d: dict) -> StoredPattern:
        return StoredPattern(
            patternId=d.get('patternId', ''),
            runId=d.get('runId', ''),
            symbol=d.get('symbol', ''),
            interval=d.get('interval', ''),
            patternLength=d.get('patternLength', 0),
            genesJson=d.get('genesJson', '[]'),
            fitness=d.get('fitness', 0),
            accuracy=d.get('accuracy', 0),
            mcCompositeScore=d.get('mcCompositeScore', 0),
            mcSharpe=d.get('mcSharpe', 0),
            mcWinRate=d.get('mcWinRate', 0),
            mcReturn=d.get('mcReturn', 0),
            rank=d.get('rank', 0),
            createdAt=d.get('createdAt', ''),
            supersededBy=d.get('supersededBy', ''),
        )

    # ──────────────────────────────────────────────────────────────────
    # Strategy management
    # ──────────────────────────────────────────────────────────────────

    def saveStrategy(self, strat: StoredStrategy):
        stratId = strat.strategyId or hashlib.md5(
            f"{strat.name}_{strat.strategyType}_{strat.source}".encode()
        ).hexdigest()[:16]
        doc = {
            'strategyId': stratId,
            'name': strat.name,
            'description': strat.description,
            'strategyType': strat.strategyType,
            'conditionsJson': strat.conditionsJson,
            'actionsJson': strat.actionsJson,
            'symbols': strat.symbols,
            'confidence': strat.confidence,
            'backtestReturn': strat.backtestReturn,
            'backtestSharpe': strat.backtestSharpe,
            'source': strat.source,
            'active': strat.active,
            'createdAt': strat.createdAt or _now().isoformat(),
            'updatedAt': strat.updatedAt or _now().isoformat(),
        }
        self.db.collection(self._COL_STRATEGIES).document(stratId).set(doc)

    def saveStrategies(self, strategies: List[StoredStrategy]):
        for s in strategies:
            self.saveStrategy(s)

    def loadActiveStrategies(self, strategyType: Optional[str] = None,
                              symbols: Optional[List[str]] = None,
                              ) -> List[StoredStrategy]:
        query = (self.db.collection(self._COL_STRATEGIES)
                 .where('active', '==', True))
        if strategyType:
            query = query.where('strategyType', '==', strategyType)
        docs = query.stream()
        results = [self._docToStrategy(d.to_dict()) for d in docs]
        if symbols:
            symSet = set(symbols)
            results = [s for s in results
                       if not s.symbols or bool(set(s.symbols) & symSet)]
        return results

    def getStrategyCount(self) -> int:
        try:
            result = (self.db.collection(self._COL_STRATEGIES)
                      .where('active', '==', True)
                      .count().get())
            return result[0][0].value
        except Exception:
            return sum(1 for _ in self.db.collection(self._COL_STRATEGIES)
                       .where('active', '==', True).stream())

    def _docToStrategy(self, d: dict) -> StoredStrategy:
        return StoredStrategy(
            strategyId=d.get('strategyId', ''),
            name=d.get('name', ''),
            description=d.get('description', ''),
            strategyType=d.get('strategyType', ''),
            conditionsJson=d.get('conditionsJson', '[]'),
            actionsJson=d.get('actionsJson', '[]'),
            symbols=d.get('symbols', []),
            confidence=d.get('confidence', 0),
            backtestReturn=d.get('backtestReturn', 0),
            backtestSharpe=d.get('backtestSharpe', 0),
            source=d.get('source', ''),
            active=d.get('active', True),
            createdAt=d.get('createdAt', ''),
            updatedAt=d.get('updatedAt', ''),
        )

    # ──────────────────────────────────────────────────────────────────
    # Stock metadata
    # ──────────────────────────────────────────────────────────────────

    def getStockMetadata(self, symbol: str) -> Optional[StockMeta]:
        doc = self.db.collection(self._COL_STOCK_META).document(symbol).get()
        if not doc.exists:
            return None
        return self._docToStockMeta(doc.to_dict())

    def getAllStockMetadata(self) -> Dict[str, StockMeta]:
        docs = self.db.collection(self._COL_STOCK_META).stream()
        result = {}
        for doc in docs:
            d = doc.to_dict()
            sym = d.get('symbol', doc.id)
            result[sym] = self._docToStockMeta(d)
        return result

    def upsertStockMetadata(self, meta: StockMeta):
        now = _now().isoformat()
        doc = {
            'symbol': meta.symbol,
            'sector': meta.sector,
            'industry': meta.industry,
            'marketCapBucket': meta.marketCapBucket,
            'description': meta.description,
            'relatedTickers': meta.relatedTickers,
            'sectorPeers': meta.sectorPeers,
            'supplyChainUp': meta.supplyChainUp,
            'supplyChainDown': meta.supplyChainDown,
            'competesWidth': meta.competesWidth,
            'updatedAt': now,
        }
        # Use merge=True so this never overwrites rich fields written by stock_research
        # (e.g. description, competitors, financialSnapshot, analystConsensus, etc.)
        self.db.collection(self._COL_STOCK_META).document(meta.symbol).set(doc, merge=True)

    def getPortfolioStocks(self, portfolioType: Optional[str] = None) -> Dict[str, StockMeta]:
        """Load stocks that are in the portfolio, optionally filtered by type."""
        query = (self.db.collection(self._COL_STOCK_META)
                 .where('inPortfolio', '==', True))
        if portfolioType:
            query = query.where('portfolioType', '==', portfolioType)
        docs = query.stream()
        result = {}
        for doc in docs:
            d = doc.to_dict()
            sym = d.get('symbol', doc.id)
            result[sym] = self._docToStockMeta(d)
        return result

    def getManualPortfolioStocks(self) -> Dict[str, StockMeta]:
        return self.getPortfolioStocks('manual')

    def getAutoPortfolioStocks(self) -> Dict[str, StockMeta]:
        return self.getPortfolioStocks('automatic')

    def addAutoStock(self, symbol: str, reason: str, parentSymbols: List[str],
                     score: float = 0.0):
        """Add a stock to the automatic portfolio (connected stock)."""
        now = _now().isoformat()
        self.db.collection(self._COL_STOCK_META).document(symbol).set({
            'symbol': symbol,
            'inPortfolio': True,
            'portfolioType': 'automatic',
            'autoAddedReason': reason,
            'autoAddedFrom': parentSymbols,
            'autoAddedAt': now,
            'autoScore': score,
            'updatedAt': now,
        }, merge=True)

    def removeAutoStock(self, symbol: str):
        """Remove a stock from the automatic portfolio."""
        self.db.collection(self._COL_STOCK_META).document(symbol).update({
            'inPortfolio': False,
            'portfolioType': '',
            'autoScore': 0.0,
            'autoRemovedAt': _now().isoformat(),
        })

    def updateAutoStockScore(self, symbol: str, score: float, reason: str = ''):
        """Update the benefit score for an automatic stock."""
        update = {'autoScore': score, 'updatedAt': _now().isoformat()}
        if reason:
            update['autoAddedReason'] = reason
        self.db.collection(self._COL_STOCK_META).document(symbol).update(update)

    def _docToStockMeta(self, d: dict) -> StockMeta:
        # Stock research writes to 'suppliers'/'customers'/'competitors',
        # while seed data and StrategyEngine write to 'supplyChainUp'/
        # 'supplyChainDown'/'competesWidth'.  Merge both naming conventions
        # so ConnectedStockManager always finds the data.
        supplyUp   = d.get('supplyChainUp', []) or d.get('suppliers', []) or []
        supplyDown = d.get('supplyChainDown', []) or d.get('customers', []) or []
        competes   = d.get('competesWidth', []) or d.get('competitors', []) or []

        return StockMeta(
            symbol=d.get('symbol', ''),
            sector=d.get('sector', ''),
            industry=d.get('industry', ''),
            marketCapBucket=d.get('marketCapBucket', ''),
            description=d.get('description', ''),
            relatedTickers=d.get('relatedTickers', []),
            sectorPeers=d.get('sectorPeers', []),
            supplyChainUp=supplyUp,
            supplyChainDown=supplyDown,
            competesWidth=competes,
            updatedAt=d.get('updatedAt', ''),
            inPortfolio=d.get('inPortfolio', False),
            portfolioType=d.get('portfolioType', ''),
            autoAddedReason=d.get('autoAddedReason', ''),
            autoAddedFrom=d.get('autoAddedFrom', []),
            autoAddedAt=d.get('autoAddedAt', ''),
            autoScore=d.get('autoScore', 0.0),
        )

    # ──────────────────────────────────────────────────────────────────
    # Cross-stock rules
    # ──────────────────────────────────────────────────────────────────

    def saveCrossStockRule(self, rule: CrossStockRule):
        ruleId = rule.ruleId or hashlib.md5(
            f"{rule.ruleType}_{rule.sourceSymbols}_{rule.targetSymbols}".encode()
        ).hexdigest()[:16]
        doc = {
            'ruleId': ruleId,
            'ruleType': rule.ruleType,
            'sourceSymbols': rule.sourceSymbols,
            'targetSymbols': rule.targetSymbols,
            'conditionJson': rule.conditionJson,
            'actionJson': rule.actionJson,
            'confidence': rule.confidence,
            'hitRate': rule.hitRate,
            'sampleSize': rule.sampleSize,
            'source': rule.source,
            'createdAt': rule.createdAt or _now().isoformat(),
        }
        self.db.collection(self._COL_CROSS_RULES).document(ruleId).set(doc)

    def saveCrossStockRules(self, rules: List[CrossStockRule]):
        for r in rules:
            self.saveCrossStockRule(r)

    def loadCrossStockRules(self, ruleType: Optional[str] = None,
                             minConfidence: float = 0.0
                             ) -> List[CrossStockRule]:
        query = (self.db.collection(self._COL_CROSS_RULES)
                 .where('confidence', '>=', minConfidence))
        if ruleType:
            query = query.where('ruleType', '==', ruleType)
        query = query.order_by('confidence', direction=firestore.Query.DESCENDING)
        docs = query.stream()
        return [self._docToRule(d.to_dict()) for d in docs]

    def _docToRule(self, d: dict) -> CrossStockRule:
        return CrossStockRule(
            ruleId=d.get('ruleId', ''),
            ruleType=d.get('ruleType', ''),
            sourceSymbols=d.get('sourceSymbols', []),
            targetSymbols=d.get('targetSymbols', []),
            conditionJson=d.get('conditionJson', '{}'),
            actionJson=d.get('actionJson', '{}'),
            confidence=d.get('confidence', 0),
            hitRate=d.get('hitRate', 0),
            sampleSize=d.get('sampleSize', 0),
            source=d.get('source', ''),
            createdAt=d.get('createdAt', ''),
        )

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    def getSummary(self) -> Dict[str, Any]:
        """Quick overview of what's in the database."""
        def _count(collection: str, **filters) -> int:
            query = self.db.collection(collection)
            for k, v in filters.items():
                query = query.where(k, '==', v)
            try:
                result = query.count().get()
                return result[0][0].value
            except Exception:
                return sum(1 for _ in query.stream())

        nRuns  = _count(self._COL_RUNS)
        nPat   = _count(self._COL_PATTERNS, supersededBy='')
        nStrat = _count(self._COL_STRATEGIES, active=True)
        nRules = _count(self._COL_CROSS_RULES)
        nMeta  = _count(self._COL_STOCK_META)

        bestRun = self.loadBestRun()
        return {
            'totalRuns': nRuns,
            'totalPatterns': nPat,
            'totalStrategies': nStrat,
            'crossStockRules': nRules,
            'stockMetadataEntries': nMeta,
            'bestRunAlpha': bestRun.alphaVsBuyHold if bestRun else 0,
            'backend': 'firestore',
        }

    def close(self):
        """No-op for Firestore (connections are managed by the SDK)."""
        pass

    def __del__(self):
        self.close()
