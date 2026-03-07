"""
LocalAgent — Desktop Pipeline Runner with GUI
===============================================

A lightweight tkinter application that:
  1. Connects to Firestore and listens for queued pipeline tasks
  2. Shows a visual queue with descriptions, status, and controls
  3. Lets you approve, skip, or run-all tasks sequentially
  4. Executes the pipeline locally and writes results back to Firestore

Usage:
  Double-click RunAgent.bat on your desktop, or:
    python backend/LocalAgent.py
"""

from __future__ import annotations

import os
# Force matplotlib to use the non-interactive Agg backend before tkinter
# (or any other matplotlib import) is loaded.  Without this, matplotlib
# auto-selects TkAgg once it detects tkinter, creating figure managers
# that fail to destroy cleanly on exit and log atexit tracebacks.
os.environ['MPLBACKEND'] = 'Agg'

import sys
import json
import hashlib
import threading
import datetime
import traceback

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    """Current time in US/Eastern (falls back to local if zoneinfo unavailable)."""
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ── Ensure backend is importable ─────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# ── Load .env from project root (API keys, etc.) ────────────────────
_env_path = os.path.join(PROJECT_DIR, '.env')
try:
    from dotenv import load_dotenv
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)  # override=True so .env always wins
        print(f"[Config] Loaded .env from {_env_path}")
        # Verify key presence
        if os.environ.get('OPENAI_API_KEY'):
            print(f"[Config] OPENAI_API_KEY is set ({os.environ['OPENAI_API_KEY'][:8]}...)")
        else:
            print("[Config] WARNING: OPENAI_API_KEY not found in .env — OpenAI features disabled")
            print(f"[Config]   .env path: {_env_path}")
            print("[Config]   Add: OPENAI_API_KEY=sk-... to that file")
    else:
        print(f"[Config] No .env found at {_env_path}")
        print("[Config]   Create a .env file there with OPENAI_API_KEY=sk-...")
except ImportError:
    print("[Config] python-dotenv not installed — reading keys from environment only")
    print("         Install with: pip install python-dotenv")

import firebase_admin
from firebase_admin import credentials, firestore

# ═════════════════════════════════════════════════════════════════════
# Firestore connection
# ═════════════════════════════════════════════════════════════════════

def _initFirestore():
    """Initialise Firebase Admin SDK and return Firestore client."""
    saPath = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
    if not saPath:
        saPath = os.path.join(
            PROJECT_DIR,
            'protfoliomanagerv2-firebase-adminsdk-fbsvc-9fbdf566d1.json',
        )
    if not os.path.isfile(saPath):
        raise FileNotFoundError(f"Firebase key not found: {saPath}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(saPath)
        firebase_admin.initialize_app(cred)

    return firestore.client()


# ═════════════════════════════════════════════════════════════════════
# Task execution — maps task types to pipeline calls
# ═════════════════════════════════════════════════════════════════════

def _buildConfig(cmd: dict) -> dict:
    """Merge website config overrides into the default PortfolioTester kwargs."""
    cfg = cmd.get('config', {})
    stocks = cmd.get('stocks', [])

    # Map website config keys → PortfolioTester constructor kwargs
    mapping = {
        'POPULATION_SIZE':              'populationSize',
        'NUM_GENERATIONS':              'numGenerations',
        'MC_NUM_SIMULATIONS':           'mcNumSimulations',
        'MC_TIME_HORIZON':              'mcSimulationPeriods',
        'ML_FORWARD_PERIODS':           'mlForwardPeriods',
        'ML_PORTFOLIO_FORWARD_PERIODS': 'mlPortfolioForwardPeriods',
        'BACKTEST_LOOKBACK_DAYS':       'backtestPeriodDays',
        'ML_TRAIN_DAYS':                'mlTrainPeriodDays',
        'DECIDER_PATTERN_WEIGHT':       'deciderPatternWeight',
        'DECIDER_PORTFOLIO_WEIGHT':     'deciderPortfolioWeight',
        'DECIDER_MIN_CONFIDENCE':       'deciderMinConfidence',
        'IFA_MAX_SLOT_ALLOCATION':      'ifaMaxSlotAllocation',
        'IFA_MAX_STOCK_ALLOCATION':     'ifaMaxStockAllocation',
        'USE_STOP_LOSS':                'useStopLoss',
        'USE_WALK_FORWARD':             'useWalkForward',
        'USE_EARNINGS_BLACKOUT':        'useEarningsBlackout',
        'USE_REGIME_DETECTION':         'useRegimeDetection',
        'USE_CORRELATION_ADJUSTMENT':   'useCorrelationAdjustment',
    }

    kwargs = {}
    for webKey, pyKey in mapping.items():
        if webKey in cfg:
            val = cfg[webKey]
            kwargs[pyKey] = int(val) if isinstance(val, float) and val == int(val) else val

    if stocks:
        # PortfolioTester expects Dict[str, float] summing to 1.0 — equal weight each stock
        weight = round(1.0 / len(stocks), 6)
        kwargs['stocks'] = {sym: weight for sym in stocks}

    # Apply .env toggles so USE_STOP_LOSS / USE_WALK_FORWARD control both backtest & pipeline
    def _env_bool(key: str) -> bool:
        return os.environ.get(key, 'true').lower() in ('true', '1', 'yes')
    kwargs['useStopLoss'] = _env_bool('USE_STOP_LOSS')
    kwargs['useWalkForward'] = _env_bool('USE_WALK_FORWARD')

    return kwargs


def _executeTask(cmd: dict, logFn, dbClient=None, stop_check=None):
    """Run the appropriate pipeline step based on task type.
    stop_check: optional callable that returns True if run should abort."""
    taskType = cmd.get('type', 'full_pipeline')
    kwargs = _buildConfig(cmd)

    # Use portfolio fund from Firestore if set (Portfolio page is source of truth)
    if dbClient is not None:
        try:
            doc = dbClient.collection('app_config').document('portfolio').get()
            if doc.exists:
                data = doc.to_dict() or {}
                totalFund = data.get('totalFund')
                if totalFund is not None and isinstance(totalFund, (int, float)) and totalFund > 0:
                    kwargs['totalFund'] = float(totalFund)
                    logFn(f"[Config] Total fund: ${totalFund:,.0f}")
                else:
                    logFn(f"[Config] No valid totalFund in app_config/portfolio "
                          f"(got {totalFund!r}) — using default ${kwargs.get('totalFund', 100000):,.0f}")
            else:
                logFn("[Config] app_config/portfolio not found — using default $100,000")
        except Exception as e:
            logFn(f"[Config] Error reading portfolio config: {e} — using default")

    # Filter out connected/automatic stocks so the heavy pipeline only runs
    # on manually-added stocks.  Connected stocks are handled by the lighter
    # Step 8b (_evaluateConnectedStocks) inside the pipeline.
    if dbClient and 'stocks' in kwargs and taskType in (
        'full_pipeline', 'backtest_only', 'mcmc_simulation',
        'retrain_ml', 'retrain_trading', 'sentiment_update',
        'strategy_refresh', 'pattern_discovery', 'incremental_update',
    ):
        try:
            manualOnly = {}
            skippedAuto = []
            for sym in kwargs['stocks']:
                metaDoc = dbClient.collection('stock_metadata').document(sym).get()
                if metaDoc.exists and (metaDoc.to_dict() or {}).get('portfolioType') == 'automatic':
                    skippedAuto.append(sym)
                else:
                    manualOnly[sym] = kwargs['stocks'][sym]
            if skippedAuto:
                logFn(f"Filtered {len(skippedAuto)} connected stocks from heavy pipeline "
                      f"(will use light model in Step 8b): {skippedAuto}")
                if manualOnly:
                    total = sum(manualOnly.values())
                    kwargs['stocks'] = {s: w / total for s, w in manualOnly.items()}
                else:
                    logFn("WARNING: All stocks are connected — no manual stocks to run pipeline on")
        except Exception as e:
            logFn(f"Stock filtering failed (proceeding with all): {e}")

    logFn(f"Building config: {json.dumps(kwargs, indent=2)[:500]}")

    from PortfolioTester import PortfolioTester

    stop_check = stop_check or (lambda: False)

    if taskType == 'full_pipeline':
        logFn("Starting: Full Pipeline")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType == 'backtest_only':
        logFn("Starting: Backtest Only")
        kwargs['mlEnabled'] = True
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType == 'mcmc_simulation':
        logFn("Starting: MCMC Simulation")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType in ('retrain_ml', 'retrain_trading'):
        logFn(f"Starting: {taskType}")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType == 'sentiment_update':
        logFn("Starting: Sentiment Update")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return {'message': 'Sentiment updated successfully'}

    elif taskType == 'strategy_refresh':
        logFn("Starting: Strategy Refresh")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return {'message': 'Strategies refreshed'}

    elif taskType == 'pattern_discovery':
        logFn("Starting: Pattern Discovery")
        tester = PortfolioTester(**kwargs)
        tester.run(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType == 'stock_research':
        # ---- Stock research via OpenAI (manual portfolio stocks only) ----
        symbols = cmd.get('stocks', [])
        if not symbols:
            return {'message': 'No stock symbol provided'}

        ticker = symbols[0].upper()

        # Only run expensive OpenAI research for manual portfolio stocks
        try:
            metaDoc = dbClient.collection('stock_metadata').document(ticker).get()
            if metaDoc.exists:
                metaData = metaDoc.to_dict()
                if metaData.get('portfolioType') == 'automatic':
                    logFn(f"[StockResearch] Skipping {ticker} — automatic stock (no OpenAI research for connected stocks)")
                    return {'message': f'Skipped: {ticker} is an automatic/connected stock — OpenAI research only runs for manually added stocks'}
        except Exception:
            pass

        logFn(f"Starting: Stock Research for {ticker}")
        result = _runStockResearch(ticker, dbClient, logFn)
        return result

    elif taskType == 'incremental_update':
        logFn("Starting: Incremental Update (lightweight refresh)")
        kwargs['mlEnabled'] = True
        tester = PortfolioTester(**kwargs)
        tester.runIncremental(verbose=True, stop_check=stop_check)
        return _extractResult(tester)

    elif taskType == 'add_stock_pipeline':
        logFn("Starting: Add Stock Pipeline")
        result = _executeAddStockPipeline(cmd, dbClient, logFn, stop_check)
        return result

    elif taskType == 'connected_stocks':
        logFn("Starting: Connected Stocks Evaluation")
        result = _evaluateConnectedStocks(cmd, dbClient, logFn)
        return result

    elif taskType in ('trading_execute', 'trading_execute_1h', 'trading_execute_1d'):
        interval = '1h' if taskType == 'trading_execute_1h' else '1d'
        logFn(f"Starting: Trading Bot Cycle [{interval}]")
        result = _executeTradingCycle(cmd, dbClient, logFn)
        return result

    elif taskType == 'cancel_orders':
        logFn("Starting: Cancel Orders")
        result = _executeCancelOrders(cmd, dbClient, logFn)
        return result

    elif taskType == 'sync_broker_orders':
        logFn("Starting: Sync Broker Orders")
        result = _executeSyncOrders(dbClient, logFn)
        return result

    elif taskType == 'daily_review':
        logFn("Starting: Daily Performance Review (fallback)")
        result = _executeDailyReview(dbClient, logFn)
        return result

    else:
        logFn(f"Unknown task type: {taskType}")
        return {'message': f'Executed task type: {taskType}'}


def _extractResult(tester) -> dict:
    """Pull summary metrics from a completed PortfolioTester."""
    result = {}
    try:
        if hasattr(tester, 'combinedResult') and tester.combinedResult:
            cr = tester.combinedResult
            result['totalReturnPct'] = getattr(cr, 'totalReturn', 0)
            result['sharpeRatio'] = getattr(cr, 'sharpeRatio', 0)
            result['numTrades'] = getattr(cr, 'totalTrades', 0)
        if hasattr(tester, 'alphaVsBuyHold'):
            result['alphaVsBuyHold'] = tester.alphaVsBuyHold
        if hasattr(tester, 'alphaVsSP500'):
            result['alphaVsSP500'] = tester.alphaVsSP500
    except Exception:
        pass
    result['message'] = 'Pipeline completed'
    return result


# ═════════════════════════════════════════════════════════════════════
# Stock Research (OpenAI)
# ═════════════════════════════════════════════════════════════════════

_STOCK_RESEARCH_PROMPT = """\
You are a senior equity research analyst at a top-tier investment bank.
Produce a comprehensive research brief for the stock ticker **{ticker}**.

Return a JSON object (NO markdown fences — raw JSON only) with these exact keys:

{{
  "companyName": "<official company name>",
  "description": "<4-6 sentence research overview covering: what the company does, its market position, competitive moat, revenue model, and any recent strategic shifts. Be specific with numbers where possible (e.g. market share %, revenue breakdown).>",
  "sector": "<GICS sector>",
  "industry": "<GICS sub-industry>",
  "mainBusiness": "<1-2 sentence core business description with revenue breakdown by segment if known>",
  "competitiveAdvantage": "<1-2 sentences on the company's moat — brand, network effects, patents, scale, etc.>",
  "revenueBreakdown": ["<segment 1: ~X% of revenue>", "<segment 2: ~Y% of revenue>"],
  "suppliers": ["<name (ticker if public) — what they supply>", "<up to 6 key suppliers>"],
  "customers": ["<name (ticker if public) — what they buy>", "<up to 6 key customers or customer segments>"],
  "competitors": ["<name (ticker) — brief competitive angle>", "<up to 6 direct competitors>"],
  "relatedTickers": ["<up to 8 tickers: competitors + supply chain + sector peers>"],
  "recentHeadlines": ["<headline 1 — include date if known, e.g. 'Jan 2026: ...' >", "<up to 8 recent significant headlines or events>"],
  "upcomingCatalysts": ["<specific catalyst with expected date>", "<up to 4>"],
  "keyRisks": ["<specific risk with brief explanation>", "<up to 4>"],
  "financialSnapshot": {{
    "marketCapBucket": "<nano | micro | small | mid | large | mega>",
    "approximateMarketCap": "<e.g. $2.8T>",
    "trailingPE": "<number or 'N/A'>",
    "forwardPE": "<number or 'N/A'>",
    "dividendYield": "<e.g. 0.5% or 'None'>",
    "revenueGrowthYoY": "<e.g. +12%>",
    "profitMargin": "<e.g. 25%>"
  }},
  "analystConsensus": "<strong buy | buy | hold | sell | strong sell — with brief reasoning>",
  "supplyChainNotes": "<1-2 sentences on key supply chain dependencies or geographic concentration>"
}}

Be factual and specific. Use recent data. If the ticker is unknown or delisted,
set companyName to "Unknown" and description to "Ticker not recognized" and leave
all arrays empty.
"""


def _runStockResearch(ticker: str, dbClient, logFn) -> dict:
    """Call OpenAI to research a stock and save results to stock_metadata."""
    import os as _os

    apiKey = _os.environ.get('OPENAI_API_KEY', '')
    if not apiKey:
        logFn("[StockResearch] OPENAI_API_KEY not set — skipping")
        return {'message': f'No OpenAI key — cannot research {ticker}'}

    logFn(f"[StockResearch] Calling OpenAI for {ticker}...")

    try:
        import openai  # type: ignore[import-untyped]
        from OpenAIRetry import with_retry
        client = openai.OpenAI(api_key=apiKey)

        def _call():
            return client.chat.completions.create(
                model='gpt-4o-mini',
                temperature=0.3,
                max_tokens=2500,
                messages=[
                    {'role': 'system', 'content': 'You are a financial research assistant. Return only valid JSON. No markdown fences.'},
                    {'role': 'user',   'content': _STOCK_RESEARCH_PROMPT.format(ticker=ticker)},
                ],
            )
        resp = with_retry(_call)

        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model adds them
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1]  # remove opening fence line
            if raw.endswith('```'):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)
        logFn(f"[StockResearch] Got data for {ticker}: {list(data.keys())}")

    except json.JSONDecodeError as e:
        logFn(f"[StockResearch] JSON parse error: {e}\nRaw: {raw[:500]}")
        return {'message': f'OpenAI returned invalid JSON for {ticker}'}
    except Exception as e:
        logFn(f"[StockResearch] OpenAI call failed: {e}")
        return {'message': f'Research failed for {ticker}: {e}'}

    # ── Extract clean ticker lists from natural-language strings ──
    import re
    def _extractTickers(entries):
        """Pull tickers from research strings like 'Qualcomm (QCOM) — chips'."""
        tickers = []
        for entry in (entries or []):
            # Parenthetical: "Company (TICK)"
            m = re.search(r'\(([A-Z]{1,6})\)', entry or '')
            if m:
                tickers.append(m.group(1))
                continue
            for token in re.split(r'[\s\-—–,;:]+', entry or ''):
                cleaned = token.strip('().,;:\'\"')
                if cleaned.isupper() and 1 <= len(cleaned) <= 6 and cleaned.isalpha():
                    tickers.append(cleaned)
                    break
        return tickers

    supplyChainUp   = _extractTickers(data.get('suppliers', []))
    supplyChainDown = _extractTickers(data.get('customers', []))
    competesWidth   = _extractTickers(data.get('competitors', []))

    # ── Write to Firestore stock_metadata/{ticker} ───────────────
    fin = data.get('financialSnapshot', {})
    try:
        docRef = dbClient.collection('stock_metadata').document(ticker)
        docRef.set({
            'symbol':              ticker,
            'companyName':         data.get('companyName', ticker),
            'description':         data.get('description', ''),
            'sector':              data.get('sector', ''),
            'industry':            data.get('industry', ''),
            'mainBusiness':        data.get('mainBusiness', ''),
            'competitiveAdvantage': data.get('competitiveAdvantage', ''),
            'revenueBreakdown':    data.get('revenueBreakdown', []),
            'suppliers':           data.get('suppliers', []),
            'customers':           data.get('customers', []),
            'competitors':         data.get('competitors', []),
            'supplyChainUp':       supplyChainUp,
            'supplyChainDown':     supplyChainDown,
            'competesWidth':       competesWidth,
            'relatedTickers':      data.get('relatedTickers', []),
            'recentHeadlines':     data.get('recentHeadlines', []),
            'upcomingCatalysts':   data.get('upcomingCatalysts', []),
            'keyRisks':            data.get('keyRisks', []),
            'analystConsensus':    data.get('analystConsensus', ''),
            'supplyChainNotes':    data.get('supplyChainNotes', ''),
            'marketCapBucket':     fin.get('marketCapBucket', data.get('marketCapBucket', '')),
            'approximateMarketCap': fin.get('approximateMarketCap', ''),
            'trailingPE':          fin.get('trailingPE', ''),
            'forwardPE':           fin.get('forwardPE', ''),
            'dividendYield':       fin.get('dividendYield', ''),
            'revenueGrowthYoY':    fin.get('revenueGrowthYoY', ''),
            'profitMargin':        fin.get('profitMargin', ''),
            'needsResearch':       False,
            'lastResearchedAt':    _now().isoformat(),
        }, merge=True)
        logFn(f"[StockResearch] Saved metadata for {ticker} to Firestore")
    except Exception as e:
        logFn(f"[StockResearch] Firestore write failed: {e}")
        return {'message': f'Research done but save failed: {e}'}

    return {'message': f'Stock research completed for {ticker}'}


# ═════════════════════════════════════════════════════════════════════
# Connected Stocks Evaluation
# ═════════════════════════════════════════════════════════════════════

def _evaluateConnectedStocks(cmd: dict, dbClient, logFn) -> dict:
    """
    Evaluate connected stocks for the automatic portfolio.
    Discovers candidates from manual portfolio supply chains,
    scores them, adds high-quality ones, and removes underperformers.
    """
    try:
        from PersistenceManager import PersistenceManager
        from ConnectedStockManager import ConnectedStockManager
        from GeneticAlgorithm import StockDataFetcher

        logFn("[ConnectedStocks] Initializing...")
        pm = PersistenceManager()
        mgr = ConnectedStockManager(pm)

        # Get manual stocks to know which parent data we need
        manualStocks = pm.getManualPortfolioStocks()
        autoStocks = pm.getAutoPortfolioStocks()
        logFn(f"[ConnectedStocks] Manual: {list(manualStocks.keys())}")
        logFn(f"[ConnectedStocks] Current auto: {list(autoStocks.keys())}")

        # Discover candidates first (to know what price data to fetch)
        candidates = mgr.discoverCandidates(verbose=False)
        candidateSyms = {c.symbol for c in candidates}
        allSyms = set(manualStocks.keys()) | set(autoStocks.keys()) | candidateSyms

        # Fetch price data for all relevant symbols
        logFn(f"[ConnectedStocks] Fetching price data for {len(allSyms)} symbols...")
        fetcher = StockDataFetcher()
        stockDataDict = {}
        for sym in allSyms:
            try:
                df = fetcher.fetchData(sym, interval='1d', period='1y')
                if df is not None and len(df) > 0:
                    stockDataDict[sym] = df
            except Exception as e:
                logFn(f"  Could not fetch {sym}: {e}")

        logFn(f"[ConnectedStocks] Got data for {len(stockDataDict)} symbols")

        # Get last run results for trading benefit scoring
        runResults = None
        try:
            latestRun = (dbClient.collection('runs')
                        .order_by('timestamp', direction='DESCENDING')
                        .limit(1).stream())
            for doc in latestRun:
                d = doc.to_dict()
                runResults = d.get('perStockResults', {})
        except Exception:
            pass

        # Run the full update cycle
        actions = mgr.updateAutoPortfolio(stockDataDict, runResults, verbose=True)

        # Log results through logFn for GUI output
        added = [s for s, a in actions.items() if a == 'added']
        removed = [s for s, a in actions.items() if a == 'removed']
        kept = [s for s, a in actions.items() if a == 'kept']

        logFn(f"[ConnectedStocks] Added: {added}")
        logFn(f"[ConnectedStocks] Removed: {removed}")
        logFn(f"[ConnectedStocks] Kept: {kept}")
        if added:
            logFn("[ConnectedStocks] Scores are preliminary (correlation-based). "
                  "Run full pipeline or incremental update to get ML-backed scores.")

        return {
            'message': f'Connected stocks evaluation complete: '
                       f'{len(added)} added, {len(removed)} removed, {len(kept)} kept',
            'added': added,
            'removed': removed,
            'kept': kept,
        }

    except Exception as e:
        logFn(f"[ConnectedStocks] Error: {e}")
        import traceback
        logFn(traceback.format_exc())
        return {'message': f'Connected stocks evaluation failed: {e}'}


# ═════════════════════════════════════════════════════════════════════
# Auto-Scheduler — Background task scheduling for live trading
# ═════════════════════════════════════════════════════════════════════

# Default schedule intervals (hours)
DEFAULT_SCHEDULE = {
    'sentiment_fetch':        4,     # Fetch headlines every 4 hours
    'broker_sync':        0.25,     # Sync Alpaca account/positions every 15 min (auto-updates P&L)
    'incremental_update':   168,     # Retrain models weekly (168h = 7 days)
    'connected_stocks':     336,     # Evaluate connected stocks biweekly
    'strategy_refresh':     336,     # Refresh strategies biweekly
    'trading_execute_1h':     1,     # Hourly trading cycle
    'trading_execute_1d':    24,     # Daily trading cycle (saves next-day summary)
    'daily_review':          24,     # Fallback: daily performance review (runs if bot missed it)
}

# Firestore collection for scheduler state
_COL_SCHEDULER = 'scheduler_state'
# Firestore collection for cached sentiment headlines
_COL_SENTIMENT_CACHE = 'sentiment_cache'


# ═════════════════════════════════════════════════════════════════════
# Add-Stock Pipeline Execution
# ═════════════════════════════════════════════════════════════════════

def _executeAddStockPipeline(cmd: dict, dbClient, logFn, stop_check=None) -> dict:
    """
    Run the full GA+MC pipeline for newly added stock(s) while reusing
    stored patterns for existing stocks.  Then retrain all ML models,
    backtest, evaluate connected stocks, and rebalance.
    """
    from PortfolioTester import PortfolioTester

    cfg = cmd.get('config', {})
    newSymbols = cfg.get('newSymbols', cmd.get('stocks', []))
    if not newSymbols:
        logFn("[AddStock] No new symbols specified")
        return {'message': 'No new symbols to add'}

    logFn(f"[AddStock] New stocks to run full pipeline for: {newSymbols}")

    # Read ALL manual portfolio stocks from Firestore
    allManualStocks = []
    try:
        snap = (dbClient.collection('stock_metadata')
                .where('inPortfolio', '==', True).stream())
        for doc in snap:
            data = doc.to_dict() or {}
            if data.get('portfolioType') != 'automatic':
                allManualStocks.append(doc.id)
    except Exception as e:
        logFn(f"[AddStock] Failed to read stock_metadata: {e}")
        allManualStocks = list(newSymbols)

    # Ensure new symbols are in the list
    for sym in newSymbols:
        if sym not in allManualStocks:
            allManualStocks.append(sym)

    if not allManualStocks:
        logFn("[AddStock] No portfolio stocks found")
        return {'message': 'No portfolio stocks'}

    logFn(f"[AddStock] Full portfolio: {allManualStocks}")

    weight = round(1.0 / len(allManualStocks), 6)
    stocks = {sym: weight for sym in allManualStocks}

    # Build kwargs from config (pipeline parameters)
    kwargs = _buildConfig(cmd)
    kwargs['stocks'] = stocks
    kwargs['mlEnabled'] = True

    # Read totalFund from Firestore
    if dbClient is not None:
        try:
            cfgDoc = dbClient.collection('app_config').document('portfolio').get()
            if cfgDoc.exists:
                data = cfgDoc.to_dict() or {}
                tf = data.get('totalFund')
                if tf and isinstance(tf, (int, float)) and tf > 0:
                    kwargs['totalFund'] = float(tf)
                    logFn(f"[AddStock] Total fund: ${tf:,.0f}")
        except Exception:
            pass

    tester = PortfolioTester(**kwargs)
    tester.runForNewStocks(
        newSymbols=list(newSymbols),
        verbose=True,
        stop_check=stop_check,
    )
    return _extractResult(tester)


# ═════════════════════════════════════════════════════════════════════
# Trading Bot Execution
# ═════════════════════════════════════════════════════════════════════

def _executeTradingCycle(cmd: dict, dbClient, logFn) -> dict:
    """
    Run one TradingBot cycle.  Reads broker mode from Firestore
    app_config/trading or falls back to 'dry_run'.
    """
    from TradingBot import TradingBot, TradingConfig

    brokerMode = 'dry_run'
    if dbClient is not None:
        try:
            doc = dbClient.collection('app_config').document('trading').get()
            if doc.exists:
                data = doc.to_dict() or {}
                if data.get('enabled'):
                    brokerMode = data.get('mode', 'paper')
                else:
                    logFn("[TradingBot] Trading is DISABLED in config — running dry_run")
            else:
                logFn("[TradingBot] No trading config found — running dry_run")
        except Exception as e:
            logFn(f"[TradingBot] Config read error: {e} — running dry_run")

    configOverrides = {}
    if 'maxPctPerStock' in cmd:
        configOverrides['maxPctPerStock'] = float(cmd['maxPctPerStock'])
    if 'dailyLossLimitPct' in cmd:
        configOverrides['dailyLossLimitPct'] = float(cmd['dailyLossLimitPct'])
    # Apply .env toggles (same as backtest) so USE_STOP_LOSS controls live trading too
    use_sl = os.environ.get('USE_STOP_LOSS', 'true').lower() in ('true', '1', 'yes')
    configOverrides['useStopLoss'] = use_sl
    use_earnings = os.environ.get('USE_EARNINGS_BLACKOUT', 'true').lower() in ('true', '1', 'yes')
    configOverrides['useEarningsBlackout'] = use_earnings
    allow_shorts = os.environ.get('ALLOW_SHORTS', 'false').lower() in ('true', '1', 'yes')
    configOverrides['allowShorts'] = allow_shorts

    config = TradingConfig(**configOverrides) if configOverrides else None

    # Interval: 1h (hourly) or 1d (daily)
    taskType = str(cmd.get('type', 'trading_execute'))
    interval = (cmd.get('config') or {}).get('interval', '1d')
    if taskType == 'trading_execute_1h':
        interval = '1h'
    elif taskType == 'trading_execute_1d':
        interval = '1d'
    if interval not in ('1h', '1d'):
        interval = '1d'

    bot = TradingBot(
        dbClient=dbClient,
        logFn=logFn,
        brokerMode=brokerMode,
        config=config,
    )
    result = bot.runCycle(interval=interval)

    # Sync broker account, positions, and orders to Firestore after trading cycle
    if brokerMode != 'dry_run':
        try:
            from BrokerClient import AlpacaBroker
            broker = AlpacaBroker()
            _syncBrokerAccount(broker, dbClient, logFn)
            _syncBrokerPositions(broker, dbClient, logFn)
            _syncBrokerOrders(broker, dbClient, logFn)
        except Exception as e:
            logFn(f"[TradingBot] Post-cycle sync failed: {e}")

    return result


def _executeDailyReview(dbClient, logFn) -> dict:
    """
    Run the daily performance review as a standalone fallback task.
    Normally runs inline at the end of TradingBot.runCycle('1d'), but
    this ensures it still executes if the trading cycle had errors.
    """
    from PersistenceManager import PersistenceManager
    from AlertManager import AlertManager
    from DailyReviewEngine import DailyReviewEngine

    pm = PersistenceManager()
    am = AlertManager(dbClient=dbClient)

    reviewer = DailyReviewEngine(
        db=dbClient,
        persistence=pm,
        alertManager=am,
        logger=logFn,
    )
    report = reviewer.runDailyReview()
    return {'status': 'completed', 'reviewDate': (report or {}).get('date', '')}


def _executeCancelOrders(cmd: dict, dbClient, logFn) -> dict:
    """
    Cancel open broker orders.  If cmd['config']['orderIds'] is provided,
    cancel only those specific orders.  Otherwise cancel ALL open orders.
    Also writes updated order list back to Firestore broker_orders collection.
    """
    from BrokerClient import AlpacaBroker, DryRunBroker

    brokerMode = 'dry_run'
    if dbClient is not None:
        try:
            doc = dbClient.collection('app_config').document('trading').get()
            if doc.exists:
                data = doc.to_dict() or {}
                brokerMode = data.get('mode', 'dry_run')
        except Exception:
            pass

    if brokerMode == 'dry_run':
        logFn("[CancelOrders] Broker mode is dry_run — nothing to cancel")
        return {'cancelled': 0, 'message': 'Dry-run mode: no real orders to cancel'}

    try:
        broker = AlpacaBroker()
    except Exception as e:
        logFn(f"[CancelOrders] Failed to init broker: {e}")
        return {'cancelled': 0, 'error': str(e)}

    orderIds = (cmd.get('config') or {}).get('orderIds', [])
    cancelled = 0

    if orderIds:
        for oid in orderIds:
            ok = broker.cancel_order(oid)
            if ok:
                cancelled += 1
                logFn(f"  Cancelled order {oid}")
            else:
                logFn(f"  Failed to cancel order {oid}")
    else:
        cancelled = broker.cancel_all_orders()
        logFn(f"  Cancelled all open orders: {cancelled}")

    # Refresh broker orders and account so the frontend sees the update
    _syncBrokerOrders(broker, dbClient, logFn)
    _syncBrokerAccount(broker, dbClient, logFn)

    return {'cancelled': cancelled, 'message': f'Cancelled {cancelled} order(s)'}


def _executeSyncOrders(dbClient, logFn) -> dict:
    """Fetch account, positions, and orders from the broker and sync to Firestore.

    Always syncs account data and positions (so the frontend can show Live
    Portfolio Value), regardless of trading mode.  Order syncing is skipped
    only when the broker is unreachable (missing API keys).
    """
    from BrokerClient import AlpacaBroker

    try:
        broker = AlpacaBroker()
    except Exception as e:
        logFn(f"[SyncOrders] Cannot connect to Alpaca: {e}")
        return {'synced': 0, 'error': str(e)}

    try:
        _syncBrokerAccount(broker, dbClient, logFn)
        _syncBrokerPositions(broker, dbClient, logFn)
        _syncBrokerOrders(broker, dbClient, logFn)
        orders = broker.get_orders(status='all')
        return {'synced': len(orders)}
    except Exception as e:
        logFn(f"[SyncOrders] Error: {e}")
        return {'synced': 0, 'error': str(e)}


def _syncBrokerAccount(broker, dbClient, logFn):
    """Fetch account (equity, day P&L) from broker and write to Firestore."""
    if dbClient is None:
        return
    try:
        account = broker.get_account()
        dbClient.collection('broker_account').document('live').set({
            'equity': account.equity,
            'cash': account.cash,
            'buyingPower': account.buyingPower,
            'portfolioValue': account.portfolioValue,
            'dayPL': account.dayPL,
            'dayPLPct': account.dayPLPct,
            'updatedAt': datetime.datetime.now().isoformat(),
            'brokerMode': broker.mode,
        }, merge=True)
        logFn(f"  Synced broker account: equity=${account.equity:,.0f}, day P&L ${account.dayPL:+,.2f}")
    except Exception as e:
        logFn(f"  [SyncAccount] Error: {e}")


def _syncBrokerPositions(broker, dbClient, logFn):
    """
    Sync Alpaca positions to trade_positions so the Positions tab shows
    actual broker holdings. Uses Alpaca as source of truth.
    """
    if dbClient is None:
        return
    try:
        positions = broker.get_positions()
        col = dbClient.collection('trade_positions')
        active_symbols = set()
        for p in positions:
            qty = int(p.qty) if p.qty else 0
            if qty <= 0:
                continue
            active_symbols.add(p.symbol)
            cost_basis = (p.avgEntryPrice or 0) * qty
            doc_data = {
                'symbol': p.symbol,
                'qty': qty,
                'avgEntryPrice': float(p.avgEntryPrice or 0),
                'currentPrice': float(p.currentPrice or 0),
                'marketValue': float(p.marketValue or 0),
                'unrealizedPL': float(p.unrealizedPL or 0),
                'unrealizedPLPct': float(p.unrealizedPLPct or 0),
                'totalInvested': cost_basis,
                'totalRealized': 0.0,
                'numBuys': 0,
                'numSells': 0,
                'lastUpdated': datetime.datetime.now().isoformat(),
            }
            col.document(p.symbol).set(doc_data, merge=True)
        # Clear positions no longer held (closed on Alpaca)
        for doc in col.stream():
            if doc.id not in active_symbols:
                col.document(doc.id).update({'qty': 0})
        logFn(f"  Synced {len(active_symbols)} positions to trade_positions")
    except Exception as e:
        logFn(f"  [SyncPositions] Error: {e}")


def _syncBrokerOrders(broker, dbClient, logFn):
    """Fetch current orders from the broker and write them to Firestore."""
    if dbClient is None:
        return
    try:
        orders = broker.get_orders(status='all')
        col = dbClient.collection('broker_orders')
        batch = dbClient.batch()
        for o in orders[:200]:
            doc_ref = col.document(o.orderId)
            batch.set(doc_ref, {
                'orderId': o.orderId,
                'symbol': o.symbol,
                'side': o.side,
                'qty': o.qty,
                'orderType': o.orderType,
                'filledQty': o.filledQty,
                'filledAvgPrice': o.filledAvgPrice,
                'status': o.status,
                'createdAt': o.createdAt,
                'limitPrice': o.limitPrice,
                'syncedAt': datetime.datetime.now().isoformat(),
            })
        batch.commit()
        logFn(f"  Synced {len(orders)} broker orders to Firestore")
    except Exception as e:
        logFn(f"  [CancelOrders] Sync error: {e}")


_TASK_COVERS_SCHEDULER = {
    'full_pipeline': [
        'last_incremental_update', 'last_connected_stocks',
        'last_strategy_refresh', 'last_sentiment_fetch',
    ],
    'incremental_update': ['last_incremental_update'],
    'connected_stocks':   ['last_connected_stocks'],
    'strategy_refresh':   ['last_strategy_refresh'],
    'sentiment_update':   ['last_sentiment_fetch'],
}


def _updateSchedulerStateAfterTask(taskType: str, db, logFn):
    """Mark scheduler keys as 'just ran' so the auto-scheduler won't re-queue."""
    keys = _TASK_COVERS_SCHEDULER.get(taskType)
    if not keys or db is None:
        return
    try:
        nowIso = _now().isoformat()
        patch = {k: nowIso for k in keys}
        db.collection(_COL_SCHEDULER).document('state').set(patch, merge=True)
        logFn(f"[Scheduler-state] Updated after '{taskType}': {list(patch.keys())}")
    except Exception as e:
        logFn(f"[Scheduler-state] Update failed (non-fatal): {e}")


class AutoScheduler:
    """
    Background scheduler that:
      1. Fetches & scores news headlines periodically (no queue needed)
      2. Queues incremental ML updates when enough time has passed
      3. Queues connected stock evaluations periodically
      4. Filters headlines for relevance before storing

    Runs as a daemon thread alongside the GUI.  State is persisted to
    Firestore so it survives agent restarts.
    """

    def __init__(self, dbClient, logFn, schedule: dict = None):
        self.db = dbClient
        self._log = logFn
        self.schedule = {**DEFAULT_SCHEDULE, **(schedule or {})}
        self._running = False
        self._thread = None
        self._stopEvent = threading.Event()
        # Track accumulated new headlines since last ML retrain
        self._newHeadlineCount = 0
        self._headlinesForRetrain = 50  # queue retrain after this many new relevant headlines

    def start(self):
        if self._running:
            return
        self._running = True
        self._stopEvent.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log("[Scheduler] Started — background tasks will run automatically")
        # Run startup catch-up in a separate thread so the GUI isn't blocked.
        # This handles the case where the laptop was off: sentiment fetch can't
        # be queued by the cloud function (it needs Python), so we catch up here.
        threading.Thread(target=self._startupCatchUp, daemon=True).start()

    def _startupCatchUp(self):
        """
        Called once when the scheduler starts.  Checks sentiment_fetch only —
        all other tasks (incremental_update, connected_stocks, strategy_refresh)
        are handled by the Firebase Cloud Function and will already be waiting
        in run_commands if they were due while the laptop was off.
        """
        try:
            self._loadConfigIntervals()
            state = self._loadState()
            now   = _now()
            lastSentiment  = state.get('last_sentiment_fetch', '')
            intervalHours  = self.schedule.get('sentiment_fetch', 4)
            if self._isOverdue(lastSentiment, intervalHours, now):
                self._log("[Scheduler] Startup catch-up: sentiment fetch is overdue, running now...")
                self._backgroundSentimentFetch(now)
                state['last_sentiment_fetch'] = now.isoformat()
                self._saveState(state)
            # Run broker sync on startup so live portfolio value updates immediately
            lastBrokerSync = state.get('last_broker_sync', '')
            if self._isOverdue(lastBrokerSync, self.schedule.get('broker_sync', 0.25), now):
                self._log("[Scheduler] Startup catch-up: broker sync is overdue, running now...")
                self._backgroundBrokerSync()
                state['last_broker_sync'] = now.isoformat()
                self._saveState(state)
        except Exception as e:
            self._log(f"[Scheduler] Startup catch-up error: {e}")

    def stop(self):
        self._running = False
        self._stopEvent.set()
        self._log("[Scheduler] Stopped")

    @property
    def isRunning(self):
        return self._running

    def _loop(self):
        """Main scheduler loop — checks what needs running every 60 seconds."""
        # Load custom intervals from Firestore config (set by frontend Settings page)
        self._loadConfigIntervals()
        while not self._stopEvent.is_set():
            try:
                self._tick()
            except Exception as e:
                self._log(f"[Scheduler] Error in tick: {e}")
            self._stopEvent.wait(60)  # check every 60 seconds

    def _loadConfigIntervals(self):
        """Load schedule intervals from Firestore (written by frontend Settings page)."""
        try:
            doc = self.db.collection('scheduler_config').document('intervals').get()
            if doc.exists:
                config = doc.to_dict()
                for key in DEFAULT_SCHEDULE:
                    if key in config and isinstance(config[key], (int, float)):
                        self.schedule[key] = config[key]
                self._log(f"[Scheduler] Loaded intervals: {self.schedule}")
        except Exception:
            pass

    def _tick(self):
        """Check each scheduled task and run/queue if overdue."""
        now = _now()
        state = self._loadState()

        # 1. Background sentiment fetch (runs directly, no queue)
        lastSentiment = state.get('last_sentiment_fetch', '')
        intervalHours = self.schedule.get('sentiment_fetch', 4)
        if self._isOverdue(lastSentiment, intervalHours, now):
            self._backgroundSentimentFetch(now)
            state['last_sentiment_fetch'] = now.isoformat()
            self._saveState(state)

        # 1b. Background broker sync (Alpaca equity, P&L) — runs directly, no queue
        lastBrokerSync = state.get('last_broker_sync', '')
        intervalHours = self.schedule.get('broker_sync', 0.25)
        if self._isOverdue(lastBrokerSync, intervalHours, now):
            self._backgroundBrokerSync()
            state['last_broker_sync'] = now.isoformat()
            self._saveState(state)

        # 2. Incremental ML update (queued)
        lastIncremental = state.get('last_incremental_update', '')
        intervalHours = self.schedule.get('incremental_update', 168)
        if self._isOverdue(lastIncremental, intervalHours, now):
            self._queueTask('incremental_update', 'Scheduled: Incremental ML update')
            state['last_incremental_update'] = now.isoformat()
            self._saveState(state)

        # 3. Connected stocks evaluation (queued)
        lastConnected = state.get('last_connected_stocks', '')
        intervalHours = self.schedule.get('connected_stocks', 336)
        if self._isOverdue(lastConnected, intervalHours, now):
            self._queueTask('connected_stocks', 'Scheduled: Connected stocks evaluation')
            state['last_connected_stocks'] = now.isoformat()
            self._saveState(state)

        # 4. Strategy refresh (queued)
        lastStrategy = state.get('last_strategy_refresh', '')
        intervalHours = self.schedule.get('strategy_refresh', 336)
        if self._isOverdue(lastStrategy, intervalHours, now):
            self._queueTask('strategy_refresh', 'Scheduled: Strategy refresh')
            state['last_strategy_refresh'] = now.isoformat()
            self._saveState(state)

        # 5a. Hourly trading cycle — automatic (runs directly) or queued
        lastTrading1h = state.get('last_trading_execute_1h', '')
        if self._isOverdue(lastTrading1h, self.schedule.get('trading_execute_1h', 1), now):
            if self._tradingScheduleMode() == 'automatic':
                self._runTradingCycleDirect('1h', now, state, 'last_trading_execute_1h')
            else:
                self._queueTask('trading_execute_1h', 'Scheduled: Trading bot cycle [1h]')
                state['last_trading_execute_1h'] = now.isoformat()
                self._saveState(state)

        # 5b. Daily trading cycle — automatic (runs directly) or queued
        lastTrading1d = state.get('last_trading_execute_1d', '')
        if self._isOverdue(lastTrading1d, self.schedule.get('trading_execute_1d', 24), now):
            if self._tradingScheduleMode() == 'automatic':
                self._runTradingCycleDirect('1d', now, state, 'last_trading_execute_1d')
            else:
                self._queueTask('trading_execute_1d', 'Scheduled: Trading bot cycle [1d]')
                state['last_trading_execute_1d'] = now.isoformat()
                self._saveState(state)

        # 5c. Daily review fallback — only runs if TradingBot didn't already
        lastReview = state.get('last_daily_review', '')
        lastDailyCycle = state.get('last_trading_execute_1d', '')
        if self._isOverdue(lastReview, self.schedule.get('daily_review', 24), now):
            # Only queue if the daily trading cycle ran (review is a post-step)
            if lastDailyCycle and not self._isOverdue(lastDailyCycle, 25, now):
                self._queueTask('daily_review', 'Scheduled: Daily performance review (fallback)')
            state['last_daily_review'] = now.isoformat()
            self._saveState(state)

        # 6. Sentiment-triggered ML retrain (when enough headlines accumulate)
        if self._newHeadlineCount >= self._headlinesForRetrain:
            self._queueTask('incremental_update',
                            f'Triggered: {self._newHeadlineCount} new headlines accumulated')
            self._newHeadlineCount = 0
            state['last_incremental_update'] = now.isoformat()
            self._saveState(state)

    def _backgroundSentimentFetch(self, now: datetime.datetime):
        """
        Fetch & score headlines for all portfolio stocks in the background.
        Only stores headlines that are RELEVANT to portfolio stocks.
        This does NOT go through the task queue — it runs directly.
        """
        try:
            from PersistenceManager import PersistenceManager
            pm = PersistenceManager()
            portfolioStocks = pm.getPortfolioStocks()
            symbols = list(portfolioStocks.keys())

            if not symbols:
                return

            self._log(f"[Sentiment] Fetching headlines for {len(symbols)} stocks...")

            from SentimentAnalysis import SentimentAnalyzer, _fetchYFinanceNews, _parseYFinanceItem

            analyzer = SentimentAnalyzer(
                openAIKey=os.environ.get('OPENAI_API_KEY', ''),
                maxOpenAIPerSymbol=2,  # conservative for background fetches
            )

            totalNew = 0
            totalSkipped = 0

            for symbol in symbols:
                try:
                    rawNews = _fetchYFinanceNews(symbol, lookbackDays=3)
                    articles = [a for item in rawNews
                                if (a := _parseYFinanceItem(item, symbol)) is not None]

                    for article in articles:
                        # Relevance filter: check if headline mentions portfolio stocks
                        # or contains financial keywords
                        if not self._isRelevantHeadline(article.headline, symbols):
                            totalSkipped += 1
                            continue

                        # Check if we already have this headline cached
                        headlineHash = hashlib.md5(
                            f"{article.headline}_{symbol}".encode()
                        ).hexdigest()[:16]

                        cacheDoc = self.db.collection(_COL_SENTIMENT_CACHE).document(headlineHash).get()
                        if cacheDoc.exists:
                            continue

                        # Score with Layer 1 + 2 only (no OpenAI for background)
                        scored = analyzer._scoreArticles([article], symbol, verbose=False)
                        if scored:
                            s = scored[0]
                            self.db.collection(_COL_SENTIMENT_CACHE).document(headlineHash).set({
                                'symbol': symbol,
                                'headline': s.headline,
                                'score': s.ensembleScore,
                                'confidence': s.ensembleConf,
                                'source': s.source,
                                'publishedAt': s.publishedAt.isoformat() if s.publishedAt else '',
                                'fetchedAt': now.isoformat(),
                                'lexiconScore': s.lexiconScore,
                                'structuralScore': s.structuralScore,
                            })
                            totalNew += 1

                except Exception as e:
                    self._log(f"[Sentiment] Error fetching {symbol}: {e}")

            self._newHeadlineCount += totalNew
            if totalNew > 0 or totalSkipped > 0:
                self._log(f"[Sentiment] {totalNew} new headlines scored, "
                          f"{totalSkipped} irrelevant skipped "
                          f"({self._newHeadlineCount} accumulated since last retrain)")

        except Exception as e:
            self._log(f"[Sentiment] Background fetch failed: {e}")

    def _backgroundBrokerSync(self):
        """
        Sync Alpaca account (equity, day P&L) and orders to Firestore.
        Runs automatically every 15 min. Always attempts Alpaca sync when
        credentials exist — live portfolio value updates regardless of
        trading mode (dry_run/paper/live).
        """
        if self.db is None:
            return
        try:
            from BrokerClient import AlpacaBroker
            broker = AlpacaBroker()
            _syncBrokerAccount(broker, self.db, self._log)
            _syncBrokerPositions(broker, self.db, self._log)
            _syncBrokerOrders(broker, self.db, self._log)
        except Exception as e:
            self._log(f"[BrokerSync] Background sync failed: {e}")

    @staticmethod
    def _isRelevantHeadline(headline: str, portfolioSymbols: list) -> bool:
        """
        Filter headlines for relevance to the portfolio.
        Returns True if the headline likely relates to portfolio stocks.
        """
        if not headline:
            return False

        headlineLower = headline.lower()

        # Check if any portfolio ticker appears in the headline
        for sym in portfolioSymbols:
            if sym.lower() in headlineLower:
                return True

        # Check for financial keywords that are broadly relevant
        FINANCIAL_KEYWORDS = {
            'earnings', 'revenue', 'profit', 'guidance', 'forecast',
            'fed', 'interest rate', 'inflation', 'gdp', 'recession',
            'market', 'stock', 'shares', 'quarterly', 'annual',
            'upgrade', 'downgrade', 'analyst', 'dividend',
            'acquisition', 'merger', 'ipo', 'sec', 'regulatory',
            'tariff', 'trade war', 'supply chain', 'semiconductor',
            'ai ', 'artificial intelligence', 'cloud', 'tech',
        }
        for kw in FINANCIAL_KEYWORDS:
            if kw in headlineLower:
                return True

        return False

    def _tradingScheduleMode(self) -> str:
        """Read scheduleMode from app_config/trading. 'automatic' = run directly; else 'queued'."""
        try:
            doc = self.db.collection('app_config').document('trading').get()
            if doc.exists:
                mode = (doc.to_dict() or {}).get('scheduleMode', 'queued')
                return mode if mode in ('automatic', 'queued') else 'queued'
        except Exception:
            pass
        return 'queued'

    def _runTradingCycleDirect(self, interval: str, now: datetime.datetime,
                               state: dict, stateKey: str):
        """Run trading cycle directly in background (no queue, no approval)."""
        def _worker():
            try:
                taskType = f'trading_execute_{interval}'
                self._log(f"[Trading] Running {interval} cycle automatically...")
                cmd = {
                    'type': taskType,
                    'config': {'interval': interval},
                    'stocks': [],
                }
                _executeTradingCycle(cmd, self.db, self._log)
                state[stateKey] = now.isoformat()
                self._saveState(state)
                self._log(f"[Trading] {interval} cycle completed.")
            except Exception as e:
                self._log(f"[Trading] {interval} cycle failed: {e}")
                import traceback
                self._log(traceback.format_exc())

        threading.Thread(target=_worker, daemon=True).start()

    def _queueTask(self, taskType: str, description: str):
        """Queue a task in Firestore run_commands (same as website does)."""
        try:
            # Check if there's already a queued task of this type
            existing = (self.db.collection('run_commands')
                       .where('type', '==', taskType)
                       .where('status', '==', 'queued')
                       .limit(1).stream())
            for _ in existing:
                return  # already queued, don't duplicate

            # Get portfolio stocks for the task
            stockDocs = (self.db.collection('stock_metadata')
                        .where('inPortfolio', '==', True)
                        .stream())
            stocks = [d.to_dict().get('symbol', d.id) for d in stockDocs]

            self.db.collection('run_commands').add({
                'type': taskType,
                'status': 'queued',
                'description': description,
                'stocks': stocks,
                'config': {},
                'source': 'scheduler',
                'createdAt': _now().isoformat(),
                'approvedAt': None,
                'startedAt': None,
                'completedAt': None,
                'result': None,
                'error': None,
            })
            self._log(f"[Scheduler] Queued: {description}")

        except Exception as e:
            self._log(f"[Scheduler] Failed to queue {taskType}: {e}")

    @staticmethod
    def _isOverdue(lastRun: str, intervalHours: float,
                   now: datetime.datetime) -> bool:
        """Check if enough time has passed since the last run."""
        if not lastRun:
            return True
        try:
            lastDt = datetime.datetime.fromisoformat(lastRun)
            if lastDt.tzinfo is None and now.tzinfo is not None:
                lastDt = lastDt.replace(tzinfo=datetime.timezone.utc)
            elif lastDt.tzinfo is not None and now.tzinfo is None:
                now = now.replace(tzinfo=datetime.timezone.utc)
            elapsed = (now - lastDt).total_seconds() / 3600
            return elapsed >= intervalHours
        except Exception:
            return True

    def _loadState(self) -> dict:
        try:
            doc = self.db.collection(_COL_SCHEDULER).document('state').get()
            return doc.to_dict() if doc.exists else {}
        except Exception:
            return {}

    def _saveState(self, state: dict):
        try:
            self.db.collection(_COL_SCHEDULER).document('state').set(state, merge=True)
        except Exception:
            pass

    def getStatus(self) -> dict:
        """Return current scheduler status for UI display."""
        state = self._loadState()
        now = _now()
        status = {'running': self._running, 'tasks': {}}

        for taskName, intervalHours in self.schedule.items():
            lastKey = f'last_{taskName}'
            lastRun = state.get(lastKey, '')
            if lastRun:
                try:
                    lastDt = datetime.datetime.fromisoformat(lastRun)
                    if lastDt.tzinfo is None and now.tzinfo is not None:
                        lastDt = lastDt.replace(tzinfo=datetime.timezone.utc)
                    elapsed = (now - lastDt).total_seconds() / 3600
                    nextIn = max(0, intervalHours - elapsed)
                except Exception:
                    elapsed = 0
                    nextIn = 0
            else:
                elapsed = float('inf')
                nextIn = 0

            status['tasks'][taskName] = {
                'intervalHours': intervalHours,
                'lastRun': lastRun,
                'nextInHours': round(nextIn, 1),
            }

        status['newHeadlines'] = self._newHeadlineCount
        status['headlinesForRetrain'] = self._headlinesForRetrain
        return status


# ═════════════════════════════════════════════════════════════════════
# GUI Application
# ═════════════════════════════════════════════════════════════════════

class AgentApp:
    """Tkinter GUI for the Local Pipeline Agent."""

    BG         = '#0d1117'
    CARD_BG    = '#161b22'
    BORDER     = '#30363d'
    TEXT       = '#e6edf3'
    MUTED      = '#8b949e'
    BLUE       = '#58a6ff'
    GREEN      = '#3fb950'
    RED        = '#f85149'
    YELLOW     = '#d29922'

    def __init__(self):
        self.db = _initFirestore()
        self.running = False
        self._currentThread = None
        self._stopRequested = threading.Event()
        self._logBuffer = []  # buffer log messages until GUI is ready

        # ── Root window ───────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Portfolio Pipeline Agent")
        self.root.geometry("820x640")
        self.root.configure(bg=self.BG)
        self.root.minsize(700, 500)

        # ── Style ─────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.',        background=self.BG, foreground=self.TEXT, fieldbackground=self.CARD_BG)
        style.configure('TFrame',   background=self.BG)
        style.configure('TLabel',   background=self.BG, foreground=self.TEXT, font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Muted.TLabel', foreground=self.MUTED, font=('Segoe UI', 9))
        style.configure('TButton',  padding=6, font=('Segoe UI', 9))
        style.map('TButton',
                   background=[('active', self.BORDER), ('!active', self.CARD_BG)],
                   foreground=[('active', self.TEXT), ('!active', self.TEXT)])

        # Treeview styling
        style.configure('Treeview',
                        background=self.CARD_BG, foreground=self.TEXT,
                        fieldbackground=self.CARD_BG, borderwidth=0,
                        font=('Segoe UI', 9), rowheight=32)
        style.configure('Treeview.Heading',
                        background=self.BG, foreground=self.MUTED,
                        font=('Segoe UI', 9, 'bold'))
        style.map('Treeview',
                   background=[('selected', '#1c2333')],
                   foreground=[('selected', self.BLUE)])

        # ── Header ────────────────────────────────────────────────────
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill='x')
        ttk.Label(header, text="Pipeline Agent", style='Title.TLabel').pack(side='left')
        self.statusLabel = ttk.Label(header, text="Idle", style='Muted.TLabel')
        self.statusLabel.pack(side='right')

        # ── Task queue table ──────────────────────────────────────────
        treeFrame = ttk.Frame(self.root, padding=(16, 0))
        treeFrame.pack(fill='both', expand=True)

        cols = ('status', 'type', 'stocks', 'description', 'created')
        self.tree = ttk.Treeview(treeFrame, columns=cols, show='headings', height=8)
        self.tree.heading('status',      text='Status',  anchor='w')
        self.tree.heading('type',        text='Type',    anchor='w')
        self.tree.heading('stocks',      text='Stocks',  anchor='w')
        self.tree.heading('description', text='Description', anchor='w')
        self.tree.heading('created',     text='Created', anchor='w')

        self.tree.column('status',      width=80,  minwidth=60)
        self.tree.column('type',        width=120, minwidth=80)
        self.tree.column('stocks',      width=160, minwidth=80)
        self.tree.column('description', width=240, minwidth=100)
        self.tree.column('created',     width=130, minwidth=80)

        scrollbar = ttk.Scrollbar(treeFrame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # ── Buttons ───────────────────────────────────────────────────
        btnFrame = ttk.Frame(self.root, padding=(16, 10))
        btnFrame.pack(fill='x')

        self.btnRun    = ttk.Button(btnFrame, text="▶  Run Selected",  command=self._onRunSelected)
        self.btnRunAll = ttk.Button(btnFrame, text="▶▶ Run All",       command=self._onRunAll)
        self.btnStop   = ttk.Button(btnFrame, text="⏹  Stop",          command=self._onStop)
        self.btnSkip   = ttk.Button(btnFrame, text="⏭  Skip",          command=self._onSkip)
        self.btnDelete = ttk.Button(btnFrame, text="✕  Delete",        command=self._onDelete)
        self.btnRefresh= ttk.Button(btnFrame, text="⟳  Refresh",       command=self._refreshQueue)

        self._showHistory = tk.BooleanVar(value=False)
        self.btnHistory = ttk.Checkbutton(
            btnFrame, text="Show History",
            variable=self._showHistory, command=self._refreshQueue,
            style='TCheckbutton',
        )

        self.btnRun.pack(side='left', padx=(0, 6))
        self.btnRunAll.pack(side='left', padx=(0, 6))
        self.btnStop.pack(side='left', padx=(0, 6))
        self.btnSkip.pack(side='left', padx=(0, 6))
        self.btnDelete.pack(side='left', padx=(0, 6))
        self.btnHistory.pack(side='left', padx=(12, 0))
        self.btnRefresh.pack(side='right')

        # ── Scheduler controls ───────────────────────────────────────
        schedFrame = ttk.Frame(self.root, padding=(16, 2))
        schedFrame.pack(fill='x')

        self._schedulerOn = tk.BooleanVar(value=True)
        self.btnScheduler = ttk.Checkbutton(
            schedFrame, text="Auto-Schedule (Live Trading Mode)",
            variable=self._schedulerOn, command=self._onToggleScheduler,
            style='TCheckbutton',
        )
        self.btnScheduler.pack(side='left')

        self.schedulerStatusLabel = ttk.Label(
            schedFrame, text="Scheduler: Off", style='Muted.TLabel'
        )
        self.schedulerStatusLabel.pack(side='right')

        # ── Log output ────────────────────────────────────────────────
        logFrame = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        logFrame.pack(fill='both', expand=True)

        ttk.Label(logFrame, text="Output Log", style='Muted.TLabel').pack(anchor='w', pady=(0, 4))
        self.logBox = scrolledtext.ScrolledText(
            logFrame, height=10, wrap='word',
            bg=self.CARD_BG, fg=self.TEXT, insertbackground=self.TEXT,
            font=('Consolas', 9), borderwidth=1, relief='solid',
            highlightbackground=self.BORDER, highlightthickness=1,
        )
        self.logBox.pack(fill='both', expand=True)

        # ── Initial data load ─────────────────────────────────────────
        self._queueData = []   # list of dicts with Firestore doc data
        self._resetOrphanedTasks()
        self._refreshQueue()

        # ── Auto-scheduler (initialized after GUI so _log works) ──────
        self.scheduler = AutoScheduler(self.db, self._log)
        self.scheduler.start()
        self._log("Auto-scheduler started automatically")

        # Start periodic scheduler status update
        self._updateSchedulerStatus()

    def _onToggleScheduler(self):
        """Toggle the auto-scheduler on/off."""
        if self._schedulerOn.get():
            self.scheduler.start()
            self._log("Auto-scheduler ENABLED — sentiment fetching every 4h, "
                      "ML updates weekly, strategies biweekly")
        else:
            self.scheduler.stop()
            self._log("Auto-scheduler DISABLED")
        self._updateSchedulerStatus()

    def _updateSchedulerStatus(self):
        """Update the scheduler status label periodically."""
        if self.scheduler.isRunning:
            status = self.scheduler.getStatus()
            tasks = status.get('tasks', {})
            nextTask = None
            nextTime = float('inf')
            for name, info in tasks.items():
                if info['nextInHours'] < nextTime:
                    nextTime = info['nextInHours']
                    nextTask = name
            headlines = status.get('newHeadlines', 0)
            headlineThresh = status.get('headlinesForRetrain', 50)
            sentimentNext = tasks.get('sentiment_fetch', {}).get('nextInHours', 0)
            label = (
                f"Scheduler: ON (sentiment fetch local) | "
                f"Sentiment in {sentimentNext:.0f}h | "
                f"Headlines: {headlines}/{headlineThresh} | "
                f"ML tasks queued by Cloud when laptop off"
            )
            self.schedulerStatusLabel.config(text=label, foreground=self.GREEN)
        else:
            self.schedulerStatusLabel.config(
                text="Scheduler: Off  (Cloud still queues ML tasks while off)",
                foreground=self.MUTED,
            )

        # Schedule next update in 30 seconds
        self.root.after(30000, self._updateSchedulerStatus)

    def _log(self, msg: str):
        """Append a message to the log box (thread-safe)."""
        def _append():
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            self.logBox.insert('end', f"[{ts}] {msg}\n")
            self.logBox.see('end')
        self.root.after(0, _append)

    def _setStatus(self, text: str):
        self.root.after(0, lambda: self.statusLabel.config(text=text))

    def _setButtons(self, enabled: bool):
        runState = 'normal' if enabled else 'disabled'
        stopState = 'disabled' if enabled else 'normal'
        self.root.after(0, lambda: [
            self.btnRun.config(state=runState),
            self.btnRunAll.config(state=runState),
            self.btnStop.config(state=stopState),
            self.btnSkip.config(state=runState),
        ])

    def _onStop(self):
        """Request stop of the currently running task(s)."""
        self._stopRequested.set()
        self._log("Stop requested — will abort after current task/stage completes.")

    # ── Reset orphaned "running" tasks on startup ─────────────────────

    def _resetOrphanedTasks(self):
        """
        At startup, any task still marked 'running' was left in that state
        because the agent was closed or crashed mid-execution.  Reset them
        back to 'queued' so they can be re-run.
        """
        try:
            docs = (self.db.collection('run_commands')
                    .where('status', '==', 'running')
                    .stream())
            reset = 0
            for doc in docs:
                doc.reference.update({'status': 'queued', 'startedAt': None})
                reset += 1
            if reset:
                msg = f"⚠ Reset {reset} orphaned 'running' task(s) back to 'queued' (agent was closed mid-run)."
                print(f"[Agent] {msg}")
                self._log(msg)
        except Exception as e:
            print(f"[Agent] WARNING: Could not reset orphaned tasks: {e}")

    # ── Refresh queue from Firestore ──────────────────────────────────

    def _refreshQueue(self):
        """Fetch tasks from Firestore. Hides completed/failed/skipped unless 'Show History' is checked."""
        self.tree.delete(*self.tree.get_children())
        self._queueData = []

        showAll = self._showHistory.get() if hasattr(self, '_showHistory') else False
        ACTIVE_STATUSES = {'queued', 'approved', 'running'}

        try:
            docs = (self.db.collection('run_commands')
                    .order_by('createdAt', direction=firestore.Query.DESCENDING)
                    .limit(50)
                    .stream())

            displayed = 0
            hidden = 0
            for doc in docs:
                d = doc.to_dict()
                d['_id'] = doc.id
                self._queueData.append(d)

                status = d.get('status', '?')

                # Filter: only show active tasks unless Show History is on
                if not showAll and status not in ACTIVE_STATUSES:
                    hidden += 1
                    continue

                taskType = (d.get('type', '') or '').replace('_', ' ')
                stocks = ', '.join(d.get('stocks', [])[:4])
                if len(d.get('stocks', [])) > 4:
                    stocks += '...'
                desc = d.get('description', '')[:50]
                created = (d.get('createdAt', '') or '')[:16]

                tag = status
                self.tree.insert('', 'end', iid=doc.id, values=(
                    status, taskType, stocks, desc, created,
                ), tags=(tag,))
                displayed += 1

            # Color tags
            self.tree.tag_configure('queued',    foreground=self.YELLOW)
            self.tree.tag_configure('approved',  foreground=self.BLUE)
            self.tree.tag_configure('running',   foreground=self.BLUE)
            self.tree.tag_configure('completed', foreground=self.GREEN)
            self.tree.tag_configure('failed',    foreground=self.RED)
            self.tree.tag_configure('skipped',   foreground=self.MUTED)

            pending = sum(1 for d in self._queueData if d.get('status') == 'queued')
            hiddenMsg = f" ({hidden} completed/failed hidden)" if hidden else ""
            self._log(f"Refreshed: {displayed} shown, {pending} pending{hiddenMsg}")

        except Exception as e:
            self._log(f"ERROR refreshing queue: {e}")

    # ── Run selected task ─────────────────────────────────────────────

    def _onRunSelected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a task from the list first.")
            return
        docId = sel[0]
        cmd = next((d for d in self._queueData if d.get('_id') == docId), None)
        if not cmd or cmd.get('status') not in ('queued', 'approved'):
            messagebox.showinfo("Invalid", "Only queued tasks can be run.")
            return
        self._runTasks([cmd])

    def _onRunAll(self):
        pending = [d for d in self._queueData if d.get('status') == 'queued']
        if not pending:
            messagebox.showinfo("Empty", "No queued tasks to run.")
            return
        if not messagebox.askyesno("Run All", f"Run all {len(pending)} queued tasks sequentially?"):
            return
        self._runTasks(pending)

    def _onSkip(self):
        sel = self.tree.selection()
        if not sel:
            return
        docId = sel[0]
        try:
            self.db.collection('run_commands').document(docId).update({
                'status': 'skipped',
            })
            self._log(f"Skipped task {docId[:8]}...")
            self._refreshQueue()
        except Exception as e:
            self._log(f"ERROR skipping: {e}")

    def _onDelete(self):
        """Delete selected task(s) from Firestore permanently."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a task to delete.")
            return
        docId = sel[0]
        cmd = next((d for d in self._queueData if d.get('_id') == docId), None)
        status = cmd.get('status', '?') if cmd else '?'
        if status == 'running':
            messagebox.showinfo("Cannot Delete", "Cannot delete a running task.")
            return
        if not messagebox.askyesno("Delete Task", f"Permanently delete this {status} task?"):
            return
        try:
            self.db.collection('run_commands').document(docId).delete()
            self._log(f"Deleted task {docId[:8]}...")
            self._refreshQueue()
        except Exception as e:
            self._log(f"ERROR deleting: {e}")

    # ── Execute tasks in a background thread ──────────────────────────

    def _runTasks(self, tasks: list):
        if self.running:
            messagebox.showinfo("Busy", "A task is already running.")
            return

        self.running = True
        self._stopRequested.clear()
        self._setButtons(False)

        def _worker():
            stop_check = lambda: self._stopRequested.is_set()
            for i, cmd in enumerate(tasks):
                if stop_check():
                    self._log("Stop requested — aborting.")
                    break
                docId = cmd['_id']
                desc = cmd.get('description', cmd.get('type', '?'))
                self._setStatus(f"Running {i+1}/{len(tasks)}: {desc[:40]}")
                self._log(f"{'='*50}")
                self._log(f"TASK {i+1}/{len(tasks)}: {desc}")
                self._log(f"{'='*50}")

                # Update status in Firestore
                try:
                    self.db.collection('run_commands').document(docId).update({
                        'status': 'running',
                        'startedAt': _now().isoformat(),
                    })
                except Exception:
                    pass

                try:
                    result = _executeTask(cmd, self._log, dbClient=self.db, stop_check=stop_check)
                    self.db.collection('run_commands').document(docId).update({
                        'status': 'completed',
                        'completedAt': _now().isoformat(),
                        'result': result or {},
                    })
                    _updateSchedulerStateAfterTask(
                        cmd.get('type', 'full_pipeline'), self.db, self._log)
                    self._log(f"COMPLETED: {desc}")
                except Exception as e:
                    errMsg = traceback.format_exc()
                    self._log(f"FAILED: {e}\n{errMsg}")
                    try:
                        self.db.collection('run_commands').document(docId).update({
                            'status': 'failed',
                            'completedAt': _now().isoformat(),
                            'error': str(e)[:500],
                        })
                    except Exception:
                        pass

            self._setStatus("Idle")
            self._setButtons(True)
            self.running = False
            self.root.after(0, self._refreshQueue)
            self._log("All tasks finished.")

        self._currentThread = threading.Thread(target=_worker, daemon=True)
        self._currentThread.start()

    # ── Run the GUI loop ──────────────────────────────────────────────

    def run(self):
        self._log("Agent started. Waiting for tasks...")
        self.root.mainloop()


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        app = AgentApp()
        app.run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
