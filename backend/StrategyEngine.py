"""
StrategyEngine — AI-Generated & Statistical Cross-Stock Strategies
===================================================================

Bridges the gap between *single-stock pattern matching* and
*portfolio-level hedge-fund thinking* by generating, storing, and
featurising **cross-stock trading strategies**.

Strategy lifecycle:
  1. **Generate** — call OpenAI once (or load from DB) to produce
     strategies like "if AAPL & MSFT both drop > 2 % in a week and
     VIX spikes, go long GOOGL (diversification play)".
  2. **Store** — persist in SQLite via PersistenceManager so the API
     is NOT called every run.
  3. **Featurise** — convert stored strategies into numeric features
     that the ML models can train on.
  4. **Validate** — backtest each strategy's conditions against real
     data and update hit-rate / confidence.

Strategy types:
  - **hedge**            : reduce risk when correlated assets move together
  - **pairs_trade**      : long laggard / short leader when pair diverges
  - **sector_rotation**  : shift capital between sectors based on macro
  - **momentum**         : ride strong trends across related names
  - **mean_reversion**   : exploit temporary dislocations
  - **supply_chain**     : trade downstream when upstream signals fire
  - **earnings_spread**  : trade correlates around one stock's earnings

Key design principle (per user request):
  **ML models make ALL trading decisions.**  OpenAI is only called to
  *generate* strategy definitions and stock metadata.  These are stored
  and converted into ML features.  The API is never in the live
  decision loop.
"""

from __future__ import annotations

import json
import os
import hashlib
import datetime
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PersistenceManager import (
    PersistenceManager, StoredStrategy, CrossStockRule, StockMeta,
)

# openai is imported lazily inside StrategyEngine.__init__ (optional dependency)

# ═══════════════════════════════════════════════════════════════════════
# OpenAI strategy generation prompt
# ═══════════════════════════════════════════════════════════════════════

_STRATEGY_SYSTEM_PROMPT = """\
You are a senior quantitative portfolio manager at a systematic hedge fund.
Your job is to generate ACTIONABLE cross-stock trading strategies that a
machine learning model can learn from.

You will receive:
  1. A list of portfolio stocks with their sectors and relationships
  2. A request to generate trading strategies

For EACH strategy, output a JSON object with:
{
  "name": "<short name>",
  "type": "<hedge|pairs_trade|sector_rotation|momentum|mean_reversion|supply_chain|earnings_spread>",
  "description": "<one paragraph explanation>",
  "conditions": [
    {"metric": "<price_change_5d|vol_spike|correlation_break|sector_move|rsi_divergence|...>",
     "symbols": ["SYM1", "SYM2"],
     "operator": "<gt|lt|between|diverge>",
     "threshold": <number>,
     "timeframe": "<1d|5d|20d>"}
  ],
  "actions": [
    {"symbol": "SYM", "signal": "<BUY|SELL|REDUCE|INCREASE>",
     "size_pct": <0.0-1.0>, "reason": "<why>"}
  ],
  "confidence": <0.0-1.0>,
  "risk_notes": "<what could go wrong>"
}

Rules:
- Generate 8-15 strategies covering ALL types
- Be SPECIFIC with thresholds (not vague)
- Include both offensive (alpha) and defensive (hedging) strategies
- Consider supply chain relationships
- Consider sector correlations and when they break
- Output ONLY a JSON array of strategy objects, nothing else
"""

_STRATEGY_USER_TEMPLATE = """\
Portfolio stocks and their metadata:

{stock_info}

Known cross-stock relationships:
{relationships}

Generate 10-15 quantitative cross-stock trading strategies for this portfolio.
Include at least:
- 2 hedging strategies (reduce risk when signals align)
- 2 pairs trading strategies (exploit divergences)
- 2 sector rotation strategies
- 2 momentum/trend strategies
- 2 mean-reversion strategies
- 1 supply-chain strategy
Focus on strategies that CONNECT multiple stocks together.
"""

_METADATA_SYSTEM_PROMPT = """\
You are a financial data specialist. For the given stock ticker, provide
accurate sector/industry classification and key relationships.

Output ONLY valid JSON:
{
  "sector": "<GICS sector>",
  "industry": "<specific industry>",
  "market_cap_bucket": "<mega|large|mid|small>",
  "description": "<one sentence>",
  "related_tickers": ["SYM1", "SYM2", ...],
  "sector_peers": ["SYM1", ...],
  "supply_chain_upstream": ["SYM1", ...],
  "supply_chain_downstream": ["SYM1", ...],
  "competes_with": ["SYM1", ...]
}
"""

# Cheap models allowed
_ALLOWED_MODELS = frozenset({
    'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
    'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-3.5-turbo',
})


# ═══════════════════════════════════════════════════════════════════════
# Statistical strategy discovery (no API needed)
# ═══════════════════════════════════════════════════════════════════════

def discoverStatisticalRules(
    stockDataDict: Dict[str, pd.DataFrame],
    stockMeta: Dict[str, StockMeta],
    minCorrelation: float = 0.4,
    minSamples: int = 30,
) -> List[CrossStockRule]:
    """
    Discover cross-stock rules purely from price data — no API calls.

    Rules discovered:
      1. Sector correlation rules (same-sector stocks move together)
      2. Lead-lag rules (stock A predicts stock B with a delay)
      3. Divergence rules (correlated stocks diverge → mean-revert)
      4. Volatility contagion (vol spike in one → vol spike in peers)
      5. Supply-chain propagation (upstream move → downstream follows)
    """
    rules: List[CrossStockRule] = []
    now = _now().isoformat()
    symbols = sorted(stockDataDict.keys())

    if len(symbols) < 2:
        return rules

    # Align close prices
    closeFrames = {}
    for sym, df in stockDataDict.items():
        if df is not None and 'close' in df.columns and len(df) > 0:
            closeFrames[sym] = df['close']
    if len(closeFrames) < 2:
        return rules

    pricesDf = pd.DataFrame(closeFrames).dropna()
    returnsDf = pricesDf.pct_change().dropna()

    if len(returnsDf) < minSamples:
        return rules

    from itertools import combinations

    for s1, s2 in combinations(symbols, 2):
        if s1 not in returnsDf.columns or s2 not in returnsDf.columns:
            continue
        r1 = returnsDf[s1].values
        r2 = returnsDf[s2].values

        # 1. Sector correlation
        corr = float(np.corrcoef(r1, r2)[0, 1]) if len(r1) > 5 else 0
        if abs(corr) >= minCorrelation:
            rule = CrossStockRule(
                ruleId=hashlib.md5(f"corr_{s1}_{s2}".encode()).hexdigest()[:12],
                ruleType='sector_correlation',
                sourceSymbols=[s1], targetSymbols=[s2],
                conditionJson=json.dumps({
                    'metric': 'correlation',
                    'value': round(corr, 3),
                    'window': 60,
                }),
                actionJson=json.dumps({
                    'signal': 'SAME_DIRECTION' if corr > 0 else 'OPPOSITE_DIRECTION',
                    'strength': round(abs(corr), 2),
                }),
                confidence=min(1.0, abs(corr)),
                hitRate=0.0,
                sampleSize=len(r1),
                source='statistical',
                createdAt=now,
            )
            rules.append(rule)

        # 2. Lead-lag detection
        for lag in range(1, 6):
            if lag >= len(r1):
                continue
            # s1 leads s2
            c12 = float(np.corrcoef(r1[:-lag], r2[lag:])[0, 1]) if len(r1) > lag + 5 else 0
            if abs(c12) >= minCorrelation:
                rule = CrossStockRule(
                    ruleId=hashlib.md5(f"leadlag_{s1}_{s2}_{lag}".encode()).hexdigest()[:12],
                    ruleType='lead_lag',
                    sourceSymbols=[s1], targetSymbols=[s2],
                    conditionJson=json.dumps({
                        'leader': s1, 'follower': s2,
                        'lag_periods': lag,
                        'correlation': round(c12, 3),
                    }),
                    actionJson=json.dumps({
                        'signal': 'FOLLOW_LEADER' if c12 > 0 else 'CONTRA_LEADER',
                        'strength': round(abs(c12), 2),
                        'delay_periods': lag,
                    }),
                    confidence=min(1.0, abs(c12) / 0.5),
                    hitRate=0.0,
                    sampleSize=len(r1) - lag,
                    source='statistical',
                    createdAt=now,
                )
                rules.append(rule)
                break  # only keep best lag per pair per direction

        # 3. Divergence / mean-reversion
        if abs(corr) >= 0.5 and len(pricesDf) >= 60:
            ratio = pricesDf[s1] / (pricesDf[s2] + 1e-9)
            ratioMean = ratio.rolling(60).mean()
            ratioStd = ratio.rolling(60).std()
            zScore = (ratio - ratioMean) / (ratioStd + 1e-9)
            # Count how often a 2-sigma divergence reverts within 10 days
            extremes = abs(zScore) > 2.0
            nExtremes = extremes.sum()
            if nExtremes >= 5:
                reverted = 0
                for idx in zScore[extremes].index:
                    loc = zScore.index.get_loc(idx)
                    if loc + 10 < len(zScore):
                        if abs(zScore.iloc[loc + 10]) < abs(zScore.iloc[loc]):
                            reverted += 1
                hitRate = reverted / max(nExtremes, 1)
                rule = CrossStockRule(
                    ruleId=hashlib.md5(f"diverge_{s1}_{s2}".encode()).hexdigest()[:12],
                    ruleType='pairs_trade',
                    sourceSymbols=[s1, s2], targetSymbols=[s1, s2],
                    conditionJson=json.dumps({
                        'metric': 'spread_z_score',
                        'threshold': 2.0,
                        'base_correlation': round(corr, 3),
                        'window': 60,
                    }),
                    actionJson=json.dumps({
                        'signal': 'MEAN_REVERT',
                        'buy_laggard': True,
                        'sell_leader': True,
                    }),
                    confidence=round(hitRate, 2),
                    hitRate=round(hitRate, 2),
                    sampleSize=int(nExtremes),
                    source='statistical',
                    createdAt=now,
                )
                rules.append(rule)

    # 4. Volatility contagion (across same-sector stocks)
    for s1, s2 in combinations(symbols, 2):
        m1 = stockMeta.get(s1)
        m2 = stockMeta.get(s2)
        if m1 and m2 and m1.sector == m2.sector and m1.sector:
            if s1 in returnsDf.columns and s2 in returnsDf.columns:
                vol1 = returnsDf[s1].rolling(5).std()
                vol2 = returnsDf[s2].rolling(5).std()
                volCorr = float(vol1.corr(vol2)) if len(vol1.dropna()) > 20 else 0
                if volCorr > 0.5:
                    rule = CrossStockRule(
                        ruleId=hashlib.md5(f"volcontagion_{s1}_{s2}".encode()).hexdigest()[:12],
                        ruleType='sector_correlation',
                        sourceSymbols=[s1], targetSymbols=[s2],
                        conditionJson=json.dumps({
                            'metric': 'volatility_correlation',
                            'sector': m1.sector,
                            'vol_corr': round(volCorr, 3),
                        }),
                        actionJson=json.dumps({
                            'signal': 'VOL_CONTAGION',
                            'description': f'Vol spike in {s1} predicts vol in {s2}',
                        }),
                        confidence=round(min(1.0, volCorr), 2),
                        hitRate=0.0,
                        sampleSize=len(vol1.dropna()),
                        source='statistical',
                        createdAt=now,
                    )
                    rules.append(rule)

    return rules


# ═══════════════════════════════════════════════════════════════════════
# Strategy Engine
# ═══════════════════════════════════════════════════════════════════════

class StrategyEngine:
    """
    Generates, stores, and featurises cross-stock trading strategies.

    OpenAI is called **only** when no strategies exist in the DB for
    the current stock set (*or* when explicitly refreshed).  On
    subsequent runs the stored strategies are loaded directly — no API
    call needed.

    Parameters
    ----------
    persistence : PersistenceManager
    openAIKey : str, optional
    openAIModel : str
    """

    def __init__(
        self,
        persistence: PersistenceManager,
        openAIKey: Optional[str] = None,
        openAIModel: str = 'gpt-4o-mini',
    ):
        self.pm = persistence
        self._client = None
        self._model = openAIModel

        key = openAIKey or os.environ.get('OPENAI_API_KEY', '')
        if key and openAIModel in _ALLOWED_MODELS:
            try:
                import openai  # type: ignore[import-untyped]
                self._client = openai.OpenAI(api_key=key)
            except ImportError:
                pass

    @property
    def hasAI(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────────────────────────────

    def ensureStrategies(
        self,
        symbols: List[str],
        stockDataDict: Dict[str, pd.DataFrame],
        forceRefresh: bool = False,
        verbose: bool = True,
    ) -> List[StoredStrategy]:
        """
        Generate fresh strategies every run and merge with existing ones.

        Flow:
          1. Load existing active strategies from DB.
          2. Always discover new statistical rules + built-in + OpenAI strategies.
          3. Merge: if a new strategy has the same ID as an existing one, keep
             whichever has higher confidence. New unique strategies are added.
          4. Deactivate stale strategies whose symbols no longer overlap the
             current stock set.
          5. Store merged result in DB.

        Returns the combined list of active strategies.
        """
        existing = self.pm.loadActiveStrategies(symbols=None)
        existingById = {s.strategyId: s for s in existing if s.strategyId}

        if verbose:
            print(f"  [StrategyEngine] {len(existing)} existing strategies in DB")
            print(f"  [StrategyEngine] Discovering fresh strategies for {symbols} ...")

        fresh: List[StoredStrategy] = []

        # A. Statistical discovery (always, no API needed)
        stockMeta = self.pm.getAllStockMetadata()
        rules = discoverStatisticalRules(stockDataDict, stockMeta)
        if rules:
            self.pm.saveCrossStockRules(rules)
            if verbose:
                print(f"    Statistical: {len(rules)} cross-stock rules discovered")

        # B. Generate built-in rule-based strategies
        fresh.extend(self._generateBuiltinStrategies(symbols, stockMeta))

        # C. OpenAI strategies (only if API available and forceRefresh or few existing AI strats)
        existingAICount = sum(1 for s in existing if s.source == 'openai')
        if self.hasAI and (forceRefresh or existingAICount < 3):
            aiStrats = self._generateOpenAIStrategies(symbols, stockMeta, verbose)
            fresh.extend(aiStrats)
        elif verbose and not self.hasAI:
            print("    OpenAI not available — using built-in + statistical only")

        # D. Merge: new strategies replace old if higher confidence, otherwise keep old
        merged = dict(existingById)
        added, updated = 0, 0
        for s in fresh:
            sid = s.strategyId
            if not sid:
                continue
            old = merged.get(sid)
            if old is None:
                merged[sid] = s
                added += 1
            elif s.confidence > old.confidence:
                s.createdAt = old.createdAt or s.createdAt
                merged[sid] = s
                updated += 1

        # E. Deactivate stale strategies that no longer match any current symbol
        symSet = set(symbols)
        deactivated = 0
        now = _now().isoformat()
        for sid, s in list(merged.items()):
            if s.symbols and not (set(s.symbols) & symSet):
                s.active = False
                s.updatedAt = now
                deactivated += 1

        active = [s for s in merged.values() if s.active]

        # F. Quick backtest each active strategy to populate backtestReturn / Sharpe
        if stockDataDict:
            self._backtestStrategies(active, stockDataDict, verbose)

        # G. Persist all changes
        for s in merged.values():
            self.pm.saveStrategy(s)

        if verbose:
            print(f"    Merge result: {added} new, {updated} upgraded, "
                  f"{deactivated} deactivated, {len(active)} active total")

        return active

    def ensureStockMetadata(
        self,
        symbols: List[str],
        verbose: bool = True,
    ):
        """
        Ensure stock metadata exists for all symbols.
        Tries DB first, then OpenAI for missing ones, then defaults.
        OpenAI is only used for manual portfolio stocks, not automatic ones.
        """
        for sym in symbols:
            existing = self.pm.getStockMetadata(sym)
            if existing and existing.sector:
                continue

            # Skip OpenAI for automatic (connected) stocks — cost control
            isAutomatic = existing and existing.portfolioType == 'automatic'

            # Try OpenAI (manual stocks only)
            if self.hasAI and not isAutomatic:
                meta = self._fetchMetadataFromAI(sym)
                if meta:
                    self.pm.upsertStockMetadata(meta)
                    if verbose:
                        print(f"    [Metadata] {sym}: {meta.sector} / {meta.industry} (from OpenAI)")
                    continue

            # Default fallback
            meta = StockMeta(
                symbol=sym, sector='Unknown', industry='Unknown',
                marketCapBucket='unknown', description=f'{sym} stock',
                updatedAt=_now().isoformat(),
            )
            self.pm.upsertStockMetadata(meta)
            if verbose:
                print(f"    [Metadata] {sym}: Unknown sector (no OpenAI key)")

    def featuriseStrategies(
        self,
        strategies: List[StoredStrategy],
        stockDataDict: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Convert strategies into numeric feature vectors for ML.

        Returns {symbol: np.array} where each array has one value per
        active strategy (the strategy's signal strength for that symbol).
        """
        result: Dict[str, np.ndarray] = {}

        for sym in symbols:
            features: List[float] = []
            for strat in strategies:
                # Check if this strategy involves this symbol
                stratSymbols = strat.symbols
                if stratSymbols and sym not in stratSymbols:
                    features.append(0.0)
                    continue

                # Check strategy conditions against current data
                signal_strength = self._evaluateStrategy(strat, sym, stockDataDict)
                features.append(signal_strength)

            result[sym] = np.array(features, dtype=float)

        return result

    def featuriseCrossStockRules(
        self,
        symbols: List[str],
        stockDataDict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build a cross-stock rule feature DataFrame aligned to the common
        date index.  Features include rule signal strength per symbol.
        """
        rules = self.pm.loadCrossStockRules()
        if not rules:
            return pd.DataFrame()

        # Find common date index
        closeFrames = {s: df['close'] for s, df in stockDataDict.items()
                       if df is not None and 'close' in df.columns}
        if len(closeFrames) < 2:
            return pd.DataFrame()

        pricesDf = pd.DataFrame(closeFrames).dropna()
        returnsDf = pricesDf.pct_change().dropna()
        feats = pd.DataFrame(index=pricesDf.index)

        ruleIdx = 0
        for rule in rules:
            if rule.ruleType == 'lead_lag':
                cond = json.loads(rule.conditionJson)
                leader = cond.get('leader', '')
                follower = cond.get('follower', '')
                lagP = cond.get('lag_periods', 1)
                corrVal = cond.get('correlation', 0)

                if leader in returnsDf.columns and follower in returnsDf.columns:
                    # Feature: leader's return shifted by lag
                    feats[f'rule_leadlag_{leader}_{follower}'] = (
                        returnsDf[leader].shift(lagP) * corrVal
                    )
                    ruleIdx += 1

            elif rule.ruleType == 'pairs_trade':
                src = rule.sourceSymbols
                if len(src) >= 2 and src[0] in pricesDf.columns and src[1] in pricesDf.columns:
                    ratio = pricesDf[src[0]] / (pricesDf[src[1]] + 1e-9)
                    zScore = (ratio - ratio.rolling(60).mean()) / (ratio.rolling(60).std() + 1e-9)
                    feats[f'rule_pair_z_{src[0]}_{src[1]}'] = zScore
                    ruleIdx += 1

            elif rule.ruleType == 'sector_correlation':
                src = rule.sourceSymbols
                tgt = rule.targetSymbols
                if src and tgt and src[0] in returnsDf.columns and tgt[0] in returnsDf.columns:
                    feats[f'rule_sectcorr_{src[0]}_{tgt[0]}'] = (
                        returnsDf[src[0]].rolling(20).corr(returnsDf[tgt[0]])
                    )
                    ruleIdx += 1

        return feats

    # ──────────────────────────────────────────────────────────────────
    # Built-in strategy templates
    # ──────────────────────────────────────────────────────────────────

    def _generateBuiltinStrategies(
        self, symbols: List[str], stockMeta: Dict[str, StockMeta]
    ) -> List[StoredStrategy]:
        """Generate universal quant strategies that work for any stock set."""
        now = _now().isoformat()
        strats: List[StoredStrategy] = []

        # 1. SECTOR HEDGE: when all same-sector stocks drop, go defensive
        sectors = {}
        for sym in symbols:
            m = stockMeta.get(sym)
            if m and m.sector:
                sectors.setdefault(m.sector, []).append(sym)

        for sector, syms in sectors.items():
            if len(syms) >= 2:
                strats.append(StoredStrategy(
                    strategyId=hashlib.md5(f"sector_hedge_{sector}".encode()).hexdigest()[:12],
                    name=f'{sector} Sector Hedge',
                    description=f'When all {sector} stocks drop >2% in 5 days, reduce exposure.',
                    strategyType='hedge',
                    conditionsJson=json.dumps([{
                        'metric': 'sector_return_5d',
                        'symbols': syms,
                        'operator': 'all_lt',
                        'threshold': -0.02,
                        'timeframe': '5d',
                    }]),
                    actionsJson=json.dumps([{
                        'signal': 'REDUCE',
                        'symbols': syms,
                        'size_pct': 0.25,
                        'reason': f'{sector} sector-wide weakness',
                    }]),
                    symbols=syms,
                    confidence=0.65,
                    source='rule_based',
                    active=True,
                    createdAt=now,
                    updatedAt=now,
                ))

        # 2. MOMENTUM DIVERGENCE: strongest stock keeps outperforming
        if len(symbols) >= 2:
            strats.append(StoredStrategy(
                strategyId=hashlib.md5(f"momentum_div_{'_'.join(symbols)}".encode()).hexdigest()[:12],
                name='Momentum Divergence',
                description='Overweight the stock with strongest 20d momentum, underweight the weakest.',
                strategyType='momentum',
                conditionsJson=json.dumps([{
                    'metric': 'relative_strength_20d',
                    'symbols': symbols,
                    'operator': 'rank_spread_gt',
                    'threshold': 0.03,
                    'timeframe': '20d',
                }]),
                actionsJson=json.dumps([{
                    'signal': 'OVERWEIGHT_LEADER',
                    'size_pct': 0.1,
                    'reason': 'Momentum continuation',
                }]),
                symbols=symbols,
                confidence=0.55,
                source='rule_based',
                active=True,
                createdAt=now,
                updatedAt=now,
            ))

        # 3. CORRELATION BREAKDOWN HEDGE
        if len(symbols) >= 2:
            strats.append(StoredStrategy(
                strategyId=hashlib.md5(f"corr_break_{'_'.join(symbols)}".encode()).hexdigest()[:12],
                name='Correlation Breakdown Warning',
                description='When portfolio correlation drops sharply, increase hedging.',
                strategyType='hedge',
                conditionsJson=json.dumps([{
                    'metric': 'avg_correlation_change_10d',
                    'symbols': symbols,
                    'operator': 'lt',
                    'threshold': -0.15,
                    'timeframe': '10d',
                }]),
                actionsJson=json.dumps([{
                    'signal': 'HEDGE',
                    'size_pct': 0.15,
                    'reason': 'Correlation regime change detected',
                }]),
                symbols=symbols,
                confidence=0.60,
                source='rule_based',
                active=True,
                createdAt=now,
                updatedAt=now,
            ))

        # 4. VOLATILITY MEAN-REVERSION
        for sym in symbols:
            strats.append(StoredStrategy(
                strategyId=hashlib.md5(f"vol_mr_{sym}".encode()).hexdigest()[:12],
                name=f'{sym} Vol Mean-Reversion',
                description=f'When {sym} vol is >2 std above mean, expect reversion (reduce size).',
                strategyType='mean_reversion',
                conditionsJson=json.dumps([{
                    'metric': 'vol_z_score',
                    'symbols': [sym],
                    'operator': 'gt',
                    'threshold': 2.0,
                    'timeframe': '20d',
                }]),
                actionsJson=json.dumps([{
                    'signal': 'REDUCE',
                    'symbol': sym,
                    'size_pct': 0.2,
                    'reason': f'{sym} vol spike — expect reversion',
                }]),
                symbols=[sym],
                confidence=0.60,
                source='rule_based',
                active=True,
                createdAt=now,
                updatedAt=now,
            ))

        # 5. PAIRS TRADE: for each pair of same-sector stocks
        for sector, syms in sectors.items():
            if len(syms) >= 2:
                from itertools import combinations
                for s1, s2 in combinations(syms, 2):
                    strats.append(StoredStrategy(
                        strategyId=hashlib.md5(f"pair_{s1}_{s2}".encode()).hexdigest()[:12],
                        name=f'Pairs: {s1}/{s2}',
                        description=f'When {s1}/{s2} spread z-score exceeds ±2, trade mean-reversion.',
                        strategyType='pairs_trade',
                        conditionsJson=json.dumps([{
                            'metric': 'spread_z_score',
                            'symbols': [s1, s2],
                            'operator': 'abs_gt',
                            'threshold': 2.0,
                            'timeframe': '60d',
                        }]),
                        actionsJson=json.dumps([
                            {'signal': 'BUY_LAGGARD', 'symbols': [s1, s2], 'size_pct': 0.1},
                            {'signal': 'SELL_LEADER', 'symbols': [s1, s2], 'size_pct': 0.1},
                        ]),
                        symbols=[s1, s2],
                        confidence=0.50,
                        source='rule_based',
                        active=True,
                        createdAt=now,
                        updatedAt=now,
                    ))

        # 6. SUPPLY CHAIN: if upstream supplier moves, trade downstream
        for sym in symbols:
            m = stockMeta.get(sym)
            if m and m.supplyChainDown:
                downstream = [d for d in m.supplyChainDown if d in symbols]
                if downstream:
                    strats.append(StoredStrategy(
                        strategyId=hashlib.md5(f"supply_{sym}_down".encode()).hexdigest()[:12],
                        name=f'Supply Chain: {sym} → {",".join(downstream)}',
                        description=f'When {sym} (supplier) surges, downstream {downstream} may follow.',
                        strategyType='supply_chain',
                        conditionsJson=json.dumps([{
                            'metric': 'price_change_3d',
                            'symbols': [sym],
                            'operator': 'gt',
                            'threshold': 0.03,
                            'timeframe': '3d',
                        }]),
                        actionsJson=json.dumps([{
                            'signal': 'BUY',
                            'symbols': downstream,
                            'size_pct': 0.05,
                            'reason': f'Supply chain propagation from {sym}',
                        }]),
                        symbols=[sym] + downstream,
                        confidence=0.45,
                        source='rule_based',
                        active=True,
                        createdAt=now,
                        updatedAt=now,
                    ))

        return strats

    # ──────────────────────────────────────────────────────────────────
    # OpenAI strategy generation
    # ──────────────────────────────────────────────────────────────────

    def _generateOpenAIStrategies(
        self, symbols: List[str], stockMeta: Dict[str, StockMeta],
        verbose: bool = True,
    ) -> List[StoredStrategy]:
        """Call OpenAI to generate advanced cross-stock strategies."""
        if not self.hasAI:
            return []

        # Build stock info string
        stockInfo = []
        for sym in symbols:
            m = stockMeta.get(sym)
            if m:
                stockInfo.append(
                    f"  {sym}: {m.sector} / {m.industry} | "
                    f"Peers: {m.sectorPeers[:5]} | "
                    f"Competes: {m.competesWidth[:5]} | "
                    f"Supply-up: {m.supplyChainUp[:3]} | "
                    f"Supply-down: {m.supplyChainDown[:3]}"
                )
            else:
                stockInfo.append(f"  {sym}: (no metadata)")

        # Build relationships string
        rules = self.pm.loadCrossStockRules()
        relStr = "None discovered yet" if not rules else "\n".join(
            f"  {r.ruleType}: {r.sourceSymbols} → {r.targetSymbols} "
            f"(conf={r.confidence:.2f})"
            for r in rules[:10]
        )

        userMsg = _STRATEGY_USER_TEMPLATE.format(
            stock_info="\n".join(stockInfo),
            relationships=relStr,
        )

        try:
            from OpenAIRetry import with_retry
            def _call():
                return self._client.chat.completions.create(
                    model=self._model,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': _STRATEGY_SYSTEM_PROMPT},
                        {'role': 'user', 'content': userMsg},
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                )
            response = with_retry(_call)
            raw = json.loads(response.choices[0].message.content)

            # Handle both {"strategies": [...]} and direct array
            if isinstance(raw, list):
                stratList = raw
            elif isinstance(raw, dict):
                stratList = raw.get('strategies', raw.get('data', [raw]))
            else:
                stratList = []

            now = _now().isoformat()
            results = []
            for item in stratList:
                if not isinstance(item, dict):
                    continue
                name = item.get('name', 'AI Strategy')
                sid = hashlib.md5(f"ai_{name}_{now}".encode()).hexdigest()[:12]
                strat = StoredStrategy(
                    strategyId=sid,
                    name=name,
                    description=item.get('description', ''),
                    strategyType=item.get('type', 'hedge'),
                    conditionsJson=json.dumps(item.get('conditions', [])),
                    actionsJson=json.dumps(item.get('actions', [])),
                    symbols=symbols,
                    confidence=float(item.get('confidence', 0.5)),
                    source='openai',
                    active=True,
                    createdAt=now,
                    updatedAt=now,
                )
                results.append(strat)

            if verbose:
                print(f"    OpenAI generated {len(results)} strategies")
            return results

        except Exception as e:
            warnings.warn(f"[StrategyEngine] OpenAI strategy generation failed: {e}")
            return []

    def _fetchMetadataFromAI(self, symbol: str) -> Optional[StockMeta]:
        """Fetch stock metadata from OpenAI for unknown tickers."""
        if not self.hasAI:
            return None
        try:
            from OpenAIRetry import with_retry
            def _call():
                return self._client.chat.completions.create(
                    model=self._model,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': _METADATA_SYSTEM_PROMPT},
                        {'role': 'user', 'content': f'Provide metadata for ticker: {symbol}'},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
            response = with_retry(_call)
            data = json.loads(response.choices[0].message.content)
            return StockMeta(
                symbol=symbol,
                sector=data.get('sector', ''),
                industry=data.get('industry', ''),
                marketCapBucket=data.get('market_cap_bucket', ''),
                description=data.get('description', ''),
                relatedTickers=data.get('related_tickers', []),
                sectorPeers=data.get('sector_peers', []),
                supplyChainUp=data.get('supply_chain_upstream', []),
                supplyChainDown=data.get('supply_chain_downstream', []),
                competesWidth=data.get('competes_with', []),
                updatedAt=_now().isoformat(),
            )
        except Exception as e:
            warnings.warn(f"[StrategyEngine] metadata fetch failed for {symbol}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    # Strategy quick-backtest (populate backtestReturn / Sharpe)
    # ──────────────────────────────────────────────────────────────────

    def _backtestStrategies(
        self,
        strategies: List[StoredStrategy],
        stockDataDict: Dict[str, pd.DataFrame],
        verbose: bool = False,
    ):
        """Evaluate each strategy over a rolling window to estimate its
        directional accuracy and hypothetical return / Sharpe."""
        WINDOW = 5  # evaluate every 5 bars
        for strat in strategies:
            try:
                syms = [s for s in (strat.symbols or []) if s in stockDataDict]
                if not syms:
                    continue
                tradeReturns: List[float] = []
                for sym in syms:
                    df = stockDataDict[sym]
                    if df is None or len(df) < 60:
                        continue
                    for end in range(60, len(df) - WINDOW, WINDOW):
                        slicedDict = {s: stockDataDict[s].iloc[:end]
                                      for s in syms if s in stockDataDict
                                      and len(stockDataDict[s]) >= end}
                        if not slicedDict:
                            continue
                        sig = self._evaluateStrategy(strat, sym, slicedDict)
                        if abs(sig) < 0.05:
                            continue
                        fwd = df.iloc[end:end + WINDOW]
                        if len(fwd) < 2:
                            continue
                        fwdRet = (fwd['close'].iloc[-1] - fwd['close'].iloc[0]) / (fwd['close'].iloc[0] + 1e-9)
                        tradeRet = fwdRet * np.sign(sig) * abs(sig)
                        tradeReturns.append(float(tradeRet))
                if len(tradeReturns) >= 3:
                    arr = np.array(tradeReturns)
                    strat.backtestReturn = float(np.sum(arr) * 100)
                    mu, sigma = float(arr.mean()), float(arr.std())
                    strat.backtestSharpe = float((mu / sigma) * (252 ** 0.5)) if sigma > 1e-9 else 0.0
                    strat.updatedAt = _now().isoformat()
                    if verbose:
                        print(f"      {strat.name}: return={strat.backtestReturn:+.2f}%, "
                              f"sharpe={strat.backtestSharpe:.2f} ({len(tradeReturns)} samples)")
            except Exception:
                continue

    # ──────────────────────────────────────────────────────────────────
    # Strategy evaluation (for featurisation)
    # ──────────────────────────────────────────────────────────────────

    def _evaluateStrategy(
        self,
        strat: StoredStrategy,
        symbol: str,
        stockDataDict: Dict[str, pd.DataFrame],
    ) -> float:
        """
        Evaluate how strongly a strategy is currently signalling for
        a given symbol.  Returns a float in [-1, +1].
        """
        try:
            conditions = json.loads(strat.conditionsJson)
            if not conditions:
                return 0.0

            totalSignal = 0.0
            nConditions = 0

            for cond in conditions:
                metric = cond.get('metric', '')
                condSymbols = cond.get('symbols', [])
                threshold = cond.get('threshold', 0)
                timeframe = cond.get('timeframe', '5d')

                # Parse timeframe to periods
                periods = self._parsePeriods(timeframe)

                if metric in ('price_change_5d', 'price_change_3d',
                              'sector_return_5d', 'relative_strength_20d'):
                    # Check if condition is met
                    signals = []
                    for s in condSymbols:
                        df = stockDataDict.get(s)
                        if df is not None and 'close' in df.columns and len(df) >= periods + 1:
                            ret = (df['close'].iloc[-1] - df['close'].iloc[-periods-1]) / \
                                  (df['close'].iloc[-periods-1] + 1e-9)
                            signals.append(float(ret))

                    if signals:
                        op = cond.get('operator', 'gt')
                        if op == 'all_lt':
                            met = all(s < threshold for s in signals)
                        elif op == 'gt':
                            met = np.mean(signals) > threshold
                        elif op == 'lt':
                            met = np.mean(signals) < threshold
                        else:
                            met = abs(np.mean(signals)) > abs(threshold)

                        if met:
                            totalSignal += strat.confidence * (1.0 if met else 0.0)
                        nConditions += 1

                elif metric == 'vol_z_score':
                    for s in condSymbols:
                        df = stockDataDict.get(s)
                        if df is not None and 'close' in df.columns and len(df) >= 60:
                            ret = df['close'].pct_change()
                            vol = ret.rolling(20).std()
                            volMean = vol.rolling(60).mean()
                            volStd = vol.rolling(60).std()
                            if len(vol.dropna()) > 0 and float(volStd.iloc[-1] or 0) > 0:
                                z = (vol.iloc[-1] - volMean.iloc[-1]) / (volStd.iloc[-1] + 1e-9)
                                if float(z) > threshold:
                                    totalSignal += strat.confidence
                            nConditions += 1

            if nConditions == 0:
                return 0.0

            # Determine direction from actions
            actions = json.loads(strat.actionsJson)
            direction = 0.0
            for act in actions:
                sig = act.get('signal', 'HOLD')
                actSymbols = act.get('symbols', act.get('symbol', ''))
                if isinstance(actSymbols, str):
                    actSymbols = [actSymbols]
                if symbol in actSymbols or not actSymbols:
                    if sig in ('BUY', 'BUY_LAGGARD', 'OVERWEIGHT_LEADER', 'INCREASE'):
                        direction = 1.0
                    elif sig in ('SELL', 'SELL_LEADER', 'REDUCE', 'HEDGE'):
                        direction = -1.0

            return float(np.clip(totalSignal / max(nConditions, 1) * direction, -1, 1))

        except Exception:
            return 0.0

    @staticmethod
    def _parsePeriods(timeframe: str) -> int:
        """Convert '5d', '20d', '1h' etc. to integer periods."""
        try:
            if timeframe.endswith('d'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1])
            elif timeframe.endswith('m'):
                return max(1, int(timeframe[:-1]) // 30)
        except ValueError:
            pass
        return 5
