"""
SentimentAnalysis — Three-Layer Financial Sentiment Engine
==========================================================

Produces a daily sentiment score per stock by combining three independent
signal layers in a confidence-weighted ensemble:

  Layer 1 — Financial Lexicon Scorer
      Rule-based keyword/phrase scorer using an embedded Loughran-McDonald
      inspired financial word list.  Transparent, zero training required,
      runs on every headline.

  Layer 2 — Structural ML Scorer
      TF-IDF n-gram (1–3) feature extraction + GradientBoosting classifier
      trained on labeled financial headlines (seed data is embedded; the
      model accumulates new labels over time via ``recordOutcome``).
      Captures phrase-level patterns that simple keyword matching misses:
      "beats estimates", "revenue miss", "better than expected", etc.

  Layer 3 — OpenAI Deep Analyst
      Selectively applied to the highest-importance articles only, gated
      by an ``ArticleImportanceScorer``.  Also used as a tiebreaker when
      Layer 1 and Layer 2 disagree significantly (|score diff| > 0.35).
      Uses GPT-4o-mini with a JSON-mode structured prompt.

Final daily score aggregation:
  - Each article in the lookback window is scored by whichever layers ran.
  - Per-article scores are ensemble-combined with adaptive layer weights.
  - Articles are decay-weighted by recency (half-life = 3 days).
  - The resulting score per symbol per day is in [-1, +1].

Public interface::

    analyser = SentimentAnalyzer(openAIKey='sk-...')
    sentimentDict = analyser.fetchAndScore(
        symbols=['AAPL', 'MSFT'],
        lookbackDays=30,
    )
    # sentimentDict['AAPL'] → pd.Series indexed by date, values in [-1, +1]
"""

from __future__ import annotations

import re
import os
import json
import math
import time
import hashlib
import warnings
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import yfinance as yf

try:
    import requests as _requests  # available via yfinance dependency
except ImportError:
    _requests = None  # type: ignore

warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ArticleSentiment:
    """Sentiment scores for a single news article across all layers."""
    headline: str
    url: str = ''
    source: str = ''
    publishedAt: Optional[datetime.datetime] = None
    symbol: str = ''

    # Layer scores (None = layer did not run for this article)
    lexiconScore:    Optional[float] = None   # Layer 1: -1 → +1
    structuralScore: Optional[float] = None   # Layer 2: -1 → +1
    openAIScore:     Optional[float] = None   # Layer 3: -1 → +1
    openAIReasoning: str = ''
    openAIDrivers:   List[str] = field(default_factory=list)

    # Ensemble output
    ensembleScore: float = 0.0     # weighted combination of available layers
    ensembleConf:  float = 0.0     # confidence (0–1) of the ensemble
    importanceScore: float = 0.0   # how important this article is (0–1)


@dataclass
class DailySentiment:
    """Aggregated sentiment for one symbol on one date."""
    symbol: str
    date: datetime.date
    score: float = 0.0          # decay-weighted ensemble score, -1 → +1
    confidence: float = 0.0     # avg ensemble confidence of articles
    articleCount: int = 0
    openAIArticleCount: int = 0
    topHeadlines: List[str] = field(default_factory=list)


# =============================================================================
# Layer 1 — Financial Lexicon Scorer
# =============================================================================

# Loughran-McDonald inspired financial word lists (embedded)
_POSITIVE_WORDS: frozenset = frozenset({
    # Earnings / revenue beats
    'beats', 'beat', 'surpasses', 'surpass', 'exceeds', 'exceed', 'topped',
    'tops', 'outperforms', 'outperform', 'outperforming', 'record',
    # Price / market action
    'rally', 'rallies', 'surge', 'surges', 'soars', 'soar', 'climbs',
    'jumps', 'jump', 'rises', 'rise', 'gains', 'gain', 'rebounds', 'rebound',
    # Corporate signals
    'upgraded', 'upgrade', 'raises', 'raised', 'boosts', 'boost', 'lifts', 'lift',
    'accelerates', 'accelerate', 'expansion', 'expands', 'expand', 'wins', 'win',
    'awarded', 'partnership', 'acquisition', 'acquires', 'dividend', 'buyback',
    'repurchase', 'reiterated', 'above', 'milestone', 'launches',
    # Qualitative
    'strong', 'strength', 'robust', 'solid', 'exceptional', 'profitability',
    'profitable', 'growth', 'growing', 'recovery', 'recovers', 'recover',
    'bullish', 'optimistic', 'confidence', 'confident', 'favorable', 'momentum',
    'opportunity', 'innovative', 'innovation', 'breakout', 'breakthrough',
    'promising', 'positive', 'improving', 'improved', 'improve',
})

_NEGATIVE_WORDS: frozenset = frozenset({
    # Earnings / guidance misses
    'misses', 'miss', 'shortfall', 'disappoints', 'disappoint',
    'disappointing', 'disappointed', 'below', 'missed',
    # Price / market action
    'drops', 'drop', 'plunges', 'plunge', 'slumps', 'slump', 'crashes', 'crash',
    'declines', 'decline', 'falls', 'fall', 'loses', 'loss', 'losses', 'selloff',
    'tumbles', 'tumble', 'sinks', 'sink', 'slides', 'slide',
    # Corporate signals
    'cuts', 'cut', 'reduces', 'reduce', 'suspended', 'suspends', 'suspend',
    'downgrade', 'downgraded', 'warns', 'warning', 'lowered', 'lowers', 'lower',
    'layoffs', 'layoff', 'fired', 'halt', 'halted',
    # Legal / regulatory
    'lawsuit', 'sued', 'probe', 'investigation', 'investigated', 'fine', 'fined',
    'penalty', 'penalized', 'violation', 'fraud', 'accused',
    # Distress
    'bankruptcy', 'bankrupt', 'defaulted', 'default', 'abandoned', 'failure',
    'failed', 'fail', 'struggling', 'struggle', 'pressure', 'headwinds',
    # Qualitative
    'weak', 'weakness', 'difficult', 'challenging', 'negative', 'bearish',
    'pessimistic', 'cautious', 'uncertain', 'uncertainty', 'volatile', 'risk',
    'recession', 'slowdown', 'contraction', 'downside',
})

# High-weight intensifiers (multiply word score when adjacent)
_INTENSIFIERS: frozenset = frozenset({
    'sharply', 'significantly', 'dramatically', 'substantially', 'massively',
    'heavily', 'greatly', 'widely', 'strongly', 'extremely', 'major', 'record',
    'historic', 'unprecedented', 'biggest', 'largest', 'worst', 'best',
})

# Negation words (flip the sign of the next scored word)
_NEGATION_WORDS: frozenset = frozenset({
    'not', "n't", 'no', 'never', 'neither', 'nor', 'without', 'fails', 'despite',
})

# High-importance trigger words (signals that article deserves OpenAI attention)
IMPORTANCE_TIER1: frozenset = frozenset({
    'earnings', 'revenue', 'profit', 'guidance', 'forecast', 'outlook',
    'fed', 'federal reserve', 'interest rate', 'inflation', 'cpi',
    'gdp', 'unemployment', 'recession', 'crisis',
})
IMPORTANCE_TIER2: frozenset = frozenset({
    'acquisition', 'merger', 'buyout', 'bankruptcy', 'default', 'lawsuit',
    'investigation', 'dividend', 'split', 'buyback', 'ipo',
})
IMPORTANCE_TIER3: frozenset = frozenset({
    'upgrade', 'downgrade', 'target', 'price target', 'beats', 'misses',
    'raises', 'cuts', 'guidance', 'layoffs', 'ceo', 'cfo',
})

# Trusted financial news sources (higher weight in aggregation)
_SOURCE_WEIGHTS: Dict[str, float] = {
    'reuters': 1.0, 'bloomberg': 1.0, 'wsj': 1.0, 'wall street journal': 1.0,
    'financial times': 1.0, 'ft': 1.0, 'barrons': 0.9, 'marketwatch': 0.85,
    'cnbc': 0.85, 'yahoo finance': 0.80, 'seeking alpha': 0.70,
    'motley fool': 0.65, 'benzinga': 0.70, 'thestreet': 0.65,
    'investing.com': 0.70, 'zacks': 0.75,
}


class KeywordLexiconScorer:
    """
    Layer 1: Fast, transparent lexicon-based headline scorer.

    Uses the embedded Loughran-McDonald inspired word lists plus
    negation handling and intensifier boosting.

    Returns a score in [-1, +1] and a confidence in [0, 1].
    Confidence reflects the fraction of tokens that were identifiable
    as sentiment-bearing (low confidence = ambiguous / neutral headline).
    """

    def score(self, headline: str) -> Tuple[float, float]:
        """
        Score a single headline.

        Returns
        -------
        (score, confidence) where score ∈ [-1, +1], confidence ∈ [0, 1]
        """
        tokens = re.findall(r"[a-z']+", headline.lower())
        if not tokens:
            return 0.0, 0.0

        totalScore = 0.0
        scoredTokens = 0
        negated = False
        intensified = False
        gapSinceModifier = 0          # tokens since last modifier

        for token in tokens:
            if token in _NEGATION_WORDS:
                negated = True
                gapSinceModifier = 0
                continue
            if token in _INTENSIFIERS:
                intensified = True
                gapSinceModifier = 0
                continue

            wordScore = 0.0
            if token in _POSITIVE_WORDS:
                wordScore = +1.0
            elif token in _NEGATIVE_WORDS:
                wordScore = -1.0

            if wordScore != 0.0:
                if negated:
                    wordScore *= -0.8    # partial flip (not fully negative)
                if intensified:
                    wordScore *= 1.4
                totalScore += wordScore
                scoredTokens += 1
                # Reset modifiers only after they are consumed by a sentiment word
                negated = False
                intensified = False
                gapSinceModifier = 0
            else:
                # Expire stale modifiers after 3 non-sentiment tokens
                gapSinceModifier += 1
                if gapSinceModifier >= 3:
                    negated = False
                    intensified = False

        if scoredTokens == 0:
            return 0.0, 0.0

        rawScore = totalScore / max(scoredTokens, 1)
        normScore = max(-1.0, min(1.0, rawScore))

        # Confidence: higher when more sentiment words found relative to headline length
        confidence = min(1.0, scoredTokens / max(len(tokens) * 0.4, 1))
        return normScore, confidence

    def batchScore(self, headlines: List[str]) -> List[Tuple[float, float]]:
        return [self.score(h) for h in headlines]


# =============================================================================
# Seed training data for Layer 2
# =============================================================================

# Label: +1 = bullish, 0 = neutral, -1 = bearish
# Covers the main headline archetypes the structural ML model must learn.
_SEED_HEADLINES: List[Tuple[str, int]] = [
    # ── Strong bullish ──────────────────────────────────────────────────────
    ("Company beats earnings estimates by 20% and raises full-year guidance", +1),
    ("Revenue surges to record high, profit margin expands significantly", +1),
    ("Stock surges after analysts upgrade to buy with increased price target", +1),
    ("Quarterly earnings smash expectations as revenue tops forecasts", +1),
    ("Company announces record dividend increase and share buyback program", +1),
    ("Strong consumer demand drives better than expected sales growth", +1),
    ("FDA approves new blockbuster drug ahead of schedule", +1),
    ("Acquisition deal expected to significantly boost earnings per share", +1),
    ("Management raises guidance citing robust demand across all segments", +1),
    ("Stock jumps after company reports best quarter in its history", +1),
    ("Analyst upgrades to overweight citing undervalued growth potential", +1),
    ("Company wins major government contract worth billions", +1),
    ("Net profit rises sharply as cost-cutting measures exceed targets", +1),
    ("Recovery accelerates faster than expected boosting investor confidence", +1),
    ("Breakthrough product launch drives unprecedented subscriber growth", +1),
    ("Services revenue grows 30% year over year beating all estimates", +1),
    ("Company expands into high-growth market with strategic partnership", +1),
    ("Institutional investors increase holdings as fundamentals strengthen", +1),
    ("Free cash flow hits record high enabling aggressive capital returns", +1),
    ("Gross margin improves as supply chain pressures ease significantly", +1),
    # ── Mild bullish ────────────────────────────────────────────────────────
    ("Company reports solid quarterly results in line with expectations", +1),
    ("Steady growth in core business provides stable revenue outlook", +1),
    ("Management remains optimistic about second half recovery", +1),
    ("Stock rises modestly after mixed results with improving margins", +1),
    ("Analyst reiterates buy rating with modest price target increase", +1),
    ("Company shows steady improvement in operating efficiency", +1),
    # ── Neutral ─────────────────────────────────────────────────────────────
    ("Company announces new product lineup scheduled for next quarter", 0),
    ("CEO speaks at annual investor day discussing long-term strategy", 0),
    ("Quarterly results meet analyst expectations with no major surprises", 0),
    ("Company appoints new chief financial officer effective next month", 0),
    ("Board of directors approves annual meeting date", 0),
    ("Company to present at upcoming industry conference", 0),
    ("Annual report filed with securities regulators as scheduled", 0),
    ("Management reaffirms existing full-year guidance range", 0),
    ("Shares trade sideways as market awaits upcoming earnings report", 0),
    ("Company completes planned share repurchase within stated parameters", 0),
    # ── Mild bearish ────────────────────────────────────────────────────────
    ("Company reports mixed results as some segments underperform", -1),
    ("Growth slows as competition intensifies in core markets", -1),
    ("Management sounds cautious tone on near-term demand outlook", -1),
    ("Analyst downgrades to neutral citing near-term headwinds", -1),
    ("Revenue growth decelerates raising questions about valuation", -1),
    ("Company faces margin pressure from rising input costs", -1),
    # ── Strong bearish ──────────────────────────────────────────────────────
    ("Company misses earnings estimates and slashes full-year guidance", -1),
    ("Revenue collapses as key product fails to gain traction with consumers", -1),
    ("Stock plunges after company reports massive unexpected quarterly loss", -1),
    ("Bankruptcy fears mount as company struggles to meet debt obligations", -1),
    ("SEC investigation launched into company accounting practices", -1),
    ("CEO resigns amid fraud allegations sending stock into freefall", -1),
    ("Company announces massive layoffs and facility closures", -1),
    ("Credit rating downgraded to junk status citing deteriorating finances", -1),
    ("Drug fails phase 3 trial wiping out years of development investment", -1),
    ("Regulator blocks acquisition deal citing antitrust concerns", -1),
    ("Company warns of significant earnings shortfall due to weak demand", -1),
    ("Major customer cancels contract putting future revenue at risk", -1),
    ("Gross margin collapses due to supply chain disruptions and rising costs", -1),
    ("Class action lawsuit filed alleging company misled investors", -1),
    ("Cash burn accelerates raising serious going concern doubts", -1),
    ("Product recall costs expected to significantly impact profitability", -1),
    ("Federal Reserve signals faster rate hikes amid persistent inflation", -1),
    ("GDP growth slows sharply as recession fears intensify", -1),
    ("Tariffs imposed sending supply chain costs surging for manufacturers", -1),
    ("Market selloff deepens as inflation data comes in hotter than expected", -1),
    # ── Mixed / tricky ──────────────────────────────────────────────────────
    ("Company beats revenue estimates but misses on earnings per share", 0),
    ("Strong sales growth offset by rising operating expenses", 0),
    ("Guidance raised for revenue but lowered for profit margins", 0),
    ("Company reports record revenue but records one-time impairment charge", 0),
    ("Market rallies on jobs data despite ongoing inflation concerns", 0),
    ("Results above expectations on top line but guidance disappoints", 0),
]


# =============================================================================
# Layer 2 — Structural ML Scorer
# =============================================================================

class StructuralMLScorer:
    """
    Layer 2: TF-IDF n-gram (1–3) + GradientBoosting classifier.

    Trained on embedded seed headline data; accumulates new labeled
    examples and retrains via ``addLabeledExamples``.

    Captures phrase-level patterns that keyword scoring misses:
      'beats estimates', 'below expectations', 'raises guidance',
      'better than expected', 'fails to meet', etc.

    Returns a score in [-1, +1] and a confidence in [0, 1].
    """

    def __init__(self, minTrainSamples: int = 20):
        self.minTrainSamples = minTrainSamples
        self._pipeline: Optional[Pipeline] = None
        self._labelEncoder = LabelEncoder()
        self._trained: bool = False
        self._corpus: List[Tuple[str, int]] = list(_SEED_HEADLINES)
        self._train()

    # ── Public ──────────────────────────────────────────────────────────────

    def score(self, headline: str) -> Tuple[float, float]:
        """
        Score one headline.
        Returns (score, confidence) where score ∈ [-1, +1].
        """
        if not self._trained or not headline.strip():
            return 0.0, 0.0
        try:
            proba = self._pipeline.predict_proba([headline])[0]
            classes = self._labelEncoder.classes_
            score = sum(
                (1.0 if cls == 1 else -1.0 if cls == -1 else 0.0) * prob
                for cls, prob in zip(classes, proba)
            )
            return float(score), float(np.max(proba))
        except Exception:
            return 0.0, 0.0

    def batchScore(self, headlines: List[str]) -> List[Tuple[float, float]]:
        if not self._trained or not headlines:
            return [(0.0, 0.0)] * len(headlines)
        try:
            probas = self._pipeline.predict_proba(headlines)
            classes = self._labelEncoder.classes_
            results = []
            for proba in probas:
                score = sum(
                    (1.0 if cls == 1 else -1.0 if cls == -1 else 0.0) * prob
                    for cls, prob in zip(classes, proba)
                )
                results.append((float(score), float(np.max(proba))))
            return results
        except Exception:
            return [(0.0, 0.0)] * len(headlines)

    def addLabeledExamples(self, examples: List[Tuple[str, int]]):
        """
        Append new labeled (headline, label) pairs and retrain.
        Label must be -1, 0, or +1.
        """
        self._corpus.extend(examples)
        self._train()

    def crossValidationScore(self) -> float:
        """Return 3-fold CV accuracy on current corpus."""
        if len(self._corpus) < 30 or not self._trained:
            return 0.0
        texts  = [h for h, _ in self._corpus]
        labels = [l for _, l in self._corpus]
        cv = cross_val_score(self._pipeline, texts, labels, cv=3, scoring='accuracy')
        return float(np.mean(cv))

    # ── Internal ────────────────────────────────────────────────────────────

    def _train(self):
        if len(self._corpus) < self.minTrainSamples:
            return
        texts  = [h for h, _ in self._corpus]
        labels = [l for _, l in self._corpus]
        encodedLabels = self._labelEncoder.fit_transform(labels)
        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=8000,
                sublinear_tf=True,
                min_df=1,
            )),
            ('clf', GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                random_state=42,
            )),
        ])
        self._pipeline.fit(texts, encodedLabels)
        self._trained = True


# =============================================================================
# Article importance scoring (gates OpenAI usage)
# =============================================================================

class ArticleImportanceScorer:
    """
    Scores article importance on [0, 1] to decide whether to send to OpenAI.

    Importance is based on:
      - Presence of Tier-1 / Tier-2 / Tier-3 financial keywords
      - Source credibility (trusted outlets score higher)
      - Recency (articles > 7 days old score lower)
      - Whether Layer 1 and Layer 2 significantly disagree (tiebreaker boost)
    """

    def score(
        self,
        headline: str,
        source: str = '',
        publishedAt: Optional[datetime.datetime] = None,
    ) -> float:
        text = headline.lower()

        kwScore = 0.0
        for kw in IMPORTANCE_TIER1:
            if kw in text:
                kwScore += 0.30
        for kw in IMPORTANCE_TIER2:
            if kw in text:
                kwScore += 0.20
        for kw in IMPORTANCE_TIER3:
            if kw in text:
                kwScore += 0.10
        kwScore = min(kwScore, 0.70)

        srcScore = 0.5
        srcLower = source.lower()
        for srcKey, weight in _SOURCE_WEIGHTS.items():
            if srcKey in srcLower:
                srcScore = weight
                break
        srcScore *= 0.20   # max 0.20 contribution

        recencyScore = 0.10
        if publishedAt is not None:
            try:
                now = _now()
                pa = publishedAt if publishedAt.tzinfo else publishedAt.replace(tzinfo=datetime.timezone.utc)
                ageDays = max(0.0, (now - pa).total_seconds() / 86400.0)
                recencyScore = 0.10 * max(0.0, 1.0 - ageDays / 7.0)
            except Exception:
                pass

        return min(1.0, kwScore + srcScore + recencyScore)

    def selectTopK(
        self,
        articles: List[ArticleSentiment],
        k: int = 5,
        disagreementThreshold: float = 0.35,
    ) -> List[ArticleSentiment]:
        """
        Select up to *k* articles for OpenAI analysis.

        Priority:
          1. Highest importance score
          2. Articles where Layer 1 and Layer 2 strongly disagree (boosted)
        """
        graded = []
        for art in articles:
            priority = art.importanceScore
            if (art.lexiconScore is not None and art.structuralScore is not None):
                if abs(art.lexiconScore - art.structuralScore) >= disagreementThreshold:
                    priority = min(1.0, priority + 0.25)
            graded.append((priority, art))

        graded.sort(key=lambda x: x[0], reverse=True)
        return [art for _, art in graded[:k]]


# =============================================================================
# Layer 3 — OpenAI Deep Analyst
# =============================================================================

_OPENAI_SYSTEM_PROMPT = """\
You are a quantitative financial analyst specialising in equity market sentiment.
Analyse financial news headlines and produce a structured JSON sentiment assessment.

Respond with ONLY valid JSON matching this exact schema:
{
  "sentiment": "<BULLISH|BEARISH|NEUTRAL>",
  "score": <float in [-1.0, +1.0]>,
  "confidence": <float in [0.0, 1.0]>,
  "key_drivers": [<list of short strings describing the main sentiment drivers>],
  "risk_flags": [<list of risk factors mentioned, empty list if none>],
  "forward_looking": <true|false>,
  "reasoning": "<one concise sentence explanation>"
}

Rules:
- score > 0.3 → BULLISH, score < -0.3 → BEARISH, else NEUTRAL
- confidence reflects how unambiguous the sentiment is (> 0.8 = very clear)
- forward_looking = true when the headline discusses future guidance or forecasts
- Do NOT include any text outside the JSON object
"""

_OPENAI_USER_TEMPLATE = """\
Analyse the sentiment of the following financial headline for {symbol}:

"{headline}"

Source: {source}
Published: {published}
"""

_OPENAI_BATCH_SYSTEM_PROMPT = """\
You are a quantitative financial analyst specialising in equity market sentiment.
Analyse each financial news headline and produce a structured JSON object with a "results" array.

Respond with ONLY valid JSON matching this schema:
{
  "results": [
    {
      "index": <integer, the 0-based index of the headline in the input list>,
      "sentiment": "<BULLISH|BEARISH|NEUTRAL>",
      "score": <float in [-1.0, +1.0]>,
      "confidence": <float in [0.0, 1.0]>,
      "key_drivers": [<list of short strings describing the main sentiment drivers>],
      "reasoning": "<one concise sentence explanation>"
    }
  ]
}

Rules:
- score > 0.3 → BULLISH, score < -0.3 → BEARISH, else NEUTRAL
- confidence reflects how unambiguous the sentiment is (> 0.8 = very clear)
- You MUST return exactly one element per headline, in the same order
- Do NOT include any text outside the JSON object
"""

_OPENAI_BATCH_USER_TEMPLATE = """\
Analyse the sentiment of the following {count} financial headlines for {symbol}.
Return a JSON array with one result per headline.

{headlines_block}
"""


# Cheap models that the system is allowed to use (prevents accidental
# use of expensive models like gpt-4o, o1, o1-pro, etc.).
_ALLOWED_OPENAI_MODELS = frozenset({
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-3.5-turbo',
})


class OpenAIAnalyst:
    """
    Layer 3: OpenAI GPT-4o-mini deep analysis for selected articles.

    Activated only for high-importance articles or when Layer 1 / Layer 2
    strongly disagree.  Rate-limited via ``maxRequestsPerSession``.

    **Cost guard**: only models in ``_ALLOWED_OPENAI_MODELS`` are accepted.
    Passing any other model raises ``ValueError``.  This prevents
    accidental use of expensive models (gpt-4o, o1, o1-pro, etc.).
    """

    def __init__(
        self,
        apiKey: Optional[str] = None,
        model: str = 'gpt-4o-mini',
        maxRequestsPerSession: int = 40,
        requestDelaySeconds: Optional[float] = None,  # None = use OPENAI_REQUEST_DELAY_SECONDS env (default 22 for 3 RPM)
    ):
        if model not in _ALLOWED_OPENAI_MODELS:
            raise ValueError(
                f"[OpenAIAnalyst] Model '{model}' is not in the allowed "
                f"cheap-model list: {sorted(_ALLOWED_OPENAI_MODELS)}.  "
                f"Edit _ALLOWED_OPENAI_MODELS in SentimentAnalysis.py "
                f"if you intentionally want to use an expensive model."
            )
        self.model = model
        self.maxRequestsPerSession = maxRequestsPerSession
        if requestDelaySeconds is not None:
            self.requestDelaySeconds = requestDelaySeconds
        else:
            try:
                self.requestDelaySeconds = float(os.environ.get('OPENAI_REQUEST_DELAY_SECONDS', '22'))
            except (TypeError, ValueError):
                self.requestDelaySeconds = 22.0  # 3 RPM free tier
        self._requestCount: int = 0
        self._client = None

        key = apiKey or os.environ.get('OPENAI_API_KEY', '')
        if key:
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=key)
            except ImportError:
                warnings.warn(
                    "[SentimentAnalysis] openai package not installed — "
                    "Layer 3 disabled.  Run: pip install openai"
                )
        else:
            warnings.warn(
                "[SentimentAnalysis] No OpenAI API key provided — "
                "Layer 3 disabled.  Pass apiKey= or set OPENAI_API_KEY."
            )

    @property
    def isAvailable(self) -> bool:
        return (self._client is not None and
                self._requestCount < self.maxRequestsPerSession)

    def analyse(self, article: ArticleSentiment) -> Tuple[float, float, str, List[str]]:
        """
        Analyse one article.
        Returns (score, confidence, reasoning, key_drivers).
        """
        if not self.isAvailable:
            return 0.0, 0.0, '', []

        userMsg = _OPENAI_USER_TEMPLATE.format(
            symbol=article.symbol or 'the company',
            headline=article.headline,
            source=article.source or 'Unknown',
            published=(str(article.publishedAt.date())
                       if article.publishedAt else 'Unknown'),
        )

        try:
            from OpenAIRetry import with_retry
            def _call():
                return self._client.chat.completions.create(
                    model=self.model,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': _OPENAI_SYSTEM_PROMPT},
                        {'role': 'user',   'content': userMsg},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
            response = with_retry(_call)
            self._requestCount += 1
            if self.requestDelaySeconds > 0:
                time.sleep(self.requestDelaySeconds)

            parsed = json.loads(response.choices[0].message.content)
            score      = max(-1.0, min(1.0, float(parsed.get('score', 0.0))))
            confidence = max(0.0,  min(1.0, float(parsed.get('confidence', 0.5))))
            reasoning  = str(parsed.get('reasoning', ''))
            drivers    = [str(d) for d in parsed.get('key_drivers', [])]
            return score, confidence, reasoning, drivers

        except json.JSONDecodeError as e:
            warnings.warn(f"[OpenAIAnalyst] JSON parse error: {e}")
            return 0.0, 0.0, '', []
        except Exception as e:
            warnings.warn(f"[OpenAIAnalyst] API error: {e}")
            return 0.0, 0.0, '', []

    def batchAnalyse(
        self,
        articles: List['ArticleSentiment'],
        batchSize: int = 8,
    ) -> List[Tuple[float, float, str, List[str]]]:
        """
        Analyse multiple articles in a single API call (batched).
        Falls back to per-article calls if parsing fails.
        Returns list of (score, confidence, reasoning, key_drivers).
        """
        if not self.isAvailable or not articles:
            return [(0.0, 0.0, '', [])] * len(articles)

        results: List[Tuple[float, float, str, List[str]]] = [(0.0, 0.0, '', [])] * len(articles)

        for batchStart in range(0, len(articles), batchSize):
            if not self.isAvailable:
                break
            batch = articles[batchStart:batchStart + batchSize]
            lines = []
            for i, art in enumerate(batch):
                pub = (str(art.publishedAt.date()) if art.publishedAt else 'Unknown')
                src = art.source or 'Unknown'
                lines.append(f"[{i}] \"{art.headline}\" (Source: {src}, Published: {pub})")

            userMsg = _OPENAI_BATCH_USER_TEMPLATE.format(
                count=len(batch),
                symbol=batch[0].symbol or 'the company',
                headlines_block='\n'.join(lines),
            )

            try:
                from OpenAIRetry import with_retry
                def _batch_call():
                    return self._client.chat.completions.create(
                        model=self.model,
                        response_format={'type': 'json_object'},
                        messages=[
                            {'role': 'system', 'content': _OPENAI_BATCH_SYSTEM_PROMPT},
                            {'role': 'user',   'content': userMsg},
                        ],
                        temperature=0.1,
                        max_tokens=200 * len(batch),
                    )
                response = with_retry(_batch_call)
                self._requestCount += 1
                if self.requestDelaySeconds > 0:
                    time.sleep(self.requestDelaySeconds)

                raw = json.loads(response.choices[0].message.content)
                parsed = raw if isinstance(raw, list) else raw.get('results', raw.get('headlines', [raw]))
                if not isinstance(parsed, list):
                    parsed = [parsed]

                for item in parsed:
                    idx = int(item.get('index', 0))
                    if 0 <= idx < len(batch):
                        score = max(-1.0, min(1.0, float(item.get('score', 0.0))))
                        conf  = max(0.0,  min(1.0, float(item.get('confidence', 0.5))))
                        reason = str(item.get('reasoning', ''))
                        drivers = [str(d) for d in item.get('key_drivers', [])]
                        results[batchStart + idx] = (score, conf, reason, drivers)

            except Exception as e:
                warnings.warn(f"[OpenAIAnalyst] Batch error: {e} — falling back to per-article")
                for i, art in enumerate(batch):
                    if not self.isAvailable:
                        break
                    score, conf, reason, drivers = self.analyse(art)
                    results[batchStart + i] = (score, conf, reason, drivers)

        return results

    def resetSession(self):
        self._requestCount = 0


# =============================================================================
# Ensemble combination
# =============================================================================

# Default layer weights.  OpenAI gets the highest weight when it fires
# because it is the most contextually aware.
_DEFAULT_LAYER_WEIGHTS = {
    'lexicon':    0.25,
    'structural': 0.35,
    'openai':     0.40,
}


def _ensembleLayers(
    lexiconScore: Optional[float],    lexiconConf: float,
    structuralScore: Optional[float], structuralConf: float,
    openAIScore: Optional[float],     openAIConf: float,
) -> Tuple[float, float]:
    """
    Combine available layer scores into a single (score, confidence).
    Adaptive: layers that did not run (None) are excluded from the denominator.
    """
    numerator   = 0.0
    totalConf   = 0.0
    totalWeight = 0.0

    def _add(score, conf, baseWeight):
        nonlocal numerator, totalConf, totalWeight
        ew = baseWeight * conf
        numerator   += score * ew
        totalConf   += conf  * baseWeight
        totalWeight += baseWeight

    if lexiconScore    is not None: _add(lexiconScore,    lexiconConf,    _DEFAULT_LAYER_WEIGHTS['lexicon'])
    if structuralScore is not None: _add(structuralScore, structuralConf, _DEFAULT_LAYER_WEIGHTS['structural'])
    if openAIScore     is not None: _add(openAIScore,     openAIConf,     _DEFAULT_LAYER_WEIGHTS['openai'])

    if totalWeight == 0:
        return 0.0, 0.0

    return (max(-1.0, min(1.0, numerator / totalWeight)),
            min(1.0, totalConf / totalWeight))


# =============================================================================
# News fetching helpers
# =============================================================================

def _fetchYFinanceNews(symbol: str, lookbackDays: int = 30) -> List[Dict]:
    """Fetch recent news from yfinance for *symbol*.

    Handles both the legacy yfinance news format (list of flat dicts with
    ``providerPublishTime``) and newer formats that may nest data under
    a ``content`` key or use ``publish_time`` instead.
    """
    try:
        ticker = yf.Ticker(symbol)
        rawNews = ticker.news
        if not rawNews:
            return []
        cutoffTs = (_now() -
                    datetime.timedelta(days=lookbackDays)).timestamp()
        filtered = []
        for item in rawNews:
            # Try multiple timestamp field names for compatibility
            ts = (item.get('providerPublishTime')
                  or item.get('publish_time')
                  or item.get('published_at', 0))
            # Some formats nest content; if ``content`` key exists, use it
            if 'content' in item and isinstance(item['content'], dict):
                merged = {**item['content'], **item}  # content fields override
                ts = ts or merged.get('providerPublishTime', 0)
                item = merged
            if isinstance(ts, str):
                try:
                    ts = datetime.datetime.fromisoformat(
                        ts.replace('Z', '+00:00')
                    ).timestamp()
                except Exception:
                    ts = 0
            if (ts or 0) >= cutoffTs:
                # Stash the resolved timestamp back for the parser
                item['_resolvedTimestamp'] = ts
                filtered.append(item)
        return filtered
    except Exception as e:
        warnings.warn(f"[SentimentAnalysis] yfinance news fetch failed for {symbol}: {e}")
        return []


def _parseYFinanceItem(item: Dict, symbol: str) -> Optional[ArticleSentiment]:
    """Convert a yfinance news dict into an ArticleSentiment.

    Compatible with both legacy and current yfinance news structures.
    """
    headline = (item.get('title') or item.get('headline', '')).strip()
    if not headline:
        return None
    publishedAt = None
    ts = item.get('_resolvedTimestamp') or item.get('providerPublishTime')
    if ts:
        try:
            if isinstance(ts, (int, float)):
                publishedAt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            elif isinstance(ts, str):
                publishedAt = datetime.datetime.fromisoformat(
                    ts.replace('Z', '+00:00')
                )
                if publishedAt.tzinfo is None:
                    publishedAt = publishedAt.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            pass
    return ArticleSentiment(
        headline=headline,
        url=item.get('link') or item.get('url', ''),
        source=item.get('publisher') or item.get('source', ''),
        publishedAt=publishedAt,
        symbol=symbol,
    )


# =============================================================================
# Alternative News API Fetchers
# =============================================================================

class _NewsAPIFetcher:
    """Fetch historical news from newsapi.org (requires ``NEWSAPI_KEY`` env var).

    Free tier: 100 req/day, articles up to 1 month old.
    Paid tier: articles up to 5 years old.
    """

    BASE_URL = 'https://newsapi.org/v2/everything'

    def __init__(self, apiKey: Optional[str] = None):
        self.apiKey = apiKey or os.environ.get('NEWSAPI_KEY', '')

    @property
    def isAvailable(self) -> bool:
        return bool(self.apiKey) and _requests is not None

    # Free-tier hard cap: articles within the last 30 days only.
    FREE_TIER_DAYS = 29

    def fetch(self, symbol: str, startDate: str, endDate: str,
              pageSize: int = 100) -> List[Dict]:
        if not self.isAvailable:
            return []
        try:
            endDt   = datetime.datetime.strptime(endDate,   '%Y-%m-%d')
            startDt = datetime.datetime.strptime(startDate, '%Y-%m-%d')

            # Clamp start date to free-tier limit (29 days before endDate).
            # If the caller asks for more history, we warn once and shrink the
            # window rather than silently returning 0 articles.
            freeCutoff = endDt - datetime.timedelta(days=self.FREE_TIER_DAYS)
            if startDt < freeCutoff:
                warnings.warn(
                    f"[NewsAPI] Free tier limits history to {self.FREE_TIER_DAYS} days. "
                    f"Clamping startDate from {startDate} to {freeCutoff.strftime('%Y-%m-%d')}. "
                    "Upgrade your NewsAPI plan to access older articles."
                )
                startDt = freeCutoff

            result: List[Dict] = []
            page = 1
            while True:
                params = {
                    'q': f'"{symbol}" OR "{symbol} stock"',
                    'from': startDt.strftime('%Y-%m-%d'),
                    'to': endDate,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': min(pageSize, 100),
                    'page': page,
                    'apiKey': self.apiKey,
                }
                resp = _requests.get(self.BASE_URL, params=params, timeout=15)
                if resp.status_code != 200:
                    warnings.warn(
                        f"[NewsAPI] HTTP {resp.status_code} for {symbol} "
                        f"(page {page}): {resp.text[:200]}"
                    )
                    break
                data = resp.json()

                # The API signals errors in the JSON body even with HTTP 200.
                if data.get('status') != 'ok':
                    err_code = data.get('code', 'unknown')
                    err_msg  = data.get('message', '')
                    warnings.warn(
                        f"[NewsAPI] API error for {symbol}: [{err_code}] {err_msg}"
                    )
                    break

                articles = data.get('articles', [])
                for art in articles:
                    result.append({
                        'title': art.get('title', ''),
                        'link': art.get('url', ''),
                        'publisher': (art.get('source') or {}).get('name', ''),
                        'publishedAt_str': art.get('publishedAt', ''),
                    })

                totalResults = data.get('totalResults', 0)
                fetched = page * min(pageSize, 100)
                # Stop paginating when we have all results or hit 500-article cap
                # (free tier caps at 100 total; paid allows more).
                if fetched >= totalResults or fetched >= 500 or len(articles) == 0:
                    break
                page += 1

            return result
        except Exception as e:
            warnings.warn(f"[NewsAPI] fetch error for {symbol}: {e}")
            return []


class _FinnhubFetcher:
    """Fetch company news from finnhub.io (requires ``FINNHUB_KEY`` env var).

    Free tier: 60 calls/min.  Historical data goes back several years.
    """

    BASE_URL = 'https://finnhub.io/api/v1/company-news'

    def __init__(self, apiKey: Optional[str] = None):
        self.apiKey = apiKey or os.environ.get('FINNHUB_KEY', '')

    @property
    def isAvailable(self) -> bool:
        return bool(self.apiKey) and _requests is not None

    def fetch(self, symbol: str, startDate: str, endDate: str) -> List[Dict]:
        if not self.isAvailable:
            return []
        try:
            params = {
                'symbol': symbol,
                'from': startDate,
                'to': endDate,
                'token': self.apiKey,
            }
            resp = _requests.get(self.BASE_URL, params=params, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
            if not isinstance(data, list):
                return []
            result = []
            for art in data:
                result.append({
                    'title': art.get('headline', ''),
                    'link': art.get('url', ''),
                    'publisher': art.get('source', ''),
                    'providerPublishTime': art.get('datetime', 0),
                })
            return result
        except Exception as e:
            warnings.warn(f"[Finnhub] fetch error for {symbol}: {e}")
            return []


class _AlphaVantageFetcher:
    """Fetch news sentiment from Alpha Vantage (requires ``ALPHAVANTAGE_KEY``).

    Free tier: 25 req/day.  Returns pre-scored sentiment alongside headlines.
    """

    BASE_URL = 'https://www.alphavantage.co/query'

    def __init__(self, apiKey: Optional[str] = None):
        self.apiKey = apiKey or os.environ.get('ALPHAVANTAGE_KEY', '')

    @property
    def isAvailable(self) -> bool:
        return bool(self.apiKey) and _requests is not None

    def fetch(self, symbol: str, startDate: str, endDate: str) -> List[Dict]:
        if not self.isAvailable:
            return []
        try:
            # Alpha Vantage wants YYYYMMDDTHHMM format
            fromDt = startDate.replace('-', '') + 'T0000'
            toDt   = endDate.replace('-', '')   + 'T2359'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': fromDt,
                'time_to': toDt,
                'limit': 200,
                'apikey': self.apiKey,
            }
            resp = _requests.get(self.BASE_URL, params=params, timeout=20)
            if resp.status_code != 200:
                return []
            data = resp.json()
            feed = data.get('feed', [])
            result = []
            for art in feed:
                # Parse time_published: "20240115T120000"
                ts_str = art.get('time_published', '')
                ts = 0
                if ts_str:
                    try:
                        dt = datetime.datetime.strptime(ts_str[:15], '%Y%m%dT%H%M%S')
                        ts = dt.timestamp()
                    except Exception:
                        pass
                result.append({
                    'title': art.get('title', ''),
                    'link': art.get('url', ''),
                    'publisher': art.get('source', ''),
                    'providerPublishTime': ts,
                    '_av_sentiment': float(art.get('overall_sentiment_score', 0)),
                })
            return result
        except Exception as e:
            warnings.warn(f"[AlphaVantage] fetch error for {symbol}: {e}")
            return []


# =============================================================================
# Historical News Feed — Multi-source Orchestrator
# =============================================================================

class HistoricalNewsFeed:
    """
    Fetches news articles for *any* date range by querying multiple APIs
    in priority order and deduplicating results.

    Priority:
      1. Finnhub  (best historical depth, free tier goes back years)
      2. NewsAPI  (good relevance ranking, limited history on free tier)
      3. Alpha Vantage  (pre-scored sentiment, limited free tier)
      4. yfinance (only recent ~14 days, always available)

    Articles from all sources are unified into ``ArticleSentiment`` objects.
    Set API keys via env vars: ``FINNHUB_KEY``, ``NEWSAPI_KEY``,
    ``ALPHAVANTAGE_KEY``, or pass them to the constructor.
    """

    def __init__(
        self,
        newsAPIKey: Optional[str] = None,
        finnhubKey: Optional[str] = None,
        alphaVantageKey: Optional[str] = None,
    ):
        self._finnhub = _FinnhubFetcher(apiKey=finnhubKey)
        self._newsAPI = _NewsAPIFetcher(apiKey=newsAPIKey)
        self._alphaVantage = _AlphaVantageFetcher(apiKey=alphaVantageKey)

    @property
    def availableSources(self) -> List[str]:
        sources = ['yfinance']
        if self._finnhub.isAvailable:
            sources.append('finnhub')
        if self._newsAPI.isAvailable:
            sources.append('newsapi')
        if self._alphaVantage.isAvailable:
            sources.append('alphavantage')
        return sources

    def fetch(
        self,
        symbol: str,
        startDate: str,
        endDate: str,
        verbose: bool = False,
    ) -> List[ArticleSentiment]:
        """
        Fetch articles from all available sources, deduplicate, and return.

        Parameters
        ----------
        symbol : str
        startDate, endDate : str  (YYYY-MM-DD)
        verbose : bool

        Returns
        -------
        List[ArticleSentiment]  (sorted by publishedAt descending)
        """
        allRaw: List[Dict] = []

        # Fetch from all news sources concurrently
        def _fetchFinnhub():
            if self._finnhub.isAvailable:
                return ('Finnhub', self._finnhub.fetch(symbol, startDate, endDate))
            return ('Finnhub', [])

        def _fetchNewsAPI():
            if self._newsAPI.isAvailable:
                return ('NewsAPI', self._newsAPI.fetch(symbol, startDate, endDate))
            return ('NewsAPI', [])

        def _fetchAlphaVantage():
            if self._alphaVantage.isAvailable:
                return ('AlphaVantage', self._alphaVantage.fetch(symbol, startDate, endDate))
            return ('AlphaVantage', [])

        def _fetchYF():
            try:
                daysAgo = (_now() -
                           datetime.datetime.strptime(startDate, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)).days
                if daysAgo <= 30:
                    return ('yfinance', _fetchYFinanceNews(symbol, lookbackDays=daysAgo))
            except Exception:
                pass
            return ('yfinance', [])

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fn) for fn in [_fetchFinnhub, _fetchNewsAPI, _fetchAlphaVantage, _fetchYF]]
            for fut in as_completed(futures):
                try:
                    name, raw = fut.result()
                    if verbose and raw:
                        print(f"      {name}: {len(raw)} articles")
                    allRaw.extend(raw)
                except Exception:
                    pass

        # Parse into ArticleSentiment and deduplicate by headline hash
        articles: List[ArticleSentiment] = []
        seen: set = set()
        for item in allRaw:
            art = self._parseItem(item, symbol)
            if art is None:
                continue
            hh = hashlib.md5(art.headline.lower().encode()).hexdigest()
            if hh in seen:
                continue
            seen.add(hh)
            # Filter to date range
            if art.publishedAt is not None:
                d = art.publishedAt.date()
                try:
                    sd = datetime.datetime.strptime(startDate, '%Y-%m-%d').date()
                    ed = datetime.datetime.strptime(endDate, '%Y-%m-%d').date()
                    if d < sd or d > ed:
                        continue
                except Exception:
                    pass
            articles.append(art)

        # Sort by date (newest first)
        _EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        articles.sort(
            key=lambda a: a.publishedAt if a.publishedAt and a.publishedAt.tzinfo else _EPOCH,
            reverse=True,
        )
        if verbose:
            print(f"      Total unique articles: {len(articles)}")
        return articles

    @staticmethod
    def _parseItem(item: Dict, symbol: str) -> Optional[ArticleSentiment]:
        """Parse a raw dict from any source into ArticleSentiment."""
        headline = (
            item.get('title') or item.get('headline', '')
        ).strip()
        if not headline:
            return None

        publishedAt = None
        # Try multiple timestamp formats
        ts = (item.get('_resolvedTimestamp')
              or item.get('providerPublishTime')
              or item.get('publish_time')
              or 0)
        ts_str = item.get('publishedAt_str', '')
        if ts_str:
            try:
                publishedAt = datetime.datetime.fromisoformat(
                    ts_str.replace('Z', '+00:00')
                )
                if publishedAt.tzinfo is None:
                    publishedAt = publishedAt.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                pass
        if publishedAt is None and ts:
            try:
                if isinstance(ts, (int, float)) and ts > 0:
                    publishedAt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                elif isinstance(ts, str):
                    publishedAt = datetime.datetime.fromisoformat(
                        ts.replace('Z', '+00:00')
                    )
                    if publishedAt.tzinfo is None:
                        publishedAt = publishedAt.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                pass

        return ArticleSentiment(
            headline=headline,
            url=item.get('link') or item.get('url', ''),
            source=item.get('publisher') or item.get('source', ''),
            publishedAt=publishedAt,
            symbol=symbol,
        )


# =============================================================================
# Synthetic Sentiment Generator (backtesting fallback)
# =============================================================================

class SyntheticSentimentGenerator:
    """
    Generates plausible daily sentiment scores from OHLCV price data.

    Used as a **backtesting fallback** when real historical news is not
    available for a given date range.  The generator produces a sentiment
    proxy that is:

      - Positively correlated with (but NOT identical to) returns
      - Smoothed with exponential decay (sentiment persists for days)
      - Amplified during volume spikes (proxy for "newsworthy" events)
      - Biased slightly negative during high-volatility regimes
      - Noisy enough that the ML model doesn't overfit to it

    The proxy is not "fake news" — it is a statistical transformation of
    real market data that captures the *footprint* of sentiment-driven
    moves.  Models trained on this proxy learn how return-correlated
    sentiment signals interact with technical features, so they benefit
    from real sentiment data when available.
    """

    @staticmethod
    def generate(
        df: pd.DataFrame,
        noiseLevel: float = 0.15,
        smoothingSpan: int = 3,
        seed: int = 42,
    ) -> pd.Series:
        """
        Generate a daily sentiment proxy from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        noiseLevel : float
            Fraction of the signal that is random noise (0=none, 1=pure noise).
        smoothingSpan : int
            EWM span for sentiment persistence (days).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.Series
            Date-indexed values in [-1, +1].
        """
        if df is None or len(df) < 10:
            return pd.Series(dtype=float)

        rng = np.random.RandomState(seed)
        closes = df['close'].values.astype(float)
        volumes = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(len(df))

        # 1. Daily log returns
        logRet = np.diff(np.log(closes + 1e-9), prepend=np.log(closes[0] + 1e-9))

        # 2. Z-score of returns (relative to 20-day rolling std)
        retSeries = pd.Series(logRet, index=df.index)
        rollingStd = retSeries.rolling(20, min_periods=5).std().fillna(
            retSeries.std()
        )
        zScores = (retSeries / (rollingStd + 1e-9)).values

        # 3. Volume spike factor: vol / rolling_mean_vol (>1 = spike)
        volSeries = pd.Series(volumes, index=df.index)
        rollingVol = volSeries.rolling(20, min_periods=5).mean().fillna(volSeries.mean())
        volSpike = np.clip((volumes / (rollingVol.values + 1e-9)) - 1.0, 0, 3)

        # 4. Raw sentiment signal: z-scored return × (1 + volume spike boost)
        rawSignal = zScores * (1.0 + 0.3 * volSpike)

        # 5. Map to [-1, +1] via tanh (saturates extreme values)
        rawSentiment = np.tanh(rawSignal * 0.5)

        # 6. Add high-vol negative bias (uncertain markets → negative sentiment)
        highVol = rollingStd.values > rollingStd.rolling(60, min_periods=20).mean().fillna(
            rollingStd.mean()
        ).values
        rawSentiment[highVol] -= 0.05

        # 7. Add calibrated noise
        noise = rng.normal(0, 0.15, size=len(rawSentiment))
        blended = rawSentiment * (1 - noiseLevel) + noise * noiseLevel

        # 8. Exponential smoothing (sentiment persists across days)
        sentSeries = pd.Series(blended, index=df.index).ewm(
            span=smoothingSpan, min_periods=1
        ).mean()

        # 9. Final clip to [-1, +1]
        sentSeries = sentSeries.clip(-1.0, 1.0)

        # Normalize index to date if needed
        if hasattr(sentSeries.index, 'normalize'):
            sentSeries.index = sentSeries.index.normalize()
        # Deduplicate on date (keep last value per day)
        sentSeries = sentSeries[~sentSeries.index.duplicated(keep='last')]

        return sentSeries


# =============================================================================
# Main orchestrator — SentimentAnalyzer
# =============================================================================

class SentimentAnalyzer:
    """
    Main entry point.  Orchestrates all three layers and produces
    date-indexed sentiment Series per symbol.

    Parameters
    ----------
    openAIKey : str, optional
        OpenAI API key.  Can also be set via ``OPENAI_API_KEY`` env var.
        If not provided, Layer 3 is disabled and Layers 1+2 run alone.
    openAIModel : str
        OpenAI model to use.  Default: 'gpt-4o-mini' (fast + cheap).
    maxOpenAIPerSymbol : int
        Maximum articles sent to OpenAI per symbol per ``fetchAndScore``
        call.  Keeps costs predictable.  Default 5.
    sentimentDecayHalfLifeDays : float
        Half-life for the exponential recency decay applied when
        aggregating article scores to a daily value.  Default 3.0 days.
    cacheDir : str, optional
        Directory to persist scored-article cache.  Prevents re-scoring
        the same headlines across calls.  None = disable.
    """

    def __init__(
        self,
        openAIKey: Optional[str] = None,
        openAIModel: str = 'gpt-4o-mini',
        maxOpenAIPerSymbol: int = 5,
        sentimentDecayHalfLifeDays: float = 3.0,
        cacheDir: Optional[str] = None,
    ):
        self._lexicon    = KeywordLexiconScorer()
        self._structural = StructuralMLScorer()
        self._openAIKey  = openAIKey or os.environ.get('OPENAI_API_KEY', '')
        self._openAIModel = openAIModel
        self._openAI     = OpenAIAnalyst(apiKey=self._openAIKey, model=openAIModel)
        self._importance = ArticleImportanceScorer()

        self.maxOpenAIPerSymbol = maxOpenAIPerSymbol
        self.decayHalfLife      = sentimentDecayHalfLifeDays
        self.cacheDir           = cacheDir
        self._cache: Dict[str, Dict] = {}
        self._structuralLock    = threading.Lock()

        if cacheDir:
            os.makedirs(cacheDir, exist_ok=True)
            self._loadCache()

    # ── Primary public API ──────────────────────────────────────────────────

    def fetchAndScore(
        self,
        symbols: List[str],
        lookbackDays: int = 30,
        verbose: bool = True,
    ) -> Dict[str, pd.Series]:
        """
        Fetch news headlines and produce a decay-weighted daily sentiment
        Series for each symbol.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols.
        lookbackDays : int
            Calendar days of history to look back.
        verbose : bool
            Print progress.

        Returns
        -------
        Dict[symbol → pd.Series]
            DatetimeIndex (daily), float values in [-1, +1].
            Days with no news have score 0.0 (via forward-fill then zero-fill).
        """
        results: Dict[str, pd.Series] = {}

        # Pre-fetch news for all symbols concurrently
        prefetched: Dict[str, List] = {}
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
            futMap = {
                pool.submit(_fetchYFinanceNews, sym, lookbackDays): sym
                for sym in symbols
            }
            for fut in as_completed(futMap):
                sym = futMap[fut]
                try:
                    prefetched[sym] = fut.result()
                except Exception:
                    prefetched[sym] = []

        for symbol in symbols:
            if verbose:
                print(f"  [Sentiment] {symbol} ...")

            rawNews = prefetched.get(symbol, [])
            articles = [a for item in rawNews
                        if (a := _parseYFinanceItem(item, symbol)) is not None]

            if not articles:
                if verbose:
                    print(f"    No news found — using zero series")
                results[symbol] = pd.Series(dtype=float)
                continue

            if verbose:
                print(f"    {len(articles)} articles — running all layers ...")

            scored = self._scoreArticles(articles, symbol, verbose=verbose)
            daily  = self._aggregateToDailySeries(scored, lookbackDays)
            results[symbol] = daily

            if verbose and len(daily):
                print(f"    Latest daily score: {daily.iloc[-1]:+.3f}")

        self._saveCache()
        return results

    def scoreHeadlines(
        self,
        headlines: List[str],
        symbol: str = '',
        sources: Optional[List[str]] = None,
        timestamps: Optional[List[datetime.datetime]] = None,
    ) -> List[ArticleSentiment]:
        """
        Score a list of raw headline strings without fetching from yfinance.
        Useful for backtesting against a pre-collected headline dataset.
        """
        articles = [
            ArticleSentiment(
                headline=h,
                symbol=symbol,
                source=(sources[i] if sources and i < len(sources) else ''),
                publishedAt=(timestamps[i] if timestamps and i < len(timestamps) else None),
            )
            for i, h in enumerate(headlines)
        ]
        return self._scoreArticles(articles, symbol)

    def fetchHistorical(
        self,
        symbols: List[str],
        startDate: str,
        endDate: str,
        priceDataDict: Optional[Dict[str, pd.DataFrame]] = None,
        newsAPIKey: Optional[str] = None,
        finnhubKey: Optional[str] = None,
        alphaVantageKey: Optional[str] = None,
        verbose: bool = True,
        maxOpenAIPerSymbol: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """
        Produce daily sentiment Series for each symbol over an arbitrary
        historical date range.  Designed for **backtesting**.

        Strategy:
          1. Try to fetch real news articles from all configured APIs
             (Finnhub, NewsAPI, Alpha Vantage, yfinance).
          2. Score real articles through Layers 1+2 (Layer 3/OpenAI is
             skipped for historical to save cost — override with
             ``useOpenAI=True`` if desired).
          3. For date ranges where real news is sparse (< 3 articles per
             30-day window), fill gaps with a synthetic sentiment proxy
             derived from OHLCV price data.
          4. Merge real and synthetic into a continuous daily Series.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols.
        startDate, endDate : str
            Date range (YYYY-MM-DD).
        priceDataDict : dict, optional
            {symbol: OHLCV DataFrame} — used for synthetic fallback.
            If not provided, data is fetched via yfinance.
        newsAPIKey, finnhubKey, alphaVantageKey : str, optional
            API keys (also read from env vars).
        verbose : bool
        maxOpenAIPerSymbol : int, optional
            Per-symbol OpenAI article cap. Use 2–3 for connected stocks to
            reduce cost; omit for main pipeline (uses 10).

        Returns
        -------
        Dict[symbol → pd.Series]
            DatetimeIndex (daily), values in [-1, +1].
        """
        feed = HistoricalNewsFeed(
            newsAPIKey=newsAPIKey,
            finnhubKey=finnhubKey,
            alphaVantageKey=alphaVantageKey,
        )
        if verbose:
            print(f"  [Sentiment] Historical mode: {startDate} → {endDate}")
            print(f"    Available news sources: {feed.availableSources}")

        results: Dict[str, pd.Series] = {}

        # OpenAI budget: use explicit override, or bump for main pipeline backtest
        origMaxOAI = self.maxOpenAIPerSymbol
        if maxOpenAIPerSymbol is not None:
            self.maxOpenAIPerSymbol = maxOpenAIPerSymbol
        else:
            self.maxOpenAIPerSymbol = max(origMaxOAI, 10)

        # ── Pre-fetch: get news for ALL symbols concurrently ─────────
        prefetched: Dict[str, List] = {}
        _t0_fetch = time.time()
        if verbose:
            print(f"    Pre-fetching news for {len(symbols)} symbols in parallel ...")
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
            futMap = {
                pool.submit(feed.fetch, sym, startDate, endDate, False): sym
                for sym in symbols
            }
            for fut in as_completed(futMap):
                sym = futMap[fut]
                try:
                    prefetched[sym] = fut.result()
                except Exception as e:
                    if verbose:
                        print(f"      {sym} fetch error: {e}")
                    prefetched[sym] = []
        if verbose:
            _fetch_sec = time.time() - _t0_fetch
            totalArts = sum(len(v) for v in prefetched.values())
            print(f"    Fetched {totalArts} total articles in {_fetch_sec:.1f}s")

        maxWorkers = min(8, len(symbols))

        def _processOneSymbol(sym: str) -> Tuple[str, pd.Series]:
            worker_oai = OpenAIAnalyst(apiKey=self._openAIKey, model=self._openAIModel)
            articles = prefetched.get(sym, [])
            realSeries = pd.Series(dtype=float)
            scored: List[ArticleSentiment] = []
            lookbackDays = (
                datetime.datetime.strptime(endDate, '%Y-%m-%d') -
                datetime.datetime.strptime(startDate, '%Y-%m-%d')
            ).days
            if articles:
                scored = self._scoreArticles(articles, sym, verbose=False, openai_analyst=worker_oai)
                realSeries = self._aggregateToDailySeries(scored, lookbackDays)
            priceDf = (priceDataDict or {}).get(sym)
            if scored and priceDf is not None and len(priceDf) > 5:
                newLabels: List[Tuple[str, int]] = []
                priceDfNorm = priceDf.copy()
                if hasattr(priceDfNorm.index, 'tz') and priceDfNorm.index.tz is not None:
                    priceDfNorm.index = priceDfNorm.index.tz_localize(None)
                if hasattr(priceDfNorm.index, 'normalize'):
                    priceDfNorm.index = priceDfNorm.index.normalize()
                closeCol = priceDfNorm['close'] if 'close' in priceDfNorm.columns else None
                if closeCol is None:
                    for col in priceDfNorm.columns:
                        cName = col[0] if isinstance(col, tuple) else col
                        if str(cName).lower() == 'close':
                            closeCol = priceDfNorm[col]
                            break
                if closeCol is not None and len(closeCol) > 3:
                    if hasattr(closeCol.index, 'tz') and closeCol.index.tz is not None:
                        closeCol = closeCol.copy()
                        closeCol.index = closeCol.index.tz_localize(None)
                    for art in scored:
                        if art.publishedAt is None:
                            continue
                        artDate = art.publishedAt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
                        loc = closeCol.index.get_indexer([artDate], method='ffill')[0]
                        if loc < 0 or loc + 3 >= len(closeCol):
                            continue
                        priceAtPub = float(closeCol.iloc[loc])
                        priceAfter = float(closeCol.iloc[min(loc + 3, len(closeCol) - 1)])
                        if priceAtPub <= 0:
                            continue
                        fwdRetPct = (priceAfter - priceAtPub) / priceAtPub * 100
                        label = +1 if fwdRetPct > 0.5 else (-1 if fwdRetPct < -0.5 else 0)
                        newLabels.append((art.headline, label))
                if newLabels:
                    with self._structuralLock:
                        self._structural.addLabeledExamples(newLabels)
                    scored = self._scoreArticles(scored, sym, verbose=False, openai_analyst=worker_oai)
                    realSeries = self._aggregateToDailySeries(scored, lookbackDays)
            priceDf = (priceDataDict or {}).get(sym)
            if priceDf is None or len(priceDf) == 0:
                try:
                    import yfinance as yf
                    priceDf = yf.download(sym, start=startDate, end=endDate, interval='1d', progress=False, auto_adjust=True)
                except Exception:
                    priceDf = None
            syntheticSeries = pd.Series(dtype=float)
            if priceDf is not None and len(priceDf) >= 10:
                syntheticSeries = SyntheticSentimentGenerator.generate(priceDf, seed=hash(sym) % (2**31))
            if len(realSeries) > 0 and len(syntheticSeries) > 0:
                sdDt = datetime.datetime.strptime(startDate, '%Y-%m-%d')
                edDt = datetime.datetime.strptime(endDate, '%Y-%m-%d')
                fullRange = pd.date_range(start=sdDt, end=edDt, freq='D')
                real = realSeries.reindex(fullRange)
                synth = syntheticSeries.reindex(fullRange)
                combined = real.fillna(synth).ffill().fillna(0.0)
                return sym, combined
            if len(realSeries) > 0:
                return sym, realSeries
            if len(syntheticSeries) > 0:
                return sym, syntheticSeries
            return sym, pd.Series(dtype=float)

        if maxWorkers > 1 and len(symbols) > 1:
            if verbose:
                print(f"    Scoring {len(symbols)} symbols in parallel ({maxWorkers} workers)...")
            with ThreadPoolExecutor(max_workers=maxWorkers) as pool:
                for sym, series in pool.map(_processOneSymbol, symbols):
                    results[sym] = series
                    if verbose:
                        arts = len(prefetched.get(sym, []))
                        s = series
                        print(f"  [Sentiment] {sym}: {len(s)} days | {arts} articles")
        else:
            for symbol in symbols:
                _t0_sym = time.time()
                if verbose:
                    print(f"\n  [Sentiment] {symbol} ...")

                # Reset OpenAI session counter so each symbol gets its full quota
                self._openAI.resetSession()

                # ── 1. Use pre-fetched articles ──────────────────────────────
                articles = prefetched.get(symbol, [])

                # ── 2. Score real articles (all layers incl. OpenAI if key set) ──
                realSeries = pd.Series(dtype=float)
                scored: List[ArticleSentiment] = []
                lookbackDays = (
                    datetime.datetime.strptime(endDate, '%Y-%m-%d') -
                    datetime.datetime.strptime(startDate, '%Y-%m-%d')
                ).days
                if articles:
                    _t0_score = time.time()
                    if verbose:
                        print(f"      Scoring {len(articles)} real articles ...")
                    scored = self._scoreArticles(articles, symbol, verbose=verbose)
                    realSeries = self._aggregateToDailySeries(scored, lookbackDays)
                    if verbose:
                        print(f"      [TIMING] Scoring: {time.time() - _t0_score:.1f}s")

                # ── 2b. Self-train Layer 2 from actual price reactions ──────
                priceDf = (priceDataDict or {}).get(symbol)
                if scored and priceDf is not None and len(priceDf) > 5:
                    newLabels: List[Tuple[str, int]] = []
                    priceDfNorm = priceDf.copy()
                    if hasattr(priceDfNorm.index, 'tz') and priceDfNorm.index.tz is not None:
                        priceDfNorm.index = priceDfNorm.index.tz_localize(None)
                    if hasattr(priceDfNorm.index, 'normalize'):
                        priceDfNorm.index = priceDfNorm.index.normalize()
                    closeCol = priceDfNorm['close'] if 'close' in priceDfNorm.columns else None
                    if closeCol is None:
                        for col in priceDfNorm.columns:
                            cName = col[0] if isinstance(col, tuple) else col
                            if str(cName).lower() == 'close':
                                closeCol = priceDfNorm[col]
                                break
                    if closeCol is not None and len(closeCol) > 3:
                        if hasattr(closeCol.index, 'tz') and closeCol.index.tz is not None:
                            closeCol = closeCol.copy()
                            closeCol.index = closeCol.index.tz_localize(None)
                        for art in scored:
                            if art.publishedAt is None:
                                continue
                            artDate = art.publishedAt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
                            loc = closeCol.index.get_indexer([artDate], method='ffill')[0]
                            if loc < 0 or loc + 3 >= len(closeCol):
                                continue
                            priceAtPub = float(closeCol.iloc[loc])
                            priceAfter = float(closeCol.iloc[min(loc + 3, len(closeCol) - 1)])
                            if priceAtPub <= 0:
                                continue
                            fwdRetPct = (priceAfter - priceAtPub) / priceAtPub * 100
                            label = +1 if fwdRetPct > 0.5 else (-1 if fwdRetPct < -0.5 else 0)
                            newLabels.append((art.headline, label))
                    if newLabels:
                        with self._structuralLock:
                            self._structural.addLabeledExamples(newLabels)
                        if verbose:
                            print(f"      Layer 2 self-trained on {len(newLabels)} headline→outcome pairs")
                        scored = self._scoreArticles(scored, symbol, verbose=False)
                        realSeries = self._aggregateToDailySeries(scored, lookbackDays)

                # ── 3. Synthetic fallback for sparse coverage ───────────────
                priceDf = (priceDataDict or {}).get(symbol)
                if priceDf is None or len(priceDf) == 0:
                    try:
                        import yfinance as yf
                        priceDf = yf.download(symbol, start=startDate, end=endDate, interval='1d', progress=False, auto_adjust=True)
                    except Exception:
                        priceDf = None

                syntheticSeries = pd.Series(dtype=float)
                if priceDf is not None and len(priceDf) >= 10:
                    syntheticSeries = SyntheticSentimentGenerator.generate(priceDf, seed=hash(symbol) % (2**31))

                # ── 4. Merge ─
                if len(realSeries) > 0 and len(syntheticSeries) > 0:
                    sdDt = datetime.datetime.strptime(startDate, '%Y-%m-%d')
                    edDt = datetime.datetime.strptime(endDate, '%Y-%m-%d')
                    fullRange = pd.date_range(start=sdDt, end=edDt, freq='D')
                    real = realSeries.reindex(fullRange)
                    synth = syntheticSeries.reindex(fullRange)
                    combined = real.fillna(synth).ffill().fillna(0.0)
                    results[symbol] = combined
                elif len(realSeries) > 0:
                    results[symbol] = realSeries
                elif len(syntheticSeries) > 0:
                    results[symbol] = syntheticSeries
                else:
                    results[symbol] = pd.Series(dtype=float)

                _sym_sec = time.time() - _t0_sym
                if verbose:
                    print(f"      [TIMING] {symbol} total: {_sym_sec:.1f}s")
                    s = results[symbol]
                    realCount = len(articles)
                    synthPct = 100 * (1 - len(realSeries) / max(len(s), 1)) if len(s) > 0 else 100
                    print(f"      → {len(s)} days | {realCount} real articles | ~{synthPct:.0f}% synthetic fill")

        self.maxOpenAIPerSymbol = origMaxOAI  # restore original cap
        self._saveCache()
        return results

    def recordOutcome(self, headline: str, actualDirection: int):
        """
        Feedback loop: provide the outcome of a trade after the fact so
        the structural ML model can retrain with real-world labels.

        Parameters
        ----------
        headline : str
        actualDirection : int
            +1 (price rose), -1 (price fell), 0 (no significant move)
        """
        self._structural.addLabeledExamples([(headline, actualDirection)])

    # ── Internal scoring pipeline ───────────────────────────────────────────

    def _scoreArticles(
        self,
        articles: List[ArticleSentiment],
        symbol: str,
        verbose: bool = False,
        openai_analyst: Optional['OpenAIAnalyst'] = None,
    ) -> List[ArticleSentiment]:
        """Run all layers on a list of articles, fill ensemble scores.
        openai_analyst: optional separate analyst for parallel workers."""
        if not articles:
            return []

        oai = openai_analyst if openai_analyst is not None else self._openAI
        headlines = [a.headline for a in articles]

        # ── Layer 1 + 2 (batch) ─────────────────────────────────────────────
        _t0_l12 = time.time()
        layer1 = self._lexicon.batchScore(headlines)
        layer2 = self._structural.batchScore(headlines)

        # Parallel arrays of per-layer confidence (kept in sync with articles)
        layer1Conf = [layer1[i][1] for i in range(len(articles))]
        layer2Conf = [layer2[i][1] for i in range(len(articles))]

        for i, art in enumerate(articles):
            cached = self._lookupCache(art.headline)
            if cached:
                art.lexiconScore    = cached.get('l1s')
                art.structuralScore = cached.get('l2s')
                art.openAIScore     = cached.get('oai_score')
                art.openAIReasoning = cached.get('oai_reason', '')
                art.openAIDrivers   = cached.get('oai_drivers', [])
                # Use cached confidences to stay consistent with cached scores
                layer1Conf[i] = cached.get('l1c', layer1Conf[i])
                layer2Conf[i] = cached.get('l2c', layer2Conf[i])
            else:
                art.lexiconScore    = layer1[i][0]
                art.structuralScore = layer2[i][0]

            art.importanceScore = self._importance.score(
                art.headline, art.source, art.publishedAt
            )

        # ── Layer 3 (selective OpenAI — batched) ────────────────────────────
        _l12_sec = time.time() - _t0_l12
        if verbose and len(articles) > 0:
            print(f"        [TIMING] L1+L2: {_l12_sec:.1f}s ({len(articles)} arts)")
        if oai.isAvailable:
            uncached = [a for a in articles if a.openAIScore is None]
            selected = self._importance.selectTopK(uncached, k=self.maxOpenAIPerSymbol)
            if selected:
                _t0_oai = time.time()
                batchResults = oai.batchAnalyse(selected, batchSize=50)
                oaiCount = 0
                for j, art in enumerate(selected):
                    score, conf, reason, drivers = batchResults[j]
                    if score == 0.0 and conf == 0.0 and not reason:
                        continue
                    art.openAIScore     = score
                    art.openAIReasoning = reason
                    art.openAIDrivers   = drivers
                    artIdx = articles.index(art)
                    self._storeCache(art.headline, {
                        'l1s': art.lexiconScore,
                        'l1c': layer1Conf[artIdx],
                        'l2s': art.structuralScore,
                        'l2c': layer2Conf[artIdx],
                        'oai_score': score,
                        'oai_conf':  conf,
                        'oai_reason': reason,
                        'oai_drivers': drivers,
                    })
                    oaiCount += 1
                _oai_sec = time.time() - _t0_oai
                if verbose and oaiCount:
                    print(f"        [TIMING] OpenAI: {_oai_sec:.1f}s ({oaiCount} arts)")

        # ── Ensemble ────────────────────────────────────────────────────────
        for i, art in enumerate(articles):
            l1c = layer1Conf[i] if art.lexiconScore    is not None else 0.0
            l2c = layer2Conf[i] if art.structuralScore is not None else 0.0
            cached = self._lookupCache(art.headline)
            oaiConf = cached.get('oai_conf', 0.8) if (cached and art.openAIScore is not None) else 0.0
            art.ensembleScore, art.ensembleConf = _ensembleLayers(
                art.lexiconScore,    l1c,
                art.structuralScore, l2c,
                art.openAIScore,     oaiConf,
            )

        return articles

    # ── Daily aggregation ───────────────────────────────────────────────────

    def _aggregateToDailySeries(
        self,
        articles: List[ArticleSentiment],
        lookbackDays: int,
    ) -> pd.Series:
        """
        Aggregate per-article ensemble scores to a daily Series using
        exponential recency decay.

        decay_weight = exp(-ln(2) / halfLife × ageDays)
        daily_score  = Σ(decay × conf × score) / Σ(decay × conf)
        """
        if not articles:
            return pd.Series(dtype=float)

        now = _now()
        lam = math.log(2) / max(self.decayHalfLife, 0.1)

        dateMap: Dict[datetime.date, List[Tuple[float, float]]] = {}
        for art in articles:
            if art.publishedAt is None:
                continue
            d = art.publishedAt.date()
            pa = art.publishedAt if art.publishedAt.tzinfo else art.publishedAt.replace(tzinfo=datetime.timezone.utc)
            ageDays = max(0.0, (now - pa).total_seconds() / 86400.0)
            weight  = math.exp(-lam * ageDays) * max(art.ensembleConf, 0.1)
            dateMap.setdefault(d, []).append((art.ensembleScore, weight))

        if not dateMap:
            return pd.Series(dtype=float)

        endDate   = now.date()
        startDate = endDate - datetime.timedelta(days=lookbackDays)
        dateRange = pd.date_range(start=startDate, end=endDate, freq='D')

        scores = []
        for dt in dateRange:
            d    = dt.date()
            pairs = dateMap.get(d)
            if pairs:
                totalW = sum(w for _, w in pairs)
                daily  = sum(s * w for s, w in pairs) / totalW if totalW else 0.0
            else:
                daily = float('nan')
            scores.append(daily)

        series = pd.Series(scores, index=dateRange)
        return series.ffill().fillna(0.0)

    # ── Utilities ────────────────────────────────────────────────────────────

    def getStructuralModelCV(self) -> float:
        """3-fold cross-validation accuracy of the structural ML model."""
        return self._structural.crossValidationScore()

    def resetOpenAISession(self):
        self._openAI.resetSession()

    @property
    def openAIAvailable(self) -> bool:
        return self._openAI.isAvailable

    def layerWeights(self) -> Dict[str, float]:
        """Return the current ensemble layer weights."""
        return dict(_DEFAULT_LAYER_WEIGHTS)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _headlineHash(self, h: str) -> str:
        return hashlib.md5(h.encode('utf-8')).hexdigest()

    def _lookupCache(self, headline: str) -> Optional[Dict]:
        return self._cache.get(self._headlineHash(headline))

    def _storeCache(self, headline: str, data: Dict):
        self._cache[self._headlineHash(headline)] = data

    def _loadCache(self):
        if not self.cacheDir:
            return
        path = os.path.join(self.cacheDir, 'sentiment_cache.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _saveCache(self):
        if not self.cacheDir:
            return
        path = os.path.join(self.cacheDir, 'sentiment_cache.json')
        try:
            with open(path, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass


# =============================================================================
# MC Synthetic Headline Generator — produces fake but realistic headlines
# for Monte Carlo simulations so the sentiment pipeline can be stress-tested
# =============================================================================

class MCSyntheticHeadlineGenerator:
    """
    Generates synthetic news headlines aligned with Monte Carlo simulated
    price movements.  Used in MC Phase 2 to test whether the sentiment
    analysis pipeline responds correctly to positive/negative news.

    For each simulated day:
      1. Decide probabilistically whether a headline exists (~35-55% chance,
         higher during large moves).
      2. If yes, pick a headline template matching the direction and
         magnitude of the simulated price change.
      3. Score the headline through the real sentiment pipeline (all 3 layers).
      4. Return a date-indexed sentiment Series ready for ML consumption.

    Headlines are NOT fetched from APIs — they are template-based to avoid
    cost and latency during MC simulations.
    """

    # ── Positive headline templates (returns > 0) ────────────────────────
    _POSITIVE_TEMPLATES = {
        'mild': [  # 0 < return <= 1%
            "{symbol} shares edge higher on steady demand",
            "{symbol} gains modestly as market sentiment improves",
            "Analysts maintain positive outlook on {symbol}",
            "{symbol} ticks up amid broader market rally",
            "Investors show renewed interest in {symbol} stock",
        ],
        'moderate': [  # 1% < return <= 3%
            "{symbol} rallies on strong quarterly results",
            "{symbol} jumps as new product launch exceeds expectations",
            "Upgrade: {symbol} raised to Outperform by major bank",
            "{symbol} surges on better-than-expected revenue growth",
            "Institutional investors increase position in {symbol}",
            "{symbol} rises sharply on positive earnings surprise",
        ],
        'strong': [  # return > 3%
            "{symbol} soars after blockbuster earnings beat",
            "Breaking: {symbol} announces major acquisition, shares surge",
            "{symbol} skyrockets on transformative partnership deal",
            "{symbol} stock explodes higher on record revenue",
            "Wall Street stunned as {symbol} crushes all estimates",
            "{symbol} posts historic gains after breakthrough announcement",
        ],
    }

    # ── Negative headline templates (returns < 0) ────────────────────────
    _NEGATIVE_TEMPLATES = {
        'mild': [  # -1% <= return < 0
            "{symbol} dips slightly amid profit-taking",
            "{symbol} edges lower on mixed market signals",
            "Traders trim {symbol} positions ahead of earnings",
            "{symbol} pulls back from recent highs",
            "Mild selling pressure on {symbol} as sector rotates",
        ],
        'moderate': [  # -3% <= return < -1%
            "{symbol} drops on disappointing guidance",
            "{symbol} falls as analysts cut price targets",
            "Concerns mount over {symbol} growth slowdown",
            "{symbol} slumps after missing quarterly expectations",
            "Downgrade: {symbol} cut to Underperform by analysts",
            "{symbol} declines on supply chain disruption fears",
        ],
        'strong': [  # return < -3%
            "{symbol} plunges on shocking earnings miss",
            "Breaking: {symbol} shares crater amid regulatory probe",
            "{symbol} tanks after CEO departure announcement",
            "{symbol} in freefall as major customer cancels contract",
            "Panic selling hits {symbol} after profit warning",
            "{symbol} collapses on fraud allegations — investigation pending",
        ],
    }

    # ── Neutral headline templates (return ≈ 0) ─────────────────────────
    _NEUTRAL_TEMPLATES = [
        "{symbol} trades flat amid low volume session",
        "No major catalysts for {symbol} today",
        "{symbol} holds steady as market awaits Fed decision",
        "Mixed signals for {symbol} keep traders cautious",
    ]

    def __init__(
        self,
        headlineProbability: float = 0.40,
        largeMoveBoost: float = 0.25,
        sentimentAnalyzer: Optional['SentimentAnalyzer'] = None,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        headlineProbability : float
            Base probability of generating a headline on any given day.
        largeMoveBoost : float
            Additional probability boost when |return| > 2%.
        sentimentAnalyzer : SentimentAnalyzer, optional
            Pre-built analyzer to score headlines.  If None, headlines
            are scored using a simple rule-based fallback (no API calls).
        seed : int
            Random seed for reproducibility.
        """
        self.baseProb = headlineProbability
        self.largeMoveBoost = largeMoveBoost
        self.analyzer = sentimentAnalyzer
        self.rng = np.random.RandomState(seed)
        self._scoreCache: Dict[str, float] = {}
        self._scoreCacheLock = threading.Lock()

    def generateForPath(
        self,
        syntheticDF: pd.DataFrame,
        symbol: str,
    ) -> pd.Series:
        """
        Generate a daily sentiment Series for a synthetic OHLCV path.

        For each row, probabilistically generates a headline matching the
        day's simulated return, scores it, and returns a Series of
        sentiment scores in [-1, +1].  Days with no headline get 0.0.

        Optimised: generates all headlines first, resolves cached scores,
        then batch-scores any remaining uncached headlines in one call.
        """
        if syntheticDF is None or len(syntheticDF) < 2:
            return pd.Series(dtype=float)

        closes = syntheticDF['close'].values.astype(float)
        returns = np.zeros(len(closes))
        returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-9)

        # Thread-safe local RNG: seeded from shared rng under lock or
        # from a deterministic hash so parallel calls don't collide.
        localSeed = hash((symbol, len(syntheticDF), float(closes[0]), float(closes[-1]))) % (2**31)
        localRng = np.random.RandomState(localSeed)

        # First pass: decide which days get headlines and pick templates
        headlineMap: Dict[int, str] = {}  # index → headline text
        for i in range(1, len(closes)):
            ret = returns[i]
            absRet = abs(ret) * 100

            prob = self.baseProb
            if absRet > 2.0:
                prob += self.largeMoveBoost
            if absRet > 5.0:
                prob += self.largeMoveBoost
            prob = min(prob, 0.85)

            if localRng.random() > prob:
                continue

            headlineMap[i] = self._pickHeadlineWithRng(ret * 100, symbol, localRng)

        scores = np.zeros(len(closes))
        if not headlineMap:
            sentSeries = pd.Series(scores, index=syntheticDF.index)
            return sentSeries.clip(-1.0, 1.0)

        # Second pass: resolve from cache, collect misses
        uncachedIndices: List[int] = []
        uncachedHeadlines: List[str] = []
        with self._scoreCacheLock:
            for idx, headline in headlineMap.items():
                cached = self._scoreCache.get(headline)
                if cached is not None:
                    scores[idx] = cached
                else:
                    uncachedIndices.append(idx)
                    uncachedHeadlines.append(headline)

        # Third pass: batch-score all uncached headlines in one call
        if uncachedHeadlines:
            batchScores = self._scoreBatch(uncachedHeadlines, symbol)
            with self._scoreCacheLock:
                for j, idx in enumerate(uncachedIndices):
                    s = batchScores[j]
                    scores[idx] = s
                    self._scoreCache[uncachedHeadlines[j]] = s

        # Apply mild EWM smoothing (sentiment persists across days)
        sentSeries = pd.Series(scores, index=syntheticDF.index)
        sentSeries = sentSeries.ewm(span=2, min_periods=1).mean()
        sentSeries = sentSeries.clip(-1.0, 1.0)

        return sentSeries

    def _pickHeadline(self, returnPct: float, symbol: str) -> str:
        """Pick a headline template matching the return direction/magnitude."""
        return self._pickHeadlineWithRng(returnPct, symbol, self.rng)

    def _pickHeadlineWithRng(self, returnPct: float, symbol: str,
                             rng: np.random.RandomState) -> str:
        """Pick a headline template using the given RNG (thread-safe)."""
        absRet = abs(returnPct)

        if absRet < 0.3:
            template = rng.choice(self._NEUTRAL_TEMPLATES)
        elif returnPct > 0:
            if absRet <= 1.0:
                templates = self._POSITIVE_TEMPLATES['mild']
            elif absRet <= 3.0:
                templates = self._POSITIVE_TEMPLATES['moderate']
            else:
                templates = self._POSITIVE_TEMPLATES['strong']
            template = rng.choice(templates)
        else:
            if absRet <= 1.0:
                templates = self._NEGATIVE_TEMPLATES['mild']
            elif absRet <= 3.0:
                templates = self._NEGATIVE_TEMPLATES['moderate']
            else:
                templates = self._NEGATIVE_TEMPLATES['strong']
            template = rng.choice(templates)

        return template.format(symbol=symbol)

    def _scoreBatch(self, headlines: List[str], symbol: str) -> List[float]:
        """Score multiple headlines at once, returning a list of floats."""
        if self.analyzer is not None:
            try:
                scored = self.analyzer.scoreHeadlines(headlines, symbol=symbol)
                if scored and len(scored) == len(headlines):
                    return [float(a.ensembleScore) for a in scored]
            except Exception:
                pass

        # Fallback: lexicon batch
        lexicon = KeywordLexiconScorer()
        results = []
        for h in headlines:
            try:
                s, _ = lexicon.score(h)
                results.append(float(np.clip(s, -1.0, 1.0)))
            except Exception:
                results.append(self._fallbackScore(h))
        return results

    def _scoreHeadline(self, headline: str, symbol: str) -> float:
        """Score a single headline (used outside generateForPath)."""
        with self._scoreCacheLock:
            cached = self._scoreCache.get(headline)
        if cached is not None:
            return cached

        result = self._scoreBatch([headline], symbol)
        score = result[0] if result else 0.0
        with self._scoreCacheLock:
            self._scoreCache[headline] = score
        return score

    @staticmethod
    def _fallbackScore(headline: str) -> float:
        """Last-resort direction from headline keywords."""
        headline_lower = headline.lower()
        positive_words = ['surge', 'rally', 'jump', 'soar', 'gain',
                          'beat', 'upgrade', 'strong', 'record', 'rise',
                          'positive', 'outperform', 'breakout', 'crush']
        negative_words = ['drop', 'fall', 'plunge', 'crash', 'tank',
                          'miss', 'downgrade', 'concern', 'slump', 'cut',
                          'decline', 'warning', 'probe', 'collapse']
        pos_count = sum(1 for w in positive_words if w in headline_lower)
        neg_count = sum(1 for w in negative_words if w in headline_lower)
        if pos_count > neg_count:
            return 0.5
        elif neg_count > pos_count:
            return -0.5
        return 0.0
