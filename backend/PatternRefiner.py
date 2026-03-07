"""
PatternRefiner — ML-Guided + Random Mutation Pattern Improvement

Uses Monte Carlo trade outcomes to learn *why* patterns succeed or fail,
then generates intelligent mutations that improve pattern performance.

Two mutation strategies run in parallel each iteration:
  1. Random mutation  — perturb gene values stochastically (exploration)
  2. ML mutation      — train a Gradient Boosting model on MC trade data,
                        analyse feature importances and partial dependence
                        to find gene values that maximise success probability
                        (exploitation)

The best-performing variants (original, random-mutant, or ML-mutant) survive
each iteration.
"""

import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from joblib import Parallel, delayed

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GeneticAlgorithm import (
    PatternChromosome, PatternGene, PatternBank, CandleType, StockDataFetcher
)
from MCMCSimulator import MCMCSimulator, SimulationPath


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RefinementResult:
    """Output of one refinement iteration for a single pattern."""
    originalScore: float
    bestScore: float
    improved: bool
    bestPattern: PatternChromosome
    method: str                     # 'original' | 'random' | 'ml'


# =============================================================================
# PatternRefiner
# =============================================================================

class PatternRefiner:
    """
    Iteratively refines patterns using MC feedback + ML intelligence.

    Usage::

        refiner = PatternRefiner(initialFund=10000, forgiveness=0.05)
        refined = refiner.refinePatterns(
            patterns=topPatterns,
            symbol='AAPL',
            interval='1d',
            calibrationPeriod='2y',
            iterations=3,
        )
    """

    def __init__(self,
                 initialFund: float = 10000.0,
                 forgiveness: float = 0.05,
                 mcMethod: str = 'bootstrap',
                 simsPerCandidate: int = 150,
                 simPeriods: int = 252,
                 randomMutantsPerPattern: int = 6,
                 mlMutantsPerPattern: int = 4,
                 mutationStrength: float = 0.3):
        """
        Args:
            initialFund:             Capital for MC simulations
            forgiveness:             Pattern-matching tolerance
            mcMethod:                'bootstrap' or 'gbm'
            simsPerCandidate:        MC paths per candidate pattern
            simPeriods:              Periods per MC path
            randomMutantsPerPattern: Random mutations generated per pattern
            mlMutantsPerPattern:     ML-guided mutations generated per pattern
            mutationStrength:        Magnitude of random perturbation (0-1)
        """
        self.initialFund = initialFund
        self.forgiveness = forgiveness
        self.mcMethod = mcMethod
        self.simsPerCandidate = simsPerCandidate
        self.simPeriods = simPeriods
        self.randomMutantsPerPattern = randomMutantsPerPattern
        self.mlMutantsPerPattern = mlMutantsPerPattern
        self.mutationStrength = mutationStrength

    # =====================================================================
    # Public API
    # =====================================================================

    def refinePatterns(self,
                      patterns: List[PatternChromosome],
                      symbol: str,
                      interval: str,
                      calibrationPeriod: str = '2y',
                      calibrationStart: Optional[str] = None,
                      calibrationEnd: Optional[str] = None,
                      iterations: int = 3,
                      verbose: bool = True
                      ) -> List[Tuple[PatternChromosome, float]]:
        """
        Run multiple refinement iterations on a set of patterns.

        Each iteration:
          1. Score every pattern via MC
          2. Collect trade-level feature data
          3. Train ML model on trade outcomes
          4. Generate random + ML-guided mutants
          5. Score mutants via MC
          6. Keep best variant per slot (original or mutant)

        Args:
            patterns:           List of patterns to refine (≤ topN per TF)
            symbol:             Stock ticker
            interval:           Timeframe ('1d', '1h', '30m')
            calibrationPeriod:  Historical window for calibration (used if
                                calibrationStart/End not provided)
            calibrationStart:   Explicit start date for calibration data.
            calibrationEnd:     Explicit end date for calibration data.
            iterations:         Number of refinement cycles
            verbose:            Print progress

        Returns:
            List of (pattern, compositeScore) sorted best-first.
        """
        if not patterns:
            return []

        # Calibrate once — prefer explicit dates over relative period
        fetcher = StockDataFetcher()
        if calibrationStart and calibrationEnd:
            df = fetcher.fetchData(symbol, interval=interval,
                                   start=calibrationStart, end=calibrationEnd)
        else:
            df = fetcher.fetchData(symbol, interval=interval, period=calibrationPeriod)
        if df is None or len(df) < 30:
            if verbose:
                print("    Insufficient data for refinement calibration")
            return [(p, 0.0) for p in patterns]

        simulator = MCMCSimulator(
            initialFund=self.initialFund,
            forgiveness=self.forgiveness,
            numSimulations=self.simsPerCandidate,
            simulationPeriods=self.simPeriods,
            method=self.mcMethod,
        )
        calibration = simulator._calibrateFromHistory(df)

        currentPatterns = [p.deepCopy() for p in patterns]

        for iteration in range(1, iterations + 1):
            if verbose:
                print(f"\n    --- Refinement iteration {iteration}/{iterations} ---")

            # 1. Score current patterns and collect trade data
            scores, allTradeData = self._scoreAndCollect(
                currentPatterns, simulator, calibration, verbose
            )

            if verbose:
                avgScore = np.mean(scores) if scores else 0
                print(f"    Current avg composite score: {avgScore:.4f}")

            # 2. Train ML model on trade data (if enough data)
            mlModel, scaler, featureNames = self._trainMLModel(allTradeData, verbose)

            # 3. Generate candidates: random + ML mutants
            candidates = []                 # (pattern, parentIdx)
            for pIdx, pattern in enumerate(currentPatterns):
                # Random mutants
                for _ in range(self.randomMutantsPerPattern):
                    mutant = pattern.mutateRandom(strength=self.mutationStrength)
                    candidates.append((mutant, pIdx))

                # ML-guided mutants
                if mlModel is not None:
                    mlMutants = self._generateMLMutants(
                        pattern, pIdx, mlModel, scaler, featureNames, verbose
                    )
                    for m in mlMutants:
                        candidates.append((m, pIdx))

            if verbose:
                print(f"    Generated {len(candidates)} candidate mutations")

            # 4. Score all candidates
            candidateScores = self._scoreBatch(
                [c[0] for c in candidates], simulator, calibration
            )

            # 5. Keep best per slot
            improved = 0
            for pIdx in range(len(currentPatterns)):
                bestScore = scores[pIdx]
                bestPattern = currentPatterns[pIdx]
                bestMethod = 'original'

                for cIdx, (cPattern, parentIdx) in enumerate(candidates):
                    if parentIdx == pIdx and candidateScores[cIdx] > bestScore:
                        bestScore = candidateScores[cIdx]
                        bestPattern = cPattern
                        bestMethod = 'mutant'

                if bestMethod != 'original':
                    currentPatterns[pIdx] = bestPattern
                    scores[pIdx] = bestScore
                    improved += 1
                    # Preserve MC composite score as the pattern's fitness
                    # so downstream steps don't see fitness=0.0 / accuracy=0.0%
                    bestPattern.fitness = bestScore * 100
                    bestPattern.totalMatches = max(bestPattern.totalMatches, 1)
                    bestPattern.correctPredictions = max(bestPattern.correctPredictions, 1)

            if verbose:
                avgScore = np.mean(scores) if scores else 0
                print(f"    Improved {improved}/{len(currentPatterns)} patterns  "
                      f"| New avg score: {avgScore:.4f}")

        # Final sort
        ranked = sorted(zip(currentPatterns, scores),
                        key=lambda x: x[1], reverse=True)
        return ranked

    # =====================================================================
    # MC scoring
    # =====================================================================

    def _scoreAndCollect(self, patterns, simulator, calibration, verbose):
        """
        Score each pattern individually via MC and collect raw trade data
        for the ML model (parallelized across patterns, sequential sims per pattern).

        Returns:
            (scores: list[float], allTradeData: list[dict])
        """
        def score_single_pattern(pIdx, pattern):
            """Score one pattern with multiple simulations (sequential)."""
            pathReturns = []
            pathWins = []
            pathSharpes = []
            tradeData = []
            
            # Run simulations sequentially (avoids nested parallelization deadlock)
            for simIdx in range(self.simsPerCandidate):
                path = simulator._simulateSinglePath(
                    pathId=simIdx,
                    patterns=[pattern],
                    calibration=calibration,
                    numPeriods=self.simPeriods,
                )
                pathReturns.append(path.totalReturnPct)
                pathWins.append(path.getWinRate())
                pathSharpes.append(path.sharpeRatio)
                
                # Collect trade-level features for ML
                for trade in path.trades:
                    tradeFeatures = self._extractTradeFeatures(pattern, trade)
                    tradeFeatures['successful'] = trade['successful']
                    tradeFeatures['returnPct'] = trade['returnPct']
                    tradeFeatures['patternIdx'] = pIdx
                    tradeData.append(tradeFeatures)

            avgReturn = float(np.mean(pathReturns))
            avgWin = float(np.mean(pathWins))
            validSharpes = [s for s in pathSharpes if np.isfinite(s)]
            avgSharpe = float(np.mean(validSharpes)) if validSharpes else 0.0

            score = self._compositeScore(avgSharpe, avgWin, avgReturn)
            return (score, tradeData)
        
        # Evaluate all patterns in parallel (but sims within each pattern are sequential)
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(score_single_pattern)(pIdx, pattern)
            for pIdx, pattern in enumerate(patterns)
        )
        
        # Unpack results
        scores = [r[0] for r in results]
        allTradeData = [td for r in results for td in r[1]]  # Flatten trade data lists
        
        return scores, allTradeData

    def _scoreBatch(self, patterns, simulator, calibration):
        """Score a batch of patterns (lighter: fewer sims for speed, parallelized across patterns)."""
        simsLight = max(50, self.simsPerCandidate // 3)
        
        def score_single_pattern_light(pattern):
            """Score one pattern with reduced simulations (sequential)."""
            pathReturns = []
            pathWins = []
            pathSharpes = []
            
            # Run simulations sequentially (avoids nested parallelization)
            for simIdx in range(simsLight):
                path = simulator._simulateSinglePath(
                    pathId=simIdx,
                    patterns=[pattern],
                    calibration=calibration,
                    numPeriods=self.simPeriods,
                )
                pathReturns.append(path.totalReturnPct)
                pathWins.append(path.getWinRate())
                pathSharpes.append(path.sharpeRatio)
            
            avgReturn = float(np.mean(pathReturns))
            avgWin = float(np.mean(pathWins))
            validSharpes = [s for s in pathSharpes if np.isfinite(s)]
            avgSharpe = float(np.mean(validSharpes)) if validSharpes else 0.0
            return self._compositeScore(avgSharpe, avgWin, avgReturn)
        
        # Parallelize across patterns
        scores = Parallel(n_jobs=-1, prefer="processes")(
            delayed(score_single_pattern_light)(pattern)
            for pattern in patterns
        )
        return scores

    @staticmethod
    def _compositeScore(sharpe, winRate, returnPct):
        """Same formula as MCMCSimulator.rankPatterns — single-pattern version."""
        # Rough normalisation to [0,1]-ish range
        nSharpe = max(0, min(1, (sharpe + 1) / 3))        # typical range -1..2
        nWin = winRate / 100.0                              # 0..100 → 0..1
        nRet = max(0, min(1, (returnPct + 50) / 150))      # typical -50..100
        return 0.4 * nSharpe + 0.3 * nWin + 0.3 * nRet

    # =====================================================================
    # Feature extraction
    # =====================================================================

    def _extractTradeFeatures(self, pattern: PatternChromosome,
                              trade: dict) -> dict:
        """
        Build a flat feature dict from pattern genes + trade context.

        Features per gene:
          gene_{i}_priceChangePct, gene_{i}_volumePct,
          gene_{i}_candleType (0/1/2), gene_{i}_minBodyPct

        Trade-level context:
          entryPrice, exitPrice, priceMove (exit-entry)/entry

        This gives the ML model visibility into *which* gene values
        correlate with success vs. failure.
        """
        features = {}
        for i, gene in enumerate(pattern.genes):
            features[f'gene_{i}_price'] = gene.expectedPriceChangePct
            features[f'gene_{i}_vol'] = gene.expectedVolumePct
            features[f'gene_{i}_candle'] = ['BULLISH', 'BEARISH', 'DOJI'].index(
                gene.candleType.value
            )
            features[f'gene_{i}_body'] = gene.minBodyPct

        features['entryPrice'] = trade.get('entryPrice', 0)
        features['exitPrice'] = trade.get('exitPrice', 0)
        entry = features['entryPrice']
        if entry > 0:
            features['priceMove'] = (features['exitPrice'] - entry) / entry
        else:
            features['priceMove'] = 0.0

        features['patternLength'] = len(pattern.genes)
        return features

    # =====================================================================
    # ML model
    # =====================================================================

    def _trainMLModel(self, tradeData: List[dict], verbose: bool):
        """
        Train a Gradient Boosting classifier on trade outcomes.

        Features = gene values for every position in the pattern
        Target   = trade success (1) or failure (0)

        Returns (model, scaler, featureNames) or (None, None, None) if
        insufficient data.
        """
        if len(tradeData) < 40:
            if verbose:
                print(f"    ML: Skipped — only {len(tradeData)} trades (need ≥40)")
            return None, None, None

        # Determine common feature set (intersection of all trade dicts)
        # excluding target columns
        excludeKeys = {'successful', 'returnPct', 'patternIdx'}
        allKeys = [set(d.keys()) - excludeKeys for d in tradeData]
        commonKeys = sorted(allKeys[0].intersection(*allKeys[1:]))

        if not commonKeys:
            return None, None, None

        X = np.array([[d[k] for k in commonKeys] for d in tradeData])
        y = np.array([1 if d['successful'] else 0 for d in tradeData])

        # Need both classes present
        if len(np.unique(y)) < 2:
            if verbose:
                print(f"    ML: Skipped — only one class in trade outcomes")
            return None, None, None

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(Xs, y)

        accuracy = model.score(Xs, y)
        if verbose:
            importances = model.feature_importances_
            topIdx = np.argsort(importances)[::-1][:5]
            topFeats = [(commonKeys[i], importances[i]) for i in topIdx]
            print(f"    ML: Trained on {len(y)} trades (acc={accuracy:.2f})")
            print(f"    ML: Top features: "
                  + ", ".join(f"{n}={v:.3f}" for n, v in topFeats))

        return model, scaler, commonKeys

    # =====================================================================
    # ML-guided mutation generation
    # =====================================================================

    def _generateMLMutants(self,
                           pattern: PatternChromosome,
                           patternIdx: int,
                           model: GradientBoostingClassifier,
                           scaler: StandardScaler,
                           featureNames: List[str],
                           verbose: bool
                           ) -> List[PatternChromosome]:
        """
        Generate mutations guided by the ML model.

        Strategy:
          1. Identify the most important gene-level features from the model
          2. For each important gene feature (price, volume, candle, body):
             - Try several candidate values around the current value
             - Ask the model: "If I change this gene value, does P(success)
               go up?"
             - Pick the value that maximises predicted success probability
          3. Assemble mutated patterns from the best candidate values

        This is essentially a **model-guided local search** — the ML model
        acts as a cheap surrogate for the full MC simulation, letting us
        explore thousands of gene tweaks quickly.
        """
        mutants: List[PatternChromosome] = []
        importances = model.feature_importances_

        # Build a baseline feature vector for this pattern
        # (use a dummy trade to get gene features, then predict)
        dummyTrade = {
            'entryPrice': 100.0, 'exitPrice': 101.0,
            'successful': True, 'returnPct': 1.0,
        }
        baseFeatures = self._extractTradeFeatures(pattern, dummyTrade)
        baseVector = np.array([baseFeatures.get(k, 0) for k in featureNames])
        baseVectorScaled = scaler.transform(baseVector.reshape(1, -1))
        baseProb = model.predict_proba(baseVectorScaled)[0][1]

        # Find the most important gene-level features
        geneFeatIndices = []
        for fIdx, fName in enumerate(featureNames):
            if fName.startswith('gene_') and importances[fIdx] > 0.01:
                geneFeatIndices.append((fIdx, fName, importances[fIdx]))
        geneFeatIndices.sort(key=lambda x: x[2], reverse=True)

        # --- Strategy A: Sweep each important feature to find optimal ---
        bestTweaks: Dict[str, Tuple[float, float]] = {}  # featureName → (bestVal, bestProb)

        for fIdx, fName, imp in geneFeatIndices[:8]:  # Top 8 most important
            currentVal = baseVector[fIdx]

            # Determine sweep range depending on feature type
            if '_candle' in fName:
                candidates = [0.0, 1.0, 2.0]  # BULLISH / BEARISH / DOJI
            elif '_body' in fName:
                candidates = np.linspace(
                    max(0, currentVal - 0.3), min(1.0, currentVal + 0.3), 7
                ).tolist()
            elif '_vol' in fName:
                candidates = np.linspace(
                    max(0.1, currentVal - 0.6), currentVal + 0.6, 7
                ).tolist()
            elif '_price' in fName:
                candidates = np.linspace(
                    currentVal - 0.06, currentVal + 0.06, 9
                ).tolist()
            else:
                continue

            bestVal, bestProb = currentVal, baseProb
            for val in candidates:
                testVec = baseVector.copy()
                testVec[fIdx] = val
                testScaled = scaler.transform(testVec.reshape(1, -1))
                prob = model.predict_proba(testScaled)[0][1]
                if prob > bestProb:
                    bestProb = prob
                    bestVal = val

            if bestVal != currentVal:
                bestTweaks[fName] = (bestVal, bestProb)

        # --- Strategy B: Apply tweaks to create mutant patterns ---
        # Mutant 1: Apply ALL best tweaks at once
        if bestTweaks:
            allTweaksMutant = pattern.deepCopy()
            self._applyTweaks(allTweaksMutant, bestTweaks)
            mutants.append(allTweaksMutant)

        # Mutant 2-N: Apply individual top tweaks
        for fName, (bestVal, _) in list(bestTweaks.items())[:self.mlMutantsPerPattern - 1]:
            singleMutant = pattern.deepCopy()
            self._applyTweaks(singleMutant, {fName: (bestVal, 0)})
            mutants.append(singleMutant)

        # Mutant N+1: Random combination of 2-3 tweaks
        if len(bestTweaks) >= 2:
            numPick = min(3, len(bestTweaks))
            pickedKeys = random.sample(list(bestTweaks.keys()), numPick)
            comboMutant = pattern.deepCopy()
            self._applyTweaks(comboMutant,
                              {k: bestTweaks[k] for k in pickedKeys})
            mutants.append(comboMutant)

        return mutants[:self.mlMutantsPerPattern]

    def _applyTweaks(self, pattern: PatternChromosome,
                     tweaks: Dict[str, Tuple[float, float]]):
        """
        Apply feature-name → value tweaks to the pattern's genes in-place.

        Feature names follow the format ``gene_{i}_{attr}`` where attr is
        one of: price, vol, candle, body.
        """
        candleMap = {0: CandleType.BULLISH, 1: CandleType.BEARISH, 2: CandleType.DOJI}

        for fName, (val, _) in tweaks.items():
            parts = fName.split('_')  # ['gene', '0', 'price']
            if len(parts) < 3 or parts[0] != 'gene':
                continue
            geneIdx = int(parts[1])
            attr = parts[2]
            if geneIdx >= len(pattern.genes):
                continue

            gene = pattern.genes[geneIdx]
            if attr == 'price':
                gene.expectedPriceChangePct = val
            elif attr == 'vol':
                gene.expectedVolumePct = max(0.1, val)
            elif attr == 'candle':
                gene.candleType = candleMap.get(int(round(val)), gene.candleType)
            elif attr == 'body':
                gene.minBodyPct = max(0.0, min(1.0, val))

        # Reset fitness
        pattern.fitness = 0.0
        pattern.totalMatches = 0
        pattern.correctPredictions = 0
