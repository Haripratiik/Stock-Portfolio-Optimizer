import { useState, useEffect } from 'react'
import { db } from '../firebase'
import {
  collection,
  query,
  orderBy,
  limit,
  onSnapshot,
  where,
  doc,
  getDoc,
  setDoc,
} from 'firebase/firestore'

/**
 * Real-time listener for a Firestore collection.
 * Returns [docs, loading] where docs is an array of {id, ...data}.
 */
export function useCollection(colName, constraints = [], limitN = 50) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const q = query(collection(db, colName), ...constraints, limit(limitN))
    const unsub = onSnapshot(q, (snap) => {
      setDocs(snap.docs.map((d) => ({ id: d.id, ...d.data() })))
      setLoading(false)
    }, (err) => {
      console.warn(`[useCollection] ${colName} query failed:`, err?.message || err)
      setLoading(false)
    })
    return unsub
  }, [colName, limitN]) // eslint-disable-line react-hooks/exhaustive-deps

  return [docs, loading]
}

/** Fetch a single document once. */
export function useDocument(colName, docId) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!docId) { setLoading(false); return }
    getDoc(doc(db, colName, docId)).then((snap) => {
      setData(snap.exists() ? { id: snap.id, ...snap.data() } : null)
      setLoading(false)
    })
  }, [colName, docId])

  return [data, loading]
}

/** Real-time listener for run_commands collection. */
export function useQueue() {
  return useCollection('run_commands', [orderBy('createdAt', 'desc')], 100)
}

/** Real-time listener for latest runs. */
export function useRuns(n = 20) {
  return useCollection('runs', [orderBy('timestamp', 'desc')], n)
}

/** Real-time listener for active patterns (supersededBy == '' means active). */
export function usePatterns(symbol = null, n = 100) {
  const constraints = [where('supersededBy', '==', '')]
  if (symbol) constraints.push(where('symbol', '==', symbol))
  constraints.push(orderBy('mcCompositeScore', 'desc'))
  return useCollection('patterns', constraints, n)
}

/**
 * Fallback: load ALL patterns (no supersededBy filter) and sort client-side.
 * Used when the composite index hasn't been created yet.
 */
export function useAllPatterns(symbol = null, n = 200) {
  const constraints = []
  if (symbol) constraints.push(where('symbol', '==', symbol))
  return useCollection('patterns', constraints, n)
}

/** Real-time listener for active strategies. */
export function useStrategies() {
  return useCollection('strategies', [where('active', '==', true)], 100)
}

/**
 * Real-time listener for portfolio stocks only.
 * Only returns documents explicitly added by the user (inPortfolio === true).
 * Pipeline-seeded metadata docs (sector peers, builtin seeds, etc.) are excluded.
 */
export function useStockMeta() {
  return useCollection('stock_metadata', [where('inPortfolio', '==', true)], 200)
}

/**
 * Real-time listener for automatic (connected) portfolio stocks.
 * Returns stocks added by the ML system based on supply chain analysis.
 */
export function useAutoStockMeta() {
  return useCollection('stock_metadata', [where('portfolioType', '==', 'automatic')], 200)
}

/** Real-time listener for config_templates. */
export function useTemplates() {
  return useCollection('config_templates', [orderBy('createdAt', 'desc')], 50)
}

/**
 * Real-time listener for charts belonging to a specific run.
 * Returns [] immediately when runId is null/undefined.
 * Uses only a single-field filter (no orderBy) to avoid requiring a
 * composite Firestore index. Results are sorted client-side.
 */
export function useRunCharts(runId) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!runId) {
      setDocs([])
      setLoading(false)
      return
    }
    setLoading(true)
    const q = query(
      collection(db, 'run_charts'),
      where('runId', '==', runId),
    )
    const unsub = onSnapshot(
      q,
      (snap) => {
        const sorted = snap.docs
          .map((d) => ({ id: d.id, ...d.data() }))
          .sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''))
        setDocs(sorted)
        setLoading(false)
      },
      (err) => {
        console.error('[useRunCharts] Firestore error:', err)
        setLoading(false)
      },
    )
    return unsub
  }, [runId])

  return [docs, loading]
}

/**
 * Real-time listener for backtest trades for a specific run.
 * Returns one doc per (symbol, interval) pair, each containing a trades array.
 */
export function useBacktestTrades(runId) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!runId) {
      setDocs([])
      setLoading(false)
      return
    }
    setLoading(true)
    const q = query(
      collection(db, 'backtest_trades'),
      where('runId', '==', runId),
    )
    const unsub = onSnapshot(
      q,
      (snap) => {
        setDocs(snap.docs.map((d) => ({ id: d.id, ...d.data() })))
        setLoading(false)
      },
      (err) => {
        console.error('[useBacktestTrades] Firestore error:', err)
        setLoading(false)
      },
    )
    return unsub
  }, [runId])

  return [docs, loading]
}

const DEFAULT_TOTAL_FUND = 100000

/**
 * Real-time listener for portfolio config (fund amount).
 * Used by the Portfolio page as the source of truth for total capital.
 */
export function usePortfolioConfig() {
  const [config, setConfig] = useState({ totalFund: DEFAULT_TOTAL_FUND })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const ref = doc(db, 'app_config', 'portfolio')
    const unsub = onSnapshot(
      ref,
      (snap) => {
        if (snap.exists()) {
          const data = snap.data()
          setConfig({
            totalFund: typeof data.totalFund === 'number' && data.totalFund > 0
              ? data.totalFund
              : DEFAULT_TOTAL_FUND,
          })
        } else {
          setConfig({ totalFund: DEFAULT_TOTAL_FUND })
        }
        setLoading(false)
      },
      () => {
        setConfig({ totalFund: DEFAULT_TOTAL_FUND })
        setLoading(false)
      },
    )
    return unsub
  }, [])

  const updateTotalFund = async (value) => {
    const num = Number(value)
    if (!Number.isFinite(num) || num <= 0) return
    await setDoc(doc(db, 'app_config', 'portfolio'), { totalFund: num }, { merge: true })
  }

  return [config, loading, updateTotalFund]
}

/**
 * Real-time listener for trading config (app_config/trading).
 * Returns [config, loading, updateConfig] where config has:
 *   enabled, mode ('paper'|'live'), lastRun, etc.
 */
export function useTradingConfig() {
  const [config, setConfig] = useState({
    enabled: false,
    mode: 'paper',
    scheduleMode: 'queued',
    lastRunAt: null,
    dailyLossLimitPct: 5,
    maxPctPerStock: 25,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const ref = doc(db, 'app_config', 'trading')
    const unsub = onSnapshot(
      ref,
      (snap) => {
        if (snap.exists()) {
          setConfig({ ...config, ...snap.data() })
        }
        setLoading(false)
      },
      () => setLoading(false),
    )
    return unsub
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const updateConfig = async (updates) => {
    await setDoc(doc(db, 'app_config', 'trading'), updates, { merge: true })
  }

  return [config, loading, updateConfig]
}

/**
 * Real-time listener for recent trade orders (trade_log collection).
 */
export function useTradeLog(n = 50) {
  return useCollection('trade_log', [orderBy('createdAt', 'desc')], n)
}

/**
 * Real-time listener for broker orders (broker_orders collection).
 * Synced from the broker by the LocalAgent.
 */
export function useBrokerOrders(n = 200) {
  return useCollection('broker_orders', [], n)
}

/**
 * Real-time listener for tracked positions (trade_positions).
 */
export function useTradePositions() {
  return useCollection('trade_positions', [], 100)
}

/**
 * Real-time listener for trade cycles (trade_cycles).
 */
export function useTradeCycles(n = 20) {
  return useCollection('trade_cycles', [orderBy('timestamp', 'desc')], n)
}

/**
 * Real-time listener for daily performance reviews (daily_reviews).
 */
export function useDailyReviews(n = 7) {
  return useCollection('daily_reviews', [orderBy('date', 'desc')], n)
}

/**
 * Real-time listener for live broker account (equity, day P&L from Alpaca).
 * Updated after each trading cycle or when user syncs broker.
 */
export function useBrokerAccount() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const ref = doc(db, 'broker_account', 'live')
    const unsub = onSnapshot(
      ref,
      (snap) => {
        setData(snap.exists() ? { id: snap.id, ...snap.data() } : null)
        setLoading(false)
        setError(null)
      },
      (err) => {
        console.warn('[useBrokerAccount] Firestore error:', err?.code, err?.message)
        setError(err?.message || 'Failed to load broker data')
        setLoading(false)
      }
    )
    return unsub
  }, [])

  return [data, loading, error]
}

/**
 * Real-time listener for daily trade summary (next day display).
 */
export function useDailyTradeSummary() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const ref = doc(db, 'app_config', 'daily_trade_summary')
    const unsub = onSnapshot(
      ref,
      (snap) => {
        setData(snap.exists() ? { id: snap.id, ...snap.data() } : null)
        setLoading(false)
      },
      () => setLoading(false)
    )
    return unsub
  }, [])

  return [data, loading]
}
