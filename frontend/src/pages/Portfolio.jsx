import { useState, useEffect, Component } from 'react'
import { useRuns, useStockMeta, useAutoStockMeta, usePortfolioConfig } from '../hooks/useFirestore'
import { db } from '../firebase'
import { collection, addDoc, doc, deleteDoc, setDoc, getDocs, getDoc, updateDoc } from 'firebase/firestore'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import StockMiniChart from '../components/StockMiniChart'
import {
  Plus, Trash2, RefreshCw, Building2, Truck, Users, Globe,
  TrendingUp, TrendingDown, Shield, AlertTriangle, Newspaper,
  ChevronDown, ChevronUp, Zap, BarChart3, DollarSign, Link2,
  Bot, Unlink, LineChart, BookOpen, Eye, GitCompare,
} from 'lucide-react'

const COLORS = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff', '#f0883e', '#56d4dd', '#db61a2']
const TABS = [
  { key: 'overview',     label: 'Overview',      icon: Eye },
  { key: 'live',         label: 'Live Charts',   icon: LineChart },
  { key: 'intelligence', label: 'Intelligence',  icon: BookOpen },
  { key: 'connected',    label: 'Connected',     icon: Bot },
]

class PortfolioErrorBoundary extends Component {
  state = { hasError: false }
  static getDerivedStateFromError() { return { hasError: true } }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 text-center">
          <p className="text-dark-muted text-lg mb-4">Something went wrong loading the Portfolio page.</p>
          <button className="px-4 py-2 bg-accent-blue text-white rounded-lg"
            onClick={() => this.setState({ hasError: false })}>
            Try Again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

function PortfolioInner() {
  const [runs] = useRuns(1)
  const [stockMeta, metaLoading] = useStockMeta()
  const [autoStockMeta, autoLoading] = useAutoStockMeta()
  const [portfolioConfig, portfolioConfigLoading, updateTotalFund] = usePortfolioConfig()
  const [newTicker, setNewTicker] = useState('')
  const [adding, setAdding] = useState(false)
  const [refreshing, setRefreshing] = useState(null)
  const [expanded, setExpanded] = useState(null)
  const [tab, setTab] = useState('overview')
  const [chartRange, setChartRange] = useState('3M')
  const [activeStock, setActiveStock] = useState(null)   // selected ticker in live tab
  const [compareMode, setCompareMode] = useState(false)  // compare overlay toggle
  const [compareSet, setCompareSet] = useState(new Set()) // stocks toggled into compare
  const [editingFund, setEditingFund] = useState(false)
  const [fundInput, setFundInput] = useState('')

  const latestRun = runs[0]
  const manualMeta = stockMeta.filter((m) => m.portfolioType !== 'automatic')
  const totalFund = portfolioConfig.totalFund ?? 100000
  const fundFromRun = (latestRun?.totalFund > 0 ? latestRun.totalFund : null) ?? totalFund
  const returnPct = latestRun?.totalReturnPct ?? 0
  const valueAfterRun = fundFromRun * (1 + returnPct / 100)

  // ── Derived symbol lists (must be declared BEFORE any useEffect that uses them) ──
  const symbols     = manualMeta.map((m) => m.symbol).sort()
  const autoSymbols = autoStockMeta.map((m) => m.symbol).sort()
  const symbolsKey  = symbols.join(',') // stable string for effect deps

  const [customTickers, setCustomTickers] = useState([])
  const [customInput, setCustomInput] = useState('')

  // All symbols for live charts: portfolio + custom
  const liveSymbols = [...symbols, ...customTickers.filter(t => !symbols.includes(t))]
  const liveSymbolsKey = liveSymbols.join(',')

  // Auto-select first stock when entering live tab
  useEffect(() => {
    if (tab === 'live' && !compareMode && !activeStock && liveSymbols.length > 0) {
      setActiveStock(liveSymbols[0])
    }
  }, [tab, liveSymbolsKey, compareMode]) // eslint-disable-line react-hooks/exhaustive-deps

  // Seed compareSet with all symbols when entering compare mode
  useEffect(() => {
    if (compareMode) setCompareSet(new Set(liveSymbols))
  }, [compareMode, liveSymbolsKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleAddCustomTicker = () => {
    const t = customInput.trim().toUpperCase()
    if (t && !liveSymbols.includes(t)) {
      setCustomTickers(prev => [...prev, t])
      setActiveStock(t)
    }
    setCustomInput('')
  }
  const handleRemoveCustom = (t) => {
    setCustomTickers(prev => {
      const updated = prev.filter(x => x !== t)
      if (activeStock === t) {
        const remaining = [...symbols, ...updated.filter(x => !symbols.includes(x))]
        setActiveStock(remaining[0] || null)
      }
      return updated
    })
  }

  useEffect(() => {
    const migrate = async () => {
      try {
        const snap = await getDocs(collection(db, 'stock_metadata'))
        await Promise.all(
          snap.docs
            .filter((d) => d.data().addedAt && (d.data().inPortfolio !== true || !d.data().portfolioType))
            .map((d) => setDoc(doc(db, 'stock_metadata', d.id), {
              inPortfolio: true,
              portfolioType: d.data().portfolioType || 'manual',
            }, { merge: true }))
        )
      } catch (e) {
        console.warn('[Portfolio] Migration error:', e)
      }
    }
    migrate()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const allocData = symbols.map((sym, i) => ({
    name: sym,
    value: +(100 / (symbols.length || 1)).toFixed(1),
    color: COLORS[i % COLORS.length],
  }))

  const handleRemoveAuto = async (sym) => {
    if (!confirm(`Remove ${sym} from automatic portfolio? The system will stop trading it.`)) return
    try {
      await updateDoc(doc(db, 'stock_metadata', sym), {
        inPortfolio: false, portfolioType: '', autoScore: 0,
        autoRemovedAt: new Date().toISOString(),
      })
    } catch (err) { alert('Failed to remove: ' + err.message) }
  }

  const handleEvalConnected = async () => {
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'connected_stocks', status: 'queued',
        description: 'Evaluate connected stocks', stocks: symbols,
        config: {}, source: 'website', createdAt: new Date().toISOString(),
        approvedAt: null, startedAt: null, completedAt: null,
        result: null, error: null,
      })
      alert('Connected stocks evaluation queued!')
    } catch (err) { alert('Failed: ' + err.message) }
  }

  const handleAdd = async () => {
    const ticker = newTicker.trim().toUpperCase()
    if (!ticker || symbols.includes(ticker)) { setNewTicker(''); return }
    setAdding(true)
    try {
      const existingDoc = await getDoc(doc(db, 'stock_metadata', ticker))
      const alreadyHasData = existingDoc.exists() && existingDoc.data().description
        && existingDoc.data().description !== 'Pending research...'

      const writeData = { symbol: ticker, addedAt: new Date().toISOString(), inPortfolio: true, portfolioType: 'manual' }
      if (!alreadyHasData) {
        writeData.sector = ''; writeData.industry = ''
        writeData.description = 'Pending research...'
        writeData.relatedTickers = []; writeData.marketCapBucket = ''
        writeData.needsResearch = true
      }
      await setDoc(doc(db, 'stock_metadata', ticker), writeData, { merge: true })

      if (!alreadyHasData) {
        await addDoc(collection(db, 'run_commands'), {
          type: 'stock_research', status: 'queued',
          description: `Research stock: ${ticker}`, stocks: [ticker],
          config: {}, source: 'website', createdAt: new Date().toISOString(),
          approvedAt: null, startedAt: null, completedAt: null, result: null, error: null,
        })
      }

      // Auto-queue the add-stock pipeline so the new stock gets patterns,
      // ML models, backtest data, and allocation without a full re-run.
      const allStocks = [...symbols, ticker]
      await addDoc(collection(db, 'run_commands'), {
        type: 'add_stock_pipeline', status: 'queued',
        description: `Add stock pipeline: ${ticker}`,
        stocks: allStocks,
        config: {
          newSymbols: [ticker],
          POPULATION_SIZE: 200,
          NUM_GENERATIONS: 20,
          MC_NUM_SIMULATIONS: 100,
          MC_TIME_HORIZON: 63,
          ML_FORWARD_PERIODS: 5,
          ML_PORTFOLIO_FORWARD_PERIODS: 5,
          BACKTEST_LOOKBACK_DAYS: 180,
          ML_TRAIN_DAYS: 365,
        },
        source: 'website', createdAt: new Date().toISOString(),
        approvedAt: null, startedAt: null, completedAt: null, result: null, error: null,
      })

      setNewTicker('')
    } catch (err) { alert('Failed to add: ' + err.message) }
    setAdding(false)
  }

  const handleRemove = async (sym) => {
    if (!confirm(`Remove ${sym} from your portfolio?`)) return
    try { await deleteDoc(doc(db, 'stock_metadata', sym)) } catch (err) { alert('Failed: ' + err.message) }
  }

  const handleRefresh = async (sym) => {
    setRefreshing(sym)
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'stock_research', status: 'queued',
        description: `Refresh research: ${sym}`, stocks: [sym],
        config: {}, source: 'website', createdAt: new Date().toISOString(),
        approvedAt: null, startedAt: null, completedAt: null, result: null, error: null,
      })
    } catch (err) { alert('Refresh failed: ' + err.message) }
    setRefreshing(null)
  }

  const handleKeyDown = (e) => { if (e.key === 'Enter') handleAdd() }

  const perStock = latestRun?.perStockResults || {}

  return (
    <div className={tab === 'live' ? 'px-6 pt-6 pb-0 space-y-3' : 'p-6 space-y-5'}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Portfolio</h1>
          <p className="text-dark-muted text-sm mt-0.5">
            {symbols.length} stock{symbols.length !== 1 ? 's' : ''}
            {autoSymbols.length > 0 && <span className="text-accent-purple"> + {autoSymbols.length} connected</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {symbols.length > 0 && (
            <button onClick={handleEvalConnected}
              className="btn-secondary flex items-center gap-1.5 text-xs px-3 py-1.5"
              title="Find and evaluate supply-chain connected stocks">
              <Bot className="w-3.5 h-3.5" /> Evaluate Connected
            </button>
          )}
        </div>
      </div>

      {/* Portfolio value — fund amount & value after last run */}
      <div className="card border-accent-blue/20 bg-dark-card/80">
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-3">
            <DollarSign className="w-5 h-5 text-accent-blue" />
            <div>
              <div className="text-xs text-dark-muted uppercase tracking-wide">Fund size</div>
              {portfolioConfigLoading ? (
                <div className="text-sm text-dark-muted">...</div>
              ) : editingFund ? (
                <div className="flex items-center gap-2 mt-0.5">
                  <input
                    type="number"
                    min={1}
                    step={1000}
                    value={fundInput}
                    onChange={(e) => setFundInput(e.target.value)}
                    className="input text-lg font-mono w-36"
                    placeholder={String(totalFund)}
                    autoFocus
                  />
                  <button
                    type="button"
                    onClick={async () => {
                      const raw = String(fundInput).replace(/,/g, '').trim()
                      const v = Number(raw)
                      if (!Number.isFinite(v) || v <= 0) {
                        alert('Please enter a valid amount (e.g. 100000)')
                        return
                      }
                      try {
                        await updateTotalFund(v)
                        setEditingFund(false)
                        setFundInput('')
                      } catch (err) {
                        console.error('Save fund size:', err)
                        alert('Failed to save: ' + (err?.message || err))
                      }
                    }}
                    className="btn-primary text-xs px-3 py-1"
                  >
                    Save
                  </button>
                  <button type="button" onClick={() => { setEditingFund(false); setFundInput('') }} className="text-xs text-dark-muted hover:text-dark-text">
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => { setEditingFund(true); setFundInput(String(totalFund)) }}
                  className="text-left text-lg font-bold font-mono text-accent-blue hover:underline"
                >
                  ${totalFund.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </button>
              )}
            </div>
          </div>
          {latestRun && (
            <>
              <div className="h-8 w-px bg-dark-border" />
              <div>
                <div className="text-xs text-dark-muted uppercase tracking-wide">Value after last run</div>
                <div className={`text-lg font-bold font-mono ${returnPct >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  ${valueAfterRun.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  <span className="text-sm font-normal ml-2 opacity-90">
                    ({returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%)
                  </span>
                </div>
                {latestRun.timestamp && (
                  <div className="text-[10px] text-dark-muted mt-0.5">
                    Run: {new Date(latestRun.timestamp).toLocaleString()}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-dark-card rounded-xl border border-dark-border p-1">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button key={key} onClick={() => setTab(key)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all flex-1 justify-center ${
              tab === key
                ? 'bg-accent-blue/15 text-accent-blue shadow-sm'
                : 'text-dark-muted hover:text-dark-text hover:bg-dark-hover/50'
            }`}>
            <Icon className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {/* ═══════════════ OVERVIEW TAB ═══════════════ */}
      {tab === 'overview' && (
        <div className="space-y-5">
          {/* Add stock + stock list side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="card space-y-4">
              <h3 className="text-sm font-medium text-dark-muted">Add Stock</h3>
              <div className="flex gap-2">
                <input className="input text-sm font-mono flex-1 uppercase"
                  placeholder="e.g. AAPL" value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                  onKeyDown={handleKeyDown} maxLength={10} />
                <button onClick={handleAdd} disabled={adding || !newTicker.trim()}
                  className="btn-primary flex items-center gap-1.5 text-sm px-4">
                  <Plus className="w-4 h-4" /> {adding ? '...' : 'Add'}
                </button>
              </div>
              <div className="space-y-0.5 max-h-52 overflow-y-auto">
                {symbols.map((sym, i) => (
                  <div key={sym} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-dark-hover/50 group">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-sm" style={{ background: COLORS[i % COLORS.length] }} />
                      <span className="font-mono text-sm font-medium text-accent-blue">{sym}</span>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => handleRefresh(sym)} className="p-1 text-dark-muted hover:text-accent-blue" title="Refresh research">
                        <RefreshCw className={`w-3 h-3 ${refreshing === sym ? 'animate-spin' : ''}`} />
                      </button>
                      <button onClick={() => handleRemove(sym)} className="p-1 text-dark-muted hover:text-accent-red" title="Remove">
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Allocation donut */}
            {allocData.length > 0 && (
              <div className="card lg:col-span-2">
                <h3 className="text-sm font-medium text-dark-muted mb-3">Allocation</h3>
                <div className="flex items-center gap-6">
                  <ResponsiveContainer width="45%" height={200}>
                    <PieChart>
                      <Pie data={allocData} cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                        paddingAngle={2} dataKey="value">
                        {allocData.map((e) => <Cell key={e.name} fill={e.color} />)}
                      </Pie>
                      <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3' }}
                        formatter={(v) => `${v}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
                    {allocData.map((d) => (
                      <div key={d.name} className="flex items-center gap-2 text-xs">
                        <div className="w-2 h-2 rounded-sm" style={{ background: d.color }} />
                        <span className="font-mono font-medium">{d.name}</span>
                        <span className="text-dark-muted">{d.value}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Per-stock performance from latest run */}
          {latestRun && Object.keys(perStock).length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-dark-muted">Latest Run Performance</h3>
                <span className="text-[10px] text-dark-muted">
                  {latestRun.timestamp ? new Date(latestRun.timestamp).toLocaleDateString() : ''}
                </span>
              </div>
              <div className="grid gap-2">
                {Object.entries(perStock).map(([sym, s]) => {
                  const ret = s.returnPct || 0
                  const wr = s.winRate || 0
                  return (
                    <div key={sym} className="flex items-center gap-4 bg-dark-hover/30 rounded-lg px-4 py-2.5">
                      <span className="font-mono text-sm font-bold text-accent-blue w-16">{sym}</span>
                      <div className="flex-1 h-1.5 bg-dark-border rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${ret >= 0 ? 'bg-accent-green' : 'bg-accent-red'}`}
                          style={{ width: `${Math.min(100, Math.abs(ret) * 3)}%` }} />
                      </div>
                      <span className={`font-mono text-sm font-bold w-20 text-right ${ret >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {ret >= 0 ? '+' : ''}{ret.toFixed(2)}%
                      </span>
                      <span className={`text-xs w-14 text-right ${wr >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {wr.toFixed(0)}% WR
                      </span>
                      <span className="text-xs text-dark-muted w-16 text-right">{s.trades || 0} trades</span>
                      <span className={`font-mono text-xs w-20 text-right ${(s.profit || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        ${(s.profit || 0) >= 0 ? '+' : ''}{(s.profit || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Quick stock cards (compact) */}
          {manualMeta.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
              {manualMeta.map((m) => {
                const hasData = m.description && m.description !== 'Pending research...'
                return (
                  <div key={m.id} className="card py-3 px-4 hover:border-accent-blue/30 transition-all cursor-pointer"
                    onClick={() => { setTab('intelligence'); setExpanded(m.symbol) }}>
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="font-mono font-bold text-accent-blue">{m.symbol}</span>
                      {m.sector
                        ? <span className="text-[9px] text-dark-muted bg-dark-hover px-1.5 py-0.5 rounded">{m.sector}</span>
                        : <span className="text-[9px] text-dark-muted/50 bg-dark-hover/50 px-1.5 py-0.5 rounded border border-dashed border-dark-border">Pending</span>
                      }
                    </div>
                    {hasData
                      ? <p className="text-[11px] text-dark-text/70 line-clamp-2 leading-relaxed">{m.description}</p>
                      : <p className="text-[11px] text-dark-muted/50 italic">Research not yet available — queue a run to fetch data.</p>
                    }
                    <div className="flex items-center gap-3 mt-2 text-[10px] text-dark-muted">
                      {m.approximateMarketCap && <span>{m.approximateMarketCap}</span>}
                      {m.analystConsensus && (
                        <span className={m.analystConsensus.toLowerCase().includes('buy') ? 'text-accent-green' :
                          m.analystConsensus.toLowerCase().includes('sell') ? 'text-accent-red' : ''}>
                          {m.analystConsensus.split(' — ')[0]}
                        </span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* ═══════════════ LIVE CHARTS TAB ═══════════════ */}
      {tab === 'live' && (
        <div className="space-y-0">
          {liveSymbols.length === 0 ? (
            <div className="card text-center text-dark-muted py-12 text-sm space-y-3">
              <p>Add stocks in the Overview tab to see live charts.</p>
              <div className="flex items-center justify-center gap-2">
                <input
                  className="input text-sm font-mono w-28 uppercase text-center"
                  placeholder="TICKER"
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value.toUpperCase())}
                  onKeyDown={(e) => e.key === 'Enter' && handleAddCustomTicker()}
                  maxLength={10}
                />
                <button onClick={handleAddCustomTicker} disabled={!customInput.trim()}
                  className="btn-primary text-xs px-3 py-1.5">
                  <Plus className="w-3.5 h-3.5 inline mr-1" /> Add Chart
                </button>
              </div>
            </div>
          ) : compareMode ? (
            /* ══════════════ COMPARE MODE — side-by-side grid ══════════════ */
            <div className="space-y-3">
              {/* Compare header bar */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-semibold">Compare</span>
                  {liveSymbols.map((sym, i) => (
                    <button
                      key={sym}
                      onClick={() => setCompareSet(prev => {
                        const next = new Set(prev)
                        if (next.has(sym) && next.size > 1) next.delete(sym)
                        else next.add(sym)
                        return next
                      })}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${
                        compareSet.has(sym)
                          ? 'border-transparent text-dark-bg font-bold'
                          : 'bg-transparent text-dark-muted border-dark-border hover:text-dark-text'
                      }`}
                      style={compareSet.has(sym) ? { background: COLORS[i % COLORS.length] } : {}}
                    >
                      <span>{sym}</span>
                    </button>
                  ))}
                  <span className="text-xs text-dark-muted">— click to toggle</span>
                </div>
                <div className="flex items-center gap-2">
                  {/* Range pills */}
                  <div className="flex gap-1">
                    {['1D', '1M', '3M', '6M', '12M', 'ALL'].map((r) => (
                      <button key={r} onClick={() => setChartRange(r)}
                        className={`px-2.5 py-1 rounded-full text-xs font-semibold transition-all ${
                          chartRange === r ? 'bg-dark-text text-dark-bg' : 'text-dark-muted hover:text-dark-text'
                        }`}>{r}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => setCompareMode(false)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent-blue/15 text-accent-blue border border-accent-blue/30"
                  >
                    <GitCompare className="w-3.5 h-3.5" /> Exit Compare
                  </button>
                </div>
              </div>

              {/* Side-by-side grid */}
              {(() => {
                const visible = liveSymbols.filter(s => compareSet.has(s))
                const cols = visible.length <= 2 ? visible.length
                  : visible.length <= 4 ? 2 : 3
                const chartH = visible.length <= 2
                  ? 'calc(100vh - 220px)'
                  : visible.length <= 4 ? 420 : 360
                return (
                  <div
                    className="grid gap-3"
                    style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
                  >
                    {visible.map((sym, i) => {
                      const meta = manualMeta.find(m => m.symbol === sym)
                      const run  = perStock[sym]
                      const isCustom = customTickers.includes(sym)
                      return (
                        <div key={sym} className="card p-0 overflow-hidden flex flex-col">
                          {/* Mini header */}
                          <div className="flex items-center justify-between px-4 py-2.5 border-b border-dark-border/50">
                            <div className="flex items-center gap-2">
                              <div className="w-2.5 h-2.5 rounded-full"
                                style={{ background: COLORS[liveSymbols.indexOf(sym) % COLORS.length] }} />
                              <span className="font-mono font-bold text-sm">{sym}</span>
                              {meta?.companyName && meta.companyName !== sym && (
                                <span className="text-xs text-dark-muted hidden sm:inline truncate max-w-[120px]">
                                  {meta.companyName}
                                </span>
                              )}
                            </div>
                            {run?.returnPct !== undefined && (
                              <span className={`text-xs font-mono font-bold ${run.returnPct >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                                {run.returnPct >= 0 ? '+' : ''}{run.returnPct.toFixed(2)}%
                              </span>
                            )}
                          </div>
                          {/* Chart */}
                          <div style={{ height: typeof chartH === 'number' ? chartH : chartH, flex: 1 }}>
                            <StockMiniChart symbol={sym} height="100%" range={chartRange} />
                          </div>
                          {/* Mini stats */}
                          {(meta?.approximateMarketCap || meta?.analystConsensus || run?.winRate !== undefined) && (
                            <div className="flex items-center gap-4 px-4 py-2 border-t border-dark-border/50 bg-dark-card/40 flex-wrap">
                              {meta?.approximateMarketCap && (
                                <span className="text-[10px] text-dark-muted">Cap: <span className="text-dark-text">{meta.approximateMarketCap}</span></span>
                              )}
                              {meta?.trailingPE && meta.trailingPE !== 'N/A' && (
                                <span className="text-[10px] text-dark-muted">P/E: <span className="text-dark-text">{meta.trailingPE}</span></span>
                              )}
                              {meta?.analystConsensus && (
                                <span className={`text-[10px] font-medium ${
                                  meta.analystConsensus.toLowerCase().includes('buy') ? 'text-accent-green' :
                                  meta.analystConsensus.toLowerCase().includes('sell') ? 'text-accent-red' : 'text-dark-muted'
                                }`}>{meta.analystConsensus.split(' — ')[0]}</span>
                              )}
                              {run?.winRate !== undefined && (
                                <span className={`text-[10px] ${run.winRate >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                                  WR: {run.winRate.toFixed(0)}%
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )
              })()}
            </div>
          ) : (
            /* ══════════════ SINGLE STOCK MODE ══════════════ */
            <div className="card p-0 overflow-hidden">
              {/* Stock tabs + Compare toggle + Custom ticker */}
              <div className="flex items-center gap-0 border-b border-dark-border bg-dark-card/60 overflow-x-auto">
                {liveSymbols.map((sym, i) => {
                  const meta = manualMeta.find(m => m.symbol === sym)
                  const ret  = perStock[sym]?.returnPct
                  const isActive = activeStock === sym
                  const isCustom = customTickers.includes(sym)
                  return (
                    <button key={sym} onClick={() => setActiveStock(sym)}
                      className={`relative flex flex-col items-start px-5 py-3 border-r border-dark-border flex-shrink-0 transition-all group ${
                        isActive ? 'bg-dark-bg text-dark-text' : 'text-dark-muted hover:text-dark-text hover:bg-dark-hover/30'
                      }`}
                    >
                      {isActive && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t"
                          style={{ background: COLORS[i % COLORS.length] }} />
                      )}
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ background: COLORS[i % COLORS.length] }} />
                        <span className="font-mono font-bold text-sm">{sym}</span>
                        {isCustom && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleRemoveCustom(sym) }}
                            className="text-dark-muted hover:text-accent-red opacity-0 group-hover:opacity-100 transition-opacity"
                            title="Remove custom chart"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                      {meta?.companyName && meta.companyName !== sym && (
                        <span className="text-[10px] text-dark-muted/70 truncate max-w-[100px] mt-0.5 pl-4">
                          {meta.companyName}
                        </span>
                      )}
                      {ret !== undefined && (
                        <span className={`text-[10px] font-mono pl-4 mt-0.5 ${ret >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                          {ret >= 0 ? '+' : ''}{ret.toFixed(2)}%
                        </span>
                      )}
                    </button>
                  )
                })}

                {/* Add custom ticker + Compare button — right side */}
                <div className="flex items-center gap-2 ml-auto px-3 flex-shrink-0">
                  <input
                    className="input text-xs font-mono w-20 uppercase text-center py-1"
                    placeholder="+ TICKER"
                    value={customInput}
                    onChange={(e) => setCustomInput(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddCustomTicker()}
                    maxLength={10}
                  />
                  <button
                    onClick={() => setCompareMode(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-dark-muted border border-dark-border hover:text-dark-text hover:border-dark-text/30 transition-all"
                    title="View all stocks side by side"
                  >
                    <GitCompare className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Compare</span>
                  </button>
                </div>
              </div>

              {/* Range pills */}
              <div className="flex items-center gap-1 px-4 py-2.5 border-b border-dark-border/50 bg-dark-bg/40">
                {['1D', '5D', '1M', '3M', '6M', '12M', 'ALL'].map((r) => (
                  <button key={r} onClick={() => setChartRange(r)}
                    className={`px-3 py-1 rounded-full text-xs font-semibold transition-all ${
                      chartRange === r ? 'bg-dark-text text-dark-bg' : 'text-dark-muted hover:text-dark-text'
                    }`}>{r}
                  </button>
                ))}
              </div>

              {/* Full-height chart */}
              <div className="bg-dark-bg" style={{ height: 'calc(100vh - 320px)', minHeight: 380 }}>
                {activeStock && <StockMiniChart symbol={activeStock} height="100%" range={chartRange} />}
              </div>

              {/* Stats row */}
              {activeStock && (
                <StockStatsRow
                  meta={manualMeta.find(m => m.symbol === activeStock) || { symbol: activeStock }}
                  run={perStock[activeStock]}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* ═══════════════ INTELLIGENCE TAB ═══════════════ */}
      {tab === 'intelligence' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-dark-muted">
              Stock Intelligence <span className="text-accent-blue">(Your Stocks)</span>
            </h3>
          </div>
          {metaLoading ? (
            <div className="text-dark-muted text-sm py-8 text-center">Loading...</div>
          ) : manualMeta.length === 0 ? (
            <div className="card text-center text-dark-muted py-8 text-sm">
              No stocks yet. Add a ticker in the Overview tab.
            </div>
          ) : (
            <div className="space-y-3">
              {manualMeta.map((m) => {
                const isExpanded = expanded === m.symbol
                const hasData = m.description && m.description !== 'Pending research...'
                return (
                  <div key={m.id} className="card hover:border-accent-blue/30 transition-all">
                    <div className="flex items-start justify-between cursor-pointer"
                      onClick={() => setExpanded(isExpanded ? null : m.symbol)}>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 flex-wrap">
                          <span className="text-lg font-mono font-bold text-accent-blue">{m.symbol}</span>
                          {m.companyName && m.companyName !== m.symbol && (
                            <span className="text-sm text-dark-text/80">{m.companyName}</span>
                          )}
                          {m.sector && <span className="badge-muted text-[10px]">{m.sector}</span>}
                          {m.industry && <span className="badge-muted text-[10px]">{m.industry}</span>}
                          {m.needsResearch && <span className="badge-yellow text-[10px]">Pending Research</span>}
                        </div>

                        {hasData && (
                          <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                            {m.approximateMarketCap && (
                              <span className="flex items-center gap-1 text-dark-muted">
                                <DollarSign className="w-3 h-3" /> {m.approximateMarketCap}
                              </span>
                            )}
                            {m.trailingPE && m.trailingPE !== 'N/A' && <span className="text-dark-muted">P/E: {m.trailingPE}</span>}
                            {m.dividendYield && m.dividendYield !== 'None' && <span className="text-dark-muted">Div: {m.dividendYield}</span>}
                            {m.revenueGrowthYoY && (
                              <span className={m.revenueGrowthYoY.includes('+') ? 'text-accent-green' : 'text-accent-red'}>
                                Rev Growth: {m.revenueGrowthYoY}
                              </span>
                            )}
                            {m.profitMargin && <span className="text-dark-muted">Margin: {m.profitMargin}</span>}
                            {m.analystConsensus && (
                              <span className={`font-medium ${
                                m.analystConsensus.toLowerCase().includes('buy') ? 'text-accent-green' :
                                m.analystConsensus.toLowerCase().includes('sell') ? 'text-accent-red' : 'text-dark-muted'
                              }`}>Analyst: {m.analystConsensus.split(' — ')[0]}</span>
                            )}
                          </div>
                        )}

                        {hasData && (
                          <p className={`text-xs text-dark-text/75 mt-2 leading-relaxed ${isExpanded ? '' : 'line-clamp-2'}`}>
                            {m.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 ml-3 flex-shrink-0">
                        <button onClick={(e) => { e.stopPropagation(); handleRefresh(m.symbol) }}
                          className="p-1.5 text-dark-muted hover:text-accent-blue rounded-lg hover:bg-dark-hover" title="Refresh">
                          <RefreshCw className={`w-4 h-4 ${refreshing === m.symbol ? 'animate-spin' : ''}`} />
                        </button>
                        {isExpanded ? <ChevronUp className="w-4 h-4 text-dark-muted" /> : <ChevronDown className="w-4 h-4 text-dark-muted" />}
                      </div>
                    </div>

                    {isExpanded && hasData && (
                      <div className="mt-4 pt-4 border-t border-dark-border space-y-4">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                          {m.mainBusiness && (
                            <InfoPanel icon={Globe} title="Core Business" color="blue">
                              <p className="text-xs text-dark-text/85 leading-relaxed">{m.mainBusiness}</p>
                            </InfoPanel>
                          )}
                          {m.competitiveAdvantage && (
                            <InfoPanel icon={Shield} title="Competitive Advantage" color="green">
                              <p className="text-xs text-dark-text/85 leading-relaxed">{m.competitiveAdvantage}</p>
                            </InfoPanel>
                          )}
                        </div>

                        {m.revenueBreakdown?.length > 0 && (
                          <InfoPanel icon={BarChart3} title="Revenue Breakdown">
                            <div className="flex flex-wrap gap-1.5">
                              {m.revenueBreakdown.map((seg, i) => (
                                <span key={i} className="px-2 py-1 rounded bg-dark-bg text-xs text-dark-text/80 border border-dark-border">{seg}</span>
                              ))}
                            </div>
                          </InfoPanel>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                          {m.suppliers?.length > 0 && (
                            <InfoPanel icon={Truck} title="Key Suppliers">
                              <ul className="space-y-0.5">
                                {m.suppliers.map((s, i) => (
                                  <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                    <span className="text-accent-blue mt-0.5">›</span> {s}
                                  </li>
                                ))}
                              </ul>
                            </InfoPanel>
                          )}
                          {m.customers?.length > 0 && (
                            <InfoPanel icon={Users} title="Key Customers">
                              <ul className="space-y-0.5">
                                {m.customers.map((c, i) => (
                                  <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                    <span className="text-accent-green mt-0.5">›</span> {c}
                                  </li>
                                ))}
                              </ul>
                            </InfoPanel>
                          )}
                        </div>

                        {m.supplyChainNotes && (
                          <InfoPanel icon={Link2} title="Supply Chain Notes">
                            <p className="text-xs text-dark-text/80 leading-relaxed">{m.supplyChainNotes}</p>
                          </InfoPanel>
                        )}

                        {m.competitors?.length > 0 && (
                          <InfoPanel icon={BarChart3} title="Competitors">
                            <ul className="space-y-0.5">
                              {m.competitors.map((c, i) => (
                                <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                  <span className="text-accent-red mt-0.5">›</span> {c}
                                </li>
                              ))}
                            </ul>
                          </InfoPanel>
                        )}

                        {m.relatedTickers?.length > 0 && (
                          <div>
                            <div className="text-xs text-dark-muted font-medium mb-1.5">Related Tickers</div>
                            <div className="flex flex-wrap gap-1.5">
                              {m.relatedTickers.map((t) => (
                                <span key={t} className="px-2 py-0.5 rounded bg-accent-blue/10 text-xs font-mono text-accent-blue border border-accent-blue/20">{t}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {m.recentHeadlines?.length > 0 && (
                          <InfoPanel icon={Newspaper} title="Recent Headlines">
                            <ul className="space-y-1">
                              {m.recentHeadlines.map((h, i) => (
                                <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                  <span className="text-dark-muted mt-0.5">•</span> {h}
                                </li>
                              ))}
                            </ul>
                          </InfoPanel>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                          {(m.upcomingCatalysts?.length > 0 || m.catalysts?.length > 0) && (
                            <div className="bg-accent-green/5 border border-accent-green/15 rounded-lg p-3">
                              <div className="text-xs font-medium flex items-center gap-1.5 mb-1.5 text-accent-green">
                                <TrendingUp className="w-3.5 h-3.5" /> Catalysts
                              </div>
                              <ul className="space-y-0.5">
                                {(m.upcomingCatalysts || m.catalysts || []).map((c, i) => (
                                  <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                    <span className="text-accent-green mt-0.5">▸</span> {c}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {m.keyRisks?.length > 0 && (
                            <div className="bg-accent-red/5 border border-accent-red/15 rounded-lg p-3">
                              <div className="text-xs font-medium flex items-center gap-1.5 mb-1.5 text-accent-red">
                                <AlertTriangle className="w-3.5 h-3.5" /> Key Risks
                              </div>
                              <ul className="space-y-0.5">
                                {m.keyRisks.map((r, i) => (
                                  <li key={i} className="text-xs text-dark-text/80 flex items-start gap-1.5">
                                    <span className="text-accent-red mt-0.5">▸</span> {r}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>

                        <div className="flex flex-wrap items-center gap-4 text-xs text-dark-muted pt-1">
                          {m.analystConsensus && <span>Consensus: <span className="text-dark-text">{m.analystConsensus}</span></span>}
                          {m.marketCapBucket && <span>Cap: <span className="text-dark-text">{m.marketCapBucket}</span></span>}
                          {m.lastResearchedAt && <span>Updated: {new Date(m.lastResearchedAt).toLocaleDateString()}</span>}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* ═══════════════ CONNECTED STOCKS TAB ═══════════════ */}
      {tab === 'connected' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-dark-muted flex items-center gap-2">
              <Bot className="w-4 h-4 text-accent-purple" />
              Connected Stocks
              <span className="text-xs text-accent-purple">(Auto-managed by ML)</span>
            </h3>
            {autoSymbols.length > 0 && (
              <span className="text-xs text-dark-muted">{autoSymbols.length} stock{autoSymbols.length !== 1 ? 's' : ''}</span>
            )}
          </div>

          {autoLoading ? (
            <div className="text-dark-muted text-sm py-6 text-center">Loading...</div>
          ) : autoStockMeta.length === 0 ? (
            <div className="card text-center text-dark-muted py-10 text-sm border-dashed space-y-2">
              <Bot className="w-8 h-8 mx-auto opacity-30" />
              <p>No connected stocks yet.</p>
              <p className="text-xs">Run "Evaluate Connected Stocks" to let ML discover supply-chain related stocks worth trading.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {autoStockMeta.map((m) => (
                <div key={m.id} className="card border-accent-purple/20 hover:border-accent-purple/40 transition-all">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-lg font-mono font-bold text-accent-purple">{m.symbol}</span>
                        {m.companyName && m.companyName !== m.symbol && (
                          <span className="text-xs text-dark-text/80">{m.companyName}</span>
                        )}
                        {m.sector && <span className="badge-muted text-[10px]">{m.sector}</span>}
                        <span className="px-2 py-0.5 rounded text-[10px] bg-accent-purple/10 text-accent-purple border border-accent-purple/20">
                          {m.autoAddedReason ? m.autoAddedReason.split(':')[0] : 'Connected'}
                        </span>
                      </div>

                      <div className="mt-2 space-y-1">
                        {m.autoAddedFrom?.length > 0 && (
                          <p className="text-xs text-dark-muted flex items-center gap-1.5">
                            <Link2 className="w-3 h-3" /> From:
                            {m.autoAddedFrom.map((t) => (
                              <span key={t} className="font-mono text-accent-blue ml-0.5">{t}</span>
                            ))}
                          </p>
                        )}
                        {m.autoScore > 0 && (() => {
                          const reason = m.autoAddedReason || ''
                          const isMLBacked = reason.toLowerCase().includes('ml evaluation')
                          return (
                            <div className="flex items-center gap-3 text-xs flex-wrap">
                              <span className="text-dark-muted">
                                {isMLBacked ? 'ML Score' : 'Prelim. Score'}: <span className={`font-medium ${
                                  m.autoScore > 0.5 ? 'text-accent-green' :
                                  m.autoScore > 0.25 ? 'text-dark-text' : 'text-accent-red'
                                }`}>{(m.autoScore * 100).toFixed(0)}%</span>
                              </span>
                              {!isMLBacked && (
                                <span className="text-[10px] text-yellow-500/80 italic">
                                  Run pipeline to get ML score
                                </span>
                              )}
                              {m.autoAddedAt && <span className="text-dark-muted">Added: {new Date(m.autoAddedAt).toLocaleDateString()}</span>}
                            </div>
                          )
                        })()}
                      </div>
                    </div>
                    <button onClick={() => handleRemoveAuto(m.symbol)}
                      className="p-1.5 text-dark-muted hover:text-accent-red rounded-lg hover:bg-dark-hover ml-3"
                      title="Stop trading this stock">
                      <Unlink className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function InfoPanel({ icon: Icon, title, color = 'muted', children }) {
  return (
    <div className="bg-dark-hover/30 rounded-lg p-3">
      <div className="text-xs text-dark-muted font-medium flex items-center gap-1.5 mb-1.5">
        {Icon && <Icon className="w-3.5 h-3.5" />} {title}
      </div>
      {children}
    </div>
  )
}

function Stat({ label, value, color }) {
  const colorClass = color === 'green' ? 'text-accent-green'
    : color === 'red' ? 'text-accent-red'
    : color === 'muted' ? 'text-dark-muted'
    : 'text-dark-text'
  return (
    <div>
      <div className="text-[9px] text-dark-muted uppercase tracking-wide mb-0.5">{label}</div>
      <div className={`text-sm font-semibold ${colorClass}`}>{value}</div>
    </div>
  )
}

function StockStatsRow({ meta, run }) {
  const hasStats = meta?.approximateMarketCap || meta?.trailingPE ||
    meta?.analystConsensus || run?.returnPct !== undefined
  if (!hasStats) return null
  return (
    <div className="px-5 py-3 border-t border-dark-border/50 flex flex-wrap items-center gap-6 bg-dark-card/40">
      {meta?.approximateMarketCap && (
        <Stat label="Market Cap" value={meta.approximateMarketCap} />
      )}
      {meta?.trailingPE && meta.trailingPE !== 'N/A' && (
        <Stat label="P/E Ratio" value={meta.trailingPE} />
      )}
      {meta?.dividendYield && meta.dividendYield !== 'None' && (
        <Stat label="Dividend" value={meta.dividendYield} />
      )}
      {meta?.revenueGrowthYoY && (
        <Stat label="Rev Growth" value={meta.revenueGrowthYoY}
          color={meta.revenueGrowthYoY.includes('+') ? 'green' : 'red'} />
      )}
      {meta?.analystConsensus && (
        <Stat label="Analyst" value={meta.analystConsensus.split(' — ')[0]}
          color={meta.analystConsensus.toLowerCase().includes('buy') ? 'green' :
            meta.analystConsensus.toLowerCase().includes('sell') ? 'red' : 'muted'} />
      )}
      {run?.returnPct !== undefined && (
        <Stat label="Last Run Return"
          value={`${run.returnPct >= 0 ? '+' : ''}${run.returnPct.toFixed(2)}%`}
          color={run.returnPct >= 0 ? 'green' : 'red'} />
      )}
      {run?.winRate !== undefined && (
        <Stat label="Win Rate" value={`${run.winRate.toFixed(0)}%`}
          color={run.winRate >= 50 ? 'green' : 'red'} />
      )}
      {run?.trades !== undefined && (
        <Stat label="Trades" value={run.trades} />
      )}
    </div>
  )
}

export default function Portfolio() {
  return (
    <PortfolioErrorBoundary>
      <PortfolioInner />
    </PortfolioErrorBoundary>
  )
}
