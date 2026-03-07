import { useRuns, useQueue, useStockMeta, useRunCharts, useStrategies } from '../hooks/useFirestore'
import { db } from '../firebase'
import { collection, getCountFromServer, query, where, doc, deleteDoc, getDocs } from 'firebase/firestore'
import { useState, useEffect, Fragment, useMemo } from 'react'
import StatsCard from '../components/StatsCard'
import { Link } from 'react-router-dom'
import {
  TrendingUp,
  BarChart3,
  Brain,
  Activity,
  Clock,
  Zap,
  Target,
  ChevronDown,
  ChevronUp,
  Trash2,
  Database,
  ImageIcon,
  ArrowRight,
  BarChart2,
  Layers,
  LineChart as LineChartIcon,
  List,
  GitCompare,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, Cell,
  PieChart, Pie,
} from 'recharts'
import { format } from 'date-fns'

const TABS = [
  { id: 'overview', label: 'Overview', icon: Layers },
  { id: 'performance', label: 'Performance', icon: LineChartIcon },
  { id: 'runs', label: 'Runs', icon: List },
  { id: 'allocation', label: 'Allocation', icon: Target },
]

export default function Dashboard() {
  const [runs, runsLoading] = useRuns(30)
  const [queue] = useQueue()
  const [stockMeta] = useStockMeta()
  const [strategies] = useStrategies()
  const [counts, setCounts] = useState({ patterns: 0, strategies: 0 })
  const [expandedRun, setExpandedRun] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteMenuOpen, setDeleteMenuOpen] = useState(null)
  const [tab, setTab] = useState('overview')
  const latestRunId = runs.length > 0 ? runs[0].id : null
  const [latestCharts] = useRunCharts(latestRunId)
  const portfolioChart = latestCharts.find((c) => c.chartType === 'portfolio') ?? null

  useEffect(() => {
    const handler = () => setDeleteMenuOpen(null)
    document.addEventListener('click', handler)
    return () => document.removeEventListener('click', handler)
  }, [])

  const handleDeleteRunOnly = async (runId) => {
    if (!confirm('Delete this run entry? (Patterns discovered during this run are kept.)')) return
    try { await deleteDoc(doc(db, 'runs', runId)) } catch (err) { alert('Delete failed: ' + err.message) }
  }
  const handleDeleteRunFull = async (runId) => {
    if (!confirm('Delete this run entry AND all patterns discovered during it?')) return
    try {
      const patSnap = await getDocs(query(collection(db, 'patterns'), where('runId', '==', runId)))
      const ops = [deleteDoc(doc(db, 'runs', runId))]
      patSnap.forEach((d) => ops.push(deleteDoc(doc(db, 'patterns', d.id))))
      await Promise.all(ops)
    } catch (err) { alert('Delete failed: ' + err.message) }
  }
  const handleClearRunsOnly = async () => {
    if (!confirm(`Delete ALL ${runs.length} run entries?`)) return
    setDeleting(true)
    try {
      const snap = await getDocs(collection(db, 'runs'))
      await Promise.all(snap.docs.map((d) => deleteDoc(doc(db, 'runs', d.id))))
    } catch (err) { alert('Bulk delete failed: ' + err.message) }
    setDeleting(false)
  }
  const handleClearAllData = async () => {
    if (!confirm('Delete ALL run entries, ALL patterns, and ALL strategies? This cannot be undone.')) return
    setDeleting(true)
    try {
      const [runSnap, patSnap, stratSnap] = await Promise.all([
        getDocs(collection(db, 'runs')),
        getDocs(collection(db, 'patterns')),
        getDocs(collection(db, 'strategies')),
      ])
      const ops = []
      runSnap.forEach((d) => ops.push(deleteDoc(doc(db, 'runs', d.id))))
      patSnap.forEach((d) => ops.push(deleteDoc(doc(db, 'patterns', d.id))))
      stratSnap.forEach((d) => ops.push(deleteDoc(doc(db, 'strategies', d.id))))
      await Promise.all(ops)
    } catch (err) { alert('Bulk delete failed: ' + err.message) }
    setDeleting(false)
  }

  useEffect(() => {
    async function fetchCounts() {
      try {
        const patSnap = await getCountFromServer(
          query(collection(db, 'patterns'), where('supersededBy', '==', ''))
        )
        const stratSnap = await getCountFromServer(
          query(collection(db, 'strategies'), where('active', '==', true))
        )
        setCounts({ patterns: patSnap.data().count, strategies: stratSnap.data().count })
      } catch { /* index may not exist */ }
    }
    fetchCounts()
  }, [runs])

  const pendingCount = queue.filter((q) => q.status === 'queued').length
  const runningCount = queue.filter((q) => q.status === 'running').length
  const bestRun = runs.length
    ? runs.reduce((a, b) => ((a.alphaVsBuyHold || 0) > (b.alphaVsBuyHold || 0) ? a : b))
    : null
  const latestRun = runs[0] || null
  const avgReturn = runs.length ? runs.reduce((s, r) => s + (r.totalReturnPct || 0), 0) / runs.length : 0
  const avgAlpha = runs.length ? runs.reduce((s, r) => s + (r.alphaVsBuyHold || 0), 0) / runs.length : 0
  const avgSharpe = runs.length ? runs.reduce((s, r) => s + (r.sharpeRatio || 0), 0) / runs.length : 0
  const avgWinRate = runs.length ? runs.reduce((s, r) => s + (r.winRate || 0), 0) / runs.length : 0

  const chartData = useMemo(() =>
    [...runs].reverse().map((r) => ({
      date: r.timestamp ? format(new Date(r.timestamp), 'MMM d HH:mm') : '',
      alpha: +(r.alphaVsBuyHold || 0).toFixed(2),
      returnPct: +(r.totalReturnPct || 0).toFixed(2),
      sharpe: +(r.sharpeRatio || 0).toFixed(2),
      winRate: +(r.winRate || 0).toFixed(1),
    })),
  [runs])

  if (runsLoading) {
    return <div className="p-6 text-center text-dark-muted py-20">Loading data...</div>
  }

  return (
    <div className="p-6 space-y-5 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Dashboard</h1>
          <p className="text-dark-muted text-sm mt-0.5">Pipeline overview &amp; performance</p>
        </div>
        <div className="flex items-center gap-2">
          {runningCount > 0 && (
            <span className="badge-blue flex items-center gap-1">
              <Activity className="w-3 h-3 animate-pulse" /> Running
            </span>
          )}
          {pendingCount > 0 && <span className="badge-yellow">{pendingCount} queued</span>}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatsCard label="Total Runs" value={runs.length} icon={Zap} color="blue" />
        <StatsCard label="Active Patterns" value={counts.patterns} icon={BarChart3} color="purple" />
        <StatsCard label="Strategies" value={counts.strategies} icon={Brain} color="green" />
        <StatsCard label="Stocks Tracked" value={stockMeta.length} icon={Target} color="yellow" />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-dark-border">
        {TABS.map((t) => {
          const Icon = t.icon
          const active = tab === t.id
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors -mb-px ${
                active
                  ? 'border-accent-blue text-accent-blue'
                  : 'border-transparent text-dark-muted hover:text-dark-text'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {t.label}
            </button>
          )
        })}
      </div>

      {/* ─── TAB: Overview ──────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="space-y-5">
          {/* Average metrics */}
          {runs.length > 0 && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <MetricCard label="Avg Return" value={`${avgReturn.toFixed(2)}%`} positive={avgReturn >= 0} />
              <MetricCard label="Avg Alpha (B&H)" value={`${avgAlpha >= 0 ? '+' : ''}${avgAlpha.toFixed(2)}%`} positive={avgAlpha >= 0} />
              <MetricCard label="Avg Sharpe" value={avgSharpe.toFixed(2)} />
              <MetricCard label="Avg Win Rate" value={`${avgWinRate.toFixed(1)}%`} positive={avgWinRate >= 50} />
            </div>
          )}

          {/* Portfolio growth chart */}
          {portfolioChart && (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-accent-blue" />
                  <h3 className="text-sm font-medium">Portfolio Growth</h3>
                  <span className="text-xs text-dark-muted">Latest run</span>
                </div>
                <a href="/charts" className="text-xs text-accent-blue hover:underline">View all charts</a>
              </div>
              <img src={portfolioChart.imageData} alt="Portfolio Growth"
                className="w-full rounded-lg border border-dark-border object-contain" style={{ maxHeight: '380px' }} />
            </div>
          )}

          {/* Latest + Best run cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {latestRun && <RunCard run={latestRun} title="Latest Run" icon={Clock} />}
            {bestRun && bestRun !== latestRun && <RunCard run={bestRun} title="Best Run (by Alpha)" icon={TrendingUp} accent />}
          </div>

          {/* Top Strategies */}
          {strategies.length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-accent-purple" />
                  <h3 className="text-sm font-medium">Top Strategies</h3>
                  <span className="text-xs text-dark-muted">({strategies.length} total)</span>
                </div>
                <Link to="/strategies" className="flex items-center gap-1 text-xs text-accent-blue hover:underline">
                  View all <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
              <div className="space-y-1.5">
                {strategies
                  .slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
                  .slice(0, 5)
                  .map((s) => <StrategyRow key={s.id} s={s} />)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─── TAB: Performance ───────────────────────────────────────── */}
      {tab === 'performance' && (
        <div className="space-y-5">
          {chartData.length > 1 ? (
            <>
              <div className="card">
                <h3 className="text-sm font-medium text-dark-muted mb-4">Return &amp; Alpha Across Runs</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                    <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3' }}
                      labelStyle={{ color: '#8b949e' }}
                    />
                    <Legend wrapperStyle={{ color: '#8b949e', fontSize: 11 }} />
                    <Line type="monotone" dataKey="alpha" stroke="#3fb950" name="Alpha vs B&H %" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="returnPct" stroke="#58a6ff" name="Return %" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="winRate" stroke="#d29922" name="Win Rate %" dot={false} strokeWidth={1} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-dark-muted mb-4">Sharpe Ratio Trend</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                    <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3' }}
                    />
                    <Bar dataKey="sharpe" name="Sharpe Ratio" radius={[4, 4, 0, 0]}>
                      {chartData.map((e, i) => (
                        <Cell key={i} fill={e.sharpe >= 0 ? '#3fb950' : '#f85149'} fillOpacity={0.7} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="card text-center py-16 text-dark-muted text-sm">
              Run the pipeline at least twice to see performance trends.
            </div>
          )}

          {/* Latest run per-stock breakdown */}
          {latestRun?.perStockResults && Object.keys(latestRun.perStockResults).length > 0 && (
            <div className="card">
              <div className="flex items-center gap-2 mb-3">
                <BarChart2 className="w-4 h-4 text-accent-blue" />
                <h3 className="text-sm font-medium">Latest Run — Per-Stock Breakdown</h3>
              </div>
              <PerStockTable perStock={latestRun.perStockResults} />
            </div>
          )}
        </div>
      )}

      {/* ─── TAB: Runs ──────────────────────────────────────────────── */}
      {tab === 'runs' && (
        <div className="space-y-4">
          {/* Actions bar */}
          {runs.length > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-dark-muted">{runs.length} run{runs.length !== 1 ? 's' : ''} recorded</span>
              <div className="flex items-center gap-2">
                <button onClick={handleClearRunsOnly} disabled={deleting}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-dark-border text-dark-muted hover:text-dark-text hover:border-dark-text/40 transition-colors">
                  <Trash2 className="w-3.5 h-3.5" />{deleting ? 'Deleting...' : 'Clear Runs'}
                </button>
                <button onClick={handleClearAllData} disabled={deleting}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-accent-red/30 text-accent-red hover:bg-accent-red/10 transition-colors">
                  <Database className="w-3.5 h-3.5" />{deleting ? 'Deleting...' : 'Clear All Data'}
                </button>
              </div>
            </div>
          )}

          {/* Runs table */}
          {runs.length >= 2 && <ABComparisonSection runs={runs} />}

          {runs.length > 0 ? (
            <div className="card p-0 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-dark-muted text-xs uppercase border-b border-dark-border bg-dark-hover/30">
                      <th className="py-2.5 px-3 w-6"></th>
                      <th className="py-2.5 px-3">Time</th>
                      <th className="py-2.5 px-3">Stocks</th>
                      <th className="py-2.5 px-3 text-right">Return</th>
                      <th className="py-2.5 px-3 text-right">Alpha B&H</th>
                      <th className="py-2.5 px-3 text-right">Alpha S&P</th>
                      <th className="py-2.5 px-3 text-right">Win Rate</th>
                      <th className="py-2.5 px-3 text-right">Sharpe</th>
                      <th className="py-2.5 px-3 text-right">Trades</th>
                      <th className="py-2.5 px-3 w-8"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-border/50">
                    {runs.slice(0, 20).map((r) => (
                      <Fragment key={r.id}>
                        <tr className="hover:bg-dark-hover/50 cursor-pointer group"
                          onClick={() => setExpandedRun(expandedRun === r.id ? null : r.id)}>
                          <td className="py-2 px-3">
                            {r.perStockResults && Object.keys(r.perStockResults).length > 0 && (
                              expandedRun === r.id
                                ? <ChevronUp className="w-3.5 h-3.5 text-dark-muted" />
                                : <ChevronDown className="w-3.5 h-3.5 text-dark-muted" />
                            )}
                          </td>
                          <td className="py-2 px-3 text-dark-muted text-xs whitespace-nowrap">
                            {r.timestamp ? format(new Date(r.timestamp), 'MMM d HH:mm') : '—'}
                          </td>
                          <td className="py-2 px-3 text-xs max-w-[200px] truncate">{r.symbols?.join(', ') || '—'}</td>
                          <td className={`py-2 px-3 text-right font-mono ${(r.totalReturnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                            {(r.totalReturnPct || 0).toFixed(2)}%
                          </td>
                          <td className={`py-2 px-3 text-right font-mono ${(r.alphaVsBuyHold || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                            {(r.alphaVsBuyHold || 0).toFixed(2)}%
                          </td>
                          <td className={`py-2 px-3 text-right font-mono ${(r.alphaVsSP500 || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                            {(r.alphaVsSP500 || 0).toFixed(2)}%
                          </td>
                          <td className={`py-2 px-3 text-right font-mono ${(r.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                            {(r.winRate || 0).toFixed(1)}%
                          </td>
                          <td className="py-2 px-3 text-right font-mono">{(r.sharpeRatio || 0).toFixed(2)}</td>
                          <td className="py-2 px-3 text-right font-mono">{r.numTrades || 0}</td>
                          <td className="py-2 px-3 relative">
                            <button
                              onClick={(e) => { e.stopPropagation(); setDeleteMenuOpen(deleteMenuOpen === r.id ? null : r.id) }}
                              className="p-1 text-dark-muted hover:text-accent-red opacity-0 group-hover:opacity-100 transition-opacity">
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                            {deleteMenuOpen === r.id && (
                              <div className="absolute right-0 top-7 z-50 bg-dark-card border border-dark-border rounded-lg shadow-xl w-56 text-xs overflow-hidden"
                                onClick={(e) => e.stopPropagation()}>
                                <button onClick={() => { handleDeleteRunOnly(r.id); setDeleteMenuOpen(null) }}
                                  className="w-full text-left px-3 py-2.5 hover:bg-dark-hover flex items-center gap-2 border-b border-dark-border/50">
                                  <Trash2 className="w-3.5 h-3.5 text-dark-muted shrink-0" /><span>Delete run entry only</span>
                                </button>
                                <button onClick={() => { handleDeleteRunFull(r.id); setDeleteMenuOpen(null) }}
                                  className="w-full text-left px-3 py-2.5 hover:bg-dark-hover flex items-center gap-2 text-accent-red">
                                  <Database className="w-3.5 h-3.5 shrink-0" /><span>Delete run + its patterns</span>
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                        {expandedRun === r.id && r.perStockResults && (
                          <tr key={`${r.id}-detail`}>
                            <td colSpan={10} className="pb-3 pt-0 px-4">
                              <PerStockTable perStock={r.perStockResults} compact />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="card text-center py-16 text-dark-muted text-sm">
              No pipeline runs yet. Run the full pipeline to see results here.
            </div>
          )}
        </div>
      )}

      {/* ─── TAB: Allocation ────────────────────────────────────────── */}
      {tab === 'allocation' && (
        <div className="space-y-5">
          {latestRun?.perStockResults && Object.keys(latestRun.perStockResults).length > 0 ? (
            <AllocationChart perStock={latestRun.perStockResults} totalFund={latestRun.totalFund || 100000} />
          ) : (
            <div className="card text-center py-16 text-dark-muted text-sm">
              Run the pipeline to see fund allocation data.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Reusable sub-components ──────────────────────────────────────────

function MetricCard({ label, value, positive }) {
  const color = positive === undefined ? '' : positive ? 'text-accent-green' : 'text-accent-red'
  return (
    <div className="card text-center py-3">
      <div className="text-xs text-dark-muted mb-1">{label}</div>
      <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
    </div>
  )
}

function RunCard({ run, title, icon: Icon, accent }) {
  const titleColor = accent ? 'text-accent-green' : 'text-dark-muted'
  return (
    <div className="card space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className={`w-4 h-4 ${titleColor}`} />
        <span className={titleColor}>{title}</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div>
          <div className="text-xs text-dark-muted">Return</div>
          <div className={`text-lg font-bold ${(run.totalReturnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(run.totalReturnPct || 0).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-dark-muted">Alpha vs B&H</div>
          <div className={`text-lg font-bold ${(run.alphaVsBuyHold || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(run.alphaVsBuyHold || 0).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-dark-muted">Win Rate</div>
          <div className={`text-lg font-bold ${(run.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(run.winRate || 0).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-dark-muted">Sharpe</div>
          <div className="text-lg font-bold">{(run.sharpeRatio || 0).toFixed(2)}</div>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-dark-muted truncate max-w-[70%]">
          {run.numTrades || 0} trades &bull; {run.symbols?.join(', ')} &bull;{' '}
          {run.timestamp ? format(new Date(run.timestamp), 'PPp') : '—'}
        </div>
        <Link to="/trades" className="flex items-center gap-1 text-xs text-accent-blue hover:underline flex-shrink-0 ml-2">
          <BarChart2 className="w-3 h-3" /> Trades
        </Link>
      </div>
    </div>
  )
}

function StrategyRow({ s }) {
  const ret = s.backtestReturn || s.backtestReturnPct || 0
  const sharpe = s.backtestSharpe || 0
  const conf = s.confidence || 0
  return (
    <div className="flex items-center gap-3 p-2 rounded-lg bg-dark-hover/30 border border-dark-border hover:border-dark-border/80 transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-xs font-medium truncate">{s.name}</span>
          <span className="text-[10px] text-dark-muted bg-dark-hover px-1.5 py-0.5 rounded capitalize flex-shrink-0">
            {(s.strategyType || '').replace(/_/g, ' ')}
          </span>
        </div>
        {s.symbols?.length > 0 && (
          <div className="flex gap-1 flex-wrap">
            {s.symbols.slice(0, 4).map((sym) => (
              <span key={sym} className="text-[10px] font-mono text-accent-blue/70">{sym}</span>
            ))}
            {s.symbols.length > 4 && <span className="text-[10px] text-dark-muted">+{s.symbols.length - 4}</span>}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 flex-shrink-0 text-right">
        {ret !== 0 && (
          <div>
            <div className="text-[10px] text-dark-muted">Return</div>
            <div className={`text-xs font-mono font-bold ${ret >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
              {ret >= 0 ? '+' : ''}{(ret * (ret > 1 ? 1 : 100)).toFixed(1)}%
            </div>
          </div>
        )}
        {sharpe !== 0 && (
          <div>
            <div className="text-[10px] text-dark-muted">Sharpe</div>
            <div className="text-xs font-mono">{sharpe.toFixed(2)}</div>
          </div>
        )}
        <div>
          <div className="text-[10px] text-dark-muted">Conf</div>
          <div className="text-xs font-mono text-accent-blue">{(conf * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  )
}

function PerStockTable({ perStock, compact }) {
  const entries = Object.entries(perStock)
  return (
    <div className={`grid gap-1 ${compact ? 'bg-dark-hover/30 rounded-lg p-3' : ''}`}>
      <div className={`grid ${compact ? 'grid-cols-6' : 'grid-cols-7'} gap-2 text-xs text-dark-muted font-medium pb-1 border-b border-dark-border/50`}>
        <span>Stock</span>
        <span className="text-right">Return</span>
        <span className="text-right">P/L</span>
        <span className="text-right">Trades</span>
        <span className="text-right">Win Rate</span>
        <span className="text-right">Alloc</span>
        {!compact && <span className="text-right">Type</span>}
      </div>
      {entries.map(([sym, s]) => (
        <div key={sym} className={`grid ${compact ? 'grid-cols-6' : 'grid-cols-7'} gap-2 text-xs font-mono py-0.5`}>
          <span className="font-bold flex items-center gap-1">
            {sym}
            {compact && s.portfolioType === 'automatic' && <span className="text-[8px] text-accent-purple font-normal">auto</span>}
          </span>
          <span className={`text-right ${(s.returnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(s.returnPct || 0) >= 0 ? '+' : ''}{(s.returnPct || 0).toFixed(2)}%
          </span>
          <span className={`text-right ${(s.profit || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            ${(s.profit || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
          <span className="text-right">{s.trades || 0}</span>
          <span className={`text-right ${(s.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
            {(s.winRate || 0).toFixed(0)}%
          </span>
          <span className="text-right text-dark-muted">{(s.allocation || 0).toFixed(1)}%</span>
          {!compact && (
            <span className="text-right">
              {s.portfolioType === 'automatic'
                ? <span className="text-accent-purple text-[10px] bg-accent-purple/10 px-1.5 py-0.5 rounded">connected</span>
                : <span className="text-dark-muted text-[10px]">main</span>}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

// ─── A/B Pipeline Comparison ──────────────────────────────────────────
function ABComparisonSection({ runs }) {
  const [runAId, setRunAId] = useState('')
  const [runBId, setRunBId] = useState('')

  const runA = runs.find(r => r.id === runAId) || null
  const runB = runs.find(r => r.id === runBId) || null

  // Build per-symbol comparison from perStockResults (already on each run doc)
  const comparison = useMemo(() => {
    if (!runA?.perStockResults || !runB?.perStockResults) return []
    const allSyms = [...new Set([
      ...Object.keys(runA.perStockResults),
      ...Object.keys(runB.perStockResults),
    ])]
    return allSyms.map(sym => ({
      symbol: sym,
      a: runA.perStockResults[sym] || { returnPct: 0, profit: 0, trades: 0, winRate: 0 },
      b: runB.perStockResults[sym] || { returnPct: 0, profit: 0, trades: 0, winRate: 0 },
    }))
  }, [runA, runB])

  const totalPnlA = comparison.reduce((s, c) => s + (c.a.profit || 0), 0)
  const totalPnlB = comparison.reduce((s, c) => s + (c.b.profit || 0), 0)

  const runLabel = (r) => {
    const date = r.timestamp ? format(new Date(r.timestamp), 'MMM d HH:mm') : r.id.slice(0, 8)
    const ret = (r.totalReturnPct || 0).toFixed(1)
    const syms = r.symbols?.join(', ') || '—'
    return `${date}  •  ${ret >= 0 ? '+' : ''}${ret}%  •  ${syms}`
  }

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-1">
        <GitCompare className="w-4 h-4 text-accent-purple" />
        <h3 className="text-sm font-semibold">A/B Run Comparison</h3>
      </div>
      <p className="text-xs text-dark-muted mb-4">
        Select two pipeline runs to compare their per-stock backtest results side-by-side.
      </p>

      <div className="grid grid-cols-2 gap-4 mb-5">
        {[
          { label: 'Run A', value: runAId, set: setRunAId, exclude: runBId, accent: 'text-accent-blue' },
          { label: 'Run B', value: runBId, set: setRunBId, exclude: runAId, accent: 'text-accent-purple' },
        ].map(({ label, value, set, exclude, accent }) => (
          <div key={label}>
            <label className={`text-[10px] font-semibold uppercase tracking-wider ${accent}`}>{label}</label>
            <select
              value={value}
              onChange={e => set(e.target.value)}
              className="w-full mt-1 px-3 py-2 text-xs rounded-lg bg-dark-hover border border-dark-border text-dark-text focus:border-accent-blue/50 focus:outline-none"
            >
              <option value="">Select a run...</option>
              {runs.filter(r => r.id !== exclude).map(r => (
                <option key={r.id} value={r.id}>{runLabel(r)}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {/* Summary totals */}
      {runA && runB && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          {[
            { run: runA, label: 'Run A', pnl: totalPnlA, accent: 'border-accent-blue/40 bg-accent-blue/5' },
            { run: runB, label: 'Run B', pnl: totalPnlB, accent: 'border-accent-purple/40 bg-accent-purple/5' },
          ].map(({ run, label, pnl, accent }) => (
            <div key={label} className={`rounded-lg border p-3 ${accent}`}>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide mb-1.5">{label}</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <div className="text-dark-muted">Return</div>
                <div className={`font-mono font-bold text-right ${(run.totalReturnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {(run.totalReturnPct || 0) >= 0 ? '+' : ''}{(run.totalReturnPct || 0).toFixed(2)}%
                </div>
                <div className="text-dark-muted">P&amp;L</div>
                <div className={`font-mono text-right ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {pnl >= 0 ? '+' : ''}${Math.round(pnl).toLocaleString()}
                </div>
                <div className="text-dark-muted">Win Rate</div>
                <div className={`font-mono text-right ${(run.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {(run.winRate || 0).toFixed(1)}%
                </div>
                <div className="text-dark-muted">Sharpe</div>
                <div className="font-mono text-right">{(run.sharpeRatio || 0).toFixed(2)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Per-stock table */}
      {comparison.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-dark-muted border-b border-dark-border">
                <th className="text-left py-2 px-2">Symbol</th>
                <th className="text-right py-2 px-2 text-accent-blue">A Return</th>
                <th className="text-right py-2 px-2 text-accent-blue">A Win%</th>
                <th className="text-right py-2 px-2 text-accent-blue">A P&amp;L</th>
                <th className="text-right py-2 px-2 text-accent-purple">B Return</th>
                <th className="text-right py-2 px-2 text-accent-purple">B Win%</th>
                <th className="text-right py-2 px-2 text-accent-purple">B P&amp;L</th>
                <th className="text-right py-2 px-2">Winner</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map(({ symbol, a, b }) => {
                const winner = a.profit > b.profit ? 'A' : b.profit > a.profit ? 'B' : '—'
                const winnerColor = winner === 'A' ? 'text-accent-blue' : winner === 'B' ? 'text-accent-purple' : 'text-dark-muted'
                return (
                  <tr key={symbol} className="border-b border-dark-border/40 hover:bg-dark-hover/30">
                    <td className="py-1.5 px-2 font-mono font-bold">{symbol}</td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(a.returnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(a.returnPct || 0) >= 0 ? '+' : ''}{(a.returnPct || 0).toFixed(2)}%
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(a.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(a.winRate || 0).toFixed(0)}%
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(a.profit || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(a.profit || 0) >= 0 ? '+' : ''}${Math.round(a.profit || 0).toLocaleString()}
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(b.returnPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(b.returnPct || 0) >= 0 ? '+' : ''}{(b.returnPct || 0).toFixed(2)}%
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(b.winRate || 0) >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(b.winRate || 0).toFixed(0)}%
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono ${(b.profit || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {(b.profit || 0) >= 0 ? '+' : ''}${Math.round(b.profit || 0).toLocaleString()}
                    </td>
                    <td className={`py-1.5 px-2 text-right font-bold ${winnerColor}`}>{winner}</td>
                  </tr>
                )
              })}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-dark-border">
                <td className="py-2 px-2 font-bold text-xs">Total P&amp;L</td>
                <td colSpan={2} />
                <td className={`py-2 px-2 text-right font-mono font-bold ${totalPnlA >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {totalPnlA >= 0 ? '+' : ''}${Math.round(totalPnlA).toLocaleString()}
                </td>
                <td colSpan={2} />
                <td className={`py-2 px-2 text-right font-mono font-bold ${totalPnlB >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {totalPnlB >= 0 ? '+' : ''}${Math.round(totalPnlB).toLocaleString()}
                </td>
                <td className={`py-2 px-2 text-right font-bold ${totalPnlA > totalPnlB ? 'text-accent-blue' : totalPnlB > totalPnlA ? 'text-accent-purple' : 'text-dark-muted'}`}>
                  {totalPnlA > totalPnlB ? 'A' : totalPnlB > totalPnlA ? 'B' : '—'}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      )}

      {runAId && runBId && comparison.length === 0 && (
        <p className="text-xs text-dark-muted text-center py-6">
          No per-stock breakdown available for one or both runs.
        </p>
      )}
    </div>
  )
}

const ALLOC_COLORS = [
  '#58a6ff', '#3fb950', '#d29922', '#e078a0', '#bc8cff',
  '#ff7f0e', '#17becf', '#9467bd', '#e377c2', '#7f7f7f',
  '#8c564b', '#bcbd22', '#56d4dd', '#f0883e', '#db61a2',
]

function AllocationChart({ perStock, totalFund }) {
  const entries = Object.entries(perStock)
    .map(([sym, s]) => ({
      sym, alloc: s.allocation || 0, returnPct: s.returnPct || 0,
      profit: s.profit || 0, type: s.portfolioType || 'manual',
    }))
    .sort((a, b) => b.alloc - a.alloc)

  const pieData = entries.map((e, i) => ({
    name: e.sym, value: +(e.alloc).toFixed(1),
    fill: ALLOC_COLORS[i % ALLOC_COLORS.length], type: e.type,
  }))
  const barData = entries.map((e, i) => ({
    sym: e.sym, returnPct: +e.returnPct.toFixed(2), profit: +e.profit.toFixed(0),
    alloc: +e.alloc.toFixed(1), fill: ALLOC_COLORS[i % ALLOC_COLORS.length], type: e.type,
  }))

  const mainCount = entries.filter(e => e.type !== 'automatic').length
  const autoCount = entries.filter(e => e.type === 'automatic').length

  return (
    <>
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-accent-purple" />
            <h3 className="text-sm font-medium">Fund Allocation</h3>
          </div>
          <div className="flex items-center gap-3 text-xs text-dark-muted">
            {mainCount > 0 && <span>{mainCount} main</span>}
            {autoCount > 0 && <span className="text-accent-purple">+ {autoCount} connected</span>}
            <span className="font-mono">${totalFund.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <div className="text-xs text-dark-muted mb-2">Allocation by Stock</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={90}
                  paddingAngle={1.5} dataKey="value"
                  label={({ name, value }) => `${name} ${value}%`} labelLine={false}>
                  {pieData.map((e, i) => (
                    <Cell key={i} fill={e.fill} stroke={e.type === 'automatic' ? '#bc8cff' : 'transparent'}
                      strokeWidth={e.type === 'automatic' ? 2 : 0} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3', fontSize: 11 }}
                  formatter={(v, name) => [`${v}% ($${Math.round(totalFund * v / 100).toLocaleString()})`, name]}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-2 gap-1 mt-2">
              {pieData.map((e, i) => (
                <div key={i} className="flex items-center gap-1.5 text-[10px]">
                  <div className="w-2 h-2 rounded-sm flex-shrink-0"
                    style={{ background: e.fill, border: e.type === 'automatic' ? '1px solid #bc8cff' : 'none' }} />
                  <span className="font-mono font-medium">{e.name}</span>
                  <span className="text-dark-muted">{e.value}%</span>
                  {e.type === 'automatic' && <span className="text-accent-purple text-[8px]">auto</span>}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-xs text-dark-muted mb-2">Return per Stock (%)</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} layout="vertical" margin={{ left: 2, right: 10, top: 0, bottom: 0 }}>
                <CartesianGrid horizontal={false} stroke="#30363d" strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="sym" width={45} tick={{ fill: '#8b949e', fontSize: 10, fontFamily: 'monospace' }} />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3', fontSize: 11 }}
                  formatter={(v, name) => [name === 'returnPct' ? `${v}%` : `$${v.toLocaleString()}`, name === 'returnPct' ? 'Return' : 'Profit']}
                />
                <Bar dataKey="returnPct" radius={[0, 4, 4, 0]}>
                  {barData.map((e, i) => (
                    <Cell key={i} fill={e.returnPct >= 0 ? '#3fb950' : '#f85149'} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-2 mt-2">
              {barData.map((e, i) => (
                <div key={i} className="flex items-center gap-1 text-[10px] bg-dark-hover/40 rounded px-1.5 py-0.5">
                  <span className="font-mono font-medium">{e.sym}</span>
                  <span className={`font-mono ${e.profit >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {e.profit >= 0 ? '+' : ''}${Math.abs(e.profit).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
