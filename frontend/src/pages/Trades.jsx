import { useState, useEffect, useMemo } from 'react'
import { useRuns, useBacktestTrades } from '../hooks/useFirestore'
import { format } from 'date-fns'
import {
  TrendingUp, TrendingDown, Minus, Activity, Clock, Filter,
  ChevronDown, ChevronUp, Loader2, BarChart2, CheckCircle, XCircle,
  DollarSign, Target,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const INTERVAL_COLORS = {
  '1d': '#58a6ff',
  '1h': '#3fb950',
  '15m': '#d29922',
  '5m': '#e078a0',
  '1wk': '#bc8cff',
}
const STOCK_PALETTE = [
  '#58a6ff', '#3fb950', '#d29922', '#e078a0', '#bc8cff',
  '#ff7f0e', '#17becf', '#9467bd', '#e377c2', '#7f7f7f',
]

// ── Helpers ──────────────────────────────────────────────────────────

function pct(v) {
  const n = v || 0
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`
}
function dollar(v) {
  const n = v || 0
  return `${n >= 0 ? '+' : '-'}$${Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
}

// Build an equity-curve series from a sorted trade list + initial balance
function buildEquityCurve(trades, initialBalance) {
  if (!trades.length) return []
  const sorted = [...trades].sort((a, b) =>
    (a.exitTs || a.ts || '').localeCompare(b.exitTs || b.ts || ''))
  const pts = [{ idx: 0, bal: initialBalance, label: 'Start' }]
  let bal = initialBalance
  sorted.forEach((t, i) => {
    bal += (t.pnl || 0)
    const label = (t.exitTs || t.ts || '').slice(0, 10) || `#${i + 1}`
    pts.push({ idx: i + 1, bal: +bal.toFixed(2), label })
  })
  return pts
}

// ── Sub-components ───────────────────────────────────────────────────

function StatChip({ label, value, color }) {
  const cls = color === 'green' ? 'text-accent-green'
            : color === 'red'   ? 'text-accent-red'
            :                     'text-dark-text'
  return (
    <div className="flex flex-col items-center bg-dark-hover/40 rounded-lg px-3 py-2">
      <span className="text-[10px] text-dark-muted mb-0.5">{label}</span>
      <span className={`text-sm font-mono font-bold ${cls}`}>{value}</span>
    </div>
  )
}

const EXIT_LABELS = {
  signal_reversal: 'Signal',
  stop_loss: 'Stop-Loss',
  max_hold: 'Max Hold',
  end_of_backtest: 'End',
}

function SignalPill({ label, sig }) {
  if (!sig) return null
  const color = sig.signal === 'BUY' ? 'text-accent-green' : sig.signal === 'SELL' ? 'text-accent-red' : 'text-dark-muted'
  return (
    <span className={`inline-block mr-1 text-[10px] ${color}`}>
      {label}:{sig.signal?.charAt(0)}{sig.confidence > 0 ? ` ${(sig.confidence * 100).toFixed(0)}%` : ''}
    </span>
  )
}

function TradeRow({ trade, index }) {
  const ok = trade.ok
  const conf = trade.conf || 0
  const deployed = trade.alloc || 0
  const signals = trade.signals || {}
  const exitReason = trade.exitReason || ''
  return (
    <>
      <tr className={`text-xs font-mono border-b border-dark-border/40 hover:bg-dark-hover/30 ${ok ? '' : 'opacity-80'}`}>
        <td className="py-1.5 px-2 text-dark-muted">{index + 1}</td>
        <td className="py-1.5 px-2 text-dark-muted whitespace-nowrap">
          {trade.ts ? trade.ts.slice(0, 10) : '—'}
        </td>
        <td className="py-1.5 px-2 text-center">
          {ok
            ? <CheckCircle className="w-3.5 h-3.5 text-accent-green inline" />
            : <XCircle    className="w-3.5 h-3.5 text-accent-red   inline" />}
        </td>
        <td className="py-1.5 px-2 text-right text-dark-muted">
          {conf > 0 ? `${(conf * 100).toFixed(0)}%` : '—'}
        </td>
        <td className={`py-1.5 px-2 text-right ${ok ? 'text-accent-green' : 'text-accent-red'}`}>
          {pct(trade.ret)}
        </td>
        <td className="py-1.5 px-2 text-right text-dark-muted">${(trade.entry || 0).toFixed(2)}</td>
        <td className="py-1.5 px-2 text-right text-dark-muted">${(trade.exit  || 0).toFixed(2)}</td>
        <td className="py-1.5 px-2 text-right text-dark-muted">
          ${deployed > 0 ? deployed.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
        </td>
        <td className={`py-1.5 px-2 text-right font-bold ${(trade.pnl || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
          {dollar(trade.pnl)}
        </td>
        <td className="py-1.5 px-2 text-right text-dark-text">${(trade.bal || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
        {exitReason && (
          <td className="py-1.5 px-2 text-right text-[10px] text-dark-muted">
            {EXIT_LABELS[exitReason] || exitReason}
          </td>
        )}
      </tr>
      {Object.keys(signals).length > 0 && (
        <tr className="border-b border-dark-border/20">
          <td colSpan={11} className="py-0.5 px-4">
            <SignalPill label="ML" sig={signals.ml} />
            <SignalPill label="Pat" sig={signals.pattern} />
            <SignalPill label="Port" sig={signals.portfolio} />
            <SignalPill label="Sent" sig={signals.sentiment} />
          </td>
        </tr>
      )}
    </>
  )
}

function StockBlock({ doc, colorIdx, expanded, onToggle }) {
  const trades = doc.trades || []
  const stats  = doc.stats  || {}
  const initBal = stats.initialBalance || 0
  const curve  = useMemo(() => buildEquityCurve(trades, initBal), [trades, initBal])
  const color  = STOCK_PALETTE[colorIdx % STOCK_PALETTE.length]

  // Derive P&L and Return from the equity curve so they ALWAYS match the graph.
  // This prevents mismatches from stale Firestore stats or corrupt trade pnl values.
  const { totalPnL, compoundReturn } = useMemo(() => {
    if (!curve.length) return { totalPnL: 0, compoundReturn: 0 }
    const start = curve[0].bal
    const end   = curve[curve.length - 1].bal
    const pnl   = end - start
    const ret   = start > 0 ? (pnl / start) * 100 : 0
    return { totalPnL: pnl, compoundReturn: ret }
  }, [curve])

  const winColor = (stats.winRate || 0) >= 50 ? 'green' : 'red'
  const retColor = compoundReturn >= 0 ? 'green' : 'red'

  return (
    <div className="card">
      {/* Header row */}
      <div
        className="flex items-center gap-4 cursor-pointer"
        onClick={onToggle}
      >
        <div className="w-2 h-8 rounded-full flex-shrink-0" style={{ background: color }} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono font-bold text-sm">{doc.symbol}</span>
            <span className="text-[10px] text-dark-muted bg-dark-hover/60 px-1.5 py-0.5 rounded border border-dark-border">
              {doc.interval}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <StatChip label="Trades"   value={stats.totalTrades || 0} />
          <StatChip label="Win Rate" value={`${(stats.winRate || 0).toFixed(1)}%`} color={winColor} />
          <StatChip label="Alloc"    value={initBal > 0 ? `$${(initBal / 1000).toFixed(1)}k` : '—'} />
          <StatChip label="Return"   value={pct(compoundReturn)} color={retColor} />
          <StatChip label="P&L"      value={dollar(totalPnL)} color={retColor} />
          {expanded
            ? <ChevronUp className="w-4 h-4 text-dark-muted" />
            : <ChevronDown className="w-4 h-4 text-dark-muted" />}
        </div>
      </div>

      {/* Expanded: equity curve + trade table */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-dark-border space-y-4">
          {/* Equity curve */}
          {curve.length > 1 && (
            <div>
              <div className="text-xs text-dark-muted font-medium mb-2">Equity Curve</div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#8b949e', fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: '#8b949e', fontSize: 10 }}
                    tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                    width={52}
                  />
                  <Tooltip
                    contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3', fontSize: 11 }}
                    formatter={(v) => [`$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Balance']}
                    labelStyle={{ color: '#8b949e' }}
                  />
                  <ReferenceLine y={stats.initialBalance || 0} stroke="#58a6ff" strokeDasharray="4 4" strokeWidth={1} opacity={0.5} />
                  <Line
                    type="stepAfter"
                    dataKey="bal"
                    stroke={color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Trade table */}
          <div className="overflow-x-auto rounded-lg border border-dark-border">
            <table className="w-full">
              <thead className="bg-dark-hover/40">
                <tr className="text-left text-[10px] text-dark-muted uppercase">
                  <th className="py-1.5 px-2">#</th>
                  <th className="py-1.5 px-2">Date</th>
                  <th className="py-1.5 px-2 text-center">Result</th>
                  <th className="py-1.5 px-2 text-right">Conf</th>
                  <th className="py-1.5 px-2 text-right">Return</th>
                  <th className="py-1.5 px-2 text-right">Entry</th>
                  <th className="py-1.5 px-2 text-right">Exit</th>
                  <th className="py-1.5 px-2 text-right">Deployed</th>
                  <th className="py-1.5 px-2 text-right">P&amp;L</th>
                  <th className="py-1.5 px-2 text-right">Stock Bal</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => <TradeRow key={i} trade={t} index={i} />)}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Portfolio-level combined equity curve ─────────────────────────────

function PortfolioEquityCurve({ tradeDocs }) {
  const chartData = useMemo(() => {
    const all = []
    tradeDocs.forEach((doc) => {
      ;(doc.trades || []).forEach((t) => {
        const exitTime = t.exitTs || t.ts || ''
        all.push({ ts: exitTime, pnl: t.pnl || 0 })
      })
    })
    all.sort((a, b) => a.ts.localeCompare(b.ts))

    const totalInit = tradeDocs.reduce((s, d) => s + (d.stats?.initialBalance || 0), 0)
    const pts = [{ label: 'Start', value: totalInit }]
    let cum = totalInit
    all.forEach((t, i) => {
      cum += t.pnl
      pts.push({ label: t.ts.slice(0, 10) || `#${i + 1}`, value: +cum.toFixed(2) })
    })
    return pts
  }, [tradeDocs])

  if (chartData.length < 2) return null

  const init = chartData[0].value
  const final = chartData[chartData.length - 1].value
  const pnl = final - init
  const retPct = init > 0 ? (pnl / init * 100) : 0

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-accent-blue" />
          <h3 className="text-sm font-semibold">Combined Portfolio Equity Curve</h3>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className={`font-mono font-bold ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {dollar(pnl)}
          </span>
          <span className={`font-mono ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {pct(retPct)}
          </span>
          <span className="text-dark-muted">{chartData.length - 1} trades total</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
          <XAxis
            dataKey="label"
            tick={{ fill: '#8b949e', fontSize: 10 }}
            interval={Math.max(1, Math.floor(chartData.length / 10))}
          />
          <YAxis
            tick={{ fill: '#8b949e', fontSize: 10 }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            width={55}
          />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, color: '#e6edf3', fontSize: 11 }}
            formatter={(v) => [`$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Portfolio Value']}
            labelStyle={{ color: '#8b949e' }}
          />
          <ReferenceLine y={init} stroke="#58a6ff" strokeDasharray="5 5" strokeWidth={1} opacity={0.5} label={{ value: 'Initial', fill: '#8b949e', fontSize: 9 }} />
          <Line type="monotone" dataKey="value" stroke="#3fb950" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────

export default function TradesPage() {
  const [runs, runsLoading] = useRuns(30)
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [tradeDocs, tradesLoading] = useBacktestTrades(selectedRunId)
  const [filterSym, setFilterSym] = useState('')
  const [filterType, setFilterType] = useState('all') // 'all' | 'main' | 'connected'
  const [expandedId, setExpandedId] = useState(null)

  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) setSelectedRunId(runs[0].id)
  }, [runs]) // eslint-disable-line react-hooks/exhaustive-deps

  const selectedRun = runs.find((r) => r.id === selectedRunId) ?? null
  const selectedRunFund = selectedRun?.totalFund ?? null

  const symbols = useMemo(() => {
    const set = new Set(tradeDocs.map((d) => d.symbol).filter(Boolean))
    return [...set].sort()
  }, [tradeDocs])

  const filtered = useMemo(() => {
    let docs = tradeDocs.filter((d) => !d.isWalkForward)
    if (filterSym) docs = docs.filter((d) => d.symbol === filterSym)
    if (filterType === 'main')      docs = docs.filter((d) => !d.portfolioType || d.portfolioType === 'manual')
    if (filterType === 'connected') docs = docs.filter((d) => d.portfolioType === 'automatic')
    return docs.sort((a, b) => `${a.symbol}${a.interval}`.localeCompare(`${b.symbol}${b.interval}`))
  }, [tradeDocs, filterSym, filterType])

  // Aggregate stats across filtered docs
  const totals = useMemo(() => {
    let trades = 0, wins = 0, pnl = 0
    filtered.forEach((d) => {
      trades += d.stats?.totalTrades || 0
      wins   += Math.round((d.stats?.totalTrades || 0) * (d.stats?.winRate || 0) / 100)
      pnl    += (d.stats?.finalBalance || 0) - (d.stats?.initialBalance || 0)
    })
    return { trades, wins, winRate: trades > 0 ? wins / trades * 100 : 0, pnl }
  }, [filtered])

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <BarChart2 className="w-5 h-5 text-dark-muted" />
        <div>
          <h1 className="text-xl font-bold">Trade History</h1>
          <p className="text-dark-muted text-sm mt-0.5">
            Backtest trade evolution — per-stock equity curves and individual trades
          </p>
        </div>
      </div>

      {/* Run selector */}
      {runsLoading ? (
        <div className="flex items-center gap-2 text-dark-muted text-sm">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading runs…
        </div>
      ) : runs.length === 0 ? (
        <div className="card text-center text-dark-muted py-10 text-sm">
          No pipeline runs yet. Run the full pipeline to generate backtest trade data.
        </div>
      ) : (
        <div className="card space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-dark-muted" />
            <span className="text-sm font-medium text-dark-muted">Select Run</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {runs.slice(0, 10).map((r) => {
              const ret = r.totalReturnPct || 0
              const sel = r.id === selectedRunId
              const fund = r.totalFund > 0 ? `$${(r.totalFund / 1000).toFixed(0)}k` : null
              return (
                <button
                  key={r.id}
                  onClick={() => { setSelectedRunId(r.id); setFilterSym(''); setExpandedId(null) }}
                  className={`px-3 py-2 rounded-lg text-left text-xs border transition-all ${
                    sel ? 'bg-accent-blue/15 border-accent-blue/40' : 'bg-dark-hover/50 border-dark-border hover:border-dark-text/30'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`font-mono font-bold ${sel ? 'text-accent-blue' : 'text-dark-text'}`}>
                      {r.symbols?.slice(0, 4).join(', ')}{(r.symbols?.length || 0) > 4 ? '…' : ''}
                    </span>
                    <span className={`font-mono ${ret >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-dark-muted">
                    <span>{r.timestamp ? format(new Date(r.timestamp), 'MMM d, HH:mm') : '—'}</span>
                    {fund && <span className="font-mono text-dark-muted/70">· {fund} fund</span>}
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {tradesLoading && selectedRunId && (
        <div className="flex items-center justify-center py-16 gap-3 text-dark-muted">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading trade data…</span>
        </div>
      )}

      {!tradesLoading && selectedRunId && tradeDocs.length === 0 && (
        <div className="card text-center py-12 text-dark-muted space-y-2">
          <BarChart2 className="w-8 h-8 mx-auto opacity-20" />
          <p className="text-sm font-medium">No trade data for this run</p>
          <p className="text-xs max-w-xs mx-auto">
            Trade history is generated by pipeline runs executed after the latest backend update.
            Re-run the pipeline to produce individual trade records.
          </p>
        </div>
      )}

      {!tradesLoading && tradeDocs.length > 0 && (
        <>
          {/* Run fund banner */}
          {selectedRunFund != null && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-hover/40 border border-dark-border text-xs text-dark-muted">
              <DollarSign className="w-3.5 h-3.5 flex-shrink-0" />
              <span>
                This run used a total fund of{' '}
                <span className="font-mono font-bold text-dark-text">
                  ${selectedRunFund.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>.
                {' '}Alloc, P&amp;L, and equity values below are all relative to this amount.
                {selectedRunFund !== (selectedRun?.totalFund) && null}
              </span>
            </div>
          )}

          {/* Portfolio equity curve */}
          <PortfolioEquityCurve tradeDocs={filtered} />

          {/* Aggregate stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatChip label="Total Trades"  value={totals.trades} />
            <StatChip label="Win Rate"      value={`${totals.winRate.toFixed(1)}%`} color={totals.winRate >= 50 ? 'green' : 'red'} />
            <StatChip label="Total P&L"     value={dollar(totals.pnl)} color={totals.pnl >= 0 ? 'green' : 'red'} />
            <StatChip label="Pairs shown"   value={`${filtered.length} stock/interval`} />
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-dark-muted" />
              <span className="text-xs text-dark-muted">Filter:</span>
            </div>

            {/* Type filter */}
            {['all', 'main', 'connected'].map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`px-3 py-1 rounded-lg text-xs border transition-all ${
                  filterType === t
                    ? 'bg-accent-blue/15 text-accent-blue border-accent-blue/30'
                    : 'bg-dark-hover text-dark-muted border-transparent hover:border-dark-border'
                }`}
              >
                {t === 'all' ? 'All stocks' : t === 'main' ? 'Main portfolio' : 'Connected stocks'}
              </button>
            ))}

            {/* Symbol filter */}
            {symbols.length > 1 && (
              <div className="flex flex-wrap gap-1.5 border-l border-dark-border pl-3 ml-1">
                <button
                  onClick={() => setFilterSym('')}
                  className={`px-2 py-0.5 rounded text-xs border transition-all ${
                    !filterSym ? 'bg-accent-blue/15 text-accent-blue border-accent-blue/30' : 'text-dark-muted border-transparent hover:border-dark-border'
                  }`}
                >All</button>
                {symbols.map((s) => (
                  <button
                    key={s}
                    onClick={() => setFilterSym(s)}
                    className={`px-2 py-0.5 rounded text-xs font-mono border transition-all ${
                      filterSym === s ? 'bg-accent-blue/15 text-accent-blue border-accent-blue/30' : 'text-dark-muted border-transparent hover:border-dark-border'
                    }`}
                  >{s}</button>
                ))}
              </div>
            )}
          </div>

          {/* Per-stock blocks */}
          <div className="space-y-3">
            {filtered.map((doc, i) => (
              <StockBlock
                key={doc.id}
                doc={doc}
                colorIdx={i}
                expanded={expandedId === doc.id}
                onToggle={() => setExpandedId(expandedId === doc.id ? null : doc.id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
