import { useState, useMemo } from 'react'
import { useStrategies } from '../hooks/useFirestore'
import {
  Brain, TrendingUp, TrendingDown, ChevronDown, ChevronUp,
  Activity, Target, Zap, BarChart3, ArrowLeftRight,
  RefreshCw, Link2, DollarSign, Shield,
} from 'lucide-react'

const TYPE_META = {
  hedge:            { label: 'Hedge',           color: 'text-accent-yellow', bg: 'bg-accent-yellow/10 border-accent-yellow/20',  icon: Shield },
  pairs_trade:      { label: 'Pairs Trade',     color: 'text-accent-blue',   bg: 'bg-accent-blue/10   border-accent-blue/20',    icon: ArrowLeftRight },
  sector_rotation:  { label: 'Sector Rotation', color: 'text-accent-purple', bg: 'bg-accent-purple/10 border-accent-purple/20',  icon: RefreshCw },
  momentum:         { label: 'Momentum',        color: 'text-accent-green',  bg: 'bg-accent-green/10  border-accent-green/20',   icon: TrendingUp },
  mean_reversion:   { label: 'Mean Reversion',  color: 'text-accent-red',    bg: 'bg-accent-red/10    border-accent-red/20',     icon: Activity },
  supply_chain:     { label: 'Supply Chain',    color: 'text-accent-blue',   bg: 'bg-accent-blue/10   border-accent-blue/20',    icon: Link2 },
  earnings_spread:  { label: 'Earnings Spread', color: 'text-dark-muted',    bg: 'bg-dark-hover        border-dark-border',       icon: DollarSign },
}

const SORT_OPTIONS = [
  { value: 'confidence', label: 'Confidence' },
  { value: 'backtestReturn', label: 'Return' },
  { value: 'backtestSharpe', label: 'Sharpe' },
  { value: 'name', label: 'Name' },
]

const TYPE_FILTERS = ['all', ...Object.keys(TYPE_META)]

function ConfidenceBar({ value }) {
  const pct = Math.round((value || 0) * 100)
  const color = pct >= 70 ? '#3fb950' : pct >= 45 ? '#d29922' : '#f85149'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-dark-hover rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono w-8 text-right" style={{ color }}>{pct}%</span>
    </div>
  )
}

function MetricPill({ label, value, colorClass }) {
  return (
    <div className="flex flex-col items-center bg-dark-hover/40 rounded-lg px-2.5 py-1.5">
      <span className="text-[10px] text-dark-muted mb-0.5">{label}</span>
      <span className={`text-xs font-mono font-bold ${colorClass || 'text-dark-text'}`}>{value}</span>
    </div>
  )
}

function StrategyCard({ s, expanded, onToggle }) {
  const meta = TYPE_META[s.strategyType] || { label: s.strategyType || '—', color: 'text-dark-muted', bg: 'bg-dark-hover border-dark-border', icon: Brain }
  const Icon = meta.icon
  const ret = s.backtestReturn ?? s.backtestReturnPct ?? null
  const sharpe = s.backtestSharpe ?? null
  const conf = s.confidence || 0

  // Normalize return — sometimes stored as decimal (0.05 = 5%) vs percentage (5.0)
  const retPct = ret !== null ? (Math.abs(ret) <= 2 ? ret * 100 : ret) : null
  const retColor = retPct !== null ? (retPct >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-dark-muted'

  return (
    <div className={`card transition-all ${expanded ? 'border-accent-blue/30' : ''}`}>
      {/* Summary row */}
      <div className="flex items-start gap-3 cursor-pointer" onClick={onToggle}>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${meta.bg} border`}>
          <Icon className={`w-4 h-4 ${meta.color}`} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-sm font-semibold">{s.name}</span>
            <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${meta.bg} ${meta.color}`}>
              {meta.label}
            </span>
            {s.source && (
              <span className="text-[10px] text-dark-muted bg-dark-hover px-1.5 py-0.5 rounded border border-dark-border">
                {s.source}
              </span>
            )}
          </div>

          <p className="text-xs text-dark-muted leading-relaxed line-clamp-2">{s.description}</p>

          {s.symbols?.length > 0 && (
            <div className="flex gap-1.5 flex-wrap mt-1.5">
              {s.symbols.map((sym) => (
                <span key={sym} className="text-[10px] font-mono text-accent-blue/80 bg-accent-blue/5 border border-accent-blue/15 px-1.5 py-0.5 rounded">
                  {sym}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {retPct !== null && (
            <MetricPill
              label="Return"
              value={`${retPct >= 0 ? '+' : ''}${retPct.toFixed(1)}%`}
              colorClass={retColor}
            />
          )}
          {sharpe !== null && (
            <MetricPill label="Sharpe" value={sharpe.toFixed(2)} />
          )}
          {expanded ? <ChevronUp className="w-4 h-4 text-dark-muted" /> : <ChevronDown className="w-4 h-4 text-dark-muted" />}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-dark-border space-y-3">
          {/* Full description */}
          {s.description && (
            <p className="text-xs text-dark-text/80 leading-relaxed">{s.description}</p>
          )}

          {/* Metrics grid */}
          <div className="flex flex-wrap gap-2">
            {retPct !== null && (
              <MetricPill label="Backtest Return" value={`${retPct >= 0 ? '+' : ''}${retPct.toFixed(2)}%`} colorClass={retColor} />
            )}
            {sharpe !== null && (
              <MetricPill label="Sharpe Ratio" value={sharpe.toFixed(3)} colorClass={sharpe >= 1 ? 'text-accent-green' : sharpe >= 0.5 ? 'text-dark-text' : 'text-accent-red'} />
            )}
            {s.confidence > 0 && (
              <MetricPill label="Confidence" value={`${(s.confidence * 100).toFixed(0)}%`} colorClass="text-accent-blue" />
            )}
            {s.backtestWinRate > 0 && (
              <MetricPill label="Win Rate" value={`${(s.backtestWinRate * 100).toFixed(1)}%`} colorClass={s.backtestWinRate >= 0.5 ? 'text-accent-green' : 'text-accent-red'} />
            )}
            {s.numTrades > 0 && (
              <MetricPill label="Trades" value={s.numTrades} />
            )}
            {s.holdingPeriod && (
              <MetricPill label="Hold Period" value={s.holdingPeriod} />
            )}
          </div>

          {/* Confidence bar */}
          <div>
            <div className="text-[10px] text-dark-muted mb-1">Confidence</div>
            <ConfidenceBar value={s.confidence} />
          </div>

          {/* Entry/exit rules */}
          {(s.entryRule || s.exitRule || s.rules) && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {s.entryRule && (
                <div className="p-2.5 rounded-lg bg-accent-green/5 border border-accent-green/15">
                  <div className="text-[10px] font-medium text-accent-green mb-1">Entry Rule</div>
                  <div className="text-xs text-dark-text/80">{s.entryRule}</div>
                </div>
              )}
              {s.exitRule && (
                <div className="p-2.5 rounded-lg bg-accent-red/5 border border-accent-red/15">
                  <div className="text-[10px] font-medium text-accent-red mb-1">Exit Rule</div>
                  <div className="text-xs text-dark-text/80">{s.exitRule}</div>
                </div>
              )}
            </div>
          )}

          {/* Conditions list */}
          {s.conditions?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-muted mb-1.5">Conditions</div>
              <ul className="space-y-1">
                {s.conditions.map((c, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-dark-text/80">
                    <span className="text-accent-blue mt-0.5 flex-shrink-0">•</span>
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Metadata footer */}
          <div className="flex flex-wrap gap-3 text-[10px] text-dark-muted pt-1 border-t border-dark-border/50">
            {s.runId && <span>Run: <span className="font-mono">{s.runId.slice(0, 8)}</span></span>}
            {s.createdAt && <span>Created: {new Date(s.createdAt).toLocaleDateString()}</span>}
            {s.active !== undefined && (
              <span className={s.active ? 'text-accent-green' : 'text-accent-red'}>
                {s.active ? 'Active' : 'Inactive'}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Strategies() {
  const [strategies, loading] = useStrategies()
  const [sortBy, setSortBy] = useState('confidence')
  const [typeFilter, setTypeFilter] = useState('all')
  const [expandedId, setExpandedId] = useState(null)

  const filtered = useMemo(() => {
    let arr = strategies
    if (typeFilter !== 'all') arr = arr.filter((s) => s.strategyType === typeFilter)
    arr = [...arr].sort((a, b) => {
      if (sortBy === 'name') return (a.name || '').localeCompare(b.name || '')
      if (sortBy === 'backtestReturn') {
        const ra = a.backtestReturn ?? a.backtestReturnPct ?? 0
        const rb = b.backtestReturn ?? b.backtestReturnPct ?? 0
        return rb - ra
      }
      return (b[sortBy] || 0) - (a[sortBy] || 0)
    })
    return arr
  }, [strategies, sortBy, typeFilter])

  if (loading) {
    return (
      <div className="p-6 space-y-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-20 rounded-xl bg-dark-hover/20 border border-dark-border animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <Brain className="w-5 h-5 text-dark-muted" />
        <div>
          <h1 className="text-xl font-bold">Strategies</h1>
          <p className="text-dark-muted text-sm mt-0.5">
            AI-generated &amp; statistically-discovered cross-stock strategies — {strategies.length} active
          </p>
        </div>
      </div>

      {strategies.length === 0 ? (
        <div className="card text-center text-dark-muted py-12 space-y-2">
          <Brain className="w-10 h-10 mx-auto opacity-20" />
          <p className="text-sm font-medium">No strategies yet.</p>
          <p className="text-xs">Run the full pipeline to generate cross-stock strategies.</p>
        </div>
      ) : (
        <>
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Sort */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-dark-muted">Sort:</span>
              {SORT_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  onClick={() => setSortBy(o.value)}
                  className={`px-2.5 py-1 rounded-lg text-xs border transition-all ${
                    sortBy === o.value
                      ? 'bg-accent-blue/15 text-accent-blue border-accent-blue/30'
                      : 'text-dark-muted border-transparent hover:border-dark-border'
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>

            {/* Type filter */}
            <div className="flex flex-wrap items-center gap-1.5 border-l border-dark-border pl-3">
              {TYPE_FILTERS.slice(0, 5).map((t) => {
                const m = TYPE_META[t]
                return (
                  <button
                    key={t}
                    onClick={() => setTypeFilter(t)}
                    className={`px-2.5 py-1 rounded-lg text-xs border capitalize transition-all ${
                      typeFilter === t
                        ? 'bg-accent-blue/15 text-accent-blue border-accent-blue/30'
                        : 'text-dark-muted border-transparent hover:border-dark-border'
                    }`}
                  >
                    {m?.label || 'All'}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Strategy list */}
          <div className="space-y-2">
            {filtered.map((s) => (
              <StrategyCard
                key={s.id}
                s={s}
                expanded={expandedId === s.id}
                onToggle={() => setExpandedId(expandedId === s.id ? null : s.id)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="text-center text-dark-muted text-sm py-8">
                No strategies match the selected filter.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
