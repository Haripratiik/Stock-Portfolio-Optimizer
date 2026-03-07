import { useState, useMemo } from 'react'
import { usePatterns, useAllPatterns } from '../hooks/useFirestore'
import PatternCandlestick from '../components/PatternCandlestick'
import { BarChart3, ChevronDown, ChevronUp, X, TrendingUp, TrendingDown, Minus } from 'lucide-react'

const CANDLE_ICONS = { BULLISH: TrendingUp, BEARISH: TrendingDown, DOJI: Minus }
const CANDLE_COLORS = { BULLISH: 'text-accent-green', BEARISH: 'text-accent-red', DOJI: 'text-dark-muted' }

export default function Patterns() {
  const [selectedSym, setSelectedSym] = useState('')
  const [sortField, setSortField] = useState('mcCompositeScore')
  const [sortAsc, setSortAsc] = useState(false)
  const [expandedId, setExpandedId] = useState(null)

  // Load ALL patterns (unfiltered) for building the symbol list and counts.
  // A second filtered set is used for the actual table display.
  const [allPatternsRaw, rawLoading] = useAllPatterns(null, 500)
  const [filteredIndexed, filteredIndexedLoading] = usePatterns(selectedSym || null, 200)
  const [filteredAll, filteredAllLoading] = useAllPatterns(selectedSym || null, 200)

  // Active patterns = not superseded
  const activeAll = useMemo(
    () => allPatternsRaw.filter((p) => !p.supersededBy || p.supersededBy === ''),
    [allPatternsRaw],
  )

  // Displayed patterns: prefer indexed query (needs composite index), fall back to client-side
  const patterns = useMemo(() => {
    if (filteredIndexed.length > 0) return filteredIndexed
    const base = selectedSym
      ? activeAll.filter((p) => p.symbol === selectedSym)
      : activeAll
    return [...base].sort((a, b) => (b.mcCompositeScore || 0) - (a.mcCompositeScore || 0))
  }, [filteredIndexed, activeAll, selectedSym])

  const loading = rawLoading && filteredIndexedLoading && filteredAllLoading

  // Symbols always derived from ALL patterns so filter buttons never disappear
  const symbols = useMemo(() => {
    const set = new Set(activeAll.map((p) => p.symbol).filter(Boolean))
    return [...set].sort()
  }, [activeAll])

  const sorted = useMemo(() => {
    const arr = [...patterns]
    arr.sort((a, b) => {
      const va = a[sortField] || 0
      const vb = b[sortField] || 0
      return sortAsc ? va - vb : vb - va
    })
    return arr
  }, [patterns, sortField, sortAsc])

  const handleSort = (field) => {
    if (sortField === field) setSortAsc((p) => !p)
    else { setSortField(field); setSortAsc(false) }
  }

  const SortIcon = ({ field }) => {
    if (sortField !== field) return null
    return sortAsc ? <ChevronUp className="w-3 h-3 inline" /> : <ChevronDown className="w-3 h-3 inline" />
  }

  const parseGenes = (p) => {
    if (!p.genesJson) return []
    try {
      const raw = typeof p.genesJson === 'string' ? JSON.parse(p.genesJson) : p.genesJson
      return Array.isArray(raw) ? raw : []
    } catch { return [] }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold">Patterns</h1>
        <p className="text-dark-muted text-sm mt-0.5">
          GA-discovered candlestick patterns ranked by MC composite score — {patterns.length} active
        </p>
        <div className="mt-2 text-xs text-dark-muted bg-dark-hover/40 border border-dark-border rounded-lg px-3 py-2 max-w-2xl">
          <span className="text-dark-text font-medium">About these metrics: </span>
          MC Return, Win% and Sharpe are from the raw pattern simulation (no ML model). Negative values are
          expected — patterns are building blocks used as inputs to the ML model, which combines them with
          market context to achieve positive returns. High MC Score means the pattern is a strong signal
          source; the Trades page shows the actual ML-driven backtest performance.
        </div>
      </div>

      {/* Symbol filter */}
      <div className="flex flex-wrap gap-2">
        <button onClick={() => setSelectedSym('')}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            !selectedSym ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/30' : 'bg-dark-hover text-dark-muted border border-transparent'
          }`}>
          All ({activeAll.length})
        </button>
        {symbols.map((s) => {
          const count = activeAll.filter((p) => p.symbol === s).length
          return (
            <button key={s} onClick={() => setSelectedSym(s)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                selectedSym === s ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/30' : 'bg-dark-hover text-dark-muted border border-transparent'
              }`}>
              {s} ({count})
            </button>
          )
        })}
      </div>

      {/* Pattern table */}
      {loading ? (
        <div className="text-dark-muted text-sm py-8 text-center">Loading patterns...</div>
      ) : sorted.length === 0 ? (
        <div className="card text-center text-dark-muted py-8">
          <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30" />
          <div className="text-sm">No active patterns found. Run the pipeline to discover them.</div>
        </div>
      ) : (
        <div className="space-y-2">
          {sorted.map((p, i) => {
            const isExpanded = expandedId === p.id
            const genes = parseGenes(p)
            return (
              <div key={p.id} className={`card transition-all ${isExpanded ? 'border-accent-blue/30' : ''}`}>
                {/* Summary row */}
                <div className="flex items-center gap-3 cursor-pointer"
                  onClick={() => setExpandedId(isExpanded ? null : p.id)}>
                  <span className="text-dark-muted text-xs w-6 text-right">{i + 1}</span>
                  <span className="font-mono text-accent-blue font-medium text-sm w-16">{p.symbol}</span>
                  <span className="text-dark-muted text-xs w-10">{p.interval}</span>

                  <div className="flex-1 flex items-center gap-4">
                    <Metric label="MC Score" value={(p.mcCompositeScore || 0).toFixed(4)} color="green" />
                    <Metric label="Sharpe" value={(p.mcSharpe || 0).toFixed(3)} />
                    <Metric label="Win%" value={`${((p.mcWinRate || 0) * 100).toFixed(1)}%`}
                      color={(p.mcWinRate || 0) >= 0.5 ? 'green' : 'red'} />
                    <Metric label="Return" value={`${((p.mcReturn || 0) * 100).toFixed(2)}%`}
                      color={(p.mcReturn || 0) >= 0 ? 'green' : 'red'} />
                    <Metric label="Fitness" value={(p.fitness || 0).toFixed(3)} />
                    <Metric label="Accuracy" value={`${((p.accuracy || 0) * 100).toFixed(1)}%`} />
                  </div>

                  <div className="flex items-center gap-1.5 text-[10px] text-dark-muted">
                    <span>{p.patternLength || genes.length} candles</span>
                    {isExpanded
                      ? <ChevronUp className="w-3.5 h-3.5" />
                      : <ChevronDown className="w-3.5 h-3.5" />}
                  </div>
                </div>

                {/* Expanded detail: candlestick chart + gene data */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t border-dark-border">
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                      {/* Candlestick chart */}
                      <div>
                        <div className="text-xs text-dark-muted font-medium mb-2">Pattern Shape (Candlestick)</div>
                        <div className="rounded-lg border border-dark-border overflow-hidden">
                          <PatternCandlestick genes={genes} height={240} symbol={p.symbol} interval={p.interval} />
                        </div>
                      </div>

                      {/* Gene table */}
                      <div>
                        <div className="text-xs text-dark-muted font-medium mb-2">
                          Gene Details ({genes.length} candle{genes.length !== 1 ? 's' : ''})
                        </div>
                        <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                          {genes.map((g, gi) => {
                            const gene = typeof g === 'string' ? JSON.parse(g) : g
                            const type = (gene.candleType || 'DOJI').toUpperCase()
                            const TypeIcon = CANDLE_ICONS[type] || Minus
                            return (
                              <div key={gi} className="flex items-center gap-3 bg-dark-hover/30 rounded-lg px-3 py-2">
                                <span className="text-xs text-dark-muted w-5 text-right font-mono">{gi + 1}</span>
                                <TypeIcon className={`w-3.5 h-3.5 ${CANDLE_COLORS[type] || ''}`} />
                                <span className={`text-xs font-medium w-16 ${CANDLE_COLORS[type] || ''}`}>{type}</span>
                                <div className="flex-1 grid grid-cols-3 gap-2 text-[10px]">
                                  <span className="text-dark-muted">
                                    Price Δ: <span className={`font-mono ${(gene.expectedPriceChangePct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                                      {(gene.expectedPriceChangePct || 0) >= 0 ? '+' : ''}{(gene.expectedPriceChangePct || 0).toFixed(2)}%
                                    </span>
                                  </span>
                                  <span className="text-dark-muted">
                                    Volume: <span className="font-mono text-dark-text">{(gene.expectedVolumePct || 0).toFixed(1)}%</span>
                                  </span>
                                  <span className="text-dark-muted">
                                    Body: <span className="font-mono text-dark-text">{(gene.minBodyPct || 0).toFixed(0)}%</span>
                                  </span>
                                </div>
                              </div>
                            )
                          })}
                        </div>

                        {/* Pattern summary stats */}
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          <div className="bg-dark-hover/30 rounded px-3 py-2 text-center">
                            <div className="text-[10px] text-dark-muted">MC Composite</div>
                            <div className="text-sm font-mono font-bold text-accent-green">{(p.mcCompositeScore || 0).toFixed(4)}</div>
                          </div>
                          <div className="bg-dark-hover/30 rounded px-3 py-2 text-center">
                            <div className="text-[10px] text-dark-muted">MC Sharpe</div>
                            <div className="text-sm font-mono font-bold">{(p.mcSharpe || 0).toFixed(3)}</div>
                          </div>
                          <div className="bg-dark-hover/30 rounded px-3 py-2 text-center">
                            <div className="text-[10px] text-dark-muted">Win Rate</div>
                            <div className={`text-sm font-mono font-bold ${(p.mcWinRate || 0) >= 0.5 ? 'text-accent-green' : 'text-accent-red'}`}>
                              {((p.mcWinRate || 0) * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-dark-hover/30 rounded px-3 py-2 text-center">
                            <div className="text-[10px] text-dark-muted">Expected Return</div>
                            <div className={`text-sm font-mono font-bold ${(p.mcReturn || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                              {((p.mcReturn || 0) * 100).toFixed(2)}%
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Pattern metadata footer */}
                    <div className="mt-3 flex flex-wrap items-center gap-4 text-[10px] text-dark-muted">
                      {p.runId && <span>Run: <span className="font-mono">{p.runId.slice(0, 8)}</span></span>}
                      {p.rank && <span>Rank: #{p.rank}</span>}
                      {p.createdAt && <span>Discovered: {new Date(p.createdAt).toLocaleDateString()}</span>}
                      <span>Fitness: {(p.fitness || 0).toFixed(4)}</span>
                      <span>Accuracy: {((p.accuracy || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value, color }) {
  const colorClass = color === 'green' ? 'text-accent-green' : color === 'red' ? 'text-accent-red' : 'text-dark-text'
  return (
    <div className="hidden sm:block">
      <div className="text-[9px] text-dark-muted leading-none mb-0.5">{label}</div>
      <div className={`text-xs font-mono ${colorClass}`}>{value}</div>
    </div>
  )
}
