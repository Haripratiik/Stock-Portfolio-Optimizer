import { useState, useEffect } from 'react'
import { useRuns, useRunCharts } from '../hooks/useFirestore'
import { format } from 'date-fns'
import {
  BarChart3, TrendingUp, Activity, Clock, ChevronDown, ChevronUp,
  ImageOff, Loader2, Info, Maximize2, X,
} from 'lucide-react'

// ── Descriptive text for each chart type ─────────────────────────────────────

const CHART_DESCRIPTIONS = {
  portfolio: {
    title: 'Portfolio Performance Overview',
    icon: TrendingUp,
    iconColor: 'text-accent-blue',
    summary: 'Shows cumulative portfolio value over the backtest period, compared against buy-and-hold and S&P 500 benchmarks.',
    detail: [
      'The top panel plots your portfolio\'s cumulative returns against passive benchmarks (buy-and-hold of the same stocks and the S&P 500 index).',
      'Below it, per-stock P&L bars show each stock\'s individual contribution to total return.',
      'The allocation band at the bottom shows how the dynamic reallocation system shifted capital between stocks over time.',
      'Green outperformance above the benchmark lines means the pattern-based trading strategy is generating alpha.',
    ],
  },
  backtest: {
    title: 'Per-Stock Backtest Performance',
    icon: BarChart3,
    iconColor: 'text-accent-purple',
    summary: 'Individual backtest equity curves for each stock, broken down by timeframe.',
    detail: [
      'Each chart shows one stock\'s cumulative P&L during the backtest period.',
      'Multiple lines represent different timeframes (daily, hourly, etc.) — this reveals which intervals have the best-performing patterns.',
      'Trade markers (dots) appear at entry/exit points. Green = profitable trade, red = loss.',
      'The smoother the upward curve, the more consistent the pattern is at predicting price direction.',
    ],
  },
  fan_chart: {
    title: 'Monte Carlo Fan Charts',
    icon: Activity,
    iconColor: 'text-accent-green',
    summary: 'Monte Carlo simulation of possible future outcomes using discovered patterns.',
    detail: [
      'Each thin line represents one simulated trading path — the model replays pattern-based trades with randomized market noise.',
      'The shaded band covers the 10th–90th percentile range, giving you a confidence corridor for likely outcomes.',
      'The median line (thick) shows the "most likely" path. If it trends upward, the pattern set has positive expected value.',
      'A wider fan indicates higher uncertainty/risk. A narrow upward fan indicates consistent, reliable pattern performance.',
      'Paths that drop below the starting balance illustrate the risk of ruin — important for position sizing.',
    ],
  },
}

function ReturnBadge({ value, label }) {
  const pos = (value || 0) >= 0
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-dark-muted mb-0.5">{label}</div>
      <div className={`text-sm font-mono font-bold ${pos ? 'text-accent-green' : 'text-accent-red'}`}>
        {pos && value !== 0 ? '+' : ''}{(value || 0).toFixed(2)}
        {label.includes('%') || label.includes('Rate') || label.includes('Alpha') || label.includes('Return') ? '%' : ''}
      </div>
    </div>
  )
}

function ChartCard({ chart, fullWidth = false }) {
  const [zoomed, setZoomed] = useState(false)

  if (!chart?.imageData) return null

  const desc = CHART_DESCRIPTIONS[chart.chartType] || {}

  return (
    <>
      <div className={`card flex flex-col gap-3 ${fullWidth ? 'col-span-full' : ''}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {desc.icon && <desc.icon className={`w-4 h-4 ${desc.iconColor || ''}`} />}
            <span className="text-sm font-semibold">{chart.label}</span>
            {chart.symbol && (
              <span className="px-2 py-0.5 rounded bg-dark-hover text-xs font-mono text-accent-blue border border-dark-border">
                {chart.symbol}
              </span>
            )}
          </div>
          <button onClick={() => setZoomed(true)}
            className="flex items-center gap-1 text-xs text-dark-muted hover:text-accent-blue px-2 py-1 rounded hover:bg-dark-hover transition-colors">
            <Maximize2 className="w-3 h-3" /> Expand
          </button>
        </div>

        <img src={chart.imageData} alt={chart.label}
          className="w-full rounded-lg border border-dark-border cursor-zoom-in object-contain"
          style={{ maxHeight: fullWidth ? '700px' : '500px' }}
          onClick={() => setZoomed(true)} />

        <div className="text-[10px] text-dark-muted flex items-start gap-1.5 leading-relaxed">
          <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>{desc.summary || ''}</span>
        </div>
      </div>

      {zoomed && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setZoomed(false)}>
          <div className="max-w-[95vw] max-h-[95vh] overflow-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-semibold text-white">{chart.label}</span>
              <button onClick={() => setZoomed(false)}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-white px-3 py-1 rounded border border-gray-600 hover:border-gray-400">
                <X className="w-3 h-3" /> Close
              </button>
            </div>
            <img src={chart.imageData} alt={chart.label}
              className="max-w-full max-h-[85vh] rounded-lg border border-gray-700 object-contain" />
          </div>
        </div>
      )}
    </>
  )
}

export default function Charts() {
  const [runs, runsLoading] = useRuns(30)
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [charts, chartsLoading] = useRunCharts(selectedRunId)
  const [showAllRuns, setShowAllRuns] = useState(false)
  const [showGuide, setShowGuide] = useState(false)

  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) setSelectedRunId(runs[0].id)
  }, [runs]) // eslint-disable-line react-hooks/exhaustive-deps

  const selectedRun = runs.find((r) => r.id === selectedRunId) ?? null
  const portfolioChart = charts.find((c) => c.chartType === 'portfolio') ?? null
  const backtestCharts = charts.filter((c) => c.chartType === 'backtest')
  const fanCharts = charts.filter((c) => c.chartType === 'fan_chart')
  const hasAnyChart = charts.length > 0

  const RUNS_VISIBLE_DEFAULT = 8
  const visibleRuns = showAllRuns ? runs : runs.slice(0, RUNS_VISIBLE_DEFAULT)

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Charts</h1>
          <p className="text-dark-muted text-sm mt-0.5">
            Pipeline output — backtest performance, portfolio overview &amp; Monte Carlo simulations
          </p>
        </div>
        <button onClick={() => setShowGuide((v) => !v)}
          className="flex items-center gap-1.5 text-xs text-dark-muted hover:text-accent-blue px-3 py-1.5 rounded-lg border border-dark-border hover:border-accent-blue/30 transition-colors">
          <Info className="w-3.5 h-3.5" /> {showGuide ? 'Hide' : 'Reading'} Guide
        </button>
      </div>

      {/* Chart reading guide */}
      {showGuide && (
        <div className="card space-y-4 border-accent-blue/20">
          <h3 className="text-sm font-medium text-accent-blue flex items-center gap-2">
            <Info className="w-4 h-4" /> How to Read These Charts
          </h3>
          {Object.entries(CHART_DESCRIPTIONS).map(([key, desc]) => (
            <div key={key} className="bg-dark-hover/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                {desc.icon && <desc.icon className={`w-4 h-4 ${desc.iconColor}`} />}
                <span className="text-sm font-medium">{desc.title}</span>
              </div>
              <ul className="space-y-1.5">
                {desc.detail.map((line, i) => (
                  <li key={i} className="text-xs text-dark-text/80 leading-relaxed flex items-start gap-2">
                    <span className="text-accent-blue mt-0.5 flex-shrink-0">•</span>
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {runsLoading && (
        <div className="flex items-center justify-center py-20 gap-3 text-dark-muted">
          <Loader2 className="w-5 h-5 animate-spin" /> <span className="text-sm">Loading runs...</span>
        </div>
      )}

      {!runsLoading && runs.length === 0 && (
        <div className="card text-center py-16 text-dark-muted space-y-2">
          <ImageOff className="w-10 h-10 mx-auto opacity-30" />
          <p className="text-sm font-medium">No pipeline runs yet</p>
          <p className="text-xs">Run the full pipeline from the <span className="text-accent-blue">Run</span> page to generate charts.</p>
        </div>
      )}

      {/* Run selector */}
      {runs.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-dark-muted" />
            <h3 className="text-sm font-medium text-dark-muted">Select Run</h3>
            <span className="text-xs text-dark-muted ml-1">({runs.length} available)</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {visibleRuns.map((r) => {
              const isSelected = r.id === selectedRunId
              const ret = r.totalReturnPct || 0
              return (
                <button key={r.id} onClick={() => setSelectedRunId(r.id)}
                  className={`px-3 py-2 rounded-lg text-left transition-all border ${
                    isSelected
                      ? 'bg-accent-blue/15 border-accent-blue/40 shadow-sm'
                      : 'bg-dark-hover/50 border-dark-border hover:border-dark-text/30 hover:bg-dark-hover'
                  }`}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-xs font-mono font-bold ${isSelected ? 'text-accent-blue' : 'text-dark-text'}`}>
                      {r.symbols?.join(', ') || '—'}
                    </span>
                    <span className={`text-[10px] font-mono ${ret >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-[10px] text-dark-muted">
                    {r.timestamp ? format(new Date(r.timestamp), 'MMM d, HH:mm') : '—'}
                  </div>
                </button>
              )
            })}
          </div>
          {runs.length > RUNS_VISIBLE_DEFAULT && (
            <button onClick={() => setShowAllRuns((v) => !v)}
              className="flex items-center gap-1 text-xs text-dark-muted hover:text-accent-blue transition-colors">
              {showAllRuns
                ? <><ChevronUp className="w-3.5 h-3.5" /> Show fewer</>
                : <><ChevronDown className="w-3.5 h-3.5" /> Show {runs.length - RUNS_VISIBLE_DEFAULT} more runs</>}
            </button>
          )}
        </div>
      )}

      {/* Selected run summary */}
      {selectedRun && (
        <div className="flex flex-wrap items-center gap-6 bg-dark-card rounded-xl border border-dark-border px-5 py-3">
          <div className="flex-1 min-w-0">
            <div className="text-xs text-dark-muted mb-0.5">Selected run</div>
            <div className="font-mono font-bold text-sm">{selectedRun.symbols?.join(', ') || '—'}</div>
          </div>
          <ReturnBadge value={selectedRun.totalReturnPct} label="Return %" />
          <ReturnBadge value={selectedRun.alphaVsBuyHold} label="Alpha B&H" />
          <ReturnBadge value={selectedRun.alphaVsSP500}   label="Alpha S&P" />
          <ReturnBadge value={selectedRun.winRate}         label="Win Rate %" />
          <div className="flex flex-col items-center">
            <div className="text-[10px] text-dark-muted mb-0.5">Sharpe</div>
            <div className="text-sm font-mono font-bold">{(selectedRun.sharpeRatio || 0).toFixed(2)}</div>
          </div>
          <div className="flex flex-col items-center">
            <div className="text-[10px] text-dark-muted mb-0.5">Trades</div>
            <div className="text-sm font-mono font-bold">{selectedRun.numTrades || 0}</div>
          </div>
          <div className="text-xs text-dark-muted ml-auto">
            {selectedRun.timestamp ? format(new Date(selectedRun.timestamp), 'PPpp') : '—'}
          </div>
        </div>
      )}

      {chartsLoading && selectedRunId && (
        <div className="flex items-center justify-center py-16 gap-3 text-dark-muted">
          <Loader2 className="w-5 h-5 animate-spin" /> <span className="text-sm">Loading charts...</span>
        </div>
      )}

      {!chartsLoading && selectedRunId && !hasAnyChart && (
        <div className="card text-center py-14 text-dark-muted space-y-2">
          <ImageOff className="w-9 h-9 mx-auto opacity-30" />
          <p className="text-sm font-medium">No charts for this run</p>
          <p className="text-xs max-w-sm mx-auto">
            Charts are generated by pipeline runs executed after the latest backend update. Re-run to produce charts.
          </p>
        </div>
      )}

      {/* Chart sections */}
      {hasAnyChart && (
        <div className="space-y-8">
          {portfolioChart && (
            <ChartSection type="portfolio">
              <ChartCard chart={portfolioChart} fullWidth />
            </ChartSection>
          )}

          {backtestCharts.length > 0 && (
            <ChartSection type="backtest" count={backtestCharts.length} run={selectedRun}>
              <div className={`grid gap-4 ${backtestCharts.length === 1 ? 'grid-cols-1' : 'grid-cols-1 xl:grid-cols-2'}`}>
                {backtestCharts.map((c) => <ChartCard key={c.id} chart={c} />)}
              </div>
            </ChartSection>
          )}

          {fanCharts.length > 0 && (
            <ChartSection type="fan_chart" count={fanCharts.length} run={selectedRun}>
              <div className={`grid gap-4 ${fanCharts.length === 1 ? 'grid-cols-1' : 'grid-cols-1 xl:grid-cols-2'}`}>
                {fanCharts.map((c) => <ChartCard key={c.id} chart={c} />)}
              </div>
            </ChartSection>
          )}
        </div>
      )}
    </div>
  )
}

function ChartSection({ type, count, run, children }) {
  const desc = CHART_DESCRIPTIONS[type] || {}
  const Icon = desc.icon
  return (
    <section className="space-y-3">
      <div className="flex items-start gap-2">
        {Icon && <Icon className={`w-4 h-4 mt-0.5 ${desc.iconColor}`} />}
        <div>
          <h2 className="text-base font-semibold">{desc.title}</h2>
          <p className="text-xs text-dark-muted mt-0.5">{desc.summary}</p>
        </div>
      </div>
      {children}
    </section>
  )
}
