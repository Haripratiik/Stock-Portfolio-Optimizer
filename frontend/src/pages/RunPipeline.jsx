import { useState } from 'react'
import { db } from '../firebase'
import { collection, addDoc } from 'firebase/firestore'
import { useTemplates, useStockMeta } from '../hooks/useFirestore'
import { Play, Save, ChevronDown, ChevronUp, Zap, Plus } from 'lucide-react'

// ─── Pipeline task types ─────────────────────────────────────────────
const TASK_TYPES = [
  { value: 'full_pipeline',       label: 'Full Pipeline',              desc: 'GA → MC → Refine → Sentiment → Strategies → ML → Backtest → Allocate' },
  { value: 'backtest_only',       label: 'Backtest Current Strategy',  desc: 'Run backtester on existing ML models and patterns' },
  { value: 'mcmc_simulation',     label: 'MCMC Simulation',            desc: 'Monte Carlo simulation on current patterns' },
  { value: 'retrain_ml',          label: 'Retrain ML Models',          desc: 'Retrain all Stock ML + Portfolio ML + Trading Decider models' },
  { value: 'retrain_trading',     label: 'Retrain Trading Decider',    desc: 'Retrain only the signal reconciliation model' },
  { value: 'sentiment_update',    label: 'Update Sentiment',           desc: 'Fetch latest news headlines & update sentiment scores' },
  { value: 'strategy_refresh',    label: 'Refresh Strategies',         desc: 'Re-generate cross-stock strategies (statistical + OpenAI)' },
  { value: 'pattern_discovery',   label: 'Pattern Discovery Only',     desc: 'Run GA to discover new patterns (no ML training)' },
  { value: 'add_stock_pipeline',  label: 'Add New Stock(s)',           desc: 'Full GA+MC for selected new stocks, reuse patterns for existing — then retrain all ML + backtest + connected stocks' },
  { value: 'incremental_update', label: 'Incremental Update',         desc: 'Lightweight refresh: reuses stored patterns, retrains ML + sentiment with latest data' },
  { value: 'connected_stocks',   label: 'Evaluate Connected Stocks',  desc: 'Discover and evaluate supply-chain connected stocks for automatic trading' },
  { value: 'trading_execute_1h', label: 'Execute Trading Cycle [1h]', desc: 'Hourly: ML signals → orders (uses pipeline patterns + allocation)' },
  { value: 'trading_execute_1d', label: 'Execute Trading Cycle [1d]', desc: 'Daily: ML signals → orders + saves next-day summary for website' },
]

// ─── Production config (proper full pipeline — 5y backtest, 3y train) ──
const DEFAULT_CONFIG = {
  POPULATION_SIZE:            2000,
  NUM_GENERATIONS:            150,
  MC_NUM_SIMULATIONS:         2000,
  MC_TIME_HORIZON:            252,
  ML_FORWARD_PERIODS:         10,
  ML_PORTFOLIO_FORWARD_PERIODS: 10,
  BACKTEST_LOOKBACK_DAYS:     1825,   // 5 years
  ML_TRAIN_DAYS:              1095,   // 3 years
  DECIDER_PATTERN_WEIGHT:     0.80,
  DECIDER_PORTFOLIO_WEIGHT:   0.60,
  DECIDER_MIN_CONFIDENCE:     0.25,
  IFA_MAX_SLOT_ALLOCATION:    0.40,
  IFA_MAX_STOCK_ALLOCATION:   0.60,
}

const QUICK_CONFIG = {
  POPULATION_SIZE:            200,
  NUM_GENERATIONS:            20,
  MC_NUM_SIMULATIONS:         100,
  MC_TIME_HORIZON:            63,
  ML_FORWARD_PERIODS:         5,
  ML_PORTFOLIO_FORWARD_PERIODS: 5,
  BACKTEST_LOOKBACK_DAYS:     180,
  ML_TRAIN_DAYS:              365,
  DECIDER_PATTERN_WEIGHT:     0.80,
  DECIDER_PORTFOLIO_WEIGHT:   0.60,
  DECIDER_MIN_CONFIDENCE:     0.25,
  IFA_MAX_SLOT_ALLOCATION:    0.40,
  IFA_MAX_STOCK_ALLOCATION:   0.60,
}

// ─── Common stock universe (loaded from Portfolio) ───────────────────

export default function RunPipeline() {
  const [templates] = useTemplates()
  const [stockMeta] = useStockMeta()
  const portfolioStocks = stockMeta.map((m) => m.symbol).sort()
  const [taskType, setTaskType] = useState('full_pipeline')
  const [config, setConfig] = useState({ ...DEFAULT_CONFIG })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [templateName, setTemplateName] = useState('')
  const [newStocks, setNewStocks] = useState(new Set())

  const taskInfo = TASK_TYPES.find((t) => t.value === taskType) || TASK_TYPES[0]
  const isAddStock = taskType === 'add_stock_pipeline'

  // Submit to Firestore queue — uses ALL portfolio stocks
  const handleSubmit = async () => {
    if (portfolioStocks.length === 0) {
      alert('No stocks in portfolio. Go to Portfolio tab to add tickers first.')
      return
    }
    if (isAddStock && newStocks.size === 0) {
      alert('Select at least one stock to run the full pipeline for.')
      return
    }
    setSubmitting(true)
    try {
      const submitConfig = isAddStock
        ? { ...config, newSymbols: [...newStocks] }
        : config
      const desc = isAddStock
        ? `${taskInfo.label}: ${[...newStocks].join(', ')} (full) + ${portfolioStocks.filter((s) => !newStocks.has(s)).length} existing`
        : `${taskInfo.label} — ${portfolioStocks.join(', ')}`

      await addDoc(collection(db, 'run_commands'), {
        type: taskType,
        status: 'queued',
        description: desc,
        stocks: portfolioStocks,
        config: submitConfig,
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: null,
        startedAt: null,
        completedAt: null,
        result: null,
        error: null,
      })
      setSubmitted(true)
      setTimeout(() => setSubmitted(false), 3000)
    } catch (err) {
      alert('Failed to queue: ' + err.message)
    }
    setSubmitting(false)
  }

  // Save as template
  const handleSaveTemplate = async () => {
    if (!templateName.trim()) return
    try {
      await addDoc(collection(db, 'config_templates'), {
        name: templateName.trim(),
        description: `${taskInfo.label} with ${portfolioStocks.length} stocks`,
        taskType,
        config,
        createdAt: new Date().toISOString(),
      })
      setTemplateName('')
      alert('Template saved!')
    } catch (err) {
      alert('Save failed: ' + err.message)
    }
  }

  // Load template
  const loadTemplate = (tpl) => {
    setTaskType(tpl.taskType || 'full_pipeline')
    setConfig({ ...DEFAULT_CONFIG, ...(tpl.config || {}) })
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold">Run Pipeline</h1>
        <p className="text-dark-muted text-sm mt-0.5">
          Configure and submit a pipeline task to the queue
        </p>
      </div>

      {/* Portfolio stocks info */}
      <div className="card">
        <h3 className="text-sm font-medium text-dark-muted mb-2">Portfolio Stocks ({portfolioStocks.length})</h3>
        {portfolioStocks.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {portfolioStocks.map((sym) => (
              <span key={sym} className="px-2.5 py-1 rounded text-xs font-mono font-medium bg-accent-blue/15 text-accent-blue border border-accent-blue/30">
                {sym}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-xs text-dark-muted">No stocks in portfolio. <a href="/portfolio" className="text-accent-blue hover:underline">Add stocks first</a>.</p>
        )}
        <p className="text-xs text-dark-muted mt-2">Pipeline will run on all portfolio stocks. Manage stocks in the Portfolio tab.</p>
      </div>

      {/* Templates */}
      {templates.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-dark-muted mb-3">Saved Templates</h3>
          <div className="flex flex-wrap gap-2">
            {templates.map((tpl) => (
              <button
                key={tpl.id}
                onClick={() => loadTemplate(tpl)}
                className="btn-secondary text-xs"
              >
                {tpl.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Quick presets */}
      <div className="flex gap-2">
        <button
          onClick={() => setConfig({ ...DEFAULT_CONFIG })}
          className="btn-secondary text-xs"
        >
          Production Config
        </button>
        <button
          onClick={() => setConfig({ ...QUICK_CONFIG })}
          className="btn-secondary text-xs"
        >
          Quick Test
        </button>
      </div>

      {/* Task type */}
      <div className="card space-y-4">
        <h3 className="text-sm font-medium">Task Type</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {TASK_TYPES.map((t) => (
            <button
              key={t.value}
              onClick={() => setTaskType(t.value)}
              className={`text-left p-3 rounded-lg border transition-all text-xs ${
                taskType === t.value
                  ? 'border-accent-blue bg-accent-blue/10 text-accent-blue'
                  : 'border-dark-border bg-dark-hover/50 text-dark-muted hover:border-dark-muted'
              }`}
            >
              <div className="font-medium">{t.label}</div>
            </button>
          ))}
        </div>
        <p className="text-dark-muted text-xs">{taskInfo.desc}</p>
      </div>

      {/* New-stock selector (only for add_stock_pipeline) */}
      {isAddStock && portfolioStocks.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2">
            <Plus className="w-4 h-4 text-accent-green" />
            <h3 className="text-sm font-medium">Select New Stocks</h3>
          </div>
          <p className="text-xs text-dark-muted">
            Check the stocks that need <strong>full pattern discovery</strong> (GA + MC).
            Unchecked stocks will reuse their existing stored patterns — much faster.
          </p>
          <div className="flex flex-wrap gap-2">
            {portfolioStocks.map((sym) => {
              const selected = newStocks.has(sym)
              return (
                <button
                  key={sym}
                  onClick={() => setNewStocks((prev) => {
                    const next = new Set(prev)
                    if (next.has(sym)) next.delete(sym)
                    else next.add(sym)
                    return next
                  })}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition-all ${
                    selected
                      ? 'bg-accent-green/15 text-accent-green border-accent-green/40'
                      : 'bg-dark-hover/50 text-dark-muted border-dark-border hover:border-dark-muted'
                  }`}
                >
                  {selected ? '✓ ' : ''}{sym}
                </button>
              )
            })}
          </div>
          {newStocks.size > 0 && (
            <div className="flex gap-4 text-xs text-dark-muted pt-1 border-t border-dark-border/50">
              <span>
                <span className="font-bold text-accent-green">{newStocks.size}</span> stock{newStocks.size > 1 ? 's' : ''} will get full GA+MC discovery
              </span>
              <span>
                <span className="font-bold text-dark-text">{portfolioStocks.length - newStocks.size}</span> will reuse stored patterns
              </span>
            </div>
          )}
        </div>
      )}

      {/* Advanced config */}
      <div className="card">
        <button
          onClick={() => setShowAdvanced((p) => !p)}
          className="flex items-center gap-2 text-sm font-medium w-full"
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          Advanced Configuration
        </button>
        {showAdvanced && (
          <div className="mt-4 grid grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(config).map(([key, val]) => (
              <div key={key}>
                <label className="text-xs text-dark-muted block mb-1">
                  {key.replace(/_/g, ' ')}
                </label>
                <input
                  className="input text-sm font-mono"
                  type="number"
                  step="any"
                  value={val}
                  onChange={(e) =>
                    setConfig((prev) => ({
                      ...prev,
                      [key]: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Save template */}
      <div className="flex items-center gap-2">
        <input
          className="input text-sm flex-1"
          placeholder="Template name (optional)..."
          value={templateName}
          onChange={(e) => setTemplateName(e.target.value)}
        />
        <button onClick={handleSaveTemplate} disabled={!templateName.trim()} className="btn-secondary flex items-center gap-1.5 text-sm">
          <Save className="w-3.5 h-3.5" /> Save Template
        </button>
      </div>

      {/* Submit */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={submitting || portfolioStocks.length === 0}
          className="btn-primary flex items-center gap-2 text-sm px-6 py-3"
        >
          {submitting ? (
            <Zap className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {submitting ? 'Submitting...' : 'Add to Queue'}
        </button>
        {submitted && (
          <span className="badge-green text-sm">Queued successfully!</span>
        )}
      </div>

    </div>
  )
}
