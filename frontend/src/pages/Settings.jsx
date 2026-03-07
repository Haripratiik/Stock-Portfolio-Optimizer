import { useState, useEffect } from 'react'
import { useTemplates } from '../hooks/useFirestore'
import { db } from '../firebase'
import { doc, deleteDoc, getDoc, setDoc, collection, getDocs, writeBatch } from 'firebase/firestore'
import {
  Trash2, FileText, Clock, Zap, Newspaper, Link2, BarChart3, Bot,
  CheckCircle, AlertCircle, Info, Settings, ChevronRight, Save,
  AlertTriangle, Database, Loader2, Bell, Mail, Send,
} from 'lucide-react'

const SCHEDULE_TASKS = [
  {
    key: 'sentiment_fetch',
    label: 'Sentiment Fetch',
    icon: Newspaper,
    desc: 'Fetch & score headlines — runs directly in background, no queue needed.',
    unit: 'hours',
    defaultVal: 4,
    color: 'text-accent-blue',
    badge: 'background',
  },
  {
    key: 'incremental_update',
    label: 'ML Model Update',
    icon: Zap,
    desc: 'Retrain ML models with fresh market data.',
    unit: 'hours',
    defaultVal: 168,
    color: 'text-accent-green',
    badge: 'queued',
  },
  {
    key: 'connected_stocks',
    label: 'Connected Stocks',
    icon: Link2,
    desc: 'Evaluate supply-chain and correlated stock relationships.',
    unit: 'hours',
    defaultVal: 336,
    color: 'text-accent-purple',
    badge: 'queued',
  },
  {
    key: 'strategy_refresh',
    label: 'Strategy Refresh',
    icon: BarChart3,
    desc: 'Regenerate cross-stock trading strategies.',
    unit: 'hours',
    defaultVal: 336,
    color: 'text-accent-purple',
    badge: 'queued',
  },
  {
    key: 'trading_execute_1h',
    label: 'Trading Cycle [1h]',
    icon: Bot,
    desc: 'Hourly cycle — ML signals → broker orders.',
    unit: 'hours',
    defaultVal: 1,
    color: 'text-accent-yellow',
    badge: 'queued',
  },
  {
    key: 'trading_execute_1d',
    label: 'Trading Cycle [1d]',
    icon: Bot,
    desc: 'Daily cycle — ML signals → orders + next-day summary.',
    unit: 'hours',
    defaultVal: 24,
    color: 'text-accent-yellow',
    badge: 'queued',
  },
]

const PIPELINE_STAGES = {
  'Full Pipeline': 'GA → MC → Refine → Sentiment → Strategies → StockML → PortfolioML → MC Phase 2 → Decider → Backtest → Allocate → Connected Stocks',
  'Incremental Update': 'Load Patterns → Sentiment → Strategies → Retrain ML → Decider → Quick Backtest → Connected Stocks',
  'Trading Cycle': 'Load Portfolio → Fetch OHLCV → StockML Predict → PortfolioML → TradingDecider → OrderBuilder → Broker Execute → Audit Log',
}

const BADGE_STYLES = {
  background: 'bg-accent-blue/10 text-accent-blue border border-accent-blue/20',
  queued: 'bg-dark-hover text-dark-muted border border-dark-border',
}

function formatInterval(hours) {
  if (!hours) return '—'
  if (hours < 24) return `Every ${hours}h`
  const days = Math.round(hours / 24)
  if (days < 7) return `Every ${days}d`
  const weeks = Math.round(days / 7)
  return `Every ${weeks}w`
}

function SectionHeader({ icon: Icon, title, subtitle }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <div className="w-8 h-8 rounded-lg bg-accent-blue/10 flex items-center justify-center flex-shrink-0">
        <Icon className="w-4 h-4 text-accent-blue" />
      </div>
      <div>
        <h2 className="text-sm font-semibold">{title}</h2>
        {subtitle && <p className="text-xs text-dark-muted mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}

const TRADE_COLLECTIONS = [
  { key: 'trade_log', label: 'Trade Log', desc: 'Individual order records' },
  { key: 'trade_positions', label: 'Positions', desc: 'Per-symbol position snapshots' },
  { key: 'trade_daily', label: 'Daily Summaries', desc: 'Daily P&L records' },
  { key: 'trade_cycles', label: 'Trade Cycles', desc: 'Cycle metadata & results' },
]

async function deleteCollection(colName) {
  const snap = await getDocs(collection(db, colName))
  if (snap.empty) return 0
  const batchSize = 400
  let deleted = 0
  const docs = snap.docs
  for (let i = 0; i < docs.length; i += batchSize) {
    const batch = writeBatch(db)
    docs.slice(i, i + batchSize).forEach((d) => batch.delete(d.ref))
    await batch.commit()
    deleted += Math.min(batchSize, docs.length - i)
  }
  return deleted
}

const ALERT_FIELDS = [
  { key: 'emailTo', label: 'Email To', type: 'text', placeholder: 'you@example.com' },
  { key: 'smtpServer', label: 'SMTP Server', type: 'text', placeholder: 'smtp.gmail.com' },
  { key: 'smtpPort', label: 'SMTP Port', type: 'number', placeholder: '587' },
  { key: 'smtpUser', label: 'SMTP User', type: 'text', placeholder: 'bot@example.com' },
  { key: 'smtpPassword', label: 'SMTP Password', type: 'password', placeholder: 'app-password' },
  { key: 'telegramBotToken', label: 'Telegram Bot Token', type: 'password', placeholder: '123456:ABC-DEF' },
  { key: 'telegramChatId', label: 'Telegram Chat ID', type: 'text', placeholder: '12345678' },
  { key: 'pnlThresholdPct', label: 'P&L Alert Threshold (%)', type: 'number', placeholder: '3.0' },
]

function AlertConfigSection() {
  const [cfg, setCfg] = useState(null)
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState(null)

  useEffect(() => {
    (async () => {
      try {
        const snap = await getDoc(doc(db, 'app_config', 'alerts'))
        setCfg(snap.exists() ? snap.data() : { enabled: false, emailEnabled: false, telegramEnabled: false, pnlThresholdPct: 3.0 })
      } catch {
        setCfg({ enabled: false, emailEnabled: false, telegramEnabled: false, pnlThresholdPct: 3.0 })
      }
    })()
  }, [])

  const save = async () => {
    if (!cfg) return
    setSaving(true)
    try {
      await setDoc(doc(db, 'app_config', 'alerts'), cfg)
      setStatus('success')
    } catch {
      setStatus('error')
    }
    setSaving(false)
    setTimeout(() => setStatus(null), 3000)
  }

  if (!cfg) return null

  return (
    <div className="card">
      <SectionHeader icon={Bell} title="Alert Notifications" subtitle="Email & Telegram alerts for trades, P&L, and system events" />

      <div className="space-y-4">
        <div className="flex items-center gap-6">
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={cfg.enabled || false} onChange={e => setCfg(p => ({ ...p, enabled: e.target.checked }))} className="accent-accent-blue" />
            <span className="font-medium">Alerts Enabled</span>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <Mail className="w-3.5 h-3.5 text-dark-muted" />
            <input type="checkbox" checked={cfg.emailEnabled || false} onChange={e => setCfg(p => ({ ...p, emailEnabled: e.target.checked }))} className="accent-accent-blue" />
            <span>Email</span>
          </label>
          <label className="flex items-center gap-2 text-xs">
            <Send className="w-3.5 h-3.5 text-dark-muted" />
            <input type="checkbox" checked={cfg.telegramEnabled || false} onChange={e => setCfg(p => ({ ...p, telegramEnabled: e.target.checked }))} className="accent-accent-blue" />
            <span>Telegram</span>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {ALERT_FIELDS.map(f => (
            <div key={f.key}>
              <label className="text-[10px] text-dark-muted uppercase tracking-wide">{f.label}</label>
              <input
                type={f.type}
                placeholder={f.placeholder}
                value={cfg[f.key] || ''}
                onChange={e => setCfg(p => ({ ...p, [f.key]: f.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value }))}
                className="w-full mt-1 px-2.5 py-1.5 text-xs rounded-md bg-dark-hover border border-dark-border text-dark-text focus:border-accent-blue focus:outline-none"
              />
            </div>
          ))}
        </div>

        <div className="flex items-center gap-2 pt-2">
          <button onClick={save} disabled={saving} className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-accent-blue/15 text-accent-blue text-xs font-medium border border-accent-blue/30 hover:bg-accent-blue/25 transition-colors disabled:opacity-40">
            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            Save Alert Config
          </button>
          {status === 'success' && <span className="text-xs text-accent-green flex items-center gap-1"><CheckCircle className="w-3 h-3" />Saved</span>}
          {status === 'error' && <span className="text-xs text-accent-red flex items-center gap-1"><AlertCircle className="w-3 h-3" />Failed</span>}
        </div>
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const [templates] = useTemplates()
  const [schedule, setSchedule] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null) // 'success' | 'error' | null
  const [clearing, setClearing] = useState(null) // collection key being cleared
  const [clearResult, setClearResult] = useState(null) // { type, msg }

  useEffect(() => {
    const defaults = {}
    SCHEDULE_TASKS.forEach((t) => { defaults[t.key] = t.defaultVal })
    const load = async () => {
      try {
        const snap = await getDoc(doc(db, 'scheduler_config', 'intervals'))
        setSchedule(snap.exists() ? { ...defaults, ...snap.data() } : defaults)
      } catch {
        setSchedule(defaults)
      }
    }
    load()
  }, [])

  const handleSave = async () => {
    if (!schedule) return
    setSaving(true)
    setSaveStatus(null)
    try {
      await setDoc(doc(db, 'scheduler_config', 'intervals'), schedule)
      setSaveStatus('success')
      setTimeout(() => setSaveStatus(null), 3000)
    } catch (e) {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus(null), 4000)
    }
    setSaving(false)
  }

  const updateInterval = (key, val) => {
    setSchedule((prev) => ({ ...prev, [key]: Math.max(1, parseInt(val) || 1) }))
  }

  const deleteTemplate = async (id) => {
    if (confirm('Delete this template?')) {
      try {
        await deleteDoc(doc(db, 'config_templates', id))
      } catch (e) {
        alert('Failed to delete: ' + e.message)
      }
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      {/* Page header */}
      <div className="flex items-center gap-3">
        <Settings className="w-5 h-5 text-dark-muted" />
        <div>
          <h1 className="text-xl font-bold">Settings</h1>
          <p className="text-dark-muted text-sm mt-0.5">
            Manage templates, scheduling, and system configuration
          </p>
        </div>
      </div>

      {/* Auto-Schedule Configuration */}
      <div className="card">
        <SectionHeader
          icon={Clock}
          title="Auto-Schedule Intervals"
          subtitle="Control how often the local agent runs each automated task"
        />

        <div className="flex items-start gap-3 p-3 rounded-lg bg-accent-blue/5 border border-accent-blue/15 mb-5">
          <Info className="w-4 h-4 text-accent-blue mt-0.5 flex-shrink-0" />
          <p className="text-xs text-dark-muted leading-relaxed">
            Enable the scheduler in the <span className="text-dark-text font-medium">Local Agent</span> app
            via the <span className="text-dark-text font-medium">Auto-Schedule</span> toggle. Sentiment
            fetching runs directly in the background; all other tasks are queued for your approval before
            execution.
          </p>
        </div>

        {schedule ? (
          <>
            <div className="space-y-2">
              {SCHEDULE_TASKS.map((task) => {
                const Icon = task.icon
                const val = schedule[task.key] ?? task.defaultVal
                return (
                  <div
                    key={task.key}
                    className="flex items-center gap-4 p-3.5 rounded-lg bg-dark-hover/30 border border-dark-border hover:border-dark-border/80 transition-colors group"
                  >
                    <div className={`w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 bg-dark-hover/60`}>
                      <Icon className={`w-3.5 h-3.5 ${task.color}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{task.label}</span>
                        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${BADGE_STYLES[task.badge]}`}>
                          {task.badge}
                        </span>
                      </div>
                      <div className="text-xs text-dark-muted mt-0.5">{task.desc}</div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <input
                        type="number"
                        className="input text-xs w-16 text-center font-mono"
                        value={val}
                        onChange={(e) => updateInterval(task.key, e.target.value)}
                        min={1}
                      />
                      <span className="text-xs text-dark-muted w-8">hrs</span>
                      <span className="text-xs text-accent-blue font-medium w-20 text-right">
                        {formatInterval(val)}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Save row */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-dark-border">
              <div className="flex items-center gap-2 h-7">
                {saveStatus === 'success' && (
                  <span className="flex items-center gap-1.5 text-xs text-accent-green">
                    <CheckCircle className="w-3.5 h-3.5" /> Saved successfully
                  </span>
                )}
                {saveStatus === 'error' && (
                  <span className="flex items-center gap-1.5 text-xs text-accent-red">
                    <AlertCircle className="w-3.5 h-3.5" /> Save failed — check permissions
                  </span>
                )}
              </div>
              <button
                onClick={handleSave}
                disabled={saving}
                className="btn-primary text-xs px-4 py-1.5 flex items-center gap-1.5"
              >
                <Save className="w-3.5 h-3.5" />
                {saving ? 'Saving…' : 'Save Intervals'}
              </button>
            </div>
          </>
        ) : (
          <div className="space-y-2">
            {SCHEDULE_TASKS.map((t) => (
              <div key={t.key} className="h-14 rounded-lg bg-dark-hover/20 border border-dark-border animate-pulse" />
            ))}
          </div>
        )}
      </div>

      {/* Saved templates */}
      <div className="card">
        <SectionHeader
          icon={FileText}
          title="Saved Config Templates"
          subtitle="Reusable pipeline configurations created from the Run Pipeline page"
        />
        {templates.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-dark-muted">
            <FileText className="w-8 h-8 mb-2 opacity-20" />
            <p className="text-sm">No templates saved yet.</p>
            <p className="text-xs mt-1 opacity-70">Create one from the Run Pipeline page.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {templates.map((tpl) => (
              <div
                key={tpl.id}
                className="flex items-center justify-between p-3.5 rounded-lg bg-dark-hover/30 border border-dark-border hover:border-dark-border/80 transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-md bg-dark-hover/60 flex items-center justify-center">
                    <FileText className="w-3.5 h-3.5 text-dark-muted" />
                  </div>
                  <div>
                    <div className="text-sm font-medium">{tpl.name}</div>
                    <div className="text-xs text-dark-muted mt-0.5 flex items-center gap-1.5">
                      <span className="capitalize">{tpl.taskType?.replace(/_/g, ' ')}</span>
                      {tpl.stocks?.length > 0 && (
                        <>
                          <span className="opacity-40">·</span>
                          <span>{tpl.stocks.join(', ')}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => deleteTemplate(tpl.id)}
                  className="p-1.5 text-dark-muted hover:text-accent-red transition-colors opacity-0 group-hover:opacity-100"
                  title="Delete template"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Danger Zone — Clear Trade Data */}
      <div className="card border-accent-red/20">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-accent-red/10 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-4 h-4 text-accent-red" />
          </div>
          <div>
            <h2 className="text-sm font-semibold">Clear Trade Data</h2>
            <p className="text-xs text-dark-muted mt-0.5">
              Delete paper trading test data from Firestore. This cannot be undone.
            </p>
          </div>
        </div>

        {clearResult && (
          <div className={`flex items-center gap-2 p-3 rounded-lg mb-4 text-xs ${
            clearResult.type === 'success'
              ? 'bg-accent-green/10 text-accent-green border border-accent-green/20'
              : 'bg-accent-red/10 text-accent-red border border-accent-red/20'
          }`}>
            {clearResult.type === 'success'
              ? <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
              : <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />}
            {clearResult.msg}
          </div>
        )}

        <div className="space-y-2">
          {TRADE_COLLECTIONS.map((col) => {
            const isClearing = clearing === col.key
            return (
              <div key={col.key}
                className="flex items-center justify-between p-3.5 rounded-lg bg-dark-hover/30 border border-dark-border"
              >
                <div className="flex items-center gap-3">
                  <Database className="w-4 h-4 text-dark-muted" />
                  <div>
                    <div className="text-sm font-medium">{col.label}</div>
                    <div className="text-xs text-dark-muted mt-0.5">
                      <code className="text-[10px] bg-dark-hover/60 px-1.5 py-0.5 rounded">{col.key}</code>
                      <span className="ml-2">{col.desc}</span>
                    </div>
                  </div>
                </div>
                <button
                  disabled={!!clearing}
                  onClick={async () => {
                    if (!confirm(`Delete ALL documents in "${col.key}"? This cannot be undone.`)) return
                    setClearing(col.key)
                    setClearResult(null)
                    try {
                      const n = await deleteCollection(col.key)
                      setClearResult({ type: 'success', msg: `Deleted ${n} documents from ${col.key}` })
                    } catch (e) {
                      setClearResult({ type: 'error', msg: `Failed: ${e.message}` })
                    }
                    setClearing(null)
                    setTimeout(() => setClearResult(null), 5000)
                  }}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md bg-accent-red/10 text-accent-red border border-accent-red/20 hover:bg-accent-red/20 transition-colors disabled:opacity-40"
                >
                  {isClearing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                  {isClearing ? 'Deleting…' : 'Clear'}
                </button>
              </div>
            )
          })}
        </div>

        {/* Clear ALL */}
        <div className="mt-4 pt-4 border-t border-dark-border flex justify-end">
          <button
            disabled={!!clearing}
            onClick={async () => {
              if (!confirm('Delete ALL trade data (log, positions, daily summaries, cycles)? This cannot be undone.')) return
              setClearing('all')
              setClearResult(null)
              try {
                let total = 0
                for (const col of TRADE_COLLECTIONS) {
                  total += await deleteCollection(col.key)
                }
                setClearResult({ type: 'success', msg: `Cleared ${total} total documents across all trade collections` })
              } catch (e) {
                setClearResult({ type: 'error', msg: `Failed: ${e.message}` })
              }
              setClearing(null)
              setTimeout(() => setClearResult(null), 6000)
            }}
            className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-md bg-accent-red/15 text-accent-red border border-accent-red/30 hover:bg-accent-red/25 transition-colors disabled:opacity-40 font-medium"
          >
            {clearing === 'all' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
            {clearing === 'all' ? 'Clearing All…' : 'Clear All Trade Data'}
          </button>
        </div>
      </div>

      {/* Alert Configuration */}
      <AlertConfigSection />

      {/* System Info */}
      <div className="card">
        <SectionHeader
          icon={Info}
          title="System Info"
          subtitle="Architecture overview and pipeline stage reference"
        />
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Backend', value: 'Firestore (cloud)' },
              { label: 'Execution', value: 'Local Agent (desktop)' },
              { label: 'Auth', value: 'Google Sign-In (whitelist)' },
            ].map(({ label, value }) => (
              <div key={label} className="p-3 rounded-lg bg-dark-hover/30 border border-dark-border">
                <div className="text-xs text-dark-muted mb-1">{label}</div>
                <div className="text-xs font-medium">{value}</div>
              </div>
            ))}
          </div>
          <div className="space-y-2 pt-1">
            {Object.entries(PIPELINE_STAGES).map(([name, stages]) => (
              <div key={name} className="p-3 rounded-lg bg-dark-hover/20 border border-dark-border">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <ChevronRight className="w-3 h-3 text-accent-blue" />
                  <span className="text-xs font-semibold text-dark-text">{name}</span>
                </div>
                <p className="text-xs text-dark-muted leading-relaxed pl-4">{stages}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
