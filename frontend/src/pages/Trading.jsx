import { useState, useMemo, useEffect } from 'react'
import { db } from '../firebase'
import { collection, addDoc, getDocs, query, orderBy, limit as fbLimit, where } from 'firebase/firestore'
import {
  useTradingConfig,
  useTradeLog,
  useTradePositions,
  useTradeCycles,
  usePortfolioConfig,
  useBrokerAccount,
  useDailyTradeSummary,
  useDailyReviews,
  useBrokerOrders,
  useRuns,
  useBacktestTrades,
} from '../hooks/useFirestore'
import {
  Bot,
  Power,
  PowerOff,
  Play,
  ShieldCheck,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  ArrowUpCircle,
  ArrowDownCircle,
  Clock,
  DollarSign,
  Activity,
  BookOpen,
  BarChart3,
  Settings,
  Calendar,
  XCircle,
  RefreshCw,
  Eye,
  Zap,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

const TABS = ['overview', 'orders', 'positions', 'cycles', 'comparison']

export default function Trading() {
  const [tradingConfig, configLoading, updateConfig] = useTradingConfig()
  const [tradeLog]      = useTradeLog(100)
  const [positions]     = useTradePositions()
  const [cycles]        = useTradeCycles(20)
  const [portfolioCfg]  = usePortfolioConfig()
  const [brokerAccount, brokerAccountLoading, brokerAccountError] = useBrokerAccount()
  const [dailySummary]  = useDailyTradeSummary()
  const [dailyReviews]  = useDailyReviews(7)
  const [brokerOrders]  = useBrokerOrders()
  const [tab, setTab]   = useState('overview')
  const [submitting, setSubmitting] = useState(false)
  const [syncAlpacaLoading, setSyncAlpacaLoading] = useState(false)
  const [syncAlpacaError, setSyncAlpacaError] = useState(null)

  const handleSyncAlpaca = async () => {
    setSyncAlpacaLoading(true)
    setSyncAlpacaError(null)
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'sync_broker_orders',
        status: 'queued',
        description: 'Sync from Alpaca',
        stocks: [],
        config: {},
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: new Date().toISOString(),
        startedAt: null,
        completedAt: null,
        result: null,
        error: null,
      })
      setSyncAlpacaError(null)
      // LocalAgent will pick it up and sync; Firestore listener will update UI
    } catch (err) {
      setSyncAlpacaError(err.message || 'Failed to queue sync')
    } finally {
      setSyncAlpacaLoading(false)
    }
  }

  const openPositions = positions.filter((p) => p.qty > 0)
  const todayStr = new Date().toISOString().slice(0, 10)
  const todayOrders = tradeLog.filter((o) => (o.createdAt || '').startsWith(todayStr))
  const latestCycle = cycles[0]

  const totalUnrealizedPL = openPositions.reduce((s, p) => s + (p.unrealizedPL || 0), 0)
  const totalMarketValue  = openPositions.reduce((s, p) => s + (p.marketValue  || 0), 0)

  const toggleEnabled = () => updateConfig({ enabled: !tradingConfig.enabled })
  const setMode = (mode) => updateConfig({ mode })
  const setScheduleMode = (mode) => updateConfig({ scheduleMode: mode })

  const triggerCycle = async (interval) => {
    setSubmitting(true)
    const taskType = interval === '1h' ? 'trading_execute_1h' : 'trading_execute_1d'
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: taskType,
        status: 'queued',
        description: `Manual: Trading bot cycle [${interval}] (${tradingConfig.mode})`,
        stocks: [],
        config: { interval },
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: null, startedAt: null, completedAt: null,
        result: null, error: null,
      })
      alert(`Trading cycle [${interval}] queued!`)
    } catch (err) {
      alert('Failed to queue: ' + err.message)
    }
    setSubmitting(false)
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Bot className="w-5 h-5 text-accent-blue" /> Trading Bot
          </h1>
          <p className="text-dark-muted text-sm mt-0.5">
            Live order execution powered by ML pipeline signals
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => triggerCycle('1h')}
            disabled={submitting}
            className="btn-secondary flex items-center gap-2 text-sm px-3 py-2"
          >
            <Play className="w-4 h-4" />
            {submitting ? 'Queuing...' : 'Run 1h'}
          </button>
          <button
            onClick={() => triggerCycle('1d')}
            disabled={submitting}
            className="btn-primary flex items-center gap-2 text-sm px-3 py-2"
          >
            <Play className="w-4 h-4" />
            {submitting ? 'Queuing...' : 'Run 1d'}
          </button>
        </div>
      </div>

      {/* Live Portfolio Value — always visible, from Alpaca */}
      <LivePortfolioBar brokerAccount={brokerAccount} />

      {/* Status cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatusCard
          label="Trading"
          value={tradingConfig.enabled ? 'ENABLED' : 'DISABLED'}
          icon={tradingConfig.enabled ? Power : PowerOff}
          color={tradingConfig.enabled ? 'text-accent-green' : 'text-accent-red'}
          onClick={toggleEnabled}
          clickLabel={tradingConfig.enabled ? 'Click to disable' : 'Click to enable'}
        />
        <StatusCard
          label="Mode"
          value={tradingConfig.mode?.toUpperCase() || 'PAPER'}
          icon={tradingConfig.mode === 'live' ? AlertTriangle : ShieldCheck}
          color={tradingConfig.mode === 'live' ? 'text-accent-yellow' : 'text-accent-blue'}
        />
        <StatusCard
          label="Open Positions"
          value={openPositions.length}
          icon={Activity}
          color="text-accent-blue"
          sub={`$${totalMarketValue.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
        />
        <StatusCard
          label="Day P&L (Alpaca)"
          value={brokerAccount ? `$${(brokerAccount.dayPL ?? 0) >= 0 ? '+' : ''}${(brokerAccount.dayPL ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
          icon={brokerAccount && (brokerAccount.dayPL ?? 0) >= 0 ? TrendingUp : TrendingDown}
          color={brokerAccount ? ((brokerAccount.dayPL ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red') : 'text-dark-muted'}
          sub={brokerAccount?.updatedAt ? `Updated ${new Date(brokerAccount.updatedAt).toLocaleTimeString()}` : 'Updates automatically'}
        />
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-dark-muted" />
            <span className="text-sm text-dark-muted">Broker Mode:</span>
            {['paper', 'live', 'dry_run'].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  tradingConfig.mode === m
                    ? m === 'live'
                      ? 'bg-accent-red/15 text-accent-red border border-accent-red/40'
                      : 'bg-accent-blue/15 text-accent-blue border border-accent-blue/40'
                    : 'bg-dark-hover/50 text-dark-muted border border-dark-border hover:border-dark-muted'
                }`}
              >
                {m === 'dry_run' ? 'Dry Run' : m.charAt(0).toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-dark-muted" />
            <span className="text-sm text-dark-muted">Schedule:</span>
            {[
              { value: 'queued', label: 'Queued', desc: 'Tasks go to queue, approve in LocalAgent' },
              { value: 'automatic', label: 'Automatic', desc: 'Runs on schedule, no approval needed' },
            ].map(({ value, label, desc }) => (
              <button
                key={value}
                onClick={() => setScheduleMode(value)}
                title={desc}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  tradingConfig.scheduleMode === value
                    ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/40'
                    : 'bg-dark-hover/50 text-dark-muted border border-dark-border hover:border-dark-muted'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-dark-muted">Kill Switch:</span>
            <button
              onClick={toggleEnabled}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                tradingConfig.enabled ? 'bg-accent-green' : 'bg-dark-border'
              }`}
            >
              <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                tradingConfig.enabled ? 'left-6' : 'left-0.5'
              }`} />
            </button>
          </div>
        </div>

        {tradingConfig.mode === 'live' && (
          <div className="mt-3 p-3 bg-accent-red/10 border border-accent-red/30 rounded-lg flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-accent-red flex-shrink-0 mt-0.5" />
            <p className="text-xs text-accent-red">
              <strong>LIVE MODE:</strong> Orders will execute with real money on your broker account.
              Ensure you have sufficient funds and understand the risks.
            </p>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-dark-border">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize transition-colors ${
              tab === t
                ? 'text-accent-blue border-b-2 border-accent-blue'
                : 'text-dark-muted hover:text-dark-text'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'overview' && <OverviewTab latestCycle={latestCycle} todayOrders={todayOrders} totalFund={portfolioCfg.totalFund} brokerAccount={brokerAccount} openPositions={openPositions} dailySummary={dailySummary} dailyReviews={dailyReviews} onSyncAlpaca={handleSyncAlpaca} syncAlpacaLoading={syncAlpacaLoading} syncAlpacaError={syncAlpacaError} brokerAccountError={brokerAccountError} />}
      {tab === 'orders'   && <OrdersTab orders={tradeLog} brokerOrders={brokerOrders} />}
      {tab === 'positions' && <PositionsTab positions={openPositions} />}
      {tab === 'cycles'   && <CyclesTab cycles={cycles} />}
      {tab === 'comparison' && <ComparisonTab />}
    </div>
  )
}

/* ─── Status Card ──────────────────────────────────────────────────── */
function StatusCard({ label, value, icon: Icon, color, sub, onClick, clickLabel }) {
  return (
    <div
      className={`card flex items-center gap-3 ${onClick ? 'cursor-pointer hover:border-dark-muted' : ''}`}
      onClick={onClick}
      title={clickLabel}
    >
      <div className={`p-2 rounded-lg bg-dark-hover ${color}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div>
        <div className="text-xs text-dark-muted">{label}</div>
        <div className={`text-sm font-bold ${color}`}>{value}</div>
        {sub && <div className="text-[10px] text-dark-muted">{sub}</div>}
      </div>
    </div>
  )
}

/* ─── Live Portfolio Bar (always visible) ──────────────────────────── */
function LivePortfolioBar({ brokerAccount }) {
  const eq = brokerAccount?.equity ?? brokerAccount?.portfolioValue
  const dayPL = brokerAccount?.dayPL
  const dayPLPct = brokerAccount?.dayPLPct
  return (
    <div className="card border-accent-blue/30 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
      <div className="flex flex-wrap items-center gap-4 sm:gap-6">
        <div>
          <div className="text-[10px] text-dark-muted uppercase tracking-wide">Live Portfolio Value (Alpaca)</div>
          <div className="text-lg font-bold font-mono text-accent-blue">
            {eq != null ? `$${eq.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
          </div>
        </div>
        {eq != null && (
          <div>
            <div className="text-[10px] text-dark-muted uppercase tracking-wide">Day P&L</div>
            <div className={`text-sm font-bold font-mono ${(dayPL ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
              {(dayPL ?? 0) >= 0 ? '+' : ''}${(dayPL ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              <span className="text-xs font-normal ml-1">({(dayPLPct ?? 0) >= 0 ? '+' : ''}{(dayPLPct ?? 0).toFixed(2)}%)</span>
            </div>
          </div>
        )}
        {brokerAccount?.updatedAt ? (
          <div className="text-xs text-dark-muted">
            Updated {new Date(brokerAccount.updatedAt).toLocaleString()} · auto-refreshes every 15 min with LocalAgent
          </div>
        ) : (
          <div className="text-xs text-dark-muted">Run SyncAlpaca.bat or LocalAgent to load data</div>
        )}
      </div>
    </div>
  )
}

/* ─── Overview Tab ─────────────────────────────────────────────────── */
function OverviewTab({ latestCycle, todayOrders, totalFund, brokerAccount, openPositions, dailySummary, dailyReviews, onSyncAlpaca, syncAlpacaLoading, syncAlpacaError, brokerAccountError }) {
  const liveEquity = brokerAccount?.equity ?? brokerAccount?.portfolioValue
  const liveDayPL = brokerAccount?.dayPL
  const liveDayPLPct = brokerAccount?.dayPLPct
  const displayFund = liveEquity != null ? liveEquity : totalFund

  return (
    <div className="space-y-4">
      {/* Live Portfolio Value — primary display from Alpaca (updates automatically) */}
      <div className="card border-accent-blue/40 bg-accent-blue/5">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-accent-blue" /> Live Portfolio Value
            </h3>
            <p className="text-xs text-dark-muted mt-0.5">
              From Alpaca broker · Sync now or run LocalAgent for auto-updates every 15 min
            </p>
          </div>
          <button
            type="button"
            onClick={onSyncAlpaca}
            disabled={syncAlpacaLoading}
            className="btn-sm bg-accent-blue/20 hover:bg-accent-blue/30 text-accent-blue flex items-center gap-1.5 disabled:opacity-60"
          >
            <RefreshCw className={`w-4 h-4 ${syncAlpacaLoading ? 'animate-spin' : ''}`} />
            {syncAlpacaLoading ? 'Queuing…' : 'Sync from Alpaca'}
          </button>
        </div>
        {(syncAlpacaError || brokerAccountError) && (
          <p className="text-sm text-accent-red mb-3">
            {brokerAccountError || syncAlpacaError}
            {brokerAccountError && String(brokerAccountError).toLowerCase().includes('permission') && ' Sign in with harieduam@gmail.com (see firestore.rules).'}
          </p>
        )}
        {brokerAccount ? (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide">Portfolio Value</div>
              <div className="text-xl font-bold font-mono text-accent-blue">
                ${(liveEquity ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide">Cash</div>
              <div className="text-lg font-mono">
                ${(brokerAccount.cash ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide">Day P&L (Alpaca)</div>
              <div className={`text-lg font-bold font-mono ${(liveDayPL ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                ${(liveDayPL ?? 0) >= 0 ? '+' : ''}{(liveDayPL ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className={`text-xs ${(liveDayPLPct ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                {(liveDayPLPct ?? 0) >= 0 ? '+' : ''}{(liveDayPLPct ?? 0).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide">Buying Power</div>
              <div className="text-lg font-mono">
                ${(brokerAccount.buyingPower ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-dark-muted uppercase tracking-wide">Last Updated</div>
              <div className="text-sm">
                {brokerAccount.updatedAt ? new Date(brokerAccount.updatedAt).toLocaleString() : '—'}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-dark-muted space-y-3">
            <p className="font-medium">No Alpaca data yet.</p>
            <p className="text-sm">Click <strong>Sync from Alpaca</strong> above to queue a sync. LocalAgent will process it when running.</p>
            {brokerAccountError && (
              <p className="text-xs text-accent-red">
                Firestore error: {brokerAccountError}. Check console for details. If it says &quot;permission-denied&quot;, ensure you are signed in with the email allowed in firestore.rules (harieduam@gmail.com).
              </p>
            )}
          </div>
        )}
      </div>

      {/* Latest cycle summary */}
      {latestCycle ? (
        <div className="card">
          <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-dark-muted" /> Latest Cycle
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
            <Stat label="Interval" value={latestCycle.interval || '1d'} />
            <Stat label="Cycle ID" value={latestCycle.cycleId?.slice(-10) || '—'} />
            <Stat label="Mode" value={latestCycle.brokerMode?.toUpperCase() || '—'} />
            <Stat
              label="Orders"
              value={`${latestCycle.numFilled || 0} filled / ${latestCycle.numRejected || 0} rejected`}
              sub={
                (latestCycle.numRejected || 0) > 0
                  ? 'Check Orders tab → Order History for rejection details.'
                  : undefined
              }
            />
            <Stat
              label="Day P&L"
              value={`$${(latestCycle.dayPL || 0) >= 0 ? '+' : ''}${(latestCycle.dayPL || 0).toFixed(2)}`}
              color={(latestCycle.dayPL || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}
            />
            <Stat label="Time" value={latestCycle.timestamp ? new Date(latestCycle.timestamp).toLocaleString() : '—'} />
          </div>
        </div>
      ) : (
        <div className="card text-center py-8">
          <Bot className="w-8 h-8 text-dark-muted mx-auto mb-2" />
          <p className="text-dark-muted text-sm">No trading cycles yet. Run your first cycle to get started.</p>
        </div>
      )}

      {/* Next Day Daily Trade Summary */}
      <div className="card">
        <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
          <Calendar className="w-4 h-4 text-dark-muted" /> Next Day Daily Trade Summary
        </h3>
        <p className="text-xs text-dark-muted mb-3">
          Planned trades from the last daily (1d) cycle — executed at next market open.
        </p>
        {dailySummary?.trades?.length > 0 ? (
          <div className="space-y-1.5">
            <div className="text-[10px] text-dark-muted mb-1">
              Updated: {dailySummary.updatedAt ? new Date(dailySummary.updatedAt).toLocaleString() : '—'} • Cycle: {dailySummary.cycleId?.slice(-12) || '—'}
            </div>
            {dailySummary.trades.map((t, i) => (
              <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-dark-border/50 last:border-0">
                <span className={`px-1.5 py-0.5 rounded font-mono font-bold ${
                  t.signal === 'BUY' ? 'bg-accent-green/15 text-accent-green' : t.signal === 'SELL' ? 'bg-accent-red/15 text-accent-red' : 'bg-dark-hover text-dark-muted'
                }`}>
                  {t.signal}
                </span>
                <span className="font-mono font-medium">{t.symbol}</span>
                <span className="text-dark-muted">@ ${(t.price || 0).toFixed(2)}</span>
                {t.executed && <span className="text-dark-muted">qty {t.qty}</span>}
                {t.ghost && <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-purple/15 text-accent-purple">Ghost</span>}
                <span className={`ml-auto px-1.5 py-0.5 rounded text-[10px] ${t.status === 'filled' ? 'bg-accent-green/15 text-accent-green' : t.status === 'ghost' ? 'bg-accent-purple/15 text-accent-purple' : 'bg-dark-hover text-dark-muted'}`}>
                  {t.status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-dark-muted text-xs">No daily summary yet. Run a daily cycle to generate.</p>
        )}
      </div>

      {/* Daily Performance Review */}
      <DailyReviewCard reviews={dailyReviews} />

      {/* Today's activity */}
      <div className="card">
        <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-dark-muted" /> Today's Activity
        </h3>
        {todayOrders.length > 0 ? (
          <div className="space-y-1.5">
            {todayOrders.slice(0, 10).map((o, i) => (
              <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-dark-border/50 last:border-0">
                <span className={`px-1.5 py-0.5 rounded font-mono font-bold ${
                  o.side === 'buy' ? 'bg-accent-green/15 text-accent-green' : 'bg-accent-red/15 text-accent-red'
                }`}>
                  {o.side?.toUpperCase()}
                </span>
                <span className="font-mono font-medium">{o.qty} {o.symbol}</span>
                <span className="text-dark-muted">@ ${(o.filledPrice || 0).toFixed(2)}</span>
                <span className={`ml-auto px-1.5 py-0.5 rounded text-[10px] ${
                  ['filled', 'submitted', 'accepted', 'new', 'dry_run'].includes(o.status)
                    ? 'bg-accent-green/15 text-accent-green'
                    : o.status === 'rejected' ? 'bg-accent-red/15 text-accent-red'
                    : o.status === 'partially_filled' ? 'bg-accent-blue/15 text-accent-blue'
                    : ['pending_new', 'held'].includes(o.status) ? 'bg-accent-yellow/15 text-accent-yellow'
                    : 'bg-dark-hover text-dark-muted'
                }`}>{o.status}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-dark-muted text-xs">No orders today</p>
        )}
      </div>

      {/* Portfolio summary — Alpaca values when available */}
      <div className="card">
        <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
          <DollarSign className="w-4 h-4 text-dark-muted" /> Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat
            label="Portfolio Value"
            value={`$${(displayFund || totalFund || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            sub={liveEquity != null ? 'from Alpaca' : 'from config'}
          />
          <Stat label="Open Positions" value={openPositions.length} />
          <Stat
            label="Day P&L"
            value={liveEquity != null
              ? `${(liveDayPL ?? 0) >= 0 ? '+' : ''}$${(liveDayPL ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${(liveDayPLPct ?? 0) >= 0 ? '+' : ''}${(liveDayPLPct ?? 0).toFixed(2)}%)`
              : '—'
            }
            color={liveEquity != null ? ((liveDayPL ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red') : undefined}
            sub={liveEquity != null ? 'from Alpaca' : undefined}
          />
          <Stat
            label="Est. Market Value (positions)"
            value={`$${openPositions.reduce((s, p) => s + (p.marketValue || 0), 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            sub=" tracked"
          />
        </div>
        <p className="text-[10px] text-dark-muted mt-2">
          P&L and Portfolio Value from Alpaca. Updates automatically every 15 min.
        </p>
      </div>
    </div>
  )
}

/* ─── Daily Review Card ───────────────────────────────────────────── */
function DailyReviewCard({ reviews }) {
  const [expanded, setExpanded] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  const latest = reviews?.[0]
  if (!latest) {
    return (
      <div className="card">
        <h3 className="text-sm font-medium flex items-center gap-2 mb-2">
          <Eye className="w-4 h-4 text-dark-muted" /> Daily Performance Review
        </h3>
        <p className="text-dark-muted text-xs">No reviews yet. The review runs automatically after each daily trading cycle.</p>
      </div>
    )
  }

  const plColor = (latest.portfolioPLPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'
  const slots = latest.slotBreakdown || []
  const actions = latest.actionsTaken || []
  const patternActions = latest.patternActions || []
  const ghostWatch = latest.ghostWatch || []
  const suggestions = latest.suggestions || []
  const history = reviews?.slice(1) || []

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium flex items-center gap-2">
          <Eye className="w-4 h-4 text-accent-blue" /> Daily Performance Review
        </h3>
        <span className="text-[10px] text-dark-muted">{latest.date}</span>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-3">
        <div className="text-center">
          <div className="text-[10px] text-dark-muted">Day P&L</div>
          <div className={`text-sm font-bold ${plColor}`}>
            {(latest.portfolioPLPct || 0) >= 0 ? '+' : ''}{(latest.portfolioPLPct || 0).toFixed(2)}%
          </div>
          <div className={`text-[10px] ${plColor}`}>${(latest.portfolioPL || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-muted">Equity</div>
          <div className="text-sm font-bold">${(latest.equity || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-muted">Active</div>
          <div className="text-sm font-bold">{latest.activeSlots || 0} slots</div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-muted">Ghost</div>
          <div className="text-sm font-bold text-accent-purple">{latest.ghostSlotCount || 0} slots</div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-muted">Filled</div>
          <div className="text-sm font-bold">{latest.numFilled || 0} / {latest.numOrders || 0}</div>
        </div>
      </div>

      {/* Actions taken */}
      {actions.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] font-semibold text-dark-muted uppercase tracking-wider mb-1">Actions Taken</div>
          <div className="space-y-1">
            {actions.map((a, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  a.action === 'phased_out' ? 'bg-accent-red/15 text-accent-red'
                    : a.action === 'reduced' ? 'bg-accent-yellow/15 text-accent-yellow'
                    : 'bg-accent-green/15 text-accent-green'
                }`}>{a.action?.replace('_', ' ').toUpperCase()}</span>
                <span className="font-mono">{a.slot}</span>
                <span className="text-dark-muted">{a.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pattern actions */}
      {patternActions.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] font-semibold text-dark-muted uppercase tracking-wider mb-1">Pattern Changes</div>
          <div className="space-y-1">
            {patternActions.map((p, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-accent-red/15 text-accent-red">SUPERSEDED</span>
                <span className="font-mono">{p.symbol}/{p.interval}</span>
                <span className="text-dark-muted">accuracy {((p.liveAccuracy || 0) * 100).toFixed(0)}% over {p.triggers} triggers</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ghost watch */}
      {ghostWatch.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] font-semibold text-dark-muted uppercase tracking-wider mb-1">Ghost Slot Watch</div>
          <div className="space-y-1">
            {ghostWatch.map((g, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-purple/15 text-accent-purple">GHOST</span>
                <span className="font-mono">{g.slot}</span>
                <span className={g.recentReturn >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                  {g.recentReturn >= 0 ? '+' : ''}{(g.recentReturn || 0).toFixed(1)}% recent
                </span>
                <span className="text-dark-muted">
                  {((g.recentWinRate || 0) * 100).toFixed(0)}% win rate / {g.recentTrades} trades
                </span>
                {g.nearRestore && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-green/15 text-accent-green">NEAR RESTORE</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expandable: full slot breakdown */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-[10px] text-accent-blue flex items-center gap-1 mb-2 hover:underline"
      >
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        {expanded ? 'Hide' : 'Show'} slot breakdown ({slots.length} slots)
      </button>
      {expanded && slots.length > 0 && (
        <div className="overflow-x-auto mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-dark-muted text-[10px] border-b border-dark-border">
                <th className="text-left py-1 pr-3">Slot</th>
                <th className="text-right py-1 px-2">Alloc%</th>
                <th className="text-right py-1 px-2">Return</th>
                <th className="text-right py-1 px-2">Win Rate</th>
                <th className="text-right py-1 px-2">Sharpe</th>
                <th className="text-right py-1 px-2">Trades</th>
                <th className="text-center py-1 px-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {slots.map((s, i) => (
                <tr key={i} className="border-b border-dark-border/30">
                  <td className="py-1 pr-3 font-mono">{s.slot}</td>
                  <td className="text-right py-1 px-2">{(s.allocation || 0).toFixed(1)}%</td>
                  <td className={`text-right py-1 px-2 ${(s.recentReturn || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {(s.recentReturn || 0) >= 0 ? '+' : ''}{(s.recentReturn || 0).toFixed(2)}%
                  </td>
                  <td className="text-right py-1 px-2">{((s.winRate || 0) * 100).toFixed(0)}%</td>
                  <td className={`text-right py-1 px-2 ${(s.recentSharpe || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {(s.recentSharpe || 0).toFixed(2)}
                  </td>
                  <td className="text-right py-1 px-2">{s.totalTrades || 0}</td>
                  <td className="text-center py-1 px-2">
                    {s.isGhost
                      ? <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-purple/15 text-accent-purple">Ghost</span>
                      : <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent-green/15 text-accent-green">Active</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] font-semibold text-dark-muted uppercase tracking-wider mb-1">Suggestions</div>
          <div className="space-y-1">
            {suggestions.map((s, i) => (
              <div key={i} className="text-xs text-dark-muted flex items-start gap-1.5">
                <Zap className="w-3 h-3 text-accent-yellow mt-0.5 flex-shrink-0" />
                <span>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History toggle */}
      {history.length > 0 && (
        <>
          <button
            onClick={() => setHistoryOpen(!historyOpen)}
            className="text-[10px] text-accent-blue flex items-center gap-1 hover:underline"
          >
            {historyOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {historyOpen ? 'Hide' : 'Show'} previous reviews ({history.length})
          </button>
          {historyOpen && (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-dark-muted text-[10px] border-b border-dark-border">
                    <th className="text-left py-1 pr-3">Date</th>
                    <th className="text-right py-1 px-2">P&L %</th>
                    <th className="text-right py-1 px-2">Active</th>
                    <th className="text-right py-1 px-2">Ghost</th>
                    <th className="text-right py-1 px-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((r, i) => (
                    <tr key={i} className="border-b border-dark-border/30">
                      <td className="py-1 pr-3 font-mono">{r.date}</td>
                      <td className={`text-right py-1 px-2 ${(r.portfolioPLPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {(r.portfolioPLPct || 0) >= 0 ? '+' : ''}{(r.portfolioPLPct || 0).toFixed(2)}%
                      </td>
                      <td className="text-right py-1 px-2">{r.activeSlots || 0}</td>
                      <td className="text-right py-1 px-2">{r.ghostSlotCount || 0}</td>
                      <td className="text-right py-1 px-2">{(r.actionsTaken || []).length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}

/* ─── Orders Tab (Order Book) ──────────────────────────────────────── */
function OrdersTab({ orders, brokerOrders }) {
  const [selected, setSelected] = useState(new Set())
  const [cancelling, setCancelling] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [view, setView] = useState('broker')

  const openBrokerOrders = useMemo(
    () => brokerOrders.filter((o) => ['new', 'partially_filled', 'accepted', 'pending_new', 'held'].includes(o.status)),
    [brokerOrders],
  )

  const toggleSelect = (id) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleAll = () => {
    if (selected.size === openBrokerOrders.length) setSelected(new Set())
    else setSelected(new Set(openBrokerOrders.map((o) => o.orderId)))
  }

  const handleCancelSelected = async () => {
    if (!selected.size) return
    setCancelling(true)
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'cancel_orders',
        status: 'queued',
        description: `Cancel ${selected.size} order(s)`,
        stocks: [],
        config: { orderIds: [...selected] },
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: new Date().toISOString(),
        startedAt: null, completedAt: null, result: null, error: null,
      })
      setSelected(new Set())
    } catch (err) {
      alert('Failed to queue cancel: ' + err.message)
    }
    setCancelling(false)
  }

  const handleCancelAll = async () => {
    if (!openBrokerOrders.length) return
    if (!confirm(`Cancel ALL ${openBrokerOrders.length} open orders?`)) return
    setCancelling(true)
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'cancel_orders',
        status: 'queued',
        description: `Cancel ALL open orders (${openBrokerOrders.length})`,
        stocks: [],
        config: {},
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: new Date().toISOString(),
        startedAt: null, completedAt: null, result: null, error: null,
      })
    } catch (err) {
      alert('Failed to queue cancel: ' + err.message)
    }
    setCancelling(false)
  }

  const handleSync = async () => {
    setSyncing(true)
    try {
      await addDoc(collection(db, 'run_commands'), {
        type: 'sync_broker_orders',
        status: 'queued',
        description: 'Sync broker orders',
        stocks: [],
        config: {},
        source: 'website',
        createdAt: new Date().toISOString(),
        approvedAt: new Date().toISOString(),
        startedAt: null, completedAt: null, result: null, error: null,
      })
    } catch (err) {
      alert('Failed to queue sync: ' + err.message)
    }
    setTimeout(() => setSyncing(false), 2000)
  }

  const statusColor = (s) => {
    if (s === 'filled') return 'bg-accent-green/15 text-accent-green'
    if (s === 'canceled' || s === 'expired') return 'bg-dark-hover text-dark-muted'
    if (s === 'rejected') return 'bg-accent-red/15 text-accent-red'
    if (s === 'new' || s === 'accepted' || s === 'pending_new') return 'bg-accent-yellow/15 text-accent-yellow'
    if (s === 'partially_filled') return 'bg-accent-blue/15 text-accent-blue'
    if (s === 'dry_run') return 'bg-accent-purple/15 text-accent-purple'
    return 'bg-dark-hover text-dark-muted'
  }

  return (
    <div className="space-y-4">
      {/* View toggle + actions */}
      <div className="flex items-center gap-3">
        <div className="flex gap-1">
          {['broker', 'history'].map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all capitalize ${
                view === v
                  ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/40'
                  : 'bg-dark-hover/50 text-dark-muted border border-dark-border hover:border-dark-muted'
              }`}
            >
              {v === 'broker' ? 'Broker Orders' : 'Order History'}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {view === 'broker' && (
            <>
              <button
                onClick={handleSync}
                disabled={syncing}
                className="btn-secondary flex items-center gap-1.5 text-xs px-3 py-1.5"
              >
                <RefreshCw className={`w-3 h-3 ${syncing ? 'animate-spin' : ''}`} />
                {syncing ? 'Syncing...' : 'Sync'}
              </button>
              {openBrokerOrders.length > 0 && (
                <>
                  <button
                    onClick={handleCancelSelected}
                    disabled={cancelling || selected.size === 0}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-medium border transition-all bg-accent-red/10 text-accent-red border-accent-red/30 hover:bg-accent-red/20 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <XCircle className="w-3 h-3" />
                    Cancel {selected.size} selected
                  </button>
                  <button
                    onClick={handleCancelAll}
                    disabled={cancelling}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-medium border transition-all bg-accent-red/10 text-accent-red border-accent-red/30 hover:bg-accent-red/20 disabled:opacity-40"
                  >
                    Cancel All
                  </button>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {/* Broker orders view */}
      {view === 'broker' && (
        <div className="card overflow-x-auto">
          <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-dark-muted" /> Broker Orders
            {openBrokerOrders.length > 0 && (
              <span className="text-[10px] bg-accent-yellow/15 text-accent-yellow px-1.5 py-0.5 rounded font-bold">
                {openBrokerOrders.length} open
              </span>
            )}
          </h3>
          {!brokerOrders.length ? (
            <div className="text-center py-8">
              <BookOpen className="w-8 h-8 text-dark-muted mx-auto mb-2" />
              <p className="text-dark-muted text-sm">No broker orders synced yet</p>
              <p className="text-dark-muted text-[10px] mt-1">Click "Sync" to fetch orders from your broker</p>
            </div>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-dark-muted border-b border-dark-border">
                  <th className="py-2 px-2 w-8">
                    <input
                      type="checkbox"
                      checked={selected.size === openBrokerOrders.length && openBrokerOrders.length > 0}
                      onChange={toggleAll}
                      className="accent-accent-blue"
                    />
                  </th>
                  <th className="text-left py-2 px-2">Symbol</th>
                  <th className="text-left py-2 px-2">Side</th>
                  <th className="text-left py-2 px-2">Type</th>
                  <th className="text-right py-2 px-2">Qty</th>
                  <th className="text-right py-2 px-2">Filled</th>
                  <th className="text-right py-2 px-2">Fill Price</th>
                  <th className="text-right py-2 px-2">Limit</th>
                  <th className="text-left py-2 px-2">Status</th>
                  <th className="text-left py-2 px-2">Time</th>
                </tr>
              </thead>
              <tbody>
                {brokerOrders
                  .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''))
                  .map((o) => {
                    const isOpen = ['new', 'partially_filled', 'accepted', 'pending_new', 'held'].includes(o.status)
                    return (
                      <tr key={o.orderId} className={`border-b border-dark-border/30 hover:bg-dark-hover/30 ${isOpen ? 'bg-accent-yellow/[0.03]' : ''}`}>
                        <td className="py-1.5 px-2">
                          {isOpen && (
                            <input
                              type="checkbox"
                              checked={selected.has(o.orderId)}
                              onChange={() => toggleSelect(o.orderId)}
                              className="accent-accent-blue"
                            />
                          )}
                        </td>
                        <td className="py-1.5 px-2 font-mono font-medium">{o.symbol}</td>
                        <td className="py-1.5 px-2">
                          <span className={`flex items-center gap-1 ${o.side === 'buy' ? 'text-accent-green' : 'text-accent-red'}`}>
                            {o.side === 'buy' ? <ArrowUpCircle className="w-3 h-3" /> : <ArrowDownCircle className="w-3 h-3" />}
                            {o.side?.toUpperCase()}
                          </span>
                        </td>
                        <td className="py-1.5 px-2 capitalize">{o.orderType}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{o.qty}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{o.filledQty || 0}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{o.filledAvgPrice ? `$${o.filledAvgPrice.toFixed(2)}` : '—'}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{o.limitPrice ? `$${o.limitPrice.toFixed(2)}` : '—'}</td>
                        <td className="py-1.5 px-2">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] ${statusColor(o.status)}`}>
                            {o.status}
                          </span>
                        </td>
                        <td className="py-1.5 px-2 text-dark-muted whitespace-nowrap">
                          {o.createdAt ? new Date(o.createdAt).toLocaleString() : '—'}
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Order history view (existing trade_log) */}
      {view === 'history' && (
        <div className="card overflow-x-auto">
          <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-dark-muted" /> Order History (Trade Log)
          </h3>
          {!orders.length ? (
            <div className="text-center py-8">
              <BookOpen className="w-8 h-8 text-dark-muted mx-auto mb-2" />
              <p className="text-dark-muted text-sm">No orders in trade log yet</p>
              <p className="text-dark-muted text-[10px] mt-1">
                Run a trading cycle (paper/live) to record orders. Sync broker to see Alpaca orders.
              </p>
            </div>
          ) : (
            <table className="w-full text-xs">
            <thead>
              <tr className="text-dark-muted border-b border-dark-border">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Symbol</th>
                <th className="text-left py-2 px-2">Side</th>
                <th className="text-right py-2 px-2">Qty</th>
                <th className="text-right py-2 px-2">Price</th>
                <th className="text-right py-2 px-2">Total</th>
                <th className="text-left py-2 px-2">Signal</th>
                <th className="text-right py-2 px-2">Conf</th>
                <th className="text-left py-2 px-2">Status</th>
                <th className="text-left py-2 px-2">Rejection Reason</th>
                <th className="text-left py-2 px-2">Mode</th>
              </tr>
            </thead>
              <tbody>
                {orders.map((o, i) => (
                  <tr key={i} className="border-b border-dark-border/30 hover:bg-dark-hover/30">
                    <td className="py-1.5 px-2 text-dark-muted whitespace-nowrap">
                      {o.createdAt ? new Date(o.createdAt).toLocaleString() : '—'}
                    </td>
                    <td className="py-1.5 px-2 font-mono font-medium">{o.symbol}</td>
                    <td className="py-1.5 px-2">
                      <span className={`flex items-center gap-1 ${o.side === 'buy' ? 'text-accent-green' : 'text-accent-red'}`}>
                        {o.side === 'buy' ? <ArrowUpCircle className="w-3 h-3" /> : <ArrowDownCircle className="w-3 h-3" />}
                        {o.side?.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono">{o.qty}</td>
                    <td className="py-1.5 px-2 text-right font-mono">${(o.filledPrice || 0).toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-right font-mono">${(o.totalCost || 0).toFixed(0)}</td>
                    <td className="py-1.5 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        o.signal === 'BUY' ? 'bg-accent-green/15 text-accent-green'
                          : o.signal === 'SELL' ? 'bg-accent-red/15 text-accent-red'
                          : 'bg-dark-hover text-dark-muted'
                      }`}>{o.signal || '—'}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono">{((o.confidence || 0) * 100).toFixed(0)}%</td>
                    <td className="py-1.5 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] ${statusColor(o.status)}`}>
                        {o.status}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 text-[10px] text-dark-muted max-w-[200px] truncate" title={o.rejectionMessage}>
                      {o.status === 'rejected' && o.rejectionMessage ? o.rejectionMessage : '—'}
                    </td>
                    <td className="py-1.5 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                        o.brokerMode === 'live' ? 'bg-accent-red/15 text-accent-red'
                          : o.brokerMode === 'paper' ? 'bg-accent-blue/15 text-accent-blue'
                          : 'bg-dark-hover text-dark-muted'
                      }`}>{o.brokerMode}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Positions Tab ────────────────────────────────────────────────── */
function PositionsTab({ positions }) {
  if (!positions.length) {
    return (
      <div className="card text-center py-8">
        <BarChart3 className="w-8 h-8 text-dark-muted mx-auto mb-2" />
        <p className="text-dark-muted text-sm">No open positions</p>
        <p className="text-dark-muted text-[10px] mt-1">
          Positions sync from Alpaca every 15 min. Run a sync in Orders tab or wait for auto-sync.
        </p>
      </div>
    )
  }

  return (
    <div className="card overflow-x-auto">
      <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
        <BarChart3 className="w-4 h-4 text-dark-muted" /> Open Positions
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-dark-muted border-b border-dark-border">
            <th className="text-left py-2 px-2">Symbol</th>
            <th className="text-right py-2 px-2">Qty</th>
            <th className="text-right py-2 px-2">Avg Entry</th>
            <th className="text-right py-2 px-2">Current</th>
            <th className="text-right py-2 px-2">Mkt Value</th>
            <th className="text-right py-2 px-2">Unrealized P&L</th>
            <th className="text-right py-2 px-2">P&L %</th>
            <th className="text-right py-2 px-2">Buys</th>
            <th className="text-right py-2 px-2">Sells</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.symbol} className="border-b border-dark-border/30 hover:bg-dark-hover/30">
              <td className="py-1.5 px-2 font-mono font-bold">{p.symbol}</td>
              <td className="py-1.5 px-2 text-right font-mono">{p.qty}</td>
              <td className="py-1.5 px-2 text-right font-mono">${(p.avgEntryPrice || 0).toFixed(2)}</td>
              <td className="py-1.5 px-2 text-right font-mono">${(p.currentPrice || 0).toFixed(2)}</td>
              <td className="py-1.5 px-2 text-right font-mono">${(p.marketValue || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className={`py-1.5 px-2 text-right font-mono font-bold ${
                (p.unrealizedPL || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'
              }`}>
                ${(p.unrealizedPL || 0) >= 0 ? '+' : ''}{(p.unrealizedPL || 0).toFixed(2)}
              </td>
              <td className={`py-1.5 px-2 text-right font-mono ${
                (p.unrealizedPLPct || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'
              }`}>
                {(p.unrealizedPLPct || 0) >= 0 ? '+' : ''}{(p.unrealizedPLPct || 0).toFixed(2)}%
              </td>
              <td className="py-1.5 px-2 text-right">{p.numBuys || 0}</td>
              <td className="py-1.5 px-2 text-right">{p.numSells || 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Cycles Tab ───────────────────────────────────────────────────── */
function CyclesTab({ cycles }) {
  if (!cycles.length) {
    return (
      <div className="card text-center py-8">
        <Clock className="w-8 h-8 text-dark-muted mx-auto mb-2" />
        <p className="text-dark-muted text-sm">No cycles recorded yet</p>
      </div>
    )
  }

  return (
    <div className="card overflow-x-auto">
      <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-dark-muted" /> Trading Cycles
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-dark-muted border-b border-dark-border">
            <th className="text-left py-2 px-2">Time</th>
            <th className="text-left py-2 px-2">Interval</th>
            <th className="text-left py-2 px-2">Cycle ID</th>
            <th className="text-left py-2 px-2">Mode</th>
            <th className="text-right py-2 px-2">Stocks</th>
            <th className="text-right py-2 px-2">Orders</th>
            <th className="text-right py-2 px-2">Filled</th>
            <th className="text-right py-2 px-2">Rejected</th>
            <th className="text-right py-2 px-2">Day P&L</th>
            <th className="text-right py-2 px-2">Equity</th>
          </tr>
        </thead>
        <tbody>
          {cycles.map((c, i) => (
            <tr key={i} className="border-b border-dark-border/30 hover:bg-dark-hover/30">
              <td className="py-1.5 px-2 text-dark-muted whitespace-nowrap">
                {c.timestamp ? new Date(c.timestamp).toLocaleString() : '—'}
              </td>
              <td className="py-1.5 px-2">
                <span className={`px-1.5 py-0.5 rounded text-[10px] ${c.interval === '1h' ? 'bg-accent-blue/15 text-accent-blue' : 'bg-accent-green/15 text-accent-green'}`}>
                  {c.interval || '1d'}
                </span>
              </td>
              <td className="py-1.5 px-2 font-mono text-[10px]">{c.cycleId?.slice(-12) || '—'}</td>
              <td className="py-1.5 px-2">
                <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                  c.brokerMode === 'live' ? 'bg-accent-red/15 text-accent-red'
                    : c.brokerMode === 'paper' ? 'bg-accent-blue/15 text-accent-blue'
                    : 'bg-dark-hover text-dark-muted'
                }`}>{c.brokerMode}</span>
              </td>
              <td className="py-1.5 px-2 text-right">{c.numStocks || 0}</td>
              <td className="py-1.5 px-2 text-right">{c.numOrders || 0}</td>
              <td className="py-1.5 px-2 text-right text-accent-green">{c.numFilled || 0}</td>
              <td className="py-1.5 px-2 text-right text-accent-red">{c.numRejected || 0}</td>
              <td className={`py-1.5 px-2 text-right font-mono font-bold ${
                (c.dayPL || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'
              }`}>
                ${(c.dayPL || 0) >= 0 ? '+' : ''}{(c.dayPL || 0).toFixed(2)}
              </td>
              <td className="py-1.5 px-2 text-right font-mono">${(c.equity || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Comparison Tab: Backtest vs Paper Trading ──────────────────── */
function ComparisonTab() {
  const [runs] = useRuns()
  const [btTrades] = useBacktestTrades(runs?.[0]?.id)
  const [paperTrades] = useTradeLog(500)

  const latestRun = runs?.[0]
  const btBySymbol = useMemo(() => {
    const m = {}
    if (!btTrades) return m
    btTrades.forEach(d => {
      const sym = d.symbol || ''
      if (!m[sym]) m[sym] = { trades: 0, pnl: 0, winRate: 0 }
      const stats = d.stats || {}
      m[sym].trades += stats.totalTrades || 0
      m[sym].pnl += (stats.finalBalance || 0) - (stats.initialBalance || 0)
      m[sym].winRate = stats.winRate || 0
    })
    return m
  }, [btTrades])

  const paperBySymbol = useMemo(() => {
    const m = {}
    paperTrades.forEach(o => {
      const sym = o.symbol || ''
      if (!m[sym]) m[sym] = { trades: 0, pnl: 0, wins: 0 }
      m[sym].trades++
      const filled = o.filledPrice || 0
      const cost = o.totalCost || 0
      if (o.side === 'sell' && filled > 0) m[sym].pnl += cost
      else if (o.side === 'buy') m[sym].pnl -= cost
      if (o.status === 'filled') m[sym].wins++
    })
    return m
  }, [paperTrades])

  const allSymbols = [...new Set([...Object.keys(btBySymbol), ...Object.keys(paperBySymbol)])]

  return (
    <div className="space-y-4">
      <div className="card">
        <h3 className="text-sm font-semibold mb-3">Backtest vs Paper Trading</h3>
        {latestRun && <p className="text-xs text-dark-muted mb-3">Comparing latest pipeline run ({latestRun.id?.slice(0,8)}) with live paper trades</p>}
        <table className="w-full text-xs">
          <thead>
            <tr className="text-dark-muted border-b border-dark-border">
              <th className="text-left py-2 px-2">Symbol</th>
              <th className="text-right py-2 px-2">BT Trades</th>
              <th className="text-right py-2 px-2">BT P&L</th>
              <th className="text-right py-2 px-2">BT Win%</th>
              <th className="text-right py-2 px-2">Paper Trades</th>
              <th className="text-right py-2 px-2">Paper P&L</th>
              <th className="text-right py-2 px-2">Deviation</th>
            </tr>
          </thead>
          <tbody>
            {allSymbols.map(sym => {
              const bt = btBySymbol[sym] || { trades: 0, pnl: 0, winRate: 0 }
              const pp = paperBySymbol[sym] || { trades: 0, pnl: 0 }
              const dev = bt.pnl !== 0 ? ((pp.pnl - bt.pnl) / Math.abs(bt.pnl) * 100) : 0
              return (
                <tr key={sym} className="border-b border-dark-border/40 hover:bg-dark-hover/30">
                  <td className="py-1.5 px-2 font-medium">{sym}</td>
                  <td className="py-1.5 px-2 text-right text-dark-muted">{bt.trades}</td>
                  <td className={`py-1.5 px-2 text-right font-mono ${bt.pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    ${bt.pnl.toFixed(0)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-dark-muted">{bt.winRate.toFixed(0)}%</td>
                  <td className="py-1.5 px-2 text-right text-dark-muted">{pp.trades}</td>
                  <td className={`py-1.5 px-2 text-right font-mono ${pp.pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    ${pp.pnl.toFixed(0)}
                  </td>
                  <td className={`py-1.5 px-2 text-right font-mono font-bold ${Math.abs(dev) > 50 ? 'text-accent-red' : 'text-dark-muted'}`}>
                    {dev !== 0 ? `${dev > 0 ? '+' : ''}${dev.toFixed(0)}%` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {allSymbols.length === 0 && (
          <p className="text-xs text-dark-muted text-center py-8">
            No data yet. Run a pipeline and execute some paper trades first.
          </p>
        )}
      </div>
    </div>
  )
}

/* ─── Helper: Stat display ─────────────────────────────────────────── */
function Stat({ label, value, color, sub }) {
  return (
    <div>
      <div className="text-[10px] text-dark-muted uppercase tracking-wider">{label}</div>
      <div className={`text-sm font-bold font-mono ${color || ''}`}>{value}</div>
      {sub && <div className="text-[10px] text-dark-muted">{sub}</div>}
    </div>
  )
}
