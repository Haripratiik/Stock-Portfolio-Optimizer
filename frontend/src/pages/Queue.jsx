import { useQueue } from '../hooks/useFirestore'
import { db } from '../firebase'
import { doc, updateDoc, deleteDoc } from 'firebase/firestore'
import {
  Clock,
  Play,
  CheckCircle,
  XCircle,
  SkipForward,
  Trash2,
  Loader,
} from 'lucide-react'
import { format } from 'date-fns'

const STATUS_CONFIG = {
  queued:    { color: 'badge-yellow', icon: Clock,       label: 'Queued' },
  approved:  { color: 'badge-blue',   icon: Play,        label: 'Approved' },
  running:   { color: 'badge-blue',   icon: Loader,      label: 'Running' },
  completed: { color: 'badge-green',  icon: CheckCircle, label: 'Completed' },
  failed:    { color: 'badge-red',    icon: XCircle,     label: 'Failed' },
  skipped:   { color: 'badge-muted',  icon: SkipForward, label: 'Skipped' },
}

export default function Queue() {
  const [queue, loading] = useQueue()

  const pending   = queue.filter((q) => q.status === 'queued')
  const active    = queue.filter((q) => q.status === 'running' || q.status === 'approved')
  const completed = queue.filter((q) => q.status === 'completed')
  const other     = queue.filter((q) => q.status === 'failed' || q.status === 'skipped')

  const updateStatus = async (id, status) => {
    await updateDoc(doc(db, 'run_commands', id), { status })
  }

  const deleteCommand = async (id) => {
    if (confirm('Remove this from the queue?')) {
      await deleteDoc(doc(db, 'run_commands', id))
    }
  }

  const renderItem = (item) => {
    const cfg = STATUS_CONFIG[item.status] || STATUS_CONFIG.queued
    const Icon = cfg.icon

    return (
      <div key={item.id} className="card-hover flex items-start gap-4">
        {/* Status icon */}
        <div className="pt-0.5">
          <Icon className={`w-5 h-5 ${
            item.status === 'running' ? 'text-accent-blue animate-spin' :
            item.status === 'completed' ? 'text-accent-green' :
            item.status === 'failed' ? 'text-accent-red' :
            'text-dark-muted'
          }`} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-sm">{item.description || item.type}</span>
            <span className={cfg.color}>{cfg.label}</span>
          </div>
          <div className="text-xs text-dark-muted space-x-3">
            <span>Type: {item.type?.replace(/_/g, ' ')}</span>
            <span>&bull;</span>
            <span>Stocks: {item.stocks?.join(', ') || 'N/A'}</span>
            <span>&bull;</span>
            <span>Source: {item.source || 'website'}</span>
          </div>
          <div className="text-xs text-dark-muted mt-1">
            Created: {item.createdAt ? format(new Date(item.createdAt), 'MMM d, yyyy HH:mm') : '—'}
            {item.startedAt && ` • Started: ${format(new Date(item.startedAt), 'HH:mm')}`}
            {item.completedAt && ` • Done: ${format(new Date(item.completedAt), 'HH:mm')}`}
          </div>

          {/* Error message */}
          {item.error && (
            <div className="mt-2 text-xs text-accent-red bg-accent-red/10 border border-accent-red/20 rounded p-2">
              {item.error}
            </div>
          )}

          {/* Result summary */}
          {item.result && item.status === 'completed' && (
            <div className="mt-2 text-xs bg-accent-green/5 border border-accent-green/20 rounded p-2 space-y-0.5">
              {item.result.totalReturnPct != null && (
                <div>Return: <span className="font-mono text-accent-green">{(item.result.totalReturnPct * 100).toFixed(2)}%</span></div>
              )}
              {item.result.alphaVsBuyHold != null && (
                <div>Alpha: <span className="font-mono text-accent-green">{(item.result.alphaVsBuyHold * 100).toFixed(2)}%</span></div>
              )}
              {item.result.message && <div>{item.result.message}</div>}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-1.5 flex-shrink-0">
          {item.status === 'queued' && (
            <>
              <button
                onClick={() => updateStatus(item.id, 'skipped')}
                className="p-1.5 text-dark-muted hover:text-accent-yellow transition-colors"
                title="Skip"
              >
                <SkipForward className="w-4 h-4" />
              </button>
              <button
                onClick={() => deleteCommand(item.id)}
                className="p-1.5 text-dark-muted hover:text-accent-red transition-colors"
                title="Delete"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </>
          )}
          {(item.status === 'completed' || item.status === 'failed' || item.status === 'skipped') && (
            <button
              onClick={() => deleteCommand(item.id)}
              className="p-1.5 text-dark-muted hover:text-accent-red transition-colors"
              title="Remove"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    )
  }

  if (loading) {
    return <div className="p-6 text-dark-muted">Loading queue...</div>
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-bold">Task Queue</h1>
        <p className="text-dark-muted text-sm mt-0.5">
          Pipeline tasks waiting for your Local Agent
        </p>
      </div>

      {/* Active */}
      {active.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-accent-blue mb-3 flex items-center gap-2">
            <Loader className="w-4 h-4 animate-spin" /> Active ({active.length})
          </h2>
          <div className="space-y-2">{active.map(renderItem)}</div>
        </section>
      )}

      {/* Pending */}
      <section>
        <h2 className="text-sm font-medium text-dark-muted mb-3 flex items-center gap-2">
          <Clock className="w-4 h-4" /> Pending ({pending.length})
        </h2>
        {pending.length === 0 ? (
          <div className="card text-center text-dark-muted py-8 text-sm">
            No tasks in queue. Go to <span className="text-accent-blue">Run Pipeline</span> to add one.
          </div>
        ) : (
          <div className="space-y-2">{pending.map(renderItem)}</div>
        )}
      </section>

      {/* Completed */}
      {completed.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-accent-green mb-3 flex items-center gap-2">
            <CheckCircle className="w-4 h-4" /> Completed ({completed.length})
          </h2>
          <div className="space-y-2">{completed.map(renderItem)}</div>
        </section>
      )}

      {/* Failed / Skipped */}
      {other.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-dark-muted mb-3">Other ({other.length})</h2>
          <div className="space-y-2">{other.map(renderItem)}</div>
        </section>
      )}
    </div>
  )
}
