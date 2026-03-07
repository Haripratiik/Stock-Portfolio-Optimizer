export default function StatsCard({ label, value, sub, icon: Icon, color = 'blue' }) {
  const colorMap = {
    blue:   'text-accent-blue   bg-accent-blue/10',
    green:  'text-accent-green  bg-accent-green/10',
    red:    'text-accent-red    bg-accent-red/10',
    yellow: 'text-accent-yellow bg-accent-yellow/10',
    purple: 'text-accent-purple bg-accent-purple/10',
  }
  const cls = colorMap[color] || colorMap.blue

  return (
    <div className="card flex items-start gap-4">
      {Icon && (
        <div className={`p-2.5 rounded-lg ${cls}`}>
          <Icon className="w-5 h-5" />
        </div>
      )}
      <div>
        <div className="text-dark-muted text-xs font-medium uppercase tracking-wider mb-1">
          {label}
        </div>
        <div className="text-2xl font-bold tabular-nums">{value}</div>
        {sub && <div className="text-dark-muted text-xs mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}
