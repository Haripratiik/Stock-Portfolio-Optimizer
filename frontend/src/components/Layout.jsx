import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useState } from 'react'
import {
  LayoutDashboard,
  Play,
  ListTodo,
  BarChart3,
  Brain,
  Settings,
  LogOut,
  TrendingUp,
  Menu,
  X,
  ImageIcon,
  Bot,
  BarChart2,
} from 'lucide-react'

const NAV = [
  { to: '/',          icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: TrendingUp,      label: 'Portfolio' },
  { to: '/trading',   icon: Bot,             label: 'Trading' },
  { to: '/run',       icon: Play,            label: 'Run' },
  { to: '/queue',     icon: ListTodo,        label: 'Queue' },
  { to: '/patterns',  icon: BarChart3,       label: 'Patterns' },
  { to: '/strategies',icon: Brain,           label: 'Strategies' },
  { to: '/trades',    icon: BarChart2,       label: 'Trades' },
  { to: '/charts',    icon: ImageIcon,       label: 'Charts' },
  { to: '/settings',  icon: Settings,        label: 'Settings' },
]

// Items shown in the bottom tab bar on mobile (most-used)
const BOTTOM_NAV = NAV.slice(0, 5)

export default function Layout() {
  const { user, logout } = useAuth()
  const [drawerOpen, setDrawerOpen] = useState(false)

  const navLinkClass = ({ isActive }) =>
    `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
      isActive
        ? 'bg-accent-blue/10 text-accent-blue'
        : 'text-dark-muted hover:text-dark-text hover:bg-dark-hover'
    }`

  return (
    <div className="flex h-screen overflow-hidden">

      {/* ─── Desktop Sidebar (hidden on mobile) ─── */}
      <aside className="hidden md:flex w-56 flex-shrink-0 bg-dark-card border-r border-dark-border flex-col">
        <div className="h-14 flex items-center px-4 border-b border-dark-border">
          <TrendingUp className="w-5 h-5 text-accent-blue mr-2" />
          <span className="font-semibold text-sm tracking-wide">PORTFOLIO MGR</span>
        </div>
        <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} end={to === '/'} className={navLinkClass}>
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-3 border-t border-dark-border">
          <div className="flex items-center gap-2">
            {user?.photoURL ? (
              <img src={user.photoURL} alt="" className="w-7 h-7 rounded-full" />
            ) : (
              <div className="w-7 h-7 rounded-full bg-accent-blue/20 flex items-center justify-center text-xs text-accent-blue font-bold">
                {user?.email?.[0]?.toUpperCase() || '?'}
              </div>
            )}
            <div className="flex-1 min-w-0">
              <div className="text-xs text-dark-text truncate">{user?.displayName || 'User'}</div>
              <div className="text-[10px] text-dark-muted truncate">{user?.email}</div>
            </div>
            <button onClick={logout} className="p-1 text-dark-muted hover:text-accent-red transition-colors" title="Sign out">
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* ─── Mobile: slide-in drawer overlay ─── */}
      {drawerOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60" onClick={() => setDrawerOpen(false)} />
          {/* Drawer panel */}
          <div className="relative w-64 bg-dark-card border-r border-dark-border flex flex-col z-10">
            <div className="h-14 flex items-center justify-between px-4 border-b border-dark-border">
              <div className="flex items-center">
                <TrendingUp className="w-5 h-5 text-accent-blue mr-2" />
                <span className="font-semibold text-sm tracking-wide">PORTFOLIO MGR</span>
              </div>
              <button onClick={() => setDrawerOpen(false)} className="text-dark-muted hover:text-dark-text">
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
              {NAV.map(({ to, icon: Icon, label }) => (
                <NavLink key={to} to={to} end={to === '/'} className={navLinkClass} onClick={() => setDrawerOpen(false)}>
                  <Icon className="w-4 h-4" />
                  {label}
                </NavLink>
              ))}
            </nav>
            <div className="p-3 border-t border-dark-border">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-accent-blue/20 flex items-center justify-center text-xs text-accent-blue font-bold">
                  {user?.email?.[0]?.toUpperCase() || '?'}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-dark-text truncate">{user?.displayName || 'User'}</div>
                  <div className="text-[10px] text-dark-muted truncate">{user?.email}</div>
                </div>
                <button onClick={logout} className="p-1 text-dark-muted hover:text-accent-red transition-colors">
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── Main content area ─── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Mobile top header */}
        <header className="md:hidden flex items-center justify-between h-12 px-4 bg-dark-card border-b border-dark-border flex-shrink-0">
          <div className="flex items-center">
            <TrendingUp className="w-4 h-4 text-accent-blue mr-2" />
            <span className="font-semibold text-sm tracking-wide">PORTFOLIO MGR</span>
          </div>
          <button onClick={() => setDrawerOpen(true)} className="text-dark-muted hover:text-dark-text p-1">
            <Menu className="w-5 h-5" />
          </button>
        </header>

        {/* Page content — add pb-16 on mobile to clear the bottom tab bar */}
        <main className="flex-1 overflow-y-auto bg-dark-bg pb-16 md:pb-0">
          <Outlet />
        </main>

        {/* ─── Mobile bottom tab bar ─── */}
        <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-dark-card border-t border-dark-border flex">
          {BOTTOM_NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex-1 flex flex-col items-center justify-center py-2 gap-0.5 text-[10px] font-medium transition-colors ${
                  isActive ? 'text-accent-blue' : 'text-dark-muted'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>

      </div>
    </div>
  )
}

