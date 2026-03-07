import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Portfolio from './pages/Portfolio'
import RunPipeline from './pages/RunPipeline'
import Queue from './pages/Queue'
import Patterns from './pages/Patterns'
import Strategies from './pages/Strategies'
import Charts from './pages/Charts'
import SettingsPage from './pages/Settings'
import Trading from './pages/Trading'
import TradesPage from './pages/Trades'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="portfolio" element={<Portfolio />} />
        <Route path="trading" element={<Trading />} />
        <Route path="run" element={<RunPipeline />} />
        <Route path="queue" element={<Queue />} />
        <Route path="patterns"    element={<Patterns />} />
        <Route path="strategies"  element={<Strategies />} />
        <Route path="charts"      element={<Charts />} />
        <Route path="trades"      element={<TradesPage />} />
        <Route path="settings"    element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
