import { useEffect, useRef, memo } from 'react'

/**
 * TradingView Advanced Chart in compare mode.
 * Overlays multiple stocks on a single normalised % chart.
 *
 * Props:
 *  - symbols   string[]   All tickers to overlay (first is the base)
 *  - height    number     Chart height (default 520)
 *  - range     string     e.g. "3M", "1D", "ALL"
 *  - colors    string[]   Optional line colours per symbol
 */

const RANGE_TO_INTERVAL = {
  '1D': '5', '5D': '15', '1M': '60', '3M': 'D',
  '6M': 'D', '12M': 'W', 'ALL': 'W',
}

const DEFAULT_COLORS = [
  '#58a6ff', '#3fb950', '#f85149', '#d29922',
  '#bc8cff', '#f0883e', '#56d4dd', '#db61a2',
]

function StockCompareChart({ symbols = [], height = '100%', range = '3M', colors = DEFAULT_COLORS }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || symbols.length === 0) return

    const container = containerRef.current
    container.innerHTML = ''

    const [base, ...rest] = symbols
    const interval = RANGE_TO_INTERVAL[range] || 'D'

    // Build compare_symbols list
    const compareSymbols = rest.map((sym, i) => ({
      symbol: sym,
      position: 'SameScale',
    }))

    const wrapper = document.createElement('div')
    wrapper.className = 'tradingview-widget-container__widget'
    wrapper.style.height = '100%'
    wrapper.style.width = '100%'

    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: base,
      interval,
      timezone: 'Etc/UTC',
      theme: 'dark',
      style: '2',            // Line style (cleaner for comparison)
      locale: 'en',
      backgroundColor: 'rgba(13, 17, 23, 1)',
      gridColor: 'rgba(48, 54, 61, 0.25)',
      hide_top_toolbar: false,
      hide_legend: false,
      hide_side_toolbar: true,
      allow_symbol_change: false,
      save_image: false,
      calendar: false,
      compare_symbols: compareSymbols,
      // Show % change so all stocks are on the same scale
      scaleMode: 'Percentage',
    })

    container.appendChild(wrapper)
    container.appendChild(script)

    return () => { container.innerHTML = '' }
  }, [symbols.join(','), range]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container"
      style={{ height: typeof height === 'number' ? `${height}px` : height, width: '100%' }}
    />
  )
}

export default memo(StockCompareChart)
