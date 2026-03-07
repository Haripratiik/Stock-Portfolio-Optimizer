import { useEffect, useRef, memo } from 'react'

/**
 * TradingView Advanced Chart — area style (Robinhood-like).
 * Uses the `range` parameter so bare tickers (AAPL, AMZN, MSFT, JPM …)
 * are resolved correctly without needing an exchange prefix.
 *
 * Props:
 *  - symbol   string   Ticker, e.g. "AAPL"
 *  - height   number   Pixel height (default: fills parent via 100%)
 *  - range    string   "1D" | "5D" | "1M" | "3M" | "6M" | "12M" | "ALL"
 */

// Map UI range labels → TradingView range + interval pairs
const RANGE_CFG = {
  '1D':  { range: '1D',   interval: '5'  },
  '5D':  { range: '5D',   interval: '15' },
  '1M':  { range: '1M',   interval: '60' },
  '3M':  { range: '3M',   interval: 'D'  },
  '6M':  { range: '6M',   interval: 'D'  },
  '12M': { range: '12M',  interval: 'W'  },
  'ALL': { range: 'ALL',  interval: 'W'  },
}

function StockMiniChart({ symbol, height = '100%', range = '3M' }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !symbol) return

    const container = containerRef.current
    container.innerHTML = ''

    const { range: tvRange, interval } = RANGE_CFG[range] || RANGE_CFG['3M']

    const wrapper = document.createElement('div')
    wrapper.className = 'tradingview-widget-container__widget'
    wrapper.style.height = '100%'
    wrapper.style.width  = '100%'

    const script = document.createElement('script')
    script.src   = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize:          true,
      symbol,                        // bare ticker — TradingView resolves it
      interval,
      range:             tvRange,
      timezone:          'Etc/UTC',
      theme:             'dark',
      style:             '3',        // Area chart (Robinhood-style)
      locale:            'en',
      backgroundColor:   'rgba(13,17,23,1)',
      gridColor:         'rgba(48,54,61,0.2)',
      hide_top_toolbar:  true,       // hide TradingView toolbar — use our own range pills
      hide_legend:       false,
      hide_side_toolbar: true,
      allow_symbol_change: false,
      save_image:        false,
      calendar:          false,
      support_host:      'https://www.tradingview.com',
    })

    container.appendChild(wrapper)
    container.appendChild(script)

    return () => { container.innerHTML = '' }
  }, [symbol, range])

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container"
      style={{ height: typeof height === 'number' ? `${height}px` : height, width: '100%' }}
    />
  )
}

export default memo(StockMiniChart)
