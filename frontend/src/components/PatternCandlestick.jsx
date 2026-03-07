import { useEffect, useRef, memo } from 'react'
import { createChart, CandlestickSeries, HistogramSeries } from 'lightweight-charts'

/**
 * Renders a candlestick chart from pattern gene data.
 * Synthesizes OHLCV candles from the gene parameters stored in Firestore.
 *
 * Props:
 *  - genes       (array)  Pattern genes from genesJson
 *  - height      (number) Chart height in px (default 220)
 *  - symbol      (string) For the title overlay
 *  - interval    (string) Timeframe label
 */
function PatternCandlestick({ genes = [], height = 220, symbol = '', interval = '' }) {
  const chartRef = useRef(null)
  const chartInstanceRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current || !genes.length) return

    const container = chartRef.current
    const chart = createChart(container, {
      height,
      layout: {
        background: { color: '#0d1117' },
        textColor: '#8b949e',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(48,54,61,0.3)' },
        horzLines: { color: 'rgba(48,54,61,0.3)' },
      },
      crosshair: {
        vertLine: { color: '#58a6ff', width: 1, style: 2 },
        horzLine: { color: '#58a6ff', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: '#30363d',
      },
      timeScale: {
        borderColor: '#30363d',
        timeVisible: false,
      },
    })
    chartInstanceRef.current = chart

    const { candles, volumes } = genesToOHLCV(genes)

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#3fb950',
      downColor: '#f85149',
      borderUpColor: '#3fb950',
      borderDownColor: '#f85149',
      wickUpColor: '#3fb950',
      wickDownColor: '#f85149',
    })
    candleSeries.setData(candles)

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    volumeSeries.setData(volumes)
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
      borderVisible: false,
    })

    chart.timeScale().fitContent()

    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth })
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartInstanceRef.current = null
    }
  }, [genes, height])

  if (!genes.length) {
    return (
      <div className="flex items-center justify-center text-dark-muted text-xs py-8"
           style={{ height }}>
        No gene data available
      </div>
    )
  }

  return (
    <div className="relative">
      {(symbol || interval) && (
        <div className="absolute top-2 left-3 z-10 flex items-center gap-2">
          {symbol && (
            <span className="text-xs font-mono font-bold text-accent-blue/80">{symbol}</span>
          )}
          {interval && (
            <span className="text-[10px] text-dark-muted bg-dark-bg/80 px-1.5 py-0.5 rounded">
              {interval}
            </span>
          )}
        </div>
      )}
      <div ref={chartRef} className="w-full rounded-lg overflow-hidden" />
    </div>
  )
}

/**
 * Convert pattern genes into synthetic OHLCV candles.
 *
 * Body sizing is intentionally visual — `expectedPriceChangePct` encodes the
 * direction/magnitude the GA expects, but it often sits near zero when the
 * GA hasn't converged on a large directional move.  We therefore derive
 * visual body size from `minBodyPct` (0-100, representing body-to-range
 * ratio) and a fixed base percentage, ensuring every candle is clearly
 * readable regardless of the raw gene values.
 *
 *   BASE_BODY = 1.8% of current price
 *   strength  = minBodyPct / 100, clamped [0.3, 1.0]
 *   bodySize  = price × BASE_BODY × strength
 *
 * Candle direction comes from candleType (BULLISH / BEARISH / DOJI).
 * DOJI uses a near-flat body with exaggerated wicks.
 */
function genesToOHLCV(genes) {
  const candles = []
  const volumes = []
  let price = 100
  const BASE_VOL = 1_000_000
  const BASE_BODY = 0.018 // 1.8% of price per candle
  const baseDate = new Date('2024-01-01')

  for (let i = 0; i < genes.length; i++) {
    const gene = typeof genes[i] === 'string' ? JSON.parse(genes[i]) : genes[i]
    const candleType = (gene.candleType || 'BULLISH').toUpperCase()
    const volumePct = (gene.expectedVolumePct || 0) / 100

    // strength: minBodyPct tells us how "decisive" the candle body is
    const strength = Math.max(0.3, Math.min(1.0, (gene.minBodyPct || 50) / 100))
    const bodySize = price * BASE_BODY * strength
    const wickSize = bodySize * 0.35

    let open, close, high, low

    if (candleType === 'BULLISH') {
      open  = price
      close = price + bodySize
      low   = open  - wickSize
      high  = close + wickSize
    } else if (candleType === 'BEARISH') {
      open  = price
      close = price - bodySize
      high  = open  + wickSize
      low   = close - wickSize
    } else {
      // DOJI — nearly flat body with prominent wicks
      const dojiBody = bodySize * 0.07
      open  = price + dojiBody / 2
      close = price - dojiBody / 2
      high  = price + bodySize * 0.75
      low   = price - bodySize * 0.75
    }

    const d = new Date(baseDate)
    d.setDate(d.getDate() + i)
    const time = d.toISOString().split('T')[0]

    candles.push({
      time,
      open:  +open.toFixed(4),
      high:  +Math.max(high, open, close).toFixed(4),
      low:   +Math.min(low,  open, close).toFixed(4),
      close: +close.toFixed(4),
    })
    volumes.push({
      time,
      value: Math.round(BASE_VOL * (1 + volumePct)),
      color: close >= open ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)',
    })

    price = close
  }

  return { candles, volumes }
}

export default memo(PatternCandlestick)
