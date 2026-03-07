# Investigation: Why Stop-Loss and Walk-Forward Degrade Performance

## Summary

Before stop-loss and walk-forward were added, the system achieved **~+20%** over the same period. After adding them, it drops to **~-4%** — a 24 percentage point swing.

---

## 1. Stop-Loss Analysis

### How It Works
- **Daily/Weekly**: Stop is checked against the bar's **closing price** (not intraday low). Exit at next bar's open.
- **Intraday (1h, 30m, etc.)**: Stop is checked against the bar's **low** (LONG) or **high** (SHORT). If touched, exit at stop price or bar open (gap-through).
- **Thresholds** (from `stopLossPct=5%` base):
  - 1d: 5.0% (close-based)
  - 1h: 3.0%
  - 30m: 1.5%

### Root Causes of Performance Degradation

#### A. Hourly 3% stop is too tight for volatile tech stocks
- Tech names (AAPL, NVDA, GOOGL, etc.) often swing 2–4% in a single hour.
- A 3% stop can trigger on normal volatility rather than genuine trend reversal.
- **Effect**: Trades that would recover and finish positive get cut at -3%, turning potential winners into small losers.

#### B. Intraday stop uses LOW/HIGH (not close)
- For 1h bars, if the price touches -3% intraday and then closes at -1%, the stop still fires.
- That exit happens on temporary noise; the bar close would have shown a smaller drawdown.
- **Effect**: More false stop-outs than a close-based check.

#### C. Extended cooldown after stop-loss (2× normal)
- After a stop, the system waits 10 bars (e.g. 10 hours) before re-entering.
- During that time, the market can reverse and create a new valid entry.
- **Effect**: Re-entry opportunities are delayed or missed.

#### D. Amplifying effect with many slots
- With several (stock × interval) slots, each hit by the same problems, losses add up.
- Slots that would have been net positive without stop-loss turn net negative with it.

### Recommended Adjustments (keep stop-loss, reduce damage)
1. **Widen hourly stop**: 3% → 4–5% for 1h to align better with typical intraday volatility.
2. **Use close-based stops for 1h when possible** (or widen further) to avoid intraday noise.
3. **ATR-adaptive stops**: Scale stop % by recent volatility (e.g. 2× ATR) instead of a fixed %.
4. **Shorter cooldown after stop**: 1.5× instead of 2×, or equal to normal cooldown.

---

## 2. Walk-Forward Analysis

### How It Works
- Walk-forward runs the ML model on rolling/expanding windows (e.g. train on year 1, test on quarter 2; train on years 1–2, test on quarter 3).
- Results are combined per slot as `WalkForward_1d`, `WalkForward_1h`.
- These entries are now **excluded** from allocation and total profit (correct behavior).

### Bugs Found in Walk-Forward Aggregation

1. **Wrong timestamp key**: Uses `t.get('entryTime')` but trades use `'timestamp'`.
2. **Wrong returnPct key**: Uses `t.get('totalReturnPct', 0)` but that is on the BacktestResult, not on trades. Trades use `'returnPct'`.

These only affect Walk-forward display/structure; they do not affect the main backtest or the -4% result.

### Impact on Performance
- Walk-forward slots are **excluded** from allocation and total P&L.
- The -4% comes from the **main** ML backtest slots only.
- Walk-forward is a validation step; it does not directly drive the -4%.

---

## 3. Fund Allocation Check

The dynamic allocator is functioning as intended:
- Poor slots (low score) are ghosted (0% allocation).
- Capital is shifted to stronger slots.
- P/L scaling and equity curve construction are correct.

Stop-loss hurts performance at the **trade level** (individual exits). The allocator then correctly identifies those slots as worse and reduces allocation, which is correct but cannot fix the underlying loss from overly tight stops.

---

## 4. Next Steps

1. **Fix Walk-forward aggregation bugs** (timestamp, returnPct) so validation results are consistent.
2. **Adjust stop-loss parameters**:
   - Widen 1h stop (e.g. 3% → 4.5%).
   - Consider reducing stop-loss cooldown from 2× to 1.5×.
3. **Optional**: Add ATR-based adaptive stops for volatility-aware thresholds.
