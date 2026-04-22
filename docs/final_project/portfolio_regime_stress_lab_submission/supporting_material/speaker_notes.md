# Speaker Notes

## 30-Second Version

I studied how stress builds and spreads across an equal-weight basket of
mega-cap technology stocks. The main result is that the portfolio breaks
into three interpretable volatility regimes, and the stress regime is
visibly different from the calm regime in volatility, correlation, and
breadth.

## Slide Narrative

1. Start with the stress-regime chart.
   Mention that Calm Expansion averaged 7.65%
   realized volatility, while Stress Contagion averaged
   45.90%.
2. Explain the custom Stress Propagation Score.
   It combines correlation, volatility, breadth, dispersion, and lead-lag
   concentration into one interpretable state variable.
3. Close with the forecast result.
   ARIMA(2, 0, 1) delivered RMSE
   0.0269 versus a naive baseline at
   0.0271.

## Back-Pocket Answers

- Why this universe? It keeps the story focused and reproducible.
- Why ARIMA? It is interpretable, rubric-friendly, and appropriate for
  realized volatility forecasting.
- Why is this unique? The custom stress score and propagation framing add
  an original research question without making speculative claims.

## Useful Lines

- Stress Contagion carried an average realized vol of 45.90% versus 7.65% in Calm Expansion.
- The Stress Propagation Score was higher in Stress Contagion (+1.49) than in Calm Expansion (-0.93).
- The strongest lead-lag relation was GOOGL leading AAPL by 3 day(s) (corr=+0.12).
