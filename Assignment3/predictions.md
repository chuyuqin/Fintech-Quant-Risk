# Predictions recorded before the main model fits

These predictions use only the assignment inputs and the permitted pre-fit diagnostics in
`preview.py` (parity, initial Greeks, sample moments, and Problem 5 CAPM/residual correlations).
The subsequent `run_analysis.py` performs IV inversion, tree pricing, return-distribution
fitting, simulation, optimization, and expanded factor regressions.

1. The sole parity flag is strike 175: call-minus-put exceeds discounted forward
   intrinsic by $0.88818, versus the specified $0.47 full-spread band. At least one
   quote there is inconsistent; parity alone cannot identify which leg. Exact parity
   implies equal call/put IV because the same volatility cancels from their price
   difference and each price is increasing in volatility. Wing IVs are unstable:
   low vega magnifies cents into large volatility changes. Calls at 215/220/225
   all have identical $0.05/$0.10 quotes, so I expect a spurious rising call wing.
2. The interest thresholds are about $0.892 and $0.225. Only the second dividend
   passes the call exercise condition. Dividends lower both calls and raise both
   puts. I expect a larger call exercise premium and a smaller put exercise premium:
   cash dividends create a call exercise incentive and make waiting more attractive
   for the put. This predicts premium changes, not just option-value changes.
3. Predicted ranking of relative delta-normal error: ShortStraddle and
   HedgedShortCall (roughly tied; major understatement), then CoveredCall
   (understatement of downside), then RiskReversal (smallest, tentative slight
   overstatement from positive local gamma). Their deltas are -0.889, -0.444,
   69.916, and 61.389 respectively; gammas are -8.092, -4.046, -3.793,
   and +0.536. All thetas are positive, so all should gain if spot stays fixed.
   The two near-delta-neutral short-gamma books should have the strongest ES/VaR
   inflation; deep tail losses are invisible to delta alone. Local gamma may not
   describe the risk reversal across both strikes.
4. PUTW's annual Sharpe is about 0.783, but skewness is -6.53 and excess kurtosis
   59.85. Its worst month is -16.72%; removing that month raises its mean and
   sharply reduces its volatility. I expect PUTW to dominate the long-only
   tangency portfolio (more than half its weight). Historical 1% ES should be
   much worse than matched-normal ES because the optimizer rewards frequent
   premiums while a few insurance losses control the downside tail.
5. FUND1 looks stronger in the CAPM (daily alpha 0.000441, t=3.20, versus FUND2's
   0.000245, t=1.83). FUND1 residuals correlate 0.801 with momentum, so adding
   factors should cut its alpha sharply. FUND2 residuals correlate 0.330 with SMB;
   I expect a smaller alpha change there, tentatively a reduction. The direction
   depends on factor premia and correlated omitted exposures, not correlation
   alone; this latter sign prediction is uncertain.
