# EXP4 - How the size of the monthly contribution X changes what the plan should hold

Script: `scripts/dca/exp4_contribution_size.py` (re-runnable, about 15 s). All tables below are in the CSVs in this folder. Prices are the adjusted total-return series in `data/ohlcv_daily`; decisions on the first trading day of each month at the open, slippage 0.0005, commission 0, the engine's `buy_underweight` policy. Nothing here is advice or a forecast.

**Headline.** The owner's intuition is half right. What X changes is (1) whether whole shares can be bought at all (a pure rounding effect, about X below roughly 1-2 times the price of the priciest fund), and (2) how long it takes new money to dilute an existing concentrated position (arithmetic that scales as 1/X). With fractional shares and zero commission, X does not limit the number of funds N, and IRR and weight tracking are exactly scale-invariant. The one thing that is independent of X is how much each extra fund diversifies (B).

## A. Whole shares vs fractional shares

Setup: target portfolios N = 1, 2, 3, 5, 8 (as specified), policy `buy_underweight`, account built from zero, monthly X. Windows: last 5 years (window start 2021-11-14, first decision 2021-11-15, 61 monthly buys) and last 10 years (window start 2016-11-14, first decision 2016-11-14, 121 buys), both ending 2026-11-13. Modes: `whole` (fractional=False), `frac` (fractional=True), `frac_min5` (fractional with $5 minimum order). X = 25 and 50 were added to the specified list to see where fractional shares start to bind. Latest adjusted closes (2026-11-13, equal to actual prices to within the dividend adjustment; adjusted prices are somewhat lower than actual ones for old dates, so the 10-year window is slightly understated for price-level effects early on):

| VOO | BRK-B | VXUS | QQQ | SCHD | VTV | BND | GLD |
|---|---|---|---|---|---|---|---|
| 693.38 | 592.94 | 82.64 | 679.89 | 36.12 | 229.51 | 70.56 | 355.45 |

Weights are measured daily as holding value / total account value (cash counts in the denominator, so idle cash shows up as a shortfall of weight). "Time-average" is the mean over every trading day of the window, **including the ramp-up from an empty account**, which is why the 5-year window looks worse than the 10-year one (the first year of a 5-year window is a larger share of the whole). Between monthly purchases the weights also drift with the market, which is X-independent.

**Time-average largest asset-weight deviation from target (pp), whole shares, last 5y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 42.8 | 23.6 | 12.0 | 5.4 | 4.5 | 2.5 | 1.1 | 0.7 | 0.4 | 0.2 |
| N=2 | 37.9 | 22.3 | 13.5 | 7.6 | 5.5 | 3.5 | 2.2 | 1.3 | 1.3 | 1.1 |
| N=3 | 41.4 | 24.7 | 14.2 | 8.0 | 5.8 | 3.5 | 1.9 | 1.3 | 1.1 | 0.9 |
| N=5 | 40.0 | 23.9 | 13.9 | 8.1 | 6.0 | 3.5 | 2.2 | 1.3 | 1.1 | 0.9 |
| N=8 | 30.0 | 20.4 | 13.2 | 7.6 | 5.8 | 3.8 | 2.1 | 1.0 | 0.9 | 0.8 |

**Time-average mean absolute weight deviation (pp, mean over assets), whole shares, last 5y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 42.8 | 23.6 | 12.0 | 5.4 | 4.5 | 2.5 | 1.1 | 0.7 | 0.4 | 0.2 |
| N=2 | 33.4 | 19.0 | 10.8 | 6.0 | 3.9 | 2.7 | 1.6 | 1.0 | 1.0 | 0.9 |
| N=3 | 23.1 | 13.3 | 7.7 | 4.3 | 3.1 | 2.0 | 1.0 | 0.7 | 0.6 | 0.6 |
| N=5 | 15.7 | 9.9 | 5.8 | 3.4 | 2.5 | 1.5 | 0.9 | 0.6 | 0.5 | 0.4 |
| N=8 | 9.8 | 7.2 | 4.5 | 2.7 | 2.0 | 1.3 | 0.7 | 0.4 | 0.4 | 0.3 |

**Time-average idle cash (% of account), whole shares, last 5y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 42.8 | 23.6 | 12.0 | 5.4 | 4.5 | 2.5 | 1.1 | 0.7 | 0.4 | 0.2 |
| N=2 | 66.7 | 37.9 | 21.5 | 11.6 | 7.4 | 4.7 | 2.3 | 0.9 | 0.8 | 0.5 |
| N=3 | 69.2 | 39.7 | 23.0 | 12.5 | 8.9 | 5.4 | 2.3 | 1.2 | 0.8 | 0.5 |
| N=5 | 78.6 | 49.4 | 28.9 | 16.8 | 12.2 | 7.0 | 3.8 | 1.9 | 1.3 | 0.8 |
| N=8 | 78.4 | 57.0 | 36.2 | 20.9 | 15.8 | 9.7 | 5.0 | 2.3 | 1.7 | 1.0 |

**Final value shortfall vs fractional (%), whole shares, last 5y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 10.6 | 2.0 | 2.1 | 1.1 | 0.5 | 0.2 | 0.1 | 0.1 | 0.1 | 0.0 |
| N=2 | 20.9 | 6.5 | 4.7 | 2.2 | 1.4 | 1.0 | 0.3 | 0.1 | 0.1 | 0.1 |
| N=3 | 17.3 | 9.7 | 4.4 | 2.4 | 1.6 | 0.5 | 0.4 | 0.1 | 0.2 | 0.1 |
| N=5 | 27.8 | 15.4 | 5.4 | 2.5 | 2.4 | 1.4 | 0.4 | 0.2 | 0.2 | 0.1 |
| N=8 | 26.0 | 15.2 | 7.2 | 3.1 | 2.7 | 1.5 | 0.5 | 0.4 | 0.2 | 0.1 |

**Months with no purchase at all, whole shares, last 5y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 58 | 55 | 48 | 35 | 21 | 4 | 0 | 0 | 0 | 0 |
| N=2 | 59 | 55 | 47 | 34 | 23 | 9 | 0 | 0 | 0 | 0 |
| N=3 | 55 | 49 | 35 | 16 | 7 | 1 | 0 | 0 | 0 | 0 |
| N=5 | 52 | 40 | 20 | 11 | 4 | 1 | 0 | 0 | 0 | 0 |
| N=8 | 52 | 41 | 21 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |


**Time-average largest asset-weight deviation from target (pp), whole shares, last 10y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 16.4 | 8.3 | 4.1 | 2.1 | 1.4 | 0.9 | 0.4 | 0.2 | 0.1 | 0.1 |
| N=2 | 16.0 | 9.3 | 5.2 | 3.2 | 2.5 | 1.8 | 1.4 | 1.2 | 1.1 | 1.0 |
| N=3 | 17.7 | 9.4 | 5.4 | 2.9 | 2.3 | 1.7 | 1.2 | 1.1 | 1.0 | 1.0 |
| N=5 | 18.8 | 9.8 | 5.2 | 3.1 | 2.2 | 1.6 | 1.2 | 1.0 | 0.9 | 0.9 |
| N=8 | 18.3 | 8.8 | 5.3 | 2.9 | 2.1 | 1.6 | 1.1 | 0.9 | 0.9 | 0.9 |

**Time-average mean absolute weight deviation (pp, mean over assets), whole shares, last 10y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 16.4 | 8.3 | 4.1 | 2.1 | 1.4 | 0.9 | 0.4 | 0.2 | 0.1 | 0.1 |
| N=2 | 13.1 | 7.4 | 4.2 | 2.5 | 2.1 | 1.4 | 1.2 | 1.1 | 1.0 | 1.0 |
| N=3 | 9.5 | 5.2 | 3.0 | 1.6 | 1.3 | 1.1 | 0.8 | 0.7 | 0.7 | 0.6 |
| N=5 | 8.2 | 4.0 | 2.2 | 1.3 | 1.0 | 0.7 | 0.6 | 0.5 | 0.4 | 0.4 |
| N=8 | 6.4 | 3.2 | 1.9 | 1.1 | 0.8 | 0.6 | 0.5 | 0.4 | 0.4 | 0.3 |

**Time-average idle cash (% of account), whole shares, last 10y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 16.4 | 8.3 | 4.1 | 2.1 | 1.4 | 0.9 | 0.4 | 0.2 | 0.1 | 0.1 |
| N=2 | 26.0 | 13.9 | 7.5 | 3.7 | 2.8 | 1.4 | 0.7 | 0.4 | 0.2 | 0.1 |
| N=3 | 27.7 | 15.0 | 8.0 | 3.9 | 2.8 | 1.8 | 0.8 | 0.4 | 0.3 | 0.2 |
| N=5 | 40.2 | 19.6 | 10.1 | 5.4 | 3.6 | 2.1 | 1.1 | 0.5 | 0.3 | 0.2 |
| N=8 | 50.9 | 25.3 | 14.1 | 7.5 | 4.9 | 3.1 | 1.6 | 0.8 | 0.5 | 0.3 |

**Final value shortfall vs fractional (%), whole shares, last 10y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 8.4 | 4.0 | 1.5 | 0.4 | 0.6 | 0.3 | 0.2 | 0.1 | 0.0 | 0.0 |
| N=2 | 11.1 | 5.6 | 2.5 | 0.9 | 0.6 | 0.4 | 0.2 | 0.1 | 0.1 | 0.1 |
| N=3 | 10.3 | 6.8 | 2.6 | 1.0 | 0.9 | 0.5 | 0.2 | 0.1 | 0.0 | 0.0 |
| N=5 | 25.0 | 10.1 | 4.3 | 1.7 | 1.4 | 0.5 | 0.3 | 0.2 | 0.1 | 0.1 |
| N=8 | 26.7 | 10.4 | 5.3 | 2.4 | 1.7 | 0.8 | 0.4 | 0.2 | 0.1 | 0.1 |

**Months with no purchase at all, whole shares, last 10y**

| index | $25 | $50 | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| N=1 | 113 | 103 | 83 | 45 | 23 | 4 | 0 | 0 | 0 | 0 |
| N=2 | 112 | 101 | 82 | 47 | 22 | 0 | 0 | 0 | 0 | 0 |
| N=3 | 102 | 84 | 56 | 27 | 14 | 2 | 0 | 0 | 0 | 0 |
| N=5 | 94 | 62 | 31 | 12 | 10 | 2 | 0 | 0 | 0 | 0 |
| N=8 | 95 | 58 | 25 | 8 | 4 | 1 | 0 | 0 | 0 | 0 |


**Fractional shares ($5 minimum order), selected X** - time-average idle cash % and largest-asset deviation pp

Idle cash %, last 5y:

| index | $25 | $50 | $100 | $200 | $500 |
|---|---|---|---|---|---|
| N=1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| N=2 | 0.02 | 0.01 | 0.0 | 0.0 | 0.0 |
| N=3 | 0.23 | 0.04 | 0.0 | 0.0 | 0.0 |
| N=5 | 1.81 | 0.34 | 0.02 | 0.0 | 0.0 |
| N=8 | 3.89 | 1.14 | 0.18 | 0.01 | 0.0 |

Idle cash %, last 10y:

| index | $25 | $50 | $100 | $200 | $500 |
|---|---|---|---|---|---|
| N=1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| N=2 | 0.02 | 0.0 | 0.0 | 0.0 | 0.0 |
| N=3 | 0.26 | 0.02 | 0.0 | 0.0 | 0.0 |
| N=5 | 0.86 | 0.23 | 0.01 | 0.0 | 0.0 |
| N=8 | 1.66 | 0.45 | 0.09 | 0.01 | 0.0 |

Largest-asset deviation pp, last 5y (plain `fractional=True` is identical to the 2nd decimal for X >= 50):

| index | $25 | $50 | $100 | $200 | $500 |
|---|---|---|---|---|---|
| N=1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| N=2 | 0.85 | 0.85 | 0.84 | 0.84 | 0.84 |
| N=3 | 0.91 | 0.77 | 0.77 | 0.77 | 0.77 |
| N=5 | 1.4 | 0.81 | 0.63 | 0.63 | 0.63 |
| N=8 | 1.39 | 0.89 | 0.61 | 0.57 | 0.57 |

Largest-asset deviation pp, last 10y:

| index | $25 | $50 | $100 | $200 | $500 |
|---|---|---|---|---|---|
| N=1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| N=2 | 0.99 | 0.98 | 0.97 | 0.97 | 0.97 |
| N=3 | 1.11 | 0.93 | 0.92 | 0.92 | 0.92 |
| N=5 | 1.23 | 0.94 | 0.83 | 0.83 | 0.83 |
| N=8 | 1.18 | 0.98 | 0.86 | 0.82 | 0.82 |

With fractional shares at X >= 100 the numbers are the same for every X: idle cash 0.0%, no month without a purchase (all 61/121 months bought), final-value shortfall vs plain fractional within +-0.03%, and a largest-asset deviation of 0.6-1.0pp (N >= 2), which is just the market moving between monthly purchases, not an X effect.

**Smallest X (on the grid) at which whole-share `buy_underweight` tracks target within 2pp** and stays within it for every larger X on the grid (fractional columns are 25 = the smallest X tested in every case but one):

| window | N | largest-asset dev (full window) | mean abs dev (full window) | largest-asset dev (last 12m only) |
|---|---|---|---|---|
| 5y | 1 | 1000 | 1000 | 300 |
| 5y | 2 | 2000 | 1000 | 300 |
| 5y | 3 | 1000 | 500 | 500 |
| 5y | 5 | 2000 | 500 | 500 |
| 5y | 8 | 2000 | 500 | 500 |
| 10y | 1 | 300 | 300 | 100 |
| 10y | 2 | 500 | 500 | none in grid |
| 10y | 3 | 500 | 200 | 200 |
| 10y | 5 | 500 | 200 | 200 |
| 10y | 8 | 500 | 100 | 200 |

"none in grid" (10y, N=2, last-12-months criterion) is not a whole-share effect: the fractional baseline itself is 2.17pp in that window, because VOO and BRK-B drifted apart between monthly purchases.

What the numbers show:
- Whole shares: with X = $500 and a 5-year window, mean idle cash is 2.5% (N=1), 4.7% (N=2), 5.4% (N=3), 7.0% (N=5), 9.7% (N=8); the full-window largest-asset deviation is 2.5-3.8pp; 0-9 of 61 months have no purchase at all. At X = $100 the same plan has 12-36% idle cash, 12-14pp mean deviation and 20-48 of 61 months with no purchase. At X >= $2000 every N is within about 1.3pp.
- The 2pp threshold on the full-window average is reached around X = $1000-2000 (5y) and $300-500 (10y, N >= 2); on the mean-abs-deviation measure around $500-1000 (5y) and $100-500 (10y). Several cells sit within 0.2pp of 2.0 (for example 5y N=2/3/5/8 at $1000 = 2.2/1.9/2.2/2.1), so the exact grid point is not meaningful; the stable reading is "tracking is within about 2pp once X is at least 1-2 times the price of the most expensive fund held ($693 VOO, $593 BRK-B)".
- More funds under whole shares cost idle cash roughly in proportion (idle cash rises with N at every X), but the dependence on X is far stronger than on N (N=1 to N=8 changes idle cash by about 4x at X = $500; going from X = $500 to $100 changes it about 4x for N=8 and 5x for N=1).
- Counter-intuitive: at small X more funds can mean FEWER empty months, because cheap funds (VXUS $83, SCHD $36, BND $71) can be bought while VOO/BRK-B cannot. At X = $200 (5y) months with no purchase are 35 (N=1), 34 (N=2), 16 (N=3), 11 (N=5), 4 (N=8). The plan then holds more cheap funds than its target, i.e. N=8 at small X is not "diversified as specified", it is skewed to the cheap sleeves. Final-value shortfall vs fractional is 0.2-1.5% at X = $500 and 2-28% at X <= $50 (5y; the 10y window is 0.3-0.8% and 4-27%).
- Fractional shares: the $5 minimum order matters only when target weight x X is a few dollars: at X = $25 and N=8 mean idle cash is 3.9% (5y) / 1.7% (10y) and deviation 1.4 / 1.2pp; at X >= $100 it is 0.2% or less. So with fractional shares X does not constrain N down to the lowest value tested ($25/month, N=8, still within 1.5pp and 4% idle).
- Windows: 5y and 10y windows overlap and share their end date, so they are not independent; both are one path of history.

## B. Diversification per extra fund (independent of X)

Common sample 2011-10-21 to 2026-11-13 (3,788 daily returns; start fixed by SCHD, first date 2011-10-20). Correlations of daily close-to-close returns:

| index | VOO | VTI | QQQ | BRK-B | VXUS | SCHD | VTV | RSP | BND | GLD |
|---|---|---|---|---|---|---|---|---|---|---|
| VOO | 1.0 | 1.0 | 0.93 | 0.74 | 0.85 | 0.88 | 0.93 | 0.94 | 0.08 | 0.09 |
| VTI | 1.0 | 1.0 | 0.92 | 0.73 | 0.86 | 0.87 | 0.92 | 0.95 | 0.08 | 0.1 |
| QQQ | 0.93 | 0.92 | 1.0 | 0.57 | 0.77 | 0.7 | 0.75 | 0.8 | 0.08 | 0.1 |
| BRK-B | 0.74 | 0.73 | 0.57 | 1.0 | 0.64 | 0.79 | 0.82 | 0.78 | 0.0 | -0.01 |
| VXUS | 0.85 | 0.86 | 0.77 | 0.64 | 1.0 | 0.77 | 0.83 | 0.85 | 0.11 | 0.23 |
| SCHD | 0.88 | 0.87 | 0.7 | 0.79 | 0.77 | 1.0 | 0.96 | 0.94 | 0.06 | 0.06 |
| VTV | 0.93 | 0.92 | 0.75 | 0.82 | 0.83 | 0.96 | 1.0 | 0.97 | 0.04 | 0.07 |
| RSP | 0.94 | 0.95 | 0.8 | 0.78 | 0.85 | 0.94 | 0.97 | 1.0 | 0.08 | 0.08 |
| BND | 0.08 | 0.08 | 0.08 | 0.0 | 0.11 | 0.06 | 0.04 | 0.08 | 1.0 | 0.29 |
| GLD | 0.09 | 0.1 | 0.1 | -0.01 | 0.23 | 0.06 | 0.07 | 0.08 | 0.29 | 1.0 |

Mixes rebalanced monthly (weights reset on the first trading day of each month, drift inside the month). Marginal columns are the change in volatility vs the previous row (the mixes change several weights at once, so this is not a pure one-fund effect). CAGR is shown for context only (the winners are known in hindsight):

| N | sleeves | cagr_pct | vol_pct | max_dd_pct | worst_12m_pct | vol_change_pp | vol_change_rel_pct |
|---|---|---|---|---|---|---|---|
| 1 | VOO 100 | 14.75 | 16.73 | -33.99 | -19.73 | - | - |
| 2 | VOO 50/BRK-B 50 | 14.86 | 16.31 | -31.67 | -18.9 | -0.42 | -2.49 |
| 3 | VOO 50/BRK-B 30/VXUS 20 | 13.45 | 15.87 | -32.47 | -20.35 | -0.44 | -2.7 |
| 5 | VOO 40/BRK-B 20/VXUS 20/QQQ 10/SCHD 10 | 13.79 | 15.86 | -32.19 | -19.19 | -0.01 | -0.06 |
| 8 | VOO 30/BRK-B 15/VXUS 15/QQQ 10/SCHD 10/VTV 5/BND 10/GLD 5 | 12.39 | 13.55 | -28.17 | -16.39 | -2.31 | -14.55 |

Long sample 1996-05-02 to 2026-11-13 (7,684 daily returns), mutual funds and BRK-A (open = close irrelevant here, only closes are used):

| mix | cagr_pct | vol_pct | max_dd_pct | worst_12m_pct | vol_change_pp |
|---|---|---|---|---|---|
| VFINX 100 | 10.25 | 19.04 | -55.25 | -47.5 | - |
| VFINX 60 / BRK-A 40 | 11.09 | 17.78 | -52.4 | -47.28 | -1.26 |
| VFINX 50 / BRK-A 30 / VGTSX 20 | 10.1 | 17.09 | -54.03 | -48.52 | -0.69 |
| VFINX 45 / BRK-A 25 / VGTSX 15 / VBMFX 15 | 9.39 | 14.43 | -47.55 | -42.39 | -2.66 |

What the numbers show:
- The U.S. large-cap and tilt funds are nearly the same exposure: VOO-VTI 1.00, VOO-RSP 0.94, VOO-VTV 0.93, VOO-QQQ 0.93, VOO-SCHD 0.88. Adding VTI/RSP/VTV/SCHD changes essentially nothing (N=5, which adds QQQ and SCHD to N=3, lowers vol by 0.01pp).
- The funds that actually change risk are BRK-B (0.74 with VOO; vol 16.73 -> 16.31, max drawdown -34.0% -> -31.7% at 50/50), VXUS (0.85; vol -0.44pp, but CAGR 14.86% -> 13.45% in this sample, a result of the U.S. outperformance 2011-2026) and above all BND and GLD (correlations 0.0-0.1 with equities and 0.29 with each other): the N=8 mix lowers vol by 2.3pp, max drawdown to -28.2%, worst 12 months to -16.4%.
- Long sample: adding BRK-A 40% lowered vol 19.0 -> 17.8% and raised CAGR (10.25% -> 11.09%) in this sample, but the max drawdown (-55.3% -> -52.4%) and worst 12 months (-47.5% -> -47.3%) barely moved; bonds (15%) are what cut the drawdown to -47.6% and worst-12m to -42.4%.
- None of this depends on X. Whether an extra fund is worth holding is a question about correlation and weight, not contribution size. All figures are from one historical path (about 15 and 30 years); BRK and QQQ are winners known in hindsight, so the CAGR column should not be read as the benefit of diversification.

## C. Catch-up from today's account

Start: NVDA $1,000, SPCX $1,000, BRK-B $2,000, VOO $700, "other" $2,300 (as QQQ) = $7,000. No selling; all new money to the core with fixed splits: core (i) VOO 70 / VXUS 30 (BRK counted as a single stock, so core starts at $700 = 10%); core (ii) VOO 50 / BRK-B 30 / VXUS 20 (core starts at $2,700 = 38.6%). The split of new money does not matter for the zero-return maths, only for the simulation. "Months" = number of monthly contributions until the condition first holds (0 = already true). The first contribution is made on the start month's open, checked at that open after the purchase.

(1) Analytic, zero returns (raw numbers; anything above 120 is beyond the cap used in the simulations). Concentration: NVDA+SPCX = $2,000 of an account that grows by X per month.

Core (i) VOO+VXUS, keep "other":

| index | 100 | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|---|
| core>=50% | 56 | 28 | 19 | 12 | 6 | 3 | 2 | 2 |
| core>=80% | 245 | 123 | 82 | 49 | 25 | 13 | 9 | 5 |
| NVDA+SPCX<=20% | 30 | 15 | 10 | 6 | 3 | 2 | 1 | 1 |
| NVDA+SPCX<=10% | 130 | 65 | 44 | 26 | 13 | 7 | 5 | 3 |

Core (i), one-off sale of the $2,300 "other" into the core at the start:

| index | 100 | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|---|
| core>=50% | 10 | 5 | 4 | 2 | 1 | 1 | 1 | 1 |
| core>=80% | 130 | 65 | 44 | 26 | 13 | 7 | 5 | 3 |
| NVDA+SPCX<=20% | 30 | 15 | 10 | 6 | 3 | 2 | 1 | 1 |
| NVDA+SPCX<=10% | 130 | 65 | 44 | 26 | 13 | 7 | 5 | 3 |

Core (ii) VOO+BRK-B+VXUS, keep "other":

| index | 100 | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|---|
| core>=50% | 16 | 8 | 6 | 4 | 2 | 1 | 1 | 1 |
| core>=80% | 145 | 73 | 49 | 29 | 15 | 8 | 5 | 3 |
| NVDA+SPCX<=20% | 30 | 15 | 10 | 6 | 3 | 2 | 1 | 1 |
| NVDA+SPCX<=10% | 130 | 65 | 44 | 26 | 13 | 7 | 5 | 3 |

Core (ii), sell "other" into the core:

| index | 100 | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|---|
| core>=50% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| core>=80% | 30 | 15 | 10 | 6 | 3 | 2 | 1 | 1 |
| NVDA+SPCX<=20% | 30 | 15 | 10 | 6 | 3 | 2 | 1 | 1 |
| NVDA+SPCX<=10% | 130 | 65 | 44 | 26 | 13 | 7 | 5 | 3 |

Selling "other" does not change the NVDA+SPCX share times (the proceeds stay in the account, so the total is unchanged); it only moves the core share. With core (ii) it makes the account already 71% core, so the 50% target is met at month 0.

(2) Simulated with real prices. Start months 2010-07 to 2026-11 (197 starts), each followed for up to 120 monthly contributions; NVDA, BRK-B, QQQ are actual; VOO is spliced with SPY before 2010-09-09 and VXUS with EFA (scaled to match) before 2011-02. SPCX is proxied (a) by TSLA, (b) by 1.5 x QQQ daily returns (close-to-close and overnight). Months are capped at 120 (a 120 means "not reached within 120 contributions"); starts whose data end before 120 months and that have not yet hit the goal are dropped as censored, so late starts only contribute when they hit quickly (n starts per cell: 78-197). Entries are median (p10-p90) over start months. Starts overlap heavily and cover a bull market, so the effective number of independent paths is roughly 16 / (horizon in years) and the p10/p90 are not tight; the two SPCX proxies give nearly the same numbers.

**core>=50%: simulated median (p10-p90) months, SPCX proxied by TSLA**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (59-120) | 28 (19-51) | 14 (11-28) | 7 (6-8) | 3 (3-4) | 2 (2-3) | 2 (2-2) |
| i: VOO+VXUS, sell other into core | 14 (7-120) | 7 (4-23) | 4 (3-7) | 3 (2-4) | 2 (2-2) | 1 (1-1) | 1 (1-1) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, keep other | 24 (10-120) | 11 (6-34) | 7 (5-15) | 4 (3-6) | 2 (2-3) | 1 (1-1) | 1 (1-1) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, sell other into core | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) |

**core>=50%: same, SPCX proxied by 1.5 x QQQ**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (38-120) | 26 (20-44) | 14 (11-20) | 7 (6-7) | 3 (3-4) | 2 (2-2) | 2 (2-2) |
| i: VOO+VXUS, sell other into core | 15 (9-120) | 7 (5-12) | 4 (3-6) | 3 (2-3) | 2 (2-2) | 1 (1-1) | 1 (1-1) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, keep other | 25 (11-120) | 11 (6-29) | 7 (5-12) | 4 (3-5) | 2 (2-2) | 1 (1-1) | 1 (1-1) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, sell other into core | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) |

**core>=80%: simulated median (p10-p90) months, SPCX proxied by TSLA**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (120-120) | 57 (28-120) | 16 (12-28) | 10 (8-14) | 6 (5-7) |
| i: VOO+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (25-120) | 18 (12-42) | 8 (6-13) | 5 (5-6) | 3 (3-4) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (41-120) | 21 (13-46) | 9 (7-15) | 6 (5-8) | 4 (3-4) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (9-120) | 21 (7-120) | 14 (6-120) | 8 (4-32) | 4 (3-8) | 2 (2-3) | 2 (2-2) | 1 (1-1) |

**core>=80%: same, SPCX proxied by 1.5 x QQQ**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (120-120) | 47 (29-120) | 15 (12-21) | 10 (8-12) | 6 (5-6) |
| i: VOO+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (31-120) | 17 (13-34) | 8 (7-9) | 5 (5-6) | 3 (3-3) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (35-120) | 20 (14-36) | 9 (7-12) | 6 (5-7) | 4 (3-4) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (9-120) | 21 (8-120) | 16 (7-120) | 8 (5-29) | 4 (3-6) | 2 (2-2) | 2 (2-2) | 1 (1-1) |

**NVDA+SPCX<=20%: simulated median (p10-p90) months, SPCX proxied by TSLA**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (10-120) | 20 (8-120) | 14 (6-120) | 8 (4-34) | 4 (3-7) | 2 (2-3) | 2 (2-2) | 1 (1-1) |
| i: VOO+VXUS, sell other into core | 120 (9-120) | 20 (8-120) | 15 (6-120) | 8 (4-34) | 4 (3-8) | 2 (2-3) | 2 (2-2) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (10-120) | 20 (8-120) | 14 (6-120) | 8 (4-34) | 4 (3-7) | 2 (2-3) | 2 (2-2) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (9-120) | 21 (7-120) | 14 (6-120) | 8 (4-32) | 4 (3-8) | 2 (2-3) | 2 (2-2) | 1 (1-1) |

**NVDA+SPCX<=20%: same, SPCX proxied by 1.5 x QQQ**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (11-120) | 22 (8-120) | 15 (7-120) | 8 (5-29) | 4 (3-6) | 2 (2-2) | 2 (2-2) | 1 (1-1) |
| i: VOO+VXUS, sell other into core | 120 (11-120) | 24 (8-120) | 16 (7-120) | 8 (5-29) | 4 (3-6) | 2 (2-2) | 2 (2-2) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (11-120) | 22 (8-120) | 15 (7-120) | 8 (5-29) | 4 (3-6) | 2 (2-2) | 2 (2-2) | 1 (1-1) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (9-120) | 21 (8-120) | 16 (7-120) | 8 (5-29) | 4 (3-6) | 2 (2-2) | 2 (2-2) | 1 (1-1) |

**NVDA+SPCX<=10%: simulated median (p10-p90) months, SPCX proxied by TSLA**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (24-120) | 120 (16-120) | 19 (10-120) | 8 (6-26) | 5 (4-9) | 3 (3-4) |
| i: VOO+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (16-120) | 19 (10-120) | 8 (6-26) | 5 (4-9) | 3 (3-5) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (24-120) | 120 (16-120) | 19 (10-120) | 8 (6-26) | 5 (4-9) | 3 (3-4) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (16-120) | 19 (10-120) | 8 (6-27) | 5 (4-9) | 3 (3-4) |

**NVDA+SPCX<=10%: same, SPCX proxied by 1.5 x QQQ**

| row | $100 | $200 | $300 | $500 | $1000 | $2000 | $3000 | $5000 |
|---|---|---|---|---|---|---|---|---|
| i: VOO+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (21-120) | 20 (11-120) | 8 (6-17) | 5 (4-8) | 3 (3-4) |
| i: VOO+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (21-120) | 20 (11-120) | 8 (6-17) | 5 (4-8) | 3 (3-4) |
| ii: VOO+BRK-B+VXUS, keep other | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (21-120) | 20 (11-120) | 8 (6-17) | 5 (4-8) | 3 (3-4) |
| ii: VOO+BRK-B+VXUS, sell other into core | 120 (120-120) | 120 (120-120) | 120 (120-120) | 120 (21-120) | 20 (11-120) | 8 (6-18) | 5 (4-8) | 3 (3-4) |


Spot check against `DCABacktest` (start 2015-01, TSLA proxy, core ii, X=500, fixed_buy 50/30/20): engine gives core >= 50% in month 4 and NVDA+SPCX <= 20% in month 8, the vectorised simulation gives 4 and 8; core >= 80% is not reached within 120 in either.

What the numbers show:
- With zero returns the time to 50% core scales as 1/X: core (i) takes 56 / 28 / 19 / 12 / 6 / 3 months at X = $100 / 200 / 300 / 500 / 1000 / 2000; 80% needs 245 / 123 / 82 / 49 / 25 / 13 months. Core (ii) is about 3-4 times faster to 50% (16 / 8 / 6 / 4 / 2 / 1).
- Real prices are materially slower than zero returns for the 80% core and the concentration goals, because the concentrated positions kept growing: the median for NVDA+SPCX <= 20% at X = $100 is 120+ (analytic 30), at $500 it is 8 (analytic 6), and NVDA+SPCX <= 10% at X = $500 is 120+ in the median (analytic 26). For core >= 80% (ii) at X = $500 the median is 120+ (analytic 29), p10 41. NVDA (and TSLA) are well-known winners in hindsight, so these simulated delays are probably an upper bound for a random stock; a stock that falls would reach the goal faster.
- Practical read, X = $500-1000: core (ii) reaches >= 50% in about 2-4 months, but dilution of NVDA+SPCX to 10% takes about 20 months at $1000 and is not reached in a typical history at $500; at X = $100 NVDA+SPCX <= 20% is not reached within 10 years in the median (analytic 30 months), and at $200 it takes a median of about 20 months, so at small X the concentration (not the core fraction) is what stays for years.
- Selling the $2,300 "other" is the biggest lever at small X: with core (i) and X = $200, the 50% core goal drops from a median of 120+ months to 7; with core (ii) the 50% goal is met immediately. It does nothing for the NVDA+SPCX share. Whether to sell is a separate (tax, conviction) question not addressed here.

## D. When do contributions stop being able to rebalance?

Constant X = $500 from an empty account (fractional, zero commission), `buy_underweight`; also contributions growing 5% a year (raised once per year, from $500). Each month: ratio = contribution / account value at the previous close (before the deposit), and the largest absolute asset-weight deviation from target measured at the last close before the contribution ("pre", the drift the contribution must correct) and at the close of the decision day ("post"). The first month is excluded (empty account). Buckets by ratio: >5% means the account is below 20 x X; 2-5% = 20-50 x X; 1-2% = 50-100 x X; 0.5-1% = 100-200 x X; <0.5% = above 200 x X (for X=$500: $100k).

**VOO 50 / BRK-B 30 / VXUS 20, 2011-02 to 2026-11 (189 months)**

| growth | bucket | months | mean_dev_pre | median_dev_pre | p90_dev_pre | max_dev_pre | share_above_2pp | share_above_5pp | max_dev_post |
|---|---|---|---|---|---|---|---|---|---|
| constant | >5% | 18 | 0.66 | 0.67 | 1.12 | 1.34 | 0.0 | 0.0 | 0.41 |
| constant | 2-5% | 19 | 0.54 | 0.52 | 0.78 | 1.35 | 0.0 | 0.0 | 0.35 |
| constant | 1-2% | 34 | 0.75 | 0.71 | 1.37 | 1.46 | 0.0 | 0.0 | 1.53 |
| constant | 0.5-1% | 46 | 0.81 | 0.66 | 1.82 | 3.0 | 8.7 | 0.0 | 2.76 |
| constant | <0.5% | 72 | 1.97 | 1.97 | 3.33 | 4.92 | 48.61 | 0.0 | 4.78 |
| growing 5%/yr | >5% | 19 | 0.66 | 0.64 | 1.1 | 1.34 | 0.0 | 0.0 | 0.41 |
| growing 5%/yr | 2-5% | 21 | 0.59 | 0.5 | 1.25 | 1.46 | 0.0 | 0.0 | 0.63 |
| growing 5%/yr | 1-2% | 38 | 0.68 | 0.65 | 1.18 | 1.42 | 0.0 | 0.0 | 1.48 |
| growing 5%/yr | 0.5-1% | 60 | 1.1 | 0.7 | 2.39 | 4.18 | 20.0 | 0.0 | 3.6 |
| growing 5%/yr | <0.5% | 51 | 1.66 | 1.47 | 2.88 | 4.43 | 35.29 | 0.0 | 4.1 |

**SPY 50 / QQQ 20 / BRK-B 30, 1999-03 to 2026-11 (332 months)**

| growth | bucket | months | mean_dev_pre | median_dev_pre | p90_dev_pre | max_dev_pre | share_above_2pp | share_above_5pp | max_dev_post |
|---|---|---|---|---|---|---|---|---|---|
| constant | >5% | 17 | 2.0 | 2.04 | 3.33 | 4.66 | 52.94 | 0.0 | 2.75 |
| constant | 2-5% | 36 | 2.1 | 1.9 | 4.13 | 5.62 | 47.22 | 2.78 | 5.49 |
| constant | 1-2% | 32 | 0.8 | 0.77 | 1.26 | 2.64 | 3.12 | 0.0 | 2.17 |
| constant | 0.5-1% | 61 | 2.06 | 1.4 | 4.53 | 6.72 | 34.43 | 9.84 | 6.69 |
| constant | <0.5% | 186 | 3.15 | 2.13 | 6.73 | 8.57 | 52.15 | 25.81 | 8.7 |
| growing 5%/yr | >5% | 18 | 1.94 | 1.73 | 3.31 | 4.66 | 44.44 | 0.0 | 2.75 |
| growing 5%/yr | 2-5% | 37 | 2.0 | 1.89 | 3.86 | 5.49 | 40.54 | 2.7 | 5.32 |
| growing 5%/yr | 1-2% | 47 | 1.2 | 0.99 | 2.01 | 5.9 | 10.64 | 4.26 | 5.6 |
| growing 5%/yr | 0.5-1% | 68 | 1.5 | 1.0 | 3.55 | 6.03 | 20.59 | 2.94 | 5.86 |
| growing 5%/yr | <0.5% | 162 | 2.53 | 2.1 | 5.1 | 6.08 | 50.0 | 12.35 | 6.22 |

Summary:

| config | growth | months | months_above_5pp | first_above_5pp | ratio_at_first | max_dev_when_ratio_below_2pp | months_above_5pp_when_ratio_below_2pp | months_ratio_below_2pp |
|---|---|---|---|---|---|---|---|---|
| SPY50/QQQ20/BRK-B30 (1999-03+) | constant | 332 | 55 | 2001-01-02 | 0.045 | 8.57 | 54 | 279 |
| SPY50/QQQ20/BRK-B30 (1999-03+) | growing 5%/yr | 332 | 25 | 2001-01-02 | 0.0464 | 6.08 | 24 | 277 |
| VOO50/BRK-B30/VXUS20 (2011-02+) | constant | 189 | 0 | - | - | 4.92 | 0 | 152 |
| VOO50/BRK-B30/VXUS20 (2011-02+) | growing 5%/yr | 189 | 0 | - | - | 4.43 | 0 | 149 |

(`months_ratio_below_2pp` = number of months with ratio below 2%.)

What the numbers show:
- For the VOO/BRK-B/VXUS plan, deviation never reached 5pp in 15 years (max 4.92pp, at a ratio below 0.5%); even at ratio >= 1% the maximum was 1.5pp. Deviation above 2pp only appeared at ratios below about 1% (account above roughly 100 x X), and in those buckets 9-49% of months exceeded 2pp.
- For the SPY/QQQ/BRK-B plan, which includes a volatile sleeve and a dot-com crash, 5pp was exceeded in 55 of 332 months (constant X); one episode was at a ratio of 4.5% (2001-01, account $11k), the rest at ratios below 1.2%: 2008 (ratio 0.7-0.9%, account about $67k), 2020-2022 (ratio about 0.1%, account $430-650k) and 2024-2026 (ratio about 0.05%, account $0.8-1.1M, up to 8.6pp). At ratio below 0.5%, 26% of months exceed 5pp (constant X, 186 months) vs 12% with 5%-per-year growing contributions (162 months, max 6.1pp).
- At ratios below 2% the contribution can correct at most about that much of the account per month, so a 5-8pp drift is out of reach for new money alone (moving 5pp of weight with new money takes roughly 5pp / ratio months of the entire contribution: about 5 months at a ratio of 1%, about 50 at 0.1%). Rule of thumb from these two paths: the contribution still pulls weights back mostly effectively while the account is below about 100 x X (ratio above 1%; one growing-contribution case still showed 5.9pp at ratio 1-2%); between 100 and 200 x X drift above 5pp starts to appear when a sleeve is volatile (QQQ; 2008), and above about 200 x X it is common for that plan (and absent for the VOO/BRK-B/VXUS plan).
- Caveat: this is two historical paths with a handful of drift episodes (2000-02, 2008, 2020-22, 2023-26), not a distribution; the ratio is correlated with calendar time, so the effect of the ratio cannot be separated cleanly from the effect of the market episode. The dot-com case shows that a large drift can occur at a high ratio when one sleeve moves 50%+ in a few months.

## E. What depends on X: IRR ranking and projected balances

Scale invariance. `rolling_dca` on VFINX, BRK-A, VGTSX, VBMFX (sample 1996-05+, 10-year horizon, 247 windows) for four mixes, once with $100 and once with $5,000 per month:

| mix | n_windows | median_irr_X100 | median_irr_X5000 | max_abs_irr_diff | rank_X100 | rank_X5000 |
|---|---|---|---|---|---|---|
| VFINX 100 | 247 | 10.789% | 10.789% | 0.0e+00 | 2 | 2 |
| 60/40 VFINX/BRK-A | 247 | 10.839% | 10.839% | 0.0e+00 | 1 | 1 |
| 50/30/20 +VGTSX | 247 | 9.625% | 9.625% | 0.0e+00 | 3 | 3 |
| 45/25/15/15 +VBMFX | 247 | 8.861% | 8.861% | 0.0e+00 | 4 | 4 |

The money-weighted IRR (and wealth multiple) is identical to machine precision, and the ranking is the same. With fractional shares, zero commission and slippage proportional to trade size, IRR is scale-invariant: the ranking of mixes does not depend on X. It would depend on X only through whole-share rounding / minimum orders (A) or fixed fees, which are zero here.

What does scale is the dollar outcome. Historical IRR of a VFINX 60 / BRK-A 40 DCA over 20-year horizons (`rolling_dca`, 321 monthly start dates 1980-03-17 to 2006-11-01, i.e. 1980-2026 data, about 2 independent 20-year paths): p10 7.9%, median 10.7%, p90 18.3%. The 1980-2006 starts are dominated by the 1980s-90s bull market and BRK-A's record, so these rates are generous.

Illustrative arithmetic only, not a forecast: balance in nominal dollars after 10 / 20 / 30 years of X per month, contributions at the start of each month, a constant annual return equal to the historical IRR below, no taxes or inflation adjustment. Contributions only (0%):

| index | 10y | 20y | 30y |
|---|---|---|---|
| $100/mo | 12,000 | 24,000 | 36,000 |
| $200/mo | 24,000 | 48,000 | 72,000 |
| $300/mo | 36,000 | 72,000 | 108,000 |
| $500/mo | 60,000 | 120,000 | 180,000 |
| $1000/mo | 120,000 | 240,000 | 360,000 |
| $2000/mo | 240,000 | 480,000 | 720,000 |
| $3000/mo | 360,000 | 720,000 | 1,080,000 |
| $5000/mo | 600,000 | 1,200,000 | 1,800,000 |

At the p10 20-year IRR (7.9%):

| index | 10y | 20y | 30y |
|---|---|---|---|
| $100/mo | 18,032 | 56,597 | 139,077 |
| $200/mo | 36,064 | 113,194 | 278,153 |
| $300/mo | 54,096 | 169,791 | 417,230 |
| $500/mo | 90,159 | 282,985 | 695,383 |
| $1000/mo | 180,319 | 565,969 | 1,390,767 |
| $2000/mo | 360,637 | 1,131,939 | 2,781,534 |
| $3000/mo | 540,956 | 1,697,908 | 4,172,301 |
| $5000/mo | 901,593 | 2,829,846 | 6,953,835 |

At the median 20-year IRR (10.7%):

| index | 10y | 20y | 30y |
|---|---|---|---|
| $100/mo | 20,930 | 78,883 | 239,341 |
| $200/mo | 41,861 | 157,765 | 478,683 |
| $300/mo | 62,791 | 236,648 | 718,024 |
| $500/mo | 104,652 | 394,414 | 1,196,707 |
| $1000/mo | 209,304 | 788,827 | 2,393,415 |
| $2000/mo | 418,608 | 1,577,654 | 4,786,829 |
| $3000/mo | 627,913 | 2,366,481 | 7,180,244 |
| $5000/mo | 1,046,521 | 3,944,136 | 11,967,073 |

What the numbers show: balances are exactly proportional to X (a plan at $500 is 5x one at $100), so X sets the dollar outcome, while the choice of mix is evaluated per dollar and is independent of X. Doubling X outweighs the difference between the p10 and median rates over these horizons (X=$1000 at p10 gives $180k after 10 years vs $105k for X=$500 at the median; $1.39M vs $1.20M after 30 years), while the spread between the two rates grows with the horizon (30 years at X=$500: $695k vs $1.20M). Those rates come from one historical path with overlapping windows, so the p10 is a statement about two or three independent 20-year spans, not a probability.

## Caveats
- Overlapping windows are highly autocorrelated; independent observations are about sample_years / horizon_years (A: about 1-2, C: about 2-3, D: 2 paths, E: about 2).
- The whole-share analysis uses current price levels on a 5-10 year past path (adjusted prices differ from actual prices by the dividends accumulated since, small near the end of the window). The $5 Webull minimum is taken as given; no other broker rules (e.g. fractional unavailable for some tickers or order types) were modelled.
- `buy_underweight` for whole shares splits cash by fractions and floors each asset separately, so the idle cash and no-buy counts are those of this simple policy; a smarter greedy whole-share rule would do somewhat better at small X.
- Survivorship: Berkshire, NVDA, QQQ, TSLA are winners known in hindsight. The SPCX proxies (TSLA, 1.5 x QQQ) are not SPCX; SPCX has only 108 trading days of its own history.
- B: marginal vol changes are not pure single-fund effects because the mixes change several weights at once.
