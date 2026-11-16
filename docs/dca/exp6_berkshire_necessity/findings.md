# EXP6: is Berkshire needed in the core, or is a pure S&P 500 core enough?

Script: `scripts/dca/exp6_berkshire_necessity.py` (about 1 minute; fixed seed). All generated tables (more than shown here) are in `tables.md` and the CSVs in this folder.
Data end: Berkshire/S&P 2026-11-13 (bootstrap pool ends 2026-10), Fama-French files 2026-09-30. BRK-A starts 1980-03, so the first full monthly return is 1980-04.

## F. Decision table (put first)

20-year monthly DCA, stationary bootstrap (mean block 18 months, 10,000 paths, fixed seed), pure S&P (VFINX) vs 70% S&P / 30% BRK-A per contribution (fixed split, no rebalancing). "Paired" differences are computed path by path; the bracket is the 5th-95th percentile of that paired difference across simulated futures (a spread of outcomes, not a confidence interval of the mean; the Monte-Carlo standard error of each mean difference is about 0.02 pp). Drawdown is on the time-weighted NAV (monthly resolution).

| Scenario for Berkshire's return process | Expected-return change: mean paired IRR diff, pp/yr [5-95%] (median) | Bad case: p10 IRR, pure -> mix (change, pp) | Median max drawdown pure -> mix; paired mean change, pp [5-95%] | Terminal wealth p10 ratio, mix/pure |
|---|---|---|---|---|
| 1. Raw 1980-2026 history | +3.43 [-0.10, +8.60] (+2.96) | 6.53 -> 9.38 (+2.84) | -36.1 -> -29.7; +4.8 [-2.0, +16.8] | 1.39 |
| 2. 2000-2026 history only | +0.92 [-0.79, +3.57] (+0.64) | 2.83 -> 4.80 (+1.97) | -46.0 -> -37.9; +7.1 [-0.1, +19.1] | 1.24 |
| 3. 2013-2026 history only | -0.15 [-1.11, +1.03] (-0.22) | 11.68 -> 11.65 (-0.03) | -24.0 -> -21.2; +1.8 [-0.9, +4.3] | 1.00 |
| 4. **No edge** (BRK mean log return = S&P's, 1980-2026 pool) | **+0.35 [-1.59, +3.52] (0.00)** | **6.53 -> 6.82 (+0.29)** | **-36.1 -> -31.5; +2.8 [-2.3, +11.2]** | **1.03** |
| 4b. No edge, 2000-2026 pool | +0.12 [-1.26, +2.33] (-0.12) | 2.83 -> 3.85 (+1.02) | -46.0 -> -39.0; +5.9 [-0.2, +16.5] | 1.12 |
| 5. Negative edge (-1 pp/yr) | +0.01 [-1.73, +2.90] (-0.31) | 6.53 -> 6.50 (-0.03) | -36.1 -> -31.9; +2.6 [-2.4, +10.4] | 1.00 |
| 6. No edge + 10% chance of a one-off -50% | +0.25 [-1.66, +3.38] (-0.09) | 6.53 -> 6.69 (+0.16) | -36.1 -> -31.7; +2.7 [-2.8, +11.0] | 1.02 |

P(mix beats pure on IRR): 94% (raw), 73% (2000+), 37% (2013+), **50.0% (no edge)**, 41% (negative edge), 47% (no edge + 10% blow-up).

Headline under "no edge" (scenario 4, 30% Berkshire vs 0%, 20 years):
- Median IRR: no change (paired median difference 0.00; median-of-mix minus median-of-pure +0.21 pp). Indistinguishable from zero.
- p10 IRR: +0.29 pp. Monte-Carlo noise across six independent 10,000-path repeats: +0.18 to +0.29 (sd 0.05). Block-length sensitivity: +0.32 (6 months), +0.29 (18), -0.04 (36). The p10 gain is above Monte-Carlo noise but as large as the block-length sensitivity and vanishes at 36 months. Treat it as "between 0 and +0.3 pp".
- Max drawdown: median -36.1% -> -31.5%; paired mean change +2.8 pp, positive at every block length (+1.8 at 6, +2.8 at 18, +3.0 at 36) and every horizon (10y +2.2, 20y +2.8, 30y +3.1). Larger than the Monte-Carlo noise (about 0.05 pp) and the block-length spread. It is the one benefit that survives. The worst-12-month figure does not improve (p10 worst: -43.3% -> -44.7%).
- A monthly-rebalanced variant (scenario 4, 20y, 30%) adds a small rebalancing premium: median IRR +0.30 pp above pure (vs 0.00 unrebalanced), p10 IRR 6.89 vs 6.53, median max DD -31.1%.

Risks that a bootstrap cannot see (none are in the table):
- Single-company risk. The "no edge" and blow-up scenarios are stylised; a real failure (reserving, a large acquisition or investment going wrong, a catastrophe year) is a left-tail event of unknown frequency. Scenario 6 says a 10% chance of a -50% event costs the 30% mix 0.1 pp of mean IRR and 0.1 pp of median IRR.
- Succession. Buffett stepped down as CEO at end-2025; the 1980-2026 history is mostly the Buffett-Munger period. The pools include only about 10 months of 2026.
- Concentration of US-situs assets. Berkshire is a US-listed stock, as are US-domiciled ETFs; for a Singapore resident both carry US estate-tax exposure above the nonresident threshold (US$60,000). An Irish-domiciled S&P or all-world fund avoids this, Berkshire cannot. Not a return effect; not modelled.
- Regime effects: the bootstrap reshuffles one history, cannot create regimes that never occurred and destroys mean reversion beyond the block length. "No edge" is an assumption, not an estimate.

Plain reading: whether Berkshire adds expected return depends entirely on whether its pre-2000 outperformance repeats. If it does not (no edge), the median outcome does not change, the bad case improves by 0 to 0.3 pp/yr, and a shallower drawdown (about 3-5 pp on a -36% median) is the only consistent effect. With a -1 pp/yr edge the mix is slightly worse at the median (-0.3 pp). Berkshire is not required for a pure S&P core to work; it is a modest drawdown diversifier whose return contribution cannot be separated from zero with this data.

## A. What is Berkshire made of? (monthly regressions of BRK-A excess return)

Sample 1980-04 to 2026-09 (n = 558). y = BRK-A monthly return minus RF. Newey-West HAC t-statistics, 6 lags. Alpha = monthly intercept x 12, pp/yr.

| era | model | n | alpha pp/yr | NW t | 95% CI | R2 |
|---|---|---|---|---|---|---|
| 1980-2026 | CAPM | 558 | 9.63 | 3.57 | [4.35, 14.91] | 0.22 |
| 1980-2026 | FF3 | 558 | 7.44 | 2.91 | [2.43, 12.44] | 0.30 |
| 1980-2026 | FF5+Mom | 558 | 5.30 | 2.10 | [0.36, 10.24] | 0.32 |
| 1980-1999 | FF3 | 237 | 10.78 | 2.34 | [1.73, 19.82] | 0.28 |
| 1980-1999 | FF5+Mom | 237 | 7.22 | 1.58 | [-1.74, 16.19] | 0.30 |
| 2000-2012 | FF3 | 156 | 5.21 | 1.34 | [-2.40, 12.82] | 0.33 |
| 2000-2012 | FF5+Mom | 156 | 5.60 | 1.23 | [-3.29, 14.48] | 0.34 |
| 2013-2026 | FF3 | 165 | 1.08 | 0.49 | [-3.28, 5.44] | 0.54 |
| 2013-2026 | FF5+Mom | 165 | 0.77 | 0.36 | [-3.48, 5.03] | 0.55 |
| 2000-2026 | CAPM | 321 | 5.45 | 2.10 | [0.36, 10.54] | 0.21 |
| 2000-2026 | FF3 | 321 | 4.08 | 1.86 | [-0.23, 8.39] | 0.41 |
| 2000-2026 | FF5+Mom | 321 | 3.06 | 1.29 | [-1.58, 7.69] | 0.42 |

Loadings (FF5+Mom; NW t in brackets):

| era | Mkt-RF | SMB | HML | RMW | CMA | Mom |
|---|---|---|---|---|---|---|
| 1980-2026 | 0.82 (11.7) | -0.22 (-2.1) | 0.42 (3.8) | 0.39 (4.3) | 0.05 (0.3) | 0.03 (0.6) |
| 1980-1999 | 1.09 (11.1) | 0.05 (0.3) | 0.46 (1.8) | 0.55 (2.5) | 0.39 (1.1) | 0.01 (0.1) |
| 2000-2012 | 0.45 (3.4) | -0.41 (-2.4) | 0.64 (3.4) | 0.08 (0.5) | -0.25 (-1.1) | -0.03 (-0.5) |
| 2013-2026 | 0.82 (8.7) | -0.34 (-2.9) | 0.48 (3.2) | 0.18 (1.6) | 0.14 (0.7) | 0.02 (0.3) |
| 2000-2026 | 0.65 (7.4) | -0.36 (-3.4) | 0.53 (4.4) | 0.23 (2.7) | -0.05 (-0.3) | -0.01 (-0.2) |

Rolling 10-year (120-month) alpha and market beta: `fig1_rolling_alpha_beta.png`, data `A_rolling_10y.csv`. The FF5+Mom alpha of the window ending 2026-09 is -0.01 pp/yr (CAPM +2.5); its market beta is 0.80. Rolling FF5+Mom alpha peaked at +19.8 (window ending 1998-06) and has trended toward zero. Its 95% band excludes zero in 62% of windows ending 1995-99 and in about 2% of later windows (never on the negative side). Rolling market beta ran from about 1.0-1.1 in the 1990s to about 0.35 around 2010 and about 0.7-0.8 now.

What this shows:
- Berkshire is a 0.65-0.8 beta, value-tilted (HML +0.5), large-cap (SMB -0.35), high-profitability (RMW +0.2 to +0.5) stock. Factors explain only 42% of post-2000 monthly variance (55% since 2013); the rest is idiosyncratic.
- Post-2000 alpha: FF5+Mom +3.1 pp/yr, t = 1.29, 95% CI [-1.6, +7.7]; FF3 +4.1, t = 1.86 (borderline); only the CAPM alpha (+5.5, t = 2.1) is significant at 5%, and it mostly credits value and large-cap exposure as alpha. On the 6-factor model post-2000 alpha is **not distinguishable from zero**.
- Post-2013 alpha: +0.8 pp/yr (FF5+Mom), t = 0.36, CI [-3.5, +5.0]: **not distinguishable from zero**, but equally not distinguishable from +4 pp/yr; 13 years of monthly data cannot resolve an alpha smaller than about 4 pp/yr.
- The decay 7.2 (1980-99) -> 5.6 (2000-12) -> 0.8 (2013-26) is a point-estimate decay. The 1980-99 6-factor alpha is itself not significant (t = 1.58) and all era CIs overlap, so "alpha disappeared" is consistent with the data but not proven. Only the full-sample FF5+Mom alpha is significant (t = 2.10, CI [0.4, 10.2]), and it is dominated by the 1980s-90s.
- Loadings are unstable (market beta 1.09 -> 0.45 -> 0.82), which matters in B.

## B. Berkshire-style sleeves without Buffett

Zero-alpha replicas: return = RF + sum(beta_k x factor_k), no intercept, built on daily factors so they can be DCA'd. Loadings estimated 2000-01..2026-09: **R6** (FF5+Mom; Mkt 0.65, SMB -0.36, HML 0.53, RMW 0.23, CMA -0.05, Mom -0.01; available 1963-07+) and **R3** (FF3; Mkt 0.62, SMB -0.45, HML 0.59; 1926-07+). **R6_oos / R3_oos** use 1980-04..1999-12 loadings (Mkt 1.09 / 1.03). Style proxies: **VALUE** (FF top-30% book/market), **MONEY** (FF financials), **LOWB3** (equal mix of the three lowest-beta industries by 1926-1999 daily beta: Telcm 0.68, NoDur 0.75, Hlth 0.84). Contrast: **TECH** (FF BusEq). R6/R3 are in-sample for 2000-26.

Tracking of actual BRK-A, monthly, 2000-01..2026-09 (n = 321); window returns are cumulative %:

| series | ann. ret % | ann. vol % | corr w BRK | tracking err % | max DD (monthly) % | 2000-03..2002-09 | 2007-11..2009-02 | 2022 | 2025-02..04 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| BRK-A | 10.2 | 18.0 | 1.00 | 0 | -44.5 | +68.0 | -40.7 | +4.0 | +13.9 | +10.9 |
| S&P (VFINX) | 8.2 | 15.2 | 0.49 | 16.9 | -51.0 | -38.3 | -51.0 | -18.2 | -7.5 | +17.7 |
| R6 | 8.2 | 11.5 | 0.65 | 13.7 | -37.7 | +31.9 | -36.2 | +1.8 | +0.5 | +18.8 |
| R3 | 7.4 | 12.0 | 0.64 | 13.7 | -41.9 | +25.3 | -40.2 | +6.5 | +2.1 | +22.9 |
| R6_oos | 14.2 | 17.7 | 0.60 | 16.0 | -52.7 | +75.0 | -49.8 | +3.0 | -7.4 | +14.3 |
| R3_oos | 10.0 | 17.0 | 0.57 | 16.2 | -58.1 | -7.9 | -56.3 | -6.3 | -5.6 | +24.4 |
| VALUE | 8.5 | 19.2 | 0.47 | 19.1 | -60.2 | -37.8 | -58.0 | -5.6 | -2.7 | +23.2 |
| MONEY | 7.7 | 19.7 | 0.57 | 17.5 | -72.7 | +14.7 | -69.9 | -10.0 | -9.6 | +15.1 |
| LOWB3 | 6.9 | 12.9 | 0.51 | 15.9 | -38.4 | -29.5 | -37.7 | -9.6 | +2.3 | +12.0 |
| TECH | 9.0 | 23.4 | 0.22 | 26.3 | -79.6 | -78.8 | -50.0 | -32.3 | -11.7 | +24.8 |

Figure: `fig2_replica_vs_actual.png`.

What this shows:
- No replica tracks Berkshire closely: best monthly correlation 0.65, tracking error about 14% a year on an 18%-vol asset. Most of what Berkshire does month to month is not a factor property.
- The in-sample factor replicas reproduce the sign of Berkshire's defensive behaviour but not the size. 2000-02: Berkshire +68%, R6 +32%, R3 +25%, S&P -38%. 2008: -41% vs R6 -36%, R3 -40%, S&P -51%. 2022: +4% vs R6 +2%, R3 +6%, S&P -18%. Part of the defensiveness (beta 0.65 plus value and large-cap tilt) is a factor property; roughly half of the 2000-02 gain is not.
- Out-of-sample (1980-99 loadings) R3_oos loses 8% in 2000-02 and 56% in 2008, and R6_oos behaves like a levered market in the GFC (-50%): loadings describing Berkshire in one era do not describe it in the next. The replicas are not a stable stand-in.
- The simple proxies do not reproduce 2000-02: VALUE -38% (like the market), LOWB3 -29% (telecoms), MONEY +15% but a -73% drawdown in 2008-09 (Berkshire -41%). In 2025-02..04 Berkshire gained 13.9% while every replica was between -10% and +2.3%: that period is Berkshire-specific. For calendar 2025 Berkshire (+10.9%) lagged the S&P (+17.7%) and every replica except LOWB3.
- Conclusion: a factor sleeve gives a rough, in-sample-only version of the defensiveness; the size of Berkshire's 2000-02 and 2025 behaviour is idiosyncratic to Berkshire.

## C. Century rolling DCA (daily FF data, monthly contributions, slippage 0.0005)

Market = `FF_MKT` (frictionless; beat VFINX by ~0.8 pp/yr on overlap). Mix = (1-w) FF_MKT + w sleeve, fixed split. Overlapping monthly-start windows: the effective number of independent windows is about 100 years / horizon (10, 5, 3 for 10/20/30-year horizons). The zero-alpha replicas are low-beta (about 0.6) sleeves, so their expected return is below the market's by construction; a negative R3 median difference is the beta effect.

Sample 1926-07+, 70% market / 30% sleeve, IRR % (market 10y/20y/30y: median 11.54 / 11.12 / 11.11, p10 4.41 / 6.62 / 9.11, min -6.95 / 2.81 / 7.63):

| H | sleeve | windows | mix median / p10 / min | % windows beating market | median diff (pp) | diff range (pp) |
|---|---|---|---|---|---|---|
| 10y | R3 | 1083 | 10.93 / 4.91 / -5.92 | 34 | -0.36 | -2.70..+4.54 |
| 20y | R3 | 963 | 11.01 / 6.98 / 4.09 | 43 | -0.21 | -1.54..+1.83 |
| 30y | R3 | 843 | 11.15 / 9.06 / 7.86 | 52 | +0.06 | -0.90..+0.77 |
| 10y | VALUE | 1083 | 12.38 / 5.13 / -7.19 | 73 | +0.67 | -1.64..+2.38 |
| 20y | VALUE | 963 | 11.79 / 7.83 / 3.73 | 89 | +0.85 | -0.81..+2.18 |
| 30y | VALUE | 843 | 12.29 / 9.68 / 8.24 | 91 | +0.95 | -0.33..+1.90 |
| 10y | MONEY | 1083 | 11.59 / 3.92 / -9.52 | 57 | +0.18 | -2.57..+2.73 |
| 20y | MONEY | 963 | 11.91 / 6.42 / 2.65 | 63 | +0.17 | -0.97..+1.69 |
| 10y | LOWB3 | 1083 | 11.63 / 4.68 / -6.02 | 60 | +0.20 | -2.39..+2.99 |
| 20y | LOWB3 | 963 | 11.07 / 7.05 / 3.76 | 68 | +0.21 | -1.13..+1.61 |
| 10y | TECH | 1083 | 12.13 / 4.28 / -7.34 | 62 | +0.35 | -2.79..+6.80 |
| 20y | TECH | 963 | 11.42 / 6.90 / 3.26 | 59 | +0.22 | -1.42..+2.81 |

Sample 1963-07+ (needed for R6; market 10y/20y/30y medians 11.92 / 11.11 / 10.73), 70/30:

| H | sleeve | windows | % beating market | median diff (pp) | mix p10 vs market p10 | mix min vs market min |
|---|---|---|---|---|---|---|
| 10y | R6 | 639 | 57 | +0.19 | 4.95 vs 3.97 | -5.33 vs -6.95 |
| 20y | R6 | 519 | 67 | +0.34 | 7.25 vs 7.29 | 4.51 vs 3.38 |
| 30y | R6 | 399 | 64 | +0.36 | 9.18 vs 9.24 | 8.09 vs 7.94 |
| 10y | R3 | 639 | 49 | -0.02 | 4.99 vs 3.97 | -5.92 vs -6.95 |
| 20y | R3 | 519 | 57 | +0.07 | 6.95 vs 7.29 | 4.09 vs 3.38 |

90/10, 80/20 and 50/50 mixes, all horizons and sleeves, are in `C_century_rolling.csv` and `tables.md` (the 50/50 pattern is the 70/30 pattern at about 1.7x the size). By start decade (`C_century_by_decade.csv`; 70/30, 20y windows, median diff, pp): R3 ranges from -1.08 (starts in the 2000s, 81 windows) to +0.80 (1950s); VALUE from +1.67 (1960s) to -0.59 (2000s); LOWB3 from -0.70 (1930s) to +1.13 (1970s); TECH from -0.86 (1970s) to +1.62 (1940s). Signs flip by decade for every sleeve except VALUE before 2000.

Non-overlapping windows (blocks of H months starting 1926-07, then every H months; "offsets" = the same grid shifted by 0, 12, ... H-12 months), 70/30:

| H | sleeve | blocks at offset 0 | % blocks beating market at offset 0 | mean % beating over offsets (min-max) | mean diff over offsets (pp) |
|---|---|---|---|---|---|
| 10y | R3 | 10 | 20 | 35 (20-44) | -0.27 |
| 20y | R3 | 5 | 40 | 43 (25-50) | -0.14 |
| 30y | R3 | 3 | 33 | 52 (0-100) | -0.03 |
| 10y | VALUE | 10 | 70 | 74 (56-89) | +0.7 |
| 20y | VALUE | 5 | 80 | 88 (75-100) | +0.8 |
| 30y | VALUE | 3 | 100 | 92 (67-100) | +0.9 |
| 10y | MONEY | 10 | 40 | 54 (22-78) | +0.1 |
| 10y | LOWB3 | 10 | 70 | 60 (44-78) | +0.2 |
| 10y | TECH | 10 | 60 | 62 (56-78) | +0.5 |

1963+ non-overlapping, R6: 6 / 3 / 2 blocks at offset 0 for 10 / 20 / 30 years; % beating 50 / 33 / 50 at offset 0 and 59 / 67 / 63 averaged over offsets (min-max over offsets 40-80, 0-100, 0-100).

What this shows:
- A zero-alpha Berkshire-like sleeve (R3, R6) changes the median IRR by a few tenths of a pp either way (-0.4 to +0.4); it does not raise the median. What it changes is the lower tail: 10-year p10 IRR rises from 4.41% to 4.91% (R3, 1926+) and from 3.97% to 4.95% (R6, 1963+); the worst 10-year window improves from -6.95% to about -5.3 to -5.9%; the worst 20-year window from 2.81% to 4.09% (R3). These tail differences come from a few episodes (1929-32 and 2000-09 starts), so rest on about 2-3 independent episodes.
- VALUE beats the market in 73-91% of overlapping windows by +0.7 to +0.95 pp (+1.1 to +1.5 at 50/50), but that is the century-long value premium and it turned negative for starts in the 2000s and 2010s. It is a different effect from "Berkshire".
- With 10, 5 and 3 independent windows, differences of 0.2-0.4 pp in medians and win rates are within noise: shifting the block grid by one year moves the win rate by 20-50 points at 20- and 30-year horizons (R3 30y: 0% to 100% across offsets).

## D. Block bootstrap of the real thing (VFINX vs BRK-A)

Method: joint monthly returns Apr-1980..Oct-2026 (n = 559), stationary bootstrap (mean block 18 months; 6 and 36 for sensitivity), 10,000 paths, common random numbers (scenarios on the same pool share the same resampled index paths), contributions of 1 at the start of each month, horizons 10/20/30 years, Berkshire share 0/10/20/30/40/50% of each contribution (fixed split; monthly-rebalanced variant at L = 18). IRR is exact for monthly flows. The blow-up multiplies Berkshire's gross return in one uniformly random month by 0.5 (probability p over the horizon).

Pool facts (monthly returns):

| pool | n | S&P mean log % | BRK mean log % | S&P vol % | BRK vol % | corr |
|---|---|---|---|---|---|---|
| 1980-2026 | 559 | 10.93 | 17.28 | 15.2 | 21.6 | 0.49 |
| 2000-2026 | 322 | 7.82 | 9.98 | 15.1 | 18.0 | 0.49 |
| 2013-2026 | 166 | 13.70 | 13.05 | 14.2 | 16.9 | 0.63 |

Scenario 4 ("no edge"), 20y, L = 18, fixed split (% unless noted; d = paired difference vs 0% Berkshire, pp):

| BRK share | IRR mean | median | p5 | p10 | p90 | d mean [5%, 95%] | P(mix > pure) | max DD median | max DD p10 worst | worst-12m median | TW p5 / p10 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11.73 | 11.99 | 4.87 | 6.53 | 16.53 | - | - | -36.1 | -51.7 | -26.7 | 1.00 / 1.00 |
| 10 | 11.88 | 12.12 | 5.08 | 6.75 | 16.68 | +0.16 | 50.0 | -34.2 | -50.6 | -24.4 | 1.02 / 1.02 |
| 20 | 12.00 | 12.20 | 5.20 | 6.79 | 16.88 | +0.27 | 50.0 | -32.5 | -50.1 | -23.3 | 1.04 / 1.03 |
| 30 | 12.08 | 12.21 | 5.25 | 6.82 | 17.13 | +0.35 [-1.59, +3.52] | 50.0 | -31.5 | -49.6 | -22.7 | 1.04 / 1.03 |
| 40 | 12.13 | 12.21 | 5.20 | 6.68 | 17.37 | +0.40 | 50.0 | -30.9 | -49.2 | -23.4 | 1.04 / 1.02 |
| 50 | 12.15 | 12.18 | 5.03 | 6.53 | 17.69 | +0.42 [-2.86, +5.22] | 50.0 | -31.3 | -48.9 | -25.1 | 1.02 / 1.00 |

Scenario comparison at 30% Berkshire, 20y, L = 18 (full 0-50% grids per scenario in `tables.md` / `D_bootstrap_results.csv`):

| scenario | IRR median (30%) | p10 (30%) | d mean | d median | d p5 / p95 | P(mix>pure) | max DD median (30%) | d max DD mean [5%, 95%] | TW p5 / p10 ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 raw | 15.12 | 9.38 | +3.43 | +2.96 | -0.10 / +8.60 | 94.1 | -29.7 | +4.8 [-2.0, +16.8] | 1.40 / 1.39 |
| 2 2000+ | 9.47 | 4.80 | +0.92 | +0.64 | -0.79 / +3.57 | 72.6 | -37.9 | +7.1 [-0.1, +19.1] | 1.28 / 1.24 |
| 3 2013+ | 14.64 | 11.65 | -0.15 | -0.22 | -1.11 / +1.03 | 36.6 | -21.2 | +1.8 [-0.9, +4.3] | 1.01 / 1.00 |
| 4 no edge | 12.21 | 6.82 | +0.35 | 0.00 | -1.59 / +3.52 | 50.0 | -31.5 | +2.8 [-2.3, +11.2] | 1.04 / 1.03 |
| 4b no edge 2000+ | 8.68 | 3.85 | +0.12 | -0.12 | -1.26 / +2.33 | 45.6 | -39.0 | +5.9 [-0.2, +16.5] | 1.15 / 1.12 |
| 5 -1 pp edge | 11.90 | 6.50 | +0.01 | -0.31 | -1.73 / +2.90 | 40.9 | -31.9 | +2.6 [-2.4, +10.4] | 1.01 / 1.00 |
| 6 no edge + 5% blow-up | 12.16 | 6.76 | +0.30 | -0.04 | -1.64 / +3.45 | 48.7 | -31.7 | +2.7 [-2.5, +11.1] | 1.04 / 1.03 |
| 6 no edge + 10% | 12.12 | 6.69 | +0.25 | -0.09 | -1.66 / +3.38 | 47.4 | -31.7 | +2.7 [-2.8, +11.0] | 1.03 / 1.02 |
| 6 no edge + 20% | 12.02 | 6.56 | +0.15 | -0.21 | -1.75 / +3.28 | 44.5 | -32.1 | +2.4 [-3.6, +10.9] | 1.02 / 1.00 |

Figures: `fig4_bootstrap_difference_distributions.png` (paired IRR and max-DD differences by scenario), `fig5_weight_curves.png` (p10 IRR, median IRR, median max DD against Berkshire weight, all scenarios).

Monte-Carlo and block-length check of the headline (scenario 4, 30%; `D_mc_repeats_noedge_w30_H20.csv`):

| source | change in p10 IRR (pp) | change in median IRR (quantile diff, pp) | paired mean change in max DD (pp) |
|---|---|---|---|
| 6 independent seeds x 10,000 paths, 20y, L = 18 | +0.18 to +0.29 (sd 0.05) | +0.20 to +0.29 | +2.80 to +2.89 |
| block length 6 / 18 / 36 (20y) | +0.32 / +0.29 / -0.04 | +0.26 / +0.21 / +0.26 | +1.84 / +2.84 / +3.01 |
| horizon 10 / 20 / 30y (L = 18) | +0.23 / +0.29 / +0.14 | +0.17 / +0.21 / +0.27 | +2.18 / +2.84 / +3.08 |

For the raw and 2000+ scenarios the same sensitivity (`tables.md`, D4) gives p10 IRR changes of +2.1 to +3.3 pp (raw) and +1.6 to +2.4 pp (2000+) across block lengths and horizons.

What this shows:
- Under "no edge" the p10 IRR change (+0.29) exceeds Monte-Carlo noise (sd 0.05) but sits inside the block-length range (-0.04 to +0.32), so it is not a reliable gain; the median IRR change is zero; P(mix beats pure) is 50%.
- The shallower drawdown is robust to block length, horizon and seed (paired mean +1.8 to +3.1 pp; median-of-paths max DD about 4.5 pp shallower). The 5-95% spread is wide (-2.3 to +11.2 pp): in about one path in ten the 30% mix has a deeper drawdown than pure S&P.
- The fixed-split mean IRR is +0.35 pp higher even with no edge. This is a design artefact: matching mean log returns leaves Berkshire's arithmetic mean about 1.2 pp higher (vol 21.6% vs 15.2%, half the variance difference); it shows up in the mean but not the median. The 2000+ no-edge shift (vol 18.0%) gives +0.12 mean and -0.12 median.
- The raw-history result (+3.4 mean, +2.96 median, 94% win rate) is almost entirely the 1980-99 outperformance. Restricting to 2000-26 cuts it to +0.9 / +0.6; to 2013-26 it is negative (-0.15, 37%): the S&P's mean log return was 13.7% against Berkshire's 13.05%.
- A one-off -50% blow-up with 5/10/20% probability lowers the 30% mix's mean IRR gain by 0.05 / 0.10 / 0.20 pp and the median difference by 0.04 / 0.09 / 0.21 pp; at 20% probability P(mix > pure) is 44.5%. It lowers the median edge but does not change the picture.
- The 2000+ no-edge case (4b) shows a larger drawdown benefit (+5.9 pp) than the full-history one (+2.8 pp): Berkshire differed most from the S&P in 2000-02 and 2008, so the benefit depends on how much weight the resampled history gives those episodes.
- Limits: 559 monthly returns (47 years) are all the data; the bootstrap preserves volatility, correlation and within-block drawdown shape but cannot create a regime that never occurred. "No edge" is an assumed return, not an estimate, and the 5-95% intervals are outcome spreads under that assumption.

## E. Singapore tax tilt (30% US dividend withholding on the S&P leg)

Measured drag (SPY vs SPY_NET30 / SPY_NET15, CAGR): 0.56 / 0.28 pp/yr over 1993-2026 (0.57 / 0.29 in 1993-99, 0.55 / 0.28 in 2000-12, 0.57 / 0.28 in 2013-26). At a 1.2% yield the drag would be 0.36 / 0.18 pp/yr (synthetic `SPY_D36`, `SPY_D18`).

Historical rolling DCA from 1999-03, SPY leg gross vs net vs BRK-B (no drag); IRR %; 213 ten-year and 93 twenty-year overlapping windows (about 2.7 and 1.2 independent windows):

| H | S&P leg | median 100% S&P | median 70/30 | median 100% BRK-B | 70/30 minus S&P, median pp | % windows 70/30 beats S&P | BRK-B minus S&P, median pp |
|---|---|---|---|---|---|---|---|
| 10y | gross SPY | 11.31 | 11.34 | 11.24 | -0.03 | 46 | -0.12 |
| 10y | NET30 | 10.69 | 10.91 | 11.24 | +0.16 | 65 | +0.52 |
| 10y | NET15 | 11.00 | 11.12 | 11.24 | +0.06 | 56 | +0.19 |
| 10y | drag 0.36 | 10.90 | 11.06 | 11.24 | +0.08 | 59 | +0.26 |
| 10y | drag 0.18 | 11.11 | 11.20 | 11.24 | +0.02 | 53 | +0.08 |
| 20y | gross SPY | 11.32 | 11.24 | 10.98 | -0.03 | 47 | -0.10 |
| 20y | NET30 | 10.72 | 10.82 | 10.98 | +0.15 | 67 | +0.49 |
| 20y | NET15 | 11.02 | 11.03 | 10.98 | +0.06 | 57 | +0.19 |
| 20y | drag 0.36 | 10.93 | 10.97 | 10.98 | +0.09 | 59 | +0.29 |
| 20y | drag 0.18 | 11.12 | 11.10 | 10.98 | +0.03 | 55 | +0.10 |

Bootstrap, scenario 4 with the S&P leg net of withholding (joint SPY / SPY_NET30 / SPY_NET15 / BRK-B pool 1996-06..2026-10, n = 365, 20y, L = 18, 10,000 paths; "no edge" shifts BRK-B to the mean log return of the GROSS S&P, so the tax drag is not neutralised). Paired mean IRR difference, mix minus pure, pp/yr (P(mix > pure)):

| S&P leg | 30% BRK vs pure S&P | 100% BRK vs 100% S&P |
|---|---|---|
| gross | +0.16 (48.6%) | -0.04 (48.6%) |
| NET30 | +0.34 (55.1%) | +0.52 (55.1%) |
| NET15 | +0.25 (51.7%) | +0.24 (51.7%) |
| drag 0.36 (1.2% yield, 30%) | +0.29 (53.0%) | +0.36 (53.0%) |
| drag 0.18 (1.2% yield, 15%) | +0.23 (50.8%) | +0.16 (50.8%) |

The 5-95% spread of the paired 30%-vs-pure difference is about -1.3 to +2.6 pp in every row. In the raw 1996+ history (BRK-B's real edge, same pool) the 30% mix's paired mean advantage is +0.56 (gross), +0.75 (NET30), +0.65 (NET15), with P(beats) 62% / 68% / 65%.

What this shows:
- The withholding drag adds about 0.56 pp/yr (0.28 for the Irish-domiciled case; 0.36 / 0.18 at a 1.2% yield) to the Berkshire side of a 100/0 comparison, and 0.3 times that, about 0.17 pp/yr (0.08 at 15%), to a 70/30 comparison against the S&P-only core (measured +0.16 to +0.18 in the bootstrap means).
- Under "no edge" it moves the mean difference from +0.16 to +0.34 and P(mix > pure) from 48.6% to 55.1%; the paired median from -0.04 to +0.13. These shifts are small relative to the ~4 pp 5-95% spread, so the tax drag does not change any conclusion: the 30% mix is not distinguishable from pure S&P on return, and the drawdown benefit is unchanged. With NET15 or 1.2%-yield drags the shift is smaller still.
- Berkshire pays no dividend, so the drag applies only to the S&P leg; an Irish-domiciled S&P fund (15%) halves it. The historical 1999+ comparison has 1-3 independent windows and is evidence only of the size of the drag.

## Caveats
- Overlapping windows are highly autocorrelated; effective independent observations are about sample_years / horizon_years (FF 1926-2026: 10 / 5 / 3 windows of 10 / 20 / 30 years; BRK 1980-2026: about 4.7 / 2.3 / 1.6).
- Berkshire and the 1980-2026 window are known in hindsight; the 1980-99 alpha estimate is the one that does not survive out of sample.
- FF series are frictionless index returns; replicas use 2000-26 loadings and are in-sample for that period. Replica and FF series have open == close (the decision-day close is the buy price in `rolling_dca`).
- Factor regressions use month-end to month-end BRK-A returns; NW lags = 6; CIs are asymptotic.
- The bootstrap works on monthly returns with contributions at the start of each month; max drawdown and worst-12-month are monthly measures and understate daily drawdowns.
- Spot checks: `rolling_dca` on FF_MKT (10y from 1950-01) matches `DCABacktest` + `dca_metrics` (IRR 17.2035% both); the bootstrap IRR solver recovers a constant 0.7%/month return exactly (8.7311%). No bug found in `src/dca.py`.

## Files
- Script: `scripts/dca/exp6_berkshire_necessity.py`
- Tables: `tables.md` (all generated tables), `A_regressions.csv`, `A_rolling_10y.csv`, `B_tracking.csv`, `C_century_rolling.csv`, `C_century_by_decade.csv`, `C_century_nonoverlapping.csv`, `D_pool_stats.csv`, `D_bootstrap_results.csv`, `D_mc_repeats_noedge_w30_H20.csv`, `E_drag_measured.csv`, `E_hist_rolling_1999.csv`, `E_bootstrap_tax.csv`
- Figures: `fig1_rolling_alpha_beta.png`, `fig2_replica_vs_actual.png`, `fig3_century_irr_differences.png`, `fig4_bootstrap_difference_distributions.png`, `fig5_weight_curves.png`
