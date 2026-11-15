# EXP3 - Should the monthly contribution chase tech or buy the laggard (Berkshire)?

Script: `scripts/dca/exp3_tilt_timing.py` (about 75 s). Tables are the CSVs in this folder. All returns are total return except `^NDX`, which is a PRICE index (no dividends; flagged wherever used). BRK-A / BRK-B are treated as the same asset. Decisions are the first trading day of each month, orders at that day's open, slippage 0.0005, no commission, $1 per month. Signals use closes strictly before the decision day (`searchsorted(..., side='left') - 1`).

**Samples.** Modern pairs (QQQ/BRK-B, SPY/BRK-B, QQQ/SPY): data 1999-03 to 2026-11-13, 333 decision months. Signals for these pairs are warmed up with spliced proxies from before 1999 (^NDX or VFINX for A, BRK-A for B, rescaled at the splice date), so every rule has a signal from the first month. Long pairs: ^NDX/BRK-A decisions 1988-11 to 2026-11 (457 months; first 36 months dropped for signal warm-up) and VFINX/BRK-A 1983-05 to 2026-11 (523 months). Window counts: modern 273 / 213 / 93 for 5y / 10y / 20y; ^NDX/BRK-A 397 / 337 / 217; VFINX/BRK-A 463 / 403 / 283.

**Overlap caveat.** Overlapping windows are highly autocorrelated. The effective number of independent observations is roughly sample years / horizon years: about 5 / 2.5 / 1.4 for the modern pairs (5y/10y/20y), and about 9 / 4.5 / 2 for VFINX/BRK-A. Percent-of-windows-beaten figures should be read with that in mind. The placebo tests below are designed to account for the overlap; the raw window counts are not.

**Hindsight caveat.** Berkshire, QQQ and NVDA-era tech are known winners picked now. Berkshire's 1980-2000 outperformance (about 2.5%/month mean in 1980-99) is not representative of a forward expectation.

---
## Part A - descriptive facts

### A1. Drawdown episodes
Episodes use a running-max definition: peak = last all-time high, trough = lowest close before the next new high; an episode is kept if depth is at least 10% (S&P, VFINX, 1980-) or 15% (NDX price, 1985-). Returns in percent. "p->t" is peak to trough; "+12m / +36m" is the return over the 12 / 36 months after the trough. Blank = asset has no data. BRK = BRK-A. QQQ columns are in `a1_*.csv`.

S&P 500 drawdowns >= 10% (20 episodes):

| peak | trough | depth % | S&P p->t | NDX p->t | BRK p->t | S&P +12m | NDX +12m | BRK +12m | S&P +36m | NDX +36m | BRK +36m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1980-02-13 | 1980-03-26 | -16 | -16 |  |  | 40 |  |  | 65 |  |  |
| 1980-11-28 | 1982-08-12 | -25 | -25 |  | 5 | 61 |  | 114 | 92 |  | 369 |
| 1983-10-10 | 1984-07-24 | -14 | -14 |  | -0 | 32 |  | 71 | 96 |  | 203 |
| 1985-12-16 | 1986-01-22 | -10 | -10 | -2 | -11 | 28 | 30 | 35 | 43 | 39 | 101 |
| 1986-12-02 | 1986-12-31 | -12 | -12 | -5 | 2 | 5 | 10 | 5 | 60 | 58 | 208 |
| 1987-08-25 | 1987-10-19 | -33 | -33 | -26 | -19 | 27 | 14 | 48 | 53 | 12 | 92 |
| 1990-01-02 | 1990-01-30 | -10 | -10 | -13 | -16 | 9 | 16 | 1 | 49 | 87 | 67 |
| 1990-07-16 | 1990-10-11 | -19 | -19 | -33 | -24 | 33 | 70 | 53 | 70 | 130 | 214 |
| 1997-10-07 | 1997-10-27 | -11 | -11 | -14 | -6 | 23 | 38 | 44 | 64 | 225 | 41 |
| 1998-07-17 | 1998-08-31 | -19 | -19 | -22 | -20 | 40 | 110 | 6 | 23 | 29 | 15 |
| 1999-07-16 | 1999-10-15 | -12 | -12 | -2 | -19 | 12 | 36 | 2 | -27 | -60 | 29 |
| 2000-03-24 | 2000-04-14 | -11 | -11 | -32 | 6 | -12 | -47 | 11 | -32 | -67 | 22 |
| 2000-09-01 | 2002-10-09 | -48 | -48 | -80 | 14 | 36 | 73 | 11 | 62 | 93 | 22 |
| 2007-10-09 | 2009-03-09 | -55 | -55 | -52 | -40 | 72 | 82 | 69 | 115 | 154 | 63 |
| 2015-07-20 | 2016-02-11 | -13 | -13 | -15 | -12 | 29 | 32 | 29 | 57 | 74 | 60 |
| 2018-01-26 | 2018-02-08 | -10 | -10 | -10 | -12 | 7 | 10 | 4 | 60 | 117 | 24 |
| 2018-09-20 | 2018-12-24 | -19 | -19 | -22 | -15 | 40 | 47 | 20 | 111 | 176 | 57 |
| 2020-02-19 | 2020-03-23 | -34 | -34 | -28 | -30 | 78 | 86 | 57 | 84 | 82 | 89 |
| 2022-01-03 | 2022-10-12 | -25 | -25 | -35 | -12 | 23 | 41 | 31 | 91 | 125 | 84 |
| 2025-02-19 | 2025-04-08 | -19 | -19 | -23 | 2 | 38 | 46 | -3 |  |  |  |

Nasdaq-100 drawdowns >= 15% (15 episodes):

| peak | trough | depth % | S&P p->t | NDX p->t | BRK p->t | S&P +12m | NDX +12m | BRK +12m | S&P +36m | NDX +36m | BRK +36m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1986-07-03 | 1986-09-15 | -18 | -7 | -18 | -11 | 30 | 50 | 43 | 51 | 64 | 218 |
| 1987-10-05 | 1987-10-26 | -40 | -31 | -40 | -31 | 28 | 37 | 65 | 48 | 36 | 102 |
| 1989-10-09 | 1990-01-30 | -17 | -9 | -17 | -18 | 9 | 16 | 1 | 49 | 87 | 67 |
| 1990-07-16 | 1990-10-11 | -33 | -19 | -33 | -24 | 33 | 70 | 53 | 70 | 130 | 214 |
| 1992-02-12 | 1992-06-26 | -19 | -2 | -19 | 3 | 14 | 25 | 68 | 46 | 85 | 165 |
| 1994-03-17 | 1994-06-24 | -16 | -5 | -16 | -0 | 27 | 55 | 49 | 117 | 180 | 195 |
| 1997-01-22 | 1997-04-02 | -15 | -4 | -15 | 3 | 52 | 57 | 86 | 108 | 461 | 58 |
| 1997-10-09 | 1997-12-24 | -18 | -4 | -18 | 0 | 33 | 92 | 48 | 46 | 159 | 45 |
| 1998-07-20 | 1998-10-08 | -23 | -19 | -23 | -23 | 41 | 126 | -1 | 15 | 13 | 24 |
| 2000-03-27 | 2002-10-07 | -83 | -47 | -83 | 40 | 35 | 73 | 5 | 60 | 93 | 15 |
| 2015-11-03 | 2016-02-09 | -16 | -12 | -16 | -7 | 27 | 32 | 28 | 55 | 75 | 56 |
| 2018-08-29 | 2018-12-24 | -23 | -19 | -23 | -11 | 40 | 47 | 20 | 111 | 176 | 57 |
| 2020-02-19 | 2020-03-20 | -28 | -32 | -28 | -25 | 73 | 84 | 49 | 79 | 80 | 77 |
| 2021-11-19 | 2022-12-28 | -36 | -18 | -36 | 9 | 28 | 58 | 19 | 90 | 140 | 62 |
| 2025-02-19 | 2025-04-08 | -23 | -19 | -23 | 2 | 38 | 46 | -3 |  |  |  |

Summary:

| set | n_episodes | n_with_BRK | BRK_fell_less_than_SP | n_with_NDX | BRK_fell_less_than_NDX | BRK_rose_in_episode | median_SP | median_NDX | median_BRK |
|---|---|---|---|---|---|---|---|---|---|
| S&P drawdowns >=10% (VFINX) | 20 | 19 | 13 | 17 | 12 | 5 | -0.174 | -0.221 | -0.121 |
| NDX drawdowns >=15% (^NDX) | 15 | 15 | 10 | 15 | 13 | 6 | -0.182 | -0.229 | -0.068 |

In 19 S&P episodes where Berkshire has data, Berkshire lost less than the S&P in 13 (68%), lost less than the NDX in 12 of 17, and actually rose in 5. In the 15 NDX episodes it lost less than the NDX in 13 and less than the S&P in 10; median peak-to-trough: NDX -23%, S&P -18%, BRK -7%. So the hypothesis "Berkshire holds up better in drawdowns" is true on average, but not reliably: it fell more than the S&P in 6 of 19 episodes (1985-86, Jan 1990, Jul-Oct 1990 (-24% vs -19%), 1998, 1999 (-19% vs -12%), early 2018) and in 2007-09 it lost -40% vs -55%, in 2020 -30% vs -34%. The standout cases are 2000-02 (BRK +14% while S&P -48% and NDX -80%), 2022 and 2025. Berkshire also did not rebound as hard: after the 2009 and 2020 troughs the 12-month returns of the NDX/S&P exceeded Berkshire's (2009: S&P +72% vs BRK +69%; 2020: S&P +78% vs BRK +57%), so being defensive costs some of the recovery. (In the 1980s and early 1990s Berkshire had the largest rebounds, e.g. +114% 12m after the 1982 trough.) Peak-to-trough returns of the NDX in 2000-02 are -83% on the price index; the QQQ ETF shows -80% from its own start.

### A2. Capture and beta of Berkshire (monthly returns, month-end closes)
Up capture = mean BRK return in months when the benchmark was up divided by the benchmark's mean in those months (arithmetic); down capture likewise for down months.

| benchmark | era | n_months | beta | corr | up_capture | down_capture | mean_a_pct | mean_b_pct |
|---|---|---|---|---|---|---|---|---|
| S&P 500 (VFINX) | full | 560 | 0.697 | 0.488 | 0.998 | 0.468 | 1.009 | 1.653 |
| S&P 500 (VFINX) | 1980-99 | 237 | 0.835 | 0.497 | 1.303 | 0.57 | 1.368 | 2.553 |
| S&P 500 (VFINX) | 2000-12 | 156 | 0.441 | 0.368 | 0.618 | 0.282 | 0.236 | 0.707 |
| S&P 500 (VFINX) | 2013-26 | 167 | 0.745 | 0.621 | 0.826 | 0.592 | 1.223 | 1.258 |
| Nasdaq-100 (^NDX price) | full | 493 | 0.25 | 0.28 | 0.488 | 0.123 | 1.35 | 1.366 |
| Nasdaq-100 (^NDX price) | 1980-99 | 170 | 0.381 | 0.364 | 0.617 | 0.215 | 2.301 | 2.076 |
| Nasdaq-100 (^NDX price) | 2000-12 | 156 | 0.061 | 0.087 | 0.271 | 0.038 | 0.104 | 0.707 |
| Nasdaq-100 (^NDX price) | 2013-26 | 167 | 0.387 | 0.402 | 0.521 | 0.183 | 1.546 | 1.258 |
| SPY (BRK-B, 1999-03+) | full | 332 | 0.586 | 0.484 | 0.715 | 0.453 | 0.774 | 0.902 |
| QQQ (BRK-B, 1999-03+) | full | 332 | 0.147 | 0.188 | 0.353 | 0.093 | 1.055 | 0.902 |

Against the S&P, Berkshire has had a down capture of about 0.47 and an up capture of about 1.0 over 1980-2026 (beta 0.70). The up/down asymmetry is strongest in 1980-99 (1.30 / 0.57) and 2000-12 (0.62 / 0.28 - the up capture is lower in that era), and weakest recently (2013-26: 0.83 / 0.59, beta 0.75). Against the Nasdaq-100 the correlation is low (0.28 full sample, 0.09 in 2000-12, 0.40 in 2013-26), so Berkshire diversifies tech mostly in the 2000-12 era. Since 2013 the S&P down capture of 0.59 and mean monthly returns (1.22% S&P vs 1.26% BRK) are much closer than the full sample suggests. Eras overlap with very few independent episodes (2000-02, 2008-09, 2020, 2022).

### A3. Is the relative return predictable?
Monthly month-end closes, r = log(A/B). Trailing k-month relative return = r_t - r_(t-k), forward h-month = r_(t+h) - r_t. Correlation over all months (overlapping). Significance: p-value from a circular-shift null (the forward series is rotated by every offset of at least max(k,h,12) months, which preserves the autocorrelation of both series and the overlap), two-sided. Each pair has 30 (k,h) cells, so about 1.5 cells would be "significant" at 5% by chance, and the cells are strongly dependent on each other. Full 6x5 matrices: `a3_corr_matrix.csv`; all cells with n, p, and non-overlapping correlations: `a3_corr_long.csv`.

| pair | cells | p05 | neg | mean_corr |
|---|---|---|---|---|
| QQQ/BRK-B | 30 | 3 | 15 | -0.037 |
| QQQ/SPY | 30 | 6 | 12 | 0.019 |
| SPY/BRK-B | 30 | 8 | 29 | -0.133 |
| VFINX/BRK-A | 30 | 0 | 25 | -0.017 |
| ^NDX/BRK-A | 30 | 6 | 15 | -0.072 |

Correlation, trailing 12m to forward 12m:

| pair | n_obs | corr | p_circ_shift |
|---|---|---|---|
| QQQ/BRK-B | 309 | -0.059 | 0.666 |
| SPY/BRK-B | 309 | -0.048 | 0.735 |
| QQQ/SPY | 309 | -0.091 | 0.429 |
| ^NDX/BRK-A | 470 | -0.03 | 0.658 |
| VFINX/BRK-A | 537 | 0.003 | 0.981 |

Correlation, trailing 36m to forward 36m:

| pair | n_obs | corr | p_circ_shift |
|---|---|---|---|
| QQQ/BRK-B | 261 | -0.283 | 0.34 |
| SPY/BRK-B | 261 | -0.625 | 0.031 |
| QQQ/SPY | 261 | 0.169 | 0.597 |
| ^NDX/BRK-A | 422 | -0.417 | 0.065 |
| VFINX/BRK-A | 489 | 0.09 | 0.675 |

Mean forward relative return (A minus B, log, percentage points) by quintile of trailing relative return (Q1 = A lagged most):

| pair | k | h | Q1_mean_pp | Q2_mean_pp | Q3_mean_pp | Q4_mean_pp | Q5_mean_pp | spread_Q5_minus_Q1_pp | p_spread_circ_shift |
|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | 12 | 12 | -2.0 | 6.1 | 2.1 | 2.6 | -13.0 | -11.0 | 0.4 |
| QQQ/BRK-B | 36 | 36 | 19.7 | 16.6 | 16.7 | 14.3 | -10.7 | -30.4 | 0.1 |
| SPY/BRK-B | 12 | 12 | -1.3 | -1.1 | -0.0 | -1.4 | -4.3 | -3.0 | 0.7 |
| SPY/BRK-B | 36 | 36 | 14.6 | 5.0 | 3.7 | -6.0 | -17.8 | -32.3 | 0.0 |
| QQQ/SPY | 12 | 12 | 1.3 | 3.7 | 3.5 | 2.4 | -6.8 | -8.1 | 0.2 |
| QQQ/SPY | 36 | 36 | 7.1 | 15.2 | 16.4 | 13.9 | 4.4 | -2.6 | 0.9 |
| ^NDX/BRK-A | 12 | 12 | -3.9 | 4.1 | -1.0 | 1.1 | -5.0 | -1.1 | 0.9 |
| ^NDX/BRK-A | 36 | 36 | 6.1 | 13.4 | 12.1 | 11.5 | -42.3 | -48.4 | 0.1 |
| VFINX/BRK-A | 12 | 12 | -7.4 | -5.8 | -3.3 | -8.0 | -3.3 | 4.1 | 0.5 |
| VFINX/BRK-A | 36 | 36 | -24.2 | -12.1 | -3.5 | -7.0 | -23.3 | 0.9 | 1.0 |

Short horizons (k and h up to 12 months): nothing distinguishable from zero in any pair. The correlations are between -0.1 and +0.1 for the S&P/Berkshire and NDX/Berkshire pairs; the QQQ/SPY pair shows small positive short-horizon momentum (0.11 to 0.19 at k,h <= 6, 6 of 30 cells at p < 0.05), consistent with chance given 150 cells in total and dependence between cells. The long-horizon cells (trailing 12-36m, forward 36m) are negative in the 1999+ samples (SPY/BRK-B: 29 of 30 cells negative; 36m-to-36m correlation -0.63, p = 0.03; QQQ/BRK-B -0.28, p = 0.34; ^NDX/BRK-A -0.42, p = 0.065). That is mean reversion of A relative to Berkshire over multi-year spans. It is driven by the 2000-02 round trip: with only about 8 non-overlapping 36-month observations in the 1999+ sample the non-overlapping correlations (-0.5 to -0.9) mean little. The 43-year VFINX/BRK-A sample shows no relation at any horizon (0 of 30 cells significant, 36m/36m correlation +0.09, quintile spread +0.9 pp, p = 1.0). Net: weak, sample-dependent hints of multi-year mean reversion between tech and Berkshire; no usable monthly predictability. The figure shows the quintile bars with the shifted-null band (`fig2_quintiles.png`).

Current position (last close 2026-11-13; as of the data): QQQ/BRK-B log relative returns over 1/3/6/12 months are -18.9 / -23.4 / -25.8 / -3.8 pp (QQQ weak lately), QQQ is 10.5% below its all-time high and just under its 200-day SMA (-0.2%), and the 36-month ratio z-score is -0.08 (neutral). The ratio is 1.31x its 1999-03 level (peak 4.0x in 2000, trough 0.4x in 2008). SPY/BRK-B z = -0.18, S&P 3.2% below its high.

| pair | as_of_close | rel1m_pp | rel3m_pp | rel6m_pp | rel12m_pp | A_vs_sma200_pct | A_drawdown_from_ath_pct | ratio_z36m |
|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | 2026-11-13 | -18.88 | -23.4 | -25.78 | -3.77 | -0.19 | -10.5 | -0.08 |
| SPY/BRK-B | 2026-11-13 | -13.91 | -18.87 | -18.96 | -2.63 | 3.44 | -3.17 | -0.18 |
| QQQ/SPY | 2026-11-13 | -4.97 | -4.53 | -6.82 | -1.14 | -0.19 | -10.5 | 0.21 |
| ^NDX/BRK-A | 2026-11-13 | -18.88 | -23.61 | -25.87 | -4.16 | -0.34 | -10.51 | -0.12 |
| VFINX/BRK-A | 2026-11-13 | -13.95 | -19.04 | -18.89 | -2.67 | 3.38 | -3.22 | -0.17 |

(`fig5_relative_strength.png`.)

---
## Part B - contribution-direction rules

Setup: each month the whole $1 goes to A with fraction f and to B with 1-f (alloc table, `rolling_dca`). "Hard" = 100/0 on the signal side, "soft" = 75/25. Rules (48 variants per pair, plus 5 fixed baselines): momentum k=1,3,6,12; contrarian k=1,3,6,12; trend (A above its 200-day SMA -> A, else B) and reverse; drawdown d=10/20/30% below A's all-time high: dip (in drawdown -> A, else B), flee (in drawdown -> B, else A), and dip/flee with "otherwise 50/50"; ratio z-score of log(A/B) vs trailing 36 months: mean-revert (z>1 -> B, z<-1 -> A, else 50/50) and the reverse. 8 + 8 + 4 + 24 + 4 = 48 variants including hard and soft (48 = 24 signal definitions x 2). Every rule was evaluated per pair per horizon (5/10/20y), 15 pair-horizon combinations, so 720 rule-evaluations in total.

**Controls.** (1) fixed 50/50. (2) The fair control: for each window, the fixed mix whose A-fraction equals that window's average fraction of dollars the rule sent to A ("matched mix"). Because wealth is exactly linear in the allocation (final value = f x all-A wealth + (1-f) x all-B wealth), this is computed exactly. The ratio rule/matched removes the effect of simply holding more of the higher-return asset. (3) Hindsight best of all-A / all-B. The script asserts equality between its linear evaluation and `rolling_dca` for every rule.

**Placebo.** For each pair and horizon, the entire signal book (all 48 allocation sequences) is circularly shifted in time (and, in half of the draws, time-reversed) by a common offset of at least 6 months. This gives 644 draws for the modern pairs, 892 for ^NDX/BRK-A, 1024 for VFINX/BRK-A. Each rule keeps its exact average A-fraction, run lengths and its correlation with the other rules, but loses its alignment with prices. The placebo statistic is the same one as for the real rule: median over windows of wealth / matched-mix wealth. For multiple testing, the placebo statistic is the maximum over the 48 rules in each draw, and the best real rule is placed in that distribution. Secondary placebo: independent block-shuffles of each rule's own sequence (block length = the rule's mean run length, at least 6, 1000 draws). The columns `pctile_shift` and `pctile_block` are in `b_placebo.csv`.

An earlier version of the script used 6-month block shuffles for everyone; that made the placebo band too narrow (nearly every rule landed outside it, because persistent rules like drawdown and trend cannot be reproduced by 6-month blocks), so it was replaced by the shift placebo, which preserves persistence exactly.

### B1. Headline: which direction rule beats which
Median wealth ratio vs matched mix, averaged over rule families (48 variants; `b_rules_summary.csv`):

| horizon | contrarian (8) | momentum (8) | trend (4) | drawdown (24) | ratio z (4) |
|---|---|---|---|---|---|
| 5y | 1.011 | 0.989 | 1.000 | 1.000 | 1.000 |
| 10y | 1.012 | 0.988 | 1.000 | 1.000 | 1.001 |
| 20y | 1.017 | 0.984 | 1.002 | 0.996 | 1.002 |

(Means across the five pairs of per-rule medians, not a statistical statement.) Momentum rules lost to their matched mix almost everywhere: 118 of 120 pair-horizon-rule medians are below 1 (maximum 1.003); e.g. mom12 hard, 10y: 0.966 to 0.982 in all five pairs, beating the matched mix in 0-22% of windows. Their mirror images (contrarian) win by the same amount with the sign flipped, by construction, because avg A-fraction mirrors; so "contrarian > momentum" is one finding, not two. About half of all 48 variants beat their matched mix, as must happen with mirrored pairs. The size is small: ±1 to ±3% of final wealth at 5-10 years.

Hard rules, per pair, 5y / 10y / 20y. Columns: avg_fracA = mean % of money sent to A; irr in % per year; wm_med = median wealth multiple (terminal value / total contributed); vs50 = rule / fixed 50/50; vsmatch = rule / matched mix with the same avg A-fraction per window; beat = % of windows with ratio > 1; vsbest_med = median of rule wealth / max(all-A, all-B) wealth.

**10-year windows**

| pair | rule | avg_fracA | irr_med | irr_p10 | irr_min | wm_med | vs50_beat | vs50_med | vsmatch_beat | vsmatch_med | vsbest_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | mom12|hard | 62.8 | 14.1 | 5.4 | -5.5 | 2.084 | 48.8 | 0.998 | 10.3 | 0.97 | 0.894 |
| QQQ/BRK-B | contra12|hard | 37.2 | 13.7 | 6.6 | -2.2 | 2.042 | 51.2 | 1.002 | 89.7 | 1.034 | 0.89 |
| QQQ/BRK-B | trend_A_above_sma200|hard | 77.6 | 15.0 | 6.0 | -4.9 | 2.192 | 81.2 | 1.031 | 22.1 | 0.986 | 0.938 |
| QQQ/BRK-B | trend_rev|hard | 22.4 | 12.5 | 6.1 | -2.7 | 1.917 | 18.8 | 0.969 | 77.9 | 1.017 | 0.853 |
| QQQ/BRK-B | dd20_dip|hard | 57.7 | 14.3 | 6.8 | -4.2 | 2.109 | 64.3 | 1.057 | 63.4 | 1.029 | 0.959 |
| QQQ/BRK-B | dd20_flee|hard | 42.3 | 12.8 | 4.6 | -3.3 | 1.944 | 35.7 | 0.943 | 17.8 | 0.971 | 0.832 |
| QQQ/BRK-B | z_mean_revert|hard | 35.0 | 13.6 | 6.7 | -1.9 | 2.03 | 52.6 | 1.008 | 94.4 | 1.036 | 0.892 |
| QQQ/BRK-B | z_momentum|hard | 65.0 | 14.0 | 5.4 | -5.9 | 2.082 | 47.4 | 0.992 | 5.6 | 0.966 | 0.901 |
| SPY/BRK-B | mom12|hard | 46.8 | 10.9 | 4.1 | -4.6 | 1.762 | 8.0 | 0.984 | 0.0 | 0.982 | 0.956 |
| SPY/BRK-B | contra12|hard | 53.2 | 11.7 | 4.3 | -3.3 | 1.836 | 92.0 | 1.016 | 100.0 | 1.018 | 0.994 |
| SPY/BRK-B | trend_A_above_sma200|hard | 77.6 | 11.1 | 3.7 | -5.3 | 1.782 | 27.7 | 0.984 | 4.2 | 0.99 | 0.968 |
| SPY/BRK-B | trend_rev|hard | 22.4 | 11.6 | 4.7 | -2.6 | 1.824 | 72.3 | 1.016 | 95.8 | 1.011 | 0.992 |
| SPY/BRK-B | dd20_dip|hard | 20.1 | 11.6 | 4.8 | -2.2 | 1.829 | 73.7 | 1.021 | 74.2 | 1.024 | 0.998 |
| SPY/BRK-B | dd20_flee|hard | 79.9 | 11.2 | 3.7 | -5.9 | 1.791 | 26.3 | 0.979 | 25.8 | 0.976 | 0.965 |
| SPY/BRK-B | z_mean_revert|hard | 49.7 | 11.7 | 4.5 | -3.0 | 1.841 | 89.7 | 1.019 | 100.0 | 1.019 | 0.996 |
| SPY/BRK-B | z_momentum|hard | 50.3 | 10.9 | 3.9 | -4.9 | 1.763 | 10.3 | 0.981 | 0.0 | 0.981 | 0.957 |
| QQQ/SPY | mom12|hard | 72.0 | 15.0 | 4.5 | -8.7 | 2.194 | 75.6 | 1.046 | 21.6 | 0.982 | 0.93 |
| QQQ/SPY | contra12|hard | 28.0 | 13.0 | 5.4 | -6.8 | 1.963 | 24.4 | 0.954 | 78.4 | 1.021 | 0.85 |
| QQQ/SPY | trend_A_above_sma200|hard | 77.6 | 15.2 | 5.0 | -8.0 | 2.218 | 90.1 | 1.043 | 25.4 | 0.987 | 0.95 |
| QQQ/SPY | trend_rev|hard | 22.4 | 12.6 | 4.9 | -7.4 | 1.924 | 9.9 | 0.957 | 74.6 | 1.013 | 0.846 |
| QQQ/SPY | dd20_dip|hard | 57.7 | 14.6 | 6.8 | -6.6 | 2.144 | 70.9 | 1.073 | 63.4 | 1.024 | 0.98 |
| QQQ/SPY | dd20_flee|hard | 42.3 | 13.0 | 2.9 | -8.8 | 1.969 | 29.1 | 0.927 | 17.8 | 0.974 | 0.828 |
| QQQ/SPY | z_mean_revert|hard | 25.7 | 12.9 | 5.4 | -6.8 | 1.954 | 24.9 | 0.934 | 77.9 | 1.026 | 0.835 |
| QQQ/SPY | z_momentum|hard | 74.3 | 15.0 | 4.5 | -8.7 | 2.197 | 75.1 | 1.066 | 22.1 | 0.979 | 0.94 |
| ^NDX/BRK-A | mom12|hard | 54.9 | 12.6 | 4.7 | -5.3 | 1.925 | 32.6 | 0.98 | 6.5 | 0.966 | 0.885 |
| ^NDX/BRK-A | contra12|hard | 45.1 | 12.6 | 6.7 | -2.2 | 1.926 | 67.4 | 1.02 | 93.5 | 1.037 | 0.917 |
| ^NDX/BRK-A | trend_A_above_sma200|hard | 74.3 | 12.9 | 4.7 | -5.0 | 1.96 | 52.2 | 1.002 | 17.8 | 0.971 | 0.92 |
| ^NDX/BRK-A | trend_rev|hard | 25.7 | 12.2 | 6.5 | -2.5 | 1.891 | 47.8 | 0.998 | 82.2 | 1.028 | 0.904 |
| ^NDX/BRK-A | dd20_dip|hard | 51.6 | 13.7 | 6.5 | -4.2 | 2.048 | 72.1 | 1.058 | 71.8 | 1.046 | 0.961 |
| ^NDX/BRK-A | dd20_flee|hard | 48.4 | 11.3 | 4.6 | -3.2 | 1.8 | 27.9 | 0.942 | 14.5 | 0.958 | 0.853 |
| ^NDX/BRK-A | z_mean_revert|hard | 43.1 | 12.5 | 6.9 | -2.0 | 1.92 | 68.6 | 1.022 | 97.9 | 1.037 | 0.92 |
| ^NDX/BRK-A | z_momentum|hard | 56.9 | 12.6 | 4.5 | -5.6 | 1.923 | 31.4 | 0.978 | 2.1 | 0.965 | 0.885 |
| VFINX/BRK-A | mom12|hard | 39.0 | 12.2 | 5.0 | -4.2 | 1.882 | 37.5 | 0.986 | 9.7 | 0.982 | 0.92 |
| VFINX/BRK-A | contra12|hard | 61.0 | 12.3 | 5.6 | -3.4 | 1.894 | 62.5 | 1.014 | 90.3 | 1.019 | 0.944 |
| VFINX/BRK-A | trend_A_above_sma200|hard | 77.8 | 11.7 | 4.7 | -5.2 | 1.841 | 14.4 | 0.956 | 22.8 | 0.989 | 0.907 |
| VFINX/BRK-A | trend_rev|hard | 22.2 | 12.5 | 5.7 | -2.5 | 1.913 | 85.6 | 1.044 | 77.2 | 1.011 | 0.964 |
| VFINX/BRK-A | dd20_dip|hard | 17.3 | 12.6 | 5.8 | -2.1 | 1.926 | 86.1 | 1.055 | 64.5 | 1.021 | 0.997 |
| VFINX/BRK-A | dd20_flee|hard | 82.6 | 11.4 | 4.3 | -5.7 | 1.812 | 13.9 | 0.945 | 27.0 | 0.979 | 0.882 |
| VFINX/BRK-A | z_mean_revert|hard | 58.5 | 12.3 | 5.7 | -3.0 | 1.894 | 62.0 | 1.015 | 82.4 | 1.02 | 0.946 |
| VFINX/BRK-A | z_momentum|hard | 41.5 | 12.2 | 4.8 | -4.7 | 1.884 | 38.0 | 0.985 | 17.6 | 0.981 | 0.925 |

Fixed baselines (10y):

| pair | rule | irr_med | irr_p10 | irr_min | wm_med | vsbest_med |
|---|---|---|---|---|---|---|
| QQQ/BRK-B | fixed_100A | 15.9 | 6.8 | -7.5 | 2.3 | 1.0 |
| QQQ/BRK-B | fixed_0A_100B | 11.2 | 4.6 | -0.7 | 1.79 | 0.774 |
| QQQ/BRK-B | fixed_50_50 | 14.0 | 6.1 | -3.8 | 2.08 | 0.886 |
| QQQ/BRK-B | fixed_70A_30B | 14.9 | 6.6 | -5.2 | 2.18 | 0.93 |
| QQQ/BRK-B | fixed_30A_70B | 13.0 | 5.8 | -2.5 | 1.96 | 0.842 |
| SPY/BRK-B | fixed_100A | 11.3 | 2.9 | -7.9 | 1.8 | 1.0 |
| SPY/BRK-B | fixed_0A_100B | 11.2 | 4.6 | -0.7 | 1.79 | 0.994 |
| SPY/BRK-B | fixed_50_50 | 11.3 | 4.2 | -3.9 | 1.8 | 0.976 |
| SPY/BRK-B | fixed_70A_30B | 11.3 | 3.8 | -5.4 | 1.8 | 0.981 |
| SPY/BRK-B | fixed_30A_70B | 11.4 | 4.7 | -2.6 | 1.81 | 0.976 |
| QQQ/SPY | fixed_100A | 15.9 | 6.8 | -7.5 | 2.3 | 1.0 |
| QQQ/SPY | fixed_0A_100B | 11.3 | 2.9 | -7.9 | 1.8 | 0.783 |
| QQQ/SPY | fixed_50_50 | 13.8 | 5.0 | -7.7 | 2.05 | 0.891 |
| QQQ/SPY | fixed_70A_30B | 14.6 | 5.7 | -7.6 | 2.15 | 0.935 |
| QQQ/SPY | fixed_30A_70B | 12.8 | 4.2 | -7.8 | 1.95 | 0.848 |
| ^NDX/BRK-A | fixed_100A | 13.9 | 3.4 | -7.7 | 2.06 | 1.0 |
| ^NDX/BRK-A | fixed_0A_100B | 11.6 | 5.8 | -0.4 | 1.83 | 0.856 |
| ^NDX/BRK-A | fixed_50_50 | 12.5 | 5.8 | -3.7 | 1.92 | 0.897 |
| ^NDX/BRK-A | fixed_70A_30B | 13.2 | 5.3 | -5.2 | 1.99 | 0.928 |
| ^NDX/BRK-A | fixed_30A_70B | 12.4 | 6.1 | -2.3 | 1.91 | 0.888 |
| VFINX/BRK-A | fixed_100A | 11.3 | 4.1 | -8.0 | 1.8 | 0.892 |
| VFINX/BRK-A | fixed_0A_100B | 12.4 | 6.0 | -0.4 | 1.91 | 1.0 |
| VFINX/BRK-A | fixed_50_50 | 12.0 | 5.2 | -3.8 | 1.87 | 0.928 |
| VFINX/BRK-A | fixed_70A_30B | 12.0 | 5.0 | -5.4 | 1.87 | 0.924 |
| VFINX/BRK-A | fixed_30A_70B | 12.3 | 5.4 | -2.4 | 1.89 | 0.948 |

**20-year windows**

| pair | rule | avg_fracA | irr_med | irr_p10 | irr_min | wm_med | vs50_beat | vs50_med | vsmatch_beat | vsmatch_med | vsbest_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | mom12|hard | 62.2 | 14.2 | 10.8 | 10.0 | 4.995 | 53.8 | 1.003 | 0.0 | 0.936 | 0.79 |
| QQQ/BRK-B | contra12|hard | 37.8 | 14.0 | 11.6 | 10.7 | 4.875 | 46.2 | 0.997 | 100.0 | 1.079 | 0.762 |
| QQQ/BRK-B | trend_A_above_sma200|hard | 77.0 | 14.8 | 11.4 | 10.6 | 5.411 | 100.0 | 1.092 | 0.0 | 0.949 | 0.859 |
| QQQ/BRK-B | trend_rev|hard | 23.0 | 13.2 | 10.9 | 10.0 | 4.453 | 0.0 | 0.908 | 100.0 | 1.07 | 0.697 |
| QQQ/BRK-B | dd20_dip|hard | 57.2 | 15.8 | 12.6 | 11.8 | 6.134 | 100.0 | 1.217 | 100.0 | 1.178 | 0.954 |
| QQQ/BRK-B | dd20_flee|hard | 42.8 | 11.7 | 9.3 | 8.3 | 3.68 | 0.0 | 0.783 | 0.0 | 0.801 | 0.605 |
| QQQ/BRK-B | z_mean_revert|hard | 35.2 | 13.9 | 11.5 | 10.7 | 4.85 | 46.2 | 0.998 | 100.0 | 1.089 | 0.758 |
| QQQ/BRK-B | z_momentum|hard | 64.8 | 14.2 | 10.8 | 10.0 | 4.985 | 53.8 | 1.002 | 0.0 | 0.92 | 0.796 |
| SPY/BRK-B | mom12|hard | 46.8 | 11.0 | 8.8 | 7.7 | 3.402 | 0.0 | 0.978 | 0.0 | 0.978 | 0.946 |
| SPY/BRK-B | contra12|hard | 53.2 | 11.4 | 9.4 | 8.0 | 3.554 | 100.0 | 1.022 | 100.0 | 1.022 | 0.992 |
| SPY/BRK-B | trend_A_above_sma200|hard | 77.6 | 11.1 | 9.1 | 7.8 | 3.427 | 34.4 | 0.984 | 0.0 | 0.977 | 0.957 |
| SPY/BRK-B | trend_rev|hard | 22.4 | 11.3 | 9.3 | 7.9 | 3.522 | 65.6 | 1.016 | 100.0 | 1.022 | 0.991 |
| SPY/BRK-B | dd20_dip|hard | 20.1 | 11.3 | 9.5 | 8.1 | 3.518 | 79.6 | 1.021 | 100.0 | 1.028 | 1.0 |
| SPY/BRK-B | dd20_flee|hard | 79.9 | 11.0 | 8.9 | 7.6 | 3.419 | 20.4 | 0.979 | 0.0 | 0.972 | 0.948 |
| SPY/BRK-B | z_mean_revert|hard | 49.4 | 11.4 | 9.4 | 8.1 | 3.566 | 100.0 | 1.025 | 100.0 | 1.025 | 0.994 |
| SPY/BRK-B | z_momentum|hard | 50.6 | 11.0 | 8.8 | 7.6 | 3.392 | 0.0 | 0.975 | 0.0 | 0.975 | 0.942 |
| QQQ/SPY | mom12|hard | 70.5 | 14.6 | 11.0 | 10.1 | 5.283 | 86.0 | 1.05 | 0.0 | 0.953 | 0.835 |
| QQQ/SPY | contra12|hard | 29.5 | 13.8 | 11.1 | 10.2 | 4.754 | 14.0 | 0.95 | 100.0 | 1.06 | 0.738 |
| QQQ/SPY | trend_A_above_sma200|hard | 77.0 | 15.0 | 11.3 | 10.4 | 5.523 | 100.0 | 1.094 | 0.0 | 0.952 | 0.868 |
| QQQ/SPY | trend_rev|hard | 23.0 | 13.4 | 10.8 | 9.9 | 4.557 | 0.0 | 0.906 | 100.0 | 1.06 | 0.703 |
| QQQ/SPY | dd20_dip|hard | 57.2 | 15.9 | 12.5 | 11.5 | 6.174 | 100.0 | 1.214 | 100.0 | 1.177 | 0.96 |
| QQQ/SPY | dd20_flee|hard | 42.8 | 12.1 | 9.2 | 8.1 | 3.859 | 0.0 | 0.786 | 0.0 | 0.803 | 0.616 |
| QQQ/SPY | z_mean_revert|hard | 27.2 | 13.7 | 11.1 | 10.2 | 4.724 | 12.9 | 0.953 | 100.0 | 1.068 | 0.74 |
| QQQ/SPY | z_momentum|hard | 72.8 | 14.6 | 11.0 | 10.1 | 5.277 | 87.1 | 1.047 | 0.0 | 0.945 | 0.842 |
| ^NDX/BRK-A | mom12|hard | 54.4 | 10.0 | 8.7 | 7.9 | 3.025 | 22.1 | 0.963 | 0.0 | 0.942 | 0.845 |
| ^NDX/BRK-A | contra12|hard | 45.6 | 11.0 | 9.4 | 8.8 | 3.402 | 77.9 | 1.037 | 100.0 | 1.061 | 0.92 |
| ^NDX/BRK-A | trend_A_above_sma200|hard | 72.6 | 10.1 | 8.7 | 7.4 | 3.045 | 40.6 | 0.986 | 0.0 | 0.964 | 0.884 |
| ^NDX/BRK-A | trend_rev|hard | 27.4 | 11.0 | 9.3 | 8.5 | 3.386 | 59.4 | 1.014 | 100.0 | 1.038 | 0.906 |
| ^NDX/BRK-A | dd20_dip|hard | 61.2 | 11.7 | 9.8 | 9.1 | 3.697 | 100.0 | 1.15 | 100.0 | 1.117 | 0.999 |
| ^NDX/BRK-A | dd20_flee|hard | 38.8 | 9.3 | 8.2 | 6.6 | 2.766 | 0.0 | 0.85 | 0.0 | 0.876 | 0.748 |
| ^NDX/BRK-A | z_mean_revert|hard | 43.9 | 10.8 | 9.3 | 8.3 | 3.308 | 73.7 | 1.025 | 96.8 | 1.052 | 0.896 |
| ^NDX/BRK-A | z_momentum|hard | 56.1 | 10.4 | 8.7 | 7.9 | 3.151 | 26.3 | 0.975 | 3.2 | 0.95 | 0.852 |
| VFINX/BRK-A | mom12|hard | 39.0 | 10.2 | 8.3 | 7.2 | 3.076 | 51.9 | 1.016 | 33.2 | 0.982 | 0.902 |
| VFINX/BRK-A | contra12|hard | 61.0 | 10.0 | 7.9 | 6.8 | 3.019 | 48.1 | 0.984 | 66.8 | 1.019 | 0.871 |
| VFINX/BRK-A | trend_A_above_sma200|hard | 75.6 | 9.5 | 7.0 | 5.3 | 2.83 | 10.6 | 0.915 | 13.1 | 0.976 | 0.817 |
| VFINX/BRK-A | trend_rev|hard | 24.4 | 11.0 | 8.8 | 7.8 | 3.419 | 89.4 | 1.085 | 86.9 | 1.023 | 0.957 |
| VFINX/BRK-A | dd20_dip|hard | 22.2 | 11.4 | 9.0 | 8.0 | 3.552 | 94.0 | 1.116 | 100.0 | 1.046 | 0.99 |
| VFINX/BRK-A | dd20_flee|hard | 77.8 | 9.1 | 6.1 | 3.2 | 2.704 | 6.0 | 0.884 | 0.0 | 0.949 | 0.795 |
| VFINX/BRK-A | z_mean_revert|hard | 57.5 | 9.9 | 7.8 | 6.4 | 2.983 | 49.5 | 0.993 | 55.8 | 1.014 | 0.882 |
| VFINX/BRK-A | z_momentum|hard | 42.5 | 10.4 | 8.3 | 7.2 | 3.145 | 50.5 | 1.007 | 44.2 | 0.988 | 0.901 |

Fixed baselines (20y):

| pair | rule | irr_med | irr_p10 | irr_min | wm_med | vsbest_med |
|---|---|---|---|---|---|---|
| QQQ/BRK-B | fixed_100A | 16.3 | 12.6 | 11.6 | 6.46 | 1.0 |
| QQQ/BRK-B | fixed_0A_100B | 11.0 | 9.2 | 7.6 | 3.39 | 0.558 |
| QQQ/BRK-B | fixed_50_50 | 14.1 | 11.2 | 10.4 | 4.94 | 0.779 |
| QQQ/BRK-B | fixed_70A_30B | 15.0 | 11.8 | 10.9 | 5.55 | 0.868 |
| QQQ/BRK-B | fixed_30A_70B | 13.0 | 10.4 | 9.5 | 4.33 | 0.691 |
| SPY/BRK-B | fixed_100A | 11.3 | 9.0 | 7.7 | 3.53 | 1.0 |
| SPY/BRK-B | fixed_0A_100B | 11.0 | 9.2 | 7.6 | 3.39 | 0.988 |
| SPY/BRK-B | fixed_50_50 | 11.2 | 9.1 | 7.9 | 3.48 | 0.97 |
| SPY/BRK-B | fixed_70A_30B | 11.2 | 9.2 | 7.8 | 3.5 | 0.971 |
| SPY/BRK-B | fixed_30A_70B | 11.1 | 9.2 | 7.9 | 3.45 | 0.973 |
| QQQ/SPY | fixed_100A | 16.3 | 12.6 | 11.6 | 6.46 | 1.0 |
| QQQ/SPY | fixed_0A_100B | 11.3 | 9.0 | 7.7 | 3.53 | 0.558 |
| QQQ/SPY | fixed_50_50 | 14.2 | 11.0 | 10.2 | 5.0 | 0.779 |
| QQQ/SPY | fixed_70A_30B | 15.1 | 11.7 | 10.8 | 5.58 | 0.868 |
| QQQ/SPY | fixed_30A_70B | 13.2 | 10.3 | 9.4 | 4.41 | 0.691 |
| ^NDX/BRK-A | fixed_100A | 10.3 | 8.5 | 5.6 | 3.12 | 1.0 |
| ^NDX/BRK-A | fixed_0A_100B | 10.2 | 8.9 | 7.6 | 3.08 | 0.9 |
| ^NDX/BRK-A | fixed_50_50 | 10.5 | 9.1 | 8.4 | 3.22 | 0.879 |
| ^NDX/BRK-A | fixed_70A_30B | 10.4 | 9.0 | 7.7 | 3.16 | 0.9 |
| ^NDX/BRK-A | fixed_30A_70B | 10.6 | 9.1 | 8.3 | 3.25 | 0.886 |
| VFINX/BRK-A | fixed_100A | 9.1 | 6.1 | 3.0 | 2.72 | 0.793 |
| VFINX/BRK-A | fixed_0A_100B | 11.1 | 9.0 | 7.6 | 3.44 | 1.0 |
| VFINX/BRK-A | fixed_50_50 | 10.0 | 8.1 | 7.4 | 3.02 | 0.892 |
| VFINX/BRK-A | fixed_70A_30B | 9.6 | 7.6 | 6.4 | 2.88 | 0.855 |
| VFINX/BRK-A | fixed_30A_70B | 10.6 | 8.6 | 7.6 | 3.25 | 0.914 |

**5-year windows**

| pair | rule | avg_fracA | irr_med | irr_p10 | irr_min | wm_med | vs50_beat | vs50_med | vsmatch_beat | vsmatch_med | vsbest_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | mom12|hard | 60.7 | 13.7 | 1.9 | -13.3 | 1.41 | 29.7 | 0.985 | 4.8 | 0.974 | 0.938 |
| QQQ/BRK-B | contra12|hard | 39.3 | 13.2 | 5.8 | -11.0 | 1.394 | 70.3 | 1.015 | 95.2 | 1.026 | 0.96 |
| QQQ/BRK-B | trend_A_above_sma200|hard | 75.4 | 14.4 | 3.2 | -14.2 | 1.435 | 50.6 | 1.001 | 18.7 | 0.982 | 0.957 |
| QQQ/BRK-B | trend_rev|hard | 24.6 | 12.5 | 4.3 | -10.1 | 1.368 | 49.4 | 0.999 | 81.3 | 1.019 | 0.938 |
| QQQ/BRK-B | dd20_dip|hard | 57.2 | 13.2 | 3.5 | -14.4 | 1.392 | 57.1 | 1.01 | 46.9 | 1.0 | 0.979 |
| QQQ/BRK-B | dd20_flee|hard | 42.8 | 14.0 | 2.8 | -9.9 | 1.42 | 42.9 | 0.99 | 2.2 | 1.0 | 0.955 |
| QQQ/BRK-B | z_mean_revert|hard | 36.4 | 13.2 | 6.2 | -10.7 | 1.391 | 65.2 | 1.021 | 99.6 | 1.024 | 0.956 |
| QQQ/BRK-B | z_momentum|hard | 63.6 | 13.8 | 1.4 | -13.5 | 1.411 | 34.8 | 0.979 | 0.4 | 0.978 | 0.94 |
| SPY/BRK-B | mom12|hard | 46.0 | 11.1 | 2.7 | -14.4 | 1.319 | 4.8 | 0.98 | 0.4 | 0.98 | 0.948 |
| SPY/BRK-B | contra12|hard | 54.0 | 12.4 | 4.1 | -13.4 | 1.364 | 95.2 | 1.02 | 99.6 | 1.02 | 0.987 |
| SPY/BRK-B | trend_A_above_sma200|hard | 76.7 | 11.6 | 1.2 | -18.3 | 1.338 | 33.7 | 0.984 | 15.8 | 0.99 | 0.964 |
| SPY/BRK-B | trend_rev|hard | 23.3 | 11.8 | 4.4 | -10.0 | 1.343 | 66.3 | 1.016 | 84.2 | 1.01 | 0.988 |
| SPY/BRK-B | dd20_dip|hard | 21.9 | 11.8 | 4.6 | -10.8 | 1.345 | 75.1 | 1.028 | 77.3 | 1.005 | 0.996 |
| SPY/BRK-B | dd20_flee|hard | 78.1 | 11.5 | 1.4 | -17.3 | 1.333 | 24.9 | 0.972 | 7.3 | 0.994 | 0.949 |
| SPY/BRK-B | z_mean_revert|hard | 49.7 | 12.3 | 4.6 | -12.5 | 1.362 | 95.6 | 1.025 | 100.0 | 1.023 | 0.989 |
| SPY/BRK-B | z_momentum|hard | 50.3 | 11.1 | 2.3 | -15.4 | 1.322 | 4.4 | 0.975 | 0.0 | 0.977 | 0.945 |
| QQQ/SPY | mom12|hard | 68.4 | 14.4 | 0.5 | -16.9 | 1.434 | 58.6 | 1.008 | 12.8 | 0.986 | 0.964 |
| QQQ/SPY | contra12|hard | 31.6 | 13.0 | 2.0 | -15.7 | 1.383 | 41.4 | 0.992 | 87.2 | 1.014 | 0.94 |
| QQQ/SPY | trend_A_above_sma200|hard | 75.4 | 14.6 | 1.6 | -16.0 | 1.44 | 73.6 | 1.012 | 20.5 | 0.993 | 0.975 |
| QQQ/SPY | trend_rev|hard | 24.6 | 12.7 | 1.0 | -16.6 | 1.373 | 26.4 | 0.988 | 79.5 | 1.007 | 0.931 |
| QQQ/SPY | dd20_dip|hard | 57.2 | 13.5 | 2.2 | -14.4 | 1.402 | 54.6 | 1.003 | 42.1 | 1.0 | 0.975 |
| QQQ/SPY | dd20_flee|hard | 42.8 | 13.2 | -1.0 | -18.3 | 1.393 | 45.4 | 0.997 | 7.0 | 1.0 | 0.957 |
| QQQ/SPY | z_mean_revert|hard | 29.4 | 12.7 | 2.1 | -15.9 | 1.373 | 38.1 | 0.983 | 83.2 | 1.009 | 0.935 |
| QQQ/SPY | z_momentum|hard | 70.6 | 14.6 | 0.3 | -16.7 | 1.442 | 61.9 | 1.017 | 16.8 | 0.991 | 0.973 |
| ^NDX/BRK-A | mom12|hard | 54.0 | 14.6 | -0.7 | -14.5 | 1.44 | 18.4 | 0.972 | 4.5 | 0.969 | 0.924 |
| ^NDX/BRK-A | contra12|hard | 46.0 | 14.8 | 5.1 | -10.8 | 1.447 | 81.6 | 1.028 | 95.5 | 1.031 | 0.959 |
| ^NDX/BRK-A | trend_A_above_sma200|hard | 75.0 | 15.0 | 0.8 | -13.9 | 1.456 | 42.3 | 0.988 | 31.5 | 0.988 | 0.949 |
| ^NDX/BRK-A | trend_rev|hard | 25.0 | 13.4 | 3.8 | -9.9 | 1.4 | 57.7 | 1.012 | 68.5 | 1.013 | 0.942 |
| ^NDX/BRK-A | dd20_dip|hard | 45.4 | 13.8 | 3.0 | -14.8 | 1.411 | 64.0 | 1.017 | 40.3 | 1.0 | 0.972 |
| ^NDX/BRK-A | dd20_flee|hard | 54.6 | 14.7 | 0.7 | -13.1 | 1.444 | 36.0 | 0.983 | 15.9 | 1.0 | 0.948 |
| ^NDX/BRK-A | z_mean_revert|hard | 44.2 | 14.6 | 5.3 | -10.6 | 1.439 | 75.8 | 1.024 | 99.0 | 1.034 | 0.957 |
| ^NDX/BRK-A | z_momentum|hard | 55.8 | 14.6 | -1.0 | -16.0 | 1.441 | 24.2 | 0.976 | 1.0 | 0.969 | 0.924 |
| VFINX/BRK-A | mom12|hard | 38.2 | 13.2 | 2.5 | -13.9 | 1.391 | 26.8 | 0.986 | 10.4 | 0.98 | 0.935 |
| VFINX/BRK-A | contra12|hard | 61.8 | 14.3 | 4.0 | -13.2 | 1.432 | 73.2 | 1.014 | 89.6 | 1.021 | 0.959 |
| VFINX/BRK-A | trend_A_above_sma200|hard | 77.7 | 13.4 | 1.0 | -18.0 | 1.398 | 24.8 | 0.971 | 31.8 | 0.994 | 0.925 |
| VFINX/BRK-A | trend_rev|hard | 22.3 | 14.0 | 4.4 | -9.5 | 1.418 | 75.2 | 1.029 | 68.2 | 1.005 | 0.969 |
| VFINX/BRK-A | dd20_dip|hard | 16.1 | 13.8 | 4.9 | -10.3 | 1.412 | 81.2 | 1.041 | 57.7 | 1.001 | 0.996 |
| VFINX/BRK-A | dd20_flee|hard | 83.9 | 12.5 | 0.6 | -17.1 | 1.368 | 18.8 | 0.959 | 13.0 | 0.999 | 0.914 |
| VFINX/BRK-A | z_mean_revert|hard | 59.4 | 13.6 | 4.6 | -12.1 | 1.405 | 69.8 | 1.015 | 89.6 | 1.023 | 0.963 |
| VFINX/BRK-A | z_momentum|hard | 40.6 | 13.2 | 2.0 | -15.0 | 1.394 | 30.2 | 0.985 | 10.4 | 0.978 | 0.939 |

Fixed baselines (5y):

| pair | rule | irr_med | irr_p10 | irr_min | wm_med | vsbest_med |
|---|---|---|---|---|---|---|
| QQQ/BRK-B | fixed_100A | 15.7 | 1.6 | -14.4 | 1.48 | 1.0 |
| QQQ/BRK-B | fixed_0A_100B | 11.4 | 2.7 | -9.9 | 1.33 | 0.923 |
| QQQ/BRK-B | fixed_50_50 | 13.4 | 4.2 | -12.1 | 1.4 | 0.946 |
| QQQ/BRK-B | fixed_70A_30B | 14.4 | 3.2 | -13.0 | 1.43 | 0.963 |
| QQQ/BRK-B | fixed_30A_70B | 12.8 | 4.7 | -11.2 | 1.38 | 0.936 |
| SPY/BRK-B | fixed_100A | 11.8 | 0.9 | -18.3 | 1.34 | 1.0 |
| SPY/BRK-B | fixed_0A_100B | 11.4 | 2.7 | -9.9 | 1.33 | 1.0 |
| SPY/BRK-B | fixed_50_50 | 11.8 | 3.4 | -13.9 | 1.34 | 0.969 |
| SPY/BRK-B | fixed_70A_30B | 11.7 | 3.2 | -15.6 | 1.34 | 0.972 |
| SPY/BRK-B | fixed_30A_70B | 11.8 | 3.5 | -12.3 | 1.34 | 0.97 |
| QQQ/SPY | fixed_100A | 15.7 | 1.6 | -14.4 | 1.48 | 1.0 |
| QQQ/SPY | fixed_0A_100B | 11.8 | 0.9 | -18.3 | 1.34 | 0.902 |
| QQQ/SPY | fixed_50_50 | 13.7 | 1.3 | -16.3 | 1.41 | 0.948 |
| QQQ/SPY | fixed_70A_30B | 14.5 | 1.3 | -15.5 | 1.44 | 0.968 |
| QQQ/SPY | fixed_30A_70B | 12.8 | 0.7 | -17.1 | 1.38 | 0.932 |
| ^NDX/BRK-A | fixed_100A | 15.6 | -3.5 | -27.2 | 1.48 | 1.0 |
| ^NDX/BRK-A | fixed_0A_100B | 12.4 | 3.4 | -9.2 | 1.36 | 0.976 |
| ^NDX/BRK-A | fixed_50_50 | 14.6 | 2.7 | -11.9 | 1.44 | 0.935 |
| ^NDX/BRK-A | fixed_70A_30B | 15.3 | 0.7 | -14.0 | 1.47 | 0.951 |
| ^NDX/BRK-A | fixed_30A_70B | 13.8 | 3.8 | -10.8 | 1.41 | 0.937 |
| VFINX/BRK-A | fixed_100A | 12.3 | 0.2 | -18.5 | 1.36 | 0.919 |
| VFINX/BRK-A | fixed_0A_100B | 13.8 | 4.4 | -9.2 | 1.41 | 1.0 |
| VFINX/BRK-A | fixed_50_50 | 14.0 | 3.3 | -13.6 | 1.42 | 0.937 |
| VFINX/BRK-A | fixed_70A_30B | 13.8 | 2.8 | -15.4 | 1.41 | 0.94 |
| VFINX/BRK-A | fixed_30A_70B | 13.8 | 3.9 | -11.8 | 1.41 | 0.948 |

Observations. (a) Against fixed 50/50 the picture is dominated by average allocation: where A has beaten B (QQQ vs BRK-B in 1999-2026) any rule that puts more money in A "wins" (e.g. trend hard, 10y: 81% of windows beat 50/50, but only 22% beat their own matched mix, median ratio 0.986). Where B beat A (VFINX vs BRK-A) the sign flips. (b) The hard trend rule beats 50/50 only because it holds about 75-78% A on average; against the matched mix it is at or below 1 in every pair (0.97-0.99 at 10y). (c) At 5y and 10y no rule's median beats the hindsight best of all-A / all-B (max vsbest_med 1.000); at 20y the best is 1.006 (a drawdown-dip variant). Most rules sit at 0.8-1.0. (d) Era split (`b_rules_by_era.csv`, windows grouped by start year <2000 / 2000-09 / 2010+): the 5y z-score mean-reversion rule beats its matched mix in all three eras in SPY/BRK-B, ^NDX/BRK-A and VFINX/BRK-A (median ratio 1.02 to 1.08; 76-100% of windows); the QQQ/BRK-B 20y drawdown dip rule gets 1.13 and 1.20 in <2000 and 2000-09 starts and there are no 2010+ 20y windows. Momentum rules (e.g. mom6 hard, QQQ/BRK-B 10y) are below 1 in all three eras (0.967 / 0.973 / 0.980).

### B2. Placebo and multiple testing
Best of 48 rules per pair/horizon vs the null distribution of the best of 48 under shifted signals; `pct_best_in_shift_max_null` = percentile of the real best among the 644-1024 null maxima (95+ would be the usual bar); `pct_best_in_block_max_null` = same against the block-shuffle max-null.

| pair | horizon | n_windows | best_rule | best_real_med_ratio | shift_null_max_p95 | pct_best_in_shift_max_null | pct_best_in_block_max_null | n_above_own_shift_p95 | n_below_own_shift_p5 |
|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | 5y | 273 | contra12 hard | 1.026 | 1.019 | 99.379 | 96.3 | 18 | 14 |
| QQQ/BRK-B | 10y | 213 | dd30_dip hard | 1.036 | 1.047 | 70.342 | 79.1 | 6 | 6 |
| QQQ/BRK-B | 20y | 93 | dd30_dip hard | 1.193 | 1.212 | 78.106 | 99.2 | 4 | 13 |
| SPY/BRK-B | 5y | 273 | z_mean_revert hard | 1.023 | 1.017 | 100.0 | 99.8 | 20 | 20 |
| SPY/BRK-B | 10y | 213 | dd20_dip hard | 1.024 | 1.023 | 96.118 | 98.9 | 18 | 18 |
| SPY/BRK-B | 20y | 93 | dd20_dip hard | 1.028 | 1.033 | 89.13 | 97.0 | 12 | 12 |
| QQQ/SPY | 5y | 273 | contra12 hard | 1.014 | 1.009 | 100.0 | 99.3 | 13 | 11 |
| QQQ/SPY | 10y | 213 | dd30_dip hard | 1.031 | 1.035 | 81.988 | 83.9 | 8 | 7 |
| QQQ/SPY | 20y | 93 | dd30_dip hard | 1.185 | 1.204 | 75.311 | 98.8 | 1 | 10 |
| ^NDX/BRK-A | 5y | 397 | z_mean_revert hard | 1.034 | 1.021 | 99.888 | 99.7 | 14 | 12 |
| ^NDX/BRK-A | 10y | 337 | dd20_dip hard | 1.046 | 1.042 | 98.543 | 96.3 | 24 | 24 |
| ^NDX/BRK-A | 20y | 217 | dd30_dip hard | 1.131 | 1.127 | 97.982 | 99.8 | 12 | 16 |
| VFINX/BRK-A | 5y | 463 | z_mean_revert hard | 1.023 | 1.016 | 99.902 | 100.0 | 18 | 18 |
| VFINX/BRK-A | 10y | 403 | dd20_dip hard | 1.021 | 1.02 | 96.289 | 98.9 | 24 | 24 |
| VFINX/BRK-A | 20y | 283 | dd20_dip hard | 1.046 | 1.048 | 93.945 | 99.5 | 12 | 12 |

Reading. At 5 years the best real rule exceeds the 99th percentile of the null maximum in all five pairs (99.4 for QQQ/BRK-B, 100 for SPY/BRK-B and QQQ/SPY, 99.9 for ^NDX/BRK-A and VFINX/BRK-A; block-shuffle placebo 96-100). The best rule is `z_mean_revert hard` for SPY/BRK-B, ^NDX/BRK-A and VFINX/BRK-A and `contra12 hard` for QQQ/BRK-B and QQQ/SPY, i.e. buying the lagging asset, with an edge of +1.4% to +3.4% in median terminal wealth vs the matched mix. At 10 years the best real rule sits at the 70th-98.5th percentile of the null maximum (QQQ/BRK-B 70, QQQ/SPY 82, SPY/BRK-B 96, VFINX/BRK-A 96, ^NDX/BRK-A 98.5), so only borderline. At 20 years: 75-78 for the two QQQ pairs (best rule is a drawdown-dip with +19%, but the null maximum's own 95th percentile is +21%, because the sample holds about 1.4 independent 20-year periods), 89 for SPY/BRK-B, 94 for VFINX/BRK-A, 98 for ^NDX/BRK-A (block placebo gives 97-100 at 20y, so the two placebos disagree at 20y). Several cells show many rules outside their own 5-95% band (e.g. ^NDX/BRK-A 10y: 24 above, 24 below of 48), but this reflects mirror-image rule pairs: if a rule is above, its reverse is below.

So the one pattern that survives the multiple-testing control at the 5-year horizon in all five samples (z-score mean reversion in three, 12-month contrarian in two) is "buy the laggard on a 36-month z-score of the ratio" (or 12-month contrarian). It is an in-sample result with about 5-9 independent 5-year periods, found as the best of 48 variants in the sample, and it does not reach comparable significance at 10-20 years. It is also consistent in sign across eras. Treat it as a hint of mean reversion in relative strength, which matches the weak A3 long-horizon correlations, not as a tested edge. Its size (about 2% of median terminal wealth over five years, i.e. about 0.4%/year) is also smaller than the effect of choosing the average allocation (all-A vs all-B differs by 30-100% in terminal wealth at 10-20y).

### B3. State-dependent rule: buy_underweight (50/50 target) vs fixed_buy 50/50
`DCABacktest`, same costs. Windows of 10y / 20y started every 12 months (n windows per row), plus one full-sample run. IRR difference in percentage points (buy_underweight minus fixed); max drawdown is the time-weighted NAV drawdown of the portfolio; final weights = value share in A at the end of the window (median).

| pair | horizon | n_windows | irr_buy_underweight | irr_fixed | irr_diff_med_pp | irr_diff_mean_pp | pct_windows_bu_better | mdd_bu | mdd_fixed | final_wA_bu | final_wA_fixed | wm_ratio_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QQQ/BRK-B | full | 1 | 0.133 | 0.133 | 0.001 | 0.001 | 100.0 | -0.52 | -0.524 | 0.571 | 0.635 | 1.0 |
| QQQ/BRK-B | 10y | 18 | 0.142 | 0.142 | 0.126 | 0.107 | 77.778 | -0.486 | -0.492 | 0.505 | 0.561 | 1.007 |
| QQQ/BRK-B | 20y | 8 | 0.138 | 0.14 | -0.088 | -0.083 | 12.5 | -0.518 | -0.523 | 0.542 | 0.626 | 0.989 |
| SPY/BRK-B | full | 1 | 0.115 | 0.114 | 0.076 | 0.076 | 100.0 | -0.537 | -0.538 | 0.499 | 0.481 | 1.014 |
| SPY/BRK-B | 10y | 18 | 0.116 | 0.114 | 0.168 | 0.152 | 88.889 | -0.501 | -0.504 | 0.495 | 0.499 | 1.009 |
| SPY/BRK-B | 20y | 8 | 0.11 | 0.109 | 0.122 | 0.117 | 100.0 | -0.536 | -0.538 | 0.498 | 0.495 | 1.015 |
| QQQ/SPY | full | 1 | 0.13 | 0.132 | -0.183 | -0.183 | 0.0 | -0.692 | -0.692 | 0.553 | 0.652 | 0.967 |
| QQQ/SPY | 10y | 18 | 0.139 | 0.14 | -0.089 | -0.083 | 22.222 | -0.518 | -0.518 | 0.504 | 0.558 | 0.995 |
| QQQ/SPY | 20y | 8 | 0.139 | 0.142 | -0.284 | -0.24 | 0.0 | -0.537 | -0.538 | 0.527 | 0.637 | 0.966 |
| ^NDX/BRK-A | full | 1 | 0.138 | 0.136 | 0.223 | 0.223 | 100.0 | -0.606 | -0.574 | 0.609 | 0.543 | 1.062 |
| ^NDX/BRK-A | 10y | 29 | 0.127 | 0.124 | 0.27 | 0.381 | 89.655 | -0.501 | -0.509 | 0.507 | 0.544 | 1.014 |
| ^NDX/BRK-A | 20y | 19 | 0.108 | 0.107 | 0.122 | 0.164 | 68.421 | -0.508 | -0.568 | 0.52 | 0.529 | 1.015 |
| VFINX/BRK-A | full | 1 | 0.133 | 0.138 | -0.526 | -0.526 | 0.0 | -0.522 | -0.518 | 0.382 | 0.206 | 0.847 |
| VFINX/BRK-A | 10y | 34 | 0.126 | 0.124 | 0.144 | -0.076 | 70.588 | -0.321 | -0.321 | 0.494 | 0.465 | 1.008 |
| VFINX/BRK-A | 20y | 24 | 0.103 | 0.104 | 0.097 | -0.186 | 62.5 | -0.524 | -0.523 | 0.49 | 0.435 | 1.011 |

Mechanical "buy whatever is below target" is close to to a fixed 50/50 split, because it never sells and contributions are small relative to the stock of holdings: the IRR difference is within about +/-0.3 pp per year in most rows (+0.27 pp median for ^NDX/BRK-A 10y; -0.28 for QQQ/SPY 20y). It steers new money to the laggard, so the final weight in A is lower than under fixed splits when A has run away (QQQ/BRK-B 10y: 50% vs 56%; 20y: 54% vs 63%), and the NAV drawdown is usually 0.1-0.6 pp shallower (up to about 6 pp for ^NDX/BRK-A 20y windows, 50.9% vs 56.8%). The sign of the IRR difference depends on which asset happened to lag: positive in SPY/BRK-B and ^NDX/BRK-A (the laggard later recovered), negative for QQQ/SPY and QQQ/BRK-B 20y and for the VFINX/BRK-A full sample (-0.53 pp/yr; underweight rule ends 38% in A versus 21% for fixed, because BRK-A compounded much faster there and the rule kept adding to the stronger-than-target side only slowly). The full-sample rows are single runs; the 10y/20y windows overlap heavily (started every 12 months).

---
## Answers to the owner's hypothesis
- "Berkshire holds up better in drawdowns": true on median (S&P -17%, NDX -18 to -22%, BRK -7 to -12% peak to trough), in about two-thirds of episodes, with large exceptions (1990, 1999, 2007-09 -40%, 2020 -30%). It rebounds less after troughs in recent decades.
- "When tech is hot, buy Berkshire while it lags": the matching test is `contra*` and `z_mean_revert` (money to the laggard). `contra12 hard` beats its matched fixed mix in all 15 pair-horizon cells (+1.4% to +7.9% median terminal wealth; 67-100% of windows); momentum rules ("chase the winner") lose by similar amounts (98% of momentum cells below 1). The edge is small at 5-10 years; the 5y best-of-48 result survives the placebo in all five pairs, the 10y/20y results only borderline.
- "When tech falls, ...": under the drawdown rules, "buy A while it is far below its high" (dip) beats its matched mix (hard dip rules: +0.6% to +4.6% at 10y, +1.8% to +19% at 20y vs matched mix, all above 1) and "flee to B in drawdowns" loses by the mirror amount. The 20y dip result for QQQ is one episode (2000-02) seen from different start dates and is within the best-of-48 null range.
- Whether ANY simple rule beats a fixed split after controlling for average allocation: a small contrarian (buy-the-laggard) edge of about 1-4% in median terminal wealth at 5-10y shows up consistently in sign in every pair; only the 5-year version clearly survives the multiple-testing placebo; none of it is large compared with the allocation choice itself, and it is derived in one 26-43 year history.

## Files
`a1_sp_drawdowns.csv`, `a1_ndx_drawdowns.csv`, `a1_summary.csv`, `a2_capture.csv`, `a3_corr_long.csv`, `a3_corr_matrix.csv`, `a3_quintiles.csv`, `b_rules_summary.csv`, `b_rules_by_era.csv`, `b_placebo.csv`, `b_placebo_max.csv`, `b_buy_underweight_windows.csv`, `b_buy_underweight_summary.csv`, `b_current_signals.csv`, `sanity_check.txt`; figures `fig1_episodes.png`, `fig2_quintiles.png`, `fig3_rules_QQQ_BRKB.png`, `fig4_rules_VFINX_BRKA.png`, `fig5_relative_strength.png`.
