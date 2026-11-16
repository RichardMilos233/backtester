# EXP7 - Out-of-sample and walk-forward tests of the round-1 conclusions

Script: `scripts/dca/exp7_walk_forward.py` (runs `exp7a_oos_replication.py`, `exp7b_wf_rules.py`, `exp7c_wf_weights.py`; shared code `exp7_common.py`; about 7 minutes on 4 cores). This file is generated from `scripts/dca/exp7_prose.md` plus the CSV tables in this folder. Bootstrap: 2000 paths of a stationary block bootstrap (mean block 18 months) of the joint daily returns of all assets, cut into calendar-month blocks (so cross-asset correlation is kept), fixed seeds; plus 500 paths with iid months (block length 1) used as a "no predictability" benchmark. All French series are frictionless index returns with open = close (trade at the decision day's close; signals use closes up to the previous day).

## D. One-screen summary: round-1 claim -> out-of-sample result -> verdict

The most important result is not a replication number but a flaw in the round-1 yardstick, so it comes first.

**The "matched fixed mix" control of exp3 is biased, and the round-1 effects are almost entirely reproduced by a world with no predictability at all.** Three independent checks (below): (1) two synthetic assets with identical iid returns, mean and volatility give mom12 = 0.971 and contra12 = 1.030 against the window-matched mix, the same numbers round 1 found on real pairs (0.966-0.982 and 1.018-1.037) and that the 66 industry pairs give (0.970 / 1.031); (2) resampling the real histories with iid months reproduces the real 66-pair medians to within +-0.002; (3) the round-1 pairs themselves land inside, or on the less extreme side of, their own iid-resampled benchmark. Two mechanical effects explain it: the window-matched control uses each window's own average allocation, which is correlated with the in-window performance the momentum signal follows (this alone costs momentum about 1.2 pp and gives contrarian about +1.3 pp at 10 years); and a contrarian split behaves like rebalancing (it buys the asset that fell, which has a median-wealth advantage even when returns are unpredictable; momentum is the opposite). The circular-shift placebo used in round 1 destroys both effects, so it was the wrong null: it made mechanics look like skill.

| round-1 claim | out-of-sample result (66 industry pairs, 1926-2026, 10y windows, median wealth ratio vs a fixed mix with the same average allocation) | verdict |
|---|---|---|
| Sending the contribution to the trailing-12m WINNER (mom12) loses 2-6% vs a matched mix | 0.970 vs window-matched mix (round 1: 0.966-0.982), 0.982 vs a single global-average mix; negative in 66/66 and 63/66 pairs; same in 1926-62, 1963-98, 1999-2026; non-overlapping windows identical (0.970) | **Pattern replicated, but it is the mechanical value for a world with no predictability** (iid-resampled 90% band 0.968-0.972 matched, 0.979-0.987 global). Not evidence that winner-chasing fails because winners mean-revert. In the 2 international pairs the sign is reversed (global control: FF_MKT/FF_DEVXUS 1.021, VFINX/PRITX 1.013; not significant). |
| Sending it to the LAGGARD (contra12) gains 1.4-7.9% | 1.031 matched (round 1: 1.018-1.037), 1.018 global; positive in 66/66 and 63/66 pairs; all eras; non-overlap 1.031 | **Pattern replicated, equals the iid benchmark** (1.029-1.033 matched, 1.013-1.021 global). Gain is a real median-wealth effect of contrarian splitting (a rebalancing-like benefit) but needs no mean reversion; it is about +1.8% over 10 years vs the best-fair control, not 2-8%. On the new named pairs the gain is 0 to +1.6% (global control) and negative for FF_MKT/FF_DEVXUS (0.981) and VFINX/PRITX (0.986). |
| mom6 / contra6 mirror this | mom6 0.978 / contra6 1.023 matched; 0.988 / 1.013 global | Same verdict as mom12 / contra12 (smaller). |
| z-score mean reversion (36m) beats its matched mix | 1.032 matched, 1.020 global, 98.5% of pairs > 1 | Pattern replicated, inside the iid band (1.029-1.035 / 1.013-1.022). |
| "Buy the dip" drawdown rules look good mainly because of 2000-02 | dd20_dip 1.008 matched / 1.016 global (round 1: 1.021-1.046); 94% / 82% of pairs > 1; windows starting 1926-62: 1.005; 1963-98: 1.011; 1999+: 1.012. dd20_flee 0.992 / 0.984. | **Round-1 magnitude NOT replicated** (1.008 vs 1.021-1.046 on the matched control; the 2000-02 episode is absent from 1926-98 windows); sign right but inside the iid band; dd20_dip is -5% to -6% on the two international pairs (VFINX/PRITX p=0.002). |
| Trend rules add nothing | trend_A_above_sma200 0.989 / 0.993, trend_rev 1.011 / 1.007 (global-control values within +-0.7%) | **Replicated (no timing value)**: a small, same-sign-in-every-era shortfall (trend_A) / gain (trend_rev) of 0.7-1.1%, equal to the iid benchmark, so it is mechanics, not skill. |
| (walk-forward rule selection) | Picking the top of 53 candidates each January: OOS terminal wealth vs fixed 50/50 = 0.91-0.94 (FF_TECH/FF_VALUE), 1.39-1.47 (FF_MKT/FF_VALUE, because it holds VALUE 100%), 0.99 (VFINX/BRK-A), 0.99-1.05 (^NDX/BRK-A); picked rule beat 50/50 by +10.1% in its training window and by +0.8% in the next 5 years | **Not reliably better than fixed 50/50.** In-sample advantage shrinks from +10% to about +1% (about 0 in 2 of 4 pairs). The one large gain is asset persistence (VALUE beat the market), not timing. |
| (walk-forward weights) | "Optimised" splits vs equal weight: within about +-1% in 1926+ 3-asset universe (winner 1.000, max-Sharpe 1.010, rank 0.940, inverse-vol 0.966), min-variance 0.40; larger gains only in the 1990+ VFINX/BRK-A universes (winner 1.28 / 1.69), intervals include 1 (5-95%: 0.81-1.91 / 0.99-2.81) | **Not better than naive equal weight, with unstable weights.** Min-variance and inverse-vol are reliably worse (P(better) 2-18%). Winner-take-all swaps 12-33% of the contribution flow every year (25-33% in the VFINX/BRK-A universes) and reached a -82% max drawdown in the 2000+ universe. |

Number of combinations examined: part A 77 pairs x 9 rules x 3 horizons = 2,079 rule-pair-horizon medians (the 66 industry pairs are built from 12 industries and are not independent: they have at most 11 independent relative-return series, and the 9 rules are 5 distinct ideas plus mirror images, mom/contra, dip/flee, trend/rev being exact mirrors in the matched control); part B 4 pairs x 53 candidates x 3 training schemes, 585 January selections (3 x 195 Januaries), plus a 10-year-start variant; part C 4 universes x 8 methods, 960 weight decisions. Nothing was tuned on the new data; rules, orientation (A = the more volatile industry in 1926-62) and the defensive mix (equal weight of Telcm, NoDur, Hlth: the three lowest-beta industries in 1926-62) were fixed beforehand.

Self-tests run every time: selftest A passed: fast evaluation equals rolling_dca (1e-9); signals at rows<=m* unchanged by scrambling closes from pos[m*] on. selftest B passed (picks unchanged by scrambling the future; chained track == DCABacktest, wm 5.0833, irr 12.4240%): picks at Januaries <= 2015-01-02 unchanged when all prices from that date on are scrambled. selftest C passed on 91 Januaries (46 checked for lookahead by scrambling prices after 1981-01-02); track == DCABacktest (wm 4584.5564, irr 12.6363%; note src.dca.dca_metrics returns NaN irr here: its +-8 bracket overflows exp() on a 90-year horizon); windows == rolling_dca.

---
## A. Pre-registered out-of-sample replication

Frozen: mom12, contra12, mom6, contra6, z_mean_revert (36m), dd20_dip, dd20_flee, trend_A_above_sma200, trend_rev; hard 100/0; signals identical to exp3 (verified: identical to `exp3.make_signals` on real data); monthly DCA through the same engine arithmetic as `rolling_dca` (verified to 1e-9). Pairs: five century US pairs (FF_TECH/FF_VALUE, FF_TECH/FF_MKT, FF_MKT/FF_VALUE, FF_GROWTH/FF_VALUE, FF_TECH/defensive mix), all 66 pairs of the 12 industries (A = more volatile in 1926-62), FF_MKT/FF_DEVXUS (1990+), VFINX/PRITX (1980+). Windows: rolling monthly starts, decisions begin 37 months after data start (signal warm-up). Rolling windows per century pair: 1,106 / 1,046 / 926 for 5y / 10y / 20y; FF_MKT/FF_DEVXUS and VFINX/PRITX at 10y: 278 and 402.

Two controls are reported. **Window-matched**: the exp3 control (fixed mix with the window's own average A-fraction). **Global**: one fixed mix with the rule's full-sample average A-fraction, which has no in-window alignment bias. Ratios are medians over windows; >1 means the rule ended with more wealth.

Significance, three references: (a) circular-shift placebo (signal books rotated/time-reversed by at least 48 months against prices, common shift for all pairs so cross-pair dependence is kept; the cross-pair median is recomputed per shift); (b) block bootstrap of history (mean block 18 months); (c) iid-month resampling and a pure iid synthetic world (the "mechanics" benchmark). Only (c) contains the mechanical effects, which is why (a) wrongly flags everything.

### A1. 66-pair panel, 10-year windows (median over pairs of per-pair medians)

Window-matched control (round-1 yardstick):

| rule | median | q25 | q75 | share>1 | p (placebo) | boot_p5 | boot_p95 | iid_p5 | iid_p95 | nonov_median | nonov_share_pairs_gt1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom12 | 0.970 | 0.966 | 0.974 | 0.000 | 0.001 | 0.967 | 0.973 | 0.968 | 0.972 | 0.970 | 0.000 |
| contra12 | 1.031 | 1.027 | 1.035 | 1.000 | 0.001 | 1.028 | 1.034 | 1.029 | 1.033 | 1.031 | 1.000 |
| mom6 | 0.978 | 0.975 | 0.983 | 0.000 | 0.001 | 0.976 | 0.980 | 0.976 | 0.979 | 0.977 | 0.000 |
| contra6 | 1.023 | 1.018 | 1.025 | 1.000 | 0.001 | 1.021 | 1.025 | 1.021 | 1.024 | 1.023 | 1.000 |
| z_mean_revert | 1.032 | 1.028 | 1.037 | 1.000 | 0.001 | 1.030 | 1.036 | 1.029 | 1.035 | 1.033 | 1.000 |
| dd20_dip | 1.008 | 1.003 | 1.014 | 0.939 | 0.001 | 1.003 | 1.019 | 1.008 | 1.022 | 1.010 | 0.939 |
| dd20_flee | 0.992 | 0.986 | 0.997 | 0.015 | 0.001 | 0.981 | 0.997 | 0.978 | 0.992 | 0.990 | 0.061 |
| trend_A_above_sma200 | 0.989 | 0.985 | 0.993 | 0.030 | 0.001 | 0.984 | 0.991 | 0.985 | 0.990 | 0.989 | 0.045 |
| trend_rev | 1.011 | 1.007 | 1.015 | 0.970 | 0.001 | 1.009 | 1.016 | 1.010 | 1.015 | 1.011 | 0.955 |

Global-average control:

| rule | median | q25 | q75 | share_pairst1 | p_shift_median | boot_p5 | boot_p95 | iid_p5 | iid_p95 | nonov_median | nonov_share_pairst1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom12 | 0.982 | 0.977 | 0.985 | 0.045 | 0.011 | 0.979 | 0.987 | 0.979 | 0.987 | 0.984 | 0.061 |
| contra12 | 1.018 | 1.015 | 1.024 | 0.955 | 0.011 | 1.013 | 1.021 | 1.013 | 1.021 | 1.016 | 0.939 |
| mom6 | 0.988 | 0.985 | 0.992 | 0.045 | 0.008 | 0.985 | 0.991 | 0.985 | 0.991 | 0.988 | 0.061 |
| contra6 | 1.013 | 1.008 | 1.015 | 0.955 | 0.008 | 1.009 | 1.015 | 1.009 | 1.015 | 1.012 | 0.939 |
| z_mean_revert | 1.020 | 1.017 | 1.025 | 0.985 | 0.012 | 1.013 | 1.022 | 1.013 | 1.022 | 1.019 | 0.985 |
| dd20_dip | 1.016 | 1.002 | 1.023 | 0.818 | 0.001 | 0.998 | 1.022 | 0.999 | 1.020 | 1.012 | 0.818 |
| dd20_flee | 0.984 | 0.976 | 0.998 | 0.182 | 0.001 | 0.978 | 1.002 | 0.980 | 1.001 | 0.987 | 0.182 |
| trend_A_above_sma200 | 0.993 | 0.988 | 1.000 | 0.258 | 0.008 | 0.989 | 0.998 | 0.990 | 0.997 | 0.993 | 0.303 |
| trend_rev | 1.007 | 1.000 | 1.012 | 0.742 | 0.008 | 1.002 | 1.011 | 1.003 | 1.010 | 1.007 | 0.697 |

`q25/q75` = inter-quartile range across the 66 pairs; `share>1` = share of pairs with median ratio above 1; `p (placebo)` = circular-shift placebo, two-sided (min possible 0.001; the shifted-signal null does not contain the mechanical effect); `boot_p5/p95` = block-bootstrap 90% interval of the cross-pair median; `iid_p5/p95` = iid-resampled-month 90% interval of the cross-pair median; `nonov_*` = non-overlapping windows (average over all 120 start phases; about 8-9 independent 10y windows per pair, and the 66 pairs are not independent). The real values sit inside the iid and bootstrap intervals for every rule except where noted below; nothing is distinguishable from the no-predictability world.

Pure synthetic check (two identical iid assets, 25% vol, corr 0.8, 300 paths of 97 years), 10y windows:

| rule | global | matched |
|---|---|---|
| mom12 | 0.984 | 0.971 |
| contra12 | 1.016 | 1.030 |
| mom6 | 0.989 | 0.979 |
| contra6 | 1.011 | 1.022 |
| z_mean_revert | 1.017 | 1.031 |
| dd20_dip | 1.006 | 1.008 |
| dd20_flee | 0.994 | 0.992 |
| trend_A_above_sma200 | 0.996 | 0.992 |
| trend_rev | 1.004 | 1.008 |

Compare with the real panel rows above (window-matched 0.970 / 1.031 / 0.978 / 1.023 / 1.032 / 1.008 / 0.992 / 0.989 / 1.011). The synthetic world has no momentum, no reversal and no mean difference, yet reproduces every number.

Era split (windows by start date; "end<1980" = windows entirely before round 1's data), global control, 10y:

| rule | start<1963 | start1963-98 | start>=1999 | end<1980 |
|---|---|---|---|---|
| mom12 | 0.981 | 0.981 | 0.986 | 0.980 |
| contra12 | 1.020 | 1.019 | 1.014 | 1.020 |
| mom6 | 0.987 | 0.986 | 0.991 | 0.987 |
| contra6 | 1.013 | 1.014 | 1.010 | 1.013 |
| z_mean_revert | 1.022 | 1.022 | 1.011 | 1.023 |
| dd20_dip | 1.011 | 1.020 | 1.003 | 1.011 |
| dd20_flee | 0.989 | 0.980 | 0.997 | 0.989 |
| trend_A_above_sma200 | 0.992 | 0.994 | 0.994 | 0.993 |
| trend_rev | 1.009 | 1.006 | 1.006 | 1.007 |

Window-matched control:

| rule | start<1963 | start1963-98 | start>=1999 | end<1980 |
|---|---|---|---|---|
| mom12 | 0.968 | 0.971 | 0.973 | 0.970 |
| contra12 | 1.032 | 1.030 | 1.028 | 1.031 |
| mom6 | 0.978 | 0.978 | 0.982 | 0.978 |
| contra6 | 1.022 | 1.022 | 1.019 | 1.022 |
| z_mean_revert | 1.033 | 1.033 | 1.029 | 1.033 |
| dd20_dip | 1.005 | 1.011 | 1.012 | 1.007 |
| dd20_flee | 0.995 | 0.989 | 0.988 | 0.993 |
| trend_A_above_sma200 | 0.987 | 0.992 | 0.987 | 0.987 |
| trend_rev | 1.014 | 1.008 | 1.013 | 1.013 |

The sign and size are the same in 1926-62, 1963-98 and 1999-2026 and in windows that end before 1980: the pattern does not depend on the 2000-02 episode (except dd20_dip, which is smallest in 1999+ on the global control, and was 1.005-1.012 on the matched control in all eras). 5y and 20y panel results:

| rule | median (matched) | share>1 (matched) | median (global) | share>1 (global) | iid p5 (global) | iid p95 (global) |
|---|---|---|---|---|---|---|
| mom12 | 0.975 | 0.000 | 0.986 | 0.045 | 0.983 | 0.989 |
| contra12 | 1.026 | 1.000 | 1.014 | 0.955 | 1.011 | 1.017 |
| mom6 | 0.981 | 0.000 | 0.989 | 0.045 | 0.987 | 0.991 |
| contra6 | 1.020 | 1.000 | 1.011 | 0.955 | 1.009 | 1.013 |
| z_mean_revert | 1.024 | 1.000 | 1.014 | 0.985 | 1.010 | 1.016 |
| dd20_dip | 1.002 | 0.848 | 1.006 | 0.773 | 1.001 | 1.013 |
| dd20_flee | 0.998 | 0.000 | 0.994 | 0.227 | 0.987 | 0.999 |
| trend_A_above_sma200 | 0.990 | 0.015 | 0.995 | 0.136 | 0.992 | 0.997 |
| trend_rev | 1.010 | 0.985 | 1.005 | 0.864 | 1.003 | 1.008 |

| rule | median (matched) | share>1 (matched) | median (global) | share>1 (global) | iid p5 (global) | iid p95 (global) |
|---|---|---|---|---|---|---|
| mom12 | 0.967 | 0.000 | 0.976 | 0.030 | 0.974 | 0.986 |
| contra12 | 1.034 | 1.000 | 1.024 | 0.970 | 1.014 | 1.027 |
| mom6 | 0.976 | 0.000 | 0.985 | 0.030 | 0.981 | 0.990 |
| contra6 | 1.024 | 1.000 | 1.016 | 0.970 | 1.011 | 1.019 |
| z_mean_revert | 1.039 | 1.000 | 1.027 | 0.970 | 1.015 | 1.029 |
| dd20_dip | 1.014 | 0.939 | 1.024 | 0.909 | 0.997 | 1.034 |
| dd20_flee | 0.986 | 0.061 | 0.974 | 0.091 | 0.966 | 1.003 |
| trend_A_above_sma200 | 0.989 | 0.076 | 0.993 | 0.303 | 0.986 | 0.997 |
| trend_rev | 1.011 | 0.924 | 1.008 | 0.697 | 1.003 | 1.014 |

At 20 years the effects roughly double for the contrarian and drawdown rules, again matching the iid worlds (`a_panel_summary.csv`).

### A2. Named pairs and international pairs, 10y windows, global control (ratio to a fixed mix; >1 favours the rule)

| pair | rolling 10y windows |
|---|---|
| FF_TECH/FF_VALUE | 1046 |
| FF_TECH/FF_MKT | 1046 |
| FF_MKT/FF_VALUE | 1046 |
| FF_GROWTH/FF_VALUE | 1046 |
| FF_TECH/DEF | 1046 |
| FF_MKT/FF_DEVXUS | 278 |
| VFINX/PRITX | 402 |

| rule | FF_GROWTH/FF_VALUE | FF_MKT/FF_DEVXUS | FF_MKT/FF_VALUE | FF_TECH/DEF | FF_TECH/FF_MKT | FF_TECH/FF_VALUE | VFINX/PRITX |
|---|---|---|---|---|---|---|---|
| mom12 | 0.999 | 1.021 | 0.997 | 0.984 | 0.989 | 0.996 | 1.013 |
| contra12 | 1.001 | 0.981 | 1.003 | 1.016 | 1.012 | 1.004 | 0.986 |
| mom6 | 1.000 | 1.017 | 0.997 | 0.992 | 0.993 | 0.997 | 0.995 |
| contra6 | 1.000 | 0.984 | 1.003 | 1.008 | 1.007 | 1.003 | 1.005 |
| z_mean_revert | 1.006 | 0.982 | 1.001 | 1.018 | 1.011 | 1.006 | 1.001 |
| dd20_dip | 1.008 | 0.940 | 1.005 | 1.023 | 1.012 | 1.013 | 0.952 |
| dd20_flee | 0.992 | 1.052 | 0.994 | 0.977 | 0.988 | 0.987 | 1.041 |
| trend_A_above_sma200 | 1.006 | 1.027 | 1.006 | 0.995 | 0.999 | 1.006 | 1.017 |
| trend_rev | 0.994 | 0.969 | 0.994 | 1.005 | 1.001 | 0.994 | 0.979 |

Same, window-matched control:

| rule | FF_GROWTH/FF_VALUE | FF_MKT/FF_DEVXUS | FF_MKT/FF_VALUE | FF_TECH/DEF | FF_TECH/FF_MKT | FF_TECH/FF_VALUE | VFINX/PRITX |
|---|---|---|---|---|---|---|---|
| mom12 | 0.981 | 0.988 | 0.988 | 0.971 | 0.981 | 0.973 | 0.978 |
| contra12 | 1.019 | 1.014 | 1.012 | 1.030 | 1.020 | 1.027 | 1.024 |
| mom6 | 0.987 | 0.983 | 0.989 | 0.983 | 0.986 | 0.979 | 0.978 |
| contra6 | 1.013 | 1.019 | 1.011 | 1.017 | 1.015 | 1.021 | 1.024 |
| z_mean_revert | 1.021 | 1.021 | 1.014 | 1.031 | 1.022 | 1.028 | 1.029 |
| dd20_dip | 1.000 | 0.991 | 0.999 | 1.020 | 1.011 | 1.007 | 0.998 |
| dd20_flee | 1.000 | 1.006 | 1.001 | 0.980 | 0.989 | 0.993 | 1.001 |
| trend_A_above_sma200 | 1.003 | 1.012 | 1.005 | 0.985 | 0.990 | 0.993 | 1.001 |
| trend_rev | 0.997 | 0.987 | 0.995 | 1.016 | 1.010 | 1.008 | 0.999 |

Circular-shift p-values (global control; shift null, so NOT the mechanical benchmark):

| rule | FF_GROWTH/FF_VALUE | FF_MKT/FF_DEVXUS | FF_MKT/FF_VALUE | FF_TECH/DEF | FF_TECH/FF_MKT | FF_TECH/FF_VALUE | VFINX/PRITX |
|---|---|---|---|---|---|---|---|
| mom12 | 0.994 | 0.715 | 0.779 | 0.158 | 0.210 | 0.851 | 0.783 |
| contra12 | 0.990 | 0.807 | 0.777 | 0.159 | 0.206 | 0.853 | 0.776 |
| mom6 | 0.940 | 0.754 | 0.522 | 0.371 | 0.319 | 0.834 | 0.937 |
| contra6 | 0.940 | 0.813 | 0.521 | 0.365 | 0.319 | 0.840 | 0.930 |
| z_mean_revert | 0.728 | 0.797 | 0.963 | 0.115 | 0.214 | 0.798 | 0.986 |
| dd20_dip | 0.948 | 0.226 | 0.954 | 0.161 | 0.354 | 0.713 | 0.002 |
| dd20_flee | 0.966 | 0.187 | 0.938 | 0.164 | 0.360 | 0.705 | 0.002 |
| trend_A_above_sma200 | 0.167 | 0.118 | 0.035 | 0.502 | 0.898 | 0.565 | 0.026 |
| trend_rev | 0.154 | 0.105 | 0.034 | 0.516 | 0.891 | 0.584 | 0.016 |

Non-overlapping 10y windows (mean over start phases; only 3-9 windows per pair, so about 3 for the international pairs):

| rule | FF_GROWTH/FF_VALUE | FF_MKT/FF_DEVXUS | FF_MKT/FF_VALUE | FF_TECH/DEF | FF_TECH/FF_MKT | FF_TECH/FF_VALUE | VFINX/PRITX |
|---|---|---|---|---|---|---|---|
| mom12 | 0.998 | 1.022 | 0.997 | 0.984 | 0.992 | 1.011 | 1.003 |
| contra12 | 1.002 | 0.977 | 1.003 | 1.016 | 1.008 | 0.989 | 0.997 |
| mom6 | 0.998 | 1.015 | 0.996 | 0.993 | 0.995 | 1.004 | 0.992 |
| contra6 | 1.002 | 0.985 | 1.004 | 1.007 | 1.005 | 0.996 | 1.008 |
| z_mean_revert | 1.005 | 0.981 | 1.002 | 1.019 | 1.009 | 0.989 | 1.011 |
| dd20_dip | 1.006 | 0.953 | 1.004 | 1.017 | 1.009 | 1.012 | 0.958 |
| dd20_flee | 0.993 | 1.049 | 0.996 | 0.983 | 0.991 | 0.988 | 1.032 |
| trend_A_above_sma200 | 1.008 | 1.028 | 1.005 | 0.998 | 1.001 | 1.010 | 1.013 |
| trend_rev | 0.993 | 0.972 | 0.995 | 1.001 | 0.999 | 0.989 | 0.985 |

Pairs with low relative volatility (FF_GROWTH/FF_VALUE, FF_MKT/FF_VALUE) show about zero on the global control, because both the rebalancing-type effect and the alignment bias scale with the relative variance of the two assets. The strongest named result, FF_TECH vs the defensive mix, is +1.6% for contra12, which is what the iid world gives for a pair with that relative volatility (`a_pairs_named_intl.csv`, column `iid_mean_g`). In the two international pairs the momentum and drawdown signs are the opposite of the US industry pattern (e.g. dd20_dip -6.0% and -4.8%): with only 278 and 402 overlapping windows (about 3 independent decades) these are not stable findings either way, but they do show the US-industry pattern is not universal.

### A3. Round-1 pairs against their own iid benchmark (same rules, no proxy warm-up, so numbers differ slightly from exp3), 10y:

| pair | rule | med | iid_mean | med_g | iid_mean_g | boot_p5_g | boot_p95_g |
|---|---|---|---|---|---|---|---|
| VFINX/BRK-A | mom12 | 0.982 | 0.965 | 0.986 | 0.980 | 0.959 | 1.006 |
| VFINX/BRK-A | contra12 | 1.019 | 1.039 | 1.014 | 1.021 | 0.993 | 1.046 |
| VFINX/BRK-A | mom6 | 0.984 | 0.972 | 0.986 | 0.984 | 0.968 | 1.003 |
| VFINX/BRK-A | contra6 | 1.016 | 1.030 | 1.014 | 1.017 | 0.997 | 1.034 |
| VFINX/BRK-A | z_mean_revert | 1.020 | 1.040 | 1.017 | 1.019 | 0.982 | 1.046 |
| VFINX/BRK-A | dd20_dip | 1.021 | 1.002 | 1.027 | 1.010 | 0.985 | 1.054 |
| VFINX/BRK-A | dd20_flee | 0.979 | 0.997 | 0.971 | 0.987 | 0.932 | 1.016 |
| VFINX/BRK-A | trend_A_above_sma200 | 0.989 | 0.997 | 0.994 | 0.999 | 0.968 | 1.017 |
| VFINX/BRK-A | trend_rev | 1.011 | 1.002 | 1.006 | 1.001 | 0.986 | 1.027 |
| ^NDX/BRK-A | mom12 | 0.966 | 0.948 | 0.974 | 0.970 | 0.942 | 1.005 |
| ^NDX/BRK-A | contra12 | 1.037 | 1.056 | 1.026 | 1.031 | 0.995 | 1.059 |
| ^NDX/BRK-A | mom6 | 0.973 | 0.961 | 0.980 | 0.979 | 0.962 | 1.000 |
| ^NDX/BRK-A | contra6 | 1.028 | 1.040 | 1.020 | 1.021 | 1.000 | 1.038 |
| ^NDX/BRK-A | z_mean_revert | 1.037 | 1.060 | 1.028 | 1.032 | 0.988 | 1.060 |
| ^NDX/BRK-A | dd20_dip | 1.046 | 1.036 | 1.057 | 1.020 | 0.936 | 1.096 |
| ^NDX/BRK-A | dd20_flee | 0.958 | 0.965 | 0.943 | 0.978 | 0.901 | 1.063 |
| ^NDX/BRK-A | trend_A_above_sma200 | 0.971 | 0.975 | 0.995 | 0.986 | 0.965 | 1.019 |
| ^NDX/BRK-A | trend_rev | 1.028 | 1.025 | 1.005 | 1.014 | 0.981 | 1.033 |
| QQQ/BRK-B | mom12 | 0.974 | 0.948 | 0.992 | 0.966 | 0.930 | 1.019 |
| QQQ/BRK-B | contra12 | 1.033 | 1.056 | 1.008 | 1.035 | 0.981 | 1.071 |
| QQQ/BRK-B | mom6 | 0.974 | 0.961 | 0.975 | 0.975 | 0.957 | 1.009 |
| QQQ/BRK-B | contra6 | 1.027 | 1.041 | 1.026 | 1.025 | 0.990 | 1.045 |
| QQQ/BRK-B | z_mean_revert | 1.035 | 1.061 | 1.007 | 1.038 | 0.977 | 1.077 |
| QQQ/BRK-B | dd20_dip | 1.037 | 1.052 | 1.063 | 1.037 | 0.942 | 1.160 |
| QQQ/BRK-B | dd20_flee | 0.963 | 0.950 | 0.937 | 0.963 | 0.842 | 1.049 |
| QQQ/BRK-B | trend_A_above_sma200 | 0.989 | 0.970 | 0.994 | 0.981 | 0.947 | 1.027 |
| QQQ/BRK-B | trend_rev | 1.015 | 1.032 | 1.006 | 1.020 | 0.972 | 1.054 |
| SPY/BRK-B | mom12 | 0.982 | 0.966 | 0.983 | 0.980 | 0.961 | 1.001 |
| SPY/BRK-B | contra12 | 1.018 | 1.035 | 1.017 | 1.020 | 0.999 | 1.040 |
| SPY/BRK-B | mom6 | 0.985 | 0.975 | 0.986 | 0.985 | 0.974 | 1.001 |
| SPY/BRK-B | contra6 | 1.015 | 1.026 | 1.014 | 1.015 | 0.999 | 1.027 |
| SPY/BRK-B | z_mean_revert | 1.019 | 1.037 | 1.018 | 1.021 | 0.995 | 1.042 |
| SPY/BRK-B | dd20_dip | 1.023 | 1.008 | 1.012 | 1.004 | 0.973 | 1.056 |
| SPY/BRK-B | dd20_flee | 0.977 | 0.992 | 0.988 | 0.994 | 0.942 | 1.027 |
| SPY/BRK-B | trend_A_above_sma200 | 0.990 | 0.992 | 0.994 | 0.997 | 0.968 | 1.012 |
| SPY/BRK-B | trend_rev | 1.011 | 1.008 | 1.006 | 1.003 | 0.988 | 1.032 |

`iid_mean` is the mean median-ratio over iid-resampled worlds built from that pair's own history. On the window-matched control the real contra12/z_mean_revert numbers (e.g. VFINX/BRK-A 1.019 / 1.020) are BELOW the iid-world values (1.039 / 1.040): round 1's headline gains were not larger than what no predictability would give.

### A4. What the bootstrap and placebo can and cannot say
The bootstrap and iid resampling reshuffle the one history we have: they keep each asset's average return gap (e.g. tech's premium) as a permanent feature, cannot create regimes that never happened, and destroy mean reversion longer than the block length (18 months). A test of multi-year mean reversion between 3 and 10 years would need independent histories; the 100-year French sample has about 10 non-overlapping decades. What the test does show is that the 12-month-signal effects found in round 1 do not need any such mean reversion to exist.

---
## B. Walk-forward rule selection

Every January, 53 candidates (48 exp3 rule variants plus 5 fixed mixes) are scored by the DCA wealth of the trailing window (expanding, rolling 10y, rolling 20y; contributions on rows strictly before the decision day, marked at the previous close), the top one is used for the next 12 contributions. All three schemes start at the same first January with 20 years of history (FF pairs 1950, VFINX/BRK-A 2004, ^NDX/BRK-A 2009 (price index), so the real out-of-sample spans are 77, 77, 23 and 18 years). A 10-year-start variant is in `b_walkforward_tracks.csv` (`universe = start_10y`).

### B1. Out-of-sample chained tracks (terminal wealth per $ contributed, marked at the last close; IRR in %)

| pair | oos_start | strategy | terminal wealth / $ | IRR % | avg A-frac |
|---|---|---|---|---|---|
| FF_TECH/FF_VALUE | 1950-01-03 | WF expanding | 947.72 | 12.51 | 0.25 |
| FF_TECH/FF_VALUE | 1950-01-03 | WF roll10y | 924.54 | 12.47 | 0.46 |
| FF_TECH/FF_VALUE | 1950-01-03 | WF roll20y | 922.32 | 12.47 | 0.42 |
| FF_TECH/FF_VALUE | 1950-01-03 | fixed 50/50 | 1013.15 | 12.62 | 0.50 |
| FF_TECH/FF_VALUE | 1950-01-03 | matched fixed (expanding avg A-fraction) | 1085.31 | 12.73 | 0.25 |
| FF_TECH/FF_VALUE | 1950-01-03 | matched fixed (roll10y avg A-fraction) | 1025.13 | 12.64 | 0.46 |
| FF_TECH/FF_VALUE | 1950-01-03 | matched fixed (roll20y avg A-fraction) | 1036.15 | 12.66 | 0.42 |
| FF_TECH/FF_VALUE | 1950-01-03 | always contra12 hard | 1062.94 | 12.70 | 0.50 |
| FF_TECH/FF_VALUE | 1950-01-03 | always mom12 hard | 963.36 | 12.54 | 0.50 |
| FF_TECH/FF_VALUE | 1950-01-03 | hindsight-best of 53 candidates in OOS span (fixed_0A_100B) | 1158.26 | 12.84 | 0.00 |
| FF_TECH/FF_VALUE | 1950-01-03 | median of 53 candidates in OOS span | 1013.15 | 12.62 |  |
| FF_TECH/FF_VALUE | 1950-01-03 | worst of 53 candidates in OOS span | 868.04 | 12.37 |  |
| FF_MKT/FF_VALUE | 1950-01-03 | WF expanding | 1158.26 | 12.84 | 0.00 |
| FF_MKT/FF_VALUE | 1950-01-03 | WF roll10y | 1099.03 | 12.75 | 0.24 |
| FF_MKT/FF_VALUE | 1950-01-03 | WF roll20y | 1136.81 | 12.81 | 0.12 |
| FF_MKT/FF_VALUE | 1950-01-03 | fixed 50/50 | 789.23 | 12.21 | 0.50 |
| FF_MKT/FF_VALUE | 1950-01-03 | matched fixed (expanding avg A-fraction) | 1158.26 | 12.84 | 0.00 |
| FF_MKT/FF_VALUE | 1950-01-03 | matched fixed (roll10y avg A-fraction) | 984.36 | 12.57 | 0.24 |
| FF_MKT/FF_VALUE | 1950-01-03 | matched fixed (roll20y avg A-fraction) | 1073.32 | 12.71 | 0.12 |
| FF_MKT/FF_VALUE | 1950-01-03 | always contra12 hard | 692.47 | 11.99 | 0.60 |
| FF_MKT/FF_VALUE | 1950-01-03 | always mom12 hard | 885.99 | 12.40 | 0.40 |
| FF_MKT/FF_VALUE | 1950-01-03 | hindsight-best of 53 candidates in OOS span (fixed_0A_100B) | 1158.26 | 12.84 | 0.00 |
| FF_MKT/FF_VALUE | 1950-01-03 | median of 53 candidates in OOS span | 789.23 | 12.21 |  |
| FF_MKT/FF_VALUE | 1950-01-03 | worst of 53 candidates in OOS span | 420.19 | 11.17 |  |
| VFINX/BRK-A | 2004-01-02 | WF expanding | 5.11 | 12.46 | 0.00 |
| VFINX/BRK-A | 2004-01-02 | WF roll10y | 5.08 | 12.42 | 0.26 |
| VFINX/BRK-A | 2004-01-02 | WF roll20y | 5.10 | 12.44 | 0.18 |
| VFINX/BRK-A | 2004-01-02 | fixed 50/50 | 5.14 | 12.50 | 0.50 |
| VFINX/BRK-A | 2004-01-02 | matched fixed (expanding avg A-fraction) | 5.11 | 12.46 | 0.00 |
| VFINX/BRK-A | 2004-01-02 | matched fixed (roll10y avg A-fraction) | 5.13 | 12.48 | 0.26 |
| VFINX/BRK-A | 2004-01-02 | matched fixed (roll20y avg A-fraction) | 5.12 | 12.47 | 0.18 |
| VFINX/BRK-A | 2004-01-02 | always contra12 hard | 5.25 | 12.65 | 0.51 |
| VFINX/BRK-A | 2004-01-02 | always mom12 hard | 5.03 | 12.35 | 0.49 |
| VFINX/BRK-A | 2004-01-02 | hindsight-best of 53 candidates in OOS span (z_mean_revert|hard) | 5.27 | 12.68 | 0.47 |
| VFINX/BRK-A | 2004-01-02 | median of 53 candidates in OOS span | 5.14 | 12.50 |  |
| VFINX/BRK-A | 2004-01-02 | worst of 53 candidates in OOS span | 5.01 | 12.32 |  |
| ^NDX/BRK-A | 2009-01-02 | WF expanding | 5.05 | 16.11 | 0.49 |
| ^NDX/BRK-A | 2009-01-02 | WF roll10y | 5.38 | 16.68 | 0.75 |
| ^NDX/BRK-A | 2009-01-02 | WF roll20y | 5.06 | 16.13 | 0.53 |
| ^NDX/BRK-A | 2009-01-02 | fixed 50/50 | 5.13 | 16.24 | 0.50 |
| ^NDX/BRK-A | 2009-01-02 | matched fixed (expanding avg A-fraction) | 5.11 | 16.22 | 0.49 |
| ^NDX/BRK-A | 2009-01-02 | matched fixed (roll10y avg A-fraction) | 5.67 | 17.18 | 0.75 |
| ^NDX/BRK-A | 2009-01-02 | matched fixed (roll20y avg A-fraction) | 5.18 | 16.34 | 0.53 |
| ^NDX/BRK-A | 2009-01-02 | always contra12 hard | 4.74 | 15.52 | 0.32 |
| ^NDX/BRK-A | 2009-01-02 | always mom12 hard | 5.51 | 16.91 | 0.68 |
| ^NDX/BRK-A | 2009-01-02 | hindsight-best of 53 candidates in OOS span (fixed_100A) | 6.23 | 18.03 | 1.00 |
| ^NDX/BRK-A | 2009-01-02 | median of 53 candidates in OOS span | 5.13 | 16.24 |  |
| ^NDX/BRK-A | 2009-01-02 | worst of 53 candidates in OOS span | 4.03 | 14.00 |  |

"hindsight-best" = the best of the 53 candidates over the same OOS span, i.e. what the in-sample best rule scores: it is always at or above every walk-forward number, and the gap is the cost of having to choose in advance (IRR gaps are small, 0.1-0.4 pp, because the candidates differ little; they are large only in wealth terms over decades).

### B2. Rolling 10y windows inside the OOS track (ratio of walk-forward wealth to fixed 50/50)

| pair | scheme | n_windows | med_vs_50_50 | share_gt1_vs_50_50 | nonov_med_vs_50_50 | nonov_n |
|---|---|---|---|---|---|---|
| FF_TECH/FF_VALUE | expanding | 801 | 0.965 | 0.395 | 0.969 | 7 |
| FF_TECH/FF_VALUE | roll10y | 801 | 0.992 | 0.473 | 1.002 | 7 |
| FF_TECH/FF_VALUE | roll20y | 801 | 0.978 | 0.419 | 0.981 | 7 |
| FF_MKT/FF_VALUE | expanding | 801 | 1.056 | 0.725 | 1.056 | 7 |
| FF_MKT/FF_VALUE | roll10y | 801 | 1.034 | 0.769 | 1.036 | 7 |
| FF_MKT/FF_VALUE | roll20y | 801 | 1.053 | 0.724 | 1.051 | 7 |
| VFINX/BRK-A | expanding | 155 | 0.989 | 0.387 | 0.981 | 2 |
| VFINX/BRK-A | roll10y | 155 | 0.981 | 0.155 | 0.977 | 2 |
| VFINX/BRK-A | roll20y | 155 | 0.988 | 0.290 | 0.979 | 2 |
| ^NDX/BRK-A | expanding | 95 | 0.951 | 0.242 | 0.948 | 1 |
| ^NDX/BRK-A | roll10y | 95 | 1.033 | 0.768 | 1.047 | 1 |
| ^NDX/BRK-A | roll20y | 95 | 0.951 | 0.242 | 0.950 | 1 |

### B3. What gets picked

| pair | scheme | n_Jan | distinct_rules | fixed_mix | momentum_type | contrarian_type | top3 |
|---|---|---|---|---|---|---|---|
| FF_MKT/FF_VALUE | expanding | 77 | 1 | 100% | 0% | 0% | fixed_0A_100B 100% |
| FF_MKT/FF_VALUE | roll10y | 77 | 10 | 69% | 0% | 9% | fixed_0A_100B 62%, z_mean_revert|hard 8%, fixed_100A 6% |
| FF_MKT/FF_VALUE | roll20y | 77 | 6 | 90% | 0% | 0% | fixed_0A_100B 79%, fixed_100A 10%, dd30_dip|hard 4% |
| FF_TECH/FF_VALUE | expanding | 77 | 5 | 83% | 0% | 1% | fixed_0A_100B 74%, dd30_flee|hard 14%, fixed_100A 9% |
| FF_TECH/FF_VALUE | roll10y | 77 | 10 | 73% | 0% | 4% | fixed_0A_100B 45%, fixed_100A 27%, dd30_flee|hard 5% |
| FF_TECH/FF_VALUE | roll20y | 77 | 8 | 68% | 0% | 1% | fixed_0A_100B 43%, fixed_100A 25%, dd30_dip|hard 14% |
| VFINX/BRK-A | expanding | 23 | 1 | 100% | 0% | 0% | fixed_0A_100B 100% |
| VFINX/BRK-A | roll10y | 23 | 8 | 17% | 0% | 9% | dd30_dip|hard 22%, dd20_dip|hard 22%, dd30_flee|hard 17% |
| VFINX/BRK-A | roll20y | 23 | 6 | 61% | 0% | 9% | fixed_0A_100B 48%, dd30_dip|hard 17%, fixed_100A 13% |
| ^NDX/BRK-A | expanding | 18 | 3 | 33% | 0% | 0% | dd30_dip|hard 67%, fixed_100A 28%, fixed_0A_100B 6% |
| ^NDX/BRK-A | roll10y | 18 | 6 | 50% | 0% | 0% | fixed_100A 44%, dd20_dip_else50|hard 22%, dd30_dip|hard 11% |
| ^NDX/BRK-A | roll20y | 18 | 4 | 33% | 0% | 0% | dd30_dip|hard 61%, fixed_100A 28%, fixed_0A_100B 6% |

Most picks are fixed mixes (a pure asset choice, usually "all VALUE" or "all BRK-A", i.e. the long-run winner), not timing rules; momentum-type rules are almost never picked and `dd*_dip` is picked in the NDX and VFINX samples. Expanding selection locks onto one fixed mix for decades (FF_MKT/FF_VALUE and VFINX/BRK-A: 100% fixed_0A_100B). Picks change frequently under rolling windows (6-10 distinct rules in 18-77 selections).

### B4. In-sample to out-of-sample shrinkage (rolling-10y pick; in-sample = trailing 10y DCA wealth vs fixed 50/50; out-of-sample = the same rule over the next 5 years vs fixed 50/50)

| pair | n | picked rule, trailing 10y vs 50/50 | picked rule, next 5y vs 50/50 | median candidate, trailing | median candidate, next 5y |
|---|---|---|---|---|---|
| FF_MKT/FF_VALUE | 72 | 0.073 | 0.016 | -0.000 | 0.000 |
| FF_TECH/FF_VALUE | 72 | 0.134 | -0.001 | -0.000 | 0.000 |
| VFINX/BRK-A | 18 | 0.075 | 0.017 | 0.000 | 0.000 |
| ^NDX/BRK-A | 13 | 0.110 | -0.001 | 0.000 | 0.000 |
| all pairs | 175 | 0.101 | 0.008 | -0.000 | 0.000 |

Picked rules look about +10% better than 50/50 in their training window and are +0.8% better in the following 5 years (+1.6-1.7% in FF_MKT/FF_VALUE and VFINX/BRK-A, -0.1% in the other two); the median candidate shows 0 by construction. See `fig4_wf_rules_shrinkage.png`.

### B5. Bootstrap (resampled histories: full walk-forward re-run on each path)

| pair | scheme | comparison | real_ratio | boot_mean_ratio | boot_p5 | boot_p95 | P_ratio_gt1 | real_pctile_in_boot |
|---|---|---|---|---|---|---|---|---|
| FF_TECH/FF_VALUE | expanding | WF / fixed 50-50 | 0.935 | 0.896 | 0.370 | 1.689 | 0.416 | 49.550 |
| FF_TECH/FF_VALUE | expanding | WF / matched mix | 0.873 | 0.782 | 0.338 | 1.028 | 0.101 | 42.800 |
| FF_TECH/FF_VALUE | roll10y | WF / fixed 50-50 | 0.913 | 0.955 | 0.606 | 1.482 | 0.385 | 39.400 |
| FF_TECH/FF_VALUE | roll10y | WF / matched mix | 0.902 | 0.875 | 0.547 | 1.195 | 0.264 | 45.000 |
| FF_TECH/FF_VALUE | roll20y | WF / fixed 50-50 | 0.910 | 0.925 | 0.471 | 1.598 | 0.403 | 44.600 |
| FF_TECH/FF_VALUE | roll20y | WF / matched mix | 0.890 | 0.820 | 0.410 | 1.174 | 0.281 | 48.350 |
| FF_MKT/FF_VALUE | expanding | WF / fixed 50-50 | 1.468 | 1.214 | 0.700 | 1.690 | 0.777 | 72.050 |
| FF_MKT/FF_VALUE | expanding | WF / matched mix | 1.000 | 0.897 | 0.577 | 1.005 | 0.074 | 53.050 |
| FF_MKT/FF_VALUE | roll10y | WF / fixed 50-50 | 1.393 | 1.121 | 0.801 | 1.582 | 0.691 | 83.950 |
| FF_MKT/FF_VALUE | roll10y | WF / matched mix | 1.116 | 0.939 | 0.662 | 1.150 | 0.413 | 89.600 |
| FF_MKT/FF_VALUE | roll20y | WF / fixed 50-50 | 1.440 | 1.181 | 0.766 | 1.666 | 0.748 | 76.400 |
| FF_MKT/FF_VALUE | roll20y | WF / matched mix | 1.059 | 0.932 | 0.610 | 1.129 | 0.482 | 78.800 |
| VFINX/BRK-A | expanding | WF / fixed 50-50 | 0.994 | 1.373 | 0.917 | 1.800 | 0.915 | 8.000 |
| VFINX/BRK-A | expanding | WF / matched mix | 1.000 | 0.970 | 0.814 | 1.005 | 0.088 | 26.150 |
| VFINX/BRK-A | roll10y | WF / fixed 50-50 | 0.989 | 1.249 | 0.832 | 1.747 | 0.814 | 17.350 |
| VFINX/BRK-A | roll10y | WF / matched mix | 0.992 | 0.947 | 0.705 | 1.086 | 0.433 | 48.550 |
| VFINX/BRK-A | roll20y | WF / fixed 50-50 | 0.991 | 1.346 | 0.902 | 1.794 | 0.883 | 10.950 |
| VFINX/BRK-A | roll20y | WF / matched mix | 0.995 | 0.965 | 0.774 | 1.032 | 0.287 | 31.850 |
| ^NDX/BRK-A | expanding | WF / fixed 50-50 | 0.986 | 0.954 | 0.546 | 1.505 | 0.452 | 52.500 |
| ^NDX/BRK-A | expanding | WF / matched mix | 0.989 | 0.912 | 0.657 | 1.005 | 0.075 | 52.300 |
| ^NDX/BRK-A | roll10y | WF / fixed 50-50 | 1.049 | 0.975 | 0.693 | 1.395 | 0.405 | 67.500 |
| ^NDX/BRK-A | roll10y | WF / matched mix | 0.948 | 0.895 | 0.668 | 1.048 | 0.179 | 56.500 |
| ^NDX/BRK-A | roll20y | WF / fixed 50-50 | 0.987 | 0.960 | 0.586 | 1.481 | 0.434 | 54.300 |
| ^NDX/BRK-A | roll20y | WF / matched mix | 0.977 | 0.890 | 0.623 | 1.020 | 0.144 | 59.250 |

`real_ratio` is the actual OOS terminal-wealth ratio, columns 5-8 describe resampled histories. The resampled worlds retain the full-history asset mean gaps, so asset selection persists by construction (VFINX/BRK-A: P(WF beats 50/50) = 0.8-0.9 in resampled worlds, but the actual result, 0.99, is at the 8th-17th percentile of that distribution): in the real history BRK-A's lead was concentrated before the OOS span, so what was learned did not carry forward. Versus the window-matched mix the resampled ratio is about 0.9 because of the control bias described above, so that comparison is not informative.

---
## C. Walk-forward weights

Universes: U1 VFINX + BRK-A, U2 VFINX + BRK-A + PRITX (both 1990-2026, training from 1980; the first year trains on 113-114 months), U3a FF_MKT + FF_TECH + FF_VALUE (1936-2026), U3b adds FF_DEVXUS (2000-2026; kept separate because it only exists from 1990). Each January the split of the next 12 buy-only contributions is set from the trailing 10 years (excess over FF_RF for Sharpe): winner-take-all (highest trailing DCA wealth), weights proportional to the rank of trailing DCA wealth, long-only max-Sharpe and min-variance (grid search on the simplex), inverse volatility. Naive: 100% first asset (market), equal weight, 70/30 first/second asset. Full-track metrics (wealth multiple, max drawdown) come from `DCABacktest` with a policy that looks up the year's weights; IRR from the same cash flows (computed with a wider-bracket solver, because `src.dca.dca_metrics` returns NaN IRR on 90-year horizons: its +-8 log-rate bracket overflows `exp` beyond ~88 years).

### C1. OOS tracks

| universe | method | months | wealth / $ | IRR % | max DD % | yearly weight change | avg max weight |
|---|---|---|---|---|---|---|---|
| U1 VFINX+BRK-A | winner-take-all | 443 | 19.83 | 12.97 | -51.5 | 0.25 | 1.00 |
| U1 VFINX+BRK-A | rank-proportional | 443 | 16.96 | 12.36 | -51.9 | 0.08 | 0.67 |
| U1 VFINX+BRK-A | max-Sharpe | 443 | 19.39 | 12.88 | -51.6 | 0.13 | 0.76 |
| U1 VFINX+BRK-A | min-variance | 443 | 11.83 | 10.95 | -53.5 | 0.03 | 0.79 |
| U1 VFINX+BRK-A | inverse-vol | 443 | 14.52 | 11.75 | -52.3 | 0.01 | 0.58 |
| U1 VFINX+BRK-A | 100% first asset | 443 | 11.22 | 10.74 | -55.3 | 0.00 | 1.00 |
| U1 VFINX+BRK-A | equal weight | 443 | 15.53 | 12.02 | -52.1 | 0.00 | 0.50 |
| U1 VFINX+BRK-A | 70/30 first/second | 443 | 13.80 | 11.56 | -52.5 | 0.00 | 0.70 |
| U2 VFINX+BRK-A+PRITX | winner-take-all | 443 | 19.47 | 12.90 | -51.5 | 0.33 | 1.00 |
| U2 VFINX+BRK-A+PRITX | rank-proportional | 443 | 13.37 | 11.43 | -52.9 | 0.08 | 0.50 |
| U2 VFINX+BRK-A+PRITX | max-Sharpe | 443 | 17.43 | 12.47 | -52.0 | 0.16 | 0.70 |
| U2 VFINX+BRK-A+PRITX | min-variance | 443 | 8.36 | 9.57 | -56.8 | 0.06 | 0.56 |
| U2 VFINX+BRK-A+PRITX | inverse-vol | 443 | 10.48 | 10.47 | -54.4 | 0.01 | 0.38 |
| U2 VFINX+BRK-A+PRITX | 100% first asset | 443 | 11.22 | 10.74 | -55.3 | 0.00 | 1.00 |
| U2 VFINX+BRK-A+PRITX | equal weight | 443 | 11.49 | 10.84 | -53.7 | 0.00 | 0.33 |
| U2 VFINX+BRK-A+PRITX | 70/30 first/second | 443 | 13.80 | 11.56 | -52.5 | 0.00 | 0.70 |
| U3a FF_MKT+FF_TECH+FF_VALUE | winner-take-all | 1089 | 4540.22 | 12.62 | -65.0 | 0.16 | 1.00 |
| U3a FF_MKT+FF_TECH+FF_VALUE | rank-proportional | 1089 | 4268.64 | 12.54 | -61.5 | 0.06 | 0.50 |
| U3a FF_MKT+FF_TECH+FF_VALUE | max-Sharpe | 1089 | 4584.56 | 12.64 | -64.8 | 0.17 | 0.90 |
| U3a FF_MKT+FF_TECH+FF_VALUE | min-variance | 1089 | 1796.61 | 11.36 | -54.6 | 0.03 | 0.96 |
| U3a FF_MKT+FF_TECH+FF_VALUE | inverse-vol | 1089 | 4387.20 | 12.58 | -62.1 | 0.01 | 0.39 |
| U3a FF_MKT+FF_TECH+FF_VALUE | 100% first asset | 1089 | 1796.10 | 11.36 | -54.6 | 0.00 | 1.00 |
| U3a FF_MKT+FF_TECH+FF_VALUE | equal weight | 1089 | 4541.15 | 12.62 | -62.2 | 0.00 | 0.33 |
| U3a FF_MKT+FF_TECH+FF_VALUE | 70/30 first/second | 1089 | 2556.76 | 11.84 | -66.2 | 0.00 | 0.70 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | winner-take-all | 321 | 7.46 | 12.76 | -82.2 | 0.12 | 1.00 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | rank-proportional | 321 | 6.93 | 12.33 | -63.6 | 0.05 | 0.40 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | max-Sharpe | 321 | 5.62 | 11.12 | -64.1 | 0.25 | 0.91 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | min-variance | 321 | 5.02 | 10.45 | -57.8 | 0.08 | 0.76 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | inverse-vol | 321 | 6.25 | 11.74 | -58.3 | 0.01 | 0.29 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | 100% first asset | 321 | 6.27 | 11.75 | -54.6 | 0.00 | 1.00 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | equal weight | 321 | 6.92 | 12.32 | -58.2 | 0.00 | 0.25 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | 70/30 first/second | 321 | 8.34 | 13.39 | -61.1 | 0.00 | 0.70 |

`yearly weight change` = average over Januaries of half the L1 distance between consecutive weight vectors (0.25 = a quarter of each year's contribution moves to a different asset). `avg max weight` = average of the largest weight. Weights over time: `fig6_wf_weights_over_time.png`; every weight vector: `c_weights_by_january.csv`.

### C2. Rolling OOS windows, 10y (method wealth / naive wealth; non-overlapping = average over start phases, n windows)

| universe | method | vs | n_windows | median_ratio | share_gt1 | nonov_median | nonov_n |
|---|---|---|---|---|---|---|---|
| U1 VFINX+BRK-A | winner-take-all | 100% first asset | 323 | 1.025 | 0.573 | 1.065 | 3 |
| U1 VFINX+BRK-A | winner-take-all | equal weight | 323 | 1.013 | 0.594 | 1.029 | 3 |
| U1 VFINX+BRK-A | rank-proportional | 100% first asset | 323 | 1.025 | 0.619 | 1.047 | 3 |
| U1 VFINX+BRK-A | rank-proportional | equal weight | 323 | 1.004 | 0.594 | 1.010 | 3 |
| U1 VFINX+BRK-A | max-Sharpe | 100% first asset | 323 | 1.013 | 0.591 | 1.023 | 3 |
| U1 VFINX+BRK-A | max-Sharpe | equal weight | 323 | 0.990 | 0.307 | 0.988 | 3 |
| U1 VFINX+BRK-A | min-variance | 100% first asset | 323 | 1.005 | 0.619 | 1.006 | 3 |
| U1 VFINX+BRK-A | min-variance | equal weight | 323 | 0.988 | 0.331 | 0.975 | 3 |
| U1 VFINX+BRK-A | inverse-vol | 100% first asset | 323 | 1.021 | 0.647 | 1.029 | 3 |
| U1 VFINX+BRK-A | inverse-vol | equal weight | 323 | 0.997 | 0.344 | 0.992 | 3 |
| U2 VFINX+BRK-A+PRITX | winner-take-all | 100% first asset | 323 | 1.000 | 0.502 | 1.036 | 3 |
| U2 VFINX+BRK-A+PRITX | winner-take-all | equal weight | 323 | 1.056 | 0.734 | 1.063 | 3 |
| U2 VFINX+BRK-A+PRITX | rank-proportional | 100% first asset | 323 | 0.968 | 0.433 | 0.989 | 3 |
| U2 VFINX+BRK-A+PRITX | rank-proportional | equal weight | 323 | 1.015 | 0.613 | 1.020 | 3 |
| U2 VFINX+BRK-A+PRITX | max-Sharpe | 100% first asset | 323 | 1.004 | 0.517 | 1.008 | 3 |
| U2 VFINX+BRK-A+PRITX | max-Sharpe | equal weight | 323 | 1.035 | 0.740 | 1.035 | 3 |
| U2 VFINX+BRK-A+PRITX | min-variance | 100% first asset | 323 | 0.977 | 0.341 | 0.987 | 3 |
| U2 VFINX+BRK-A+PRITX | min-variance | equal weight | 323 | 1.013 | 0.579 | 1.017 | 3 |
| U2 VFINX+BRK-A+PRITX | inverse-vol | 100% first asset | 323 | 0.952 | 0.347 | 0.971 | 3 |
| U2 VFINX+BRK-A+PRITX | inverse-vol | equal weight | 323 | 1.000 | 0.489 | 0.997 | 3 |
| U3a FF_MKT+FF_TECH+FF_VALUE | winner-take-all | 100% first asset | 969 | 1.092 | 0.752 | 1.101 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | winner-take-all | equal weight | 969 | 1.015 | 0.546 | 1.007 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | rank-proportional | 100% first asset | 969 | 1.079 | 0.832 | 1.085 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | rank-proportional | equal weight | 969 | 1.007 | 0.536 | 1.002 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | max-Sharpe | 100% first asset | 969 | 1.086 | 0.802 | 1.088 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | max-Sharpe | equal weight | 969 | 1.020 | 0.524 | 1.018 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | min-variance | 100% first asset | 969 | 1.000 | 0.467 | 1.000 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | min-variance | equal weight | 969 | 0.933 | 0.240 | 0.932 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | inverse-vol | 100% first asset | 969 | 1.075 | 0.831 | 1.081 | 9 |
| U3a FF_MKT+FF_TECH+FF_VALUE | inverse-vol | equal weight | 969 | 0.996 | 0.374 | 0.996 | 9 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | winner-take-all | 100% first asset | 201 | 0.967 | 0.413 | 1.055 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | winner-take-all | equal weight | 201 | 0.925 | 0.443 | 1.071 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | rank-proportional | 100% first asset | 201 | 1.003 | 0.502 | 1.000 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | rank-proportional | equal weight | 201 | 0.979 | 0.453 | 1.018 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | max-Sharpe | 100% first asset | 201 | 0.920 | 0.358 | 0.977 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | max-Sharpe | equal weight | 201 | 0.939 | 0.313 | 0.991 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | min-variance | 100% first asset | 201 | 0.977 | 0.134 | 0.965 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | min-variance | equal weight | 201 | 0.981 | 0.338 | 0.983 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | inverse-vol | 100% first asset | 201 | 0.967 | 0.308 | 0.971 | 2 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | inverse-vol | equal weight | 201 | 0.991 | 0.199 | 0.989 | 2 |

20-year windows are in `c_rolling_oos_windows.csv` (only 2-9 independent windows exist).

### C3. Paired difference vs naive mixes, bootstrap of resampled histories (full walk-forward re-run per path)

| universe | method | vs | actual wealth ratio | boot 5% | boot 95% | P(ratio>1) | IRR diff pp | IRR diff 5% | IRR diff 95% |
|---|---|---|---|---|---|---|---|---|---|
| U1 VFINX+BRK-A | winner-take-all | 100% first asset | 1.768 | 1.327 | 26.906 | 0.980 | 2.229 | 1.103 | 12.658 |
| U1 VFINX+BRK-A | winner-take-all | equal weight | 1.277 | 0.811 | 1.914 | 0.865 | 0.952 | -0.804 | 2.491 |
| U1 VFINX+BRK-A | rank-proportional | 100% first asset | 1.512 | 1.316 | 19.280 | 0.982 | 1.621 | 1.089 | 11.440 |
| U1 VFINX+BRK-A | rank-proportional | equal weight | 1.092 | 0.937 | 1.305 | 0.865 | 0.345 | -0.250 | 1.019 |
| U1 VFINX+BRK-A | max-Sharpe | 100% first asset | 1.729 | 1.213 | 20.130 | 0.980 | 2.142 | 0.767 | 11.547 |
| U1 VFINX+BRK-A | max-Sharpe | equal weight | 1.249 | 0.638 | 1.707 | 0.637 | 0.865 | -1.717 | 2.043 |
| U1 VFINX+BRK-A | min-variance | 100% first asset | 1.054 | 1.103 | 6.786 | 0.985 | 0.210 | 0.384 | 7.391 |
| U1 VFINX+BRK-A | min-variance | equal weight | 0.762 | 0.313 | 0.898 | 0.017 | -1.067 | -4.453 | -0.426 |
| U1 VFINX+BRK-A | inverse-vol | 100% first asset | 1.294 | 1.233 | 13.003 | 0.984 | 1.015 | 0.847 | 9.911 |
| U1 VFINX+BRK-A | inverse-vol | equal weight | 0.935 | 0.791 | 0.972 | 0.018 | -0.262 | -0.892 | -0.109 |
| U2 VFINX+BRK-A+PRITX | winner-take-all | 100% first asset | 1.735 | 1.347 | 29.486 | 0.980 | 2.157 | 1.164 | 12.952 |
| U2 VFINX+BRK-A+PRITX | winner-take-all | equal weight | 1.694 | 0.994 | 2.813 | 0.948 | 2.061 | -0.025 | 3.957 |
| U2 VFINX+BRK-A+PRITX | rank-proportional | 100% first asset | 1.192 | 1.157 | 15.657 | 0.974 | 0.692 | 0.569 | 10.553 |
| U2 VFINX+BRK-A+PRITX | rank-proportional | equal weight | 1.164 | 1.004 | 1.458 | 0.952 | 0.596 | 0.016 | 1.444 |
| U2 VFINX+BRK-A+PRITX | max-Sharpe | 100% first asset | 1.553 | 1.221 | 20.700 | 0.977 | 1.726 | 0.769 | 11.643 |
| U2 VFINX+BRK-A+PRITX | max-Sharpe | equal weight | 1.516 | 0.838 | 2.400 | 0.879 | 1.630 | -0.684 | 3.353 |
| U2 VFINX+BRK-A+PRITX | min-variance | 100% first asset | 0.745 | 0.883 | 7.176 | 0.900 | -1.171 | -0.488 | 7.588 |
| U2 VFINX+BRK-A+PRITX | min-variance | equal weight | 0.727 | 0.348 | 1.059 | 0.096 | -1.267 | -4.036 | 0.226 |
| U2 VFINX+BRK-A+PRITX | inverse-vol | 100% first asset | 0.934 | 0.983 | 9.275 | 0.946 | -0.270 | -0.068 | 8.566 |
| U2 VFINX+BRK-A+PRITX | inverse-vol | equal weight | 0.912 | 0.763 | 0.972 | 0.013 | -0.366 | -1.037 | -0.108 |
| U3a FF_MKT+FF_TECH+FF_VALUE | winner-take-all | 100% first asset | 2.528 | 1.305 | 10.668 | 0.985 | 1.266 | 0.348 | 3.106 |
| U3a FF_MKT+FF_TECH+FF_VALUE | winner-take-all | equal weight | 1.000 | 0.696 | 2.008 | 0.713 | -0.000 | -0.475 | 0.919 |
| U3a FF_MKT+FF_TECH+FF_VALUE | rank-proportional | 100% first asset | 2.377 | 1.365 | 8.034 | 0.993 | 1.182 | 0.416 | 2.735 |
| U3a FF_MKT+FF_TECH+FF_VALUE | rank-proportional | equal weight | 0.940 | 0.863 | 1.268 | 0.628 | -0.085 | -0.192 | 0.310 |
| U3a FF_MKT+FF_TECH+FF_VALUE | max-Sharpe | 100% first asset | 2.553 | 1.281 | 9.448 | 0.991 | 1.280 | 0.329 | 2.947 |
| U3a FF_MKT+FF_TECH+FF_VALUE | max-Sharpe | equal weight | 1.010 | 0.588 | 1.964 | 0.649 | 0.013 | -0.688 | 0.887 |
| U3a FF_MKT+FF_TECH+FF_VALUE | min-variance | 100% first asset | 1.000 | 1.000 | 1.514 | 0.928 | 0.000 | 0.000 | 0.546 |
| U3a FF_MKT+FF_TECH+FF_VALUE | min-variance | equal weight | 0.396 | 0.137 | 0.799 | 0.016 | -1.266 | -2.614 | -0.293 |
| U3a FF_MKT+FF_TECH+FF_VALUE | inverse-vol | 100% first asset | 2.443 | 1.378 | 6.805 | 0.994 | 1.219 | 0.425 | 2.520 |
| U3a FF_MKT+FF_TECH+FF_VALUE | inverse-vol | equal weight | 0.966 | 0.861 | 1.034 | 0.184 | -0.047 | -0.196 | 0.044 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | winner-take-all | 100% first asset | 1.191 | 0.816 | 3.302 | 0.814 | 1.008 | -1.272 | 6.729 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | winner-take-all | equal weight | 1.079 | 0.843 | 2.313 | 0.820 | 0.436 | -1.043 | 4.721 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | rank-proportional | 100% first asset | 1.105 | 0.880 | 1.964 | 0.808 | 0.578 | -0.780 | 3.822 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | rank-proportional | equal weight | 1.001 | 0.949 | 1.304 | 0.824 | 0.006 | -0.307 | 1.502 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | max-Sharpe | 100% first asset | 0.898 | 0.808 | 2.619 | 0.772 | -0.630 | -1.306 | 5.469 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | max-Sharpe | equal weight | 0.813 | 0.790 | 1.925 | 0.716 | -1.202 | -1.375 | 3.751 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | min-variance | 100% first asset | 0.801 | 0.611 | 1.002 | 0.051 | -1.298 | -2.924 | 0.010 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | min-variance | equal weight | 0.726 | 0.453 | 1.003 | 0.054 | -1.870 | -4.531 | 0.022 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | inverse-vol | 100% first asset | 0.998 | 0.830 | 1.447 | 0.613 | -0.011 | -1.139 | 2.100 |
| U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS | inverse-vol | equal weight | 0.904 | 0.860 | 1.010 | 0.095 | -0.583 | -0.864 | 0.066 |

The wealth ratio is the real OOS ratio; the 5-95% interval and P(ratio>1) come from the resampled histories; IRR differences are in percentage points per year (resampled worlds use a 252-day year). As in part B, the resampled worlds keep each asset's long-run mean, so selection-based methods are favoured relative to the real out-of-sample history. Even so, versus equal weight the intervals include 1 for winner-take-all, rank and max-Sharpe in every universe, and min-variance / inverse-vol are below 1 with high probability.

Observations: (1) against "100% first asset" (the index) the selecting methods win (winner / rank / max-Sharpe) by +19% to +155% in U1-U3a (U3b: -10% to +19%), but that mainly says BRK-A / tech / value did better than the index in these histories (hindsight winners are in the universe), because equal weight, which selects nothing, wins nearly as much. (2) Against equal weight, only in U1/U2 (VFINX + BRK-A, 37 years) do winner/max-Sharpe come out ahead (+25-28% in U1, +51-69% in U2 at the end), and 5-95% intervals include 1; in the 90-year U3a they tie (1.00-1.01) and rank-proportional and inverse-vol lose 3-6%; in the 26-year U3b max-Sharpe loses 19%. (3) Max drawdown is not improved by optimisation except min-variance (U3a -54.6% vs -62.2% equal weight, but it holds 100% FF_MKT most years: it is the naive market mix). Winner-take-all concentrates 100% in one asset, hit a -82% drawdown in U3b, and moves 12-33% of the yearly contribution flow between assets.

---
## Caveats
- All French series are frictionless index returns with open = close (trades at the decision day's close, no fees); FF_MKT beat VFINX by about 0.8 pp/yr. Slippage 5 bp per buy is applied to every strategy. ^NDX is a price index without dividends.
- The matched control is retained because the brief required it, but it is biased against momentum rules and for contrarian ones; the global-average control and the iid benchmarks are the fair yardsticks. The same flaw affects round 1's reading of `b_rules_summary.csv` (its "vs 50/50" and "vs best" columns do not have this problem).
- Overlapping windows: the 66 pairs x rolling windows are far from independent; effective independent decades ~10 per pair, 11 independent industry return series. p-values from the shift placebo are not comparable with p-values from the iid benchmark.
- The verdict wording "inside the iid band" means the real value lies within the 5-95% range of iid-resampled worlds: the data cannot distinguish the rule from a no-predictability world; it does not prove there is no predictability (power at 100 years is limited, around 1 pp of relative return).
- Part B/C out-of-sample spans for the 1980+ data are short (18-37 years); the 100-year runs are on French index data.

## Files
Tables: `a_verdicts.csv`, `a_panel_summary.csv`, `a_pairs_named_intl.csv`, `a_pairs_panel66.csv`, `a_round1_pairs_vs_iid.csv`, `a_mechanics_iid_demo.csv`, `b_walkforward_tracks.csv`, `b_pick_frequency.csv`, `b_rolling_oos_windows.csv`, `b_shrinkage_picks.csv`, `b_bootstrap_wf_vs_fixed.csv`, `c_oos_tracks.csv`, `c_weights_by_january.csv`, `c_rolling_oos_windows.csv`, `c_bootstrap_vs_naive.csv`. Figures: `fig1_oos_replication.png` (per rule, 7 new pairs, with intervals), `fig2_panel66_distribution.png`, `fig3_panel_eras.png`, `fig4_wf_rules_shrinkage.png`, `fig5_wf_rules_equity.png`, `fig6_wf_weights_over_time.png`, `fig7_wf_weights_equity.png`, `fig8_wf_weights_bootstrap.png`.
