# EXP7 - Out-of-sample and walk-forward tests of the round-1 conclusions

Script: `scripts/dca/exp7_walk_forward.py` (runs `exp7a_oos_replication.py`, `exp7b_wf_rules.py`, `exp7c_wf_weights.py`; shared code `exp7_common.py`; about 7 minutes on 4 cores). This file is generated from `scripts/dca/exp7_prose.md` plus the CSV tables in this folder. Bootstrap: {{nboot}} paths of a stationary block bootstrap (mean block 18 months) of the joint daily returns of all assets, cut into calendar-month blocks (so cross-asset correlation is kept), fixed seeds; plus {{niid}} paths with iid months (block length 1) used as a "no predictability" benchmark. All French series are frictionless index returns with open = close (trade at the decision day's close; signals use closes up to the previous day).

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

Self-tests run every time: {{msgA}}. {{msgB}}. {{msgC}}.

---
## A. Pre-registered out-of-sample replication

Frozen: mom12, contra12, mom6, contra6, z_mean_revert (36m), dd20_dip, dd20_flee, trend_A_above_sma200, trend_rev; hard 100/0; signals identical to exp3 (verified: identical to `exp3.make_signals` on real data); monthly DCA through the same engine arithmetic as `rolling_dca` (verified to 1e-9). Pairs: five century US pairs (FF_TECH/FF_VALUE, FF_TECH/FF_MKT, FF_MKT/FF_VALUE, FF_GROWTH/FF_VALUE, FF_TECH/defensive mix), all 66 pairs of the 12 industries (A = more volatile in 1926-62), FF_MKT/FF_DEVXUS (1990+), VFINX/PRITX (1980+). Windows: rolling monthly starts, decisions begin 37 months after data start (signal warm-up). Rolling windows per century pair: 1,106 / 1,046 / 926 for 5y / 10y / 20y; FF_MKT/FF_DEVXUS and VFINX/PRITX at 10y: 278 and 402.

Two controls are reported. **Window-matched**: the exp3 control (fixed mix with the window's own average A-fraction). **Global**: one fixed mix with the rule's full-sample average A-fraction, which has no in-window alignment bias. Ratios are medians over windows; >1 means the rule ended with more wealth.

Significance, three references: (a) circular-shift placebo (signal books rotated/time-reversed by at least 48 months against prices, common shift for all pairs so cross-pair dependence is kept; the cross-pair median is recomputed per shift); (b) block bootstrap of history (mean block 18 months); (c) iid-month resampling and a pure iid synthetic world (the "mechanics" benchmark). Only (c) contains the mechanical effects, which is why (a) wrongly flags everything.

### A1. 66-pair panel, 10-year windows (median over pairs of per-pair medians)

Window-matched control (round-1 yardstick):

{{table:panel10_matched}}

Global-average control:

{{table:panel10_global}}

`q25/q75` = inter-quartile range across the 66 pairs; `share>1` = share of pairs with median ratio above 1; `p (placebo)` = circular-shift placebo, two-sided (min possible 0.001; the shifted-signal null does not contain the mechanical effect); `boot_p5/p95` = block-bootstrap 90% interval of the cross-pair median; `iid_p5/p95` = iid-resampled-month 90% interval of the cross-pair median; `nonov_*` = non-overlapping windows (average over all 120 start phases; about 8-9 independent 10y windows per pair, and the 66 pairs are not independent). The real values sit inside the iid and bootstrap intervals for every rule except where noted below; nothing is distinguishable from the no-predictability world.

Pure synthetic check (two identical iid assets, 25% vol, corr 0.8, 300 paths of 97 years), 10y windows:

{{table:demo10}}

Compare with the real panel rows above (window-matched 0.970 / 1.031 / 0.978 / 1.023 / 1.032 / 1.008 / 0.992 / 0.989 / 1.011). The synthetic world has no momentum, no reversal and no mean difference, yet reproduces every number.

Era split (windows by start date; "end<1980" = windows entirely before round 1's data), global control, 10y:

{{table:panel_eras}}

Window-matched control:

{{table:panel_eras_matched}}

The sign and size are the same in 1926-62, 1963-98 and 1999-2026 and in windows that end before 1980: the pattern does not depend on the 2000-02 episode (except dd20_dip, which is smallest in 1999+ on the global control, and was 1.005-1.012 on the matched control in all eras). 5y and 20y panel results:

{{table:panel_5y}}

{{table:panel_20y}}

At 20 years the effects roughly double for the contrarian and drawdown rules, again matching the iid worlds (`a_panel_summary.csv`).

### A2. Named pairs and international pairs, 10y windows, global control (ratio to a fixed mix; >1 favours the rule)

{{table:named_nwin}}

{{table:named10_global}}

Same, window-matched control:

{{table:named10_matched}}

Circular-shift p-values (global control; shift null, so NOT the mechanical benchmark):

{{table:named10_p}}

Non-overlapping 10y windows (mean over start phases; only 3-9 windows per pair, so about 3 for the international pairs):

{{table:named10_nonov}}

Pairs with low relative volatility (FF_GROWTH/FF_VALUE, FF_MKT/FF_VALUE) show about zero on the global control, because both the rebalancing-type effect and the alignment bias scale with the relative variance of the two assets. The strongest named result, FF_TECH vs the defensive mix, is +1.6% for contra12, which is what the iid world gives for a pair with that relative volatility (`a_pairs_named_intl.csv`, column `iid_mean_g`). In the two international pairs the momentum and drawdown signs are the opposite of the US industry pattern (e.g. dd20_dip -6.0% and -4.8%): with only 278 and 402 overlapping windows (about 3 independent decades) these are not stable findings either way, but they do show the US-industry pattern is not universal.

### A3. Round-1 pairs against their own iid benchmark (same rules, no proxy warm-up, so numbers differ slightly from exp3), 10y:

{{table:round1_vs_iid}}

`iid_mean` is the mean median-ratio over iid-resampled worlds built from that pair's own history. On the window-matched control the real contra12/z_mean_revert numbers (e.g. VFINX/BRK-A 1.019 / 1.020) are BELOW the iid-world values (1.039 / 1.040): round 1's headline gains were not larger than what no predictability would give.

### A4. What the bootstrap and placebo can and cannot say
The bootstrap and iid resampling reshuffle the one history we have: they keep each asset's average return gap (e.g. tech's premium) as a permanent feature, cannot create regimes that never happened, and destroy mean reversion longer than the block length (18 months). A test of multi-year mean reversion between 3 and 10 years would need independent histories; the 100-year French sample has about 10 non-overlapping decades. What the test does show is that the 12-month-signal effects found in round 1 do not need any such mean reversion to exist.

---
## B. Walk-forward rule selection

Every January, 53 candidates (48 exp3 rule variants plus 5 fixed mixes) are scored by the DCA wealth of the trailing window (expanding, rolling 10y, rolling 20y; contributions on rows strictly before the decision day, marked at the previous close), the top one is used for the next 12 contributions. All three schemes start at the same first January with 20 years of history (FF pairs 1950, VFINX/BRK-A 2004, ^NDX/BRK-A 2009 (price index), so the real out-of-sample spans are 77, 77, 23 and 18 years). A 10-year-start variant is in `b_walkforward_tracks.csv` (`universe = start_10y`).

### B1. Out-of-sample chained tracks (terminal wealth per $ contributed, marked at the last close; IRR in %)

{{table:b_tracks}}

"hindsight-best" = the best of the 53 candidates over the same OOS span, i.e. what the in-sample best rule scores: it is always at or above every walk-forward number, and the gap is the cost of having to choose in advance (IRR gaps are small, 0.1-0.4 pp, because the candidates differ little; they are large only in wealth terms over decades).

### B2. Rolling 10y windows inside the OOS track (ratio of walk-forward wealth to fixed 50/50)

{{table:b_rolling}}

### B3. What gets picked

{{table:b_picks}}

Most picks are fixed mixes (a pure asset choice, usually "all VALUE" or "all BRK-A", i.e. the long-run winner), not timing rules; momentum-type rules are almost never picked and `dd*_dip` is picked in the NDX and VFINX samples. Expanding selection locks onto one fixed mix for decades (FF_MKT/FF_VALUE and VFINX/BRK-A: 100% fixed_0A_100B). Picks change frequently under rolling windows (6-10 distinct rules in 18-77 selections).

### B4. In-sample to out-of-sample shrinkage (rolling-10y pick; in-sample = trailing 10y DCA wealth vs fixed 50/50; out-of-sample = the same rule over the next 5 years vs fixed 50/50)

{{table:b_shrink}}

Picked rules look about +10% better than 50/50 in their training window and are +0.8% better in the following 5 years (+1.6-1.7% in FF_MKT/FF_VALUE and VFINX/BRK-A, -0.1% in the other two); the median candidate shows 0 by construction. See `fig4_wf_rules_shrinkage.png`.

### B5. Bootstrap (resampled histories: full walk-forward re-run on each path)

{{table:b_boot}}

`real_ratio` is the actual OOS terminal-wealth ratio, columns 5-8 describe resampled histories. The resampled worlds retain the full-history asset mean gaps, so asset selection persists by construction (VFINX/BRK-A: P(WF beats 50/50) = 0.8-0.9 in resampled worlds, but the actual result, 0.99, is at the 8th-17th percentile of that distribution): in the real history BRK-A's lead was concentrated before the OOS span, so what was learned did not carry forward. Versus the window-matched mix the resampled ratio is about 0.9 because of the control bias described above, so that comparison is not informative.

---
## C. Walk-forward weights

Universes: U1 VFINX + BRK-A, U2 VFINX + BRK-A + PRITX (both 1990-2026, training from 1980; the first year trains on 113-114 months), U3a FF_MKT + FF_TECH + FF_VALUE (1936-2026), U3b adds FF_DEVXUS (2000-2026; kept separate because it only exists from 1990). Each January the split of the next 12 buy-only contributions is set from the trailing 10 years (excess over FF_RF for Sharpe): winner-take-all (highest trailing DCA wealth), weights proportional to the rank of trailing DCA wealth, long-only max-Sharpe and min-variance (grid search on the simplex), inverse volatility. Naive: 100% first asset (market), equal weight, 70/30 first/second asset. Full-track metrics (wealth multiple, max drawdown) come from `DCABacktest` with a policy that looks up the year's weights; IRR from the same cash flows (computed with a wider-bracket solver, because `src.dca.dca_metrics` returns NaN IRR on 90-year horizons: its +-8 log-rate bracket overflows `exp` beyond ~88 years).

### C1. OOS tracks

{{table:c_tracks}}

`yearly weight change` = average over Januaries of half the L1 distance between consecutive weight vectors (0.25 = a quarter of each year's contribution moves to a different asset). `avg max weight` = average of the largest weight. Weights over time: `fig6_wf_weights_over_time.png`; every weight vector: `c_weights_by_january.csv`.

### C2. Rolling OOS windows, 10y (method wealth / naive wealth; non-overlapping = average over start phases, n windows)

{{table:c_rolling}}

20-year windows are in `c_rolling_oos_windows.csv` (only 2-9 independent windows exist).

### C3. Paired difference vs naive mixes, bootstrap of resampled histories (full walk-forward re-run per path)

{{table:c_boot}}

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
