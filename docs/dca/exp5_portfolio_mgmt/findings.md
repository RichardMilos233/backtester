# EXP5 - Portfolio management: transition, rebalancing rules, satellite sleeve

Engine: `src/dca.py` `DCABacktest` (slippage 0.0005, commission 0, fractional). Decisions on the first trading day of each month at the open, using data up to the previous close. Scripts: `scripts/dca/exp5_portfolio_mgmt.py` (runs `exp5a_transition.py`, `exp5b_rebalance.py`, `exp5c_satellite.py`; helpers in `exp5_common.py`). Data end 2026-11-13. `^NDX` is a price index (no dividends) wherever it appears (B, long NDX set only).

## Summary table (D)

| Part | What | Numbers |
|---|---|---|
| A | 6-month transition, T1 "80/20", 6-month window, all monthly starts (QQQ as "other") | End value vs doing nothing, median: lump -1.6% / 3 steps -1.3% / 6 steps -1.0% (1.5xQQQ proxy, n=326); lump -2.1% / -1.5% / -1.1% (TSLA proxy, n=191). p10 about -4% to -7%, p90 about +2% to +3%. Median in-window max drawdown: hold -11.4%, T1 about -10%. |
| A | Lump beats 6-step gradual (T1, 6m) | 34% of windows (1.5xQQQ), 31% (TSLA). In "hot tech" starts: 25% / 30% (n=63, overlapping, few distinct episodes). |
| A | Sensitivity: "other" held as cash instead of QQQ | The sign flips: T1 median +0.3% to +0.7%, T2 +1.0% to +2.2% vs hold, but the worst in-window drawdown is larger than holding cash (-50% vs -37% on 1.5xQQQ). The answer depends on what "other" really is. |
| B | Rebalancing rules, rolling 10y IRR (median across windows) | fixed_buy 10.8-13.3% vs other rules 11.0-13.1%. Paired gap +0.1 to +0.2 pp/yr in 4 of 5 target sets (-0.12 pp in VOO/BRK-B/VXUS, 12 windows) against a cross-window IRR std of about 4 pp. Max drawdown differences are 0-0.5 pp except where fixed_buy drifts a lot (NDX set: -49% fixed vs -43% to -46%). |
| B | Effort | band_10pt sells nothing in the median 10y window on 4 of 5 sets; band_5pt sells 0-3 times; rebalance_12m sells on 4-6 days per 10y (median USD 1.8k-3.8k, 3-6% of USD 60k contributed). fixed_buy end-of-window weights are off target by a median 1.5-6 pp (p90 5-8 pp). |
| C | 10y rolling, SPY60/BRK-B40 core + satellite, s=20% (median IRR change vs s=0) | QQQ +1.0 pp, SMH +1.5, MSFT +1.0, bad stock-picker -0.75, wipe-out -4.2 pp (wealth x0.80), NVDA +8.9, TSLA +13.1, AAPL +5.4 (the last three are hindsight winners). |
| C | Same, s=10% / worst case | s=10%: bad stock-picker -0.37 pp, wipe-out -2.0 pp (wealth x0.90). Account drawdown worsens in the worst 10y window by 8-16 pp for s=10% in a single volatile stock (NVDA -16, AAPL -12, TSLA -15), and by 24-26 pp at s=20%. |

## A. Six-month transition with no new cash

Setup. P0 (of $7,000): NVDA 14.3, SPCX 14.3, BRK-B 28.6, VOO 10, other 32.9 (weights normalised to 100%). Roles: NVDA, BRK-B, QQQ real; VOO = SPY spliced to VOO at 2010-09; intl = SPY (to 2001-08) -> EFA -> VXUS (2011-02), splices matched on the splice-date close. SPCX is proxied two ways: TSLA (sample 2010-07..2026-11, 191 monthly starts for a 6m window, 185 for 12m) and a synthetic 1.5 x QQQ daily return (no financing drag, no vol-decay beyond what daily compounding gives; sample 1999-04..2026-11, 326 / 320 starts). The first partial month of each sample is dropped. "Hold" = P0 with no trades.

Paths (all via `{'target': ...}` orders, zero contributions): lump = move to the end-state weights on day 0; "3 steps" / "6 steps" = on each of the first 3 / 6 monthly decision dates move 1/r of the remaining weight gap (r = steps left), reaching the end-state weights exactly on the last step. Note this also trims drift in assets that are already in the end state (e.g. NVDA in T2), so T2 is not strictly "touch only the others". Window return and drawdown come from daily total value starting at $7,000 at the day-0 open; "end value vs hold" is the end value divided by hold's end value minus 1. Spot-checked one window (TSLA sample, start 2015-07, T2 lump) by hand: engine and manual end value are identical ($7,387.57).

All starts, 6-month window, "other" = QQQ. Distribution of end value vs hold (%):

| Proxy (n) | End state | Path | median | p10 | p90 | min | median max DD | worst max DD | P(beat hold) |
|---|---|---|---|---|---|---|---|---|---|
| 1.5xQQQ (326) | hold P0 | - | 0 | 0 | 0 | 0 | -11.5% | -52.6% | - |
| | T1 80/20 | lump | -1.6 | -5.7 | +3.2 | -21.9 | -10.0% | -50.2% | 29% |
| | | 3 steps | -1.3 | -5.2 | +3.0 | -23.2 | -10.0% | -51.2% | 30% |
| | | 6 steps | -1.0 | -4.2 | +2.2 | -19.0 | -10.2% | -51.4% | 35% |
| | T2 others->VOO | lump | -0.7 | -2.9 | +2.5 | -15.7 | -10.9% | -51.5% | 36% |
| | | 6 steps | -0.4 | -2.5 | +1.7 | -14.2 | -10.8% | -52.3% | 40% |
| | T3 others->VOO+intl | lump / 6 steps | -0.8 / -0.4 | -3.3 / -2.5 | +2.6 / +1.9 | -15.7 / -14.2 | -10.9% / -10.8% | -52.2% / -52.6% | 35% / 38% |
| | T4 50/50 core | lump | -3.3 | -15.1 | +8.5 | -45.5 | -7.4% | -45.2% | 34% |
| | | 6 steps | -2.0 | -8.7 | +5.6 | -35.3 | -9.1% | -47.4% | 33% |
| TSLA (191) | hold P0 | - | 0 | 0 | 0 | 0 | -11.4% | -38.8% | - |
| | T1 80/20 | lump | -2.1 | -7.3 | +2.0 | -12.9 | -9.1% | -37.7% | 21% |
| | | 3 steps | -1.5 | -6.8 | +2.1 | -16.7 | -9.3% | -36.9% | 26% |
| | | 6 steps | -1.1 | -5.5 | +1.6 | -15.9 | -9.5% | -35.8% | 34% |
| | T2 others->VOO | lump | -0.8 | -2.3 | +1.3 | -5.0 | -10.5% | -39.9% | 29% |
| | | 6 steps | -0.4 | -3.1 | +1.6 | -11.7 | -10.3% | -36.3% | 38% |
| | T3 | lump / 6 steps | -1.0 / -0.6 | -2.9 / -3.4 | +1.3 / +1.6 | -5.6 / -12.0 | -10.0% / -10.3% | -40.2% / -36.3% | 26% / 32% |
| | T4 50/50 core | lump | -3.9 | -18.4 | +5.5 | -33.9 | -6.8% | -31.7% | 30% |
| | | 6 steps | -2.4 | -11.7 | +3.6 | -24.9 | -8.3% | -34.0% | 31% |

(3-step rows for T2-T4 and everything for 12m are in `a_transition_summary.csv`.) 12-month windows look the same, slightly wider: T1 lump median -3.1% (1.5xQQQ, n=320) / -4.2% (TSLA, n=185), 6 steps -2.0% / -3.4%; median max DD hold -16.6% / -14.6% vs T1 about -14.2% / -11.7%.

Lump vs gradual (end value, 6 months, T1 80/20, all starts): P(lump > 6-step) = 34% (1.5xQQQ) and 31% (TSLA); median (lump / 6-step - 1) = -0.6% and -0.7%, p10 -2.7%, p90 +1.8% / +1.3%. For T2 it is 42% / 46%, median about -0.2%. P(lump > 3-step) is 40-51%. In these samples, where the P0 holdings had on average outperformed the destination assets, the more you moved early the more you gave up, but the gaps between paths (about 0.5 pp median) are small against the spread across starts (p10-p90 about 4-5 pp), so lump vs gradual is not distinguishable from noise on this evidence. The larger and more consistent effect is the choice of end state (T4 and T1 vs T2/T3), which determines how much of the concentrated winners you give up.

What the numbers mainly show: transitioning costs a median 1-2% of the portfolio over 6 months (T1) or 0.4-0.8% (T2) relative to holding, mostly because P0 is full of historically strong assets (NVDA, TSLA/high-beta proxy, QQQ); the transition cuts median in-window drawdown by about 1.5 pp (T1) or 4 pp (T4) and in-window volatility from about 19% to 16-17% (T1). It does not remove tail drawdowns: the worst in-window drawdown (e.g. 2000-02, -50% on the 1.5xQQQ proxy) is barely changed because T1/T2/T3 still keep NVDA and the proxy at 10-14% each plus 45-70% in equity beta.

Conditional subsets (QQQ trailing 12m return > 20% and QQQ within 5% of its high = "hot tech"; QQQ > 15% below its high = "bust"; QQQ features use closes strictly before the start date and QQQ's own history from 1999-03). Windows overlap heavily, so n greatly overstates the independent evidence (hot tech: only a handful of distinct episodes, e.g. 1999, 2013-14, 2017, 2020-21, 2023-24):

| Subset, 6m, T1 80/20 | Proxy | n | hold median max DD / p10 / worst | T1 lump | T1 3 steps | T1 6 steps | median vs hold (lump / 3 / 6) | P(lump > 6-step) |
|---|---|---|---|---|---|---|---|---|
| hot tech | 1.5xQQQ | 63 | -11.5% / -25.8% / -31.5% | -9.8% / -22.3% / -33.3% | -10.0% / -22.2% / -33.4% | -10.6% / -23.8% / -32.7% | -1.5% / -1.5% / -0.9% | 25% |
| hot tech | TSLA | 63 | -11.9% / -22.2% / -37.1% | -10.5% / -19.4% / -36.9% | -10.5% / -20.0% / -35.3% | -10.8% / -20.8% / -35.8% | -1.4% / -1.4% / -1.2% | 30% |
| QQQ >15% off high | 1.5xQQQ | 184 | -11.5% / -31.6% / -52.6% | -10.3% / -22.8% / -50.2% | -10.0% / -25.7% / -51.2% | -10.4% / -27.8% / -51.4% | -0.9% / -0.8% / -0.5% | 43% |
| QQQ >15% off high | TSLA | 63 | -8.8% / -20.1% / -24.9% | -8.4% / -19.4% / -21.7% | -8.0% / -19.7% / -23.3% | -8.2% / -20.0% / -23.1% | -2.1% / -1.4% / -0.7% | 35% |

(12-month versions and T2 are in the CSVs.) In hot-tech starts the transition did not systematically beat holding: the median is about -1.5% (6m) and -2.5% (12m, T1 lump) and P(beat hold) is 25-38%. It lowered the p10 in-window drawdown by 2-3.5 pp (1.5xQQQ) or 2-3 pp (TSLA) but the worst case in the subset was the same. Counterfactually, the "selling after a hot run" intuition is not supported strongly by these samples, which are dominated by a few episodes. In the bust subset, the transition trailed hold by less in median (-0.5% to -0.9% for 6 steps/lump on 1.5xQQQ), consistent with the sold assets having mean-reverted upward on average, but again with wide p10-p90.

Cash sensitivity (other = 0%-return cash, n same): hold has lower volatility (14% vs 19%) and median max DD -8.5%; transitions now put cash into equities, so the median end value vs hold is slightly positive (T1 lump +0.5% to +0.7%, T2 lump +1.7% to +2.2%, p10 -3.5% to -4.4% for T1) and drawdown and volatility rise (T2 lump median DD -10.4% to -10.9% vs -8.0% to -8.5% hold; worst -50% on 1.5xQQQ vs -37% hold). That scenario is a different question (deploying cash), not de-risking.

Caveats. Proxies for SPCX are guesses (TSLA is a single high-vol stock; 1.5xQQQ has no financing or borrow cost and no idiosyncratic risk). Both proxy samples are strong-equity-market periods and NVDA, QQQ, BRK-B are known winners. Effective independent observations for 6m windows over 16 (TSLA) or 27 (1.5xQQQ) years are about 30-55, not 191-326, and the conditional subsets are a handful of episodes. VOO role uses SPY before 2010-09.

## B. Long-run rebalancing rule (DCA $500/month)

Rules: `fixed_buy` (fixed split of each deposit; weights drift), `buy_underweight` (new money only), `rebalance_to(every=12)` (full rebalance every 12th decision, otherwise buy_underweight), `rebalance_to(band=0.05 / 0.10)` (full rebalance when any weight is more than 5 / 10 pp off, checked each month on the weights including that month's deposit). Windows: whole sample and rolling 10y and 20y windows started every 6 months (n shown). Sold dollars/sell events count only sells. Sanity check: fixed_buy IRR equals `rolling_dca` IRR to 6 decimals for one 10y window (assert in script).

Rolling 10y windows, median across windows (full table incl. 20y, p10/p90, TWR, vol, worst 12m in `b_rebalance_summary.csv`):

| Target set (sample, n windows 10y / 20y) | Rule | IRR | max DD (median / worst) | USD sold (median) | sell days (median) | end weight dev. from target (median / p90) |
|---|---|---|---|---|---|---|
| SPY60/BRK-B40 (1996-06+, 41/21) | fixed_buy | 10.83% | -32.0% / -53.8% | 0 | 0 | 1.5 / 5.2 pp |
| | buy_underweight | 11.00% | -32.0% / -53.8% | 0 | 0 | 1.1 / 3.6 pp |
| | rebalance 12m | 11.01% | -32.0% / -53.7% | 2,410 | 5 | 1.1 / 3.7 pp |
| | band 5pp | 11.04% | -32.0% / -53.5% | 3,807 | 2 | 0.8 / 2.0 pp |
| | band 10pp | 11.00% | -32.0% / -53.8% | 0 | 0 | 1.1 / 3.6 pp |
| SPY50/QQQ20/BRK-B30 (1999-04+, 36/16) | fixed_buy | 12.15% | -51.2% / -53.2% | 0 | 0 | 4.6 / 7.2 pp |
| | buy_underweight | 12.26% | -50.9% / -53.1% | 0 | 0 | 1.2 / 2.6 pp |
| | rebalance 12m | 12.28% | -50.9% / -52.9% | 3,274 | 5.5 | 1.1 / 3.1 pp |
| | band 5pp | 12.27% | -50.9% / -52.8% | 280 | 1 | 1.2 / 2.6 pp |
| | band 10pp | 12.26% | -50.9% / -53.1% | 0 | 0 | 1.2 / 2.6 pp |
| VFINX60/BRK-A40 (1980-04+, 48/28) | fixed_buy | 10.21% | -32.5% / -52.7% | 0 | 0 | 1.8 / 8.1 pp |
| | buy_underweight | 10.40% | -32.4% / -52.7% | 0 | 0 | 0.8 / 3.1 pp |
| | rebalance 12m | 10.40% | -32.4% / -52.7% | 1,801 | 4 | 0.8 / 2.3 pp |
| | band 5pp | 10.44% | -32.4% / -52.4% | 3,656 | 2.5 | 0.7 / 2.1 pp |
| VFINX50/^NDX20/BRK-A30 (1985-11+, price-index NDX; 48/28) | fixed_buy | 10.58% | -49.5% / -52.4% | 0 | 0 | 3.9 / 7.9 pp |
| | buy_underweight | 11.05% | -46.0% / -52.4% | 0 | 0 | 1.1 / 4.2 pp |
| | rebalance 12m | 11.10% | -43.6% / -52.4% | 3,794 | 6 | 1.1 / 3.1 pp |
| | band 5pp | 11.13% | -43.3% / -51.8% | 1,034 | 1 | 1.1 / 3.0 pp |
| VOO50/BRK-B30/VXUS20 (2011-02+, 12 / none) | fixed_buy | 13.25% | -32.5% / -32.5% | 0 | 0 | 6.0 / 6.5 pp |
| | buy_underweight | 13.10% | -32.5% / -32.5% | 0 | 0 | 1.2 / 2.3 pp |
| | rebalance 12m | 13.09% | -32.5% / -32.5% | 2,280 | 5 | 0.8 / 2.2 pp |
| | band 5 / 10pp | 13.10% | -32.5% / -32.5% | 0 | 0 | 1.2 / 2.3 pp |

(band 10pp for VFINX sets in `b_rebalance_summary.csv`; it also sells 0 in the median 10y window.) Full-sample runs (IRR, SPY60/BRK-B40, 1996-06..2026-11): fixed_buy 10.95%, buy_underweight 11.05%, rebalance 12m 11.16% (25 sells, USD 177k sold = about 97% of contributions), band 5pp 11.21% (10 sells, USD 161k). For the long VFINX60/BRK-A40 full sample: 11.20 / 11.27 / 11.34 / 11.53 (band 5pp, 16 sells) %.

Paired differences vs fixed_buy (same window, 10y): `b_rule_vs_fixed_buy.csv`. Median IRR gain from any rule is +0.10 to +0.20 pp/yr in the SPY/BRK-B, the two VFINX sets, and +0.10 pp (SPY/QQQ/BRK-B, but only 81-83% of windows positive); it is negative, -0.12 pp, in VOO/BRK-B/VXUS (12 windows from a single 2011-2026 stretch, a period in which US large caps outran international, so any rule that buys more VXUS lags). In 20y windows the gain is +0.05 to +0.15 pp (VFINX/BRK-A, SPY/BRK-B) but ranges from -0.07 to +0.01 pp for SPY/QQQ/BRK-B (QQQ outperformed after 1999 so drift toward QQQ helped fixed_buy). 

Interpretation. Differences between rules are small relative to the dispersion of IRR across windows (std 4.1-4.8 pp for 10y, 1.3-2.1 pp for 20y windows), and the sign is determined by whether the asset that drifted up kept outperforming, which differs by sample. Within a given target set the paired differences are consistent in 80-100% of overlapping windows, but those windows overlap by 9.5 years of 10 and share the same few market episodes, so effective independent evidence is only about 2-3 observations (30-45 years / 10-20 y horizons). The reasonable reading is: rules do not matter much for return in this data; what they do change is staying at target (median end-of-window deviation 0.7-1.5 pp for any rule that uses new money vs up to 4-6 pp for fixed_buy and p90 5-8 pp) and, for the NDX set (where drift is large), 3-6 pp less median drawdown. With $500/month on a modest account, buy_underweight alone keeps the weights nearly as close to target as full rebalancing, and a 10pp band almost never triggers a sale (0 sells in the median window), while an annual rebalance sells on about 5 days per decade (USD 1.8-3.8k per 10y, 3-6% of contributions; much larger over 20-30y as the account grows: USD 25-38k, 20-32% of contributions over 20y). Costs of selling in a taxable account (not modelled) would reduce the small gain from rebalancing; slippage of 5 bp is included but is negligible here.

Figures: `b1_rebalance_rules.png` (IRR and drawdown by rule, 10y), `b2_selling_effort.png`, `b3_weight_drift_fixed_buy.png`.

## C. Satellite ("play money") sleeve

Core SPY 60 / BRK-B 40; each month $500 total, of which share s goes to the satellite and 1-s to the core (60/40). Scenarios use their own availability (inner join with SPY and BRK-B, first full month onward): QQQ/NVDA/MSFT/AAPL/SMH from 1999-2000, TSLA 2010-07, ARKK and the NVDA/TSLA/ARKK equal mix 2014-11 (so only 5 rolling 10y windows and 15 5y windows: treat those as anecdotes), "bad stock-picker" = 1.5 x SPY daily return minus 8%/yr (SPY sample, 1996+), "wipe-out" = the satellite share of each deposit is simply lost (IRR measured against the full $500 deposits; drawdown shown for the core only, because the loss is booked as a deposit that never invested). Cap variant: once a year (every 12th decision) if the satellite is above 1.5 x s of the account, sell the satellite back to s of the account and put the proceeds plus that month's deposit in the core, 60/40. IRR change = IRR(s) - IRR(0) over the identical window, in percentage points per year.

Rolling 10y windows, s = 10% and 20% (uncapped; capped in brackets where different). "worst d(maxDD)" is the change in max drawdown (pp) in the worst window vs s=0 (negative = worse).

| Satellite | n | s=0 IRR (median) | s=10%: IRR change med (p10..p90) | s=20%: IRR change med (p10..p90) | s=20% wealth ratio med / min | s=20% median d(maxDD) / worst | s=20% end sat. share med / p90 / max, uncapped [capped] |
|---|---|---|---|---|---|---|---|
| QQQ | 36 | 11.2% | +0.5 (+0.2..+0.8) | +1.0 (+0.4..+1.6) | 1.06 / 0.98 | +0.6 / -0.4 | 24 / 26 / 29 [same] |
| SMH | 33 | 11.8% | +0.7 (-0.3..+2.3) | +1.5 (-0.7..+4.3) | 1.08 / 0.94 | +0.2 / -1.0 | 26 / 37 / 50 [23 / 29 / 36] |
| MSFT | 41 | 10.8% | +0.5 (-0.3..+2.7) | +1.0 (-0.6..+5.0) | 1.06 / 0.95 | -0.3 / -8.9 | 24 / 39 / 43 [21 / 28 / 33] |
| AAPL | 41 | 10.8% | +2.9 (+1.6..+13.3) | +5.4 (+3.1..+20.9) | 1.34 / 1.10 | -1.2 / -22.0 | 40 / 74 / 79 [23 / 33 / 40] |
| NVDA | 36 | 11.4% | +5.0 (-0.4..+18.3) | +8.9 (-0.8..+27.4) | 1.62 / 0.96 | -10.0 / -24.2 | 51 / 82 / 87 [24 / 36 / 49] |
| TSLA | 13 | 13.9% | +7.7 (+5.3..+17.3) | +13.1 (+9.3..+26.2) | 2.02 / 1.48 | -22.1 / -26.3 | 60 / 80 / 83 [20 / 41 / 70] |
| ARKK | 5 | 14.6% | -0.4 (-0.7..-0.1) | -0.8 (-1.4..-0.1) | 0.96 / 0.93 | -3.5 / -4.3 | 17 / 19 / 21 [12 / 13 / 14] |
| mix NVDA/TSLA/ARKK | 5 | 14.6% | +8.4 (+6.9..+11.4) | +14.2 (+11.9..+18.4) | 2.15 / 1.80 | -11.1 / -12.5 | 63 / 70 / 71 [31 / 37 / 38] |
| bad picker (1.5xSPY-8%) | 41 | 10.8% | -0.4 (-0.7..-0.2) | -0.75 (-1.5..-0.4) | 0.96 / 0.89 | -3.3 / -7.9 | 17 / 18 / 20 [same] |
| wipe-out | 41 | 10.8% | -2.0 (-2.1..-2.0) | -4.2 (-4.4..-4.2) | 0.80 / 0.80 | 0 (core DD) | 0 |

Share of 10y windows in which s=20% had a higher IRR than s=0: QQQ 94%, SMH 70%, MSFT 63%, AAPL 100%, NVDA 81%, TSLA 100%, ARKK 20% (n=5), bad picker 0%, wipe-out 0%. 5y rolling windows (n=43-51 for the 1999-start assets): s=20% IRR change median QQQ +0.7, SMH +0.7, MSFT +0.4, AAPL +3.6, NVDA +6.8, TSLA +10.3 (n=23), bad picker -0.8, wipe-out -8.9 pp; the worst 5y account drawdown change is -13 pp (QQQ, i.e. the 2000-02 window), -31 pp (NVDA), -26 pp (AAPL). Full-period single runs and a common 2014-11..2026-11 run for all scenarios (s=20% IRR change: QQQ +0.9, SMH +5.3, MSFT +2.3, AAPL +3.6, NVDA +24.2, TSLA +7.5, ARKK -0.7, mix +14.0, bad picker -0.5, wipe-out -3.4 pp) are in `c_satellite_summary.csv` (horizon `full`, `common_full`). Linearity: the IRR change is roughly proportional to s for each scenario (s=30% is about 1.4-1.5x the s=20% value), see `c1_satellite_irr_delta.png`.

What the numbers show. (1) A 10% sleeve changes the account's IRR by only about -0.4 pp (bad stock-picker) to -2 pp (total loss of the sleeve) on the downside and +0.5 pp (QQQ/MSFT) in the typical case on the upside; at 20% the downside is -0.75 to -4.2 pp. Wipe-out is exactly the share lost (wealth x0.9 / x0.8); the bad picker costs 2-4% of final wealth at 10-20%. (2) The big positive deltas (NVDA, TSLA, AAPL, mix) are the survivorship/hindsight effect of choosing assets already known to have been historic winners; these are not forecasts, and even then the account's worst-window drawdown deepened 12-26 pp. The broad-index satellites (QQQ, SMH) changed IRR by less than about 1.5 pp at 20%, and QQQ's account drawdown was within 1 pp (the QQQ-heavy 1999-2000 window is the worst case: -13 pp over 5y). (3) Uncapped, a winner can balloon: with s=20% the median end-of-10y satellite share is 40% (AAPL), 51% (NVDA), 60% (TSLA) of the account, p90 74-82%, max 79-87%; the annual 1.5x cap holds it to a median of 20-24% (p90 33-41%, TSLA max 70% because a single year's spike exceeds the cap). The cap gave up about 0.4-0.6 pp (AAPL, NVDA) to 1 pp (TSLA) of the median IRR gain and cut the median drawdown increase by about 55-60% (NVDA -10.0 -> -4.2 pp; TSLA -22.1 -> -8.0 pp); capped runs sold USD 54-77k over the TSLA full sample, a taxable-event cost not modelled. (4) Dispersion across windows is large (e.g. NVDA 10y s=20% p10 -0.8, p90 +27.4 pp), and NVDA/TSLA/ARKK sample sizes (13, 5, 5 windows) are tiny and overlapping; only AAPL/MSFT/QQQ/SMH/bad-picker/wipe-out windows span 25-30 years (about 2-3 independent 10y periods).

Caveats: satellites are chosen with hindsight; synthetic bad picker adds no idiosyncratic risk beyond 1.5x leverage and the 8% drag; wipe-out accounting is stylised (see above); satellite sleeves are rebuilt monthly (no satellite-internal rebalancing); IRR is annualised on dated $500 deposits; ARKK/mix windows are mostly within one 2014-2026 cycle that included the 2021-22 ARKK collapse (explains the negative ARKK deltas).

Figures: `c1_satellite_irr_delta.png` (IRR delta vs s), `c2_satellite_end_share.png` (end satellite share, uncapped vs capped, s=20%).

## Files
`a_transition_summary.csv`, `a_lump_vs_gradual.csv`, `a_transition_windows.csv.gz` (all window rows, with QQQ features); `b_rebalance_summary.csv`, `b_rule_vs_fixed_buy.csv`, `b_rebalance_windows.csv.gz`; `c_satellite_summary.csv`, `c_satellite_windows.csv.gz`; figures `a1_transition_outcomes.png`, `a2_transition_drawdown_subsets.png`, `b1..b3`, `c1..c2`.
