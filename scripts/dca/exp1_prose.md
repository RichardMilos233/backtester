# EXP1 - Stress test of the current portfolio

Script: `scripts/dca/exp1_current_portfolio.py` (prose source: `scripts/dca/exp1_prose.md`; tables below are regenerated on each run, the prose numbers are not). All data end 2026-11-13. Percent figures are in % unless a column says otherwise.

**Holdings modelled (US$7,000):** NVDA 1,000 / SPCX 1,000 / BRK-B 2,000 / VOO 700 / "other" 2,300, with "other" as (a) cash earning 0%, (b) QQQ, (c) SPY (so (c) = US$3,000 S&P 500 fund in total). Weights: NVDA 14.3%, SPCX 14.3%, BRK-B 28.6%, S&P 10.0% + other.

**Other portfolios:** 100% SPY/VOO; 100% BRK-B; 100% QQQ; 50/50 and 60/40 SPY/BRK-B; "target 80/20" = SPY 45 / BRK-B 25 / intl 10 (VXUS, EFA before 2011; before 2001-08 that 10% is reallocated to SPY) / NVDA 10 / SPCX-proxy 10; 60/40 SPY/bonds (AGG, VBMFX before 2003-09).

**Series used (all adjusted total-return closes):** S&P fund = VOO (2010-09+), else SPY (1993-02+), else VFINX; BRK = BRK-B, else BRK-A (before 1996-05; same company, different share class); QQQ from 1999-03 (no pre-1999 Nasdaq stand-in used, so QQQ-containing portfolios are n/a before then). Cash earns 0% (understates a real cash/T-bill yield).

**Global caveats:** Berkshire, NVDA and QQQ are known winners chosen in hindsight (survivorship). SPCX has 107 daily returns, so every SPCX-specific number is a small-sample estimate and its proxies are assumptions, not data. Episodes are few (about 10 distinct since 1999) and overlap across the S&P and QQQ lists; they are not independent samples. D uses overlapping daily windows for worst 1m/3m/12m.

## 1. SPCX from its own history

{{table:spcx_summary}}

{{table:spcx_stats}}

SPCX (first close 2026-06-12, last 2026-11-13) has 107 daily returns. Its annualised daily volatility is about 82%, against roughly 24% for QQQ, 14% for SPY, 41% for NVDA and 52% for TSLA over the same days. Beta to QQQ is 1.61 (standard error 0.29), to SPY 2.62 (s.e. 0.50), with correlation only 0.48 to QQQ and 0.45 to SPY (R2 about 0.2): most of SPCX's daily variance in this window is not explained by the index. Correlation is highest with ARKK (0.56) and lowest with NVDA (0.28). The synthetic proxy ("SYN") therefore uses beta 1.61 x QQQ daily return (the estimate exceeds the 1.5 floor, so the floor did not bind). SYN has the right market sensitivity but none of SPCX's idiosyncratic volatility, so it understates SPCX risk in calm markets and may or may not do so in a crash. The first close-to-last total return is -13.6% with a worst drawdown of -48.8% from the closing peak (2026-06-16).

## A. Market state as of 2026-11-13

{{table:A_market_state|pct}}

"Current drawdown" for a series is the same thing as distance from its running all-time high of the adjusted close; the 52-week-high distance is shown as a separate column. Column `since_start` is a multiple-of-initial-value minus 1, times 100.

Over the last 12 months SPY is +13.5%, QQQ +12.3%, BRK-B +15.6%, NVDA -12.6%, SMH +41.6%. SPY is 3.2% below its high (2026-10-06), QQQ 10.5% below, NVDA 31.9% below and SMH 27.2% below its 2026-06-22 high. BRK-B closed at its all-time high on the last day. NVDA fell about 28% in each of the last 1-, 3- and 6-month windows while SMH fell 17% in 1-3 months, i.e. the recent weakness is concentrated in semiconductors/AI names. SPCX is 34% below its 2026-06-16 closing high. Neither SPY (-3.2%) nor QQQ (-10.5%) is currently in a drawdown that meets the thresholds below.

{{table:A_relative_12m}}

(Percentile = share of days, within each series' own history, on which the trailing 252-trading-day relative return was below today's value; histories start 1997-05 (vs SPY) and 2000-03 (vs QQQ). Windows overlap daily, so the effective number of independent observations is about 29 and 26.) Today BRK-B leads SPY by 6.4 points and QQQ by 8.0 points over 12 months, the 66th and 72nd percentile of their own histories, i.e. unremarkable. Berkshire's relative return has ranged from -58 to +84 points against SPY and -177 to +132 points against QQQ (the latter dominated by 1999-2002).

### S&P drawdown episodes >= 10% (VFINX 1980-1993, SPY 1993+; close-based, adjusted)

{{table:A_episodes_SP_10pct_since1980|pct}}

(The 1980-2026 sample has 20 such episodes. Depths are in %, `peak_to_trough_days`/`trough_to_recovery_days` are calendar days; recovery = first close at or above the old peak. SPY-only detection since 1993 gives the same peak and trough dates, with depths within 0.3 points and one differing recovery date (2018-01 episode), see `A_episodes_SPYonly_10pct.csv`.) Three episodes since 1999 exceed 30%: 2000-03 (-47.5%, 929 days down and 1,478 days to recover), 2007-10 (-55.2%) and 2020 (-34.0%); 2022 was -24.5%. The 2025 episode (-18.7%, 48 days down, 79 days up) is the only one in the data since 2022 that is >= 10%; there is none in 2026 so far.

### QQQ drawdown episodes >= 15% (since 1999-03)

{{table:A_episodes_QQQ_15pct|pct}}

QQQ's dot-com episode is -83.0% over 926 days, with recovery to the old (adjusted) high only on 2015-02-20, 4,517 days after the trough. The 2021-12 to 2022-11 episode is -35.1%. The first row (2000-01-03 to 2000-01-06, -15.5% in three days) is the start of the same bubble period and is a distinct episode only because the March 2000 high was higher.

## B. Episode table

Buy-and-hold from the close of the peak date, start weights, no rebalancing. "Ret" = return peak to trough; "MaxDD" = maximum drawdown of the portfolio value inside [peak, recovery] (or to the end of data if not recovered; all listed episodes recovered). Episodes: the S&P list (SP) and the QQQ list (QQ), labelled by peak > trough dates. n/a = a component did not exist at the peak date. Full long table: `B_episode_portfolios_long.csv`.

### B1. Per-asset returns peak to trough (%)

{{table:B_episode_asset_returns|pct}}

SYN is the synthetic beta x QQQ proxy; INTL = EFA/VXUS; BOND = VBMFX/AGG. BRK-B had positive returns in the 2000-2002 episode (+25.9%) and 2025 (+1.8%), and a -43.5% return in 2007-09. NVDA fell 68% (2000-02), 77% (2007-09), 62% (2022) and 31% (2025). TSLA fell 38-53% in the 2015, 2020, 2022 and 2025 episodes but only 1-8% in 2018's two episodes.

### B2. Portfolios without SPCX - return peak to trough (%)

{{table:B_noSPCX_ret_peak_to_trough|pct}}

### B3. Portfolios without SPCX - max drawdown peak to recovery (%)

{{table:B_noSPCX_maxdd_peak_to_recovery|pct}}

### B4. Portfolios with SPCX-proxy - return peak to trough (%), by proxy

TSLA proxy (episodes from 2015):

{{table:B_SPCX_TSLA_ret_peak_to_trough|pct}}

ARKK proxy (episodes from 2015-12 or later, ARKK data start 2014-11):

{{table:B_SPCX_ARKK_ret_peak_to_trough|pct}}

Synthetic beta x QQQ proxy (episodes from 1999-03):

{{table:B_SPCX_SYN_ret_peak_to_trough|pct}}

### B5. Portfolios with SPCX-proxy - max drawdown peak to recovery (%)

TSLA:

{{table:B_SPCX_TSLA_maxdd_peak_to_recovery|pct}}

ARKK:

{{table:B_SPCX_ARKK_maxdd_peak_to_recovery|pct}}

SYN:

{{table:B_SPCX_SYN_maxdd_peak_to_recovery|pct}}

Across the four large episodes (2000-03, 2007-10, 2022, 2025) results differ by episode and by what "other" is. 2022-01 to 2022-10: cur(a) -21% to -24%, cur(b) -33% to -35%, cur(c) -29% to -32%, 100% SPY -24.5%, 50/50 SPY/BRK -18.4%, 100% QQQ -34.3% (range across the three SPCX proxies). 2025: cur(a) -11%, cur(b) -18% to -19%, cur(c) -17%, SPY -18.7%, 50/50 -8.4%. 2000-02 (SYN only): cur(a) -20.8%, cur(c) -36.4%, cur(b) -48.1%, SPY -47.5%, BRK-B +25.9%, 50/50 -10.8%. 2007-10 (SYN only): cur(a) -39%, cur(c) -57%, cur(b) -56%, SPY -55%, 50/50 -49%. The result depends heavily on what "other" is: in the dot-com and 2022 cases the cash version beats everything except BRK-B-heavy mixes, simply because cash does not fall. The target 80/20 was within about 2 points of cur(c) in every episode that has a proxy.

The three SPCX proxies move the 2022 and 2025 portfolio results by about 1-2 points (e.g. 2022 cur(c): -29.4% TSLA, -31.8% ARKK, -30.1% SYN) because SPCX is 14% of the portfolio; only a proxy that differs by tens of points matters. For the pre-2010 episodes only SYN exists, and in 2000-02 it lost 95.8% (1.6 x an -83% QQQ), which is a strong assumption.

## C. Risk decomposition

Method: for portfolio risk, annualised covariances (252 days) are computed from daily returns over the trailing 252 (1y) and 756 (3y) trading days ending 2026-11-13, with fixed current weights (equivalent to daily rebalancing). SPCX covariances use only its 107-day overlap, pairwise, with all other pairs on the full window (FLAGGED: mixed-window covariance matrix; min eigenvalue was still positive in all cases). A third consistent-window estimate, `spcx_overlap`, uses only the 107 SPCX days for every pair. Cash has zero risk. Betas are portfolio covariance with SPY/QQQ divided by that index's variance in the same window.

### C1. Volatility and betas (vol in %)

{{table:C_risk|pct|noindex}}

The trailing numbers for current-(a/b/c) are inflated by one estimate: SPCX's 82% volatility (from 107 days). On the 3y window, ann. vol is 16.8% for cur(a), 22.8% for cur(b), 20.8% for cur(c), 17.9% for the target and 15.1% for 100% SPY. Betas to SPY (3y): 0.81 / 1.23 / 1.13 / 1.03. With the 1y window the portfolio betas to QQQ are 0.50 / 0.83 / 0.69 / 0.64. All three window choices give the same ranking cur(b) > cur(c) > target > cur(a) in volatility.

### C2. Share of portfolio variance (risk contribution, %)

{{table:C_risk_contribution|pct|noindex}}

SPCX, 14.3% of the dollars, accounts for 40-67% of the variance in all four portfolios shown (1y: 67% in cur(a), 47% in cur(b), 53% in cur(c), 40% in target; 3y: 59% / 40% / 44% / 33%). NVDA adds 17-25% in the current portfolios. Together, the two single stocks (28.6% of dollars) are 84-87% of the variance in cur(a) and 61-71% in cur(b/c). BRK-B, 28.6% of dollars, contributes 3-9.5% of variance in the current portfolios because of its low and often negative correlation with the others. In the target 80/20 the 10% SPCX weight still carries 33-40% of variance (again driven by the 82% volatility estimate). These shares move substantially if SPCX's volatility estimate is wrong; the 107-day estimate has a large standard error (about 7% of itself) and may well decline as lock-ups and the IPO period roll off.

### C3. Correlation of the holdings (trailing 1y, pairwise; SPCX on its overlap)

{{table:C_corr_1y|nd2}}

3y:

{{table:C_corr_3y|nd2}}

SPY-QQQ correlation is 0.93 (1y) and 0.95 (3y); NVDA-QQQ 0.67 / 0.73; BRK-B-SPY only 0.05 over 1y and 0.37 over 3y, and BRK-B-QQQ -0.12 / 0.20; SPCX-QQQ 0.48 and SPCX-NVDA 0.28 on the overlap (identical in both windows because they use the same 107 days).

### C4. How much of the S&P fund is "tech": regression of SPY on QQQ

Daily SPY returns regressed on daily QQQ returns over rolling 756-day windows ending 2002-01 to 2026-11 (figure `fig5_rolling_spy_on_qqq.png`).

{{table:C_rolling_spy_on_qqq_samples}}

{{table:C_rolling_spy_on_qqq_summary}}

{{table:C_spy_on_qqq_trailing}}

SPY's beta to QQQ has ranged between 0.35 (2002-04) and 0.99 (2011-10) with median 0.74; the latest 3y value is 0.70 with R2 0.90, the 1y value 0.59 with R2 0.86. In words: in the current 3y window about 90% of SPY's daily variance is explained by QQQ moves, and the S&P moves about 0.7% per 1% QQQ move. (Reverse: QQQ moves 1.29% per 1% SPY move on 3y, 1.46% on 1y.) R2 rose from 0.66-0.75 in 2002-2005 to 0.85-0.92 since 2010. This does not itself say how much tech weight the fund holds, only how closely it co-moves with QQQ; it does mean that adding QQQ to an S&P fund adds mostly the same exposure.

## D. Full-history simulation, monthly rebalanced fixed weights

Weights reset on the first trading day of each month (equivalently at the prior close), no costs or slippage, cash 0%. Three samples, differing by SPCX proxy: SYN from 1999-03-12, TSLA from 2010-07-01, ARKK from 2014-11-03. Portfolios without SPCX are rerun on each sample so they are comparable. In the SYN sample the intl weight is reallocated to SPY until 2001-08. Notes: the SYN series is 1.6 x daily QQQ, so it is a daily-reset leveraged series rebalanced monthly to 14% weight; NVDA (+4,300x over the data) and QQQ are hindsight winners, so CAGR of any portfolio with NVDA is an upper-bound-style hindsight number, not a forecast.

{{table:D_full_history|pct|noindex}}

CAGR, volatility, max drawdown and worst 1m/3m/12m returns are all in %; `end_multiple` is the final value over initial value. Highlights: since 1999-03 (SYN) the current portfolios gave CAGR 13.0% (a), 16.3% (b), 15.6% (c) with max drawdowns -46%, -65%, -59% against 100% SPY 8.5% / -55%, BRK-B 9.1% / -54%, QQQ 10.6% / -83%, target 80/20 13.3% / -58%, 60/40 SPY/bonds 6.8% / -36%. Since 2010-07 (TSLA) the same portfolios had CAGR 20.5% / 27.3% / 25.8% with max drawdowns -26% / -34% / -36%, versus SPY 15.0% / -34%, BRK-B 13.1% / -30%, target 21.8% / -35%. Since 2014-11 (ARKK) they had 16.5% / 22.6% / 21.0% and -28% / -38% / -35%. Cash helped drawdowns: cur(a) had the lowest max drawdown of the three, at the price of the lowest CAGR. The ranking by CAGR is mostly NVDA and the proxy's contribution (SPCX proxy and NVDA are about 29% of the current portfolios), which is a hindsight outcome; in the worst 12 months the current portfolios lost 22-54% depending on sample, versus about 17-20% (2010+) and 47-50% (1999+) for 100% SPY or BRK-B.

## E. Hypothetical shock table: actual asset moves applied to the current dollars

Each asset's actual peak-to-trough return in the S&P episode (peak/trough dates below) applied to the current weights of the US$7,000 portfolio ((a/b/c) as defined; target 80/20 and 50/50 SPY/BRK on the same US$7,000). Dollar loss (percent loss). SPCX is mapped to each proxy in turn; "flat -50%" assumes SPCX loses exactly half in every scenario for reference. n/a: proxy did not exist at the time.

{{table:E_shock_episodes|pct}}

{{table:E_shock_table}}

For the 2022 scenario the loss on the current portfolio is US$1,491-2,450 (21-35%) depending on the version of "other" and the proxy; the target 80/20 would have lost US$1,932-2,052 (28-29%) and 50/50 SPY/BRK US$1,286 (18.4%). In the 2025 episode: current US$747-1,311 (11-19%), target US$1,093-1,121 (16%), 50/50 US$590 (8.4%). In 2008 (SYN or flat -50% only) the current portfolio would have lost US$2,531-4,017 (36-57%) against 49% for 50/50. In the dot-com scenario cur(a) would have lost US$1,457 (21%), cur(c) US$2,550 (36%), and 50/50 SPY/BRK just US$758 (11%), because Berkshire gained 26% over that period. 50/50 SPY/BRK has a smaller loss than all three current versions in 2000-02, 2022 and 2025, but a larger one than cur(a) (36-39%) in 2008, when cash held its value and Berkshire fell 43.5%. These numbers use realised paths of assets that survived; they say nothing about a loss that is larger than anything in the data (a single-stock such as SPCX can go to zero).

## Deviations and implementation notes

- No engine (`DCABacktest`) used: everything is buy-and-hold or fixed-weight returns on closes. No decisions are made, so no lookahead question arises; rebalances reset at the previous close.
- "Dot-com 2000-02" is taken from the detected S&P episode (peak 2000-03-24, trough 2002-10-09), not a hard-coded date; likewise the other three shock episodes (2007-10 > 2009-03, 2022-01 > 2022-10, 2025-02 > 2025-04).
- QQQ drawdown-related pre-1999 values are n/a; `^NDX` price index was not used.
- S&P fund series spliced as VOO > SPY > VFINX; so "SPY" in the portfolio tables is VOO from 2010-09.
- Cash earns 0% (T-bills have yielded several percent in parts of the sample; this flatters equity-heavy portfolios in relative terms and understates cur(a) returns).
