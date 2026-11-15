# EXP1 - Stress test of the current portfolio

Script: `scripts/dca/exp1_current_portfolio.py` (prose source: `scripts/dca/exp1_prose.md`; tables below are regenerated on each run, the prose numbers are not). All data end 2026-11-13. Percent figures are in % unless a column says otherwise.

**Holdings modelled (US$7,000):** NVDA 1,000 / SPCX 1,000 / BRK-B 2,000 / VOO 700 / "other" 2,300, with "other" as (a) cash earning 0%, (b) QQQ, (c) SPY (so (c) = US$3,000 S&P 500 fund in total). Weights: NVDA 14.3%, SPCX 14.3%, BRK-B 28.6%, S&P 10.0% + other.

**Other portfolios:** 100% SPY/VOO; 100% BRK-B; 100% QQQ; 50/50 and 60/40 SPY/BRK-B; "target 80/20" = SPY 45 / BRK-B 25 / intl 10 (VXUS, EFA before 2011; before 2001-08 that 10% is reallocated to SPY) / NVDA 10 / SPCX-proxy 10; 60/40 SPY/bonds (AGG, VBMFX before 2003-09).

**Series used (all adjusted total-return closes):** S&P fund = VOO (2010-09+), else SPY (1993-02+), else VFINX; BRK = BRK-B, else BRK-A (before 1996-05; same company, different share class); QQQ from 1999-03 (no pre-1999 Nasdaq stand-in used, so QQQ-containing portfolios are n/a before then). Cash earns 0% (understates a real cash/T-bill yield).

**Global caveats:** Berkshire, NVDA and QQQ are known winners chosen in hindsight (survivorship). SPCX has 107 daily returns, so every SPCX-specific number is a small-sample estimate and its proxies are assumptions, not data. Episodes are few (about 10 distinct since 1999) and overlap across the S&P and QQQ lists; they are not independent samples. D uses overlapping daily windows for worst 1m/3m/12m.

## 1. SPCX from its own history

| index | value |
|---|---|
| first_date | 2026-06-12 |
| last_date | 2026-11-13 |
| daily_returns_used | 107 |
| ann_vol | 0.815 |
| total_return_first_close_to_last | -0.136 |
| max_drawdown_from_peak_close | -0.488 |
| beta_QQQ_estimate | 1.605 |
| beta_used_for_SYN | 1.605 |

| vs | beta | beta_se | corr | R2 | peer_ann_vol_same_days |
|---|---|---|---|---|---|
| QQQ | 1.61 | 0.29 | 0.479 | 0.23 | 0.243 |
| SPY | 2.62 | 0.50 | 0.453 | 0.21 | 0.141 |
| NVDA | 0.56 | 0.19 | 0.283 | 0.08 | 0.410 |
| TSLA | 0.51 | 0.14 | 0.328 | 0.11 | 0.519 |
| ARKK | 1.12 | 0.16 | 0.555 | 0.31 | 0.403 |

SPCX (first close 2026-06-12, last 2026-11-13) has 107 daily returns. Its annualised daily volatility is about 82%, against roughly 24% for QQQ, 14% for SPY, 41% for NVDA and 52% for TSLA over the same days. Beta to QQQ is 1.61 (standard error 0.29), to SPY 2.62 (s.e. 0.50), with correlation only 0.48 to QQQ and 0.45 to SPY (R2 about 0.2): most of SPCX's daily variance in this window is not explained by the index. Correlation is highest with ARKK (0.56) and lowest with NVDA (0.28). The synthetic proxy ("SYN") therefore uses beta 1.61 x QQQ daily return (the estimate exceeds the 1.5 floor, so the floor did not bind). SYN has the right market sensitivity but none of SPCX's idiosyncratic volatility, so it understates SPCX risk in calm markets and may or may not do so in a crash. The first close-to-last total return is -13.6% with a worst drawdown of -48.8% from the closing peak (2026-06-16).

## A. Market state as of 2026-11-13

| ticker | first_date | 1m | 3m | 6m | 12m | YTD | 3y_ann | 5y_ann | since_start | ATH_date | dist_from_ATH | dist_from_52w_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY | 1993-01-29 | -3.0 | -2.8 | 2.1 | 13.5 | 11.5 | 21.1 | 11.5 | 3036.2 | 2026-10-06 | -3.2 | -3.2 |
| QQQ | 1999-03-10 | -8.6 | -7.0 | -4.7 | 12.3 | 11.1 | 22.4 | 12.2 | 1481.8 | 2026-10-06 | -10.5 | -10.5 |
| BRK-B | 1996-05-09 | 13.9 | 17.0 | 22.1 | 15.6 | 18.0 | 19.2 | 15.7 | 2455.8 | 2026-11-13 | 0.0 | 0.0 |
| NVDA | 1999-01-22 | -28.7 | -27.6 | -27.7 | -12.6 | -12.5 | 49.8 | 40.0 | 433968.8 | 2026-10-06 | -31.9 | -31.9 |
| SPCX | 2026-06-12 | -11.1 | -1.5 | n/a | n/a | n/a | n/a | n/a | -13.6 | 2026-06-16 | -34.2 | -34.2 |
| VXUS | 2011-01-28 | -2.2 | -5.6 | -2.3 | 13.1 | 10.4 | 18.9 | 7.8 | 165.8 | 2026-09-04 | -6.4 | -6.4 |
| SMH | 2000-06-05 | -17.2 | -17.4 | -15.0 | 41.6 | 35.2 | 46.7 | 27.2 | 1061.7 | 2026-06-22 | -27.2 | -27.2 |
| BND | 2007-04-10 | 0.7 | -1.6 | -1.5 | -1.1 | -1.5 | 4.3 | -0.4 | 74.2 | 2020-08-06 | -4.0 | -3.2 |
| GLD | 2004-11-18 | -6.9 | -10.9 | -17.4 | -7.2 | -10.3 | 25.3 | 15.3 | 700.9 | 2026-01-29 | -28.3 | -28.3 |

"Current drawdown" for a series is the same thing as distance from its running all-time high of the adjusted close; the 52-week-high distance is shown as a separate column. Column `since_start` is a multiple-of-initial-value minus 1, times 100.

Over the last 12 months SPY is +13.5%, QQQ +12.3%, BRK-B +15.6%, NVDA -12.6%, SMH +41.6%. SPY is 3.2% below its high (2026-10-06), QQQ 10.5% below, NVDA 31.9% below and SMH 27.2% below its 2026-06-22 high. BRK-B closed at its all-time high on the last day. NVDA fell about 28% in each of the last 1-, 3- and 6-month windows while SMH fell 17% in 1-3 months, i.e. the recent weakness is concentrated in semiconductors/AI names. SPCX is 34% below its 2026-06-16 closing high. Neither SPY (-3.2%) nor QQQ (-10.5%) is currently in a drawdown that meets the thresholds below.

| series | current_12m_diff | percentile_in_own_history | hist_min | hist_median | hist_max | n_days | history_from | overlapping_12m_windows_note | current_BRK_12m | current_SPY_12m | current_QQQ_12m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRK_minus_SPY | 0.064 | 66.339 | -0.576 | 0.006 | 0.839 | 7427 | 1997-05-08 | daily overlapping | 0.180 | 0.116 | 0.100 |
| BRK_minus_QQQ | 0.080 | 71.980 | -1.769 | -0.042 | 1.321 | 6713 | 2000-03-08 | daily overlapping | 0.180 | 0.116 | 0.100 |

(Percentile = share of days, within each series' own history, on which the trailing 252-trading-day relative return was below today's value; histories start 1997-05 (vs SPY) and 2000-03 (vs QQQ). Windows overlap daily, so the effective number of independent observations is about 29 and 26.) Today BRK-B leads SPY by 6.4 points and QQQ by 8.0 points over 12 months, the 66th and 72nd percentile of their own histories, i.e. unremarkable. Berkshire's relative return has ranged from -58 to +84 points against SPY and -177 to +132 points against QQQ (the latter dominated by 1999-2002).

### S&P drawdown episodes >= 10% (VFINX 1980-1993, SPY 1993+; close-based, adjusted)

| series | peak_date | trough_date | depth | recovery_date | peak_to_trough_days | trough_to_recovery_days |
|---|---|---|---|---|---|---|
| S&P (VFINX->SPY->VOO) | 1980-02-13 | 1980-03-26 | -16.0 | 1980-06-18 | 42 | 84 |
| S&P (VFINX->SPY->VOO) | 1980-11-28 | 1982-08-12 | -25.2 | 1982-10-13 | 622 | 62 |
| S&P (VFINX->SPY->VOO) | 1983-10-10 | 1984-07-24 | -14.5 | 1984-11-06 | 288 | 105 |
| S&P (VFINX->SPY->VOO) | 1985-12-16 | 1986-01-22 | -10.1 | 1986-02-27 | 37 | 36 |
| S&P (VFINX->SPY->VOO) | 1986-12-02 | 1986-12-31 | -11.6 | 1987-01-28 | 29 | 28 |
| S&P (VFINX->SPY->VOO) | 1987-08-25 | 1987-10-19 | -33.1 | 1989-05-19 | 55 | 578 |
| S&P (VFINX->SPY->VOO) | 1990-01-02 | 1990-01-30 | -10.0 | 1990-05-21 | 28 | 111 |
| S&P (VFINX->SPY->VOO) | 1990-07-16 | 1990-10-11 | -19.2 | 1991-02-11 | 87 | 123 |
| S&P (VFINX->SPY->VOO) | 1997-02-18 | 1997-04-11 | -10.0 | 1997-05-05 | 52 | 24 |
| S&P (VFINX->SPY->VOO) | 1997-10-07 | 1997-10-27 | -11.2 | 1997-12-05 | 20 | 39 |
| S&P (VFINX->SPY->VOO) | 1998-07-20 | 1998-08-31 | -19.0 | 1998-11-23 | 42 | 84 |
| S&P (VFINX->SPY->VOO) | 1999-07-16 | 1999-10-15 | -11.7 | 1999-11-17 | 91 | 33 |
| S&P (VFINX->SPY->VOO) | 2000-03-24 | 2002-10-09 | -47.5 | 2006-10-26 | 929 | 1478 |
| S&P (VFINX->SPY->VOO) | 2007-10-09 | 2009-03-09 | -55.2 | 2012-08-16 | 517 | 1256 |
| S&P (VFINX->SPY->VOO) | 2015-07-20 | 2016-02-11 | -13.0 | 2016-04-18 | 206 | 67 |
| S&P (VFINX->SPY->VOO) | 2018-01-26 | 2018-02-08 | -10.1 | 2018-07-25 | 13 | 167 |
| S&P (VFINX->SPY->VOO) | 2018-09-20 | 2018-12-24 | -19.5 | 2019-04-12 | 95 | 109 |
| S&P (VFINX->SPY->VOO) | 2020-02-19 | 2020-03-23 | -34.0 | 2020-08-10 | 33 | 140 |
| S&P (VFINX->SPY->VOO) | 2022-01-03 | 2022-10-12 | -24.5 | 2023-12-13 | 282 | 427 |
| S&P (VFINX->SPY->VOO) | 2025-02-19 | 2025-04-08 | -18.7 | 2025-06-26 | 48 | 79 |

(The 1980-2026 sample has 20 such episodes. Depths are in %, `peak_to_trough_days`/`trough_to_recovery_days` are calendar days; recovery = first close at or above the old peak. SPY-only detection since 1993 gives the same peak and trough dates, with depths within 0.3 points and one differing recovery date (2018-01 episode), see `A_episodes_SPYonly_10pct.csv`.) Three episodes since 1999 exceed 30%: 2000-03 (-47.5%, 929 days down and 1,478 days to recover), 2007-10 (-55.2%) and 2020 (-34.0%); 2022 was -24.5%. The 2025 episode (-18.7%, 48 days down, 79 days up) is the only one in the data since 2022 that is >= 10%; there is none in 2026 so far.

### QQQ drawdown episodes >= 15% (since 1999-03)

| series | peak_date | trough_date | depth | recovery_date | peak_to_trough_days | trough_to_recovery_days |
|---|---|---|---|---|---|---|
| QQQ | 2000-01-03 | 2000-01-06 | -15.5 | 2000-01-19 | 3 | 13 |
| QQQ | 2000-03-27 | 2002-10-09 | -83.0 | 2015-02-20 | 926 | 4517 |
| QQQ | 2015-12-01 | 2016-02-09 | -16.1 | 2016-07-27 | 70 | 169 |
| QQQ | 2018-08-29 | 2018-12-24 | -22.8 | 2019-04-12 | 117 | 109 |
| QQQ | 2020-02-19 | 2020-03-16 | -28.6 | 2020-06-03 | 26 | 79 |
| QQQ | 2021-12-27 | 2022-11-03 | -35.1 | 2023-12-13 | 311 | 405 |
| QQQ | 2025-02-19 | 2025-04-08 | -22.8 | 2025-06-24 | 48 | 77 |

QQQ's dot-com episode is -83.0% over 926 days, with recovery to the old (adjusted) high only on 2015-02-20, 4,517 days after the trough. The 2021-12 to 2022-11 episode is -35.1%. The first row (2000-01-03 to 2000-01-06, -15.5% in three days) is the start of the same bubble period and is a distinct episode only because the March 2000 high was higher.

## B. Episode table

Buy-and-hold from the close of the peak date, start weights, no rebalancing. "Ret" = return peak to trough; "MaxDD" = maximum drawdown of the portfolio value inside [peak, recovery] (or to the end of data if not recovered; all listed episodes recovered). Episodes: the S&P list (SP) and the QQQ list (QQ), labelled by peak > trough dates. n/a = a component did not exist at the peak date. Full long table: `B_episode_portfolios_long.csv`.

### B1. Per-asset returns peak to trough (%)

| index | list | peak | trough | SP | QQQ | BRK | NVDA | TSLA | ARKK | SYN | INTL | BOND | GLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | SP>=10% | 1980-02-13 | 1980-03-26 | -16.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 1 | SP>=10% | 1980-11-28 | 1982-08-12 | -25.2 | n/a | 4.7 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | SP>=10% | 1983-10-10 | 1984-07-24 | -14.5 | n/a | -0.4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | SP>=10% | 1985-12-16 | 1986-01-22 | -10.1 | n/a | -11.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | SP>=10% | 1986-12-02 | 1986-12-31 | -11.6 | n/a | 1.8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | SP>=10% | 1987-08-25 | 1987-10-19 | -33.1 | n/a | -18.7 | n/a | n/a | n/a | n/a | n/a | -4.5 | n/a |
| 6 | SP>=10% | 1990-01-02 | 1990-01-30 | -10.0 | n/a | -15.9 | n/a | n/a | n/a | n/a | n/a | -2.1 | n/a |
| 7 | SP>=10% | 1990-07-16 | 1990-10-11 | -19.2 | n/a | -24.2 | n/a | n/a | n/a | n/a | n/a | 0.3 | n/a |
| 8 | SP>=10% | 1997-02-18 | 1997-04-11 | -10.0 | n/a | 2.1 | n/a | n/a | n/a | n/a | n/a | -2.1 | n/a |
| 9 | SP>=10% | 1997-10-07 | 1997-10-27 | -11.2 | n/a | -6.2 | n/a | n/a | n/a | n/a | n/a | 0.3 | n/a |
| 10 | SP>=10% | 1998-07-20 | 1998-08-31 | -19.0 | n/a | -21.1 | n/a | n/a | n/a | n/a | n/a | 2.3 | n/a |
| 11 | SP>=10% | 1999-07-16 | 1999-10-15 | -11.7 | -2.4 | -17.4 | -2.6 | n/a | n/a | -4.9 | n/a | -0.6 | n/a |
| 12 | SP>=10% | 2000-03-24 | 2002-10-09 | -47.5 | -82.9 | 25.8 | -68.4 | n/a | n/a | -95.8 | n/a | 27.0 | n/a |
| 13 | SP>=10% | 2007-10-09 | 2009-03-09 | -55.2 | -51.6 | -43.5 | -77.5 | n/a | n/a | -71.6 | -60.2 | 7.4 | 23.9 |
| 14 | SP>=10% | 2015-07-20 | 2016-02-11 | -13.0 | -14.8 | -13.4 | 29.5 | -46.7 | -31.2 | -23.8 | -20.9 | 2.9 | 12.6 |
| 15 | SP>=10% | 2018-01-26 | 2018-02-08 | -10.1 | -10.2 | -11.9 | -10.6 | -8.1 | -9.4 | -16.1 | -9.9 | -1.1 | -2.4 |
| 16 | SP>=10% | 2018-09-20 | 2018-12-24 | -19.5 | -22.0 | -15.3 | -52.2 | -1.0 | -26.9 | -33.6 | -15.1 | 1.6 | 5.0 |
| 17 | SP>=10% | 2020-02-19 | 2020-03-23 | -34.0 | -27.9 | -29.3 | -32.4 | -52.7 | -36.5 | -42.7 | -33.4 | -1.3 | -3.6 |
| 18 | SP>=10% | 2022-01-03 | 2022-10-12 | -24.5 | -34.3 | -12.2 | -61.8 | -45.7 | -62.8 | -51.0 | -27.6 | -14.4 | -7.3 |
| 19 | SP>=10% | 2025-02-19 | 2025-04-08 | -18.7 | -22.8 | 1.8 | -30.8 | -38.5 | -38.4 | -34.5 | -11.4 | 0.9 | 1.6 |
| 20 | QQQ>=15% | 2000-01-03 | 2000-01-06 | -5.3 | -15.5 | 2.2 | -12.0 | n/a | n/a | -24.1 | n/a | 0.1 | n/a |
| 21 | QQQ>=15% | 2000-03-27 | 2002-10-09 | -47.0 | -83.0 | 32.4 | -66.1 | n/a | n/a | -95.8 | n/a | 27.1 | n/a |
| 22 | QQQ>=15% | 2015-12-01 | 2016-02-09 | -11.5 | -16.1 | -6.4 | -22.2 | -37.5 | -28.7 | -25.0 | -13.4 | 1.3 | 11.0 |
| 23 | QQQ>=15% | 2018-08-29 | 2018-12-24 | -18.9 | -22.8 | -11.1 | -54.3 | -3.2 | -28.8 | -34.8 | -15.7 | 0.9 | 5.1 |
| 24 | QQQ>=15% | 2020-02-19 | 2020-03-16 | -29.4 | -28.6 | -22.5 | -37.6 | -51.5 | -40.0 | -43.3 | -31.1 | -0.4 | -6.7 |
| 25 | QQQ>=15% | 2021-12-27 | 2022-11-03 | -21.4 | -35.1 | -4.3 | -56.6 | -41.0 | -62.5 | -52.2 | -24.7 | -15.9 | -10.4 |
| 26 | QQQ>=15% | 2025-02-19 | 2025-04-08 | -18.7 | -22.8 | 1.8 | -30.8 | -38.5 | -38.4 | -34.5 | -11.4 | 0.9 | 1.6 |

SYN is the synthetic beta x QQQ proxy; INTL = EFA/VXUS; BOND = VBMFX/AGG. BRK-B had positive returns in the 2000-2002 episode (+25.9%) and 2025 (+1.8%), and a -43.5% return in 2007-09. NVDA fell 68% (2000-02), 77% (2007-09), 62% (2022) and 31% (2025). TSLA fell 38-53% in the 2015, 2020, 2022 and 2025 episodes but only 1-8% in 2018's two episodes.

### B2. Portfolios without SPCX - return peak to trough (%)

| episode | 100% SPY | 100% BRK-B | 100% QQQ | 50/50 | 60/40 SPY/BRK | 60/40 SPY/bond |
|---|---|---|---|---|---|---|
| SP 1980-02-13 > 1980-03-26 | -16.0 | n/a | n/a | n/a | n/a | n/a |
| SP 1980-11-28 > 1982-08-12 | -25.2 | 4.7 | n/a | -10.3 | -13.3 | n/a |
| SP 1983-10-10 > 1984-07-24 | -14.5 | -0.4 | n/a | -7.4 | -8.8 | n/a |
| SP 1985-12-16 > 1986-01-22 | -10.1 | -11.0 | n/a | -10.6 | -10.5 | n/a |
| SP 1986-12-02 > 1986-12-31 | -11.6 | 1.8 | n/a | -4.9 | -6.3 | n/a |
| SP 1987-08-25 > 1987-10-19 | -33.1 | -18.7 | n/a | -25.9 | -27.3 | -21.6 |
| SP 1990-01-02 > 1990-01-30 | -10.0 | -15.9 | n/a | -13.0 | -12.4 | -6.9 |
| SP 1990-07-16 > 1990-10-11 | -19.2 | -24.2 | n/a | -21.7 | -21.2 | -11.4 |
| SP 1997-02-18 > 1997-04-11 | -10.0 | 2.1 | n/a | -4.0 | -5.2 | -6.8 |
| SP 1997-10-07 > 1997-10-27 | -11.2 | -6.2 | n/a | -8.7 | -9.2 | -6.6 |
| SP 1998-07-20 > 1998-08-31 | -19.0 | -21.1 | n/a | -20.1 | -19.9 | -10.5 |
| SP 1999-07-16 > 1999-10-15 | -11.7 | -17.4 | -2.4 | -14.6 | -14.0 | -7.3 |
| SP 2000-03-24 > 2002-10-09 | -47.5 | 25.8 | -82.9 | -10.8 | -18.2 | -17.7 |
| SP 2007-10-09 > 2009-03-09 | -55.2 | -43.5 | -51.6 | -49.3 | -50.5 | -30.2 |
| SP 2015-07-20 > 2016-02-11 | -13.0 | -13.4 | -14.8 | -13.2 | -13.2 | -6.6 |
| SP 2018-01-26 > 2018-02-08 | -10.1 | -11.9 | -10.2 | -11.0 | -10.8 | -6.5 |
| SP 2018-09-20 > 2018-12-24 | -19.5 | -15.3 | -22.0 | -17.4 | -17.8 | -11.0 |
| SP 2020-02-19 > 2020-03-23 | -34.0 | -29.3 | -27.9 | -31.6 | -32.1 | -20.9 |
| SP 2022-01-03 > 2022-10-12 | -24.5 | -12.2 | -34.3 | -18.4 | -19.6 | -20.5 |
| SP 2025-02-19 > 2025-04-08 | -18.7 | 1.8 | -22.8 | -8.4 | -10.5 | -10.9 |
| QQ 2000-01-03 > 2000-01-06 | -5.3 | 2.2 | -15.5 | -1.5 | -2.3 | -3.1 |
| QQ 2000-03-27 > 2002-10-09 | -47.0 | 32.4 | -83.0 | -7.3 | -15.2 | -17.3 |
| QQ 2015-12-01 > 2016-02-09 | -11.5 | -6.4 | -16.1 | -8.9 | -9.5 | -6.4 |
| QQ 2018-08-29 > 2018-12-24 | -18.9 | -11.1 | -22.8 | -15.0 | -15.8 | -11.0 |
| QQ 2020-02-19 > 2020-03-16 | -29.4 | -22.5 | -28.6 | -25.9 | -26.6 | -17.8 |
| QQ 2021-12-27 > 2022-11-03 | -21.4 | -4.3 | -35.1 | -12.9 | -14.6 | -19.2 |
| QQ 2025-02-19 > 2025-04-08 | -18.7 | 1.8 | -22.8 | -8.4 | -10.5 | -10.9 |

### B3. Portfolios without SPCX - max drawdown peak to recovery (%)

| episode | 100% SPY | 100% BRK-B | 100% QQQ | 50/50 | 60/40 SPY/BRK | 60/40 SPY/bond |
|---|---|---|---|---|---|---|
| SP 1980-02-13 > 1980-03-26 | -16.0 | n/a | n/a | n/a | n/a | n/a |
| SP 1980-11-28 > 1982-08-12 | -25.2 | -27.1 | n/a | -23.2 | -22.2 | n/a |
| SP 1983-10-10 > 1984-07-24 | -14.5 | -11.9 | n/a | -10.8 | -10.9 | n/a |
| SP 1985-12-16 > 1986-01-22 | -10.1 | -14.3 | n/a | -11.3 | -10.8 | n/a |
| SP 1986-12-02 > 1986-12-31 | -11.6 | -4.1 | n/a | -6.5 | -7.4 | n/a |
| SP 1987-08-25 > 1987-10-19 | -33.1 | -37.1 | n/a | -33.5 | -33.1 | -21.6 |
| SP 1990-01-02 > 1990-01-30 | -10.0 | -23.7 | n/a | -15.3 | -13.7 | -6.9 |
| SP 1990-07-16 > 1990-10-11 | -19.2 | -24.2 | n/a | -21.7 | -21.2 | -11.4 |
| SP 1997-02-18 > 1997-04-11 | -10.0 | -4.6 | n/a | -5.5 | -6.3 | -6.8 |
| SP 1997-10-07 > 1997-10-27 | -11.2 | -6.2 | n/a | -8.7 | -9.2 | -6.6 |
| SP 1998-07-20 > 1998-08-31 | -19.0 | -24.6 | n/a | -20.7 | -20.2 | -10.5 |
| SP 1999-07-16 > 1999-10-15 | -11.7 | -21.0 | -11.8 | -14.9 | -14.0 | -7.3 |
| SP 2000-03-24 > 2002-10-09 | -47.5 | -23.3 | -83.0 | -24.8 | -27.6 | -19.9 |
| SP 2007-10-09 > 2009-03-09 | -55.2 | -53.9 | -53.4 | -53.8 | -53.8 | -30.2 |
| SP 2015-07-20 > 2016-02-11 | -13.0 | -14.1 | -16.1 | -13.2 | -13.2 | -6.9 |
| SP 2018-01-26 > 2018-02-08 | -10.1 | -14.9 | -10.5 | -11.0 | -10.8 | -6.5 |
| SP 2018-09-20 > 2018-12-24 | -19.5 | -16.1 | -22.7 | -17.4 | -17.8 | -11.0 |
| SP 2020-02-19 > 2020-03-23 | -34.0 | -29.3 | -28.6 | -31.6 | -32.1 | -20.9 |
| SP 2022-01-03 > 2022-10-12 | -24.5 | -26.6 | -34.8 | -24.2 | -23.7 | -20.6 |
| SP 2025-02-19 > 2025-04-08 | -18.7 | -10.4 | -22.8 | -10.0 | -11.0 | -10.9 |
| QQ 2000-01-03 > 2000-01-06 | -5.3 | -6.6 | -15.5 | -3.7 | -3.7 | -3.1 |
| QQ 2000-03-27 > 2002-10-09 | -55.2 | -53.9 | -83.0 | -53.7 | -53.7 | -25.9 |
| QQ 2015-12-01 > 2016-02-09 | -12.7 | -9.0 | -16.1 | -10.5 | -10.9 | -7.0 |
| QQ 2018-08-29 > 2018-12-24 | -19.5 | -16.1 | -22.8 | -17.3 | -17.8 | -11.1 |
| QQ 2020-02-19 > 2020-03-16 | -34.0 | -29.3 | -28.6 | -31.6 | -32.1 | -20.9 |
| QQ 2021-12-27 > 2022-11-03 | -24.5 | -26.6 | -35.1 | -24.2 | -23.7 | -20.9 |
| QQ 2025-02-19 > 2025-04-08 | -18.7 | -10.4 | -22.8 | -10.0 | -11.0 | -10.9 |

### B4. Portfolios with SPCX-proxy - return peak to trough (%), by proxy

TSLA proxy (episodes from 2015):

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 2015-07-20 > 2016-02-11 | -7.6 | -12.5 | -11.9 | -13.0 |
| SP 2018-01-26 > 2018-02-08 | -7.1 | -10.4 | -10.4 | -10.4 |
| SP 2018-09-20 > 2018-12-24 | -13.9 | -21.1 | -20.3 | -19.4 |
| SP 2020-02-19 > 2020-03-23 | -23.9 | -33.1 | -35.1 | -34.5 |
| SP 2022-01-03 > 2022-10-12 | -21.3 | -32.6 | -29.4 | -27.6 |
| SP 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |
| QQ 2015-12-01 > 2016-02-09 | -11.5 | -16.8 | -15.3 | -14.1 |
| QQ 2018-08-29 > 2018-12-24 | -13.3 | -20.8 | -19.5 | -18.6 |
| QQ 2020-02-19 > 2020-03-16 | -22.1 | -31.5 | -31.7 | -30.8 |
| QQ 2021-12-27 > 2022-11-03 | -17.3 | -28.9 | -24.3 | -22.9 |
| QQ 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |

ARKK proxy (episodes from 2015-12 or later, ARKK data start 2014-11):

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 2015-07-20 > 2016-02-11 | -5.4 | -10.3 | -9.7 | -11.5 |
| SP 2018-01-26 > 2018-02-08 | -7.3 | -10.6 | -10.6 | -10.5 |
| SP 2018-09-20 > 2018-12-24 | -17.6 | -24.8 | -24.0 | -22.0 |
| SP 2020-02-19 > 2020-03-23 | -21.6 | -30.8 | -32.8 | -32.8 |
| SP 2022-01-03 > 2022-10-12 | -23.7 | -35.0 | -31.8 | -29.3 |
| SP 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |
| QQ 2015-12-01 > 2016-02-09 | -10.2 | -15.5 | -14.0 | -13.2 |
| QQ 2018-08-29 > 2018-12-24 | -16.9 | -24.4 | -23.1 | -21.2 |
| QQ 2020-02-19 > 2020-03-16 | -20.4 | -29.8 | -30.1 | -29.7 |
| QQ 2021-12-27 > 2022-11-03 | -20.4 | -31.9 | -27.4 | -25.1 |
| QQ 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |

Synthetic beta x QQQ proxy (episodes from 1999-03):

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 1999-07-16 > 1999-10-15 | -7.2 | -8.0 | -11.1 | -11.5 |
| SP 2000-03-24 > 2002-10-09 | -20.8 | -48.1 | -36.4 | -36.1 |
| SP 2007-10-09 > 2009-03-09 | -39.3 | -56.2 | -57.4 | -56.6 |
| SP 2015-07-20 > 2016-02-11 | -4.3 | -9.2 | -8.6 | -10.7 |
| SP 2018-01-26 > 2018-02-08 | -8.2 | -11.6 | -11.5 | -11.2 |
| SP 2018-09-20 > 2018-12-24 | -18.6 | -25.8 | -25.0 | -22.7 |
| SP 2020-02-19 > 2020-03-23 | -22.5 | -31.7 | -33.7 | -33.5 |
| SP 2022-01-03 > 2022-10-12 | -22.1 | -33.3 | -30.1 | -28.1 |
| SP 2025-02-19 > 2025-04-08 | -10.7 | -18.2 | -16.8 | -15.6 |
| QQ 2000-01-03 > 2000-01-06 | -5.1 | -10.1 | -6.8 | -6.0 |
| QQ 2000-03-27 > 2002-10-09 | -18.6 | -45.8 | -34.0 | -33.9 |
| QQ 2015-12-01 > 2016-02-09 | -9.7 | -15.0 | -13.5 | -12.8 |
| QQ 2018-08-29 > 2018-12-24 | -17.8 | -25.3 | -24.0 | -21.8 |
| QQ 2020-02-19 > 2020-03-16 | -20.9 | -30.3 | -30.6 | -30.0 |
| QQ 2021-12-27 > 2022-11-03 | -18.9 | -30.5 | -26.0 | -24.1 |
| QQ 2025-02-19 > 2025-04-08 | -10.7 | -18.2 | -16.8 | -15.6 |

### B5. Portfolios with SPCX-proxy - max drawdown peak to recovery (%)

TSLA:

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 2015-07-20 > 2016-02-11 | -12.9 | -17.8 | -16.7 | -15.5 |
| SP 2018-01-26 > 2018-02-08 | -9.1 | -12.0 | -12.2 | -11.4 |
| SP 2018-09-20 > 2018-12-24 | -14.7 | -22.1 | -21.0 | -19.7 |
| SP 2020-02-19 > 2020-03-23 | -23.9 | -33.1 | -35.1 | -34.5 |
| SP 2022-01-03 > 2022-10-12 | -23.6 | -32.9 | -30.7 | -28.6 |
| SP 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |
| QQ 2015-12-01 > 2016-02-09 | -12.1 | -17.3 | -16.3 | -15.2 |
| QQ 2018-08-29 > 2018-12-24 | -14.4 | -21.8 | -20.7 | -19.5 |
| QQ 2020-02-19 > 2020-03-16 | -23.9 | -33.1 | -35.1 | -34.5 |
| QQ 2021-12-27 > 2022-11-03 | -23.7 | -32.9 | -30.7 | -28.6 |
| QQ 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |

ARKK:

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 2015-07-20 > 2016-02-11 | -12.5 | -17.3 | -16.2 | -15.1 |
| SP 2018-01-26 > 2018-02-08 | -7.3 | -10.6 | -10.6 | -10.5 |
| SP 2018-09-20 > 2018-12-24 | -17.7 | -25.1 | -24.0 | -22.0 |
| SP 2020-02-19 > 2020-03-23 | -21.6 | -30.8 | -32.8 | -32.8 |
| SP 2022-01-03 > 2022-10-12 | -24.0 | -35.0 | -31.8 | -29.3 |
| SP 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |
| QQ 2015-12-01 > 2016-02-09 | -11.2 | -16.5 | -15.3 | -14.4 |
| QQ 2018-08-29 > 2018-12-24 | -17.4 | -24.7 | -23.7 | -21.8 |
| QQ 2020-02-19 > 2020-03-16 | -21.6 | -30.8 | -32.8 | -32.8 |
| QQ 2021-12-27 > 2022-11-03 | -23.9 | -34.9 | -31.6 | -29.2 |
| QQ 2025-02-19 > 2025-04-08 | -11.2 | -18.7 | -17.4 | -16.0 |

SYN:

| episode | cur(a) cash | cur(b) QQQ | cur(c) SPY | target 80/20 |
|---|---|---|---|---|
| SP 1999-07-16 > 1999-10-15 | -8.9 | -11.4 | -12.1 | -12.1 |
| SP 2000-03-24 > 2002-10-09 | -38.1 | -50.6 | -46.5 | -41.5 |
| SP 2007-10-09 > 2009-03-09 | -41.9 | -57.9 | -58.9 | -58.0 |
| SP 2015-07-20 > 2016-02-11 | -12.1 | -16.9 | -15.8 | -14.8 |
| SP 2018-01-26 > 2018-02-08 | -8.2 | -11.6 | -11.5 | -11.2 |
| SP 2018-09-20 > 2018-12-24 | -19.1 | -26.4 | -25.4 | -22.8 |
| SP 2020-02-19 > 2020-03-23 | -22.5 | -31.7 | -33.7 | -33.5 |
| SP 2022-01-03 > 2022-10-12 | -23.8 | -33.3 | -31.0 | -28.8 |
| SP 2025-02-19 > 2025-04-08 | -10.7 | -18.2 | -16.8 | -15.6 |
| QQ 2000-01-03 > 2000-01-06 | -5.1 | -10.1 | -6.8 | -6.0 |
| QQ 2000-03-27 > 2002-10-09 | -52.8 | -62.0 | -61.4 | -59.6 |
| QQ 2015-12-01 > 2016-02-09 | -10.8 | -16.0 | -14.8 | -14.0 |
| QQ 2018-08-29 > 2018-12-24 | -18.7 | -26.1 | -25.0 | -22.5 |
| QQ 2020-02-19 > 2020-03-16 | -22.5 | -31.7 | -33.7 | -33.5 |
| QQ 2021-12-27 > 2022-11-03 | -23.7 | -33.3 | -30.9 | -28.7 |
| QQ 2025-02-19 > 2025-04-08 | -10.7 | -18.2 | -16.8 | -15.6 |

Across the four large episodes (2000-03, 2007-10, 2022, 2025) results differ by episode and by what "other" is. 2022-01 to 2022-10: cur(a) -21% to -24%, cur(b) -33% to -35%, cur(c) -29% to -32%, 100% SPY -24.5%, 50/50 SPY/BRK -18.4%, 100% QQQ -34.3% (range across the three SPCX proxies). 2025: cur(a) -11%, cur(b) -18% to -19%, cur(c) -17%, SPY -18.7%, 50/50 -8.4%. 2000-02 (SYN only): cur(a) -20.8%, cur(c) -36.4%, cur(b) -48.1%, SPY -47.5%, BRK-B +25.9%, 50/50 -10.8%. 2007-10 (SYN only): cur(a) -39%, cur(c) -57%, cur(b) -56%, SPY -55%, 50/50 -49%. The result depends heavily on what "other" is: in the dot-com and 2022 cases the cash version beats everything except BRK-B-heavy mixes, simply because cash does not fall. The target 80/20 was within about 2 points of cur(c) in every episode that has a proxy.

The three SPCX proxies move the 2022 and 2025 portfolio results by about 1-2 points (e.g. 2022 cur(c): -29.4% TSLA, -31.8% ARKK, -30.1% SYN) because SPCX is 14% of the portfolio; only a proxy that differs by tens of points matters. For the pre-2010 episodes only SYN exists, and in 2000-02 it lost 95.8% (1.6 x an -83% QQQ), which is a strong assumption.

## C. Risk decomposition

Method: for portfolio risk, annualised covariances (252 days) are computed from daily returns over the trailing 252 (1y) and 756 (3y) trading days ending 2026-11-13, with fixed current weights (equivalent to daily rebalancing). SPCX covariances use only its 107-day overlap, pairwise, with all other pairs on the full window (FLAGGED: mixed-window covariance matrix; min eigenvalue was still positive in all cases). A third consistent-window estimate, `spcx_overlap`, uses only the 107 SPCX days for every pair. Cash has zero risk. Betas are portfolio covariance with SPY/QQQ divided by that index's variance in the same window.

### C1. Volatility and betas (vol in %)

| window | n_days | portfolio | ann_vol | beta_SPY | beta_QQQ | uses_SPCX_overlap_cov |
|---|---|---|---|---|---|---|
| 1y | 252 | cur_a_cash | 15.7 | 0.77 | 0.50 | True |
| 1y | 252 | cur_b_qqq | 21.2 | 1.24 | 0.83 | True |
| 1y | 252 | cur_c_spy | 19.0 | 1.09 | 0.69 | True |
| 1y | 252 | target_80/20 | 16.4 | 1.03 | 0.64 | True |
| 1y | 252 | 50/50_SPY/BRK | 10.5 | 0.53 | 0.25 | False |
| 1y | 252 | 100_VOO/SPY | 13.6 | 0.99 | 0.59 | False |
| 1y | 252 | 100_QQQ | 21.5 | 1.46 | 1.00 | False |
| 1y | 252 | 100_BRK-B | 15.1 | 0.06 | -0.08 | False |
| 1y | 252 | 60/40_SPY/BRK | 10.4 | 0.62 | 0.32 | False |
| 1y | 252 | 60/40_SPY/bonds | 8.9 | 0.64 | 0.37 | False |
| 3y | 756 | cur_a_cash | 16.8 | 0.81 | 0.65 | True |
| 3y | 756 | cur_b_qqq | 22.8 | 1.23 | 0.98 | True |
| 3y | 756 | cur_c_spy | 20.8 | 1.13 | 0.88 | True |
| 3y | 756 | target_80/20 | 17.9 | 1.03 | 0.78 | True |
| 3y | 756 | 50/50_SPY/BRK | 12.9 | 0.68 | 0.41 | False |
| 3y | 756 | 100_VOO/SPY | 15.1 | 0.97 | 0.68 | False |
| 3y | 756 | 100_QQQ | 21.0 | 1.29 | 1.00 | False |
| 3y | 756 | 100_BRK-B | 16.1 | 0.39 | 0.15 | False |
| 3y | 756 | 60/40_SPY/BRK | 12.9 | 0.74 | 0.47 | False |
| 3y | 756 | 60/40_SPY/bonds | 9.6 | 0.61 | 0.42 | False |
| spcx_overlap | 107 | cur_a_cash | 15.8 | 0.74 | 0.42 | True |
| spcx_overlap | 107 | cur_b_qqq | 21.8 | 1.27 | 0.74 | True |
| spcx_overlap | 107 | cur_c_spy | 19.1 | 1.07 | 0.59 | True |
| spcx_overlap | 107 | target_80/20 | 16.4 | 1.00 | 0.54 | True |
| spcx_overlap | 107 | 50/50_SPY/BRK | 10.8 | 0.54 | 0.22 | False |
| spcx_overlap | 107 | 100_VOO/SPY | 14.0 | 0.99 | 0.53 | False |
| spcx_overlap | 107 | 100_QQQ | 24.3 | 1.60 | 1.00 | False |
| spcx_overlap | 107 | 100_BRK-B | 15.3 | 0.08 | -0.08 | False |
| spcx_overlap | 107 | 60/40_SPY/BRK | 10.8 | 0.63 | 0.28 | False |
| spcx_overlap | 107 | 60/40_SPY/bonds | 8.9 | 0.62 | 0.33 | False |

The trailing numbers for current-(a/b/c) are inflated by one estimate: SPCX's 82% volatility (from 107 days). On the 3y window, ann. vol is 16.8% for cur(a), 22.8% for cur(b), 20.8% for cur(c), 17.9% for the target and 15.1% for 100% SPY. Betas to SPY (3y): 0.81 / 1.23 / 1.13 / 1.03. With the 1y window the portfolio betas to QQQ are 0.50 / 0.83 / 0.69 / 0.64. All three window choices give the same ranking cur(b) > cur(c) > target > cur(a) in volatility.

### C2. Share of portfolio variance (risk contribution, %)

| window | portfolio | SP | QQQ | BRK | NVDA | SPCX | INTL | BOND | GLD |
|---|---|---|---|---|---|---|---|---|---|
| 1y | cur_a_cash | 5.9 | 0.0 | 6.8 | 20.0 | 67.4 | 0.0 | 0.0 | 0.0 |
| 1y | cur_b_qqq | 5.2 | 28.2 | 2.9 | 16.8 | 46.9 | 0.0 | 0.0 | 0.0 |
| 1y | cur_c_spy | 24.4 | 0.0 | 4.9 | 18.0 | 52.7 | 0.0 | 0.0 | 0.0 |
| 1y | target_80/20 | 32.4 | 0.0 | 5.2 | 14.8 | 39.7 | 7.8 | 0.0 | 0.0 |
| 1y | 50/50_SPY/BRK | 45.1 | 0.0 | 54.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1y | 100_VOO/SPY | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1y | 100_QQQ | 0.0 | 100.0 | -0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1y | 100_BRK-B | 0.0 | -0.0 | 100.0 | -0.0 | 0.0 | -0.0 | 0.0 | -0.0 |
| 1y | 60/40_SPY/BRK | 63.9 | 0.0 | 36.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1y | 60/40_SPY/bonds | 90.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9.8 | 0.0 |
| 3y | cur_a_cash | 6.7 | 0.0 | 9.4 | 25.3 | 58.6 | 0.0 | 0.0 | 0.0 |
| 3y | cur_b_qqq | 5.6 | 27.4 | 6.3 | 20.3 | 40.4 | 0.0 | 0.0 | 0.0 |
| 3y | cur_c_spy | 26.3 | 0.0 | 8.1 | 21.6 | 43.9 | 0.0 | 0.0 | 0.0 |
| 3y | target_80/20 | 33.9 | 0.0 | 9.2 | 17.5 | 33.0 | 6.5 | 0.0 | 0.0 |
| 3y | 50/50_SPY/BRK | 47.6 | 0.0 | 52.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3y | 100_VOO/SPY | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3y | 100_QQQ | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3y | 100_BRK-B | 0.0 | 0.0 | 100.0 | -0.0 | 0.0 | 0.0 | 0.0 | -0.0 |
| 3y | 60/40_SPY/BRK | 62.1 | 0.0 | 37.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3y | 60/40_SPY/bonds | 91.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.1 | 0.0 |
| spcx_overlap | cur_a_cash | 5.9 | 0.0 | 6.6 | 20.9 | 66.6 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | cur_b_qqq | 5.3 | 30.6 | 2.5 | 17.4 | 44.3 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | cur_c_spy | 24.7 | 0.0 | 4.9 | 18.5 | 51.8 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | target_80/20 | 32.9 | 0.0 | 5.3 | 15.2 | 39.4 | 7.2 | 0.0 | 0.0 |
| spcx_overlap | 50/50_SPY/BRK | 45.8 | 0.0 | 54.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | 100_VOO/SPY | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | 100_QQQ | 0.0 | 100.0 | -0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | 100_BRK-B | 0.0 | -0.0 | 100.0 | -0.0 | 0.0 | -0.0 | 0.0 | 0.0 |
| spcx_overlap | 60/40_SPY/BRK | 64.1 | 0.0 | 35.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spcx_overlap | 60/40_SPY/bonds | 91.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.1 | 0.0 |

SPCX, 14.3% of the dollars, accounts for 40-67% of the variance in all four portfolios shown (1y: 67% in cur(a), 47% in cur(b), 53% in cur(c), 40% in target; 3y: 59% / 40% / 44% / 33%). NVDA adds 17-25% in the current portfolios. Together, the two single stocks (28.6% of dollars) are 84-87% of the variance in cur(a) and 61-71% in cur(b/c). BRK-B, 28.6% of dollars, contributes 3-9.5% of variance in the current portfolios because of its low and often negative correlation with the others. In the target 80/20 the 10% SPCX weight still carries 33-40% of variance (again driven by the 82% volatility estimate). These shares move substantially if SPCX's volatility estimate is wrong; the 107-day estimate has a large standard error (about 7% of itself) and may well decline as lock-ups and the IPO period roll off.

### C3. Correlation of the holdings (trailing 1y, pairwise; SPCX on its overlap)

| index | SP | QQQ | BRK | NVDA | SPCX | INTL | BOND | GLD |
|---|---|---|---|---|---|---|---|---|
| SP | 1.00 | 0.93 | 0.05 | 0.63 | 0.46 | 0.84 | 0.37 | 0.36 |
| QQQ | 0.93 | 1.00 | -0.12 | 0.67 | 0.48 | 0.81 | 0.27 | 0.34 |
| BRK | 0.05 | -0.12 | 1.00 | -0.23 | 0.06 | -0.02 | 0.10 | -0.04 |
| NVDA | 0.63 | 0.67 | -0.23 | 1.00 | 0.28 | 0.51 | 0.16 | 0.26 |
| SPCX | 0.46 | 0.48 | 0.06 | 0.28 | 1.00 | 0.39 | 0.04 | 0.18 |
| INTL | 0.84 | 0.81 | -0.02 | 0.51 | 0.39 | 1.00 | 0.48 | 0.53 |
| BOND | 0.37 | 0.27 | 0.10 | 0.16 | 0.04 | 0.48 | 1.00 | 0.29 |
| GLD | 0.36 | 0.34 | -0.04 | 0.26 | 0.18 | 0.53 | 0.29 | 1.00 |

3y:

| index | SP | QQQ | BRK | NVDA | SPCX | INTL | BOND | GLD |
|---|---|---|---|---|---|---|---|---|
| SP | 1.00 | 0.95 | 0.37 | 0.66 | 0.46 | 0.78 | 0.19 | 0.20 |
| QQQ | 0.95 | 1.00 | 0.20 | 0.73 | 0.48 | 0.75 | 0.13 | 0.20 |
| BRK | 0.37 | 0.20 | 1.00 | -0.00 | 0.06 | 0.25 | 0.07 | -0.02 |
| NVDA | 0.66 | 0.73 | -0.00 | 1.00 | 0.28 | 0.46 | 0.01 | 0.12 |
| SPCX | 0.46 | 0.48 | 0.06 | 0.28 | 1.00 | 0.39 | 0.04 | 0.18 |
| INTL | 0.78 | 0.75 | 0.25 | 0.46 | 0.39 | 1.00 | 0.33 | 0.44 |
| BOND | 0.19 | 0.13 | 0.07 | 0.01 | 0.04 | 0.33 | 1.00 | 0.23 |
| GLD | 0.20 | 0.20 | -0.02 | 0.12 | 0.18 | 0.44 | 0.23 | 1.00 |

SPY-QQQ correlation is 0.93 (1y) and 0.95 (3y); NVDA-QQQ 0.67 / 0.73; BRK-B-SPY only 0.05 over 1y and 0.37 over 3y, and BRK-B-QQQ -0.12 / 0.20; SPCX-QQQ 0.48 and SPCX-NVDA 0.28 on the overlap (identical in both windows because they use the same 107 days).

### C4. How much of the S&P fund is "tech": regression of SPY on QQQ

Daily SPY returns regressed on daily QQQ returns over rolling 756-day windows ending 2002-01 to 2026-11 (figure `fig5_rolling_spy_on_qqq.png`).

| index | date | beta_SPY_on_QQQ | R2 |
|---|---|---|---|
| 0 | 2002-12-31 | 0.38 | 0.68 |
| 1 | 2005-12-30 | 0.59 | 0.75 |
| 2 | 2008-12-31 | 0.88 | 0.82 |
| 3 | 2010-12-31 | 0.95 | 0.86 |
| 4 | 2013-12-31 | 0.86 | 0.86 |
| 5 | 2016-12-30 | 0.77 | 0.86 |
| 6 | 2019-12-31 | 0.69 | 0.86 |
| 7 | 2022-12-30 | 0.79 | 0.87 |
| 8 | 2024-12-31 | 0.70 | 0.91 |
| 9 | 2025-12-31 | 0.73 | 0.90 |
| 10 | 2026-11-13 | 0.70 | 0.90 |

| index | min | min_date | max | max_date | median | latest |
|---|---|---|---|---|---|---|
| beta_SPY_on_QQQ | 0.349 | 2002-04-09 | 0.988 | 2011-10-06 | 0.743 | 0.697 |
| R2 | 0.663 | 2002-10-09 | 0.918 | 2025-04-25 | 0.851 | 0.896 |

| index | window | beta_SPY_on_QQQ | R2 | beta_QQQ_on_SPY |
|---|---|---|---|---|
| 0 | 1y | 0.59 | 0.86 | 1.46 |
| 1 | 3y | 0.70 | 0.90 | 1.29 |

SPY's beta to QQQ has ranged between 0.35 (2002-04) and 0.99 (2011-10) with median 0.74; the latest 3y value is 0.70 with R2 0.90, the 1y value 0.59 with R2 0.86. In words: in the current 3y window about 90% of SPY's daily variance is explained by QQQ moves, and the S&P moves about 0.7% per 1% QQQ move. (Reverse: QQQ moves 1.29% per 1% SPY move on 3y, 1.46% on 1y.) R2 rose from 0.66-0.75 in 2002-2005 to 0.85-0.92 since 2010. This does not itself say how much tech weight the fund holds, only how closely it co-moves with QQQ; it does mean that adding QQQ to an S&P fund adds mostly the same exposure.

## D. Full-history simulation, monthly rebalanced fixed weights

Weights reset on the first trading day of each month (equivalently at the prior close), no costs or slippage, cash 0%. Three samples, differing by SPCX proxy: SYN from 1999-03-12, TSLA from 2010-07-01, ARKK from 2014-11-03. Portfolios without SPCX are rerun on each sample so they are comparable. In the SYN sample the intl weight is reallocated to SPY until 2001-08. Notes: the SYN series is 1.6 x daily QQQ, so it is a daily-reset leveraged series rebalanced monthly to 14% weight; NVDA (+4,300x over the data) and QQQ are hindsight winners, so CAGR of any portfolio with NVDA is an upper-bound-style hindsight number, not a forecast.

| sample | portfolio | start | CAGR | ann_vol | max_drawdown | maxdd_date | worst_1m | worst_3m | worst_12m | end_multiple |
|---|---|---|---|---|---|---|---|---|---|---|
| SYN_1999-03+ | cur_a_cash | 1999-03-12 | 13.0 | 17.9 | -46.4 | 2009-03-09 | -22.2 | -33.2 | -42.0 | 29.61 |
| SYN_1999-03+ | cur_b_qqq | 1999-03-12 | 16.3 | 25.9 | -64.7 | 2002-10-09 | -31.0 | -46.4 | -54.2 | 66.04 |
| SYN_1999-03+ | cur_c_spy | 1999-03-12 | 15.6 | 23.5 | -58.9 | 2009-03-09 | -32.2 | -45.0 | -53.6 | 55.36 |
| SYN_1999-03+ | 100_VOO/SPY | 1999-03-12 | 8.5 | 19.2 | -55.2 | 2009-03-09 | -33.0 | -41.5 | -47.4 | 9.49 |
| SYN_1999-03+ | 100_BRK-B | 1999-03-12 | 9.1 | 21.8 | -53.9 | 2009-03-05 | -33.6 | -32.3 | -50.1 | 11.22 |
| SYN_1999-03+ | 100_QQQ | 1999-03-12 | 10.6 | 26.9 | -83.0 | 2002-10-09 | -31.1 | -46.1 | -69.7 | 16.13 |
| SYN_1999-03+ | 50/50_SPY/BRK | 1999-03-12 | 9.2 | 18.1 | -53.4 | 2009-03-09 | -31.2 | -36.5 | -48.3 | 11.38 |
| SYN_1999-03+ | 60/40_SPY/BRK | 1999-03-12 | 9.1 | 17.9 | -53.4 | 2009-03-09 | -31.6 | -37.5 | -48.0 | 11.15 |
| SYN_1999-03+ | target_80/20 | 1999-03-12 | 13.3 | 21.6 | -57.9 | 2009-03-09 | -32.2 | -43.6 | -51.6 | 31.94 |
| SYN_1999-03+ | 60/40_SPY/bonds | 1999-03-12 | 6.8 | 11.5 | -35.8 | 2009-03-09 | -22.6 | -27.1 | -30.3 | 6.13 |
| TSLA_2010-07+ | cur_a_cash | 2010-07-01 | 20.5 | 16.3 | -25.9 | 2020-03-18 | -24.7 | -21.4 | -22.2 | 21.16 |
| TSLA_2010-07+ | cur_b_qqq | 2010-07-01 | 27.3 | 22.3 | -34.1 | 2022-12-28 | -32.4 | -28.3 | -32.8 | 51.99 |
| TSLA_2010-07+ | cur_c_spy | 2010-07-01 | 25.7 | 21.2 | -35.8 | 2020-03-23 | -34.5 | -26.4 | -28.3 | 42.54 |
| TSLA_2010-07+ | 100_VOO/SPY | 2010-07-01 | 15.0 | 17.0 | -34.0 | 2020-03-23 | -33.0 | -30.0 | -19.7 | 9.83 |
| TSLA_2010-07+ | 100_BRK-B | 2010-07-01 | 13.1 | 18.9 | -29.6 | 2020-03-23 | -29.3 | -28.0 | -20.3 | 7.52 |
| TSLA_2010-07+ | 100_QQQ | 2010-07-01 | 19.4 | 20.7 | -35.1 | 2022-11-03 | -27.6 | -23.5 | -34.8 | 18.34 |
| TSLA_2010-07+ | 50/50_SPY/BRK | 2010-07-01 | 14.3 | 16.8 | -31.7 | 2020-03-23 | -31.2 | -29.0 | -18.9 | 8.92 |
| TSLA_2010-07+ | 60/40_SPY/BRK | 2010-07-01 | 14.5 | 16.7 | -32.1 | 2020-03-23 | -31.6 | -29.2 | -18.9 | 9.15 |
| TSLA_2010-07+ | target_80/20 | 2010-07-01 | 21.8 | 19.3 | -35.0 | 2020-03-23 | -33.8 | -23.8 | -24.3 | 25.09 |
| TSLA_2010-07+ | 60/40_SPY/bonds | 2010-07-01 | 9.9 | 10.3 | -21.8 | 2020-03-23 | -21.3 | -18.3 | -17.5 | 4.70 |
| ARKK_2014-11+ | cur_a_cash | 2014-11-03 | 16.5 | 15.4 | -28.4 | 2022-10-14 | -21.3 | -21.8 | -24.6 | 6.29 |
| ARKK_2014-11+ | cur_b_qqq | 2014-11-03 | 22.5 | 22.2 | -38.2 | 2022-10-14 | -29.5 | -28.8 | -34.6 | 11.53 |
| ARKK_2014-11+ | cur_c_spy | 2014-11-03 | 21.0 | 20.9 | -34.8 | 2022-10-14 | -31.4 | -26.9 | -30.2 | 9.90 |
| ARKK_2014-11+ | 100_VOO/SPY | 2014-11-03 | 13.5 | 17.5 | -34.0 | 2020-03-23 | -33.0 | -30.0 | -19.7 | 4.57 |
| ARKK_2014-11+ | 100_BRK-B | 2014-11-03 | 12.7 | 18.9 | -29.6 | 2020-03-23 | -29.3 | -28.0 | -19.2 | 4.21 |
| ARKK_2014-11+ | 100_QQQ | 2014-11-03 | 18.0 | 21.9 | -35.1 | 2022-11-03 | -27.6 | -23.5 | -34.8 | 7.31 |
| ARKK_2014-11+ | 50/50_SPY/BRK | 2014-11-03 | 13.4 | 17.0 | -31.7 | 2020-03-23 | -31.2 | -29.0 | -18.9 | 4.52 |
| ARKK_2014-11+ | 60/40_SPY/BRK | 2014-11-03 | 13.4 | 16.9 | -32.1 | 2020-03-23 | -31.6 | -29.2 | -18.9 | 4.55 |
| ARKK_2014-11+ | target_80/20 | 2014-11-03 | 18.1 | 19.2 | -32.9 | 2020-03-23 | -31.6 | -27.3 | -26.4 | 7.43 |
| ARKK_2014-11+ | 60/40_SPY/bonds | 2014-11-03 | 8.8 | 10.8 | -21.8 | 2020-03-23 | -21.3 | -18.3 | -17.5 | 2.77 |

CAGR, volatility, max drawdown and worst 1m/3m/12m returns are all in %; `end_multiple` is the final value over initial value. Highlights: since 1999-03 (SYN) the current portfolios gave CAGR 13.0% (a), 16.3% (b), 15.6% (c) with max drawdowns -46%, -65%, -59% against 100% SPY 8.5% / -55%, BRK-B 9.1% / -54%, QQQ 10.6% / -83%, target 80/20 13.3% / -58%, 60/40 SPY/bonds 6.8% / -36%. Since 2010-07 (TSLA) the same portfolios had CAGR 20.5% / 27.3% / 25.8% with max drawdowns -26% / -34% / -36%, versus SPY 15.0% / -34%, BRK-B 13.1% / -30%, target 21.8% / -35%. Since 2014-11 (ARKK) they had 16.5% / 22.6% / 21.0% and -28% / -38% / -35%. Cash helped drawdowns: cur(a) had the lowest max drawdown of the three, at the price of the lowest CAGR. The ranking by CAGR is mostly NVDA and the proxy's contribution (SPCX proxy and NVDA are about 29% of the current portfolios), which is a hindsight outcome; in the worst 12 months the current portfolios lost 22-54% depending on sample, versus about 17-20% (2010+) and 47-50% (1999+) for 100% SPY or BRK-B.

## E. Hypothetical shock table: actual asset moves applied to the current dollars

Each asset's actual peak-to-trough return in the S&P episode (peak/trough dates below) applied to the current weights of the US$7,000 portfolio ((a/b/c) as defined; target 80/20 and 50/50 SPY/BRK on the same US$7,000). Dollar loss (percent loss). SPCX is mapped to each proxy in turn; "flat -50%" assumes SPCX loses exactly half in every scenario for reference. n/a: proxy did not exist at the time.

| index | peak | trough | SP_depth |
|---|---|---|---|
| dot-com | 2000-03-24 | 2002-10-09 | -47.5 |
| 2008 | 2007-10-09 | 2009-03-09 | -55.2 |
| 2022 | 2022-01-03 | 2022-10-12 | -24.5 |
| 2025 | 2025-02-19 | 2025-04-08 | -18.7 |

| portfolio | spcx_proxy | dot-com | 2008 | 2022 | 2025 |
|---|---|---|---|---|---|
| 50/50_SPY/BRK | - | -758 (-10.8%) | -3,454 (-49.3%) | -1,286 (-18.4%) | -590 (-8.4%) |
| cur_a_cash | ARKK | n/a | n/a | -1,662 (-23.7%) | -786 (-11.2%) |
| cur_a_cash | SYN | -1,457 (-20.8%) | -2,748 (-39.3%) | -1,545 (-22.1%) | -747 (-10.7%) |
| cur_a_cash | TSLA | n/a | n/a | -1,491 (-21.3%) | -787 (-11.2%) |
| cur_a_cash | flat -50% | -999 (-14.3%) | -2,531 (-36.2%) | -1,534 (-21.9%) | -902 (-12.9%) |
| cur_b_qqq | ARKK | n/a | n/a | -2,450 (-35.0%) | -1,310 (-18.7%) |
| cur_b_qqq | SYN | -3,365 (-48.1%) | -3,933 (-56.2%) | -2,333 (-33.3%) | -1,271 (-18.2%) |
| cur_b_qqq | TSLA | n/a | n/a | -2,280 (-32.6%) | -1,311 (-18.7%) |
| cur_b_qqq | flat -50% | -2,907 (-41.5%) | -3,717 (-53.1%) | -2,323 (-33.2%) | -1,426 (-20.4%) |
| cur_c_spy | ARKK | n/a | n/a | -2,226 (-31.8%) | -1,216 (-17.4%) |
| cur_c_spy | SYN | -2,550 (-36.4%) | -4,017 (-57.4%) | -2,109 (-30.1%) | -1,177 (-16.8%) |
| cur_c_spy | TSLA | n/a | n/a | -2,055 (-29.4%) | -1,217 (-17.4%) |
| cur_c_spy | flat -50% | -2,092 (-29.9%) | -3,801 (-54.3%) | -2,098 (-30.0%) | -1,332 (-19.0%) |
| target_80/20 | ARKK | n/a | n/a | -2,052 (-29.3%) | -1,121 (-16.0%) |
| target_80/20 | SYN | -2,526 (-36.1%) | -3,965 (-56.6%) | -1,970 (-28.1%) | -1,093 (-15.6%) |
| target_80/20 | TSLA | n/a | n/a | -1,932 (-27.6%) | -1,121 (-16.0%) |
| target_80/20 | flat -50% | -2,206 (-31.5%) | -3,814 (-54.5%) | -1,962 (-28.0%) | -1,202 (-17.2%) |

For the 2022 scenario the loss on the current portfolio is US$1,491-2,450 (21-35%) depending on the version of "other" and the proxy; the target 80/20 would have lost US$1,932-2,052 (28-29%) and 50/50 SPY/BRK US$1,286 (18.4%). In the 2025 episode: current US$747-1,311 (11-19%), target US$1,093-1,121 (16%), 50/50 US$590 (8.4%). In 2008 (SYN or flat -50% only) the current portfolio would have lost US$2,531-4,017 (36-57%) against 49% for 50/50. In the dot-com scenario cur(a) would have lost US$1,457 (21%), cur(c) US$2,550 (36%), and 50/50 SPY/BRK just US$758 (11%), because Berkshire gained 26% over that period. 50/50 SPY/BRK has a smaller loss than all three current versions in 2000-02, 2022 and 2025, but a larger one than cur(a) (36-39%) in 2008, when cash held its value and Berkshire fell 43.5%. These numbers use realised paths of assets that survived; they say nothing about a loss that is larger than anything in the data (a single-stock such as SPCX can go to zero).

## Deviations and implementation notes

- No engine (`DCABacktest`) used: everything is buy-and-hold or fixed-weight returns on closes. No decisions are made, so no lookahead question arises; rebalances reset at the previous close.
- "Dot-com 2000-02" is taken from the detected S&P episode (peak 2000-03-24, trough 2002-10-09), not a hard-coded date; likewise the other three shock episodes (2007-10 > 2009-03, 2022-01 > 2022-10, 2025-02 > 2025-04).
- QQQ drawdown-related pre-1999 values are n/a; `^NDX` price index was not used.
- S&P fund series spliced as VOO > SPY > VFINX; so "SPY" in the portfolio tables is VOO from 2010-09.
- Cash earns 0% (T-bills have yielded several percent in parts of the sample; this flatters equity-heavy portfolios in relative terms and understates cur(a) returns).
