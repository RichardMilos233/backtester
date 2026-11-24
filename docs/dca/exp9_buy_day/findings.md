# EXP9: which day of the month to buy

Money arrives on the first trading day of each month and is spent once that month at a daily close; idle cash earns T-bill interest. Numbers are the median difference in annualised money-weighted return (pp/yr) versus buying on the first trading day, over rolling 10-year windows unless stated. Samples: FF_MKT 1926-07 to 2026-09; VFINX 1980-01 to 2026-10; SPY 1993-02 to 2026-10; QQQ 1999-04 to 2026-10; BRK-B 1996-06 to 2026-10.

## Fixed calendar day (10y, pp/yr vs day 1)

|                 |   FF_MKT |   VFINX |   SPY |   QQQ |   BRK-B |
|:----------------|---------:|--------:|------:|------:|--------:|
| calendar day 1  |     0    |    0    |  0    |  0    |    0    |
| calendar day 2  |    -0.02 |   -0.02 | -0.01 | -0.01 |    0    |
| calendar day 3  |    -0.03 |   -0.02 | -0.01 | -0.01 |   -0.01 |
| calendar day 4  |    -0.03 |   -0.01 | -0.01 | -0.02 |   -0.01 |
| calendar day 5  |    -0.03 |   -0.02 | -0.02 | -0.03 |   -0.02 |
| calendar day 6  |    -0.04 |   -0.02 | -0.02 | -0.02 |   -0.04 |
| calendar day 7  |    -0.04 |   -0.01 | -0.01 |  0.01 |   -0.01 |
| calendar day 8  |    -0.05 |   -0.02 | -0.03 | -0.01 |   -0.01 |
| calendar day 9  |    -0.05 |   -0.01 |  0.01 |  0    |   -0.01 |
| calendar day 10 |    -0.06 |   -0    |  0    | -0.01 |   -0.01 |
| calendar day 11 |    -0.06 |   -0.01 | -0.01 | -0.02 |   -0.03 |
| calendar day 12 |    -0.06 |   -0.02 | -0.02 | -0.03 |   -0.02 |
| calendar day 13 |    -0.06 |   -0.03 | -0.03 | -0.05 |   -0    |
| calendar day 14 |    -0.06 |   -0.03 | -0.02 | -0.04 |    0.02 |
| calendar day 15 |    -0.07 |   -0.04 | -0.05 | -0.06 |    0.03 |
| calendar day 16 |    -0.07 |   -0.04 | -0.06 | -0.08 |    0    |
| calendar day 17 |    -0.07 |   -0.04 | -0.07 | -0.08 |   -0.02 |
| calendar day 18 |    -0.07 |   -0.06 | -0.07 | -0.09 |   -0.03 |
| calendar day 19 |    -0.06 |   -0.04 | -0.06 | -0.07 |    0.01 |
| calendar day 20 |    -0.06 |   -0.03 | -0.04 | -0.03 |    0.04 |
| calendar day 21 |    -0.05 |   -0.03 | -0.03 | -0.04 |    0.03 |
| calendar day 22 |    -0.05 |   -0.02 | -0.05 | -0.06 |    0.04 |
| calendar day 23 |    -0.04 |   -0.02 | -0.04 | -0.08 |    0.06 |
| calendar day 24 |    -0.05 |   -0.03 | -0.03 | -0.08 |    0.02 |
| calendar day 25 |    -0.05 |   -0.02 | -0.03 | -0.09 |    0.02 |
| calendar day 26 |    -0.04 |   -0.04 | -0.04 | -0.11 |    0    |
| calendar day 27 |    -0.04 |   -0.03 | -0.04 | -0.1  |   -0.03 |
| calendar day 28 |    -0.05 |   -0.04 | -0.05 | -0.11 |   -0.03 |
| calendar day 29 |    -0.05 |   -0.05 | -0.05 | -0.11 |   -0.07 |
| calendar day 30 |    -0.07 |   -0.06 | -0.06 | -0.12 |   -0.09 |
| calendar day 31 |    -0.08 |   -0.07 | -0.06 | -0.12 |   -0.1  |

Spread between the best and worst fixed day (pp/yr): BRK-B 0.16, FF_MKT 0.08, QQQ 0.132, SPY 0.081, VFINX 0.071

## Hindsight and dip-waiting rules (10y, pp/yr vs day 1)

|                        |   FF_MKT |   VFINX |   SPY |   QQQ |   BRK-B |
|:-----------------------|---------:|--------:|------:|------:|--------:|
| hindsight low          |     0.45 |    0.48 |  0.58 |  0.62 |    0.62 |
| hindsight intraday low |   nan    |  nan    |  0.7  |  0.75 |    0.81 |
| hindsight high         |    -0.49 |   -0.49 | -0.51 | -0.61 |   -0.58 |
| limit 1pct             |    -0.16 |   -0.12 | -0.14 | -0.17 |   -0.11 |
| limit 2pct             |    -0.12 |   -0.07 | -0.07 | -0.16 |   -0.09 |
| limit 3pct             |    -0.12 |   -0.08 | -0.06 | -0.12 |   -0.07 |
| limit 5pct             |    -0.12 |   -0.11 | -0.12 | -0.14 |   -0.04 |
| after 1pct down day    |    -0.07 |   -0.03 | -0.03 | -0.08 |    0.01 |
| after 3 down days      |    -0.08 |   -0.03 | -0.01 | -0.01 |   -0.02 |

## Same, 20-year windows

|                        |   FF_MKT |   VFINX |   SPY |   QQQ |   BRK-B |
|:-----------------------|---------:|--------:|------:|------:|--------:|
| hindsight low          |     0.21 |    0.23 |  0.25 |  0.26 |    0.27 |
| hindsight intraday low |   nan    |  nan    |  0.31 |  0.33 |    0.36 |
| hindsight high         |    -0.21 |   -0.22 | -0.22 | -0.27 |   -0.25 |
| limit 1pct             |    -0.06 |   -0.05 | -0.05 | -0.07 |   -0.05 |
| limit 2pct             |    -0.05 |   -0.03 | -0.02 | -0.06 |   -0.04 |
| limit 3pct             |    -0.04 |   -0.03 | -0.02 | -0.04 |   -0.03 |
| limit 5pct             |    -0.05 |   -0.04 | -0.03 | -0.04 |   -0.02 |
| after 1pct down day    |    -0.02 |   -0.01 | -0.01 | -0.03 |   -0    |
| after 3 down days      |    -0.03 |   -0.01 |  0    |  0    |   -0.01 |

## Share of 10y windows that beat buying on day 1

|                        |   BRK-B |   FF_MKT |   QQQ |   SPY |   VFINX |
|:-----------------------|--------:|---------:|------:|------:|--------:|
| hindsight low          |    1    |     1    |  1    |  1    |    1    |
| hindsight intraday low |    1    |   nan    |  1    |  1    |  nan    |
| hindsight high         |    0    |     0    |  0    |  0    |    0    |
| limit 1pct             |    0    |     0    |  0    |  0    |    0    |
| limit 2pct             |    0.05 |     0.02 |  0    |  0.24 |    0.11 |
| limit 3pct             |    0.09 |     0.05 |  0    |  0.29 |    0.14 |
| limit 5pct             |    0.24 |     0.04 |  0.05 |  0.23 |    0.13 |
| after 1pct down day    |    0.53 |     0.16 |  0.17 |  0.31 |    0.21 |
| after 3 down days      |    0.35 |     0.13 |  0.47 |  0.41 |    0.3  |

## Average extra shares per month vs day 1 (%)

|                     |   BRK-B |   FF_MKT |   QQQ |   SPY |   VFINX |
|:--------------------|--------:|---------:|------:|------:|--------:|
| after 1pct down day |   -0.14 |    -0.37 | -0.36 | -0.19 |   -0.15 |
| calendar day 15     |   -0.11 |    -0.33 | -0.28 | -0.24 |   -0.2  |
| hindsight high      |   -3.52 |    -2.95 | -4.07 | -2.74 |   -2.75 |
| hindsight low       |    3.58 |     3.11 |  4.53 |  3.07 |    3.02 |
| limit 2pct          |   -0.62 |    -0.69 | -1.08 | -0.43 |   -0.43 |

Caveats: closes only for the dip rules (limit fills are taken at the limit price on the first close at or below it, which misses intraday touches); FF_MKT and VFINX have open == close; overlapping windows; no costs (all strategies trade once a month).
