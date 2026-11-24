"""EXP9: does the day of the month you buy on matter for a monthly DCA?

Money for month m arrives on its first trading day. Each strategy spends it once
inside month m, at a daily close. While the money waits it earns T-bill interest
(FF_RF, last daily rate carried forward after the French data end).

Strategies
- calendar day d (1..31): first trading day on or after day d of the month,
  else the month's last trading day
- k-th trading day (1..21), capped at the last trading day
- hindsight best (lowest close of the month) and worst (highest close);
  for SPY also the lowest intraday low
- realistic "wait for a dip" rules, decided only from closes already seen:
  * limit x%: buy at the limit price on the first close at or below
    (first-day close)*(1-x); otherwise at the last trading day's close
  * down day: after the first close down >= 1% on the day, buy at the next
    close; otherwise last day
  * 3 down days in a row: buy at the next close; otherwise last day

Metric: for rolling windows of H months, terminal wealth of the strategy divided
by terminal wealth of buying on the first trading day (same deposits, same end
date, so only the purchase prices differ), and the matching IRR difference.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data', 'ohlcv_daily')
OUT = os.path.join(ROOT, 'docs', 'dca', 'exp9_buy_day')
SERIES = {
    'FF_MKT': 'US market (French, 1926-2026)',
    'VFINX': 'S&P 500 fund (1980-2026)',
    'SPY': 'SPY (1993-2026)',
    'QQQ': 'QQQ (1999-2026)',
    'BRK-B': 'BRK-B (1996-2026)',
}
HORIZONS = [10, 20]
LIMITS = [0.01, 0.02, 0.03, 0.05]


def load(ticker):
    df = pd.read_csv(os.path.join(DATA, f'{ticker}.csv'), index_col='date', parse_dates=True)
    return df[['low', 'close']].dropna()


def rf_index(index):
    rf = load('FF_RF')['close']
    daily = rf.pct_change().fillna(0.0)
    daily = daily.reindex(daily.index.union(index)).ffill()
    last = daily.loc[:rf.index[-1]].iloc[-21:].mean()
    daily.loc[daily.index > rf.index[-1]] = last
    return (1 + daily).cumprod().reindex(index)


def month_groups(df):
    keys = df.index.to_period('M')
    groups = [g for _, g in df.groupby(keys)]
    # drop a partial first month (data starting after day 5) and the current month
    if groups[0].index[0].day > 5:
        groups = groups[1:]
    if groups[-1].index[-1] >= df.index[-1] and groups[-1].index[-1].day < 25:
        groups = groups[:-1]
    return groups


def strategies(g, rf, has_low):
    """Units bought with $1 that arrived on the month's first trading day, per strategy."""
    c = g['close'].to_numpy()
    r = rf.loc[g.index].to_numpy() / rf.loc[g.index[0]]
    days = g.index.day.to_numpy()
    n = len(c)
    out = {}

    def at(i, price=None):
        return r[i] / (c[i] if price is None else price)

    for d in range(1, 32):
        idx = np.nonzero(days >= d)[0]
        out[f'cal_{d:02d}'] = at(idx[0] if len(idx) else n - 1)
    for k in range(1, 22):
        out[f'td_{k:02d}'] = at(min(k, n) - 1)
    out['hindsight_low'] = at(int(np.argmin(c)))
    out['hindsight_high'] = at(int(np.argmax(c)))
    if has_low:
        lo = g['low'].to_numpy()
        j = int(np.argmin(lo))
        out['hindsight_intraday_low'] = r[j] / lo[j]
    for x in LIMITS:
        lim = c[0] * (1 - x)
        hit = np.nonzero(c[1:] <= lim)[0]
        out[f'limit_{int(x * 100)}pct'] = at(hit[0] + 1, lim) if len(hit) else at(n - 1)
    ret = np.r_[0.0, c[1:] / c[:-1] - 1]
    hit = np.nonzero(ret <= -0.01)[0]
    out['after_1pct_down_day'] = at(min(hit[0] + 1, n - 1)) if len(hit) else at(n - 1)
    down = ret < 0
    run = np.nonzero(down[2:] & down[1:-1] & down[:-2])[0]
    out['after_3_down_days'] = at(min(run[0] + 3, n - 1)) if len(run) else at(n - 1)
    return out


def money_duration(t_flow, t_end, irr):
    """Value-weighted average time each dollar is invested; d(ln W) = D * d(irr)."""
    g = np.log1p(irr)
    w = np.exp(g * (t_end - t_flow))
    return float(np.sum(w * (t_end - t_flow)) / np.sum(w))


def irr_monthly(t_flow, t_end, final):
    lo, hi = -0.99, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = np.sum(np.exp(np.log1p(mid) * (t_end - t_flow)))
        lo, hi = (mid, hi) if v < final else (lo, mid)
    return 0.5 * (lo + hi)


def analyse(ticker):
    df = load(ticker)
    has_low = ticker in ('SPY', 'QQQ', 'BRK-B')
    rf = rf_index(df.index)
    groups = month_groups(df)
    units = pd.DataFrame([strategies(g, rf, has_low) for g in groups])
    first = np.array([g.index[0] for g in groups])
    last_close = np.array([g['close'].iloc[-1] for g in groups])
    last_date = np.array([g.index[-1] for g in groups])
    t = np.array([(d - first[0]).days / 365.25 for d in first])
    t_end_all = np.array([(d - first[0]).days / 365.25 for d in last_date])

    # average price advantage per month vs day 1 (extra units bought)
    per_month = units.div(units['td_01'], axis=0) - 1

    rows = []
    cum = np.vstack([np.zeros(units.shape[1]), units.cumsum().to_numpy()])
    for H in HORIZONS:
        M = H * 12
        n_win = len(groups) - M + 1
        if n_win < 12:
            continue
        s = np.arange(n_win)
        tot = cum[s + M] - cum[s]                       # (n_win, n_strat)
        base = tot[:, units.columns.get_loc('td_01')]
        ratio = tot / base[:, None]
        # IRR of the day-1 strategy per window and its money duration
        dur = np.empty(n_win)
        for i in s:
            tf = t[i:i + M]
            te = t_end_all[i + M - 1]
            final = base[i] * last_close[i + M - 1]
            irr0 = irr_monthly(tf, te, final)
            dur[i] = money_duration(tf, te, irr0)
        dirr = np.log(ratio) / dur[:, None]
        # non-overlapping windows starting at the first month
        no = s[::M]
        for j, name in enumerate(units.columns):
            rows.append({
                'series': ticker, 'horizon': H, 'strategy': name, 'n_windows': n_win,
                'n_nonoverlap': len(no),
                'ratio_median': np.median(ratio[:, j]), 'ratio_p10': np.percentile(ratio[:, j], 10),
                'ratio_p90': np.percentile(ratio[:, j], 90), 'ratio_min': ratio[:, j].min(),
                'ratio_max': ratio[:, j].max(), 'beat_day1': np.mean(ratio[:, j] > 1),
                'dirr_median_pp': 100 * np.median(dirr[:, j]),
                'dirr_p10_pp': 100 * np.percentile(dirr[:, j], 10),
                'dirr_p90_pp': 100 * np.percentile(dirr[:, j], 90),
                'nonoverlap_ratio_mean': ratio[no, j].mean(),
            })
    full = pd.DataFrame(rows)
    pm = pd.DataFrame({
        'series': ticker, 'strategy': units.columns,
        'months': len(groups),
        'mean_extra_units_pct': 100 * per_month.mean().to_numpy(),
        'median_extra_units_pct': 100 * per_month.median().to_numpy(),
        'share_months_better': (per_month > 0).mean().to_numpy(),
    })
    period = f'{first[0]:%Y-%m} to {last_date[-1]:%Y-%m}'
    return full, pm, period


def label(name):
    if name.startswith('cal_'):
        return f'calendar day {int(name[4:])}'
    if name.startswith('td_'):
        return f'trading day {int(name[3:])}'
    return name.replace('_', ' ')


def main():
    os.makedirs(OUT, exist_ok=True)
    fulls, pms, periods = [], [], {}
    for tk in SERIES:
        f, p, per = analyse(tk)
        fulls.append(f)
        pms.append(p)
        periods[tk] = per
        print(tk, per, 'done')
    full = pd.concat(fulls, ignore_index=True)
    pm = pd.concat(pms, ignore_index=True)
    full.round(5).to_csv(os.path.join(OUT, 'windows_vs_day1.csv'), index=False)
    pm.round(4).to_csv(os.path.join(OUT, 'per_month_vs_day1.csv'), index=False)

    # figure 1: 10y median IRR difference by calendar day
    fig, ax = plt.subplots(figsize=(10, 5))
    for tk in SERIES:
        sub = full[(full.series == tk) & (full.horizon == 10) & full.strategy.str.startswith('cal_')]
        ax.plot(range(1, 32), sub['dirr_median_pp'].to_numpy(), marker='o', ms=3, label=SERIES[tk])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Calendar day of the month you buy on')
    ax.set_ylabel('Median 10y DCA IRR minus buying on day 1 (pp/yr)')
    ax.set_title('Fixed buy day vs buying on the first trading day (10-year rolling DCA)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_by_calendar_day.png'), dpi=130)
    plt.close(fig)

    # figure 2: rules vs hindsight
    keep = ['hindsight_low', 'hindsight_intraday_low', 'limit_1pct', 'limit_2pct', 'limit_3pct',
            'limit_5pct', 'after_1pct_down_day', 'after_3_down_days', 'cal_15', 'td_21', 'hindsight_high']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.16
    for i, tk in enumerate(SERIES):
        sub = full[(full.series == tk) & (full.horizon == 10)].set_index('strategy')
        vals = [sub['dirr_median_pp'].get(k, np.nan) for k in keep]
        ax.barh(np.arange(len(keep)) + i * width, vals, height=width, label=SERIES[tk])
    ax.set_yticks(np.arange(len(keep)) + 2 * width)
    ax.set_yticklabels([label(k) for k in keep])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Median 10y DCA IRR minus buying on day 1 (pp/yr)')
    ax.set_title('Waiting-for-a-dip rules vs perfect hindsight')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_rules_vs_hindsight.png'), dpi=130)
    plt.close(fig)

    # findings tables
    def table(h, names):
        sub = full[(full.horizon == h) & full.strategy.isin(names)]
        piv = sub.pivot(index='strategy', columns='series', values='dirr_median_pp').reindex(names)
        piv = piv[[c for c in SERIES if c in piv.columns]]
        piv.index = [label(n) for n in piv.index]
        return piv.round(2).to_markdown()

    cal = [f'cal_{d:02d}' for d in range(1, 32)]
    rules = ['hindsight_low', 'hindsight_intraday_low', 'hindsight_high', 'limit_1pct', 'limit_2pct',
             'limit_3pct', 'limit_5pct', 'after_1pct_down_day', 'after_3_down_days']
    cal_stats = full[(full.horizon == 10) & full.strategy.isin(cal)].groupby('series')['dirr_median_pp']
    spread = (cal_stats.max() - cal_stats.min()).round(3)
    beat = full[(full.horizon == 10) & full.strategy.isin(rules)].pivot(
        index='strategy', columns='series', values='beat_day1').reindex(rules)
    beat.index = [label(n) for n in beat.index]
    pmt = pm[pm.strategy.isin(['hindsight_low', 'hindsight_high', 'limit_2pct', 'after_1pct_down_day', 'cal_15'])]
    pmt = pmt.pivot(index='strategy', columns='series', values='mean_extra_units_pct')
    pmt.index = [label(n) for n in pmt.index]

    lines = ['# EXP9: which day of the month to buy', '',
             'Money arrives on the first trading day of each month and is spent once that month at a '
             'daily close; idle cash earns T-bill interest. Numbers are the median difference in '
             'annualised money-weighted return (pp/yr) versus buying on the first trading day, over '
             'rolling 10-year windows unless stated. Samples: ' +
             '; '.join(f'{k} {v}' for k, v in periods.items()) + '.', '',
             '## Fixed calendar day (10y, pp/yr vs day 1)', '', table(10, cal), '',
             'Spread between the best and worst fixed day (pp/yr): ' +
             ', '.join(f'{k} {v}' for k, v in spread.items()), '',
             '## Hindsight and dip-waiting rules (10y, pp/yr vs day 1)', '', table(10, rules), '',
             '## Same, 20-year windows', '', table(20, rules), '',
             '## Share of 10y windows that beat buying on day 1', '', beat.round(2).to_markdown(), '',
             '## Average extra shares per month vs day 1 (%)', '', pmt.round(2).to_markdown(), '',
             'Caveats: closes only for the dip rules (limit fills are taken at the limit price on the '
             'first close at or below it, which misses intraday touches); FF_MKT and VFINX have '
             'open == close; overlapping windows; no costs (all strategies trade once a month).']
    with open(os.path.join(OUT, 'findings.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
