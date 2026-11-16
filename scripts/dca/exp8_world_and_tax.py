"""EXP8 - should a world equity index be part of the core, and what does Singapore tax do to the fund choice?

A. longest US vs international record (calendar years, 5y blocks, rolling gaps, regime switches)
B. world-index DCA: rolling windows, non-overlapping windows, path risk, stationary block bootstrap, scenarios
C. how different is a world index from the S&P today (beta / R2 / implied US weight / tracking difference)
D. Singapore tax and vehicle comparison (withholding, Irish UCITS, USD/SGD, new SGX lines)
E. three-way core comparison net of Singapore tax
F. (findings.md, written by hand from the tables this script produces)

Run from the repo root:  python scripts/dca/exp8_world_and_tax.py
Outputs: docs/dca/exp8_world_and_tax/*.csv, *.png, findings_tables.md (all tables, auto-generated)

Conventions: decision dates = first trading day of each month, buys at that day's open + 5 bp slippage (rolling_dca /
DCABacktest). Fixed-fraction DCA = each month's contribution is split by constant fractions, never rebalanced (so the
mix drifts like a real never-sell plan). Constant fractions need no signal, hence no lookahead. The bootstrap works
on monthly returns (deposit at month start, return over the month, 5 bp slippage) with the same fixed-fraction rule.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dca import DCABacktest, dca_metrics, fixed_buy, load_prices, month_starts, rolling_dca

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca', 'exp8_world_and_tax')
os.makedirs(OUT, exist_ok=True)
DPI = 130
SLIP = 0.0005
R_PATHS = 2000
BLOCK = 18          # mean block length, months
SEED = 8
TABLES = []         # (title, df, note) in order of creation -> findings_tables.md


# ------------------------------------------------------------------------------------------------ helpers
def md(df, floatfmt=None):
    cols = [df.index.name or ''] + [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for idx, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                vals.append('' if not np.isfinite(v) else f'{v:.4g}' if abs(v) < 1e-3 and v != 0 else f'{v:.3f}'.rstrip('0').rstrip('.') if abs(v) < 10 else f'{v:.1f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join([str(idx)] + vals) + ' |')
    return '\n'.join(lines)


def save(name, df, title, note='', index=True, dec=3):
    d = df.copy()
    d.to_csv(os.path.join(OUT, name + '.csv'), index=index, float_format=f'%.{dec}f')
    TABLES.append((name, title, d if index else d.set_index(d.columns[0]), note))
    return d


def pct(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def monthly_returns(closes):
    m = closes.resample('ME').last()
    return m.pct_change().iloc[1:]


def ann_ret(c):
    """CAGR from a level series with a DatetimeIndex."""
    days = (c.index[-1] - c.index[0]).days
    return (c.iloc[-1] / c.iloc[0]) ** (365.25 / days) - 1


def px(tickers, start=None, end=None, how='inner'):
    d = load_prices(tickers, start, end, how)
    return d['open'], d['close']


def synth(ret_daily, drag_ann=0.0):
    """Level series (base 100, open==close) from daily returns minus a constant annual drag."""
    r = ret_daily - drag_ann / 252.0
    lv = 100 * (1 + r.fillna(0)).cumprod()
    return lv


def ppy(x):
    return 100 * x


# ------------------------------------------------------------------------------------------------ A
def roll_ann(r, n):
    return np.exp(np.log1p(r).rolling(n).sum() * 12 / n) - 1


def zigzag(logw, thr):
    """Pivot indices of a zigzag filter with reversal threshold `thr` (log units). Returns [0, pivots..., last]."""
    a = np.asarray(logw)
    hi = lo = 0
    direction = 0
    piv = [0]
    ext = 0
    for i, x in enumerate(a):
        if direction == 0:
            if x > a[hi]:
                hi = i
            if x < a[lo]:
                lo = i
            if x - a[lo] >= thr:
                direction, ext = 1, i
            elif a[hi] - x >= thr:
                direction, ext = -1, i
        elif direction == 1:
            if x > a[ext]:
                ext = i
            elif a[ext] - x >= thr:
                piv.append(ext)
                direction, ext = -1, i
        else:
            if x < a[ext]:
                ext = i
            elif x - a[ext] >= thr:
                piv.append(ext)
                direction, ext = 1, i
    piv.append(len(a) - 1)      # the last (unconfirmed) retracement is folded into the final leg
    out = []
    for p in piv:
        if not out or out[-1] != p:
            out.append(p)
    return out


def regime_table(r_us, r_ex, thr):
    lw = np.log1p(r_us).cumsum() - np.log1p(r_ex).cumsum()
    lw = pd.concat([pd.Series([0.0], index=[lw.index[0] - pd.offsets.MonthEnd(1)]), lw])
    p = zigzag(lw.values, thr)
    rows = []
    for a, b in zip(p[:-1], p[1:]):
        ch = lw.iloc[b] - lw.iloc[a]
        rows.append({'from': lw.index[a].strftime('%Y-%m'), 'to': lw.index[b].strftime('%Y-%m'),
                     'months': b - a, 'leader': 'US' if ch > 0 else 'International',
                     'US_minus_intl_cum_log_pp': 100 * ch})
    return pd.DataFrame(rows)


def part_a():
    print('A ...')
    names = ['FF_MKT', 'FF_DEVXUS', 'VFINX', 'PRITX', 'VWIGX', 'VEA', 'EFA', 'VXUS', 'VWO']
    mon = {}
    for t in names:
        _, c = px([t])
        mon[t] = monthly_returns(c[t])
    # ---- calendar-year table (complete years only; 2026 = partial)
    def cal(r, last_complete=2025):
        g = r.groupby(r.index.year)
        n = g.size()
        ret = g.apply(lambda x: (1 + x).prod() - 1)
        ret = ret[(n == 12)]
        ret.loc[r.index[-1].year] = (1 + r[r.index.year == r.index[-1].year]).prod() - 1 if r.index[-1].month != 12 else np.nan
        return ret.sort_index()
    cy = pd.DataFrame({t: cal(mon[t]) for t in names})
    cy = cy.loc[1980:]
    ydf = (100 * cy).round(1)
    ydf['VFINX-PRITX'] = ydf['VFINX'] - ydf['PRITX']
    ydf['FF_MKT-FF_DEVXUS'] = ydf['FF_MKT'] - ydf['FF_DEVXUS']
    ydf.index.name = 'year'
    save('A1_calendar_year_returns_pct', ydf, 'A1. Calendar-year total returns, USD, %',
         '2026 is year-to-date (FF series to 2026-09-30, funds/ETFs to 2026-11-13). Blank = partial first year. PRITX/VWIGX are active mutual funds net of fees; FF series are frictionless indices.', dec=1)

    # ---- 5-year blocks
    def blocks(r_us, r_ex, first):
        rows = []
        for y in range(first, 2026, 5):
            idx = (r_us.index.year >= y) & (r_us.index.year < y + 5)
            if idx.sum() < 60:
                continue
            a = (1 + r_us[idx]).prod() ** (12 / idx.sum()) - 1
            b = (1 + r_ex[idx]).prod() ** (12 / idx.sum()) - 1
            rows.append({'block': f'{y}-{y + 4}', 'US': 100 * a, 'Intl': 100 * b, 'US_minus_Intl_pp': 100 * (a - b)})
        idx = r_us.index.year >= 2026
        return pd.DataFrame(rows).set_index('block')
    pairs = {'FF_MKT vs FF_DEVXUS': ('FF_MKT', 'FF_DEVXUS', 1991), 'VFINX vs PRITX': ('VFINX', 'PRITX', 1981),
             'VFINX vs VWIGX': ('VFINX', 'VWIGX', 1982)}
    bl = []
    for k, (u, e, f) in pairs.items():
        j = pd.concat([mon[u], mon[e]], axis=1, join='inner')
        b = blocks(j[u], j[e], f)
        b.insert(0, 'pair', k)
        bl.append(b)
    bl = pd.concat(bl)
    save('A2_five_year_blocks', bl, 'A2. Annualised return by 5-year block (US, international, difference in pp/yr)',
         'Blocks start in the first full calendar year after each pair\'s common start. Complete 5-year blocks only (2026 YTD excluded).', dec=1)

    # ---- ETF table (common windows)
    etf_rows = []
    for t in ['VEA', 'EFA', 'VXUS', 'VWO']:
        j = pd.concat([mon['VFINX'], mon[t]], axis=1, join='inner')
        n = len(j)
        a = (1 + j.iloc[:, 0]).prod() ** (12 / n) - 1
        b = (1 + j.iloc[:, 1]).prod() ** (12 / n) - 1
        etf_rows.append({'ETF': t, 'start': j.index[0].strftime('%Y-%m'), 'VFINX_ann': 100 * a, f'ETF_ann': 100 * b,
                         'US_minus_ETF_pp': 100 * (a - b)})
    save('A3_etf_vs_vfinx', pd.DataFrame(etf_rows).set_index('ETF'), 'A3. International ETFs vs VFINX on each ETF\'s own history (annualised %, to 2026-11)',
         'VFINX used as the US leg for all ETFs (monthly returns, through 2026-10 + partial Nov).', dec=2)

    # ---- rolling gaps
    roll = {}
    for k, (u, e, f) in pairs.items():
        j = pd.concat([mon[u], mon[e]], axis=1, join='inner')
        for n in (60, 120):
            roll[(k, n)] = 100 * (roll_ann(j[u], n) - roll_ann(j[e], n))
    # ---- regimes
    reg_rows = []
    reg_tabs = {}
    for k, (u, e, f) in pairs.items():
        j = pd.concat([mon[u], mon[e]], axis=1, join='inner')
        for thr in (0.15, 0.25, 0.40):
            t = regime_table(j[u], j[e], thr)
            reg_rows.append({'pair': k, 'sample': f'{j.index[0]:%Y-%m}..{j.index[-1]:%Y-%m}', 'reversal_threshold': f'{int(thr * 100)}%',
                             'legs': len(t), 'switches_of_leader': len(t) - 1})
            if thr == 0.25:
                reg_tabs[k] = t
    save('A4_regime_counts', pd.DataFrame(reg_rows).set_index('pair'),
         'A4. Regime count: zigzag of cumulative US/international relative wealth',
         'A "leg" is a run in which the relative-wealth ratio moves in one direction; a new leg is confirmed once the ratio retraces by the threshold (15/25/40% of relative wealth). Switches = legs - 1. Results depend on the threshold; the first and last legs may be incomplete.', index=True)
    for k, t in reg_tabs.items():
        save('A5_regimes_' + k.split(' ')[0] + '_' + k.split(' ')[2], t.set_index('from'),
             f'A5. Relative-performance legs, {k} (25% reversal threshold)', 'US_minus_intl_cum_log_pp = change in log(US wealth / intl wealth) x 100 over the leg.', dec=1)
    # share of months when trailing rolling windows had international ahead
    sh = []
    for (k, n), s in roll.items():
        s = s.dropna()
        sh.append({'pair': k, 'window': f'{n // 12}y', 'n_months': len(s), 'pct_intl_ahead': 100 * (s < 0).mean(),
                   'median_gap_pp': s.median(), 'min_gap_pp': s.min(), 'max_gap_pp': s.max(), 'latest_gap_pp': s.iloc[-1]})
    save('A6_rolling_gap_summary', pd.DataFrame(sh).set_index('pair'), 'A6. Rolling annualised US-minus-international gap (pp/yr)',
         'Gap = annualised US return minus annualised international return over the trailing window; negative = international ahead.', dec=1)

    # ---- figure 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for ax, ks, ttl in [(axes[0], ['FF_MKT vs FF_DEVXUS'], 'French factor-library indices (frictionless), 1990-2026'),
                        (axes[1], ['VFINX vs PRITX', 'VFINX vs VWIGX'], 'Mutual funds (net of fees), 1980-2026')]:
        for k, c in zip(ks, ['#1f77b4', '#d62728']):
            ax.plot(roll[(k, 60)].dropna(), color=c, lw=1.4, label=f'{k}: 5y')
            ax.plot(roll[(k, 120)].dropna(), color=c, lw=2.4, ls='--', alpha=.75, label=f'{k}: 10y')
        ax.axhline(0, color='k', lw=.8)
        ax.set_title(ttl)
        ax.set_ylabel('US minus international, pp/yr')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=.3)
        ax.text(0.99, 0.04, 'below zero = international ahead', transform=ax.transAxes, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_rolling_us_minus_intl.png'), dpi=DPI)
    plt.close(fig)
    return reg_rows, roll


# ------------------------------------------------------------------------------------------------ bootstrap engine
def boot_idx(N, R, n, L, rng):
    idx = np.empty((R, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, N, R)
    new = rng.random((R, n)) < 1.0 / L
    rnd = rng.integers(0, N, (R, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], rnd[:, t], (idx[:, t - 1] + 1) % N)
    return idx


def solve_irr(fv, H, lo=-0.3, hi=0.3, iters=60):
    """Monthly IRR g with (1+g)*((1+g)^H-1)/g = fv (deposits at the start of each of H months, value at the end)."""
    fv = np.asarray(fv, dtype=float)
    a = np.full(fv.shape, lo)
    b = np.full(fv.shape, hi)

    def f(g):
        small = np.abs(g) < 1e-9
        gs = np.where(small, 1.0, g)
        val = (1 + g) * np.expm1(H * np.log1p(g)) / gs
        return np.where(small, H, val) - fv
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = f(m) > 0          # value too high -> rate too high
        b = np.where(up, m, b)
        a = np.where(up, a, m)
    g = 0.5 * (a + b)
    return (1 + g) ** 12 - 1


def dca_irr(rets, W, H, first_only=False, slip=SLIP):
    """rets (R,n,k) monthly returns; W (P,k) fixed fractions. Returns annual IRR (R, windows, P)."""
    R, n, k = rets.shape
    L = np.concatenate([np.ones((R, 1, k)), np.cumprod(1 + rets, axis=1)], axis=1)
    S = np.concatenate([np.zeros((R, 1, k)), np.cumsum(1.0 / L[:, :n], axis=1)], axis=1)
    starts = np.array([0]) if first_only else np.arange(0, n - H + 1)
    fv_i = L[:, starts + H, :] * (S[:, starts + H, :] - S[:, starts, :])      # (R,Wn,k)
    fv = np.einsum('rwk,pk->rwp', fv_i, W) / (1 + slip)
    return solve_irr(fv, H)


def max_dd(rets, W, n_months):
    R, n, k = rets.shape
    P = W.shape[0]
    v = np.zeros((R, P, k))
    nav = np.ones((R, P))
    peak = np.ones((R, P))
    dd = np.zeros((R, P))
    for t in range(n_months):
        v = v + W[None, :, :]
        before = v.sum(axis=2)
        v = v * (1 + rets[:, t, None, :])
        nav = nav * v.sum(axis=2) / before
        peak = np.maximum(peak, nav)
        dd = np.minimum(dd, nav / peak - 1)
    return dd


def wmat(portfolios, cols):
    W = np.zeros((len(portfolios), len(cols)))
    for i, w in enumerate(portfolios.values()):
        for t, x in w.items():
            W[i, cols.index(t)] = x
    return W


def summarize_diff(d):
    d = np.asarray(d)
    return {'median': np.median(d), 'mean': d.mean(), 'p5': np.percentile(d, 5), 'p95': np.percentile(d, 95), 'P(>0)': (d > 0).mean()}


def boot_core(arr, cols, portfolios, seed=SEED, R=R_PATHS, L=BLOCK, H=240, n_long=480):
    """arr (N,k) monthly returns. Returns dict of per-path metrics for each portfolio."""
    rng = np.random.default_rng(seed)
    idx = boot_idx(len(arr), R, n_long, L, rng)
    rets = arr[idx]
    W = wmat(portfolios, cols)
    irr_w = dca_irr(rets, W, H)                         # (R, n_long-H+1, P)
    irr20 = irr_w[:, 0, :]
    p10 = np.percentile(irr_w, 10, axis=1)              # within-path p10 across rolling 20y windows
    mn = irr_w.min(axis=1)
    dd = max_dd(rets, W, H)
    return {'names': list(portfolios), 'irr20': irr20, 'p10path': p10, 'minpath': mn, 'dd': dd}


def paired_table(res, pairs, label=''):
    nm = res['names']
    rows = []
    for a, b in pairs:
        ia, ib = nm.index(a), nm.index(b)
        di = 100 * (res['irr20'][:, ia] - res['irr20'][:, ib])
        dp = 100 * (res['p10path'][:, ia] - res['p10path'][:, ib])
        dd = 100 * (res['dd'][:, ia] - res['dd'][:, ib])
        s, p, d = summarize_diff(di), summarize_diff(dp), summarize_diff(dd)
        rows.append({'A minus B': f'{a} - {b}', 'IRR20 diff median pp': s['median'], '90% lo': s['p5'], '90% hi': s['p95'], 'P(A>B)': s['P(>0)'],
                     'path-p10 diff pp': p['median'], 'p10 90% lo': p['p5'], 'p10 90% hi': p['p95'], 'P(p10 A>B)': p['P(>0)'],
                     'maxDD diff pp (+ = A shallower)': d['median'], 'DD 90% lo': d['p5'], 'DD 90% hi': d['p95'], 'P(DD A shallower)': d['P(>0)']})
    return pd.DataFrame(rows).set_index('A minus B')


def level_table(res):
    nm = res['names']
    rows = []
    for i, n in enumerate(nm):
        rows.append({'portfolio': n, 'IRR20 median %': 100 * np.median(res['irr20'][:, i]), 'IRR20 p10 across paths %': 100 * pct(res['irr20'][:, i], 10),
                     'IRR20 p5 %': 100 * pct(res['irr20'][:, i], 5), 'IRR20 min %': 100 * res['irr20'][:, i].min(),
                     'median path-p10 %': 100 * np.median(res['p10path'][:, i]),
                     'median maxDD %': 100 * np.median(res['dd'][:, i]), 'p10 maxDD % (worse tail)': 100 * pct(res['dd'][:, i], 10)})
    return pd.DataFrame(rows).set_index('portfolio')


# ------------------------------------------------------------------------------------------------ rolling DCA helpers
def alloc_for(ms, cols, w):
    a = pd.DataFrame(0.0, index=ms, columns=cols)
    for t, x in w.items():
        a[t] = x
    return a


def rolling_set(opens, closes, portfolios, horizons_y, start=None):
    idx = opens.index if start is None else opens.loc[start:].index
    ms = month_starts(idx)
    out = {}
    for name, w in portfolios.items():
        al = alloc_for(ms, list(opens.columns), w)
        for h in horizons_y:
            r = rolling_dca(opens, closes, al, h * 12)
            if len(r):
                out[(name, h)] = r.set_index('start')['irr']
    return out, ms


def rolling_stats(out, ref, horizons_y, nonover_ms=None):
    rows, dec_rows, no_rows = [], [], []
    names = sorted({k[0] for k in out}, key=lambda n: list(dict.fromkeys(k[0] for k in out)).index(n))
    for h in horizons_y:
        base = out.get((ref, h))
        for n in names:
            s = out.get((n, h))
            if s is None:
                continue
            d = (s - base).dropna() if base is not None else None
            rows.append({'portfolio': n, 'H': h, 'windows': len(s), 'median %': 100 * s.median(), 'p10 %': 100 * pct(s, 10), 'min %': 100 * s.min(),
                         'median diff vs ref pp': 100 * d.median(), '% windows beating ref': 100 * (d > 0).mean() if n != ref else np.nan,
                         'first start': f'{s.index[0]:%Y-%m}', 'last start': f'{s.index[-1]:%Y-%m}'})
            if n != ref:
                for dec, g in d.groupby(d.index.year // 10 * 10):
                    dec_rows.append({'portfolio': n, 'H': h, 'start decade': f'{dec}s', 'windows': len(g), 'median diff pp': 100 * g.median(),
                                     '% beating ref': 100 * (g > 0).mean(), 'median IRR %': 100 * s.loc[g.index].median(), 'ref median IRR %': 100 * base.loc[g.index].median()})
            # non-overlapping windows: start 0, H*12, 2*H*12 ...
            pos = np.arange(0, len(s), h * 12)
            ss = s.iloc[pos]
            bb = base.iloc[pos]
            for (dt, v), b in zip(ss.items(), bb.values):
                no_rows.append({'portfolio': n, 'H': h, 'window': f'{dt:%Y-%m}..{dt + pd.DateOffset(years=h):%Y-%m}', 'IRR %': 100 * v, 'ref IRR %': 100 * b,
                                'diff vs ref pp': 100 * (v - b)})
    return pd.DataFrame(rows), pd.DataFrame(dec_rows), pd.DataFrame(no_rows)


def path_risk(opens, closes, portfolios, start=None):
    idx = opens.index if start is None else opens.loc[start:].index
    ms = month_starts(idx)
    o, c = opens.loc[ms[0]:], closes.loc[ms[0]:]
    rows = []
    for name, w in portfolios.items():
        led = DCABacktest(slippage_rate=SLIP).run(o, c, fixed_buy(w), pd.Series(1.0, index=ms))
        m = dca_metrics(led)
        rows.append({'portfolio': name, 'years': m['years'], 'irr %': 100 * m['irr'], 'wealth multiple': m['wealth_multiple'],
                     'ann vol of nav %': 100 * m['annualized_volatility'], 'max drawdown %': 100 * m['max_drawdown'], 'worst 12m %': 100 * m['worst_12m_return']})
    return pd.DataFrame(rows).set_index('portfolio')


# ------------------------------------------------------------------------------------------------ B
def part_b(world_cap_exus):
    print('B ...')
    res = {}
    # --- FF set 1990+
    o, c = px(['FF_MKT', 'FF_DEV', 'FF_DEVXUS'])
    P1 = {'US only (FF_MKT)': {'FF_MKT': 1.0}, 'Developed world (FF_DEV)': {'FF_DEV': 1.0},
          'US 80 / exUS 20': {'FF_MKT': .8, 'FF_DEVXUS': .2}, 'US 60 / exUS 40': {'FF_MKT': .6, 'FF_DEVXUS': .4},
          'exUS only (FF_DEVXUS)': {'FF_DEVXUS': 1.0}}
    ref1 = 'US only (FF_MKT)'
    out1, ms1 = rolling_set(o, c, P1, [10, 20, 30])
    st1, dec1, no1 = rolling_stats(out1, ref1, [10, 20, 30])
    save('B1_rolling_FF_1990', st1.set_index('portfolio'), 'B1. Rolling monthly DCA, French indices 1990-07..2026-09 (IRR %, all start months)',
         'Windows overlap heavily: the number of independent observations is about 36y/H = 3.6 (10y), 1.8 (20y), 1.2 (30y). Ref = US only. exUS/US mixes are fixed-fraction, never rebalanced.', index=True, dec=2)
    save('B2_by_start_decade_FF_1990', dec1.set_index('portfolio'), 'B2. Same, by start decade (difference vs US only)', '', dec=2)
    save('B3_nonoverlapping_FF_1990', no1.set_index('portfolio'), 'B3. Non-overlapping windows (FF 1990+): each start H years after the previous one', 'Only complete windows.', dec=2)
    risk1 = path_risk(o, c, P1)
    save('B4_path_risk_FF_1990', risk1, 'B4. Full-sample DCABacktest path (1990-07..2026-09, $1/month, one 36-year run)', 'nav = time-weighted value; vol/max drawdown/worst 12m from nav.', dec=2)
    # --- sanity check
    ms = month_starts(o.index)
    k = 60
    start, end = ms[k], ms[k + 120]
    led = DCABacktest(slippage_rate=SLIP).run(o.loc[start:end - pd.Timedelta(days=1)], c.loc[start:end - pd.Timedelta(days=1)], fixed_buy({'FF_MKT': .8, 'FF_DEVXUS': .2}),
                                              pd.Series(1.0, index=ms[k:k + 120]))
    chk = pd.DataFrame({'rolling_dca irr %': [100 * out1[('US 80 / exUS 20', 10)].iloc[k]], 'DCABacktest irr %': [100 * dca_metrics(led)['irr']],
                        'window': [f'{start:%Y-%m}..{end:%Y-%m}']})
    save('B0_spotcheck', chk.set_index('window'), 'B0. Spot check: one 10-year window, rolling_dca vs full DCABacktest', 'US80/exUS20 on FF series.', dec=3)

    # --- fund pairs 1980+
    fund_sets = {}
    for ex in ['PRITX', 'VWIGX']:
        o2, c2 = px(['VFINX', ex])
        P2 = {'US only (VFINX)': {'VFINX': 1.0}, 'US 80 / exUS 20': {'VFINX': .8, ex: .2}, 'US 60 / exUS 40': {'VFINX': .6, ex: .4},
              f'exUS only ({ex})': {ex: 1.0}}
        out2, _ = rolling_set(o2, c2, P2, [10, 20, 30])
        st2, dec2, no2 = rolling_stats(out2, 'US only (VFINX)', [10, 20, 30])
        save(f'B5_rolling_funds_{ex}', st2.set_index('portfolio'), f'B5. Rolling monthly DCA, VFINX / {ex} mutual funds ({o2.index[0]:%Y-%m}..{o2.index[-1]:%Y-%m}), IRR %',
             f'Independent observations: about {(o2.index[-1] - o2.index[0]).days / 365.25:.0f}y/H. Active mutual fund with fees on the international leg.', dec=2)
        save(f'B6_nonoverlap_funds_{ex}', no2.set_index('portfolio'), f'B6. Non-overlapping windows, VFINX / {ex}', '', dec=2)
        save(f'B6b_by_decade_funds_{ex}', dec2.set_index('portfolio'), f'B6b. By start decade, VFINX / {ex}', '', dec=2)
        save(f'B7_path_risk_funds_{ex}', path_risk(o2, c2, P2), f'B7. Full-sample DCABacktest path, VFINX / {ex}', '', dec=2)
        fund_sets[ex] = (o2, c2, P2, out2)

    # --- bootstrap FF
    cols = ['FF_MKT', 'FF_DEV', 'FF_DEVXUS']
    mon = monthly_returns(c[cols])
    res_ff = boot_core(mon.values, cols, P1)
    save('B8_bootstrap_levels_FF', level_table(res_ff), f'B8. Stationary block bootstrap ({R_PATHS} paths, mean block {BLOCK}m), FF 1990-07..2026-09: 20-year monthly DCA outcomes',
         'IRR20 = first 20 years of a 40-year simulated path; "path-p10" = 10th percentile of the 241 rolling 20y windows inside a simulated 40-year path; maxDD of the DCA nav over the 20 years.', dec=2)
    pairs = [('Developed world (FF_DEV)', 'US only (FF_MKT)'), ('US 80 / exUS 20', 'US only (FF_MKT)'), ('US 60 / exUS 40', 'US only (FF_MKT)'),
             ('exUS only (FF_DEVXUS)', 'US only (FF_MKT)')]
    pt_ff = paired_table(res_ff, pairs)
    save('B9_bootstrap_paired_FF', pt_ff, 'B9. Paired bootstrap differences vs US only (FF 1990+), 90% intervals = 5th..95th percentile across paths',
         'The interval reflects block-bootstrap sampling uncertainty of the 36 years observed; it cannot represent regimes never seen.', dec=2)
    # block length sensitivity
    sens = []
    for L in (12, 24, 36):
        r = boot_core(mon.values, cols, P1, L=L, R=1000)
        for a in ['Developed world (FF_DEV)', 'US 80 / exUS 20']:
            ia, ib = r['names'].index(a), r['names'].index(ref1)
            d = summarize_diff(100 * (r['irr20'][:, ia] - r['irr20'][:, ib]))
            sens.append({'block': L, 'portfolio': a, 'IRR20 diff median pp': d['median'], '90% lo': d['p5'], '90% hi': d['p95'], 'P(>US)': d['P(>0)']})
    save('B10_block_length_sensitivity', pd.DataFrame(sens).set_index('block'), 'B10. Block-length sensitivity (1000 paths each), FF 1990+', '', dec=2)

    # --- bootstrap fund pairs
    pt_funds = {}
    res_fund = {}
    for ex, (o2, c2, P2, _) in fund_sets.items():
        cols2 = ['VFINX', ex]
        mon2 = monthly_returns(c2[cols2])
        r2 = boot_core(mon2.values, cols2, P2)
        res_fund[ex] = (r2, mon2)
        base = 'US only (VFINX)'
        pairs2 = [(n, base) for n in r2['names'] if n != base]
        pt = paired_table(r2, pairs2)
        pt_funds[ex] = pt
        save(f'B11_bootstrap_levels_funds_{ex}', level_table(r2), f'B11. Bootstrap levels, VFINX / {ex} ({mon2.index[0]:%Y-%m}..{mon2.index[-1]:%Y-%m}), 20-year DCA', '', dec=2)
        save(f'B12_bootstrap_paired_funds_{ex}', pt, f'B12. Paired bootstrap differences vs VFINX, VFINX / {ex}', '', dec=2)

    # --- scenarios
    sc_all = []
    sc_curves = {}
    grid = [round(0.1 * i, 1) for i in range(11)]
    datasets = {'FF 1990-2026 (FF_MKT / FF_DEVXUS)': (mon[['FF_MKT', 'FF_DEVXUS']].values, ['US', 'EX']),
                'Funds 1980-2026 (VFINX / PRITX)': (res_fund['PRITX'][1].values, ['US', 'EX'])}
    for dname, (arr, cn) in datasets.items():
        gap = (arr[:, 1].mean() - arr[:, 0].mean()) * 12
        d1 = arr[:, 1].mean() - arr[:, 0].mean()
        for sname, delta in [('history as is', 0.0), ('no US edge (equal mean)', d1), ('reversal (US 2pp/yr below exUS)', d1 - 0.02 / 12)]:
            a = arr.copy()
            a[:, 0] = a[:, 0] + delta
            shares = grid + [round(world_cap_exus, 3)]
            port = {f'{s:.3f}': {'US': 1 - s, 'EX': s} for s in shares}
            W = wmat(port, cn)
            rng = np.random.default_rng(SEED + 1)
            idx = boot_idx(len(a), R_PATHS, 240, BLOCK, rng)
            irr = dca_irr(a[idx], W, 240, first_only=True)[:, 0, :]
            ddv = max_dd(a[idx], W, 240)
            base = irr[:, 0]
            for j, s in enumerate(shares):
                d = irr[:, j] - base
                row = {'dataset': dname, 'scenario': sname, 'exUS share': s, 'median IRR %': 100 * np.median(irr[:, j]), 'p10 IRR %': 100 * pct(irr[:, j], 10),
                       'p5 IRR %': 100 * pct(irr[:, j], 5), 'median vs 0% pp': 100 * np.median(d), '90% lo': 100 * pct(d, 5), '90% hi': 100 * pct(d, 95),
                       'P(beats 0% exUS)': (d > 0).mean() if j else np.nan, 'p10 vs 0% pp': 100 * (pct(irr[:, j], 10) - pct(irr[:, 0], 10)),
                       'median maxDD %': 100 * np.median(ddv[:, j]), 'US mean - exUS mean after shift, pp/yr': -100 * 12 * (a[:, 1].mean() - a[:, 0].mean())}
                row['is_world_cap'] = (j == len(shares) - 1)
                sc_all.append(row)
        sc_curves[dname] = gap
    sc = pd.DataFrame(sc_all)
    sc.to_csv(os.path.join(OUT, 'B13_scenarios_full.csv'), index=False, float_format='%.3f')
    show = sc[sc['exUS share'].isin([0.0, 0.2, 0.4, round(world_cap_exus, 3)])].copy()
    show['exUS share'] = show.apply(lambda r: f"{r['exUS share']:.2f}{' (world-cap)' if r['is_world_cap'] else ''}", axis=1)
    show = show.drop(columns=['is_world_cap'])
    save('B13_scenarios', show.set_index('dataset'), 'B13. Scenario analysis: 20-year DCA, bootstrap of shifted returns',
         'US leg shifted by a constant monthly amount before resampling so that (no US edge) both legs share the same arithmetic mean return, (reversal) the US mean is 2pp/yr below ex-US. "world-cap" = today\'s MSCI-World-like ex-US share (implied from Part C). Geometric returns differ slightly because ex-US volatility is higher.', index=True, dec=2)
    return dict(res_ff=res_ff, pt_ff=pt_ff, out1=out1, P1=P1, sc=sc, res_fund=res_fund, st1=st1, ref1=ref1, risk1=risk1)


def fig_b(B):
    # fig2: IRR distributions of rolling windows
    out1 = B['out1']
    names = list(B['P1'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    cols = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    for ax, h in zip(axes, [10, 20]):
        data = [100 * out1[(n, h)].dropna().values for n in names]
        bp = ax.boxplot(data, tick_labels=[n.replace(' (', '\n(').replace('Developed', 'Dev.') for n in names], showfliers=False, patch_artist=True, whis=(0, 100))
        for p, c in zip(bp['boxes'], cols):
            p.set_facecolor(c)
            p.set_alpha(.5)
        ax.set_title(f'{h}-year monthly DCA IRR, all start months 1990-2026\n(box = IQR, whiskers = min/max; heavily overlapping windows)', fontsize=10)
        ax.set_ylabel('IRR % per year')
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_dca_irr_distributions.png'), dpi=DPI)
    plt.close(fig)
    # fig3: bootstrap difference histogram
    res = B['res_ff']
    nm = res['names']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for lab, c in [('Developed world (FF_DEV)', '#2ca02c'), ('US 80 / exUS 20', '#ff7f0e'), ('US 60 / exUS 40', '#9467bd')]:
        d = 100 * (res['irr20'][:, nm.index(lab)] - res['irr20'][:, nm.index(B['ref1'])])
        axes[0].hist(d, bins=60, alpha=.45, color=c, label=f'{lab}: median {np.median(d):+.2f}pp, P(>US)={np.mean(d > 0):.2f}')
    axes[0].axvline(0, color='k', lw=1)
    axes[0].set_title('Bootstrap: 20y DCA IRR, option minus US-only (FF 1990-2026)', fontsize=10)
    axes[0].set_xlabel('pp per year')
    axes[0].legend(fontsize=7)
    r2, _ = B['res_fund']['PRITX']
    nm2 = r2['names']
    for lab, c in [('US 80 / exUS 20', '#ff7f0e'), ('US 60 / exUS 40', '#9467bd'), ('exUS only (PRITX)', '#d62728')]:
        d = 100 * (r2['irr20'][:, nm2.index(lab)] - r2['irr20'][:, nm2.index('US only (VFINX)')])
        axes[1].hist(d, bins=60, alpha=.45, color=c, label=f'{lab}: median {np.median(d):+.2f}pp, P(>US)={np.mean(d > 0):.2f}')
    axes[1].axvline(0, color='k', lw=1)
    axes[1].set_title('Same, VFINX / PRITX mutual funds (1980-2026)', fontsize=10)
    axes[1].set_xlabel('pp per year')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_bootstrap_difference.png'), dpi=DPI)
    plt.close(fig)
    # fig4: scenario curves
    sc = B['sc']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, dname in enumerate(sc['dataset'].unique()):
        for k, (metric, ttl) in enumerate([('median IRR %', 'median'), ('p10 IRR %', '10th percentile')]):
            ax = axes[k, j]
            for sname, c in zip(['history as is', 'no US edge (equal mean)', 'reversal (US 2pp/yr below exUS)'], ['#1f77b4', '#7f7f7f', '#d62728']):
                s = sc[(sc.dataset == dname) & (sc.scenario == sname) & (~sc.is_world_cap)]
                ax.plot(100 * s['exUS share'], s[metric], marker='o', color=c, label=sname)
            ax.set_title(f'{dname}\n{ttl} 20y DCA IRR', fontsize=9)
            ax.set_ylabel('IRR % per year')
            ax.grid(alpha=.3)
            if k == 1:
                ax.set_xlabel('ex-US share of each contribution, %')
            if j == 0 and k == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_scenario_curves.png'), dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ C
def roll_beta(y, x, w):
    cov = y.rolling(w).cov(x)
    var = x.rolling(w).var()
    cor = y.rolling(w).corr(x)
    return cov / var, cor ** 2, cor


def part_c():
    print('C ...')
    cl = load_prices(['VOO', 'VEA', 'VXUS', 'URTH', 'ACWI', 'VT', 'IWDA.L'], how='outer')['close']
    rows, hist = [], {}
    specs = [('URTH', 'VEA', 1), ('ACWI', 'VXUS', 1), ('VT', 'VXUS', 1), ('IWDA.L', 'VEA', 5)]
    for w, ex, step in specs:
        d = cl[['VOO', ex, w]].dropna()
        if step > 1:
            d = d.iloc[::step]
        r = d.pct_change().dropna()
        win3 = 756 // step
        win1 = 252 // step
        for lab, win in [('latest 3y', win3), ('latest 1y', win1), ('full sample', len(r))]:
            rr = r.iloc[-win:]
            b = np.cov(rr[w], rr['VOO'])[0, 1] / rr['VOO'].var()
            c = rr[w].corr(rr['VOO'])
            yy = rr[w] - rr[ex]
            xx = rr['VOO'] - rr[ex]
            uw = np.cov(yy, xx)[0, 1] / xx.var()
            rows.append({'world index': w + (' (5-day returns)' if step > 1 else ''), 'window': lab, 'start': f'{rr.index[0]:%Y-%m-%d}', 'n obs': len(rr),
                         'beta on VOO': b, 'R2': c * c, 'corr': c, 'ex-US leg': ex, 'implied US weight': uw})
        # rolling implied weight
        yy = r[w] - r[ex]
        xx = r['VOO'] - r[ex]
        iw = yy.rolling(win3).cov(xx) / xx.rolling(win3).var()
        hist[w] = iw.dropna()
        rb, rr2, rc = roll_beta(r[w], r['VOO'], win3)
        hist[w + '_beta'] = rb.dropna()
        hist[w + '_r2'] = rr2.dropna()
    save('C1_beta_r2_implied_us_weight', pd.DataFrame(rows).set_index('world index'), 'C1. World index vs VOO: beta, R2, correlation, implied US weight',
         'Implied US weight = slope of (world - exUS) on (VOO - exUS), weights sum to 1, intercept free. Rolling windows are 3 years of daily returns (156 five-day returns for IWDA.L, to avoid London/NY close-timing noise). ex-US leg: VEA for developed-only indices, VXUS for all-world. Only a regression estimate; the true index weights come from the index provider.', dec=3)
    # history table at year ends
    ye = pd.DataFrame({k: v.groupby(v.index.year).last() for k, v in hist.items() if not k.endswith('_beta') and not k.endswith('_r2')})
    ye.index.name = 'year (last obs)'
    save('C2_implied_us_weight_by_year', ye, 'C2. Rolling-3y implied US weight, last observation of each year', '', dec=3)
    # tracking difference FF_DEV vs URTH, IWDA.L
    mon = monthly_returns(load_prices(['FF_DEV', 'URTH', 'IWDA.L', 'FF_DEVXUS', 'FF_MKT'])['close'])
    tr = []
    for t in ['URTH', 'IWDA.L']:
        j = mon[['FF_DEV', t]].dropna()
        n = len(j)
        a = (1 + j['FF_DEV']).prod() ** (12 / n) - 1
        b = (1 + j[t]).prod() ** (12 / n) - 1
        d = j[t] - j['FF_DEV']
        tr.append({'ETF': t, 'overlap': f'{j.index[0]:%Y-%m}..{j.index[-1]:%Y-%m}', 'months': n, 'FF_DEV ann %': 100 * a, 'ETF ann %': 100 * b,
                   'ETF - FF_DEV pp/yr': 100 * (b - a), 'monthly corr': j.corr().iloc[0, 1], 'tracking error pp/yr': 100 * d.std() * np.sqrt(12)})
    save('C3_tracking_FF_DEV', pd.DataFrame(tr).set_index('ETF'), 'C3. FF_DEV (Fama-French developed incl. US, frictionless) vs MSCI World ETFs, monthly returns',
         'FF_DEV covers developed markets only and uses French\'s own country/size coverage; differences are index construction + fees + dividend taxation inside the ETF (gross of the 30% US-withholding, adjusted close).', dec=3)
    # fig5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, c in zip(['URTH', 'IWDA.L', 'ACWI', 'VT'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
        axes[0].plot(100 * hist[k], label=k, color=c)
        axes[1].plot(hist[k + '_r2'], label=k, color=c)
    axes[0].set_title('Implied US weight (rolling 3y regression, sums to 1 with ex-US leg)', fontsize=10)
    axes[0].set_ylabel('% US')
    axes[1].set_title('Rolling 3y R-squared of the world index on VOO', fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_implied_us_weight.png'), dpi=DPI)
    plt.close(fig)
    latest = {k: float(hist[k].iloc[-1]) for k in ['URTH', 'IWDA.L', 'ACWI', 'VT']}
    return latest


# ------------------------------------------------------------------------------------------------ D
def part_d():
    print('D ...')
    groups = {
        'S&P 500': (['VOO', 'VOO_NET30', 'VOO_NET15', 'CSPX.L'], 'VOO'),
        'MSCI World': (['URTH', 'URTH_NET30', 'IWDA.L'], 'URTH'),
        'All-world': (['VT', 'VT_NET30', 'ACWI', 'ACWI_NET30', 'VWRA.L'], None),
        'Ex-US': (['VXUS', 'VXUS_NET30'], 'VXUS'),
    }
    vrows, drows, dcarows = [], [], []
    drag_for_fig = []
    for g, (tk, ref) in groups.items():
        o, c = px(tk)
        extra = {}
        # constructed "NET15-style" lines: halfway between gross and NET30 returns
        for gross, net in [('URTH', 'URTH_NET30'), ('VXUS', 'VXUS_NET30'), ('VT', 'VT_NET30'), ('ACWI', 'ACWI_NET30')]:
            if gross in c and net in c:
                r = (c[gross].pct_change() + c[net].pct_change()) / 2
                extra[gross + '_NET15x'] = 100 * (1 + r.fillna(0)).cumprod()
        c2 = c.copy()
        for k, v in extra.items():
            c2[k] = v
        refs = [ref] if ref else ['VT', 'ACWI']
        days = (c.index[-1] - c.index[0]).days
        a = {t: ann_ret(c2[t]) for t in c2}
        for t in c2:
            r_ref = refs[0] if (not ref and t in ('VT', 'VT_NET30')) else (refs[-1] if not ref else ref)
            if not ref and t.startswith('VWRA'):
                r_ref = 'VT'
            vrows.append({'group': g, 'vehicle': t, 'common start': f'{c.index[0]:%Y-%m-%d}', 'end': f'{c.index[-1]:%Y-%m-%d}', 'years': days / 365.25,
                          'ann return %': 100 * a[t], 'reference gross': r_ref, 'drag vs gross pp/yr': 100 * (a[r_ref] - a[t]),
                          'kept % of gross return': 100 * a[t] / a[r_ref]})
            drag_for_fig.append((g, t, 100 * (a[r_ref] - a[t])))
        # DCA IRR full span and rolling
        ms = month_starts(c.index)
        for t in c2:
            pass
        ref_series = {'S&P 500': 'VOO', 'MSCI World': 'URTH', 'Ex-US': 'VXUS'}.get(g, 'VT')
        irr_full = {}
        rolls = {}
        o2 = o.copy()
        for k in extra:
            o2[k] = c2[k]
        for t in c2:
            al = alloc_for(ms, list(c2.columns), {t: 1.0})
            r = rolling_dca(o2, c2, al, len(ms) - 1)
            irr_full[t] = float(r['irr'].iloc[0])
            for h in (5, 10):
                if len(ms) > h * 12 + 12:
                    rr = rolling_dca(o2, c2, al, h * 12)
                    rolls[(t, h)] = rr.set_index('start')['irr']
        for t in c2:
            row = {'group': g, 'vehicle': t, 'full-span DCA irr %': 100 * irr_full[t], 'diff vs gross ref pp': 100 * (irr_full[t] - irr_full[ref_series])}
            for h in (5, 10):
                if (t, h) in rolls:
                    d = (rolls[(t, h)] - rolls[(ref_series, h)]).dropna()
                    row[f'{h}y rolling median diff pp'] = 100 * d.median()
                    row[f'{h}y windows'] = len(d)
            dcarows.append(row)
    vt = save('D1_vehicle_returns', pd.DataFrame(vrows).set_index('group'), 'D1. Annualised return on common dates, by vehicle (USD)',
              'Common trading days within each group, CAGR of adjusted total-return closes. NET30 = 30% of every dividend withheld (US-listed ETF held by a Singapore resident); NET15 = 15% (Irish UCITS treaty rate inside the fund). *_NET15x = constructed halfway between gross and NET30 returns (documented approximation of the Irish-fund treatment of the US part only; foreign WHT on non-US holdings is already inside every fund\'s NAV). London lines close 4.30pm London time, so endpoints carry a few hours of timing noise; with 10-16 years the effect on CAGR is ~0.01-0.05 pp/yr. VOO expense ratio 0.03% vs CSPX 0.07%, IWDA 0.20%, URTH 0.24%, VWRA 0.22% are inside these prices.', dec=3)
    save('D2_vehicle_dca_irr', pd.DataFrame(dcarows).set_index('group'), 'D2. DCA IRR by vehicle: full common span and rolling 5y/10y median difference vs the gross US-listed reference (pp/yr)',
         'Monthly DCA at the first trading day\'s open, 5 bp slippage. For London lines the open is London\'s (before New York opens), which adds one-day timing noise per deposit that averages out over windows.', dec=3)

    # fig6 drag bars
    dfp = pd.DataFrame(drag_for_fig, columns=['group', 'vehicle', 'drag']).query("vehicle not in ['VOO','URTH','VXUS','VT','ACWI']")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728' if '_NET30' in v else '#ff7f0e' if 'NET15' in v else '#2ca02c' for v in dfp.vehicle]
    ax.bar(range(len(dfp)), dfp.drag, color=colors)
    ax.set_xticks(range(len(dfp)))
    ax.set_xticklabels([f'{v}\n({g})' for g, v in zip(dfp.group, dfp.vehicle)], rotation=60, ha='right', fontsize=8)
    for i, v in enumerate(dfp.drag):
        ax.text(i, v + 0.01 * np.sign(v), f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='k', lw=.8)
    ax.set_ylabel('annual return drag vs gross US-listed ETF, pp/yr')
    ax.set_title('What Singapore withholding / fund domicile costs per year (red = 30% WHT, orange = 15% constructed, green = Irish line, incl. fee gap)', fontsize=9)
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_vehicle_drag.png'), dpi=DPI)
    plt.close(fig)
    return pd.DataFrame(vrows)


def part_d2():
    # weekly tracking CSPX.L vs VOO_NET15; USD/SGD; XUS/XWR
    rows = []
    for a, b in [('CSPX.L', 'VOO_NET15'), ('CSPX.L', 'VOO'), ('IWDA.L', 'URTH')]:
        d = load_prices([a, b])['close']
        w = d.iloc[::5].pct_change().dropna()
        diff = w[a] - w[b]
        yr = d.groupby(d.index.year).last().pct_change().dropna()
        rows.append({'pair': f'{a} vs {b}', 'common start': f'{d.index[0]:%Y-%m}', '5-day return corr': w.corr().iloc[0, 1],
                     'tracking error (5-day, annualised) pp': 100 * diff.std() * np.sqrt(52), 'mean diff pp/yr': 100 * diff.mean() * 52,
                     'worst calendar-year diff pp': 100 * (yr[a] - yr[b]).min(), 'best calendar-year diff pp': 100 * (yr[a] - yr[b]).max()})
    save('D3_irish_line_tracking', pd.DataFrame(rows).set_index('pair'), 'D3. Irish accumulating line vs US-listed ETF (common trading days, 5-day returns)',
         'CSPX.L should equal VOO_NET15 minus the fee gap (CSPX 0.07% vs VOO 0.03% expense ratio) if the 15% treaty rate is the only difference; securities lending and index-provider differences are inside the residual.', dec=3)

    # SGD
    fx = pd.read_csv(os.path.join(os.path.dirname(OUT), '..', '..', 'data', 'ohlcv_daily', 'SGD_X.csv'), index_col=0, parse_dates=True)['close']
    fx = fx.dropna()
    yrs = (fx.index[-1] - fx.index[0]).days / 365.25
    chg = (fx.iloc[-1] / fx.iloc[0]) ** (1 / yrs) - 1
    vol = np.log(fx).diff().std() * np.sqrt(252)
    ann = fx.groupby(fx.index.year).last()
    ye = ann.pct_change() * 100
    spy = load_prices(['SPY_NET30', 'SPY'])['close']
    spy = spy.loc['2003-12-01':]
    j = pd.concat([spy, fx.rename('fx')], axis=1, join='inner').dropna()
    r_usd = j['SPY_NET30'].pct_change()
    r_sgd = (1 + r_usd) * (j['fx'].shift(1) / j['fx']) - 1       # USD per SGD up -> SGD return; fx = SGD per USD
    # SGD return = (1+r_usd) * (fx_t / fx_{t-1}) - 1 since value in SGD = USD value * fx
    r_sgd = (1 + r_usd) * (j['fx'] / j['fx'].shift(1)) - 1
    a_usd = (1 + r_usd.dropna()).prod() ** (365.25 / (j.index[-1] - j.index[0]).days) - 1
    a_sgd = (1 + r_sgd.dropna()).prod() ** (365.25 / (j.index[-1] - j.index[0]).days) - 1
    voo = load_prices(['VOO_NET30'])['close']['VOO_NET30']
    jv = pd.concat([voo, fx.rename('fx')], axis=1, join='inner').dropna()
    rv = jv['VOO_NET30'].pct_change()
    rv_s = (1 + rv) * (jv['fx'] / jv['fx'].shift(1)) - 1
    dv = (jv.index[-1] - jv.index[0]).days
    fxtab = pd.DataFrame({
        'metric': ['USD/SGD start (SGD per USD)', 'USD/SGD end', 'annualised change of USD vs SGD %', 'annualised vol of USD/SGD %', 'period',
                   'SPY_NET30 ann return, USD %', 'SPY_NET30 ann return, SGD %', 'effect of FX on SPY_NET30 pp/yr', 'ann vol SPY_NET30 USD %', 'ann vol SGD %',
                   'VOO_NET30 ann return USD % (2010-09+)', 'VOO_NET30 ann return SGD %', 'effect of FX on VOO_NET30 pp/yr'],
        'value': [fx.iloc[0], fx.iloc[-1], 100 * chg, 100 * vol, f'{fx.index[0]:%Y-%m}..{fx.index[-1]:%Y-%m}',
                  100 * a_usd, 100 * a_sgd, 100 * (a_sgd - a_usd), 100 * r_usd.std() * np.sqrt(252), 100 * r_sgd.std() * np.sqrt(252),
                  100 * ((1 + rv.dropna()).prod() ** (365.25 / dv) - 1), 100 * ((1 + rv_s.dropna()).prod() ** (365.25 / dv) - 1),
                  100 * (((1 + rv_s.dropna()).prod() ** (365.25 / dv)) - ((1 + rv.dropna()).prod() ** (365.25 / dv)))]}).set_index('metric')
    fxtab['value'] = [f'{v:.3f}' if isinstance(v, (float, np.floating)) else v for v in fxtab['value']]
    save('D4_usdsgd', fxtab, 'D4. USD/SGD and its effect on SGD-terms returns of a US S&P fund (Singapore-net of 30% WHT)',
         'Yahoo FX closes; returns compounded on common days. An SGD-quoted line (SGX XUS/XWR) does not hedge anything: the underlying is still USD assets, only the quote currency differs, so SGD-terms returns have the same FX effect.', dec=3)
    byy = pd.DataFrame({'USD/SGD year-end': ann, 'USD change %': ye})
    ry = (1 + r_usd.dropna()).groupby(r_usd.dropna().index.year).prod() - 1
    rs = (1 + r_sgd.dropna()).groupby(r_sgd.dropna().index.year).prod() - 1
    byy['SPY_NET30 USD %'] = 100 * ry
    byy['SPY_NET30 SGD %'] = 100 * rs
    byy.index.name = 'year'
    save('D5_usdsgd_by_year', byy.loc[2004:], 'D5. USD/SGD year by year and SPY_NET30 return in USD vs SGD', '2026 = year to date.', dec=2)

    # XUS / XWR
    raw = {t: pd.read_csv(os.path.join(os.path.dirname(OUT), '..', '..', 'data', 'ohlcv_daily', f'{t}.csv'), index_col=0, parse_dates=True) for t in ['XUS.SI', 'XWR.SI']}
    us = load_prices(['VOO', 'CSPX.L', 'URTH', 'IWDA.L'], how='outer')['close']
    rows = []
    for t, refs in [('XUS.SI', ['VOO', 'CSPX.L']), ('XWR.SI', ['IWDA.L', 'URTH'])]:
        d = raw[t].iloc[1:]       # first row is a zero-volume placeholder price from Yahoo
        usd = (d['close'] / fx.reindex(d.index).ffill()).rename(t + '_usd')
        for rf in refs:
            ref = us[rf].dropna()
            for lag in (0, 1):
                s = ref.shift(lag).reindex(usd.index) if lag else ref.reindex(usd.index)
                s = ref.reindex(ref.index.union(usd.index)).ffill().reindex(usd.index) if lag == 0 else ref.shift(1).reindex(ref.index.union(usd.index)).ffill().reindex(usd.index)
                dd = pd.concat([usd, s], axis=1).dropna()
                rr = dd.pct_change().dropna()
                cum = dd.iloc[-1] / dd.iloc[0] - 1
                rows.append({'line': t, 'reference': rf + (' (prior-day close)' if lag else ' (same date)'), 'n daily returns': len(rr),
                             'daily corr': rr.corr().iloc[0, 1], 'cum return line USD %': 100 * cum.iloc[0], 'cum return ref %': 100 * cum.iloc[1],
                             'cum difference pp': 100 * (cum.iloc[0] - cum.iloc[1]), 'period': f'{dd.index[0]:%m-%d}..{dd.index[-1]:%m-%d}'})
    save('D6_sgx_lines_tracking', pd.DataFrame(rows).set_index('line'), 'D6. New SGX lines (XUS.SI, XWR.SI), converted to USD with SGD_X, vs US/London lines. TINY SAMPLE (22-23 rows)',
         'The SGX session closes ~5pm Singapore time, hours before New York opens, so a Singapore date\'s close reflects the previous New York close: compare against the prior-day close (lag 1). FX = Yahoo USD/SGD close. First Yahoo row (2026-10-12, zero volume) dropped. 22 daily returns: correlations and cumulative differences are indicative only; they say nothing about long-run tracking.', dec=3)
    liq = []
    for t, px_ in [('XUS.SI', 'SGD'), ('XWR.SI', 'SGD')]:
        d = raw[t].iloc[1:]
        v = d['volume']
        liq.append({'line': t, 'rows': len(d), 'mean daily volume (shares)': v.mean(), 'median daily volume': v.median(), 'min daily volume': v.min(),
                    'mean daily traded value (SGD)': (d['close'] * v).mean(), 'median daily traded value (SGD)': (d['close'] * v).median(),
                    'last close SGD': d['close'].iloc[-1], 'intraday range % (mean high-low / close)': 100 * ((d['high'] - d['low']) / d['close']).mean()})
    for t, ccy in [('CSPX.L', 'USD'), ('IWDA.L', 'USD'), ('VOO', 'USD')]:
        d = pd.read_csv(os.path.join(os.path.dirname(OUT), '..', '..', 'data', 'ohlcv_daily', f'{t}.csv'), index_col=0, parse_dates=True).loc['2026-10-13':]
        v = d['volume']
        liq.append({'line': t + ' (same dates, reference)', 'rows': len(d), 'mean daily volume (shares)': v.mean(), 'median daily volume': v.median(), 'min daily volume': v.min(),
                    'mean daily traded value (SGD)': np.nan, 'median daily traded value (SGD)': (d['close'] * v).median(), 'last close SGD': d['close'].iloc[-1],
                    'intraday range % (mean high-low / close)': 100 * ((d['high'] - d['low']) / d['close']).mean()})
    save('D7_sgx_liquidity', pd.DataFrame(liq).set_index('line'), 'D7. Daily trading volume of the new SGX lines over the available rows (and reference lines; reference traded value in USD, not SGD)',
         'Volume in shares as reported by Yahoo; the first-listing days may be unrepresentative. Intraday range includes listing-week volatility.', dec=1)


# ------------------------------------------------------------------------------------------------ E
def part_e(drags):
    print('E ...')
    out_tabs = {}
    d_sp, d_world, d_ex = drags
    raw = load_prices(['FF_MKT', 'FF_DEV', 'FF_DEVXUS'])['close']
    rr = raw.pct_change().dropna()
    brk_o, brk_c = px(['BRK-A'])
    spy_o, spy_c = px(['SPY_NET15'])

    def build(start, sp_source):
        lv = pd.DataFrame({'WORLD': synth(rr['FF_DEV'], d_world), 'EXUS': synth(rr['FF_DEVXUS'], d_ex)})
        lv['SP'] = synth(rr['FF_MKT'], d_sp) if sp_source == 'FF' else np.nan
        cl = lv.copy()
        op = lv.copy()
        if sp_source == 'SPY':
            cl['SP'] = spy_c['SPY_NET15']
            op['SP'] = spy_o['SPY_NET15']
        cl['BRK'] = brk_c['BRK-A']
        op['BRK'] = brk_o['BRK-A']
        cl = cl.dropna().loc[start:]
        op = op.reindex(cl.index)
        return op[['SP', 'WORLD', 'EXUS', 'BRK']], cl[['SP', 'WORLD', 'EXUS', 'BRK']]
    P = {'1 S&P only': {'SP': 1.0}, '2 S&P 70 / BRK 30': {'SP': .7, 'BRK': .3}, '3 World only': {'WORLD': 1.0},
         '4 World 70 / BRK 30': {'WORLD': .7, 'BRK': .3}, '5 S&P 50 / exUS 20 / BRK 30': {'SP': .5, 'EXUS': .2, 'BRK': .3}}
    pairs = [('2 S&P 70 / BRK 30', '1 S&P only'), ('3 World only', '1 S&P only'), ('4 World 70 / BRK 30', '2 S&P 70 / BRK 30'),
             ('5 S&P 50 / exUS 20 / BRK 30', '2 S&P 70 / BRK 30'), ('4 World 70 / BRK 30', '1 S&P only'), ('5 S&P 50 / exUS 20 / BRK 30', '1 S&P only'),
             ('4 World 70 / BRK 30', '3 World only')]
    keep = {}
    for tag, start, src in [('1990', '1990-07-02', 'FF'), ('1999', '1999-03-10', 'SPY')]:
        o, c = build(start, src)
        out, ms = rolling_set(o, c, P, [10, 20])
        st, _, no = rolling_stats(out, '1 S&P only', [10, 20])
        save(f'E1_rolling_{tag}', st.set_index('portfolio'), f'E1. Rolling DCA, net of Singapore tax, window starting {c.index[0]:%Y-%m} ({"FF indices, S&P = FF_MKT" if src == "FF" else "S&P = SPY_NET15 real series, world/exUS = FF_DEV/FF_DEVXUS constructed"}; BRK-A gross)',
             'IRR %. Independent observations ~ sample years / H.', dec=2)
        save(f'E2_nonoverlap_{tag}', no.set_index('portfolio'), f'E2. Non-overlapping windows, start {c.index[0]:%Y-%m}', '', dec=2)
        save(f'E3_path_risk_{tag}', path_risk(o, c, P), f'E3. Full-sample DCABacktest path risk, {c.index[0]:%Y-%m}..{c.index[-1]:%Y-%m}', '', dec=2)
        mon = monthly_returns(c)
        res = boot_core(mon.values, list(c.columns), P)
        save(f'E4_bootstrap_levels_{tag}', level_table(res), f'E4. Bootstrap levels (20-year DCA, {R_PATHS} paths, sample {mon.index[0]:%Y-%m}..{mon.index[-1]:%Y-%m})', '', dec=2)
        save(f'E5_bootstrap_paired_{tag}', paired_table(res, pairs), f'E5. Paired bootstrap differences, window {tag}', '90% interval = 5th..95th percentile across paths.', dec=2)
        keep[tag] = (out, res, mon)
    # real-vehicle window
    o, c = px(['CSPX.L', 'IWDA.L', 'BRK-B'])
    c.columns = ['SP', 'WORLD', 'BRK']
    o.columns = c.columns
    P3 = {k: v for k, v in P.items() if 'EXUS' not in str(v)}
    out, ms = rolling_set(o, c, P3, [10])
    st, _, no = rolling_stats(out, '1 S&P only', [10])
    save('E6_rolling_real_vehicles', st.set_index('portfolio'), f'E6. Real vehicles only: CSPX.L (Irish S&P), IWDA.L (Irish World), BRK-B, window {c.index[0]:%Y-%m}..{c.index[-1]:%Y-%m}, 10y IRR %',
         'Only ~5 years of start dates, essentially 1 independent decade: indicative. London lines trade at London opens (one-day timing noise per deposit).', dec=2)
    save('E7_path_risk_real_vehicles', path_risk(o, c, P3), 'E7. Path risk of real vehicles', '', dec=2)
    # fig7
    out, res, mon = keep['1990']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(P)
    cols = ['#1f77b4', '#17becf', '#2ca02c', '#98df8a', '#ff7f0e']
    data = [100 * res['irr20'][:, i] for i in range(len(names))]
    bp = axes[0].boxplot(data, tick_labels=[n.replace(' / ', '/').replace(' ', '\n', 1) for n in names], showfliers=False, patch_artist=True, whis=(5, 95))
    for p, c_ in zip(bp['boxes'], cols):
        p.set_facecolor(c_)
        p.set_alpha(.55)
    axes[0].set_title('Bootstrap 20y DCA IRR (FF 1990+, net of SG tax; whiskers 5-95%)', fontsize=10)
    axes[0].set_ylabel('IRR % per year')
    axes[0].tick_params(axis='x', labelsize=7)
    axes[0].grid(alpha=.3)
    data = [100 * keep['1990'][0][(n, 10)].dropna().values for n in names]
    bp = axes[1].boxplot(data, tick_labels=[n.replace(' / ', '/').replace(' ', '\n', 1) for n in names], showfliers=False, patch_artist=True, whis=(0, 100))
    for p, c_ in zip(bp['boxes'], cols):
        p.set_facecolor(c_)
        p.set_alpha(.55)
    axes[1].set_title('Rolling 10y DCA IRR, actual history 1990-2026 (min..max)', fontsize=10)
    axes[1].tick_params(axis='x', labelsize=7)
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_core_comparison.png'), dpi=DPI)
    plt.close(fig)
    return keep


def drag_estimates():
    """Constant annual drags used to build net-of-15% (Irish-style) versions of FF series."""
    def drag(g, n, f=1.0):
        o, c = px([g, n])
        r = c.pct_change().dropna()
        return float((r[g] - r[n]).mean() * 252 * f), c.index[0]
    spy, d0 = drag('SPY', 'SPY_NET15')
    urth, d1 = drag('URTH', 'URTH_NET30', 0.5)
    vea, d2 = drag('VEA', 'VEA_NET30', 0.5)
    t = pd.DataFrame({'series': ['S&P (FF_MKT, SPY_NET15)', 'World (FF_DEV)', 'ex-US (FF_DEVXUS)'],
                      'drag used pp/yr': [100 * spy, 100 * urth, 100 * vea],
                      'derived from': [f'SPY - SPY_NET15 mean daily return x252 since {d0:%Y-%m}', f'0.5 x (URTH - URTH_NET30) since {d1:%Y-%m}',
                                       f'0.5 x (VEA - VEA_NET30) since {d2:%Y-%m}']}).set_index('series')
    save('E0_drag_assumptions', t, 'E0. Constant 15%-style dividend drag applied to French series (documented approximation)',
         'Applied as a constant daily deduction over the whole history. True dividend yields were higher in 1990-2000 (S&P ~3% then vs ~1.3-1.8% now), so the early-period drag is understated; foreign withholding on non-US holdings is already inside real ETF NAVs but NOT inside the French series (it would lower the ex-US/world lines further, by roughly 0.1-0.3 pp/yr).', dec=3)
    return spy, urth, vea


# ------------------------------------------------------------------------------------------------ main
def main():
    reg_rows, roll = part_a()
    latest_w = part_c()
    world_cap = 1 - latest_w['URTH']
    print('latest implied US weights', latest_w, 'world cap exUS share', world_cap)
    B = part_b(world_cap)
    fig_b(B)
    part_d()
    part_d2()
    drags = drag_estimates()
    keep = part_e(drags)
    # world cap info
    save('X0_world_cap_share_used', pd.DataFrame({'quantity': ['implied US weight URTH (3y)', 'implied US weight IWDA.L', 'implied US weight ACWI', 'implied US weight VT', 'ex-US share used in scenarios (1 - URTH)'],
                                                   'value': [latest_w['URTH'], latest_w['IWDA.L'], latest_w['ACWI'], latest_w['VT'], world_cap]}).set_index('quantity'), 'X0. Ex-US share used as "world-cap" in scenarios', '', dec=3)
    with open(os.path.join(OUT, 'findings_tables.md'), 'w') as f:
        f.write('# EXP8 auto-generated tables\n\n')
        for name, title, df, note in TABLES:
            f.write(f'### {title}\n\n')
            if note:
                f.write(f'_{note}_\n\n')
            f.write(md(df) + '\n\n')
    print('done;', len(TABLES), 'tables')


if __name__ == '__main__':
    main()
