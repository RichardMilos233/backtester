"""EXP2 - which core asset / fixed mix to DCA into.

Fixed-fraction monthly DCA (constant `alloc` rows, never sells) evaluated on every possible monthly start
date with `src.dca.rolling_dca`. Decision dates = first trading day of each month, buys at that day's open
(+5 bp slippage). Constant fractions need no signal, so there is no lookahead by construction.

Run from the repo root:  python scripts/dca/exp2_core_choice.py
Outputs: docs/dca/exp2_core_choice/*.csv, *.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dca import DCABacktest, buy_underweight, dca_metrics, fixed_buy, load_prices, month_starts, rolling_dca

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca', 'exp2_core_choice')
os.makedirs(OUT, exist_ok=True)
DPI = 130
DISP = {'^NDX': 'NDX(px)'}
MIN_WIN = 12


def dn(t):
    return DISP.get(t, t)


def label_of(w):
    items = [(t, x) for t, x in w.items() if x > 1e-12]
    if len(items) == 1:
        return dn(items[0][0])
    return ' / '.join(f'{dn(t)} {int(round(x * 100))}' for t, x in items)


def simplex(cols, step=10):
    out = []
    for i in range(step + 1):
        for j in range(step + 1 - i):
            k = step - i - j
            out.append(dict(zip(cols, [i / step, j / step, k / step])))
    return out


def pair_grid(a, b, step=10):
    return [{a: (step - i) / step, b: i / step} for i in range(step + 1)]


# ------------------------------------------------------------------ core machinery
def run_windows(opens, closes, mixes, horizons_y, start=None):
    """mixes: list of weight dicts. Returns {(label, H_years): DataFrame indexed by start date}."""
    idx = opens.index if start is None else opens.loc[start:].index
    ms = month_starts(idx)
    res = {}
    for w in mixes:
        lab = label_of(w)
        alloc = pd.DataFrame(0.0, index=ms, columns=list(opens.columns))
        for t, x in w.items():
            alloc[t] = x
        for hy in horizons_y:
            r = rolling_dca(opens, closes, alloc, hy * 12)
            if len(r) >= 1:
                res[(lab, hy)] = r.set_index('start')[['end', 'final_value', 'wealth_multiple', 'irr']].rename(
                    columns={'wealth_multiple': 'wm', 'final_value': 'final'})
    return res, ms


def pct(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def stats_row(df, best, worst, ref):
    irr = df['irr'].to_numpy()
    row = dict(n_windows=len(df),
               irr_median=pct(irr, 50), irr_mean=float(np.nanmean(irr)), irr_p10=pct(irr, 10),
               irr_min=float(np.nanmin(irr)), irr_max=float(np.nanmax(irr)),
               wm_median=float(df['wm'].median()), wm_min=float(df['wm'].min()))
    if ref is not None:
        row['pct_beat_ref'] = float((df['final'] > ref * (1 + 1e-9)).mean() * 100)
    else:
        row['pct_beat_ref'] = np.nan
    rb = (df['final'] / best).to_numpy()
    rw = (df['final'] / worst).to_numpy()
    row.update(regret_best_min=float(np.min(rb)), regret_best_p10=pct(rb, 10), regret_best_median=pct(rb, 50),
               ratio_worst_min=float(np.min(rw)), ratio_worst_p10=pct(rw, 10), ratio_worst_median=pct(rw, 50))
    return row


def summarize(W, mixes, singles, ref, horizons_y, sample_years, group=None, min_windows=MIN_WIN):
    """singles: list of single-asset labels for the best/worst-single regret. ref: label of the S&P reference.
    group: None, or function(start_timestamp_index)->array of group names (e.g. start decade)."""
    rows = []
    cols = sorted({t for w in mixes for t in w})
    for hy in horizons_y:
        if not all((s, hy) in W for s in singles):
            continue
        sf = pd.concat({s: W[(s, hy)]['final'] for s in singles}, axis=1)
        best, worst = sf.max(axis=1), sf.min(axis=1)
        reff = W[(ref, hy)]['final'] if (ref, hy) in W else None
        for w in mixes:
            lab = label_of(w)
            if (lab, hy) not in W:
                continue
            df = W[(lab, hy)]
            parts = [('all', df)] if group is None else [(g, df[group(df.index) == g]) for g in sorted(set(group(df.index)))]
            for g, d in parts:
                if len(d) < (min_windows if g == 'all' else 6):
                    continue
                r = stats_row(d, best.loc[d.index], worst.loc[d.index], None if reff is None else reff.loc[d.index])
                if lab == ref:
                    r['pct_beat_ref'] = np.nan
                r.update(mix=lab, horizon_y=hy, eff_indep_obs=round(sample_years / hy, 1))
                if group is not None:
                    r['start_decade'] = g
                for c in cols:
                    r['w_' + c] = w.get(c, 0.0)
                rows.append(r)
    out = pd.DataFrame(rows)
    first = ['mix', 'horizon_y'] + (['start_decade'] if group is not None else []) + ['n_windows', 'eff_indep_obs']
    return out[first + [c for c in out.columns if c not in first]]


def decade(idx):
    return np.array([f'{(y // 10) * 10}s' for y in idx.year])


def save(df, name, nd=4):
    p = os.path.join(OUT, name)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == 'f':
            out[c] = out[c].round(nd if not (c.startswith('pct') or c.startswith('wm')) else 2)
    out.to_csv(p, index=False)
    return out


# ------------------------------------------------------------------ path risk
def path_risk(opens, closes, mixes, amount=500.0, start=None):
    ms = month_starts(opens.index if start is None else opens.loc[start:].index)
    contrib = pd.Series(amount, index=ms)
    rows = []
    for w in mixes:
        tgt = {t: x for t, x in w.items() if x > 1e-12}
        led = DCABacktest().run(opens, closes, buy_underweight(tgt), contrib)
        m = dca_metrics(led)
        tv = led['total_value']
        peak = tv.cummax()
        rows.append(dict(mix=label_of(w), years=m['years'], total_contributed=m['total_contributed'],
                         final_value=m['final_value'], wealth_multiple=m['wealth_multiple'], irr=m['irr'],
                         twr=m['twr_annualized'], vol=m['annualized_volatility'], max_drawdown_nav=m['max_drawdown'],
                         worst_12m=m['worst_12m_return'], max_drawdown_dollar_value=float((tv / peak - 1).min()),
                         max_dollar_drop=float((peak - tv).max())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plotting helpers
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf', '#7f7f7f', '#e377c2', '#bcbd22',
           '#393b79', '#843c39']


def boxplots(W, labels, horizons_y, title, fname, note=''):
    fig, axes = plt.subplots(1, len(horizons_y), figsize=(5.2 * len(horizons_y), 6.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, hy in zip(axes, horizons_y):
        data, labs = [], []
        for l in labels:
            if (l, hy) in W:
                data.append(W[(l, hy)]['irr'].dropna().to_numpy() * 100)
                labs.append(l)
        bp = ax.boxplot(data, orientation="horizontal", whis=(0, 100), patch_artist=True, widths=0.6)
        for i, b in enumerate(bp['boxes']):
            b.set_facecolor(PALETTE[i % len(PALETTE)])
            b.set_alpha(0.55)
        ax.set_yticks(range(1, len(labs) + 1))
        ax.set_yticklabels(labs, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='k', lw=0.6)
        ax.set_title(f'{hy}-year horizon (n={len(data[0])} windows)', fontsize=10)
        ax.set_xlabel('IRR of monthly DCA, % per year  (box = p25-p75, line = median, whiskers = min-max)', fontsize=7)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle(title + note, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, fname), dpi=DPI)
    plt.close(fig)


def simplex_heatmap(summ, cols, fname, title):
    """x = weight in cols[1], y = weight in cols[2], the rest in cols[0]."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    for r, hy in enumerate([10, 20]):
        for c, (metric, nm) in enumerate([('irr_median', 'median IRR'), ('irr_p10', 'p10 IRR')]):
            ax = axes[r, c]
            s = summ[summ.horizon_y == hy]
            grid = np.full((11, 11), np.nan)
            for _, row in s.iterrows():
                i = int(round(row['w_' + cols[1]] * 10))
                j = int(round(row['w_' + cols[2]] * 10))
                grid[j, i] = row[metric] * 100
            im = ax.imshow(grid, origin='lower', cmap='viridis', aspect='equal')
            for j in range(11):
                for i in range(11):
                    if np.isfinite(grid[j, i]):
                        ax.text(i, j, f'{grid[j, i]:.1f}', ha='center', va='center', fontsize=7,
                                color='w' if grid[j, i] < np.nanmean(grid) else 'k')
            ax.set_xticks(range(11))
            ax.set_xticklabels([str(10 * i) for i in range(11)], fontsize=7)
            ax.set_yticks(range(11))
            ax.set_yticklabels([str(10 * i) for i in range(11)], fontsize=7)
            ax.set_xlabel(f'{dn(cols[1])} weight, %', fontsize=8)
            ax.set_ylabel(f'{dn(cols[2])} weight, %', fontsize=8)
            ax.set_title(f'{hy}y horizon: {nm} (% p.a.)   [{dn(cols[0])} = remainder]', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, fname), dpi=DPI)
    plt.close(fig)


def frontier_scatter(summs, fname, title):
    """summs: list of (name, summary_df, cols). Panels per horizon x sample."""
    n = len(summs)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 11))
    for k, (nm, s, cols) in enumerate(summs):
        for r, hy in enumerate([10, 20]):
            ax = axes[r, k]
            d = s[s.horizon_y == hy]
            single = d[d[['w_' + c for c in cols]].max(axis=1) > 0.999]
            col = {c: PALETTE[i] for i, c in enumerate(cols)}
            for _, row in d.iterrows():
                nz = [c for c in cols if row['w_' + c] > 1e-9]
                color = col[nz[0]] if len(nz) == 1 else ('#999999' if len(nz) == 3 else '#e0a030')
                ax.scatter(row['irr_median'] * 100, row['irr_p10'] * 100, s=18 if len(nz) > 1 else 70, color=color,
                           edgecolor='k' if len(nz) == 1 else 'none', lw=0.8, zorder=3 if len(nz) == 1 else 2, alpha=0.85)
            for _, row in single.iterrows():
                ax.annotate(row['mix'], (row['irr_median'] * 100, row['irr_p10'] * 100), fontsize=8,
                            xytext=(5, 4), textcoords='offset points')
            lo = min(d['irr_median'].min(), d['irr_p10'].min()) * 100
            hi = max(d['irr_median'].max(), d['irr_p10'].max()) * 100
            ax.plot([lo, hi], [lo, hi], color='k', lw=0.5, ls=':')
            ax.set_xlabel('median IRR, % p.a.')
            ax.set_ylabel('p10 IRR, % p.a.')
            ax.set_title(f'{nm}, {hy}y horizon', fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle(title + '\n(black-edged = single assets; orange = two-asset mixes; grey = three-asset mixes)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), dpi=DPI)
    plt.close(fig)


def regret_chart(items, fname, title):
    """items: list of (sample_name, summary, labels)."""
    fig, axes = plt.subplots(len(items), 1, figsize=(13, 5.2 * len(items)))
    axes = np.atleast_1d(axes)
    hs = [5, 10, 15, 20]
    for ax, (nm, s, labels) in zip(axes, items):
        x = np.arange(len(labels))
        width = 0.2
        for i, hy in enumerate(hs):
            d = s[s.horizon_y == hy].set_index('mix')
            v = [d.loc[l, 'regret_best_min'] if l in d.index else np.nan for l in labels]
            p = [d.loc[l, 'regret_best_p10'] if l in d.index else np.nan for l in labels]
            ax.bar(x + (i - 1.5) * width, v, width, label=f'{hy}y min', color=PALETTE[i], alpha=0.9)
            ax.scatter(x + (i - 1.5) * width, p, marker='_', s=120, color='k', zorder=4,
                       label='p10 (same colour = same horizon)' if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('final value / final value of best single asset\n(same window)')
        ax.set_ylim(0, 1.2)
        ax.axhline(1, color='k', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'{nm}: worst-case (bar) and p10 (black tick) ratio to the best single asset in each window', fontsize=10)
        ax.legend(fontsize=8, ncol=5, loc='upper left')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, fname), dpi=DPI)
    plt.close(fig)


# ================================================================== main
def main():
    horizons = [5, 10, 15, 20]
    all_summ = {}

    # ---------------------------------------------------------------- sample 1: modern
    mc = ['SPY', 'QQQ', 'BRK-B']
    px = load_prices(mc, how='inner')
    o1, c1 = px['open'], px['close']
    yrs1 = (o1.index[-1] - o1.index[0]).days / 365.25
    print('modern sample', o1.index[0].date(), o1.index[-1].date(), round(yrs1, 1), 'years')
    simp1 = simplex(mc)
    W1, ms1 = run_windows(o1, c1, simp1, horizons)
    singles1 = ['SPY', 'QQQ', 'BRK-B']
    S1 = summarize(W1, simp1, singles1, 'SPY', horizons, yrs1)
    save(S1, 's1_modern_all_mixes.csv')
    # SPY/BRK grid subset
    S1_pair = S1[(S1['w_QQQ'] == 0)]
    save(S1_pair, 's1b_modern_spy_brk_grid.csv')
    S1d = summarize(W1, simp1, singles1, 'SPY', horizons, yrs1, group=decade)
    save(S1d, 's1_modern_by_start_decade.csv')
    all_summ['modern'] = S1

    # ---------------------------------------------------------------- sample 2: long
    lc = ['VFINX', '^NDX', 'BRK-A']
    px = load_prices(lc, how='inner')
    o2, c2 = px['open'], px['close']
    yrs2 = (o2.index[-1] - o2.index[0]).days / 365.25
    print('long sample', o2.index[0].date(), o2.index[-1].date(), round(yrs2, 1), 'years')
    simp2 = simplex(lc)
    W2, ms2 = run_windows(o2, c2, simp2, horizons)
    singles2 = ['VFINX', 'NDX(px)', 'BRK-A']
    S2 = summarize(W2, simp2, singles2, 'VFINX', horizons, yrs2)
    save(S2, 's2_long_all_mixes.csv')
    S2d = summarize(W2, simp2, singles2, 'VFINX', horizons, yrs2, group=decade)
    save(S2d, 's2_long_by_start_decade.csv')
    all_summ['long'] = S2

    # VFINX / BRK-A from 1980-03
    px = load_prices(['VFINX', 'BRK-A'], how='inner', start='1980-03-01')
    o2b, c2b = px['open'], px['close']
    yrs2b = (o2b.index[-1] - o2b.index[0]).days / 365.25
    print('VFINX/BRK-A sample', o2b.index[0].date(), round(yrs2b, 1))
    pg = pair_grid('VFINX', 'BRK-A')
    W2b, _ = run_windows(o2b, c2b, pg, horizons)
    S2b = summarize(W2b, pg, ['VFINX', 'BRK-A'], 'VFINX', horizons, yrs2b)
    save(S2b, 's2b_vfinx_brka_1980_grid.csv')
    S2bd = summarize(W2b, pg, ['VFINX', 'BRK-A'], 'VFINX', horizons, yrs2b, group=decade)
    save(S2bd, 's2b_vfinx_brka_1980_by_start_decade.csv')

    # ---------------------------------------------------------------- sample 3: broad funds
    funds = ['VOO', 'VTI', 'VT', 'VXUS', 'SCHD', 'RSP', 'VTV', 'VUG', 'QQQ', 'BRK-B']
    bc = funds + ['BND']
    px = load_prices(bc, how='inner', start='2011-11-01')
    o3, c3 = px['open'], px['close']
    yrs3 = (o3.index[-1] - o3.index[0]).days / 365.25
    print('broad common sample', o3.index[0].date(), o3.index[-1].date(), round(yrs3, 1))
    mixes3 = [{f: 1.0} for f in funds] + [{'BND': 1.0}] + [
        {'VOO': .8, 'VXUS': .2}, {'VOO': .6, 'VXUS': .4},
        {'VOO': .5, 'BRK-B': .3, 'VXUS': .2}, {'VOO': .4, 'BRK-B': .4, 'VXUS': .2},
        {'VOO': .7, 'BRK-B': .3}, {'VOO': .5, 'BRK-B': .5},
        {'VOO': .5, 'QQQ': .2, 'BRK-B': .3}, {'VOO': .8, 'BND': .2}, {'VOO': .5, 'BRK-B': .3, 'BND': .2}]
    W3, _ = run_windows(o3, c3, mixes3, [5, 10, 15])
    S3 = summarize(W3, mixes3, funds, 'VOO', [5, 10, 15], yrs3, min_windows=1)
    save(S3, 's3_broad_common_2011.csv')

    # pairwise vs SPY over each fund's full life
    rows = []
    for f in funds + ['BND']:
        px = load_prices([f, 'SPY'], how='inner')
        o, c = px['open'], px['close']
        yrs = (o.index[-1] - o.index[0]).days / 365.25
        Wp, _ = run_windows(o, c, [{f: 1.0}, {'SPY': 1.0}], [5, 10, 15])
        for hy in [5, 10, 15]:
            if (f, hy) not in Wp or len(Wp[(f, hy)]) < 6:
                continue
            a, b = Wp[(f, hy)], Wp[('SPY', hy)]
            rows.append(dict(fund=f, first_date=o.index[0].date(), horizon_y=hy, n_windows=len(a),
                             irr_fund_median=a['irr'].median(), irr_spy_median=b['irr'].median(),
                             irr_diff_median=(a['irr'] - b['irr']).median(), irr_diff_p10=pct((a['irr'] - b['irr']), 10),
                             irr_diff_min=(a['irr'] - b['irr']).min(), irr_diff_max=(a['irr'] - b['irr']).max(),
                             pct_fund_beats_spy=float((a['final'] > b['final']).mean() * 100),
                             fund_irr_p10=pct(a['irr'], 10), spy_irr_p10=pct(b['irr'], 10)))
    P3 = pd.DataFrame(rows)
    save(P3, 's3_pairwise_vs_spy_full_life.csv', nd=5)

    # ---------------------------------------------------------------- sample 4: international
    px = load_prices(['VFINX', 'BRK-A', 'VGTSX'], how='inner', start='1996-05-01')
    o4, c4 = px['open'], px['close']
    yrs4 = (o4.index[-1] - o4.index[0]).days / 365.25
    print('intl sample', o4.index[0].date(), round(yrs4, 1))
    mixes4 = [{'VFINX': 1.0}, {'BRK-A': 1.0}, {'VGTSX': 1.0}, {'VFINX': .8, 'VGTSX': .2}, {'VFINX': .6, 'VGTSX': .4},
              {'VFINX': .5, 'BRK-A': .3, 'VGTSX': .2}, {'VFINX': .7, 'BRK-A': .3}]
    W4, _ = run_windows(o4, c4, mixes4, [10, 20])
    S4 = summarize(W4, mixes4, ['VFINX', 'BRK-A', 'VGTSX'], 'VFINX', [10, 20], yrs4)
    save(S4, 's4_intl_1996.csv')
    S4d = summarize(W4, mixes4, ['VFINX', 'BRK-A', 'VGTSX'], 'VFINX', [10, 20], yrs4, group=decade)
    save(S4d, 's4_intl_by_start_decade.csv')

    # ---------------------------------------------------------------- spot check one window
    s0 = ms1[ms1.get_indexer([pd.Timestamp('2007-01-02')], method='bfill')[0]]
    s_i = list(ms1).index(s0)
    H = 120
    chk = W1[('SPY / QQQ 40'.replace('SPY / QQQ 40', 'SPY 50 / QQQ 20 / BRK-B 30'), 10)].loc[s0]
    dates = ms1[s_i:s_i + H]
    led = DCABacktest().run(o1, c1, fixed_buy({'SPY': .5, 'QQQ': .2, 'BRK-B': .3}), pd.Series(1.0, index=dates))
    end = ms1[s_i + H]
    led = led.loc[:end - pd.Timedelta(days=1)]
    led = led[led.index < end]
    m = dca_metrics(led)
    print('SPOT CHECK window start', s0.date(), 'rolling final', round(chk['final'], 4), 'irr', round(chk['irr'], 5),
          '| DCABacktest final', round(m['final_value'], 4), 'irr', round(m['irr'], 5))
    assert abs(chk['final'] - m['final_value']) < 1e-6 * m['final_value'] and abs(chk['irr'] - m['irr']) < 1e-6

    # ---------------------------------------------------------------- path risk
    def heads(cols):  # cols = [S, Q, B]
        S, Q, B = cols
        tr = lambda a, b, c: {S: a / 10, Q: b / 10, B: c / 10}
        return [tr(10, 0, 0), tr(0, 10, 0), tr(0, 0, 10), tr(8, 0, 2), tr(7, 0, 3), tr(5, 0, 5), tr(3, 0, 7),
                tr(7, 3, 0), tr(4, 3, 3), tr(5, 2, 3), tr(6, 2, 2), tr(4, 2, 4)]
    PR = {}
    PR['modern_1999'] = path_risk(o1, c1, heads(mc))
    PR['long_1985'] = path_risk(o2, c2, heads(lc))
    PR['vfinx_brka_1980'] = path_risk(o2b, c2b, [{'VFINX': 1.0}, {'BRK-A': 1.0}, {'VFINX': .7, 'BRK-A': .3},
                                                 {'VFINX': .5, 'BRK-A': .5}])
    PR['broad_2011'] = path_risk(o3, c3, mixes3)
    PR['intl_1996'] = path_risk(o4, c4, mixes4)
    allpr = []
    for k, d in PR.items():
        d.insert(0, 'sample', k)
        allpr.append(d)
    save(pd.concat(allpr), 'path_risk_full_sample.csv')

    # ---------------------------------------------------------------- Berkshire edge decay
    px = load_prices(['VFINX', 'BRK-A', '^NDX', 'SPY', 'QQQ', 'BRK-B'], how='outer')
    cl = px['close']
    ye = cl.groupby(cl.index.year).last()          # year-end close
    vb = load_prices(['VFINX', 'BRK-A'], how='inner', start='1980-03-01')['close']
    yb = vb.groupby(vb.index.year).last()
    yb_first = vb.iloc[0]
    # 5-year blocks 1981-85 ... 2021-25 (year-end to year-end)
    rows = []
    for y0 in range(1981, 2026, 5):
        y1 = y0 + 4
        n = 5
        rb = (yb.loc[y1, 'BRK-A'] / yb.loc[y0 - 1, 'BRK-A']) ** (1 / n) - 1
        rs = (yb.loc[y1, 'VFINX'] / yb.loc[y0 - 1, 'VFINX']) ** (1 / n) - 1
        rows.append(dict(block=f'{y0}-{y1}', brk_a_ann=rb, vfinx_ann=rs, excess=rb - rs))
    # YTD 2026
    last = vb.iloc[-1]
    ytd_years = (vb.index[-1] - pd.Timestamp('2025-12-31')).days / 365.25
    rows.append(dict(block=f'2026 YTD to {vb.index[-1].date()} (not annualised)', brk_a_ann=last['BRK-A'] / yb.loc[2025, 'BRK-A'] - 1,
                     vfinx_ann=last['VFINX'] / yb.loc[2025, 'VFINX'] - 1,
                     excess=last['BRK-A'] / yb.loc[2025, 'BRK-A'] - last['VFINX'] / yb.loc[2025, 'VFINX']))
    blocks = pd.DataFrame(rows)
    save(blocks, 'brk_5y_blocks_vs_vfinx.csv')
    # whole-period + sub-period annualised
    sub = []
    for a, b in [('1980-03-01', '2025-12-31'), ('1980-03-01', '1999-12-31'), ('2000-01-01', '2012-12-31'),
                 ('2013-01-01', '2026-11-13'), ('2000-01-01', '2026-11-13'), ('2010-01-01', '2026-11-13')]:
        d = vb.loc[a:b]
        yy = (d.index[-1] - d.index[0]).days / 365.25
        r = (d.iloc[-1] / d.iloc[0]) ** (1 / yy) - 1
        sub.append(dict(period=f'{d.index[0].date()} to {d.index[-1].date()}', years=yy, brk_a_ann=r['BRK-A'],
                        vfinx_ann=r['VFINX'], excess=r['BRK-A'] - r['VFINX']))
    save(pd.DataFrame(sub), 'brk_subperiod_annualised.csv')

    # rolling 10y excess (month-end sampled)
    me = vb.resample('ME').last()
    r10 = (me / me.shift(120)) ** (1 / 10) - 1
    ex10 = (r10['BRK-A'] - r10['VFINX']).dropna()
    r5 = (me / me.shift(60)) ** (1 / 5) - 1
    ex5 = (r5['BRK-A'] - r5['VFINX']).dropna()
    ndx = load_prices(['^NDX', 'BRK-A'], how='inner')['close'].resample('ME').last()
    rn10 = (ndx / ndx.shift(120)) ** (1 / 10) - 1
    exn10 = (rn10['BRK-A'] - rn10['^NDX']).dropna()
    roll = pd.DataFrame({'brk_a_minus_vfinx_10y': ex10, 'brk_a_minus_vfinx_5y': ex5, 'brk_a_minus_ndx_price_10y': exn10})
    roll.index.name = 'month_end'
    save(roll.reset_index(), 'brk_rolling_excess.csv')
    pos_share = float((ex10 > 0).mean())
    print('rolling 10y excess: share>0 =', round(pos_share, 3), 'last', round(ex10.iloc[-1], 4), 'min', round(ex10.min(), 3),
          'max', round(ex10.max(), 3))

    # calendar-year tables
    def cal(a_col, b_col, src, first_year):
        yy = src.groupby(src.index.year).last()
        ret = yy.pct_change()
        return ret, yy

    yrs_ret = yb.pct_change()
    ytd = {'VFINX': last['VFINX'] / yb.loc[2025, 'VFINX'] - 1, 'BRK-A': last['BRK-A'] / yb.loc[2025, 'BRK-A'] - 1}
    t = pd.DataFrame({'year': yrs_ret.index, 'vfinx': yrs_ret['VFINX'].values, 'brk_a': yrs_ret['BRK-A'].values})
    t = t[t.year >= 1981]
    t.loc[t.year == 2026, ['vfinx', 'brk_a']] = [ytd['VFINX'], ytd['BRK-A']]
    # 2026 row exists only if yb has 2026 (it does: partial year last close)
    t['brk_minus_vfinx'] = t['brk_a'] - t['vfinx']
    t['partial_year'] = t.year == 2026
    down = t[t.vfinx < 0].copy()
    save(down, 'brk_vs_sp500_in_down_years.csv')
    save(t, 'brk_vs_sp500_all_years.csv')

    nd = load_prices(['^NDX', 'BRK-A', 'QQQ', 'BRK-B'], how='outer')['close']
    ny = nd.groupby(nd.index.year).last()
    nr = ny.pct_change()
    tn = pd.DataFrame({'year': nr.index, 'ndx_price': nr['^NDX'].values, 'qqq_total': nr['QQQ'].values,
                       'brk_a': nr['BRK-A'].values})
    tn = tn[tn.year >= 1986]
    tn['brk_minus_ndx_price'] = tn['brk_a'] - tn['ndx_price']
    tn['brk_minus_qqq'] = tn['brk_a'] - tn['qqq_total']
    tn['partial_year'] = tn.year == 2026
    # partial first QQQ year (1999) is year-end 1999 vs NaN -> NaN automatically; 2026 uses last close
    save(tn, 'brk_vs_ndx_all_years.csv')
    save(tn[tn.ndx_price < 0], 'brk_vs_ndx_in_ndx_down_years.csv')
    print(tn[tn.ndx_price < 0].round(3).to_string())
    print(down.round(3).to_string())

    # ---------------------------------------------------------------- figures
    head_labels1 = [label_of(w) for w in heads(mc)]
    head_labels2 = [label_of(w) for w in heads(lc)]
    boxplots(W1, head_labels1, [5, 10, 15, 20],
             'Modern sample (SPY/QQQ/BRK-B, DCA starts 1999-03 to 2026): distribution of DCA IRR over all monthly start dates',
             'fig1_irr_boxplots_modern.png', '\nOverlapping windows - effective independent observations ~ sample years / horizon')
    boxplots(W2, head_labels2, [5, 10, 15, 20],
             'Long sample (VFINX / BRK-A / NDX price index*, starts 1985-10 to 2026): distribution of DCA IRR over all start dates',
             'fig2_irr_boxplots_long.png', '\n*NDX excludes dividends (understates total return by ~1% p.a.)')
    simplex_heatmap(S1, mc, 'fig3_simplex_heatmap_modern.png',
                    'SPY / QQQ / BRK-B mixes (weights in 10% steps), modern sample 1999-03+: median and p10 DCA IRR')
    frontier_scatter([('Modern SPY/QQQ/BRK-B', S1, mc), ('Long VFINX/NDX(px)/BRK-A', S2, lc)], 'fig4_frontier_scatter.png',
                     'Median vs p10 DCA IRR for each fixed mix (higher and to the right is better)')
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    ax[0].bar(range(len(blocks) - 1), (blocks['excess'].iloc[:-1] * 100).to_numpy(), color=[
        '#2ca02c' if v > 0 else '#d62728' for v in blocks['excess'].iloc[:-1]])
    ax[0].set_xticks(range(len(blocks) - 1))
    ax[0].set_xticklabels(blocks['block'].iloc[:-1], fontsize=8)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_ylabel('BRK-A minus VFINX,\nannualised, pct points')
    ax[0].set_title('Non-overlapping 5-year blocks (year-end to year-end)')
    ax[0].grid(axis='y', alpha=0.3)
    ax[1].plot(ex10.index, ex10 * 100, color='#1f77b4', lw=1.5, label='BRK-A minus VFINX (total return)')
    ax[1].plot(exn10.index, exn10 * 100, color='#ff7f0e', lw=1.0, ls='--', label='BRK-A minus NDX (price index, no dividends)')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_ylabel('rolling 10y annualised excess,\npct points')
    ax[1].set_title('Rolling 10-year annualised excess return of Berkshire A (month-end sampled)')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].plot(ex5.index, ex5 * 100, color='#1f77b4', lw=1.0)
    ax[2].axhline(0, color='k', lw=0.6)
    ax[2].set_ylabel('rolling 5y annualised excess,\npct points')
    ax[2].set_title('Rolling 5-year annualised excess return, BRK-A minus VFINX')
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_brk_edge_decay.png'), dpi=DPI)
    plt.close(fig)
    regret_chart([('Modern (SPY/QQQ/BRK-B)', S1, head_labels1), ('Long (VFINX/NDX(px)/BRK-A)', S2, head_labels2)],
                 'fig6_regret_vs_best_single.png', 'Minimax-regret view of fixed mixes (final value / best single asset in the same window)')

    # decade heat table: median IRR by start decade at 10y, long sample
    d = S2d[S2d.horizon_y == 10]
    piv = d.pivot(index='mix', columns='start_decade', values='irr_median').reindex(head_labels2) * 100
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8)
    ax.set_xlabel('decade in which the 10-year DCA started')
    ax.set_title('Long sample (VFINX / NDX price / BRK-A): median 10y DCA IRR (% p.a.) by start decade')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_decade_dependence_long_10y.png'), dpi=DPI)
    plt.close(fig)

    # ---------------------------------------------------------------- console digest
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 400)
    cols = ['mix', 'horizon_y', 'n_windows', 'irr_median', 'irr_p10', 'irr_min', 'wm_median', 'wm_min', 'pct_beat_ref',
            'regret_best_min', 'regret_best_p10']
    for nm, s in [('MODERN singles+pair', S1[S1['w_QQQ'] == 0]), ('LONG all', S2[S2['w_^NDX'] == 0]), ('S2b', S2b), ('S3', S3), ('S4', S4)]:
        print('\n==', nm)
        print(s[cols].round(3).to_string(index=False))
    print(P3.round(3).to_string(index=False))
    for k, v in PR.items():
        print(k)
        print(v.round(3).to_string(index=False))
    print(blocks.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
