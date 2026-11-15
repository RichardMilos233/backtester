"""EXP4 - how does the size of the monthly contribution X change what a DCA plan should hold?

Sections: A whole shares vs fractional, B diversification per extra fund, C catch-up from today's
account, D when contributions stop being able to rebalance, E scale-invariance of IRR + projections.
Run from the repo root:  python scripts/dca/exp4_contribution_size.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dca import DCABacktest, buy_underweight, fixed_buy, load_prices, month_starts, rolling_dca

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca', 'exp4_contribution_size')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
END = '2026-11-13'
SLIP = 0.0005
XS = [100, 200, 300, 500, 1000, 2000, 3000, 5000]
XS_A = [25, 50] + XS            # 25 and 50 are extras, to find where fractional shares stop working
DPI = 130
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def save_csv(df, name, **kw):
    df.to_csv(os.path.join(OUT, name), **kw)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=DPI)
    plt.close(fig)


def heat(ax, M, rows, cols, title, fmt='{:.1f}', cmap='YlOrRd', vmax=None, xlabel='', ylabel=''):
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=vmax if vmax else np.nanmax(M))
    ax.set_xticks(range(len(cols)), [str(c) for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(rows)), [str(r) for r in rows])
    thr = (vmax if vmax else np.nanmax(M)) * 0.6
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=7.5,
                    color='white' if M[i, j] > thr else 'black')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


# =========================================================================== A
PORTS = {
    1: {'VOO': 1.0},
    2: {'VOO': .5, 'BRK-B': .5},
    3: {'VOO': .5, 'BRK-B': .3, 'VXUS': .2},
    5: {'VOO': .4, 'BRK-B': .2, 'VXUS': .2, 'QQQ': .1, 'SCHD': .1},
    8: {'VOO': .3, 'BRK-B': .15, 'VXUS': .15, 'QQQ': .1, 'SCHD': .1, 'VTV': .05, 'BND': .1, 'GLD': .05},
}
MODES = {'whole': dict(fractional=False, min_order=0.0),
         'frac': dict(fractional=True, min_order=0.0),
         'frac_min5': dict(fractional=True, min_order=5.0)}


def run_A():
    allt = list(PORTS[8])
    px = load_prices(allt, start='2016-01-01', end=END)
    last = px['close'].iloc[-1].round(2)
    save_csv(last.rename('last_adjusted_close').to_frame(), 'A_latest_prices.csv')
    print('Latest adjusted closes', px['close'].index[-1].date(), dict(last))
    wins = {'5y': '2021-11-14', '10y': '2016-11-14'}
    rows = []
    for wname, wstart in wins.items():
        for N, tgt in PORTS.items():
            tick = list(tgt)
            p = load_prices(tick, start=wstart, end=END)
            o, c = p['open'], p['close']
            dec = month_starts(o.index)
            tw = np.array([tgt[t] for t in tick])
            for X in XS_A:
                res = {}
                for mname, kw in MODES.items():
                    led = DCABacktest(SLIP, 0.0, **kw).run(o, c, buy_underweight(tgt), pd.Series(float(X), index=dec))
                    tv = led['total_value'].to_numpy()
                    W = np.column_stack([led[f'value_{t}'].to_numpy() / tv for t in tick])
                    dev = np.abs(W - tw)
                    last252 = slice(-252, None)
                    res[mname] = dict(
                        idle_pct=100 * np.mean(led['cash'].to_numpy() / tv),
                        idle_pct_last12m=100 * np.mean(led['cash'].to_numpy()[last252] / tv[last252]),
                        mad_pp=100 * np.mean(dev.mean(axis=1)),
                        maxdev_pp=100 * np.mean(dev.max(axis=1)),
                        maxdev_pp_last12m=100 * np.mean(dev.max(axis=1)[last252]),
                        mad_pp_last12m=100 * np.mean(dev.mean(axis=1)[last252]),
                        no_buy_months=int((led.loc[dec, 'n_trades'] == 0).sum()),
                        n_months=len(dec), final=tv[-1], final_cash=led['cash'].iloc[-1],
                        contributed=led['cum_contribution'].iloc[-1])
                for mname, r in res.items():
                    r = dict(r)
                    r['shortfall_pct'] = 100 * (res['frac']['final'] - r['final']) / res['frac']['final']
                    rows.append(dict(window=wname, N=N, X=X, mode=mname, **r))
    df = pd.DataFrame(rows)
    save_csv(df.round(3), 'A_whole_vs_fractional.csv', index=False)
    return df


def fig_A(df):
    for metric, fname, title, vmax in [('maxdev_pp', 'fig1_weight_deviation_heatmap.png',
                                        'Time-average of the largest asset-weight deviation from target (pp)', 15),
                                       ('idle_pct', 'fig2_idle_cash_heatmap.png',
                                        'Time-average idle cash, % of account', 15)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, w in zip(axes, ['5y', '10y']):
            d = df[(df['mode'] == 'whole') & (df['window'] == w)].pivot(index='N', columns='X', values=metric)
            im = heat(ax, d.values, [f'N={n}' for n in d.index], d.columns, f'whole shares, last {w}',
                      vmax=vmax, xlabel='monthly contribution X ($)')
        fig.suptitle(title + ' - buy_underweight, from zero, whole shares', fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(os.path.join(OUT, fname), dpi=DPI, bbox_inches='tight')
        plt.close(fig)


def summarise_A(df):
    out = []
    for w in ['5y', '10y']:
        for N in PORTS:
            for metric in ['maxdev_pp', 'mad_pp', 'maxdev_pp_last12m']:
                row = {'window': w, 'N': N, 'metric': metric}
                for mode in ['whole', 'frac', 'frac_min5']:
                    d = df[(df.window == w) & (df.N == N) & (df['mode'] == mode)].sort_values('X')
                    ok = (d[metric] <= 2.0).to_numpy()
                    # smallest X such that it (and every larger X on the grid) is within 2pp
                    sm = np.nan
                    for i in range(len(ok)):
                        if ok[i:].all():
                            sm = d['X'].iloc[i]
                            break
                    row[mode] = sm
                out.append(row)
    s = pd.DataFrame(out)
    save_csv(s, 'A_smallest_X_within_2pp.csv', index=False)
    return s


# =========================================================================== B
SLEEVES = ['VOO', 'VTI', 'QQQ', 'BRK-B', 'VXUS', 'SCHD', 'VTV', 'RSP', 'BND', 'GLD']


def monthly_rebal_nav(rets: pd.DataFrame, w: dict) -> pd.Series:
    cols = list(w)
    wv = np.array([w[c] for c in cols])
    r = rets[cols]
    key = r.index.year * 12 + r.index.month
    level = 1.0
    out = np.empty(len(r))
    for k in pd.unique(key):
        m = key == k
        g = np.cumprod(1 + r.to_numpy()[m], axis=0)
        path = level * (g @ wv)
        out[m] = path
        level = path[-1]
    return pd.Series(out, index=r.index)


def nav_stats(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr_pct=100 * ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1),
                vol_pct=100 * r.std() * np.sqrt(252),
                max_dd_pct=100 * (nav / nav.cummax() - 1).min(),
                worst_12m_pct=100 * nav.pct_change(252).min())


def run_B():
    p = load_prices(SLEEVES, start='2011-10-20', end=END)['close']
    rets = p.pct_change().dropna()
    corr = rets.corr()
    save_csv(corr.round(3), 'B_correlation_matrix.csv')
    print('B common sample', rets.index[0].date(), rets.index[-1].date(), len(rets))
    rows = []
    prev = None
    for N, tgt in PORTS.items():
        s = nav_stats(monthly_rebal_nav(rets, tgt))
        s.update(N=N, sleeves='/'.join(f'{k} {v*100:.0f}' for k, v in tgt.items()))
        s['vol_change_pp'] = np.nan if prev is None else s['vol_pct'] - prev
        s['vol_change_rel_pct'] = np.nan if prev is None else 100 * (s['vol_pct'] / prev - 1)
        prev = s['vol_pct']
        rows.append(s)
    modern = pd.DataFrame(rows)[['N', 'sleeves', 'cagr_pct', 'vol_pct', 'max_dd_pct', 'worst_12m_pct',
                                 'vol_change_pp', 'vol_change_rel_pct']]
    save_csv(modern.round(2), 'B_mix_risk_2011.csv', index=False)

    lp = load_prices(['VFINX', 'BRK-A', 'VGTSX', 'VBMFX'], start='1996-05-01', end=END)['close']
    lr = lp.pct_change().dropna()
    LONG = {'VFINX 100': {'VFINX': 1.0},
            'VFINX 60 / BRK-A 40': {'VFINX': .6, 'BRK-A': .4},
            'VFINX 50 / BRK-A 30 / VGTSX 20': {'VFINX': .5, 'BRK-A': .3, 'VGTSX': .2},
            'VFINX 45 / BRK-A 25 / VGTSX 15 / VBMFX 15': {'VFINX': .45, 'BRK-A': .25, 'VGTSX': .15, 'VBMFX': .15}}
    rows, prev = [], None
    for name, w in LONG.items():
        s = nav_stats(monthly_rebal_nav(lr, w))
        s.update(mix=name)
        s['vol_change_pp'] = np.nan if prev is None else s['vol_pct'] - prev
        prev = s['vol_pct']
        rows.append(s)
    long_df = pd.DataFrame(rows)[['mix', 'cagr_pct', 'vol_pct', 'max_dd_pct', 'worst_12m_pct', 'vol_change_pp']]
    save_csv(long_df.round(2), 'B_mix_risk_long_1996.csv', index=False)
    print('B long sample', lr.index[0].date(), lr.index[-1].date(), len(lr))

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr.values, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v < 0.6 else 'black')
    ax.set_title(f'Daily-return correlation, {rets.index[0].date()} to {rets.index[-1].date()}')
    fig.colorbar(im, shrink=0.8)
    savefig(fig, 'fig5_correlation_heatmap.png')

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, col, lab in zip(axes, ['vol_pct', 'max_dd_pct', 'worst_12m_pct'],
                            ['Annualised volatility (%)', 'Max drawdown (%)', 'Worst 12-month return (%)']):
        ax.plot(modern['N'], modern[col], 'o-', color='#2a6f97')
        ax.set_xticks(modern['N'])
        ax.set_xlabel('number of funds N')
        ax.set_title(lab, fontsize=9)
    fig.suptitle('Risk of the N-fund mixes (monthly rebalanced), common sample 2011-11 to 2026-11', fontsize=10)
    savefig(fig, 'fig6_risk_by_N.png')
    return corr, modern, long_df


# =========================================================================== C
C_START = {'NVDA': 1000., 'SPCX': 1000., 'BRK-B': 2000., 'VOO': 700., 'QQQ': 2300.}
ASSETS_C = ['NVDA', 'SPCX', 'BRK-B', 'VOO', 'QQQ', 'VXUS']
CORE_DEF = {'i: VOO+VXUS': {'VOO': .7, 'VXUS': .3},
            'ii: VOO+BRK-B+VXUS': {'VOO': .5, 'BRK-B': .3, 'VXUS': .2}}
CORE_SET = {'i: VOO+VXUS': ['VOO', 'VXUS'], 'ii: VOO+BRK-B+VXUS': ['VOO', 'BRK-B', 'VXUS']}
CAP = 120
GOALS = {'core>=50%': ('core', .5), 'core>=80%': ('core', .8), 'NVDA+SPCX<=20%': ('conc', .2),
         'NVDA+SPCX<=10%': ('conc', .1)}


def analytic_months(X, goal, core0, total0=7000., conc0=2000.):
    kind, thr = GOALS[goal]
    if kind == 'core':
        need = (thr * total0 - core0) / (X * (1 - thr))
    else:
        need = (conc0 / thr - total0) / X
    return max(0, int(np.ceil(need - 1e-12)))


def build_C_prices():
    tk = ['NVDA', 'TSLA', 'BRK-B', 'VOO', 'QQQ', 'VXUS', 'SPY', 'EFA']
    p = load_prices(tk, start='2010-06-29', end=END, how='outer')
    o, c = p['open'].copy(), p['close'].copy()

    def splice(new, old):
        first = c[new].first_valid_index()
        k = c.loc[first, new] / c.loc[first, old]
        for fr in (o, c):
            fr[new] = fr[new].where(fr[new].notna(), fr[old] * k)
    splice('VOO', 'SPY')
    splice('VXUS', 'EFA')
    # proxy for SPCX built from 1.5 x QQQ (close-to-close and overnight), self-consistent index
    qc, qo = c['QQQ'], o['QQQ']
    r_cc = qc.pct_change().fillna(0)
    r_on = (qo / qc.shift(1) - 1).fillna(0)
    close15 = 100 * (1 + 1.5 * r_cc).cumprod()
    open15 = close15.shift(1) * (1 + 1.5 * r_on)
    open15.iloc[0] = close15.iloc[0]
    o['SPCX_Q15'], c['SPCX_Q15'] = open15, close15
    o['SPCX_TSLA'], c['SPCX_TSLA'] = o['TSLA'], c['TSLA']
    return o, c


def sim_C(Pm, proxy, coredef, sale, X, slip=SLIP):
    """Vectorised over start months. Pm: (T, 6) monthly open prices in ASSETS_C order (NVDA, SPCX, BRK-B, VOO, QQQ, VXUS).
    Contribution k (k=0..119) is spent at the open of row s+k; condition checked after the purchase, marked at that open.
    Returns dict goal -> months array (NaN = censored by end of data)."""
    T = len(Pm)
    S = np.arange(T)
    w = np.array([CORE_DEF[coredef].get(a, 0.0) for a in ASSETS_C])
    core_mask = np.array([a in CORE_SET[coredef] for a in ASSETS_C])
    # a core (ii) includes BRK-B; core (i) does not count BRK-B
    v0 = np.array([C_START.get(a, 0.0) for a in ASSETS_C])
    units = v0[None, :] / Pm[S]
    if sale:
        sold = units[:, 4] * Pm[S, 4] * (1 - slip)
        units[:, 4] = 0.0
        units += (sold[:, None] * w[None, :]) / (Pm[S] * (1 + slip))
    res = {g: np.full(T, np.nan) for g in GOALS}
    done = {g: np.zeros(T, bool) for g in GOALS}
    cens_end = np.zeros(T, bool)

    def check(val, k):
        tot = val.sum(axis=1)
        core = val[:, core_mask].sum(axis=1) / tot
        conc = (val[:, 0] + val[:, 1]) / tot
        for g, (kind, thr) in GOALS.items():
            hit = (core >= thr) if kind == 'core' else (conc <= thr)
            new = hit & ~done[g] & valid
            res[g][new] = k
            done[g] |= new

    valid = np.ones(T, bool)
    check(units * Pm[S], 0)
    for k in range(CAP):
        idx = S + k
        valid = idx < T
        P = Pm[np.minimum(idx, T - 1)]
        units = units + (X * w[None, :]) / (P * (1 + slip))
        check(units * P, k + 1)
    avail = np.minimum(CAP, T - S)
    for g in GOALS:
        miss = ~done[g]
        res[g][miss & (avail >= CAP)] = CAP          # not reached within the cap
        # misses with < CAP months of data remain NaN (censored)
    return res


def run_C():
    o, c = build_C_prices()
    ms = month_starts(o.loc['2010-07-01':].index)
    ms = ms[:]
    analytic = []
    for dname, cs in CORE_SET.items():
        for sale in (False, True):
            core0 = sum(C_START.get(a, 0) for a in cs) + (2300 if sale else 0)
            for X in XS:
                row = dict(core=dname, sell_other=sale, X=X)
                for g in GOALS:
                    row[g] = analytic_months(X, g, core0)
                analytic.append(row)
    an = pd.DataFrame(analytic)
    save_csv(an, 'C_analytic_zero_return.csv', index=False)

    sims = []
    for proxy in ['SPCX_TSLA', 'SPCX_Q15']:
        cols = ['NVDA', proxy, 'BRK-B', 'VOO', 'QQQ', 'VXUS']
        Pm = o.loc[ms, cols].to_numpy(dtype=float)
        assert np.isfinite(Pm).all()
        for dname in CORE_DEF:
            for sale in (False, True):
                for X in XS:
                    r = sim_C(Pm, proxy, dname, sale, float(X))
                    for g, a in r.items():
                        a2 = a[np.isfinite(a)]
                        sims.append(dict(proxy=proxy, core=dname, sell_other=sale, X=X, goal=g,
                                         n_starts=len(a2), n_censored=int(np.isnan(a).sum()),
                                         median=np.median(a2), p10=np.percentile(a2, 10),
                                         p90=np.percentile(a2, 90), share_not_reached=np.mean(a2 >= CAP)))
    sm = pd.DataFrame(sims)
    save_csv(sm.round(2), 'C_simulated_months.csv', index=False)
    print('C starts', ms[0].date(), ms[-1].date(), len(ms))

    # spot check one start against the engine
    spot = spot_check_C(o, c, ms)
    return an, sm, spot


def spot_check_C(o, c, ms):
    proxy, dname, X, start = 'SPCX_TSLA', 'ii: VOO+BRK-B+VXUS', 500, pd.Timestamp('2015-01-02')
    cols = ['NVDA', proxy, 'BRK-B', 'VOO', 'QQQ', 'VXUS']
    oo, cc = o[cols].loc[start:], c[cols].loc[start:]
    oo = oo.rename(columns={proxy: 'SPCX'})
    cc = cc.rename(columns={proxy: 'SPCX'})
    dec = month_starts(oo.index)[:120]
    w = CORE_DEF[dname]
    led = DCABacktest(SLIP, 0.0).run(oo, cc, fixed_buy(w), pd.Series(float(X), index=dec), initial_values=dict(C_START))
    ld = led.loc[dec]
    out = {}
    core = (ld['value_VOO'] + ld['value_BRK-B'] + ld['value_VXUS']) / ld['total_value']
    conc = (ld['value_NVDA'] + ld['value_SPCX']) / ld['total_value']
    out['engine_core50'] = int(np.argmax(core.to_numpy() >= .5) + 1) if (core >= .5).any() else None
    out['engine_core80'] = int(np.argmax(core.to_numpy() >= .8) + 1) if (core >= .8).any() else None
    out['engine_conc20'] = int(np.argmax(conc.to_numpy() <= .2) + 1) if (conc <= .2).any() else None
    T = len(ms)
    i0 = list(ms).index(start)
    Pm = o.loc[ms, cols].to_numpy(dtype=float)
    r = sim_C(Pm, proxy, dname, False, float(X))
    out['sim_core50'], out['sim_core80'], out['sim_conc20'] = r['core>=50%'][i0], r['core>=80%'][i0], r['NVDA+SPCX<=20%'][i0]
    print('C spot check (start 2015-01, TSLA proxy, core ii, X=500):', out)
    return out


def fig_C(an, sm):
    for goals, fname, ttl in [(['core>=50%', 'core>=80%'], 'fig3_catchup_core_share.png', 'Months until core share reaches target'),
                              (['NVDA+SPCX<=20%', 'NVDA+SPCX<=10%'], 'fig3b_catchup_concentration.png',
                               'Months until NVDA+SPCX share falls to target')]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for i, g in enumerate(goals):
            for j, dname in enumerate(CORE_DEF):
                ax = axes[i, j]
                a = an[(an.core == dname) & (~an.sell_other)]
                ax.plot(a.X, np.minimum(a[g], CAP + 10), 'k--', label='zero returns (analytic)')
                for proxy, col in [('SPCX_TSLA', '#2a6f97'), ('SPCX_Q15', '#c1121f')]:
                    d = sm[(sm.proxy == proxy) & (sm.core == dname) & (~sm.sell_other) & (sm.goal == g)]
                    ax.plot(d.X, d['median'], 'o-', color=col, label=f'median, SPCX~{proxy[5:]}')
                    ax.fill_between(d.X, d.p10, d.p90, color=col, alpha=.12)
                ax.axhline(CAP, color='grey', lw=.7)
                ax.set_xscale('log')
                ax.set_xticks(XS, [str(x) for x in XS], rotation=45)
                ax.set_ylim(0, CAP + 15)
                ax.set_title(f'{g} | core {dname}', fontsize=9)
                if i == 1:
                    ax.set_xlabel('monthly contribution X ($)')
                ax.set_ylabel('months (capped at 120)')
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(ttl + ' (band = p10-p90 over historical start months 2010-07 on)', fontsize=10)
        savefig(fig, fname)


# =========================================================================== D
def run_D():
    cfgs = {'VOO50/BRK-B30/VXUS20 (2011-02+)': ({'VOO': .5, 'BRK-B': .3, 'VXUS': .2}, '2011-02-01'),
            'SPY50/QQQ20/BRK-B30 (1999-03+)': ({'SPY': .5, 'QQQ': .2, 'BRK-B': .3}, '1999-03-01')}
    recs = []
    for cname, (tgt, start) in cfgs.items():
        tick = list(tgt)
        p = load_prices(tick, start=start, end=END)
        o, c = p['open'], p['close']
        dec = month_starts(o.index)
        tw = np.array([tgt[t] for t in tick])
        for growth in ('constant', 'growing 5%/yr'):
            if growth == 'constant':
                contrib = pd.Series(500.0, index=dec)
            else:
                yrs = (dec - dec[0]).days / 365.25
                contrib = pd.Series(500.0 * 1.05 ** np.floor(yrs), index=dec)   # raise once a year
            led = DCABacktest(SLIP, 0.0, fractional=True).run(o, c, buy_underweight(tgt), contrib)
            tv = led['total_value'].to_numpy()
            W = np.column_stack([led[f'value_{t}'].to_numpy() / tv for t in tick])
            dev = np.abs(W - tw).max(axis=1)
            pos = led.index.get_indexer(dec)
            for k in range(1, len(dec)):
                recs.append(dict(config=cname, growth=growth, date=dec[k], X=contrib.iloc[k],
                                 value_before=tv[pos[k] - 1], ratio=contrib.iloc[k] / tv[pos[k] - 1],
                                 dev_pre_pp=100 * dev[pos[k] - 1], dev_post_pp=100 * dev[pos[k]]))
    d = pd.DataFrame(recs)
    d['bucket'] = pd.cut(d['ratio'], [0, .005, .01, .02, .05, np.inf],
                         labels=['<0.5%', '0.5-1%', '1-2%', '2-5%', '>5%'])
    save_csv(d.round(4), 'D_monthly_drift_records.csv', index=False)
    order = ['>5%', '2-5%', '1-2%', '0.5-1%', '<0.5%']
    g = d.groupby(['config', 'growth', 'bucket'], observed=True)
    t = g.agg(months=('dev_pre_pp', 'size'), mean_dev_pre=('dev_pre_pp', 'mean'),
              median_dev_pre=('dev_pre_pp', 'median'),
              p90_dev_pre=('dev_pre_pp', lambda x: np.percentile(x, 90)), max_dev_pre=('dev_pre_pp', 'max'),
              share_above_2pp=('dev_pre_pp', lambda x: 100 * np.mean(x > 2)),
              share_above_5pp=('dev_pre_pp', lambda x: 100 * np.mean(x > 5)),
              mean_dev_post=('dev_post_pp', 'mean'), max_dev_post=('dev_post_pp', 'max')).reset_index()
    t['bucket'] = pd.Categorical(t['bucket'], order, ordered=True)
    t = t.sort_values(['config', 'growth', 'bucket'])
    save_csv(t.round(2), 'D_drift_by_ratio_bucket.csv', index=False)
    # when does drift >5pp first appear at low ratio, and the largest observed deviations
    ex = []
    for (cn, gr), x in d.groupby(['config', 'growth']):
        big = x[x.dev_pre_pp > 5]
        low = x[x.ratio < 0.02]
        ex.append(dict(config=cn, growth=gr, months=len(x), months_above_5pp=len(big),
                       first_above_5pp=big.date.min().date() if len(big) else None,
                       ratio_at_first=round(big.sort_values('date').ratio.iloc[0], 4) if len(big) else None,
                       max_ratio_when_above_5pp_after_ratio_below_2pp=None,
                       max_dev_when_ratio_below_2pp=round(low.dev_pre_pp.max(), 2) if len(low) else None,
                       months_above_5pp_when_ratio_below_2pp=int((low.dev_pre_pp > 5).sum()),
                       months_ratio_below_2pp=len(low)))
    ex = pd.DataFrame(ex).drop(columns='max_ratio_when_above_5pp_after_ratio_below_2pp')
    save_csv(ex, 'D_drift_summary.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    for i, cn in enumerate(cfgs):
        for j, gr in enumerate(['constant', 'growing 5%/yr']):
            ax = axes[i, j]
            x = d[(d.config == cn) & (d.growth == gr)]
            sc = ax.scatter(100 * x.ratio, x.dev_pre_pp, c=x.date.dt.year + x.date.dt.month / 12, s=8, cmap='viridis')
            ax.axhline(5, color='#c1121f', lw=.8, ls='--')
            ax.axhline(2, color='grey', lw=.8, ls=':')
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_title(f'{cn}, contributions {gr}', fontsize=8.5)
            ax.set_ylabel('max |weight - target| just before the contribution (pp)')
            if i == 1:
                ax.set_xlabel('monthly contribution / account value (%, log; account grows to the right)')
            fig.colorbar(sc, ax=ax, label='year')
    savefig(fig, 'fig4_drift_vs_contribution_ratio.png')
    return t, ex


# =========================================================================== E
def run_E():
    p = load_prices(['VFINX', 'BRK-A'], end=END)
    o, c = p['open'], p['close']
    ms = month_starts(o.index)
    alloc = pd.DataFrame({'VFINX': .6, 'BRK-A': .4}, index=ms)
    print('E sample', o.index[0].date(), o.index[-1].date())
    stats = {}
    for H in (120, 240):
        r = rolling_dca(o, c, alloc, H, 100.0)
        stats[H] = r
    r20 = stats[240]
    q = r20['irr'].quantile([.1, .5, .9])
    st = pd.DataFrame({'horizon_years': 20, 'n_windows': len(r20), 'first_start': r20.start.min().date(),
                       'last_start': r20.start.max().date(),
                       'irr_p10': q[.1], 'irr_median': q[.5], 'irr_p90': q[.9]}, index=[0])
    save_csv(st.round(4), 'E_vfinx_brka_6040_20y_irr.csv', index=False)

    # scale-invariance check: 4 mixes, 10-year horizon, 1996-05+ sample, amount 100 vs 5000
    p2 = load_prices(['VFINX', 'BRK-A', 'VGTSX', 'VBMFX'], start='1996-05-01', end=END)
    o2, c2 = p2['open'], p2['close']
    ms2 = month_starts(o2.index)
    mixes = {'VFINX 100': {'VFINX': 1.0}, '60/40 VFINX/BRK-A': {'VFINX': .6, 'BRK-A': .4},
             '50/30/20 +VGTSX': {'VFINX': .5, 'BRK-A': .3, 'VGTSX': .2},
             '45/25/15/15 +VBMFX': {'VFINX': .45, 'BRK-A': .25, 'VGTSX': .15, 'VBMFX': .15}}
    rows, maxdiff = [], 0
    for name, w in mixes.items():
        al = pd.DataFrame({a: w.get(a, 0.0) for a in o2.columns}, index=ms2)
        small = rolling_dca(o2, c2, al, 120, 100.0)
        big = rolling_dca(o2, c2, al, 120, 5000.0)
        maxdiff = max(maxdiff, float((small.irr - big.irr).abs().max()))
        rows.append(dict(mix=name, n_windows=len(small), median_irr_X100=small.irr.median(),
                         median_irr_X5000=big.irr.median(),
                         max_abs_irr_diff=float((small.irr - big.irr).abs().max()),
                         wealth_multiple_median_X100=small.wealth_multiple.median(),
                         wealth_multiple_median_X5000=big.wealth_multiple.median()))
    sc = pd.DataFrame(rows)
    sc['rank_X100'] = sc.median_irr_X100.rank(ascending=False).astype(int)
    sc['rank_X5000'] = sc.median_irr_X5000.rank(ascending=False).astype(int)
    save_csv(sc.round(6), 'E_scale_invariance_check.csv', index=False)

    # projections: contributions at the start of each month, constant monthly-equivalent rate = historical IRR
    proj = []
    for lab, irr in [('0% (contributions only)', 0.0), (f'p10 20y IRR {q[.1]*100:.1f}%', q[.1]),
                     (f'median 20y IRR {q[.5]*100:.1f}%', q[.5])]:
        i = (1 + irr) ** (1 / 12) - 1
        for X in XS:
            row = dict(rate=lab, X=X)
            for yrs in (10, 20, 30):
                n = 12 * yrs
                row[f'{yrs}y'] = X * n if irr == 0 else X * ((1 + i) ** n - 1) / i * (1 + i)
            proj.append(row)
    pr = pd.DataFrame(proj)
    save_csv(pr.round(0), 'E_projected_balances.csv', index=False)
    return st, sc, pr, maxdiff


# =========================================================================== main
if __name__ == '__main__':
    pd.set_option('display.width', 250, 'display.max_columns', 50, 'display.max_rows', 500)
    print('=== A'); dfA = run_A(); fig_A(dfA); sA = summarise_A(dfA)
    print(sA.to_string())
    print('=== B'); corr, modern, long_df = run_B()
    print(modern.round(2).to_string()); print(long_df.round(2).to_string())
    print('=== C'); an, sm, spot = run_C(); fig_C(an, sm)
    print('=== D'); tD, exD = run_D()
    print(tD.round(2).to_string()); print(exD.to_string())
    print('=== E'); st, sc, pr, md = run_E()
    print(st.to_string()); print(sc.round(5).to_string()); print('max irr diff', md)
    print(pr.to_string())
