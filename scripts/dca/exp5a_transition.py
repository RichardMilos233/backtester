"""EXP5-A: six-month transition from a concentrated portfolio, no new cash."""
import multiprocessing as mp
import time

from exp5_common import *  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOTAL = 7000.0
P0 = {'NVDA': 14.3, 'SPCX': 14.3, 'BRK-B': 28.6, 'VOO': 10.0, 'QQQ': 32.9}
STATES = {
    'T1 80/20': {'VOO': 45, 'BRK-B': 25, 'INTL': 10, 'NVDA': 10, 'SPCX': 10},
    'T2 sell others->VOO': {'NVDA': 14.3, 'SPCX': 14.3, 'BRK-B': 28.6, 'VOO': 42.9},
    'T3 others->VOO+intl': {'NVDA': 14.3, 'SPCX': 14.3, 'BRK-B': 28.6, 'VOO': 32.9, 'INTL': 10},
    'T4 core 50/50': {'VOO': 50, 'BRK-B': 50},
}
PATHS = {'lump': 1, 'g3': 3, 'g6': 6}
ASSETS = ['NVDA', 'SPCX', 'BRK-B', 'VOO', 'QQQ', 'INTL']
PROXIES = ['TSLA', '1.5xQQQ']
OTHERS = ['QQQ', 'cash']
HS = (6, 12)


def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


P0N = norm(P0)
STATESN = {k: norm(v) for k, v in STATES.items()}

RAW = None
FR = {}
QQ = None
QFEAT = None


def build():
    global RAW, QQ
    o, c = load_outer(['NVDA', 'TSLA', 'BRK-B', 'SPY', 'VOO', 'QQQ', 'EFA', 'VXUS'])
    voo = splice(o, c, [('SPY', '1993-01-01'), ('VOO', '2010-09-09')])
    intl = splice(o, c, [('SPY', '1993-01-01'), ('EFA', '2001-08-27'), ('VXUS', '2011-02-01')])
    qqq = (o['QQQ'].dropna(), c['QQQ'].dropna())
    base = {'NVDA': (o['NVDA'].dropna(), c['NVDA'].dropna()), 'BRK-B': (o['BRK-B'].dropna(), c['BRK-B'].dropna()),
            'VOO': voo, 'QQQ': qqq, 'INTL': intl}
    for px in PROXIES:
        if px == 'TSLA':
            sp = (o['TSLA'].dropna(), c['TSLA'].dropna())
        else:
            sp = synth(*qqq, 1.5)
        FR[px] = frames({'NVDA': base['NVDA'], 'SPCX': sp, 'BRK-B': base['BRK-B'], 'VOO': base['VOO'],
                         'QQQ': base['QQQ'], 'INTL': base['INTL']})
    qc = c['QQQ'].dropna()
    QQ = qc
    return FR


def qqq_features(date):
    """QQQ trailing 12m return and distance from running high, using closes strictly before `date`."""
    pos = QQ.index.searchsorted(date) - 1
    if pos < 252:
        return np.nan, np.nan
    r12 = QQ.iloc[pos] / QQ.iloc[pos - 252] - 1
    dd = QQ.iloc[pos] / QQ.iloc[:pos + 1].max() - 1
    return r12, dd


def make_policy(target, n_steps):
    assets = ASSETS

    def policy(ctx):
        k = ctx.step
        if n_steps == 0 or k >= n_steps:
            return {}
        total = ctx.cash + float(ctx.values.sum())
        if total <= 0:
            return {}
        r = n_steps - k
        w = {}
        for a in assets:
            cur = float(ctx.values[a]) / total
            w[a] = cur + (target.get(a, 0.0) - cur) / r
        if r == 1:
            w = {a: target.get(a, 0.0) for a in assets}
        return {'target': w}
    return policy


def run_one(opens, closes, ms, i, H, other, policy):
    end = ms[i + H] if i + H < len(ms) else None
    sl = (opens.index >= ms[i]) & ((opens.index < end) if end is not None else True)
    o, c = opens[sl], closes[sl]
    dec = ms[i:i + H]
    contrib = pd.Series(0.0, index=dec)
    iv = {a: TOTAL * w for a, w in P0N.items() if not (a == 'QQQ' and other == 'cash')}
    ic = TOTAL * P0N['QQQ'] if other == 'cash' else 0.0
    led = ENGINE.run(o, c, policy, contrib, initial_values=iv, initial_cash=ic)
    return led


def series_stats(led, cut):
    tv = led['total_value'].to_numpy()[: cut]
    eq = np.r_[TOTAL, tv]
    r = np.diff(eq) / eq[:-1]
    return tv[-1], max_dd(eq), float(np.std(r[1:], ddof=1) * np.sqrt(252))


def task(args):
    proxy, other, i = args
    opens, closes = FR[proxy]
    ms = MS[proxy]
    rows = []
    Hmax = 12 if i + 12 < len(ms) else 6
    paths = {'hold': (None, 0)}
    for sname, tgt in STATESN.items():
        for pname, n in PATHS.items():
            paths[(sname, pname)] = (tgt, n)
    res = {}
    for key, (tgt, n) in paths.items():
        led = run_one(opens, closes, ms, i, Hmax, other, make_policy(tgt or {}, n))
        out = {}
        for H in HS:
            if H > Hmax:
                continue
            cut = int((led.index < ms[i + H]).sum())
            out[H] = series_stats(led, cut)
        res[key] = out
    r12, dd = qqq_features(ms[i])
    for key, out in res.items():
        for H, (ev, mdd, vol) in out.items():
            hold_ev = res['hold'][H][0]
            sname, pname = ('hold', 'hold') if key == 'hold' else key
            rows.append(dict(proxy=proxy, other=other, start=ms[i], H=H, state=sname, path=pname,
                             end_value=ev, rel_hold=ev / hold_ev - 1, mdd=mdd, vol=vol,
                             qqq_r12=r12, qqq_dd=dd))
    return rows


MS = {}


def main():
    t0 = time.time()
    build()
    for px in PROXIES:
        o, c = FR[px]
        ms = full_month_starts(o.index, load_prices(['SPY'])['open'].index)
        MS[px] = ms[ms > o.index[0]]  # drop first partial month
        print(px, 'sample', o.index[0].date(), '->', o.index[-1].date(), 'month starts', len(MS[px]))
    tasks = []
    for px in PROXIES:
        for other in OTHERS:
            for i in range(len(MS[px]) - 6):
                tasks.append((px, other, i))
    ctx = mp.get_context('fork')
    with ctx.Pool(4) as pool:
        out = pool.map(task, tasks, chunksize=8)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_pickle('/tmp/claude-0/-home-user-backtester/33fa7352-a928-54df-8b07-5f65696c4528/scratchpad/exp5a_raw.pkl')
    print('runs done', len(df), round(time.time() - t0), 's')
    analyse(df)


def subset_masks(d):
    return {
        'all': np.ones(len(d), bool),
        'hot tech (QQQ 12m>20% & within 5% of high)': ((d.qqq_r12 > 0.20) & (d.qqq_dd > -0.05)).to_numpy(),
        'QQQ >15% below high': (d.qqq_dd < -0.15).to_numpy(),
    }


def analyse(df):
    df = df.copy()
    df.to_csv(f'{OUT}/a_transition_windows.csv.gz', index=False, float_format='%.5f', compression='gzip')
    summ = []
    lump_vs_g6 = []
    for (proxy, other, H), g in df.groupby(['proxy', 'other', 'H']):
        # one row per (start, state, path)
        starts_all = g[['start', 'qqq_r12', 'qqq_dd']].drop_duplicates('start').reset_index(drop=True)
        masks = subset_masks(starts_all)
        for sname, mask in masks.items():
            sel_starts = set(starts_all.start[mask])
            gg = g[g.start.isin(sel_starts)]
            n = len(sel_starts)
            if n == 0:
                continue
            hold = gg[gg.path == 'hold']
            summ.append(dict(proxy=proxy, other=other, H=H, subset=sname, state='P0', path='hold', n=n,
                             rel_med=0, rel_p10=0, rel_p90=0, rel_min=0,
                             mdd_med=hold.mdd.median(), mdd_p10=q(hold.mdd, 10), mdd_worst=hold.mdd.min(),
                             vol_med=hold.vol.median(), end_med=hold.end_value.median(), p_beat_hold=np.nan))
            for (sname2, pname), x in gg[gg.path != 'hold'].groupby(['state', 'path']):
                summ.append(dict(proxy=proxy, other=other, H=H, subset=sname, state=sname2, path=pname, n=n,
                                 rel_med=x.rel_hold.median(), rel_p10=q(x.rel_hold, 10), rel_p90=q(x.rel_hold, 90),
                                 rel_min=x.rel_hold.min(), mdd_med=x.mdd.median(), mdd_p10=q(x.mdd, 10),
                                 mdd_worst=x.mdd.min(), vol_med=x.vol.median(), end_med=x.end_value.median(),
                                 p_beat_hold=(x.rel_hold > 0).mean()))
            for sname2 in STATES:
                x = gg[gg.state == sname2].pivot(index='start', columns='path', values='end_value')
                lump_vs_g6.append(dict(proxy=proxy, other=other, H=H, subset=sname, state=sname2, n=n,
                                       p_lump_gt_g6=(x.lump > x.g6).mean(), p_lump_gt_g3=(x.lump > x.g3).mean(),
                                       med_lump_minus_g6=(x.lump / x.g6 - 1).median(),
                                       p10_lump_minus_g6=q(x.lump / x.g6 - 1, 10), p90_lump_minus_g6=q(x.lump / x.g6 - 1, 90)))
    S = pd.DataFrame(summ)
    L = pd.DataFrame(lump_vs_g6)
    S.round(4).to_csv(f'{OUT}/a_transition_summary.csv', index=False)
    L.round(4).to_csv(f'{OUT}/a_lump_vs_gradual.csv', index=False)
    figures(df, S)


def figures(df, S):
    # Fig 1: box plots of end value relative to hold by path, 6-month, T1 and T2, both proxies (other=QQQ)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
    for r, px in enumerate(PROXIES):
        for cidx, st in enumerate(['T1 80/20', 'T2 sell others->VOO']):
            ax = axes[r, cidx]
            g = df[(df.proxy == px) & (df.other == 'QQQ') & (df.H == 6) & (df.state == st)]
            data = [g[g.path == p].rel_hold.to_numpy() * 100 for p in ['lump', 'g3', 'g6']]
            ax.boxplot(data, tick_labels=['lump', '3 steps', '6 steps'], showfliers=True, whis=(10, 90),
                       flierprops=dict(markersize=2, alpha=.4))
            ax.axhline(0, color='k', lw=.8)
            ax.set_title(f'{px} proxy, {st}, 6m (n={g.start.nunique()})', fontsize=10)
            ax.set_ylabel('end value vs hold-P0 (%)')
            ax.grid(alpha=.3)
    fig.suptitle('Transition outcome vs doing nothing (whiskers = p10/p90)')
    fig.tight_layout()
    savefig(fig, 'a1_transition_outcomes.png')
    plt.close(fig)

    # Fig 2: median max drawdown by path in the subsets (T1), 6m
    subs = ['all', 'hot tech (QQQ 12m>20% & within 5% of high)', 'QQQ >15% below high']
    labels = ['all starts', 'hot tech', 'QQQ >15% off high']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, px in zip(axes, PROXIES):
        w = 0.2
        for k, (pth, lab) in enumerate([('hold', 'hold P0'), ('lump', 'T1 lump'), ('g3', 'T1 3 steps'), ('g6', 'T1 6 steps')]):
            vals, ns = [], []
            for sb in subs:
                x = S[(S.proxy == px) & (S.other == 'QQQ') & (S.H == 6) & (S.subset == sb) &
                      (S.path == pth) & (S.state.isin(['P0', 'T1 80/20']))]
                vals.append(x.mdd_med.iloc[0] * 100 if len(x) else np.nan)
                ns.append(x.n.iloc[0] if len(x) else 0)
            ax.bar(np.arange(3) + (k - 1.5) * w, vals, w, label=lab)
        ax.set_xticks(range(3))
        ns = [int(S[(S.proxy == px) & (S.other == 'QQQ') & (S.H == 6) & (S.subset == sb) & (S.path == 'hold')].n.iloc[0]) for sb in subs]
        ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labels, ns)])
        ax.set_title(f'{px} proxy: median max drawdown within 6m')
        ax.set_ylabel('max drawdown (%)')
        ax.grid(alpha=.3, axis='y')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'a2_transition_drawdown_subsets.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
