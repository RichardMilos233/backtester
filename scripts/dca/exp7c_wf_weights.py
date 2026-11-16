"""EXP7 part C: walk-forward weights.  Is an "optimised" contribution split better than a fixed naive one?

Each January the split of the next 12 monthly (buy-only) contributions is chosen from the trailing 10 years
(information strictly before the decision day) by: winner-take-all (best trailing DCA wealth), weights proportional to
the rank of trailing DCA wealth, long-only max-Sharpe, long-only min-variance, inverse volatility.  Naive comparators:
100% first asset, equal weight, 70/30 first/second asset.
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_common import *  # noqa
from exp7_common import irr_rows  # noqa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_BOOT = int(os.environ.get('EXP7_NBOOT', 2000))
MEAN_BLOCK = 18
SEED = 20261118
TRAIN = 120
JMIN = 114            # first January with >= 113 monthly returns of history (~9.5y; 10y when available)
METHODS = ['winner', 'rank', 'maxsharpe', 'minvar', 'invvol']
NAIVE = ['mkt100', 'equal', 'mkt70_30']
ALLM = METHODS + NAIVE
LABEL = {'winner': 'winner-take-all', 'rank': 'rank-proportional', 'maxsharpe': 'max-Sharpe', 'minvar': 'min-variance',
         'invvol': 'inverse-vol', 'mkt100': '100% first asset', 'equal': 'equal weight', 'mkt70_30': '70/30 first/second'}
UNIVERSES = {
    'U1 VFINX+BRK-A': ['VFINX', 'BRK-A'],
    'U2 VFINX+BRK-A+PRITX': ['VFINX', 'BRK-A', 'PRITX'],
    'U3a FF_MKT+FF_TECH+FF_VALUE': ['FF_MKT', 'FF_TECH', 'FF_VALUE'],
    'U3b FF_MKT+FF_TECH+FF_VALUE+FF_DEVXUS': ['FF_MKT', 'FF_TECH', 'FF_VALUE', 'FF_DEVXUS'],
}
_G = {}


def build_c_world(assets):
    p = load_prices(assets)
    o, c = p['open'].copy(), p['close'].copy()
    rf = load_prices(['FF_RF'])['close']['FF_RF']
    rf = rf.reindex(rf.index.union(c.index)).ffill().reindex(c.index)
    o['RF'] = rf.values
    c['RF'] = rf.values
    return World.from_frames(o, c)


def simplex_grid(N, step):
    k = int(round(1 / step))
    pts = []

    def rec(prefix, left, d):
        if d == N - 1:
            pts.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, d + 1)
    rec([], k, 0)
    return np.array(pts, dtype=float) / k


GRID = {2: simplex_grid(2, 0.01), 3: simplex_grid(3, 0.02), 4: simplex_grid(4, 0.05)}


def month_returns(W):
    pe = W.c[np.maximum(W.pos - 1, 0)]                  # close before each decision day
    r = np.full_like(pe, np.nan)
    r[1:-1] = pe[2:] / pe[1:-1] - 1                     # r[m]: return of month m (pe[m] -> pe[m+1]); row 0 has no prior close
    return pe, r


def weights_at(W, pe, r, N, j):
    """Weights of every method for the January decision row j, using rows < j and closes <= pos[j]-1 only."""
    L = min(TRAIN, j - 1)
    lo = j - L
    R = r[lo:j, :N]
    ex = R - r[lo:j, N][:, None]
    assert np.isfinite(R).all() and lo >= 1
    wealth = W.inv[lo:j, :N].sum(axis=0) * pe[j, :N] / (j - lo)      # DCA wealth multiple per $ over the training window
    out = {}
    w = np.zeros(N); w[int(np.argmax(wealth))] = 1.0
    out['winner'] = w
    rk = np.argsort(np.argsort(wealth)) + 1.0
    out['rank'] = rk / rk.sum()
    G = GRID[N]
    mu = ex.mean(axis=0)
    cov = np.cov(ex, rowvar=False)
    var = np.einsum('ij,jk,ik->i', G, cov, G)
    sd = np.sqrt(np.maximum(var, 1e-18))
    out['maxsharpe'] = G[int(np.argmax(G @ mu / sd))]
    out['minvar'] = G[int(np.argmin(var))]
    iv = 1.0 / R.std(axis=0, ddof=1)
    out['invvol'] = iv / iv.sum()
    e = np.zeros(N); e[0] = 1.0
    out['mkt100'] = e
    out['equal'] = np.full(N, 1.0 / N)
    m = np.zeros(N); m[0], m[1] = 0.7, 0.3
    out['mkt70_30'] = m
    return out


def jan_rows_c(W):
    d = W.dates[W.pos]
    J = np.flatnonzero(np.asarray(d.month) == 1)
    return J[J >= JMIN]


def run_universe(W, N, J, real=True):
    """-> dict(weights {method: (nJ, N)}, rows, wt {method: (nrows, N)}, stats {method: (wm, irr)})"""
    pe, r = month_returns(W)
    Wm = {m: np.zeros((len(J), N)) for m in ALLM}
    for i, j in enumerate(J):
        ws = weights_at(W, pe, r, N, j)
        for m in ALLM:
            Wm[m][i] = ws[m]
    n = W.n
    rows = np.arange(J[0], n)
    idx = np.searchsorted(J, rows, side='right') - 1
    wt = {m: Wm[m][idx] for m in ALLM}
    fin, irr, wm = {}, {}, {}
    for m in ALLM:
        units = (wt[m] * W.inv[rows, :N]).sum(axis=0)
        fin[m] = float((units * W.c[-1, :N]).sum())
        wm[m] = fin[m] / len(rows)
    ks = list(ALLM)
    pos = W.pos[rows]
    if real and W.dates is not None:
        t = np.asarray((W.dates[pos] - W.dates[pos[0]]).days, dtype=float) / 365.0
        hor = (W.dates[-1] - W.dates[pos[0]]).days / 365.0
    else:
        t = (pos - pos[0]) / 252.0
        hor = (W.T - 1 - pos[0]) / 252.0
    irrs = irr_rows(np.tile(t, (len(ks), 1)), np.ones((len(ks), len(rows))), np.array([fin[m] for m in ks]),
                      np.full(len(ks), hor))
    irr = dict(zip(ks, irrs))
    return dict(weights=Wm, rows=rows, wt=wt, wm=wm, irr=irr, J=J, pe=pe)


def _boot_job(b):
    rng = np.random.default_rng([SEED, b])
    out = {}
    for u, (W, N, J) in _G['u'].items():
        Wb = _G['src'][u].draw(rng, MEAN_BLOCK)[0]
        res = run_universe(Wb, N, J, real=False)
        out[u] = (np.array([res['wm'][m] for m in ALLM]), np.array([res['irr'][m] for m in ALLM]),
                  np.array([np.abs(np.diff(res['weights'][m], axis=0)).sum(axis=1).mean() * 0.5 for m in ALLM]))
    return b, out


def selftest(W, N, J, res):
    """(1) no lookahead in weights, (2) fast track == DCABacktest, (3) windows == rolling_dca."""
    from src.dca import DCABacktest, dca_metrics, rolling_dca
    jstar = J[len(J) // 2]
    cut = W.pos[jstar]
    o, c = W.o.copy(), W.c.copy()
    rng = np.random.default_rng(11)
    c[cut:] *= rng.uniform(0.3, 3, size=(W.T - cut, W.N))
    o[cut:] = c[cut:]
    W2 = World(o, c, W.pos, dates=W.dates)
    pe2, r2 = month_returns(W2)
    pe, r = month_returns(W)
    k = 0
    for j in J[J <= jstar]:
        a, b = weights_at(W, pe, r, N, j), weights_at(W2, pe2, r2, N, j)
        for m in ALLM:
            assert np.array_equal(a[m], b[m]), f'lookahead in {m} at {W.dates[W.pos[j]].date()}'
        k += 1
    # DCABacktest on one method
    m = 'maxsharpe'
    rows = res['rows']
    D = W.dates[W.pos[rows]]
    names = W.names[:N]
    wmap = {d: dict(zip(names, w)) for d, w in zip(D, res['wt'][m])}
    pol = lambda ctx: {'buy': wmap[ctx.date]}
    op = pd.DataFrame(W.o[:, :N], index=W.dates, columns=names).loc[D[0]:]
    cl = pd.DataFrame(W.c[:, :N], index=W.dates, columns=names).loc[D[0]:]
    led = DCABacktest(slippage_rate=SLIP).run(op, cl, pol, pd.Series(1.0, index=D))
    mt = dca_metrics(led)
    assert abs(mt.wealth_multiple - res['wm'][m]) < 1e-9 * mt.wealth_multiple, (mt.wealth_multiple, res['wm'][m])
    dep = led['cum_contribution'].diff().fillna(led['cum_contribution'].iloc[0])
    dep = dep[dep > 0]
    tt = ((dep.index - led.index[0]).days.to_numpy(dtype=float) / 365.0)[None, :]
    irr_led = irr_rows(tt, dep.to_numpy()[None, :], np.array([led['total_value'].iloc[-1]]), np.array([(led.index[-1] - led.index[0]).days / 365.0]))[0]
    assert abs(irr_led - res['irr'][m]) < 1e-6, (irr_led, res['irr'][m])
    alloc = pd.DataFrame(res['wt'][m][:-1], index=D[:-1], columns=names) if False else pd.DataFrame(res['wt'][m], index=D, columns=names)
    rd = rolling_dca(op, cl, alloc, 120, 1.0, SLIP)
    fast = windows_wealth(W, N, rows, res['wt'][m], 120)
    assert np.allclose(rd.wealth_multiple.values, fast, rtol=1e-9)
    return (f'selftest C passed on {len(J)} Januaries ({k} checked for lookahead by scrambling prices after '
            f'{W.dates[cut].date()}); track == DCABacktest (wm {mt.wealth_multiple:.4f}, irr {irr_led:.4%}; note src.dca.dca_metrics returns NaN irr here: its +-8 bracket overflows exp() on a 90-year horizon); windows == rolling_dca')


def windows_wealth(W, N, rows, wt, H):
    cum = np.vstack([np.zeros(N), np.cumsum(wt * W.inv[rows, :N], axis=0)])
    nrow = len(rows)
    Wn = nrow - H
    s = np.arange(Wn)
    e = s + H
    Pend = W.c[W.pos[rows[e]] - 1, :N]
    return (Pend * (cum[e] - cum[s])).sum(axis=1) / H


def dcabt_metrics(W, N, res, m):
    from src.dca import DCABacktest, dca_metrics
    rows = res['rows']
    D = W.dates[W.pos[rows]]
    names = W.names[:N]
    wmap = {d: dict(zip(names, w)) for d, w in zip(D, res['wt'][m])}
    op = pd.DataFrame(W.o[:, :N], index=W.dates, columns=names).loc[D[0]:]
    cl = pd.DataFrame(W.c[:, :N], index=W.dates, columns=names).loc[D[0]:]
    led = DCABacktest(slippage_rate=SLIP).run(op, cl, lambda ctx: {'buy': wmap[ctx.date]}, pd.Series(1.0, index=D))
    return dca_metrics(led), led


def run():
    t0 = time.time()
    worlds = {u: build_c_world(a) for u, a in UNIVERSES.items()}
    U = {}
    for u, W in worlds.items():
        N = len(UNIVERSES[u])
        J = jan_rows_c(W)
        U[u] = (W, N, J)
        print(f'{u}: data {W.dates[0].date()}..{W.dates[-1].date()}, {len(J)} Januaries from {W.dates[W.pos[J[0]]].date()}', flush=True)
    _G.update(u=U, src={u: BlockSource(w) for u, (w, _, _) in U.items()})
    res = {u: run_universe(W, N, J) for u, (W, N, J) in U.items()}
    msg = selftest(*U['U3a FF_MKT+FF_TECH+FF_VALUE'], res['U3a FF_MKT+FF_TECH+FF_VALUE'])
    print(msg, flush=True)

    trows, wrows, rrows, wyrows = [], [], [], []
    for u, (W, N, J) in U.items():
        R = res[u]
        names = W.names[:N]
        d0 = W.dates[W.pos[R['rows'][0]]]
        for m in ALLM:
            mt, led = dcabt_metrics(W, N, R, m)
            assert abs(mt.wealth_multiple - R['wm'][m]) < 1e-8 * mt.wealth_multiple
            Wm = R['weights'][m]
            turn = np.abs(np.diff(Wm, axis=0)).sum(axis=1).mean() * 0.5
            trows.append(dict(universe=u, method=LABEL[m], oos_start=str(d0.date()), oos_end=str(W.dates[-1].date()), months=len(R['rows']),
                              wealth_multiple=mt.wealth_multiple, irr=R['irr'][m], max_drawdown=mt.max_drawdown, ann_vol=mt.annualized_volatility,
                              worst_12m=mt.worst_12m_return, mean_annual_turnover=turn, mean_max_weight=Wm.max(axis=1).mean(),
                              n_januaries=len(J)))
            for i, j in enumerate(J):
                wyrows.append(dict(universe=u, method=LABEL[m], january=str(W.dates[W.pos[j]].date()), **{a: Wm[i, k] for k, a in enumerate(names)}))
        # rolling OOS windows vs naive mixes
        for h, H in (('10y', 120), ('20y', 240)):
            if len(R['rows']) - H < 12:
                continue
            ww = {m: windows_wealth(W, N, R['rows'], R['wt'][m], H) for m in ALLM}
            for m in METHODS:
                for nv in NAIVE:
                    q = ww[m] / ww[nv]
                    no = stationary_median_nonoverlap(q, H)
                    rrows.append(dict(universe=u, method=LABEL[m], vs=LABEL[nv], horizon=h, n_windows=len(q), median_ratio=np.median(q),
                                      share_gt1=np.mean(q > 1), p10=np.percentile(q, 10), p90=np.percentile(q, 90),
                                      nonov_median=no[0], nonov_n=len(q[::H]), nonov_share_gt1=np.mean(q[::H] > 1)))
    T = pd.DataFrame(trows)
    T.round(4).to_csv(os.path.join(OUT, 'c_oos_tracks.csv'), index=False)
    pd.DataFrame(wyrows).round(3).to_csv(os.path.join(OUT, 'c_weights_by_january.csv'), index=False)
    RW = pd.DataFrame(rrows)
    RW.round(4).to_csv(os.path.join(OUT, 'c_rolling_oos_windows.csv'), index=False)
    print(T.round(4).to_string(), flush=True)

    print(f'bootstrap x{N_BOOT} ...', flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(4) as pool:
        bres = dict(pool.map(_boot_job, range(N_BOOT), chunksize=20))
    brow = []
    for u in U:
        wm = np.stack([bres[b][u][0] for b in range(N_BOOT)])      # (B, 8)
        irr = np.stack([bres[b][u][1] for b in range(N_BOOT)])
        tn = np.stack([bres[b][u][2] for b in range(N_BOOT)])
        R = res[u]
        for m in METHODS:
            for nv in NAIVE:
                im, inn = ALLM.index(m), ALLM.index(nv)
                lr = np.log(wm[:, im] / wm[:, inn])
                di = (irr[:, im] - irr[:, inn]) * 100
                real_r = R['wm'][m] / R['wm'][nv]
                real_d = (R['irr'][m] - R['irr'][nv]) * 100
                brow.append(dict(universe=u, method=LABEL[m], vs=LABEL[nv], real_wealth_ratio=real_r, boot_mean_ratio=np.exp(lr.mean()),
                                 boot_p5=np.exp(np.percentile(lr, 5)), boot_p95=np.exp(np.percentile(lr, 95)), P_ratio_gt1=np.mean(lr > 0),
                                 real_pctile=np.mean(lr < np.log(real_r)) * 100, real_irr_diff_pp=real_d, boot_irr_diff_mean_pp=di.mean(),
                                 boot_irr_p5_pp=np.percentile(di, 5), boot_irr_p95_pp=np.percentile(di, 95), n_paths=N_BOOT))
    BB = pd.DataFrame(brow)
    BB.round(4).to_csv(os.path.join(OUT, 'c_bootstrap_vs_naive.csv'), index=False)
    print(BB.round(3).to_string(), flush=True)

    figs(U, res, BB, T)
    print(f'part C done {time.time() - t0:.0f}s', flush=True)
    return dict(T=T, RW=RW, BB=BB, msg=msg, n_boot=N_BOOT, res=res, U=U)


def figs(U, res, BB, T):
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # weights over time
    ulist = [u for u in U if not u.startswith('U3b')]
    fig, axes = plt.subplots(len(ulist), len(METHODS), figsize=(17, 8.5), sharey=True)
    for r_, u in enumerate(ulist):
        W, N, J = U[u]
        yrs = np.asarray(W.dates[W.pos[J]].year)
        for c_, m in enumerate(METHODS):
            ax = axes[r_, c_]
            ax.stackplot(yrs, res[u]['weights'][m].T, labels=W.names[:N], colors=cols[:N], alpha=0.85, step='post')
            ax.set_xlim(yrs[0], yrs[-1] + 1)
            if r_ == 0:
                ax.set_title(LABEL[m], fontsize=10)
            if c_ == 0:
                ax.set_ylabel(u.split(' ')[0] + ' weight')
            ax.set_ylim(0, 1)
        axes[r_, -1].legend(W.names[:N], fontsize=6, loc='upper right')
    fig.suptitle('Walk-forward weights chosen each January from the trailing 10 years (stacked share of the next 12 contributions)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_wf_weights_over_time.png'), dpi=130)
    plt.close(fig)

    # equity curves relative to equal weight
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (u, (W, N, J)) in zip(axes.ravel(), U.items()):
        R = res[u]
        rows = R['rows']
        n = len(rows)
        pe_ = np.r_[W.pos[rows[1:]], W.T] - 1
        d = W.dates[W.pos[rows]]

        def curve(wt):
            cum = np.cumsum(wt * W.inv[rows, :N], axis=0)
            return (cum * W.c[pe_, :N]).sum(axis=1) / np.arange(1, n + 1)
        base = curve(R['wt']['equal'])
        for m in ALLM:
            ax.plot(d, curve(R['wt'][m]) / base, lw=1.6 if m in METHODS else 1.0, ls='-' if m in METHODS else '--', label=LABEL[m])
        ax.axhline(1, color='k', lw=0.5)
        ax.set_title(u, fontsize=9)
        ax.set_yscale('log')
        ax.set_ylabel('value / equal-weight value')
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle('Out-of-sample chained DCA tracks relative to the equal-weight DCA', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_wf_weights_equity.png'), dpi=130)
    plt.close(fig)

    # bootstrap distribution summary vs equal weight and vs 100% market
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, nv in zip(axes, ['equal weight', '100% first asset']):
        sub = BB[BB.vs == nv]
        us = list(dict.fromkeys(sub.universe))
        x = np.arange(len(METHODS))
        for k, u in enumerate(us):
            g = sub[sub.universe == u].set_index('method').loc[[LABEL[m] for m in METHODS]]
            xx = x + (k - 1.5) * 0.18
            ax.vlines(xx, (g.boot_p5 - 1) * 100, (g.boot_p95 - 1) * 100, color=cols[k], lw=3, alpha=0.5)
            ax.scatter(xx, (g.real_wealth_ratio - 1) * 100, color=cols[k], zorder=5, s=35, label=u.split(' ')[0] + ' actual')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=8)
        ax.set_title(f'wealth vs {nv} DCA; bars = bootstrap 5-95% (resampled histories)')
        ax.set_ylabel('terminal wealth ratio - 1, %')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig8_wf_weights_bootstrap.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    run()
