"""EXP3: should each month's contribution chase tech (A) or buy the laggard / Berkshire (B)?

Part A: descriptive facts (drawdown episodes, capture ratios, predictability of relative returns).
Part B: contribution-direction rules evaluated with rolling_dca, matched-allocation controls,
        block-shuffle placebo, and a DCABacktest buy_underweight vs fixed 50/50 comparison.

Run from the repo root:  python scripts/dca/exp3_tilt_timing.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dca import (DCABacktest, buy_underweight, dca_metrics, fixed_buy, load_prices,
                     month_starts, rolling_dca)

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca', 'exp3_tilt_timing')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

SLIP = 0.0005
HORIZONS = {'5y': 60, '10y': 120, '20y': 240}
N_PLACEBO = 1000
BLOCK = 6
RNG = np.random.default_rng(20261115)
EPS = 1e-9

# pair name -> (A, B, start, proxyA, proxyB)  proxies are used only for signal warm-up before the pair's data begins
PAIRS = {
    'QQQ/BRK-B': ('QQQ', 'BRK-B', '1999-03-01', '^NDX', 'BRK-A'),
    'SPY/BRK-B': ('SPY', 'BRK-B', '1999-03-01', 'VFINX', 'BRK-A'),
    'QQQ/SPY': ('QQQ', 'SPY', '1999-03-01', '^NDX', 'VFINX'),
    '^NDX/BRK-A': ('^NDX', 'BRK-A', None, None, None),
    'VFINX/BRK-A': ('VFINX', 'BRK-A', None, None, None),
}


def save(df, name, **kw):
    df.to_csv(os.path.join(OUT, name), **kw)


def pct(x, nd=1):
    return np.round(100 * np.asarray(x, dtype=float), nd)


# =========================================================================== PART A
def episodes(close: pd.Series, thr: float) -> pd.DataFrame:
    """Drawdown episodes (running-max definition): peak -> trough while below the previous high, until a new high.
    Episodes whose max depth is >= thr are kept."""
    cm = close.cummax()
    dd = close / cm - 1
    at_high = dd.values == 0
    rows = []
    i, n = 0, len(close)
    idx = close.index
    while i < n:
        if at_high[i]:
            i += 1
            continue
        j = i
        while j < n and not at_high[j]:
            j += 1
        peak = idx[i - 1] if i > 0 else idx[0]
        seg = dd.iloc[i:j]
        tdate = seg.idxmin()
        depth = seg.min()
        if depth <= -thr:
            rows.append(dict(peak=peak, trough=tdate, depth=depth,
                             recovered=idx[j] if j < n else pd.NaT))
        i = j
    return pd.DataFrame(rows)


def asof_px(s: pd.Series, d):
    if d < s.index[0] or d > s.index[-1]:
        return np.nan
    return s.asof(d)


def ep_returns(eps: pd.DataFrame, assets: dict) -> pd.DataFrame:
    out = eps.copy()
    for nm, s in assets.items():
        pk = [asof_px(s, d) for d in eps['peak']]
        tr = [asof_px(s, d) for d in eps['trough']]
        out[f'{nm}_p2t'] = np.array(tr) / np.array(pk) - 1
        for m in (12, 36):
            fut = [asof_px(s, d + pd.DateOffset(months=m)) if d + pd.DateOffset(months=m) <= s.index[-1] else np.nan
                   for d in eps['trough']]
            out[f'{nm}_post{m}m'] = np.array(fut) / np.array(tr) - 1
        # BRK must exist at the peak date for p2t to be meaningful
        bad = np.array([d < s.index[0] for d in eps['peak']])
        out.loc[bad, [f'{nm}_p2t', f'{nm}_post12m', f'{nm}_post36m']] = np.nan
    return out


def part_a1(px):
    sp, ndx, brk, qqq = px['VFINX'], px['^NDX'], px['BRK-A'], px['QQQ']
    assets = {'SP': sp, 'NDX': ndx, 'QQQ': qqq, 'BRK': brk}
    e_sp = ep_returns(episodes(sp, 0.10), assets)
    e_nd = ep_returns(episodes(ndx, 0.15), assets)
    summ = []
    for nm, e, ref in (('S&P drawdowns >=10% (VFINX)', e_sp, 'SP'), ('NDX drawdowns >=15% (^NDX)', e_nd, 'NDX')):
        ok = e['BRK_p2t'].notna()
        n = int(ok.sum())
        summ.append(dict(set=nm, n_episodes=len(e), n_with_BRK=n,
                         BRK_fell_less_than_SP=int((e.loc[ok, 'BRK_p2t'] > e.loc[ok, 'SP_p2t']).sum()),
                         n_with_NDX=int((ok & e['NDX_p2t'].notna()).sum()),
                         BRK_fell_less_than_NDX=int(((e['BRK_p2t'] > e['NDX_p2t']) & ok & e['NDX_p2t'].notna()).sum()),
                         BRK_rose_in_episode=int((e.loc[ok, 'BRK_p2t'] > 0).sum()),
                         median_SP=np.nanmedian(e['SP_p2t']), median_NDX=np.nanmedian(e['NDX_p2t']),
                         median_BRK=np.nanmedian(e['BRK_p2t'])))
    summ = pd.DataFrame(summ)
    for nm, e in (('a1_sp_drawdowns.csv', e_sp), ('a1_ndx_drawdowns.csv', e_nd)):
        o = e.copy()
        for c in o.columns:
            if c.endswith(('_p2t', '_post12m', '_post36m')) or c == 'depth':
                o[c] = o[c].round(4)
        for c in ('peak', 'trough', 'recovered'):
            o[c] = o[c].dt.strftime('%Y-%m-%d')
        save(o, nm, index=False)
    save(summ.round(4), 'a1_summary.csv', index=False)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    for ax, e, ttl in ((axes[0], e_sp, 'S&P 500 (VFINX) drawdowns >= 10%'),
                       (axes[1], e_nd, 'Nasdaq-100 (^NDX, price) drawdowns >= 15%')):
        lab = [f"{p:%Y-%m}->{t:%Y-%m} ({d*100:.0f}%)" for p, t, d in zip(e['peak'], e['trough'], e['depth'])]
        y = np.arange(len(e))[::-1]
        h = 0.27
        for off, nm, col in ((h, 'SP', '#1f77b4'), (0, 'NDX', '#2ca02c'), (-h, 'BRK', '#d62728')):
            ax.barh(y + off, e[f'{nm}_p2t'] * 100, height=h, color=col,
                    label={'SP': 'S&P 500', 'NDX': 'Nasdaq-100', 'BRK': 'Berkshire A'}[nm])
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=7)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel('Return peak -> trough, %  (blank bar = asset has no data at that peak)')
        ax.set_title(ttl)
        ax.legend(loc='lower left', fontsize=8)
    fig.suptitle('Drawdown episodes: how S&P, Nasdaq-100 and Berkshire behaved from peak to trough', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_episodes.png'), dpi=130)
    plt.close(fig)
    return e_sp, e_nd, summ


def monthly_close(s):
    return s.resample('ME').last().dropna()


def capture(a: pd.Series, b: pd.Series):
    """b vs benchmark a: beta, corr, up/down capture (ratio of mean monthly returns)."""
    d = pd.concat([a, b], axis=1, keys=['a', 'b'], sort=True).dropna()
    if len(d) < 12:
        return None
    up, dn = d[d.a > 0], d[d.a < 0]
    beta = np.cov(d.a, d.b)[0, 1] / d.a.var()
    return dict(n_months=len(d), beta=beta, corr=d.a.corr(d.b),
                up_capture=up.b.mean() / up.a.mean(), down_capture=dn.b.mean() / dn.a.mean(),
                mean_a_pct=d.a.mean() * 100, mean_b_pct=d.b.mean() * 100,
                ann_vol_a=d.a.std() * np.sqrt(12) * 100, ann_vol_b=d.b.std() * np.sqrt(12) * 100)


def part_a2(px):
    m = {k: monthly_close(v).pct_change().dropna() for k, v in px.items()}
    eras = {'full': (None, None), '1980-99': ('1980', '1999'), '2000-12': ('2000', '2012'), '2013-26': ('2013', '2026')}
    rows = []
    for bench, bn in (('VFINX', 'S&P 500 (VFINX)'), ('^NDX', 'Nasdaq-100 (^NDX price)')):
        for era, (s, e) in eras.items():
            r = capture(m[bench].loc[s:e], m['BRK-A'].loc[s:e])
            if r:
                rows.append(dict(benchmark=bn, era=era, **r))
    # modern ETF sample as a cross-check
    mm = {k: monthly_close(px[k].loc['1999-03':]).pct_change().dropna() for k in ('SPY', 'QQQ', 'BRK-B')}
    for bench in ('SPY', 'QQQ'):
        r = capture(mm[bench], mm['BRK-B'])
        rows.append(dict(benchmark=f'{bench} (BRK-B, 1999-03+)', era='full', **r))
    t = pd.DataFrame(rows).round(3)
    save(t, 'a2_capture.csv', index=False)
    return t


def circ_p(x, y, stat, nmin):
    """p-value for stat(x, y) by circularly shifting y (keeps autocorrelation of both series)."""
    obs = stat(x, y)
    m = len(y)
    vals = []
    for s in range(nmin, m - nmin + 1):
        vals.append(stat(x, np.roll(y, s)))
    vals = np.array(vals)
    return obs, (1 + np.sum(np.abs(vals) >= abs(obs) - 1e-15)) / (1 + len(vals)), vals


def corr_stat(x, y):
    return np.corrcoef(x, y)[0, 1]


def quint_means(x, y):
    q = pd.qcut(pd.Series(x).rank(method='first'), 5, labels=False).values
    return np.array([y[q == i].mean() for i in range(5)])


def part_a3(pairs_data):
    KS = [1, 3, 6, 12, 24, 36]
    HS = [1, 3, 6, 12, 36]
    rows, qrows = [], []
    qplot = {}
    for pname, d in pairs_data.items():
        c = d['closes']
        mcl = c.resample('ME').last().dropna()
        r = np.log(mcl['A'] / mcl['B']).values
        n = len(r)
        for k in KS:
            for h in HS:
                t0, t1 = k, n - 1 - h
                if t1 - t0 < 40:
                    continue
                ts = np.arange(t0, t1 + 1)
                x = r[ts] - r[ts - k]
                y = r[ts + h] - r[ts]
                nmin = max(k, h, 12)
                obs, p, _ = circ_p(x, y, corr_stat, nmin)
                # non-overlapping sample: every max(k,h) months
                step = max(k, h)
                xs, ys = x[::step], y[::step]
                n_no = len(xs)
                c_no = np.corrcoef(xs, ys)[0, 1] if n_no > 4 else np.nan
                rows.append(dict(pair=pname, k=k, h=h, n_obs=len(x), corr=obs, p_circ_shift=p,
                                 n_nonoverlap=n_no, corr_nonoverlap=c_no))
                # quintiles (labels fixed from x; y circularly shifted)
                q = pd.qcut(pd.Series(x).rank(method='first'), 5, labels=False).values
                M = np.array([(q == i) / (q == i).sum() for i in range(5)])      # 5 x m
                qm = M @ y
                spread = qm[4] - qm[0]
                shifts = range(nmin, len(y) - nmin + 1)
                null = np.array([M @ np.roll(y, s_) for s_ in shifts])
                ns = null[:, 4] - null[:, 0]
                ps = (1 + np.sum(np.abs(ns) >= abs(spread) - 1e-15)) / (1 + len(ns))
                lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
                qrows.append(dict(pair=pname, k=k, h=h, n_obs=len(x),
                                  **{f'Q{i+1}_mean_pp': qm[i] * 100 for i in range(5)},
                                  spread_Q5_minus_Q1_pp=spread * 100, p_spread_circ_shift=ps))
                qplot[(pname, k, h)] = (qm * 100, lo * 100, hi * 100)
    cor = pd.DataFrame(rows)
    qt = pd.DataFrame(qrows)
    save(cor.round(4), 'a3_corr_long.csv', index=False)
    save(qt.round(3), 'a3_quintiles.csv', index=False)
    # matrices (corr) per pair
    mats = []
    for pname in pairs_data:
        s = cor[cor.pair == pname]
        mm = s.pivot(index='k', columns='h', values='corr').round(3)
        mm.insert(0, 'pair', pname)
        mats.append(mm.reset_index())
    save(pd.concat(mats), 'a3_corr_matrix.csv', index=False)

    # figure: quintile bars
    pn = list(pairs_data)
    fig, axes = plt.subplots(2, len(pn), figsize=(3.2 * len(pn), 6.2), sharey='row')
    for j, p in enumerate(pn):
        for i, (k, h) in enumerate(((12, 12), (36, 36))):
            ax = axes[i, j]
            qm, lo, hi = qplot[(p, k, h)]
            ax.bar(range(1, 6), qm, color='#4c78a8')
            ax.fill_between([0.5, 5.5], np.min(lo), np.max(hi), color='gray', alpha=0.25,
                            label='95% band, shifted-y null')
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(range(1, 6))
            pv = qt[(qt.pair == p) & (qt.k == k) & (qt.h == h)].p_spread_circ_shift.iloc[0]
            ax.set_title(f'{p}\ntrailing {k}m -> fwd {h}m  (p={pv:.2f})', fontsize=8)
            if j == 0:
                ax.set_ylabel('Mean fwd A-minus-B, pp (log)')
            ax.set_xlabel('Quintile of trailing relative return (1=A lagged most)', fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Forward relative return of A (tech/market) vs B (Berkshire) by quintile of trailing relative return', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_quintiles.png'), dpi=130)
    plt.close(fig)
    return cor, qt


# =========================================================================== PART B
def splice(proxy: pd.Series, actual: pd.Series) -> pd.Series:
    t0 = actual.index[0]
    scale = actual.iloc[0] / proxy.asof(t0)
    pre = proxy[proxy.index < t0] * scale
    return pd.concat([pre, actual])


def prev_idx(index, dates):
    """position of the last bar strictly before each date"""
    return index.searchsorted(pd.DatetimeIndex(dates), side='left') - 1


def asof_pos(index, dates):
    """position of the last bar on or before each date"""
    return index.searchsorted(pd.DatetimeIndex(dates), side='right') - 1


def make_signals(sigA: pd.Series, sigB: pd.Series, D: pd.DatetimeIndex):
    """All signals for decision dates D, from closes strictly before each date. Returns dict of arrays (NaN = unavailable)."""
    com = sigA.index.intersection(sigB.index)
    a, b = sigA.loc[com], sigB.loc[com]
    la, lb = np.log(a.values), np.log(b.values)
    ip = prev_idx(com, D)
    out = {}
    for k in (1, 3, 6, 12):
        look = D - pd.DateOffset(months=k)
        il = asof_pos(com, look)
        ok = (il >= 0) & (ip >= 0)
        ra = np.where(ok, la[ip] - la[np.maximum(il, 0)], np.nan)
        rb = np.where(ok, lb[ip] - lb[np.maximum(il, 0)], np.nan)
        out[f'rel{k}'] = ra - rb
    sma = a.rolling(200, min_periods=200).mean().values
    out['trend'] = np.where(ip >= 0, a.values[ip] / sma[ip] - 1, np.nan)
    dd = (a / a.cummax() - 1).values
    out['dd'] = np.where(ip >= 0, dd[ip], np.nan)
    ratio = pd.Series(la - lb, index=com)
    mu = ratio.rolling(756, min_periods=700).mean()
    sd = ratio.rolling(756, min_periods=700).std()
    z = ((ratio - mu) / sd).values
    out['z'] = np.where(ip >= 0, z[ip], np.nan)
    out['ratio'] = np.where(ip >= 0, ratio.values[ip], np.nan)
    return out


def sign_rules(sg):
    """name -> (family, s array in {+1 favor A, -1 favor B, 0 neutral})"""
    R = {}
    for k in (1, 3, 6, 12):
        s = np.where(sg[f'rel{k}'] > 0, 1.0, -1.0)
        R[f'mom{k}'] = ('momentum', s)
        R[f'contra{k}'] = ('contrarian', -s)
    s = np.where(sg['trend'] > 0, 1.0, -1.0)
    R['trend_A_above_sma200'] = ('trend', s)
    R['trend_rev'] = ('trend', -s)
    for d in (0.10, 0.20, 0.30):
        isdd = sg['dd'] <= -d
        tag = int(d * 100)
        R[f'dd{tag}_dip'] = ('drawdown', np.where(isdd, 1.0, -1.0))
        R[f'dd{tag}_flee'] = ('drawdown', np.where(isdd, -1.0, 1.0))
        R[f'dd{tag}_dip_else50'] = ('drawdown', np.where(isdd, 1.0, 0.0))
        R[f'dd{tag}_flee_else50'] = ('drawdown', np.where(isdd, -1.0, 0.0))
    z = sg['z']
    R['z_mean_revert'] = ('ratio_z', np.where(z > 1, -1.0, np.where(z < -1, 1.0, 0.0)))
    R['z_momentum'] = ('ratio_z', np.where(z > 1, 1.0, np.where(z < -1, -1.0, 0.0)))
    return R


FIXED = {'fixed_100A': 1.0, 'fixed_0A_100B': 0.0, 'fixed_50_50': 0.5, 'fixed_70A_30B': 0.7, 'fixed_30A_70B': 0.3}


def make_alloc_table(D, fA):
    return pd.DataFrame({'A': fA, 'B': 1 - fA}, index=D)


class Ctx:
    """Per pair/horizon arrays for the fast linear evaluation used for placebo (verified against rolling_dca)."""

    def __init__(self, opens, closes, D, H):
        self.H, self.n = H, len(D)
        self.W = self.n - H
        oA = opens['A'].reindex(D).values
        oB = opens['B'].reindex(D).values
        self.invA = 1.0 / (oA * (1 + SLIP))
        self.invB = 1.0 / (oB * (1 + SLIP))
        ends = np.arange(self.W) + H
        pe = opens.index.get_indexer(D[ends])
        cf = closes.reindex(opens.index).ffill()
        self.PA = cf['A'].values[pe - 1]
        self.PB = cf['B'].values[pe - 1]
        self.s = np.arange(self.W)
        self.e = ends
        self.WA, _ = self._wealth(np.ones((1, self.n)))
        self.WB, _ = self._wealth(np.zeros((1, self.n)))
        self.WA, self.WB = self.WA[0], self.WB[0]

    def _wealth(self, F):
        cA = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(F * self.invA, axis=1)], axis=1)
        cB = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum((1 - F) * self.invB, axis=1)], axis=1)
        cF = np.concatenate([np.zeros((F.shape[0], 1)), np.cumsum(F, axis=1)], axis=1)
        w = (self.PA * (cA[:, self.e] - cA[:, self.s]) + self.PB * (cB[:, self.e] - cB[:, self.s])) / self.H
        fav = (cF[:, self.e] - cF[:, self.s]) / self.H
        return w, fav

    def wealth_and_matched(self, F):
        w, fav = self._wealth(F)
        matched = fav * self.WA + (1 - fav) * self.WB
        return w, fav, matched


def era_of(ts):
    y = pd.DatetimeIndex(ts).year
    return np.where(y < 2000, '<2000', np.where(y < 2010, '2000-09', '2010+'))


def eval_rule(name, family, version, fA, D, opens, closes, ctx, H):
    alloc = make_alloc_table(D, fA)
    res = rolling_dca(opens, closes, alloc, H, amount=1.0, slippage_rate=SLIP)
    w, fav, matched = ctx.wealth_and_matched(fA[None, :])
    w, fav, matched = w[0], fav[0], matched[0]
    assert np.allclose(w, res['wealth_multiple'].values, rtol=1e-9), name
    w50 = 0.5 * ctx.WA + 0.5 * ctx.WB
    best = np.maximum(ctx.WA, ctx.WB)
    irr = res['irr'].values
    rm = w / matched
    r5 = w / w50
    rec = dict(rule=name, family=family, version=version, n_windows=len(w), avg_fracA=fav.mean(),
               irr_med=np.nanmedian(irr), irr_p10=np.nanpercentile(irr, 10), irr_min=np.nanmin(irr),
               wm_med=np.median(w),
               vs50_beat=np.mean(r5 > 1 + EPS), vs50_med=np.median(r5), vs50_mean=np.mean(r5),
               vsmatch_beat=np.mean(rm > 1 + EPS), vsmatch_med=np.median(rm), vsmatch_mean=np.mean(rm),
               vsbest_med=np.median(w / best), vsbest_beat=np.mean(w > best + EPS),
               vsAllA_med=np.median(w / ctx.WA), vsAllB_med=np.median(w / ctx.WB))
    eras = era_of(res['start'])
    era_rows = []
    for e in ('<2000', '2000-09', '2010+'):
        mk = eras == e
        if mk.sum() >= 3:
            era_rows.append(dict(rule=name, era=e, n_windows=int(mk.sum()), avg_fracA=fav[mk].mean(),
                                 vsmatch_beat=np.mean(rm[mk] > 1 + EPS), vsmatch_med=np.median(rm[mk]),
                                 vs50_beat=np.mean(r5[mk] > 1 + EPS), vs50_med=np.median(r5[mk]),
                                 irr_med=np.nanmedian(irr[mk])))
    return rec, era_rows


def block_perm_index(n, L, R, rng):
    """R x n index matrix: blocks of length L of range(n) randomly permuted and concatenated (multiset preserved)."""
    blocks = [np.arange(i, min(i + L, n)) for i in range(0, n, L)]
    nb = len(blocks)
    out = np.empty((R, n), dtype=np.int64)
    for r in range(R):
        out[r] = np.concatenate([blocks[j] for j in rng.permutation(nb)])
    return out


def lambda_w(ctx, F):
    w, fav, matched = ctx.wealth_and_matched(F)
    return w / matched


def placebo_for_rule(fA, ctx, L, R, rng):
    idx = block_perm_index(len(fA), L, R, rng)
    F = fA[idx]
    w, fav, matched = ctx.wealth_and_matched(F)
    return np.median(w / matched, axis=1)


def run_part_b(pairs_data):
    summary, era_all, plc_rows, plcmax_rows, sens_rows = [], [], [], [], []
    store = {}
    for pname, d in pairs_data.items():
        D, opens, closes = d['D'], d['opens'], d['closes']
        sg = make_signals(d['sigA'], d['sigB'], D)
        assert not any(np.isnan(v).any() for v in sg.values()), f'NaN signals in {pname}'
        rules = sign_rules(sg)
        seqs = {}
        for nm, (fam, s) in rules.items():
            seqs[f'{nm}|hard'] = (nm, fam, 'hard', 0.5 + 0.5 * s)
            seqs[f'{nm}|soft'] = (nm, fam, 'soft', 0.5 + 0.25 * s)
        n_var = len(seqs)
        for hname, H in HORIZONS.items():
            if len(D) - H < 24:
                continue
            ctx = Ctx(opens, closes, D, H)
            recs = []
            for fx, v in FIXED.items():
                rec, er = eval_rule(fx, 'fixed', '-', np.full(len(D), v), D, opens, closes, ctx, H)
                rec.update(pair=pname, horizon=hname)
                recs.append(rec)
                for e in er:
                    e.update(pair=pname, horizon=hname)
                    era_all.append(e)
            real = {}
            for key, (nm, fam, ver, fA) in seqs.items():
                rec, er = eval_rule(f'{nm}|{ver}', fam, ver, fA, D, opens, closes, ctx, H)
                rec.update(pair=pname, horizon=hname)
                recs.append(rec)
                for e in er:
                    e.update(pair=pname, horizon=hname)
                    era_all.append(e)
                real[key] = rec['vsmatch_med']
            keys = list(seqs)
            real_arr = np.array([real[k] for k in keys])
            # placebo 1 (primary): the whole signal book is circularly shifted (and optionally time-reversed) by a common
            # offset, so every rule keeps its own average fraction, run lengths and its correlation with the other rules,
            # but loses its alignment with the price data.
            nn = len(D)
            smin = 6
            sh = np.arange(smin, nn - smin + 1)
            ar = np.arange(nn)
            idx_sh = np.vstack([(ar[None, :] + sh[:, None]) % nn, (ar[::-1][None, :] + sh[:, None]) % nn])
            mat = np.vstack([np.median(lambda_w(ctx, seqs[k][3][idx_sh]), axis=1) for k in keys])
            # placebo 2 (secondary): independent block shuffles per rule, block length = rule's mean run length (>=6)
            matb = []
            for k in keys:
                f = seqs[k][3]
                runs = 1 + np.sum(np.diff(f) != 0)
                L = max(BLOCK, int(round(len(f) / runs)))
                matb.append(placebo_for_rule(f, ctx, L, N_PLACEBO, RNG))
            matb = np.vstack(matb)
            for i, k in enumerate(keys):
                plc_rows.append(dict(pair=pname, horizon=hname, rule=f'{seqs[k][0]}|{seqs[k][2]}',
                                     real_med_ratio=real_arr[i], n_windows=ctx.W, n_placebo_shift=mat.shape[1],
                                     shift_mean=mat[i].mean(), shift_p5=np.percentile(mat[i], 5),
                                     shift_p95=np.percentile(mat[i], 95), pctile_shift=np.mean(mat[i] < real_arr[i]) * 100,
                                     block_p5=np.percentile(matb[i], 5), block_p95=np.percentile(matb[i], 95),
                                     pctile_block=np.mean(matb[i] < real_arr[i]) * 100))
            mx, mn = mat.max(axis=0), mat.min(axis=0)
            mxb, mnb = matb.max(axis=0), matb.min(axis=0)
            hi = np.array([np.percentile(mat[i], 95) for i in range(len(keys))])
            lo = np.array([np.percentile(mat[i], 5) for i in range(len(keys))])
            plcmax_rows.append(dict(pair=pname, horizon=hname, n_rule_variants=n_var, n_windows=ctx.W,
                                    n_placebo_shift=mat.shape[1],
                                    best_rule=keys[int(real_arr.argmax())].replace('|', ' '),
                                    best_real_med_ratio=real_arr.max(),
                                    shift_null_max_median=np.median(mx), shift_null_max_p95=np.percentile(mx, 95),
                                    pct_best_in_shift_max_null=np.mean(mx < real_arr.max()) * 100,
                                    block_null_max_p95=np.percentile(mxb, 95),
                                    pct_best_in_block_max_null=np.mean(mxb < real_arr.max()) * 100,
                                    worst_rule=keys[int(real_arr.argmin())].replace('|', ' '),
                                    worst_real_med_ratio=real_arr.min(),
                                    pct_worst_in_shift_min_null=np.mean(mn < real_arr.min()) * 100,
                                    n_rules_real_gt_1=int((real_arr > 1).sum()),
                                    n_above_own_shift_p95=int((real_arr > hi).sum()),
                                    n_below_own_shift_p5=int((real_arr < lo).sum())))
            summary.extend(recs)
            store[(pname, hname)] = dict(ctx=ctx)
            print(f'  {pname} {hname}: windows={ctx.W}, variants={n_var}', flush=True)
    S = pd.DataFrame(summary)
    cols = ['pair', 'horizon', 'rule', 'family', 'version', 'n_windows', 'avg_fracA', 'irr_med', 'irr_p10', 'irr_min',
            'wm_med', 'vs50_beat', 'vs50_med', 'vs50_mean', 'vsmatch_beat', 'vsmatch_med', 'vsmatch_mean',
            'vsbest_med', 'vsbest_beat', 'vsAllA_med', 'vsAllB_med']
    S = S[cols]
    save(S.round(4), 'b_rules_summary.csv', index=False)
    E = pd.DataFrame(era_all)
    E = E[['pair', 'horizon', 'rule', 'era', 'n_windows', 'avg_fracA', 'vsmatch_beat', 'vsmatch_med',
           'vs50_beat', 'vs50_med', 'irr_med']]
    save(E.round(4), 'b_rules_by_era.csv', index=False)
    P = pd.DataFrame(plc_rows)
    save(P.round(4), 'b_placebo.csv', index=False)
    PM = pd.DataFrame(plcmax_rows)
    save(PM.round(4), 'b_placebo_max.csv', index=False)
    return S, E, P, PM


def fig_rules(S, P, pname, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 11), sharey=False)
    cmap = {'momentum': '#1f77b4', 'contrarian': '#ff7f0e', 'trend': '#2ca02c', 'drawdown': '#d62728', 'ratio_z': '#9467bd'}
    for ax, h in zip(axes, ('10y', '20y')):
        s = S[(S.pair == pname) & (S.horizon == h) & (S.family != 'fixed')].copy()
        if s.empty:
            ax.set_visible(False)
            continue
        p = P[(P.pair == pname) & (P.horizon == h)].set_index('rule')
        s = s.sort_values('vsmatch_med')
        y = np.arange(len(s))
        ax.barh(y, (s.vsmatch_med.values - 1) * 100, color=[cmap[f] for f in s.family], height=0.7)
        lo = (p.loc[s.rule, 'shift_p5'].values - 1) * 100
        hi = (p.loc[s.rule, 'shift_p95'].values - 1) * 100
        ax.hlines(y, lo, hi, color='k', lw=1.6, alpha=0.7, label='placebo 5-95% (signals shifted in time)')
        ax.set_yticks(y)
        ax.set_yticklabels(s.rule, fontsize=6)
        ax.axvline(0, color='k', lw=0.6)
        nw = int(s.n_windows.iloc[0])
        ax.set_title(f'{pname}, {h} windows (n={nw})', fontsize=10)
        ax.set_xlabel('Median wealth ratio vs fixed mix with same A-fraction, % (0 = no timing value)')
        if ax is axes[0]:
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
            ax.legend(handles + [ax.lines[0] if False else plt.Line2D([0], [0], color='k')],
                      list(cmap) + ['placebo 5-95%'], fontsize=7, loc='lower right')
    fig.suptitle(f'Contribution-direction rules: median wealth ratio vs matched fixed mix  ({pname}; 48 variants per panel)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- buy_underweight
def part_b_underweight(pairs_data):
    rows = []
    for pname, d in pairs_data.items():
        D, opens, closes = d['D'], d['opens'], d['closes']
        n = len(D)
        windows = [('full', 0, n)]
        for hname, H in (('10y', 120), ('20y', 240)):
            for s in range(0, n - H, 12):
                windows.append((hname, s, s + H))
        for hname, s, e in windows:
            if hname == 'full':
                d0, d1 = D[0], None
                contrib = pd.Series(1.0, index=D)
                o = opens.loc[D[0]:]
                c = closes.loc[D[0]:]
                cut = None
            else:
                contrib = pd.Series(1.0, index=D[s:e])
                o = opens.loc[D[s]:D[e]]
                c = closes.loc[D[s]:D[e]]
                cut = D[e]
            out = {}
            for pol_name, pol in (('buy_underweight', buy_underweight({'A': 0.5, 'B': 0.5})),
                                  ('fixed_buy', fixed_buy({'A': 0.5, 'B': 0.5}))):
                led = DCABacktest(slippage_rate=SLIP).run(o.rename(columns={}), c, pol, contrib)
                if cut is not None:
                    led = led.loc[led.index < cut]
                m = dca_metrics(led)
                last = led.iloc[-1]
                out[pol_name] = dict(irr=m['irr'], mdd=m['max_drawdown'], wm=m['wealth_multiple'],
                                     wA=last['value_A'] / last['total_value'])
            rows.append(dict(pair=pname, horizon=hname, start=D[s] if hname != 'full' else D[0],
                             irr_bu=out['buy_underweight']['irr'], irr_fx=out['fixed_buy']['irr'],
                             mdd_bu=out['buy_underweight']['mdd'], mdd_fx=out['fixed_buy']['mdd'],
                             wm_bu=out['buy_underweight']['wm'], wm_fx=out['fixed_buy']['wm'],
                             wA_bu=out['buy_underweight']['wA'], wA_fx=out['fixed_buy']['wA']))
    W = pd.DataFrame(rows)
    save(W.assign(start=W.start.dt.strftime('%Y-%m-%d')).round(5), 'b_buy_underweight_windows.csv', index=False)
    g = []
    for (p, h), x in W.groupby(['pair', 'horizon'], sort=False):
        g.append(dict(pair=p, horizon=h, n_windows=len(x),
                      irr_buy_underweight=x.irr_bu.median(), irr_fixed=x.irr_fx.median(),
                      irr_diff_med_pp=(x.irr_bu - x.irr_fx).median() * 100,
                      irr_diff_mean_pp=(x.irr_bu - x.irr_fx).mean() * 100,
                      pct_windows_bu_better=np.mean((x.wm_bu > x.wm_fx + EPS)) * 100,
                      mdd_bu=x.mdd_bu.median(), mdd_fixed=x.mdd_fx.median(),
                      final_wA_bu=x.wA_bu.median(), final_wA_fixed=x.wA_fx.median(),
                      wm_ratio_med=(x.wm_bu / x.wm_fx).median()))
    G = pd.DataFrame(g)
    save(G.round(4), 'b_buy_underweight_summary.csv', index=False)
    return W, G


# --------------------------------------------------------------------------- relative strength + current signals
def fig_relative_strength(pairs_data, px):
    d = pairs_data['QQQ/BRK-B']
    c = d['closes']
    ratio = c['A'] / c['B']
    ratio = ratio / ratio.iloc[0]
    sg = d['sigA'], d['sigB']
    com = sg[0].index.intersection(sg[1].index)
    lr = np.log(sg[0].loc[com] / sg[1].loc[com])
    mu = lr.rolling(756, min_periods=700).mean()
    sd = lr.rolling(756, min_periods=700).std()
    z = ((lr - mu) / sd).loc['1999-03':]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    ax = axes[0]
    ax.plot(ratio.index, ratio.values, color='#1f77b4')
    ax.set_yscale('log')
    ax.scatter([ratio.index[-1]], [ratio.iloc[-1]], color='red', zorder=5, s=40)
    ax.annotate(f'{ratio.index[-1]:%Y-%m-%d}: {ratio.iloc[-1]:.2f}x', (ratio.index[-1], ratio.iloc[-1]),
                textcoords='offset points', xytext=(-95, 8), color='red', fontsize=9)
    ax.set_title('QQQ / BRK-B relative strength (total return, rebased to 1 on 1999-03; log scale)')
    ax.grid(alpha=0.3)
    ax = axes[1]
    nd = (px['^NDX'] / px['BRK-A']).dropna()
    nd = (nd / nd.iloc[0])
    ax.plot(nd.index, nd.values, color='#2ca02c')
    ax.set_yscale('log')
    ax.scatter([nd.index[-1]], [nd.iloc[-1]], color='red', zorder=5, s=40)
    ax.set_title('^NDX (price index) / BRK-A, rebased to 1 on 1985-10 (log scale)')
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(z.index, z.values, color='#9467bd', lw=0.9)
    ax.axhline(0, color='k', lw=0.6)
    for lv in (1, -1):
        ax.axhline(lv, color='gray', ls='--', lw=0.8)
    ax.scatter([z.index[-1]], [z.iloc[-1]], color='red', zorder=5, s=40)
    ax.annotate(f'z = {z.iloc[-1]:.2f}', (z.index[-1], z.iloc[-1]), textcoords='offset points', xytext=(-50, 8),
                color='red', fontsize=9)
    ax.set_title('z-score of log(QQQ/BRK-B) vs its trailing 36-month mean/std (signal used by the ratio rule)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_relative_strength.png'), dpi=130)
    plt.close(fig)


def current_signals(pairs_data):
    rows = []
    for pname, d in pairs_data.items():
        last = d['closes'].index[-1]
        dnext = pd.DatetimeIndex([last + pd.Timedelta(days=1)])
        sg = make_signals(d['sigA'], d['sigB'], dnext)
        rows.append(dict(pair=pname, as_of_close=last.date(),
                         rel1m_pp=sg['rel1'][0] * 100, rel3m_pp=sg['rel3'][0] * 100, rel6m_pp=sg['rel6'][0] * 100,
                         rel12m_pp=sg['rel12'][0] * 100, A_vs_sma200_pct=sg['trend'][0] * 100,
                         A_drawdown_from_ath_pct=sg['dd'][0] * 100, ratio_z36m=sg['z'][0]))
    t = pd.DataFrame(rows).round(2)
    save(t, 'b_current_signals.csv', index=False)
    return t


# =========================================================================== main
def main():
    # ---- price data
    long_t = ['VFINX', 'BRK-A', '^NDX', 'QQQ', 'SPY', 'BRK-B']
    px = {}
    for t in long_t:
        px[t] = load_prices([t])['close'][t]
    print('Part A1 ...', flush=True)
    e_sp, e_nd, a1 = part_a1(px)
    print(a1.round(3).to_string())
    print('Part A2 ...', flush=True)
    a2 = part_a2(px)
    print(a2.to_string())

    # ---- pair data
    pairs_data = {}
    for pname, (A, B, start, pa, pb) in PAIRS.items():
        p = load_prices([A, B], start=start)
        opens = p['open'].rename(columns={A: 'A', B: 'B'})
        closes = p['close'].rename(columns={A: 'A', B: 'B'})
        sigA = closes['A'] if pa is None else splice(px[pa], closes['A'])
        sigB = closes['B'] if pb is None else splice(px[pb], closes['B'])
        D = month_starts(opens.index)
        if pa is None:   # no proxies: need 36 months of signal history
            D = D[D >= opens.index[0] + pd.DateOffset(months=37)]
        pairs_data[pname] = dict(opens=opens, closes=closes, sigA=sigA, sigB=sigB, D=D)
        print(f'{pname}: data {opens.index[0].date()}..{opens.index[-1].date()}, decisions {D[0].date()}..{D[-1].date()} (n={len(D)})')

    print('Part A3 ...', flush=True)
    cor, qt = part_a3(pairs_data)
    print(cor.assign(sig=cor.p_circ_shift < 0.05).groupby('pair').agg(cells=('corr', 'size'), n_p05=('sig', 'sum')))

    print('Part B rules ...', flush=True)
    S, E, P, PM = run_part_b(pairs_data)
    print(PM.round(3).to_string())
    fig_rules(S, P, 'QQQ/BRK-B', 'fig3_rules_QQQ_BRKB.png')
    fig_rules(S, P, 'VFINX/BRK-A', 'fig4_rules_VFINX_BRKA.png')

    print('Part B buy_underweight ...', flush=True)
    W, G = part_b_underweight(pairs_data)
    print(G.round(4).to_string())

    fig_relative_strength(pairs_data, px)
    cs = current_signals(pairs_data)
    print(cs.to_string())

    # ---- sanity: one window by hand with DCABacktest, rule = mom6 soft on QQQ/BRK-B, 10y
    d = pairs_data['QQQ/BRK-B']
    D = d['D']
    sg = make_signals(d['sigA'], d['sigB'], D)
    s6 = np.where(sg['rel6'] > 0, 1.0, -1.0)
    fA = 0.5 + 0.25 * s6
    alloc = make_alloc_table(D, fA)
    H = 120
    res = rolling_dca(d['opens'], d['closes'], alloc, H, 1.0, SLIP)
    k = 30
    pol = lambda ctx: {'buy': {'A': float(alloc.loc[ctx.date, 'A']), 'B': float(alloc.loc[ctx.date, 'B'])}}
    contrib = pd.Series(1.0, index=D[k:k + H])
    led = DCABacktest(slippage_rate=SLIP).run(d['opens'].loc[D[k]:D[k + H]], d['closes'].loc[D[k]:D[k + H]], pol, contrib)
    led = led.loc[led.index < D[k + H]]
    m = dca_metrics(led)
    lines = [f'sanity window start={D[k].date()} H={H}',
             f'rolling_dca wealth_multiple={res.wealth_multiple.iloc[k]:.6f} irr={res.irr.iloc[k]:.6f}',
             f'DCABacktest  wealth_multiple={m.wealth_multiple:.6f} irr={m.irr:.6f}',
             f'fixed 50/50 via linearity: {0.5*Ctx(d["opens"], d["closes"], D, H).WA[k] + 0.5*Ctx(d["opens"], d["closes"], D, H).WB[k]:.6f}']
    r50 = rolling_dca(d['opens'], d['closes'], make_alloc_table(D, np.full(len(D), 0.5)), H, 1.0, SLIP)
    lines.append(f'rolling_dca fixed 50/50 wealth_multiple={r50.wealth_multiple.iloc[k]:.6f}')
    open(os.path.join(OUT, 'sanity_check.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    assert abs(res.wealth_multiple.iloc[k] - m.wealth_multiple) < 1e-6
    print('done')


if __name__ == '__main__':
    main()
