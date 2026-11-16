"""EXP7 part A: pre-registered out-of-sample replication of the round-1 contribution-direction conclusions.

Frozen rules (hard 100/0, exp3 definitions): mom12, contra12, mom6, contra6, z_mean_revert, dd20_dip, dd20_flee,
trend_A_above_sma200, trend_rev.  Applied to pairs never used in round 1 (century US French data, 66 industry pairs,
international pairs).  Metric: median over rolling windows of wealth / wealth of the fixed mix with the same average
A-fraction in that window ("matched").  Significance: (1) circular-shift placebo (signals rotated/reversed against prices,
common shift for all pairs so cross-pair dependence is preserved), (2) stationary block bootstrap of joint monthly blocks
of daily returns (mean block 18 months, 2000 paths).
"""
import itertools
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_common import *  # noqa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_BOOT = int(os.environ.get('EXP7_NBOOT', 2000))
MEAN_BLOCK = 18
SEED = 20261116
IND = ['BusEq', 'Chems', 'Durbl', 'Enrgy', 'Hlth', 'Manuf', 'Money', 'NoDur', 'Other', 'Shops', 'Telcm', 'Utils']
HN = list(HORIZ)
ERAS = ['start<1963', 'start1963-98', 'start>=1999', 'end<1980']
EXPECT = {'mom12': -1, 'mom6': -1, 'contra12': 1, 'contra6': 1, 'z_mean_revert': 1, 'dd20_dip': 1, 'dd20_flee': -1,
          'trend_A_above_sma200': 0, 'trend_rev': 0}

_G = {}   # globals shared with forked workers


# ------------------------------------------------------------------------------------------------ worlds
def build_french():
    names = ['FF_MKT', 'FF_VALUE', 'FF_GROWTH', 'FF_TECH'] + ['FF_IND_' + x for x in IND]
    p = load_prices(names)
    cl = p['close'].copy()
    mon = cl.resample('ME').last().pct_change().loc[:'1962-12']
    beta = {}
    for x in IND:
        d = pd.concat([mon['FF_IND_' + x], mon['FF_MKT']], axis=1).dropna()
        beta[x] = np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var()
    low3 = sorted(beta, key=beta.get)[:3]
    dr = cl[['FF_IND_' + x for x in low3]].pct_change().fillna(0.0).mean(axis=1)
    cl['DEF'] = 100 * (1 + dr).cumprod()
    W = World(cl.values, cl.values, cl.index.get_indexer(month_starts(cl.index)), dates=cl.index, names=list(cl.columns))
    vol = cl.loc[:'1962-12'].pct_change().std()
    return W, beta, low3, vol


def build_all():
    W, beta, low3, vol = build_french()
    named = [('FF_TECH', 'FF_VALUE'), ('FF_TECH', 'FF_MKT'), ('FF_MKT', 'FF_VALUE'), ('FF_GROWTH', 'FF_VALUE'),
             ('FF_TECH', 'DEF')]
    panel = []
    for a, b in itertools.combinations(IND, 2):
        a, b = 'FF_IND_' + a, 'FF_IND_' + b
        if vol[a] < vol[b]:
            a, b = b, a          # A = the more volatile industry over 1926-1962 (orientation fixed before looking at results)
        panel.append((a, b))
    worlds = {'fr': W,
              'dev': load_world(['FF_MKT', 'FF_DEVXUS']),
              'pr': load_world(['VFINX', 'PRITX']),
              'vb': load_world(['VFINX', 'BRK-A']), 'nb': load_world(['^NDX', 'BRK-A']),
              'qb': load_world(['QQQ', 'BRK-B']), 'sb': load_world(['SPY', 'BRK-B'])}
    pairs = []   # (group, label, world key, ia, ib)
    for a, b in named:
        pairs.append(('named', f'{a}/{b}', 'fr', W.col(a), W.col(b)))
    for a, b in panel:
        pairs.append(('panel', f'{a[7:]}/{b[7:]}', 'fr', W.col(a), W.col(b)))
    pairs.append(('intl', 'FF_MKT/FF_DEVXUS', 'dev', 0, 1))
    pairs.append(('intl', 'VFINX/PRITX', 'pr', 0, 1))
    for wk, lab in (('vb', 'VFINX/BRK-A'), ('nb', '^NDX/BRK-A'), ('qb', 'QQQ/BRK-B'), ('sb', 'SPY/BRK-B')):
        pairs.append(('r1', lab + ' (round-1 pair, no proxy warm-up)', wk, 0, 1))
    info = dict(beta=beta, low3=low3, vol=vol)
    return worlds, pairs, info


# ------------------------------------------------------------------------------------------------ per-pair statistics
def shift_index(nv, step=2, smin=48):
    sh = np.arange(smin, nv - smin + 1, step)
    ar = np.arange(nv)
    return np.vstack([(ar[None, :] + sh[:, None]) % nv, (ar[::-1][None, :] + sh[:, None]) % nv])


def pair_stats(wkey, ia, ib, shifts=False):
    W = _G['worlds'][wkey]
    sg = pair_signals(W, ia, ib)
    F = frozen_fracs(sg)
    pe = PairEval(W, ia, ib)
    res = pe.ratios(F)
    d0 = W.dates[W.pos[M0:]]
    out = {}
    for h, (r, fav, w, rg) in res.items():
        hh = pe.h[h]
        sy = np.asarray(d0[hh['s']].year)
        masks = {'start<1963': sy < 1963, 'start1963-98': (sy >= 1963) & (sy <= 1998), 'start>=1999': sy >= 1999,
                 'end<1980': np.asarray(d0[hh['e']] < pd.Timestamp('1980-01-01'))}
        for k, rule in enumerate(FROZEN):
            rec = dict(n=len(r[k]), avg_fracA=float(fav[k].mean()))
            for ctl, arr in (('', r), ('_g', rg)):
                rr = arr[k]
                no = stationary_median_nonoverlap(rr, hh['H'])
                rec.update({'med' + ctl: float(np.median(rr)), 'share_gt1' + ctl: float(np.mean(rr > 1 + EPS)),
                            'nonov_med' + ctl: no[0], 'nonov_min' + ctl: no[1], 'nonov_max' + ctl: no[2],
                            'nonov_share' + ctl: float(np.mean(rr[::hh['H']] > 1 + EPS)), 'nonov_n': len(rr[::hh['H']])})
                for en, m in masks.items():
                    rec[en + ctl] = float(np.median(rr[m])) if m.sum() >= 12 else np.nan
                    rec[en + '_n'] = int(m.sum())
            out[(rule, h)] = rec
    null = None
    if shifts:
        idx = shift_index(pe.nv)
        nh = len(pe.h)
        null = np.empty((len(FROZEN), nh, 2, len(idx)), dtype=np.float32)
        for k in range(len(FROZEN)):
            rs = pe.ratios(F[k][idx])
            for j, h in enumerate(pe.h):
                null[k, j, 0] = np.median(rs[h][0], axis=1)
                null[k, j, 1] = np.median(rs[h][3], axis=1)
    return out, null


def _stats_job(i):
    g, lab, wk, ia, ib = _G['pairs'][i]
    return i, pair_stats(wk, ia, ib, shifts=True)


def _boot_job(args):
    b, mean_len = args
    rng = np.random.default_rng([SEED, b, mean_len])
    cache = {}
    res = np.full((len(_G['pairs']), len(FROZEN), len(HN), 2), np.nan, dtype=np.float32)
    for i, (g, lab, wk, ia, ib) in enumerate(_G['pairs']):
        if wk not in cache:
            cache[wk] = _G['src'][wk].draw(rng, mean_len)[0]
        W = cache[wk]
        sg = pair_signals(W, ia, ib)
        F = frozen_fracs(sg)
        pe = PairEval(W, ia, ib)
        rs = pe.ratios(F)
        for j, h in enumerate(HN):
            if h in rs:
                res[i, :, j, 0] = np.median(rs[h][0], axis=1)
                res[i, :, j, 1] = np.median(rs[h][3], axis=1)
    return b, res


def pmean(x, null):
    mu = null.mean(axis=-1)
    return (1 + np.sum(np.abs(null - mu[..., None]) >= np.abs(x - mu)[..., None] - 1e-15, axis=-1)) / (1 + null.shape[-1])


# ------------------------------------------------------------------------------------------------ run
def selftest(worlds):
    """(1) fast evaluation == rolling_dca ; (2) no lookahead: signals at rows <= m* unchanged when closes from
    pos[m*] onward are scrambled."""
    from src.dca import rolling_dca
    W = worlds['dev']
    sg = pair_signals(W, 0, 1)
    F = frozen_fracs(sg)
    pe = PairEval(W, 0, 1)
    r = pe.ratios(F)
    D = W.dates[W.pos[M0:]]
    for k in (1, 5):
        alloc = pd.DataFrame({'A': F[k], 'B': 1 - F[k]}, index=D)
        op = pd.DataFrame(W.o, index=W.dates, columns=['A', 'B'])
        cl = pd.DataFrame(W.c, index=W.dates, columns=['A', 'B'])
        res = rolling_dca(op, cl, alloc, 120, 1.0, SLIP)
        assert np.allclose(res.wealth_multiple.values, r['10y'][2][k], rtol=1e-9)
    ms = W.n - 150
    W2 = World(W.o.copy(), W.c.copy(), W.pos, dates=W.dates)
    cut = W.pos[ms]
    rng = np.random.default_rng(1)
    W2.c[cut:] *= rng.uniform(0.5, 2, size=(W.T - cut, 2))
    W2.o[cut:] = W2.c[cut:]
    W2 = World(W2.o, W2.c, W.pos, dates=W.dates)
    sg2 = pair_signals(W2, 0, 1)
    for k in sg:
        a, b = sg[k][:ms + 1], sg2[k][:ms + 1]
        assert np.allclose(a, b, equal_nan=True), f'lookahead in {k}'
    return 'selftest A passed: fast evaluation equals rolling_dca (1e-9); signals at rows<=m* unchanged by scrambling closes from pos[m*] on'


def mechanics_demo(n_paths=300, months=1170, seed=7):
    """Two synthetic assets with IDENTICAL iid daily returns (mu 9%, vol 25%, corr 0.8): no predictability, no mean
    difference.  Shows what the frozen rules score against both controls purely from mechanics."""
    rng = np.random.default_rng(seed)
    sig, rho, mu = 0.25 / np.sqrt(252), 0.8, 0.09 / 252
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    T = months * 21
    rows = {(r, h, c): [] for r in FROZEN for h in HN for c in ('matched', 'global')}
    for _ in range(n_paths):
        z = rng.standard_normal((T, 2)) @ L.T
        lc = np.cumsum(mu - 0.5 * sig ** 2 + sig * z, axis=0)
        c = np.exp(lc)
        W = World(c, c, np.arange(months) * 21)
        sg = pair_signals(W, 0, 1)
        F = frozen_fracs(sg)
        rs = PairEval(W, 0, 1).ratios(F)
        for h in HN:
            if h in rs:
                for k, r in enumerate(FROZEN):
                    rows[(r, h, 'matched')].append(np.median(rs[h][0][k]))
                    rows[(r, h, 'global')].append(np.median(rs[h][3][k]))
    out = [dict(rule=r, horizon=h, control=c, mean_of_median_ratio=np.mean(v), p5=np.percentile(v, 5), p95=np.percentile(v, 95))
           for (r, h, c), v in rows.items() if v]
    D = pd.DataFrame(out)
    D.round(4).to_csv(os.path.join(OUT, 'a_mechanics_iid_demo.csv'), index=False)
    return D


def run():
    t0 = time.time()
    worlds, pairs, info = build_all()
    _G.update(worlds=worlds, pairs=pairs, src={k: BlockSource(w) for k, w in worlds.items()})
    msg = selftest(worlds)
    print(msg, flush=True)
    print('defensive mix = equal weight of', info['low3'], flush=True)
    npairs = len(pairs)
    ctx = mp.get_context('fork')
    print(f'{npairs} pairs; computing real-data stats + shift nulls ...', flush=True)
    with ctx.Pool(4) as pool:
        got = dict(pool.map(_stats_job, range(npairs), chunksize=2))
    print(f'  done {time.time() - t0:.0f}s; bootstrap x{N_BOOT} (mean block {MEAN_BLOCK}m) ...', flush=True)
    with ctx.Pool(4) as pool:
        bres = dict(pool.map(_boot_job, [(b, MEAN_BLOCK) for b in range(N_BOOT)], chunksize=20))
        BOOT = np.stack([bres[b] for b in range(N_BOOT)])      # (B, pairs, rules, H, 2 controls)
        n_iid = min(N_BOOT, 500)
        bres = dict(pool.map(_boot_job, [(b, 1) for b in range(n_iid)], chunksize=20))
        IID = np.stack([bres[b] for b in range(n_iid)])        # iid-month resampling: pure "no predictability" world
    print(f'  done {time.time() - t0:.0f}s', flush=True)

    rows = []
    NULL = {}
    for i, (g, lab, wk, ia, ib) in enumerate(pairs):
        st, null = got[i]
        hs = [h for h in HN if (FROZEN[0], h) in st]
        NULL[i] = (hs, null)
        for k, rule in enumerate(FROZEN):
            for j, h in enumerate(hs):
                rec = st[(rule, h)]
                jh = HN.index(h)
                rec.update(group=g, pair=lab, rule=rule, horizon=h)
                for c, ctl in enumerate(('', '_g')):
                    nl = null[k, j, c].astype(float)
                    bt, it = BOOT[:, i, k, jh, c], IID[:, i, k, jh, c]
                    mu = nl.mean()
                    rec.update({'shift_mean' + ctl: mu, 'shift_p5' + ctl: np.percentile(nl, 5), 'shift_p95' + ctl: np.percentile(nl, 95),
                                'p_shift' + ctl: (1 + np.sum(np.abs(nl - mu) >= abs(rec['med' + ctl] - mu) - 1e-15)) / (1 + len(nl)),
                                'boot_mean' + ctl: np.nanmean(bt), 'boot_p5' + ctl: np.nanpercentile(bt, 5),
                                'boot_p95' + ctl: np.nanpercentile(bt, 95), 'boot_p_gt1' + ctl: float(np.nanmean(bt > 1)),
                                'iid_mean' + ctl: np.nanmean(it)})
                rows.append(rec)
    T = pd.DataFrame(rows)
    keep = ['group', 'pair', 'rule', 'horizon', 'n', 'avg_fracA']
    for ctl in ('', '_g'):
        keep += ['med' + ctl, 'share_gt1' + ctl, 'shift_mean' + ctl, 'shift_p5' + ctl, 'shift_p95' + ctl, 'p_shift' + ctl,
                 'boot_mean' + ctl, 'boot_p5' + ctl, 'boot_p95' + ctl, 'boot_p_gt1' + ctl, 'iid_mean' + ctl,
                 'nonov_med' + ctl, 'nonov_share' + ctl] + [e + ctl for e in ERAS]
    keep += ['nonov_n'] + [e + '_n' for e in ERAS]
    T = T[keep]
    T[T.group != 'panel'].round(4).to_csv(os.path.join(OUT, 'a_pairs_named_intl.csv'), index=False)
    T[T.group == 'panel'][['pair', 'rule', 'horizon', 'n', 'med', 'med_g', 'share_gt1', 'share_gt1_g', 'p_shift_g'] + ERAS + [e + '_g' for e in ERAS] + ['nonov_med', 'nonov_med_g']].round(4).to_csv(
        os.path.join(OUT, 'a_pairs_panel66.csv'), index=False)

    pidx = [i for i, p in enumerate(pairs) if p[0] == 'panel']
    srows = []
    for k, rule in enumerate(FROZEN):
        for h in HN:
            jh = HN.index(h)
            sub = T[(T.group == 'panel') & (T.rule == rule) & (T.horizon == h)]
            if sub.empty:
                continue
            rec = dict(rule=rule, horizon=h, n_pairs=len(sub))
            for c, ctl in enumerate(('', '_g')):
                med = sub['med' + ctl].values
                j = NULL[pidx[0]][0].index(h)
                nl = np.stack([NULL[i][1][k, NULL[i][0].index(h), c].astype(float) for i in pidx])          # (66, S)
                cross_null = np.median(nl, axis=0)
                share_null = (nl > 1).mean(axis=0)
                mu = cross_null.mean()
                real = np.median(med)
                bt = np.nanmedian(BOOT[:, pidx, k, jh, c], axis=1)
                btshare = np.nanmean(BOOT[:, pidx, k, jh, c] > 1, axis=1)
                it = np.nanmedian(IID[:, pidx, k, jh, c], axis=1)
                rec.update({'median_of_pair_medians' + ctl: real, 'share_pairs_gt1' + ctl: np.mean(med > 1),
                            'q25' + ctl: np.percentile(med, 25), 'q75' + ctl: np.percentile(med, 75), 'min' + ctl: med.min(), 'max' + ctl: med.max(),
                            'shift_null_mean' + ctl: mu, 'shift_null_p5' + ctl: np.percentile(cross_null, 5), 'shift_null_p95' + ctl: np.percentile(cross_null, 95),
                            'p_shift_median' + ctl: (1 + np.sum(np.abs(cross_null - mu) >= abs(real - mu) - 1e-15)) / (1 + len(cross_null)),
                            'boot_median_mean' + ctl: np.mean(bt), 'boot_p5' + ctl: np.percentile(bt, 5), 'boot_p95' + ctl: np.percentile(bt, 95),
                            'boot_P_median_gt1' + ctl: np.mean(bt > 1), 'iid_median_mean' + ctl: np.mean(it),
                            'iid_p5' + ctl: np.percentile(it, 5), 'iid_p95' + ctl: np.percentile(it, 95),
                            'nonov_median' + ctl: np.median(sub['nonov_med' + ctl].values),
                            'nonov_share_pairs_gt1' + ctl: np.mean(sub['nonov_med' + ctl].values > 1)})
                for en in ERAS:
                    v = sub[en + ctl].dropna().values
                    rec[en + '_median' + ctl] = np.median(v) if len(v) else np.nan
                    rec[en + '_share_gt1' + ctl] = np.mean(v > 1) if len(v) else np.nan
            srows.append(rec)
    S = pd.DataFrame(srows)
    S.round(4).to_csv(os.path.join(OUT, 'a_panel_summary.csv'), index=False)

    r1 = pd.read_csv(os.path.join(ROOT, 'docs', 'dca', 'exp3_tilt_timing', 'b_rules_summary.csv'))
    r1 = r1[r1.pair.isin(['QQQ/BRK-B', 'SPY/BRK-B', '^NDX/BRK-A', 'VFINX/BRK-A'])]
    ref = {}
    for rule in FROZEN:
        for h in HN:
            x = r1[(r1.rule == rule + '|hard') & (r1.horizon == h)].vsmatch_med
            ref[(rule, h)] = (x.min(), x.max())

    # --------------------------------------------------------------- verdicts (pre-registered criteria, see findings.md)
    vrows = []
    for rule in FROZEN:
        h = '10y'
        s = S[(S.rule == rule) & (S.horizon == h)].iloc[0]
        e = EXPECT[rule]
        nm = T[T.group.isin(['named', 'intl']) & (T.rule == rule) & (T.horizon == h)]
        out = dict(rule=rule, round1_expect={1: '>1', -1: '<1', 0: '~1'}[e], round1_range_10y_matched=f'{ref[(rule, h)][0]:.3f}-{ref[(rule, h)][1]:.3f}')
        verd = {}
        for ctl, lab in (('', 'matched'), ('_g', 'global')):
            dev = s['median_of_pair_medians' + ctl] - 1
            sig = s['p_shift_median' + ctl] < 0.05
            side_ok = (np.sign(dev) == e) if e != 0 else None
            eras_ok = sum(1 for en in ERAS[:3] if not np.isnan(s[en + '_median' + ctl]) and (np.sign(s[en + '_median' + ctl] - 1) == e))
            sh = s['share_pairs_gt1' + ctl]
            # a rule only counts as replicated if the effect is also outside the iid-resampling band (mechanical bias excluded)
            beyond_iid = (s['median_of_pair_medians' + ctl] > s['iid_p95' + ctl]) if e > 0 else (s['median_of_pair_medians' + ctl] < s['iid_p5' + ctl]) if e < 0 else None
            n_side = int(sum(np.sign(nm['med' + ctl].values - 1) == e)) if e != 0 else int(sum(np.abs(nm['med' + ctl].values - 1) < 0.005))
            if e != 0:
                pattern = bool(sig and side_ok and ((sh >= 0.6) if e > 0 else (sh <= 0.4)) and eras_ok >= 2)
                if pattern and beyond_iid:
                    v = 'replicated (beyond iid null)'
                elif pattern:
                    v = 'pattern replicated, but inside the iid no-predictability band'
                elif sig and not side_ok:
                    v = 'reversed'
                elif side_ok:
                    v = 'not replicated (right sign; fails significance / consistency)'
                else:
                    v = 'not replicated (wrong sign, not significant)'
            else:
                v = ('replicated (no effect)' if (not sig and abs(dev) < 0.01) else
                     ('small effect, equals iid band' if not (s['median_of_pair_medians' + ctl] > s['iid_p95' + ctl] or s['median_of_pair_medians' + ctl] < s['iid_p5' + ctl]) else 'contradicted: effect beyond iid band'))
            out.update({f'panel_median{ctl}': s['median_of_pair_medians' + ctl], f'share_pairs_gt1{ctl}': sh,
                        f'shift_band{ctl}': f"{s['shift_null_p5' + ctl]:.3f}-{s['shift_null_p95' + ctl]:.3f}",
                        f'p_shift{ctl}': s['p_shift_median' + ctl],
                        f'boot_90{ctl}': f"{s['boot_p5' + ctl]:.3f}-{s['boot_p95' + ctl]:.3f}",
                        f'iid_90{ctl}': f"{s['iid_p5' + ctl]:.3f}-{s['iid_p95' + ctl]:.3f}",
                        f'eras_same_sign{ctl}': f'{eras_ok}/3', f'named_intl_same_side{ctl}': f'{n_side}/{len(nm)}',
                        f'verdict{ctl}': v})
        vrows.append(out)
    V = pd.DataFrame(vrows)
    V.round(4).to_csv(os.path.join(OUT, 'a_verdicts.csv'), index=False)
    pd.set_option('display.width', 250)
    print(V[['rule', 'panel_median', 'iid_90', 'verdict', 'panel_median_g', 'share_pairs_gt1_g', 'shift_band_g', 'p_shift_g', 'boot_90_g', 'iid_90_g', 'eras_same_sign_g', 'verdict_g']].to_string(), flush=True)

    rr = T[T.group == 'r1'][['pair', 'rule', 'horizon', 'n', 'med', 'iid_mean', 'med_g', 'iid_mean_g', 'boot_p5', 'boot_p95', 'p_shift', 'p_shift_g']]
    rr.round(4).to_csv(os.path.join(OUT, 'a_round1_pairs_vs_iid.csv'), index=False)
    demo = mechanics_demo()
    print(demo[demo.horizon == '10y'].round(4).to_string())
    figs(T, S, ref, pairs)
    print(f'part A done {time.time() - t0:.0f}s', flush=True)
    return dict(T=T, S=S, V=V, demo=demo, ref=ref, info=info, msg=msg, npairs=npairs, pairs=pairs, n_boot=N_BOOT, n_iid=n_iid)


# ------------------------------------------------------------------------------------------------ figures
def figs(T, S, ref, pairs):
    nm = T[T.group.isin(['named', 'intl'])]
    labs = list(dict.fromkeys(nm.pair))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, rule in zip(axes.ravel(), FROZEN):
        sub = nm[(nm.rule == rule) & (nm.horizon == '10y')].set_index('pair').loc[labs]
        x = np.arange(len(labs))
        lo, hi = ref[(rule, '10y')]
        ax.axhspan((lo - 1) * 100, (hi - 1) * 100, color='#f2c94c', alpha=0.35, label='round 1 range (window-matched, 4 pairs)')
        ax.vlines(x - 0.12, (sub.shift_p5_g - 1) * 100, (sub.shift_p95_g - 1) * 100, color='k', lw=4, alpha=0.25, label='placebo 5-95% (global mix)')
        ax.vlines(x + 0.12, (sub.boot_p5_g - 1) * 100, (sub.boot_p95_g - 1) * 100, color='#1f77b4', lw=2, alpha=0.6, label='bootstrap 5-95% (global mix)')
        ax.scatter(x, (sub.med_g - 1) * 100, color='#d62728', zorder=5, s=30, label='actual, vs global-average mix')
        ax.scatter(x, (sub.med - 1) * 100, facecolors='none', edgecolors='k', zorder=5, s=36, label='actual, vs window-matched mix')
        ax.scatter(x, (sub.iid_mean - 1) * 100, marker='_', color='green', s=120, zorder=4, label='iid-resampled world, window-matched')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(rule, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('FF_', '') for l in labs], rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('median wealth ratio - 1, %')
    axes[0, 0].legend(fontsize=5.5, loc='best')
    fig.suptitle('Out-of-sample replication, 10y windows (frozen hard rules, 7 new pairs)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_oos_replication.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for r_, (ctl, cname) in enumerate((('', 'window-matched mix (exp3 control)'), ('_g', 'global-average mix'))):
        for ax, h in zip(axes[r_], ['10y', '20y']):
            data = []
            for k, rule in enumerate(FROZEN):
                v = T[(T.group == 'panel') & (T.rule == rule) & (T.horizon == h)]['med' + ctl].values
                data.append((v - 1) * 100)
                s = S[(S.rule == rule) & (S.horizon == h)].iloc[0]
                ax.fill_between([k - 0.4, k + 0.4], (s['shift_null_p5' + ctl] - 1) * 100, (s['shift_null_p95' + ctl] - 1) * 100, color='k', alpha=0.15,
                                label='placebo 5-95% (cross-pair median)' if k == 0 else None)
                ax.hlines((s['iid_median_mean' + ctl] - 1) * 100, k - 0.4, k + 0.4, color='green', lw=2, label='iid-resampled world (cross-pair median)' if k == 0 else None)
                if ctl == '':
                    lo, hi = ref[(rule, h)]
                    ax.hlines([(lo - 1) * 100, (hi - 1) * 100], k - 0.4, k + 0.4, color='#e0a800', lw=2, label='round-1 range' if k == 0 else None)
            ax.boxplot(data, positions=range(len(FROZEN)), widths=0.5, showfliers=False)
            for k, d in enumerate(data):
                ax.scatter(np.full(len(d), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(d)), d, s=6, color='#1f77b4', alpha=0.5)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(range(len(FROZEN)))
            ax.set_xticklabels(FROZEN, rotation=40, ha='right', fontsize=8)
            ax.set_title(f'66 industry pairs, {h} windows, control: {cname}', fontsize=9)
            ax.set_ylabel('median wealth ratio - 1, %')
        axes[r_, 0].legend(fontsize=6)
    fig.suptitle('Cross-pair distribution (the 66 pairs share 12 industries: not 66 independent tests)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_panel66_distribution.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cols = ['start<1963', 'start1963-98', 'start>=1999', 'end<1980']
    s10 = S[S.horizon == '10y'].set_index('rule').loc[FROZEN]
    for ax, ctl, cname in zip(axes, ('', '_g'), ('window-matched control', 'global-average control')):
        M = np.array([[(s10.loc[r, c + '_median' + ctl] - 1) * 100 for c in cols] for r in FROZEN])
        im = ax.imshow(M, cmap='RdBu', vmin=-4, vmax=4, aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(['starts\n1926-62', 'starts\n1963-98', 'starts\n1999-2026', 'windows\nending <1980'], fontsize=8)
        ax.set_yticks(range(len(FROZEN))); ax.set_yticklabels(FROZEN)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=8)
        ax.set_title(f'{cname}', fontsize=9)
    fig.suptitle('66-pair panel, 10y windows: median over pairs of (wealth / control - 1), %, by era', y=0.999)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_panel_eras.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    run()
