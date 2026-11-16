"""EXP7 part B: walk-forward rule selection.

Every January (first decision row of the year) all 48 exp3 rule variants + 5 fixed mixes are scored by the DCA wealth of
the trailing window (expanding, rolling 10y, rolling 20y; contributions on rows strictly before the decision row,
marked at the previous close).  The top-scoring candidate's allocation fractions are used for the next 12 contributions.
The 12-month-chained out-of-sample allocations form one DCA track record.
"""
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
SEED = 20261117
SCHEMES = {'expanding': None, 'roll10y': 120, 'roll20y': 240}
PAIRS_B = [('FF_TECH/FF_VALUE', 'fr', 'FF_TECH', 'FF_VALUE'), ('FF_MKT/FF_VALUE', 'fr', 'FF_MKT', 'FF_VALUE'),
           ('VFINX/BRK-A', 'vb', 'VFINX', 'BRK-A'), ('^NDX/BRK-A', 'nb', '^NDX', 'BRK-A')]
_G = {}


class Cand:
    """All candidates (names, F (K, nv)) for a pair in a world, plus cumulative arrays for fast window wealth."""

    def __init__(self, W, ia, ib, m0=M0):
        self.W, self.ia, self.ib, self.m0 = W, ia, ib, m0
        sg = pair_signals(W, ia, ib)
        self.names, self.F = all_variants(sg, m0)
        assert all(np.isfinite(v[m0:]).all() for v in sg.values())
        self.nv = W.n - m0
        self.invA, self.invB = W.inv[m0:, ia], W.inv[m0:, ib]
        z = np.zeros((self.F.shape[0], 1))
        self.cA = np.concatenate([z, np.cumsum(self.F * self.invA, axis=1)], axis=1)
        self.cB = np.concatenate([z, np.cumsum((1 - self.F) * self.invB, axis=1)], axis=1)
        self.cF = np.concatenate([z, np.cumsum(self.F, axis=1)], axis=1)
        self.ix = {n: i for i, n in enumerate(self.names)}

    def prev_close(self, j):
        """closes of A, B strictly before decision row j (relative index)"""
        p = self.W.pos[self.m0 + j] - 1
        return self.W.c[p, self.ia], self.W.c[p, self.ib]

    def score(self, j, L):
        """DCA wealth multiple per $ of every candidate over rows [j-L, j) (L None = expanding from row 0)."""
        lo = 0 if L is None else j - L
        assert lo >= 0, 'training window longer than available history'
        PA, PB = self.prev_close(j)
        return (PA * (self.cA[:, j] - self.cA[:, lo]) + PB * (self.cB[:, j] - self.cB[:, lo])) / (j - lo)


def jan_rows(W, m0=M0):
    d = W.dates[W.pos[m0:]]
    return np.flatnonzero(np.asarray(d.month) == 1)


def select(cand, J, jmin, short_ok=False):
    """picks[scheme] = array of candidate indices for Januaries J[J>=jmin]; returns (Jsel, picks).
    short_ok: with a 10y start the rolling-20y scheme uses all available history (<20y) until 20y exist."""
    Jsel = J[J >= jmin]
    picks = {}
    for sch, L in SCHEMES.items():
        picks[sch] = np.array([int(np.argmax(cand.score(j, None if (L is not None and short_ok and j < L) else L))) for j in Jsel])
    return Jsel, picks


def chain(cand, Jsel, pk):
    """OOS allocation fractions for rows Jsel[0]..nv-1 (each pick used until the next January)."""
    nv = cand.nv
    Fo = np.empty(nv - Jsel[0])
    ends = np.r_[Jsel[1:], nv]
    for j0, j1, k in zip(Jsel, ends, pk):
        Fo[j0 - Jsel[0]:j1 - Jsel[0]] = cand.F[k, j0:j1]
    return Fo


def track(W, cand, j0, Fo):
    """final wealth per $ contributed over rows j0.. with fractions Fo, marked at the last close of the world."""
    uA = (Fo * cand.invA[j0:]).sum()
    uB = ((1 - Fo) * cand.invB[j0:]).sum()
    return (uA * W.c[-1, cand.ia] + uB * W.c[-1, cand.ib]) / len(Fo)


def track_stats(W, cand, j0, Fo):
    n = len(Fo)
    rows = cand.m0 + j0 + np.arange(n)
    fin = track(W, cand, j0, Fo) * n
    irr = track_irr(W, rows, fin)
    return fin / n, irr


def comparators(W, cand, j0):
    n = cand.nv - j0
    sl = slice(j0, None)
    wA = cand.invA[sl].sum() * W.c[-1, cand.ia] / n
    wB = cand.invB[sl].sum() * W.c[-1, cand.ib] / n
    return wA, wB


def _boot_job(b):
    rng = np.random.default_rng([SEED, b])
    cache = {}
    res = np.full((len(PAIRS_B), len(SCHEMES), 3), np.nan)
    for i, (lab, wk, a, bnm) in enumerate(PAIRS_B):
        if wk not in cache:
            cache[wk] = _G['src'][wk].draw(rng, MEAN_BLOCK)[0]
        W = cache[wk]
        cand = Cand(W, _G['ia'][i], _G['ib'][i])
        Jsel, picks = select(cand, _G['J'][i], _G['jmin'][i])
        j0 = Jsel[0]
        wA, wB = comparators(W, cand, j0)
        w50 = 0.5 * wA + 0.5 * wB
        for s, sch in enumerate(SCHEMES):
            Fo = chain(cand, Jsel, picks[sch])
            w = track(W, cand, j0, Fo)
            fav = Fo.mean()
            res[i, s] = (np.log(w / w50), np.log(w / (fav * wA + (1 - fav) * wB)), fav)
    return b, res


def selftest(worlds):
    """No lookahead: scrambling prices from pos[j*] onward leaves every pick at Januaries <= j* unchanged."""
    W = worlds['vb']
    cand = Cand(W, 0, 1)
    J = jan_rows(W)
    Jsel, picks = select(cand, J, 240)
    jstar = Jsel[len(Jsel) // 2]
    cut = W.pos[M0 + jstar]
    o, c = W.o.copy(), W.c.copy()
    rng = np.random.default_rng(5)
    c[cut:] *= rng.uniform(0.3, 3, size=(W.T - cut, W.N))
    o[cut:] = c[cut:]
    W2 = World(o, c, W.pos, dates=W.dates)
    cand2 = Cand(W2, 0, 1)
    Jsel2, picks2 = select(cand2, J, 240)
    k = int(np.searchsorted(Jsel, jstar, side='right'))
    for sch in SCHEMES:
        assert np.array_equal(picks[sch][:k], picks2[sch][:k]), f'lookahead in selection ({sch})'
    assert any(not np.array_equal(picks[s][k:], picks2[s][k:]) for s in SCHEMES), 'scrambled future should change later picks'
    # chained track equals a DCABacktest run driven by the same fractions (policy looks up the date's fraction)
    from src.dca import DCABacktest, dca_metrics
    j0 = Jsel[0]
    Fo = chain(cand, Jsel, picks['roll10y'])
    D = W.dates[W.pos[M0 + j0:]]
    fr = dict(zip(D, Fo))
    pol = lambda ctx: {'buy': {'A': float(fr[ctx.date]), 'B': float(1 - fr[ctx.date])}}
    op = pd.DataFrame(W.o, index=W.dates, columns=['A', 'B']).loc[D[0]:]
    cl = pd.DataFrame(W.c, index=W.dates, columns=['A', 'B']).loc[D[0]:]
    led = DCABacktest(slippage_rate=SLIP).run(op, cl, pol, pd.Series(1.0, index=D))
    m = dca_metrics(led)
    w, irr = track_stats(W, cand, j0, Fo)
    assert abs(m.wealth_multiple - w) < 1e-9 * w and abs(m.irr - irr) < 1e-6, (m.wealth_multiple, w, m.irr, irr)
    return f'selftest B passed (picks unchanged by scrambling the future; chained track == DCABacktest, wm {w:.4f}, irr {irr:.4%}): picks at Januaries <= {W.dates[cut].date()} unchanged when all prices from that date on are scrambled'


def run():
    t0 = time.time()
    from exp7a_oos_replication import build_all
    worlds, _, _ = build_all()
    msg = selftest(worlds)
    print(msg, flush=True)
    _G.update(src={k: BlockSource(w) for k, w in worlds.items()}, ia=[], ib=[], J=[], jmin=[])
    cands, rows_out, picks_out, roll_rows, shr_rows, eq = {}, [], [], [], [], {}
    for (lab, wk, a, bnm) in PAIRS_B:
        W = worlds[wk]
        ia, ib = W.col(a) if W.names else 0, W.col(bnm)
        _G['ia'].append(ia); _G['ib'].append(ib)
    for i, (lab, wk, a, bnm) in enumerate(PAIRS_B):
        W = worlds[wk]
        ia, ib = _G['ia'][i], _G['ib'][i]
        cand = Cand(W, ia, ib)
        J = jan_rows(W)
        _G['J'].append(J)
        _G['jmin'].append(240)
        pe = PairEval(W, ia, ib)
        d0 = W.dates[W.pos[M0:]]
        for tag, jmin in (('common_start_20y', 240), ('start_10y', 120)):
            Jsel, picks = select(cand, J, jmin, tag == 'start_10y')
            j0 = Jsel[0]
            wA, wB = comparators(W, cand, j0)
            n = cand.nv - j0
            rows_ = np.arange(M0 + j0, M0 + cand.nv)
            # all-candidate OOS-span track (hindsight best)
            allc = []
            for k, nm in enumerate(cand.names):
                w, irr = track_stats(W, cand, j0, cand.F[k, j0:])
                allc.append((nm, w, irr))
            allc = pd.DataFrame(allc, columns=['rule', 'wm', 'irr'])
            best = allc.loc[allc.irr.idxmax()]
            recs = {}
            for sch in SCHEMES:
                Fo = chain(cand, Jsel, picks[sch])
                w, irr = track_stats(W, cand, j0, Fo)
                fav = Fo.mean()
                recs[sch] = (w, irr, fav, Fo)
            # fixed comparators
            f50 = track_stats(W, cand, j0, np.full(n, 0.5))
            fm = {sch: track_stats(W, cand, j0, np.full(n, recs[sch][2])) for sch in SCHEMES}
            c12 = track_stats(W, cand, j0, cand.F[cand.ix['contra12|hard'], j0:])
            m12 = track_stats(W, cand, j0, cand.F[cand.ix['mom12|hard'], j0:])
            base = dict(pair=lab, oos_start=str(d0[j0].date()), oos_end=str(W.dates[-1].date()), oos_months=n, universe=tag)
            for sch in SCHEMES:
                w, irr, fav, Fo = recs[sch]
                rows_out.append(dict(base, strategy=f'WF {sch}', wm=w, irr=irr, avg_fracA=fav))
            rows_out += [dict(base, strategy='fixed 50/50', wm=f50[0], irr=f50[1], avg_fracA=0.5)]
            for sch in SCHEMES:
                rows_out.append(dict(base, strategy=f'matched fixed ({sch} avg A-fraction)', wm=fm[sch][0], irr=fm[sch][1], avg_fracA=recs[sch][2]))
            rows_out += [dict(base, strategy='always contra12 hard', wm=c12[0], irr=c12[1], avg_fracA=cand.F[cand.ix['contra12|hard'], j0:].mean()),
                         dict(base, strategy='always mom12 hard', wm=m12[0], irr=m12[1], avg_fracA=cand.F[cand.ix['mom12|hard'], j0:].mean()),
                         dict(base, strategy=f'hindsight-best of 53 candidates in OOS span ({best.rule})', wm=best.wm, irr=best.irr, avg_fracA=cand.F[cand.ix[best.rule], j0:].mean()),
                         dict(base, strategy='median of 53 candidates in OOS span', wm=allc.wm.median(), irr=allc.irr.median(), avg_fracA=np.nan),
                         dict(base, strategy='worst of 53 candidates in OOS span', wm=allc.wm.min(), irr=allc.irr.min(), avg_fracA=np.nan)]
            if tag == 'common_start_20y':
                # pick frequencies
                for sch in SCHEMES:
                    cnt = pd.Series(np.array(cand.names)[picks[sch]]).value_counts()
                    for nm_, c_ in cnt.items():
                        picks_out.append(dict(pair=lab, scheme=sch, rule=nm_, n_picks=int(c_), share=c_ / len(Jsel), n_januaries=len(Jsel)))
                # rolling windows of the OOS chain (windows starting at/after OOS start)
                for sch in SCHEMES:
                    Ff = np.full(cand.nv, 0.5)
                    Ff[j0:] = recs[sch][3]
                    rs = pe.ratios(Ff)
                    for h, (rm, fav, w, rg) in rs.items():
                        hh = pe.h[h]
                        msk = hh['s'] >= j0
                        if msk.sum() < 6:
                            continue
                        w50 = 0.5 * hh['WA'] + 0.5 * hh['WB']
                        r50 = (w[0] / w50)[msk]
                        rmm = rm[0][msk]
                        no50 = stationary_median_nonoverlap(r50, hh['H'])
                        roll_rows.append(dict(pair=lab, scheme=sch, horizon=h, n_windows=int(msk.sum()),
                                              med_vs_50_50=np.median(r50), share_gt1_vs_50_50=np.mean(r50 > 1), nonov_med_vs_50_50=no50[0],
                                              nonov_n=len(r50[::hh['H']]),
                                              med_vs_matched=np.median(rmm), share_gt1_vs_matched=np.mean(rmm > 1)))
                # shrinkage: roll10y picks, in-sample (training 120m) ratio to fixed 50/50 vs forward 60m ratio of the same rule
                for jj, k in zip(Jsel, picks['roll10y']):
                    sc = cand.score(jj, 120)
                    k50 = cand.ix['fixed_50_50']
                    ins = sc[k] / sc[k50]
                    if jj + 60 <= cand.nv - 1:
                        PA, PB = (W.c[W.pos[M0 + jj + 60] - 1, ia], W.c[W.pos[M0 + jj + 60] - 1, ib])
                        fw = lambda q: (PA * (cand.cA[q, jj + 60] - cand.cA[q, jj]) + PB * (cand.cB[q, jj + 60] - cand.cB[q, jj])) / 60
                        out_ = fw(k) / fw(k50)
                        # same for the best-in-training-by-rank median candidate (how a random candidate fares)
                        shr_rows.append(dict(pair=lab, jan=str(d0[jj].date()), rule=cand.names[k], in_sample_ratio=ins, fwd5y_ratio=out_,
                                             median_cand_train_ratio=np.median(sc / sc[k50]),
                                             median_cand_fwd_ratio=np.median(fw(np.arange(len(cand.names))) / fw(k50))))
                eq[lab] = dict(d=d0[j0:], rec=recs, cand=cand, j0=j0, W=W, f50=0.5)
    R = pd.DataFrame(rows_out)
    R.round(4).to_csv(os.path.join(OUT, 'b_walkforward_tracks.csv'), index=False)
    P = pd.DataFrame(picks_out)
    P.sort_values(['pair', 'scheme', 'n_picks'], ascending=[True, True, False]).round(3).to_csv(os.path.join(OUT, 'b_pick_frequency.csv'), index=False)
    RR = pd.DataFrame(roll_rows)
    RR.round(4).to_csv(os.path.join(OUT, 'b_rolling_oos_windows.csv'), index=False)
    SH = pd.DataFrame(shr_rows)
    SH.round(4).to_csv(os.path.join(OUT, 'b_shrinkage_picks.csv'), index=False)
    print(R[R.universe == 'common_start_20y'].round(4).to_string(), flush=True)

    print(f'bootstrap x{N_BOOT} ...', flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(4) as pool:
        bres = dict(pool.map(_boot_job, range(N_BOOT), chunksize=20))
    BOOT = np.stack([bres[b] for b in range(N_BOOT)])         # (B, pairs, schemes, 3)
    brow = []
    for i, (lab, *_r) in enumerate(PAIRS_B):
        for s, sch in enumerate(SCHEMES):
            for c, cn in enumerate(('WF / fixed 50-50', 'WF / matched mix')):
                x = BOOT[:, i, s, c]
                real_w = R[(R.pair == lab) & (R.universe == 'common_start_20y')].set_index('strategy')
                wf = real_w.loc[f'WF {sch}', 'wm']
                if c == 0:
                    real = wf / real_w.loc['fixed 50/50', 'wm']
                else:
                    real = wf / real_w.loc[f'matched fixed ({sch} avg A-fraction)', 'wm']
                brow.append(dict(pair=lab, scheme=sch, comparison=cn, real_ratio=real, boot_mean_ratio=np.exp(x.mean()),
                                 boot_p5=np.exp(np.percentile(x, 5)), boot_p95=np.exp(np.percentile(x, 95)), P_ratio_gt1=np.mean(x > 0),
                                 real_pctile_in_boot=np.mean(x < np.log(real)) * 100, n_paths=N_BOOT))
    BB = pd.DataFrame(brow)
    BB.round(4).to_csv(os.path.join(OUT, 'b_bootstrap_wf_vs_fixed.csv'), index=False)
    print(BB.round(3).to_string(), flush=True)

    figs(R, SH, eq, BB, worlds)
    print(f'part B done {time.time() - t0:.0f}s', flush=True)
    return dict(R=R, P=P, RR=RR, SH=SH, BB=BB, msg=msg, n_boot=N_BOOT)


def figs(R, SH, eq, BB, worlds):
    # fig4: in-sample vs OOS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    cols = {'FF_TECH/FF_VALUE': '#1f77b4', 'FF_MKT/FF_VALUE': '#2ca02c', 'VFINX/BRK-A': '#d62728', '^NDX/BRK-A': '#9467bd'}
    for lab, g in SH.groupby('pair'):
        ax.scatter((g.in_sample_ratio - 1) * 100, (g.fwd5y_ratio - 1) * 100, s=14, alpha=0.6, color=cols[lab], label=lab)
    ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('picked rule: trailing-10y DCA wealth / fixed 50-50, % (in-sample)')
    ax.set_ylabel('same rule, next 5y DCA wealth / fixed 50-50, % (out-of-sample)')
    m_in, m_out = (SH.in_sample_ratio - 1).mean() * 100, (SH.fwd5y_ratio - 1).mean() * 100
    ax.set_title(f'Each January pick (rolling 10y): mean in-sample {m_in:+.1f}%, mean out-of-sample {m_out:+.1f}%')
    ax.legend(fontsize=7)
    ax = axes[1]
    Rc = R[R.universe == 'common_start_20y']
    strat = ['fixed 50/50', 'WF expanding', 'WF roll10y', 'WF roll20y', 'always contra12 hard', 'always mom12 hard']
    pairs = list(dict.fromkeys(Rc.pair))
    x = np.arange(len(pairs))
    wdt = 0.12
    for k, st in enumerate(strat + ['hindsight']):
        vals = []
        for p in pairs:
            g = Rc[Rc.pair == p].set_index('strategy')
            if st == 'hindsight':
                vals.append(g[g.index.str.startswith('hindsight')].irr.iloc[0] * 100)
            else:
                vals.append(g.loc[st, 'irr'] * 100)
        ax.bar(x + (k - 3) * wdt, vals, wdt, label=st)
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel('OOS DCA IRR, % / yr')
    lo = Rc.irr.min() * 100
    ax.set_ylim(lo - 1.5, Rc.irr.max() * 100 + 1.5)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('OOS-span IRR: walk-forward vs fixed vs hindsight-best candidate')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_wf_rules_shrinkage.png'), dpi=130)
    plt.close(fig)

    # fig5: OOS equity (wealth ratio vs fixed 50/50) over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (lab, e) in zip(axes.ravel(), eq.items()):
        W, cand, j0 = e['W'], e['cand'], e['j0']
        n = cand.nv - j0
        def curve(Fo):
            cum_a = np.cumsum(Fo * cand.invA[j0:])
            cum_b = np.cumsum((1 - Fo) * cand.invB[j0:])
            pos = W.pos[M0 + j0:]
            # value at the previous close of each next decision row
            pe_ = np.r_[W.pos[M0 + j0 + 1:], W.T] - 1
            return (cum_a * W.c[pe_, cand.ia] + cum_b * W.c[pe_, cand.ib]) / np.arange(1, n + 1)
        base = curve(np.full(n, 0.5))
        d = W.dates[W.pos[M0 + j0:]]
        for sch, st in (('expanding', '-'), ('roll10y', '--'), ('roll20y', ':')):
            ax.plot(d, curve(e['rec'][sch][3]) / base, st, label=f'WF {sch}')
        ax.plot(d, curve(cand.F[cand.ix['contra12|hard'], j0:]) / base, color='gray', lw=0.8, label='always contra12')
        ax.plot(d, curve(cand.F[cand.ix['mom12|hard'], j0:]) / base, color='k', lw=0.8, label='always mom12')
        ax.axhline(1, color='r', lw=0.6)
        ax.set_title(lab, fontsize=10)
        ax.set_ylabel('portfolio value / fixed 50-50 value')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Out-of-sample chained DCA tracks (value per $ contributed, relative to fixed 50/50)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_wf_rules_equity.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    run()
