"""Shared machinery for EXP7 (out-of-sample / walk-forward tests of the round-1 DCA conclusions).

Everything is built on a `World`: daily open/close arrays for N assets plus the decision rows (first trading day of each
calendar month).  Wealth of any allocation sequence is exactly linear in the allocation (units bought are
fraction / (open * (1 + slippage)), no dependence on other months), so windows are evaluated with cumulative sums.
The fast evaluation is verified against `rolling_dca` and `DCABacktest` in `selftest()`.

No-lookahead conventions (same as exp3):
  * decision row m trades on day pos[m] at that day's open;
  * every signal for row m is computed from closes with index <= pos[m] - 1 (the previous close);
  * a training window ending at decision row j contains contributions of rows < j only and is marked at close[pos[j]-1].
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.dca import _irr_solve, load_prices, month_starts

SLIP = 0.0005
HORIZ = {'5y': 60, '10y': 120, '20y': 240}
M0 = 37                      # first usable decision row (36 months of signal history, as exp3 did for proxy-less pairs)
EPS = 1e-9
OUT = os.path.join(ROOT, 'docs', 'dca', 'exp7_walk_forward')
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------------------------------------ world
class World:
    def __init__(self, opens, closes, pos, dates=None, names=None):
        self.o = np.ascontiguousarray(opens, dtype=float)
        self.c = np.ascontiguousarray(closes, dtype=float)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.dates = dates
        self.names = list(names) if names is not None else None
        self.T, self.N = self.c.shape
        self.n = len(self.pos)
        self.lc = np.log(self.c)
        self.inv = 1.0 / (self.o[self.pos] * (1 + SLIP))      # (n, N): units per $1 bought on each decision row

    @staticmethod
    def from_frames(o_df, c_df):
        pos = o_df.index.get_indexer(month_starts(o_df.index))
        return World(o_df.values, c_df.values, pos, dates=o_df.index, names=list(o_df.columns))

    def col(self, name):
        return self.names.index(name)

    def windows(self, H, m0=M0):
        nv = self.n - m0
        W = nv - H
        s = np.arange(W)
        e = s + H
        Pend = self.c[self.pos[m0 + e] - 1]      # close before the window's end decision date
        return s, e, Pend

    def decision_dates(self, m0=0):
        return self.dates[self.pos[m0:]]


def load_world(tickers, start=None, end=None, extra=None):
    """Real-data world; opens/closes from load_prices (inner join)."""
    p = load_prices(list(tickers), start=start, end=end)
    return World.from_frames(p['open'], p['close'])


# ------------------------------------------------------------------------------------------------ bootstrap worlds
class BlockSource:
    """Daily log returns of a real World cut into calendar-month blocks (decision row m -> rows pos[m] .. pos[m+1]-1)."""

    def __init__(self, W: World):
        self.lr = np.vstack([np.zeros((1, W.N)), np.diff(W.lc, axis=0)])
        pos = W.pos
        self.starts = pos.copy()
        ends = np.r_[pos[1:], W.T]
        self.lens = ends - pos
        self.nm = W.n
        self.names = W.names

    def stationary_idx(self, rng, mean_len):
        nm = self.nm
        new = rng.random(nm) < 1.0 / mean_len
        st = rng.integers(nm, size=nm)
        idx = np.empty(nm, dtype=np.int64)
        idx[0] = st[0]
        for i in range(1, nm):
            idx[i] = st[i] if new[i] else (idx[i - 1] + 1) % nm
        return idx

    def draw(self, rng, mean_len=18):
        idx = self.stationary_idx(rng, mean_len)
        lens = self.lens[idx]
        cum = np.cumsum(lens)
        begin = cum - lens
        rows = np.repeat(self.starts[idx], lens) + (np.arange(cum[-1]) - np.repeat(begin, lens))
        lc = np.cumsum(self.lr[rows], axis=0)
        c = np.exp(lc)
        return World(c, c, begin), idx


# ------------------------------------------------------------------------------------------------ signals
def rolling_z(ratio, win=756, minp=700):
    T = len(ratio)
    cs1 = np.r_[0.0, np.cumsum(ratio)]
    cs2 = np.r_[0.0, np.cumsum(ratio ** 2)]
    t = np.arange(T)
    nw = np.minimum(win, t + 1)
    lo = t + 1 - nw
    s1 = cs1[t + 1] - cs1[lo]
    s2 = cs2[t + 1] - cs2[lo]
    mu = s1 / nw
    var = np.maximum((s2 - s1 * s1 / nw) / np.maximum(nw - 1, 1), 0)
    sd = np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (ratio - mu) / sd
    z[nw < minp] = np.nan
    return z


def pair_signals(W: World, ia: int, ib: int, ks=(1, 3, 6, 12)):
    """Signals for every decision row, from closes strictly before the decision day (index pos-1).
    rel{k}: trailing k-month log relative return A/B between previous-close of row m-k and previous-close of row m."""
    prev = W.pos - 1
    ok = prev >= 0
    pv = np.maximum(prev, 0)
    la, lb = W.lc[:, ia], W.lc[:, ib]
    LA, LB = la[pv], lb[pv]
    n = W.n
    out = {}
    for k in ks:
        r = np.full(n, np.nan)
        if W.dates is not None:
            # exp3 definition: last bar on or before (decision date - k calendar months)
            look = pd.DatetimeIndex(W.dates[W.pos]) - pd.DateOffset(months=k)
            il = W.dates.searchsorted(look, side='right') - 1
            good = (il >= 0) & ok
            ilc = np.maximum(il, 0)
            r = np.where(good, (la[pv] - la[ilc]) - (lb[pv] - lb[ilc]), np.nan)
        else:   # synthetic worlds: previous close of decision row m-k
            r[k:] = (LA[k:] - LA[:-k]) - (LB[k:] - LB[:-k])
            r[~ok] = np.nan
        out[f'rel{k}'] = r
    c = W.c[:, ia]
    cs = np.r_[0.0, np.cumsum(c)]
    sma = np.full(W.T, np.nan)
    sma[199:] = (cs[200:] - cs[:-200]) / 200.0
    out['trend'] = np.where(ok, c[pv] / sma[pv] - 1, np.nan)
    dd = c / np.maximum.accumulate(c) - 1
    out['dd'] = np.where(ok, dd[pv], np.nan)
    z = rolling_z(la - lb)
    out['z'] = np.where(ok, z[pv], np.nan)
    return out


def sign_rules(sg):
    """name -> (family, s) with s in {+1 favour A, -1 favour B, 0 neutral}; identical to exp3.sign_rules."""
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


FROZEN = ['mom12', 'contra12', 'mom6', 'contra6', 'z_mean_revert', 'dd20_dip', 'dd20_flee',
          'trend_A_above_sma200', 'trend_rev']
FIXED = {'fixed_100A': 1.0, 'fixed_0A_100B': 0.0, 'fixed_50_50': 0.5, 'fixed_70A_30B': 0.7, 'fixed_30A_70B': 0.3}


def frozen_fracs(sg, m0=M0):
    """(9, nv) hard 100/0 allocation fractions to A (0.5 + 0.5 s) for the pre-registered rules, valid rows only."""
    R = sign_rules(sg)
    F = np.vstack([0.5 + 0.5 * R[nm][1][m0:] for nm in FROZEN])
    sgv = {k: v[m0:] for k, v in sg.items()}
    assert all(np.isfinite(v).all() for v in sgv.values()), 'NaN signal in valid rows'
    return F


def all_variants(sg, m0=M0):
    """48 rule variants (hard/soft) + 5 fixed mixes -> names, (53, nv) fractions."""
    R = sign_rules(sg)
    names, rows = [], []
    nv = len(sg['z']) - m0
    for nm, (fam, s) in R.items():
        s = s[m0:]
        names.append(f'{nm}|hard'); rows.append(0.5 + 0.5 * s)
        names.append(f'{nm}|soft'); rows.append(0.5 + 0.25 * s)
    for nm, v in FIXED.items():
        names.append(nm); rows.append(np.full(nv, v))
    assert all(np.isfinite(v).all() for v in sg.values() if len(v) == len(sg['z'])) or True
    return names, np.vstack(rows)


# ------------------------------------------------------------------------------------------------ two-asset evaluation
class PairEval:
    """Window wealth of allocation sequences F (fraction to A each valid decision row) for a pair (ia, ib)."""

    def __init__(self, W: World, ia: int, ib: int, Hs=HORIZ, m0=M0, need=24):
        self.W, self.m0, self.ia, self.ib = W, m0, ia, ib
        self.invA = W.inv[m0:, ia]
        self.invB = W.inv[m0:, ib]
        self.nv = W.n - m0
        z = np.zeros(1)
        cA = np.r_[z, np.cumsum(self.invA)]
        cB = np.r_[z, np.cumsum(self.invB)]
        self.h = {}
        for name, H in Hs.items():
            if self.nv - H < need:
                continue
            s, e, Pend = W.windows(H, m0)
            PA, PB = Pend[:, ia], Pend[:, ib]
            self.h[name] = dict(H=H, s=s, e=e, PA=PA, PB=PB,
                                WA=PA * (cA[e] - cA[s]) / H, WB=PB * (cB[e] - cB[s]) / H)

    def ratios(self, F):
        """F (K, nv) -> {horizon: (ratio_to_window_matched_mix (K,W), fav (K,W), wealth (K,W), ratio_to_global_mix (K,W))}."""
        F = np.atleast_2d(F)
        z = np.zeros((F.shape[0], 1))
        cA = np.concatenate([z, np.cumsum(F * self.invA, axis=1)], axis=1)
        cB = np.concatenate([z, np.cumsum((1 - F) * self.invB, axis=1)], axis=1)
        cF = np.concatenate([z, np.cumsum(F, axis=1)], axis=1)
        out = {}
        for name, h in self.h.items():
            s, e, H = h['s'], h['e'], h['H']
            w = (h['PA'] * (cA[:, e] - cA[:, s]) + h['PB'] * (cB[:, e] - cB[:, s])) / H
            fav = (cF[:, e] - cF[:, s]) / H
            matched = fav * h['WA'] + (1 - fav) * h['WB']
            fg = F.mean(axis=1, keepdims=True)       # control 2: ONE fixed mix with the rule's full-sample average fraction
            glob = fg * h['WA'] + (1 - fg) * h['WB']
            out[name] = (w / matched, fav, w, w / glob)
        return out


# ------------------------------------------------------------------------------------------------ N-asset tracks
def track_final(W: World, rows, Wt, upto=None):
    """Chain of $1 monthly contributions on decision rows `rows` with weights Wt (len(rows), N). Returns
    (units (N,), final value marked at the last close of the world)."""
    units = (Wt * W.inv[rows]).sum(axis=0)
    return units, float((units * W.c[-1 if upto is None else upto]).sum())


def irr_rows(t, flows, final, horizon, lo=-1.0, hi=1.0, iters=80):
    """Money-weighted annual return per row by bisection on g = ln(1+irr) in [lo, hi] (irr -63%..+172%).  Same equation as
    src.dca._irr_solve, but with a bracket that does not overflow on 90-year horizons.  t, flows (R, K); final, horizon (R,)."""
    final = np.asarray(final, dtype=float)
    horizon = np.asarray(horizon, dtype=float)

    def f(g):
        return final * np.exp(-g * horizon) - np.sum(flows * np.exp(-g[:, None] * t), axis=1)
    a = np.full(len(final), lo)
    b = np.full(len(final), hi)
    ok = (f(a) > 0) & (f(b) < 0)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = f(m) > 0
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    return np.where(ok, np.exp(0.5 * (a + b)) - 1.0, np.nan)


def track_irr(W: World, rows, final, real_dates=True):
    pos = W.pos[rows]
    if real_dates and W.dates is not None:
        d = np.asarray((W.dates[pos] - W.dates[pos[0]]).days, dtype=float) / 365.0
        hor = (W.dates[-1] - W.dates[pos[0]]).days / 365.0
    else:
        d = (pos - pos[0]) / 252.0
        hor = (W.T - 1 - pos[0]) / 252.0
    return float(irr_rows(d[None, :], np.ones((1, len(rows))), np.array([final]), np.array([hor]))[0])


def stationary_median_nonoverlap(r, H):
    """Mean over the H possible start phases of the median ratio of non-overlapping windows (every H-th window)."""
    meds = [np.median(r[p::H]) for p in range(min(H, len(r)))]
    return float(np.mean(meds)), float(np.min(meds)), float(np.max(meds))
