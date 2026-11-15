"""Shared helpers for EXP5 (portfolio management)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.dca import (DCABacktest, load_prices, month_starts, dca_metrics, _irr_solve,  # noqa: F401
                     fixed_buy, buy_underweight, rebalance_to, rolling_dca)

OUT = os.path.join(ROOT, 'docs', 'dca', 'exp5_portfolio_mgmt')
os.makedirs(OUT, exist_ok=True)
SLIP = 0.0005
ENGINE = DCABacktest(slippage_rate=SLIP)


def load_outer(tickers):
    p = load_prices(list(tickers), how='outer')
    return p['open'], p['close']


def splice(o, c, segs):
    """Chain several tickers into one price series (open, close).

    segs = [(ticker, first_date), ...] ascending. Each later segment is scaled so
    its close equals the previous segment's close on the splice date.
    """
    outs_o, outs_c = [], []
    scale = 1.0
    for k, (t, d0) in enumerate(segs):
        d0 = pd.Timestamp(d0)
        d1 = pd.Timestamp(segs[k + 1][1]) if k + 1 < len(segs) else None
        cc, oo = c[t].dropna(), o[t].dropna()
        if k > 0:
            tp, _ = segs[k - 1]
            prev_close_at = c[tp].dropna().loc[d0:].iloc[0]
            new_close_at = cc.loc[d0:].iloc[0]
            scale = scale * prev_close_at / new_close_at   # new scaled close == previous scaled close on d0
        sel_c = cc[(cc.index >= d0) & ((cc.index < d1) if d1 is not None else True)]
        sel_o = oo[(oo.index >= d0) & ((oo.index < d1) if d1 is not None else True)]
        outs_c.append(sel_c * scale)
        outs_o.append(sel_o * scale)
    return pd.concat(outs_o), pd.concat(outs_c)


def synth(o, c, mult, drag_annual=0.0):
    """Synthetic daily-return series: mult x the underlying daily return, minus a yearly drag.

    Close-to-close return = mult*r - drag/252; the overnight (prev close -> open) part is mult*r_overnight.
    """
    cc = c.dropna()
    oo = o.reindex(cc.index)
    rc = cc.pct_change()
    ro = oo / cc.shift() - 1
    new_c = 100 * (1 + (mult * rc - drag_annual / 252).fillna(0)).cumprod()
    new_o = new_c.shift() * (1 + mult * ro)
    return new_o.iloc[1:], new_c.iloc[1:]


def frames(cols: dict):
    """cols: {name: (open_series, close_series)} -> inner-joined (opens, closes) DataFrames."""
    op = pd.concat({k: v[0] for k, v in cols.items()}, axis=1).dropna()
    cl = pd.concat({k: v[1] for k, v in cols.items()}, axis=1).reindex(op.index)
    cl = cl.dropna()
    op = op.reindex(cl.index)
    return op, cl


def full_month_starts(idx, master_idx):
    """First trading day of each month in idx, keeping only months whose first day is the true first trading day."""
    master = set(month_starts(master_idx))
    return pd.DatetimeIndex([d for d in month_starts(idx) if d in master])


def max_dd(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def irr_deposits(dates, amounts, final, end_date) -> float:
    d0 = dates[0]
    t = np.array([(d - d0).days / 365.0 for d in dates])[None, :]
    horizon = np.array([(end_date - d0).days / 365.0])
    return float(_irr_solve(t, np.asarray(amounts, dtype=float)[None, :], np.array([float(final)]), horizon)[0])


def sold_stats(ledger, opens, assets):
    u = ledger[[f'units_{a}' for a in assets]].to_numpy()
    d = np.diff(u, axis=0)
    px = opens.reindex(ledger.index)[assets].to_numpy()[1:]
    sold_row = (np.maximum(-d, 0) * np.nan_to_num(px)).sum(axis=1)
    return float(sold_row.sum()), int((sold_row > 0.01).sum())


def q(x, p):
    return float(np.nanpercentile(np.asarray(x, dtype=float), p))


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches='tight')
