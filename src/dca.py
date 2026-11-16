"""Dollar-cost-averaging simulation on daily bars.

Conventions mirror `src/backtest.py`: a decision is made from information up to
the previous close and executed at the decision day's open. Buys fill at
`open * (1 + slippage)`, sells at `open * (1 - slippage)`, commission is a
rate on traded notional. Prices are expected to be split/dividend adjusted.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.data import DATA_DIR

_EPS = 1e-12


# --------------------------------------------------------------------------- data

def _ticker_path(ticker: str):
    name = ('_' + ticker[1:]) if ticker.startswith('^') else ticker
    return DATA_DIR / f'{name}.csv'


def load_prices(tickers: list[str], start=None, end=None, how: str = 'inner') -> dict[str, pd.DataFrame]:
    """Return {'open': df, 'close': df} date-indexed (plain dates, no 16:00 offset), one column per ticker.

    how='inner' keeps only dates where all tickers have data; how='outer' keeps
    the union (NaN before a ticker starts). A leading '^' in a ticker maps to
    '_' in the file name.
    """
    if isinstance(tickers, str):
        raise TypeError("tickers must be a list. One ticker is a one-item list.")
    if how not in ('inner', 'outer'):
        raise ValueError("how must be 'inner' or 'outer'")
    frames = {}
    for ticker in tickers:
        path = _ticker_path(ticker)
        if not path.exists():
            raise FileNotFoundError(f'Market data file not found: {path}')
        frames[ticker] = pd.read_csv(path, index_col='date', parse_dates=True)[['open', 'close']]
    out = {}
    for field in ('open', 'close'):
        df = pd.concat({t: f[field] for t, f in frames.items()}, axis=1, join=how, sort=True)
        df = df[list(tickers)].sort_index()
        if how == 'inner':
            df = df.dropna()
        out[field] = df
    # open and close must share one index, and (inner) one set of valid rows
    idx = out['open'].index.intersection(out['close'].index) if how == 'inner' else out['open'].index.union(out['close'].index)
    out = {k: v.reindex(idx).loc[start:end] for k, v in out.items()}
    return out


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each calendar month present in index."""
    index = pd.DatetimeIndex(index).sort_values()
    key = index.year * 12 + index.month
    first = np.r_[True, key[1:] != key[:-1]]
    return index[first]


# --------------------------------------------------------------------------- engine

@dataclass
class Context:
    """What a policy may know on a decision date: only information from before that day's open."""
    date: pd.Timestamp        # the trading day on which orders will execute at the open
    history: pd.DataFrame     # adjusted closes up to and including the PREVIOUS trading day's close
    values: pd.Series         # dollar value per asset of current holdings, marked at the previous close
    cash: float               # cash available to spend now (uninvested cash + today's contribution)
    contribution: float       # today's deposit
    step: int                 # 0-based index of this decision date


class DCABacktest:
    def __init__(self, slippage_rate: float = 0.0005, commission_rate: float = 0.0,
                 fractional: bool = True, min_order: float = 0.0):
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.fractional = fractional
        self.min_order = min_order

    def _units(self, units: float) -> float:
        return units if self.fractional else float(np.floor(units + 1e-9))

    def run(self, opens: pd.DataFrame, closes: pd.DataFrame, policy, contributions: pd.Series,
            initial_values: dict[str, float] | None = None, initial_cash: float = 0.0) -> pd.DataFrame:
        """Simulate `policy` on the decision dates `contributions.index`.

        Returns a daily ledger from the first decision date onward. `contribution`
        holds the deposit passed in `contributions` (0 on other days);
        `cum_contribution` additionally includes `initial_values` and
        `initial_cash` as a day-0 deposit. `nav` is a time-weighted unit value:
        nav = 1.0 on the first day (day 0 is the anchor, its return is not
        counted). On a later decision day the return is split at the open:
        overnight (previous close -> open) on the old holdings, then
        open -> close on holdings plus that day's deposit, so deposits never earn
        a return they were not exposed to.
        """
        assets = list(opens.columns)
        if list(closes.columns) != assets:
            closes = closes[assets]
        index = opens.index
        contributions = contributions.sort_index().astype(float)
        if (contributions < 0).any():
            raise ValueError('contributions must be non-negative')
        pos = index.get_indexer(contributions.index)
        if (pos < 0).any():
            raise KeyError('every contribution date must be a trading date in opens.index')
        if len(pos) == 0:
            raise ValueError('no decision dates')

        n_assets = len(assets)
        open_arr = opens.to_numpy(dtype=float)
        close_arr = closes.reindex(index).to_numpy(dtype=float)
        close_ff = closes.reindex(index).ffill().to_numpy(dtype=float)

        units = np.zeros(n_assets)
        initial_total = float(initial_cash)
        if initial_values:
            first_open = open_arr[pos[0]]
            for asset, value in initial_values.items():
                j = assets.index(asset)
                if not np.isfinite(first_open[j]) or first_open[j] <= 0:
                    raise ValueError(f'no open price for {asset} on the first date')
                units[j] = value / first_open[j]
                initial_total += value
        cash = float(initial_cash)

        n_dec = len(pos)
        units_rec = np.zeros((n_dec, n_assets))
        cash_rec = np.zeros(n_dec)
        traded = np.zeros(n_dec)
        ntrades = np.zeros(n_dec)
        comm_rec = np.zeros(n_dec)
        pre_open = np.zeros(n_dec)
        flow = np.zeros(n_dec)

        slip, comm = self.slippage_rate, self.commission_rate
        for k, p in enumerate(pos):
            o = open_arr[p]
            tradable = np.isfinite(o) & (o > 0)
            # open price, falling back to the last known close, for valuing untradable holdings
            ref_open = np.where(tradable, o, close_ff[p - 1] if p > 0 else np.nan)
            prev_close = close_ff[p - 1] if p > 0 else np.full(n_assets, np.nan)
            mark = np.where(np.isfinite(prev_close), prev_close, ref_open)
            values = np.where(units > 0, units * np.nan_to_num(mark), 0.0)
            deposit = float(contributions.iloc[k])
            flow[k] = deposit + (initial_total if k == 0 else 0.0)
            pre_open[k] = cash + float(np.sum(np.where(units > 0, units * np.nan_to_num(ref_open), 0.0)))
            cash_avail = cash + deposit

            ctx = Context(
                date=index[p],
                history=closes.iloc[:p],
                values=pd.Series(values, index=assets),
                cash=cash_avail,
                contribution=deposit,
                step=k,
            )
            order = policy(ctx) or {}
            cash = cash_avail
            day_traded = day_comm = 0.0
            day_n = 0

            if 'buy' in order:
                fractions = order['buy']
                if sum(fractions.values()) > 1 + 1e-9:
                    raise ValueError('buy fractions sum to more than 1')
                budget = cash_avail
                for asset, frac in fractions.items():
                    j = assets.index(asset)
                    spend = frac * budget
                    if frac <= 0 or not tradable[j] or spend < max(self.min_order, _EPS):
                        continue
                    price = o[j] * (1 + slip)
                    n_units = self._units(spend / (price * (1 + comm)))
                    if n_units <= 0:
                        continue
                    cost = n_units * price
                    c = cost * comm
                    cash -= cost + c
                    units[j] += n_units
                    day_traded += n_units * o[j]
                    day_comm += c
                    day_n += 1
            elif 'target' in order:
                target = order['target']
                if sum(target.values()) > 1 + 1e-9:
                    raise ValueError('target weights sum to more than 1')
                total = cash + float(np.sum(np.where(units > 0, units * np.nan_to_num(ref_open), 0.0)))
                held_val = np.where(units > 0, units * np.nan_to_num(ref_open), 0.0)
                w = np.array([target.get(a, 0.0) for a in assets], dtype=float)
                desired = total * w
                delta = np.where(tradable, desired - held_val, 0.0)
                # sells first
                for j in np.nonzero(delta < -_EPS)[0]:
                    if w[j] == 0:
                        n_units = units[j]
                    else:
                        n_units = min(units[j], self._units(-delta[j] / o[j]))
                    if n_units <= 0 or n_units * o[j] < max(self.min_order, _EPS):
                        continue
                    price = o[j] * (1 - slip)
                    proceeds = n_units * price
                    c = proceeds * comm
                    cash += proceeds - c
                    units[j] -= n_units
                    day_traded += n_units * o[j]
                    day_comm += c
                    day_n += 1
                # then buys, scaled down if cash falls short (slippage on the sells)
                buys = []
                for j in np.nonzero(delta > _EPS)[0]:
                    n_units = self._units(delta[j] / (o[j] * (1 + slip) * (1 + comm)))
                    if n_units <= 0 or n_units * o[j] < max(self.min_order, _EPS):
                        continue
                    buys.append((j, n_units))
                need = sum(nu * o[j] * (1 + slip) * (1 + comm) for j, nu in buys)
                if need > cash and need > 0:
                    scale = cash / need
                    buys = [(j, self._units(nu * scale)) for j, nu in buys]
                for j, n_units in buys:
                    if n_units <= 0:
                        continue
                    price = o[j] * (1 + slip)
                    cost = n_units * price
                    c = cost * comm
                    cash -= cost + c
                    units[j] += n_units
                    day_traded += n_units * o[j]
                    day_comm += c
                    day_n += 1
            elif order:
                raise ValueError("policy must return {'buy': ...}, {'target': ...} or {}")

            cash = max(cash, 0.0) if cash > -1e-9 else cash
            units_rec[k] = units
            cash_rec[k] = cash
            traded[k] = day_traded
            ntrades[k] = day_n
            comm_rec[k] = day_comm

        # ---- vectorised daily valuation
        ledger_index = index[pos[0]:]
        start = pos[0]
        rows = pos - start
        n = len(ledger_index)
        units_full = np.full((n, n_assets), np.nan)
        units_full[rows] = units_rec
        units_full = pd.DataFrame(units_full).ffill().to_numpy()
        cash_full = np.full(n, np.nan)
        cash_full[rows] = cash_rec
        cash_full = pd.Series(cash_full).ffill().to_numpy()

        px = close_ff[start:]
        val = np.where(units_full > 0, units_full * np.nan_to_num(px), 0.0)
        total_value = cash_full + val.sum(axis=1)

        contrib = np.zeros(n)
        contrib[rows] = contributions.to_numpy()
        init_dep = np.zeros(n)
        init_dep[0] = initial_total
        cum_contrib = np.cumsum(contrib + init_dep)

        def per_day(arr):
            out = np.zeros(n)
            out[rows] = arr
            return out

        # time-weighted nav (see docstring)
        nav_ret = np.zeros(n)
        pre = per_day(pre_open)
        flow_d = per_day(flow)
        is_dec = np.zeros(n, dtype=bool)
        is_dec[rows] = True
        for t in range(1, n):
            vprev = total_value[t - 1]
            over = 1.0
            base = vprev
            if is_dec[t]:
                base = pre[t] + flow_d[t]
                if vprev > 0:
                    over = pre[t] / vprev
            if base > 0:
                nav_ret[t] = over * total_value[t] / base - 1.0
        nav = np.cumprod(1.0 + nav_ret)

        data = {
            'total_value': total_value,
            'cash': cash_full,
            'contribution': contrib,
            'cum_contribution': cum_contrib,
            'traded_value': per_day(traded),
            'n_trades': per_day(ntrades).astype(int),
            'commission': per_day(comm_rec),
            'nav': nav,
        }
        for j, asset in enumerate(assets):
            data[f'value_{asset}'] = val[:, j]
        for j, asset in enumerate(assets):
            data[f'units_{asset}'] = units_full[:, j]
        return pd.DataFrame(data, index=ledger_index)


# --------------------------------------------------------------------------- metrics

def _irr_solve(t: np.ndarray, flows: np.ndarray, final: np.ndarray, horizon: np.ndarray,
               lo: float = -8.0, hi: float = 8.0, iters: int = 100) -> np.ndarray:
    """Vectorised money-weighted annual return by bisection on the log rate g = ln(1 + irr).

    t, flows: (W, K) deposit times in years and amounts; final, horizon: (W,)
    terminal value and its time in years. Solves
    sum(flows * exp(-g t)) = final * exp(-g horizon) for each row. NaN when
    there is no sign change in [lo, hi], except total loss which gives -1.
    """
    def f(g):
        return final * np.exp(-g * horizon) - np.sum(flows * np.exp(-g[:, None] * t), axis=1)

    # Keep |g * horizon| small enough that exp() cannot overflow on long horizons.
    bound = 600.0 / np.maximum(np.asarray(horizon, dtype=float), 1e-9)
    a = np.maximum(np.full(len(final), lo), -bound)
    b = np.minimum(np.full(len(final), hi), bound)
    fa, fb = f(a), f(b)
    ok = (fa > 0) & (fb < 0)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        up = fm > 0
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    g = 0.5 * (a + b)
    irr = np.where(ok, np.exp(g) - 1.0, np.nan)
    irr = np.where(final <= 0, -1.0, irr)
    return irr


def dca_metrics(ledger: pd.DataFrame) -> pd.Series:
    """Summary of a DCA ledger.

    irr is the annualised money-weighted return of the dated deposits (ACT/365)
    against the final total_value. twr_annualized, annualized_volatility,
    max_drawdown and worst_12m_return come from the time-weighted `nav`
    (252 trading days per year; the 12-month window is 252 rows).
    """
    nav = ledger['nav']
    total_contributed = float(ledger['cum_contribution'].iloc[-1])
    final_value = float(ledger['total_value'].iloc[-1])
    days = (ledger.index[-1] - ledger.index[0]).days

    dep = ledger['cum_contribution'].diff()
    dep.iloc[0] = ledger['cum_contribution'].iloc[0]
    dep = dep[dep > 0]
    t = ((dep.index - ledger.index[0]).days.to_numpy(dtype=float) / 365.0)[None, :]
    if days > 0 and len(dep):
        irr = float(_irr_solve(t, dep.to_numpy()[None, :], np.array([final_value]),
                               np.array([days / 365.0]))[0])
    else:
        irr = np.nan

    returns = nav.pct_change().dropna()
    n_ret = len(nav) - 1
    twr = nav.iloc[-1] ** (252 / n_ret) - 1 if n_ret > 0 else np.nan
    drawdown = nav / nav.cummax() - 1
    return pd.Series({
        'total_contributed': total_contributed,
        'final_value': final_value,
        'wealth_multiple': final_value / total_contributed if total_contributed > 0 else np.nan,
        'irr': irr,
        'twr_annualized': twr,
        'annualized_volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan,
        'max_drawdown': drawdown.min(),
        'worst_12m_return': nav.pct_change(252).min() if len(nav) > 252 else np.nan,
        'years': days / 365.25,
    })


# --------------------------------------------------------------------------- policies

def fixed_buy(weights: dict[str, float]):
    """Split each contribution by fixed fractions of the deposit; never sells.

    Weights summing to 1 spend all available cash (leftovers from whole-share
    rounding carry over). Weights summing to < 1 apply to the deposit only; the
    remainder stays idle cash that is never re-spent.
    """
    weights = dict(weights)
    full = sum(weights.values()) >= 1 - 1e-9

    def policy(ctx: Context) -> dict:
        if ctx.cash <= 0:
            return {}
        # fully invested weights spend all cash (so whole-share leftovers carry over);
        # otherwise fractions apply to today's deposit and the idle remainder is not re-spent
        share = 1.0 if full else ctx.contribution / ctx.cash
        return {'buy': {a: w * share for a, w in weights.items()}}
    return policy


def _water_fill(values: dict[str, float], target: dict[str, float], cash: float) -> dict[str, float]:
    """Spend `cash` on the assets furthest below target (equalise value/target-weight).

    Raises the level L = (value + spend) / weight of the lowest-ratio assets
    until the cash is used up or L reaches the full target level (portfolio
    value + cash). Returns dollars per asset.
    """
    cap = sum(values.values()) + cash
    names = [a for a, w in target.items() if w > 0]
    if cash <= 0 or not names:
        return {}
    ratio = {a: values.get(a, 0.0) / target[a] for a in names}
    order = sorted(names, key=lambda a: ratio[a])
    level, active = None, []
    for i, a in enumerate(order):
        active.append(a)
        lvl = (cash + sum(values.get(x, 0.0) for x in active)) / sum(target[x] for x in active)
        nxt = ratio[order[i + 1]] if i + 1 < len(order) else np.inf
        if lvl <= nxt:
            level = lvl
            break
    level = min(level, cap)
    spend = {a: level * target[a] - values.get(a, 0.0) for a in names}
    return {a: s for a, s in spend.items() if s > _EPS}


def buy_underweight(target: dict[str, float]):
    """Contribution-only rebalancing (water-filling toward target weights); never sells.

    Cash goes first to the assets with the lowest value/target ratio. If all
    assets reach their target (weights summing to < 1), the remainder stays cash.
    """
    target = dict(target)

    def policy(ctx: Context) -> dict:
        if ctx.cash <= 0:
            return {}
        spend = _water_fill(ctx.values.to_dict(), target, ctx.cash)
        return {'buy': {a: s / ctx.cash for a, s in spend.items()}}
    return policy


def rebalance_to(target: dict[str, float], every: int = 12, band: float | None = None):
    """Like buy_underweight each month, plus periodic full rebalances.

    Without `band`, a full {'target': ...} rebalance happens on decision steps
    every, 2*every, ... (step % every == 0, step > 0), i.e. after `every`
    contributions. With `band`, `every` is ignored and a full rebalance happens
    whenever any asset's weight (values / (holdings + cash)) is more than
    `band` (absolute) away from target.
    """
    target = dict(target)
    underweight = buy_underweight(target)

    def policy(ctx: Context) -> dict:
        if band is None:
            full = ctx.step > 0 and ctx.step % every == 0
        else:
            total = float(ctx.values.sum()) + ctx.cash
            if total <= 0:
                full = False
            else:
                w = ctx.values / total
                full = any(abs(w.get(a, 0.0) - target.get(a, 0.0)) > band for a in w.index)
        return {'target': target} if full else underweight(ctx)
    return policy


# --------------------------------------------------------------------------- rolling fast path

def rolling_dca(opens, closes, alloc: pd.DataFrame, horizon_months: int, amount: float = 1.0,
                slippage_rate: float = 0.0005) -> pd.DataFrame:
    """Evaluate every full window of `horizon_months` monthly contributions at once.

    `alloc` is indexed by month-start decision dates (rows of `opens.index`),
    one column per asset, rows = fractions of that month's `amount`
    (sum <= 1, remainder idle cash; a fraction on an asset with no open price
    also stays idle). Units bought in a month do not depend on the start date,
    so they are computed once and each window is a difference of cumulative sums.

    Window s covers contributions on alloc rows s .. s+H-1 (H = horizon_months).
    `end` is the decision date at row s+H (not a contribution date). Final
    value is marked at the PREVIOUS trading day's close before `end` (the same
    point DCABacktest's ledger reaches if it is cut just before `end`), with
    idle cash added at face value, no liquidation cost. `irr` is the annualised
    money-weighted return with each deposit at its actual decision date and the
    final value at that previous close (ACT/365). Note: this uses actual dates
    rather than a uniform monthly grid, so it agrees with `dca_metrics`.
    Columns: start, end, total_contributed, final_value, wealth_multiple, irr.
    """
    H = int(horizon_months)
    assets = list(alloc.columns)
    n = len(alloc)
    if n <= H:
        return pd.DataFrame(columns=['start', 'end', 'total_contributed', 'final_value',
                                     'wealth_multiple', 'irr'])
    open_m = opens.reindex(alloc.index)[assets].to_numpy(dtype=float)
    frac = alloc.to_numpy(dtype=float)
    ok = np.isfinite(open_m) & (open_m > 0) & np.isfinite(frac) & (frac > 0)
    spent = np.where(ok, frac * amount, 0.0)
    units = np.where(ok, spent / np.where(ok, open_m, 1.0) / (1 + slippage_rate), 0.0)
    idle = amount - spent.sum(axis=1)

    cum_units = np.vstack([np.zeros(len(assets)), np.cumsum(units, axis=0)])
    cum_idle = np.r_[0.0, np.cumsum(idle)]
    W = n - H
    starts = np.arange(W)
    ends = starts + H

    held = cum_units[ends] - cum_units[starts]
    idle_w = cum_idle[ends] - cum_idle[starts]

    pos_end = opens.index.get_indexer(alloc.index[ends])
    if (pos_end <= 0).any():
        raise KeyError('alloc dates must be trading dates in opens.index (and not the first row)')
    close_ff = closes.reindex(opens.index)[assets].ffill().to_numpy(dtype=float)
    mark_pos = pos_end - 1
    final = np.where(held > 0, held * np.nan_to_num(close_ff[mark_pos]), 0.0).sum(axis=1) + idle_w

    days = ((alloc.index - pd.Timestamp('1970-01-01')).days).to_numpy(dtype=float)
    win = np.lib.stride_tricks.sliding_window_view(days, H)[:W]
    t = (win - days[:W, None]) / 365.0
    mark_days = ((opens.index[mark_pos] - pd.Timestamp('1970-01-01')).days).to_numpy(dtype=float)
    horizon = (mark_days - days[:W]) / 365.0
    flows = np.full((W, H), float(amount))
    irr = _irr_solve(t, flows, final, np.maximum(horizon, 1e-9))

    total = amount * H
    return pd.DataFrame({
        'start': alloc.index[:W],
        'end': alloc.index[ends],
        'total_contributed': total,
        'final_value': final,
        'wealth_multiple': final / total,
        'irr': irr,
    })
