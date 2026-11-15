"""EXP1: stress test of the owner's current portfolio (NVDA 1000, SPCX 1000, BRK-B 2000, VOO 700, other 2300).

Run from repo root:  python scripts/dca/exp1_current_portfolio.py
Outputs: docs/dca/exp1_current_portfolio/*.csv, *.png, findings.md
(findings.md = scripts/dca/exp1_prose.md with {{table:name}} placeholders replaced by the generated tables.)

Method notes
- All prices are adjusted (total-return) closes from data/ohlcv_daily. Everything here is buy-and-hold / weights x returns
  on closes (no engine, no decisions, so no lookahead issue). Monthly rebalance in D resets weights on the first
  trading day of each month (i.e. at the prior close); no slippage/commission modelled in D.
- Spliced return series (earlier source used only where the later one has no return):
    SP   = VOO (2010-09+) -> SPY (1993-02+) -> VFINX (1980+)
    BRK  = BRK-B (1996-05+) -> BRK-A (same company, earlier)
    INTL = VXUS (2011-01+) -> EFA (2001-08+)       [NaN before 2001-08; portfolios reallocate that weight to SP]
    BOND = AGG (2003-09+) -> VBMFX
- SPCX proxies: TSLA (2010-07+), ARKK (2014-11+), SYN = max(beta,1.5) x QQQ daily return (1999-03+).
- CASH earns 0%.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DATA_DIR

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca', 'exp1_current_portfolio')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
ASOF = pd.Timestamp('2026-11-13')
TABLES = {}


def save(name, df, index=True, nd=4):
    TABLES[name] = df.copy()
    df.round(nd).to_csv(os.path.join(OUT, name + '.csv'), index=index)


def md(df, index=True, nd=3):
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append('n/a' if np.isnan(v) else f'{v:.{nd}f}')
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime('%Y-%m-%d'))
            elif v is pd.NaT:
                cells.append('n/a')
            else:
                cells.append(str(v))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] + rows)


def pct(df):
    return (df * 100)


# ------------------------------------------------------------------ data
def load_close(t):
    name = ('_' + t[1:]) if t.startswith('^') else t
    return pd.read_csv(DATA_DIR / f'{name}.csv', index_col='date', parse_dates=True)['close']


RAW = {t: load_close(t) for t in ['SPY', 'VOO', 'VFINX', 'QQQ', 'BRK-B', 'BRK-A', 'NVDA', 'TSLA', 'ARKK', 'SPCX', 'VXUS',
                                  'EFA', 'SMH', 'BND', 'AGG', 'VBMFX', 'GLD']}
CAL = RAW['VFINX'].index.union(RAW['SPY'].index).union(RAW['BRK-B'].index)
CAL = CAL[CAL <= ASOF]


def ret(t):
    s = RAW[t].reindex(CAL)
    first, last = RAW[t].index[0], RAW[t].index[-1]
    s = s.ffill()
    s[(s.index < first)] = np.nan
    return s.pct_change(fill_method=None)


R = pd.DataFrame({t: ret(t) for t in RAW})
SPCX_R = RAW['SPCX'].pct_change().dropna()   # trading days of its own (first return = day 2)
ret_df = pd.DataFrame(index=CAL)
ret_df['SP'] = R['VOO'].combine_first(R['SPY']).combine_first(R['VFINX'])
ret_df['SPY'] = R['SPY']
ret_df['QQQ'] = R['QQQ']
ret_df['BRK'] = R['BRK-B'].combine_first(R['BRK-A'])
ret_df['NVDA'] = R['NVDA']
ret_df['TSLA'] = R['TSLA']
ret_df['ARKK'] = R['ARKK']
ret_df['INTL'] = R['VXUS'].combine_first(R['EFA'])
ret_df['BOND'] = R['AGG'].combine_first(R['VBMFX'])
ret_df['GLD'] = R['GLD']
ret_df['SPCX'] = SPCX_R.reindex(CAL)
ret_df['CASH'] = 0.0

# ------------------------------------------------------------------ step 1: SPCX stats
ov = pd.concat([SPCX_R.rename('SPCX'), ret_df[['QQQ', 'SPY', 'NVDA', 'TSLA', 'ARKK']]], axis=1, join='inner').dropna()
rows = []
for m in ['QQQ', 'SPY', 'NVDA', 'TSLA', 'ARKK']:
    x, y = ov[m], ov['SPCX']
    beta = np.cov(y, x)[0, 1] / x.var()
    resid = y - y.mean() - beta * (x - x.mean())
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    rows.append({'vs': m, 'beta': beta, 'beta_se': se, 'corr': y.corr(x), 'R2': y.corr(x) ** 2,
                 'peer_ann_vol_same_days': x.std() * np.sqrt(252)})
spcx_stats = pd.DataFrame(rows).set_index('vs')
N_SPCX = len(ov)
SPCX_VOL = ov['SPCX'].std() * np.sqrt(252)
BETA_Q = spcx_stats.loc['QQQ', 'beta']
BETA_USED = max(BETA_Q, 1.5)
print(f'SPCX: {N_SPCX} daily returns, ann vol {SPCX_VOL:.3f}, beta to QQQ {BETA_Q:.2f} -> synthetic beta {BETA_USED:.2f}')
save('spcx_stats', spcx_stats, nd=3)
spcx_summary = pd.DataFrame({'value': {
    'first_date': str(RAW['SPCX'].index[0].date()), 'last_date': str(RAW['SPCX'].index[-1].date()),
    'daily_returns_used': N_SPCX, 'ann_vol': round(SPCX_VOL, 3),
    'total_return_first_close_to_last': round(RAW['SPCX'].iloc[-1] / RAW['SPCX'].iloc[0] - 1, 3),
    'max_drawdown_from_peak_close': round((RAW['SPCX'] / RAW['SPCX'].cummax() - 1).min(), 3),
    'beta_QQQ_estimate': round(BETA_Q, 3), 'beta_used_for_SYN': round(BETA_USED, 3)}})
save('spcx_summary', spcx_summary, nd=3)
ret_df['SYN'] = BETA_USED * ret_df['QQQ']
ret_df['VOLCHK'] = 0.0
LV = (1 + ret_df.drop(columns=['VOLCHK'])).cumprod()      # levels, NaN preserved before first valid return
LV['CASH'] = 1.0

# ------------------------------------------------------------------ A: market state
tick = ['SPY', 'QQQ', 'BRK-B', 'NVDA', 'SPCX', 'VXUS', 'SMH', 'BND', 'GLD']


def at_or_before(s, d):
    s2 = s.loc[:d]
    return s2.iloc[-1] if len(s2) and s2.index[0] <= d else np.nan


rows = []
for t in tick:
    s = RAW[t]
    last = s.iloc[-1]
    row = {'ticker': t, 'first_date': s.index[0].date()}
    for lab, mo in [('1m', 1), ('3m', 3), ('6m', 6), ('12m', 12)]:
        d = ASOF - pd.DateOffset(months=mo)
        row[lab] = last / at_or_before(s, d) - 1 if s.index[0] <= d else np.nan
    ye = pd.Timestamp('2025-12-31')
    row['YTD'] = last / at_or_before(s, ye) - 1 if s.index[0] <= ye else np.nan
    for lab, yr in [('3y_ann', 3), ('5y_ann', 5)]:
        d = ASOF - pd.DateOffset(years=yr)
        row[lab] = (last / at_or_before(s, d)) ** (1 / yr) - 1 if s.index[0] <= d else np.nan
    row['since_start'] = last / s.iloc[0] - 1
    ath = s.max()
    row['ATH_date'] = s.idxmax().date()
    row['dist_from_ATH'] = last / ath - 1
    hi52 = s.loc[ASOF - pd.DateOffset(months=12):].max()
    row['dist_from_52w_high'] = last / hi52 - 1
    rows.append(row)
mstate = pd.DataFrame(rows).set_index('ticker')
save('A_market_state', mstate)

# relative 12m return BRK-B vs SPY / QQQ and percentile (252 trading-day returns)
brk, spy, qqq = RAW['BRK-B'], RAW['SPY'], RAW['QQQ']
r12 = lambda s: s / s.shift(252) - 1
rel = pd.concat([r12(brk), r12(spy), r12(qqq)], axis=1, keys=['BRK', 'SPY', 'QQQ'])
rel['BRK_minus_SPY'] = rel['BRK'] - rel['SPY']
rel['BRK_minus_QQQ'] = rel['BRK'] - rel['QQQ']
rows = []
for c, startnote in [('BRK_minus_SPY', rel['BRK_minus_SPY'].dropna().index[0]), ('BRK_minus_QQQ', rel['BRK_minus_QQQ'].dropna().index[0])]:
    s = rel[c].dropna()
    cur = s.iloc[-1]
    rows.append({'series': c, 'current_12m_diff': cur, 'percentile_in_own_history': (s < cur).mean() * 100,
                 'hist_min': s.min(), 'hist_median': s.median(), 'hist_max': s.max(),
                 'n_days': len(s), 'history_from': s.index[0].date(), 'overlapping_12m_windows_note': 'daily overlapping'})
rel_tbl = pd.DataFrame(rows).set_index('series')
rel_tbl['current_BRK_12m'] = rel['BRK'].iloc[-1]
rel_tbl['current_SPY_12m'] = rel['SPY'].iloc[-1]
rel_tbl['current_QQQ_12m'] = rel['QQQ'].iloc[-1]
save('A_relative_12m', rel_tbl)


# episodes
def find_episodes(px, thr):
    px = px.dropna()
    peak = px.cummax()
    dd = px / peak - 1
    hi = np.flatnonzero((dd == 0).values)
    n = len(px)
    out = []
    for a, b in zip(hi, list(hi[1:]) + [n]):
        if b - a <= 1:
            continue
        seg = dd.iloc[a:b]
        if seg.min() <= -thr:
            tr = seg.idxmin()
            out.append({'peak_date': px.index[a], 'trough_date': tr, 'depth': seg.min(),
                        'recovery_date': px.index[b] if b < n else pd.NaT,
                        'peak_to_trough_days': (tr - px.index[a]).days,
                        'trough_to_recovery_days': (px.index[b] - tr).days if b < n else np.nan})
    return pd.DataFrame(out)


sp_ep = find_episodes(LV['SP'].dropna() if False else (1 + ret_df['SP'].fillna(0)).cumprod(), 0.10)
sp_ep.insert(0, 'series', 'S&P (VFINX->SPY->VOO)')
qq_ep = find_episodes(RAW['QQQ'], 0.15)
qq_ep.insert(0, 'series', 'QQQ')
spy_only = find_episodes(RAW['SPY'], 0.10)
spy_only.insert(0, 'series', 'SPY only')
save('A_episodes_SP_10pct_since1980', sp_ep.set_index('series'))
save('A_episodes_QQQ_15pct', qq_ep.set_index('series'))
save('A_episodes_SPYonly_10pct', spy_only.set_index('series'))
cur_dd = {t: RAW[t].iloc[-1] / RAW[t].max() - 1 for t in ['SPY', 'QQQ']}

# ------------------------------------------------------------------ portfolio definitions
CUR = {'SP': 700, 'BRK': 2000, 'NVDA': 1000, 'SPCX': 1000}
PORT = {
    'cur_a_cash': {**CUR, 'CASH': 2300},
    'cur_b_qqq': {**CUR, 'QQQ': 2300},
    'cur_c_spy': {**CUR, 'SP': 3000},
    '100_VOO/SPY': {'SP': 1},
    '100_BRK-B': {'BRK': 1},
    '100_QQQ': {'QQQ': 1},
    '50/50_SPY/BRK': {'SP': .5, 'BRK': .5},
    '60/40_SPY/BRK': {'SP': .6, 'BRK': .4},
    'target_80/20': {'SP': .45, 'BRK': .25, 'INTL': .10, 'NVDA': .10, 'SPCX': .10},
    '60/40_SPY/bonds': {'SP': .6, 'BOND': .4},
}
PORT = {k: {a: w / sum(v.values()) for a, w in v.items()} for k, v in PORT.items()}   # normalise to weights
HAS_SPCX = {k: 'SPCX' in v for k, v in PORT.items()}
PROXIES = ['TSLA', 'ARKK', 'SYN']
SHORT = {'cur_a_cash': 'cur(a) cash', 'cur_b_qqq': 'cur(b) QQQ', 'cur_c_spy': 'cur(c) SPY', '100_VOO/SPY': '100% SPY',
         '100_BRK-B': '100% BRK-B', '100_QQQ': '100% QQQ', '50/50_SPY/BRK': '50/50', '60/40_SPY/BRK': '60/40 SPY/BRK',
         'target_80/20': 'target 80/20', '60/40_SPY/bonds': '60/40 SPY/bond'}


def resolve(w, proxy, date):
    """Map weights to level columns on `date`. INTL missing -> SP. Any other missing asset -> None."""
    out = {}
    for a, x in w.items():
        col = proxy if a == 'SPCX' else a
        if col is None or pd.isna(LV[col].get(date, np.nan)):
            if a == 'INTL':
                out['SP'] = out.get('SP', 0) + x
                continue
            return None
        out[col] = out.get(col, 0) + x
    return out


def bh_stats(w, proxy, d0, d1, d_end):
    """buy&hold from d0 close: return to d1 (trough), max drawdown within [d0, d_end] (d_end=recovery or last date)."""
    ww = resolve(w, proxy, d0)
    if ww is None:
        return np.nan, np.nan
    sub = LV.loc[d0:d_end, list(ww)]
    if sub.isna().any().any():
        return np.nan, np.nan
    v = (sub / sub.iloc[0] * pd.Series(ww)).sum(axis=1)
    return v.loc[d1] - 1, (v / v.cummax() - 1).min()


# ------------------------------------------------------------------ B: episodes
eps = pd.concat([sp_ep.assign(list='SP>=10%'), qq_ep.assign(list='QQQ>=15%')], ignore_index=True)
eps['end'] = eps['recovery_date'].fillna(ASOF)
rows = []
for _, e in eps.iterrows():
    for k, w in PORT.items():
        for px in (PROXIES if HAS_SPCX[k] else [None]):
            r1, m1 = bh_stats(w, px, e.peak_date, e.trough_date, e.end)
            rows.append({'list': e['list'], 'peak': e.peak_date.date(), 'trough': e.trough_date.date(),
                         'recovery': (e.recovery_date.date() if pd.notna(e.recovery_date) else 'not recovered'),
                         'portfolio': k, 'spcx_proxy': px or '-', 'ret_peak_to_trough': r1, 'maxdd_peak_to_recovery': m1})
epp = pd.DataFrame(rows)
save('B_episode_portfolios_long', epp, index=False)
# per-asset
rows = []
for _, e in eps.iterrows():
    row = {'list': e['list'], 'peak': e.peak_date.date(), 'trough': e.trough_date.date()}
    for a in ['SP', 'QQQ', 'BRK', 'NVDA', 'TSLA', 'ARKK', 'SYN', 'INTL', 'BOND', 'GLD']:
        p0, p1 = LV[a].get(e.peak_date, np.nan), LV[a].get(e.trough_date, np.nan)
        row[a] = p1 / p0 - 1 if pd.notna(p0) and pd.notna(p1) else np.nan
    rows.append(row)
ep_assets = pd.DataFrame(rows)
save('B_episode_asset_returns', ep_assets, index=False)
ep_assets['depth_chk'] = eps['depth'].values


# pivots for the write-up
epp['episode'] = epp['list'].str[:2] + ' ' + epp.peak.astype(str) + ' > ' + epp.trough.astype(str)
order = [PORT_ for PORT_ in PORT]
np_ = epp[epp.spcx_proxy == '-']
for nm, val in [('ret_peak_to_trough', 'ret_peak_to_trough'), ('maxdd_peak_to_recovery', 'maxdd_peak_to_recovery')]:
    pv = np_.pivot_table(index='episode', columns='portfolio', values=val, aggfunc='first', dropna=False)
    cols = [c for c in order if c in pv.columns]
    pv = pv[cols].loc[epp.drop_duplicates('episode').episode]
    pv.columns = [SHORT[c] for c in pv.columns]
    save('B_noSPCX_' + nm, pv)
    for px in PROXIES:
        q = epp[(epp.spcx_proxy == px)]
        pv2 = q.pivot_table(index='episode', columns='portfolio', values=val, aggfunc='first', dropna=False)
        pv2 = pv2[[c for c in order if c in pv2.columns]].loc[[e for e in epp.drop_duplicates('episode').episode if e in pv2.index]]
        pv2.columns = [SHORT[c] for c in pv2.columns]
        pv2 = pv2.dropna(how='all')
        save(f'B_SPCX_{px}_' + nm, pv2)

# ------------------------------------------------------------------ C: risk decomposition
COLS = ['SP', 'QQQ', 'BRK', 'NVDA', 'SPCX', 'INTL', 'BOND', 'GLD']
MKT = ['SPY', 'QQQ']
last_dates = CAL
windows = {'1y': ret_df.loc[CAL[-252]:], '3y': ret_df.loc[CAL[-756]:], 'spcx_overlap': ret_df.loc[RAW['SPCX'].index[1]:]}
wts_C = {k: PORT[k] for k in ['cur_a_cash', 'cur_b_qqq', 'cur_c_spy', 'target_80/20', '50/50_SPY/BRK', '100_VOO/SPY', '100_QQQ',
                              '100_BRK-B', '60/40_SPY/BRK', '60/40_SPY/bonds']}
risk_rows, rc_rows, corr_out = [], [], {}
for wn, W in windows.items():
    cov = W[COLS + ['SPY']].cov(min_periods=20) * 252
    eig = np.linalg.eigvalsh(cov.fillna(0).values).min()
    print(f'cov window {wn}: n={len(W)} min eigenvalue {eig:.2e}')
    corr_out[wn] = W[COLS].corr(min_periods=20)
    save(f'C_corr_{wn}', corr_out[wn], nd=2)
    for k, w in wts_C.items():
        wv = pd.Series({c: w.get(c, 0.0) for c in COLS})
        if wn == 'spcx_overlap' and not HAS_SPCX[k]:
            pass
        S = cov.loc[COLS, COLS]
        var = float(wv @ S @ wv)
        cov_spy = float(wv @ cov.loc[COLS, 'SPY'])
        cov_q = float(wv @ cov.loc[COLS, 'QQQ'])
        risk_rows.append({'window': wn, 'n_days': len(W), 'portfolio': k, 'ann_vol': np.sqrt(var),
                          'beta_SPY': cov_spy / cov.loc['SPY', 'SPY'] if 'SPY' in cov.index else np.nan,
                          'beta_QQQ': cov_q / cov.loc['QQQ', 'QQQ'],
                          'uses_SPCX_overlap_cov': HAS_SPCX[k]})
        rc = wv * (S @ wv) / var
        d = {'window': wn, 'portfolio': k}
        d.update(rc.round(4).to_dict())
        rc_rows.append(d)
risk = pd.DataFrame(risk_rows)
# fix beta_SPY (SPY row exists as an extra column in cov)
save('C_risk', risk, index=False)
rcdf = pd.DataFrame(rc_rows)
save('C_risk_contribution', rcdf, index=False)
# market-vol sanity
mkt_vol = {wn: {m: W[m].std() * np.sqrt(252) for m in ['SPY', 'QQQ', 'NVDA', 'BRK', 'SPCX']} for wn, W in windows.items()}
save('C_asset_vols', pd.DataFrame(mkt_vol).T, nd=3)

# rolling SPY on QQQ (3y)
pair = ret_df[['SPY', 'QQQ']].dropna()
win = 756
rcov = pair['SPY'].rolling(win).cov(pair['QQQ'])
rbeta = rcov / pair['QQQ'].rolling(win).var()
rr2 = pair['SPY'].rolling(win).corr(pair['QQQ']) ** 2
roll = pd.DataFrame({'beta_SPY_on_QQQ': rbeta, 'R2': rr2}).dropna()
roll = roll.loc['2002-01-01':]
sel = ['2002-12-31', '2005-12-30', '2008-12-31', '2010-12-31', '2013-12-31', '2016-12-30', '2019-12-31', '2022-12-30',
       '2024-12-31', '2025-12-31', ASOF.strftime('%Y-%m-%d')]
rs = pd.DataFrame([{'date': roll.index[roll.index.get_indexer([pd.Timestamp(d)], method='ffill')[0]].date(),
                    **roll.loc[roll.index[roll.index.get_indexer([pd.Timestamp(d)], method='ffill')[0]]].to_dict()} for d in sel])
save('C_rolling_spy_on_qqq_samples', rs, index=False)
roll_sum = pd.DataFrame({'min': roll.min(), 'min_date': roll.idxmin().dt.date, 'max': roll.max(), 'max_date': roll.idxmax().dt.date,
                         'median': roll.median(), 'latest': roll.iloc[-1]})
save('C_rolling_spy_on_qqq_summary', roll_sum)
roll.round(4).to_csv(os.path.join(OUT, 'C_rolling_spy_on_qqq_daily.csv'))
# whole-sample and 1y/3y regression too
reg = []
for wn in ['1y', '3y']:
    W = windows[wn][['SPY', 'QQQ']].dropna()
    b = W['SPY'].cov(W['QQQ']) / W['QQQ'].var()
    reg.append({'window': wn, 'beta_SPY_on_QQQ': b, 'R2': W['SPY'].corr(W['QQQ']) ** 2, 'beta_QQQ_on_SPY': W['SPY'].cov(W['QQQ']) / W['SPY'].var()})
save('C_spy_on_qqq_trailing', pd.DataFrame(reg), index=False)


# ------------------------------------------------------------------ D: full-history simulation (monthly rebalance)
def month_first(idx):
    key = idx.year * 12 + idx.month
    return np.r_[True, key[1:] != key[:-1]]


def simulate(w, proxy, start, rets):
    r = rets.loc[start:].copy()
    mf = month_first(r.index)
    cols = list(r.columns)
    arr = np.nan_to_num(r.to_numpy(float))
    avail = r.notna().to_numpy()
    ci = {c: i for i, c in enumerate(cols)}
    val = np.zeros(len(cols))
    nav = np.empty(len(r))
    total = 1.0
    for t in range(len(r)):
        if mf[t]:
            ww = {}
            for a, x in w.items():
                c = proxy if a == 'SPCX' else a
                if not avail[t, ci[c]] and not (t == 0):
                    pass
                ok = r[c].iloc[t] == r[c].iloc[t] or c == 'CASH'
                if not ok and a == 'INTL':
                    ww['SP'] = ww.get('SP', 0) + x
                elif not ok:
                    ww = None
                    break
                else:
                    ww[c] = ww.get(c, 0) + x
            if ww is None:
                nav[t] = np.nan
                continue
            val = np.zeros(len(cols))
            for c, x in ww.items():
                val[ci[c]] = total * x
        val = val * (1 + arr[t])
        total = val.sum()
        nav[t] = total
    return pd.Series(nav, index=r.index).dropna()


SAMPLES = {'SYN_1999-03+': ('SYN', '1999-03-12'), 'TSLA_2010-07+': ('TSLA', '2010-07-01'), 'ARKK_2014-11+': ('ARKK', '2014-11-03')}
sim_rows, NAVS = [], {}
for sn, (px, start) in SAMPLES.items():
    for k, w in PORT.items():
        rets = ret_df[[c for c in ['SP', 'QQQ', 'BRK', 'NVDA', 'INTL', 'BOND', 'CASH', 'GLD', px]]]
        # skip rows before every non-INTL asset has data
        need = [(px if a == 'SPCX' else a) for a in w if a != 'INTL']
        first = rets[need].dropna().index[0] if need else rets.index[0]
        s = max(pd.Timestamp(start), first)
        nav = simulate(w, px, s, rets)
        NAVS[(sn, k)] = nav
        dd = nav / nav.cummax() - 1
        dr = nav.pct_change().dropna()
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        sim_rows.append({'sample': sn, 'portfolio': k, 'start': nav.index[0].date(), 'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
                         'ann_vol': dr.std() * np.sqrt(252), 'max_drawdown': dd.min(), 'maxdd_date': dd.idxmin().date(),
                         'worst_1m': (nav / nav.shift(21) - 1).min(), 'worst_3m': (nav / nav.shift(63) - 1).min(),
                         'worst_12m': (nav / nav.shift(252) - 1).min(), 'end_multiple': nav.iloc[-1] / nav.iloc[0]})
sim = pd.DataFrame(sim_rows)
save('D_full_history', sim, index=False)
for m in ['CAGR', 'max_drawdown', 'worst_12m']:
    save('D_piv_' + m, sim.pivot_table(index='portfolio', columns='sample', values=m).loc[list(PORT)])

# ------------------------------------------------------------------ E: shocks
def match(year_lo, year_hi):
    m = sp_ep[(sp_ep.peak_date.dt.year >= year_lo) & (sp_ep.peak_date.dt.year <= year_hi)]
    return m.sort_values('depth').iloc[0]


SHOCKS = {'dot-com': match(2000, 2000), '2008': match(2007, 2008), '2022': match(2022, 2022), '2025': match(2025, 2025)}
shock_dates = pd.DataFrame({k: {'peak': v.peak_date.date(), 'trough': v.trough_date.date(), 'SP_depth': v.depth} for k, v in SHOCKS.items()}).T
shock_dates['SP_depth'] = shock_dates['SP_depth'].astype(float)
save('E_shock_episodes', shock_dates)
TOTAL = 7000.0
rows = []
for sk, e in SHOCKS.items():
    for k in ['cur_a_cash', 'cur_b_qqq', 'cur_c_spy', 'target_80/20', '50/50_SPY/BRK']:
        w = PORT[k]
        for px in (PROXIES + ['flat -50%'] if HAS_SPCX[k] else [None]):
            ww = {}
            ok = True
            for a, x in w.items():
                if a == 'SPCX' and px == 'flat -50%':
                    ww['FLAT'] = x
                    continue
                c = px if a == 'SPCX' else a
                if pd.isna(LV[c].get(e.peak_date, np.nan)):
                    if a == 'INTL':
                        ww['SP'] = ww.get('SP', 0) + x
                        continue
                    ok = False
                    break
                ww[c] = ww.get(c, 0) + x
            if not ok:
                pl = np.nan
            else:
                pl = sum(x * (-0.5 if c == 'FLAT' else LV[c].loc[e.trough_date] / LV[c].loc[e.peak_date] - 1) for c, x in ww.items())
            rows.append({'shock': sk, 'portfolio': k, 'spcx_proxy': px or '-', 'pct_loss': pl, 'dollar_loss': pl * TOTAL})
shock = pd.DataFrame(rows)
save('E_shock_long', shock, index=False)
shock['cell'] = shock.apply(lambda r: 'n/a' if pd.isna(r.pct_loss) else f'{r.dollar_loss:,.0f} ({r.pct_loss * 100:.1f}%)', axis=1)
shock_piv = shock.pivot_table(index=['portfolio', 'spcx_proxy'], columns='shock', values='cell', aggfunc='first')[list(SHOCKS)]
shock_piv.to_csv(os.path.join(OUT, 'E_shock_table.csv'))
TABLES['E_shock_table'] = shock_piv

# ------------------------------------------------------------------ figures
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.grid': True, 'grid.alpha': .3})
PLOT = ['cur_a_cash', 'cur_b_qqq', 'cur_c_spy', '100_VOO/SPY', '100_BRK-B', '100_QQQ', 'target_80/20', '50/50_SPY/BRK']
COL = dict(zip(PLOT, ['#d62728', '#ff7f0e', '#e377c2', '#1f77b4', '#2ca02c', '#9467bd', '#000000', '#8c564b']))
for sn, fn in [('TSLA_2010-07+', 'fig1_drawdowns_2010plus.png'), ('SYN_1999-03+', 'fig2_drawdowns_1999plus.png')]:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in PLOT:
        nav = NAVS[(sn, k)]
        ax.plot(nav.index, (nav / nav.cummax() - 1) * 100, label=SHORT[k], color=COL[k], lw=1.1, ls='--' if k.startswith('cur') else '-')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'Drawdowns, monthly-rebalanced fixed weights ({sn}; SPCX proxy = {SAMPLES[sn][0]}; cash earns 0%)')
    ax.legend(ncol=4, fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fn))
    plt.close(fig)

# episode bars: SP episodes since 1999, SYN proxy
sub = epp[(epp['list'] == 'SP>=10%') & (pd.to_datetime(epp.peak) >= '1999-03-01') & (epp.spcx_proxy.isin(['-', 'SYN']))]
sub = sub[sub.portfolio.isin(PLOT)]
lab = sub.drop_duplicates(['peak', 'trough'])[['peak', 'trough']]
fig, ax = plt.subplots(figsize=(12.5, 5.5))
nP = len(PLOT)
for j, k in enumerate(PLOT):
    vals = [sub[(sub.peak == p) & (sub.trough == t) & (sub.portfolio == k)].ret_peak_to_trough.iloc[0] * 100 for p, t in zip(lab.peak, lab.trough)]
    ax.bar(np.arange(len(lab)) + (j - nP / 2) * 0.1, vals, width=0.1, label=SHORT[k], color=COL[k])
ax.set_xticks(np.arange(len(lab)))
ax.set_xticklabels([f'{p}\n-> {t}' for p, t in zip(lab.peak, lab.trough)], fontsize=7)
ax.set_ylabel('Buy-and-hold return peak -> trough (%)')
ax.set_title('S&P drawdown episodes >=10% since 1999: portfolio returns (SPCX proxy = synthetic beta x QQQ; no rebalancing)')
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_episode_bars.png'))
plt.close(fig)

# risk contributions
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, wn in zip(axs, ['1y', '3y']):
    d = rcdf[(rcdf.window == wn) & rcdf.portfolio.isin(['cur_a_cash', 'cur_b_qqq', 'cur_c_spy', 'target_80/20'])].set_index('portfolio')[COLS] * 100
    d = d.loc[:, (d.abs().sum() > 0)]
    d.index = [SHORT[i] for i in d.index]
    d.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', width=.7)
    ax.set_title(f'Share of portfolio variance, trailing {wn} (SPCX cov from {N_SPCX}-day overlap)')
    ax.set_ylabel('% of variance')
    ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_risk_contribution.png'))
plt.close(fig)

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(roll.index, roll.beta_SPY_on_QQQ, color='#1f77b4')
axs[0].set_ylabel('Beta of SPY on QQQ')
axs[0].set_title('Rolling 3-year (756-day) regression of SPY daily returns on QQQ daily returns')
axs[1].plot(roll.index, roll.R2, color='#d62728')
axs[1].set_ylabel('R-squared')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_rolling_spy_on_qqq.png'))
plt.close(fig)

# ------------------------------------------------------------------ findings assembly
prose_path = os.path.join(os.path.dirname(__file__), 'exp1_prose.md')
if os.path.exists(prose_path):
    txt = open(prose_path).read()

    def rep(m):
        name, *fmt = m.group(1).split('|')
        df = TABLES[name]
        nd = 3
        if 'pct' in fmt:
            df = df.copy()
            for c in df.columns:
                if df[c].dtype.kind == 'f' and not any(s in c for s in ['beta', 'corr', 'R2', 'multiple', 'sample']):
                    df[c] = df[c] * 100
            nd = 1
        if 'nd2' in fmt:
            nd = 2
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == 'f' and any(t in str(c) for t in ['beta', 'R2', 'multiple']):
                df[c] = df[c].map(lambda v: 'n/a' if pd.isna(v) else f'{v:.2f}')
        idx = not (len(fmt) and 'noindex' in fmt)
        return md(df, index=idx, nd=nd)
    txt = re.sub(r'\{\{table:([^}]+)\}\}', rep, txt)
    open(os.path.join(OUT, 'findings.md'), 'w').write(txt)
print('done; outputs in', OUT)
