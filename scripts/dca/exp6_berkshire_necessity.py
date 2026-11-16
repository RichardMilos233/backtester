"""EXP6: is Berkshire needed in the core, or is a pure S&P 500 core enough?

A  factor decomposition of BRK-A (FF3 / FF5+Mom, eras, rolling 10y alpha/beta)
B  zero-alpha factor replicas + simple style proxies, extended back to 1926/1963
C  century rolling DCA (and non-overlapping windows) of market vs market/sleeve mixes
D  stationary block bootstrap of VFINX + BRK-A (six scenarios), 10/20/30y monthly DCA
E  Singapore withholding tax tilt (SPY_NET30 / SPY_NET15 vs BRK-B)

Run from the repo root:  python scripts/dca/exp6_berkshire_necessity.py
Outputs: docs/dca/exp6_berkshire_necessity/
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DATA_DIR
from src.dca import load_prices, month_starts, rolling_dca

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'dca',
                                   'exp6_berkshire_necessity'))
os.makedirs(OUT, exist_ok=True)
FR = os.path.abspath(os.path.join(DATA_DIR, '..', 'french'))
SLIP = 0.0005
SEED = 20261116
T0 = time.time()
MD = []   # markdown fragments for tables.md


def md_table(df, floatfmt='{:.2f}', index=True):
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('' if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def emit(title, df, floatfmt='{:.2f}', index=True, note=None):
    MD.append(f'### {title}\n')
    if note:
        MD.append(note + '\n')
    MD.append(md_table(df, floatfmt, index) + '\n')


def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


def save(df, name, nd=4):
    df.round(nd).to_csv(os.path.join(OUT, name))


# ------------------------------------------------------------------ data helpers
def mret(close):
    """Month-end to month-end simple returns indexed by Period('M')."""
    m = close.resample('ME').last().dropna()
    m.index = m.index.to_period('M')
    return m.pct_change().dropna()


def read_factors(name):
    d = pd.read_csv(os.path.join(FR, name), index_col='date', parse_dates=True)
    return d


def fmonthly(name):
    d = read_factors(name)
    d.index = d.index.to_period('M')
    return d


def ols_hac(y, X, lags=6):
    """OLS with intercept; Newey-West (Bartlett) HAC standard errors. X: (T,k). Returns dict."""
    T = len(y)
    Z = np.column_stack([np.ones(T), X])
    XtXi = np.linalg.inv(Z.T @ Z)
    b = XtXi @ Z.T @ y
    u = y - Z @ b
    g = Z * u[:, None]
    S = g.T @ g
    for l in range(1, lags + 1):
        G = g[l:].T @ g[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u @ u) / (((y - y.mean()) ** 2).sum())
    return dict(b=b, se=se, t=b / se, r2=r2, n=T, resid=u)


# =============================================================== A. factor decomposition
log('loading')
px = load_prices(['VFINX', 'BRK-A'], how='outer')['close']
brk_m = mret(px['BRK-A'].dropna())
spx_m = mret(px['VFINX'].dropna())

ff5 = fmonthly('ff5_monthly.csv')
ff3 = fmonthly('ff3_monthly.csv')
mom = fmonthly('mom_monthly.csv')
F6 = ff5.join(mom, how='inner')                       # Mkt-RF SMB HML RMW CMA RF Mom
RF_M = ff3['RF']
LAST = pd.Period('2026-09', 'M')                       # last month of the factor files

d = pd.DataFrame({'brk': brk_m}).join(F6, how='inner').loc[:LAST]
d['ex'] = d['brk'] - d['RF']
d = d.loc[pd.Period('1980-04', 'M'):]
log(f'regression sample {d.index[0]}..{d.index[-1]} n={len(d)}')

MODELS = {
    'CAPM': ['Mkt-RF'],
    'FF3': ['Mkt-RF', 'SMB', 'HML'],
    'FF5+Mom': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'],
}
ERAS = {
    '1980-2026': ('1980-04', '2026-09'),
    '1980-1999': ('1980-04', '1999-12'),
    '2000-2012': ('2000-01', '2012-12'),
    '2013-2026': ('2013-01', '2026-09'),
    '2000-2026': ('2000-01', '2026-09'),
}
rows = []
for era, (a, b) in ERAS.items():
    sub = d.loc[pd.Period(a, 'M'):pd.Period(b, 'M')]
    for mname, fac in MODELS.items():
        r = ols_hac(sub['ex'].to_numpy(), sub[fac].to_numpy())
        row = dict(era=era, model=mname, n=r['n'], alpha_pp_yr=r['b'][0] * 12 * 100,
                   alpha_t=r['t'][0], alpha_ci95_lo=(r['b'][0] - 1.96 * r['se'][0]) * 1200,
                   alpha_ci95_hi=(r['b'][0] + 1.96 * r['se'][0]) * 1200, R2=r['r2'],
                   mean_excess_ann_pp=sub['ex'].mean() * 1200)
        for k, f in enumerate(fac):
            row[f'b_{f}'] = r['b'][k + 1]
            row[f't_{f}'] = r['t'][k + 1]
        rows.append(row)
regs = pd.DataFrame(rows)
save(regs, 'A_regressions.csv')
show = regs[['era', 'model', 'n', 'alpha_pp_yr', 'alpha_t', 'alpha_ci95_lo', 'alpha_ci95_hi', 'R2']]
emit('A1. BRK-A excess return alpha (annualised, pp), Newey-West t (6 lags), 95% CI', show.set_index(['era', 'model']),
     note='y = BRK-A monthly return minus RF; alpha x 12.')
loadcols = ['era', 'model', 'b_Mkt-RF', 'b_SMB', 'b_HML', 'b_RMW', 'b_CMA', 'b_Mom', 't_Mkt-RF', 't_SMB', 't_HML', 't_RMW', 't_CMA', 't_Mom']
emit('A2. Factor loadings (and NW t-stats)', regs[loadcols].set_index(['era', 'model']), floatfmt='{:.2f}')

# rolling 120-month
W = 120
roll = []
for i in range(W, len(d) + 1):
    sub = d.iloc[i - W:i]
    r6 = ols_hac(sub['ex'].to_numpy(), sub[MODELS['FF5+Mom']].to_numpy())
    r1 = ols_hac(sub['ex'].to_numpy(), sub[['Mkt-RF']].to_numpy())
    roll.append(dict(end=sub.index[-1].to_timestamp('M'), alpha6=r6['b'][0] * 1200, alpha6_se=r6['se'][0] * 1200,
                     beta6=r6['b'][1], beta6_se=r6['se'][1], alpha1=r1['b'][0] * 1200, beta1=r1['b'][1],
                     beta1_se=r1['se'][1], hml6=r6['b'][3], rmw6=r6['b'][4]))
roll = pd.DataFrame(roll).set_index('end')
save(roll, 'A_rolling_10y.csv')

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(roll.index, roll['alpha6'], color='#1f4e79', label='alpha, FF5+Mom (ann., pp)')
ax[0].fill_between(roll.index, roll['alpha6'] - 1.96 * roll['alpha6_se'], roll['alpha6'] + 1.96 * roll['alpha6_se'],
                   color='#1f4e79', alpha=0.15, label='95% band (Newey-West)')
ax[0].plot(roll.index, roll['alpha1'], color='#c55a11', lw=1, label='alpha, CAPM (ann., pp)')
ax[0].axhline(0, color='k', lw=0.8)
ax[0].set_ylabel('annualised alpha, pp')
ax[0].set_title('BRK-A rolling 10-year alpha (window ends on the date shown)')
ax[0].legend(fontsize=8, loc='upper right')
ax[1].plot(roll.index, roll['beta1'], color='#c55a11', label='market beta, CAPM')
ax[1].plot(roll.index, roll['beta6'], color='#1f4e79', label='market beta, FF5+Mom')
ax[1].fill_between(roll.index, roll['beta6'] - 1.96 * roll['beta6_se'], roll['beta6'] + 1.96 * roll['beta6_se'],
                   color='#1f4e79', alpha=0.15)
ax[1].axhline(1, color='grey', lw=0.8, ls='--')
ax[1].set_ylabel('market beta')
ax[1].legend(fontsize=8)
ax[1].set_xlabel('end of 10-year window')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_rolling_alpha_beta.png'), dpi=130)
plt.close(fig)
log('A done')

# =============================================================== B. replicas
FFm = load_prices(['FF_MKT', 'FF_RF', 'FF_VALUE', 'FF_TECH', 'FF_IND_Money'], how='inner')['close']
inds = ['BusEq', 'Chems', 'Durbl', 'Enrgy', 'Hlth', 'Manuf', 'Money', 'NoDur', 'Other', 'Shops', 'Telcm', 'Utils']
INDC = load_prices([f'FF_IND_{i}' for i in inds], how='inner')['close']
INDC.columns = inds
mk_ret = FFm['FF_MKT'].pct_change()
pre = INDC.pct_change().loc[:'1999-12-31']
betas_ind = {c: np.cov(pre[c].dropna(), mk_ret.loc[pre.index][pre[c].notna()])[0, 1] / mk_ret.loc[pre.index].var()
             for c in inds}
betas_ind = pd.Series(betas_ind).sort_values()
LOW3 = list(betas_ind.index[:3])
log(f'industry betas (daily, 1926-1999):\n{betas_ind.round(2).to_dict()}  LOW3={LOW3}')
emit('B0. Industry betas vs FF_MKT (daily returns, 1926-07..1999-12; used only to pick the low-beta trio)',
     betas_ind.rename('beta').to_frame(), floatfmt='{:.2f}')
low3_ret = INDC[LOW3].pct_change().mean(axis=1)           # daily-rebalanced equal mix
low3 = (1 + low3_ret.fillna(0)).cumprod() * 100

# daily factors
f3d = read_factors('ff3_daily.csv')
f5d = read_factors('ff5_daily.csv')
mod = read_factors('mom_daily.csv')
f6d = f5d.join(mod, how='inner')

# loadings
ESTW = ('2000-01', '2026-09')
sub = d.loc[pd.Period(ESTW[0], 'M'):pd.Period(ESTW[1], 'M')]
fit6 = ols_hac(sub['ex'].to_numpy(), sub[MODELS['FF5+Mom']].to_numpy())
fit3 = ols_hac(sub['ex'].to_numpy(), sub[MODELS['FF3']].to_numpy())
sub0 = d.loc[pd.Period('1980-04', 'M'):pd.Period('1999-12', 'M')]
fit6o = ols_hac(sub0['ex'].to_numpy(), sub0[MODELS['FF5+Mom']].to_numpy())
fit3o = ols_hac(sub0['ex'].to_numpy(), sub0[MODELS['FF3']].to_numpy())
LOAD = {'R6': (MODELS['FF5+Mom'], fit6['b'][1:]), 'R3': (MODELS['FF3'], fit3['b'][1:]),
        'R6_oos': (MODELS['FF5+Mom'], fit6o['b'][1:]), 'R3_oos': (MODELS['FF3'], fit3o['b'][1:])}
ld = pd.DataFrame({k: pd.Series(v[1], index=v[0]) for k, v in LOAD.items()}).T
emit('B1. Replica loadings (R6/R3 estimated 2000-01..2026-09; *_oos estimated 1980-04..1999-12)', ld, floatfmt='{:.3f}')


def replica(fac_df, names, beta):
    r = fac_df['RF'] + fac_df[names].to_numpy() @ np.asarray(beta)
    return (1 + r).cumprod() * 100


SL = {}                                                   # daily level series
SL['R6'] = replica(f6d, *LOAD['R6'])
SL['R3'] = replica(f3d, *LOAD['R3'])
SL['R6_oos'] = replica(f6d, *LOAD['R6_oos'])
SL['R3_oos'] = replica(f3d, *LOAD['R3_oos'])
SL['VALUE'] = FFm['FF_VALUE']
SL['MONEY'] = FFm['FF_IND_Money']
SL['LOWB3'] = low3
SL['TECH'] = FFm['FF_TECH']
MKT = FFm['FF_MKT']

# tracking vs BRK-A
SERIES_M = {k: mret(v) for k, v in SL.items()}
SERIES_M['MKT'] = mret(MKT)
SERIES_M['BRK-A'] = brk_m
SERIES_M['VFINX'] = spx_m
cmn = pd.concat({k: v for k, v in SERIES_M.items()}, axis=1).loc[pd.Period('2000-01', 'M'):LAST].dropna()
WINS = {'2000-03..2002-09': ('2000-03', '2002-09'), '2007-11..2009-02': ('2007-11', '2009-02'),
        '2022 (Jan-Dec)': ('2022-01', '2022-12'), '2025-02..2025-04': ('2025-02', '2025-04'),
        '2025 (Jan-Dec)': ('2025-01', '2025-12')}


def mdd(r):
    nav = (1 + r).cumprod()
    nav = pd.concat([pd.Series([1.0]), nav.reset_index(drop=True)])
    return (nav / nav.cummax() - 1).min()


trk = []
for k in ['BRK-A', 'MKT', 'VFINX', 'R6', 'R3', 'R6_oos', 'R3_oos', 'VALUE', 'MONEY', 'LOWB3', 'TECH']:
    r = cmn[k]
    row = {'series': k, 'ann_ret_%': ((1 + r).prod() ** (12 / len(r)) - 1) * 100,
           'ann_vol_%': r.std() * np.sqrt(12) * 100, 'corr_w_BRK': r.corr(cmn['BRK-A']),
           'TE_vs_BRK_%': (r - cmn['BRK-A']).std() * np.sqrt(12) * 100,
           'beta_to_BRK': np.cov(r, cmn['BRK-A'])[0, 1] / cmn['BRK-A'].var(),
           'maxDD_monthly_%': mdd(r) * 100}
    for w, (a, b) in WINS.items():
        rr = r.loc[pd.Period(a, 'M'):pd.Period(b, 'M')]
        row[f'ret {w} %'] = ((1 + rr).prod() - 1) * 100
    trk.append(row)
trk = pd.DataFrame(trk).set_index('series')
save(trk, 'B_tracking.csv', 2)
emit('B2. Tracking of actual BRK-A, monthly, 2000-01..2026-09 (n=%d); window returns are cumulative %%' % len(cmn),
     trk, floatfmt='{:.1f}')
# corr columns want 2 decimals
trk_c = trk[['corr_w_BRK', 'beta_to_BRK']]
emit('B2b. Correlation / beta to BRK-A (monthly)', trk_c, floatfmt='{:.2f}')

# figure 2: drawdown curves and window bars
fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.5, 1]})
cols = {'BRK-A': 'k', 'MKT': '#7f7f7f', 'R6': '#1f77b4', 'R3': '#2ca02c', 'VALUE': '#d62728', 'MONEY': '#9467bd',
        'LOWB3': '#ff7f0e'}
for k, c in cols.items():
    nav = (1 + cmn[k]).cumprod()
    dd = nav / nav.cummax() - 1
    ax[0].plot(dd.index.to_timestamp(), dd * 100, color=c, lw=2 if k == 'BRK-A' else 1.1, label=k)
ax[0].set_title('Drawdown from running peak, monthly, 2000-2026 (replicas are in-sample)')
ax[0].set_ylabel('%')
ax[0].legend(fontsize=8, ncol=2)
wl = list(WINS)[:4]
x = np.arange(len(wl))
bw = 0.11
for i, (k, c) in enumerate(cols.items()):
    ax[1].bar(x + (i - 3) * bw, [trk.loc[k, f'ret {w} %'] for w in wl], bw, color=c, label=k)
ax[1].set_xticks(x)
ax[1].set_xticklabels(wl, rotation=15, fontsize=8)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_title('Cumulative return in stress windows (%)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_replica_vs_actual.png'), dpi=130)
plt.close(fig)
log('B done')

# =============================================================== C. century rolling DCA
HOR = {'10y': 120, '20y': 240, '30y': 360}
WTS = [0.10, 0.20, 0.30, 0.50]
SLEEVES = ['R6', 'R3', 'VALUE', 'MONEY', 'LOWB3', 'TECH']


def run_roll(sleeve, H, start, wts):
    df = pd.concat({'M': MKT, 'S': SL[sleeve]}, axis=1).dropna()
    if start:
        df = df.loc[start:]
    ms = month_starts(df.index)
    out = {}
    for w in [0.0] + wts:
        alloc = pd.DataFrame({'M': 1 - w, 'S': w}, index=ms)
        out[w] = rolling_dca(df, df, alloc, H, slippage_rate=SLIP).set_index('start')['irr']
    return pd.DataFrame(out)


cent_rows, dec_rows, nono_rows = [], [], []
diffs_store = {}
for sample, start in [('1926+', None), ('1963+', '1963-07-01')]:
    for sl in SLEEVES:
        for hn, H in HOR.items():
            if sl == 'R6' and sample == '1926+':
                continue
            R = run_roll(sl, H, start, WTS)
            base = R[0.0]
            for w in WTS:
                m = R[w]
                dif = (m - base) * 100
                diffs_store[(sample, sl, hn, w)] = dif
                cent_rows.append(dict(sample=sample, sleeve=sl, H=hn, w=w, n_windows=len(R),
                                      mkt_median=base.median() * 100, mkt_p10=base.quantile(.1) * 100,
                                      mkt_min=base.min() * 100, mix_median=m.median() * 100,
                                      mix_p10=m.quantile(.1) * 100, mix_min=m.min() * 100,
                                      pct_beat=(m > base).mean() * 100, med_diff_pp=dif.median(),
                                      min_diff_pp=dif.min(), max_diff_pp=dif.max()))
                dec = (R.index.year // 10) * 10
                for dc in sorted(set(dec)):
                    sel = dec == dc
                    dec_rows.append(dict(sample=sample, sleeve=sl, H=hn, w=w, start_decade=dc, n=int(sel.sum()),
                                         mkt_median=base[sel].median() * 100, mix_median=m[sel].median() * 100,
                                         med_diff_pp=dif[sel].median(), pct_beat=(m[sel] > base[sel]).mean() * 100))
                # non-overlapping: every H months, offsets in whole years
                offs = []
                for o in range(0, H, 12):
                    idx = np.arange(o, len(R), H)
                    dd_ = dif.iloc[idx]
                    offs.append((len(idx), (dd_ > 0).mean() * 100, dd_.median(), dd_.mean()))
                offs = np.array(offs)
                o0 = dif.iloc[np.arange(0, len(R), H)]
                nono_rows.append(dict(sample=sample, sleeve=sl, H=hn, w=w, n_blocks_offset0=len(o0),
                                      pct_beat_offset0=(o0 > 0).mean() * 100, med_diff_offset0=o0.median(),
                                      n_offsets=len(offs), mean_blocks=offs[:, 0].mean(),
                                      pct_beat_mean_over_offsets=offs[:, 1].mean(),
                                      pct_beat_min_offset=offs[:, 1].min(), pct_beat_max_offset=offs[:, 1].max(),
                                      mean_diff_over_offsets=offs[:, 3].mean()))
    log(f'C {sample} done')
cent = pd.DataFrame(cent_rows)
decd = pd.DataFrame(dec_rows)
nono = pd.DataFrame(nono_rows)
save(cent, 'C_century_rolling.csv', 3)
save(decd, 'C_century_by_decade.csv', 3)
save(nono, 'C_century_nonoverlapping.csv', 3)

for sample in ['1926+', '1963+']:
    for w in [0.30, 0.50]:
        t = cent[(cent['sample'] == sample) & (cent['w'] == w)].copy()
        t = t.set_index(['H', 'sleeve'])[['n_windows', 'mkt_median', 'mkt_p10', 'mkt_min', 'mix_median', 'mix_p10',
                                          'mix_min', 'pct_beat', 'med_diff_pp', 'min_diff_pp', 'max_diff_pp']]
        t.index = t.index.set_levels(t.index.levels[0], level=0)
        emit(f'C1. Rolling DCA, sample {sample}, market vs market/sleeve {int((1 - w) * 100)}/{int(w * 100)} '
             '(IRR %, overlapping monthly-start windows)', t, floatfmt='{:.2f}')
        t2 = nono[(nono['sample'] == sample) & (nono['w'] == w)].set_index(['H', 'sleeve'])[
            ['n_blocks_offset0', 'pct_beat_offset0', 'med_diff_offset0', 'n_offsets', 'mean_blocks',
             'pct_beat_mean_over_offsets', 'pct_beat_min_offset', 'pct_beat_max_offset', 'mean_diff_over_offsets']]
        emit(f'C2. Non-overlapping windows, sample {sample}, {int((1 - w) * 100)}/{int(w * 100)}: blocks start 1st of '
             'sample and every H months; offsets = start shifted by 0,12,24..H-12 months', t2, floatfmt='{:.1f}')
t3 = decd[(decd['sample'] == '1926+') & (decd['w'] == 0.30) & (decd['H'] == '20y')]
t3 = t3.pivot(index='start_decade', columns='sleeve', values='med_diff_pp')
n3 = decd[(decd['sample'] == '1926+') & (decd['w'] == 0.30) & (decd['H'] == '20y') & (decd['sleeve'] == 'R3')].set_index('start_decade')['n']
t3.insert(0, 'n_windows', n3)
emit('C3. 20y windows, 70/30, median IRR difference vs 100% market (pp/yr) by start decade, sample 1926+', t3,
     floatfmt='{:.2f}')
t4 = decd[(decd['sample'] == '1926+') & (decd['w'] == 0.30) & (decd['H'] == '10y')]
t4 = t4.pivot(index='start_decade', columns='sleeve', values='med_diff_pp')
n4 = decd[(decd['sample'] == '1926+') & (decd['w'] == 0.30) & (decd['H'] == '10y') & (decd['sleeve'] == 'R3')].set_index('start_decade')['n']
t4.insert(0, 'n_windows', n4)
emit('C4. 10y windows, 70/30, median IRR difference vs 100% market (pp/yr) by start decade, sample 1926+', t4,
     floatfmt='{:.2f}')

# figure 3
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for a, hn in zip(ax, ['10y', '20y', '30y']):
    labs, data = [], []
    for sl in SLEEVES:
        for sample in ['1926+', '1963+']:
            key = (sample, sl, hn, 0.30)
            if key in diffs_store and (sample == '1926+' or sl == 'R6'):
                labs.append(f'{sl}\n{sample}')
                data.append(diffs_store[key].dropna().to_numpy())
    bp = a.boxplot(data, tick_labels=labs, showfliers=False, whis=(5, 95), patch_artist=True)
    for p in bp['boxes']:
        p.set_facecolor('#9ecae1')
    a.axhline(0, color='k', lw=0.8)
    a.set_title(f'{hn} DCA: IRR(70% mkt/30% sleeve) - IRR(100% mkt), pp/yr', fontsize=9)
    a.tick_params(axis='x', labelsize=7)
fig.suptitle('Century rolling windows (box = IQR, whiskers = 5-95%), overlapping starts', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_century_irr_differences.png'), dpi=130)
plt.close(fig)
log('C done')

# =============================================================== D. bootstrap
P = 10000
WB = np.array([0, .1, .2, .3, .4, .5])


def stationary_idx(N, T, L, n_paths, rng):
    new = rng.random((n_paths, T)) < 1.0 / L
    new[:, 0] = True
    starts = rng.integers(0, N, (n_paths, T))
    idx = np.empty((n_paths, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % N)
    return idx


def irr_monthly(final, H):
    """annualised IRR for unit deposits at the start of each of H months, final at end of month H."""
    g_lo = np.full(final.shape, -0.7)
    g_hi = np.full(final.shape, 0.5)

    def S(g):
        g = np.where(np.abs(g) < 1e-9, 1e-9, g)
        return np.exp(g) * np.expm1(g * H) / np.expm1(g)

    for _ in range(60):
        m = 0.5 * (g_lo + g_hi)
        up = S(m) < final            # need larger g
        g_lo = np.where(up, m, g_lo)
        g_hi = np.where(up, g_hi, m)
    g = 0.5 * (g_lo + g_hi)
    return np.exp(12 * g) - 1


def sim(S, B, wB, H, rebalanced=False):
    """S, B: (P,H) simple monthly returns. Contributions of 1 at the start of each month.
    Returns dict of (P,len(wB)) arrays."""
    Pn = S.shape[0]
    wB = np.asarray(wB)[None, :]
    va = np.zeros((Pn, wB.shape[1]))
    vb = np.zeros_like(va)
    tot = np.zeros_like(va)
    nav = np.ones_like(va)
    peak = nav.copy()
    maxdd = np.zeros_like(va)
    hist = [nav.copy()]
    worst12 = np.zeros_like(va)
    for t in range(H):
        s = S[:, t, None]
        b = B[:, t, None]
        pre = tot
        if rebalanced:
            tot_in = pre + 1.0
            tot = tot_in * ((1 - wB) * (1 + s) + wB * (1 + b))
        else:
            va = (va + (1 - wB)) * (1 + s)
            vb = (vb + wB) * (1 + b)
            tot_in = pre + 1.0
            tot = va + vb
        nav = nav * tot / tot_in
        peak = np.maximum(peak, nav)
        maxdd = np.minimum(maxdd, nav / peak - 1)
        hist.append(nav.copy())
        if t + 1 >= 12:
            worst12 = np.minimum(worst12, nav / hist[t + 1 - 12] - 1)
    return dict(irr=irr_monthly(tot, H), maxdd=maxdd, worst12=worst12, final=tot)


def summ(res, wB):
    base = {k: v[:, 0] for k, v in res.items()}
    rows = []
    for j, w in enumerate(wB):
        irr = res['irr'][:, j]
        dif = (irr - base['irr']) * 100
        r = dict(w_brk=w, irr_mean=irr.mean() * 100, irr_median=np.median(irr) * 100,
                 irr_p5=np.quantile(irr, .05) * 100, irr_p10=np.quantile(irr, .10) * 100,
                 irr_p90=np.quantile(irr, .90) * 100,
                 d_irr_mean=dif.mean(), d_irr_median=np.median(dif), d_irr_p5=np.quantile(dif, .05),
                 d_irr_p95=np.quantile(dif, .95), p_mix_gt_pure=(dif > 0).mean() * 100,
                 d_irr_mean_mcse=dif.std(ddof=1) / np.sqrt(len(dif)),
                 maxdd_median=np.median(res['maxdd'][:, j]) * 100, maxdd_p10worst=np.quantile(res['maxdd'][:, j], .10) * 100,
                 d_maxdd_mean=(res['maxdd'][:, j] - base['maxdd']).mean() * 100,
                 d_maxdd_p5=np.quantile(res['maxdd'][:, j] - base['maxdd'], .05) * 100,
                 d_maxdd_p95=np.quantile(res['maxdd'][:, j] - base['maxdd'], .95) * 100,
                 d_maxdd_median=np.median(res['maxdd'][:, j] - base['maxdd']) * 100,
                 worst12_median=np.median(res['worst12'][:, j]) * 100,
                 worst12_p10worst=np.quantile(res['worst12'][:, j], .10) * 100,
                 tw_p5_ratio=np.quantile(res['final'][:, j], .05) / np.quantile(res['final'][:, 0], .05),
                 tw_p10_ratio=np.quantile(res['final'][:, j], .10) / np.quantile(res['final'][:, 0], .10),
                 tw_median_ratio=np.median(res['final'][:, j]) / np.median(res['final'][:, 0]),
                 d_irr_p10_quantiles=(np.quantile(irr, .10) - np.quantile(base['irr'], .10)) * 100,
                 d_irr_median_quantiles=(np.median(irr) - np.median(base['irr'])) * 100)
        rows.append(r)
    return pd.DataFrame(rows)


# joint monthly pool
pool = pd.DataFrame({'S': spx_m, 'B': brk_m}).dropna()
pool = pool.loc[:pd.Period('2026-10', 'M')]
log(f'bootstrap pool {pool.index[0]}..{pool.index[-1]} n={len(pool)}')
lS_all, lB_all = np.log1p(pool['S']), np.log1p(pool['B'])
pool_stats = {}
for nm, a in [('1980-2026', '1980-04'), ('2000-2026', '2000-01'), ('2013-2026', '2013-01')]:
    sub_ = pool.loc[pd.Period(a, 'M'):]
    pool_stats[nm] = dict(n=len(sub_), S_mean_log_ann=np.log1p(sub_['S']).mean() * 1200,
                          B_mean_log_ann=np.log1p(sub_['B']).mean() * 1200,
                          S_vol=sub_['S'].std() * np.sqrt(12) * 100, B_vol=sub_['B'].std() * np.sqrt(12) * 100,
                          corr=sub_['S'].corr(sub_['B']))
ps = pd.DataFrame(pool_stats).T
emit('D0. Bootstrap pools (monthly VFINX / BRK-A): annualised mean log return (pp), vol (%), correlation',
     ps, floatfmt='{:.2f}')
save(ps, 'D_pool_stats.csv', 3)


def make_scn(name):
    """returns (pool_start, shift_log_per_month, blow_p)"""
    if name == 'S1_raw':
        return '1980-04', 0.0, 0.0
    if name == 'S2_2000+':
        return '2000-01', 0.0, 0.0
    if name == 'S3_2013+':
        return '2013-01', 0.0, 0.0
    if name == 'S4_noedge':
        return '1980-04', 'edge0', 0.0
    if name == 'S4b_noedge_2000+':
        return '2000-01', 'edge0', 0.0
    if name == 'S5_negedge1pp':
        return '1980-04', 'edge-1pp', 0.0
    if name.startswith('S6_blow'):
        return '1980-04', 'edge0', float(name.split('blow')[1]) / 100
    raise KeyError(name)


SCN = ['S1_raw', 'S2_2000+', 'S3_2013+', 'S4_noedge', 'S4b_noedge_2000+', 'S5_negedge1pp', 'S6_blow5', 'S6_blow10',
       'S6_blow20']
_idx_cache = {}


def get_paths(pool_start, L, H, seed_off=0):
    key = (pool_start, L, seed_off)
    if key not in _idx_cache:
        sub_ = pool.loc[pd.Period(pool_start, 'M'):]
        rng = np.random.default_rng(SEED + 1000 * seed_off + 17 * L + (1 if pool_start == '2000-01' else 2 if pool_start == '2013-01' else 0))
        idx = stationary_idx(len(sub_), 360, L, P, rng)
        if len(_idx_cache) > 6:
            _idx_cache.clear()
        _idx_cache[key] = (sub_, idx)
    return _idx_cache[key]


def scenario_returns(name, L, H, seed_off=0):
    pool_start, shift, blow = make_scn(name)
    sub_, idx = get_paths(pool_start, L, H, seed_off)
    idx = idx[:, :H]
    lS = np.log1p(sub_['S'].to_numpy())
    lB = np.log1p(sub_['B'].to_numpy())
    if shift == 'edge0':
        lB = lB - lB.mean() + lS.mean()
    elif shift == 'edge-1pp':
        lB = lB - lB.mean() + lS.mean() - 0.01 / 12
    S = np.expm1(lS)[idx]
    B = np.expm1(lB)[idx]
    if blow > 0:
        rng = np.random.default_rng(SEED + 99 + int(blow * 100) + H + seed_off)
        hit = rng.random(P) < blow
        mth = rng.integers(0, H, P)
        rows_ = np.nonzero(hit)[0]
        B[rows_, mth[hit]] = (1 + B[rows_, mth[hit]]) * 0.5 - 1
    return S, B


boot_rows = []
for L in [18, 6, 36]:
    for H in [20, 10, 30]:
        for scn in SCN:
            if L != 18 and scn not in ('S1_raw', 'S4_noedge', 'S2_2000+'):
                continue
            for reb in [False, True]:
                if reb and L != 18:
                    continue
                S, B = scenario_returns(scn, L, H * 12)
                res = sim(S, B, WB, H * 12, rebalanced=reb)
                t = summ(res, WB)
                t.insert(0, 'variant', 'rebalanced' if reb else 'fixed')
                t.insert(0, 'H', H)
                t.insert(0, 'L', L)
                t.insert(0, 'scenario', scn)
                boot_rows.append(t)
    log(f'D block length {L} done')
boot = pd.concat(boot_rows, ignore_index=True)
save(boot, 'D_bootstrap_results.csv', 3)

# MC repeat for headline
rep = []
for sd in range(6):
    S, B = scenario_returns('S4_noedge', 18, 240, seed_off=sd + 1)
    res = sim(S, B, WB, 240)
    t = summ(res, WB)
    t = t[t.w_brk == 0.3].iloc[0]
    rep.append(dict(seed=sd + 1, d_irr_p10_q=t['d_irr_p10_quantiles'], d_irr_median_q=t['d_irr_median_quantiles'],
                    d_irr_mean=t['d_irr_mean'], d_maxdd_mean=t['d_maxdd_mean'],
                    maxdd_median_mix=t['maxdd_median'], p_mix_gt_pure=t['p_mix_gt_pure']))
    # maxdd median of pure for the same seed
    rep[-1]['maxdd_median_pure'] = np.median(res['maxdd'][:, 0]) * 100
    rep[-1]['d_maxdd_median'] = rep[-1]['maxdd_median_mix'] - rep[-1]['maxdd_median_pure']
rep = pd.DataFrame(rep).set_index('seed')
save(rep, 'D_mc_repeats_noedge_w30_H20.csv', 3)
emit('D5. Monte-Carlo repeat of the headline (no edge, 20y, L=18, 30% BRK vs 0%), 6 independent seeds x 10,000 paths',
     rep, floatfmt='{:.3f}')
log('D done')

hdr = ['w_brk', 'irr_mean', 'irr_median', 'irr_p5', 'irr_p10', 'irr_p90', 'd_irr_mean', 'd_irr_median', 'd_irr_p5',
       'd_irr_p95', 'p_mix_gt_pure', 'maxdd_median', 'maxdd_p10worst', 'd_maxdd_mean', 'd_maxdd_p5', 'd_maxdd_p95', 'worst12_median', 'worst12_p10worst',
       'tw_p5_ratio', 'tw_p10_ratio']
base_tab = boot[(boot.L == 18) & (boot.H == 20) & (boot.variant == 'fixed')]
for scn in SCN:
    t = base_tab[base_tab.scenario == scn][hdr].set_index('w_brk')
    emit(f'D1. {scn}: 20y monthly DCA, L=18, fixed split, 10,000 paths (IRR %, drawdown %, differences in pp vs 0% BRK)',
         t, floatfmt='{:.2f}')
for H in [10, 30]:
    t = boot[(boot.scenario == 'S4_noedge') & (boot.L == 18) & (boot.H == H) & (boot.variant == 'fixed')][hdr].set_index('w_brk')
    emit(f'D2. S4_noedge, H={H}y, L=18, fixed', t, floatfmt='{:.2f}')
t = boot[(boot.scenario == 'S4_noedge') & (boot.L == 18) & (boot.H == 20) & (boot.variant == 'rebalanced')][hdr].set_index('w_brk')
emit('D3. S4_noedge, H=20y, L=18, monthly-rebalanced variant', t, floatfmt='{:.2f}')
sens = boot[(boot.w_brk == 0.3) & (boot.variant == 'fixed') & (boot.scenario.isin(['S4_noedge', 'S1_raw', 'S2_2000+']))]
sens = sens[['scenario', 'H', 'L', 'd_irr_p10_quantiles', 'd_irr_median_quantiles', 'd_irr_mean', 'p_mix_gt_pure',
             'd_maxdd_mean', 'maxdd_median', 'tw_p10_ratio']].sort_values(['scenario', 'H', 'L'])
emit('D4. Block-length and horizon sensitivity at 30% BRK (fixed split). d_irr_p10/median_q = quantile of mix minus quantile of pure (pp)',
     sens.set_index(['scenario', 'H', 'L']), floatfmt='{:.2f}')

# figures 4 and 5 (20y, L=18, fixed)
colors = {'S1_raw': '#1f77b4', 'S2_2000+': '#ff7f0e', 'S3_2013+': '#2ca02c', 'S4_noedge': 'k', 'S4b_noedge_2000+': '#7f7f7f',
          'S5_negedge1pp': '#d62728', 'S6_blow5': '#9467bd', 'S6_blow10': '#8c564b', 'S6_blow20': '#e377c2'}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
data_i, data_d, labs = [], [], []
for scn in SCN:
    S, B = scenario_returns(scn, 18, 240)
    res = sim(S, B, [0, 0.3], 240)
    data_i.append((res['irr'][:, 1] - res['irr'][:, 0]) * 100)
    data_d.append((res['maxdd'][:, 1] - res['maxdd'][:, 0]) * 100)
    labs.append(scn.replace('S', 'S', 1).replace('_', '\n', 1))
for a, dat, ttl in [(ax[0], data_i, 'IRR(70/30) - IRR(pure S&P), pp/yr'), (ax[1], data_d, 'maxDD(70/30) - maxDD(pure), pp (positive = shallower)')]:
    bp = a.boxplot(dat, tick_labels=labs, showfliers=False, whis=(5, 95), patch_artist=True)
    for p, scn in zip(bp['boxes'], SCN):
        p.set_facecolor(colors[scn])
        p.set_alpha(0.5)
    a.axhline(0, color='k', lw=0.8)
    a.set_title(ttl + ' (20y, L=18)', fontsize=10)
    a.tick_params(axis='x', labelsize=7)
fig.suptitle('Paired bootstrap differences, 10,000 paths; box = IQR, whiskers = 5-95%', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_bootstrap_difference_distributions.png'), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for scn in SCN:
    t = base_tab[base_tab.scenario == scn]
    ax[0].plot(t.w_brk * 100, t.irr_p10, marker='o', ms=3, color=colors[scn], label=scn)
    ax[1].plot(t.w_brk * 100, t.irr_median, marker='o', ms=3, color=colors[scn], label=scn)
    ax[2].plot(t.w_brk * 100, t.maxdd_median, marker='o', ms=3, color=colors[scn], label=scn)
for a, ttl in zip(ax, ['p10 IRR (%)', 'median IRR (%)', 'median max drawdown (%)']):
    a.set_title(ttl + ' vs Berkshire share, 20y DCA')
    a.set_xlabel('Berkshire share of each contribution, %')
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_weight_curves.png'), dpi=130)
plt.close(fig)

# =============================================================== E. Singapore tax tilt
tk = ['SPY', 'SPY_NET30', 'SPY_NET15', 'BRK-B']
pr = load_prices(tk, how='inner')
gaps = {}
for nm, (a, b) in {'1993-2026': ('1993-01-29', '2026-11-13'), '1993-1999': ('1993-01-29', '1999-12-31'),
                   '2000-2012': ('2000-01-01', '2012-12-31'), '2013-2026': ('2013-01-01', '2026-11-13')}.items():
    c = pr['close'].loc[a:b]
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    cg = (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1
    gaps[nm] = dict(SPY_cagr=cg['SPY'] * 100, NET30_cagr=cg['SPY_NET30'] * 100, NET15_cagr=cg['SPY_NET15'] * 100,
                    drag30_pp=(cg['SPY'] - cg['SPY_NET30']) * 100, drag15_pp=(cg['SPY'] - cg['SPY_NET15']) * 100)
gaps = pd.DataFrame(gaps).T
emit('E0. Measured withholding drag, SPY vs SPY_NET30 / SPY_NET15 (CAGR, pp/yr)', gaps, floatfmt='{:.2f}')
save(gaps, 'E_drag_measured.csv', 3)

# E1 historical rolling 1999-03+
op, cl = pr['open'], pr['close']
start = '1999-03-01'
# synthetic 1.2%-yield drag lines
syn = {}
for nm, dr in [('SPY_D36', 0.0036), ('SPY_D18', 0.0018)]:
    r = cl['SPY'].loc[start:].pct_change().fillna(0)
    lvl = (1 + r).cumprod() * np.exp(-dr * np.arange(len(r)) / 252.0)
    syn[nm] = lvl
cl2 = cl.loc[start:].copy()
op2 = op.loc[start:].copy()
for nm in syn:
    cl2[nm] = syn[nm]
    op2[nm] = syn[nm]
ms = month_starts(cl2.index)
hist_rows = []
for H in [120, 240]:
    for leg in ['SPY', 'SPY_NET30', 'SPY_NET15', 'SPY_D36', 'SPY_D18']:
        out = {}
        for label, wts in [('100% S&P', {leg: 1.0}), ('70/30', {leg: 0.7, 'BRK-B': 0.3}), ('100% BRK-B', {'BRK-B': 1.0})]:
            alloc = pd.DataFrame(0.0, index=ms, columns=[leg, 'BRK-B'])
            for k, v in wts.items():
                alloc[k] = v
            out[label] = rolling_dca(op2, cl2, alloc, H, slippage_rate=SLIP).set_index('start')['irr']
        o = pd.DataFrame(out)
        hist_rows.append(dict(H=f'{H // 12}y', sp_leg=leg, n_windows=len(o),
                              median_100SP=o['100% S&P'].median() * 100, median_70_30=o['70/30'].median() * 100,
                              median_100BRK=o['100% BRK-B'].median() * 100,
                              d_70_30_vs_SP_median_pp=(o['70/30'] - o['100% S&P']).median() * 100,
                              d_70_30_vs_SP_mean_pp=(o['70/30'] - o['100% S&P']).mean() * 100,
                              pct_70_30_beats=((o['70/30'] > o['100% S&P']).mean() * 100),
                              d_BRK_vs_SP_median_pp=(o['100% BRK-B'] - o['100% S&P']).median() * 100,
                              pct_BRK_beats=((o['100% BRK-B'] > o['100% S&P']).mean() * 100),
                              p10_100SP=o['100% S&P'].quantile(.1) * 100, p10_70_30=o['70/30'].quantile(.1) * 100,
                              min_100SP=o['100% S&P'].min() * 100, min_70_30=o['70/30'].min() * 100))
hist = pd.DataFrame(hist_rows)
save(hist, 'E_hist_rolling_1999.csv', 3)
emit('E1. Historical rolling DCA from 1999-03 (SPY leg gross / net 30% / net 15% / synthetic 0.36 and 0.18 pp drag) vs BRK-B (no drag); IRR %',
     hist.set_index(['H', 'sp_leg']), floatfmt='{:.2f}')

# E2 bootstrap on 1996+ real joint pool with net S&P leg
pool_t = pd.DataFrame({'S': mret(cl['SPY']), 'N30': mret(cl['SPY_NET30']), 'N15': mret(cl['SPY_NET15']),
                       'B': mret(cl['BRK-B'])}).dropna().loc[:pd.Period('2026-10', 'M')]
log(f'tax pool {pool_t.index[0]}..{pool_t.index[-1]} n={len(pool_t)}')
lS = np.log1p(pool_t['S'].to_numpy())
lB = np.log1p(pool_t['B'].to_numpy())
lB_ne = lB - lB.mean() + lS.mean()          # no edge vs GROSS S&P
rng = np.random.default_rng(SEED + 555)
tax_rows = []
for H in [20, 10]:
    idx = stationary_idx(len(pool_t), H * 12, 18, P, rng)
    for scn, bb in [('raw 1996+', lB), ('no edge vs gross S&P', lB_ne)]:
        B = np.expm1(bb)[idx]
        legs = {'gross': np.expm1(lS)[idx], 'net30': pool_t['N30'].to_numpy()[idx], 'net15': pool_t['N15'].to_numpy()[idx]}
        for dname, dr in [('drag 0.36pp (1.2% yield, 30%)', 0.0036), ('drag 0.18pp (1.2% yield, 15%)', 0.0018)]:
            legs[dname] = np.expm1(lS - dr / 12)[idx]
        for ln, Sx in legs.items():
            res = sim(Sx, B, [0, 0.3, 0.5, 1.0], H * 12)
            t = summ(res, [0, 0.3, 0.5, 1.0])
            for _, r in t[t.w_brk > 0].iterrows():
                tax_rows.append(dict(H=H, scenario=scn, sp_leg=ln, w_brk=r.w_brk, irr_median_pure=np.median(res['irr'][:, 0]) * 100,
                                     irr_median_mix=r.irr_median, d_irr_mean=r.d_irr_mean, d_irr_median=r.d_irr_median,
                                     d_irr_p5=r.d_irr_p5, d_irr_p95=r.d_irr_p95, p_mix_gt_pure=r.p_mix_gt_pure,
                                     d_p10_q=r.d_irr_p10_quantiles, maxdd_median=r.maxdd_median))
tax = pd.DataFrame(tax_rows)
save(tax, 'E_bootstrap_tax.csv', 3)
tt = tax[(tax.H == 20) & (tax.w_brk.isin([0.3, 1.0]))].set_index(['scenario', 'sp_leg', 'w_brk'])[
    ['irr_median_pure', 'irr_median_mix', 'd_irr_mean', 'd_irr_median', 'd_irr_p5', 'd_irr_p95', 'p_mix_gt_pure', 'd_p10_q']]
emit('E2. Bootstrap (joint SPY/BRK-B pool 1996-06..2026-10, 20y, L=18, 10,000 paths): effect of S&P-leg withholding. '
     'w_brk=1.0 is the 100% Berkshire vs 100% S&P comparison; no-edge shifts BRK-B to the GROSS S&P mean log return',
     tt, floatfmt='{:.2f}')
tt2 = tax[(tax.H == 10) & (tax.w_brk.isin([0.3]))].set_index(['scenario', 'sp_leg', 'w_brk'])[
    ['irr_median_pure', 'irr_median_mix', 'd_irr_mean', 'd_irr_median', 'd_irr_p5', 'd_irr_p95', 'p_mix_gt_pure', 'd_p10_q']]
emit('E3. Same, 10y horizon, 30% BRK', tt2, floatfmt='{:.2f}')

with open(os.path.join(OUT, 'tables.md'), 'w') as f:
    f.write('# EXP6 generated tables (auto-written by the script)\n\n' + '\n'.join(MD))
log('all done')
