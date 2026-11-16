"""EXP7 - out-of-sample and walk-forward tests of the round-1 conclusions.  Runs parts A, B, C and writes findings.md.

    python scripts/dca/exp7_walk_forward.py            (about 6-8 minutes on 4 cores)

Parts live in exp7a_oos_replication.py, exp7b_wf_rules.py, exp7c_wf_weights.py (shared code: exp7_common.py).
The prose of findings.md is in exp7_prose.md; {{table:name}} placeholders are filled from the generated CSVs.
Set EXP7_NBOOT to change the number of bootstrap paths (default 2000).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp7_common import OUT, ROOT  # noqa

import pandas as pd

import exp7a_oos_replication as A
import exp7b_wf_rules as B
import exp7c_wf_weights as C


def md(df, floatfmt=None):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append('' if pd.isna(v) else (floatfmt.get(c, '{:.3f}') if floatfmt else '{:.3f}').format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def tables(rA, rB, rC):
    """name -> markdown, used by exp7_prose.md placeholders."""
    T = {}
    S, V, TT = rA['S'], rA['V'], rA['T']
    # headline verdict table (global-average control is primary)
    v = V.copy()
    T['verdicts'] = md(v[['rule', 'round1_expect', 'round1_range_10y_matched', 'panel_median', 'iid_90', 'panel_median_g', 'share_pairs_gt1_g',
                          'shift_band_g', 'iid_90_g', 'eras_same_sign_g', 'verdict_g']].rename(columns={
        'round1_expect': 'round-1 claim (vs matched)', 'round1_range_10y_matched': 'round-1 10y range', 'panel_median': '66-pair median (matched)',
        'iid_90': 'iid-world 90% (matched)', 'panel_median_g': '66-pair median (global mix)', 'share_pairs_gt1_g': 'share of pairs >1',
        'shift_band_g': 'placebo 5-95%', 'iid_90_g': 'iid-world 90% (global)', 'eras_same_sign_g': 'eras same sign', 'verdict_g': 'verdict'}))
    s10 = S[S.horizon == '10y']
    t = s10[['rule', 'median_of_pair_medians', 'q25', 'q75', 'share_pairs_gt1', 'p_shift_median', 'boot_p5', 'boot_p95', 'iid_p5', 'iid_p95',
             'nonov_median', 'nonov_share_pairs_gt1']]
    T['panel10_matched'] = md(t.rename(columns={'median_of_pair_medians': 'median', 'share_pairs_gt1': 'share>1', 'p_shift_median': 'p (placebo)'}))
    t = s10[['rule', 'median_of_pair_medians_g', 'q25_g', 'q75_g', 'share_pairs_gt1_g', 'p_shift_median_g', 'boot_p5_g', 'boot_p95_g', 'iid_p5_g', 'iid_p95_g',
             'nonov_median_g', 'nonov_share_pairs_gt1_g']]
    T['panel10_global'] = md(t.rename(columns=lambda c: c.replace('median_of_pair_medians_g', 'median').replace('_g', '')))
    for h in ('5y', '20y'):
        t = S[S.horizon == h][['rule', 'median_of_pair_medians', 'share_pairs_gt1', 'median_of_pair_medians_g', 'share_pairs_gt1_g', 'iid_p5_g', 'iid_p95_g']]
        T[f'panel_{h}'] = md(t.rename(columns={'median_of_pair_medians': 'median (matched)', 'share_pairs_gt1': 'share>1 (matched)',
                                               'median_of_pair_medians_g': 'median (global)', 'share_pairs_gt1_g': 'share>1 (global)',
                                               'iid_p5_g': 'iid p5 (global)', 'iid_p95_g': 'iid p95 (global)'}))
    e = s10[['rule'] + [c + '_median_g' for c in A.ERAS]]
    T['panel_eras'] = md(e.rename(columns=lambda c: c.replace('_median_g', '')))
    e = s10[['rule'] + [c + '_median' for c in A.ERAS]]
    T['panel_eras_matched'] = md(e.rename(columns=lambda c: c.replace('_median', '')))
    nm = TT[TT.group.isin(['named', 'intl']) & (TT.horizon == '10y')]
    pv = nm.pivot(index='rule', columns='pair', values='med_g').loc[A.FROZEN]
    T['named10_global'] = md(pv.reset_index())
    pv = nm.pivot(index='rule', columns='pair', values='med').loc[A.FROZEN]
    T['named10_matched'] = md(pv.reset_index())
    nm20 = TT[TT.group.isin(['named', 'intl']) & (TT.horizon == '20y')]
    pv = nm20.pivot(index='rule', columns='pair', values='med_g').loc[A.FROZEN]
    T['named20_global'] = md(pv.reset_index())
    pv = nm.pivot(index='rule', columns='pair', values='p_shift_g').loc[A.FROZEN]
    T['named10_p'] = md(pv.reset_index())
    pv = nm.pivot(index='rule', columns='pair', values='nonov_med_g').loc[A.FROZEN]
    T['named10_nonov'] = md(pv.reset_index())
    nn = nm[['pair', 'rule', 'n']].drop_duplicates('pair')[['pair', 'n']]
    T['named_nwin'] = md(nn.rename(columns={'n': 'rolling 10y windows'}))
    d = rA['demo']
    d10 = d[d.horizon == '10y'].pivot(index='rule', columns='control', values='mean_of_median_ratio').loc[A.FROZEN].reset_index()
    T['demo10'] = md(d10)
    r1 = TT[TT.group == 'r1']
    r1 = r1[r1.horizon == '10y'][['pair', 'rule', 'med', 'iid_mean', 'med_g', 'iid_mean_g', 'boot_p5_g', 'boot_p95_g']]
    r1['pair'] = r1.pair.str.replace(' (round-1 pair, no proxy warm-up)', '', regex=False)
    T['round1_vs_iid'] = md(r1)
    # B
    R = rB['R']
    Rc = R[R.universe == 'common_start_20y'].copy()
    Rc['irr'] = Rc.irr * 100
    T['b_tracks'] = md(Rc[['pair', 'oos_start', 'strategy', 'wm', 'irr', 'avg_fracA']].rename(columns={'wm': 'terminal wealth / $', 'irr': 'IRR %', 'avg_fracA': 'avg A-frac'}),
                       {'terminal wealth / $': '{:.2f}', 'IRR %': '{:.2f}', 'avg A-frac': '{:.2f}'})
    P = rB['P']
    rows = []
    for (pair, sch), g in P.groupby(['pair', 'scheme']):
        g = g.sort_values('n_picks', ascending=False)
        fixed = g[g.rule.str.startswith('fixed')].share.sum()
        mom = g[g.rule.str.startswith(('mom', 'z_momentum'))].share.sum()
        con = g[g.rule.str.startswith(('contra', 'z_mean'))].share.sum()
        top = ', '.join(f'{r} {s:.0%}' for r, s in zip(g.rule.head(3), g.share.head(3)))
        rows.append(dict(pair=pair, scheme=sch, n_Jan=int(g.n_januaries.iloc[0]), distinct_rules=len(g), fixed_mix=f'{fixed:.0%}', momentum_type=f'{mom:.0%}',
                         contrarian_type=f'{con:.0%}', top3=top))
    T['b_picks'] = md(pd.DataFrame(rows))
    BB = rB['BB']
    T['b_boot'] = md(BB[['pair', 'scheme', 'comparison', 'real_ratio', 'boot_mean_ratio', 'boot_p5', 'boot_p95', 'P_ratio_gt1', 'real_pctile_in_boot']])
    SH = rB['SH']
    g = SH.groupby('pair').agg(n=('rule', 'size'), in_sample=('in_sample_ratio', lambda x: x.mean() - 1), oos=('fwd5y_ratio', lambda x: x.mean() - 1),
                               median_candidate_train=('median_cand_train_ratio', lambda x: x.mean() - 1),
                               median_candidate_fwd=('median_cand_fwd_ratio', lambda x: x.mean() - 1)).reset_index()
    tot = pd.DataFrame([dict(pair='all pairs', n=len(SH), in_sample=SH.in_sample_ratio.mean() - 1, oos=SH.fwd5y_ratio.mean() - 1,
                             median_candidate_train=SH.median_cand_train_ratio.mean() - 1, median_candidate_fwd=SH.median_cand_fwd_ratio.mean() - 1)])
    T['b_shrink'] = md(pd.concat([g, tot]).rename(columns={'in_sample': 'picked rule, trailing 10y vs 50/50', 'oos': 'picked rule, next 5y vs 50/50',
                                                            'median_candidate_train': 'median candidate, trailing', 'median_candidate_fwd': 'median candidate, next 5y'}))
    RR = rB['RR']
    T['b_rolling'] = md(RR[RR.horizon == '10y'][['pair', 'scheme', 'n_windows', 'med_vs_50_50', 'share_gt1_vs_50_50', 'nonov_med_vs_50_50', 'nonov_n']])
    # C
    Tc = rC['T'].copy()
    Tc['irr'] = Tc.irr * 100
    Tc['max_drawdown'] = Tc.max_drawdown * 100
    T['c_tracks'] = md(Tc[['universe', 'method', 'months', 'wealth_multiple', 'irr', 'max_drawdown', 'mean_annual_turnover', 'mean_max_weight']].rename(
        columns={'wealth_multiple': 'wealth / $', 'irr': 'IRR %', 'max_drawdown': 'max DD %', 'mean_annual_turnover': 'yearly weight change', 'mean_max_weight': 'avg max weight'}),
        {'wealth / $': '{:.2f}', 'IRR %': '{:.2f}', 'max DD %': '{:.1f}', 'yearly weight change': '{:.2f}', 'avg max weight': '{:.2f}'})
    BC = rC['BB']
    sub = BC[BC.vs.isin(['equal weight', '100% first asset'])]
    T['c_boot'] = md(sub[['universe', 'method', 'vs', 'real_wealth_ratio', 'boot_p5', 'boot_p95', 'P_ratio_gt1', 'real_irr_diff_pp', 'boot_irr_p5_pp', 'boot_irr_p95_pp']].rename(
        columns={'real_wealth_ratio': 'actual wealth ratio', 'boot_p5': 'boot 5%', 'boot_p95': 'boot 95%', 'P_ratio_gt1': 'P(ratio>1)', 'real_irr_diff_pp': 'IRR diff pp',
                 'boot_irr_p5_pp': 'IRR diff 5%', 'boot_irr_p95_pp': 'IRR diff 95%'}))
    RW = rC['RW']
    x = RW[(RW.vs == '100% first asset') | (RW.vs == 'equal weight')]
    T['c_rolling'] = md(x[x.horizon == '10y'][['universe', 'method', 'vs', 'n_windows', 'median_ratio', 'share_gt1', 'nonov_median', 'nonov_n']])
    T['c_rolling20'] = md(x[x.horizon == '20y'][['universe', 'method', 'vs', 'n_windows', 'median_ratio', 'share_gt1', 'nonov_median', 'nonov_n']])
    return T


def main():
    rA = A.run()
    rB = B.run()
    rC = C.run()
    T = tables(rA, rB, rC)
    for k, v in T.items():
        open(os.path.join(OUT, f'_table_{k}.md'), 'w').write(v + '\n')
    prose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp7_prose.md')
    if os.path.exists(prose_path):
        txt = open(prose_path).read()
        txt = re.sub(r'\{\{table:(\w+)\}\}', lambda m: T[m.group(1)], txt)
        extra = dict(nboot=rA['n_boot'], niid=rA['n_iid'], npairs=rA['npairs'], msgA=rA['msg'], msgB=rB['msg'], msgC=rC['msg'])
        txt = re.sub(r'\{\{(\w+)\}\}', lambda m: str(extra.get(m.group(1), m.group(0))), txt)
        open(os.path.join(OUT, 'findings.md'), 'w').write(txt)
    print('done')


if __name__ == '__main__':
    main()
