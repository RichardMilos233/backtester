"""EXP5-C: satellite ("play money") sleeve on top of a SPY 60 / BRK-B 40 core, $500/month."""
import multiprocessing as mp
import time

from exp5_common import *  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AMT = 500.0
CORE = {'SPY': .6, 'BRK-B': .4}
SHARES = [0.0, 0.10, 0.20, 0.30]
CAP_MULT = 1.5
SCEN = {   # name -> satellite columns (equal weight of each new satellite dollar)
    'QQQ': ['QQQ'], 'SMH': ['SMH'], 'NVDA': ['NVDA'], 'TSLA': ['TSLA'], 'ARKK': ['ARKK'],
    'MSFT': ['MSFT'], 'AAPL': ['AAPL'], 'MIX NVDA/TSLA/ARKK': ['NVDA', 'TSLA', 'ARKK'],
    'BADPICK 1.5xSPY-8%/yr': ['BADPICK'], 'WIPEOUT (sat money lost)': [],
}
D = {}


def build():
    tick = ['SPY', 'BRK-B', 'QQQ', 'SMH', 'NVDA', 'TSLA', 'ARKK', 'MSFT', 'AAPL']
    o, c = load_outer(tick)
    cols = {t: (o[t].dropna(), c[t].dropna()) for t in tick}
    cols['BADPICK'] = synth(o['SPY'].dropna(), c['SPY'].dropna(), 1.5, 0.08)
    master = o['SPY'].dropna().index
    for name, sat in SCEN.items():
        use = ['SPY', 'BRK-B'] + sat
        op, cl = frames({k: cols[k] for k in use})
        ms = full_month_starts(op.index, master)
        D[name] = (op, cl, ms)
        # common sample for all scenarios: 2014-11 onward
        op2, cl2 = op[op.index >= '2014-11-01'], cl[cl.index >= '2014-11-01']
        D[name + '|common'] = (op2, cl2, full_month_starts(op2.index, master))


def make_policy(sat, s, cap):
    def policy(ctx):
        cash = ctx.cash
        if cash <= 0:
            return {}
        w = {'SPY': (1 - s) * CORE['SPY'], 'BRK-B': (1 - s) * CORE['BRK-B']}
        for a in sat:
            w[a] = s / len(sat)
        if cap and s > 0 and sat and ctx.step > 0 and ctx.step % 12 == 0:
            v = ctx.values
            total = float(v.sum()) + cash
            sat_val = float(sum(v[a] for a in sat))
            if sat_val / total > CAP_MULT * s and sat_val > 0:
                core_cur = float(v['SPY'] + v['BRK-B']) / total
                X = (1 - s) - core_cur
                tgt = {a: s * float(v[a]) / sat_val for a in sat}
                tgt['SPY'] = float(v['SPY']) / total + CORE['SPY'] * X
                tgt['BRK-B'] = float(v['BRK-B']) / total + CORE['BRK-B'] * X
                return {'target': tgt}
        return {'buy': {a: x for a, x in w.items() if x > 0}}
    return policy


def run(name, s, cap, i, H):
    sample = name
    op, cl, ms = D[sample]
    sat = SCEN[name.split('|')[0]]
    wipe = name.startswith('WIPEOUT')
    end = ms[i + H] if (H and i + H < len(ms)) else None
    sl = (op.index >= ms[i]) & ((op.index < end) if end is not None else True)
    dec = ms[i:i + H] if H else ms[i:]
    if wipe:
        contrib = pd.Series(AMT * (1 - s), index=dec)
        led = ENGINE.run(op[sl], cl[sl], make_policy([], 0.0, False), contrib)
    else:
        led = ENGINE.run(op[sl], cl[sl], make_policy(sat, s, cap), pd.Series(AMT, index=dec))
    final = float(led['total_value'].iloc[-1])
    irr = irr_deposits(list(dec), [AMT] * len(dec), final, led.index[-1])
    mdd = max_dd(led['nav'].to_numpy())
    sat_val = sum(led[f'value_{a}'].iloc[-1] for a in sat) if sat else 0.0
    sold = 0.0
    if cap and sat:
        sold, _ = sold_stats(led, op[sl], [a for a in led.columns.str.replace('units_', '', regex=False)
                                           if f'units_{a}' in led.columns and not a.startswith('value')])
    return dict(irr=irr, final=final, mdd=mdd, sat_share=sat_val / final, sold=sold)


def task(a):
    name, kind, i, H = a
    key = name
    out = {}
    base = run(key, 0.0, False, i, H)
    rows = []
    ms = D[key][2]
    base_row = dict(scen=name, kind=kind, start=ms[i], s=0.0, cap=False, **base)
    rows.append(base_row)
    for s in SHARES[1:]:
        for cap in ([False] if name.startswith('WIPEOUT') else [False, True]):
            r = run(key, s, cap, i, H)
            rows.append(dict(scen=name, kind=kind, start=ms[i], s=s, cap=cap, **r,
                             d_irr_pp=(r['irr'] - base['irr']) * 100, wealth_ratio=r['final'] / base['final'],
                             d_mdd_pp=(r['mdd'] - base['mdd']) * 100, base_irr=base['irr'], base_final=base['final'],
                             base_mdd=base['mdd']))
    return rows


def main():
    t0 = time.time()
    build()
    tasks = []
    for name in D:
        scen = name.split('|')[0]
        ms = D[name][2]
        n = len(ms)
        if name.endswith('|common'):
            tasks.append((name, 'common_full', 0, 0))
            continue
        tasks.append((name, 'full', 0, 0))
        for H, kind in ((60, '5y'), (120, '10y')):
            for i in range(0, n - H, 6):
                tasks.append((name, kind, i, H))
    print(len(tasks), 'tasks')
    with mp.get_context('fork').Pool(4) as pool:
        out = pool.map(task, tasks, chunksize=4)
    df = pd.DataFrame([r for rows in out for r in rows])
    df['scen'] = df.scen.str.replace('|common', '', regex=False)
    print('done', round(time.time() - t0), 's')
    df.to_pickle('/tmp/claude-0/-home-user-backtester/33fa7352-a928-54df-8b07-5f65696c4528/scratchpad/exp5c_raw.pkl')
    df.round(5).to_csv(f'{OUT}/c_satellite_windows.csv.gz', index=False, compression='gzip')
    analyse(df)


def analyse(df):
    d = df[df.s > 0]
    rows = []
    for (scen, kind, s, cap), g in d.groupby(['scen', 'kind', 's', 'cap']):
        rows.append(dict(scen=scen, horizon=kind, s=s, capped=cap, n=len(g),
                         d_irr_pp_med=g.d_irr_pp.median(), d_irr_pp_p10=q(g.d_irr_pp, 10), d_irr_pp_p90=q(g.d_irr_pp, 90),
                         frac_irr_up=(g.d_irr_pp > 0).mean(),
                         wealth_ratio_med=g.wealth_ratio.median(), wealth_ratio_p10=q(g.wealth_ratio, 10),
                         wealth_ratio_p90=q(g.wealth_ratio, 90), wealth_ratio_min=g.wealth_ratio.min(),
                         d_mdd_pp_med=g.d_mdd_pp.median(), d_mdd_pp_worst=g.d_mdd_pp.min(),
                         sat_share_med=g.sat_share.median(), sat_share_p90=q(g.sat_share, 90), sat_share_max=g.sat_share.max(),
                         base_irr_med=g.base_irr.median(), sold_med=g.sold.median()))
    S = pd.DataFrame(rows)
    S.round(4).to_csv(f'{OUT}/c_satellite_summary.csv', index=False)

    # Fig 1: IRR delta vs s, by scenario (uncapped), medians, 5y and 10y
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, kind, ttl in zip(axes, ['5y', '10y', 'common_full'],
                             ['rolling 5y windows (median)', 'rolling 10y windows (median)', 'one full run 2014-11..2026-11 (all scenarios)']):
        for scen in SCEN:
            vals = [0.0]
            for s in SHARES[1:]:
                x = S[(S.scen == scen) & (S.horizon == kind) & (S.s == s) & (~S.capped)]
                vals.append(x.d_irr_pp_med.iloc[0] if len(x) else np.nan)
            if np.isnan(vals[1]):
                continue
            ax.plot([x * 100 for x in SHARES], vals, marker='o', label=scen, lw=1.5)
        ax.axhline(0, color='k', lw=.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('satellite share of monthly contribution (%)')
        ax.set_ylabel('IRR change vs s=0 (pp / yr)')
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, 'c1_satellite_irr_delta.png')
    plt.close(fig)

    # Fig 2: end-of-window satellite share, uncapped vs capped, s=20%, 10y windows
    names = [s for s in SCEN if not s.startswith('WIPEOUT')]
    fig, ax = plt.subplots(figsize=(13, 5))
    pos = np.arange(len(names))
    for k, (cap, lab, col) in enumerate([(False, 'uncapped', 'tab:red'), (True, 'annual cap (sell back to s if > 1.5 s)', 'tab:blue')]):
        data = [df[(df.scen == n) & (df.kind == '10y') & (df.s == .2) & (df.cap == cap)].sat_share.to_numpy() * 100 for n in names]
        bp = ax.boxplot(data, positions=pos + (k - .5) * .35, widths=.3, whis=(10, 90), showfliers=False, patch_artist=True)
        for b in bp['boxes']:
            b.set_facecolor(col); b.set_alpha(.5)
        ax.plot([], [], color=col, lw=6, alpha=.5, label=lab)
    ax.axhline(20, color='k', ls='--', lw=.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([n.replace(' ', '\n', 1) for n in names], fontsize=8)
    ax.set_ylabel('satellite % of account at end of 10y window')
    ax.set_title('s = 20%: satellite share of the account after 10 years (box p25-p75, whiskers p10-p90)')
    ax.legend()
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    savefig(fig, 'c2_satellite_end_share.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
