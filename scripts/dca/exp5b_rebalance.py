"""EXP5-B: long-run rebalancing rule while DCA-ing $500/month."""
import multiprocessing as mp
import time

from exp5_common import *  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AMT = 500.0
SETS = {
    'SPY60/BRKB40': dict(w={'SPY': .6, 'BRK-B': .4}, start='1996-06-01'),
    'SPY50/QQQ20/BRKB30': dict(w={'SPY': .5, 'QQQ': .2, 'BRK-B': .3}, start='1999-04-01'),
    'VFINX60/BRKA40 (long)': dict(w={'VFINX': .6, 'BRK-A': .4}, start='1980-04-01'),
    'VFINX50/NDX20/BRKA30 (long, NDX=price index)': dict(w={'VFINX': .5, '^NDX': .2, 'BRK-A': .3}, start='1985-11-01'),
    'VOO50/BRKB30/VXUS20': dict(w={'VOO': .5, 'BRK-B': .3, 'VXUS': .2}, start='2011-02-01'),
}
RULES = {
    'fixed_buy': lambda t: fixed_buy(t),
    'buy_underweight': lambda t: buy_underweight(t),
    'rebalance_12m': lambda t: rebalance_to(t, every=12),
    'band_5pt': lambda t: rebalance_to(t, band=0.05),
    'band_10pt': lambda t: rebalance_to(t, band=0.10),
}
D = {}   # set -> (opens, closes, ms)


def build():
    master = load_prices(['SPY'])['open'].index
    for name, s in SETS.items():
        p = load_prices(list(s['w']), start=s['start'], how='inner')
        ms = full_month_starts(p['open'].index, master)
        D[name] = (p['open'], p['close'], ms)


def run_window(name, rule, i, H):
    opens, closes, ms = D[name]
    t = SETS[name]['w']
    end = ms[i + H] if (H and i + H < len(ms)) else None
    sl = (opens.index >= ms[i]) & ((opens.index < end) if end is not None else True)
    o, c = opens[sl], closes[sl]
    dec = ms[i:i + H] if H else ms[i:]
    led = ENGINE.run(o, c, RULES[rule](t), pd.Series(AMT, index=dec))
    return led


def metrics(led, opens, name):
    t = SETS[name]['w']
    assets = list(t)
    m = dca_metrics(led)
    sold, ev = sold_stats(led, opens, assets)
    last = led.iloc[-1]
    w = {a: last[f'value_{a}'] / last['total_value'] for a in assets}
    return dict(irr=m['irr'], twr=m['twr_annualized'], vol=m['annualized_volatility'], mdd=m['max_drawdown'],
                worst12=m['worst_12m_return'], final=m['final_value'], contributed=m['total_contributed'],
                sold=sold, sell_events=ev, years=m['years'],
                max_weight_dev=max(abs(w[a] - t[a]) for a in assets),
                **{f'w_{a}': w[a] for a in assets})


def task(a):
    name, rule, kind, i, H = a
    led = run_window(name, rule, i, H)
    r = metrics(led, D[name][0], name)
    r.update(set=name, rule=rule, kind=kind, start=D[name][2][i])
    return r


def main():
    t0 = time.time()
    build()
    # sanity: fixed_buy irr equals rolling_dca
    name = 'SPY50/QQQ20/BRKB30'
    opens, closes, ms = D[name]
    alloc = pd.DataFrame([SETS[name]['w']] * len(ms), index=ms)
    rd = rolling_dca(opens, closes, alloc, 120, amount=AMT)
    r0 = task((name, 'fixed_buy', 'chk', 0, 120))
    print('sanity IRR  engine', round(r0['irr'], 6), ' rolling_dca', round(rd.irr.iloc[0], 6))
    assert abs(r0['irr'] - rd.irr.iloc[0]) < 2e-4

    tasks = []
    for name in SETS:
        ms = D[name][2]
        n = len(ms)
        for rule in RULES:
            tasks.append((name, rule, 'full', 0, 0))
            for H, kind in ((120, '10y'), (240, '20y')):
                for i in range(0, n - H, 6):
                    tasks.append((name, rule, kind, i, H))
    print(len(tasks), 'runs')
    with mp.get_context('fork').Pool(4) as pool:
        rows = pool.map(task, tasks, chunksize=6)
    df = pd.DataFrame(rows)
    print('done', round(time.time() - t0), 's')
    df.round(5).to_csv(f'{OUT}/b_rebalance_windows.csv.gz', index=False, compression='gzip')
    analyse(df)
    drift_figure()


def analyse(df):
    summ, paired = [], []
    for (name, kind, rule), g in df.groupby(['set', 'kind', 'rule']):
        summ.append(dict(set=name, horizon=kind, rule=rule, n=len(g),
                         irr_med=g.irr.median(), irr_p10=q(g.irr, 10), irr_p90=q(g.irr, 90),
                         irr_std=g.irr.std(), twr_med=g.twr.median(), vol_med=g.vol.median(),
                         mdd_med=g.mdd.median(), mdd_worst=g.mdd.min(), worst12_med=g.worst12.median(),
                         worst12_worst=g.worst12.min(), final_med=g.final.median(),
                         sold_med=g.sold.median(), sold_p90=q(g.sold, 90), sell_events_med=g.sell_events.median(),
                         sell_events_max=g.sell_events.max(),
                         sold_pct_of_contrib=(g.sold / g.contributed).median(), maxdev_med=g.max_weight_dev.median(),
                         maxdev_p90=q(g.max_weight_dev, 90)))
    S = pd.DataFrame(summ)
    S.round(4).to_csv(f'{OUT}/b_rebalance_summary.csv', index=False)
    for (name, kind), g in df.groupby(['set', 'kind']):
        base = g[g.rule == 'fixed_buy'].set_index('start')
        for rule in RULES:
            if rule == 'fixed_buy':
                continue
            x = g[g.rule == rule].set_index('start')
            d_irr = (x.irr - base.irr) * 100
            d_mdd = (x.mdd - base.mdd) * 100
            d_fin = x.final / base.final - 1
            paired.append(dict(set=name, horizon=kind, rule=rule, n=len(x),
                               d_irr_pp_med=d_irr.median(), d_irr_pp_p10=q(d_irr, 10), d_irr_pp_p90=q(d_irr, 90),
                               frac_irr_better=(d_irr > 0).mean(),
                               d_mdd_pp_med=d_mdd.median(), d_final_med=d_fin.median(),
                               irr_dispersion_pp=base.irr.std() * 100,
                               signal_to_noise=abs(d_irr.median()) / max(d_irr.std(), 1e-9)))
    P = pd.DataFrame(paired)
    P.round(4).to_csv(f'{OUT}/b_rule_vs_fixed_buy.csv', index=False)
    # figure: median IRR and median mdd by rule, rolling 10y, per set (error bars p10-p90 of irr)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    sets = [s for s in SETS]
    for ax, metric, ttl in ((axes[0], 'irr', 'IRR (median, bars p10-p90 across rolling 10y windows)'),
                            (axes[1], 'mdd', 'Max drawdown within window (median, bars p10-worst)')):
        w = 0.16
        for k, rule in enumerate(RULES):
            meds, lo, hi = [], [], []
            for s in sets:
                g = df[(df.set == s) & (df.kind == '10y') & (df.rule == rule)][metric] * 100
                meds.append(g.median())
                if metric == 'irr':
                    lo.append(g.median() - q(g, 10)); hi.append(q(g, 90) - g.median())
                else:
                    lo.append(g.median() - g.min()); hi.append(q(g, 90) - g.median() if False else 0)
            ax.bar(np.arange(len(sets)) + (k - 2) * w, meds, w, yerr=[lo, hi], capsize=2, label=rule)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s.replace(' (long', '\n(long') for s in sets], fontsize=8)
        ax.set_title(ttl)
        ax.grid(alpha=.3, axis='y')
        ax.set_ylabel('%')
    axes[0].legend(fontsize=8, ncol=5)
    fig.tight_layout()
    savefig(fig, 'b1_rebalance_rules.png')
    plt.close(fig)

    # figure: sold dollars and events
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, rule in enumerate(['rebalance_12m', 'band_5pt', 'band_10pt']):
        sd = [df[(df.set == s) & (df.kind == '10y') & (df.rule == rule)].sold.median() / (AMT * 120) * 100 for s in sets]
        ev = [df[(df.set == s) & (df.kind == '10y') & (df.rule == rule)].sell_events.median() for s in sets]
        axes[0].bar(np.arange(len(sets)) + (k - 1) * .25, sd, .25, label=rule)
        axes[1].bar(np.arange(len(sets)) + (k - 1) * .25, ev, .25, label=rule)
    for ax, t in zip(axes, ['Median USD sold per rolling 10y window (% of $60k contributed)', 'Median number of sell days per 10y window']):
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s.split(' ')[0] for s in sets], fontsize=8, rotation=15)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=.3, axis='y')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'b2_selling_effort.png')
    plt.close(fig)


def drift_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), ['SPY60/BRKB40', 'SPY50/QQQ20/BRKB30', 'VFINX60/BRKA40 (long)',
                                       'VFINX50/NDX20/BRKA30 (long, NDX=price index)']):
        opens, closes, ms = D[name]
        led = ENGINE.run(opens, closes, fixed_buy(SETS[name]['w']), pd.Series(AMT, index=ms))
        for a in SETS[name]['w']:
            ax.plot(led.index, led[f'value_{a}'] / led['total_value'] * 100, label=f'{a} (target {SETS[name]["w"][a]*100:.0f}%)')
        ax.set_title(f'fixed_buy weight drift: {name}', fontsize=9)
        ax.set_ylabel('% of account')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, 'b3_weight_drift_fixed_buy.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
