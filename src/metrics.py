import pandas as pd
import numpy as np

def compute_strategy_metrics(
    backtest_result: pd.DataFrame, 
    risk_free_rate: float = 0.0
) -> pd.Series:
    T = 252
    N = len(backtest_result)
    total_value = backtest_result['total_value']
    returns = total_value.pct_change()
    trade_shares = backtest_result['trade_shares']
    open_price = backtest_result['open']
    daily_pnl = backtest_result['daily_pnl']
    total_return = total_value.iloc[-1] / total_value.iloc[0] - 1
    annualized_return = (total_value.iloc[-1] / total_value.iloc[0])**(T/N) - 1
    annualized_volatility = returns.std() * np.sqrt(T)
    peak = total_value.cummax()
    drawdown = (total_value - peak) / peak
    max_drawdown = drawdown.min()
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    calmar = annualized_return / abs(max_drawdown)
    trade_count = (trade_shares != 0).sum()
    turnover = abs(open_price * trade_shares).sum() / total_value.mean()
    daily_win_rate = (daily_pnl > 0).sum() / (daily_pnl != 0).sum()
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'trade_count': trade_count,
        'turnover': turnover,
        'daily_win_rate': daily_win_rate,
    }
    return pd.Series(metrics)

def compute_yearly_breakdown(res_strat: pd.DataFrame, res_bh: pd.DataFrame) -> pd.DataFrame:
    records = []
    strat_daily_ret = res_strat['total_value'].pct_change()
    spy_daily_ret = res_bh['total_value'].pct_change()
    for year, sub_strat in res_strat.groupby(res_strat.index.year):
        sub_bh = res_bh.loc[sub_strat.index]
        sub_strat_ret = strat_daily_ret.loc[sub_strat.index]
        sub_spy_ret = spy_daily_ret.loc[sub_strat.index]

        r_strat = (1 + sub_strat_ret.fillna(0)).prod() - 1
        r_spy = (1 + sub_spy_ret.fillna(0)).prod() - 1
        r_excess = r_strat - r_spy

        peak_strat = sub_strat['total_value'].cummax()
        strat_mdd = ((sub_strat['total_value'] - peak_strat) / peak_strat).min()
        peak_spy = sub_bh['total_value'].cummax()
        spy_mdd = ((sub_bh['total_value'] - peak_spy) / peak_spy).min()

        trades = (sub_strat['trade_shares'] != 0).sum()

        records.append({
            'year': year,
            'strat_return': r_strat,
            'spy_return': r_spy,
            'excess_return': r_excess,
            'strat_mdd': strat_mdd,
            'spy_mdd': spy_mdd,
            'trades': trades,
        })
    return pd.DataFrame(records).set_index('year')

def _calc_regime_stats(group: pd.DataFrame) -> pd.Series:
    ann_ret = group['return'].mean() * 252
    ann_vol = group['return'].std() * (252 ** 0.5)
    
    return pd.Series({
        'days': len(group),
        'exposure': (group['position'] > 0).mean(),
        'total_pnl': group['daily_pnl'].sum(),
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': ann_ret / ann_vol if ann_vol > 0 else 0.0,
        'trades': (group['trade_shares'] != 0).sum(),
    })

def compute_conditional_metrics(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby('regime').apply(_calc_regime_stats)