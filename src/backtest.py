import pandas as pd
import numpy as np


class Backtest:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        slippage_rate: float = 0.0005,
        commission_rate: float = 0.0005,
        rebalance_tolerance: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.rebalance_tolerance = rebalance_tolerance

    def run(self, data: pd.DataFrame, signals: pd.Series, dividend_events: pd.DataFrame,) -> pd.DataFrame:
        """
        运行回测引擎
        :param data: 行情 DataFrame，必须包含 'open' 和 'close'，索引为日期
        :param signals: 目标仓位权重 Series (0.0~1.0，支持连续权重与离散 0/1)，索引为日期
        :return: 包含 6 大账本、摩擦成本、市场状态与日收益率的明细 DataFrame
        """
        target_signals = signals.shift(1)
        current_cash = self.initial_capital
        current_position = 0
        prev_total_value = self.initial_capital
        records = []
        events = dividend_events.set_index('ex_date')
        pending_dividends = {}

        for date in data.index:
            open_price = data['open'][date]
            close_price = data['close'][date]
            signal = target_signals[date]

            dividend_accrued = 0.0
            if date in events.index:
                event = events.loc[date]
                dividend_accrued = current_position * event['dividend_per_share']
                pay_date = event['pay_date']

                pending_dividends[pay_date] = (
                    pending_dividends.get(pay_date, 0.0) + dividend_accrued
                )

            dividend_receivable = sum(pending_dividends.values())

            # 每股采购综合成本预算（含滑点与手续费，严防资金穿仓）
            cost_per_share = open_price * (1 + self.slippage_rate) * (1 + self.commission_rate)
            v_open = current_cash + current_position * open_price + dividend_receivable

            # 目标仓位撮合逻辑（兼容 0/1 离散信号与 0.0~1.0 连续权重）
            if pd.isna(signal):
                trade_shares = 0
            elif signal == 0:
                trade_shares = -current_position
            else:
                # 目标权重发生调整（加仓、减仓或建仓）
                target_value = v_open * signal
                current_value = current_position * open_price
                delta_value = target_value - current_value
                drift_ratio = delta_value / v_open if v_open > 0 else 0.0

                if abs(drift_ratio) < self.rebalance_tolerance:
                    trade_shares = 0
                elif delta_value > 0:
                    desired_buy = delta_value // cost_per_share
                    max_buy = current_cash // cost_per_share
                    trade_shares = max(0, min(desired_buy, max_buy))
                elif delta_value < 0:
                    desired_sell = abs(delta_value) // open_price
                    trade_shares = -min(current_position, desired_sell)
                else:
                    trade_shares = 0


            # 实际执行价格计算（考虑买卖方向滑点）
            if trade_shares > 0:
                execution_price = open_price * (1 + self.slippage_rate)
            elif trade_shares < 0:
                execution_price = open_price * (1 - self.slippage_rate)
            else:
                execution_price = open_price

            # 佣金扣除与资金/持仓结算
            commission = abs(trade_shares) * execution_price * self.commission_rate
            current_cash -= trade_shares * execution_price + commission
            current_position += trade_shares

            dividend_paid = 0.0
            for pay_date in list(pending_dividends):
                if pay_date <= date:
                    dividend_paid += pending_dividends.pop(pay_date)

            current_cash += dividend_paid
            dividend_receivable = sum(pending_dividends.values())

            # 收盘盯市结算（Mark to Market）
            asset_value = current_position * close_price
            total_value = current_cash + asset_value + dividend_receivable
            pnl = total_value - prev_total_value
            prev_total_value = total_value

            records.append({
                'date': date,
                'open': open_price,
                'close': close_price,
                'signal': signal,
                'position': current_position,
                'cash': current_cash,
                'asset_value': asset_value,
                'total_value': total_value,
                'daily_pnl': pnl,
                'trade_shares': trade_shares,
                'commission': commission,
                'regime': data['regime'][date] if 'regime' in data.columns else None,
                'dividend_accrued': dividend_accrued,
                'dividend_paid': dividend_paid,
                'dividend_receivable': dividend_receivable,
            })

        df_records = pd.DataFrame(records).set_index('date')
        df_records['return'] = df_records['total_value'].pct_change()
        return df_records
