import pandas as pd


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

    def run(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Run the backtest on adjusted OHLC. Dividends are already in the returns.
        :param data: market DataFrame with 'open' and 'close', indexed by date
        :param signals: target weight Series from 0.0 to 1.0, indexed by date
        :return: ledger with positions, friction, regime, and daily returns
        """
        target_signals = signals.shift(1)
        current_cash = self.initial_capital
        current_position = 0
        prev_total_value = self.initial_capital
        records = []

        for date in data.index:
            open_price = data['open'][date]
            close_price = data['close'][date]
            signal = target_signals[date]

            # All-in cost per share, including slippage and commission.
            cost_per_share = open_price * (1 + self.slippage_rate) * (1 + self.commission_rate)
            v_open = current_cash + current_position * open_price

            # Target weight: 0 liquidates; any other weight is a fraction of open equity.
            if pd.isna(signal):
                trade_shares = 0
            elif signal == 0:
                trade_shares = -current_position
            else:
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

            # Execution price includes directional slippage.
            if trade_shares > 0:
                execution_price = open_price * (1 + self.slippage_rate)
            elif trade_shares < 0:
                execution_price = open_price * (1 - self.slippage_rate)
            else:
                execution_price = open_price

            commission = abs(trade_shares) * execution_price * self.commission_rate
            current_cash -= trade_shares * execution_price + commission
            current_position += trade_shares

            asset_value = current_position * close_price
            total_value = current_cash + asset_value
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
            })

        df_records = pd.DataFrame(records).set_index('date')
        df_records['return'] = df_records['total_value'].pct_change()
        return df_records
