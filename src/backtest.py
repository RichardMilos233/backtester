import pandas as pd
class Backtest():
    def __init__(self, initial_capital, slippage_rate = 0.0005, commission_rate = 0.0005):
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate

    def run(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        target_signals = signals.shift(1)
        current_cash = self.initial_capital
        current_position = 0
        prev_total_value = self.initial_capital
        records = []
        for date in data.index[1:]:
            open_price = data['open'][date]
            close_price = data['close'][date]
            signal = target_signals[date]
            # signal: all in when 1, all out when 0
            if signal == 1:
                if current_position == 0:
                    cost_per_share = open_price * (1+self.slippage_rate) * (1+self.commission_rate)
                    trade_shares = current_cash // cost_per_share
                else:
                    trade_shares = 0
            elif signal == 0:
                trade_shares = - current_position
            else:
                trade_shares = 0
            
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
                'cash':current_cash,
                'asset_value': asset_value,
                'total_value': total_value,
                'daily_pnl': pnl,
                'trade_shares': trade_shares,
                'commission': commission,
            })
        return pd.DataFrame(records).set_index('date')