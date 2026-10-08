import math

import pandas as pd


class Backtest:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        slippage_rate: float = 0.0005,
        commission_rate: float = 0.0005,
        rebalance_tolerance: float = 0.01,
    ):
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.rebalance_tolerance = rebalance_tolerance

    def run(
        self,
        opens: pd.DataFrame,
        closes: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """Trade toward target weights at the next open.

        `opens`, `closes`, and `weights` share one column per asset. One asset
        is a one-column table. Weights are decided at the close and traded at
        the next open. A positive weight is a long, a negative weight is a short.
        Shares are whole numbers, truncated toward zero so the cash spent stays
        inside the theoretical budget. `tracking_gap` is the open-price dollar
        gap between those target weights and the shares actually held.
        """
        assets = list(weights.columns)
        missing = [
            asset for asset in assets
            if asset not in opens.columns or asset not in closes.columns
        ]
        if missing:
            raise KeyError(f"Missing open or close prices for: {missing}")

        target_weights = weights.shift(1)
        cash = self.initial_capital
        positions = {asset: 0.0 for asset in assets}
        prev_total_value = self.initial_capital
        records = []

        for date in weights.index:
            v_open = cash
            for asset in assets:
                open_price = opens.at[date, asset]
                if pd.notna(open_price):
                    v_open += positions[asset] * open_price

            day_commission = 0.0
            traded_value = 0.0
            n_trades = 0

            for asset in assets:
                weight = target_weights.at[date, asset]
                open_price = opens.at[date, asset]
                if pd.isna(weight) or pd.isna(open_price) or open_price == 0 or v_open <= 0:
                    continue

                delta_value = v_open * weight - positions[asset] * open_price
                if delta_value == 0 or (weight != 0 and abs(delta_value) < self.rebalance_tolerance * v_open):
                    continue

                if delta_value > 0:
                    execution_price = open_price * (1 + self.slippage_rate)
                else:
                    execution_price = open_price * (1 - self.slippage_rate)

                unit_cash = execution_price * (1 + self.commission_rate)
                if weight == 0:
                    trade_units = -positions[asset]
                else:
                    trade_units = math.trunc(delta_value / unit_cash)
                    # Never cross zero when the target stays on the same side.
                    if positions[asset] * weight > 0 and (positions[asset] + trade_units) * weight < 0:
                        trade_units = -positions[asset]
                        
                if trade_units == 0:
                    continue
                cash -= trade_units * execution_price + abs(trade_units) * execution_price * self.commission_rate
                positions[asset] += trade_units
                day_commission += abs(trade_units) * execution_price * self.commission_rate
                traded_value += abs(trade_units * open_price)
                n_trades += 1

            tracking_gap = 0.0
            for asset in assets:
                weight = target_weights.at[date, asset]
                open_price = opens.at[date, asset]
                if pd.isna(weight) or pd.isna(open_price):
                    continue
                tracking_gap += abs(positions[asset] * open_price - v_open * weight)

            asset_value = 0.0
            gross_exposure = 0.0
            for asset in assets:
                close_price = closes.at[date, asset]
                if pd.notna(close_price):
                    market_value = positions[asset] * close_price
                    asset_value += market_value
                    gross_exposure += abs(market_value)

            total_value = cash + asset_value
            daily_pnl = total_value - prev_total_value
            prev_total_value = total_value

            records.append({
                "date": date,
                "cash": cash,
                "asset_value": asset_value,
                "gross_exposure": gross_exposure / total_value if total_value else 0.0,
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "traded_value": traded_value,
                "tracking_gap": tracking_gap,
                "n_trades": n_trades,
                "commission": day_commission,
            })

        ledger = pd.DataFrame(records).set_index("date")
        ledger["return"] = ledger["total_value"].pct_change()
        return ledger
