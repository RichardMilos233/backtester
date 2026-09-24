from pathlib import Path

import pandas as pd


def fetch_dividends():
    # State Street 官方历史分红表：除息日、派息日和每股分红金额。
    url = (
        "https://www.ssga.com/library-content/products/fund-data/"
        "etfs/us/spdr-etf-historical-distributions.xlsx"
    )
    raw = pd.read_excel(url, sheet_name="dividend")
    spy = raw.loc[raw["TICKER"].astype(str).str.strip().eq("SPY")]
    events = spy[["EX-DATE", "PAYABLE DATE", "DIVIDEND ($)"]].rename(
        columns={
            "EX-DATE": "ex_date",
            "PAYABLE DATE": "pay_date",
            "DIVIDEND ($)": "dividend_per_share",
        }
    )
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events["pay_date"] = pd.to_datetime(events["pay_date"])
    events["dividend_per_share"] = pd.to_numeric(events["dividend_per_share"])
    if events["ex_date"].isna().any():
        raise ValueError("SPY 分红事件缺失除息日")
    events = events.loc[events["ex_date"] >= "2015-01-01"].sort_values("ex_date")

    if events.empty or events.isna().any().any():
        raise ValueError("SPY 分红事件为空或包含缺失数据")
    if events["ex_date"].duplicated().any():
        raise ValueError("SPY 分红事件存在重复除息日")
    if (events["pay_date"] < events["ex_date"]).any() or (events["dividend_per_share"] <= 0).any():
        raise ValueError("SPY 分红事件的日期或金额无效")

    output_path = Path(__file__).resolve().parents[1] / "data" / "SPY_dividends.csv"
    events.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"已保存 {len(events)} 条 SPY 分红事件至 {output_path}")


if __name__ == "__main__":
    fetch_dividends()
