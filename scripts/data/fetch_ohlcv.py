#!/usr/bin/env python3
"""
Fetch daily OHLCV for benchmark tickers from Yahoo's chart endpoint and write
adjusted CSVs in the same schema as data/ohlcv_daily/SPY.csv
(date,open,high,low,close,volume with all prices split/dividend adjusted).

  .venv/bin/python scripts/data/fetch_ohlcv.py QQQ TLT GLD IWM EFA
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "ohlcv_daily"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def fetch(ticker: str, start: str) -> pd.DataFrame:
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        js = json.loads(r.read())
    res = js["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame({
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"], "adj_close": adj,
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None))
    df.index.name = "date"
    df = df.dropna(subset=["close", "adj_close"])
    factor = df["adj_close"] / df["close"]
    out = pd.DataFrame({
        "open": df["open"] * factor, "high": df["high"] * factor,
        "low": df["low"] * factor, "close": df["adj_close"],
        "volume": df["volume"].fillna(0).astype("int64"),
    })
    return out[~out.index.duplicated(keep="last")].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in args.tickers:
        path = OUT_DIR / f"{t}.csv"
        if path.exists() and not args.force:
            print(f"{t}: exists, skip ({path})")
            continue
        df = fetch(t, args.start)
        df.to_csv(path, float_format="%.6f")
        print(f"{t}: {len(df)} rows {df.index[0].date()}..{df.index[-1].date()} -> {path}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
