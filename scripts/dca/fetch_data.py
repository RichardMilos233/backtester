"""Download daily history from Yahoo Finance into data/ohlcv_daily/<TICKER>.csv.

Prices are adjusted for splits and dividends (total return): open/high/low/close
are multiplied by Adj Close / Close. Output format matches
src/data.py::load_market_data (date,open,high,low,close,volume).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).resolve().parents[2] / 'data' / 'ohlcv_daily'

TICKERS = [
    'SPY', 'VOO', 'VTI', 'QQQ', 'BRK-B', 'BRK-A', 'VXUS', 'VEA', 'VWO', 'EFA', 'VT',
    'SCHD', 'VTV', 'VUG', 'RSP', 'XLK', 'VGT', 'SMH', 'NVDA', 'MSFT', 'AAPL', 'TSLA',
    'ARKK', 'BND', 'AGG', 'TLT', 'IEF', 'SHY', 'BIL', 'GLD', 'VFINX', 'VGTSX', 'VBMFX',
    '^GSPC', '^NDX', '^IXIC', '^VIX',
    'SPCX',  # Space Exploration Technologies Corp. (NASDAQ), IPO June 2026
]


def file_name(ticker: str) -> str:
    return (('_' + ticker[1:]) if ticker.startswith('^') else ticker) + '.csv'


def fetch(ticker: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).history(start='1980-01-01', auto_adjust=False, actions=False)
            if len(raw) > 0:
                return raw
            err = 'empty result'
        except Exception as exc:  # network / rate limit
            err = repr(exc)
        wait = 2 ** attempt
        print(f'  {ticker}: {err}; retry in {wait}s', file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f'failed to download {ticker}')


def adjust(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=['Close'])
    factor = (raw['Adj Close'] / raw['Close']) if 'Adj Close' in raw else 1.0
    out = pd.DataFrame({
        'open': raw['Open'] * factor,
        'high': raw['High'] * factor,
        'low': raw['Low'] * factor,
        'close': raw['Close'] * factor,
        'volume': raw['Volume'],
    })
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = 'date'
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out.dropna(subset=['close'])


def main(tickers: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker in tickers:
        df = adjust(fetch(ticker))
        name = file_name(ticker)
        df.to_csv(OUT_DIR / name)
        rows.append({'ticker': ticker, 'file': name, 'first_date': df.index[0].date(),
                     'last_date': df.index[-1].date(), 'rows': len(df)})
        print(f'{ticker}: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} rows)')
        time.sleep(0.5)
    pd.DataFrame(rows).to_csv(OUT_DIR / '_manifest.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:] or TICKERS)
