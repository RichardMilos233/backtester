"""Extend the data layer with long / broad histories.

Usage (repo root):  python scripts/dca/fetch_long_history.py [french] [jst] [yahoo] [net] [sanity]
No argument = all steps.  Re-runnable; everything lands in the git-ignored data/ dir.

  french : Ken French daily/monthly files -> data/french/*.csv and synthetic total-return
           price series FF_* in data/ohlcv_daily/
  jst    : Jorda-Schularick-Taylor Macrohistory (annual) -> data/jst/ (see note in fetch_jst)
  yahoo  : extra UCITS / SGX / mutual-fund / FX tickers (adjusted, same format as fetch_data.py)
  net    : <T>_NET30 / <T>_NET15 series (dividends taxed at withholding rate, from yfinance dividends)
  sanity : cross-checks, printed and written to data/french/_sanity.txt

Prices written to data/ohlcv_daily/<T>.csv as date,open,high,low,close,volume so
src/dca.py::load_prices([...]) works unchanged. data/ohlcv_daily/_manifest.csv is extended with
columns currency,notes (existing rows are kept; rows for tickers handled here are replaced).
"""
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OHLCV = ROOT / 'data' / 'ohlcv_daily'
FRENCH = ROOT / 'data' / 'french'
JST = ROOT / 'data' / 'jst'
MANIFEST = OHLCV / '_manifest.csv'
FRENCH_BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
JST_PAGE = 'https://www.macrohistory.net/database/'

YAHOO_EXTRAS = ['CSPX.L', 'VUAA.L', 'VWRA.L', 'IWDA.L', 'SWRD.L', 'XDWD.L', 'SPYL.L', 'S27.SI',
                'XUS.SI', 'XWR.SI', 'XND.SI', 'EUS.SI', 'ACWI', 'URTH', 'PRITX', 'VWIGX', 'IVV',
                'SGD=X', 'GBPUSD=X']
NET30 = ['VOO', 'SPY', 'IVV', 'VTI', 'VT', 'VXUS', 'QQQ', 'SCHD', 'VEA', 'ACWI', 'URTH']
NET15 = ['VOO', 'SPY']

# (zip name without _CSV.zip, local tidy name)
FRENCH_DAILY = {
    'ff3': 'F-F_Research_Data_Factors_daily',
    'ff5': 'F-F_Research_Data_5_Factors_2x3_daily',
    'mom': 'F-F_Momentum_Factor_daily',
    'dev_exus_3f': 'Developed_ex_US_3_Factors_Daily',
    'dev_3f': 'Developed_3_Factors_Daily',
}
FRENCH_MONTHLY = {
    'ff3': 'F-F_Research_Data_Factors',
    'ff5': 'F-F_Research_Data_5_Factors_2x3',
    'mom': 'F-F_Momentum_Factor',
}
IND12 = '12_Industry_Portfolios_daily'
BM = 'Portfolios_Formed_on_BE-ME_daily'


def file_name(ticker: str) -> str:
    return (('_' + ticker[1:]) if ticker.startswith('^') else ticker) + '.csv'


def safe_name(ticker: str) -> str:
    return ticker.replace('=', '_')  # SGD=X -> SGD_X.csv


# ---------------------------------------------------------------- manifest
def update_manifest(rows: list[dict]) -> None:
    cols = ['ticker', 'file', 'first_date', 'last_date', 'rows', 'currency', 'notes']
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in old:
            old[c] = ''
    new = pd.DataFrame(rows)
    old = old[~old['ticker'].isin(new['ticker'])]
    out = pd.concat([old[cols], new[cols]], ignore_index=True)
    out.to_csv(MANIFEST, index=False)


def write_prices(ticker: str, df: pd.DataFrame, currency: str = '', notes: str = '') -> dict:
    OHLCV.mkdir(parents=True, exist_ok=True)
    name = safe_name(file_name(ticker))
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.index.name = 'date'
    df.to_csv(OHLCV / name)
    return {'ticker': ticker, 'file': name, 'first_date': df.index[0].date(),
            'last_date': df.index[-1].date(), 'rows': len(df), 'currency': currency, 'notes': notes}


def synthetic_prices(ret: pd.Series) -> pd.DataFrame:
    """Total-return price index from simple returns: 100 on the first date, then compounded."""
    ret = ret.dropna().sort_index()
    px = 100.0 * (1 + ret).cumprod() / (1 + ret.iloc[0])
    return pd.DataFrame({'open': px, 'high': px, 'low': px, 'close': px, 'volume': 0})


# ---------------------------------------------------------------- Ken French
def http_get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and r.content:
                return r.content
            err = f'HTTP {r.status_code}'
        except Exception as exc:
            err = repr(exc)
        wait = 2 ** attempt
        print(f'  {url}: {err}; retry in {wait}s', file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f'failed to download {url}')


def french_text(name: str) -> str:
    z = zipfile.ZipFile(io.BytesIO(http_get(f'{FRENCH_BASE}{name}_CSV.zip')))
    member = z.namelist()[0]
    return z.read(member).decode('latin-1')


def french_sections(text: str) -> list[tuple[str, pd.DataFrame]]:
    """Split a multi-section French CSV into [(title, DataFrame)].

    A section = header line starting with ',' followed by data rows whose first field is a
    4/6/8 digit date, ended by a blank line. Title = last non-blank text line before the header.
    -99.99 / -999 become NaN. Values stay in percent (converted by the caller).
    """
    lines = text.splitlines()
    out, i, last_title = [], 0, ''
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith(','):
            cols = [c.strip() for c in ln.split(',')[1:]]
            recs, i = [], i + 1
            while i < len(lines) and lines[i].strip():
                parts = [p.strip() for p in lines[i].split(',')]
                if re.fullmatch(r'\d{4}|\d{6}|\d{8}', parts[0]):
                    recs.append(parts)
                i += 1
            if recs:
                df = pd.DataFrame([r[1:len(cols) + 1] for r in recs], columns=cols,
                                  index=[r[0] for r in recs]).apply(pd.to_numeric, errors='coerce')
                df = df.mask((df <= -99.0))  # -99.99 and -999
                out.append((last_title, df))
            continue
        if ln and not re.match(r'^\d', ln):
            last_title = ln
        i += 1
    return out


def to_dates(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df.index[0])
    if n == 8:
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
    elif n == 6:
        df.index = pd.to_datetime(df.index, format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        raise ValueError('annual section')
    df.index.name = 'date'
    return df


def first_section(text: str, title_contains: str = '', min_len_digits: int = 6) -> pd.DataFrame:
    for title, df in french_sections(text):
        if title_contains.lower() in title.lower() and len(df.index[0]) >= min_len_digits:
            return to_dates(df)
    raise ValueError(f'section {title_contains!r} not found')


def fetch_french() -> None:
    FRENCH.mkdir(parents=True, exist_ok=True)
    rows = []

    def tidy(df, name):
        (df / 100.0).round(8).to_csv(FRENCH / f'{name}.csv')

    daily = {}
    for key, nm in FRENCH_DAILY.items():
        print('French daily', nm)
        df = first_section(french_text(nm))
        daily[key] = df
        tidy(df, f'{key}_daily')
    for key, nm in FRENCH_MONTHLY.items():
        print('French monthly', nm)
        tidy(first_section(french_text(nm)), f'{key}_monthly')
    print('French industries / BE-ME')
    ind = first_section(french_text(IND12), 'Average Value Weighted Returns')
    tidy(ind, 'ind12_vw_daily')
    bm = first_section(french_text(BM), 'Average Value Weighted Returns')
    tidy(bm, 'bm_vw_daily')

    lag = 'French data lags ~1-2 months behind today; '

    def add(ticker, ret_pct, note, currency='USD'):
        ret = ret_pct / 100.0
        nmiss = int(ret.isna().sum())
        n = note + (f' {nmiss} missing days dropped (not filled).' if nmiss else '')
        rows.append(write_prices(ticker, synthetic_prices(ret), currency, lag + n))
        print(f'  {ticker}: {rows[-1]["first_date"]} -> {rows[-1]["last_date"]}')

    f3 = daily['ff3']
    add('FF_MKT', f3['Mkt-RF'] + f3['RF'], 'Synthetic: CRSP US market total return = Mkt-RF+RF (Ken French daily).')
    add('FF_RF', f3['RF'], 'Synthetic: 1-month T-bill (cash), Ken French RF.')
    for col in ind.columns:
        note = f'Synthetic: 12-industry value-weighted daily return "{col}".'
        add(f'FF_IND_{col}', ind[col], note)
    add('FF_TECH', ind['BusEq'], 'Synthetic: 12-industry "BusEq" (business equipment incl. computers/software/electronics); same as FF_IND_BusEq.')
    add('FF_VALUE', bm['Hi 30'], 'Synthetic: value-weighted Hi 30 B/M portfolio.')
    add('FF_GROWTH', bm['Lo 30'], 'Synthetic: value-weighted Lo 30 B/M portfolio.')
    dx, dv = daily['dev_exus_3f'], daily['dev_3f']
    add('FF_DEVXUS', dx['Mkt-RF'] + dx['RF'], 'Synthetic: Developed ex-US market (USD) = Mkt-RF+RF.')
    add('FF_DEV', dv['Mkt-RF'] + dv['RF'], 'Synthetic: Developed incl. US market (USD) = Mkt-RF+RF.')
    update_manifest(rows)


# ---------------------------------------------------------------- JST
def fetch_jst() -> None:
    """Download the JST xlsx and write data/jst/jst_equity.csv.

    NOTE: macrohistory.net answers the download link with a 302 to sfff...jimcontent.com (a Jimdo CDN host).
    In the sandbox that host is blocked by the egress proxy. If the download fails, put the file
    (JSTdatasetR6.xlsx) in data/jst/ manually and re-run this step; it will be tidied from disk.
    """
    JST.mkdir(parents=True, exist_ok=True)
    local = sorted(JST.glob('JSTdataset*.xlsx'))
    if not local:
        try:
            html = requests.get(JST_PAGE, timeout=60).text
            m = re.search(r'href="(/app/download/\d+/JSTdataset[^"]*\.xlsx[^"]*)"', html)
            url = 'https://www.macrohistory.net' + m.group(1)
            data = http_get(url, retries=2)
            path = JST / re.search(r'(JSTdataset[^/?]*\.xlsx)', url).group(1)
            path.write_bytes(data)
            local = [path]
        except Exception as exc:
            print(f'JST download failed ({exc!r}); place the xlsx in {JST} manually.', file=sys.stderr)
            return
    raw = pd.read_excel(local[-1])
    raw.columns = [str(c).strip() for c in raw.columns]
    bill = 'bill_rate' if 'bill_rate' in raw else 'stir'  # R6 names the short rate 'stir'
    keep = [c for c in ['country', 'iso', 'year', 'eq_tr', 'cpi', 'xrusd', 'gdp', 'rgdpmad', 'pop', bill] if c in raw]
    df = raw[keep].rename(columns={bill: 'bill_rate'}).sort_values(['iso', 'year']).reset_index(drop=True)
    g = df.groupby('iso')
    # xrusd = local currency units per USD, so USD value of 1 local unit = 1/xrusd.
    df['eq_tr_usd'] = (1 + df['eq_tr']) * g['xrusd'].shift(1) / df['xrusd'] - 1
    df['eq_tr_real'] = (1 + df['eq_tr']) * g['cpi'].shift(1) / df['cpi'] - 1
    df.to_csv(JST / 'jst_equity.csv', index=False)
    spans = []
    for iso, d in df.groupby('iso'):
        miss = d[d['eq_tr'].isna()]['year'].tolist()
        spans.append(f'{iso}: eq_tr first {d.dropna(subset=["eq_tr"])["year"].min()}, missing years {miss}')
    (JST / '_missing_spans.txt').write_text('\n'.join(spans) + '\n')
    print('\n'.join(spans))


# ---------------------------------------------------------------- Yahoo
def yf_raw(ticker: str, retries: int = 5) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).history(start='1970-01-01', auto_adjust=False, actions=True)
            if len(raw) > 0:
                return raw
            err = 'empty result'
        except Exception as exc:
            err = repr(exc)
        wait = 2 ** attempt * 2
        print(f'  {ticker}: {err}; retry in {wait}s', file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f'failed to download {ticker}')


def yf_currency(ticker: str) -> str:
    import yfinance as yf
    for _ in range(3):
        try:
            c = yf.Ticker(ticker).fast_info.get('currency')
            if c:
                return str(c)
        except Exception:
            pass
        time.sleep(2)
    try:
        return str(yf.Ticker(ticker).info.get('currency', ''))
    except Exception:
        return ''


def clean_index(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=['Close']).copy()
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    raw.index.name = 'date'
    return raw[~raw.index.duplicated(keep='last')].sort_index()


def adjust_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = clean_index(raw)
    f = raw['Adj Close'] / raw['Close']
    return pd.DataFrame({'open': raw['Open'] * f, 'high': raw['High'] * f, 'low': raw['Low'] * f,
                         'close': raw['Close'] * f, 'volume': raw['Volume']}).dropna(subset=['close'])


_RAW: dict[str, pd.DataFrame] = {}


def raw_cached(ticker: str) -> pd.DataFrame:
    if ticker not in _RAW:
        _RAW[ticker] = clean_index(yf_raw(ticker))
        time.sleep(1.5)  # be gentle with Yahoo
    return _RAW[ticker]


def fetch_yahoo() -> None:
    rows = []
    for t in YAHOO_EXTRAS:
        try:
            raw = raw_cached(t)
        except RuntimeError as exc:
            print(f'{t}: FAILED {exc}', file=sys.stderr)
            continue
        df = adjust_raw(raw)
        cur = yf_currency(t)
        notes = ''
        if cur in ('GBp', 'GBP', 'GBX'):
            notes = f'WARNING: trades in {cur} (pence/pounds, not USD).'
        elif t.endswith('.SI') and cur != 'SGD':
            notes = f'SGX line quoted in {cur}'
        if t.endswith('=X'):
            notes = 'FX rate (quote currency in currency column), not a total-return asset; Yahoo history only from 2003-12.'
        if len(df) < 250:
            notes = (notes + ' ' if notes else '') + f'Only {len(df)} rows on Yahoo (short history).'
        rows.append(write_prices(t, df, cur, notes))
        print(f'{t}: {rows[-1]["first_date"]} -> {rows[-1]["last_date"]} ({len(df)} rows) {cur} {notes}')
    update_manifest(rows)


# ---------------------------------------------------------------- net-of-withholding series
def net_series(ticker: str, keep: float) -> pd.DataFrame:
    """Daily return (close_t + keep*div_t)/close_{t-1} - 1 from split-adjusted, dividend-unadjusted closes."""
    raw = raw_cached(ticker)
    close = raw['Close']
    div = raw['Dividends'].reindex(close.index).fillna(0.0)
    ret = (close + keep * div) / close.shift(1) - 1
    px = 100.0 * (1 + ret.fillna(0.0)).cumprod()
    factor = px / close
    return pd.DataFrame({'open': raw['Open'] * factor, 'high': raw['High'] * factor,
                         'low': raw['Low'] * factor, 'close': px, 'volume': raw['Volume']}).dropna(subset=['close'])


def cagr(close: pd.Series) -> float:
    yrs = (close.index[-1] - close.index[0]).days / 365.25
    return (close.iloc[-1] / close.iloc[0]) ** (1 / yrs) - 1


def fetch_net() -> list[str]:
    rows, report = [], []
    jobs = [(t, 0.70, 'NET30') for t in NET30] + [(t, 0.85, 'NET15') for t in NET15]
    for t, keep, tag in jobs:
        df = net_series(t, keep)
        rows.append(write_prices(f'{t}_{tag}', df, 'USD',
                                 f'Synthetic: {t} with {(1 - keep):.0%} of each dividend withheld; '
                                 f'(close+{keep:.2f}*div)/prev close - 1, split-adjusted unadjusted-for-div closes.'))
        if tag == 'NET30':
            gross = net_series(t, 1.0)
            adj = adjust_raw(raw_cached(t))
            common = gross.index.intersection(adj.index)
            g, a, n = gross.loc[common, 'close'], adj.loc[common, 'close'], df.loc[common, 'close']
            line = (f'{t:5s} {common[0].date()}->{common[-1].date()}  CAGR adj={cagr(a):.4%} gross_rebuilt={cagr(g):.4%} '
                    f'(rebuilt-adj {1e4 * (cagr(g) - cagr(a)):+.1f}bp/yr)  NET30={cagr(n):.4%}  '
                    f'gap gross-NET30={1e4 * (cagr(g) - cagr(n)):.1f}bp/yr')
            report.append(line)
            print(line)
    update_manifest(rows)
    OHLCV.parent.joinpath('french').mkdir(exist_ok=True)
    (FRENCH / '_net_sanity.txt').write_text('\n'.join(report) + '\n')
    return report


# ---------------------------------------------------------------- sanity
def load_close(t: str) -> pd.Series:
    return pd.read_csv(OHLCV / safe_name(file_name(t)), index_col='date', parse_dates=True)['close']


def ann_rets(c: pd.Series) -> pd.Series:
    ye = c.resample('YE').last()
    return ye.pct_change().dropna()


def maxdd(c: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    dd = c / c.cummax() - 1
    trough = dd.idxmin()
    return dd.min(), c.loc[:trough].idxmax(), trough


def sanity() -> None:
    out = []

    def p(s=''):
        print(s)
        out.append(s)

    mkt = load_close('FF_MKT')
    d, pk, tr = maxdd(mkt)
    p(f'FF_MKT {mkt.index[0].date()}->{mkt.index[-1].date()} CAGR {cagr(mkt):.3%}; max DD {d:.2%} ({pk.date()} -> {tr.date()})')
    m1929 = mkt.loc['1929-01-01':'1933-12-31']
    d2, pk2, tr2 = maxdd(m1929)
    p(f'  1929-33 window max DD {d2:.2%} ({pk2.date()} -> {tr2.date()})')

    for other in ['VFINX', 'SPY']:
        o = load_close(other)
        common = mkt.index.intersection(o.index)
        a = ann_rets(mkt.loc[common])
        b = ann_rets(o.loc[common])
        j = pd.concat([a, b], axis=1, keys=['FF_MKT', other]).dropna().iloc[1:]  # first partial year pct_change base ok
        gap = j['FF_MKT'] - j[other]
        p(f'FF_MKT vs {other}: {len(j)} full calendar years {j.index[0].year}-{j.index[-1].year}; '
          f'mean gap {gap.mean() * 100:+.2f}pp, mean |gap| {gap.abs().mean() * 100:.2f}pp, max |gap| {gap.abs().max() * 100:.2f}pp, corr {j.corr().iloc[0, 1]:.4f}; '
          f'CAGR on common dates FF_MKT {cagr(mkt.loc[common]):.3%} vs {other} {cagr(o.loc[common]):.3%}')

    tech = load_close('FF_TECH')
    for other in ['_NDX', 'QQQ']:
        t = '^NDX' if other == '_NDX' else other
        o = load_close(t)
        common = tech.index.intersection(o.index)
        common = common[common >= '1999-03-10']
        mt = tech.loc[common].resample('ME').last().pct_change().dropna()
        mo = o.loc[common].resample('ME').last().pct_change().dropna()
        p(f'FF_TECH vs {t} monthly return corr (from {common[0].date()}): {mt.corr(mo):.4f} ({len(mt)} months)')

    dx = load_close('FF_DEVXUS')
    for t in ['EFA', 'VEA']:
        o = load_close(t)
        common = dx.index.intersection(o.index)
        a = ann_rets(dx.loc[common])
        b = ann_rets(o.loc[common])
        j = pd.concat([a, b], axis=1, keys=['FF_DEVXUS', t]).dropna()
        gap = j['FF_DEVXUS'] - j[t]
        p(f'FF_DEVXUS vs {t}: {j.index[0].year}-{j.index[-1].year} ({len(j)} yrs) mean gap {gap.mean() * 100:+.2f}pp, '
          f'mean |gap| {gap.abs().mean() * 100:.2f}pp, corr {j.corr().iloc[0, 1]:.4f}; CAGR FF {cagr(dx.loc[common]):.3%} vs {t} {cagr(o.loc[common]):.3%}')

    jst_path = JST / 'jst_equity.csv'
    if jst_path.exists():
        jst = pd.read_csv(jst_path)
        us = jst[jst['iso'] == 'USA'].set_index('year')['eq_tr']
        ff = (mkt.resample('YE').last().pct_change().dropna())
        ff.index = ff.index.year
        j = pd.concat([us, ff], axis=1, keys=['JST_USA', 'FF_MKT']).dropna()
        j = j.loc[1927:]
        j = j[j.index < mkt.index[-1].year]
        p(f'JST USA eq_tr vs FF_MKT calendar years {j.index[0]}-{j.index[-1]} ({len(j)}): corr {j.corr().iloc[0, 1]:.4f}, '
          f'mean gap JST-FF {100 * (j.JST_USA - j.FF_MKT).mean():+.2f}pp')
    else:
        p('JST file missing: JST sanity skipped')

    cs, vo = load_close('CSPX.L'), load_close('VOO')
    common = cs.index.intersection(vo.index)
    p(f'CSPX.L vs VOO on {len(common)} common dates {common[0].date()}->{common[-1].date()}: CAGR {cagr(cs.loc[common]):.3%} vs {cagr(vo.loc[common]):.3%}; '
      f'gap {1e4 * (cagr(cs.loc[common]) - cagr(vo.loc[common])):+.0f}bp/yr (London/NY holiday gaps and non-sync closes included)')

    man = pd.read_csv(MANIFEST)
    new = man[man['ticker'].str.startswith(('FF_',)) | man['ticker'].str.contains('_NET') | man['ticker'].isin(YAHOO_EXTRAS)]
    p('\nLast dates of new series:')
    for _, r in new.iterrows():
        p(f'  {r.ticker:14s} {r.first_date} -> {r.last_date}  {r.currency}')
    FRENCH.mkdir(parents=True, exist_ok=True)
    (FRENCH / '_sanity.txt').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    steps = sys.argv[1:] or ['french', 'jst', 'yahoo', 'net', 'sanity']
    for s in steps:
        print(f'===== {s}')
        {'french': fetch_french, 'jst': fetch_jst, 'yahoo': fetch_yahoo, 'net': fetch_net, 'sanity': sanity}[s]()
