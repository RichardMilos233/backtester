import yfinance as yf
import pandas as pd

ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", auto_adjust=False, actions=True)
# change to single index
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# rename columns
df.columns = df.columns.str.lower().str.replace(' ', '_')
# monotone in time
df = df.sort_index(ascending=True)
# remove duplicates and nan
df = df[~df.index.duplicated(keep='first')]
df = df.dropna()
# sanity check
invalid_mask = (
    (df['high'] < df['low']) |
    (df['high'] < df['open']) |
    (df['high'] < df['close']) |
    (df['low'] > df['open']) |
    (df['low'] > df['close']) |
    (df['volume'] < 0) |
    (df['close'] <= 0)
)
df = df[~invalid_mask]
# index / column names
df.index.name = 'date'
df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends']
df = df[cols]
# save to csv
df.to_csv(f"data/{ticker}.csv")