# data/

`ohlcv_daily/<TICKER>.csv` — daily adjusted OHLCV (`date,open,high,low,close,volume`). Not committed.

```bash
.venv/bin/python scripts/data/fetch_ohlcv.py SPY QQQ TLT GLD IWM EFA
.venv/bin/python scripts/data/make_synthetic.py      # SYNTH_RW zero-drift control
```

Layer 1 scoring uses the last 1500 bars of SPY (TLT for `qcb_027`, IWM for `qcb_004`). The Layer 2 tasks and
their windows are listed in `benchmark/layer2_search/tasks.py`.
