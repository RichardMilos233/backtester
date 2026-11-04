# QuantCode-Bench screen (56 cleaned tasks)

Source: `/tmp/qcb_clean56.json`. Screened against our sandbox: single asset, daily OHLCV, long-only, `build_strategy(df)` returns a position series in [0, 1], scored bar-by-bar against a golden implementation.

Conventions used in `params_to_pin`: all price tests use `close` unless stated; "size -> 1.0" means any fixed-capital sizing collapses to full position; stops/take-profits are evaluated on close and exit happens on the bar they trigger; re-entry allowed from the next bar. `LB>=200` flags lookbacks of 200+ bars (on ~1500 bars that eats 13%+ of the sample as warm-up).

| id | difficulty | source | usable | state_complexity | trade_freq | core_logic | params_to_pin |
|---|---|---|---|---|---|---|---|
| 4 | easy | stackexchange | yes | stateless | ok (LB>=200) | Long when close/close[-252]*100 > 100, flat when < 100 | equality (==100) -> hold prior state; 95% size -> 1.0 |
| 8 | hard | stackexchange | no | - | - | DQN trained online decides buy/sell/hold | - |
| 13 | medium | stackexchange | maybe | multi_state | ok | Close>EMA50; enter on RSI re-cross above 50; add 1 unit when RSI hits 70 then falls to 60 (max 4); partial exits at breakeven/+5%; full exit RSI<40 after >60 or close<EMA50 | 1h -> daily; unit = 0.25; breakeven partial fires only once and only after position was underwater; +5% measured on avg entry; "reaches 70" = RSI>=70 |
| 27 | medium | stackexchange | maybe | coupled_state | ok | Long when `longcondition` AND close[1] < last exit price; exit when `longcondition` false | `longcondition` undefined -> pin close > SMA(20); first trade exempt from exit-price test; size -> 1.0 |
| 39 | medium | stackexchange | yes | stateless | ok | Long on SMA10 crossing above SMA30, flat on cross below | none (drop "dynamic params" ask) |
| 42 | easy | reddit | yes | stateless | ok/low (LB>=200) | Long on close crossing above SMA200, flat on cross below | 1% size -> 1.0; duplicate of 241 |
| 44 | easy | reddit | maybe | multi_state | ok | Close>EMA50; enter on 1.5*ATR pullback from recent high; add up to 4 units on further pullbacks; TP 2.5*ATR, SL 1.5*ATR, exit close<EMA50 | "recent high" -> highest high since last cross above EMA50; unit = 0.25; TP/SL from avg entry; ATR frozen at entry |
| 49 | easy | stackexchange | no | - | - | Logistic regression on RSI/momentum, 80/20 train split | - |
| 52 | medium | synthetic | yes | stateless | ok | Long on MACD(12,26,9) line crossing above signal, flat on cross below | none; duplicate of 178 |
| 64 | medium | synthetic | yes | coupled_state | ok | Long when EMA 8>13>21>34>55 becomes fully ordered; exit when ordering breaks or SL -2% / TP +4% | "stop being ordered" -> min(EMA8,13,21) < max(EMA34,55); 2% size -> 1.0 |
| 89 | medium | stackexchange | maybe | coupled_state | ok | Long on SMA20 crossing above SMA50; exit rule depends on open P&L <10% vs >10% | `sell_below10()`/`sell_abv10()` undefined -> pin: <10% exit on SMA cross down; >10% exit on 5% drawdown from trade peak close |
| 92 | easy | reddit | yes | coupled_state | ok | Long when close > upper BB(20,2); exit at +6% TP or -3% SL | TP/SL on close; re-entry next bar if still above band; size -> 1.0 |
| 95 | medium | stackexchange | no | - | - | Count VaR exceedances, no positions | - |
| 101 | hard | reddit | yes | coupled_state | ok/high | Close>SMA9 and MACD hist crosses above 0 -> long; exit on 2*ATR14 trailing stop from highest high or MACD hist crossing below 0 | MACD params typo (12,9,26) -> (12,26,9); drop optional volume filter; initial stop = entry - 2*ATR; stop only ratchets up |
| 103 | medium | stackexchange | no | - | - | Up/down-day counts above/below EMA200 | - |
| 110 | medium | synthetic | yes | simple_state | ok/low | Long when RSI14 crosses above 30; exit when RSI crosses below 70 | "crosses 70 from above downward" kept literally (exit only after RSI has exceeded 70, then drops below); hold otherwise |
| 117 | easy | reddit | maybe | multi_state | ok | Pre-existing long; sell 50% at new ATH; exit close<SMA20 with volume>1.2x avg; stop close<SMA50; RSI divergence exit | needs entry rule -> pin close crosses above SMA50; ATH partial fires once per trade; drop divergence rule; 2% size -> 1.0 |
| 118 | hard | reddit | maybe | coupled_state | low | Enter when >=3 of 4 (RSI<30, close>SMA50, close<lower BB, MACD cross up); exit RSI>70, MACD cross down, or 2*ATR stop; size inverse to ATR | drop regime weighting; size -> min(1, 0.02*close/ATR14); 3-of-4 consensus pinned |
| 126 | easy | stackexchange | yes | simple_state | ok (LB>=200) | Long when SuperTrend(12,3) flips up and close>DEMA200; exit when SuperTrend flips down | SuperTrend uses hl2 and standard band ratchet; DEMA warm-up ~400 bars; size -> 1.0 |
| 129 | easy | reddit | maybe | coupled_state | low/ok | Breakout above falling Donchian(20) upper; TP = channel height; SL = lower band at entry | pick Donchian over BB; "falling" = upper[t-1] < upper[t-4] (3 lower highs) and same for lower; drop drawdown and volume filters; 4h -> daily |
| 137 | easy | stackexchange | yes | stateless | ok | Long on SMA10 crossing above SMA50, flat on cross below | none (signal-diff framing is implementation detail) |
| 162 | easy | reddit | maybe | coupled_state | ok | Close>SMA50, RSI dipped <50 within 5 bars and re-crosses 50, volume>SMA20, ATR filter, no gap; exits 1.5*ATR SL/TP, upper BB touch in 5 bars, RSI<40, fast +1*ATR in 3 bars | 1m -> daily; ATR frozen at entry; "bars since entry" counters pinned; size -> 1.0 |
| 163 | easy | reddit | yes | simple_state | low/ok (LB>=200) | Close>DEMA200 and close crosses above EMA10+3*ATR10 -> long; exit close crosses below EMA10-3*ATR10 or close<DEMA200 | "price" -> close; 1h -> daily |
| 168 | easy | reddit | maybe | multi_state | ok | Hold long; exit on +10% TP, 380 bars held, BB-width minus ATR% > 0.10, or event proxy (gain>5%, volume>1.5x, Stoch>80); re-enter after 10-bar wait if within 3% of exit and SMA50>SMA200 | initial entry -> first bar where SMA50>SMA200; Stochastic -> %K(14); 25% size -> 1.0 |
| 175 | easy | reddit | yes | coupled_state | high | Close<EMA20, RSI<40, ATR/close<4%, no -1% day -> long next bar; exit close>=entry+0.5*ATR or close>=EMA20; stop entry-min(ATR, 2%); 5-day time stop | ATR frozen at entry; "next bar" -> position set on signal bar close; size -> 1.0 |
| 178 | medium | stackexchange | yes | stateless | ok | MACD(12,26,9) cross up -> long, cross down -> flat | none; duplicate of 52 |
| 181 | hard | github | maybe | coupled_state | ok/high | 5-bar return >=3% and ATR14/close>=1% -> long; exit +6% TP / -3% SL | 1m -> daily; 100 USDT -> 1.0; ATR filter flag -> on |
| 182 | medium | stackexchange | maybe | simple_state | low (LB>=200, 756) | CAGR_3Y>=5% and close crosses above SMA200 -> long; exit cross below SMA200 or CAGR<0 | 756-bar lookback leaves ~750 bars; consider CAGR window 252 (1y) to keep the task |
| 183 | hard | stackexchange | maybe | coupled_state | high | Close > max(close[-10..-1]) -> long; exit close < min(close[-10..-1]) or -2% stop | 5m -> daily; window excludes current bar; 1% size -> 1.0 |
| 187 | medium | stackexchange | no | - | - | Always 100% long, rebalance every 12 weeks | - |
| 195 | hard | stackexchange | no | - | - | 3+3 limit-order grid around SMA20 | - |
| 198 | easy | stackexchange | maybe | simple_state | low (LB>=200, 260) | Long when Stoch %K(260) crosses above 85; exit when %K<1 | exit %K<1 almost never fires (1-3 trades); consider %K<15 or add SMA exit |
| 204 | easy | reddit | maybe | multi_state | low | SMA200 trend, BB squeeze (<0.08), 20-bar range <2.5*ATR, rising 5-bar low, MACD/RSI momentum, volume 1.5x; stop-entry at base_high+0.1*ATR; SL base_low-ATR; TP1 50% at +base_height (SL to base_high); TP2 +2*base_height; fail-safe; 30-day time exit | stop order -> close >= trigger; all filters on breakout bar; size -> 1.0 then 0.5 after TP1 |
| 207 | easy | stackexchange | yes | stateless | high | Long when close > upper BB(10, 0.674); flat when close < upper | none (entry-cross + exit-below collapses to close>top) |
| 212 | easy | reddit | yes | coupled_state | ok | Base long 1.0; sell 5% at upper BB touch with ATR>0.5*SMA(ATR,30); buy back at -5% from anchor high or +3% from sell price; 30-bar cooldown | positions in {0.95, 1.0}; "touch" -> high >= upper; anchor high = high of sell bar |
| 226 | hard | reddit | yes | coupled_state | ok (LB>=200, 252) | Close>SMA252 and SMA(ATR14,252)/close<=5% -> long; exit close<SMA252, -2% SL, or +5% TP | 10% size -> 1.0; SL/TP on close; re-entry blocked while ATR%>5% |
| 227 | medium | stackexchange | yes | stateless | high (continuous) (LB>=200, 252) | Daily weight = wealth-weighted mean over w in {0,0.1,...,1} of Π(1+w*r) over last 252 returns | fractions pinned already; no trade before 252 returns |
| 241 | easy | reddit | yes | stateless | ok/low (LB>=200) | Long on close crossing above SMA200, flat on cross below | none; duplicate of 42 |
| 243 | easy | reddit | yes | coupled_state | ok/high | Close > upper BB(20,2) after 3 closes at/below it and SMA20 rising -> long; exit close<upper band or +2% TP | TP on close; size -> 1.0 |
| 246 | medium | reddit | maybe | multi_state | low | >=15% 5-day drop with a 7x-volume down day; within 10 bars enter on up day, RSI crosses 30, volume>=0.8x, range>=3%, close>SMA20; exit TP2 (50% retrace) or SMA50 touch; stop L-ATR | 7x volume almost never fires on a single asset -> 3x; use single exit at TP2/SMA50; cap-day redefinition kept |
| 253 | easy | reddit | maybe | simple_state | low (LB>=200) | Close>SMA200, close<lower BB, RSI<=30, vol rising -> long; exit close>upper BB and RSI>=65 and vol falling | simultaneous exit conds rarely co-occur -> drop vol condition from exit; size -> 1.0 |
| 255 | easy | reddit | maybe | coupled_state | low | 20-bar return>=15%, EMA50 rising, low<=EMA20 with close>EMA20 and >prev high, RSI 40-60, volume 1.2x, R:R>=2 -> long; TP HH20; SL swing-low-0.5*ATR; exit close<EMA20 | 15% 20-bar return -> 5%; risk size -> 1.0; TP on high or close -> close |
| 256 | easy | reddit | yes | coupled_state | ok/high (LB>=200) | Close>SMA50 and SMA200, ATR/close<=1.5% -> long; TP min(+5%, +2.5*ATR); exit close<SMA50 | drop 10% equity drawdown kill switch; 5% size -> 1.0; ATR frozen at entry |
| 257 | easy | reddit | no | - | - | Library of 30 Buy/Sell/Hold signals | - |
| 259 | easy | reddit | maybe | multi_state | ok (LB>=200) | Lower BB touch, EMA50>SMA200, RSI<35, BB width 2-8% -> long; TP +0.5R; hard stop -1R (1.5*ATR); time exit; exit if BB width +20% | DTE 60/30/10/0 -> single 60-bar time stop; drop soft stop and drawdown filter; size -> 1.0 |
| 263 | medium | stackexchange | maybe | coupled_state | low (LB>=200) | Close>SMA200, golden cross within 5 bars, %B<=0.2, RSI 40-60, volume filters, ATR/close<=4% -> long; exits +20% TP, -10% SL, death cross, RSI>=70 or upper BB touch, ATR/close>6% | "golden cross within 5 bars" -> SMA50>SMA200; drop absolute volume 100000; size -> 1.0 |
| 267 | medium | reddit | maybe | multi_state | low | Crash filter (close <= 30% of 252-high); enter on close/RSI re-cross; pyramid up to 3 adds on EMA20 pullbacks; partial TPs +50%/+100%; trailing 3*ATR; 30-bar cooldown | 70% drawdown filter never fires -> 30%; unit = 0.25; a lot of pinning, lowest-priority maybe |
| 281 | easy | reddit | maybe | coupled_state | low | BB squeeze (W<0.75*SMA(W,20), ATR<SMA(ATR,20)), close near mid, RSI<35, 3-day volume 1.5x, close crosses above mid -> long; TP high>=entry+2*ATR; SL low<=min(lower BB, entry-1.5*ATR); exit close>upper and RSI>50; re-entry only after new squeeze | RSI<35 with cross above mid rarely coincides -> RSI<50; TP/SL on high/low kept (intrabar fill at level is fine for position series) |
| 283 | easy | reddit | maybe | multi_state | low | Buy at bar 0; each time position value hits initial*2^n sell half | positions 1, 0.5, 0.25...; after first halving price must 8x for next -> pin thresholds on price multiples of entry (2x, 4x, ...) |
| 284 | hard | stackexchange | yes | coupled_state | high | Close crosses above upper BB(20,2) and RSI>55 -> long; trailing stop = max(prev stop, highest high*0.99); exit close<=stop | highest includes entry bar high; 1 share -> 1.0; re-entry next bar |
| 286 | easy | reddit | maybe | multi_state | ok | Core long; sell 20% tranches on upper BB cross with RSI>60 and volume; 7-bar "term" with rolling; buy back below mid or RSI<45; full sell above upper+ATR; re-enter below mid with RSI<50 | heavy pinning: 1 tranche/day, max 5, roll = extend 7 bars; positions in {0,0.2,...,1}; lowest-priority maybe |
| 295 | easy | reddit | maybe | multi_state | ok | Close<=lower BB, RSI 25-45 -> long (<=50% capital); exit mid band or +1*ATR; SL 1.5*ATR; 25-bar time stop with roll; convert to core if close < lower BB-ATR or RSI<25; core exit close>SMA50 or RSI>60 | position 0.5; time stop at bar 25; roll = re-enter same bar if close<=lower BB; core = no time stop |
| 297 | easy | reddit | maybe | coupled_state | ok | Two strategies; pick CC-proxy: EMA20 crosses above SMA50 with close>EMA20 and volume>SMA20 -> long; TP upper BB touch or +k; after +0.5k trail to EMA20; SL close<EMA20-ATR or reverse cross; 20-bar time-out | port Strategy 2 only; k=clip(ATR/close,0.01,0.05) frozen at entry; size -> 1.0 |
| 317 | medium | tradingview | yes | coupled_state | high | Long at first trading day of month if close>SMA20; exit at 5th trading day or -2% stop | needs datetime index; first/5th = bar count within calendar month; stop on close |
| 338 | medium | tradingview | maybe | coupled_state | low | Long first trading day of January; exit first trading day of February or -2% stop | ~6 trades on 1500 bars; SMA20 is "for analysis" only -> drop |
| 353 | hard | tradingview | yes | simple_state | ok/low | Close>VWAP20 and MACD cross up and RSI<70 -> long; exit close<VWAP20 and MACD cross down and RSI>30 | rolling VWAP = sum(tp*vol,20)/sum(vol,20); exit "and" kept literally |

## Recommended to port first

Medium/hard with coupled or multi state, or otherwise distinctive mechanics:

1. **101** hard, coupled: ATR trailing stop plus MACD-histogram exit. Clean spec after fixing the MACD parameter typo.
2. **284** hard, coupled: 1% ratcheting trailing stop on highest high. High trade count, precise spec.
3. **226** hard, coupled: SMA252 regime with ATR% gate, 2%/5% stop/TP, re-entry blocking.
4. **64** medium, coupled: EMA ribbon ordering with SL/TP. One pin on "stop being ordered".
5. **317** medium, coupled: calendar entry/exit with stop. Tests datetime handling, high trade count.
6. **89** medium, coupled: profit-dependent exit rules. Needs two exit functions pinned but the state logic is good.
7. **227** medium, stateless fractional: universal-portfolio weights. The only continuous-position task.
8. **13** medium, multi: pyramiding with partial exits. Needs the breakeven rule fixed.
9. **44** easy, multi: ATR pullback scale-in to 4 units with TP/SL. Good multi-state after pinning "recent high".
10. **212** easy, coupled: 0.95/1.0 rotation with anchor and cooldown. Small position deltas but exact.
11. **175** easy, coupled: mean-reversion with three exits and a time stop. High trade count.
12. **27** medium, coupled: exit-price memory. Trivial once `longcondition` is pinned.

Good stateless/simple fillers already clean: 4, 39, 52/178, 126, 163, 207, 353, 92, 243, 256.

## Definitely no

- **8**: DQN training pipeline
- **49**: logistic regression training
- **95**: VaR exceedance counting
- **103**: pure statistics report
- **187**: always fully invested
- **195**: limit-order grid, intrabar
- **257**: signal library, no strategy

## Low-priority maybes

267 (70% crash filter), 286 (covered-call tranche rolling), 283 (price must 8x), 198 (exit never fires), 338 (~6 trades), 182 (756-bar lookback). Each needs a parameter change that alters the original strategy's character before it would produce a usable number of trades.
