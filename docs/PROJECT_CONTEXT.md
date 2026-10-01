# Project context

## 1. Goal
Follow the course outline in Project_proposal_version_4.pdf, Sessions 1-10, and build a transparent object-oriented CTA backtester from scratch. No lookahead.

## 2. References
1. **Jump Trading (ICML 2026 Expo)**:
   - Keep each step a small loop that can be measured.
   - Decisions at time T use only information known at T. Include trading costs, slippage, and next-day execution.
2. **Google Research / MIT (arXiv:2512.08296)**:
   - On separable financial tasks, checking results in one place cut error amplification from 17.2x to 4.4x.
   - This project is the deterministic tool layer those checks would run on.

## 3. Sessions
- Session 1: turn a trading idea into a testable baseline.
- Session 2: a class-based backtester with cash, position, asset value, equity, PnL, and commission.
- Session 3-4: SMA and MACD signals, plus Sharpe, drawdown, skew, and turnover.
- Session 5: commission, slippage, and next-day fills.
- Session 6-7: four regimes from rolling volume and volatility, then conditional performance.
- Session 8: volatility scaling and the strategy revision.
- Session 9-10: out-of-sample machine learning (Ridge, Lasso, tree models) and whether it adds anything.

## 4. How we work
- Do not drop a finished file in one shot.
- Walk through the logic one step at a time, and let me write it.
