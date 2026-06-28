# Pairs Trading Project

A small, end-to-end **statistical-arbitrage** pipeline: find two stocks that move
together, confirm their spread is mean-reverting, forecast it with linear
regression, turn that into trade signals, and backtest the strategy in dollars.

The headline pair is **AAPL / SPY** — the one that passes the cointegration test.

## Pipeline

| Step | File | What it does |
|------|------|--------------|
| 1. Data | `data_extraction.py` | Pulls ~6 years of daily closes from the FMP API and builds correlation matrices, heatmaps, and pair plots (raw price + daily returns). |
| 2. Cointegration | `cointegration.py` | Engle–Granger cointegration + ADF stationarity tests. Computes the OLS hedge ratio and the spread `A − β·B`. |
| 3. Forecasting | `forecasting.py` | Linear-regression **autoregressive (AR)** model on lagged values (5 lags) to predict the next-day ratio, spread, and stock prices. |
| 4. Signals | `trading_signals.py` | Rolling **z-score** of the spread for entry/exit timing, gated by the LR forecast agreeing on the direction of reversion. |
| 5. Backtest | `backtest_2025.py` | Out-of-sample 2025 backtest of a **$1000** account, in dollars, with transaction costs. |

## The model

To predict a time series with regression, each day's value is modeled from its
previous few days (lagging) — an AR model in regression form:

```
y[t] = c + w1·y[t-1] + w2·y[t-2] + ... + wn·y[t-n]
```

Trained chronologically (never shuffled) so the model is always tested on the
future it hasn't seen. Forecasting the raw **price** is nearly trivial
(tomorrow ≈ today), so the strategy trades the mean-reverting **spread** via its
z-score rather than the price level itself.

## Trading rules (AAPL/SPY spread, β ≈ 0.42)

- z ≤ −2 **and** model predicts a rise → **LONG** spread (buy AAPL, sell β·SPY)
- z ≥ +2 **and** model predicts a fall → **SHORT** spread (sell AAPL, buy β·SPY)
- |z| ≤ 0.5 → **CLOSE** (edge gone, go flat)

The z-score uses a trailing 30-day window so only past data informs each day.

## 2025 backtest (out-of-sample, $1000 start)

Model trained only on pre-2025 data; signals and P&L generated for 2025.

| Metric | Value |
|--------|-------|
| Gross profit (no costs) | +$133.75 (+13.4%) |
| Transaction costs (5 bps) | −$17.65 |
| **Net ending capital** | **$1,116.10** |
| **Net profit** | **+$116.10 (+11.6%)** |
| Round-trip trades | 8 |
| Time in market | 43% of the year |
| Worst drawdown | −10.8% |
| Benchmark: buy & hold AAPL | +11.5% |
| Benchmark: buy & hold SPY | +16.6% |

The strategy roughly matches buy-and-hold AAPL and trails SPY, but with market
exposure only **43%** of the time — the appeal of a market-neutral pair trade is
lower risk, not necessarily higher raw return.

### Cost & sizing assumptions

- On each entry, current equity is committed as the AAPL-leg notional; the SPY
  leg is sized by the hedge ratio. Profits compound into the next trade's size.
- The short leg is assumed financed on margin; net P&L is tracked on the long-leg
  notional.
- Costs (`COST_BPS`, default 5 bps) are charged on both legs each time a position
  opens or closes, and compound. Bump `COST_BPS` to stress-test turnover.

## Running it

```bash
# add your FMP key to a .env file:  FMP_API_KEY=your_key
python data_extraction.py     # correlations, heatmaps, pair plots
python cointegration.py       # cointegration + stationarity tests
python forecasting.py         # LR forecast accuracy (R², RMSE)
python trading_signals.py     # signals + spread-unit backtest, today's call
python backtest_2025.py       # $1000 dollar backtest with costs + charts
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`,
`requests`, `python-dotenv`, `seaborn`.

## Outputs

- `raw_correlation/`, `returns_correlation/` — correlation matrices, heatmaps, pair plots
- `spread_AAPL_SPY.png` — spread with ±2 std-dev bands
- `signals_spread_AAPL_SPY.png`, `signals_ratio_AAPL_SPY.png` — trade signals + equity
- `model_vs_stocks_2025.png` — AAPL, SPY, and spread vs their LR forecasts (2025)
- `equity_2025.png` — the $1000 account curve through 2025

## Caveats

Educational, not investment advice. No borrow fees, financing, or market-impact
modeling; single pair over a single year; results are sensitive to the cost
assumption and the entry/exit thresholds.
