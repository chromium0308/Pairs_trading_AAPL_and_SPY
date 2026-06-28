"""Visualise the linear-regression model against the two stocks and run a
dollar-denominated 2025 backtest starting from $1000.

This builds directly on the existing pieces:

  * forecasting.py     -- lagged linear-regression (AR) forecaster
  * cointegration.py   -- the AAPL - beta*SPY spread and hedge ratio
  * trading_signals.py -- z-score + model-confirmed entry/exit logic

Two things are produced:

  1. model_vs_stocks_2025.png
     AAPL and SPY with their one-step-ahead LR forecasts, plus the spread
     (actual vs forecast). The model is trained ONLY on pre-2025 data so the
     2025 curve is a genuine out-of-sample forecast, not an in-sample fit.

  2. equity_2025.png + printed summary
     What $1000 would have become trading the AAPL/SPY spread through 2025.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from data_extraction import extract_data
from cointegration import spread
from forecasting import make_lagged_dataset, N_LAGS

PAIR = ("AAPL", "SPY")
ZSCORE_WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
START_CASH = 1000.0
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
# Round-trip-ish cost charged on the dollar notional traded, each time we open
# or close a leg. 5 bps (0.05%) is a reasonable all-in estimate of commission +
# bid/ask slippage for liquid names like AAPL/SPY. Bump it to stress-test.
COST_BPS = 5.0


def lr_forecast_oos(series, split_date, n_lags=N_LAGS):
    """One-step-ahead LR forecast, trained only on data before `split_date`.

    Returns a Series of predicted values aligned to the day each prediction is
    FOR. The model never sees the test period during training, so the forecast
    over the test window is honest (no look-ahead).
    """
    X, y = make_lagged_dataset(series, n_lags=n_lags)
    train_mask = X.index < split_date
    model = LinearRegression().fit(X[train_mask], y[train_mask])
    preds = pd.Series(model.predict(X), index=X.index)
    return preds, model


def generate_signals_oos(series, split_date):
    """Z-score + model-confirmed positions, with the LR model trained pre-2025.

    Mirrors trading_signals.generate_signals but the forecast comes from a model
    that only saw pre-test data, and positions are only opened in the test window.
    """
    mean = series.rolling(ZSCORE_WINDOW).mean()
    std = series.rolling(ZSCORE_WINDOW).std()
    z = ((series - mean) / std).rename("zscore")

    preds, _ = lr_forecast_oos(series, split_date)
    # predicted next value minus most recent known value (lag_1)
    X, _ = make_lagged_dataset(series)
    predicted_move = (preds - X["lag_1"]).rename("predicted_move")

    frame = pd.DataFrame({"value": series, "zscore": z,
                          "predicted_move": predicted_move}).dropna()

    positions, position = [], 0
    for _, row in frame.iterrows():
        in_test = row.name >= pd.Timestamp(split_date)
        if position == 0:
            if not in_test:
                positions.append(0)
                continue
            if row.zscore <= -ENTRY_Z and row.predicted_move > 0:
                position = 1          # spread too low -> expect rise
            elif row.zscore >= ENTRY_Z and row.predicted_move < 0:
                position = -1         # spread too high -> expect fall
        else:
            if abs(row.zscore) <= EXIT_Z:
                position = 0
        positions.append(position)

    frame["position"] = positions
    return frame


def dollar_backtest(frame, data, a, b, hedge_ratio):
    """Compound a $1000 account trading the spread, leg by leg, in dollars.

    On each entry we commit the current account equity as the AAPL-leg notional:
        shares_a = equity / price_a        (long if position +1, short if -1)
        shares_b = -position * hedge_ratio * shares_a   (the offsetting SPY leg)
    Holding shares_a of A and -hedge_ratio*shares_a of B exactly replicates one
    unit of the spread, so the daily P&L is shares_a * d(spread). The short leg is
    assumed financed on margin; we track net P&L on the long-leg notional.

    Every time the position changes we trade BOTH legs (open or close), and pay
    COST_BPS on the dollar notional that changes hands on each leg.
    """
    f = frame.loc[(frame.index >= TEST_START) & (frame.index <= TEST_END)].copy()
    d_spread = f["value"].diff()
    cost_rate = COST_BPS / 10_000.0

    def leg_notional(shares, day):
        """Total $ traded across both legs to open/close `shares` of A."""
        return abs(shares) * (data[a].loc[day] + hedge_ratio * data[b].loc[day])

    equity = START_CASH
    shares_a = 0.0
    total_cost = 0.0
    equity_curve, cost_curve, prev_pos = [], [], 0
    for day, row in f.iterrows():
        # capture today's move using yesterday's holding
        if prev_pos != 0 and not np.isnan(d_spread.loc[day]):
            equity += shares_a * d_spread.loc[day]

        pos = row["position"]
        if pos != prev_pos:                       # open and/or close => trade legs
            cost = cost_rate * leg_notional(shares_a, day)   # close current legs
            new_shares = pos * (equity / data[a].loc[day]) if pos != 0 else 0.0
            cost += cost_rate * leg_notional(new_shares, day)  # open new legs
            equity -= cost
            total_cost += cost
            shares_a = new_shares
        equity_curve.append(equity)
        cost_curve.append(total_cost)
        prev_pos = pos

    f["equity"] = equity_curve
    f["cum_cost"] = cost_curve
    return f


def summarise(f):
    final = f["equity"].iloc[-1]
    profit = final - START_CASH
    ret = profit / START_CASH
    entries = ((f["position"] != 0) & (f["position"].shift(1).fillna(0) == 0)).sum()
    days_in = (f["position"] != 0).mean()
    peak = f["equity"].cummax()
    max_dd = ((f["equity"] - peak) / peak).min()

    # buy-and-hold $1000 in each stock over the same window, for comparison
    print("\n" + "=" * 60)
    print(f"  AAPL/SPY spread strategy -- 2025 ({f.index[0].date()} to {f.index[-1].date()})")
    print("=" * 60)
    total_cost = f["cum_cost"].iloc[-1]
    gross_profit = profit + total_cost
    print(f"  starting capital:        ${START_CASH:,.2f}")
    print(f"  ending capital:          ${final:,.2f}")
    print(f"  gross profit (no costs): ${gross_profit:+,.2f}   ({gross_profit/START_CASH:+.1%})")
    print(f"  transaction costs ({COST_BPS:.0f} bps): -${total_cost:,.2f}")
    print(f"  net profit / loss:       ${profit:+,.2f}   ({ret:+.1%})")
    print(f"  round-trip trades taken: {entries}")
    print(f"  fraction of year in a trade: {days_in:.0%}")
    print(f"  worst drawdown:          {max_dd:.1%}")
    return profit, ret


def main():
    data = extract_data().dropna()
    a, b = PAIR
    pair_spread, hedge_ratio = spread(a, b, data)
    pair_spread = pair_spread.rename("value")

    # ---- (1) model vs the two stocks --------------------------------------
    aapl_pred, _ = lr_forecast_oos(data[a], TEST_START)
    spy_pred, _ = lr_forecast_oos(data[b], TEST_START)
    spread_pred, _ = lr_forecast_oos(pair_spread, TEST_START)

    win = lambda s: s.loc[(s.index >= TEST_START) & (s.index <= TEST_END)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), dpi=150, sharex=True)
    for ax, ticker, pred in [(axes[0], a, aapl_pred), (axes[1], b, spy_pred)]:
        ax.plot(win(data[ticker]).index, win(data[ticker]), color="steelblue",
                label=f"{ticker} actual")
        ax.plot(win(pred).index, win(pred), color="darkorange", linewidth=1,
                label=f"{ticker} LR forecast (out-of-sample)")
        ax.set_ylabel("price ($)")
        ax.set_title(f"{ticker}: actual vs linear-regression forecast (2025)")
        ax.legend()

    axes[2].plot(win(pair_spread).index, win(pair_spread), color="seagreen",
                 label="spread actual")
    axes[2].plot(win(spread_pred).index, win(spread_pred), color="crimson",
                 linewidth=1, label="spread LR forecast")
    axes[2].axhline(pair_spread.mean(), color="black", linewidth=0.8, label="long-run mean")
    axes[2].set_title(f"{a} - {hedge_ratio:.2f}*{b} spread: actual vs LR forecast (2025)")
    axes[2].set_ylabel("spread")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("model_vs_stocks_2025.png")
    plt.close()
    print("saved model_vs_stocks_2025.png")

    # ---- (2) $1000 backtest over 2025 -------------------------------------
    frame = generate_signals_oos(pair_spread, TEST_START)
    f = dollar_backtest(frame, data, a, b, hedge_ratio)
    profit, ret = summarise(f)

    # buy-and-hold benchmarks
    for ticker in (a, b):
        s = win(data[ticker])
        bh = START_CASH * (s.iloc[-1] / s.iloc[0])
        print(f"  [benchmark] $1000 buy & hold {ticker:<4}: ${bh:,.2f} "
              f"({bh/START_CASH-1:+.1%})")

    entries = f[(f["position"] != 0) & (f["position"].shift(1).fillna(0) == 0)]
    exits = f[(f["position"] == 0) & (f["position"].shift(1).fillna(0) != 0)]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    ax.plot(f.index, f["equity"], color="purple", label="account equity")
    ax.axhline(START_CASH, color="black", linewidth=0.8, label="$1000 start")
    ax.scatter(entries.index, entries["equity"], marker="^", color="green",
               s=80, zorder=5, label="enter trade")
    ax.scatter(exits.index, exits["equity"], marker="v", color="red",
               s=80, zorder=5, label="exit trade")
    ax.set_title(f"$1000 -> ${f['equity'].iloc[-1]:,.0f}: AAPL/SPY spread strategy, 2025")
    ax.set_ylabel("account value ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("equity_2025.png")
    plt.close()
    print("saved equity_2025.png")


if __name__ == "__main__":
    main()
