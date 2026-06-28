"""Turn the linear-regression forecasts into actual trade timing.

forecasting.py answers "what will the spread/ratio be tomorrow?". This module
answers the question you actually care about: "*when should I put a trade on,
and which way?*"

The idea has two ingredients that have to AGREE before we trade:

    1. STATISTICS -- a rolling z-score of the spread tells us how stretched the
       pair is right now. A z-score of +2 means the spread sits two standard
       deviations above its recent average, i.e. AAPL is unusually expensive
       relative to SPY. Mean-reverting pairs tend to snap back, so extreme
       z-scores are the candidate entry points.

    2. THE MODEL -- the linear-regression forecast from forecasting.py predicts
       the spread's next-day value. We only take a trade when the model agrees
       the spread is about to move back toward its mean. This filters out the
       times the spread is stretched but still trending away from us.

A trade on the spread (= AAPL - beta*SPY) means:

    z very HIGH  -> SHORT the spread (short AAPL, long beta*SPY) -- expect fall
    z very LOW   -> LONG  the spread (long  AAPL, short beta*SPY) -- expect rise
    z near ZERO  -> CLOSE the position -- the edge is gone

We compute the z-score on a *rolling* window rather than the whole history so
that, on any given day, only past data is used -- otherwise the backtest would
be cheating by using the future to define "normal".
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from data_extraction import extract_data
from cointegration import spread
from forecasting import make_lagged_dataset, N_LAGS

PAIR = ("AAPL", "SPY")
ZSCORE_WINDOW = 30   # trading days used to define "normal" for the z-score
ENTRY_Z = 2.0        # open a trade when |z| crosses this
EXIT_Z = 0.5         # close it when |z| falls back inside this


def rolling_zscore(series, window=ZSCORE_WINDOW):
    """How many rolling std-devs the series sits from its rolling mean.

    Uses only the trailing `window` days at each point, so it is honest about
    what you would actually have known in real time.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ((series - mean) / std).rename("zscore")


def forecast_next_day_move(series, n_lags=N_LAGS):
    """One-step-ahead forecast of the change in `series` for every day.

    We fit the same lagged linear model as forecasting.py, but here we want the
    *predicted direction*: positive means the model expects the value to rise
    tomorrow, negative means it expects a fall. Returns a Series aligned to the
    day the prediction is FOR (so it can be compared with that day's z-score).
    """
    X, y = make_lagged_dataset(series, n_lags=n_lags)
    model = LinearRegression().fit(X, y)
    predicted = pd.Series(model.predict(X), index=y.index)

    # predicted next value minus the most recent known value (lag_1)
    return (predicted - X["lag_1"]).rename("predicted_move")


def generate_signals(series, window=ZSCORE_WINDOW,
                     entry_z=ENTRY_Z, exit_z=EXIT_Z, use_model=True):
    """Build a day-by-day position (+1 long spread, -1 short, 0 flat).

    Walk forward through time. Enter when the spread is stretched AND (optionally)
    the model's forecast points back toward the mean. Hold the position until the
    z-score decays back inside `exit_z`, then go flat.
    """
    z = rolling_zscore(series, window)
    move = forecast_next_day_move(series) if use_model else None

    frame = pd.DataFrame({"value": series, "zscore": z})
    if use_model:
        frame["predicted_move"] = move
    frame = frame.dropna()

    positions = []
    position = 0
    for _, row in frame.iterrows():
        if position == 0:
            # Look for an entry. Model must agree the spread will revert.
            model_says_up = (not use_model) or row.get("predicted_move", 0) > 0
            model_says_down = (not use_model) or row.get("predicted_move", 0) < 0

            if row.zscore <= -entry_z and model_says_up:
                position = 1            # spread too low -> expect it to rise
            elif row.zscore >= entry_z and model_says_down:
                position = -1           # spread too high -> expect it to fall
        else:
            # In a trade -- exit once we have mean-reverted back near zero.
            if abs(row.zscore) <= exit_z:
                position = 0
        positions.append(position)

    frame["position"] = positions
    return frame


def backtest(frame):
    """Compute the P&L of holding `position` in the spread, one day at a time.

    Each day we earn the change in the spread times yesterday's position
    (you must already be in the trade to capture today's move). Returns the
    frame with daily and cumulative P&L columns added.
    """
    daily_change = frame["value"].diff()
    frame = frame.copy()
    frame["daily_pnl"] = frame["position"].shift(1) * daily_change
    frame["cumulative_pnl"] = frame["daily_pnl"].cumsum()
    return frame


def summarise(frame, label):
    """Print headline stats: how often we trade, hit-rate, and total P&L."""
    pnl = frame["daily_pnl"].dropna()
    trades = (frame["position"].diff().abs() > 0).sum()
    days_in_market = (frame["position"] != 0).mean()
    win_rate = (pnl[frame["position"].shift(1) != 0] > 0).mean()

    print(f"\n{label}")
    print(f"  position changes (entries+exits): {trades}")
    print(f"  fraction of days in a trade:       {days_in_market:6.1%}")
    print(f"  daily win rate while in a trade:   {win_rate:6.1%}")
    print(f"  total spread P&L over test:        {frame['cumulative_pnl'].iloc[-1]:8.2f}")


def plot_signals(frame, a, b, filename):
    """Plot the spread with entry (^) and exit (v) markers and the equity curve."""
    entries = frame[(frame["position"] != 0) & (frame["position"].shift(1) == 0)]
    exits = frame[(frame["position"] == 0) & (frame["position"].shift(1) != 0)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(frame.index, frame["value"], color="steelblue", label="spread")
    ax1.scatter(entries.index, entries["value"], marker="^", s=90,
                color="green", zorder=5, label="enter")
    ax1.scatter(exits.index, exits["value"], marker="v", s=90,
                color="red", zorder=5, label="exit")
    ax1.set_title(f"{a} - {b} spread: model-confirmed trade signals")
    ax1.set_ylabel("spread")
    ax1.legend()

    ax2.plot(frame.index, frame["cumulative_pnl"], color="purple")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("cumulative spread P&L (backtest)")
    ax2.set_ylabel("P&L")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def todays_call(frame, a, b):
    """Translate the latest row into a plain-English recommendation."""
    last = frame.iloc[-1]
    z = last.zscore
    when = frame.index[-1].date()

    if last.position == 1:
        action = f"LONG the spread  (BUY {a}, SELL {b})"
    elif last.position == -1:
        action = f"SHORT the spread (SELL {a}, BUY {b})"
    else:
        action = "STAY FLAT -- spread is near fair value, no edge"

    print(f"\nToday's call ({when}):")
    print(f"  spread z-score: {z:+.2f}  ->  {action}")


if __name__ == "__main__":
    data = extract_data().dropna()
    a, b = PAIR

    pair_spread, hedge_ratio = spread(a, b, data)
    pair_spread = pair_spread.rename("spread")
    ratio = (data[a] / data[b]).rename("ratio")

    print(f"Trade-timing for the {a}/{b} pair "
          f"(hedge ratio {hedge_ratio:.3f}, z-window {ZSCORE_WINDOW}d, "
          f"entry |z|>{ENTRY_Z}, exit |z|<{EXIT_Z}):")

    # Model the spread and the ratio; trade whichever signal you prefer.
    for series, name in [(pair_spread, "SPREAD"), (ratio, "RATIO")]:
        frame = backtest(generate_signals(series, use_model=True))
        summarise(frame, f"{name} (model-confirmed)")
        plot_signals(frame, a, b, f"signals_{name.lower()}_{a}_{b}.png")
        todays_call(frame, a, b)
