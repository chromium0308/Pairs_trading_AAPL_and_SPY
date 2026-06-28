"""Linear-regression forecasting for a trading pair.

The goal of this module is to *predict the next day's value* of four things,
in increasing order of difficulty:

    1. the price RATIO   (AAPL / SPY)   -- the cleanest mean-reverting signal
    2. the SPREAD        (AAPL - beta*SPY)
    3. & 4. the STOCKS themselves (AAPL, SPY)

We use the simplest possible model -- ordinary linear regression -- but we
turn a *time series* into a regression problem with a standard trick called
"lagging": to predict the value at day t, we feed the model the values from
the previous few days (t-1, t-2, ...). A linear model on lagged inputs is
just an autoregressive (AR) model written in regression form:

    y[t] = c + w1*y[t-1] + w2*y[t-2] + ... + wn*y[t-n]

Everything below is built around that one idea.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from data_extraction import extract_data
from cointegration import spread

# Which pair to model. AAPL/SPY is the one that passed the cointegration test,
# so its ratio/spread genuinely mean-revert -- the best case for forecasting.
PAIR = ("AAPL", "SPY")
N_LAGS = 5          # use the previous 5 trading days as predictors
TRAIN_FRACTION = 0.8  # first 80% of history to train, last 20% to test


def make_lagged_dataset(series, n_lags=N_LAGS):
    """Turn one time series into a supervised (X, y) regression problem.

    For every day t we build a feature row of its `n_lags` previous values and
    a target equal to the value on day t itself:

        X row  = [y[t-1], y[t-2], ..., y[t-n_lags]]
        y value = y[t]

    The first `n_lags` rows have no complete history, so pandas fills them with
    NaN and we drop them.
    """
    df = pd.DataFrame({"target": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)      # value from `lag` days ago
    df = df.dropna()                              # drop rows with missing history

    X = df.drop(columns="target")
    y = df["target"]
    return X, y


def train_test_split_ordered(X, y, train_fraction=TRAIN_FRACTION):
    """Split chronologically -- NEVER shuffle time-series data.

    The model must be trained on the past and tested on the *future* it has
    never seen. Random shuffling would let it peek at future days and give
    dishonestly good scores.
    """
    split = int(len(X) * train_fraction)
    return X[:split], X[split:], y[:split], y[split:]


def fit_and_evaluate(series, name):
    """Fit a linear AR model on one series and report how well it predicts.

    Returns the trained model plus the held-out predictions so the caller can
    plot or trade on them.
    """
    X, y = make_lagged_dataset(series)
    X_train, X_test, y_train, y_test = train_test_split_ordered(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)               # learn the lag weights from the past

    predictions = model.predict(X_test)       # forecast the unseen future
    r2 = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print(f"{name:<16} R2={r2:6.3f}   RMSE={rmse:.4f}")
    return model, pd.Series(predictions, index=y_test.index)


if __name__ == "__main__":
    data = extract_data().dropna()
    a, b = PAIR

    # The four target series, easiest to hardest.
    ratio = (data[a] / data[b]).rename("ratio")
    pair_spread, _ = spread(a, b, data)
    pair_spread = pair_spread.rename("spread")

    print(f"Linear-regression forecasts for the {a}/{b} pair "
          f"({N_LAGS} lags, {int(TRAIN_FRACTION*100)}% train):\n")

    fit_and_evaluate(ratio, "ratio")
    fit_and_evaluate(pair_spread, "spread")
    fit_and_evaluate(data[a], f"{a} price")
    fit_and_evaluate(data[b], f"{b} price")
