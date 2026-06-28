"""Cointegration and stationarity tests for pairs-trading candidates.

Correlation tells us two stocks move together; cointegration tells us their
*spread* is mean-reverting, which is what actually makes a pair tradeable.
"""
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from data_extraction import extract_data

# Returns-correlation candidates surfaced in data_extraction.py
CANDIDATE_PAIRS = [
    ("DAL", "UAL"),
    ("GM", "F"),
    ("KO", "PEP"),
    ("AAPL", "SPY"),
    ("SPY", "MSFT"),
    ("SPY", "GS"),
]


def adf_test(series, name):
    """Augmented Dickey-Fuller test. p < 0.05 => stationary (mean-reverting)."""
    stat, pvalue, *_ = adfuller(series.dropna())
    verdict = "stationary" if pvalue < 0.05 else "non-stationary (has a unit root)"
    print(f"  ADF {name:<12} stat={stat:7.3f}  p={pvalue:.4f}  -> {verdict}")
    return pvalue


def spread(a, b, data):
    """Spread = a - hedge_ratio * b, where hedge_ratio comes from OLS of a on b."""
    model = sm.OLS(data[a], sm.add_constant(data[b])).fit()
    hedge_ratio = model.params[b]
    return data[a] - hedge_ratio * data[b], hedge_ratio


def plot_spread(a, b, data, filename=None):
    """Plot the pair's spread with its mean and +/-2 standard deviation bands."""
    pair_spread, _ = spread(a, b, data)
    mean, std = pair_spread.mean(), pair_spread.std()

    plt.figure(figsize=(11, 5), dpi=150)
    pair_spread.plot(label="spread", color="steelblue")
    plt.axhline(mean, color="black", label="mean")
    plt.axhline(mean + 2 * std, color="red", linestyle="--", label="+2 std")
    plt.axhline(mean - 2 * std, color="green", linestyle="--", label="-2 std")
    plt.title(f"{a} - {b} spread with 2 std-dev bands")
    plt.ylabel("spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename or f"spread_{a}_{b}.png")
    plt.close()


def cointegration_test(a, b, data):
    """Engle-Granger cointegration test. p < 0.05 => the pair is cointegrated."""
    _, pvalue, _ = coint(data[a], data[b])
    verdict = "COINTEGRATED" if pvalue < 0.05 else "not cointegrated"
    print(f"{a:>4} - {b:<4}  coint p={pvalue:.4f}  -> {verdict}")
    return pvalue


if __name__ == "__main__":
    data = extract_data().dropna()

    print("Engle-Granger cointegration test (p < 0.05 = tradeable spread):")
    for a, b in CANDIDATE_PAIRS:
        cointegration_test(a, b, data)

    print("\nADF stationarity tests for DAL / UAL:")
    adf_test(data["DAL"], "DAL price")
    adf_test(data["UAL"], "UAL price")

    pair_spread, hedge_ratio = spread("DAL", "UAL", data)
    print(f"\n  hedge ratio (DAL ~ UAL): {hedge_ratio:.3f}")
    adf_test(pair_spread, "DAL-UAL spread")
