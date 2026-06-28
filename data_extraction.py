import os
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from dotenv import load_dotenv

# FMP API key — add a line `FMP_API_KEY=your_key` to a .env file
load_dotenv()
API_KEY = os.environ["FMP_API_KEY"]
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

TICKERS = [
    "DPZ", "AAPL", "GOOG", "AMD", "SPY", "NFLX", "BA", "INTC", "WMT",
    "GS", "XOM", "NKE", "META", "BRK-B", "MSFT",
    "KO", "PEP", "GM", "F", "DAL", "UAL",
]


def fetch_close_prices(ticker, start, end):
    """Return a Series of daily close prices for one ticker, or None if unavailable."""
    response = requests.get(
        BASE_URL,
        params={"symbol": ticker, "from": start, "to": end, "apikey": API_KEY},
    )
    if not response.ok:
        print(f"  skipping {ticker}: {response.status_code} (not available on this plan)")
        return None

    prices = pd.DataFrame(response.json())
    if prices.empty:
        print(f"  skipping {ticker}: no data returned")
        return None

    prices["date"] = pd.to_datetime(prices["date"])
    return prices.set_index("date")["close"].rename(ticker).sort_index()


def extract_data(tickers=TICKERS, years=6):
    """Fetch daily closes for every ticker and combine them into one DataFrame."""
    end = date.today()
    start = end - timedelta(days=365 * years)

    closes = [fetch_close_prices(t, start, end) for t in tickers]
    return pd.concat([s for s in closes if s is not None], axis=1)


def correlation_matrix(data, returns=False):
    """Absolute correlation matrix, computed on daily returns or raw prices."""
    series = data.pct_change().dropna() if returns else data
    return series.corr().abs()


def save_heatmap(corr_matrix, folder):
    """Save an annotated heatmap of the correlation matrix."""
    plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(corr_matrix, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "heatmap.png"))
    plt.close()


def high_corr_pairs(corr_matrix, threshold=0.9):
    """Return [(ticker_a, ticker_b, corr), ...] for pairs above the threshold."""
    pairs = [
        (a, b, corr_matrix.loc[a, b])
        for i, a in enumerate(corr_matrix.columns)
        for b in corr_matrix.columns[i + 1:]
        if corr_matrix.loc[a, b] > threshold
    ]
    return sorted(pairs, key=lambda p: -p[2])


def save_pair_plots(data, pairs, folder):
    """Save each highly-correlated pair as its own image (prices rebased to 100)."""
    pair_folder = os.path.join(folder, "pair_plots")
    os.makedirs(pair_folder, exist_ok=True)
    rebased = data / data.iloc[0] * 100

    for a, b, corr in pairs:
        plt.figure(figsize=(8, 5), dpi=150)
        rebased[[a, b]].plot(ax=plt.gca())
        plt.title(f"{a} vs {b}  (corr = {corr:.2f})")
        plt.ylabel("price (rebased to 100)")
        plt.tight_layout()
        plt.savefig(os.path.join(pair_folder, f"{a}_{b}.png"))
        plt.close()


def run_analysis(data, folder, returns=False, threshold=0.65):
    """Save the correlation matrix, heatmap, and pair plots above the threshold."""
    os.makedirs(folder, exist_ok=True)
    label = "returns" if returns else "raw price"

    corr = correlation_matrix(data, returns=returns)
    corr.to_csv(os.path.join(folder, "correlation_matrix.csv"))
    save_heatmap(corr, folder)

    pairs = high_corr_pairs(corr, threshold=threshold)
    save_pair_plots(data, pairs, folder)

    print(f"\n{label} correlation -> {folder}/  ({len(pairs)} pairs > {threshold})")
    for a, b, corr_value in pairs:
        print(f"  {a} - {b}: {corr_value:.2f}")


if __name__ == "__main__":
    data = extract_data()
    print("Daily close prices:")
    print(data)

    run_analysis(data, "raw_correlation", returns=False, threshold=0.65)
    run_analysis(data, "returns_correlation", returns=True, threshold=0.65)
