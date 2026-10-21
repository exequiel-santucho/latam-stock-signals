"""Step 4 - Trading simulations on the out-of-sample predictions (all amounts in USD).

1. Vectorised simulations (course approach, one row = one possible trade):
   sim1_*  : $100 on every signal with probability above a threshold
   sim2_*  : $100 on the top-k stocks of each day
   sim3_*  : sim2 only when the stock's market is in a bull regime (>50% of stocks above SMA200)
   sim4_*  : long-short: long the top-k and short the bottom-k of each day (market neutral)
   For each strategy and prediction: fields sim{N}_gross_rev / sim{N}_fees / sim{N}_net_rev,
   and a summary with the approximate CAGR on the capital needed to run it.
2. Exact simulation (day by day, `exact_simulation`): starts with $100k, invests 1/20 of the equity
   each day in the top-k signals, holds 20 trading days, reinvests gains, optional stop-loss /
   take-profit (checked on daily closes), position size proportional to the probability, max one
   open position per ticker, and an optional market-regime filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg

H = cfg.HORIZON
FEE = cfg.FEE
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Vectorised simulations
# ---------------------------------------------------------------------------
def rank_in_day(df: pd.DataFrame, pred: str, ascending: bool = False) -> pd.Series:
    return df.groupby("Date")[pred].rank(ascending=ascending, method="first")


def add_vector_sims(df: pd.DataFrame, pred: str, threshold: float = 0.55, k: int = 5) -> pd.DataFrame:
    """Add sim1..sim4 columns for one prediction column (investment of $100 per trade)."""
    g = df["growth_future_20d"]
    top = rank_in_day(df, pred) <= k
    bottom = rank_in_day(df, pred, ascending=True) <= k
    signals = {
        "sim1": df[pred] > threshold,
        "sim2": top,
        "sim3": top & (df["market_regime"] == "bull"),
    }
    for name, mask in signals.items():
        df[f"{name}_{pred}_gross_rev"] = np.where(mask, 100 * (g - 1), 0.0)
        df[f"{name}_{pred}_fees"] = np.where(mask, -100 * FEE * (1 + g), 0.0)
        df[f"{name}_{pred}_net_rev"] = df[f"{name}_{pred}_gross_rev"] + df[f"{name}_{pred}_fees"]
        df[f"{name}_{pred}_count"] = mask.astype(int)
    # long-short: +100 long the top-k, -100 short the bottom-k (a short gains when the price falls)
    ls = np.where(top, 100 * (g - 1), 0.0) + np.where(bottom, -100 * (g - 1), 0.0)
    fees = np.where(top | bottom, -100 * FEE * (1 + g), 0.0)
    df[f"sim4_{pred}_gross_rev"], df[f"sim4_{pred}_fees"] = ls, fees
    df[f"sim4_{pred}_net_rev"] = ls + fees
    df[f"sim4_{pred}_count"] = (top | bottom).astype(int)
    return df


def summarize_vector_sims(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    """One row per (strategy, prediction): trades, net revenue, capital and approximate CAGR.

    Capital = $100 x 20 holding days x Q75 of daily trades (as in the course): the money needed
    to keep the strategy running most days. CAGR = ((capital + net) / capital) ^ (1 / years)."""
    years = (df["Date"].max() - df["Date"].min()).days / 365.25
    rows = []
    for pred in preds:
        for sim in ("sim1", "sim2", "sim3", "sim4"):
            c = f"{sim}_{pred}"
            daily = df.groupby("Date")[f"{c}_count"].sum()
            trades = int(daily.sum())
            if trades == 0:
                continue
            net = df[f"{c}_net_rev"].sum()
            capital = 100 * H * max(daily.quantile(0.75), 1)
            wins = (df.loc[df[f"{c}_count"] == 1, f"{c}_net_rev"] > 0).mean()
            rows.append({"strategy": sim, "prediction": pred, "trades": trades,
                         "trades_per_day": round(trades / len(daily), 2), "win_rate": wins,
                         "gross_rev": df[f"{c}_gross_rev"].sum(), "fees": df[f"{c}_fees"].sum(),
                         "net_rev": net, "capital": capital,
                         "cagr": ((capital + net) / capital) ** (1 / years)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exact (iterative) simulation
# ---------------------------------------------------------------------------
def exact_simulation(df: pd.DataFrame, pred: str, k: int = 5, threshold: float = 0.5,
                     stop_loss: float | None = None, take_profit: float | None = None,
                     size_by_prob: bool = False, regime_filter: bool = False,
                     capital: float = 100_000.0) -> tuple[pd.Series, pd.DataFrame]:
    """Day-by-day portfolio simulation. Returns (daily equity, trades)."""
    prices = df.pivot_table(index="Date", columns="Ticker", values="Close_usd").sort_index().ffill()
    dates = prices.index
    by_day = {d: g for d, g in df.groupby("Date")}
    cash, positions, trades, equity = capital, [], [], []

    for i, d in enumerate(dates):
        px = prices.loc[d]
        # 1) exits: holding period reached, stop-loss or take-profit on the close
        still_open = []
        for p in positions:
            price = px[p["ticker"]]
            r = price / p["entry_price"]
            reason = None
            if i - p["entry_i"] >= H:
                reason = "time"
            elif stop_loss is not None and r <= 1 - stop_loss:
                reason = "stop_loss"
            elif take_profit is not None and r >= 1 + take_profit:
                reason = "take_profit"
            if reason:
                proceeds = p["units"] * price * (1 - FEE)
                cash += proceeds
                trades.append({**p, "exit_date": d, "exit_price": price, "reason": reason,
                               "pnl": proceeds - p["cost"], "return": proceeds / p["cost"] - 1})
            else:
                still_open.append(p)
        positions = still_open
        value = cash + sum(p["units"] * px[p["ticker"]] for p in positions)
        equity.append(value)

        # 2) entries: top-k signals of the day above the threshold
        day = by_day.get(d)
        if day is None or i >= len(dates) - 1:
            continue
        cand = day[day[pred] >= threshold]
        if regime_filter:
            cand = cand[cand["market_regime"] == "bull"]
        held = {p["ticker"] for p in positions}
        cand = cand[~cand["Ticker"].isin(held)].nlargest(k, pred)
        if cand.empty:
            continue
        budget = min(value / H, cash)          # ~1/20 of the equity per day -> fully invested after 20 days
        w = (cand[pred] - threshold + 0.01) if size_by_prob else pd.Series(1.0, index=cand.index)
        w = w / w.sum()
        for (_, row), wi in zip(cand.iterrows(), w):
            cost = budget * wi
            if cost < 1:
                continue
            price = row["Close_usd"]
            positions.append({"ticker": row["Ticker"], "market": row["Market"], "entry_date": d, "entry_i": i,
                              "entry_price": price, "units": cost * (1 - FEE) / price, "cost": cost,
                              "prob": row[pred]})
            cash -= cost

    return pd.Series(equity, index=dates, name=pred), pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def performance(equity: pd.Series, rf_annual: float = 0.0) -> dict:
    """CAGR, annual volatility, Sharpe ratio, max drawdown and Calmar ratio of an equity curve."""
    equity = equity.dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ret = equity.pct_change().dropna()
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years)
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS - rf_annual) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {"cagr": cagr, "annual_vol": vol, "sharpe": sharpe, "max_drawdown": dd.min(),
            "calmar": (cagr - 1) / abs(dd.min()) if dd.min() < 0 else np.nan,
            "total_return": equity.iloc[-1] / equity.iloc[0] - 1}


def benchmark_equity(market: pd.DataFrame, dates: pd.DatetimeIndex, capital: float = 100_000.0) -> pd.DataFrame:
    """Buy & hold equity curves of the benchmarks over the same dates."""
    cols = {"etf_latam40": "ILF (Latin America 40)", "etf_brazil": "EWZ (Brazil)", "sp500": "S&P 500"}
    m = market[list(cols)].ffill().reindex(dates, method="ffill")
    return (m / m.iloc[0] * capital).rename(columns=cols)


def risk_free_rate(market: pd.DataFrame, dates) -> float:
    """Average US 3-month T-bill yield (^IRX, in %) over the simulated dates, as an annual rate."""
    return float(market["us3m"].reindex(pd.DatetimeIndex(dates).unique()).mean() / 100)


def rolling_returns(equity: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    return equity / equity.shift(window) - 1
