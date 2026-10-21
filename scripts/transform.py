"""Step 2 - Transformations: one table with features and targets for every (Ticker, Date).

Feature groups (column lists are kept in FEATURE_SETS):
- GROWTH       : past USD returns (course features, computed on USD prices)
- TECHNICAL    : moving averages, volatility, volume, 52-week high/low distance
- TALIB        : scale-free TA-Lib momentum / cycle indicators
- PATTERNS     : the 61 TA-Lib candlestick patterns (-100/0/100)
- RELATIVE     : relative strength vs the country ETF and the S&P 500, rolling beta
- CROSS        : cross-sectional ranks and breadth inside each market and day
- MARKET       : ETFs, FX, commodities, US rates, VIX (same for all stocks of a day)
- MACRO        : Brazil Selic / inflation, Argentina reserves (lagged to avoid look-ahead)
- CATEGORICAL  : Market, Sector, month, weekday and binned variables -> dummies in train.py
Targets (TO_PREDICT): growth_future_20d, excess_future_20d, is_outperform_20d, is_positive_growth_20d
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import talib

import config as cfg
from scripts.data_repo import DataRepository

warnings.filterwarnings("ignore")

H = cfg.HORIZON
GROWTH_WINDOWS = [1, 3, 5, 10, 20, 60, 120, 250]
REL_WINDOWS = [5, 20, 60, 120, 250]
PATTERN_FUNCS = talib.get_function_groups()["Pattern Recognition"]


def clean_market(market: pd.DataFrame, tolerance: float = 0.2) -> pd.DataFrame:
    """Remove isolated bad prints in market series (e.g. CLP=X has days at ~1/100 of the real rate):
    a value more than 20% away from its centred 5-day median is set to NaN and forward-filled.
    Rates and VIX are excluded (they can legitimately move more than 20% in a day)."""
    m = market.copy()
    for col in m.columns:
        if col in ("vix", "us10y", "us5y", "us3m"):
            continue
        med = m[col].rolling(5, center=True, min_periods=3).median()
        m.loc[(m[col] / med - 1).abs() > tolerance, col] = np.nan
    return m.ffill()


# ---------------------------------------------------------------------------
# Market-wide features (one row per date)
# ---------------------------------------------------------------------------
def market_features(market: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    m = market.ffill()
    out = pd.DataFrame(index=m.index)
    for col in ["etf_latam40", "sp500", "etf_brazil", "etf_mexico", "etf_chile", "etf_argentina",
                "brent", "copper", "gold", "soybeans", "iron_ore", "dollar_index",
                "usdbrl", "usdmxn", "usdclp"]:
        for w in (5, 20, 60):
            out[f"{col}_growth_{w}d"] = m[col] / m[col].shift(w)
    for col in ["usdbrl", "usdmxn", "usdclp"]:
        out[f"{col}_vol_20d"] = np.log(m[col]).diff().rolling(20).std() * np.sqrt(252)
    out["vix"] = m["vix"]
    out["vix_change_20d"] = m["vix"] - m["vix"].shift(20)
    out["us10y"] = m["us10y"]
    out["us3m"] = m["us3m"]
    out["us_curve_10y_3m"] = m["us10y"] - m["us3m"]
    out["us10y_change_20d"] = m["us10y"] - m["us10y"].shift(20)
    out["sp500_above_sma200"] = (m["sp500"] > m["sp500"].rolling(200).mean()).astype("int8")

    # Macro: publication lags to avoid look-ahead bias.
    mac = macro.copy()
    monthly = mac[["ipca_mom", "ipca_12m"]].dropna(how="all")
    monthly.index = monthly.index + pd.DateOffset(days=45)       # IPCA is published ~10 days after month end
    daily = mac[["selic_target", "bcra_reserves_musd"]].shift(1)  # known the next day
    mac = pd.concat([daily, monthly], axis=1, sort=True).ffill()
    mac = mac.reindex(out.index, method="ffill")
    out["selic_target"] = mac["selic_target"]
    out["ipca_12m"] = mac["ipca_12m"]
    out["selic_real"] = mac["selic_target"] - mac["ipca_12m"]
    out["selic_minus_us3m"] = mac["selic_target"] - m["us3m"]
    out["bcra_reserves_growth_20d"] = mac["bcra_reserves_musd"] / mac["bcra_reserves_musd"].shift(20)
    out["bcra_reserves_growth_250d"] = mac["bcra_reserves_musd"] / mac["bcra_reserves_musd"].shift(250)
    return out


# ---------------------------------------------------------------------------
# Per-ticker features
# ---------------------------------------------------------------------------
def talib_features(df: pd.DataFrame) -> dict:
    """Scale-free TA-Lib indicators (values comparable across stocks with different prices)."""
    o, h, l, c, v = (df[k].values.astype("float64") for k in ["Open", "High", "Low", "Close", "Volume"])
    f = {
        "adx": talib.ADX(h, l, c, 14), "adxr": talib.ADXR(h, l, c, 14),
        "aroonosc": talib.AROONOSC(h, l, 14), "bop": talib.BOP(o, h, l, c),
        "cci": talib.CCI(h, l, c, 14), "cmo": talib.CMO(c, 14), "dx": talib.DX(h, l, c, 14),
        "mfi": talib.MFI(h, l, c, v, 14), "minus_di": talib.MINUS_DI(h, l, c, 14),
        "plus_di": talib.PLUS_DI(h, l, c, 14), "ppo": talib.PPO(c, 12, 26),
        "roc_10": talib.ROC(c, 10), "rsi": talib.RSI(c, 14), "trix": talib.TRIX(c, 30),
        "ultosc": talib.ULTOSC(h, l, c, 7, 14, 28), "willr": talib.WILLR(h, l, c, 14),
        "natr": talib.NATR(h, l, c, 14),
        "ht_dcperiod": talib.HT_DCPERIOD(c), "ht_dcphase": talib.HT_DCPHASE(c),
        "ht_trendmode": talib.HT_TRENDMODE(c),
    }
    aroon_down, aroon_up = talib.AROON(h, l, 14)
    f["aroon_up"], f["aroon_down"] = aroon_up, aroon_down
    slowk, slowd = talib.STOCH(h, l, c)
    f["stoch_slowk"], f["stoch_slowd"] = slowk, slowd
    fastk, fastd = talib.STOCHRSI(c, 14)
    f["stochrsi_k"], f["stochrsi_d"] = fastk, fastd
    macd, signal, hist = talib.MACD(c, 12, 26, 9)
    f["macd_hist_rel"] = hist / c          # normalised by price
    sine, leadsine = talib.HT_SINE(c)
    f["ht_sine"], f["ht_leadsine"] = sine, leadsine
    return f


MIN_TURNOVER_USD = 1e6
MAX_LOG_JUMP = 1.0   # |log return| > 1 in one day (x2.7 or /2.7) = broken split adjustment in Yahoo data


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bad prints from Yahoo data (mostly early years of illiquid Brazilian stocks):
    1. days without trading (Volume 0: stale prices),
    2. the history before the last impossible one-day jump (unadjusted splits / unit changes).
    Real crashes (e.g. LATAM -58% in 2020, |log r| = 0.58) are kept."""
    df = df[(df["Volume"] > 0) & (df["Close"] > 0)].sort_values("Date")
    r = np.log(df["Close"]).diff().abs()
    breaks = df.loc[r > MAX_LOG_JUMP, "Date"]
    if len(breaks):
        df = df[df["Date"] >= breaks.max()]
    return df


def ticker_features(df: pd.DataFrame, fx: pd.Series | None, bench: pd.Series, sp500: pd.Series) -> pd.DataFrame:
    """Features and targets for one ticker. Prices are converted to USD with the daily FX rate."""
    df = df.sort_values("Date").set_index("Date")
    rate = fx.reindex(df.index, method="ffill") if fx is not None else 1.0
    usd = df["Close"] / rate
    b = bench.reindex(df.index, method="ffill")
    sp = sp500.reindex(df.index, method="ffill")
    ret = np.log(usd).diff()
    b_ret = np.log(b).diff()

    f = {"Close": df["Close"], "Close_usd": usd, "Volume": df["Volume"]}
    # --- growth (course features, in USD)
    for w in GROWTH_WINDOWS:
        f[f"growth_{w}d"] = usd / usd.shift(w)
    # --- technical
    for w in (10, 20, 50, 200):
        f[f"close_to_sma{w}"] = usd / usd.rolling(w).mean()
    f["sma10_above_sma20"] = (usd.rolling(10).mean() > usd.rolling(20).mean()).astype("int8")
    f["high_minus_low_relative"] = (df["High"] - df["Low"]) / df["Close"]
    f["volatility_20d"] = ret.rolling(20).std() * np.sqrt(252)
    f["volatility_60d"] = ret.rolling(60).std() * np.sqrt(252)
    f["ln_volume"] = np.log1p(df["Volume"])
    dollar_vol = usd * df["Volume"]
    f["ln_turnover_usd_20d"] = np.log1p(dollar_vol.rolling(20).mean())
    f["volume_ratio_5_60"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(60).mean() + 1)
    f["amihud_20d"] = (ret.abs() / (dollar_vol / 1e6 + 1e-9)).rolling(20).mean()
    f["dist_52w_high"] = usd / usd.rolling(250, min_periods=120).max() - 1
    f["dist_52w_low"] = usd / usd.rolling(250, min_periods=120).min() - 1
    # --- relative strength vs the country ETF and the S&P 500
    for w in REL_WINDOWS:
        f[f"rel_bench_{w}d"] = (usd / usd.shift(w)) / (b / b.shift(w)) - 1
    for w in (20, 60):
        f[f"rel_sp500_{w}d"] = (usd / usd.shift(w)) / (sp / sp.shift(w)) - 1
    cov = ret.rolling(120).cov(b_ret)
    var = b_ret.rolling(120).var()
    f["beta_120d"] = cov / var
    f["idio_vol_120d"] = (ret - f["beta_120d"] * b_ret).rolling(120).std() * np.sqrt(252)
    # --- TA-Lib
    f.update(talib_features(df.reset_index()))
    for name in PATTERN_FUNCS:
        func = getattr(talib, name)
        f[name.lower()] = func(*(df[k].values.astype("float64") for k in ["Open", "High", "Low", "Close"]))
    # --- targets (future information: only used as labels, never as features)
    f["growth_future_20d"] = usd.shift(-H) / usd
    f["bench_growth_future_20d"] = b.shift(-H) / b
    # max / min of the next H days, used by the stop-loss / take-profit simulation
    f["max_future_20d"] = df["High"][::-1].rolling(H, min_periods=1).max()[::-1].shift(-1) / df["Close"]
    f["min_future_20d"] = df["Low"][::-1].rolling(H, min_periods=1).min()[::-1].shift(-1) / df["Close"]

    out = pd.DataFrame({k: (pd.Series(v, index=df.index) if isinstance(v, np.ndarray) else v)
                        for k, v in f.items()})
    return out


# ---------------------------------------------------------------------------
class TransformData:
    """Builds data/processed/dataset.parquet from a DataRepository."""

    def __init__(self, repo: DataRepository | None = None):
        self.repo = repo
        self.df: pd.DataFrame | None = None

    def transform(self, min_date: str = "2005-01-01") -> "TransformData":
        repo = self.repo
        mkt = clean_market(repo.market)
        uni = repo.universe.set_index("Ticker")
        frames, t0 = [], time.time()
        removed = 0
        for ticker, g in repo.stocks.groupby("Ticker"):
            n0 = len(g)
            g = clean_prices(g)
            removed += n0 - len(g)
            if len(g) < 300 or ticker not in uni.index:
                continue
            market = uni.loc[ticker, "Market"]
            mcfg = cfg.MARKETS[market]
            fx = mkt[cfg.MARKET_SYMBOLS[mcfg["fx"]]] if mcfg["fx"] else None
            bench = mkt[cfg.MARKET_SYMBOLS[mcfg["benchmark"]]]
            f = ticker_features(g, fx, bench, mkt["sp500"])
            f["Ticker"], f["Market"], f["Sector"] = ticker, market, uni.loc[ticker, "Sector"]
            num = f.select_dtypes("number").columns
            f[num] = f[num].astype("float32")
            frames.append(f.reset_index())
        df = pd.concat(frames, ignore_index=True)
        print(f"  cleaning removed {removed:,} bad rows; per-ticker features: {df.shape} ({time.time() - t0:.0f}s)")

        # pattern columns as int8 (values -200..200 / 100)
        for name in PATTERN_FUNCS:
            df[name.lower()] = (df[name.lower()] / 100).fillna(0).astype("int8")

        # --- targets
        df["excess_future_20d"] = df["growth_future_20d"] - df["bench_growth_future_20d"]
        df["is_outperform_20d"] = (df["excess_future_20d"] > 0).astype("int8")
        df["is_positive_growth_20d"] = (df["growth_future_20d"] > 1).astype("int8")
        no_future = df["growth_future_20d"].isna() | df["bench_growth_future_20d"].isna()
        df.loc[no_future, ["is_outperform_20d", "is_positive_growth_20d"]] = -1   # unknown yet (last 20 days)

        # --- cross-sectional features inside each (Market, Date)
        grp = df.groupby(["Market", "Date"])
        for col in ["growth_20d", "growth_60d", "growth_250d", "volatility_20d", "rsi",
                    "ln_turnover_usd_20d", "dist_52w_high"]:
            df[f"rank_{col}"] = grp[col].rank(pct=True).astype("float32")
        df["market_breadth_sma200"] = grp["close_to_sma200"].transform(lambda s: (s > 1).mean()).astype("float32")
        df["market_median_growth_20d"] = grp["growth_20d"].transform("median").astype("float32")
        df["market_n_stocks"] = grp["Ticker"].transform("count").astype("int16")

        # --- market and macro features
        mf = market_features(mkt, repo.macro).astype("float32")
        df = df.merge(mf, left_on="Date", right_index=True, how="left")
        bench_col = df["Market"].map({k: cfg.MARKET_SYMBOLS[v["benchmark"]] for k, v in cfg.MARKETS.items()})
        bench_growth_20 = np.select([bench_col == c for c in bench_col.unique()],
                                    [df[f"{c}_growth_20d"] for c in bench_col.unique()])
        df["bench_growth_20d"] = bench_growth_20.astype("float32")

        # --- categorical (binned) features
        df["month"] = df["Date"].dt.month.astype(str)
        df["weekday"] = df["Date"].dt.dayofweek.astype(str)
        df["rsi_zone"] = pd.cut(df["rsi"], [-1, 30, 70, 101], labels=["oversold", "neutral", "overbought"]).astype(str)
        df["vol_quintile"] = (np.ceil(df["rank_volatility_20d"] * 5).clip(1, 5)).fillna(0).astype(int).astype(str)
        df["momentum_quintile"] = (np.ceil(df["rank_growth_60d"] * 5).clip(1, 5)).fillna(0).astype(int).astype(str)
        df["near_52w_high"] = pd.cut(df["dist_52w_high"], [-1.01, -0.3, -0.1, -0.02, 0.01],
                                     labels=["far", "mid", "close", "at_high"]).astype(str)
        df["vix_regime"] = pd.cut(df["vix"], [0, 15, 25, 200], labels=["calm", "normal", "stress"]).astype(str)
        df["market_regime"] = np.where(df["market_breadth_sma200"] > 0.5, "bull", "bear")

        # tradable universe: average traded value of at least US$1M per day in the last 20 days
        df["is_liquid"] = (np.expm1(df["ln_turnover_usd_20d"]) >= MIN_TURNOVER_USD).astype("int8")

        df = df[df["Date"] >= min_date].reset_index(drop=True)
        self.df = df
        print(f"  dataset: {df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory "
              f"({time.time() - t0:.0f}s)")
        return self

    # --- persistence ----------------------------------------------------------
    def persist(self, path=cfg.PROCESSED_DIR / "dataset.parquet") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, compression="brotli", index=False)

    def load(self, path=cfg.PROCESSED_DIR / "dataset.parquet") -> "TransformData":
        self.df = pd.read_parquet(path)
        return self


# Feature sets ----------------------------------------------------------------
GROWTH = [f"growth_{w}d" for w in GROWTH_WINDOWS]
TECHNICAL = ["close_to_sma10", "close_to_sma20", "close_to_sma50", "close_to_sma200", "sma10_above_sma20",
             "high_minus_low_relative", "volatility_20d", "volatility_60d", "ln_volume", "ln_turnover_usd_20d",
             "volume_ratio_5_60", "amihud_20d", "dist_52w_high", "dist_52w_low"]
TALIB = ["adx", "adxr", "aroonosc", "bop", "cci", "cmo", "dx", "mfi", "minus_di", "plus_di", "ppo", "roc_10",
         "rsi", "trix", "ultosc", "willr", "natr", "ht_dcperiod", "ht_dcphase", "ht_trendmode", "aroon_up",
         "aroon_down", "stoch_slowk", "stoch_slowd", "stochrsi_k", "stochrsi_d", "macd_hist_rel", "ht_sine",
         "ht_leadsine"]
PATTERNS = [p.lower() for p in PATTERN_FUNCS]
RELATIVE = [f"rel_bench_{w}d" for w in REL_WINDOWS] + ["rel_sp500_20d", "rel_sp500_60d", "beta_120d",
                                                         "idio_vol_120d", "bench_growth_20d"]
CROSS = ["rank_growth_20d", "rank_growth_60d", "rank_growth_250d", "rank_volatility_20d", "rank_rsi",
         "rank_ln_turnover_usd_20d", "rank_dist_52w_high", "market_breadth_sma200", "market_median_growth_20d",
         "market_n_stocks"]
MARKET = ([f"{c}_growth_{w}d" for c in ["etf_latam40", "sp500", "etf_brazil", "etf_mexico", "etf_chile",
                                         "etf_argentina", "brent", "copper", "gold", "soybeans", "iron_ore",
                                         "dollar_index", "usdbrl", "usdmxn", "usdclp"] for w in (5, 20, 60)]
          + ["usdbrl_vol_20d", "usdmxn_vol_20d", "usdclp_vol_20d", "vix", "vix_change_20d", "us10y", "us3m",
             "us_curve_10y_3m", "us10y_change_20d", "sp500_above_sma200"])
MACRO = ["selic_target", "ipca_12m", "selic_real", "selic_minus_us3m", "bcra_reserves_growth_20d",
         "bcra_reserves_growth_250d"]
CATEGORICAL = ["Market", "Sector", "month", "weekday", "rsi_zone", "vol_quintile", "momentum_quintile",
               "near_52w_high", "vix_regime", "market_regime"]
TO_PREDICT = ["growth_future_20d", "excess_future_20d", "is_outperform_20d", "is_positive_growth_20d"]
FEATURE_SETS = {"GROWTH": GROWTH, "TECHNICAL": TECHNICAL, "TALIB": TALIB, "PATTERNS": PATTERNS,
                "RELATIVE": RELATIVE, "CROSS": CROSS, "MARKET": MARKET, "MACRO": MACRO}
NUMERIC = [c for cols in FEATURE_SETS.values() for c in cols]


if __name__ == "__main__":
    repo = DataRepository().load()
    td = TransformData(repo).transform()
    td.persist()
    print({k: len(v) for k, v in FEATURE_SETS.items()}, "numeric:", len(NUMERIC), "categorical:", len(CATEGORICAL))
