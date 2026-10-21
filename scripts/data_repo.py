"""Step 1 - Data repository.

Downloads every raw dataset, stores it as parquet under data/raw/ and supports
incremental updates (only the last days are downloaded on each run).

Sources
- B3 (Brasil, Bolsa, Balcão) website API: constituents and sectors of the IBrA index.
- Yahoo Finance (yfinance): daily OHLCV of ~240 Latin American stocks/ADRs, country ETFs,
  local indexes, FX rates, commodities, US rates and VIX.
- Banco Central do Brasil (SGS API): Selic target rate and inflation (IPCA).
- Banco Central de la República Argentina (API v4.0): international reserves.
"""
from __future__ import annotations

import base64
import json
import time
import warnings
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)

B3_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/"
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
BCRA_URL = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/{code}"

OVERLAP_DAYS = 10        # days re-downloaded on an incremental update to detect price adjustments
ADJ_TOLERANCE = 0.002    # >0.2% difference in an overlapping Close => dividend/split => full reload


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def b3_sector(segment: str) -> str:
    """Translate the B3 economic sector (Portuguese) to a common English sector."""
    s = segment.split("/")[0].strip().lower()
    for key, sector in cfg.B3_SECTOR_MAP.items():
        if s.startswith(key):
            return sector
    return "Other"


def get_b3_universe(index: str = "IBRA") -> pd.DataFrame:
    """Constituents of a B3 index (IBrA = ~150 most liquid Brazilian stocks) with their sector.

    The B3 page calls an internal endpoint whose only parameter is a base64-encoded JSON.
    """
    params = {"language": "en-us", "pageNumber": 1, "pageSize": 300, "index": index, "segment": "2"}
    token = base64.b64encode(json.dumps(params).encode()).decode()
    rows = requests.get(B3_URL + token, timeout=60).json()["results"]
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "Ticker": df["cod"].str.strip() + ".SA",
        "Market": "BR",
        "Sector": df["segment"].map(b3_sector),
        "Name": df["asset"].str.strip().str.title(),
        "IndexWeight": pd.to_numeric(df["part"], errors="coerce"),
    })


def build_universe() -> pd.DataFrame:
    """All stocks: Brazil (from B3) + Mexico, Chile, Argentine and Andean ADRs (from config)."""
    try:
        br = get_b3_universe()
    except Exception as exc:  # B3 site down -> reuse the last saved universe
        print(f"  B3 API failed ({exc}); using the saved universe")
        saved = pd.read_csv(cfg.RAW_DIR / "universe.csv")
        br = saved[saved.Market == "BR"]
    others = []
    for market, stocks in (("MX", cfg.MX_STOCKS), ("CL", cfg.CL_STOCKS), ("AR", cfg.AR_STOCKS),
                            ("AN", cfg.AN_STOCKS)):
        suffix = cfg.MARKETS[market]["suffix"]
        for t, sector in stocks.items():
            others.append({"Ticker": t + suffix, "Market": market, "Sector": sector, "Name": t})
    return pd.concat([br, pd.DataFrame(others)], ignore_index=True).drop_duplicates("Ticker")


# ---------------------------------------------------------------------------
# Yahoo Finance helpers
# ---------------------------------------------------------------------------
def yf_download(tickers: list[str], start: str, batch: int = 60, retries: int = 3) -> pd.DataFrame:
    """Download daily adjusted OHLCV for many tickers; returns a long DataFrame
    (Date, Ticker, Open, High, Low, Close, Volume)."""
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(retries):
            try:
                wide = yf.download(chunk, start=start, group_by="ticker", auto_adjust=True,
                                   progress=False, threads=True)
                break
            except Exception as exc:
                print(f"  yfinance error ({exc}), retry {attempt + 1}")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        if wide.empty:
            continue
        long = wide.stack(level=0, future_stack=True).reset_index()
        long.columns.name = None
        long = long.rename(columns={"level_1": "Ticker"})
        frames.append(long.dropna(subset=["Close"]))
    if not frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def incremental_prices(old: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """Update a long price table, downloading only the last OVERLAP_DAYS for known tickers.

    Yahoo's adjusted prices change backwards after a dividend or split, so the overlapping
    days are compared with the stored ones: if they differ, the full history of that ticker
    is downloaded again.
    """
    if old is None or old.empty:
        return yf_download(tickers, cfg.START_DATE)

    known = set(old.Ticker)
    new_tickers = [t for t in tickers if t not in known]
    start = (old.Date.max() - timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
    recent = yf_download([t for t in tickers if t in known], start)

    # the last stored day may have been an intraday (incomplete) price: leave it out of the check
    stored = old.loc[old.Date < old.Date.max(), ["Date", "Ticker", "Close"]]
    merged = recent.merge(stored, on=["Date", "Ticker"], suffixes=("", "_old"), how="inner")
    diff = (merged.Close / merged.Close_old - 1).abs()
    adjusted = sorted(set(merged.loc[diff > ADJ_TOLERANCE, "Ticker"]))
    reload = new_tickers + adjusted
    print(f"  incremental: {len(recent)} recent rows, {len(new_tickers)} new tickers, "
          f"{len(adjusted)} re-adjusted tickers")

    keep = old[~old.Ticker.isin(adjusted) & old.Ticker.isin(tickers)]
    parts = [keep, recent[~recent.Ticker.isin(adjusted)]]
    if reload:
        parts.append(yf_download(reload, cfg.START_DATE))
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["Date", "Ticker"], keep="last").sort_values(["Ticker", "Date"])


# ---------------------------------------------------------------------------
# Central banks
# ---------------------------------------------------------------------------
def get_json(url: str, params: dict, ok, retries: int = 5):
    """GET a JSON API with retries (the central bank APIs fail or time out now and then)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.ok:
                js = r.json()
                if ok(js):
                    return js
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def fetch_bcb(code: int, name: str, start: str = cfg.START_DATE) -> pd.DataFrame:
    """Banco Central do Brasil SGS series. Daily series accept at most 10 years per request."""
    frames, y0 = [], pd.Timestamp(start).year
    while y0 <= date.today().year:
        y1 = min(y0 + 4, date.today().year)
        params = {"formato": "json", "dataInicial": f"01/01/{y0}", "dataFinal": f"31/12/{y1}"}
        rows = get_json(BCB_URL.format(code=code), params, ok=lambda js: isinstance(js, list))
        if rows is None:
            print(f"  WARNING: BCB series {code} failed for {y0}-{y1}")
        else:
            frames.append(pd.DataFrame(rows))
        y0 = y1 + 1
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df[name] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[df.Date <= pd.Timestamp.today()]   # the Selic target is published until the next meeting
    return df[["Date", name]].drop_duplicates("Date").set_index("Date")


def fetch_bcra(code: int, name: str, start: str = cfg.START_DATE) -> pd.DataFrame:
    """BCRA monetary series (paginated by year)."""
    frames = []
    for year in range(pd.Timestamp(start).year, date.today().year + 1):
        params = {"desde": f"{year}-01-01", "hasta": f"{year}-12-31", "limit": 3000}
        js = get_json(BCRA_URL.format(code=code), params, ok=lambda js: "results" in js)
        if js is None:
            print(f"  WARNING: BCRA series {code} failed for {year}")
            continue
        for res in js["results"]:
            frames.append(pd.DataFrame(res["detalle"]))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["fecha"])
    df[name] = pd.to_numeric(df["valor"], errors="coerce")
    return df[["Date", name]].drop_duplicates("Date").set_index("Date").sort_index()


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------
class DataRepository:
    """Holds the raw datasets: universe, stocks, market (wide), macro (wide)."""

    def __init__(self, data_dir=cfg.RAW_DIR):
        self.dir = data_dir
        self.universe: pd.DataFrame | None = None
        self.stocks: pd.DataFrame | None = None
        self.market: pd.DataFrame | None = None
        self.macro: pd.DataFrame | None = None

    # --- download -----------------------------------------------------------
    def fetch(self, incremental: bool = True) -> "DataRepository":
        """Download all sources. With incremental=True, previously saved data is updated."""
        if incremental and (self.dir / "stocks.parquet").exists():
            self.load()
        else:
            self.stocks = self.market = None

        print("  universe (B3 API + config)...")
        self.universe = build_universe()
        tickers = self.universe.Ticker.tolist()

        print(f"  stocks: {len(tickers)} tickers (yfinance)...")
        self.stocks = incremental_prices(self.stocks, tickers)
        missing = sorted(set(tickers) - set(self.stocks.Ticker))
        if missing:
            print(f"  no data for {len(missing)} tickers: {missing}")

        print("  market series (yfinance)...")
        old_market = None
        if self.market is not None:
            old_market = (self.market.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")
                          .dropna())
            inv = {v: k for k, v in cfg.MARKET_SYMBOLS.items()}
            old_market["Ticker"] = old_market.Ticker.map(inv)
            old_market = old_market.dropna(subset=["Ticker"])
        market_long = incremental_prices(old_market, list(cfg.MARKET_SYMBOLS))
        self.market = (market_long.pivot(index="Date", columns="Ticker", values="Close")
                       .rename(columns=cfg.MARKET_SYMBOLS).sort_index())

        print("  macro (Banco Central do Brasil + BCRA)...")
        macro = [fetch_bcb(c, n) for c, n in cfg.BCB_SERIES.items()]
        macro += [fetch_bcra(c, n) for c, n in cfg.BCRA_SERIES.items()]
        self.macro = pd.concat(macro, axis=1, sort=True)
        return self

    # --- persistence ----------------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.universe.to_csv(self.dir / "universe.csv", index=False)
        self.stocks.to_parquet(self.dir / "stocks.parquet", compression="brotli", index=False)
        self.market.to_parquet(self.dir / "market.parquet", compression="brotli")
        self.macro.to_parquet(self.dir / "macro.parquet", compression="brotli")

    def load(self) -> "DataRepository":
        self.universe = pd.read_csv(self.dir / "universe.csv")
        self.stocks = pd.read_parquet(self.dir / "stocks.parquet")
        self.market = pd.read_parquet(self.dir / "market.parquet")
        self.macro = pd.read_parquet(self.dir / "macro.parquet")
        return self

    def summary(self) -> str:
        s = self.stocks
        return (f"stocks: {len(s):,} rows, {s.Ticker.nunique()} tickers, {s.Date.min().date()} -> "
                f"{s.Date.max().date()} | market: {self.market.shape} | macro: {self.macro.shape}")


if __name__ == "__main__":
    t0 = time.time()
    repo = DataRepository().fetch(incremental=True)
    repo.persist()
    print(repo.summary(), f"({time.time() - t0:.0f}s)")
