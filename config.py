"""Project settings: stock universe, market symbols, paths and model constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # not committed to GitHub (see .gitignore)
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = ROOT / "output"        # predictions, simulations and daily report

START_DATE = "2004-01-01"           # BRL=X starts in Dec-2003
HORIZON = 20                        # prediction horizon in trading days (~1 month)
FEE = 0.001                         # 0.1% per side (buy and sell)

# ---------------------------------------------------------------------------
# Markets: suffix used by Yahoo Finance, FX pair to convert to USD and the
# country ETF (in USD) used as the "local index" benchmark.
# ---------------------------------------------------------------------------
MARKETS = {
    "BR": {"suffix": ".SA", "fx": "BRL=X", "benchmark": "EWZ", "index": "^BVSP"},
    "MX": {"suffix": ".MX", "fx": "MXN=X", "benchmark": "EWW", "index": "^MXX"},
    "CL": {"suffix": ".SN", "fx": "CLP=X", "benchmark": "ECH", "index": None},   # ^IPSA no longer on Yahoo
    "AR": {"suffix": "",    "fx": None,    "benchmark": "ARGT", "index": "^MERV"},  # NYSE/Nasdaq ADRs, already in USD
    "AN": {"suffix": "",    "fx": None,    "benchmark": "ILF",  "index": None},     # Andean ADRs (Colombia, Peru)
}

# Brazil: constituents of the IBrA index are downloaded from the B3 website (data_repo.get_b3_universe).
# Mexico (S&P/BMV IPC), Chile (S&P IPSA) and Argentine ADRs: ticker -> sector
MX_STOCKS = {
    "AC": "Consumer Staples", "ALSEA": "Consumer Discretionary", "AMXB": "Communication",
    "ASURB": "Industrials", "BBAJIOO": "Financials", "BIMBOA": "Consumer Staples",
    "BOLSAA": "Financials", "CEMEXCPO": "Materials", "CHDRAUIB": "Consumer Staples",
    "CUERVO": "Consumer Staples", "FEMSAUBD": "Consumer Staples", "GAPB": "Industrials",
    "GCARSOA1": "Industrials", "GCC": "Materials", "GENTERA": "Financials",
    "GFINBURO": "Financials", "GFNORTEO": "Financials", "GMEXICOB": "Materials",
    "GRUMAB": "Consumer Staples", "KIMBERA": "Consumer Staples", "KOFUBL": "Consumer Staples",
    "LABB": "Health Care", "LIVEPOLC-1": "Consumer Discretionary", "MEGACPO": "Communication",
    "OMAB": "Industrials", "ORBIA": "Materials", "PE&OLES": "Materials", "PINFRA": "Industrials",
    "Q": "Financials", "RA": "Financials", "TLEVISACPO": "Communication", "VESTA": "Real Estate",
    "WALMEX": "Consumer Staples", "ALPEKA": "Materials", "FIBRAMQ12": "Real Estate", "FUNO11": "Real Estate",
}
CL_STOCKS = {
    "AGUAS-A": "Utilities", "ANDINA-B": "Consumer Staples", "BCI": "Financials",
    "BSANTANDER": "Financials", "CAP": "Materials", "CCU": "Consumer Staples",
    "CENCOSUD": "Consumer Staples", "CENCOMALLS": "Real Estate", "CHILE": "Financials",
    "CMPC": "Materials", "COLBUN": "Utilities", "CONCHATORO": "Consumer Staples",
    "COPEC": "Energy", "ECL": "Utilities", "ENELAM": "Utilities", "ENELCHILE": "Utilities",
    "ENTEL": "Communication", "FALABELLA": "Consumer Discretionary", "IAM": "Utilities",
    "ITAUCL": "Financials", "LTM": "Industrials", "MALLPLAZA": "Real Estate",
    "PARAUCO": "Real Estate", "QUINENCO": "Industrials", "RIPLEY": "Consumer Discretionary",
    "SMU": "Consumer Staples", "SONDA": "Technology", "SQM-B": "Materials",
    "VAPORES": "Industrials", "SALFACORP": "Industrials",
}
AR_STOCKS = {
    "GGAL": "Financials", "BMA": "Financials", "SUPV": "Financials", "BBAR": "Financials",
    "YPF": "Energy", "VIST": "Energy", "PAM": "Utilities", "CEPU": "Utilities", "EDN": "Utilities",
    "TGS": "Energy", "CRESY": "Real Estate", "IRS": "Real Estate", "LOMA": "Materials",
    "TEO": "Communication", "CAAP": "Industrials", "TX": "Materials",
    "MELI": "Consumer Discretionary", "GLOB": "Technology",
}

AN_STOCKS = {
    "EC": "Energy", "CIB": "Financials", "AVAL": "Financials", "BAP": "Financials",
    "IFS": "Financials", "SCCO": "Materials", "BVN": "Materials",
}

# B3 economic sectors (Portuguese, as returned by the B3 API) -> common English sector
B3_SECTOR_MAP = {
    "consumo c": "Consumer Discretionary", "cons n": "Consumer Staples",
    "financ": "Financials", "bens ind": "Industrials", "utilidade": "Utilities",
    "mats b": "Materials", "sa": "Health Care", "petr": "Energy",
    "tec": "Technology", "comput": "Technology", "telecom": "Communication",
    "diversos": "Other",
}

# ---------------------------------------------------------------------------
# Market-wide series from Yahoo Finance (all daily Close prices)
# ---------------------------------------------------------------------------
MARKET_SYMBOLS = {
    # country / region ETFs (USD) and the US benchmark
    "EWZ": "etf_brazil", "EWW": "etf_mexico", "ECH": "etf_chile", "ARGT": "etf_argentina",
    "ILF": "etf_latam40", "^GSPC": "sp500",
    # local indexes (local currency)
    "^BVSP": "ibovespa", "^MXX": "ipc_mexico", "^MERV": "merval",
    # FX: local currency per 1 USD
    "BRL=X": "usdbrl", "MXN=X": "usdmxn", "CLP=X": "usdclp", "DX-Y.NYB": "dollar_index",
    # commodities
    "BZ=F": "brent", "HG=F": "copper", "GC=F": "gold", "ZS=F": "soybeans", "TIO=F": "iron_ore",
    # US rates and volatility (replace FRED, which blocks automated downloads)
    "^TNX": "us10y", "^FVX": "us5y", "^IRX": "us3m", "^VIX": "vix",
}

# Banco Central do Brasil - SGS time series (https://dadosabertos.bcb.gov.br)
BCB_SERIES = {432: "selic_target", 433: "ipca_mom", 13522: "ipca_12m"}
# Banco Central de la República Argentina - API v4.0 (https://api.bcra.gob.ar)
BCRA_SERIES = {1: "bcra_reserves_musd"}

# Random seed for reproducibility
SEED = 42

# Google Drive link to data snapshot (data/ zipped by scripts/snapshot.py) for `main.py --fast`
SNAPSHOT_URL = ""

# Strategy used by main.py (chosen in notebooks/03_simulation.ipynb on the test period)
STRATEGY = {"prediction": "pred_xgb", "k": 5, "threshold": 0.5, "stop_loss": 0.10, "take_profit": None,
            "size_by_prob": True, "regime_filter": False}
