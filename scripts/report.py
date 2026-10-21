"""Step 5 - Daily report: the latest trade signals, model health and strategy results.

Writes output/latest_signals.csv and output/daily_report.md. If the environment variables
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set, the summary is also sent to Telegram
(create a bot with @BotFather, then get the chat id from https://api.telegram.org/bot<TOKEN>/getUpdates).
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests

import config as cfg


def latest_signals(pred: pd.DataFrame, pred_col: str, k: int = 5, threshold: float = 0.5) -> pd.DataFrame:
    """Top-k stocks of the last available day (the trades to open at the next session)."""
    last = pred["Date"].max()
    day = pred[pred["Date"] == last]
    picks = day[day[pred_col] >= threshold].nlargest(k, pred_col)
    cols = ["Date", "Ticker", "Market", "Sector", "Close", "Close_usd", pred_col, "market_regime"]
    return picks[cols].reset_index(drop=True)


def model_health(pred: pd.DataFrame, pred_col: str, days: int = 120) -> dict:
    """Precision of the top-5 picks over the last `days` days with a known outcome (monitoring)."""
    known = pred[pred["is_outperform_20d"] >= 0]
    recent = known[known["Date"] >= known["Date"].max() - pd.Timedelta(days=days)]
    top = recent[recent.groupby("Date")[pred_col].rank(ascending=False, method="first") <= 5]
    return {"window_days": days, "top5_precision": float(top["is_outperform_20d"].mean()),
            "base_rate": float(recent["is_outperform_20d"].mean()),
            "avg_excess_return_top5": float(top["excess_future_20d"].mean())}


def write_report(signals: pd.DataFrame, health: dict, summary: pd.DataFrame | None, pred_col: str) -> str:
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    signals.to_csv(cfg.OUTPUT_DIR / "latest_signals.csv", index=False)
    date = signals["Date"].max().date() if len(signals) else "n/a"
    lines = [f"# LatAm stock signals - {date}", "",
             f"Model: `{pred_col}` | horizon: {cfg.HORIZON} trading days | generated {datetime.now():%Y-%m-%d %H:%M}",
             "", "## Buy candidates (beat the country ETF in the next 20 days)", ""]
    if len(signals):
        show = signals.assign(prob=signals[pred_col].round(3))[["Ticker", "Market", "Sector", "Close", "prob",
                                                                 "market_regime"]]
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("No signal above the threshold today.")
    lines += ["", "## Model health (last 120 days with known outcome)", "",
              f"- Top-5 precision: {health['top5_precision']:.1%} (base rate {health['base_rate']:.1%})",
              f"- Average excess return of the top-5 vs the country ETF: {health['avg_excess_return_top5']:+.2%}"]
    if summary is not None and len(summary):
        lines += ["", "## Strategy (test period, exact simulation)", "", summary.round(3).to_markdown(index=False)]
    text = "\n".join(lines) + "\n"
    (cfg.OUTPUT_DIR / "daily_report.md").write_text(text, encoding="utf-8")
    return text


def send_telegram(signals: pd.DataFrame, pred_col: str) -> bool:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    if len(signals):
        body = "\n".join(f"{r.Ticker} ({r.Market}) p={getattr(r, pred_col):.2f}" for r in signals.itertuples())
    else:
        body = "no signals"
    msg = f"LatAm signals {signals['Date'].max().date() if len(signals) else ''}\n{body}"
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat, "text": msg},
                      timeout=30)
    return r.ok
