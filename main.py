"""End-to-end workflow: data -> transformations -> models -> predictions -> simulation -> daily report.

Usage
  python main.py                # daily run: incremental download + transform + predict with the saved models
  python main.py --retrain      # same, and also tune and refit the models (weekly / monthly)
  python main.py --fast         # no download: use the dataset and models already in data/
                                # (downloaded from the Google Drive snapshot if data/ is empty)
  python main.py --full-reload  # download the full history again (ignores the saved raw data)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config as cfg
from scripts.data_repo import DataRepository
from scripts.report import latest_signals, model_health, send_telegram, write_report
from scripts.simulate import benchmark_equity, exact_simulation, performance, risk_free_rate
from scripts.snapshot import ensure_snapshot
from scripts.train import TrainModel
from scripts.transform import TransformData


def run(fast: bool = False, retrain: bool = False, full_reload: bool = False) -> None:
    t0 = time.time()
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) Data ------------------------------------------------------------------
    print("Step 1: data")
    if fast:
        ensure_snapshot()
        repo = DataRepository().load()
    else:
        repo = DataRepository().fetch(incremental=not full_reload)
        repo.persist()
    print(" ", repo.summary())

    # 2) Transformations ----------------------------------------------------------
    print("Step 2: transformations")
    td = TransformData(repo)
    if fast:
        td.load()
    else:
        td.transform()
        td.persist()

    # 3) Models -------------------------------------------------------------------
    print("Step 3: models")
    tm = TrainModel(td.df).prepare()
    del td
    if retrain or not (cfg.MODELS_DIR / "models.joblib").exists():
        tm.select_features().tune().fit_final()
    else:
        tm.load()
    pred = tm.predict()
    tm.persist()

    # 4) Simulation of the chosen strategy on the test period ---------------------
    print("Step 4: simulation (test period)")
    st = cfg.STRATEGY
    test = pred[pred["split"] == "test"]
    equity, trades = exact_simulation(test, st["prediction"], k=st["k"], threshold=st["threshold"],
                                      stop_loss=st["stop_loss"], take_profit=st["take_profit"],
                                      size_by_prob=st["size_by_prob"], regime_filter=st["regime_filter"])
    bench = benchmark_equity(repo.market, equity.index)
    rf = risk_free_rate(repo.market, equity.index)
    summary = pd.DataFrame([{"portfolio": "strategy", **performance(equity, rf)}]
                           + [{"portfolio": c, **performance(bench[c], rf)} for c in bench.columns])
    summary.to_csv(cfg.OUTPUT_DIR / "strategy_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # 5) Latest signals and report ---------------------------------------------------
    print("Step 5: latest signals")
    signals = latest_signals(pred, st["prediction"], k=st["k"], threshold=st["threshold"])
    health = model_health(pred, st["prediction"])
    write_report(signals, health, summary, st["prediction"])
    print(signals.to_string(index=False))
    if send_telegram(signals, st["prediction"]):
        print("  sent to Telegram")
    print(f"Done in {time.time() - t0:.0f}s -> output/daily_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatAm stock signals - daily workflow")
    parser.add_argument("--fast", action="store_true", help="use the saved dataset and models (no download)")
    parser.add_argument("--retrain", action="store_true", help="tune and refit the models")
    parser.add_argument("--full-reload", action="store_true", help="download the full history again")
    args = parser.parse_args()
    run(fast=args.fast, retrain=args.retrain, full_reload=args.full_reload)
