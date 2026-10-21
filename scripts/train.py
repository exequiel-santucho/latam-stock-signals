"""Step 3 - Modeling: DecisionTree, RandomForest and XGBoost with hyperparameter tuning.

- Universe: liquid rows only (>= US$1M traded per day), see transform.py.
- Target: is_outperform_20d (the stock beats its country ETF in USD over the next 20 trading days).
- Temporal split with a 20-day embargo between sets (targets look 20 days ahead, so the last
  20 days before each cut would leak information into the next set):
    train      2005-01-01 .. 2018-12-31
    validation 2019-01-01 .. 2021-12-31   (hyperparameter tuning)
    test       2022-01-01 .. last date with a known target (never used for any decision)
- Tuning, step 1: choose the feature set with a fixed XGBoost (validation ROC AUC). Market-wide and
  macro features are identical for every stock of a day, so they let trees memorise periods.
- Tuning, step 2: grid per model family, chosen by validation ROC AUC. The best configuration of each
  family is refitted on train + validation and scored on test; the same models predict the
  latest days (no target yet), which are the live signals.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as cfg
from scripts.transform import CATEGORICAL, FEATURE_SETS, NUMERIC

TARGET = "is_outperform_20d"
SPLITS = {"train": ("2005-01-01", "2018-12-31"),
          "validation": ("2019-01-01", "2021-12-31"),
          "test": ("2022-01-01", "2100-01-01")}
EMBARGO_DAYS = 30   # calendar days (~20 trading days) removed at the end of train and validation

# columns kept next to the predictions (used by the simulations and the report)
KEEP = ["Date", "Ticker", "Market", "Sector", "Close", "Close_usd", "growth_future_20d",
        "bench_growth_future_20d", "excess_future_20d", "is_outperform_20d", "is_positive_growth_20d",
        "max_future_20d", "min_future_20d", "market_regime", "vix", "rsi", "ln_turnover_usd_20d"]

# time-level dummies (same value for all the stocks of a day)
DATE_LEVEL_DUMMIES = ("month_", "weekday_", "vix_regime_", "market_regime_")

GRIDS = {
    "dt": [{"max_depth": d, "min_samples_leaf": leaf} for d in (4, 6, 8) for leaf in (500, 3000)],
    "rf": [{"max_depth": d, "max_features": mf, "n_estimators": 150, "min_samples_leaf": 200,
            "max_samples": 0.3} for d in (8, 14) for mf in ("sqrt", 0.3)],
    "xgb": [{"max_depth": d, "learning_rate": 0.03, "n_estimators": 800, "subsample": 0.7,
             "colsample_bytree": cs, "min_child_weight": 50} for d in (3, 5, 7) for cs in (0.3, 0.6)],
}


def make_model(family: str, params: dict):
    if family == "dt":
        return DecisionTreeClassifier(random_state=cfg.SEED, **params)
    if family == "rf":
        return RandomForestClassifier(random_state=cfg.SEED, n_jobs=-1, **params)
    if family == "xgb":
        return XGBClassifier(random_state=cfg.SEED, n_jobs=-1, tree_method="hist", eval_metric="auc", **params)
    raise ValueError(family)


def precision_at_top(df: pd.DataFrame, pred: str, k: int = 10) -> float:
    """Share of winners among the k highest-probability stocks of each day."""
    top = df[df.groupby("Date")[pred].rank(ascending=False, method="first") <= k]
    return float(top[TARGET].mean())


class TrainModel:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df
        self.features: list[str] = []
        self.models: dict = {}
        self.tuning: list[dict] = []
        self.predictions: pd.DataFrame | None = None
        self.val_probs: dict = {}   # family -> (validation probabilities of the model fitted on train only, auc)

    # --- data -------------------------------------------------------------------
    def prepare(self) -> "TrainModel":
        """Liquid rows, dummies for categorical features, split labels."""
        df = self.df[self.df["is_liquid"] == 1].copy()
        dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype="int8")
        df = pd.concat([df.drop(columns=[c for c in CATEGORICAL if c not in KEEP]), dummies], axis=1)
        self.features = NUMERIC + dummies.columns.tolist()
        stock_dummies = [c for c in dummies.columns if not c.startswith(DATE_LEVEL_DUMMIES)]
        self.feature_sets = {
            "all": self.features,
            "no_market_macro": [c for c in self.features if c not in FEATURE_SETS["MARKET"] + FEATURE_SETS["MACRO"]],
            "stock_only": [c for s in ("GROWTH", "TECHNICAL", "TALIB", "RELATIVE", "CROSS") for c in FEATURE_SETS[s]]
                          + stock_dummies,
        }
        df[NUMERIC] = df[NUMERIC].replace([np.inf, -np.inf], np.nan)

        known = df[TARGET] >= 0
        # rows without a target: "live" if they are the last days (outcome not known yet),
        # "no_target" otherwise (e.g. Argentine ADRs before ARGT existed in 2011)
        recent = df["Date"] > df["Date"].max() - pd.Timedelta(days=45)
        df["split"] = np.where(recent, "live", "no_target")
        for name, (start, end) in SPLITS.items():
            df.loc[known & df["Date"].between(start, end), "split"] = name
        for name in ("train", "validation"):  # embargo
            end = pd.Timestamp(SPLITS[name][1])
            df.loc[(df["split"] == name) & (df["Date"] > end - pd.Timedelta(days=EMBARGO_DAYS)), "split"] = "embargo"
        self.data = df
        counts = df["split"].value_counts().to_dict()
        print(f"  features: {len(self.features)} | rows per split: {counts}")
        return self

    def xy(self, split: str | list[str]):
        splits = [split] if isinstance(split, str) else split
        d = self.data[self.data["split"].isin(splits)]
        return d[self.features], d[TARGET].astype(int), d

    # --- tuning -------------------------------------------------------------------
    def select_features(self) -> "TrainModel":
        """Step 1: compare feature sets with the same XGBoost on the validation period."""
        _, y_tr, d_tr = self.xy("train")
        _, y_va, d_va = self.xy("validation")
        best = (None, -1)
        for name, cols in self.feature_sets.items():
            model = make_model("xgb", {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
                                       "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 50})
            model.fit(d_tr[cols], y_tr)
            p = model.predict_proba(d_va[cols])[:, 1]
            auc = roc_auc_score(y_va, p)
            self.tuning.append({"family": "feature_set", "params": json.dumps({"feature_set": name, "n": len(cols)}),
                                "auc_val": auc, "precision_val_p50": precision_score(y_va, p > 0.5, zero_division=0),
                                "precision_val_top10": precision_at_top(d_va.assign(p=p), "p", 10), "seconds": 0})
            print(f"  feature set {name} ({len(cols)}) -> AUC {auc:.4f}")
            if auc > best[1]:
                best = (name, auc)
        self.feature_set = best[0]
        self.features = self.feature_sets[best[0]]
        print(f"  selected feature set: {best[0]}")
        return self

    def tune(self, families=("dt", "rf", "xgb")) -> "TrainModel":
        """Step 2: hyperparameter grid of each model family on the selected features."""
        X_tr, y_tr, _ = self.xy("train")
        X_va, y_va, d_va = self.xy("validation")
        for fam in families:
            for params in GRIDS[fam]:
                t0 = time.time()
                model = make_model(fam, params)
                if fam == "xgb":
                    model.set_params(early_stopping_rounds=100)
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    params = {**params, "n_estimators": int(model.best_iteration) + 1}
                else:
                    model.fit(X_tr, y_tr)
                p = model.predict_proba(X_va)[:, 1]
                d_va = d_va.assign(p=p)
                row = {"family": fam, "params": json.dumps(params), "auc_val": roc_auc_score(y_va, p),
                       "precision_val_p50": precision_score(y_va, p > 0.5, zero_division=0),
                       "precision_val_top10": precision_at_top(d_va, "p", 10),
                       "seconds": round(time.time() - t0)}
                self.tuning.append(row)
                # keep the out-of-sample validation probabilities of the best configuration of each family
                if row["auc_val"] > self.val_probs.get(fam, (None, -1))[1]:
                    self.val_probs[fam] = (pd.Series(p, index=d_va.index, dtype="float32"), row["auc_val"])
                print(f"  {fam} {params} -> AUC {row['auc_val']:.4f}  top10 {row['precision_val_top10']:.3f} "
                      f"({row['seconds']}s)")
        return self

    def best_params(self) -> dict:
        t = pd.DataFrame(self.tuning)
        t = t[t["family"] != "feature_set"]
        best = t.loc[t.groupby("family")["auc_val"].idxmax()]
        return {r.family: json.loads(r.params) for r in best.itertuples()}

    # --- final models -------------------------------------------------------------
    def fit_final(self, params: dict | None = None) -> "TrainModel":
        """Refit the best configuration of each family on train + validation."""
        params = params or self.best_params()
        X, y, _ = self.xy(["train", "validation"])
        for fam, p in params.items():
            t0 = time.time()
            self.models[fam] = make_model(fam, p).fit(X, y)
            print(f"  final {fam} fitted ({time.time() - t0:.0f}s)")
        self.params = params
        return self

    def predict(self) -> pd.DataFrame:
        """Probabilities of every model for validation, test and live rows.

        Test and live rows: final models (fitted on train + validation).
        Validation rows: the tuned models fitted on train only, so that validation predictions are
        also out-of-sample and can be used to choose the trading strategy without touching test."""
        d = self.data[self.data["split"].isin(["validation", "test", "live"])]
        out = d[KEEP + ["split"]].copy()
        for fam, model in self.models.items():
            col = f"pred_{fam}"
            out[col] = model.predict_proba(d[self.features])[:, 1].astype("float32")
            if fam in self.val_probs:
                val = self.val_probs[fam][0]
                out.loc[val.index, col] = val.values
        out["pred_ensemble"] = out[[f"pred_{f}" for f in self.models]].mean(axis=1)
        self.predictions = out.reset_index(drop=True)
        return self.predictions

    def scores(self) -> pd.DataFrame:
        rows = []
        p = self.predictions
        for split in ("validation", "test"):
            s = p[p["split"] == split]
            for col in [c for c in p.columns if c.startswith("pred_")]:
                rows.append({"split": split, "model": col, "auc": roc_auc_score(s[TARGET], s[col]),
                             "precision_p50": precision_score(s[TARGET], s[col] > 0.5, zero_division=0),
                             "precision_top10": precision_at_top(s, col, 10),
                             "base_rate": s[TARGET].mean()})
        return pd.DataFrame(rows)

    def feature_importance(self, top: int = 30) -> pd.DataFrame:
        imp = pd.DataFrame({"feature": self.features})
        for fam in ("rf", "xgb"):
            if fam in self.models:
                imp[fam] = self.models[fam].feature_importances_
        return imp.sort_values(imp.columns[-1], ascending=False).head(top)

    # --- persistence ----------------------------------------------------------------
    def persist(self, models_dir=cfg.MODELS_DIR) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features, "params": self.params,
                     "feature_set": getattr(self, "feature_set", None),
                     "val_probs": {f: v[0].values for f, v in self.val_probs.items()},
                     "val_index": self.xy("validation")[2].index.values},
                    models_dir / "models.joblib", compress=3)
        if self.tuning:
            pd.DataFrame(self.tuning).to_csv(cfg.OUTPUT_DIR / "tuning_results.csv", index=False)
        if self.predictions is not None:
            self.predictions.to_parquet(cfg.PROCESSED_DIR / "predictions.parquet", index=False)

    def load(self, models_dir=cfg.MODELS_DIR) -> "TrainModel":
        bundle = joblib.load(models_dir / "models.joblib")
        self.models, self.params = bundle["models"], bundle["params"]
        # validation probabilities are reused only if the validation rows did not change
        val_index = self.xy("validation")[2].index.values
        if len(val_index) == len(bundle["val_index"]) and (val_index == bundle["val_index"]).all():
            self.val_probs = {f: (pd.Series(v, index=val_index), None) for f, v in bundle["val_probs"].items()}
        missing = set(bundle["features"]) - set(self.features)
        for col in missing:            # a dummy level absent from the current data
            self.data[col] = 0
        self.features = bundle["features"]
        return self


if __name__ == "__main__":
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(cfg.PROCESSED_DIR / "dataset.parquet")
    tm = TrainModel(df).prepare()
    del df
    tm.select_features().tune().fit_final()
    tm.predict()
    tm.persist()
    sc = tm.scores()
    sc.to_csv(cfg.OUTPUT_DIR / "model_scores.csv", index=False)
    print(sc.round(4).to_string(index=False))
    print(tm.feature_importance(20).round(4).to_string(index=False))
