"""Shared helpers for the baseline model notebooks (CHS logit, XGBoost variants).

All baselines operate on the quarterly firm panel:
  - features: data/clean/node_features_quarterly_standardized.parquet
  - labels:   data/clean/default_labels_quarterly.parquet
  - splits:   data/clean/split_assignments.parquet (calendar year x quarter)

Conventions
-----------
* Calendar quarter of a panel row = calendar quarter of `datadate` (fiscal
  quarter end). One row per (gvkey, year, quarter); when a firm reports two
  fiscal quarter-ends inside one calendar quarter we keep the latest.
* Labels: `default_next_{h}q` for h = 1..8 (default within the next h
  quarters). h in {1, 4, 8} come from phase4's parquet; the intermediate
  horizons are rebuilt here with the same rule (earliest default event per
  gvkey, day-count thresholds) and validated against the phase4 columns.
  The primary horizon for tuning and headline numbers is 4 quarters.
* Right-censoring: rows whose h-quarter label window extends past the last
  default event in the data cannot be labelled reliably; filter each
  horizon's sample with `horizon_mask`.
* Splits: pretrain <1990 | train 1990-2006 | val 2007-2012 | test 2013+.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT_ROOT / "data" / "clean"
RESULTS = PROJECT_ROOT / "results" / "models"

HORIZONS = list(range(1, 9))
# day-count thresholds per horizon; 1/4/8 match phase4_default_labels.py exactly
HORIZON_DAYS = {1: 90, 2: 182, 3: 274, 4: 365, 5: 456, 6: 548, 7: 639, 8: 730}
PRIMARY_HORIZON = 4

LABEL_COL = f"default_next_{PRIMARY_HORIZON}q"
LABEL_HORIZON_Q = PRIMARY_HORIZON


def label_col(horizon: int) -> str:
    return f"default_next_{horizon}q"

# The 53 model features used by the graph pipeline (graph_metadata.json).
with open(CLEAN / "graph_metadata.json") as _f:
    GRAPH_METADATA = json.load(_f)
FEATURES_53: list[str] = GRAPH_METADATA["feature_columns"]

# XGBoost baselines exclude the credit-spread feature (it is a target-adjacent
# market price of default risk, reserved for the spread-prediction task).
XGB_FEATURES: list[str] = [c for c in FEATURES_53 if c != "log_credit_spread"]

ID_COLS = ["gvkey", "datadate", "year", "quarter", "split", LABEL_COL]


def _calendar_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["year"] = df["datadate"].dt.year.astype("int16")
    df["quarter"] = df["datadate"].dt.quarter.astype("int8")
    return df


def load_splits() -> pd.DataFrame:
    s = pd.read_parquet(CLEAN / "split_assignments.parquet")
    return s[["year", "quarter", "split"]]


def last_event_date() -> pd.Timestamp:
    de = pd.read_parquet(CLEAN / "default_events.parquet")
    return pd.to_datetime(de["default_date"]).max()


def horizon_mask(df: pd.DataFrame, horizon: int) -> pd.Series:
    """True where the h-quarter forward label window is fully observable."""
    cutoff = last_event_date() - pd.Timedelta(days=HORIZON_DAYS[horizon])
    return df["datadate"] <= cutoff


def add_horizon_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add default_next_{h}q for every h in HORIZONS (phase4 rule).

    Horizons already present (1, 4, 8 from the labels parquet) are kept and
    used to validate the reconstruction; the rest are built from
    default_events with identical logic.
    """
    de = pd.read_parquet(CLEAN / "default_events.parquet")
    first_default = pd.to_datetime(de.groupby("gvkey")["default_date"].min())
    days = (panel["gvkey"].map(first_default) - panel["datadate"]).dt.days

    for h in HORIZONS:
        col = label_col(h)
        rebuilt = ((days > 0) & (days <= HORIZON_DAYS[h])).astype("int8")
        if col in panel.columns:
            mismatch = int((panel[col] != rebuilt).sum())
            if mismatch:
                raise AssertionError(
                    f"{col}: {mismatch} rows disagree with phase4 labels")
        else:
            panel[col] = rebuilt
    return panel


def load_panel(standardized: bool = True) -> pd.DataFrame:
    """Quarterly panel with calendar (year, quarter), split, and labels attached.

    One row per (gvkey, year, quarter), with default_next_{1..8}q columns.
    No right-censoring is applied here: filter each horizon's sample with
    `horizon_mask(panel, h)` before fitting or scoring.
    """
    fname = (
        "node_features_quarterly_standardized.parquet"
        if standardized
        else "node_features_quarterly.parquet"
    )
    panel = pd.read_parquet(CLEAN / fname)
    panel = _calendar_quarter(panel)

    labels = pd.read_parquet(CLEAN / "default_labels_quarterly.parquet")
    labels["datadate"] = pd.to_datetime(labels["datadate"])
    panel = panel.merge(labels, on=["gvkey", "datadate"], how="inner")

    # one row per firm-calendar-quarter (keep the latest fiscal quarter end)
    panel = (
        panel.sort_values("datadate")
        .drop_duplicates(["gvkey", "year", "quarter"], keep="last")
        .reset_index(drop=True)
    )

    panel = panel.merge(load_splits(), on=["year", "quarter"], how="left")
    panel = add_horizon_labels(panel)
    return panel


def split_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: panel[panel["split"] == name] for name in ["train", "val", "test"]}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def capture_at_pct(y_true: np.ndarray, y_score: np.ndarray, pct: float) -> float:
    """Share of all defaulters captured in the top `pct` fraction of scores."""
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan
    k = max(1, int(np.ceil(len(y_score) * pct)))
    top = np.argsort(-y_score)[:k]
    return float(y_true[top].sum() / n_pos)


def evaluate_scores(y_true, y_score) -> dict:
    """Standard metric block for an imbalanced default-prediction task."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
    }
    if y_true.sum() in (0, len(y_true)):
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))
    for pct in (0.01, 0.05, 0.10):
        out[f"capture_top{int(pct * 100)}pct"] = capture_at_pct(y_true, y_score, pct)
    return out


def evaluate_by_split(panel: pd.DataFrame, score_col: str,
                      label_col: str = LABEL_COL) -> pd.DataFrame:
    rows = {}
    for name in ["train", "val", "test"]:
        sub = panel[panel["split"] == name]
        rows[name] = evaluate_scores(sub[label_col].values, sub[score_col].values)
    return pd.DataFrame(rows).T


def evaluate_by_year(panel: pd.DataFrame, score_col: str,
                     label_col: str = LABEL_COL,
                     splits: tuple = ("val", "test")) -> pd.DataFrame:
    sub = panel[panel["split"].isin(splits)]
    rows = {}
    for year, grp in sub.groupby("year"):
        rows[year] = evaluate_scores(grp[label_col].values, grp[score_col].values)
    return pd.DataFrame(rows).T


def save_results(model_name: str, metrics: dict, predictions: pd.DataFrame | None = None,
                 extra: dict | None = None) -> Path:
    """Write metrics JSON (+ optional predictions parquet) to results/models/."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "label": LABEL_COL,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    out = RESULTS / f"{model_name}_metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if predictions is not None:
        predictions.to_parquet(RESULTS / f"{model_name}_predictions.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

def xgb_random_search(X_train, y_train, X_val, y_val, n_iter: int = 20,
                      seed: int = 42, base_params: dict | None = None,
                      verbose: bool = True) -> tuple[list[dict], dict]:
    """Random search with early stopping on validation PR-AUC.

    Returns (all trial records, best trial record). Each record holds the
    sampled params, best_iteration, and val metrics.
    """
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    trials = []
    for i in range(n_iter):
        params = {
            "max_depth": int(rng.integers(3, 9)),
            "learning_rate": float(10 ** rng.uniform(-2, -0.6)),
            "min_child_weight": float(rng.choice([1, 5, 10, 20, 50])),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_lambda": float(10 ** rng.uniform(-1, 1)),
            "scale_pos_weight": float(rng.choice([1.0, np.sqrt(pos_weight), pos_weight])),
        }
        if base_params:
            params.update(base_params)
        model = xgb.XGBClassifier(
            n_estimators=2000,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=seed,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_scores = model.predict_proba(X_val)[:, 1]
        rec = {
            **params,
            "best_iteration": int(model.best_iteration),
            **{f"val_{k}": v for k, v in evaluate_scores(y_val, val_scores).items()
               if k in ("roc_auc", "pr_auc")},
        }
        trials.append(rec)
        if verbose:
            print(f"[{i + 1:2d}/{n_iter}] val PR-AUC={rec['val_pr_auc']:.4f} "
                  f"ROC-AUC={rec['val_roc_auc']:.4f} depth={params['max_depth']} "
                  f"lr={params['learning_rate']:.3f} spw={params['scale_pos_weight']:.1f} "
                  f"iters={rec['best_iteration']}")

    best = max(trials, key=lambda r: r["val_pr_auc"])
    return trials, best


def fit_final_xgb(best: dict, X_train, y_train, X_val, y_val, seed: int = 42):
    """Refit an XGBClassifier with the best sampled params (early stop on val)."""
    import xgboost as xgb

    param_keys = ["max_depth", "learning_rate", "min_child_weight", "subsample",
                  "colsample_bytree", "reg_lambda", "scale_pos_weight"]
    params = {k: best[k] for k in param_keys}
    params["max_depth"] = int(params["max_depth"])  # may arrive as float from CSV
    model = xgb.XGBClassifier(
        n_estimators=2000,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=seed,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
