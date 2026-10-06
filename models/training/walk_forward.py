"""Walk-forward (expanding window) training for the three baselines.

Protocol
--------
Annual refit anchors 2007..2025. The model that scores calendar year Y is
trained only on information available before Y starts:

* Label embargo: a training row's h-quarter label window must be fully closed
  before Y-01-01, i.e. datadate <= Y-01-01 - HORIZON_DAYS[h]. Without this,
  rows near the boundary would carry labels realized inside the prediction
  window.
* The training window expands (start fixed at 1990, matching the fixed-split
  baselines); only the end moves forward.
* Hyperparameters are NOT re-tuned per anchor: each model/horizon reuses the
  configuration tuned once on the 2007-2012 validation window, and the tree
  count of its fixed-split refit (early stopping on a short trailing window
  proved too noisy — it stopped at ~50 trees where the full validation window
  chose ~320). For anchors inside 2007-2012 this embeds a little in-window
  selection; scores there are otherwise genuinely out-of-sample, which the
  fixed-split protocol could not offer at all.

Walk-forward scores therefore make 2007+ one continuous out-of-sample period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import baseline_utils as bu
import chs_features as chs

ANCHORS = list(range(2007, 2026))
TRAIN_START = 1990


def embargo_cutoff(anchor_year: int, horizon: int) -> pd.Timestamp:
    return (pd.Timestamp(f"{anchor_year}-01-01")
            - pd.Timedelta(days=bu.HORIZON_DAYS[horizon]))


def _anchor_window(panel: pd.DataFrame, anchor: int) -> pd.Series:
    lo = pd.Timestamp(f"{anchor}-01-01")
    hi = pd.Timestamp(f"{anchor + 1}-01-01")
    return (panel["datadate"] >= lo) & (panel["datadate"] < hi)


def walk_forward_xgb(panel: pd.DataFrame, features: list[str], horizon: int,
                     params: dict, n_trees: int,
                     anchors: list[int] = ANCHORS, verbose: bool = True) -> np.ndarray:
    """Walk-forward scores for one XGB model at one horizon (NaN outside 2007+)."""
    import xgboost as xgb

    col = bu.label_col(horizon)
    scores = np.full(len(panel), np.nan)
    for anchor in anchors:
        cutoff = embargo_cutoff(anchor, horizon)
        tr = panel[(panel["year"] >= TRAIN_START) & (panel["datadate"] <= cutoff)]
        model = xgb.XGBClassifier(
            n_estimators=n_trees, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=42,
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=params["scale_pos_weight"],
        )
        model.fit(tr[features], tr[col].values, verbose=False)
        win = _anchor_window(panel, anchor).values
        if win.any():
            scores[win] = model.predict_proba(panel.loc[win, features])[:, 1]
        if verbose:
            print(f"    h={horizon}q anchor={anchor}: "
                  f"{len(tr):,} train rows ({int(tr[col].sum())} pos), "
                  f"{int(win.sum()):,} scored", flush=True)
    return scores


def walk_forward_chs(df: pd.DataFrame, horizon: int,
                     anchors: list[int] = ANCHORS) -> np.ndarray:
    """Walk-forward CHS logit scores at one horizon.

    `df` must hold the raw (unwinsorized) CHS variables, complete cases only.
    Winsorization bounds (5/95) are re-estimated per anchor on the eligible
    training window — fully point-in-time.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    col = bu.label_col(horizon)
    scores = np.full(len(df), np.nan)
    for anchor in anchors:
        cutoff = embargo_cutoff(anchor, horizon)
        tr = df[(df["year"] >= TRAIN_START) & (df["datadate"] <= cutoff)]
        lo = tr[chs.CHS_VARS].quantile(0.05)
        hi = tr[chs.CHS_VARS].quantile(0.95)

        X_tr = tr[chs.CHS_VARS].clip(lower=lo, upper=hi, axis=1).values
        scaler = StandardScaler().fit(X_tr)
        logit = LogisticRegression(C=1e6, max_iter=2000)
        logit.fit(scaler.transform(X_tr), tr[col].values)

        win = _anchor_window(df, anchor).values
        if win.any():
            X_win = df.loc[win, chs.CHS_VARS].clip(lower=lo, upper=hi, axis=1).values
            scores[win] = logit.predict_proba(scaler.transform(X_win))[:, 1]
    return scores


# ---------------------------------------------------------------------------
# Resumable drivers: scores cached per model, one horizon at a time, so a
# killed run resumes where it stopped. Delete the cache files to recompute.
# ---------------------------------------------------------------------------

def _cache_path(model: str):
    return bu.RESULTS / f"wf_scores_{model}.parquet"


def select_n_trees(panel: pd.DataFrame, model: str, features: list[str],
                   params: dict) -> dict[int, int]:
    """Tree count per horizon: one fit on train (1990-2006) with early stopping
    on val (2007-2012). Early stopping inside each walk-forward anchor is too
    noisy (short trailing windows stop after ~50 trees where the full
    validation window chooses ~320), so the count is selected once here and
    held fixed across anchors. Cached to results/models/<model>_n_trees.json;
    the selection fits are also saved as <model>_h{h}q.ubj.
    """
    import json

    path = bu.RESULTS / f"{model}_n_trees.json"
    if path.exists():
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    counts = {}
    for h in bu.HORIZONS:
        col = bu.label_col(h)
        mh = bu.horizon_mask(panel, h)
        tr = panel[mh & (panel["split"] == "train")]
        va = panel[mh & (panel["split"] == "val")]
        m = bu.fit_final_xgb(params, tr[features], tr[col].values,
                             va[features], va[col].values)
        m.save_model(str(bu.RESULTS / f"{model}_h{h}q.ubj"))
        counts[h] = int(m.best_iteration) + 1
        print(f"  n_trees h={h}q: {counts[h]}", flush=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in counts.items()}, f)
    return counts


def build_xgb_wf_scores(panel: pd.DataFrame, model: str, features: list[str],
                        params: dict, n_trees: dict[int, int]) -> pd.DataFrame:
    """Walk-forward scores for one XGB model, all horizons, cached per horizon."""
    import time

    path = _cache_path(model)
    out = (pd.read_parquet(path) if path.exists()
           else panel[["gvkey", "datadate"]].copy())
    for h in bu.HORIZONS:
        col = f"{model}_wf_h{h}"
        if col in out.columns:
            print(f"  {col}: cached", flush=True)
            continue
        t0 = time.time()
        out[col] = walk_forward_xgb(panel, features, h, params, n_trees[h],
                                    verbose=False)
        out.to_parquet(path, index=False)
        print(f"  {col}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def build_chs_wf_scores(chs_df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward CHS scores, all horizons, cached per horizon."""
    import time

    path = _cache_path("chs_logit")
    out = (pd.read_parquet(path) if path.exists()
           else chs_df[["gvkey", "datadate"]].copy())
    for h in bu.HORIZONS:
        col = f"chs_logit_wf_h{h}"
        if col in out.columns:
            print(f"  {col}: cached", flush=True)
            continue
        t0 = time.time()
        out[col] = walk_forward_chs(chs_df, h)
        out.to_parquet(path, index=False)
        print(f"  {col}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def main():
    """Compute all walk-forward scores (resumable; run as a script)."""
    import neighbor_features as nf

    panel = bu.load_panel()
    nbr = nf.build_neighbor_features(panel)
    nbr_cols = nf.neighbor_feature_columns(nbr)
    panel = panel.merge(nbr, on=["gvkey", "year", "quarter"], how="left")

    label_cols = [bu.label_col(h) for h in bu.HORIZONS]
    chs_vars = chs.build_chs_variables(panel[["gvkey", "datadate", "permno"]])
    chs_df = panel[["gvkey", "datadate", "year", "split"] + label_cols].merge(
        chs_vars, on=["gvkey", "datadate"], how="left")
    chs_df = chs_df.dropna(subset=chs.CHS_VARS).reset_index(drop=True)

    print("CHS logit walk-forward:", flush=True)
    build_chs_wf_scores(chs_df)

    feature_sets = {"xgb_node_features": bu.XGB_FEATURES,
                    "xgb_neighbor_features": bu.XGB_FEATURES + nbr_cols}
    for model, feats in feature_sets.items():
        params = (pd.read_csv(bu.RESULTS / f"{model}_trials.csv")
                  .sort_values("val_pr_auc", ascending=False).iloc[0].to_dict())
        n_trees = select_n_trees(panel, model, feats, params)
        print(f"{model} walk-forward (trees per horizon: {n_trees}):", flush=True)
        build_xgb_wf_scores(panel, model, feats, params, n_trees)
    print("WF SCORES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
