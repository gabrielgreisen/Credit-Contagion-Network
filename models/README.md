# Models

Training code for the credit-contagion prediction models. One notebook per
model; shared code lives in importable modules next to the notebooks.

## Layout

```
models/
└── training/
    ├── baseline_utils.py          panel loading, splits, metrics, XGB tuning helpers
    ├── chs_features.py            builds the 8 CHS (2008) variables from raw WRDS files
    ├── neighbor_features.py       builds per-layer network features (cached)
    ├── walk_forward.py            the training engine: expanding-window refits with
    │                              per-horizon label embargo, resumable score caches
    ├── chs_logit.ipynb            baseline 1 — CHS reduced-form hazard logit
    ├── xgb_node_features.ipynb    baseline 2 — XGBoost on the firm's own 52 features
    ├── xgb_neighbor_features.ipynb  baseline 3 — XGBoost + hand-built neighbor features
    └── regime_analysis.ipynb      ROC-AUC by horizon x regime/year (+ bankruptcy-only
                                   label robustness; no refitting)
```

Outputs go to `results/models/`: `<model>_metrics.json` with per-horizon
val-era/test blocks, `<model>_predictions.parquet` with walk-forward
`score_h1..score_h8` (2007+), walk-forward score caches
(`wf_scores_<model>.parquet`), tuning-trial CSVs, tree-count selections
(`<model>_n_trees.json`, with the selection fits saved as `<model>_h{h}q.ubj`),
and a final-anchor 4q booster per XGB model for inspection. Feature caches go
to `data/clean/` (`chs_variables_quarterly.parquet`,
`neighbor_features_quarterly.parquet`).

## Common protocol

- **Task:** predict `default_next_hq` (default within h quarters) per
  firm-quarter, at **every horizon h = 1..8**. Labels for h ∈ {1, 4, 8} come
  from phase4; the intermediate horizons are rebuilt with the identical rule
  in `baseline_utils.add_horizon_labels` and validated against the phase4
  columns. The 4-quarter horizon is the primary one for tuning and headline
  numbers; each notebook ends with the full term structure.
- **Training is walk-forward** (`walk_forward.py`): every model is trained
  with an expanding window and annual refit anchors 2007–2025, with a
  per-horizon label embargo — training rows' h-quarter label windows must
  close before the scored year starts. Scores exist for 2007+ only and all of
  2007–2025 is out-of-sample; they are cached per model/horizon
  (`results/models/wf_scores_<model>.parquet`, resumable — delete to retrain).
  `python3 walk_forward.py` precomputes every model's scores from the CLI.
- **Evaluation windows:** val-era 2007–2012 and test 2013–2025. For XGBoost,
  hyperparameters and tree counts are selected on 2007–2012 (see Tuning), so
  treat that window accordingly; 2013+ is untouched by any selection. The CHS
  logit has no tuned quantities — each anchor re-estimates winsorization
  bounds, scaler and coefficients point-in-time.
- **Right-censoring:** per horizon — rows whose h-quarter label window extends
  past the last observed default event are excluded from that horizon's sample
  (`baseline_utils.horizon_mask`).
- **Metrics:** ROC-AUC, PR-AUC (average precision), and capture rates (share of
  defaulters ranked in the top 1/5/10% of scores), reported per horizon.
- **Tuning (XGBoost):** a 20-draw random search with early stopping, selected
  on validation PR-AUC at the 4q horizon (cached to
  `results/models/<model>_trials.csv`), held fixed across anchors and
  horizons. Tree counts are then selected once per horizon by a train
  (1990–2006) fit with early stopping on val (cached to
  `<model>_n_trees.json`) — early stopping inside each anchor's short trailing
  window proved far too noisy.
- **Known timing caveat:** features sit at fiscal quarter-end `datadate` with
  no reporting lag, so absolute short-horizon numbers are optimistic for all
  models equally; CHS (2008) lag accounting data by two months for this
  reason. Not yet applied here.

## The three baselines

1. **CHS logit** (`chs_logit.ipynb`) — Campbell–Hilscher–Szilagyi (2008) logistic
   hazard on eight variables, the standard academic benchmark. Shows where the
   literature stands. (No Merton distance-to-default variant yet: DD is not in
   the 53-feature set and would need a KMV-style iterative solve.)
2. **XGBoost, node features** (`xgb_node_features.ipynb`) — gradient-boosted trees
   on the full 53-feature set minus the credit spread. The strongest
   own-information model; the bar any graph model must clear. The gap over CHS
   isolates the contribution of more features + nonlinearity.
3. **XGBoost + neighbor features** (`xgb_neighbor_features.ipynb`) — same model plus
   hand-built per-layer network features measured at or before t: degree, recent
   neighbor defaults (anchored at the defaulter's last active quarter, since
   defaulters exit the panel), and mean neighbor distress stats for each of the
   7 non-empty layers. The gap over baseline 2 isolates the value of network
   information; a GNN must beat *this* to justify message passing.
