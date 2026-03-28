"""
================================================================================
MODULE 4: MACHINE LEARNING MODEL + BACKTESTING
File: model.py

Trains an XGBoost classifier on features.csv.
Target: won (1 = horse won, 0 = did not win)
Implements Time-Series cross-validation to backtest on 2025 data.
Saves the trained model to model/xgb_racing_model.json.

Output:
  model/xgb_racing_model.json  — serialised model
  model/feature_importance.csv — feature importance scores
================================================================================
"""

import logging
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FEATURES_CSV  = Path("./data/features.csv")
MODEL_DIR     = Path("./model")
MODEL_PATH    = MODEL_DIR / "xgb_racing_model.json"
IMPORTANCE_CSV= MODEL_DIR / "feature_importance.csv"
METRICS_JSON  = MODEL_DIR / "backtest_metrics.json"

# ── Feature columns fed to the model ─────────────────────────────────────────
ML_FEATURES = [
    "horse_win_rate_5",
    "jockey_win_rate_5",
    "trainer_win_rate_5",  # <--- NEW
    "weight_diff",
    "days_since_last_run",
    "course_dist_win",
    "avg_pos_last3",
    "sp_prob",
    "age",
    "weight_lbs",
    "rpr",                 # <--- NEW
    "or",                  # <--- NEW
]

TARGET = "won"

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "n_estimators":      500,
    "max_depth":         5,
    "learning_rate":     0.03,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,
    "gamma":             1.0,
    "scale_pos_weight":  8,    # compensate for class imbalance (winners rare)
    "random_state":      42,
    "n_jobs":           -1,
    "use_label_encoder": False,
    "verbosity":         0,
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run feature_builder.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True, ignore_index=True)
    log.info(f"Loaded {len(df):,} rows | date range: "
             f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def _select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and y, keeping only available feature columns."""
    avail = [c for c in ML_FEATURES if c in df.columns]
    missing = set(ML_FEATURES) - set(avail)
    if missing:
        log.warning(f"Missing feature columns (will skip): {missing}")
    X = df[avail].copy()
    y = df[TARGET].copy()
    return X, y


# ── Time-Series Backtesting ───────────────────────────────────────────────────

def backtest(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Time-series cross-validation (walk-forward).
    Each fold uses older data for training and newer data for validation.
    Prints per-fold and average metrics.

    Returns dict of average metrics.
    """
    log.info("\n" + "=" * 60)
    log.info("  TIME-SERIES CROSS-VALIDATION (BACKTEST)")
    log.info("=" * 60)

    X, y = _select_features(df)

    tscv    = TimeSeriesSplit(n_splits=n_splits, gap=0)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Skip folds with too few positives
        if y_tr.sum() < 10 or y_val.sum() < 5:
            log.warning(f"  Fold {fold}: Too few winners — skipping")
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs  = model.predict_proba(X_val)[:, 1]
        preds  = (probs >= 0.5).astype(int)

        ll     = log_loss(y_val, probs)
        acc    = accuracy_score(y_val, preds)
        # Accuracy of top-predicted horse per race (simulated betting accuracy)
        auc    = roc_auc_score(y_val, probs)

        # Simulate: for each 'race group' pick top-prob horse, check if won
        val_df = df.iloc[val_idx].copy()
        val_df["pred_prob"] = probs
        # Group by course+time as proxy for race
        if "course" in val_df.columns and "race_time" in val_df.columns:
            val_df["race_group"] = val_df["course"] + "_" + val_df["race_time"].astype(str)
        elif "course" in val_df.columns:
            val_df["race_group"] = val_df["course"] + "_" + val_df["date"].astype(str)
        else:
            val_df["race_group"] = val_df.index // 10   # fallback

        pick_acc = (
            val_df
            .groupby("race_group")
            .apply(lambda g: g.loc[g["pred_prob"].idxmax(), TARGET], include_groups=False)
            .mean()
        )

        date_range = (
            f"{df.iloc[val_idx[0]]['date'].date()} → "
            f"{df.iloc[val_idx[-1]]['date'].date()}"
        )

        log.info(
            f"\n  Fold {fold}  [{date_range}]"
            f"\n    Rows (train/val) : {len(train_idx):,} / {len(val_idx):,}"
            f"\n    Winners in val   : {y_val.sum():,} ({y_val.mean():.2%})"
            f"\n    Log Loss         : {ll:.4f}   ← lower is better"
            f"\n    AUC-ROC          : {auc:.4f}  ← higher is better"
            f"\n    Accuracy         : {acc:.4f}"
            f"\n    Pick Accuracy    : {pick_acc:.4f}  ← top horse wins?"
        )

        metrics.append({
            "fold":         fold,
            "log_loss":     round(ll,       4),
            "auc_roc":      round(auc,      4),
            "accuracy":     round(acc,      4),
            "pick_accuracy":round(pick_acc, 4),
            "val_rows":     len(val_idx),
            "val_winners":  int(y_val.sum()),
        })

    if not metrics:
        log.error("No valid folds produced. Check your data.")
        return {}

    avg = {
        "avg_log_loss":      round(np.mean([m["log_loss"]      for m in metrics]), 4),
        "avg_auc_roc":       round(np.mean([m["auc_roc"]       for m in metrics]), 4),
        "avg_accuracy":      round(np.mean([m["accuracy"]       for m in metrics]), 4),
        "avg_pick_accuracy": round(np.mean([m["pick_accuracy"]  for m in metrics]), 4),
        "folds":             metrics,
    }

    log.info(
        f"\n{'='*60}"
        f"\n  BACKTEST SUMMARY (avg over {len(metrics)} folds)"
        f"\n    Avg Log Loss    : {avg['avg_log_loss']:.4f}"
        f"\n    Avg AUC-ROC     : {avg['avg_auc_roc']:.4f}"
        f"\n    Avg Accuracy    : {avg['avg_accuracy']:.4f}"
        f"\n    Avg Pick Acc    : {avg['avg_pick_accuracy']:.4f}"
        f"\n{'='*60}\n"
    )

    log.info("Interpreting results:")
    if avg["avg_log_loss"] < 0.25:
        log.info("  ✅ Log Loss: GOOD — model is well-calibrated")
    elif avg["avg_log_loss"] < 0.40:
        log.info("  ⚠️  Log Loss: MODERATE — room to improve features")
    else:
        log.info("  ❌ Log Loss: HIGH — model needs more/better data")

    if avg["avg_pick_accuracy"] > 0.25:
        log.info("  ✅ Pick Accuracy >25%: Better than random for ~7-runner fields")
    else:
        log.info("  ⚠️  Pick Accuracy low — try adding more features")

    return avg


# ── Full Model Training ───────────────────────────────────────────────────────

def train_final_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train the final XGBoost model on ALL available data.
    This is the model used for daily predictions.
    """
    log.info("\n" + "=" * 60)
    log.info("  TRAINING FINAL MODEL (all data)")
    log.info("=" * 60)

    X, y = _select_features(df)

    log.info(f"Training on {len(X):,} samples | {y.sum():,} winners ({y.mean():.2%})")
    log.info(f"Features: {list(X.columns)}")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    log.info("✅ Final model trained.")

    # Feature importance
    importance = pd.DataFrame({
        "feature":   X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("\n  Feature Importance:")
    log.info("\n" + importance.to_string(index=False))

    return model, importance


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: xgb.XGBClassifier, importance: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    importance.to_csv(IMPORTANCE_CSV, index=False)
    log.info(f"✅ Model saved → {MODEL_PATH}")
    log.info(f"✅ Importance  → {IMPORTANCE_CSV}")


def load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run model.py first."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    log.info(f"✅ Model loaded from {MODEL_PATH}")
    return model


def get_feature_list() -> list[str]:
    """Return the ML feature columns this model expects."""
    return [c for c in ML_FEATURES if True]   # same as ML_FEATURES


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  MODULE 4 — ML Model + Backtesting")
    log.info("=" * 60)

    df = load_features()

    # Step 1: Backtest (walk-forward CV)
    log.info("\n[1/2] Running backtest …")
    metrics = backtest(df, n_splits=5)

    # Step 2: Train final model on all data
    log.info("\n[2/2] Training final model …")
    model, importance = train_final_model(df)

    # Save
    save_model(model, importance)

    # Save metrics
    if metrics:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_JSON.write_text(json.dumps(metrics, indent=2))
        log.info(f"✅ Metrics → {METRICS_JSON}")

    log.info("\n✅ Module 4 complete. Run main.py for today's predictions.")
