"""
train_rf_xgboost.py
Binary attack detection (Normal=0, Attack=1) using Random Forest and XGBoost.

Expected preprocessing outputs:
    processed/train.parquet
    processed/val.parquet
    processed/test.parquet

Usage:
    pip install pandas pyarrow scikit-learn xgboost matplotlib joblib
    python train_rf_xgboost.py --data_dir processed --output_dir models_binary

Model selection is performed on the validation set using F1-score.
The test set is used only for the final evaluation.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer


FEATURES = [
    "sender_spd",
    "sender_acl",
    "time_delta",
    "distance",
    "heading_change",
    "speed_consistency",
    "accel_consistency",
    "heading_trajectory_consistency",
    "tx_delay",
    "mean_sender_receiver_dist",
    "n_receivers",
    "sender_dist_to_road_edge",
]
TARGET = "attacker"
SEED = 42


def load_split(path: Path):
    df = pd.read_parquet(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    X = df[FEATURES].copy()
    # Coerce any unexpected non-numeric values; infinities become missing.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna() & y.isin([0, 1])
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype("int32").reset_index(drop=True)

    return X, y


def metrics_for_model(model, X, y, name, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities))
        if y.nunique() == 2 else None,
    }
    return metrics, predictions, probabilities


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Normal (0)", "Attack (1)"])
    ax.set_yticks([0, 1], ["Normal (0)", "Attack (1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_importance(model, feature_names, title, path):
    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def choose_by_validation(candidates, X_train, y_train, X_val, y_val, model_name):
    """Fit candidate configurations and select the one with highest validation F1."""
    best_model = None
    best_params = None
    best_metrics = None

    for index, params in enumerate(candidates, start=1):
        print(f"  [{model_name}] Candidate {index}/{len(candidates)}: {params}")

        model = params.pop("_model_factory")(**params) if "_model_factory" in params else None
        if model is None:
            raise RuntimeError("Candidate is missing its model factory.")

        model.fit(X_train, y_train)
        val_metrics, _, _ = metrics_for_model(model, X_val, y_val, model_name)

        print(
            f"    Validation: F1={val_metrics['f1']:.4f}, "
            f"Recall={val_metrics['recall']:.4f}, "
            f"Precision={val_metrics['precision']:.4f}, "
            f"ROC-AUC={val_metrics['roc_auc']:.4f}"
        )

        if best_metrics is None or val_metrics["f1"] > best_metrics["f1"]:
            best_model = model
            best_params = {k: v for k, v in params.items()}
            best_metrics = val_metrics

    return best_model, best_params, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest and XGBoost binary classifiers.")
    parser.add_argument("--data_dir", required=True, help="Directory containing train/val/test parquet files.")
    parser.add_argument("--output_dir", default="models_binary", help="Directory for models, metrics, and plots.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for attack class.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        file_path = data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {file_path}")

    print("Loading processed datasets...")
    X_train, y_train = load_split(data_dir / "train.parquet")
    X_val, y_val = load_split(data_dir / "val.parquet")
    X_test, y_test = load_split(data_dir / "test.parquet")

    print(f"Train: {len(y_train):,} rows | class counts: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Val:   {len(y_val):,} rows | class counts: {y_val.value_counts().sort_index().to_dict()}")
    print(f"Test:  {len(y_test):,} rows | class counts: {y_test.value_counts().sort_index().to_dict()}")

    # Imputation is fitted on training data only to avoid validation/test leakage.
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=FEATURES)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)
    joblib.dump(imputer, output_dir / "feature_imputer.joblib")

    # ----- Random Forest candidate configurations -----
    rf_candidates = [
        {
            "_model_factory": RandomForestClassifier,
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": args.seed,
        },
        {
            "_model_factory": RandomForestClassifier,
            "n_estimators": 300,
            "max_depth": 25,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": args.seed,
        },
        {
            "_model_factory": RandomForestClassifier,
            "n_estimators": 500,
            "max_depth": 35,
            "min_samples_leaf": 1,
            "max_features": 0.7,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": args.seed,
        },
    ]

    print("\nSelecting Random Forest configuration using validation F1...")
    # Copy dicts because candidate evaluation removes the factory key.
    rf_candidates = [dict(c) for c in rf_candidates]
    rf_model, rf_params, rf_val_metrics = choose_by_validation(
        rf_candidates, X_train, y_train, X_val, y_val, "RandomForest"
    )

    # ----- XGBoost candidate configurations -----
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from exc

    xgb_candidates = [
        {
            "_model_factory": XGBClassifier,
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": args.seed,
        },
        {
            "_model_factory": XGBClassifier,
            "n_estimators": 600,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": args.seed,
        },
        {
            "_model_factory": XGBClassifier,
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": args.seed,
        },
    ]

    print("\nSelecting XGBoost configuration using validation F1...")
    xgb_candidates = [dict(c) for c in xgb_candidates]
    xgb_model, xgb_params, xgb_val_metrics = choose_by_validation(
        xgb_candidates, X_train, y_train, X_val, y_val, "XGBoost"
    )

    # ----- Final test evaluation -----
    results = []
    models = {
        "RandomForest": (rf_model, rf_params, rf_val_metrics),
        "XGBoost": (xgb_model, xgb_params, xgb_val_metrics),
    }

    for name, (model, params, val_metrics) in models.items():
        print(f"\n{'=' * 70}\nFinal test evaluation: {name}\n{'=' * 70}")
        test_metrics, test_predictions, test_probabilities = metrics_for_model(
            model, X_test, y_test, name, threshold=args.threshold
        )

        print(json.dumps(test_metrics, indent=2))
        print("\nClassification report:")
        print(classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        ))

        save_confusion_matrix(
            y_test,
            test_predictions,
            f"{name} — Test Confusion Matrix",
            output_dir / f"{name.lower()}_confusion_matrix.png",
        )
        save_feature_importance(
            model,
            FEATURES,
            f"{name} — Feature Importance",
            output_dir / f"{name.lower()}_feature_importance.png",
        )

        # Save model in common format; XGBoost additionally gets native JSON.
        joblib.dump(model, output_dir / f"{name.lower()}_model.joblib")
        if name == "XGBoost":
            model.save_model(str(output_dir / "xgboost_model.json"))

        results.append({
            **test_metrics,
            "validation_f1": float(val_metrics["f1"]),
            "selected_params": json.dumps(params, default=str),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("RF vs XGBoost — Test Results")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]].to_string(index=False))
    print(f"\nSaved models, metrics, and plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
