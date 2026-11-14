"""
generate_adversarial_examples.py

Generate constrained, model-agnostic adversarial examples for saved Random Forest
and XGBoost binary classifiers (Normal=0, Attack=1).

Method:
  * Query-based greedy feature perturbation (black-box, score-based).
  * For each attack sample, adjust features toward their training-set Normal
    medians, in order of model feature importance.
  * Each feature is constrained to an epsilon fraction of its robust training
    range (Q99-Q01), and clipped to the training-set Q01-Q99 bounds.
  * A candidate perturbation is retained only if it lowers P(Attack=1).
  * Stop once the model predicts Normal, or after the configured iterations.

This is a research/evaluation script. The generated rows are feature-space
examples and may not correspond to physically realizable vehicle trajectories.

Expected files:
  Adversarial/models_binary/randomforest_model.joblib
  Adversarial/models_binary/xgboost_model.joblib
  Adversarial/models_binary/feature_imputer.joblib (optional)
  data/processed/train.parquet
  data/processed/test.parquet

Usage:
  pip install pandas pyarrow scikit-learn xgboost joblib
  python generate_adversarial_examples.py \
     --data_dir results/all/models_binary
      --output_dir adversarial_output \
      --epsilon 0.05 \
      --max_samples 1000

Use --max_samples 0 to process all attack rows (may be slow).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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


def read_features(df, imputer=None):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if imputer is not None:
        X = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)
    else:
        X = X.fillna(X.median())

    return X.astype(np.float32)


def get_training_constraints(train_df):
    """Compute robust bounds, scales, and Normal-class feature medians."""
    train_X = read_features(train_df)

    if TARGET not in train_df.columns:
        raise ValueError(f"Training parquet must contain target column '{TARGET}'.")

    normal_mask = pd.to_numeric(train_df[TARGET], errors="coerce") == 0
    if not normal_mask.any():
        raise ValueError("No Normal (attacker=0) rows found in train.parquet.")

    normal_medians = train_X.loc[normal_mask, FEATURES].median().to_numpy(dtype=float)
    q01 = train_X.quantile(0.01).to_numpy(dtype=float)
    q99 = train_X.quantile(0.99).to_numpy(dtype=float)
    robust_ranges = q99 - q01

    # Prevent zero-range features from receiving a nonzero perturbation budget.
    robust_ranges = np.where(robust_ranges > 1e-12, robust_ranges, 0.0)
    return normal_medians, q01, q99, robust_ranges


def importance_order(model):
    """Tree model feature_importances_ ordering, descending."""
    if not hasattr(model, "feature_importances_"):
        return np.arange(len(FEATURES))
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(FEATURES):
        raise ValueError(
            "Model feature count differs from the script's FEATURES list. "
            "Ensure the saved model was trained with these 12 features in this order."
        )
    return np.argsort(importances)[::-1]


def greedy_query_attack(
    model,
    x_original,
    normal_medians,
    q01,
    q99,
    robust_ranges,
    epsilon=0.05,
    step_fraction=0.20,
    max_iterations=30,
):
    """
    Greedy score-based attack against a tree classifier.

    For each feature (ordered by feature importance), try moving a fraction
    of its allowed budget toward the Normal median. Keep the change if it
    reduces predicted attack probability. Iterate until evasion or exhaustion.
    """
    x_adv = np.asarray(x_original, dtype=np.float32).copy()
    x_orig = x_adv.copy()

    budgets = epsilon * robust_ranges
    budgets = np.maximum(budgets, 0.0)

    lower = np.maximum(q01, x_orig.astype(float) - budgets)
    upper = np.minimum(q99, x_orig.astype(float) + budgets)

    # A single-row prediction helper.
    def attack_score(x):
        return float(model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0, 1])

    initial_score = attack_score(x_adv)
    current_score = initial_score
    success = int(model.predict(pd.DataFrame([x_adv], columns=FEATURES))[0]) == 0

    if success:
        return x_adv, initial_score, current_score, True, 0

    order = importance_order(model)
    iterations_used = 0

    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        improved_any = False

        for feature_idx in order:
            if budgets[feature_idx] <= 0:
                continue

            # Candidate targets progressively move toward the Normal median.
            target = normal_medians[feature_idx]
            direction = target - x_adv[feature_idx]
            if abs(direction) < 1e-12:
                continue

            step = step_fraction * budgets[feature_idx]
            candidate_value = x_adv[feature_idx] + np.sign(direction) * min(
                abs(direction), step
            )
            candidate_value = np.clip(
                candidate_value, lower[feature_idx], upper[feature_idx]
            )

            if abs(candidate_value - x_adv[feature_idx]) < 1e-12:
                # If the small step does not move, try the median within budget.
                candidate_value = np.clip(target, lower[feature_idx], upper[feature_idx])

            candidate = x_adv.copy()
            candidate[feature_idx] = candidate_value
            candidate_score = attack_score(candidate)

            # Accept only strictly improved score.
            if candidate_score + 1e-8 < current_score:
                x_adv = candidate
                current_score = candidate_score
                improved_any = True

                if int(model.predict(pd.DataFrame([x_adv], columns=FEATURES))[0]) == 0:
                    success = True
                    return x_adv, initial_score, current_score, success, iterations_used

        if not improved_any:
            break

    success = int(model.predict(pd.DataFrame([x_adv], columns=FEATURES))[0]) == 0
    return x_adv, initial_score, current_score, success, iterations_used


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial feature examples for saved RF and XGBoost models."
    )
    parser.add_argument("--data_dir", default="results/all/process", help="Directory containing train.parquet and test.parquet (default: data/processed).")
    parser.add_argument("--model_dir", default="results/all/models_binary", help="Directory containing saved model joblib files (default: Adversarial/models_binary).")
    parser.add_argument("--output_dir", default="adversarial_output")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Per-feature perturbation budget as fraction of train Q99-Q01 range.")
    parser.add_argument("--step_fraction", type=float, default=0.20,
                        help="Greedy per-iteration step as fraction of epsilon budget.")
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum attack rows to process; 0 means all attack rows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epsilon < 0:
        raise ValueError("--epsilon must be non-negative.")
    if not (0 < args.step_fraction <= 1):
        raise ValueError("--step_fraction must be in (0, 1].")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.parquet and test.parquet in --data_dir.")

    print("Loading train/test datasets and saved preprocessing imputer...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    imputer_path = model_dir / "feature_imputer.joblib"
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None

    normal_medians, q01, q99, robust_ranges = get_training_constraints(train_df)

    if TARGET not in test_df.columns:
        raise ValueError(f"Test data must contain '{TARGET}'.")

    attack_mask = pd.to_numeric(test_df[TARGET], errors="coerce") == 1
    attack_df = test_df.loc[attack_mask].copy()

    if attack_df.empty:
        raise ValueError("No Attack (attacker=1) rows in test.parquet.")

    if args.max_samples > 0 and len(attack_df) > args.max_samples:
        attack_df = attack_df.sample(
            n=args.max_samples, random_state=args.seed
        ).copy()

    # Preserve source_row if it already exists in the parquet. Otherwise use
    # the original dataframe index as the source-row identifier. Reset the
    # dataframe index without inserting another column (avoids duplicate names).
    if "source_row" not in attack_df.columns:
        attack_df["source_row"] = attack_df.index
    attack_df = attack_df.reset_index(drop=True)
    X_attack = read_features(attack_df, imputer=imputer)
    X_attack_np = X_attack.to_numpy(dtype=np.float32)

    model_files = {
        "RandomForest": model_dir / "randomforest_model.joblib",
        "XGBoost": model_dir / "xgboost_model.joblib",
    }

    summary = []

    for model_name, model_path in model_files.items():
        if not model_path.exists():
            print(f"[SKIP] Saved model not found: {model_path}")
            continue

        print(f"\nLoading {model_name} from {model_path}...")
        model = joblib.load(model_path)

        # Sanity check the model input feature count where available.
        if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURES):
            raise ValueError(
                f"{model_name} expects {model.n_features_in_} features, "
                f"but script supplies {len(FEATURES)}."
            )

        rows = []
        successful = 0
        initially_detected = 0

        for idx, x in enumerate(X_attack_np):
            original_prediction = int(model.predict(pd.DataFrame([x], columns=FEATURES))[0])
            original_probability = float(model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0, 1])
            initially_detected += int(original_prediction == 1)

            x_adv, p_before, p_after, success, iterations = greedy_query_attack(
                model=model,
                x_original=x,
                normal_medians=normal_medians,
                q01=q01,
                q99=q99,
                robust_ranges=robust_ranges,
                epsilon=args.epsilon,
                step_fraction=args.step_fraction,
                max_iterations=args.max_iterations,
            )
            # Count an evasion only when the original model detected the attack
            # and the perturbation changed that prediction to Normal.
            evasion_success = bool(original_prediction == 1 and success)
            successful += int(evasion_success)

            record = {
                "model": model_name,
                "source_row": int(attack_df.loc[idx, "source_row"]),
                "true_label": 1,
                "original_prediction": original_prediction,
                "original_attack_probability": original_probability,
                "adversarial_prediction": int(model.predict(pd.DataFrame([x_adv], columns=FEATURES))[0]),
                "adversarial_attack_probability": float(
                    model.predict_proba(pd.DataFrame([x_adv], columns=FEATURES))[0, 1]
                ),
                "attack_success": evasion_success,
                "iterations": int(iterations),
            }

            for j, feature in enumerate(FEATURES):
                record[f"original_{feature}"] = float(x[j])
                record[f"adv_{feature}"] = float(x_adv[j])
                record[f"delta_{feature}"] = float(x_adv[j] - x[j])

            rows.append(record)

            if (idx + 1) % 100 == 0 or idx + 1 == len(X_attack_np):
                print(f"  Processed {idx + 1:,}/{len(X_attack_np):,} attack samples")

        result_df = pd.DataFrame(rows)
        out_csv = output_dir / f"{model_name.lower()}_adversarial_examples.csv"
        out_parquet = output_dir / f"{model_name.lower()}_adversarial_examples.parquet"
        result_df.to_csv(out_csv, index=False)
        result_df.to_parquet(out_parquet, index=False)

        success_rate = successful / initially_detected if initially_detected else 0.0
        metrics = {
            "model": model_name,
            "attack_method": "greedy_query_based_feature_perturbation",
            "n_attack_samples": int(len(X_attack_np)),
            "initially_predicted_attack": int(initially_detected),
            "successful_evasions": int(successful),
            "evasion_success_rate": float(success_rate),
            "evasion_success_rate_denominator": "samples initially predicted as Attack (1)",
            "epsilon": float(args.epsilon),
            "step_fraction": float(args.step_fraction),
            "max_iterations": int(args.max_iterations),
            "outputs": [str(out_csv), str(out_parquet)],
        }
        summary.append(metrics)

        print(
            f"\n{model_name}: successful evasions={successful:,}/{len(X_attack_np):,} "
            f"({success_rate:.2%})"
        )
        print(f"  CSV: {out_csv}")
        print(f"  Parquet: {out_parquet}")

    summary_path = output_dir / "adversarial_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    if not summary:
        print("No saved model files were found. Check --model_dir and model filenames.")


if __name__ == "__main__":
    main()
