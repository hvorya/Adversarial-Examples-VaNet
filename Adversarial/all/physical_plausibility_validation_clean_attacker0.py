"""Extract successful RandomForest adversarial attacks and run physical/consistency checks.

Input accepts randomforest_adversarial_examples.xls/.xlsx/.csv. Only rows whose
attack_success value is True are processed and written to output.

Statistical checks require train.parquet (normal-class feature distribution).
Coordinate/trajectory and exact sequential-feature recomputation checks require
 test.parquet to map source_row to the original preprocessing records.

Example:
  python physical_plausibility_check_with_consistency_updated.py \
      --input randomforest_adversarial_examples.xls \
      --data_dir data/processed \
      --output_dir physical_check_output
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "sender_spd", "sender_acl", "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency", "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]
LABEL = "attacker"
ALIASES = {
    "scenario": ["scenario", "scenario_id", "scenarioid", "scene"],
    "sender": ["sender", "sender_id", "station_id", "vehicle_id", "pseudonym", "pseudonym_id"],
    "time": ["sendTime", "send_time", "timestamp", "time", "sim_time", "event_time", "message_time"],
    "x": ["x", "sender_pos_x", "pos_x", "position_x", "sender_x", "x_position"],
    "y": ["y", "sender_pos_y", "pos_y", "position_y", "sender_y", "y_position"],
    "receiver_x": ["receiver_x", "rx", "receiver_mean_pos_x", "receiver_pos_x", "receiver_position_x", "x_receiver", "x_rx"],
    "receiver_y": ["receiver_y", "ry", "receiver_mean_pos_y", "receiver_pos_y", "receiver_position_y", "y_receiver", "y_ry"],
    "speed": ["speed", "sender_spd", "velocity", "sender_speed"],
}


def read_table(path: Path) -> pd.DataFrame:
    """Read Excel (.xls/.xlsx) or CSV based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension {suffix!r}; use .xls, .xlsx, or .csv")


def detect_col(df: pd.DataFrame, aliases: list[str]):
    cols = {str(c).strip().lower(): c for c in df.columns}
    return next((cols[a.lower()] for a in aliases if a.lower() in cols), None)


def numeric_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    names = [f"{prefix}{feature}" for feature in FEATURES]
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    out = df[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out.columns = FEATURES
    return out


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = x - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv_cov, delta), 0))


def bool_success(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "successful"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="randomforest_adversarial_examples.xls",
                        help="RandomForest adversarial examples file (.xls/.xlsx/.csv)")
    parser.add_argument("--data_dir", default="data/processed",
                        help="Directory containing train.parquet and test.parquet")
    parser.add_argument("--output_dir", default="physical_check_output")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="Quantile threshold for normal-class Mahalanobis distance")
    parser.add_argument("--bounds_tolerance", type=float, default=0.0,
                        help="Fraction of Q99-Q01 range allowed beyond feature bounds")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Absolute tolerance for formula and geometry residuals")
    parser.add_argument("--max_speed_mps", type=float, default=55.56,
                        help="Maximum derived trajectory speed, m/s (default ~200 km/h)")
    parser.add_argument("--reported_speed_tolerance_mps", type=float, default=8.0,
                        help="Allowed difference between reported and derived speed, m/s")
    parser.add_argument("--clean_data", action="store_true",
                        help="Validate only attacker==0 rows from the supplied test CSV; use their clean features as the baseline.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Helpful fallback: use a matching CSV/XLSX if requested .xls is absent.
        candidates = [input_path.with_suffix(ext) for ext in (".xlsx", ".csv", ".xls")]
        input_path = next((candidate for candidate in candidates if candidate.exists()), input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_examples = read_table(input_path)
    if args.clean_data:
        required_clean_cols = {LABEL, "source_row", *FEATURES}
        absent = required_clean_cols - set(all_examples.columns)
        if absent:
            raise ValueError(f"Clean-data input is missing required columns: {sorted(absent)}")
        clean_mask = pd.to_numeric(all_examples[LABEL], errors="coerce").eq(0)
        clean_rows = all_examples.loc[clean_mask].copy().reset_index(drop=True)
        successful = pd.DataFrame({
            "attack_success": True,
            "source_row": pd.to_numeric(clean_rows["source_row"], errors="coerce"),
        })
        for feature in FEATURES:
            successful[f"adv_{feature}"] = pd.to_numeric(clean_rows[feature], errors="coerce")
        # For unchanged clean rows, no feature delta was introduced.
        successful["delta_heading_trajectory_consistency"] = 0.0
        successful["delta_heading_change"] = 0.0
        successful["delta_time_delta"] = 0.0
        print(f"Read {len(all_examples)} test rows; selected {len(clean_rows)} clean rows with {LABEL}==0.")
    else:
        required_cols = {"attack_success", "source_row"}
        absent = required_cols - set(all_examples.columns)
        if absent:
            raise ValueError(f"Input is missing required columns: {sorted(absent)}")
        successful = all_examples[bool_success(all_examples["attack_success"])].copy().reset_index(drop=True)
        if successful.empty:
            raise ValueError("No rows with attack_success=True were found in the input file.")
        print(f"Read {len(all_examples)} total attack rows; extracted {len(successful)} successful attacks.")

    # Initialize each check as unavailable, and never treat unavailable checks as a pass.
    checks = [
        "source_row_valid", "feature_bounds_pass", "normal_mahalanobis_pass",
        "euclidean_distance_consistency_pass", "trajectory_dynamics_pass",
        "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass",
        "heading_trajectory_dependency_heuristic_pass",
    ]
    for check in checks:
        successful[check] = pd.NA
    successful["check_availability_notes"] = ""

    data_dir = Path(args.data_dir)
    train_path = next((p for p in [data_dir / "train.csv", data_dir / "train.parquet"] if p.exists()), None)
    test_path = next((p for p in [data_dir / "test(1).csv", data_dir / "test.csv", data_dir / "test.parquet"] if p.exists()), None)
    train = test = None

    def load_dataset(path):
        if path is None:
            return None
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            warnings.warn(f"Parquet support unavailable ({exc}). Install pyarrow or fastparquet.")
            return None

    train = load_dataset(train_path)
    test = load_dataset(test_path)

    # Statistical checks: feature Q01-Q99 range and Normal-class Mahalanobis distance.
    if train is not None and LABEL in train.columns:
        train_x = numeric_features(train)
        medians = train_x.median()
        train_x = train_x.fillna(medians)
        normal_x = train_x.loc[pd.to_numeric(train[LABEL], errors="coerce").eq(0)]
        if len(normal_x) >= 2:
            q01, q99 = train_x.quantile(.01), train_x.quantile(.99)
            scale = (q99 - q01).replace(0, 1e-9)
            lower = q01 - args.bounds_tolerance * scale
            upper = q99 + args.bounds_tolerance * scale
            n = normal_x.to_numpy(float)
            mean = n.mean(axis=0)
            cov = np.cov(n, rowvar=False)
            ridge = max(float(np.trace(cov) / len(FEATURES)) * 1e-6, 1e-9)
            inv_cov = np.linalg.pinv(cov + np.eye(len(FEATURES)) * ridge)
            threshold = float(np.quantile(mahalanobis_distance(n, mean, inv_cov), args.quantile))

            adv_x = numeric_features(successful, prefix="adv_").fillna(medians)
            bounds_pass = ((adv_x >= lower) & (adv_x <= upper)).all(axis=1)
            md = mahalanobis_distance(adv_x.to_numpy(float), mean, inv_cov)
            md_pass = md <= threshold
            successful["feature_bounds_pass"] = bounds_pass
            successful["normal_mahalanobis_distance"] = md
            successful["normal_mahalanobis_threshold"] = threshold
            successful["normal_mahalanobis_pass"] = md_pass
        else:
            successful["check_availability_notes"] += "Insufficient Normal training rows for Mahalanobis; "
    else:
        successful["check_availability_notes"] += "train.parquet unavailable or lacks attacker label; statistical checks unavailable; "

    # Original source-row mapping enables geometric and sequence checks.
    if test is None:
        successful["check_availability_notes"] += "test.parquet unavailable; coordinate/trajectory/formula checks unavailable; "
    else:
        source = pd.to_numeric(successful["source_row"], errors="coerce")
        # source_row is a data column created by preprocessing; it is not necessarily
        # the dataframe row index after CSV export/shuffling. Map by that key when present.
        if "source_row" in test.columns:
            source_key = pd.to_numeric(test["source_row"], errors="coerce")
            source_lookup = test.assign(_source_row_numeric=source_key).dropna(subset=["_source_row_numeric"])
            source_lookup = source_lookup.drop_duplicates("_source_row_numeric").set_index("_source_row_numeric", drop=False)
            valid = source.notna() & source.isin(source_lookup.index)
            raw = source_lookup.reindex(source.where(valid).to_numpy()).reset_index(drop=True)
        else:
            valid = source.notna() & source.astype("Int64").isin(test.index)
            raw = test.reindex(source.where(valid).astype("Int64").to_numpy()).reset_index(drop=True)
        successful["source_row_valid"] = valid.to_numpy()
        detected = {key: detect_col(test, aliases) for key, aliases in ALIASES.items()}

        # Receiver-distance consistency: compare the adversarial feature to the
        # same preprocessing feature in the source row. This avoids the invalid
        # comparison between trajectory displacement and sender-to-mean-receiver
        # coordinate distance (and avoids recomputing a mean of distances from a
        # distance-to-mean-coordinate approximation).
        receiver_distance_feature = detect_col(test, ["mean_sender_receiver_dist"])
        if receiver_distance_feature and "adv_mean_sender_receiver_dist" in successful.columns:
            raw_receiver_dist = pd.to_numeric(raw[receiver_distance_feature], errors="coerce")
            adv_receiver_dist = pd.to_numeric(successful["adv_mean_sender_receiver_dist"], errors="coerce")
            residual = (adv_receiver_dist - raw_receiver_dist).abs()
            successful["recomputed_euclidean_distance"] = raw_receiver_dist
            successful["euclidean_distance_residual"] = residual
            successful["euclidean_distance_consistency_pass"] = residual.le(args.tolerance) & np.isfinite(residual)
        else:
            successful["check_availability_notes"] += "Source mean_sender_receiver_dist unavailable for like-for-like Euclidean check; "

        # Exact sequential engineered-feature formula checks and dependency heuristic.
        seq_available = all(detected[key] for key in ["scenario", "sender", "x", "y", "speed"])
        heading_col = detect_col(test, ["sender_hed", "sender_heading", "heading", "hed"])
        accel_col = detect_col(test, ["sender_acl", "sender_accel", "acceleration", "acl"])
        sendtime_col = detect_col(test, ["sendTime", "send_time"])
        seq_available = seq_available and bool(heading_col and accel_col and sendtime_col)
        if seq_available:
            base = test.copy()
            group_cols = [detected["scenario"], detected["sender"]]
            numeric_cols = [sendtime_col, detected["x"], detected["y"], detected["speed"], accel_col, heading_col]
            for col in numeric_cols:
                base[col] = pd.to_numeric(base[col], errors="coerce")
            base["_source_index"] = pd.to_numeric(base["source_row"], errors="coerce") if "source_row" in base.columns else base.index
            base = base.sort_values(group_cols + [sendtime_col], kind="stable")
            grouped = base.groupby(group_cols, dropna=False, sort=False)
            for field, col in [("_prev_speed", detected["speed"]), ("_prev_x", detected["x"]),
                               ("_prev_y", detected["y"]), ("_prev_time", sendtime_col)]:
                base[field] = grouped[col].shift(1)
            state = base.set_index("_source_index").reindex(source.where(valid).astype("Int64").to_numpy()).reset_index(drop=True)
            # Formula validation uses the adversarial values for current-row
            # engineered inputs, and the true preceding source-row speed for
            # acceleration. The preprocessing threshold is 0.05 seconds.
            adv_speed = pd.to_numeric(successful["adv_sender_spd"], errors="coerce")
            adv_accel = pd.to_numeric(successful["adv_sender_acl"], errors="coerce")
            adv_dist = pd.to_numeric(successful["adv_distance"], errors="coerce")
            adv_dt = pd.to_numeric(successful["adv_time_delta"], errors="coerce")
            reliable = adv_dt.ge(0.05) & np.isfinite(adv_dt)

            expected_speed_consistency = np.where(reliable, adv_speed - adv_dist / adv_dt.where(reliable), 0.0)
            actual_speed_consistency = pd.to_numeric(successful["adv_speed_consistency"], errors="coerce")
            speed_residual = (actual_speed_consistency - expected_speed_consistency).abs()
            successful["speed_consistency_formula_residual"] = speed_residual
            successful["speed_consistency_formula_pass"] = speed_residual.le(args.tolerance) & np.isfinite(speed_residual)

            prev_speed = pd.to_numeric(state["_prev_speed"], errors="coerce")
            has_prev = prev_speed.notna()
            derived_accel = np.where(reliable & has_prev, (adv_speed - prev_speed) / adv_dt.where(reliable & has_prev), 0.0)
            expected_accel_consistency = np.where(reliable & has_prev, adv_accel - derived_accel, 0.0)
            actual_accel_consistency = pd.to_numeric(successful["adv_accel_consistency"], errors="coerce")
            accel_residual = (actual_accel_consistency - expected_accel_consistency).abs()
            successful["accel_consistency_formula_residual"] = accel_residual
            successful["accel_consistency_formula_pass"] = accel_residual.le(args.tolerance) & np.isfinite(accel_residual)

            dx = pd.to_numeric(raw[detected["x"]], errors="coerce") - pd.to_numeric(state["_prev_x"], errors="coerce")
            dy = pd.to_numeric(raw[detected["y"]], errors="coerce") - pd.to_numeric(state["_prev_y"], errors="coerce")
            displacement = np.sqrt(dx ** 2 + dy ** 2)
            movement_heading = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0
            reported_heading = pd.to_numeric(raw[heading_col], errors="coerce") % 360.0
            angle_diff = (reported_heading - movement_heading).abs()
            expected_heading_consistency = np.minimum(angle_diff, 360.0 - angle_diff)
            expected_heading_consistency = expected_heading_consistency.where(displacement >= 0.1, 0.0)
            expected_heading_consistency = expected_heading_consistency.where(state["_prev_x"].notna() & state["_prev_y"].notna(), 0.0)
            actual_heading_consistency = pd.to_numeric(successful["adv_heading_trajectory_consistency"], errors="coerce")
            heading_residual = (actual_heading_consistency - expected_heading_consistency).abs()
            successful["heading_trajectory_consistency_formula_residual"] = heading_residual
            successful["heading_trajectory_consistency_formula_pass"] = heading_residual.le(args.tolerance) & np.isfinite(heading_residual)
            successful["heading_trajectory_consistency_formula_inconsistency_flag"] = ~successful["heading_trajectory_consistency_formula_pass"].astype(bool)
            successful["sequential_formula_checks_available"] = True
        else:
            successful["check_availability_notes"] += "Required raw columns for exact sequential formula checks unavailable; "
            successful["sequential_formula_checks_available"] = False

        # Dependency heuristic flags changed heading trajectory consistency if proxy fields
        # heading_change and time_delta are both unmodified. Formula recomputation is stronger.
        htc_changed = pd.to_numeric(successful.get("delta_heading_trajectory_consistency"), errors="coerce").abs().gt(args.tolerance)
        proxy_inputs_changed = pd.concat([
            pd.to_numeric(successful.get("delta_heading_change"), errors="coerce").abs().gt(args.tolerance),
            pd.to_numeric(successful.get("delta_time_delta"), errors="coerce").abs().gt(args.tolerance),
        ], axis=1).any(axis=1)
        dependency_flag = htc_changed & ~proxy_inputs_changed
        successful["heading_trajectory_dependency_inconsistency_flag"] = dependency_flag
        successful["heading_trajectory_dependency_heuristic_pass"] = ~dependency_flag
        if "heading_trajectory_consistency_formula_inconsistency_flag" in successful.columns:
            successful["heading_trajectory_consistency_inconsistency_flag"] = (
                successful["heading_trajectory_consistency_formula_inconsistency_flag"] | dependency_flag
            )

        # Physical trajectory dynamics is calculated over the FULL source test
        # table, not merely the subset of successful adversarial source rows.
        # Results are then mapped back using source_row to preserve predecessor
        # context and avoid false jumps between sparsely selected attack rows.
        trajectory_available = all(detected[key] for key in ["sender", "time", "x", "y"])
        if trajectory_available and "source_row" in test.columns:
            full = test.copy()
            group = [detected["sender"]]
            if detected["scenario"]:
                group.insert(0, detected["scenario"])
            time_col, x_col, y_col = detected["time"], detected["x"], detected["y"]
            numeric = [time_col, x_col, y_col] + ([detected["speed"]] if detected["speed"] else [])
            for col in numeric:
                full[col] = pd.to_numeric(full[col], errors="coerce")
            full["_source_row_key"] = pd.to_numeric(full["source_row"], errors="coerce")
            full = full.sort_values(group + [time_col], kind="stable")
            fg = full.groupby(group, dropna=False, sort=False)
            dt_dyn = fg[time_col].diff() * 1e-9
            dx_dyn, dy_dyn = fg[x_col].diff(), fg[y_col].diff()
            derived_speed = np.sqrt(dx_dyn ** 2 + dy_dyn ** 2) / dt_dyn.where(dt_dyn > 0)

            # In clean-data mode, calibrate the two trajectory thresholds from
            # attacker==0 observations only, using the 99th percentile of
            # derived speed and absolute reported-vs-derived speed residual.
            max_speed_threshold = args.max_speed_mps
            reported_speed_tolerance = args.reported_speed_tolerance_mps
            if args.clean_data and LABEL in full.columns and detected["speed"]:
                clean_cal_mask = pd.to_numeric(full[LABEL], errors="coerce").eq(0)
                reliable_cal = clean_cal_mask & dt_dyn.ge(0.05) & np.isfinite(derived_speed)
                clean_speed_values = derived_speed.loc[reliable_cal]
                clean_reported = pd.to_numeric(full.loc[reliable_cal, detected["speed"]], errors="coerce")
                clean_residual_values = (clean_reported - clean_speed_values).abs().replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_speed_values) and len(clean_residual_values):
                    max_speed_threshold = float(clean_speed_values.quantile(0.99))
                    reported_speed_tolerance = float(clean_residual_values.quantile(0.99))
                successful["trajectory_max_speed_threshold_mps"] = max_speed_threshold
                successful["trajectory_reported_speed_tolerance_mps"] = reported_speed_tolerance

            valid_dynamics = dt_dyn.gt(0) & np.isfinite(derived_speed)
            dynamics_ok = pd.Series(pd.NA, index=full.index, dtype="boolean")
            dynamics_result = derived_speed.le(max_speed_threshold)
            if detected["speed"]:
                reported = pd.to_numeric(full[detected["speed"]], errors="coerce")
                dynamics_result &= reported.isna() | ((reported - derived_speed).abs() <= reported_speed_tolerance)
            dynamics_ok.loc[valid_dynamics] = dynamics_result.loc[valid_dynamics].astype("boolean")
            full["_dynamics_ok"] = dynamics_ok
            dynamics_lookup = full.dropna(subset=["_source_row_key"]).drop_duplicates("_source_row_key").set_index("_source_row_key")["_dynamics_ok"]
            selected_keys = pd.to_numeric(successful["source_row"], errors="coerce")
            successful["trajectory_dynamics_pass"] = selected_keys.map(dynamics_lookup).astype("boolean")
            successful.loc[successful["trajectory_dynamics_pass"].isna(), "check_availability_notes"] += "No valid predecessor/time delta for dynamics; "
        else:
            successful["check_availability_notes"] += "Sender/time/x/y/source_row columns unavailable for trajectory dynamics check; "

    # Report a graded assessment rather than a brittle all-checks-must-pass
    # binary verdict. Statistical screens and the heading dependency heuristic
    # are advisory; the primary score reflects the available formula/dynamics
    # checks. Missing checks remain explicitly indeterminate.
    all_check_cols = [
        "feature_bounds_pass", "normal_mahalanobis_pass", "euclidean_distance_consistency_pass",
        "trajectory_dynamics_pass", "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass", "heading_trajectory_dependency_heuristic_pass",
    ]
    primary_check_cols = [
        "euclidean_distance_consistency_pass", "trajectory_dynamics_pass",
        "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass",
    ]
    successful["checks_available_count"] = successful[all_check_cols].notna().sum(axis=1)
    successful["checks_failed_count"] = successful[all_check_cols].apply(
        lambda row: sum(value is False or (isinstance(value, (np.bool_, bool)) and not value) for value in row), axis=1
    )
    successful["all_available_checks_pass"] = successful[all_check_cols].apply(
        lambda row: all(pd.isna(value) or bool(value) for value in row), axis=1
    )
    successful["complete_check_coverage"] = successful[all_check_cols].notna().all(axis=1)
    successful["primary_checks_available_count"] = successful[primary_check_cols].notna().sum(axis=1)
    successful["primary_checks_pass_count"] = successful[primary_check_cols].apply(
        lambda row: sum(bool(value) for value in row if pd.notna(value)), axis=1
    )
    successful["primary_checks_pass_rate"] = (
        successful["primary_checks_pass_count"] / successful["primary_checks_available_count"].replace(0, np.nan)
    )
    successful["primary_checks_complete_coverage"] = successful[primary_check_cols].notna().all(axis=1)
    successful["physical_plausibility_assessment"] = np.select(
        [
            successful["primary_checks_available_count"].eq(0),
            ~successful["primary_checks_complete_coverage"],
            successful["primary_checks_pass_rate"].ge(0.8),
            successful["primary_checks_pass_rate"].lt(0.5),
        ],
        ["Indeterminate - no primary checks available", "Indeterminate - partial primary coverage",
         "Mostly consistent - full primary coverage", "Potential inconsistencies - primary checks"],
        default="Mixed evidence - review failed checks",
    )
    # Retain legacy boolean as a conservative indicator, but do not treat it as
    # the headline scientific conclusion. It requires every check to be present.
    successful["physically_plausible_screen"] = (
        successful["primary_checks_complete_coverage"] & successful["primary_checks_pass_rate"].eq(1.0)
    )

    # Export all extracted successes, with full columns and additional check results.
    csv_out = output_dir / "randomforest_successful_attacks_physical_checks.csv"
    xlsx_out = output_dir / "randomforest_successful_attacks_physical_checks.xlsx"
    summary_out = output_dir / "physical_plausibility_summary.json"
    successful.to_csv(csv_out, index=False)
    summary = {
        "input_file": str(input_path),
        "input_rows": int(len(all_examples)),
        "successful_attack_rows_extracted": int(len(successful)),
        "successful_rows_passing_all_available_checks": int(successful["all_available_checks_pass"].sum()),
        "successful_rows_passing_all_checks_with_complete_coverage": int(successful["physically_plausible_screen"].sum()),
        "physical_plausibility_assessment_counts": successful["physical_plausibility_assessment"].value_counts(dropna=False).to_dict(),
        "mean_primary_checks_pass_rate": float(successful["primary_checks_pass_rate"].mean()),
        "check_columns": all_check_cols,
        "check_coverage_complete_count": int(successful["complete_check_coverage"].sum()),
        "notes": "A check marked NA is unavailable. Headline assessment is graded on primary geometry/trajectory/formula checks; statistical bounds/Mahalanobis and dependency heuristic are advisory. Legacy physically_plausible_screen is a conservative boolean, not the sole scientific conclusion.",
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        successful.to_excel(writer, sheet_name="Successful Attacks Checks", index=False)
    print(f"Extracted successful attacks: {len(successful)} / {len(all_examples)}")
    print(f"CSV results: {csv_out}")
    print(f"Excel results: {xlsx_out}")
    print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
    main()
