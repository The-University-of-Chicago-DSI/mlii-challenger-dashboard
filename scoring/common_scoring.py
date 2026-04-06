from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr).reshape(-1)

    if suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            return df.iloc[:, 0].to_numpy().reshape(-1)
        for col in ["label", "rating", "target", "y", "prediction", "pred", "y_pred"]:
            if col in df.columns:
                return df[col].to_numpy().reshape(-1)
        return df.iloc[:, -1].to_numpy().reshape(-1)

    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy().reshape(-1)


def load_prediction(path: Path) -> np.ndarray:
    return load_array(path)


def extract_student_name(path: Path, name_overrides: Optional[Dict[str, str]] = None) -> str:
    name_overrides = name_overrides or {}
    stem = path.stem

    match = re.search(r"y_pred[_-]([A-Za-z]+)$", stem, flags=re.IGNORECASE)
    if match:
        name = match.group(1)
    else:
        parts = re.split(r"[_-]+", stem)
        name = parts[-1]

    name = name.strip().title()
    return name_overrides.get(name, name)


def compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if metric_name == "Accuracy":
        if y_pred.dtype.kind in "fc":
            unique_vals = np.unique(y_pred[~pd.isna(y_pred)])
            rounded = np.round(unique_vals).astype(int)
            if set(rounded).issubset(set(range(100))):
                y_pred = np.round(y_pred).astype(int)
        return float(accuracy_score(y_true, y_pred))

    if metric_name == "MSE":
        return float(mean_squared_error(y_true, y_pred))

    if metric_name == "MAE":
        return float(mean_absolute_error(y_true, y_pred))

    raise ValueError(f"Unsupported metric: {metric_name}")


def sort_scores(df: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    valid_df = df[df["status"] == "Valid"].copy()
    invalid_df = df[df["status"] != "Valid"].copy()

    valid_df = valid_df.sort_values("score", ascending=not higher_is_better)
    invalid_df = invalid_df.sort_values("student_name", ascending=True)

    out = pd.concat([valid_df, invalid_df], ignore_index=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def score_submissions(
    *,
    week: str,
    title: str,
    metric_name: str,
    higher_is_better: bool,
    submissions_dir: Path,
    submission_pattern: str,
    target_path: Path,
    model_family_map: Optional[Dict[str, str]] = None,
    name_overrides: Optional[Dict[str, str]] = None,
    expected_length: Optional[int] = None,
    scores_output: Optional[Path] = None,
    leaderboard_rows_output: Optional[Path] = None,
):
    model_family_map = model_family_map or {}
    name_overrides = name_overrides or {}

    y_true = load_array(target_path)
    expected_length = expected_length if expected_length is not None else len(y_true)

    submission_files = sorted(submissions_dir.glob(submission_pattern))

    results = []
    for path in submission_files:
        student_name = extract_student_name(path, name_overrides=name_overrides)

        try:
            y_pred = load_prediction(path)

            if len(y_pred) != expected_length:
                raise ValueError(
                    f"Found input variables with inconsistent numbers of samples: "
                    f"[{len(y_true)}, {len(y_pred)}]"
                )

            score = compute_metric(metric_name, y_true, y_pred)

            results.append(
                {
                    "student_name": student_name,
                    "filename": path.name,
                    "score": score,
                    "status": "Valid",
                    "notes": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "student_name": student_name,
                    "filename": path.name,
                    "score": np.nan,
                    "status": "Invalid format",
                    "notes": str(exc),
                }
            )

    scores_df = pd.DataFrame(results)
    if scores_df.empty:
        scores_df = pd.DataFrame(columns=["student_name", "filename", "score", "status", "notes"])
    scores_df = sort_scores(scores_df, higher_is_better=higher_is_better)

    leaderboard_rows = scores_df.copy()
    leaderboard_rows["week"] = week
    leaderboard_rows["metric_name"] = metric_name
    leaderboard_rows["higher_is_better"] = higher_is_better
    leaderboard_rows["model_family"] = leaderboard_rows["student_name"].map(model_family_map).fillna("")
    leaderboard_rows = leaderboard_rows[
        ["week", "student_name", "score", "metric_name", "higher_is_better", "model_family", "status", "notes"]
    ]

    if scores_output is not None:
        scores_df.to_csv(scores_output, index=False)

    if leaderboard_rows_output is not None:
        leaderboard_rows.to_csv(leaderboard_rows_output, index=False)

    return scores_df, leaderboard_rows


def print_summary(
    *,
    week: str,
    title: str,
    metric_name: str,
    higher_is_better: bool,
    target_path: Path,
    submissions_dir: Path,
    submission_pattern: str,
    scores_df: pd.DataFrame,
    leaderboard_rows: pd.DataFrame,
    scores_output: Optional[Path] = None,
    leaderboard_rows_output: Optional[Path] = None,
) -> None:
    print(f"\n=== Week {week} Scoring: {title} ===")
    print(f"Metric: {metric_name} ({'higher is better' if higher_is_better else 'lower is better'})")
    print(f"Target: {target_path}")
    print(f"Submissions directory: {submissions_dir}")
    print(f"Pattern: {submission_pattern}")
    print(f"Files scored: {len(scores_df)}\n")

    if not scores_df.empty:
        print(scores_df.to_string(index=False))
    else:
        print("No submission files matched the pattern.")

    if scores_output is not None:
        print(f"\nSaved detailed scores to: {scores_output.resolve()}")

    if leaderboard_rows_output is not None:
        print(f"Saved leaderboard rows to: {leaderboard_rows_output.resolve()}")

    print("\nCopy these rows into leaderboard_master.csv:\n")
    print(leaderboard_rows.to_csv(index=False))


def build_default_base_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submissions-dir", type=Path, default=None)
    parser.add_argument("--target-path", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default=None)
