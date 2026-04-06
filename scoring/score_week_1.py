from __future__ import annotations

import argparse
from pathlib import Path

from common_scoring import (
    add_common_cli_args,
    build_default_base_dir,
    print_summary,
    score_submissions,
)

WEEK = "1"
TITLE = "KNN Classification"
METRIC_NAME = "Accuracy"
HIGHER_IS_BETTER = True
SUBMISSION_PATTERN = "*Week_1_y_pred_*.*"
DEFAULT_TARGET_FILENAME = "Week_1_y_test.csv"

MODEL_FAMILY_MAP = {
    # "Smith": "KNN",
}

NAME_OVERRIDES = {
    # "Joliphant": "Oliphant",
}

EXPECTED_LENGTH = None


def main() -> None:
    base_dir = build_default_base_dir()

    parser = argparse.ArgumentParser(description=f"Score Week 1 challenge submissions.")
    add_common_cli_args(parser)
    parser.add_argument("--scores-output", type=Path, default=Path(f"week_1_scores.csv"))
    parser.add_argument("--leaderboard-output", type=Path, default=Path(f"week_1_leaderboard_rows.csv"))
    args = parser.parse_args()

    submissions_dir = args.submissions_dir or (base_dir / "submissions" / f"week_1")
    target_path = args.target_path or (base_dir / "hidden" / DEFAULT_TARGET_FILENAME)
    pattern = args.pattern or SUBMISSION_PATTERN

    scores_df, leaderboard_rows = score_submissions(
        week=WEEK,
        title=TITLE,
        metric_name=METRIC_NAME,
        higher_is_better=HIGHER_IS_BETTER,
        submissions_dir=submissions_dir,
        submission_pattern=pattern,
        target_path=target_path,
        model_family_map=MODEL_FAMILY_MAP,
        name_overrides=NAME_OVERRIDES,
        expected_length=EXPECTED_LENGTH,
        scores_output=args.scores_output,
        leaderboard_rows_output=args.leaderboard_output,
    )

    print_summary(
        week=WEEK,
        title=TITLE,
        metric_name=METRIC_NAME,
        higher_is_better=HIGHER_IS_BETTER,
        target_path=target_path,
        submissions_dir=submissions_dir,
        submission_pattern=pattern,
        scores_df=scores_df,
        leaderboard_rows=leaderboard_rows,
        scores_output=args.scores_output,
        leaderboard_rows_output=args.leaderboard_output,
    )


if __name__ == "__main__":
    main()
