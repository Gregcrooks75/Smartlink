from __future__ import annotations
from pathlib import Path
import pandas as pd
from mtx.run_config import RunConfig
def validate_transformer_candidates(
    config: RunConfig,
    candidate_files: dict[str, Path],
    group_files: dict[str, Path],
) -> None:
    """
    Validate candidate integrity and candidate coverage.
    """
    for division in candidate_files:
        print(
            "\n"
            + "=" * 80
        )
        print(
            f"VALIDATING CANDIDATES: {division}"
        )
        candidate_df = pd.read_parquet(
            candidate_files[
                division
            ]
        )
        groups = pd.read_parquet(
            group_files[
                division
            ]
        )
        required = {
            "meter_number",
            "division",
            "candidate_transformer",
            "distance_ft",
            "tx_group_id",
            "tx_group_size",
            "tx_group_source",
            "meter_type",
        }
        missing = (
            required
            - set(
                candidate_df.columns
            )
        )
        if missing:
            raise AssertionError(
                f"{division}: missing "
                f"candidate columns "
                f"{sorted(missing)}"
            )
        # ----------------------------------------------
        # Duplicate pair check
        # ----------------------------------------------
        duplicates = (
            candidate_df
            .duplicated(
                [
                    "meter_number",
                    "candidate_transformer",
                ]
            )
        )
        if duplicates.any():
            raise AssertionError(
                f"{division}: duplicate "
                "meter-transformer candidates."
            )
        # ----------------------------------------------
        # Distance check
        # ----------------------------------------------
        if (
            candidate_df[
                "distance_ft"
            ] < 0
        ).any():
            raise AssertionError(
                f"{division}: negative distance."
            )
        max_allowed = (
            config
            .transformer_candidate_radius_ft
            + 0.1
        )
        if (
            candidate_df[
                "distance_ft"
            ]
            > max_allowed
        ).any():
            raise AssertionError(
                f"{division}: candidate beyond "
                "configured radius."
            )
        # ----------------------------------------------
        # Coverage
        # ----------------------------------------------
        input_meters = set(
            groups[
                "meter_number"
            ]
            .astype(str)
        )
        candidate_meters = set(
            candidate_df[
                "meter_number"
            ]
            .astype(str)
        )
        missing_meters = (
            input_meters
            - candidate_meters
        )
        coverage = (
            len(
                candidate_meters
            )
            /
            max(
                len(
                    input_meters
                ),
                1,
            )
        )
        print(
            f"Input meters: "
            f"{len(input_meters):,}"
        )
        print(
            f"Meters with candidates: "
            f"{len(candidate_meters):,}"
        )
        print(
            f"Meters with zero candidates: "
            f"{len(missing_meters):,}"
        )
        print(
            f"Candidate coverage: "
            f"{coverage:.2%}"
        )
        per_meter = (
            candidate_df
            .groupby(
                "meter_number"
            )[
                "candidate_transformer"
            ]
            .nunique()
        )
        print(
            "\nCandidate count per meter:"
        )
        print(
            per_meter.describe()
        )
    print(
        "\nTRANSFORMER CANDIDATE "
        "VALIDATION PASSED")