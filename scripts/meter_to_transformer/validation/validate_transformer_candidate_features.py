from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
KEY = [
    "meter_number",
    "candidate_transformer",
]
def pairset(
    df: pd.DataFrame,
) -> set[tuple[str, str]]:
    return set(
        map(
            tuple,
            df[
                KEY
            ]
            .astype(str)
            .to_numpy(),
        )
    )
def validate_transformer_candidate_features(
    feature_files: dict[str, Path],
    candidate_files: dict[str, Path],
) -> None:
    for division in feature_files:
        print(
            "\n"
            + "=" * 80
        )
        print(
            f"VALIDATING FEATURES: {division}"
        )
        before = pd.read_parquet(
            candidate_files[
                division
            ]
        )
        after = pd.read_parquet(
            feature_files[
                division
            ]
        )
        if (
            pairset(before)
            !=
            pairset(after)
        ):
            raise AssertionError(
                f"{division}: candidate population "
                "changed during feature engineering."
            )
        if (
            after
            .duplicated(
                KEY
            )
            .any()
        ):
            raise AssertionError(
                f"{division}: duplicate "
                "meter-transformer pairs."
            )
        closest_counts = (
            after
            .groupby(
                "meter_number"
            )[
                "is_closest"
            ]
            .sum()
        )
        if not (
            closest_counts
            == 1
        ).all():
            raise AssertionError(
                f"{division}: not exactly one "
                "closest transformer per meter."
            )
        numeric = (
            after
            .select_dtypes(
                include="number"
            )
        )
        infinite = (
            np.isinf(
                numeric
            )
        )
        if infinite.any().any():
            bad_columns = (
                infinite
                .any()
            )
            bad_columns = (
                bad_columns[
                    bad_columns
                ]
                .index
                .tolist()
            )
            raise AssertionError(
                f"{division}: infinite values "
                f"in {bad_columns}"
            )
        percentage_columns = [
            "group_transformer_representation_pct",
            "group_nearest_transformer_share_pct",
        ]
        for col in percentage_columns:
            if col not in after.columns:
                continue
            values = (
                after[
                    col
                ]
                .dropna()
            )
            if not (
                values
                .between(
                    0,
                    100,
                )
                .all()
            ):
                raise AssertionError(
                    f"{division}: invalid values "
                    f"in {col}"
                )
        print(
            f"Rows: {len(after):,}"
        )
        if (
            "transformer_bank_check"
            in after.columns
        ):
            print(
                "\nTransformer bank check:"
            )
            print(
                after[
                    "transformer_bank_check"
                ]
                .value_counts(
                    dropna=False
                )
            )
    print(
        "\nCANDIDATE FEATURE "
        "VALIDATION PASSED"
    )