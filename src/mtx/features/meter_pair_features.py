"""Create selectable AMI features for nearby meter pairs."""

from collections.abc import Sequence
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


def similarity_percentage(
    first_value: object,
    second_value: object,
) -> float:
    """Calculate case-insensitive string similarity."""
    if pd.isna(first_value) or pd.isna(second_value):
        return np.nan

    first_text = str(first_value).upper().strip()
    second_text = str(second_value).upper().strip()

    return round(
        SequenceMatcher(
            None,
            first_text,
            second_text,
        ).ratio()
        * 100,
        2,
    )


def exact_match(
    first_series: pd.Series,
    second_series: pd.Series,
) -> pd.Series:
    """Compare two columns while preserving missing values."""
    result = pd.Series(
        pd.NA,
        index=first_series.index,
        dtype="boolean",
    )

    valid = (
        first_series.notna()
        & second_series.notna()
    )

    first_clean = (
        first_series
        .astype("string")
        .str.upper()
        .str.strip()
    )

    second_clean = (
        second_series
        .astype("string")
        .str.upper()
        .str.strip()
    )

    result.loc[valid] = (
        first_clean.loc[valid]
        == second_clean.loc[valid]
    )

    return result


def build_meter_pair_features(
    pair_df: pd.DataFrame,
    ami_df: pd.DataFrame,
    selected_ami_columns: Sequence[str],
    similarity_columns: Sequence[str] = (),
    exact_match_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Add selected AMI attributes and pair-comparison features.

    Parameters
    ----------
    pair_df:
        Nearby meter pairs containing meter_1 and meter_2.
    ami_df:
        AMI meter data.
    selected_ami_columns:
        AMI columns to attach to both meters.
    similarity_columns:
        Selected columns requiring percentage similarity.
    exact_match_columns:
        Selected columns requiring exact-match indicators.

    Returns
    -------
    pandas.DataFrame
        Enriched meter-pair feature data.
    """
    pair_required = {
        "meter_1",
        "meter_2",
    }

    pair_missing = (
        pair_required - set(pair_df.columns)
    )

    if pair_missing:
        raise ValueError(
            "Pair data is missing columns: "
            f"{sorted(pair_missing)}"
        )

    ami = ami_df.copy()

    if "MeterID" in ami.columns:
        ami = ami.rename(
            columns={
                "MeterID": "meter_number",
            }
        )

    requested_columns = list(
        dict.fromkeys(
            [
                *selected_ami_columns,
                *similarity_columns,
                *exact_match_columns,
            ]
        )
    )

    required_ami_columns = {
        "meter_number",
        *requested_columns,
    }

    missing_ami_columns = (
        required_ami_columns - set(ami.columns)
    )

    if missing_ami_columns:
        raise ValueError(
            "AMI data is missing columns: "
            f"{sorted(missing_ami_columns)}"
        )

    ami = ami[
        ["meter_number"] + requested_columns
    ].copy()

    ami["meter_number"] = (
        ami["meter_number"]
        .astype("string")
        .str.strip()
    )

    if ami["meter_number"].duplicated().any():
        raise ValueError(
            "AMI data contains duplicate meter numbers."
        )

    features = pair_df.copy()

    features["meter_1"] = (
        features["meter_1"]
        .astype("string")
        .str.strip()
    )

    features["meter_2"] = (
        features["meter_2"]
        .astype("string")
        .str.strip()
    )

    meter_1_ami = ami.rename(
        columns={
            "meter_number": "meter_1",
            **{
                column: f"meter_1_{column}"
                for column in requested_columns
            },
        }
    )

    meter_2_ami = ami.rename(
        columns={
            "meter_number": "meter_2",
            **{
                column: f"meter_2_{column}"
                for column in requested_columns
            },
        }
    )

    original_row_count = len(features)

    features = features.merge(
        meter_1_ami,
        on="meter_1",
        how="left",
        validate="many_to_one",
    )

    features = features.merge(
        meter_2_ami,
        on="meter_2",
        how="left",
        validate="many_to_one",
    )

    if len(features) != original_row_count:
        raise ValueError(
            "AMI joins unexpectedly changed the pair row count."
        )

    for column in similarity_columns:
        feature_name = (
            f"{column.lower()}_similarity_pct"
        )

        features[feature_name] = [
            similarity_percentage(first, second)
            for first, second in zip(
                features[f"meter_1_{column}"],
                features[f"meter_2_{column}"],
            )
        ]

    for column in exact_match_columns:
        feature_name = (
            f"same_{column.lower()}"
        )

        features[feature_name] = exact_match(
            features[f"meter_1_{column}"],
            features[f"meter_2_{column}"],
        )

    return features