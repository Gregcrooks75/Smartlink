"""Create candidate-transformer features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


METER = "meter_number"
CAND = "candidate_transformer"
DIST = "distance_ft"
GROUP = "tx_group_id"
SIZE = "tx_group_size"


def floc_suffix(value) -> str:
    """Return the final two FLOC components."""
    if pd.isna(value):
        return ""

    parts = str(value).split("-")

    if len(parts) >= 2:
        return "-".join(parts[-2:])

    return str(value)


def enrich_candidate_rows(
    df: pd.DataFrame,
    ami_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add meter and transformer metadata required
    for feature engineering.
    """
    out = df.copy()
    ami = ami_df.copy()
    transformers = transformer_df.copy()

    # ==================================================
    # AMI
    # ==================================================

    rename_map = {}

    if "MeterID" in ami.columns:
        rename_map["MeterID"] = "meter_number"

    if "SuperiorFLOC" in ami.columns:
        rename_map["SuperiorFLOC"] = "meter_floc"

    ami = ami.rename(columns=rename_map)

    required_ami = {
        "meter_number",
        "LATITUDE",
        "LONGITUDE",
    }

    missing = required_ami - set(ami.columns)

    if missing:
        raise ValueError(
            "AMI missing columns needed "
            f"for enrichment: {sorted(missing)}"
        )

    ami_cols = [
        "meter_number",
        "LATITUDE",
        "LONGITUDE",
    ]

    if "meter_floc" in ami.columns:
        ami_cols.append("meter_floc")

    meter_lookup = (
        ami[ami_cols]
        .drop_duplicates(
            subset="meter_number",
            keep="first",
        )
    )

    out = out.merge(
        meter_lookup,
        on="meter_number",
        how="left",
        validate="many_to_one",
    )

    # ==================================================
    # Transformer
    # ==================================================

    required_transformer = {
        "FuncLoc",
        "Location_X",
        "Location_Y",
    }

    missing = required_transformer - set(transformers.columns)

    if missing:
        raise ValueError(
            "Transformer data missing "
            f"columns: {sorted(missing)}"
        )

    transformer_cols = [
        "FuncLoc",
        "Location_X",
        "Location_Y",
    ]

    if "Transformer_kVA" in transformers.columns:
        transformer_cols.append("Transformer_kVA")

    transformer_lookup = (
        transformers[transformer_cols]
        .sort_values("FuncLoc")
        .drop_duplicates(
            subset="FuncLoc",
            keep="first",
        )
    )

    out = (
        out
        .merge(
            transformer_lookup,
            left_on=CAND,
            right_on="FuncLoc",
            how="left",
            validate="many_to_one",
        )
        .drop(
            columns=["FuncLoc"],
            errors="ignore",
        )
    )

    return out


def add_distance_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add meter-level distance features."""
    out = df.copy()

    out = (
        out
        .sort_values(
            [
                METER,
                DIST,
                CAND,
            ]
        )
        .drop_duplicates(
            subset=[
                METER,
                CAND,
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    out["distance_rank"] = (
        out
        .groupby(METER)[DIST]
        .rank(method="first")
    )

    out["is_closest"] = (
        out["distance_rank"] == 1
    ).astype(int)

    out["candidate_count_per_meter"] = (
        out
        .groupby(METER)[CAND]
        .transform("nunique")
    )

    closest = (
        out
        .groupby(METER)[DIST]
        .transform("min")
    )

    first_two = out[
        out["distance_rank"] <= 2
    ]

    gap = (
        first_two
        .groupby(METER)[DIST]
        .agg(
            lambda values: (
                values.max()
                - values.min()
            )
        )
        .rename("distance_gap_to_second")
    )

    out = out.merge(
        gap,
        on=METER,
        how="left",
    )

    # A meter with only one candidate has no meaningful
    # second-candidate distance gap.
    one_candidate = (
        out["candidate_count_per_meter"] == 1
    )

    out.loc[
        one_candidate,
        "distance_gap_to_second",
    ] = np.nan

    out["relative_distance_to_closest_pct"] = (
        closest
        / out[DIST].replace(0, np.nan)
        * 100
    )

    out["relative_distance_to_closest_pct"] = (
        out["relative_distance_to_closest_pct"]
        .fillna(100.0)
    )

    out["distance_gap_pct"] = (
        out["distance_gap_to_second"]
        / closest.replace(0, np.nan)
        * 100
    )

    out["distance_gap_ratio"] = (
        out["distance_gap_to_second"]
        / out[DIST].replace(0, np.nan)
    )

    out["candidate_density"] = (
        out["candidate_count_per_meter"]
        / out["distance_gap_to_second"].replace(
            0,
            np.nan,
        )
    )

    out["is_closest_gap_50ft"] = (
        (out["is_closest"] == 1)
        & (out["distance_gap_to_second"] > 50)
    ).astype(int)

    out["is_closest_gap_100ft"] = (
        (out["is_closest"] == 1)
        & (out["distance_gap_to_second"] > 100)
    ).astype(int)

    return out


def add_ring_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add distance-ring indicators."""
    out = df.copy()

    out["inner_ring"] = (
        out[DIST] < 150
    ).astype(int)

    out["middle_ring"] = (
        (out[DIST] >= 150)
        & (out[DIST] < 300)
    ).astype(int)

    out["outer_ring"] = (
        (out[DIST] >= 300)
        & (out[DIST] < 550)
    ).astype(int)

    out["out_of_bounds"] = (
        out[DIST] >= 550
    ).astype(int)

    out["is_singleton_group"] = (
        out[SIZE] == 1
    ).astype(int)

    out["is_not_single_within_150ft"] = (
        (out[SIZE] > 1)
        & (out["inner_ring"] == 1)
    ).astype(int)

    return out


def add_floc_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add existing-meter-FLOC comparison features."""
    out = df.copy()

    if "meter_floc" not in out.columns:
        return out

    out["floc_similarity_individual"] = [
        fuzz.token_sort_ratio(
            floc_suffix(meter_floc),
            floc_suffix(candidate_transformer),
        )
        for meter_floc, candidate_transformer in zip(
            out["meter_floc"],
            out[CAND],
        )
    ]

    out["meter_floc_matches_transformer_floc"] = (
        out["meter_floc"].astype(str)
        == out[CAND].astype(str)
    ).astype(int)

    out["is_single_closest_same_floc"] = (
        (out["is_singleton_group"] == 1)
        & (out["is_closest"] == 1)
        & (
            out["meter_floc_matches_transformer_floc"]
            == 1
        )
    ).astype(int)

    return out


def add_group_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add group-level candidate features.

    The group may originate from either an ITRON
    Tx subgroup or a SmartLink cluster, depending
    on the value of tx_group_source.
    """
    out = df.copy()

    grouped = out[GROUP].notna()

    if not grouped.any():
        return out

    base = (
        out.loc[grouped]
        .drop_duplicates(
            subset=[
                GROUP,
                METER,
                CAND,
            ]
        )
        .copy()
    )

    # ==================================================
    # Group size
    # ==================================================

    actual_group_size = (
        base
        .groupby(GROUP)[METER]
        .nunique()
        .rename("group_meter_count")
    )

    # ==================================================
    # Candidate representation
    # ==================================================

    aggregate = (
        base
        .groupby(
            [
                GROUP,
                CAND,
            ]
        )
        .agg(
            group_candidate_meter_count=(
                METER,
                "nunique",
            ),
            group_avg_distance_to_candidate=(
                DIST,
                "mean",
            ),
            group_median_distance_to_candidate=(
                DIST,
                "median",
            ),
            group_distance_std_to_candidate=(
                DIST,
                "std",
            ),
            group_best_rank_share=(
                "is_closest",
                "mean",
            ),
            group_inner_ring_pct=(
                "inner_ring",
                "mean",
            ),
            group_middle_ring_pct=(
                "middle_ring",
                "mean",
            ),
            group_outer_ring_pct=(
                "outer_ring",
                "mean",
            ),
            group_out_of_bounds_pct=(
                "out_of_bounds",
                "mean",
            ),
            avg_rank_in_group=(
                "distance_rank",
                "mean",
            ),
        )
        .reset_index()
        .merge(
            actual_group_size,
            on=GROUP,
            how="left",
        )
    )

    aggregate[
        "group_transformer_representation_pct"
    ] = (
        aggregate["group_candidate_meter_count"]
        / aggregate["group_meter_count"].replace(
            0,
            np.nan,
        )
        * 100
    )

    out = out.merge(
        aggregate,
        on=[
            GROUP,
            CAND,
        ],
        how="left",
    )

    # ==================================================
    # True nearest-transformer share
    # ==================================================

    nearest = (
        base.loc[
            base["is_closest"] == 1
        ]
        .drop_duplicates(
            subset=[
                GROUP,
                METER,
            ]
        )
    )

    nearest_counts = (
        nearest
        .groupby(
            [
                GROUP,
                CAND,
            ]
        )[METER]
        .nunique()
        .rename(
            "group_nearest_transformer_count"
        )
        .reset_index()
    )

    out = out.merge(
        nearest_counts,
        on=[
            GROUP,
            CAND,
        ],
        how="left",
    )

    out["group_nearest_transformer_count"] = (
        out["group_nearest_transformer_count"]
        .fillna(0)
    )

    out["group_nearest_transformer_share_pct"] = (
        out["group_nearest_transformer_count"]
        / out[SIZE].replace(0, np.nan)
        * 100
    )

    # ==================================================
    # Nearest-transformer entropy
    # ==================================================

    nearest_distribution = (
        nearest
        .groupby(
            [
                GROUP,
                CAND,
            ]
        )[METER]
        .nunique()
        .rename("n")
        .reset_index()
    )

    def calculate_entropy(
        values: pd.Series,
    ) -> float:
        """Calculate entropy from candidate counts."""
        probability = (
            values
            / values.sum()
        )

        return float(
            -(
                probability
                * np.log(
                    probability
                    + 1e-12
                )
            ).sum()
        )

    group_entropy = (
        nearest_distribution
        .groupby(GROUP)["n"]
        .apply(calculate_entropy)
        .rename(
            "group_nearest_transformer_entropy"
        )
        .reset_index()
    )

    out = out.merge(
        group_entropy,
        on=GROUP,
        how="left",
    )

    return out


def add_transformer_features(
    df: pd.DataFrame,
    transformer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add kVA and multi-unit transformer-bank features."""
    out = df.copy()
    transformers = transformer_df.copy()

    # ==================================================
    # Transformer kVA
    # ==================================================

    if (
        "Transformer_kVA" in transformers.columns
        and "Transformer_kVA" not in out.columns
    ):
        kva = (
            transformers[
                [
                    "FuncLoc",
                    "Transformer_kVA",
                ]
            ]
            .drop_duplicates(
                subset="FuncLoc",
                keep="first",
            )
        )

        out = (
            out
            .merge(
                kva,
                left_on=CAND,
                right_on="FuncLoc",
                how="left",
            )
            .drop(
                columns=["FuncLoc"],
                errors="ignore",
            )
        )

    # ==================================================
    # Transformer bank
    # ==================================================
    #
    # Do not deduplicate FuncLoc before calculating this.
    # Otherwise every transformer unit count becomes 1.
    # ==================================================

    required_bank_columns = {
        "FuncLoc",
        "SAPID",
    }

    if required_bank_columns.issubset(
        transformers.columns
    ):
        bank = (
            transformers
            .groupby("FuncLoc")["SAPID"]
            .nunique()
            .rename("transformer_unit_count")
            .reset_index()
        )

        bank["transformer_bank_check"] = (
            bank["transformer_unit_count"] > 1
        ).astype(int)

        out = (
            out
            .merge(
                bank,
                left_on=CAND,
                right_on="FuncLoc",
                how="left",
            )
            .drop(
                columns=["FuncLoc"],
                errors="ignore",
            )
        )

    return out


def add_customer_features(
    df: pd.DataFrame,
    ami_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add current-customer-count and kVA-ratio features."""
    out = df.copy()
    ami = ami_df.copy()

    required_customer_columns = {
        "SuperiorFLOC",
        "MeterID",
    }

    if required_customer_columns.issubset(
        ami.columns
    ):
        customers = (
            ami
            .groupby("SuperiorFLOC")["MeterID"]
            .nunique()
            .rename("customer_count")
            .reset_index()
        )

        out = (
            out
            .merge(
                customers,
                left_on=CAND,
                right_on="SuperiorFLOC",
                how="left",
            )
            .drop(
                columns=["SuperiorFLOC"],
                errors="ignore",
            )
        )

        out["customer_count"] = (
            out["customer_count"]
            .fillna(0)
        )

        out["is_zero_customer_transformer"] = (
            out["customer_count"] == 0
        ).astype(int)

        out["customer_count_to_group_size_ratio"] = (
            out["customer_count"]
            / out[SIZE].replace(0, np.nan)
        )

    if "Transformer_kVA" in out.columns:
        if "customer_count" in out.columns:
            out["kva_per_customer"] = (
                out["Transformer_kVA"]
                / out["customer_count"].replace(
                    0,
                    np.nan,
                )
            ).fillna(-1)

        out["kva_per_group_meter"] = (
            out["Transformer_kVA"]
            / out[SIZE].replace(
                0,
                np.nan,
            )
        ).fillna(-1)

    return out


def add_bearing_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add directional meter-to-transformer features."""
    out = df.copy()

    required = {
        "Location_Y",
        "Location_X",
        "LATITUDE",
        "LONGITUDE",
    }

    if not required.issubset(out.columns):
        return out

    lat1 = np.radians(
        pd.to_numeric(
            out["Location_Y"],
            errors="coerce",
        )
    )

    lat2 = np.radians(
        pd.to_numeric(
            out["LATITUDE"],
            errors="coerce",
        )
    )

    lon1 = np.radians(
        pd.to_numeric(
            out["Location_X"],
            errors="coerce",
        )
    )

    lon2 = np.radians(
        pd.to_numeric(
            out["LONGITUDE"],
            errors="coerce",
        )
    )

    dlon = lon2 - lon1

    x = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    y = (
        np.cos(lat1)
        * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    out["bearing"] = (
        np.degrees(
            np.arctan2(
                x,
                y,
            )
        )
        + 360
    ) % 360

    out["bearing_rad"] = np.radians(
        out["bearing"]
    )

    out["bearing_sin"] = np.sin(
        out["bearing_rad"]
    )

    out["bearing_cos"] = np.cos(
        out["bearing_rad"]
    )

    grouped = out[GROUP].notna()

    if grouped.any():
        bearing_group = (
            out.loc[grouped]
            .groupby(
                [
                    GROUP,
                    CAND,
                ]
            )
            .agg(
                mean_bearing_sin=(
                    "bearing_sin",
                    "mean",
                ),
                mean_bearing_cos=(
                    "bearing_cos",
                    "mean",
                ),
            )
            .reset_index()
        )

        bearing_group[
            "bearing_resultant_strength"
        ] = np.sqrt(
            bearing_group["mean_bearing_sin"] ** 2
            + bearing_group["mean_bearing_cos"] ** 2
        )

        bearing_group["bearing_dispersion"] = (
            1
            - bearing_group[
                "bearing_resultant_strength"
            ]
        )

        out = out.merge(
            bearing_group,
            on=[
                GROUP,
                CAND,
            ],
            how="left",
        )

    return out


def add_candidate_features(
    df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    ami_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete candidate-transformer feature pipeline."""
    out = enrich_candidate_rows(
        df=df,
        ami_df=ami_df,
        transformer_df=transformer_df,
    )

    out = add_distance_features(out)

    out = add_ring_features(out)

    out = add_floc_features(out)

    out = add_group_features(out)

    out = add_transformer_features(
        out,
        transformer_df=transformer_df,
    )

    out = add_customer_features(
        out,
        ami_df=ami_df,
    )

    out = add_bearing_features(out)

    out = (
        out
        .sort_values(
            [
                METER,
                DIST,
                CAND,
            ]
        )
        .drop_duplicates(
            subset=[
                METER,
                CAND,
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return out