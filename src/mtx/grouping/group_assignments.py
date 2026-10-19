from __future__ import annotations
import numpy as np
import pandas as pd
GROUP_COLUMNS = [
    "meter_number",
    "division",
    "tx_group_id",
    "tx_group_size",
    "tx_group_source",
    "meter_type",
]
def normalizemeter_population(
    meter_population: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize AMI meter population to:
        meter_number
        division
    """
    df = meter_population.copy()
    rename_map = {}
    if "MeterID" in df.columns:
        rename_map["MeterID"] = "meter_number"
    if "Division" in df.columns:
        rename_map["Division"] = "division"
    df = df.rename(columns=rename_map)
    required = {
        "meter_number",
        "division",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Meter population missing columns: "
            f"{sorted(missing)}"
        )
    df = (
        df[
            [
                "meter_number",
                "division",
            ]
        ]
        .dropna(
            subset=[
                "meter_number",
                "division",
            ]
        )
        .drop_duplicates(
            subset=["meter_number"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    return df
def build_itron_group_assignments(
    meter_population: pd.DataFrame,
    itron_groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach ITRON Tx_subgroup_ID to the meter population.
    Meters without an ITRON subgroup are treated as standalone.
    """
    population = normalizemeter_population(
        meter_population
    )
    groups = itron_groups.copy()
    groups = groups.rename(
        columns={
            "MeterNumber": "meter_number",
            "Division": "itron_division",
        }
    )
    required = {
        "meter_number",
        "Tx_Subgroup_ID",
    }
    missing = required - set(groups.columns)
    if missing:
        raise ValueError(
            "ITRON grouping data missing columns: "
            f"{sorted(missing)}"
        )
    # --------------------------------------------------
    # Detect conflicting ITRON subgroup membership
    # --------------------------------------------------
    conflicts = (
        groups
        .dropna(
            subset=["Tx_Subgroup_ID"]
        )
        .groupby("meter_number")[
            "Tx_Subgroup_ID"
        ]
        .nunique()
    )
    conflicts = conflicts[
        conflicts > 1
    ]
    if not conflicts.empty:
        examples = (
            conflicts
            .head(20)
            .index
            .tolist()
        )
        raise ValueError(
            "Some meters belong to multiple ITRON "
            "Tx_subgroup_ID values. "
            f"Conflict count={len(conflicts):,}. "
            f"Examples={examples}"
        )
    groups = (
        groups
        .sort_values(
            [
                "meter_number",
                "Tx_Subgroup_ID",
            ]
        )
        .drop_duplicates(
            subset=["meter_number"],
            keep="first",
        )
    )
    keep = [
        "meter_number",
        "Tx_Subgroup_ID",
    ]
    if "itron_division" in groups.columns:
        keep.append(
            "itron_division"
        )
    groups = groups[keep]
    # --------------------------------------------------
    # Join to actual SmartLink meter population
    # --------------------------------------------------
    df = population.merge(
        groups,
        on="meter_number",
        how="left",
    )
    # --------------------------------------------------
    # Build globally unique group ID
    # --------------------------------------------------
    has_group = (
        df["Tx_Subgroup_ID"]
        .notna()
    )
    df["tx_group_id"] = pd.NA
    df.loc[
        has_group,
        "tx_group_id",
    ] = (
        df.loc[
            has_group,
            "division",
        ]
        .astype(str)
        .str.strip()
        +
        "::ITRON::"
        +
        df.loc[
            has_group,
            "Tx_Subgroup_ID",
        ]
        .astype(str)
        .str.strip()
    )
    # --------------------------------------------------
    # Calculate group size from actual population
    # --------------------------------------------------
    group_sizes = (
        df
        .dropna(
            subset=["tx_group_id"]
        )
        .groupby(
            "tx_group_id"
        )[
            "meter_number"
        ]
        .nunique()
        .rename(
            "tx_group_size"
        )
    )
    df = df.merge(
        group_sizes,
        on="tx_group_id",
        how="left",
    )
    df[
        "tx_group_size"
    ] = (
        df[
            "tx_group_size"
        ]
        .fillna(1)
        .astype(int)
    )
    # --------------------------------------------------
    # Clustered versus standalone
    # --------------------------------------------------
    df[
        "meter_type"
    ] = np.where(
        df[
            "tx_group_size"
        ] > 1,
        "clustered",
        "standalone",
    )
    df[
        "tx_group_source"
    ] = "itron"
    return (
        df[
            GROUP_COLUMNS
        ]
        .sort_values(
            [
                "division",
                "meter_number",
            ]
        )
        .reset_index(drop=True)
    )
def build_smartlink_group_assignments(
    clustered: pd.DataFrame,
    standalone: pd.DataFrame,
    division: str,
) -> pd.DataFrame:
    """
    Convert SmartLink meter clusters to the same schema
    used by ITRON groups.
    """
    clustered = clustered.copy()
    standalone = standalone.copy()
    required_cluster = {
        "meter_number",
        "cluster_id",
    }
    missing = (
        required_cluster
        - set(clustered.columns)
    )
    if missing:
        raise ValueError(
            "Clustered meter file missing columns: "
            f"{sorted(missing)}"
        )
    # --------------------------------------------------
    # Clustered meters
    # --------------------------------------------------
    clustered[
        "division"
    ] = division
    clustered[
        "tx_group_id"
    ] = (
        division
        + "::SMARTLINK::"
        + clustered[
            "cluster_id"
        ]
        .astype(str)
    )
    group_sizes = (
        clustered
        .groupby(
            "tx_group_id"
        )[
            "meter_number"
        ]
        .nunique()
        .rename(
            "tx_group_size"
        )
    )
    clustered = clustered.merge(
        group_sizes,
        on="tx_group_id",
        how="left",
    )
    clustered[
        "tx_group_source"
    ] = "smartlink"
    clustered[
        "meter_type"
    ] = "clustered"
    # --------------------------------------------------
    # Standalone meters
    # --------------------------------------------------
    if "meter_number" not in standalone.columns:
        raise ValueError(
            "Standalone meter file is missing "
            "'meter_number'."
        )
    standalone = (
        standalone[
            ["meter_number"]
        ]
        .copy()
    )
    standalone[
        "division"
    ] = division
    standalone[
        "tx_group_id"
    ] = pd.NA
    standalone[
        "tx_group_size"
    ] = 1
    standalone[
        "tx_group_source"
    ] = "smartlink"
    standalone[
        "meter_type"
    ] = "standalone"
    out = pd.concat(
        [
            clustered[
                GROUP_COLUMNS
            ],
            standalone[
                GROUP_COLUMNS
            ],
        ],
        ignore_index=True,
    )
    duplicates = (
        out
        .duplicated(
            subset=[
                "division",
                "meter_number",
            ]
        )
    )
    if duplicates.any():
        raise ValueError(
            "A meter occurs more than once "
            "in SmartLink group output."
        )
    return (
        out
        .sort_values(
            [
                "division",
                "meter_number",
            ]
        )
        .reset_index(drop=True)
    )