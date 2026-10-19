from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "candidate_features"
)


# ============================================================
# CONSTANTS
# ============================================================

METER = "meter_number"
CAND = "candidate_transformer"
DIST = "distance_ft"
GROUP = "tx_group_id"
SIZE = "tx_group_size"


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def rename_if_needed(
    df: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """Rename known aliases when the standardized name is not present."""

    out = df.copy()
    rename = {}

    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            rename[old] = new

    if rename:
        out = out.rename(
            columns=rename
        )

    return out


def normalize_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize candidate-pair columns."""

    df = rename_if_needed(
        df,
        {
            "Division": "division",
        },
    )

    required = {
        METER,
        CAND,
        DIST,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    df[METER] = (
        df[METER]
        .astype("string")
        .str.strip()
    )

    df[CAND] = (
        df[CAND]
        .astype("string")
        .str.strip()
    )

    df[DIST] = pd.to_numeric(
        df[DIST],
        errors="coerce",
    )

    return (
        df
        .sort_values(
            [
                METER,
                DIST,
                CAND,
            ]
        )
        .drop_duplicates(
            [
                METER,
                CAND,
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


def normalize_ami(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize AMI columns needed for candidate features."""

    df = rename_if_needed(
        df,
        {
            "MeterID": METER,
            "Division": "division",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "SuperiorFLOC": "current_meter_floc",
            "MeterFLOC": "current_meter_floc",
        },
    )

    if METER not in df.columns:
        raise ValueError(
            f"AMI data must contain '{METER}'."
        )

    df[METER] = (
        df[METER]
        .astype("string")
        .str.strip()
    )

    return df


def normalize_transformers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize transformer columns needed for candidate features."""

    df = rename_if_needed(
        df,
        {
            "FuncLoc": "transformer_floc",
            "Division": "division",
            "DivName": "division",
            "Location_Y": "latitude",
            "Location_X": "longitude",
            "Transformer_kVA": "transformer_kva",
        },
    )

    if "transformer_floc" not in df.columns:
        raise ValueError(
            "Transformer data must contain 'transformer_floc'."
        )

    df["transformer_floc"] = (
        df["transformer_floc"]
        .astype("string")
        .str.strip()
    )

    return df


def normalize_groups(
    df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Standardize meter-group assignment columns."""

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                METER,
                GROUP,
                SIZE,
                "tx_group_source",
                "meter_type",
            ]
        )

    df = rename_if_needed(
        df,
        {
            "Division": "division",
            "cluster_id": GROUP,
            "cluster_size": SIZE,
        },
    )

    if METER not in df.columns:
        raise ValueError(
            f"Grouping data must contain '{METER}'."
        )

    df[METER] = (
        df[METER]
        .astype("string")
        .str.strip()
    )

    keep = [
        column
        for column in [
            METER,
            GROUP,
            SIZE,
            "tx_group_source",
            "meter_type",
        ]
        if column in df.columns
    ]

    return (
        df[keep]
        .drop_duplicates(
            subset=[METER],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FLOC HELPERS
# ============================================================

def floc_suffix(
    value,
) -> str:
    """Return the last two portions of a functional location."""

    if pd.isna(value):
        return ""

    parts = str(value).split("-")

    if len(parts) >= 2:
        return "-".join(
            parts[-2:]
        )

    return str(value)


def floc_similarity(
    meter_floc,
    candidate_floc,
) -> float:
    """Calculate fuzzy similarity between meter and transformer FLOCs."""

    if (
        pd.isna(meter_floc)
        or pd.isna(candidate_floc)
    ):
        return np.nan

    return fuzz.token_sort_ratio(
        floc_suffix(meter_floc),
        floc_suffix(candidate_floc),
    )


# ============================================================
# BASE ENRICHMENT
# ============================================================

def enrich_base_data(
    df: pd.DataFrame,
    groups: pd.DataFrame,
    ami_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add grouping, meter, and transformer information."""

    # --------------------------------------------------------
    # GROUP INFORMATION
    # --------------------------------------------------------

    if not groups.empty:
        df = df.merge(
            groups,
            on=METER,
            how="left",
            validate="many_to_one",
        )

    if SIZE in df.columns:
        # Legacy compatibility with previous model/features.
        df["cluster_size"] = df[SIZE]

    # --------------------------------------------------------
    # METER INFORMATION
    # --------------------------------------------------------

    meter_columns = [
        column
        for column in [
            METER,
            "latitude",
            "longitude",
            "current_meter_floc",
        ]
        if column in ami_df.columns
    ]

    meter_lookup = (
        ami_df[meter_columns]
        .drop_duplicates(
            subset=[METER],
            keep="first",
        )
        .copy()
    )

    meter_lookup = meter_lookup.rename(
        columns={
            "latitude": "meter_latitude",
            "longitude": "meter_longitude",
        }
    )

    df = df.merge(
        meter_lookup,
        on=METER,
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # TRANSFORMER BANK COUNTS
    # --------------------------------------------------------

    if "SAPID" in transformer_df.columns:
        bank = (
            transformer_df
            .groupby(
                "transformer_floc"
            )["SAPID"]
            .nunique()
            .rename(
                "transformer_unit_count"
            )
            .reset_index()
        )

    else:
        bank = (
            transformer_df
            .groupby(
                "transformer_floc"
            )
            .size()
            .rename(
                "transformer_unit_count"
            )
            .reset_index()
        )

    bank["transformer_bank_check"] = (
        bank["transformer_unit_count"] > 1
    ).astype("int8")

    # --------------------------------------------------------
    # UNIQUE TRANSFORMER LOOKUP
    # --------------------------------------------------------

    transformer_columns = [
        column
        for column in [
            "transformer_floc",
            "latitude",
            "longitude",
            "transformer_kva",
        ]
        if column in transformer_df.columns
    ]

    transformer_lookup = (
        transformer_df[transformer_columns]
        .sort_values(
            "transformer_floc"
        )
        .drop_duplicates(
            subset=["transformer_floc"],
            keep="first",
        )
        .rename(
            columns={
                "latitude": "transformer_latitude",
                "longitude": "transformer_longitude",
            }
        )
    )

    transformer_lookup = transformer_lookup.merge(
        bank,
        on="transformer_floc",
        how="left",
    )

    df = df.merge(
        transformer_lookup,
        left_on=CAND,
        right_on="transformer_floc",
        how="left",
        validate="many_to_one",
    )

    df = df.drop(
        columns=["transformer_floc"],
        errors="ignore",
    )

    # Legacy column expected by previous model code.
    if "transformer_kva" in df.columns:
        df["Transformer_kVA"] = (
            df["transformer_kva"]
        )

    # Legacy coordinates useful for downstream code.
    if "meter_latitude" in df.columns:
        df["LATITUDE"] = (
            df["meter_latitude"]
        )

    if "meter_longitude" in df.columns:
        df["LONGITUDE"] = (
            df["meter_longitude"]
        )

    if "transformer_latitude" in df.columns:
        df["Location_Y"] = (
            df["transformer_latitude"]
        )

    if "transformer_longitude" in df.columns:
        df["Location_X"] = (
            df["transformer_longitude"]
        )

    return df


# ============================================================
# DISTANCE FEATURES
# ============================================================

def add_distance_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add meter-level candidate distance features."""

    df = (
        df
        .sort_values(
            [
                METER,
                DIST,
                CAND,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Since the rows are already sorted, cumcount provides
    # a deterministic distance rank.
    df["distance_rank"] = (
        df
        .groupby(METER)
        .cumcount()
        .add(1)
        .astype("int32")
    )

    df["is_closest"] = (
        df["distance_rank"] == 1
    ).astype("int8")

    df["candidate_count_per_meter"] = (
        df
        .groupby(METER)[CAND]
        .transform("size")
        .astype("int32")
    )

    nearest = (
        df
        .groupby(METER)[DIST]
        .transform("first")
    )

    second = (
        df.loc[
            df["distance_rank"] == 2,
            [
                METER,
                DIST,
            ],
        ]
        .drop_duplicates(
            subset=[METER]
        )
        .set_index(METER)[DIST]
    )

    second_distance = (
        df[METER]
        .map(second)
        .fillna(nearest)
    )

    df["distance_gap_to_second"] = (
        second_distance
        - nearest
    )

    df["relative_distance_to_closest_pct"] = (
        nearest
        / df[DIST].replace(0, np.nan)
        * 100
    ).fillna(100)

    df["distance_gap_pct"] = (
        df["distance_gap_to_second"]
        / nearest.replace(0, np.nan)
        * 100
    )

    df["distance_gap_ratio"] = (
        df["distance_gap_to_second"]
        / df[DIST].replace(0, np.nan)
    )

    df["candidate_density"] = (
        df["candidate_count_per_meter"]
        / df["distance_gap_to_second"].replace(
            0,
            np.nan,
        )
    )

    df["is_closest_gap_50ft"] = (
        (df["is_closest"] == 1)
        & (
            df["distance_gap_to_second"]
            > 50
        )
    ).astype("int8")

    df["is_closest_gap_100ft"] = (
        (df["is_closest"] == 1)
        & (
            df["distance_gap_to_second"]
            > 100
        )
    ).astype("int8")

    return df


# ============================================================
# RING FEATURES
# ============================================================

def add_ring_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add distance-ring indicators."""

    df["inner_ring"] = (
        df[DIST] < 150
    ).astype("int8")

    df["middle_ring"] = (
        (df[DIST] >= 150)
        & (df[DIST] < 300)
    ).astype("int8")

    df["outer_ring"] = (
        (df[DIST] >= 300)
        & (df[DIST] < 550)
    ).astype("int8")

    df["out_of_bounds"] = (
        df[DIST] >= 550
    ).astype("int8")

    if SIZE in df.columns:
        df["is_singleton_cluster"] = (
            df[SIZE] == 1
        ).astype("int8")

        df["is_not_single_within_150ft"] = (
            (df[SIZE] > 1)
            & (df["inner_ring"] == 1)
        ).astype("int8")

    else:
        df["is_singleton_cluster"] = 0
        df["is_not_single_within_150ft"] = 0

    return df


# ============================================================
# FLOC FEATURES
# ============================================================

def add_floc_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add FLOC similarity and identity features."""

    if "current_meter_floc" not in df.columns:
        return df

    df["floc_similarity_individual"] = [
        floc_similarity(
            meter_floc,
            candidate_floc,
        )
        for meter_floc, candidate_floc in zip(
            df["current_meter_floc"],
            df[CAND],
        )
    ]

    df["meter_floc_matches_transformer_floc"] = (
        df["current_meter_floc"]
        .astype("string")
        .eq(
            df[CAND].astype("string")
        )
        .fillna(False)
        .astype("int8")
    )

    df["is_single_closest_same_floc"] = (
        (df["is_singleton_cluster"] == 1)
        & (df["is_closest"] == 1)
        & (
            df[
                "meter_floc_matches_transformer_floc"
            ] == 1
        )
    ).astype("int8")

    return df


# ============================================================
# GROUP / CLUSTER FEATURES
# ============================================================

def add_group_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add group-level candidate features."""

    if GROUP not in df.columns:
        return df

    valid = df[GROUP].notna()

    if not valid.any():
        return df

    base = (
        df.loc[valid]
        .drop_duplicates(
            subset=[
                GROUP,
                METER,
                CAND,
            ]
        )
        .copy()
    )

    cluster_n = (
        base
        .groupby(GROUP)[METER]
        .nunique()
        .rename(
            "cluster_meter_count"
        )
        .reset_index()
    )

    aggregate = (
        base
        .groupby(
            [
                GROUP,
                CAND,
            ],
            dropna=False,
        )
        .agg(
            cluster_candidate_meter_count=(
                METER,
                "nunique",
            ),
            cluster_avg_distance_to_candidate=(
                DIST,
                "mean",
            ),
            cluster_median_distance_to_candidate=(
                DIST,
                "median",
            ),
            cluster_distance_std_to_candidate=(
                DIST,
                "std",
            ),
            cluster_best_rank_share=(
                "is_closest",
                "mean",
            ),
            cluster_inner_ring_pct=(
                "inner_ring",
                "mean",
            ),
            cluster_middle_ring_pct=(
                "middle_ring",
                "mean",
            ),
            cluster_outer_ring_pct=(
                "outer_ring",
                "mean",
            ),
            cluster_out_of_bounds_pct=(
                "out_of_bounds",
                "mean",
            ),
            avg_rank_in_cluster=(
                "distance_rank",
                "mean",
            ),
        )
        .reset_index()
        .merge(
            cluster_n,
            on=GROUP,
            how="left",
        )
    )

    aggregate[
        "cluster_transformer_representation_pct"
    ] = (
        aggregate[
            "cluster_candidate_meter_count"
        ]
        / aggregate[
            "cluster_meter_count"
        ].replace(0, np.nan)
        * 100
    )

    df = df.merge(
        aggregate,
        on=[
            GROUP,
            CAND,
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # GROUP ENTROPY
    # --------------------------------------------------------

    counts = (
        base
        .groupby(
            [
                GROUP,
                CAND,
            ]
        )[METER]
        .nunique()
        .reset_index(
            name="n"
        )
    )

    entropy = (
        counts
        .groupby(GROUP)["n"]
        .apply(
            lambda values: -(
                (
                    values / values.sum()
                )
                * np.log(
                    values / values.sum()
                    + 1e-9
                )
            ).sum()
        )
        .rename(
            "cluster_entropy"
        )
        .reset_index()
    )

    df = df.merge(
        entropy,
        on=GROUP,
        how="left",
        validate="many_to_one",
    )

    return df


# ============================================================
# CUSTOMER / KVA FEATURES
# ============================================================

def add_customer_features(
    df: pd.DataFrame,
    ami_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add customer-count and transformer-capacity features."""

    if "current_meter_floc" in ami_df.columns:
        customers = (
            ami_df
            .dropna(
                subset=[
                    "current_meter_floc",
                    METER,
                ]
            )
            .groupby(
                "current_meter_floc"
            )[METER]
            .nunique()
            .rename(
                "customer_count"
            )
            .reset_index()
        )

        df = df.merge(
            customers,
            left_on=CAND,
            right_on="current_meter_floc",
            how="left",
            validate="many_to_one",
            suffixes=(
                "",
                "_candidate_customer",
            ),
        )

        df = df.drop(
            columns=[
                "current_meter_floc_candidate_customer"
            ],
            errors="ignore",
        )

        df["customer_count"] = (
            df["customer_count"]
            .fillna(0)
            .astype("int32")
        )

    else:
        df["customer_count"] = 0

    df["is_zero_customer_transformer"] = (
        df["customer_count"] == 0
    ).astype("int8")

    if SIZE in df.columns:
        df["customer_count_to_group_size_ratio"] = (
            df["customer_count"]
            / df[SIZE].replace(
                0,
                np.nan,
            )
        )

    else:
        df[
            "customer_count_to_group_size_ratio"
        ] = np.nan

    if "Transformer_kVA" in df.columns:
        df["kva_per_customer"] = (
            df["Transformer_kVA"]
            / df["customer_count"].replace(
                0,
                np.nan,
            )
        ).fillna(-1)

        if SIZE in df.columns:
            df["kva_per_cluster_meter"] = (
                df["Transformer_kVA"]
                / df[SIZE].replace(
                    0,
                    np.nan,
                )
            ).fillna(-1)

        else:
            df["kva_per_cluster_meter"] = np.nan

    return df


# ============================================================
# DOMINANCE FEATURES
# ============================================================

def add_dominance_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add meter and group-level distance preference features."""

    df["best_by_distance"] = (
        df[DIST]
        == df
        .groupby(METER)[DIST]
        .transform("min")
    ).astype("int8")

    if GROUP not in df.columns:
        return df

    valid = df[GROUP].notna()

    if not valid.any():
        return df

    preference = (
        df.loc[valid]
        .drop_duplicates(
            subset=[
                GROUP,
                METER,
                CAND,
            ]
        )
        .groupby(
            [
                GROUP,
                CAND,
            ]
        )["best_by_distance"]
        .mean()
        .mul(100)
        .rename(
            "cluster_dominance_distance"
        )
        .reset_index()
    )

    df = df.merge(
        preference,
        on=[
            GROUP,
            CAND,
        ],
        how="left",
        validate="many_to_one",
    )

    if "kva_per_cluster_meter" in df.columns:
        df["dominance_div_kva_per_cluster"] = (
            df["cluster_dominance_distance"]
            / df["kva_per_cluster_meter"].replace(
                0,
                np.nan,
            )
        )

    return df


# ============================================================
# BEARING FEATURES
# ============================================================

def add_bearing_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add directional and group-level bearing features."""

    required = {
        "meter_latitude",
        "meter_longitude",
        "transformer_latitude",
        "transformer_longitude",
    }

    if not required.issubset(df.columns):
        return df

    # Same direction as the prior feature implementation:
    # transformer to meter.
    transformer_latitude = np.radians(
        df["transformer_latitude"]
    )

    meter_latitude = np.radians(
        df["meter_latitude"]
    )

    delta_longitude = np.radians(
        df["meter_longitude"]
        - df["transformer_longitude"]
    )

    x = (
        np.sin(delta_longitude)
        * np.cos(meter_latitude)
    )

    y = (
        np.cos(transformer_latitude)
        * np.sin(meter_latitude)
        - np.sin(transformer_latitude)
        * np.cos(meter_latitude)
        * np.cos(delta_longitude)
    )

    df["bearing"] = (
        np.degrees(
            np.arctan2(
                x,
                y,
            )
        )
        + 360
    ) % 360

    df["bearing_rad"] = np.radians(
        df["bearing"]
    )

    df["bearing_sin"] = np.sin(
        df["bearing_rad"]
    )

    df["bearing_cos"] = np.cos(
        df["bearing_rad"]
    )

    # --------------------------------------------------------
    # GROUP BEARING CONSISTENCY
    # --------------------------------------------------------

    if (
        GROUP in df.columns
        and df[GROUP].notna().any()
    ):
        valid = df[GROUP].notna()

        bearing_group = (
            df.loc[valid]
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
            bearing_group[
                "mean_bearing_sin"
            ] ** 2
            + bearing_group[
                "mean_bearing_cos"
            ] ** 2
        )

        bearing_group[
            "bearing_dispersion"
        ] = (
            1
            - bearing_group[
                "bearing_resultant_strength"
            ]
        )

        df = df.merge(
            bearing_group,
            on=[
                GROUP,
                CAND,
            ],
            how="left",
            validate="many_to_one",
        )

    return df


# ============================================================
# MAIN FEATURE FUNCTION
# ============================================================

def add_candidate_features(
    candidates: pd.DataFrame,
    groups: pd.DataFrame,
    transformer_df: pd.DataFrame,
    ami_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate-level model features."""

    candidates = normalize_candidates(
        candidates
    )

    ami_df = normalize_ami(
        ami_df
    )

    transformer_df = normalize_transformers(
        transformer_df
    )

    groups = normalize_groups(
        groups
    )

    df = enrich_base_data(
        df=candidates,
        groups=groups,
        ami_df=ami_df,
        transformer_df=transformer_df,
    )

    df = add_distance_features(df)
    df = add_ring_features(df)
    df = add_floc_features(df)
    df = add_group_features(df)

    df = add_customer_features(
        df,
        ami_df,
    )

    df = add_dominance_features(df)
    df = add_bearing_features(df)

    # Final duplicate protection.
    df = (
        df
        .sort_values(
            [
                METER,
                DIST,
                CAND,
            ]
        )
        .drop_duplicates(
            [
                METER,
                CAND,
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Write and validate a candidate-feature Parquet file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        ".tmp.parquet"
    )

    try:
        df.to_parquet(
            temp,
            index=False,
            compression="zstd",
        )

        # Validate without loading millions of rows again.
        rows_written = (
            pq.ParquetFile(temp)
            .metadata
            .num_rows
        )

        if rows_written != len(df):
            raise ValueError(
                f"Parquet validation failed for {path}. "
                f"Expected {len(df):,}, "
                f"found {rows_written:,}."
            )

        temp.replace(path)

    finally:
        temp.unlink(
            missing_ok=True
        )


# ============================================================
# DIVISION FILTER
# ============================================================

def division_subset(
    df: pd.DataFrame,
    division: str,
) -> pd.DataFrame:
    """Return rows for one division when a division column is present."""

    if "Division" in df.columns:
        return df.loc[
            df["Division"] == division
        ].copy()

    if "division" in df.columns:
        return df.loc[
            df["division"] == division
        ].copy()

    return df.copy()


# ============================================================
# PIPELINE STAGE
# ============================================================

def run_candidate_feature_stage(
    config,
    candidate_files: dict[str, Path],
    group_files: dict[str, Path],
    ami_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    snapshot_date: date | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Build and cache candidate features for every division."""

    snapshot_date = (
        snapshot_date
        if snapshot_date is not None
        else date.today()
    )

    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else FEATURE_CACHE_DIR
    )

    day_dir = (
        cache_root
        / snapshot_date.isoformat()
    )

    complete_file = (
        day_dir
        / "_complete.json"
    )

    overwrite = bool(
        getattr(
            config,
            "overwrite_m2t_outputs",
            False,
        )
    )

    # --------------------------------------------------------
    # REUSE TODAY'S COMPLETED FEATURE CACHE
    # --------------------------------------------------------

    if (
        complete_file.exists()
        and not overwrite
    ):
        print(
            "Candidate-feature cache already exists "
            f"for {snapshot_date}."
        )

        print(
            "Reusing completed feature files."
        )

        return {
            file.stem: file
            for file in day_dir.glob(
                "*.parquet"
            )
        }

    day_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs: dict[str, Path] = {}

    manifest = {
        "date": snapshot_date.isoformat(),
        "divisions": {},
    }

    # --------------------------------------------------------
    # PROCESS EVERY DIVISION
    # --------------------------------------------------------

    for division in sorted(candidate_files):
        print(
            f"\n[{division}] "
            "building candidate features..."
        )

        candidate_path = Path(
            candidate_files[division]
        )

        candidates = pd.read_parquet(
            candidate_path
        )

        original_rows = len(
            candidates
        )

        if division in group_files:
            groups = pd.read_parquet(
                group_files[division]
            )

        else:
            print(
                f"[{division}] no grouping file found. "
                "Group features will be null."
            )

            groups = pd.DataFrame()

        ami_division = division_subset(
            ami_df,
            division,
        )

        transformer_division = division_subset(
            transformer_df,
            division,
        )

        features = add_candidate_features(
            candidates=candidates,
            groups=groups,
            transformer_df=transformer_division,
            ami_df=ami_division,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(features) != original_rows:
            raise ValueError(
                f"[{division}] candidate feature row "
                f"count changed from {original_rows:,} "
                f"to {len(features):,}."
            )

        duplicate_pairs = int(
            features
            .duplicated(
                [
                    METER,
                    CAND,
                ]
            )
            .sum()
        )

        if duplicate_pairs:
            raise ValueError(
                f"[{division}] found "
                f"{duplicate_pairs:,} duplicate "
                "meter-transformer pairs."
            )

        output_path = (
            day_dir
            / f"{division}.parquet"
        )

        save_parquet(
            features,
            output_path,
        )

        outputs[division] = (
            output_path
        )

        manifest["divisions"][division] = {
            "candidate_rows": original_rows,
            "feature_rows": len(features),
            "meters": int(
                features[METER].nunique()
            ),
            "transformers": int(
                features[CAND].nunique()
            ),
            "feature_columns": len(
                features.columns
            ),
            "file": str(output_path),
        }

        print(
            f"[{division}] "
            f"{len(features):,} candidate rows | "
            f"{features[METER].nunique():,} meters | "
            f"{len(features.columns):,} columns"
        )

    # --------------------------------------------------------
    # COMPLETE MARKER
    # --------------------------------------------------------

    temp_complete = complete_file.with_suffix(
        ".tmp.json"
    )

    try:
        with temp_complete.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )

        temp_complete.replace(
            complete_file
        )

    finally:
        temp_complete.unlink(
            missing_ok=True
        )

    print(
        "\nCandidate feature generation complete."
    )

    print(
        f"Feature cache: {day_dir}"
    )

    return outputs