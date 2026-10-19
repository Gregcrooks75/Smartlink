from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    r"C:\Users\u355394\OneDrive - IBERDROLA S.A"
    r"\Documents - Asset Data Strategy"
    r"\Asset Data Quality\SmartLink"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
)


# ============================================================
# HELPERS
# ============================================================

def search_parquet_files(
    root: Path,
    column: str,
    value: str,
) -> pd.DataFrame:
    """
    Search every parquet file below root for a specific value.
    """

    if not root.exists():
        return pd.DataFrame()

    results = []

    for file in root.rglob("*.parquet"):

        try:
            df = pd.read_parquet(file)

            if column not in df.columns:
                continue

            matches = df[
                df[column].astype(str)
                == str(value)
            ].copy()

            if matches.empty:
                continue

            matches["history_file"] = str(file)

            results.append(matches)

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# METER
# ============================================================

def get_meter_history(
    meter_number: str,
) -> dict[str, pd.DataFrame]:

    meter_number = str(meter_number)

    history = {}

    # AMI

    history["ami"] = search_parquet_files(
        CACHE_DIR / "ami" / "history",
        "meter_number",
        meter_number,
    )

    # ITRON source history

    history["itron"] = search_parquet_files(
        CACHE_DIR
        / "grouping"
        / "itron"
        / "history",
        "meter_number",
        meter_number,
    )

    # Resulting meter-group history

    history["grouping"] = search_parquet_files(
        CACHE_DIR
        / "grouping"
        / "history",
        "meter_number",
        meter_number,
    )

    # Candidate history

    history["candidates"] = search_parquet_files(
        CACHE_DIR
        / "candidates"
        / "history"
        / "meters",
        "meter_number",
        meter_number,
    )

    # Adam / training history

    history["adam"] = search_parquet_files(
        CACHE_DIR
        / "adam"
        / "history",
        "meter_number",
        meter_number,
    )

    return history


# ============================================================
# TRANSFORMER
# ============================================================

def get_transformer_history(
    transformer_floc: str,
) -> dict[str, pd.DataFrame]:

    transformer_floc = str(transformer_floc)

    history = {}

    # Transformer source history

    history["transformer"] = search_parquet_files(
        CACHE_DIR
        / "transformers"
        / "history",
        "transformer_floc",
        transformer_floc,
    )

    # Candidate-history transformer perspective

    history["candidates"] = search_parquet_files(
        CACHE_DIR
        / "candidates"
        / "history"
        / "transformers",
        "transformer_floc",
        transformer_floc,
    )

    # Candidate history may use candidate_transformer

    candidate_pairs = search_parquet_files(
        CACHE_DIR
        / "candidates"
        / "history"
        / "meters",
        "candidate_transformer",
        transformer_floc,
    )

    history[
        "candidate_relationships"
    ] = candidate_pairs

    return history


# ============================================================
# GENERIC LOOKUP
# ============================================================

def get_asset_history(
    asset_id: str,
    asset_type: str,
) -> dict[str, pd.DataFrame]:

    asset_type = (
        asset_type
        .lower()
        .strip()
    )

    if asset_type == "meter":
        return get_meter_history(asset_id)

    if asset_type == "transformer":
        return get_transformer_history(asset_id)

    raise ValueError(
        "asset_type must be "
        "'meter' or 'transformer'."
    )