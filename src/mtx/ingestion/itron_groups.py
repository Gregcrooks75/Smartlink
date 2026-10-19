"""Extract and normalize ITRON transformer subgroup assignments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


SOURCE_REQUIRED_COLUMNS = {
    "meter_number",
    "Tx_Subgroup_ID",
}

OUTPUT_REQUIRED_COLUMNS = {
    "meter_number",
    "Tx_Subgroup_ID",
    "tx_group_id",
    "tx_group_size",
    "group_source",
    "group_type",
}


def _default_source_loader() -> pd.DataFrame:
    """Load ITRON meter assignments from the established NYAMI loader."""
    from mtx.db.db_AMI import get_itron_meter_data

    return get_itron_meter_data(meter_ids=None)


def _to_pandas(data: Any) -> pd.DataFrame:
    """Return a defensive pandas copy of a pandas or Polars frame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        import polars as pl
    except ImportError:
        pl = None

    if pl is not None and isinstance(data, pl.DataFrame):
        return data.to_pandas()

    raise TypeError(
        "The ITRON source loader must return a pandas or Polars "
        f"DataFrame, not {type(data).__name__}."
    )


def _clean_identifier(series: pd.Series) -> pd.Series:
    """Normalize identifier text while retaining missing values."""
    cleaned = series.astype("string").str.strip()

    return cleaned.mask(
        cleaned.str.lower().isin(
            {"", "nan", "none", "null", "<na>"}
        )
    )


def _validate_source(source: pd.DataFrame) -> None:
    """Validate the raw ITRON rows before grouping."""
    if source.empty:
        raise ValueError(
            "The ITRON source query returned zero rows."
        )

    missing = SOURCE_REQUIRED_COLUMNS - set(source.columns)

    if missing:
        raise ValueError(
            "The ITRON source is missing required columns: "
            f"{sorted(missing)}"
        )

    missing_meter = source["meter_number"].isna()

    if missing_meter.any():
        raise ValueError(
            "The ITRON source contains "
            f"{int(missing_meter.sum()):,} rows without meter_number."
        )

    assignments = source.dropna(subset=["Tx_Subgroup_ID"])

    conflicting = (
        assignments.groupby(
            "meter_number",
            dropna=False
        )["Tx_Subgroup_ID"]
        .nunique(dropna=True)
        .gt(1)
    )

    if conflicting.any():
        examples = (
            conflicting[conflicting]
            .index.astype(str)
            .tolist()[:10]
        )

        raise ValueError(
            "Meters have conflicting ITRON subgroup assignments. "
            f"Examples: {examples}"
        )


def _scoped_group_ids(source: pd.DataFrame) -> pd.Series:
    """Create collision-safe group IDs scoped to the meter division."""

    if "division" in source.columns:
        division = (
            _clean_identifier(source["division"])
            .fillna("UNKNOWN")
        )
    else:
        division = pd.Series(
            "UNKNOWN",
            index=source.index,
            dtype="string"
        )

    subgroup = source["Tx_Subgroup_ID"]
    has_subgroup = subgroup.notna()

    group_id = pd.Series(
        pd.NA,
        index=source.index,
        dtype="string"
    )

    group_id.loc[has_subgroup] = (
        "itron::"
        + division.loc[has_subgroup]
        + "::"
        + subgroup.loc[has_subgroup]
    )

    group_id.loc[~has_subgroup] = (
        "standalone::"
        + source.loc[~has_subgroup, "meter_number"]
    )

    return group_id


def build_itron_tx_subgroups(
    source_loader: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """
    Build one normalized transformer-group assignment per ITRON meter.

    Meters without ``Tx_Subgroup_ID`` are retained as one-meter standalone
    groups. The raw ITRON subgroup remains available for traceability.
    """

    loader = source_loader or _default_source_loader

    print(
        "[ITRON] Loading raw ITRON data...",
        flush=True,
    )

    source = _to_pandas(loader())

    aliases = {
        "Meter_Number": "meter_number",
        "ServiceCenterName": "division",
        "SuperiorFLOC": "current_meter_floc",
    }

    source = source.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in source.columns
            and new not in source.columns
        }
    )

    if "meter_number" in source.columns:
        source["meter_number"] = _clean_identifier(
            source["meter_number"]
        )

    if "Tx_Subgroup_ID" in source.columns:
        source["Tx_Subgroup_ID"] = _clean_identifier(
            source["Tx_Subgroup_ID"]
        )

    _validate_source(source)

    # Prefer a real subgroup assignment when duplicate source rows contain
    # both a null and a populated subgroup for the same meter.
    source = (
        source.assign(
            _has_subgroup=source["Tx_Subgroup_ID"].notna()
        )
        .sort_values(
            ["meter_number", "_has_subgroup"],
            ascending=[True, False]
        )
        .drop_duplicates(
            "meter_number",
            keep="first"
        )
        .drop(columns="_has_subgroup")
        .reset_index(drop=True)
    )

    source["tx_group_id"] = _scoped_group_ids(source)

    source["tx_group_size"] = (
        source.groupby("tx_group_id")["meter_number"]
        .transform("nunique")
        .astype("int32")
    )

    source["group_source"] = "itron"
    source["group_type"] = "standalone"

    clustered = (
        source["Tx_Subgroup_ID"].notna()
        & source["tx_group_size"].gt(1)
    )

    source.loc[
        clustered,
        "group_type"
    ] = "clustered"

    optional_columns = [
        column
        for column in (
            "division",
            "opco",
            "current_meter_floc",
        )
        if column in source.columns
    ]

    result = source[
        [
            "meter_number",
            "Tx_Subgroup_ID",
            "tx_group_id",
            "tx_group_size",
            "group_source",
            "group_type",
            *optional_columns,
        ]
    ].sort_values(
        "meter_number",
        ignore_index=True,
    )

    missing_output = (
        OUTPUT_REQUIRED_COLUMNS - set(result.columns)
    )

    if missing_output:
        raise RuntimeError(
            "ITRON subgroup construction failed to create columns: "
            f"{sorted(missing_output)}"
        )

    print(
        "[ITRON] Built "
        f"{len(result):,} meter assignments across "
        f"{result['tx_group_id'].nunique():,} groups.",
        flush=True,
    )

    return result