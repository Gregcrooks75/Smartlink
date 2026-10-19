# association_simple.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def _norm_str(s: pd.Series) -> pd.Series:
    """Normalize strings for reliable comparisons."""
    return s.astype(str).str.strip().replace({"": np.nan})


def add_unique_subgroups_floc(
    df: pd.DataFrame,
    meter_floc_col: str = "MeterFLOC",
    subgroup_col: str = "Tx_Subgroup_ID",
    out_col: str = "Unique_Subgroups_FLOC",
) -> pd.DataFrame:
    """
    Adds per-FLOC DISTINCT subgroup counts (NULL-safe).
    Returns a new DataFrame.
    """
    d = df.copy()
    fl = _norm_str(d[meter_floc_col]) if meter_floc_col in d else pd.Series(np.nan, index=d.index)
    d["_mfloc_"] = fl

    counts = (
        d.loc[d["_mfloc_"].notna(), ["_mfloc_", subgroup_col]]
        .dropna(subset=[subgroup_col])
        .groupby("_mfloc_")[subgroup_col]
        .nunique()
    )

    d[out_col] = d["_mfloc_"].map(counts).fillna(0).astype(int)
    d.drop(columns=["_mfloc_"], inplace=True)
    return d


def apply_confidence_logic(
    df: pd.DataFrame,
    la_col: str = "LA_Transformer_ID",
    gis_col: str = "GIS_Transformer_ID",
    near_col: str = "Nearest_Transformer_ID",
    meter_floc_col: str = "MeterFLOC",
    xfmr_floc_col: str = "TransformerFLOC",
    subgroup_col: str = "Tx_Subgroup_ID",
    total_phases_col: str = "TotalPhasesInPole",
    out_conf_col: str = "GISAssociationConfidencePct",
    out_label_col: str = "Xfmr_Discrepancy",
) -> pd.DataFrame:
    """
    Applies confidence logic based on transformer ID matches and FLOC consistency.
    """
    d = df.copy()

    if "Unique_Subgroups_FLOC" not in d.columns:
        d = add_unique_subgroups_floc(
            d,
            meter_floc_col=meter_floc_col,
            subgroup_col=subgroup_col
        )

    la = d[la_col]
    gis = d[gis_col]
    near = d[near_col]
    eq = lambda x, y: x.eq(y) & x.notna() & y.notna()

    mfloc = _norm_str(d[meter_floc_col]) if meter_floc_col in d else pd.Series(np.nan, index=d.index)
    xfloc = _norm_str(d[xfmr_floc_col]) if xfmr_floc_col in d else pd.Series(np.nan, index=d.index)
    floc_ok = mfloc.eq(xfloc) & mfloc.notna() & xfloc.notna()

    phases = d[total_phases_col] if total_phases_col in d else pd.Series(np.nan, index=d.index)
    subgs = d["Unique_Subgroups_FLOC"]
    ge_true = phases.isna() | (phases >= subgs)

    all_match = eq(la, gis) & eq(la, near)
    gis_near = eq(gis, near) & ~eq(la, near)
    gis_la = eq(gis, la) & ~eq(near, gis)
    la_near = eq(la, near) & ~eq(gis, near)

    conf_hi = np.select(
        [all_match & floc_ok, gis_near & floc_ok, gis_la & floc_ok, la_near & floc_ok],
        [100, 90, 60, 30],
        default=10
    )
    conf_lo = np.select(
        [all_match & floc_ok, gis_near & floc_ok, gis_la & floc_ok, la_near & floc_ok],
        [90, 80, 50, 20],
        default=0
    )

    d[out_conf_col] = np.where(ge_true, conf_hi, conf_lo).astype(int)

    labels = np.select(
        [all_match & floc_ok, gis_near & floc_ok, gis_la & floc_ok, la_near & floc_ok],
        [
            "Accurate: All FLOCs Match",
            "Accurate: GIS=Nearest & FLOC",
            "GIS=LA & FLOC",
            "LA=Nearest & FLOC"
        ],
        default="Discrepancy"
    )
    d[out_label_col] = labels

    return d