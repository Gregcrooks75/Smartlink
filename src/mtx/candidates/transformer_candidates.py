from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
EARTH_RADIUS_FT = 20_902_231.0
def validcoordinates(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """
    Convert coordinates to numeric and remove
    invalid latitude/longitude values.
    """
    out = df.copy()
    out[
        lat_col
    ] = pd.to_numeric(
        out[
            lat_col
        ],
        errors="coerce",
    )
    out[
        lon_col
    ] = pd.to_numeric(
        out[
            lon_col
        ],
        errors="coerce",
    )
    out = out.dropna(
        subset=[
            lat_col,
            lon_col,
        ]
    )
    out = out[
        out[
            lat_col
        ].between(
            -90,
            90,
        )
        &
        out[
            lon_col
        ].between(
            -180,
            180,
        )
    ]
    return out.reset_index(
        drop=True
    )
def find_transformer_candidates(
    meter_groups: pd.DataFrame,
    ami_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    division: str,
    radius_ft: float = 550.0,
    transformer_division_col: str = "DivName",
) -> pd.DataFrame:
    """
    Find transformers within radius_ft of each meter.
    One output row =
        meter_number x candidate_transformer
    """
    groups = (
        meter_groups
        .copy()
    )
    ami = (
        ami_df
        .copy()
    )
    transformers = (
        transformer_df
        .copy()
    )
    # ==================================================
    # NORMALIZE AMI
    # ==================================================
    if "MeterID" in ami.columns:
        ami = ami.rename(
            columns={
                "MeterID": "meter_number",
            }
        )
    required_ami = {
        "meter_number",
        "LATITUDE",
        "LONGITUDE",
    }
    missing = (
        required_ami
        - set(ami.columns)
    )
    if missing:
        raise ValueError(
            "AMI missing columns: "
            f"{sorted(missing)}"
        )
    # ==================================================
    # NORMALIZE GROUPS
    # ==================================================
    required_groups = {
        "meter_number",
        "division",
        "tx_group_id",
        "tx_group_size",
        "tx_group_source",
        "meter_type",
    }
    missing = (
        required_groups
        - set(groups.columns)
    )
    if missing:
        raise ValueError(
            "Meter group file missing columns: "
            f"{sorted(missing)}"
        )
    groups = (
        groups[
            groups[
                "division"
            ]
            == division
        ]
        .copy()
    )
    # ==================================================
    # JOIN METER COORDINATES
    # ==================================================
    meter_xy = (
        ami[
            [
                "meter_number",
                "LATITUDE",
                "LONGITUDE",
            ]
        ]
        .drop_duplicates(
            "meter_number"
        )
    )
    meters = (
        groups
        .merge(
            meter_xy,
            on="meter_number",
            how="left",
        )
    )
    meters = validcoordinates(
        meters,
        lat_col="LATITUDE",
        lon_col="LONGITUDE",
    )
    # ==================================================
    # FILTER TRANSFORMERS TO TERRITORY
    # ==================================================
    required_transformers = {
        "FuncLoc",
        "Location_Y",
        "Location_X",
        'DivName'
    }
    missing = (
        required_transformers
        - set(transformers.columns)
    )
    print(f'Transformer division column is {transformer_division_col}')
    if missing:
        raise ValueError(
            "Transformer dataframe missing columns: "
            f"{sorted(missing)}"
        )
    if (
        transformer_division_col
        not in transformers.columns
    ):
        raise ValueError(
            "Transformer territory column "
            f"'{transformer_division_col}' "
            "does not exist. "
            "Set CONFIG.transformer_division_col "
            "to the correct field."
        )
    # ==================================================
    # MAP SMARTLINK DIVISIONS TO GIS DIVISIONS
    # ==================================================

    DIVISION_TO_GIS_DIVISION = {
    "Central": "Central/Rochester",
    "Sodus": "Lakeshore/Sodus",
    "Fillmore": "Genesee Valley/Filmore",
    }

    gis_division = DIVISION_TO_GIS_DIVISION.get(
    division,
    division,
    )

    print(
    f"Division mapping: "
    f"{division} -> {gis_division}"
    )

    # Normalize both sides before comparison
    transformer_divisions = (
    transformers[transformer_division_col]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.casefold()
    )

    target_division = (
    str(gis_division)
    .strip()
    .casefold()
    )

    transformers = transformers[
    transformer_divisions == target_division
    ].copy()

    print(
    f"Found {len(transformers):,} transformers "
    f"for division {division}"
    )

    transformers = (
    transformers
    .sort_values("FuncLoc")
    .drop_duplicates(
    "FuncLoc",
    keep="first",
    )
    )

    transformers = validcoordinates(
    transformers,
    lat_col="Location_Y",
    lon_col="Location_X",
    )

    print(
    f"After coordinate validation: "
    f"{len(transformers):,} transformers "
    f"for {division}"
    )



    # ==================================================
    # EMPTY CHECK
    # ==================================================
    if meters.empty:
        raise ValueError(
            f"No valid meters for {division}."
        )
    if transformers.empty:
        raise ValueError(
            f"No valid transformers for {division}."
        )
    # ==================================================
    # BUILD BALLTREE
    # ==================================================
    transformer_coords = np.radians(
        transformers[
            [
                "Location_Y",
                "Location_X",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )
    meter_coords = np.radians(
        meters[
            [
                "LATITUDE",
                "LONGITUDE",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )
    tree = BallTree(
        transformer_coords,
        metric="haversine",
    )
    radius_rad = (
        radius_ft
        / EARTH_RADIUS_FT
    )
    candidate_idx, candidate_dist = (
        tree.query_radius(
            meter_coords,
            r=radius_rad,
            return_distance=True,
            sort_results=True,
        )
    )
    # ==================================================
    # BUILD OUTPUT
    # ==================================================
    transformer_flocs = (
        transformers[
            "FuncLoc"
        ]
        .astype(str)
        .to_numpy()
    )
    rows = []
    group_columns = [
        "meter_number",
        "division",
        "tx_group_id",
        "tx_group_size",
        "tx_group_source",
        "meter_type",
    ]
    meter_metadata = (
        meters[
            group_columns
        ]
        .reset_index(
            drop=True
        )
    )
    for meter_idx, (
        tx_indices,
        distances,
    ) in enumerate(
        zip(
            candidate_idx,
            candidate_dist,
        )
    ):
        metadata = (
            meter_metadata
            .iloc[
                meter_idx
            ]
            .to_dict()
        )
        for (
            tx_idx,
            distance_rad,
        ) in zip(
            tx_indices,
            distances,
        ):
            row = {
                **metadata,
                "candidate_transformer":
                    transformer_flocs[
                        tx_idx
                    ],
                "distance_ft":
                    float(
                        distance_rad
                        * EARTH_RADIUS_FT
                    ),
            }
            rows.append(
                row
            )
    columns = (
        group_columns
        + [
            "candidate_transformer",
            "distance_ft",
        ]
    )
    if not rows:
        return pd.DataFrame(
            columns=columns
        )
    out = pd.DataFrame(
        rows
    )
    out = (
        out
        .sort_values(
            [
                "meter_number",
                "distance_ft",
                "candidate_transformer",
            ]
        )
        .drop_duplicates(
            subset=[
                "meter_number",
                "candidate_transformer",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )
    return out