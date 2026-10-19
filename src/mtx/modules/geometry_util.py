from pyproj import Geod
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
import geopandas as gpd
import math
# --------------------------------------------------
# Haversine distance function (feet)
# --------------------------------------------------
def haversine_feet(lat1, lon1, lat2, lon2):
    R = 6_371_000  # Earth radius in meters

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return R * c * 3.28084  # feet

def invert_coords(line):
    if line and not line.is_empty:
        return LineString([(lat, lon) for lon, lat in line.coords])
    else:
        return line

# Create LineString from comma-separated lat/lon strings
from pyproj import Geod
def create_geometry(lat_str, lon_str):
    if pd.isna(lat_str) or pd.isna(lon_str):
        return None

    latitudes = list(map(float, lat_str.split(',')))
    longitudes = list(map(float, lon_str.split(',')))

    if len(latitudes) >= 2:
        return LineString(zip(latitudes, longitudes))  # Correct order: (lon, lat)
    else:
        return Point(latitudes[0], longitudes[0])

def calculate_length(line):
    geod = Geod(ellps='WGS84')
    if line.geometry.type == 'MultiLineString':
        return sum(geod.geometry_length(geom) for geom in line.geomtry.geoms)
    else:
        return geod.geometry_length(line.geometry)


def find_line_and_distance(lat1, lat2, lon1, lon2):
    """
    Correct version:
    - Takes lon1, lon2, lat1, lat2 (in that order)
    - Builds LineString using (lon, lat)
    - Returns a DataFrame with 'Line' and 'Distance' in feet
    """
    geod = Geod(ellps='WGS84')
    results = {'Line': [], 'Distance': []}

    for lo1, lo2, la1, la2 in zip(lon1, lon2, lat1, lat2):
        if pd.notna(lo1) and pd.notna(lo2) and pd.notna(la1) and pd.notna(la2):
            # Create LineString in (lon, lat) format
            line = LineString([(lo1, la1), (lo2, la2)])
            distance = geod.geometry_length(line) * 3.28084  # Convert meters to feet
            results['Line'].append(line)
            results['Distance'].append(distance)
        else:
            results['Line'].append(None)
            results['Distance'].append(None)

    return pd.DataFrame(results)





def add_distance(df, lat1, lat2, lon1, lon2, name_prefix):
    df[[f'Line_{name_prefix}', f'Distance_{name_prefix}']] = find_line_and_distance(
        df[lat1],
        df[lat2],
        df[lon1],
        df[lon2]
    )
    return df

def create_point(df, lat_col, lon_col, new_col):
    df[new_col] = df.apply(
        lambda row: Point(row[lat_col], row[lon_col]),
        axis=1
    )
    return df

def Discrepancy_check(row, distance_col):
    distance = row[distance_col]

    if pd.isna(distance):
        return 'No FLOC Match'
    
    if distance >= 550:
        return '550 Feet or above'
    else:
        return 'Under 550 Feet'



def extract_xy(df, geometry_col, prefix):
    """
    Adds X and Y columns from a shapely Point geometry column.

    Args:
        df (pd.DataFrame): Input DataFrame with a geometry column.
        geometry_col (str): Name of the column containing shapely Point objects.
        prefix (str): Prefix for the new 'X' and 'Y' columns.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns '<prefix>_X' and '<prefix>_Y'.
    """
    geom_series = gpd.GeoSeries(df[geometry_col])
    df[f'{prefix}_X'] = geom_series.y
    df[f'{prefix}_Y'] = geom_series.x
    return df

import geopandas as gpd

def get_nearest_in_circuit(
    meters_gdf,
    transformers_gdf,
    circuit_col,
    transformer_geom_col,
    meter_geom_col,
    max_distance=None
):
    """
    Match each meter to the nearest transformer within the same circuit.

    Parameters:
    - meters_gdf (GeoDataFrame): GeoDataFrame of meters with geometry and circuit info.
    - transformers_gdf (GeoDataFrame): GeoDataFrame of transformers with geometry and circuit info.
    - circuit_col (str): Column name for circuit ID (must be present in both DataFrames).
    - transformer_geom_col (str): Name of the geometry column for transformers.
    - max_distance (float): Optional max search distance in units of the CRS.

    Returns:
    - GeoDataFrame: meters_gdf enriched with nearest transformer info in the same circuit.
    """
    original_crs = meters_gdf.crs
    project_crs = 'EPSG:32619'
    meters_gdf = meters_gdf.to_crs(project_crs)
    transformers_gdf = transformers_gdf.to_crs(project_crs)
    # Filter to essential columns and duplicate geometry
    transformers_gdf["Closest_Transformer_Location"] = transformers_gdf[transformer_geom_col]
    transformers_gdf = transformers_gdf[['SAPID', 'FuncLoc', circuit_col, transformer_geom_col, "Closest_Transformer_Location"]]

    # Perform spatial join by nearest (within same CRS)
    joined = gpd.sjoin_nearest(
        gpd.GeoDataFrame(meters_gdf, geometry=meter_geom_col),
        gpd.GeoDataFrame(transformers_gdf, geometry=transformer_geom_col),
        how="left",
        distance_col="nearest_distance",
        max_distance=max_distance
    )

    # Rename result columns for clarity
    joined = joined.rename(columns={
        'MeterFLOC_left': 'MeterFLOC',
        'SAPID_left': 'SAPID',
        'Transformer_Location_left': 'Transformer_Location',
        'All_Transformers_Location_right': 'Closest_Transformer_Location',
        'SAPID_right': 'Closest SAPID',
        'FuncLoc': 'Closest Transformer FLOC',
        'SAP_Address': 'Meter_Address'
    })
    
    joined = joined.to_crs(original_crs)

    # Drop technical join columns
    return joined.drop(columns=['index_right', 'nearest_distance'])


def process_all_circuits(transformers_gdf, meters_gdf, get_nearest_func):
    results = []
    
    # Get only circuits that exist in both dataframes
    common_circuits = set(transformers_gdf['CircuitID'].unique()) & set(meters_gdf['CircuitID'].unique())
    
    for circuit in common_circuits:
        # Filter data for this circuit
        transformers_sample = transformers_gdf[transformers_gdf['CircuitID'] == circuit]
        meters_sample = meters_gdf[meters_gdf['CircuitID'] == circuit]
        
        # Skip if one of the samples is empty
        if transformers_sample.empty or meters_sample.empty:
            continue
        
        # Run the nearest function
        matched = get_nearest_func(
            meters_gdf=meters_sample,
            transformers_gdf=transformers_sample,
            circuit_col="CircuitID",  # assumed to exist in both
            transformer_geom_col="All_Transformers_Location",
            meter_geom_col="Meter_Location",
            max_distance=None
        ).reset_index(drop=True)
        
        # Optional: add circuit column to keep track
        matched['CircuitID'] = circuit
        results.append(matched)
    
    # Combine all results into one GeoDataFrame
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def create_buffer(df, x_col, y_col, epsg_in=4326, epsg_out=32619, radius_ft=550):
    df = df.copy()
    df['geometry'] = df.apply(lambda row: Point(row[y_col], row[x_col]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{epsg_in}").to_crs(epsg=epsg_out)
    gdf['geometry'] = gdf.buffer(radius_ft * 0.3048)  # convert feet to meters
    return gdf.to_crs(epsg=epsg_in)['geometry']



import numpy as np

def get_candidate_transformers_from_center(
    center_lat,
    center_lon,
    df_transformer,
    radius_ft=None,
    tx_lat_col="Location_Y",
    tx_lon_col="Location_X",
):
    """
    Returns transformer candidates within a given radius (feet)
    of a single center point using a bounding box + Haversine distance.
    """

    # Earth radius in meters
    R = 6_371_000

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )

        return 2 * R * np.arcsin(np.sqrt(a))

    # --- Convert radius to meters ---
    radius_m = radius_ft * 0.3048

    # --- Fast bounding box filter ---
    d_lat = radius_m / 111_320.0
    d_lon = radius_m / (111_320.0 * np.cos(np.radians(center_lat)))

    df_tx = df_transformer.copy()
    df_tx[tx_lon_col] = pd.to_numeric(df_tx[tx_lon_col], errors="coerce")
    df_tx[tx_lat_col] = pd.to_numeric(df_tx[tx_lat_col], errors="coerce")

    df_tx = df_tx[
        df_tx[tx_lon_col].between(center_lon - d_lon, center_lon + d_lon)
        & df_tx[tx_lat_col].between(center_lat - d_lat, center_lat + d_lat)
    ].copy()

    if df_tx.empty:
        return df_tx.reset_index(drop=True)

    # --- Precise distance in feet ---
    df_tx["candidate_transformer_distance_from_center"] = (
        haversine(
            center_lat,
            center_lon,
            df_tx[tx_lat_col],
            df_tx[tx_lon_col],
        )
        * 3.28084
    )

    df_tx = (
        df_tx[
            df_tx["candidate_transformer_distance_from_center"] <= radius_ft
        ]
        .sort_values("candidate_transformer_distance_from_center")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    return df_tx
    
import os


def build_transformer_candidates_with_resume(
    df_groups,
    df_transformer,
    output_pickle_path,
    radius_ft=750,
    batch_size=100,
    group_label_col="Group_Label",
    center_lat_col="Centroid_Latitude",
    center_lon_col="Centroid_Longitude",
    tx_lat_col="Location_Y",
    tx_lon_col="Location_X",
):
    """
    Builds transformer candidate matches for all groups using centroid distance.
    Supports resume-safe batching via pickle persistence.

    Returns:
        pd.DataFrame with all candidate transformers.
    """

    results = []

    # --------------------------------------------------
    # 1. Load existing data (if exists)
    # --------------------------------------------------
    if os.path.exists(output_pickle_path):
        df_existing = pd.read_pickle(output_pickle_path)
        processed_groups = set(df_existing[group_label_col].unique())
        print(f"Resuming: {len(processed_groups)} groups already processed")
    else:
        df_existing = None
        processed_groups = set()

    # --------------------------------------------------
    # 2. Loop over groups (skip processed)
    # --------------------------------------------------
    for i, (_, row) in enumerate(df_groups.iterrows(), start=1):
        group_label = row[group_label_col]

        # 🔥 Skip already processed groups
        if group_label in processed_groups:
            continue

        center_lat = pd.to_numeric(row[center_lat_col], errors="coerce")
        center_lon = pd.to_numeric(row[center_lon_col], errors="coerce")

        if pd.isna(center_lat) or pd.isna(center_lon):
            continue

        df_candidates_group = get_candidate_transformers_from_center(
            center_lat=center_lat,
            center_lon=center_lon,
            df_transformer=df_transformer,
            radius_ft=radius_ft,
            tx_lat_col=tx_lat_col,
            tx_lon_col=tx_lon_col,
        )

        if not df_candidates_group.empty:
            df_candidates_group[group_label_col] = group_label
            results.append(df_candidates_group)

        # --------------------------------------------------
        # 3. Save every batch
        # --------------------------------------------------
        if len(results) >= batch_size:
            batch_df = pd.concat(results, ignore_index=True)

            if df_existing is not None:
                df_existing = pd.concat(
                    [df_existing, batch_df], ignore_index=True
                )
            else:
                df_existing = batch_df

            df_existing.to_pickle(output_pickle_path)
            print(f"Saved batch | total rows: {len(df_existing)}")

            processed_groups.update(
                batch_df[group_label_col].unique()
            )
            results = []

    # --------------------------------------------------
    # 4. Save remaining
    # --------------------------------------------------
    if results:
        batch_df = pd.concat(results, ignore_index=True)

        if df_existing is not None:
            df_existing = pd.concat(
                [df_existing, batch_df], ignore_index=True
            )
        else:
            df_existing = batch_df

        df_existing.to_pickle(output_pickle_path)
        print(f"Final save | total rows: {len(df_existing)}")

    return df_existing
