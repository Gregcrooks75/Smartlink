### SECTION: Importing Packages
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from pathlib import Path
from mtx.modules.geometry_util import (
    add_distance, create_point, Discrepancy_check, extract_xy)
from mtx.utils.config import load_config
from mtx.data.loaders import match_meters_to_transformers, load_transformers_data, load_ami_data

## Pickles
base_dir = Path.cwd()
dir = base_dir / "pickles"

def get_final_gdf():
    ### SECTION: Data Extraction
    pkl_path = dir / "Final_gdf.pkl"
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        return df
    else:
        df_meter_transformer = match_meters_to_transformers()

        ### SECTION: Distance - Meter to Transformer
        df_meter_transformer = add_distance(
            df_meter_transformer, 'MeterY', 'XfmrY', 'MeterX', 'XfmrX', 'Meter_Transformer_Matched'
        )

        ### SECTION: Filtering Discrepancies (> 550 Feet)
        df_meter_transformer['Discrepancy_check'] = df_meter_transformer.apply(
            lambda row: Discrepancy_check(row, 'Distance_Meter_Transformer_Matched'), axis=1
        )
                
        df_meter_transformer_discrepancies = df_meter_transformer[
            df_meter_transformer['Discrepancy_check'] == '550 Feet or above'
        ].reset_index(drop=True)

        ### SECTION: Geocoded Data Creation
        df_meter_transformer_discrepancies = create_point(
            df_meter_transformer_discrepancies, "MeterY", "MeterX", "Meter_Location"
        )
        df_meter_transformer_discrepancies = create_point(
            df_meter_transformer_discrepancies, "XfmrY", "XfmrX", "Transformer_Location"
        )
        print(df_meter_transformer_discrepancies)
        ### SECTION: Setup for Nearest Transformer Match
        Transformer_copy = load_transformers_data()
        Transformer_copy.rename(columns={'POINT_X': 'XfmrX', 'POINT_Y': 'XfmrY'}, inplace=True)
        Transformer_copy = create_point(Transformer_copy, "XfmrY", "XfmrX", "Transformer_Location")
        print(Transformer_copy)
        All_Meters_Copy = df_meter_transformer_discrepancies.copy().rename(columns={'Circuit': 'CircuitID'})
        Transformers = Transformer_copy[['Transformer_Location', 'CircuitID', 'SAPID', 'FuncLoc']]
        Transformers['All_Transformers_Location'] = Transformers['Transformer_Location']
        Transformers.rename(columns={
            'FuncLoc': 'XfmrFLOC',
            'CircuitID': 'Transformer_Circuit_ID'
        }, inplace=True)

        All_Meters_Copy.drop(columns=['OperatingVoltage', 'POLEID', 'TransformerType'], inplace=True)

        All_Transf = Transformer_copy[['FuncLoc', 'CircuitID']].rename(
            columns={'CircuitID': 'Transformer_Circuit_ID_Matched'}
        )
        All_Meters_Copy = All_Meters_Copy.merge(
            All_Transf, how='left', left_on='XfmrFLOC', right_on='FuncLoc'
        )

        ### SECTION: Format Columns and Drop Duplicates
        columns_to_keep = [
            'MeterID', 'CircuitID', 'Transformer_Circuit_ID_Matched', 'SAPID',
            'MeterFLOC', 'XfmrFLOC', 'SAP_Address', 'SAP_Town', 'ZIP', 'DivisionName',
            'OPCO', 'MeterX', 'MeterY', 'XfmrX', 'XfmrY',
            'Line_Meter_Transformer_Matched', 'Distance_Meter_Transformer_Matched',
            'Discrepancy_check', 'Meter_Location', 'Transformer_Location'
        ]
        All_Meters_Copy = All_Meters_Copy[columns_to_keep].drop_duplicates('MeterID')

        ### SECTION: Nearest Transformer Spatial Join
        Transformers['Transformer_Location'] = Transformers['Transformer_Location'].apply(
            lambda geom: Point(geom.coords[0][::-1])
        )
        All_Meters_Copy['Meter_Location'] = All_Meters_Copy['Meter_Location'].apply(
            lambda geom: Point(geom.coords[0][::-1])
        )

        Transformers = gpd.GeoDataFrame(Transformers, geometry='Transformer_Location', crs='EPSG:4326').to_crs(epsg=32619)
        All_Meters_Copy = gpd.GeoDataFrame(All_Meters_Copy, geometry='Meter_Location', crs='EPSG:4326').to_crs(epsg=32619)

        results = []
        for circuit in All_Meters_Copy.CircuitID.unique():
            gdf_meter_sub = All_Meters_Copy[All_Meters_Copy['CircuitID'] == circuit]
            gdf_transformers_sub = Transformers[Transformers['Transformer_Circuit_ID'] == circuit]
            if gdf_transformers_sub.empty or gdf_meter_sub.empty:
                continue
            joined = gpd.sjoin_nearest(
                gdf_meter_sub, gdf_transformers_sub, how='left', distance_col='Distance Meters'
            )
            results.append(joined)

        nearest = pd.concat(results).reset_index(drop=True)
        print(nearest)
        ### SECTION: Format Nearest Transformer Output
        nearest = gpd.GeoDataFrame(nearest, geometry='All_Transformers_Location', crs='EPSG:4326').to_crs(epsg=4326)
        nearest = gpd.GeoDataFrame(nearest, geometry='Meter_Location', crs='EPSG:32619').to_crs(epsg=4326)

        nearest['Distance Meters'] = nearest['Distance Meters'] * 3.28084  # meters to feet
        nearest['Distance_Closest_Transformer'] = nearest['Distance Meters'].round(6)

        nearest.drop(columns=['Distance Meters', 'index_right'], inplace=True)
        nearest.rename(columns={
            'SAPID_right': 'Closest SAPID',
            'XfmrFLOC_right': 'Closest FLOC',
            'All_Transformers_Location': 'Closest_Transformer_Location',
            'Transformer_Circuit_ID': 'Closest_Transformer_Circuit_ID'
        }, inplace=True)

        nearest.drop_duplicates('MeterID', inplace=True)
        nearest = nearest.reset_index(drop=True)

        ### SECTION: Extract X/Y Coordinates and Add Distances
        gpd_closest_transformer = nearest
        gpd_closest_transformer['Meter_Location'] = gpd_closest_transformer['Meter_Location'].apply(
            lambda geom: Point(geom.coords[0][::-1])
        )

        gpd_closest_transformer = extract_xy(gpd_closest_transformer, 'Transformer_Location', 'Transformer_Location')
        gpd_closest_transformer = extract_xy(gpd_closest_transformer, 'Meter_Location', 'Meter')
        gpd_closest_transformer = extract_xy(gpd_closest_transformer, 'Closest_Transformer_Location', 'Closest_Transformer')

        gpd_closest_transformer = add_distance(
            gpd_closest_transformer, 'XfmrY', 'Meter_Y', 'XfmrX', 'Meter_X', 'Meter_Transformer_Matched'
        )
        gpd_closest_transformer = add_distance(
            gpd_closest_transformer, 'Closest_Transformer_Y', 'Meter_Y',
            'Closest_Transformer_X', 'Meter_X', 'Meter_Closest_Transformer'
        )
        gpd_closest_transformer.to_pickle(pkl_path)
        return gpd_closest_transformer


if __name__ == '__main__':
    df = get_final_gdf()
    