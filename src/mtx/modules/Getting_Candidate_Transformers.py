from mtx.data.extract_gis_data import get_devices_with_structures
from mtx.data.extract_ami_data import ami_data_ny, adam_data
import math
from reading_cluster_data import saving_by_chunks
from mtx.modules.geometry_util import build_transformer_candidates_with_resume
import os
import pandas as pd
from typing import Optional


transformer_data = get_devices_with_structures(
    server='nygisprd19',
    database= 'NYGIS',
    
    device='transformer')
df_ami = ami_data_ny()
df_adam = adam_data()
cluster_dir_path = r"C:\Users\u355394\OneDrive - IBERDROLA S.A\Documents - Asset Data Strategy\Asset Data Quality\Meter to Transformer Analysis\inputs\April 2026 - Data Lake Cluster Data"


# PARAMETERS

COLUMNS_TO_CONCAT = [
    'meter_number',
    'meter_floc',
    'READING_DATE',
    'avg_intra_cluster_corr',
    'incluster_zscore_corr',
    'Group Size',
    'Group_Label',
    'Division',
]


### DIVISION MAPPING
division_mapping = {
    # Exact matches
    "Plattsburgh": "Plattsburgh",
    "Liberty": "Liberty",
    "Ithaca": "Ithaca",
    "Oneonta": "Oneonta",
    "Brewster": "Brewster",
    "Mechanicville": "Mechanicville",
    "Lancaster": "Lancaster",
    "Geneva": "Geneva",
    "Binghamton": "Binghamton",
    "Hornell": "Hornell",
    "Elmira": "Elmira",
    "Auburn": "Auburn",
    "Lockport": "Lockport",
    "Canandaigua": "Canandaigua",

    # Normalizations / merges
    "Central": "Rochester",
    "Central / Rochester": "Rochester",
    "Central/Rochester": "Rochester",

    "Lakeshore / Sodus": "Sodus",
    "Lakeshore/Sodus": "Sodus",
    "Lakeshore": "Sodus",

    "Fillmore": "Fillmore",
    "Genesee Valley/Filmore": "Fillmore",
    "Genesee Valley / Filmore": "Fillmore",
}


### APPLYING DIVISION MAPPING
df_ami["Division"] = (
    df_ami["DivisionName"]
    .map(division_mapping)
    .fillna(df_ami["DivisionName"])
)
df_adam["Division"] = (
    df_adam["Division"]
    .map(division_mapping)
    .fillna(df_adam["Division"])
)
transformer_data["Division"] = (
    transformer_data["DivName"]
    .map(division_mapping)
    .fillna(transformer_data["DivName"])
)


#FUNCTION PARAMETERS
INPUT_DIRECTORY = r"C:\Users\u355394\OneDrive - IBERDROLA S.A\Documents - Asset Data Strategy\Asset Data Quality\Meter to Transformer Analysis\inputs\April 2026 - Data Lake Cluster Data" # Where we have our cluster data from data lake
BIG_SIZE_FILE_DIR = r'C:\Users\u355394\OneDrive - IBERDROLA S.A\Documents - Asset Data Strategy\Asset Data Quality\Meter to Transformer Analysis\pickles\Cluster Data' 
DIVISION = ["Rochester", 'Ithaca', 'Canandaigua'] # Optional - Specific Divisions to select
DIVISION_LIST = list(set(division_mapping.values())) # Optional - List of all our divisions
CANDIDATES_PATH = r'C:\Users\u355394\OneDrive - IBERDROLA S.A\Documents - Asset Data Strategy\Asset Data Quality\Meter to Transformer Analysis\pickles\Candidate_Data' # Where I store my candidate data

def Data_Reshape(df_cluster, df_candidates):
        df_candidates['Location_X'] = df_candidates['Location_X'].astype(str)
        df_candidates['Location_Y'] = df_candidates['Location_Y'].astype(str)
        df_candidates['candidate_transformer_distance_from_center'] = df_candidates['candidate_transformer_distance_from_center'].astype(str)
        reshaped_candidate_X = df_candidates.groupby('Group_Label')['Location_X'].agg(lambda x: ', '.join(set(x))).reset_index()
        reshaped_candidate_Y = df_candidates.groupby('Group_Label')['Location_Y'].agg(lambda x: ', '.join(set(x))).reset_index()
        reshaped_candidate_Distances = df_candidates.groupby('Group_Label')['candidate_transformer_distance_from_center'].agg(lambda x: ', '.join(set(x))).reset_index()
        reshaped_candidate_FLOCs = df_candidates.groupby('Group_Label')['FuncLoc'].agg(lambda x: ', '.join(set(x))).reset_index()
        map = {
            'Location_Y':'All_Candidate_Latitudes',
            'Location_X':'All_Candidate_Longitudes',
            'FuncLoc':'All_Candidate_FLOCs',
            'candidate_transformer_distance_from_center':'All_Distances_to_Group'
            
        }
        df1 = reshaped_candidate_Distances.merge(reshaped_candidate_FLOCs, on='Group_Label')
        df2 = df1.merge(reshaped_candidate_X, on='Group_Label')
        df3 = reshaped_candidate_Y.merge(df2, on='Group_Label').rename(columns=map)

        return df_cluster.merge(df3, on='Group_Label').drop(columns=['Centroid_Latitude', 'Centroid_Longitude'])
    
    
def finding_nearest_candidates(df_cluster, path_candidates_pickle, radius_ft, Transformer_Data):
    if os.path.exists(path_candidates_pickle):
        df_candidates = pd.read_pickle(path_candidates_pickle)
    else:
        group_columns = ['Group_Label', 'Centroid_Latitude', 'Centroid_Longitude']
        df_rochester_cluster_groups = df_cluster.drop_duplicates(subset=group_columns)[group_columns]
        df_candidates = build_transformer_candidates_with_resume(
            df_groups=df_rochester_cluster_groups,
            df_transformer=Transformer_Data,
            output_pickle_path=path_candidates_pickle,
            radius_ft=radius_ft,
            batch_size=100,
        )
        return df_candidates
        
def finding_centroid(df_cluster, df_ami):
    df_cluster = df_cluster.merge(df_ami[['MeterID', 'LATITUDE', 'LONGITUDE']], left_on='meter_number', right_on='MeterID').drop(columns='MeterID')
    df_cluster["Centroid_Latitude"] = (
        df_cluster
            .groupby("Group_Label")["LATITUDE"]
            .transform("mean")
    )

    df_cluster["Centroid_Longitude"] = (
        df_cluster
            .groupby("Group_Label")["LONGITUDE"]
            .transform("mean")
    )
    return df_cluster



def directory_has_file_with_strings(path, required_strings):
    """
    Return True if the directory exists and contains at least one file
    whose filename includes all required strings.
    """
    if not os.path.isdir(path):
        return False

    required = [s.lower() for s in required_strings]

    for filename in os.listdir(path):
        filename_lower = filename.lower()
        if all(s in filename_lower for s in required):
            return True

    return False

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


# --------------------------------------------------
# Distance calculation per row
# --------------------------------------------------
def get_candidate_distances_for_row(row):
    try:
        base_latitude = float(row["LATITUDE"])
        base_longitude = float(row["LONGITUDE"])

        lat_list = [
            float(x.strip())
            for x in str(row["All_Candidate_Latitudes"]).split(",")
        ]
        lon_list = [
            float(x.strip())
            for x in str(row["All_Candidate_Longitudes"]).split(",")
        ]

        if len(lat_list) != len(lon_list):
            return []

        distances = [
            haversine_feet(base_latitude, base_longitude, cand_lat, cand_lon)
            for cand_lat, cand_lon in zip(lat_list, lon_list)
        ]

        return distances

    except Exception:
        return []






def getting_around_seed_transformer(
    df_ami,
    all_divisions_list,
    Transformer_Data,
    chunk_size,
    radius_around_meter_centroid,
    input_directory,
    candidate_path,
    divisions_to_process: Optional[list] = None,
    df_adam=None,
    Big_Size=None
):
    dfs = []

    # --------------------------------------------------
    # Decide which divisions to load
    # --------------------------------------------------
    if divisions_to_process is not None:
        print(f"Processing only: {divisions_to_process}")
        divisions_list = divisions_to_process
    else:
        divisions_list = all_divisions_list

    # --------------------------------------------------
    # Load cluster data per division
    # --------------------------------------------------
    for division in divisions_list:
        input_path = os.path.join(input_directory, f"{division}.csv")

        if not os.path.exists(input_path):
            print(f"⚠ File not found for {division}: {input_path}")
            continue

        try:
            df_div = saving_by_chunks(
                input_path,
                division,
                chunk_size=chunk_size,
            )
            dfs.append(df_div)

        except Exception as e:
            print(f"⚠ Skipping division ({division}): {e}")

    if not dfs:
        raise ValueError("No division files were loaded.")

    final_df = pd.concat(dfs, ignore_index=True)

    df_cluster = (
        final_df[COLUMNS_TO_CONCAT]
        .drop_duplicates(subset=['meter_number', 'Group_Label'])
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Optional Adam filter
    # --------------------------------------------------
    if df_adam is not None:
        print("Getting Adam's Data to downsample the data for training")

        df_adam = adam_data()
        df_adam = (
            df_adam
            .drop_duplicates(subset=['meter_number', 'Truth'])
            .dropna(how='any')
            .reset_index(drop=True)
        )

        meter_filter = df_adam['meter_number'].tolist()

        df_cluster = (
            df_cluster[df_cluster['meter_number'].isin(meter_filter)]
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # Create centroids
    # --------------------------------------------------
    df_cluster = finding_centroid(df_cluster, df_ami)

    # --------------------------------------------------
    # Build / load candidate data per division
    # --------------------------------------------------
    candidate_dfs = []

    for division in df_cluster["Division"].dropna().unique():
        print(f"\nProcessing candidates for division: {division}")

        suffix = "_adam" if df_adam is not None else "" 
        pickle_path = os.path.join( candidate_path, f"df_candidates{suffix}_{division}.pkl" )

        df_division_cluster = (
            df_cluster[df_cluster["Division"] == division]
            .copy()
        )

        group_columns = [
            'Group_Label',
            'Centroid_Latitude',
            'Centroid_Longitude',
        ]

        df_division_cluster_groups = (
            df_division_cluster
            .drop_duplicates(subset=group_columns)[group_columns]
            .reset_index(drop=True)
        )

        df_division_transformers = (
            Transformer_Data[
                Transformer_Data['Division'] == division
            ]
            .reset_index(drop=True)
        )

        if os.path.exists(pickle_path):
            print(f"Reading existing pickle for {division}: {pickle_path}")
            df_candidates = pd.read_pickle(pickle_path)

        else:
            print(f"No pickle found for {division}. Building candidates...")
            df_candidates = build_transformer_candidates_with_resume(
                df_groups=df_division_cluster_groups,
                df_transformer=df_division_transformers,
                output_pickle_path=pickle_path,
                radius_ft=radius_around_meter_centroid,
                batch_size=100,
            )

        df_candidates["Division"] = division
        candidate_dfs.append(df_candidates)

    if not candidate_dfs:
        raise ValueError("No candidate dataframes were created.")
    ## df containing all candidate transformers i.e. transformers that are within the radius around the group centroids
    candidate_transformers_df = pd.concat(
        candidate_dfs,
        ignore_index=True,
    )[['FuncLoc', 'candidate_transformer_distance_from_center', 'Group_Label', 'Location_X', 'Location_Y']].rename(columns={'FuncLoc':'Candidate_Transformer', 'candidate_transformer_distance_from_center':'Distance_to_Group'})
    
    
    ### Concatenating that df to df_cluster containing the cluster data
    df_candidates = candidate_transformers_df.merge(df_cluster, on='Group_Label')
     ## df containing all candidate transformers i.e. transformers that are within the radius around the group centroids
    
    
        ### Big Volume Handling
    if Big_Size != None:
            ### Concatenating that df to df_cluster containing the cluster data
        df_candidates = Data_Reshape(df_cluster, df_candidates)
        return df_candidates
    else:
        df_candidates = candidate_transformers_df.merge(df_cluster, on='Group_Label')
        return df_candidates



def Main():
    final_candidate_df = getting_around_seed_transformer(
        df_ami=df_ami,
        all_divisions_list=DIVISION_LIST,
        Transformer_Data=transformer_data,
        chunk_size=2_000_000,
        radius_around_meter_centroid=700,
        input_directory=INPUT_DIRECTORY,
        candidate_path=CANDIDATES_PATH,
        divisions_to_process=None,   # or None to process all
        df_adam=df_adam,                  # or None to skip Adam filtering
        Big_Size=None
        )
            
