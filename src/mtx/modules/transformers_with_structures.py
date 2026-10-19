import os
import pandas as pd
from mtx.modules.extract_transformer_from_db import get_transformer_data
from pathlib import Path

base_dir = Path.cwd()
dir = base_dir / "inputs"
# === DIVISION CODE LOOKUPS ===
def get_location_name(div_code, region='CMP'):
    div_code_map_cmp = {
        79: 'Brunswick', 80: 'Penobscot/Rockland', 81: 'Alfred', 82: 'Portland',
        83: 'Augusta', 84: 'Fairfield', 85: 'Lewiston', 86: 'Farmington',
        87: 'Dover', 88: 'Skowhegan', 89: 'Bridgton'
    }

    div_code_map_ny = {
        13: 'Canandaigua', 14: 'Central/Rochester', 15: 'Genesee Valley/Filmore',
        18: 'Lakeshore/Sodus', 26: 'Brewster', 30: 'Mechanicville',
        34: 'Plattsburgh', 38: 'Oneonta', 48: 'Liberty', 51: 'Lancaster',
        55: 'Lockport', 57: 'Elmira', 61: 'Hornell', 65: 'Geneva',
        68: 'Ithaca', 70: 'Auburn', 71: 'Binghamton',
        **div_code_map_cmp  # CMP codes reused for NY if needed
    }

    if region == 'CMP':
        return div_code_map_cmp.get(div_code, 'Unknown')
    else:
        return div_code_map_ny.get(div_code, 'Unknown Division Code')


# === LOAD STRUCTURE SHEETS ===
def load_structure_sheet(filepath, sheet_name, region):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df['Divname'] = df['DivCode'].map(lambda x: get_location_name(x, region))
    df['StructureType'] = sheet_name.split('_')[1]  # 'Surface', 'Support', etc.
    return df[['GISID', 'StructureType', 'FuncLoc', 'RoadName', 'TownName', 'TrueX', 'TrueY', 'POINT_X', 'POINT_Y']]


def load_all_structures():
    file_path = dir / "Structures NY - CMP.xlsx"

    # NY structures
    ny_underground = load_structure_sheet(file_path, 'NY_Underground_Structure', region='NY')
    ny_surface = load_structure_sheet(file_path, 'NY_Surface_Structure', region='NY')
    ny_support = load_structure_sheet(file_path, 'NY_Support_Structure', region='NY')

    # CMP structures
    cmp_underground = load_structure_sheet(file_path, 'ME_Underground_Structure', region='CMP')
    cmp_surface = load_structure_sheet(file_path, 'ME_Surface_Structure', region='CMP')
    cmp_support = load_structure_sheet(file_path, 'ME_Support_Structure', region='CMP')

    # Combine all
    return pd.concat([
        ny_underground, ny_surface, ny_support,
        cmp_underground, cmp_surface, cmp_support
    ]).reset_index(drop=True)


# === MAIN MERGE OPERATION ===
def merge_transformer_and_structure():
    transformer_df = get_transformer_data()
    structure_df = load_all_structures()
    merged_df = transformer_df.merge(
        structure_df,
        how='left',
        left_on='POLEID',
        right_on='GISID'
    )
    return merged_df


# === RUN IF MAIN ===
if __name__ == "__main__":
    final_df = merge_transformer_and_structure()
    print(final_df.head())  # For preview
