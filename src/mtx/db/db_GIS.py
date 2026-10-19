from __future__ import annotations
import numpy as np
# devices_with_structures.py
# -------------------------------------------------
# Build structures once, pull one or all device tables,
# join on PoleID == GISID, and return a dict of DataFrames
# plus a combined DataFrame.
import pyodbc
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path
# -----------------------------
# Connection helpers
# -----------------------------

def get_connection(server: str, database: str):
    conn_str = (
        "Driver={SQL Server};"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)

def run_query_trusted(query: str, server: str, database: str) -> pd.DataFrame:
    with get_connection(server, database) as conn:
        return pd.read_sql_query(query, conn)

def run_query_with_temp_trusted(query_setup: str, query_final: str, server: str, database: str) -> pd.DataFrame:
    conn = get_connection(server, database)
    cur = conn.cursor()
    cur.execute("SET NOCOUNT ON; " + query_setup)
    while cur.nextset():
        pass
    df = pd.read_sql_query(query_final, conn)
    cur.close()
    conn.close()
    return df

# -----------------------------
# Reference maps
# -----------------------------

DIV_CODE_MAP = {
    13: 'Canandaigua', 14: 'Central/Rochester', 15: 'Genesee Valley/Filmore',
    18: 'Lakeshore/Sodus', 26: 'Brewster', 30: 'Mechanicville',
    34: 'Plattsburgh', 38: 'Oneonta', 48: 'Liberty', 51: 'Lancaster',
    55: 'Lockport', 57: 'Elmira', 61: 'Hornell', 65: 'Geneva',
    68: 'Ithaca', 70: 'Auburn', 71: 'Binghamton', 79: 'Brunswick',
    80: 'Penobscot/Rockland', 81: 'Alfred', 82: 'Portland', 83: 'Augusta',
    84: 'Fairfield', 85: 'Lewiston', 86: 'Farmington', 87: 'Dover',
    88: 'Skowhegan', 89: 'Bridgton'
}

OPCO_MAP = {
    9301: 'NYSEG',
    9302: 'RGE',
    9310: 'CMP'
}

phase_map = {
    1: 'C',
    2: 'B',
    3: 'BC',
    4: 'A',
    5: 'AC',
    6: 'AB',
    7: 'ABC'
}


# -----------------------------
# Device specifications & SQL
# -----------------------------

DEVICE_SPECS: Dict[str, Dict[str, str]] = {
    "transformer": {
        "unit_table": "Electric.XfmrUnit_evw",
        "parent_table": "Electric.Transformer_evw",
        "unit_id": "TransformerGUID",
        "parent_id": "GlobalID"
    },
    "recloser": {
        "unit_table": "Electric.RECLOSERUNIT_EVW",
        "parent_table": "Electric.DYNAMICPROTECTIVEDEVICE_EVW",
        "unit_id": "RecloserGUID",
        "parent_id": "GlobalID"
    },
    "sectionalizer": {
        "unit_table": "Electric.SECTIONALIZERUNIT",
        "parent_table": "Electric.DYNAMICPROTECTIVEDEVICE_EVW",
        "unit_id": "SectionalizerGUID",
        "parent_id": "GlobalID"
    },
    "line_sensor": {
        "unit_table": "Electric.LineSensorUnit_evw",
        "parent_table": "Electric.DYNAMICPROTECTIVEDEVICE_EVW",
        "unit_id": "SensorGUID",
        "parent_id": "GlobalID"
    },
    "capacitor": {
        "unit_table": "Electric.CapacitorUnit_evw",
        "parent_table": "Electric.PFCorrectingEquipment_evw",
        "unit_id": "CapacitorBankGUID",
        "parent_id": "GlobalID"
    },
    "voltage_regulator": {
        "unit_table": "Electric.VOLTAGEREGULATORUNIT_EVW",
        "parent_table": "Electric.VOLTAGEREGULATOR_EVW",
        "unit_id": "VoltageRegulatorGUID",
        "parent_id": "GlobalID"
    },
    "fuse_cutout": {
        "unit_table": "Electric.FUSEUNIT_EVW",
        "parent_table": "Electric.Fuse_evw",
        "unit_id": "FuseObjectGUID",
        "parent_id": "GlobalID"
    },
    "switch": {
        "unit_table": "Electric.SWITCHUNIT_EVW",
        "parent_table": "Electric.SWITCH_EVW",
        "unit_id": "SwitchGUID",
        "parent_id": "GlobalID"
    },
    "fault_indicator": {
        "unit_table": "Electric.FaultIndUnit_evw",
        "parent_table": "Electric.MISCNETWORKFEATURE_EVW",
        "unit_id": "MiscObjectGUID",
        "parent_id": "GlobalID"
    },
    "terminating_cabinet": {
        "unit_table": "Electric.TermCabinetUnit_evw",
        "parent_table": "Electric.MISCNETWORKFEATURE_EVW",
        "unit_id": "MiscObjectGUID",
        "parent_id": "GlobalID"
    },
    "switchgear": {
        "unit_table": "Electric.SwitchGearUnit_evw",
        "parent_table": "Electric.MISCNETWORKFEATURE_EVW",
        "unit_id": "MiscObjectGUID",
        "parent_id": "GlobalID"
    },
    "net_protector": {
        "unit_table": "Electric.NetProtectorUnit_evw",
        "parent_table": "Electric.MISCNETWORKFEATURE_EVW",
        "unit_id": "MiscObjectGUID",
        "parent_id": "GlobalID"
    }
}

DEVICE_SQL_TEMPLATE = """
SELECT 
    ru.[GISID], ru.[SAPID], ru.[CUDesc], ru.[SerialNumber], ru.[WorkOrderID], 
    ru.[InSvcDate], ru.[{unit_id}] AS DeviceGUID, ru.[ConstructionStatus], 
    dpd.[OperatingVoltage], dpd.[PoleID], dpd.PhaseDesignation, dpd.SubtypeCD, dpd.FeederID
FROM {unit_table} ru
LEFT JOIN {parent_table} dpd
    ON ru.[{unit_id}] = dpd.[{parent_id}];
"""

TRANSFORMER_SQL_TEMPLATE = """
SELECT 
    ru.[GISID], ru.[SAPID], ru.[CUDesc], ru.[SerialNumber], ru.[WorkOrderID], 
    ru.[InSvcDate],
    ru.[RatedKVA] AS Transformer_kVA,
    ru.[TransformerGUID] AS DeviceGUID,
    ru.[ConstructionStatus],
    dpd.[OperatingVoltage], dpd.[PoleID], dpd.PhaseDesignation, dpd.SubtypeCD, dpd.FeederID
FROM Electric.XfmrUnit_evw ru
LEFT JOIN Electric.Transformer_evw dpd
    ON ru.[TransformerGUID] = dpd.[GlobalID];
"""


# -----------------------------
# Structures query (with X/Y)
# -----------------------------

STRUCTURES_QUERY_SETUP = """
DROP TABLE IF EXISTS #Structure;

SELECT [COCode], [SAPID], [GISID], [DivCode], [FuncLoc],
Shape.STX AS Location_X,
Shape.STY AS Location_Y,
TrueX,
TrueY,
 'Pole' AS Type
INTO #Structure
FROM Electric.SupportStructure_evw

UNION ALL

SELECT [COCode], [SAPID], [GISID], [DivCode], [FuncLoc],
Shape.STX AS Location_X,
Shape.STY AS Location_Y,

TrueX,
TrueY,
CASE 
 WHEN [SubtypeCD] = 1 THEN 'Handhole'
 WHEN [SubtypeCD] = 2 THEN 'Mandhole'
 WHEN [SubtypeCD] = 3 THEN 'Vault'
 ELSE 'NA_Underground'
 END AS Type
FROM Electric.UndergroundStructure_evw

UNION ALL

SELECT [COCode], [SAPID], [GISID], [DivCode], [FuncLoc],
Shape.STX AS Location_X,
Shape.STY AS Location_Y,
TrueX,
TrueY,
 'Vault Pad' AS Type
FROM Electric.SURFACESTRUCTURE_EVW

UNION ALL

SELECT [COCode], [SAPID], [GISID], NULL AS [DivCode], [FuncLoc],
 NULL AS Location_X, NULL AS Location_Y,
 NULL AS TrueX,
NULL AS TrueY,
 'Vert' AS Type
FROM Electric.Vert;
"""

STRUCTURES_QUERY_FINAL = "SELECT * FROM #Structure;"

query_transformers_smweb = """
SELECT SAPID AS SAPID, 
	Latitude AS trans_Lat,
	Longitude AS trans_Long
FROM [UAD].[Inputs].[GIS_Transformers_SMWeb]
"""


def get_most_recent_folder(base_path):
    base = Path(base_path)
    
    # List only directories and exclude those ending with '_report'
    dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.lower().endswith('_report')]
    
    if not dirs:
        return None
    
    # Get the most recently created directory
    most_recent = max(dirs, key=lambda d: d.stat().st_ctime)
    return most_recent

# -----------------------------
# Public API
# -----------------------------

def get_devices_with_structures(
    server: str,
    database: str,
    structures: Optional[str] = None,
    device: Optional[str] = None,
    save_to_csv: Optional[str] = None,
    csv_path: Optional[str] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fetch one or all device tables, join with structures on PoleID==GISID, and return:
    - dict of dataframes per device
    - combined dataframe of all devices
    """

    all_dfs = []
    if structures == 'structures':
        df_structures = run_query_with_temp_trusted(query_setup=STRUCTURES_QUERY_SETUP, query_final=STRUCTURES_QUERY_FINAL, server=server, database=database)
        return df_structures
    
    if device is None and structures is None:
        df_structures = run_query_with_temp_trusted(query_setup=STRUCTURES_QUERY_SETUP, query_final=STRUCTURES_QUERY_FINAL, server=server, database=database)
        for device, spec in DEVICE_SPECS.items():
            sql = DEVICE_SQL_TEMPLATE.format(**spec)
            df = run_query_trusted(query=sql, server=server, database=database)
        
            # Store individual DataFrame if needed
            globals()[f"df_{device}"] = df
            df['Device'] = f'{device}'
            # Append to list
            all_dfs.append(df)
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all.merge(
        df_structures[['GISID', 'COCode', 'DivCode', 'FuncLoc', 'Type', 'Location_X', 'Location_Y']],
        left_on='PoleID',
        right_on='GISID',
        how='left'
        ).rename(columns={'GISID_x': 'GISID', 'Type':'StructureType'}).drop(columns=['GISID_y']).reset_index(drop=True)
        #df_all['PhaseClass'] = df_all['PhaseDesignation'].map(phase_map)
        df_all["DivName"] = df_all["DivCode"].astype(float).map(DIV_CODE_MAP)
        df_all['OPCO'] = df_all["COCode"].astype(float).map(OPCO_MAP)
        df_all = df_all.drop(columns=['DivCode', 'COCode'])
        # Specifying whether the equipment is 3-phase or not
        #df["Three_Phase_Status"] = np.where(df["PhaseClass"] == 'ABC', "3-Phase", "Not 3-Phase")
        if save_to_csv and csv_path:
            try:
                df_all.to_csv(csv_path, index=False)
                print(f"Saved to CSV at: {csv_path}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

        return df_all
    else:
        device = device.lower()
        df_structures = run_query_with_temp_trusted(query_setup=STRUCTURES_QUERY_SETUP, query_final=STRUCTURES_QUERY_FINAL, server=server, database=database)

        if device in DEVICE_SPECS:
            spec = DEVICE_SPECS[device]
            if device == "transformer":
                sql = TRANSFORMER_SQL_TEMPLATE
            else:
                sql = DEVICE_SQL_TEMPLATE.format(**spec)
            df = run_query_trusted(query=sql, server=server, database=database)
            df = df.merge(
                df_structures[['GISID', 'COCode', 'DivCode', 'FuncLoc', 'Type', 'Location_X', 'Location_Y']],
                left_on='PoleID',
                right_on='GISID',
                how='left'
            ).rename(columns={'GISID_x': 'GISID', 'Type':'StructureType', 'FeederID':'CircuitID'}).drop(columns=['GISID_y']).reset_index(drop=True)
            df['Device'] = device
            df['PhaseClass'] = df['PhaseDesignation'].map(phase_map)
            df['OPCO'] = df["COCode"].astype(float).map(OPCO_MAP)

            if save_to_csv and csv_path:
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"Saved to CSV at: {csv_path}")
                except Exception as e:
                    print(f"Failed to save CSV: {e}")
            # Adding Division Names
            df["DivName"] = df["DivCode"].astype(float).map(DIV_CODE_MAP)
            df = df.drop(columns=['DivCode', 'COCode', 'PhaseDesignation'])
        
            # Specifying whether the equipment is 3-phase or not
            df["Three_Phase_Status"] = np.where(df["PhaseClass"] == 'ABC', "3-Phase", "Not 3-Phase")
            return df

        else:
            print(f"Device '{device}' not found. Available options: {list(DEVICE_SPECS.keys())}")
            return None, None

# -----------------------------
# Optional: list available devices
# -----------------------------

def list_devices() -> List[str]:
    return list(DEVICE_SPECS.keys())