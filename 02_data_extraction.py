"""
LTPP Pavement Deterioration Under Climate Variability
Script 2: Data Extraction and Merging from SQL Server
Author: Metehan Alp Memis
Journal: Transportation Geotechnics (TRGEO-D-26-01076)

Requirements:
    pip install pyodbc sqlalchemy pandas numpy
"""

import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATABASE CONNECTION
# ============================================================

SERVER = r'ADMIN\SQLEXPRESS01'  # Update with your instance name

def get_engine(database):
    return create_engine(
        f'mssql+pyodbc://{SERVER}/{database}'
        '?driver=ODBC+Driver+17+for+SQL+Server'
        '&trusted_connection=yes'
    )

engine_main  = get_engine('LTPP_MAIN')
engine_merra = get_engine('LTPP_MERRA')
engine_mat   = get_engine('LTPP_MATERIAL')

print('Database connections established')

# ============================================================
# 2. LOAD RAW TABLES
# ============================================================

print('\n=== LOADING RAW DATA ===\n')

# IRI performance data
df_iri = pd.read_sql('SELECT * FROM ANALYSIS_IRI', engine_main)
print(f'IRI:      {df_iri.shape[0]:,} rows, {df_iri.shape[1]} cols')

# Rutting data
df_rut = pd.read_sql('SELECT * FROM ANALYSIS_RUTTING', engine_main)
print(f'Rutting:  {df_rut.shape[0]:,} rows, {df_rut.shape[1]} cols')

# Traffic data
df_trf = pd.read_sql('SELECT * FROM TRF_TREND', engine_main)
print(f'Traffic:  {df_trf.shape[0]:,} rows, {df_trf.shape[1]} cols')

# Section info
df_shrp = pd.read_sql('SELECT * FROM SHRP_INFO', engine_main)
print(f'Sections: {df_shrp.shape[0]:,} rows, {df_shrp.shape[1]} cols')

# MERRA climate grid mapping
df_grid = pd.read_sql('SELECT * FROM MERRA_GRID_SECTION', engine_merra)
print(f'Grid:     {df_grid.shape[0]:,} rows, {df_grid.shape[1]} cols')

# MERRA climate variables (annual)
df_temp   = pd.read_sql('SELECT * FROM VW_MERRA_TEMP_YEAR',   engine_merra)
df_precip = pd.read_sql('SELECT * FROM VW_MERRA_PRECIP_YEAR', engine_merra)
df_humid  = pd.read_sql('SELECT * FROM VW_MERRA_HUMID_YEAR',  engine_merra)
df_wind   = pd.read_sql('SELECT * FROM VW_MERRA_WIND_YEAR',   engine_merra)
print(f'Climate (temp/precip/humid/wind): {df_temp.shape[0]:,} rows each')

# Subgrade/unbound materials
df_unbound = pd.read_sql('SELECT * FROM ANALYSIS_TST_UNBOUND', engine_mat)
print(f'Unbound:  {df_unbound.shape[0]:,} rows, {df_unbound.shape[1]} cols')

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print('\n=== FEATURE ENGINEERING ===\n')

# 3a. IRI: annual average from visit measurements
df_iri['YEAR'] = pd.to_datetime(df_iri['VISIT_DATE']).dt.year
df_iri_annual = df_iri.groupby(['STATE_CODE', 'SHRP_ID', 'YEAR']).agg(
    IRI_LEFT  = ('IRI_LEFT_WHEEL_PATH',  'mean'),
    IRI_RIGHT = ('IRI_RIGHT_WHEEL_PATH', 'mean')
).reset_index()
df_iri_annual['IRI_AVG'] = (
    df_iri_annual['IRI_LEFT'] + df_iri_annual['IRI_RIGHT']
) / 2
print(f'IRI annual:   {df_iri_annual.shape[0]:,} rows')

# 3b. Rutting: annual average
df_rut['YEAR'] = pd.to_datetime(df_rut['SURVEY_DATE']).dt.year
df_rut_annual = df_rut.groupby(['STATE_CODE', 'SHRP_ID', 'YEAR']).agg(
    RUTTING_LEFT  = ('LLH_DEPTH_1_8_MEAN', 'mean'),
    RUTTING_RIGHT = ('RLH_DEPTH_1_8_MEAN', 'mean')
).reset_index()
df_rut_annual['RUTTING_AVG'] = (
    df_rut_annual['RUTTING_LEFT'] + df_rut_annual['RUTTING_RIGHT']
) / 2
print(f'Rutting annual: {df_rut_annual.shape[0]:,} rows')

# 3c. Traffic: already annual
df_trf_annual = df_trf.groupby(['STATE_CODE', 'SHRP_ID', 'YEAR']).agg(
    AADTT           = ('AADTT_ALL_TRUCKS_TREND',    'mean'),
    ANNUAL_TRUCK_VOL= ('ANNUAL_TRUCK_VOLUME_TREND', 'mean')
).reset_index()
print(f'Traffic annual: {df_trf_annual.shape[0]:,} rows')

# 3d. Climate: merge MERRA grid with annual variables
df_temp_grid = df_grid.merge(df_temp,   on='MERRA_ID', how='inner')
df_climate   = df_temp_grid.merge(df_precip, on=['MERRA_ID', 'YEAR'], how='inner')
df_climate   = df_climate.merge(df_humid,    on=['MERRA_ID', 'YEAR'], how='inner')
df_climate   = df_climate.merge(df_wind,     on=['MERRA_ID', 'YEAR'], how='inner')
print(f'Climate merged: {df_climate.shape[0]:,} rows')

# 3e. Subgrade: SS = Subgrade Soil layer type
numeric_cols = ['PLASTICITY_INDEX', 'RESILIENT_MODULUS', 'LIQUID_LIMIT']
df_sub_raw = df_unbound[df_unbound['LAYER_TYPE'] == 'SS'].copy()
for col in numeric_cols:
    df_sub_raw[col] = pd.to_numeric(df_sub_raw[col], errors='coerce')

df_sub = df_sub_raw.groupby(['STATE_CODE', 'SHRP_ID']).agg(
    AASHTO_CLASS      = ('AASHTO_SOIL_CLASS', 'first'),
    PLASTICITY_INDEX  = ('PLASTICITY_INDEX',  'mean'),
    RESILIENT_MODULUS = ('RESILIENT_MODULUS',  'mean'),
    LIQUID_LIMIT      = ('LIQUID_LIMIT',       'mean')
).reset_index()
print(f'Subgrade: {df_sub.shape[0]:,} rows')

# ============================================================
# 4. MERGE ALL DATASETS
# ============================================================

print('\n=== MERGING DATASETS ===\n')

climate_cols = [
    'STATE_CODE', 'SHRP_ID', 'YEAR',
    'LATITUDE', 'LONGITUDE',
    'TEMP_AVG', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'
]

df = df_iri_annual.copy()
df = df.merge(
    df_rut_annual[['STATE_CODE', 'SHRP_ID', 'YEAR', 'RUTTING_AVG']],
    on=['STATE_CODE', 'SHRP_ID', 'YEAR'], how='left'
)
df = df.merge(df_trf_annual, on=['STATE_CODE', 'SHRP_ID', 'YEAR'], how='left')
df = df.merge(df_climate[climate_cols], on=['STATE_CODE', 'SHRP_ID', 'YEAR'], how='left')
df = df.merge(df_sub, on=['STATE_CODE', 'SHRP_ID'], how='left')

print(f'Merged dataset: {df.shape[0]:,} rows, {df.shape[1]} cols')
print(f'\nMissing value rates (%):\n{(df.isnull().sum() / len(df) * 100).round(1)}')

# ============================================================
# 5. CLEAN AND SAVE
# ============================================================

print('\n=== CLEANING AND SAVING ===\n')

keep_cols = [
    'STATE_CODE', 'SHRP_ID', 'YEAR',
    'IRI_AVG', 'RUTTING_AVG',
    'AADTT', 'ANNUAL_TRUCK_VOL',
    'LATITUDE', 'LONGITUDE',
    'TEMP_AVG', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION'
]

df_clean = df[keep_cols].copy()
df_clean = df_clean.dropna(subset=[
    'IRI_AVG', 'TEMP_AVG', 'PRECIPITATION',
    'FREEZE_INDEX', 'FREEZE_THAW'
])

print(f'Clean dataset: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} cols')

df_clean.to_csv('LTPP_merged_clean.csv', index=False)
print('Saved: LTPP_merged_clean.csv')
