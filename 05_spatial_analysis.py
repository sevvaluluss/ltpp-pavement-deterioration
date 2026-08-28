"""
LTPP Pavement Deterioration Under Climate Variability
Script 5: Spatial and Regional Risk Analysis
Author: Metehan Alp Memis
Journal: Transportation Geotechnics (TRGEO-D-26-01076)

Requirements:
    pip install pandas numpy matplotlib plotly joblib scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. STATE-LEVEL RISK MAPPING
# ============================================================

df_ml = pd.read_csv('LTPP_ml_dataset.csv')

FEATURES = [
    'TEMP_AVG', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION',
    'AADTT', 'ANNUAL_TRUCK_VOL', 'PAVEMENT_AGE', 'YEAR'
]

imputer   = joblib.load('imputer.pkl')
best_model= joblib.load('model_Random_Forest.pkl')

X_all = df_ml[FEATURES].fillna(df_ml[FEATURES].median())
X_imp = imputer.transform(X_all)
df_ml['PRED_PROB'] = best_model.predict_proba(X_imp)[:, 1]

# State code to abbreviation mapping
STATE_MAPPING = {
    1:'AL', 2:'AK', 4:'AZ', 5:'AR', 6:'CA', 8:'CO', 9:'CT',
    10:'DE', 12:'FL', 13:'GA', 15:'HI', 16:'ID', 17:'IL',
    18:'IN', 19:'IA', 20:'KS', 21:'KY', 22:'LA', 23:'ME',
    24:'MD', 25:'MA', 26:'MI', 27:'MN', 28:'MS', 29:'MO',
    30:'MT', 31:'NE', 32:'NV', 33:'NH', 34:'NJ', 35:'NM',
    36:'NY', 37:'NC', 38:'ND', 39:'OH', 40:'OK', 41:'OR',
    42:'PA', 44:'RI', 45:'SC', 46:'SD', 47:'TN', 48:'TX',
    49:'UT', 50:'VT', 51:'VA', 53:'WA', 54:'WV', 55:'WI',
    56:'WY', 72:'PR', 83:'MB', 86:'SK', 87:'AB', 89:'ON', 90:'QC'
}

# LTPP climatic zone assignment (US states only)
CLIMATIC_ZONES = {
    'AL':4,'AK':2,'AZ':3,'AR':4,'CA':3,'CO':2,'CT':1,'DE':1,
    'FL':4,'GA':4,'HI':4,'ID':2,'IL':1,'IN':1,'IA':1,'KS':2,
    'KY':1,'LA':4,'ME':1,'MD':1,'MA':1,'MI':1,'MN':1,'MS':4,
    'MO':1,'MT':2,'NE':2,'NV':3,'NH':1,'NJ':1,'NM':3,'NY':1,
    'NC':4,'ND':1,'OH':1,'OK':4,'OR':2,'PA':1,'RI':1,'SC':4,
    'SD':1,'TN':4,'TX':4,'UT':2,'VT':1,'VA':1,'WA':2,'WV':1,
    'WI':1,'WY':2
}
ZONE_NAMES = {
    1:'Wet-Freeze', 2:'Dry-Freeze',
    3:'Dry-No Freeze', 4:'Wet-No Freeze'
}

# State-level risk aggregation
state_risk = df_ml.groupby('STATE_CODE').agg(
    DETERI_RISK     = ('PRED_PROB',    'mean'),
    FREEZE_IDX_AVG  = ('FREEZE_INDEX', 'mean'),
    TEMP_AVG        = ('TEMP_AVG',     'mean'),
    PRECIP_AVG      = ('PRECIPITATION','mean'),
    N_SECTIONS      = ('SHRP_ID',      'nunique')
).reset_index().sort_values('DETERI_RISK', ascending=False)

state_risk['STATE_ABBR'] = state_risk['STATE_CODE'].map(STATE_MAPPING)

# ============================================================
# 2. STATE RISK BAR CHART (Top 20)
# ============================================================

state_risk_us = state_risk[state_risk['STATE_CODE'] <= 90].copy()
state_risk_us['CLIMATE_ZONE'] = state_risk_us['STATE_ABBR'].map(CLIMATIC_ZONES)
state_risk_us['CLIMATE_NAME'] = state_risk_us['CLIMATE_ZONE'].map(ZONE_NAMES)

top20 = state_risk_us.head(20)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(top20['STATE_ABBR'], top20['DETERI_RISK'],
              color='steelblue', edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top20['DETERI_RISK']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('State / Province', fontsize=12)
ax.set_ylabel('Mean Deterioration Risk Probability', fontsize=12)
ax.set_title(
    'Pavement Deterioration Risk by State and Province\n'
    '(Random Forest Classifier, LTPP 2,574 Sections)',
    fontsize=12
)
ax.set_ylim(0, 1.0)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('State_Risk_v2.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: State_Risk_v2.png')

# ============================================================
# 3. CHOROPLETH MAP (US States)
# ============================================================

fig_map = px.choropleth(
    state_risk_us[state_risk_us['STATE_CODE'] <= 56],
    locations='STATE_ABBR',
    locationmode='USA-states',
    color='DETERI_RISK',
    scope='usa',
    color_continuous_scale='RdYlGn_r',
    title='Pavement Deterioration Risk by State (LTPP Analysis)',
    labels={'DETERI_RISK': 'Risk Probability'}
)
fig_map.update_layout(title_font_size=16)
fig_map.write_html('State_Risk_Map.html')
print('Saved: State_Risk_Map.html')

# ============================================================
# 4. CLIMATIC REGION RISK
# ============================================================

region_risk = state_risk_us.groupby('CLIMATE_NAME').agg(
    RISK_MEAN = ('DETERI_RISK', 'mean'),
    RISK_STD  = ('DETERI_RISK', 'std'),
    N_STATES  = ('STATE_ABBR',  'count')
).reset_index()

print('\nClimatic Region Risk:')
print(region_risk.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors_region = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']
bars = ax.bar(
    region_risk['CLIMATE_NAME'], region_risk['RISK_MEAN'],
    yerr=region_risk['RISK_STD'], capsize=5,
    color=colors_region, alpha=0.85, edgecolor='black'
)
for bar, (_, row) in zip(bars, region_risk.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + row['RISK_STD'] + 0.02,
        f'n={int(row["N_STATES"])}', ha='center', fontsize=10
    )
ax.set_ylabel('Mean Deterioration Risk Probability', fontsize=12)
ax.set_xlabel('Climatic Region (LTPP Classification)', fontsize=12)
ax.set_title(
    'Pavement Deterioration Risk by Climatic Region\n'
    '(LTPP National Dataset)',
    fontsize=12
)
ax.set_ylim(0, 0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Climatic_Region_Risk.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: Climatic_Region_Risk.png')

# ============================================================
# 5. CLIMATE-RISK SCATTER PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Freeze Index vs Risk
sc1 = axes[0].scatter(
    state_risk_us['FREEZE_IDX_AVG'],
    state_risk_us['DETERI_RISK'],
    c=state_risk_us['TEMP_AVG'],
    cmap='RdYlBu_r', s=100, alpha=0.8,
    edgecolors='black', linewidth=0.5
)
plt.colorbar(sc1, ax=axes[0], label='Mean Annual Temperature (°C)')
for _, row in state_risk_us.iterrows():
    if pd.notna(row['STATE_ABBR']):
        axes[0].annotate(
            row['STATE_ABBR'],
            (row['FREEZE_IDX_AVG'], row['DETERI_RISK']),
            fontsize=7, ha='center', va='bottom'
        )
axes[0].set_xlabel('Mean Freeze Index (°C days)', fontsize=11)
axes[0].set_ylabel('Mean Deterioration Risk Probability', fontsize=11)
axes[0].set_title('Freeze Index vs Deterioration Risk by State', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Precipitation vs Risk
sc2 = axes[1].scatter(
    state_risk_us['PRECIP_AVG'],
    state_risk_us['DETERI_RISK'],
    c=state_risk_us['FREEZE_IDX_AVG'],
    cmap='Blues', s=100, alpha=0.8,
    edgecolors='black', linewidth=0.5
)
plt.colorbar(sc2, ax=axes[1], label='Freeze Index (°C days)')
for _, row in state_risk_us.iterrows():
    if pd.notna(row['STATE_ABBR']):
        axes[1].annotate(
            row['STATE_ABBR'],
            (row['PRECIP_AVG'], row['DETERI_RISK']),
            fontsize=7, ha='center', va='bottom'
        )
axes[1].set_xlabel('Mean Annual Precipitation (mm)', fontsize=11)
axes[1].set_ylabel('Mean Deterioration Risk Probability', fontsize=11)
axes[1].set_title('Precipitation vs Deterioration Risk by State', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    'Climate-Deterioration Risk Relationships (LTPP National Dataset)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('Climate_Risk_Scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: Climate_Risk_Scatter.png')
