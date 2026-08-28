"""
LTPP Pavement Deterioration Under Climate Variability
Script 3: Machine Learning Pipeline
Author: Metehan Alp Memis
Journal: Transportation Geotechnics (TRGEO-D-26-01076)

Requirements:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix,
    root_mean_squared_error
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================

df_clean = pd.read_csv('LTPP_merged_clean.csv')
print(f'Loaded: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} cols')

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
df_clean['IRI_AVG'].hist(bins=50, color='steelblue', ax=axes[0])
axes[0].set_title('IRI Distribution')
axes[0].set_xlabel('IRI (m/km)')

df_clean['FREEZE_THAW'].hist(bins=50, color='coral', ax=axes[1])
axes[1].set_title('Freeze-Thaw Cycles')

df_clean['PRECIPITATION'].hist(bins=50, color='green', ax=axes[2])
axes[2].set_title('Annual Precipitation (mm)')

plt.tight_layout()
plt.savefig('EDA_distributions.png', dpi=150)
plt.close()
print('Saved: EDA_distributions.png')

# ============================================================
# 3. TARGET VARIABLE: 5-YEAR DETERIORATION LABEL
# ============================================================

# Sort by section and year
df_sorted = df_clean.sort_values(['STATE_CODE', 'SHRP_ID', 'YEAR']).copy()

# Compute pavement age (years since first observation per section)
df_sorted['IRI_BASELINE'] = df_sorted.groupby(
    ['STATE_CODE', 'SHRP_ID']
)['IRI_AVG'].transform('first')

df_sorted['PAVEMENT_AGE'] = df_sorted.groupby(
    ['STATE_CODE', 'SHRP_ID']
)['YEAR'].transform(lambda x: x - x.min())

# 5-year IRI change
def rolling_iri_change(group):
    group = group.sort_values('YEAR')
    group['IRI_5YR_CHANGE'] = group['IRI_AVG'].shift(-5) - group['IRI_AVG']
    return group

df_5yr = df_sorted.groupby(
    ['STATE_CODE', 'SHRP_ID'], group_keys=False
).apply(rolling_iri_change, include_groups=False)

df_5yr = df_5yr.dropna(subset=['IRI_5YR_CHANGE', 'TEMP_AVG'])
df_5yr = df_5yr[df_5yr['IRI_5YR_CHANGE'] >= 0]  # Remove rehabilitation events

# Binary classification target
# 1 = Deteriorating: IRI increases >= 0.5 m/km OR future IRI >= 2.0 m/km
CRITICAL_IRI = 2.0
df_5yr['WILL_DETERIORATE'] = (
    (df_5yr['IRI_5YR_CHANGE'] >= 0.5) |
    (df_5yr['IRI_AVG'] + df_5yr['IRI_5YR_CHANGE'] >= CRITICAL_IRI)
).astype(int)

print(f'\n5-year dataset: {df_5yr.shape[0]:,} rows')
print(f'Class distribution:\n{df_5yr["WILL_DETERIORATE"].value_counts(normalize=True).round(3)}')

# Outlier removal (99th percentile IRI)
iri_99 = df_5yr['IRI_AVG'].quantile(0.99)
df_ml = df_5yr[df_5yr['IRI_AVG'] <= iri_99].copy()
print(f'After outlier removal: {df_ml.shape[0]:,} rows')

# ============================================================
# 4. FEATURE SELECTION AND TRAIN/TEST SPLIT
# ============================================================

FEATURES = [
    'TEMP_AVG', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION',
    'AADTT', 'ANNUAL_TRUCK_VOL', 'PAVEMENT_AGE', 'YEAR'
]

X = df_ml[FEATURES]
y = df_ml['WILL_DETERIORATE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

print(f'\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}')

# ============================================================
# 5. MODEL TRAINING AND EVALUATION
# ============================================================

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, random_state=42,
        n_jobs=-1, eval_metric='logloss'
    ),
}

print('\n=== CLASSIFICATION RESULTS ===\n')
results = {}
for name, model in models.items():
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)
    y_prob = model.predict_proba(X_test_imp)[:, 1]
    acc  = (y_pred == y_test).mean()
    auc  = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob
    }
    print(f'{name}:')
    print(f'  Accuracy = {acc:.3f} | AUC-ROC = {auc:.3f}')
    print(classification_report(
        y_test, y_pred,
        target_names=['Stable', 'Deteriorate']
    ))

# ============================================================
# 6. ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['steelblue', 'coral', 'green']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    auc = roc_auc_score(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(
    'ROC Curves — Pavement Deterioration Risk Prediction\n'
    '(LTPP National Dataset, 2,574 Sections)',
    fontsize=12
)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ROC_curves.png')

# Save imputer and models for later use
import joblib
joblib.dump(imputer, 'imputer.pkl')
for name, res in results.items():
    safe_name = name.replace(' ', '_')
    joblib.dump(res['model'], f'model_{safe_name}.pkl')
    print(f'Saved: model_{safe_name}.pkl')

# Save processed dataset
df_ml.to_csv('LTPP_ml_dataset.csv', index=False)
print('Saved: LTPP_ml_dataset.csv')
