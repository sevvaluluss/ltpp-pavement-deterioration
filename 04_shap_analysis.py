"""
LTPP Pavement Deterioration Under Climate Variability
Script 4: SHAP Explainability Analysis
Author: Metehan Alp Memis
Journal: Transportation Geotechnics (TRGEO-D-26-01076)

Requirements:
    pip install shap pandas numpy matplotlib joblib scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================

df_ml = pd.read_csv('LTPP_ml_dataset.csv')

FEATURES = [
    'TEMP_AVG', 'FREEZE_INDEX', 'FREEZE_THAW',
    'PRECIPITATION', 'PRECIP_DAYS', 'EVAPORATION',
    'AADTT', 'ANNUAL_TRUCK_VOL', 'PAVEMENT_AGE', 'YEAR'
]

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X = df_ml[FEATURES]
y = df_ml['WILL_DETERIORATE']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = joblib.load('imputer.pkl')
X_test_imp = imputer.transform(X_test)

# Load best model: Random Forest
best_model = joblib.load('model_Random_Forest.pkl')

# ============================================================
# 2. COMPUTE SHAP VALUES
# ============================================================

print('Computing SHAP values...')
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_imp)

# Class 1 = Deteriorate
shap_class1 = shap_values[:, :, 1]

# ============================================================
# 3. GLOBAL FEATURE IMPORTANCE (BAR PLOT)
# ============================================================

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_class1, X_test_imp,
    feature_names=FEATURES,
    plot_type='bar',
    show=False
)
plt.title(
    'Feature Importance — Pavement Deterioration Risk\n(SHAP Values)',
    fontsize=13
)
plt.tight_layout()
plt.savefig('SHAP_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: SHAP_importance.png')

# ============================================================
# 4. BEESWARM PLOT
# ============================================================

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_class1, X_test_imp,
    feature_names=FEATURES,
    show=False
)
plt.title(
    'SHAP Summary Plot — Deterioration Risk Factors',
    fontsize=13
)
plt.tight_layout()
plt.savefig('SHAP_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: SHAP_beeswarm.png')

# ============================================================
# 5. MEAN ABSOLUTE SHAP VALUES TABLE
# ============================================================

mean_shap = pd.DataFrame({
    'Feature': FEATURES,
    'Mean_SHAP': np.abs(shap_class1).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

print('\nGlobal Feature Importance (Mean |SHAP|):')
print(mean_shap.to_string(index=False))
mean_shap.to_csv('SHAP_importance_table.csv', index=False)
print('Saved: SHAP_importance_table.csv')
