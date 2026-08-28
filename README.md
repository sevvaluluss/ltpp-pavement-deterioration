# LTPP Pavement Deterioration Under Climate Variability

**Paper:** Machine Learning Based Prediction of Pavement Deterioration Risk Under Climate Variability: A National Scale Analysis Using the Long Term Pavement Performance Database

**Journal:** Transportation Geotechnics  
**Manuscript No:** TRGEO-D-26-01076  
**Authors:** Metehan Alp Memis, Sevval Ulus Memis  
**Affiliation:** Southern Illinois University Edwardsville (SIUE)

---

## Repository Structure

```
pavement_deterioration/
├── 01_sql_restore.sql          # SQL Server database restore scripts
├── 02_data_extraction.py       # Data extraction and merging from LTPP database
├── 03_ml_pipeline.py           # ML model training and evaluation
├── 04_shap_analysis.py         # SHAP explainability analysis
├── 05_spatial_analysis.py      # State-level and regional risk mapping
└── README.md                   # This file
```

---

## Data Source

Data retrieved from the **LTPP InfoPave** portal (Standard Data Release 39):
- https://infopave.fhwa.dot.gov
- Bucket #141968: IRI, Traffic, Performance (ARPED + ARTRD)
- Bucket #141969: MERRA Climate (ARCLD)
- Bucket #141971: Materials (ARMAD)

---

## Requirements

```bash
pip install pyodbc sqlalchemy pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly joblib
```

SQL Server Express (free): https://www.microsoft.com/en-us/sql-server/sql-server-downloads  
SSMS: https://aka.ms/ssmsfullsetup

---

## Workflow

1. Download `.bak` files from LTPP InfoPave
2. Run `01_sql_restore.sql` in SSMS to restore databases
3. Run `02_data_extraction.py` to merge all tables → `LTPP_merged_clean.csv`
4. Run `03_ml_pipeline.py` to train models → ROC curves, metrics
5. Run `04_shap_analysis.py` for SHAP explainability → importance plots
6. Run `05_spatial_analysis.py` for state/regional risk maps

---

## Key Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Random Forest | 0.699 | **0.760** |
| XGBoost | 0.694 | 0.750 |
| Gradient Boosting | 0.658 | 0.707 |

- **Freeze Index** is the dominant climatic predictor (SHAP rank #1 climate variable)
- **Dry-Freeze** regions show highest mean deterioration risk (0.454)
- Dataset: 2,574 sections, 27,023 annual observations, 1990-2024

---

## Citation

```
Memis, M.A., Ulus Memis, S. (2026). Machine Learning Based Prediction of 
Pavement Deterioration Risk Under Climate Variability: A National Scale 
Analysis Using the Long Term Pavement Performance Database. 
Transportation Geotechnics. TRGEO-D-26-01076.
```
