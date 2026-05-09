# NPSLE Analysis Reproduction Notes

Data source: `data/SLEmatrix_merged.csv`
Latest completed run: 2026-05-05 06:05:41; runtime 593.9 seconds.
Visits: 79655; patients: 9420
Primary NPSLE label: any positive value in acr_神经_癫痫, acr_神经_精神, acr_神经_脊髓炎, acr_神经_脑血管.
Visit-level events: 907; patient-level events: 617.

The current merged CSV differs from the manuscript's historical Results directory. These outputs are recomputed directly from the merged file and should be treated as the authoritative reproduction for this data source.

Key regenerated outputs:
- Univariate significant features (FDR<0.05): 36
- Best cross-sectional model: GradientBoosting AUC=0.863
- First-visit LASSO CV AUC=0.972+/-0.022
- EWS risk groups written to `ews_risk_stratification.csv`.
