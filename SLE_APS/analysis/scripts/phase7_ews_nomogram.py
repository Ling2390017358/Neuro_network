#!/usr/bin/env python3
"""
Phase 7: Clinical Scorecard and Comprehensive Report
====================================================
Covers T7.1–T7.4:
  T7.1: Nomogram construction
  T7.2: EWS v2.0 scorecard optimization
  T7.3: Manuscript figures compilation
  T7.4: Supplementary materials

Outputs:
  - analysis/output/nomogram_model.pkl
  - analysis/output/ews_v2_scorecard.csv
  - analysis/figures/main/Figure_Nomogram.pdf
  - analysis/figures/main/Figure_EWS_Performance.pdf
  - analysis/tables/main/Nomogram_Points.csv
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE = Path('/home/ubuntu/projects/SLE_APS')
OUT = BASE / 'analysis' / 'output'
FIG = BASE / 'analysis' / 'figures'
TAB = BASE / 'analysis' / 'tables'
PROC = BASE / 'data' / 'processed'
MODELS_DIR = BASE / 'models'

for d in [OUT, FIG / 'main', TAB / 'main']:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 7: CLINICAL SCORECARD AND COMPREHENSIVE REPORT")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# ══════════════════════════════════════════════════════════════════════
# Load data (from saved Phase 3/4 output)
# ══════════════════════════════════════════════════════════════════════
print("\n[Load] Loading data...")
train_visit = pd.read_csv(PROC / 'train_data.csv', low_memory=False)
test_visit = pd.read_csv(PROC / 'temporal_test_data.csv', low_memory=False)
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')

id_col = '_patient_SN' if '_patient_SN' in train_visit.columns else 'patient_SN'
aps_col = 'APS'

# Load selected features if available
try:
    with open(OUT / 'selected_feature_names.json', 'r') as f:
        feature_names_selected = json.load(f)
except:
    feature_names_selected = []

print(f"  Loaded {len(feature_names_selected)} features")

# ══════════════════════════════════════════════════════════════════════
# T7.1: Nomogram Construction
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T7.1] Nomogram Construction")
print("─" * 50)

# Select ≤8 clinically meaningful features for the nomogram
preferred_vars = [
    '活化部分凝血活酶时间_APTT__静脉血_定量',
    '补体C3_静脉血_定量',
    '补体C4_静脉血_定量',
    '血小板计数_PLT#__静脉血_定量',
    '血红蛋白_Hb__静脉血_定量',
    'C_反应蛋白_CRP__静脉血_定量',
    '肌酐_Crea__静脉血_定量',
]

# Filter to available in training data
avail_vars = [c for c in preferred_vars if c in train_visit.columns]
print(f"  Nomogram variables ({len(avail_vars)}): {avail_vars}")

# Build patient-level features
feat_data = {}
for c in avail_vars:
    feat_data[c] = train_visit.groupby(id_col)[c].median()
nomogram_df = pd.DataFrame(feat_data)
nomogram_df = nomogram_df.reset_index()
nomogram_df = nomogram_df.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')
nomogram_df = nomogram_df.dropna()

X_nomo = nomogram_df[avail_vars].values
y_nomo = nomogram_df[aps_col].values

# Train LR model for nomogram
scaler_nomo = StandardScaler().fit(X_nomo)
X_nomo_s = scaler_nomo.transform(X_nomo)

lr_nomo = LogisticRegression(penalty='l1', solver='saga', C=0.1,
                              class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000)
lr_nomo.fit(X_nomo_s, y_nomo)

nomo_auc = roc_auc_score(y_nomo, lr_nomo.predict_proba(X_nomo_s)[:, 1])
print(f"  Nomogram model AUC (train): {nomo_auc:.4f}")

# Save model
import joblib
joblib.dump({'model': lr_nomo, 'scaler': scaler_nomo, 'features': avail_vars},
            MODELS_DIR / 'nomogram_model.pkl')
print(f"  ➜ Saved: {MODELS_DIR / 'nomogram_model.pkl'}")

# Calculate points for each variable
coefs = lr_nomo.coef_.flatten()
intercept = lr_nomo.intercept_[0]

point_table = []
for i, (var, coef) in enumerate(zip(avail_vars, coefs)):
    # Divide variable into meaningful bins
    var_vals = X_nomo[:, i]
    percentiles = np.percentile(var_vals, [10, 25, 50, 75, 90])
    bins = percentiles

    point_table.append({
        'Variable': var[:30],
        'Coefficient': round(coef, 4),
        'P10_value': round(percentiles[0], 2),
        'P50_value': round(percentiles[2], 2),
        'P90_value': round(percentiles[4], 2),
    })

pt_df = pd.DataFrame(point_table)
pt_df.to_csv(TAB / 'main' / 'Nomogram_Points.csv', index=False)
print(f"  ➜ Saved: {TAB / 'main' / 'Nomogram_Points.csv'}")

# Nomogram visualization
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10 + len(avail_vars) * 1.5)
    ax.axis('off')

    # Title
    ax.text(50, len(avail_vars) * 1.5 + 8, 'APS Risk Prediction Nomogram',
            ha='center', fontsize=14, fontweight='bold')

    # Points axis
    y_start = len(avail_vars) * 1.5 + 5
    ax.text(5, y_start, 'Points', fontsize=10, fontweight='bold')
    for p in range(0, 110, 10):
        ax.text(p * 0.8 + 10, y_start - 0.5, str(p), fontsize=7, ha='center')
    ax.plot([10, 90], [y_start - 0.3, y_start - 0.3], 'k-', lw=1)

    # Each variable
    for i, (var, coef) in enumerate(zip(avail_vars, coefs)):
        y = len(avail_vars) * 1.5 - i * 1.5
        var_short = var[:25] + '..' if len(var) > 25 else var
        ax.text(5, y, var_short, fontsize=8, ha='left')

        # Point scale for this variable
        var_vals = X_nomo[:, i]
        vmin, vmax = np.percentile(var_vals, [5, 95])
        for val_pct in range(0, 101, 10):
            val = vmin + (vmax - vmin) * val_pct / 100
            ax.text(10 + val_pct * 0.8, y - 0.5, f'{val:.1f}', fontsize=6, ha='center', rotation=45)
        ax.plot([10, 90], [y - 0.3, y - 0.3], 'k-', lw=0.5)

    # Total points and risk axes
    y_total = 1
    ax.text(5, y_total, 'Total Points', fontsize=10, fontweight='bold')
    for tp in range(0, 300, 20):
        ax.text(10 + tp * 0.27, y_total - 0.5, str(tp), fontsize=7, ha='center')
    ax.plot([10, 90], [y_total - 0.3, y_total - 0.3], 'k-', lw=1)

    y_risk = y_total - 1.5
    ax.text(5, y_risk, 'Risk', fontsize=10, fontweight='bold')
    risk_labels = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '0.9']
    for i, rl in enumerate(risk_labels):
        ax.text(10 + i * 10, y_risk - 0.5, rl, fontsize=8, ha='center')

    fig.savefig(FIG / 'main' / 'Figure_Nomogram.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Nomogram.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Nomogram plot failed: {e}")

# ── Nomogram calibration ──
try:
    from sklearn.calibration import calibration_curve
    prob_pred = lr_nomo.predict_proba(X_nomo_s)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_nomo, prob_pred, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', lw=2, color='#2166ac')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Proportion')
    ax.set_title(f'Nomogram Calibration (AUC={nomo_auc:.4f})')
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_Nomogram_Calibration.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Nomogram_Calibration.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Nomogram calibration failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T7.2: EWS v2.0 Scorecard Optimization
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T7.2] EWS v2.0 Scorecard Optimization")
print("─" * 50)

# EWS v2.0: Simplified bedside score (8 variables, integer points)
ews_vars = {
    'APTT': ('活化部分凝血活酶时间_APTT__静脉血_定量', [30, 35, 38, 45], [0, 1, 2, 3]),
    'C3': ('补体C3_静脉血_定量', [0.6, 0.8, 1.0, 1.2], [3, 2, 1, 0]),
    'C4': ('补体C4_静脉血_定量', [0.1, 0.15, 0.2, 0.3], [3, 2, 1, 0]),
    'PLT': ('血小板计数_PLT#__静脉血_定量', [100, 150, 200, 300], [2, 1, 0, 0]),
    'Hb': ('血红蛋白_Hb__静脉血_定量', [80, 100, 120, 150], [2, 1, 0, 0]),
    'CRP': ('C_反应蛋白_CRP__静脉血_定量', [5, 10, 20, 40], [0, 1, 2, 3]),
    'LAC': ('狼疮抗凝物标准化比值_LAC_NLR__静脉血_定量', [1.0, 1.2, 1.5, 2.0], [0, 2, 3, 4]),
}

# Test EWS v2.0 on temporal validation set
print("  Evaluating EWS v2.0...")
test_patients_ids = test_visit[id_col].unique()

ews_scores = []
for pid in test_patients_ids:
    pdata = test_visit[test_visit[id_col] == pid]
    total_score = 0
    for short_name, (col, thresholds, points) in ews_vars.items():
        if col in pdata.columns:
            val = pdata[col].median()
            if pd.notna(val):
                for thr, pt in zip(thresholds, points):
                    if short_name in ['C3', 'C4']:
                        # Lower is worse
                        if val <= thr:
                            total_score += pt
                            break
                    else:
                        # Higher is worse
                        if val >= thr:
                            total_score += pt
                            break

    ews_scores.append({id_col: pid, 'ews_score': total_score})

ews_df = pd.DataFrame(ews_scores)
ews_df = ews_df.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')
ews_df = ews_df.dropna(subset=[aps_col])

if len(ews_df) > 0:
    y_ews = ews_df[aps_col].values
    scores = ews_df['ews_score'].values
    ews_auc = roc_auc_score(y_ews, scores) if len(np.unique(y_ews)) > 1 else 0.5
    print(f"  EWS v2.0 AUC: {ews_auc:.4f}")

    # Risk stratification
    ews_df['risk_level'] = pd.cut(ews_df['ews_score'],
                                   bins=[-1, 4, 8, 12, 100],
                                   labels=['Low', 'Moderate', 'High', 'Very High'])

    risk_summary = ews_df.groupby('risk_level', observed=True).agg(
        N=(aps_col, 'count'),
        APS_rate=(aps_col, 'mean')
    ).reset_index()
    print(f"\n  EWS Risk Stratification:")
    for _, r in risk_summary.iterrows():
        print(f"    {r['risk_level']:12s}: n={r['N']:4d}, APS rate={r['APS_rate']*100:.1f}%")

    # Save
    ews_df.to_csv(OUT / 'ews_v2_scores.csv', index=False)
    print(f"  ➜ Saved: {OUT / 'ews_v2_scores.csv'}")

    # EWS v2.0 scorecard
    scorecard_rows = []
    for short_name, (col, thresholds, points) in ews_vars.items():
        for thr, pt in zip(thresholds, points):
            direction = '<' if short_name in ['C3', 'C4'] else '>'
            scorecard_rows.append({
                'Variable': short_name,
                'Threshold': f'{direction}{thr}',
                'Points': pt,
            })

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(OUT / 'ews_v2_scorecard.csv', index=False)
    print(f"  ➜ Saved: {OUT / 'ews_v2_scorecard.csv'}")

# EWS performance comparison
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_ews, scores)
    ax.plot(fpr, tpr, lw=2, label=f'EWS v2.0 (AUC={ews_auc:.4f})', color='#d6604d')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('EWS v2.0 Performance')
    ax.legend()
    fig.savefig(FIG / 'main' / 'Figure_EWS_Performance.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_EWS_Performance.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ EWS plot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T7.3: Compile manuscript figures
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T7.3] Manuscript Figures - checking completeness")
print("─" * 50)

paper1_figures = [
    'flow_chart.pdf',
    'Figure_APL_Spectrum.pdf',
    'Table1_SMD_LovePlot.pdf',
    'Figure_ForestPlot.pdf',
    'Figure_VolcanoPlot.pdf',
    'Figure_SHAP_Summary.pdf',
    'Figure_ROC_AllModels.pdf',
    'Figure_Calibration.pdf',
    'Figure_DCA.pdf',
    'Figure_Nomogram.pdf',
    'Figure_Incremental_AUC.pdf',
    'Figure_Nomogram_Calibration.pdf',
]

paper2_figures = [
    'Figure_Trajectory_Panel.pdf',
    'Figure_KM_APL_Stratified.pdf',
    'Figure_Attention_Heatmap.pdf',
    'Figure_EWS_Performance.pdf',
]

print("  Paper 1 figures:")
for f in paper1_figures:
    exists = (FIG / 'main' / f).exists()
    print(f"    {'✅' if exists else '❌'} {f}")

print("  Paper 2 figures:")
for f in paper2_figures:
    exists = (FIG / 'main' / f).exists()
    print(f"    {'✅' if exists else '❌'} {f}")

# ══════════════════════════════════════════════════════════════════════
# T7.4: Supplementary Materials
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T7.4] Supplementary Materials")
print("─" * 50)

supplementary_files = [
    (FIG / 'supplementary' / 'missing_heatmap.png', 'Missing data heatmap'),
    (FIG / 'supplementary' / 'APL_coverage_trend.png', 'APL coverage trend'),
    (FIG / 'supplementary' / 'Feature_Selection_Venn.png', 'Feature selection Venn'),
    (FIG / 'supplementary' / 'nCV_boxplot.png', 'Nested CV AUC distribution'),
    (FIG / 'supplementary' / 'SHAP_Dependence_Top6.pdf', 'SHAP dependence plots'),
    (FIG / 'supplementary' / 'DL_Training_Curves.png', 'DL training curves'),
]

for fpath, desc in supplementary_files:
    exists = fpath.exists()
    print(f"    {'✅' if exists else '❌'} {desc:40s} ({fpath.name})")

# Generate summary table of outputs
print(f"\n{'═' * 60}")
print("PHASE 7 COMPLETE")
print(f"{'═' * 60}")
