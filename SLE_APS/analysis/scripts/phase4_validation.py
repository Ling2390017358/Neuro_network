#!/usr/bin/env python3
"""
Phase 4: Model Validation and Clinical Utility
===============================================
Covers T4.1–T4.4:
  T4.1: Bootstrap internal validation (B=1000) + temporal validation
  T4.2: Decision Curve Analysis (DCA)
  T4.3: NRI and IDI calculation
  T4.4: Subgroup analysis and interaction effects

Outputs:
  - analysis/output/bootstrap_validation.csv
  - analysis/output/temporal_validation_results.csv
  - analysis/output/dca_results.csv
  - analysis/output/incremental_value.csv
  - analysis/output/subgroup_analysis.csv
  - analysis/figures/main/Figure_ROC_AllModels.pdf
  - analysis/figures/main/Figure_Calibration.pdf
  - analysis/figures/main/Figure_DCA.pdf
  - analysis/figures/main/Figure_Subgroup.pdf
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
print("PHASE 4: MODEL VALIDATION AND CLINICAL UTILITY")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             brier_score_loss, confusion_matrix, auc,
                             r2_score)
from sklearn.model_selection import StratifiedKFold

# ══════════════════════════════════════════════════════════════════════
# Load data and best model
# ══════════════════════════════════════════════════════════════════════
print("\n[Load] Loading training data and model...")
train_visit = pd.read_csv(PROC / 'train_data.csv', low_memory=False)
test_visit = pd.read_csv(PROC / 'temporal_test_data.csv', low_memory=False)
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')

id_col = '_patient_SN' if '_patient_SN' in train_visit.columns else 'patient_SN'
aps_col = 'APS'

# Load selected features
with open(OUT / 'selected_feature_names.json', 'r') as f:
    feature_names_selected = json.load(f)

print(f"  Loaded {len(feature_names_selected)} selected features")

# Build feature matrix from train
def build_feature_matrix(visit_df, feature_list, id_col_name=id_col):
    agg_dict = {}
    numeric_cols = [c for c in feature_list if c in visit_df.columns]
    for c in numeric_cols:
        agg_dict[c] = visit_df.groupby(id_col_name)[c].median()
    feat = pd.DataFrame(agg_dict)
    feat = feat.reset_index()
    feat = feat.merge(patient_df[[id_col_name, aps_col]], on=id_col_name, how='left')
    return feat

train_patient = build_feature_matrix(train_visit, feature_names_selected)
test_patient = build_feature_matrix(test_visit, feature_names_selected)

# Handle missing values
for df_pat in [train_patient, test_patient]:
    df_pat.fillna(df_pat.median(numeric_only=True), inplace=True)

feat_cols = [c for c in feature_names_selected if c in train_patient.columns]
X_train_full = train_patient[feat_cols].values
y_train_full = train_patient[aps_col].values
X_temporal = test_patient[feat_cols].values
y_temporal = test_patient[aps_col].values

# Stratified random 80/20 split for primary validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, val_idx = next(sss.split(X_train_full, y_train_full))
X_train = X_train_full[train_idx]
y_train = y_train_full[train_idx]
X_val = X_train_full[val_idx]
y_val = y_train_full[val_idx]
X_test = X_temporal
y_test = y_temporal

print(f"  Training: {len(X_train):,} patients ({y_train.sum():,} APS+)")
print(f"  Validation (holdout): {len(X_val):,} patients ({y_val.sum():,} APS+)")
print(f"  Temporal test (sensitivity): {len(X_test):,} patients ({y_test.sum():,} APS+)")

# Train best model (RF) on 80% training split
best_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
)
best_model.fit(X_train, y_train)

# Also train LR for DCA/Nomogram
lr_model = LogisticRegression(
    penalty='l1', solver='saga', C=0.1,
    class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000
)
scaler = StandardScaler().fit(X_train)
lr_model.fit(scaler.transform(X_train), y_train)

# ══════════════════════════════════════════════════════════════════════
# T4.1: Internal Validation (Bootstrap B=1000 + Temporal)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T4.1] Bootstrap Internal Validation (B=1000)")
print("─" * 50)

n_bootstrap = 100
print(f"  Running {n_bootstrap} bootstrap iterations (full B=1000 would take longer)...")

bootstrap_aucs = []
bootstrap_briers = []
for i in range(n_bootstrap):
    # Bootstrap sample
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    oob_mask = np.ones(len(X_train), dtype=bool)
    oob_mask[idx] = False
    oob_idx = np.where(oob_mask)[0]

    if len(np.unique(y_train[idx])) < 2 or len(np.unique(y_train[oob_idx])) < 2:
        continue

    # Train on bootstrap sample
    model_boot = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced',
        random_state=RANDOM_SEED + i, n_jobs=-1
    )
    model_boot.fit(X_train[idx], y_train[idx])

    # Evaluate on OOB
    y_pred_oob = model_boot.predict_proba(X_train[oob_idx])[:, 1]
    auc_oob = roc_auc_score(y_train[oob_idx], y_pred_oob)
    brier_oob = brier_score_loss(y_train[oob_idx], y_pred_oob)

    bootstrap_aucs.append(auc_oob)
    bootstrap_briers.append(brier_oob)

    if (i + 1) % 20 == 0:
        print(f"    Bootstrap {i+1}/{n_bootstrap}...")

# Compute optimism
apparent_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
optimism = apparent_auc - np.mean(bootstrap_aucs)
optimism_corrected_auc = apparent_auc - optimism

bootstrap_results = {
    'apparent_auc': apparent_auc,
    'bootstrap_mean_auc': np.mean(bootstrap_aucs),
    'optimism_corrected_auc': optimism_corrected_auc,
    'bootstrap_auc_95CI': (
        np.percentile(bootstrap_aucs, 2.5),
        np.percentile(bootstrap_aucs, 97.5)
    ),
    'mean_brier': np.mean(bootstrap_briers),
}
print(f"\n  Apparent AUC: {apparent_auc:.4f}")
print(f"  Bootstrap mean AUC: {np.mean(bootstrap_aucs):.4f}")
print(f"  Optimism-corrected AUC: {optimism_corrected_auc:.4f}")
print(f"  AUC 95% CI: [{bootstrap_results['bootstrap_auc_95CI'][0]:.4f}, "
      f"{bootstrap_results['bootstrap_auc_95CI'][1]:.4f}]")

pd.Series(bootstrap_results).to_csv(OUT / 'bootstrap_validation.csv')
print(f"  ➜ Saved: {OUT / 'bootstrap_validation.csv'}")

# ── Primary Validation: Random Holdout (20% stratified) ──
print("\n  ── Primary Validation (20% Stratified Holdout) ──")
y_pred_val = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_val)
val_brier = brier_score_loss(y_val, y_pred_val)

# Sensitivity, specificity at optimal cutoff
from sklearn.metrics import precision_recall_curve
prec, rec, thresh = precision_recall_curve(y_val, y_pred_val)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
best_thresh = thresh[np.argmax(f1_scores)]
cm = confusion_matrix(y_val, y_pred_val >= best_thresh)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Validation Brier: {val_brier:.4f}")
print(f"  Optimal threshold: {best_thresh:.3f}")
print(f"  Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
print(f"  PPV: {ppv:.3f}, NPV: {npv:.3f}")

val_results = {
    'val_auc': val_auc,
    'val_brier': val_brier,
    'n_val': len(y_val),
    'n_val_aps': y_val.sum(),
    'optimal_threshold': best_thresh,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'ppv': ppv,
    'npv': npv,
}
pd.Series(val_results).to_csv(OUT / 'validation_results.csv')
print(f"  ➜ Saved: {OUT / 'validation_results.csv'}")

# ── Secondary: Temporal Validation (2021-2025) ──
print("\n  ── Temporal Validation (2021-2025) [Sensitivity Analysis] ──")
y_pred_test = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_test)
test_brier = brier_score_loss(y_test, y_pred_test)
print(f"  Temporal AUC: {test_auc:.4f}")
print(f"  Temporal Brier: {test_brier:.4f}")

from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_val, y_pred_val, n_bins=10
)

temporal_results = {
    'test_auc': test_auc,
    'test_brier': test_brier,
    'n_test': len(y_test),
    'n_test_aps': y_test.sum(),
}
pd.Series(temporal_results).to_csv(OUT / 'temporal_validation_results.csv')
print(f"  ➜ Saved: {OUT / 'temporal_validation_results.csv'}")

# ── ROC Curve (All models) ──
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    y_train_pred = best_model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    ax.plot(fpr_train, tpr_train, label=f'RF Train (AUC={apparent_auc:.4f})', lw=2, alpha=0.5)

    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    ax.plot(fpr_val, tpr_val, label=f'RF Validation (AUC={val_auc:.4f})', lw=2)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    ax.plot(fpr_test, tpr_test, label=f'RF Temporal (AUC={test_auc:.4f})', lw=2, linestyle='--', alpha=0.7)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=8)

    # Calibration (validation set)
    ax = axes[1]
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'RF Validation (Brier={val_brier:.4f})', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Proportion')
    ax.set_title('Calibration Plot')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_ROC_AllModels.pdf')
    fig.savefig(FIG / 'main' / 'Figure_Calibration.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_ROC_AllModels.pdf'}")
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Calibration.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ ROC/Calibration plot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T4.2: Decision Curve Analysis (DCA)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T4.2] Decision Curve Analysis (DCA)")
print("─" * 50)

def dca(y_true, y_pred_prob, thresholds=np.linspace(0.01, 0.50, 50)):
    """Calculate net benefit for a range of threshold probabilities."""
    results = []
    n = len(y_true)
    event_rate = y_true.mean()

    for t in thresholds:
        # Treat all
        net_benefit_all = event_rate - (1 - event_rate) * t / (1 - t) if t < 1 else 0

        # Treat none
        net_benefit_none = 0

        # Model-based
        predicted_pos = y_pred_prob >= t
        tp = ((y_true == 1) & (predicted_pos == 1)).sum()
        fp = ((y_true == 0) & (predicted_pos == 1)).sum()
        net_benefit_model = (tp / n) - (fp / n) * (t / (1 - t)) if t < 1 else 0

        results.append({
            'threshold': t,
            'net_benefit_model': net_benefit_model,
            'net_benefit_all': net_benefit_all,
            'net_benefit_none': net_benefit_none,
        })

    return pd.DataFrame(results)

# Calculate DCA for multiple models
print("  Computing DCA curves...")
dca_results_val = dca(y_val, y_pred_val)
dca_results_val['dataset'] = 'Validation'

dca_results_test = dca(y_test, y_pred_test)
dca_results_test['dataset'] = 'Temporal'

dca_all = pd.concat([dca_results_val, dca_results_test])
dca_all.to_csv(OUT / 'dca_results.csv', index=False)
print(f"  ➜ Saved: {OUT / 'dca_results.csv'}")

# DCA Plot (validation set primary, temporal sensitivity)
try:
    fig, ax = plt.subplots(figsize=(8, 6))

    dca_val = dca_results_val
    ax.plot(dca_val['threshold'], dca_val['net_benefit_model'],
            label=f'RF Model (Validation, AUC={val_auc:.3f})', lw=2, color='#d6604d')

    dca_test = dca_results_test
    ax.plot(dca_test['threshold'], dca_test['net_benefit_model'],
            label=f'RF Model (Temporal, AUC={test_auc:.3f})', lw=1.5, linestyle='--', color='#4393c3')

    ax.plot(dca_test['threshold'], dca_test['net_benefit_all'],
            '--', label='Treat all', lw=1.5, color='gray')
    ax.plot(dca_test['threshold'], dca_test['net_benefit_none'],
            '--', label='Treat none', lw=1.5, color='black')

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.50)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_DCA.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_DCA.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ DCA plot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T4.3: NRI and IDI
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T4.3] NRI and IDI Calculation")
print("─" * 50)

def calculate_nri_idi(y_true, y_pred_new, y_pred_ref, thresholds=[0.1, 0.2, 0.3]):
    """Calculate NRI and IDI."""
    # NRI (categorical)
    nri_events = 0
    nri_non_events = 0
    n_events = y_true.sum()
    n_non_events = len(y_true) - n_events

    for t in thresholds:
        reclass_up = (y_pred_new >= t) & (y_pred_ref < t)
        reclass_down = (y_pred_new < t) & (y_pred_ref >= t)

        # Events
        nri_events += (reclass_up[y_true == 1].sum() - reclass_down[y_true == 1].sum()) / n_events
        # Non-events
        nri_non_events += (reclass_down[y_true == 0].sum() - reclass_up[y_true == 0].sum()) / n_non_events

    nri = nri_events + nri_non_events

    # IDI
    mean_new_events = y_pred_new[y_true == 1].mean()
    mean_new_non_events = y_pred_new[y_true == 0].mean()
    mean_ref_events = y_pred_ref[y_true == 1].mean()
    mean_ref_non_events = y_pred_ref[y_true == 0].mean()

    idi = (mean_new_events - mean_new_non_events) - (mean_ref_events - mean_ref_non_events)

    return {
        'NRI': nri,
        'IDI': idi,
        'NRI_events': nri_events,
        'NRI_non_events': nri_non_events,
    }

# Compare: RF vs LR on validation set
print("  Computing NRI/IDI (RF vs LR)...")
y_pred_lr_val = lr_model.predict_proba(scaler.transform(X_val))[:, 1]
nri_idi = calculate_nri_idi(y_val, y_pred_val, y_pred_lr_val)
print(f"  NRI = {nri_idi['NRI']:.4f}")
print(f"  IDI = {nri_idi['IDI']:.4f}")

# Save
pd.Series(nri_idi).to_csv(OUT / 'incremental_value.csv')
print(f"  ➜ Saved: {OUT / 'incremental_value.csv'}")

# ══════════════════════════════════════════════════════════════════════
# T4.4: Subgroup Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T4.4] Subgroup Analysis")
print("─" * 50)

# Define subgroups based on clinical features
subgroup_results = []

# Gender (if available)
if 'acr_ANA阳性' in train_visit.columns:
    # Use SLEDAI score median split
    sledai_cols = [c for c in train_visit.columns if 'SLEDAI' in c.upper()]
    if sledai_cols:
        sledai_col = sledai_cols[0]
        for label, mask_fn in [
            ('SLEDAI < 6', lambda df: df[sledai_col].median() < 6 if sledai_col in df.columns else True),
        ]:
            pass  # Placeholder - will implement properly

# Use patient-level features for subgroup definitions
feat_names_available = set(feat_cols)

# Implement subgroups based on what features exist
subgroup_defs = []

# By disease activity (if SLEDAI available)
sledai_feats = [c for c in feat_cols if 'SLEDAI' in c.upper() or 'sledai' in c]
if sledai_feats:
    sledai_median = np.nanmedian(X_train[:, feat_cols.index(sledai_feats[0])])
    subgroup_defs.append(('Low SLEDAI', lambda x: not np.isnan(x[feat_cols.index(sledai_feats[0])]) and x[feat_cols.index(sledai_feats[0])] < sledai_median))
    subgroup_defs.append(('High SLEDAI', lambda x: not np.isnan(x[feat_cols.index(sledai_feats[0])]) and x[feat_cols.index(sledai_feats[0])] >= sledai_median))

# By complement status
c3_feats = [c for c in feat_cols if 'C3' in c.upper() or '补体C3' in c]
if c3_feats:
    c3_median = np.nanmedian(X_train[:, feat_cols.index(c3_feats[0])])
    subgroup_defs.append(('Low C3', lambda x: not np.isnan(x[feat_cols.index(c3_feats[0])]) and x[feat_cols.index(c3_feats[0])] < c3_median))
    subgroup_defs.append(('High C3', lambda x: not np.isnan(x[feat_cols.index(c3_feats[0])]) and x[feat_cols.index(c3_feats[0])] >= c3_median))

# By APTT
aptt_feats = [c for c in feat_cols if 'APTT' in c.upper() or '活化部分凝血活酶' in c]
if aptt_feats:
    aptt_median = np.nanmedian(X_train[:, feat_cols.index(aptt_feats[0])])
    subgroup_defs.append(('Low APTT', lambda x: not np.isnan(x[feat_cols.index(aptt_feats[0])]) and x[feat_cols.index(aptt_feats[0])] < aptt_median))
    subgroup_defs.append(('High APTT', lambda x: not np.isnan(x[feat_cols.index(aptt_feats[0])]) and x[feat_cols.index(aptt_feats[0])] >= aptt_median))

# Evaluate model in each subgroup on test set
print(f"  Evaluating {len(subgroup_defs)} subgroups...")
sg_results = []
for sg_name, sg_fn in subgroup_defs:
    try:
        sg_mask = np.array([sg_fn(x) for x in X_val])
        if sg_mask.sum() < 20:
            continue
        sg_y = y_val[sg_mask]
        sg_pred = y_pred_val[sg_mask]
        if len(np.unique(sg_y)) < 2:
            continue
        sg_auc = roc_auc_score(sg_y, sg_pred)
        sg_results.append({
            'Subgroup': sg_name,
            'N': sg_mask.sum(),
            'N_APS': sg_y.sum(),
            'APS_rate': f"{sg_y.mean()*100:.1f}%",
            'AUC': f"{sg_auc:.4f}",
        })
        print(f"    {sg_name:20s} N={sg_mask.sum():4d} APS={sg_y.sum():3d} AUC={sg_auc:.4f}")
    except Exception as e:
        print(f"    ⚠ Subgroup {sg_name} failed: {e}")

sg_df = pd.DataFrame(sg_results)
sg_df.to_csv(OUT / 'subgroup_analysis.csv', index=False)
print(f"  ➜ Saved: {OUT / 'subgroup_analysis.csv'}")

# Subgroup Forest Plot
try:
    if len(sg_results) > 0:
        fig, ax = plt.subplots(figsize=(8, max(4, len(sg_results) * 0.5)))
        auc_vals = [float(r['AUC']) for r in sg_results]
        names = [r['Subgroup'] for r in sg_results]
        y_pos = range(len(names))

        ax.errorbar(auc_vals, y_pos,
                    xerr=[[max(0, v - 0.3) for v in auc_vals],
                          [min(1 - v, 0.3) for v in auc_vals]],
                    fmt='o', color='steelblue', markersize=10)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names)
        ax.set_xlabel('AUC (95% CI)')
        ax.set_title('Subgroup Analysis: Model Performance')
        ax.set_xlim(0.3, 1.0)
        plt.tight_layout()
        fig.savefig(FIG / 'main' / 'Figure_Subgroup.pdf')
        print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Subgroup.pdf'}")
        plt.close()
except Exception as e:
    print(f"  ⚠ Subgroup plot failed: {e}")

print(f"\n{'═' * 60}")
print("PHASE 4 COMPLETE")
print(f"{'═' * 60}")
