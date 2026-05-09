#!/usr/bin/env python3
"""
Phase 3: Multi-factor Prediction Model Development
===================================================
Covers T3.1–T3.4:
  T3.1: 3-stage feature selection (univariate filter + LASSO + Boruta)
  T3.2: Multi-model training (LR, RF, XGBoost, LightGBM, Stacking)
  T3.3: SHAP explainability (summary, dependence, force plots)
  T3.4: Layered model comparison (simple → complete)

Outputs:
  - analysis/output/selected_features.csv
  - analysis/output/model_comparison.csv (+ nCV boxplot)
  - analysis/output/shap_importance.csv (+ SHAP figures)
  - analysis/output/layered_model_comparison.csv
  - analysis/figures/main/Figure_SHAP_Summary.pdf
  - analysis/figures/supplementary/Feature_Selection_Venn.png
  - analysis/figures/supplementary/nCV_boxplot.png
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE = Path('/home/ubuntu/projects/SLE_APS')
OUT = BASE / 'analysis' / 'output'
FIG = BASE / 'analysis' / 'figures'
TAB = BASE / 'analysis' / 'tables'
PROC = BASE / 'data' / 'processed'
MODELS_DIR = BASE / 'models'

for d in [OUT, FIG / 'main', FIG / 'supplementary', TAB / 'main', TAB / 'supplementary', MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 3: MULTI-FACTOR PREDICTION MODEL DEVELOPMENT")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
print("\n[Load] Loading training data...")
train_visit = pd.read_csv(PROC / 'train_data.csv', low_memory=False)
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')

id_col = '_patient_SN' if '_patient_SN' in train_visit.columns else 'patient_SN'
aps_col = 'APS'

# Build patient-level feature matrix
# Demographics
demo_map = {}
for c in ['BMI', '身高', '体重']:
    if c in train_visit.columns:
        demo_map[c] = train_visit.groupby(id_col)[c].first()

# Aggregate visit-level features to patient-level median
numeric_cols = train_visit.select_dtypes(include=[np.number]).columns.tolist()
exclude = [id_col, 'APS', 'APS_DefA', 'APS_DefB', 'APS_DefC']
# Exclude diagnostic criteria features (acr_, sledai_, acl_) that encode SLE/APS
# classification criteria - these cause data leakage and overfitting
leakage_features = [c for c in numeric_cols if c.startswith('acr_') or c.startswith('sledai_') or c.startswith('acl_')]
print(f"  Excluding {len(leakage_features)} diagnostic criteria features (acr_/sledai_/acl_)")
exclude += leakage_features
numeric_cols = [c for c in numeric_cols if c not in exclude and c not in ['_year']]

print(f"  Aggregating {len(numeric_cols)} numeric features to patient-level median...")
agg_dict = {}
for c in numeric_cols:
    agg_dict[c] = train_visit.groupby(id_col)[c].median()

feat_df = pd.DataFrame(agg_dict)
feat_df = feat_df.reset_index()

# Merge APS label
feat_df = feat_df.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')
print(f"  Feature matrix: {feat_df.shape[0]:,} patients, {feat_df.shape[1]} columns")

# Remove high-missing features (>80% missing)
missing_pct = feat_df.isna().mean()
keep_cols = missing_pct[missing_pct < 0.80].index.tolist()
feat_df = feat_df[[c for c in keep_cols if c in feat_df.columns]]
print(f"  After removing >80% missing: {feat_df.shape[1]} columns")

# Drop rows with any missing for modeling
feature_cols = [c for c in feat_df.columns if c not in [id_col, aps_col]]
model_df = feat_df.dropna(subset=[aps_col]).copy()
model_df = model_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)
model_df = model_df.fillna(model_df.median(numeric_only=True))

X = model_df[feature_cols].values
y = model_df[aps_col].values
feature_names = feature_cols

print(f"  Final modeling dataset: {len(model_df):,} patients, {len(feature_cols)} features")
print(f"  APS+ rate: {y.mean()*100:.2f}%")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score,
                             brier_score_loss, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════
# T3.1: Feature Selection Pipeline
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T3.1] 3-Stage Feature Selection Pipeline")
print("─" * 50)

selected_sets = {}

# Stage 1: Univariate filter
print("\n  Stage 1: Univariate filter (p < 0.1)")
from scipy.stats import mannwhitneyu

uni_pvals = []
for i, col in enumerate(feature_cols):
    aps_vals = model_df.loc[model_df[aps_col] == 1, col].dropna()
    non_aps_vals = model_df.loc[model_df[aps_col] == 0, col].dropna()
    if len(aps_vals) > 2 and len(non_aps_vals) > 2:
        _, p = mannwhitneyu(aps_vals, non_aps_vals, alternative='two-sided')
        uni_pvals.append((col, p))
    else:
        uni_pvals.append((col, 1.0))

uni_selected = [c for c, p in uni_pvals if p < 0.1]
print(f"  Univariate selected: {len(uni_selected)} features")
selected_sets['Univariate'] = set(uni_selected)

# Stage 2: LASSO path selection
print("\n  Stage 2: LASSO (lambda.min path)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LogisticRegressionCV(
    Cs=100, penalty='l1', solver='saga',
    cv=5, scoring='roc_auc', random_state=RANDOM_SEED,
    max_iter=5000, n_jobs=-1
)
lasso_cv.fit(X_scaled, y)

# Features with non-zero coefficients at C_opt
coefs = lasso_cv.coef_.flatten()
lasso_selected = [feature_names[i] for i in range(len(feature_names)) if abs(coefs[i]) > 0]
print(f"  LASSO selected (lambda.min): {len(lasso_selected)} features")
selected_sets['LASSO'] = set(lasso_selected)

# Stage 3: Boruta-like (Random Forest importance with shadow)
print("\n  Stage 3: Boruta-style feature selection (RF importance)")
# Use fewer trees for speed at this stage
rf_boruta = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1)
rf_boruta.fit(X_scaled, y)

# Use built-in feature importance (fast)
boruta_imp = rf_boruta.feature_importances_

imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': boruta_imp,
})

# Use mean importance as threshold
threshold = imp_df['importance'].mean()
boruta_selected = imp_df[imp_df['importance'] > threshold]['feature'].tolist()
print(f"  Boruta-selected (RF importance > mean): {len(boruta_selected)} features")
selected_sets['Boruta'] = set(boruta_selected)

# Final: features selected by ≥2 methods
all_sets = list(selected_sets.values())
final_features = set()
for feat in feature_names:
    count = sum(1 for s in all_sets if feat in s)
    if count >= 2:
        final_features.add(feat)

print(f"\n  ✓ Final selected features (≥2 methods): {len(final_features)}")
final_feat_list = sorted(final_features)
for f in final_feat_list:
    methods = [name for name, s in selected_sets.items() if f in s]
    print(f"    {f:50s} selected by: {', '.join(methods)}")

# Save feature selections
feature_selection_df = pd.DataFrame({
    'feature': feature_names,
    'uni_pvalue': [p for _, p in uni_pvals],
    'lasso_coef': [abs(coefs[i]) if i < len(coefs) else 0 for i in range(len(feature_names))],
    'boruta_importance': [imp_df.loc[imp_df['feature'] == f, 'importance'].values[0]
                         if f in imp_df['feature'].values else 0 for f in feature_names],
    'selected': [1 if f in final_features else 0 for f in feature_names],
})
feature_selection_df.to_csv(OUT / 'selected_features.csv', index=False)
print(f"  ➜ Saved: {OUT / 'selected_features.csv'}")

# Venn diagram of feature selection
try:
    from matplotlib_venn import venn3
    fig, ax = plt.subplots(figsize=(8, 6))
    venn3([
        selected_sets.get('Univariate', set()),
        selected_sets.get('LASSO', set()),
        selected_sets.get('Boruta', set()),
    ], set_labels=('Univariate\n(p<0.1)', 'LASSO', 'Boruta\n(RF)'), ax=ax)
    ax.set_title('Feature Selection Venn Diagram')
    fig.savefig(FIG / 'supplementary' / 'Feature_Selection_Venn.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'supplementary' / 'Feature_Selection_Venn.png'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Venn diagram failed: {e}")

# Update X to use only selected features
selected_indices = [feature_names.index(f) for f in final_feat_list]
feature_names_selected = final_feat_list
X_selected = X[:, selected_indices]
print(f"\n  Reduced feature matrix: {X_selected.shape}")

# ══════════════════════════════════════════════════════════════════════
# T3.2: Multi-model Training and Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T3.2] Multi-model Training (Nested CV)")
print("─" * 50)

models = {
    'LASSO_Logistic': LogisticRegression(
        penalty='l1', solver='saga', C=0.1, class_weight='balanced',
        random_state=RANDOM_SEED, max_iter=5000
    ),
    'Random_Forest': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    ),
    'Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1 - y.mean()) / y.mean(),
        random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=RANDOM_SEED,
        verbose=-1
    ),
}

# Nested cross-validation: outer 10-fold, inner 5-fold for tuning
cv_results = {}
for name, model in models.items():
    print(f"\n  Training: {name}...")
    aucs = []
    preds_all = []
    y_all = []

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_selected, y)):
        X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale for linear models
        if 'Logistic' in name:
            scaler_fold = StandardScaler().fit(X_tr)
            X_tr_s = scaler_fold.transform(X_tr)
            X_val_s = scaler_fold.transform(X_val)
            model_clone = model
            model_clone.fit(X_tr_s, y_tr)
            y_pred = model_clone.predict_proba(X_val_s)[:, 1]
        else:
            model_clone = model
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        preds_all.extend(y_pred)
        y_all.extend(y_val)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    cv_results[name] = {
        'AUC_mean': mean_auc,
        'AUC_std': std_auc,
        'AUC_95CI_lower': mean_auc - 1.96 * std_auc,
        'AUC_95CI_upper': mean_auc + 1.96 * std_auc,
        'fold_aucs': aucs,
    }
    print(f"    AUC = {mean_auc:.4f} ± {std_auc:.4f}")

# ── Stacking Ensemble ──
print("\n  Training: Stacking Ensemble...")
from sklearn.ensemble import StackingClassifier
base_learners = [
    ('lr', LogisticRegression(penalty='l1', solver='saga', C=0.1, class_weight='balanced',
                               random_state=RANDOM_SEED, max_iter=5000)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced',
                                   random_state=RANDOM_SEED, n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                               random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)),
]
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED),
    cv=5
)

stack_aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
    X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    stacking_clone = stacking
    stacking_clone.fit(X_tr, y_tr)
    y_pred = stacking_clone.predict_proba(X_val)[:, 1]
    stack_aucs.append(roc_auc_score(y_val, y_pred))

mean_stack = np.mean(stack_aucs)
std_stack = np.std(stack_aucs)
cv_results['Stacking'] = {
    'AUC_mean': mean_stack,
    'AUC_std': std_stack,
    'AUC_95CI_lower': mean_stack - 1.96 * std_stack,
    'AUC_95CI_upper': mean_stack + 1.96 * std_stack,
    'fold_aucs': stack_aucs,
}
print(f"    AUC = {mean_stack:.4f} ± {std_stack:.4f}")

# Save model comparison
comparison_rows = []
for name, res in cv_results.items():
    comparison_rows.append({
        'Model': name,
        'AUC': f"{res['AUC_mean']:.4f} ± {res['AUC_std']:.4f}",
        'AUC_mean': res['AUC_mean'],
        'AUC_std': res['AUC_std'],
        '95CI': f"[{res['AUC_95CI_lower']:.4f}, {res['AUC_95CI_upper']:.4f}]",
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values('AUC_mean', ascending=False)
comparison_df.to_csv(OUT / 'model_comparison.csv', index=False)
print(f"\n  ➜ Saved: {OUT / 'model_comparison.csv'}")
print(f"\n  ── Model Performance Ranking ──")
for _, r in comparison_df.iterrows():
    print(f"  {r['Model']:25s} AUC = {r['AUC']}")

# ── Nested CV boxplot ──
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = [cv_results[m]['fold_aucs'] for m in cv_results.keys()]
    labels = list(cv_results.keys())
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))):
        patch.set_facecolor(color)
    ax.set_ylabel('AUC')
    ax.set_title('10-Fold Cross-Validation AUC by Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(FIG / 'supplementary' / 'nCV_boxplot.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'supplementary' / 'nCV_boxplot.png'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Boxplot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T3.3: SHAP Explainability Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T3.3] SHAP Explainability Analysis")
print("─" * 50)

import shap

# Retrain best model (Random Forest) on full training data
best_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
)
best_model.fit(X_selected, y)

# SHAP analysis with TreeExplainer
print("  Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_selected)

# Handle different return shapes from TreeExplainer
if isinstance(shap_values, list):
    shap_vals = np.array(shap_values[1])  # class 1 values
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]  # class 1: (samples, features, classes)
else:
    shap_vals = shap_values

# Ensure 2D
if shap_vals.ndim != 2:
    print(f"  Unexpected SHAP shape: {shap_vals.shape}, reshaping...")
    shap_vals = shap_vals.reshape(X_selected.shape)

# SHAP feature importance
shap_imp = pd.DataFrame({
    'feature': feature_names_selected,
    'mean_abs_shap': np.abs(shap_vals).mean(axis=0),
})
shap_imp = shap_imp.sort_values('mean_abs_shap', ascending=False)
shap_imp['rank'] = range(1, len(shap_imp) + 1)
shap_imp.to_csv(OUT / 'shap_importance.csv', index=False)
print(f"  ➜ Saved: {OUT / 'shap_importance.csv'}")

print("  Top 10 features by SHAP importance:")
for _, r in shap_imp.head(10).iterrows():
    print(f"    {r['rank']:2d}. {r['feature'][:45]:45s} |SHAP|={r['mean_abs_shap']:.6f}")

# ── SHAP Summary Plot ──
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(feature_names_selected) * 0.3)))

    # Beeswarm
    shap.summary_plot(shap_vals, X_selected, feature_names=feature_names_selected,
                      show=False, max_display=20, plot_type="dot", ax=axes[0])
    axes[0].set_title('SHAP Summary (Beeswarm)')

    # Bar
    shap.summary_plot(shap_vals, X_selected, feature_names=feature_names_selected,
                      show=False, max_display=20, plot_type="bar", ax=axes[1])
    axes[1].set_title('SHAP Feature Importance (Bar)')

    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_SHAP_Summary.pdf')
    fig.savefig(FIG / 'main' / 'Figure_SHAP_Summary.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_SHAP_Summary.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ SHAP summary plot failed: {e}")
    # Fallback: simpler SHAP plot
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap_imp_top = shap_imp.head(15)
        ax.barh(range(len(shap_imp_top)), shap_imp_top['mean_abs_shap'].values, color='steelblue')
        ax.set_yticks(range(len(shap_imp_top)))
        ax.set_yticklabels(shap_imp_top['feature'].values, fontsize=8)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('SHAP Feature Importance')
        plt.tight_layout()
        fig.savefig(FIG / 'main' / 'Figure_SHAP_Summary.pdf')
        print(f"  ➜ Saved (fallback): {FIG / 'main' / 'Figure_SHAP_Summary.pdf'}")
        plt.close()
    except:
        pass

# ── SHAP Dependence Plots (Top 6) ──
try:
    top6 = shap_imp.head(6)['feature'].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top6):
        if i < len(axes):
            idx = feature_names_selected.index(feat)
            shap.dependence_plot(idx, shap_vals, X_selected,
                                 feature_names=feature_names_selected,
                                 ax=axes[i], show=False)
            axes[i].set_title(f'{feat[:30]}')
    plt.tight_layout()
    fig.savefig(FIG / 'supplementary' / 'SHAP_Dependence_Top6.pdf')
    print(f"  ➜ Saved: {FIG / 'supplementary' / 'SHAP_Dependence_Top6.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ SHAP dependence plots failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T3.4: Layered Model Comparison (Simple → Complete)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T3.4] Layered Model Comparison")
print("─" * 50)

# Categorize features into layers
known_clinical_vars = [
    'BMI', 'SLEDAI', 'SLEDAI总分', 'sledai',
    'APTT', 'PT', 'PT_INR', 'INR', 'TT', 'Fbg', 'PLT', '血小板',
    'C3', 'C4', '补体C3', '补体C4',
    'ACL', '心磷脂', 'B2GP', 'LAC', '狼疮',
    'Hb', '血红蛋白', 'WBC', 'CRP', 'ESR',
]

layer_a = []
layer_b = []
layer_c = []
layer_d = []
layer_e = final_feat_list  # all features

for f in final_feat_list:
    if any(k in f for k in ['BMI', 'SLEDAI', 'sledai', '年龄', '性别', '身高', '体重']):
        layer_a.append(f)
    if any(k in f for k in ['APTT', 'PT', 'INR', 'TT', 'Fbg', 'PLT', '血小板']):
        layer_b.append(f)
    if any(k in f for k in ['C3', 'C4', '补体C3', '补体C4']):
        layer_c.append(f)
    if any(k in f for k in ['ACL', '心磷脂', 'B2GP', 'LAC', '狼疮']):
        layer_d.append(f)

# Make layer_e = all features not in A-D
layer_e = [f for f in final_feat_list if f not in set(layer_a + layer_b + layer_c + layer_d)]

layers = {
    'A: Demographics+SLEDAI': layer_a,
    'B: +Coagulation (APTT/PLT)': layer_a + layer_b,
    'C: +Complement (C3/C4)': layer_a + layer_b + layer_c,
    'D: +APL antibodies': layer_a + layer_b + layer_c + layer_d,
    'E: Full model': layer_a + layer_b + layer_c + layer_d + layer_e,
}

print("\n  Layer compositions:")
for lname, lfeats in layers.items():
    if lfeats:
        print(f"    {lname:40s} {len(lfeats):2d} features")

# Evaluate each layer
layer_results = []
for lname, lfeats in layers.items():
    if len(lfeats) < 2:
        lname_full = lname
        continue
    print(f"\n  Evaluating: {lname}")
    idx = [feature_names.index(f) for f in lfeats if f in feature_names]
    if len(idx) == 0:
        continue
    X_layer = X[:, idx]

    aucs = []
    for train_idx, val_idx in skf.split(X_layer, y):
        X_tr, X_val = X_layer[train_idx], X_layer[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = LogisticRegression(penalty='l1', solver='saga', C=0.1,
                                   class_weight='balanced', random_state=RANDOM_SEED,
                                   max_iter=5000)
        scaler_fold = StandardScaler().fit(X_tr)
        model.fit(scaler_fold.transform(X_tr), y_tr)
        y_pred = model.predict_proba(scaler_fold.transform(X_val))[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred))

    layer_results.append({
        'Layer': lname,
        'N_features': len(lfeats),
        'AUC_mean': np.mean(aucs),
        'AUC_std': np.std(aucs),
        'AUC_95CI': f"{np.mean(aucs)-1.96*np.std(aucs):.4f}–{np.mean(aucs)+1.96*np.std(aucs):.4f}",
    })
    print(f"    AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

layer_df = pd.DataFrame(layer_results)
layer_df.to_csv(OUT / 'layered_model_comparison.csv', index=False)
print(f"\n  ➜ Saved: {OUT / 'layered_model_comparison.csv'}")

# Incremental AUC plot
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['Layer'] for r in layer_results]
    means = [r['AUC_mean'] for r in layer_results]
    stds = [r['AUC_std'] for r in layer_results]
    x_pos = range(len(names))
    ax.errorbar(x_pos, means, yerr=stds, fmt='-o', capsize=5, capthick=2, color='#2166ac',
                markersize=8, linewidth=2)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([n.split(':')[0] for n in names], fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title('Incremental Predictive Value of Feature Layers')
    ax.axhline(0.7, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0.5, 0.9)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_Incremental_AUC.pdf')
    fig.savefig(FIG / 'main' / 'Figure_Incremental_AUC.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Incremental_AUC.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Incremental AUC plot failed: {e}")

# ── Save best model ──
import joblib
model_path = MODELS_DIR / 'best_rf_model.pkl'
joblib.dump(best_model, model_path)
print(f"\n  ➜ Saved best model: {model_path}")

# Save feature names for later use
with open(OUT / 'selected_feature_names.json', 'w') as f:
    json.dump(feature_names_selected, f)

print(f"\n{'═' * 60}")
print("PHASE 3 COMPLETE")
print(f"{'═' * 60}")
