#!/usr/bin/env python3
"""
Phase 2: Baseline Characteristics and Univariate Analysis
==========================================================
Covers T2.1–T2.3:
  T2.1: Standardized Table 1 (APS vs non-APS, with SMD)
  T2.2: APL antibody deep profile analysis
  T2.3: Univariate analysis (forest plot, volcano plot)

Outputs:
  - analysis/output/Table1.csv / Table1_formatted.csv
  - analysis/output/univariate_results.csv
  - analysis/figures/main/Table1_SMD_LovePlot.png
  - analysis/figures/main/Figure_ForestPlot.pdf
  - analysis/figures/main/Figure_VolcanoPlot.pdf
  - analysis/figures/main/Figure_APL_Spectrum.pdf
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path('/home/ubuntu/projects/SLE_APS')
OUT = BASE / 'analysis' / 'output'
FIG = BASE / 'analysis' / 'figures'
TAB = BASE / 'analysis' / 'tables'
RAW = BASE / 'data' / 'raw' / 'SLEmatrix_merged.csv'

for d in [OUT, FIG / 'main', TAB / 'main', FIG / 'supplementary']:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 2: BASELINE CHARACTERISTICS AND UNIVARIATE ANALYSIS")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# Load processed patient-level data
# ══════════════════════════════════════════════════════════════════════
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')
df_raw = pd.read_csv(RAW, encoding='utf-8-sig', low_memory=False)

id_col = '_patient_SN' if '_patient_SN' in df_raw.columns else 'patient_SN'
visit_date_col = '_visit_date' if '_visit_date' in df_raw.columns else 'visit_date'
df_raw[visit_date_col] = pd.to_datetime(df_raw[visit_date_col], errors='coerce')

print(f"\nLoaded patient-level data: {patient_df.shape[0]:,} patients, {patient_df.shape[1]} columns")

# APS label
aps_col = 'APS'  # Definition A (2023 ACR/EULAR)
print(f"APS+ (Def A): {patient_df[aps_col].sum():,} ({patient_df[aps_col].mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════
# T2.1: Standardized Table 1
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T2.1] Standardized Table 1 Generation")
print("─" * 50)

# Define key variables for Table 1
# Demographic and clinical variables
key_vars_raw = {
    'Demographics': {
        '定量': ['BMI', '年龄', '身高', '体重'],
    },
    'Disease Activity': {
        '定量': ['SLEDAI'],
    },
    'Coagulation': {
        '定量': ['APTT', 'PT', 'TT', 'PTA', 'Fbg', 'PLT', 'INR'],
    },
    'Complement': {
        '定量': ['C3', 'C4'],
    },
    'Renal': {
        '定量': ['Crea', 'UA', 'BUN', '尿蛋白', 'eGFR'],
    },
    'Inflammation': {
        '定量': ['CRP', 'ESR', 'Hb', 'WBC', 'RBC'],
    },
}

# Map from raw column names to simplified keys
def find_cols(keywords, df):
    found = {}
    for kw in keywords:
        matches = [c for c in df.columns if kw in c]
        for m in matches:
            if '定量' in m:
                found[kw] = m
                break
        if kw not in found:
            # try without 定量 filter
            for m in matches:
                if kw not in found:
                    found[kw] = m
    return found

# For visit-level data, aggregate to patient level (median)
all_found_cols = {}
patient_data_for_table = pd.DataFrame({id_col: patient_df[id_col], 'APS': patient_df[aps_col]})

for category, var_types in key_vars_raw.items():
    for vtype, varlist in var_types.items():
        col_map = find_cols(varlist, df_raw)
        for simple_name, raw_col in col_map.items():
            if raw_col in df_raw.columns and df_raw[raw_col].dtype in ['float64', 'int64']:
                # Aggregate to patient median
                med_vals = df_raw.groupby(id_col)[raw_col].median()
                patient_data_for_table[f"{simple_name}"] = patient_data_for_table[id_col].map(med_vals)
                all_found_cols[f"{simple_name}"] = raw_col

print(f"  Found {len(all_found_cols)} variables for Table 1:")
for name, raw in sorted(all_found_cols.items()):
    print(f"    {name:15s} ← {raw}")

# Compute Table 1 statistics
table1_rows = []
for var_name in sorted(all_found_cols.keys()):
    if var_name not in patient_data_for_table.columns:
        continue
    data = patient_data_for_table[var_name].dropna()
    aps_data = patient_data_for_table.loc[patient_data_for_table['APS'] == 1, var_name].dropna()
    non_aps_data = patient_data_for_table.loc[patient_data_for_table['APS'] == 0, var_name].dropna()

    if len(data) == 0:
        continue

    # Check normality (heuristic: if median close to mean)
    is_normal = abs(data.mean() - data.median()) / data.std() < 0.1 if data.std() > 0 else False

    if is_normal:
        overall = f"{data.mean():.2f} ± {data.std():.2f}"
        aps_str = f"{aps_data.mean():.2f} ± {aps_data.std():.2f}" if len(aps_data) > 0 else "N/A"
        non_aps_str = f"{non_aps_data.mean():.2f} ± {non_aps_data.std():.2f}" if len(non_aps_data) > 0 else "N/A"
    else:
        overall = f"{data.median():.2f} ({data.quantile(0.25):.2f}-{data.quantile(0.75):.2f})"
        aps_str = f"{aps_data.median():.2f} ({aps_data.quantile(0.25):.2f}-{aps_data.quantile(0.75):.2f})" if len(aps_data) > 0 else "N/A"
        non_aps_str = f"{non_aps_data.median():.2f} ({non_aps_data.quantile(0.25):.2f}-{non_aps_data.quantile(0.75):.2f})" if len(non_aps_data) > 0 else "N/A"

    # Statistical test: Mann-Whitney U
    from scipy.stats import mannwhitneyu
    if len(aps_data) > 0 and len(non_aps_data) > 0:
        stat, p_val = mannwhitneyu(aps_data, non_aps_data, alternative='two-sided')
    else:
        p_val = np.nan

    # SMD (Cohen's d)
    if len(aps_data) > 1 and len(non_aps_data) > 1:
        pooled_std = np.sqrt(
            ((len(aps_data) - 1) * aps_data.std()**2 +
             (len(non_aps_data) - 1) * non_aps_data.std()**2) /
            (len(aps_data) + len(non_aps_data) - 2)
        )
        smd = (aps_data.mean() - non_aps_data.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        smd = np.nan

    table1_rows.append({
        'Variable': var_name,
        'Overall': overall,
        'APS+': aps_str,
        'APS-': non_aps_str,
        'p_value': f"{p_val:.2e}" if not np.isnan(p_val) else "N/A",
        'SMD': f"{smd:.3f}" if not np.isnan(smd) else "N/A",
        'N_APS+': len(aps_data),
        'N_APS-': len(non_aps_data),
    })

table1_df = pd.DataFrame(table1_rows)
table1_df.to_csv(TAB / 'main' / 'Table1.csv', index=False)
print(f"  ➜ Saved: {TAB / 'main' / 'Table1.csv'}")

# Format for Word (clean version)
table1_clean = table1_df[['Variable', 'APS+', 'APS-', 'p_value', 'SMD']].copy()
table1_clean.to_csv(TAB / 'main' / 'Table1_formatted.csv', index=False)
print(f"  ➜ Saved: {TAB / 'main' / 'Table1_formatted.csv'}")

# ── SMD Love Plot ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    table1_plot = table1_df.copy()
    table1_plot['SMD_num'] = pd.to_numeric(table1_plot['SMD'], errors='coerce')
    table1_plot = table1_plot.dropna(subset=['SMD_num'])
    table1_plot = table1_plot.sort_values('SMD_num')

    fig, ax = plt.subplots(figsize=(8, max(5, len(table1_plot) * 0.3)))
    colors = ['#d6604d' if abs(s) > 0.1 else '#4393c3' for s in table1_plot['SMD_num']]
    ax.barh(range(len(table1_plot)), table1_plot['SMD_num'], color=colors, alpha=0.7)
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='SMD=0.1 (threshold)')
    ax.axvline(-0.1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_yticks(range(len(table1_plot)))
    ax.set_yticklabels(table1_plot['Variable'], fontsize=9)
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('SMD Love Plot: APS+ vs APS-')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Table1_SMD_LovePlot.png', dpi=200)
    fig.savefig(FIG / 'main' / 'Table1_SMD_LovePlot.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Table1_SMD_LovePlot.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Love plot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T2.2: APL Antibody Deep Profile Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T2.2] APL Antibody Deep Profile Analysis")
print("─" * 50)

# APL quantitative columns
apl_cols = {
    'ACL_IgG': '抗心磷脂抗体IgG_ACL_IgG__静脉血_定量',
    'ACL_IgM': '抗心磷脂抗体IgM_ACL_IgM__静脉血_定量',
    'ACL_IgA': '抗心磷脂抗体IgA_ACL_IgA__静脉血_定量',
    'B2GP1_IgG': '抗β2糖蛋白1抗体IgG_β2_GP1_IgG__静脉血_定量',
    'B2GP1_IgM': '抗β2糖蛋白1抗体IgM_β2_GP1_IgM__静脉血_定量',
    'LAC': '狼疮抗凝物标准化比值_LAC_NLR__静脉血_定量',
}

# Thresholds for positivity
thresholds = {
    'ACL_IgG': (12, 20, 40),
    'ACL_IgM': (12, 20, 40),
    'ACL_IgA': (12, 20, 40),
    'B2GP1_IgG': (20, 40, 80),
    'B2GP1_IgM': (20, 40, 80),
    'LAC': (1.2, 1.5, 2.0),
}

# Per-patient APL positivity at multiple thresholds
apl_results = []
for name, col in apl_cols.items():
    if col not in df_raw.columns:
        print(f"  ⚠ Column not found: {col}")
        continue
    t1, t2, t3 = thresholds[name]
    # Per-patient: any visit exceeding threshold
    patient_max = df_raw.groupby(id_col)[col].max()
    n_tested = patient_max.notna().sum()
    n_pos_t1 = (patient_max > t1).sum()
    n_pos_t2 = (patient_max > t2).sum()
    n_pos_t3 = (patient_max > t3).sum()
    pct_tested = n_tested / len(patient_max) * 100

    # APS+ vs APS- positivity
    aps_patients = set(patient_df[patient_df[aps_col] == 1][id_col])
    non_aps = set(patient_df[patient_df[aps_col] == 0][id_col])

    apl_results.append({
        'Marker': name,
        'Tested_n': n_tested,
        'Tested_%': pct_tested,
        f'Pos_>{t1}': f"{n_pos_t1} ({n_pos_t1/max(n_tested,1)*100:.1f}%)",
        f'Pos_>{t2}': f"{n_pos_t2} ({n_pos_t2/max(n_tested,1)*100:.1f}%)",
        f'Pos_>{t3}': f"{n_pos_t3} ({n_pos_t3/max(n_tested,1)*100:.1f}%)",
        'Median_titer': f"{patient_max.median():.2f}" if n_tested > 0 else "N/A",
    })

apl_table = pd.DataFrame(apl_results)
apl_table.to_csv(TAB / 'main' / 'APL_Antibody_Profile.csv', index=False)
print(f"  ➜ Saved: {TAB / 'main' / 'APL_Antibody_Profile.csv'}")
for _, r in apl_table.iterrows():
    print(f"  {r['Marker']:10s} tested={r['Tested_n']:5d} ({r['Tested_%']:4.1f}%)")

# ── APL Spectrum Figure ──
try:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (name, col) in enumerate(apl_cols.items()):
        if col not in df_raw.columns or idx >= len(axes):
            continue
        ax = axes[idx]
        vals = df_raw[col].dropna()
        vals = vals[vals < vals.quantile(0.99)]  # clip for visualization
        ax.hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(thresholds[name][0], color='red', linestyle='--', alpha=0.7, label=f'cutoff={thresholds[name][0]}')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Distribution')
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_APL_Spectrum.pdf')
    fig.savefig(FIG / 'main' / 'Figure_APL_Spectrum.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_APL_Spectrum.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ APL spectrum plot failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T2.3: Univariate Analysis and Effect Size Visualization
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T2.3] Univariate Analysis")
print("─" * 50)

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression

# Identify numeric candidate features
id_set = set(patient_df[id_col])
candidate_cols = []
for c in df_raw.columns:
    if df_raw[c].dtype not in ['float64', 'int64']:
        continue
    if c in [id_col, visit_date_col]:
        continue
    if df_raw[c].isna().mean() > 0.9:  # skip >90% missing
        continue
    candidate_cols.append(c)

print(f"  Testing {len(candidate_cols)} candidate features...")

univariate_results = []
for c in candidate_cols:
    aggr = df_raw.groupby(id_col)[c].median()
    df_merge = pd.DataFrame({id_col: patient_df[id_col], 'APS': patient_df[aps_col]})
    df_merge[c] = df_merge[id_col].map(aggr)
    df_merge = df_merge.dropna(subset=[c])

    if len(df_merge) < 50:
        continue

    aps_vals = df_merge.loc[df_merge['APS'] == 1, c]
    non_aps_vals = df_merge.loc[df_merge['APS'] == 0, c]

    if len(aps_vals) < 5 or len(non_aps_vals) < 5:
        continue

    # Mann-Whitney U test
    stat, p_val = mannwhitneyu(aps_vals, non_aps_vals, alternative='two-sided')

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        ((len(aps_vals) - 1) * aps_vals.std()**2 +
         (len(non_aps_vals) - 1) * non_aps_vals.std()**2) /
        max(len(aps_vals) + len(non_aps_vals) - 2, 1)
    )
    cohens_d = (aps_vals.mean() - non_aps_vals.mean()) / pooled_std if pooled_std > 0 else 0

    # Percentage difference
    pct_diff = (aps_vals.median() - non_aps_vals.median()) / max(abs(non_aps_vals.median()), 0.001) * 100

    univariate_results.append({
        'Feature': c,
        'N_total': len(df_merge),
        'N_APS+': len(aps_vals),
        'N_APS-': len(non_aps_vals),
        'Median_APS+': f"{aps_vals.median():.4f}",
        'Median_APS-': f"{non_aps_vals.median():.4f}",
        'Mean_APS+': f"{aps_vals.mean():.4f}",
        'Mean_APS-': f"{non_aps_vals.mean():.4f}",
        'Cohen_d': f"{cohens_d:.4f}",
        'Pct_diff': f"{pct_diff:.2f}",
        'p_value': p_val,
        'log10_p': -np.log10(max(p_val, 1e-300)),
    })

uni_df = pd.DataFrame(univariate_results)

# FDR correction
from scipy.stats import false_discovery_control as fdr_bh

if len(uni_df) > 0:
    p_adj = fdr_bh(uni_df['p_value'].values)
    uni_df['p_adj'] = p_adj
    uni_df['significant'] = p_adj < 0.05
    n_sig = uni_df['significant'].sum()
    print(f"  FDR significant: {n_sig}/{len(uni_df)} (p_adj < 0.05)")
else:
    uni_df['p_adj'] = np.nan
    uni_df['significant'] = False
    n_sig = 0

uni_df.to_csv(OUT / 'univariate_results.csv', index=False)
print(f"  ➜ Saved: {OUT / 'univariate_results.csv'}")

# ── Volcano Plot ──
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    cohen_d_vals = pd.to_numeric(uni_df['Cohen_d'], errors='coerce')
    log10p = uni_df['log10_p'].values
    sig = uni_df['significant'].values

    ax.scatter(cohen_d_vals[~sig], log10p[~sig], c='gray', alpha=0.4, s=20, label='Not significant')
    ax.scatter(cohen_d_vals[sig], log10p[sig], c='#d6604d', alpha=0.7, s=30, label=f'FDR significant (n={n_sig})')

    # Label top features
    top_idx = uni_df.nlargest(10, 'log10_p').index
    for i in top_idx:
        ax.annotate(uni_df.loc[i, 'Feature'][:20], (cohen_d_vals[i], log10p[i]),
                   fontsize=7, alpha=0.8, ha='center')

    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_ylabel('−log₁₀(p-value)')
    ax.set_title('Univariate Analysis: APS+ vs APS-')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_VolcanoPlot.pdf')
    fig.savefig(FIG / 'main' / 'Figure_VolcanoPlot.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_VolcanoPlot.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Volcano plot failed: {e}")

# ── Forest Plot (Top features by clinical domain) ──
try:
    fig, ax = plt.subplots(figsize=(10, max(6, n_sig * 0.25)))
    plot_df = uni_df[uni_df['significant']].copy()
    if len(plot_df) > 30:
        plot_df = plot_df.nlargest(30, 'log10_p')

    plot_df = plot_df.sort_values('log10_p')
    y_pos = range(len(plot_df))

    ax.barh(y_pos, plot_df['log10_p'].values, color='steelblue', alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([str(n)[:30] for n in plot_df['Feature']], fontsize=8)
    ax.set_xlabel('−log₁₀(p-value)')
    ax.set_title('Top Significant Features (FDR corrected)')
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_ForestPlot.pdf')
    fig.savefig(FIG / 'main' / 'Figure_ForestPlot.png', dpi=200)
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_ForestPlot.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Forest plot failed: {e}")

# ── Print top results ──
print(f"\n  ── Top 15 Features by −log₁₀(p) ──")
top15 = uni_df.nlargest(15, 'log10_p')
for _, r in top15.iterrows():
    marker = '✅' if r['significant'] else '❌'
    print(f"  {marker} {r['Feature'][:40]:40s} d={float(r['Cohen_d']):6.4f} "
          f"p={r['p_value']:.2e} p_adj={r['p_adj']:.2e}")

# ── Save univariate summary ──
summary = {
    'total_features_tested': len(uni_df),
    'fdr_significant': n_sig,
    'proportion_significant': f"{n_sig/max(len(uni_df),1)*100:.1f}%",
    'aps_positive': int(patient_df[aps_col].sum()),
    'aps_negative': len(patient_df) - int(patient_df[aps_col].sum()),
    'total_patients': len(patient_df),
}
pd.Series(summary).to_csv(OUT / 'univariate_summary.csv')
print(f"  ➜ Saved: {OUT / 'univariate_summary.csv'}")

print(f"\n{'═' * 60}")
print("PHASE 2 COMPLETE")
print(f"{'═' * 60}")
