#!/usr/bin/env python3
"""
Phase 1: APS Phenotype Definition and Cohort Construction
==========================================================
Covers T1.1–T1.3:
  T1.1: 2023 ACR/EULAR APS criteria implementation (definitions A/B/C)
  T1.2: Cohort flow chart (CONSORT-style)
  T1.3: Time-based split (train 2008-2020 / temporal validation 2021-2025)

Outputs:
  - analysis/output/aps_classification_detailed.csv
  - analysis/output/patient_level_data.csv (definitive cohort with APS labels)
  - analysis/output/cohort_flow_counts.csv
  - data/processed/train_data.csv
  - data/processed/temporal_test_data.csv
  - analysis/figures/main/flow_chart.png (CONSORT-style)
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
RAW = BASE / 'data' / 'raw' / 'SLEmatrix_merged.csv'
OUT = BASE / 'analysis' / 'output'
PROC = BASE / 'data' / 'processed'
FIG = BASE / 'analysis' / 'figures'

for d in [OUT, PROC, FIG / 'main']:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 1: APS PHENOTYPE DEFINITION AND COHORT CONSTRUCTION")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv(RAW, encoding='utf-8-sig', low_memory=False)
print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Identify key columns
id_col = '_patient_SN' if '_patient_SN' in df.columns else 'patient_SN'
visit_date_col = '_visit_date' if '_visit_date' in df.columns else 'visit_date'
df[visit_date_col] = pd.to_datetime(df[visit_date_col], errors='coerce')

# Identify APS-relevant columns
# APL antibody quantitative columns
apl_keywords = ['ACL', 'B2GP', 'LAC', '抗心磷脂', '抗β2', '狼疮抗凝']
apl_cols = [c for c in df.columns if any(k in c.upper() for k in apl_keywords)]
apl_quant = [c for c in apl_cols if '定量' in c]
print(f"APL quantitative columns ({len(apl_quant)}):")
for c in apl_quant:
    print(f"  {c}")

# ══════════════════════════════════════════════════════════════════════
# T1.1: 2023 ACR/EULAR APS Criteria Implementation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T1.1] 2023 ACR/EULAR APS Criteria Implementation")
print("─" * 50)

# ── Definition C: Any APL positive (broad definition, used previously) ──
# Map columns to standard APL markers
def find_col(keywords, cols):
    """Find column matching any of the keywords."""
    for c in cols:
        for k in keywords:
            if k in c:
                return c
    return None

# ACL columns
acl_igG_col = find_col(['ACL_IgG', 'ACL-IgG', '抗心磷脂抗体IgG', 'ACL IgG'], apl_quant)
acl_igM_col = find_col(['ACL_IgM', 'ACL-IgM', '抗心磷脂抗体IgM', 'ACL IgM'], apl_quant)
acl_igA_col = find_col(['ACL_IgA', 'ACL-IgA', '抗心磷脂抗体IgA', 'ACL IgA'], apl_quant)
b2gp1_igG_col = find_col(['β2_GP1_IgG', 'B2GP1_IgG', '抗β2糖蛋白1抗体IgG', 'B2GP1-IgG'], apl_quant)
b2gp1_igM_col = find_col(['β2_GP1_IgM', 'B2GP1_IgM', '抗β2糖蛋白1抗体IgM', 'B2GP1-IgM'], apl_quant)
lac_col = find_col(['LAC_NLR', '狼疮抗凝物', 'LAC'], apl_quant)

print(f"  ACL-IgG: {acl_igG_col}")
print(f"  ACL-IgM: {acl_igM_col}")
print(f"  ACL-IgA: {acl_igA_col}")
print(f"  B2GP1-IgG: {b2gp1_igG_col}")
print(f"  B2GP1-IgM: {b2gp1_igM_col}")
print(f"  LAC: {lac_col}")

# Per-patient APL positivity using definition C
def patient_has_apl(patient_df):
    """Check if patient has any APL positivity across all visits."""
    result = {}
    markers = {
        'ACL_IgG': acl_igG_col, 'ACL_IgM': acl_igM_col, 'ACL_IgA': acl_igA_col,
        'B2GP1_IgG': b2gp1_igG_col, 'B2GP1_IgM': b2gp1_igM_col, 'LAC': lac_col
    }
    for name, col in markers.items():
        if col and col in patient_df.columns:
            vals = patient_df[col].dropna()
            result[name] = (vals > 12).any()  # typical cutoff
        else:
            result[name] = False
    result['any_APL_positive'] = any(result.values())
    result['has_APL_test'] = any(
        patient_df[col].notna().any()
        for col in markers.values() if col
    )
    return pd.Series(result)

# Compute per-patient APL status
print("  Computing per-patient APL status...")
apl_status = df.groupby(id_col).apply(patient_has_apl).reset_index()
print(f"  Any APL positive (Def C): {apl_status['any_APL_positive'].sum():,} / {len(apl_status):,}")

# ── Definition A: 2023 ACR/EULAR criteria (approximation) ──
# Clinical domain scoring (≥3 points):
#   - Thrombosis (arterial/venous) - need diagnosis column
#   - Pregnancy morbidity
#   - Microvascular involvement
# Lab domain scoring (≥3 points):
#   - LAC positive: 5 points (if high-titer) or 3 points
#   - ACL-IgG/IgM high-titer (>40): 4 points, moderate (20-40): 2 points
#   - B2GP1-IgG/IgM high-titer (>40): 4 points, moderate (20-40): 2 points

# For an approximation based on available lab data:
def compute_2023_score(patient_df):
    """Compute approximate 2023 ACR/EULAR APS lab domain score."""
    score = 0
    details = {}

    # LAC scoring
    if lac_col and lac_col in patient_df.columns:
        lac_vals = patient_df[lac_col].dropna()
        if len(lac_vals) > 0:
            max_lac = lac_vals.max()
            if max_lac > 1.5:  # high positive
                score += 5
                details['LAC'] = f'high_pos({max_lac:.2f})'
            elif max_lac > 1.2:  # moderate
                score += 3
                details['LAC'] = f'mod_pos({max_lac:.2f})'
            else:
                details['LAC'] = 'neg'

    # ACL scoring
    if acl_igG_col and acl_igG_col in patient_df.columns:
        acl_vals = patient_df[acl_igG_col].dropna()
        if len(acl_vals) > 0:
            max_acl = acl_vals.max()
            if max_acl > 40:
                score += 4
                details['ACL_IgG'] = f'high({max_acl:.1f})'
            elif max_acl > 20:
                score += 2
                details['ACL_IgG'] = f'mod({max_acl:.1f})'

    if acl_igM_col and acl_igM_col in patient_df.columns:
        acl_vals = patient_df[acl_igM_col].dropna()
        if len(acl_vals) > 0:
            max_acl = acl_vals.max()
            if max_acl > 40:
                score += 4
                details['ACL_IgM'] = f'high({max_acl:.1f})'
            elif max_acl > 20:
                score += 2
                details['ACL_IgM'] = f'mod({max_acl:.1f})'

    # B2GP1 scoring
    if b2gp1_igG_col and b2gp1_igG_col in patient_df.columns:
        b2gp1_vals = patient_df[b2gp1_igG_col].dropna()
        if len(b2gp1_vals) > 0:
            max_b2 = b2gp1_vals.max()
            if max_b2 > 40:
                score += 4
                details['B2GP1_IgG'] = f'high({max_b2:.1f})'
            elif max_b2 > 20:
                score += 2
                details['B2GP1_IgG'] = f'mod({max_b2:.1f})'

    if b2gp1_igM_col and b2gp1_igM_col in patient_df.columns:
        b2gp1_vals = patient_df[b2gp1_igM_col].dropna()
        if len(b2gp1_vals) > 0:
            max_b2 = b2gp1_vals.max()
            if max_b2 > 40:
                score += 4
                details['B2GP1_IgM'] = f'high({max_b2:.1f})'
            elif max_b2 > 20:
                score += 2
                details['B2GP1_IgM'] = f'mod({max_b2:.1f})'

    # Triple-positive bonus (≥4.5 points automatically)
    pos_count = 0
    for col in [lac_col, acl_igG_col, acl_igM_col, b2gp1_igG_col, b2gp1_igM_col]:
        if col and col in patient_df.columns:
            if (patient_df[col].dropna() > 20).any():
                pos_count += 1
    if pos_count >= 3:
        score = max(score, 5)  # triple positive guarantees high score

    return pd.Series({'lab_score': score, 'lab_details': str(details)})

print("  Computing 2023 ACR/EULAR lab domain scores...")
lab_scores = df.groupby(id_col).apply(compute_2023_score).reset_index()

# Merge APL status and lab scores
patient_df = apl_status.merge(lab_scores, on=id_col)

# Define APS definitions
patient_df['APS_DefC'] = patient_df['any_APL_positive'].astype(int)

# Definition A: lab_score >= 3 (clinical domain needs diagnosis data)
# We approximate: lab_score >= 3 as likely APS
patient_df['APS_DefA'] = (patient_df['lab_score'] >= 3).astype(int)

# Definition B: Sydney 2006 (at least 1 clinical + 1 lab criteria)
# Approximate: any APL positive (same as DefC for lab)
patient_df['APS_DefB'] = patient_df['APS_DefC'].copy()

# Print comparison
print(f"\n  APS Definition Comparison:")
print(f"    Def A (2023 ACR/EULAR, lab≥3): {patient_df['APS_DefA'].sum():,} ({patient_df['APS_DefA'].mean()*100:.1f}%)")
print(f"    Def B (Sydney 2006):            {patient_df['APS_DefB'].sum():,} ({patient_df['APS_DefB'].mean()*100:.1f}%)")
print(f"    Def C (Any APL+):               {patient_df['APS_DefC'].sum():,} ({patient_df['APS_DefC'].mean()*100:.1f}%)")

# Cross-tabulation
print(f"\n  Cross-tabulation (DefA vs DefC):")
ct = pd.crosstab(patient_df['APS_DefA'], patient_df['APS_DefC'],
                 rownames=['DefA'], colnames=['DefC'])
print(ct)

# Save classification details
patient_df.to_csv(OUT / 'aps_classification_detailed.csv', index=False)
print(f"\n  ➜ Saved: {OUT / 'aps_classification_detailed.csv'}")

# ── Use Definition A as primary for downstream ──
# For patients without sufficient APL data, they default to non-APS
# This is a conservative approach
patient_df['APS'] = patient_df['APS_DefA']
print(f"\n  Using Definition A (2023 ACR/EULAR) as primary APS definition")
print(f"  APS patients: {patient_df['APS'].sum():,} / {len(patient_df):,} ({patient_df['APS'].mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════
# T1.2: Cohort Flow Chart (CONSORT-style)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T1.2] Cohort Flow Chart")
print("─" * 50)

flow_steps = []

# Step 1: Total database
total_patients = df[id_col].nunique()
total_visits = len(df)
flow_steps.append(('Total SLE patients in database', total_patients, total_visits))
print(f"  {total_patients:,} patients, {total_visits:,} visits")

# Step 2: ≥2 visits (required for longitudinal analysis)
visit_counts = df.groupby(id_col).size()
multi_visit = visit_counts[visit_counts >= 2].index
n_multi_visit = len(multi_visit)
multi_visit_visits = df[df[id_col].isin(multi_visit)].shape[0]
flow_steps.append(('≥2 visits (longitudinal cohort)', n_multi_visit, multi_visit_visits))

# Step 3: Has APL test data
has_apl = patient_df[patient_df['has_APL_test']]
n_has_apl = has_apl[id_col].nunique()
# count visits for these patients
has_apl_visits = df[df[id_col].isin(has_apl[id_col])].shape[0]
flow_steps.append(('Has ≥1 APL antibody test', n_has_apl, has_apl_visits))

# Step 4: Final analysis cohort (has APL and ≥2 visits)
analysis_patients = set(has_apl[id_col]) & set(multi_visit)
n_analysis = len(analysis_patients)
analysis_visits = df[df[id_col].isin(analysis_patients)].shape[0]
flow_steps.append(('Final analysis cohort', n_analysis, analysis_visits))

# Save flow counts
flow_df = pd.DataFrame(flow_steps, columns=['Step', 'Patients', 'Visits'])
flow_df.to_csv(OUT / 'cohort_flow_counts.csv', index=False)
print(f"  ➜ Saved: {OUT / 'cohort_flow_counts.csv'}")

# Generate CONSORT-style flow chart
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0']
    y_positions = [14, 11, 8, 5]
    labels = [
        f"Total SLE Database\n{n_has_apl + (total_patients - n_has_apl):,} patients\n{total_visits:,} visits",
        f"≥2 Visits\n{n_multi_visit:,} patients\n{multi_visit_visits:,} visits",
        f"Has APL Test Data\n{n_has_apl:,} patients\n{has_apl_visits:,} visits",
        f"Final Analysis Cohort\n{n_analysis:,} patients\n{analysis_visits:,} visits\n"
        f"APS+ = {patient_df[patient_df[id_col].isin(analysis_patients)]['APS'].sum():,} "
        f"({patient_df[patient_df[id_col].isin(analysis_patients)]['APS'].mean()*100:.1f}%)"
    ]

    for i, (y, label, color) in enumerate(zip(y_positions, labels, colors)):
        box = FancyBboxPatch((2, y - 0.8), 6, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='gray', alpha=0.8)
        ax.add_patch(box)
        ax.text(5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrow down
        if i < len(y_positions) - 1:
            ax.annotate('', xy=(5, y_positions[i+1] + 1.0), xytext=(5, y - 0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.set_title('SLE-APS Study Cohort Flow Diagram', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(FIG / 'main' / 'flow_chart.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIG / 'main' / 'flow_chart.pdf', bbox_inches='tight')
    print(f"  ➜ Saved: {FIG / 'main' / 'flow_chart.png/.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Flow chart visual failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T1.3: Time-based Split (2008-2020 Train / 2021-2025 Temporal Validation)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T1.3] Time-based Data Split")
print("─" * 50)

# Determine each patient's first visit date
first_visit = df.groupby(id_col)[visit_date_col].min().reset_index()
first_visit.columns = [id_col, 'first_visit_date']
first_visit['year'] = first_visit['first_visit_date'].dt.year

# Split patients by first visit year
train_patients = first_visit[first_visit['year'] <= 2020][id_col]
test_patients = first_visit[first_visit['year'] >= 2021][id_col]

print(f"  Train patients (first visit ≤2020): {len(train_patients):,}")
print(f"  Temporal test patients (first visit ≥2021): {len(test_patients):,}")

# Create visit-level train/test sets
visit_level = df.copy()
visit_level['is_train'] = visit_level[id_col].isin(train_patients)
visit_level['is_test'] = visit_level[id_col].isin(test_patients)

train_data = visit_level[visit_level['is_train']].drop(columns=['is_train', 'is_test'])
test_data = visit_level[visit_level['is_test']].drop(columns=['is_train', 'is_test'])

print(f"  Train visits: {len(train_data):,}")
print(f"  Test visits: {len(test_data):,}")

# Merge APS labels
aps_labels = patient_df[[id_col, 'APS']].copy()
train_data = train_data.merge(aps_labels, on=id_col, how='left')
test_data = test_data.merge(aps_labels, on=id_col, how='left')

# Save
train_data.to_csv(PROC / 'train_data.csv', index=False)
test_data.to_csv(PROC / 'temporal_test_data.csv', index=False)
print(f"  ➜ Saved: {PROC / 'train_data.csv'} ({len(train_data):,} rows)")
print(f"  ➜ Saved: {PROC / 'temporal_test_data.csv'} ({len(test_data):,} rows)")

# ── Patient-level dataset with APS labels ──
# Start from aps_classification_detailed and add basic demographics
patient_final = patient_df.copy()

# Add demographics from raw data
demo_cols = [c for c in df.columns if any(k in c for k in ['BMI', '性别', '年龄', 'age', 'height', 'weight', '身高', '体重'])]
print(f"\n  Available demographics: {demo_cols[:10]}")

# Add first visit year
patient_final = patient_final.merge(first_visit[[id_col, 'first_visit_date']], on=id_col, how='left')

# Add visit-level aggregates
visit_agg = df.groupby(id_col).agg(
    total_visits=(visit_date_col, 'count'),
    first_visit=(visit_date_col, 'min'),
    last_visit=(visit_date_col, 'max'),
).reset_index()
visit_agg['followup_years'] = (visit_agg['last_visit'] - visit_agg['first_visit']).dt.days / 365.25
patient_final = patient_final.merge(visit_agg, on=id_col, how='left')

print(f"  Patient-level dataset: {len(patient_final):,} × {patient_final.shape[1]}")

# Add SLEDAI if available
sledai_cols = [c for c in df.columns if 'SLEDAI' in c.upper()]
if sledai_cols:
    sledai_mean = df.groupby(id_col)[sledai_cols[0]].mean().reset_index()
    sledai_mean.columns = [id_col, 'mean_SLEDAI']
    patient_final = patient_final.merge(sledai_mean, on=id_col, how='left')

patient_final.to_csv(OUT / 'patient_level_data.csv', index=False)
print(f"  ➜ Saved: {OUT / 'patient_level_data.csv'}")

# ── Train/Validation split summary ──
print(f"\n  ── Cohort Summary ──")
print(f"  Total patients: {len(patient_final):,}")
print(f"  APS+ (Def A): {patient_final['APS'].sum():,} ({patient_final['APS'].mean()*100:.2f}%)")
print(f"  APS+ (Def C): {patient_final['APS_DefC'].sum():,} ({patient_final['APS_DefC'].mean()*100:.2f}%)")
print(f"  Train patients: {train_patients.shape[0]:,}")
print(f"  Test patients: {test_patients.shape[0]:,}")

# Check APS rate in train/test
train_aps = patient_final[patient_final[id_col].isin(train_patients)]['APS'].mean()
test_aps = patient_final[patient_final[id_col].isin(test_patients)]['APS'].mean()
print(f"  APS rate in train: {train_aps*100:.2f}%")
print(f"  APS rate in test: {test_aps*100:.2f}%")

print("\n" + "═" * 60)
print("PHASE 1 COMPLETE")
print("═" * 60)
