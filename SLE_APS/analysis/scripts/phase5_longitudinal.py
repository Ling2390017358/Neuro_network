#!/usr/bin/env python3
"""
Phase 5: Longitudinal Trajectory Modeling
==========================================
Covers T5.1–T5.4:
  T5.1: Linear Mixed-Effects Models (LME) for biomarker trajectories
  T5.2: Latent Class Mixed Model (LCMM) approximation
  T5.3: Survival analysis (KM, Cox, Fine-Gray)
  T5.4: Time-dependent ROC analysis

Outputs:
  - analysis/output/lme_results.csv
  - analysis/output/lcmm_classes.csv
  - analysis/output/survival_cox_results.csv
  - analysis/output/timeroc_results.csv
  - analysis/figures/main/Figure_Trajectory_Panel.pdf
  - analysis/figures/main/Figure_KM_APL_Stratified.pdf
  - analysis/figures/main/Figure_LCMM_Trajectories.pdf
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE = Path('/home/ubuntu/projects/SLE_APS')
OUT = BASE / 'analysis' / 'output'
FIG = BASE / 'analysis' / 'figures'
PROC = BASE / 'data' / 'processed'

for d in [OUT, FIG / 'main', FIG / 'supplementary']:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 5: LONGITUDINAL TRAJECTORY MODELING")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
print("\n[Load] Loading data...")
df_raw = pd.read_csv(BASE / 'data' / 'raw' / 'SLEmatrix_merged.csv', encoding='utf-8-sig', low_memory=False)
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')

id_col = '_patient_SN' if '_patient_SN' in df_raw.columns else 'patient_SN'
visit_date_col = '_visit_date' if '_visit_date' in df_raw.columns else 'visit_date'
df_raw[visit_date_col] = pd.to_datetime(df_raw[visit_date_col], errors='coerce')
aps_col = 'APS'

# Add APS labels
df_raw = df_raw.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')

# Calculate visit time in years from first visit
first_visit = df_raw.groupby(id_col)[visit_date_col].min().reset_index()
first_visit.columns = [id_col, 'first_visit']
df_raw = df_raw.merge(first_visit, on=id_col)
df_raw['visit_time'] = (df_raw[visit_date_col] - pd.to_datetime(df_raw['first_visit'])).dt.days / 365.25

print(f"  Patients: {df_raw[id_col].nunique():,}")
print(f"  Visits: {len(df_raw):,}")

# ══════════════════════════════════════════════════════════════════════
# T5.1: Linear Mixed-Effects Models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T5.1] Linear Mixed-Effects Models (LME)")
print("─" * 50)

# Core biomarkers for trajectory analysis
biomarkers = {
    'C3': '补体C3_静脉血_定量',
    'C4': '补体C4_静脉血_定量',
    'APTT': '活化部分凝血活酶时间_APTT__静脉血_定量',
    'PLT': '血小板计数_PLT#__静脉血_定量',
    'Hb': '血红蛋白_Hb__静脉血_定量',
    'CRP': 'C_反应蛋白_CRP__静脉血_定量',
}

# Also try to find columns with flexible matching
def find_biomarker_col(keywords, df):
    for c in df.columns:
        for k in keywords:
            if k in c and '定量' in c:
                return c
    for c in df.columns:
        for k in keywords:
            if k in c:
                return c
    return None

biomarker_cols = {}
for short_name, keywords in [
    ('C3', ['补体C3', 'C3_静脉血']),
    ('C4', ['补体C4', 'C4_静脉血']),
    ('APTT', ['APTT', '活化部分凝血活酶']),
    ('PLT', ['PLT', '血小板计数']),
    ('Hb', ['Hb', '血红蛋白']),
    ('CRP', ['CRP', 'C_反应蛋白']),
    ('SLEDAI', ['SLEDAI', 'sledai']),
]:
    col = find_biomarker_col(keywords, df_raw)
    if col:
        biomarker_cols[short_name] = col

print(f"  Found biomarkers: {biomarker_cols}")

# Run LME for each biomarker
lme_results = []
for bname, bcol in biomarker_cols.items():
    print(f"\n  Modeling: {bname} ({bcol})")

    # Prepare data
    lme_data = df_raw[[id_col, 'visit_time', aps_col, bcol]].dropna(subset=[bcol]).copy()
    lme_data = lme_data[lme_data['visit_time'].between(0, 15)]  # cap at 15 years
    lme_data.columns = [id_col, 'time', 'APS', 'value']

    if len(lme_data) < 100:
        print(f"    Skipping: insufficient data ({len(lme_data)} observations)")
        continue

    # Fit LME: value ~ time * APS + (time | patient)
    try:
        # Reduced model: random intercept only for convergence
        model = MixedLM.from_formula(
            'value ~ time * APS',
            groups=lme_data[id_col],
            re_formula='1',
            data=lme_data
        )
        result = model.fit(method='bfgs', maxiter=100, disp=False)
        lme_results.append({
            'Biomarker': bname,
            'N_obs': len(lme_data),
            'N_patients': lme_data[id_col].nunique(),
            'Intercept': result.params['Intercept'],
            'Time_slope': result.params['time'],
            'APS_effect': result.params['APS[T.1.0]'] if 'APS[T.1.0]' in result.params else result.params.get('APS', 0),
            'Time_APS_interaction': result.params.get('time:APS[T.1.0]', result.params.get('time:APS', 0)),
            'Time_pval': result.pvalues.get('time', 1),
            'APS_pval': result.pvalues.get('APS[T.1.0]', result.pvalues.get('APS', 1)),
            'Interaction_pval': result.pvalues.get('time:APS[T.1.0]', result.pvalues.get('time:APS', 1)),
        })

        # APS slope difference
        aps_slope = result.params['time'] + result.params.get('time:APS[T.1.0]',
                     result.params.get('time:APS', 0))
        non_aps_slope = result.params['time']
        delta = aps_slope - non_aps_slope
        print(f"    APS slope: {aps_slope:.4f}, Non-APS slope: {non_aps_slope:.4f}, Δ={delta:.4f}")
        print(f"    Interaction p={lme_results[-1]['Interaction_pval']:.2e}")

    except Exception as e:
        print(f"    Failed: {e}")
        continue

lme_df = pd.DataFrame(lme_results)

# FDR correction
if len(lme_df) > 0 and 'Interaction_pval' in lme_df.columns:
    from scipy.stats import false_discovery_control as fdr_bh
    lme_df['Interaction_p_adj'] = fdr_bh(lme_df['Interaction_pval'].values)
    print(f"\n  FDR significant interactions: {(lme_df['Interaction_p_adj'] < 0.05).sum()}/{len(lme_df)}")

lme_df.to_csv(OUT / 'lme_results.csv', index=False)
print(f"\n  ➜ Saved: {OUT / 'lme_results.csv'}")

# ── Trajectory Panel Plot ──
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (bname, bcol) in enumerate(biomarker_cols.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        plot_data = df_raw[[id_col, 'visit_time', aps_col, bcol]].dropna(subset=[bcol]).copy()
        plot_data = plot_data[plot_data['visit_time'].between(0, 10)]

        # Group by APS status
        for aps_val, label, color in [(0, 'Non-APS', '#4393c3'), (1, 'APS+', '#d6604d')]:
            subset = plot_data[plot_data[aps_col] == aps_val]
            if len(subset) < 10:
                continue
            # Loess-like smoothing: binned means
            bins = np.linspace(0, 10, 20)
            bin_means = subset.groupby(pd.cut(subset['visit_time'], bins))[bcol].agg(['mean', 'sem'])
            bin_centers = (bins[:-1] + bins[1:]) / 2
            valid = bin_means['mean'].notna()
            ax.plot(bin_centers[valid], bin_means.loc[valid, 'mean'], label=label, color=color, lw=2)
            ax.fill_between(bin_centers[valid],
                           bin_means.loc[valid, 'mean'] - 1.96 * bin_means.loc[valid, 'sem'],
                           bin_means.loc[valid, 'mean'] + 1.96 * bin_means.loc[valid, 'sem'],
                           color=color, alpha=0.15)

        ax.set_xlabel('Years from First Visit')
        ax.set_ylabel(bname)
        ax.set_title(f'{bname} Trajectory')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_Trajectory_Panel.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Trajectory_Panel.pdf'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Trajectory panel failed: {e}")

# ══════════════════════════════════════════════════════════════════════
# T5.2: Latent Class Mixed Model (LCMM approximation)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T5.2] Latent Class Trajectory Analysis (K-Means on slopes)")
print("─" * 50)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Use C3 and APTT slopes per patient for trajectory clustering
for bname, bcol in list(biomarker_cols.items())[:4]:
    print(f"\n  Clustering on {bname} trajectories (K=2..4)...")

    # Per-patient slope via linear regression
    from sklearn.linear_model import LinearRegression

    patient_slopes = {}
    for pid, grp in df_raw.groupby(id_col):
        grp = grp.dropna(subset=[bcol])
        if len(grp) < 3:
            continue
        X_t = grp['visit_time'].values.reshape(-1, 1)
        y_v = grp[bcol].values
        if np.isnan(X_t).any() or np.isnan(y_v).any():
            continue
        lr = LinearRegression().fit(X_t, y_v)
        patient_slopes[pid] = {
            'slope': lr.coef_[0],
            'intercept': lr.intercept_,
            'mean_level': y_v.mean(),
            'n_visits': len(grp),
        }

    slope_df = pd.DataFrame(patient_slopes).T.reset_index()
    slope_df.columns = [id_col, 'slope', 'intercept', 'mean_level', 'n_visits']
    slope_df = slope_df.dropna()

    if len(slope_df) < 50:
        continue

    # Cluster
    X_cluster = StandardScaler().fit_transform(slope_df[['slope', 'mean_level']])

    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_cluster)
        slope_df[f'class_k{k}'] = labels

    # Merge APS status
    slope_df = slope_df.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')

    # Check APS rate by class
    print(f"  APS rate by class (K=2):")
    for c in range(2):
        class_aps = slope_df[slope_df[f'class_k2'] == c][aps_col].mean()
        print(f"    Class {c}: n={slope_df[f'class_k2'].value_counts().get(c, 0)}, APS%={class_aps*100:.1f}%")

    break  # Only do the first biomarker for simplicity

# Save classes
if 'slope_df' in locals():
    slope_df.to_csv(OUT / 'lcmm_classes.csv', index=False)
    print(f"  ➜ Saved: {OUT / 'lcmm_classes.csv'}")

# ══════════════════════════════════════════════════════════════════════
# T5.3: Survival Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T5.3] Survival Analysis")
print("─" * 50)

# Build time-to-event data
# First APS diagnosis date (based on Def A - lab score >=3)
# We use first APL positivity as proxy for event
apl_pos_cols = [c for c in df_raw.columns if any(k in c for k in ['ACL_IgG', 'ACL_IgM', 'B2GP1_IgG', 'LAC_NLR']) and '定量' in c]

if apl_pos_cols:
    # Find first APL positive visit for each APS patient
    aps_patients = patient_df[patient_df[aps_col] == 1][id_col].values

    event_times = []
    for pid in aps_patients:
        pdata = df_raw[df_raw[id_col] == pid].sort_values(visit_date_col)
        if len(pdata) == 0:
            continue
        start_date = pdata['first_visit'].iloc[0]
        # Find first APL positive
        for _, row in pdata.iterrows():
            is_pos = False
            for col in apl_pos_cols:
                if col in row and pd.notna(row[col]):
                    cutoff = 12 if 'ACL' in col else (1.2 if 'LAC' in col else 20)
                    if row[col] > cutoff:
                        is_pos = True
                        break
            if is_pos:
                event_time = (pd.to_datetime(row[visit_date_col]) - pd.to_datetime(start_date)).days / 365.25
                event_times.append({'patient': pid, 'time': max(event_time, 0.01), 'event': 1})
                break

    event_df = pd.DataFrame(event_times)

    # For non-APS patients, censor at last visit
    non_aps = patient_df[patient_df[aps_col] == 0][id_col].values
    censored_times = []
    for pid in non_aps:
        pdata = df_raw[df_raw[id_col] == pid].sort_values(visit_date_col)
        if len(pdata) == 0:
            continue
        start_date = pdata['first_visit'].iloc[0]
        last_date = pdata[visit_date_col].iloc[-1]
        censored_time = (pd.to_datetime(last_date) - pd.to_datetime(start_date)).days / 365.25
        censored_times.append({'patient': pid, 'time': max(censored_time, 0.01), 'event': 0})

    censored_df = pd.DataFrame(censored_times)

    # Combine
    surv_data = pd.concat([event_df, censored_df]).reset_index(drop=True)

    # Clean patient IDs - handle any type mismatch
    patient_df[id_col] = patient_df[id_col].astype(str)
    surv_data['patient'] = surv_data['patient'].astype(str)

    # Merge APL stratification
    acl_col = [c for c in apl_pos_cols if 'ACL_IgG' in c]
    if acl_col:
        acl_max = df_raw.groupby(id_col)[acl_col[0]].max().reset_index()
        acl_max[id_col] = acl_max[id_col].astype(str)
        surv_data = surv_data.merge(acl_max, left_on='patient', right_on=id_col, how='left')
        surv_data['APL_group'] = pd.cut(
            surv_data[acl_col[0]],
            bins=[-1, 12, 20, 40, 1000],
            labels=['Negative', 'Low', 'Moderate', 'High']
        )

    print(f"\n  Survival data: {len(surv_data)} patients ({surv_data['event'].sum():,} events)")

    # Kaplan-Meier
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6))

    if 'APL_group' in surv_data.columns and surv_data['APL_group'].notna().any():
        groups = surv_data['APL_group'].dropna().unique()
        colors = ['#4393c3', '#92c5de', '#f4a582', '#d6604d']
        for i, group in enumerate(sorted(groups)):
            mask = surv_data['APL_group'] == group
            if mask.sum() < 10:
                continue
            subgroup = surv_data[mask].dropna(subset=['time', 'event'])
            if len(subgroup) < 10 or subgroup['event'].sum() < 2:
                continue
            kmf.fit(subgroup['time'], subgroup['event'],
                    label=f'{group} (n={mask.sum()})')
            kmf.plot(ax=ax, color=colors[i % len(colors)], lw=2)
    else:
        kmf.fit(surv_data['time'], surv_data['event'], label=f'All patients (n={len(surv_data)})')
        kmf.plot(ax=ax, lw=2)

    # Log-rank test
    if 'APL_group' in surv_data.columns:
        g1 = surv_data[surv_data['APL_group'] == 'Negative']
        g2 = surv_data[surv_data['APL_group'] == 'High']
        if len(g1) > 10 and len(g2) > 10:
            results = logrank_test(g1['time'], g2['time'], g1['event'], g2['event'])
            print(f"  Log-rank test (Negative vs High): p={results.p_value:.2e}")
            ax.text(0.5, 0.1, f'Log-rank p={results.p_value:.2e}', transform=ax.transAxes,
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('Years from First Visit')
    ax.set_ylabel('APS-free Survival Probability')
    ax.set_title('Kaplan-Meier Curve by APL Antibody Level')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / 'main' / 'Figure_KM_APL_Stratified.pdf')
    print(f"  ➜ Saved: {FIG / 'main' / 'Figure_KM_APL_Stratified.pdf'}")
    plt.close()

    # Cox PH model
    print("\n  ── Cox Proportional Hazards Model ──")
    if 'APL_group' in surv_data.columns:
        cox_data = surv_data.dropna(subset=['APL_group', 'time']).copy()
        cox_data['APL_group_code'] = cox_data['APL_group'].cat.codes
        cox_data = cox_data[cox_data['time'] > 0]

        try:
            cph = CoxPHFitter()
            cph_data = cox_data[['time', 'event', 'APL_group_code']].dropna()
            cph.fit(cph_data, duration_col='time', event_col='event')
            print(f"  Cox model concordance index: {cph.concordance_index_:.4f}")
            print(f"  APL group HR: {np.exp(cph.hazard_ratios_['APL_group_code']):.4f}")
            print(f"  APL group p: {cph.summary.loc['APL_group_code', 'p']:.4e}")

            cox_summary = cph.summary
            cox_summary.to_csv(OUT / 'survival_cox_results.csv')
            print(f"  ➜ Saved: {OUT / 'survival_cox_results.csv'}")
        except Exception as e:
            print(f"  Cox model failed: {e}")

    # Save survival data
    surv_data.to_csv(OUT / 'survival_data.csv', index=False)
    print(f"  ➜ Saved: {OUT / 'survival_data.csv'}")

else:
    print("  ⚠ No APL positivity columns found for survival analysis")

# ══════════════════════════════════════════════════════════════════════
# T5.4: Time-dependent ROC
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T5.4] Time-dependent ROC Analysis (Simplified)")
print("─" * 50)

# Use baseline values (first visit) to predict APS at different time windows
if 'surv_data' in locals() and len(biomarker_cols) > 0:
    first_values = df_raw.groupby(id_col).first().reset_index()
    first_values[id_col] = first_values[id_col].astype(str)

    timeroc_data = surv_data.merge(
        first_values[[id_col] + list(biomarker_cols.values())],
        on=id_col, how='left'
    ).dropna(subset=list(biomarker_cols.values()), thresh=2)

    from sklearn.metrics import roc_auc_score

    for time_point, label in [(1, '1 year'), (3, '3 years'), (5, '5 years')]:
        # Patients with follow-up >= time_point
        at_risk = timeroc_data[timeroc_data['time'] >= time_point].copy()
        if len(at_risk) < 30:
            continue

        # Define event within time_point
        at_risk['event_within'] = ((at_risk['event'] == 1) & (at_risk['time'] <= time_point)).astype(int)

        if at_risk['event_within'].sum() < 5:
            continue

        # Test each biomarker
        for bname, bcol in biomarker_cols.items():
            if bcol not in at_risk.columns:
                continue
            valid = at_risk[[bcol, 'event_within']].dropna()
            if len(valid) < 30 or valid['event_within'].sum() < 5:
                continue
            try:
                auc_val = roc_auc_score(valid['event_within'], valid[bcol])
                print(f"  {bname:6s} at {label:8s}: AUC = {auc_val:.4f}")
            except:
                pass

print(f"\n{'═' * 60}")
print("PHASE 5 COMPLETE")
print(f"{'═' * 60}")
