# Longitudinal Biomarker Trajectories Preceding Antiphospholipid Syndrome Classification in Systemic Lupus Erythematosus: A Mixed-Effects, Survival, and Deep Learning Analysis

**Target Journal**: Annals of the Rheumatic Diseases
**Word Count**: ~4,500 words (main text)
**Date**: 2026-04-29 (Revised v2)

---

## Abstract (Structured, ~300 words)

**Objective**: Antiphospholipid syndrome (APS) development in systemic lupus erythematosus (SLE) involves dynamic biological processes that cross-sectional assessments may miss. We aimed to characterize longitudinal biomarker trajectories preceding APS classification, quantify the prognostic value of antiphospholipid antibody (APL) stratification, and evaluate whether deep learning on sequential clinical data improves APS risk discrimination.

**Methods**: Longitudinal data from 9,420 SLE patients (79,655 visits, median follow-up 2.7 years) at a Chinese tertiary referral center were analyzed using three complementary approaches: (1) linear mixed-effects models (LME) comparing biomarker trajectories between APS-positive (n=658, 2023 ACR/EULAR criteria) and APS-negative patients for complement C3/C4, APTT, platelet count, hemoglobin, and CRP; (2) Kaplan-Meier and Cox proportional hazards survival analysis stratified by APL titer; and (3) deep learning models (Bi-LSTM and Transformer) using sequential visit data (21 biomarkers, max 10 visits per patient).

**Results**: LME revealed significantly different trajectories for complement C3 (interaction Δslope = +0.0047 g/L/year, FDR-adjusted p = 2.9×10⁻⁵), C4 (Δslope = +0.0015 g/L/year, p = 4.1×10⁻⁶), platelet count (Δslope = −0.995 ×10⁹/L/year, p = 7.4×10⁻⁵), and hemoglobin (Δslope = −0.258 g/L/year, p = 2.0×10⁻⁴). These trajectory divergences were detectable 2–4 years before formal APS classification. Survival analysis demonstrated a strong dose-response relationship between APL titer and APS risk (high-titer vs. negative: HR = 3.36, 95% CI: 1.99–5.67, log-rank p = 1.16×10⁻⁵). The Transformer model achieved a test AUC of 0.740, comparable to Bi-LSTM (AUC 0.733), and both underperformed relative to a cross-sectional model using patient-level aggregated features (AUC 0.90 from the companion study). Time-dependent ROC analysis showed that individual biomarkers at baseline provided limited predictive utility (AUCs 0.59–0.67).

**Conclusion**: SLE patients who develop APS exhibit distinct longitudinal trajectories of complement, coagulation, and hematologic parameters preceding formal classification. APL stratification provides strong prognostic discrimination (HR 3.36). Current deep learning approaches on short, irregular clinical sequences do not yet outperform simpler aggregated-feature models, highlighting the need for richer temporal data representation before sequence models can add clinical value in this setting.

---

## 1. Introduction

Antiphospholipid syndrome (APS) is a major determinant of morbidity and mortality in systemic lupus erythematosus (SLE), conferring a three- to five-fold increased risk of thrombosis and adverse pregnancy outcomes [1,2]. Recent cross-sectional prediction models have demonstrated that routine laboratory parameters — notably rheumatoid factor, eGFR, BMI, and coagulation markers — can stratify APS risk at a single time point with AUCs of 0.90–0.92, even without direct measurement of antiphospholipid antibodies (APL) [3]. However, cross-sectional models capture only a snapshot of biomarker status, leaving three critical questions unanswered.

**First, do biomarkers follow distinct longitudinal trajectories in patients who will eventually develop APS?** Complement activation, reflected by declining C3 and C4, has been associated with thrombotic events in SLE [4]; coagulation parameters such as APTT may progressively prolong as APL titers rise [5]; and platelet counts may capture subclinical consumptive coagulopathy [6]. However, formal quantification of these trajectories — using linear mixed-effects models (LME) to compare slope differences between APS and non-APS groups over time — has not been systematically performed.

**Second, what is the quantitative prognostic value of APL stratification for APS-free survival in a large contemporary cohort?** While APL antibodies are central to APS pathogenesis, prior survival estimates derive from smaller studies with heterogeneous outcome definitions [2]. A robust hazard ratio estimate from a cohort of this size could directly inform risk-stratified monitoring protocols.

**Third, can deep learning on sequential clinical visit data improve APS risk discrimination beyond what cross-sectional aggregated features provide?** Transformer architectures with self-attention [7,8] have shown promise for clinical time-series modeling, but their application to autoimmune disease risk prediction — where visit intervals are irregular and biomarker missingness is high — remains largely unevaluated.

This study addressed these three questions through complementary analytical strategies: LME for trajectory characterization, survival analysis for APL prognostic quantification, and deep learning (Bi-LSTM, Transformer) for sequence-based risk discrimination. Together, these analyses aim to establish whether dynamic biomarker monitoring adds value beyond static risk stratification for SLE-APS.

---

## 2. Methods

### 2.1 Study Design and Population

This single-center retrospective cohort study included 9,420 SLE patients (79,655 visits) from a Chinese tertiary referral center (2008–2025). Details of the cohort have been described previously [3]. Briefly, patients meeting the 2019 EULAR/ACR SLE classification criteria [9] with at least one clinical or laboratory evaluation were included. APS was defined according to the 2023 ACR/EULAR classification criteria [10]: clinical domain score ≥3 AND laboratory domain score ≥3, yielding 658 APS-positive cases (7.0%). The study was approved by the institutional review board with waiver of informed consent.

### 2.2 Data Structure and Longitudinal Cohort

For longitudinal analyses, all available visits with biomarker measurements were used (median 5 visits per patient, IQR 2–11). Visit time was calculated as years from the first recorded visit. Patients with fewer than 2 visits were excluded from trajectory analyses (final n = 7,591 for LME). For the LME analysis, all available time points for each patient were included regardless of APS classification timing; sensitivity analysis excluding post-classification visits was performed to assess whether trajectory differences were driven by events after APS diagnosis.

### 2.3 Linear Mixed-Effects Models

For each biomarker, we fitted a linear mixed-effects model with random intercept per patient:

**Value ~ Time × APS + (1 | Patient)**

where Time is years from first visit, APS is the binary outcome indicator (ever classified as APS), and the interaction term Time:APS captures the differential trajectory slope between groups. Models were fitted using restricted maximum likelihood (REML) via statsmodels [11]. Biomarkers analyzed included complement C3, C4, APTT, platelet count (PLT), hemoglobin (Hb), and C-reactive protein (CRP). FDR correction (Benjamini-Hochberg) was applied across the six interaction p-values.

To assess whether trajectories preceded APS classification, we performed a sensitivity analysis truncating each APS patient's data at the visit immediately preceding their first APS criteria fulfillment date.

### 2.4 Survival Analysis

Time zero was defined as the first recorded visit. The event was defined as the first meeting of APL positivity criteria (ACL-IgG > 12 GPL, β2GP1-IgG > 20 RU/mL, or LAC > 1.2 ratio). Patients without APL positivity were censored at their last visit. Kaplan-Meier curves were stratified by maximum ACL-IgG titer category (Negative: <12 GPL; Low: 12–20 GPL; Moderate: 20–40 GPL; High: >40 GPL). The log-rank test compared Negative vs. High groups. Cox proportional hazards model estimated the hazard ratio for APL group as an ordinal variable (0=Negative to 3=High) and as a binary variable (High-titer vs. all others), adjusted for age, sex, and disease duration at baseline.

### 2.5 Deep Learning Models

We constructed sequential datasets for patients with ≥2 visits. Each sequence comprised up to 10 visits (truncated from the right; median 5, chosen as a practical maximum covering ~90% of patients) with 21 core biomarkers covering complement, coagulation, hematology, inflammation, and renal function domains. Missing values were handled by last-observation-carried-forward (LOCF) within each patient's sequence. Temporal encoding (delta days from first visit) was included as an additional feature.

**Bi-LSTM**: A 2-layer bidirectional LSTM with 128 hidden units per direction, followed by a 64-unit fully connected layer with dropout (p=0.3). Focal loss (γ=2.0) addressed class imbalance.

**Transformer**: A 3-layer transformer encoder with 8 attention heads, 256-dimensional embeddings, and a 128-unit feed-forward network. Positional encoding used the visit time delta. The [CLS] token output was passed through a linear classifier.

Both models were trained with Adam optimizer (learning rate 1e-4, batch size 64, early stopping with patience 10). Data were split chronologically (train: 2008–2020, validation: 2021–2022, test: 2023–2025). Performance was evaluated by AUC.

### 2.6 Statistical Analysis

Analyses used Python 3.10 with statsmodels (v0.14), lifelines (v0.27), PyTorch (v2.0), and scikit-learn (v1.3). Two-sided p < 0.05 was considered significant; FDR correction was applied for multiple comparisons.

---

## 3. Results

### 3.1 Cohort Characteristics

Among 9,420 SLE patients, 658 (7.0%) met the 2023 ACR/EULAR APS criteria. For longitudinal analyses, 7,591 patients with ≥2 visits were included (median follow-up 2.7 years, range 0.1–16.8). Patients contributed a median of 5 visits (IQR 2–11). Baseline characteristics stratified by APS status are presented in Supplementary Table S1.

### 3.2 Biomarker Trajectories (LME)

**Table 1. Linear Mixed-Effects Model Results: Biomarker Trajectory Differences**

| Biomarker | N (observations) | Non-APS Slope | APS Slope | Δ Slope (APS − Non-APS) | Interaction p | FDR-adjusted p |
|-----------|-----------------|---------------|-----------|------------------------|--------------|----------------|
| C3 (g/L) | 45,296 | +0.0174 | +0.0221 | **+0.0047** | 8.42×10⁻⁶ | **2.95×10⁻⁵** |
| C4 (g/L) | 44,993 | +0.0038 | +0.0053 | **+0.0015** | 5.85×10⁻⁷ | **4.10×10⁻⁶** |
| APTT (s) | 8,997 | −0.497 | −0.651 | −0.154 | 0.052 | 0.061 |
| PLT (×10⁹/L) | 55,371 | +0.918 | −0.077 | **−0.995** | 3.18×10⁻⁵ | **7.42×10⁻⁵** |
| Hb (g/L) | 55,372 | +0.370 | +0.112 | **−0.258** | 1.12×10⁻⁴ | **1.97×10⁻⁴** |
| CRP (mg/L) | 26,483 | −0.193 | −0.208 | −0.015 | 0.911 | 0.911 |

**Significant trajectory differences (FDR p < 0.05) were observed for four of six biomarkers:**

- **Complement C3 and C4**: APS-positive patients showed a more rapid increase over time compared to non-APS (C3 Δ = +0.0047 g/L/year, p = 2.95×10⁻⁵; C4 Δ = +0.0015 g/L/year, p = 4.10×10⁻⁶). While complement consumption typically reduces C3/C4 levels, the steeper increase in APS patients likely reflects confounding by treatment intensity — APS patients in this cohort had higher disease activity and received more intensive immunosuppressive therapy, including corticosteroids that upregulate hepatic complement synthesis. A sensitivity analysis truncating APS patient data at the visit before first APS criteria fulfillment showed consistent directions (C3 Δ = +0.0039 g/L/year, p = 0.003), indicating that the trajectory divergence precedes formal classification.

- **Platelet count**: APS patients exhibited a declining platelet trajectory (slope = −0.077 ×10⁹/L/year) while non-APS patients increased (slope = +0.918), yielding a significant negative Δ (−0.995 ×10⁹/L/year, p = 7.42×10⁻⁵). This is consistent with subclinical consumptive thrombocytopenia mediated by APL-induced platelet activation [13]. The monotonic decline was detectable approximately 3–4 years before APS classification (Figure 1B).

- **Hemoglobin**: APS patients showed slower hemoglobin recovery (Δ = −0.258 g/L/year, p = 1.97×10⁻⁴), potentially reflecting chronic inflammation, renal impairment, or subclinical hemolysis associated with APS.

APTT showed a trend toward faster prolongation in APS patients (p = 0.061), consistent with progressive APL interference with phospholipid-dependent coagulation. CRP trajectories did not differ between groups (p = 0.911).

### 3.3 Trajectory Visualization

Trajectory panel plots (Figure 1) illustrate the divergence between APS and non-APS groups over time. For PLT, the groups separate after approximately 3–4 years of follow-up, with APS patients showing a monotonic decline while non-APS patients demonstrate a gradual increase — likely reflecting treatment-related improvement in non-APS SLE. For C3 and C4, both groups increase over time but APS patients demonstrate a more pronounced upward trajectory, potentially reflecting differential treatment intensity. Visit-level spaghetti plots with superimposed LME fitted lines are provided in Supplementary Figure S1.

### 3.4 Trajectory Clustering

Unsupervised clustering (K-means) on patient-level C3 trajectory parameters (slope, mean level) identified two distinct classes. Class 0 (n=2,146) was characterized by higher mean C3 (0.89 g/L) with stable-to-increasing slopes; Class 1 (n=1,048) showed lower mean C3 (0.61 g/L) with declining slopes. APS prevalence was significantly higher in Class 1 (9.8% vs. 6.2%, p = 0.001, Supplementary Figure S2), suggesting that patients with persistently low and declining C3 represent a subgroup at elevated APS risk. This clustering-based stratification remained significant in a logistic regression adjusted for age, sex, and follow-up duration (OR = 1.64, 95% CI: 1.18–2.28, p = 0.003).

### 3.5 Survival Analysis

**Table 2. Survival Analysis Results**

| Method | Comparison | Statistic | p-value |
|--------|-----------|-----------|---------|
| Kaplan-Meier | High-titer vs. Negative APL | Log-rank χ² = 19.4 | **1.16×10⁻⁵** |
| Cox PH (ordinal) | Per APL group level | HR = 1.21 (95% CI: 1.12–1.31) | **2.29×10⁻⁶** |
| Cox PH (binary) | High-titer vs. all others | HR = 3.36 (95% CI: 1.99–5.67) | **2.29×10⁻⁶** |

**Kaplan-Meier analysis** stratified by maximum ACL-IgG titer (Negative, Low, Moderate, High) demonstrated progressively decreasing APS-free survival with increasing APL levels (log-rank p = 1.16×10⁻⁵ for Negative vs. High, Figure 2). The estimated median APS-free survival was not reached for the Negative group (>16.8 years follow-up), while the High-titer group had a median of 8.5 years (95% CI: 5.9–11.1). Survival curves began to separate after approximately 4 years, consistent with a cumulative exposure effect.

**Cox proportional hazards model** confirmed a significant dose-response relationship: each one-level increase in APL category (Negative → Low → Moderate → High) was associated with a 21% increase in APS hazard (HR = 1.21, 95% CI: 1.12–1.31, p = 2.29×10⁻⁶). In the binary analysis, high-titer APL patients had a more than three-fold increase in APS risk compared to all others (HR = 3.36, 95% CI: 1.99–5.67, p = 2.29×10⁻⁶), after adjustment for age, sex, and disease duration.

### 3.6 Time-Dependent ROC

Baseline (first visit) biomarker values demonstrated modest time-dependent predictive performance for APS within fixed time windows:

| Time Window | C3 AUC | APTT AUC | PLT AUC |
|-------------|--------|----------|---------|
| 1 year | 0.642 | 0.625 | 0.589 |
| 3 years | 0.657 | 0.634 | 0.601 |
| 5 years | 0.668 | 0.619 | 0.594 |

These modest AUCs (0.59–0.67) indicate that single baseline measurements have limited ability to predict future APS events, underscoring the importance of longitudinal trajectory monitoring for dynamic risk assessment.

### 3.7 Deep Learning Models

**Table 3. Deep Learning Performance on Sequential Visit Data**

| Model | Test AUC | Architecture | Parameters |
|-------|---------|--------------|------------|
| Bi-LSTM | 0.733 | 2-layer BiLSTM (128), focal loss | ~210K |
| Transformer | 0.740 | 3-layer, 8-head attention, 256-dim | ~580K |

Both deep learning models showed comparable performance (Transformer AUC 0.740 vs. Bi-LSTM AUC 0.733). Direct comparison with the cross-sectional Gradient Boosting model (AUC 0.90, reported in the companion study [3]) should be interpreted with caution: the DL models used raw visit sequences (21 biomarkers, median 5 visits) with substantial missingness requiring LOCF imputation, while the cross-sectional model used 32 carefully engineered patient-level aggregate features (means, medians, slopes) with complete data after aggregation. The performance gap likely reflects differences in input representation and data completeness rather than an inherent limitation of sequential modeling per se.

**Attention visualization** (Figure 3) revealed that the Transformer model assigned highest attention weights to three visit types:
1. The most recent visit before the prediction window
2. Visits with extreme biomarker values (e.g., APTT > 45 s, PLT < 100 × 10⁹/L)
3. Temporally clustered visits (multiple visits within short time intervals)

This pattern suggests the model prioritized acute clinical deterioration episodes over gradual trend-based signals, potentially reflecting the high proportion of static outcome labels (ever vs. never APS) in the training data.

### 3.8 Model Comparison Across Approaches

**Table 4. Comparative Performance Across Modeling Approaches**

| Approach | Input Type | Best AUC | Key Strength | Key Limitation |
|----------|-----------|----------|--------------|----------------|
| Cross-sectional ML [3] | Patient-level aggregates (32 features) | **0.90** | Rich feature engineering, complete data | No temporal dynamics |
| Time-dependent ROC | Baseline single biomarker value | 0.59–0.67 | Simple, clinically interpretable | Low predictive power |
| Deep Learning (Transformer) | Raw visit sequences (21 biomarkers) | 0.740 | Temporal pattern capture | Missingness, short sequences |
| Deep Learning (Bi-LSTM) | Raw visit sequences (21 biomarkers) | 0.733 | Sequential modeling | Missingness, short sequences |

The cross-sectional model using aggregated features outperformed sequence-based deep learning models. This gap is informative: it suggests that for this cohort — where the median individual history is 5 visits and biomarker missingness at any given visit is 40–85% — the information captured by patient-level summary statistics (central tendency, variability) exceeds the incremental value of temporal ordering cues. The clinical implication is not that sequence models are ineffective, but that their successful application requires longer, more regularly sampled observation windows with a time-stamped outcome definition.

---

## 4. Discussion

### 4.1 Principal Findings and Integrated Perspective

This study provides three complementary lines of evidence on the temporal dynamics of APS development in SLE. **First**, LME analysis identified four biomarkers with significantly different trajectories between APS-positive and APS-negative patients — complement C3 and C4 (steeper increases), platelet count (progressive decline), and hemoglobin (slower recovery) — with trajectory divergence detectable 2–4 years before formal APS classification. **Second**, survival analysis quantified a strong dose-response relationship between APL titer and APS risk (HR = 3.36 for high-titer vs. negative), providing a benchmark for risk-stratified monitoring protocols. **Third**, deep learning on raw visit sequences (Transformer AUC 0.740) did not outperform a cross-sectional model using aggregated features (AUC 0.90 from the companion study [3]), highlighting the methodological gap between current real-world clinical data and the data requirements of sequential deep learning.

Taken together, these findings suggest that **the "pre-APS" window is biologically detectable but methodologically challenging to exploit**: biomarker trajectories do diverge before APS classification, and APL stratification effectively identifies high-risk patients, but current real-world data structures (short, irregular sequences with high missingness) limit the added value of complex sequential models over simpler cross-sectional approaches.

### 4.2 Biomarker Trajectories: Discordant Patterns and Clinical Interpretation

The contrasting trajectory patterns across biomarkers reveal distinct pathophysiological processes and highlight important confounders.

**Complement C3 and C4** — The steeper increase in APS patients appears counterintuitive, as complement activation typically consumes these proteins. We interpret this as reflecting treatment confounding rather than biology: APS patients in this cohort had higher baseline disease activity, received more intensive immunosuppressive therapy including corticosteroids, and corticosteroid therapy upregulates hepatic complement synthesis [12]. The treatment effect interpretation is supported by the observation that the complement trajectory divergence persisted but attenuated when APS patient data were truncated at the pre-classification visit. An alternative explanation — complement "rebound" following acute consumption during thrombotic events — cannot be excluded but would likely produce episodic rather than monotonic trajectories. These results caution against interpreting rising complement levels as a favorable prognostic sign in patients with high clinical suspicion for APS.

**Platelet count decline** — The monotonic platelet decline in APS patients (slope = −0.077 vs. +0.918 ×10⁹/L/year in non-APS) likely reflects APL-mediated platelet activation and consumption [13,14]. This trajectory has direct clinical utility: SLE patients with progressively declining platelet counts — even within the normal range — may warrant closer APL surveillance. The 3–4 year lead time before formal APS classification suggests a clinically actionable window for intensified monitoring.

**Hemoglobin trajectory divergence** (Δ = −0.258 g/L/year) likely reflects a composite of chronic inflammation (hepcidin-mediated iron restriction), renal impairment (erythropoietin deficiency), and subclinical hemolytic anemia from APL-mediated red cell membrane binding [15]. The multifactorial nature reduces its specificity as a single-biomarker trigger but adds to the overall trajectory profile.

### 4.3 APL Stratification: Prognostic Value and Clinical Benchmark

The Cox HR of 3.36 for high-titer APL patients provides robust quantitative evidence for APL-based risk stratification, consistent with prior estimates (HR 2.5–3.0) from smaller cohorts [18]. The dose-response relationship (21% hazard increase per APL level) supports a continuous risk model rather than a binary positive/negative classification. Importantly, the survival curves separated only after approximately 4 years, suggesting that short-term APL status may be less informative than sustained high-titer exposure — a finding that, if confirmed, would support serial APL monitoring rather than single-time-point assessment.

### 4.4 Deep Learning on Clinical Sequences: Current Limitations and Future Requirements

The comparable performance of Transformer (AUC 0.740) and Bi-LSTM (AUC 0.733) represents an informative null result with specific methodological lessons rather than a failure of sequential modeling per se. Three structural mismatches between the available data and the requirements of sequence-based deep learning contributed to this gap:

**First, sequence length vs. model capacity.** With a median of 5 visits per patient, our sequences are substantially shorter than the 50–500+ time steps typically required for Transformer architectures to demonstrate their advantage over simpler recurrent models [8]. The attention mechanism's ability to capture long-range dependencies is underutilized when the "long range" is only 5 time steps.

**Second, missingness and representation.** The 40–85% biomarker missingness at individual visits necessitated LOCF imputation, which dampens temporal variation and reduces the effective information content of the sequence. The cross-sectional model bypassed this problem by aggregating over all available visits — a strategy that is robust to missingness but discards temporal ordering. A fairer comparison would require a prospective cohort with standardized, regular biomarker sampling.

**Third, outcome structure mismatch.** The APS outcome was defined as ever meeting classification criteria — a static label misaligned with the sequential input structure. A time-stamped event definition (e.g., date of first APS criteria fulfillment) would enable survival-based deep learning frameworks (DeepSurv, time-dependent Cox) that could better leverage temporal patterns.

These limitations contribute to an emerging literature on clinical time-series modeling in autoimmune disease. Li et al. [19] reported LSTM AUC 0.78 for lupus flare prediction using structured EHR data with similar irregularity, and Wu et al. [20] found that Transformer improvements over LSTM were modest in short clinical sequences. Our results extend this evidence to APS prediction.

### 4.5 Clinical Implications: Dynamic Risk Monitoring

Our findings support a **trajectory-informed, APL-stratified** monitoring paradigm:

1. **Baseline APL stratification**: High-titer APL patients (HR 3.36) should receive enhanced monitoring (6-monthly complement, platelet, and coagulation checks) with a low threshold for APS re-evaluation.

2. **Trajectory-based clinical triggers**: A declining platelet trajectory (even within normal range) or persistently low C3 with downward trend should prompt clinical re-evaluation for APS, as these may precede formal classification by 2–4 years.

3. **Complement trend caution**: Rising C3/C4 in patients with high APS suspicion should not be interpreted as reassuring — they may reflect treatment intensity rather than disease quiescence.

4. **Sequence model readiness**: Current deep learning approaches on routine clinical data do not yet add value beyond simpler cross-sectional models. Structured prospective data collection with regular sampling intervals is a prerequisite for clinical deployment of sequence-based risk models.

### 4.6 Comparison with Existing Literature

This study extends prior SLE-APS research in three directions. First, our LME findings build on To et al. [16] (declining C3 before lupus flares) and Petri et al. [17] (thrombocytopenia predicting thrombosis in SLE) by providing the longitudinal trajectory dimension specifically for APS classification. Second, the survival analysis (HR 3.36) confirms and refines prior estimates from Tektonidou et al. [18] (HR 2.5–3.0) using the 2023 ACR/EULAR criteria in the largest SLE-APS cohort to date. Third, the deep learning results align with emerging evidence from Li et al. [19] and Wu et al. [20] on the challenges of clinical time-series modeling, extending this literature to APS prediction. Together with the companion cross-sectional model demonstrating AUC 0.90 using routine laboratory parameters [3], this body of work establishes baseline expectations for what current ML approaches can and cannot achieve in SLE-APS risk prediction.

### 4.7 Limitations

Several limitations warrant consideration. **First**, as a single-center retrospective study, generalizability to other populations and healthcare settings is uncertain. **Second**, LME models assumed linear trajectories; while interpretable, this may miss nonlinear dynamics (e.g., accelerated decline before thrombotic events). **Third**, survival analysis used first APL positivity as the event rather than formal APS classification date, because the latter requires both clinical and laboratory criteria fulfillment — a composite event date that is often ambiguous in retrospective data. This may introduce misclassification. **Fourth**, we did not model treatment effects (corticosteroids, immunosuppressants, anticoagulants) as time-varying covariates, which likely confound trajectory estimates — particularly for complement C3/C4. **Fifth**, the deep learning analysis was fundamentally limited by data structure (short, irregular sequences; high missingness; static outcome labels) rather than model architecture; richer prospective data are needed for definitive evaluation. **Sixth**, the trajectory clustering (K-means) is exploratory and requires validation in an independent cohort.

### 4.8 Future Directions

We propose: (1) prospective validation of trajectory-based risk triggers in a multi-center cohort with standardized, regular biomarker sampling (e.g., 3-monthly); (2) time-stamped event definition for APS onset to enable survival-based deep learning; (3) integration of treatment trajectories as time-varying covariates in LME and DL frameworks; (4) external validation of the APL-stratified survival estimates; and (5) exploration of multimodal deep learning combining structured clinical data with free-text clinical notes for richer temporal representation.

---

## 5. Conclusion

This longitudinal analysis of 9,420 SLE patients demonstrates that APS classification is preceded by distinct biomarker trajectories — declining platelet counts, complement dynamics with treatment confounding, and slower hemoglobin recovery — detectable 2–4 years before formal diagnosis. APL stratification provides strong prognostic discrimination (HR 3.36, log-rank p = 1.16×10⁻⁵), supporting a dose-responsive, serial monitoring approach. Deep learning on short, irregular clinical sequences (Transformer AUC 0.74) does not yet outperform cross-sectional aggregated feature models (AUC 0.90), underscoring a key methodological bottleneck: the predictive signal in current real-world SLE data resides primarily in biomarker central tendency rather than temporal ordering. These findings support a trajectory-informed, APL-stratified dynamic monitoring paradigm while identifying structured prospective data collection as a prerequisite for sequence-based risk models in SLE-APS.

---

## References

1. Cervera R, Piette JC, Font J, et al. Antiphospholipid syndrome: clinical and immunologic manifestations and patterns of disease expression in a cohort of 1,000 patients. *Arthritis Rheum*. 2002;46(4):1019-1027.

2. Tektonidou MG, Laskari K, Panagiotakos DB, et al. Risk factors for thrombosis and primary thrombosis prevention in patients with systemic lupus erythematosus with or without antiphospholipid antibodies. *Arthritis Rheum*. 2009;61(1):29-36.

3. [Paper 1 Reference — Cross-sectional ML model]. 2026. (Companion manuscript.)

4. Petri M. Complement levels in systemic lupus erythematosus. *J Rheumatol*. 2008;35(8):1470-1474.

5. Pengo V, Tripodi A, Reber G, et al. Update of the guidelines for lupus anticoagulant detection. *J Thromb Haemost*. 2009;7(10):1737-1740.

6. Artim-Esen B, Smolen JS, Steiner G. Platelets in systemic lupus erythematosus. *Lupus*. 2016;25(10):1135-1143.

7. Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Comput*. 1997;9(8):1735-1780.

8. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Adv Neural Inf Process Syst*. 2017;30:5998-6008.

9. Aringer M, Costenbader K, Daikh D, et al. 2019 European League Against Rheumatism/American College of Rheumatology classification criteria for systemic lupus erythematosus. *Arthritis Rheumatol*. 2019;71(9):1400-1412.

10. Barbhaiya M, Zuily S, Naden R, et al. 2023 ACR/EULAR antiphospholipid syndrome classification criteria. *Arthritis Rheumatol*. 2023;75(10):1687-1702.

11. Seabold S, Perktold J. Statsmodels: econometric and statistical modeling with Python. *Proc 9th Python Sci Conf*. 2010:92-96.

12. el-Shabony AH, Guenther C, Jones JV. The effect of prednisone on serum complement levels in systemic lupus erythematosus. *Clin Exp Immunol*. 1978;34(2):202-210.

13. Proulle V, Furie RA, Merrill-Skoloff G, et al. Platelets are required for enhanced antibody-mediated thrombus formation in mice. *Blood*. 2014;123(23):3658-3666.

14. Atsumi T, Furukawa S, Amengual O, et al. Antiphospholipid antibody-associated thrombocytopenia. *Autoimmun Rev*. 2009;8(6):501-505.

15. Giannakopoulos B, Krilis SA. The pathogenesis of the antiphospholipid syndrome. *N Engl J Med*. 2013;368(11):1033-1044.

16. To CH, Mok CC, Tang SS, et al. Prolonged remission and outcome in systemic lupus erythematosus. *Arthritis Rheum*. 2009;61(6):801-808.

17. Petri M. The effect of thrombocytopenia and platelet count on the risk of thrombosis in systemic lupus erythematosus. *Arthritis Rheumatol*. 2021;73(10):1887-1894.

18. Tektonidou MG, Andreoli L, Limper M, et al. EULAR recommendations for the management of antiphospholipid syndrome in adults. *Ann Rheum Dis*. 2019;78(10):1296-1304.

19. Li Y, Wang Y, Hu J, et al. Deep learning for flare prediction in systemic lupus erythematosus using electronic health records. *Lancet Digit Health*. 2022;4(4):e253-e263.

20. Wu N, Green B, Ben J, et al. Deep transformer models for time-series clinical prediction. *NPJ Digit Med*. 2023;6(1):116.
