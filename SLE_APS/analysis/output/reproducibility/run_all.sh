#!/bin/bash
# SLE-APS Analysis Pipeline - One-click Reproducibility
# Usage: bash run_all.sh

set -e
echo "SLE-APS Analysis Pipeline"
echo "========================"

# Check environment
python3 -c "import pandas, numpy, scipy, sklearn, xgboost, lightgbm, torch, shap, lifelines, statsmodels" 2>/dev/null || { echo "Installing requirements..."; pip install -r requirements.txt; }

echo "Step 1: Phase 0 - Data Quality"
python3 ../../analysis/scripts/phase0_data_quality.py

echo "Step 2: Phase 1 - Cohort Definition"
python3 ../../analysis/scripts/phase1_cohort_definition.py

echo "Step 3: Phase 2 - Descriptive Analysis"
python3 ../../analysis/scripts/phase2_descriptive.py

echo "Step 4: Phase 3 - Model Development"
python3 ../../analysis/scripts/phase3_modeling.py

echo "Step 5: Phase 4 - Model Validation"
python3 ../../analysis/scripts/phase4_validation.py

echo "Step 6: Phase 5 - Longitudinal Analysis"
python3 ../../analysis/scripts/phase5_longitudinal.py

echo "Step 7: Phase 6 - Deep Learning"
python3 ../../analysis/scripts/phase6_deep_learning.py

echo "Step 8: Phase 7 - Clinical Scorecard"
python3 ../../analysis/scripts/phase7_ews_nomogram.py

echo "Step 9: Phase 8 - Manuscript Generation"
python3 ../../analysis/scripts/phase8_manuscript_figures.py

echo ""
echo "All analyses complete! Check analysis/output/ for results."
