# NPSLE Risk Prediction Analysis

This project reproduces the NPSLE risk prediction analyses from the merged SLE matrix.

## Project Layout

- `data/SLEmatrix_merged.csv`: merged longitudinal SLE visit matrix.
- `scripts/run_npsle_pipeline.py`: main reproducible analysis pipeline.
- `scripts/generate_npsle_figures.py`: manuscript figure generation from `Results/`.
- `Results/`: regenerated model outputs, summaries, diagnostics, and figures.
- `Reports/论文草稿-统一投稿版.md`: current manuscript draft and authoritative reporting version.
- `Reports/NPSLE风险预测综合研究报告：横断面、纵向轨迹与深度学习多维度分析.md`: historical report retained for comparison only.

## Reproduce

Create an environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-deep-learning.txt
```

Run the full analysis:

```bash
.venv/bin/python scripts/run_npsle_pipeline.py
```

Regenerate figures:

```bash
.venv/bin/python scripts/generate_npsle_figures.py
```

The main script resolves `data/` and `Results/` relative to this project directory, so it can be launched from either the project root or another working directory.

## Current Analysis Definition

- Data source: `data/SLEmatrix_merged.csv`.
- Cohort window: 2008-08-17 to 2025-02-14.
- Primary NPSLE endpoint: any positive value in `acr_神经_癫痫`, `acr_神经_精神`, `acr_神经_脊髓炎`, or `acr_神经_脑血管`.
- Leakage control: SLEDAI neurologic items are excluded from predictors.
- Random seed: `20260504`.

The historical comprehensive report contains older numbers from a previous data/result version. Use the unified manuscript draft and `Results/` files for current reporting.
