#!/usr/bin/env python3
"""Reproduce NPSLE analyses from data/SLEmatrix_merged.csv.

Outputs are written to Results/. The primary NPSLE label is based on the
neuropsychiatric ACR columns available in the merged matrix. SLEDAI neurologic
items are retained as source data but excluded from model predictors to avoid
label leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except Exception:  # pragma: no cover
    CoxPHFitter = None
    concordance_index = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

SEED = 20260504
ACR_NPSLE_COLS = ["acr_神经_癫痫", "acr_神经_精神", "acr_神经_脊髓炎", "acr_神经_脑血管"]
SLEDAI_NEURO_COLS = [
    "sledai_器质性脑病",
    "sledai_狼疮头痛",
    "sledai_癫痫",
    "sledai_精神症状",
    "sledai_脑血管意外",
    "sledai_视觉障碍",
    "sledai_颅神经病变",
]
ID_COL = "patient_SN"
DATE_COLS = ["检验报告日期", "_visit_date"]
VISIT_COL = "就诊标识（医渡云计算）"
METADATA_FEATURES = ["n_visits", "followup_years", "visit_date_parsed"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "SLEmatrix_merged.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "Results"

FEATURE_ALIASES = {
    "补体C3": "补体C3_静脉血_定量",
    "补体C4": "补体C4_静脉血_定量",
    "CRP": "C_反应蛋白_CRP__静脉血_定量",
    "粒细胞": "粒细胞计数_静脉血_定量",
    "白细胞": "白细胞计数_WBC#__静脉血_定量",
    "淋巴细胞": "淋巴细胞计数_Lymph#__静脉血_定量",
    "单核细胞": "单核细胞计数_Mono#__静脉血_定量",
    "血红蛋白": "血红蛋白_Hb__静脉血_定量",
    "血小板": "血小板计数_PLT#__静脉血_定量",
    "总蛋白": "总蛋白_TP__静脉血_定量",
    "白蛋白": "白蛋白_ALB__静脉血_定量",
    "肌酐": "肌酐_Crea__静脉血_定量",
    "胱抑素C": "胱抑素_Cys_C__静脉血_定量",
    "尿酸": "尿酸_UA__静脉血_定量",
    "eGFR": "估算肾小球滤过率_CKD_EPI公式__静脉血_定量",
    "APTT": "活化部分凝血活酶时间_APTT__静脉血_定量",
    "PT": "凝血酶原时间_PT__静脉血_定量",
    "PT_INR": "凝血酶原国际标准化比值_PT_INR__静脉血_定量",
    "LAC_NLR": "狼疮抗凝物标准化比值_LAC_NLR__静脉血_定量",
    "GGT": "γ_谷氨酰基转移酶_GGT__静脉血_定量",
    "甘油三酯": "甘油三酯_TG__静脉血_定量",
    "SLEDAI总分": "sledai_总分",
    "蛋白尿": "sledai_蛋白尿",
    "抗dsDNA": "acr_抗dsDNA阳性",
    "低补体": "acr_低补体",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def project_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype("string").str.strip()
    mapped = pd.Series(np.nan, index=s.index, dtype="float64")
    neg = text.str.contains(r"阴性|未见|正常|negative|^-$|^0$|无", case=False, na=False)
    pos = text.str.contains(r"阳性|positive|检出|异常|\+", case=False, na=False)
    mapped[neg] = 0.0
    mapped[pos] = 1.0
    plus = text.str.extract(r"(\d)\+")[0]
    mapped[plus.notna()] = plus.dropna().astype(float)
    num = pd.to_numeric(text.str.extract(r"([-+]?\d+(?:\.\d+)?)")[0], errors="coerce")
    mapped[num.notna()] = num[num.notna()]
    return mapped


def robust_date(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in DATE_COLS if c in df.columns), None)
    if col is None:
        raise ValueError("No visit date column found")
    return pd.to_datetime(df[col], errors="coerce")


def is_positive_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    num = pd.DataFrame({c: to_numeric_series(df[c]) for c in cols}, index=df.index)
    return num.fillna(0).gt(0).any(axis=1)


def winsorize_df(x: pd.DataFrame) -> pd.DataFrame:
    q01 = x.quantile(0.01, numeric_only=True)
    q99 = x.quantile(0.99, numeric_only=True)
    return x.clip(q01, q99, axis=1)


def clean_feature_name(name: str) -> str:
    s = name
    for old, new in [("_静脉血_定量", ""), ("__静脉血_定量", ""), ("_尿液_定量", ""), ("__尿液_定量", ""), ("_定量", "")]:
        s = s.replace(old, new)
    return s


def prepare_visit_matrix(df: pd.DataFrame, max_missing: float = 0.70) -> tuple[pd.DataFrame, list[str]]:
    exclude = set([ID_COL, "_patient_SN", VISIT_COL] + DATE_COLS + METADATA_FEATURES + ACR_NPSLE_COLS + SLEDAI_NEURO_COLS)
    exclude.update([c for c in df.columns if "脑脊液" in c])
    candidate_cols = [c for c in df.columns if c not in exclude]
    numeric = {}
    for c in candidate_cols:
        arr = to_numeric_series(df[c])
        if arr.notna().mean() >= (1 - max_missing) and arr.nunique(dropna=True) > 1:
            numeric[c] = arr
    x = pd.DataFrame(numeric, index=df.index)
    x = winsorize_df(x)
    return x, list(x.columns)


def aggregate_patient_features(x_visit: pd.DataFrame, df: pd.DataFrame, y_patient: pd.Series) -> pd.DataFrame:
    tmp = x_visit.copy()
    tmp[ID_COL] = df[ID_COL].values
    agg = tmp.groupby(ID_COL).mean(numeric_only=True)
    agg["n_visits"] = df.groupby(ID_COL).size()
    dates = robust_date(df)
    follow = dates.groupby(df[ID_COL]).agg(["min", "max"])
    agg["followup_years"] = (follow["max"] - follow["min"]).dt.days.clip(lower=0) / 365.25
    agg["npsle"] = y_patient.reindex(agg.index).astype(int)
    return agg


def metrics_from_probs(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def calibration_by_quantile(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "prob": prob}).dropna()
    if df.empty:
        return pd.DataFrame()
    try:
        df["bin"] = pd.qcut(df["prob"], q=min(n_bins, df["prob"].nunique()), duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["prob"], bins=min(n_bins, max(1, df["prob"].nunique())), duplicates="drop")
    cal = df.groupby("bin", observed=False).agg(
        n=("y_true", "size"),
        mean_pred=("prob", "mean"),
        observed_rate=("y_true", "mean"),
    )
    return cal.reset_index(drop=True).reset_index(names="bin")


def decision_curve_rows(y_true, prob, thresholds: np.ndarray) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    n = len(y)
    if n == 0:
        return []
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        odds = t / (1 - t)
        rows.append({
            "threshold": float(t),
            "net_benefit": float(tp / n - fp / n * odds),
            "treat_all_net_benefit": float(prevalence - (1 - prevalence) * odds),
            "treat_none_net_benefit": 0.0,
        })
    return rows


def save_prediction_diagnostics(pred: pd.DataFrame, out: Path, prefix: str, model_prob_cols: dict[str, str]) -> None:
    pred.to_csv(out / f"{prefix}_predictions.csv", index=False)
    thresholds = np.linspace(0.01, 0.99, 99)
    roc_rows, pr_rows, cal_rows, dca_rows = [], [], [], []
    y_true = pred["y_true"].astype(int).to_numpy()
    for model_name, col in model_prob_cols.items():
        prob = pd.to_numeric(pred[col], errors="coerce").to_numpy()
        valid = np.isfinite(prob)
        if valid.sum() == 0 or len(np.unique(y_true[valid])) < 2:
            continue
        auc = float(roc_auc_score(y_true[valid], prob[valid]))
        fpr, tpr, roc_thresholds = roc_curve(y_true[valid], prob[valid])
        roc_rows.extend({
            "model": model_name,
            "fpr": float(fp),
            "tpr": float(tp),
            "threshold": float(th),
            "auc": auc,
        } for fp, tp, th in zip(fpr, tpr, roc_thresholds))

        ap = float(average_precision_score(y_true[valid], prob[valid]))
        precision, recall, pr_thresholds = precision_recall_curve(y_true[valid], prob[valid])
        pr_thresholds = np.r_[pr_thresholds, np.nan]
        pr_rows.extend({
            "model": model_name,
            "precision": float(prec),
            "recall": float(rec),
            "threshold": float(th) if np.isfinite(th) else np.nan,
            "average_precision": ap,
        } for prec, rec, th in zip(precision, recall, pr_thresholds))

        cal = calibration_by_quantile(y_true[valid], prob[valid])
        for row in cal.to_dict(orient="records"):
            cal_rows.append({"model": model_name, **row})

        for row in decision_curve_rows(y_true[valid], prob[valid], thresholds):
            dca_rows.append({"model": model_name, **row})

    pd.DataFrame(roc_rows).to_csv(out / f"{prefix}_roc_curve.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(out / f"{prefix}_precision_recall_curve.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / f"{prefix}_calibration_curve.csv", index=False)
    pd.DataFrame(dca_rows).to_csv(out / f"{prefix}_decision_curve.csv", index=False)


def model_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def run_univariate(patient_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    y = patient_df["npsle"].astype(int)
    rows = []
    for c in [c for c in patient_df.columns if c != "npsle" and c not in METADATA_FEATURES]:
        a = patient_df.loc[y == 1, c].dropna()
        b = patient_df.loc[y == 0, c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        except Exception:
            u_p = np.nan
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        effect = (a.mean() - b.mean()) / pooled if pooled and not np.isnan(pooled) and pooled > 0 else np.nan
        rows.append({
            "feature": c,
            "feature_clean": clean_feature_name(c),
            "n_npsle": int(a.shape[0]),
            "n_non_npsle": int(b.shape[0]),
            "mean_npsle": float(a.mean()),
            "sd_npsle": float(a.std(ddof=1)),
            "mean_non_npsle": float(b.mean()),
            "sd_non_npsle": float(b.std(ddof=1)),
            "standardized_effect": float(effect) if not np.isnan(effect) else None,
            "p_value": float(u_p) if not np.isnan(u_p) else None,
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        mask = res["p_value"].notna()
        res.loc[mask, "fdr_p"] = multipletests(res.loc[mask, "p_value"], method="fdr_bh")[1]
        res["significant_fdr_0.05"] = res["fdr_p"] < 0.05
        res = res.sort_values(["fdr_p", "p_value"]).reset_index(drop=True)
    res.to_csv(out / "univariate_results.csv", index=False)
    save_json(out / "univariate_summary.json", {
        "tested_features": int(len(res)),
        "significant_fdr_0.05": int(res.get("significant_fdr_0.05", pd.Series(dtype=bool)).sum()),
        "top10": res.head(10).to_dict(orient="records"),
    })
    return res


def run_cross_sectional(patient_df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = ["npsle"] + [c for c in METADATA_FEATURES if c in patient_df.columns]
    x = patient_df.drop(columns=drop_cols)
    y = patient_df["npsle"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=SEED, stratify=y)
    models = {
        "LASSO_logistic": model_pipeline(LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced", max_iter=5000), scale=True),
        "RandomForest": model_pipeline(RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=5, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        "GradientBoosting": model_pipeline(GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=SEED)),
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    fitted = {}
    pred = pd.DataFrame({ID_COL: x_test.index.astype(str), "y_true": y_test.astype(int).values})
    for name, pipe in models.items():
        scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        if name == "GradientBoosting":
            weights = compute_sample_weight("balanced", y_train)
            pipe.fit(x_train, y_train, model__sample_weight=weights)
        else:
            pipe.fit(x_train, y_train)
        prob = pipe.predict_proba(x_test)[:, 1]
        pred[f"{name}_prob"] = prob
        m = metrics_from_probs(y_test.values, prob)
        m.update({"model": name, "cv_auc_mean": float(scores.mean()), "cv_auc_sd": float(scores.std(ddof=1))})
        rows.append(m)
        fitted[name] = pipe
    perf = pd.DataFrame(rows).sort_values("auc", ascending=False)
    perf.to_csv(out / "model_performance.csv", index=False)
    save_json(out / "model_performance.json", perf.to_dict(orient="records"))
    save_prediction_diagnostics(
        pred,
        out,
        "cross_sectional",
        {name: f"{name}_prob" for name in models},
    )

    best_name = perf.iloc[0]["model"]
    best = fitted[best_name]
    feature_rows = []
    model = best.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        vals = np.zeros(x.shape[1])
    for f, v in sorted(zip(x.columns, vals), key=lambda z: z[1], reverse=True):
        feature_rows.append({"feature": f, "feature_clean": clean_feature_name(f), "importance": float(v), "model": best_name})
    imp = pd.DataFrame(feature_rows)
    imp.to_csv(out / "feature_importance.csv", index=False)
    return perf, imp


def run_first_visit(df: pd.DataFrame, x_visit: pd.DataFrame, y_visit: pd.Series, out: Path) -> pd.DataFrame:
    dates = robust_date(df)
    order = pd.DataFrame({ID_COL: df[ID_COL], "_date": dates}).sort_values([ID_COL, "_date"])
    first_idx = order.groupby(ID_COL).head(1).index
    x = x_visit.loc[first_idx].copy()
    y = y_visit.loc[first_idx].astype(int)
    x.index = df.loc[first_idx, ID_COL].values
    model = model_pipeline(LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced", max_iter=5000), scale=True)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    aucs = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=SEED, stratify=y)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = pd.DataFrame({
        ID_COL: x_test.index.astype(str),
        "y_true": y_test.astype(int).values,
        "first_visit_LASSO_logistic_prob": prob,
    })
    save_prediction_diagnostics(
        pred,
        out,
        "first_visit",
        {"first_visit_LASSO_logistic": "first_visit_LASSO_logistic_prob"},
    )
    row = metrics_from_probs(y_test.values, prob)
    row.update({"model": "first_visit_LASSO_logistic", "cv_auc_mean": float(aucs.mean()), "cv_auc_sd": float(aucs.std(ddof=1)), "n": int(len(y)), "events": int(y.sum())})
    res = pd.DataFrame([row])
    res.to_csv(out / "first_visit_validation_results.csv", index=False)
    save_json(out / "first_visit_validation_results.json", row)
    return res


def linear_slope(years: np.ndarray, vals: np.ndarray) -> float | None:
    mask = np.isfinite(years) & np.isfinite(vals)
    if mask.sum() < 3 or np.unique(years[mask]).size < 2:
        return None
    return float(stats.linregress(years[mask], vals[mask]).slope)


def run_trajectory_cox(df: pd.DataFrame, x_visit: pd.DataFrame, y_visit: pd.Series, y_patient: pd.Series, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = robust_date(df)
    tmp = pd.DataFrame({ID_COL: df[ID_COL], "date": dates, "visit_npsle": y_visit.astype(int)})
    for label, col in [("C3", FEATURE_ALIASES["补体C3"]), ("CRP", FEATURE_ALIASES["CRP"]), ("Granulocyte", FEATURE_ALIASES["粒细胞"]), ("CysC", FEATURE_ALIASES["胱抑素C"]), ("SLEDAI", FEATURE_ALIASES["SLEDAI总分"]), ("LAC_NLR", FEATURE_ALIASES["LAC_NLR"]), ("UA", FEATURE_ALIASES["尿酸"]), ("eGFR", FEATURE_ALIASES["eGFR"]), ("APTT", FEATURE_ALIASES["APTT"])]:
        if col in df.columns:
            tmp[label] = to_numeric_series(df[col])
    first_date = tmp.groupby(ID_COL)["date"].min()
    tmp = tmp.join(first_date.rename("first_date"), on=ID_COL)
    tmp["years"] = (tmp["date"] - tmp["first_date"]).dt.days / 365.25
    rows = []
    event_date = tmp.loc[tmp["visit_npsle"] == 1].groupby(ID_COL)["date"].min()
    last_date = tmp.groupby(ID_COL)["date"].max()
    for pid, g in tmp.groupby(ID_COL):
        row = {ID_COL: pid, "n_visits": int(len(g)), "npsle": int(y_patient.get(pid, 0))}
        for label in ["C3", "CRP", "Granulocyte"]:
            if label in g:
                row[f"{label}_slope"] = linear_slope(g["years"].to_numpy(dtype=float), g[label].to_numpy(dtype=float))
                row[f"{label}_baseline"] = g.sort_values("date")[label].dropna().iloc[0] if g[label].notna().any() else np.nan
                row[f"{label}_mean"] = g[label].mean()
        for label in ["CysC", "SLEDAI", "LAC_NLR", "UA", "eGFR", "APTT"]:
            if label in g:
                row[f"{label}_baseline"] = g.sort_values("date")[label].dropna().iloc[0] if g[label].notna().any() else np.nan
        end = event_date.get(pid, pd.NaT) if row["npsle"] else pd.NaT
        if pd.isna(end):
            end = last_date.get(pid, pd.NaT)
        start = first_date.get(pid, pd.NaT)
        row["duration_years"] = max((end - start).days / 365.25, 0.001) if pd.notna(start) and pd.notna(end) else np.nan
        rows.append(row)
    slopes = pd.DataFrame(rows).set_index(ID_COL)
    slopes.to_csv(out / "trajectory_slopes.csv")

    comps = []
    for marker in ["C3", "CRP", "Granulocyte"]:
        col = f"{marker}_slope"
        if col not in slopes:
            continue
        a = slopes.loc[slopes["npsle"] == 1, col].dropna()
        b = slopes.loc[slopes["npsle"] == 0, col].dropna()
        comps.append({
            "marker": marker,
            "n_npsle": int(len(a)), "n_non_npsle": int(len(b)),
            "mean_slope_npsle": float(a.mean()) if len(a) else None,
            "sd_slope_npsle": float(a.std(ddof=1)) if len(a) > 1 else None,
            "mean_slope_non_npsle": float(b.mean()) if len(b) else None,
            "sd_slope_non_npsle": float(b.std(ddof=1)) if len(b) > 1 else None,
            "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else None,
            "ttest_p": float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue) if len(a) > 1 and len(b) > 1 else None,
        })
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(out / "slope_comparison_npsle.csv", index=False)

    cox_df = slopes[["duration_years", "npsle"]].copy()
    for c in ["C3_slope", "CRP_slope", "Granulocyte_slope", "C3_baseline", "CRP_baseline", "Granulocyte_baseline", "CysC_baseline", "SLEDAI_baseline", "LAC_NLR_baseline", "UA_baseline", "eGFR_baseline", "APTT_baseline"]:
        if c in slopes.columns:
            cox_df[c] = slopes[c]
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration_years", "npsle"])
    feature_cols = [c for c in cox_df.columns if c not in ["duration_years", "npsle"]]
    cox_df[feature_cols] = SimpleImputer(strategy="median").fit_transform(cox_df[feature_cols])
    cox_df[feature_cols] = StandardScaler().fit_transform(cox_df[feature_cols])
    cox_rows = []
    c_index = None
    if CoxPHFitter is not None and cox_df["npsle"].sum() > 5:
        try:
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(cox_df, duration_col="duration_years", event_col="npsle")
            s = cph.summary.reset_index().rename(columns={"covariate": "variable", "exp(coef)": "HR", "p": "p_value"})
            cox_rows = s[["variable", "HR", "exp(coef) lower 95%", "exp(coef) upper 95%", "p_value"]].to_dict(orient="records")
            c_index = float(cph.concordance_index_)
        except Exception as e:
            cox_rows = [{"error": str(e)}]
    cox_out = pd.DataFrame(cox_rows)
    cox_out.to_csv(out / "cox_hazard_ratios.csv", index=False)
    save_json(out / "cox_model_summary.json", {"c_index": c_index, "n": int(len(cox_df)), "events": int(cox_df["npsle"].sum()), "variables": feature_cols})

    patterns = build_trajectory_patterns(slopes)
    patterns.to_csv(out / "trajectory_patterns.csv", index=False)
    return comp_df, cox_out


def build_trajectory_patterns(slopes: pd.DataFrame) -> pd.DataFrame:
    req = ["C3_baseline", "CRP_slope", "Granulocyte_slope", "C3_slope"]
    if not all(c in slopes for c in req):
        return pd.DataFrame()
    work = slopes.copy()
    c3_low = work["C3_baseline"] < work["C3_baseline"].median(skipna=True)
    crp_up = work["CRP_slope"] > work["CRP_slope"].quantile(0.75)
    gran_up = work["Granulocyte_slope"] > work["Granulocyte_slope"].quantile(0.75)
    c3_down = work["C3_slope"] < work["C3_slope"].quantile(0.25)
    conditions = [
        ("促炎症型(CRP↑+粒细胞↑)", crp_up & gran_up),
        ("低C3+高CRP+高粒细胞", c3_low & crp_up & gran_up),
        ("低C3下降型", c3_low & c3_down),
        ("低C3+高粒细胞", c3_low & gran_up),
        ("稳定型", ~(crp_up | gran_up | c3_down)),
    ]
    rows = []
    for name, mask in conditions:
        sub = work.loc[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "pattern": name,
            "n": int(len(sub)),
            "npsle_events": int(sub["npsle"].sum()),
            "npsle_rate_percent": float(sub["npsle"].mean() * 100),
            "median_duration_years": float(sub["duration_years"].median()),
        })
    return pd.DataFrame(rows).sort_values("npsle_rate_percent", ascending=False)


def score_ews(row: pd.Series) -> int:
    def val(alias):
        col = FEATURE_ALIASES.get(alias)
        return row.get(col, np.nan)
    score = 0
    # 9-item transparent bedside score, 0-3 per domain. Thresholds match the manuscript scale where feasible.
    c3 = val("补体C3")
    score += 3 if pd.notna(c3) and c3 < 0.3 else 2 if pd.notna(c3) and c3 < 0.5 else 1 if pd.notna(c3) and c3 < 0.8 else 0
    c4 = val("补体C4")
    score += 3 if pd.notna(c4) and c4 < 0.05 else 2 if pd.notna(c4) and c4 < 0.10 else 1 if pd.notna(c4) and c4 < 0.16 else 0
    crp = val("CRP")
    score += 3 if pd.notna(crp) and crp >= 50 else 2 if pd.notna(crp) and crp >= 20 else 1 if pd.notna(crp) and crp >= 8 else 0
    cysc = val("胱抑素C")
    score += 3 if pd.notna(cysc) and cysc >= 2.0 else 2 if pd.notna(cysc) and cysc >= 1.5 else 1 if pd.notna(cysc) and cysc >= 1.1 else 0
    ua = val("尿酸")
    score += 3 if pd.notna(ua) and ua >= 600 else 2 if pd.notna(ua) and ua >= 480 else 1 if pd.notna(ua) and ua >= 360 else 0
    egfr = val("eGFR")
    score += 3 if pd.notna(egfr) and egfr < 30 else 2 if pd.notna(egfr) and egfr < 60 else 1 if pd.notna(egfr) and egfr < 90 else 0
    aptt = val("APTT")
    score += 3 if pd.notna(aptt) and aptt >= 60 else 2 if pd.notna(aptt) and aptt >= 45 else 1 if pd.notna(aptt) and aptt >= 35 else 0
    lac = val("LAC_NLR")
    score += 3 if pd.notna(lac) and lac >= 2.0 else 2 if pd.notna(lac) and lac >= 1.5 else 1 if pd.notna(lac) and lac >= 1.2 else 0
    sledai = val("SLEDAI总分")
    score += 3 if pd.notna(sledai) and sledai >= 12 else 2 if pd.notna(sledai) and sledai >= 8 else 1 if pd.notna(sledai) and sledai >= 4 else 0
    return int(score)


def run_ews(df: pd.DataFrame, y_visit: pd.Series, out: Path) -> pd.DataFrame:
    dates = robust_date(df)
    first_idx = pd.DataFrame({ID_COL: df[ID_COL], "_date": dates}).sort_values([ID_COL, "_date"]).groupby(ID_COL).head(1).index
    first = df.loc[first_idx].copy()
    y = y_visit.loc[first_idx].astype(int).values
    first["ews_score"] = first.apply(score_ews, axis=1)
    score = first["ews_score"].astype(float).values
    prob = score / 27.0
    pred = pd.DataFrame({
        ID_COL: first[ID_COL].astype(str).values,
        "y_true": y.astype(int),
        "ews_score": first["ews_score"].astype(int).values,
        "EWS_prob": prob,
    })
    save_prediction_diagnostics(pred, out, "ews", {"EWS": "EWS_prob"})
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    thresholds = sorted(first["ews_score"].dropna().unique())
    best = None
    for t in thresholds:
        m = metrics_from_probs(y, prob, threshold=t / 27.0)
        youden = (m["sensitivity"] or 0) + (m["specificity"] or 0) - 1
        if best is None or youden > best["youden"]:
            best = {**m, "score_cutoff": int(t), "youden": float(youden)}
    cats = pd.cut(first["ews_score"], bins=[-1, 5, 10, 15, 27], labels=["low", "moderate", "high", "very_high"])
    strat = first.assign(npsle=y, risk_group=cats).groupby("risk_group", observed=False).agg(n=(ID_COL, "count"), events=("npsle", "sum"), mean_score=("ews_score", "mean"))
    strat["event_rate_percent"] = strat["events"] / strat["n"] * 100
    strat = strat.reset_index()
    strat.to_csv(out / "ews_risk_stratification.csv", index=False)
    summary = {
        "auc": auc,
        "n": int(len(y)),
        "events": int(y.sum()),
        "best_cutoff": best,
        "risk_groups": strat.to_dict(orient="records"),
        "prediction_outputs": {
            "predictions": "ews_predictions.csv",
            "roc": "ews_roc_curve.csv",
            "precision_recall": "ews_precision_recall_curve.csv",
            "calibration": "ews_calibration_curve.csv",
            "decision_curve": "ews_decision_curve.csv",
        },
    }
    save_json(out / "ews_performance.json", summary)
    return strat


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int, max_len: int):
        super().__init__()
        self.inp = nn.Linear(n_features, 64)
        self.pos = nn.Parameter(torch.zeros(1, max_len, 64))
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.3, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        z = self.inp(x) + self.pos[:, : x.shape[1], :]
        z = self.enc(z)
        return self.head(z[:, -1]).squeeze(-1)


def build_sequences(df: pd.DataFrame, feature_cols: list[str], y_patient: pd.Series, seq_len: int = 8, max_patients: int | None = None):
    dates = robust_date(df)
    use = pd.DataFrame({ID_COL: df[ID_COL], "date": dates})
    for c in feature_cols:
        use[c] = to_numeric_series(df[c])
    med = use[feature_cols].median(numeric_only=True)
    use[feature_cols] = use[feature_cols].fillna(med)
    patients = y_patient.index.to_list()
    if max_patients and len(patients) > max_patients:
        rng = np.random.default_rng(SEED)
        pos = [p for p in patients if y_patient.loc[p] == 1]
        neg = [p for p in patients if y_patient.loc[p] == 0]
        keep_neg = rng.choice(neg, size=max(0, max_patients - len(pos)), replace=False).tolist()
        patients = pos + keep_neg
    seqs, labels, patient_ids = [], [], []
    for pid in patients:
        g = use.loc[use[ID_COL] == pid].sort_values("date")
        vals = g[feature_cols].to_numpy(dtype="float32")
        if vals.shape[0] == 0:
            continue
        if vals.shape[0] >= seq_len:
            vals = vals[-seq_len:]
        else:
            pad = np.repeat(vals[:1], seq_len - vals.shape[0], axis=0)
            vals = np.vstack([pad, vals])
        seqs.append(vals)
        labels.append(int(y_patient.loc[pid]))
        patient_ids.append(str(pid))
    x = np.stack(seqs)
    y = np.asarray(labels, dtype="int64")
    mu = np.nanmean(x.reshape(-1, x.shape[-1]), axis=0)
    sd = np.nanstd(x.reshape(-1, x.shape[-1]), axis=0)
    sd[sd == 0] = 1
    x = (x - mu) / sd
    return x.astype("float32"), y, feature_cols, patient_ids


def train_torch_model(model, x_train, y_train, x_val, y_val, epochs: int, batch_size: int, lr: float = 1e-3) -> tuple[dict, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.astype("float32")))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_loss, best_state, patience, waits = float("inf"), None, 8, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val, device=device))
            val_loss = float(crit(logits, torch.tensor(y_val.astype("float32"), device=device)).item())
            prob = torch.sigmoid(logits).detach().cpu().numpy()
        history.append({"epoch": epoch, "train_loss": total / len(ds), "val_loss": val_loss, "val_auc": float(roc_auc_score(y_val, prob))})
        if val_loss < best_loss:
            best_loss, waits = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waits += 1
            if waits >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x_val, device=device))).detach().cpu().numpy()
    return {"history": history, "epochs_trained": len(history), "device": str(device)}, prob


def run_deep_learning(df: pd.DataFrame, y_patient: pd.Series, out: Path, epochs: int = 25, max_patients: int | None = None) -> None:
    dl_dir = out / "lstm_transformer"
    dl_dir.mkdir(parents=True, exist_ok=True)
    if torch is None:
        save_json(dl_dir / "model_comparison.json", {"status": "skipped", "reason": "PyTorch not available"})
        return
    feat_names = ["尿酸", "eGFR", "甘油三酯", "补体C4", "肌酐", "胱抑素C", "补体C3", "CRP", "APTT", "SLEDAI总分", "粒细胞", "白细胞"]
    feature_cols = [FEATURE_ALIASES[n] for n in feat_names if FEATURE_ALIASES.get(n) in df.columns]
    x, y, cols, patient_ids = build_sequences(df, feature_cols, y_patient, seq_len=8, max_patients=max_patients)
    patient_ids = np.asarray(patient_ids)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.20, random_state=SEED, stratify=y)
    x_train, y_train, x_test, y_test = x[train_idx], y[train_idx], x[test_idx], y[test_idx]
    models = {
        "Bi-LSTM": LSTMClassifier(x.shape[-1]),
        "Transformer": TransformerClassifier(x.shape[-1], x.shape[1]),
    }
    results, probs = {}, {}
    for name, model in models.items():
        hist, prob = train_torch_model(model, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=64)
        results[name] = {**metrics_from_probs(y_test, prob), **hist}
        probs[name] = prob
    ens = (probs["Bi-LSTM"] + probs["Transformer"]) / 2
    probs["Ensemble"] = ens
    results["Ensemble"] = metrics_from_probs(y_test, ens)
    pred = pd.DataFrame({
        ID_COL: patient_ids[test_idx],
        "y_true": y_test.astype(int),
        "Bi-LSTM_prob": probs["Bi-LSTM"],
        "Transformer_prob": probs["Transformer"],
        "Ensemble_prob": probs["Ensemble"],
    })
    save_prediction_diagnostics(
        pred,
        dl_dir,
        "deep_learning",
        {
            "Bi-LSTM": "Bi-LSTM_prob",
            "Transformer": "Transformer_prob",
            "Ensemble": "Ensemble_prob",
        },
    )
    results["metadata"] = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "test_n": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "features": cols,
        "seq_len": 8,
        "epochs_requested": epochs,
        "prediction_outputs": {
            "predictions": "deep_learning_predictions.csv",
            "roc": "deep_learning_roc_curve.csv",
            "precision_recall": "deep_learning_precision_recall_curve.csv",
            "calibration": "deep_learning_calibration_curve.csv",
            "decision_curve": "deep_learning_decision_curve.csv",
        },
    }
    save_json(dl_dir / "model_comparison.json", results)

    # Permutation importance on Transformer test set, one pass per feature.
    base = results["Transformer"]["auc"]
    imps = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Re-train a fresh Transformer to access model object for importance with deterministic split.
    model = TransformerClassifier(x.shape[-1], x.shape[1])
    _, _ = train_torch_model(model, x_train, y_train, x_test, y_test, epochs=max(3, min(epochs, 10)), batch_size=64)
    model = model.to(device).eval()
    rng = np.random.default_rng(SEED)
    for j, c in enumerate(cols):
        xp = x_test.copy()
        order = rng.permutation(xp.shape[0])
        xp[:, :, j] = xp[order, :, j]
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(xp, device=device))).cpu().numpy()
        auc = float(roc_auc_score(y_test, p))
        imps.append({"feature": c, "feature_clean": clean_feature_name(c), "delta_auc": float(base - auc), "permuted_auc": auc})
    pd.DataFrame(imps).sort_values("delta_auc", ascending=False).to_csv(dl_dir / "feature_importance_transformer.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_RESULTS))
    parser.add_argument("--dl-epochs", type=int, default=25)
    parser.add_argument("--dl-max-patients", type=int, default=None, help="Optional down-sampling for quick tests")
    args = parser.parse_args()

    started = time.time()
    set_seed(SEED)
    data_path = Path(args.data)
    data_label = project_display_path(data_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, low_memory=False)
    if ID_COL not in df.columns:
        raise ValueError(f"Missing patient id column: {ID_COL}")
    df[ID_COL] = df[ID_COL].astype(str)
    df["visit_date_parsed"] = robust_date(df)
    y_visit = is_positive_frame(df, ACR_NPSLE_COLS).astype(int)
    y_patient = y_visit.groupby(df[ID_COL]).max().astype(int)

    x_visit, features = prepare_visit_matrix(df)
    patient_df = aggregate_patient_features(x_visit, df, y_patient)
    patient_df.to_csv(out / "patient_level_feature_matrix.csv")

    summary = {
        "data_source": data_label,
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "rows_visits": int(len(df)),
        "patients": int(df[ID_COL].nunique()),
        "visit_level_npsle_events_acr_definition": int(y_visit.sum()),
        "patient_level_npsle_events_acr_definition": int(y_patient.sum()),
        "npsle_label_columns": ACR_NPSLE_COLS,
        "sledai_neuro_columns_excluded_from_predictors": SLEDAI_NEURO_COLS,
        "candidate_features_after_missing_filter": int(len(features)),
        "date_min": str(df["visit_date_parsed"].min()),
        "date_max": str(df["visit_date_parsed"].max()),
    }
    save_json(out / "analysis_summary.json", summary)

    descriptive = patient_df["npsle"].value_counts().rename(index={0: "non_npsle", 1: "npsle"}).to_dict()
    save_json(out / "descriptive_stats.json", {"patient_label_counts": descriptive, "visits": summary["rows_visits"], "patients": summary["patients"]})

    uni = run_univariate(patient_df, out)
    perf, imp = run_cross_sectional(patient_df, out)
    first = run_first_visit(df, x_visit, y_visit, out)
    slopes, cox = run_trajectory_cox(df, x_visit, y_visit, y_patient, out)
    ews = run_ews(df, y_visit, out)
    run_deep_learning(df, y_patient, out, epochs=args.dl_epochs, max_patients=args.dl_max_patients)

    notes = [
        "# NPSLE Analysis Reproduction Notes",
        "",
        f"Data source: `{data_label}`",
        f"Visits: {summary['rows_visits']}; patients: {summary['patients']}",
        f"Primary NPSLE label: any positive value in {', '.join(ACR_NPSLE_COLS)}.",
        f"Visit-level events: {summary['visit_level_npsle_events_acr_definition']}; patient-level events: {summary['patient_level_npsle_events_acr_definition']}.",
        "",
        "The current merged CSV differs from the manuscript's historical Results directory. These outputs are recomputed directly from the merged file and should be treated as the authoritative reproduction for this data source.",
        "",
        "Key regenerated outputs:",
        f"- Univariate significant features (FDR<0.05): {int(uni.get('significant_fdr_0.05', pd.Series(dtype=bool)).sum())}",
        f"- Best cross-sectional model: {perf.iloc[0]['model']} AUC={perf.iloc[0]['auc']:.3f}",
        f"- First-visit LASSO CV AUC={first.iloc[0]['cv_auc_mean']:.3f}+/-{first.iloc[0]['cv_auc_sd']:.3f}",
        f"- EWS risk groups written to `ews_risk_stratification.csv`.",
    ]
    (out / "reproduction_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    summary["runtime_seconds"] = round(time.time() - started, 2)
    save_json(out / "analysis_summary.json", summary)
    (out / "run_log.txt").write_text(f"Completed in {summary['runtime_seconds']} seconds\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
