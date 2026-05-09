#!/usr/bin/env python3
"""Generate core manuscript figures from regenerated NPSLE outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_project_root(start: Path) -> Path:
    for parent in [start.parent, *start.parents]:
        if (parent / "Results" / "Results").exists() or (parent / "Results").exists():
            if (parent / "data" / "data").exists() or (parent / "data").exists():
                return parent
    return Path.cwd()


ROOT = find_project_root(Path(__file__).resolve())
RESULTS = ROOT / "Results" / "Results" if (ROOT / "Results" / "Results").exists() else ROOT / "Results"
FIGURES = RESULTS / "figures"

PATTERN_LABELS = {
    "低C3+高CRP+高粒细胞": "Low C3 + high CRP/granulocytes",
    "促炎症型(CRP↑+粒细胞↑)": "Pro-inflammatory pattern",
    "低C3+高粒细胞": "Low C3 + high granulocytes",
    "低C3下降型": "Low C3 with declining C3",
    "稳定型": "Stable pattern",
}

FEATURE_LABELS = {
    "sledai_总分": "SLEDAI total score",
    "补体C3": "Complement C3",
    "估算肾小球滤过率_CKD_EPI公式_": "eGFR",
    "胱抑素_Cys_C_": "Cystatin C",
    "活化部分凝血活酶时间_APTT_": "APTT",
    "尿酸_UA_": "Uric acid",
    "甘油三酯_TG_": "Triglycerides",
    "C_反应蛋白_CRP_": "CRP",
    "白细胞计数_WBC#_": "White blood cell count",
    "粒细胞计数": "Granulocyte count",
}

MODEL_LABELS = {
    "LASSO_logistic": "LASSO logistic",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
    "first_visit_LASSO_logistic": "First-visit LASSO",
    "EWS": "EWS",
    "Bi-LSTM": "Bi-LSTM",
    "Transformer": "Transformer",
    "Ensemble": "DL ensemble",
}

CURVE_SOURCES = {
    "cross_sectional": RESULTS,
    "first_visit": RESULTS,
    "ews": RESULTS,
    "deep_learning": RESULTS / "lstm_transformer",
}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def display_model(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def read_curve(kind: str) -> pd.DataFrame:
    pieces = []
    suffix = {
        "roc": "_roc_curve.csv",
        "pr": "_precision_recall_curve.csv",
        "calibration": "_calibration_curve.csv",
        "dca": "_decision_curve.csv",
    }[kind]
    for prefix, base in CURVE_SOURCES.items():
        path = base / f"{prefix}{suffix}"
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                df["source"] = prefix
                pieces.append(df)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def plot_model_performance() -> None:
    ml = pd.read_csv(RESULTS / "model_performance.csv")[["model", "auc"]]
    with (RESULTS / "lstm_transformer" / "model_comparison.json").open(encoding="utf-8") as f:
        dl = json.load(f)
    dl_rows = [{"model": k, "auc": v["auc"]} for k, v in dl.items() if isinstance(v, dict) and "auc" in v]
    ews = json.loads((RESULTS / "ews_performance.json").read_text(encoding="utf-8"))
    cox = json.loads((RESULTS / "cox_model_summary.json").read_text(encoding="utf-8"))
    rows = pd.concat(
        [
            ml.assign(group="Cross-sectional ML"),
            pd.DataFrame(dl_rows).assign(group="Temporal deep learning"),
            pd.DataFrame([{"model": "EWS", "auc": ews["auc"], "group": "Clinical score"}]),
            pd.DataFrame([{"model": "Cox PH", "auc": cox["c_index"], "group": "Survival model"}]),
        ],
        ignore_index=True,
    ).sort_values("auc", ascending=False)

    plt.figure(figsize=(7.2, 4.8))
    ax = sns.barplot(data=rows, y="model", x="auc", hue="group", dodge=False, palette="Set2")
    ax.set_xlim(0.72, 0.92)
    ax.set_xlabel("AUC / C-index")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower right", frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    savefig(FIGURES / "fig1_model_performance.png")


def plot_trajectory_patterns() -> None:
    df = pd.read_csv(RESULTS / "trajectory_patterns.csv").sort_values("npsle_rate_percent")
    df["pattern_label"] = df["pattern"].map(PATTERN_LABELS).fillna(df["pattern"])
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.barplot(data=df, y="pattern_label", x="npsle_rate_percent", color="#4C78A8")
    ax.set_xlabel("NPSLE event rate (%)")
    ax.set_ylabel("")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.npsle_rate_percent + 0.3, i, f"n={row.n}", va="center", fontsize=8)
    savefig(FIGURES / "fig2_trajectory_patterns.png")


def plot_ews_risk() -> None:
    df = pd.read_csv(RESULTS / "ews_risk_stratification.csv")
    order = ["low", "moderate", "high", "very_high"]
    df["risk_group"] = pd.Categorical(df["risk_group"], categories=order, ordered=True)
    df = df.sort_values("risk_group")
    plt.figure(figsize=(6.4, 4.6))
    ax = sns.barplot(data=df, x="risk_group", y="event_rate_percent", hue="risk_group", palette="Reds", legend=False)
    ax.set_xlabel("EWS risk group")
    ax.set_ylabel("NPSLE event rate (%)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)
    savefig(FIGURES / "fig3_ews_risk_stratification.png")


def plot_transformer_importance() -> None:
    df = pd.read_csv(RESULTS / "lstm_transformer" / "feature_importance_transformer.csv")
    df = df.sort_values("delta_auc", ascending=False).head(10).sort_values("delta_auc")
    df["feature_label"] = df["feature_clean"].map(FEATURE_LABELS).fillna(df["feature_clean"])
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.barplot(data=df, y="feature_label", x="delta_auc", color="#59A14F")
    ax.set_xlabel("Permutation importance (Delta AUC)")
    ax.set_ylabel("")
    savefig(FIGURES / "fig4_transformer_permutation_importance.png")


def plot_roc_curves() -> None:
    df = read_curve("roc")
    if df.empty:
        return
    plt.figure(figsize=(6.4, 5.2))
    ax = plt.gca()
    for model, g in df.groupby("model", sort=False):
        auc = g["auc"].dropna().iloc[0] if g["auc"].notna().any() else None
        label = display_model(model) if auc is None else f"{display_model(model)} (AUC {auc:.3f})"
        ax.plot(g["fpr"], g["tpr"], lw=1.8, label=label)
    ax.plot([0, 1], [0, 1], color="0.55", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="", frameon=False, fontsize=8, loc="lower right")
    savefig(FIGURES / "fig5_roc_curves.png")


def plot_precision_recall_curves() -> None:
    df = read_curve("pr")
    if df.empty:
        return
    plt.figure(figsize=(6.4, 5.2))
    ax = plt.gca()
    for model, g in df.groupby("model", sort=False):
        ap = g["average_precision"].dropna().iloc[0] if g["average_precision"].notna().any() else None
        label = display_model(model) if ap is None else f"{display_model(model)} (AP {ap:.3f})"
        ax.plot(g["recall"], g["precision"], lw=1.8, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="", frameon=False, fontsize=8, loc="upper right")
    savefig(FIGURES / "fig6_precision_recall_curves.png")


def plot_calibration_curves() -> None:
    df = read_curve("calibration")
    if df.empty:
        return
    plt.figure(figsize=(6.4, 5.2))
    ax = plt.gca()
    for model, g in df.groupby("model", sort=False):
        g = g.sort_values("mean_pred")
        ax.plot(g["mean_pred"], g["observed_rate"], marker="o", ms=3.5, lw=1.6, label=display_model(model))
    ax.plot([0, 1], [0, 1], color="0.55", lw=1, linestyle="--")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="", frameon=False, fontsize=8, loc="upper left")
    savefig(FIGURES / "fig7_calibration_curves.png")


def plot_decision_curves() -> None:
    df = read_curve("dca")
    if df.empty:
        return
    df = df[df["threshold"].between(0.01, 0.50)].copy()
    if df.empty:
        return
    plt.figure(figsize=(6.8, 5.0))
    ax = plt.gca()
    first = df.sort_values("threshold").drop_duplicates("threshold")
    ax.plot(first["threshold"], first["treat_none_net_benefit"], color="0.25", lw=1.2, linestyle=":", label="Treat none")
    ax.plot(first["threshold"], first["treat_all_net_benefit"], color="0.55", lw=1.2, linestyle="--", label="Treat all")
    for model, g in df.groupby("model", sort=False):
        g = g.sort_values("threshold")
        ax.plot(g["threshold"], g["net_benefit"], lw=1.8, label=display_model(model))
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.01, 0.50)
    lower = max(-0.05, df["net_benefit"].quantile(0.02) - 0.01)
    upper = df["net_benefit"].quantile(0.98) + 0.01
    ax.set_ylim(lower, upper)
    ax.legend(title="", frameon=False, fontsize=8, loc="upper right")
    savefig(FIGURES / "fig8_decision_curve_analysis.png")


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plot_model_performance()
    plot_trajectory_patterns()
    plot_ews_risk()
    plot_transformer_importance()
    plot_roc_curves()
    plot_precision_recall_curves()
    plot_calibration_curves()
    plot_decision_curves()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
