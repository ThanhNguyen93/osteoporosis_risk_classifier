"""Tables and printed reports built from the run frames returned by `RUN_VERSION`."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score


def REPORT(run_df, label, y):
    print(f"########## {label} ##########")
    for name in run_df.index:
        row = run_df.loc[name]
        print(f"\n===== {name} =====")
        print(classification_report(y, row["y_pred"]))
        print(confusion_matrix(y, row["y_pred"]))
        print("AUC           :", round(row["AUC"], 4))
        print("mean train acc:", round(np.mean(row["training_acc"]), 4))
        print("mean test acc :", round(np.mean(row["testing_acc"]), 4))
    print()


def AUC_TABLE(RUNS):
    """Long-to-wide on `colset` only; the two versions are not pivoted against
    each other."""
    rows = []
    for (version, colset), run_df in RUNS.items():
        for model_name in run_df.index:
            rows.append({"version": version,
                         "colset":  colset,
                         "model":   model_name,
                         "AUC":     run_df.loc[model_name]["AUC"]})

    all_runs = pd.DataFrame(rows)

    auc_table = all_runs.pivot(index=["version", "model"], columns="colset", values="AUC")
    auc_table = auc_table[["with_ever", "no_ever"]]
    auc_table["delta"] = auc_table["no_ever"] - auc_table["with_ever"]

    return auc_table


def DELTA_TABLE(with_ever_df, no_ever_df, version, y):
    rows = []

    for name in with_ever_df.index:
        w = with_ever_df.loc[name]
        n = no_ever_df.loc[name]

        rows.append({
            "version":               version,
            "model":                 name,
            "auc_with_ever":         w["AUC"],
            "auc_no_ever":           n["AUC"],
            "auc_delta":             n["AUC"] - w["AUC"],
            "case_recall_with_ever": recall_score(y, w["y_pred"], pos_label=1),
            "case_recall_no_ever":   recall_score(y, n["y_pred"], pos_label=1),
            "ctrl_recall_with_ever": recall_score(y, w["y_pred"], pos_label=0),
            "ctrl_recall_no_ever":   recall_score(y, n["y_pred"], pos_label=0),
        })

    return pd.DataFrame(rows).set_index(["version", "model"])


def TAKEAWAY_TABLE(run_df, y):
    """AUC and out-of-fold recall split by class, one row per model."""
    takeaway_rows = []

    for name in run_df.index:
        row = run_df.loc[name]
        takeaway_rows.append({
            "model":       name,
            "AUC":         row["AUC"],
            "case_recall": recall_score(y, row["y_pred"], pos_label=1),
            "ctrl_recall": recall_score(y, row["y_pred"], pos_label=0),
        })

    return pd.DataFrame(takeaway_rows).set_index("model")


def PRED_RATE_TABLE(run_df, y):
    """What fraction of rows does each model call positive, vs. the 0.50 the
    design implies?"""
    rate_rows = []

    for name in run_df.index:
        row = run_df.loc[name]
        rate_rows.append({
            "model":         name,
            "pred_pos_rate": (row["y_prob"] > 0.5).mean(),
            "prob_median":   np.median(row["y_prob"]),
            "case_recall":   recall_score(y, row["y_pred"], pos_label=1),
            "ctrl_recall":   recall_score(y, row["y_pred"], pos_label=0),
        })

    return pd.DataFrame(rate_rows).set_index("model")


def THRESHOLD_SWEEP(run_df, y, cuts=(0.30, 0.35, 0.40, 0.45, 0.50)):
    """Same probabilities, different cutoffs — AUC is fixed, the recalls are not."""
    for name in run_df.index:
        prob = run_df.loc[name]["y_prob"]
        print(f"\n===== {name} (AUC {run_df.loc[name]['AUC']:.4f}, fixed) =====")
        for cut in cuts:
            pred = (prob > cut).astype(int)
            print(f"  cut {cut:.2f}  case {recall_score(y, pred, pos_label=1):.4f}"
                  f"  ctrl {recall_score(y, pred, pos_label=0):.4f}")
