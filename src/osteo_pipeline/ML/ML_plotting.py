"""ROC plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def PLOT_MULTIPLE_ROC(result_df, title="ROC Curve Comparison", savepath=None):
    fig = plt.figure(figsize=(8, 6))

    for i in result_df.index:
        plt.plot(result_df.loc[i]["FPR"],
                 result_df.loc[i]["TPR"],
                 label="{}, AUC={:.3f}".format(i, result_df.loc[i]["AUC"]))

    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 11}, loc="lower right")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.show()


def LABELLED(RUNS, version):
    """Stack the two column sets of one version into a single plotting frame."""
    pieces = []
    for (v, colset), run_df in RUNS.items():
        if v != version:
            continue
        piece = run_df.copy()
        piece.index = [f"{colset} / {m}" for m in piece.index]
        pieces.append(piece)
    return pd.concat(pieces)
