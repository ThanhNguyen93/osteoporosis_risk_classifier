"""
Learning-curve plots.

Fold-level curves only — aggregate ROC curves across model families live in
``osteo_pipeline.plotting``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt


def PLOT_FOLD_HISTORY(
    history: Mapping[str, Sequence[float]],
    fold_number: int,
    metrics: Sequence[str] = ("accuracy", "loss"),
    save_to: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    One panel per metric for a single fold, train against validation.

    Parameters
    ----------
    history
        A Keras ``History.history`` dict. Each entry in ``metrics`` needs a
        matching ``val_<metric>`` key.
    save_to
        Directory for PNG files named ``fold<N>_<metric>.png``. ``None`` skips
        saving. The directory must already exist — empty directories do not
        survive a git clone, so creating one silently would hide the problem.
    """
    for metric in metrics:
        plt.plot(history[metric])
        plt.plot(history["val_" + metric])
        plt.title(f"model {metric} — fold {fold_number}")
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")

        if save_to is not None:
            plt.savefig(Path(save_to) / f"fold{fold_number}_{metric}.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
