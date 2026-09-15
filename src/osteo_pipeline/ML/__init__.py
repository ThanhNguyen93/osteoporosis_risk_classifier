"""scikit-learn / XGBoost stage — notebook 05 (`baseline vs. scaling`).

    from osteo_pipeline.ML import RUN_VERSION, REPORT, AUC_TABLE, PLOT_MULTIPLE_ROC

Fold assignment, feature construction, and configuration are not here — they are
shared with the ANN stage and live at the package root.
"""

from .ML_models import HAS_XGB, boost, make_model
from .ML_plotting import LABELLED, PLOT_MULTIPLE_ROC
from .ML_reporting import (
    AUC_TABLE,
    DELTA_TABLE,
    PRED_RATE_TABLE,
    REPORT,
    TAKEAWAY_TABLE,
    THRESHOLD_SWEEP,
)
from .ML_training import RUN_VERSION, TRAIN_MODEL_ML

__all__ = [
    "HAS_XGB",
    "boost",
    "make_model",
    "TRAIN_MODEL_ML",
    "RUN_VERSION",
    "REPORT",
    "AUC_TABLE",
    "DELTA_TABLE",
    "TAKEAWAY_TABLE",
    "PRED_RATE_TABLE",
    "THRESHOLD_SWEEP",
    "PLOT_MULTIPLE_ROC",
    "LABELLED",
]
