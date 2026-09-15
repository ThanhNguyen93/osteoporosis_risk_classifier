"""Shared code for the osteoporosis / hyponatremia matched case-control project.

The package root holds what both modelling stages need — seeds and paths, the
column groups, feature construction, fold assignment, frame checks. The stages
themselves sit underneath:

    osteo_pipeline.ml    scikit-learn / XGBoost families   (notebook 05)
    osteo_pipeline.ann   feed-forward network              (notebook 06)

Importing the root never pulls in TensorFlow; `osteo_pipeline.ann` does.

    from osteo_pipeline import load_column_groups, build_ohe_frame, check_frame
    from osteo_pipeline.ml import RUN_VERSION, AUC_TABLE
"""

from .checks import check_all_predicted, check_frame
from .config import (
    SEED,
    N_SPLITS,
    VAL_FRAC,
    STRATA_COL,
    LABEL_COL,
    PROJECT_ROOT,
    DATA_DIR,
    SOURCE_FILE,
    OHE_FILE,
    load_column_groups,
    make_group_of,
)
from .features import build_ohe_frame, keep_intact_pairs, split_ever_columns
from .folds import FoldStrata, check_pairs_intact, strata_folds, strata_splits

__all__ = [
    # config
    "SEED",
    "N_SPLITS",
    "VAL_FRAC",
    "STRATA_COL",
    "LABEL_COL",
    "PROJECT_ROOT",
    "DATA_DIR",
    "SOURCE_FILE",
    "OHE_FILE",
    "load_column_groups",
    "make_group_of",
    # features
    "build_ohe_frame",
    "keep_intact_pairs",
    "split_ever_columns",
    # folds
    "strata_folds",
    "strata_splits",
    "FoldStrata",
    "check_pairs_intact",
    # checks
    "check_frame",
    "check_all_predicted",
]
