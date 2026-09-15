"""Feed-forward network stage — notebook 06.

    from osteo_pipeline.ANN import ANNSpec, make_ann, TrainConfig, TRAIN_MODEL_ANN

Kept in its own subpackage so the TensorFlow dependency stays out of everything
else: importing `osteo_pipeline` or `osteo_pipeline.ML` does not pull in
TensorFlow, and importing this does.

Fold assignment, feature construction, frame checks, and configuration are not
here — they are shared with the ML stage and live at the package root.
"""

from .ANN_plotting import PLOT_FOLD_HISTORY
from .ANN_reporting import REPORT_OOF
from .ANN_training import ANNRun, TrainConfig, TRAIN_MODEL_ANN
from .nets import ANNSpec, make_ann, summarize_ann
from .scaling import apply_scaler, fit_scaler, standardize

__all__ = [
    # architecture
    "ANNSpec",
    "make_ann",
    "summarize_ann",
    # fold-local preprocessing
    "fit_scaler",
    "apply_scaler",
    "standardize",
    # cross-validation
    "TrainConfig",
    "ANNRun",
    "TRAIN_MODEL_ANN",
    # reporting and plots
    "REPORT_OOF",
    "PLOT_FOLD_HISTORY",
]
