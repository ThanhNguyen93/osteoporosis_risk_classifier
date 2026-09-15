"""Estimator factory. `make_model` is the only place the two versions diverge."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..config import SEED

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_XGB = False


def boost(seed):
    if HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(random_state=seed)


def make_model(name, version, seed=SEED):
    """One factory. `name` picks the family, `version` picks the settings."""

    if name == 'LR':
        if version == 'baseline':
            return LogisticRegression(solver='liblinear', random_state=seed)
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(solver="liblinear",
                                                    random_state=seed))])

    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      n_jobs=-1, random_state=seed)

    if name == 'SVM':
        svm = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=seed, loss='hinge')
        if version == 'baseline':
            return CalibratedClassifierCV(svm)
        return Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(svm))])

    if name == 'XGB':
        return boost(seed)          # identical in both versions

    if name == 'AdaBoost':
        if version != 'baseline':
            raise ValueError("AdaBoost is baseline-only")
        base = LogisticRegression(solver='lbfgs', random_state=seed)
        return AdaBoostClassifier(n_estimators=100, estimator=base,
                                  learning_rate=1, random_state=seed)

    raise ValueError(f"unknown model: {name}")
