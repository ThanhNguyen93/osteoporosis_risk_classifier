# Matched Case-Control Data and Machine Learning: Asymmetry Under 1:1 Matching

### Why do five different classifiers recall controls better than cases when the dataset is exactly 50/50?

This project investigates a persistent **case-vs-control recall asymmetry** in a matched case-control cohort of osteoporosis patients. Despite perfectly balanced classes, logistic regression, linear SVM, random forest, XGBoost, and a feed-forward ANN all recall controls substantially better than cases.

The project combines **statistical reproduction, multiple machine-learning approaches, and diagnostic analysis** to determine whether the pattern is caused by class imbalance, model choice, or properties of the matched clinical data.

---

## Key finding

Across model families, the models consistently perform better on controls than on cases:

* **Control recall:** 0.82–0.84
* **Case recall:** 0.51–0.60
* The pattern remained stable across **nine runs**.

The dataset is exactly **50/50 cases and controls by construction**, so conventional class imbalance cannot explain the result.

In every model, predicted probabilities for missed cases overlap with those of controls (P ≈ 0.30–0.50), below the 0.5 cutoff.

For a clinical screening application, this matters: at the default 0.5 cutoff, models with AUC up to 0.79 still miss 40–49% of cases. A good ranking score does not guarantee that a model catches most patients with the disease.

---

## Study and dataset

The source study is Usala et al. (2015), which used a matched case-control design to investigate the association between chronic hyponatremia and osteoporosis.

Each osteoporosis case was matched 1:1 to a control on:

* Age
* Sex
* Race

The published analysis estimated an odds ratio of **3.97** for chronic hyponatremia.

> Usala RL, et al. *Journal of Clinical Endocrinology & Metabolism*, 2015.
> DOI: [10.1210/jc.2015-1261](https://doi.org/10.1210/jc.2015-1261)

### Analytic cohort

* **30,513 matched pairs**
* **61,026 patients**
* Five multiple-imputation datasets for missing BMI

---

## Project workflow

### Stage 1 — Reproduce the published estimate

Rebuild the source analysis using conditional logistic regression on the matched strata and pool estimates across five imputations using Rubin's rules.

### Stage 2 — Apply machine-learning models

Evaluate multiple classifiers using out-of-fold predictions and matched-pair-aware cross-validation:

* Logistic regression
* Linear SVM
* Random forest
* XGBoost
* Feed-forward ANN

### Stage 3 — Diagnose the recall asymmetry

Investigate whether the persistent difference between case and control recall can be explained by:

* Pairwise ranking accuracy
* Class probability distributions
* Feature differences between confident and low-confidence cases.
  
---

## Stage 1: Reproducing the published estimate

The published association was reproduced using **conditional logistic regression**, stratified on the matched pairs. The five imputed datasets were pooled using Rubin's rules on the log-odds scale.

| Analysis          | Odds ratio |          95% CI |
| ----------------- | ---------: | --------------: |
| Published         |       3.97 |               — |
| This reproduction |  **3.981** | **3.602–4.399** |

The reproduced estimate closely matches the published result.

### Why BMI imputation matters

BMI is missing for approximately **68% of patients**, and missingness is strongly associated with case status:

* Observed BMI among cases: **13.2%**
* Observed BMI among controls: **50.8%**

Across the five imputations, **26,491 case values** and **15,007 control values** were imputed.

The imputed means moved toward one another:

|          | Observed mean | Imputed mean |
| -------- | ------------: | -----------: |
| Cases    |         26.95 |        27.63 |
| Controls |         28.28 |        27.77 |

Despite the substantial missingness, BMI had little effect on the published association: unadjusted odds ratios matched to two decimal places.

This provides a useful check that the reproduction is not being driven by the BMI imputation procedure.

---

## Methodological considerations: ML on matched cohorts and EHR data

Applying machine learning to observational, matched case-control data requires safeguards against leakage and inflated performance.

### 1. Preventing temporal leakage (index-date enforcement)

The source data summarizes labs over two windows: before the index date, and over the full patient record. The full-record window can include measurements taken **after** diagnosis.

* **Pre-index window:** labs measured before the index date.
* **Full-record window:** the share of patients with a value rises sharply once post-index measurements are included.

| Lab     | Pre-index | Full record |
| ------- | --------: | ----------: |
| Calcium |     58.4% |       88.9% |
| Sodium  |     67.4% |       99.6% |

**Action taken:** all full-record features were excluded from the ML feature set.

### 2. Matched-pair-grouped cross-validation

Random row-level splitting can place a case in the training set and its matched control in the test set, so information from the same matched pair crosses the train/test boundary.

**Action taken:** folds are assigned by matched stratum, so both members of a pair always fall in the same fold.

---

## Stage 2: Machine-learning results

Models were evaluated using **10-fold cross-validation**, with folds split at the matched-stratum level so that members of the same matched pair never appear in both training and test sets.

Results below are from the scaling version with `_Ever` columns excluded.

| Model               |    AUC | Control recall | Case recall |  Gap |
| ------------------- | -----: | -------------: | ----------: | ---: |
| Logistic regression | 0.7490 |         0.8300 |      0.5423 | 0.29 |
| Linear SVM          | 0.7380 |         0.8382 |      0.5121 | 0.33 |
| Random forest       | 0.7886 |         0.8242 |      0.5987 | 0.23 |
| XGBoost             | 0.7900 |         0.8306 |      0.5927 | 0.24 |

### The important result

**Every model recalls controls better than cases.**

The pattern is remarkably consistent:

* Linear models: 0.83–0.84 control recall vs. 0.51–0.54 case recall
* Tree-based models: 0.82–0.83 control recall vs. 0.59–0.60 case recall

Moving from logistic regression to XGBoost improves AUC from **0.749 to 0.790** and case recall from 0.54 to 0.59, but the recall gap remains at 0.24.

---

## Stage 3: Diagnosing the recall asymmetry

Stage 3 investigates why five model families consistently recall controls (0.82–0.84) better than cases (0.51–0.60) despite exact 50/50 class balance. The analysis so far covers pairwise ranking accuracy, class probability distributions, and feature differences between confident and low-confidence cases.

### Within-pair separation

* **Tree ensembles lead:** Within each matched pair, XGBoost ($0.7865$) and Random Forest ($0.7840$) rank the case above its matched control more often than the linear models ($0.7364$–$0.7469$).
* **Above chance:** All models score well above $0.50$, the value expected from random ranking.

### Class probability distributions

* **Consistent controls:** In all models, most controls score below the $0.50$ cutoff, peaking near $P \approx 0.35$.
* **Bimodal cases (tree models):** Random Forest and XGBoost score a large group of cases near $P \approx 1.0$. The remaining cases overlap with controls between $0.30$ and $0.50$.
* **Compressed linear models:** Logistic regression and SVM concentrate predictions around $0.35$–$0.40$, with smaller case peaks near $0.8$–$0.9$.

### Confident vs. low-confidence cases (XGBoost)

Cases were split into high-confidence ($P \ge 0.60$) and low-confidence ($P < 0.60$) groups:

* **Hyponatremia and opiates:** Hyponatremia flags (`Chronic_Hyponatremia`: 13.3% vs. 1.3%) and prior opiate use (`Drug_Opiates_prior`: 16.7% vs. 5.7%) are more common in high-confidence cases.
* **Lab testing:** High-confidence cases were tested for sodium more often (82.8% vs. 62.4%) and for calcium less often (`Calcium_Closest_Osteo_measured`: 38.6% vs. 62.4%).
* **Missingness in the encoding:** Missing labs are coded as decile 0, so lab deciles mix "not measured" with low values.

These are group-level differences, not per-prediction feature contributions.

### Key takeaway

The recall gap is not explained by class imbalance or by a single algorithm. The evidence so far points to case heterogeneity: some cases carry strong signals the models catch, while the rest overlap with controls in this feature space.

Two checks are pending:
* separating missingness flags from lab value deciles
* computing SHAP values for the two confidence groups

> For figures, tables, and full results, see the [investigation report](presentation/investigation_report.md).

---

## Repository

- `config.yaml` — model specification, covariate choices, and exclusion tiers, each with a
  written justification, so the code and the documentation cannot drift apart
- `config_ml.yaml` — feature definitions, derived-column transforms, and cross-validation
  settings for the machine learning pipeline
- `notebooks/` — source verification, imputation checks, conditional logistic regression,
  feature engineering, model comparison
- `notes/` — dated session notes and a register of open and settled methodological questions

### Reproducibility

Every notebook passes a clean **restart-and-run-all** before commit. Notebook outputs are stripped with `nbstripout`.

The configuration files are intended to keep the documented specification and implementation aligned.

---

## Data availability

The underlying dataset is institutional health-system data and **is not redistributable**.

Only aggregate outputs are included in the repository, including:

* Coefficient tables
* Model performance metrics
* Figures

No row-level patient data is committed or displayed. Cells containing counts below five are suppressed.

The configuration files document the analysis specification precisely enough to rebuild an equivalent extract when access to the underlying data is available.

---

## Status

**Stages 1 and 2 complete. Stage 3 in progress.** See *Where the investigation stands* above.
