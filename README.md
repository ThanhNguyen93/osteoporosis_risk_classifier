# Matched Case-Control Data and Machine Learning

Osteoporosis is usually diagnosed after a fracture, which makes earlier
identification from routine health records an appealing target for a
classifier. This project builds one on a health-system cohort of 30,513
osteoporosis cases and their matched controls — and then asks whether the
resulting metrics mean what they appear to mean.

They don't, and the reason is structural. The cohort was assembled as a
matched case-control study, a design that deliberately strips information out
of the data to make a causal estimate valid. That same stripping breaks
assumptions standard classification pipelines rely on. The design that makes
the epidemiological estimate trustworthy is the design that makes the ML
metrics misleading.

---

## The study

Usala et al. (2015) matched every osteoporosis case to one control on age, sex, and race,
and estimated the association with chronic hyponatremia at an odds ratio of **3.97**.

> Usala RL, et al. *Journal of Clinical Endocrinology & Metabolism*, 2015.
> DOI: [10.1210/jc.2015-1261](https://doi.org/10.1210/jc.2015-1261)

Analytic cohort as rebuilt here: **30,513 matched pairs (61,026 patients)**, with five
multiple-imputation datasets covering missing BMI.

---

## Reproducing the estimate

Conditional logistic regression, stratified on the matched pairs, with the five
imputations pooled by Rubin's rules on the log-odds scale.

| | Odds ratio | 95% CI |
|---|---|---|
| Published | 3.97 | — |
| This replication | **3.981** | 3.602 – 4.399 |

Two things worth noting from the pooling step:

**Rubin's rules earn their keep unevenly.** BMI's fraction of missing information was
~0.53–0.59; the hyponatremia coefficient's was ~0.00006. On one covariate the pooling is
doing real work, on the other it is arithmetic. Reporting both makes clear where the
uncertainty actually lives.

**The imputation left a signature.** BMI missingness was strongly associated with case
status, and the imputed means collapsed toward a common value — diluting the observed
case/control gap in a way consistent with an imputation model that omitted the outcome.

---

## What the matched design does to machine learning

A matched design deliberately removes information. That is the point: age, sex, and race
are held identical within each pair so they cannot confound the exposure estimate. But
those same properties break assumptions that classification pipelines take for granted.

**Matching variables carry no outcome information.** Within a pair they are identical by
construction. A chi-square screen scores all three at exactly **0.000, p = 1.000**. They
are not weak predictors — they are structurally incapable of predicting, and including
them adds only noise for tree models to fit.

**Exposure windows have to respect the index date.** The source data carries lab summaries
in two forms: restricted to before the index date, and spanning the full record. Measured
rates for the same patients and the same assay:

| Lab | Before index | Full record |
|---|---|---|
| Calcium | 58.4% | 88.9% |
| Sodium | 67.4% | 99.6% |

The full-record sodium column is measured on essentially everyone, because a diagnosis
generates lab draws. On the chi-square screen a full-record calcium decile scores
**3,724** against chronic hyponatremia — the study's actual exposure — at **971**. A
post-diagnosis artifact outscores the hypothesis four to one. Full-record columns are
excluded.

**Folds must split on pairs, not rows.** Cross-validation assigns whole strata to folds,
so a case and its matched control never land on opposite sides of a split.

### Results

Out-of-fold predictions, 10-fold cross-validation, folds split on strata:

| Model | AUC | Control recall | Case recall | Train acc | Test acc |
|---|---|---|---|---|---|
| Logistic regression | 0.808 | 0.78 | 0.67 | 0.723 | 0.723 |
| Linear SVM | 0.807 | 0.78 | 0.66 | 0.723 | 0.723 |
| Random forest | 0.845 | 0.82 | 0.68 | 0.848 | 0.752 |
| XGBoost | 0.848 | 0.81 | 0.70 | 0.773 | 0.755 |

### The finding

**Every model is systematically better at controls than at cases, on a perfectly balanced
sample.** All four catch roughly four in five controls and miss roughly one in three
cases.

Class imbalance cannot explain it — the matched design guarantees exactly 50/50. The
asymmetry lives in the feature space, not the label distribution. Cases are heterogeneous
in a way controls are not, and no reweighting fixes a problem that was never about
prevalence.

Model capacity does not fix it either. Moving from logistic regression to gradient
boosting buys about four AUC points and leaves the recall gap intact — the extra capacity
lifts both sides slightly rather than closing the distance between them. For a screening
application that distinction is the entire question: a model missing a third of cases is
not made acceptable by a better AUC. The near-identical logistic regression and linear SVM
results mark the linear ceiling on this feature set.

---

## Repository

- `config.yaml` — model specification, covariate choices, and exclusion tiers, each with a
  written justification, so the code and the documentation cannot drift apart
- `config_ml.yaml` — feature definitions, derived-column transforms, and cross-validation
  settings for the machine learning pipeline
- `notebooks/` — source verification, imputation checks, conditional logistic regression,
  feature engineering, model comparison
- `notes/` — dated session notes and a register of open and settled methodological questions

Every notebook passes a clean restart-and-run-all before commit; outputs are stripped with
`nbstripout`.

---

## Data availability

The underlying dataset is institutional health-system data and **is not redistributable**.
Only aggregate outputs — coefficient tables, performance metrics, figures — appear here.
No row-level data is committed or displayed, and cells with counts below 5 are suppressed.
The configuration files document the specification precisely enough to rebuild an
equivalent extract.

---

## Status

Replication complete. The machine learning analysis is in progress: a leakage audit and a
decision on index-anchored lab columns are outstanding, so the metrics above are current
results rather than final ones.
