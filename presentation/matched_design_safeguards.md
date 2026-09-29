# Methodological Safeguards for ML on Matched Case-Control Data

**Design:** 1:1 matched case-control cohort, matched on age (±1 year), sex, race and duration of patient record (±1 month).
**Cohort:** the ML stage uses imputation 1 only: 61,026 patients in 30,513 matched strata. The conditional logistic regression stage uses all 5 imputations (5 × 61,026 = 305,130 rows).

**Legend:** ✅ implemented and verified · ⚠️ implemented with a disclosed limitation · ⬜ pending · — not applicable

---

## 1. Design and cohort integrity

| Safeguard | Status | Evidence / details |
| :--- | :---: | :--- |
| **Stratum integrity** | ✅ | Every stratum keeps exactly one case and one control (`keep_intact_pairs`; asserted as 2 rows per stratum). |
| **Invalid strata removed** | ✅ | 4 strata removed before any modeling: 2 self-matched (same patient as case and control) and 2 oversized (3 rows each, with duplicate rows). |
| **Reference benchmark** | ✅ | Conditional logistic regression on the matched strata, pooled across 5 imputations with Rubin's rules on the log-odds scale: OR 3.981 (3.602–4.399) vs published 3.97. |
| **Repeat encounters** | ⚠️ | 6 patients have records in more than one stratum (repeat encounters, not duplicate rows). Kept, since these are legitimate longitudinal records. A patient can therefore appear in both a training and a test fold. Affects a handful of the 61,026 rows; disclosed, not corrected. |

---

## 2. Information leakage controls

| Safeguard | Status | Evidence / details |
| :--- | :---: | :--- |
| **Temporal anchoring** | ✅ | Post-index (`_Ever`) features excluded; pre-index (`_Prior`) features retained. |
| **Leakage quantification** | ✅ | Removing `_Ever` features lowers AUC by ~0.059, uniformly across model families (0.056–0.067), almost entirely through case recall. |
| **Feature allow-list** | ✅ | Model features are an explicit list in `config_ml.yaml`; outcome-defining columns (e.g. the diagnosis code, which separates cases from controls perfectly) are excluded. |
| **Imputation and the outcome label** | ⚠️ | The BMI imputation model included the outcome. Valid for inference, but imputed BMI can carry label information into a predictive model. See below. |
| **Systematic leakage audit** | ⬜ | Behavior-based audit of remaining candidate columns (those whose missingness pattern alone identifies the outcome). |
| **Anchoring consistency** | ⬜ | `*_Closest_Osteo*` labs are anchored to a real diagnosis date for cases and a constructed date for controls. Decision pending: exclude, or report results both ways. |

### BMI imputation and prediction

- SAS PROC MI imputed BMI with the outcome (`Group`) in the imputation model. This is required for valid inference; omitting the outcome biases associations toward zero.
- For prediction, each patient's imputed BMI was computed using their own label. BMI is observed for 13.2% of cases and 50.8% of controls, so about 87% of case values and 49% of control values are imputed this way.
- The imputation was fit once on the full dataset, not within each CV fold.
- The effect size is unknown. BMI is one continuous column among 43 features, so it is probably small. Planned check: drop BMI, rerun, and compare AUC and `pair_acc`.

### Temporal leakage: index-date enforcement

The source data summarizes labs over two windows: before the index date (`_Avg_Prior`), and over the full patient record (`_Avg_Ever`). The full-record window can include measurements taken after diagnosis. The share of patients with at least one lab value rises sharply when it is used:

| Lab | Pre-index | Full record |
| :--- | ---: | ---: |
| Calcium | 58.4% | 88.9% |
| Sodium | 67.4% | 99.6% |

Source: feature-engineering notebook (`04`), section 4.1. All full-record features were excluded from the ML feature set.

---

## 3. Cross-validation and data partitioning

| Safeguard | Status | Evidence / details |
| :--- | :---: | :--- |
| **Group-aware CV** | ✅ | 10 folds built from unique `Strata` (`strata_folds`), so both members of a pair are always in the same fold. Pair integrity is checked in code (`check_pairs_intact`). |
| **Out-of-fold scoring** | ✅ | Every row is predicted once, by a model that never saw its stratum. AUC is computed from probabilities, not hard labels. |
| **Identical folds across models** | ✅ | Same folds and fixed seed (`12345`) for every model, so results are directly comparable. |
| **Fold-safe scaling** | ✅ | `StandardScaler` sits inside a `Pipeline` and is fit on training folds only. |
| **No tuning on evaluation folds** | ✅ | Fixed conventional hyperparameters, so no nested CV is needed. Reported performance is untuned. |
| **Decile cut points** | ⚠️ | `qcut` deciles are computed on the full frame before splitting. Unsupervised (no outcome used), but the cut points see test-fold values. Effect not measured. |
| **Other splitters** | — | `StratifiedGroupKFold` matters only for variable cluster sizes (1:N matching). Leave-one-pair-out CV is for very small samples (< 100 pairs). |

---

## 4. Metrics and evaluation

| Safeguard | Status | Evidence / details |
| :--- | :---: | :--- |
| **Within-pair discrimination** | ✅ | `pair_acc`: how often a model scores a case above its own matched control. In 1:1 matching this equals within-stratum AUC and the C-index. LR 0.747, SVM 0.736, ANN 0.774, RF 0.784, XGB 0.787. |
| **Both classes reported** | ✅ | Case recall, control recall, AUC, PR-AUC, Brier score and calibration curves; threshold sweep (0.30–0.50) separates ranking quality from cutoff placement. |
| **Ties in `pair_acc`** | ⬜ | Ties (case and control with identical probability) are currently scored as misses; tie rate not reported. Plan: report the rate and score ties as 0.5. |
| **Confidence intervals** | ⬜ | Bootstrap over strata (not rows) for AUC and `pair_acc`, including paired model differences. |

---

## 5. Objectives and scope

| Safeguard / limitation | Status | Evidence / details |
| :--- | :---: | :--- |
| **Inference benchmark** | ✅ | Conditional logistic regression provides the design-appropriate association estimate. |
| **Row-wise ML objectives** | ⚠️ | The ML models use row-wise losses that ignore the pair structure. They are the naive arm of the comparison and keep the matching variables as features by design. |
| **Pair-aware ML comparator** | ⬜ | Conditional logit cross-validated by stratum and scored on `pair_acc`; tree models with a pairwise ranking objective grouped by `Strata`. |
| **Demographics ablation** | ⬜ | Refit without the matching variables (sex, race, age). |
| **Prevalence** | ⚠️ | 50/50 balance is a property of the design, not the population. Recall and precision at a 0.5 cutoff describe this sample; PPV and NPV do not transfer to a screening setting. |
| **External validation** | ⚠️ | Internal validation only: single health system, no external or temporal validation. Lab-ordering patterns may be practice-specific. |
| **Feature attribution** | ⚠️ | SHAP analysis pending (`08`). Confident vs low-confidence case comparisons are univariate group differences, not per-prediction contributions. |
