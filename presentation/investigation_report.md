
# 1. Pairwise ranking
This evaluation measures pairwise ranking accuracy (or matched pair accuracy), which tests how often each model correctly assigns a higher predicted probability to a case compared to its matched control within the same stratum.

Another name is `Direct Pairwise Concordance` ($C$-Index): For 1:1 matched pairs, the pair_acc metric is mathematically equivalent to the conditional area under the ROC curve (or **concordance index**). 

![Pairwise accuracy ranking](../outputs/figures/pairwise_acc_ranking.png)


**Key Takeaways**

**Tree-Based Models Lead:** XGBoost (0.7865) and Random Forest (0.7840) perform best, correctly ranking the case higher than its matched control in ~78.6% of strata. Their performance is virtually neck-and-neck.

Linear/Margin Models Lag: Logistic Regression (0.7469) and SVM (0.7364) show a ~4–5% drop in pairwise accuracy compared to the tree ensembles

Interpretation of Metric: Because this metric explicitly evaluates within-stratum ranking, it directly mirrors the pairwise concordance probability (equivalent to the Area Under the ROC Curve evaluated within matched pairs). A score of 0.5000 would indicate random guessing, so all models demonstrate substantial predictive power above baseline.

**Conclusions**
Model Selection: Gradient Boosting (XGB) or Random Forest (RF) should be prioritized for tuning, as tree ensembles better capture the underlying structure of this matched dataset.

Pairwise Separation: All models demonstrate strong discriminative capacity well above baseline (>73%), confirming that model-predicted risk scores reliably distinguish cases from matched controls within the same stratum.

---
# 2. Probability distribution by true class

![Probability](../outputs/figures/predicted_probability.png)

Key Findings: 
- **Tree Models (RF & XGB) Drive Sharp Bimodality**:
  - XGB and RF push a huge cluster of true cases (orange) sharply toward $P(\text{case}) \approx 1.0$, creating a strong right peak.
  - This matches the earlier confidence breakdown: when high-signal features are present (like hyponatremia and opiate history), XGB predicts with extreme certainty ($P > 0.6$ or near $1.0$), driving up pair accuracy.
  
- **Linear & Kernel Models (LR & SVM) Struggle with Discrimination:**
  - **Logistic Regression**: Shows a severe spike around $P \approx 0.35 - 0.40$ where controls and cases heavily overlap, with a much weaker peak near $0.9$. 
  - **SVM**: Extremely compressed predictions around $P \approx 0.35 - 0.40$, with an artificial secondary spike near $0.8$. It lacks a smooth risk continuum and does not to push true cases to near $1.0$.
 
- **Control Group Distribution is Consistent Across Models:**
  - In all models, the vast majority of controls (blue) fall safely below the $0.5$ threshold, peaking around $P \approx 0.35$.
  - The main discriminator between models isn't how they treat controls, but how effectively they can identify and pull cases out of that overlapping middle zone ($P \approx 0.35 - 0.50$).
  
- **Explaining Low-Confidence Cases tested only on XGBoost:**
  - The overlap region between $P = 0.30$ and $P = 0.50$ represents the "low-confidence cases" from your previous breakdown (e.g., cases without hyponatremia or opiate signals). In this region, case and control distributions overlap significantly across all models. 
  

**Takeaway**

Tree-based models excel here because they split true cases into two distinct clinical populations: an easily identifiable, high-risk group ($P \to 1.0$) driven by non-linear risk factors, and a subtle low-risk group that overlaps with controls. Linear/kernel models fail to isolate that high-risk group cleanly.

---
# 3. Two groups of cases: highlighting the features that drive strong model certainty

![features_drive_confidence_prediction](../outputs/figures/features_drive_confidence_prediction.png)

This analysis compares true cases that `XGBoost` predicted with high confidence ($P(\text{case}) \ge 0.6$) against those it predicted with low confidence ($P(\text{case}) < 0.6$), highlighting the features that drive strong model certainty.

**Primary Insights**

- **Hyponatremia & Sodium Signals Drive High Confidence:**
  - `Chronic_Hyponatremia` (+0.120 diff) and `Recent_Hyponatremia` (+0.118 diff) are vastly more prevalent among high-confidence cases—occurring in ~13–15% of high-confidence predictions compared to only ~1–3% of low-confidence predictions.
  - Measures of prior hyponatremia (`Hyponatremia_Prior`: 0.230 vs. 0.092) and sodium measurement flags (`Sodium_Closest_Osteo_measured`: 0.828 vs. 0.624) strongly push XGBoost toward higher predicted case probabilities.

- **Opiate Use & BMI Contribution:**
  - Prior opiate exposure (Drug_Opiates_prior: 0.167 vs. 0.057) is roughly 3x higher in the high-confidence tier, serving as a key risk marker.
  - BMI_Avg is slightly lower (27.2 vs. 27.9, -0.736 diff) in high-confidence cases.
  
**Methodological & Operational Implications**

- **Subgroup Identification:** XGBoost relies heavily on electrolyte disturbances (hyponatremia indicators) and prescription history (opiates) as strong anchors for confident case predictions.
- **Feature Importance vs. Explanation:** While this table isolates univariate differences between confidence tiers, using SHAP (SHapley Additive exPlanations) on these same cases would quantify exact non-linear feature contributions per individual prediction, complementing group mean differences.

---
# 4. Informative missingness / ascertainment bias

With this feature engineering context—where decile 0 contains missing values and _measured equals 1 when a test was performed—the apparent drop in Calcium decile actually reveals an informative missingness / ascertainment bias story rather than a physiological one.

**Key Findings**

**High-Confidence Cases Have Fewer Calcium Labs Measured:** 
- `Calcium_Closest_Osteo_measured` drops from 62.4% in low-confidence cases to 38.6% in high-confidence cases (-0.238 diff).
- `Calcium_Avg_Prior_measured` drops similarly (41.0% vs. 61.9%, -0.209 diff).
- High-confidence predictions occur far more frequently in patients **who did not have calcium measured at all**.

**Lower Calcium Deciles are Driven directly by Missing Values:**
- Because missing values are dumped into decile 0, the lower mean decile for high-confidence cases (Calcium_Avg_Prior_decile: 2.050 vs. 3.082, -1.032 diff) is a mathematical artifact of having ~21–24% more missing calcium labs, not lower physiological serum calcium.
  
**Opposite Trend for Sodium Testing (Active Testing Signal):**
- High-confidence cases are more likely to have sodium measured (`Sodium_Closest_Osteo_measured`: 82.8% vs. 62.4%, +0.204 diff).Combined with `Chronic_Hyponatremia` (13.3% vs. 1.3%) and `Recent_Hyponatremia` (14.8% vs. 3.0%), high model confidence is strongly anchored in active workups for electrolyte imbalances/hyponatremia.

**Takeaway & Model Behavior**

**Two Distinct Profiles for Confident Predictions:**
1. The Hyponatremia/Symptomatic Phenotype: Active lab testing for sodium + positive hyponatremia flag + prior opiate exposure ($16.7\%$ vs $5.7\%$).
2. Missing Calcium Data Profile: Patients without recent serum calcium tests are getting flagged with high risk, likely because calcium screening is missing in specific clinical settings or patient trajectories where osteoporosis risk is already high/evident via other markers.

**Methodological Caution:** Because decile 0 conflates true low values with missing values, XGBoost is using decile 0 as a proxy for "unmeasured." Splitting missingness indicator flags from actual value deciles (or using explicit missing-value handling like native XGBoost split direction) will give a cleaner physiological interpretation.

---
