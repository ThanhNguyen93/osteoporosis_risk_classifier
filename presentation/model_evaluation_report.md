# Model evaluation report

# Observation for ROC-AUC and PR-AUC

### 1. Model Hierarchy & Superiority of Non-Linear Ensembles

**XGBoost and Random Forest lead across both metrics:**
* **ROC Curve:** XGBoost achieves the highest ROC-AUC ($\text{AUC} = 0.7900$), closely followed by Random Forest ($\text{AUC} = 0.7886$).
* **PR Curve:** XGBoost also maintains top-tier Precision-Recall performance ($\text{PR-AUC} = 0.8187$), virtually tied with Random Forest ($\text{PR-AUC} = 0.8165$).

**Linear & Margin Models Underperform:**
* Logistic Regression ($\text{ROC-AUC} = 0.7490$; $\text{PR-AUC} = 0.7792$) and SVM ($\text{ROC-AUC} = 0.7380$; $\text{PR-AUC} = 0.7600$) lag significantly behind the tree ensembles and the neural network. 


---

### 2. Deep Learning (ANN) vs. Tree Ensembles

* **Competitive, but sub-optimal relative to XGBoost:** The ANN achieves strong discrimination ($\text{ROC-AUC} = 0.7767$; $\text{PR-AUC} = 0.8077$), clearly outperforming linear models.


---

### 3. Precision-Recall Asymmetry & Class Balance Thresholds

* **Baseline Precision:** The "no-skill" baseline on the PR curve sits at $0.500$, indicating a 1:1 balanced target distribution (matched case-control design).
* **High Precision at Low Recall:** For top-ranked predictions ($\text{Recall} \le 0.20$), XGBoost, Random Forest, and ANN achieve near-perfect precision ($\approx 0.95\text{–}1.00$).
* **Precision Decay:** As Recall scales past $0.40$, Precision drops off almost linearly across all models, illustrating the difficulty in capturing the remaining 50–60% of cases without introducing false positives (reflecting the low-confidence/missed case sub-population).


---

# Calibration plot and reliability diagram

The figure displays calibration curves overlaid with support histograms (predicted probability distributions) for all five models under the `no_ever` version.

Here are the key insights and takeaways from these plots:

### 1. Calibration Quality & Probability Reliability

* **XGBoost, ANN, and Random Forest are exceptionally well-calibrated:**
  * Their calibration curves closely track the ideal diagonal reference line ($y = x$) across almost the entire probability spectrum.
  * XGBoost ($\text{Brier} = 0.1825$) achieves the best overall calibration score, followed closely by Random Forest ($0.1842$) and ANN ($0.1876$).
  * This means that predicted probabilities from these models can be directly interpreted as true risk/empirical probabilities (e.g., a patient assigned a predicted probability of $0.70$ actually has an $\approx 70\%$ empirical chance of being a case).

* **LR and SVM suffer from poor calibration at extreme boundaries:**
  * **Logistic Regression** ($\text{Brier} = 0.1982$) exhibits a noticeable spike near $P < 0.10$, predicting near-zero probability for patients who actually have a significantly higher positive fraction ($\approx 0.45$).
  * **SVM** ($\text{Brier} = 0.2057$) shows severe instability and poor calibration at low probability thresholds ($0.0\text{–}0.3$), rendering its raw probability outputs uncalibrated and unreliable.

---

### 2. Empirical Visual Proof of Probability Bimodality

Looking at the gray support histograms behind the calibration lines, we see a clear structural pattern across all models:

* **Massive Concentration Around $P \approx 0.35\text{–}0.40$:**
  * The largest bar in every model’s histogram occurs between $0.30$ and $0.40$.
  * This represents the bulk of controls combined with low-confidence / unmonitored cases that lack lab testing flags (`_measured = 0`).

* **Secondary Peak at High Confidence ($P > 0.80\text{–}0.90$):**
  * Every model exhibits a distinct uptick/peak in sample support at the upper right end ($0.80\text{–}1.00$).
  * This visually captures the high-confidence case mode—the subset of true cases with complete lab draws (`_measured = 1`) and extreme risk markers.

* **Consensus Across Architectures:**
  * The fact that all five models share this exact bimodal density profile confirms that the bimodal split is an intrinsic property of the dataset's clinical feature structure (specifically informative missingness), rather than an artifact of a specific learning algorithm.

---

### Summary Takeaways

| Model | Brier Score | Calibration Quality | Reliability of Probabilities |
| :--- | :---: | :--- | :--- |
| **XGBoost (XGB)** | **0.1825** | Best (Flawless alignment to diagonal) | High (Directly interpretable as risk) |
| **Random Forest (RF)** | 0.1842 | Excellent | High |
| **ANN** | 0.1876 | Excellent | High |
| **Logistic Regression (LR)** | 0.1982 | Fair (Distorted at $P < 0.10$) | Moderate |
| **SVM** | 0.2057 | Poor (Severe distortion at $P < 0.30$) | Unreliable without Platt scaling / Isotonic regression |

---
# Calibration slope and intercept

This table shows the formal logistic calibration metrics—specifically the **Calibration Intercept** (calibration-in-the-large) and **Calibration Slope**—along with their standard errors (`_se`) across all five models.

These quantitative metrics complement the visual calibration curves from earlier and provide strict statistical checks on risk estimation:

### 1. Ideal Calibration Targets

* **Ideal Intercept ($\alpha = 0.0$):** Measures overall calibration-in-the-large (systematic over- or under-estimation of overall risk). An intercept near $0$ means the average predicted probability equals the overall sample event rate.
* **Ideal Slope ($\beta = 1.0$):** Measures model spread/concentration (over- or under-confidence).
  * A slope of $1.0$ indicates perfect discrimination scaling.
  * A slope $< 1.0$ means predictions are over-fitted/over-confident (extreme probabilities are pushed too far toward $0$ or $1$).
  * A slope $> 1.0$ means predictions are under-confident (probabilities are conservative and pulled toward the center).

---

### 2. Key Takeaways from the Table

#### A. Calibration Intercepts: All Models Are Overall Unbiased
Every model exhibits an intercept extremely close to $0.0000$ (ranging from $-0.0001$ to $0.0129$), well within their respective standard error bounds ($\approx 0.009$).

> **Meaning:** None of the models suffer from global calibration drift or systematic baseline bias—on average, the mean predicted probabilities match the $50\%$ target prevalence in the dataset.

#### B. Calibration Slopes: Tree Ensembles & ANN vs. Linear Models

* **Logistic Regression ($\text{slope} = 0.9974$) & SVM ($1.0162$):**
  * Both hover virtually at the ideal $1.0000$ line.
  * While LR and SVM showed localized non-linear distortions on the visual calibration curves (e.g., SVM's instability at $P < 0.3$), their overall global linear risk scaling is well-centered.

* **XGBoost ($\text{slope} = 1.0582$) & ANN ($1.0682$):**
  * Both display a slight slope elevated above $1.0$ (roughly $+5.8\%$ and $+6.8\%$ over ideal).
  * **Meaning:** Predictions are slightly conservative / under-confident, but overall remarkably well-scaled given their non-linear complexity.

* **Random Forest ($\text{slope} = 1.2342$):**
  * Displays a significantly elevated slope ($1.2342$, with $\text{SE} = 0.0131$).
  * **Meaning:** Random Forest suffers from classic tree-bagging probability compression. Because standard random forests average terminal leaf nodes across hundreds of decision trees, extreme predicted probabilities get pulled inward toward the mean ($0.50$), making the raw probabilities under-confident.

---

### Summary Table

| Model | Intercept ($\approx 0.0$) | Slope ($\approx 1.0$) | Primary Takeaway |
| :--- | :---: | :---: | :--- |
| **LR** | $0.0001$ | $0.9974$ | Unbiased global baseline; linear fit. |
| **RF** | $0.0036$ | $1.2342$ | Unbiased intercept, but under-confident probabilities due to tree-averaging. |
| **SVM** | $0.0000$ | $1.0162$ | Unbiased overall, despite local non-linear distortions at lower probabilities. |
| **XGB** | $-0.0001$ | $1.0582$ | Best overall balance—virtually zero intercept and nearly ideal slope. |
| **ANN** | $0.0129$ | $1.0682$ | Strong, well-calibrated non-linear scaling. |