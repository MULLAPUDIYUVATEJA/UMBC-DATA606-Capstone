# AI ReliabilityGuard: Failure Forecasting and Reliability Modeling for Machine Learning Systems

**Author:** Yuvateja Mullapudi
**Semester:** Spring 2026
**Course:** DATA 606 — Capstone Project, UMBC Data Science Master Degree
**Instructor:** Dr. Chaojie (Jay) Wang

---

> 📺 **[YouTube Presentation](#)** | 📊 **[PowerPoint Slides (docs/presentation.pptx)](#)** | 💻 **[GitHub Repository](#)**

---

## Table of Contents

1. [Background](#1-background)
2. [Data](#2-data)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Model Training](#4-model-training)
5. [Application of the Trained Models](#5-application-of-the-trained-models)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)

---

## 1. Background

### What is this project about?

AI ReliabilityGuard is a **Trustworthy AI framework** designed to forecast when a machine learning model is likely to make incorrect predictions. Modern AI systems often achieve strong overall accuracy but may still produce highly confident errors. These failures reduce trust and create risks when AI models are used in automated or decision-support environments — particularly in high-stakes domains such as financial fraud detection.

This project builds an **AI self-evaluation system** that predicts the reliability of model predictions. Rather than improving task accuracy alone, the system learns patterns associated with model failures and estimates whether a prediction should be trusted. The IEEE-CIS fraud detection dataset serves as the experimental environment, where realistic model errors naturally occur due to feature complexity, noise, and class imbalance.

The framework has three conceptual components:

- **Primary Prediction Model** — A supervised ensemble model trained on the IEEE-CIS dataset to classify fraudulent transactions.
- **Failure Forecasting Model** — A secondary meta-learning model that predicts whether the primary model's prediction will be correct or incorrect.
- **Reliability Assessment Layer** — Produces a reliability score (0–1) representing the trustworthiness of each prediction.

### Why does it matter?

Traditional ML evaluation focuses on aggregate metrics such as accuracy, precision, or ROC-AUC. However, these metrics do not explain:

- *When* a model is likely to fail
- *Which* individual predictions are unreliable
- Whether the model's confidence is actually justified

In real-world deployments, AI systems must be able to estimate their own reliability. A fraud detection model that flags 95% of cases correctly but fails silently on 5% of high-value transactions can cause massive financial damage. Failure forecasting enables:

- Early identification of unreliable predictions before they cause harm
- Reduction of overconfident, high-stakes errors
- Improved transparency and auditability in AI systems
- Safer human-in-the-loop decision making

This project contributes to the growing field of **Trustworthy AI**, where reliability and safety are as important as raw performance metrics.

### Research Questions

1. Can a meta-learning model accurately predict when a machine learning model will make incorrect predictions?
2. Which characteristics of model outputs and input data best indicate future prediction errors?
3. How effectively can reliability scores distinguish between trustworthy and unreliable predictions?
4. Can failure forecasting improve system safety compared to traditional accuracy-based evaluation?

---

## 2. Data

### Data Sources

Two datasets were used in this project:

**Dataset 1 — IEEE-CIS Fraud Detection Dataset (Experimental Benchmark)**

- **Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **Role:** Provides a large-scale, complex dataset used to train the primary prediction model and generate realistic prediction errors for reliability analysis.

**Dataset 2 — Model Prediction Metadata (Generated)**

- **Source:** Generated internally during model evaluation on the IEEE-CIS test split.
- **Role:** Primary dataset for training the failure forecasting model.
- **Target Variable:** `Model_Error` — 1 if the primary model predicted incorrectly, 0 if correctly.

---

### IEEE-CIS Dataset Description

| Property | Value |
|---|---|
| Total Samples | ~590,000 |
| Features | 390+ |
| Working Sample Used | 100,000 (random sample, `random_state=42`) |
| Time Period | Covered by `TransactionDT` (seconds offset) |
| Files | `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` |
| Join Key | `TransactionID` (left join on identity table) |

**What does each row represent?**
Each row represents one **financial transaction**, including behavioral signals, device metadata, card information, and the binary fraud label (`isFraud`).

---

### Data Dictionary — Key Variables

| Column | Data Type | Description | Values |
|---|---|---|---|
| `TransactionID` | Integer | Unique transaction identifier | — |
| `isFraud` | Integer | **Target label** — fraud indicator | 0 = Legitimate, 1 = Fraud |
| `TransactionDT` | Integer | Transaction timestamp offset (seconds) | Numeric |
| `TransactionAmt` | Float | Dollar amount of the transaction | Continuous, 0–$1,000+ |
| `ProductCD` | Categorical | Product code for the transaction | W, H, C, S, R |
| `card1–card6` | Mixed | Card-related anonymized features | Numeric & Categorical |
| `card4` | Categorical | Card network | Visa, Mastercard, Discover, American Express |
| `card6` | Categorical | Card type | Debit, Credit |
| `P_emaildomain` | Categorical | Purchaser email domain | gmail.com, yahoo.com, etc. |
| `R_emaildomain` | Categorical | Recipient email domain | gmail.com, yahoo.com, etc. |
| `DeviceType` | Categorical | Device used for transaction | Desktop, Mobile |
| `DeviceInfo` | Categorical | Specific device string | Freeform string |
| `V1–V339` | Float | Vesta-engineered anonymized features | Numeric |
| `C1–C14` | Float | Counting features (addresses, cards) | Numeric |
| `D1–D15` | Float | Timedelta features | Numeric |
| `M1–M9` | Categorical | Match features (name, address) | T/F |
| `id_01–id_38` | Mixed | Identity features | Mixed |

**Target Variable:** `isFraud`

**Selected Features for Primary Model:** Top 150 features by LightGBM feature importance, including transaction amount, Vesta features (V1–V339), counting features (C1–C14), and card metadata.

**Features for Failure Forecasting Model:**

| Feature | Description |
|---|---|
| `confidence` | Ensemble predicted probability |
| `entropy` | Prediction uncertainty (Shannon entropy) |
| `distance` | Distance from decision boundary (`|prob - 0.5|`) |
| `missing_count` | Number of missing features per transaction |
| `disagreement` | Std deviation across LightGBM, RF, and LR predictions |

---

## 3. Exploratory Data Analysis (EDA)

All EDA was performed in Google Colab using pandas, matplotlib, and seaborn on the merged `train_transaction + train_identity` dataset.

### 3.1 Target Variable Distribution

The dataset exhibits **severe class imbalance**:

- **~96.5% Legitimate** transactions (isFraud = 0)
- **~3.5% Fraudulent** transactions (isFraud = 1)

This imbalance is a key challenge — a naive model predicting "all legitimate" achieves 96.5% accuracy while detecting zero fraud. Stratified splits and careful metric selection (ROC-AUC over accuracy) were used to address this.

### 3.2 Missing Values

Missing values were extensive across the dataset:

- Identity table features (`id_01` through `id_38`) had up to 75%+ missingness — largely because many transactions had no associated identity record (left join).
- V-series features (Vesta engineered) also exhibited moderate missingness.
- A key EDA finding: **missingness itself is informative** — fraudulent transactions showed *higher* missingness rates on certain identity features, suggesting that fraudsters avoid leaving identity traces.

All missing values were imputed with `-999` as a sentinel value and encoded as a feature (`missing_count`).

### 3.3 Transaction Amount Distribution

The transaction amount distribution is heavily right-skewed. After a `log1p` transformation, the distribution approaches normality with a peak around $50–$200. Fraudulent transactions did not show a dramatically different amount distribution, confirming that amount alone is insufficient for fraud detection.

### 3.4 Temporal Patterns

`TransactionDT` was converted to day-level granularity (`TransactionDay`). The fraud rate showed temporal variation across the observation window:

- A **rolling 10-day average** smoothed noise and revealed gradual shifts in fraud rates over time.
- This temporal variation indicates the dataset spans a meaningful time window and motivates the need for temporally-robust models.

### 3.5 Email Domain Mismatch

Transactions where the purchaser's email domain (`P_emaildomain`) **did not match** the recipient's (`R_emaildomain`) were associated with notably higher fraud rates. This engineered binary feature (`email_match`) provided a meaningful fraud signal.

### 3.6 Product Type

Fraud rates varied by `ProductCD`. Product category **C** showed a disproportionately high fraud rate compared to categories W, H, S, and R — a strong categorical predictor included in the model.

### 3.7 Confidence vs. Model Error

A preliminary Random Forest model (30 estimators, max depth 10) was trained on 100,000 samples to study prediction reliability:

**Key finding:** Model errors were distributed *across all confidence levels*, not just at low confidence. The presence of errors at high probability values (>0.8) confirms **model overconfidence** — the model assigns high probability to wrong predictions. This is the core motivation for the failure forecasting layer.

The error rate by confidence bin showed:
- High error rates at mid-confidence (0.4–0.6) as expected
- Non-trivial error rates even at high confidence (>0.8) — the overconfidence problem

### 3.8 Data Quality Summary

| Issue | Finding | Action Taken |
|---|---|---|
| Missing values | Extensive (up to 75%+ in identity features) | Imputed with `-999` sentinel |
| Class imbalance | ~96.5% legitimate, ~3.5% fraud | Stratified splits, ROC-AUC metric |
| Duplicate rows | None detected | — |
| Categorical encoding | Multiple object columns | LabelEncoder applied |
| Memory | Large dataset (590K × 390+) | Memory reduction via downcasting |
| Temporal structure | `TransactionDT` is time-based | Analyzed as day-level time series |

---

## 4. Model Training

### 4.1 Overview

Model training followed a **two-phase meta-learning pipeline**:

- **Phase 1:** Train an ensemble primary model on the IEEE-CIS fraud classification task.
- **Phase 2:** Train a failure forecasting meta-model to predict when the primary model will be wrong.

### 4.2 Primary Model — Ensemble Classifier

**Models used:**
- LightGBM (`LGBMClassifier`, 200 estimators, lr=0.05)
- Random Forest (`RandomForestClassifier`, 100 estimators)
- Logistic Regression (`LogisticRegression`, max_iter=1000)

**Training Strategy:**
- 5-Fold Stratified Cross-Validation (`StratifiedKFold`, shuffle=True)
- Out-of-fold (OOF) predictions generated for each model
- Final ensemble prediction = average of LightGBM + RF + LR probabilities
- Test predictions averaged across 5 folds

**Train/Test Split:** 80% train / 20% test, stratified on `isFraud`

**Feature Selection:** Top 150 features selected by LightGBM feature importance on a pre-screening pass (100 estimators), reducing dimensionality from 390+ to 150.

**Development Environment:** Google Colab (GPU runtime for LightGBM acceleration)

**Python Packages:** `scikit-learn`, `lightgbm`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pickle`

**Primary Model Results:**

| Metric | Value |
|---|---|
| Base Ensemble ROC-AUC | High (computed at runtime) |
| Base Accuracy | High (computed at runtime) |

### 4.3 Failure Forecasting Model — Meta-Model

**Target:** `Model_Error` — 1 if the primary ensemble predicted incorrectly, 0 if correctly.

**Meta-features constructed from OOF predictions:**

| Feature | Formula |
|---|---|
| `confidence` | Ensemble probability `p` |
| `entropy` | `-(p·log(p) + (1-p)·log(1-p))` |
| `distance` | `|p - 0.5|` (margin from decision boundary) |
| `missing_count` | Count of `-999` values per row |
| `disagreement` | `std(LightGBM_prob, RF_prob, LR_prob)` |

**Meta-model:** LightGBM Classifier (200 estimators, `random_state=42`)

**Meta-model Evaluation Metrics:**

| Metric | Description |
|---|---|
| **ROC-AUC** | Discriminative power of failure forecasting |
| **Brier Score** | Calibration of predicted failure probabilities |
| **ECE (Expected Calibration Error)** | Alignment between confidence and correctness rate |

**Reliability Score:** `reliability_score = 1 - meta_test_prob`
A score near 1.0 means the primary model is very likely correct. A score near 0.0 signals high risk of prediction failure.

### 4.4 Threshold Sweep

A reliability threshold sweep was conducted over the range [0.40, 0.95]:

- For each threshold `t`, transactions with `reliability_score > t` were retained ("covered").
- Coverage (fraction retained) and accuracy on the retained subset were tracked.
- **Key finding:** As the reliability threshold increases, coverage decreases but accuracy on retained predictions increases — demonstrating that the reliability score effectively separates trustworthy from untrustworthy predictions.

### 4.5 Performance Summary

| Component | Metric | Result |
|---|---|---|
| Primary Ensemble | ROC-AUC | Evaluated on 20% hold-out |
| Primary Ensemble | Accuracy | Evaluated on 20% hold-out |
| Failure Forecasting | Meta ROC-AUC | Evaluated on test meta-features |
| Failure Forecasting | Brier Score | Lower = better calibration |
| Failure Forecasting | ECE | Lower = better calibration |

---

## 5. Application of the Trained Models

### 5.1 Streamlit Dashboard — AI ReliabilityGuard

A fully interactive web dashboard was built using **Streamlit**, deployed on Google Colab via `localtunnel`. The dashboard provides:

**Single-Transaction Fraud Verdict**
Users can input a custom reliability score, error rate, and transaction amount via sidebar sliders. The app instantly returns:
- A fraud/legitimate verdict card (color-coded)
- Composite risk percentage
- Individual metric display

**Fraud decision rule:**
```
IF reliability_score < 0.5 AND error > 0.2 → FRAUD
ELSE → LEGITIMATE
```

**Aggregate Dashboard**
Sidebar filters allow users to filter transactions by card network, card type, device type, product type, and transaction amount (capped at $1,000). The dashboard displays:
- Transaction count, average reliability, error rate, and fraud rate metrics
- Risk status banner (🔴 High / 🟡 Moderate / 🟢 Low)

**Visual Analytics (2×2 chart grid)**

| Chart | Description |
|---|---|
| Reliability Distribution | Histogram of reliability scores with mean line |
| Fraud vs Legitimate | Bar chart of predicted class counts |
| Error Distribution | Histogram of error rates |
| Reliability vs Error Scatter | Color-coded scatter plot with decision boundary lines |

**Transaction Risk Table**
A filtered table showing up to 30 individual transactions with reliability score, error, fraud prediction, amount, card info, and device type — enabling row-level audit of flagged transactions.

### 5.2 Tools Used

| Component | Tool |
|---|---|
| Dashboard Framework | Streamlit |
| Data Serialization | Python `pickle` |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Google Colab |
| Deployment | Streamlit + localtunnel |

---

## 6. Conclusion

### Summary

AI ReliabilityGuard demonstrates that a machine learning system can be augmented with a **meta-learning failure forecasting layer** that predicts its own likelihood of making incorrect predictions. Using the IEEE-CIS fraud detection dataset as an experimental benchmark, this project:

- Trained a 3-model ensemble (LightGBM + Random Forest + Logistic Regression) using 5-fold cross-validation
- Engineered uncertainty and disagreement features from ensemble predictions
- Trained a secondary LightGBM meta-model to predict model errors
- Produced per-transaction reliability scores that effectively separate correct from incorrect predictions
- Built an interactive Streamlit dashboard for real-time reliability monitoring and fraud flagging

The threshold sweep analysis confirmed that **reliability-aware filtering can significantly improve precision** on retained predictions — a practical benefit for any high-stakes deployment scenario.

### Potential Applications

- **Financial services:** Flag unreliable fraud decisions for human review before acting
- **Healthcare:** Surface uncertain diagnostic model predictions for clinician verification
- **Cybersecurity:** Monitor intrusion detection model reliability in real time
- **Autonomous systems:** Identify when self-driving perception models should defer to backup logic

### Limitations

- The failure forecasting model is trained on the same data distribution as the primary model. Out-of-distribution failures (concept drift, novel fraud patterns) may not be reliably detected.
- The fraud prediction rule in the dashboard (`reliability < 0.5 AND error > 0.2`) is a simplified heuristic. In production, the raw `meta_test_prob` output should be used with a tuned threshold.
- The 100,000-sample subset used for training is a fraction of the full 590,000-row dataset due to Colab memory constraints, which may limit model generalization.
- Missing value imputation with `-999` sentinel is a simplification — more sophisticated imputation strategies (KNN, MICE) could improve feature quality.
- The reliability score is only as good as the meta-features — if the primary model's confidence is systematically miscalibrated, the meta-model inherits that bias.

### Lessons Learned

- **Overconfidence is a real and measurable problem** — model errors occur even at high predicted probability, motivating explicit failure forecasting.
- **Ensemble disagreement is a powerful meta-feature** — the standard deviation across LightGBM, RF, and LR predictions captures epistemic uncertainty effectively.
- **Missingness is informative, not just noise** — treating missing value counts as features added signal to the meta-model.
- **Memory management is critical at scale** — downcast dtypes and stratified sampling were essential to work within Colab's constraints.
- End-to-end pipeline design (primary model → metadata generation → meta-model → dashboard) requires careful data leakage prevention between phases.

### Future Research Directions

- **Temporal reliability monitoring:** Track reliability score drift over time to detect concept drift before accuracy degrades.
- **Conformal prediction integration:** Replace heuristic reliability scores with statistically guaranteed prediction intervals.
- **Online learning:** Retrain the meta-model continuously as new labeled transaction data arrives.
- **Multi-domain generalization:** Apply the framework to healthcare (diagnostic errors) and NLP (hallucination detection in LLMs).
- **Advanced imputation:** Replace `-999` sentinel with model-based missing value imputation to reduce noise in meta-features.
- **Explainability layer:** Integrate SHAP values to explain *why* a prediction is flagged as unreliable, not just that it is.

---

## 7. References

1. Kaggle. (2019). *IEEE-CIS Fraud Detection Competition*. https://www.kaggle.com/c/ieee-fraud-detection

2. Anthropic. (2024). *Trustworthy AI and Model Reliability*. https://www.anthropic.com

3. Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830. https://scikit-learn.org

5. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. https://arxiv.org/abs/2107.07511

7. Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML 2005.

8. Wang, C. (2026). *DATA 606 Capstone Project Guidelines*. UMBC Data Science Program.

---

*Report prepared for UMBC Data Science Master Degree Capstone — DATA 606, Spring 2026.*
*Submitted by Yuvateja Mullapudi | Instructor: Dr. Chaojie (Jay) Wang*
