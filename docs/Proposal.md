# AI ReliabilityGuard: Failure Forecasting and Reliability Modeling for Machine Learning Systems

**Author:** Yuvateja Mullapudi
**Course:** DATA 606 — Capstone Project
**Instructor:** Dr. Chaojie (Jay) Wang
**Semester:** Spring 2026

---

## 1. Project Background

### What is this project about?

AI ReliabilityGuard is a **Trustworthy AI framework** designed to forecast when a machine learning model is likely to make incorrect predictions. Modern AI systems often achieve strong overall accuracy but may still produce highly confident errors — a phenomenon known as model overconfidence. These silent failures reduce trust and create serious risks when AI models are deployed in automated or decision-support environments, particularly in high-stakes domains like financial fraud detection.

This project builds an **AI self-evaluation system** that predicts the reliability of individual model predictions. Rather than improving task accuracy alone, the system learns patterns associated with model failures and estimates whether any given prediction should be trusted before it is acted upon.

The framework was fully implemented and validated using the IEEE-CIS Fraud Detection dataset, where realistic model errors naturally arise due to high dimensionality, missing data, noise, and severe class imbalance.

### Framework Components

The system consists of three integrated components:

**1. Primary Prediction Model**
An ensemble of supervised learning models — LightGBM, Random Forest, and Logistic Regression — trained using 5-fold stratified cross-validation on the IEEE-CIS dataset to classify transactions as fraudulent or legitimate. Feature selection reduced the input space from 390+ features to the top 150 by LightGBM importance score.

**2. Failure Forecasting Model**
A secondary meta-learning model (LightGBM) trained to predict whether the primary model's prediction will be correct or incorrect. It is trained on five engineered meta-features derived from the primary model's out-of-fold predictions: confidence (predicted probability), entropy (uncertainty), distance from the decision boundary, missing value count, and model disagreement (standard deviation across base model outputs).

**3. Reliability Assessment Layer**
Converts the meta-model's failure probability into a per-prediction trust score:

> **Reliability Score = 1 − P(Model Error)**

A score near **1.0** indicates a trustworthy prediction safe for automation. A score near **0.0** flags the prediction for human review. A threshold sweep analysis over [0.40, 0.95] was conducted to identify optimal operating points that balance coverage and accuracy.

---

## 2. Why Does It Matter?

Traditional machine learning evaluation focuses on aggregate metrics such as accuracy, precision, or ROC-AUC. These metrics measure overall performance but cannot explain:

- *When* a specific prediction is likely to be wrong
- *Which* individual outputs are unreliable
- Whether the model's stated confidence is actually justified

During this project, EDA confirmed that model errors occurred even at predicted probabilities above 80% — demonstrating that high confidence does not guarantee correctness. This overconfidence problem is well-documented in the Trustworthy AI literature and is especially dangerous in automated financial systems where incorrect decisions carry real consequences.

Failure forecasting addresses this gap by enabling:

- Early identification of unreliable predictions before they cause harm
- Reduction of overconfident, high-stakes errors
- Improved transparency and auditability of AI decisions
- Safer human-in-the-loop workflows — automate what is safe, route what is uncertain

This project contributes to the field of **Trustworthy AI**, where reliability, calibration, and safety are as important as raw predictive performance.

---

## 3. Research Questions

1. Can a meta-learning model accurately predict when a machine learning model will make incorrect predictions?
2. Which characteristics of model outputs and input data best indicate future prediction errors?
3. How effectively can reliability scores distinguish between trustworthy and unreliable predictions?
4. Can failure forecasting improve system safety compared to traditional accuracy-based evaluation?

---

## 4. Data Sources

### Dataset 1 — IEEE-CIS Fraud Detection Dataset (Experimental Benchmark)

| Property | Detail |
|---|---|
| Source | Kaggle IEEE-CIS Fraud Detection Competition |
| URL | https://www.kaggle.com/c/ieee-fraud-detection |
| Total Samples | ~590,000 transactions |
| Features | 390+ (transaction + identity tables joined) |
| Working Sample | 100,000 (stratified, `random_state=42`) |
| Join Key | TransactionID (left join on identity table) |
| Fraud Rate | ~3.5% (severely imbalanced) |
| Data Type | Real-world financial transaction records |

**Role:** Serves as the experimental environment to train the primary prediction model and generate realistic prediction errors for reliability analysis. The goal is not to achieve maximum fraud detection accuracy but to study AI model behavior and failure patterns under real-world complexity.

**Key features used:**

| Group | Columns | Description |
|---|---|---|
| Transaction | TransactionAmt, TransactionDT | Amount and timestamp offset |
| Product | ProductCD | Transaction product category (W/H/C/S/R) |
| Card | card1–card6 | Card network, type, issuer details |
| Device | DeviceType, DeviceInfo | Desktop or mobile, device string |
| Vesta Engineered | V1–V339 | Anonymized behavioral features |
| Counting | C1–C14 | Address/card usage counts |
| Timedelta | D1–D15 | Days since prior events |
| Match | M1–M9 | Name/address match flags |
| Identity | id_01–id_38 | Browser, OS, network identity features |

**Target variable:** `isFraud` — 0 (legitimate) / 1 (fraudulent)

---

### Dataset 2 — Model Prediction Metadata (Generated)

| Property | Detail |
|---|---|
| Source | Generated during primary model evaluation (out-of-fold predictions) |
| Training samples | 80,000 |
| Failure rate | ~2.64% |

**Role:** Primary dataset for training the failure forecasting meta-model. Transforms the problem from fraud prediction into AI behavior modeling — learning when and why the primary model fails.

**Target variable:** `Model_Error` — 1 (incorrect prediction) / 0 (correct prediction)

**Engineered meta-features:**

| Feature | Formula / Description |
|---|---|
| `confidence` | Ensemble predicted probability `p` |
| `entropy` | `−(p·log p + (1−p)·log(1−p))` — prediction uncertainty |
| `distance` | `|p − 0.5|` — margin from decision boundary |
| `missing_count` | Count of `−999` sentinel values per transaction row |
| `disagreement` | `std(LightGBM_prob, RF_prob, LR_prob)` — cross-model variance |

---

## 5. Methodology

### Phase 1 — Data Preparation

- Merged `train_transaction.csv` and `train_identity.csv` on `TransactionID` (left join)
- Applied `LabelEncoder` to all categorical features
- Imputed all missing values with `−999` sentinel value (missingness treated as informative)
- Engineered `missing_count` as an explicit numeric feature
- Applied `log1p` transformation to `TransactionAmt` to reduce skewness
- Downcast numeric dtypes to reduce memory footprint in Colab
- Stratified 80/20 train-test split on `isFraud` to preserve class ratios
- Strict data leakage prevention between primary model and meta-model phases

### Phase 2 — Primary Model Training

- Pre-selected top 150 features using LightGBM feature importance (100-estimator pre-screen)
- Trained three base models using `StratifiedKFold(n_splits=5, shuffle=True)`:
  - **LightGBM** — 200 estimators, learning rate 0.05
  - **Random Forest** — 100 estimators, `n_jobs=−1`
  - **Logistic Regression** — `max_iter=1000`
- Generated out-of-fold (OOF) predictions for each model to avoid leakage into meta-features
- Final ensemble prediction = average of LightGBM + RF + LR probabilities across all 5 folds

**Primary model results:**

| Metric | Value |
|---|---|
| ROC-AUC | **0.9027** |
| Accuracy | **97.42%** |
| Training samples per fold | ~64,000 |
| Fraud ratio | ~3.57% |

### Phase 3 — Failure Forecasting

- Defined failure as: primary ensemble prediction does not match `isFraud` ground truth
- Engineered 5 meta-features from OOF predictions (confidence, entropy, distance, missing_count, disagreement)
- Trained a secondary **LightGBM meta-model** to predict `Model_Error`
- Evaluated on held-out test meta-features

**Meta-model results:**

| Metric | Value | Interpretation |
|---|---|---|
| Failure Prediction ROC-AUC | **0.8608** | Strong discrimination of correct vs. incorrect predictions |
| Brier Score | **0.0203** | Very accurate failure probabilities |
| ECE (Expected Calibration Error) | **0.0018** | Near-perfect calibration — confidence matches reality |

### Phase 4 — Reliability Scoring

- Computed `reliability_score = 1 − meta_model_failure_probability` for every transaction
- Conducted threshold sweep over [0.40, 0.95] in increments of 0.05
- At each threshold: tracked coverage (fraction of transactions retained) and accuracy on retained subset
- **Key finding:** At threshold 0.80, the system retains ~60% of transactions with ~99% accuracy — low-reliability predictions routed to human review

---

## 6. Application — Streamlit Dashboard

An interactive web dashboard was built using **Streamlit** to demonstrate the full reliability pipeline in real time.

**Features:**

- **Single-transaction fraud verdict** — enter reliability score, error rate, and transaction amount via sidebar sliders; receive instant Fraud / Legitimate verdict with composite risk percentage
- **Aggregate monitoring** — filter by card network, card type, device, product, and transaction amount (capped at $1,000); live metrics update for fraud rate, average reliability, and error rate
- **Visual analytics (2×2 chart grid)** — reliability histogram, fraud vs. legitimate bar chart, error distribution, and reliability-vs-error scatter map with decision boundary
- **Transaction risk table** — row-level audit view of up to 30 transactions with fraud label, reliability score, and error for each

**Fraud decision rule (dashboard):**
```
IF reliability_score < 0.5 AND error > 0.2 → 🔴 FRAUD
ELSE → ✅ LEGITIMATE
```

**Tools:** Streamlit, Matplotlib, Seaborn, Python pickle, Google Colab, localtunnel

---

## 7. Results Summary

| Component | Metric | Result |
|---|---|---|
| Primary Ensemble | ROC-AUC | **0.9027** |
| Primary Ensemble | Accuracy | **97.42%** |
| Failure Forecasting Meta-Model | ROC-AUC | **0.8608** |
| Failure Forecasting Meta-Model | Brier Score | **0.0203** |
| Failure Forecasting Meta-Model | ECE | **0.0018** |
| Reliability Threshold (0.80) | Coverage | ~60% |
| Reliability Threshold (0.80) | Accuracy on retained | ~99% |

---

## 8. Outcomes Achieved

- A machine learning system that predicts its own likelihood of failure at the individual prediction level
- Near-perfect calibration between predicted failure probability and actual model correctness
- Identification of five key meta-features that signal model errors: confidence, entropy, decision boundary distance, missingness, and model disagreement
- A reliability scoring framework validated through threshold sweep analysis
- An interactive Streamlit dashboard prototype demonstrating reliability-aware AI evaluation

---

## 9. Contributions

- Practical implementation of AI failure forecasting on a large-scale real-world financial dataset
- Demonstrates AI self-monitoring capability using meta-learning on model behavior signals
- Advances reliability-aware machine learning evaluation beyond aggregate metrics
- Provides a generalizable framework applicable to healthcare (diagnostic errors), cybersecurity (intrusion detection), NLP (hallucination detection), and autonomous systems
- Shows that ensemble disagreement and prediction entropy are strong, practical signals for failure forecasting without requiring ground truth at inference time

---

## 10. Limitations & Future Work

### Limitations

- The failure forecasting model is trained on the same distribution as the primary model — out-of-distribution failures (concept drift, novel fraud patterns) may not be reliably detected
- The 100,000-sample subset may limit generalization relative to the full 590,000-row dataset
- Missing value imputation with `−999` sentinel is a simplification; model-based imputation (KNN, MICE) could improve meta-feature quality
- The dashboard fraud rule is a simplified heuristic; in production the raw `meta_test_prob` with a tuned threshold should be used

### Future Research Directions

- **Real-time deployment** with temporal drift monitoring to detect when fraud patterns shift
- **Conformal prediction** to replace heuristic reliability scores with statistically guaranteed prediction intervals
- **SHAP explainability** to explain *why* a prediction is flagged as unreliable, not just that it is
- **Online learning** — retrain the meta-model continuously as new labeled transaction data arrives
- **Cross-domain validation** — apply the framework to healthcare diagnostics and LLM hallucination detection

---

## 11. References

1. Kaggle. (2019). *IEEE-CIS Fraud Detection Competition*. https://www.kaggle.com/c/ieee-fraud-detection
2. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
4. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction*. arXiv:2107.07511.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
6. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io
7. Wang, C. (2026). *DATA 606 Capstone Project Guidelines*. UMBC Data Science Program.

---

*Submitted for DATA 606 Capstone — Spring 2026 | UMBC Data Science Master's Program*
