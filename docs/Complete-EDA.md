# AI Reliability Guard  
## Failure Forecasting and Reliability Modeling for Machine Learning Systems

---

## Project Overview

This project introduces a **Trustworthy AI framework** that predicts when a machine learning model is likely to make incorrect predictions.

Instead of focusing only on accuracy, this system answers a critical question:

>  *When should we NOT trust the model?*

The project uses the **IEEE-CIS Fraud Detection dataset** to analyze model behavior and build a **failure forecasting system**.

---

## 🎯 Objectives

- 🔍 Identify when a model is likely to fail  
- 📉 Analyze patterns in incorrect predictions  
- 📊 Evaluate model confidence vs correctness  
- 🧠 Build a **meta-model** to predict model errors  
- 🔐 Develop a **reliability scoring system**

---

## 📂 Dataset Description

The dataset comes from the IEEE-CIS Fraud Detection competition.

| File | Rows | Features |
|------|------|---------|
| train_transaction | 590,540 | 394 |
| train_identity | 144,233 | 41 |
| test_transaction | 506,691 | 393 |
| test_identity | 141,907 | 41 |

- Data merged using `TransactionID`
- High dimensional (~400+ features)
- Highly imbalanced target variable

### Target Variable

- `isFraud = 1` → Fraud  
- `isFraud = 0` → Non-Fraud  

Fraud cases represent only **~3–4%** of the dataset.

---

## Data Preparation

- Merged transaction and identity datasets  
- Cleaned column names (`id-01 → id_01`)  
- Retained missing values for analysis  
- Filled missing values with `-999` for modeling  
- Sampled **100,000 rows** for efficient computation  

---

## 📊 Exploratory Data Analysis (EDA)

### Key Findings

#### 1. Missing Values
- Many features have **>80% missing values**
- Missingness is **not random**
- Acts as an important signal

#### 2. Class Imbalance
- Severe imbalance (~96% non-fraud)
- Leads to biased predictions

#### 3. Transaction Amount
- Highly skewed distribution  
- Fraud often linked to unusual values  

#### 4. Temporal Patterns
- Fraud rate changes over time  
- Indicates **concept drift**

#### 5. Feature Complexity
- Weak correlations  
- Non-linear relationships dominate  

---

## 🤖 Model Behavior Analysis (Core Contribution)

A baseline **Random Forest model** was trained to study prediction behavior.

## 📉 Failure Analysis

![AI Reliability](https://img.shields.io/badge/Focus-AI%20Reliability-blue?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green?style=for-the-badge)
![Key Idea](https://img.shields.io/badge/Key%20Idea-Failure%20Prediction-red?style=for-the-badge)

---

###  Key Insights

#### Confidence vs Error
- Errors occur across **all confidence levels**
- Even predictions with **>90% confidence can be wrong**
- **Confidence ≠ Correctness**

---

#### High-Confidence Errors
- Most critical failure cases  
- Model is **highly confident but incorrect**  
- High risk in:
  - 💳 Finance  
  - 🏥 Healthcare  
  - 🔐 Security  

---

#### Error Rate vs Confidence

> **Confidence ↑ does NOT mean Error ↓**

- Error rate does **not consistently decrease** with confidence  
- Mid/high confidence predictions can still fail  

---

#### Calibration Curve

| Ideal Model | Your Model |
|------------|-----------|
| Confidence = Reality | Overconfident predictions |
| Perfect alignment | Deviates from diagonal |

- Model is **not well-calibrated**
- Shows **overconfidence**

---

## Key Takeaways

![Accuracy](https://img.shields.io/badge/Accuracy-Not%20Enough-red?style=for-the-badge)
![Confidence](https://img.shields.io/badge/Confidence-Unreliable-orange?style=for-the-badge)
![Errors](https://img.shields.io/badge/Errors-Predictable-blue?style=for-the-badge)
![Calibration](https://img.shields.io/badge/Calibration-Critical-green?style=for-the-badge)

---

- **Accuracy alone is misleading**  
- **Model confidence is not reliable**  
- **Errors are predictable**  
- **Calibration is critical**  

---

## Core Insight

> A reliable AI system is not just accurate — it knows when it might fail.

