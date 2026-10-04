# Bank Account Fraud Detection with AdaBoost

**Multi-Source Financial Risk & Bank Account Fraud Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-AdaBoost-orange)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-BAF%20NeurIPS%202022-green)](https://arxiv.org/abs/2211.13358)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This project applies **AdaBoost** ensemble learning to the [Bank Account Fraud (BAF) dataset](https://arxiv.org/abs/2211.13358) — a NeurIPS 2022 benchmark comprising **1 million real-world bank account opening transactions** with a fraud rate of approximately **1.1%**.

The core finding is a demonstration of the **accuracy paradox**: a globally accurate model (99%) can be nearly useless for fraud detection (recall = 0.03). This project explores why that happens, what the top predictive signals are, and what it means ethically when the most important feature is a proxy for housing status.

---

## The Problem

> Standard accuracy metrics are misleading in fraud detection.

With 1.1% fraud prevalence, a model that simply predicts "no fraud" on every transaction achieves 98.9% accuracy — and catches zero fraudulent accounts. This project demonstrates that **precision, recall, and F1 for the fraud class are the metrics that matter**, and that building a model requires deliberate handling of severe class imbalance.

---

## Dataset

**Bank Account Fraud (BAF) Suite** — Jesus et al., NeurIPS 2022  
[arXiv:2211.13358](https://arxiv.org/abs/2211.13358)

| Property | Value |
|---|---|
| Records | 1,000,000 |
| Fraud rate | ~1.1% (~11,000 fraud cases) |
| Features | 32 (numeric + categorical) |
| Target | `fraud_bool` (0 = legitimate, 1 = fraud) |
| Missing values | Encoded as `-1` → replaced with `NaN` |

**Categorical features encoded:** `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`

---

## Methodology

### Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Replace BAF missing value placeholders with NaN
df.replace(-1, np.nan, inplace=True)

X = df.drop(columns=['fraud_bool'])
y = df['fraud_bool']

categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
numeric_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
```

### Train/Test Split

Stratified 80/20 split to preserve the 1.1% fraud ratio in both sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Model

```python
from sklearn.ensemble import AdaBoostClassifier

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(n_estimators=50, random_state=42))
])

model.fit(X_train, y_train)
```

AdaBoost iteratively reweights training examples, focusing subsequent estimators on the cases the previous ones misclassified — making it theoretically well-suited for imbalanced data. In practice, the degree of imbalance here (1.1%) remains a challenge that reweighting alone does not fully resolve.

---

## Results

### The Accuracy Paradox

| Metric | Value |
|---|---|
| **Overall Accuracy** | **0.99** |
| Fraud Precision | 0.54 |
| **Fraud Recall** | **0.03** |
| Fraud F1 | 0.05 |
| Legitimate Recall | ~1.00 |

**The model achieves 99% accuracy but catches only 3% of fraud cases.**

This is the accuracy paradox in practice: because legitimate accounts vastly outnumber fraudulent ones, a model can score extremely well by essentially ignoring the minority class. The high precision (0.54) tells us that *when* the model does flag fraud, it's right more than half the time — but it almost never flags it.

### Top Features by Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | `housing_status_BA` | ~0.40 |
| 2 | `current_address_months_count` | — |
| 3 | `velocity_6h` | — |
| 4 | `device_os_windows` | — |
| 5 | `has_other_cards` | — |
| 6 | `prev_address_months_count` | — |
| 7 | `device_distinct_emails_8w` | — |
| 8 | `velocity_4w` | — |
| 9 | `keep_alive_session` | — |
| 10 | `income` | — |

---

## ⚠️ Ethical Considerations

The dominant feature — `housing_status_BA` (approximately 0.40 importance) — is a direct proxy for **socioeconomic status**. This raises a critical fairness concern: a model that flags fraud based heavily on housing status may systematically disadvantage applicants from lower-income backgrounds, creating disparate impact regardless of actual fraudulent intent.

This finding is grounded in Rudin (2019): *"Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead."* For financial risk systems, interpretability is not a luxury — it is a prerequisite for accountability and regulatory compliance.

**Recommended mitigations:**
- Fairness-constrained retraining with disparate impact thresholds
- Post-hoc auditing for demographic parity across `housing_status` groups
- Feature exclusion analysis: retrain without `housing_status` and measure precision/recall tradeoff

---

## Limitations & Next Steps

| Limitation | Proposed Solution |
|---|---|
| Recall = 0.03 for fraud class | Apply **SMOTE** oversampling before training |
| Default AdaBoost hyperparameters | Grid search over `n_estimators`, `learning_rate` |
| No fairness constraints in training | α-Fairness Regularization or reweighted loss |
| Single model evaluated | Compare with XGBoost, LightGBM, Random Forest |
| No temporal validation | Implement time-based train/test split |

---

## Project Structure

```
fraud-detection-adaboost/
├── notebooks/
│   └── DSC609_Phase2_Final_Project.ipynb   # Main analysis notebook
├── reports/
│   └── Final_Project_Phase_Final_Modeling_Testing.pdf
├── requirements.txt
└── README.md
```

---

## Requirements

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
imbalanced-learn>=0.11   # for SMOTE in future work
```

Install: `pip install -r requirements.txt`

---

## References

- Jesus, S., et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022. [arXiv:2211.13358](https://arxiv.org/abs/2211.13358)
- Dietterich, T. G. (2002). *Ensemble Learning.* The Handbook of Brain Theory and Neural Networks.
- Rudin, C. (2019). *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.* Nature Machine Intelligence.

---

## Author

**Dr. Elizabeth Taylor, DBA**  
M.S. Data Science, Utica University (August 2026)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lizbtaylor-blue?style=flat&logo=linkedin)](https://linkedin.com/in/lizbtaylor/)  
[![GitHub](https://img.shields.io/badge/GitHub-ebtaylor--star-black?style=flat&logo=github)](https://github.com/ebtaylor-star)

*DSCI 609 — Machine Learning, Utica University, April 2026*  
*Instructor: Dr. Joshua White*

---

> *"The goal is not to achieve 99% accuracy. The goal is to catch fraud."*
