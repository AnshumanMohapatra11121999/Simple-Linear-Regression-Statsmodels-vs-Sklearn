# 📊 Simple Linear Regression Using Different Approaches — Statsmodels API & Scikit-learn

> **Author:** Anshuman Mohapatra  
> **Dataset:** Advertising Dataset (TV, Radio, Newspaper → Sales)  
> **📥 Download Dataset:** [https://drive.google.com/drive/folders/19ccHQ5ekip1o33mtzkzKO4g67LaEK1cn?usp=sharing] 
> **Tools Used:** Python · Pandas · NumPy · Matplotlib · Seaborn · Statsmodels · Scikit-learn  
> **Project Type:** Supervised Learning — Regression

---

## 📌 Table of Contents

1. [Objective](#-objective)
2. [Dataset Overview](#-dataset-overview)
3. [Exploratory Data Analysis & Visualization](#-exploratory-data-analysis--visualization)
4. [Simple Linear Regression — The Theory](#-simple-linear-regression--the-theory)
5. [Model Building — Step by Step](#-model-building--step-by-step)
6. [Model Evaluation & Key Statistics](#-model-evaluation--key-statistics)
7. [Residual Analysis](#-residual-analysis)
8. [Predictions on Test Set](#-predictions-on-test-set)
9. [Sklearn vs Statsmodels — Two Approaches](#-sklearn-vs-statsmodels--two-approaches)
10. [Key Q&A — Common Doubts Addressed](#-key-qa--common-doubts-addressed)
11. [Conclusion & Key Takeaways](#-conclusion--key-takeaways)

---

## 🎯 Objective

Build a **Simple Linear Regression** model to predict **Sales** using the most appropriate predictor variable from an advertising dataset. The goal is to understand the entire workflow of a regression problem — from data exploration to model evaluation.

---

## 📋 Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | Advertising Dataset |
| **Shape** | 200 rows × 4 columns |
| **Features** | TV, Radio, Newspaper (all `float64`) |
| **Target** | Sales (`float64`) |
| **Missing Values** | None ✅ |

### Descriptive Statistics

| Statistic | TV | Radio | Newspaper | Sales |
|-----------|------|-------|-----------|-------|
| **Count** | 200 | 200 | 200 | 200 |
| **Mean** | 147.04 | 23.26 | 30.55 | 15.13 |
| **Std** | 85.85 | 14.85 | 21.78 | 5.28 |
| **Min** | 0.70 | 0.00 | 0.30 | 1.60 |
| **Max** | 296.40 | 49.60 | 114.00 | 27.00 |

### Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

---

## 🔍 Exploratory Data Analysis & Visualization

### Key Visualizations Performed

1. **Pairplot** — Scatter plots of TV, Radio, and Newspaper against Sales
2. **Heatmap** — Correlation matrix of all variables

### Insights from Correlation Analysis

| Feature Pair | Correlation |
|-------------|-------------|
| **TV ↔ Sales** | **0.90** ⭐ (Strongest) |
| Radio ↔ Sales | 0.35 |
| Newspaper ↔ Sales | 0.16 |

> **🔑 Key Finding:** The variable **TV** shows the strongest correlation with **Sales** (r ≈ 0.90). This makes TV the best candidate for Simple Linear Regression.
>
> Radio and Newspaper show weaker correlations, making them less suitable as solo predictors.

---

## 📐 Simple Linear Regression — The Theory

### The Equation

The general equation of linear regression:

$$y = c + m_1 x_1 + m_2 x_2 + \ldots + m_n x_n$$

Where:
- **y** = Response/Target variable (Sales)
- **c** = Intercept (the value of y when x = 0)
- **m₁, m₂, ..., mₙ** = Coefficients/Model parameters

For our Simple Linear Regression case:

$$\text{Sales} = c + m_1 \times \text{TV}$$

### Key Concepts Learned

| Concept | Description |
|---------|-------------|
| **Intercept (c)** | The baseline value of Sales when TV advertising spend is zero |
| **Coefficient (m)** | The change in Sales for each unit increase in TV spend |
| **OLS** | Ordinary Least Squares — minimizes the sum of squared residuals |
| **Residuals** | Difference between actual and predicted values |

---

## 🔧 Model Building — Step by Step

### Step 1: Assign Variables
```python
X = advertising['TV']    # Feature/Predictor
y = advertising['Sales'] # Target/Response
```

### Step 2: Train-Test Split (70/30)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=100
)
```

> **💡 Best Practice:** 70% training data, 30% testing data. The `random_state` parameter ensures reproducibility.

### Step 3: Add Constant & Fit Model (Statsmodels)
```python
# Add constant for intercept
X_train_sm = sm.add_constant(X_train)

# Fit the model using OLS
lr = sm.OLS(y_train, X_train_sm).fit()
```

> **⚠️ Important:** By default, `statsmodels` fits a line through the origin. You must manually use `sm.add_constant()` to include an intercept term.

### Step 4: Extract Parameters
```python
lr.params
# const    6.948683
# TV       0.054546
```

### The Fitted Equation

$$\boxed{\text{Sales} = 6.948 + 0.054 \times \text{TV}}$$

**Interpretation:** For every additional unit spent on TV advertising, Sales increase by approximately **0.054 units**.

---

## 📈 Model Evaluation & Key Statistics

### OLS Regression Summary (Training Set)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R-squared** | **0.816** | 81.6% of variance in Sales is explained by TV ✅ |
| **Adj. R-squared** | **0.814** | Adjusted for number of predictors |
| **F-statistic** | **611.2** | Very high — model is statistically significant |
| **Prob (F-stat)** | **1.52e-52** | Practically zero — fit is NOT by chance ✅ |
| **TV Coefficient** | **0.0545** | Statistically significant (p ≈ 0.000) |
| **TV t-statistic** | **24.722** | Strong evidence against null hypothesis |
| **Intercept** | **6.9487** | Statistically significant (p ≈ 0.000) |

### Key Takeaways from Summary

1. ✅ **Coefficient is significant** — The p-value for TV coefficient is practically zero, confirming the association is NOT by chance
2. ✅ **R² = 0.816** — A decent value; 81.6% of Sales variance is explained
3. ✅ **F-statistic is significant** — The overall model fit is statistically significant

---

## 🔬 Residual Analysis

### Why Residual Analysis Matters

Residual analysis validates the **assumptions** of linear regression:
- Residuals should be **normally distributed**
- Residuals should have **constant variance** (homoscedasticity)
- Residuals should be **independent**

### What Was Observed

```python
y_train_pred = lr.predict(X_train_sm)
res = y_train - y_train_pred
```

| Check | Result |
|-------|--------|
| **Normality of residuals** | Approximately normal ✅ |
| **Variance pattern** | Variance increases with X (heteroscedasticity) ⚠️ |

> **📝 Note:** The normality of residuals allows statistical inference on coefficients. However, the increasing variance of residuals with X indicates there is significant variation the model cannot explain.

---

## 🎯 Predictions on Test Set

### Process
```python
# Add constant to test set
X_test_sm = sm.add_constant(X_test)

# Predict
y_pred = lr.predict(X_test_sm)
```

### Test Set Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | **2.019** | Average prediction error of ≈2 units |
| **R² (Test)** | **0.792** | 79.2% variance explained on unseen data |

> **💡 Insight:** R² dropped slightly from 0.816 (train) to 0.792 (test), which is expected and indicates the model generalizes reasonably well without severe overfitting.

---

## ⚙️ Sklearn vs Statsmodels — Two Approaches

### Approach 1: Statsmodels
- Provides **detailed statistical summary** (p-values, confidence intervals, F-statistic)
- Requires manually adding constant via `sm.add_constant()`
- Best for **statistical inference and analysis**

### Approach 2: Scikit-learn
```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train_lm, y_train_lm)

print(lm.intercept_)  # 6.948683
print(lm.coef_)       # [0.05454575]
```

- Same equation result: **Sales = 6.948 + 0.054 × TV** ✅
- Compatible with sklearn utilities (cross-validation, grid search, pipelines)
- Best for **prediction and production pipelines**

> **⚠️ Common Error Encountered:** `'Series' object has no attribute 'reshape'`  
> **Fix:** Use `.values.reshape(-1,1)` instead of `.reshape(-1,1)` when working with Pandas Series.

---

## ❓ Key Q&A — Common Doubts Addressed

### Q1: Why is it called "R-squared"?

**Answer:** R-squared is literally the **square of Pearson's correlation coefficient (r)**.

```python
corrs = np.corrcoef(X_train, y_train)
# Correlation r = 0.903
# r² = 0.903² ≈ 0.816 → matches R-squared!
```

| Metric | Value |
|--------|-------|
| Pearson's r (correlation) | 0.903 |
| r² (R-squared) | 0.816 |

---

### Q2: What is a good RMSE?

**Answer:**
- RMSE depends on the **units of the Y variable** — it is NOT a normalized measure
- It cannot alone tell you the goodness of a model
- It is useful for **comparing models** against each other
- **R-squared is a better measure** for evaluating model goodness since it is normalized (ranges 0 to 1)

---

### Q3: Does scaling affect the model? When should I scale?

**Answer:** After applying **Standard Scaling** to the features:

| Aspect | Before Scaling | After Scaling |
|--------|---------------|---------------|
| Intercept | 6.9487 | ≈ 0 |
| Coefficient | 0.0545 | 0.9032 |
| **R-squared** | **0.816** | **0.816** ✅ |
| **F-statistic** | **611.2** | **611.2** ✅ |
| **p-values** | Same | Same ✅ |

> **🔑 Key Finding:** Model statistics and goodness of fit **remain unchanged** after scaling!

**So why scale?**
1. **Helps with interpretation** — especially in multiple regression when comparing feature importance
2. **Faster convergence** of gradient descent algorithms

---

## 🏁 Conclusion & Key Takeaways

### What I Learned

| # | Learning |
|---|---------|
| 1 | How to perform **EDA** using pairplots and heatmaps to identify the best predictor |
| 2 | The **mathematics** behind Simple Linear Regression (OLS method) |
| 3 | How to **split data** into training and testing sets (70/30 split) |
| 4 | Building regression models using **two different libraries** (statsmodels & sklearn) |
| 5 | Interpreting key statistics: **R², F-statistic, p-values, coefficients** |
| 6 | Performing **residual analysis** to validate model assumptions |
| 7 | Evaluating model performance using **RMSE and R²** on test data |
| 8 | Understanding the **impact of feature scaling** on linear regression |
| 9 | R-squared is literally the **square of Pearson's correlation coefficient** |
| 10 | RMSE is for **model comparison**, R² is for **model evaluation** |

### Model Summary

```
┌─────────────────────────────────────────────┐
│   Sales = 6.948 + 0.054 × TV               │
│                                             │
│   R² (Train) = 0.816  │  R² (Test) = 0.792 │
│   RMSE (Test) = 2.019                       │
│                                             │
│   ✅ Model is statistically significant     │
│   ✅ Coefficients are significant           │
│   ✅ Generalizes well to unseen data        │
└─────────────────────────────────────────────┘
```

---

> *These notes were prepared as a learning showcase for the Simple Linear Regression project in Python.*
