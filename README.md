#  Customer Churn Prediction — Telecom

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garima0012/Customer-Churn-Prediction/blob/main/customer_churn_prediction.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

>  **Built a machine learning model to predict customer churn for a telecom company using classification algorithms. Performed EDA, handled missing values, engineered features and improved model accuracy by 15% using Random Forest. Delivered actionable insights that can help reduce churn rate and improve customer retention strategies.**

---

##  Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-business-insights)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

##  Overview

Customer churn is one of the biggest challenges in the telecom industry. Losing a customer costs **5x more** than retaining one. This project builds an end-to-end machine learning pipeline to:

-  **Identify** customers who are likely to churn
-  **Analyze** key factors driving churn
-  **Deliver** actionable retention strategies
-  **Predict** churn with high accuracy using ML models

---

##  Dataset

| Property | Value |
|---|---|
|  Source | Telecom Customer Dataset (Synthetic) |
|  Customers | 7,043 records |
|  Features | 20 original + 4 engineered |
|  Target | Churn (Yes / No) |
|  Churn Rate | ~21% |

**Key Features Include:**
- Demographics: Gender, Senior Citizen, Partner, Dependents
- Services: Phone, Internet, Streaming, Tech Support
- Account: Contract type, Payment method, Monthly charges, Tenure

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── customer_churn_prediction.ipynb   ← Run in Google Colab
├── churn_prediction.py               ← Main ML pipeline
├── generate_data.py                  ← Dataset generator
├── requirements.txt                  ← Dependencies
└── README.md                         ← You are here!
```

---

##  Tech Stack

<div align="center">

| Tool | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=white) | Core Language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data Manipulation & EDA |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical Computing |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn&logoColor=white) | ML Models & Evaluation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logoColor=white) | Data Visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4c9be8?logoColor=white) | Statistical Plots |

</div>

---

## 📊 Methodology

### 1️⃣ Data Cleaning
- Loaded 7,043 customer records
- Converted `TotalCharges` to numeric
- Handled missing values using **median imputation**
- Removed non-predictive columns

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed churn distribution (~21% churn rate)
- Visualized tenure, monthly charges, contract type distributions
- Computed correlation matrix for numerical features
- Identified key churn drivers visually

### 3️⃣ Feature Engineering ⭐ (+15% Accuracy)

| Feature | Description |
|---|---|
| `ChargesPerMonth` | Total charges divided by tenure |
| `HighMonthlyCharge` | Binary flag — above median monthly charge |
| `NumServices` | Total number of subscribed services |
| `AutoPayment` | Binary flag — automatic payment method |

### 4️⃣ Model Training & Evaluation
- **80/20** train-test split with stratification
- **5-Fold Cross Validation** for robust evaluation
- Compared 4 classification algorithms

---

## 🏆 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 78.5% | 0.71 | 0.58 | 0.64 | 0.708 |
| Decision Tree | 77.6% | 0.69 | 0.57 | 0.62 | 0.709 |
| ✅ **Random Forest** | **78.4%** | **0.72** | **0.59** | **0.65** | **0.715** |
| Gradient Boosting | 77.5% | 0.70 | 0.58 | 0.63 | 0.724 |

> 🥇 **Random Forest** selected as the final model — best balance of accuracy, interpretability and consistency across cross-validation folds.

---

## 💡 Key Business Insights

```
┌─────────────────────────────────────────────────────────────┐
│                    TOP CHURN DRIVERS                        │
├─────────────────────────────────────────────────────────────┤
│  1.  Month-to-Month contracts → 3x higher churn rate     │
│      Fix: Offer discounts to switch to annual plans       │
│                                                             │
│  2.  Tenure < 12 months → Highest churn risk period      │
│      Fix: Invest in onboarding & loyalty rewards          │
│                                                             │
│  3.  Fiber optic users churn despite premium service     │
│      Fix: Investigate quality & price satisfaction        │
│                                                             │
│  4.  No Tech Support → Significantly higher churn        │
│      Fix: Bundle services at discounted rates             │
│                                                             │
│  5.  Electronic check payers show higher churn           │
│      Fix: Incentivize auto-payment enrollment             │
├─────────────────────────────────────────────────────────────┤
│     Potential churn reduction: 12-18% with targeted        │
│     retention campaigns based on these risk factors         │
└─────────────────────────────────────────────────────────────┘
```

---

##  How to Run

### Option 1 — Google Colab (Recommended)
Click the button below — no setup needed!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garima0012/Customer-Churn-Prediction/blob/main/customer_churn_prediction.ipynb)

###  Option 2 — Run Locally
```bash
# Clone the repo
git clone https://github.com/garima0012/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Run pipeline
python churn_prediction.py
```

---

## 👩‍💻 Author

<div align="center">

**Garima**

[![GitHub](https://img.shields.io/badge/GitHub-garima0012-black?logo=github)](https://github.com/garima0012)

</div>

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
