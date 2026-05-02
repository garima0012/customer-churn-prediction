import numpy as np
import pandas as pd

np.random.seed(42)
n = 7043

data = {
    'customerID': [f'CUST-{str(i).zfill(5)}' for i in range(1, n+1)],
    'gender': np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], n),
    'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
    'tenure': np.random.randint(0, 72, n),
    'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.39, 0.39, 0.22]),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n,
        p=[0.34, 0.23, 0.22, 0.21]
    ),
    'MonthlyCharges': np.round(np.random.uniform(18, 120, n), 2),
}

# Realistic TotalCharges based on tenure and monthly
data['TotalCharges'] = np.round(
    data['tenure'] * data['MonthlyCharges'] + np.random.uniform(-50, 50, n), 2
)
data['TotalCharges'] = np.clip(data['TotalCharges'], 0, None)

# Churn logic: higher for short tenure, month-to-month, high charges
churn_prob = (
    0.05
    + 0.15 * (np.array(data['tenure']) < 12)
    + 0.20 * (np.array(data['Contract']) == 'Month-to-month')
    + 0.10 * (np.array(data['InternetService']) == 'Fiber optic')
    + 0.05 * (np.array(data['SeniorCitizen']) == 1)
    - 0.10 * (np.array(data['Contract']) == 'Two year')
)
churn_prob = churn_prob.clip(0, 0.85)
data['Churn'] = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

# Inject ~10 missing values in TotalCharges
missing_idx = np.random.choice(n, 10, replace=False)
tc = data['TotalCharges'].tolist()
for i in missing_idx:
    tc[i] = np.nan
data['TotalCharges'] = tc

df = pd.DataFrame(data)
df.to_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {(df['Churn']=='Yes').mean():.2%}")
