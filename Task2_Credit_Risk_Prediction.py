# ============================================
# TASK 2: CREDIT RISK PREDICTION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the dataset
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Basic exploration
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

df['Married'] = df['Married'].fillna(df['Married'].mode()[0])

df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 1. Loan Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['LoanAmount'], bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.show()

# 2. Loan Status by Education
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Education', hue='Loan_Status')
plt.title('Loan Status by Education')
plt.show()

# 3. Income vs Loan Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status')
plt.title('Income vs Loan Amount')
plt.show()

# Encode categorical columns
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

# Features and target
X = df[['Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Conclusion
print("""
TASK 2 - KEY INSIGHTS:
1. Dataset had missing values - filled using median/mode.
2. Credit history is the most important factor for loan approval.
3. Graduates have higher loan approval rates.
4. Logistic Regression model trained successfully.
5. Model accuracy printed above.
""")