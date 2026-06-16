# ============================================
# TASK 5: PERSONAL LOAN ACCEPTANCE PREDICTION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Load dataset
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# Drop ID and ZIP Code columns (not useful for prediction)
df = df.drop(['ID', 'ZIP Code'], axis=1)

print("\nData after cleaning:")
print(df.head())
print("\nLoan Acceptance Rate:")
print(df['Personal Loan'].value_counts())

# 1. Age Distribution by Loan Acceptance
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Personal Loan', bins=30, kde=True)
plt.title('Age Distribution by Loan Acceptance')
plt.xlabel('Age')
plt.show()

# 2. Income vs Loan Acceptance
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Personal Loan', y='Income')
plt.title('Income vs Loan Acceptance')
plt.xlabel('Personal Loan (0=No, 1=Yes)')
plt.ylabel('Income')
plt.show()

# 3. Education vs Loan Acceptance
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Education', hue='Personal Loan')
plt.title('Education Level vs Loan Acceptance')
plt.xlabel('Education (1=Undergrad, 2=Graduate, 3=Advanced)')
plt.show()

# 4. Job (Experience) vs Loan Acceptance
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Experience', y='Income', hue='Personal Loan')
plt.title('Experience vs Income (colored by Loan Acceptance)')
plt.show()

# Features and target
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

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
plt.title('Confusion Matrix - Loan Acceptance')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
feat_importance = pd.Series(
    model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Feature Importance - What Drives Loan Acceptance?')
plt.show()

# Conclusion
print("""
TASK 5 - KEY INSIGHTS:
1. Income is the strongest predictor of loan acceptance.
2. Higher education customers are more likely to accept loans.
3. Customers with higher income and education = best targets.
4. Decision Tree model trained and evaluated successfully.
5. Model accuracy printed above.
""")