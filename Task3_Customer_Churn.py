# ============================================
# TASK 3: CUSTOMER CHURN PREDICTION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Drop columns not needed for prediction
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical columns
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

print("\nData after encoding:")
print(df.head())

# 1. Churn Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Exited')
plt.title('Churn Count (0=Stayed, 1=Left)')
plt.xlabel('Exited')
plt.show()

# 2. Age vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Exited', y='Age')
plt.title('Age vs Churn')
plt.xlabel('Exited (0=No, 1=Yes)')
plt.show()

# 3. Geography vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Geography', hue='Exited')
plt.title('Churn by Geography')
plt.xlabel('Geography (0=France, 1=Germany, 2=Spain)')
plt.show()

# Features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Feature Importance - What Causes Churn?')
plt.show()

# Conclusion
print("""
TASK 3 - KEY INSIGHTS:
1. Dataset has 10,000 customers with 20% churn rate.
2. Age is the most important factor - older customers churn more.
3. German customers churn more than French or Spanish.
4. Credit score and balance also influence churn.
5. Random Forest model trained and evaluated successfully.
""")