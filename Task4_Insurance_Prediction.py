# ============================================
# TASK 4: PREDICTING INSURANCE CLAIM AMOUNTS
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv('insurance.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# 1. BMI vs Charges
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
plt.title('BMI vs Insurance Charges')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# 2. Age vs Charges
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.title('Age vs Insurance Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# 3. Smoker vs Charges
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='smoker', y='charges')
plt.title('Smoker vs Insurance Charges')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()

# 4. Charges Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['charges'], bins=30, kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.show()

# Encode categorical columns
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({
    'northeast': 0, 'northwest': 1,
    'southeast': 2, 'southwest': 3
})

print("\nData after encoding:")
print(df.head())

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
y_pred = model.predict(X_test)

# MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Insurance Charges')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.show()

# Conclusion
print("""
TASK 4 - KEY INSIGHTS:
1. Smokers pay significantly higher insurance charges.
2. Age and BMI both positively correlate with charges.
3. Linear Regression model trained successfully.
4. MAE and RMSE printed above show prediction error.
5. Higher BMI + smoker = highest insurance charges.
""")