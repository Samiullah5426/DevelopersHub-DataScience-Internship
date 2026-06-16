import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the famous Iris dataset (built into seaborn)
df = sns.load_dataset('iris')
print(df.head())

# Shape: (rows, columns)
print("Shape of dataset:", df.shape)

# Column names
print("\nColumn names:", df.columns.tolist())

# First 5 rows
df.head()

# Data types and missing values
df.info()

# Statistical summary
df.describe()

# Scatter plot: relationship between sepal length and sepal width
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Histogram: distribution of petal length
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='petal_length', bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Count')
plt.show()

# Box plot: detect outliers and spread for each feature by species
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='petal_length')
plt.title('Petal Length by Species (Box Plot)')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()
# ============================================
# CONCLUSION
# ============================================
print("""
TASK 1 - KEY INSIGHTS:
1. Dataset has 150 rows and 5 columns with NO missing values.
2. Three species: Setosa, Versicolor, Virginica.
3. Scatter Plot: Setosa is clearly separated from other species.
4. Histogram: Petal length has two groups - small (Setosa) and larger (others).
5. Box Plot: Virginica has the longest petals, Setosa has the shortest.
""")