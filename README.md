# DevelopersHub-DataScience-Internship
Data Science and Analytics Internship Tasks
# DevelopersHub Data Science & Analytics Internship

## Task 1: Exploring and Visualizing the Iris Dataset

### Objective
Understand how to read, summarize, and visualize a dataset.

### Dataset
Iris Dataset (loaded via seaborn library)

### Approach
- Loaded dataset using pandas
- Explored structure using .shape, .columns, .head()
- Created Scatter Plot, Histogram, and Box Plot

### Results & Insights
- Dataset has 150 rows and 5 columns with no missing values
- Three flower species: Setosa, Versicolor, Virginica
- Setosa is clearly separated from other species in scatter plot
- Virginica has the longest petals, Setosa has the shortest
## Task 2: Credit Risk Prediction
### Objective
Predict whether a loan applicant will default on a loan.
### Approach
- Handled missing values using median/mode
- Visualized loan amount, education, and income
- Trained Logistic Regression model
### Results
- Credit history is the strongest predictor of loan approval
- Graduates have higher approval rates
- Model evaluated using accuracy and confusion matrix

---

## Task 3: Customer Churn Prediction
### Objective
Identify bank customers likely to leave.
### Approach
- Encoded Geography and Gender using Label Encoding
- Trained Random Forest Classifier
- Analyzed feature importance
### Results
- Age is the #1 factor driving churn
- German customers churn more than others
- Model achieved good accuracy on test data
