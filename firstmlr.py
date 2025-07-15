# Machine Learning with Python - Regression Models

## Multiple Linear Regression

# Import necessary libraries for data processing, visualization, and machine learning
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Plotting graphs
import seaborn as sns  # Statistical data visualization
from scipy import stats as stats  # Statistical functions
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Display plots within the notebook (if using Jupyter)
%matplotlib inline

# Load the dataset into a pandas DataFrame
dataset = pd.read_csv('Loan.csv')

# Display the dataset contents
print(dataset)

# Show basic statistical details like percentile, mean, std etc.
print(dataset.describe())

# Show the column names in the dataset
print(dataset.columns)

# Provide a concise summary of the DataFrame
print(dataset.info())

# Print data types of each column
print(dataset.dtypes)

# Visual exploration - plot histograms for selected numeric columns
viz = dataset[['ApplicantIncome', 'CoapplicantIncome', 'Credit_History', 'Property_Area']]
viz.hist()
plt.show()

# Scatter plot between ApplicantIncome and LoanAmount
plt.scatter(dataset['ApplicantIncome'], dataset['LoanAmount'], color='green')
plt.xlabel('Primary Income')
plt.ylabel('Loan Amount')
plt.show()

# Pairplot to visualize relationships between all features
sns.pairplot(dataset, diag_kind='kde')
plt.show()

# Scatter plot using seaborn to examine relation between ApplicantIncome and LoanAmount
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', data=dataset)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Applicant Income vs Loan Amount')
plt.show()

# Scatter plots of various features vs LoanAmount
columns = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Property_Area']
for i in columns:
    y = dataset['LoanAmount']
    x = dataset[i]
    plt.scatter(x, y)
    plt.xlabel(i)
    plt.ylabel('Loan Amount')
    plt.title(i + ' VS Loan Amount')
    plt.show()

# Box plots to visualize distribution and outliers
plt.figure(figsize=(20, 12))
plt.subplot(3, 3, 1)
sns.boxplot(x='ApplicantIncome', y='LoanAmount', data=dataset)
plt.subplot(3, 3, 2)
sns.boxplot(x='CoapplicantIncome', y='LoanAmount', data=dataset)
plt.subplot(3, 3, 3)
sns.boxplot(x='Dependents', y='LoanAmount', data=dataset)
plt.subplot(3, 3, 4)
sns.boxplot(x='Credit_History', y='LoanAmount', data=dataset)
plt.subplot(3, 3, 5)
sns.boxplot(x='Property_Area', y='LoanAmount', data=dataset)
plt.subplot(3, 3, 6)
sns.boxplot(x='Loan_Status', y='LoanAmount', data=dataset)
plt.show()

# Histogram + KDE for LoanAmount
sns.histplot(dataset['LoanAmount'].dropna(), kde=True)
plt.xlabel('Loan Amount')
plt.title('Distribution of Loan Amount')
plt.show()

# Count plot for Loan_Status
sns.countplot(x='Loan_Status', data=dataset)
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Count of Loan Status')
plt.show()

# Display missing values per column
print(dataset.isnull().sum())

# Fill missing values using median for numerical and mode for categorical fields
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
print(dataset.isnull().sum())

# Encode categorical columns to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Check encoding
print(dataset.head())

# Define feature matrix X and target variable y
X = dataset.drop(columns=['Loan_ID', 'LoanAmount', 'Loan_Amount_Term'])
y = dataset['LoanAmount']
print(X.head())

# Create correlation matrix and visualize as heatmap after removing categorical columns
dataset1 = dataset.drop(columns=['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'])
corr_matrix = dataset1.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Plot distributions for income and loan amount
flg, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
sns.distplot(dataset['ApplicantIncome'], ax=axes[0]).set_title('ApplicantIncome Distribution')
sns.distplot(dataset['CoapplicantIncome'], color="r", ax=axes[1]).set_title('CoapplicantIncome Distribution')
sns.distplot(dataset['LoanAmount'], color="g", ax=axes[2]).set_title('LoanAmount Distribution')
plt.tight_layout()
plt.show()
plt.gcf().clear()
