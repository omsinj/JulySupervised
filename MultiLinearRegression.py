# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For statistical data visualization
from scipy import stats as stats  # For statistical functions
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Enable inline plotting for Jupyter notebooks
%matplotlib inline

# Load the dataset
dataset = pd.read_csv('Loan 4.csv')

# Display basic statistical details of the dataset
dataset.describe()

# Visualize the distribution of selected features using histograms
viz = dataset[['ApplicantIncome', 'CoapplicantIncome', 'Credit_History', 'Property_Area']]
viz.hist()
plt.show()

# Scatter plot to visualize the relationship between ApplicantIncome and LoanAmount
plt.scatter(dataset['ApplicantIncome'], dataset['LoanAmount'], color='green')
plt.xlabel('Primary Income')
plt.ylabel('Loan Amount')
plt.show()

# Pair Plot: Visualize pairwise relationships in the dataset
# Useful for exploring relationships between multiple variables
sns.pairplot(dataset, diag_kind='kde')
plt.show()

# Scatter plot using seaborn for ApplicantIncome vs LoanAmount
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', data=dataset)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Applicant Income vs Loan Amount')
plt.show()

# Define columns for further analysis
columns = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Property_Area']

# Loop through columns to create scatter plots against LoanAmount
for i in columns:
    y = dataset['LoanAmount']
    x = dataset[i]
    plt.scatter(x, y)
    plt.xlabel(i)
    plt.ylabel('Loan Amount')
    plt.title(i + ' VS Loan Amount')
    plt.show()

# Box plots to visualize the distribution of LoanAmount against various features
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

# Distribution plot for LoanAmount
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

# Handling missing values
# Check for missing values in the dataset
print(dataset.isnull().sum())

# Impute missing values using median for numerical columns and mode for categorical columns
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

# Verify that missing values have been handled
print(dataset.isnull().sum())

# Label Encoding: Convert categorical variables into numerical format
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Display the first few rows of the encoded dataset
dataset.head()

# Define Features and Target Variables
# Define features (independent variables)
X = dataset.drop(columns=['Loan_ID', 'LoanAmount', 'Loan_Amount_Term'])

# Define target variable (dependent variable)
y = dataset['LoanAmount']

# Display the first few rows of the feature set
X.head()

# Drop non-numeric columns for correlation analysis
dataset1 = dataset.drop(columns=['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'])

# Compute the correlation matrix
corr_matrix = dataset1.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Distribution plots for ApplicantIncome, CoapplicantIncome, and LoanAmount
flg, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
sns.distplot(dataset['ApplicantIncome'], ax=axes[0]).set_title('ApplicantIncome Distribution')
axes[0].set_ylabel('ApplicantIncome Count')
sns.distplot(dataset['CoapplicantIncome'], color="r", ax=axes[1]).set_title('CoapplicantIncome Distribution')
axes[1].set_ylabel('CoapplicantIncome Count')
sns.distplot(dataset['LoanAmount'], color="g", ax=axes[2]).set_title('LoanAmount Distribution')
axes[2].set_ylabel('LoanAmount Count')
plt.tight_layout()
plt.show()
plt.gcf().clear()

# Remove rows with missing values
dataset = dataset.dropna()

# Proceed with splitting the data and training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the Linear Regression model for loan amount prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error and R² Score
mse_amount = mean_squared_error(y_test, y_pred)
r2_amount = r2_score(y_test, y_pred)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')

# Visualize the actual vs predicted loan amounts
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()

# Display the columns of the dataset
dataset.columns

# Train a Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor

# Create and train the Decision Tree Regressor model
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X_train, y_train)

# Make predictions using the Decision Tree model
y_pred2 = model2.predict(X_test)

# Visualize the actual vs predicted loan amounts for the Decision Tree model
plt.scatter(y_test, y_pred2, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Decision Tree Regressor)')
plt.legend()
plt.show()

# Evaluate the Decision Tree model's performance
mse_amount = mean_squared_error(y_test, y_pred2)
r2_amount = r2_score(y_test, y_pred2)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
