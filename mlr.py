# Machine Learning with Python - regression models
## Multiple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as stats
from sklearn.preprocessing import StandardScaler
%matplotlib inline
dataset = pd.read_csv('Loan.csv')
dataset
dataset.describe()
dataset.columns
dataset.info()
dataset.dtypes
viz =dataset[['ApplicantIncome', 'CoapplicantIncome', 
        'Credit_History', 'Property_Area']]
viz.hist()
plt.show()

plt.scatter(dataset['ApplicantIncome' ],dataset['LoanAmount'], color='green')
plt.xlabel('primary income')
plt.ylabel('Loan amount')
plt.show()
#### Few Visualizations
#. Pair Plot
#Visualize pairwise relationships in a dataset. 
#It is useful for exploring the relationships between multiple variables

# Pair plot
sns.pairplot(dataset, diag_kind='kde')
plt.show()
# Scatter plot for ApplicantIncome vs LoanAmount
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', data=dataset)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Applicant Income vs Loan Amount')
plt.show()
columns=[ 'Dependents', 'ApplicantIncome', 'CoapplicantIncome','Loan_Amount_Term', 'Property_Area']

for i in columns:
    y=dataset['LoanAmount']
    x=dataset[i]
    # creating a scatterplot
    plt.scatter(x,y)

    # add lebels and title
    plt.xlabel(i)
    plt.ylabel('Loan Amount')
    plt.title(i+' VS Loan Amount')

    # display chart
    plt.show()
### Box Plot
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x='ApplicantIncome',y='LoanAmount',data=dataset)
plt.subplot(3,3,2)
sns.boxplot(x='CoapplicantIncome',y='LoanAmount',data=dataset)
plt.subplot(3,3,3)
sns.boxplot(x='Dependents',y='LoanAmount',data=dataset)
plt.subplot(3,3,4)
sns.boxplot(x='Credit_History',y='LoanAmount',data=dataset)
plt.subplot(3,3,5)
sns.boxplot(x='Property_Area',y='LoanAmount',data=dataset)
plt.subplot(3,3,6)
sns.boxplot(x='Loan_Status',y='LoanAmount',data=dataset)
plt.show()
#Distribution Plot
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
### Handling Missing Values

dataset.isnull().sum()


# Check for missing values
print(dataset.isnull().sum())

# Impute missing values (example: fill with median for numerical columns, mode for categorical columns)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
print(dataset.isnull().sum())
#Label Encoding
#Encode Categorical Variables

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']

for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])
### Encoded dataset
We have encoded categorical attributes into numerical values
dataset.head()
#Define Features and Target Variables

# Define features
X = dataset.drop(columns=['Loan_ID', 'LoanAmount', 'Loan_Amount_Term'])

# Define target variables
y = dataset['LoanAmount']


X.head()


# Drop non-numeric columns
dataset1 = dataset.drop(columns=['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'])

# Correlation matrix
corr_matrix = dataset1.corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
flg, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (14,6))

sns.distplot(dataset['ApplicantIncome'], ax = axes[0]).set_title('ApplicantIncome Distribution')
axes[0].set_ylabel('ApplicantIncomee Count')

sns.distplot(dataset['CoapplicantIncome'], color = "r", ax = axes[1]).set_title('CoapplicantIncome Distribution')
axes[1].set_ylabel('CoapplicantIncome Count')

sns.distplot(dataset['LoanAmount'],color = "g", ax = axes[2]).set_title('LoanAmount Distribution')
axes[2].set_ylabel('LoanAmount Count')

plt.tight_layout()
plt.show()
plt.gcf().clear()

# Remove rows with missing values
dataset = dataset.dropna()

# Proceed with splitting the data and training the model


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Assuming 'dataset' is your DataFrame
# dataset = pd.read_csv('your_dataset.csv')

# Check for missing values
print(dataset.isnull().sum())

# Impute missing values
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Specify the features and target variable
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Term', 'Credit_History']
target = 'LoanAmount'

# Separate the features and target variable
X = dataset[features]
y = dataset[target]

# Check for missing values in the target variable
if y.isnull().sum() > 0:
    print("Missing values found in the target variable. Imputing missing values.")
    y.fillna(y.median(), inplace=True)

# Ensure consistent indices
X = X.loc[y.index]

# Verify the number of samples in X and y
print(f"Number of samples in X: {X.shape[0]}")
print(f"Number of samples in y: {y.shape[0]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#Train Machine Learning Models

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the model for loan amount prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse_amount = mean_squared_error(y_test, y_pred)
r2_amount = r2_score(y_test, y_pred)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
#visualize


import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()
dataset.columns


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Assuming 'dataset' is your DataFrame
# dataset = pd.read_csv('your_dataset.csv')

# Check for missing values
print(dataset.isnull().sum())

# Impute missing values
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Specify the features and target variable
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Term', 'Credit_History','ApplicantIncome', 'CoapplicantIncome']
target = 'LoanAmount'

# Separate the features and target variable
X = dataset[features]
y = dataset[target]

# Check for missing values in the target variable
if y.isnull().sum() > 0:
    print("Missing values found in the target variable. Imputing missing values.")
    y.fillna(y.median(), inplace=True)

# Ensure consistent indices
X = X.loc[y.index]

# Verify the number of samples in X and y
print(f"Number of samples in X: {X.shape[0]}")
print(f"Number of samples in y: {y.shape[0]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
from sklearn.tree import DecisionTreeRegressor

# Create and train the decision tree regressor model
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test, y_pred2, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Decision Tree Regressor)')
plt.legend()
plt.show()
# Evaluate the model
mse_amount = mean_squared_error(y_test, y_pred2)
r2_amount = r2_score(y_test, y_pred2)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
### Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# Transform the features to polynomial features
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
from sklearn.linear_model import LinearRegression

# Create and train the polynomial regression model
model3 = LinearRegression()
model3.fit(X_train_poly, y_train)
# Make predictions on the test set
y_pred3 = model3.predict(X_test_poly)
import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test, y_pred3, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Polynomial Regression)')
plt.legend()
plt.show()
# Evaluate the model
mse_amount = mean_squared_error(y_test, y_pred3)
r2_amount = r2_score(y_test, y_pred3)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
# Zscore for Loan Amount <br>


from scipy.stats import zscore

# Calculate Z-scores for ApplicantIncome, LoanAmount, and Loan_Amount_Term
dataset['ApplicantIncome_zscore'] = zscore(dataset['ApplicantIncome'])
dataset['LoanAmount_zscore'] = zscore(dataset['LoanAmount'])
dataset['Loan_Amount_Term_zscore'] = zscore(dataset['Loan_Amount_Term'])

# Define a threshold for identifying outliers
threshold = 2.9

# Filter out rows where the Z-score is greater than the threshold
dataset_no_outliers = dataset[
    (dataset['ApplicantIncome_zscore'].abs() < threshold) &
    (dataset['LoanAmount_zscore'].abs() < threshold) &
    (dataset['Loan_Amount_Term_zscore'].abs() < threshold)
]

# Drop the Z-score columns as they are no longer needed
dataset_no_outliers = dataset_no_outliers.drop(columns=['ApplicantIncome_zscore', 'LoanAmount_zscore', 'Loan_Amount_Term_zscore'])

print(dataset_no_outliers)

#Z Score for Loan term
# Now we use new dataset devoid of outliers and rerun the model we can change variables if needed.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

# Step 1: Load the Data
dataset = pd.read_csv('Loan.csv')

# Step 2: Visualize the Data
plt.scatter(dataset['ApplicantIncome'], dataset['LoanAmount'], color='green')
plt.xlabel('Primary Income')
plt.ylabel('Loan Amount')
plt.show()

# Step 3: Check for Missing Values
print(dataset.isnull().sum())

# Step 4: Impute Missing Values
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

# Step 5: Encode Categorical Variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Step 6: Remove Outliers
# Calculate Z-scores for ApplicantIncome, LoanAmount, and Loan_Amount_Term
dataset['ApplicantIncome_zscore'] = zscore(dataset['ApplicantIncome'])
dataset['LoanAmount_zscore'] = zscore(dataset['LoanAmount'])
dataset['Loan_Amount_Term_zscore'] = zscore(dataset['Loan_Amount_Term'])
dataset['CoapplicantIncome_zscore'] = zscore(dataset['CoapplicantIncome'])

# Define a threshold for identifying outliers
threshold = 2.9

# Filter out rows where the Z-score is greater than the threshold
dataset_no_outliers = dataset[
    (dataset['ApplicantIncome_zscore'].abs() < threshold) &
    (dataset['LoanAmount_zscore'].abs() < threshold) &
    (dataset['Loan_Amount_Term_zscore'].abs() < threshold) &
    (dataset['CoapplicantIncome_zscore'].abs() < threshold)
]

# Specify the features and target variable
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']
target = 'LoanAmount'

# Step 7: Define Features and Target Variables
X_cleaned = dataset_no_outliers[features]
y_cleaned = dataset_no_outliers[target]

# Drop the Z-score columns as they are no longer needed
dataset_no_outliers = dataset_no_outliers.drop(columns=['ApplicantIncome_zscore', 'LoanAmount_zscore', 'Loan_Amount_Term_zscore', 'CoapplicantIncome_zscore'])

# Check for missing values in the target variable
if y_cleaned.isnull().sum() > 0:
    print("Missing values found in the target variable. Imputing missing values.")
    y_cleaned.fillna(y_cleaned.median(), inplace=True)

X_cleaned = X_cleaned.loc[y_cleaned.index]

# Verify the number of samples in X and y
print(f"Number of samples in X: {X_cleaned.shape[0]}")
print(f"Number of samples in y: {y_cleaned.shape[0]}")

# Step 8: Split the Data into Training and Testing Sets
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=0)

# Step 9: Scale the Features
scaler = StandardScaler()

# Fit and transform the training data
X_train_n = scaler.fit_transform(X_train_n)

# Transform the testing data
X_test_n = scaler.transform(X_test_n)

# Step 10: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train_n, y_train_n)

# Step 11: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test_n)

# Evaluate the model
mse_amount = mean_squared_error(y_test_n, y_pred)
r2_amount = r2_score(y_test_n, y_pred)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
#Split the Dataset

from sklearn.model_selection import train_test_split

# Split the dataset for loan amount prediction
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=0)

scaler = StandardScaler()

# Fit and transform the training data
X_train_n = scaler.fit_transform(X_train_n)

# Transform the testing data
X_test_n = scaler.transform(X_test_n)
from sklearn.preprocessing import PolynomialFeatures

# Transform the features to polynomial features
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_n)
X_test_poly = poly.transform(X_test_n)
from sklearn.linear_model import LinearRegression

# Create and train the polynomial regression model
model4 = LinearRegression()
model4.fit(X_train_poly, y_train_n)
# Make predictions on the test set
y_pred4 = model4.predict(X_test_poly)
import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test_n, y_pred4, color='blue', label='Predicted vs Actual')
plt.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Polynomial Regression)')
plt.legend()
plt.show()
# Evaluate the model
mse_amount = mean_squared_error(y_test_n, y_pred4)
r2_amount = r2_score(y_test_n, y_pred4)

print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluation_metrics(y_test, y_pred, model_name):
    print("MSE for ", str(model_name), mean_squared_error(y_test, y_pred))
    print("R2 for ", str(model_name),r2_score(y_test, y_pred))
    print("RMSE for ", str(model_name),np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE for ", str(model_name), mean_absolute_error(y_test, y_pred))

#evaluation_metrics(y_test, y_pred, "multiple linear regression")
#evaluation_metrics(y_test, y_pred2, "Decision Tree Regression")
#evaluation_metrics(y_test, y_pred3, "polynomial linear regression")
evaluation_metrics(y_test_n, y_pred4, "polynomial linear regression with outliers removed")
### Using data without outliers for Multiple linear regression
model5 = LinearRegression()
model5.fit(X_train_n, y_train_n)

# Make predictions
y_pred_5 = model5.predict(X_test_n)
import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test_n, y_pred_5, color='blue')
plt.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], color='red', linewidth=2)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()
evaluation_metrics(y_test_n, y_pred_5, "multiple linear regression")
from sklearn.tree import DecisionTreeRegressor

# Create and train the decision tree regressor model
model6 = DecisionTreeRegressor(random_state=42)
model6.fit(X_train_n, y_train_n)
y_pred6 = model6.predict(X_test_n)
import matplotlib.pyplot as plt

# Plot actual vs predicted loan amounts
plt.scatter(y_test_n, y_pred6, color='blue', label='Predicted vs Actual')
plt.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Decision Tree Regressor)')
plt.legend()
plt.show()
evaluation_metrics(y_test_n, y_pred6, "multiple linear regression")


