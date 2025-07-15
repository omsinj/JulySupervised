# (Previous section continues above...)

# Remove rows with any remaining missing values
dataset = dataset.dropna()

# Import required libraries for modeling and evaluation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Recheck for any missing values
print(dataset.isnull().sum())

# Impute any missing values just in case (should be redundant here)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

# Encode categorical variables again (repeating in case of fresh reload)
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
label_encoder = LabelEncoder()
for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Define feature set and target variable
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area',
            'Loan_Amount_Term', 'Credit_History']
target = 'LoanAmount'

# Separate dataset into X (features) and y (target)
X = dataset[features]
y = dataset[target]

# Handle potential missing values in target
y.fillna(y.median(), inplace=True)
X = X.loc[y.index]  # Ensure matching indices

# Print dimensions of the feature set and target
declared_samples = X.shape[0]
print(f"Number of samples in X: {declared_samples}")
print(f"Number of samples in y: {y.shape[0]}")

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model using MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize actual vs predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()

# Add more features (income columns) to improve the model
features += ['ApplicantIncome', 'CoapplicantIncome']
X = dataset[features]
y = dataset[target]

# Redo train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train again with new features
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate again
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Train Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

# Visualize predictions vs actual for Decision Tree
plt.scatter(y_test, y_pred2, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Decision Tree Regressor)')
plt.legend()
plt.show()

# Evaluate Decision Tree
mse_amount = mean_squared_error(y_test, y_pred2)
r2_amount = r2_score(y_test, y_pred2)
print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')

# Polynomial Regression using higher-order features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train polynomial regression model
model3 = LinearRegression()
model3.fit(X_train_poly, y_train)
y_pred3 = model3.predict(X_test_poly)

# Visualize polynomial regression predictions
plt.scatter(y_test, y_pred3, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount (Polynomial Regression)')
plt.legend()
plt.show()

# Evaluate polynomial regression
mse_amount = mean_squared_error(y_test, y_pred3)
r2_amount = r2_score(y_test, y_pred3)
print(f'Loan Amount Prediction - Mean Squared Error: {mse_amount}')
print(f'Loan Amount Prediction - R² Score: {r2_amount}')
