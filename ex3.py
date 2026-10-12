
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("housing.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display dataset information
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(
    df['total_bedrooms'].median()
)

print("\nMissing values after handling:")
print(df.isnull().sum())


# SIMPLE LINEAR REGRESSION

# Select one input feature
X_simple = df[['median_income']]
y = df['median_house_value']

# Split the data
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train_simple))
print("Testing samples:", len(X_test_simple))

# Create the model
simple_model = LinearRegression()

# Train the model
simple_model.fit(X_train_simple, y_train_simple)

print("Simple Linear Regression model trained successfully!")

# Make predictions
y_pred_simple = simple_model.predict(X_test_simple)

# Evaluate the model
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("Simple Linear Regression Results")
print("----------------------------------")
print("MSE:", mse_simple)
print("R2 Score:", r2_simple)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_simple, y_pred_simple, alpha=0.5)

plt.plot(
    [y_test_simple.min(), y_test_simple.max()],
    [y_test_simple.min(), y_test_simple.max()]
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Simple Linear Regression: Actual vs Predicted")
plt.show()


# MULTIPLE LINEAR REGRESSION

# Select multiple input features
X_multiple = df[
    [
        'longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income'
    ]
]

y_multiple = df['median_house_value']

# Split the data
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple, y_multiple, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train_multiple))
print("Testing samples:", len(X_test_multiple))

# Create the model
multiple_model = LinearRegression()

# Train the model
multiple_model.fit(X_train_multiple, y_train_multiple)

print("Multiple Linear Regression model trained successfully!")

# Make predictions
y_pred_multiple = multiple_model.predict(X_test_multiple)

# Evaluate the model
mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple)
r2_multiple = r2_score(y_test_multiple, y_pred_multiple)

print("Multiple Linear Regression Results")
print("-----------------------------------")
print("MSE:", mse_multiple)
print("R2 Score:", r2_multiple)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_multiple, y_pred_multiple, alpha=0.5)

plt.plot(
    [y_test_multiple.min(), y_test_multiple.max()],
    [y_test_multiple.min(), y_test_multiple.max()]
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.show()


# COMPARE BOTH MODELS

results = pd.DataFrame({
    'Model': [
        'Simple Linear Regression',
        'Multiple Linear Regression'
    ],
    'MSE': [mse_simple, mse_multiple],
    'R2 Score': [r2_simple, r2_multiple]
})

print(results)

print("\nMODEL COMPARISON")

print("\nSimple Linear Regression:")
print("MSE:", mse_simple)
print("R2 Score:", r2_simple)

print("\nMultiple Linear Regression:")
print("MSE:", mse_multiple)
print("R2 Score:", r2_multiple)

print("\nCOMPARISON RESULT")

if mse_simple < mse_multiple:
    print("Simple Linear Regression has lower MSE.")
else:
    print("Multiple Linear Regression has lower MSE.")

if r2_simple > r2_multiple:
    print("Simple Linear Regression has higher R2.")
else:
    print("Multiple Linear Regression has higher R2.")
