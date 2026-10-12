
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display first 5 rows
print("First 5 rows:")
print(X.head())

# Display dataset shape
print("\nDataset shape:")
print(X.shape)

# Display dataset information
print("\nDataset information:")
print(X.info())

# Check missing values
print("\nMissing values:")
print(X.isnull().sum())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# DECISION TREE MODEL

# Create the model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# Make predictions
dt_pred = dt_model.predict(X_test)

# Evaluate the model
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Results")
print("---------------------")
print("Accuracy:", dt_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("Classification Report:")
print(classification_report(y_test, dt_pred, target_names=data.target_names))

# Plot Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    dt_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)
plt.title("Decision Tree")
plt.show()


# RANDOM FOREST MODEL

# Create the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("---------------------")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred, target_names=data.target_names))


# COMPARE BOTH MODELS

results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [dt_accuracy, rf_accuracy]
})

print("\nMODEL COMPARISON")
print(results)

if dt_accuracy > rf_accuracy:
    print("\nDecision Tree has higher accuracy.")
elif rf_accuracy > dt_accuracy:
    print("\nRandom Forest has higher accuracy.")
else:
    print("\nBoth models have the same accuracy.")