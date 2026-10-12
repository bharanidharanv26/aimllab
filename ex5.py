import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# STEP 1: LOAD DATASET
iris = load_iris()
print("Dataset Loaded Successfully!")

# STEP 2: CREATE DATAFRAME
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("\nFirst 5 Rows:")
print(df.head())

# STEP 3: EXPLORE DATASET
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# STEP 4: CHECK MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# STEP 5: CHECK AND REMOVE DUPLICATES
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("Number of Duplicate Rows After Removal:")
print(df.duplicated().sum())

print("Dataset Shape After Removing Duplicates:")
print(df.shape)

# STEP 6: SELECT TWO CLASSES
# Setosa = 0, Versicolor = 1
df = df[df["target"].isin([0, 1])].reset_index(drop=True)

print("\nSelected Classes:")
print(iris.target_names[[0, 1]])

print("\nTarget Value Counts:")
print(df["target"].value_counts())

# STEP 7: SEPARATE FEATURES AND TARGET
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.columns)

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# STEP 8: VISUALIZE DATA
plt.figure(figsize=(8, 6))

for i, name in zip([0, 1], iris.target_names[[0, 1]]):
    plt.scatter(
        X.loc[y == i, X.columns[2]],
        X.loc[y == i, X.columns[3]],
        label=name
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - Binary Classification")
plt.legend()
plt.grid(True)
plt.show()

# STEP 9: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# STEP 10: FEATURE SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# STEP 11: CREATE SVM MODEL
svm_model = SVC(
    kernel="linear",
    C=1.0
)

print("SVM Model Created Successfully!")

# STEP 12: TRAIN SVM MODEL
svm_model.fit(X_train_scaled, y_train)
print("SVM Model Trained Successfully!")

# STEP 13: PREDICTION
y_pred = svm_model.predict(X_test_scaled)

print("\nPredictions Generated Successfully!")

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.values[:10])

# STEP 14: ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# STEP 15: PRECISION, RECALL AND F1-SCORE
precision = precision_score(
    y_test, y_pred, average="weighted"
)

recall = recall_score(
    y_test, y_pred, average="weighted"
)

f1 = f1_score(
    y_test, y_pred, average="weighted"
)

print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# STEP 16: CLASSIFICATION REPORT
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names[[0, 1]]
    )
)

# STEP 17: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names[[0, 1]]
)

disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()

# STEP 18: SELECT TWO FEATURES FOR VISUALIZATION
X_2d = df.iloc[:, [2, 3]]
y_2d = df["target"]

# STEP 19: SCALE TWO FEATURES
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

# STEP 20: TRAIN LINEAR SVM
linear_svm = SVC(
    kernel="linear",
    C=1.0
)

linear_svm.fit(X_2d_scaled, y_2d)

print("\nLinear SVM for Visualization Trained Successfully!")

# STEP 21: GET HYPERPLANE PARAMETERS
w = linear_svm.coef_[0]
b = linear_svm.intercept_[0]

# STEP 22: CREATE HYPERPLANE AND MARGIN LINES
x_values = np.linspace(
    X_2d_scaled[:, 0].min() - 1,
    X_2d_scaled[:, 0].max() + 1,
    500
)

# Hyperplane
hyperplane = -(w[0] * x_values + b) / w[1]

# Positive margin
margin_positive = (
    -(w[0] * x_values + b - 1) / w[1]
)

# Negative margin
margin_negative = (
    -(w[0] * x_values + b + 1) / w[1]
)

# STEP 23: PLOT HYPERPLANE, MARGIN AND SUPPORT VECTORS
plt.figure(figsize=(10, 7))

# Plot Setosa
plt.scatter(
    X_2d_scaled[y_2d.values == 0, 0],
    X_2d_scaled[y_2d.values == 0, 1],
    label="Setosa"
)

# Plot Versicolor
plt.scatter(
    X_2d_scaled[y_2d.values == 1, 0],
    X_2d_scaled[y_2d.values == 1, 1],
    label="Versicolor"
)

# Plot hyperplane
plt.plot(
    x_values,
    hyperplane,
    linewidth=2,
    label="Hyperplane"
)

# Plot positive margin
plt.plot(
    x_values,
    margin_positive,
    linestyle="--",
    label="Margin"
)

# Plot negative margin
plt.plot(
    x_values,
    margin_negative,
    linestyle="--"
)

# Highlight support vectors
plt.scatter(
    linear_svm.support_vectors_[:, 0],
    linear_svm.support_vectors_[:, 1],
    s=120,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors"
)

plt.xlabel("Petal Length (Scaled)")
plt.ylabel("Petal Width (Scaled)")

plt.title(
    "Linear SVM: Hyperplane, Margin and Support Vectors"
)

plt.legend()
plt.grid(True)
plt.show()