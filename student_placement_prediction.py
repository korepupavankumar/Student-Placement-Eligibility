import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve


# Load the dataset
df = pd.read_csv("dataset_07_student_placement_eligibility.csv")

print(df.head())
print("\nShape of dataset:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget values:")
print(df["target"].value_counts())


# Selecting input columns and target
X = df[[
    "cgpa",
    "attendance_pct",
    "coding_score",
    "projects_completed",
    "internship_months",
    "backlogs"
]]

y = df["target"]


# Filling missing values if any
X = X.fillna(X.median())


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Scaling the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Results")
print("----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Eligible", "Eligible"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Checking feature coefficients
coef = model.coef_[0]

result = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coef,
    "Odds Ratio": np.exp(coef)
})

print("\nFeature Coefficients:")
print(result)


# Feature importance plot
plt.barh(X.columns, coef)

plt.xlabel("Coefficient")
plt.ylabel("Features")
plt.title("Logistic Regression Coefficients")
plt.axvline(0, linestyle="--")

plt.show()


# Simple interpretation
print("\nFeature Interpretation")

for i in range(len(X.columns)):
    if coef[i] > 0:
        print(X.columns[i], "has a positive effect on placement eligibility.")
    else:
        print(X.columns[i], "has a negative effect on placement eligibility.")