
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print(df.head())


df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# ------------------------------
# Encode Categorical Columns
# ------------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset")
print(df.head())

# ------------------------------
# Exploratory Data Analysis
# ------------------------------

# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Gender vs Churn
plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# Contract Type Analysis
plt.figure(figsize=(8,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# Feature Selection
# ------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ------------------------------
# Feature Scaling
# ------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------
# Model Training
# ------------------------------

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(n_estimators=100,
                               random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "XGBoost":
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        ),

    "Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=42
        )
}

results = {}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nMODEL ACCURACIES")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ------------------------------
# Accuracy Comparison Table
# ------------------------------
comparison_df = pd.DataFrame(
    results.items(),
    columns=["Model", "Accuracy"]
)

print("\nModel Comparison")
print(comparison_df)

# ------------------------------
# Evaluation of Best Model
# ------------------------------
best_predictions = best_model.predict(X_test)

print("\nBest Model:", best_model_name)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        best_predictions
    )
)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    best_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title(
    f"Confusion Matrix - {best_model_name}"
)
plt.show()

# ------------------------------
# Save Best Model
# ------------------------------
joblib.dump(
    best_model,
    "churn_model.pkl"
)

print(
    f"\nBest Model Saved: {best_model_name}"
)

# ------------------------------
# Load Model and Predict
# ------------------------------
loaded_model = joblib.load(
    "churn_model.pkl"
)

sample = [X_test[0]]

prediction = loaded_model.predict(
    sample
)

print(
    "\nSample Prediction:",
    prediction
)