import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier

# Step 1: Load the dataset
file_path = "Invoices_Mock_Data_Final.csv"
df = pd.read_csv(file_path)

# Step 2: Preprocessing
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
df.drop(columns=['Invoice ID', 'Invoice Description', 'Invoice Name', 'Invoice Date', 'Rule-Based Prediction',
                 'Department', 'Project Name', 'Vendor Name', "Record Owner","Region"], inplace=True, errors='ignore')
df['Useful Life Expectancy'] = df['Useful Life Expectancy'].fillna(0)


# Clean and convert Recurring Expense? to numeric 1/0
df['Recurring Expense?'] = df['Recurring Expense?'].astype(str).str.lower().map({'true': 1, 'false': 0})
df['Recurring Expense?'] = df['Recurring Expense?'].fillna(0).astype(int)


df = df[df['True Value'].notna()]


# Step 3: Encode categorical columns
cat_cols = ['Project Type', 'Project Stage', 'True Value']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(df)

# Step 4: Split the data
X = df.drop(columns=['True Value'])
y = df['True Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Step 5: Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")


# Step 6: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Diagnostic: Check value distributions
print("\n🔍 Feature Distributions:")
for col in X.columns:
    print(f"{col}: {X[col].nunique()} unique values")
    print(X[col].value_counts())
    print("-----")


    # Diagnostic: Visualize influence of Useful Life Expectancy
df_analysis = X.copy()
df_analysis['True Value'] = y
plt.figure(figsize=(8, 5))
sns.boxplot(x='True Value', y='Useful Life Expectancy', data=df_analysis)
plt.title("Distribution of Useful Life Expectancy by True Value")
plt.show()

# Define and train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Evaluate and store results
def evaluate_model(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

results = []
best_model = None
best_f1 = 0

for name, model in models.items():
    if name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    # Save the best performing model
    score = f1_score(y_test, preds, average='weighted', zero_division=0)
    if score > best_f1:
        best_model = model
        best_f1 = score

    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    results.append(evaluate_model(name, y_test, preds))

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(results_df)

best_model = models['Decision Tree']

# Predict and save results using best model
final_preds = best_model.predict(X_test_scaled if best_model in [models['SVM'], models['XGBoost']] else X_test)
decoded_preds = label_encoders['True Value'].inverse_transform(final_preds)
decoded_true = label_encoders['True Value'].inverse_transform(y_test)


importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Decision Tree):")
print(importance_df)

# Optional: visualize
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Predictive Power")
plt.ylabel("Feature")
plt.tight_layout()
# plt.show()


test_results = X_test.copy()
test_results['True'] = decoded_true
test_results['Predicted'] = decoded_preds
print(test_results.head(10))



# Save best model and encoders
joblib.dump(best_model, 'model.pkl')
joblib.dump(label_encoders, "label_encoder.pkl")
