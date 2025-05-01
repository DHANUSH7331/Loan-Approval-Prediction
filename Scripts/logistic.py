import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Preprocessing
df['Employment_Type_Salaried'] = df['Employment_Type'].apply(lambda x: 1 if x.lower() == 'salaried' else 0)
df['Existing_Loans'] = pd.to_numeric(df['Existing_Loans'], errors='coerce')
df['Existing_Loans_Yes'] = df['Existing_Loans'].apply(lambda x: 1 if x > 0 else 0)

# Features and target
X = df[['Annual_Income', 'Loan_Amount', 'Credit_Score', 'Employment_Type_Salaried', 'Loan_Term', 'Existing_Loans_Yes']]
y = df['Approval_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler (separately and correctly)
with open("loan_model_logistic.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
