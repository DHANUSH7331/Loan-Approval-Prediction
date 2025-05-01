import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset
df = pd.read_csv('loan_approval_dataset.csv')  # Replace with the actual path to your dataset

# Define features and target variable
X = df[['Annual_Income', 'Loan_Amount', 'Credit_Score', 'Employment_Type', 'Loan_Term', 'Existing_Loans']]

# Convert string labels to numeric (1: Approved, 0: Rejected)
y = df['Approval_Status'].map({'Approved': 1, 'Rejected': 0})  # Ensure correct column name

# One-hot encode the Employment_Type column (salaried/self-employed)
X = pd.get_dummies(X, drop_first=True)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('loan_model_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optionally save the scaler as well
with open('scaler.pkl', 'wb') as f:
    pickle.dump((scaler, X.columns.tolist()), f)

print("XGBoost model and scaler saved successfully!")
