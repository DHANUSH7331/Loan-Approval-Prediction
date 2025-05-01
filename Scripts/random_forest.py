import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset (adjust the path to your dataset)
df = pd.read_csv('loan_approval_dataset.csv')  # Replace with your actual dataset

# Define features and target variable
X = df[['Annual_Income', 'Loan_Amount', 'Credit_Score', 'Employment_Type', 'Loan_Term', 'Existing_Loans']]
y = df['Approval_Status']  # Assuming 'Loan_Status' is the target (0: Rejected, 1: Approved)

# One-hot encode the Employment_Type column (salaried/self-employed)
X = pd.get_dummies(X, drop_first=True)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('loan_model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optionally save the scaler as well
with open('scaler.pkl', 'wb') as f:
    pickle.dump((scaler, X.columns.tolist()), f)

print("Model and scaler saved successfully!")
