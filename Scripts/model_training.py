import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

# Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')

# Drop Full_Name (it should not be part of training or scaling)
data = data.drop(columns=['Full_Name'])

# Separate features and target
X = data.drop('Approval_Status', axis=1)
y = data['Approval_Status']

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Encode target variable
if y.dtype == 'object':
    y = le.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression()
rf = RandomForestClassifier()
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train models
log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
xgboost_model.fit(X_train_scaled, y_train)

# Save models and scaler
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(xgboost_model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('scaler.pkl', 'wb') as f:
    pickle.dump((scaler, X.columns.tolist()), f)

print("✅ Logistic, Random Forest, XGBoost models and scaler saved successfully without Full_Name!")
