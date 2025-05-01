import pandas as pd
import pickle
import sys
from sklearn.preprocessing import StandardScaler

# Load models and scaler
with open('loan_model_logistic.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

with open('loan_model_rf.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

with open('loan_model_xgb.pkl', 'rb') as model_file:
    xgboost_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler, feature_columns = pickle.load(scaler_file)

# Preprocess user input
def preprocess_input(user_input):
    user_input_df = pd.DataFrame([user_input])
    user_input_df = pd.get_dummies(user_input_df)
    user_input_df = user_input_df.reindex(columns=feature_columns, fill_value=0)
    user_input_scaled = scaler.transform(user_input_df)
    return user_input_scaled

# Predict
def make_prediction(model, user_input):
    user_data_scaled = preprocess_input(user_input)
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[0][1]  # Probability of Approved (class 1)
    return prediction, probability

# Main function
def main():
    # Check if arguments are passed
    if len(sys.argv) == 8:
        # Get from arguments (Server PHP Mode)
        user_input = {
            'Annual_Income': float(sys.argv[1]),
            'Loan_Amount': float(sys.argv[2]),
            'Credit_Score': float(sys.argv[3]),
            'Employment_Type': sys.argv[4],
            'Loan_Term': int(sys.argv[5]),
            'Existing_Loans': sys.argv[6]
        }
        model_choice = int(sys.argv[7])

    else:
        # Get from input() (CMD Mode)
        user_input = {
            'Annual_Income': float(input("Enter Annual Income: ")),
            'Loan_Amount': float(input("Enter Loan Amount: ")),
            'Credit_Score': float(input("Enter Credit Score: ")),
            'Employment_Type': input("Enter Employment Type (salaried/self-employed): "),
            'Loan_Term': int(input("Enter Loan Term (years): ")),
            'Existing_Loans': input("Do you have Existing Loans? (Yes/No): ")
        }
        print("Select Model:\n1. Logistic Regression\n2. Random Forest\n3. XGBoost")
        model_choice = int(input("Enter choice (1/2/3): "))

    # Select model
    if model_choice == 1:
        model = logistic_model
    elif model_choice == 2:
        model = random_forest_model
    elif model_choice == 3:
        model = xgboost_model
    else:
        print("Invalid choice. Defaulting to Random Forest.")
        model = random_forest_model

    prediction,probability = make_prediction(model, user_input)

    # Output prediction
    print(f"{'Approved' if prediction[0] == 1 else 'Rejected'}")
    print(f"{probability * 100:.2f}") 

if __name__ == "__main__":
    main()
