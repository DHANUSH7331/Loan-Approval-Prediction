import joblib

# These are your model columns - change if needed
model_columns = [
    'Annual_Income', 
    'Loan_Amount', 
    'Credit_Score', 
    'Employment_Type', 
    'Loan_Term', 
    'Number_of_Existing_Loans'
]

# Save it
joblib.dump(model_columns, 'model_coloumns.pkl')
print("model_coloumns.pkl file saved successfully!")
