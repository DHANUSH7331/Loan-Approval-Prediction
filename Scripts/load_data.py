import pandas as pd

# Path to your dataset
file_path = r'C:\xampp\htdocs\loan_prediction\loan_approval_dataset.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Print the first few rows of the dataset
print(data.head())

# Check for any missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check the shape of the dataset (number of rows and columns)
print("\nShape of the dataset:", data.shape)

# Check the column names
print("\nColumn names:", data.columns)

# Display the types of the columns
print("\nColumn data types:")
print(data.dtypes)
