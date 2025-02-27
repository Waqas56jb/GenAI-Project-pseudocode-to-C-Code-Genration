import pandas as pd

# Load the dataset
file_path = "spoc-train.tsv"  # Update with the correct path
df = pd.read_csv(file_path, sep='\t')

# Display basic information
print("Dataset Overview:\n")
print(df.info())  # Shows column names, null values, and data types

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())  # Displays the first few rows

print("\nSummary of Null Values:\n")
print(df.isnull().sum())  # Shows count of missing values in each column
