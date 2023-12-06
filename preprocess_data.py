import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the raw patient data into a pandas DataFrame
raw_data = pd.read_csv('patient_data.csv')

# Handle missing values
clean_data = raw_data.dropna()

# Normalize features
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(clean_data), columns=clean_data.columns)

# Save the preprocessed data to a new CSV file
normalized_data.to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing complete!")
