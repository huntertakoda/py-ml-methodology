import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# load the dataset

file_path = "C:\puredata\ml_methodology_dataset.csv"
data = pd.read_csv(file_path)

# preview the dataset

print("Initial Dataset Preview:")
print(data.head())

# handle missing values (if any)

data.fillna(data.mean(numeric_only=True), inplace=True)

# encode categorical features

label_encoder = LabelEncoder()
data['feature_4'] = label_encoder.fit_transform(data['feature_4'])

# scale numerical features

scaler = StandardScaler()
scaled_features = ['feature_1', 'feature_2', 'feature_3']
data[scaled_features] = scaler.fit_transform(data[scaled_features])

# save the preprocessed dataset

output_path = "C:\puredata\preprocessed_dataset.csv"  # replace with the desired output path
data.to_csv(output_path, index=False)

print(f"Preprocessed dataset saved to {output_path}")