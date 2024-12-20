import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# load the preprocessed dataset

file_path = r"C:\puredata\preprocessed_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# create interaction terms

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(data[['feature_1', 'feature_2', 'feature_3']])
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(['feature_1', 'feature_2', 'feature_3']))

# concatenate interaction terms with original dataset

data = pd.concat([data, interaction_df], axis=1)

# save the dataset with new features

output_path = r"C:\puredata\feature_engineered_dataset.csv"  # output path
data.to_csv(output_path, index=False)

print(f"Feature engineered dataset saved to {output_path}")
