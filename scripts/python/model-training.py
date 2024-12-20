import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load the feature-engineered dataset

file_path = r"C:\puredata\feature_engineered_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# split the data

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a random forest classifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# save the trained model

import joblib
model_path = r"C:\puredata\random_forest_model.pkl"  # output path
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")