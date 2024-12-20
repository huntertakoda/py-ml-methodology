import joblib
import pandas as pd

# load the trained model

model_path = r"C:\puredata\random_forest_model.pkl"
model = joblib.load(model_path)

# load the feature-engineered dataset

feature_engineered_path = r"C:\puredata\feature_engineered_dataset.csv"
data = pd.read_csv(feature_engineered_path)

# extract the feature names

feature_names = data.drop(columns=['target']).columns

# create a sample input with matching feature names

sample_input = pd.DataFrame([[0.5, -0.3, 1.2, 1, 0.15, 0.6, -0.36, 0.25, -0.09, 1.44]], columns=feature_names)

# a prediction

prediction = model.predict(sample_input)
print("Sample Input Prediction:", prediction)
