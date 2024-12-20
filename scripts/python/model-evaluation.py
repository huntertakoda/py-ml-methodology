import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # Ensure this is imported

# load the test data

file_path = r"C:\puredata\feature_engineered_dataset.csv"  # dataset path
data = pd.read_csv(file_path)
X = data.drop(columns=['target'])
y = data['target']

# split the data

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load the trained model

model_path = r"C:\puredata\random_forest_model.pkl"  # model path
model = joblib.load(model_path)

# evaluate the model

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
