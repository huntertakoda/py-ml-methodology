import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load the dataset

file_path = r"C:\puredata\ml_methodology_dataset.csv"  
data = pd.read_csv(file_path)

# encode categorical features

label_encoder = LabelEncoder()
data['feature_4'] = label_encoder.fit_transform(data['feature_4'])  

# split the data

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define simplified base learners

base_learners = [
    ('random_forest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),  # Simplified Random Forest
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),  # Simplified Gradient Boosting
    ('svm', SVC(probability=True, kernel='linear', random_state=42))  # SVM remains as is
]

# define stacking classifier with reduced cross-validation

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),  # Final estimator
    cv=3  
)

# train the stacking model

stacking_model.fit(X_train, y_train)

# evaluate the model

y_pred = stacking_model.predict(X_test)
print("Stacking Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))