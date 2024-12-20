import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load the dataset

file_path = r"C:\puredata\ml_methodology_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# encode categorical features

label_encoder = LabelEncoder()
data['feature_4'] = label_encoder.fit_transform(data['feature_4'])  # encode 'High', 'Medium', 'Low'

# split the data

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# grid search optimization

grid_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters from Grid Search:", grid_search.best_params_)

# evaluate the best grid search model

best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
print("Grid Search Accuracy:", accuracy_score(y_test, y_pred_grid))
