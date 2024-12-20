import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# train a Random Forest model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# initialize SHAP explainer

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# visualize global feature importance

shap.summary_plot(shap_values.values, X_test, plot_type="bar")

# visualize individual predictions (for the first class of a specific sample)

sample_idx = 0  
shap.waterfall_plot(shap_values[sample_idx], max_display=10)

# visualize feature contributions across all samples

shap.summary_plot(shap_values.values, X_test)

# save SHAP summary plot

plt.savefig(r"C:\puredata\shap_summary_plot.png")
plt.close()
