import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the preprocessed dataset

file_path = r"C:\puredata\preprocessed_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# preview the dataset

print("Dataset Preview:")
print(data.head())

# basic statistics

print("Dataset Statistics:")
print(data.describe())

# histogram of numerical features

data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# correlation heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# class distribution

sns.countplot(x='target', data=data)
plt.title("Target Class Distribution")
plt.show()