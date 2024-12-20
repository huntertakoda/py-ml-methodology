import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# generate synthetic data for demonstration

n_samples = 300
outliers_fraction = 0.1
clusters = 3

X, _ = make_blobs(n_samples=n_samples, centers=clusters, cluster_std=1.0, random_state=42)

# introduce outliers

n_outliers = int(outliers_fraction * n_samples)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-8, high=8, size=(n_outliers, X.shape[1]))
X = np.vstack([X, outliers])

# visualize data

plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Data with Outliers")
plt.show()

# fit Isolation Forest

iso_forest = IsolationForest(contamination=outliers_fraction, random_state=42)
iso_forest.fit(X)

# predict anomalies (-1: anomaly, 1: normal)

predictions = iso_forest.predict(X)
X_normal = X[predictions == 1]
X_anomalies = X[predictions == -1]

# visualize anomalies

plt.scatter(X_normal[:, 0], X_normal[:, 1], label="Normal Data", alpha=0.7)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], label="Anomalies", color="red", alpha=0.7)
plt.title("Anomaly Detection using Isolation Forest")
plt.legend()
plt.show()

# output results

results = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
results["Anomaly"] = predictions
print("Anomalies Detected:")
print(results[results["Anomaly"] == -1])
