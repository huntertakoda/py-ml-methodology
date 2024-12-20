import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# generate synthetic time series data

np.random.seed(42)
time = np.arange(100)
data = np.sin(0.2 * time) + 0.5 * np.random.normal(size=time.shape)

# visualize the time series

plt.plot(time, data, label="Time Series Data")
plt.legend()
plt.show()

# prepare the dataset

sequence_length = 10  
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length])
X, y = np.array(X), np.array(y)

# split the data into train and test sets

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# build the LSTM model

model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# train the model

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# plot training history

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training History")
plt.show()

# make predictions

y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# visualize predictions

plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label="True Values")
plt.plot(range(len(y_pred_rescaled)), y_pred_rescaled, label="Predictions")
plt.legend()
plt.title("LSTM Time Series Forecasting")
plt.show()
