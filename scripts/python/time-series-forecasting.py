import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# load the dataset

file_path = r"C:\puredata\ml_methodology_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# use one column as the time series (e.g., feature_1)

time_series = data['feature_1'].values

# visualize the time series data

plt.plot(time_series)
plt.title("Time Series Data: feature_1")
plt.show()

# scale the data to (0, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.reshape(-1, 1))

# create sequences for LSTM

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60  
X, y = create_sequences(scaled_data, sequence_length)

# split into training and testing sets

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# reshape input data to fit LSTM [samples, time steps, features]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# build the LSTM model

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

# train the model

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# plot training and validation loss

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

# make predictions

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # invert scaling

# plot actual vs predicted values

actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))  # invert scaling
plt.plot(actual_values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted")
plt.show()

