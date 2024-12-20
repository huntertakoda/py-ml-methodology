import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# load the feature-engineered dataset

file_path = r"C:\puredata\feature_engineered_dataset.csv"  # dataset path
data = pd.read_csv(file_path)

# split the data

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the model

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# save the model

model_path = r"C:\puredata\deep_learning_model.h5"  # output path
model.save(model_path)

print(f"Deep learning model saved to {model_path}")
