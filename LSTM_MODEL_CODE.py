import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your scaled data
df = pd.read_csv("D:/SEM_3_STUDY/Projects/LSTM_PROJECT/Data/scaled_data.csv")

# Sort data
df["Hour_Number"] = df["Hour_Label"].apply(lambda x: int(x[1:]))
df = df.sort_values(by=["Day_Label", "Hour_Number"]).reset_index(drop=True)

# Extract input features and create sequences
features = ['CPU_Usage', 'Disk_Usage', 'Number_of_Nodes']
data = df[features].values

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # CPU_Usage
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(data, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test[:], label="Actual")
plt.plot(y_pred[:], label="Predicted")
plt.title("CPU Usage Prediction (Test Samples)")
plt.xlabel("Time Step")
plt.ylabel("CPU Usage (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
