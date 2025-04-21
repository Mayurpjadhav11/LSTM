import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load CS
df = pd.read_csv("D:/SEM_3_STUDY/LSTM_PROJECT/Data/ fabricated.csv", parse_dates=["DateTime"])

# Check missing
df = df.fillna(method='ffill')

# Reset index to keep DateTime as a column
df = df.reset_index(drop=True)

# Extract Date and Time
df["Date"] = df["DateTime"].dt.date
df["Time"] = df["DateTime"].dt.time

# Normalize 
numerical_cols = ['CPU_Usage', 'Disk_Usage', 'Number_of_Nodes']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[numerical_cols])
df_scaled = pd.DataFrame(scaled, columns=numerical_cols)

# Add DateTime, Date, and Time back
df_scaled["DateTime"] = df["DateTime"]
df_scaled["Date"] = df["Date"]
df_scaled["Time"] = df["Time"]

# Save scaled DataFrame
df_scaled.to_csv("D:/SEM_3_STUDY/LSTM_PROJECT/Data/scaled_data.csv", index=False)

print("✅ Preprocessed data saved successfully.")



