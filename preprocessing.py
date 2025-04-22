import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load CS
df = pd.read_csv("D:\SEM_3_STUDY\Projects\LSTM_PROJECT\Data\ fabricated.csv", parse_dates=["DateTime"])

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



df_scaled["Hour_Label"] = ["T" + str((i % 24) + 1) for i in range(len(df_scaled))]


unique_dates = df_scaled["Date"].unique()
date_mapping = {date: f"D{i+1}" for i, date in enumerate(unique_dates)}


df_scaled["Day_Label"] = df_scaled["Date"].map(date_mapping)


#rearder
cols = df_scaled.columns.tolist()
cols_reordered = ['Day_Label', 'Hour_Label'] + [i for i in cols if i not in ['Day_Label', 'Hour_Label']]
df_scaled = df_scaled[cols_reordered]
df_scaled.set_index("Day_Label", inplace=False)

#drip
for col in ["DateTime", "Date", "Time"]:
    if col in df_scaled.columns:
        df_scaled.drop(col, axis=1, inplace=True)


# Save scaled DataFrame
df_scaled.to_csv("D:\SEM_3_STUDY\Projects\LSTM_PROJECT\Data\scaled_data.csv", index=False)

print("✅ Preprocessed data saved successfully.")

