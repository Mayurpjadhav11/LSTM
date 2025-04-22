import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


data_visual = pd.read_csv("D:\SEM_3_STUDY\Projects\LSTM_PROJECT\Data\scaled_data.csv")

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

# Load data
data_visual = pd.read_csv("D:/SEM_3_STUDY/Projects/LSTM_PROJECT/Data/scaled_data.csv")

# Scatter plot of CPU Usage vs Hour_Label
plt.figure(figsize=(10, 6))
plt.scatter(data_visual["Hour_Label"], data_visual["CPU_Usage"], color="red", marker="x")
plt.xlabel("Hour_Label")
plt.ylabel("CPU_Usage")
plt.title("CPU Usage by Hour")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()






hour_order = [f"T{i}" for i in range(1, 25)]

# Create pivot and apply correct order
heatmap_data = data_visual.pivot(index="Day_Label", columns="Hour_Label", values="CPU_Usage")
heatmap_data = heatmap_data[hour_order]

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, linewidths=0.5)
plt.title("Heatmap of CPU Usage (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day")
plt.tight_layout()
plt.show()






plt.figure(figsize=(12, 6))
sns.boxplot(data=data_visual, x="Hour_Label", y="CPU_Usage")
plt.title("Distribution of CPU Usage by Hour")
plt.xlabel("Hour (T1 to T24)")
plt.ylabel("CPU Usage (Scaled)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
