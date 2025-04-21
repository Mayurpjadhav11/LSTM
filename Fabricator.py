import numpy as np
import pandas as pd

def simulate_usage(hour, high_range, mid_range, low_range):
    """
    Simulates usage based on the hour of the day.
    
    Parameters:
    - hour: Hour of the day (0 to 23).
    - ranges: A dictionary of ranges for 'high', 'mid', and 'low' usage.
    
    Returns:
    - A random integer value representing usage.
    """
    if 9 <= hour < 14:
        return np.random.randint(*high_range)
    elif 8 <= hour < 9 or 14 <= hour < 16:
        return np.random.randint(*mid_range)
    else:
        return np.random.randint(*low_range)

def generate_data(days, high_range_cpu=(80, 90), mid_range_cpu=(30, 40), low_range_cpu=(20, 30),
                  high_range_disk=(70, 85), mid_range_disk=(40, 55), low_range_disk=(10, 30),
                  start_date='2024-01-01'):
    """
    Generates a DataFrame containing CPU, Disk usage, and Number of Nodes data for a given number of days.
    
    Parameters:
    - days: Number of days to generate data for.
    - high_range_cpu: Tuple representing the high range for CPU usage.
    - mid_range_cpu: Tuple representing the mid range for CPU usage.
    - low_range_cpu: Tuple representing the low range for CPU usage.
    - high_range_disk: Tuple representing the high range for Disk usage.
    - mid_range_disk: Tuple representing the mid range for Disk usage.
    - low_range_disk: Tuple representing the low range for Disk usage.
    - start_date: Start date in 'YYYY-MM-DD' format (default '2024-01-01').
    
    Returns:
    - A DataFrame containing the generated data.
    """
    np.random.seed(24)
    hours = days * 24
    date_range = pd.date_range(start=start_date, periods=hours, freq='h')

    # Generate CPU usage and Disk usage independently using the simulate_usage function
    cpu_usage = [simulate_usage(dt.hour, high_range_cpu, mid_range_cpu, low_range_cpu) for dt in date_range]
    disk_usage = [simulate_usage(dt.hour, high_range_disk, mid_range_disk, low_range_disk) for dt in date_range]

    # Calculate Number_of_Nodes based on CPU_Usage
    number_of_nodes = [cpu // 10 for cpu in cpu_usage]

    # Combine data into DataFrame
    cpu_data = pd.DataFrame({
        'DateTime': date_range,
        'CPU_Usage': cpu_usage,
        'Disk_Usage': disk_usage,
        'Number_of_Nodes': number_of_nodes
    })

    return cpu_data

def save_data(data, file_path):
    """
    Saves the generated DataFrame to a CSV file.
    
    Parameters:
    - data: The DataFrame to be saved.
    - file_path: The path where the CSV file will be saved.
    
    Returns:
    - None
    """
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}. Fabricated {len(data)} rows.")

# Example usage:
cpu_data = generate_data(days=300)
save_data(cpu_data, "D:\SEM_3_STUDY\LSTM_PROJECT\Data\ fabricated.csv")