import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv("./data/pi_taylor_parallel_execution_times.csv")

# Calculate mean and standard deviation for each thread count
stats = df.groupby("Threads")["Execution Time (s)"].agg(["mean", "std"]).reset_index()

# Calculate Coefficient of Variation (CV) as a percentage for each thread count
stats["Coefficient"] = (stats["std"] / stats["mean"]) * 100

# Scalability analysis: calculate speedup relative to the 1-thread case
single_thread_time = stats[stats["Threads"] == 1]["mean"].values[0]
stats["Speedup"] = single_thread_time / stats["mean"]

# Exclude 16 threads from scalability analysis since it's over-subscribed
scalability_data = stats[stats["Threads"] < 16]

# Save the `stats` DataFrame to a new CSV file, including Speedup
output_csv_path = "./results/pi_taylor_parallel_statistics.csv"
stats.to_csv(output_csv_path, index=False)
print(f"Statistics saved to {output_csv_path}")

# Plotting Speedup for Scalability Analysis
plt.figure(figsize=(10, 6))
plt.plot(scalability_data["Threads"], scalability_data["Speedup"], marker='o', label="Speedup")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Scalability Analysis - Speedup vs. Threads")
plt.grid(True)
plt.legend()
plt.show()

# Plotting Coefficient of Variation
plt.figure(figsize=(10, 6))
plt.plot(stats["Threads"], stats["Coefficient"], marker='o', color='red', label="CV (%)")
plt.xlabel("Number of Threads")
plt.ylabel("Coefficient of Variation (%)")
plt.title("Variability Analysis - Coefficient of Variation vs. Threads")
plt.grid(True)
plt.legend()
plt.show()
