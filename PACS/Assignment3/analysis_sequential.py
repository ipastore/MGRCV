import pandas as pd
import matplotlib.pyplot as plt

# Load the execution times data
df = pd.read_csv("./data/pi_taylor_sequential_execution_times.csv")

# Calculate mean and standard deviation for each step size
summary_stats = df.groupby("Steps")["Execution Time (s)"].agg(["mean", "std"]).reset_index()

# Save the summary statistics to a new CSV file
output_csv_path = "./results/pi_taylor_sequential_summary_statistics.csv"
summary_stats.to_csv(output_csv_path, index=False)
print(f"Summary statistics saved to {output_csv_path}")

# Plotting mean execution time for each step size
plt.figure(figsize=(10, 6))
plt.plot(summary_stats["Steps"], summary_stats["mean"], marker='o', label="Mean Execution Time")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Steps (log scale)")
plt.ylabel("Mean Execution Time (s) (log scale)")
plt.title("Mean Execution Time vs Steps")
plt.grid(True)
plt.legend()
plt.show()

# Plotting standard deviation for each step size
plt.figure(figsize=(10, 6))
plt.plot(summary_stats["Steps"], summary_stats["std"], marker='o', color="red", label="Standard Deviation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Steps (log scale)")
plt.ylabel("Standard Deviation (s) (log scale)")
plt.title("Standard Deviation of Execution Time vs Steps")
plt.grid(True)
plt.legend()
plt.show()
