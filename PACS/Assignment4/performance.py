import subprocess
import numpy as np
import re

# Configurations to test
configs = [
    (1, 1), (2, 1), (1, 2),  # Rows and Columns
    (2, 2), (4, 4), (8, 8),  # Increasingly smaller tiles
    (128, 128), (256, 256),  # Larger tiles
]

execution_times = []

# Function to extract execution time from output
def extract_execution_time(output):
    match = re.search(r"Execution time:\s*(\d+)\s*ms", output)
    if match:
        return float(match.group(1))  # Return the time in milliseconds
    return None

# Iterate over the configurations
for w_div, h_div in configs:
    times = []
    for _ in range(3):  # Run each configuration 3 times for averaging
        # Run the command and capture its output
        result = subprocess.run(
            ["./build/smallpt_thread_pool", str(w_div), str(h_div)],
            capture_output=True, text=True
        )
        # Extract execution time from the output
        execution_time = extract_execution_time(result.stdout)
        if execution_time is not None:
            times.append(execution_time)
        else:
            print(f"Error parsing execution time for config ({w_div}, {h_div}).")
            print("Output was:")
            print(result.stdout)
    # Store the average execution time for the current configuration
    if times:
        execution_times.append(np.mean(times))
    else:
        execution_times.append(None)  # None if no valid times were recorded

# Save the results to a CSV file for plotting
with open("results.csv", "w") as file:
    file.write("w_div,h_div,execution_time\n")
    for (w_div, h_div), time in zip(configs, execution_times):
        if time is not None:
            file.write(f"{w_div},{h_div},{time:.6f}\n")
        else:
            file.write(f"{w_div},{h_div},N/A\n")

print("Testing complete. Results saved to results.csv.")