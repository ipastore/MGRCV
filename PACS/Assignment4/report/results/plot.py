import pandas as pd
import matplotlib.pyplot as plt

# Read results
df = pd.read_csv("results.csv")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['execution_time'], marker='o', label='Execution Time')
plt.xticks(df.index, [f"{w}x{h}" for w, h in zip(df['w_div'], df['h_div'])], rotation=45)
plt.xlabel("Region Size (w_div x h_div)")
plt.ylabel("Execution Time (s)")
plt.title("Impact of Region Sizes on Performance")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()