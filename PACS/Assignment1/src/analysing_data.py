import pandas as pd

# Load your CSV data
# df = pd.read_csv('../test/results_ubuntu_heap.csv')
df = pd.read_csv('../test/results_macos_heap.csv')


# Group by 'Executable' and 'Matrix Size', then calculate mean and standard deviation
stats = df.groupby(['Executable', 'Matrix Size']).agg(
    mean_real_time=('Real Time', 'mean'),
    std_real_time=('Real Time', 'std'),
    mean_user_time=('User Time', 'mean'),
    std_user_time=('User Time', 'std'),
    mean_system_time=('System Time', 'mean'),
    std_system_time=('System Time', 'std')
).reset_index()

# Save the result to a new CSV file
stats.to_csv('../test/means_macos_heap.csv', index=False)

print("Statistics saved to 'path_to_save_statistics.csv'.")