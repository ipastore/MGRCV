import pandas as pd

# Load your CSV data
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

# Get the unique executables
unique_executables = stats['Executable'].unique()

# Create individual CSV for each executable
for executable in unique_executables:
    # Filter the dataframe for the current executable
    executable_df = stats[stats['Executable'] == executable]
    
    # Save each filtered dataframe to a separate CSV file
    file_name = f'../test/macos/{executable}_stats.csv'
    executable_df.to_csv(file_name, index=False)
    print(f"Statistics for {executable} saved to '{file_name}'.")

