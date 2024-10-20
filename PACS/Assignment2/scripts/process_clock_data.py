import pandas as pd
import os

# Load your CSV data
df = pd.read_csv('../test/clock_tests_results.csv')

# Group by 'Executable' and 'Matrix Size', then calculate mean and standard deviation
stats = df.groupby(['Executable', 'Matrix Size']).agg(
    mean_clock_time=('Clock Time', 'mean'),
    std_clock_time=('Clock Time', 'std'),
    mean_real_time=('Real Time', 'mean'),
    std_real_time=('Real Time', 'std'),
    mean_user_time=('User Time', 'mean'),
    std_user_time=('User Time', 'std'),
    mean_system_time=('System Time', 'mean'),
    std_system_time=('System Time', 'std')
).reset_index()

# Ensure the directory exists
os.makedirs('../test/clock', exist_ok=True)
# Save the statistics to a CSV file
stats.to_csv('../test/clock/clock_stats.csv', index=False)
print(f"Statistics of clock processed and saved to '{'../test/clock/clock_stats.csv'}'.")
# Convierte a formato LaTeX
latex_table = stats.style.to_latex()
# Guarda la tabla en un archivo .tex
with open(f"../test/clock/clock_stats.tex", "w") as f:
    f.write(latex_table)

# Get the unique executables
unique_executables = stats['Executable'].unique()

# Create individual CSV for each executable
for executable in unique_executables:
    # Filter the dataframe for the current executable
    executable_df = stats[stats['Executable'] == executable]
    
    # Save each filtered dataframe to a separate CSV file
    file_name = f'../test/clock/{executable}_stats.csv'
    executable_df.to_csv(file_name, index=False)
    print(f"Statistics for {executable} saved to '{file_name}'.")
    # Convierte a formato LaTeX
    latex_table = executable_df.style.to_latex()
    # Guarda la tabla en un archivo .tex
    with open(f"../test/clock/{executable}.tex", "w") as f:
        f.write(latex_table)
