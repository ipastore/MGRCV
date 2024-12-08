import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

########## PREPROCESSING ####################

# df = pd.read_csv('./data/cpu_sobel_results.csv')

# # Combine 'width' and 'height' into a single column 'width_height'
# df['width_height'] = df['width'].astype(str) + "x" + df['height'].astype(str)

# # Drop 'width' and 'height' columns as they are no longer needed
# df = df.drop(columns=['width', 'height'])

# # Reorder columns to make 'width_height' the first column
# df = df[['width_height'] + [col for col in df.columns if col != 'width_height']]

# # Group by 'width_height' and 'local_size' and compute mean of relevant columns
# df = df.groupby(['width_height']).mean().reset_index()

# # Save the processed DataFrame to a CSV file
# df.to_csv('./data/processed_cpu.csv', index=False)

# print("Processed dataset saved to 'processed_dataset.csv'.")

########## PREPROCESSING ####################

############### SEABORN PLOT ################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Configurar el estilo de Seaborn
# sns.set(style="whitegrid")

# df = pd.read_csv('./data/processed_dataset_mean.csv')

# # Define the desired order for 'width_height'
# desired_order = ["2048x4096", "1024x1024", "720x1280", "512x512", "256x512", "256x256"]

# # Convert 'width_height' into a categorical column with the desired order
# df['width_height'] = pd.Categorical(df['width_height'], categories=desired_order, ordered=True)

# # Generate a unique color for each facet
# palette = sns.color_palette("husl", len(df['width_height'].unique()))
# color_map = dict(zip(df['width_height'].unique(), palette))

# # Crear un FacetGrid para "Kernel Execution Time vs Local Size" por cada tipo de imagen
# g = sns.FacetGrid(df, col="width_height", col_wrap=3, height=4, sharey=False)
# g.map_dataframe(sns.lineplot, "l_size", "total_exec", color=None)

# # Apply colors to each facet
# for ax, width_height in zip(g.axes.flat, df['width_height'].unique()):
#     ax.lines[0].set_color(color_map[width_height])

# # Set specific x-ticks (local sizes) and apply them to all plots
# for ax in g.axes.flat:
#     ax.set_xticks([2, 4, 8, 16])  # Set tick positions
#     ax.set_xticklabels([2, 4, 8, 16])  # Set tick labels

# # Set global axis labels
# g.set_axis_labels("Local Size", "Program Execution Time (ms)")
# g.set_titles("{col_name}")

# # Force x-axis to display even for the upper rows
# for ax in g.axes.flat:
#     ax.tick_params(labelbottom=True)  # Ensure x-axis labels are shown for all rows

# # Adjust layout for better spacing
# g.tight_layout()

# # Show the plot
# plt.show()
############### SEABORN PLOT ################


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# cpu_df = pd.read_csv('./data/processed_cpu.csv')

# # Define the desired order for 'width_height'
# desired_order = ["2048x4096", "1024x1024", "720x1280", "512x512", "256x512", "256x256"]

# # Convert 'width_height' into a categorical column with the desired order
# cpu_df['width_height'] = pd.Categorical(cpu_df['width_height'], categories=desired_order, ordered=True)

# # Melt the DataFrame to long format for Seaborn
# cpu_df_melted = cpu_df.melt(id_vars=['width_height'], value_vars=['total_exec', 'kernel_exec'], 
#                             var_name='Execution Type', value_name='Execution Time')

# # Create a FacetGrid
# g = sns.FacetGrid(cpu_df_melted, col='width_height', col_wrap=3, sharey=False, height=4, aspect=1.5)

# # Map the barplot to the FacetGrid
# g.map(sns.barplot, 'Execution Type', 'Execution Time', order=['total_exec', 'kernel_exec'], palette=['blue', 'orange'])

# # Add labels and title
# g.set_axis_labels("Execution Type", "Execution Time (ms)")
# g.set_titles("{col_name}")

# # Force y-axis to display for all subfigures
# for ax in g.axes.flat:
#     ax.yaxis.set_visible(True)

# # Adjust layout for better spacing
# g.tight_layout()

# # Show the plot
# plt.show()

########### CPU PLOT #######################


######### PERFORMANCE Table #################

import pandas as pd
df = pd.read_csv('./data/processed_dataset_mean.csv')

# Chose only the 16 local size and make a mean for bandwith, throughput and memory footprint
df = df[df['l_size'] == 16]

df = df.drop(columns=['total_exec', 'kernel_exec', 'l_size'])

df = df.groupby(['width_height']).mean().reset_index()

df.to_csv('./data/processed_dataset_mean_16.csv', index=False)

######### PERFORMANCE Table #################


