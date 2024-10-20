import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace with your actual path)
df_ubu = pd.read_csv('../test/means_ubuntu_heap.csv')
df_mac = pd.read_csv('../test/means_macos_heap.csv')

# Set up the figure and subplots for 3 different graphs
fig, ax = plt.subplots(3, 1, figsize=(10, 18))


# Plot Mean Real Time
# for executable in df_ubu['Executable'].unique():
#     subset = df_ubu[df_ubu['Executable'] == executable]
#     ax.errorbar(subset['Matrix Size'], subset['mean_real_time'], yerr=subset['std_real_time'], label=executable)

executable = 'standard_matrix_heap'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_O2'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_O3'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[0].set_xlabel('Matrix Size')
ax[0].set_ylabel('Mean Real Time (s)')
ax[0].set_title('Mean Real Time vs Matrix Size')
ax[0].legend()
ax[0].grid(True)

######################################3

executable = 'standard_matrix_heap_flattened'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_flattened_version_O2'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_flattened_version_O3'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[1].set_xlabel('Matrix Size')
ax[1].set_ylabel('Mean Real Time (s)')
ax[1].set_title('Mean Real Time vs Matrix Size')
ax[1].legend()
ax[1].grid(True)

######################################

executable = 'my_eigen_matmult'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'my_eigen_matmult_O2'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'my_eigen_matmult_O3'
subset = df_ubu[df_ubu['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[2].set_xlabel('Matrix Size')
ax[2].set_ylabel('Mean Real Time (s)')
ax[2].set_title('Mean Real Time vs Matrix Size')
ax[2].legend()
ax[2].grid(True)

# # Adjust layout and show the plots
plt.tight_layout()
plt.figure()

# Set up the figure and subplots for 3 different graphs
fig, ax = plt.subplots(3, 1, figsize=(10, 18))

########################################################################################################################3
# Plot Mean Real Time
# for executable in df_ubu['Executable'].unique():
#     subset = df_ubu[df_ubu['Executable'] == executable]
#     ax.errorbar(subset['Matrix Size'], subset['mean_real_time'], yerr=subset['std_real_time'], label=executable)

executable = 'standard_matrix_heap'
subset = df_mac[df_mac['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_O2'
subset = df_mac[df_mac['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_O3'
subset = df_mac[df_mac['Executable'] == executable]
ax[0].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[0].set_xlabel('Matrix Size')
ax[0].set_ylabel('Mean Real Time (s)')
ax[0].set_title('Mean Real Time vs Matrix Size')
ax[0].legend()
ax[0].grid(True)

######################################3

executable = 'standard_matrix_heap_flattened'
subset = df_mac[df_mac['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_flattened_version_O2'
subset = df_mac[df_mac['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'standard_matrix_heap_flattened_version_O3'
subset = df_mac[df_mac['Executable'] == executable]
ax[1].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[1].set_xlabel('Matrix Size')
ax[1].set_ylabel('Mean Real Time (s)')
ax[1].set_title('Mean Real Time vs Matrix Size')
ax[1].legend()
ax[1].grid(True)

######################################

executable = 'my_eigen_matmult'
subset = df_mac[df_mac['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'my_eigen_matmult_O2'
subset = df_mac[df_mac['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

executable = 'my_eigen_matmult_O3'
subset = df_mac[df_mac['Executable'] == executable]
ax[2].plot(subset['Matrix Size'], subset['mean_real_time'], label=executable)

ax[2].set_xlabel('Matrix Size')
ax[2].set_ylabel('Mean Real Time (s)')
ax[2].set_title('Mean Real Time vs Matrix Size')
ax[2].legend()
ax[2].grid(True)

# # Adjust layout and show the plots
plt.tight_layout()
plt.show()

# # Plot Mean Real Time
# for executable in df_ubu['Executable'].unique():
#     subset = df_ubu[df_ubu['Executable'] == executable]
#     axs[0].errorbar(subset['Matrix Size'], subset['mean_real_time'], yerr=subset['std_real_time'], label=executable)

# axs[0].set_xlabel('Matrix Size')
# axs[0].set_ylabel('Mean Real Time (s)')
# axs[0].set_title('Mean Real Time vs Matrix Size')
# axs[0].legend()
# axs[0].grid(True)

# # Plot Mean User Time
# for executable in df_ubu['Executable'].unique():
#     subset = df_ubu[df_ubu['Executable'] == executable]
#     axs[1].errorbar(subset['Matrix Size'], subset['mean_user_time'], yerr=subset['std_user_time'], label=executable)

# axs[1].set_xlabel('Matrix Size')
# axs[1].set_ylabel('Mean User Time (s)')
# axs[1].set_title('Mean User Time vs Matrix Size')
# axs[1].legend()
# axs[1].grid(True)

# # Plot Mean System Time
# for executable in df_ubu['Executable'].unique():
#     subset = df_ubu[df_ubu['Executable'] == executable]
#     axs[2].errorbar(subset['Matrix Size'], subset['mean_system_time'], yerr=subset['std_system_time'], label=executable)

# axs[2].set_xlabel('Matrix Size')
# axs[2].set_ylabel('Mean System Time (s)')
# axs[2].set_title('Mean System Time vs Matrix Size')
# axs[2].legend()
# axs[2].grid(True)

# # Adjust layout and show the plots
# plt.tight_layout()
# plt.show()

