#!/bin/bash


# Define executable names
executables=("standard_matrix_stack" "standard_matrix_stack_O2" "standard_matrix_stack_O3" "standard_matrix_stack_flattened_array" "standard_matrix_stack_flattened_array_O2" "standard_matrix_stack_flattened_array_O3")

# # Check if running on macOS or Ubuntu
# if [[ "$OSTYPE" == "darwin"* ]]; then
time_cmd="/usr/bin/time -p"
csv_file="test/results_macos_stack.csv"
# else
#     time_cmd="/usr/bin/time"
#     csv_file="test/results_ubuntu_heap.csv"
# fi

# Define matrix sizes and iterations
iterations=10
matrix_sizes=(100 250 500 600 700 800 900 1000 1500 2000)


# Define executable names

# Create results folder
mkdir -p test

# Create CSV file with a header
echo "Executable,Matrix Size,Iteration,Real Time,User Time,System Time" > "$csv_file"

# Run tests
for N in "${matrix_sizes[@]}"
do
    for executable in "${executables[@]}"
    do
        for ((i=1; i<=iterations; i++))
        do
            echo "Running $executable for N=$N, iteration $i"
            # Use the appropriate time command in the script
            output=$( { $time_cmd ./build/"$executable" $N 2>&1; } 2>&1 )
            
            # Parse the output into variables
            real_time=$(echo "$output" | grep real | awk '{print $2}')
            user_time=$(echo "$output" | grep user | awk '{print $2}')
            sys_time=$(echo "$output" |  grep sys | awk '{print $2}')

            # Append results to CSV
            echo "$executable,$N,$i,$real_time,$user_time,$sys_time" >> "$csv_file"
        done
    done
done

echo "All tests completed. Check 'test/results.csv' for results."
