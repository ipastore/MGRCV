#!/bin/bash

# Set locale to ensure proper decimal handling
export LC_NUMERIC="C"

# Define the matrix sizes and iterations
matrix_sizes=(100 250 500 600 700 800 900 1000 1200 1500)
iterations=10

# Define the executables
executables=("my_eigen_matmult_gettimeofday" "standard_matrix_heap_gettimeofday")

# Create a results directory if it doesn't exist
mkdir -p ../test

# Create CSV file for output
csv_file="../test/gettimeofday_tests_results.csv"
echo "Executable,Matrix Size,Iteration,Initialization Time,Multiplication Time,Real Time,User Time,System Time" > "$csv_file"

# Loop through each matrix size
for N in "${matrix_sizes[@]}"
do
    for executable in "${executables[@]}"
    do
        for ((i=1; i<=iterations; i++))
        do
            echo "Running $executable for matrix size $N, iteration $i"

            # Call the unit test script and capture the output
            output=$(./unit_gettimeofday_tests.sh "$executable" $N)

            # Parse the output into the respective times
            init_time=$(echo "$output" | awk '{print $1}')
            mult_time=$(echo "$output" | awk '{print $2}')
            real_time=$(echo "$output" | awk '{print $3}')
            user_time=$(echo "$output" | awk '{print $4}')
            sys_time=$(echo "$output" | awk '{print $5}')

            # Write the results to the CSV file
            echo "$executable,$N,$i,$init_time,$mult_time,$real_time,$user_time,$sys_time" >> "$csv_file"
        done
    done
done

echo "All tests completed. Results saved to $csv_file"
