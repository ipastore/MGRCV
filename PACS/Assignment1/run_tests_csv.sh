#!/bin/bash

# Define the matrix sizes and number of iterations
matrix_sizes=(10 100 1000)
iterations=10

# Define the executable names
executables=("standard_matrix" "standard_matrix_O2" "standard_matrix_O3" "my_eigen_matmult" "my_eigen_matmult_O2" "my_eigen_matmult_O3" "static_standard_matrix" "standard_matrix_threading" "my_eigen_matmult_threading")

# Create a folder to store results
mkdir -p test_results

# Create CSV-like formatted .txt file and header
csv_file="test_results/results.csv.txt"
echo "Executable,Matrix Size,Iteration,Real Time,User Time,System Time" > "$csv_file"

# Iterate over all matrix sizes
for N in "${matrix_sizes[@]}"
do
    # Run each executable 10 times
    for executable in "${executables[@]}"
    do
        for ((i=1; i<=iterations; i++))
        do
            echo "Running $executable for N=$N, iteration $i"
            # Capture time command output and append to CSV-like formatted .txt
            { /usr/bin/time -f "$executable,$N,$i,%e,%U,%S" ./build/$executable $N; } 2>> "$csv_file"
        done
    done
done

echo "All tests completed. Check 'test_results/results.csv.txt' for results."