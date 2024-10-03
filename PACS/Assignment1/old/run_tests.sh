#!/bin/bash

# Define the matrix sizes and number of iterations
# matrix_sizes=(50 100 250 500 1000)
matrix_sizes=(2000)

iterations=1

# Define the executable names
executables=("standard_matrix_heap" "standard_matrix_heap_O2" "standard_matrix_heap_O3" "standard_matrix_heap_flattened_version" "standard_matrix_heap_flattened_version_O2" "standard_matrix_heap_flattened_version_O3" "my_eigen_matmult" "my_eigen_matmult_O2" "my_eigen_matmult_O3")

# Create a folder to store results
mkdir -p test

# Iterate over all matrix sizes
for N in "${matrix_sizes[@]}"
do
    # Create a file to store results for this matrix size
    result_file="test/results_N${N}.txt"
    echo "Results for matrix size N=${N}" > "$result_file"
    
    # Run each executable 10 times
    for executable in "${executables[@]}"
    do
        echo "Running $executable for N=$N" >> "$result_file"
        
        # Repeat test for the given number of iterations
        for ((i=1; i<=iterations; i++))
        do
            echo "Iteration $i:" >> "$result_file"
            { time ./build/"$executable" $N; } 2>> "$result_file"
            echo "----------------------------------------" >> "$result_file"
        done
    done
done

echo "All tests completed. Check the 'test' folder for results."
