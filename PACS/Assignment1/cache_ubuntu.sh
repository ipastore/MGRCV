#!/bin/bash

# Define matrix sizes and iterations
matrix_sizes=(100 250 500 600 700 800 900 1000 1500 2000)
executables=("standard_matrix_heap" "standard_matrix_heap_O2" "standard_matrix_heap_O3" "standard_matrix_heap_flattened_version" "standard_matrix_heap_flattened_version_O2" "standard_matrix_heap_flattened_version_O3" "my_eigen_matmult" "my_eigen_matmult_O2" "my_eigen_matmult_O3")
iterations=3

# Create results folder
mkdir -p test

# Create CSV file with header
csv_file="test/cache_analysis_results.csv"
echo "Executable,Matrix Size,Iteration,Cache References,Cache Misses,Cache Miss Ratio" > "$csv_file"

# Run tests
for N in "${matrix_sizes[@]}"
do
    for executable in "${executables[@]}"
    do
        for ((i=1; i<=iterations; i++))
        do
            echo "Running $executable for N=$N, iteration $i"
            
            # Run Valgrind Cachegrind and capture output
            valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out ./build/"$executable" $N &> /dev/null
            
            # Check if cachegrind.out exists
            if [ ! -f cachegrind.out ]; then
                echo "Cachegrind output not found for $executable, N=$N, iteration $i"
                continue
            fi
            
            # Extract cachegrind data
            cache_refs=$(cg_annotate cachegrind.out | grep "I   refs" | awk '{print $5}')
            cache_misses=$(cg_annotate cachegrind.out | grep "D1  misses" | awk '{print $5}')
            miss_ratio=$(cg_annotate cachegrind.out | grep "D1  miss rate" | awk '{print $4}')
            
            # Debug: Print extracted values
            echo "Cache Refs: $cache_refs, Cache Misses: $cache_misses, Miss Ratio: $miss_ratio"

            # Append results to CSV if data is available
            if [ -n "$cache_refs" ] && [ -n "$cache_misses" ] && [ -n "$miss_ratio" ]; then
                echo "$executable,$N,$i,$cache_refs,$cache_misses,$miss_ratio" >> "$csv_file"
            else
                echo "$executable,$N,$i,,,," >> "$csv_file"  # Append empty if no data
            fi
        done
    done
done

echo "Cache analysis completed. Check 'test_results/cache_analysis_results.csv' for results."
