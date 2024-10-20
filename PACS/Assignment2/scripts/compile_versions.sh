#!/bin/bash

# Create the build directory if it doesn't exist
mkdir -p ../build


# Standard version in heap using clock
g++ -std=c++17  -O2 ../src/clock/standard_matrix_heap_clock.cpp -o  ../build/standard_matrix_heap_clock

# Using EIGEN library using clock
g++ -std=c++17 -O2 -I ../eigen-3.4.0 ../src/clock/my_eigen_matmult_clock.cpp -o ../build/my_eigen_matmult_clock

# Standard version in heap using gettimeofday
g++ -std=c++17  -O2 ../src/gettimeofday/standard_matrix_heap_gettimeofday.cpp -o ../build/standard_matrix_heap_gettimeofday

# Using EIGEN library using gettimeofday
g++ -std=c++17 -O2 -I ../eigen-3.4.0 ../src/gettimeofday/my_eigen_matmult_gettimeofday.cpp -o ../build/my_eigen_matmult_gettimeofday

echo "All programs compiled and placed in 'build' folder."
