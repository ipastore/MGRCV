#!/bin/bash

# This script runs a single test for the specified executable and matrix size
executable=$1
matrix_size=$2

# Run the test and capture the clock time and time output
output=$( { /usr/bin/time -f "%e %U %S" ./$executable $matrix_size 2>&1; } )

# Extract clock time from the first line
clock_time=$(echo "$output" | grep "Seconds of execution" | awk '{print $4}')

# Extract real, user, and sys times from the second line
real_time=$(echo "$output" | tail -n 1 | awk '{print $1}')
user_time=$(echo "$output" | tail -n 1 | awk '{print $2}')
sys_time=$(echo "$output" | tail -n 1 | awk '{print $3}')

# Output the extracted values
echo "$clock_time $real_time $user_time $sys_time"