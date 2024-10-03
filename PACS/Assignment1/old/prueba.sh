#!/bin/bash

# Define the matrix size and the executable name
N=2000

executable="standard_matrix_heap_O2"

# Run the test once
echo "Running $executable for N=$N"

# # Capture time command output and display it
output=$( { /usr/bin/time -f "%e,%U,%S" ./build/"$executable" $N 2>&1; } 2>&1 )
# output=$( { LC_NUMERIC="C" /usr/bin/time -f "%e,%U,%S" ./build/"$executable" $N 2>&1; } 2>&1 )

# output=$( { /usr/bin/time -f "%0.3e,%0.3U,%0.3S" ./build/"$executable" $N 2>&1; } 2>&1 )


# output=$( { time ./build/"$executable" $N 2>&1; } )

# Print the output to debug
echo "Raw output: $output"

# Parse the output into real, user, and sys times
real_time=$(echo "$output" | awk -F',' '{print $1}')
user_time=$(echo "$output" | awk -F',' '{print $2}')
sys_time=$(echo "$output" | awk -F',' '{print $3}')

# Display the parsed times
echo "Real time: $real_time"
echo "User time: $user_time"
echo "System time: $sys_time"