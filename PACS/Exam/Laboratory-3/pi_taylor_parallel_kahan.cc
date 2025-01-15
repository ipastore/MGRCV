#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>   

using my_float = float;

// Function implementing Kahan summation
void pi_taylor_chunk(std::vector<my_float> &output, size_t thread_id, size_t start_step, size_t stop_step) {
    my_float sum = 0.0f;
    my_float c = 0.0f;

    for (size_t i = start_step; i < stop_step; ++i) {
        my_float term = (i % 2 == 0 ? 1.0f : -1.0f) / (2.0f * i + 1.0f);
        my_float y = term - c; 
        my_float t = sum + y; 
        c = (t - sum) - y; 
        sum = t;
    }

    output[thread_id] = sum;
}

// Parsing steps and threads from command line
std::pair<size_t, size_t>
usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 3) {
        std::cerr << "Invalid syntax: pi_taylor_parallel <steps> <threads>" << std::endl;
        exit(1);
    }

    size_t steps = std::stoll(argv[1]);
    size_t threads = std::stoll(argv[2]);

    if (steps < threads ){
        std::cerr << "The number of steps should be larger than the number of threads" << std::endl;
        exit(1);

    }
    return std::make_pair(steps, threads);
}

// Main function
int main(int argc, const char *argv[]) {

    // Parse command line arguments
    auto ret_pair = usage(argc, argv);
    size_t steps = ret_pair.first;
    size_t threads = ret_pair.second;

    // Vector to store the partial results from each thread
    std::vector<my_float> partial_sums(threads, 0.0);

    // Vector to hold thread objects
    std::vector<std::thread> thread_pool;

    // Determine chunk size for each thread
    size_t chunk_size = steps / threads;
    // For distributing remaining steps
    size_t remaining_steps = steps % threads; 

    size_t start_step = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Create and launch threads
    for (size_t thread_id = 0; thread_id < threads; ++thread_id) {
        size_t stop_step = start_step + chunk_size + (thread_id < remaining_steps ? 1 : 0);

        // Launch the thread to compute its chunk
        thread_pool.emplace_back(pi_taylor_chunk, std::ref(partial_sums), thread_id, start_step, stop_step);

        // Update start step for the next chunk
        start_step = stop_step;
    }

    // Wait for all threads to complete
    for (auto &t : thread_pool) {
        t.join();
    }

    // Final reduction using Kahan summation
    my_float pi = 0.0f;
    my_float c = 0.0f;

    for (size_t t = 0; t < threads; ++t) {
        my_float y = partial_sums[t] - c;
        my_float t_new = pi + y;
        c = (t_new - pi) - y;
        pi = t_new;
    }

    pi *= 4.0f; 


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Output the final result
    std::cout << "For " << steps << " steps and " << threads << " threads, pi value: "
              << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
              << pi << " cost " << duration.count() << " microseconds" << std::endl;
}

