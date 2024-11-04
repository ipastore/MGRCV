
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>

using my_float = long double;

typedef struct {
    size_t large_chunk;
    size_t small_chunk;
    size_t split_item;
} chunk_info;

void pi_taylor_chunk(std::vector<my_float> &output, size_t thread_id, size_t start_step, size_t stop_step) {
		my_float partial_sum = 0.0;
		int sign = (start_step % 2 == 0) ? 1 : -1;
		
		for (size_t n = start_step; n < stop_step; ++n) {
        partial_sum += sign / (2.0 * n + 1.0);
        sign = -sign;  // alternate between +1 and -1
    }
    output[thread_id] = partial_sum;
 }

// Que es constexpr ????
constexpr chunk_info split_evenly(size_t steps, size_t threads) {
	return {steps / threads + 1, steps / threads, steps % threads};
}

std::pair<size_t, size_t> get_chunk_begin_end(const chunk_info& ci, size_t index) {
    size_t begin = 0, end = 0;
    if (index < ci.split_item) {
        begin = index * ci.large_chunk;
        end = begin + ci.large_chunk;
    } else {
        begin = ci.split_item * ci.large_chunk + (index - ci.split_item) * ci.small_chunk;
        end = begin + ci.small_chunk;
    }
    return std::make_pair(begin, end);
}

std::pair<size_t, size_t> usage(int argc, const char *argv[]) {
    // read the number of steps from the command line
    if (argc != 3) {
        std::cerr << "Invalid syntax: pi_taylor <steps> <threads>" << std::endl;
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



int main(int argc, const char *argv[]) {


    auto ret_pair = usage(argc, argv);
    auto steps = ret_pair.first;
    auto threads = ret_pair.second;

    my_float pi;
    
    std::vector<my_float> splitted_results(threads, 0.0); // Vector where results are stored.
    std::vector<std::thread> thread_pool; // Vector to hold thread objects

    auto chunks = split_evenly(steps, threads); // Divide de work

    auto start = std::chrono::steady_clock::now();
    
    // Create threads and assign each a chunk of work
    for (size_t i = 0; i < threads; ++i) {
        auto [start_step, stop_step] = get_chunk_begin_end(chunks, i);
        thread_pool.emplace_back(pi_taylor_chunk, std::ref(splitted_results), i, start_step, stop_step);
    }
    
    // Wait for all threads to finish
    for (auto &t : thread_pool) {
        t.join();
    }
    
    // Acumulating results
    pi = 4 * std::accumulate(splitted_results.begin(), splitted_results.end(), my_float(0.0));

	// Stop timing the computation
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;


    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;

    std::cout << "Execution time: " << elapsed_seconds.count() << " seconds" << std::endl;

}
