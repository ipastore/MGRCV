#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <algorithm> // for std::min, etc.
#include <cstdint>   // for size_t

int main()
{
    const size_t N = 1024*8;    // array size
    const size_t m_buckets = 32; // number of histogram buckets
    const size_t n_threads = 8; // number of threads

    // 1) We'll assume "array" is already filled with integer values.
    //    In a real program, you'd do array[i] = something.
    std::vector<int> array(N);
    // Example initialization
    for (size_t i = 0; i < N; ++i) {
        array[i] = rand() % 100;  // or something
    }

    // 2) Our final global histogram, stored in atomic<int>
    std::vector<std::atomic<int>> histogram(m_buckets);
    // Initialize them to 0
    for (size_t b = 0; b < m_buckets; ++b) {
        histogram[b].store(0, std::memory_order_relaxed);
    }

    // 3) We'll launch threads in a typical fork-join style
    //    We'll partition the array into slices.
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    // Compute the chunk size for each thread
    size_t chunk_size = (N + n_threads - 1) / n_threads; // ceiling division

    // Lambda to run on each thread
    auto worker = [&](size_t startIdx, size_t endIdx) {
        // Each thread uses a local histogram (non-atomic)
        std::vector<int> local_hist(m_buckets, 0);

        // Fill the local histogram
        for (size_t i = startIdx; i < endIdx; ++i) {
            int val = array[i];
            // Some logic to map 'val' into [0..m_buckets-1].
            // For simplicity, clamp if val is out of range.
            if (val < 0) val = 0;
            if (val >= (int)m_buckets) val = m_buckets - 1;
            local_hist[val]++;
        }

        // Merge local histogram into the global atomic array
        for (size_t b = 0; b < m_buckets; ++b) {
            if (local_hist[b] != 0) {
                histogram[b].fetch_add(local_hist[b], std::memory_order_relaxed);
            }
        }
    };

    // Spawn n_threads
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end   = std::min(start + chunk_size, N);
        threads.emplace_back(worker, start, end);
    }

    // Join
    for (auto &th : threads) {
        th.join();
    }

    // 4) histogram[] now holds the global result
    // Print or use it
    for (size_t b = 0; b < m_buckets; ++b) {
        std::cout << "Bucket " << b << " => " << histogram[b].load() << std::endl;
    }

    return 0;
}

// Explanation
// 	1.	We keep a global array of std::atomic<int> buckets, but we do not directly increment them in the main loop for each element. That would cause heavy contention if all threads increment the same atomic counters frequently.
// 	2.	Instead, each thread builds its local (non‐atomic) histogram for its slice of the array. These increments are extremely cheap (just normal int++).
// 	3.	Only at the end do we merge each local histogram into the global atomic array (one atomic add per bucket). Thus the total number of atomic ops is at most n_threads × m_buckets, which is usually much smaller than N.

// (b) Maximum Speed‐up

// Under ideal conditions (large N, well‐balanced slices, relatively small number of buckets), the parallel histogram is dominated by each thread’s local pass through its portion of the data. The merge step does m_buckets atomic increments per thread, which is negligible if N >> m_buckets.

// Hence, in a best‐case scenario, the speed‐up approaches the number of threads:


// \text{Maximum Speedup} \;\approx\; n\_threads.


// Why not more than n_threads?
// 	•	In a fork‐join model, the best you can do is scale linearly with the number of workers. If all threads do roughly equal work without interfering, you can get near‐perfect scaling.
// 	•	The merge step is small compared to the entire pass over N, so it does not significantly reduce overall speedup when N is large.

// In reality, overheads (thread creation, synchronization) and any load imbalances (uneven data distributions, etc.) can reduce actual speed‐up below n_threads, but the theoretical maximum is close to linear scaling.