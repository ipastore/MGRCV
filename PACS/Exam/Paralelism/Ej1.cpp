#include <vector>
#include <thread>
#include <numeric>   // for std::accumulate
#include <iostream>
#include <stdexcept>

//-----------------------------------------
// 1) Helper function that each thread runs
//-----------------------------------------
template <typename T>
void partialDotProduct(
    const std::vector<T>& a, 
    const std::vector<T>& b,
    T& partialSum,
    size_t startIndex, 
    size_t endIndex
) {
    // Each thread accumulates to a local sum
    T localSum = T(); // zero of type T
    for (size_t i = startIndex; i < endIndex; ++i) {
        localSum += a[i] * b[i];
    }
    // Store result in partialSum (unique to each thread)
    partialSum = localSum;
}

//-----------------------------------------
// 2) Main parallel dot product function
//-----------------------------------------
template <typename T>
T dot_product_parallel(const std::vector<T>& a, const std::vector<T>& b)
{
    // 1. Check that both vectors have the same size
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same length.");
    }
    const size_t length = a.size();

    // 2. Decide how many threads to use
    unsigned int numThreads = std:🧵:hardware_concurrency();

    // 3. Compute chunk size for static partitioning
    const size_t chunkSize = length / numThreads;
    const size_t remainder = length % numThreads;

    // 4. Partial sums (one entry per thread)
    std::vector<T> partialSums(numThreads, T());

    // 5. Launch threads
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    size_t startIndex = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        // This thread’s chunk is [startIndex, endIndex)
        size_t endIndex = startIndex + chunkSize + ((t < remainder) ? 1 : 0);

        // Create a thread to process [startIndex, endIndex) using our helper function
        threads.emplace_back(
            partialDotProduct<T>,
            std::cref(a),
            std::cref(b),
            std::ref(partialSums[t]),
            startIndex,
            endIndex
        );

        // Advance startIndex for the next thread
        startIndex = endIndex;
    }

    // 6. Join threads
    for (auto& th : threads) {
        th.join();
    }

    // 7. Sum up partial results
    T result = std::accumulate(partialSums.begin(), partialSums.end(), T());
    return result;
}

//-----------------------------------------
// Example usage
//-----------------------------------------
int main()
{
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};

    float dot = dot_product_parallel(A, B);
    std::cout << "Dot product = " << dot << "\n"; 
    // Should print 70  (5 + 12 + 21 + 32)

    return 0;
}