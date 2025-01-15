#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>  // for std::min, std::max, etc.
#include <cmath>      // for std::floor

// Insertion sort as given:
template<typename T>
void insertion_sort(std::vector<T>& array)
{
    for(size_t i = 1; i < array.size(); ++i) {
        for(size_t j = i; j > 0 && (array[j-1] > array[j]); --j) {
            std::swap(array[j], array[j-1]);
        }
    }
}

// A parallel bucket sort. 
// - "num_buckets" can be e.g. = # of threads, or some factor
// - We do integer-based buckets, but it can also adapt to floating types if needed
template<typename T>
void parallel_bucket_sort(std::vector<T>& arr, size_t num_buckets)
{
    if (arr.empty() || num_buckets == 0) {
        return;  // trivial case
    }

    // 1) Find min & max to compute bucket ranges
    //    Or you can assume min=0 if non-negative. Here, we do a general approach.
    auto min_it = std::min_element(arr.begin(), arr.end());
    auto max_it = std::max_element(arr.begin(), arr.end());
    T min_val = *min_it;
    T max_val = *max_it;
    if (min_val == max_val) {
        // All elements identical => already sorted
        return;
    }

    // Each bucket covers a range = (max_val - min_val+1) / num_buckets 
    // We'll cast to float if T is integral, but in general we want the bucket "width"
    double range = static_cast<double>(max_val - min_val + 1) / num_buckets;

    // 2) Create the buckets
    std::vector<std::vector<T>> buckets(num_buckets);

    // 3) We can do the bucket insertion in parallel or single-threaded.
    //    Here we show single-threaded insertion for simplicity,
    //    or you can do parallel for if the array is large.
    for (const auto& val : arr) {
        size_t bucket_index = static_cast<size_t>(
            std::floor((val - min_val) / range)
        );
        // if val == max_val, floor(...) might give num_buckets, so clamp
        if (bucket_index >= num_buckets) {
            bucket_index = num_buckets - 1;
        }
        buckets[bucket_index].push_back(val);
    }

    // 4) Sort each bucket in parallel
    //    We'll spawn "num_buckets" threads, each sorts a bucket.
    std::vector<std::thread> threads;
    threads.reserve(num_buckets);

    for (size_t i = 0; i < num_buckets; ++i) {
        threads.emplace_back([&, i]() {
            insertion_sort(buckets[i]);  // or std::sort(buckets[i].begin(), buckets[i].end());
        });
    }

    // Join all
    for (auto &th : threads) {
        th.join();
    }

    // 5) Concatenate the buckets back into "arr"
    //    We'll do a simple linear pass in ascending bucket order.
    size_t idx = 0;
    for (size_t i = 0; i < num_buckets; ++i) {
        for (auto & val : buckets[i]) {
            arr[idx++] = val;
        }
    }
}

// Example usage
int main()
{
    std::vector<int> data = {31, 18, 23, 4, 12, 28, 45, 63, 2};
    size_t num_buckets = 3; // e.g. 3 buckets
    parallel_bucket_sort(data, num_buckets);

    for (auto val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

// Comments on Implementation
// 	1.	Bucket Range: We used (max_val - min_val + 1)/num_buckets. If val == max_val, we clamp the bucket index to num_buckets-1.
// 	2.	Parallel Sorting: we spawn num_buckets threads, each sorts one bucket. If the distribution is skewed, some buckets might have more elements than others (leading to load imbalance).
// 	3.	Insertion Sort: good for small buckets. For large buckets, you might prefer std::sort.

// Part (b): Pathological Case for No Speedup

// Yes, there are cases where the parallel version might not be faster. Examples:
// 	1.	Very small input: If the array is very small, the overhead of spawning threads (or distributing work) can dominate any potential parallel speedup.
// 	2.	Skewed data / load imbalance: If the data distribution is heavily skewed so that one bucket ends up with most of the elements, that single thread is forced to do almost all the sorting while other threads finish quickly and remain idle. This can kill parallel performance.
// 	3.	Data is already in a form that leads to a single bucket: E.g., if all elements are in a small numeric range that maps them all into the same bucket, you effectively lose parallelism in sorting that one large bucket.
// 	4.	Huge overhead or memory constraints: If you have high overhead in memory copying or thread creation, it may overshadow the parallel gains.

// In all these scenarios, the overhead or load imbalance can cause the parallel approach to be equal or even slower than a well‐optimized sequential sort.