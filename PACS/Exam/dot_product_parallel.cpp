
#include <iostream>
#include <vector>


///////////////////////////////////////// SEQUENTIAL ////////////////////////////////////////////
// Sequential dot product
template<typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same size.");
    }
    T dot_p = T();  // 0-initialized for any numeric type
    for (size_t i = 0; i < a.size(); ++i) {
        dot_p += a[i] * b[i];
    }
    return dot_p;
}
///////////////////////////////////////// SEQUENTIAL ////////////////////////////////////////////



///////////////////////////////////////// Parallel1 (best approach) ////////////////////////////////////////////

// Threads with static partitioning
// This is a better approach than the other with a mutex and locking. But it lacks of modularity.
template<typename T>
// devuelve un valor de tipo T, recibe dos vectores de tipo T y un tamaño de threads
T dot_product_parallel(const std::vector<T>& a, const std::vector<T>& b, size_t num_threads) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same size.");
    }
    // n es el tamaño del vector a
    size_t n = a.size();
    // tamaño de bloque
    size_t block_size = n / num_threads;
    // vector de tamaño num_threads de tipo T
    std::vector<T> partial_sums(num_threads, T());
    // vector de threads
    std::vector<std::thread> threads;

    // recorre el número de threads y crea un thread por cada uno
    for (size_t i = 0; i < num_threads; ++i) {
        // empieza en i * tamaño de bloque. Emplace back es un metodo de vector que añade un elemento al final del vector
        // el elemento es un thread que recibe una lambda function.
        threads.emplace_back([i, block_size, &a, &b, &partial_sums] {
            size_t start = i * block_size;
            size_t end = (i == partial_sums.size() - 1) ? a.size() : start + block_size;
            T partial_sum = T();
            for (size_t j = start; j < end; ++j) {
                partial_sum += a[j] * b[j];
            }
            partial_sums[i] = partial_sum;
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // accumulate suma los elementos de un vector
    // The initial value for the accumulation. This calls the default constructor of type T, 
    // which initializes the value to zero for numeric types.
    return std::accumulate(partial_sums.begin(), partial_sums.end(), T());
}   
///////////////////////////////////////// Parallel1 (best approach) ////////////////////////////////////////////

///////////////////////////////////////// Parallel2 with mutex and lock ////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
// Another option with static partitioning and two functions

// function for a partial dot product
template<typename T>
void partial_dot(const std::vector<T>& a,const std::vector<T>& b, 
                size_t start, size_t end,  T& result,
                std::mutex& result_mutex)
{
    T local_sum = T();
    for (size_t i = start; i < end; ++i) {
        local_sum += a[i] * b[i];
    }
    // Protect shared result with mutex
    // result is the shared result, lock is locking the mutex of the result
    std::lock_guard<std::mutex> lock(result_mutex);
    result += local_sum;
}

template<typename T>
T dot_product_static(const std::vector<T>& a, const std::vector<T>& b,size_t num_threads)
{
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same size.");
    }

    size_t length = a.size();
    // e.g. chunk size for each thread
    size_t block_size = length / num_threads;  
    T result = T();
    std::mutex result_mutex;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * block_size;
        // last thread picks up the remainder
        size_t end   = (t == num_threads - 1) ? length : (t+1) * block_size;

        // emplace back threads with the lambda function
        threads.emplace_back(partial_dot<T>, std::cref(a), std::cref(b), start, end, std::ref(result), std::ref(result_mutex));
    }

    // Join all
    for (auto & th : threads) {
        th.join();
    }

    return result;
}
///////////////////////////////////////// Parallel2 with mutex and lock ////////////////////////////////////////////



// Dot product parallel: Thread pool + Thread safe queue

#include <thread_pool.hpp>
#include <threadsafe_queue.hpp>

template<typename T>
T dot_product_thread_pool(const std::vector<T>& a, const std::vector<T>& b, size_t num_threads)
{
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same size.");
    }

    size_t length = a.size();
    size_t block_size = length / num_threads;
    T result = T();

    thread_pool pool(num_threads);

    threadsafe_queue<T> results;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * block_size;
        size_t end   = (t == num_threads - 1) ? length : (t+1) * block_size;

        pool.submit([start, end, &a, &b, &results] {
            T local_sum = T();
            for (size_t i = start; i < end; ++i) {
                local_sum += a[i] * b[i];
            }
            results.push(local_sum);
        });
    }

    pool.wait();

    while (!results.empty()) {
        T local_sum;
        results.wait_and_pop(local_sum);
        result += local_sum;
    }

    return result;
}