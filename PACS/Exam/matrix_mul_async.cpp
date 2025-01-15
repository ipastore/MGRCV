size_t rows() const; 
size_t cols() const;
float& operator()(size_t i, size_t j);
float operator()(size_t i, size_t j) const;

// naive 3-loop multiplication
fmatrix matrix_multiply(const fmatrix &a, const fmatrix &b)
{
    fmatrix c(a.rows(), b.cols());  // assume constructor sets size a.rows() x b.cols()
    for(size_t i = 0; i < a.rows(); ++i) {
        for(size_t j = 0; j < b.cols(); ++j) {
            float val = 0.0f;
            for(size_t k = 0; k < a.cols(); ++k) {
                val += a(i, k) * b(k, j);
            }
            c(i, j) = val;
        }
    }
    return c;
}

#include <future>   // for std::async, std::future
#include <vector>
#include <algorithm> // for std::min
#include <thread>   

fmatrix async_col_matrix_multiply(const fmatrix &a, const fmatrix &b)
{
    // if the machine concurrency is 1 => just do it serially
    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency <= 1) {
        return matrix_multiply(a, b); // the original serial version
    }

    // We spawn one std::async per column in b
    size_t rowsA = a.rows();
    size_t colsA = a.cols(); 
    size_t colsB = b.cols(); // must be the same as rowsB in a valid multiplication
    fmatrix c(rowsA, colsB);

    // We'll store a future for each column
    std::vector<std::future<void>> futures;
    futures.reserve(colsB);

    for (size_t j = 0; j < colsB; ++j) {
        // launch an async task that computes column j of c
        auto fut = std::async(std::launch::async, 
            [&, j]() {
                // compute c(*, j) => for each row i in [0..rowsA)
                for (size_t i = 0; i < rowsA; ++i) {
                    float val = 0.0f;
                    for (size_t k = 0; k < colsA; ++k) {
                        val += a(i, k) * b(k, j);
                    }
                    c(i, j) = val;
                }
            }
        );
        futures.push_back(std::move(fut));
    }

    // wait for all
    
    for (auto &f : futures) {
        f.get();
    }

    return c;
}

// Key Points:
// 	•	Each task iterates over i in [0..rowsA) and does the full sum over k from [0..colsA).
// 	•	So we have colsB concurrent tasks (in the sense of std::async).
// 	•	If colsB is large, we might saturate the machine. If colsB is small, the concurrency is limited.


// Scenario:
// 	•	A 1024‐core machine, but a.cols() (the same as b.rows()) is always < 128.
// 	•	In the “column‐based approach,” we spawn at most 128 tasks. That means we cannot exploit more than 128 cores concurrently.

// Hence the maximum speedup from that approach is limited to about 128 (assuming perfect load balance). If colsB = 128, we can run at most 128 tasks in parallel, so we never use more than 128 of the 1024 cores simultaneously.

// A More Fine‐Grained Approach

// To fully exploit 1024 cores, we can break the work into more pieces—e.g., one task per block of (rows × columns). Or we could do a 2D tiling approach, etc.

// Example: We can create tasks for each row or create tasks for sub‐blocks. For instance, if the matrix is large, we might do:
// 	1.	Partition the result matrix into a grid of tile size, say (16 x 16), so each tile is computed by one std::async.
// 	2.	The total number of tasks could then be (rowsA / 16) * (colsB / 16) (plus some remainder blocks).
// 	3.	If that number is >= 1024, we can saturate the machine.

// A simple approach is one task per row (like we did per column). Then we get rowsA tasks. If rowsA is huge, we can approach 1024 concurrency. If both rowsA and colsB are small, we might do a 2D block approach.

// Sketch:

// A version that spawns one task per row (rather than per column)
fmatrix async_row_matrix_multiply(const fmatrix &a, const fmatrix &b)
{
    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency <= 1) {
        return matrix_multiply(a, b);
    }

    size_t rowsA = a.rows();
    size_t colsA = a.cols(); 
    size_t colsB = b.cols();
    fmatrix c(rowsA, colsB);

    std::vector<std::future<void>> futures;
    futures.reserve(rowsA);

    for (size_t i = 0; i < rowsA; ++i) {
        // each task does row i
        auto fut = std::async(std::launch::async,
            [&, i]() {
                for (size_t j = 0; j < colsB; ++j) {
                    float val = 0.0f;
                    for (size_t k = 0; k < colsA; ++k) {
                        val += a(i, k) * b(k, j);
                    }
                    c(i, j) = val;
                }
            }
        );
        futures.push_back(std::move(fut));
    }

    for (auto &f : futures) {
        f.get();
    }
    return c;
}

// (c) std::async vs. Thread‐Pool: Why Could a Thread‐Pool be Faster?

// If every call to std::async spawns a new thread (the standard does not guarantee a pool, though many lib implementations might do some pooling), we incur:
// 	1.	Thread‐creation overhead for each task,
// 	2.	Possibly significant scheduling overhead,
// 	3.	Large spikes in concurrency if many tasks start simultaneously.

// A thread‐pool approach typically:
// 	•	Creates a fixed set of threads once,
// 	•	Reuses those threads for different tasks,
// 	•	Minimizes creation/join overhead,
// 	•	Might keep tasks in a queue, letting them run as soon as a worker is free.

// Hence, if you do a large number of tasks (like 1024 or more), using a thread‐pool can be significantly more efficient than creating and destroying 1024 separate threads with std::async.
// That’s why a thread‐pool implementation (like TBB, std::execution::par policies, or a custom pool) can be faster and more scalable.