#include <vector>
#include <future>      // std::async
#include <cstddef>     // size_t
#include <algorithm>   // std::transform, etc.

class event {
    std::pair<size_t,size_t> _pos;
    bool _polarity;
    std::chrono::time_point<std::chrono::steady_clock> _timestamp;
public:
    event() : _pos(), _polarity(false), _timestamp() {}
    event(bool polarity) : _pos(), _polarity(polarity), _timestamp() {}
    bool polarity() const { return _polarity; }
};

using stream = std::vector<event>;

/**
 * count_polarity: returns how many events in s have polarity == true,
 * using a recursive divide-and-conquer approach with std::async.
 *
 * This version uses sub-vector copies at each split.
 */
size_t count_polarity(const stream& s, size_t min_threshold = 128)
{
    // Base case: if small enough, do sequential
    if (s.size() <= min_threshold) {
        size_t count = 0;
        for (auto &ev : s) {
            if (ev.polarity()) count++;
        }
        return count;
    }

    // Otherwise, split the vector into two halves
    auto midIt = s.begin() + (s.size() / 2);

    // Create sub-vectors (copies)
    stream leftPart(s.begin(), midIt);
    stream rightPart(midIt, s.end());

    // Launch left side asynchronously
    auto futureLeft = std::async(std::launch::async,
                                 count_polarity,
                                 leftPart,        // pass by value
                                 min_threshold);

    // Process right side in current thread
    size_t rightVal = count_polarity(rightPart, min_threshold);

    // Combine results
    return futureLeft.get() + rightVal;
}