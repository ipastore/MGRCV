#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
    std::queue<T> _job_queue;
    mutable std::mutex _m;
    std::condition_variable _cv;

  public:
    threadsafe_queue() {}

    threadsafe_queue(const threadsafe_queue& other)
    {
        std::lock_guard<std::mutex> lk(other._m);
        _job_queue = other._job_queue;
    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    void push(T new_value)
    {
        // add a new value to the queue
        // first lock the queue
        std::lock_guard<std::mutex> lk(_m); // lock guard will unlock the mutex when it goes out of scope
        _job_queue.push(new_value);
    }

    bool try_pop(T& value)
    {
        // check if i can pop
        std::lock_guard<std::mutex> lk(_m);
        // if i can pop, pop it
        if (_job_queue.empty()) // if i can't pop, return false
            return false;
        value = _job_queue.front();
        _job_queue.pop();
        return true;
    }

    void wait_and_pop(T& value)
    {
        // wait until i can pop
        std::unique_lock<std::mutex> lk(_m);
        _cv.wait(lk, [this] { return !_job_queue.empty(); });
        value = _job_queue.front();
        _job_queue.pop();
    }

    std::shared_ptr<T> wait_and_pop()
    {
        std::unique_lock<std::mutex> lk(_m);
        _cv.wait(lk, [this] { return !_job_queue.empty(); });
        std::shared_ptr<T> res(std::make_shared<T>(_job_queue.front()));    // res is a shared pointer to a T
        _job_queue.pop();
        return res;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lk(_m);
        return _job_queue.empty();
    }
};
