#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
    std::queue<T> _queue;               
    mutable std::mutex _mutex;          
    std::condition_variable _cond_var;  
  public:
    threadsafe_queue() {}

    threadsafe_queue(const threadsafe_queue& other)
    {
        std::lock_guard<std::mutex> lock(other._mutex);
        _queue = other._queue;                         
    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    void push(T new_value)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(new_value));
        _cond_var.notify_one();
    }

    bool try_pop(T& value)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.empty()) return false;
        value = std::move(_queue.front());
        _queue.pop();
        return true;
    }

    void wait_and_pop(T& value)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _cond_var.wait(lock, [this] { return !_queue.empty(); });
        value = std::move(_queue.front());
        _queue.pop();
    }


    std::shared_ptr<T> wait_and_pop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _cond_var.wait(lock, [this] { return !_queue.empty(); });
        auto result = std::make_shared<T>(std::move(_queue.front()));
        // std::shared_ptr<T> res(std::make_shared<T>(_job_queue.front()));    // res is a shared pointer to a T

        _queue.pop();
        return result;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }
};
