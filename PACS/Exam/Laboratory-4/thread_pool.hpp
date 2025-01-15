#pragma once

#include <atomic>
#include <functional>
#include <vector>
#include <future>
#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

class thread_pool
{
  // Termination flag
  std::atomic<bool> _done;
  // Number of threads
  size_t _thread_count;
  // Queue of tasks using threadsafe_queue
  // Assume no-param no-return task
  threadsafe_queue<std::function<void()>> _task_queue;
  // Threads to execute the tasks
  std::vector<std::thread> _workers;

  // Promise
  std::promise<void> barrier;
  std::future<void> barrier_future;

  // to check if the pool is still running
  join_threads _joiner;

  using task_type = void();

  private:

void worker_thread() {
    // Keep working until the pool is done
    while (!_done) {
        // Get the task
        std::function<task_type> task;
        if (_task_queue.try_pop(task)) {
            task();
        } else {
            std::this_thread::yield();
        }
    }
}


  public:
  // Constructor
  thread_pool(size_t num_threads = std::thread::hardware_concurrency())
    : _done(false), _thread_count(num_threads), _joiner(_workers)
  {
      barrier_future = barrier.get_future();
      for (size_t i = 0; i < _thread_count; ++i)
      {
        //Call the workers
        //_workers.push_back(std::thread(&thread_pool::worker_thread, this));
        _workers.emplace_back(&thread_pool::worker_thread, this);
      }

  }

  // Destructor
  ~thread_pool() {

    // Set the done flag
    barrier_future.wait();
  }

  // ~thread_pool()
  // {
  //   wait();
  // }

  void wait()
  {
      // wait until the queue is empty
      while(!_task_queue.empty())
      {
      // active waiting
        std::this_thread::yield();
      }
      _done = true;

  }

  template<typename F>
    void submit(F f)
    {
      // if the pool is done, do not accept new tasks
      if(_done ){
        return;
      }
      // push the task to the queue
      _task_queue.push(std::function<task_type>(f));
    }

    void finish_work()
    {
      submit([this]{
        finish();
      });
    }

    void finish(){
      barrier.set_value();
      _done = true;
    }
};
