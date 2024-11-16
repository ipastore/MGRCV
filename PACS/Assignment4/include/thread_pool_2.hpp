#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

class thread_pool
{
  std::atomic_bool _done;
  size_t _thread_count;
  threadsafe_queue<std::function<void()>> _work_queue;
  std::vector<std::thread> _threads;
  join_threads _joiner;

  using task_type = void();

  void worker_thread()
  {
    while (!_done)
    {
      std::function<task_type> task;
      if (_work_queue.try_pop(task))
      {
        // if i can pop, then do the task
        task();
      }
      else
      {
        // if i can't pop, then yield ()
        std::this_thread::yield();
      }
    }
  }

  public:
  thread_pool(size_t num_threads = std::thread::hardware_concurrency())
    : _done(false), _thread_count(num_threads), _joiner(_threads)
  {
    std::cout << "thread_pool::thread_pool() num_threads = " << num_threads << std::endl;
    for (size_t i = 0; i < _thread_count; ++i)
      _threads.push_back(std::thread(&thread_pool::worker_thread, this));
  }

  ~thread_pool()
  {
    wait();
    // then wait for the threads to finish their work
    // destroy _joiner object
    _done = true;
  }

  void wait()
  {
      // wait for completion
      // active waiting
      // wait for the completion of all the tasks:
      while (!_work_queue.empty()){ // this waits for the queue to be empty
        std::this_thread::yield();
      }
      // _done = true;
  }

  template<typename F>
    void submit(F f)
    {
      _work_queue.push(std::function<task_type>(f));
    }
};
