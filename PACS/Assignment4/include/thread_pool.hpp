#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

class thread_pool
{
  std::atomic<bool> _done; 
  threadsafe_queue<std::function<void()>> _task_queue;
  std::vector<std::thread> _threads;
  join_threads _joiner;

  
  using task_type = std::function<void()>; 


  void worker_thread()
  {
    while (!_done)
    {
      task_type task; 
      if (_task_queue.try_pop(task))
      {
        task();
      }
      else
      {
        std::this_thread::yield();
      }
    }
  }

  public:
  thread_pool(size_t num_threads = std::thread::hardware_concurrency())
    : _done(false), _joiner(_threads)
  {
    std::cout << "thread_pool::thread_pool() num_threads = " << num_threads << std::endl;
    for (size_t i = 0; i < num_threads; ++i)
      _threads.emplace_back(&thread_pool::worker_thread, this); 

  }

  ~thread_pool()
  {
    wait();
    _done = true;
  }

  void wait()
  {
      while (!_task_queue.empty()){ 
        std::this_thread::yield();
      }
  }

  template<typename F>
    void submit(F f)
    {
      _task_queue.push(task_type(f));
    }
};
