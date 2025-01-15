#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
    
    // Mutex para controlar el acceso a la cola
    mutable std::mutex mutex;
    // Cola de datos
    std::queue<T> dataQ;
    // control de espera
    std::condition_variable control;

  public:
    threadsafe_queue() {}

    // Constructor copia
    threadsafe_queue(const threadsafe_queue& other)
    {
        // Para evitar problemas de concurrencia, se bloquea el mutex
        std::lock_guard<std::mutex> lock(other.mutex);
        dataQ = other.dataQ;
    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    // Añadir un nuevo valor a la cola
    void push(T new_value)
    {
        // Se bloquea el mutex
        std::lock_guard<std::mutex> lock(mutex);
        // Se añade el valor a la cola
        dataQ.push(new_value);
        // Se notifica a los hilos que estén esperando
        control.notify_one();
    }

    // Intenta extraer un valor de la cola
    bool try_pop(T& value)
    {
        // Se bloquea el mutex
        std::lock_guard<std::mutex> lock(mutex);
        if (dataQ.empty())
        {
            return false;
        }
        // Se extrae el valor de la cola
        value = dataQ.front();
        dataQ.pop();
        return true;
    }

    // Espera hasta que haya un valor en la cola y lo extrae
    void wait_and_pop(T& value)
    {
        // Se bloquea el mutex
        std::unique_lock<std::mutex> lock(mutex);
        // Se espera a que haya elementos en la cola
        control.wait(lock, [this]{return !dataQ.empty();});
        // Se extrae el valor de la cola
        value = dataQ.front();
        dataQ.pop();    
    }

    // Intenta extraer un valor de la cola y devuelve un puntero compartido
    std::shared_ptr<T> wait_and_pop()
    {
        // Se bloquea el mutex
        std::unique_lock<std::mutex> lock(mutex);
        // Se espera a que haya elementos en la cola
        control.wait(lock, [this]{return !dataQ.empty();});
        // Se extrae el valor de la cola
        std::shared_ptr<T> resultado(std::make_shared<T>(dataQ.front()));
        dataQ.pop();
        return resultado;
    }

    bool empty() const
    {
        // Se bloquea el mutex
        std::lock_guard<std::mutex> lock(mutex);
        // Se devuelve si la cola está vacía
        return dataQ.empty();
    }
};
