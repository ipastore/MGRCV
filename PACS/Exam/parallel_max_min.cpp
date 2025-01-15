// Given an std::vector<int> array, could you please write a parallel algorithm that finds the minimum
//and maximum values of the array.

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

int main() {
    std::vector<int> array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int min = array[0];
    int max = array[0];
    #pragma omp parallel for
    for (int i = 0; i < array.size(); i++) {
        #pragma omp critical
        {
            if (array[i] < min) {
                min = array[i];
            }
            if (array[i] > max) {
                max = array[i];
            }
        }
    }
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    return 0;
}

// // Given an std::vector<int> array, could you please write a parallel algorithm that finds the minimum
//and maximum values of the array. without omp in a function


#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

int main() {
    std::vector<int> array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int min = array[0];
    int max = array[0];
    std::mutex mtx;
    std::vector<std::thread> threads;
    for (int i = 0; i < array.size(); i++) {
        threads.push_back(std::thread([&](){
            if (array[i] < min) {
                mtx.lock();
                min = array[i];
                mtx.unlock();
            }
            if (array[i] > max) {
                mtx.lock();
                max = array[i];
                mtx.unlock();
            }
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    return 0;
}

// without omp in a function
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

void find_min_max(std::vector<int> array, int& min, int& max) {
    min = array[0];
    max = array[0];
    std::mutex mtx;
    std::vector<std::thread> threads;
    for (int i = 0; i < array.size(); i++) {
        threads.push_back(std::thread([&](){
            if (array[i] < min) {
                mtx.lock();
                min = array[i];
                mtx.unlock();
            }
            if (array[i] > max) {
                mtx.lock();
                max = array[i];
                mtx.unlock();
            }
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
}


