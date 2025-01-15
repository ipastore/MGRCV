#include <iostream>

using namespace std;

void min_max(int minLocal, int maxLocal, std::vector<int> a, int start_vec, int end_vec){
    minLocal = a[start_vec];
    maxLocal = a[start_vec];
    for(int i=start_vec + 1; i<end_vec + 1; i++){
        minLocal = a[i] < minLocal ? a[i] : minLocal;
        maxLocal = a[i] > maxLocal ? a[i] : maxLocal;
    }
}

std::pair<int,int> minmax_paralel_threads(std::vector<int> a){
    
    
    size_t num_threads = std:🧵:hardware_concurrency();

    std::vector<int> minLocal(num_threads);
    std::vector<int> maxLocal(num_threads);

    std::vector<std::thread> thread_pool;

    const size_t chunk_size = a.length() / num_threads;
    const size_t remaining_parts = a.length() % num_threads;

    size_t start_vec = 0;

    for(int i = 0; i < num_threads; i++){

        size_t end_vec = start_vec + chunk_size + (i < remaining_parts) ? 1 : 0;

        thread_pool.emplace_back(std::thread(min_max, std::ref(minLocal[i]), std::ref(maxLocal[i]), a, start_vec, end_vec));

        start_vec = end_vec;
    }

    for(size_t i=0; i<num_threads; ++i){
        thread_pool[i].join();
    }

    int minGlobal = minLocal[0];
    int maxGlobal = maxLocal[0];

    for (int i = 1; i< minLocal.size(), i++){
        minGlobal = minLocal[i] < minGlobal ? minLocal[i] : minGlobal;
        maxGlobal = maxLocal[i] > maxGlobal ? maxLocal[i] : minGlobal;
    }

    return {minGLobal, maxGlobal};
}