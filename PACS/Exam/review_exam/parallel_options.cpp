// 3
std::mutex m;
std::condition_variable cv;
bool ready = false;  // global or static

void producer() {
    {
        std::lock_guard<std::mutex> lg(m);
        ready = true;
    }
    cv.notify_one();
}

void consumer() {
    std::unique_lock<std::mutex> lk(m);
    // wait until "ready == true"
    while (!ready) {
    cv.wait(lk); // atomically unlock 'm' and wait
    }
    // cv.wait(lk, []{ return ready; });
    // now do something
}
int main() {
    std::thread tp(producer);
    std::thread tc(consumer);
    tp.join();
    tc.join();
}

// 2
int fib(int n) {
    if (n < 2) return n;

    // This is just demonstration, not good for large n
    auto f1 = std::async(std::launch::async, fib, n - 1);
    int f2 = fib(n - 2);
    return f1.get() + f2;
}

// 1
void f() { /* ... */ }

int main() {
    std::thread t(f);
    t.detach();
    t.join(); // WRONG: can't do both!
}