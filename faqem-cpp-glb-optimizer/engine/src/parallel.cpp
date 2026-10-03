#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace faqem {

namespace {

thread_local bool t_in_job = false;

class Pool {
public:
    explicit Pool(int threads) : threads_(threads) {
        for (int t = 1; t < threads_; t++) workers_.emplace_back([this, t] { work(t); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> l(m_);
            stop_ = true;
            gen_.fetch_add(1);
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int threads() const { return threads_; }

    void run(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t, int)>& body) {
        std::lock_guard<std::mutex> one(dispatch_);  // one job at a time
        job_ = &body;
        n_ = n;
        grain_ = grain;
        next_.store(0);
        pending_.store(threads_ - 1);
        {
            std::lock_guard<std::mutex> l(m_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        chunks(0);
        // wait for the workers (spin: jobs are short)
        while (pending_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
        job_ = nullptr;
    }

private:
    void chunks(int tid) {
        t_in_job = true;
        for (;;) {
            int64_t s = next_.fetch_add(grain_);
            if (s >= n_) break;
            (*job_)(s, std::min(n_, s + grain_), tid);
        }
        t_in_job = false;
    }
    void work(int tid) {
        uint64_t seen = 0;
        for (;;) {
            // spin briefly (collapse checks arrive in quick succession), then sleep
            uint64_t g = gen_.load(std::memory_order_acquire);
            for (int i = 0; g == seen && i < 20000; i++) {
                std::this_thread::yield();
                g = gen_.load(std::memory_order_acquire);
            }
            if (g == seen) {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return gen_.load() != seen; });
                g = gen_.load();
            }
            seen = g;
            if (stop_) return;
            chunks(tid);
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    int threads_;
    std::vector<std::thread> workers_;
    std::mutex m_, dispatch_;
    std::condition_variable cv_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int64_t> next_{0};
    std::atomic<int> pending_{0};
    const std::function<void(int64_t, int64_t, int)>* job_ = nullptr;
    int64_t n_ = 0, grain_ = 1;
    bool stop_ = false;
};

std::mutex g_cfg;
int g_requested = 0;  // 0 = all
std::unique_ptr<Pool> g_pool;
std::atomic<bool> g_collapse_parallel{true};

int resolve(int n) {
    int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    int t = n > 0 ? n : hw;
    if (const char* e = std::getenv("FAQEM_THREADS")) {
        int cap = std::atoi(e);
        if (cap > 0) t = std::min(t, cap);
    }
    return std::max(1, t);
}

Pool* pool() {
    std::lock_guard<std::mutex> l(g_cfg);
    int want = resolve(g_requested);
    if (!g_pool || g_pool->threads() != want) {
        g_pool.reset();
        g_pool.reset(new Pool(want));
    }
    return g_pool.get();
}

}  // namespace

void set_num_threads(int n) {
    std::lock_guard<std::mutex> l(g_cfg);
    g_requested = std::max(0, n);
}

void set_collapse_parallel(bool on) { g_collapse_parallel.store(on); }
bool collapse_parallel() { return g_collapse_parallel.load(); }

int num_threads() {
    std::lock_guard<std::mutex> l(g_cfg);
    return resolve(g_requested);
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t, int)>& body, int64_t grain) {
    if (n <= 0) return;
    grain = std::max<int64_t>(1, grain);
    Pool* p = t_in_job ? nullptr : pool();
    if (!p || p->threads() == 1 || n <= grain) {
        body(0, n, 0);
        return;
    }
    p->run(n, grain, body);
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& body, int64_t grain) {
    parallel_for(n, [&](int64_t s, int64_t e, int) { body(s, e); }, grain);
}

}  // namespace faqem
