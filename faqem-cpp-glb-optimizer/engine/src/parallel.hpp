// Persistent thread pool. Used where work items are independent and their results are combined
// in an order-independent way, so the outcome is the same bits for any number of threads.
#pragma once
#include <cstdint>
#include <functional>

namespace faqem {

// 0 = all hardware threads. The environment variable FAQEM_THREADS caps the count.
void set_num_threads(int n);
int num_threads();

// whether the per-collapse local check of the simplifier uses the pool (default true); the
// metrics and the bake always do
void set_collapse_parallel(bool on);
bool collapse_parallel();

// body(begin, end, thread_id) over [0, n) in chunks of `grain`; thread_id < num_threads().
// Runs inline when one thread is configured, the work is small, or it is called from inside a job.
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t, int)>& body, int64_t grain = 256);
// convenience overload without the thread id
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& body, int64_t grain = 256);

}  // namespace faqem
