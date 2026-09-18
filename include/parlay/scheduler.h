
#ifndef PARLAY_SCHEDULER_H_
#define PARLAY_SCHEDULER_H_

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <chrono>         // IWYU pragma: keep
#include <memory>
#include <thread>
#include <type_traits>    // IWYU pragma: keep
#include <utility>
#include <vector>

#include "internal/work_stealing_deque.h"         // IWYU pragma: keep
#include "internal/work_stealing_job.h"

// IWYU pragma: no_include <bits/chrono.h>
// IWYU pragma: no_include <bits/this_thread_sleep.h>



// True if the scheduler should scale the number of awake workers
// proportional to the amount of work to be done. This saves CPU
// time if there is not any parallel work available, but may cause
// some startup lag when more parallelism becomes available.
//
// Default: true
// ehb2: idle workers must stay awake, because they are the only thing that
// ever triggers a promotion (there is no timer to create work that would wake
// a sleeper).  An idle worker keeps scanning deques and sending steal
// requests instead of sleeping on the futex.  (Measured: sleeping with a few
// workers kept awake as requesters is time-neutral and saves almost no CPU on
// PBBS, since a worker sleeps only after 10 ms of failed stealing.)
#ifndef PARLAY_ELASTIC_PARALLELISM
#define PARLAY_ELASTIC_PARALLELISM false
#endif


namespace spork {
  // Defined in internal/spork_scheduler.h.
  void init_steal_requests(unsigned int num_workers);
  void register_worker(unsigned int id) noexcept;
  void set_worker_busy(bool busy) noexcept;
  bool promotion_is_blocked() noexcept;
  void set_promotion_blocked(bool b) noexcept;
  // steal-request protocol
  bool steal_request_candidate(unsigned int victim) noexcept;
  bool steal_request_begin(unsigned int victim) noexcept;
  bool steal_request_answered(unsigned int victim) noexcept;
  void steal_request_end(unsigned int victim) noexcept;
  unsigned int steal_request_max_attempts() noexcept;
  unsigned int steal_request_after() noexcept;
  unsigned int steal_request_after_abs() noexcept;
  unsigned int join_leapfrog() noexcept;
  unsigned int join_request_after() noexcept;
  unsigned int join_fallback_after() noexcept;
  template <typename LambdaL, typename LambdaR>
  void par(const LambdaL&& lamL, const LambdaR&& lamR);
  template <typename idx, typename BodyLambda>
  void parfor(idx i, idx j, const BodyLambda&& body);
}

// PARLAY_ELASTIC_STEAL_TIMEOUT sets the number of microseconds
// that a worker will attempt to steal jobs, such that if no
// jobs are successfully stolen, it will go to sleep.
//
// Default: 10000 (10 milliseconds)
#ifndef PARLAY_ELASTIC_STEAL_TIMEOUT
#define PARLAY_ELASTIC_STEAL_TIMEOUT 10000
#endif


#if PARLAY_ELASTIC_PARALLELISM
#include "internal/atomic_wait.h"
#endif

namespace parlay {


template <typename Job>
struct scheduler {

  using worker_id_type = unsigned int;

 private:
  static_assert(std::is_invocable_r_v<void, Job&>);

  struct workerInfo {
    static constexpr worker_id_type UNINITIALIZED = std::numeric_limits<worker_id_type>::max();

    worker_id_type worker_id;
    scheduler* my_scheduler;

    workerInfo() : worker_id(UNINITIALIZED), my_scheduler(nullptr) {}
    workerInfo(std::size_t worker_id_, scheduler* s) : worker_id(worker_id_), my_scheduler(s) {}

    workerInfo& operator=(const workerInfo&) = delete;
    workerInfo(const workerInfo&) = delete;

    workerInfo& operator=(workerInfo&& w) noexcept {
      if (this != &w) {
        worker_id = std::exchange(w.worker_id, UNINITIALIZED);
        my_scheduler = std::exchange(w.my_scheduler, nullptr);
      }
      return *this;
    }

    workerInfo(workerInfo&& w) noexcept { *this = std::move(w); }
  };

  // After YIELD_FACTOR * P unsuccessful steal attempts, a
  // a worker will sleep briefly for SLEEP_FACTOR * P nanoseconds
  // to give other threads a chance to work and save some cycles.
  constexpr static size_t YIELD_FACTOR = 200;
  constexpr static size_t SLEEP_FACTOR = 200;

  // The length of time that a worker must fail to steal anything
  // before it goes to sleep to save CPU time.
  constexpr static std::chrono::microseconds STEAL_TIMEOUT{PARLAY_ELASTIC_STEAL_TIMEOUT};

  static inline thread_local workerInfo worker_info{};

 public:

  const worker_id_type num_threads;

  // If the current thread is a worker of an existing scheduler, or the thread that spawned
  // a scheduler, return the most recent such scheduler.  Otherwise, returns null.
  static scheduler* get_current_scheduler() {
    return worker_info.my_scheduler;
  }

  explicit scheduler(size_t num_workers)
      : num_threads(num_workers),
        num_deques(num_threads),
        num_awake_workers(num_threads),
        parent_worker_info(std::exchange(worker_info, workerInfo{0, this})),
        deques(num_deques),
        attempts(num_deques),
        waiting_on(num_deques),
        spawned_threads(),
        finished_flag(false) {

    // Steal-request state for every worker, then register this thread as
    // worker 0.  It is busy from now on: it runs the program's top-level
    // code, which is where the first work to promote lives.
    spork::init_steal_requests(num_threads);
    spork::register_worker(0);
    spork::set_worker_busy(true);

    // Spawn num_threads many threads on startup
    for (worker_id_type i = 1; i < num_threads; ++i) {
      spawned_threads.emplace_back([&, i]() {
        worker_info = {i, this};
        spork::register_worker(i);
        worker();
      });
    }
  }

  ~scheduler() {
    spork::set_worker_busy(false);
    shutdown();
    worker_info = std::move(parent_worker_info);
  }

  // Push onto local stack.
  void spawn(Job* job) {
    int id = worker_id();
    [[maybe_unused]] bool first = deques[id].push_bottom(job);
#if PARLAY_ELASTIC_PARALLELISM
    if (first) wake_up_a_worker();
#endif
  }

  // Wait until the given condition is true.
  //
  // If conservative, this thread will simply busy wait. Otherwise,
  // it will look for work to steal and keep itself occupied. This
  // can deadlock if the stolen work wants a lock held by the code
  // that is waiting, so avoid that.
  template <typename F>
  void wait_until(F&& done, bool conservative = false) {
    // Conservative avoids deadlock if scheduler is used in conjunction
    // with user locks enclosing a wait.
    if (conservative) {
      while (!done())
        std::this_thread::yield();
    }
    // If not conservative, schedule within the wait.
    // Can deadlock if a stolen job uses same lock as encloses the wait.
    else {
      do_work_until(std::forward<F>(done));
    }
  }

  // Pop from local stack.
  Job* get_own_job() {
    auto id = worker_id();
    // A steal request served by signal during this pop would push onto the
    // same deque; the owner-side push and pop are not reentrant, so hold
    // promotions off for the duration (the request is answered as empty).
    const bool was_blocked = spork::promotion_is_blocked();
    spork::set_promotion_blocked(true);
    Job* job = deques[id].pop_bottom();
    spork::set_promotion_blocked(was_blocked);
    return job;
  }

  worker_id_type num_workers() { return num_threads; }
  worker_id_type worker_id() { return worker_info.worker_id; }

  bool finished() const noexcept {
    return finished_flag.load(std::memory_order_acquire);
  }

 private:
  // Align to avoid false sharing.
  struct alignas(128) attempt {
    size_t val;
  };

  int num_deques;
  std::atomic<size_t> num_awake_workers;
  workerInfo parent_worker_info;
  std::vector<internal::Deque<Job>> deques;
  std::vector<attempt> attempts;
  // The job each worker is currently waiting for at a leapfrogging join, or
  // null.  A joiner follows these to find the worker holding its descendants.
  struct alignas(128) waiting_slot { std::atomic<const Job*> job{nullptr}; };
  std::vector<waiting_slot> waiting_on;
  std::vector<std::thread> spawned_threads;
  std::atomic<int> finished_flag;

  std::atomic<size_t> wake_up_counter{0};
  std::atomic<size_t> num_finished_workers{0};

  // Start an individual worker task, stealing work if no local
  // work is available. May go to sleep if no work is available
  // for a long time, until woken up again when notified that
  // new work is available.
  void worker() {
#if PARLAY_ELASTIC_PARALLELISM
    wait_for_work();
#endif
    while (!finished()) {
      Job* job = get_job([&]() { return finished(); }, PARLAY_ELASTIC_PARALLELISM);
      if (job)(*job)();
#if PARLAY_ELASTIC_PARALLELISM
      else if (!finished()) {
        // If no job was stolen, the worker should go to
        // sleep and wait until more work is available
        wait_for_work();
      }
#endif
    }
    assert(finished());
    num_finished_workers.fetch_add(1);
  }

  // Runs tasks until done(), stealing work if necessary.
  //
  // Does not sleep or time out since this can be called
  // by the main thread and by join points, for which sleeping
  // would cause deadlock, and timing out could cause a join
  // point to resume execution before the job it was waiting
  // on has completed.
  template <typename F>
  void do_work_until(F&& done) {
    while (true) {
      Job* job = get_job(done, false);  // timeout MUST BE false
      if (!job) return;
      (*job)();
    }
    assert(done());
  }

  // Find a job, first trying local stack, then random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  template <typename F>
  Job* get_job(F&& break_early, bool timeout) {
    if (break_early()) return nullptr;
    Job* job = get_own_job();
    if (job) return job;
    else job = steal_job(std::forward<F>(break_early), timeout);
    return job;
  }
  
  // Find a job with random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  template<typename F>
  Job* steal_job(F&& break_early, bool timeout) {
    size_t id = worker_id();
    const auto start_time = std::chrono::steady_clock::now();
    // Steal requests: after every burst of request_after failed random
    // steals, ask a busy worker to promote.
    const size_t request_after = spork::steal_request_after_abs()
                                   ? spork::steal_request_after_abs()
                                   : num_deques * spork::steal_request_after();
    size_t since_request = 0;
    do {
      // By coupon collector's problem, this should touch all.
      for (size_t i = 0; i <= YIELD_FACTOR * num_deques; i++) {
        if (break_early()) return nullptr;
        Job* job = try_steal(id);
        if (job) return job;
        if (num_deques > 1 && ++since_request >= request_after) {
          since_request = 0;
          job = request_promotion(id);
          if (job) return job;
        }
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(num_deques * 100));
    } while (!timeout || std::chrono::steady_clock::now() - start_time < STEAL_TIMEOUT);
    return nullptr;
  }

  // Pick a random busy worker, interrupt it so that it promotes, and take the
  // work it promotes.  Returns nullptr if no suitable victim was found, or the
  // victim did not answer within a bounded number of attempts.  While waiting
  // the requester keeps stealing randomly, so a request never costs it work
  // appearing elsewhere.
  Job* request_promotion(size_t id) {
    size_t target = (hash(id) + hash(attempts[id].val)) % num_deques;
    attempts[id].val++;
    if (target == id || !spork::steal_request_candidate(target)) return nullptr;
    return request_promotion_from(id, target, true);
  }

  // Ask `target` specifically.  With steal_elsewhere, keep stealing at random
  // while waiting; a leapfrogging joiner must not, since unrelated work is
  // exactly what it is avoiding.
  Job* request_promotion_from(size_t id, size_t target, bool steal_elsewhere,
                              const Job* ancestor = nullptr) {
    if (!spork::steal_request_begin(target)) return nullptr;
    auto accept = [ancestor](const Job* const* a) { return ancestor == nullptr || Job::descends(a, ancestor); };
    Job* job = nullptr;
    for (unsigned int n = spork::steal_request_max_attempts(); n > 0; --n) {
      if ((job = deques[target].pop_top_if(accept).first)) break;
      if (spork::steal_request_answered(target)) { job = deques[target].pop_top_if(accept).first; break; }
      if (steal_elsewhere && (job = try_steal(id))) break;
    }
    spork::steal_request_end(target);
    return job;
  }

  // One random steal attempt that only takes a descendant of `ancestor`.
  Job* try_steal_descendant(size_t id, const Job* ancestor) {
    size_t target = (hash(id) + hash(attempts[id].val)) % num_deques;
    attempts[id].val++;
    return deques[target].pop_top_if([ancestor](const Job* const* a) { return Job::descends(a, ancestor); }).first;
  }

 public:
  // Wait for `job` (stolen by another worker) to finish, helping only with
  // work descended from it: steal from the deque of the worker that took it,
  // or, if that worker is itself waiting at a join, from the thief of the job
  // it waits for, and so on down the chain.  When the worker at the end of
  // the chain is running with nothing in its deque, ask it to promote.  Any
  // job obtained this way is part of the subtree the join is waiting for.
  // Modes (spork::join_leapfrog()): 1 steal descendants only; 2 also send a
  // targeted request after a burst of empty polls; 3 as 2 plus one random
  // steal as a fallback on every empty poll; 4 as 2 with the fallback only
  // after the burst; 5 as 2 but every job taken is checked to descend from
  // the awaited one (closing the race where the thief has moved on); 6 as 5
  // plus random steal attempts that accept only descendants, so descendant
  // work held by workers the chain cannot see is found too; 9 just spin.
  void wait_leapfrog(const Job* job) {
    const size_t id = worker_id();
    const unsigned mode = spork::join_leapfrog();
    const bool checked = (mode == 5 || mode == 6);
    const Job* ancestor = checked ? job : nullptr;
    auto accept = [ancestor](const Job* const* a) { return ancestor == nullptr || Job::descends(a, ancestor); };
    const size_t request_after = spork::join_request_after();   // empty polls before asking the thief
    const size_t fallback_after = spork::join_fallback_after(); // empty polls before unrelated work (mode 4)
    size_t empty_polls = 0;
    waiting_on[id].job.store(job, std::memory_order_release);
    while (!job->finished()) {
      if (mode == 9) { __builtin_ia32_pause(); continue; }
      size_t target = job->thief;
      Job* got = nullptr;
      for (int hops = 0; hops < 8 && target < static_cast<size_t>(num_deques) && target != id; ++hops) {
        if ((got = deques[target].pop_top_if(accept).first)) break;
        const Job* w = waiting_on[target].job.load(std::memory_order_acquire);
        if (w == nullptr) break;   // running: it may hold promotable descendants
        target = w->thief;         // waiting: its descendants live with that thief
      }
      // One request attempt per burst of empty polls.  The counter is reset
      // whether or not the attempt succeeds: the candidate check reads the
      // victim's spork-deque pointer, a line the victim writes on every
      // fork, and polling it on every iteration from many joiners slowed
      // the victim's own loops by 2-3x (wordCounts at 80 cores).
      if (!got && mode >= 2 && ++empty_polls >= request_after) {
        empty_polls = 0;
        if (target < static_cast<size_t>(num_deques) && target != id &&
            spork::steal_request_candidate(target))
          got = request_promotion_from(id, target, false, ancestor);
      }
      if (!got && mode == 6) got = try_steal_descendant(id, job);
      // Fallback to unrelated work: mode 3 on every empty poll, mode 4 only
      // once a burst of empty polls has passed without a targeted request
      // yielding anything (the request fires on the same threshold).
      if (!got && (mode == 3 || (mode == 4 && empty_polls + 1 >= fallback_after))) got = try_steal(id);
      if (got) { empty_polls = 0; (*got)(); continue; }
      __builtin_ia32_pause();
    }
    waiting_on[id].job.store(nullptr, std::memory_order_release);
  }

 private:

  Job* try_steal(size_t id) {
    // use hashing to get "random" target
    size_t target = (hash(id) + hash(attempts[id].val)) % num_deques;
    attempts[id].val++;
    auto [job, empty] = deques[target].pop_top();
#if PARLAY_ELASTIC_PARALLELISM
    if (!empty) wake_up_a_worker();
#endif
    return job;
  }

#if PARLAY_ELASTIC_PARALLELISM

  // Wakes up at least one sleeping worker (more than one
  // worker may be woken up depending on the implementation).
  void wake_up_a_worker() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_one(&wake_up_counter);
    }
  }
  
  // Wake up all sleeping workers
  void wake_up_all_workers() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_all(&wake_up_counter);
    }
  }
  
  // Wait until notified to wake up
  void wait_for_work() {
    num_awake_workers.fetch_sub(1);
    parlay::atomic_wait(&wake_up_counter, wake_up_counter.load());
    num_awake_workers.fetch_add(1);
  }

#endif

  size_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<size_t>(x);
  }
  
  void shutdown() {
    finished_flag.store(true, std::memory_order_release);
#if PARLAY_ELASTIC_PARALLELISM
    // We must spam wake all workers until they finish in
    // case any of them are just about to fall asleep, since
    // they might therefore miss the flag to finish
    while (num_finished_workers.load() < num_threads - 1) {
      wake_up_all_workers();
      std::this_thread::yield();
    }
#endif
    for (worker_id_type i = 1; i < num_threads; ++i) {
      spawned_threads[i - 1].join();
    }
  }
};

}  // namespace parlay


namespace parlay {

class fork_join_scheduler {
  using Job = WorkStealingJob;
  using scheduler_t = scheduler<Job>;

 public:

  // Fork two thunks and wait until they both finish using Spork's par
  template <typename L, typename R>
  static void pardo(scheduler_t&, L&& left, R&& right, bool = false) {
    spork::par([&]() { std::forward<L>(left)(); },
               [&]() { std::forward<R>(right)(); });
  }

  // Parallel loop from start to end using Spork's parfor (ignoring granularity)
  template <typename F>
  static void parfor(scheduler_t&, size_t start, size_t end, F&& f, size_t = 0, bool = false) {
    if (end <= start) return;
    if (start + 1 == end) {
      f(start);
      return;
    }
    spork::parfor(start, end, [&f](size_t i) { f(i); });
  }

 private:
  template <typename F>
  static void parfor_(scheduler_t& scheduler, size_t start, size_t end, F& f, size_t = 0, bool conservative = false) {
    parfor(scheduler, start, end, f, 0, conservative);
  }
};

}  // namespace parlay

#endif  // PARLAY_SCHEDULER_H_
