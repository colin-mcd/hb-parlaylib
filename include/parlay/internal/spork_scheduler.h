#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

#include "work_stealing_deque.h"
#include "../monoid.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define USE_SIGNAL_SAFE_ATOMIC
#define RECORD_HEARTBEAT_STATS
#define fwd(x) std::forward<std::remove_reference_t<decltype(x)>>(x)

namespace parlay {
  template <typename Job>
  class scheduler;
  namespace internal {
    unsigned int init_num_workers();
  }
}

namespace spork {

  inline constexpr unsigned int TOKENS_PER_HEARTBEAT = 30;
  inline constexpr unsigned int HEARTBEAT_INTERVAL_US = 500;
  inline constexpr unsigned int MAX_HEARTBEAT_TOKENS = TOKENS_PER_HEARTBEAT * 1;
  inline constinit thread_local volatile unsigned int heartbeat_tokens = 0;
  inline constinit thread_local volatile bool disable_heartbeats = false;

inline void start_heartbeats() noexcept;
inline void pause_heartbeats() noexcept;

struct WorkStealingJob {
  using scheduler_t = parlay::scheduler<WorkStealingJob>;
  static scheduler_t& get_current_scheduler() {
    scheduler_t* current_scheduler = scheduler_t::get_current_scheduler();
    if (current_scheduler == nullptr) {
      static thread_local scheduler_t local_scheduler(parlay::internal::init_num_workers());
      return local_scheduler;
    }
    return *current_scheduler;
  }
  static unsigned int worker_id() {
    return get_current_scheduler().worker_id();
  }
  static unsigned int num_workers() {
    return get_current_scheduler().num_workers();
  }

  WorkStealingJob() {}

  void operator()() {
    heartbeat_tokens = hbt;
    start_heartbeats();
    run();
    pause_heartbeats();
    bool was_done = done.test_and_set(std::memory_order_release);
    assert(!was_done);
  }

  [[nodiscard]] bool finished() volatile const noexcept {
    return done.test(std::memory_order_acquire);
  }

  void wait() const noexcept {
    pause_heartbeats();
    auto done = [&] () { return finished(); };
    get_current_scheduler().wait_until(done);
    start_heartbeats();
  }

  void enqueue(unsigned int with_tokens = 0) {
    hbt = with_tokens;
    if (with_tokens) heartbeat_tokens = heartbeat_tokens - with_tokens;
    get_current_scheduler().spawn(this);
  }

  static bool try_dequeue() {
    return get_current_scheduler().get_own_job() != nullptr;
  }

  void fast_clone(bool reclaim_tokens) {
    if (reclaim_tokens) heartbeat_tokens = heartbeat_tokens + hbt;
    const_cast<WorkStealingJob*>(this)->run();
  }

  void sync(bool reclaim_tokens) {
    if (try_dequeue()) { // unstolen
      fast_clone(reclaim_tokens);
    } else { // stolen
      if (!finished()) wait();
    }
  }

  bool sync_is_stolen() {
    if (!try_dequeue()) {
      if (!finished()) wait();
      return true;
    }
    return false;
  }

  virtual void run() = 0;
  volatile std::atomic_flag done;
  volatile unsigned int hbt; // heartbeat tokens
};

  template <typename T>
  class async_signal_safe_pointer {
    static_assert(std::atomic<T*>::is_always_lock_free,
                  "async_signal_safe_pointer relies on being always lock free!");
    private:
    std::atomic<T*> ptr;
    public:

    inline void store(T* p) noexcept {
      std::atomic_signal_fence(std::memory_order_release);
      ptr.store(p, std::memory_order_relaxed);
    }

    inline T* load() const noexcept {
      T* p = ptr.load(std::memory_order_relaxed);
      std::atomic_signal_fence(std::memory_order_acquire);
      return p;
    }

    inline consteval async_signal_safe_pointer() noexcept : ptr(nullptr) {}

    inline async_signal_safe_pointer(T* p) noexcept {
      store(p);
    }

    inline T& operator*() {
      return *load();
    }

    inline const T& operator*() const {
      return *load();
    }

    inline T* operator->() noexcept {
      return load();
    }

    inline const T* operator->() const noexcept {
      return load();
    }

    inline async_signal_safe_pointer<T>& operator=(async_signal_safe_pointer<T>&& other) noexcept {
      store(other.load());
      return *this;
    }

    inline bool operator==(async_signal_safe_pointer<T>&& other) const noexcept {
      return load() == other.load();
    }
  };

  struct PromFn {
    virtual void operator()() const = 0;
  };

  struct SporkSlot {
    volatile bool promoted;
    const PromFn* promfn;
    async_signal_safe_pointer<SporkSlot>* prev;
    async_signal_safe_pointer<SporkSlot> next;

    consteval explicit SporkSlot() :
      promoted(true), promfn(nullptr), prev(nullptr), next(async_signal_safe_pointer<SporkSlot>()) {}

    explicit SporkSlot(const PromFn* _promfn);
    static void reset();
    bool close();
    void promote();
    static void promote_front();

    template <typename PromLambda>
    void eager_promote(const PromLambda&& prom);
  };

  inline constinit thread_local SporkSlot spork_deque_front{};
  inline constinit thread_local async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>> spork_deque_back;

  inline SporkSlot::SporkSlot(const PromFn* _promfn)
    : promoted(false), promfn(_promfn) {
    prev = spork_deque_back.load();
    *prev = this;
    auto x = &next;
    spork_deque_back = &next;
  }

  inline void SporkSlot::reset() {
    spork_deque_back = &spork_deque_front.next;
  }

  inline bool SporkSlot::close() {
    spork_deque_back = prev;
    return promoted;
  }

  inline void SporkSlot::promote() {
    heartbeat_tokens = heartbeat_tokens - 1;
    promoted = true;
    (*promfn)();
  }

  inline void SporkSlot::promote_front() {
    if (&spork_deque_front.next == spork_deque_back) return;
    SporkSlot* slot = spork_deque_front.next.load();
    while (heartbeat_tokens) {
      if (!slot->promoted) slot->promote();
      if (&slot->next == spork_deque_back) break;
      slot = slot->next.load();
    }
  }

  template <typename PromLambda>
  __attribute__((noinline))
  void SporkSlot::eager_promote(const PromLambda&& prom) {
    bool before = disable_heartbeats;
    disable_heartbeats = true;
    if (heartbeat_tokens && !promoted) {
      heartbeat_tokens = heartbeat_tokens - 1;
      promoted = true;
      fwd(prom)();
    }
    disable_heartbeats = before;
  }

  template <typename PromLambda>
  struct PromSpork : PromFn {
    const PromLambda&& prom;
    void operator()() const override {
      fwd(prom)();
    }
    PromSpork(const PromLambda&& _prom) : prom(fwd(_prom)) {}
  };

  inline volatile unsigned int* num_heartbeats = nullptr;
  inline volatile unsigned int* missed_heartbeats = nullptr;

  inline void init_heartbeat_stats() {
#ifdef RECORD_HEARTBEAT_STATS
    static bool initialized = false;
    if (!initialized) {
      initialized = true;
      unsigned int nw = parlay::internal::init_num_workers();
      num_heartbeats = new unsigned int[nw];
      missed_heartbeats = new unsigned int[nw];
      for (unsigned int wi = 0; wi < nw; ++wi) {
        num_heartbeats[wi] = 0;
        missed_heartbeats[wi] = 0;
      }
    }
#endif
  }

  inline void reset_heartbeat_stats() {
#ifdef RECORD_HEARTBEAT_STATS
    unsigned int nw = WorkStealingJob::num_workers();
    for (unsigned int wi = 0; wi < nw; ++wi) {
      num_heartbeats[wi] = 0;
      missed_heartbeats[wi] = 0;
    }
#endif
  }

  inline void heartbeat_handler(int sig) {
    int saved_errno = errno;
    if (!disable_heartbeats) {
#ifdef RECORD_HEARTBEAT_STATS
      volatile unsigned int& hbs = num_heartbeats[spork::WorkStealingJob::worker_id()];
      hbs = hbs + 1;
#endif
      heartbeat_tokens = heartbeat_tokens + TOKENS_PER_HEARTBEAT;
      if (heartbeat_tokens > MAX_HEARTBEAT_TOKENS) {
        heartbeat_tokens = MAX_HEARTBEAT_TOKENS;
      }
      SporkSlot::promote_front();
    } else {
#ifdef RECORD_HEARTBEAT_STATS
      volatile unsigned int& mhbs = missed_heartbeats[spork::WorkStealingJob::worker_id()];
      mhbs = mhbs + 1;
#endif
    }
    errno = saved_errno;
  }

  inline constinit thread_local timer_t heartbeat_timer;
  inline constinit itimerspec heartbeat_its_zero = {};

  consteval itimerspec init_heartbeat_its() {
    itimerspec its = {};
    its.it_value   .tv_nsec = HEARTBEAT_INTERVAL_US * 1000;
    its.it_interval.tv_nsec = HEARTBEAT_INTERVAL_US * 1000;
    return its;
  }
  inline constinit itimerspec heartbeat_its = init_heartbeat_its();

inline void start_heartbeats() noexcept {
  constinit static thread_local bool thread_initialized = false;
  if (!thread_initialized) { // only first time
    thread_initialized = true;
    SporkSlot::reset();

    struct sigaction sa = {};
    sa.sa_flags |= SA_RESTART;
    sa.sa_handler = heartbeat_handler;
    sigaction(SIGALRM, &sa, nullptr);

    struct sigevent sev{};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo  = SIGALRM;
    sev._sigev_un._tid = gettid();

    timer_create(CLOCK_MONOTONIC, &sev, &heartbeat_timer);
  }

  timer_settime(heartbeat_timer, 0, &heartbeat_its, nullptr);
}

inline void pause_heartbeats() noexcept {
  timer_settime(heartbeat_timer, 0, &heartbeat_its_zero, nullptr);
}

template <typename BodyLambda, typename PromLambda>
__attribute__((always_inline))
inline bool with_prom_handler(const BodyLambda&& body, const PromLambda&& prom) {
  static_assert(std::is_invocable_v<BodyLambda&&>);
  static_assert(std::is_invocable_v<PromLambda&&>);

  const PromSpork<PromLambda> promfn(fwd(prom));

  SporkSlot slot(&promfn);
  if (heartbeat_tokens) [[unlikely]]
    slot.eager_promote(fwd(prom));
  fwd(body)();
  return slot.close();
}

} // namespace spork

namespace parlay {
  using WorkStealingJob = spork::WorkStealingJob;
}

#endif // PARLAY_INTERNAL_SPORK_SCHEDULER_H_
