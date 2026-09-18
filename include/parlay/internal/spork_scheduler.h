#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

#include "work_stealing_deque.h"
#include "spork_fiber.h"
#include "../monoid.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
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

  inline constexpr unsigned int TOKENS_PER_HEARTBEAT = 64;
  inline constexpr unsigned int HEARTBEAT_INTERVAL_US = 250;
  // inline constexpr unsigned int MAX_HEARTBEAT_TOKENS = TOKENS_PER_HEARTBEAT * 100000;
  inline constinit thread_local volatile unsigned int heartbeat_tokens = 0;
  // true => this *job* must not be promoted; used during eager promotions to prevent signal-delivered heartbeats from promoting simultaneously
  inline constinit thread_local volatile bool disable_heartbeats = false;
  // true => this *processor* (= thread) is between jobs; not fiber state
  inline constinit thread_local volatile bool heartbeats_idle = true;
  // true => the *scheduler* is done, e.g., at end of program
  inline constinit std::atomic<bool> heartbeats_done = false;
  // At a join, add the tokens a stolen child had left when it finished to
  // the continuation's pool, so tokens are never lost on a thief.
  // SPORK_JOIN_COLLECT=0 disables.
  inline bool JOIN_COLLECT_TOKENS = true;

  // ---- stack handoff, thread state ---------------------------------------
  struct Fiber;
  // The fiber this thread is currently running, or null when it is running on
  // its own stack (the main thread's, or a worker's idle loop).
  inline constinit thread_local Fiber* current_fiber = nullptr;
  // Where a fiber goes when it has nothing more to run on this thread.
  // This thread's home handle while it runs a fiber: one-shot, consumed by
  // whichever context jumps back to the idle loop.
  inline constinit thread_local fcontext_t fiber_home_ctx = nullptr;
  // A fiber abandoned by whoever last switched to us, waiting to be recycled.
  inline constinit thread_local Fiber* fiber_pending_free = nullptr;
  // A join whose counter must be decremented once we are off the stack that
  // the decrement makes available to the other arriver.
  inline constinit thread_local WorkStealingJob* fiber_post_join = nullptr;
  // Set while the thread-local view of the computation does not match the
  // stack in use.  Read by the heartbeat handler.
  inline constinit thread_local volatile bool fiber_switching = false;

  // Only a fiber can be handed to another thread.  A thread's own stack is
  // pinned, because the frames beneath the parallel region belong to that
  // thread: main(), or the scheduler's worker loop.
  inline bool handoff_possible() noexcept { return current_fiber != nullptr; }

inline void start_heartbeats() noexcept;
inline void pause_heartbeats() noexcept;
inline void stop_heartbeats() noexcept;

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
    // This job may run nested inside a join's wait(), on a thread that already
    // holds tokens belonging to the frame that is waiting.  Run on this job's
    // own budget and put the enclosing frame's back on the way out, so a job
    // never spends, nor carries off to its own join, tokens that belong to
    // another computation.  Both boundaries run with heartbeats paused, so no
    // beat can be lost in the window.
    const unsigned int enclosing = heartbeat_tokens;
    heartbeat_tokens = hbt;
    start_heartbeats();
    run();
    pause_heartbeats();
    if (JOIN_COLLECT_TOKENS) {   // hand the unspent budget to the joining parent
      leftover = heartbeat_tokens;
    }                            // otherwise it is dropped here (the A/B case)
    heartbeat_tokens = enclosing;
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
    handoff = handoff_possible();
    if (handoff) join_count.store(2, std::memory_order_relaxed);
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

  // Defined below, once fibers are available.
  void sync(bool reclaim_tokens);

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
  volatile unsigned int leftover = 0; // tokens this job had left when it finished

  // ---- stack handoff -----------------------------------------------------
  // Set when this job is enqueued from a stack that can be suspended (that
  // is, from a fiber).  Jobs enqueued from a pinned stack, which is any
  // thread's own stack, keep the older discipline: the parent waits and the
  // parent resumes.
  bool handoff;
  // Counts down from two: the parent arriving at the join and the thief
  // finishing the job each decrement it once.  Whichever decrement reads one
  // belongs to the second arriver, who continues the parent's continuation.
  std::atomic<int> join_count;
  // The parent's continuation.  It exists only once the parent has left that
  // stack, because an fcontext handle is produced by the jump that vacates it.
  fcontext_t parent_ctx;
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

    inline constexpr async_signal_safe_pointer() noexcept : ptr(nullptr) {}

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
    volatile bool promoted; // TODO: change this to an integer, record the stealing thread? or maybe needs to be separate
    const PromFn* promfn;
    async_signal_safe_pointer<SporkSlot>* prev;
    async_signal_safe_pointer<SporkSlot> next;

    constexpr explicit SporkSlot() :
      promoted(true), promfn(nullptr), prev(nullptr), next(async_signal_safe_pointer<SporkSlot>()) {}

    explicit SporkSlot(const PromFn* _promfn);
    static void reset();
    bool close();
    void promote();
    static void promote_front();

    template <typename PromLambda>
    void eager_promote(const PromLambda&& prom);
  };

  // Slots live in the frames of the stack that created them, so the list
  // head belongs to that stack too: a fiber owns its own sentinel, and
  // spork_deque_front is the sentinel for a thread's own (pinned) stack.
  // spork_deque_front_p names whichever is in use on this thread right now,
  // and both it and the back pointer are saved and restored across a handoff.
  inline constinit thread_local SporkSlot spork_deque_front{};
  inline constinit thread_local SporkSlot* spork_deque_front_p = nullptr;
  inline constinit thread_local async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>> spork_deque_back;

  inline SporkSlot::SporkSlot(const PromFn* _promfn)
    : promoted(false), promfn(_promfn) {
    prev = spork_deque_back.load();
    *prev = this;
    auto x = &next;
    spork_deque_back = &next;
  }

  inline void SporkSlot::reset() {
    spork_deque_back = &spork_deque_front_p->next;
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
    if (&spork_deque_front_p->next == spork_deque_back) return;
    SporkSlot* slot = spork_deque_front_p->next.load();
    while (heartbeat_tokens) {
      if (!slot->promoted) slot->promote();
      if (&slot->next == spork_deque_back) break;
      slot = slot->next.load();
    }
  }

  // ---- fibers -------------------------------------------------------------
  //
  // Every job runs on a stack of its own, so that a join can suspend it.  The
  // Fiber object sits at the top of the stack it describes; the stack grows
  // down from just below it towards a guard page.

  struct Fiber {
    void* map_base;
    void* stack_top;
    WorkStealingJob* job;
    Fiber* next_free;
    // This stack's computation state while it is not running.
    async_signal_safe_pointer<SporkSlot>* saved_back;
    unsigned int saved_tokens;
    bool saved_disable;
    fcontext_t entry;        // enters fiber_trampoline on this stack
    std::size_t stack_size;      // context that enters fiber_body for the first time
    SporkSlot deque_front;      // head of this stack's spork deque
  };

  [[noreturn]] void fiber_body(Fiber* f);

  inline constinit thread_local Fiber* fiber_free_list = nullptr;

  inline Fiber* fiber_acquire() {
    if (Fiber* f = fiber_free_list) { fiber_free_list = f->next_free; return f; }
    void* base = fiber_map_stack();
    auto top = reinterpret_cast<std::uintptr_t>(base) + FIBER_STACK_SIZE - sizeof(Fiber);
    // default-initialised: SporkSlot's default constructor is explicit
    Fiber* f = new (reinterpret_cast<void*>(top & ~std::uintptr_t{63})) Fiber;
    f->map_base = base;
    f->stack_size = reinterpret_cast<std::uintptr_t>(f) - reinterpret_cast<std::uintptr_t>(base) - FIBER_GUARD_SIZE;
    f->stack_top = reinterpret_cast<void*>((top & ~std::uintptr_t{63}) & ~std::uintptr_t{15});
    return f;
  }

  inline void fiber_release(Fiber* f) noexcept {
    f->next_free = fiber_free_list;
    fiber_free_list = f;
  }

  void fiber_trampoline(transfer_t t);

  inline void fiber_prepare(Fiber* f) noexcept {
    f->entry = make_fcontext(f->stack_top, f->stack_size, fiber_trampoline);
  }

  // A thread's own stack gets an empty spork deque of its own.
  inline void spork_thread_init() noexcept {
    if (spork_deque_front_p == nullptr) {
      spork_deque_front_p = &spork_deque_front;
      SporkSlot::reset();
    }
  }

  // Make f's computation the one this thread is running.
  inline void fiber_install(Fiber* f) noexcept {
    current_fiber = f;
    spork_deque_front_p = &f->deque_front;
    spork_deque_back = f->saved_back;
    heartbeat_tokens = f->saved_tokens;
    disable_heartbeats = f->saved_disable;
  }

  inline void fiber_save(Fiber* f) noexcept {
    f->saved_back = spork_deque_back.load();
    f->saved_tokens = heartbeat_tokens;
    f->saved_disable = disable_heartbeats;
  }

  // Recycle the fiber abandoned by whoever switched to us.  At most one can be
  // waiting, because a fiber is abandoned only immediately before a switch.
  inline void fiber_drain() noexcept {
    if (fiber_pending_free) { fiber_release(fiber_pending_free); fiber_pending_free = nullptr; }
  }

  // Boost hands an entering context a handle to whoever jumped in, which is
  // this thread's idle loop: where this fiber goes when it is done.
  inline void fiber_trampoline(transfer_t t) {
    fiber_home_ctx = t.fctx;
    fiber_body(static_cast<Fiber*>(t.data));
  }

  [[noreturn]] inline void fiber_body(Fiber* f) {
    WorkStealingJob* job = f->job;
    f->deque_front.promoted = true;
    f->deque_front.next.store(nullptr);
    f->saved_back = &f->deque_front.next;
    f->saved_tokens = job->hbt;
    f->saved_disable = false;
    fiber_install(f);
    fiber_switching = false;
    fiber_drain();
    start_heartbeats();

    job->run();

    if (JOIN_COLLECT_TOKENS) job->leftover = heartbeat_tokens;
    fiber_switching = true;
    if (job->handoff) {
      if (job->join_count.fetch_sub(1, std::memory_order_acq_rel) != 2) {
        // Second to the join: take over the parent's continuation on the stack
        // the parent left behind.  Heartbeats stay armed for it.
        fiber_pending_free = f;
        jump_fcontext(job->parent_ctx, RESUME_FROM_THIEF);
        __builtin_unreachable();
      }
      // First to the join: the parent has not arrived and will continue itself.
    } else {
      job->done.test_and_set(std::memory_order_release);
    }
    fiber_pending_free = f;
    jump_fcontext(fiber_home_ctx, nullptr);
    __builtin_unreachable();
  }

  // Run one job on a fiber of its own.  Returns once some fiber has handed
  // control back to this thread.
  inline void run_job(WorkStealingJob* job) {
    spork_thread_init();
    // The caller's own computation, a pinned stack waiting at a join or a
    // worker's idle loop, is suspended for the duration.
    SporkSlot* caller_front = spork_deque_front_p;
    auto* caller_back = spork_deque_back.load();
    unsigned int caller_tokens = heartbeat_tokens;
    bool caller_disable = disable_heartbeats;
    Fiber* caller_fiber = current_fiber;

    Fiber* f = fiber_acquire();
    f->job = job;
    fiber_prepare(f);
    fiber_switching = true;
    transfer_t t = jump_fcontext(f->entry, f);
    // A fiber handed control back.  If a parent left a join for us to settle,
    // settle it here, off that parent's stack: only once the counter says the
    // parent was first is its stack free for the thief to jump onto.
    while (fiber_post_join != nullptr) {
      WorkStealingJob* j = fiber_post_join;
      fiber_post_join = nullptr;
      // t.fctx is that parent's continuation, which exists only because the
      // parent has already vacated its stack.  Publish it before the counter
      // makes the stack available to the thief.
      j->parent_ctx = t.fctx;
      if (j->join_count.fetch_sub(1, std::memory_order_acq_rel) == 2) break;  // the parent was first
      t = jump_fcontext(j->parent_ctx, RESUME_FROM_HOME);   // the thief beat us; hand it back
    }
    pause_heartbeats();
    fiber_drain();
    current_fiber = caller_fiber;
    spork_deque_front_p = caller_front;
    spork_deque_back = caller_back;
    heartbeat_tokens = caller_tokens;
    disable_heartbeats = caller_disable;
    fiber_switching = false;
  }

  inline void WorkStealingJob::sync(bool reclaim_tokens) {
    if (try_dequeue()) {              // never stolen: run it here, on our budget
      fast_clone(reclaim_tokens);
      return;
    }
    if (!handoff) {                   // parent's stack is pinned: wait for the child
      if (!finished()) wait();
      if (JOIN_COLLECT_TOKENS) heartbeat_tokens = heartbeat_tokens + leftover;
      return;
    }
    if (join_count.load(std::memory_order_acquire) != 1) {
      // The thief has not finished, so we may be first.  Capture this
      // continuation and get off the stack before saying so: the moment the
      // counter reaches one, the thief may jump onto this stack.
      Fiber* me = current_fiber;
      fiber_switching = true;
      fiber_save(me);
      fiber_post_join = this;
      transfer_t t = jump_fcontext(fiber_home_ctx, nullptr);
      // Resumed.  A thief resuming us runs on a thread whose home handle its
      // own fiber already installed; our own idle loop hands us a fresh one.
      if (t.data == RESUME_FROM_HOME) fiber_home_ctx = t.fctx;
      // Resumed: by the thief if it arrived second, or by our own thread's
      // idle loop if the counter showed we were second after all.
      fiber_install(me);
      fiber_switching = false;
      fiber_drain();
    }
    // Otherwise the thief had already finished, so we are trivially second
    // and never left this stack.
    if (JOIN_COLLECT_TOKENS) heartbeat_tokens = heartbeat_tokens + leftover;
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
    if (const char* e = std::getenv("SPORK_JOIN_COLLECT")) JOIN_COLLECT_TOKENS = std::atoi(e) != 0;
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
    if (heartbeats_done.load(std::memory_order_relaxed)) return;
    int saved_errno = errno;
    // fiber_switching covers the window in which the thread-local view of the
    // computation (deque pointers, tokens) does not match the stack it is
    // about to run on.  A beat lost there is harmless; promoting from the
    // wrong stack is not.
    if (!disable_heartbeats && !heartbeats_idle && !fiber_switching) {
#ifdef RECORD_HEARTBEAT_STATS
      volatile unsigned int& hbs = num_heartbeats[spork::WorkStealingJob::worker_id()];
      hbs = hbs + 1;
#endif
      heartbeat_tokens = heartbeat_tokens + TOKENS_PER_HEARTBEAT;
      // if (heartbeat_tokens > MAX_HEARTBEAT_TOKENS) {
      //   heartbeat_tokens = MAX_HEARTBEAT_TOKENS;
      // }
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
  inline constinit thread_local bool heartbeat_timer_live = false;
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
    spork_thread_init();

    struct sigaction sa = {};
    sa.sa_flags |= SA_RESTART;
    sa.sa_handler = heartbeat_handler;
    sigaction(SIGALRM, &sa, nullptr);

    struct sigevent sev{};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo  = SIGALRM;
    sev._sigev_un._tid = gettid();

    timer_create(CLOCK_MONOTONIC, &sev, &heartbeat_timer);
    heartbeat_timer_live = true;
    timer_settime(heartbeat_timer, 0, &heartbeat_its, nullptr);
  }

  heartbeats_idle = false;
}

// Disarm for good: called when a worker leaves the scheduler, and by the main
// thread when the scheduler shuts down.
inline void stop_heartbeats() noexcept {
  heartbeats_idle = true;
  if (heartbeat_timer_live) {
    timer_settime(heartbeat_timer, 0, &heartbeat_its_zero, nullptr);
    timer_delete(heartbeat_timer);
    heartbeat_timer_live = false;
  }
}

inline void pause_heartbeats() noexcept {
  heartbeats_idle = true;
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
