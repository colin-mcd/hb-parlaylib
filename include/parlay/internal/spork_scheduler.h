#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

#include "work_stealing_deque.h"
#include "../monoid.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define fwd(x) std::forward<std::remove_reference_t<decltype(x)>>(x)
// This library sets itself up when the scheduler is constructed; programs need
// no startup call (pbbsbench/common/spork_benchmark_startup.h checks this).
#define SPORK_SELF_STARTING 1

namespace parlay {
  template <typename Job>
  class scheduler;
  namespace internal {
    unsigned int init_num_workers();
  }
}

namespace spork {

  // Promotion budget.  A worker may promote a spork (turn a pending fork or
  // the remainder of a loop into a stealable job) only while it holds tokens.
  // ehb2 has no timer: tokens enter the system solely when a steal request is
  // served, and they travel with the jobs a promotion creates, so one request
  // seeds a cascade of promotions across the thieves that pick those jobs up.
  // ehb2 is self-starting: constructing the scheduler sets everything up, so
  // programs need no startup call.
  inline constexpr unsigned int REQUEST_GRANT_TOKENS = 60;   // SPORK_STEAL_REQUEST_TOKENS overrides
  inline constinit thread_local volatile unsigned int promotion_tokens = 0;
  // Budget a worker grants itself after waiting at a join for a stolen job.
  // SPORK_JOIN_WARM_TOKENS overrides; 0 disables.
  inline unsigned int JOIN_WARM_TOKENS = 128;
  inline bool at_outermost_level() noexcept;
  // Set while this thread must not be interrupted by a promotion: while it
  // promotes a slot itself (so a request cannot promote the same slot twice)
  // and while it pops its own deque at a join (the deque's owner-side push
  // and pop are not reentrant, and a promotion pushes).  A request arriving
  // meanwhile is answered as empty.
  inline constinit thread_local volatile bool promotion_blocked = false;
  inline bool promotion_is_blocked() noexcept { return promotion_blocked; }
  inline void set_promotion_blocked(bool b) noexcept { promotion_blocked = b; }

// Publish whether the calling worker is running a job (only busy workers are
// asked for work).  The scheduler registers every worker thread, including
// the thread that created it, with register_worker() before it runs anything.
inline void set_worker_busy(bool busy) noexcept;
inline long long steal_request_now_ns() noexcept;

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
    promotion_tokens = hbt;
    set_worker_busy(true);
    run();
    set_worker_busy(false);
    bool was_done = done.test_and_set(std::memory_order_release);
    assert(!was_done);
  }

  [[nodiscard]] bool finished() volatile const noexcept {
    return done.test(std::memory_order_acquire);
  }

  void wait() const noexcept {
    set_worker_busy(false);
    auto done = [&] () { return finished(); };
    get_current_scheduler().wait_until(done);
    set_worker_busy(true);
  }

  void enqueue(unsigned int with_tokens = 0) {
    hbt = with_tokens;
    if (with_tokens) promotion_tokens = promotion_tokens - with_tokens;
    get_current_scheduler().spawn(this);
  }

  static bool try_dequeue() {
    return get_current_scheduler().get_own_job() != nullptr;
  }

  void fast_clone(bool reclaim_tokens) {
    if (reclaim_tokens) promotion_tokens = promotion_tokens + hbt;
    const_cast<WorkStealingJob*>(this)->run();
  }

  void sync(bool reclaim_tokens) {
    if (try_dequeue()) { // unstolen
      fast_clone(reclaim_tokens);
    } else { // stolen
      if (!finished()) {
        wait();
        // Join warm start: having just waited for a thief at its outermost
        // join (no enclosing sporks, so this is the end of a parallel region
        // rather than a join inside one), this worker knows idle workers are
        // hungry and carries a budget into the next region, which then
        // spreads eagerly without waiting for a steal request.
        if (at_outermost_level() && promotion_tokens < JOIN_WARM_TOKENS) promotion_tokens = JOIN_WARM_TOKENS;
      }
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
  volatile unsigned int hbt; // promotion tokens this job carries to whoever runs it
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
    static void promote_front(unsigned int max_promotions);

    template <typename PromLambda>
    void eager_promote(const PromLambda&& prom);
  };

  inline constinit thread_local SporkSlot spork_deque_front{};
  inline constinit thread_local async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>> spork_deque_back;
  inline bool at_outermost_level() noexcept { return spork_deque_back.load() == &spork_deque_front.next; }

  // ---------------------------------------------------------------------
  // Steal requests.
  //
  // In ehb2 steal requests are the only way work gets promoted: there is no
  // timer.  An idle worker that finds nothing to steal picks a busy worker,
  // posts a request to it (which also locks out other requesters), and
  // interrupts it with SIGALRM sent by tgkill.  The victim's handler promotes
  // its outermost sporks and clears the request, which is the acknowledgement;
  // the requester then steals the promoted work from the victim's own deque.
  // Promotion thus happens within microseconds of a core going idle, and
  // never otherwise.
  //
  // The victim never blocks and never waits; the requester's wait for an
  // answer is bounded by STEAL_REQUEST_MAX_ATTEMPTS steal attempts.
  // ---------------------------------------------------------------------
  struct alignas(64) WorkerRequestState {
    std::atomic<bool> busy{false};       // running a job: only busy workers are asked
    // The request, and the lock on the victim: a requester acquires the
    // victim by setting this from false to true, and the victim clears it
    // once it has promoted (or found nothing), which both acknowledges the
    // request and releases the victim for the next requester.
    std::atomic<bool> pending{false};
    // Requesters skip this victim until this steady-clock time: one request
    // per STEAL_REQUEST_GAP_US per victim, however many thieves are idle.
    std::atomic<long long> not_before_ns{0};
    pid_t tid = 0;
    // The victim's spork deque bounds, so a requester can cheaply see whether
    // there is anything to promote before interrupting it.  Compared, never
    // dereferenced, by other threads.
    const async_signal_safe_pointer<SporkSlot>* deque_front_next = nullptr;
    const async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>>* deque_back = nullptr;
  };
  inline WorkerRequestState* worker_request_states = nullptr;
  inline unsigned int num_worker_request_states = 0;
  // A thread that is not a registered worker points at this never-busy,
  // never-requested state, so the busy stores and the handler need no checks.
  inline constinit WorkerRequestState unregistered_request_state{};
  inline constinit thread_local WorkerRequestState* my_request_state = &unregistered_request_state;

  // Tunables.  Two are runtime (environment) because they are the ones that
  // trade off against each other across workloads; the rest are fixed.
  inline bool steal_requests_enabled = true;                   // SPORK_STEAL_REQUESTS=0 disables
  inline unsigned int STEAL_REQUEST_TOKENS = REQUEST_GRANT_TOKENS;  // SPORK_STEAL_REQUEST_TOKENS
  inline unsigned int STEAL_REQUEST_AFTER = 64;                // SPORK_STEAL_REQUEST_AFTER: failed steal
                                                               // passes (x workers) before the first request
  // Bound on waiting for an answer, in steal attempts rather than time so the
  // wait loop never reads a clock.  An attempt is a handful of loads (the
  // victim's deque, its request flag, one random deque) and takes about ten
  // nanoseconds when they are all empty, so this is a few hundred
  // microseconds: comfortably longer than signal delivery under load, which
  // is what matters, because a requester that gives up before its signal
  // lands has wasted the interrupt.  Measured at 80 cores: 500 and 2000
  // attempts lose promotions and slow delaunay by 18% and 5%; from 8000 on
  // the bound is indistinguishable from the old 100 us clock timeout.
  inline constexpr unsigned int STEAL_REQUEST_MAX_ATTEMPTS = 32000;
  inline constexpr unsigned int STEAL_REQUEST_GAP_US = 20;       // minimum spacing of requests per victim
  inline unsigned int steal_request_after() noexcept { return STEAL_REQUEST_AFTER; }
  inline unsigned int steal_request_max_attempts() noexcept { return STEAL_REQUEST_MAX_ATTEMPTS; }
  inline long long steal_request_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  inline WorkerRequestState* request_state(unsigned int worker) noexcept {
    return (worker_request_states && worker < num_worker_request_states)
             ? &worker_request_states[worker] : nullptr;
  }

  // Requester side ------------------------------------------------------
  // Is `victim` worth interrupting: running a job, has a non-empty spork
  // deque, past its gap, and not already being asked by someone else.
  inline bool steal_request_candidate(unsigned int victim) noexcept {
    WorkerRequestState* st = request_state(victim);
    if (!st || !steal_requests_enabled) return false;
    if (!st->busy.load(std::memory_order_relaxed)) return false;
    if (st->pending.load(std::memory_order_relaxed)) return false;
    if (steal_request_now_ns() < st->not_before_ns.load(std::memory_order_relaxed)) return false;
    if (st->deque_back == nullptr ||
        st->deque_back->load() == st->deque_front_next) return false;  // nothing to promote
    return true;
  }

  // Acquire the victim by posting the request, then interrupt it.
  inline bool steal_request_begin(unsigned int victim) noexcept {
    WorkerRequestState* st = request_state(victim);
    if (!st) return false;
    bool expected = false;
    if (!st->pending.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return false;
    if (!st->busy.load(std::memory_order_acquire) || st->tid == 0) {
      st->pending.store(false, std::memory_order_release);
      return false;
    }
    static const pid_t pid = getpid();
    syscall(SYS_tgkill, pid, st->tid, SIGALRM);
    return true;
  }

  // Has the victim answered (cleared the request)?
  inline bool steal_request_answered(unsigned int victim) noexcept {
    WorkerRequestState* st = request_state(victim);
    return st && !st->pending.load(std::memory_order_acquire);
  }

  // Done with the victim.  If it answered, it has already stamped its gap
  // and cleared the request; if the requester gave up waiting, stamp the gap
  // here and withdraw the request.  A late answer to a withdrawn request can clear
  // a newer requester's flag early; that requester then just checks the deque
  // and moves on, so it is harmless.
  inline void steal_request_end(unsigned int victim) noexcept {
    WorkerRequestState* st = request_state(victim);
    if (!st) return;
    if (st->pending.load(std::memory_order_acquire)) {
      st->not_before_ns.store(steal_request_now_ns() + STEAL_REQUEST_GAP_US * 1000LL, std::memory_order_relaxed);
      st->pending.store(false, std::memory_order_release);
    }
  }

  inline void steal_request_handler(int);

  // Called once by the scheduler constructor: read the tunables, allocate the
  // per-worker request states and install the request signal handler.
  inline void init_steal_requests(unsigned int nw) {
    if (worker_request_states) return;
    if (const char* e = std::getenv("SPORK_STEAL_REQUESTS")) steal_requests_enabled = std::atoi(e) != 0;
    if (const char* e = std::getenv("SPORK_STEAL_REQUEST_TOKENS")) STEAL_REQUEST_TOKENS = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("SPORK_STEAL_REQUEST_AFTER")) STEAL_REQUEST_AFTER = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("SPORK_JOIN_WARM_TOKENS")) JOIN_WARM_TOKENS = std::max(0, std::atoi(e));
    worker_request_states = new WorkerRequestState[nw];
    num_worker_request_states = nw;
    struct sigaction sa = {};
    sa.sa_flags |= SA_RESTART;
    sa.sa_handler = steal_request_handler;
    sigaction(SIGALRM, &sa, nullptr);
  }

  // Called on each worker thread (the scheduler's own thread is worker 0)
  // before it runs any job: start its spork deque empty and publish where a
  // requester can find it.
  inline void register_worker(unsigned int id) noexcept {
    SporkSlot::reset();
    WorkerRequestState* st = request_state(id);
    if (!st) return;
    st->tid = gettid();
    st->deque_front_next = &spork_deque_front.next;
    st->deque_back = &spork_deque_back;
    my_request_state = st;
  }

  inline SporkSlot::SporkSlot(const PromFn* _promfn)
    : promoted(false), promfn(_promfn) {
    prev = spork_deque_back.load();
    *prev = this;
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
    promotion_tokens = promotion_tokens - 1;
    promoted = true;
    (*promfn)();
  }

  // Promote unpromoted slots from the outermost inward while tokens remain,
  // at most max_promotions of them.
  inline void SporkSlot::promote_front(unsigned int max_promotions) {
    if (&spork_deque_front.next == spork_deque_back) return;
    SporkSlot* slot = spork_deque_front.next.load();
    unsigned int n = 0;
    while (promotion_tokens && n < max_promotions) {
      if (!slot->promoted) { slot->promote(); ++n; }
      if (&slot->next == spork_deque_back) break;
      slot = slot->next.load();
    }
  }

  template <typename PromLambda>
  __attribute__((noinline))
  void SporkSlot::eager_promote(const PromLambda&& prom) {
    bool before = promotion_blocked;
    promotion_blocked = true;
    if (promotion_tokens && !promoted) {
      promotion_tokens = promotion_tokens - 1;
      promoted = true;
      fwd(prom)();
    }
    promotion_blocked = before;
  }

  template <typename PromLambda>
  struct PromSpork : PromFn {
    const PromLambda&& prom;
    void operator()() const override {
      fwd(prom)();
    }
    PromSpork(const PromLambda&& _prom) : prom(fwd(_prom)) {}
  };

  // Runs on the victim in signal context.  SIGALRM only ever arrives because
  // an idle worker sent it with tgkill as a steal request, so: promote the
  // outermost unpromoted slots, up to STEAL_REQUEST_TOKENS of them, and clear
  // the request, which is the acknowledgement.  Everything touched is
  // thread-local or a lock-free atomic.
  inline void steal_request_handler(int) {
    int saved_errno = errno;
    if (my_request_state->pending.load(std::memory_order_acquire)) {
      if (!promotion_blocked) {   // otherwise the thread is mid-promotion or mid-pop: answer empty
        const unsigned int saved = promotion_tokens;
        promotion_tokens = STEAL_REQUEST_TOKENS;
        SporkSlot::promote_front(STEAL_REQUEST_TOKENS);
        promotion_tokens = saved;   // no budget left behind for eager promotion
      }
      // Stamp the gap before answering: clearing `pending` releases this
      // worker to the next requester, so the spacing must already be in place.
      my_request_state->not_before_ns.store(steal_request_now_ns() + STEAL_REQUEST_GAP_US * 1000LL,
                                            std::memory_order_relaxed);
      my_request_state->pending.store(false, std::memory_order_release);
    }
    errno = saved_errno;
  }

inline void set_worker_busy(bool busy) noexcept {
  my_request_state->busy.store(busy, std::memory_order_release);
}

template <typename BodyLambda, typename PromLambda>
__attribute__((always_inline))
inline bool with_prom_handler(const BodyLambda&& body, const PromLambda&& prom) {
  static_assert(std::is_invocable_v<BodyLambda&&>);
  static_assert(std::is_invocable_v<PromLambda&&>);

  const PromSpork<PromLambda> promfn(fwd(prom));

  SporkSlot slot(&promfn);
  if (promotion_tokens) [[unlikely]]
    slot.eager_promote(fwd(prom));
  fwd(body)();
  return slot.close();
}

} // namespace spork

namespace parlay {
  using WorkStealingJob = spork::WorkStealingJob;
}

#endif // PARLAY_INTERNAL_SPORK_SCHEDULER_H_
