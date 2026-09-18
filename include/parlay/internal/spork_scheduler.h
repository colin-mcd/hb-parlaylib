#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

#include "work_stealing_deque.h"
#include "spork_fiber.h"
#include "async_signal_safe_pointer.h"
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
  inline constexpr unsigned int REQUEST_GRANT_TOKENS = 64;   // SPORK_STEAL_REQUEST_TOKENS overrides; swept 8..256 at 80 cores, 64 best
  inline constinit thread_local volatile unsigned int promotion_tokens = 0;
  // Budget a worker grants itself after waiting at a join for a stolen job.
  // SPORK_JOIN_WARM_TOKENS overrides; 0 disables.
  inline unsigned int JOIN_WARM_TOKENS = 128;
  // How the warm budget is chosen (SPORK_JOIN_WARM_MODE): 0 none; 1 the
  // fixed JOIN_WARM_TOKENS; 2 an estimate of the number of idle workers, so
  // that every warm token is charged to an idle processor; 3 and 4 as 1 and
  // 2 but only at a true region end (the worker is running no job), not at
  // the outermost join of a stolen job.  The estimate samples
  // JOIN_IDLE_SAMPLES workers' busy flags (no shared writes: a shared
  // counter cost 20% on delaunay from contention at region boundaries).
  inline unsigned int JOIN_WARM_MODE = 2;
  inline unsigned int JOIN_IDLE_SAMPLES = 16;   // SPORK_JOIN_IDLE_SAMPLES
  inline unsigned int estimate_idle_workers() noexcept;
  struct WorkStealingJob;
  struct Fiber;
  // The fiber this thread is running, or null on a thread's own stack.
  inline constinit thread_local Fiber* current_fiber = nullptr;
  // This thread's home handle while it runs a fiber: one-shot, consumed by
  // whichever context jumps back to the idle loop.
  inline constinit thread_local fcontext_t fiber_home_ctx = nullptr;
  inline constinit thread_local Fiber* fiber_pending_free = nullptr;
  inline constinit thread_local WorkStealingJob* fiber_post_join = nullptr;
  // Set while this thread's view of the computation does not match the stack
  // it is using.  Read by the steal-request handler.
  inline constinit thread_local volatile bool fiber_switching = false;
  // Only a fiber can be handed to another thread; a thread's own stack holds
  // frames that belong to that thread (main(), or the worker loop).
  inline bool handoff_possible() noexcept { return current_fiber != nullptr; }
  // The job the calling worker is currently executing (null outside any job).
  // A job created by a promotion records it as its parent, which lets a
  // joiner test whether a job it finds descends from the child it waits for.
  inline constinit thread_local const WorkStealingJob* current_job = nullptr;
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

  // Jobs are run by fiber_body(), on a stack of their own.  This is kept only
  // because parlay::scheduler requires its Job type to be invocable; nothing
  // calls it.
  void operator()() {
    promotion_tokens = hbt;
    set_worker_busy(true);
    run();
    set_worker_busy(false);
    done.test_and_set(std::memory_order_release);
  }

  [[nodiscard]] bool finished() volatile const noexcept {
    return done.test(std::memory_order_acquire);
  }

  // Only a pinned stack ever waits at a join; a fiber hands its stack over.
  void wait() const noexcept {
    set_worker_busy(false);
    auto done = [&] () { return finished(); };
    get_current_scheduler().wait_until(done);
    set_worker_busy(true);
  }

  void enqueue(unsigned int with_tokens = 0) {
    handoff = handoff_possible();
    if (handoff) join_count.store(2, std::memory_order_relaxed);
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

  // Defined below, once fibers are available.
  void sync(bool reclaim_tokens);

  virtual void run() = 0;
  volatile std::atomic_flag done;
  volatile unsigned int hbt; // promotion tokens this job carries to whoever runs it

  // ---- stack handoff -----------------------------------------------------
  // Set when enqueued from a stack that can be suspended, that is from a
  // fiber.  A job enqueued from a pinned stack keeps the older discipline:
  // the parent waits and the parent resumes.
  bool handoff;
  // Two arrivals: the parent reaching the join and the thief finishing the
  // job.  Whichever decrement reads one belongs to the second arriver, who
  // continues the parent's continuation.
  std::atomic<int> join_count;
  // The parent's continuation.  It exists only once the parent has left that
  // stack, because an fcontext handle is produced by the jump that vacates it.
  fcontext_t parent_ctx;
};

  struct PromFn {
    virtual void operator()() const = 0;
  };

  struct SporkSlot {
    volatile bool promoted;
    const PromFn* promfn;
    async_signal_safe_pointer<SporkSlot>* prev;
    async_signal_safe_pointer<SporkSlot> next;

    constexpr explicit SporkSlot() :
      promoted(true), promfn(nullptr), prev(nullptr), next(async_signal_safe_pointer<SporkSlot>()) {}

    explicit SporkSlot(const PromFn* _promfn);
    static void reset();
    bool close();
    void promote();
    static void promote_front(unsigned int max_promotions);

    template <typename PromLambda>
    void eager_promote(const PromLambda&& prom);
  };

  // Slots live in the frames of the stack that made them, so the list head
  // belongs to that stack too: a fiber owns its own sentinel, and
  // spork_deque_front is the sentinel for a thread's own (pinned) stack.
  inline constinit thread_local SporkSlot spork_deque_front{};
  inline constinit thread_local SporkSlot* spork_deque_front_p = nullptr;
  inline constinit thread_local async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>> spork_deque_back;
  inline bool at_outermost_level() noexcept { return spork_deque_back.load() == &spork_deque_front_p->next; }

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
    bool inflight_slot = false;          // this request holds an in-flight slot (requester-private while pending)
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
  // Request threshold: a constant number of failed steals
  // (SPORK_STEAL_REQUEST_AFTER_ABS, 0 = unused) or, failing that, a multiple
  // of the worker count (SPORK_STEAL_REQUEST_AFTER).  The multiple exists to
  // bound signal pressure; with an in-flight cap that job moves to the cap
  // and the threshold only sets request latency.
  inline unsigned int STEAL_REQUEST_AFTER_ABS = 256;   // grid-swept with the cap at 80 cores: 256 x K=4 best
  // At most this many requests in flight process-wide (0 = unbounded), so
  // signal delivery, which serialises on the kernel's per-process lock, has
  // bounded latency independent of the worker count.  SPORK_STEAL_REQUEST_INFLIGHT.
  inline unsigned int STEAL_REQUEST_INFLIGHT = 4;
  inline constinit std::atomic<unsigned int> inflight_requests{0};
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
  inline unsigned int steal_request_after_abs() noexcept { return STEAL_REQUEST_AFTER_ABS; }
  inline unsigned int steal_request_max_attempts() noexcept { return STEAL_REQUEST_MAX_ATTEMPTS; }
  inline long long steal_request_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  // Number of idle workers: exact when there are at most JOIN_IDLE_SAMPLES
  // workers (every busy flag is read), otherwise estimated from that many
  // distinct, evenly spaced workers starting at a random one.
  inline unsigned int estimate_idle_workers() noexcept {
    const unsigned int n = num_worker_request_states;
    if (n == 0) return 0;
    unsigned int idle = 0;
    if (n <= JOIN_IDLE_SAMPLES) {
      for (unsigned int i = 0; i < n; i++)
        if (!worker_request_states[i].busy.load(std::memory_order_relaxed)) idle++;
      return idle;
    }
    static thread_local unsigned int seed = 0x9E3779B9u * (WorkStealingJob::worker_id() + 1);
    seed = seed * 1664525u + 1013904223u;
    const unsigned int k = JOIN_IDLE_SAMPLES, step = n / k;   // step * k <= n: indices are distinct
    unsigned int i = (seed >> 8) % n;
    for (unsigned int j = 0; j < k; j++, i += step) {
      if (i >= n) i -= n;
      if (!worker_request_states[i].busy.load(std::memory_order_relaxed)) idle++;
    }
    return (unsigned int)(((unsigned long long)idle * n) / k);
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
    if (STEAL_REQUEST_INFLIGHT) {   // claim an in-flight slot, or withdraw
      unsigned int n = inflight_requests.load(std::memory_order_relaxed);
      while (true) {
        if (n >= STEAL_REQUEST_INFLIGHT) { st->pending.store(false, std::memory_order_release); return false; }
        if (inflight_requests.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel)) break;
      }
      st->inflight_slot = true;
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
    if (st->inflight_slot) { st->inflight_slot = false; inflight_requests.fetch_sub(1, std::memory_order_acq_rel); }
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
    if (const char* e = std::getenv("SPORK_STEAL_REQUEST_AFTER_ABS")) STEAL_REQUEST_AFTER_ABS = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("SPORK_STEAL_REQUEST_INFLIGHT")) STEAL_REQUEST_INFLIGHT = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("SPORK_JOIN_WARM_TOKENS")) JOIN_WARM_TOKENS = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("SPORK_JOIN_WARM_MODE")) JOIN_WARM_MODE = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("SPORK_JOIN_IDLE_SAMPLES")) JOIN_IDLE_SAMPLES = std::max(1, std::atoi(e));
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
    spork_deque_front_p = &spork_deque_front;
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
    spork_deque_back = &spork_deque_front_p->next;
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
    if (&spork_deque_front_p->next == spork_deque_back) return;
    SporkSlot* slot = spork_deque_front_p->next.load();
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
    if (!fiber_switching && my_request_state->pending.load(std::memory_order_acquire)) {
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

  // ---- fibers -------------------------------------------------------------
  //
  // Every job runs on a stack of its own, so a join can suspend it and hand it
  // to whichever branch arrives second.  The Fiber object sits at the top of
  // the stack it describes; the stack grows down from below it to a guard page.

  struct Fiber {
    void* map_base;
    void* stack_top;
    WorkStealingJob* job;
    Fiber* next_free;
    // This stack's computation state while it is not running.
    async_signal_safe_pointer<SporkSlot>* saved_back;
    unsigned int saved_tokens;
    fcontext_t entry;        // enters fiber_trampoline on this stack
    std::size_t stack_size;   // context that enters fiber_body for the first time
    SporkSlot deque_front;   // head of this stack's spork deque
  };

  [[noreturn]] void fiber_body(Fiber* f);

  inline constinit thread_local Fiber* fiber_free_list = nullptr;

  inline Fiber* fiber_acquire() {
    if (Fiber* f = fiber_free_list) { fiber_free_list = f->next_free; return f; }
    void* base = fiber_map_stack();
    auto top = (reinterpret_cast<std::uintptr_t>(base) + FIBER_STACK_SIZE - sizeof(Fiber)) & ~std::uintptr_t{63};
    // default-initialised: SporkSlot's default constructor is explicit
    Fiber* f = new (reinterpret_cast<void*>(top)) Fiber;
    f->map_base = base;
    f->stack_size = reinterpret_cast<std::uintptr_t>(f) - reinterpret_cast<std::uintptr_t>(base) - FIBER_GUARD_SIZE;
    f->stack_top = reinterpret_cast<void*>(top & ~std::uintptr_t{15});
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

  // Point this thread at a stack's spork deque, and tell requesters where to
  // look when they ask whether this worker has anything to promote.
  inline void install_deque(SporkSlot* front, async_signal_safe_pointer<SporkSlot>* back) noexcept {
    spork_deque_front_p = front;
    spork_deque_back = back;
    my_request_state->deque_front_next = &front->next;
  }

  // A thread's own stack starts with an empty spork deque of its own.
  inline void spork_thread_init() noexcept {
    if (spork_deque_front_p == nullptr) {
      spork_deque_front_p = &spork_deque_front;
      SporkSlot::reset();
    }
  }

  inline void fiber_install(Fiber* f) noexcept {
    current_fiber = f;
    install_deque(&f->deque_front, f->saved_back);
    promotion_tokens = f->saved_tokens;
  }

  inline void fiber_save(Fiber* f) noexcept {
    f->saved_back = spork_deque_back.load();
    f->saved_tokens = promotion_tokens;
  }

  // Recycle the fiber abandoned by whoever switched to us.  At most one can be
  // waiting, because a fiber is abandoned only immediately before a switch.
  inline void fiber_drain() noexcept {
    if (fiber_pending_free) { fiber_release(fiber_pending_free); fiber_pending_free = nullptr; }
  }

  // A stolen job means other workers are hungry, so a worker continuing past
  // its outermost join carries a budget into the next region.
  inline void warm_start_after_join() noexcept {
    if (JOIN_WARM_MODE == 0 || !at_outermost_level()) return;
    if (JOIN_WARM_MODE > 2 && current_fiber != nullptr) return;   // modes 3 and 4: true region ends only
    const unsigned int warm = (JOIN_WARM_MODE == 2 || JOIN_WARM_MODE == 4) ? estimate_idle_workers()
                            : JOIN_WARM_TOKENS;
    if (promotion_tokens < warm) promotion_tokens = warm;
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
    fiber_install(f);
    promotion_blocked = false;
    fiber_switching = false;
    fiber_drain();
    set_worker_busy(true);

    job->run();

    fiber_switching = true;
    if (job->handoff) {
      if (job->join_count.fetch_sub(1, std::memory_order_acq_rel) != 2) {
        // Second to the join: continue the parent on the stack it left behind.
        fiber_pending_free = f;
        jump_fcontext(job->parent_ctx, RESUME_FROM_THIEF);
        __builtin_unreachable();
      }
      // First to the join: the parent has not arrived and will continue itself.
    } else {
      job->done.test_and_set(std::memory_order_release);
    }
    set_worker_busy(false);
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
    unsigned int caller_tokens = promotion_tokens;
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
    fiber_drain();
    current_fiber = caller_fiber;
    install_deque(caller_front, caller_back);
    promotion_tokens = caller_tokens;
    set_worker_busy(false);
    fiber_switching = false;
  }

  inline void WorkStealingJob::sync(bool reclaim_tokens) {
    if (try_dequeue()) {              // never stolen: run it here, on our budget
      fast_clone(reclaim_tokens);
      return;
    }
    if (!handoff) {                   // parent's stack is pinned: wait for the child
      if (!finished()) wait();
      warm_start_after_join();
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
      // Resumed: by the thief if it arrived second, or by our own idle loop if
      // the counter showed we were second after all.
      fiber_install(me);
      fiber_switching = false;
      fiber_drain();
      set_worker_busy(true);
    }
    // Otherwise the thief had already finished and we never left this stack.
    warm_start_after_join();
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
