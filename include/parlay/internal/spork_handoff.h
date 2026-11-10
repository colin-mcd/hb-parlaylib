#ifndef PARLAY_INTERNAL_SPORK_HANDOFF_H_
#define PARLAY_INTERNAL_SPORK_HANDOFF_H_

// Stack handoff: every job runs on a fiber of its own, and at the join of a
// stolen job whichever side arrives second continues the parent.  Running a
// job (run_job, fiber_body), and the joins (WorkStealingJob::join, abandon,
// join_stolen).

#include "spork_api.h"
#include "spork_deque.h"
#include "spork_fiber.h"
#include "spork_job.h"
#include "spork_timing.h"
#include "spork_tokens.h"
#include "../portability.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>

namespace spork {

// ---- fibers -------------------------------------------------------------
//
// Every job runs on a stack of its own, so that a join can suspend it.  The
// Fiber object sits at the top of the stack it describes; the stack grows
// down from just below it towards a guard page.

// The C++ runtime's per-thread exception state (the Itanium ABI's
// __cxa_eh_globals, the same in libc++abi and libstdc++ on x86-64): the
// exceptions being handled, innermost first, and how many have been thrown
// and not yet caught.  It belongs to the computation, not the thread, so
// each fiber carries its own across switches: a catch block that joins can
// resume on another thread, and its __cxa_end_catch must find its own
// exception there.  Declared under another name (the asm label binds it),
// so as not to clash with either runtime's cxxabi.h.
struct EhState {
  void* caught;
  unsigned int uncaught;
};
void* spork_cxa_get_globals() noexcept __asm__("__cxa_get_globals");
// noinline: a switch can resume a computation on another thread, so the
// runtime is asked afresh each time (libstdc++ declares __cxa_get_globals
// const, which would let the compiler reuse an earlier answer).
PARLAY_NOINLINE inline EhState eh_state_save() noexcept {
  return *static_cast<EhState*>(spork_cxa_get_globals());
}
PARLAY_NOINLINE inline void eh_state_install(EhState e) noexcept {
  *static_cast<EhState*>(spork_cxa_get_globals()) = e;
}

// The state of a stack's computation that this thread's thread-locals hold
// while it runs here: its spork deque, its balance and mark, whether it has
// beats disabled, and its exception state.  A suspended computation keeps it
// in its Fiber (fiber_save, fiber_install), and run_job keeps that of the
// computation it suspends.
struct ComputationState {
  SporkSlot* back;
  SporkSlot* frontier;
  unsigned int tokens;
  std::uint64_t mark;
  bool disable;
  EhState eh;

  // Take it from this thread, whose running computation it is.  Field by
  // field, in place: built whole and then copied, it would keep five values
  // live across eh_state_save's call (callee-saved pushes in join_stolen).
  void save() noexcept {
    back = spork_deque_back.load();
    frontier = spork_deque_frontier.load();
    tokens = heartbeat_tokens;
    mark = task_mark;
    disable = disable_heartbeats;
    eh = eh_state_save();
  }

  // A computation that starts now on a stack whose sentinel is `front`, with
  // t tokens: an empty deque, and a mark at now (a right side's mark is when
  // it starts).
  void start(SporkSlot* front, unsigned int t) noexcept {
    back = front;
    frontier = front;
    tokens = t;
    mark = NO_CLOCK ? 0 : monotonic_ns();
    disable = false;
    eh = EhState{};
  }

  // Make it the computation running on this thread.
  void install() const noexcept {
    spork_deque_back = back;
    spork_deque_frontier = frontier;
    heartbeat_tokens = tokens;
    task_mark = mark;
    disable_heartbeats = disable;
    eh_state_install(eh);
  }
};

struct Fiber {
  void* map_base;
  void* stack_top;
  WorkStealingJob* job;
  Fiber* next_free;
  ComputationState saved;   // this stack's computation while it is not running
  fcontext_t entry;         // enters fiber_trampoline on this stack
  std::size_t stack_size;   // usable stack, from the guard page up to this Fiber
  SporkSlot deque_front;    // head of this stack's spork deque
};

[[noreturn]] void fiber_body(Fiber* f);

// Free fibers.  Each thread caches up to FIBER_CACHE_MAX of them.  A thread
// whose cache overflows moves FIBER_BATCH to a shared pool, and one whose
// cache is empty takes up to FIBER_BATCH back before it maps a new stack.
// A fiber is freed on whichever thread ends up leaving it, not the one that
// acquired it, so with purely per-thread lists some threads hoard stacks
// while others keep mapping new ones: three rounds of gbbs BFS on the 3D
// grid mapped 52,000 stacks for a peak of 187 live.  The pool's lock is
// taken only to move a batch, and never from the heartbeat handler.
//
// Both functions are noinline so that the thread-local lists are addressed
// afresh on every call: they run after context switches, which can resume a
// computation on a different thread.
inline constexpr unsigned int FIBER_CACHE_MAX = 16;
inline constexpr unsigned int FIBER_BATCH = FIBER_CACHE_MAX / 2;
inline constinit thread_local Fiber* fiber_free_list = nullptr;
inline constinit thread_local unsigned int fiber_free_count = 0;

struct FiberPool {
  std::mutex lock;
  Fiber* head = nullptr;
};
inline FiberPool fiber_pool;

PARLAY_NOINLINE inline Fiber* fiber_acquire() {
  if (fiber_free_list == nullptr) {
    std::lock_guard<std::mutex> g(fiber_pool.lock);
    if (Fiber* h = fiber_pool.head) {
      Fiber* t = h;
      unsigned int n = 1;
      while (n < FIBER_BATCH && t->next_free != nullptr) { t = t->next_free; n++; }
      fiber_pool.head = t->next_free;
      t->next_free = nullptr;
      fiber_free_list = h;
      fiber_free_count = n;
    }
  }
  if (Fiber* f = fiber_free_list) {
    fiber_free_list = f->next_free;
    fiber_free_count--;
    return f;
  }
  void* base = fiber_map_stack();
  auto top = reinterpret_cast<std::uintptr_t>(base) + FIBER_STACK_SIZE - sizeof(Fiber);
  // default-initialised: SporkSlot's default constructor is explicit
  Fiber* f = new (reinterpret_cast<void*>(top & ~std::uintptr_t{63})) Fiber;
  f->map_base = base;
  f->stack_size = reinterpret_cast<std::uintptr_t>(f) - reinterpret_cast<std::uintptr_t>(base) - FIBER_GUARD_SIZE;
  f->stack_top = reinterpret_cast<void*>((top & ~std::uintptr_t{63}) & ~std::uintptr_t{15});
  return f;
}

PARLAY_NOINLINE inline void fiber_release(Fiber* f) noexcept {
  f->next_free = fiber_free_list;
  fiber_free_list = f;
  if (++fiber_free_count > FIBER_CACHE_MAX) {
    Fiber* h = fiber_free_list;
    Fiber* t = h;
    for (unsigned int i = 1; i < FIBER_BATCH; i++) t = t->next_free;
    fiber_free_list = t->next_free;
    fiber_free_count -= FIBER_BATCH;
    std::lock_guard<std::mutex> g(fiber_pool.lock);
    t->next_free = fiber_pool.head;
    fiber_pool.head = h;
  }
}

void fiber_trampoline(transfer_t t);

inline void fiber_prepare(Fiber* f) noexcept {
  f->entry = make_fcontext(f->stack_top, f->stack_size, fiber_trampoline);
}

// Make f's computation the one this thread is running.
inline void fiber_install(Fiber* f) noexcept {
  current_fiber = f;
  f->saved.install();
}

inline void fiber_save(Fiber* f) noexcept {
  f->saved.save();
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
  f->saved.start(&f->deque_front, job->tokens);
  fiber_install(f);
  fiber_switching = false;
  fiber_drain();
  start_heartbeats();

  job->thief.store(WorkStealingJob::worker_id(), std::memory_order_relaxed);
  SPORK_TRY {
    job->run();
  } SPORK_CATCH_ALL {
    // Nothing beneath this frame could catch it.  The parent throws it on
    // once joined, and the steps below publish it to the parent.
    job->keep_exception();
  }

  fiber_switching = true;
  // For the join, if we are second to it.  Written before either way of
  // telling the parent we are done, and read only by a parent that waited.
  job->tokens = heartbeat_tokens;
  job->child_progress = task_elapsed(job->child_end);
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
  Fiber* const caller_fiber = current_fiber;
  ComputationState caller;
  caller.save();

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
  caller.install();
  fiber_switching = false;
}

// A job nobody stole is run here (fast_clone); at the join of a stolen one
// the continuation's tokens and mark come from both sides (join_tokens).
inline bool WorkStealingJob::join() {
#ifdef SPORK_HEARTBEAT_TIMING
  HeartbeatTiming* const timing = heartbeat_timing_mine;
  if (timing) timing_add(timing->joins, 1);
#endif
  if (try_dequeue()) {              // never stolen: run it here
    fast_clone();
    return threw;
  }
#ifdef SPORK_HEARTBEAT_TIMING
  if (timing) timing_add(timing->stolen_joins, 1);
#endif
  join_stolen();
  return threw;
}

// The parent's own side threw before the join, and its exception comes
// first: serially, this job would never have run.  So a job nobody stole
// is dropped, and its tokens go back to the parent as at the join of a job
// that took no time.  A stolen one is running against the parent's frame,
// which must outlive it, so the parent waits for it, and drops whatever it
// threw.
inline void WorkStealingJob::abandon() {
  if (try_dequeue()) {
    join_tokens_running(0, tokens);
    return;
  }
  join_stolen();
  if (threw) [[unlikely]] kept_exception().~exception_ptr();
}

// The join of a job some thief took.
inline void WorkStealingJob::join_stolen() {
  if (!handoff) {                   // parent's stack is pinned: wait for the child
    if (!finished()) {              // the child will be second
      pause_heartbeats();           // before reading our clock and balance: no beat may change them now
      const std::uint64_t my_progress = task_elapsed();
      const unsigned int my_tokens = heartbeat_tokens;
      get_current_scheduler().wait_for_job(this);
      join_tokens(my_progress, my_tokens, child_progress, tokens);
      start_heartbeats();
      return;
    }
    join_tokens_running(child_progress, tokens);
    return;
  }
  if (join_count.load(std::memory_order_acquire) != 1) {
    // The thief has not finished, so we may be first.  Capture this
    // continuation and get off the stack before saying so: the moment the
    // counter reaches one, the thief may jump onto this stack.
    Fiber* me = current_fiber;
    fiber_switching = true;
    fiber_save(me);
    parent_progress = task_elapsed();
    fiber_post_join = this;
    transfer_t t = jump_fcontext(fiber_home_ctx, nullptr);
    // Resumed.  A thief resuming us runs on a thread whose home handle its
    // own fiber already installed; our own idle loop hands us a fresh one.
    if (t.data == RESUME_FROM_HOME) fiber_home_ctx = t.fctx;
    // Resumed: by the thief if it arrived second, or by our own thread's
    // idle loop if the counter showed we were second after all.
    fiber_install(me);   // our balance is back in heartbeat_tokens
    if (t.data == RESUME_FROM_THIEF)   // the thief was second: its reading is the join's time
      join_tokens(parent_progress, heartbeat_tokens, child_progress, tokens, child_end);
    else                               // we were second after all
      join_tokens(child_progress, tokens, parent_progress, heartbeat_tokens);
    fiber_switching = false;
    fiber_drain();
    return;
  }
  // Otherwise the thief had already finished, so we are trivially second
  // and never left this stack.
  join_tokens_running(child_progress, tokens);
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_HANDOFF_H_
