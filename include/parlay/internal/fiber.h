#ifndef PARLAY_INTERNAL_FIBER_H_
#define PARLAY_INTERNAL_FIBER_H_

// Stack handoff at joins.
//
// A join here is not a blocking point: whichever of the two branches arrives
// second continues the parent's continuation, as Cilk does.  A continuation is
// a return address and a live frame on one thread's stack, so the only way to
// give it to another thread is to give that thread the stack.  Every job
// therefore runs on a stack of its own, taken from a pool of lazily committed
// mappings, and a join hands one over.
//
// Switching is Boost.Context's fcontext.  A jump pushes the callee-saved
// registers onto the stack being left and hands back a one-shot handle to it,
// so a handle only exists once its stack has been vacated.  The protocol
// depends on that: a parent must be off its stack before the join counter
// tells the thief that the stack is available.

#include <boost/context/detail/fcontext.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>

#include <sys/mman.h>

#include "work_stealing_job.h"

namespace parlay {

using fcontext_t = boost::context::detail::fcontext_t;
using transfer_t = boost::context::detail::transfer_t;
using boost::context::detail::make_fcontext;
using boost::context::detail::jump_fcontext;

// Why a suspended continuation is being resumed, passed as the jump's data
// word.  One resumed by a thief keeps this thread's home handle, which the
// thief's own fiber installed; one resumed by its own thread's idle loop is
// handed a fresh home handle by that jump.
inline void* const RESUME_FROM_HOME  = reinterpret_cast<void*>(1);
inline void* const RESUME_FROM_THIEF = reinterpret_cast<void*>(2);

inline constexpr std::size_t FIBER_STACK_SIZE = 8u << 20;
inline constexpr std::size_t FIBER_GUARD_SIZE = 4096;

struct Fiber {
  void* map_base;
  void* stack_top;
  std::size_t stack_size;
  WorkStealingJob* job;
  Fiber* next_free;
  fcontext_t entry;
};

// The fiber this thread is running, or null on a thread's own stack.  Only a
// fiber can be handed to another thread: a thread's own stack holds frames
// that belong to that thread (main(), or the worker loop).
inline constinit thread_local Fiber* current_fiber = nullptr;
// This thread's idle loop while it runs a fiber: one-shot, consumed by
// whichever context jumps back to it.
inline constinit thread_local fcontext_t fiber_home_ctx = nullptr;
// A fiber abandoned by whoever last switched to us, waiting to be recycled.
inline constinit thread_local Fiber* fiber_pending_free = nullptr;
// A join whose counter must be settled once we are off the stack that the
// settling makes available to the other arriver.
inline constinit thread_local WorkStealingJob* fiber_post_join = nullptr;
inline constinit thread_local Fiber* fiber_free_list = nullptr;

[[noreturn]] void fiber_body(Fiber* f);

inline Fiber* fiber_acquire() {
  if (Fiber* f = fiber_free_list) { fiber_free_list = f->next_free; return f; }
  void* base = mmap(nullptr, FIBER_STACK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
  if (base == MAP_FAILED) { std::cerr << "parlay: could not reserve a fiber stack\n"; std::abort(); }
  mprotect(base, FIBER_GUARD_SIZE, PROT_NONE);
  auto top = (reinterpret_cast<std::uintptr_t>(base) + FIBER_STACK_SIZE - sizeof(Fiber)) & ~std::uintptr_t{63};
  Fiber* f = new (reinterpret_cast<void*>(top)) Fiber;
  f->map_base = base;
  f->stack_top = reinterpret_cast<void*>(top & ~std::uintptr_t{15});
  f->stack_size = top - reinterpret_cast<std::uintptr_t>(base) - FIBER_GUARD_SIZE;
  return f;
}

inline void fiber_release(Fiber* f) noexcept { f->next_free = fiber_free_list; fiber_free_list = f; }

// Recycle the fiber abandoned by whoever switched to us.  At most one can be
// waiting, because a fiber is abandoned only immediately before a switch.
inline void fiber_drain() noexcept {
  if (fiber_pending_free) { fiber_release(fiber_pending_free); fiber_pending_free = nullptr; }
}

// Boost hands an entering context a handle to whoever jumped in, which is this
// thread's idle loop: where this fiber goes when it is done.
inline void fiber_trampoline(transfer_t t) {
  fiber_home_ctx = t.fctx;
  fiber_body(static_cast<Fiber*>(t.data));
}

[[noreturn]] inline void fiber_body(Fiber* f) {
  WorkStealingJob* job = f->job;
  current_fiber = f;
  fiber_drain();

  (*job)();

  if (job->handoff) {
    if (job->join_count.fetch_sub(1, std::memory_order_acq_rel) != 2) {
      // Second to the join: continue the parent on the stack it left behind.
      fiber_pending_free = f;
      jump_fcontext(job->parent_ctx, RESUME_FROM_THIEF);
      __builtin_unreachable();
    }
    // First to the join: the parent has not arrived and will continue itself.
  }
  fiber_pending_free = f;
  jump_fcontext(fiber_home_ctx, nullptr);
  __builtin_unreachable();
}

// Run one job on a fiber of its own, so a join can suspend it.  Returns once
// some fiber has handed control back to this thread.
inline void run_job(WorkStealingJob* job) {
  Fiber* caller = current_fiber;
  Fiber* f = fiber_acquire();
  f->job = job;
  f->entry = make_fcontext(f->stack_top, f->stack_size, fiber_trampoline);
  transfer_t t = jump_fcontext(f->entry, f);
  // A fiber handed control back.  If a parent left a join for us to settle,
  // settle it here, off that parent's stack: only once the counter says the
  // parent was first is its stack free for the thief to jump onto.
  while (fiber_post_join != nullptr) {
    WorkStealingJob* j = fiber_post_join;
    fiber_post_join = nullptr;
    j->parent_ctx = t.fctx;   // exists only because the parent has vacated
    if (j->join_count.fetch_sub(1, std::memory_order_acq_rel) == 2) break;  // the parent was first
    t = jump_fcontext(j->parent_ctx, RESUME_FROM_HOME);   // the thief beat us; hand it back
  }
  fiber_drain();
  current_fiber = caller;
}

// Wait for a stolen job by giving up this stack rather than blocking on it.
inline void join_stolen(WorkStealingJob* j) {
  if (j->join_count.load(std::memory_order_acquire) == 1) return;  // the thief already finished
  Fiber* me = current_fiber;
  fiber_post_join = j;
  transfer_t t = jump_fcontext(fiber_home_ctx, nullptr);
  // Resumed: by the thief if it arrived second, or by our own idle loop if the
  // counter showed we were second after all.
  if (t.data == RESUME_FROM_HOME) fiber_home_ctx = t.fctx;
  current_fiber = me;
  fiber_drain();
}

}  // namespace parlay

#endif  // PARLAY_INTERNAL_FIBER_H_
