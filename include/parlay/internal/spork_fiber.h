#ifndef PARLAY_INTERNAL_SPORK_FIBER_H_
#define PARLAY_INTERNAL_SPORK_FIBER_H_

// Stack handoff for heartbeat scheduling.
//
// A join in the heartbeat/Cilk design is not a blocking point: whichever of
// the two branches arrives second continues the parent's continuation.  In a
// library (no compiler support for splitting a function at a spawn) the only
// way to give the continuation to a different thread is to give that thread
// the stack the continuation lives on.  This header holds the machine-level
// half of that: context switching, lazily committed stacks, and which fiber
// each thread is running.  The protocol that uses them is in
// spork_handoff.h.
//
// Switching is Boost.Context's fcontext, vendored in fcontext.h.  A jump
// pushes the callee-saved registers onto the stack being left and hands back
// a one-shot handle to it,
// so a handle only comes into existence once its stack has been vacated; the
// protocol relies on that, because a parent must be off its stack before it
// tells the thief the stack is available.  fcontext also preserves the MXCSR
// and x87 control words, which are callee-saved under the System V ABI.

#include "fcontext.h"
#include "spork_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <sys/mman.h>
#include <unistd.h>

namespace spork {

// Why a suspended continuation is being resumed, passed as the jump's data
// word.  A continuation resumed by a thief keeps this thread's home handle,
// which the thief's fiber already installed; one resumed by its own thread's
// idle loop is handed a fresh home handle by that jump.
inline void* const RESUME_FROM_HOME  = reinterpret_cast<void*>(1);
inline void* const RESUME_FROM_THIEF = reinterpret_cast<void*>(2);

// Stacks are reserved, not committed, so an idle fiber costs address space
// rather than memory.  The size matches a default pthread stack, because a
// fiber runs one job, whose stack depth is that of the same subtree in the
// sequential program.
inline constexpr std::size_t FIBER_STACK_SIZE = 8u << 20;
inline constexpr std::size_t FIBER_GUARD_SIZE = 4096;

// Reserve one stack, with a guard page at the low end.
inline void* fiber_map_stack() {
  void* base = mmap(nullptr, FIBER_STACK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
  if (base == MAP_FAILED) {
    std::cerr << "spork: could not reserve a fiber stack\n";
    std::abort();
  }
  mprotect(base, FIBER_GUARD_SIZE, PROT_NONE);
  return base;
}

// ---- which fiber this thread is running ------------------------------------
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

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_FIBER_H_
