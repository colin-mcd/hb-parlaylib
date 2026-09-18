#ifndef PARLAY_INTERNAL_SPORK_FIBER_H_
#define PARLAY_INTERNAL_SPORK_FIBER_H_

// Stack handoff for heartbeat scheduling.
//
// A join in the heartbeat/Cilk design is not a blocking point: whichever of
// the two branches arrives second continues the parent's continuation.  In a
// library (no compiler support for splitting a function at a spawn) the only
// way to give the continuation to a different thread is to give that thread
// the stack the continuation lives on.  This header holds the machine-level
// half of that: context switching and a pool of lazily committed stacks.  The
// protocol that uses them is in spork_scheduler.h.
//
// Switching is Boost.Context's fcontext.  A jump pushes the callee-saved
// registers onto the stack being left and hands back a one-shot handle to it,
// so a handle only comes into existence once its stack has been vacated; the
// protocol relies on that, because a parent must be off its stack before it
// tells the thief the stack is available.  fcontext also preserves the MXCSR
// and x87 control words, which are callee-saved under the System V ABI.

#include <boost/context/detail/fcontext.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <sys/mman.h>
#include <unistd.h>

namespace spork {

using fcontext_t = boost::context::detail::fcontext_t;
using transfer_t = boost::context::detail::transfer_t;
using boost::context::detail::make_fcontext;
using boost::context::detail::jump_fcontext;

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

inline void fiber_unmap_stack(void* base) {
  munmap(base, FIBER_STACK_SIZE);
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_FIBER_H_
