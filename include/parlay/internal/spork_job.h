#ifndef PARLAY_INTERNAL_SPORK_JOB_H_
#define PARLAY_INTERNAL_SPORK_JOB_H_

// The job a promotion spawns (WorkStealingJob), and how an exception thrown
// on either side of a fork reaches its join.

#include "spork_api.h"
#include "spork_fiber.h"
#include "spork_tokens.h"
#include "work_stealing_deque.h"
#include "../portability.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <new>
#include <utility>

// An exception thrown by either side of a fork reaches its join, and is
// thrown on from there (see par, parfor_ and WorkStealingJob::join).  Built
// without exceptions, the handlers compile to nothing.
#if defined(__cpp_exceptions)
#define SPORK_TRY try
#define SPORK_CATCH_ALL catch (...)
#else
#define SPORK_TRY if (true)
#define SPORK_CATCH_ALL else
#endif

namespace parlay {
template <typename Job>
class scheduler;
namespace internal {
unsigned int init_num_workers();
}
}

namespace spork {

// Throw on an exception kept in raw storage (WorkStealingJob::thrown, or a
// fork's own while it joins its job), destroying the kept copy.  Every
// exception spork carries across a join goes on from here, outside any
// handler: a join can resume a frame on another thread, and the C++
// runtime keeps the exceptions being handled per thread.
//
// It returns if the exception kept is null.  It never is, but the
// compiler cannot know that, and so believes the handlers in par and
// parfor_ may lead back to their normal return.  A landing pad that never
// returns stops LLVM from shrink-wrapping the function (ShrinkWrap gives
// up on it), and a recursive function's base case then pays the whole
// prologue and epilogue: five more instructions per fork in fib.
PARLAY_NOINLINE PARLAY_COLD inline void throw_kept(std::exception_ptr& kept) {
  std::exception_ptr e = std::move(kept);
  kept.~exception_ptr();
  if (e) std::rethrow_exception(e);
}

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

  [[nodiscard]] bool finished() const noexcept {
    return done.test(std::memory_order_acquire);
  }

  // Spawn this job with t tokens of its own (from split_tokens).  Through
  // the deque cached for this thread; the scheduler is looked up (and made,
  // if need be) only on a thread that has none yet.
  void enqueue(unsigned int t) {
    tokens = t;
    threw = false;
    handoff = handoff_possible();
    if (handoff) join_count.store(2, std::memory_order_relaxed);
    thief.store(NO_THIEF, std::memory_order_relaxed);
    if (!scheduler_t::spawn_here(this)) [[unlikely]] get_current_scheduler().spawn(this);
  }

  static bool try_dequeue() {
    if (auto* d = scheduler_t::own_deque()) [[likely]] return d->pop_bottom() != nullptr;
    return get_current_scheduler().get_own_job() != nullptr;
  }

  // Run by the parent, which finds the job not stolen.  It is a join like
  // any other: the parent (left side) has finished, with its elapsed time
  // and balance; the job (right side) starts with a fresh mark and the
  // tokens it was given, as a thief would start it; and once it finishes,
  // join_tokens pools the two.
  void fast_clone() {
    std::uint64_t e1;
    unsigned int t1;
    {
      // With beats disabled, so that no beat can land between reading the
      // parent's clock and balance and replacing them with the job's.  The
      // parent's clock runs up to now, and the job's starts now.
      const HeartbeatsDisabled beats_off;
      const std::uint64_t now = NO_CLOCK ? 0 : monotonic_ns();
      e1 = now - task_mark;
      t1 = heartbeat_tokens;
      if constexpr (!NO_CLOCK) task_mark = now;
      heartbeat_tokens = tokens;
    }
    // If the job throws, its exception is kept in it, as a thief would keep
    // it (fiber_body), and the join happens all the same.
    SPORK_TRY {
      const_cast<WorkStealingJob*>(this)->run();
    } SPORK_CATCH_ALL {
      keep_exception();
    }
    join_tokens_running(e1, t1);   // the job, still running here, is the second side
  }

  // join waits for the job, running it here if nobody stole it, and returns
  // whether it threw, keeping the exception in the job; sync throws it on.
  // abandon is the join for a parent whose own side threw first.  Defined
  // in spork_handoff.h, with the fibers, as is join_stolen, the part of join
  // and abandon for a job some thief took.
  bool join();
  void sync() {
    if (join()) [[unlikely]] throw_kept(kept_exception());
  }
  void abandon();
  void join_stolen();

  // Whoever ran a job whose run() threw keeps the exception in the job
  // (fiber_body, fast_clone).
  PARLAY_NOINLINE PARLAY_COLD void keep_exception() noexcept {
    ::new (static_cast<void*>(thrown)) std::exception_ptr(std::current_exception());
    threw = true;
  }
  std::exception_ptr& kept_exception() noexcept {
    return *std::launder(reinterpret_cast<std::exception_ptr*>(thrown));
  }

  virtual void run() = 0;
  std::atomic_flag done;
  // The worker running this job once one has started it (NO_THIEF until
  // then).  Written by that worker and read by a parent waiting at a pinned
  // join (steal_job_patient), so atomic; relaxed, as it is only a hint.
  std::atomic<unsigned int> thief;
  // The tokens the job starts with, set by enqueue; once it has run, the
  // balance it finished with, for a join at which it arrives second.
  unsigned int tokens;
  // Once stolen, each side's elapsed time since its mark (task_elapsed) when
  // it arrived at the join: the thief's when the job finished, and the
  // parent's when it left its stack at a handoff join.
  std::uint64_t child_progress;
  std::uint64_t parent_progress;
  // When the thief read the time for child_progress: the join's time if the
  // thief arrives second and continues the parent (see join_tokens).
  std::uint64_t child_end;

  // ---- stack handoff -----------------------------------------------------
  // Set when this job is enqueued from a stack that can be suspended (that
  // is, from a fiber).  Jobs enqueued from a pinned stack, which is any
  // thread's own stack, keep the older discipline: the parent waits and the
  // parent resumes.
  bool handoff;
  // Whether run() threw, in which case whoever ran it left the exception in
  // `thrown` for the parent (join reports it, sync throws it on, abandon
  // drops it).  Set by enqueue.
  bool threw;
  // Counts down from two: the parent arriving at the join and the thief
  // finishing the job each decrement it once.  Whichever decrement reads one
  // belongs to the second arriver, who continues the parent's continuation.
  std::atomic<int> join_count;
  // The parent's continuation.  It exists only once the parent has left that
  // stack, because an fcontext handle is produced by the jump that vacates it.
  fcontext_t parent_ctx;
  // A std::exception_ptr once threw is set, in raw storage, so that a job
  // stays trivially destructible.
  alignas(std::exception_ptr) unsigned char thrown[sizeof(std::exception_ptr)];
};

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_JOB_H_
