
#ifndef PARLAY_INTERNAL_WORK_STEALING_JOB_H_
#define PARLAY_INTERNAL_WORK_STEALING_JOB_H_

#include <cassert>
#include <cstdint>

#include <atomic>
#include <thread>
#include <type_traits>    // IWYU pragma: keep


namespace parlay {

// Jobs are thunks -- i.e., functions that take no arguments
// and return nothing. Could be a lambda, e.g. [] () {}.

struct WorkStealingJob {
  WorkStealingJob() : done{false} { }
  virtual ~WorkStealingJob() = default;
  void operator()() {
    assert(done.load(std::memory_order_relaxed) == false);
    execute();
    done.store(true, std::memory_order_release);
  }
  [[nodiscard]] bool finished() const noexcept {
    return done.load(std::memory_order_acquire);
  }
  void wait() const noexcept {
    while (!finished())
      std::this_thread::yield();
  }
  // ---- stack handoff -----------------------------------------------------
  // Set when spawned from a stack that can be suspended, that is from a fiber.
  // A job spawned from a pinned stack keeps the older discipline: the parent
  // waits, and the parent resumes.
  bool handoff;
  // Two arrivals: the parent reaching the join, and the thief finishing the
  // job.  Whichever decrement reads one belongs to the second arriver, who
  // continues the parent's continuation.
  std::atomic<int> join_count;
  // The parent's continuation, published once the parent is off that stack.
  // Typed as void* so this header needs no fcontext dependency.
  void* parent_ctx;

 protected:
  virtual void execute() = 0;
  std::atomic<bool> done;
};

// Holds a type-specific reference to a callable object
template<typename F>
struct JobImpl : WorkStealingJob {
  static_assert(std::is_invocable_v<F&>);
  explicit JobImpl(F& _f) : WorkStealingJob(), f(_f) { }
  void execute() override {
    f();
  }
 private:
  F& f;
};

template<typename F>
JobImpl<F> make_job(F& f) { return JobImpl(f); }

}

#endif  // PARLAY_INTERNAL_WORK_STEALING_JOB_H_
