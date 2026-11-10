#ifndef PARLAY_INTERNAL_SPORK_API_H_
#define PARLAY_INTERNAL_SPORK_API_H_

// What parlay's scheduler (scheduler.h) uses of spork, declared once for it
// and for the spork headers that define it: parallel.h includes scheduler.h
// before them.

#include <atomic>

namespace spork {

// A job's `thief` before any worker has started running it.
inline constexpr unsigned int NO_THIEF = ~0u;

// true => the *scheduler* is done, e.g., at end of program
inline constinit std::atomic<bool> heartbeats_done = false;

struct WorkStealingJob;
inline void run_job(WorkStealingJob* job);   // runs a job on a fiber (spork_handoff.h)

// The heartbeat timer (spork_heartbeat.h).
inline void start_heartbeats() noexcept;
inline void pause_heartbeats() noexcept;
inline void stop_heartbeats() noexcept;

template <typename LambdaL, typename LambdaR>
void par(const LambdaL& lamL, const LambdaR& lamR);   // spork_par.h
template <typename idx, typename BodyLambda>
void parfor(idx i, idx j, const BodyLambda& body);    // spork_parfor.h

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_API_H_
