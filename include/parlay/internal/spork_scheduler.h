#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

// Spork's heartbeat scheduler, by layer:
//   spork_api.h        what parlay's scheduler (scheduler.h) uses of it
//   spork_fiber.h      context switching, stacks, which fiber each thread runs
//   spork_tokens.h     tokens, credit clocks and settle; disabling beats
//   spork_timing.h     what heartbeats cost (-DSPORK_HEARTBEAT_TIMING)
//   spork_deque.h      promotable forks, and promoting them
//   spork_job.h        the job a promotion spawns; exceptions across joins
//   spork_handoff.h    running jobs on fibers, and the joins
//   spork_heartbeat.h  each thread's timer and the signal handler
// The forks themselves, par and parfor, are in spork_par.h and spork_parfor.h.

#include "spork_api.h"
#include "spork_fiber.h"
#include "spork_tokens.h"
#include "spork_timing.h"
#include "spork_deque.h"
#include "spork_job.h"
#include "spork_handoff.h"
#include "spork_heartbeat.h"

namespace parlay {
using WorkStealingJob = spork::WorkStealingJob;
}

#endif  // PARLAY_INTERNAL_SPORK_SCHEDULER_H_
