#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

// Spork's heartbeat scheduler, by layer:
//   ../spork_api.h        what parlay's scheduler (parlay/scheduler.h) uses of it
//   fiber.h               context switching, stacks, which fiber each thread runs
//   tokens.h              tokens, credit clocks and settle; disabling beats
//   timing.h              what heartbeats cost (-DSPORK_HEARTBEAT_TIMING)
//   prom_handler_deque.h  promotable forks, and promoting them
//   job.h                 the job a promotion spawns; exceptions across joins
//   handoff.h             running jobs on fibers, and the joins
//   heartbeat.h           each thread's timer and the signal handler
// The forks themselves, par and parfor, are in par.h and parfor.h.

#include "../spork_api.h"
#include "fiber.h"
#include "tokens.h"
#include "timing.h"
#include "prom_handler_deque.h"
#include "job.h"
#include "handoff.h"
#include "heartbeat.h"

namespace parlay {
using WorkStealingJob = spork::WorkStealingJob;
}

#endif  // PARLAY_INTERNAL_SPORK_SCHEDULER_H_
