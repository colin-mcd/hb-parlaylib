
#ifndef PARLAY_INTERNAL_WORK_STEALING_JOB_H_
#define PARLAY_INTERNAL_WORK_STEALING_JOB_H_

#include <cassert>

#include <atomic>
#include <thread>
#include <type_traits>    // IWYU pragma: keep


namespace spork {
  struct WorkStealingJob;
}

namespace parlay {

using WorkStealingJob = spork::WorkStealingJob;

}

#endif  // PARLAY_INTERNAL_WORK_STEALING_JOB_H_
