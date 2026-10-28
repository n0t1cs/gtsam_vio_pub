#pragma once
#include "init_align.h"

namespace vio {

// Rotates world so gravity = [0,0,-9.81], bakes in metric scale,
// computes T_wb body poses for backend initialization.
AlignedInit& vins_realign(AlignedInit& aligned);

} // namespace vio