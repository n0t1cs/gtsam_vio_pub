#pragma once
#include "init_sfm.h"

namespace vio {

// Refines SfM poses + landmarks via LM bundle adjustment.
// Modifies sfm in place AND returns it.
SfmResult& bundle_adjust(SfmResult& sfm, int max_iters = 20);

} // namespace vio