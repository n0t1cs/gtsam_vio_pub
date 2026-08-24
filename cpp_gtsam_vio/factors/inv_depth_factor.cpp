#include "inv_depth_factor.h"
#include <gtsam/base/numericalDerivative.h>
#include <functional>

namespace vio {

using gtsam::Pose3;
using gtsam::Point2;
using gtsam::Point3;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::Matrix;
using Cam = gtsam::PinholeCamera<gtsam::Cal3_S2>;

// ---------------------------------------------------------------- anchor (binary)
Vector InvDepthAnchorFactor::residual(const Pose3& X_anchor, const Vector3& l) const {
    try {
        const Pose3 wTc = X_anchor.compose(T_bc_);          // anchor camera in world
        const Point3 Pc = invdepth_to_cam_point(l);          // point in anchor cam frame
        const Point3 Pw = wTc.transformFrom(Pc);             // -> world
        Cam cam(wTc, *K_);
        return cam.project(Pw) - measured_;                  // reproject into anchor cam
    } catch (gtsam::CheiralityException&) {
        return Vector::Ones(2) * 2.0 * K_->fx();             // push behind-camera out
    }
}

Vector InvDepthAnchorFactor::evaluateError(const Pose3& X_anchor, const Vector3& l,
        boost::optional<Matrix&> H1, boost::optional<Matrix&> H2) const {
    if (H1) {
        *H1 = gtsam::numericalDerivative21<Vector, Pose3, Vector3>(
            std::bind(&InvDepthAnchorFactor::residual, this,
                      std::placeholders::_1, std::placeholders::_2),
            X_anchor, l);
    }
    if (H2) {
        *H2 = gtsam::numericalDerivative22<Vector, Pose3, Vector3>(
            std::bind(&InvDepthAnchorFactor::residual, this,
                      std::placeholders::_1, std::placeholders::_2),
            X_anchor, l);
    }
    return residual(X_anchor, l);
}

// ---------------------------------------------------------------- proj (ternary)
Vector InvDepthProjFactor::residual(const Pose3& X_anchor, const Pose3& X_obs,
                                    const Vector3& l) const {
    try {
        const Pose3 wTc_a = X_anchor.compose(T_bc_);         // anchor cam in world
        const Pose3 wTc_o = X_obs.compose(T_bc_);            // observing cam in world
        const Point3 Pc = invdepth_to_cam_point(l);          // point in anchor cam frame
        const Point3 Pw = wTc_a.transformFrom(Pc);           // -> world
        Cam cam(wTc_o, *K_);
        return cam.project(Pw) - measured_;                  // reproject into obs cam
    } catch (gtsam::CheiralityException&) {
        return Vector::Ones(2) * 2.0 * K_->fx();
    }
}

Vector InvDepthProjFactor::evaluateError(const Pose3& X_anchor, const Pose3& X_obs,
        const Vector3& l,
        boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
        boost::optional<Matrix&> H3) const {
    auto f = [this](const Pose3& a, const Pose3& o, const Vector3& p) {
        return this->residual(a, o, p);
    };
    if (H1) *H1 = gtsam::numericalDerivative31<Vector, Pose3, Pose3, Vector3>(f, X_anchor, X_obs, l);
    if (H2) *H2 = gtsam::numericalDerivative32<Vector, Pose3, Pose3, Vector3>(f, X_anchor, X_obs, l);
    if (H3) *H3 = gtsam::numericalDerivative33<Vector, Pose3, Pose3, Vector3>(f, X_anchor, X_obs, l);
    return residual(X_anchor, X_obs, l);
}

} // namespace vio
