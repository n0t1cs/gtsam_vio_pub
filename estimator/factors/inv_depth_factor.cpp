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

// ---------------------------------------------------------------- re-anchor (quaternary)
Vector3 reanchor_invdepth(const Vector3& l, const Pose3& wTc_a, const Pose3& wTc_b,
                          bool* in_front) {
    const double theta = l(0), phi = l(1), rho = l(2);
    const double cphi = std::cos(phi);
    const Vector3 m(cphi * std::sin(theta), std::sin(phi), cphi * std::cos(theta));
    const gtsam::Matrix3 R_bw = wTc_b.rotation().transpose();
    const Vector3 h = R_bw * (wTc_a.rotation().matrix() * m)
                    + rho * (R_bw * (wTc_a.translation() - wTc_b.translation()));
    const double n = h.norm();
    if (in_front) *in_front = h.z() > 0.0;
    return Vector3(std::atan2(h.x(), h.z()), std::asin(h.y() / n), rho / n);
}

Vector InvDepthReanchorFactor::residual(const Vector3& l_a, const Pose3& X_a,
                                        const Pose3& X_b, const Vector3& l_b) const {
    const Vector3 g = reanchor_invdepth(l_a, X_a.compose(T_bc_), X_b.compose(T_bc_));
    Vector3 r = l_b - g;
    // theta is an atan2 angle: wrap the difference so a point near the image border of
    // camera b cannot make the residual jump by 2 pi.
    r(0) = std::atan2(std::sin(r(0)), std::cos(r(0)));
    return r;
}

Vector InvDepthReanchorFactor::evaluateError(const Vector3& l_a, const Pose3& X_a,
        const Pose3& X_b, const Vector3& l_b,
        boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
        boost::optional<Matrix&> H3, boost::optional<Matrix&> H4) const {
    auto f = [this](const Vector3& la, const Pose3& xa, const Pose3& xb, const Vector3& lb) {
        return this->residual(la, xa, xb, lb);
    };
    if (H1) *H1 = gtsam::numericalDerivative41<Vector, Vector3, Pose3, Pose3, Vector3>(f, l_a, X_a, X_b, l_b);
    if (H2) *H2 = gtsam::numericalDerivative42<Vector, Vector3, Pose3, Pose3, Vector3>(f, l_a, X_a, X_b, l_b);
    if (H3) *H3 = gtsam::numericalDerivative43<Vector, Vector3, Pose3, Pose3, Vector3>(f, l_a, X_a, X_b, l_b);
    if (H4) *H4 = gtsam::numericalDerivative44<Vector, Vector3, Pose3, Pose3, Vector3>(f, l_a, X_a, X_b, l_b);
    return residual(l_a, X_a, X_b, l_b);
}

} // namespace vio
