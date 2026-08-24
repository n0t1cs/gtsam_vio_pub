#pragma once
// Inverse-depth landmark factors, parameterized
// as L = (theta, phi, rho) in the ANCHOR CAMERA frame:
//
//   point_in_anchor_cam = (cos(phi)sin(theta)/rho, sin(phi)/rho, cos(phi)cos(theta)/rho)
//
// Unlike gtsam_unstable's InvDepthFactorVariant3, these compose the rigid
// body->camera extrinsic T_bc, because our optimization variables are BODY
// (IMU) poses, not camera poses:  T_w_cam = X_body * T_bc.
//
// Two factors per feature:
//   InvDepthAnchorFactor  (binary)  : the anchor keyframe's own measurement.
//   InvDepthProjFactor    (ternary) : every other observing keyframe.
//
// Jacobians are numerical (sufficient at VIO rates; matches the line factors).

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>
#include <Eigen/Dense>

namespace vio {

// (theta, phi, rho) -> 3D point in the anchor camera frame.
inline gtsam::Point3 invdepth_to_cam_point(const gtsam::Vector3& l) {
    const double theta = l(0), phi = l(1), rho = l(2);
    const double cphi = std::cos(phi);
    return gtsam::Point3(cphi * std::sin(theta) / rho,
                         std::sin(phi) / rho,
                         cphi * std::cos(theta) / rho);
}

// Convenience: build (theta,phi,rho) in the anchor camera frame from a 3D point
// expressed in that same anchor camera frame (depth = +Z).
inline gtsam::Vector3 cam_point_to_invdepth(const gtsam::Point3& Pc) {
    const double x = Pc.x(), y = Pc.y(), z = Pc.z();
    const double r = Pc.norm();
    const double theta = std::atan2(x, z);
    const double phi   = std::asin(y / r);
    const double rho   = 1.0 / Pc.norm();   // inverse range
    return gtsam::Vector3(theta, phi, rho);
}

// ---- binary: anchor keyframe's own measurement ----
class InvDepthAnchorFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>;

    InvDepthAnchorFactor(gtsam::Key anchorPoseKey, gtsam::Key landmarkKey,
                         const gtsam::Point2& measured, const gtsam::Pose3& T_bc,
                         const gtsam::Cal3_S2::shared_ptr& K,
                         const gtsam::SharedNoiseModel& model)
        : Base(model, anchorPoseKey, landmarkKey),
          measured_(measured), T_bc_(T_bc), K_(K) {}

    gtsam::Vector residual(const gtsam::Pose3& X_anchor, const gtsam::Vector3& l) const;

    gtsam::Vector evaluateError(const gtsam::Pose3& X_anchor, const gtsam::Vector3& l,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override;

private:
    gtsam::Point2 measured_;
    gtsam::Pose3 T_bc_;
    gtsam::Cal3_S2::shared_ptr K_;
};

// ---- ternary: any non-anchor observing keyframe ----
class InvDepthProjFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> {
public:
    using Base = gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3>;

    InvDepthProjFactor(gtsam::Key anchorPoseKey, gtsam::Key obsPoseKey, gtsam::Key landmarkKey,
                       const gtsam::Point2& measured, const gtsam::Pose3& T_bc,
                       const gtsam::Cal3_S2::shared_ptr& K,
                       const gtsam::SharedNoiseModel& model)
        : Base(model, anchorPoseKey, obsPoseKey, landmarkKey),
          measured_(measured), T_bc_(T_bc), K_(K) {}

    gtsam::Vector residual(const gtsam::Pose3& X_anchor, const gtsam::Pose3& X_obs,
                           const gtsam::Vector3& l) const;

    gtsam::Vector evaluateError(const gtsam::Pose3& X_anchor, const gtsam::Pose3& X_obs,
        const gtsam::Vector3& l,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none,
        boost::optional<gtsam::Matrix&> H3 = boost::none) const override;

private:
    gtsam::Point2 measured_;
    gtsam::Pose3 T_bc_;
    gtsam::Cal3_S2::shared_ptr K_;
};

} // namespace vio
