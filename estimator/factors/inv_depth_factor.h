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

// Re-express (theta, phi, rho) from anchor camera a to anchor camera b, in homogeneous
// form so it stays finite as rho -> 0:
//   h = R_ba m(theta, phi) + rho R_bw (t_wa - t_wb),   p_b = h / rho,
//   theta' = atan2(h_x, h_z), phi' = asin(h_y / |h|), rho' = rho / |h|.
// *in_front is false when the point is not in front of camera b (h_z <= 0).
gtsam::Vector3 reanchor_invdepth(const gtsam::Vector3& l, const gtsam::Pose3& wTc_a,
                                 const gtsam::Pose3& wTc_b, bool* in_front = nullptr);

// ---- quaternary: re-anchoring as a change of variables ----
// Ties the new variable L_b of a landmark, anchored at keyframe b, to its old variable
// L_a, anchored at keyframe a:  r = L_b - g(L_a, X_a, X_b),  g = reanchor_invdepth.
// The map L_a <-> L_b is invertible for fixed poses, so with a tight sigma eliminating
// L_a moves all of its information into L_b, including the part already folded into a
// marginal, and leaves the information on the poses unchanged.
// No measurement is re-emitted, unlike the depth transfer (point_depth_transfer), which
// re-attaches views that are already inside the marginal. A finite sigma only adds noise,
// so the approximation errs on the conservative side.
class InvDepthReanchorFactor
    : public gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> {
public:
    using Base = gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3>;

    InvDepthReanchorFactor(gtsam::Key oldLandmarkKey, gtsam::Key oldAnchorPoseKey,
                           gtsam::Key newAnchorPoseKey, gtsam::Key newLandmarkKey,
                           const gtsam::Pose3& T_bc, const gtsam::SharedNoiseModel& model)
        : Base(model, oldLandmarkKey, oldAnchorPoseKey, newAnchorPoseKey, newLandmarkKey),
          T_bc_(T_bc) {}

    gtsam::Vector residual(const gtsam::Vector3& l_a, const gtsam::Pose3& X_a,
                           const gtsam::Pose3& X_b, const gtsam::Vector3& l_b) const;

    gtsam::Vector evaluateError(const gtsam::Vector3& l_a, const gtsam::Pose3& X_a,
        const gtsam::Pose3& X_b, const gtsam::Vector3& l_b,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none,
        boost::optional<gtsam::Matrix&> H3 = boost::none,
        boost::optional<gtsam::Matrix&> H4 = boost::none) const override;

private:
    gtsam::Pose3 T_bc_;
};

} // namespace vio
