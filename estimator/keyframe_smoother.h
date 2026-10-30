#pragma once
// KeyframeSmoother — standalone KEYFRAME-INDEXED fixed-lag smoother on GTSAM.
//
// A peer of gtsam::BatchFixedLagSmoother / IncrementalFixedLagSmoother, but the
// sliding window is a fixed number of KEYFRAMES (N_kf) rather than a wall-clock lag.
// It owns its factor graph, its linearization point, the window, and an explicit
// Schur-complement marginalization with optional First-Estimate Jacobians.
//
// APPLICATION-AGNOSTIC BY CONSTRUCTION. The smoother knows nothing about poses,
// velocities, biases or landmarks, and never inspects gtsam::Symbol characters.
// Everything it needs about "when" a variable belongs to is supplied by the caller
// through a KeyIndexMap — the keyframe-index analog of GTSAM's KeyTimestampMap:
//
//     Key -> keyframe index
//
// The map is persistent and incremental, exactly like GTSAM's timestamps: entries
// for keys present in the passed map are updated, all others keep their previous
// index. That single contract expresses both eviction policies the VIO needs:
//   * a state stamped ONCE at creation ages out purely by index (poses, velocities,
//     biases: index i evicts when currentKf - i >= N_kf);
//   * a variable RE-STAMPED on every use evicts by inactivity (landmarks: re-stamp
//     on each observation and it survives while tracked, then goes stale).
// The staleness policy therefore lives in the caller, where the domain knowledge is,
// and the smoother stays generic.
//
// Which variables are held at first estimates is likewise a caller decision, passed
// as a FejKeyFilter predicate (default: every key).
//
// update() mirrors the GTSAM smoothers' surface:
//     update(newFactors, newTheta, keyIndices, factorsToRemove) -> Result

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace vio {

class KeyframeSmoother {
public:
    // Key -> keyframe index. The keyframe-window analog of
    // gtsam::FixedLagSmoother::KeyTimestampMap.
    using KeyIndexMap = std::map<gtsam::Key, std::uint64_t>;

    // Predicate selecting which keys are held at their FIRST estimate when they enter
    // a marginalization prior. Empty = every key.
    using FejKeyFilter = std::function<bool(gtsam::Key)>;

    // Predicate selecting which keys the SOLVE-GUARD may drop when they make the window's
    // linear system indeterminate. Empty = drop nothing (the indeterminate solve is
    // rethrown, preserving the pre-guard behaviour). The caller passes the classes that
    // are safe to cull — for the VIO, landmarks only; never a pose/velocity/bias, whose
    // degeneracy is a real failure the guard must not mask.
    using CullKeyFilter = std::function<bool(gtsam::Key)>;

    // Per-update meta (the subset of gtsam::FixedLagSmoother::Result we populate).
    struct Result {
        std::size_t iterations = 0;
        std::size_t nonlinearVariables = 0;
        double      error = 0.0;
    };

    // Conditioning diagnostic (runtime.kf_diag_cond). One record per marginalization, only
    // populated when setCondDiagnostics(true). Measures the conditioning of the system
    // the Schur step actually eliminates, which is the quantity the square-root argument
    // is about: Cholesky forms the normal equations and therefore works at cond(J)^2,
    // while a QR/square-root elimination works at cond(J). Recording cond(J) here gives
    // the margin to the float32 cliff without needing a float32 build.
    // NOTE: computing singular values per marginalization is EXPENSIVE (dense SVD).
    // Runs with this enabled are diagnostic only — their timing is NOT comparable to a
    // normal run and must never be quoted in a Pareto claim.
    struct CondDiag {
        std::uint64_t kf        = 0;   // cutoff keyframe index of this marginalization
        int    margKeys         = 0;   // variables eliminated
        int    rows             = 0;   // rows of the eliminated system's Jacobian J
        int    cols             = 0;   // cols of J
        double smax             = 0.0; // largest  singular value of J
        double smin             = 0.0; // smallest singular value of J
        double cond             = 0.0; // smax/smin = cond(J)
        int    rankJ            = 0;   // numerical rank of J at tol = max(m,n)*eps*smax
        double sminEff          = 0.0; // smallest singular value ABOVE tol
        double condEff          = 0.0; // smax/sminEff = effective cond(J) on its row space
        int    priorRows        = 0;   // resulting marginal prior (already square-root form)
        int    priorCols        = 0;
        double priorSmin        = 0.0;
        double priorCond        = 0.0; // raw smax/smin of the prior
        int    priorRank        = 0;
        double priorCondEff     = 0.0; // effective cond of the prior
        double priorCondScaled  = 0.0; // effective cond AFTER unit-norm column scaling
                                       // (Jacobi): separates genuine ill-conditioning
                                       // from the mixed units of a VI state vector
                                       // (m, rad, m/s, bias, inverse depth).
    };

    // windowSize = N_kf: keyframes kept before the oldest is evicted.
    // useFej: hold the marginalization at first-estimate Jacobians (see fejValues_).
    // fejFilter: which keys FEJ applies to; default (empty) = all of them.
    // maxIterations: window re-optimization iterations per update(). This is the
    //   filter<->smoother knob: 10 = a re-linearizing smoother; 1 = a single
    //   linearization per step, i.e. an (iterated-)EKF update. Combined with a small
    //   windowSize it degenerates this class toward a filter on the same code path.
    // useGaussNewton: solve each window with an undamped Gauss-Newton step instead of
    //   Levenberg-Marquardt. With maxIterations=1 this makes the update a single EKF-style
    //   linearized correction (no LM damping). Default false = LM (unchanged behaviour).
    // cullFilter: which keys the solve-guard may drop on an indeterminate window (see
    //   CullKeyFilter; default empty = drop nothing = pre-guard behaviour).
    // useQrMarginalization: eliminate the evicted keys with EliminateQR (square-root form,
    //   operates on the Jacobian at cond(J)) instead of GTSAM's default
    //   EliminatePreferCholesky (normal equations, cond(J)^2). Default false = unchanged
    //   behaviour. In double precision the two are algebraically equivalent; this is the
    //   knob for study S1, which tests whether that equivalence actually holds here.
    explicit KeyframeSmoother(int windowSize, bool useFej = true,
                              FejKeyFilter fejFilter = FejKeyFilter(),
                              int maxIterations = 10,
                              bool useGaussNewton = false,
                              CullKeyFilter cullFilter = CullKeyFilter(),
                              bool useQrMarginalization = false)
        : window_(windowSize), useFej_(useFej), fejFilter_(std::move(fejFilter)),
          maxIter_(maxIterations), useGn_(useGaussNewton),
          cullFilter_(std::move(cullFilter)), useQrMarg_(useQrMarginalization) {}

    // Add factors/values, refresh keyframe indices, optimize, then bound the window.
    // Argument order mirrors gtsam::FixedLagSmoother::update.
    Result update(const gtsam::NonlinearFactorGraph& newFactors = gtsam::NonlinearFactorGraph(),
                  const gtsam::Values& newTheta = gtsam::Values(),
                  const KeyIndexMap& keyIndices = KeyIndexMap(),
                  const gtsam::FactorIndices& factorsToRemove = gtsam::FactorIndices());

    gtsam::Values calculateEstimate() const { return theta_; }

    // Marginal covariance of a key (NEES study), from the current bounded graph +
    // linearization point. Throws on an indeterminate system (caller catches).
    // NOTE: re-factorizes the whole graph per call — fine for one key per keyframe,
    // unusable at landmark scale.
    gtsam::Matrix marginalCovariance(gtsam::Key key) const;

    // Joint marginal covariance over `keys`, returned as one dense block matrix in
    // the order given (for two Pose3 keys: 12x12, tangent [rot,trans] per block).
    // Needed because the RELATIVE uncertainty between two states is NOT recoverable
    // from their separate marginals -- those are dominated by common-mode drift and
    // by the VI-unobservable global position/yaw, so differencing them double-counts
    // the shared part. Only the cross-block carries the correlation that cancels.
    gtsam::Matrix jointMarginalCovariance(const gtsam::KeyVector& keys) const;

    const gtsam::NonlinearFactorGraph& getFactors() const { return graph_; }
    const gtsam::Values& getLinearizationPoint() const { return theta_; }
    // Newest keyframe index the smoother has been told about.
    std::uint64_t currentKeyframe() const { return currentKf_; }

    // S2: enable/read the per-marginalization conditioning log. Off by default; the
    // committed system pays nothing for it.
    void setCondDiagnostics(bool on) { diagCond_ = on; }
    const std::vector<CondDiag>& condLog() const { return condLog_; }

private:
    // Schur-marginalize every key whose keyframe index is older than `cutoff`.
    // The marginal on the surviving separator is appended as LinearContainerFactors,
    // linearized per the FEJ policy.
    void marginalizeOlderThan(std::uint64_t cutoff);

    // Is this variable held at its first estimate? (fejFilter_; empty = all.)
    bool isFejKey(gtsam::Key k) const { return !fejFilter_ || fejFilter_(k); }

    // May the solve-guard drop this variable? (cullFilter_; empty = none.)
    bool isCullKey(gtsam::Key k) const { return cullFilter_ && cullFilter_(k); }

    // Drop every factor touching `key` and erase its state from the window. Called by the
    // solve-guard on a rank-deficient landmark; graph_ is rebuilt compactly (consistent
    // with marginalizeOlderThan — the caller passes no stable factor indices here).
    void cullKey(gtsam::Key key);

    int          window_;                 // N_kf (keyframes kept)
    bool         useFej_;                 // hold marginalization at first estimates
    FejKeyFilter fejFilter_;              // which keys FEJ covers (empty = all)
    int          maxIter_ = 10;           // window re-opt iterations (1 = EKF-like)
    bool         useGn_ = false;          // Gauss-Newton (undamped) instead of LM
    CullKeyFilter cullFilter_;            // keys the solve-guard may drop (empty = none)
    bool         useQrMarg_ = false;      // marginalize via EliminateQR (square-root) vs Cholesky
    bool         diagCond_  = false;      // S2: record conditioning at each marginalization
    std::vector<CondDiag> condLog_;       // S2: one record per marginalization

    gtsam::NonlinearFactorGraph graph_;   // accumulated, bounded to the window
    gtsam::Values theta_;                 // current linearization point / estimate

    // The smoother's keyframe clock: caller-supplied index per key, and the newest
    // index seen. Monotone — it never runs backwards.
    KeyIndexMap   keyIndex_;
    std::uint64_t currentKf_ = 0;

    // FEJ store (boundary-only): the FIRST estimate of every variable that has entered
    // a marginalization prior, frozen when it joined the separator and never updated.
    // Marginalization linearizes at these values so successive marginals share one
    // tangent frame per variable — which is what stops information leaking into the
    // unobservable position/yaw nullspace. Keys leave when they are marginalized out.
    gtsam::Values fejValues_;
};

} // namespace vio
