#include "keyframe_smoother.h"

#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/inference/Symbol.h>

#include <algorithm>
#include <set>
#include <vector>

namespace vio {

KeyframeSmoother::Result KeyframeSmoother::update(
        const gtsam::NonlinearFactorGraph& newFactors,
        const gtsam::Values& newTheta,
        const KeyIndexMap& keyIndices,
        const gtsam::FactorIndices& factorsToRemove) {

    // 1. new variables (all keys are fresh per the caller contract: the new keyframe's
    //    states + newly-initialized structure; re-observations append a factor only,
    //    never re-insert a value).
    theta_.insert(newTheta);

    // 2. defensive factor removal.
    for (auto idx : factorsToRemove)
        if (idx < graph_.size()) graph_.remove(idx);

    // 3. append this keyframe's new factors.
    graph_.push_back(newFactors);

    // 4. advance the keyframe clock from the caller's map. Persistent + incremental,
    //    exactly like GTSAM's timestamps: keys named here are re-stamped, every other
    //    key keeps the index it already had. A state stamped once at creation
    //    therefore ages out by pure index; a variable re-stamped on each use evicts by
    //    inactivity. The smoother needs no knowledge of which is which.
    for (const auto& kv : keyIndices) {
        keyIndex_[kv.first] = kv.second;
        currentKf_ = std::max(currentKf_, kv.second);   // monotone
    }
    const std::uint64_t currentKf = currentKf_;

    // 5. optimize the (bounded) window, warm-started from theta_. Gauss-Newton takes the full
    //    undamped step, so a SINGLE GN iteration IS one EKF update (making the "SLKS is a
    //    filter in disguise" claim exact). iSAM2 (the incremental arm) is likewise GN-based.
    // ...under a SOLVE-GUARD (step 6 folded in). An underconstrained landmark (too few
    //    effective views, degenerate geometry) can make the window's linear system
    //    indeterminate; GTSAM then throws IndeterminantLinearSystemException and, uncaught,
    //    ABORTS the process. Instead we read the offending variable off the exception and,
    //    if the caller has declared its class cullable (cullFilter_ — the VIO passes
    //    landmarks only), drop it and retry. An underconstrained landmark carries no
    //    information in the singular direction, so removing it is numerically free. Both
    //    the optimize (step 5) and the marginalization (step 6) can raise it, so both live
    //    inside the try; neither commits its mutation before the throw point (optimize
    //    assigns theta_ only on success; marginalizeOlderThan assigns graph_ only after the
    //    elimination), so a retry re-runs cleanly on the reduced window. A genuine
    //    pose/velocity/bias degeneracy is NOT cullable and still surfaces (rethrown), as
    //    does exhausting the retry budget.
    std::size_t iters = 0;
    const int kMaxGuard = 8;
    for (int attempt = 0; ; ++attempt) {
        try {
            // 5. optimize the (bounded) window, warm-started from theta_.
            gtsam::GaussNewtonParams p;
            p.setMaxIterations(maxIter_);
            gtsam::GaussNewtonOptimizer opt(graph_, theta_, p);
            theta_ = opt.optimize();
            iters = opt.iterations();

            // 6. bound the window to the newest `window_` keyframes. Poses older than the
            //    cutoff, and landmarks not observed since the cutoff, are marginalized out.
            if (currentKf >= static_cast<std::uint64_t>(window_))
                marginalizeOlderThan(currentKf - static_cast<std::uint64_t>(window_));
            break;   // window solved and bounded cleanly
        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            const gtsam::Key bad = e.nearbyVariable();
            if (attempt >= kMaxGuard || !isCullKey(bad)) throw;   // not ours to fix
            cullKey(bad);   // drop the degenerate landmark; the surviving theta_ is intact
        }
    }

    Result r;
    r.iterations = iters;
    r.nonlinearVariables = theta_.size();
    return r;
}

void KeyframeSmoother::marginalizeOlderThan(std::uint64_t cutoff) {
    // ---- select what leaves the window ----
    // ONE uniform rule: a variable leaves when its keyframe index falls behind the
    // cutoff. The caller's stamping policy decides what that means per variable class
    // (stamp-once = age out; re-stamp-on-use = evict on inactivity). A key with no
    // index yet is never evicted — the caller has not told us when it belongs to.
    gtsam::KeyVector margKeys;
    for (const gtsam::Key key : theta_.keys()) {
        const auto it = keyIndex_.find(key);
        if (it != keyIndex_.end() && it->second < cutoff) margKeys.push_back(key);
    }
    if (margKeys.empty()) return;

    // ---- split the window: factors touching an evicted key feed the Schur
    //      complement; the rest are kept verbatim. ----
    const std::set<gtsam::Key> margSet(margKeys.begin(), margKeys.end());
    gtsam::NonlinearFactorGraph removed, kept;
    for (const auto& f : graph_) {
        if (!f) continue;
        bool touches = false;
        for (const gtsam::Key k : f->keys())
            if (margSet.count(k)) { touches = true; break; }
        (touches ? removed : kept).push_back(f);
    }

    // ---- FIRST-ESTIMATE JACOBIANS (boundary-only). Build the linearization point
    //      used for this marginalization: any variable that has ALREADY entered a
    //      marginalization prior is linearized at its FROZEN first estimate; a
    //      variable joining the boundary for the first time uses the current
    //      estimate. Without this, the information the new factors contribute about a
    //      separator variable is expressed in a different tangent frame than the
    //      information the existing marginal already holds about it, and the
    //      difference accumulates as spurious certainty along the unobservable
    //      position/yaw directions. With useFej_ = false this degrades to the plain
    //      current-estimate linearization (the noFEJ A/B arm). ----
    gtsam::Values linPoint;
    for (const gtsam::Key k : removed.keys()) {
        if (useFej_ && isFejKey(k) && fejValues_.exists(k))
            linPoint.insert(k, fejValues_.at(k));
        else if (theta_.exists(k))
            linPoint.insert(k, theta_.at(k));
    }

    // ---- Build the linear system for the Schur complement.
    //      TRUE FEJ (DM-VIO FEJNoiseModelFactor semantics): the JACOBIAN A is evaluated
    //      at the FROZEN first estimates, but the RESIDUAL b is evaluated at the CURRENT
    //      estimate:
    //          A = d(error)/dx |_fej ,   b = -error(x_current)
    //      Freezing b as well would tell the estimator its error is what it was a
    //      keyframe ago — a pure accuracy loss with no consistency benefit (it is the
    //      frozen A that preserves the observability nullspace). Since b is taken at
    //      theta_, increments are measured from theta_, so the marginal is wrapped at
    //      theta_ below (NOT at the frozen point).
    //      Factors that do not linearize to a JacobianFactor fall back to the plain
    //      current-estimate linearization. ----
    gtsam::GaussianFactorGraph linear;
    if (!useFej_) {
        linear = *removed.linearize(theta_);
    } else {
        const gtsam::GaussianFactorGraph::shared_ptr linFej = removed.linearize(linPoint);
        const gtsam::GaussianFactorGraph::shared_ptr linCur = removed.linearize(theta_);
        for (size_t i = 0; i < linCur->size(); ++i) {
            const auto jfF = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(linFej->at(i));
            const auto jfC = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(linCur->at(i));
            if (jfF && jfC && jfF->rows() == jfC->rows()) {
                std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms;
                for (auto it = jfF->begin(); it != jfF->end(); ++it)
                    terms.emplace_back(*it, gtsam::Matrix(jfF->getA(it)));   // A at FEJ
                linear.push_back(boost::make_shared<gtsam::JacobianFactor>(
                    terms, gtsam::Vector(jfC->getb())));                     // b at current
            } else {
                linear.push_back(linCur->at(i));
            }
        }
    }

    // ---- Schur complement: eliminate the evicted keys and wrap the resulting marginal
    //      on the separator in LinearContainerFactors. This preserves the evicted
    //      information instead of discarding it (dropping it outright starves the
    //      window and diverges). GTSAM's EliminatePreferCholesky (EliminateQR gives
    //      the identical estimate at twice the cost). ----
    const auto elim = linear.eliminatePartialMultifrontal(
        gtsam::Ordering(margKeys),
        static_cast<gtsam::GaussianFactorGraph::Eliminate>(gtsam::EliminatePreferCholesky));

    const gtsam::NonlinearFactorGraph marginal =
        gtsam::LinearContainerFactor::ConvertLinearGraph(*elim.second, theta_);

    kept.push_back(marginal);
    graph_ = kept;

    // ---- freeze the first estimate of every variable that just JOINED the boundary
    //      (i.e. appears in the new marginal and was not already frozen). Variables
    //      already in the store keep their original first estimate — that is the FEJ
    //      invariant. ----
    if (useFej_) {
        for (const gtsam::Key k : elim.second->keys())
            if (isFejKey(k) && !fejValues_.exists(k) && linPoint.exists(k))
                fejValues_.insert(k, linPoint.at(k));
    }

    // ---- evicted keys leave the window and the boundary ----
    for (const gtsam::Key k : margKeys) {
        if (theta_.exists(k))      theta_.erase(k);
        if (fejValues_.exists(k))  fejValues_.erase(k);
        keyIndex_.erase(k);
    }
}

void KeyframeSmoother::cullKey(gtsam::Key key) {
    // Rebuild the graph without any factor that touches `key`, then erase its state,
    // its frozen first estimate and its keyframe stamp. The factor rebuild mirrors
    // marginalizeOlderThan's kept/removed split; because a keyframe smoother is never
    // handed stable factor indices (its graph is already rebuilt on every
    // marginalization), compacting here is safe. Only reached for keys the caller marked
    // cullable (landmarks): a rank-deficient landmark contributes no information in the
    // singular direction it created, so dropping it — rather than Schur-preserving it —
    // loses nothing and is what lets the retry converge.
    gtsam::NonlinearFactorGraph pruned;
    for (const auto& f : graph_) {
        if (!f) continue;
        bool touches = false;
        for (const gtsam::Key fk : f->keys())
            if (fk == key) { touches = true; break; }
        if (!touches) pruned.push_back(f);
    }
    graph_ = pruned;

    if (theta_.exists(key))     theta_.erase(key);
    if (fejValues_.exists(key)) fejValues_.erase(key);
    keyIndex_.erase(key);
}

gtsam::Matrix KeyframeSmoother::marginalCovariance(gtsam::Key key) const {
    gtsam::Marginals marg(graph_, theta_);
    return marg.marginalCovariance(key);
}

gtsam::Matrix KeyframeSmoother::jointMarginalCovariance(const gtsam::KeyVector& keys) const {
    gtsam::Marginals marg(graph_, theta_);
    gtsam::JointMarginal jm = marg.jointMarginalCovariance(keys);
    // Assemble explicitly from the blocks rather than trusting fullMatrix()'s
    // internal ordering, which follows elimination order, not `keys`.
    std::vector<int> dim; dim.reserve(keys.size());
    int n = 0;
    for (const gtsam::Key k : keys) { const int d = (int)jm.at(k, k).rows(); dim.push_back(d); n += d; }
    gtsam::Matrix out = gtsam::Matrix::Zero(n, n);
    int ro = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        int co = 0;
        for (size_t j = 0; j < keys.size(); ++j) {
            out.block(ro, co, dim[i], dim[j]) = jm.at(keys[i], keys[j]);
            co += dim[j];
        }
        ro += dim[i];
    }
    return out;
}

} // namespace vio
