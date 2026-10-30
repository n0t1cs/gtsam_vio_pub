#include "keyframe_smoother.h"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/SVD>
#include <Eigen/QR>

#include <algorithm>
#include <limits>
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

    // 5. optimize the (bounded) window, warm-started from theta_. The optimizer is the
    //    companion of the filter<->smoother knob: LM (default) damps each step; Gauss-Newton
    //    takes the full undamped step, so a SINGLE GN iteration IS one EKF update (making the
    //    "SLKS is a filter in disguise" claim exact). GN has no damping to rescue an
    //    ill-conditioned window, so it is the robustness side of the GN-vs-LM study; iSAM2
    //    (the incremental arm) is likewise GN-based, so this also isolates the LM/GN axis
    //    between our two proven back-ends.
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
            // 5. optimize the (bounded) window, warm-started from theta_. LM (default)
            //    damps each step; a single Gauss-Newton iteration IS one EKF update.
            if (useGn_) {
                gtsam::GaussNewtonParams p;
                p.setMaxIterations(maxIter_);
                gtsam::GaussNewtonOptimizer opt(graph_, theta_, p);
                theta_ = opt.optimize();
                iters = opt.iterations();
            } else {
                gtsam::LevenbergMarquardtParams p;
                p.setMaxIterations(maxIter_);   // 1 = single linearization/step (EKF-like); 10 = smoother
                gtsam::LevenbergMarquardtOptimizer opt(graph_, theta_, p);
                theta_ = opt.optimize();
                iters = opt.iterations();
            }

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
    r.error = graph_.error(theta_);
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
    //      window and diverges). ----
    //      The eliminate function is selectable (runtime.kf_marg_elim). GTSAM's default for a
    //      GaussianFactorGraph is EliminatePreferCholesky: it forms the normal equations,
    //      so the marginal is computed at cond(J)^2 (it does fall back to QR when the
    //      Cholesky attempt fails). EliminateQR stays in square-root form throughout,
    //      working on the Jacobian at cond(J). In double precision the two are
    //      algebraically equivalent — measuring that equivalence is the point.
    const auto elimFn = useQrMarg_ ? static_cast<gtsam::GaussianFactorGraph::Eliminate>(gtsam::EliminateQR)
                                   : static_cast<gtsam::GaussianFactorGraph::Eliminate>(gtsam::EliminatePreferCholesky);
    const auto elim = linear.eliminatePartialMultifrontal(gtsam::Ordering(margKeys), elimFn);

    // ---- Conditioning diagnostic (opt-in, runtime.kf_diag_cond). Measure the
    //      conditioning of the system this Schur step eliminates. cond(J) is the number
    //      the square-root argument turns on: a Cholesky elimination forms J^T J and so
    //      operates at cond(J)^2, and single precision (eps^-1 ~ 1.2e7) fails once
    //      cond(J)^2 approaches that. Recording cond(J) in double therefore gives the
    //      float32 margin WITHOUT needing a float32 build.
    //      Dense SVD per marginalization — diagnostic runs only, timing not comparable. ----
    if (diagCond_) {
        CondDiag d;
        d.kf       = cutoff;
        d.margKeys = static_cast<int>(margKeys.size());

        // Assemble the dense Jacobian OURSELVES, factor by factor, instead of calling
        // GaussianFactorGraph::jacobian(). That convenience path goes through Scatter and
        // trips an Eigen assertion (which abort()s — it cannot be caught) on the graphs
        // produced here after a solve-guard cull. Building the block layout directly from
        // each factor's own keys and dimensions is equivalent and fully under our control.
        auto denseJacobian = [](const gtsam::GaussianFactorGraph& g) -> gtsam::Matrix {
            std::map<gtsam::Key, std::size_t> dim, off;
            std::vector<std::pair<gtsam::Matrix, gtsam::Vector>> blocks;
            blocks.reserve(g.size());
            std::size_t totalRows = 0;
            for (const auto& f : g) {
                if (!f) continue;
                std::pair<gtsam::Matrix, gtsam::Vector> ab = f->jacobian();
                for (auto it = f->begin(); it != f->end(); ++it)
                    if (!dim.count(*it)) dim[*it] = f->getDim(it);
                totalRows += static_cast<std::size_t>(ab.first.rows());
                blocks.push_back(std::move(ab));
            }
            std::size_t totalCols = 0;
            for (const auto& kv : dim) { off[kv.first] = totalCols; totalCols += kv.second; }
            if (totalRows == 0 || totalCols == 0) return gtsam::Matrix();
            gtsam::Matrix A = gtsam::Matrix::Zero(totalRows, totalCols);
            std::size_t r = 0, bi = 0;
            for (const auto& f : g) {
                if (!f) continue;
                const gtsam::Matrix& Af = blocks[bi].first;
                std::size_t c = 0;
                for (auto it = f->begin(); it != f->end(); ++it) {
                    const std::size_t dk = dim[*it];
                    if (c + dk <= static_cast<std::size_t>(Af.cols()))
                        A.block(r, off[*it], Af.rows(), dk) = Af.block(0, c, Af.rows(), dk);
                    c += dk;
                }
                r += static_cast<std::size_t>(Af.rows());
                ++bi;
            }
            return A;
        };

        // Effective conditioning over the NUMERICALLY NON-ZERO spectrum. A raw smax/smin
        // is meaningless here: the eliminated subgraph is rank-deficient by construction
        // (that is exactly why the Schur step projects onto the separator), so smin is
        // legitimately 0 and the raw ratio is inf. What governs numerical behaviour is the
        // spread across the directions the system actually constrains, at the standard
        // tolerance tol = max(m,n)*eps*smax.
        // Eigen's BDCSVD has a defect in perturbCol0() that indexes out of range on
        // rank-deficient matrices with repeated singular values — precisely the case
        // here — and it abort()s rather than throwing. GTSAM ships that Eigen version.
        // So: the large eliminated system is characterised with a RANK-REVEALING
        // column-pivoted QR (robust, O(mn^2), rank + a condition ESTIMATE from the R
        // diagonal), while the small marginal prior — the quantity the float32 argument
        // actually turns on — gets an exact JacobiSVD.
        auto spectrumQR = [](const gtsam::Matrix& M, int& rank, double& smax,
                             double& sminEff, double& condEff) {
            rank = 0; smax = sminEff = condEff = 0.0;
            if (M.rows() == 0 || M.cols() == 0) return;
            Eigen::ColPivHouseholderQR<gtsam::Matrix> qr(M);
            rank = static_cast<int>(qr.rank());
            const gtsam::Matrix R = qr.matrixR();
            const int n = static_cast<int>(std::min(R.rows(), R.cols()));
            if (n == 0) return;
            smax = std::abs(R(0, 0));
            const int last = std::max(0, std::min(rank, n) - 1);
            sminEff = std::abs(R(last, last));
            condEff = (sminEff > 0.0) ? (smax / sminEff)
                                      : std::numeric_limits<double>::infinity();
        };
        auto spectrumSVD = [](const gtsam::Matrix& M, int& rank, double& smax,
                              double& sminEff, double& condEff) {
            rank = 0; smax = sminEff = condEff = 0.0;
            if (M.rows() == 0 || M.cols() == 0) return;
            Eigen::JacobiSVD<gtsam::Matrix> svd(M);
            const gtsam::Vector sv = svd.singularValues();
            if (sv.size() == 0) return;
            smax = sv(0);
            const double tol = static_cast<double>(std::max(M.rows(), M.cols())) *
                               std::numeric_limits<double>::epsilon() * smax;
            for (int i = 0; i < sv.size(); ++i)
                if (sv(i) > tol) { ++rank; sminEff = sv(i); }
            condEff = (sminEff > 0.0) ? (smax / sminEff)
                                      : std::numeric_limits<double>::infinity();
        };

        try {
            const gtsam::Matrix A = denseJacobian(linear);
            d.rows = static_cast<int>(A.rows());
            d.cols = static_cast<int>(A.cols());
            double smax = 0.0;
            spectrumQR(A, d.rankJ, smax, d.sminEff, d.condEff);
            d.smax = smax;
            d.smin = (d.rankJ == d.cols) ? d.sminEff : 0.0;   // exactly 0 when rank-deficient
            d.cond = (d.smin > 0.0) ? (smax / d.smin)
                                    : std::numeric_limits<double>::infinity();

            if (elim.second) {
                gtsam::Matrix P = denseJacobian(*elim.second);
                d.priorRows = static_cast<int>(P.rows());
                d.priorCols = static_cast<int>(P.cols());
                double psmax = 0.0;
                spectrumSVD(P, d.priorRank, psmax, d.priorSmin, d.priorCondEff);
                d.priorCond = d.priorCondEff;
                // Jacobi (unit-column-norm) scaling strips the arbitrary unit scaling of a
                // mixed VI state vector (m, rad, m/s, bias, inverse depth), leaving the
                // geometric conditioning.
                if (P.rows() > 0 && P.cols() > 0) {
                    for (int c = 0; c < P.cols(); ++c) {
                        const double n = P.col(c).norm();
                        if (n > 0.0) P.col(c) /= n;
                    }
                    int r2 = 0; double s2 = 0.0, se2 = 0.0;
                    spectrumSVD(P, r2, s2, se2, d.priorCondScaled);
                }
            }
        } catch (const std::exception&) {
            // a diagnostic must never take the estimator down; leave the record partial
        }
        condLog_.push_back(d);
    }
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
