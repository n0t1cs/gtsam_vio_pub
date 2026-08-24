#pragma once
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "loop_keyframe.h"
#include "DBoW2.h"   // BriefVocabulary, BriefDatabase (loop/ThirdParty/DBoW)

namespace vio {

// Owns the BRIEF bag-of-words vocabulary + database and the harvested keyframe
// store. Stage 1 only indexes; loop querying/verification is added in Stage 2.
class LoopDatabase {
public:
    LoopDatabase(const std::string& vocab_path, const LoopParams& lp = LoopParams());

    // Store a harvested keyframe and index its FAST-corner descriptors. The BoW
    // entry id equals the keyframes() index (both grow together).
    void add(const LoopKeyframe& kf);

    size_t size() const { return kfs_.size(); }
    const std::vector<LoopKeyframe>& keyframes() const { return kfs_; }

    // ONLINE per-keyframe loop query: DBoW-query the EARLIER stored keyframes
    // (temporal neighbours skipped) for `cur`, then BRIEF-match + PnP-RANSAC
    // verify. Call BEFORE add(cur) so `cur` is not yet in the database. T_BC is
    // the camera-in-body extrinsic (for the PnP guess + body-pose recovery).
    std::vector<LoopEdge> query_loops(const LoopKeyframe& cur, const Eigen::Matrix4d& T_BC);

private:
    // Verify a DBoW candidate (old) against the current KF; fill `edge` on success.
    bool verify_loop(const LoopKeyframe& cur, const LoopKeyframe& old,
                     const Eigen::Matrix4d& T_BC, LoopEdge& edge) const;
    std::unique_ptr<BriefVocabulary> voc_;
    BriefDatabase db_;
    std::vector<LoopKeyframe> kfs_;
    LoopParams lp_;
};

} // namespace vio
