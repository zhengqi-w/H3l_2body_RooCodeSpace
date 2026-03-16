#ifndef MODE_POLICY_H
#define MODE_POLICY_H

#include <stdexcept>
#include <string>

namespace UnifiedAnalysis {

struct ModePolicy {
    std::string mode;
    std::string profileKey;
    bool useCentrality{false};
    bool usePt{false};
    bool useCt{false};
    bool useTopologyArray{false};
};

inline ModePolicy GetModePolicy(const std::string &modeRaw) {
    const std::string mode = modeRaw.empty() ? "bdt_spectrum" : modeRaw;
    if (mode == "bdt_spectrum" || mode == "spectrum") {
        return ModePolicy{"bdt_spectrum", "bdt_spectrum", true, true, false, false};
    }
    if (mode == "topology_spectrum") {
        return ModePolicy{"topology_spectrum", "topology_spectrum", true, true, false, true};
    }
    if (mode == "ct_extraction" || mode == "crosssection" || mode == "pt_ct") {
        return ModePolicy{"pt_ct", "pt_ct", false, true, true, false};
    }
    if (mode == "ct_single") {
        return ModePolicy{"ct_single", "ct_single", false, false, true, false};
    }
    throw std::runtime_error("Unsupported execution.mode: " + modeRaw);
}

} // namespace UnifiedAnalysis

#endif // MODE_POLICY_H
