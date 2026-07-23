#ifndef CHECKS_CONFIG_H
#define CHECKS_CONFIG_H

#include <unordered_map>
#include <string>
#include <vector>

namespace UnifiedAnalysis {

struct AxisSpec {
    int nBins{80};
    double min{0.0};
    double max{1.0};
    std::string title;
};

struct Hist2DPair {
    std::string x;
    std::string y;
};

struct CheckBlockConfig {
    bool enable{false};
    std::string file;
    std::vector<std::string> files;
    std::string tree;
    std::string selection;
    std::vector<std::string> variables;
    std::vector<Hist2DPair> hist2dPairs;
};

struct ChecksConfig {
    bool enabled{false};
    bool savePdf{false};
    std::string outputRootFile{"./Outputs/Checks/checks.root"};
    std::unordered_map<std::string, AxisSpec> axisPool;
    CheckBlockConfig mcChecks;
    CheckBlockConfig dataAllChecks;
    CheckBlockConfig onTheFlyChecks;
};

} // namespace UnifiedAnalysis

#endif // CHECKS_CONFIG_H
