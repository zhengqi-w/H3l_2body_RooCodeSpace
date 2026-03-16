#ifndef BIN_PLAN_H
#define BIN_PLAN_H

#include <string>
#include <vector>

namespace UnifiedAnalysis {

struct BinPlanItem {
    bool hasCen{false};
    bool hasPt{false};
    bool hasCt{false};

    double cenMin{-1.0};
    double cenMax{-1.0};
    double ptMin{-1.0};
    double ptMax{-1.0};
    double ctMin{-1.0};
    double ctMax{-1.0};

    std::string mode;
    std::string label;
    std::string snapshotDataPath;
    std::string snapshotMcPath;
    std::string topologySelection;
};

struct BinPlan {
    std::string mode;
    std::vector<BinPlanItem> items;

    std::vector<double> cenEdges;
    std::vector<double> ptEdges;
    std::vector<double> ctEdges;
};

} // namespace UnifiedAnalysis

#endif // BIN_PLAN_H
