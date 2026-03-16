#ifndef BIN_PLAN_BUILDER_H
#define BIN_PLAN_BUILDER_H

#include "BinPlan.h"
#include "../policies/ModePolicy.h"
#include "../GeneralHelper.hpp"

namespace UnifiedAnalysis {

class BinPlanBuilder {
public:
    BinPlan Build(const GeneralHelper::Json &cfg, const ModePolicy &policy) const;

private:
    static std::string FormatEdge(double value);
    static std::string MakeLabel(const BinPlanItem &item);
    static std::string BuildDataSnapshotPath(const std::string &dir, const BinPlanItem &item);
    static std::string BuildMcSnapshotPath(const std::string &dir, const BinPlanItem &item);
    static std::string JoinSelection(const std::string &a, const std::string &b);
};

} // namespace UnifiedAnalysis

#endif // BIN_PLAN_BUILDER_H
