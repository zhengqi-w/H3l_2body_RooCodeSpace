#ifndef UNIFIED_TASK_RUNNER_H
#define UNIFIED_TASK_RUNNER_H

#include "../GeneralHelper.hpp"
#include "../binning/BinPlan.h"
#include "../policies/ModePolicy.h"

namespace UnifiedAnalysis {

class UnifiedTaskRunner {
public:
    int Run(const GeneralHelper::Json &cfg, const ModePolicy &policy, const BinPlan &plan) const;

private:
    int RunBdtSpectrum(const GeneralHelper::Json &cfg, const BinPlan &plan) const;
    int RunTopologySpectrum(const GeneralHelper::Json &cfg, const BinPlan &plan) const;
    int RunCtExtraction(const GeneralHelper::Json &cfg, const BinPlan &plan) const;
    int RunRadCt(const GeneralHelper::Json &cfg, const BinPlan &plan) const;
    int RunCtSingle(const GeneralHelper::Json &cfg, const BinPlan &plan) const;
};

} // namespace UnifiedAnalysis

#endif // UNIFIED_TASK_RUNNER_H
