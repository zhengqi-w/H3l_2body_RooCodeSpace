#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#include "GeneralHelper.hpp"
#include "binning/BinPlan.h"
#include "binning/BinPlanBuilder.h"
#include "checks/ChecksConfig.h"
#include "checks/ChecksEngine.h"
#include "output/OutputWriter.h"
#include "policies/ModePolicy.h"
#include "tasks/UnifiedTaskRunner.h"

#include <string>

namespace UnifiedAnalysis {

class AnalysisEngine {
public:
    int Run(const std::string &configPath, const std::string &modeOverride = "") const;

private:
    static std::string ResolveMode(const GeneralHelper::Json &cfg, const std::string &overrideMode);
    static ChecksConfig BuildChecksConfig(const GeneralHelper::Json &cfg);
};

} // namespace UnifiedAnalysis

#endif // ANALYSIS_ENGINE_H
