#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#include <string>

namespace UnifiedAnalysis {

struct DataSourceInfo {
    std::string dataTreeName;
    std::string mcTreeName;
    std::string absorptionTreeName;
    std::string snapshotDir;
    std::string rawDataPath;
    std::string mcAcceptancePath;
    std::string mcAbsorptionPath;
};

} // namespace UnifiedAnalysis

#endif // DATA_SOURCE_H
