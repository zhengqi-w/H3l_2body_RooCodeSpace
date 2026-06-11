#include <TCanvas.h>
#include <TSystem.h>

R__LOAD_LIBRARY(libHist)
R__LOAD_LIBRARY(libGraf)
R__LOAD_LIBRARY(libGpad)

#include "../Tools/AnalysisEngine.h"
#include "../Tools/AnalysisEngine.cxx"
#include "../Tools/binning/BinPlanBuilder.cxx"
#include "../Tools/tasks/UnifiedTaskRunner.cxx"

#include <exception>
#include <iostream>
#include <string>

int ProcessAnalysis(const char *configPath = "../configs/general_config.json",
                    const char *modeOverride = "") {
    if (!configPath || std::string(configPath).empty()) {
        std::cerr << "[ProcessAnalysis] config path is empty" << std::endl;
        return 1;
    }

    try {
        UnifiedAnalysis::AnalysisEngine engine;
        return engine.Run(configPath, modeOverride ? std::string(modeOverride) : std::string());
    } catch (const std::exception &ex) {
        std::cerr << "[ProcessAnalysis] Error: " << ex.what() << std::endl;
        return 1;
    }
}
