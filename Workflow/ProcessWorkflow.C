// Unified workflow orchestrator: one config schema + one analysis entry.
// Usage:
//   root -l -b -q 'Workflow/ProcessWorkflow.C("configs/general_config.json")'

#include <TSystem.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

namespace {

using Json = GeneralHelper::Json;

std::string EscapeForRootString(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

int RunProcessAnalysis(const std::string &cfgPath, const std::string &modeOverride) {
    std::string invocation = "Tasks/ProcessAnalysis.C(\"" + EscapeForRootString(cfgPath) + "\"";
    if (!modeOverride.empty()) {
        invocation += ", \"" + EscapeForRootString(modeOverride) + "\")";
    } else {
        invocation += ", \"\")";
    }
    const std::string cmd = "root -l -b -q '" + invocation + "'";
    return gSystem->Exec(cmd.c_str());
}

} // namespace

int ProcessWorkflow(const char *workflowConfigPath = "../configs/general_config.json") {
    if (!workflowConfigPath || std::string(workflowConfigPath).empty()) {
        std::cerr << "[ProcessWorkflow] Empty config path" << std::endl;
        return 1;
    }

    try {
        const std::filesystem::path cfgPath = std::filesystem::weakly_canonical(workflowConfigPath);
        const Json general = GeneralHelper::LoadJsonFile(cfgPath.string());

        const auto execution = general.value("execution", Json::object());
        const auto workflow = general.value("workflow", Json::object());
        const bool stopOnError = execution.value("stop_on_error", true);

        std::vector<std::string> modes;
        if (workflow.contains("order") && workflow["order"].is_array() && !workflow["order"].empty()) {
            for (const auto &m : workflow["order"]) {
                if (!m.is_string()) continue;
                modes.push_back(m.get<std::string>());
            }
        }
        if (modes.empty()) {
            modes.push_back(execution.value("mode", std::string("bdt_spectrum")));
        }

        for (const auto &mode : modes) {
            std::cout << "[ProcessWorkflow] Running mode: " << mode << std::endl;
            const int ret = RunProcessAnalysis(cfgPath.string(), mode);
            if (ret != 0) {
                std::cerr << "[ProcessWorkflow] Mode failed: " << mode << ", code=" << ret << std::endl;
                if (stopOnError) return ret;
            }
        }

        std::cout << "[ProcessWorkflow] Completed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[ProcessWorkflow] Exception: " << e.what() << std::endl;
        return 1;
    }
}
