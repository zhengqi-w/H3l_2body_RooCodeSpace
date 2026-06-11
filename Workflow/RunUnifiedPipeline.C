// Unified pipeline entry macro.
// Usage:
//   root -l -b -q 'Workflow/RunUnifiedPipeline.C("configs/general_config.json", "all")'

#include <TSystem.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

std::string NormalizeMode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::replace(mode.begin(), mode.end(), '-', '_');
    if (mode == "ptct") return "pt_ct";
    if (mode == "cenpt") return "cen_pt";
    if (mode == "ptctsingle") return "pt_ct_single";
    if (mode == "ptsingle") return "pt_single";
    if (mode == "ctsingle") return "ct_single";
    return mode;
}

bool IsAllowedMode(const std::string &mode) {
    static const std::unordered_set<std::string> kAllowed = {
        "pt_ct", "cen_pt", "pt_ct_single", "pt_single", "ct_single", "bdt_spectrum", "topology_spectrum"
    };
    return kAllowed.find(mode) != kAllowed.end();
}

std::string PreprocessModeForAnalysisMode(const std::string &mode) {
    if (mode == "bdt_spectrum") return "cen_pt";
    return mode;
}

std::string EscapeForShellSingleQuoted(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

void WriteStageManifest(const std::string &stage,
                        const std::string &configPath,
                        const std::string &trainingMode,
                        const std::string &wpMode,
                        const std::string &analysisMode,
                        const std::string &command,
                        int exitCode,
                        bool dryRun) {
    Json manifest = Json::object();
    manifest["stage"] = stage;
    manifest["config"] = configPath;
    manifest["training_mode"] = trainingMode;
    manifest["wp_mode"] = wpMode;
    manifest["analysis_mode"] = analysisMode;
    manifest["command"] = command;
    manifest["exit_code"] = exitCode;
    manifest["dry_run"] = dryRun;
    manifest["timestamp_unix"] = static_cast<long long>(std::time(nullptr));

    const std::filesystem::path outDir = "Workflow/manifests";
    const std::filesystem::path outPath = outDir / ("run_manifest_" + stage + ".json");
    GeneralHelper::SaveJsonFile(outPath.string(), manifest, 2);
}

int RunStage(const std::string &stage,
             const std::string &command,
             const std::string &configPath,
             const std::string &trainingMode,
             const std::string &wpMode,
             const std::string &analysisMode,
             bool dryRun) {
    std::cout << "[RunUnifiedPipeline] stage=" << stage << std::endl;
    std::cout << "[RunUnifiedPipeline] cmd=" << command << std::endl;

    int code = 0;
    if (!dryRun) {
        code = gSystem->Exec(command.c_str());
    }

    WriteStageManifest(stage, configPath, trainingMode, wpMode, analysisMode, command, code, dryRun);
    if (code != 0) {
        std::cerr << "[RunUnifiedPipeline] stage failed: " << stage << ", code=" << code << std::endl;
        return code;
    }
    return 0;
}

} // namespace

int RunUnifiedPipeline(const char *configPath = "configs/general_config.json",
                       const char *stage = "all",
                       bool dryRun = false) {
    if (!configPath || std::string(configPath).empty()) {
        std::cerr << "[RunUnifiedPipeline] Empty config path" << std::endl;
        return 1;
    }

    try {
        const std::filesystem::path defaultCfgPath = std::filesystem::weakly_canonical("configs/general_config.json");
        const std::filesystem::path cfgPath = std::filesystem::weakly_canonical(configPath);
        if (cfgPath != defaultCfgPath) {
            std::cerr << "[RunUnifiedPipeline] Info: default config is configs/general_config.json;"
                      << " using config: " << cfgPath << std::endl;
        }
        const Json config = GeneralHelper::LoadJsonFile(cfgPath.string());
        const auto execution = config.value("execution", Json::object());

        const std::string trainingMode = NormalizeMode(
            execution.value("training_mode", std::string("ct_single")));
        const std::string wpMode = NormalizeMode(
            execution.value("wp_mode", std::string("ct_single")));
        const std::string analysisMode = NormalizeMode(
            execution.value("analysis_mode", std::string("ct_single")));

        if (!IsAllowedMode(trainingMode) || !IsAllowedMode(wpMode) || !IsAllowedMode(analysisMode)) {
            throw std::runtime_error("Unsupported mode in execution.{training_mode,wp_mode,analysis_mode}");
        }
        const std::string trainingPreprocessMode = PreprocessModeForAnalysisMode(trainingMode);
        const std::string wpPreprocessMode = PreprocessModeForAnalysisMode(wpMode);

        std::string stageOpt = stage ? std::string(stage) : std::string("all");
        stageOpt = NormalizeMode(stageOpt);

        const std::string pythonCmd =
            "python3 PreProcess/BDTPreProcess.py --config-file " + EscapeForShellSingleQuoted(cfgPath.string()) +
            " --mix-mode " + EscapeForShellSingleQuoted(trainingPreprocessMode);
        const std::string wpCmd =
            "root -l -b -q 'PreProcess/ProcessWP.C(\"" + EscapeForRootString(cfgPath.string()) + "\", \"" +
            EscapeForRootString(wpPreprocessMode) + "\")'";
        const std::string analysisCmd =
            "root -l -b -q 'Tasks/ProcessAnalysis.C(\"" + EscapeForRootString(cfgPath.string()) + "\", \"" +
            EscapeForRootString(analysisMode) + "\")'";

        std::vector<std::pair<std::string, std::string>> stages;
        if (stageOpt == "all") {
            stages = {
                {"train", pythonCmd},
                {"wp", wpCmd},
                {"analysis", analysisCmd},
            };
        } else if (stageOpt == "train") {
            stages = {{"train", pythonCmd}};
        } else if (stageOpt == "wp") {
            stages = {{"wp", wpCmd}};
        } else if (stageOpt == "analysis") {
            stages = {{"analysis", analysisCmd}};
        } else {
            std::cerr << "[RunUnifiedPipeline] Unsupported stage: " << stageOpt << std::endl;
            return 2;
        }

        for (const auto &item : stages) {
            const int code = RunStage(item.first, item.second, cfgPath.string(), trainingMode, wpMode, analysisMode, dryRun);
            if (code != 0) {
                return code;
            }
        }

        std::cout << "[RunUnifiedPipeline] Completed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[RunUnifiedPipeline] Exception: " << e.what() << std::endl;
        return 1;
    }
}
