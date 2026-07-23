#include "AnalysisEngine.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <vector>

namespace UnifiedAnalysis {

namespace {

std::string SanitizeCheckTag(std::string tag) {
    for (auto &c : tag) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') c = '_';
    }
    return tag;
}

std::string BuildChecksPeriodTag(const GeneralHelper::Json &cfg) {
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    if (execution.value("combine_period", false) && periods.is_array() && !periods.empty()) {
        std::string out;
        for (size_t ip = 0; ip < periods.size(); ++ip) {
            const auto &period = periods[ip];
            std::string tag = "period_" + std::to_string(ip);
            if (period.is_object()) tag = period.value("tag", tag);
            tag = SanitizeCheckTag(tag);
            if (tag.empty()) continue;
            if (!out.empty()) out += "_";
            out += tag;
        }
        return out.empty() ? "combined_period" : out;
    }
    const auto tags = common.value("tags", GeneralHelper::Json::object());
    return tags.value("period", std::string("period")) + "_" +
           tags.value("period_mark", std::string("mark"));
}

std::vector<std::string> CollectPeriodPaths(const GeneralHelper::Json &cfg,
                                            const std::string &key,
                                            const std::string &fallback) {
    std::vector<std::string> out;
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    if (execution.value("combine_period", false) && periods.is_array() && !periods.empty()) {
        for (const auto &period : periods) {
            if (!period.is_object()) continue;
            const std::string path = period.value(key, std::string());
            if (!path.empty()) out.push_back(path);
        }
    }
    if (out.empty() && !fallback.empty()) out.push_back(fallback);
    return out;
}

std::string MergeCheckSelection(const std::string &base, const std::string &extra) {
    if (extra.empty()) return base;
    if (base.empty()) return extra;
    return "(" + base + ") && (" + extra + ")";
}

std::string MatterSelectionForChecks(const std::string &isMatter) {
    if (isMatter == "matter") return "fIsMatter > 0";
    if (isMatter == "antimatter") return "fIsMatter <= 0";
    return "";
}

} // namespace

int AnalysisEngine::Run(const std::string &configPath, const std::string &modeOverride) const {
    const auto cfg = GeneralHelper::LoadJsonFile(configPath);
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    const bool enableMainMode = execution.value("enable", true);
    const bool enableImplicitMT = execution.value("enable_implicit_mt", false);
    if (enableImplicitMT) {
        GeneralHelper::EnableImplicitMTWithPreferredThreads();
    }

    const std::string mode = ResolveMode(cfg, modeOverride);
    const ModePolicy policy = GetModePolicy(mode);

    BinPlan plan;
    plan.mode = policy.mode;
    if (enableMainMode) {
        BinPlanBuilder builder;
        plan = builder.Build(cfg, policy);
    }

    std::cout << "[AnalysisEngine] mode=" << policy.mode << ", bins=" << plan.items.size();
    if (!enableMainMode) {
        std::cout << " [main mode disabled: checks-only]";
    }
    std::cout << std::endl;

    OutputWriter writer(cfg.value("common", GeneralHelper::Json::object())
                            .value("path", GeneralHelper::Json::object())
                            .value("output_dir", std::string("./Outputs")));
    writer.WriteRunSummary(policy.mode, plan.items.size());

    const ChecksConfig checksCfg = BuildChecksConfig(cfg);
    ChecksEngine checks;
    checks.Run(checksCfg, plan);

    if (!enableMainMode) {
        return 0;
    }

    UnifiedTaskRunner runner;
    return runner.Run(cfg, policy, plan);
}

std::string AnalysisEngine::ResolveMode(const GeneralHelper::Json &cfg, const std::string &overrideMode) {
    if (!overrideMode.empty()) return overrideMode;
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    return execution.value("analysis_mode", std::string("bdt_spectrum"));
}

ChecksConfig AnalysisEngine::BuildChecksConfig(const GeneralHelper::Json &cfg) {
    const auto jc = cfg.value("checks", GeneralHelper::Json::object());
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto trees = common.value("tree_names", GeneralHelper::Json::object());
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto selection = analysis.value("selection", GeneralHelper::Json::object());
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());

    auto parsePairs = [](const GeneralHelper::Json &arr) {
        std::vector<Hist2DPair> pairs;
        if (!arr.is_array()) return pairs;
        for (const auto &row : arr) {
            if (!row.is_array() || row.size() < 2) continue;
            if (!row[0].is_string() || !row[1].is_string()) continue;
            pairs.push_back(Hist2DPair{row[0].get<std::string>(), row[1].get<std::string>()});
        }
        return pairs;
    };

    auto parseBlock = [&](const GeneralHelper::Json &b) {
        CheckBlockConfig out;
        out.enable = b.value("enable", false);
        out.file = b.value("file", std::string());
        out.tree = b.value("tree", std::string());
        out.selection = b.value("selection", std::string());
        if (b.contains("variables") && b["variables"].is_array()) {
            out.variables = b["variables"].get<std::vector<std::string>>();
        }
        out.hist2dPairs = parsePairs(b.value("hist2d_pairs", GeneralHelper::Json::array()));
        return out;
    };

    ChecksConfig out;
    out.enabled = jc.value("enabled", true);
    out.savePdf = jc.value("save_pdf", false);

    const std::string outputDir = path.value("output_dir", std::string("./Outputs"));
    const std::string periodTag = BuildChecksPeriodTag(cfg);
    const std::string mode = execution.value("analysis_mode", std::string("bdt_spectrum"));
    const std::string isMatter = selection.value("is_matter", std::string("both"));
    out.outputRootFile = (std::filesystem::path(outputDir) / periodTag / mode / isMatter / "Checks" / "checks.root").string();

    const auto general = jc.value("general", GeneralHelper::Json::object());
    const auto axisPool = general.value("axis_pool", GeneralHelper::Json::object());
    if (axisPool.is_object()) {
        for (auto it = axisPool.begin(); it != axisPool.end(); ++it) {
            if (!it.value().is_object()) continue;
            AxisSpec ax;
            ax.nBins = it.value().value("nbins", 80);
            ax.min = it.value().value("min", 0.0);
            ax.max = it.value().value("max", 1.0);
            ax.title = it.value().value("title", it.key());
            out.axisPool[it.key()] = ax;
        }
    }

    out.mcChecks = parseBlock(jc.value("mc_checks", GeneralHelper::Json::object()));
    if (out.mcChecks.file.empty()) out.mcChecks.file = path.value("mc_path", std::string());
    out.mcChecks.files = CollectPeriodPaths(cfg, "mc_path", out.mcChecks.file);
    if (out.mcChecks.tree.empty()) out.mcChecks.tree = trees.value("mc", std::string("O2mchypcands"));

    out.dataAllChecks = parseBlock(jc.value("data_all_checks", GeneralHelper::Json::object()));
    out.dataAllChecks.file = path.value("data_path", out.dataAllChecks.file);
    out.dataAllChecks.files = CollectPeriodPaths(cfg, "data_path", out.dataAllChecks.file);
    out.dataAllChecks.tree = trees.value("data", std::string("O2hypcands"));

    const std::string matterSelection = MatterSelectionForChecks(isMatter);
    if (!matterSelection.empty()) {
        out.mcChecks.selection = MergeCheckSelection(out.mcChecks.selection, matterSelection);
        out.dataAllChecks.selection = MergeCheckSelection(out.dataAllChecks.selection, matterSelection);
    }

    out.onTheFlyChecks = parseBlock(jc.value("hypertriton_onthefly_checks", GeneralHelper::Json::object()));
    if (!out.onTheFlyChecks.enable) {
        out.onTheFlyChecks.variables = {};
        out.onTheFlyChecks.hist2dPairs = {};
    }
    return out;
}

} // namespace UnifiedAnalysis
