// Unified workflow orchestrator: single entry + task registry + single config tree.
// Usage:
//   root -l -b -q 'Workflow/ProcessWorkflow.C("configs/general_config.json")'

#include <TROOT.h>
#include <TSystem.h>

#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
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

std::filesystem::path ResolvePath(const std::filesystem::path &baseDir, const std::string &p) {
    std::filesystem::path path(p);
    if (path.is_absolute()) return path;
    std::error_code ec;
    auto full = baseDir / path;
    auto canon = std::filesystem::weakly_canonical(full, ec);
    return ec ? full : canon;
}

std::string GetString(const Json &j, const std::string &key, const std::string &def = "") {
    return (j.contains(key) && j[key].is_string()) ? j[key].get<std::string>() : def;
}

double GetNumber(const Json &j, const std::string &key, double def = 0.0) {
    return (j.contains(key) && j[key].is_number()) ? j[key].get<double>() : def;
}

Json BuildBdtConfigFromGeneral(const Json &g) {
    const Json jc = g.value("common", Json::object());
    const Json path = jc.value("path", Json::object());
    const Json tree = jc.value("tree_names", Json::object());
    const Json tags = jc.value("tags", Json::object());
    const Json pars = jc.value("parameters", Json::object());
    const Json oth = jc.value("others", Json::object());
    const Json jt = g.value("bdt_spectrum", Json::object());
    const Json jb = jt.value("binnings", Json::object());
    const Json jsw = jt.value("switches", Json::object());
    const Json jsy = jt.value("systematics", Json::object());

    const std::string period = GetString(tags, "period", "Run3");
    const std::string mark = GetString(tags, "period_mark", "default");
    const std::string outBase = GetString(path, "output_dir", "./Outputs") + "/" + period + "_" + mark;
    const std::string wpDir = GetString(path, "working_point_dir", "");

    Json out = Json::object();
    out["snapshot_dir"] = GetString(path, "input_snapshot_dir", "");
    out["analysis_results_file"] = GetString(path, "analysis_results_file", "");
    out["mc_file_for_acceptance"] = GetString(path, "mc_file_for_acceptance", "");
    out["mc_file_for_absorption"] = GetString(path, "mc_file_for_absorption", "");
    std::string wpSpectrum;
    if (!wpDir.empty()) {
        const std::filesystem::path wpTest = std::filesystem::path(wpDir) / "WorkingPoint_SpectrumTest.txt";
        const std::filesystem::path wpAllCent = std::filesystem::path(wpDir) / "WorkingPoint_Spectrum_AllCent.txt";
        if (std::filesystem::exists(wpTest)) {
            wpSpectrum = wpTest.string();
        } else {
            wpSpectrum = wpAllCent.string();
        }
    }
    out["working_point_file"] = wpSpectrum;
    out["reweight_pt_file"] = GetString(path, "reweight_pt_file", "");
    out["tree_name"] = GetString(tree, "data", "O2hypcands");
    out["tree_name_mc"] = GetString(tree, "mc", "O2mchypcands");
    out["tree_name_absorption"] = GetString(tree, "absorption", "he3candidates");
    out["output_dir"] = outBase + "/BdtSpectrum";
    out["cen_bins"] = jb.value("cen_bins", Json::array());
    out["pt_bins_by_centrality"] = jb.value("pt_bins_by_centrality", Json::array());
    out["basic_selection_data_for_mc_eff"] = GetString(oth, "basic_selection_data_for_mc_eff", "");
    out["is_matter"] = GetString(jsw, "is_matter", "both");
    out["bkg_fit_func"] = GetString(oth, "bkg_fit_func", "pol2");
    out["signal_fit_func"] = GetString(oth, "signal_fit_func", "dscb");
    out["sigma_range_mc_to_data"] = oth.value("sigma_range_mc_to_data", Json::array({1.0, 1.5}));
    out["branching_ratio"] = GetNumber(pars, "branching_ratio", 0.25);
    out["delta_rap"] = GetNumber(pars, "delta_rap", 2.0);
    out["mass_min"] = GetNumber(pars, "mass_min", 2.96);
    out["mass_max"] = GetNumber(pars, "mass_max", 3.04);
    out["n_events_hist"] = GetString(oth, "n_events_hist", "");
    out["enable_implicit_mt"] = oth.value("enable_implicit_mt", true);
    out["do_QA_afterward"] = jsw.value("do_QA_afterward", true);
    out["do_systematics"] = jsw.value("do_systematics", false);
    out["random_seed"] = jsy.value("random_seed", 42);
    out["syst_ntrails"] = jsy.value("syst_ntrails", 200);
    out["syst_thrashold_chi2ndf"] = jsy.value("syst_thrashold_chi2ndf", 2.0);
    out["syst_thrashold_significance"] = jsy.value("syst_thrashold_significance", 2.5);
    out["syst_bdt_score_npoints"] = jsy.value("syst_bdt_score_npoints", 30);
    out["syst_efficiency_array_path"] = jsy.value("syst_efficiency_array_path", "");
    out["syst_bkg_funcs"] = jsy.value("syst_bkg_funcs", Json::array());
    out["syst_signal_funcs"] = jsy.value("syst_signal_funcs", Json::array());
    out["syst_absorption_files"] = jsy.value("syst_absorption_files", Json::array());
    out["syst_absorption_file_labels"] = jsy.value("syst_absorption_file_labels", Json::array());
    return out;
}

Json BuildTopologyConfigFromGeneral(const Json &g) {
    Json out = BuildBdtConfigFromGeneral(g);
    const Json jt = g.value("topology_spectrum", Json::object());
    const Json jb = jt.value("binnings", Json::object());
    const Json jsw = jt.value("switches", Json::object());
    const Json jsel = jt.value("selections", Json::object());
    const Json jc = g.value("common", Json::object());
    const Json path = jc.value("path", Json::object());
    const Json tags = jc.value("tags", Json::object());
    const std::string period = GetString(tags, "period", "Run3");
    const std::string mark = GetString(tags, "period_mark", "default");
    const std::string outBase = GetString(path, "output_dir", "./Outputs") + "/" + period + "_" + mark;

    out["output_dir"] = outBase + "/TopologySpectrum";
    out["cen_bins"] = jb.value("cen_bins", Json::array());
    out["pt_bins_by_centrality"] = jb.value("pt_bins_by_centrality", Json::array());
    out["is_matter"] = GetString(jsw, "is_matter", "both");
    out["do_QA_afterward"] = jsw.value("do_QA_afterward", true);
    out["do_systematics"] = jsw.value("do_systematics", false);
    out["data_selection_topology"] = jsel.value("data_selection_topology", Json::array());
    return out;
}

Json BuildCtExtractionConfigFromGeneral(const Json &g) {
    const Json jc = g.value("common", Json::object());
    const Json path = jc.value("path", Json::object());
    const Json tree = jc.value("tree_names", Json::object());
    const Json tags = jc.value("tags", Json::object());
    const Json pars = jc.value("parameters", Json::object());
    const Json oth = jc.value("others", Json::object());
    const Json jt = g.value("ct_extraction", Json::object());
    const Json jb = jt.value("binnings", Json::object());
    const Json jsw = jt.value("switches", Json::object());

    const std::string period = GetString(tags, "period", "Run3");
    const std::string mark = GetString(tags, "period_mark", "default");
    const std::string outBase = GetString(path, "output_dir", "./Outputs") + "/" + period + "_" + mark;
    const std::string wpDir = GetString(path, "working_point_dir", "");
    Json ptBins = jb.value("pt_bins", Json::array());
    Json ctBinsByPt = jb.value("ct_bins_by_pt", Json::array());
    Json sigmaRanges = Json::array();
    const Json sigmaSingle = oth.value("sigma_range_mc_to_data", Json::array({1.0, 1.5}));
    const size_t nPtBins = ptBins.size() > 0 ? ptBins.size() - 1 : 0;
    for (size_t i = 0; i < nPtBins; ++i) sigmaRanges.push_back(sigmaSingle);

    Json out = Json::object();
    out["data_snapshot_dir"] = GetString(path, "input_snapshot_dir", "");
    out["snapshot_tree_name"] = GetString(tree, "data", "O2hypcands");
    out["mc_file"] = GetString(path, "mc_file_for_acceptance", "");
    out["mc_tree_name"] = GetString(tree, "mc", "O2mchypcands");
    out["mc_snapshot_dir"] = GetString(path, "input_snapshot_dir", "");
    out["mc_snapshot_tree_name"] = GetString(tree, "mc", "O2mchypcands");
    out["mc_snapshot_pattern"] = "mc_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root";
    out["mc_reweight_file"] = GetString(path, "reweight_pt_file", "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root");
    out["mc_reweight_func"] = "BlastWave_H3L_10_30";
    out["working_point_file"] = wpDir.empty() ? "" : (wpDir + "/WorkingPoint_Crosssection_CustomV0s.txt");
    out["output_dir"] = outBase + "/CtExtraction";
    out["output_file"] = "ct_analysis";
    out["trial_suffix"] = "";
    out["is_matter"] = GetString(jsw, "is_matter", "both");
    out["mass_column"] = "fMassH3L";
    out["bdt_score_column"] = "model_output";
    out["run_period_label"] = period;
    out["colliding_system"] = GetString(tags, "collision_system", "Pb--Pb");
    out["sqrtsnn_label"] = "#sqrt{s_{NN}}";
    out["data_set_label"] = mark;
    out["alice_performance"] = false;
    out["collision_energy_tev"] = 5.36;
    out["features"] = Json::array({"fDcaV0Daug", "fDcaHe", "fDcaPi", "fCosPA", "fNSigmaHe"});
    out["basic_selection_data_for_mc_eff"] = GetString(oth, "basic_selection_data_for_mc_eff", "");
    out["pt_bins"] = ptBins;
    out["ct_bins"] = ctBinsByPt;
    out["mass_range"] = Json::array({GetNumber(pars, "mass_min", 2.96), GetNumber(pars, "mass_max", 3.04)});
    out["mass_nbins_mc"] = oth.value("mass_nbins_mc", 80);
    out["mass_nbins_data"] = oth.value("mass_nbins_data", 50);
    out["min_entries_for_fit"] = 10;
    out["bdt_score_shift"] = 0.0;
    out["snapshot_pattern"] = "data_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root";
    out["sigma_mc_to_data_range"] = sigmaRanges;
    out["bdt_overrides"] = Json::array();
    return out;
}

using Runner = std::function<int(const std::filesystem::path &, const std::filesystem::path &, const Json &, const Json &)>;

int RunRootTask(const std::filesystem::path &macroPath,
                const std::string &argsExpr) {
    const std::string invocation = macroPath.string() + "(" + argsExpr + ")";
    std::string rootExe = "root";
    const char *envRootExe = gSystem->Getenv("ROOT_EXECUTABLE");
    if (envRootExe && std::string(envRootExe).size() > 0) {
        rootExe = envRootExe;
    } else {
        rootExe = "/opt/anaconda3/envs/MLenv/bin/root";
    }
    const std::string cmd = rootExe + " -q -b -l '" + invocation + "'";
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
        const std::filesystem::path cfgDir = cfgPath.parent_path();
        const std::filesystem::path codeSpaceDir = cfgDir.parent_path();

        const Json general = GeneralHelper::LoadJsonFile(cfgPath.string());
        const Json workflowNode = general.value("workflow", Json::object());
        const bool stopOnError = workflowNode.value("stop_on_error", true);

        std::filesystem::path generatedDir = codeSpaceDir / "Workflow" / "generated_configs";
        if (workflowNode.contains("generated_config_dir") && workflowNode["generated_config_dir"].is_string()) {
            generatedDir = ResolvePath(cfgDir, workflowNode["generated_config_dir"].get<std::string>());
        }
        GeneralHelper::EnsureDir(generatedDir.string());

        std::filesystem::path taskDir = codeSpaceDir / "Tasks";
        if (workflowNode.contains("tasks_dir") && workflowNode["tasks_dir"].is_string()) {
            taskDir = ResolvePath(cfgDir, workflowNode["tasks_dir"].get<std::string>());
        }

        std::vector<std::string> order{"bdt_spectrum", "topology_spectrum", "ct_extraction", "ct_single"};
        if (workflowNode.contains("order")) {
            order = workflowNode["order"].get<std::vector<std::string>>();
        }

        auto isEnabled = [&](const std::string &taskKey) {
            if (!general.contains(taskKey)) return false;
            const auto sec = general[taskKey];
            if (sec.contains("enabled") && sec["enabled"].is_boolean()) return sec["enabled"].get<bool>();
            return true;
        };

        std::map<std::string, Runner> registry;
        registry["bdt_spectrum"] = [](const std::filesystem::path &taskDir,
                                       const std::filesystem::path &cfgOut,
                                       const Json &,
                                       const Json &) {
            const auto macro = taskDir / "ProcessBdtSpectrum.C";
            const std::string args = "\"" + EscapeForRootString(cfgOut.string()) + "\"";
            return RunRootTask(macro, args);
        };

        registry["topology_spectrum"] = [](const std::filesystem::path &taskDir,
                                            const std::filesystem::path &cfgOut,
                                            const Json &,
                                            const Json &) {
            const auto macro = taskDir / "ProcessTopologySpectrum.C";
            const std::string args = "\"" + EscapeForRootString(cfgOut.string()) + "\"";
            return RunRootTask(macro, args);
        };

        registry["ct_extraction"] = [](const std::filesystem::path &taskDir,
                                        const std::filesystem::path &cfgOut,
                                        const Json &,
                                        const Json &general) {
            const auto macro = taskDir / "ProcessCtSpectrum.C";
            bool mt = true;
            if (general.contains("common") && general["common"].contains("others")) {
                mt = general["common"]["others"].value("enable_implicit_mt", true);
            }
            const std::string args = "\"" + EscapeForRootString(cfgOut.string()) + "\", " +
                                     std::string(mt ? "true" : "false");
            return RunRootTask(macro, args);
        };

        registry["ct_single"] = [](const std::filesystem::path &taskDir,
                                    const std::filesystem::path &,
                                    const Json &,
                                    const Json &general) {
            const auto macro = taskDir / "ProcessCtSingleSpectrum.C";
            const std::string cfg = EscapeForRootString(general.value("_config_path", std::string("")));
            const std::string args = "\"" + cfg + "\"";
            return RunRootTask(macro, args);
        };

        for (const auto &taskKey : order) {
            if (!isEnabled(taskKey)) {
                std::cout << "[ProcessWorkflow] Skip disabled task: " << taskKey << std::endl;
                continue;
            }
            if (!registry.count(taskKey)) {
                std::cerr << "[ProcessWorkflow] Unregistered task: " << taskKey << std::endl;
                if (stopOnError) return 2;
                continue;
            }

            Json effectiveCfg = Json::object();
            std::filesystem::path cfgOut;
            if (taskKey == "bdt_spectrum") {
                effectiveCfg = BuildBdtConfigFromGeneral(general);
                cfgOut = generatedDir / "bdt_spectrum_effective.json";
                GeneralHelper::SaveJsonFile(cfgOut.string(), effectiveCfg, 2);
            } else if (taskKey == "topology_spectrum") {
                effectiveCfg = BuildTopologyConfigFromGeneral(general);
                cfgOut = generatedDir / "topology_spectrum_effective.json";
                GeneralHelper::SaveJsonFile(cfgOut.string(), effectiveCfg, 2);
            } else if (taskKey == "ct_extraction") {
                effectiveCfg = BuildCtExtractionConfigFromGeneral(general);
                cfgOut = generatedDir / "ct_extraction_effective.json";
                GeneralHelper::SaveJsonFile(cfgOut.string(), effectiveCfg, 2);
            }

            Json runtimeGeneral = general;
            runtimeGeneral["_config_path"] = cfgPath.string();
            std::cout << "[ProcessWorkflow] Running task " << taskKey;
            if (!cfgOut.empty()) std::cout << " with config " << cfgOut;
            std::cout << std::endl;
            int ret = registry[taskKey](taskDir, cfgOut, general.value(taskKey, Json::object()), runtimeGeneral);
            if (ret != 0) {
                std::cerr << "[ProcessWorkflow] Task failed: " << taskKey << ", code=" << ret << std::endl;
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
