// Helper utilities for BDT-based spectrum extraction.
// Provides bin key/labels and working-point lookup from ProcessWP summaries.

#ifndef BDT_SPECTRUM_HELPER_H
#define BDT_SPECTRUM_HELPER_H

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

struct Config {
    std::string snapshotDir;
    std::string treeNameData{"O2hypcands"};
    std::string treeNameMc{"O2mchypcands"};
    std::string treeNameAbsorption{"h3l_spectrum"};
    std::string wpFile;
    std::string outputDir{"Outputs"};
    std::vector<double> ptBins;
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCen;
    std::string isMatter{"both"};
    std::string bkgFunc{"pol2"};
    std::string sigFunc{"dscb"};
    std::vector<double> sigmaRangeMcToData{1.0, 1.5};
    double branchingRatio{0.25};
    double deltaRap{2.0};
    double massMin{2.96};
    double massMax{3.04};
    bool doSystematics{true};
    std::vector<double> bdtScoreRelShifts{-0.10, 0.0, 0.10};
    std::vector<std::string> bkgFuncSyst{"pol2", "pol1", "expo"};
    std::string nEventsFile;
    std::string nEventsHist;
    std::string mcFileForAcceptance;
    std::string mcFileForAbsorption;
    std::string reweightPtFile;
    std::string basicSelectionDataForMCEff;
    bool enableImplicitMT{false};
    bool do_QA_afterward{false};
    int randomSeed{42};
    int systBdtScoreNPoints{20};
    int systNtrails{800};
    double systThrChi2Ndf{2.0};
    double systThrSignificance{2.5};
    std::string systEfficiencyArrayPath;
    std::vector<std::string> systSignalFuncs;
    std::vector<std::string> systAbsorptionFiles;
    std::vector<std::string> systAbsorptionFileLabels;
};

inline Config LoadConfig(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Config file not found: " + path);
    }
    std::ifstream ifs(path);
    json j; ifs >> j;
    Config cfg;
    auto get_string = [&](const char *key, const std::string &fallback) {
        if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
        return fallback;
    };
    auto get_double = [&](const char *key, double fallback) {
        if (j.contains(key) && j[key].is_number()) return j[key].get<double>();
        return fallback;
    };
    auto get_int = [&](const char *key, int fallback) {
        if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
        if (j.contains(key) && j[key].is_number()) return static_cast<int>(j[key].get<double>());
        return fallback;
    };
    auto get_bool = [&](const char *key, bool fallback) {
        if (j.contains(key) && j[key].is_boolean()) return j[key].get<bool>();
        return fallback;
    };
    auto get_double_vec = [&](const char *key, const std::vector<double> &fallback) {
        if (!j.contains(key) || !j[key].is_array()) return fallback;
        std::vector<double> out;
        for (const auto &v : j[key]) if (v.is_number()) out.push_back(v.get<double>());
        return out.empty() ? fallback : out;
    };
    auto get_string_vec = [&](const char *key, const std::vector<std::string> &fallback) {
        if (!j.contains(key) || !j[key].is_array()) return fallback;
        std::vector<std::string> out;
        for (const auto &v : j[key]) if (v.is_string()) out.push_back(v.get<std::string>());
        return out.empty() ? fallback : out;
    };
    auto get_2d_double_vec = [&](const char *key, const std::vector<std::vector<double>> &fallback) {
        if (!j.contains(key) || !j[key].is_array()) return fallback;
        std::vector<std::vector<double>> out;
        for (const auto &row : j[key]) {
            if (!row.is_array()) continue;
            std::vector<double> r;
            for (const auto &v : row) if (v.is_number()) r.push_back(v.get<double>());
            if (!r.empty()) out.push_back(std::move(r));
        }
        return out.empty() ? fallback : out;
    };
    cfg.snapshotDir = j.at("snapshot_dir").get<std::string>();
    cfg.treeNameData = get_string("tree_name", cfg.treeNameData);
    cfg.treeNameMc = get_string("tree_name_mc", cfg.treeNameMc);
    cfg.treeNameAbsorption = get_string("tree_name_absorption", cfg.treeNameAbsorption);
    cfg.wpFile = j.at("working_point_file").get<std::string>();
    cfg.outputDir = get_string("output_dir", cfg.outputDir);
    cfg.ptBins = get_double_vec("pt_bins", std::vector<double>{});
    cfg.cenBins = get_double_vec("cen_bins", std::vector<double>{});
    cfg.ptBinsByCen = get_2d_double_vec("pt_bins_by_centrality", std::vector<std::vector<double>>{});
    cfg.isMatter = get_string("is_matter", cfg.isMatter);
    cfg.bkgFunc = get_string("bkg_fit_func", cfg.bkgFunc);
    cfg.sigFunc = get_string("signal_fit_func", cfg.sigFunc);
    cfg.sigmaRangeMcToData = get_double_vec("sigma_range_mc_to_data", cfg.sigmaRangeMcToData);
    cfg.branchingRatio = get_double("branching_ratio", cfg.branchingRatio);
    cfg.deltaRap = get_double("delta_rap", cfg.deltaRap);
    cfg.massMin = get_double("mass_min", cfg.massMin);
    cfg.massMax = get_double("mass_max", cfg.massMax);
    cfg.doSystematics = get_bool("do_systematics", cfg.doSystematics);
    cfg.bdtScoreRelShifts = get_double_vec("syst_bdt_score_rel_shifts", cfg.bdtScoreRelShifts);
    cfg.bkgFuncSyst = get_string_vec("syst_bkg_funcs", cfg.bkgFuncSyst);
    cfg.nEventsFile = get_string("analysis_results_file", "");
    cfg.nEventsHist = get_string("n_events_hist", "");
    cfg.mcFileForAcceptance = get_string("mc_file_for_acceptance", "");
    cfg.mcFileForAbsorption = get_string("mc_file_for_absorption", "");
    cfg.reweightPtFile = get_string("reweight_pt_file", "");
    cfg.basicSelectionDataForMCEff = get_string("basic_selection_data_for_mc_eff", "");
    cfg.enableImplicitMT = get_bool("enable_implicit_mt", cfg.enableImplicitMT);
    cfg.do_QA_afterward = get_bool("do_QA_afterward", get_bool("do_QA_afterwords", cfg.do_QA_afterward));
    cfg.randomSeed = get_int("random_seed", cfg.randomSeed);
    cfg.systBdtScoreNPoints = get_int("syst_bdt_score_npoints", cfg.systBdtScoreNPoints);
    cfg.systNtrails = get_int("syst_ntrails", cfg.systNtrails);
    cfg.systThrChi2Ndf = get_double("syst_thrashold_chi2ndf", cfg.systThrChi2Ndf);
    cfg.systThrSignificance = get_double("syst_thrashold_significance", cfg.systThrSignificance);
    cfg.systEfficiencyArrayPath = get_string("syst_efficiency_array_path", cfg.systEfficiencyArrayPath);
    cfg.systSignalFuncs = get_string_vec("syst_signal_funcs", cfg.systSignalFuncs);
    cfg.systAbsorptionFiles = get_string_vec("syst_absorption_files", cfg.systAbsorptionFiles);
    cfg.systAbsorptionFileLabels = get_string_vec("syst_absorption_file_labels", cfg.systAbsorptionFileLabels);
    return cfg;
}

struct BinKey {
    double cenMin{-1.0};
    double cenMax{-1.0};
    double ptMin{0.0};
    double ptMax{0.0};
    double ctMin{-1.0};
    double ctMax{-1.0};

    bool operator<(const BinKey &other) const {
        return std::tie(cenMin, cenMax, ptMin, ptMax, ctMin, ctMax) <
               std::tie(other.cenMin, other.cenMax, other.ptMin, other.ptMax, other.ctMin, other.ctMax);
    }
};

struct WorkingPoint {
    double score{0.0};
    double efficiency{0.0};
    double significance{0.0};
};

inline std::string FormatEdge(double v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << v;
    std::string s = os.str();
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    if (s.empty()) s = "0";
    return s;
}

inline std::string MakeLabel(const BinKey &key) {
    std::string label;
    if (key.cenMin >= 0.0 && key.cenMax >= 0.0) {
        label += "cen_" + FormatEdge(key.cenMin) + "_" + FormatEdge(key.cenMax) + "_";
    }
    if (key.ptMax > key.ptMin) {
        label += "pt_" + FormatEdge(key.ptMin) + "_" + FormatEdge(key.ptMax) + "_";
    }
    if (key.ctMax > key.ctMin && key.ctMin >= 0.0) {
        label += "ct_" + FormatEdge(key.ctMin) + "_" + FormatEdge(key.ctMax) + "_";
    }
    if (!label.empty() && label.back() == '_') label.pop_back();
    if (label.empty()) label = "all";
    return label;
}

#endif // BDT_SPECTRUM_HELPER_H