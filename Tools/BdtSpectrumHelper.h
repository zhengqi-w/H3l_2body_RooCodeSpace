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
    bool enableImplicitMT{false};
};

inline Config LoadConfig(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Config file not found: " + path);
    }
    std::ifstream ifs(path);
    json j; ifs >> j;
    Config cfg;
    cfg.snapshotDir = j.at("snapshot_dir").get<std::string>();
    cfg.treeNameData = j.value("tree_name", cfg.treeNameData);
    cfg.treeNameMc = j.value("tree_name_mc", cfg.treeNameMc);
    cfg.treeNameAbsorption = j.value("tree_name_absorption", cfg.treeNameAbsorption);
    cfg.wpFile = j.at("working_point_file").get<std::string>();
    cfg.outputDir = j.value("output_dir", cfg.outputDir);
    cfg.ptBins = j.value("pt_bins", std::vector<double>{});
    cfg.cenBins = j.value("cen_bins", std::vector<double>{});
    cfg.ptBinsByCen = j.value("pt_bins_by_centrality", std::vector<std::vector<double>>{});
    cfg.isMatter = j.value("is_matter", cfg.isMatter);
    cfg.bkgFunc = j.value("bkg_fit_func", cfg.bkgFunc);
    cfg.sigFunc = j.value("signal_fit_func", cfg.sigFunc);
    cfg.sigmaRangeMcToData = j.value("sigma_range_mc_to_data", cfg.sigmaRangeMcToData);
    cfg.branchingRatio = j.value("branching_ratio", cfg.branchingRatio);
    cfg.deltaRap = j.value("delta_rap", cfg.deltaRap);
    cfg.massMin = j.value("mass_min", cfg.massMin);
    cfg.massMax = j.value("mass_max", cfg.massMax);
    cfg.doSystematics = j.value("do_systematics", cfg.doSystematics);
    cfg.bdtScoreRelShifts = j.value("syst_bdt_score_rel_shifts", cfg.bdtScoreRelShifts);
    cfg.bkgFuncSyst = j.value("syst_bkg_funcs", cfg.bkgFuncSyst);
    cfg.nEventsFile = j.value("analysis_results_file", "");
    cfg.nEventsHist = j.value("n_events_hist", "");
    cfg.mcFileForAcceptance = j.value("mc_file_for_acceptance", "");
    cfg.mcFileForAbsorption = j.value("mc_file_for_absorption", "");
    cfg.reweightPtFile = j.value("reweight_pt_file", "");
    cfg.enableImplicitMT = j.value("enable_implicit_mt", cfg.enableImplicitMT);
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

enum class WPFormat {
    Full,   // cen pt ct
    CenPt,  // cen pt
    PtCt    // pt ct
};

inline bool CloseEnough(double a, double b, double tol = 1e-6) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::abs(a + 1.0) < tol && std::abs(b + 1.0) < tol) return true; // sentinel -1 pair
    return std::abs(a - b) < tol;
}

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

class WPSummaryReader {
public:
    explicit WPSummaryReader(const std::string &path, double tol = 1e-6) : fPath(path), fTol(tol) {
        Load();
    }

    WorkingPoint Lookup(const BinKey &key) const {
        for (const auto &[k, wp] : fMap) {
            if (CloseEnough(k.cenMin, key.cenMin, fTol) && CloseEnough(k.cenMax, key.cenMax, fTol) &&
                CloseEnough(k.ptMin, key.ptMin, fTol) && CloseEnough(k.ptMax, key.ptMax, fTol) &&
                CloseEnough(k.ctMin, key.ctMin, fTol) && CloseEnough(k.ctMax, key.ctMax, fTol)) {
                return wp;
            }
        }
        throw std::runtime_error("Working point not found for label: " + MakeLabel(key));
    }

    std::vector<BinKey> Keys() const {
        std::vector<BinKey> out;
        out.reserve(fMap.size());
        for (const auto &kv : fMap) out.push_back(kv.first);
        return out;
    }

private:
    void Load() {
        namespace fs = std::filesystem;
        if (!fs::exists(fPath)) {
            throw std::runtime_error("WP file not found: " + fPath);
        }
        std::ifstream ifs(fPath);
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                ParseFormatHint(line);
                continue;
            }
            std::istringstream ss(line);
            std::vector<double> vals;
            double v;
            while (ss >> v) vals.push_back(v);
            if (vals.empty()) continue;

            if (vals.size() >= 9) {
                BinKey key{vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]};
                fMap[key] = WorkingPoint{vals[6], vals[7], vals[8]};
                continue;
            }

            // 4 boundaries + 3 numbers (score/eff/sig)
            if (vals.size() >= 7) {
                if (format_ == WPFormat::PtCt) {
                    double cenMin = -1.0, cenMax = -1.0;
                    double ptMin = vals[0], ptMax = vals[1];
                    double ctMin = vals[2], ctMax = vals[3];
                    fMap[{cenMin, cenMax, ptMin, ptMax, ctMin, ctMax}] = WorkingPoint{vals[4], vals[5], vals[6]};
                } else { // CenPt default
                    double cenMin = vals[0], cenMax = vals[1];
                    double ptMin = vals[2], ptMax = vals[3];
                    double ctMin = -1.0, ctMax = -1.0;
                    fMap[{cenMin, cenMax, ptMin, ptMax, ctMin, ctMax}] = WorkingPoint{vals[4], vals[5], vals[6]};
                }
            }
        }
    }

    void ParseFormatHint(const std::string &line) {
        if (line.find("ptmin ptmax ctmin ctmax") != std::string::npos) {
            format_ = WPFormat::PtCt;
        } else if (line.find("cenmin cenmax ptmin ptmax") != std::string::npos) {
            format_ = WPFormat::CenPt;
        } else if (line.find("ctmin ctmax") != std::string::npos) {
            format_ = WPFormat::Full;
        }
    }

    std::string fPath;
    double fTol{1e-6};
    std::map<BinKey, WorkingPoint> fMap;
    WPFormat format_{WPFormat::Full};
};

#endif // BDT_SPECTRUM_HELPER_H