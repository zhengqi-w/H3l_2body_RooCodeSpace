#include "UnifiedTaskRunner.h"

#include "BinningCorrectionHelper.h"
#include "IntegralYieldHelper.h"
#include "ProcessOneBin.h"
#include "SpectrumPlotHelper.h"
#include "../checks/ChecksConfig.h"

#include <ROOT/RDataFrame.hxx>
#include <RooMsgService.h>
#include <RooPlot.h>
#include <TCanvas.h>
#include <TBox.h>
#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TMath.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TTree.h>

#include "../../include/AliPWGFunc.h"
#include "../../include/AliPWGFunc.cxx"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace UnifiedAnalysis {

namespace {

constexpr double kSpeedOfLightCmPerPs = 0.0299792458;

std::string GetS(const GeneralHelper::Json &j, const char *key, const std::string &def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

double GetD(const GeneralHelper::Json &j, const char *key, double def = 0.0) {
    if (j.contains(key) && j[key].is_number()) return j[key].get<double>();
    return def;
}

int GetI(const GeneralHelper::Json &j, const char *key, int def = 0) {
    if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
    if (j.contains(key) && j[key].is_number()) return static_cast<int>(j[key].get<double>());
    return def;
}

std::string NormalizeEventSignalLossMethod(std::string method) {
    method.erase(std::remove_if(method.begin(), method.end(), [](unsigned char c) {
                     return c == '-' || c == '_' || std::isspace(c);
                 }),
                 method.end());
    std::transform(method.begin(), method.end(), method.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (method == "impact" || method == "impactparameter") return "impactparameter";
    return "multiplicity";
}

std::map<std::string, SpectrumFunctionConfig> ParseSpectrumFitParameterConfigs(const GeneralHelper::Json &j) {
    std::map<std::string, SpectrumFunctionConfig> out;
    if (!j.is_object()) return out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!it.value().is_object()) continue;
        SpectrumFunctionConfig cfg;
        const auto &v = it.value();
        const char *initialKeys[] = {"initial", "init", "parameters_initial"};
        for (const char *key : initialKeys) {
            if (v.contains(key) && v[key].is_array()) {
                cfg.initial = v[key].get<std::vector<double>>();
                break;
            }
        }
        const char *limitKeys[] = {"limits", "ranges", "parameter_limits"};
        for (const char *key : limitKeys) {
            if (!v.contains(key) || !v[key].is_array()) continue;
            for (const auto &row : v[key]) {
                if (!row.is_array() || row.size() < 2) continue;
                if (!row[0].is_number() || !row[1].is_number()) continue;
                cfg.limits.emplace_back(row[0].get<double>(), row[1].get<double>());
            }
            break;
        }
        if (!cfg.initial.empty() || !cfg.limits.empty()) out[it.key()] = std::move(cfg);
    }
    return out;
}

std::string BuildRangeLabel(const std::string &prefix, double v1, double v2) {
    std::ostringstream os;
    os << prefix << '_' << v1 << '_' << v2;
    return os.str();
}

std::string FormatTitleNumber(double v) {
    std::ostringstream os;
    os << std::setprecision(6) << std::defaultfloat << v;
    return os.str();
}

std::string BuildSimpleBinTitle(const BinPlanItem &item) {
    std::vector<std::string> parts;
    if (item.hasCen) {
        parts.push_back("Centrality " + FormatTitleNumber(item.cenMin) + "-" + FormatTitleNumber(item.cenMax) + "%");
    }
    if (item.hasPt) {
        parts.push_back("p_{T} " + FormatTitleNumber(item.ptMin) + "-" + FormatTitleNumber(item.ptMax) + " GeV/c");
    }
    if (item.hasCt) {
        parts.push_back("ct " + FormatTitleNumber(item.ctMin) + "-" + FormatTitleNumber(item.ctMax) + " cm");
    }
    if (parts.empty()) return item.label;

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += " ";
        out += parts[i];
    }
    return out;
}

std::string BuildSimpleGroupTitle(const std::vector<BinPlanItem> &items,
                                  const std::string &fallbackLabel) {
    if (items.empty()) return fallbackLabel;
    if (items.size() == 1) return BuildSimpleBinTitle(items.front());

    const auto &first = items.front();
    bool sameCen = first.hasCen;
    bool samePt = first.hasPt;
    bool sameCt = first.hasCt;
    for (size_t i = 1; i < items.size(); ++i) {
        const auto &it = items[i];
        sameCen = sameCen && it.hasCen && it.cenMin == first.cenMin && it.cenMax == first.cenMax;
        samePt = samePt && it.hasPt && it.ptMin == first.ptMin && it.ptMax == first.ptMax;
        sameCt = sameCt && it.hasCt && it.ctMin == first.ctMin && it.ctMax == first.ctMax;
    }

    BinPlanItem fixed;
    fixed.label = fallbackLabel;
    fixed.hasCen = sameCen;
    fixed.hasPt = samePt;
    fixed.hasCt = sameCt;
    fixed.cenMin = first.cenMin;
    fixed.cenMax = first.cenMax;
    fixed.ptMin = first.ptMin;
    fixed.ptMax = first.ptMax;
    fixed.ctMin = first.ctMin;
    fixed.ctMax = first.ctMax;

    const std::string title = BuildSimpleBinTitle(fixed);
    if (!title.empty() && title != fallbackLabel) return title;
    return BuildSimpleBinTitle(first);
}

std::string NormalizeWpFilename(std::string name) {
    const std::string kPrefix = "WorkingPoint_";
    const std::string kDoublePrefix = kPrefix + kPrefix;
    while (name.rfind(kDoublePrefix, 0) == 0) {
        name.erase(0, kPrefix.size());
    }
    if (!name.empty() && !(name.size() > 4 && name.substr(name.size() - 4) == ".txt")) {
        name += ".txt";
    }
    return name;
}

bool IsCombinePeriodEnabled(const GeneralHelper::Json &cfg) {
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    return execution.value("combine_period", false);
}

std::string SanitizePeriodTag(std::string tag) {
    for (auto &c : tag) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') c = '_';
    }
    return tag;
}

std::string BuildCombinedPeriodTag(const GeneralHelper::Json &cfg) {
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    std::string out;
    if (!periods.is_array()) return "combined_period";
    for (size_t ip = 0; ip < periods.size(); ++ip) {
        const auto &period = periods[ip];
        std::string tag = "period_" + std::to_string(ip);
        if (period.is_object()) tag = GetS(period, "tag", tag);
        tag = SanitizePeriodTag(tag);
        if (tag.empty()) continue;
        if (!out.empty()) out += "_";
        out += tag;
    }
    return out.empty() ? "combined_period" : out;
}

std::string CombinedTopDir(const std::string &path, const std::string &tag) {
    return (std::filesystem::path(path).parent_path() / tag).string();
}

std::string CombinedSubDir(const std::string &path, const std::string &tag) {
    const std::filesystem::path p(path);
    return (p.parent_path().parent_path() / tag / p.filename()).string();
}

std::string ResolveCombinedPath(const GeneralHelper::Json &cfg,
                                const std::string &path,
                                bool isTopDir) {
    if (path.empty() || !IsCombinePeriodEnabled(cfg)) return path;
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    if (!periods.is_array() || periods.empty()) return path;
    const std::string tag = BuildCombinedPeriodTag(cfg);
    return isTopDir ? CombinedTopDir(path, tag) : CombinedSubDir(path, tag);
}

std::string ResolveWpFile(const GeneralHelper::Json &cfg, const std::string &mode) {
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto wpFiles = common.value("wp_files", GeneralHelper::Json::object());
    const std::string wpDir = ResolveCombinedPath(cfg, GetS(path, "wp_dir", ""), false);
    if (wpDir.empty()) return std::string();

    const std::string wpFilename = NormalizeWpFilename(GetS(wpFiles, mode.c_str(), ""));
    if (!wpFilename.empty()) {
        return (std::filesystem::path(wpDir) / wpFilename).string();
    }

    if (mode == "bdt_spectrum") {
        const std::filesystem::path testPath = std::filesystem::path(wpDir) / "WorkingPoint_SpectrumTest.txt";
        const std::filesystem::path allPath = std::filesystem::path(wpDir) / "WorkingPoint_Spectrum_AllCent.txt";
        return std::filesystem::exists(testPath) ? testPath.string() : allPath.string();
    }
    if (mode == "ct_single") {
        return (std::filesystem::path(wpDir) / "WorkingPoint_CtSingle.txt").string();
    }
    if (mode == "pt_ct") {
        return (std::filesystem::path(wpDir) / "WorkingPoint_Crosssection_CustomV0s.txt").string();
    }
    return std::string();
}

std::string ResolveScoreEffDir(const GeneralHelper::Json &cfg, const std::string &mode) {
    (void)mode;
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    return ResolveCombinedPath(cfg, GetS(path, "wp_dir", ""), false);
}

double GetNEvents(const GeneralHelper::Json &cfg, double cenMin, double cenMax) {
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    const auto eventHist = common.value("event_hist", GeneralHelper::Json::object());
    const std::string histPath = GetS(eventHist, "n_events_hist", "");
    if (histPath.empty()) return 0.0;

    std::vector<std::string> files;
    if (IsCombinePeriodEnabled(cfg) && periods.is_array() && !periods.empty()) {
        for (const auto &period : periods) {
            if (!period.is_object()) continue;
            const std::string fp = GetS(period, "analysisresults_path", "");
            if (!fp.empty()) files.push_back(fp);
        }
    }
    if (files.empty()) {
        const std::string filePath = GetS(path, "analysisresults_path", "");
        if (!filePath.empty()) files.push_back(filePath);
    }

    double sum = 0.0;
    for (const auto &filePath : files) {
        TFile f(filePath.c_str(), "READ");
        if (f.IsZombie()) continue;
        TH1 *h = dynamic_cast<TH1 *>(f.Get(histPath.c_str()));
        if (!h) continue;
        const int bmin = h->GetXaxis()->FindBin(cenMin + 1e-3);
        const int bmax = h->GetXaxis()->FindBin(cenMax - 1e-3);
        sum += h->Integral(bmin, bmax);
    }
    return sum;
}

std::vector<std::pair<std::string, double>> GetPeriodEventWeights(const GeneralHelper::Json &cfg,
                                                                  double cenMin,
                                                                  double cenMax) {
    std::vector<std::pair<std::string, double>> out;
    if (!IsCombinePeriodEnabled(cfg)) return out;
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto periods = common.value("periods", GeneralHelper::Json::array());
    const auto eventHist = common.value("event_hist", GeneralHelper::Json::object());
    const std::string histPath = GetS(eventHist, "n_events_hist", "");
    if (!periods.is_array() || periods.empty() || histPath.empty()) return out;

    double total = 0.0;
    for (size_t ip = 0; ip < periods.size(); ++ip) {
        const auto &period = periods[ip];
        if (!period.is_object()) continue;
        const std::string tag = GetS(period, "tag", "period_" + std::to_string(ip));
        const std::string filePath = GetS(period, "analysisresults_path", "");
        double count = 0.0;
        if (!filePath.empty()) {
            TFile f(filePath.c_str(), "READ");
            if (!f.IsZombie()) {
                TH1 *h = dynamic_cast<TH1 *>(f.Get(histPath.c_str()));
                if (h) {
                    const int bmin = h->GetXaxis()->FindBin(cenMin + 1e-3);
                    const int bmax = h->GetXaxis()->FindBin(cenMax - 1e-3);
                    count = h->Integral(bmin, bmax);
                }
            }
        }
        out.emplace_back(tag, count);
        total += count;
    }
    if (total > 0.0) {
        for (auto &kv : out) kv.second /= total;
    }
    return out;
}

struct GroupContext {
    std::string key;
    std::string dirName;
    std::vector<BinPlanItem> items;
};

struct RunConfig {
    std::string mode;
    std::string dataTree;
    std::string mcTree;
    std::string wpFile;
    std::string outputRoot;
    std::string bkgFunc;
    std::string sigFunc;
    std::string isMatter;
    std::string mcFileForAcceptance;
    std::string eventSignalLossFile;
    std::string eventSignalLossMethod{"multiplicity"};
    std::vector<double> cenBins;
    std::vector<double> relatedMultiplicityCenter;
    std::vector<std::vector<double>> ptBinsByCentrality;
    std::vector<std::vector<std::string>> topologySelectionsByCentrality;
    std::vector<bool> addEventSignalLossCenPt;
    std::string mcFileForAbsorption;
    double originalCtaoAbsorption{7.6};
    std::string absorptionTree;
    std::string scoreEffDir;
    std::string basicSelectionDataForMcEff;
    std::string additionalDataSelectionGeneral;
    std::string additionalDataSelection;
    std::string centralitySelection;
    bool enableOneBinQa{false};
    bool onTheFlyChecksEnabled{false};
    bool saveChecksPdf{false};
    bool saveResultsToPdf{false};
    double onTheFlyMassWindowNSigmas{0.0};
    std::vector<std::string> oneBinQaColumns;
    std::vector<std::string> onTheFlyVariables;
    std::vector<Hist2DPair> oneBinQaPairs;
    std::unordered_map<std::string, AxisSpec> oneBinQaAxisPool;
    std::string onTheFlyChecksRoot;
    std::string onTheFlyChecksPdfDir;
    std::vector<double> ptBins;
    std::vector<std::vector<double>> ctBinsByPt;
    bool addAbsorptionCorrectionPtCt{false};

    bool doSystematics{false};
    int randomSeed{40};
    int systNtrails{200};
    int nTrailsForIntegralSyst{1000};
    int nCombinationsForIntegralSystExtrapolation{500};
    int nBinsForFit{100};
    int systBdtScoreNPoints{20};
    double systThrChi2Ndf{2.0};
    double systThrSignificance{2.5};
    std::vector<std::string> systBkgFuncs;
    std::vector<std::string> systSigFuncs;
    std::vector<std::string> systAbsorptionFiles;
    std::vector<std::string> systAbsorptionFileLabels;
    std::vector<std::string> integralFitFuncs;
    std::map<std::string, SpectrumFunctionConfig> integralFitParameters;
    std::string integralFitFunc{"fBGBW"};
    bool rejectIntegralFitFuncByChi2{false};
    double integralFitFuncMaxChi2Ndf{5.0};
    double integralFitFuncFallbackFraction{0.20};
    double integralGaussFitMaxChi2Ndf{3.0};
    double perPtBinGaussFitMaxChi2Ndf{3.0};
    double integralExtrapToyMaxChi2Ndf{-1.0};
    double integralFitRangeMin{0.0};
    double integralFitRangeMax{10.0};
    double integratedYieldRangeMin{0.0};
    double integratedYieldRangeMax{10.0};
    double integralLowPtMaxFactor{10.0};
    bool integralFitUseMinosErrors{false};
    double absorptionLength{0.5};

    double branchingRatio{0.25};
    double branchingRatioFractionalUncertainty{0.0};
    double deltaRap{2.0};

    GeneralHelper::MassFitConfig fitCfg;
    PlotLabelConfig plotLabels;
};

std::string JoinSelectionParts(const std::vector<std::string> &parts) {
    std::ostringstream os;
    bool first = true;
    for (const auto &part : parts) {
        if (part.empty()) continue;
        if (!first) os << " && ";
        os << "(" << part << ")";
        first = false;
    }
    return os.str();
}

RunConfig BuildRunConfig(const GeneralHelper::Json &cfg, const std::string &mode) {
    RunConfig out;
    out.mode = mode;

    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto trees = common.value("tree_names", GeneralHelper::Json::object());
    const auto tags = common.value("tags", GeneralHelper::Json::object());
    const auto params = common.value("parameters", GeneralHelper::Json::object());
    const auto commonSel = common.value("selection", GeneralHelper::Json::object());

    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto execution = cfg.value("execution", GeneralHelper::Json::object());
    const auto checks = cfg.value("checks", GeneralHelper::Json::object());
    const auto checksGeneral = checks.value("general", GeneralHelper::Json::object());
    const auto onFlyChecks = checks.value("hypertriton_onthefly_checks", GeneralHelper::Json::object());
    const auto fit = analysis.value("fit", GeneralHelper::Json::object());
    const auto selection = analysis.value("selection", GeneralHelper::Json::object());
    const auto syst = analysis.value("systematics", GeneralHelper::Json::object());
    const auto modeProfiles = analysis.value("mode_profiles", GeneralHelper::Json::object());
    const auto profile = modeProfiles.value(mode, GeneralHelper::Json::object());
    const auto commonBinning = common.value("binning", GeneralHelper::Json::object());

    out.dataTree = GetS(trees, "data", "O2hypcands");
    out.mcTree = GetS(trees, "mc", "O2mchypcands");
    out.absorptionTree = GetS(trees, "absorption", "he3candidates");
    out.wpFile = ResolveWpFile(cfg, mode);
    out.scoreEffDir = ResolveScoreEffDir(cfg, mode);
    out.bkgFunc = GetS(fit, "bkg_fit_func", "pol2");
    out.sigFunc = GetS(fit, "signal_fit_func", "dscb");
    out.isMatter = GetS(selection, "is_matter", "both");
    out.basicSelectionDataForMcEff = GetS(commonSel, "basic_selection_data", "");
    out.additionalDataSelectionGeneral = GetS(selection, "additional_data_selection_general", "");
    out.additionalDataSelection = GetS(profile, "additional_data_selection", "");
    out.centralitySelection = GetS(profile, "centrality_selection", "");
    const bool checksEnabled = checks.value("enabled", false);
    out.onTheFlyChecksEnabled = checksEnabled && onFlyChecks.value("enable", false);
    out.enableOneBinQa = out.onTheFlyChecksEnabled;
    out.saveChecksPdf = checks.value("save_pdf", false);
    out.saveResultsToPdf = execution.value("save_results_to_pdf", false);
    out.onTheFlyMassWindowNSigmas = onFlyChecks.value("nsigmas_mass_window", 0.0);
    if (onFlyChecks.contains("variables") && onFlyChecks["variables"].is_array()) {
        out.onTheFlyVariables = onFlyChecks["variables"].get<std::vector<std::string>>();
    }
    if (onFlyChecks.contains("variables") && onFlyChecks["variables"].is_array()) {
        out.oneBinQaColumns = onFlyChecks["variables"].get<std::vector<std::string>>();
    } else if (checksGeneral.contains("variables") && checksGeneral["variables"].is_array()) {
        out.oneBinQaColumns = checksGeneral["variables"].get<std::vector<std::string>>();
    } else if (checksGeneral.contains("varpool") && checksGeneral["varpool"].is_array()) {
        out.oneBinQaColumns = checksGeneral["varpool"].get<std::vector<std::string>>();
    }

    const auto pairsJson = onFlyChecks.value("hist2d_pairs", GeneralHelper::Json::array());
    if (pairsJson.is_array()) {
        for (const auto &row : pairsJson) {
            if (!row.is_array() || row.size() < 2) continue;
            if (!row[0].is_string() || !row[1].is_string()) continue;
            out.oneBinQaPairs.push_back(Hist2DPair{row[0].get<std::string>(), row[1].get<std::string>()});
        }
    }

    const auto axisPool = checksGeneral.value("axis_pool", GeneralHelper::Json::object());
    if (axisPool.is_object()) {
        for (auto it = axisPool.begin(); it != axisPool.end(); ++it) {
            if (!it.value().is_object()) continue;
            AxisSpec ax;
            ax.nBins = it.value().value("nbins", 80);
            ax.min = it.value().value("min", 0.0);
            ax.max = it.value().value("max", 1.0);
            ax.title = it.value().value("title", it.key());
            out.oneBinQaAxisPool[it.key()] = ax;
        }
    }
    if (commonBinning.contains("pt_bins") && commonBinning["pt_bins"].is_array()) {
        out.ptBins = commonBinning["pt_bins"].get<std::vector<double>>();
    }
    if (commonBinning.contains("ct_bins_by_pt") && commonBinning["ct_bins_by_pt"].is_array()) {
        out.ctBinsByPt = commonBinning["ct_bins_by_pt"].get<std::vector<std::vector<double>>>();
    }
    out.addAbsorptionCorrectionPtCt = profile.value("add_absorption_correction", false);
    out.mcFileForAcceptance = GetS(path, "mc_path", "");
    out.eventSignalLossFile = GetS(path, "event_signal_loss_file", "");
    out.eventSignalLossMethod = NormalizeEventSignalLossMethod(GetS(analysis, "event_signal_loss_method", "multiplicity"));
    if (commonBinning.contains("cen_bins") && commonBinning["cen_bins"].is_array()) {
        out.cenBins = commonBinning["cen_bins"].get<std::vector<double>>();
    }
    if (commonBinning.contains("related_multiplicity_center") && commonBinning["related_multiplicity_center"].is_array()) {
        out.relatedMultiplicityCenter = commonBinning["related_multiplicity_center"].get<std::vector<double>>();
    }
    if (commonBinning.contains("pt_bins_by_centrality") && commonBinning["pt_bins_by_centrality"].is_array()) {
        out.ptBinsByCentrality = commonBinning["pt_bins_by_centrality"].get<std::vector<std::vector<double>>>();
    }
    if (profile.contains("data_selection_topology") && profile["data_selection_topology"].is_array()) {
        out.topologySelectionsByCentrality.clear();
        for (const auto &row : profile["data_selection_topology"]) {
            std::vector<std::string> rowSelections;
            if (row.is_array()) {
                for (const auto &cell : row) {
                    rowSelections.push_back(cell.is_string() ? cell.get<std::string>() : std::string{});
                }
            }
            out.topologySelectionsByCentrality.push_back(std::move(rowSelections));
        }
    }
    if (params.contains("add_event_signal_loss_cen_pt") && params["add_event_signal_loss_cen_pt"].is_array()) {
        out.addEventSignalLossCenPt = params["add_event_signal_loss_cen_pt"].get<std::vector<bool>>();
    }
    out.mcFileForAbsorption = GetS(path, "mc_file_for_absorption", "");
    out.originalCtaoAbsorption = GetD(params, "original_ctao_absorption", out.originalCtaoAbsorption);

    const bool doSystExec = execution.value("do_systematics", false);
    out.doSystematics = doSystExec && mode != "topology_spectrum";
    out.randomSeed = GetI(syst, "random_seed", out.randomSeed);
    out.systNtrails = GetI(syst, "syst_ntrails", out.systNtrails);
    out.nTrailsForIntegralSyst = GetI(syst, "n_trails_for_integral_syst", out.nTrailsForIntegralSyst);
    out.nCombinationsForIntegralSystExtrapolation =
        GetI(syst, "n_combinations_for_integral_syst_extrapolation", out.nCombinationsForIntegralSystExtrapolation);
    out.nBinsForFit = GetI(syst, "n_bins_for_fit", out.nBinsForFit);
    out.systBdtScoreNPoints = GetI(syst, "syst_bdt_score_npoints", out.systBdtScoreNPoints);
    out.systThrChi2Ndf = GetD(syst, "syst_thrashold_chi2ndf", out.systThrChi2Ndf);
    out.systThrSignificance = GetD(syst, "syst_thrashold_significance", out.systThrSignificance);
    if (syst.contains("syst_bkg_funcs") && syst["syst_bkg_funcs"].is_array()) {
        out.systBkgFuncs = syst["syst_bkg_funcs"].get<std::vector<std::string>>();
    }
    if (syst.contains("syst_signal_funcs") && syst["syst_signal_funcs"].is_array()) {
        out.systSigFuncs = syst["syst_signal_funcs"].get<std::vector<std::string>>();
    }
    if (syst.contains("syst_absorption_files") && syst["syst_absorption_files"].is_array()) {
        out.systAbsorptionFiles = syst["syst_absorption_files"].get<std::vector<std::string>>();
    }
    if (syst.contains("syst_absorption_file_labels") && syst["syst_absorption_file_labels"].is_array()) {
        out.systAbsorptionFileLabels = syst["syst_absorption_file_labels"].get<std::vector<std::string>>();
    }
    out.absorptionLength = GetD(syst, "absorption_length", out.absorptionLength);
    out.integralFitFunc = GetS(fit, "integral_fit_func", out.integralFitFunc);
    if (syst.contains("integral_fit_funcs") && syst["integral_fit_funcs"].is_array()) {
        out.integralFitFuncs = syst["integral_fit_funcs"].get<std::vector<std::string>>();
    }
    if (out.integralFitFuncs.empty()) out.integralFitFuncs.push_back(out.integralFitFunc);
    out.rejectIntegralFitFuncByChi2 = syst.value("reject_integral_fit_func_by_chi2", out.rejectIntegralFitFuncByChi2);
    out.integralFitFuncMaxChi2Ndf = GetD(syst, "integral_fit_func_max_chi2ndf", out.integralFitFuncMaxChi2Ndf);
    out.integralFitFuncFallbackFraction =
        GetD(syst, "integral_fit_func_fallback_fraction", out.integralFitFuncFallbackFraction);
    out.integralGaussFitMaxChi2Ndf =
        GetD(syst, "integral_gauss_fit_max_chi2ndf", out.integralGaussFitMaxChi2Ndf);
    out.perPtBinGaussFitMaxChi2Ndf =
        GetD(syst, "per_ptbin_gauss_fit_max_chi2ndf", out.perPtBinGaussFitMaxChi2Ndf);
    out.integralExtrapToyMaxChi2Ndf =
        GetD(syst, "integral_extrap_toy_max_chi2ndf", out.integralExtrapToyMaxChi2Ndf);
    if (syst.contains("integral_fit_range") && syst["integral_fit_range"].is_array() &&
        syst["integral_fit_range"].size() >= 2) {
        out.integralFitRangeMin = syst["integral_fit_range"][0].get<double>();
        out.integralFitRangeMax = syst["integral_fit_range"][1].get<double>();
    }
    if (syst.contains("integrated_yield_range") && syst["integrated_yield_range"].is_array() &&
        syst["integrated_yield_range"].size() >= 2) {
        out.integratedYieldRangeMin = syst["integrated_yield_range"][0].get<double>();
        out.integratedYieldRangeMax = syst["integrated_yield_range"][1].get<double>();
    }
    out.integralLowPtMaxFactor =
        GetD(syst, "integral_lowpt_max_factor", out.integralLowPtMaxFactor);
    out.integralFitUseMinosErrors =
        syst.value("integral_fit_use_minos_errors", out.integralFitUseMinosErrors);
    out.integralFitParameters = ParseSpectrumFitParameterConfigs(fit.value("integral_fit_parameters", GeneralHelper::Json::object()));
    auto systIntegralFitParameters =
        ParseSpectrumFitParameterConfigs(syst.value("integral_fit_parameters", GeneralHelper::Json::object()));
    for (auto &kv : systIntegralFitParameters) {
        out.integralFitParameters[kv.first] = std::move(kv.second);
    }
    out.branchingRatio = GetD(params, "branching_ratio", out.branchingRatio);
    out.branchingRatioFractionalUncertainty =
        GetD(params, "branching_ratio_fractional_uncertainty",
             GetD(syst, "branching_ratio_fractional_uncertainty", out.branchingRatioFractionalUncertainty));
    out.deltaRap = GetD(params, "delta_rap", out.deltaRap);

    out.fitCfg.massMin = GetD(params, "mass_min", 2.96);
    out.fitCfg.massMax = GetD(params, "mass_max", 3.04);
    out.fitCfg.sigmaRangeMcToData = params.value("sigma_range_mc_to_data", std::vector<double>{1.0, 1.5});
    out.fitCfg.nBinsMcFrame = params.value("mass_nbins_mc", out.fitCfg.nBinsMcFrame);
    out.fitCfg.nBinsDataFrame = params.value("mass_nbins_data", out.fitCfg.nBinsDataFrame);
    out.fitCfg.useBinnedDataFit = params.value("mass_fit_use_binned_data", out.fitCfg.useBinnedDataFit);
    out.fitCfg.prefitSidebands = params.value("mass_fit_prefit_sidebands", out.fitCfg.prefitSidebands);
    out.fitCfg.sidebandExclusionSigma = GetD(params, "mass_fit_sideband_exclusion_sigma", out.fitCfg.sidebandExclusionSigma);

    out.plotLabels.usePerformance = tags.value("use_performance", false);
    out.plotLabels.performanceLabel = GetS(tags, "performance_label", "");
    out.plotLabels.period = GetS(tags, "period", "");
    out.plotLabels.periodMark = GetS(tags, "period_mark", "");
    out.plotLabels.collisionSystem = GetS(tags, "collision_system", "");
    out.plotLabels.collisionEnergy = GetS(tags, "collision_energy", "");

    std::string periodTag;
    if (IsCombinePeriodEnabled(cfg)) {
        const auto commonPeriods = common.value("periods", GeneralHelper::Json::array());
        periodTag = (commonPeriods.is_array() && !commonPeriods.empty()) ? BuildCombinedPeriodTag(cfg) : "combined_period";
        out.plotLabels.period = periodTag;
        out.plotLabels.periodMark.clear();
    } else {
        const std::string period = GetS(tags, "period", "period");
        const std::string periodMark = GetS(tags, "period_mark", "mark");
        periodTag = period + "_" + periodMark;
    }
    const std::string outputBase = GetS(path, "output_dir", "./Outputs");
    const std::string outputDir = outputBase + "/" + periodTag + "/" + mode + "/" + out.isMatter;
    out.outputRoot = outputDir + "/spectrum.root";
    out.onTheFlyChecksRoot = outputDir + "/Checks_hypertriton/checks_hypertriton.root";
    out.onTheFlyChecksPdfDir = outputDir + "/Checks_hypertriton";

    if (out.onTheFlyVariables.empty()) {
        out.onTheFlyVariables = out.oneBinQaColumns;
    }
    if (out.onTheFlyChecksEnabled && out.onTheFlyMassWindowNSigmas > 0.0) {
        const std::string massCol = "fMassH3L";
        if (std::find(out.oneBinQaColumns.begin(), out.oneBinQaColumns.end(), massCol) == out.oneBinQaColumns.end()) {
            out.oneBinQaColumns.push_back(massCol);
        }
    }
    return out;
}

std::string ResolveAcceptanceMcFileForGroup(const RunConfig &runCfg, const GroupContext &) {
    return runCfg.mcFileForAcceptance;
}

bool IsAbsorptionCorrectionEnabled(const RunConfig &runCfg) {
    // add_absorption_correction switch is currently defined for pt_ct/ct_single profiles.
    if (runCfg.mode == "pt_ct" || runCfg.mode == "ct_single") {
        return runCfg.addAbsorptionCorrectionPtCt;
    }
    return true;
}

std::vector<GroupContext> BuildGroups(const std::string &mode, const BinPlan &plan) {
    std::map<std::string, GroupContext> grouped;

    for (const auto &item : plan.items) {
        GroupContext ctx;
        if (mode == "bdt_spectrum" || mode == "topology_spectrum") {
            ctx.key = BuildRangeLabel("cen", item.cenMin, item.cenMax);
            ctx.dirName = ctx.key;
        } else if (mode == "pt_ct") {
            if (item.hasCen) {
                std::ostringstream os;
                os << BuildRangeLabel("cen", item.cenMin, item.cenMax) << "_"
                   << BuildRangeLabel("pt", item.ptMin, item.ptMax);
                ctx.key = os.str();
                ctx.dirName = ctx.key;
            } else {
                ctx.key = BuildRangeLabel("pt", item.ptMin, item.ptMax);
                ctx.dirName = ctx.key;
            }
        } else {
            ctx.key = "ct_single";
            ctx.dirName = "ct_single";
        }

        auto it = grouped.find(ctx.key);
        if (it == grouped.end()) {
            grouped[ctx.key] = ctx;
        }
        grouped[ctx.key].items.push_back(item);
    }

    std::vector<GroupContext> out;
    out.reserve(grouped.size());
    for (auto &kv : grouped) {
        auto &items = kv.second.items;
        if (mode == "ct_single" || mode == "pt_ct") {
            std::sort(items.begin(), items.end(), [](const BinPlanItem &a, const BinPlanItem &b) { return a.ctMin < b.ctMin; });
        } else {
            std::sort(items.begin(), items.end(), [](const BinPlanItem &a, const BinPlanItem &b) { return a.ptMin < b.ptMin; });
        }
        out.push_back(std::move(kv.second));
    }
    return out;
}

std::vector<double> BuildAxisEdges(const std::string &mode, const std::vector<BinPlanItem> &items) {
    std::vector<double> edges;
    edges.reserve(items.size() + 1);
    if (items.empty()) return edges;

    const bool useCtAxis = (mode == "ct_single" || mode == "pt_ct");
    edges.push_back(useCtAxis ? items.front().ctMin : items.front().ptMin);
    for (const auto &item : items) {
        edges.push_back(useCtAxis ? item.ctMax : item.ptMax);
    }
    return edges;
}

GeneralHelper::WorkingPointResult GetWorkingPointForItem(const std::string &mode,
                                                          const std::string &wpFile,
                                                          const BinPlanItem &item) {
    if (wpFile.empty()) return GeneralHelper::WorkingPointResult{};

    if (mode == "ct_single") {
        return GeneralHelper::GetWpForCtSingle(wpFile, item.ctMin, item.ctMax);
    }
    if (mode == "pt_ct") {
        return GeneralHelper::GetWpForPtCt(wpFile, item.ptMin, item.ptMax, item.ctMin, item.ctMax, item.cenMin, item.cenMax);
    }
    return GeneralHelper::GetWpForCenPt(wpFile, item.cenMin, item.cenMax, item.ptMin, item.ptMax);
}

struct CorrectedCountsResult {
    double value{0.0};
    double error{0.0};
    double bdtEfficiency{1.0};
    double acceptance{1.0};
    double absorption{1.0};
    double eventLoss{1.0};
    double eventSplitting{1.0};
    double signalLoss{1.0};
    double branchingRatio{1.0};
    double deltaRapidity{1.0};
    double matterRatio{1.0};
    double nEvents{1.0};
    double binWidth{1.0};
};

CorrectedCountsResult ComputeCorrectedCounts(const std::string &mode,
                                             const BinPlanItem &item,
                                             const GeneralHelper::MassFitResult &fitRes,
                                             double acc,
                                             double abso,
                                             double bdtEff,
                                             double eventLoss,
                                             double eventSplitting,
                                             double signalLoss,
                                             double nEvents,
                                             double branchingRatio,
                                             double deltaRap,
                                             const std::string &isMatter,
                                             bool addAbsorptionCorrectionPtCt) {
    const double eff = (bdtEff > 0.0) ? bdtEff : 1.0;
    const double accVal = (acc > 0.0) ? acc : 1.0;
    const double absoVal = (abso > 0.0) ? abso : 1.0;
    const double raw = fitRes.signal;
    const double rawErr = fitRes.signalErr;

    CorrectedCountsResult out;
    out.bdtEfficiency = eff;
    out.acceptance = accVal;
    out.absorption = absoVal;
    out.eventLoss = (eventLoss > 0.0) ? eventLoss : 1.0;
    out.eventSplitting = (eventSplitting > 0.0) ? eventSplitting : 1.0;
    out.signalLoss = (signalLoss > 0.0) ? signalLoss : 1.0;

    if (mode == "bdt_spectrum" || mode == "topology_spectrum") {
        const double lossCorr = out.eventLoss / out.signalLoss;
        const double base = raw / accVal / absoVal / eff * lossCorr;
        const double baseErr = rawErr / accVal / absoVal / eff * lossCorr;
        const double binWidth = item.ptMax - item.ptMin;
        const double matterRatio = (isMatter == "both") ? 2.0 : 1.0;
        const double norm = std::max(1.0, nEvents) * std::max(1e-12, branchingRatio) * std::max(1e-12, deltaRap) *
                            std::max(1e-12, binWidth) * matterRatio;
        out.value = base / norm;
        out.error = baseErr / norm;
        out.branchingRatio = std::max(1e-12, branchingRatio);
        out.deltaRapidity = std::max(1e-12, deltaRap);
        out.matterRatio = matterRatio;
        out.nEvents = std::max(1.0, nEvents);
        out.binWidth = std::max(1e-12, binWidth);
        return out;
    } else if (mode == "ct_single") {
        const double binWidth = item.ctMax - item.ctMin;
        const double absoFactor = addAbsorptionCorrectionPtCt ? absoVal : 1.0;
        const double base = raw / accVal / absoFactor / eff;
        const double baseErr = rawErr / accVal / absoFactor / eff;
        out.value = base / std::max(1e-12, binWidth);
        out.error = baseErr / std::max(1e-12, binWidth);
        out.absorption = absoFactor;
        out.binWidth = std::max(1e-12, binWidth);
        return out;
    } else if (mode == "pt_ct") {
        const double binwidthCt = item.ctMax - item.ctMin;
        const double absoFactor = addAbsorptionCorrectionPtCt ? absoVal : 1.0;
        const double base = raw / accVal / absoFactor / eff;
        const double baseErr = rawErr / accVal / absoFactor / eff;
        out.value = base / std::max(1e-12, binwidthCt);
        out.error = baseErr / std::max(1e-12, binwidthCt);
        out.absorption = absoFactor;
        out.binWidth = std::max(1e-12, binwidthCt);
        return out;
    }
    return out;
}

std::string BuildPdfNameWithSuffix(const std::string &baseName,
                                   const std::string &relPath,
                                   std::unordered_map<std::string, int> &usedNames) {
    auto sanitize = [](std::string s) {
        for (char &c : s) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
        }
        return s;
    };

    const std::string stem = sanitize(baseName);
    const bool hasBinInfoInName =
        (stem.find("cen_") != std::string::npos) ||
        (stem.find("_pt_") != std::string::npos) ||
        (stem.find("_ct_") != std::string::npos);

    std::string candidate = stem;
    if (!hasBinInfoInName) {
        std::string relTag = relPath;
        std::replace(relTag.begin(), relTag.end(), '/', '_');
        relTag = sanitize(relTag);
        if (!relTag.empty()) {
            candidate += "_" + relTag;
        }
    }

    auto it = usedNames.find(candidate);
    if (it == usedNames.end()) {
        usedNames[candidate] = 1;
        return candidate;
    }

    int idx = ++it->second;
    std::string candidateIndexed = candidate + "_" + std::to_string(idx);
    while (usedNames.count(candidateIndexed)) {
        ++idx;
        candidateIndexed = candidate + "_" + std::to_string(idx);
    }
    usedNames[candidateIndexed] = 1;
    return candidateIndexed;
}

void ExportRootObjectsToPdf(TDirectory *dir,
                            const std::filesystem::path &outRoot,
                            const std::string &relPath,
                            bool inStdDir,
                            std::unordered_map<std::string, int> &usedNames) {
    if (!dir) return;
    TIter next(dir->GetListOfKeys());
    while (auto *key = dynamic_cast<TKey *>(next())) {
        if (!key) continue;
        TObject *obj = key->ReadObj();
        if (!obj) continue;

        if (obj->InheritsFrom(TDirectory::Class())) {
            const std::string nextRel = relPath.empty() ? obj->GetName() : (relPath + "/" + obj->GetName());
            const bool nextInStd = inStdDir || (std::string(obj->GetName()) == "std");
            ExportRootObjectsToPdf(dynamic_cast<TDirectory *>(obj), outRoot, nextRel, nextInStd, usedNames);
            continue;
        }

        if (obj->InheritsFrom(TCanvas::Class())) {
            auto *c = dynamic_cast<TCanvas *>(obj);
            if (!c) continue;
            const std::string outName = BuildPdfNameWithSuffix(c->GetName(), relPath, usedNames);
            const std::filesystem::path outFile = outRoot / (outName + ".pdf");
            c->SaveAs(outFile.string().c_str());
            continue;
        }

        if (inStdDir && obj->InheritsFrom(TH1::Class())) {
            auto *h = dynamic_cast<TH1 *>(obj);
            if (!h) continue;
            const std::string hname = h->GetName();
            if (hname.rfind("h_raw_over_nevents", 0) == 0) {
                std::string cname = hname;
                cname[0] = 'c';
                TObject *cObj = dir->Get(cname.c_str());
                auto *cFromDir = dynamic_cast<TCanvas *>(cObj);
                if (cFromDir) {
                    const std::string outName = BuildPdfNameWithSuffix(hname, relPath, usedNames);
                    const std::filesystem::path outFile = outRoot / (outName + ".pdf");
                    cFromDir->SaveAs(outFile.string().c_str());
                    continue;
                }
            }
            TCanvas cTmp((std::string("c_pdf_") + h->GetName()).c_str(), h->GetName(), 900, 700);
            cTmp.cd();
            if (h->InheritsFrom(TH2::Class())) h->Draw("COLZ");
            else h->Draw("E1");
            const std::string outName = BuildPdfNameWithSuffix(h->GetName(), relPath, usedNames);
            const std::filesystem::path outFile = outRoot / (outName + ".pdf");
            cTmp.SaveAs(outFile.string().c_str());
        }
    }
}

void SaveResultsToPdfIfRequested(const RunConfig &runCfg) {
    if (!runCfg.saveResultsToPdf) return;
    const std::filesystem::path outputDir = std::filesystem::path(runCfg.outputRoot).parent_path();
    const std::filesystem::path outPlotsDir = outputDir / "OutPlots";
    std::filesystem::create_directories(outPlotsDir);

    TFile fin(runCfg.outputRoot.c_str(), "READ");
    if (fin.IsZombie()) {
        std::cerr << "[save_results_to_pdf] cannot open ROOT output: " << runCfg.outputRoot << std::endl;
        return;
    }
    std::unordered_map<std::string, int> usedNames;
    ExportRootObjectsToPdf(&fin, outPlotsDir, "", false, usedNames);
    fin.Close();
}

std::vector<std::pair<double, double>> LoadScoreEfficiencyArray(const std::string &path) {
    std::vector<std::pair<double, double>> rows;
    if (path.empty()) return rows;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return rows;

    double score = 0.0;
    double eff = 0.0;
    while (ifs >> score >> eff) {
        rows.emplace_back(score, eff);
    }
    return rows;
}

struct BdtCandidate {
    double score{0.0};
    double efficiency{0.0};
};

std::vector<BdtCandidate> BuildBdtCandidates(double wpScore,
                                             double wpEff,
                                             int totalPoints,
                                             const std::vector<std::pair<double, double>> &arr) {
    std::vector<BdtCandidate> out;
    if (totalPoints <= 1 || arr.empty()) {
        out.push_back({wpScore, wpEff});
        return out;
    }

    size_t centerIdx = 0;
    double bestDiff = std::numeric_limits<double>::max();
    for (size_t i = 0; i < arr.size(); ++i) {
        double diff = std::abs(arr[i].first - wpScore);
        if (diff < bestDiff) {
            bestDiff = diff;
            centerIdx = i;
        }
    }

    out.push_back({arr[centerIdx].first, arr[centerIdx].second});
    size_t left = centerIdx;
    size_t right = centerIdx;
    while (out.size() < static_cast<size_t>(totalPoints)) {
        bool progressed = false;
        if (left > 0) {
            --left;
            out.push_back({arr[left].first, arr[left].second});
            progressed = true;
            if (out.size() >= static_cast<size_t>(totalPoints)) break;
        }
        if (right + 1 < arr.size()) {
            ++right;
            out.push_back({arr[right].first, arr[right].second});
            progressed = true;
        }
        if (!progressed) break;
    }
    return out;
}

std::vector<std::tuple<size_t, size_t, size_t>> BuildTrailCombos(size_t nBdt, size_t nBkg, size_t nSig) {
    std::vector<std::tuple<size_t, size_t, size_t>> combos;
    combos.reserve(nBdt * nBkg * nSig);
    for (size_t i = 0; i < nBdt; ++i) {
        for (size_t j = 0; j < nBkg; ++j) {
            for (size_t k = 0; k < nSig; ++k) {
                combos.emplace_back(i, j, k);
            }
        }
    }
    return combos;
}

BinningCorrectionConfig BuildCorrectionConfig(const RunConfig &runCfg) {
    BinningCorrectionConfig cfg;
    cfg.mode = runCfg.mode;
    cfg.isMatter = runCfg.isMatter;
    cfg.eventSignalLossMethod = runCfg.eventSignalLossMethod;
    cfg.mcTree = runCfg.mcTree;
    cfg.absorptionTree = runCfg.absorptionTree;
    cfg.mcFileForAcceptance = runCfg.mcFileForAcceptance;
    cfg.mcFileForAbsorption = runCfg.mcFileForAbsorption;
    if (!IsAbsorptionCorrectionEnabled(runCfg)) {
        cfg.mcFileForAbsorption.clear();
    }
    cfg.mcEfficiencySelection = JoinSelectionParts({runCfg.basicSelectionDataForMcEff,
                                                    runCfg.additionalDataSelectionGeneral,
                                                    runCfg.additionalDataSelection});
    cfg.originalCtaoAbsorption = runCfg.originalCtaoAbsorption;
    cfg.cenBins = runCfg.cenBins;
    cfg.ptBinsByCentrality = runCfg.ptBinsByCentrality;
    cfg.topologySelectionsByCentrality = runCfg.topologySelectionsByCentrality;
    cfg.ptBins = runCfg.ptBins;
    cfg.ctBinsByPt = runCfg.ctBinsByPt;
    return cfg;
}

struct SysBinArtifacts {
    std::string subBinName;
    std::unique_ptr<TH1D> hCorrDist;
    std::unique_ptr<TH1D> hTrailBdtEff;
    std::unique_ptr<TH1D> hCorrVsAbso;
    std::vector<double> acceptedCorrValues;
    std::vector<double> corrValuesByTrial;
    std::vector<std::unique_ptr<RooPlot>> trailFrames;
    std::unique_ptr<TCanvas> cCorrDist;
    std::unique_ptr<TCanvas> cCorrVsAbso;
    double selectionRms{0.0};
    double absorptionRms{0.0};
};

std::string SanitizeName(const std::string &input) {
    std::string out = input;
    for (char &c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return out;
}

AxisSpec ResolveQaAxis(const std::string &var,
                       const std::unordered_map<std::string, AxisSpec> &axisPool) {
    auto it = axisPool.find(var);
    if (it != axisPool.end()) return it->second;
    AxisSpec ax;
    ax.title = var;
    if (var == "fPt") {
        ax = AxisSpec{80, 0.0, 12.0, var};
    } else if (var == "fCt") {
        ax = AxisSpec{80, 0.0, 40.0, var};
    } else if (var == "fCosPA") {
        ax = AxisSpec{100, 0.95, 1.0, var};
    } else if (var == "fNSigmaHe") {
        ax = AxisSpec{100, -10.0, 10.0, var};
    }
    return ax;
}

std::string MakeSysSubBinName(const std::string &mode, const BinPlanItem &item) {
    if (mode == "pt_ct" || mode == "ct_single") {
        return BuildRangeLabel("ct", item.ctMin, item.ctMax);
    }
    return BuildRangeLabel("pt", item.ptMin, item.ptMax);
}

void WriteGroupOutput(TFile &fout,
                      const std::string &dirName,
                      TH1D *hRaw,
                      TH1D *hCorr,
                      TH1D *hBdtEff,
                      TH1D *hAcc,
                      TH1D *hAbso,
                      TH1D *hEventLoss,
                      TH1D *hEventSplitting,
                      TH1D *hSignalLoss,
                      TH1D *hFitSigma,
                      TH1D *hFitMean,
                      TH1D *hFitChi2,
                      TH1D *hPeriodEventWeights,
                      TH1D *hRawOverNevts,
                      TCanvas *cRawOverNevts,
                      TH1D *hSysSelection,
                      TH1D *hSysAbsorption,
                      TH1D *hSysBranching,
                      TH1D *hSysTotal,
                      TH1D *hFinalStat,
                      TGraphAsymmErrors *gFinalSys,
                      const std::vector<SysBinArtifacts> *sysArtifacts,
                      const std::vector<std::unique_ptr<RooPlot>> *fitFrames = nullptr,
                      const std::vector<std::unique_ptr<TCanvas>> *fitCanvases = nullptr,
                      const std::vector<std::unique_ptr<TF1>> *shapeFits = nullptr,
                      const std::vector<std::unique_ptr<TCanvas>> *shapeCanvases = nullptr,
                      TF1 *finalStdFit = nullptr,
                      const std::string &finalXAxisTitle = "",
                      const std::string &finalExtraText = "",
                      const PlotLabelConfig *finalLabelCfg = nullptr,
                      const std::string &finalIsMatter = "",
                      double finalNEvents = 0.0,
                      TH1D *tauPerPt = nullptr) {
    // Keep the file content from other groups/modes and rewrite only this first-level group.
    if (fout.GetDirectory(dirName.c_str())) {
        fout.cd();
        fout.Delete((dirName + ";*").c_str());
    }
    TDirectory *dir = fout.GetDirectory(dirName.c_str());
    if (!dir) dir = fout.mkdir(dirName.c_str());
    if (!dir) throw std::runtime_error("Cannot create output directory: " + dirName);

    TDirectory *stdDir = dir->GetDirectory("std");
    if (!stdDir) stdDir = dir->mkdir("std");
    if (stdDir) {
        stdDir->cd();
        if (hRaw) hRaw->Write("h_raw_counts", TObject::kOverwrite);
        if (hCorr) hCorr->Write("h_corrected_counts", TObject::kOverwrite);
        if (hBdtEff) hBdtEff->Write("h_bdt_efficiency", TObject::kOverwrite);
        if (hAcc) hAcc->Write("h_acceptance", TObject::kOverwrite);
        if (hAbso) hAbso->Write("h_absorption", TObject::kOverwrite);
        if (hEventLoss) hEventLoss->Write("h_event_loss", TObject::kOverwrite);
        if (hEventSplitting) hEventSplitting->Write("h_event_splitting", TObject::kOverwrite);
        if (hSignalLoss) hSignalLoss->Write("h_signal_loss", TObject::kOverwrite);
        if (hFitSigma) hFitSigma->Write("h_fit_sigma", TObject::kOverwrite);
        if (hFitMean) hFitMean->Write("h_fit_mean", TObject::kOverwrite);
        if (hFitChi2) hFitChi2->Write("h_fit_chi2ndf", TObject::kOverwrite);
        if (hPeriodEventWeights) hPeriodEventWeights->Write("h_period_event_weights", TObject::kOverwrite);
        if (hRawOverNevts) hRawOverNevts->Write("h_raw_over_nevents", TObject::kOverwrite);
        if (cRawOverNevts) cRawOverNevts->Write("c_raw_over_nevents", TObject::kOverwrite);
        if (shapeFits) {
            for (const auto &fn : *shapeFits) {
                if (fn) fn->Write(fn->GetName(), TObject::kOverwrite);
            }
        }
        if (shapeCanvases) {
            for (const auto &cv : *shapeCanvases) {
                if (cv) cv->Write(cv->GetName(), TObject::kOverwrite);
            }
        }
        if (tauPerPt) tauPerPt->Write("tau_per_ptbin", TObject::kOverwrite);
    }

    if (fitFrames && !fitFrames->empty()) {
        TDirectory *fitDir = stdDir ? stdDir->GetDirectory("fit_frames") : nullptr;
        if (!fitDir && stdDir) fitDir = stdDir->mkdir("fit_frames");
        if (fitDir) {
            fitDir->cd();
            for (const auto &frame : *fitFrames) {
                if (frame) frame->Write(frame->GetName(), TObject::kOverwrite);
            }
        }
    }

    if (fitCanvases && !fitCanvases->empty()) {
        TDirectory *fitCanvasDir = stdDir ? stdDir->GetDirectory("fit_canvases") : nullptr;
        if (!fitCanvasDir && stdDir) fitCanvasDir = stdDir->mkdir("fit_canvases");
        if (fitCanvasDir) {
            fitCanvasDir->cd();
            for (const auto &canvas : *fitCanvases) {
                if (canvas) canvas->Write(canvas->GetName(), TObject::kOverwrite);
            }
        }
    }

    TDirectory *sysDir = dir->GetDirectory("sys");
    if (!sysDir) sysDir = dir->mkdir("sys");
    if (sysDir && sysArtifacts) {
        for (const auto &art : *sysArtifacts) {
            TDirectory *subDir = sysDir->GetDirectory(art.subBinName.c_str());
            if (!subDir) subDir = sysDir->mkdir(art.subBinName.c_str());
            if (!subDir) continue;
            subDir->cd();
            if (art.hCorrDist) art.hCorrDist->Write("h_corr_syst_dist", TObject::kOverwrite);
            if (art.hTrailBdtEff) art.hTrailBdtEff->Write("h_trail_bdt_eff", TObject::kOverwrite);
            if (art.hCorrVsAbso) art.hCorrVsAbso->Write("h_corr_vs_absorption_scale", TObject::kOverwrite);
            if (!art.trailFrames.empty()) {
                TDirectory *trailDir = subDir->GetDirectory("trails_FitFrames");
                if (!trailDir) trailDir = subDir->mkdir("trails_FitFrames");
                if (trailDir) {
                    trailDir->cd();
                    for (const auto &fr : art.trailFrames) {
                        if (fr) fr->Write(fr->GetName(), TObject::kOverwrite);
                    }
                }
            }
            // Ensure corr-dist canvas is stored directly under sys/subbin, not in trails_FitFrames.
            subDir->cd();
            if (art.cCorrDist) art.cCorrDist->Write(art.cCorrDist->GetName(), TObject::kOverwrite);
            if (art.cCorrVsAbso) art.cCorrVsAbso->Write(art.cCorrVsAbso->GetName(), TObject::kOverwrite);
        }
    }

    dir->cd();
    if (hFinalStat) hFinalStat->Write("h_final_spectrum_stat", TObject::kOverwrite);
    if (gFinalSys) gFinalSys->Write("g_final_spectrum_sys", TObject::kOverwrite);
    if (hSysSelection) hSysSelection->Write("h_sys_selection", TObject::kOverwrite);
    if (hSysAbsorption) hSysAbsorption->Write("h_sys_absorption", TObject::kOverwrite);
    if (hSysBranching) hSysBranching->Write("h_sys_branching_ratio", TObject::kOverwrite);
    if (hSysTotal) hSysTotal->Write("h_sys_total", TObject::kOverwrite);

    auto cSys = MakeSystematicsDetailCanvas("c_sys_details", hSysSelection, hSysAbsorption, hSysBranching, hSysTotal);
    if (cSys) {
        cSys->SetTitle(hSysTotal ? hSysTotal->GetTitle() : dirName.c_str());
        cSys->Write("c_sys_details", TObject::kOverwrite);
    }
    auto cFinal = MakeFinalSpectrumCanvas("c_final_spectrum",
                                          hFinalStat,
                                          gFinalSys,
                                          finalStdFit,
                                          finalXAxisTitle,
                                          finalExtraText,
                                          finalLabelCfg ? *finalLabelCfg : PlotLabelConfig{},
                                          finalIsMatter,
                                          finalNEvents);
    const bool isExpCtFit = finalStdFit && std::string(finalStdFit->GetName()).rfind("f_exp_", 0) == 0;
    if (cFinal && isExpCtFit) {
        constexpr double kSpeedOfLightCmPerPsLocal = 0.0299792458;
        const double tauCm = finalStdFit->GetParameter(1);
        const double tauCmErr = finalStdFit->GetParError(1);
        const double tauPs = tauCm / kSpeedOfLightCmPerPsLocal;
        const double tauPsErr = tauCmErr / kSpeedOfLightCmPerPsLocal;
        const double chi2 = finalStdFit->GetChisquare();
        const int ndf = finalStdFit->GetNDF();
        const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;

        cFinal->cd();
        TLatex latex;
        latex.SetNDC();
        latex.SetTextFont(42);
        latex.SetTextSize(0.035);
        latex.SetTextAlign(13);
        double y = 0.88;
        latex.DrawLatex(0.60, y, Form("#tau = %.1f #pm %.1f ps", tauPs, tauPsErr));
        y -= 0.05;
        latex.DrawLatex(0.60, y, Form("#chi^{2}/ndf = %.2f/%d", chi2, ndf));
        y -= 0.05;
        latex.DrawLatex(0.60, y, Form("Fit probability = %.3f", fitProb));
        cFinal->Modified();
        cFinal->Update();
    }
    if (cFinal) cFinal->Write("c_final_spectrum", TObject::kOverwrite);
}


void AppendCorrectionsCsv(const std::string &csvPath,
                          const std::string &mode,
                          const std::string &group,
                          const BinPlanItem &item,
                          const GeneralHelper::MassFitResult &fitRes,
                          const CorrectedCountsResult &corrRes,
                          double sysSelection,
                          double sysAbsorption,
                          double sysBranching,
                          double sysTotal) {
    const bool exists = std::filesystem::exists(csvPath);
    std::ofstream ofs(csvPath, std::ios::app);
    if (!ofs.is_open()) return;
    ofs << std::setprecision(12);

    if (!exists) {
        ofs << "mode,group,label,raw,raw_err,chi2_ndf,significance,corrected,corrected_err,";
        ofs << "bdt_efficiency,acceptance,absorption,event_loss,event_splitting,signal_loss,branching_ratio,delta_rapidity,matter_ratio,n_events,bin_width,";
        ofs << "syst_selection_abs,syst_absorption_abs,syst_branching_abs,syst_total_abs,";
        ofs << "syst_selection_ratio,syst_absorption_ratio,syst_branching_ratio,syst_total_ratio\n";
    }

    const double den = std::abs(corrRes.value);
    const auto ratio = [den](double v) { return den > 0.0 ? v / den : 0.0; };
    ofs << mode << ',' << group << ',' << item.label << ','
        << fitRes.signal << ',' << fitRes.signalErr << ','
        << fitRes.chi2Data << ',' << fitRes.significance << ','
        << corrRes.value << ',' << corrRes.error << ','
        << corrRes.bdtEfficiency << ','
        << corrRes.acceptance << ','
        << corrRes.absorption << ','
        << corrRes.eventLoss << ','
        << corrRes.eventSplitting << ','
        << corrRes.signalLoss << ','
        << corrRes.branchingRatio << ','
        << corrRes.deltaRapidity << ','
        << corrRes.matterRatio << ','
        << corrRes.nEvents << ','
        << corrRes.binWidth << ','
        << sysSelection << ','
        << sysAbsorption << ','
        << sysBranching << ','
        << sysTotal << ','
        << ratio(sysSelection) << ','
        << ratio(sysAbsorption) << ','
        << ratio(sysBranching) << ','
        << ratio(sysTotal) << '\n';
}

void AppendIntegratedYieldCsvRow(std::ofstream &ofs,
                                 double cenMin,
                                 double cenMax,
                                 double value,
                                 double statErr,
                                 double systExtrap,
                                 double systFitFunc,
                                 double systAbsorption,
                                 double systTrails,
                                 double systBranching,
                                 double systTotal) {
    const double den = std::abs(value);
    const auto ratio = [den](double v) { return den > 0.0 ? v / den : 0.0; };
    ofs << cenMin << ',' << cenMax << ','
        << value << ',' << statErr << ','
        << systExtrap << ',' << systFitFunc << ',' << systAbsorption << ','
        << systTrails << ',' << systBranching << ',' << systTotal << ','
        << ratio(statErr) << ','
        << ratio(systExtrap) << ',' << ratio(systFitFunc) << ',' << ratio(systAbsorption) << ','
        << ratio(systTrails) << ',' << ratio(systBranching) << ',' << ratio(systTotal) << '\n';
}

void WriteOnTheFlyChecksOutput(const std::string &rootFilePath,
                               const std::vector<std::string> &qaColumns,
                               const std::vector<ProcessOneBinResult::H3lQAVars> &qaCandidates,
                               const std::vector<std::string> &variables,
                               const std::vector<Hist2DPair> &pairs,
                               const std::unordered_map<std::string, AxisSpec> &axisPool,
                               bool savePdf,
                               const std::string &pdfBaseDir) {
    if (qaColumns.empty()) return;

    const std::filesystem::path baseDir = std::filesystem::path(rootFilePath).parent_path();
    std::filesystem::path pdfDir;
    if (savePdf) {
        std::error_code ec;
        std::filesystem::remove_all(std::filesystem::path(pdfBaseDir), ec);
        pdfDir = std::filesystem::path(pdfBaseDir);
    }

    std::filesystem::create_directories(baseDir);
    if (savePdf) std::filesystem::create_directories(pdfDir);

    TFile checksFile(rootFilePath.c_str(), "RECREATE");
    if (checksFile.IsZombie()) {
        std::cerr << "[on_the_fly_checks] cannot open output ROOT: " << rootFilePath << std::endl;
        return;
    }

    const std::string dirName = "all_candidates";
    TDirectory *groupDir = checksFile.GetDirectory(dirName.c_str());
    if (groupDir) {
        checksFile.cd();
        checksFile.Delete((dirName + ";*").c_str());
    }
    groupDir = checksFile.mkdir(dirName.c_str());
    if (!groupDir) {
        checksFile.Close();
        return;
    }

    groupDir->cd();
    std::unordered_map<std::string, size_t> colIndex;
    for (size_t i = 0; i < qaColumns.size(); ++i) colIndex[qaColumns[i]] = i;

    for (const auto &var : variables) {
        auto it = colIndex.find(var);
        if (it == colIndex.end()) continue;
        const auto ax = ResolveQaAxis(var, axisPool);
        const std::string xTitle = ax.title.empty() ? var : ax.title;
        const std::string hTitle = ";" + xTitle + ";Counts";
        TH1D h(("h1_" + var).c_str(), hTitle.c_str(), ax.nBins, ax.min, ax.max);
        h.SetDirectory(nullptr);
        for (const auto &row : qaCandidates) {
            if (it->second < row.values.size()) h.Fill(row.values[it->second]);
        }
        h.Write("", TObject::kOverwrite);

        if (savePdf) {
            TCanvas c(("c_" + std::string(h.GetName())).c_str(), h.GetTitle(), 900, 700);
            h.Draw("HIST");
            c.SaveAs((pdfDir / (std::string(h.GetName()) + ".pdf")).string().c_str());
        }
    }

    for (const auto &p : pairs) {
        auto ix = colIndex.find(p.x);
        auto iy = colIndex.find(p.y);
        if (ix == colIndex.end() || iy == colIndex.end()) continue;
        const auto ax = ResolveQaAxis(p.x, axisPool);
        const auto ay = ResolveQaAxis(p.y, axisPool);
        const std::string hname = "h2_" + p.x + "_vs_" + p.y;
        const std::string xTitle = ax.title.empty() ? p.x : ax.title;
        const std::string yTitle = ay.title.empty() ? p.y : ay.title;
        const std::string hTitle = ";" + xTitle + ";" + yTitle + ";Counts";
        TH2D h2(hname.c_str(), hTitle.c_str(), ax.nBins, ax.min, ax.max, ay.nBins, ay.min, ay.max);
        h2.SetDirectory(nullptr);
        for (const auto &row : qaCandidates) {
            if (ix->second < row.values.size() && iy->second < row.values.size()) {
                h2.Fill(row.values[ix->second], row.values[iy->second]);
            }
        }
        h2.Write("", TObject::kOverwrite);

        if (savePdf) {
            TCanvas c(("c_" + hname).c_str(), hname.c_str(), 900, 700);
            h2.Draw("COLZ");
            c.SaveAs((pdfDir / (hname + ".pdf")).string().c_str());
        }
    }

    // Also persist the filtered QA candidates as a tree in the same output file.
    groupDir->cd();
    TTree tree("hypertriton_qa_all", "hypertriton_qa_all");
    std::vector<double> scalars(qaColumns.size(), 0.0);
    for (size_t ic = 0; ic < qaColumns.size(); ++ic) {
        tree.Branch(qaColumns[ic].c_str(), &scalars[ic]);
    }

    for (const auto &row : qaCandidates) {
        const size_t n = std::min(scalars.size(), row.values.size());
        for (size_t ic = 0; ic < n; ++ic) scalars[ic] = row.values[ic];
        for (size_t ic = n; ic < scalars.size(); ++ic) scalars[ic] = 0.0;
        tree.Fill();
    }
    tree.Write("", TObject::kOverwrite);

    checksFile.Close();
}

SysBinArtifacts RunSystematicsForBin(const RunConfig &runCfg,
                                     const GroupContext &group,
                                     const BinPlanItem &item,
                                     const std::unordered_map<std::string, std::shared_ptr<ROOT::RDataFrame>> &rdfCache,
                                     double stdCorr,
                                     double stdCorrErr,
                                     double acc,
                                     double abso,
                                     double eventLoss,
                                     double signalLoss,
                                     double nEvents,
                                     int histBin,
                                     std::ofstream *trailLog,
                                     const std::map<std::string, std::vector<double>> *absoVecByFile = nullptr,
                                     bool enableAbsorptionSystematic = true) {
    SysBinArtifacts out;
    out.subBinName = MakeSysSubBinName(runCfg.mode, item);

    const auto wp = GetWorkingPointForItem(runCfg.mode, runCfg.wpFile, item);
    const double wpScore = wp.found ? wp.score : 0.0;
    const double wpEff = wp.found ? wp.eff : 1.0;

    const std::string scoreArrPath = runCfg.scoreEffDir.empty()
        ? std::string()
        : (runCfg.scoreEffDir + "/score_efficiency_array_" + item.label + ".txt");
    const auto scoreEffArray = LoadScoreEfficiencyArray(scoreArrPath);
    auto bdtCandidates = BuildBdtCandidates(wpScore, wpEff, runCfg.systBdtScoreNPoints, scoreEffArray);
    if (bdtCandidates.empty()) bdtCandidates.push_back({wpScore, wpEff});

    const std::vector<std::string> bkgFuncs = runCfg.systBkgFuncs.empty()
        ? std::vector<std::string>{runCfg.bkgFunc}
        : runCfg.systBkgFuncs;
    const std::vector<std::string> sigFuncs = runCfg.systSigFuncs.empty()
        ? std::vector<std::string>{runCfg.sigFunc}
        : runCfg.systSigFuncs;

    auto combos = BuildTrailCombos(bdtCandidates.size(), bkgFuncs.size(), sigFuncs.size());
    std::mt19937 rng(static_cast<unsigned>(runCfg.randomSeed + histBin));
    std::shuffle(combos.begin(), combos.end(), rng);
    const size_t nUse = std::min(combos.size(), static_cast<size_t>(std::max(1, runCfg.systNtrails)));

    const int nBinsCorrDist = std::max(10, runCfg.nBinsForFit);
    out.hCorrDist = std::make_unique<TH1D>(("h_corr_syst_" + item.label).c_str(), ";Corrected counts;Entries", nBinsCorrDist,
                                           stdCorr - 5.0 * std::max(1e-12, stdCorrErr),
                                           stdCorr + 5.0 * std::max(1e-12, stdCorrErr));
    out.hCorrDist->SetDirectory(nullptr);
    out.hCorrDist->SetStats(false);
    const std::string simpleBinTitle = BuildSimpleBinTitle(item);
    out.hCorrDist->SetTitle((simpleBinTitle + ";Corrected counts;Entries").c_str());
    out.hTrailBdtEff = std::make_unique<TH1D>(("h_trail_bdt_eff_" + item.label).c_str(), ";trail index;BDT efficiency", static_cast<int>(nUse), 0.5, static_cast<double>(nUse) + 0.5);
    out.hTrailBdtEff->SetDirectory(nullptr);
    out.hTrailBdtEff->SetStats(false);
    out.hTrailBdtEff->SetTitle((simpleBinTitle + ";trail index;BDT efficiency").c_str());
    out.corrValuesByTrial.assign(nUse, std::numeric_limits<double>::quiet_NaN());

    auto itData = rdfCache.find(runCfg.dataTree + "@" + item.snapshotDataPath);
    auto itMc = rdfCache.find(runCfg.mcTree + "@" + item.snapshotMcPath);
    if (itData == rdfCache.end() || itMc == rdfCache.end()) {
        return out;
    }

    std::vector<std::string> selParts;
    if (!runCfg.centralitySelection.empty()) selParts.push_back(runCfg.centralitySelection);
    if (!runCfg.additionalDataSelectionGeneral.empty()) selParts.push_back(runCfg.additionalDataSelectionGeneral);
    if (!runCfg.additionalDataSelection.empty()) selParts.push_back(runCfg.additionalDataSelection);
    if (!item.topologySelection.empty()) selParts.push_back(item.topologySelection);
    if (runCfg.isMatter == "matter") selParts.push_back("fIsMatter > 0");
    else if (runCfg.isMatter == "antimatter") selParts.push_back("fIsMatter <= 0");
    std::string dataSelectionExpr;
    if (!selParts.empty()) {
        std::ostringstream expr;
        for (size_t j = 0; j < selParts.size(); ++j) {
            if (j) expr << " && ";
            expr << "(" << selParts[j] << ")";
        }
        dataSelectionExpr = expr.str();
    }

    std::vector<ROOT::RDF::RResultPtr<std::vector<double>>> dataMassCache;
    dataMassCache.reserve(bdtCandidates.size());
    for (const auto &cand : bdtCandidates) {
        std::ostringstream bdtCutExpr;
        bdtCutExpr << "model_output > " << cand.score;
        if (dataSelectionExpr.empty()) {
            dataMassCache.emplace_back(itData->second->Filter(bdtCutExpr.str()).Take<double>("fMassH3L"));
        } else {
            dataMassCache.emplace_back(itData->second->Filter(dataSelectionExpr).Filter(bdtCutExpr.str()).Take<double>("fMassH3L"));
        }
    }
    std::string mcMassSel = "fMassH3L>2.95 && fMassH3L<3.02";
    if (runCfg.isMatter == "matter") mcMassSel += " && fIsMatter > 0";
    else if (runCfg.isMatter == "antimatter") mcMassSel += " && fIsMatter <= 0";
    auto mcMass = itMc->second->Filter(mcMassSel).Take<double>("fMassH3L");

    auto isHigherOrderPolynomial = [](const std::string &name) {
        return name == "pol2" || name == "pol3" || name == "pol4";
    };
    auto hasCleanSidebands = [](const std::vector<double> &masses,
                                const GeneralHelper::MassFitConfig &fitCfg) {
        int nSignalWindow = 0;
        int nSideband = 0;
        constexpr double signalLow = 2.985;
        constexpr double signalHigh = 2.997;
        for (double m : masses) {
            if (m < fitCfg.massMin || m > fitCfg.massMax) continue;
            if (m >= signalLow && m <= signalHigh) {
                ++nSignalWindow;
            } else {
                ++nSideband;
            }
        }
        if (nSignalWindow < 3) return false;
        return nSideband <= 5 ||
               static_cast<double>(nSideband) / static_cast<double>(std::max(1, nSignalWindow)) < 0.25;
    };

    for (size_t i = 0; i < nUse; ++i) {
        const auto [ibdt, ibkg, isig] = combos[i];
        const auto &trialDataMasses = *dataMassCache[ibdt];
        std::string trialBkgFunc = bkgFuncs[ibkg];
        if (isHigherOrderPolynomial(trialBkgFunc) && hasCleanSidebands(trialDataMasses, runCfg.fitCfg)) {
            trialBkgFunc = "pol1";
        }

        GeneralHelper::MassFitResult trialFit;
        try {
            trialFit = GeneralHelper::FitMassSpectrum(trialDataMasses, *mcMass, runCfg.fitCfg, trialBkgFunc, sigFuncs[isig]);
            if (!std::isfinite(trialFit.chi2Data) && trialBkgFunc != "pol1" && isHigherOrderPolynomial(bkgFuncs[ibkg])) {
                trialBkgFunc = "pol1";
                trialFit = GeneralHelper::FitMassSpectrum(trialDataMasses, *mcMass, runCfg.fitCfg, trialBkgFunc, sigFuncs[isig]);
            }
        } catch (const std::exception &) {
            continue;
        }

        const int barLen = 20;
        const int filled = static_cast<int>(std::round(static_cast<double>(i + 1) / static_cast<double>(nUse) * barLen));
        std::string bar(static_cast<size_t>(std::max(0, filled)), '<');
        std::cout << "\r[Info] Bin " << item.label << " trails " << (i + 1) << "/" << nUse << " " << bar << std::flush;

        const auto corrRes = ComputeCorrectedCounts(
            runCfg.mode,
            item,
            trialFit,
            acc,
            abso,
            bdtCandidates[ibdt].efficiency,
            eventLoss,
            1.0,
            signalLoss,
            nEvents,
            runCfg.branchingRatio,
            runCfg.deltaRap,
            runCfg.isMatter,
            runCfg.addAbsorptionCorrectionPtCt);
        const double corr = corrRes.value;
        if (std::isfinite(corr) && i < out.corrValuesByTrial.size()) {
            out.corrValuesByTrial[i] = corr;
        }

        const double corrErr = corrRes.error;
        if (out.hTrailBdtEff) out.hTrailBdtEff->SetBinContent(static_cast<int>(i + 1), bdtCandidates[ibdt].efficiency);

        if (trialFit.frame) {
            const std::string frameName = "trail_frame_" + item.label + "_" + std::to_string(i);
            auto *cloned = dynamic_cast<RooPlot *>(trialFit.frame->Clone(frameName.c_str()));
            if (cloned) out.trailFrames.emplace_back(cloned);
        }

        const bool pass = std::isfinite(corr) &&
                          std::isfinite(trialFit.chi2Data) &&
                          trialFit.chi2Data < runCfg.systThrChi2Ndf;
        if (pass) {
            if (out.hCorrDist) out.hCorrDist->Fill(corr);
            out.acceptedCorrValues.push_back(corr);
        }

        if (trailLog && trailLog->is_open()) {
            (*trailLog)
                << (i + 1) << ','
                << group.dirName << ','
                << item.label << ','
                << bdtCandidates[ibdt].score << ','
                << bdtCandidates[ibdt].efficiency << ','
                << trialBkgFunc << ','
                << sigFuncs[isig] << ','
                << trialFit.chi2Data << ','
                << trialFit.significance << ','
                << trialFit.signal << ','
                << trialFit.signalErr << ','
                << corr << ','
                << corrErr << ','
                << (pass ? 1 : 0)
                << '\n';
        }
    }
    std::cout << std::endl;
    out.selectionRms = (out.hCorrDist && out.hCorrDist->GetEntries() > 2) ? out.hCorrDist->GetRMS() : 0.0;
    if (out.hCorrDist) {
        const std::string cName = "c_corr_syst_" + item.label;
        out.cCorrDist = MakeSystematicsCorrDistCanvas(cName,
                                                      out.hCorrDist.get(),
                                                      stdCorr,
                                                      stdCorrErr,
                                                      item.cenMin,
                                                      item.cenMax,
                                                      simpleBinTitle,
                                                      runCfg.perPtBinGaussFitMaxChi2Ndf,
                                                      &out.selectionRms);
    }

    if (!enableAbsorptionSystematic) {
        out.hCorrVsAbso = std::make_unique<TH1D>(("h_corr_vs_abso_" + item.label).c_str(),
                                                 ";n x #sigma_{He3};Corrected counts",
                                                 1,
                                                 0.5,
                                                 1.5);
        out.hCorrVsAbso->SetDirectory(nullptr);
        out.hCorrVsAbso->SetStats(false);
        out.hCorrVsAbso->SetTitle((simpleBinTitle + ";n x #sigma_{He3};Corrected counts").c_str());
        out.hCorrVsAbso->GetXaxis()->SetBinLabel(1, "disabled");
        out.hCorrVsAbso->SetBinContent(1, stdCorr);
        out.absorptionRms = 0.0;
        return out;
    }

    std::vector<std::string> absoFiles = runCfg.systAbsorptionFiles;
    if (absoFiles.empty() && !runCfg.mcFileForAbsorption.empty()) {
        absoFiles.push_back(runCfg.mcFileForAbsorption);
    }
    if (!runCfg.mcFileForAbsorption.empty() &&
        std::find(absoFiles.begin(), absoFiles.end(), runCfg.mcFileForAbsorption) == absoFiles.end()) {
        absoFiles.push_back(runCfg.mcFileForAbsorption);
    }

    out.hCorrVsAbso = std::make_unique<TH1D>(("h_corr_vs_abso_" + item.label).c_str(), ";n x #sigma_{He3};Corrected counts", static_cast<int>(absoFiles.size()), 0.5, static_cast<double>(absoFiles.size()) + 0.5);
    out.hCorrVsAbso->SetDirectory(nullptr);
    out.hCorrVsAbso->SetStats(false);
    out.hCorrVsAbso->SetTitle((simpleBinTitle + ";n x #sigma_{He3};Corrected counts").c_str());
    std::vector<double> corrVariants;
    for (size_t iabso = 0; iabso < absoFiles.size(); ++iabso) {
        const auto &absoFile = absoFiles[iabso];
        const std::vector<double> *absoVec = nullptr;
        if (absoVecByFile) {
            auto it = absoVecByFile->find(absoFile);
            if (it != absoVecByFile->end()) absoVec = &it->second;
        }
        const int binx = static_cast<int>(iabso + 1);
        std::string label = (iabso < runCfg.systAbsorptionFileLabels.size() && !runCfg.systAbsorptionFileLabels[iabso].empty())
            ? runCfg.systAbsorptionFileLabels[iabso]
            : ("var" + std::to_string(iabso));
        out.hCorrVsAbso->GetXaxis()->SetBinLabel(binx, label.c_str());
        if (!absoVec || static_cast<size_t>(histBin - 1) >= absoVec->size()) continue;
        const double absoVar = (*absoVec)[static_cast<size_t>(histBin - 1)];
        if (absoVar <= 0.0 || abso <= 0.0) continue;
        const double corrVar = stdCorr * (abso / absoVar);
        out.hCorrVsAbso->SetBinContent(binx, corrVar);
        corrVariants.push_back(corrVar);
    }
    if (!corrVariants.empty()) {
        const auto mm = std::minmax_element(corrVariants.begin(), corrVariants.end());
        out.absorptionRms = std::max(0.0, runCfg.absorptionLength) * (*mm.second - *mm.first);
    }

    if (out.hCorrVsAbso) {
        const std::string cName = "c_absorption_source_" + item.label;
        out.cCorrVsAbso = std::make_unique<TCanvas>(cName.c_str(), cName.c_str(), 960, 720);
        out.cCorrVsAbso->SetTitle(simpleBinTitle.c_str());
        out.cCorrVsAbso->cd();
        out.cCorrVsAbso->SetLeftMargin(0.14);
        out.cCorrVsAbso->SetBottomMargin(0.18);
        out.cCorrVsAbso->SetRightMargin(0.05);
        out.cCorrVsAbso->SetTopMargin(0.08);
        out.cCorrVsAbso->SetTicks(1, 1);

        out.hCorrVsAbso->SetLineWidth(2);
        out.hCorrVsAbso->SetLineColor(kBlue + 1);
        out.hCorrVsAbso->GetXaxis()->LabelsOption("v");
        out.hCorrVsAbso->GetXaxis()->SetLabelSize(0.04);
        out.hCorrVsAbso->GetYaxis()->SetTitleOffset(1.35);
        out.hCorrVsAbso->Draw("HIST");

        out.cCorrVsAbso->Modified();
        out.cCorrVsAbso->Update();
    }
    return out;
}

struct BlastWavePostFit {
    std::unique_ptr<TF1> fit;
    std::unique_ptr<TCanvas> canvas;
};

BlastWavePostFit BuildBlastWavePostFit(const GroupContext &group,
                                       TH1D *hCorr,
                                       const std::map<std::string, SpectrumFunctionConfig> &fitParameters) {
    BlastWavePostFit out;
    if (!hCorr || group.items.empty()) return out;

    const double fitMin = group.items.front().ptMin;
    const double fitMax = group.items.back().ptMax;
    if (!(fitMax > fitMin)) return out;

    const std::string cleanName = SanitizeName(group.dirName);
    auto raw = BuildSpectrumFunction("fBGBW", cleanName + "_std_bgbw", fitParameters);
    if (!raw) return out;
    FitHistogramWithFunction(hCorr, raw.get(), fitMin, fitMax);
    raw->SetName(("f_bgbw_" + cleanName).c_str());
    out.fit = std::move(raw);

    out.fit->SetRange(0.0, 10.0);
    out.canvas = MakeBlastWaveFitCanvas("c_bgbw_fit_" + cleanName,
                                        hCorr,
                                        out.fit.get(),
                                        group.items.front().cenMin,
                                        group.items.front().cenMax);
    return out;
}

struct ExpoPostFit {
    std::unique_ptr<TF1> fit;
    std::unique_ptr<TCanvas> canvas;
    bool hasTau{false};
    double tauPs{0.0};
    double tauPsErr{0.0};
};

ExpoPostFit BuildExponentialPostFit(const std::string &tag,
                                    TH1D *hCorr,
                                    double fitMin,
                                    double fitMax,
                                    const std::string &xTitle) {
    ExpoPostFit out;
    if (!hCorr || !(fitMax > fitMin)) return out;

    const std::string cleanName = SanitizeName(tag);
    out.fit = std::make_unique<TF1>(("f_exp_" + cleanName).c_str(), "[0]*exp(-x/[1])", fitMin, fitMax);
    out.fit->SetParName(0, "N_{0}");
    out.fit->SetParName(1, "ct");
    out.fit->SetParameter(0, std::max(1.0, hCorr->GetMaximum()));
    out.fit->SetParameter(1, 8.0);
    out.fit->SetParLimits(0, 0.0, std::max(1e9, hCorr->GetMaximum() * 1e4));
    out.fit->SetParLimits(1, 1e-6, 1e4);

    hCorr->Fit(out.fit.get(), "QIS0R");
    const double tauCm = out.fit->GetParameter(1);
    const double tauCmErr = out.fit->GetParError(1);
    out.tauPs = tauCm / kSpeedOfLightCmPerPs;
    out.tauPsErr = tauCmErr / kSpeedOfLightCmPerPs;
    out.hasTau = std::isfinite(out.tauPs) && std::isfinite(out.tauPsErr);

    out.canvas = MakeExponentialFitCanvas("c_exp_fit_" + cleanName,
                                          hCorr,
                                          out.fit.get(),
                                          xTitle,
                                          out.tauPs,
                                          out.tauPsErr);
    return out;
}

int RunSpectrumMode(const GeneralHelper::Json &cfg,
                    const BinPlan &plan,
                    const std::string &modeName) {
    if (plan.items.empty()) {
        throw std::runtime_error("Bin plan is empty for mode: " + modeName);
    }

    const RunConfig runCfg = BuildRunConfig(cfg, modeName);
    const auto corrCfg = BuildCorrectionConfig(runCfg);
    const std::vector<GroupContext> groups = BuildGroups(modeName, plan);
    const std::string outputDir = std::filesystem::path(runCfg.outputRoot).parent_path().string();
    const std::string csvPath = outputDir + "/corrections_all.csv";
    const std::string integratedYieldCsvPath = outputDir + "/integrated_yield_summary.csv";

    // Clear previous CSV before appending new rows for this run.
    {
        std::error_code ec;
        std::filesystem::remove(csvPath, ec);
        std::filesystem::remove(integratedYieldCsvPath, ec);
    }

    const bool isSpectrumMode = (modeName == "bdt_spectrum" || modeName == "topology_spectrum");
    const bool isLifetimeMode = (modeName == "ct_single" || modeName == "pt_ct");
    const bool applyBrSystematic = runCfg.doSystematics &&
                                   (isSpectrumMode || isLifetimeMode) &&
                                   runCfg.branchingRatioFractionalUncertainty > 0.0;
    const std::string periodTag = runCfg.plotLabels.periodMark.empty()
                                      ? runCfg.plotLabels.period
                                      : runCfg.plotLabels.period + "_" + runCfg.plotLabels.periodMark;
    double totalRawOverNevts = 0.0;
    double totalEventsOverNevts = 0.0;
    double totalRawErr2OverNevts = 0.0;

    std::filesystem::create_directories(std::filesystem::path(runCfg.outputRoot).parent_path());
    TFile fout(runCfg.outputRoot.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        throw std::runtime_error("Cannot open output ROOT: " + runCfg.outputRoot);
    }

    PtCtAcceptanceCache ptCtAccCache;
    PtCtAbsorptionCache ptCtAbsoCache;
    SpectrumAcceptanceCache spectrumAccCache;
    SpectrumEventSignalLossCache spectrumEvtSigLossCache;
    if (modeName == "pt_ct") {
        ptCtAccCache = BuildPtCtAcceptanceCache(corrCfg, runCfg.mcFileForAcceptance);
        if (runCfg.addAbsorptionCorrectionPtCt) {
            ptCtAbsoCache = BuildPtCtAbsorptionCache(corrCfg, runCfg.mcFileForAbsorption);
        }
    } else if (isSpectrumMode) {
        const std::string mcFileForAccSpectrum = ResolveAcceptanceMcFileForGroup(runCfg, GroupContext{});
        spectrumAccCache = BuildSpectrumAcceptanceCache(corrCfg, mcFileForAccSpectrum);
        spectrumEvtSigLossCache = BuildSpectrumEventSignalLossCache(corrCfg,
                                                                    runCfg.eventSignalLossFile,
                                                                    runCfg.mcFileForAcceptance,
                                                                    runCfg.addEventSignalLossCenPt);
    }

    std::unordered_map<std::string, std::shared_ptr<ROOT::RDataFrame>> rdfCache;
    std::vector<std::string> onTheFlyQaColumns = runCfg.oneBinQaColumns;
    std::vector<ProcessOneBinResult::H3lQAVars> onTheFlyQaCandidates;

    std::ofstream trailLog;
    if (runCfg.doSystematics) {
        const std::filesystem::path sysOutDir = std::filesystem::path(runCfg.outputRoot).parent_path() / "sys";
        std::filesystem::create_directories(sysOutDir);
        const std::filesystem::path trailLogPath = sysOutDir / "trails.log";
        trailLog.open(trailLogPath.string(), std::ios::out | std::ios::trunc);
        if (trailLog.is_open()) {
            trailLog << "trail,group,label,bdt_score,bdt_eff,bkg_func,sig_func,chi2ndf,significance,raw,raw_err,corr,corr_err,pass\n";
        }
    }

    std::unique_ptr<TH1D> hTauPerPt;
    if (modeName == "pt_ct" && runCfg.ptBins.size() >= 2) {
        hTauPerPt = std::make_unique<TH1D>("tau_per_ptbin",
                                           ";#it{p}_{T} (GeV/#it{c});#tau (ps)",
                                           static_cast<int>(runCfg.ptBins.size() - 1),
                                           runCfg.ptBins.data());
        hTauPerPt->SetDirectory(nullptr);
        hTauPerPt->SetStats(false);
    }

    struct IntegralSummaryRow {
        double cenMin{0.0};
        double cenMax{0.0};
        double value{0.0};
        double statErr{0.0};
        double systExtrap{0.0};
        double systFitFunc{0.0};
        double systAbsorption{0.0};
        double systTrails{0.0};
        double systBranching{0.0};
        double systTotal{0.0};
    };
    std::vector<IntegralSummaryRow> integralSummaryRows;

    struct IntegralFitParameterSummaryRow {
        double cenMin{0.0};
        double cenMax{0.0};
        IntegralFitParameterRow fitRow;
    };
    std::vector<IntegralFitParameterSummaryRow> integralFitParameterRows;

    for (const auto &group : groups) {
        const auto edges = BuildAxisEdges(modeName, group.items);
        if (edges.size() < 2) continue;

        double rawSumGroup = 0.0;
        double rawSumGroupErr2 = 0.0;

        const bool useCtAxis = (modeName == "ct_single" || modeName == "pt_ct");
        auto hRaw = std::make_unique<TH1D>("h_raw_counts", useCtAxis ? ";ct;N_{raw}" : ";p_{T};N_{raw}",
                                           static_cast<int>(edges.size() - 1), edges.data());
        auto hCorr = std::make_unique<TH1D>("h_corrected_counts", useCtAxis ? ";ct;N_{corr}" : ";p_{T};N_{corr}",
                                            static_cast<int>(edges.size() - 1), edges.data());
        auto hBdtEff = std::make_unique<TH1D>("h_bdt_efficiency", useCtAxis ? ";ct;BDT eff" : ";p_{T};BDT eff",
                                              static_cast<int>(edges.size() - 1), edges.data());
        auto hAcc = std::make_unique<TH1D>("h_acceptance", useCtAxis ? ";ct;Accptance #times efficency" : ";p_{T};Accptance #times efficency",
                           static_cast<int>(edges.size() - 1), edges.data());
        auto hAbso = std::make_unique<TH1D>("h_absorption", useCtAxis ? ";ct;#epsilon_{abso}" : ";p_{T};#epsilon_{abso}",
                            static_cast<int>(edges.size() - 1), edges.data());
        auto hEventLoss = std::make_unique<TH1D>("h_event_loss", useCtAxis ? ";ct;Event loss / event splitting" : ";p_{T};Event loss / event splitting",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hEventSplitting = std::make_unique<TH1D>("h_event_splitting", useCtAxis ? ";ct;Event splitting" : ";p_{T};Event splitting",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hSignalLoss = std::make_unique<TH1D>("h_signal_loss", useCtAxis ? ";ct;Signal loss correction" : ";p_{T};Signal loss correction",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hFitSigma = std::make_unique<TH1D>("h_fit_sigma", useCtAxis ? ";ct;#sigma_{fit}" : ";p_{T};#sigma_{fit}",
                     static_cast<int>(edges.size() - 1), edges.data());
        auto hFitMean = std::make_unique<TH1D>("h_fit_mean", useCtAxis ? ";ct;#mu_{fit}" : ";p_{T};#mu_{fit}",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hFitChi2 = std::make_unique<TH1D>("h_fit_chi2ndf", useCtAxis ? ";ct;#chi^{2}/ndf" : ";p_{T};#chi^{2}/ndf",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hSysSelection = std::make_unique<TH1D>("h_sys_selection", useCtAxis ? ";ct;#sigma_{syst}^{selection}" : ";p_{T};#sigma_{syst}^{selection}",
                               static_cast<int>(edges.size() - 1), edges.data());
        auto hSysAbsorption = std::make_unique<TH1D>("h_sys_absorption", useCtAxis ? ";ct;#sigma_{syst}^{abso}" : ";p_{T};#sigma_{syst}^{abso}",
                            static_cast<int>(edges.size() - 1), edges.data());
        auto hSysBranching = std::make_unique<TH1D>("h_sys_branching_ratio", useCtAxis ? ";ct;#sigma_{syst}^{BR}" : ";p_{T};#sigma_{syst}^{BR}",
                    static_cast<int>(edges.size() - 1), edges.data());
        auto hSysTotal = std::make_unique<TH1D>("h_sys_total", useCtAxis ? ";ct;#sigma_{syst}^{total}" : ";p_{T};#sigma_{syst}^{total}",
                             static_cast<int>(edges.size() - 1), edges.data());
        const std::string groupBinTitle = BuildSimpleGroupTitle(group.items, group.key);
        hRaw->SetTitle((groupBinTitle + (useCtAxis ? ";ct;N_{raw}" : ";p_{T};N_{raw}")).c_str());
        hCorr->SetTitle((groupBinTitle + (useCtAxis ? ";ct;N_{corr}" : ";p_{T};N_{corr}")).c_str());
        hBdtEff->SetTitle((groupBinTitle + (useCtAxis ? ";ct;BDT eff" : ";p_{T};BDT eff")).c_str());
        hAcc->SetTitle((groupBinTitle + (useCtAxis ? ";ct;Accptance #times efficency" : ";p_{T};Accptance #times efficency")).c_str());
        hAbso->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#epsilon_{abso}" : ";p_{T};#epsilon_{abso}")).c_str());
        hEventLoss->SetTitle((groupBinTitle + (useCtAxis ? ";ct;Event loss / event splitting" : ";p_{T};Event loss / event splitting")).c_str());
        hEventSplitting->SetTitle((groupBinTitle + (useCtAxis ? ";ct;Event splitting" : ";p_{T};Event splitting")).c_str());
        hSignalLoss->SetTitle((groupBinTitle + (useCtAxis ? ";ct;Signal loss correction" : ";p_{T};Signal loss correction")).c_str());
        hFitSigma->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#sigma_{fit}" : ";p_{T};#sigma_{fit}")).c_str());
        hFitMean->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#mu_{fit}" : ";p_{T};#mu_{fit}")).c_str());
        hFitChi2->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#chi^{2}/ndf" : ";p_{T};#chi^{2}/ndf")).c_str());
        hSysSelection->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#sigma_{syst}^{selection}" : ";p_{T};#sigma_{syst}^{selection}")).c_str());
        hSysAbsorption->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#sigma_{syst}^{abso}" : ";p_{T};#sigma_{syst}^{abso}")).c_str());
        hSysBranching->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#sigma_{syst}^{BR}" : ";p_{T};#sigma_{syst}^{BR}")).c_str());
        hSysTotal->SetTitle((groupBinTitle + (useCtAxis ? ";ct;#sigma_{syst}^{total}" : ";p_{T};#sigma_{syst}^{total}")).c_str());
        if (modeName == "ct_single") {
            // ct_single group-level summary histograms should be generic (no first-bin title).
            hRaw->SetTitle(";ct;N_{raw}");
            hCorr->SetTitle(";ct;N_{corr}");
            hBdtEff->SetTitle(";ct;BDT eff");
            hAcc->SetTitle(";ct;Acceptance #times efficiency");
            hAbso->SetTitle(";ct;#epsilon_{abso}");
            hEventLoss->SetTitle(";ct;Event loss / event splitting");
            hEventSplitting->SetTitle(";ct;Event splitting");
            hSignalLoss->SetTitle(";ct;Signal loss correction");
            hFitSigma->SetTitle(";ct;#sigma_{fit}");
            hFitMean->SetTitle(";ct;#mu_{fit}");
            hFitChi2->SetTitle(";ct;#chi^{2}/ndf");
            hSysSelection->SetTitle(";ct;#sigma_{syst}^{selection}");
            hSysAbsorption->SetTitle(";ct;#sigma_{syst}^{abso}");
            hSysBranching->SetTitle(";ct;#sigma_{syst}^{BR}");
            hSysTotal->SetTitle(";ct;#sigma_{syst}^{total}");
        }
        hRaw->SetDirectory(nullptr);
        hCorr->SetDirectory(nullptr);
        hBdtEff->SetDirectory(nullptr);
        hAcc->SetDirectory(nullptr);
        hAbso->SetDirectory(nullptr);
        hEventLoss->SetDirectory(nullptr);
        hEventSplitting->SetDirectory(nullptr);
        hSignalLoss->SetDirectory(nullptr);
        hFitSigma->SetDirectory(nullptr);
        hFitMean->SetDirectory(nullptr);
        hFitChi2->SetDirectory(nullptr);
        hSysSelection->SetDirectory(nullptr);
        hSysAbsorption->SetDirectory(nullptr);
        hSysTotal->SetDirectory(nullptr);
        hRaw->SetStats(false);
        hCorr->SetStats(false);
        hBdtEff->SetStats(false);
        hAcc->SetStats(false);
        hAbso->SetStats(false);
        hEventLoss->SetStats(false);
        hEventSplitting->SetStats(false);
        hSignalLoss->SetStats(false);
        hFitSigma->SetStats(false);
        hFitMean->SetStats(false);
        hFitChi2->SetStats(false);
        hSysSelection->SetStats(false);
        hSysAbsorption->SetStats(false);
        hSysBranching->SetStats(false);
        hSysTotal->SetStats(false);
        std::vector<std::unique_ptr<RooPlot>> fitFrames;
        std::vector<std::unique_ptr<TCanvas>> stdDataFitCanvases;
        std::vector<std::unique_ptr<TF1>> shapeFits;
        std::vector<std::unique_ptr<TCanvas>> shapeFitCanvases;
        std::vector<SysBinArtifacts> sysArtifacts;
        std::unique_ptr<TH1D> hPeriodEventWeights;

        const std::string mcFileForAccGroup = ResolveAcceptanceMcFileForGroup(runCfg, group);
        const auto accVec = ComputeAcceptancePerBinWithErrors(corrCfg,
                                      group.items,
                                      edges,
                                      mcFileForAccGroup,
                                      &ptCtAccCache,
                                      &spectrumAccCache);
        const bool applyAbsorptionCorrection = IsAbsorptionCorrectionEnabled(runCfg);
        std::vector<BinValueWithError> absoVec(group.items.size(), BinValueWithError{});
        if (applyAbsorptionCorrection) {
            absoVec = ComputeAbsorptionPerBinWithErrors(corrCfg, group.items, edges, "", &ptCtAbsoCache);
        }

        std::map<std::string, std::vector<double>> absoVecByFile;
        const bool enableAbsorptionSystematic = applyAbsorptionCorrection;
        if (runCfg.doSystematics && enableAbsorptionSystematic) {
            std::vector<std::string> absoFiles = runCfg.systAbsorptionFiles;
            if (absoFiles.empty() && !runCfg.mcFileForAbsorption.empty()) {
                absoFiles.push_back(runCfg.mcFileForAbsorption);
            }
            if (!runCfg.mcFileForAbsorption.empty() &&
                std::find(absoFiles.begin(), absoFiles.end(), runCfg.mcFileForAbsorption) == absoFiles.end()) {
                absoFiles.push_back(runCfg.mcFileForAbsorption);
            }
            for (const auto &absoFile : absoFiles) {
                const PtCtAbsorptionCache *cachePtr =
                    (modeName == "pt_ct" && absoFile == runCfg.mcFileForAbsorption) ? &ptCtAbsoCache : nullptr;
                absoVecByFile[absoFile] = ComputeAbsorptionPerBin(corrCfg, group.items, edges, absoFile, cachePtr);
            }
        }

        double groupNEvents = 1.0;
        if ((modeName == "bdt_spectrum" || modeName == "topology_spectrum") && !group.items.empty()) {
            groupNEvents = GetNEvents(cfg, group.items.front().cenMin, group.items.front().cenMax);
            if (groupNEvents <= 0.0) groupNEvents = 1.0;
            auto periodWeights = GetPeriodEventWeights(cfg, group.items.front().cenMin, group.items.front().cenMax);
            if (!periodWeights.empty()) {
                hPeriodEventWeights = std::make_unique<TH1D>("h_period_event_weights",
                                                             ";Period;Data event weight",
                                                             static_cast<int>(periodWeights.size()),
                                                             0.5,
                                                             static_cast<double>(periodWeights.size()) + 0.5);
                hPeriodEventWeights->SetDirectory(nullptr);
                hPeriodEventWeights->SetStats(false);
                for (size_t ip = 0; ip < periodWeights.size(); ++ip) {
                    hPeriodEventWeights->SetBinContent(static_cast<int>(ip + 1), periodWeights[ip].second);
                    hPeriodEventWeights->GetXaxis()->SetBinLabel(static_cast<int>(ip + 1), periodWeights[ip].first.c_str());
                }
            }
        }

        for (const auto &item : group.items) {
            auto wp = GetWorkingPointForItem(modeName, runCfg.wpFile, item);
            const double bdtScore = wp.found ? wp.score : 0.0;
            const double bdtEff = wp.found ? wp.eff : 1.0;

            ProcessOneBinOptions oneBinOpt;
            oneBinOpt.dataTreeName = runCfg.dataTree;
            oneBinOpt.mcTreeName = runCfg.mcTree;
            oneBinOpt.massColumn = "fMassH3L";
            oneBinOpt.bdtScoreColumn = "model_output";
            oneBinOpt.mcMassSelection = "fMassH3L>2.95 && fMassH3L<3.02";
            oneBinOpt.useBdtCut = true;
            oneBinOpt.bdtCut = bdtScore;
            oneBinOpt.isMatter = runCfg.isMatter;
            oneBinOpt.enableQACapture = runCfg.enableOneBinQa;
            if (!runCfg.oneBinQaColumns.empty()) {
                oneBinOpt.qaColumns = runCfg.oneBinQaColumns;
            }
            oneBinOpt.throwOnError = IsCombinePeriodEnabled(cfg);
            std::vector<std::string> selParts;
            if (!runCfg.centralitySelection.empty()) selParts.push_back(runCfg.centralitySelection);
            if (!runCfg.additionalDataSelectionGeneral.empty()) selParts.push_back(runCfg.additionalDataSelectionGeneral);
            if (!runCfg.additionalDataSelection.empty()) selParts.push_back(runCfg.additionalDataSelection);
            if (!item.topologySelection.empty()) selParts.push_back(item.topologySelection);
            if (!selParts.empty()) {
                std::ostringstream expr;
                for (size_t j = 0; j < selParts.size(); ++j) {
                    if (j) expr << " && ";
                    expr << "(" << selParts[j] << ")";
                }
                oneBinOpt.dataSelection = expr.str();
            }

            const auto oneBinRes = ProcessOneBin(item, oneBinOpt, runCfg.fitCfg, runCfg.bkgFunc, runCfg.sigFunc, &rdfCache);
            if (!oneBinRes.success) {
                std::cerr << "[" << modeName << "] Skip bin " << item.label << ": " << oneBinRes.error << "\n";
                continue;
            }

            if (oneBinRes.massFit.frame) {
                const std::string frameName = "frame_data_" + item.label;
                auto *cloned = dynamic_cast<RooPlot *>(oneBinRes.massFit.frame->Clone(frameName.c_str()));
                if (cloned) {
                    const std::string simpleBinTitle = BuildSimpleBinTitle(item);
                    cloned->SetTitle(simpleBinTitle.c_str());
                    fitFrames.emplace_back(cloned);
                    const std::string canvasName = "canvas_" + frameName;
                    auto cData = MakeDecoratedFitCanvas(canvasName,
                                                        cloned,
                                                        false,
                                                        runCfg.plotLabels,
                                                        runCfg.isMatter,
                                                        groupNEvents);
                    if (cData) {
                        cData->SetTitle(simpleBinTitle.c_str());
                        stdDataFitCanvases.push_back(std::move(cData));
                    }
                }
            }
            if (oneBinRes.massFit.frameMc) {
                const std::string frameNameMc = "frame_mc_" + item.label;
                auto *clonedMc = dynamic_cast<RooPlot *>(oneBinRes.massFit.frameMc->Clone(frameNameMc.c_str()));
                if (clonedMc) fitFrames.emplace_back(clonedMc);
            }

            if (runCfg.enableOneBinQa && !oneBinRes.qaCandidates.empty()) {
                if (runCfg.onTheFlyChecksEnabled) {
                    if (onTheFlyQaColumns.empty()) {
                        onTheFlyQaColumns = oneBinRes.qaColumns;
                    }
                    std::unordered_map<std::string, size_t> srcIndex;
                    for (size_t i = 0; i < oneBinRes.qaColumns.size(); ++i) {
                        srcIndex[oneBinRes.qaColumns[i]] = i;
                    }

                    auto appendRow = [&](const ProcessOneBinResult::H3lQAVars &srcRow) {
                        ProcessOneBinResult::H3lQAVars dstRow;
                        dstRow.values.assign(onTheFlyQaColumns.size(), 0.0);
                        for (size_t ic = 0; ic < onTheFlyQaColumns.size(); ++ic) {
                            auto it = srcIndex.find(onTheFlyQaColumns[ic]);
                            if (it != srcIndex.end() && it->second < srcRow.values.size()) {
                                dstRow.values[ic] = srcRow.values[it->second];
                            }
                        }
                        onTheFlyQaCandidates.push_back(std::move(dstRow));
                    };

                    const bool applyMassWindow = runCfg.onTheFlyMassWindowNSigmas > 0.0 &&
                                                 oneBinRes.massFit.sigmaData > 0.0;
                    if (!applyMassWindow) {
                        for (const auto &srcRow : oneBinRes.qaCandidates) {
                            appendRow(srcRow);
                        }
                    } else {
                        const auto itMass = srcIndex.find("fMassH3L");
                        if (itMass == srcIndex.end()) {
                            for (const auto &srcRow : oneBinRes.qaCandidates) {
                                appendRow(srcRow);
                            }
                        } else {
                            const double massMin = oneBinRes.massFit.meanData - runCfg.onTheFlyMassWindowNSigmas * oneBinRes.massFit.sigmaData;
                            const double massMax = oneBinRes.massFit.meanData + runCfg.onTheFlyMassWindowNSigmas * oneBinRes.massFit.sigmaData;
                            for (const auto &srcRow : oneBinRes.qaCandidates) {
                                if (itMass->second >= srcRow.values.size()) continue;
                                const double m = srcRow.values[itMass->second];
                                if (m < massMin || m > massMax) continue;
                                appendRow(srcRow);
                            }
                        }
                    }
                }
            }

            const auto &fitRes = oneBinRes.massFit;
            const double xCenter = useCtAxis ? 0.5 * (item.ctMin + item.ctMax) : 0.5 * (item.ptMin + item.ptMax);
            const int ib = hRaw->FindBin(xCenter);
            const double acc = (static_cast<size_t>(ib - 1) < accVec.size()) ? accVec[static_cast<size_t>(ib - 1)].value : 1.0;
            const double accErr = (static_cast<size_t>(ib - 1) < accVec.size()) ? accVec[static_cast<size_t>(ib - 1)].error : 0.0;
            const double abso = (static_cast<size_t>(ib - 1) < absoVec.size()) ? absoVec[static_cast<size_t>(ib - 1)].value : 1.0;
            const double absoErr = (static_cast<size_t>(ib - 1) < absoVec.size()) ? absoVec[static_cast<size_t>(ib - 1)].error : 0.0;
            const auto evtLossCorr = isSpectrumMode ? GetSpectrumEventLossForBin(spectrumEvtSigLossCache, item) : BinValueWithError{1.0, 0.0};
            const auto evtSplittingCorr = isSpectrumMode ? GetSpectrumEventSplittingForBin(spectrumEvtSigLossCache, item) : BinValueWithError{1.0, 0.0};
            const auto sigLossCorr = isSpectrumMode ? GetSpectrumSignalLossForBin(spectrumEvtSigLossCache, item) : BinValueWithError{1.0, 0.0};

            const auto corrRes = ComputeCorrectedCounts(
                modeName,
                item,
                fitRes,
                acc,
                abso,
                bdtEff,
                evtLossCorr.value,
                evtSplittingCorr.value,
                sigLossCorr.value,
                groupNEvents,
                runCfg.branchingRatio,
                runCfg.deltaRap,
                runCfg.isMatter,
                runCfg.addAbsorptionCorrectionPtCt);
            const double corr = corrRes.value;

            hRaw->SetBinContent(ib, fitRes.signal);
            hRaw->SetBinError(ib, fitRes.signalErr);
            rawSumGroup += fitRes.signal;
            rawSumGroupErr2 += fitRes.signalErr * fitRes.signalErr;
            hCorr->SetBinContent(ib, corr);
            hCorr->SetBinError(ib, corrRes.error);
            hBdtEff->SetBinContent(ib, bdtEff);
            hBdtEff->SetBinError(ib, 0.0);
            hAcc->SetBinContent(ib, acc);
            hAcc->SetBinError(ib, accErr);
            hAbso->SetBinContent(ib, abso);
            hAbso->SetBinError(ib, absoErr);
            hEventLoss->SetBinContent(ib, evtLossCorr.value);
            hEventLoss->SetBinError(ib, evtLossCorr.error);
            hEventSplitting->SetBinContent(ib, evtSplittingCorr.value);
            hEventSplitting->SetBinError(ib, evtSplittingCorr.error);
            hSignalLoss->SetBinContent(ib, sigLossCorr.value);
            hSignalLoss->SetBinError(ib, sigLossCorr.error);
            hFitSigma->SetBinContent(ib, fitRes.sigmaData);
            hFitSigma->SetBinError(ib, fitRes.sigmaDataErr);
            hFitMean->SetBinContent(ib, fitRes.meanData);
            hFitMean->SetBinError(ib, fitRes.meanDataErr);
            hFitChi2->SetBinContent(ib, fitRes.chi2Data);
            if (applyBrSystematic) {
                hSysBranching->SetBinContent(ib, std::abs(corr) * runCfg.branchingRatioFractionalUncertainty);
            } else {
                hSysBranching->SetBinContent(ib, 0.0);
            }

            if (runCfg.doSystematics) {
                auto sysOut = RunSystematicsForBin(runCfg, group, item, rdfCache,
                                                   corr,
                                                   hCorr->GetBinError(ib),
                                                   acc,
                                                   abso,
                                                   evtLossCorr.value,
                                                   sigLossCorr.value,
                                                   groupNEvents,
                                                   ib,
                                                   runCfg.doSystematics ? &trailLog : nullptr,
                                                   &absoVecByFile,
                                                   enableAbsorptionSystematic);
                hSysSelection->SetBinContent(ib, sysOut.selectionRms);
                hSysAbsorption->SetBinContent(ib, sysOut.absorptionRms);
                sysArtifacts.push_back(std::move(sysOut));
            }
            const double sysSelection = hSysSelection->GetBinContent(ib);
            const double sysAbsorption = hSysAbsorption->GetBinContent(ib);
            const double sysBranching = hSysBranching->GetBinContent(ib);
            const double sysTotal = std::sqrt(sysSelection * sysSelection +
                                              sysAbsorption * sysAbsorption +
                                              sysBranching * sysBranching);
            AppendCorrectionsCsv(csvPath, modeName, group.dirName, item, fitRes, corrRes,
                                 sysSelection, sysAbsorption, sysBranching, sysTotal);
        }

        for (int ib = 1; ib <= hSysTotal->GetNbinsX(); ++ib) {
            const double fitV = hSysSelection->GetBinContent(ib);
            const double absoV = hSysAbsorption->GetBinContent(ib);
            const double brV = hSysBranching->GetBinContent(ib);
            hSysTotal->SetBinContent(ib, std::sqrt(fitV * fitV + absoV * absoV + brV * brV));
        }

        auto hFinalStat = std::unique_ptr<TH1D>(static_cast<TH1D *>(hCorr->Clone("h_final_spectrum_stat")));
        if (hFinalStat) hFinalStat->SetDirectory(nullptr);
        if (hFinalStat) hFinalStat->SetStats(false);
        std::unique_ptr<TGraphAsymmErrors> gFinalSys;
        if (hFinalStat) {
            gFinalSys = std::make_unique<TGraphAsymmErrors>(hFinalStat->GetNbinsX());
            gFinalSys->SetName("g_final_spectrum_sys");
            for (int ib = 1; ib <= hFinalStat->GetNbinsX(); ++ib) {
                const double x = hFinalStat->GetXaxis()->GetBinCenter(ib);
                const double y = hFinalStat->GetBinContent(ib);
                const double ex = 0.5 * hFinalStat->GetXaxis()->GetBinWidth(ib);
                const double ey = hSysTotal->GetBinContent(ib);
                gFinalSys->SetPoint(ib - 1, x, y);
                gFinalSys->SetPointError(ib - 1, ex, ex, ey, ey);
            }
        }

        if (modeName == "bdt_spectrum" || modeName == "topology_spectrum") {
            auto bwOut = BuildBlastWavePostFit(group, hCorr.get(), runCfg.integralFitParameters);
            if (bwOut.fit) shapeFits.push_back(std::move(bwOut.fit));
            if (bwOut.canvas) shapeFitCanvases.push_back(std::move(bwOut.canvas));
        }
        if (modeName == "pt_ct") {
            auto expoOut = BuildExponentialPostFit(group.dirName,
                                                   hCorr.get(),
                                                   edges.front(),
                                                   edges.back(),
                                                   "#it{c}t (cm)");
            if (expoOut.fit) shapeFits.push_back(std::move(expoOut.fit));
            if (expoOut.canvas) shapeFitCanvases.push_back(std::move(expoOut.canvas));
            if (hTauPerPt && expoOut.hasTau && !group.items.empty()) {
                const double ptCenter = 0.5 * (group.items.front().ptMin + group.items.front().ptMax);
                const int ptBin = hTauPerPt->FindBin(ptCenter);
                hTauPerPt->SetBinContent(ptBin, expoOut.tauPs);
                hTauPerPt->SetBinError(ptBin, expoOut.tauPsErr);
            }
        }
        if (modeName == "ct_single") {
            auto expoOut = BuildExponentialPostFit(group.dirName,
                                                   hCorr.get(),
                                                   edges.front(),
                                                   edges.back(),
                                                   "#it{c}t (cm)");
            if (expoOut.fit) shapeFits.push_back(std::move(expoOut.fit));
            if (expoOut.canvas) shapeFitCanvases.push_back(std::move(expoOut.canvas));

            const std::filesystem::path outDir = std::filesystem::path(runCfg.outputRoot).parent_path();
            std::filesystem::create_directories(outDir);
            const std::string txtName = "final_spectrum_expfit_" + SanitizeName(group.dirName) + ".txt";
            std::ofstream fitTxt(outDir / txtName, std::ios::out | std::ios::trunc);
            if (fitTxt.is_open()) {
                fitTxt << std::fixed << std::setprecision(6);
                fitTxt << "mode=ct_single\n";
                fitTxt << "group=" << group.dirName << "\n";
                if (!shapeFits.empty() && shapeFits.back()) {
                    const double chi2 = shapeFits.back()->GetChisquare();
                    const int ndf = shapeFits.back()->GetNDF();
                    const double chi2Ndf = (ndf > 0) ? (chi2 / static_cast<double>(ndf)) : 0.0;
                    const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;
                    fitTxt << "lifetime_ps=" << expoOut.tauPs << "\n";
                    fitTxt << "lifetime_err_ps=" << expoOut.tauPsErr << "\n";
                    fitTxt << "chi2=" << chi2 << "\n";
                    fitTxt << "ndf=" << ndf << "\n";
                    fitTxt << "chi2_ndf=" << chi2Ndf << "\n";
                    fitTxt << "fit_probability=" << fitProb << "\n";
                } else {
                    fitTxt << "status=exp_fit_missing\n";
                }
            }
        }

        IntegralYieldResult integralResult;
        if (isSpectrumMode && !group.items.empty()) {
            std::vector<std::unique_ptr<TH1D>> absorptionVariantOwned;
            std::vector<TH1D*> absorptionVariantPtrs;
            std::vector<std::string> absorptionVariantLabels;
            for (const auto &kv : absoVecByFile) {
                const auto &absoVariant = kv.second;
                if (absoVariant.empty()) continue;
                auto hVar = std::unique_ptr<TH1D>(static_cast<TH1D *>(hCorr->Clone(("h_corr_absvar_" + SanitizeName(group.dirName) + "_" + SanitizeName(kv.first)).c_str())));
                hVar->SetDirectory(nullptr);
                for (int ib = 1; ib <= hVar->GetNbinsX(); ++ib) {
                    if (static_cast<size_t>(ib - 1) >= absoVariant.size()) continue;
                    const double absoStd = hAbso->GetBinContent(ib);
                    const double absoVar = absoVariant[static_cast<size_t>(ib - 1)];
                    if (absoStd <= 0.0 || absoVar <= 0.0) continue;
                    const double scale = absoStd / absoVar;
                    hVar->SetBinContent(ib, hCorr->GetBinContent(ib) * scale);
                    hVar->SetBinError(ib, hCorr->GetBinError(ib) * std::abs(scale));
                }
                absorptionVariantPtrs.push_back(hVar.get());
                {
                    auto itLabel = std::find(runCfg.systAbsorptionFiles.begin(), runCfg.systAbsorptionFiles.end(), kv.first);
                    if (itLabel != runCfg.systAbsorptionFiles.end()) {
                        const size_t idxLabel = static_cast<size_t>(std::distance(runCfg.systAbsorptionFiles.begin(), itLabel));
                        if (idxLabel < runCfg.systAbsorptionFileLabels.size() && !runCfg.systAbsorptionFileLabels[idxLabel].empty()) {
                            absorptionVariantLabels.push_back(runCfg.systAbsorptionFileLabels[idxLabel]);
                        } else {
                            absorptionVariantLabels.push_back(kv.first);
                        }
                    } else {
                        absorptionVariantLabels.push_back(kv.first);
                    }
                }
                absorptionVariantOwned.push_back(std::move(hVar));
            }

            std::vector<std::vector<double>> trailValuesByBin(static_cast<size_t>(hCorr->GetNbinsX()));
            std::vector<std::vector<double>> trailValuesByTrialBin(static_cast<size_t>(hCorr->GetNbinsX()));
            if (runCfg.doSystematics) {
                for (size_t iItem = 0; iItem < group.items.size(); ++iItem) {
                    const std::string subName = MakeSysSubBinName(modeName, group.items[iItem]);
                    auto itArt = std::find_if(sysArtifacts.begin(), sysArtifacts.end(),
                                              [&](const SysBinArtifacts &a) { return a.subBinName == subName; });
                    if (itArt != sysArtifacts.end() && iItem < trailValuesByBin.size()) {
                        trailValuesByBin[iItem] = itArt->acceptedCorrValues;
                        trailValuesByTrialBin[iItem] = itArt->corrValuesByTrial;
                    }
                }
            }

            IntegralYieldInput iInput;
            iInput.groupTag = group.dirName;
            iInput.cenMin = group.items.front().cenMin;
            iInput.cenMax = group.items.front().cenMax;
            iInput.hCorrected = hCorr.get();
            iInput.hCorrectedSyst = hSysTotal.get();
            iInput.absorptionVariants = absorptionVariantPtrs;
            iInput.absorptionVariantLabels = absorptionVariantLabels;
            iInput.measuredPtEdges = edges;
            iInput.trailCorrectedValuesByBin = trailValuesByBin;
            iInput.trailCorrectedValuesByTrialBin = trailValuesByTrialBin;

            IntegralYieldConfig iCfg;
            iCfg.nominalFitFunc = runCfg.integralFitFunc;
            iCfg.fitFuncCandidates = runCfg.integralFitFuncs;
            iCfg.fitFuncParameters = runCfg.integralFitParameters;
            iCfg.doSystematics = runCfg.doSystematics;
            iCfg.nTrailsForIntegralSyst = runCfg.nTrailsForIntegralSyst;
            iCfg.nCombinationsForExtrapolation = runCfg.nCombinationsForIntegralSystExtrapolation;
            iCfg.rejectFitFuncByChi2 = runCfg.rejectIntegralFitFuncByChi2;
            iCfg.fitFuncMaxChi2Ndf = runCfg.integralFitFuncMaxChi2Ndf;
            iCfg.fitFuncFallbackFraction = runCfg.integralFitFuncFallbackFraction;
            iCfg.gaussFitMaxChi2Ndf = runCfg.integralGaussFitMaxChi2Ndf;
            iCfg.extrapToyMaxChi2Ndf = runCfg.integralExtrapToyMaxChi2Ndf;
            iCfg.fitRangeMin = runCfg.integralFitRangeMin;
            iCfg.fitRangeMax = runCfg.integralFitRangeMax;
            iCfg.integrateMin = runCfg.integratedYieldRangeMin;
            iCfg.integrateMax = runCfg.integratedYieldRangeMax;
            iCfg.lowPtMaxFactor = runCfg.integralLowPtMaxFactor;
            iCfg.useMinosErrors = runCfg.integralFitUseMinosErrors;
            iCfg.branchingRatioFractionalUncertainty = runCfg.branchingRatioFractionalUncertainty;
            iCfg.absorptionLength = runCfg.absorptionLength;
            iCfg.usePerformanceLabel = runCfg.plotLabels.usePerformance;
            iCfg.performanceLabel = runCfg.plotLabels.performanceLabel;
            iCfg.collisionSystem = runCfg.plotLabels.collisionSystem;
            iCfg.collisionEnergy = runCfg.plotLabels.collisionEnergy;
            iCfg.period = runCfg.plotLabels.period;
            iCfg.periodMark = runCfg.plotLabels.periodMark;
            iCfg.isMatter = runCfg.isMatter;

            std::cout << "[Info] Run integrated-yield for " << group.dirName
                      << " with nominal=" << iCfg.nominalFitFunc
                      << ", candidates=" << iCfg.fitFuncCandidates.size()
                      << ", nTrails=" << iCfg.nTrailsForIntegralSyst
                      << std::endl;
            integralResult = ComputeIntegralYield(iInput, iCfg, runCfg.randomSeed + static_cast<int>(group.items.front().cenMin * 10.0));
            if (integralResult.ok) {
                integralSummaryRows.push_back(IntegralSummaryRow{
                    iInput.cenMin,
                    iInput.cenMax,
                    integralResult.value,
                    integralResult.statErr,
                    integralResult.systExtrapolation,
                    integralResult.systFitFunction,
                    integralResult.systAbsorption,
                    integralResult.systTrails,
                    integralResult.systBranchingRatio,
                    integralResult.systTotal});
                for (const auto &fitRow : integralResult.fitParameterRows) {
                    integralFitParameterRows.push_back(IntegralFitParameterSummaryRow{
                        iInput.cenMin,
                        iInput.cenMax,
                        fitRow});
                }
            } else {
                std::cout << "[Warn] Integrated-yield skipped for " << group.dirName
                          << " (see IntegralYield logs above for failure step)"
                          << std::endl;
            }
        }

        std::unique_ptr<TH1D> hRawOverNevts;
        std::unique_ptr<TCanvas> cRawOverNevts;
        if (isSpectrumMode) {
            const double rawErrGroup = std::sqrt(std::max(0.0, rawSumGroupErr2));
            hRawOverNevts = MakeRawOverNevtsHist("h_raw_over_nevents", rawSumGroup, rawErrGroup, groupNEvents, group.dirName);
            cRawOverNevts = MakeRawOverNevtsCanvas("c_raw_over_nevents", hRawOverNevts.get(), rawSumGroup, rawErrGroup, groupNEvents, group.dirName, periodTag);
            totalRawOverNevts += rawSumGroup;
            totalEventsOverNevts += groupNEvents;
            totalRawErr2OverNevts += rawSumGroupErr2;
        }

        TF1 *finalStdFit = shapeFits.empty() ? nullptr : shapeFits.front().get();
        const std::string finalXAxisTitle = useCtAxis ? "#it{c}t (cm)" : "#it{p}_{T} (GeV/#it{c})";
        const std::string finalExtraText = group.dirName;

        WriteGroupOutput(fout,
                         group.dirName,
                         hRaw.get(),
                         hCorr.get(),
                         hBdtEff.get(),
                         hAcc.get(),
                         hAbso.get(),
                         hEventLoss.get(),
                         hEventSplitting.get(),
                         hSignalLoss.get(),
                         hFitSigma.get(),
                         hFitMean.get(),
                         hFitChi2.get(),
                         hPeriodEventWeights.get(),
                         hRawOverNevts.get(),
                         cRawOverNevts.get(),
                         hSysSelection.get(),
                         hSysAbsorption.get(),
                         hSysBranching.get(),
                         hSysTotal.get(),
                         hFinalStat.get(),
                         gFinalSys.get(),
                         runCfg.doSystematics ? &sysArtifacts : nullptr,
                         &fitFrames,
                         &stdDataFitCanvases,
                         &shapeFits,
                         &shapeFitCanvases,
                         finalStdFit,
                         finalXAxisTitle,
                         finalExtraText,
                         &runCfg.plotLabels,
                         runCfg.isMatter,
                         groupNEvents);

        if (isSpectrumMode && integralResult.ok) {
            TDirectory *groupDir = fout.GetDirectory(group.dirName.c_str());
            if (groupDir) {
                TDirectory *intDir = groupDir->GetDirectory("integral_yield");
                if (!intDir) intDir = groupDir->mkdir("integral_yield");
                if (intDir) {
                    intDir->cd();
                    if (integralResult.fNominal) integralResult.fNominal->Write("f_integral_nominal", TObject::kOverwrite);
                    for (size_t i = 0; i < integralResult.fFitCandidates.size(); ++i) {
                        if (integralResult.fFitCandidates[i]) {
                            integralResult.fFitCandidates[i]->Write(Form("f_integral_candidate_%zu", i), TObject::kOverwrite);
                        }
                    }
                    if (integralResult.hExtrapRatioDist) integralResult.hExtrapRatioDist->Write("h_integral_extrap_ratio", TObject::kOverwrite);
                    if (integralResult.hIntegralTrailDist) integralResult.hIntegralTrailDist->Write("h_integral_trails", TObject::kOverwrite);
                    if (integralResult.hAbsorptionYieldScan) integralResult.hAbsorptionYieldScan->Write("h_integral_absorption_scan", TObject::kOverwrite);
                    if (integralResult.hSystSources) integralResult.hSystSources->Write("h_integral_syst_sources", TObject::kOverwrite);
                    if (integralResult.hSystSourceFractions) integralResult.hSystSourceFractions->Write("h_integral_syst_source_fractions", TObject::kOverwrite);
                    if (integralResult.hIntegralYieldOneBin) integralResult.hIntegralYieldOneBin->Write("h_integral_yield_onebin", TObject::kOverwrite);
                    if (integralResult.cNominalAndFunctions) integralResult.cNominalAndFunctions->Write("c_integral_fit_functions", TObject::kOverwrite);
                    if (integralResult.cFitFunctionParameters) integralResult.cFitFunctionParameters->Write("c_integral_fit_function_parameters", TObject::kOverwrite);
                    if (integralResult.cExtrapolation) integralResult.cExtrapolation->Write("c_integral_extrapolation", TObject::kOverwrite);
                    if (integralResult.cAbsorption) integralResult.cAbsorption->Write("c_integral_absorption", TObject::kOverwrite);
                    if (integralResult.cAbsorptionYieldScan) integralResult.cAbsorptionYieldScan->Write("c_integral_absorption_scan", TObject::kOverwrite);
                    if (integralResult.cTrails) integralResult.cTrails->Write("c_integral_trails", TObject::kOverwrite);
                    if (integralResult.cSources) integralResult.cSources->Write("c_integral_syst_sources", TObject::kOverwrite);
                    if (integralResult.cIntegralYieldOneBin) integralResult.cIntegralYieldOneBin->Write("c_integral_yield_onebin", TObject::kOverwrite);
                }
            }
        }

    }

    if (isSpectrumMode && totalEventsOverNevts > 0.0) {
        const double rawErrAll = std::sqrt(std::max(0.0, totalRawErr2OverNevts));
        auto hAll = MakeRawOverNevtsHist("h_raw_over_nevents_all",
                                         totalRawOverNevts,
                                         rawErrAll,
                                         totalEventsOverNevts,
                                         "all_centralities");
        auto cAll = MakeRawOverNevtsCanvas("c_raw_over_nevents_all",
                                           hAll.get(),
                                           totalRawOverNevts,
                                           rawErrAll,
                                           totalEventsOverNevts,
                                           "all_centralities",
                                           periodTag);
        TDirectory *summaryDir = fout.GetDirectory("summary");
        if (summaryDir) {
            fout.cd();
            fout.Delete("summary;*");
        }
        summaryDir = fout.GetDirectory("summary");
        if (!summaryDir) summaryDir = fout.mkdir("summary");
        if (summaryDir) {
            TDirectory *stdDir = summaryDir->GetDirectory("std");
            if (!stdDir) stdDir = summaryDir->mkdir("std");
            if (stdDir) {
                stdDir->cd();
                if (hAll) hAll->Write("h_raw_over_nevents_all", TObject::kOverwrite);
                if (cAll) cAll->Write("c_raw_over_nevents_all", TObject::kOverwrite);
            }
        }
    }

    if (isSpectrumMode && !integralSummaryRows.empty()) {
        std::vector<double> cenEdges = runCfg.cenBins;
        if (cenEdges.size() < 2) {
            cenEdges.clear();
            cenEdges.push_back(integralSummaryRows.front().cenMin);
            for (const auto &row : integralSummaryRows) cenEdges.push_back(row.cenMax);
        }

        auto hIntegralStat = std::make_unique<TH1D>("h_integral_yield_stat", ";Centrality (%);Integrated yield", static_cast<int>(cenEdges.size() - 1), cenEdges.data());
        auto hIntegralSys = std::make_unique<TH1D>("h_integral_yield_sys", ";Centrality (%);Integrated yield", static_cast<int>(cenEdges.size() - 1), cenEdges.data());
        auto hIntegralSysRatio = std::make_unique<TH1D>("h_integral_yield_sys_ratio", ";Centrality (%);Total sys / integrated yield", static_cast<int>(cenEdges.size() - 1), cenEdges.data());
        auto hSystFractionSummary = std::make_unique<TH1D>("h_integral_syst_fraction_summary", ";Systematic source;Average fraction (%)", 5, 0.5, 5.5);
        std::array<std::unique_ptr<TH1D>, 6> hSystFractionVsCen;
        const std::array<std::string, 6> systNames = {"Extrap", "FitFunc", "Absorption", "CorrTrails", "Branching", "Total"};
        const std::array<int, 6> systColors = {kRed + 1, kBlue + 1, kMagenta + 1, kGreen + 2, kOrange + 7, kBlack};
        for (size_t i = 0; i < hSystFractionVsCen.size(); ++i) {
            hSystFractionVsCen[i] = std::make_unique<TH1D>(("h_integral_syst_fraction_vs_centrality_" + systNames[i]).c_str(),
                                                           ";Centrality (%);Uncertainty / integrated yield (%)",
                                                           static_cast<int>(cenEdges.size() - 1),
                                                           cenEdges.data());
            hSystFractionVsCen[i]->SetDirectory(nullptr);
            hSystFractionVsCen[i]->SetStats(false);
            hSystFractionVsCen[i]->SetLineColor(systColors[i]);
            hSystFractionVsCen[i]->SetMarkerColor(systColors[i]);
            hSystFractionVsCen[i]->SetMarkerStyle(i == 5 ? 20 : 24 + static_cast<int>(i));
            hSystFractionVsCen[i]->SetLineWidth(i == 5 ? 3 : 2);
        }

        hIntegralStat->SetDirectory(nullptr);
        hIntegralSys->SetDirectory(nullptr);
        hIntegralSysRatio->SetDirectory(nullptr);
        hSystFractionSummary->SetDirectory(nullptr);
        hIntegralStat->SetStats(false);
        hIntegralSys->SetStats(false);
        hIntegralSysRatio->SetStats(false);
        hSystFractionSummary->SetStats(false);
        hSystFractionSummary->GetXaxis()->SetBinLabel(1, "Extrap");
        hSystFractionSummary->GetXaxis()->SetBinLabel(2, "FitFunc");
        hSystFractionSummary->GetXaxis()->SetBinLabel(3, "Absorption");
        hSystFractionSummary->GetXaxis()->SetBinLabel(4, "CorrTrails");
        hSystFractionSummary->GetXaxis()->SetBinLabel(5, "Branching");

        std::array<double, 5> fracSums{0.0, 0.0, 0.0, 0.0, 0.0};
        int fracCount = 0;
        for (const auto &row : integralSummaryRows) {
            const double x = 0.5 * (row.cenMin + row.cenMax);
            const int ib = hIntegralStat->FindBin(x);
            hIntegralStat->SetBinContent(ib, row.value);
            hIntegralStat->SetBinError(ib, row.statErr);
            hIntegralSys->SetBinContent(ib, row.value);
            hIntegralSys->SetBinError(ib, row.systTotal);
            hIntegralSysRatio->SetBinContent(ib, (std::abs(row.value) > 0.0) ? (row.systTotal / std::abs(row.value)) : 0.0);

            if (std::abs(row.value) > 0.0) {
                const std::array<double, 6> fracs = {
                    100.0 * row.systExtrap / std::abs(row.value),
                    100.0 * row.systFitFunc / std::abs(row.value),
                    100.0 * row.systAbsorption / std::abs(row.value),
                    100.0 * row.systTrails / std::abs(row.value),
                    100.0 * row.systBranching / std::abs(row.value),
                    100.0 * row.systTotal / std::abs(row.value)};
                for (size_t i = 0; i < fracs.size(); ++i) {
                    hSystFractionVsCen[i]->SetBinContent(ib, fracs[i]);
                }
                fracSums[0] += fracs[0];
                fracSums[1] += fracs[1];
                fracSums[2] += fracs[2];
                fracSums[3] += fracs[3];
                fracSums[4] += fracs[4];
                ++fracCount;
            }
        }
        if (fracCount > 0) {
            for (int i = 0; i < 5; ++i) {
                hSystFractionSummary->SetBinContent(i + 1, fracSums[static_cast<size_t>(i)] / static_cast<double>(fracCount));
            }
        }

        auto sortedIntegralSummaryRows = integralSummaryRows;
        std::sort(sortedIntegralSummaryRows.begin(), sortedIntegralSummaryRows.end(),
                  [](const IntegralSummaryRow &a, const IntegralSummaryRow &b) {
                      if (a.cenMin != b.cenMin) return a.cenMin < b.cenMin;
                      return a.cenMax < b.cenMax;
                  });

        {
            std::ofstream intCsv(integratedYieldCsvPath);
            if (intCsv.is_open()) {
                intCsv << std::setprecision(12);
                intCsv << "centrality_min,centrality_max,integrated_yield,stat_err,";
                intCsv << "syst_extrapolation_abs,syst_fit_function_abs,syst_absorption_abs,syst_correction_trails_abs,syst_branching_abs,syst_total_abs,";
                intCsv << "stat_ratio,syst_extrapolation_ratio,syst_fit_function_ratio,syst_absorption_ratio,syst_correction_trails_ratio,syst_branching_ratio,syst_total_ratio\n";
                for (const auto &row : sortedIntegralSummaryRows) {
                    AppendIntegratedYieldCsvRow(intCsv,
                                                row.cenMin,
                                                row.cenMax,
                                                row.value,
                                                row.statErr,
                                                row.systExtrap,
                                                row.systFitFunc,
                                                row.systAbsorption,
                                                row.systTrails,
                                                row.systBranching,
                                                row.systTotal);
                }
            }
        }

        const auto formatCentrality = [](double cmin, double cmax) {
            std::ostringstream os;
            os << FormatTitleNumber(cmin) << "--" << FormatTitleNumber(cmax);
            return os.str();
        };
        const auto formatPercent = [](double ratio) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(2) << 100.0 * ratio;
            return os.str();
        };
        const auto formatScientificLatex = [](double value) {
            if (!std::isfinite(value) || value == 0.0) return std::string("$0$");
            const double absValue = std::abs(value);
            const int exponent = static_cast<int>(std::floor(std::log10(absValue)));
            const double mantissa = value / std::pow(10.0, exponent);
            std::ostringstream os;
            os << "$" << std::fixed << std::setprecision(3) << mantissa
               << " \\times 10^{" << exponent << "}$";
            return os.str();
        };
        const auto formatParameterValue = [](double value) {
            if (!std::isfinite(value)) return std::string("--");
            const double absValue = std::abs(value);
            std::ostringstream os;
            if ((absValue > 0.0 && absValue < 1e-3) || absValue >= 1e4) {
                os << std::scientific << std::setprecision(3) << value;
            } else {
                os << std::fixed << std::setprecision(4) << value;
            }
            return os.str();
        };
        const auto formatParameterLatex = [&](double value, double err) {
            if (!std::isfinite(value)) return std::string("--");
            if (std::isfinite(err) && err > 0.0) {
                return "$" + formatParameterValue(value) + " \\pm " + formatParameterValue(err) + "$";
            }
            return "$" + formatParameterValue(value) + "$";
        };
        const auto formatParameterNameLatex = [](const std::string &name) {
            if (name == "#beta") return std::string("$\\beta$");
            if (name == "Norm") return std::string("Norm.");
            return std::string("$") + name + "$";
        };
        const auto formatChi2NdfLatex = [&](double chi2, int ndf) {
            if (!std::isfinite(chi2) || ndf < 0) return std::string("--");
            return formatParameterValue(chi2) + " (" + std::to_string(ndf) + ")";
        };
        const auto formatFitStatus = [](const IntegralFitParameterRow &row) {
            if (row.isNominal) return std::string("nominal");
            if (row.rejectedByChi2 && row.rejectedByLowPt) return std::string("rejected chi2 low-pT");
            if (row.rejectedByChi2) return std::string("rejected chi2");
            if (row.rejectedByLowPt) return std::string("rejected low-pT");
            return std::string("accepted");
        };

        const std::filesystem::path integratedYieldDir = std::filesystem::path(integratedYieldCsvPath).parent_path();
        const std::filesystem::path integratedYieldSystTexPath = integratedYieldDir / "integrated_yield_syst_table.tex";
        const std::filesystem::path integratedYieldFinalTexPath = integratedYieldDir / "integrated_yield_final_table.tex";
        const std::filesystem::path fitParameterCsvPath = integratedYieldDir / "fit_function_parameters.csv";
        const std::filesystem::path fitParameterTexPath = integratedYieldDir / "fit_function_parameters_table.tex";
        const std::filesystem::path bgbwParameterTexPath = integratedYieldDir / "bgbw_fit_parameters_table.tex";

        {
            std::ofstream systTex(integratedYieldSystTexPath);
            if (systTex.is_open()) {
                systTex << "% Auto-generated by UnifiedTaskRunner from integrated_yield_summary.csv.\n";
                systTex << "% Do not edit by hand; rerun the workflow instead.\n";
                systTex << "\\begin{tabular}{c|cccccc}\n";
                systTex << "    \\hline\n";
                systTex << "    Centrality (\\%) & Extrapolation (\\%) & Fit function (\\%) & Absorption (\\%) & Selection/trials (\\%) & Branching ratio (\\%) & Total syst. (\\%) \\\\\n";
                systTex << "    \\hline\n";
                for (const auto &row : sortedIntegralSummaryRows) {
                    const double den = std::abs(row.value);
                    const auto ratio = [den](double v) { return den > 0.0 ? v / den : 0.0; };
                    systTex << "    " << formatCentrality(row.cenMin, row.cenMax)
                            << " & " << formatPercent(ratio(row.systExtrap))
                            << " & " << formatPercent(ratio(row.systFitFunc))
                            << " & " << formatPercent(ratio(row.systAbsorption))
                            << " & " << formatPercent(ratio(row.systTrails))
                            << " & " << formatPercent(ratio(row.systBranching))
                            << " & " << formatPercent(ratio(row.systTotal))
                            << " \\\\\n";
                }
                systTex << "    \\hline\n";
                systTex << "\\end{tabular}\n";
                std::cout << "[Info] Wrote LaTeX table fragment: " << integratedYieldSystTexPath << std::endl;
            }
        }

        {
            std::ofstream finalTex(integratedYieldFinalTexPath);
            if (finalTex.is_open()) {
                finalTex << "% Auto-generated by UnifiedTaskRunner from integrated_yield_summary.csv.\n";
                finalTex << "% Do not edit by hand; rerun the workflow instead.\n";
                finalTex << "\\begin{tabular}{c|ccc}\n";
                finalTex << "    \\hline\n";
                finalTex << "    Centrality (\\%) & Integrated yield & Stat. unc. & Syst. unc. \\\\\n";
                finalTex << "    \\hline\n";
                for (const auto &row : sortedIntegralSummaryRows) {
                    finalTex << "    " << formatCentrality(row.cenMin, row.cenMax)
                             << " & " << formatScientificLatex(row.value)
                             << " & " << formatScientificLatex(row.statErr)
                             << " & " << formatScientificLatex(row.systTotal)
                             << " \\\\\n";
                }
                finalTex << "    \\hline\n";
                finalTex << "\\end{tabular}\n";
                std::cout << "[Info] Wrote LaTeX table fragment: " << integratedYieldFinalTexPath << std::endl;
            }
        }

        if (!integralFitParameterRows.empty()) {
            auto sortedFitParameterRows = integralFitParameterRows;
            std::sort(sortedFitParameterRows.begin(), sortedFitParameterRows.end(),
                      [](const IntegralFitParameterSummaryRow &a, const IntegralFitParameterSummaryRow &b) {
                          if (a.cenMin != b.cenMin) return a.cenMin < b.cenMin;
                          if (a.cenMax != b.cenMax) return a.cenMax < b.cenMax;
                          if (a.fitRow.isNominal != b.fitRow.isNominal) return a.fitRow.isNominal > b.fitRow.isNominal;
                          if (a.fitRow.functionName != b.fitRow.functionName) return a.fitRow.functionName < b.fitRow.functionName;
                          return a.fitRow.parameterName < b.fitRow.parameterName;
                      });

            {
                std::ofstream fitCsv(fitParameterCsvPath);
                if (fitCsv.is_open()) {
                    fitCsv << std::setprecision(12);
                    fitCsv << "centrality_min,centrality_max,function,parameter,parameter_index,value,error,limit_min,limit_max,has_limits,chi2,ndf,chi2_ndf,is_nominal,rejected_by_chi2,rejected_by_lowpt,status\n";
                    for (const auto &row : sortedFitParameterRows) {
                        const auto &fitRow = row.fitRow;
                        fitCsv << row.cenMin << ',' << row.cenMax << ','
                               << fitRow.functionName << ','
                               << fitRow.parameterName << ','
                               << fitRow.parameterIndex << ','
                               << fitRow.value << ','
                               << fitRow.error << ','
                               << fitRow.limitMin << ','
                               << fitRow.limitMax << ','
                               << (fitRow.hasLimits ? 1 : 0) << ','
                               << fitRow.chi2 << ','
                               << fitRow.ndf << ','
                               << fitRow.chi2Ndf << ','
                               << (fitRow.isNominal ? 1 : 0) << ','
                               << (fitRow.rejectedByChi2 ? 1 : 0) << ','
                               << (fitRow.rejectedByLowPt ? 1 : 0) << ','
                               << formatFitStatus(fitRow) << '\n';
                    }
                    std::cout << "[Info] Wrote fit-function parameter CSV: " << fitParameterCsvPath << std::endl;
                }
            }

            {
                std::ofstream fitTex(fitParameterTexPath);
                if (fitTex.is_open()) {
                    fitTex << "% Auto-generated by UnifiedTaskRunner from fit_function_parameters.csv.\n";
                    fitTex << "% Do not edit by hand; rerun the workflow instead.\n";
                    fitTex << "\\begin{tabular}{c|ccccc}\n";
                    fitTex << "    \\hline\n";
                    fitTex << "    Centrality (\\%) & Function & Parameter & Value & $\\chi^{2}$ (ndf) & Status \\\\\n";
                    fitTex << "    \\hline\n";
                    for (const auto &row : sortedFitParameterRows) {
                        const auto &fitRow = row.fitRow;
                        fitTex << "    " << formatCentrality(row.cenMin, row.cenMax)
                               << " & " << fitRow.functionName
                               << " & " << formatParameterNameLatex(fitRow.parameterName)
                               << " & " << formatParameterLatex(fitRow.value, fitRow.error)
                               << " & " << formatChi2NdfLatex(fitRow.chi2, fitRow.ndf)
                               << " & " << formatFitStatus(fitRow)
                               << " \\\\\n";
                    }
                    fitTex << "    \\hline\n";
                    fitTex << "\\end{tabular}\n";
                    std::cout << "[Info] Wrote LaTeX table fragment: " << fitParameterTexPath << std::endl;
                }
            }

            {
                std::map<std::pair<double, double>, std::map<std::string, IntegralFitParameterRow>> bgbwRowsByCen;
                for (const auto &row : sortedFitParameterRows) {
                    if (!row.fitRow.isNominal || row.fitRow.functionName != "fBGBW") continue;
                    bgbwRowsByCen[{row.cenMin, row.cenMax}][row.fitRow.parameterName] = row.fitRow;
                }

                std::ofstream bgbwTex(bgbwParameterTexPath);
                if (bgbwTex.is_open()) {
                    bgbwTex << "% Auto-generated by UnifiedTaskRunner from fit_function_parameters.csv.\n";
                    bgbwTex << "% Do not edit by hand; rerun the workflow instead.\n";
                    bgbwTex << "\\begin{tabular}{c|cccc}\n";
                    bgbwTex << "    \\hline\n";
                    bgbwTex << "    Centrality (\\%) & $\\beta$ & $T$ & $n$ & $\\chi^{2}$ (ndf) \\\\\n";
                    bgbwTex << "    \\hline\n";
                    for (const auto &kv : bgbwRowsByCen) {
                        const auto &cen = kv.first;
                        const auto &pars = kv.second;
                        const auto getPar = [&](const std::string &name) {
                            auto it = pars.find(name);
                            if (it == pars.end()) return std::string("--");
                            return formatParameterLatex(it->second.value, it->second.error);
                        };
                        double chi2 = std::numeric_limits<double>::quiet_NaN();
                        int ndf = -1;
                        if (!pars.empty()) {
                            chi2 = pars.begin()->second.chi2;
                            ndf = pars.begin()->second.ndf;
                        }
                        bgbwTex << "    " << formatCentrality(cen.first, cen.second)
                                << " & " << getPar("#beta")
                                << " & " << getPar("T")
                                << " & " << getPar("n")
                                << " & " << formatChi2NdfLatex(chi2, ndf)
                                << " \\\\\n";
                    }
                    bgbwTex << "    \\hline\n";
                    bgbwTex << "\\end{tabular}\n";
                    std::cout << "[Info] Wrote LaTeX table fragment: " << bgbwParameterTexPath << std::endl;
                }
            }
        }

        auto gIntegralSys = std::make_unique<TGraphAsymmErrors>(hIntegralStat->GetNbinsX());
        gIntegralSys->SetName("g_integral_yield_sys");
        gIntegralSys->SetLineColor(kAzure + 2);
        gIntegralSys->SetLineStyle(kDashed);
        gIntegralSys->SetLineWidth(2);
        for (int ib = 1; ib <= hIntegralStat->GetNbinsX(); ++ib) {
            const double x = hIntegralStat->GetXaxis()->GetBinCenter(ib);
            const double y = hIntegralStat->GetBinContent(ib);
            const double ex = 0.5 * hIntegralStat->GetXaxis()->GetBinWidth(ib);
            const double ey = hIntegralSys->GetBinError(ib);
            gIntegralSys->SetPoint(ib - 1, x, y);
            gIntegralSys->SetPointError(ib - 1, ex, ex, ey, ey);
        }

        struct IntegralMultiplicityPoint {
            double mult{0.0};
            double yield{0.0};
            double stat{0.0};
            double syst{0.0};
            double cenMin{0.0};
            double cenMax{0.0};
        };
        std::vector<IntegralMultiplicityPoint> integralMultiplicityPoints;
        if (runCfg.relatedMultiplicityCenter.size() + 1 == runCfg.cenBins.size()) {
            for (const auto &row : integralSummaryRows) {
                int cenIdx = -1;
                for (size_t ic = 0; ic + 1 < runCfg.cenBins.size(); ++ic) {
                    if (std::abs(row.cenMin - runCfg.cenBins[ic]) < 1e-6 &&
                        std::abs(row.cenMax - runCfg.cenBins[ic + 1]) < 1e-6) {
                        cenIdx = static_cast<int>(ic);
                        break;
                    }
                }
                if (cenIdx < 0 || static_cast<size_t>(cenIdx) >= runCfg.relatedMultiplicityCenter.size()) continue;
                integralMultiplicityPoints.push_back({runCfg.relatedMultiplicityCenter[static_cast<size_t>(cenIdx)],
                                                       row.value,
                                                       row.statErr,
                                                       row.systTotal,
                                                       row.cenMin,
                                                       row.cenMax});
            }
            std::sort(integralMultiplicityPoints.begin(),
                      integralMultiplicityPoints.end(),
                      [](const auto &a, const auto &b) { return a.mult < b.mult; });
        } else if (!runCfg.relatedMultiplicityCenter.empty()) {
            Warning("UnifiedTaskRunner",
                    "related_multiplicity_center size (%zu) does not match centrality bin count (%zu); skip multiplicity summary canvas",
                    runCfg.relatedMultiplicityCenter.size(),
                    runCfg.cenBins.size() > 0 ? runCfg.cenBins.size() - 1 : 0);
        }

        auto cIntegral = std::make_unique<TCanvas>("c_integral_yield_vs_centrality", "", 900, 700);
        cIntegral->SetLeftMargin(0.14);
        cIntegral->SetBottomMargin(0.12);
        cIntegral->SetRightMargin(0.05);
        cIntegral->SetTopMargin(0.06);
        cIntegral->SetTicks(1, 1);
        double yIntMin = std::numeric_limits<double>::infinity();
        double yIntMax = 0.0;
        for (int ib = 1; ib <= hIntegralStat->GetNbinsX(); ++ib) {
            const double y = hIntegralStat->GetBinContent(ib);
            const double e = std::hypot(hIntegralStat->GetBinError(ib), hIntegralSys->GetBinError(ib));
            if (y <= 0.0) continue;
            yIntMin = std::min(yIntMin, std::max(1e-20, y - e));
            yIntMax = std::max(yIntMax, y + e);
        }
        if (!std::isfinite(yIntMin)) yIntMin = 0.0;
        if (!(yIntMax > yIntMin)) yIntMax = yIntMin + 1e-12;
        hIntegralStat->GetYaxis()->SetRangeUser(std::max(0.0, yIntMin * 0.75), yIntMax * 1.35);
        hIntegralStat->SetTitle("");
        hIntegralStat->SetMarkerStyle(20);
        hIntegralStat->SetMarkerColor(kBlack);
        hIntegralStat->SetLineColor(kBlack);
        hIntegralStat->Draw("E1 X0");
        for (int ib = 1; ib <= hIntegralStat->GetNbinsX(); ++ib) {
            const double x1 = hIntegralStat->GetXaxis()->GetBinLowEdge(ib);
            const double x2 = hIntegralStat->GetXaxis()->GetBinUpEdge(ib);
            const double y = hIntegralStat->GetBinContent(ib);
            const double ey = hIntegralSys->GetBinError(ib);
            if (!std::isfinite(y) || !std::isfinite(ey) || ey <= 0.0) continue;
            TBox box(x1, std::max(0.0, y - ey), x2, y + ey);
            box.SetFillStyle(0);
            box.SetLineColor(kAzure + 2);
            box.SetLineStyle(kDashed);
            box.SetLineWidth(2);
            box.DrawClone("l");
        }
        hIntegralStat->Draw("E1 X0 SAME");
        {
            TLegend leg(0.56, 0.74, 0.90, 0.90);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            leg.SetTextFont(42);
            leg.SetTextSize(0.035);
            leg.AddEntry(hIntegralStat.get(), "Integrated yield (stat.)", "lep");
            leg.AddEntry(gIntegralSys.get(), "Total syst.", "l");
            leg.DrawClone();
        }
        {
            TPaveText text(0.16, 0.64, 0.56, 0.90, "NDC");
            text.SetBorderSize(0);
            text.SetFillStyle(0);
            text.SetTextAlign(12);
            text.SetTextFont(42);
            text.SetTextSize(0.035);
            if (runCfg.plotLabels.usePerformance && !runCfg.plotLabels.performanceLabel.empty()) {
                text.AddText(runCfg.plotLabels.performanceLabel.c_str());
            }
            if (!runCfg.plotLabels.collisionSystem.empty() || !runCfg.plotLabels.collisionEnergy.empty()) {
                text.AddText((runCfg.plotLabels.collisionSystem + " " + runCfg.plotLabels.collisionEnergy).c_str());
            }
            if (!runCfg.plotLabels.period.empty() || !runCfg.plotLabels.periodMark.empty()) {
                text.AddText((runCfg.plotLabels.period + " " + runCfg.plotLabels.periodMark).c_str());
            }
            const std::string decay = BuildDecayString(runCfg.isMatter);
            if (!decay.empty()) {
                text.AddText(decay.c_str());
            }
            text.DrawClone();
        }

        std::unique_ptr<TGraphAsymmErrors> gIntegralMultiplicityStat;
        std::unique_ptr<TGraphAsymmErrors> gIntegralMultiplicitySys;
        std::unique_ptr<TCanvas> cIntegralMultiplicity;
        std::unique_ptr<TH1D> hFrameMult;
        if (!integralMultiplicityPoints.empty()) {
            const int nMult = static_cast<int>(integralMultiplicityPoints.size());
            gIntegralMultiplicityStat = std::make_unique<TGraphAsymmErrors>(nMult);
            gIntegralMultiplicitySys = std::make_unique<TGraphAsymmErrors>(nMult);
            gIntegralMultiplicityStat->SetName("g_integral_yield_vs_multiplicity_stat");
            gIntegralMultiplicitySys->SetName("g_integral_yield_vs_multiplicity_sys");
            gIntegralMultiplicityStat->SetMarkerStyle(20);
            gIntegralMultiplicityStat->SetMarkerColor(kBlack);
            gIntegralMultiplicityStat->SetLineColor(kBlack);
            gIntegralMultiplicitySys->SetLineColor(kAzure + 2);
            gIntegralMultiplicitySys->SetLineStyle(kDashed);
            gIntegralMultiplicitySys->SetLineWidth(2);

            double multMin = std::numeric_limits<double>::infinity();
            double multMax = 0.0;
            double yMultMin = std::numeric_limits<double>::infinity();
            double yMultMax = 0.0;
            for (int ip = 0; ip < nMult; ++ip) {
                const auto &p = integralMultiplicityPoints[static_cast<size_t>(ip)];
                gIntegralMultiplicityStat->SetPoint(ip, p.mult, p.yield);
                gIntegralMultiplicityStat->SetPointError(ip, 0.0, 0.0, p.stat, p.stat);
                gIntegralMultiplicitySys->SetPoint(ip, p.mult, p.yield);
                gIntegralMultiplicitySys->SetPointError(ip, 0.0, 0.0, p.syst, p.syst);
                multMin = std::min(multMin, p.mult);
                multMax = std::max(multMax, p.mult);
                const double e = std::hypot(p.stat, p.syst);
                if (p.yield > 0.0) yMultMin = std::min(yMultMin, std::max(1e-20, p.yield - e));
                yMultMax = std::max(yMultMax, p.yield + e);
            }
            if (!std::isfinite(multMin) || !(multMax > multMin)) {
                multMin = 0.0;
                multMax = std::max(1.0, multMax);
            }
            if (!std::isfinite(yMultMin)) yMultMin = 0.0;
            if (!(yMultMax > yMultMin)) yMultMax = yMultMin + 1e-12;
            const double xPad = 0.08 * (multMax - multMin);
            const double boxHalfWidth = 0.018 * (multMax - multMin);

            cIntegralMultiplicity = std::make_unique<TCanvas>("c_integral_yield_vs_multiplicity", "", 900, 700);
            cIntegralMultiplicity->SetLeftMargin(0.14);
            cIntegralMultiplicity->SetBottomMargin(0.12);
            cIntegralMultiplicity->SetRightMargin(0.05);
            cIntegralMultiplicity->SetTopMargin(0.06);
            cIntegralMultiplicity->SetTicks(1, 1);
            hFrameMult = std::make_unique<TH1D>("h_frame_integral_yield_vs_multiplicity",
                                                ";#LTd#it{N}_{ch}/d#eta#GT;Integrated yield",
                                                100,
                                                std::max(0.0, multMin - xPad),
                                                multMax + xPad);
            hFrameMult->SetDirectory(nullptr);
            hFrameMult->SetStats(false);
            hFrameMult->GetYaxis()->SetRangeUser(std::max(0.0, yMultMin * 0.75), yMultMax * 1.35);
            hFrameMult->Draw("AXIS");
            for (const auto &p : integralMultiplicityPoints) {
                if (!std::isfinite(p.yield) || !std::isfinite(p.syst) || p.syst <= 0.0) continue;
                TBox box(std::max(0.0, p.mult - boxHalfWidth),
                         std::max(0.0, p.yield - p.syst),
                         p.mult + boxHalfWidth,
                         p.yield + p.syst);
                box.SetFillStyle(0);
                box.SetLineColor(kAzure + 2);
                box.SetLineStyle(kDashed);
                box.SetLineWidth(2);
                box.DrawClone("l");
            }
            gIntegralMultiplicityStat->Draw("P SAME");
            {
                TLegend leg(0.56, 0.74, 0.90, 0.90);
                leg.SetBorderSize(0);
                leg.SetFillStyle(0);
                leg.SetTextFont(42);
                leg.SetTextSize(0.035);
                leg.AddEntry(gIntegralMultiplicityStat.get(), "Integrated yield (stat.)", "lep");
                leg.AddEntry(gIntegralMultiplicitySys.get(), "Total syst.", "l");
                leg.DrawClone();
            }
            {
                TPaveText text(0.16, 0.64, 0.56, 0.90, "NDC");
                text.SetBorderSize(0);
                text.SetFillStyle(0);
                text.SetTextAlign(12);
                text.SetTextFont(42);
                text.SetTextSize(0.035);
                if (runCfg.plotLabels.usePerformance && !runCfg.plotLabels.performanceLabel.empty()) {
                    text.AddText(runCfg.plotLabels.performanceLabel.c_str());
                }
                if (!runCfg.plotLabels.collisionSystem.empty() || !runCfg.plotLabels.collisionEnergy.empty()) {
                    text.AddText((runCfg.plotLabels.collisionSystem + " " + runCfg.plotLabels.collisionEnergy).c_str());
                }
                if (!runCfg.plotLabels.period.empty() || !runCfg.plotLabels.periodMark.empty()) {
                    text.AddText((runCfg.plotLabels.period + " " + runCfg.plotLabels.periodMark).c_str());
                }
                const std::string decay = BuildDecayString(runCfg.isMatter);
                if (!decay.empty()) {
                    text.AddText(decay.c_str());
                }
                text.DrawClone();
            }
        }

        auto cSysFrac = std::make_unique<TCanvas>("c_integral_syst_fraction_summary", "", 900, 700);
        cSysFrac->SetLeftMargin(0.12);
        cSysFrac->SetBottomMargin(0.14);
        hSystFractionSummary->SetFillColor(kOrange - 3);
        hSystFractionSummary->SetLineColor(kBlack);
        hSystFractionSummary->Draw("HIST");

        auto cSysFracVsCen = std::make_unique<TCanvas>("c_integral_syst_fraction_vs_centrality", "", 900, 700);
        cSysFracVsCen->SetLeftMargin(0.12);
        cSysFracVsCen->SetBottomMargin(0.12);
        double yFracMax = 0.0;
        for (const auto &hFrac : hSystFractionVsCen) {
            if (hFrac) yFracMax = std::max(yFracMax, hFrac->GetMaximum());
        }
        if (yFracMax <= 0.0) yFracMax = 1.0;
        hSystFractionVsCen.back()->GetYaxis()->SetRangeUser(0.0, yFracMax * 1.25);
        hSystFractionVsCen.back()->Draw("HIST");
        for (size_t i = 0; i + 1 < hSystFractionVsCen.size(); ++i) {
            hSystFractionVsCen[i]->Draw("HIST SAME");
        }
        hSystFractionVsCen.back()->Draw("HIST SAME");
        {
            TLegend leg(0.56, 0.58, 0.90, 0.90);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            for (size_t i = 0; i < hSystFractionVsCen.size(); ++i) {
                leg.AddEntry(hSystFractionVsCen[i].get(),
                             (systNames[i] + (i == hSystFractionVsCen.size() - 1 ? " (quadrature)" : "")).c_str(),
                             "l");
            }
            leg.DrawClone();
        }

        TDirectory *summaryDir = fout.GetDirectory("summary");
        if (!summaryDir) summaryDir = fout.mkdir("summary");
        if (summaryDir) {
            TDirectory *intDir = summaryDir->GetDirectory("integral_yield");
            if (!intDir) intDir = summaryDir->mkdir("integral_yield");
            if (intDir) {
                intDir->cd();
                hIntegralStat->Write("h_integral_yield_stat", TObject::kOverwrite);
                hIntegralSys->Write("h_integral_yield_sys", TObject::kOverwrite);
                hIntegralSysRatio->Write("h_integral_yield_sys_ratio", TObject::kOverwrite);
                hSystFractionSummary->Write("h_integral_syst_fraction_summary", TObject::kOverwrite);
                for (size_t i = 0; i < hSystFractionVsCen.size(); ++i) {
                    if (hSystFractionVsCen[i]) {
                        hSystFractionVsCen[i]->Write(("h_integral_syst_fraction_vs_centrality_" + systNames[i]).c_str(), TObject::kOverwrite);
                    }
                }
                gIntegralSys->Write("g_integral_yield_sys", TObject::kOverwrite);
                if (gIntegralMultiplicityStat) {
                    gIntegralMultiplicityStat->Write("g_integral_yield_vs_multiplicity_stat", TObject::kOverwrite);
                }
                if (gIntegralMultiplicitySys) {
                    gIntegralMultiplicitySys->Write("g_integral_yield_vs_multiplicity_sys", TObject::kOverwrite);
                }
                cIntegral->Write("c_integral_yield_vs_centrality", TObject::kOverwrite);
                if (cIntegralMultiplicity) {
                    cIntegralMultiplicity->Write("c_integral_yield_vs_multiplicity", TObject::kOverwrite);
                }
                cSysFrac->Write("c_integral_syst_fraction_summary", TObject::kOverwrite);
                cSysFracVsCen->Write("c_integral_syst_fraction_vs_centrality", TObject::kOverwrite);
            }
        }
    }

    if (modeName == "pt_ct" && hTauPerPt) {
        fout.cd();
        hTauPerPt->Write("tau_per_ptbin", TObject::kOverwrite);
        auto cTau = std::make_unique<TCanvas>("c_tau_per_ptbin", "c_tau_per_ptbin", 960, 720);
        cTau->SetLeftMargin(0.14);
        cTau->SetBottomMargin(0.12);
        cTau->SetTicks(1, 1);
        cTau->SetGridy(true);
        hTauPerPt->SetMarkerStyle(20);
        hTauPerPt->SetMarkerColor(kBlack);
        hTauPerPt->SetLineColor(kBlack);
        hTauPerPt->Draw("E1");
        cTau->Write("c_tau_per_ptbin", TObject::kOverwrite);
    }

    if (runCfg.onTheFlyChecksEnabled) {
        WriteOnTheFlyChecksOutput(runCfg.onTheFlyChecksRoot,
                                  onTheFlyQaColumns,
                                  onTheFlyQaCandidates,
                                  runCfg.onTheFlyVariables,
                                  runCfg.oneBinQaPairs,
                                  runCfg.oneBinQaAxisPool,
                                  runCfg.saveChecksPdf,
                                  runCfg.onTheFlyChecksPdfDir);
    }

    fout.Close();
    SaveResultsToPdfIfRequested(runCfg);

    return 0;
}

} // namespace

int UnifiedTaskRunner::RunBdtSpectrum(const GeneralHelper::Json &cfg, const BinPlan &plan) const {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
    return RunSpectrumMode(cfg, plan, "bdt_spectrum");
}

int UnifiedTaskRunner::RunTopologySpectrum(const GeneralHelper::Json &cfg, const BinPlan &plan) const {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
    return RunSpectrumMode(cfg, plan, "topology_spectrum");
}

int UnifiedTaskRunner::RunCtExtraction(const GeneralHelper::Json &cfg, const BinPlan &plan) const {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
    return RunSpectrumMode(cfg, plan, "pt_ct");
}

int UnifiedTaskRunner::RunCtSingle(const GeneralHelper::Json &cfg, const BinPlan &plan) const {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
    return RunSpectrumMode(cfg, plan, "ct_single");
}

int UnifiedTaskRunner::Run(const GeneralHelper::Json &cfg,
                           const ModePolicy &policy,
                           const BinPlan &plan) const {
    if (policy.mode == "bdt_spectrum") {
        return RunBdtSpectrum(cfg, plan);
    }
    if (policy.mode == "topology_spectrum") {
        return RunTopologySpectrum(cfg, plan);
    }
    if (policy.mode == "pt_ct") {
        return RunCtExtraction(cfg, plan);
    }
    if (policy.mode == "ct_single") {
        return RunCtSingle(cfg, plan);
    }
    throw std::runtime_error("UnifiedTaskRunner does not support mode: " + policy.mode);
}

} // namespace UnifiedAnalysis
