#include "UnifiedTaskRunner.h"

#include "BinningCorrectionHelper.h"
#include "ProcessOneBin.h"
#include "SpectrumPlotHelper.h"
#include "../checks/ChecksConfig.h"

#include <ROOT/RDataFrame.hxx>
#include <RooMsgService.h>
#include <RooPlot.h>
#include <TCanvas.h>
#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TTree.h>

#include "../../include/AliPWGFunc.h"
#include "../../include/AliPWGFunc.cxx"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
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

std::string BuildRangeLabel(const std::string &prefix, double v1, double v2) {
    std::ostringstream os;
    os << prefix << '_' << v1 << '_' << v2;
    return os.str();
}

std::string ResolveWpFile(const GeneralHelper::Json &cfg, const std::string &mode) {
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto binning = analysis.value("binning", GeneralHelper::Json::object());
    const auto profiles = binning.value("mode_profiles", GeneralHelper::Json::object());
    const auto profile = profiles.value(mode, GeneralHelper::Json::object());
    const std::string wpFromProfile = GetS(profile, "wp_file", "");
    if (!wpFromProfile.empty()) return wpFromProfile;

    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const std::string wpDir = GetS(path, "working_point_dir", "");
    if (wpDir.empty()) return std::string();

    if (mode == "bdt_spectrum" || mode == "topology_spectrum") {
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
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto binning = analysis.value("binning", GeneralHelper::Json::object());
    const auto profiles = binning.value("mode_profiles", GeneralHelper::Json::object());
    const auto profile = profiles.value(mode, GeneralHelper::Json::object());

    std::string dir = GetS(profile, "score_efficiency_dir", "");
    if (!dir.empty()) return dir;

    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    dir = GetS(path, "score_efficiency_dir", "");
    if (!dir.empty()) return dir;

    const auto syst = analysis.value("systematics", GeneralHelper::Json::object());
    dir = GetS(syst, "syst_efficiency_array_path", "");
    return dir;
}

double GetNEvents(const GeneralHelper::Json &cfg, double cenMin, double cenMax) {
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto eventHist = common.value("event_hist", GeneralHelper::Json::object());
    const std::string filePath = GetS(path, "analysis_results_file", "");
    const std::string histPath = GetS(eventHist, "n_events_hist", "");
    if (filePath.empty() || histPath.empty()) return 0.0;

    TFile f(filePath.c_str(), "READ");
    if (f.IsZombie()) return 0.0;
    TH1 *h = dynamic_cast<TH1 *>(f.Get(histPath.c_str()));
    if (!h) return 0.0;
    const int bmin = h->GetXaxis()->FindBin(cenMin + 1e-3);
    const int bmax = h->GetXaxis()->FindBin(cenMax - 1e-3);
    return h->Integral(bmin, bmax);
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
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCentrality;
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
    int nBinsForFit{100};
    int systBdtScoreNPoints{20};
    double systThrChi2Ndf{2.0};
    double systThrSignificance{2.5};
    std::vector<std::string> systBkgFuncs;
    std::vector<std::string> systSigFuncs;
    std::vector<std::string> systAbsorptionFiles;
    std::vector<std::string> systAbsorptionFileLabels;

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
    const auto correction = analysis.value("correction", GeneralHelper::Json::object());
    const auto syst = analysis.value("systematics", GeneralHelper::Json::object());
    const auto binning = analysis.value("binning", GeneralHelper::Json::object());
    const auto profiles = binning.value("mode_profiles", GeneralHelper::Json::object());
    const auto profile = profiles.value(mode, GeneralHelper::Json::object());

    out.dataTree = GetS(trees, "data", "O2hypcands");
    out.mcTree = GetS(trees, "mc", "O2mchypcands");
    out.absorptionTree = GetS(trees, "absorption", "he3candidates");
    out.wpFile = ResolveWpFile(cfg, mode);
    out.scoreEffDir = ResolveScoreEffDir(cfg, mode);
    out.bkgFunc = GetS(fit, "bkg_fit_func", "pol2");
    out.sigFunc = GetS(fit, "signal_fit_func", "dscb");
    out.isMatter = GetS(selection, "is_matter", "both");
    out.basicSelectionDataForMcEff = GetS(commonSel, "basic_selection_data_for_mc_eff", "");
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
    if (correction.contains("onebin_qa_columns") && correction["onebin_qa_columns"].is_array()) {
        out.oneBinQaColumns = correction["onebin_qa_columns"].get<std::vector<std::string>>();
    } else if (onFlyChecks.contains("variables") && onFlyChecks["variables"].is_array()) {
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
    if (profile.contains("pt_bins") && profile["pt_bins"].is_array()) {
        out.ptBins = profile["pt_bins"].get<std::vector<double>>();
    }
    if (profile.contains("ct_bins_by_pt") && profile["ct_bins_by_pt"].is_array()) {
        out.ctBinsByPt = profile["ct_bins_by_pt"].get<std::vector<std::vector<double>>>();
    }
    out.addAbsorptionCorrectionPtCt = profile.value("add_absorption_correction", false);
    out.mcFileForAcceptance = GetS(path, "mc_file_for_acceptance", "");
    if (profile.contains("cen_bins") && profile["cen_bins"].is_array()) {
        out.cenBins = profile["cen_bins"].get<std::vector<double>>();
    }
    if (profile.contains("pt_bins_by_centrality") && profile["pt_bins_by_centrality"].is_array()) {
        out.ptBinsByCentrality = profile["pt_bins_by_centrality"].get<std::vector<std::vector<double>>>();
    }
    out.mcFileForAbsorption = GetS(path, "mc_file_for_absorption", "");
    out.originalCtaoAbsorption = GetD(params, "original_ctao_absorption", out.originalCtaoAbsorption);

    const bool doSystExec = execution.value("do_systematics", correction.value("do_systematics", false));
    out.doSystematics = doSystExec && mode != "topology_spectrum";
    out.randomSeed = GetI(syst, "random_seed", out.randomSeed);
    out.systNtrails = GetI(syst, "syst_ntrails", out.systNtrails);
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

    out.plotLabels.usePerformance = tags.value("use_performance", false);
    out.plotLabels.performanceLabel = GetS(tags, "performance_label", "");
    out.plotLabels.period = GetS(tags, "period", "");
    out.plotLabels.periodMark = GetS(tags, "period_mark", "");
    out.plotLabels.collisionSystem = GetS(tags, "collision_system", "");
    out.plotLabels.collisionEnergy = GetS(tags, "collision_energy", "");

    const std::string period = GetS(tags, "period", "period");
    const std::string periodMark = GetS(tags, "period_mark", "mark");
    const std::string periodTag = period + "_" + periodMark;
    const std::string outputDir = GetS(path, "output_dir", "./Outputs") + "/" + periodTag + "/" + mode + "/" + out.isMatter;
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

    if (mode == "bdt_spectrum" || mode == "topology_spectrum") {
        const double base = raw / accVal / absoVal / eff;
        const double baseErr = rawErr / accVal / absoVal / eff;
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
    cfg.ptBins = runCfg.ptBins;
    cfg.ctBinsByPt = runCfg.ctBinsByPt;
    return cfg;
}

struct SysBinArtifacts {
    std::string subBinName;
    std::unique_ptr<TH1D> hCorrDist;
    std::unique_ptr<TH1D> hTrailBdtEff;
    std::unique_ptr<TH1D> hCorrVsAbso;
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
                      TH1D *hFitSigma,
                      TH1D *hFitMean,
                      TH1D *hFitChi2,
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
        if (hFitSigma) hFitSigma->Write("h_fit_sigma", TObject::kOverwrite);
        if (hFitMean) hFitMean->Write("h_fit_mean", TObject::kOverwrite);
        if (hFitChi2) hFitChi2->Write("h_fit_chi2ndf", TObject::kOverwrite);
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
    if (cSys) cSys->Write("c_sys_details", TObject::kOverwrite);
    auto cFinal = MakeFinalSpectrumCanvas("c_final_spectrum",
                                          hFinalStat,
                                          gFinalSys,
                                          finalStdFit,
                                          finalXAxisTitle,
                                          finalExtraText,
                                          finalLabelCfg ? *finalLabelCfg : PlotLabelConfig{},
                                          finalIsMatter,
                                          finalNEvents);
    if (cFinal) cFinal->Write("c_final_spectrum", TObject::kOverwrite);
}


void AppendCorrectionsCsv(const std::string &csvPath,
                          const std::string &mode,
                          const std::string &group,
                          const BinPlanItem &item,
                          const GeneralHelper::MassFitResult &fitRes,
                          const CorrectedCountsResult &corrRes) {
    const bool exists = std::filesystem::exists(csvPath);
    std::ofstream ofs(csvPath, std::ios::app);
    if (!ofs.is_open()) return;

    if (!exists) {
        ofs << "mode,group,label,raw,raw_err,chi2_ndf,significance,corrected,corrected_err,";
        ofs << "bdt_efficiency,acceptance,absorption,branching_ratio,delta_rapidity,matter_ratio,n_events,bin_width\n";
    }

    ofs << mode << ',' << group << ',' << item.label << ','
        << fitRes.signal << ',' << fitRes.signalErr << ','
        << fitRes.chi2Data << ',' << fitRes.significance << ','
        << corrRes.value << ',' << corrRes.error << ','
        << corrRes.bdtEfficiency << ','
        << corrRes.acceptance << ','
        << corrRes.absorption << ','
        << corrRes.branchingRatio << ','
        << corrRes.deltaRapidity << ','
        << corrRes.matterRatio << ','
        << corrRes.nEvents << ','
        << corrRes.binWidth << '\n';
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
    out.hTrailBdtEff = std::make_unique<TH1D>(("h_trail_bdt_eff_" + item.label).c_str(), ";trail index;BDT efficiency", static_cast<int>(nUse), 0.5, static_cast<double>(nUse) + 0.5);
    out.hTrailBdtEff->SetDirectory(nullptr);
    out.hTrailBdtEff->SetStats(false);

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

    for (size_t i = 0; i < nUse; ++i) {
        const auto [ibdt, ibkg, isig] = combos[i];

        GeneralHelper::MassFitResult trialFit;
        try {
            trialFit = GeneralHelper::FitMassSpectrum(*dataMassCache[ibdt], *mcMass, runCfg.fitCfg, bkgFuncs[ibkg], sigFuncs[isig]);
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
            nEvents,
            runCfg.branchingRatio,
            runCfg.deltaRap,
            runCfg.isMatter,
            runCfg.addAbsorptionCorrectionPtCt);
        const double corr = corrRes.value;

        const double corrErr = corrRes.error;
        if (out.hTrailBdtEff) out.hTrailBdtEff->SetBinContent(static_cast<int>(i + 1), bdtCandidates[ibdt].efficiency);

        if (trialFit.frame) {
            const std::string frameName = "trail_frame_" + item.label + "_" + std::to_string(i);
            auto *cloned = dynamic_cast<RooPlot *>(trialFit.frame->Clone(frameName.c_str()));
            if (cloned) out.trailFrames.emplace_back(cloned);
        }

        const bool pass = trialFit.chi2Data < runCfg.systThrChi2Ndf &&
                  trialFit.significance > runCfg.systThrSignificance &&
                          std::isfinite(corr);
        if (pass && out.hCorrDist) out.hCorrDist->Fill(corr);

        if (trailLog && trailLog->is_open()) {
            (*trailLog)
                << (i + 1) << ','
                << group.dirName << ','
                << item.label << ','
                << bdtCandidates[ibdt].score << ','
                << bdtCandidates[ibdt].efficiency << ','
                << bkgFuncs[ibkg] << ','
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
                                                      item.label);
    }

    if (!enableAbsorptionSystematic) {
        out.hCorrVsAbso = std::make_unique<TH1D>(("h_corr_vs_abso_" + item.label).c_str(),
                                                 ";n x #sigma_{He3};Corrected counts",
                                                 1,
                                                 0.5,
                                                 1.5);
        out.hCorrVsAbso->SetDirectory(nullptr);
        out.hCorrVsAbso->SetStats(false);
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
        double mean = 0.0;
        for (double v : corrVariants) mean += v;
        mean /= static_cast<double>(corrVariants.size());
        double var = 0.0;
        for (double v : corrVariants) {
            const double d = v - mean;
            var += d * d;
        }
        var /= static_cast<double>(corrVariants.size());
        out.absorptionRms = std::sqrt(std::max(0.0, var));
    }

    if (out.hCorrVsAbso) {
        const std::string cName = "c_absorption_source_" + item.label;
        out.cCorrVsAbso = std::make_unique<TCanvas>(cName.c_str(), cName.c_str(), 960, 720);
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
                                       TH1D *hCorr) {
    BlastWavePostFit out;
    if (!hCorr || group.items.empty()) return out;

    const double fitMin = group.items.front().ptMin;
    const double fitMax = group.items.back().ptMax;
    if (!(fitMax > fitMin)) return out;

    const std::string cleanName = SanitizeName(group.dirName);
    static std::atomic<unsigned long> sBwFitCallId{0};
    static std::mutex sBwFitMutex;
    const unsigned long fitId = sBwFitCallId.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::string tmpName = "f_bgbw_tmp_" + cleanName + "_" + std::to_string(fitId);
    AliPWGFunc bwHelper;
    bwHelper.SetVarType(AliPWGFunc::kdNdpt);
    TF1 *raw = bwHelper.GetBGBW(2.991, 0.6, 0.12, 0.5, 1.0,
                                tmpName.c_str());
    if (!raw) return out;
    raw->SetRange(0.0, 10.0);
    raw->SetNpx(1200);
    raw->SetParameter(1, 0.6);
    raw->SetParameter(2, 0.2);
    raw->SetParameter(3, 1.0);
    raw->SetParameter(4, 1.0);

    raw->SetParLimits(1, 0.2, 0.9);
    raw->SetParLimits(2, 0.1, 0.6);
    raw->SetParLimits(3, 0.01, 5.0);
    // Requested norm scale: ~1e-2 to 1e3, typical around 1e0.
    raw->SetParLimits(4, 1e-2, 1e3);

    const bool imtWasEnabled = ROOT::IsImplicitMTEnabled();
    const unsigned int imtThreads = ROOT::GetThreadPoolSize();
    if (imtWasEnabled) ROOT::DisableImplicitMT();
    {
        // Keep BW fit thread-safe under ROOT implicit MT.
        std::lock_guard<std::mutex> lock(sBwFitMutex);
        hCorr->Fit(raw, "SRQ0", "", fitMin, fitMax);
    }
    if (imtWasEnabled) ROOT::EnableImplicitMT(imtThreads > 0 ? imtThreads : 1);

    auto *fittedClone = dynamic_cast<TF1 *>(raw->Clone(("f_bgbw_" + cleanName).c_str()));
    if (!fittedClone) return out;
    out.fit.reset(fittedClone);

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
    const std::string csvPath = std::filesystem::path(runCfg.outputRoot).parent_path().string() + "/corrections_all.csv";

    // Clear previous CSV before appending new rows for this run.
    {
        std::error_code ec;
        std::filesystem::remove(csvPath, ec);
    }

    const bool isSpectrumMode = (modeName == "bdt_spectrum" || modeName == "topology_spectrum");
    const bool isLifetimeMode = (modeName == "ct_single" || modeName == "pt_ct");
    const bool applyBrSystematic = runCfg.doSystematics &&
                                   (isSpectrumMode || isLifetimeMode) &&
                                   runCfg.branchingRatioFractionalUncertainty > 0.0;
    const std::string periodTag = runCfg.plotLabels.period + "_" + runCfg.plotLabels.periodMark;
    double totalRawOverNevts = 0.0;
    double totalEventsOverNevts = 0.0;
    double totalRawErr2OverNevts = 0.0;

    std::filesystem::create_directories(std::filesystem::path(runCfg.outputRoot).parent_path());
    TFile fout(runCfg.outputRoot.c_str(), "UPDATE");
    if (fout.IsZombie()) {
        throw std::runtime_error("Cannot open output ROOT: " + runCfg.outputRoot);
    }

    PtCtAcceptanceCache ptCtAccCache;
    PtCtAbsorptionCache ptCtAbsoCache;
    SpectrumAcceptanceCache spectrumAccCache;
    if (modeName == "pt_ct") {
        ptCtAccCache = BuildPtCtAcceptanceCache(corrCfg, runCfg.mcFileForAcceptance);
        if (runCfg.addAbsorptionCorrectionPtCt) {
            ptCtAbsoCache = BuildPtCtAbsorptionCache(corrCfg, runCfg.mcFileForAbsorption);
        }
    } else if (isSpectrumMode) {
        const std::string mcFileForAccSpectrum = ResolveAcceptanceMcFileForGroup(runCfg, GroupContext{});
        spectrumAccCache = BuildSpectrumAcceptanceCache(corrCfg, mcFileForAccSpectrum);
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
        hRaw->SetDirectory(nullptr);
        hCorr->SetDirectory(nullptr);
        hBdtEff->SetDirectory(nullptr);
        hAcc->SetDirectory(nullptr);
        hAbso->SetDirectory(nullptr);
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
            oneBinOpt.throwOnError = false;
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
                    fitFrames.emplace_back(cloned);
                    const std::string canvasName = "canvas_" + frameName;
                    auto cData = MakeDecoratedFitCanvas(canvasName,
                                                        cloned,
                                                        false,
                                                        runCfg.plotLabels,
                                                        runCfg.isMatter,
                                                        groupNEvents);
                    if (cData) {
                        cData->SetTitle(("Invariant mass fit " + item.label).c_str());
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

            const auto corrRes = ComputeCorrectedCounts(
                modeName,
                item,
                fitRes,
                acc,
                abso,
                bdtEff,
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

            AppendCorrectionsCsv(csvPath, modeName, group.dirName, item, fitRes, corrRes);

            if (runCfg.doSystematics) {
                auto sysOut = RunSystematicsForBin(runCfg, group, item, rdfCache,
                                                   corr,
                                                   hCorr->GetBinError(ib),
                                                   acc,
                                                   abso,
                                                   groupNEvents,
                                                   ib,
                                                   runCfg.doSystematics ? &trailLog : nullptr,
                                                   &absoVecByFile,
                                                   enableAbsorptionSystematic);
                hSysSelection->SetBinContent(ib, sysOut.selectionRms);
                hSysAbsorption->SetBinContent(ib, sysOut.absorptionRms);
                sysArtifacts.push_back(std::move(sysOut));
            }
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
            auto bwOut = BuildBlastWavePostFit(group, hCorr.get());
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
                         hFitSigma.get(),
                         hFitMean.get(),
                         hFitChi2.get(),
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
