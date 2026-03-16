#include <ROOT/RDataFrame.hxx>

#include <RooAbsReal.h>
#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TMath.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "../Tools/GeneralHelper.hpp"

namespace {

using json = nlohmann::json;

struct BinKey {
    double ptMin{0.0};
    double ptMax{0.0};
    double ctMin{0.0};
    double ctMax{0.0};

    std::string ToString() const {
        std::ostringstream ss;
        ss << "pt_" << std::fixed << std::setprecision(3) << ptMin
           << "_" << ptMax << "__ct_" << ctMin << "_" << ctMax;
        return ss.str();
    }

    bool operator<(const BinKey &other) const {
        return std::tie(ptMin, ptMax, ctMin, ctMax) <
               std::tie(other.ptMin, other.ptMax, other.ctMin, other.ctMax);
    }
};

struct WorkingPoint {
    double score{0.0};
    double efficiency{0.0};
    double significance{0.0};
};

struct Config {
    std::string dataSnapshotDir;
    std::string snapshotTreeName;
    std::string snapshotPattern;
    std::string mcSnapshotDir;
    std::string mcSnapshotTreeName;
    std::string mcSnapshotPattern;
    std::string workingPointFile;
    std::string outputDir;
    std::string isMatter{"both"};
    std::string massColumn{"fMassH3L"};
    std::string bdtScoreColumn{"model_output"};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> ctBins;
    std::vector<double> massRange{2.95, 3.05};
    int massBinsData{50};
    int massBinsMc{80};
    double minEntriesForFit{60.0};
    double bdtScoreShift{0.0};
};

struct FitOutputs {
    double sigmaData{0.0};
    double sigmaDataErr{0.0};
    double sigmaMc{0.0};
    double sigmaMcErr{0.0};
    double ratio{0.0};
    double ratioErr{0.0};
    double signalYield3Sigma{0.0};
    double signalYield3SigmaErr{0.0};
    double backgroundYield3Sigma{0.0};
    double backgroundYield3SigmaErr{0.0};
    double significance3Sigma{0.0};
    double significance3SigmaErr{0.0};
    size_t entriesData{0};
    size_t entriesMc{0};
    std::unique_ptr<RooPlot> dataFrame;
    std::unique_ptr<RooPlot> mcFrame;
};

struct BinSample {
    size_t ptIndex{0};
    size_t ctIndex{0};
    BinKey key;
    std::vector<double> dataMasses;
    std::vector<double> mcMasses;
    int entriesBefore{0};
    int entriesAfter{0};
};

class SigmaMcDataMatcher {
public:
    SigmaMcDataMatcher(const std::string &configPath,
                       const std::string &outputOverride);

    void Run();

private:
    static constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

    Config fCfg;
    json fCfgJson;
    std::filesystem::path fConfigDir;
    std::filesystem::path fCombinedDir;
    std::filesystem::path fPerBinDir;

    std::map<BinKey, WorkingPoint> fWorkingPoints;
    std::map<BinKey, double> fOverrides;

    std::vector<BinSample> fSamples;
    std::vector<std::vector<size_t>> fSampleIndexGrid;

    static std::string FormatEdge(double value);
    std::string ExpandPattern(const std::string &pattern, const BinKey &key) const;
    std::string BuildPath(const std::string &dir,
                          const std::string &pattern,
                          const BinKey &key) const;
    std::string ResolvePath(const std::string &path) const;

    void LoadConfig(const std::string &configPath);
    void LoadWorkingPoints();
    void PrepareOutputDirs(const std::string &overridePath);

    WorkingPoint GetWorkingPoint(const BinKey &key) const;
    double ResolveBDTScore(const BinKey &key) const;

    void LoadSamples();
    std::vector<double> CollectMassValues(const BinKey &key,
                                          double bdtCut,
                                          bool applyBdt,
                                          bool isMc,
                                          int &entriesBefore,
                                          int &entriesAfter) const;

    FitOutputs FitMassSpectra(const std::vector<double> &dataMasses,
                              const std::vector<double> &mcMasses,
                              const std::string &tag) const;

    const BinSample &GetSample(size_t ptIndex, size_t ctIndex) const;

    void RunCombinedFit();
    void RunPerBinFits();
};

SigmaMcDataMatcher::SigmaMcDataMatcher(const std::string &configPath,
                                       const std::string &outputOverride) {
    if (configPath.empty()) {
        throw std::runtime_error("SigmaMcDataMatcher: config path is empty");
    }
    const auto configFull = std::filesystem::absolute(std::filesystem::path(configPath));
    if (!std::filesystem::exists(configFull)) {
        throw std::runtime_error("SigmaMcDataMatcher: config file not found -> " + configFull.string());
    }
    fConfigDir = configFull.parent_path();
    LoadConfig(configFull.string());
    LoadWorkingPoints();
    PrepareOutputDirs(outputOverride);
}

void SigmaMcDataMatcher::Run() {
    GeneralHelper::EnableImplicitMTWithPreferredThreads();
    if (!gROOT->IsBatch()) {
        gROOT->SetBatch(true);
    }
    LoadSamples();
    RunCombinedFit();
    RunPerBinFits();
}

std::string SigmaMcDataMatcher::FormatEdge(double value) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << value;
    std::string out = ss.str();
    while (!out.empty() && out.back() == '0') {
        out.pop_back();
    }
    if (!out.empty() && out.back() == '.') {
        out.pop_back();
    }
    if (out.empty()) {
        out = "0";
    }
    return out;
}

std::string SigmaMcDataMatcher::ExpandPattern(const std::string &pattern,
                                              const BinKey &key) const {
    std::string out = pattern;
    auto replace_all = [&](const std::string &token, double value) {
        const std::string formatted = FormatEdge(value);
        size_t pos = 0;
        while ((pos = out.find(token, pos)) != std::string::npos) {
            out.replace(pos, token.size(), formatted);
            pos += formatted.size();
        }
    };
    replace_all("%PTMIN%", key.ptMin);
    replace_all("%PTMAX%", key.ptMax);
    replace_all("%CTMIN%", key.ctMin);
    replace_all("%CTMAX%", key.ctMax);
    return out;
}

std::string SigmaMcDataMatcher::BuildPath(const std::string &dir,
                                          const std::string &pattern,
                                          const BinKey &key) const {
    std::filesystem::path base(dir);
    base /= ExpandPattern(pattern, key);
    return base.lexically_normal().string();
}

std::string SigmaMcDataMatcher::ResolvePath(const std::string &path) const {
    if (path.empty()) {
        return path;
    }
    std::filesystem::path p(path);
    if (p.is_absolute()) {
        return p.lexically_normal().string();
    }
    return (fConfigDir / p).lexically_normal().string();
}

void SigmaMcDataMatcher::LoadConfig(const std::string &configPath) {
    std::ifstream ifs(configPath);
    if (!ifs) {
        throw std::runtime_error("SigmaMcDataMatcher: failed to open config -> " + configPath);
    }
    fCfgJson = json::parse(ifs, nullptr, true, true);

    auto require_string = [&](const char *key) {
        if (!fCfgJson.contains(key)) {
            throw std::runtime_error(std::string("Missing required key '") + key + "' in config");
        }
        return fCfgJson.value(key, std::string());
    };

    fCfg.dataSnapshotDir = ResolvePath(require_string("data_snapshot_dir"));
    fCfg.snapshotTreeName = fCfgJson.value("snapshot_tree_name", "O2hypcands");
    fCfg.snapshotPattern = fCfgJson.value("snapshot_pattern", "data_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root");
    fCfg.mcSnapshotDir = ResolvePath(fCfgJson.value("mc_snapshot_dir", fCfgJson.value("data_snapshot_dir", "")));
    fCfg.mcSnapshotTreeName = fCfgJson.value("mc_snapshot_tree_name", "O2mchypcands");
    fCfg.mcSnapshotPattern = fCfgJson.value("mc_snapshot_pattern", "mc_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root");
    fCfg.workingPointFile = ResolvePath(require_string("working_point_file"));
    fCfg.outputDir = ResolvePath(fCfgJson.value("output_dir", "results/ct_extraction"));
    fCfg.isMatter = fCfgJson.value("is_matter", std::string("both"));
    fCfg.massColumn = fCfgJson.value("mass_column", std::string("fMassH3L"));
    fCfg.bdtScoreColumn = fCfgJson.value("bdt_score_column", std::string("model_output"));
    fCfg.ptBins = fCfgJson.value("pt_bins", std::vector<double>{});
    fCfg.ctBins = fCfgJson.value("ct_bins", std::vector<std::vector<double>>{});
    fCfg.massRange = fCfgJson.value("mass_range", std::vector<double>{2.95, 3.05});
    fCfg.massBinsData = fCfgJson.value("mass_nbins_data", 50);
    fCfg.massBinsMc = fCfgJson.value("mass_nbins_mc", 80);
    fCfg.minEntriesForFit = fCfgJson.value("min_entries_for_fit", 60.0);
    fCfg.bdtScoreShift = fCfgJson.value("bdt_score_shift", 0.0);

    if (fCfg.ptBins.size() < 2) {
        throw std::runtime_error("Config error: pt_bins must contain at least two edges");
    }
    if (fCfg.ctBins.size() != fCfg.ptBins.size() - 1) {
        throw std::runtime_error("Config error: ct_bins size mismatch with pt bins");
    }
    if (fCfg.massRange.size() < 2 || fCfg.massRange[0] >= fCfg.massRange[1]) {
        throw std::runtime_error("Config error: invalid mass_range");
    }
    auto ensure_exists = [](const std::string &path, const std::string &tag) {
        if (path.empty()) {
            throw std::runtime_error(tag + " is empty");
        }
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(tag + " does not exist -> " + path);
        }
    };
    ensure_exists(fCfg.dataSnapshotDir, "data_snapshot_dir");
    ensure_exists(fCfg.mcSnapshotDir, "mc_snapshot_dir");
    ensure_exists(fCfg.workingPointFile, "working_point_file");
}

void SigmaMcDataMatcher::LoadWorkingPoints() {
    std::ifstream ifs(fCfg.workingPointFile);
    if (!ifs) {
        throw std::runtime_error("Failed to open working point file -> " + fCfg.workingPointFile);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        double ptMin, ptMax, ctMin, ctMax, score, eff, sig;
        ss >> ptMin >> ptMax >> ctMin >> ctMax >> score >> eff >> sig;
        if (ss.fail()) {
            continue;
        }
        BinKey key{ptMin, ptMax, ctMin, ctMax};
        fWorkingPoints[key] = WorkingPoint{score, eff, sig};
    }

    if (fCfgJson.contains("bdt_overrides")) {
        for (const auto &entry : fCfgJson["bdt_overrides"]) {
            if (!entry.contains("pt") || !entry.contains("ct") || !entry.contains("score")) {
                continue;
            }
            auto pt = entry["pt"].get<std::vector<double>>();
            auto ct = entry["ct"].get<std::vector<double>>();
            if (pt.size() != 2 || ct.size() != 2) {
                continue;
            }
            BinKey key{pt[0], pt[1], ct[0], ct[1]};
            fOverrides[key] = entry["score"].get<double>();
        }
    }
}

void SigmaMcDataMatcher::PrepareOutputDirs(const std::string &overridePath) {
    std::filesystem::path base;
    if (!overridePath.empty()) {
        base = std::filesystem::absolute(std::filesystem::path(overridePath));
    } else {
        base = std::filesystem::path(fCfg.outputDir) / "SigmaMatching";
    }
    fCombinedDir = base / "combined";
    fPerBinDir = base / "per_bin";
    GeneralHelper::EnsureDir(fCombinedDir.string());
    GeneralHelper::EnsureDir(fPerBinDir.string());
}

WorkingPoint SigmaMcDataMatcher::GetWorkingPoint(const BinKey &key) const {
    auto it = fWorkingPoints.find(key);
    if (it == fWorkingPoints.end()) {
        throw std::runtime_error("Missing working point for bin " + key.ToString());
    }
    return it->second;
}

double SigmaMcDataMatcher::ResolveBDTScore(const BinKey &key) const {
    auto it = fOverrides.find(key);
    if (it != fOverrides.end()) {
        return it->second;
    }
    return GetWorkingPoint(key).score + fCfg.bdtScoreShift;
}

void SigmaMcDataMatcher::LoadSamples() {
    const size_t nPt = fCfg.ptBins.size() - 1;
    fSamples.clear();
    fSampleIndexGrid.assign(nPt, {});

    for (size_t ipt = 0; ipt < nPt; ++ipt) {
        const auto &ctEdges = fCfg.ctBins.at(ipt);
        if (ctEdges.size() < 2) {
            throw std::runtime_error("ct bin list too short for pt index " + std::to_string(ipt));
        }
        const size_t nCt = ctEdges.size() - 1;
        fSampleIndexGrid[ipt].assign(nCt, kInvalidIndex);

        for (size_t ict = 0; ict < nCt; ++ict) {
            BinKey key{fCfg.ptBins[ipt], fCfg.ptBins[ipt + 1], ctEdges[ict], ctEdges[ict + 1]};
            const double bdtCut = ResolveBDTScore(key);

            int before = 0;
            int after = 0;
            auto dataMasses = CollectMassValues(key, bdtCut, true, false, before, after);
            int mcBefore = 0;
            int mcAfter = 0;
            auto mcMasses = CollectMassValues(key, 0.0, false, true, mcBefore, mcAfter);

            if (static_cast<int>(mcMasses.size()) < fCfg.minEntriesForFit) {
                std::ostringstream err;
                err << "MC dataset too small for bin " << key.ToString()
                    << " (entries=" << mcMasses.size() << ")";
                throw std::runtime_error(err.str());
            }
            if (static_cast<int>(dataMasses.size()) < fCfg.minEntriesForFit) {
                std::ostringstream err;
                err << "Data dataset too small after BDT for bin " << key.ToString()
                    << " (entries=" << dataMasses.size() << ")";
                throw std::runtime_error(err.str());
            }

            BinSample sample;
            sample.ptIndex = ipt;
            sample.ctIndex = ict;
            sample.key = key;
            sample.dataMasses = std::move(dataMasses);
            sample.mcMasses = std::move(mcMasses);
            sample.entriesBefore = before;
            sample.entriesAfter = after;

            const size_t idx = fSamples.size();
            fSamples.emplace_back(std::move(sample));
            fSampleIndexGrid[ipt][ict] = idx;
        }
    }
}

std::vector<double> SigmaMcDataMatcher::CollectMassValues(const BinKey &key,
                                                          double bdtCut,
                                                          bool applyBdt,
                                                          bool isMc,
                                                          int &entriesBefore,
                                                          int &entriesAfter) const {
    const std::string &dir = isMc ? fCfg.mcSnapshotDir : fCfg.dataSnapshotDir;
    const std::string &pattern = isMc ? fCfg.mcSnapshotPattern : fCfg.snapshotPattern;
    const std::string &treeName = isMc ? fCfg.mcSnapshotTreeName : fCfg.snapshotTreeName;
    const std::string filePath = BuildPath(dir, pattern, key);

    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(std::string("Snapshot not found -> ") + filePath);
    }

    ROOT::RDataFrame df(treeName, filePath);
    ROOT::RDF::RNode node = df;
    if (fCfg.isMatter == "matter") {
        node = node.Filter("fIsMatter > 0.5");
    } else if (fCfg.isMatter == "antimatter") {
        node = node.Filter("fIsMatter < 0.5");
    }

    ROOT::RDF::RNode filtered = node;
    if (!isMc && applyBdt) {
        filtered = node.Filter([bdtCut](float score) {
            return static_cast<double>(score) > bdtCut;
        }, {fCfg.bdtScoreColumn});
    }

    auto beforeFuture = node.Count();
    auto afterFuture = filtered.Count();

    struct SlotBuffer {
        std::vector<double> masses;
    };
    std::mutex bufMutex;
    std::vector<std::unique_ptr<SlotBuffer>> buffers;

    auto acquire = [&](unsigned slot) -> SlotBuffer & {
        std::lock_guard<std::mutex> guard(bufMutex);
        if (slot >= buffers.size()) {
            buffers.resize(slot + 1);
        }
        if (!buffers[slot]) {
            buffers[slot] = std::make_unique<SlotBuffer>();
        }
        return *buffers[slot];
    };

    filtered.ForeachSlot(
        [&](unsigned slot, double mass) {
            auto &buf = acquire(slot);
            buf.masses.push_back(mass);
        },
        std::vector<std::string>{fCfg.massColumn});

    entriesBefore = static_cast<int>(beforeFuture.GetValue());
    entriesAfter = static_cast<int>(afterFuture.GetValue());

    size_t total = 0;
    for (const auto &slot : buffers) {
        if (!slot) {
            continue;
        }
        total += slot->masses.size();
    }

    std::vector<double> masses;
    masses.reserve(total);
    for (const auto &slot : buffers) {
        if (!slot) {
            continue;
        }
        masses.insert(masses.end(), slot->masses.begin(), slot->masses.end());
    }
    return masses;
}

FitOutputs SigmaMcDataMatcher::FitMassSpectra(const std::vector<double> &dataMasses,
                                              const std::vector<double> &mcMasses,
                                              const std::string &tag) const {
    FitOutputs out;
    const double massMin = fCfg.massRange[0];
    const double massMax = fCfg.massRange[1];

    auto massVar = std::make_shared<RooRealVar>("mass", "invariant mass", massMin, massMax);
    RooRealVar &mass = *massVar;
    RooArgSet vars(mass);

    RooDataSet dataSet("data", "data", vars);
    for (double value : dataMasses) {
        if (value < massMin || value > massMax) {
            continue;
        }
        mass.setVal(value);
        dataSet.add(vars);
    }

    RooDataSet mcSet("mc", "mc", vars);
    for (double value : mcMasses) {
        if (value < massMin || value > massMax) {
            continue;
        }
        mass.setVal(value);
        mcSet.add(vars);
    }

    if (static_cast<int>(dataSet.numEntries()) < fCfg.minEntriesForFit) {
        throw std::runtime_error("Data entries below threshold for tag " + tag);
    }
    if (static_cast<int>(mcSet.numEntries()) < fCfg.minEntriesForFit) {
        throw std::runtime_error("MC entries below threshold for tag " + tag);
    }

    RooRealVar alphaL("alphaL", "alphaL", 1.5, 0.1, 10.0);
    RooRealVar nL("nL", "nL", 5.0, 0.5, 30.0);
    RooRealVar alphaR("alphaR", "alphaR", 1.5, 0.1, 10.0);
    RooRealVar nR("nR", "nR", 5.0, 0.5, 30.0);
    RooRealVar meanMC("meanMC", "meanMC", 2.991, massMin, massMax);
    RooRealVar sigmaMC("sigmaMC", "sigmaMC", 1.5e-3, 1.1e-3, 1.8e-3);

    RooCrystalBall signalMC("signalMC", "signalMC", mass, meanMC, sigmaMC, alphaL, nL, alphaR, nR);
    signalMC.fitTo(mcSet, RooFit::Save(true), RooFit::PrintLevel(-1));

    alphaL.setConstant(true);
    nL.setConstant(true);
    alphaR.setConstant(true);
    nR.setConstant(true);
    sigmaMC.setConstant(true);

    std::unique_ptr<RooPlot> mcFrame(mass.frame(fCfg.massBinsMc));
    mcSet.plotOn(mcFrame.get());
    signalMC.plotOn(mcFrame.get(),
                    RooFit::LineColor(kRed + 1),
                    RooFit::LineWidth(2),
                    RooFit::Name("signalMC"));

    constexpr int nMcFloatParams = 6;
    const int ndfMc = std::max(1, static_cast<int>(fCfg.massBinsMc) - nMcFloatParams);
    const double chi2Mc = mcFrame->chiSquare("signalMC", nullptr, nMcFloatParams);
    auto mcInfo = std::make_unique<TPaveText>(0.58, 0.55, 0.9, 0.88, "NDC");
    mcInfo->SetBorderSize(0);
    mcInfo->SetFillStyle(0);
    mcInfo->SetTextAlign(11);
    mcInfo->SetTextFont(42);
    mcInfo->AddText(Form("#mu = %.2f #pm %.2f MeV/#it{c}^{2}",
                         meanMC.getVal() * 1e3,
                         meanMC.getError() * 1e3));
    mcInfo->AddText(Form("#sigma = %.2f #pm %.2f MeV/#it{c}^{2}",
                         sigmaMC.getVal() * 1e3,
                         sigmaMC.getError() * 1e3));
    mcInfo->AddText(Form("#alpha_{L} = %.2f #pm %.2f",
                         alphaL.getVal(),
                         alphaL.getError()));
    mcInfo->AddText(Form("#alpha_{R} = %.2f #pm %.2f",
                         alphaR.getVal(),
                         alphaR.getError()));
    mcInfo->AddText(Form("n_{L} = %.2f #pm %.2f",
                         nL.getVal(),
                         nL.getError()));
    mcInfo->AddText(Form("n_{R} = %.2f #pm %.2f",
                         nR.getVal(),
                         nR.getError()));
    mcInfo->AddText(Form("#chi^{2}/NDF = %.3f (NDF: %d)", chi2Mc, ndfMc));
    mcInfo->AddText(Form("Entries = %d", mcSet.numEntries()));
    mcFrame->addObject(mcInfo.release());

    RooRealVar mean("mean", "mean", meanMC.getVal(), massMin, massMax);
    RooRealVar sigma("sigma", "sigma", sigmaMC.getVal(), 5e-4, 1e-2);
    RooCrystalBall signal("signal", "signal", mass, mean, sigma, alphaL, nL, alphaR, nR);

    RooRealVar c0("c0", "c0", 0.0, -1.5, 1.5);
    RooRealVar c1("c1", "c1", 0.0, -1.5, 1.5);
    RooRealVar c2("c2", "c2", 0.0, -1.5, 1.5);
    RooChebychev background("background", "background", mass, RooArgList(c0, c1, c2));

    RooRealVar nsig("nsig", "signal yield", std::max(1.0, dataSet.sumEntries() * 0.5),
                    0.0, std::max(10.0, dataSet.sumEntries() * 10.0));
    RooRealVar nbkg("nbkg", "background yield", std::max(1.0, dataSet.sumEntries() * 0.5),
                    0.0, std::max(10.0, dataSet.sumEntries() * 10.0));

    RooAddPdf model("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg));
    model.fitTo(dataSet, RooFit::Save(true), RooFit::PrintLevel(-1));

    double ratioVal = 0.0;
    double ratioErr = 0.0;
    const bool hasRatio = (sigmaMC.getVal() > 0.0 && sigma.getVal() > 0.0);
    if (hasRatio) {
        ratioVal = sigma.getVal() / sigmaMC.getVal();
        const double relData = sigma.getError() / sigma.getVal();
        const double relMc = sigmaMC.getError() / sigmaMC.getVal();
        ratioErr = ratioVal * std::sqrt(relData * relData + relMc * relMc);
    }

    const double muVal = mean.getVal();
    const double sigmaVal = sigma.getVal();
    double windowMin = std::max(massMin, muVal - 3.0 * sigmaVal);
    double windowMax = std::min(massMax, muVal + 3.0 * sigmaVal);
    if (windowMax <= windowMin) {
        windowMin = massMin;
        windowMax = massMax;
    }
    mass.setRange("sigWindow", windowMin, windowMax);

    std::unique_ptr<RooAbsReal> sigIntegral(signal.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
    std::unique_ptr<RooAbsReal> bkgIntegral(background.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));

    const double sigFrac3s = sigIntegral ? sigIntegral->getVal() : 0.0;
    const double bkgFrac3s = bkgIntegral ? bkgIntegral->getVal() : 0.0;

    const double signalCounts = nsig.getVal();
    const double signalCountsErr = nsig.getError();
    const double signalInt3s = signalCounts * sigFrac3s;
    const double signalIntErr3s = signalCountsErr * sigFrac3s;

    const double bkgInt3s = nbkg.getVal() * bkgFrac3s;
    const double bkgIntErr3s = nbkg.getError() * bkgFrac3s;

    double sOverB = 0.0;
    double sOverBErr = 0.0;
    const bool validSOverB = bkgInt3s > 0.0;
    if (validSOverB) {
        sOverB = signalInt3s / bkgInt3s;
        const double dSd = 1.0 / bkgInt3s;
        const double dBd = -signalInt3s / (bkgInt3s * bkgInt3s);
        sOverBErr = std::sqrt(std::pow(dSd * signalIntErr3s, 2) + std::pow(dBd * bkgIntErr3s, 2));
    }

    double significance3s = 0.0;
    double significanceErr3s = 0.0;
    const double sumSB = signalInt3s + bkgInt3s;
    if (sumSB > 0.0) {
        significance3s = signalInt3s / std::sqrt(sumSB);
        const double denomPow = std::pow(sumSB, 1.5);
        if (denomPow > 0.0) {
            const double dFdS = (signalInt3s + 2.0 * bkgInt3s) / (2.0 * denomPow);
            const double dFdB = -signalInt3s / (2.0 * denomPow);
            significanceErr3s = std::sqrt(std::pow(dFdS * signalIntErr3s, 2) + std::pow(dFdB * bkgIntErr3s, 2));
        }
    }

    std::unique_ptr<RooPlot> dataFrame(mass.frame(fCfg.massBinsData));
    dataSet.plotOn(dataFrame.get());
    model.plotOn(dataFrame.get(),
                 RooFit::LineColor(kBlue + 1),
                 RooFit::LineWidth(2),
                 RooFit::Name("totalPDF"));
    model.plotOn(dataFrame.get(),
                 RooFit::Components(background.GetName()),
                 RooFit::LineStyle(kDashed),
                 RooFit::LineColor(kGreen + 2),
                 RooFit::Name("background"));
    model.plotOn(dataFrame.get(),
                 RooFit::Components(signal.GetName()),
                 RooFit::LineStyle(kDashDotted),
                 RooFit::LineColor(kOrange + 7),
                 RooFit::Name("signal"));

    constexpr int nDataFloatParams = 7;
    const int ndfData = std::max(1, static_cast<int>(fCfg.massBinsData) - nDataFloatParams);
    const double chi2Data = dataFrame->chiSquare("totalPDF", nullptr, nDataFloatParams);
    auto dataInfo = std::make_unique<TPaveText>(0.55, 0.50, 0.88, 0.86, "NDC");
    dataInfo->SetBorderSize(0);
    dataInfo->SetFillStyle(0);
    dataInfo->SetTextAlign(11);
    dataInfo->SetTextFont(42);
    dataInfo->AddText(Form("#mu = %.2f #pm %.2f MeV/#it{c}^{2}",
                           mean.getVal() * 1e3,
                           mean.getError() * 1e3));
    dataInfo->AddText(Form("#sigma_{data} = %.2f #pm %.2f MeV/#it{c}^{2}",
                           sigma.getVal() * 1e3,
                           sigma.getError() * 1e3));
    dataInfo->AddText(Form("#sigma_{MC} = %.2f #pm %.2f MeV/#it{c}^{2}",
                           sigmaMC.getVal() * 1e3,
                           sigmaMC.getError() * 1e3));
    if (hasRatio) {
        dataInfo->AddText(Form("#sigma_{data}/#sigma_{MC} = %.3f #pm %.3f",
                               ratioVal,
                               ratioErr));
    }
    dataInfo->AddText(Form("Signal (3 #sigma): %.0f #pm %.0f",
                           signalInt3s,
                           signalIntErr3s));
    if (validSOverB) {
        dataInfo->AddText(Form("S/B (3 #sigma): %.1f #pm %.1f",
                               sOverB,
                               sOverBErr));
    } else {
        dataInfo->AddText("S/B (3 #sigma): n/a");
    }
    if (sumSB > 0.0) {
        dataInfo->AddText(Form("S/#sqrt{S+B} (3 #sigma): %.1f #pm %.1f",
                               significance3s,
                               significanceErr3s));
    } else {
        dataInfo->AddText("S/#sqrt{S+B} (3 #sigma): n/a");
    }
    dataInfo->AddText(Form("Entries = %d", dataSet.numEntries()));
    dataInfo->AddText(Form("#chi^{2}/NDF = %.3f (NDF: %d)", chi2Data, ndfData));
    dataFrame->addObject(dataInfo.release());

    out.sigmaData = sigma.getVal();
    out.sigmaDataErr = sigma.getError();
    out.sigmaMc = sigmaMC.getVal();
    out.sigmaMcErr = sigmaMC.getError();
    out.ratio = hasRatio ? ratioVal : 0.0;
    out.ratioErr = hasRatio ? ratioErr : 0.0;
    out.signalYield3Sigma = signalInt3s;
    out.signalYield3SigmaErr = signalIntErr3s;
    out.backgroundYield3Sigma = bkgInt3s;
    out.backgroundYield3SigmaErr = bkgIntErr3s;
    out.significance3Sigma = significance3s;
    out.significance3SigmaErr = significanceErr3s;
    out.entriesData = static_cast<size_t>(dataSet.numEntries());
    out.entriesMc = static_cast<size_t>(mcSet.numEntries());
    out.dataFrame = std::move(dataFrame);
    out.mcFrame = std::move(mcFrame);
    return out;
}

const BinSample &SigmaMcDataMatcher::GetSample(size_t ptIndex, size_t ctIndex) const {
    if (ptIndex >= fSampleIndexGrid.size()) {
        throw std::runtime_error("SigmaMcDataMatcher: pt index out of range");
    }
    if (ctIndex >= fSampleIndexGrid[ptIndex].size()) {
        throw std::runtime_error("SigmaMcDataMatcher: ct index out of range");
    }
    size_t idx = fSampleIndexGrid[ptIndex][ctIndex];
    if (idx == kInvalidIndex || idx >= fSamples.size()) {
        throw std::runtime_error("SigmaMcDataMatcher: missing sample for requested bin");
    }
    return fSamples[idx];
}

void SigmaMcDataMatcher::RunCombinedFit() {
    std::vector<double> allData;
    std::vector<double> allMc;
    size_t totalBefore = 0;
    size_t totalAfter = 0;
    size_t totalMcEntries = 0;

    for (const auto &sample : fSamples) {
        allData.insert(allData.end(), sample.dataMasses.begin(), sample.dataMasses.end());
        allMc.insert(allMc.end(), sample.mcMasses.begin(), sample.mcMasses.end());
        totalBefore += std::max(0, sample.entriesBefore);
        totalAfter += std::max(0, sample.entriesAfter);
        totalMcEntries += sample.mcMasses.size();
    }

    auto fit = FitMassSpectra(allData, allMc, "combined");

    const auto txtPath = fCombinedDir / "sigma_summary.txt";
    std::ofstream ofs(txtPath);
    ofs << "# Combined sigma summary\n";
    ofs << "# entries_total_before entries_total_after entries_mc sigma_data sigma_data_err sigma_mc sigma_mc_err ratio ratio_err signal_3sigma signal_3sigma_err background_3sigma background_3sigma_err significance_3sigma significance_3sigma_err\n";
    ofs << totalBefore << " " << totalAfter << " " << totalMcEntries << " "
        << fit.sigmaData << " " << fit.sigmaDataErr << " "
        << fit.sigmaMc << " " << fit.sigmaMcErr << " "
        << fit.ratio << " " << fit.ratioErr << " "
        << fit.signalYield3Sigma << " " << fit.signalYield3SigmaErr << " "
        << fit.backgroundYield3Sigma << " " << fit.backgroundYield3SigmaErr << " "
        << fit.significance3Sigma << " " << fit.significance3SigmaErr << "\n";

    auto makeCanvas = [](const std::string &name, RooPlot *frame, const char *title) {
        auto canvas = std::make_unique<TCanvas>(name.c_str(), title, 900, 700);
        canvas->cd();
        frame->GetXaxis()->SetTitle("m(^{3}H_{#Lambda}) (GeV/c^{2})");
        frame->GetYaxis()->SetTitle("Candidates");
        frame->Draw();
        return canvas;
    };

    auto cData = makeCanvas("c_combined_data", fit.dataFrame.get(), "Combined data mass fit");
    auto cMc = makeCanvas("c_combined_mc", fit.mcFrame.get(), "Combined MC mass fit");

    const std::string dataPdf = (fCombinedDir / "combined_data_fit.pdf").string();
    const std::string mcPdf = (fCombinedDir / "combined_mc_fit.pdf").string();
    cData->SaveAs(dataPdf.c_str());
    cMc->SaveAs(mcPdf.c_str());

    const auto rootPath = (fCombinedDir / "combined_mass_fit.root").string();
    TFile outFile(rootPath.c_str(), "RECREATE");
    if (!outFile.IsZombie()) {
        cData->Write("combined_data_fit");
        cMc->Write("combined_mc_fit");
        outFile.Write();
    }
}

void SigmaMcDataMatcher::RunPerBinFits() {
    const size_t nPt = fCfg.ptBins.size() - 1;
    std::vector<std::unique_ptr<TH1D>> ratioHists;
    ratioHists.reserve(nPt);

    for (size_t ipt = 0; ipt < nPt; ++ipt) {
        const auto &ctEdges = fCfg.ctBins.at(ipt);
        auto hist = std::make_unique<TH1D>(
            Form("h_sigma_ratio_pt_%zu", ipt),
            Form("#sigma_{data}/#sigma_{MC} vs ct (%g < p_{T} < %g)",
                 fCfg.ptBins[ipt], fCfg.ptBins[ipt + 1]),
            static_cast<int>(ctEdges.size()) - 1,
            ctEdges.data());
        hist->GetXaxis()->SetTitle("ct (cm)");
        hist->GetYaxis()->SetTitle("#sigma_{data}/#sigma_{MC}");
        hist->Sumw2();
        hist->SetDirectory(nullptr);
        ratioHists.emplace_back(std::move(hist));
    }

    struct BinSigmaSummary {
        BinKey key;
        double sigmaData;
        double sigmaDataErr;
        double sigmaMc;
        double sigmaMcErr;
        double ratio;
        double ratioErr;
        double signal3Sigma;
        double signal3SigmaErr;
        double background3Sigma;
        double background3SigmaErr;
        double significance3Sigma;
        double significance3SigmaErr;
        size_t entriesData;
        size_t entriesMc;
    };

    std::vector<BinSigmaSummary> binSummaries;

    const auto rootPath = (fPerBinDir / "per_bin_fits.root").string();
    TFile rootFile(rootPath.c_str(), "RECREATE");
    TDirectory *massDir = rootFile.mkdir("mass_fits");
    TDirectory *ratioDir = rootFile.mkdir("ratio_vs_ct");

    auto ensurePtDir = [&](TDirectory *parent, size_t ipt) -> TDirectory * {
        const std::string name = Form("pt_%s_%s",
                                      FormatEdge(fCfg.ptBins[ipt]).c_str(),
                                      FormatEdge(fCfg.ptBins[ipt + 1]).c_str());
        TDirectory *dir = parent->GetDirectory(name.c_str());
        if (!dir) {
            parent->cd();
            dir = parent->mkdir(name.c_str());
        }
        return dir;
    };

    auto appendRatioSummary = [&](size_t ipt, double value, double error) {
        const auto txtPath = fPerBinDir / "pt_ratio_summary.txt";
        std::ofstream ratioTxt;
        if (ipt == 0) {
            ratioTxt.open(txtPath, std::ios::out);
            ratioTxt << "# pt_min pt_max constant_ratio constant_ratio_err\n";
        } else {
            ratioTxt.open(txtPath, std::ios::app);
        }
        ratioTxt << fCfg.ptBins[ipt] << " " << fCfg.ptBins[ipt + 1] << " "
                 << value << " " << error << "\n";
    };

    for (size_t ipt = 0; ipt < nPt; ++ipt) {
        const size_t nCt = fCfg.ctBins.at(ipt).size() - 1;
        for (size_t ict = 0; ict < nCt; ++ict) {
            const auto &sample = GetSample(ipt, ict);
            const std::string tag = sample.key.ToString();
            auto fit = FitMassSpectra(sample.dataMasses, sample.mcMasses, tag);

            ratioHists[ipt]->SetBinContent(static_cast<int>(ict) + 1, fit.ratio);
            ratioHists[ipt]->SetBinError(static_cast<int>(ict) + 1, fit.ratioErr);

            BinSigmaSummary summary{
                sample.key,
                fit.sigmaData,
                fit.sigmaDataErr,
                fit.sigmaMc,
                fit.sigmaMcErr,
                fit.ratio,
                fit.ratioErr,
                fit.signalYield3Sigma,
                fit.signalYield3SigmaErr,
                fit.backgroundYield3Sigma,
                fit.backgroundYield3SigmaErr,
                fit.significance3Sigma,
                fit.significance3SigmaErr,
                fit.entriesData,
                fit.entriesMc};
            binSummaries.push_back(summary);

            auto makeCanvas = [&](const std::string &name, RooPlot *frame, const char *title) {
                auto canvas = std::make_unique<TCanvas>(name.c_str(), title, 900, 700);
                canvas->cd();
                frame->GetXaxis()->SetTitle("m(^{3}H_{#Lambda}) (GeV/c^{2})");
                frame->Draw();
                return canvas;
            };

            auto cData = makeCanvas(Form("c_data_%s", tag.c_str()), fit.dataFrame.get(), "Data mass fit");
            auto cMc = makeCanvas(Form("c_mc_%s", tag.c_str()), fit.mcFrame.get(), "MC mass fit");

            if (massDir) {
                TDirectory *ptDir = ensurePtDir(massDir, ipt);
                ptDir->cd();
                cData->Write(Form("data_fit_ct_%g_%g", sample.key.ctMin, sample.key.ctMax));
                cMc->Write(Form("mc_fit_ct_%g_%g", sample.key.ctMin, sample.key.ctMax));
            }

            const auto outPrefix = fPerBinDir / Form("%s", tag.c_str());
            cData->SaveAs((outPrefix.string() + "_data.pdf").c_str());
            cMc->SaveAs((outPrefix.string() + "_mc.pdf").c_str());
        }

        if (ratioDir) {
            ratioDir->cd();
            ratioHists[ipt]->Write();
        }

        TF1 fConst(Form("f_sigma_ratio_pt_%zu", ipt), "[0]",
                   ratioHists[ipt]->GetXaxis()->GetXmin(),
                   ratioHists[ipt]->GetXaxis()->GetXmax());
        fConst.SetParameter(0, 1.0);
        ratioHists[ipt]->Fit(&fConst, "QS");

        auto cRatio = std::make_unique<TCanvas>(
            Form("c_ratio_pt_%zu", ipt),
            Form("#sigma_{data}/#sigma_{MC} (%g < p_{T} < %g)",
                 fCfg.ptBins[ipt], fCfg.ptBins[ipt + 1]),
            900, 700);
        cRatio->cd();
        ratioHists[ipt]->SetStats(0);
        ratioHists[ipt]->Draw("E1");
        fConst.SetLineColor(kRed + 1);
        fConst.SetLineWidth(2);
        fConst.Draw("SAME");
        cRatio->cd();
        auto ratioLabel = new TPaveText(0.55, 0.78, 0.88, 0.88, "NDC");
        ratioLabel->SetBorderSize(0);
        ratioLabel->SetFillStyle(0);
        ratioLabel->SetTextFont(42);
        ratioLabel->SetTextAlign(12);
        ratioLabel->AddText(Form("Const = %.3f #pm %.3f",
                     fConst.GetParameter(0),
                     fConst.GetParError(0)));
        ratioLabel->Draw("same");

        if (ratioDir) {
            ratioDir->cd();
            cRatio->Write();
        }
        const auto pdfPath = fPerBinDir / Form("pt_%s_%s_ratio.pdf",
                                               FormatEdge(fCfg.ptBins[ipt]).c_str(),
                                               FormatEdge(fCfg.ptBins[ipt + 1]).c_str());
        cRatio->SaveAs(pdfPath.string().c_str());

        appendRatioSummary(ipt, fConst.GetParameter(0), fConst.GetParError(0));
    }

    const auto summaryPath = fPerBinDir / "per_bin_sigma_summary.txt";
    std::ofstream summary(summaryPath);
    summary << "# pt_min pt_max ct_min ct_max entries_data entries_mc sigma_data sigma_data_err sigma_mc sigma_mc_err ratio ratio_err signal_3sigma signal_3sigma_err background_3sigma background_3sigma_err significance_3sigma significance_3sigma_err\n";
    for (const auto &entry : binSummaries) {
        summary << entry.key.ptMin << " " << entry.key.ptMax << " "
                << entry.key.ctMin << " " << entry.key.ctMax << " "
                << entry.entriesData << " " << entry.entriesMc << " "
                << entry.sigmaData << " " << entry.sigmaDataErr << " "
                << entry.sigmaMc << " " << entry.sigmaMcErr << " "
                << entry.ratio << " " << entry.ratioErr << " "
                << entry.signal3Sigma << " " << entry.signal3SigmaErr << " "
                << entry.background3Sigma << " " << entry.background3SigmaErr << " "
                << entry.significance3Sigma << " " << entry.significance3SigmaErr << "\n";
    }

    if (!rootFile.IsZombie()) {
        rootFile.Write();
    }
}

} // namespace

void SigmaMcDataMatching(const char *configPath = "../configs/ct_extraction.json",
                         const char *outputDirOverride = "") {
    try {
        SigmaMcDataMatcher matcher(configPath ? configPath : "", outputDirOverride ? outputDirOverride : "");
        matcher.Run();
    } catch (const std::exception &ex) {
        std::cerr << "[SigmaMcDataMatching] Error: " << ex.what() << std::endl;
        throw;
    }
}