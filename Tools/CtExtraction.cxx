#include "CtExtraction.h"

#include "AcceptanceHelper.h"
#include "GeneralHelper.hpp"

#include <ROOT/RDataFrame.hxx>

#include <TCanvas.h>
#include <TChain.h>
#include <TDirectory.h>
#include <TError.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TMath.h>
#include <TPaveText.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "GeneralHelper.hpp"

using json = nlohmann::json;

namespace {
constexpr double kSpeedOfLightCmPerPs = 0.0299792458; // c * 1 ps

std::unique_ptr<TF1> MakeExpoFitFunction(const std::string &name, double xmin, double xmax) {
    auto fn = std::make_unique<TF1>(name.c_str(), "[0]*exp(-x/[1])", xmin, xmax);
    fn->SetParName(0, "N_{0}");
    fn->SetParName(1, "ct");
    fn->SetParameter(0, 1.0);
    fn->SetParameter(1, 8.0);
    fn->SetLineColor(kOrange + 1);
    fn->SetLineWidth(2);
    return fn;
}
} // namespace

CtExtraction::CtExtraction(const std::string &configPath) {
    LoadConfig(configPath);
    ValidateConfig();
    LoadWorkingPoints();

    fInputMcFile = TFile::Open(fCfg.mcFile.c_str(), "READ");
    if (!fInputMcFile || fInputMcFile->IsZombie()) {
        throw std::runtime_error("Failed to open MC file: " + fCfg.mcFile);
    }
}

CtExtraction::~CtExtraction() {
    if (fInputMcFile) {
        fInputMcFile->Close();
        delete fInputMcFile;
        fInputMcFile = nullptr;
    }
    if (fOutputFile) {
        fOutputFile->Write();
        fOutputFile->Close();
    }
}

void CtExtraction::Run() {
    PrepareOutputFile();
    BuildAcceptance();
    cout << "CtExtraction: Starting main extraction loop..." << endl;

    if (!fOutputFile) {
        throw std::runtime_error("Output ROOT file is not ready");
    }

    std::string stdDirName = "std";
    if (!fCfg.trialSuffix.empty()) {
        stdDirName += "_" + fCfg.trialSuffix;
    }
    if (auto existingStdDir = fOutputFile->GetDirectory(stdDirName.c_str()); existingStdDir) {
        fOutputFile->Delete(Form("%s;*", stdDirName.c_str()));
    }
    TDirectory *stdDir = fOutputFile->mkdir(stdDirName.c_str());

    if (!stdDir) {
        throw std::runtime_error("Failed to create directory " + stdDirName);
    }

    auto hTauVsPt = std::make_unique<TH1D>("tau_per_ptbin",
                                            ";#it{p}_{T} (GeV/c);#tau (ps)",
                                            static_cast<int>(fCfg.ptBins.size()) - 1,
                                            fCfg.ptBins.data());
    auto hTauErrVsPt = std::make_unique<TH1D>("tau_err_per_ptbin",
                                              ";#it{p}_{T} (GeV/c);#sigma_{#tau} (ps)",
                                              static_cast<int>(fCfg.ptBins.size()) - 1,
                                              fCfg.ptBins.data());
    hTauVsPt->SetDirectory(nullptr);
    hTauErrVsPt->SetDirectory(nullptr);

    for (size_t ipt = 0; ipt + 1 < fCfg.ptBins.size(); ++ipt) {
        double ptMin = fCfg.ptBins[ipt];
        double ptMax = fCfg.ptBins[ipt + 1];
        const auto &ctEdges = fCfg.ctBins.at(ipt);
        if (ctEdges.size() < 2) {
            throw std::runtime_error("ct_edges empty for pt bin index " + std::to_string(ipt));
        }

        std::string ptDirName = std::string("pt_") + FormatEdge(ptMin) + "_" + FormatEdge(ptMax);
        TDirectory *ptDir = stdDir->mkdir(ptDirName.c_str());
        if (!ptDir) {
            throw std::runtime_error("Failed to create directory " + ptDirName);
        }

        auto makeHist = [&](const std::string &base, const std::string &title) {
            auto h = std::make_unique<TH1D>((base + ptDirName).c_str(), title.c_str(), static_cast<int>(ctEdges.size()) - 1, ctEdges.data());
            h->Sumw2();
            h->SetDirectory(nullptr);
            h->GetXaxis()->SetTitle("ct (cm)");
            return h;
        };

        auto hRaw = makeHist("h_raw_counts_", Form("Raw counts (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hAcc = makeHist("h_acc_eff_", Form("Acc. x Eff. (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hCorr = makeHist("h_ct_spectrum_", Form("Corrected ct spectrum (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hBdtEff = makeHist("h_bdt_eff_", Form("BDT efficiency (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hEffAll = makeHist("h_eff_all_", Form("Total efficiency (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hRawScaled = makeHist("h_raw_counts_scaled_", Form("Scaled Raw counts (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hSigmaData = makeHist("h_sigma_data_", Form("Sigma Data (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hSigmaMc = makeHist("h_sigma_mc_", Form("Sigma MC (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hSigmaMcData = makeHist("h_sigma_mc_data_", Form("Sigma MC/Data (%g < #it{p}_{T} < %g)", ptMin, ptMax));
        auto hFitChi2Data = makeHist("h_fit_chi2_data_", Form("Fit Chi2 Data (%g < #it{p}_{T} < %g)", ptMin, ptMax));

        for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
            auto result = ProcessOneBin(ipt, ict);

            const int binIdx = static_cast<int>(ict) + 1;
            hRaw->SetBinContent(binIdx, result.rawYield);
            hRaw->SetBinError(binIdx, result.rawYieldErr);
            hAcc->SetBinContent(binIdx, result.acceptance);
            hAcc->SetBinError(binIdx, result.acceptanceErr);
            hCorr->SetBinContent(binIdx, result.correctedYield);
            hCorr->SetBinError(binIdx, result.correctedYieldErr);
            hBdtEff->SetBinContent(binIdx, result.bdtEfficiency);
            hBdtEff->SetBinError(binIdx, 0.0);
            hEffAll->SetBinContent(binIdx, result.acceptance * result.bdtEfficiency);
            hEffAll->SetBinError(binIdx, result.acceptanceErr * result.bdtEfficiency);
            hRawScaled->SetBinContent(binIdx, result.rawYield / (result.key.ctMax - result.key.ctMin));
            hRawScaled->SetBinError(binIdx, result.rawYieldErr / (result.key.ctMax - result.key.ctMin));
            hSigmaData->SetBinContent(binIdx, result.fittedSigma);
            hSigmaData->SetBinError(binIdx, result.fittedSigmaErr);
            hSigmaMc->SetBinContent(binIdx, result.fittedSigmaMC);
            hSigmaMc->SetBinError(binIdx, result.fittedSigmaMCErr);
            if (result.fittedSigmaMC > 0.0) {
                hSigmaMcData->SetBinContent(binIdx, result.fittedSigmaMC / result.fittedSigma);
                // Propagate error
                double relErr = std::sqrt(
                    std::pow(result.fittedSigmaMCErr / result.fittedSigmaMC, 2) +
                    std::pow(result.fittedSigmaErr / result.fittedSigma, 2));
                hSigmaMcData->SetBinError(binIdx, relErr * (result.fittedSigmaMC / result.fittedSigma));
            } else {
                hSigmaMcData->SetBinContent(binIdx, 0.0);
                hSigmaMcData->SetBinError(binIdx, 0.0);
            }
            if (result.fittedChi2 >= 0.0 && !std::isnan(result.fittedChi2)) {
                hFitChi2Data->SetBinContent(binIdx, result.fittedChi2);
            } else {
                hFitChi2Data->SetBinContent(binIdx, 0.0);
            }
            hFitChi2Data->SetBinError(binIdx, 0.0);


            
            ptDir->cd();
            if (result.mcMassFrame) {
                result.mcMassFrame->SetName(Form("mc_massfit_pt_%s", result.key.ToString().c_str()));
                result.mcMassFrame->SetTitle(Form("MC h3l invariant mass fit (pt_%s)", result.key.ToString().c_str()));
                result.mcMassFrame->Write();
            }
            if (result.dataMassFrame) {
                result.dataMassFrame->SetName(Form("data_massfit_pt_%s", result.key.ToString().c_str()));
                result.dataMassFrame->SetTitle(Form("Data h3l invariant mass fit (pt_%s)", result.key.ToString().c_str()));
                result.dataMassFrame->Write();
            }
        }

        ptDir->cd();
        hRaw->Write();
        hAcc->Write();
        hCorr->Write();
        hBdtEff->Write();
        hEffAll->Write();
        hRawScaled->Write();
        hSigmaData->Write();
        hSigmaMc->Write();
        hSigmaMcData->Write();
        hFitChi2Data->Write();

        auto fitFunc = MakeExpoFitFunction("f_exp_" + ptDirName,
                                           hCorr->GetXaxis()->GetXmin(),
                                           hCorr->GetXaxis()->GetXmax());
        fitFunc->SetParameter(0, std::max(1.0, hCorr->GetMaximum()));
        fitFunc->SetParameter(1, 8.0);
        hCorr->Fit(fitFunc.get(), "QIS");
        double tauCm = fitFunc->GetParameter(1);
        double tauCmErr = fitFunc->GetParError(1);
        double tauPs = tauCm / kSpeedOfLightCmPerPs;
        double tauPsErr = tauCmErr / kSpeedOfLightCmPerPs;

        hTauVsPt->SetBinContent(static_cast<int>(ipt) + 1, tauPs);
        hTauVsPt->SetBinError(static_cast<int>(ipt) + 1, tauPsErr);
        hTauErrVsPt->SetBinContent(static_cast<int>(ipt) + 1, tauPsErr);
        hTauErrVsPt->SetBinError(static_cast<int>(ipt) + 1, 0.0);

        auto canvas = std::make_unique<TCanvas>(Form("c_ct_fit_%s", ptDirName.c_str()),
                        Form("CT fit %s", ptDirName.c_str()),
                        900, 650);
        canvas->SetLeftMargin(0.14);
        canvas->SetBottomMargin(0.12);
        canvas->SetRightMargin(0.05);
        canvas->SetTopMargin(0.05);
        canvas->SetTicks(1, 1);
        canvas->SetGridy(true);
        canvas->SetLogy();
        hCorr->SetStats(false);
        hCorr->SetMinimum(std::max(1e-3, hCorr->GetMinimum(1) * 0.5));
        hCorr->SetLineColor(kAzure + 2);
        hCorr->SetMarkerColor(kAzure + 2);
        hCorr->SetMarkerStyle(20);
        hCorr->SetMarkerSize(1.1);
        hCorr->GetXaxis()->SetTitle("#it{c}t (cm)");
        hCorr->GetXaxis()->SetTitleSize(0.05);
        hCorr->GetXaxis()->SetLabelSize(0.045);
        hCorr->GetYaxis()->SetTitle("Corrected counts");
        hCorr->GetYaxis()->SetTitleSize(0.05);
        hCorr->GetYaxis()->SetTitleOffset(1.25);
        hCorr->GetYaxis()->SetLabelSize(0.045);
        hCorr->Draw("E1");
        fitFunc->SetLineColor(kRed + 1);
        fitFunc->SetLineWidth(3);
        fitFunc->Draw("SAME");

        auto legend = std::make_unique<TLegend>(0.60, 0.70, 0.90, 0.90);
        legend->SetBorderSize(0);
        legend->SetFillStyle(0);
        legend->SetTextSize(0.045);
        legend->AddEntry(hCorr.get(), "Corrected spectrum", "lep");
        legend->AddEntry(fitFunc.get(), "Exp fit", "l");
        legend->Draw();

        auto pave = std::make_unique<TPaveText>(0.18, 0.70, 0.55, 0.90, "NDC");
        pave->SetFillStyle(0);
        pave->SetBorderSize(0);
        pave->SetTextAlign(12);
        pave->SetTextSize(0.045);
        const double chi2 = fitFunc->GetChisquare();
        const int ndf = fitFunc->GetNDF();
        const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;
        pave->AddText(Form("#tau = %.2f #pm %.2f ps", tauPs, tauPsErr));
        pave->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
        pave->AddText(Form("Fit prob. = %.3f", fitProb));

        pave->Draw();

        canvas->Write();
    }

    stdDir->cd();
    hTauVsPt->Write();
    hTauErrVsPt->Write();
}

void CtExtraction::SetBDTScoreOverride(double ptMin, double ptMax,
                                       double ctMin, double ctMax,
                                       double score) {
    BinKey key{ptMin, ptMax, ctMin, ctMax};
    fUserOverrides[key] = score;
}

void CtExtraction::ClearBDTOverrides() {
    fUserOverrides.clear();
}

std::string CtExtraction::BinKey::ToString() const {
    return FormatEdge(ptMin) + "_" + FormatEdge(ptMax) + "_ct_" + FormatEdge(ctMin) + "_" + FormatEdge(ctMax);
}

bool CtExtraction::BinKey::operator<(const BinKey &other) const {
    return std::tie(ptMin, ptMax, ctMin, ctMax) < std::tie(other.ptMin, other.ptMax, other.ctMin, other.ctMax);
}

void CtExtraction::LoadConfig(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    fCfgJson = json::parse(ifs, nullptr, true, true);

    auto get_string = [&](const char *key, const std::string &fallback = std::string()) {
        return fCfgJson.value(key, fallback);
    };

    fCfg.dataSnapshotDir = get_string("data_snapshot_dir");
    fCfg.snapshotTreeName = get_string("snapshot_tree_name", "O2hypcands");
    fCfg.mcFile = get_string("mc_file");
    fCfg.mcTreeName = get_string("mc_tree_name", "O2mchypcands");
    fCfg.mcSnapshotDir = get_string("mc_snapshot_dir", fCfg.dataSnapshotDir);
    fCfg.mcSnapshotTreeName = get_string("mc_snapshot_tree_name", "O2mchypcands");
    fCfg.mcSnapshotPattern = get_string("mc_snapshot_pattern", "mc_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root");
    fCfg.workingPointFile = get_string("working_point_file");
    fCfg.outputDir = get_string("output_dir", "results/ct_extraction");
    fCfg.outputFile = get_string("output_file", "ct_analysis");
    fCfg.trialSuffix = get_string("trial_suffix");
    fCfg.isMatter = get_string("is_matter", "both");
    fCfg.massColumn = get_string("mass_column", "fMassH3L");
    fCfg.bdtScoreColumn = get_string("bdt_score_column", "model_output");
    fCfg.snapshotPattern = get_string("snapshot_pattern", "data_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root");
    fCfg.mcReweightFile = get_string("mc_reweight_file", "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root");
    fCfg.mcReweightFunc = get_string("mc_reweight_func", "BlastWave_H3L_10_30");

    fCfg.ptBins = fCfgJson.value("pt_bins", std::vector<double>{});
    fCfg.ctBins = fCfgJson.value("ct_bins", std::vector<std::vector<double>>{});
    fCfg.massRange = fCfgJson.value("mass_range", std::vector<double>{2.95, 3.05});
    fCfg.massBins = fCfgJson.value("mass_nbins_data", 50);
    fCfg.mcMassBins = fCfgJson.value("mass_nbins_mc", 80);

    fCfg.minEntriesForFit = fCfgJson.value("min_entries_for_fit", 60.0);
    fCfg.minScoreShift = fCfgJson.value("bdt_score_shift", 0.0);
    fCfg.runPeriodLabel = get_string("run_period_label", "Run 3");
    fCfg.collidingSystem = get_string("colliding_system", "Pb-Pb");
    fCfg.sqrtsLabel = get_string("sqrtsnn_label", "#sqrt{s_{NN}}");
    fCfg.dataSetLabel = get_string("data_set_label", "LHC23_PbPb_pass5");
    fCfg.collisionEnergyTeV = fCfgJson.value("collision_energy_tev", 5.36);
    fCfg.alicePerformance = fCfgJson.value("alice_performance", false);
    fCfg.sigmaRangeMcToData = fCfgJson.value("sigma_mc_to_data_range", std::vector<std::vector<double>>{ {0.9, 1.5}, {0.9, 1.5} , {0.9, 1.5} , {0.9, 1.5} });

    if (fCfgJson.contains("bdt_overrides")) {
        for (const auto &item : fCfgJson["bdt_overrides"]) {
            if (!item.contains("pt") || !item.contains("ct") || !item.contains("score")) {
                continue;
            }
            auto pt = item["pt"].get<std::vector<double>>();
            auto ct = item["ct"].get<std::vector<double>>();
            if (pt.size() != 2 || ct.size() != 2) {
                continue;
            }
            BinKey key{pt[0], pt[1], ct[0], ct[1]};
            fUserOverrides[key] = item["score"].get<double>();
        }
    }
}

void CtExtraction::ValidateConfig() const {
    auto ensure_exists = [](const std::string &path, const std::string &tag) {
        if (path.empty()) {
            throw std::runtime_error(tag + " path is empty");
        }
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(tag + " path does not exist: " + path);
        }
    };

    ensure_exists(fCfg.dataSnapshotDir, "data_snapshot_dir");
    ensure_exists(fCfg.mcFile, "mc_file");
    ensure_exists(fCfg.workingPointFile, "working_point_file");
    ensure_exists(fCfg.mcSnapshotDir, "mc_snapshot_dir");
    ensure_exists(fCfg.mcReweightFile, "mc_reweight_file");

    if (fCfg.ptBins.size() < 2) {
        throw std::runtime_error("pt_bins must contain at least two edges");
    }
    if (fCfg.ctBins.size() != fCfg.ptBins.size() - 1) {
        throw std::runtime_error("ct_bins must have (nPt-1) entries");
    }
    for (const auto &edges : fCfg.ctBins) {
        if (edges.size() < 2) {
            throw std::runtime_error("Each ct bin list must contain at least two edges");
        }
    }
    if (fCfg.massRange.size() < 2 || fCfg.massRange[0] >= fCfg.massRange[1]) {
        throw std::runtime_error("Invalid mass_range specification");
    }
    if (fCfg.massColumn.empty()) {
        throw std::runtime_error("mass_column cannot be empty");
    }
    if (fCfg.bdtScoreColumn.empty()) {
        throw std::runtime_error("bdt_score_column cannot be empty");
    }
    if (fCfg.snapshotPattern.empty()) {
        throw std::runtime_error("snapshot_pattern cannot be empty");
    }
    if (fCfg.mcSnapshotPattern.empty()) {
        throw std::runtime_error("mc_snapshot_pattern cannot be empty");
    }
}

void CtExtraction::LoadWorkingPoints() {
    std::ifstream ifs(fCfg.workingPointFile);
    if (!ifs) {
        throw std::runtime_error("Failed to open working point file: " + fCfg.workingPointFile);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        double ptMin, ptMax, ctMin, ctMax, bestScore, bestEff, bestSig;
        ss >> ptMin >> ptMax >> ctMin >> ctMax >> bestScore >> bestEff >> bestSig;
        if (ss.fail()) {
            continue;
        }
        BinKey key{ptMin, ptMax, ctMin, ctMax};
        fWorkingPoints[key] = WorkingPoint{bestScore, bestEff, bestSig};
    }
}

void CtExtraction::PrepareOutputFile() {
    std::filesystem::path baseDir = std::filesystem::path(fCfg.outputDir);
    const std::string matterComponent = fCfg.isMatter.empty() ? "both" : fCfg.isMatter;
    baseDir /= matterComponent;
    GeneralHelper::EnsureDir(baseDir.string());

    const std::string outPath = (baseDir / (fCfg.outputFile + ".root")).string();
    fOutputFile.reset(TFile::Open(outPath.c_str(), "RECREATE"));
    if (!fOutputFile || fOutputFile->IsZombie()) {
        throw std::runtime_error("Failed to create output ROOT file: " + outPath);
    }
}

void CtExtraction::BuildAcceptance() {
    if (!fInputMcFile) {
        throw std::runtime_error("MC input file pointer is null");
    }

    TChain mcChain(fCfg.mcTreeName.c_str());
    GeneralHelper::fillChainFromAO2D(mcChain, fInputMcFile);
    if (mcChain.GetEntries() <= 0) {
        const std::string directPath = fCfg.mcFile + "/" + fCfg.mcTreeName;
        if (mcChain.Add(directPath.c_str()) == 0 || mcChain.GetEntries() <= 0) {
            throw std::runtime_error("Failed to locate tree '" + fCfg.mcTreeName +
                                     "' inside " + fCfg.mcFile);
        }
    }

    ROOT::RDataFrame mcFrame(mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(mcFrame, false, true, false);
    auto fileCloser = [](TFile *file) {
        if (file) {
            file->Close();
            delete file;
        }
    };
    std::unique_ptr<TFile, decltype(fileCloser)> reweightFile(TFile::Open(fCfg.mcReweightFile.c_str(), "READ"), fileCloser);
    if (!reweightFile || reweightFile->IsZombie()) {
        throw std::runtime_error("Failed to open MC reweight file: " + fCfg.mcReweightFile);
    }
    TF1 *reweightFunc = static_cast<TF1*>(reweightFile->Get(fCfg.mcReweightFunc.c_str()));
    if (!reweightFunc) {
        throw std::runtime_error("Failed to locate reweight histogram: " + fCfg.mcReweightFunc);
    }
    auto mcReweighted = GeneralHelper::ReWeightSpectrum(mcReady, reweightFunc, "fAbsGenPt");
    auto accResult = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcReweighted,
        fCfg.ptBins,
        std::vector<double>{},
        fCfg.ctBins,
        std::vector<double>{},
        std::vector<std::vector<double>>{}
    );

    const std::vector<TH1D*> *perPt = nullptr;
    if (fCfg.isMatter == "matter") {
        perPt = &accResult.acc_ct_per_pt_matter;
    } else if (fCfg.isMatter == "antimatter") {
        perPt = &accResult.acc_ct_per_pt_antimatter;
    } else {
        perPt = &accResult.acc_ct_per_pt;
    }

    fAcceptancePerPt.clear();
    fAcceptancePerPt.reserve(perPt->size());
    for (size_t i = 0; i < perPt->size(); ++i) {
        TH1D *src = perPt->at(i);
        if (!src) {
            fAcceptancePerPt.emplace_back();
            continue;
        }
        auto clone = std::unique_ptr<TH1D>(static_cast<TH1D*>(src->Clone(Form("acc_per_pt_%zu", i))));
        clone->SetDirectory(nullptr);
        fAcceptancePerPt.emplace_back(std::move(clone));
    }

    // clean up dynamically allocated histograms inside accResult to avoid leaks
    accResult.Clear();
}

CtExtraction::WorkingPoint CtExtraction::GetWorkingPoint(const BinKey &key) const {
    auto it = fWorkingPoints.find(key);
    if (it == fWorkingPoints.end()) {
        throw std::runtime_error("Missing working point entry for bin " + key.ToString());
    }
    return it->second;
}

double CtExtraction::ResolveBDTScore(const BinKey &key) const {
    auto overrideIt = fUserOverrides.find(key);
    if (overrideIt != fUserOverrides.end()) {
        return overrideIt->second;
    }

    const WorkingPoint wp = GetWorkingPoint(key);
    return wp.score + fCfg.minScoreShift;
}


CtExtraction::BinComputationResult CtExtraction::ProcessOneBin(size_t ptIndex, size_t ctIndex) {
    BinKey key{fCfg.ptBins[ptIndex], fCfg.ptBins[ptIndex + 1],
               fCfg.ctBins[ptIndex][ctIndex], fCfg.ctBins[ptIndex][ctIndex + 1]};

    WorkingPoint wp = GetWorkingPoint(key);
    double scoreToUse = ResolveBDTScore(key);

    int nBefore = 0;
    int nAfter = 0;
    auto masses = CollectMassValues(key, scoreToUse, nBefore, nAfter);
    auto mcMasses = CollectMCMasses(key);

    if (masses.size() < static_cast<size_t>(fCfg.minEntriesForFit)) {
        throw std::runtime_error("Not enough entries after BDT cut for bin " + key.ToString());
    }
    if (mcMasses.size() < static_cast<size_t>(fCfg.minEntriesForFit)) {
        throw std::runtime_error("Not enough MC entries for bin " + key.ToString());
    }

    auto result = FitSpectrum(key, wp, masses, mcMasses, nBefore, nAfter, fCfg.sigmaRangeMcToData[ptIndex]);
    result.bdtEfficiency = wp.efficiency;
    result.bdtScore = scoreToUse;

    double accErr = 0.0;
    result.acceptance = LookupAcceptance(key, accErr);
    result.acceptanceErr = accErr;

    if (result.acceptance > 0.0) {
        double binWidth = result.key.ctMax - result.key.ctMin;
        if (binWidth <= 0.0) {
            throw std::runtime_error("Invalid ct bin width for " + result.key.ToString());
        }
        if (result.bdtEfficiency <= 0.0) {
            throw std::runtime_error("Invalid BDT efficiency (<=0) for " + result.key.ToString());
        }
        result.correctedYield = result.rawYield / result.acceptance / result.bdtEfficiency / binWidth;
        result.correctedYieldErr = result.rawYieldErr / result.acceptance / result.bdtEfficiency / binWidth;
    }

    return result;
}

std::vector<double> CtExtraction::CollectMassValues(const BinKey &key,
                                                     double bdtScore,
                                                     int &entriesBefore,
                                                     int &entriesAfter) const {
    const std::string filePath = BuildPath(fCfg.dataSnapshotDir, fCfg.snapshotPattern, key);
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("Snapshot file not found: " + filePath);
    }

    ROOT::RDataFrame df(fCfg.snapshotTreeName, filePath);
    ROOT::RDF::RNode node = df;
    if (fCfg.isMatter == "matter") {
        node = node.Filter("fIsMatter > 0.5");
    } else if (fCfg.isMatter == "antimatter") {
        node = node.Filter("fIsMatter < 0.5");
    }

    auto filtered = node.Filter([bdtScore](float score) { return static_cast<double>(score) > bdtScore; }, {fCfg.bdtScoreColumn});

    auto countBefore = node.Count();
    auto countAfter = filtered.Count();

    struct SlotMassBuffer {
        std::vector<double> masses;
    };
    std::mutex slotMutex;
    std::vector<std::unique_ptr<SlotMassBuffer>> buffers;

    auto acquire = [&](unsigned slot) -> SlotMassBuffer & {
        std::lock_guard<std::mutex> guard(slotMutex);
        if (slot >= buffers.size()) {
            buffers.resize(slot + 1);
        }
        if (!buffers[slot]) {
            buffers[slot] = std::make_unique<SlotMassBuffer>();
        }
        return *buffers[slot];
    };

    filtered.ForeachSlot(
        [&](unsigned slot, double mass) {
            auto &buf = acquire(slot);
            buf.masses.push_back(mass);
        },
        std::vector<std::string>{fCfg.massColumn});

    entriesBefore = static_cast<int>(countBefore.GetValue());
    entriesAfter = static_cast<int>(countAfter.GetValue());

    std::vector<double> massValues;
    size_t total = 0;
    for (const auto &slotBuf : buffers) {
        if (!slotBuf) {
            continue;
        }
        total += slotBuf->masses.size();
    }
    massValues.reserve(total);
    for (const auto &slotBuf : buffers) {
        if (!slotBuf) {
            continue;
        }
        massValues.insert(massValues.end(), slotBuf->masses.begin(), slotBuf->masses.end());
    }
    return massValues;
}

std::vector<double> CtExtraction::CollectMCMasses(const BinKey &key) const {
    const std::string filePath = BuildPath(fCfg.mcSnapshotDir, fCfg.mcSnapshotPattern, key);
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("MC snapshot file not found: " + filePath);
    }

    ROOT::RDataFrame df(fCfg.mcSnapshotTreeName, filePath);
    ROOT::RDF::RNode node = df;
    if (fCfg.isMatter == "matter") {
        node = node.Filter("fIsMatter > 0.5");
    } else if (fCfg.isMatter == "antimatter") {
        node = node.Filter("fIsMatter < 0.5");
    }

    struct SlotMassBuffer {
        std::vector<double> masses;
    };
    std::mutex slotMutex;
    std::vector<std::unique_ptr<SlotMassBuffer>> buffers;

    auto acquire = [&](unsigned slot) -> SlotMassBuffer & {
        std::lock_guard<std::mutex> guard(slotMutex);
        if (slot >= buffers.size()) {
            buffers.resize(slot + 1);
        }
        if (!buffers[slot]) {
            buffers[slot] = std::make_unique<SlotMassBuffer>();
        }
        return *buffers[slot];
    };

    node.ForeachSlot(
        [&](unsigned slot, double mass) {
            auto &buf = acquire(slot);
            buf.masses.push_back(mass);
        },
        std::vector<std::string>{fCfg.massColumn});

    std::vector<double> massValues;
    size_t total = 0;
    for (const auto &slotBuf : buffers) {
        if (!slotBuf) {
            continue;
        }
        total += slotBuf->masses.size();
    }
    massValues.reserve(total);
    for (const auto &slotBuf : buffers) {
        if (!slotBuf) {
            continue;
        }
        massValues.insert(massValues.end(), slotBuf->masses.begin(), slotBuf->masses.end());
    }
    return massValues;
}

CtExtraction::BinComputationResult CtExtraction::FitSpectrum(const BinKey &key,
                                                             const WorkingPoint &wp,
                                                             const std::vector<double> &massValues,
                                                             const std::vector<double> &mcMassValues,
                                                             int entriesBefore,
                                                             int entriesAfter,
                                                             std::vector<double> sigmaRange) const {
    BinComputationResult res;
    res.key = key;
    res.entriesBeforeBDT = entriesBefore;
    res.entriesAfterBDT = entriesAfter;

    const double massMin = fCfg.massRange[0];
    const double massMax = fCfg.massRange[1];
    auto massVar = std::make_shared<RooRealVar>("mass", "invariant mass", massMin, massMax);
    RooRealVar &mass = *massVar;
    RooArgSet vars(mass);

    RooDataSet dataSet("data", "data", vars);
    for (double value : massValues) {
        if (value < massMin || value > massMax) {
            continue;
        }
        mass.setVal(value);
        dataSet.add(vars);
    }

    RooDataSet mcSet("mc", "mc", vars);
    for (double value : mcMassValues) {
        if (value < massMin || value > massMax) {
            continue;
        }
        mass.setVal(value);
        mcSet.add(vars);
    }

    if (dataSet.numEntries() < fCfg.minEntriesForFit) {
        throw std::runtime_error("Dataset too small to fit for bin " + key.ToString());
    }
    if (mcSet.numEntries() < fCfg.minEntriesForFit) {
        throw std::runtime_error("MC dataset too small to fit for bin " + key.ToString());
    }
    // Fit MC spectrum
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

    std::unique_ptr<RooPlot> mcFrame(mass.frame(fCfg.mcMassBins));
    mcSet.plotOn(mcFrame.get());
    signalMC.plotOn(mcFrame.get(), RooFit::LineColor(kRed + 1), RooFit::LineWidth(2), RooFit::Name("signalMC"), RooFit::LineStyle(kSolid));
    auto fitParamMc = std::make_unique<TPaveText>(0.6, 0.43, 0.9, 0.85, "NDC");
    fitParamMc->SetBorderSize(0);
    fitParamMc->SetFillStyle(0);
    fitParamMc->SetTextAlign(12);
    fitParamMc->AddText(Form("#mu = %.2f #pm %.2f MeV/#it{c}^{2}",
                             meanMC.getVal() * 1e3,
                             meanMC.getError() * 1e3));
    fitParamMc->AddText(Form("#sigma = %.2f #pm %.2f MeV/#it{c}^{2}",
                             sigmaMC.getVal() * 1e3,
                             sigmaMC.getError() * 1e3));
    fitParamMc->AddText(Form("#alpha_{L} = %.2f #pm %.2f",
                             alphaL.getVal(),
                             alphaL.getError()));
    fitParamMc->AddText(Form("#alpha_{R} = %.2f #pm %.2f",
                             alphaR.getVal(),
                             alphaR.getError()));
    fitParamMc->AddText(Form("n_{L} = %.2f #pm %.2f",
                             nL.getVal(),
                             nL.getError()));
    fitParamMc->AddText(Form("n_{R} = %.2f #pm %.2f",
                             nR.getVal(),
                             nR.getError()));
    constexpr int nMcFloatParams = 6;
    const int ndfMc = std::max(1, static_cast<int>(fCfg.mcMassBins) - nMcFloatParams);
    const double chi2OverNdfMc = mcFrame->chiSquare(signalMC.GetName(), nullptr, nMcFloatParams);
    fitParamMc->AddText(Form("#chi^{2} / NDF = %.3f (NDF: %d)", chi2OverNdfMc, ndfMc));
    mcFrame->addObject(fitParamMc.release());
    
    // Fit data spectrum
    RooRealVar mean("mean", "mean", meanMC.getVal(), massMin, massMax);
    RooRealVar sigma("sigma", "sigma", 1.05 * sigmaMC.getVal(), sigmaRange[0] * sigmaMC.getVal(), sigmaRange[1] * sigmaMC.getVal());
    RooCrystalBall signal("signal", "signal", mass, mean, sigma, alphaL, nL, alphaR, nR);

    RooRealVar c0("c0", "c0", 0.0, -1.5, 1.5);
    RooRealVar c1("c1", "c1", 0.0, -1.5, 1.5);
    RooRealVar c2("c2", "c2", 0.0, -1.5, 1.5);
    RooChebychev background("background", "background", mass, RooArgList(c0, c1, c2));

    const double entries = dataSet.sumEntries();
    RooRealVar nsig("nsig", "signal yield", std::max(1.0, entries * 0.1), 0.0, std::max(10.0, entries * 10.0));
    RooRealVar nbkg("nbkg", "background yield", std::max(1.0, entries * 0.9 + 1.0), 0.0, std::max(10.0, entries * 10.0));

    RooAddPdf model("model", "signal+background", RooArgList(signal, background), RooArgList(nsig, nbkg));
    model.fitTo(dataSet, RooFit::Save(true), RooFit::PrintLevel(-1));

    std::unique_ptr<RooPlot> dataFrame(mass.frame(fCfg.massBins));
    dataSet.plotOn(dataFrame.get());
    const Int_t kOrangeC = TColor::GetColor("#ff7f00");
    model.plotOn(dataFrame.get(), RooFit::LineColor(kAzure + 1), RooFit::LineWidth(3), RooFit::Name("totalPDF"), RooFit::LineStyle(kSolid));
    model.plotOn(dataFrame.get(), RooFit::Components(background.GetName()), RooFit::LineStyle(kDashed), RooFit::LineColor(kOrangeC), RooFit::LineWidth(3), RooFit::Name("background"));
    model.plotOn(dataFrame.get(),RooFit::Components(signal.GetName()), RooFit::LineColor(kGreen + 2), RooFit::LineWidth(3), RooFit::Name("signal"), RooFit::LineStyle(kDashDotted));
    const double scoreUsed = wp.score;
    const double effHere = wp.efficiency;
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
    const double signalIntVal3s = signalCounts * sigFrac3s;
    const double signalIntErr3s = signalCountsErr * sigFrac3s;

    const double bkgIntVal3s = nbkg.getVal() * bkgFrac3s;
    const double bkgIntErr3s = nbkg.getError() * bkgFrac3s;

    double sOverB = 0.0;
    double sOverBErr = 0.0;
    const bool validSOverB = bkgIntVal3s > 0.0;
    if (validSOverB) {
        sOverB = signalIntVal3s / bkgIntVal3s;
        const double dSd = 1.0 / bkgIntVal3s;
        const double dBd = -signalIntVal3s / (bkgIntVal3s * bkgIntVal3s);
        sOverBErr = std::sqrt(std::pow(dSd * signalIntErr3s, 2) + std::pow(dBd * bkgIntErr3s, 2));
    }

    double significance = 0.0;
    double significanceErr = 0.0;
    const double sumSB = signalIntVal3s + bkgIntVal3s;
    if (sumSB > 0.0) {
        significance = signalIntVal3s / std::sqrt(sumSB);
        const double denomPow = std::pow(sumSB, 1.5);
        if (denomPow > 0.0) {
            const double dFdS = (signalIntVal3s + 2.0 * bkgIntVal3s) / (2.0 * denomPow);
            const double dFdB = -signalIntVal3s / (2.0 * denomPow);
            significanceErr = std::sqrt(std::pow(dFdS * signalIntErr3s, 2) + std::pow(dFdB * bkgIntErr3s, 2));
        }
    }


    constexpr int nDataFloatParams = 7;
    const int ndfData = std::max(1, static_cast<int>(fCfg.massBins) - nDataFloatParams);
    const double chi2Data = dataFrame->chiSquare("totalPDF", nullptr, nDataFloatParams);

    auto pinfoVals = std::make_unique<TPaveText>(0.592, 0.50, 0.892, 0.85, "NDC");
    pinfoVals->SetBorderSize(0);
    pinfoVals->SetFillStyle(0);
    pinfoVals->SetTextAlign(11);
    pinfoVals->SetTextFont(42);
    pinfoVals->AddText(Form("BDT score > %.3f   Effi(#it{BDT}) = %.3f", scoreUsed, effHere));
    pinfoVals->AddText(Form("Signal Counts(3 #sigma): %.0f #pm %.0f", signalIntVal3s, signalIntErr3s));
    if (validSOverB) {
        pinfoVals->AddText(Form("S/B (3 #sigma): %.1f #pm %.1f", sOverB, sOverBErr));
    } else {
        pinfoVals->AddText("S/B (3 #sigma): n/a");
    }
    if (sumSB > 0.0) {
        pinfoVals->AddText(Form("S/#sqrt{S+B} (3 #sigma): %.1f #pm %.1f", significance, significanceErr));
    } else {
        pinfoVals->AddText("S/#sqrt{S+B} (3 #sigma): n/a");
    }
    pinfoVals->AddText(Form("#mu = %.2f #pm %.2f MeV/#it{c}^{2}", muVal * 1e3, mean.getError() * 1e3));
    pinfoVals->AddText(Form("#sigma Range Fixed to [%.2f, %.2f] #times #sigma_{MC}(%.2f MeV/#it{c}^{2})", sigmaRange[0], sigmaRange[1], sigmaMC.getVal() * 1e3));
    pinfoVals->AddText(Form("#sigma = %.2f #pm %.2f MeV/#it{c}^{2}", sigmaVal * 1e3, sigma.getError() * 1e3));
    pinfoVals->AddText(Form("#chi^{2} / NDF = %.3f (NDF: %d)", chi2Data, ndfData));
    dataFrame->addObject(pinfoVals.release());

    const std::string &runPeriod = fCfg.runPeriodLabel;
    const std::string &collidingSystem = fCfg.collidingSystem;
    const std::string &sqrtsLabel = fCfg.sqrtsLabel;
    const double collisionEnergy = fCfg.collisionEnergyTeV;
    const std::string &dataSetLabel = fCfg.dataSetLabel;
    const std::string decayString = fCfg.isMatter == "matter"
                                        ? "{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}"
                                        : (fCfg.isMatter == "antimatter"
                                               ? "{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}"
                                               : "{}^{3}_{#Lambda}H({}^{3}_{#bar{#Lambda}}#bar{H}) #rightarrow ^{3}He+#pi(^{3}#bar{He}+#pi^{+})");

    auto pinfoAlice = std::make_unique<TPaveText>(0.10, 0.6, 0.38, 0.85, "NDC");
    pinfoAlice->SetBorderSize(0);
    pinfoAlice->SetFillStyle(0);
    pinfoAlice->SetTextAlign(11);
    pinfoAlice->SetTextFont(42);
    if (fCfg.alicePerformance) {
        pinfoAlice->AddText("ALICE Performance");
        pinfoAlice->AddText(Form("%s, %s @ %s = %.2f TeV",
                                 runPeriod.c_str(),
                                 collidingSystem.c_str(),
                                 sqrtsLabel.c_str(),
                                 collisionEnergy));
    }
    else {
       pinfoAlice->AddText(dataSetLabel.c_str());
       pinfoAlice->AddText(decayString.c_str());
    }
    
    dataFrame->addObject(pinfoAlice.release());

    res.rawYield = signalIntVal3s;
    res.rawYieldErr = signalIntErr3s;
    res.fittedMean = mean.getVal();
    res.fittedSigma = sigma.getVal();
    res.fittedSigmaErr = sigma.getError();
    res.fittedSigmaMC = sigmaMC.getVal();
    res.fittedSigmaMCErr = sigmaMC.getError();
    res.fittedChi2 = chi2Data;
    res.mcMassFrame = std::move(mcFrame);
    res.dataMassFrame = std::move(dataFrame);
    res.massAxis = std::move(massVar);

    return res;
}

double CtExtraction::LookupAcceptance(const BinKey &key, double &err) const {
    err = 0.0;
    auto it = std::find(fCfg.ptBins.begin(), fCfg.ptBins.end(), key.ptMin);
    if (it == fCfg.ptBins.end()) {
        throw std::runtime_error("Cannot locate pt bin edge for " + key.ToString());
    }
    size_t ptIndex = std::distance(fCfg.ptBins.begin(), it);
    if (ptIndex >= fAcceptancePerPt.size()) {
        throw std::runtime_error("Acceptance histogram missing for pt bin " + key.ToString());
    }
    TH1D *hist = fAcceptancePerPt[ptIndex].get();
    if (!hist) {
        throw std::runtime_error("Acceptance histogram is null for pt index " + std::to_string(ptIndex));
    }
    const double center = 0.5 * (key.ctMin + key.ctMax);
    int bin = hist->FindBin(center);
    err = hist->GetBinError(bin);
    return hist->GetBinContent(bin);
}

std::string CtExtraction::FormatEdge(double value) {
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

std::string CtExtraction::ExpandPattern(const std::string &pattern, const BinKey &key) const {
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

std::string CtExtraction::BuildPath(const std::string &dir,
                                    const std::string &pattern,
                                    const BinKey &key) const {
    std::filesystem::path base(dir);
    base /= ExpandPattern(pattern, key);
    return base.string();
}
