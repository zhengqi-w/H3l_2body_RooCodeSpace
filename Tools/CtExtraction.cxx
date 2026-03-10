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

    fInputMcFile = TFile::Open(fCfg.mcFile.c_str(), "READ");
    if (!fInputMcFile || fInputMcFile->IsZombie()) {
        throw std::runtime_error("Failed to open MC file: " + fCfg.mcFile);
    }
}

CtExtraction::~CtExtraction() {
    for (auto *h : fAcceptancePerPt) {
        delete h;
    }
    fAcceptancePerPt.clear();

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

        TLegend legend(0.60, 0.70, 0.90, 0.90);
        legend.SetBorderSize(0);
        legend.SetFillStyle(0);
        legend.SetTextSize(0.045);
        legend.AddEntry(hCorr.get(), "Corrected spectrum", "lep");
        legend.AddEntry(fitFunc.get(), "Exp fit", "l");
        legend.Draw();

        TPaveText pave(0.18, 0.70, 0.55, 0.90, "NDC");
        pave.SetFillStyle(0);
        pave.SetBorderSize(0);
        pave.SetTextAlign(12);
        pave.SetTextSize(0.045);
        const double chi2 = fitFunc->GetChisquare();
        const int ndf = fitFunc->GetNDF();
        const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;
        pave.AddText(Form("#tau = %.2f #pm %.2f ps", tauPs, tauPsErr));
        pave.AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
        pave.AddText(Form("Fit prob. = %.3f", fitProb));

        pave.Draw();

        canvas->Write();
    }

    stdDir->cd();
    hTauVsPt->Write();
    hTauErrVsPt->Write();

    // Explicitly flush and close the output file so downstream readers do not hit
    // the ROOT recovery path (avoids "file probably not closed" warnings).
    if (fOutputFile) {
        fOutputFile->cd();
        fOutputFile->Write();
        fOutputFile->Flush();
        fOutputFile->Close();
        fOutputFile.reset();
    }
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
        if (fCfgJson.contains(key) && fCfgJson[key].is_string()) {
            return fCfgJson[key].get<std::string>();
        }
        return fallback;
    };
    auto get_double = [&](const char *key, double fallback) {
        if (fCfgJson.contains(key) && fCfgJson[key].is_number()) {
            return fCfgJson[key].get<double>();
        }
        return fallback;
    };
    auto get_int = [&](const char *key, int fallback) {
        if (fCfgJson.contains(key) && fCfgJson[key].is_number_integer()) {
            return fCfgJson[key].get<int>();
        }
        if (fCfgJson.contains(key) && fCfgJson[key].is_number()) {
            return static_cast<int>(fCfgJson[key].get<double>());
        }
        return fallback;
    };
    auto get_bool = [&](const char *key, bool fallback) {
        if (fCfgJson.contains(key) && fCfgJson[key].is_boolean()) {
            return fCfgJson[key].get<bool>();
        }
        return fallback;
    };
    auto get_double_vec = [&](const char *key, const std::vector<double> &fallback) {
        if (!fCfgJson.contains(key) || !fCfgJson[key].is_array()) {
            return fallback;
        }
        std::vector<double> out;
        for (const auto &v : fCfgJson[key]) {
            if (v.is_number()) {
                out.push_back(v.get<double>());
            }
        }
        return out.empty() ? fallback : out;
    };
    auto get_nested_double_vec = [&](const char *key, const std::vector<std::vector<double>> &fallback) {
        if (!fCfgJson.contains(key) || !fCfgJson[key].is_array()) {
            return fallback;
        }
        std::vector<std::vector<double>> out;
        for (const auto &entry : fCfgJson[key]) {
            if (!entry.is_array()) {
                continue;
            }
            std::vector<double> row;
            for (const auto &v : entry) {
                if (v.is_number()) {
                    row.push_back(v.get<double>());
                }
            }
            if (!row.empty()) {
                out.push_back(std::move(row));
            }
        }
        return out.empty() ? fallback : out;
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
    fCfg.basicSelectionDataForMCEff = get_string("basic_selection_data_for_mc_eff", "");

    fCfg.ptBins = get_double_vec("pt_bins", std::vector<double>{});
    fCfg.ctBins = get_nested_double_vec("ct_bins", std::vector<std::vector<double>>{});
    fCfg.massRange = get_double_vec("mass_range", std::vector<double>{2.95, 3.05});
    fCfg.massBins = get_int("mass_nbins_data", 50);
    fCfg.mcMassBins = get_int("mass_nbins_mc", 80);

    fCfg.minEntriesForFit = get_double("min_entries_for_fit", 60.0);
    fCfg.minScoreShift = get_double("bdt_score_shift", 0.0);
    fCfg.runPeriodLabel = get_string("run_period_label", "Run 3");
    fCfg.collidingSystem = get_string("colliding_system", "Pb-Pb");
    fCfg.sqrtsLabel = get_string("sqrtsnn_label", "#sqrt{s_{NN}}");
    fCfg.dataSetLabel = get_string("data_set_label", "LHC23_PbPb_pass5");
    fCfg.collisionEnergyTeV = get_double("collision_energy_tev", 5.36);
    fCfg.alicePerformance = get_bool("alice_performance", false);
    fCfg.sigmaRangeMcToData = get_nested_double_vec("sigma_mc_to_data_range", std::vector<std::vector<double>>{ {0.9, 1.5}, {0.9, 1.5} , {0.9, 1.5} , {0.9, 1.5} });

    if (fCfgJson.contains("bdt_overrides")) {
        for (const auto &item : fCfgJson["bdt_overrides"]) {
            if (!item.contains("pt") || !item.contains("ct") || !item.contains("score")) {
                continue;
            }
            std::vector<double> pt;
            std::vector<double> ct;
            if (item["pt"].is_array()) {
                for (const auto &v : item["pt"]) {
                    if (v.is_number()) pt.push_back(v.get<double>());
                }
            }
            if (item["ct"].is_array()) {
                for (const auto &v : item["ct"]) {
                    if (v.is_number()) ct.push_back(v.get<double>());
                }
            }
            if (pt.size() != 2 || ct.size() != 2) {
                continue;
            }
            BinKey key{pt[0], pt[1], ct[0], ct[1]};
            if (item["score"].is_number()) {
                fUserOverrides[key] = item["score"].get<double>();
            }
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
        std::vector<std::vector<double>>{},
        fCfg.basicSelectionDataForMCEff
    );

    const std::vector<TH1D*> *perPt = nullptr;
    if (fCfg.isMatter == "matter") {
        perPt = &accResult.acc_ct_per_pt_matter;
    } else if (fCfg.isMatter == "antimatter") {
        perPt = &accResult.acc_ct_per_pt_antimatter;
    } else {
        perPt = &accResult.acc_ct_per_pt;
    }

    for (auto *h : fAcceptancePerPt) {
        delete h;
    }
    fAcceptancePerPt.clear();
    fAcceptancePerPt.reserve(perPt->size());
    for (size_t i = 0; i < perPt->size(); ++i) {
        TH1D *src = perPt->at(i);
        if (!src) {
            fAcceptancePerPt.emplace_back(nullptr);
            continue;
        }
        TH1D *clone = static_cast<TH1D*>(src->Clone(Form("acc_per_pt_%zu", i)));
        if (clone) {
            clone->SetDirectory(nullptr);
        }
        fAcceptancePerPt.emplace_back(clone);
    }

    // clean up dynamically allocated histograms inside accResult to avoid leaks
    accResult.Clear();
}

CtExtraction::WorkingPoint CtExtraction::GetWorkingPoint(const BinKey &key) const {
    auto wp = GeneralHelper::GetWpForPtCt(fCfg.workingPointFile, key.ptMin, key.ptMax, key.ctMin, key.ctMax);
    if (!wp.found) {
        throw std::runtime_error("Missing working point entry for bin " + key.ToString() + " in " + fCfg.workingPointFile);
    }
    return WorkingPoint{wp.score, wp.eff, wp.significance};
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

    const std::string bdtCutExpr = fCfg.bdtScoreColumn + " > " + std::to_string(bdtScore);
    auto filtered = node.Filter(bdtCutExpr);

    auto countBefore = node.Count();
    auto countAfter = filtered.Count();

    entriesBefore = static_cast<int>(countBefore.GetValue());
    entriesAfter = static_cast<int>(countAfter.GetValue());
    auto massValues = filtered.Take<double>(fCfg.massColumn);
    return massValues.GetValue();
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

    auto massValues = node.Take<double>(fCfg.massColumn);
    return massValues.GetValue();
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

    GeneralHelper::MassFitConfig fitCfg;
    fitCfg.massMin = fCfg.massRange[0];
    fitCfg.massMax = fCfg.massRange[1];
    fitCfg.sigmaRangeMcToData = sigmaRange;

    auto fit = GeneralHelper::FitMassSpectrum(massValues, mcMassValues, fitCfg, "pol2", "dscb");

    res.rawYield = fit.signal;
    res.rawYieldErr = fit.signalErr;
    res.fittedMean = fit.meanData;
    res.fittedSigma = fit.sigmaData;
    res.fittedSigmaErr = fit.sigmaDataErr;
    res.fittedSigmaMC = fit.sigmaMc;
    res.fittedSigmaMCErr = fit.sigmaMcErr;
    res.fittedChi2 = fit.chi2Data;
    res.mcMassFrame = std::move(fit.frameMc);
    res.dataMassFrame = std::move(fit.frame);
    res.massAxis = std::move(fit.massAxis);

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
    TH1D *hist = fAcceptancePerPt[ptIndex];
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
