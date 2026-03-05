#include "../Tools/AcceptanceHelper.h"
#include "../Tools/AbsorptionHelper.h"
#include "../Tools/GeneralHelper.hpp"
#include "../Tools/SpectrumCalculator.h"

#include <TChain.h>
#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TLine.h>
#include <TString.h>
#include <TSystem.h>
#include <TLatex.h>
#include <TBox.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <RooMsgService.h>

#include <ROOT/RDataFrame.hxx>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace {

std::string MakeDecayString(const std::string &mode) {
    if (mode == "matter") {
        return "{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}";
    }
    if (mode == "antimatter") {
        return "{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}";
    }
    if (mode == "both") {
        return "{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi";
    }
    return std::string();
}

void AddLatexLine(RooPlot *frame, double x, double y, const std::string &text) {
    if (!frame) return;
    auto latex = std::make_unique<TLatex>(x, y, text.c_str());
    latex->SetNDC();
    latex->SetTextFont(42);
    latex->SetTextSize(0.035);
    latex->SetTextAlign(11);
    latex->SetTextColor(kBlack);
    frame->addObject(latex.release());
    y -= 0.04;
}

void AddTrailAnnotation(RooPlot *frame, const std::string &bkg, const std::string &sig,
                        double bdtScore, double bdtEff) {
    if (!frame) return;
    auto pave = std::make_unique<TPaveText>(0.58, 0.12, 0.88, 0.28, "NDC");
    pave->SetBorderSize(0);
    pave->SetFillStyle(0);
    pave->SetTextAlign(12);
    pave->SetTextFont(42);
    pave->AddText(Form("Bkg: %s", bkg.c_str()));
    pave->AddText(Form("Sig: %s", sig.c_str()));
    pave->AddText(Form("BDT cut: %.4f", bdtScore));
    pave->AddText(Form("BDT eff: %.3f", bdtEff));
    frame->addObject(pave.release());
}

std::string Sanitize(const std::string &s) {
    std::string out = s;
    for (char &c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '-') c = '_';
    }
    return out;
}

void AppendCorrectionsCsv(const SpectrumResult &res, const std::pair<double, double> &cenRange,
                         const Config &cfg, const std::string &tag, const std::string &outPath) {
    if (res.corrections.empty()) {
        std::cerr << "[Warn] No correction rows to write for " << outPath << "\n";
        return;
    }
    const bool fileExists = std::filesystem::exists(outPath);
    std::ofstream ofs(outPath, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Warn] Failed to open CSV for writing: " << outPath << "\n";
        return;
    }
    if (!fileExists) {
        ofs << "tag,cen_min,cen_max,pt_min,pt_max,label,raw,raw_err,acc,abso,bdt_eff,bin_width,n_events,branching_ratio,delta_rap,matter_ratio,corrected,corrected_err\n";
    }
    ofs << std::defaultfloat << std::setprecision(8);
    for (const auto &row : res.corrections) {
        ofs << tag << ','
            << cenRange.first << ',' << cenRange.second << ','
            << row.ptMin << ',' << row.ptMax << ','
            << row.label << ','
            << row.raw << ',' << row.rawErr << ','
            << row.acc << ',' << row.abso << ','
            << row.bdtEff << ','
            << row.binWidth << ','
            << std::scientific << std::setprecision(6) << row.nEvents << ','
            << std::defaultfloat << std::setprecision(8)
            << cfg.branchingRatio << ','
            << cfg.deltaRap << ','
            << row.matterRatio << ','
            << std::scientific << std::setprecision(6) << row.corrected << ',' << row.correctedErr << '\n';
        ofs << std::defaultfloat << std::setprecision(8);
    }
}

void AnnotateSpectrumFrames(SpectrumResult &res, const Config &cfg, double nEvents) {
    if (res.frames.empty()) return;
    const std::string experimentLine = "LHC23_PbPb_pass5 (#sqrt{#it{s_{NN}}} = 5.36TeV)";
    const std::string decayLine = MakeDecayString(cfg.isMatter);
    std::string eventsLine = Form("N_{ev} = %.2e", nEvents);
    for (const auto &frame : res.frames) {
        if (!frame) continue;
        AddLatexLine(frame.get(), 0.15, 0.85, experimentLine);
        if (!decayLine.empty()) {
            AddLatexLine(frame.get(), 0.15, 0.8, decayLine);
        }
        AddLatexLine(frame.get(), 0.15, 0.75, eventsLine);
    }
}

void RefreshCanvases(SpectrumResult &res, const SpectrumCalculator &calc) {
    for (size_t i = 0; i < res.canvases.size() && i < res.frames.size(); ++i) {
        calc.RedrawFrameCanvas(res.canvases[i].get(), res.frames[i].get(), false);
    }
    for (size_t i = 0; i < res.canvasesMc.size() && i < res.framesMc.size(); ++i) {
        calc.RedrawFrameCanvas(res.canvasesMc[i].get(), res.framesMc[i].get(), true);
    }
}

double GetNEvents(const Config &cfg, const std::pair<double, double> &cenRange) {
    if (cfg.nEventsFile.empty() || cfg.nEventsHist.empty()) {
        throw std::runtime_error("analysis_results_file and n_events_hist are required");
    }
    TFile f(cfg.nEventsFile.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Failed to open " + cfg.nEventsFile);
    }
    TH1 *h = dynamic_cast<TH1 *>(f.Get(cfg.nEventsHist.c_str()));
    if (!h) {
        throw std::runtime_error("Histogram not found: " + cfg.nEventsHist);
    }
    int bmin = h->GetXaxis()->FindBin(cenRange.first + 1e-3);
    int bmax = h->GetXaxis()->FindBin(cenRange.second - 1e-3);
    return h->Integral(bmin, bmax);
}

std::unique_ptr<TChain> MakeChainFromFile(const std::string &file, const std::string &tree) {
    auto chain = std::make_unique<TChain>(tree.c_str());
    TFile f(file.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Failed to open " + file);
    }
    TTree *t = dynamic_cast<TTree *>(f.Get(tree.c_str()));
    if (t) {
        chain->Add(file.c_str());
    } else {
        GeneralHelper::fillChainFromAO2D(*chain, &f);
    }
    if (chain->GetEntries() == 0) {
        throw std::runtime_error("No entries found for tree " + tree + " in " + file);
    }
    return chain;
}

std::unique_ptr<TH1D> BuildAcceptance(const Config &cfg, const std::pair<double, double> &cenRange,
                                      const std::vector<double> &ptEdges) {
    if (cfg.mcFileForAcceptance.empty()) {
        auto h = std::make_unique<TH1D>("h_acceptance", ";p_{T};A\times#epsilon_{geo}",
                                        static_cast<int>(ptEdges.size() - 1), ptEdges.data());
        h->SetDirectory(nullptr); // keep alive outside any current directory
        h->Reset("ICES");
        h->Add(h.get(), 0.0); // ensure zeroed content
        h->Add(h.get(), 0.0);
        for (int i = 1; i <= h->GetNbinsX(); ++i) h->SetBinContent(i, 1.0);
        return h;
    }

    if (cfg.enableImplicitMT) ROOT::EnableImplicitMT();
    auto mcChain = MakeChainFromFile(cfg.mcFileForAcceptance, cfg.treeNameMc);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    std::unique_ptr<TF1> reweightFunc;
    std::unique_ptr<TFile> reweightFile;
    if (!cfg.reweightPtFile.empty()) {
        reweightFile.reset(TFile::Open(cfg.reweightPtFile.c_str(), "READ"));
        if (!reweightFile || reweightFile->IsZombie()) {
            std::cerr << "[Warn] Failed to open reweight file: " << cfg.reweightPtFile << std::endl;
        } else {
            auto pickName = [&](double cmin, double cmax) {
                if (std::abs(cmin - 0.0) < 1e-3 && std::abs(cmax - 10.0) < 1e-3) return std::string("BlastWave_H3L_0_10");
                if (std::abs(cmin - 10.0) < 1e-3 && std::abs(cmax - 30.0) < 1e-3) return std::string("BlastWave_H3L_10_30");
                if (std::abs(cmin - 30.0) < 1e-3 && std::abs(cmax - 50.0) < 1e-3) return std::string("BlastWave_H3L_30_50");
                return std::string("BlastWave_H3L_0_10");
            };
            std::string funcName = pickName(cenRange.first, cenRange.second);
            TF1 *tmp = dynamic_cast<TF1 *>(reweightFile->Get(funcName.c_str()));
            if (!tmp) tmp = dynamic_cast<TF1 *>(reweightFile->Get("BlastWave_H3L_0_10"));
            if (tmp) {
                reweightFunc.reset(static_cast<TF1 *>(tmp->Clone()));
            } else {
                std::cerr << "[Warn] Reweight TF1 not found, skip reweight" << std::endl;
            }
        }
    }
    ROOT::RDF::RNode mcReadyNode(mcReady);
    ROOT::RDF::RNode mcReweighted = reweightFunc ? ROOT::RDF::RNode(GeneralHelper::ReWeightSpectrum(mcReadyNode, reweightFunc.get(), "fAbsGenPt")) : mcReadyNode;
    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcReweighted,
        std::vector<double>{ptEdges},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        std::vector<double>{}, // MC efficiency should not be related to Centrality Just do reweighting for each centrality
        std::vector<std::vector<double>>{},
        cfg.basicSelectionDataForMCEff // pass basic selection for data to calculate MC efficiency
    );
    TH1D *src = nullptr;
    if (cfg.isMatter == "matter") {
        if (!(accRes.acc_pt_matter == nullptr) ) {
            src = accRes.acc_pt_matter;
        }
    } else if (cfg.isMatter == "antimatter") {
        if (!(accRes.acc_pt_antimatter == nullptr) ) {
            src = accRes.acc_pt_antimatter;
        }
    } else if (cfg.isMatter == "both") {
        if (!(accRes.acc_pt_both == nullptr) ) {
            src = accRes.acc_pt_both;
        }
    }

    auto h = std::make_unique<TH1D>("h_acceptance", ";p_{T};A\times#epsilon_{geo}",
                                    static_cast<int>(ptEdges.size() - 1), ptEdges.data());
    h->SetDirectory(nullptr); // detach so histogram survives file closures
    if (src) {
        for (int i = 1; i <= h->GetNbinsX(); ++i) {
            h->SetBinContent(i, src->GetBinContent(i));
            h->SetBinError(i, src->GetBinError(i));
        }
    } else {
        std::cerr << "[Warn] Acceptance histogram missing, fallback to 1" << std::endl;
        for (int i = 1; i <= h->GetNbinsX(); ++i) h->SetBinContent(i, 1.0);
    }
    return h;
}

std::unique_ptr<TH1D> BuildAbsorption(const Config &cfg, const std::pair<double, double> &cenRange,
                                       const std::vector<double> &ptEdges) {
    auto h = std::make_unique<TH1D>("h_absorption", ";p_{T};#epsilon_{abso}",
                                    static_cast<int>(ptEdges.size() - 1), ptEdges.data());
    h->SetDirectory(nullptr); // detach so histogram is not owned by a transient TFile
    if (cfg.mcFileForAbsorption.empty()) {
        for (int i = 1; i <= h->GetNbinsX(); ++i) h->SetBinContent(i, 1.0);
        return h;
    }

    auto chain = MakeChainFromFile(cfg.mcFileForAbsorption, cfg.treeNameAbsorption);
    ROOT::RDataFrame rdf(*chain);

    std::unique_ptr<TF1> reweightFunc;
    std::unique_ptr<TFile> reweightFile;
    if (!cfg.reweightPtFile.empty()) {
        reweightFile.reset(TFile::Open(cfg.reweightPtFile.c_str(), "READ"));
        if (!reweightFile || reweightFile->IsZombie()) {
            std::cerr << "[Warn] Failed to open reweight file: " << cfg.reweightPtFile << std::endl;
        } else {
            auto pickName = [&](double cmin, double cmax) {
                if (std::abs(cmin - 0.0) < 1e-3 && std::abs(cmax - 10.0) < 1e-3) return std::string("BlastWave_H3L_0_10");
                if (std::abs(cmin - 10.0) < 1e-3 && std::abs(cmax - 30.0) < 1e-3) return std::string("BlastWave_H3L_10_30");
                if (std::abs(cmin - 30.0) < 1e-3 && std::abs(cmax - 50.0) < 1e-3) return std::string("BlastWave_H3L_30_50");
                return std::string("BlastWave_H3L_0_10");
            };
            std::string funcName = pickName(cenRange.first, cenRange.second);
            TF1 *tmp = dynamic_cast<TF1 *>(reweightFile->Get(funcName.c_str()));
            if (!tmp) tmp = dynamic_cast<TF1 *>(reweightFile->Get("BlastWave_H3L_0_10"));
            if (tmp) {
                reweightFunc.reset(static_cast<TF1 *>(tmp->Clone()));
            } else {
                std::cerr << "[Warn] Reweight TF1 not found, skip reweight" << std::endl;
            }
        }
    }

    ROOT::RDF::RNode rdfBase(rdf);
    ROOT::RDF::RNode rdfWeighted = reweightFunc ? ROOT::RDF::RNode(GeneralHelper::ReWeightSpectrum(rdfBase, reweightFunc.get(), "pt")) : rdfBase;

    Absorption::SpectrumAbsorptionCalculator calc(rdfWeighted, ptEdges, 7.6);
    calc.Calculate();

    std::string key = cfg.isMatter.empty() ? std::string("both") : cfg.isMatter;
    if (key != "both" && key != "matter" && key != "antimatter") key = "both";

    const auto &ratioMap = calc.Ratio();
    auto it = ratioMap.find(key);
    if (it == ratioMap.end()) {
        std::cerr << "[Warn] Absorption ratio missing, fallback to 1" << std::endl;
        for (int i = 1; i <= h->GetNbinsX(); ++i) h->SetBinContent(i, 1.0);
        return h;
    }

    const TH1F &src = it->second;
    for (int i = 1; i <= h->GetNbinsX(); ++i) {
        h->SetBinContent(i, src.GetBinContent(i));
        h->SetBinError(i, src.GetBinError(i));
    }
    return h;
}

std::shared_ptr<ROOT::RDataFrame> MakeSnapshotRdf(const std::string &path, const std::string &tree) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Snapshot not found: " + path);
    }
    return std::make_shared<ROOT::RDataFrame>(tree, path);
}

std::vector<BinInput> BuildBins(const Config &cfg, const WPSummaryReader &wpReader,
                                const std::pair<double, double> &cenRange,
                                const std::vector<double> &ptEdges,
                                const TH1D *hAcc, const TH1D *hAbso) {
    std::vector<BinInput> bins;
    for (size_t i = 0; i + 1 < ptEdges.size(); ++i) {
        BinKey key{cenRange.first, cenRange.second, ptEdges[i], ptEdges[i + 1], -1.0, -1.0};
        std::string label = MakeLabel(key);
        std::string dataPath = cfg.snapshotDir + "/data_" + label + ".root";
        std::string mcPath = cfg.snapshotDir + "/mc_" + label + ".root";

        BinInput bin;
        bin.ptMin = ptEdges[i];
        bin.ptMax = ptEdges[i + 1];
        bin.dfData = MakeSnapshotRdf(dataPath, cfg.treeNameData);
        bin.dfMc = MakeSnapshotRdf(mcPath, cfg.treeNameMc);
        bin.wp = wpReader.Lookup(key);
        bin.acceptance = hAcc ? hAcc->GetBinContent(static_cast<int>(i + 1)) : 1.0;
        bin.absorption = hAbso ? hAbso->GetBinContent(static_cast<int>(i + 1)) : 1.0;
        bin.label = label;
        bins.push_back(std::move(bin));
    }
    return bins;
}

std::vector<double> CollectEdges(const std::vector<BinInput> &bins) {
    std::vector<double> edges;
    edges.reserve(bins.size() + 1);
    for (size_t i = 0; i < bins.size(); ++i) {
        if (i == 0) edges.push_back(bins[i].ptMin);
        edges.push_back(bins[i].ptMax);
    }
    return edges;
}

void WriteSpectrum(const SpectrumResult &res, TDirectory *dir, bool writeFrames) {
    if (!dir) return;
    TDirectory::TContext ctx(dir);
    auto writeHist = [dir](auto &h) {
        if (!h) return;
        h->SetDirectory(dir);
        h->Write();
        h->SetDirectory(nullptr);
    };
    writeHist(res.hRaw);
    writeHist(res.hCorr);
    writeHist(res.hAcc);
    writeHist(res.hAbso);
    writeHist(res.hBdtEff);
    if (writeFrames) {
        for (const auto &f : res.frames) {
            if (f) dir->WriteObject(f.get(), f->GetName());
        }
        for (const auto &f : res.framesMc) {
            if (f) dir->WriteObject(f.get(), f->GetName());
        }
        for (const auto &c : res.canvases) {
            if (c) dir->WriteObject(c.get(), c->GetName());
        }
        for (const auto &c : res.canvasesMc) {
            if (c) dir->WriteObject(c.get(), c->GetName());
        }
    }
}

std::vector<BinInput> ShiftWorkingPoints(const std::vector<BinInput> &bins, double relShift) {
    std::vector<BinInput> out = bins;
    for (auto &b : out) {
        b.wp.score *= (1.0 + relShift);
        b.wp.efficiency *= (1.0 + relShift);
    }
    return out;
}

struct BdtCandidate {
    double score{0.0};
    double efficiency{0.0};
};

std::vector<std::pair<double, double>> LoadScoreEfficiencyArray(const std::string &path) {
    std::vector<std::pair<double, double>> rows;
    if (path.empty()) return rows;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[Warn] score-efficiency array not found: " << path << std::endl;
        return rows;
    }
    double score = 0.0, eff = 0.0;
    while (ifs >> score >> eff) {
        rows.emplace_back(score, eff);
    }
    if (rows.empty()) {
        std::cerr << "[Warn] score-efficiency array empty: " << path << std::endl;
    }
    return rows;
}

std::vector<BdtCandidate> BuildBdtCandidates(double wpScore, double wpEff, int totalPoints,
                                             const std::vector<std::pair<double, double>> &arr) {
    std::vector<BdtCandidate> out;
    if (totalPoints <= 0) {
        out.push_back({wpScore, wpEff});
        return out;
    }
    if (arr.empty()) {
        out.push_back({wpScore, wpEff});
        return out;
    }
    // find nearest score index to WP
    size_t centerIdx = 0;
    double bestDiff = std::numeric_limits<double>::max();
    for (size_t i = 0; i < arr.size(); ++i) {
        double diff = std::abs(arr[i].first - wpScore);
        if (diff < bestDiff) {
            bestDiff = diff;
            centerIdx = i;
        }
    }
    const size_t totalDesired = static_cast<size_t>(totalPoints);
    const size_t belowTarget = (totalDesired - 1) / 2;
    const size_t aboveTarget = totalDesired - 1 - belowTarget;

    out.push_back({arr[centerIdx].first, arr[centerIdx].second});
    size_t belowAdded = 0, aboveAdded = 0;
    size_t belowIdx = centerIdx;
    size_t aboveIdx = centerIdx;
    while (out.size() < totalDesired) {
        bool tookBelow = false;
        bool tookAbove = false;
        if (belowAdded < belowTarget && belowIdx > 0) {
            --belowIdx;
            out.push_back({arr[belowIdx].first, arr[belowIdx].second});
            ++belowAdded;
            tookBelow = true;
        }
        if (out.size() >= totalDesired) break;
        if (aboveAdded < aboveTarget && aboveIdx + 1 < arr.size()) {
            ++aboveIdx;
            out.push_back({arr[aboveIdx].first, arr[aboveIdx].second});
            ++aboveAdded;
            tookAbove = true;
        }
        if (!tookBelow && belowAdded < belowTarget && aboveAdded >= aboveTarget && aboveIdx + 1 < arr.size()) {
            ++aboveIdx;
            out.push_back({arr[aboveIdx].first, arr[aboveIdx].second});
            ++aboveAdded;
        } else if (!tookAbove && aboveAdded < aboveTarget && belowAdded >= belowTarget && belowIdx > 0) {
            --belowIdx;
            out.push_back({arr[belowIdx].first, arr[belowIdx].second});
            ++belowAdded;
        } else if (!tookBelow && !tookAbove) {
            break; // cannot expand further
        }
    }
    return out;
}

std::vector<std::tuple<size_t, size_t, size_t, size_t>> BuildTrailCombos(size_t nBdt, size_t nBkg,
                                                                         size_t nSig, size_t nAbso) {
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> combos;
    combos.reserve(nBdt * nBkg * nSig * nAbso);
    for (size_t ibdt = 0; ibdt < nBdt; ++ibdt) {
        for (size_t ibkg = 0; ibkg < nBkg; ++ibkg) {
            for (size_t isig = 0; isig < nSig; ++isig) {
                for (size_t iabso = 0; iabso < nAbso; ++iabso) {
                    combos.emplace_back(ibdt, ibkg, isig, iabso);
                }
            }
        }
    }
    return combos;
}

std::string BuildScoreEffPath(const Config &cfg, const std::string &label) {
    if (cfg.systEfficiencyArrayPath.empty()) return std::string();
    return cfg.systEfficiencyArrayPath + "/score_efficiency_array_" + label + ".txt";
}

} // namespace

int BdtSpectrumExtraction(const char *cfgPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/bdt_spectrum.json") {
    if (!cfgPath) {
        std::cerr << "Usage: root -l -b -q 'BdtSpectrumExtraction.C(\"config.json\")'\n";
        return 1;
    }

    // Silence RooFit info prints during systematic trails
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);

    Config cfg = LoadConfig(cfgPath);
    if (cfg.enableImplicitMT) ROOT::EnableImplicitMT();
    std::filesystem::create_directories(cfg.outputDir);
    std::string mergedCsvPath;
    if (cfg.do_QA_afterward) {
        mergedCsvPath = cfg.outputDir + "/corrections_all.csv";
        std::error_code ec;
        std::filesystem::remove(mergedCsvPath, ec); // clear previous content if present
    }

    WPSummaryReader wpReader(cfg.wpFile);
    SpectrumCalculator calculator(cfg);

    auto saveHistPdf = [](TH1 &h, const std::string &path, const std::string &opt = "HIST") {
        auto cTmp = std::make_unique<TCanvas>(Form("c_tmp_%s", h.GetName()), h.GetTitle(), 900, 700);
        h.Draw(opt.c_str());
        cTmp->SaveAs(path.c_str());
    };

    for (size_t icen = 0; icen + 1 < cfg.cenBins.size(); ++icen) {
        std::pair<double, double> cenRange{cfg.cenBins[icen], cfg.cenBins[icen + 1]};
        std::vector<double> ptEdges = (!cfg.ptBinsByCen.empty() && icen < cfg.ptBinsByCen.size()) ? cfg.ptBinsByCen[icen] : cfg.ptBins;
        if (ptEdges.size() < 2) {
            std::cerr << "[Warn] Skip centrality bin due to empty pt bins" << std::endl;
            continue;
        }
        cout << "[Info] Processing centrality " << cenRange.first << "-" << cenRange.second << " with pt bins:";
        for (double e : ptEdges) cout << " " << e;
        cout << std::endl;

        auto hAcc = BuildAcceptance(cfg, cenRange, ptEdges);
        auto hAbso = BuildAbsorption(cfg, cenRange, ptEdges);
        double nEvents = GetNEvents(cfg, cenRange);
        auto bins = BuildBins(cfg, wpReader, cenRange, ptEdges, hAcc.get(), hAbso.get());
        if (bins.empty()) {
            std::cerr << "[Warn] No bins built for centrality " << cenRange.first << "-" << cenRange.second << std::endl;
            continue;
        }

        std::string cenDirName = Form("cen%d-%d", static_cast<int>(cenRange.first), static_cast<int>(cenRange.second));
        std::filesystem::create_directories(cfg.outputDir + "/" + cenDirName);
        std::string outPath = cfg.outputDir + "/" + cenDirName + "/pt_analysis_pbpb.root";

        TFile fout(outPath.c_str(), "RECREATE");
        TDirectory *stdDir = fout.mkdir("std");
        SpectrumResult resStd = calculator.Calculate(bins, nEvents, cfg.bkgFunc, cfg.sigFunc, cfg.isMatter, true, "_std");
        AnnotateSpectrumFrames(resStd, cfg, nEvents);
        RefreshCanvases(resStd, calculator);
        if (cfg.do_QA_afterward) {
            AppendCorrectionsCsv(resStd, cenRange, cfg, "std", mergedCsvPath);
        }
        WriteSpectrum(resStd, stdDir, true);

        if (cfg.do_QA_afterward) {
            std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
            if (resStd.hRaw) saveHistPdf(*resStd.hRaw, basePdfDir + "/h_raw_std.pdf", "E1");
            if (resStd.hCorr) saveHistPdf(*resStd.hCorr, basePdfDir + "/h_corr_std.pdf", "E1");
            if (resStd.hAcc) saveHistPdf(*resStd.hAcc, basePdfDir + "/h_acc_std.pdf");
            if (resStd.hAbso) saveHistPdf(*resStd.hAbso, basePdfDir + "/h_abso_std.pdf");
            if (resStd.hBdtEff) saveHistPdf(*resStd.hBdtEff, basePdfDir + "/h_bdteff_std.pdf");
            for (const auto &c : resStd.canvases) {
                if (c) c->SaveAs((basePdfDir + "/" + std::string(c->GetName()) + ".pdf").c_str());
            }
        }

        fout.cd();
        std::vector<double> edges = CollectEdges(bins);
        TH1D hSystFit("h_systematics_fit", ";p_{T};#sigma_{syst}^{fit}", static_cast<int>(edges.size() - 1), edges.data());
        TH1D hSystAbso("h_systematics_absorption", ";p_{T};#sigma_{syst}^{abso}", static_cast<int>(edges.size() - 1), edges.data());
        TH1D hSystTotal("h_systematics_total", ";p_{T};#sigma_{syst}^{total}", static_cast<int>(edges.size() - 1), edges.data());
        hSystFit.SetDirectory(nullptr);
        hSystAbso.SetDirectory(nullptr);
        hSystTotal.SetDirectory(nullptr);
        hSystFit.SetStats(false);
        hSystAbso.SetStats(false);
        hSystTotal.SetStats(false);

        if (cfg.doSystematics) {
            std::cout << "[Info] Start systematics for centrality " << cenRange.first << "-" << cenRange.second << std::endl;
            std::mt19937 rng(static_cast<unsigned int>(cfg.randomSeed));
            TDirectory *sysDir = fout.mkdir("sys");
            std::filesystem::create_directories(cfg.outputDir + "/" + cenDirName + "/sys");
            std::ofstream logFile(cfg.outputDir + "/" + cenDirName + "/sys/trails.log");
            if (logFile.is_open()) {
                logFile << "cen,ptmin,ptmax,trail,bdt_score,bdt_eff,bkg_func,sig_func,abso_file,abso,chi2ndf,significance,raw,raw_err,corr,corr_err,pass\n";
            }

            const std::vector<std::string> bkgFuncs = cfg.bkgFuncSyst.empty() ? std::vector<std::string>{cfg.bkgFunc} : cfg.bkgFuncSyst;
            const std::vector<std::string> sigFuncs = cfg.systSignalFuncs.empty() ? std::vector<std::string>{cfg.sigFunc} : cfg.systSignalFuncs;
            const std::vector<std::string> absoFiles = cfg.systAbsorptionFiles.empty() ? std::vector<std::string>{cfg.mcFileForAbsorption} : cfg.systAbsorptionFiles;


            std::map<std::string, std::unique_ptr<TH1D>> absoHists;
            for (const auto &absoFile : absoFiles) {
                Config cfgAbso = cfg;
                cfgAbso.mcFileForAbsorption = absoFile;
                absoHists[absoFile] = BuildAbsorption(cfgAbso, cenRange, ptEdges);
            }

            const double matterRatio = (cfg.isMatter == "both") ? 2.0 : 1.0;
            for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
                const auto &bin = bins[ibin];
                const std::string binLabel = bin.label;
                TDirectory *binDir = sysDir ? sysDir->mkdir(binLabel.c_str()) : nullptr;

                std::string arrayPath = BuildScoreEffPath(cfg, binLabel);
                auto scoreArray = LoadScoreEfficiencyArray(arrayPath);
                auto bdtCandidates = BuildBdtCandidates(bin.wp.score, bin.wp.efficiency, cfg.systBdtScoreNPoints, scoreArray);
                if (bdtCandidates.empty()) bdtCandidates.push_back({bin.wp.score, bin.wp.efficiency});

                auto mcMass = bin.dfMc->Filter("fMassH3L>2.95 && fMassH3L<3.02").Take<double>("fMassH3L");
                std::vector<ROOT::RDF::RResultPtr<std::vector<double>>> dataMassCache;
                dataMassCache.reserve(bdtCandidates.size());
                for (const auto &cand : bdtCandidates) {
                    std::string cut = Form("model_output > %f", cand.score);
                    dataMassCache.emplace_back(bin.dfData->Filter(cut).Take<double>("fMassH3L"));
                }

                // Fit-only systematics: fix absorption to the nominal one during trail studies
                auto combos = BuildTrailCombos(bdtCandidates.size(), bkgFuncs.size(), sigFuncs.size(), 1);
                std::shuffle(combos.begin(), combos.end(), rng);
                if (combos.empty()) {
                    std::cerr << "[Warn] No systematic combos for bin " << binLabel << std::endl;
                    continue;
                }
                const size_t nUse = std::min(combos.size(), static_cast<size_t>(std::max(1, cfg.systNtrails)));

                const double stdCorr = resStd.hCorr ? resStd.hCorr->GetBinContent(static_cast<int>(ibin + 1)) : 0.0;
                const double stdCorrErr = resStd.hCorr ? resStd.hCorr->GetBinError(static_cast<int>(ibin + 1)) : 0.0;
                double range = stdCorrErr > 0 ? 5.0 * stdCorrErr : (std::abs(stdCorr) * 0.6);
                //if (range < 1e-6) range = 1.0;
                TH1D hCorrDist(Form("h_corr_syst_%s", binLabel.c_str()), ";Corrected counts;Entries", 100, stdCorr - range, stdCorr + range);
                hCorrDist.SetDirectory(nullptr);
                hCorrDist.SetStats(false);
                const double baselineAbso = (resStd.hAbso) ? resStd.hAbso->GetBinContent(static_cast<int>(ibin + 1)) : 1.0;
                TLine *lineCorr = nullptr;
                TBox *bandCorr = nullptr;
                TBox *bandGauss = nullptr;
                TF1 *fgaus = nullptr;
                double gausMean = std::numeric_limits<double>::quiet_NaN();
                double gausSigma = std::numeric_limits<double>::quiet_NaN();

                size_t trailCounter = 0;
                for (size_t icombo = 0; icombo < nUse; ++icombo) {
                    const auto [bdtIdx, bkgIdx, sigIdx, absoIdx] = combos[icombo];
                    const auto &cand = bdtCandidates.at(bdtIdx);
                    const std::string bkg = bkgFuncs.at(bkgIdx);
                    const std::string sig = sigFuncs.at(sigIdx);
                    const std::string absoFile = cfg.mcFileForAbsorption;
                    int barLen = 20;
                    int filled = static_cast<int>(std::round(static_cast<double>(trailCounter + 1) / nUse * barLen));
                    std::string bar(filled, '<');
                    std::cout << "\r[Info] Bin " << binLabel << " trails " << (trailCounter + 1) << "/" << nUse
                              << " " << bar << std::flush;
                    double absoVal = 1.0;
                    auto itAbso = absoHists.find(absoFile);
                    if (itAbso != absoHists.end() && itAbso->second) {
                        absoVal = itAbso->second->GetBinContent(static_cast<int>(ibin + 1));
                    }
                    // absorption variation fixed to nominal during fit trails
                    const double eff = (cand.efficiency > 0) ? cand.efficiency : bin.wp.efficiency;

                    FitResult fit = calculator.FitMassPublic(*dataMassCache.at(bdtIdx), *mcMass, bkg, sig);
                    const double bw = bin.ptMax - bin.ptMin;
                    const double acc = (bin.acceptance > 0) ? bin.acceptance : 1.0;
                    double corr = 0.0;
                    double corrErr = 0.0;
                    if (bw > 0 && acc > 0 && absoVal > 0 && eff > 0) {
                        double corrBase = fit.signal / acc / absoVal / eff / bw / nEvents / cfg.branchingRatio / cfg.deltaRap;
                        double corrErrBase = fit.signalErr / acc / absoVal / eff / bw / nEvents / cfg.branchingRatio / cfg.deltaRap;
                        corrBase /= matterRatio;
                        corrErrBase /= matterRatio;
                        corr = corrBase;
                        corrErr = corrErrBase;
                    }

                    const bool withinStd = (stdCorrErr > 0) ? (std::abs(corr - stdCorr) <= 5.0 * stdCorrErr) : true;
                    const bool pass = (fit.chi2Data < cfg.systThrChi2Ndf) && (fit.significance > cfg.systThrSignificance) && withinStd && std::isfinite(corr) && std::isfinite(corrErr);
                    if (pass) {
                        hCorrDist.Fill(corr);
                    }

                    if (logFile.is_open()) {
                        logFile << cenRange.first << '-' << cenRange.second << ',' << bin.ptMin << ',' << bin.ptMax << ','
                                << trailCounter << ',' << cand.score << ',' << eff << ',' << bkg << ',' << sig << ','
                                << absoFile << ',' << absoVal << ',' << fit.chi2Data << ',' << fit.significance << ','
                                << fit.signal << ',' << fit.signalErr << ',' << corr << ',' << corrErr << ',' << pass << '\n';
                    }

                    if (binDir && fit.frame) {
                        TDirectory::TContext ctx(binDir);
                        std::string name = Form("data_frame_%s_trail%zu", binLabel.c_str(), trailCounter);
                        AddTrailAnnotation(fit.frame.get(), bkg, sig, cand.score, eff);
                        fit.frame->SetName(name.c_str());
                        fit.frame->SetTitle(name.c_str());
                        binDir->WriteObject(fit.frame.get(), name.c_str());
                    }
                    ++trailCounter;
                }
                std::cout << std::endl;

                // annotate distributions with baseline markers
                if (std::isfinite(stdCorr)) {
                    lineCorr = new TLine(stdCorr, 0.0, stdCorr, hCorrDist.GetMaximum() * 1.05);
                    lineCorr->SetLineColor(kRed); // deep red for std line
                    lineCorr->SetLineStyle(kDashed);
                    bandCorr = new TBox(stdCorr - stdCorrErr, 0.0, stdCorr + stdCorrErr, hCorrDist.GetMaximum() * 1.02);
                    bandCorr->SetFillColorAlpha(kOrange + 2, 0.36); // light orange band
                    bandCorr->SetLineColor(kRed + 2);
                }
                double sigma = 0.0;
                if (hCorrDist.GetEntries() > 2) {
                    const double initMean = hCorrDist.GetMean();
                    const double initSigma = std::max(hCorrDist.GetRMS(), 1e-9);
                    TF1 gausFunc("gaus", "gaus", hCorrDist.GetXaxis()->GetXmin(), hCorrDist.GetXaxis()->GetXmax());
                    gausFunc.SetParameters(hCorrDist.GetMaximum(), initMean, initSigma); // p0=amp, p1=mean, p2=sigma
                    auto fitRes = hCorrDist.Fit(&gausFunc, "QS");
                    fgaus = hCorrDist.GetFunction("gaus");
                    if (fitRes == 0 && fgaus) {
                        fgaus->SetLineColor(kGreen + 3); // deep green for Gauss fit
                        fgaus->SetLineWidth(2);
                        fgaus->SetLineStyle(kSolid);
                        sigma = fgaus->GetParameter(2);
                        gausMean = fgaus->GetParameter(1);
                        gausSigma = sigma;
                        bandGauss = new TBox(gausMean - gausSigma, 0.0, gausMean + gausSigma, hCorrDist.GetMaximum() * 1.02);
                        bandGauss->SetFillColorAlpha(kGreen + 1, 0.18); // light green band
                        bandGauss->SetLineColor(kGreen + 3);
                        std::cout << "[Info] Bin " << binLabel << " Gauss fit ok: entries=" << hCorrDist.GetEntries()
                                  << " mean=" << gausMean << " sigma=" << gausSigma << " rms=" << hCorrDist.GetRMS() << std::endl;
                    } else {
                        sigma = hCorrDist.GetRMS();
                        std::cout << "[Warning] Bin " << binLabel << " Gauss fit failed, using RMS=" << sigma << std::endl;
                    }
                }
                hSystFit.SetBinContent(static_cast<int>(ibin + 1), sigma);

                if (sysDir) {
                    TDirectory::TContext ctx(sysDir);
                    hCorrDist.Write();
                    auto cCorr = std::make_unique<TCanvas>(Form("c_corr_syst_%s", binLabel.c_str()), Form("c_corr_syst_%s", binLabel.c_str()), 900, 700);
                    if (cCorr) {
                        cCorr->cd();
                        hCorrDist.SetTitle(Form("%s", binLabel.c_str()));
                        hCorrDist.Draw("HIST SAME");
                        if (bandCorr) {
                            bandCorr->SetFillStyle(3004);
                            bandCorr->SetFillColor(kOrange - 2);
                            bandCorr->SetLineColor(kRed + 2);
                            bandCorr->Draw("same");
                        }
                        if (bandGauss) {
                            bandGauss->SetFillStyle(3005);
                            bandGauss->SetFillColor(kGreen + 1);
                            bandGauss->SetLineColor(kGreen + 3);
                            bandGauss->Draw("same");
                        }
                        if (fgaus) {
                            fgaus->Draw("same");
                        }
                        if (lineCorr) lineCorr->Draw("same");
                        auto leg = std::make_unique<TLegend>(0.58, 0.70, 0.88, 0.88);
                        leg->SetBorderSize(0);
                        leg->SetFillStyle(0);
                        leg->SetTextFont(42);
                        leg->AddEntry(&hCorrDist, "Trails passing cuts", "lep");
                        if (lineCorr) leg->AddEntry(lineCorr, "Std value", "l");
                        if (bandCorr) leg->AddEntry(bandCorr, "Std stat band", "f");
                        if (bandGauss) leg->AddEntry(bandGauss, "Gauss #pm1#sigma", "f");
                        if (fgaus) leg->AddEntry(fgaus, "Gauss fit", "l");
                        leg->Draw("same");

                        auto pave = std::make_unique<TPaveText>(0.14, 0.74, 0.44, 0.90, "NDC");
                        pave->SetBorderSize(0);
                        pave->SetFillStyle(0);
                        pave->SetTextAlign(12);
                        pave->SetTextFont(42);
                        if (std::isfinite(gausMean) && std::isfinite(gausSigma)) {
                            pave->AddText(Form("Gauss #mu = %.3e", gausMean));
                            pave->AddText(Form("Gauss #sigma = %.3e", gausSigma));
                        } else {
                            pave->AddText("Gauss fit: n/a");
                        }
                        pave->AddText(Form("RMS = %.3e", hCorrDist.GetRMS()));
                        pave->AddText(Form("Central = %.3e", hCorrDist.GetMean()));
                        pave->Draw("same");

                        //cCorr->SetGrid();
                        cCorr->Write();
                        if (cfg.do_QA_afterward) {
                            std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
                            cCorr->SaveAs((basePdfDir + "/c_corr_syst_" + binLabel + ".pdf").c_str());
                        }
                    }
                }
            }

            // absorption-only systematics evaluated separately from fit trails
            TDirectory *sysAbsoDir = sysDir ? sysDir->mkdir("absorption") : nullptr;
            std::vector<std::string> absoScanFiles = absoFiles;
            if (!cfg.mcFileForAbsorption.empty() && std::find(absoScanFiles.begin(), absoScanFiles.end(), cfg.mcFileForAbsorption) == absoScanFiles.end()) {
                absoScanFiles.push_back(cfg.mcFileForAbsorption);
                Config cfgAbso = cfg;
                cfgAbso.mcFileForAbsorption = cfg.mcFileForAbsorption;
                absoHists[cfg.mcFileForAbsorption] = BuildAbsorption(cfgAbso, cenRange, ptEdges);
            }
            for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
                const std::string binLabel = bins[ibin].label;
                const double baselineCorr = resStd.hCorr ? resStd.hCorr->GetBinContent(static_cast<int>(ibin + 1)) : 0.0;
                const double baselineAbso = resStd.hAbso ? resStd.hAbso->GetBinContent(static_cast<int>(ibin + 1)) : 1.0;

                const int nAbsoVar = static_cast<int>(absoScanFiles.size());
                TH1D hAbsoCorrDist(Form("h_corr_abso_syst_%s", binLabel.c_str()), ";n#times#sigma(He3);Corrected counts", nAbsoVar, 0.5, static_cast<double>(nAbsoVar) + 0.5);
                hAbsoCorrDist.SetDirectory(nullptr);
                hAbsoCorrDist.SetStats(false);

                double minCorr = std::numeric_limits<double>::infinity();
                double maxCorr = -std::numeric_limits<double>::infinity();
                for (size_t iabso = 0; iabso < absoScanFiles.size(); ++iabso) {
                    const auto &absoFile = absoScanFiles[iabso];
                    double absoVal = 1.0;
                    auto itAbso = absoHists.find(absoFile);
                    if (itAbso != absoHists.end() && itAbso->second) {
                        absoVal = itAbso->second->GetBinContent(static_cast<int>(ibin + 1));
                    }
                    if (absoVal <= 0.0 || baselineAbso <= 0.0) {
                        continue;
                    }
                    const double corrVariant = baselineCorr * (baselineAbso / absoVal);
                    const int fillBin = static_cast<int>(iabso + 1);
                    hAbsoCorrDist.SetBinContent(fillBin, corrVariant);
                    std::string label = Sanitize(absoFile);
                    if (iabso < cfg.systAbsorptionFileLabels.size()) {
                        label = cfg.systAbsorptionFileLabels[iabso];
                    }
                    hAbsoCorrDist.GetXaxis()->SetBinLabel(fillBin, label.c_str());
                    minCorr = std::min(minCorr, corrVariant);
                    maxCorr = std::max(maxCorr, corrVariant);
                }

                const double absoSyst = std::isfinite(minCorr) && std::isfinite(maxCorr) ? (maxCorr - minCorr) : 0.0;
                hSystAbso.SetBinContent(static_cast<int>(ibin + 1), absoSyst);

                if (sysAbsoDir) {
                    TDirectory::TContext ctx(sysAbsoDir);
                    hAbsoCorrDist.Write();
                }
                if (cfg.do_QA_afterward) {
                    std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
                    saveHistPdf(hAbsoCorrDist, basePdfDir + "/h_corr_abso_syst_" + binLabel + ".pdf");
                }
            }

            // combine sources in quadrature
            for (int ibin = 1; ibin <= hSystTotal.GetNbinsX(); ++ibin) {
                const double fitVal = hSystFit.GetBinContent(ibin);
                const double absoVal = hSystAbso.GetBinContent(ibin);
                hSystTotal.SetBinContent(ibin, std::sqrt(fitVal * fitVal + absoVal * absoVal));
            }

            // final spectrum with stat (bars) + total syst (boxes)
            auto hStat = std::unique_ptr<TH1D>(static_cast<TH1D *>(resStd.hCorr ? resStd.hCorr->Clone("h_final_spectrum_stat") : nullptr));
            if (hStat) hStat->SetDirectory(nullptr);
            auto gSys = std::make_unique<TGraphAsymmErrors>(static_cast<int>(bins.size()));
            if (gSys) {
                gSys->SetName("g_final_spectrum_sys");
                gSys->SetTitle("Final spectrum with systematics");
                gSys->SetFillStyle(0); // transparent fill
                gSys->SetLineColor(kRed);
                gSys->SetLineStyle(kDotted);
                for (int i = 0; i < gSys->GetN(); ++i) {
                    double x = 0.0, y = 0.0;
                    if (hStat) {
                        x = hStat->GetXaxis()->GetBinCenter(i + 1);
                        y = hStat->GetBinContent(i + 1);
                        const double xerr = hStat->GetXaxis()->GetBinWidth(i + 1) * 0.5;
                        const double yerr = hSystTotal.GetBinContent(i + 1);
                        gSys->SetPoint(i, x, y);
                        gSys->SetPointError(i, xerr, xerr, yerr, yerr);
                    }
                }
            }

            auto cFinal = std::make_unique<TCanvas>("c_final_spectrum", "c_final_spectrum", 900, 700);
            if (cFinal) {
                if (hStat) hStat->Draw("E1");
                if (gSys) gSys->Draw("E2 SAME");
                cFinal->SetGrid();
                cFinal->SetLogy(true);
            }

            {
                TDirectory::TContext ctx(&fout);
                if (hStat) hStat->Write();
                if (gSys) gSys->Write();
                if (cFinal) cFinal->Write();
            }

            if (cfg.do_QA_afterward) {
                std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
                if (hStat) saveHistPdf(*hStat, basePdfDir + "/h_final_spectrum_stat.pdf", "E1");
                if (cFinal) cFinal->SaveAs((basePdfDir + "/c_final_spectrum.pdf").c_str());
            }
        }

        {
            TDirectory::TContext ctx(&fout);
            hSystFit.Write();
            hSystAbso.Write();
            hSystTotal.Write();
        }

        if (cfg.doSystematics) {
            auto cSystOverlay = std::make_unique<TCanvas>("c_systematics_overlay", "c_systematics_overlay", 900, 700);
            if (cSystOverlay) {
                cSystOverlay->cd();
                hSystTotal.SetLineColor(kBlack);
                hSystTotal.SetLineWidth(3);
                hSystTotal.Draw("HIST");

                hSystFit.SetLineColor(kRed + 1);
                hSystFit.SetLineWidth(2);
                hSystFit.Draw("HIST SAME");

                hSystAbso.SetLineColor(kAzure + 2);
                hSystAbso.SetLineWidth(2);
                hSystAbso.Draw("HIST SAME");

                auto leg = std::make_unique<TLegend>(0.58, 0.70, 0.88, 0.88);
                leg->SetBorderSize(0);
                leg->SetFillStyle(0);
                leg->SetTextFont(42);
                leg->AddEntry(&hSystTotal, "Total syst", "l");
                leg->AddEntry(&hSystFit, "Fit syst", "l");
                leg->AddEntry(&hSystAbso, "Absorption syst", "l");
                leg->Draw("same");
                cSystOverlay->SetGrid();

                {
                    TDirectory::TContext ctx(&fout);
                    cSystOverlay->Write();
                }

                if (cfg.do_QA_afterward) {
                    std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
                    saveHistPdf(hSystFit, basePdfDir + "/h_systematics_fit.pdf");
                    saveHistPdf(hSystAbso, basePdfDir + "/h_systematics_absorption.pdf");
                    saveHistPdf(hSystTotal, basePdfDir + "/h_systematics_total.pdf");
                    cSystOverlay->SaveAs((basePdfDir + "/c_systematics_overlay.pdf").c_str());
                }
            }
        }

        std::cout << "Saved " << outPath << "\n";
    }

    return 0;
}