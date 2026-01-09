#include "../Tools/AcceptanceHelper.h"
#include "../Tools/AbsorptionHelper.h"
#include "../Tools/GeneralHelper.hpp"
#include "../Tools/SpectrumCalculator.h"

#include <TChain.h>
#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TString.h>
#include <TSystem.h>
#include <TLatex.h>

#include <ROOT/RDataFrame.hxx>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

void AnnotateSpectrumFrames(SpectrumResult &res, const Config &cfg, double nEvents) {
    if (res.frames.empty()) return;
    const std::string experimentLine = "LHC23_PbPb_pass5 (#sqrt{#it{s_{NN}}} = 5.36TeV)";
    const std::string decayLine = MakeDecayString(cfg.isMatter);
    std::string eventsLine = Form("N_{ev} = %.0f", nEvents);
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
        std::vector<double>{},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        std::vector<double>{cenRange.first, cenRange.second},
        std::vector<std::vector<double>>{ptEdges});

    TH1D *src = nullptr;
    if (cfg.isMatter == "matter") {
        if (!accRes.acc_pt_per_cent_matter.empty()) {
            src = accRes.acc_pt_per_cent_matter.front();
        }
    } else if (cfg.isMatter == "antimatter") {
        if (!accRes.acc_pt_per_cent_antimatter.empty()) {
            src = accRes.acc_pt_per_cent_antimatter.front();
        }
    } else if (cfg.isMatter == "both") {
        if (!accRes.acc_pt_per_cent.empty()) {
            src = accRes.acc_pt_per_cent.front();
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

} // namespace

int BdtSpectrumExtraction(const char *cfgPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/bdt_spectrum.json") {
    if (!cfgPath) {
        std::cerr << "Usage: root -l -b -q 'BdtSpectrumExtraction.C(\"config.json\")'\n";
        return 1;
    }

    Config cfg = LoadConfig(cfgPath);
    if (cfg.enableImplicitMT) ROOT::EnableImplicitMT();
    std::filesystem::create_directories(cfg.outputDir);

    WPSummaryReader wpReader(cfg.wpFile);
    SpectrumCalculator calculator(cfg);

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
        SpectrumResult resStd = calculator.Calculate(bins, nEvents, cfg.bkgFunc, cfg.sigFunc, true, "_std");
        AnnotateSpectrumFrames(resStd, cfg, nEvents);
        RefreshCanvases(resStd, calculator);
        WriteSpectrum(resStd, stdDir, true);

        std::vector<std::vector<double>> trailValues(ptEdges.size() - 1);
        int trailIdx = 0;
        if (cfg.doSystematics) {
            for (double relShift : cfg.bdtScoreRelShifts) {
                for (const auto &bkg : cfg.bkgFuncSyst) {
                    if (std::abs(relShift) < 1e-6 && bkg == cfg.bkgFunc) continue; // avoid duplicating nominal
                    auto binsVar = ShiftWorkingPoints(bins, relShift);
                    std::string tag = Form("%s_trail%d_%s_shift%+.3f", cenDirName.c_str(), trailIdx, bkg.c_str(), relShift);
                    SpectrumResult resVar = calculator.Calculate(binsVar, nEvents, bkg, cfg.sigFunc, true, tag);
                    TDirectory *d = fout.mkdir(Form("trail%d", trailIdx++));
                    WriteSpectrum(resVar, d, true);
                    if (resVar.hCorr) {
                        for (int ib = 1; ib <= resVar.hCorr->GetNbinsX(); ++ib) {
                            trailValues[ib - 1].push_back(resVar.hCorr->GetBinContent(ib));
                        }
                    }
                }
            }
        }

        fout.cd();
        std::vector<double> edges = CollectEdges(bins);
        TH1D hSyst("h_systematics", ";p_{T};#sigma_{syst}", static_cast<int>(edges.size() - 1), edges.data());
        hSyst.SetDirectory(nullptr); // avoid ownership issues on close
        for (size_t i = 0; i < trailValues.size(); ++i) {
            const auto &vals = trailValues[i];
            if (vals.empty()) continue;
            double vmin = *std::min_element(vals.begin(), vals.end());
            double vmax = *std::max_element(vals.begin(), vals.end());
            double margin = 0.2 * std::max(1e-6, vmax - vmin);
            TH1D hDist(Form("h_trail_dist_pt%zu", i), ";Y_{corr};Counts", 40, vmin - margin, vmax + margin);
            hDist.SetDirectory(nullptr);
            for (double v : vals) hDist.Fill(v);
            double sigma = 0.0;
            if (hDist.GetEntries() > 5) {
                auto fitRes = hDist.Fit("gaus", "QS");
                if (fitRes == 0 && hDist.GetFunction("gaus")) {
                    sigma = hDist.GetFunction("gaus")->GetParameter(2);
                } else {
                    sigma = hDist.GetRMS();
                }
            }
            {
                TDirectory::TContext ctx(&fout);
                hDist.Write();
            }
            hSyst.SetBinContent(static_cast<int>(i + 1), sigma);
        }
        {
            TDirectory::TContext ctx(&fout);
            hSyst.Write();
        }
        std::cout << "Saved " << outPath << "\n";
    }

    return 0;
}