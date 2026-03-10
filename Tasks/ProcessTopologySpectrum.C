#include "../Tools/GeneralHelper.hpp"
#include "../Tools/AbsorptionHelper.h"
#include "../Tools/SpectrumCalculator.h"
#include "../Tools/BdtSpectrumHelper.h"

#include <ROOT/RDataFrame.hxx>
#include <TChain.h>
#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TH1D.h>
#include <TKey.h>
#include <TSystem.h>
#include <RooMsgService.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

struct TopoConfig {
    std::string snapshotDir;
    std::string nEventsFile;
    std::string nEventsHist;
    std::string mcFileForAbsorption;
    std::string mcFileForAcceptance;
    std::string reweightPtFile;
    std::string treeNameData{"O2hypcands"};
    std::string treeNameMc{"O2mchypcands"};
    std::string treeNameAbsorption{"he3candidates"};
    std::string outputDir{"Outputs"};
    std::vector<double> ptBins;
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCen;
    std::vector<std::vector<std::string>> dataSelectionTopo;
    std::string isMatter{"both"};
    std::string bkgFunc{"pol2"};
    std::string sigFunc{"dscb"};
    std::string basicSelectionDataForMcEff;
    double branchingRatio{0.25};
    double deltaRap{2.0};
    double massMin{2.96};
    double massMax{3.04};
    bool enableImplicitMT{false};
    bool do_QA_afterward{false};
};

TopoConfig LoadTopoConfig(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Config file not found: " + path);
    }
    std::ifstream ifs(path);
    json j; ifs >> j;
    TopoConfig cfg;
    auto get_string = [&](const char *key, const std::string &fallback) {
        if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
        return fallback;
    };
    auto get_double = [&](const char *key, double fallback) {
        if (j.contains(key) && j[key].is_number()) return j[key].get<double>();
        return fallback;
    };
    auto get_bool = [&](const char *key, bool fallback) {
        if (j.contains(key) && j[key].is_boolean()) return j[key].get<bool>();
        return fallback;
    };
    auto get_double_vec = [&](const char *key) {
        std::vector<double> out;
        if (!j.contains(key) || !j[key].is_array()) return out;
        for (const auto &v : j[key]) if (v.is_number()) out.push_back(v.get<double>());
        return out;
    };
    auto get_2d_double_vec = [&](const char *key) {
        std::vector<std::vector<double>> out;
        if (!j.contains(key) || !j[key].is_array()) return out;
        for (const auto &row : j[key]) {
            if (!row.is_array()) continue;
            std::vector<double> r;
            for (const auto &v : row) if (v.is_number()) r.push_back(v.get<double>());
            if (!r.empty()) out.push_back(std::move(r));
        }
        return out;
    };
    auto get_2d_string_vec = [&](const char *key) {
        std::vector<std::vector<std::string>> out;
        if (!j.contains(key) || !j[key].is_array()) return out;
        for (const auto &row : j[key]) {
            if (!row.is_array()) continue;
            std::vector<std::string> r;
            for (const auto &v : row) if (v.is_string()) r.push_back(v.get<std::string>());
            if (!r.empty()) out.push_back(std::move(r));
        }
        return out;
    };

    cfg.snapshotDir = get_string("snapshot_dir", "");
    cfg.nEventsFile = get_string("analysis_results_file", "");
    cfg.nEventsHist = get_string("n_events_hist", "");
    cfg.mcFileForAbsorption = get_string("mc_file_for_absorption", "");
    cfg.mcFileForAcceptance = get_string("mc_file_for_acceptance", "");
    cfg.reweightPtFile = get_string("reweight_pt_file", "");
    cfg.treeNameData = get_string("tree_name", cfg.treeNameData);
    cfg.treeNameMc = get_string("tree_name_mc", cfg.treeNameMc);
    cfg.treeNameAbsorption = get_string("tree_name_absorption", cfg.treeNameAbsorption);
    cfg.outputDir = get_string("output_dir", cfg.outputDir);
    cfg.ptBins = get_double_vec("pt_bins");
    cfg.cenBins = get_double_vec("cen_bins");
    cfg.ptBinsByCen = get_2d_double_vec("pt_bins_by_centrality");
    cfg.dataSelectionTopo = get_2d_string_vec("data_selection_topology");
    cfg.isMatter = get_string("is_matter", cfg.isMatter);
    cfg.bkgFunc = get_string("bkg_fit_func", cfg.bkgFunc);
    cfg.sigFunc = get_string("signal_fit_func", cfg.sigFunc);
    cfg.basicSelectionDataForMcEff = get_string("basic_selection_data_for_mc_eff", "");
    cfg.branchingRatio = get_double("branching_ratio", cfg.branchingRatio);
    cfg.deltaRap = get_double("delta_rap", cfg.deltaRap);
    cfg.massMin = get_double("mass_min", cfg.massMin);
    cfg.massMax = get_double("mass_max", cfg.massMax);
    cfg.enableImplicitMT = get_bool("enable_implicit_mt", cfg.enableImplicitMT);
    cfg.do_QA_afterward = get_bool("do_QA_afterward", get_bool("do_QA_afterwords", cfg.do_QA_afterward));
    return cfg;
}

double GetNEvents(const TopoConfig &cfg, const std::pair<double, double> &cenRange) {
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

std::unique_ptr<TH1D> BuildAbsorption(const TopoConfig &cfg, const std::vector<double> &ptEdges) {
    auto h = std::make_unique<TH1D>("h_absorption", ";p_{T};#epsilon_{abso}", static_cast<int>(ptEdges.size() - 1), ptEdges.data());
    h->SetDirectory(nullptr);
    if (cfg.mcFileForAbsorption.empty()) {
        for (int i = 1; i <= h->GetNbinsX(); ++i) h->SetBinContent(i, 1.0);
        return h;
    }

    TChain chain(cfg.treeNameAbsorption.c_str());
    chain.Add(cfg.mcFileForAbsorption.c_str());
    ROOT::RDataFrame rdf(chain);

    std::unique_ptr<TF1> reweightFunc;
    std::unique_ptr<TFile> reweightFile;
    if (!cfg.reweightPtFile.empty()) {
        reweightFile.reset(TFile::Open(cfg.reweightPtFile.c_str(), "READ"));
        if (!reweightFile || reweightFile->IsZombie()) {
            std::cerr << "[Warn] Failed to open reweight file: " << cfg.reweightPtFile << std::endl;
        } else {
            reweightFunc.reset(dynamic_cast<TF1 *>(reweightFile->Get("BlastWave_H3L_0_10")));
            if (!reweightFunc) std::cerr << "[Warn] TF1 BlastWave_H3L not found in reweight file" << std::endl;
        }
    }

    ROOT::RDF::RNode base(rdf);
    ROOT::RDF::RNode weighted = reweightFunc ? ROOT::RDF::RNode(GeneralHelper::ReWeightSpectrum(base, reweightFunc.get(), "pt")) : base;

    Absorption::SpectrumAbsorptionCalculator calc(weighted, ptEdges, 7.6);
    calc.Calculate();
    const auto &ratioMap = calc.Ratio();
    std::string key = cfg.isMatter;
    if (key != "matter" && key != "antimatter" && key != "both") key = "both";
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

std::unique_ptr<TF1> LoadReweightFunc(const TopoConfig &cfg, double cenMin, double cenMax) {
    if (cfg.reweightPtFile.empty()) return nullptr;
    auto f = std::unique_ptr<TFile>(TFile::Open(cfg.reweightPtFile.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[Warn] Failed to open reweight file: " << cfg.reweightPtFile << std::endl;
        return nullptr;
    }
    auto pickName = [&](double cmin, double cmax) {
        if (std::abs(cmin - 0.0) < 1e-3 && std::abs(cmax - 10.0) < 1e-3) return std::string("BlastWave_H3L_0_10");
        if (std::abs(cmin - 10.0) < 1e-3 && std::abs(cmax - 30.0) < 1e-3) return std::string("BlastWave_H3L_10_30");
        if (std::abs(cmin - 30.0) < 1e-3 && std::abs(cmax - 50.0) < 1e-3) return std::string("BlastWave_H3L_30_50");
        if (std::abs(cmin - 50.0) < 1e-3 && std::abs(cmax - 80.0) < 1e-3) return std::string("BlastWave_H3L_50_80");
        return std::string("BlastWave_H3L");
    };
    std::string name = pickName(cenMin, cenMax);
    TF1 *func = dynamic_cast<TF1 *>(f->Get(name.c_str()));
    if (!func) {
        std::cerr << "[Warn] TF1 " << name << " not found, fallback to BlastWave_H3L_0_10" << std::endl;
        func = dynamic_cast<TF1 *>(f->Get("BlastWave_H3L_0_10"));
    }
    if (!func) return nullptr;
    return std::unique_ptr<TF1>(static_cast<TF1 *>(func->Clone())) ;
}

struct TopoResult {
    std::unique_ptr<TH1D> hRaw;
    std::unique_ptr<TH1D> hCorr;
    std::unique_ptr<TH1D> hEff;
    std::unique_ptr<TH1D> hAbso;
    std::vector<std::unique_ptr<RooPlot>> frames;
    std::vector<std::unique_ptr<RooPlot>> framesMc;
    std::vector<std::unique_ptr<TCanvas>> canvases;
    std::vector<std::unique_ptr<TCanvas>> canvasesMc;
    std::vector<std::shared_ptr<RooRealVar>> massAxes;
};

struct EffCounts {
    uint64_t num{0};
    uint64_t den{0};
};

std::vector<EffCounts> ComputeMcEfficiencyAll(ROOT::RDF::RNode base, const std::vector<double> &ptEdges, const std::vector<std::string> &topoCuts, const std::string &basicSelection) {
    size_t nbins = ptEdges.size() > 1 ? ptEdges.size() - 1 : 0;
    std::vector<EffCounts> effs(nbins);
    std::vector<ROOT::RDF::RResultHandle> meanHandles;
    meanHandles.reserve(nbins);
    std::vector<ROOT::RDF::RResultPtr<double>> meanPtrs;
    meanPtrs.reserve(nbins);

    // First pass: compute mean per pT bin so they can be scheduled together.
    for (size_t ibin = 0; ibin < nbins; ++ibin) {
        std::string ptCutMc = Form("fAbsGenPt > %f && fAbsGenPt <= %f", ptEdges[ibin], ptEdges[ibin + 1]);
        ROOT::RDF::RNode nodeBin = base.Filter(ptCutMc);
        auto meanPtr = nodeBin.Mean("fNSigmaHe");
        meanPtrs.push_back(meanPtr);
        meanHandles.push_back(meanPtr);
    }
    ROOT::RDF::RunGraphs(meanHandles);

    std::vector<ROOT::RDF::RResultHandle> handles;
    handles.reserve(nbins * 2);
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> numPtrs;
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> denPtrs;
    numPtrs.reserve(nbins);
    denPtrs.reserve(nbins);

    for (size_t ibin = 0; ibin < nbins; ++ibin) {
        std::string ptCutMc = Form("fAbsGenPt > %f && fAbsGenPt <= %f", ptEdges[ibin], ptEdges[ibin + 1]);
        ROOT::RDF::RNode nodeBin = base.Filter(ptCutMc);
        const double mean = meanPtrs[ibin].GetValue();
        ROOT::RDF::RNode nodeCentered = nodeBin.Redefine("fNSigmaHe", [mean](float x) { return static_cast<double>(x) - mean; }, {"fNSigmaHe"});
        auto den = nodeBin.Filter("fIsSurvEvSel>0").Count();
        ROOT::RDF::RNode nodeNum = nodeCentered.Filter("fIsReco>0");
        if (!basicSelection.empty()) nodeNum = nodeNum.Filter(basicSelection);
        if (ibin < topoCuts.size() && !topoCuts[ibin].empty()) nodeNum = nodeNum.Filter(topoCuts[ibin]);
        auto num = nodeNum.Count();
        denPtrs.push_back(den);
        numPtrs.push_back(num);
        handles.push_back(den);
        handles.push_back(num);
    }

    ROOT::RDF::RunGraphs(handles);
    for (size_t ibin = 0; ibin < nbins; ++ibin) {
        effs[ibin].den = denPtrs[ibin].GetValue();
        effs[ibin].num = numPtrs[ibin].GetValue();
    }
    return effs;
}

std::shared_ptr<ROOT::RDataFrame> MakeSnapshotRdf(const std::string &path, const std::string &tree) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Snapshot not found: " + path);
    }
    return std::make_shared<ROOT::RDataFrame>(tree, path);
}

std::unique_ptr<TChain> MakeAo2dChain(const std::string &file, const std::string &tree) {
    auto chain = std::make_unique<TChain>(tree.c_str());
    TFile f(file.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Failed to open AO2D file: " + file);
    }
    bool added = false;
    TIter nextDir(f.GetListOfKeys());
    while (TObject *obj = nextDir()) {
        auto key = dynamic_cast<TKey *>(obj);
        if (!key) continue;
        if (std::string(key->GetClassName()) != "TDirectoryFile") continue;
        std::string path = file + "/" + key->GetName() + "/" + tree;
        chain->Add(path.c_str());
        added = true;
    }
    if (!added) {
        throw std::runtime_error("No matching trees added from AO2D: " + file);
    }
    return chain;
}

int ProcessTopologySpectrum(const char *cfgPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/topology_spectrum_V0s.json") {
    if (!cfgPath) {
        std::cerr << "Config path is null" << std::endl;
        return 1;
    }

    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);

    TopoConfig cfg = LoadTopoConfig(cfgPath);
    if (cfg.enableImplicitMT) ROOT::EnableImplicitMT();
    std::filesystem::create_directories(cfg.outputDir);

    if (cfg.snapshotDir.empty()) {
        throw std::runtime_error("snapshot_dir is required for topology spectrum extraction");
    }
    if (cfg.mcFileForAbsorption.empty()) {
        std::cerr << "[Warn] mc_file_for_absorption missing, absorption defaults to 1" << std::endl;
    }
    if (cfg.mcFileForAcceptance.empty()) {
        throw std::runtime_error("mc_file_for_acceptance is required for efficiency computation");
    }

    auto mcAccChain = MakeAo2dChain(cfg.mcFileForAcceptance, cfg.treeNameMc);
    ROOT::RDataFrame rdfMcAcc(*mcAccChain);
    auto mcAccBase = GeneralHelper::CorrectAndConvertRDF(rdfMcAcc, false, true, false);

    const double matterRatio = (cfg.isMatter == "both") ? 2.0 : 1.0;

    auto saveHistPdf = [](TH1 &h, const std::string &path, const std::string &opt = "HIST") {
        auto cTmp = std::make_unique<TCanvas>(Form("c_tmp_%s", h.GetName()), h.GetTitle(), 900, 700);
        h.Draw(opt.c_str());
        cTmp->SaveAs(path.c_str());
    };

    for (size_t icen = 0; icen + 1 < cfg.cenBins.size(); ++icen) {
        std::pair<double, double> cenRange{cfg.cenBins[icen], cfg.cenBins[icen + 1]};
        std::string centCut = Form("fCentralityFT0C>=%f && fCentralityFT0C<%f", cenRange.first, cenRange.second);
        std::vector<double> ptEdges = (!cfg.ptBinsByCen.empty() && icen < cfg.ptBinsByCen.size()) ? cfg.ptBinsByCen[icen] : cfg.ptBins;
        if (ptEdges.size() < 2) {
            std::cerr << "[Warn] Skip centrality bin due to empty pt bins" << std::endl;
            continue;
        }

        auto hAbso = BuildAbsorption(cfg, ptEdges);
        double nEvents = GetNEvents(cfg, cenRange);

        auto reweightFunc = LoadReweightFunc(cfg, cenRange.first, cenRange.second);
        ROOT::RDF::RNode effBase(mcAccBase);
        if (reweightFunc) {
            effBase = GeneralHelper::ReWeightSpectrum(effBase, reweightFunc.get(), "fAbsGenPt");
        }
        auto effPrecomputed = ComputeMcEfficiencyAll(effBase, ptEdges, (icen < cfg.dataSelectionTopo.size()) ? cfg.dataSelectionTopo[icen] : std::vector<std::string>{}, cfg.basicSelectionDataForMcEff);

        std::vector<double> edges = ptEdges;
        TH1D hRaw("h_raw_counts", ";p_{T};N_{raw}", static_cast<int>(edges.size() - 1), edges.data());
        TH1D hCorr("h_corrected_counts", ";p_{T};#frac{1}{N_{ev}} dN/dy dp_{T}", static_cast<int>(edges.size() - 1), edges.data());
        TH1D hEff("h_efficiency", ";p_{T};#epsilon_{reco}", static_cast<int>(edges.size() - 1), edges.data());
        TH1D hAbsoHist("h_absorption", ";p_{T};#epsilon_{abso}", static_cast<int>(edges.size() - 1), edges.data());
        hRaw.SetDirectory(nullptr);
        hCorr.SetDirectory(nullptr);
        hEff.SetDirectory(nullptr);
        hAbsoHist.SetDirectory(nullptr);
        hRaw.SetStats(false);
        hCorr.SetStats(false);
        hEff.SetStats(false);
        hAbsoHist.SetStats(false);

        TopoResult res;

        for (size_t ibin = 0; ibin + 1 < ptEdges.size(); ++ibin) {
            double ptMin = ptEdges[ibin];
            double ptMax = ptEdges[ibin + 1];
            BinKey key{cenRange.first, cenRange.second, ptMin, ptMax, -1.0, -1.0};
            std::string label = MakeLabel(key);
            std::string dataPath = cfg.snapshotDir + "/data_" + label + ".root";
            std::string mcPath = cfg.snapshotDir + "/mc_" + label + ".root";
            auto dfData = MakeSnapshotRdf(dataPath, cfg.treeNameData);
            auto dfMc = MakeSnapshotRdf(mcPath, cfg.treeNameMc);

            std::string topoCut;
            if (icen < cfg.dataSelectionTopo.size() && ibin < cfg.dataSelectionTopo[icen].size()) {
                topoCut = cfg.dataSelectionTopo[icen][ibin];
            } else if (icen < cfg.dataSelectionTopo.size() && !cfg.dataSelectionTopo[icen].empty()) {
                topoCut = cfg.dataSelectionTopo[icen].back();
            }
            std::string ptCutData = Form("fPt > %f && fPt <= %f", ptMin, ptMax);
            std::string ptCutMc = Form("fAbsGenPt > %f && fAbsGenPt <= %f", ptMin, ptMax);
            std::string massCut = Form("fMassH3L>%f && fMassH3L<%f", cfg.massMin, cfg.massMax);

            ROOT::RDF::RNode nodeData(*dfData);
            ROOT::RDF::RNode nodeMc(*dfMc);
            if (!topoCut.empty()) {
                nodeData = nodeData.Filter(topoCut);
                //nodeMc = nodeMc.Filter(topoCut);
            }
            nodeData = nodeData.Filter(ptCutData);
            nodeMc = nodeMc.Filter(ptCutMc);
            if (reweightFunc) {
                nodeMc = GeneralHelper::ReWeightSpectrum(nodeMc, reweightFunc.get(), "fAbsGenPt");
            }

            auto dataMass = nodeData.Take<double>("fMassH3L");
            auto mcMass = nodeMc.Filter(massCut).Take<double>("fMassH3L");

            EffCounts effCounts;
            if (ibin < effPrecomputed.size()) {
                effCounts = effPrecomputed[ibin];
            }

            if (effCounts.den == 0 || dataMass->empty() || mcMass->empty()) {
                hRaw.SetBinContent(static_cast<int>(ibin + 1), 0.0);
                hCorr.SetBinContent(static_cast<int>(ibin + 1), 0.0);
                hEff.SetBinContent(static_cast<int>(ibin + 1), 0.0);
                hAbsoHist.SetBinContent(static_cast<int>(ibin + 1), hAbso ? hAbso->GetBinContent(static_cast<int>(ibin + 1)) : 1.0);
                continue;
            }

            const double eff = static_cast<double>(effCounts.num) / static_cast<double>(effCounts.den);
            const double bw = ptMax - ptMin;
            const double absoVal = hAbso ? hAbso->GetBinContent(static_cast<int>(ibin + 1)) : 1.0;

            Config cfgFit;
            cfgFit.massMin = cfg.massMin;
            cfgFit.massMax = cfg.massMax;
            cfgFit.sigmaRangeMcToData = {1.0, 1.5};
            FitResult fit = SpectrumCalculator(cfgFit).FitMassPublic(*dataMass, *mcMass, cfg.bkgFunc, cfg.sigFunc);

            double corr = 0.0;
            double corrErr = 0.0;
            if (bw > 0 && eff > 0 && absoVal > 0) {
                corr = fit.signal / eff / absoVal / bw / nEvents / cfg.branchingRatio / cfg.deltaRap;
                corrErr = fit.signalErr / eff / absoVal / bw / nEvents / cfg.branchingRatio / cfg.deltaRap;
                corr /= matterRatio;
                corrErr /= matterRatio;
            }

            hRaw.SetBinContent(static_cast<int>(ibin + 1), fit.signal);
            hRaw.SetBinError(static_cast<int>(ibin + 1), fit.signalErr);
            hCorr.SetBinContent(static_cast<int>(ibin + 1), corr);
            hCorr.SetBinError(static_cast<int>(ibin + 1), corrErr);
            hEff.SetBinContent(static_cast<int>(ibin + 1), eff);
            hAbsoHist.SetBinContent(static_cast<int>(ibin + 1), absoVal);

            res.frames.push_back(std::move(fit.frame));
            res.framesMc.push_back(std::move(fit.frameMc));
            res.massAxes.push_back(std::move(fit.massAxis));
            const std::string suffix = label;
            if (res.frames.back()) {
                auto c = std::make_unique<TCanvas>(Form("c_data_fit_%s", suffix.c_str()), Form("c_data_fit_%s", suffix.c_str()), 800, 600);
                res.frames.back()->SetName(Form("frame_data_%s", suffix.c_str()));
                c->cd();
                res.frames.back()->Draw();
                res.canvases.push_back(std::move(c));
            }
            if (res.framesMc.back()) {
                auto cMc = std::make_unique<TCanvas>(Form("c_mc_fit_%s", suffix.c_str()), Form("c_mc_fit_%s", suffix.c_str()), 800, 600);
                res.framesMc.back()->SetName(Form("frame_mc_%s", suffix.c_str()));
                cMc->cd();
                res.framesMc.back()->Draw();
                res.canvasesMc.push_back(std::move(cMc));
            }
        }

        std::string cenDirName = Form("cen%d-%d", static_cast<int>(cenRange.first), static_cast<int>(cenRange.second));
        std::filesystem::create_directories(cfg.outputDir + "/" + cenDirName);
        std::string outPath = cfg.outputDir + "/" + cenDirName + "/pt_analysis_pbpb.root";
        TFile fout(outPath.c_str(), "RECREATE");
        TDirectory *stdDir = fout.mkdir("std");
        TDirectory *targetDir = stdDir ? stdDir : static_cast<TDirectory*>(&fout);
        targetDir->cd();
        hRaw.Write();
        hCorr.Write();
        hEff.Write();
        hAbsoHist.Write();
        for (const auto &f : res.frames) {
            if (f) targetDir->WriteObject(f.get(), f->GetName());
        }
        for (const auto &f : res.framesMc) {
            if (f) targetDir->WriteObject(f.get(), f->GetName());
        }
        for (const auto &c : res.canvases) {
            if (c) targetDir->WriteObject(c.get(), c->GetName());
        }
        for (const auto &c : res.canvasesMc) {
            if (c) targetDir->WriteObject(c.get(), c->GetName());
        }

        if (cfg.do_QA_afterward) {
            std::string basePdfDir = cfg.outputDir + "/" + cenDirName;
            saveHistPdf(hRaw, basePdfDir + "/h_raw_topo.pdf", "E1");
            saveHistPdf(hCorr, basePdfDir + "/h_corr_topo.pdf", "E1");
            saveHistPdf(hEff, basePdfDir + "/h_eff_topo.pdf");
            saveHistPdf(hAbsoHist, basePdfDir + "/h_abso_topo.pdf");
            for (const auto &c : res.canvases) if (c) c->SaveAs((basePdfDir + "/" + c->GetName() + ".pdf").c_str());
            for (const auto &c : res.canvasesMc) if (c) c->SaveAs((basePdfDir + "/" + c->GetName() + ".pdf").c_str());
        }

        std::cout << "Saved " << outPath << "\n";
    }

    return 0;
}
