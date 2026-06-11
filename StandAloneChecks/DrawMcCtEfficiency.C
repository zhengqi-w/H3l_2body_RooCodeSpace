// DrawMcCtEfficiency.C
// Quick check: compute and plot MC reconstruction efficiency in ct bins.
// Usage (ROOT): root -l -b -q 'DrawMcCtEfficiency.C()'
// Optional args:
//   DrawMcCtEfficiency({0,2,4,6,8,10},
//                      "/path/to/AO2D.root",
//                      "O2mchypcands",
//                      "basic_sel_expr",
//                      "/path/to/reweight.root",
//                      "both",
//                      false,
//                      "extra_cut",
//                      "mc_eff_ct.pdf")

#include <ROOT/RDataFrame.hxx>
#include <TChain.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TKey.h>
#include <TString.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cmath>

#include "../Tools/GeneralHelper.hpp"

std::unique_ptr<TF1> LoadReweightFunc(const std::string &reweightPtFile) {
    if (reweightPtFile.empty()) return nullptr;
    auto f = std::unique_ptr<TFile>(TFile::Open(reweightPtFile.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[Warn] Failed to open reweight file: " << reweightPtFile << std::endl;
        return nullptr;
    }
    TF1 *func = dynamic_cast<TF1 *>(f->Get("BlastWave_H3L_0_10"));
    if (!func) return nullptr;
    return std::unique_ptr<TF1>(static_cast<TF1 *>(func->Clone()));
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

TH1D ComputeMcEffVsCt(const std::string &mcFileForAcceptance,
                      const std::string &treeNameMc,
                      const std::vector<double> &ctEdges,
                      const std::string &basicSelectionDataForMcEff,
                      const std::string &extraCut,
                      const std::string &reweightPtFile,
                      const std::string &isMatter) {
    if (ctEdges.size() < 2) {
        throw std::runtime_error("Need at least two ct edges");
    }
    auto chain = MakeAo2dChain(mcFileForAcceptance, treeNameMc);
    ROOT::RDataFrame rdf(*chain);
    auto baseConverted = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode base(baseConverted);

    if (isMatter == "matter") base = base.Filter("fIsMatter>0");
    else if (isMatter == "antimatter") base = base.Filter("fIsMatter<1");

    auto reweightFunc = LoadReweightFunc(reweightPtFile);
    if (reweightFunc) {
        base = GeneralHelper::ReWeightSpectrum(base, reweightFunc.get(), "fAbsGenPt");
    }

    TH1D hEff("h_mc_eff_ct", ";ct (cm);#epsilon_{reco}", static_cast<int>(ctEdges.size()) - 1, ctEdges.data());
    hEff.SetDirectory(nullptr);
    hEff.SetStats(false);

    std::vector<ROOT::RDF::RResultHandle> handles;
    handles.reserve(ctEdges.size() * 4);
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> numPtrs;
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> denPtrs;
    numPtrs.reserve(ctEdges.size());
    denPtrs.reserve(ctEdges.size());
    std::vector<ROOT::RDF::RResultPtr<TH1D>> hBeforePtrs;
    std::vector<ROOT::RDF::RResultPtr<TH1D>> hAfterPtrs;
    hBeforePtrs.reserve(ctEdges.size());
    hAfterPtrs.reserve(ctEdges.size());

    for (size_t i = 0; i + 1 < ctEdges.size(); ++i) {
        std::string ctCut = Form("fGenCt>=%f && fGenCt<%f", ctEdges[i], ctEdges[i + 1]);
        ROOT::RDF::RNode nodeBin = base.Filter(ctCut);
        auto den = nodeBin.Filter("fIsSurvEvSel>0").Count();
        ROOT::RDF::RNode nodeRecoBase = nodeBin.Filter("fIsReco>0");
        if (!basicSelectionDataForMcEff.empty()) nodeRecoBase = nodeRecoBase.Filter(basicSelectionDataForMcEff);
        ROOT::RDF::RNode nodeBefore = nodeRecoBase;
        ROOT::RDF::RNode nodeAfter = nodeRecoBase;
        if (!extraCut.empty()) nodeAfter = nodeAfter.Filter(extraCut);
        auto num = nodeAfter.Count();

        auto hBeforePtr = nodeBefore.Histo1D({Form("hGenDecRad_before_%zu", i), ";fGenDecRad (cm);Counts", 300, 0.0, 30.0}, "fGenDecRad");
        auto hAfterPtr = nodeAfter.Histo1D({Form("hGenDecRad_after_%zu", i), ";fGenDecRad (cm);Counts", 300, 0.0, 30.0}, "fGenDecRad");

        denPtrs.push_back(den);
        numPtrs.push_back(num);
        hBeforePtrs.push_back(hBeforePtr);
        hAfterPtrs.push_back(hAfterPtr);
        handles.push_back(den);
        handles.push_back(num);
        handles.push_back(hBeforePtr);
        handles.push_back(hAfterPtr);
    }

    ROOT::RDF::RunGraphs(handles);
    for (size_t i = 0; i + 1 < ctEdges.size(); ++i) {
        const double den = static_cast<double>(denPtrs[i].GetValue());
        const double num = static_cast<double>(numPtrs[i].GetValue());
        const double eff = (den > 0.0) ? num / den : 0.0;
        const double err = (den > 0.0) ? std::sqrt(eff * (1.0 - eff) / den) : 0.0;
        hEff.SetBinContent(static_cast<int>(i + 1), eff);
        hEff.SetBinError(static_cast<int>(i + 1), err);
        std::cout << Form("ct %.3f-%.3f: eff = %.4f +/- %.4f (num=%g, den=%g)\n",
                          ctEdges[i], ctEdges[i + 1], eff, err, num, den);

        TH1D *hBefore = hBeforePtrs[i].GetPtr();
        TH1D *hAfter = hAfterPtrs[i].GetPtr();
        if (hBefore && hAfter) {
            hBefore->SetStats(false);
            hAfter->SetStats(false);
            TCanvas cRad(Form("c_genDecrad_ctbin%zu", i), "fGenDecRad", 800, 600);
            cRad.SetLeftMargin(0.12);
            cRad.SetBottomMargin(0.12);
            hBefore->SetLineColor(kAzure+2);
            hBefore->SetMarkerColor(kAzure+2);
            hBefore->SetMarkerStyle(20);
            hBefore->Draw("HIST");
            hAfter->SetLineColor(kRed+1);
            hAfter->SetMarkerColor(kRed+1);
            hAfter->SetMarkerStyle(24);
            hAfter->Draw("HIST SAME");
            auto leg = new TLegend(0.58, 0.75, 0.90, 0.90);
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->AddEntry(hBefore, "before extraCut", "l");
            leg->AddEntry(hAfter,  "after extraCut",  "l");
            leg->Draw();
            cRad.SaveAs(Form("fGenDecRad_ct_%.3f_%.3f.pdf", ctEdges[i], ctEdges[i + 1]));
        }
    }

    return hEff;
}

int DrawMcCtEfficiency(const std::vector<double> &ctEdges = {0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 3},
                       const char *mcFileForAcceptance = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root",
                       const char *treeNameMc = "O2mchypcands",
                       const char *basicSelectionDataForMcEff = "fTPCsignalPi<1000 && fCosPA>0.99 && fAvgClusterSizeHe > 5",
                       const char *reweightPtFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root",
                       const char *isMatter = "both",
                       bool enableImplicitMT = false,
                       const char *extraCut = "fGenDecRad < 2.1",
                       const char *outpath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/MCEfficiency") {
    try {
        if (!mcFileForAcceptance || std::string(mcFileForAcceptance).empty()) {
            throw std::runtime_error("mc_file_for_acceptance is required");
        }
        if (enableImplicitMT) ROOT::EnableImplicitMT();

        auto edges = ctEdges;
        if (edges.size() < 2) edges = {0.0, 10.0};
        std::sort(edges.begin(), edges.end());

        auto hEff = ComputeMcEffVsCt(mcFileForAcceptance,
                         treeNameMc ? treeNameMc : "O2mchypcands",
                         edges,
                         basicSelectionDataForMcEff ? basicSelectionDataForMcEff : "",
                         extraCut ? extraCut : "",
                         reweightPtFile ? reweightPtFile : "",
                         isMatter ? isMatter : "both");

        TCanvas c("c_mc_eff_ct", "MC efficiency vs ct", 900, 700);
        c.SetLeftMargin(0.12);
        c.SetBottomMargin(0.12);
        c.SetGridx();
        c.SetGridy();
        hEff.SetMarkerStyle(20);
        hEff.SetMarkerSize(1.1);
        hEff.SetLineColor(kAzure+2);
        hEff.SetMarkerColor(kAzure+2);
        hEff.Draw("E1");

        auto leg = new TLegend(0.58, 0.78, 0.90, 0.90);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->AddEntry(&hEff, "MC reco eff", "lep");
        if (extraCut && std::string(extraCut).size() > 0) leg->AddEntry((TObject*)nullptr, Form("Extra: %s", extraCut), "");
        leg->Draw();

        std::string outDir = outpath ? std::string(outpath) : std::string(".");
        if (!outDir.empty()) std::filesystem::create_directories(outDir);
        std::string effPdf = outDir + "/mc_eff_ct.pdf";
        c.SaveAs(effPdf.c_str());
        std::cout << "Saved " << effPdf << std::endl;

        // Move per-bin rad plots into outDir
        for (size_t i = 0; i + 1 < edges.size(); ++i) {
            std::string src = Form("fGenDecRad_ct_%.3f_%.3f.pdf", edges[i], edges[i + 1]);
            if (gSystem->AccessPathName(src.c_str())) continue;
            std::string dst = outDir + "/" + src;
            gSystem->Rename(src.c_str(), dst.c_str());
        }
    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
