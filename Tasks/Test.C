#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>

#include <ROOT/RDataFrame.hxx>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooPlot.h>
#include <RooArgSet.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "../Tools/GeneralHelper.hpp"
#include "../Tools/AcceptanceHelper.h"
using namespace AcceptanceHelper;
using namespace GeneralHelper;
using namespace std;

void Test() {
    unsigned int preferred = std::thread::hardware_concurrency();
    if (preferred == 0) {
        preferred = 4;
    }
    const unsigned int nThreads = std::clamp(preferred, 2u, 12u);
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(nThreads);
        std::cout << "[Test] Enabled ROOT implicit MT with " << nThreads << " threads\n";
    }

    std::string filepath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root";
    TChain chain("O2mchypcands");
    fillChainFromAO2D(chain, TFile::Open(filepath.c_str(), "READ"));
    ROOT::RDataFrame genRDF(chain);
    auto genRDFConverted = CorrectAndConvertRDF(genRDF, false, true, false);
    auto cloumnNames = genRDFConverted.GetColumnNames();
    cout << "Columns after conversion:\n";
    for (const auto& name : cloumnNames) {
        cout << "  " << name << "\n";
    }
    double mean = 3.0;
    double sigma = 1.0;
    TF1* distribution = new TF1("distribution", "gaus", 0.0, 10.0);
    distribution->SetParameters(1.0, mean, sigma);
    // normalize to unit area on [0,10]
    double area = distribution->Integral(0.0, 10.0);
    if (area > 0) distribution->SetParameter(0, distribution->GetParameter(0)/area);
    auto rdfReweighted = ReWeightSpectrum(genRDFConverted, distribution, "fAbsGenPt");
    auto hGentPt = rdfReweighted.Histo1D({"fGenPt", "Generated p_{T};p_{T} (GeV/c);Counts", 100, -10, 10}, "fGenPt").GetValue();
    auto hGentCt = rdfReweighted.Histo1D({"fGenCt", "Generated ct;ct (cm);Counts", 600, -30, 30}, "fGenCt").GetValue();
    TCanvas* c1 = new TCanvas("c1", "Generated C_{t}", 800, 600);
    hGentPt.Draw("hist");
    c1->SaveAs("generated_pt_reweighted.png");
    auto mass = rdfReweighted.Mean("fMassH3L").GetValue();
    auto massarr = rdfReweighted.Take<double>("fMassH3L");
    cout << "Mean mass of 3He from reweighted RDF: " << mass << " GeV/c^2\n";
    AcceptanceResult res = AcceptanceHelper::ComputeAcceptanceFlexible(
        rdfReweighted,
        {2, 3.5, 5, 8}, // ptBins
        {},                             // ctBins1D
        { {1, 3, 6, 9, 12, 18, 30},
          {1, 3, 6, 9, 12, 18, 25},
          {1, 3, 6, 9, 15, 25}},                             // ctBinsPerPt
        {},                             // centBins1D
        {}                              // ptBinsPerCent
    );
    TCanvas* c2 = new TCanvas("c2", "Acceptance vs ct per pt bin", 800, 600);
    res.acc_ct_per_pt_antimatter[0]->SetLineColor(kRed);
    res.acc_ct_per_pt_antimatter[0]->SetMarkerColor(kRed);
    res.acc_ct_per_pt_antimatter[0]->SetMarkerStyle(20);
    res.acc_ct_per_pt_antimatter[0]->SetTitle("Acceptance vs ct in 2-3.5 GeV/c pt bin;ct (cm);Acceptance");
    res.acc_ct_per_pt_antimatter[0]->Draw("PE");
    c2->SaveAs("acceptance_ct_ptbin1.png");
    TCanvas* c3 = new TCanvas("c3", "Counts evsel vs reco", 800, 600);
    res.evsel_ct_per_pt_antimatter[0]->SetLineColor(kBlue);
    res.evsel_ct_per_pt_antimatter[0]->SetMarkerColor(kBlue);
    res.evsel_ct_per_pt_antimatter[0]->SetMarkerStyle(20);
    res.evsel_ct_per_pt_antimatter[0]->SetTitle("Counts evsel vs reco in 2-3.5 GeV/c pt bin;ct (cm);Counts");
    res.evsel_ct_per_pt_antimatter[0]->Draw("SAME PE0");
    res.reco_ct_per_pt_antimatter[0]->SetLineColor(kGreen+2);
    res.reco_ct_per_pt_antimatter[0]->SetMarkerColor(kGreen+2);
    res.reco_ct_per_pt_antimatter[0]->SetMarkerStyle(21);
    res.reco_ct_per_pt_antimatter[0]->Draw("SAME PE0 ");
    c3->SaveAs("counts_evsel_reco_ptbin1.png");
    TCanvas* c4 = new TCanvas("c4", "Acceptance vs ct in 3.5-5 GeV/c pt bin;ct (cm);Acceptance", 800, 600);
    res.acc_ct_per_pt_antimatter[1]->SetLineColor(kRed);
    res.acc_ct_per_pt_antimatter[1]->SetMarkerColor(kRed);
    res.acc_ct_per_pt_antimatter[1]->SetMarkerStyle(20);
    res.acc_ct_per_pt_antimatter[1]->SetTitle("Acceptance vs ct in 3.5-5 GeV/c pt bin;ct (cm);Acceptance");
    res.acc_ct_per_pt_antimatter[1]->Draw("PE");
    c4->SaveAs("acceptance_ct_ptbin2.png");
    TCanvas* c5  = new TCanvas("c5", "Acceptance vs ct in 5-8 GeV/c pt bin;ct (cm);Acceptance", 800, 600);
    res.acc_ct_per_pt_antimatter[2]->SetLineColor(kRed);
    res.acc_ct_per_pt_antimatter[2]->SetMarkerColor(kRed);
    res.acc_ct_per_pt_antimatter[2]->SetMarkerStyle(20);
    res.acc_ct_per_pt_antimatter[2]->SetTitle("Acceptance vs ct in 5-8 GeV/c pt bin;ct (cm);Acceptance");
    res.acc_ct_per_pt_antimatter[2]->Draw("PE");
    c5->SaveAs("acceptance_ct_ptbin3.png");


    // filter the RDF by model_output and build a per-slot RooDataSet, then merge and plot
    TFile *SnapshotFile = TFile::Open("/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID/data_pt_2_3_ct_18_30.root", "READ");
    ROOT::RDataFrame rdfSnapshot("O2hypcands", SnapshotFile);
    auto rdfForRoo = rdfSnapshot.Filter("model_output > 4.747", "select high model score");

    std::vector<RooDataSet*> slotDatasets(nThreads, nullptr);
    std::vector<RooRealVar*> slotMassVar(nThreads, nullptr);

    rdfForRoo.ForeachSlot(
        [&](unsigned slot, double massVal) {
            if (!slotMassVar[slot]) {
                // create a slot-local RooRealVar and RooDataSet (use same variable name "mass" for all slots)
                // set a sensible range around the expected mass (adjust if needed)
                double minRange = massVal - 0.5;
                double maxRange = massVal + 0.5;
                slotMassVar[slot] = new RooRealVar("mass", "mass (GeV/c^{2})", minRange, maxRange);
                std::string dsName = std::string("slot_ds_") + std::to_string(slot);
                slotDatasets[slot] = new RooDataSet(dsName.c_str(), "slot dataset", RooArgSet(*slotMassVar[slot]));
            }
            slotMassVar[slot]->setVal(massVal);
            slotDatasets[slot]->add(RooArgSet(*slotMassVar[slot]));
        },
        {"fMassH3L"}
    );

    // merge slot datasets into a single RooDataSet
    // create a master RooRealVar with the full range covering all slot vars
    double globalMin = 1e6, globalMax = -1e6;
    for (unsigned i = 0; i < slotMassVar.size(); ++i) {
        if (!slotMassVar[i]) continue;
        globalMin = std::min(globalMin, slotMassVar[i]->getMin());
        globalMax = std::max(globalMax, slotMassVar[i]->getMax());
    }
    if (globalMin > globalMax) { // no entries
        std::cout << "[Test] No entries passed the model_output filter, skipping RooDataset creation.\n";
    } else {
        RooRealVar massVar("mass", "mass (GeV/c^{2})", 2.96, 3.04);
        RooDataSet* merged = new RooDataSet("roodataset", "RooDataset fMassH3L (model_output>4.747)", RooArgSet(massVar));
        for (auto ds : slotDatasets) {
            if (!ds) continue;
            merged->append(*ds);
        }

        // plot and save to PDF
        TCanvas* c_roo = new TCanvas("c_roo", "RooDataset mass", 800, 600);
        RooPlot* frame = massVar.frame(50);
        merged->plotOn(frame);
        frame->Draw();
        c_roo->SaveAs("roodataset_mass.pdf");
    }

}