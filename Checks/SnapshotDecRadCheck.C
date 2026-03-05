#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TH1D.h>
#include <TString.h>
#include <TPaveText.h>
#include <RooAddPdf.h>
#include <RooAbsReal.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataSet.h>
#include <RooPlot.h>
#include <RooRealVar.h>
#include <TFile.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

std::string Clean(double v) {
    std::ostringstream ss;
    ss << std::setprecision(12) << v;
    std::string s = ss.str();
    auto pos = s.find('.');
    if (pos != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

void SnapshotDecRadCheck(const std::string &snapshotDir = "../../SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID",
                         const std::string &mcfile = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root",
                         const std::string &wpFile = "../../Outputs/MLProcess/LHC23_PbPb_pass5_CustomV0s_HadronPID/WorkingPoint/WorkingPoint_SpectrumTest.txt",
                         const std::string &treeName = "O2hypcands",
                         const std::string &treeNameMc = "O2mchypcands",
                         const std::string &outDir = "../../Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/Checks/DecRad",
                         const std::string &isMatter = "both", // "matter", "antimatter", "both"
                         const std::vector<double> &ctBins = {0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 3}) {
    using namespace GeneralHelper;
    SetDefaultStyle();
    EnableImplicitMTWithPreferredThreads();
    std::filesystem::create_directories(outDir);

    const double massMin = 2.96;
    const double massMax = 3.04;
    const int massBins = 80;

    std::vector<double> massAllNoCut;
    std::vector<double> massAllDec;
    std::vector<double> massAllMcDec;
    std::vector<std::pair<double, double>> dataMassCt;
    std::vector<std::pair<double, double>> dataMassDecRad; // (mass, decRad) before decRad cut
    std::vector<std::pair<double, double>> mcMassDecRad;   // (mass, decRad) raw MC with fIsReco>0
    bool haveAnyData = false;
    bool haveAnyMc = false;
    bool warnedCtMissing = false;
    bool warnedIsMatterMissing = false;
    bool warnedMcIsMatterMissing = false;
    bool warnedMcNoModel = false;

    std::string isMatterLower = isMatter;
    std::transform(isMatterLower.begin(), isMatterLower.end(), isMatterLower.begin(), ::tolower);

    std::vector<double> cenBins{0, 10, 30, 50, 80};
    std::vector<std::vector<double>> ptBinsByCen{
        {2, 3, 3.5, 4, 4.5, 5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 5, 8}
    };

    auto hasTree = [](const std::string &filePath, const std::string &tree) {
        std::unique_ptr<TFile> f(TFile::Open(filePath.c_str(), "READ"));
        if (!f || f->IsZombie()) return false;
        return f->GetListOfKeys()->Contains(tree.c_str());
    };

    for (size_t ic = 0; ic + 1 < cenBins.size(); ++ic) {
        double cenMin = cenBins[ic];
        double cenMax = cenBins[ic + 1];
        const auto &pts = ptBinsByCen.at(ic);
        for (size_t ip = 0; ip + 1 < pts.size(); ++ip) {
            double ptMin = pts[ip];
            double ptMax = pts[ip + 1];

            auto wp = GetWpForCenPt(wpFile, cenMin, cenMax, ptMin, ptMax);
            if (!wp.found) {
                std::cerr << "[Skip] WP not found for cen " << cenMin << "-" << cenMax
                          << " pt " << ptMin << "-" << ptMax << "\n";
                continue;
            }

            std::string filePath = snapshotDir + "/data_cen_" + Clean(cenMin) + "_" + Clean(cenMax) +
                                   "_pt_" + Clean(ptMin) + "_" + Clean(ptMax) + ".root";
            if (!std::filesystem::exists(filePath)) {
                std::cerr << "[Skip] Snapshot file not found: " << filePath << "\n";
                continue;
            }
            if (!hasTree(filePath, treeName)) {
                std::cerr << "[Skip] Tree " << treeName << " not found in " << filePath << "\n";
                continue;
            }

            ROOT::RDataFrame df(treeName, filePath);
            auto cols = df.GetColumnNames();
            bool hasDecRad = std::find(cols.begin(), cols.end(), std::string("fDecRad")) != cols.end();
            bool hasX = std::find(cols.begin(), cols.end(), std::string("fXDecVtx")) != cols.end();
            bool hasY = std::find(cols.begin(), cols.end(), std::string("fYDecVtx")) != cols.end();
            if (!hasDecRad && (!hasX || !hasY)) {
                std::cerr << "[Skip] fDecRad and decay-vertex coords missing in " << filePath << "\n";
                continue;
            }
            auto dfWithDecRad = hasDecRad ? df
                                          : df.Define("fDecRad", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx)");

            ROOT::RDF::RNode dfMatter = dfWithDecRad;
            bool hasIsMatter = std::find(cols.begin(), cols.end(), std::string("fIsMatter")) != cols.end();
            if (isMatterLower == "matter") {
                if (hasIsMatter) {
                    dfMatter = dfMatter.Filter("fIsMatter > 0", "matter only");
                } else if (!warnedIsMatterMissing) {
                    std::cerr << "[Warn] fIsMatter missing in data, skipping matter filter.\n";
                    warnedIsMatterMissing = true;
                }
            } else if (isMatterLower == "antimatter") {
                if (hasIsMatter) {
                    dfMatter = dfMatter.Filter("fIsMatter <= 0", "antimatter only");
                } else if (!warnedIsMatterMissing) {
                    std::cerr << "[Warn] fIsMatter missing in data, skipping antimatter filter.\n";
                    warnedIsMatterMissing = true;
                }
            }

            auto dfWp = dfMatter.Filter([score = wp.score](float m) { return m > score; }, {"model_output"});
            auto massNoCut = dfWp.Take<double>("fMassH3L");
            auto decRadNoCut = dfWp.Take<float>("fDecRad");
            massAllNoCut.insert(massAllNoCut.end(), massNoCut->begin(), massNoCut->end());
            for (size_t i = 0; i < massNoCut->size() && i < decRadNoCut->size(); ++i) {
                dataMassDecRad.emplace_back(massNoCut->at(i), static_cast<double>(decRadNoCut->at(i)));
            }

            auto dfDec = dfWp.Filter("fDecRad < 2.1"); // here 
            auto massDec = dfDec.Take<double>("fMassH3L");
            massAllDec.insert(massAllDec.end(), massDec->begin(), massDec->end());
            bool hasCt = std::find(cols.begin(), cols.end(), std::string("fCt")) != cols.end();
            if (hasCt) {
                auto ctDec = dfDec.Take<double>("fCt");
                for (size_t irow = 0; irow < massDec->size(); ++irow) {
                    dataMassCt.emplace_back(massDec->at(irow), ctDec->at(irow));
                }
            } else if (!warnedCtMissing) {
                std::cerr << "[Warn] fCt column missing in data snapshots, skip ct fill.\n";
                warnedCtMissing = true;
            }
            haveAnyData = true;

        }
    }

    // Load raw MC once (no snapshots), convert and keep only reconstructed candidates
    try {
        TChain mcChain(treeNameMc.c_str());
        auto mcFilePtr = std::unique_ptr<TFile>(TFile::Open(mcfile.c_str(), "READ"));
        if (mcFilePtr && !mcFilePtr->IsZombie()) {
            fillChainFromAO2D(mcChain, mcFilePtr.get());
            ROOT::RDataFrame mcRdf(mcChain);
            auto mcReady = CorrectAndConvertRDF(mcRdf, false, true, false).Filter("fIsReco > 0", "MC reco only");

            auto mcCols = mcReady.GetColumnNames();
            bool hasGenPt = std::find(mcCols.begin(), mcCols.end(), std::string("fGenPt")) != mcCols.end();
            ROOT::RDF::RNode mcSelected = mcReady;
            if (isMatterLower == "matter") {
                if (hasGenPt) {
                    mcSelected = mcSelected.Filter("fGenPt > 0", "MC matter only");
                } else if (!warnedMcIsMatterMissing) {
                    std::cerr << "[Warn] fGenPt missing in MC, skipping matter filter.\n";
                    warnedMcIsMatterMissing = true;
                }
            } else if (isMatterLower == "antimatter") {
                if (hasGenPt) {
                    mcSelected = mcSelected.Filter("fGenPt < 0", "MC antimatter only");
                } else if (!warnedMcIsMatterMissing) {
                    std::cerr << "[Warn] fGenPt missing in MC, skipping antimatter filter.\n";
                    warnedMcIsMatterMissing = true;
                }
            }

            auto mcMass = mcSelected.Take<double>("fMassH3L");
            auto mcDecRadCol = mcSelected.Take<float>("fDecRad");
            massAllMcDec.insert(massAllMcDec.end(), mcMass->begin(), mcMass->end());
            for (size_t i = 0; i < mcMass->size() && i < mcDecRadCol->size(); ++i) {
                mcMassDecRad.emplace_back(mcMass->at(i), static_cast<double>(mcDecRadCol->at(i)));
            }
            haveAnyMc = true;
        } else {
            std::cerr << "[Error] Cannot open MC file: " << mcfile << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "[Error] Exception while loading MC: " << e.what() << "\n";
    }

    if (!haveAnyData) {
        std::cerr << "[Error] No data snapshots processed, abort.\n";
        return;
    }

    TH1D totalNoCut("h_total_noCut", "fMassH3L;fMassH3L (GeV/c^{2});Entries", massBins, massMin, massMax);
    TH1D totalDec("h_total_dec", "fMassH3L;fMassH3L (GeV/c^{2});Entries", massBins, massMin, massMax);
    totalNoCut.SetDirectory(nullptr);
    totalDec.SetDirectory(nullptr);
    for (double v : massAllNoCut) totalNoCut.Fill(v);
    for (double v : massAllDec) totalDec.Fill(v);

    TCanvas cTotal("c_total", "Combined mass", 800, 600);
    totalNoCut.SetLineColor(kBlue + 2);
    totalNoCut.SetLineWidth(2);
    totalNoCut.Draw("hist");
    totalDec.SetLineColor(kRed + 1);
    totalDec.SetLineWidth(2);
    totalDec.Draw("hist same");

    TLegend legTot(0.60, 0.70, 0.88, 0.88);
    legTot.SetBorderSize(0);
    legTot.AddEntry(&totalNoCut, "model_output > WP (all bins)", "l");
    legTot.AddEntry(&totalDec, "WP + fDecRad < 2.1 (all bins)", "l");
    legTot.Draw();

    TLatex latTot;
    latTot.SetNDC();
    latTot.SetTextSize(0.035);
    latTot.DrawLatex(0.15, 0.85, "Combined over all WP bins");

    std::filesystem::create_directories(outDir);
    std::string outTot = outDir + "/mass_all_bins.pdf";
    cTotal.SaveAs(outTot.c_str());

    if (!haveAnyMc || massAllMcDec.empty()) {
        std::cerr << "[Error] No MC snapshots processed, cannot perform tail-constrained fit.\n";
        return;
    }
    if (massAllDec.empty()) {
        std::cerr << "[Error] No data after fDecRad cut, abort fit.\n";
        return;
    }

    RooRealVar mass("m", "Mass(H3L)", massMin, massMax, "GeV/c^{2}");
    RooDataSet data("data", "data", RooArgSet(mass));
    for (double v : massAllDec) {
        if (v < massMin || v > massMax) continue;
        mass.setVal(v);
        data.add(RooArgSet(mass));
    }
    RooDataSet mc("mc", "mc", RooArgSet(mass));
    for (double v : massAllMcDec) {
        if (v < massMin || v > massMax) continue;
        mass.setVal(v);
        mc.add(RooArgSet(mass));
    }

    RooRealVar muMc("muMc", "muMc", 2.991, 2.97, 3.01);
    RooRealVar sigmaMcVar("sigmaMc", "sigmaMc", 1.5e-3, 1.1e-3, 2.1e-3);
    RooRealVar a1McVar("a1Mc", "a1Mc", 1.5, 0.1, 10.0);
    RooRealVar a2McVar("a2Mc", "a2Mc", 1.5, 0.1, 10.0);
    RooRealVar n1McVar("n1Mc", "n1Mc", 5.0, 0.5, 30.0);
    RooRealVar n2McVar("n2Mc", "n2Mc", 5.0, 0.5, 30.0);
    RooCrystalBall sigMc("sigMc", "sigMc", mass, muMc, sigmaMcVar, a1McVar, n1McVar, a2McVar, n2McVar);
    sigMc.fitTo(mc, RooFit::Range(2.97, 3.01), RooFit::Save(true), RooFit::PrintLevel(-1));
    a1McVar.setConstant();
    a2McVar.setConstant();
    n1McVar.setConstant();
    n2McVar.setConstant();

    if (massAllNoCut.empty()) {
        std::cerr << "[Error] No data before fDecRad cut, abort fit." << "\n";
        return;
    }

    // Fit data before fDecRad cut, sharing MC DSCB tails
    RooDataSet dataNoCut("dataNoCut", "dataNoCut", RooArgSet(mass));
    for (double v : massAllNoCut) {
        if (v < massMin || v > massMax) continue;
        mass.setVal(v);
        dataNoCut.add(RooArgSet(mass));
    }
    RooRealVar muNoCut("muNoCut", "muNoCut", muMc.getVal(), 2.985, 2.992);
    RooRealVar sigmaNoCut("sigmaNoCut", "sigmaNoCut", sigmaMcVar.getVal(), 1.0e-3, 3.0e-3);
    RooCrystalBall sigNoCut("sigNoCut", "sigNoCut", mass, muNoCut, sigmaNoCut, a1McVar, n1McVar, a2McVar, n2McVar);
    RooRealVar c0Pre("c0Pre", "c0Pre", 0.0, -1.5, 1.5);
    RooRealVar c1Pre("c1Pre", "c1Pre", 0.0, -1.5, 1.5);
    RooRealVar c2Pre("c2Pre", "c2Pre", 0.0, -1.5, 1.5);
    RooChebychev bkgPre("bkgPre", "bkgPre", mass, RooArgList(c0Pre, c1Pre, c2Pre));
    const double nSigInitPre = std::max(1.0, 0.5 * static_cast<double>(dataNoCut.numEntries()));
    const double nBkgInitPre = std::max(1.0, 0.5 * static_cast<double>(dataNoCut.numEntries()));
    RooRealVar nSigPre("nSigPre", "nSigPre", nSigInitPre, 0.0, 1e6);
    RooRealVar nBkgPre("nBkgPre", "nBkgPre", nBkgInitPre, 0.0, 1e6);
    RooAddPdf modelPre("modelPre", "modelPre", RooArgList(sigNoCut, bkgPre), RooArgList(nSigPre, nBkgPre));
    modelPre.fitTo(dataNoCut, RooFit::Extended(true), RooFit::Save(true), RooFit::PrintLevel(-1));

    const double muDataPre = muNoCut.getVal();
    const double sigmaDataPre = sigmaNoCut.getVal();
    const double windowMin3Pre = muDataPre - 3.0 * sigmaDataPre;
    const double windowMax3Pre = muDataPre + 3.0 * sigmaDataPre;
    const double windowMin2Pre = muDataPre - 2.0 * sigmaDataPre;
    const double windowMax2Pre = muDataPre + 2.0 * sigmaDataPre;
    mass.setRange("sigWindowPre", windowMin3Pre, windowMax3Pre);
    std::unique_ptr<RooAbsReal> sigIntegralPre(sigNoCut.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindowPre")));
    std::unique_ptr<RooAbsReal> bkgIntegralPre(bkgPre.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindowPre")));
    const double sigFracPre = sigIntegralPre ? sigIntegralPre->getVal() : 0.0;
    const double bkgFracPre = bkgIntegralPre ? bkgIntegralPre->getVal() : 0.0;
    const double signalCounts3sPre = nSigPre.getVal() * sigFracPre;
    const double signalCounts3sErrPre = nSigPre.getError() * sigFracPre;
    const double bkgCounts3sPre = nBkgPre.getVal() * bkgFracPre;
    const double bkgCounts3sErrPre = nBkgPre.getError() * bkgFracPre;
    double significancePre = 0.0;
    double significancePreErr = 0.0;
    if (signalCounts3sPre + bkgCounts3sPre > 0.0) {
        significancePre = signalCounts3sPre / std::sqrt(signalCounts3sPre + bkgCounts3sPre);
        const double dSdSigPre = std::sqrt(signalCounts3sPre + bkgCounts3sPre) - (signalCounts3sPre / (2.0 * std::sqrt(signalCounts3sPre + bkgCounts3sPre)));
        const double dBdSigPre = -(signalCounts3sPre / (2.0 * std::sqrt(signalCounts3sPre + bkgCounts3sPre)));
        significancePreErr = std::sqrt(std::pow(dSdSigPre * signalCounts3sErrPre, 2) + std::pow(dBdSigPre * bkgCounts3sErrPre, 2));
    }

    std::unique_ptr<RooPlot> framePre(mass.frame(40));
    dataNoCut.plotOn(framePre.get(), RooFit::Name("data_noCut"));
    modelPre.plotOn(framePre.get(), RooFit::Name("total_noCut"));
    modelPre.plotOn(framePre.get(), RooFit::Components(bkgPre), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed + 1), RooFit::Name("bkg_noCut"));
    modelPre.plotOn(framePre.get(), RooFit::Components(sigNoCut), RooFit::LineStyle(kDotted), RooFit::LineColor(kGreen + 2), RooFit::Name("sig_noCut"));
    auto textDataPre = std::make_unique<TPaveText>(0.58, 0.36, 0.88, 0.88, "NDC");
    textDataPre->SetBorderSize(0);
    textDataPre->SetFillStyle(0);
    textDataPre->SetTextAlign(12);
    textDataPre->AddText(Form("Data fit (no fDecRad cut)"));
    textDataPre->AddText(Form(" S (3#sigma) = %.1f #pm %.1f", signalCounts3sPre, signalCounts3sErrPre));
    textDataPre->AddText(Form(" B (3#sigma) = %.1f #pm %.1f", bkgCounts3sPre, bkgCounts3sErrPre));
    textDataPre->AddText(Form(" S/#sqrt{S+B} = %.2f #pm %.2f", significancePre, significancePreErr));
    textDataPre->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muDataPre * 1e3, muNoCut.getError() * 1e3));
    textDataPre->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaDataPre * 1e3, sigmaNoCut.getError() * 1e3));
    framePre->addObject(textDataPre.release());

    TCanvas cDataPre("c_data_pre", "Data fit before fDecRad cut", 800, 600);
    framePre->Draw();
    cDataPre.SaveAs((outDir + "/fit_data_noDecRad.pdf").c_str());

    RooRealVar mu("mu", "mu", muMc.getVal(), 2.985, 2.992);
    RooRealVar sigma("sigma", "sigma", sigmaMcVar.getVal(), 1.0e-3, 3.0e-3);
    RooCrystalBall sig("sig", "sig", mass, mu, sigma, a1McVar, n1McVar, a2McVar, n2McVar);

    RooRealVar c0("c0", "c0", 0.0, -1.5, 1.5);
    RooRealVar c1("c1", "c1", 0.0, -1.5, 1.5);
    RooRealVar c2("c2", "c2", 0.0, -1.5, 1.5);
    RooChebychev bkg("bkg", "bkg", mass, RooArgList(c0, c1, c2));

    const double nSigInit = std::max(1.0, 0.5 * static_cast<double>(data.numEntries()));
    const double nBkgInit = std::max(1.0, 0.5 * static_cast<double>(data.numEntries()));
    RooRealVar nSig("nSig", "nSig", nSigInit, 0.0, 1e6);
    RooRealVar nBkg("nBkg", "nBkg", nBkgInit, 0.0, 1e6);
    RooAddPdf model("model", "total_pdf", RooArgList(sig, bkg), RooArgList(nSig, nBkg));
    model.fitTo(data, RooFit::Extended(true), RooFit::Save(true), RooFit::PrintLevel(-1));

    const double muData = mu.getVal();
    const double sigmaData = sigma.getVal();
    const double windowMin3 = muData - 3.0 * sigmaData;
    const double windowMax3 = muData + 3.0 * sigmaData;
    mass.setRange("sigWindow", windowMin3, windowMax3);
    std::unique_ptr<RooAbsReal> sigIntegral(sig.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
    std::unique_ptr<RooAbsReal> bkgIntegral(bkg.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
    const double sigFrac = sigIntegral ? sigIntegral->getVal() : 0.0;
    const double bkgFrac = bkgIntegral ? bkgIntegral->getVal() : 0.0;
    const double signalCounts3s = nSig.getVal() * sigFrac;
    const double signalCounts3sErr = nSig.getError() * sigFrac;
    const double bkgCounts3s = nBkg.getVal() * bkgFrac;
    const double bkgCounts3sErr = nBkg.getError() * bkgFrac;
    double significance = 0.0;
    double significanceErr = 0.0;
    if (signalCounts3s + bkgCounts3s > 0.0) {
        significance = signalCounts3s / std::sqrt(signalCounts3s + bkgCounts3s);
        const double dSdSig = std::sqrt(signalCounts3s + bkgCounts3s) - (signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
        const double dBdSig = -(signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
        significanceErr = std::sqrt(std::pow(dSdSig * signalCounts3sErr, 2) + std::pow(dBdSig * bkgCounts3sErr, 2));
    }

    std::unique_ptr<RooPlot> frameMc(mass.frame(80));
    mc.plotOn(frameMc.get(), RooFit::Name("mc"));
    sigMc.plotOn(frameMc.get(), RooFit::LineColor(kRed), RooFit::LineStyle(kDashed), RooFit::Name("sig_fit_mc"));
    auto textMc = std::make_unique<TPaveText>(0.60, 0.43, 0.90, 0.85, "NDC");
    textMc->SetBorderSize(0);
    textMc->SetFillStyle(0);
    textMc->SetTextAlign(12);
    textMc->AddText(Form("MC DSCB"));
    textMc->AddText(Form(" #mu = %.3f MeV/c^{2}", muMc.getVal() * 1e3));
    textMc->AddText(Form(" #sigma = %.3f MeV/c^{2}", sigmaMcVar.getVal() * 1e3));
    textMc->AddText(Form(" #alpha_{l} = %.2f", a1McVar.getVal()));
    textMc->AddText(Form(" n_{l} = %.2f", n1McVar.getVal()));
    textMc->AddText(Form(" #alpha_{r} = %.2f", a2McVar.getVal()));
    textMc->AddText(Form(" n_{r} = %.2f", n2McVar.getVal()));
    frameMc->addObject(textMc.release());

    std::unique_ptr<RooPlot> frame(mass.frame(40));
    data.plotOn(frame.get(), RooFit::Name("data"));
    model.plotOn(frame.get(), RooFit::Name("total"));
    model.plotOn(frame.get(), RooFit::Components(bkg), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed + 1), RooFit::Name("bkg"));
    model.plotOn(frame.get(), RooFit::Components(sig), RooFit::LineStyle(kDotted), RooFit::LineColor(kGreen + 2), RooFit::Name("sig"));
    auto textData = std::make_unique<TPaveText>(0.58, 0.36, 0.88, 0.88, "NDC");
    textData->SetBorderSize(0);
    textData->SetFillStyle(0);
    textData->SetTextAlign(12);
    textData->AddText(Form("Data fit (fDecRad<2.1)"));
    textData->AddText(Form(" S (3#sigma) = %.1f #pm %.1f", signalCounts3s, signalCounts3sErr));
    textData->AddText(Form(" B (3#sigma) = %.1f #pm %.1f", bkgCounts3s, bkgCounts3sErr));
    textData->AddText(Form(" S/#sqrt{S+B} = %.2f #pm %.2f", significance, significanceErr));
    textData->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muData * 1e3, mu.getError() * 1e3));
    textData->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaData * 1e3, sigma.getError() * 1e3));
    frame->addObject(textData.release());

    TCanvas cMc("c_mc", "MC fit", 800, 600);
    frameMc->Draw();
    cMc.SaveAs((outDir + "/fit_mc.pdf").c_str());

    TCanvas cData("c_data", "Data fit", 800, 600);
    frame->Draw();
    cData.SaveAs((outDir + "/fit_data.pdf").c_str());

    const double windowMin2 = muData - 2.0 * sigmaData;
    const double windowMax2 = muData + 2.0 * sigmaData;
    if (ctBins.size() < 2) {
        std::cerr << "[Error] Need at least two ct bin edges, got " << ctBins.size() << "\n";
        return;
    }
    const int nCtBins = static_cast<int>(ctBins.size() - 1);
    const double ctMin = ctBins.front();
    const double ctMax = ctBins.back();
    TH1D hCtData("h_ct_data", ";c#tau (cm);Entries", nCtBins, ctBins.data());
    hCtData.SetDirectory(nullptr);
    for (const auto &p : dataMassCt) {
        if (p.first >= windowMin2 && p.first <= windowMax2 && p.second >= ctMin && p.second <= ctMax) {
            hCtData.Fill(p.second);
        }
    }
    TCanvas cCtData("c_ct_data", "ct_data", 800, 600);
    hCtData.SetLineColor(kBlue + 2);
    hCtData.SetLineWidth(2);
    hCtData.Draw("hist");
    cCtData.SaveAs((outDir + "/ct_data_2sigma.pdf").c_str());

    // Decay radius distributions (no decay-radius cut) within 2#sigma mass window
    const int decRadBins = 180;
    const double decRadMin = 0.0;
    const double decRadMax = 5.0;

    TH1D hDecRadData("h_decRad_data", ";fDecRad (cm);Entries", decRadBins, decRadMin, decRadMax);
    hDecRadData.SetDirectory(nullptr);
    for (const auto &p : dataMassDecRad) {
        if (p.first >= windowMin2Pre && p.first <= windowMax2Pre) {
            hDecRadData.Fill(p.second);
        }
    }
    TCanvas cDecRadData("c_decRad_data", "DecRad data 2sigma (no cut)", 800, 600);
    hDecRadData.SetLineColor(kBlue + 2);
    hDecRadData.SetLineWidth(2);
    hDecRadData.Draw("hist");
    cDecRadData.SaveAs((outDir + "/decRad_data_preCut_2sigma.pdf").c_str());

    TH1D hDecRadMc("h_decRad_mc", ";fDecRad (cm);Entries", decRadBins, decRadMin, decRadMax);
    hDecRadMc.SetDirectory(nullptr);
    const double muMcVal = muMc.getVal();
    const double sigmaMcVal = sigmaMcVar.getVal();
    const double mcWinMin2 = muMcVal - 2.0 * sigmaMcVal;
    const double mcWinMax2 = muMcVal + 2.0 * sigmaMcVal;
    for (const auto &p : mcMassDecRad) {
        if (p.first >= mcWinMin2 && p.first <= mcWinMax2) {
            hDecRadMc.Fill(p.second);
        }
    }
    TCanvas cDecRadMc("c_decRad_mc", "DecRad MC 2sigma (no cut)", 800, 600);
    hDecRadMc.SetLineColor(kRed + 1);
    hDecRadMc.SetLineWidth(2);
    hDecRadMc.Draw("hist");
    cDecRadMc.SaveAs((outDir + "/decRad_mc_preCut_2sigma.pdf").c_str());
}
