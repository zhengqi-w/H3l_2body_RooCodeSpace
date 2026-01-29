#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TH1D.h>
#include <TString.h>
#include <TPaveText.h>
#include <TChain.h>
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
#include <thread>
#include <limits>

#include "../Tools/GeneralHelper.hpp"
#include "../Tools/AcceptanceHelper.h"
#include <TLine.h>

using namespace AcceptanceHelper;
using namespace GeneralHelper;

void MCEffCheck(const string &stdmcpath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root",
                const vector<string> &comparepaths = {"/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_CustomV0s.root",
                                                      "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_V0s_full.root",
                                                      "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_V0s_full.root"},
                const vector<string> &labels = {"CustomV0s(with H3l interaction)", "CustomV0s(without H3l interaction)",  "V0s(with H3l interaction)", "V0s(without H3l interaction)"},
                const vector<double> ptbins = {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
                const vector<double> ctbins = {1, 3, 6, 9, 12, 18, 30},
                const vector<double> ptbinsforct = {2, 3, 4, 5.5, 8},
                const vector<vector<double>> ctbinsforpt = { {1, 3, 6, 9, 12, 18, 30},
                                                             {1, 3, 6, 9, 12, 18, 25},
                                                             {1, 3, 6, 9, 15, 25},
                                                             {1, 3, 6, 10, 23} },
                const string &ptreweightpath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root",
                const string &outputpath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MCEfficiency",
                const string &matterOpt = "both") {
    // Basic MT setup
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }

    auto loadAcc = [&](const std::string &path) {
        TChain chain("O2mchypcands");
        auto f = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            throw std::runtime_error("Cannot open MC file: " + path);
        }
        fillChainFromAO2D(chain, f.get());
        ROOT::RDataFrame rdf(chain);
        auto ready = CorrectAndConvertRDF(rdf, false, true, false);
        // Separate calls for pt, ct, and ct-per-pt scenarios to satisfy API constraints
        AcceptanceHelper::AcceptanceResult resPt      = AcceptanceHelper::ComputeAcceptanceFlexible(ready, ptbins, {}, {}, {}, {});
        AcceptanceHelper::AcceptanceResult resCt      = AcceptanceHelper::ComputeAcceptanceFlexible(ready, {}, ctbins, {}, {}, {});
        AcceptanceHelper::AcceptanceResult resCtPerPt = AcceptanceHelper::ComputeAcceptanceFlexible(ready, ptbinsforct, {}, ctbinsforpt, {}, {});

        AcceptanceHelper::AcceptanceResult out;
        // pt efficiencies
        out.acc_pt_both        = resPt.acc_pt_both;
        out.acc_pt_matter      = resPt.acc_pt_matter;
        out.acc_pt_antimatter  = resPt.acc_pt_antimatter;
        // ct efficiencies
        out.acc_ct_both        = resCt.acc_ct_both;
        out.acc_ct_matter      = resCt.acc_ct_matter;
        out.acc_ct_antimatter  = resCt.acc_ct_antimatter;
        // ct per pt
        out.acc_ct_per_pt              = resCtPerPt.acc_ct_per_pt;
        out.acc_ct_per_pt_matter       = resCtPerPt.acc_ct_per_pt_matter;
        out.acc_ct_per_pt_antimatter   = resCtPerPt.acc_ct_per_pt_antimatter;
        return out;
    };

    auto cloneHist = [](TH1D *h, const std::string &name) -> std::unique_ptr<TH1D> {
        if (!h) return nullptr;
        auto c = std::unique_ptr<TH1D>(static_cast<TH1D*>(h->Clone(name.c_str())));
        c->SetDirectory(nullptr);
        return c;
    };

    auto pickPt = [&](const AcceptanceHelper::AcceptanceResult &res) -> TH1D* {
        if (matterOpt == "antimatter" && res.acc_pt_antimatter) return res.acc_pt_antimatter;
        if (matterOpt == "matter" && res.acc_pt_matter) return res.acc_pt_matter;
        return res.acc_pt_both ? res.acc_pt_both : (res.acc_pt_matter ? res.acc_pt_matter : res.acc_pt_antimatter);
    };
    auto pickCt = [&](const AcceptanceHelper::AcceptanceResult &res) -> TH1D* {
        if (matterOpt == "antimatter" && res.acc_ct_antimatter) return res.acc_ct_antimatter;
        if (matterOpt == "matter" && res.acc_ct_matter) return res.acc_ct_matter;
        return res.acc_ct_both ? res.acc_ct_both : (res.acc_ct_matter ? res.acc_ct_matter : res.acc_ct_antimatter);
    };

    AcceptanceHelper::AcceptanceResult stdRes = loadAcc(stdmcpath);
    std::vector<AcceptanceHelper::AcceptanceResult> compRes;
    compRes.reserve(comparepaths.size());
    for (const auto &p : comparepaths) compRes.emplace_back(loadAcc(p));

    std::filesystem::create_directories(outputpath);

    const std::vector<Color_t> colors = {kBlack, kRed + 1, kAzure + 2, kGreen + 2};
    const std::vector<Style_t> markers = {20, 21, 22, 23};

    auto drawSet = [&](const std::string &cname, const std::string &title,
                       TH1D *stdH, const std::vector<TH1D*> &compH,
                       const std::string &ratioTitle) {
        if (!stdH) return;
        TCanvas c(cname.c_str(), title.c_str(), 1200, 600);
        TPad pLeft("pleft","pleft",0,0,0.65,1); pLeft.Draw();
        TPad pRight("pright","pright",0.65,0,1,1); pRight.Draw();

        auto minmaxHists = [](const std::vector<TH1D*> &hs) {
            double minv = std::numeric_limits<double>::max();
            double maxv = std::numeric_limits<double>::lowest();
            for (auto *h : hs) {
                if (!h) continue;
                const int nb = h->GetNbinsX();
                for (int i=1;i<=nb;++i) {
                    double v = h->GetBinContent(i);
                    minv = std::min(minv, v);
                    maxv = std::max(maxv, v);
                }
            }
            if (minv == std::numeric_limits<double>::max()) {
                minv = 0.0; maxv = 1.0;
            }
            return std::make_pair(minv, maxv);
        };

        // Left: efficiencies
        pLeft.cd();
        bool isPtCanvas = (cname == "eff_pt");
        double legX1 = isPtCanvas ? 0.15 : 0.55;
        double legY1 = isPtCanvas ? 0.65 : 0.65;
        double legX2 = isPtCanvas ? 0.48 : 0.88;
        double legY2 = isPtCanvas ? 0.88 : 0.88;
        stdH->SetTitle((title+";"+stdH->GetXaxis()->GetTitle()+";Efficiency").c_str());
        stdH->SetStats(false);
        stdH->SetLineColor(colors[0]);
        stdH->SetMarkerColor(colors[0]);
        stdH->SetMarkerStyle(markers[0]);
        stdH->Draw("PE");
        TLegend leg(legX1, legY1, legX2, legY2);
        leg.SetFillStyle(0);
        leg.SetBorderSize(0);
        leg.SetFillColorAlpha(0,0);
        leg.AddEntry(stdH, labels[0].c_str(), "lep");
        for (size_t i=0;i<compH.size();++i) {
            auto h = compH[i]; if (!h) continue;
            h->SetStats(false);
            h->SetLineColor(colors[i+1]);
            h->SetMarkerColor(colors[i+1]);
            h->SetMarkerStyle(markers[i+1]);
            h->Draw("SAME PE");
            leg.AddEntry(h, labels[i+1].c_str(), "lep");
        }
        {
            std::vector<TH1D*> allH; allH.push_back(stdH); for (auto *h: compH) allH.push_back(h);
            auto [mn,mx] = minmaxHists(allH);
            stdH->SetMinimum(0.7*mn);
            stdH->SetMaximum(1.2*mx);
        }
        leg.Draw();

        // Right: ratios (compare / std)
        pRight.cd();
        std::vector<std::unique_ptr<TH1D>> ratioH;
        ratioH.reserve(compH.size());
        for (size_t i=0;i<compH.size();++i) {
            if (!compH[i]) { ratioH.push_back(nullptr); continue; }
            auto r = cloneHist(compH[i], Form("ratio_%s_%zu", cname.c_str(), i));
            r->Divide(stdH);
            r->SetStats(false);
            r->SetTitle("");
            r->SetLineColor(colors[i+1]);
            r->SetMarkerColor(colors[i+1]);
            r->SetMarkerStyle(markers[i+1]);
            r->GetYaxis()->SetTitle(ratioTitle.c_str());
            ratioH.push_back(std::move(r));
        }
        std::vector<TH1D*> ratioPtrs; for (auto &r: ratioH) if (r) ratioPtrs.push_back(r.get());
        auto [rmin, rmax] = minmaxHists(ratioPtrs);
        double rLow = 0.7*rmin;
        double rHigh = 1.2*rmax;
        bool firstDrawn = false;
        for (auto &r : ratioH) {
            if (!r) continue;
            r->SetMinimum(rLow);
            r->SetMaximum(rHigh);
            if (!firstDrawn) { r->Draw("PE"); firstDrawn=true; }
            else r->Draw("SAME PE");
        }
        TLine line(stdH->GetXaxis()->GetXmin(),1,stdH->GetXaxis()->GetXmax(),1);
        line.SetLineStyle(2); line.Draw();
        c.SaveAs((outputpath + "/" + cname + ".pdf").c_str());
    };

    // pt efficiency
    auto stdPt = cloneHist(pickPt(stdRes), "std_pt");
    std::vector<TH1D*> compPt;
    for (size_t i=0;i<compRes.size();++i) compPt.push_back(pickPt(compRes[i]));
    std::vector<std::unique_ptr<TH1D>> compPtOwned;
    for (size_t i=0;i<compPt.size();++i) compPtOwned.push_back(cloneHist(compPt[i], Form("comp_pt_%zu", i)));
    std::vector<TH1D*> compPtPtrs; for (auto &p: compPtOwned) compPtPtrs.push_back(p.get());
    drawSet("eff_pt", Form("Efficiency vs p_{T} (%s)", matterOpt.c_str()), stdPt.get(), compPtPtrs, "ratio to std");

    // ct efficiency
    auto stdCt = cloneHist(pickCt(stdRes), "std_ct");
    std::vector<TH1D*> compCt;
    for (size_t i=0;i<compRes.size();++i) compCt.push_back(pickCt(compRes[i]));
    std::vector<std::unique_ptr<TH1D>> compCtOwned;
    for (size_t i=0;i<compCt.size();++i) compCtOwned.push_back(cloneHist(compCt[i], Form("comp_ct_%zu", i)));
    std::vector<TH1D*> compCtPtrs; for (auto &p: compCtOwned) compCtPtrs.push_back(p.get());
    drawSet("eff_ct", Form("Efficiency vs ct (%s)", matterOpt.c_str()), stdCt.get(), compCtPtrs, "ratio to std");

    // ct per pt-bin
    auto pickVec = [&](const AcceptanceHelper::AcceptanceResult &res) -> const std::vector<TH1D*>& {
        if (!res.acc_ct_per_pt_antimatter.empty()) return res.acc_ct_per_pt_antimatter;
        if (!res.acc_ct_per_pt_matter.empty()) return res.acc_ct_per_pt_matter;
        return res.acc_ct_per_pt;
    };
    const auto &stdCtPerPt = pickVec(stdRes);
    std::vector<const std::vector<TH1D*>*> compCtPerPt;
    for (auto &r : compRes) compCtPerPt.push_back(&pickVec(r));

    for (size_t ipt=0; ipt<stdCtPerPt.size(); ++ipt) {
        auto stdH = cloneHist(stdCtPerPt[ipt], Form("std_ct_pt_%zu", ipt));
        if (!stdH) continue;
        std::vector<std::unique_ptr<TH1D>> compsOwned;
        std::vector<TH1D*> compsPtrs;
        for (size_t ic=0; ic<compCtPerPt.size(); ++ic) {
            TH1D *h = nullptr;
            if (ipt < compCtPerPt[ic]->size()) h = (*compCtPerPt[ic])[ipt];
            compsOwned.push_back(cloneHist(h, Form("comp_ct_pt_%zu_%zu", ipt, ic)));
        }
        for (auto &p : compsOwned) compsPtrs.push_back(p.get());
        std::string ttl = Form("Efficiency vs ct (%.1f-%.1f GeV/c, %s)", ptbinsforct[ipt], ptbinsforct[ipt+1], matterOpt.c_str());
        std::string cname = Form("eff_ct_ptbin_%zu", ipt);
        drawSet(cname, ttl, stdH.get(), compsPtrs, "ratio to std");
    }
}