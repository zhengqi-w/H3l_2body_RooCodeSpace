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
#include <TPad.h>

#include <algorithm>
#include <cctype>
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

void MCEffCheck(const string &stdmcpath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
                const vector<string> &comparepaths = { "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root"},
                const vector<bool> &TwoBodySelections = {true, true},
                const vector<string> &labels = {"Pass5_NoClusterSizeCut",  "Pass5_ClusterSizeCut"},
                const vector<double> cenbins = {0, 10, 30, 50, 80},
                const vector<vector<double>> ptbinsforcent = { 
                                           {2, 3, 3.5, 4, 4.5, 5, 6, 8},
                                           {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
                                           {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
                                           {2, 2.5, 3, 3.5, 4, 5, 8} },
                const vector<double> ptbins = {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
                const vector<double> ctbins = {1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 27, 33},
                const vector<double> ptbinsforct = {2, 3, 4, 5.5, 8},
                const vector<vector<double>> ctbinsforpt = { {1, 3, 6, 9, 12, 18, 30},
                                                             {1, 3, 6, 9, 12, 18, 25},
                                                             {1, 3, 6, 9, 15, 25},
                                                             {1, 3, 6, 10, 23} },
                const vector<string> &basic_selection_data_for_mc_eff = {"fDecRad > 0.8", "fDecRad > 0.8 && fAvgClusterSizeHe > 5"},
                const string &outputpath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MCEfficiency_ClusterSizeCheck",
                const string &matterOpt = "both") {
    // Basic MT setup
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }

    auto loadAcc = [&](const std::string &path, const std::string &basicSel, bool selTwoBody) {
        TChain chain("O2mchypcands");
        auto f = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            throw std::runtime_error("Cannot open MC file: " + path);
        }
        fillChainFromAO2D(chain, f.get());
        ROOT::RDataFrame rdf(chain);
        auto ready = CorrectAndConvertRDF(rdf, false, true, false);
        // Separate calls for pt, ct, and ct-per-pt scenarios to satisfy API constraints
        AcceptanceHelper::AcceptanceResult resPt      = AcceptanceHelper::ComputeAcceptanceFlexible(ready, ptbins, {}, {}, {}, {}, basicSel, {}, selTwoBody);
        AcceptanceHelper::AcceptanceResult resCt      = AcceptanceHelper::ComputeAcceptanceFlexible(ready, {}, ctbins, {}, {}, {}, basicSel, {}, selTwoBody);
        AcceptanceHelper::AcceptanceResult resCtPerPt = AcceptanceHelper::ComputeAcceptanceFlexible(ready, ptbinsforct, {}, ctbinsforpt, {}, {}, basicSel, {}, selTwoBody);
        AcceptanceHelper::AcceptanceResult resPtPerCent = AcceptanceHelper::ComputeAcceptanceFlexible(ready, {}, {}, {}, cenbins, ptbinsforcent, basicSel, {}, selTwoBody);

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
        // pt per centrality
        out.acc_pt_per_cent            = resPtPerCent.acc_pt_per_cent;
        out.acc_pt_per_cent_matter     = resPtPerCent.acc_pt_per_cent_matter;
        out.acc_pt_per_cent_antimatter = resPtPerCent.acc_pt_per_cent_antimatter;
        return out;
    };

    auto cloneHist = [](TH1D *h, const std::string &name) -> std::unique_ptr<TH1D> {
        if (!h) return nullptr;
        auto c = std::unique_ptr<TH1D>(static_cast<TH1D*>(h->Clone(name.c_str())));
        c->SetDirectory(nullptr);
        return c;
    };

    auto hasSameBinning = [](const TH1D *a, const TH1D *b) {
        if (!a || !b) return false;
        if (a->GetNbinsX() != b->GetNbinsX()) return false;
        const int nb = a->GetNbinsX();
        for (int i = 1; i <= nb + 1; ++i) {
            if (std::abs(a->GetXaxis()->GetBinLowEdge(i) - b->GetXaxis()->GetBinLowEdge(i)) > 1e-9) {
                return false;
            }
        }
        return true;
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

    auto pickBasicSel = [&](size_t idx) {
        if (basic_selection_data_for_mc_eff.empty()) return std::string("1");
        if (idx < basic_selection_data_for_mc_eff.size()) return basic_selection_data_for_mc_eff[idx];
        return basic_selection_data_for_mc_eff.back();
    };
    auto pickTwoBody = [&](size_t idx) -> bool {
        if (TwoBodySelections.empty()) return true;
        if (idx < TwoBodySelections.size()) return static_cast<bool>(TwoBodySelections[idx]);
        return static_cast<bool>(TwoBodySelections.back());
    };

    AcceptanceHelper::AcceptanceResult stdRes = loadAcc(stdmcpath, pickBasicSel(0), pickTwoBody(0));
    std::vector<AcceptanceHelper::AcceptanceResult> compRes;
    compRes.reserve(comparepaths.size());
    for (size_t i = 0; i < comparepaths.size(); ++i) {
        compRes.emplace_back(loadAcc(comparepaths[i], pickBasicSel(i + 1), pickTwoBody(i + 1)));
    }

    std::filesystem::create_directories(outputpath);

    constexpr int kCanvasW = 1000;
    constexpr int kCanvasH = 750;

    auto edgeToToken = [](double x) {
        std::ostringstream os;
        os << std::setprecision(6) << std::defaultfloat << x;
        std::string s = os.str();
        std::replace(s.begin(), s.end(), '.', 'p');
        return s;
    };

    auto sanitizeLabel = [](std::string s) {
        for (auto &ch : s) {
            if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')) ch = '_';
        }
        std::string out;
        out.reserve(s.size());
        bool prevUnderscore = false;
        for (char ch : s) {
            if (ch == '_') {
                if (!prevUnderscore) out.push_back(ch);
                prevUnderscore = true;
            } else {
                out.push_back(ch);
                prevUnderscore = false;
            }
        }
        while (!out.empty() && out.front() == '_') out.erase(out.begin());
        while (!out.empty() && out.back() == '_') out.pop_back();
        return out.empty() ? std::string("set") : out;
    };

    const std::vector<Color_t> colors = {kBlack, kRed + 1, kAzure + 2, kGreen + 2};
    const std::vector<Style_t> markers = {20, 21, 22, 23};
    const float legendTextSize = 0.04f;
    const Float_t prevEndErrorSize = gStyle->GetEndErrorSize();
    gStyle->SetEndErrorSize(4.0f);

    auto drawSet = [&](const std::string &cname, const std::string &title,
                       TH1D *stdH, const std::vector<TH1D*> &compH,
                       const std::string &ratioTitle,
                       bool useTopBottomPads = false) {
        if (!stdH) return;
        TCanvas c(cname.c_str(), title.c_str(), kCanvasW, kCanvasH);
        TPad pLeft("pleft","pleft",0,0,0.65,1);
        TPad pRight("pright","pright",0.65,0,1,1);
        TPad pTop("ptop", "ptop", 0.0, 0.30, 1.0, 1.0);
        TPad pBot("pbot", "pbot", 0.0, 0.0, 1.0, 0.30);
        if (useTopBottomPads) {
            pTop.SetBottomMargin(0.02);
            pBot.SetTopMargin(0.02);
            pBot.SetBottomMargin(0.32);
            pTop.Draw();
            pBot.Draw();
        } else {
            pLeft.Draw();
            pRight.Draw();
        }

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
        if (useTopBottomPads) pTop.cd();
        else pLeft.cd();
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
        stdH->SetLineWidth(3);
        if (useTopBottomPads) {
            stdH->GetXaxis()->SetLabelSize(0.0);
            stdH->GetXaxis()->SetTitleSize(0.0);
        }
        stdH->Draw("E1 P");
        TLegend leg(legX1, legY1, legX2, legY2);
        leg.SetFillStyle(0);
        leg.SetBorderSize(0);
        leg.SetFillColorAlpha(0,0);
        leg.SetTextSize(legendTextSize);
        leg.AddEntry(stdH, labels[0].c_str(), "lep");
        for (size_t i=0;i<compH.size();++i) {
            auto h = compH[i]; if (!h) continue;
            h->SetStats(false);
            h->SetLineColor(colors[i+1]);
            h->SetMarkerColor(colors[i+1]);
            h->SetMarkerStyle(markers[i+1]);
            h->SetLineWidth(3);
            h->Draw("SAME E1 P");
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
        if (useTopBottomPads) pBot.cd();
        else pRight.cd();
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
            if (useTopBottomPads) {
                r->GetYaxis()->SetTitleSize(0.10);
                r->GetYaxis()->SetTitleOffset(0.50);
                r->GetYaxis()->SetLabelSize(0.09);
                r->GetXaxis()->SetTitleSize(0.12);
                r->GetXaxis()->SetTitleOffset(1.0);
                r->GetXaxis()->SetLabelSize(0.10);
            }
            ratioH.push_back(std::move(r));
        }
        std::vector<TH1D*> ratioPtrs; for (auto &r: ratioH) if (r) ratioPtrs.push_back(r.get());
        auto [rmin, rmax] = minmaxHists(ratioPtrs);
        double rLow = 0.7*rmin;
        double rHigh = 1.2*rmax;
        // Always keep y=1 visible for ratio pads.
        rLow = std::min(rLow, 0.95);
        rHigh = std::max(rHigh, 1.05);

        auto ratioFrame = cloneHist(stdH, Form("ratio_frame_%s", cname.c_str()));
        if (ratioFrame) {
            ratioFrame->Reset("ICES");
            ratioFrame->SetStats(false);
            ratioFrame->SetMinimum(rLow);
            ratioFrame->SetMaximum(rHigh);
            ratioFrame->SetTitle((";" + std::string(stdH->GetXaxis()->GetTitle()) + ";" + ratioTitle).c_str());
            if (useTopBottomPads) {
                ratioFrame->GetYaxis()->SetTitleSize(0.10);
                ratioFrame->GetYaxis()->SetTitleOffset(0.50);
                ratioFrame->GetYaxis()->SetLabelSize(0.09);
                ratioFrame->GetXaxis()->SetTitleSize(0.12);
                ratioFrame->GetXaxis()->SetTitleOffset(1.0);
                ratioFrame->GetXaxis()->SetLabelSize(0.10);
            }
            ratioFrame->Draw("HIST");
        }

        bool firstDrawn = false;
        for (auto &r : ratioH) {
            if (!r) continue;
            r->SetMinimum(rLow);
            r->SetMaximum(rHigh);
            r->SetLineWidth(3);
            if (!firstDrawn) { r->Draw("E1 P"); firstDrawn=true; }
            else r->Draw("SAME E1 P");
        }
        TLine line(stdH->GetXaxis()->GetXmin(),1,stdH->GetXaxis()->GetXmax(),1);
        line.SetLineColor(kBlack);
        line.SetLineStyle(2);
        line.SetLineWidth(3);
        line.Draw();
        c.SaveAs((outputpath + "/" + cname + ".pdf").c_str());
    };

    // pt efficiency
    auto stdPt = cloneHist(pickPt(stdRes), "std_pt");
    std::vector<TH1D*> compPt;
    for (size_t i=0;i<compRes.size();++i) compPt.push_back(pickPt(compRes[i]));
    std::vector<std::unique_ptr<TH1D>> compPtOwned;
    for (size_t i=0;i<compPt.size();++i) compPtOwned.push_back(cloneHist(compPt[i], Form("comp_pt_%zu", i)));
    std::vector<TH1D*> compPtPtrs; for (auto &p: compPtOwned) compPtPtrs.push_back(p.get());
    drawSet("eff_pt", Form("Efficiency vs p_{T} (%s)", matterOpt.c_str()), stdPt.get(), compPtPtrs, "ratio to std", true);

    // ct efficiency
    auto stdCt = cloneHist(pickCt(stdRes), "std_ct");
    std::vector<TH1D*> compCt;
    for (size_t i=0;i<compRes.size();++i) compCt.push_back(pickCt(compRes[i]));
    std::vector<std::unique_ptr<TH1D>> compCtOwned;
    for (size_t i=0;i<compCt.size();++i) compCtOwned.push_back(cloneHist(compCt[i], Form("comp_ct_%zu", i)));
    std::vector<TH1D*> compCtPtrs; for (auto &p: compCtOwned) compCtPtrs.push_back(p.get());
    drawSet("eff_ct", Form("Efficiency vs ct (%s)", matterOpt.c_str()), stdCt.get(), compCtPtrs, "ratio to std", true);

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
        std::string cname = Form("eff_ct_pt_%s_%s", edgeToToken(ptbinsforct[ipt]).c_str(), edgeToToken(ptbinsforct[ipt + 1]).c_str());
        drawSet(cname, ttl, stdH.get(), compsPtrs, "ratio to std", true);
    }

    // pt efficiency per centrality
    auto pickPtPerCent = [&](const AcceptanceHelper::AcceptanceResult &res) -> const std::vector<TH1D*>& {
        if (matterOpt == "antimatter" && !res.acc_pt_per_cent_antimatter.empty()) return res.acc_pt_per_cent_antimatter;
        if (matterOpt == "matter" && !res.acc_pt_per_cent_matter.empty()) return res.acc_pt_per_cent_matter;
        if (!res.acc_pt_per_cent.empty()) return res.acc_pt_per_cent;
        if (!res.acc_pt_per_cent_matter.empty()) return res.acc_pt_per_cent_matter;
        return res.acc_pt_per_cent_antimatter;
    };

    std::vector<const std::vector<TH1D*>*> perCentSets;
    perCentSets.push_back(&pickPtPerCent(stdRes));
    for (auto &r : compRes) perCentSets.push_back(&pickPtPerCent(r));

    const std::vector<Style_t> centMarkers = {20, 21, 22, 23, 33, 34, 29, 47};
    const std::vector<Style_t> centLineStyles = {1, 2, 7, 9, 10, 5, 6, 3};

    double minv = std::numeric_limits<double>::max();
    double maxv = std::numeric_limits<double>::lowest();
    for (size_t iset = 0; iset < perCentSets.size(); ++iset) {
        const auto &vec = *perCentSets[iset];
        for (size_t ic = 0; ic < vec.size(); ++ic) {
            TH1D *h = vec[ic];
            if (!h) continue;
            for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
                const double v = h->GetBinContent(ib);
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
        }
    }
    if (minv == std::numeric_limits<double>::max()) {
        minv = 0.0;
        maxv = 1.0;
    }
    const double yMaxAll = (maxv > 0.0) ? (2.0 * maxv) : 1.0;
    const size_t nCentBins = cenbins.size() > 1 ? (cenbins.size() - 1) : 0;

    auto setLabel = [&](size_t iset) {
        return (iset < labels.size()) ? labels[iset] : std::string(Form("set_%zu", iset));
    };

    const Double_t prevErrorX = gStyle->GetErrorX();
    gStyle->SetErrorX(0.5);

    // 1) nCentBin plots: each plot compares different dataframes in one centrality bin
    for (size_t ic = 0; ic < nCentBins; ++ic) {
        TCanvas cCentBin(Form("eff_pt_centbin_%zu", ic), "Efficiency vs p_{T} by dataframe", kCanvasW, kCanvasH);
        TPad pTop("pTop_centbin", "pTop_centbin", 0.0, 0.30, 1.0, 1.0);
        TPad pBot("pBot_centbin", "pBot_centbin", 0.0, 0.0, 1.0, 0.30);
        pTop.SetBottomMargin(0.02);
        pBot.SetTopMargin(0.02);
        pBot.SetBottomMargin(0.32);
        pTop.Draw();
        pBot.Draw();

        // Top pad: keep only Y error bars for efficiency points.
        gStyle->SetErrorX(0.0);
        pTop.cd();
        TLegend leg(0.14, 0.68, 0.88, 0.88);
        leg.SetNColumns(2);
        leg.SetFillStyle(0);
        leg.SetBorderSize(0);
        leg.SetTextSize(legendTextSize);

        std::vector<std::unique_ptr<TH1D>> owned;
        owned.reserve(perCentSets.size());
        bool firstDrawnCent = false;
        for (size_t iset = 0; iset < perCentSets.size(); ++iset) {
            const auto &vec = *perCentSets[iset];
            TH1D *src = (ic < vec.size()) ? vec[ic] : nullptr;
            auto h = cloneHist(src, Form("h_centbin_%zu_set_%zu", ic, iset));
            if (!h)
                continue;
            const Color_t color = colors[std::min(iset, colors.size() - 1)];
            h->SetStats(false);
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(markers[iset % markers.size()]);
            h->SetLineStyle(1);
            h->SetLineWidth(2);
            h->SetTitle(Form("Efficiency vs p_{T} (Cent %.0f-%.0f%%, %s);p_{T} (GeV/c);Efficiency", cenbins[ic], cenbins[ic + 1], matterOpt.c_str()));
            h->SetMinimum(0.0);
            h->SetMaximum(yMaxAll);
            h->GetXaxis()->SetLabelSize(0.0);
            h->GetXaxis()->SetTitleSize(0.0);

            if (!firstDrawnCent) {
                h->Draw("E1 X0 P");
                h->Draw("SAME HIST L");
                firstDrawnCent = true;
            } else {
                h->Draw("SAME E1 X0 P");
                h->Draw("SAME HIST L");
            }
            leg.AddEntry(h.get(), setLabel(iset).c_str(), "lp");
            owned.push_back(std::move(h));
        }
        if (firstDrawnCent) {
            leg.Draw();

            // Bottom pad: enable X error bars for ratio points.
            gStyle->SetErrorX(0.5);
            pBot.cd();
            std::vector<std::unique_ptr<TH1D>> ratioOwned;
            ratioOwned.reserve(owned.size());
            TH1D *href = nullptr;
            for (auto &h : owned) {
                if (h) {
                    href = h.get();
                    break;
                }
            }
            bool firstRatio = false;
            double ratioMin = std::numeric_limits<double>::max();
            double ratioMax = std::numeric_limits<double>::lowest();
            if (href) {
                for (size_t i = 0; i < owned.size(); ++i) {
                    if (i == 0) {
                        ratioOwned.push_back(nullptr);
                        continue;
                    }
                    if (!owned[i]) {
                        ratioOwned.push_back(nullptr);
                        continue;
                    }
                    if (!hasSameBinning(owned[i].get(), href)) {
                        ratioOwned.push_back(nullptr);
                        continue;
                    }
                    auto r = cloneHist(owned[i].get(), Form("h_ratio_centbin_%zu_set_%zu", ic, i));
                    r->Divide(href);
                    r->SetLineColor(owned[i]->GetLineColor());
                    r->SetMarkerColor(owned[i]->GetMarkerColor());
                    r->SetMarkerStyle(owned[i]->GetMarkerStyle());
                    r->SetLineWidth(3);
                    r->SetTitle(";p_{T} (GeV/c);ratio");
                    r->GetYaxis()->SetTitleSize(0.10);
                    r->GetYaxis()->SetTitleOffset(0.50);
                    r->GetYaxis()->SetLabelSize(0.09);
                    r->GetXaxis()->SetTitleSize(0.12);
                    r->GetXaxis()->SetTitleOffset(1.0);
                    r->GetXaxis()->SetLabelSize(0.10);
                    for (int ib = 1; ib <= r->GetNbinsX(); ++ib) {
                        const double v = r->GetBinContent(ib);
                        if (!std::isfinite(v) || v <= 0.0) continue;
                        ratioMin = std::min(ratioMin, v);
                        ratioMax = std::max(ratioMax, v);
                    }
                    ratioOwned.push_back(std::move(r));
                }
                if (ratioMin == std::numeric_limits<double>::max()) {
                    ratioMin = 0.8;
                    ratioMax = 1.2;
                }
                double lo = std::max(0.0, 0.9 * ratioMin);
                double hi = 1.1 * ratioMax;
                // Keep the y=1 baseline inside visible range.
                lo = std::min(lo, 0.95);
                hi = std::max(hi, 1.05);

                auto ratioFrame = cloneHist(href, Form("h_ratio_frame_centbin_%zu", ic));
                if (ratioFrame) {
                    ratioFrame->Reset("ICES");
                    ratioFrame->SetStats(false);
                    ratioFrame->SetMinimum(lo);
                    ratioFrame->SetMaximum(hi);
                    ratioFrame->SetTitle(";p_{T} (GeV/c);ratio");
                    ratioFrame->GetYaxis()->SetTitleSize(0.10);
                    ratioFrame->GetYaxis()->SetTitleOffset(0.50);
                    ratioFrame->GetYaxis()->SetLabelSize(0.09);
                    ratioFrame->GetXaxis()->SetTitleSize(0.12);
                    ratioFrame->GetXaxis()->SetTitleOffset(1.0);
                    ratioFrame->GetXaxis()->SetLabelSize(0.10);
                    ratioFrame->SetLineColor(kWhite);
                    ratioFrame->SetMarkerColor(kWhite);
                    ratioFrame->Draw("HIST");
                }

                for (auto &r : ratioOwned) {
                    if (!r) continue;
                    r->SetMinimum(lo);
                    r->SetMaximum(hi);
                    if (!firstRatio) {
                        r->Draw("E1 P");
                        firstRatio = true;
                    } else {
                        r->Draw("SAME E1 P");
                    }
                }
                TLine l(href->GetXaxis()->GetXmin(), 1.0, href->GetXaxis()->GetXmax(), 1.0);
                l.SetLineColor(kBlack);
                l.SetLineStyle(2);
                l.SetLineWidth(4);
                l.DrawClone();
            }
            const std::string pdfName = "eff_pt_cen_" + edgeToToken(cenbins[ic]) + "_" + edgeToToken(cenbins[ic + 1]) + ".pdf";
            cCentBin.SaveAs((outputpath + "/" + pdfName).c_str());
        }
    }

    // 2) nDataFrame plots: each plot compares different centrality bins in one dataframe
    for (size_t iset = 0; iset < perCentSets.size(); ++iset) {
        TCanvas cSet(Form("eff_pt_per_cent_set_%zu", iset), "Efficiency vs p_{T} by centrality", kCanvasW, kCanvasH);
        TLegend leg(0.14, 0.64, 0.88, 0.88);
        leg.SetNColumns(2);
        leg.SetFillStyle(0);
        leg.SetBorderSize(0);
        leg.SetTextSize(legendTextSize);

        std::vector<std::unique_ptr<TH1D>> owned;
        const auto &vec = *perCentSets[iset];
        owned.reserve(vec.size());
        const Color_t color = colors[std::min(iset, colors.size() - 1)];
        bool firstDrawnSet = false;
        for (size_t ic = 0; ic < vec.size(); ++ic) {
            auto h = cloneHist(vec[ic], Form("h_set_%zu_cent_%zu", iset, ic));
            if (!h)
                continue;
            h->SetStats(false);
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(centMarkers[ic % centMarkers.size()]);
            h->SetLineStyle(centLineStyles[ic % centLineStyles.size()]);
            h->SetLineWidth(2);
            h->SetTitle(Form("Efficiency vs p_{T} (%s, %s);p_{T} (GeV/c);Efficiency", setLabel(iset).c_str(), matterOpt.c_str()));
            h->SetMinimum(0.0);
            h->SetMaximum(yMaxAll);

            if (!firstDrawnSet) {
                h->Draw("E1 X0 P");
                h->Draw("SAME HIST L");
                firstDrawnSet = true;
            } else {
                h->Draw("SAME E1 X0 P");
                h->Draw("SAME HIST L");
            }
            if (ic + 1 < cenbins.size())
                leg.AddEntry(h.get(), Form("%.0f-%.0f%%", cenbins[ic], cenbins[ic + 1]), "lp");
            else
                leg.AddEntry(h.get(), Form("cent bin %zu", ic), "lp");
            owned.push_back(std::move(h));
        }
        if (firstDrawnSet) {
            leg.Draw();
            const std::string labelSuffix = sanitizeLabel(setLabel(iset));
            cSet.SaveAs((outputpath + "/eff_pt_per_cent_" + labelSuffix + ".pdf").c_str());
        }
    }

    // 3) all points in one combined plot
    gStyle->SetErrorX(0.0); // Combined efficiency plot keeps only Y error bars.
    TCanvas cCent("eff_pt_per_cent_combined", "Efficiency vs p_{T} per centrality", kCanvasW, kCanvasH);

    std::vector<std::vector<std::unique_ptr<TH1D>>> ownedPerCent(perCentSets.size());
    bool firstDrawn = false;
    TLegend leg(0.12, 0.58, 0.9, 0.88);
    leg.SetNColumns(2);
    leg.SetFillStyle(0);
    leg.SetBorderSize(0);
    leg.SetTextSize(legendTextSize);

    for (size_t iset = 0; iset < perCentSets.size(); ++iset) {
        const auto &vec = *perCentSets[iset];
        ownedPerCent[iset].reserve(vec.size());
        const Color_t color = colors[std::min(iset, colors.size() - 1)];
        std::string setLabel = (iset < labels.size()) ? labels[iset] : Form("set_%zu", iset);
        for (size_t ic = 0; ic < vec.size(); ++ic) {
            auto hc = cloneHist(vec[ic], Form("pt_per_cent_set%zu_cent%zu", iset, ic));
            if (!hc) {
                ownedPerCent[iset].push_back(nullptr);
                continue;
            }
            hc->SetStats(false);
            hc->SetLineColor(color);
            hc->SetMarkerColor(color);
            hc->SetMarkerStyle(centMarkers[ic % centMarkers.size()]);
            hc->SetLineStyle(centLineStyles[ic % centLineStyles.size()]);
            hc->SetLineWidth(2);
            hc->SetTitle(Form("Efficiency vs p_{T} per centrality (%s);p_{T} (GeV/c);Efficiency", matterOpt.c_str()));
            hc->SetMinimum(0.0);
            hc->SetMaximum(yMaxAll);
            if (!firstDrawn) {
                hc->Draw("E1 X0 P");
                hc->Draw("SAME HIST L");
                firstDrawn = true;
            } else {
                hc->Draw("SAME E1 X0 P");
                hc->Draw("SAME HIST L");
            }
            if (ic + 1 < cenbins.size()) {
                leg.AddEntry(hc.get(), Form("%s, %.0f-%.0f%%", setLabel.c_str(), cenbins[ic], cenbins[ic + 1]), "lep");
            } else {
                leg.AddEntry(hc.get(), Form("%s, cent bin %zu", setLabel.c_str(), ic), "lep");
            }
            ownedPerCent[iset].push_back(std::move(hc));
        }
    }

    if (firstDrawn) {
        leg.Draw();
        cCent.SaveAs((outputpath + "/eff_pt_per_cent_combined.pdf").c_str());
    }

    gStyle->SetEndErrorSize(prevEndErrorSize);
    gStyle->SetErrorX(prevErrorX);
}