#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TLegend.h>
#include <TLine.h>
#include <TH1D.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Tools/EventSignalLossHelper.h"

std::string EventSignalLossCentTag(double low, double high) {
    return Form("%.0f_%.0f", low, high);
}

std::string EventSignalLossCentLabel(double low, double high) {
    return Form("%.0f-%.0f%%", low, high);
}

void StyleHist(TH1D *h, int color, int marker, int lineStyle = 1) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.0);
    h->SetLineStyle(lineStyle);
    h->SetLineWidth(2);
}

void SetRangeFromHists(TH1D *frameHist, const std::vector<TH1D *> &hists, double floor = 0.0, double fallbackMax = 1.2) {
    if (!frameHist) return;
    double ymax = 0.0;
    for (auto *h : hists) {
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double y = h->GetBinContent(ib);
            const double e = h->GetBinError(ib);
            if (std::isfinite(y)) ymax = std::max(ymax, y + e);
        }
    }
    frameHist->GetYaxis()->SetRangeUser(floor, ymax > floor ? ymax * 1.25 : fallbackMax);
}

std::vector<int> MakeRainbowColors(size_t n) {
    std::vector<int> colors;
    colors.reserve(n);
    gStyle->SetPalette(kRainBow);
    const int nPalette = std::max(1, gStyle->GetNumberOfColors());
    for (size_t i = 0; i < n; ++i) {
        const double frac = (n <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
        colors.push_back(TColor::GetColorPalette(static_cast<int>(frac * (nPalette - 1))));
    }
    return colors;
}

std::unique_ptr<TH1D> MakeFinalCorrectionHist(const char *name,
                                               TH1D *signalLoss,
                                               double eventLoss,
                                               double eventLossErr,
                                               double eventSplitting,
                                               double eventSplittingErr) {
    auto out = std::unique_ptr<TH1D>(static_cast<TH1D *>(signalLoss->Clone(name)));
    out->SetDirectory(nullptr);
    out->Reset("ICES");
    const auto eventFactor = EventSignalLossHelper::RatioWithError(eventLoss, eventLossErr, eventSplitting, eventSplittingErr);
    for (int ib = 1; ib <= signalLoss->GetNbinsX(); ++ib) {
        const double sig = signalLoss->GetBinContent(ib);
        const double sigErr = signalLoss->GetBinError(ib);
        const auto finalFactor = EventSignalLossHelper::RatioWithError(eventFactor.first, eventFactor.second, sig, sigErr);
        out->SetBinContent(ib, finalFactor.first);
        out->SetBinError(ib, finalFactor.second);
    }
    return out;
}

int EventSignalLossCheck(
    const char *eventSignalLossPath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/EventLoss/AnalysisResults.root",
    const std::vector<double> &cenBins = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
    const std::vector<std::vector<double>> &ptBinsByCent = {
        {2, 3, 3.5, 4, 4.5, 5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 5, 8}},
    const char *outDir = "") {
    try {
        if (!eventSignalLossPath || std::string(eventSignalLossPath).empty()) {
            throw std::runtime_error("EventSignalLossCheck: input file path is empty");
        }
        if (ptBinsByCent.size() != cenBins.size() - 1) {
            throw std::runtime_error("EventSignalLossCheck: ptBinsByCent size must match centrality bins");
        }

        const std::string outputDir = (outDir && std::string(outDir).size() > 0)
                                          ? outDir
                                          : "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/EventSignalLoss";
        std::filesystem::create_directories(outputDir);

        auto evRes = EventSignalLossHelper::ComputeEventLoss(eventSignalLossPath, cenBins);
        auto sigRes = EventSignalLossHelper::ComputeSignalLossCenPt(eventSignalLossPath, cenBins, ptBinsByCent);

        const std::string outRootPath = outputDir + "/event_signal_loss_check.root";
        TFile outRoot(outRootPath.c_str(), "RECREATE");
        if (outRoot.IsZombie()) {
            throw std::runtime_error("EventSignalLossCheck: failed to create output ROOT file");
        }

        StyleHist(evRes.hMultiplicity, kAzure + 2, 20, 1);
        StyleHist(evRes.hImpact, kRed + 1, 24, 2);
        StyleHist(evRes.hEventSplitting, kGreen + 2, 21, 1);

        {
            TCanvas c("c_event_loss_vs_centrality", "Event loss vs centrality", 900, 700);
            c.SetLeftMargin(0.12);
            c.SetBottomMargin(0.12);
            evRes.hMultiplicity->SetTitle("Event Loss vs Centrality;Centrality (%);Event loss");
            SetRangeFromHists(evRes.hMultiplicity, {evRes.hMultiplicity, evRes.hImpact}, 0.0, 1.2);
            evRes.hMultiplicity->Draw("E1");
            evRes.hImpact->Draw("E1 SAME");
            TLegend leg(0.55, 0.74, 0.88, 0.88);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            leg.AddEntry(evRes.hMultiplicity, "Multiplicity, #eta < 0.8", "lep");
            leg.AddEntry(evRes.hImpact, "Impact parameter", "lep");
            leg.Draw();
            c.Write();
            c.SaveAs((outputDir + "/event_loss_vs_centrality.pdf").c_str());
        }

        {
            TCanvas c("c_event_splitting_vs_centrality", "Event splitting vs centrality", 900, 700);
            c.SetLeftMargin(0.12);
            c.SetBottomMargin(0.12);
            evRes.hEventSplitting->SetTitle("Event Splitting vs Centrality;Centrality (%);Event splitting");
            SetRangeFromHists(evRes.hEventSplitting, {evRes.hEventSplitting}, 0.0, 1.2);
            evRes.hEventSplitting->Draw("E1");
            TLine unity(cenBins.front(), 1.0, cenBins.back(), 1.0);
            unity.SetLineStyle(2);
            unity.SetLineColor(kGray + 2);
            unity.Draw("SAME");
            c.Write();
            c.SaveAs((outputDir + "/event_splitting_vs_centrality.pdf").c_str());
        }

        const std::vector<int> colors = MakeRainbowColors(cenBins.size() - 1);
        std::vector<TH1D *> hSigMultAll;
        std::vector<TH1D *> hSigImpactAll;
        std::vector<TH1D *> hFinalMultAll;
        std::vector<TH1D *> hFinalImpactAll;
        std::vector<std::unique_ptr<TH1D>> ownedFinalHists;

        for (size_t ic = 0; ic + 1 < cenBins.size(); ++ic) {
            const std::string tag = EventSignalLossCentTag(cenBins[ic], cenBins[ic + 1]);
            const std::string label = EventSignalLossCentLabel(cenBins[ic], cenBins[ic + 1]);
            TH1D *hSigMult = sigRes.multiplicity_pt_per_cent[ic];
            TH1D *hSigImpact = sigRes.impact_pt_per_cent[ic];
            if (!hSigMult || !hSigImpact) continue;

            hSigMult->SetName(Form("h_signal_loss_multiplicity_cen_%s", tag.c_str()));
            hSigImpact->SetName(Form("h_signal_loss_impact_cen_%s", tag.c_str()));
            hSigMult->SetTitle(Form("Signal Loss %.0f-%.0f%%;#it{p}_{T} (GeV/#it{c});Signal loss", cenBins[ic], cenBins[ic + 1]));
            hSigImpact->SetTitle(hSigMult->GetTitle());
            StyleHist(hSigMult, kAzure + 2, 20, 1);
            StyleHist(hSigImpact, kRed + 1, 24, 2);

            auto hFinalMult = MakeFinalCorrectionHist(Form("h_final_event_signal_correction_multiplicity_cen_%s", tag.c_str()),
                                                       hSigMult,
                                                       evRes.multiplicityValue[ic],
                                                       evRes.multiplicityError[ic],
                                                       evRes.eventSplittingValue[ic],
                                                       evRes.eventSplittingError[ic]);
            auto hFinalImpact = MakeFinalCorrectionHist(Form("h_final_event_signal_correction_impact_cen_%s", tag.c_str()),
                                                         hSigImpact,
                                                         evRes.impactValue[ic],
                                                         evRes.impactError[ic],
                                                         evRes.eventSplittingValue[ic],
                                                         evRes.eventSplittingError[ic]);
            hFinalMult->SetTitle(Form("Final Event/Signal Correction %.0f-%.0f%%;#it{p}_{T} (GeV/#it{c});Event loss / event splitting / signal loss", cenBins[ic], cenBins[ic + 1]));
            hFinalImpact->SetTitle(hFinalMult->GetTitle());
            StyleHist(hFinalMult.get(), kAzure + 2, 20, 1);
            StyleHist(hFinalImpact.get(), kRed + 1, 24, 2);

            hSigMultAll.push_back(hSigMult);
            hSigImpactAll.push_back(hSigImpact);
            hFinalMultAll.push_back(hFinalMult.get());
            hFinalImpactAll.push_back(hFinalImpact.get());

            {
                TCanvas c(Form("c_signal_loss_cen_%s", tag.c_str()), "", 900, 700);
                c.SetLeftMargin(0.12);
                c.SetBottomMargin(0.12);
                SetRangeFromHists(hSigMult, {hSigMult, hSigImpact}, 0.0, 1.2);
                hSigMult->Draw("E1");
                hSigImpact->Draw("E1 SAME");
                TLegend leg(0.55, 0.74, 0.88, 0.88);
                leg.SetBorderSize(0);
                leg.SetFillStyle(0);
                leg.AddEntry(hSigMult, "Multiplicity, #eta < 0.8", "lep");
                leg.AddEntry(hSigImpact, "Impact parameter", "lep");
                leg.Draw();
                c.Write();
                c.SaveAs((outputDir + "/signal_loss_pt_cen_" + tag + ".pdf").c_str());
            }

            {
                TCanvas c(Form("c_final_event_signal_correction_cen_%s", tag.c_str()), "", 900, 700);
                c.SetLeftMargin(0.12);
                c.SetBottomMargin(0.12);
                SetRangeFromHists(hFinalMult.get(), {hFinalMult.get(), hFinalImpact.get()}, 0.0, 1.5);
                hFinalMult->Draw("E1");
                hFinalImpact->Draw("E1 SAME");
                TLegend leg(0.50, 0.72, 0.88, 0.88);
                leg.SetBorderSize(0);
                leg.SetFillStyle(0);
                leg.AddEntry(hFinalMult.get(), "Multiplicity, #eta < 0.8", "lep");
                leg.AddEntry(hFinalImpact.get(), "Impact parameter", "lep");
                leg.Draw();
                c.Write();
                c.SaveAs((outputDir + "/final_event_signal_correction_cen_" + tag + ".pdf").c_str());
            }

            hSigMult->Write();
            hSigImpact->Write();
            hFinalMult->Write();
            hFinalImpact->Write();
            ownedFinalHists.push_back(std::move(hFinalMult));
            ownedFinalHists.push_back(std::move(hFinalImpact));

            std::cout << Form("cent %s | event loss mult %.5f | impact %.5f | event splitting %.5f",
                              label.c_str(),
                              evRes.multiplicityValue[ic],
                              evRes.impactValue[ic],
                              evRes.eventSplittingValue[ic])
                      << std::endl;
        }

        auto drawAll = [&](const char *canvasName,
                           const char *title,
                           const char *yTitle,
                           const std::string &pdfName,
                           const std::vector<TH1D *> &hMultVec,
                           const std::vector<TH1D *> &hImpactVec) {
            TCanvas c(canvasName, title, 1000, 800);
            c.SetLeftMargin(0.12);
            c.SetRightMargin(0.04);
            c.SetBottomMargin(0.12);
            c.SetTopMargin(0.08);
            TLegend legCent(0.58, 0.18, 0.94, 0.72);
            legCent.SetBorderSize(0);
            legCent.SetFillStyle(0);
            legCent.SetTextSize(0.030);
            legCent.SetNColumns(2);
            TLegend legMethod(0.17, 0.18, 0.48, 0.31);
            legMethod.SetBorderSize(0);
            legMethod.SetFillStyle(0);
            legMethod.SetTextSize(0.034);
            bool first = true;
            std::vector<TH1D *> allHists;
            allHists.insert(allHists.end(), hMultVec.begin(), hMultVec.end());
            allHists.insert(allHists.end(), hImpactVec.begin(), hImpactVec.end());
            for (size_t ic = 0; ic < hMultVec.size(); ++ic) {
                TH1D *hMult = hMultVec[ic];
                TH1D *hImpact = ic < hImpactVec.size() ? hImpactVec[ic] : nullptr;
                if (!hMult || !hImpact) continue;
                const int color = colors[ic % colors.size()];
                StyleHist(hMult, color, 20, 1);
                StyleHist(hImpact, color, 24, 2);
                hMult->SetTitle(Form("%s;#it{p}_{T} (GeV/#it{c});%s", title, yTitle));
                SetRangeFromHists(hMult, allHists, 0.0, 1.2);
                hMult->Draw(first ? "E1" : "E1 SAME");
                hImpact->Draw("E1 SAME");
                first = false;
                if (ic + 1 < cenBins.size()) legCent.AddEntry(hMult, EventSignalLossCentLabel(cenBins[ic], cenBins[ic + 1]).c_str(), "lep");
            }
            TH1D hMethodMult("h_method_mult_legend", "", 1, 0.0, 1.0);
            TH1D hMethodImpact("h_method_impact_legend", "", 1, 0.0, 1.0);
            StyleHist(&hMethodMult, kBlack, 20, 1);
            StyleHist(&hMethodImpact, kBlack, 24, 2);
            legMethod.AddEntry(&hMethodMult, "Multiplicity, #eta < 0.8", "lep");
            legMethod.AddEntry(&hMethodImpact, "Impact parameter", "lep");
            legMethod.Draw();
            legCent.Draw();
            c.Write();
            c.SaveAs((outputDir + "/" + pdfName).c_str());
        };

        drawAll("c_signal_loss_all_centralities",
                "Signal Loss for all centrality bins",
                "Signal loss",
                "signal_loss_pt_all_centralities.pdf",
                hSigMultAll,
                hSigImpactAll);
        drawAll("c_final_event_signal_correction_all_centralities",
                "Final Event/Signal Correction for all centrality bins",
                "Event loss / event splitting / signal loss",
                "final_event_signal_correction_all_centralities.pdf",
                hFinalMultAll,
                hFinalImpactAll);

        evRes.hMultiplicity->Write("h_event_loss_multiplicity");
        evRes.hImpact->Write("h_event_loss_impact");
        evRes.hEventSplitting->Write("h_event_splitting");
        outRoot.Close();

        sigRes.Clear();
        evRes.Clear();

        std::cout << "[Info] Output ROOT saved: " << outRootPath << std::endl;
        std::cout << "[Info] Output PDF dir: " << outputDir << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
}
