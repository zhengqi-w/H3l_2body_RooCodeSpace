#ifndef SPECTRUM_PLOT_HELPER_H
#define SPECTRUM_PLOT_HELPER_H

#include <TCanvas.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <RooPlot.h>
#include <TObject.h>
#include <TLine.h>
#include <TBox.h>
#include <TF1.h>
#include <TMath.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace UnifiedAnalysis {

struct PlotLabelConfig {
    bool usePerformance{false};
    std::string performanceLabel;
    std::string period;
    std::string periodMark;
    std::string collisionSystem;
    std::string collisionEnergy;
};

struct TrailAnnotationInfo {
    std::string bkgFunc;
    std::string sigFunc;
    double bdtScore{0.0};
    double bdtEff{0.0};
    bool enabled{false};
};

inline std::string BuildDecayString(const std::string &isMatter) {
    if (isMatter == "matter") {
        return "{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}";
    }
    if (isMatter == "antimatter") {
        return "{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}";
    }
    if (isMatter == "both") {
        return "{}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}He + #pi^{-} (^{3}#bar{He} + #pi^{+})";
    }
    return std::string();
}

inline std::string BuildCentralityText(const std::string &tag) {
    if (tag.rfind("cen_", 0) == 0 && tag.size() > 4) {
        std::string range = tag.substr(4);
        for (char &ch : range) {
            if (ch == '_') ch = '-';
        }
        return "CentralityFT0C " + range + "%";
    }
    return std::string();
}

inline std::string BuildEventLine(double nEvents) {
    std::ostringstream os;
    os << "N_{ev} = " << std::scientific << std::setprecision(2) << nEvents;
    return os.str();
}

inline std::vector<std::string> BuildExperimentLines(const PlotLabelConfig &cfg) {
    std::vector<std::string> lines;
    std::string coll;
    if (!cfg.collisionSystem.empty()) coll += cfg.collisionSystem;
    if (!cfg.collisionEnergy.empty()) {
        if (!coll.empty()) coll += " ";
        coll += cfg.collisionEnergy;
    }

    if (cfg.usePerformance) {
        if (!cfg.performanceLabel.empty()) lines.push_back(cfg.performanceLabel);
    } else {
        std::string periodLine;
        if (!cfg.period.empty()) periodLine += cfg.period;
        if (!cfg.periodMark.empty()) {
            if (!periodLine.empty()) periodLine += "_";
            periodLine += cfg.periodMark;
        }
        if (!periodLine.empty()) lines.push_back(periodLine);
    }
    if (!coll.empty()) lines.push_back(coll);
    return lines;
}

inline TObject *FindPlotObject(RooPlot *frame, const char *name) {
    if (!frame || !name) return nullptr;
    return frame->findObject(name);
}

inline std::unique_ptr<TLegend> MakeFitLegend(RooPlot *frame, bool isMc) {
    if (!frame) return nullptr;
    const auto entries = isMc ? std::vector<std::tuple<const char *, const char *, const char *>>{
                                    {"mc", "MC", "l"},
                                    {"sig_fit_mc", "Signal (MC)", "l"}}
                              : std::vector<std::tuple<const char *, const char *, const char *>>{
                                    {"data", "Data", "lep"},
                                    {"total", "Total fit", "l"},
                                    {"bkg", "Background", "l"},
                                    {"sig", "Signal", "l"}};

    auto legend = std::make_unique<TLegend>(0.14, 0.50, 0.50, 0.70);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.035);

    bool added = false;
    for (const auto &[name, label, option] : entries) {
        if (auto *obj = FindPlotObject(frame, name)) {
            legend->AddEntry(obj, label, option);
            added = true;
        }
    }
    if (!added) return nullptr;
    return legend;
}

inline void DrawHeaderText(const PlotLabelConfig &cfg,
                           const std::string &isMatter,
                           double nEvents) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.035);
    latex.SetTextAlign(11);
    latex.SetTextColor(kBlack);

    double x = 0.15;
    double y = 0.85;
    const auto expLines = BuildExperimentLines(cfg);
    for (const auto &line : expLines) {
        latex.DrawLatex(x, y, line.c_str());
        y -= 0.045;
    }

    const std::string decay = BuildDecayString(isMatter);
    if (!decay.empty()) {
        latex.DrawLatex(x, y, decay.c_str());
        y -= 0.045;
    }

    const std::string nev = BuildEventLine(nEvents);
    latex.DrawLatex(x, y, nev.c_str());
}

inline void DrawTrailAnnotation(const TrailAnnotationInfo &trail) {
    if (!trail.enabled) return;
    auto pave = std::make_unique<TPaveText>(0.58, 0.12, 0.88, 0.28, "NDC");
    pave->SetBorderSize(0);
    pave->SetFillStyle(0);
    pave->SetTextAlign(12);
    pave->SetTextFont(42);
    pave->SetTextSize(0.03);
    pave->AddText(Form("Bkg: %s", trail.bkgFunc.c_str()));
    pave->AddText(Form("Sig: %s", trail.sigFunc.c_str()));
    pave->AddText(Form("BDT cut: %.4f", trail.bdtScore));
    pave->AddText(Form("BDT eff: %.3f", trail.bdtEff));
    pave->DrawClone();
}

inline std::unique_ptr<TCanvas> MakeDecoratedFitCanvas(const std::string &name,
                                                        RooPlot *frame,
                                                        bool isMc,
                                                        const PlotLabelConfig &labelCfg,
                                                        const std::string &isMatter,
                                                        double nEvents,
                                                        const TrailAnnotationInfo &trail = TrailAnnotationInfo{}) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!frame) return c;

    c->cd();
    c->SetTicks(1, 1);
    frame->Draw();
    auto legend = MakeFitLegend(frame, isMc);
    if (legend) legend->DrawClone();
    DrawHeaderText(labelCfg, isMatter, nEvents);
    DrawTrailAnnotation(trail);
    c->Modified();
    c->Update();
    return c;
}

inline std::unique_ptr<TCanvas> MakeSystematicsCorrDistCanvas(const std::string &name,
                                                              TH1D *hCorrDist,
                                                              double stdCorr,
                                                              double stdCorrErr,
                                                              double cenMin,
                                                              double cenMax,
                                                              const std::string &binLabel) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 900, 700);
    if (!hCorrDist) return c;

    c->cd();
    hCorrDist->SetStats(false);
    hCorrDist->SetTitle(binLabel.c_str());
    hCorrDist->SetLineColor(kBlack);
    hCorrDist->SetLineWidth(2);
    hCorrDist->Draw("HIST");

    TLine *lineCorr = nullptr;
    TBox *bandCorr = nullptr;
    TBox *bandGauss = nullptr;
    TF1 *fgaus = nullptr;
    double gausMean = std::numeric_limits<double>::quiet_NaN();
    double gausSigma = std::numeric_limits<double>::quiet_NaN();

    if (std::isfinite(stdCorr)) {
        const double ymax = std::max(1.0, hCorrDist->GetMaximum()) * 1.02;
        lineCorr = new TLine(stdCorr, 0.0, stdCorr, ymax);
        lineCorr->SetLineColor(kRed + 2);
        lineCorr->SetLineWidth(2);

        if (std::isfinite(stdCorrErr) && stdCorrErr > 0.0) {
            bandCorr = new TBox(stdCorr - stdCorrErr, 0.0, stdCorr + stdCorrErr, ymax);
            bandCorr->SetFillStyle(3004);
            bandCorr->SetFillColor(kOrange - 2);
            bandCorr->SetLineColor(kRed + 2);
        }
    }

    if (hCorrDist->GetEntries() > 2) {
        const double initMean = hCorrDist->GetMean();
        const double initSigma = std::max(hCorrDist->GetRMS(), 1e-9);
        TF1 gausFunc("gaus", "gaus", hCorrDist->GetXaxis()->GetXmin(), hCorrDist->GetXaxis()->GetXmax());
        gausFunc.SetParameters(hCorrDist->GetMaximum(), initMean, initSigma);
        auto fitRes = hCorrDist->Fit(&gausFunc, "QS");
        fgaus = hCorrDist->GetFunction("gaus");
        if (fitRes == 0 && fgaus) {
            fgaus->SetLineColor(kGreen + 3);
            fgaus->SetLineWidth(2);
            fgaus->SetLineStyle(kSolid);
            gausMean = fgaus->GetParameter(1);
            gausSigma = std::abs(fgaus->GetParameter(2));
            bandGauss = new TBox(gausMean - gausSigma, 0.0, gausMean + gausSigma,
                                 std::max(1.0, hCorrDist->GetMaximum()) * 1.02);
            bandGauss->SetFillStyle(3005);
            bandGauss->SetFillColor(kGreen + 1);
            bandGauss->SetLineColor(kGreen + 3);
        }
    }

    if (bandCorr) bandCorr->Draw("same");
    if (bandGauss) bandGauss->Draw("same");
    if (fgaus) fgaus->Draw("same");
    if (lineCorr) lineCorr->Draw("same");
    hCorrDist->Draw("HIST SAME");

    auto leg = std::make_unique<TLegend>(0.58, 0.70, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->AddEntry(hCorrDist, "Trails passing cuts", "l");
    if (lineCorr) leg->AddEntry(lineCorr, "Std value", "l");
    if (bandCorr) leg->AddEntry(bandCorr, "Std stat band", "f");
    if (bandGauss) leg->AddEntry(bandGauss, "Gauss #pm1#sigma", "f");
    if (fgaus) leg->AddEntry(fgaus, "Gauss fit", "l");
    leg->DrawClone();

    auto pave = std::make_unique<TPaveText>(0.14, 0.68, 0.47, 0.90, "NDC");
    pave->SetBorderSize(0);
    pave->SetFillStyle(0);
    pave->SetTextAlign(12);
    pave->SetTextFont(42);
    pave->AddText(Form("Cen: %.0f-%.0f%%", cenMin, cenMax));
    if (std::isfinite(gausMean) && std::isfinite(gausSigma)) {
        pave->AddText(Form("Gauss #mu = %.3e", gausMean));
        pave->AddText(Form("Gauss #sigma = %.3e", gausSigma));
    } else {
        pave->AddText("Gauss fit: n/a");
    }
    pave->AddText(Form("RMS = %.3e", hCorrDist->GetRMS()));
    pave->AddText(Form("Central = %.3e", hCorrDist->GetMean()));
    pave->DrawClone();

    c->Modified();
    c->Update();
    return c;
}

inline std::unique_ptr<TCanvas> MakeSystematicsDetailCanvas(const std::string &name,
                                                             TH1D *hSel,
                                                             TH1D *hAbso,
                                                             TH1D *hBr,
                                                             TH1D *hTot) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!hSel || !hAbso || !hTot) return c;

    hSel->SetStats(false);
    hAbso->SetStats(false);
    if (hBr) hBr->SetStats(false);
    hTot->SetStats(false);
    hTot->SetLineColor(kBlack);
    hTot->SetLineWidth(3);
    hSel->SetLineColor(kRed + 1);
    hSel->SetLineWidth(2);
    hAbso->SetLineColor(kAzure + 2);
    hAbso->SetLineWidth(2);
    if (hBr) {
        hBr->SetLineColor(kGreen + 2);
        hBr->SetLineWidth(2);
    }

    hTot->Draw("HIST");
    hSel->Draw("HIST SAME");
    hAbso->Draw("HIST SAME");
    if (hBr) hBr->Draw("HIST SAME");

    auto leg = std::make_unique<TLegend>(0.62, 0.70, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.035);
    leg->AddEntry(hTot, "Total sys", "l");
    leg->AddEntry(hSel, "Selection sys", "l");
    leg->AddEntry(hAbso, "Absorption sys", "l");
    if (hBr) leg->AddEntry(hBr, "BR sys", "l");
    leg->DrawClone();
    c->SetGrid();
    return c;
}

inline std::unique_ptr<TCanvas> MakeFinalSpectrumCanvas(const std::string &name,
                                                        TH1D *hStat,
                                                        TGraphAsymmErrors *gSys,
                                                        TF1 *fStdFit = nullptr,
                                                        const std::string &xTitle = "",
                                                        const std::string &extraText = "",
                                                        const PlotLabelConfig &labelCfg = PlotLabelConfig{},
                                                        const std::string &isMatter = "",
                                                        double nEvents = 0.0) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!hStat) return c;

    c->cd();
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.12);
    c->SetRightMargin(0.05);
    c->SetTopMargin(0.08);
    c->SetTicks(1, 1);
    c->SetGridy(false);
    c->SetLogy();

    const bool isPtSpectrum = (extraText.rfind("cen_", 0) == 0);

    hStat->SetStats(false);
    hStat->SetMarkerStyle(20);
    hStat->SetMarkerSize(1.0);
    hStat->SetMarkerColor(kBlack);
    hStat->SetLineColor(kBlack);
    hStat->SetLineWidth(2);
    hStat->SetMinimum(std::max(1e-12, hStat->GetMinimum(1) * 0.5));
    hStat->SetTitle("");
    hStat->GetXaxis()->SetTitle(xTitle.empty() ? "x" : xTitle.c_str());
    hStat->GetYaxis()->SetTitle(isPtSpectrum ? "#frac{1}{N_{ev}} #frac{d^{2}N}{d#it{p}_{T}d#it{y}} ((GeV/#it{c})^{-1})"
                                             : "Corrected counts");
    hStat->GetXaxis()->SetTitleSize(0.05);
    hStat->GetYaxis()->SetTitleSize(0.05);
    hStat->GetYaxis()->SetTitleOffset(1.35);
    if (isPtSpectrum) {
        hStat->GetXaxis()->SetRangeUser(0.0, hStat->GetXaxis()->GetXmax());
    }
    hStat->Draw("E1 X0");

    if (gSys) {
        const int n = gSys->GetN();
        for (int i = 0; i < n; ++i) {
            double x = 0.0;
            double y = 0.0;
            gSys->GetPoint(i, x, y);
            const double exl = gSys->GetErrorXlow(i);
            const double exh = gSys->GetErrorXhigh(i);
            const double eyl = gSys->GetErrorYlow(i);
            const double eyh = gSys->GetErrorYhigh(i);
            if (!std::isfinite(x) || !std::isfinite(y)) continue;
            const double yMin = std::max(1e-20, y - eyl);
            const double yMax = std::max(yMin, y + eyh);
            TBox box(x - exl, yMin, x + exh, yMax);
            box.SetFillStyle(0);
            box.SetLineColor(kBlue + 2);
            box.SetLineWidth(2);
            box.SetLineStyle(kDashed);
            box.DrawClone("l");
        }
    }

    if (fStdFit) {
        fStdFit->SetLineColor(kRed + 1);
        fStdFit->SetLineWidth(3);
        fStdFit->Draw("SAME");
    }

    // Draw all text info at bottom left
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.035);
    latex.SetTextAlign(11);
    double textY = 0.33;
    const auto expLines = BuildExperimentLines(labelCfg);
    for (const auto &line : expLines) {
        latex.DrawLatex(0.16, textY, line.c_str());
        textY -= 0.045;
    }
    const std::string decay = BuildDecayString(isMatter);
    if (!decay.empty()) {
        latex.DrawLatex(0.16, textY, decay.c_str());
        textY -= 0.045;
    }
    const std::string cenText = BuildCentralityText(extraText);
    if (!cenText.empty()) {
        latex.DrawLatex(0.16, textY, cenText.c_str());
        textY -= 0.045;
    }
    latex.DrawLatex(0.16, textY, BuildEventLine(nEvents).c_str());

    // Draw Legend above text block
    TLegend *leg = new TLegend(0.16, 0.40, 0.50, 0.58);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.035);
    leg->AddEntry(hStat, "Data points", "p");
    if (gSys) {
        TLine *sysLine = new TLine();
        sysLine->SetLineColor(kBlue + 2);
        sysLine->SetLineStyle(kDashed);
        sysLine->SetLineWidth(2);
        leg->AddEntry(sysLine, "Sys. box", "l");
    }
    if (fStdFit) {
        leg->AddEntry(fStdFit, isPtSpectrum ? "Blast-Wave fit" : "Std fit", "l");
    }
    leg->Draw();

    return c;
}

inline std::unique_ptr<TH1D> MakeRawOverNevtsHist(const std::string &name,
                                                  double rawSum,
                                                  double rawErr,
                                                  double nEvents,
                                                  const std::string &label) {
    auto h = std::make_unique<TH1D>(name.c_str(), ";bin;N_{raw}/N_{ev}", 1, 0.5, 1.5);
    h->SetDirectory(nullptr);
    h->SetStats(false);
    const double val = (nEvents > 0.0) ? rawSum / nEvents : 0.0;
    const double err = (nEvents > 0.0) ? rawErr / nEvents : 0.0;
    h->SetBinContent(1, val);
    h->SetBinError(1, err);
    h->GetXaxis()->SetBinLabel(1, label.c_str());
    return h;
}

inline std::unique_ptr<TCanvas> MakeRawOverNevtsCanvas(const std::string &name,
                                                       TH1D *hist,
                                                       double rawSum,
                                                       double rawErr,
                                                       double nEvents,
                                                       const std::string &label,
                                                       const std::string &periodTag) {
    if (!hist) return nullptr;
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 900, 700);
    c->cd();
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.12);
    c->SetTicks(1, 1);
    hist->SetMarkerStyle(20);
    hist->SetMarkerColor(kBlue + 1);
    hist->SetLineColor(kBlue + 1);
    hist->GetYaxis()->SetTitleOffset(1.4);
    hist->Draw("E1");

    const double x = 0.20;
    double y = 0.80;
    auto addLine = [&](const std::string &text, double yPos) {
        auto *t = new TLatex(x, yPos, text.c_str());
        t->SetNDC();
        t->SetTextSize(0.04);
        t->SetTextFont(42);
        t->Draw();
    };
    addLine(label, y);
    y -= 0.06;
    if (!periodTag.empty()) {
        addLine(periodTag, y);
        y -= 0.06;
    }
    std::ostringstream os;
    os << "N_{ev} = " << nEvents;
    addLine(os.str(), y);
    y -= 0.06;
    os.str("");
    os << "#Sigma raw = " << rawSum;
    addLine(os.str(), y);
    y -= 0.06;
    os.str("");
    const double ratio = (nEvents > 0.0) ? rawSum / nEvents : 0.0;
    const double ratioErr = (nEvents > 0.0) ? rawErr / nEvents : 0.0;
    os << "N_{raw}/N_{ev} = " << ratio << " #pm " << ratioErr;
    addLine(os.str(), y);
    c->Modified();
    c->Update();
    return c;
}

inline std::unique_ptr<TCanvas> MakeBlastWaveFitCanvas(const std::string &name,
                                                       TH1D *hCorr,
                                                       TF1 *fBw,
                                                       double cenMin,
                                                       double cenMax) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!hCorr || !fBw) return c;

    auto hDraw = std::unique_ptr<TH1D>(static_cast<TH1D *>(hCorr->Clone((name + "_hdraw").c_str())));
    if (!hDraw) return c;
    hDraw->SetDirectory(nullptr);
    const std::string plotTitle = Form("Blast-Wave Fit for %.0f-%.0f%%", cenMin, cenMax);
    c->SetTitle(plotTitle.c_str());

    c->cd();
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.12);
    c->SetRightMargin(0.05);
    c->SetTopMargin(0.10);
    c->SetTicks(1, 1);
    c->SetGridy(true);
    c->SetLogy();

    hDraw->SetStats(false);
    hDraw->SetMarkerStyle(20);
    hDraw->SetMarkerSize(1.0);
    hDraw->SetLineColor(kBlack);
    hDraw->SetMarkerColor(kBlack);
    hDraw->SetMinimum(std::max(1e-12, hDraw->GetMinimum(1) * 0.5));
    hDraw->SetMaximum(hDraw->GetMaximum() * 1.4);
    hDraw->SetTitle(plotTitle.c_str());
    hDraw->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    hDraw->GetYaxis()->SetTitle("Corrected counts");
    hDraw->GetXaxis()->SetRangeUser(fBw->GetXmin(), fBw->GetXmax());
    hDraw->GetXaxis()->SetTitleSize(0.05);
    hDraw->GetXaxis()->SetTitleOffset(1.25);
    hDraw->GetYaxis()->SetTitleSize(0.05);
    hDraw->GetYaxis()->SetTitleOffset(1.35);
    auto *hDrawObj = dynamic_cast<TH1D *>(hDraw->DrawCopy("E1"));

    fBw->SetLineColor(kRed + 1);
    fBw->SetLineWidth(3);
    fBw->Draw("SAME");

    auto pave = std::make_unique<TPaveText>(0.12, 0.14, 0.62, 0.40, "NDC");
    pave->SetBorderSize(0);
    pave->SetFillStyle(0);
    pave->SetTextAlign(12);
    pave->SetTextSize(0.036);
    const double chi2 = fBw->GetChisquare();
    const int ndf = fBw->GetNDF();
    const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;
    pave->AddText(Form("Cen: %.0f-%.0f%%", cenMin, cenMax));
    pave->AddText(Form("#beta = %.3f #pm %.3f", fBw->GetParameter(1), fBw->GetParError(1)));
    pave->AddText(Form("T = %.3f #pm %.3f", fBw->GetParameter(2), fBw->GetParError(2)));
    pave->AddText(Form("n = %.3f #pm %.3f", fBw->GetParameter(3), fBw->GetParError(3)));
    pave->AddText(Form("norm = %.3g #pm %.3g", fBw->GetParameter(4), fBw->GetParError(4)));
    pave->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
    pave->AddText(Form("Fit prob. = %.3f", fitProb));
    pave->DrawClone();

    auto leg = std::make_unique<TLegend>(0.62, 0.76, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->AddEntry(hDrawObj ? hDrawObj : hCorr, "Corrected spectrum", "lep");
    leg->AddEntry(fBw, "BGBW fit", "l");
    leg->DrawClone();

    c->Modified();
    c->Update();
    return c;
}

inline std::unique_ptr<TCanvas> MakeExponentialFitCanvas(const std::string &name,
                                                         TH1D *hCorr,
                                                         TF1 *fExpo,
                                                         const std::string &xTitle,
                                                         double tauPs,
                                                         double tauPsErr) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!hCorr || !fExpo) return c;

    c->cd();
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.12);
    c->SetRightMargin(0.05);
    c->SetTopMargin(0.05);
    c->SetTicks(1, 1);
    c->SetGridy(true);
    c->SetLogy();

    hCorr->SetStats(false);
    hCorr->SetMinimum(std::max(1e-3, hCorr->GetMinimum(1) * 0.5));
    hCorr->SetLineColor(kAzure + 2);
    hCorr->SetMarkerColor(kAzure + 2);
    hCorr->SetMarkerStyle(20);
    hCorr->SetMarkerSize(1.1);
    hCorr->GetXaxis()->SetTitle(xTitle.c_str());
    hCorr->GetXaxis()->SetTitleSize(0.05);
    hCorr->GetXaxis()->SetLabelSize(0.045);
    hCorr->GetYaxis()->SetTitle("Corrected counts");
    hCorr->GetYaxis()->SetTitleSize(0.05);
    hCorr->GetYaxis()->SetTitleOffset(1.25);
    hCorr->GetYaxis()->SetLabelSize(0.045);
    hCorr->Draw("E1");

    fExpo->SetLineColor(kRed + 1);
    fExpo->SetLineWidth(3);
    fExpo->Draw("SAME");

    auto leg = std::make_unique<TLegend>(0.60, 0.70, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.042);
    leg->AddEntry(hCorr, "Corrected spectrum", "lep");
    leg->AddEntry(fExpo, "Exp fit", "l");
    leg->DrawClone();

    const double chi2 = fExpo->GetChisquare();
    const int ndf = fExpo->GetNDF();
    const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;

    auto pave = std::make_unique<TPaveText>(0.18, 0.70, 0.56, 0.90, "NDC");
    pave->SetFillStyle(0);
    pave->SetBorderSize(0);
    pave->SetTextAlign(12);
    pave->SetTextSize(0.042);
    pave->AddText(Form("#tau = %.2f #pm %.2f ps", tauPs, tauPsErr));
    pave->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
    pave->AddText(Form("Fit prob. = %.3f", fitProb));
    pave->DrawClone();

    c->Modified();
    c->Update();
    return c;
}

} // namespace UnifiedAnalysis

#endif // SPECTRUM_PLOT_HELPER_H
