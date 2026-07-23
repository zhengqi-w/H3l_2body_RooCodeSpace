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
#include <TFitResultPtr.h>
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

inline double ComputeGaussianChi2Ndf(TF1 *f) {
    if (!f || f->GetNDF() <= 0) return std::numeric_limits<double>::infinity();
    return f->GetChisquare() / static_cast<double>(f->GetNDF());
}

inline bool AcceptSystematicGaussianFit(const TFitResultPtr &fitRes, TF1 *f, double maxChi2Ndf) {
    const int fitStatus = static_cast<int>(fitRes);
    const double chi2Ndf = ComputeGaussianChi2Ndf(f);
    const double sigma = f ? std::abs(f->GetParameter(2)) : 0.0;
    return fitStatus == 0 &&
           std::isfinite(chi2Ndf) &&
           chi2Ndf <= maxChi2Ndf &&
           std::isfinite(sigma) &&
           sigma > 0.0;
}

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

inline std::string BuildMassXAxisTitle(const std::string &isMatter) {
    if (isMatter == "matter") {
        return "M(^{3}He+#pi^{-}) (GeV/c^{2})";
    }
    if (isMatter == "antimatter") {
        return "M(^{3}#bar{He}+#pi^{+}) (GeV/c^{2})";
    }
    return "M(^{3}He+#pi^{-}, ^{3}#bar{He}+#pi^{+}) (GeV/c^{2})";
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
    if (frame->GetXaxis()) {
        frame->GetXaxis()->SetTitle(BuildMassXAxisTitle(isMatter).c_str());
    }
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
                                                              const std::string &binLabel,
                                                              double gaussFitMaxChi2Ndf,
                                                              double *usedUncertainty = nullptr) {
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
    std::unique_ptr<TF1> fgaus;
    double gausMean = std::numeric_limits<double>::quiet_NaN();
    double gausSigma = std::numeric_limits<double>::quiet_NaN();
    double gausChi2Ndf = std::numeric_limits<double>::infinity();
    bool gausAccepted = false;

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
        fgaus = std::make_unique<TF1>((name + "_gaus").c_str(),
                                      "gaus",
                                      hCorrDist->GetXaxis()->GetXmin(),
                                      hCorrDist->GetXaxis()->GetXmax());
        fgaus->SetParameters(hCorrDist->GetMaximum(), initMean, initSigma);
        auto fitRes = hCorrDist->Fit(fgaus.get(), "QSN0");
        gausChi2Ndf = ComputeGaussianChi2Ndf(fgaus.get());
        if (fitRes == 0 && fgaus) {
            fgaus->SetLineColor(kGreen + 3);
            fgaus->SetLineWidth(2);
            gausAccepted = AcceptSystematicGaussianFit(fitRes, fgaus.get(), gaussFitMaxChi2Ndf);
            fgaus->SetLineStyle(gausAccepted ? kSolid : kDashed);
            gausMean = fgaus->GetParameter(1);
            gausSigma = std::abs(fgaus->GetParameter(2));
            bandGauss = new TBox(gausMean - gausSigma, 0.0, gausMean + gausSigma,
                                 std::max(1.0, hCorrDist->GetMaximum()) * 1.02);
            bandGauss->SetFillStyle(3005);
            bandGauss->SetFillColor(kGreen + 1);
            bandGauss->SetLineColor(kGreen + 3);
        }
    }

    if (usedUncertainty) {
        *usedUncertainty = (gausAccepted && std::isfinite(gausSigma) && gausSigma > 0.0)
                               ? gausSigma
                               : hCorrDist->GetRMS();
    }

    if (bandCorr) bandCorr->Draw("same");
    if (bandGauss) bandGauss->Draw("same");
    if (fgaus) fgaus->DrawCopy("same");
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
    if (fgaus) leg->AddEntry(fgaus.get(), gausAccepted ? "Gauss fit" : "Gauss fit rejected", "l");
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
    if (std::isfinite(gausChi2Ndf)) {
        pave->AddText(Form("Gauss #chi^{2}/ndf = %.2f (%s)",
                           gausChi2Ndf,
                           gausAccepted ? "used" : "RMS used"));
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
                                                             TH1D *hTot,
                                                             TH1D *hRef = nullptr) {
    auto c = std::make_unique<TCanvas>(name.c_str(), name.c_str(), 960, 720);
    if (!hSel || !hAbso || !hTot) return c;

    auto makeFraction = [&](TH1D *src, const char *histName, const char *title) -> std::unique_ptr<TH1D> {
        if (!src) return nullptr;
        auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(src->Clone(histName)));
        h->SetDirectory(nullptr);
        h->SetTitle(title);
        h->SetStats(false);
        h->GetYaxis()->SetTitle("Systematic uncertainty (%)");
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double denom = hRef ? hRef->GetBinContent(ib) : 0.0;
            const double value = h->GetBinContent(ib);
            const double frac = (std::isfinite(denom) && std::abs(denom) > 0.0 && std::isfinite(value))
                                    ? 100.0 * value / std::abs(denom)
                                    : 0.0;
            h->SetBinContent(ib, frac);
            h->SetBinError(ib, 0.0);
        }
        return h;
    };

    auto hSelFrac = makeFraction(hSel, (name + "_selection_fraction").c_str(), hTot->GetTitle());
    auto hAbsoFrac = makeFraction(hAbso, (name + "_absorption_fraction").c_str(), hTot->GetTitle());
    auto hBrFrac = makeFraction(hBr, (name + "_br_fraction").c_str(), hTot->GetTitle());
    auto hTotFrac = makeFraction(hTot, (name + "_total_fraction").c_str(), hTot->GetTitle());
    if (!hSelFrac || !hAbsoFrac || !hTotFrac) return c;

    hTotFrac->SetLineColor(kBlack);
    hTotFrac->SetLineWidth(3);
    hSelFrac->SetLineColor(kRed + 1);
    hSelFrac->SetLineWidth(2);
    hAbsoFrac->SetLineColor(kAzure + 2);
    hAbsoFrac->SetLineWidth(2);
    if (hBrFrac) {
        hBrFrac->SetLineColor(kGreen + 2);
        hBrFrac->SetLineWidth(2);
    }

    double yMax = hTotFrac->GetMaximum();
    yMax = std::max(yMax, hSelFrac->GetMaximum());
    yMax = std::max(yMax, hAbsoFrac->GetMaximum());
    if (hBrFrac) yMax = std::max(yMax, hBrFrac->GetMaximum());
    if (!(yMax > 0.0)) yMax = 1.0;
    hTotFrac->GetYaxis()->SetRangeUser(0.0, yMax * 1.25);

    c->SetLeftMargin(0.12);
    c->SetBottomMargin(0.12);
    c->SetTicks(1, 1);
    c->SetGrid(false, false);

    hTotFrac->Draw("HIST");
    hSelFrac->Draw("HIST SAME");
    hAbsoFrac->Draw("HIST SAME");
    if (hBrFrac) hBrFrac->Draw("HIST SAME");

    auto leg = std::make_unique<TLegend>(0.58, 0.64, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.035);
    leg->AddEntry(hTotFrac.get(), "Total sys", "l");
    leg->AddEntry(hSelFrac.get(), "Selection sys", "l");
    leg->AddEntry(hAbsoFrac.get(), "Absorption sys", "l");
    if (hBrFrac) leg->AddEntry(hBrFrac.get(), "BR sys", "l");
    leg->DrawClone();

    c->Modified();
    c->Update();
    // The canvas keeps pointers to drawn histograms. Transfer ownership to ROOT
    // so the percentage clones remain alive when this helper returns.
    hTotFrac->SetBit(TObject::kCanDelete);
    hSelFrac->SetBit(TObject::kCanDelete);
    hAbsoFrac->SetBit(TObject::kCanDelete);
    if (hBrFrac) hBrFrac->SetBit(TObject::kCanDelete);
    hTotFrac.release();
    hSelFrac.release();
    hAbsoFrac.release();
    if (hBrFrac) hBrFrac.release();
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

    // Draw all text info in one object.
    TPaveText textBox(0.16, 0.18, 0.52, 0.38, "NDC");
    textBox.SetBorderSize(0);
    textBox.SetFillStyle(0);
    textBox.SetTextAlign(12);
    textBox.SetTextFont(42);
    textBox.SetTextSize(0.035);
    const auto expLines = BuildExperimentLines(labelCfg);
    for (const auto &line : expLines) {
        textBox.AddText(line.c_str());
    }
    const std::string decay = BuildDecayString(isMatter);
    if (!decay.empty()) {
        textBox.AddText(decay.c_str());
    }
    const std::string cenText = BuildCentralityText(extraText);
    if (!cenText.empty()) {
        textBox.AddText(cenText.c_str());
    }
    textBox.AddText(BuildEventLine(nEvents).c_str());
    textBox.DrawClone();

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
                                                       double cenMax,
                                                       const std::vector<std::string> &parameterTextLines = {}) {
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
    c->SetLogy();

    hDraw->SetStats(false);
    hDraw->SetMarkerStyle(20);
    hDraw->SetMarkerSize(1.15);
    hDraw->SetLineWidth(2);
    hDraw->SetLineColor(kBlack);
    hDraw->SetMarkerColor(kBlack);

    const double xMin = 0.0;
    const double xMax = 10.0;
    double yMin = std::numeric_limits<double>::infinity();
    double yMax = 0.0;
    for (int ib = 1; ib <= hDraw->GetNbinsX(); ++ib) {
        const double x = hDraw->GetXaxis()->GetBinCenter(ib);
        if (x < xMin || x > xMax) continue;
        const double y = hDraw->GetBinContent(ib);
        const double e = hDraw->GetBinError(ib);
        if (std::isfinite(y) && y > 0.0) {
            yMin = std::min(yMin, std::max(1e-20, y - e));
            yMax = std::max(yMax, y + e);
        }
    }
    for (double x = xMin; x <= xMax; x += 0.05) {
        const double y = fBw->Eval(x);
        if (std::isfinite(y) && y > 0.0) {
            yMin = std::min(yMin, y);
            yMax = std::max(yMax, y);
        }
    }
    if (!std::isfinite(yMin) || yMin <= 0.0) yMin = std::max(1e-14, hDraw->GetMinimum(1));
    if (!std::isfinite(yMax) || yMax <= yMin) yMax = std::max(yMin * 10.0, hDraw->GetMaximum());
    yMin *= 0.55;
    yMax *= 1.8;
    if (yMax / yMin > 1e6) yMin = yMax / 1e6;

    hDraw->SetMinimum(yMin);
    hDraw->SetMaximum(yMax);
    hDraw->SetTitle("");
    hDraw->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    hDraw->GetYaxis()->SetTitle("Corrected counts");
    hDraw->GetXaxis()->SetRangeUser(xMin, xMax);
    hDraw->GetXaxis()->SetTitleSize(0.05);
    hDraw->GetXaxis()->SetTitleOffset(1.25);
    hDraw->GetYaxis()->SetTitleSize(0.05);
    hDraw->GetYaxis()->SetTitleOffset(1.35);
    auto *hDrawObj = dynamic_cast<TH1D *>(hDraw->DrawCopy("E1"));

    fBw->SetLineColor(kRed + 1);
    fBw->SetLineWidth(3);
    fBw->SetLineStyle(kSolid);
    fBw->SetRange(xMin, xMax);
    fBw->Draw("SAME");

    auto leg = std::make_unique<TLegend>(0.62, 0.76, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.036);
    leg->AddEntry(hDrawObj ? hDrawObj : hCorr, "Corrected spectrum", "lep");
    leg->AddEntry(fBw, "BGBW fit", "l");
    leg->DrawClone();

    TLatex title;
    title.SetNDC();
    title.SetTextFont(42);
    title.SetTextSize(0.040);
    title.SetTextAlign(13);
    title.DrawLatex(0.16, 0.91, plotTitle.c_str());

    auto pave = std::make_unique<TPaveText>(0.16, 0.15, 0.58, 0.49, "NDC");
    pave->SetBorderSize(0);
    pave->SetFillStyle(0);
    pave->SetTextAlign(12);
    pave->SetTextFont(42);
    pave->SetTextSize(0.030);
    const double chi2 = fBw->GetChisquare();
    const int ndf = fBw->GetNDF();
    const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;
    pave->AddText(Form("Cen: %.0f-%.0f%%", cenMin, cenMax));
    if (!parameterTextLines.empty()) {
        for (const auto &line : parameterTextLines) pave->AddText(line.c_str());
    } else {
        pave->AddText(Form("#beta = %.3f #pm %.3f", fBw->GetParameter(1), fBw->GetParError(1)));
        pave->AddText(Form("T = %.3f #pm %.3f", fBw->GetParameter(2), fBw->GetParError(2)));
        pave->AddText(Form("n = %.3f #pm %.3f", fBw->GetParameter(3), fBw->GetParError(3)));
        pave->AddText(Form("Norm = %.3g #pm %.3g", fBw->GetParameter(4), fBw->GetParError(4)));
    }
    pave->AddText(Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
    pave->AddText(Form("Fit prob. = %.3f", fitProb));
    pave->DrawClone();

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
