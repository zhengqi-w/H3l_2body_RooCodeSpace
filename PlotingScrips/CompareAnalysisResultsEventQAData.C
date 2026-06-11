#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TPad.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TString.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
struct PeriodInput {
    const char *fileName;
    const char *label;
    Color_t color;
    Style_t marker;
    bool thirdEventBinIsRejected{false};
};

std::unique_ptr<TH1> CloneHist(TFile *f, const std::vector<const char *> &paths, const char *newName) {
    if (!f) return nullptr;
    TH1 *h = nullptr;
    for (const auto *path : paths) {
        h = dynamic_cast<TH1 *>(f->Get(path));
        if (h) break;
    }
    if (!h) {
        std::cerr << "ERROR: missing histogram. Tried paths:" << std::endl;
        for (const auto *path : paths) std::cerr << "  " << path << std::endl;
        return nullptr;
    }
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(h->Clone(newName)));
    if (out) out->SetDirectory(nullptr);
    return out;
}

void NormalizeHist(TH1 *h) {
    if (!h) return;
    const double sum = h->Integral(0, h->GetNbinsX() + 1);
    if (sum > 0.0) h->Scale(1.0 / sum);
}

void NormalizeHistInRange(TH1 *h, double xmin, double xmax) {
    if (!h) return;
    const int bmin = h->GetXaxis()->FindBin(xmin);
    const int bmax = h->GetXaxis()->FindBin(std::nextafter(xmax, xmin));
    const double sum = h->Integral(bmin, bmax);
    if (sum > 0.0) h->Scale(1.0 / sum);
}

void NormalizeEventsByFirstBin(TH1 *h, bool thirdBinIsRejected) {
    if (!h) return;
    if (thirdBinIsRejected && h->GetNbinsX() >= 3) {
        const double selectedRaw = h->GetBinContent(2);
        const double rejected = h->GetBinContent(3);
        h->SetBinContent(3, std::max(0.0, selectedRaw - rejected));
        h->SetBinError(3, std::sqrt(std::max(0.0, selectedRaw + rejected)));
    }
    const double norm = h->GetBinContent(1);
    if (norm > 0.0) h->Scale(1.0 / norm);
}

void StyleHist(TH1 *h, Color_t color, Style_t marker) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.1);
    h->SetLineWidth(2);
}

double GetMaxInRange(const TH1 *h, double xmin, double xmax) {
    if (!h) return 0.0;
    const int bmin = h->GetXaxis()->FindBin(xmin);
    const int bmax = h->GetXaxis()->FindBin(std::nextafter(xmax, xmin));
    double maxv = 0.0;
    for (int ib = bmin; ib <= bmax; ++ib) {
        maxv = std::max(maxv, h->GetBinContent(ib));
    }
    return maxv;
}

std::pair<double, double> GetRatioRange(const std::vector<std::unique_ptr<TH1>> &ratios, double xmin, double xmax) {
    double ymin = 1.0;
    double ymax = 1.0;
    bool seen = false;
    for (const auto &ratio : ratios) {
        if (!ratio) continue;
        const int bmin = ratio->GetXaxis()->FindBin(xmin);
        const int bmax = ratio->GetXaxis()->FindBin(std::nextafter(xmax, xmin));
        for (int ib = bmin; ib <= bmax; ++ib) {
            const double y = ratio->GetBinContent(ib);
            const double ey = ratio->GetBinError(ib);
            if (!std::isfinite(y) || y <= 0.0) continue;
            ymin = seen ? std::min(ymin, y - ey) : y - ey;
            ymax = seen ? std::max(ymax, y + ey) : y + ey;
            seen = true;
        }
    }
    if (!seen) return {0.5, 1.5};
    const double span = std::max(ymax - ymin, 0.2);
    ymin = std::max(0.0, ymin - 0.25 * span);
    ymax += 0.25 * span;
    ymin = std::min(ymin, 0.95);
    ymax = std::max(ymax, 1.05);
    return {ymin, ymax};
}

std::unique_ptr<TH1> MakeRatio(const TH1 *num, const TH1 *den, const char *name) {
    if (!num || !den) return nullptr;
    auto ratio = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(num->Clone(name)));
    if (!ratio) return nullptr;
    ratio->SetDirectory(nullptr);
    ratio->Divide(den);
    return ratio;
}

void ConfigureTopPad(TPad *pad) {
    pad->SetBottomMargin(0.02);
    pad->SetLeftMargin(0.13);
    pad->SetRightMargin(0.04);
    pad->SetTopMargin(0.08);
}

void ConfigureRatioPad(TPad *pad) {
    pad->SetTopMargin(0.02);
    pad->SetBottomMargin(0.30);
    pad->SetLeftMargin(0.13);
    pad->SetRightMargin(0.04);
    pad->SetGridy(true);
}

void StyleRatio(TH1 *h, Color_t color, Style_t marker) {
    StyleHist(h, color, marker);
    h->SetTitle("");
    h->GetYaxis()->SetTitle("Ratio / ref.");
    h->GetYaxis()->SetTitleSize(0.085);
    h->GetYaxis()->SetTitleOffset(0.55);
    h->GetYaxis()->SetLabelSize(0.075);
    h->GetYaxis()->SetNdivisions(505);
    h->GetXaxis()->SetTitleSize(0.095);
    h->GetXaxis()->SetLabelSize(0.085);
    h->GetXaxis()->SetTitleOffset(1.05);
}

void DrawComparisonWithRatio(const char *canvasName,
                             const char *outputName,
                             const char *title,
                             const char *xTitle,
                             const char *yTitle,
                             std::vector<std::unique_ptr<TH1>> &hists,
                             const std::vector<PeriodInput> &periods,
                             const std::vector<double> &nEvents,
                             double xmin,
                             double xmax,
                             bool useXRange,
                             const char *outDir,
                             bool drawAsHist = false,
                             bool drawMarkersOnHist = false) {
    std::vector<std::unique_ptr<TH1>> ratios;
    for (size_t i = 1; i < hists.size(); ++i) {
        ratios.push_back(MakeRatio(hists[i].get(), hists[0].get(), Form("%s_ratio_%zu", canvasName, i)));
        if (ratios.back()) StyleRatio(ratios.back().get(), periods[i].color, periods[i].marker);
    }

    TCanvas c(canvasName, title, 900, 850);
    TPad topPad(Form("%s_top", canvasName), "", 0.0, 0.32, 1.0, 1.0);
    TPad ratioPad(Form("%s_ratio", canvasName), "", 0.0, 0.0, 1.0, 0.32);
    ConfigureTopPad(&topPad);
    ConfigureRatioPad(&ratioPad);
    topPad.Draw();
    ratioPad.Draw();

    topPad.cd();
    double ymax = 0.0;
    for (const auto &hist : hists) {
        if (useXRange) hist->GetXaxis()->SetRangeUser(xmin, xmax);
        ymax = std::max(ymax, useXRange ? GetMaxInRange(hist.get(), xmin, xmax) : hist->GetMaximum());
    }

    hists[0]->SetTitle(Form("%s;%s;%s", title, xTitle, yTitle));
    hists[0]->SetMinimum(0.0);
    hists[0]->SetMaximum((ymax > 0.0) ? 1.30 * ymax : 1.0);
    hists[0]->GetXaxis()->SetLabelSize(0.0);
    hists[0]->GetXaxis()->SetTitleSize(0.0);
    hists[0]->GetYaxis()->SetTitleSize(0.055);
    hists[0]->GetYaxis()->SetTitleOffset(1.05);
    hists[0]->GetYaxis()->SetLabelSize(0.045);
    const char *drawOpt = drawAsHist ? "HIST" : "E1";
    hists[0]->Draw(drawOpt);
    for (size_t i = 1; i < hists.size(); ++i) {
        hists[i]->Draw(Form("%s SAME", drawOpt));
    }
    if (drawAsHist && drawMarkersOnHist) {
        for (auto &hist : hists) hist->Draw("P SAME");
    }

    TLegend leg(0.55, 0.68, 0.90, 0.90);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.040);
    for (size_t i = 0; i < hists.size(); ++i) {
        leg.AddEntry(hists[i].get(), periods[i].label, drawAsHist ? "lp" : "lep");
    }
    leg.Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.034);
    latex.DrawLatex(0.16, 0.88, Form("Reference ratio: %s", periods[0].label));
    for (size_t i = 0; i < periods.size(); ++i) {
        latex.DrawLatex(0.16, 0.82 - 0.045 * i, Form("NEvents %s = %.3e", periods[i].label, nEvents[i]));
    }

    ratioPad.cd();
    auto yrange = GetRatioRange(ratios, useXRange ? xmin : hists[0]->GetXaxis()->GetXmin(),
                                useXRange ? xmax : hists[0]->GetXaxis()->GetXmax());
    bool firstRatio = true;
    for (auto &ratio : ratios) {
        if (!ratio) continue;
        if (useXRange) ratio->GetXaxis()->SetRangeUser(xmin, xmax);
        ratio->GetXaxis()->SetTitle(xTitle);
        ratio->SetMinimum(yrange.first);
        ratio->SetMaximum(yrange.second);
        ratio->Draw(firstRatio ? drawOpt : Form("%s SAME", drawOpt));
        firstRatio = false;
    }
    if (drawAsHist && drawMarkersOnHist) {
        for (auto &ratio : ratios) {
            if (ratio) ratio->Draw("P SAME");
        }
    }

    TLine one(useXRange ? xmin : hists[0]->GetXaxis()->GetXmin(), 1.0,
              useXRange ? xmax : hists[0]->GetXaxis()->GetXmax(), 1.0);
    one.SetLineStyle(2);
    one.SetLineColor(kGray + 2);
    one.Draw("SAME");

    c.SaveAs(Form("%s/%s", outDir, outputName));
}
} // namespace

void CompareThreePeriodsEventQA(
    const char *period1File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/data/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
    const char *period2File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/data/AnalysisResults_CustomV0s_HadronPID.root",
    const char *period3File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/data/AnalysisResults_CustomV0s_HadronPID.root",
    const char *period1Label = "LHC23 pass5",
    const char *period2Label = "LHC24ar pass3",
    const char *period3Label = "LHC25 pass1",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/CompareAnalysisResultsEventQAData") {

    gSystem->mkdir(outDir, true);
    gStyle->SetOptStat(0);

    std::vector<PeriodInput> periods = {
        {period1File, period1Label, kBlack, 20, false},
        {period2File, period2Label, kRed + 1, 24, false},
        {period3File, period3Label, kAzure + 2, 25, false},
    };

    std::vector<std::unique_ptr<TFile>> files;
    for (const auto &period : periods) {
        files.emplace_back(TFile::Open(period.fileName, "READ"));
        if (!files.back() || files.back()->IsZombie()) {
            std::cerr << "ERROR: cannot open file for " << period.label << ": " << period.fileName << std::endl;
            return;
        }
    }

    const std::vector<const char *> zvtxPaths = {"hyper-reco-task/hZvtx", "hyper-reco-task/hVtxZ", "hyper-reco-task/hVtz"};
    const std::vector<const char *> centPaths = {"hyper-reco-task/hCentFT0C", "hyper-reco-task/fCentralityFT0C"};
    const std::vector<const char *> eventPaths = {"hyper-reco-task/hEvents"};

    std::vector<std::unique_ptr<TH1>> hEvents;
    std::vector<std::unique_ptr<TH1>> hZvtx;
    std::vector<std::unique_ptr<TH1>> hCent;
    std::vector<double> nEvents;
    for (size_t i = 0; i < periods.size(); ++i) {
        hEvents.push_back(CloneHist(files[i].get(), eventPaths, Form("hEvents_%zu", i)));
        hZvtx.push_back(CloneHist(files[i].get(), zvtxPaths, Form("hZvtx_%zu", i)));
        hCent.push_back(CloneHist(files[i].get(), centPaths, Form("hCentFT0C_%zu", i)));
        if (!hEvents.back() || !hZvtx.back() || !hCent.back()) return;

        nEvents.push_back(hEvents.back()->GetBinContent(1));
        NormalizeEventsByFirstBin(hEvents.back().get(), periods[i].thirdEventBinIsRejected);
        NormalizeHist(hZvtx.back().get());
        NormalizeHistInRange(hCent.back().get(), 0.0, 90.0);
        StyleHist(hEvents.back().get(), periods[i].color, periods[i].marker);
        StyleHist(hZvtx.back().get(), periods[i].color, periods[i].marker);
        StyleHist(hCent.back().get(), periods[i].color, periods[i].marker);
    }

    DrawComparisonWithRatio("c_hEvents_three_periods",
                            "hEvents_three_periods_norm_by_first_bin_ratio.pdf",
                            "Event selection comparison",
                            "hEvents bin",
                            "Counts / hEvents bin 1",
                            hEvents,
                            periods,
                            nEvents,
                            0.0,
                            0.0,
                            false,
                            outDir,
                            true,
                            true);

    DrawComparisonWithRatio("c_hZvtx_three_periods",
                            "hZvtx_three_periods_norm_ratio.pdf",
                            "Vtx_{z} normalized comparison",
                            "Vtx_{z} (cm)",
                            "Normalized counts",
                            hZvtx,
                            periods,
                            nEvents,
                            0.0,
                            0.0,
                            false,
                            outDir);

    DrawComparisonWithRatio("c_hCentFT0C_three_periods",
                            "fCentralityFT0C_three_periods_norm_ratio_0_90.pdf",
                            "fCentralityFT0C normalized comparison (0-90%)",
                            "fCentralityFT0C (%)",
                            "Normalized counts",
                            hCent,
                            periods,
                            nEvents,
                            0.0,
                            90.0,
                            true,
                            outDir);

    std::cout << "Saved: " << outDir << "/hZvtx_three_periods_norm_ratio.pdf" << std::endl;
    std::cout << "Saved: " << outDir << "/fCentralityFT0C_three_periods_norm_ratio_0_90.pdf" << std::endl;
    std::cout << "Saved: " << outDir << "/hEvents_three_periods_norm_by_first_bin_ratio.pdf" << std::endl;
}

void CompareAnalysisResultsEventQAData(
    const char *period1File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/data/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
    const char *period2File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/data/AnalysisResults_CustomV0s_HadronPID.root",
    const char *period3File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/data/AnalysisResults_CustomV0s_HadronPID.root",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/CompareAnalysisResultsEventQAData") {
    CompareThreePeriodsEventQA(period1File,
                               period2File,
                               period3File,
                               "LHC23 pass5",
                               "LHC24ar pass3",
                               "LHC25 pass1",
                               outDir);
}
