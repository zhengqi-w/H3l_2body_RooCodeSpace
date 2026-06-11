#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
struct InputSpec {
    const char *fileName;
    const char *label;
    Color_t color;
    Style_t marker;
    bool thirdEventBinIsRejected{false};
};

std::unique_ptr<TH1> CloneHist(TFile *f, const char *path, const char *newName) {
    if (!f) return nullptr;
    auto *h = dynamic_cast<TH1 *>(f->Get(path));
    if (!h) {
        std::cerr << "[Error] Missing histogram: " << path << std::endl;
        return nullptr;
    }
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(h->Clone(newName)));
    if (out) out->SetDirectory(nullptr);
    return out;
}

void StyleHist(TH1 *h, Color_t color, Style_t marker) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.05);
    h->SetLineWidth(2);
}

void Normalize(TH1 *h, double xmin = 0.0, double xmax = 0.0, bool useRange = false) {
    if (!h) return;
    double sum = 0.0;
    if (useRange) {
        const int bmin = h->GetXaxis()->FindBin(xmin);
        const int bmax = h->GetXaxis()->FindBin(std::nextafter(xmax, xmin));
        sum = h->Integral(bmin, bmax);
    } else {
        sum = h->Integral(0, h->GetNbinsX() + 1);
    }
    if (sum > 0.0) h->Scale(1.0 / sum);
}

std::unique_ptr<TH1> MakeEventsForPlot(const TH1 *src, const char *name, bool thirdBinIsRejected) {
    if (!src) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(src->Clone(name)));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    if (thirdBinIsRejected && out->GetNbinsX() >= 3) {
        const double selectedRaw = out->GetBinContent(2);
        const double rejected = out->GetBinContent(3);
        const double keptAfterCut = std::max(0.0, selectedRaw - rejected);
        out->SetBinContent(3, keptAfterCut);
        out->SetBinError(3, std::sqrt(std::max(0.0, selectedRaw + rejected)));
    }
    const double norm = out->GetBinContent(1);
    if (norm > 0.0) {
        out->Scale(1.0 / norm);
    }
    return out;
}

double AllEvents(const TH1 *hEvents) {
    if (!hEvents || hEvents->GetNbinsX() < 1) return 0.0;
    return hEvents->GetBinContent(1);
}

double MaxVisible(const TH1 *h, double xmin, double xmax, bool useRange) {
    if (!h) return 0.0;
    if (!useRange) return h->GetMaximum();
    const int bmin = h->GetXaxis()->FindBin(xmin);
    const int bmax = h->GetXaxis()->FindBin(std::nextafter(xmax, xmin));
    double ymax = 0.0;
    for (int ib = bmin; ib <= bmax; ++ib) ymax = std::max(ymax, h->GetBinContent(ib) + h->GetBinError(ib));
    return ymax;
}

std::unique_ptr<TH1> MakeRatio(const TH1 *num, const TH1 *den, const char *name) {
    if (!num || !den) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(num->Clone(name)));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    out->Divide(den);
    out->SetTitle("");
    out->GetYaxis()->SetTitle("MC / data");
    out->GetYaxis()->SetTitleSize(0.085);
    out->GetYaxis()->SetTitleOffset(0.55);
    out->GetYaxis()->SetLabelSize(0.075);
    out->GetYaxis()->SetNdivisions(505);
    out->GetXaxis()->SetTitleSize(0.095);
    out->GetXaxis()->SetLabelSize(0.085);
    out->GetXaxis()->SetTitleOffset(1.05);
    return out;
}

std::pair<double, double> RatioRange(const std::vector<std::unique_ptr<TH1>> &ratios, double xmin, double xmax, bool useRange) {
    double ymin = 1.0;
    double ymax = 1.0;
    bool seen = false;
    for (const auto &h : ratios) {
        if (!h) continue;
        const int bmin = useRange ? h->GetXaxis()->FindBin(xmin) : 1;
        const int bmax = useRange ? h->GetXaxis()->FindBin(std::nextafter(xmax, xmin)) : h->GetNbinsX();
        for (int ib = bmin; ib <= bmax; ++ib) {
            const double y = h->GetBinContent(ib);
            const double ey = h->GetBinError(ib);
            if (!std::isfinite(y) || y <= 0.0) continue;
            ymin = seen ? std::min(ymin, y - ey) : y - ey;
            ymax = seen ? std::max(ymax, y + ey) : y + ey;
            seen = true;
        }
    }
    if (!seen) return {0.5, 1.5};
    const double span = std::max(0.2, ymax - ymin);
    return {std::max(0.0, ymin - 0.25 * span), ymax + 0.25 * span};
}

void DrawWithRatio(const char *canvasName,
                   const char *outPath,
                   const char *title,
                   const char *xTitle,
                   const char *yTitle,
                   std::vector<std::unique_ptr<TH1>> &hists,
                   const std::vector<InputSpec> &inputs,
                   const std::vector<double> &normalizationEvents,
                   double xmin = 0.0,
                   double xmax = 0.0,
                   bool useRange = false,
                   bool logY = false,
                   bool drawAsHist = false,
                   bool drawMarkersOnHist = false) {
    if (hists.empty() || !hists.front()) return;

    std::vector<std::unique_ptr<TH1>> ratios;
    for (size_t i = 1; i < hists.size(); ++i) {
        ratios.emplace_back(MakeRatio(hists[i].get(), hists[0].get(), Form("%s_ratio_%zu", canvasName, i)));
        if (ratios.back()) StyleHist(ratios.back().get(), inputs[i].color, inputs[i].marker);
    }

    TCanvas c(canvasName, title, 900, 850);
    TPad top(Form("%s_top", canvasName), "", 0.0, 0.32, 1.0, 1.0);
    TPad bottom(Form("%s_bottom", canvasName), "", 0.0, 0.0, 1.0, 0.32);
    top.SetBottomMargin(0.02);
    top.SetLeftMargin(0.13);
    top.SetRightMargin(0.04);
    top.SetTopMargin(0.08);
    bottom.SetTopMargin(0.02);
    bottom.SetBottomMargin(0.30);
    bottom.SetLeftMargin(0.13);
    bottom.SetRightMargin(0.04);
    bottom.SetGridy(true);
    top.Draw();
    bottom.Draw();

    top.cd();
    if (logY) top.SetLogy();
    double ymax = 0.0;
    double ymin = logY ? 1e-6 : 0.0;
    for (const auto &h : hists) {
        if (useRange) h->GetXaxis()->SetRangeUser(xmin, xmax);
        ymax = std::max(ymax, MaxVisible(h.get(), xmin, xmax, useRange));
    }
    hists[0]->SetTitle(Form("%s;%s;%s", title, xTitle, yTitle));
    hists[0]->SetMinimum(ymin);
    hists[0]->SetMaximum((ymax > 0.0) ? ymax * (logY ? 8.0 : 1.35) : 1.0);
    hists[0]->GetXaxis()->SetLabelSize(0.0);
    hists[0]->GetXaxis()->SetTitleSize(0.0);
    hists[0]->GetYaxis()->SetTitleSize(0.055);
    hists[0]->GetYaxis()->SetTitleOffset(1.05);
    hists[0]->GetYaxis()->SetLabelSize(0.045);
    const char *drawOpt = drawAsHist ? "HIST" : "E1";
    hists[0]->Draw(drawOpt);
    for (size_t i = 1; i < hists.size(); ++i) hists[i]->Draw(Form("%s SAME", drawOpt));
    if (drawAsHist && drawMarkersOnHist) {
        for (auto &hist : hists) hist->Draw("P SAME");
    }

    TLegend leg(0.58, 0.74, 0.90, 0.90);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.040);
    for (size_t i = 0; i < hists.size(); ++i) leg.AddEntry(hists[i].get(), inputs[i].label, drawAsHist ? "lp" : "lep");
    leg.Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.034);
    latex.DrawLatex(0.16, 0.88, "LHC23 Pb--Pb pass5");
    latex.DrawLatex(0.16, 0.83, Form("Reference ratio: %s", inputs[0].label));
    for (size_t i = 0; i < normalizationEvents.size(); ++i) {
        latex.DrawLatex(0.16, 0.77 - 0.045 * i, Form("N_{all} %s = %.3e", inputs[i].label, normalizationEvents[i]));
    }

    bottom.cd();
    const auto yr = RatioRange(ratios, xmin, xmax, useRange);
    bool first = true;
    for (auto &ratio : ratios) {
        if (!ratio) continue;
        if (useRange) ratio->GetXaxis()->SetRangeUser(xmin, xmax);
        ratio->SetMinimum(yr.first);
        ratio->SetMaximum(yr.second);
        ratio->GetXaxis()->SetTitle(xTitle);
        ratio->Draw(first ? drawOpt : Form("%s SAME", drawOpt));
        first = false;
    }
    if (drawAsHist && drawMarkersOnHist) {
        for (auto &ratio : ratios) {
            if (ratio) ratio->Draw("P SAME");
        }
    }
    TLine one(useRange ? xmin : hists[0]->GetXaxis()->GetXmin(), 1.0,
              useRange ? xmax : hists[0]->GetXaxis()->GetXmax(), 1.0);
    one.SetLineStyle(2);
    one.SetLineColor(kGray + 2);
    one.Draw("SAME");

    c.SaveAs(outPath);
    TString pngPath(outPath);
    if (pngPath.EndsWith(".pdf")) {
        pngPath.ReplaceAll(".pdf", ".png");
        c.SaveAs(pngPath.Data());
    }
}
} // namespace

void CompareAnalysisResultsEventQAMCData(
    const char *dataPath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/data/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
    const char *mcPath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/AnalysisResults.root",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/CompareAnalysisResultsEventQAMCData_LHC23Pass5") {
    gStyle->SetOptStat(0);
    gSystem->mkdir(outDir, true);

    std::vector<InputSpec> inputs = {
        {dataPath, "Data", kBlack, 20, false},
        {mcPath, "MC LHC25g11_G4list", kRed + 1, 24, false},
    };

    std::vector<std::unique_ptr<TFile>> files;
    for (const auto &in : inputs) {
        files.emplace_back(TFile::Open(in.fileName, "READ"));
        if (!files.back() || files.back()->IsZombie()) {
            std::cerr << "[Error] Cannot open " << in.fileName << std::endl;
            return;
        }
    }

    std::vector<std::unique_ptr<TH1>> hEvents;
    std::vector<std::unique_ptr<TH1>> hCent;
    std::vector<std::unique_ptr<TH1>> hZvtx;
    std::vector<double> normalizationEvents;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto rawEvents = CloneHist(files[i].get(), "hyper-reco-task/hEvents", Form("hEvents_raw_%zu", i));
        auto cent = CloneHist(files[i].get(), "hyper-reco-task/hCentFT0C", Form("hCentFT0C_%zu", i));
        auto zvtx = CloneHist(files[i].get(), "hyper-reco-task/hZvtx", Form("hZvtx_%zu", i));
        if (!rawEvents || !cent || !zvtx) return;

        normalizationEvents.push_back(AllEvents(rawEvents.get()));
        hEvents.emplace_back(MakeEventsForPlot(rawEvents.get(), Form("hEvents_plot_%zu", i), inputs[i].thirdEventBinIsRejected));
        hCent.emplace_back(std::move(cent));
        hZvtx.emplace_back(std::move(zvtx));

        Normalize(hCent.back().get(), 0.0, 90.0, true);
        Normalize(hZvtx.back().get());
        StyleHist(hEvents.back().get(), inputs[i].color, inputs[i].marker);
        StyleHist(hCent.back().get(), inputs[i].color, inputs[i].marker);
        StyleHist(hZvtx.back().get(), inputs[i].color, inputs[i].marker);
    }

    DrawWithRatio("c_lhc23_pass5_data_mc_events",
                  Form("%s/lhc23_pass5_data_mc_hEvents.pdf", outDir),
                  "Event selection QA; each dataset normalized by hEvents bin 1",
                  "hEvents bin",
                  "Counts / hEvents bin 1",
                  hEvents,
                  inputs,
                  normalizationEvents,
                  0.0,
                  0.0,
                  false,
                  false,
                  true,
                  true);

    DrawWithRatio("c_lhc23_pass5_data_mc_cent",
                  Form("%s/lhc23_pass5_data_mc_hCentFT0C_norm_0_90.pdf", outDir),
                  "FT0C centrality normalized comparison",
                  "FT0C centrality (%)",
                  "Normalized counts",
                  hCent,
                  inputs,
                  normalizationEvents,
                  0.0,
                  90.0,
                  true,
                  false,
                  false,
                  false);

    DrawWithRatio("c_lhc23_pass5_data_mc_zvtx",
                  Form("%s/lhc23_pass5_data_mc_hZvtx_norm.pdf", outDir),
                  "Vtx_{z} normalized comparison",
                  "Vtx_{z} (cm)",
                  "Normalized counts",
                  hZvtx,
                  inputs,
                  normalizationEvents,
                  0.0,
                  0.0,
                  false,
                  false,
                  false,
                  false);

    std::cout << "Saved plots in: " << outDir << std::endl;
}
