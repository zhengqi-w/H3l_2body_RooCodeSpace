#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TH1.h>
#include <TKey.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TPad.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TBox.h>
#include <ROOT/RDataFrame.hxx>
#include <TChain.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../Tools/AcceptanceHelper.h"
#include "../Tools/GeneralHelper.hpp"

namespace {
struct Sample {
    const char *label;
    const char *fileName;
    Color_t color;
    Style_t marker;
};

struct RawSummary {
    double raw{0.0};
    double rawErr{0.0};
    double nEvents{0.0};
    double rawOverEvents{0.0};
    double rawOverEventsErr{0.0};
};

struct RatioBundle {
    std::unique_ptr<TCanvas> canvas;
    std::vector<std::unique_ptr<TH1>> ratios;
};

std::unique_ptr<TH1> GetHist(TFile *f, const std::string &path, const std::string &name) {
    if (!f) return nullptr;
    auto *h = dynamic_cast<TH1 *>(f->Get(path.c_str()));
    if (!h) {
        std::cerr << "[Warn] missing hist: " << path << std::endl;
        return nullptr;
    }
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(h->Clone(name.c_str())));
    if (out) out->SetDirectory(nullptr);
    return out;
}

std::unique_ptr<TGraphAsymmErrors> GetGraph(TFile *f, const std::string &path, const std::string &name) {
    if (!f) return nullptr;
    auto *g = dynamic_cast<TGraphAsymmErrors *>(f->Get(path.c_str()));
    if (!g) {
        std::cerr << "[Warn] missing graph: " << path << std::endl;
        return nullptr;
    }
    return std::unique_ptr<TGraphAsymmErrors>(dynamic_cast<TGraphAsymmErrors *>(g->Clone(name.c_str())));
}

std::unique_ptr<TGraphAsymmErrors> GraphFromHistSys(const TH1 *h, const std::string &name) {
    if (!h) return nullptr;
    auto g = std::make_unique<TGraphAsymmErrors>(h->GetNbinsX());
    g->SetName(name.c_str());
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const int ip = ib - 1;
        const double x = h->GetBinCenter(ib);
        const double ex = 0.5 * h->GetBinWidth(ib);
        const double y = h->GetBinContent(ib);
        const double ey = h->GetBinError(ib);
        g->SetPoint(ip, x, y);
        g->SetPointError(ip, ex, ex, ey, ey);
    }
    return g;
}

TGraphAsymmErrors *MakeStatGraphWithBinWidth(const TH1 *h, const std::string &name, const Sample &s) {
    if (!h) return nullptr;
    auto *g = new TGraphAsymmErrors(h->GetNbinsX());
    g->SetName(name.c_str());
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const int ip = ib - 1;
        const double x = h->GetBinCenter(ib);
        const double y = h->GetBinContent(ib);
        const double ex = 0.5 * h->GetBinWidth(ib);
        const double ey = h->GetBinError(ib);
        g->SetPoint(ip, x, y);
        g->SetPointError(ip, ex, ex, ey, ey);
    }
    g->SetLineColor(s.color);
    g->SetMarkerColor(s.color);
    g->SetMarkerStyle(s.marker);
    g->SetMarkerSize(1.12);
    g->SetLineWidth(3);
    return g;
}

void StyleHist(TH1 *h, const Sample &s) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(s.color);
    h->SetMarkerColor(s.color);
    h->SetMarkerStyle(s.marker);
    h->SetMarkerSize(1.0);
    h->SetLineWidth(2);
}

void StyleFrame(TH1 *h, double titleSize = 0.047, double labelSize = 0.041, double yOffset = 1.25) {
    if (!h) return;
    h->SetStats(false);
    h->GetXaxis()->SetTitleSize(titleSize);
    h->GetYaxis()->SetTitleSize(titleSize);
    h->GetXaxis()->SetLabelSize(labelSize);
    h->GetYaxis()->SetLabelSize(labelSize);
    h->GetYaxis()->SetTitleOffset(yOffset);
    h->GetXaxis()->SetTitleOffset(1.05);
}

void StyleSys(TGraphAsymmErrors *g, const Sample &s) {
    if (!g) return;
    g->SetLineColor(s.color);
    g->SetMarkerColor(s.color);
    g->SetFillColorAlpha(s.color, 0.22);
    g->SetLineWidth(1);
}

void StyleGraphStat(TGraphAsymmErrors *g, const Sample &s) {
    if (!g) return;
    g->SetLineColor(s.color);
    g->SetMarkerColor(s.color);
    g->SetMarkerStyle(s.marker);
    g->SetMarkerSize(1.1);
    g->SetLineWidth(2);
}

double MaxHist(const TH1 *h) {
    if (!h) return 0.0;
    double ymax = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        ymax = std::max(ymax, h->GetBinContent(ib) + h->GetBinError(ib));
    }
    return ymax;
}

double MaxGraph(const TGraphAsymmErrors *g) {
    if (!g) return 0.0;
    double ymax = 0.0;
    for (int ip = 0; ip < g->GetN(); ++ip) {
        double x = 0.0, y = 0.0;
        g->GetPoint(ip, x, y);
        ymax = std::max(ymax, y + g->GetErrorYhigh(ip));
    }
    return ymax;
}

std::pair<double, double> XRangeFromHists(const std::vector<std::unique_ptr<TH1>> &hists) {
    double xmin = 1e30;
    double xmax = -1e30;
    for (const auto &h : hists) {
        if (!h) continue;
        xmin = std::min(xmin, h->GetXaxis()->GetXmin());
        xmax = std::max(xmax, h->GetXaxis()->GetXmax());
    }
    if (xmin > xmax) return {0.0, 1.0};
    return {xmin, xmax};
}

std::unique_ptr<TH1> MakeRatioHist(const TH1 *num, const TH1 *den, const std::string &name, const std::string &xTitle = "#it{p}_{T} (GeV/#it{c})") {
    if (!num || !den) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(num->Clone(name.c_str())));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    out->Divide(den);
    out->SetTitle("");
    out->GetYaxis()->SetTitle("Ratio / LHC23");
    out->GetYaxis()->SetTitleSize(0.085);
    out->GetYaxis()->SetTitleOffset(0.55);
    out->GetYaxis()->SetLabelSize(0.075);
    out->GetYaxis()->SetNdivisions(505);
    out->GetXaxis()->SetTitle(xTitle.c_str());
    out->GetXaxis()->SetTitleSize(0.095);
    out->GetXaxis()->SetLabelSize(0.085);
    out->GetXaxis()->SetTitleOffset(1.05);
    return out;
}

std::pair<double, double> RatioRange(const std::vector<std::unique_ptr<TH1>> &ratios) {
    double ymin = 1.0;
    double ymax = 1.0;
    bool seen = false;
    for (const auto &h : ratios) {
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
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
    ymin = std::max(0.0, ymin - 0.25 * span);
    ymax += 0.25 * span;
    ymin = std::min(ymin, 0.95);
    ymax = std::max(ymax, 1.05);
    return {ymin, ymax};
}

std::pair<double, double> RatioRangeRaw(const std::vector<TH1 *> &ratios) {
    double ymin = 1.0;
    double ymax = 1.0;
    bool seen = false;
    for (const auto *h : ratios) {
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
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
    ymin = std::max(0.0, ymin - 0.25 * span);
    ymax += 0.25 * span;
    ymin = std::min(ymin, 0.95);
    ymax = std::max(ymax, 1.05);
    return {ymin, ymax};
}

RawSummary BuildRawSummary(const TH1 *hRaw, const TH1 *hRawOverNevt) {
    RawSummary out;
    if (!hRaw) return out;
    double err2 = 0.0;
    for (int ib = 1; ib <= hRaw->GetNbinsX(); ++ib) {
        out.raw += hRaw->GetBinContent(ib);
        err2 += hRaw->GetBinError(ib) * hRaw->GetBinError(ib);
    }
    out.rawErr = std::sqrt(err2);
    if (hRawOverNevt && hRawOverNevt->GetBinContent(1) > 0.0) {
        out.rawOverEvents = hRawOverNevt->GetBinContent(1);
        out.rawOverEventsErr = hRawOverNevt->GetBinError(1);
        out.nEvents = out.raw / out.rawOverEvents;
    }
    return out;
}

std::unique_ptr<TH1> MakeRawOverBdtOverEvents(const TH1 *hRaw, const TH1 *hBdt, double nEvents, const std::string &name) {
    if (!hRaw || !hBdt || nEvents <= 0.0) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(hRaw->Clone(name.c_str())));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    out->SetTitle(";#it{p}_{T} (GeV/#it{c});N_{raw}/(#epsilon_{BDT} N_{evt})");
    for (int ib = 1; ib <= out->GetNbinsX(); ++ib) {
        const double raw = hRaw->GetBinContent(ib);
        const double rawErr = hRaw->GetBinError(ib);
        const double eff = hBdt->GetBinContent(ib);
        const double effErr = hBdt->GetBinError(ib);
        if (eff <= 0.0) {
            out->SetBinContent(ib, 0.0);
            out->SetBinError(ib, 0.0);
            continue;
        }
        const double val = raw / eff / nEvents;
        double rel2 = 0.0;
        if (raw > 0.0) rel2 += (rawErr / raw) * (rawErr / raw);
        if (eff > 0.0 && effErr > 0.0) rel2 += (effErr / eff) * (effErr / eff);
        out->SetBinContent(ib, val);
        out->SetBinError(ib, val * std::sqrt(rel2));
    }
    return out;
}

std::unique_ptr<TH1> MakeRawOverEventsPtHist(const TH1 *hRaw, double nEvents, const std::string &name) {
    if (!hRaw || nEvents <= 0.0) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(hRaw->Clone(name.c_str())));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    out->SetTitle(";#it{p}_{T} (GeV/#it{c});N_{raw}/N_{evt}");
    out->Scale(1.0 / nEvents);
    return out;
}

void DrawExperimentLabel(double x, double y, const char *extra) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.040);
    latex.DrawLatex(x, y, "ALICE Run 3 internal");
    latex.SetTextSize(0.034);
    latex.DrawLatex(x, y - 0.050, "Pb#minusPb #sqrt{#it{s}_{NN}} = 5.36 TeV");
    latex.DrawLatex(x, y - 0.095, extra);
}

std::string CentralityLabel(std::string cent) {
    std::replace(cent.begin(), cent.end(), '_', '-');
    return cent + "%";
}

std::unique_ptr<TCanvas> DrawSpectrum(const std::vector<Sample> &samples,
                                      const std::vector<std::unique_ptr<TH1>> &stats,
                                      const std::vector<std::unique_ptr<TGraphAsymmErrors>> &sys,
                                      const std::string &cent) {
    auto c = std::make_unique<TCanvas>(("c_spectrum_" + cent).c_str(), "spectrum", 980, 900);
    c->SetTicks(1, 1);
    auto *top = new TPad(("top_spectrum_" + cent).c_str(), "", 0.0, 0.34, 1.0, 1.0);
    auto *bot = new TPad(("bot_ratio_spectrum_" + cent).c_str(), "", 0.0, 0.0, 1.0, 0.34);
    top->SetLogy();
    top->SetBottomMargin(0.02);
    top->SetLeftMargin(0.14);
    top->SetRightMargin(0.04);
    top->SetTopMargin(0.06);
    top->SetTicks(1, 1);
    bot->SetTopMargin(0.02);
    bot->SetBottomMargin(0.30);
    bot->SetLeftMargin(0.14);
    bot->SetRightMargin(0.04);
    bot->SetGridy(true);
    bot->SetTicks(1, 1);
    top->Draw();
    bot->Draw();

    top->cd();
    auto xr = XRangeFromHists(stats);
    double ymax = 0.0;
    double ymin = 1e30;
    for (size_t i = 0; i < stats.size(); ++i) {
        ymax = std::max(ymax, std::max(MaxHist(stats[i].get()), MaxGraph(sys[i].get())));
        if (!stats[i]) continue;
        for (int ib = 1; ib <= stats[i]->GetNbinsX(); ++ib) {
            const double y = stats[i]->GetBinContent(ib);
            if (y > 0.0) ymin = std::min(ymin, y);
        }
    }
    if (ymin == 1e30) ymin = 1e-10;
    auto *frame = new TH1D(("frame_spec_" + cent).c_str(),
        Form(";%s;%s", "#it{p}_{T} (GeV/#it{c})", "#frac{1}{N_{evt}} #frac{d^{2}N}{d#it{p}_{T}d#it{y}} (#it{c}/GeV)"),
        100, xr.first, xr.second);
    frame->SetDirectory(nullptr);
    frame->SetMinimum(std::max(ymin * 0.25, 1e-12));
    frame->SetMaximum(ymax > 0.0 ? ymax * 6.0 : 1.0);
    StyleFrame(frame, 0.050, 0.043, 1.22);
    frame->GetXaxis()->SetLabelSize(0.0);
    frame->GetXaxis()->SetTitleSize(0.0);
    frame->Draw();

    auto *leg = new TLegend(0.56, 0.67, 0.91, 0.91);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.034);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (sys[i]) sys[i]->Draw("2 SAME");
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!stats[i]) continue;
        stats[i]->Draw("E1 SAME");
        leg->AddEntry(stats[i].get(), samples[i].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.16, 0.86, Form("Centrality %s%%, stat. bars + syst. boxes", cent.c_str()));

    bot->cd();
    auto *rFrame = new TH1D(("frame_ratio_spec_" + cent).c_str(), ";#it{p}_{T} (GeV/#it{c});Ratio / LHC23",
                            100, xr.first, xr.second);
    rFrame->SetDirectory(nullptr);
    rFrame->SetStats(false);
    std::vector<TH1 *> ratios;
    for (size_t i = 1; i < stats.size(); ++i) {
        auto ratioOwned = MakeRatioHist(stats[i].get(), stats[0].get(), Form("h_ratio_spec_%zu_%s", i, cent.c_str()));
        if (!ratioOwned) continue;
        TH1 *ratio = ratioOwned.release();
        ratio->SetDirectory(nullptr);
        StyleHist(ratio, samples[i]);
        ratios.push_back(ratio);
    }
    const auto ratioRange = RatioRangeRaw(ratios);
    rFrame->SetMinimum(ratioRange.first);
    rFrame->SetMaximum(ratioRange.second);
    StyleFrame(rFrame, 0.090, 0.075, 0.58);
    rFrame->GetYaxis()->SetNdivisions(505);
    rFrame->Draw();
    auto *one = new TLine(xr.first, 1.0, xr.second, 1.0);
    one->SetLineStyle(2);
    one->SetLineColor(kGray + 2);
    one->SetLineWidth(2);
    one->Draw("SAME");
    for (auto *ratio : ratios) {
        if (!ratio) continue;
        ratio->Draw("E1 SAME");
    }

    return c;
}

std::unique_ptr<TCanvas> DrawRawAndEvents(const std::vector<Sample> &samples,
                                          const std::vector<std::unique_ptr<TH1>> &raw,
                                          const std::vector<RawSummary> &summaries,
                                          const std::string &cent) {
    auto c = std::make_unique<TCanvas>(("c_raw_nevents_" + cent).c_str(), "raw counts and events", 900, 850);
    auto *top = new TPad("top_raw", "", 0.0, 0.33, 1.0, 1.0);
    auto *bot = new TPad("bot_evt", "", 0.0, 0.0, 1.0, 0.33);
    top->SetBottomMargin(0.03);
    top->SetLeftMargin(0.13);
    top->SetRightMargin(0.04);
    bot->SetTopMargin(0.03);
    bot->SetBottomMargin(0.28);
    bot->SetLeftMargin(0.13);
    bot->SetRightMargin(0.04);
    top->Draw();
    bot->Draw();

    top->cd();
    top->SetLogy();
    auto xr = XRangeFromHists(raw);
    double ymax = 0.0;
    double ymin = 1e30;
    for (const auto &h : raw) {
        ymax = std::max(ymax, MaxHist(h.get()));
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double y = h->GetBinContent(ib);
            if (y > 0.0) ymin = std::min(ymin, y);
        }
    }
    if (ymin == 1e30) ymin = 1.0;
    auto *frame = new TH1D(("frame_raw_" + cent).c_str(), ";#it{p}_{T} (GeV/#it{c});N_{raw}", 100, xr.first, xr.second);
    frame->SetDirectory(nullptr);
    frame->SetMinimum(std::max(0.2, ymin * 0.3));
    frame->SetMaximum(ymax > 0.0 ? ymax * 6.0 : 1.0);
    frame->GetXaxis()->SetLabelSize(0.0);
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->SetStats(false);
    frame->Draw();

    auto *leg = new TLegend(0.55, 0.62, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.035);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!raw[i]) continue;
        raw[i]->Draw("E1 SAME");
        leg->AddEntry(raw[i].get(), samples[i].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.16, 0.86, Form("Centrality %s%%", cent.c_str()));

    bot->cd();
    auto *hEvt = new TH1D(("h_events_" + cent).c_str(), ";sample;N_{evt}", static_cast<int>(samples.size()), 0.5, samples.size() + 0.5);
    hEvt->SetDirectory(nullptr);
    hEvt->SetStats(false);
    hEvt->GetYaxis()->SetTitleOffset(0.75);
    hEvt->GetYaxis()->SetTitleSize(0.075);
    hEvt->GetYaxis()->SetLabelSize(0.065);
    hEvt->GetXaxis()->SetLabelSize(0.075);
    hEvt->GetXaxis()->LabelsOption("v");
    double evtMax = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        hEvt->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), samples[i].label);
        hEvt->SetBinContent(static_cast<int>(i + 1), summaries[i].nEvents);
        evtMax = std::max(evtMax, summaries[i].nEvents);
    }
    hEvt->SetMaximum(evtMax > 0.0 ? evtMax * 1.35 : 1.0);
    hEvt->SetFillColor(kGray + 1);
    hEvt->Draw("BAR");
    return c;
}

RatioBundle DrawPeriodPtRatio(const std::vector<Sample> &samples,
                              const std::vector<std::unique_ptr<TH1>> &hists,
                              const std::string &cent,
                              const std::string &canvasStem,
                              const std::string &yTitle,
                              const std::string &labelExtra,
                              bool logTop = true,
                              const std::string &xTitle = "#it{p}_{T} (GeV/#it{c})",
                              const std::string &displayLabel = "") {
    RatioBundle out;
    out.canvas = std::make_unique<TCanvas>(("c_" + canvasStem + "_" + cent).c_str(), canvasStem.c_str(), 1050, 920);
    auto *c = out.canvas.get();
    c->SetTicks(1, 1);
    auto *top = new TPad(("top_" + canvasStem + "_" + cent).c_str(), "", 0.0, 0.34, 1.0, 1.0);
    auto *bot = new TPad(("bot_ratio_" + canvasStem + "_" + cent).c_str(), "", 0.0, 0.0, 1.0, 0.34);
    if (logTop) top->SetLogy();
    top->SetBottomMargin(0.02);
    top->SetLeftMargin(0.14);
    top->SetRightMargin(0.04);
    top->SetTopMargin(0.06);
    top->SetTicks(1, 1);
    bot->SetTopMargin(0.02);
    bot->SetBottomMargin(0.30);
    bot->SetLeftMargin(0.14);
    bot->SetRightMargin(0.04);
    bot->SetGridy(true);
    bot->SetTicks(1, 1);
    top->Draw();
    bot->Draw();

    top->cd();
    auto xr = XRangeFromHists(hists);
    double ymax = 0.0;
    double ymin = 1e30;
    for (const auto &h : hists) {
        ymax = std::max(ymax, MaxHist(h.get()));
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double y = h->GetBinContent(ib);
            if (y > 0.0) ymin = std::min(ymin, y);
        }
    }
    if (ymin == 1e30) ymin = 1e-12;
    auto *frame = new TH1D(("frame_" + canvasStem + "_" + cent).c_str(),
                           Form(";%s;%s", xTitle.c_str(), yTitle.c_str()),
                           100,
                           xr.first,
                           xr.second);
    frame->SetDirectory(nullptr);
    frame->SetMinimum(logTop ? std::max(ymin * 0.25, 1e-14) : 0.0);
    frame->SetMaximum(ymax > 0.0 ? (logTop ? ymax * 4.8 : std::min(1.0, ymax * 1.35)) : 1.0);
    StyleFrame(frame, 0.050, 0.043, 1.22);
    frame->GetXaxis()->SetLabelSize(0.0);
    frame->GetXaxis()->SetTitleSize(0.0);
    frame->Draw();

    auto *leg = new TLegend(0.56, 0.68, 0.91, 0.91);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.034);
    const Float_t prevEndErrorSizeTop = gStyle->GetEndErrorSize();
    gStyle->SetEndErrorSize(4.0f);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!hists[i]) continue;
        hists[i]->SetLineColor(samples[i].color);
        hists[i]->SetMarkerColor(samples[i].color);
        hists[i]->SetMarkerStyle(samples[i].marker);
        hists[i]->SetMarkerSize(1.12);
        hists[i]->SetLineWidth(3);
        auto *gTop = MakeStatGraphWithBinWidth(hists[i].get(), Form("g_top_%s_%zu_%s", canvasStem.c_str(), i, cent.c_str()), samples[i]);
        if (!gTop) continue;
        gTop->Draw("P SAME");
        leg->AddEntry(gTop, samples[i].label, "lep");
    }
    gStyle->SetEndErrorSize(prevEndErrorSizeTop);
    leg->Draw();
    const std::string label = displayLabel.empty() ? ("Centrality " + CentralityLabel(cent)) : displayLabel;
    DrawExperimentLabel(0.16, 0.86, Form("%s, %s", label.c_str(), labelExtra.c_str()));

    bot->cd();
    auto *rFrame = new TH1D(("frame_ratio_" + canvasStem + "_" + cent).c_str(),
                            Form(";%s;Ratio / LHC23", xTitle.c_str()),
                            100,
                            xr.first,
                            xr.second);
    rFrame->SetDirectory(nullptr);
    rFrame->SetStats(false);
    std::vector<TH1 *> ratios;
    std::vector<size_t> ratioSampleIndex;
    for (size_t i = 1; i < hists.size(); ++i) {
        auto ratioOwned = MakeRatioHist(hists[i].get(), hists[0].get(),
                                        Form("h_ratio_%s_%zu_%s", canvasStem.c_str(), i, cent.c_str()),
                                        xTitle);
        if (!ratioOwned) continue;
        TH1 *ratio = ratioOwned.get();
        StyleHist(ratio, samples[i]);
        ratio->SetMarkerSize(1.05);
        ratios.push_back(ratio);
        ratioSampleIndex.push_back(i);
        out.ratios.push_back(std::move(ratioOwned));
    }
    const auto ratioRange = RatioRangeRaw(ratios);
    rFrame->SetMinimum(ratioRange.first);
    rFrame->SetMaximum(ratioRange.second);
    StyleFrame(rFrame, 0.090, 0.075, 0.58);
    rFrame->GetYaxis()->SetNdivisions(505);
    rFrame->Draw();
    auto *one = new TLine(xr.first, 1.0, xr.second, 1.0);
    one->SetLineStyle(2);
    one->SetLineColor(kGray + 2);
    one->SetLineWidth(2);
    one->Draw("SAME");
    std::vector<TF1 *> fits;
    std::vector<size_t> fitSampleIndex;
    for (size_t ir = 0; ir < ratios.size(); ++ir) {
        auto *ratio = ratios[ir];
        if (ratio) ratio->Draw("E1 SAME");
        if (!ratio) continue;
        const size_t sampleIndex = ratioSampleIndex[ir];
        auto *fit = new TF1(Form("f_pol0_%s_%zu_%s", canvasStem.c_str(), sampleIndex, cent.c_str()), "pol0", xr.first, xr.second);
        fit->SetLineColor(samples[sampleIndex].color);
        fit->SetLineWidth(2);
        fit->SetLineStyle(1);
        const int fitStatus = ratio->Fit(fit, "Q0");
        if (fitStatus == 0) {
            fit->Draw("SAME");
            fits.push_back(fit);
            fitSampleIndex.push_back(sampleIndex);
        } else {
            delete fit;
        }
    }
    TLatex fitText;
    fitText.SetNDC();
    fitText.SetTextFont(42);
    fitText.SetTextSize(0.057);
    double yText = 0.83;
    for (size_t ifit = 0; ifit < fits.size(); ++ifit) {
        const size_t sampleIndex = fitSampleIndex[ifit];
        fitText.SetTextColor(samples[sampleIndex].color);
        fitText.DrawLatex(0.18, yText,
                          Form("%s/LHC23: %.3f #pm %.3f",
                               samples[sampleIndex].label,
                               fits[ifit]->GetParameter(0),
                               fits[ifit]->GetParError(0)));
        yText -= 0.115;
    }
    fitText.SetTextColor(kBlack);
    return out;
}

std::unique_ptr<TCanvas> DrawBdtAndRawOverEff(const std::vector<Sample> &samples,
                                              const std::vector<std::unique_ptr<TH1>> &bdt,
                                              const std::vector<std::unique_ptr<TH1>> &rawOverEff,
                                              const std::string &cent) {
    auto c = std::make_unique<TCanvas>(("c_bdt_raw_over_eff_" + cent).c_str(), "bdt and raw over bdt", 900, 850);
    auto *top = new TPad("top_bdt", "", 0.0, 0.50, 1.0, 1.0);
    auto *bot = new TPad("bot_raw_eff", "", 0.0, 0.0, 1.0, 0.50);
    top->SetBottomMargin(0.03);
    top->SetLeftMargin(0.13);
    top->SetRightMargin(0.04);
    bot->SetTopMargin(0.03);
    bot->SetBottomMargin(0.16);
    bot->SetLeftMargin(0.13);
    bot->SetRightMargin(0.04);
    top->Draw();
    bot->Draw();

    auto xr = XRangeFromHists(bdt);
    top->cd();
    auto *frameB = new TH1D(("frame_bdt_" + cent).c_str(), ";#it{p}_{T} (GeV/#it{c});#epsilon_{BDT}", 100, xr.first, xr.second);
    frameB->SetDirectory(nullptr);
    frameB->SetStats(false);
    frameB->SetMinimum(0.0);
    frameB->SetMaximum(1.05);
    frameB->GetXaxis()->SetLabelSize(0.0);
    frameB->GetYaxis()->SetTitleOffset(1.15);
    frameB->Draw();
    auto *leg = new TLegend(0.55, 0.58, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.035);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!bdt[i]) continue;
        bdt[i]->Draw("E1 SAME");
        leg->AddEntry(bdt[i].get(), samples[i].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.16, 0.86, Form("Centrality %s%%", cent.c_str()));

    bot->cd();
    double ymax = 0.0;
    for (const auto &h : rawOverEff) ymax = std::max(ymax, MaxHist(h.get()));
    auto *frameR = new TH1D(("frame_raw_eff_" + cent).c_str(), ";#it{p}_{T} (GeV/#it{c});N_{raw}/(#epsilon_{BDT} N_{evt})", 100, xr.first, xr.second);
    frameR->SetDirectory(nullptr);
    frameR->SetStats(false);
    frameR->SetMinimum(0.0);
    frameR->SetMaximum(ymax > 0.0 ? ymax * 1.35 : 1.0);
    frameR->GetYaxis()->SetTitleOffset(1.15);
    frameR->Draw();
    for (const auto &h : rawOverEff) {
        if (h) h->Draw("E1 SAME");
    }
    return c;
}

std::vector<double> EdgesFromHist(const TH1 *h) {
    std::vector<double> edges;
    if (!h) return edges;
    edges.reserve(h->GetNbinsX() + 1);
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        edges.push_back(h->GetXaxis()->GetBinLowEdge(ib));
    }
    edges.push_back(h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()));
    return edges;
}

std::vector<std::vector<std::unique_ptr<TH1>>> ComputeMCEfficiencies(
    const std::vector<Sample> &samples,
    const std::vector<const char *> &mcPaths,
    const std::vector<double> &centBins,
    const std::vector<std::vector<double>> &ptBinsPerCent) {
    std::vector<std::vector<std::unique_ptr<TH1>>> out(samples.size());
    for (auto &v : out) v.resize(ptBinsPerCent.size());

    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }

    for (size_t is = 0; is < samples.size(); ++is) {
        auto f = std::unique_ptr<TFile>(TFile::Open(mcPaths[is], "READ"));
        if (!f || f->IsZombie()) {
            std::cerr << "[Warn] cannot open MC file for " << samples[is].label << ": " << mcPaths[is] << std::endl;
            continue;
        }
        TChain chain("O2mchypcands");
        GeneralHelper::fillChainFromAO2D(chain, f.get());
        ROOT::RDataFrame rdf(chain);
        auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
        auto res = AcceptanceHelper::ComputeAcceptanceFlexible(ready,
                                                               {},
                                                               {},
                                                               {},
                                                               centBins,
                                                               ptBinsPerCent,
                                                               "fDecRad > 0.8",
                                                               {},
                                                               {},
                                                               true);
        for (size_t ic = 0; ic < ptBinsPerCent.size() && ic < res.acc_pt_per_cent.size(); ++ic) {
            TH1D *src = res.acc_pt_per_cent[ic];
            if (!src) continue;
            out[is][ic] = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(src->Clone(Form("h_mc_eff_%zu_%zu", is, ic))));
            if (!out[is][ic]) continue;
            out[is][ic]->SetDirectory(nullptr);
            out[is][ic]->SetTitle(";#it{p}_{T} (GeV/#it{c});MC efficiency");
            StyleHist(out[is][ic].get(), samples[is]);
        }
        res.Clear();
    }
    return out;
}

std::vector<std::unique_ptr<TH1>> ComputeMCEfficiencyCt(const std::vector<Sample> &samples,
                                                       const std::vector<const char *> &mcPaths,
                                                       const std::vector<double> &ctBins) {
    std::vector<std::unique_ptr<TH1>> out(samples.size());
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }

    for (size_t is = 0; is < samples.size(); ++is) {
        auto f = std::unique_ptr<TFile>(TFile::Open(mcPaths[is], "READ"));
        if (!f || f->IsZombie()) {
            std::cerr << "[Warn] cannot open MC file for " << samples[is].label << ": " << mcPaths[is] << std::endl;
            continue;
        }
        TChain chain("O2mchypcands");
        GeneralHelper::fillChainFromAO2D(chain, f.get());
        ROOT::RDataFrame rdf(chain);
        auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
        auto res = AcceptanceHelper::ComputeAcceptanceFlexible(ready,
                                                               {},
                                                               ctBins,
                                                               {},
                                                               {},
                                                               {},
                                                               "fDecRad > 0.8",
                                                               {},
                                                               {},
                                                               true);
        if (res.acc_ct_both) {
            out[is] = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(res.acc_ct_both->Clone(Form("h_mc_eff_ct_%zu", is))));
            if (out[is]) {
                out[is]->SetDirectory(nullptr);
                out[is]->SetTitle(";#it{c}t (cm);MC efficiency");
                StyleHist(out[is].get(), samples[is]);
            }
        }
        res.Clear();
    }
    return out;
}

RatioBundle DrawMCEfficiencyRatio(const std::vector<Sample> &samples,
                                  const std::vector<std::unique_ptr<TH1>> &mcEff,
                                  const std::string &cent) {
    return DrawPeriodPtRatio(samples, mcEff, cent, "mc_efficiency", "MC efficiency", "MC efficiency", false);
}

RatioBundle DrawMCEfficiencyCtRatio(const std::vector<Sample> &samples,
                                    const std::vector<std::unique_ptr<TH1>> &mcEffCt) {
    return DrawPeriodPtRatio(samples, mcEffCt, "ct_single", "mc_efficiency_ct", "MC efficiency", "ct single", false, "#it{c}t (cm)", "ct single");
}

std::unique_ptr<TH1> DivideRatioHists(const TH1 *num, const TH1 *den, const std::string &name, const std::string &yTitle) {
    if (!num || !den) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(num->Clone(name.c_str())));
    if (!out) return nullptr;
    out->SetDirectory(nullptr);
    out->Divide(den);
    out->SetTitle(Form(";#it{p}_{T} (GeV/#it{c});%s", yTitle.c_str()));
    out->GetYaxis()->SetTitle(yTitle.c_str());
    out->GetYaxis()->SetTitleSize(0.055);
    out->GetYaxis()->SetLabelSize(0.045);
    out->GetYaxis()->SetTitleOffset(1.05);
    out->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    out->GetXaxis()->SetTitleSize(0.055);
    out->GetXaxis()->SetLabelSize(0.045);
    return out;
}

Style_t ClosedMarkerForSample(size_t sampleIndex) {
    if (sampleIndex == 1) return kFullSquare;
    if (sampleIndex == 2) return kFullTriangleUp;
    return kFullCircle;
}

Style_t OpenMarkerForSample(size_t sampleIndex) {
    if (sampleIndex == 1) return kOpenSquare;
    if (sampleIndex == 2) return kOpenTriangleUp;
    return kOpenCircle;
}

std::pair<double, double> RatioRangePair(const TH1 *a, const TH1 *b) {
    std::vector<TH1 *> hists;
    if (a) hists.push_back(const_cast<TH1 *>(a));
    if (b) hists.push_back(const_cast<TH1 *>(b));
    return RatioRangeRaw(hists);
}

std::unique_ptr<TCanvas> DrawRatioComparisonPanels(const std::vector<Sample> &samples,
                                                   const std::vector<std::unique_ptr<TH1>> &dataRatios,
                                                   const std::vector<std::unique_ptr<TH1>> &mcRatios,
                                                   const std::string &cent,
                                                   const std::string &canvasStem,
                                                   const std::string &dataLabel,
                                                   const std::string &labelExtra,
                                                   const std::string &xTitle = "#it{p}_{T} (GeV/#it{c})",
                                                   const std::string &displayLabel = "") {
    auto c = std::make_unique<TCanvas>(("c_" + canvasStem + "_" + cent).c_str(), canvasStem.c_str(), 1240, 620);
    c->SetTicks(1, 1);
    c->Divide(2, 1, 0.015, 0.0);

    for (size_t ir = 0; ir < 2; ++ir) {
        const size_t sampleIndex = ir + 1;
        if (ir >= dataRatios.size() || ir >= mcRatios.size()) continue;
        TH1 *hData = dataRatios[ir].get();
        TH1 *hMc = mcRatios[ir].get();
        if (!hData || !hMc) continue;

        c->cd(static_cast<int>(ir + 1));
        gPad->SetTicks(1, 1);
        gPad->SetGridy(true);
        gPad->SetLeftMargin(ir == 0 ? 0.13 : 0.10);
        gPad->SetRightMargin(0.04);
        gPad->SetTopMargin(0.08);
        gPad->SetBottomMargin(0.14);

        std::vector<std::unique_ptr<TH1>> rangeHists;
        rangeHists.emplace_back(std::unique_ptr<TH1>(dynamic_cast<TH1 *>(hData->Clone(Form("h_range_data_%s_%zu", cent.c_str(), ir)))));
        rangeHists.emplace_back(std::unique_ptr<TH1>(dynamic_cast<TH1 *>(hMc->Clone(Form("h_range_mc_%s_%zu", cent.c_str(), ir)))));
        auto xr = XRangeFromHists(rangeHists);
        auto yr = RatioRangePair(hData, hMc);
        const double span = std::max(0.15, yr.second - yr.first);
        yr.first = std::max(0.0, yr.first - 0.10 * span);
        yr.second += 0.10 * span;

        auto *frame = new TH1D(Form("frame_%s_%s_%zu", canvasStem.c_str(), cent.c_str(), ir),
                               Form(";%s;Ratio / LHC23", xTitle.c_str()),
                               100,
                               xr.first,
                               xr.second);
        frame->SetDirectory(nullptr);
        frame->SetStats(false);
        frame->SetMinimum(yr.first);
        frame->SetMaximum(yr.second);
        StyleFrame(frame, 0.052, 0.044, ir == 0 ? 1.05 : 0.90);
        frame->GetYaxis()->SetNdivisions(505);
        frame->Draw();

        auto *one = new TLine(xr.first, 1.0, xr.second, 1.0);
        one->SetLineStyle(2);
        one->SetLineColor(kGray + 2);
        one->SetLineWidth(2);
        one->Draw("SAME");

        hData->SetLineColor(samples[sampleIndex].color);
        hData->SetMarkerColor(samples[sampleIndex].color);
        hData->SetMarkerStyle(ClosedMarkerForSample(sampleIndex));
        hData->SetMarkerSize(1.08);
        hData->SetLineWidth(2);

        hMc->SetLineColor(kGray + 2);
        hMc->SetMarkerColor(kGray + 2);
        hMc->SetMarkerStyle(OpenMarkerForSample(sampleIndex));
        hMc->SetMarkerSize(1.08);
        hMc->SetLineWidth(2);

        hData->Draw("E1 SAME");
        hMc->Draw("E1 SAME");

        auto *fitData = new TF1(Form("f_pol0_%s_data_%zu_%s", canvasStem.c_str(), ir, cent.c_str()), "pol0", xr.first, xr.second);
        fitData->SetLineColor(samples[sampleIndex].color);
        fitData->SetLineWidth(3);
        fitData->SetLineStyle(1);
        const int fitDataStatus = hData->Fit(fitData, "Q0");
        if (fitDataStatus == 0) fitData->Draw("SAME");
        else delete fitData;

        auto *fitMc = new TF1(Form("f_pol0_%s_mc_%zu_%s", canvasStem.c_str(), ir, cent.c_str()), "pol0", xr.first, xr.second);
        fitMc->SetLineColor(kGray + 2);
        fitMc->SetLineWidth(3);
        fitMc->SetLineStyle(7);
        const int fitMcStatus = hMc->Fit(fitMc, "Q0");
        if (fitMcStatus == 0) fitMc->Draw("SAME");
        else delete fitMc;

        auto *leg = new TLegend(0.18, 0.66, 0.88, 0.88);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextFont(42);
        leg->SetTextSize(0.034);
        leg->AddEntry(hData, Form("%s %s", samples[sampleIndex].label, dataLabel.c_str()), "lep");
        leg->AddEntry(hMc, Form("%s MC eff. ratio", samples[sampleIndex].label), "lep");
        if (fitDataStatus == 0) leg->AddEntry(fitData, "counts pol0", "l");
        if (fitMcStatus == 0) leg->AddEntry(fitMc, "MC pol0", "l");
        leg->Draw();

        TLatex title;
        title.SetNDC();
        title.SetTextFont(42);
        title.SetTextSize(0.040);
        const std::string label = displayLabel.empty() ? CentralityLabel(cent) : displayLabel;
        title.DrawLatex(0.18, 0.93, Form("%s / LHC23, %s", samples[sampleIndex].label, label.c_str()));
        title.SetTextSize(0.030);
        title.DrawLatex(0.18, 0.895, labelExtra.c_str());
    }
    return c;
}

std::unique_ptr<TCanvas> DrawEventCountsVsCentrality(const std::vector<Sample> &samples,
                                                     const std::vector<std::string> &cents,
                                                     const std::vector<std::vector<RawSummary>> &allSummaries) {
    auto c = std::make_unique<TCanvas>("c_events_vs_centrality", "events vs centrality", 900, 700);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    auto *frame = new TH1D("frame_evt_cent", ";Centrality (%);N_{evt}", 100, 0.0, 80.0);
    frame->SetDirectory(nullptr);
    frame->SetStats(false);
    double ymax = 0.0;
    for (const auto &perSample : allSummaries) {
        for (const auto &s : perSample) ymax = std::max(ymax, s.nEvents);
    }
    frame->SetMaximum(ymax > 0.0 ? ymax * 1.35 : 1.0);
    frame->SetMinimum(0.0);
    frame->Draw();
    auto *leg = new TLegend(0.55, 0.63, 0.90, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.035);
    for (size_t is = 0; is < samples.size(); ++is) {
        auto *h = new TH1D(Form("h_evt_cent_%zu", is), ";Centrality (%);N_{evt}", 4, 0.0, 80.0);
        h->SetDirectory(nullptr);
        h->SetStats(false);
        StyleHist(h, samples[is]);
        for (size_t ic = 0; ic < allSummaries[is].size(); ++ic) {
            h->SetBinContent(static_cast<int>(ic + 1), allSummaries[is][ic].nEvents);
        }
        h->Draw("P E1 SAME");
        leg->AddEntry(h, samples[is].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.16, 0.86, "Event counts from N_{raw}/(N_{raw}/N_{evt})");
    return c;
}

std::unique_ptr<TCanvas> DrawIntegralYield(const std::vector<Sample> &samples,
                                           const std::vector<std::unique_ptr<TH1>> &stat,
                                           const std::vector<std::unique_ptr<TGraphAsymmErrors>> &sys) {
    auto c = std::make_unique<TCanvas>("c_integral_yield_period_merged", "integral yield", 980, 780);
    c->SetTicks(1, 1);
    c->SetGridy(true);
    c->SetLeftMargin(0.14);
    c->SetRightMargin(0.04);
    c->SetTopMargin(0.06);
    c->SetBottomMargin(0.12);
    double ymax = 0.0;
    for (size_t i = 0; i < stat.size(); ++i) ymax = std::max(ymax, std::max(MaxHist(stat[i].get()), MaxGraph(sys[i].get())));
    auto *frame = new TH1D("frame_integral_yield", ";Centrality (%);dN/dy", 100, 0.0, 80.0);
    frame->SetDirectory(nullptr);
    frame->SetStats(false);
    frame->SetMinimum(0.0);
    frame->SetMaximum(ymax > 0.0 ? ymax * 1.45 : 1.0);
    StyleFrame(frame, 0.048, 0.042, 1.25);
    frame->Draw();
    auto *leg = new TLegend(0.62, 0.68, 0.91, 0.91);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.032);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (sys[i]) sys[i]->Draw("2 SAME");
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!stat[i]) continue;
        stat[i]->Draw("E1 SAME");
        leg->AddEntry(stat[i].get(), samples[i].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.17, 0.86, "Integrated yield");
    return c;
}

std::unique_ptr<TCanvas> DrawIntegralYieldVsNcharged(const std::vector<Sample> &samples,
                                                     const std::vector<std::unique_ptr<TGraphAsymmErrors>> &stat,
                                                     const std::vector<std::unique_ptr<TGraphAsymmErrors>> &sys) {
    auto c = std::make_unique<TCanvas>("c_integral_yield_vs_ncharged_periods", "integral yield vs ncharged", 980, 780);
    c->SetTicks(1, 1);
    c->SetGridy(true);
    c->SetLeftMargin(0.14);
    c->SetRightMargin(0.04);
    c->SetTopMargin(0.06);
    c->SetBottomMargin(0.12);

    double xmin = 1e30;
    double xmax = -1e30;
    double ymax = 0.0;
    for (size_t i = 0; i < stat.size(); ++i) {
        if (!stat[i]) continue;
        ymax = std::max(ymax, std::max(MaxGraph(stat[i].get()), MaxGraph(sys[i].get())));
        for (int ip = 0; ip < stat[i]->GetN(); ++ip) {
            double x = 0.0, y = 0.0;
            stat[i]->GetPoint(ip, x, y);
            xmin = std::min(xmin, x - stat[i]->GetErrorXlow(ip));
            xmax = std::max(xmax, x + stat[i]->GetErrorXhigh(ip));
        }
    }
    if (!(xmin < xmax)) {
        xmin = 0.0;
        xmax = 2000.0;
    }
    const double xpad = 0.08 * (xmax - xmin);

    auto *frame = new TH1D("frame_integral_yield_vs_ncharged",
                           ";#LTd#it{N}_{ch}/d#eta#GT;dN/dy",
                           100,
                           std::max(0.0, xmin - xpad),
                           xmax + xpad);
    frame->SetDirectory(nullptr);
    frame->SetStats(false);
    frame->SetMinimum(0.0);
    frame->SetMaximum(ymax > 0.0 ? ymax * 1.45 : 1.0);
    StyleFrame(frame, 0.048, 0.042, 1.25);
    frame->Draw();

    auto *leg = new TLegend(0.62, 0.68, 0.91, 0.91);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.032);
    const double boxHalfWidth = 0.018 * (xmax - xmin);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!sys[i]) continue;
        for (int ip = 0; ip < sys[i]->GetN(); ++ip) {
            double x = 0.0, y = 0.0;
            sys[i]->GetPoint(ip, x, y);
            const double eyLow = sys[i]->GetErrorYlow(ip);
            const double eyHigh = sys[i]->GetErrorYhigh(ip);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(eyLow) || !std::isfinite(eyHigh)) continue;
            auto *box = new TBox(std::max(0.0, x - boxHalfWidth),
                                 std::max(0.0, y - eyLow),
                                 x + boxHalfWidth,
                                 y + eyHigh);
            box->SetFillColorAlpha(samples[i].color, 0.20);
            box->SetLineColor(samples[i].color);
            box->SetLineWidth(1);
            box->Draw("SAME");
        }
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!stat[i]) continue;
        stat[i]->Draw("P E1 SAME");
        leg->AddEntry(stat[i].get(), samples[i].label, "lep");
    }
    leg->Draw();
    DrawExperimentLabel(0.17, 0.86, "Integrated yield vs multiplicity");
    return c;
}
} // namespace

void ComparePeriodMergedSpectrumQA(
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/PeriodMergedQA",
    const char *lhc23File = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
    const char *lhc24File = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC24ar_pass3_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
    const char *lhc25File = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC25_PbPb_pass1_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
    const char *lhc23CtFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/ct_single/both/spectrum.root",
    const char *lhc24CtFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC24ar_pass3_CustomV0s_HadronPID/ct_single/both/spectrum.root",
    const char *lhc25CtFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC25_PbPb_pass1_CustomV0s_HadronPID/ct_single/both/spectrum.root") {
    gSystem->mkdir(outDir, true);
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.0);

    std::vector<Sample> samples = {
        {"LHC23 pass5", lhc23File, kBlack, kFullCircle},
        {"LHC24ar pass3", lhc24File, kOrange + 7, kOpenSquare},
        {"LHC25 pass1", lhc25File, kAzure + 2, kOpenTriangleUp},
    };
    std::vector<Sample> mcSamples = {
        {"LHC23 pass5 MC", "", kBlack, kFullCircle},
        {"LHC24ar pass3 MC", "", kOrange + 7, kOpenSquare},
        {"LHC25 pass1 MC", "", kAzure + 2, kOpenTriangleUp},
    };
    std::vector<std::unique_ptr<TFile>> files;
    for (const auto &s : samples) {
        files.emplace_back(TFile::Open(s.fileName, "READ"));
        if (!files.back() || files.back()->IsZombie()) {
            std::cerr << "ERROR: cannot open " << s.fileName << std::endl;
            return;
        }
    }
    std::vector<const char *> ctFileNames = {lhc23CtFile, lhc24CtFile, lhc25CtFile};
    std::vector<std::unique_ptr<TFile>> ctFiles;
    for (const auto *fileName : ctFileNames) {
        ctFiles.emplace_back(TFile::Open(fileName, "READ"));
        if (!ctFiles.back() || ctFiles.back()->IsZombie()) {
            std::cerr << "ERROR: cannot open ct file " << fileName << std::endl;
            return;
        }
    }

    const std::vector<std::string> cents = {"0_10", "10_30", "30_50", "50_80"};
    std::vector<std::vector<RawSummary>> summaries(samples.size());
    for (auto &v : summaries) v.resize(cents.size());

    const std::vector<double> centBins = {0.0, 10.0, 30.0, 50.0, 80.0};
    std::vector<std::vector<double>> ptBinsPerCent;
    ptBinsPerCent.reserve(cents.size());
    for (const auto &cent : cents) {
        auto hRawBinning = GetHist(files[0].get(), "cen_" + cent + "/std/h_raw_counts", "h_raw_binning_" + cent);
        ptBinsPerCent.push_back(EdgesFromHist(hRawBinning.get()));
    }
    const std::vector<const char *> mcPaths = {
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5_G4list/reweighted/AO2D_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6_G4list/reweighted/AO2D_combined_reweighted.root",
    };
    auto mcEffAll = ComputeMCEfficiencies(mcSamples, mcPaths, centBins, ptBinsPerCent);
    auto hCtBinning = GetHist(ctFiles[0].get(), "ct_single/std/h_raw_counts", "h_ct_binning");
    const std::vector<double> ctBins = EdgesFromHist(hCtBinning.get());
    auto mcEffCt = ComputeMCEfficiencyCt(mcSamples, mcPaths, ctBins);

    for (size_t ic = 0; ic < cents.size(); ++ic) {
        const auto &cent = cents[ic];
        std::vector<std::unique_ptr<TH1>> stat(samples.size());
        std::vector<std::unique_ptr<TGraphAsymmErrors>> sys(samples.size());
        std::vector<std::unique_ptr<TH1>> raw(samples.size());
        std::vector<std::unique_ptr<TH1>> bdt(samples.size());
        std::vector<std::unique_ptr<TH1>> rawOverNevtSummary(samples.size());
        std::vector<std::unique_ptr<TH1>> rawOverNevtPt(samples.size());
        std::vector<std::unique_ptr<TH1>> rawOverEff(samples.size());

        for (size_t is = 0; is < samples.size(); ++is) {
            const std::string base = "cen_" + cent;
            stat[is] = GetHist(files[is].get(), base + "/h_final_spectrum_stat", Form("h_spec_%zu_%s", is, cent.c_str()));
            sys[is] = GetGraph(files[is].get(), base + "/g_final_spectrum_sys", Form("g_spec_sys_%zu_%s", is, cent.c_str()));
            raw[is] = GetHist(files[is].get(), base + "/std/h_raw_counts", Form("h_raw_%zu_%s", is, cent.c_str()));
            bdt[is] = GetHist(files[is].get(), base + "/std/h_bdt_efficiency", Form("h_bdt_%zu_%s", is, cent.c_str()));
            rawOverNevtSummary[is] = GetHist(files[is].get(), base + "/std/h_raw_over_nevents", Form("h_raw_nevt_summary_%zu_%s", is, cent.c_str()));
            summaries[is][ic] = BuildRawSummary(raw[is].get(), rawOverNevtSummary[is].get());

            StyleHist(stat[is].get(), samples[is]);
            StyleHist(raw[is].get(), samples[is]);
            StyleHist(bdt[is].get(), samples[is]);
            StyleSys(sys[is].get(), samples[is]);
            rawOverNevtPt[is] = MakeRawOverEventsPtHist(raw[is].get(), summaries[is][ic].nEvents,
                                                        Form("h_raw_over_nevt_pt_%zu_%s", is, cent.c_str()));
            StyleHist(rawOverNevtPt[is].get(), samples[is]);
            rawOverEff[is] = MakeRawOverBdtOverEvents(raw[is].get(), bdt[is].get(), summaries[is][ic].nEvents,
                                                      Form("h_raw_over_eff_nevt_%zu_%s", is, cent.c_str()));
            StyleHist(rawOverEff[is].get(), samples[is]);
        }

        auto cSpec = DrawSpectrum(samples, stat, sys, cent);
        cSpec->SaveAs((std::string(outDir) + "/spectrum_compare_cen_" + cent + ".pdf").c_str());

        std::vector<RawSummary> centSummaries;
        centSummaries.reserve(samples.size());
        for (size_t is = 0; is < samples.size(); ++is) centSummaries.push_back(summaries[is][ic]);
        auto cRaw = DrawRawAndEvents(samples, raw, centSummaries, cent);
        cRaw->SaveAs((std::string(outDir) + "/raw_counts_and_nevents_cen_" + cent + ".pdf").c_str());

        auto rawOverNevtBundle = DrawPeriodPtRatio(samples,
                                                   rawOverNevtPt,
                                                   cent,
                                                   "raw_over_nevents_pt",
                                                   "N_{raw}/N_{evt}",
                                                   "N_{raw}/N_{evt}");
        rawOverNevtBundle.canvas->SaveAs((std::string(outDir) + "/raw_over_nevents_pt_cen_" + cent + ".pdf").c_str());

        auto rawOverBdtBundle = DrawPeriodPtRatio(samples,
                                                  rawOverEff,
                                                  cent,
                                                  "raw_over_bdt_nevents_pt",
                                                  "N_{raw}/(#epsilon_{BDT} N_{evt})",
                                                  "N_{raw}/(#epsilon_{BDT} N_{evt})");
        rawOverBdtBundle.canvas->SaveAs((std::string(outDir) + "/raw_over_bdt_nevents_pt_cen_" + cent + ".pdf").c_str());

        std::vector<std::unique_ptr<TH1>> mcEff(samples.size());
        for (size_t is = 0; is < samples.size(); ++is) {
            if (is >= mcEffAll.size() || ic >= mcEffAll[is].size() || !mcEffAll[is][ic]) continue;
            mcEff[is] = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(mcEffAll[is][ic]->Clone(Form("h_mc_eff_draw_%zu_%zu", is, ic))));
            if (!mcEff[is]) continue;
            mcEff[is]->SetDirectory(nullptr);
            StyleHist(mcEff[is].get(), mcSamples[is]);
        }
        auto mcEffBundle = DrawMCEfficiencyRatio(mcSamples, mcEff, cent);
        mcEffBundle.canvas->SaveAs((std::string(outDir) + "/mc_efficiency_pt_cen_" + cent + ".pdf").c_str());

        auto cRawNevtVsMc = DrawRatioComparisonPanels(samples,
                                                      rawOverNevtBundle.ratios,
                                                      mcEffBundle.ratios,
                                                      cent,
                                                      "raw_over_nevents_ratio_vs_mc_eff_ratio",
                                                      "counts ratio",
                                                      "N_{raw}/N_{evt} ratio vs MC efficiency ratio");
        cRawNevtVsMc->SaveAs((std::string(outDir) + "/raw_over_nevents_ratio_vs_mc_eff_ratio_cen_" + cent + ".pdf").c_str());

        auto cRawBdtVsMc = DrawRatioComparisonPanels(samples,
                                                     rawOverBdtBundle.ratios,
                                                     mcEffBundle.ratios,
                                                     cent,
                                                     "raw_over_bdt_nevents_ratio_vs_mc_eff_ratio",
                                                     "counts/BDT ratio",
                                                     "N_{raw}/(#epsilon_{BDT}N_{evt}) ratio vs MC efficiency ratio");
        cRawBdtVsMc->SaveAs((std::string(outDir) + "/raw_over_bdt_nevents_ratio_vs_mc_eff_ratio_cen_" + cent + ".pdf").c_str());
    }

    std::vector<std::unique_ptr<TH1>> rawCt(samples.size());
    std::vector<std::unique_ptr<TH1>> bdtCt(samples.size());
    std::vector<std::unique_ptr<TH1>> rawOverNevtCt(samples.size());
    std::vector<std::unique_ptr<TH1>> rawOverBdtNevtCt(samples.size());
    for (size_t is = 0; is < samples.size(); ++is) {
        rawCt[is] = GetHist(ctFiles[is].get(), "ct_single/std/h_raw_counts", Form("h_raw_ct_%zu", is));
        bdtCt[is] = GetHist(ctFiles[is].get(), "ct_single/std/h_bdt_efficiency", Form("h_bdt_ct_%zu", is));
        StyleHist(rawCt[is].get(), samples[is]);
        StyleHist(bdtCt[is].get(), samples[is]);
        double nEvents = 0.0;
        for (const auto &summary : summaries[is]) nEvents += summary.nEvents;
        rawOverNevtCt[is] = MakeRawOverEventsPtHist(rawCt[is].get(), nEvents, Form("h_raw_over_nevt_ct_%zu", is));
        if (rawOverNevtCt[is]) rawOverNevtCt[is]->SetTitle(";#it{c}t (cm);N_{raw}/N_{evt}");
        StyleHist(rawOverNevtCt[is].get(), samples[is]);
        rawOverBdtNevtCt[is] = MakeRawOverBdtOverEvents(rawCt[is].get(), bdtCt[is].get(), nEvents, Form("h_raw_over_bdt_nevt_ct_%zu", is));
        if (rawOverBdtNevtCt[is]) rawOverBdtNevtCt[is]->SetTitle(";#it{c}t (cm);N_{raw}/(#epsilon_{BDT} N_{evt})");
        StyleHist(rawOverBdtNevtCt[is].get(), samples[is]);
    }

    auto rawOverNevtCtBundle = DrawPeriodPtRatio(samples,
                                                 rawOverNevtCt,
                                                 "ct_single",
                                                 "raw_over_nevents_ct",
                                                 "N_{raw}/N_{evt}",
                                                 "N_{raw}/N_{evt}",
                                                 true,
                                                 "#it{c}t (cm)",
                                                 "ct single");
    rawOverNevtCtBundle.canvas->SaveAs((std::string(outDir) + "/raw_over_nevents_ct.pdf").c_str());

    auto rawOverBdtCtBundle = DrawPeriodPtRatio(samples,
                                                rawOverBdtNevtCt,
                                                "ct_single",
                                                "raw_over_bdt_nevents_ct",
                                                "N_{raw}/(#epsilon_{BDT} N_{evt})",
                                                "N_{raw}/(#epsilon_{BDT} N_{evt})",
                                                true,
                                                "#it{c}t (cm)",
                                                "ct single");
    rawOverBdtCtBundle.canvas->SaveAs((std::string(outDir) + "/raw_over_bdt_nevents_ct.pdf").c_str());

    auto mcEffCtBundle = DrawMCEfficiencyCtRatio(mcSamples, mcEffCt);
    mcEffCtBundle.canvas->SaveAs((std::string(outDir) + "/mc_efficiency_ct.pdf").c_str());

    auto cRawNevtCtVsMc = DrawRatioComparisonPanels(samples,
                                                    rawOverNevtCtBundle.ratios,
                                                    mcEffCtBundle.ratios,
                                                    "ct_single",
                                                    "raw_over_nevents_ct_ratio_vs_mc_eff_ratio",
                                                    "counts ratio",
                                                    "N_{raw}/N_{evt} ratio vs MC efficiency ratio",
                                                    "#it{c}t (cm)",
                                                    "ct single");
    cRawNevtCtVsMc->SaveAs((std::string(outDir) + "/raw_over_nevents_ct_ratio_vs_mc_eff_ratio.pdf").c_str());

    auto cRawBdtCtVsMc = DrawRatioComparisonPanels(samples,
                                                   rawOverBdtCtBundle.ratios,
                                                   mcEffCtBundle.ratios,
                                                   "ct_single",
                                                   "raw_over_bdt_nevents_ct_ratio_vs_mc_eff_ratio",
                                                   "counts/BDT ratio",
                                                   "N_{raw}/(#epsilon_{BDT}N_{evt}) ratio vs MC efficiency ratio",
                                                   "#it{c}t (cm)",
                                                   "ct single");
    cRawBdtCtVsMc->SaveAs((std::string(outDir) + "/raw_over_bdt_nevents_ct_ratio_vs_mc_eff_ratio.pdf").c_str());

    auto cEvt = DrawEventCountsVsCentrality(samples, cents, summaries);
    cEvt->SaveAs((std::string(outDir) + "/nevents_vs_centrality.pdf").c_str());

    std::vector<std::unique_ptr<TH1>> iyStat(samples.size());
    std::vector<std::unique_ptr<TGraphAsymmErrors>> iySys(samples.size());
    for (size_t is = 0; is < samples.size(); ++is) {
        iyStat[is] = GetHist(files[is].get(), "summary/integral_yield/h_integral_yield_stat", Form("h_iy_stat_%zu", is));
        auto hSys = GetHist(files[is].get(), "summary/integral_yield/h_integral_yield_sys", Form("h_iy_sys_%zu", is));
        iySys[is] = GraphFromHistSys(hSys.get(), Form("g_iy_sys_%zu", is));
        StyleHist(iyStat[is].get(), samples[is]);
        StyleSys(iySys[is].get(), samples[is]);
    }
    auto cIy = DrawIntegralYield(samples, iyStat, iySys);
    cIy->SaveAs((std::string(outDir) + "/integral_yield_vs_centrality.pdf").c_str());

    std::vector<std::unique_ptr<TGraphAsymmErrors>> iyMultStat(samples.size());
    std::vector<std::unique_ptr<TGraphAsymmErrors>> iyMultSys(samples.size());
    for (size_t is = 0; is < samples.size(); ++is) {
        iyMultStat[is] = GetGraph(files[is].get(),
                                  "summary/integral_yield/g_integral_yield_vs_multiplicity_stat",
                                  Form("g_iy_mult_stat_%zu", is));
        iyMultSys[is] = GetGraph(files[is].get(),
                                 "summary/integral_yield/g_integral_yield_vs_multiplicity_sys",
                                 Form("g_iy_mult_sys_%zu", is));
        StyleGraphStat(iyMultStat[is].get(), samples[is]);
        StyleSys(iyMultSys[is].get(), samples[is]);
    }
    auto cIyMult = DrawIntegralYieldVsNcharged(samples, iyMultStat, iyMultSys);
    cIyMult->SaveAs((std::string(outDir) + "/integral_yield_vs_ncharged.pdf").c_str());

    std::cout << "Saved QA plots under: " << outDir << std::endl;
}
