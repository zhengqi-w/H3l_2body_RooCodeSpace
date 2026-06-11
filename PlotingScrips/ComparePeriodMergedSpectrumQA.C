#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

std::unique_ptr<TH1> MakeRatioHist(const TH1 *num, const TH1 *den, const std::string &name) {
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
    out->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
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
    const char *lhc25File = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC25_PbPb_pass1_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root") {
    gSystem->mkdir(outDir, true);
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.0);

    std::vector<Sample> samples = {
        {"LHC23 pass5", lhc23File, kBlack, 20},
        {"LHC24ar pass3", lhc24File, kOrange + 7, 21},
        {"LHC25 pass1", lhc25File, kAzure + 2, 22},
    };
    std::vector<std::unique_ptr<TFile>> files;
    for (const auto &s : samples) {
        files.emplace_back(TFile::Open(s.fileName, "READ"));
        if (!files.back() || files.back()->IsZombie()) {
            std::cerr << "ERROR: cannot open " << s.fileName << std::endl;
            return;
        }
    }

    const std::vector<std::string> cents = {"0_10", "10_30", "30_50", "50_80"};
    std::vector<std::vector<RawSummary>> summaries(samples.size());
    for (auto &v : summaries) v.resize(cents.size());

    for (size_t ic = 0; ic < cents.size(); ++ic) {
        const auto &cent = cents[ic];
        std::vector<std::unique_ptr<TH1>> stat(samples.size());
        std::vector<std::unique_ptr<TGraphAsymmErrors>> sys(samples.size());
        std::vector<std::unique_ptr<TH1>> raw(samples.size());
        std::vector<std::unique_ptr<TH1>> bdt(samples.size());
        std::vector<std::unique_ptr<TH1>> rawOverEff(samples.size());

        for (size_t is = 0; is < samples.size(); ++is) {
            const std::string base = "cen_" + cent;
            stat[is] = GetHist(files[is].get(), base + "/h_final_spectrum_stat", Form("h_spec_%zu_%s", is, cent.c_str()));
            sys[is] = GetGraph(files[is].get(), base + "/g_final_spectrum_sys", Form("g_spec_sys_%zu_%s", is, cent.c_str()));
            raw[is] = GetHist(files[is].get(), base + "/std/h_raw_counts", Form("h_raw_%zu_%s", is, cent.c_str()));
            bdt[is] = GetHist(files[is].get(), base + "/std/h_bdt_efficiency", Form("h_bdt_%zu_%s", is, cent.c_str()));
            auto hRawOverNevt = GetHist(files[is].get(), base + "/std/h_raw_over_nevents", Form("h_raw_nevt_%zu_%s", is, cent.c_str()));
            summaries[is][ic] = BuildRawSummary(raw[is].get(), hRawOverNevt.get());

            StyleHist(stat[is].get(), samples[is]);
            StyleHist(raw[is].get(), samples[is]);
            StyleHist(bdt[is].get(), samples[is]);
            StyleSys(sys[is].get(), samples[is]);
            rawOverEff[is] = MakeRawOverBdtOverEvents(raw[is].get(), bdt[is].get(), summaries[is][ic].nEvents,
                                                      Form("h_raw_over_eff_nevt_%zu_%s", is, cent.c_str()));
            StyleHist(rawOverEff[is].get(), samples[is]);
        }

        auto cSpec = DrawSpectrum(samples, stat, sys, cent);
        cSpec->SaveAs((std::string(outDir) + "/spectrum_compare_cen_" + cent + ".pdf").c_str());
        cSpec->SaveAs((std::string(outDir) + "/spectrum_compare_cen_" + cent + ".png").c_str());

        std::vector<RawSummary> centSummaries;
        centSummaries.reserve(samples.size());
        for (size_t is = 0; is < samples.size(); ++is) centSummaries.push_back(summaries[is][ic]);
        auto cRaw = DrawRawAndEvents(samples, raw, centSummaries, cent);
        cRaw->SaveAs((std::string(outDir) + "/raw_counts_and_nevents_cen_" + cent + ".pdf").c_str());

        auto cBdt = DrawBdtAndRawOverEff(samples, bdt, rawOverEff, cent);
        cBdt->SaveAs((std::string(outDir) + "/bdt_eff_and_raw_over_bdt_nevents_cen_" + cent + ".pdf").c_str());
    }

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
    cIy->SaveAs((std::string(outDir) + "/integral_yield_vs_centrality.png").c_str());

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
    cIyMult->SaveAs((std::string(outDir) + "/integral_yield_vs_ncharged.png").c_str());

    std::cout << "Saved QA plots under: " << outDir << std::endl;
}
