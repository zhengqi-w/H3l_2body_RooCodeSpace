#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TROOT.h>
#include <TString.h>
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
struct SpectrumItem {
    double cenMin{0.0};
    double cenMax{0.0};
    int color{kBlack};
    std::unique_ptr<TH1D> stat;
    std::unique_ptr<TGraphAsymmErrors> sys;
};

struct Pol0FitResult {
    TF1 *func{nullptr};
    double value{0.0};
    double error{0.0};
    double chi2{0.0};
    int ndf{0};
    bool ok{false};
};

std::string EdgeToken(double x) {
    TString s = Form("%.6g", x);
    s.ReplaceAll(".", "p");
    return s.Data();
}

std::string CenDir(double min, double max) {
    return "cen_" + EdgeToken(min) + "_" + EdgeToken(max);
}

std::string CenLabel(double min, double max) {
    return Form("%.0f-%.0f%%", min, max);
}

int RainbowColor(size_t i, size_t n) {
    const double frac = (n <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    // Reverse the rainbow so the most central bin starts from the deep-red end.
    return TColor::GetColorPalette(static_cast<int>((1.0 - frac) * (gStyle->GetNumberOfColors() - 1)));
}

Style_t CentralityMarker(size_t i) {
    static const Style_t markers[] = {20, 21, 22, 23, 29, 33, 34, 47, 24, 25, 26, 32};
    return markers[i % (sizeof(markers) / sizeof(markers[0]))];
}

template <class T>
std::unique_ptr<T> CloneObject(TFile &file, const std::string &path, const std::string &name) {
    auto *obj = dynamic_cast<T *>(file.Get(path.c_str()));
    if (!obj) return nullptr;
    auto *clone = dynamic_cast<T *>(obj->Clone(name.c_str()));
    return std::unique_ptr<T>(clone);
}

std::vector<SpectrumItem> LoadSpectra(const char *fileName, const std::vector<double> &edges, const char *tag) {
    std::vector<SpectrumItem> out;
    std::unique_ptr<TFile> file(TFile::Open(fileName, "READ"));
    if (!file || file->IsZombie()) {
        std::cerr << "[Error] Cannot open " << fileName << std::endl;
        return out;
    }

    const size_t nCent = (edges.size() > 1) ? edges.size() - 1 : 0;
    for (size_t i = 0; i < nCent; ++i) {
        const std::string dir = CenDir(edges[i], edges[i + 1]);
        SpectrumItem item;
        item.cenMin = edges[i];
        item.cenMax = edges[i + 1];
        item.color = RainbowColor(i, nCent);
        item.stat = CloneObject<TH1D>(*file, dir + "/h_final_spectrum_stat", Form("h_%s_%s", tag, dir.c_str()));
        item.sys = CloneObject<TGraphAsymmErrors>(*file, dir + "/g_final_spectrum_sys", Form("g_%s_%s", tag, dir.c_str()));
        if (!item.stat) {
            std::cerr << "[Warn] Missing " << dir << "/h_final_spectrum_stat in " << fileName << std::endl;
            continue;
        }
        item.stat->SetDirectory(nullptr);
        item.stat->SetStats(false);
        item.stat->SetLineColor(item.color);
        item.stat->SetMarkerColor(item.color);
        item.stat->SetLineWidth(2);
        item.stat->SetMarkerSize(1.15);
        if (item.sys) {
            item.sys->SetLineColor(item.color);
            item.sys->SetFillColorAlpha(item.color, 0.22);
            item.sys->SetMarkerColor(item.color);
            item.sys->SetLineWidth(1);
        }
        out.push_back(std::move(item));
    }
    return out;
}

void ApplyMarkerStyle(std::vector<SpectrumItem> &items, Style_t marker) {
    for (auto &item : items) {
        item.stat->SetMarkerStyle(marker);
        if (item.sys) item.sys->SetMarkerStyle(marker);
    }
}

void FindRange(const std::vector<const std::vector<SpectrumItem> *> &sets, double &xmin, double &xmax, double &ymin, double &ymax) {
    xmin = 1e9;
    xmax = -1e9;
    ymin = 1e9;
    ymax = -1e9;
    for (auto *set : sets) {
        if (!set) continue;
        for (const auto &item : *set) {
            auto *h = item.stat.get();
            for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
                const double y = h->GetBinContent(ib);
                if (y <= 0.0) continue;
                const double x = h->GetBinCenter(ib);
                const double ex = 0.5 * h->GetBinWidth(ib);
                const double ey = h->GetBinError(ib);
                xmin = std::min(xmin, x - ex);
                xmax = std::max(xmax, x + ex);
                ymin = std::min(ymin, std::max(1e-30, y - ey));
                ymax = std::max(ymax, y + ey);
            }
            if (item.sys) {
                for (int ip = 0; ip < item.sys->GetN(); ++ip) {
                    double x = 0.0, y = 0.0;
                    item.sys->GetPoint(ip, x, y);
                    if (y <= 0.0) continue;
                    xmin = std::min(xmin, x - item.sys->GetErrorXlow(ip));
                    xmax = std::max(xmax, x + item.sys->GetErrorXhigh(ip));
                    ymin = std::min(ymin, std::max(1e-30, y - item.sys->GetErrorYlow(ip)));
                    ymax = std::max(ymax, y + item.sys->GetErrorYhigh(ip));
                }
            }
        }
    }
    if (!(xmin < xmax) || !(ymin < ymax)) {
        xmin = 1.5;
        xmax = 8.5;
        ymin = 1e-9;
        ymax = 1e-2;
    }
}

void DrawHeader(double x, double y) {
    TLatex text;
    text.SetNDC();
    text.SetTextFont(42);
    text.SetTextSize(0.026);
    text.DrawLatex(x, y, "ALICE Work In Progress");
    text.SetTextSize(0.022);
    text.DrawLatex(x, y - 0.042, "Pb--Pb, #sqrt{#it{s}_{NN}} = 5.36 TeV");
    text.DrawLatex(x, y - 0.084, "Merged periods:");
    text.DrawLatex(x, y - 0.122, "LHC23 pass5");
    text.DrawLatex(x, y - 0.160, "LHC24ar pass3");
    text.DrawLatex(x, y - 0.198, "LHC25 pass1");
    text.DrawLatex(x, y - 0.248, "{}^{3}_{#Lambda}H #rightarrow {}^{3}He + #pi");
}

TLegend *BuildCentralityLegend(const std::vector<SpectrumItem> &items, double x1, double y1, double x2, double y2) {
    auto *leg = new TLegend(x1, y1, x2, y2);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.028);
    for (const auto &item : items) {
        leg->AddEntry(item.stat.get(), CenLabel(item.cenMin, item.cenMax).c_str(), "p");
    }
    return leg;
}

void DrawSysThenStat(const std::vector<SpectrumItem> &items, const char *statOpt) {
    for (const auto &item : items) {
        if (!item.sys) continue;
        for (int ip = 0; ip < item.sys->GetN(); ++ip) {
            double x = 0.0;
            double y = 0.0;
            item.sys->GetPoint(ip, x, y);
            if (!std::isfinite(x) || !std::isfinite(y) || y <= 0.0) continue;
            auto *box = new TBox(x - item.sys->GetErrorXlow(ip),
                                 std::max(1e-30, y - item.sys->GetErrorYlow(ip)),
                                 x + item.sys->GetErrorXhigh(ip),
                                 y + item.sys->GetErrorYhigh(ip));
            box->SetFillColorAlpha(item.color, 0.20);
            box->SetLineColor(item.color);
            box->SetLineWidth(1);
            box->Draw("SAME");
        }
    }
    for (const auto &item : items) {
        item.stat->Draw(statOpt);
    }
}

const SpectrumItem *FindSpectrumItem(const std::vector<SpectrumItem> &items, double cenMin, double cenMax) {
    for (const auto &item : items) {
        if (std::fabs(item.cenMin - cenMin) < 1e-6 && std::fabs(item.cenMax - cenMax) < 1e-6) return &item;
    }
    return nullptr;
}

std::vector<SpectrumItem> MakeAntimatterMatterRatios(const std::vector<SpectrumItem> &matter,
                                                     const std::vector<SpectrumItem> &antimatter) {
    std::vector<SpectrumItem> out;
    for (const auto &anti : antimatter) {
        const auto *mat = FindSpectrumItem(matter, anti.cenMin, anti.cenMax);
        if (!mat || !mat->stat || !anti.stat) {
            std::cerr << "[Warn] Cannot build antimatter/matter ratio for "
                      << CenLabel(anti.cenMin, anti.cenMax) << std::endl;
            continue;
        }
        SpectrumItem item;
        item.cenMin = anti.cenMin;
        item.cenMax = anti.cenMax;
        item.color = anti.color;
        item.stat = std::unique_ptr<TH1D>(static_cast<TH1D *>(anti.stat->Clone(
            Form("h_ratio_antimatter_matter_%s", CenDir(anti.cenMin, anti.cenMax).c_str()))));
        item.stat->SetDirectory(nullptr);
        item.stat->Divide(mat->stat.get());
        item.stat->SetStats(false);
        item.stat->SetLineColor(item.color);
        item.stat->SetMarkerColor(item.color);
        item.stat->SetMarkerStyle(CentralityMarker(out.size()));
        item.stat->SetMarkerSize(1.25);
        item.stat->SetLineWidth(2);
        if (anti.sys && mat->sys && anti.sys->GetN() >= anti.stat->GetNbinsX() && mat->sys->GetN() >= mat->stat->GetNbinsX()) {
            item.sys = std::make_unique<TGraphAsymmErrors>(item.stat->GetNbinsX());
            item.sys->SetName(Form("g_ratio_antimatter_matter_sys_%s", CenDir(anti.cenMin, anti.cenMax).c_str()));
            item.sys->SetLineColor(item.color);
            item.sys->SetFillStyle(0);
            item.sys->SetMarkerColor(item.color);
            item.sys->SetLineStyle(kDashed);
            item.sys->SetLineWidth(2);
            for (int ib = 1; ib <= item.stat->GetNbinsX(); ++ib) {
                const int ip = ib - 1;
                const double a = anti.stat->GetBinContent(ib);
                const double m = mat->stat->GetBinContent(ib);
                const double r = item.stat->GetBinContent(ib);
                const double x = item.stat->GetBinCenter(ib);
                const double ex = 0.5 * item.stat->GetBinWidth(ib);
                double ax = 0.0, ay = 0.0, mx = 0.0, my = 0.0;
                anti.sys->GetPoint(ip, ax, ay);
                mat->sys->GetPoint(ip, mx, my);
                double eyLow = 0.0;
                double eyHigh = 0.0;
                if (a > 0.0 && m > 0.0 && r > 0.0) {
                    const double aLow = anti.sys->GetErrorYlow(ip);
                    const double aHigh = anti.sys->GetErrorYhigh(ip);
                    const double mLow = mat->sys->GetErrorYlow(ip);
                    const double mHigh = mat->sys->GetErrorYhigh(ip);
                    eyLow = std::sqrt((aLow / m) * (aLow / m) + (a * mHigh / (m * m)) * (a * mHigh / (m * m)));
                    eyHigh = std::sqrt((aHigh / m) * (aHigh / m) + (a * mLow / (m * m)) * (a * mLow / (m * m)));
                }
                item.sys->SetPoint(ip, x, r);
                item.sys->SetPointError(ip, ex, ex, eyLow, eyHigh);
            }
        }
        out.push_back(std::move(item));
    }
    return out;
}

void FindHistOnlyRange(const std::vector<SpectrumItem> &items, double &xmin, double &xmax, double &ymin, double &ymax) {
    xmin = 1e9;
    xmax = -1e9;
    ymin = 1e9;
    ymax = -1e9;
    for (const auto &item : items) {
        if (!item.stat) continue;
        auto *h = item.stat.get();
        xmin = std::min(xmin, h->GetXaxis()->GetXmin());
        xmax = std::max(xmax, h->GetXaxis()->GetXmax());
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double y = h->GetBinContent(ib);
            const double ey = h->GetBinError(ib);
            if (!std::isfinite(y) || y <= 0.0) continue;
            ymin = std::min(ymin, std::max(0.0, y - ey));
            ymax = std::max(ymax, y + ey);
        }
        if (item.sys) {
            for (int ip = 0; ip < item.sys->GetN(); ++ip) {
                double x = 0.0, y = 0.0;
                item.sys->GetPoint(ip, x, y);
                if (!std::isfinite(y) || y <= 0.0) continue;
                ymin = std::min(ymin, std::max(0.0, y - item.sys->GetErrorYlow(ip)));
                ymax = std::max(ymax, y + item.sys->GetErrorYhigh(ip));
            }
        }
    }
    if (!(xmin < xmax)) {
        xmin = 1.5;
        xmax = 8.5;
    }
    if (!(ymin < ymax)) {
        ymin = 0.0;
        ymax = 2.0;
    } else {
        const double span = std::max(0.2, ymax - ymin);
        ymin = std::max(0.0, ymin - 0.25 * span);
        ymax += 0.25 * span;
        ymin = std::min(ymin, 0.9);
        ymax = std::max(ymax, 1.1);
    }
}

void DrawRatioSysBoxes(const SpectrumItem &item, bool filled) {
    if (!item.sys) return;
    for (int ip = 0; ip < item.sys->GetN(); ++ip) {
        double x = 0.0, y = 0.0;
        item.sys->GetPoint(ip, x, y);
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        auto *box = new TBox(x - item.sys->GetErrorXlow(ip),
                             std::max(0.0, y - item.sys->GetErrorYlow(ip)),
                             x + item.sys->GetErrorXhigh(ip),
                             y + item.sys->GetErrorYhigh(ip));
        if (filled) {
            box->SetFillColorAlpha(item.color, 0.22);
            box->SetLineColor(item.color);
            box->SetLineWidth(1);
        } else {
            box->SetFillStyle(0);
            box->SetLineColor(item.color);
            box->SetLineStyle(kDashed);
            box->SetLineWidth(2);
        }
        box->Draw("SAME");
    }
}

Pol0FitResult FitRatioPol0(const SpectrumItem &item, const std::string &name) {
    Pol0FitResult out;
    if (!item.stat) return out;
    const double xmin = item.stat->GetXaxis()->GetXmin();
    const double xmax = item.stat->GetXaxis()->GetXmax();
    auto *fit = new TF1(name.c_str(), "pol0", xmin, xmax);
    fit->SetLineColor(item.color);
    fit->SetLineStyle(1);
    fit->SetLineWidth(2);
    const int status = item.stat->Fit(fit, "Q0");
    out.func = fit;
    out.value = fit->GetParameter(0);
    out.error = fit->GetParError(0);
    out.chi2 = fit->GetChisquare();
    out.ndf = fit->GetNDF();
    out.ok = (status == 0);
    return out;
}

} // namespace

void DrawMergedCentralitySpectra(
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/MergedCentralitySpectra",
    const char *matterFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/matter/spectrum.root",
    const char *antimatterFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/antimatter/spectrum.root",
    const char *bothFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root") {
    // Manually adjust the centrality binning here when needed.
    std::vector<double> centralityEdges = {0, 5, 10, 20, 30, 40, 50, 60, 70, 90};

    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    gSystem->mkdir(outDir, true);

    auto matter = LoadSpectra(matterFile, centralityEdges, "matter");
    auto antimatter = LoadSpectra(antimatterFile, centralityEdges, "antimatter");
    auto both = LoadSpectra(bothFile, centralityEdges, "both");
    ApplyMarkerStyle(matter, kFullCircle);
    ApplyMarkerStyle(antimatter, kOpenSquare);
    ApplyMarkerStyle(both, kFullDiamond);

    double xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0;
    FindRange({&matter, &antimatter}, xmin, xmax, ymin, ymax);
    auto cMA = std::make_unique<TCanvas>("c_merged_matter_antimatter_spectra", "", 1200, 850);
    cMA->SetLogy();
    cMA->SetLeftMargin(0.14);
    cMA->SetRightMargin(0.26);
    cMA->SetTopMargin(0.05);
    cMA->SetBottomMargin(0.12);
    auto *frameMA = cMA->DrawFrame(xmin * 0.92, ymin * 0.35, xmax * 1.08, ymax * 8.0,
                                   ";#it{p}_{T} (GeV/#it{c});1/#it{N}_{ev} d^{2}#it{N}/d#it{y}d#it{p}_{T} [(GeV/#it{c})^{-1}]");
    frameMA->GetYaxis()->SetTitleOffset(1.55);
    DrawSysThenStat(matter, "E SAME");
    DrawSysThenStat(antimatter, "E SAME");
    DrawHeader(0.76, 0.94);

    auto *legCentMA = BuildCentralityLegend(matter.empty() ? antimatter : matter, 0.76, 0.16, 0.98, 0.54);
    legCentMA->Draw();
    auto *legParticle = new TLegend(0.76, 0.57, 0.98, 0.66);
    legParticle->SetBorderSize(0);
    legParticle->SetFillStyle(0);
    legParticle->SetTextFont(42);
    legParticle->SetTextSize(0.032);
    if (!matter.empty()) legParticle->AddEntry(matter.front().stat.get(), "{}^{3}_{#Lambda}H", "pe");
    if (!antimatter.empty()) legParticle->AddEntry(antimatter.front().stat.get(), "{}^{3}_{#bar{#Lambda}}#bar{H}", "pe");
    legParticle->Draw();
    cMA->SaveAs((std::string(outDir) + "/merged_centrality_spectra_matter_antimatter.pdf").c_str());

    FindRange({&both}, xmin, xmax, ymin, ymax);
    auto cBoth = std::make_unique<TCanvas>("c_merged_both_spectra", "", 1200, 850);
    cBoth->SetLogy();
    cBoth->SetLeftMargin(0.14);
    cBoth->SetRightMargin(0.26);
    cBoth->SetTopMargin(0.05);
    cBoth->SetBottomMargin(0.12);
    auto *frameBoth = cBoth->DrawFrame(xmin * 0.92, ymin * 0.35, xmax * 1.08, ymax * 8.0,
                                       ";#it{p}_{T} (GeV/#it{c});1/#it{N}_{ev} d^{2}#it{N}/d#it{y}d#it{p}_{T} [(GeV/#it{c})^{-1}]");
    frameBoth->GetYaxis()->SetTitleOffset(1.55);
    DrawSysThenStat(both, "E SAME");
    DrawHeader(0.76, 0.94);

    auto *legCentBoth = BuildCentralityLegend(both, 0.76, 0.16, 0.98, 0.54);
    legCentBoth->Draw();
    auto *legBoth = new TLegend(0.76, 0.57, 0.98, 0.66);
    legBoth->SetBorderSize(0);
    legBoth->SetFillStyle(0);
    legBoth->SetTextFont(42);
    legBoth->SetTextSize(0.032);
    if (!both.empty()) legBoth->AddEntry(both.front().stat.get(), "({}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H})/2", "pe");
    legBoth->Draw();
    cBoth->SaveAs((std::string(outDir) + "/merged_centrality_spectra_both.pdf").c_str());

    auto ratios = MakeAntimatterMatterRatios(matter, antimatter);
    FindHistOnlyRange(ratios, xmin, xmax, ymin, ymax);
    auto cRatio = std::make_unique<TCanvas>("c_merged_antimatter_matter_ratio", "", 1200, 850);
    cRatio->SetLeftMargin(0.14);
    cRatio->SetRightMargin(0.26);
    cRatio->SetTopMargin(0.05);
    cRatio->SetBottomMargin(0.12);
    auto *frameRatio = cRatio->DrawFrame(xmin * 0.92, ymin, xmax * 1.08, ymax,
                                         ";#it{p}_{T} (GeV/#it{c});{}^{3}_{#bar{#Lambda}}#bar{H} / {}^{3}_{#Lambda}H");
    frameRatio->GetYaxis()->SetTitleOffset(1.55);
    for (const auto &item : ratios) {
        DrawRatioSysBoxes(item, true);
    }
    for (const auto &item : ratios) {
        item.stat->Draw("E SAME");
    }
    TF1 *overviewFitDummy = nullptr;
    for (const auto &item : ratios) {
        auto fit = FitRatioPol0(item, Form("f_pol0_ratio_overview_%s", CenDir(item.cenMin, item.cenMax).c_str()));
        if (fit.func) {
            fit.func->Draw("SAME");
            if (!overviewFitDummy) overviewFitDummy = fit.func;
        }
    }
    auto *unity = new TLine(xmin * 0.92, 1.0, xmax * 1.08, 1.0);
    unity->SetLineStyle(2);
    unity->SetLineColor(kGray + 2);
    unity->Draw("SAME");
    DrawHeader(0.76, 0.94);
    auto *legCentRatio = BuildCentralityLegend(ratios, 0.76, 0.16, 0.98, 0.54);
    legCentRatio->Draw();
    auto *legRatio = new TLegend(0.76, 0.57, 0.98, 0.66);
    legRatio->SetBorderSize(0);
    legRatio->SetFillStyle(0);
    legRatio->SetTextFont(42);
    legRatio->SetTextSize(0.030);
    if (!ratios.empty()) {
        legRatio->AddEntry(ratios.front().stat.get(), "stat.", "pe");
        if (ratios.front().sys) legRatio->AddEntry(ratios.front().sys.get(), "syst.", "f");
        if (overviewFitDummy) legRatio->AddEntry(overviewFitDummy, "pol0 fit", "l");
    }
    legRatio->Draw();
    cRatio->SaveAs((std::string(outDir) + "/merged_centrality_antimatter_over_matter_ratio.pdf").c_str());

    for (const auto &item : ratios) {
        std::vector<SpectrumItem> oneItem;
        SpectrumItem rangeItem;
        rangeItem.cenMin = item.cenMin;
        rangeItem.cenMax = item.cenMax;
        rangeItem.color = item.color;
        rangeItem.stat = std::unique_ptr<TH1D>(static_cast<TH1D *>(item.stat->Clone(
            Form("h_range_ratio_%s", CenDir(item.cenMin, item.cenMax).c_str()))));
        rangeItem.stat->SetDirectory(nullptr);
        if (item.sys) {
            rangeItem.sys = std::unique_ptr<TGraphAsymmErrors>(static_cast<TGraphAsymmErrors *>(item.sys->Clone(
                Form("g_range_ratio_sys_%s", CenDir(item.cenMin, item.cenMax).c_str()))));
        }
        oneItem.push_back(std::move(rangeItem));
        double rxmin = 0.0, rxmax = 0.0, rymin = 0.0, rymax = 0.0;
        FindHistOnlyRange(oneItem, rxmin, rxmax, rymin, rymax);

        const std::string tag = CenDir(item.cenMin, item.cenMax);
        auto cSingleRatio = std::make_unique<TCanvas>(("c_antimatter_matter_ratio_" + tag).c_str(), "", 900, 750);
        cSingleRatio->SetLeftMargin(0.14);
        cSingleRatio->SetRightMargin(0.04);
        cSingleRatio->SetTopMargin(0.06);
        cSingleRatio->SetBottomMargin(0.12);
        auto *frameSingle = cSingleRatio->DrawFrame(rxmin * 0.92, rymin, rxmax * 1.08, rymax,
                                                    ";#it{p}_{T} (GeV/#it{c});{}^{3}_{#bar{#Lambda}}#bar{H} / {}^{3}_{#Lambda}H");
        frameSingle->GetYaxis()->SetTitleOffset(1.35);
        DrawRatioSysBoxes(item, false);
        item.stat->Draw("E SAME");
        auto fit = FitRatioPol0(item, Form("f_pol0_ratio_%s", tag.c_str()));
        if (fit.func) fit.func->Draw("SAME");
        auto *singleUnity = new TLine(rxmin * 0.92, 1.0, rxmax * 1.08, 1.0);
        singleUnity->SetLineStyle(2);
        singleUnity->SetLineColor(kGray + 2);
        singleUnity->Draw("SAME");

        TLatex singleText;
        singleText.SetNDC();
        singleText.SetTextFont(42);
        singleText.SetTextSize(0.038);
        singleText.DrawLatex(0.18, 0.88, "ALICE Work In Progress");
        singleText.SetTextSize(0.032);
        singleText.DrawLatex(0.18, 0.83, "Pb--Pb, #sqrt{#it{s}_{NN}} = 5.36 TeV");
        singleText.DrawLatex(0.18, 0.78, Form("Centrality %s", CenLabel(item.cenMin, item.cenMax).c_str()));
        if (fit.func) {
            singleText.DrawLatex(0.18, 0.73, Form("pol0 = %.3f #pm %.3f", fit.value, fit.error));
            singleText.DrawLatex(0.18, 0.68, Form("#chi^{2}/NDF = %.1f/%d", fit.chi2, fit.ndf));
        }

        auto *singleLeg = new TLegend(0.60, 0.74, 0.90, 0.88);
        singleLeg->SetBorderSize(0);
        singleLeg->SetFillStyle(0);
        singleLeg->SetTextFont(42);
        singleLeg->SetTextSize(0.034);
        singleLeg->AddEntry(item.stat.get(), "stat.", "pe");
        auto *singleSysDummy = new TLine(0.0, 0.0, 1.0, 0.0);
        singleSysDummy->SetLineColor(item.color);
        singleSysDummy->SetLineStyle(kDashed);
        singleSysDummy->SetLineWidth(2);
        singleLeg->AddEntry(singleSysDummy, "syst.", "l");
        if (fit.func) singleLeg->AddEntry(fit.func, "pol0 fit", "l");
        singleLeg->Draw();

        cSingleRatio->SaveAs((std::string(outDir) + "/antimatter_over_matter_ratio_" + tag + ".pdf").c_str());
    }

    std::cout << "Saved plots in: " << outDir << std::endl;
}
