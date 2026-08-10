#include <TCanvas.h>
#include <TColor.h>
#include <TDirectory.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kCCmPerPs = 0.0299792458;

struct RadSpectrum {
    double radMin{0.};
    double radMax{0.};
    std::string dirName;
    std::unique_ptr<TH1D> hist;
    std::unique_ptr<TF1> fit;
};

bool ParseRadDirName(const std::string &name, double &radMin, double &radMax)
{
    if (name.rfind("rad_", 0) != 0) return false;
    std::string rest = name.substr(4);
    std::vector<double> vals;
    size_t pos = 0;
    while (pos < rest.size()) {
        size_t next = rest.find('_', pos);
        const std::string token = rest.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        try {
            vals.push_back(std::stod(token));
        } catch (...) {
            return false;
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    if (vals.size() != 2) return false;
    radMin = vals[0];
    radMax = vals[1];
    return true;
}

TF1 *FindFirstTF1(TDirectory *dir)
{
    if (!dir) return nullptr;
    TIter next(dir->GetListOfKeys());
    while (auto *obj = next()) {
        auto *key = dynamic_cast<TKey *>(obj);
        if (!key) continue;
        TObject *item = key->ReadObj();
        if (auto *f = dynamic_cast<TF1 *>(item)) {
            return f;
        }
        delete item;
    }
    return nullptr;
}

TGraphAsymmErrors *HistToGraph(const TH1D *h, const std::string &name, Color_t color, Style_t marker)
{
    if (!h) return nullptr;
    auto *g = new TGraphAsymmErrors(h->GetNbinsX());
    g->SetName(name.c_str());
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const int ip = ib - 1;
        g->SetPoint(ip, h->GetBinCenter(ib), h->GetBinContent(ib));
        g->SetPointError(ip, 0.5 * h->GetBinWidth(ib), 0.5 * h->GetBinWidth(ib),
                         h->GetBinError(ib), h->GetBinError(ib));
    }
    g->SetLineColor(color);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(0.95);
    g->SetLineWidth(2);
    return g;
}

std::vector<Color_t> MakeColors(size_t n)
{
    std::vector<Color_t> colors;
    colors.reserve(n);
    std::vector<double> stops = {0.00, 0.18, 0.36, 0.54, 0.72, 1.00};
    std::vector<double> red   = {0.72, 0.93, 0.95, 0.15, 0.05, 0.38};
    std::vector<double> green = {0.05, 0.42, 0.80, 0.62, 0.25, 0.08};
    std::vector<double> blue  = {0.05, 0.06, 0.16, 0.75, 0.78, 0.52};
    const int first = TColor::CreateGradientColorTable(static_cast<int>(stops.size()),
                                                       stops.data(), red.data(), green.data(), blue.data(),
                                                       std::max(2, static_cast<int>(n)));
    for (size_t i = 0; i < n; ++i) colors.push_back(first + static_cast<int>(i));
    return colors;
}

Color_t MakePaleColor(Color_t baseColor, double alpha = 0.55)
{
    auto *base = gROOT->GetColor(baseColor);
    if (!base) return baseColor;
    const double r = base->GetRed() * alpha + (1.0 - alpha);
    const double g = base->GetGreen() * alpha + (1.0 - alpha);
    const double b = base->GetBlue() * alpha + (1.0 - alpha);
    return TColor::GetColor(static_cast<Float_t>(r),
                            static_cast<Float_t>(g),
                            static_cast<Float_t>(b));
}

double PositiveMin(const std::vector<RadSpectrum> &items)
{
    double ymin = 1e300;
    for (const auto &item : items) {
        if (!item.hist) continue;
        for (int ib = 1; ib <= item.hist->GetNbinsX(); ++ib) {
            const double y = item.hist->GetBinContent(ib);
            if (y > 0.) ymin = std::min(ymin, y);
        }
    }
    return std::isfinite(ymin) && ymin < 1e299 ? ymin : 1e-8;
}

double MaxWithErrorAndFit(const std::vector<RadSpectrum> &items)
{
    double ymax = 0.;
    for (const auto &item : items) {
        if (item.hist) {
            for (int ib = 1; ib <= item.hist->GetNbinsX(); ++ib) {
                ymax = std::max(ymax, item.hist->GetBinContent(ib) + item.hist->GetBinError(ib));
            }
        }
        if (item.fit) {
            const double xmin = item.fit->GetXmin();
            const double xmax = item.fit->GetXmax();
            for (int i = 0; i <= 200; ++i) {
                const double x = xmin + (xmax - xmin) * i / 200.;
                const double y = item.fit->Eval(x);
                if (std::isfinite(y) && y > 0.) ymax = std::max(ymax, y);
            }
        }
    }
    return ymax > 0. ? ymax : 1.;
}

std::string TauTextLine(const RadSpectrum &item)
{
    if (!item.fit || item.fit->GetNpar() < 2) {
        return Form("%.1f-%.1f: no fit", item.radMin, item.radMax);
    }
    const double tau = item.fit->GetParameter(1) / kCCmPerPs;
    const double tauErr = item.fit->GetParError(1) / kCCmPerPs;
    if (std::isfinite(tauErr) && tauErr > 0.) {
        return Form("%.1f-%.1f: %.1f #pm %.1f ps", item.radMin, item.radMax, tau, tauErr);
    }
    return Form("%.1f-%.1f: %.1f ps", item.radMin, item.radMax, tau);
}

std::unique_ptr<TCanvas> DrawTauPerRadBin(const TH1D *tauHist, const std::string &outputDir)
{
    if (!tauHist) return nullptr;
    auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(tauHist->Clone("h_tau_per_radbin_beautified")));
    h->SetDirectory(nullptr);
    h->SetStats(false);
    h->SetTitle("");
    h->GetXaxis()->SetTitle("#it{R}_{dec} (cm)");
    h->GetYaxis()->SetTitle("#tau (ps)");
    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetXaxis()->SetLabelSize(0.038);
    h->GetYaxis()->SetLabelSize(0.038);
    h->GetYaxis()->SetTitleOffset(1.18);
    h->SetMarkerStyle(kFullCircle);
    h->SetMarkerColor(kAzure + 2);
    h->SetLineColor(kAzure + 2);
    h->SetLineWidth(2);
    h->SetMarkerSize(1.15);

    double ymin = 1e9;
    double ymax = -1e9;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double y = h->GetBinContent(ib);
        if (y <= 0.) continue;
        ymin = std::min(ymin, y - h->GetBinError(ib));
        ymax = std::max(ymax, y + h->GetBinError(ib));
    }
    if (!std::isfinite(ymin) || !std::isfinite(ymax) || ymax <= ymin) {
        ymin = 0.;
        ymax = 320.;
    }
    ymin = std::max(0., ymin - 0.18 * (ymax - ymin));
    ymax = ymax + 0.28 * (ymax - ymin);
    h->SetMinimum(ymin);
    h->SetMaximum(ymax);

    auto c = std::make_unique<TCanvas>("c_tau_per_radbin_beautified", "", 960, 720);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.06);
    c->SetTicks(1, 1);

    h->Draw("E1 X0");

    TLine lifetimeLine(h->GetXaxis()->GetXmin(), 253., h->GetXaxis()->GetXmax(), 253.);
    lifetimeLine.SetLineColor(kGray + 2);
    lifetimeLine.SetLineStyle(7);
    lifetimeLine.SetLineWidth(2);
    lifetimeLine.Draw("same");
    h->Draw("E1 X0 SAME");

    TLegend leg(0.56, 0.72, 0.90, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.032);
    leg.AddEntry(h.get(), "Fit #tau per #it{R}_{dec} bin", "pe");
    leg.AddEntry(&lifetimeLine, "#tau = 253 ps", "l");
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, "Pb--Pb #sqrt{#it{s}_{NN}} = 5.36 TeV");
    text.DrawLatex(0.16, 0.78, "Run 3 merged, rad-ct both");

    c->SaveAs((outputDir + "/c_tau_per_radbin_beautified.pdf").c_str());
    h->Write();
    c->Write();
    return c;
}

} // namespace

void DrawRadCtFinalSpectrumOverlay(
    const char *inputPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/rad_ct/both/spectrum.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtFinalSpectrumOverlay")
{
    std::filesystem::create_directories(outputDir);

    TFile input(inputPath, "READ");
    if (input.IsZombie()) {
        std::cerr << "[DrawRadCtFinalSpectrumOverlay] cannot open " << inputPath << "\n";
        return;
    }

    std::vector<RadSpectrum> spectra;
    TIter next(input.GetListOfKeys());
    while (auto *obj = next()) {
        auto *key = dynamic_cast<TKey *>(obj);
        if (!key) continue;
        if (std::string(key->GetClassName()) != "TDirectoryFile") continue;
        double rmin = 0., rmax = 0.;
        const std::string dirName = key->GetName();
        if (!ParseRadDirName(dirName, rmin, rmax)) continue;
        auto *dir = dynamic_cast<TDirectory *>(input.Get(dirName.c_str()));
        if (!dir) continue;

        auto *h = dynamic_cast<TH1D *>(dir->Get("h_final_spectrum_stat"));
        auto *stdDir = dynamic_cast<TDirectory *>(dir->Get("std"));
        auto *f = FindFirstTF1(stdDir);
        if (!h && !f) continue;

        RadSpectrum item;
        item.radMin = rmin;
        item.radMax = rmax;
        item.dirName = dirName;
        if (h) {
            item.hist.reset(static_cast<TH1D *>(h->Clone(("h_overlay_" + dirName).c_str())));
            item.hist->SetDirectory(nullptr);
        }
        if (f) {
            item.fit.reset(static_cast<TF1 *>(f->Clone(("f_overlay_" + dirName).c_str())));
            item.fit->SetNpx(800);
        }
        spectra.push_back(std::move(item));
    }

    std::sort(spectra.begin(), spectra.end(), [](const RadSpectrum &a, const RadSpectrum &b) {
        return a.radMin < b.radMin;
    });
    if (spectra.empty()) {
        std::cerr << "[DrawRadCtFinalSpectrumOverlay] no rad spectra found in " << inputPath << "\n";
        return;
    }

    gStyle->SetOptStat(0);
    auto colors = MakeColors(spectra.size());
    const std::vector<Style_t> markers = {
        kFullCircle, kFullSquare, kFullTriangleUp, kFullDiamond,
        kOpenCircle, kOpenSquare, kOpenTriangleUp, kOpenDiamond
    };

    TCanvas c("c_rad_ct_final_spectrum_overlay", "", 1250, 860);
    c.SetLeftMargin(0.13);
    c.SetRightMargin(0.30);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetLogy();
    c.SetTicks(1, 1);

    const double ymin = PositiveMin(spectra) * 0.30;
    const double ymax = MaxWithErrorAndFit(spectra) * 8.0;
    TH1D frame("h_frame_rad_ct_final_spectrum_overlay",
               ";#it{c}t (cm);Final spectrum / corrected counts",
               100, 0., 42.);
    frame.SetMinimum(ymin);
    frame.SetMaximum(ymax);
    frame.GetXaxis()->SetTitleSize(0.045);
    frame.GetYaxis()->SetTitleSize(0.045);
    frame.GetXaxis()->SetLabelSize(0.038);
    frame.GetYaxis()->SetLabelSize(0.038);
    frame.GetYaxis()->SetTitleOffset(1.35);
    frame.Draw("AXIS");

    TLegend leg(0.72, 0.11, 0.98, 0.91);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.022);
    leg.SetNColumns(1);

    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    for (size_t i = 0; i < spectra.size(); ++i) {
        const Color_t color = colors[i];
        const Color_t fitColor = MakePaleColor(color, 0.58);
        const Style_t marker = markers[i % markers.size()];
        auto &item = spectra[i];
        if (item.fit) {
            item.fit->SetLineColor(fitColor);
            item.fit->SetLineWidth(2);
            item.fit->SetLineStyle(7);
            item.fit->Draw("SAME");
        }
        if (item.hist) {
            auto g = std::unique_ptr<TGraphAsymmErrors>(
                HistToGraph(item.hist.get(), "g_" + item.dirName, color, marker));
            g->Draw("PZ SAME");
            leg.AddEntry(g.get(), Form("%.1f < #it{R}_{dec} < %.1f cm", item.radMin, item.radMax), "pe");
            graphs.push_back(std::move(g));
        } else if (item.fit) {
            leg.AddEntry(item.fit.get(), Form("%.1f < #it{R}_{dec} < %.1f cm", item.radMin, item.radMax), "l");
        }
    }

    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.034);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.030);
    text.DrawLatex(0.16, 0.83, "Pb--Pb #sqrt{#it{s}_{NN}} = 5.36 TeV");
    text.DrawLatex(0.16, 0.78, "Run 3 merged, H^{3}_{#Lambda} + #bar{H}^{3}_{#Lambda}, rad-ct both");
    text.DrawLatex(0.16, 0.73, "Points: final spectrum, lines: exponential fits");
    text.SetTextSize(0.016);
    text.DrawLatex(0.16, 0.305, "Exp-fit #tau values:");
    const size_t nRows = (spectra.size() + 1) / 2;
    for (size_t i = 0; i < spectra.size(); ++i) {
        const bool secondCol = i >= nRows;
        const double x = secondCol ? 0.38 : 0.16;
        const double y = 0.282 - 0.021 * static_cast<double>(secondCol ? i - nRows : i);
        text.DrawLatex(x, y, TauTextLine(spectra[i]).c_str());
    }

    const std::string outPdf = std::string(outputDir) + "/rad_ct_final_spectrum_all_radbins_overlay.pdf";
    const std::string outRoot = std::string(outputDir) + "/rad_ct_final_spectrum_all_radbins_overlay.root";
    c.SaveAs(outPdf.c_str());

    TFile out(outRoot.c_str(), "RECREATE");
    c.Write();
    for (const auto &item : spectra) {
        if (item.hist) item.hist->Write();
        if (item.fit) item.fit->Write();
    }
    auto *tauHist = dynamic_cast<TH1D *>(input.Get("tau_per_radbin"));
    auto tauCanvas = DrawTauPerRadBin(tauHist, outputDir);
    out.Close();
    std::cout << "[DrawRadCtFinalSpectrumOverlay] wrote " << outPdf << "\n";
}
