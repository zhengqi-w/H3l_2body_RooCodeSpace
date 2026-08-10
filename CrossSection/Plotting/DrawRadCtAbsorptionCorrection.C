#include "../../Tools/AbsorptionHelper.h"
#include "../../Tools/GeneralHelper.hpp"
#include "../../Tools/tasks/BinningCorrectionHelper.h"

#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr double kCcmPerPs = 0.0299792458;

std::vector<double> JsonVector(const GeneralHelper::Json &j)
{
    if (!j.is_array()) return {};
    return j.get<std::vector<double>>();
}

std::vector<std::vector<double>> JsonVector2D(const GeneralHelper::Json &j)
{
    if (!j.is_array()) return {};
    return j.get<std::vector<std::vector<double>>>();
}

std::vector<double> ParametersBaseRadBins()
{
    return {0.8, 1.1, 1.5, 2.1, 3, 4, 5, 6, 8, 10, 12.5, 15, 17.5, 20, 25, 30, 35};
}

std::vector<std::vector<double>> ParametersBaseCtBinsByRad()
{
    return {
        {0.4, 0.6, 0.8, 0.9, 1, 1.1, 1.3, 1.5},
        {0.5, 0.8, 1, 1.2, 1.3, 1.6, 1.8, 2.1, 3},
        {0.5, 1, 1.5, 1.7, 2, 2.5, 3.5},
        {1, 1.5, 2, 2.2, 2.5, 2.9, 3.5, 4.5, 6},
        {1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 9},
        {1.5, 2, 3, 4, 4.5, 5, 6, 7, 9},
        {2, 3, 4, 5, 5.5, 6, 6.5, 7.5, 10},
        {2, 4, 5, 6, 6.5, 7, 7.5, 8, 10, 14},
        {3, 5, 7, 8, 9, 10, 12, 16},
        {5, 7, 9, 11, 12, 13, 15, 20},
        {5, 7, 10, 12, 14, 16, 18, 22},
        {7, 10, 12, 14, 16, 18, 25},
        {10, 13, 17, 20, 23, 30},
        {10, 15, 18, 21, 26, 33},
        {15, 17, 20, 23, 27, 35},
        {15, 20, 25, 30, 40}
    };
}

void SetPlotStyle()
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetTitleSize(0.045, "XYZ");
    gStyle->SetLabelSize(0.038, "XYZ");
    gStyle->SetPadLeftMargin(0.13);
    gStyle->SetPadRightMargin(0.15);
    gStyle->SetPadBottomMargin(0.12);
    gStyle->SetPalette(kBird);
}

std::vector<double> BuildGlobalCtEdges(const std::vector<std::vector<double>> &ctBinsByRad)
{
    std::vector<double> edges;
    for (const auto &row : ctBinsByRad) {
        edges.insert(edges.end(), row.begin(), row.end());
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-9;
    }), edges.end());
    return edges;
}

TH1D *MakeDensityClone(const TH1F *src, const std::string &name)
{
    auto *h = new TH1D(name.c_str(), src->GetTitle(), src->GetNbinsX(), src->GetXaxis()->GetXbins()->GetArray());
    h->SetDirectory(nullptr);
    for (int ib = 1; ib <= src->GetNbinsX(); ++ib) {
        const double width = src->GetXaxis()->GetBinWidth(ib);
        const double val = width > 0.0 ? src->GetBinContent(ib) / width : src->GetBinContent(ib);
        const double err = width > 0.0 ? src->GetBinError(ib) / width : src->GetBinError(ib);
        h->SetBinContent(ib, val);
        h->SetBinError(ib, err);
    }
    return h;
}

int CountNonZeroBins(const TH1 *h)
{
    int n = 0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        if (h->GetBinContent(ib) > 0.0) ++n;
    }
    return n;
}

void DrawStamp(double x, double y, const std::string &text, double size = 0.035)
{
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(size);
    latex.DrawLatex(x, y, text.c_str());
}

} // namespace

void DrawRadCtAbsorptionCorrection(
    const std::string configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json",
    const std::string outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/AbsorptionMCQA",
    const std::vector<double> radBinsOverride = {},
    const std::string parametersBasePath = "",
    const std::string absorptionFileOverride = "",
    const std::string outputPrefix = "")
{
    SetPlotStyle();

    const auto cfg = GeneralHelper::LoadJsonFile(configPath);
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto path = common.value("path", GeneralHelper::Json::object());
    const auto trees = common.value("tree_names", GeneralHelper::Json::object());
    const auto params = common.value("parameters", GeneralHelper::Json::object());
    const auto binning = common.value("binning", GeneralHelper::Json::object());

    const std::string absorptionFileFromConfig = path.value("mc_file_for_absorption", std::string(""));
    const std::string absorptionFile = absorptionFileOverride.empty() ? absorptionFileFromConfig : absorptionFileOverride;
    const std::string treeName = trees.value("absorption", std::string("he3candidates"));
    const double originalCtau = params.value("original_ctao_absorption", 7.6);
    (void)parametersBasePath;
    auto radBins = ParametersBaseRadBins();
    auto ctBinsByRad = ParametersBaseCtBinsByRad();
    std::cout << "[DrawRadCtAbsorptionCorrection] Using hard-coded ParametersBase rad-ct binning\n";
    if (radBinsOverride.size() >= 2) {
        radBins = radBinsOverride;
        auto sharedCtBins = JsonVector(binning.value("ct_bins_single", GeneralHelper::Json::array()));
        if (sharedCtBins.size() < 2) {
            sharedCtBins = {1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 40};
        }
        ctBinsByRad.assign(radBins.size() - 1, sharedCtBins);
    }

    if (absorptionFile.empty() || radBins.size() < 2 || ctBinsByRad.size() != radBins.size() - 1) {
        std::cerr << "[DrawRadCtAbsorptionCorrection] Invalid config for rad-ct absorption plotting\n";
        return;
    }

    std::filesystem::create_directories(outputDir);
    const std::string filePrefix = outputPrefix.empty() ? std::string("") : outputPrefix + "_";

    auto chain = UnifiedAnalysis::MakeChainFromFileForCorrection(absorptionFile, treeName);
    ROOT::RDataFrame rdf(*chain);
    gRandom->SetSeed(0);
    Absorption::RadCtAbsorptionCalculator calc(&rdf, radBins, ctBinsByRad, originalCtau, "_plot");
    calc.Calculate();

    const auto &ratioMap = calc.HistRatio();
    const auto &countsMap = calc.HistCounts();
    const auto &survivedMap = calc.HistCountsAbsorb();
    auto ratioIt = ratioMap.find("both");
    auto countsIt = countsMap.find("both");
    auto survivedIt = survivedMap.find("both");
    if (ratioIt == ratioMap.end() || countsIt == countsMap.end() || survivedIt == survivedMap.end()) {
        std::cerr << "[DrawRadCtAbsorptionCorrection] Missing both histograms\n";
        return;
    }
    const auto &ratioHists = ratioIt->second;
    const auto &countsHists = countsIt->second;
    const auto &survivedHists = survivedIt->second;

    const auto globalCtEdges = BuildGlobalCtEdges(ctBinsByRad);
    auto hMap = std::make_unique<TH2D>("h_rad_ct_absorption_survival",
                                       ";decay c#tau (cm);R_{abs} = #sqrt{x_{abs}^{2}+y_{abs}^{2}} (cm);survival efficiency",
                                       static_cast<int>(globalCtEdges.size() - 1), globalCtEdges.data(),
                                       static_cast<int>(radBins.size() - 1), radBins.data());
    hMap->SetDirectory(nullptr);
    hMap->SetMinimum(0.0);
    hMap->SetMaximum(1.05);

    for (size_t ir = 0; ir < ratioHists.size(); ++ir) {
        const TH1F *h = ratioHists[ir];
        for (int lb = 1; lb <= h->GetNbinsX(); ++lb) {
            const double lo = h->GetXaxis()->GetBinLowEdge(lb);
            const double hi = h->GetXaxis()->GetBinUpEdge(lb);
            const double val = h->GetBinContent(lb);
            for (int gx = 1; gx <= hMap->GetNbinsX(); ++gx) {
                const double center = hMap->GetXaxis()->GetBinCenter(gx);
                if (center >= lo && center < hi) {
                    hMap->SetBinContent(gx, static_cast<int>(ir + 1), val);
                }
            }
        }
    }

    {
        TCanvas c("c_rad_ct_absorption_map", "c_rad_ct_absorption_map", 950, 760);
        c.SetTicks(1, 1);
        hMap->Draw("COLZ");
        DrawStamp(0.16, 0.93, "Absorption correction factor: numerator survived (absoCt > decay ct), denominator generated decay ct");
        c.SaveAs((outputDir + "/" + filePrefix + "rad_ct_absorption_correction_factor_map.pdf").c_str());
    }

    const std::vector<int> colors = {
        kRed + 1, kBlue + 1, kGreen + 2, kMagenta + 1, kOrange + 7,
        kCyan + 2, kViolet + 1, kPink + 1, kTeal + 3
    };

    {
        TCanvas c("c_rad_ct_absorption_curves", "c_rad_ct_absorption_curves", 950, 760);
        c.SetTicks(1, 1);
        c.SetGridx();
        c.SetGridy();
        TH1D frame("frame_rad_ct_absorption_curves", ";decay c#tau (cm);survival efficiency", 100, globalCtEdges.front(), globalCtEdges.back());
        frame.SetMinimum(0.0);
        frame.SetMaximum(1.08);
        frame.Draw("AXIS");
        TLegend leg(0.55, 0.18, 0.86, 0.46);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.SetTextFont(42);
        leg.SetTextSize(0.032);
        for (size_t ir = 0; ir < ratioHists.size(); ++ir) {
            TH1F *h = ratioHists[ir];
            h->SetLineColor(colors[ir % colors.size()]);
            h->SetMarkerColor(colors[ir % colors.size()]);
            h->SetMarkerStyle(20 + static_cast<int>(ir % 7));
            h->SetMarkerSize(0.95);
            h->SetLineWidth(2);
            h->Draw("E1 SAME");
            leg.AddEntry(h, Form("%.1f < R_{abs} < %.1f cm", radBins[ir], radBins[ir + 1]), "lep");
        }
        leg.Draw();
        DrawStamp(0.16, 0.93, "Absorption correction factor = survived / generated decay-ct trials");
        c.SaveAs((outputDir + "/" + filePrefix + "rad_ct_absorption_correction_factor_curves.pdf").c_str());
    }

    const std::filesystem::path perRadDir = std::filesystem::path(outputDir) / "rad_ct_absorption_efficiency_per_radbin";
    std::filesystem::create_directories(perRadDir);
    for (size_t ir = 0; ir < ratioHists.size(); ++ir) {
        TH1F *h = ratioHists[ir];
        TCanvas c(Form("c_rad_ct_absorption_efficiency_rad%zu", ir),
                  Form("c_rad_ct_absorption_efficiency_rad%zu", ir), 820, 680);
        c.SetTicks(1, 1);
        c.SetGridx();
        c.SetGridy();
        h->SetTitle(Form("%.1f < R_{abs} < %.1f cm;decay c#tau (cm);survival efficiency", radBins[ir], radBins[ir + 1]));
        h->SetLineColor(kBlue + 1);
        h->SetMarkerColor(kBlue + 1);
        h->SetMarkerStyle(20);
        h->SetMarkerSize(1.0);
        h->SetLineWidth(2);
        h->SetMinimum(0.0);
        h->SetMaximum(1.08);
        h->Draw("E1");
        DrawStamp(0.17, 0.92, "Absorption efficiency = survived / generated decay-ct trials", 0.033);
        const std::string fname = Form("%srad_ct_absorption_efficiency_rad_%.1f_%.1f.pdf", filePrefix.c_str(), radBins[ir], radBins[ir + 1]);
        c.SaveAs((perRadDir / fname).string().c_str());
    }

    auto hTau = std::make_unique<TH1D>("h_tau_vs_absorption_radius",
                                       ";R_{abs} = #sqrt{x_{abs}^{2}+y_{abs}^{2}} (cm);#tau from survived c#tau (ps)",
                                       static_cast<int>(radBins.size() - 1), radBins.data());
    hTau->SetDirectory(nullptr);
    hTau->SetMarkerStyle(20);
    hTau->SetMarkerSize(1.1);
    hTau->SetMarkerColor(kBlack);
    hTau->SetLineColor(kBlack);

    std::vector<std::unique_ptr<TH1D>> densityHists;
    std::vector<std::unique_ptr<TF1>> fits;
    densityHists.reserve(survivedHists.size());
    fits.reserve(survivedHists.size());

    {
        TCanvas c("c_survived_ct_rad_tau_fit", "c_survived_ct_rad_tau_fit", 980, 780);
        c.SetTicks(1, 1);
        c.SetLogy();
        TH1D frame("frame_survived_ct_rad", ";decay c#tau (cm);survived candidates / cm", 100, globalCtEdges.front(), globalCtEdges.back());
        frame.SetMinimum(1.0);
        frame.SetMaximum(1e8);
        frame.Draw("AXIS");

        TLegend leg(0.50, 0.46, 0.88, 0.86);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.SetTextFont(42);
        leg.SetTextSize(0.029);

        double ymax = 1.0;
        for (size_t ir = 0; ir < survivedHists.size(); ++ir) {
            densityHists.emplace_back(MakeDensityClone(survivedHists[ir], Form("h_survived_ct_density_rad%zu", ir)));
            auto *h = densityHists.back().get();
            h->SetLineColor(colors[ir % colors.size()]);
            h->SetMarkerColor(colors[ir % colors.size()]);
            h->SetMarkerStyle(20 + static_cast<int>(ir % 7));
            h->SetMarkerSize(0.9);
            h->SetLineWidth(2);
            ymax = std::max(ymax, h->GetMaximum());
        }
        frame.SetMaximum(std::max(10.0, ymax * 8.0));

        for (size_t ir = 0; ir < densityHists.size(); ++ir) {
            auto *h = densityHists[ir].get();
            const double xmin = h->GetXaxis()->GetXmin();
            const double xmax = h->GetXaxis()->GetXmax();
            fits.emplace_back(std::make_unique<TF1>(Form("f_survived_ct_rad%zu", ir), "[0]*exp(-x/[1])", xmin, xmax));
            auto *fit = fits.back().get();
            fit->SetParameters(std::max(1.0, h->GetMaximum()), originalCtau);
            fit->SetParLimits(1, 0.2, 200.0);
            fit->SetLineColor(colors[ir % colors.size()]);
            fit->SetLineWidth(2);
            fit->SetLineStyle(2);
            if (CountNonZeroBins(h) >= 3) {
                h->Fit(fit, "RQ0");
                const double tauCt = fit->GetParameter(1);
                const double tauCtErr = fit->GetParError(1);
                hTau->SetBinContent(static_cast<int>(ir + 1), tauCt / kCcmPerPs);
                hTau->SetBinError(static_cast<int>(ir + 1), tauCtErr / kCcmPerPs);
            }
            h->Draw("E1 SAME");
            if (CountNonZeroBins(h) >= 3) fit->Draw("SAME");
            leg.AddEntry(h, Form("%.1f < R_{abs} < %.1f cm, #tau = %.0f #pm %.0f ps",
                                 radBins[ir], radBins[ir + 1],
                                 hTau->GetBinContent(static_cast<int>(ir + 1)),
                                 hTau->GetBinError(static_cast<int>(ir + 1))), "lep");
        }
        leg.Draw();
        DrawStamp(0.16, 0.93, "Survived distribution: absoCt > generated decay c#tau");
        c.SaveAs((outputDir + "/" + filePrefix + "rad_ct_survived_ct_tau_fit.pdf").c_str());
    }

    {
        TCanvas c("c_tau_vs_absorption_radius", "c_tau_vs_absorption_radius", 900, 700);
        c.SetTicks(1, 1);
        c.SetGridy();
        hTau->SetMinimum(0.0);
        hTau->SetMaximum(std::max(320.0, hTau->GetMaximum() * 1.35));
        hTau->Draw("E1");
        DrawStamp(0.16, 0.93, "Tau from exponential fits to survived c#tau distributions");
        c.SaveAs((outputDir + "/" + filePrefix + "rad_ct_survived_tau_vs_rad.pdf").c_str());
    }

    TFile fout((outputDir + "/" + filePrefix + "rad_ct_absorption_correction.root").c_str(), "RECREATE");
    hMap->Write();
    hTau->Write();
    for (auto *h : ratioHists) h->Write();
    for (auto *h : countsHists) h->Write();
    for (auto *h : survivedHists) h->Write();
    for (const auto &h : densityHists) h->Write();
    fout.Close();

    std::cout << "[DrawRadCtAbsorptionCorrection] Output written to " << outputDir << std::endl;
}
