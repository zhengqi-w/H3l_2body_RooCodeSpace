// DrawPtLifetime.C
// Usage:
//   root -l -b -q 'DrawPtLifetime.C()'

#include <ROOT/RDataFrame.hxx>

#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TLorentzVector.h>
#include <TPaveText.h>
#include <TRandom3.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "../../include/include.h"
#include "../../Tools/AbsorptionHelper.h"
#include "../../Tools/GeneralHelper.hpp"

using namespace Absorption;
using namespace GeneralHelper;
using namespace Physics;

namespace {

constexpr double kTauPlotYMin = 210.0;
constexpr double kTauPlotYMax = 270.0;

struct LifetimeFit {
    double tauPs = 0.0;
    double tauErrPs = 0.0;
    double chi2 = 0.0;
    int ndf = 0;
    TF1* func = nullptr;
};

std::string Tag(double value)
{
    std::string s = Form("%g", value);
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

double ExtractMultiplierFromFilename(const std::string& fname)
{
    std::string base = fname;
    size_t p = base.find_last_of("/\\");
    if (p != std::string::npos) base = base.substr(p + 1);

    std::regex re(R"([xX]([+-]?[0-9]+(?:\.[0-9]+)?))");
    std::smatch m;
    std::string::const_iterator start = base.cbegin();
    double value = 1.0;
    bool found = false;
    while (std::regex_search(start, base.cend(), m, re)) {
        value = std::stod(m[1].str());
        found = true;
        start = m.suffix().first;
    }
    if (!found) {
        std::cerr << "Warning: cannot extract multiplier from " << fname << ", using 1.0\n";
    }
    return value;
}

std::string MacroDir()
{
    std::string macroPath = __FILE__;
    size_t pos = macroPath.find_last_of("/\\");
    return (pos == std::string::npos) ? "." : macroPath.substr(0, pos);
}

std::string ResolveInputPath(const std::string& path)
{
    if (path.empty() || gSystem->IsAbsoluteFileName(path.c_str()) || !gSystem->AccessPathName(path.c_str())) {
        return path;
    }
    const std::string candidate = MacroDir() + "/" + path;
    return gSystem->AccessPathName(candidate.c_str()) ? path : candidate;
}

std::string ResolveOutputPath(const std::string& path)
{
    if (path.empty() || gSystem->IsAbsoluteFileName(path.c_str())) {
        return path;
    }
    return MacroDir() + "/" + path;
}

void StyleHist(TH1* h, Color_t color, Style_t marker)
{
    h->SetStats(0);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(0.9);
    h->SetLineWidth(2);
    h->GetXaxis()->SetTitle("c#tau (cm)");
    h->GetYaxis()->SetTitle("Counts / cm");
    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleOffset(1.25);
}

std::unique_ptr<TH1> MakeDensityClone(const TH1* hist, const std::string& name)
{
    if (!hist) {
        return nullptr;
    }
    auto* clone = dynamic_cast<TH1*>(hist->Clone(name.c_str()));
    if (!clone) {
        return nullptr;
    }
    clone->SetDirectory(nullptr);
    for (int b = 1; b <= clone->GetNbinsX(); ++b) {
        const double width = clone->GetXaxis()->GetBinWidth(b);
        if (width <= 0.0) continue;
        clone->SetBinContent(b, clone->GetBinContent(b) / width);
        clone->SetBinError(b, clone->GetBinError(b) / width);
    }
    return std::unique_ptr<TH1>(clone);
}

LifetimeFit FitLifetime(TH1* hist, const std::string& name)
{
    LifetimeFit out;
    if (!hist || hist->Integral() <= 0.0) {
        return out;
    }

    auto density = MakeDensityClone(hist, name + "_density");
    if (!density) {
        return out;
    }

    const double xmin = density->GetXaxis()->GetXmin();
    const double xmax = density->GetXaxis()->GetXmax();
    auto* fit = new TF1(name.c_str(), "[0]*exp(-x/[1])", xmin, xmax);
    fit->SetParameters(density->GetMaximum(), 253.0 * c_cm_per_ps);
    fit->SetParLimits(0, 0.0, 10.0 * std::max(1.0, density->GetMaximum()));
    fit->SetParLimits(1, 1.0, 50.0);
    fit->SetLineWidth(2);
    fit->SetLineColor(kRed + 1);
    density->Fit(fit, "RQ0");

    out.func = fit;
    out.tauPs = fit->GetParameter(1) / c_cm_per_ps;
    out.tauErrPs = fit->GetParError(1) / c_cm_per_ps;
    out.chi2 = fit->GetChisquare();
    out.ndf = fit->GetNDF();
    return out;
}

void DrawDistributionCanvas(TH1* before,
                            TH1* after,
                            const LifetimeFit& fitBefore,
                            const LifetimeFit& fitAfter,
                            double ptMin,
                            double ptMax,
                            double sigma,
                            const std::string& outDir)
{
    auto beforeDensity = MakeDensityClone(before, Form("%s_density_draw", before->GetName()));
    auto afterDensity = MakeDensityClone(after, Form("%s_density_draw", after->GetName()));
    if (!beforeDensity || !afterDensity) {
        return;
    }

    auto canvas = std::make_unique<TCanvas>(Form("c_ct_pt_%s_%s_x%s", Tag(ptMin).c_str(), Tag(ptMax).c_str(), Tag(sigma).c_str()),
                                            "ct distribution", 900, 700);
    canvas->SetLeftMargin(0.12);
    canvas->SetRightMargin(0.04);
    canvas->SetTopMargin(0.07);
    canvas->SetBottomMargin(0.12);
    canvas->SetTicks();
    canvas->SetLogy();

    StyleHist(beforeDensity.get(), kAzure + 2, 20);
    StyleHist(afterDensity.get(), kOrange + 7, 24);
    beforeDensity->SetTitle(Form("%.1f #leq p_{T} < %.1f GeV/c, %.1f #times #sigma_{abso}", ptMin, ptMax, sigma));
    beforeDensity->Draw("E");
    afterDensity->Draw("E SAME");
    if (fitBefore.func) fitBefore.func->Draw("SAME");
    if (fitAfter.func) {
        fitAfter.func->SetLineColor(kMagenta + 2);
        fitAfter.func->Draw("SAME");
    }

    auto legend = std::make_unique<TLegend>(0.55, 0.68, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.035);
    legend->AddEntry(beforeDensity.get(), "Before absorption", "lep");
    legend->AddEntry(afterDensity.get(), "After absorption", "lep");
    if (fitBefore.func) legend->AddEntry(fitBefore.func, Form("Before fit: %.1f ps", fitBefore.tauPs), "l");
    if (fitAfter.func) legend->AddEntry(fitAfter.func, Form("After fit: %.1f ps", fitAfter.tauPs), "l");
    legend->Draw();

    auto info = std::make_unique<TPaveText>(0.16, 0.18, 0.48, 0.35, "NDC");
    info->SetBorderSize(0);
    info->SetFillStyle(0);
    info->SetTextAlign(12);
    info->SetTextFont(42);
    info->SetTextSize(0.032);
    info->AddText(Form("#tau_{input} = 253 ps"));
    info->AddText(Form("#chi^{2}/ndf before = %.1f/%d", fitBefore.chi2, fitBefore.ndf));
    info->AddText(Form("#chi^{2}/ndf after = %.1f/%d", fitAfter.chi2, fitAfter.ndf));
    info->Draw();

    canvas->SaveAs(Form("%s/ct_distribution_pt_%s_%s_x%s.pdf",
                        outDir.c_str(), Tag(ptMin).c_str(), Tag(ptMax).c_str(), Tag(sigma).c_str()));
}

void DrawTauGraph(TGraphErrors* graph, const std::string& title, const std::string& outPath)
{
    auto canvas = std::make_unique<TCanvas>(Form("c_%s", graph->GetName()), title.c_str(), 850, 650);
    canvas->SetLeftMargin(0.12);
    canvas->SetRightMargin(0.04);
    canvas->SetTopMargin(0.07);
    canvas->SetBottomMargin(0.12);
    canvas->SetGridx();
    canvas->SetGridy();
    canvas->SetTicks();

    graph->SetTitle("");
    graph->GetXaxis()->SetTitle("p_{T} (GeV/c)");
    graph->GetYaxis()->SetTitle("Fitted #tau (ps)");
    graph->GetYaxis()->SetRangeUser(kTauPlotYMin, kTauPlotYMax);
    graph->SetLineWidth(2);
    graph->SetMarkerSize(1.15);
    graph->Draw("AP");

    auto legend = std::make_unique<TLegend>(0.58, 0.78, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.04);
    legend->AddEntry(graph, title.c_str(), "lep");
    legend->Draw();

    canvas->SaveAs(outPath.c_str());
}

} // namespace

void DrawPtLifetime(const std::vector<std::string>& absorptionPaths = {
                        "../../../../AbsorptionTrees/absorption_tree_x1.root",
                        "../../../../AbsorptionTrees/absorption_tree_x2.root",
                        "../../../../AbsorptionTrees/absorption_tree_x3.root",
                        "../../../../AbsorptionTrees/absorption_tree_x4.root",
                        "../../../../AbsorptionTrees/absorption_tree_x5.root"},
                    const char* treeName = "he3candidates",
                    const char* outputDir = "../../../Outputs/CrossSection/Plotting")
{
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    const std::string resolvedOutputDir = ResolveOutputPath(outputDir ? std::string(outputDir) : std::string("../../../Outputs/CrossSection/Plotting"));
    EnsureDir(resolvedOutputDir);

    const std::vector<double> ptBins = {2.0, 3.0, 4.0, 5.5, 8.0};
    const std::vector<double> ctBins = [] {
        std::vector<double> bins;
        for (double x = 0.0; x < 10.0; x += 0.5) bins.push_back(x);
        for (double x = 10.0; x < 20.0; x += 1.0) bins.push_back(x);
        for (double x = 20.0; x <= 40.0; x += 2.0) bins.push_back(x);
        return bins;
    }();
    const double originalTauCt = 253.0 * c_cm_per_ps;

    std::vector<TGraphErrors*> graphs;
    const int colors[] = {kBlack, kAzure + 2, kOrange + 7, kGreen + 2, kMagenta + 2, kRed + 1};
    const int markers[] = {20, 21, 22, 23, 29, 33};

    for (size_t iFile = 0; iFile < absorptionPaths.size(); ++iFile) {
        const std::string path = ResolveInputPath(absorptionPaths[iFile]);
        std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
        if (!file || file->IsZombie()) {
            std::cerr << "Cannot open absorption file: " << path << "\n";
            continue;
        }
        auto* tree = dynamic_cast<TTree*>(file->Get(treeName));
        if (!tree) {
            std::cerr << "Cannot find tree " << treeName << " in " << path << "\n";
            continue;
        }

        const double sigma = ExtractMultiplierFromFilename(path);
        ROOT::RDataFrame rdf(*tree);
        std::vector<std::unique_ptr<TH1F>> hBefore;
        std::vector<std::unique_ptr<TH1F>> hAfter;
        hBefore.reserve(ptBins.size() - 1);
        hAfter.reserve(ptBins.size() - 1);
        for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
            hBefore.emplace_back(MakeTH1(Form("h_ct_before_pt%zu_x%s", ipt, Tag(sigma).c_str()),
                                         Form("Before absorption %.1f #leq p_{T} < %.1f;pseudoc#tau;Counts", ptBins[ipt], ptBins[ipt + 1]),
                                         ctBins));
            hAfter.emplace_back(MakeTH1(Form("h_ct_after_pt%zu_x%s", ipt, Tag(sigma).c_str()),
                                        Form("After absorption %.1f #leq p_{T} < %.1f;pseudoc#tau;Counts", ptBins[ipt], ptBins[ipt + 1]),
                                        ctBins));
            hBefore.back()->Sumw2();
            hAfter.back()->Sumw2();
            hBefore.back()->SetDirectory(nullptr);
            hAfter.back()->SetDirectory(nullptr);
        }

        TRandom3 rng(0);
        rdf.Foreach([&](float pt, float eta, float phi, float ax, float ay, float az) {
            TLorentzVector lv;
            lv.SetPtEtaPhiM(pt, eta, phi, HE3_MASS);
            const double he3p = lv.P();
            const double absoL = std::sqrt(ax * ax + ay * ay + az * az);
            const double absoCt = (he3p != 0.0) ? absoL * HE3_MASS / he3p : 1e9;
            const double decCt = -originalTauCt * std::log(std::max(1e-12, rng.Uniform()));
            for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
                if (pt < ptBins[ipt] || pt >= ptBins[ipt + 1]) continue;
                hBefore[ipt]->Fill(decCt);
                if (absoCt > decCt) {
                    hAfter[ipt]->Fill(decCt);
                }
                break;
            }
        }, {"pt", "eta", "phi", "absoX", "absoY", "absoZ"});

        auto* graph = new TGraphErrors(static_cast<int>(ptBins.size() - 1));
        graph->SetName(Form("gr_tau_vs_pt_x%s", Tag(sigma).c_str()));
        graph->SetTitle(Form("%.1f #times #sigma_{abso}", sigma));
        graph->SetMarkerStyle(markers[iFile % 6]);
        graph->SetMarkerColor(colors[iFile % 6]);
        graph->SetLineColor(colors[iFile % 6]);

        for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
            LifetimeFit fitBefore = FitLifetime(hBefore[ipt].get(), Form("fit_before_pt%zu_x%s", ipt, Tag(sigma).c_str()));
            LifetimeFit fitAfter = FitLifetime(hAfter[ipt].get(), Form("fit_after_pt%zu_x%s", ipt, Tag(sigma).c_str()));
            DrawDistributionCanvas(hBefore[ipt].get(), hAfter[ipt].get(), fitBefore, fitAfter,
                                   ptBins[ipt], ptBins[ipt + 1], sigma, resolvedOutputDir);

            const double ptCenter = 0.5 * (ptBins[ipt] + ptBins[ipt + 1]);
            const double ptHalfWidth = 0.5 * (ptBins[ipt + 1] - ptBins[ipt]);
            graph->SetPoint(static_cast<int>(ipt), ptCenter, fitAfter.tauPs);
            graph->SetPointError(static_cast<int>(ipt), ptHalfWidth, fitAfter.tauErrPs);
        }

        DrawTauGraph(graph,
                     Form("%.1f #times #sigma_{abso}", sigma),
                     Form("%s/tau_vs_pt_x%s.pdf", resolvedOutputDir.c_str(), Tag(sigma).c_str()));
        graphs.push_back(graph);
    }

    auto canvas = std::make_unique<TCanvas>("c_tau_vs_pt_overlay", "tau vs pt overlay", 900, 700);
    canvas->SetLeftMargin(0.12);
    canvas->SetRightMargin(0.04);
    canvas->SetTopMargin(0.07);
    canvas->SetBottomMargin(0.12);
    canvas->SetGridx();
    canvas->SetGridy();
    canvas->SetTicks();

    auto frame = std::make_unique<TH1F>("h_tau_overlay_frame", ";p_{T} (GeV/c);Fitted #tau (ps)", 1, 2.0, 8.0);
    frame->SetMinimum(kTauPlotYMin);
    frame->SetMaximum(kTauPlotYMax);
    frame->SetStats(0);
    frame->Draw();

    auto legend = std::make_unique<TLegend>(0.58, 0.62, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.036);
    for (auto* graph : graphs) {
        graph->Draw("P SAME");
        legend->AddEntry(graph, graph->GetTitle()[0] ? graph->GetTitle() : graph->GetName(), "lep");
    }
    legend->Draw();
    canvas->SaveAs(Form("%s/tau_vs_pt_overlay.pdf", resolvedOutputDir.c_str()));

    TFile outFile(Form("%s/pt_lifetime_results.root", resolvedOutputDir.c_str()), "RECREATE");
    for (auto* graph : graphs) graph->Write();
    canvas->Write();
    outFile.Close();
}
