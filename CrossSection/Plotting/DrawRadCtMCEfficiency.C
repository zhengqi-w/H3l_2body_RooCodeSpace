#include <ROOT/RDataFrame.hxx>
#include <TBox.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TColor.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../Tools/AcceptanceHelper.h"
#include "../../Tools/GeneralHelper.hpp"

namespace {

std::string MacroDir()
{
    std::filesystem::path p(__FILE__);
    if (p.is_relative()) {
        p = std::filesystem::current_path() / p;
    }
    return std::filesystem::weakly_canonical(p).parent_path().string();
}

std::string ResolvePath(const std::string &path)
{
    if (path.empty()) {
        return path;
    }
    std::filesystem::path p(path);
    if (p.is_absolute()) {
        return p.string();
    }

    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::current_path() / p,
        std::filesystem::path(MacroDir()) / p
    };
    for (const auto &candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::weakly_canonical(candidate).string();
        }
    }
    return (std::filesystem::path(MacroDir()) / p).string();
}

std::string JoinEdges(const std::vector<double> &edges)
{
    std::ostringstream os;
    os << std::setprecision(4);
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i) {
            os << ", ";
        }
        os << edges[i];
    }
    return os.str();
}

std::string BinLabel(double lo, double hi, const char *unit = "")
{
    return Form("%.1f-%.1f%s", lo, hi, unit);
}

TGraphAsymmErrors *MakeGraphWithBinWidth(const TH1D *h, const std::string &name,
                                         Color_t color, Style_t marker)
{
    if (!h) {
        return nullptr;
    }
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
    g->SetMarkerSize(1.15);
    g->SetLineWidth(3);
    return g;
}

double MaxHistValue(const std::vector<TH1D *> &hists)
{
    double ymax = 0.;
    for (const auto *h : hists) {
        if (!h) {
            continue;
        }
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            ymax = std::max(ymax, h->GetBinContent(ib) + h->GetBinError(ib));
        }
    }
    return ymax;
}

void StyleFrame(TH1 *frame)
{
    frame->SetStats(false);
    frame->GetXaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.045);
    frame->GetXaxis()->SetLabelSize(0.038);
    frame->GetYaxis()->SetLabelSize(0.038);
    frame->GetYaxis()->SetTitleOffset(1.15);
}

void SaveCurves(const std::vector<double> &ptBins,
                const std::vector<TH1D *> &accHists,
                const std::string &outPdf,
                const std::string &sampleLabel,
                const std::string &basicSelection)
{
    const std::vector<Color_t> colors = {
        kRed + 1, kOrange + 7, kGreen + 2, kAzure + 1, kBlue + 1, kViolet + 1
    };
    const std::vector<Style_t> markers = {
        kFullCircle, kFullSquare, kFullTriangleUp, kFullDiamond, kOpenCircle, kOpenSquare
    };

    const double yMax = std::max(0.08, MaxHistValue(accHists) * 1.35);
    TCanvas c("c_rad_ct_eff_curves", "", 1050, 780);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);

    TH1D frame("h_frame_rad_ct_eff", ";#it{c}t (cm);MC efficiency", 100, 0., 25.);
    StyleFrame(&frame);
    frame.SetMinimum(0.);
    frame.SetMaximum(yMax);
    frame.Draw("AXIS");

    TLegend leg(0.58, 0.55, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.031);

    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    graphs.reserve(accHists.size());
    for (size_t ip = 0; ip < accHists.size(); ++ip) {
        if (!accHists[ip]) {
            continue;
        }
        auto g = std::unique_ptr<TGraphAsymmErrors>(
            MakeGraphWithBinWidth(accHists[ip], Form("g_acc_pt_%zu", ip),
                                  colors[ip % colors.size()], markers[ip % markers.size()]));
        g->Draw("PZ SAME");
        leg.AddEntry(g.get(), Form("%s GeV/#it{c}", BinLabel(ptBins[ip], ptBins[ip + 1]).c_str()), "pe");
        graphs.push_back(std::move(g));
    }

    leg.Draw();
    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, sampleLabel.c_str());
    text.DrawLatex(0.16, 0.78, Form("Basic cut: %s", basicSelection.c_str()));
    text.DrawLatex(0.16, 0.73, "Config pt-ct binning");
    c.SaveAs(outPdf.c_str());
}

void SaveMap(const std::vector<double> &ptBins,
             const std::vector<TH1D *> &accHists,
             const std::string &outPdf,
             const std::string &sampleLabel,
             const std::string &basicSelection)
{
    double ctMin = 1e9;
    double ctMax = -1e9;
    double zMax = 0.;
    for (const auto *h : accHists) {
        if (!h) {
            continue;
        }
        ctMin = std::min(ctMin, h->GetXaxis()->GetBinLowEdge(1));
        ctMax = std::max(ctMax, h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()));
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            zMax = std::max(zMax, h->GetBinContent(ib));
        }
    }
    if (ctMax <= ctMin) {
        ctMin = 0.;
        ctMax = 25.;
    }
    zMax = std::max(zMax, 1e-6);

    TCanvas c("c_rad_ct_eff_map", "", 1120, 790);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.14);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    gStyle->SetPalette(kBird);

    TH2D frame("h_frame_rad_ct_eff_map", ";#it{c}t (cm);#it{p}_{T} (GeV/#it{c})",
               100, ctMin, ctMax, 100, ptBins.front(), ptBins.back());
    StyleFrame(&frame);
    frame.SetStats(false);
    frame.Draw("AXIS");

    TLatex binText;
    binText.SetTextAlign(22);
    binText.SetTextSize(0.026);

    std::vector<std::unique_ptr<TBox>> boxes;
    for (size_t ip = 0; ip < accHists.size(); ++ip) {
        const auto *h = accHists[ip];
        if (!h) {
            continue;
        }
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x1 = h->GetXaxis()->GetBinLowEdge(ib);
            const double x2 = h->GetXaxis()->GetBinUpEdge(ib);
            const double y1 = ptBins[ip];
            const double y2 = ptBins[ip + 1];
            const double z = h->GetBinContent(ib);
            const int palIdx = std::clamp(static_cast<int>(std::round(254. * z / zMax)), 0, 254);
            const int color = TColor::GetPalette()[palIdx];
            auto box = std::make_unique<TBox>(x1, y1, x2, y2);
            box->SetFillColor(color);
            box->SetLineColor(kGray + 2);
            box->SetLineWidth(2);
            box->Draw("same");
            boxes.push_back(std::move(box));
            binText.SetTextColor(z / zMax > 0.55 ? kWhite : kBlack);
            binText.DrawLatex(0.5 * (x1 + x2), 0.5 * (y1 + y2), Form("%.3f", z));
        }
    }
    frame.Draw("AXIS SAME");

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.034);
    text.DrawLatex(0.15, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.030);
    text.DrawLatex(0.15, 0.83, sampleLabel.c_str());
    text.DrawLatex(0.15, 0.78, Form("Basic cut: %s", basicSelection.c_str()));
    text.DrawLatex(0.15, 0.73, Form("Max efficiency = %.4f", zMax));

    c.SaveAs(outPdf.c_str());
}

void WriteOutputRoot(const std::string &outRoot,
                     const std::vector<double> &ptBins,
                     const AcceptanceHelper::AcceptanceResult &res)
{
    TFile fout(outRoot.c_str(), "RECREATE");
    for (size_t ip = 0; ip < res.acc_ct_per_pt.size(); ++ip) {
        auto *acc = res.acc_ct_per_pt[ip];
        auto *reco = ip < res.reco_ct_per_pt.size() ? res.reco_ct_per_pt[ip] : nullptr;
        auto *evsel = ip < res.evsel_ct_per_pt.size() ? res.evsel_ct_per_pt[ip] : nullptr;
        if (acc) {
            acc->SetName(Form("h_mc_eff_pt_%.1f_%.1f", ptBins[ip], ptBins[ip + 1]));
            acc->Write();
        }
        if (reco) {
            reco->SetName(Form("h_reco_pt_%.1f_%.1f", ptBins[ip], ptBins[ip + 1]));
            reco->Write();
        }
        if (evsel) {
            evsel->SetName(Form("h_evsel_pt_%.1f_%.1f", ptBins[ip], ptBins[ip + 1]));
            evsel->Write();
        }
    }
    fout.Close();
}

} // namespace

void DrawRadCtMCEfficiency(
    const char *configPath = "../../configs/crosssection_config.json",
    const char *mcPath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtMCEfficiency")
{
    const std::string cfgPath = ResolvePath(configPath ? configPath : "");
    const std::string mcFilePath = ResolvePath(mcPath ? mcPath : "");
    const std::string outDir = ResolvePath(outputDir ? outputDir : "RadCtMCEfficiency");
    const std::string basicSelection = "fDecRad > 0.8";

    const auto cfg = GeneralHelper::LoadJsonFile(cfgPath);
    const auto ptBins = cfg.at("ptbins").get<std::vector<double>>();
    const auto ctBinsPerPt = cfg.at("ctbins").get<std::vector<std::vector<double>>>();
    if (ptBins.size() < 2 || ctBinsPerPt.size() != ptBins.size() - 1) {
        throw std::runtime_error("DrawRadCtMCEfficiency: config ptbins/ctbins size mismatch");
    }

    std::filesystem::create_directories(outDir);

    ROOT::DisableImplicitMT();
    auto f = std::unique_ptr<TFile>(TFile::Open(mcFilePath.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        throw std::runtime_error("Cannot open MC file: " + mcFilePath);
    }
    TChain chain("O2mchypcands");
    GeneralHelper::fillChainFromAO2D(chain, f.get());
    if (chain.GetNtrees() <= 0) {
        throw std::runtime_error("No O2mchypcands trees found in MC file: " + mcFilePath);
    }

    std::cout << "[DrawRadCtMCEfficiency] config: " << cfgPath << "\n";
    std::cout << "[DrawRadCtMCEfficiency] MC: " << mcFilePath << "\n";
    std::cout << "[DrawRadCtMCEfficiency] ptbins: " << JoinEdges(ptBins) << "\n";
    for (size_t ip = 0; ip < ctBinsPerPt.size(); ++ip) {
        std::cout << "  pt " << ptBins[ip] << "-" << ptBins[ip + 1]
                  << " ct: " << JoinEdges(ctBinsPerPt[ip]) << "\n";
    }

    ROOT::RDataFrame rdf(chain);
    auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    auto res = AcceptanceHelper::ComputeAcceptanceFlexible(ready,
                                                           ptBins,
                                                           {},
                                                           ctBinsPerPt,
                                                           {},
                                                           {},
                                                           basicSelection,
                                                           {},
                                                           {},
                                                           true);

    const std::string sampleLabel = "LHC23 pass5, LHC25g11_G4list reweighted MC";
    SaveCurves(ptBins, res.acc_ct_per_pt,
               outDir + "/rad_ct_mc_efficiency_curves.pdf",
               sampleLabel, basicSelection);
    SaveMap(ptBins, res.acc_ct_per_pt,
            outDir + "/rad_ct_mc_efficiency_map.pdf",
            sampleLabel, basicSelection);
    WriteOutputRoot(outDir + "/rad_ct_mc_efficiency.root", ptBins, res);

    res.Clear();
    std::cout << "[DrawRadCtMCEfficiency] wrote outputs to " << outDir << "\n";
}
