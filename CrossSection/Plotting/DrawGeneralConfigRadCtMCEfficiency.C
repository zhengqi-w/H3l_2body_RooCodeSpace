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

struct Sample {
    std::string tag;
    std::string mcPath;
    std::string analysisResultsPath;
    Color_t color{kBlack};
    Style_t marker{kFullCircle};
};

std::string MacroDir()
{
    std::filesystem::path p(__FILE__);
    if (p.is_relative()) p = std::filesystem::current_path() / p;
    return std::filesystem::weakly_canonical(p).parent_path().string();
}

std::string ResolvePath(const std::string &path)
{
    if (path.empty()) return path;
    std::filesystem::path p(path);
    if (p.is_absolute()) return p.string();
    const auto fromCwd = std::filesystem::current_path() / p;
    if (std::filesystem::exists(fromCwd)) return std::filesystem::weakly_canonical(fromCwd).string();
    const auto fromMacro = std::filesystem::path(MacroDir()) / p;
    if (std::filesystem::exists(fromMacro)) return std::filesystem::weakly_canonical(fromMacro).string();
    return fromMacro.string();
}

std::string EdgesText(const std::vector<double> &edges)
{
    std::ostringstream os;
    os << std::setprecision(4);
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i) os << ", ";
        os << edges[i];
    }
    return os.str();
}

std::vector<double> NormalizedWeights(size_t n)
{
    return std::vector<double>(n, n ? 1.0 / static_cast<double>(n) : 0.0);
}

std::vector<double> EventWeights(const std::vector<Sample> &samples,
                                 const std::string &histPath,
                                 double centralityMin,
                                 double centralityMax)
{
    std::vector<double> weights(samples.size(), 0.0);
    double total = 0.0;
    for (size_t is = 0; is < samples.size(); ++is) {
        if (samples[is].analysisResultsPath.empty() || histPath.empty()) continue;
        TFile input(samples[is].analysisResultsPath.c_str(), "READ");
        if (input.IsZombie()) continue;
        auto *hist = dynamic_cast<TH1 *>(input.Get(histPath.c_str()));
        if (!hist) continue;
        const int firstBin = hist->GetXaxis()->FindBin(centralityMin + 1e-3);
        const int lastBin = hist->GetXaxis()->FindBin(centralityMax - 1e-3);
        weights[is] = hist->Integral(firstBin, lastBin);
        total += weights[is];
    }
    if (total <= 0.0) return NormalizedWeights(samples.size());
    for (auto &weight : weights) weight /= total;
    return weights;
}

std::vector<TH1D *> PickBoth(AcceptanceHelper::AcceptanceResult &res)
{
    return res.acc_ct_per_pt;
}

void StyleHist(TH1D *h, Color_t color, Style_t marker)
{
    if (!h) return;
    h->SetDirectory(nullptr);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.0);
    h->SetLineWidth(3);
    h->SetStats(false);
}

TGraphAsymmErrors *MakeGraph(const TH1D *h, const std::string &name, Color_t color, Style_t marker)
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
    g->SetLineWidth(3);
    return g;
}

double MaxWithError(const std::vector<TH1D *> &hists)
{
    double out = 0.;
    for (const auto *h : hists) {
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            out = std::max(out, h->GetBinContent(ib) + h->GetBinError(ib));
        }
    }
    return out;
}

std::string FileEdge(double value)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << value;
    std::string out = os.str();
    while (!out.empty() && out.back() == '0') out.pop_back();
    if (!out.empty() && out.back() == '.') out.pop_back();
    std::replace(out.begin(), out.end(), '.', '_');
    return out;
}

std::vector<std::unique_ptr<TH1D>> CloneHistVector(const std::vector<TH1D *> &src,
                                                   const std::string &prefix,
                                                   const std::vector<double> &radBins)
{
    std::vector<std::unique_ptr<TH1D>> out;
    out.reserve(src.size());
    for (size_t ir = 0; ir < src.size(); ++ir) {
        if (!src[ir]) {
            out.emplace_back(nullptr);
            continue;
        }
        auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(src[ir]->Clone(
            Form("%s_rad_%.2f_%.2f", prefix.c_str(), radBins[ir], radBins[ir + 1]))));
        h->SetDirectory(nullptr);
        out.push_back(std::move(h));
    }
    return out;
}

std::vector<std::unique_ptr<TH1D>> MakeWeightedAverage(
    const std::vector<std::vector<std::unique_ptr<TH1D>>> &periodHists,
    const std::vector<double> &weights,
    const std::vector<double> &radBins)
{
    std::vector<std::unique_ptr<TH1D>> out;
    if (periodHists.empty()) return out;
    out.reserve(periodHists.front().size());
    for (size_t ir = 0; ir < periodHists.front().size(); ++ir) {
        const TH1D *tmpl = nullptr;
        for (const auto &vec : periodHists) {
            if (ir < vec.size() && vec[ir]) {
                tmpl = vec[ir].get();
                break;
            }
        }
        if (!tmpl) {
            out.emplace_back(nullptr);
            continue;
        }
        auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(tmpl->Clone(
            Form("h_merged_rad_%.2f_%.2f", radBins[ir], radBins[ir + 1]))));
        h->Reset("ICES");
        h->SetDirectory(nullptr);
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            double value = 0.;
            double err2 = 0.;
            for (size_t is = 0; is < periodHists.size(); ++is) {
                if (ir >= periodHists[is].size() || !periodHists[is][ir]) continue;
                const double w = is < weights.size() ? weights[is] : 0.;
                value += w * periodHists[is][ir]->GetBinContent(ib);
                err2 += w * w * std::pow(periodHists[is][ir]->GetBinError(ib), 2);
            }
            h->SetBinContent(ib, value);
            h->SetBinError(ib, std::sqrt(err2));
        }
        out.push_back(std::move(h));
    }
    return out;
}

void DrawCurves(const std::vector<double> &radBins,
                const std::vector<TH1D *> &hists,
                const std::string &title,
                const std::string &note,
                const std::string &outPdf)
{
    const std::vector<Color_t> colors = {
        kRed + 1, kOrange + 7, kSpring + 5, kGreen + 2, kTeal + 3, kCyan + 2,
        kAzure + 1, kBlue + 1, kViolet + 1, kMagenta + 1, kPink + 7, kGray + 2
    };
    const std::vector<Style_t> markers = {
        kFullCircle, kFullSquare, kFullTriangleUp, kFullDiamond, kOpenCircle, kOpenSquare,
        kOpenTriangleUp, kOpenDiamond, kFullCross, kFullStar, kOpenCross, kOpenStar
    };

    TCanvas c("c_rad_ct_eff_curves", "", 1150, 820);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.28);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);

    TH1D frame("h_frame_rad_ct_eff_curves", ";#it{c}t_{gen} (cm);MC efficiency", 100, 0., 42.);
    frame.SetStats(false);
    frame.SetMinimum(0.);
    frame.SetMaximum(std::max(0.05, MaxWithError(hists) * 1.35));
    frame.GetXaxis()->SetTitleSize(0.045);
    frame.GetYaxis()->SetTitleSize(0.045);
    frame.GetYaxis()->SetTitleOffset(1.15);
    frame.Draw("AXIS");

    TLegend leg(0.74, 0.16, 0.97, 0.91);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.022);

    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    for (size_t ir = 0; ir < hists.size(); ++ir) {
        if (!hists[ir]) continue;
        auto g = std::unique_ptr<TGraphAsymmErrors>(MakeGraph(
            hists[ir], Form("g_rad_%zu", ir), colors[ir % colors.size()], markers[ir % markers.size()]));
        g->Draw("PZ SAME");
        leg.AddEntry(g.get(), Form("%.1f < #it{R}_{dec} < %.1f cm", radBins[ir], radBins[ir + 1]), "pe");
        graphs.push_back(std::move(g));
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.034);
    text.DrawLatex(0.15, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.030);
    text.DrawLatex(0.15, 0.83, title.c_str());
    text.DrawLatex(0.15, 0.78, note.c_str());
    c.SaveAs(outPdf.c_str());
}

void DrawPeriodComparisonForRad(
    size_t ir,
    const std::vector<double> &radBins,
    const std::vector<Sample> &samples,
    const std::vector<std::vector<std::unique_ptr<TH1D>>> &periodHists,
    const std::vector<std::unique_ptr<TH1D>> &mergedHists,
    const std::vector<std::unique_ptr<TH1D>> &equalWeightHists,
    const std::string &note,
    const std::string &outPdf,
    const std::string &multipagePdf)
{
    std::vector<TH1D *> hists;
    for (const auto &period : periodHists) {
        if (ir < period.size() && period[ir]) hists.push_back(period[ir].get());
    }
    if (ir < mergedHists.size() && mergedHists[ir]) hists.push_back(mergedHists[ir].get());
    if (ir < equalWeightHists.size() && equalWeightHists[ir]) hists.push_back(equalWeightHists[ir].get());
    if (hists.empty()) return;

    double ctMin = hists.front()->GetXaxis()->GetBinLowEdge(1);
    double ctMax = hists.front()->GetXaxis()->GetBinUpEdge(hists.front()->GetNbinsX());
    for (const auto *h : hists) {
        ctMin = std::min(ctMin, h->GetXaxis()->GetBinLowEdge(1));
        ctMax = std::max(ctMax, h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()));
    }

    TCanvas c(Form("c_period_comparison_rad_%zu", ir), "", 920, 720);
    c.SetLeftMargin(0.13);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);

    TH1D frame(Form("h_frame_period_comparison_rad_%zu", ir),
               ";#it{c}t_{gen} (cm);MC efficiency", 100, ctMin, ctMax);
    frame.SetStats(false);
    frame.SetMinimum(0.0);
    frame.SetMaximum(std::max(0.05, MaxWithError(hists) * 1.35));
    frame.GetXaxis()->SetTitleSize(0.047);
    frame.GetYaxis()->SetTitleSize(0.047);
    frame.GetXaxis()->SetLabelSize(0.040);
    frame.GetYaxis()->SetLabelSize(0.040);
    frame.GetYaxis()->SetTitleOffset(1.18);
    frame.Draw("AXIS");

    TLegend leg(0.57, 0.66, 0.94, 0.90);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.031);
    leg.SetNColumns(1);

    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    for (size_t is = 0; is < samples.size(); ++is) {
        if (is >= periodHists.size() || ir >= periodHists[is].size() || !periodHists[is][ir]) continue;
        auto graph = std::unique_ptr<TGraphAsymmErrors>(MakeGraph(
            periodHists[is][ir].get(),
            Form("g_period_%zu_rad_%zu", is, ir),
            samples[is].color,
            samples[is].marker));
        graph->Draw("PZ SAME");
        leg.AddEntry(graph.get(), samples[is].tag.c_str(), "pe");
        graphs.push_back(std::move(graph));
    }

    if (ir < mergedHists.size() && mergedHists[ir]) {
        auto graph = std::unique_ptr<TGraphAsymmErrors>(MakeGraph(
            mergedHists[ir].get(),
            Form("g_merged_rad_%zu", ir),
            kBlack,
            kOpenCircle));
        graph->SetMarkerSize(1.15);
        graph->SetLineWidth(3);
        graph->Draw("PZ SAME");
        leg.AddEntry(graph.get(), "Merged event weighted", "pe");
        graphs.push_back(std::move(graph));
    }
    if (ir < equalWeightHists.size() && equalWeightHists[ir]) {
        auto graph = std::unique_ptr<TGraphAsymmErrors>(MakeGraph(
            equalWeightHists[ir].get(),
            Form("g_equal_weight_rad_%zu", ir),
            kGray + 2,
            kOpenSquare));
        graph->SetMarkerSize(1.05);
        graph->SetLineStyle(2);
        graph->SetLineWidth(2);
        graph->Draw("PZ SAME");
        leg.AddEntry(graph.get(), "Merged equal weight", "pe");
        graphs.push_back(std::move(graph));
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.036);
    text.DrawLatex(0.17, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.032);
    text.DrawLatex(0.17, 0.82,
                   Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.SetTextSize(0.028);
    text.DrawLatex(0.17, 0.77, note.c_str());

    if (!outPdf.empty()) c.SaveAs(outPdf.c_str());
    if (!multipagePdf.empty()) c.SaveAs(multipagePdf.c_str());
}

void DrawMap(const std::vector<double> &radBins,
             const std::vector<TH1D *> &hists,
             const std::string &title,
             const std::string &note,
             const std::string &outPdf)
{
    double ctMin = 1e9;
    double ctMax = -1e9;
    double zMax = 0.;
    for (const auto *h : hists) {
        if (!h) continue;
        ctMin = std::min(ctMin, h->GetXaxis()->GetBinLowEdge(1));
        ctMax = std::max(ctMax, h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()));
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            zMax = std::max(zMax, h->GetBinContent(ib));
        }
    }
    if (ctMax <= ctMin) {
        ctMin = 0.;
        ctMax = 42.;
    }
    zMax = std::max(zMax, 1e-8);

    TCanvas c("c_rad_ct_eff_map", "", 1200, 850);
    c.SetLeftMargin(0.11);
    c.SetRightMargin(0.05);
    c.SetBottomMargin(0.11);
    c.SetTopMargin(0.06);
    gStyle->SetPalette(kBird);

    TH2D frame("h_frame_rad_ct_eff_map", ";#it{c}t_{gen} (cm);#it{R}_{dec}^{gen} (cm)",
               100, ctMin, ctMax, 100, radBins.front(), radBins.back());
    frame.SetStats(false);
    frame.GetXaxis()->SetTitleSize(0.043);
    frame.GetYaxis()->SetTitleSize(0.043);
    frame.GetYaxis()->SetTitleOffset(1.08);
    frame.Draw("AXIS");

    std::vector<std::unique_ptr<TBox>> boxes;
    TLatex valueText;
    valueText.SetTextAlign(22);
    valueText.SetTextSize(0.014);
    for (size_t ir = 0; ir < hists.size(); ++ir) {
        const auto *h = hists[ir];
        if (!h) continue;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x1 = h->GetXaxis()->GetBinLowEdge(ib);
            const double x2 = h->GetXaxis()->GetBinUpEdge(ib);
            const double y1 = radBins[ir];
            const double y2 = radBins[ir + 1];
            const double z = h->GetBinContent(ib);
            const int palIdx = std::clamp(static_cast<int>(std::round(254. * z / zMax)), 0, 254);
            auto box = std::make_unique<TBox>(x1, y1, x2, y2);
            box->SetFillColor(TColor::GetPalette()[palIdx]);
            box->SetLineColor(kGray + 2);
            box->SetLineWidth(1);
            box->Draw("same");
            boxes.push_back(std::move(box));
            valueText.SetTextColor((z / zMax) > 0.55 ? kWhite : kBlack);
            valueText.DrawLatex(0.5 * (x1 + x2), 0.5 * (y1 + y2), Form("%.3f", z));
        }
    }
    frame.Draw("AXIS SAME");

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.034);
    text.DrawLatex(0.14, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.030);
    text.DrawLatex(0.14, 0.83, title.c_str());
    text.DrawLatex(0.14, 0.78, note.c_str());
    text.DrawLatex(0.14, 0.73, Form("Max efficiency = %.4f", zMax));
    c.SaveAs(outPdf.c_str());
}

void WriteRoot(const std::string &outRoot,
               const std::vector<Sample> &samples,
               const std::vector<double> &radBins,
               const std::vector<std::vector<std::unique_ptr<TH1D>>> &periodHists,
               const std::vector<std::unique_ptr<TH1D>> &mergedHists,
               const std::vector<std::unique_ptr<TH1D>> &equalWeightHists)
{
    TFile out(outRoot.c_str(), "RECREATE");
    for (size_t is = 0; is < periodHists.size(); ++is) {
        out.mkdir(samples[is].tag.c_str());
        out.cd(samples[is].tag.c_str());
        for (size_t ir = 0; ir < periodHists[is].size(); ++ir) {
            if (!periodHists[is][ir]) continue;
            auto *h = static_cast<TH1D *>(periodHists[is][ir]->Clone(
                Form("h_eff_rad_%.2f_%.2f", radBins[ir], radBins[ir + 1])));
            h->Write();
            delete h;
        }
        out.cd();
    }
    out.mkdir("merged_event_weighted");
    out.cd("merged_event_weighted");
    for (size_t ir = 0; ir < mergedHists.size(); ++ir) {
        if (!mergedHists[ir]) continue;
        auto *h = static_cast<TH1D *>(mergedHists[ir]->Clone(
            Form("h_eff_rad_%.2f_%.2f", radBins[ir], radBins[ir + 1])));
        h->Write();
        delete h;
    }
    out.cd();
    out.mkdir("merged_equal_weight");
    out.cd("merged_equal_weight");
    for (size_t ir = 0; ir < equalWeightHists.size(); ++ir) {
        if (!equalWeightHists[ir]) continue;
        auto *h = static_cast<TH1D *>(equalWeightHists[ir]->Clone(
            Form("h_eff_rad_%.2f_%.2f", radBins[ir], radBins[ir + 1])));
        h->Write();
        delete h;
    }
    out.Close();
}

} // namespace

void DrawGeneralConfigRadCtMCEfficiency(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config_merged.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/GeneralConfigRadCtMCEfficiency")
{
    const std::string cfgPath = ResolvePath(configPath ? configPath : "");
    const std::string outDir = ResolvePath(outputDir ? outputDir : "");
    std::filesystem::create_directories(outDir);

    const auto cfg = GeneralHelper::LoadJsonFile(cfgPath);
    const auto common = cfg.at("common");
    const auto binning = common.at("binning");
    const auto path = common.at("path");
    const auto eventHist = common.value("event_hist", GeneralHelper::Json::object());
    const std::string mcTree = common.value("tree_names", GeneralHelper::Json::object()).value("mc", std::string("O2mchypcands"));
    const std::string eventHistPath = eventHist.value("n_events_hist", std::string());
    const std::string basicSelection = common.value("selection", GeneralHelper::Json::object())
                                           .value("basic_selection_data", std::string("fDecRad > 0.8"));
    const bool requireTwoBody = common.value("selection", GeneralHelper::Json::object())
                                    .value("mc_acceptance_require_two_body", true);
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto modeProfiles = analysis.value("mode_profiles", GeneralHelper::Json::object());
    const auto radCtProfile = modeProfiles.value("rad_ct", GeneralHelper::Json::object());
    const bool constrainDenominatorOuterBin =
        radCtProfile.value("mc_acceptance_outer_bin_cut_denominator",
                           common.value("selection", GeneralHelper::Json::object())
                               .value("mc_acceptance_outer_bin_cut_denominator", true));

    const auto radBins = binning.at("rad_bins").get<std::vector<double>>();
    const auto ctBinsByRad = binning.at("ct_bins_by_rad").get<std::vector<std::vector<double>>>();
    if (radBins.size() < 2 || ctBinsByRad.size() != radBins.size() - 1) {
        throw std::runtime_error("rad_bins / ct_bins_by_rad size mismatch in config");
    }

    std::vector<Sample> samples;
    const std::vector<Color_t> colors = {kRed + 1, kAzure + 1, kGreen + 2, kMagenta + 1};
    const std::vector<Style_t> markers = {kFullCircle, kFullSquare, kFullTriangleUp, kFullDiamond};
    if (common.contains("periods") && common["periods"].is_array()) {
        size_t i = 0;
        for (const auto &period : common["periods"]) {
            samples.push_back({
                period.value("tag", Form("period_%zu", i)),
                ResolvePath(period.value("mc_path", std::string())),
                ResolvePath(period.value("analysisresults_path", std::string())),
                colors[i % colors.size()],
                markers[i % markers.size()]
            });
            ++i;
        }
    }
    if (samples.empty()) {
        samples.push_back({
            "config_mc",
            ResolvePath(path.value("mc_path", std::string())),
            ResolvePath(path.value("analysisresults_path", std::string())),
            kBlack,
            kFullCircle
        });
    }

    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] config: " << cfgPath << "\n";
    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] tree: " << mcTree << "\n";
    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] selection: " << basicSelection
              << ", requireTwoBody=" << requireTwoBody
              << ", denominatorOuterBinCut=" << constrainDenominatorOuterBin << "\n";
    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] rad bins: " << EdgesText(radBins) << "\n";

    ROOT::DisableImplicitMT();
    std::vector<std::vector<std::unique_ptr<TH1D>>> periodHists;
    periodHists.reserve(samples.size());

    for (size_t is = 0; is < samples.size(); ++is) {
        std::cout << "  sample " << samples[is].tag << ": " << samples[is].mcPath << "\n";
        auto f = std::unique_ptr<TFile>(TFile::Open(samples[is].mcPath.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            std::cerr << "[Warn] cannot open " << samples[is].mcPath << "\n";
            periodHists.emplace_back();
            continue;
        }
        TChain chain(mcTree.c_str());
        GeneralHelper::fillChainFromAO2D(chain, f.get());
        if (chain.GetNtrees() <= 0) {
            std::cerr << "[Warn] no " << mcTree << " trees found in " << samples[is].mcPath << "\n";
            periodHists.emplace_back();
            continue;
        }
        ROOT::RDataFrame rdf(chain);
        auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
        auto res = AcceptanceHelper::ComputeAcceptanceFlexible(ready,
                                                               radBins,
                                                               {},
                                                               ctBinsByRad,
                                                               {},
                                                               {},
                                                               basicSelection,
                                                               {},
                                                               {},
                                                               requireTwoBody,
                                                               "fCentralityFT0C",
                                                               "fIsSurvEvSel",
                                                               "fIsReco",
                                                               "fGenDecRad",
                                                               "fGenCt",
                                                               "fGenPt",
                                                               constrainDenominatorOuterBin);
        auto cloned = CloneHistVector(PickBoth(res), "h_eff_" + samples[is].tag, radBins);
        for (auto &h : cloned) StyleHist(h.get(), samples[is].color, samples[is].marker);
        periodHists.push_back(std::move(cloned));
        res.Clear();
    }

    const auto centralityBins = binning.value("cen_bins", std::vector<double>{0.0, 100.0});
    const double centralityMin = centralityBins.size() >= 2 ? centralityBins.front() : 0.0;
    const double centralityMax = centralityBins.size() >= 2 ? centralityBins.back() : 100.0;
    const auto eventWeights = EventWeights(samples, eventHistPath, centralityMin, centralityMax);
    const auto equalWeights = NormalizedWeights(samples.size());
    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] period event weights:";
    for (size_t is = 0; is < samples.size(); ++is) {
        std::cout << " " << samples[is].tag << "="
                  << (is < eventWeights.size() ? eventWeights[is] : 0.0);
    }
    std::cout << "\n";

    auto mergedHists = MakeWeightedAverage(periodHists, eventWeights, radBins);
    auto equalWeightHists = MakeWeightedAverage(periodHists, equalWeights, radBins);
    for (auto &h : mergedHists) StyleHist(h.get(), kBlack, kFullCircle);
    for (auto &h : equalWeightHists) StyleHist(h.get(), kGray + 2, kOpenSquare);

    std::vector<TH1D *> mergedPtrs;
    mergedPtrs.reserve(mergedHists.size());
    for (auto &h : mergedHists) mergedPtrs.push_back(h.get());

    const std::string note = Form("rad-ct mode, %s", basicSelection.c_str());
    DrawCurves(radBins, mergedPtrs,
               "Merged event-weighted MC efficiency",
               note,
               outDir + "/general_config_merged_rad_ct_mc_efficiency_curves.pdf");
    DrawMap(radBins, mergedPtrs,
            "Merged event-weighted MC efficiency",
            note,
            outDir + "/general_config_merged_rad_ct_mc_efficiency_map.pdf");

    const std::string multipage = outDir + "/general_config_merged_rad_ct_mc_efficiency_by_period.pdf";
    for (size_t is = 0; is < samples.size(); ++is) {
        std::vector<TH1D *> ptrs;
        for (auto &h : periodHists[is]) ptrs.push_back(h.get());
        const std::string pageTarget = multipage + (is == 0 ? "(" : "");
        DrawMap(radBins, ptrs,
                samples[is].tag + " MC efficiency",
                note,
                pageTarget);
    }
    const std::string mergedPageTarget = multipage + ")";
    DrawMap(radBins, mergedPtrs,
            "Merged event-weighted MC efficiency",
            note,
            mergedPageTarget);

    const std::string perRadDir = outDir + "/per_rad_bin";
    std::filesystem::create_directories(perRadDir);
    for (size_t ir = 0; ir + 1 < radBins.size(); ++ir) {
        const std::string perRadPdf =
            perRadDir + "/period_comparison_rad_" +
            FileEdge(radBins[ir]) + "_" + FileEdge(radBins[ir + 1]) + ".pdf";
        DrawPeriodComparisonForRad(ir,
                                   radBins,
                                   samples,
                                   periodHists,
                                   mergedHists,
                                   equalWeightHists,
                                   note,
                                   perRadPdf,
                                   "");
    }

    const std::string perRadMultipage =
        outDir + "/general_config_merged_rad_ct_mc_efficiency_period_comparison_per_rad.pdf";
    for (size_t ir = 0; ir + 1 < radBins.size(); ++ir) {
        const bool firstPage = ir == 0;
        const bool lastPage = ir + 2 == radBins.size();
        const std::string pageTarget =
            perRadMultipage + (firstPage ? "(" : (lastPage ? ")" : ""));
        DrawPeriodComparisonForRad(ir,
                                   radBins,
                                   samples,
                                   periodHists,
                                   mergedHists,
                                   equalWeightHists,
                                   note,
                                   "",
                                   pageTarget);
    }

    WriteRoot(outDir + "/general_config_merged_rad_ct_mc_efficiency.root",
              samples, radBins, periodHists, mergedHists, equalWeightHists);
    std::cout << "[DrawGeneralConfigRadCtMCEfficiency] wrote outputs to " << outDir << "\n";
}
