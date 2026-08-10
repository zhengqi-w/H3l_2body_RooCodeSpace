#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPad.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../Tools/GeneralHelper.hpp"

namespace {

struct GroupInput {
    std::string label;
    std::vector<std::string> paths;
    Color_t color{kBlack};
    Style_t marker{kFullCircle};
};

struct GroupHists {
    std::string label;
    Color_t color{kBlack};
    Style_t marker{kFullCircle};
    std::vector<std::unique_ptr<TH1D>> denomRad;
    std::vector<std::unique_ptr<TH1D>> numerRad;
    std::unique_ptr<TH1D> efficiency;
};

std::string ReplaceFirst(std::string s, const std::string &from, const std::string &to)
{
    const size_t pos = s.find(from);
    if (pos != std::string::npos) s.replace(pos, from.size(), to);
    return s;
}

std::vector<std::string> BuildG4PathsFromNoG4(const std::vector<std::string> &paths)
{
    std::vector<std::string> out;
    out.reserve(paths.size());
    for (auto p : paths) {
        p = ReplaceFirst(p, "LHC25g11/NCrossedRows", "LHC25g11_G4list/NCrossedRows");
        p = ReplaceFirst(p, "LHC26e5/reweighted", "LHC26e5_G4list/reweighted");
        p = ReplaceFirst(p, "LHC26e6/reweighted", "LHC26e6_G4list/reweighted");
        out.push_back(p);
    }
    return out;
}

std::unique_ptr<TH1D> MakeRadHist(const std::string &name, double rmin, double rmax)
{
    const int nBins = std::max(12, static_cast<int>(std::ceil((rmax - rmin) / 0.03)));
    auto h = std::make_unique<TH1D>(name.c_str(), ";#it{R}_{dec}^{gen} (cm);Counts", nBins, rmin, rmax);
    h->Sumw2();
    h->SetDirectory(nullptr);
    return h;
}

bool InRange(double x, double lo, double hi)
{
    return x >= lo && x < hi;
}

bool InAnyCtBin(double ct, const std::vector<double> &ctEdges)
{
    if (ctEdges.size() < 2) return false;
    return InRange(ct, ctEdges.front(), ctEdges.back());
}

void AddAODFileToChain(TChain &chain, const std::string &path)
{
    auto f = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[Warn] cannot open " << path << "\n";
        return;
    }
    GeneralHelper::fillChainFromAO2D(chain, f.get());
}

void StyleHist(TH1D *h, Color_t color, Style_t marker, int lineStyle = 1)
{
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(0.95);
    h->SetLineWidth(3);
    h->SetLineStyle(lineStyle);
}

std::unique_ptr<TH1D> MakeRatio(const TH1D *num, const TH1D *den, const std::string &name)
{
    if (!num || !den || num->GetNbinsX() != den->GetNbinsX()) return nullptr;
    auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(num->Clone(name.c_str())));
    h->SetDirectory(nullptr);
    h->Divide(den);
    h->SetStats(false);
    h->GetYaxis()->SetTitle("G4list / no-G4list");
    return h;
}

double MaxHist(const std::vector<TH1D *> &hists)
{
    double ymax = 0.;
    for (const auto *h : hists) {
        if (!h) continue;
        ymax = std::max(ymax, h->GetMaximum());
    }
    return ymax > 0. ? ymax : 1.;
}

GroupHists BuildGroup(const GroupInput &input,
                      const std::vector<double> &radBins,
                      const std::vector<std::vector<double>> &ctBinsByRad,
                      const std::string &treeName,
                      const bool requireTwoBody)
{
    GroupHists out;
    out.label = input.label;
    out.color = input.color;
    out.marker = input.marker;
    const int nRad = static_cast<int>(radBins.size()) - 1;
    out.denomRad.reserve(nRad);
    out.numerRad.reserve(nRad);
    for (int ir = 0; ir < nRad; ++ir) {
        out.denomRad.push_back(MakeRadHist(Form("h_den_%s_rad_%d", input.label.c_str(), ir), radBins[ir], radBins[ir + 1]));
        out.numerRad.push_back(MakeRadHist(Form("h_num_%s_rad_%d", input.label.c_str(), ir), radBins[ir], radBins[ir + 1]));
    }
    out.efficiency = std::make_unique<TH1D>(Form("h_eff_%s", input.label.c_str()),
                                            ";#it{R}_{dec}^{gen} (cm);Numerator / denominator",
                                            nRad, radBins.data());
    out.efficiency->Sumw2();
    out.efficiency->SetDirectory(nullptr);

    TChain chain(treeName.c_str());
    for (const auto &path : input.paths) {
        AddAODFileToChain(chain, path);
    }
    if (chain.GetNtrees() <= 0) {
        std::cerr << "[Warn] no trees for " << input.label << "\n";
        return out;
    }

    ROOT::RDataFrame rdf(chain);
    auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode node(ready);
    auto cols = node.GetColumnNames();
    const bool haveRecoMCCollision = std::find(cols.begin(), cols.end(), "fIsRecoMCCollision") != cols.end();
    const bool haveEvSel = std::find(cols.begin(), cols.end(), "fIsSurvEvSel") != cols.end();
    const bool haveTwoBody = std::find(cols.begin(), cols.end(), "fIsTwoBodyDecay") != cols.end();

    std::cout << "[BuildGroup] " << input.label << ": entries=" << chain.GetEntries()
              << ", trees=" << chain.GetNtrees() << "\n";

    node.Foreach(
        [&](double genRad, double genCt, bool isReco, double recoRad, bool evSel, bool recoMCCollision, bool isTwoBody) {
            if (requireTwoBody && !isTwoBody) return;
            const auto upper = std::upper_bound(radBins.begin(), radBins.end(), genRad);
            if (upper == radBins.begin() || upper == radBins.end()) return;
            const int ir = static_cast<int>(upper - radBins.begin()) - 1;
            if (ir < 0 || ir >= nRad) return;
            if (!InAnyCtBin(genCt, ctBinsByRad[static_cast<size_t>(ir)])) return;

            const bool passDenom = evSel && recoMCCollision;
            const bool passNumer = isReco && recoRad > 0.8;
            if (passDenom) out.denomRad[static_cast<size_t>(ir)]->Fill(genRad);
            if (passNumer) out.numerRad[static_cast<size_t>(ir)]->Fill(genRad);
        },
        {"fGenDecRad", "fGenCt", "fIsReco", "fDecRad",
         haveEvSel ? "fIsSurvEvSel" : "fIsReco",
         haveRecoMCCollision ? "fIsRecoMCCollision" : "fIsReco",
         haveTwoBody ? "fIsTwoBodyDecay" : "fIsReco"});

    for (int ir = 0; ir < nRad; ++ir) {
        const double den = out.denomRad[static_cast<size_t>(ir)]->Integral();
        const double num = out.numerRad[static_cast<size_t>(ir)]->Integral();
        const double eff = den > 0. ? num / den : 0.;
        const double err = den > 0. ? std::sqrt(std::max(0., eff * (1. - eff) / den)) : 0.;
        out.efficiency->SetBinContent(ir + 1, eff);
        out.efficiency->SetBinError(ir + 1, err);
    }

    StyleHist(out.efficiency.get(), input.color, input.marker);
    return out;
}

void DrawOneRadBin(const std::vector<GroupHists> &groups,
                   const std::vector<double> &radBins,
                   size_t ir,
                   const std::string &outPdf)
{
    TCanvas c(Form("c_rad_num_den_%zu", ir), "", 1000, 850);
    auto *pTop = new TPad("pTop", "", 0., 0.32, 1., 1.);
    auto *pBot = new TPad("pBot", "", 0., 0., 1., 0.32);
    pTop->SetLeftMargin(0.12);
    pTop->SetRightMargin(0.04);
    pTop->SetBottomMargin(0.02);
    pTop->SetTopMargin(0.07);
    pTop->SetLogy();
    pBot->SetLeftMargin(0.12);
    pBot->SetRightMargin(0.04);
    pBot->SetBottomMargin(0.32);
    pBot->SetTopMargin(0.03);
    pTop->Draw();
    pBot->Draw();

    pTop->cd();
    std::vector<TH1D *> drawHists;
    for (const auto &g : groups) {
        if (ir < g.denomRad.size()) drawHists.push_back(g.denomRad[ir].get());
        if (ir < g.numerRad.size()) drawHists.push_back(g.numerRad[ir].get());
    }
    TH1D frame("h_frame", ";#it{R}_{dec}^{gen} (cm);Counts", 100, radBins[ir], radBins[ir + 1]);
    frame.SetStats(false);
    frame.SetMinimum(0.5);
    frame.SetMaximum(MaxHist(drawHists) * 10.);
    frame.GetYaxis()->SetTitleOffset(1.1);
    frame.GetXaxis()->SetLabelSize(0.);
    frame.Draw("AXIS");

    TLegend leg(0.50, 0.55, 0.92, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.032);

    for (const auto &g : groups) {
        if (ir >= g.denomRad.size() || ir >= g.numerRad.size()) continue;
        StyleHist(g.denomRad[ir].get(), g.color, g.marker, 1);
        StyleHist(g.numerRad[ir].get(), g.color, g.marker, 7);
        g.denomRad[ir]->Draw("HIST E SAME");
        g.numerRad[ir]->Draw("HIST E SAME");
        leg.AddEntry(g.denomRad[ir].get(), (g.label + " denominator").c_str(), "l");
        leg.AddEntry(g.numerRad[ir].get(), (g.label + " numerator").c_str(), "l");
    }
    leg.Draw();
    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.036);
    text.DrawLatex(0.16, 0.87, "ALICE Work In Progress");
    text.SetTextSize(0.032);
    text.DrawLatex(0.16, 0.81, Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.DrawLatex(0.16, 0.75, "Solid: denominator, dashed: numerator");

    pBot->cd();
    std::unique_ptr<TH1D> rDen;
    std::unique_ptr<TH1D> rNum;
    if (groups.size() >= 2 && ir < groups[0].denomRad.size() && ir < groups[1].denomRad.size()) {
        rDen = MakeRatio(groups[1].denomRad[ir].get(), groups[0].denomRad[ir].get(), "r_den");
        rNum = MakeRatio(groups[1].numerRad[ir].get(), groups[0].numerRad[ir].get(), "r_num");
    }
    TH1D rFrame("h_rframe", ";#it{R}_{dec}^{gen} (cm);G4 / no-G4", 100, radBins[ir], radBins[ir + 1]);
    rFrame.SetStats(false);
    rFrame.SetMinimum(0.4);
    rFrame.SetMaximum(1.6);
    rFrame.GetXaxis()->SetTitleSize(0.10);
    rFrame.GetYaxis()->SetTitleSize(0.09);
    rFrame.GetXaxis()->SetLabelSize(0.08);
    rFrame.GetYaxis()->SetLabelSize(0.075);
    rFrame.GetYaxis()->SetTitleOffset(0.55);
    rFrame.Draw("AXIS");
    if (rDen) {
        StyleHist(rDen.get(), kBlack, kFullCircle, 1);
        rDen->Draw("E SAME");
    }
    if (rNum) {
        StyleHist(rNum.get(), kRed + 1, kOpenSquare, 7);
        rNum->Draw("E SAME");
    }
    c.SaveAs(outPdf.c_str());
}

void DrawEfficiencyCompare(const std::vector<GroupHists> &groups, const std::string &outPdf)
{
    TCanvas c("c_rad_eff_compare", "", 950, 720);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);

    double ymax = 0.;
    for (const auto &g : groups) ymax = std::max(ymax, g.efficiency ? g.efficiency->GetMaximum() : 0.);
    auto *frame = groups.front().efficiency ? static_cast<TH1D *>(groups.front().efficiency->Clone("h_eff_frame")) : nullptr;
    if (!frame) return;
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.);
    frame->SetMaximum(std::max(0.05, ymax * 1.35));
    frame->Draw("AXIS");
    TLegend leg(0.58, 0.70, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.035);
    for (const auto &g : groups) {
        if (!g.efficiency) continue;
        g.efficiency->Draw("E1 X0 SAME");
        leg.AddEntry(g.efficiency.get(), g.label.c_str(), "pe");
    }
    leg.Draw();
    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, "Rad-ct MC efficiency, numerator / denominator");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

} // namespace

void DrawRadCtMCEffNumeratorDenominatorRadQA(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config_merged.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtMCEffNumeratorDenominatorRadQA")
{
    std::filesystem::create_directories(outputDir);
    const auto cfg = GeneralHelper::LoadJsonFile(configPath);
    const auto common = cfg.at("common");
    const auto binning = common.at("binning");
    const auto radBins = binning.at("rad_bins").get<std::vector<double>>();
    const auto ctBinsByRad = binning.at("ct_bins_by_rad").get<std::vector<std::vector<double>>>();
    const std::string treeName = common.value("tree_names", GeneralHelper::Json::object()).value("mc", std::string("O2mchypcands"));
    const bool requireTwoBody = common.value("selection", GeneralHelper::Json::object()).value("mc_acceptance_require_two_body", true);

    std::vector<std::string> noG4Paths;
    for (const auto &p : common.at("periods")) {
        noG4Paths.push_back(p.value("mc_path", std::string()));
    }
    std::vector<GroupInput> inputs = {
        {"no-G4list merged", noG4Paths, kAzure + 2, kFullCircle},
        {"G4list merged", BuildG4PathsFromNoG4(noG4Paths), kRed + 1, kFullSquare}
    };

    std::vector<GroupHists> groups;
    for (const auto &input : inputs) {
        groups.push_back(BuildGroup(input, radBins, ctBinsByRad, treeName, requireTwoBody));
    }

    const std::string multiPdf = std::string(outputDir) + "/rad_ct_num_den_rad_distribution_by_bin.pdf";
    TCanvas opener("c_open", "", 10, 10);
    opener.SaveAs((multiPdf + "[").c_str());
    for (size_t ir = 0; ir + 1 < radBins.size(); ++ir) {
        DrawOneRadBin(groups, radBins, ir, multiPdf);
    }
    opener.SaveAs((multiPdf + "]").c_str());

    DrawEfficiencyCompare(groups, std::string(outputDir) + "/rad_ct_efficiency_g4list_vs_nog4list.pdf");

    TFile out((std::string(outputDir) + "/rad_ct_num_den_rad_distribution_qa.root").c_str(), "RECREATE");
    for (const auto &g : groups) {
        out.mkdir(g.label.c_str());
        out.cd(g.label.c_str());
        if (g.efficiency) g.efficiency->Write();
        for (size_t ir = 0; ir < g.denomRad.size(); ++ir) {
            g.denomRad[ir]->Write(Form("h_den_rad_%zu", ir));
            g.numerRad[ir]->Write(Form("h_num_rad_%zu", ir));
        }
        out.cd();
    }
    out.Close();

    std::cout << "[DrawRadCtMCEffNumeratorDenominatorRadQA] wrote outputs to " << outputDir << "\n";
}
