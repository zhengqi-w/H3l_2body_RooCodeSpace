#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLine.h>
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
};

struct GroupH2 {
    std::string label;
    std::unique_ptr<TH2D> denom;
    std::unique_ptr<TH2D> numer;
    std::unique_ptr<TH2D> eff;
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

void AddAODFileToChain(TChain &chain, const std::string &path)
{
    auto f = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[Warn] cannot open " << path << "\n";
        return;
    }
    GeneralHelper::fillChainFromAO2D(chain, f.get());
}

bool InAnyCtBin(double ct, const std::vector<double> &ctEdges)
{
    return ctEdges.size() >= 2 && ct >= ctEdges.front() && ct < ctEdges.back();
}

int FindRadBin(const std::vector<double> &radBins, double genRad)
{
    if (radBins.size() < 2 || genRad < radBins.front() || genRad >= radBins.back()) return -1;
    auto upper = std::upper_bound(radBins.begin(), radBins.end(), genRad);
    return static_cast<int>(upper - radBins.begin()) - 1;
}

ROOT::RDF::RNode DefineIntFlag(ROOT::RDF::RNode node, const std::string &source, const std::string &alias)
{
    auto cols = node.GetColumnNames();
    if (std::find(cols.begin(), cols.end(), source) == cols.end()) {
        return node.Define(alias, []() { return 1; });
    }
    return node.Define(alias, "(" + source + ") ? 1 : 0");
}

ROOT::RDF::RNode DefineDoubleAlias(ROOT::RDF::RNode node, const std::string &source, const std::string &alias)
{
    return node.Define(alias, "static_cast<double>(" + source + ")");
}

std::unique_ptr<TH2D> MakeH2(const std::string &name,
                             const std::string &title,
                             double radMin,
                             double radMax,
                             double ctMax)
{
    auto h = std::make_unique<TH2D>(name.c_str(),
                                    title.c_str(),
                                    220, radMin, radMax,
                                    220, 0., ctMax);
    h->SetDirectory(nullptr);
    h->Sumw2();
    h->GetXaxis()->SetTitle("#it{R}_{dec}^{gen} (cm)");
    h->GetYaxis()->SetTitle("#it{c}t_{gen} (cm)");
    h->GetZaxis()->SetTitle("Counts");
    return h;
}

GroupH2 BuildGroup(const GroupInput &input,
                   const std::vector<double> &radBins,
                   const std::vector<std::vector<double>> &ctBinsByRad,
                   const std::string &treeName,
                   const bool requireTwoBody)
{
    GroupH2 out;
    out.label = input.label;
    double ctMax = 0.;
    for (const auto &edges : ctBinsByRad) {
        if (!edges.empty()) ctMax = std::max(ctMax, edges.back());
    }
    ctMax = std::max(40., ctMax);
    out.denom = MakeH2("h2_den_" + input.label, input.label + " denominator;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)", radBins.front(), radBins.back(), ctMax);
    out.numer = MakeH2("h2_num_" + input.label, input.label + " numerator;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)", radBins.front(), radBins.back(), ctMax);
    out.eff = MakeH2("h2_eff_" + input.label, input.label + " efficiency;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)", radBins.front(), radBins.back(), ctMax);
    out.eff->GetZaxis()->SetTitle("Numerator / denominator");

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
    node = DefineIntFlag(node, "fIsReco", "__is_reco_int");
    node = DefineIntFlag(node, "fIsSurvEvSel", "__evsel_int");
    node = DefineIntFlag(node, "fIsRecoMCCollision", "__reco_mc_collision_int");
    node = DefineIntFlag(node, "fIsTwoBodyDecay", "__two_body_int");
    node = DefineDoubleAlias(node, "fGenDecRad", "__gen_rad_double");
    node = DefineDoubleAlias(node, "fGenCt", "__gen_ct_double");
    node = DefineDoubleAlias(node, "fDecRad", "__reco_rad_double");

    std::cout << "[BuildGroup2D] " << input.label << ": entries=" << chain.GetEntries()
              << ", trees=" << chain.GetNtrees() << "\n";

    node.Foreach(
        [&](double genRad, double genCt, int isReco, double recoRad, int evSel, int recoMCCollision, int isTwoBody) {
            if (requireTwoBody && !isTwoBody) return;
            const int ir = FindRadBin(radBins, genRad);
            if (ir < 0) return;
            if (!InAnyCtBin(genCt, ctBinsByRad[static_cast<size_t>(ir)])) return;
            if (evSel && recoMCCollision) out.denom->Fill(genRad, genCt);
            if (isReco && recoRad > 0.8) out.numer->Fill(genRad, genCt);
        },
        {"__gen_rad_double", "__gen_ct_double", "__is_reco_int", "__reco_rad_double", "__evsel_int", "__reco_mc_collision_int", "__two_body_int"});

    out.eff->Divide(out.numer.get(), out.denom.get(), 1.0, 1.0, "B");
    return out;
}

std::unique_ptr<TH2D> MakeRatio2D(const TH2D *num, const TH2D *den, const std::string &name, const std::string &title)
{
    if (!num || !den) return nullptr;
    auto out = std::unique_ptr<TH2D>(static_cast<TH2D *>(num->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->SetTitle(title.c_str());
    out->Divide(den);
    out->GetZaxis()->SetTitle("G4list / no-G4list");
    return out;
}

void DrawRadCtGrid(const std::vector<double> &radBins, const std::vector<std::vector<double>> &ctBinsByRad)
{
    std::vector<std::unique_ptr<TLine>> lines;
    lines.reserve(256);
    auto addLine = [&](double x1, double y1, double x2, double y2, int style = 1) {
        auto line = std::make_unique<TLine>(x1, y1, x2, y2);
        line->SetLineColor(kGray + 2);
        line->SetLineWidth(1);
        line->SetLineStyle(style);
        line->Draw("same");
        lines.push_back(std::move(line));
    };
    const double ctGlobalMax = 40.;
    for (double r : radBins) {
        addLine(r, 0., r, ctGlobalMax, 7);
    }
    for (size_t ir = 0; ir < ctBinsByRad.size() && ir + 1 < radBins.size(); ++ir) {
        for (double ct : ctBinsByRad[ir]) {
            addLine(radBins[ir], ct, radBins[ir + 1], ct, 1);
        }
    }
    gPad->Modified();
}

void DrawH2Page(TH2D *h,
                const std::vector<double> &radBins,
                const std::vector<std::vector<double>> &ctBinsByRad,
                const std::string &label,
                const std::string &outPdf,
                bool logz,
                double zMin = -1.,
                double zMax = -1.)
{
    if (!h) return;
    TCanvas c(("c_" + label).c_str(), "", 1050, 850);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.16);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.07);
    c.SetTicks(1, 1);
    c.SetLogz(logz);
    h->SetStats(false);
    if (zMin >= 0.) h->SetMinimum(zMin);
    if (zMax > zMin) h->SetMaximum(zMax);
    h->GetXaxis()->SetTitleSize(0.043);
    h->GetYaxis()->SetTitleSize(0.043);
    h->GetZaxis()->SetTitleSize(0.038);
    h->GetYaxis()->SetTitleOffset(1.18);
    h->Draw("COLZ");
    DrawRadCtGrid(radBins, ctBinsByRad);
    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, label.c_str());
    c.SaveAs(outPdf.c_str());
}

} // namespace

void DrawRadCtMCEffNumeratorDenominator2DQA(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config_merged.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtMCEffNumeratorDenominator2DQA")
{
    std::filesystem::create_directories(outputDir);
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBird);

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
        {"no-G4list merged", noG4Paths},
        {"G4list merged", BuildG4PathsFromNoG4(noG4Paths)}
    };
    std::vector<GroupH2> groups;
    for (const auto &input : inputs) groups.push_back(BuildGroup(input, radBins, ctBinsByRad, treeName, requireTwoBody));

    const std::string multiPdf = std::string(outputDir) + "/rad_ct_num_den_2d_g4list_comparison.pdf";
    TCanvas opener("c_open_2d", "", 10, 10);
    opener.SaveAs((multiPdf + "[").c_str());
    for (auto &g : groups) {
        DrawH2Page(g.denom.get(), radBins, ctBinsByRad, g.label + " denominator", multiPdf, true, 0.5);
        DrawH2Page(g.numer.get(), radBins, ctBinsByRad, g.label + " numerator", multiPdf, true, 0.5);
        DrawH2Page(g.eff.get(), radBins, ctBinsByRad, g.label + " efficiency", multiPdf, false, 0., 1.);
    }
    auto rDen = MakeRatio2D(groups[1].denom.get(), groups[0].denom.get(), "h2_ratio_den_g4_over_nog4", "Denominator ratio;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)");
    auto rNum = MakeRatio2D(groups[1].numer.get(), groups[0].numer.get(), "h2_ratio_num_g4_over_nog4", "Numerator ratio;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)");
    auto rEff = MakeRatio2D(groups[1].eff.get(), groups[0].eff.get(), "h2_ratio_eff_g4_over_nog4", "Efficiency ratio;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)");
    DrawH2Page(rDen.get(), radBins, ctBinsByRad, "G4list / no-G4list denominator", multiPdf, false, 0.5, 1.5);
    DrawH2Page(rNum.get(), radBins, ctBinsByRad, "G4list / no-G4list numerator", multiPdf, false, 0.5, 1.5);
    DrawH2Page(rEff.get(), radBins, ctBinsByRad, "G4list / no-G4list efficiency", multiPdf, false, 0.5, 1.5);
    opener.SaveAs((multiPdf + "]").c_str());

    TFile out((std::string(outputDir) + "/rad_ct_num_den_2d_qa.root").c_str(), "RECREATE");
    for (auto &g : groups) {
        if (g.denom) g.denom->Write();
        if (g.numer) g.numer->Write();
        if (g.eff) g.eff->Write();
    }
    if (rDen) rDen->Write();
    if (rNum) rNum->Write();
    if (rEff) rEff->Write();
    out.Close();
    std::cout << "[DrawRadCtMCEffNumeratorDenominator2DQA] wrote outputs to " << outputDir << "\n";
}
