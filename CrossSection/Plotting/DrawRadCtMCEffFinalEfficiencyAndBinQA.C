#include <ROOT/RDataFrame.hxx>
#include <TBox.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPad.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../Tools/AcceptanceHelper.h"
#include "../../Tools/GeneralHelper.hpp"

namespace {

struct GroupInput {
    std::string label;
    std::vector<std::string> paths;
    Color_t color{kBlack};
    Style_t marker{kFullCircle};
};

struct GroupQA {
    std::string label;
    Color_t color{kBlack};
    Style_t marker{kFullCircle};
    std::vector<std::unique_ptr<TH1D>> effVsCt;
    std::vector<std::unique_ptr<TH1D>> effVsCtDenRadReleased;
    std::vector<std::unique_ptr<TH1D>> denomCountsVsCt;
    std::vector<std::unique_ptr<TH1D>> denomRadReleasedCountsVsCt;
    std::vector<std::unique_ptr<TH1D>> numerCountsVsCt;
    std::vector<std::unique_ptr<TH1D>> denomPtVsRad;
    std::vector<std::unique_ptr<TH1D>> numerPtVsRad;
    std::unique_ptr<TH1D> denomNoRadCutFineCt;
    std::unique_ptr<TH1D> numerNoRadCutFineCt;
    std::vector<std::vector<std::unique_ptr<TH2D>>> denom2D;
    std::vector<std::vector<std::unique_ptr<TH2D>>> numer2D;
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
        p = ReplaceFirst(p, "LHC25g11/AO2D", "LHC25g11_G4list/AO2D");
        p = ReplaceFirst(p, "LHC26e5/reweighted", "LHC26e5_G4list/reweighted");
        p = ReplaceFirst(p, "LHC26e5/AO2D", "LHC26e5_G4list/AO2D");
        p = ReplaceFirst(p, "LHC26e6/reweighted", "LHC26e6_G4list/reweighted");
        p = ReplaceFirst(p, "LHC26e6/AO2D", "LHC26e6_G4list/AO2D");
        p = ReplaceFirst(p, "updatedMC/AO2D_LHC25g11.root", "updatedMC_G4list/AO2D_LHC25g11_G4list.root");
        p = ReplaceFirst(p, "updatedMC/AO2D_LHC26e5.root", "updatedMC_G4list/AO2D_LHC26e5_G4list.root");
        p = ReplaceFirst(p, "updatedMC/AO2D_LHC26e6.root", "updatedMC_G4list/AO2D_LHC26e6_G4list.root");
        out.push_back(p);
    }
    return out;
}

std::vector<std::string> BuildNoG4PathsFromConfig(const std::vector<std::string> &paths)
{
    std::vector<std::string> out;
    out.reserve(paths.size());
    for (auto p : paths) {
        p = ReplaceFirst(p, "LHC25g11_G4list/NCrossedRows", "LHC25g11/NCrossedRows");
        p = ReplaceFirst(p, "LHC26e5_G4list/reweighted", "LHC26e5/reweighted");
        p = ReplaceFirst(p, "LHC26e6_G4list/reweighted", "LHC26e6/reweighted");
        p = ReplaceFirst(p, "updatedMC_G4list/AO2D_LHC25g11_G4list.root", "updatedMC/AO2D_LHC25g11.root");
        p = ReplaceFirst(p, "updatedMC_G4list/AO2D_LHC26e5_G4list.root", "updatedMC/AO2D_LHC26e5.root");
        p = ReplaceFirst(p, "updatedMC_G4list/AO2D_LHC26e6_G4list.root", "updatedMC/AO2D_LHC26e6.root");
        out.push_back(p);
    }
    return out;
}

std::vector<std::string> BuildUnweightedPaths(const std::vector<std::string> &paths)
{
    std::vector<std::string> out;
    out.reserve(paths.size());
    for (auto p : paths) {
        p = ReplaceFirst(p, "LHC25g11/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
                         "LHC25g11/NCrossedRows/AO2D_CustomV0s.root");
        p = ReplaceFirst(p, "LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
                         "LHC25g11_G4list/NCrossedRows/AO2D_CustomV0s.root");
        p = ReplaceFirst(p, "LHC26e5/reweighted/AO2D_combined_reweighted.root",
                         "LHC26e5/AO2D.root");
        p = ReplaceFirst(p, "LHC26e5_G4list/reweighted/AO2D_combined_reweighted.root",
                         "LHC26e5_G4list/AO2D.root");
        p = ReplaceFirst(p, "LHC26e6/reweighted/AO2D_combined_reweighted.root",
                         "LHC26e6/AO2D.root");
        p = ReplaceFirst(p, "LHC26e6_G4list/reweighted/AO2D_combined_reweighted.root",
                         "LHC26e6_G4list/AO2D.root");
        out.push_back(p);
    }
    return out;
}

std::string SiblingConfigPath(const std::string &configPath, const std::string &name)
{
    std::filesystem::path p(configPath);
    if (!p.has_parent_path()) return name;
    return (p.parent_path() / name).string();
}

std::string ExtractJsonArrayAfterKey(const std::string &text, const std::string &key)
{
    const std::string needle = "\"" + key + "\"";
    const size_t keyPos = text.find(needle);
    if (keyPos == std::string::npos) return {};
    const size_t firstBracket = text.find('[', keyPos);
    if (firstBracket == std::string::npos) return {};

    int depth = 0;
    bool inString = false;
    for (size_t i = firstBracket; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '"' && (i == 0 || text[i - 1] != '\\')) inString = !inString;
        if (inString) continue;
        if (ch == '[') ++depth;
        if (ch == ']') {
            --depth;
            if (depth == 0) {
                return text.substr(firstBracket, i - firstBracket + 1);
            }
        }
    }
    return {};
}

std::string RemoveCppLineComments(const std::string &text)
{
    std::istringstream input(text);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        const size_t pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        output << line << '\n';
    }
    return output.str();
}

bool LoadRadCtBinningFromParametersBase(const std::string &path,
                                        std::vector<double> &radBins,
                                        std::vector<std::vector<double>> &ctBinsByRad)
{
    std::ifstream input(path);
    if (!input.is_open()) return false;
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = RemoveCppLineComments(buffer.str());
    const std::string radArray = ExtractJsonArrayAfterKey(text, "rad_bins");
    const std::string ctArray = ExtractJsonArrayAfterKey(text, "ct_bins_by_rad");
    if (radArray.empty() || ctArray.empty()) return false;
    try {
        const auto radJson = GeneralHelper::Json::parse(radArray);
        const auto ctJson = GeneralHelper::Json::parse(ctArray);
        auto parsedRad = radJson.get<std::vector<double>>();
        auto parsedCt = ctJson.get<std::vector<std::vector<double>>>();
        if (parsedRad.size() >= 2 && parsedCt.size() == parsedRad.size() - 1) {
            radBins = std::move(parsedRad);
            ctBinsByRad = std::move(parsedCt);
            return true;
        }
    } catch (const std::exception &e) {
        std::cerr << "[Warn] failed parsing ParametersBase.md rad-ct binning: " << e.what() << "\n";
    }
    return false;
}

void LoadFullRadCtBinningIfNeeded(const std::string &configPath,
                                  std::vector<double> &radBins,
                                  std::vector<std::vector<double>> &ctBinsByRad)
{
    if (radBins.size() > 2) return;
    const std::string parametersPath = SiblingConfigPath(configPath, "ParametersBase.md");
    if (std::filesystem::exists(parametersPath) &&
        LoadRadCtBinningFromParametersBase(parametersPath, radBins, ctBinsByRad)) {
        std::cout << "[Info] current config has only target rad bin(s); use full rad-ct binning from "
                  << parametersPath << "\n";
    } else {
        std::cerr << "[Warn] failed to load full rad-ct binning from " << parametersPath << "\n";
    }
}

std::string EdgeTag(double x)
{
    std::string s = Form("%.4g", x);
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

std::string RadTag(double rMin, double rMax)
{
    return "rad_" + EdgeTag(rMin) + "_" + EdgeTag(rMax);
}

std::string RadCtTag(double rMin, double rMax, double ctMin, double ctMax)
{
    return RadTag(rMin, rMax) + "_ct_" + EdgeTag(ctMin) + "_" + EdgeTag(ctMax);
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

int FindBin(const std::vector<double> &edges, double value)
{
    if (edges.size() < 2 || value < edges.front() || value >= edges.back()) return -1;
    auto upper = std::upper_bound(edges.begin(), edges.end(), value);
    return static_cast<int>(upper - edges.begin()) - 1;
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

std::unique_ptr<TH2D> MakeBinH2(const std::string &name,
                                const std::string &title,
                                double radMin,
                                double radMax,
                                double ctMin,
                                double ctMax)
{
    auto h = std::make_unique<TH2D>(name.c_str(), title.c_str(),
                                    120, radMin, radMax,
                                    120, ctMin, ctMax);
    h->SetDirectory(nullptr);
    h->Sumw2();
    h->GetXaxis()->SetTitle("#it{R}_{dec}^{gen} (cm)");
    h->GetYaxis()->SetTitle("#it{c}t_{gen} (cm)");
    h->GetZaxis()->SetTitle("Counts");
    return h;
}

void StyleEff(TH1D *h, Color_t color, Style_t marker)
{
    if (!h) return;
    h->SetDirectory(nullptr);
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.05);
    h->SetLineWidth(3);
    h->GetXaxis()->SetTitle("#it{c}t_{gen} (cm)");
    h->GetYaxis()->SetTitle("Final MC efficiency");
}

void StyleCounts(TH1D *h, Color_t color, Style_t marker, Style_t lineStyle)
{
    if (!h) return;
    h->SetDirectory(nullptr);
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.05);
    h->SetLineStyle(lineStyle);
    h->SetLineWidth(3);
    h->GetXaxis()->SetTitle("#it{c}t_{gen} (cm)");
    h->GetYaxis()->SetTitle("Counts");
}

GroupQA BuildGroup(const GroupInput &input,
                   const std::vector<double> &radBins,
                   const std::vector<std::vector<double>> &ctBinsByRad,
                   const std::vector<double> &fineCtBins,
                   const std::vector<double> &ptBins,
                   const std::string &treeName,
                   bool requireTwoBody,
                   bool constrainDenominatorOuterBin)
{
    GroupQA out;
    out.label = input.label;
    out.color = input.color;
    out.marker = input.marker;
    const int nRad = static_cast<int>(radBins.size()) - 1;
    out.effVsCt.reserve(nRad);
    out.effVsCtDenRadReleased.reserve(nRad);
    out.denomCountsVsCt.reserve(nRad);
    out.denomRadReleasedCountsVsCt.reserve(nRad);
    out.numerCountsVsCt.reserve(nRad);
    out.denomPtVsRad.reserve(nRad);
    out.numerPtVsRad.reserve(nRad);
    if (fineCtBins.size() >= 2) {
        out.denomNoRadCutFineCt = std::make_unique<TH1D>(
            Form("h_den_no_rad_cut_fine_ct_%s", input.label.c_str()),
            ";#it{c}t_{gen} (cm);Counts",
            static_cast<int>(fineCtBins.size()) - 1,
            fineCtBins.data());
        out.numerNoRadCutFineCt = std::make_unique<TH1D>(
            Form("h_num_no_rad_cut_fine_ct_%s", input.label.c_str()),
            ";#it{c}t_{gen} (cm);Counts",
            static_cast<int>(fineCtBins.size()) - 1,
            fineCtBins.data());
        out.denomNoRadCutFineCt->Sumw2();
        out.numerNoRadCutFineCt->Sumw2();
        StyleCounts(out.denomNoRadCutFineCt.get(), input.color, input.marker, 1);
        StyleCounts(out.numerNoRadCutFineCt.get(), input.color, input.marker, 2);
    }
    out.denom2D.resize(nRad);
    out.numer2D.resize(nRad);

    for (int ir = 0; ir < nRad; ++ir) {
        const auto &ctEdges = ctBinsByRad[static_cast<size_t>(ir)];
        auto hEff = std::make_unique<TH1D>(Form("h_eff_%s_rad_%d", input.label.c_str(), ir),
                                           ";#it{c}t_{gen} (cm);Final MC efficiency",
                                           static_cast<int>(ctEdges.size()) - 1,
                                           ctEdges.data());
        hEff->Sumw2();
        StyleEff(hEff.get(), input.color, input.marker);
        out.effVsCt.push_back(std::move(hEff));
        auto hEffDenRadReleased = std::make_unique<TH1D>(
            Form("h_eff_den_rad_released_%s_rad_%d", input.label.c_str(), ir),
            ";#it{c}t_{gen} (cm);Final MC efficiency",
            static_cast<int>(ctEdges.size()) - 1,
            ctEdges.data());
        hEffDenRadReleased->Sumw2();
        StyleEff(hEffDenRadReleased.get(), input.color, input.marker);
        out.effVsCtDenRadReleased.push_back(std::move(hEffDenRadReleased));
        auto hDenCounts = std::make_unique<TH1D>(Form("h_den_counts_%s_rad_%d", input.label.c_str(), ir),
                                                 ";#it{c}t_{gen} (cm);Counts",
                                                 static_cast<int>(ctEdges.size()) - 1,
                                                 ctEdges.data());
        auto hDenRadReleasedCounts = std::make_unique<TH1D>(
            Form("h_den_rad_released_counts_%s_rad_%d", input.label.c_str(), ir),
            ";#it{c}t_{gen} (cm);Counts",
            static_cast<int>(ctEdges.size()) - 1,
            ctEdges.data());
        auto hNumCounts = std::make_unique<TH1D>(Form("h_num_counts_%s_rad_%d", input.label.c_str(), ir),
                                                 ";#it{c}t_{gen} (cm);Counts",
                                                 static_cast<int>(ctEdges.size()) - 1,
                                                 ctEdges.data());
        hDenCounts->Sumw2();
        hDenRadReleasedCounts->Sumw2();
        hNumCounts->Sumw2();
        StyleCounts(hDenCounts.get(), input.color, input.marker, 1);
        StyleCounts(hDenRadReleasedCounts.get(), input.color, input.marker, 1);
        StyleCounts(hNumCounts.get(), input.color, input.marker, 2);
        out.denomCountsVsCt.push_back(std::move(hDenCounts));
        out.denomRadReleasedCountsVsCt.push_back(std::move(hDenRadReleasedCounts));
        out.numerCountsVsCt.push_back(std::move(hNumCounts));
        if (ptBins.size() >= 2) {
            auto hDenPt = std::make_unique<TH1D>(Form("h_den_pt_%s_rad_%d", input.label.c_str(), ir),
                                                 ";#it{p}_{T}^{gen} (GeV/#it{c});Counts",
                                                 static_cast<int>(ptBins.size()) - 1,
                                                 ptBins.data());
            auto hNumPt = std::make_unique<TH1D>(Form("h_num_pt_%s_rad_%d", input.label.c_str(), ir),
                                                 ";#it{p}_{T}^{gen} (GeV/#it{c});Counts",
                                                 static_cast<int>(ptBins.size()) - 1,
                                                 ptBins.data());
            hDenPt->Sumw2();
            hNumPt->Sumw2();
            StyleCounts(hDenPt.get(), input.color, input.marker, 1);
            StyleCounts(hNumPt.get(), input.color, input.marker, 2);
            hDenPt->GetXaxis()->SetTitle("#it{p}_{T}^{gen} (GeV/#it{c})");
            hNumPt->GetXaxis()->SetTitle("#it{p}_{T}^{gen} (GeV/#it{c})");
            out.denomPtVsRad.push_back(std::move(hDenPt));
            out.numerPtVsRad.push_back(std::move(hNumPt));
        }
        out.denom2D[static_cast<size_t>(ir)].reserve(ctEdges.size() - 1);
        out.numer2D[static_cast<size_t>(ir)].reserve(ctEdges.size() - 1);
        for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
            const double radWidth = radBins[ir + 1] - radBins[ir];
            const double ctWidth = ctEdges[ict + 1] - ctEdges[ict];
            const double radPad = std::max(0.02, 0.08 * radWidth);
            const double ctPad = std::max(0.02, 0.08 * ctWidth);
            const double plotRadMin = std::max(0.0, radBins[ir] - radPad);
            const double plotRadMax = radBins[ir + 1] + radPad;
            const double plotCtMin = std::max(0.0, ctEdges[ict] - ctPad);
            const double plotCtMax = ctEdges[ict + 1] + ctPad;
            out.denom2D[static_cast<size_t>(ir)].push_back(
                MakeBinH2(Form("h2_den_%s_rad_%d_ct_%zu", input.label.c_str(), ir, ict),
                          "denominator",
                          plotRadMin, plotRadMax, plotCtMin, plotCtMax));
            out.numer2D[static_cast<size_t>(ir)].push_back(
                MakeBinH2(Form("h2_num_%s_rad_%d_ct_%zu", input.label.c_str(), ir, ict),
                          "numerator",
                          plotRadMin, plotRadMax, plotCtMin, plotCtMax));
        }
    }

    TChain chain(treeName.c_str());
    for (const auto &path : input.paths) AddAODFileToChain(chain, path);
    if (chain.GetNtrees() <= 0) {
        std::cerr << "[Warn] no trees for " << input.label << "\n";
        return out;
    }

    ROOT::RDataFrame rdf(chain);
    auto ready = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(ready,
                                                              radBins,
                                                              {},
                                                              ctBinsByRad,
                                                              {},
                                                              {},
                                                              "fDecRad > 0.8",
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
    const auto &accHists = accRes.acc_ct_per_pt;
    for (size_t ir = 0; ir < out.effVsCt.size() && ir < accHists.size(); ++ir) {
        if (!accHists[ir]) continue;
        out.effVsCt[ir].reset(static_cast<TH1D *>(accHists[ir]->Clone(
            Form("h_final_eff_%s_rad_%zu", input.label.c_str(), ir))));
        StyleEff(out.effVsCt[ir].get(), input.color, input.marker);
    }
    accRes.Clear();

    ROOT::RDF::RNode node(ready);
    node = DefineIntFlag(node, "fIsReco", "__is_reco_int");
    node = DefineIntFlag(node, "fIsSurvEvSel", "__evsel_int");
    node = DefineIntFlag(node, "fIsRecoMCCollision", "__reco_mc_collision_int");
    node = DefineIntFlag(node, "fIsTwoBodyDecay", "__two_body_int");
    node = DefineDoubleAlias(node, "fGenDecRad", "__gen_rad_double");
    node = DefineDoubleAlias(node, "fGenCt", "__gen_ct_double");
    node = DefineDoubleAlias(node, "fGenPt", "__gen_pt_double");
    node = DefineDoubleAlias(node, "fDecRad", "__reco_rad_double");

    std::cout << "[BuildGroupRadCtBinQA] " << input.label
              << ": entries=" << chain.GetEntries()
              << ", trees=" << chain.GetNtrees() << "\n";

    node.Foreach(
        [&](double genRad, double genCt, double genPt, int isReco, double recoRad, int evSel, int recoMCCollision, int isTwoBody) {
            if (requireTwoBody && !isTwoBody) return;
            const bool passDenNoRad = evSel && recoMCCollision;
            const bool passNumNoRad = isReco;
            if (passDenNoRad && out.denomNoRadCutFineCt) out.denomNoRadCutFineCt->Fill(genCt);
            if (passNumNoRad && out.numerNoRadCutFineCt) out.numerNoRadCutFineCt->Fill(genCt);
            if (passDenNoRad) {
                for (size_t jrad = 0; jrad < ctBinsByRad.size() && jrad < out.denomRadReleasedCountsVsCt.size(); ++jrad) {
                    if (FindBin(ctBinsByRad[jrad], genCt) >= 0) {
                        out.denomRadReleasedCountsVsCt[jrad]->Fill(genCt);
                    }
                }
            }
            const int ir = FindBin(radBins, genRad);
            if (ir < 0) return;
            const auto &ctEdges = ctBinsByRad[static_cast<size_t>(ir)];
            const int ict = FindBin(ctEdges, genCt);
            if (ict < 0) return;
            const bool passDen = evSel && recoMCCollision;
            const bool passNum = isReco && recoRad > 0.8;
            if (passDen) {
                out.denomCountsVsCt[static_cast<size_t>(ir)]->Fill(genCt);
                if (static_cast<size_t>(ir) < out.denomPtVsRad.size()) out.denomPtVsRad[static_cast<size_t>(ir)]->Fill(genPt);
                out.denom2D[static_cast<size_t>(ir)][static_cast<size_t>(ict)]->Fill(genRad, genCt);
            }
            if (passNum) {
                out.numerCountsVsCt[static_cast<size_t>(ir)]->Fill(genCt);
                if (static_cast<size_t>(ir) < out.numerPtVsRad.size()) out.numerPtVsRad[static_cast<size_t>(ir)]->Fill(genPt);
                out.numer2D[static_cast<size_t>(ir)][static_cast<size_t>(ict)]->Fill(genRad, genCt);
            }
        },
        {"__gen_rad_double", "__gen_ct_double", "__gen_pt_double", "__is_reco_int", "__reco_rad_double",
         "__evsel_int", "__reco_mc_collision_int", "__two_body_int"});
    for (size_t ir = 0; ir < out.effVsCtDenRadReleased.size(); ++ir) {
        if (!out.effVsCtDenRadReleased[ir] || ir >= out.numerCountsVsCt.size() || ir >= out.denomRadReleasedCountsVsCt.size()) continue;
        out.effVsCtDenRadReleased[ir]->Divide(out.numerCountsVsCt[ir].get(),
                                              out.denomRadReleasedCountsVsCt[ir].get(),
                                              1.0,
                                              1.0,
                                              "B");
    }
    return out;
}

std::unique_ptr<TGraphAsymmErrors> MakeOffsetGraph(const TH1D *h,
                                                   const std::string &name,
                                                   Color_t color,
                                                   Style_t marker,
                                                   double offsetFraction)
{
    if (!h) return nullptr;
    auto g = std::make_unique<TGraphAsymmErrors>(h->GetNbinsX());
    g->SetName(name.c_str());
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double width = h->GetBinWidth(ib);
        const double x = h->GetBinCenter(ib) + offsetFraction * width;
        const int ip = ib - 1;
        g->SetPoint(ip, x, h->GetBinContent(ib));
        g->SetPointError(ip, 0.5 * width, 0.5 * width, h->GetBinError(ib), h->GetBinError(ib));
    }
    g->SetLineColor(color);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.12);
    g->SetLineWidth(3);
    return g;
}

void DrawEfficiencyVsCtForRad(const std::vector<GroupQA> &groups,
                              const std::vector<double> &radBins,
                              size_t ir,
                              const std::string &outPdf)
{
    TCanvas c(Form("c_eff_vs_ct_rad_%zu", ir), "", 950, 740);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);

    double ymax = 0.;
    for (const auto &g : groups) {
        if (ir < g.effVsCt.size() && g.effVsCt[ir]) ymax = std::max(ymax, g.effVsCt[ir]->GetMaximum());
    }
    auto *frame = groups.front().effVsCt[ir] ? static_cast<TH1D *>(groups.front().effVsCt[ir]->Clone("h_frame_eff")) : nullptr;
    if (!frame) return;
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.);
    frame->SetMaximum(std::max(0.04, 1.35 * ymax));
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->Draw("AXIS");

    TLegend leg(0.56, 0.70, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.035);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const auto &g = groups[ig];
        if (ir >= g.effVsCt.size() || !g.effVsCt[ir]) continue;
        const double offset = (groups.size() == 2) ? (ig == 0 ? -0.055 : 0.055) : 0.;
        auto graph = MakeOffsetGraph(g.effVsCt[ir].get(),
                                     Form("g_eff_%zu_rad_%zu", ig, ir),
                                     g.color,
                                     g.marker,
                                     offset);
        if (!graph) continue;
        graph->Draw("PZ SAME");
        leg.AddEntry(graph.get(), g.label.c_str(), "pe");
        graphs.push_back(std::move(graph));
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.DrawLatex(0.16, 0.78, "Final MC efficiency: numerator / denominator");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

void DrawEfficiencyVsCtDenRadReleasedForRad(const std::vector<GroupQA> &groups,
                                            const std::vector<double> &radBins,
                                            size_t ir,
                                            const std::string &outPdf)
{
    TCanvas c(Form("c_eff_vs_ct_den_rad_released_rad_%zu", ir), "", 950, 740);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);

    double ymax = 0.;
    for (const auto &g : groups) {
        if (ir < g.effVsCtDenRadReleased.size() && g.effVsCtDenRadReleased[ir]) {
            ymax = std::max(ymax, g.effVsCtDenRadReleased[ir]->GetMaximum());
        }
    }
    auto *frame = (groups.front().effVsCtDenRadReleased.size() > ir && groups.front().effVsCtDenRadReleased[ir])
                      ? static_cast<TH1D *>(groups.front().effVsCtDenRadReleased[ir]->Clone("h_frame_eff_den_rad_released"))
                      : nullptr;
    if (!frame) return;
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.);
    frame->SetMaximum(std::max(0.04, 1.35 * ymax));
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->GetYaxis()->SetTitle("Final MC efficiency");
    frame->Draw("AXIS");

    TLegend leg(groups.size() > 2 ? 0.42 : 0.50, groups.size() > 2 ? 0.58 : 0.68, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(groups.size() > 2 ? 0.025 : 0.033);
    if (groups.size() > 2) leg.SetNColumns(2);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const auto &g = groups[ig];
        if (ir >= g.effVsCtDenRadReleased.size() || !g.effVsCtDenRadReleased[ir]) continue;
        const double offset = (groups.size() == 2)
                                  ? (ig == 0 ? -0.055 : 0.055)
                                  : (static_cast<double>(ig) - 0.5 * static_cast<double>(groups.size() - 1)) * 0.022;
        auto graph = MakeOffsetGraph(g.effVsCtDenRadReleased[ir].get(),
                                     Form("g_eff_den_rad_released_%zu_rad_%zu", ig, ir),
                                     g.color,
                                     g.marker,
                                     offset);
        if (!graph) continue;
        graph->Draw("PZ SAME");
        leg.AddEntry(graph.get(), g.label.c_str(), "pe");
        graphs.push_back(std::move(graph));
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, Form("Numerator: %.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.DrawLatex(0.16, 0.78, "Denominator radius released, same #it{c}t binning");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

std::unique_ptr<TGraphAsymmErrors> MakeCountsGraph(const TH1D *h,
                                                   const std::string &name,
                                                   Color_t color,
                                                   Style_t marker)
{
    if (!h) return nullptr;
    auto g = std::make_unique<TGraphAsymmErrors>(h->GetNbinsX());
    g->SetName(name.c_str());
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const int ip = ib - 1;
        const double y = h->GetBinContent(ib);
        const double ey = h->GetBinError(ib) > 0. ? h->GetBinError(ib) : std::sqrt(std::max(0., y));
        g->SetPoint(ip, h->GetBinCenter(ib), y);
        g->SetPointError(ip, 0.5 * h->GetBinWidth(ib), 0.5 * h->GetBinWidth(ib), ey, ey);
    }
    g->SetLineColor(color);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.05);
    g->SetLineWidth(2);
    return g;
}

std::unique_ptr<TH1D> MakeBinWidthScaledCounts(const TH1D *h,
                                               const std::string &name,
                                               Color_t color,
                                               Style_t marker,
                                               Style_t lineStyle)
{
    if (!h) return nullptr;
    auto out = std::unique_ptr<TH1D>(static_cast<TH1D *>(h->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->Scale(1.0, "width");
    StyleCounts(out.get(), color, marker, lineStyle);
    out->GetYaxis()->SetTitle("Counts / cm");
    return out;
}

void DrawCountsVsCtForRad(const std::vector<GroupQA> &groups,
                          const std::vector<double> &radBins,
                          size_t ir,
                          const std::string &outPdf)
{
    if (groups.empty() || ir >= groups.front().denomCountsVsCt.size() || !groups.front().denomCountsVsCt[ir]) return;
    TCanvas c(Form("c_counts_vs_ct_rad_%zu", ir), "", 980, 760);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);
    c.SetLogy();

    double ymax = 0.;
    std::vector<std::unique_ptr<TH1D>> scaledHists;
    for (const auto &g : groups) {
        if (ir < g.denomCountsVsCt.size() && g.denomCountsVsCt[ir]) {
            auto h = MakeBinWidthScaledCounts(g.denomCountsVsCt[ir].get(),
                                              Form("h_den_counts_width_scaled_%s_rad_%zu", g.label.c_str(), ir),
                                              g.color,
                                              g.marker,
                                              1);
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
        if (ir < g.numerCountsVsCt.size() && g.numerCountsVsCt[ir]) {
            auto h = MakeBinWidthScaledCounts(g.numerCountsVsCt[ir].get(),
                                              Form("h_num_counts_width_scaled_%s_rad_%zu", g.label.c_str(), ir),
                                              g.color,
                                              g.marker,
                                              2);
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
    }
    if (scaledHists.empty()) return;

    auto *frame = static_cast<TH1D *>(scaledHists.front()->Clone("h_frame_counts"));
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.5);
    frame->SetMaximum(std::max(10., 6.0 * ymax));
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->Draw("AXIS");

    TLegend leg(0.50, 0.64, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.030);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    size_t ih = 0;
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const auto &g = groups[ig];
        if (ir < g.denomCountsVsCt.size() && g.denomCountsVsCt[ir]) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_den_counts_%zu_rad_%zu", ig, ir),
                                         g.color,
                                         g.marker);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " denominator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
        if (ir < g.numerCountsVsCt.size() && g.numerCountsVsCt[ir]) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_num_counts_%zu_rad_%zu", ig, ir),
                                         g.color,
                                         ig == 0 ? kOpenCircle : kOpenSquare);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " numerator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.DrawLatex(0.16, 0.78, "MC numerator and denominator counts");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

void DrawPtVsRadForRad(const std::vector<GroupQA> &groups,
                       const std::vector<double> &radBins,
                       size_t ir,
                       const std::string &outPdf)
{
    if (groups.empty() || ir >= groups.front().denomPtVsRad.size() || !groups.front().denomPtVsRad[ir]) return;
    TCanvas c(Form("c_pt_vs_rad_%zu", ir), "", 980, 760);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);
    c.SetLogy();

    double ymax = 0.;
    std::vector<std::unique_ptr<TH1D>> scaledHists;
    for (const auto &g : groups) {
        if (ir < g.denomPtVsRad.size() && g.denomPtVsRad[ir]) {
            auto h = MakeBinWidthScaledCounts(g.denomPtVsRad[ir].get(),
                                              Form("h_den_pt_width_scaled_%s_rad_%zu", g.label.c_str(), ir),
                                              g.color,
                                              g.marker,
                                              1);
            h->GetXaxis()->SetTitle("#it{p}_{T}^{gen} (GeV/#it{c})");
            h->GetYaxis()->SetTitle("Counts / (GeV/#it{c})");
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
        if (ir < g.numerPtVsRad.size() && g.numerPtVsRad[ir]) {
            auto h = MakeBinWidthScaledCounts(g.numerPtVsRad[ir].get(),
                                              Form("h_num_pt_width_scaled_%s_rad_%zu", g.label.c_str(), ir),
                                              g.color,
                                              g.marker,
                                              2);
            h->GetXaxis()->SetTitle("#it{p}_{T}^{gen} (GeV/#it{c})");
            h->GetYaxis()->SetTitle("Counts / (GeV/#it{c})");
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
    }
    if (scaledHists.empty()) return;

    auto *frame = static_cast<TH1D *>(scaledHists.front()->Clone("h_frame_pt_counts"));
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.5);
    frame->SetMaximum(std::max(10., 6.0 * ymax));
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->Draw("AXIS");

    TLegend leg(0.48, 0.62, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.030);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    size_t ih = 0;
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const auto &g = groups[ig];
        if (ir < g.denomPtVsRad.size() && g.denomPtVsRad[ir]) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_den_pt_%zu_rad_%zu", ig, ir),
                                         g.color,
                                         g.marker);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " denominator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
        if (ir < g.numerPtVsRad.size() && g.numerPtVsRad[ir]) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_num_pt_%zu_rad_%zu", ig, ir),
                                         g.color,
                                         ig == 0 ? kOpenCircle : kOpenSquare);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " numerator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm", radBins[ir], radBins[ir + 1]));
    text.DrawLatex(0.16, 0.78, "MC numerator and denominator counts / bin width");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

void DrawNoRadCutFineCtCounts(const std::vector<GroupQA> &groups,
                              const std::string &outPdf)
{
    if (groups.empty()) return;
    TCanvas c("c_no_rad_cut_fine_ct_counts", "", 980, 760);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetTicks(1, 1);
    c.SetLogy();

    double ymax = 0.;
    std::vector<std::unique_ptr<TH1D>> scaledHists;
    for (const auto &g : groups) {
        if (g.denomNoRadCutFineCt) {
            auto h = MakeBinWidthScaledCounts(g.denomNoRadCutFineCt.get(),
                                              Form("h_den_no_rad_cut_width_scaled_%s", g.label.c_str()),
                                              g.color,
                                              g.marker,
                                              1);
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
        if (g.numerNoRadCutFineCt) {
            auto h = MakeBinWidthScaledCounts(g.numerNoRadCutFineCt.get(),
                                              Form("h_num_no_rad_cut_width_scaled_%s", g.label.c_str()),
                                              g.color,
                                              g.marker,
                                              2);
            ymax = std::max(ymax, h->GetMaximum());
            scaledHists.push_back(std::move(h));
        }
    }
    if (scaledHists.empty()) return;

    auto *frame = static_cast<TH1D *>(scaledHists.front()->Clone("h_frame_no_rad_cut_counts"));
    frame->Reset("ICES");
    frame->SetStats(false);
    frame->SetMinimum(0.5);
    frame->SetMaximum(std::max(10., 6.0 * ymax));
    frame->GetYaxis()->SetTitleOffset(1.15);
    frame->Draw("AXIS");

    TLegend leg(0.48, 0.62, 0.91, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.030);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphs;
    size_t ih = 0;
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const auto &g = groups[ig];
        if (g.denomNoRadCutFineCt) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_den_no_rad_cut_%zu", ig),
                                         g.color,
                                         g.marker);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " denominator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
        if (g.numerNoRadCutFineCt) {
            TH1D *h = scaledHists[ih++].get();
            h->Draw("HIST SAME");
            auto graph = MakeCountsGraph(h,
                                         Form("g_num_no_rad_cut_%zu", ig),
                                         g.color,
                                         ig == 0 ? kOpenCircle : kOpenSquare);
            graph->Draw("PZ SAME");
            leg.AddEntry(graph.get(), (g.label + " numerator").c_str(), "pe");
            graphs.push_back(std::move(graph));
        }
    }
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.035);
    text.DrawLatex(0.16, 0.88, "ALICE Work In Progress");
    text.SetTextSize(0.031);
    text.DrawLatex(0.16, 0.83, "No radius cut, fine #it{c}t binning");
    text.DrawLatex(0.16, 0.78, "MC numerator and denominator counts / bin width");
    c.SaveAs(outPdf.c_str());
    delete frame;
}

void DrawOneBinSubplot(const std::vector<GroupQA> &groups,
                       const std::vector<double> &radBins,
                       const std::vector<std::vector<double>> &ctBinsByRad,
                       size_t ir,
                       size_t ict,
                       const std::string &outPdf)
{
    TCanvas c(Form("c_bin_rad_%zu_ct_%zu", ir, ict), "", 1150, 920);
    c.Divide(2, 2, 0.006, 0.006);
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBird);

    const char *rowNames[2] = {"no-G4list", "G4list"};
    const char *colNames[2] = {"denominator", "numerator"};
    const auto &ctEdges = ctBinsByRad[ir];
    for (int ig = 0; ig < 2; ++ig) {
        if (static_cast<size_t>(ig) >= groups.size()) continue;
        for (int in = 0; in < 2; ++in) {
            const int padIdx = ig * 2 + in + 1;
            c.cd(padIdx);
            gPad->SetLeftMargin(0.12);
            gPad->SetRightMargin(0.15);
            gPad->SetBottomMargin(0.12);
            gPad->SetTopMargin(0.12);
            gPad->SetLogz();
            TH2D *h = nullptr;
            if (in == 0 && ir < groups[ig].denom2D.size() && ict < groups[ig].denom2D[ir].size()) {
                h = groups[ig].denom2D[ir][ict].get();
            }
            if (in == 1 && ir < groups[ig].numer2D.size() && ict < groups[ig].numer2D[ir].size()) {
                h = groups[ig].numer2D[ir][ict].get();
            }
            if (!h) continue;
            h->SetStats(false);
            h->SetMinimum(0.5);
            h->SetTitle(Form("%s %s;#it{R}_{dec}^{gen} (cm);#it{c}t_{gen} (cm)",
                             rowNames[ig], colNames[in]));
            h->GetXaxis()->SetTitleSize(0.045);
            h->GetYaxis()->SetTitleSize(0.045);
            h->GetXaxis()->SetLabelSize(0.038);
            h->GetYaxis()->SetLabelSize(0.038);
            h->Draw("COLZ");
            TBox selected(radBins[ir], ctEdges[ict], radBins[ir + 1], ctEdges[ict + 1]);
            selected.SetFillStyle(0);
            selected.SetLineColor(kRed + 1);
            selected.SetLineWidth(3);
            selected.Draw("same");
            TLatex text;
            text.SetNDC();
            text.SetTextSize(0.045);
            text.DrawLatex(0.15, 0.92, Form("N = %.0f", h->Integral()));
        }
    }
    c.cd();
    TLatex title;
    title.SetNDC();
    title.SetTextAlign(22);
    title.SetTextSize(0.030);
    title.DrawLatex(0.50, 0.985,
                    Form("%.1f < #it{R}_{dec}^{gen} < %.1f cm, %.2g < #it{c}t_{gen} < %.2g cm",
                         radBins[ir], radBins[ir + 1], ctEdges[ict], ctEdges[ict + 1]));
    c.SaveAs(outPdf.c_str());
}

} // namespace

void DrawRadCtMCEffFinalEfficiencyAndBinQA(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config_merged.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtMCEffFinalEfficiencyAndBinQA",
    bool usePtWeight = true)
{
    const std::filesystem::path outBase(outputDir);
    const std::filesystem::path effDir = outBase / "efficiency_vs_ct";
    const std::filesystem::path effDenRadReleasedDir = outBase / "efficiency_vs_ct_den_rad_released";
    const std::filesystem::path effDenRadReleasedPeriodDir = outBase / "efficiency_vs_ct_den_rad_released_by_period";
    const std::filesystem::path countsDir = outBase / "counts_vs_ct";
    const std::filesystem::path ptDir = outBase / "pt_counts_per_rad";
    const std::filesystem::path noRadCutDir = outBase / "no_radius_cut_fine_ct_counts";
    const std::filesystem::path binDir = outBase / "rad_ct_bin_num_den_subplots";
    std::filesystem::remove_all(effDir);
    std::filesystem::remove_all(effDenRadReleasedDir);
    std::filesystem::remove_all(effDenRadReleasedPeriodDir);
    std::filesystem::remove_all(countsDir);
    std::filesystem::remove_all(ptDir);
    std::filesystem::remove_all(noRadCutDir);
    std::filesystem::remove_all(binDir);
    std::filesystem::create_directories(outputDir);
    std::filesystem::create_directories(effDir);
    std::filesystem::create_directories(effDenRadReleasedDir);
    std::filesystem::create_directories(effDenRadReleasedPeriodDir);
    std::filesystem::create_directories(countsDir);
    std::filesystem::create_directories(ptDir);
    std::filesystem::create_directories(noRadCutDir);
    std::filesystem::create_directories(binDir);

    const auto cfg = GeneralHelper::LoadJsonFile(configPath);
    const auto common = cfg.at("common");
    const auto binning = common.at("binning");
    auto radBins = binning.at("rad_bins").get<std::vector<double>>();
    auto ctBinsByRad = binning.at("ct_bins_by_rad").get<std::vector<std::vector<double>>>();
    auto fineCtBins = binning.value("ct_bins_single", std::vector<double>{});
    auto ptBins = binning.value("pt_bins_single", std::vector<double>{});
    LoadFullRadCtBinningIfNeeded(configPath, radBins, ctBinsByRad);
    const std::string treeName = common.value("tree_names", GeneralHelper::Json::object()).value("mc", std::string("O2mchypcands"));
    const bool requireTwoBody = common.value("selection", GeneralHelper::Json::object()).value("mc_acceptance_require_two_body", true);
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto modeProfiles = analysis.value("mode_profiles", GeneralHelper::Json::object());
    const auto radCtProfile = modeProfiles.value("rad_ct", GeneralHelper::Json::object());
    const bool constrainDenominatorOuterBin =
        radCtProfile.value("mc_acceptance_outer_bin_cut_denominator",
                           common.value("selection", GeneralHelper::Json::object())
                               .value("mc_acceptance_outer_bin_cut_denominator", true));

    std::vector<std::string> configMcPaths;
    for (const auto &p : common.at("periods")) configMcPaths.push_back(p.value("mc_path", std::string()));
    std::vector<std::string> noG4Paths = BuildNoG4PathsFromConfig(configMcPaths);
    if (!usePtWeight) noG4Paths = BuildUnweightedPaths(noG4Paths);
    std::vector<std::string> g4Paths = BuildG4PathsFromNoG4(noG4Paths);
    if (!usePtWeight) g4Paths = BuildUnweightedPaths(g4Paths);
    std::cout << "[Info] use pT weight: " << (usePtWeight ? "yes" : "no") << "\n";
    std::cout << "[Info] no-G4list paths:\n";
    for (const auto &p : noG4Paths) std::cout << "  " << p << "\n";
    std::cout << "[Info] G4list paths:\n";
    for (const auto &p : g4Paths) std::cout << "  " << p << "\n";
    std::cout << "[Info] rad bins to draw:";
    for (double r : radBins) std::cout << " " << r;
    std::cout << "\n";
    std::vector<GroupInput> inputs = {
        {usePtWeight ? "no-G4list merged" : "no-G4list merged no pT weight", noG4Paths, kAzure + 2, kFullCircle},
        {usePtWeight ? "G4list merged" : "G4list merged no pT weight", g4Paths, kRed + 1, kFullSquare}
    };

    std::vector<GroupQA> groups;
    for (const auto &input : inputs) {
        groups.push_back(BuildGroup(input,
                                    radBins,
                                    ctBinsByRad,
                                    fineCtBins,
                                    ptBins,
                                    treeName,
                                    requireTwoBody,
                                    constrainDenominatorOuterBin));
    }

    DrawNoRadCutFineCtCounts(groups, (noRadCutDir / "no_radius_cut_fine_ct_counts_width_scaled.pdf").string());

    for (size_t ir = 0; ir + 1 < radBins.size(); ++ir) {
        const std::string effPdf = (effDir / ("efficiency_vs_ct_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string();
        DrawEfficiencyVsCtForRad(groups, radBins, ir, effPdf);
        const std::string effReleasedPdf =
            (effDenRadReleasedDir / ("efficiency_vs_ct_den_rad_released_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string();
        DrawEfficiencyVsCtDenRadReleasedForRad(groups, radBins, ir, effReleasedPdf);
        const std::string countsPdf = (countsDir / ("counts_vs_ct_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string();
        DrawCountsVsCtForRad(groups, radBins, ir, countsPdf);
        const std::string ptPdf = (ptDir / ("pt_counts_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string();
        DrawPtVsRadForRad(groups, radBins, ir, ptPdf);
        for (size_t ict = 0; ict + 1 < ctBinsByRad[ir].size(); ++ict) {
            const std::string binPdf =
                (binDir / ("num_den_subplots_" +
                           RadCtTag(radBins[ir], radBins[ir + 1],
                                    ctBinsByRad[ir][ict], ctBinsByRad[ir][ict + 1]) +
                           ".pdf")).string();
            DrawOneBinSubplot(groups, radBins, ctBinsByRad, ir, ict, binPdf);
        }
    }

    TFile out((outBase / "rad_ct_final_efficiency_and_bin_qa.root").string().c_str(), "RECREATE");
    for (auto &g : groups) {
        out.mkdir(g.label.c_str());
        out.cd(g.label.c_str());
        if (g.denomNoRadCutFineCt) g.denomNoRadCutFineCt->Write("h_den_no_radius_cut_fine_ct");
        if (g.numerNoRadCutFineCt) g.numerNoRadCutFineCt->Write("h_num_no_radius_cut_fine_ct");
        for (size_t ir = 0; ir < g.effVsCt.size(); ++ir) {
            if (g.effVsCt[ir]) g.effVsCt[ir]->Write(Form("h_eff_vs_ct_rad_%zu", ir));
            if (ir < g.effVsCtDenRadReleased.size() && g.effVsCtDenRadReleased[ir]) {
                g.effVsCtDenRadReleased[ir]->Write(Form("h_eff_vs_ct_den_rad_released_rad_%zu", ir));
            }
            if (ir < g.denomCountsVsCt.size() && g.denomCountsVsCt[ir]) {
                g.denomCountsVsCt[ir]->Write(Form("h_den_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.denomRadReleasedCountsVsCt.size() && g.denomRadReleasedCountsVsCt[ir]) {
                g.denomRadReleasedCountsVsCt[ir]->Write(Form("h_den_rad_released_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.numerCountsVsCt.size() && g.numerCountsVsCt[ir]) {
                g.numerCountsVsCt[ir]->Write(Form("h_num_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.denomPtVsRad.size() && g.denomPtVsRad[ir]) {
                g.denomPtVsRad[ir]->Write(Form("h_den_pt_rad_%zu", ir));
            }
            if (ir < g.numerPtVsRad.size() && g.numerPtVsRad[ir]) {
                g.numerPtVsRad[ir]->Write(Form("h_num_pt_rad_%zu", ir));
            }
            for (size_t ict = 0; ict < g.denom2D[ir].size(); ++ict) {
                if (g.denom2D[ir][ict]) g.denom2D[ir][ict]->Write(Form("h2_den_rad_%zu_ct_%zu", ir, ict));
                if (g.numer2D[ir][ict]) g.numer2D[ir][ict]->Write(Form("h2_num_rad_%zu_ct_%zu", ir, ict));
            }
        }
        out.cd();
    }
    out.Close();

    std::cout << "[DrawRadCtMCEffFinalEfficiencyAndBinQA] wrote outputs to " << outputDir << "\n";
}

void DrawRadCtMCEffFinalEfficiencyAndBinQA_DirectOriginalMC(
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/RadCtMCEffFinalEfficiencyAndBinQA",
    bool usePtWeight = true)
{
    const std::filesystem::path outBase(outputDir);
    const std::filesystem::path effDir = outBase / "efficiency_vs_ct";
    const std::filesystem::path effDenRadReleasedDir = outBase / "efficiency_vs_ct_den_rad_released";
    const std::filesystem::path effDenRadReleasedPeriodDir = outBase / "efficiency_vs_ct_den_rad_released_by_period";
    const std::filesystem::path countsDir = outBase / "counts_vs_ct";
    const std::filesystem::path ptDir = outBase / "pt_counts_per_rad";
    const std::filesystem::path noRadCutDir = outBase / "no_radius_cut_fine_ct_counts";
    const std::filesystem::path binDir = outBase / "rad_ct_bin_num_den_subplots";
    std::filesystem::remove_all(effDir);
    std::filesystem::remove_all(effDenRadReleasedDir);
    std::filesystem::remove_all(effDenRadReleasedPeriodDir);
    std::filesystem::remove_all(countsDir);
    std::filesystem::remove_all(ptDir);
    std::filesystem::remove_all(noRadCutDir);
    std::filesystem::remove_all(binDir);
    std::filesystem::create_directories(outputDir);
    std::filesystem::create_directories(effDir);
    std::filesystem::create_directories(effDenRadReleasedDir);
    std::filesystem::create_directories(effDenRadReleasedPeriodDir);
    std::filesystem::create_directories(countsDir);
    std::filesystem::create_directories(ptDir);
    std::filesystem::create_directories(noRadCutDir);
    std::filesystem::create_directories(binDir);

    const std::vector<double> radBins{0.8, 1.1, 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
                                      8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0};
    const std::vector<std::vector<double>> ctBinsByRad{
        {0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5},
        {0.5, 0.8, 1.0, 1.2, 1.3, 1.6, 1.8, 2.1},
        {0.5, 1.0, 1.5, 1.7, 2.0, 2.5},
        {1.0, 1.5, 2.0, 2.2, 2.5, 2.9, 3.5, 4.5},
        {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0},
        {1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0},
        {2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.5},
        {2.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0},
        {3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0},
        {5.0, 7.0, 9.0, 11.0, 12.0, 13.0, 15.0},
        {5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0},
        {7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 25.0},
        {10.0, 13.0, 17.0, 20.0, 23.0, 30.0},
        {10.0, 15.0, 18.0, 21.0, 26.0, 33.0},
        {15.0, 17.0, 20.0, 23.0, 27.0, 35.0},
        {15.0, 20.0, 25.0, 30.0, 40.0}
    };
    const std::vector<double> fineCtBins{1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 27, 33};
    const std::vector<double> ptBins{2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8};
    const std::string treeName = "O2mchypcands";
    const bool requireTwoBody = true;
    const bool constrainDenominatorOuterBin = true;

    std::vector<std::string> noG4Paths = {
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5/reweighted/AO2D_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6/reweighted/AO2D_combined_reweighted.root"
    };
    std::vector<std::string> g4Paths = {
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5_G4list/reweighted/AO2D_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6_G4list/reweighted/AO2D_combined_reweighted.root"
    };
    const std::vector<std::string> periodLabels{"LHC23", "LHC24ar", "LHC25"};
    if (!usePtWeight) {
        noG4Paths = BuildUnweightedPaths(noG4Paths);
        g4Paths = BuildUnweightedPaths(g4Paths);
    }

    std::cout << "[Info] Direct original MC paths, use pT weight: " << (usePtWeight ? "yes" : "no") << "\n";
    std::cout << "[Info] no-G4list paths:\n";
    for (const auto &p : noG4Paths) std::cout << "  " << p << "\n";
    std::cout << "[Info] G4list paths:\n";
    for (const auto &p : g4Paths) std::cout << "  " << p << "\n";

    std::vector<GroupInput> inputs = {
        {usePtWeight ? "no-G4list merged" : "no-G4list merged no pT weight", noG4Paths, kAzure + 2, kFullCircle},
        {usePtWeight ? "G4list merged" : "G4list merged no pT weight", g4Paths, kRed + 1, kFullSquare}
    };

    std::vector<GroupQA> groups;
    for (const auto &input : inputs) {
        groups.push_back(BuildGroup(input,
                                    radBins,
                                    ctBinsByRad,
                                    fineCtBins,
                                    ptBins,
                                    treeName,
                                    requireTwoBody,
                                    constrainDenominatorOuterBin));
    }

    std::vector<GroupInput> periodInputs;
    const std::vector<Color_t> noG4PeriodColors{kAzure + 2, kTeal + 3, kViolet + 1};
    const std::vector<Color_t> g4PeriodColors{kRed + 1, kOrange + 7, kMagenta + 1};
    for (size_t ip = 0; ip < periodLabels.size() && ip < noG4Paths.size() && ip < g4Paths.size(); ++ip) {
        periodInputs.push_back({periodLabels[ip] + " no-G4list", {noG4Paths[ip]}, noG4PeriodColors[ip], kFullCircle});
        periodInputs.push_back({periodLabels[ip] + " G4list", {g4Paths[ip]}, g4PeriodColors[ip], kFullSquare});
    }

    std::vector<GroupQA> periodGroups;
    for (const auto &input : periodInputs) {
        periodGroups.push_back(BuildGroup(input,
                                          radBins,
                                          ctBinsByRad,
                                          fineCtBins,
                                          ptBins,
                                          treeName,
                                          requireTwoBody,
                                          constrainDenominatorOuterBin));
    }

    DrawNoRadCutFineCtCounts(groups, (noRadCutDir / "no_radius_cut_fine_ct_counts_width_scaled.pdf").string());
    for (size_t ir = 0; ir + 1 < radBins.size(); ++ir) {
        DrawEfficiencyVsCtForRad(groups,
                                 radBins,
                                 ir,
                                 (effDir / ("efficiency_vs_ct_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string());
        DrawEfficiencyVsCtDenRadReleasedForRad(
            groups,
            radBins,
            ir,
            (effDenRadReleasedDir / ("efficiency_vs_ct_den_rad_released_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string());
        DrawEfficiencyVsCtDenRadReleasedForRad(
            periodGroups,
            radBins,
            ir,
            (effDenRadReleasedPeriodDir / ("efficiency_vs_ct_den_rad_released_by_period_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string());
        DrawCountsVsCtForRad(groups,
                             radBins,
                             ir,
                             (countsDir / ("counts_vs_ct_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string());
        DrawPtVsRadForRad(groups,
                          radBins,
                          ir,
                          (ptDir / ("pt_counts_" + RadTag(radBins[ir], radBins[ir + 1]) + ".pdf")).string());
        for (size_t ict = 0; ict + 1 < ctBinsByRad[ir].size(); ++ict) {
            DrawOneBinSubplot(groups,
                              radBins,
                              ctBinsByRad,
                              ir,
                              ict,
                              (binDir / ("num_den_subplots_" +
                                         RadCtTag(radBins[ir], radBins[ir + 1],
                                                  ctBinsByRad[ir][ict], ctBinsByRad[ir][ict + 1]) +
                                         ".pdf"))
                                  .string());
        }
    }

    TFile out((outBase / "rad_ct_final_efficiency_and_bin_qa.root").string().c_str(), "RECREATE");
    for (auto &g : groups) {
        out.mkdir(g.label.c_str());
        out.cd(g.label.c_str());
        if (g.denomNoRadCutFineCt) g.denomNoRadCutFineCt->Write("h_den_no_radius_cut_fine_ct");
        if (g.numerNoRadCutFineCt) g.numerNoRadCutFineCt->Write("h_num_no_radius_cut_fine_ct");
        for (size_t ir = 0; ir < g.effVsCt.size(); ++ir) {
            if (g.effVsCt[ir]) g.effVsCt[ir]->Write(Form("h_eff_vs_ct_rad_%zu", ir));
            if (ir < g.effVsCtDenRadReleased.size() && g.effVsCtDenRadReleased[ir]) {
                g.effVsCtDenRadReleased[ir]->Write(Form("h_eff_vs_ct_den_rad_released_rad_%zu", ir));
            }
            if (ir < g.denomCountsVsCt.size() && g.denomCountsVsCt[ir]) {
                g.denomCountsVsCt[ir]->Write(Form("h_den_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.denomRadReleasedCountsVsCt.size() && g.denomRadReleasedCountsVsCt[ir]) {
                g.denomRadReleasedCountsVsCt[ir]->Write(Form("h_den_rad_released_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.numerCountsVsCt.size() && g.numerCountsVsCt[ir]) {
                g.numerCountsVsCt[ir]->Write(Form("h_num_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.denomPtVsRad.size() && g.denomPtVsRad[ir]) {
                g.denomPtVsRad[ir]->Write(Form("h_den_pt_rad_%zu", ir));
            }
            if (ir < g.numerPtVsRad.size() && g.numerPtVsRad[ir]) {
                g.numerPtVsRad[ir]->Write(Form("h_num_pt_rad_%zu", ir));
            }
            for (size_t ict = 0; ict < g.denom2D[ir].size(); ++ict) {
                if (g.denom2D[ir][ict]) g.denom2D[ir][ict]->Write(Form("h2_den_rad_%zu_ct_%zu", ir, ict));
                if (g.numer2D[ir][ict]) g.numer2D[ir][ict]->Write(Form("h2_num_rad_%zu_ct_%zu", ir, ict));
            }
        }
        out.cd();
    }
    out.mkdir("by_period_den_rad_released");
    out.cd("by_period_den_rad_released");
    for (auto &g : periodGroups) {
        gDirectory->mkdir(g.label.c_str());
        gDirectory->cd(g.label.c_str());
        for (size_t ir = 0; ir < g.effVsCtDenRadReleased.size(); ++ir) {
            if (g.effVsCtDenRadReleased[ir]) {
                g.effVsCtDenRadReleased[ir]->Write(Form("h_eff_vs_ct_den_rad_released_rad_%zu", ir));
            }
            if (ir < g.denomRadReleasedCountsVsCt.size() && g.denomRadReleasedCountsVsCt[ir]) {
                g.denomRadReleasedCountsVsCt[ir]->Write(Form("h_den_rad_released_counts_vs_ct_rad_%zu", ir));
            }
            if (ir < g.numerCountsVsCt.size() && g.numerCountsVsCt[ir]) {
                g.numerCountsVsCt[ir]->Write(Form("h_num_counts_vs_ct_rad_%zu", ir));
            }
        }
        out.cd("by_period_den_rad_released");
    }
    out.cd();
    out.Close();

    std::cout << "[DrawRadCtMCEffFinalEfficiencyAndBinQA_DirectOriginalMC] wrote outputs to " << outputDir << "\n";
}
