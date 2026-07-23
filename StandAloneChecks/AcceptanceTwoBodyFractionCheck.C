#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TString.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

namespace {

using Json = GeneralHelper::Json;

std::vector<double> ReadDoubleArray(const Json &j)
{
    std::vector<double> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
        if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
}

std::vector<std::vector<double>> ReadDoubleArray2D(const Json &j)
{
    std::vector<std::vector<double>> out;
    if (!j.is_array()) return out;
    for (const auto &row : j) out.push_back(ReadDoubleArray(row));
    return out;
}

std::string GetString(const Json &j, const char *key, const std::string &fallback = "")
{
    return (j.contains(key) && j[key].is_string()) ? j[key].get<std::string>() : fallback;
}

std::string FormatEdge(double x)
{
    std::ostringstream os;
    os << std::setprecision(6) << std::defaultfloat << x;
    std::string s = os.str();
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

std::unique_ptr<TH1D> MakeHist(const std::string &name, const std::vector<double> &edges)
{
    auto h = std::make_unique<TH1D>(name.c_str(), "", static_cast<int>(edges.size()) - 1, edges.data());
    h->SetDirectory(nullptr);
    h->Sumw2();
    return h;
}

std::unique_ptr<TH1D> MakeRatio(const TH1D *num, const TH1D *den, const std::string &name, const std::string &title)
{
    if (!num || !den) return nullptr;
    auto out = std::unique_ptr<TH1D>(static_cast<TH1D *>(num->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->SetTitle(title.c_str());
    out->Divide(num, den, 1.0, 1.0, "B");
    out->SetStats(false);
    return out;
}

double Ratio(double a, double b)
{
    return b > 0.0 ? a / b : 0.0;
}

void StyleHist(TH1D *h, int color, int marker)
{
    if (!h) return;
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(0.9);
    h->SetLineWidth(2);
    h->SetStats(false);
}

void DrawRatios(const std::string &outPdf,
                const std::string &title,
                const std::vector<TH1D *> &hists,
                const std::vector<std::string> &labels,
                double yMin = 0.0,
                double yMax = 3.5)
{
    if (hists.empty()) return;
    TCanvas c(("c_" + title).c_str(), "", 900, 700);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);

    bool first = true;
    TLegend leg(0.50, 0.70, 0.88, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1D *h = hists[i];
        if (!h) continue;
        h->GetYaxis()->SetRangeUser(yMin, yMax);
        h->GetYaxis()->SetTitle("ratio");
        h->GetXaxis()->SetTitle(title.c_str());
        h->Draw(first ? "E1" : "E1 SAME");
        first = false;
        if (i < labels.size()) leg.AddEntry(h, labels[i].c_str(), "lep");
    }
    TLine line(hists.front()->GetXaxis()->GetXmin(), 1.0, hists.front()->GetXaxis()->GetXmax(), 1.0);
    line.SetLineStyle(2);
    line.SetLineColor(kGray + 2);
    line.Draw("SAME");
    leg.Draw();
    c.SaveAs(outPdf.c_str());
}

void DrawDistributions(const std::string &outPdf,
                       const std::string &title,
                       const std::vector<TH1D *> &hists,
                       const std::vector<std::string> &labels,
                       bool normalize)
{
    if (hists.empty()) return;
    TCanvas c(("c_dist_" + title + (normalize ? "_norm" : "")).c_str(), "", 900, 700);
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);
    c.SetLogy(!normalize);

    std::vector<std::unique_ptr<TH1D>> clones;
    clones.reserve(hists.size());
    double ymax = 0.0;
    for (size_t i = 0; i < hists.size(); ++i) {
        if (!hists[i]) continue;
        clones.emplace_back(static_cast<TH1D *>(hists[i]->Clone(Form("%s_draw_%zu", hists[i]->GetName(), i))));
        auto *h = clones.back().get();
        h->SetDirectory(nullptr);
        h->SetStats(false);
        if (normalize && h->Integral() > 0.0) h->Scale(1.0 / h->Integral());
        ymax = std::max(ymax, h->GetMaximum());
    }
    if (clones.empty()) return;

    TLegend leg(0.58, 0.72, 0.88, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    bool first = true;
    for (size_t i = 0; i < clones.size(); ++i) {
        auto *h = clones[i].get();
        h->GetXaxis()->SetTitle(title.c_str());
        h->GetYaxis()->SetTitle(normalize ? "normalized counts" : "counts");
        h->GetYaxis()->SetRangeUser(normalize ? 0.0 : 0.5, ymax * (normalize ? 1.35 : 8.0));
        h->Draw(first ? "HIST E" : "HIST E SAME");
        first = false;
        if (i < labels.size()) leg.AddEntry(h, labels[i].c_str(), "l");
    }
    leg.Draw();
    c.SaveAs(outPdf.c_str());
}

template <typename RNode>
std::unique_ptr<TH1D> MakeCountHist(RNode node,
                                    const std::string &name,
                                    const std::string &var,
                                    const std::vector<double> &edges,
                                    const std::string &selection)
{
    auto model = ROOT::RDF::TH1DModel(name.c_str(), "", static_cast<int>(edges.size()) - 1, edges.data());
    auto h = node.Filter(selection).Histo1D(model, var);
    auto out = std::unique_ptr<TH1D>(static_cast<TH1D *>(h->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->Sumw2();
    return out;
}

void WriteRows(std::ofstream &csv,
               const std::string &axis,
               const std::string &ptLabel,
               const std::vector<double> &edges,
               const TH1D *denBefore,
               const TH1D *denAfter,
               const TH1D *numBefore,
               const TH1D *numAfter,
               const TH1D *numBeforeNested,
               const TH1D *numAfterNested)
{
    for (int ib = 1; ib <= denBefore->GetNbinsX(); ++ib) {
        const double den0 = denBefore->GetBinContent(ib);
        const double den1 = denAfter->GetBinContent(ib);
        const double num0 = numBefore->GetBinContent(ib);
        const double num1 = numAfter->GetBinContent(ib);
        const double num0n = numBeforeNested ? numBeforeNested->GetBinContent(ib) : 0.0;
        const double num1n = numAfterNested ? numAfterNested->GetBinContent(ib) : 0.0;
        csv << axis << ','
            << ptLabel << ','
            << edges[static_cast<size_t>(ib - 1)] << ','
            << edges[static_cast<size_t>(ib)] << ','
            << den0 << ','
            << den1 << ','
            << num0 << ','
            << num1 << ','
            << num0n << ','
            << num1n << ','
            << Ratio(den1, den0) << ','
            << Ratio(num1, num0) << ','
            << Ratio(num1n, num0n) << ','
            << Ratio(num0, den0) << ','
            << Ratio(num1, den1) << ','
            << Ratio(num0n, den0) << ','
            << Ratio(num1n, den1) << '\n';
    }
}

} // namespace

int AcceptanceTwoBodyFractionCheck(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/AcceptanceTwoBodyFractionCheck")
{
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }
    gStyle->SetOptStat(0);
    std::filesystem::create_directories(outputDir);

    const Json cfg = GeneralHelper::LoadJsonFile(configPath);
    const auto common = cfg.value("common", Json::object());
    const auto path = common.value("path", Json::object());
    const auto trees = common.value("tree_names", Json::object());
    const auto binning = common.value("binning", Json::object());
    const auto selection = common.value("selection", Json::object());

    const std::string mcPath = GetString(path, "mc_path");
    const std::string mcTree = GetString(trees, "mc", "O2mchypcands");
    const std::string basicSel = GetString(selection, "basic_selection_data", "1");
    const std::vector<double> ptBins = ReadDoubleArray(binning.value("pt_bins", Json::array()));
    const std::vector<double> ctBins = ReadDoubleArray(binning.value("ct_bins_single", Json::array()));
    const std::vector<std::vector<double>> ctBinsByPt = ReadDoubleArray2D(binning.value("ct_bins_by_pt", Json::array()));

    if (mcPath.empty()) throw std::runtime_error("AcceptanceTwoBodyFractionCheck: empty common.path.mc_path");
    if (ptBins.size() < 2) throw std::runtime_error("AcceptanceTwoBodyFractionCheck: invalid pt_bins");
    if (ctBins.size() < 2) throw std::runtime_error("AcceptanceTwoBodyFractionCheck: invalid ct_bins_single");

    TChain chain(mcTree.c_str());
    auto file = std::unique_ptr<TFile>(TFile::Open(mcPath.c_str(), "READ"));
    if (!file || file->IsZombie()) throw std::runtime_error("Cannot open MC file: " + mcPath);
    GeneralHelper::fillChainFromAO2D(chain, file.get());
    if (chain.GetEntries() <= 0) throw std::runtime_error("No MC entries found in chain: " + mcTree);

    ROOT::RDataFrame rdf(chain);
    auto readyBase = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    auto ready = readyBase
        .Define("__den_before", "(fIsSurvEvSel && fIsRecoMCCollision) ? 1 : 0")
        .Define("__den_after", "(fIsSurvEvSel && fIsRecoMCCollision && fIsTwoBodyDecay > 0) ? 1 : 0")
        .Define("__num_before", ("((" + basicSel + ") && fIsReco) ? 1 : 0").c_str())
        .Define("__num_after", ("((" + basicSel + ") && fIsReco && fIsTwoBodyDecay > 0) ? 1 : 0").c_str())
        .Define("__num_before_nested", ("((" + basicSel + ") && fIsReco && fIsSurvEvSel && fIsRecoMCCollision) ? 1 : 0").c_str())
        .Define("__num_after_nested", ("((" + basicSel + ") && fIsReco && fIsSurvEvSel && fIsRecoMCCollision && fIsTwoBodyDecay > 0) ? 1 : 0").c_str());

    const std::string outRootPath = std::string(outputDir) + "/acceptance_two_body_fraction_check.root";
    TFile fout(outRootPath.c_str(), "RECREATE");
    if (fout.IsZombie()) throw std::runtime_error("Cannot create output ROOT: " + outRootPath);

    std::ofstream csv(std::string(outputDir) + "/acceptance_two_body_fraction_check.csv");
    csv << "axis,pt_range,bin_min,bin_max,"
        << "den_before,den_after,num_before,num_after,num_before_nested,num_after_nested,"
        << "den_after_over_before,num_after_over_before,num_after_nested_over_before_nested,"
        << "eff_before,eff_after,eff_before_nested,eff_after_nested\n";

    auto process_axis = [&](const std::string &axis,
                            const std::string &var,
                            const std::vector<double> &edges,
                            const std::string &extraSel,
                            const std::string &ptLabel) {
        const std::string baseSel = extraSel.empty() ? "1" : extraSel;
        auto denBefore = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_den_before", var, edges, baseSel + " && __den_before");
        auto denAfter = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_den_after", var, edges, baseSel + " && __den_after");
        auto numBefore = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_num_before", var, edges, baseSel + " && __num_before");
        auto numAfter = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_num_after", var, edges, baseSel + " && __num_after");
        auto numBeforeNested = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_num_before_nested", var, edges, baseSel + " && __num_before_nested");
        auto numAfterNested = MakeCountHist(ready, "h_" + axis + "_" + ptLabel + "_num_after_nested", var, edges, baseSel + " && __num_after_nested");

        auto denFrac = MakeRatio(denAfter.get(), denBefore.get(), "h_" + axis + "_" + ptLabel + "_den_after_over_before", "");
        auto numFrac = MakeRatio(numAfter.get(), numBefore.get(), "h_" + axis + "_" + ptLabel + "_num_after_over_before", "");
        auto numNestedFrac = MakeRatio(numAfterNested.get(), numBeforeNested.get(), "h_" + axis + "_" + ptLabel + "_num_after_nested_over_before_nested", "");
        auto effBefore = MakeRatio(numBefore.get(), denBefore.get(), "h_" + axis + "_" + ptLabel + "_eff_before", "");
        auto effAfter = MakeRatio(numAfter.get(), denAfter.get(), "h_" + axis + "_" + ptLabel + "_eff_after", "");
        auto effBeforeNested = MakeRatio(numBeforeNested.get(), denBefore.get(), "h_" + axis + "_" + ptLabel + "_eff_before_nested", "");
        auto effAfterNested = MakeRatio(numAfterNested.get(), denAfter.get(), "h_" + axis + "_" + ptLabel + "_eff_after_nested", "");
        auto effChange = MakeRatio(effAfter.get(), effBefore.get(), "h_" + axis + "_" + ptLabel + "_eff_after_over_before", "");
        auto effNestedChange = MakeRatio(effAfterNested.get(), effBeforeNested.get(), "h_" + axis + "_" + ptLabel + "_eff_after_over_before_nested", "");

        StyleHist(denFrac.get(), kBlack, kFullCircle);
        StyleHist(numFrac.get(), kRed + 1, kFullSquare);
        StyleHist(numNestedFrac.get(), kBlue + 1, kOpenSquare);
        StyleHist(effChange.get(), kMagenta + 2, kFullDiamond);
        StyleHist(effNestedChange.get(), kGreen + 2, kOpenDiamond);

        WriteRows(csv, axis, ptLabel, edges,
                  denBefore.get(), denAfter.get(),
                  numBefore.get(), numAfter.get(),
                  numBeforeNested.get(), numAfterNested.get());

        fout.cd();
        denBefore->Write();
        denAfter->Write();
        numBefore->Write();
        numAfter->Write();
        numBeforeNested->Write();
        numAfterNested->Write();
        denFrac->Write();
        numFrac->Write();
        numNestedFrac->Write();
        effBefore->Write();
        effAfter->Write();
        effBeforeNested->Write();
        effAfterNested->Write();
        effChange->Write();
        effNestedChange->Write();

        DrawRatios(std::string(outputDir) + "/two_body_fractions_" + axis + "_" + ptLabel + ".pdf",
                   axis,
                   {denFrac.get(), numFrac.get(), numNestedFrac.get()},
                   {"den after/before", "num after/before", "nested num after/before"},
                   0.0,
                   1.15);
        DrawRatios(std::string(outputDir) + "/efficiency_change_" + axis + "_" + ptLabel + ".pdf",
                   axis,
                   {effChange.get(), effNestedChange.get()},
                   {"current num/den: eff after/before", "nested num/den: eff after/before"},
                   0.0,
                   4.0);
    };

    process_axis("pt", "fAbsGenPt", ptBins, "", "allpt");
    process_axis("ct", "fGenCt", ctBins, "", "allpt");

    auto hDenCtAll = MakeCountHist(ready, "h_denominator_fGenCt_all", "fGenCt", ctBins, "__den_before");
    auto hDenCtTwoBody = MakeCountHist(ready, "h_denominator_fGenCt_two_body", "fGenCt", ctBins, "__den_after");
    auto hDenCtThreeBody = MakeCountHist(ready, "h_denominator_fGenCt_three_body", "fGenCt", ctBins, "__den_before && !(fIsTwoBodyDecay > 0)");
    StyleHist(hDenCtAll.get(), kBlack, kFullCircle);
    StyleHist(hDenCtTwoBody.get(), kRed + 1, kFullSquare);
    StyleHist(hDenCtThreeBody.get(), kBlue + 1, kOpenSquare);
    fout.cd();
    hDenCtAll->Write();
    hDenCtTwoBody->Write();
    hDenCtThreeBody->Write();
    DrawDistributions(std::string(outputDir) + "/denominator_fGenCt_all_twobody_threebody.pdf",
                      "fGenCt",
                      {hDenCtAll.get(), hDenCtTwoBody.get(), hDenCtThreeBody.get()},
                      {"all denominator", "two-body denominator", "three-body denominator"},
                      false);
    DrawDistributions(std::string(outputDir) + "/denominator_fGenCt_all_twobody_threebody_normalized.pdf",
                      "fGenCt",
                      {hDenCtAll.get(), hDenCtTwoBody.get(), hDenCtThreeBody.get()},
                      {"all denominator", "two-body denominator", "three-body denominator"},
                      true);

    for (size_t ip = 0; ip + 1 < ptBins.size() && ip < ctBinsByPt.size(); ++ip) {
        const auto &ctEdges = ctBinsByPt[ip];
        if (ctEdges.size() < 2) continue;
        const std::string ptLabel = "pt_" + FormatEdge(ptBins[ip]) + "_" + FormatEdge(ptBins[ip + 1]);
        const std::string ptSel = Form("(fAbsGenPt >= %.17g && fAbsGenPt < %.17g)", ptBins[ip], ptBins[ip + 1]);
        process_axis("ct", "fGenCt", ctEdges, ptSel, ptLabel);
    }

    fout.Close();
    csv.close();
    std::cout << "[AcceptanceTwoBodyFractionCheck] Output dir: " << outputDir << std::endl;
    std::cout << "[AcceptanceTwoBodyFractionCheck] CSV: "
              << std::string(outputDir) + "/acceptance_two_body_fraction_check.csv" << std::endl;
    return 0;
}
