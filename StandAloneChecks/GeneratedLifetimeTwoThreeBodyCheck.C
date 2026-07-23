#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

namespace {

using Json = GeneralHelper::Json;
constexpr double kSpeedOfLightCmPerPs = 0.0299792458;

std::string GetString(const Json &j, const char *key, const std::string &fallback = "")
{
    return (j.contains(key) && j[key].is_string()) ? j[key].get<std::string>() : fallback;
}

std::vector<double> ReadDoubleArray(const Json &j)
{
    std::vector<double> out;
    if (!j.is_array()) return out;
    for (const auto &v : j) {
        if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
}

std::vector<double> MakeUniformEdges(double min, double max, double width)
{
    std::vector<double> edges;
    if (width <= 0.0 || max <= min) return edges;
    const int nBins = static_cast<int>(std::ceil((max - min) / width));
    edges.reserve(nBins + 1);
    for (int i = 0; i <= nBins; ++i) {
        edges.push_back(std::min(max, min + i * width));
    }
    if (edges.size() < 2 || edges.back() < max) edges.push_back(max);
    return edges;
}

template <typename RNode>
std::unique_ptr<TH1D> MakeCtHist(RNode node,
                                 const std::string &name,
                                 const std::vector<double> &edges,
                                 const std::string &selection)
{
    auto model = ROOT::RDF::TH1DModel(name.c_str(), ";#it{c}t_{gen} (cm);dN/d#it{c}t", static_cast<int>(edges.size()) - 1, edges.data());
    auto h = node.Filter(selection).Histo1D(model, "fGenCt");
    auto out = std::unique_ptr<TH1D>(static_cast<TH1D *>(h->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->Sumw2();
    for (int ib = 1; ib <= out->GetNbinsX(); ++ib) {
        const double width = out->GetBinWidth(ib);
        if (width <= 0.0) continue;
        out->SetBinContent(ib, out->GetBinContent(ib) / width);
        out->SetBinError(ib, out->GetBinError(ib) / width);
    }
    return out;
}

void Style(TH1D *h, Color_t color, Style_t marker)
{
    if (!h) return;
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(0.9);
    h->SetLineWidth(2);
    h->SetStats(false);
}

struct FitSummary {
    std::string sample;
    std::string channel;
    double entries{0.0};
    double tauCm{0.0};
    double tauCmErr{0.0};
    double chi2Ndf{0.0};
    int ndf{0};
    int status{-1};
};

double TauPs(const FitSummary &fit)
{
    return fit.tauCm / kSpeedOfLightCmPerPs;
}

double TauPsErr(const FitSummary &fit)
{
    return fit.tauCmErr / kSpeedOfLightCmPerPs;
}

FitSummary FitExpo(TH1D *h,
                   const std::string &sample,
                   const std::string &channel,
                   double fitMin,
                   double fitMax,
                   Color_t color)
{
    FitSummary out;
    out.sample = sample;
    out.channel = channel;
    out.entries = h ? h->Integral("width") : 0.0;
    if (!h) return out;
    auto *f = new TF1((std::string("f_exp_") + sample + "_" + channel).c_str(), "[0]*exp(-x/[1])", fitMin, fitMax);
    f->SetParNames("N0", "tau");
    f->SetParameters(std::max(1.0, h->GetMaximum()), 7.6);
    f->SetParLimits(0, 0.0, 1e12);
    f->SetParLimits(1, 0.1, 50.0);
    f->SetLineColor(color);
    f->SetLineWidth(2);
    const int status = h->Fit(f, "Q0RS", "", fitMin, fitMax);
    out.status = status;
    out.tauCm = f->GetParameter(1);
    out.tauCmErr = f->GetParError(1);
    out.ndf = f->GetNDF();
    out.chi2Ndf = out.ndf > 0 ? f->GetChisquare() / out.ndf : 0.0;
    h->GetListOfFunctions()->Add(f);
    return out;
}

void DrawSet(const std::string &outPdf,
             const std::string &title,
             const std::vector<TH1D *> &hists,
             const std::vector<FitSummary> &fits,
             double fitMin,
             double fitMax)
{
    TCanvas c(("c_" + title).c_str(), "", 900, 720);
    c.SetLogy();
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.04);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.06);

    double ymax = 0.0;
    double ymin = 1e30;
    for (auto *h : hists) {
        if (!h) continue;
        ymax = std::max(ymax, h->GetMaximum());
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double y = h->GetBinContent(ib);
            if (y > 0.0) ymin = std::min(ymin, y);
        }
    }
    if (ymax <= 0.0) ymax = 1.0;
    if (ymin >= 1e29) ymin = 0.5;

    bool first = true;
    TLegend leg(0.16, 0.16, 0.43, 0.34);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.032);
    for (auto *h : hists) {
        if (!h) continue;
        h->GetYaxis()->SetRangeUser(ymin * 0.35, ymax * 7.0);
        h->GetXaxis()->SetRangeUser(fitMin, fitMax);
        h->Draw(first ? "E1" : "E1 SAME");
        first = false;
        leg.AddEntry(h, h->GetTitle(), "lep");
    }
    leg.Draw();

    TPaveText text(0.48, 0.64, 0.88, 0.88, "NDC");
    text.SetFillStyle(0);
    text.SetBorderSize(0);
    text.SetTextAlign(12);
    text.SetTextSize(0.030);
    text.AddText(title.c_str());
    for (const auto &fit : fits) {
        text.AddText(Form("%s: #tau = %.1f #pm %.1f ps, #chi^{2}/ndf = %.2f",
                          fit.channel.c_str(), TauPs(fit), TauPsErr(fit), fit.chi2Ndf));
    }
    text.Draw();
    c.SaveAs(outPdf.c_str());
}

} // namespace

int GeneratedLifetimeTwoThreeBodyCheck(
    const char *configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/GeneratedLifetimeTwoThreeBodyCheck",
    double fitMin = 1.0,
    double fitMax = 33.0)
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

    const std::string mcPath = GetString(path, "mc_path");
    const std::string mcTree = GetString(trees, "mc", "O2mchypcands");
    auto ctEdges = ReadDoubleArray(binning.value("ct_bins_single", Json::array()));
    if (ctEdges.size() < 2) {
        ctEdges = {1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 27, 33};
    }
    fitMin = std::max(fitMin, ctEdges.front());
    fitMax = std::min(fitMax, ctEdges.back());
    auto fineCtEdges = MakeUniformEdges(ctEdges.front(), ctEdges.back(), 1.0);
    if (fineCtEdges.size() < 2) fineCtEdges = ctEdges;

    TChain chain(mcTree.c_str());
    auto file = std::unique_ptr<TFile>(TFile::Open(mcPath.c_str(), "READ"));
    if (!file || file->IsZombie()) throw std::runtime_error("Cannot open MC file: " + mcPath);
    GeneralHelper::fillChainFromAO2D(chain, file.get());
    if (chain.GetEntries() <= 0) throw std::runtime_error("No entries found in MC chain");

    ROOT::RDataFrame rdf(chain);
    auto readyBase = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    auto ready = readyBase
        .Define("__is_two_body", "(fIsTwoBodyDecay > 0)")
        .Define("__is_three_body", "!(fIsTwoBodyDecay > 0)")
        .Define("__denominator", "(fIsSurvEvSel && fIsRecoMCCollision)");

    auto hGenAll = MakeCtHist(ready, "h_gen_ct_all", ctEdges, "1");
    auto hGenTwo = MakeCtHist(ready, "h_gen_ct_two_body", ctEdges, "__is_two_body");
    auto hGenThree = MakeCtHist(ready, "h_gen_ct_three_body", ctEdges, "__is_three_body");
    auto hPureGenTwoFine = MakeCtHist(ready, "h_pure_generated_ct_two_body_fine", fineCtEdges, "fIsReco == false && __is_two_body");
    auto hDenAll = MakeCtHist(ready, "h_den_ct_all", ctEdges, "__denominator");
    auto hDenTwo = MakeCtHist(ready, "h_den_ct_two_body", ctEdges, "__denominator && __is_two_body");
    auto hDenThree = MakeCtHist(ready, "h_den_ct_three_body", ctEdges, "__denominator && __is_three_body");
    auto hSigAll = MakeCtHist(ready, "h_signal_ct_all", ctEdges, "fIsSignal");
    auto hSigTwo = MakeCtHist(ready, "h_signal_ct_two_body", ctEdges, "fIsSignal && __is_two_body");
    auto hSigThree = MakeCtHist(ready, "h_signal_ct_three_body", ctEdges, "fIsSignal && __is_three_body");

    hGenAll->SetTitle("all generated");
    hGenTwo->SetTitle("two-body generated");
    hGenThree->SetTitle("three-body generated");
    hPureGenTwoFine->SetTitle("two-body pure generated");
    hDenAll->SetTitle("all denominator");
    hDenTwo->SetTitle("two-body denominator");
    hDenThree->SetTitle("three-body denominator");
    hSigAll->SetTitle("all signal");
    hSigTwo->SetTitle("two-body signal");
    hSigThree->SetTitle("three-body signal");
    Style(hGenAll.get(), kBlack, kFullCircle);
    Style(hGenTwo.get(), kRed + 1, kFullSquare);
    Style(hGenThree.get(), kBlue + 1, kOpenSquare);
    Style(hPureGenTwoFine.get(), kRed + 1, kFullSquare);
    Style(hDenAll.get(), kBlack, kFullCircle);
    Style(hDenTwo.get(), kRed + 1, kFullSquare);
    Style(hDenThree.get(), kBlue + 1, kOpenSquare);
    Style(hSigAll.get(), kBlack, kFullCircle);
    Style(hSigTwo.get(), kRed + 1, kFullSquare);
    Style(hSigThree.get(), kBlue + 1, kOpenSquare);

    std::vector<FitSummary> genFits;
    genFits.push_back(FitExpo(hGenAll.get(), "gen", "all", fitMin, fitMax, kBlack));
    genFits.push_back(FitExpo(hGenTwo.get(), "gen", "two-body", fitMin, fitMax, kRed + 1));
    genFits.push_back(FitExpo(hGenThree.get(), "gen", "three-body", fitMin, fitMax, kBlue + 1));
    std::vector<FitSummary> pureGenTwoFineFits;
    pureGenTwoFineFits.push_back(FitExpo(hPureGenTwoFine.get(), "pure_generated_fine", "two-body", fitMin, fitMax, kRed + 1));
    std::vector<FitSummary> denFits;
    denFits.push_back(FitExpo(hDenAll.get(), "den", "all", fitMin, fitMax, kBlack));
    denFits.push_back(FitExpo(hDenTwo.get(), "den", "two-body", fitMin, fitMax, kRed + 1));
    denFits.push_back(FitExpo(hDenThree.get(), "den", "three-body", fitMin, fitMax, kBlue + 1));
    std::vector<FitSummary> sigFits;
    sigFits.push_back(FitExpo(hSigAll.get(), "signal", "all", fitMin, fitMax, kBlack));
    sigFits.push_back(FitExpo(hSigTwo.get(), "signal", "two-body", fitMin, fitMax, kRed + 1));
    sigFits.push_back(FitExpo(hSigThree.get(), "signal", "three-body", fitMin, fitMax, kBlue + 1));

    const std::string rootPath = std::string(outputDir) + "/generated_lifetime_two_three_body.root";
    TFile fout(rootPath.c_str(), "RECREATE");
    hGenAll->Write();
    hGenTwo->Write();
    hGenThree->Write();
    hPureGenTwoFine->Write();
    hDenAll->Write();
    hDenTwo->Write();
    hDenThree->Write();
    hSigAll->Write();
    hSigTwo->Write();
    hSigThree->Write();
    fout.Close();

    std::ofstream csv(std::string(outputDir) + "/generated_lifetime_two_three_body.csv");
    csv << "sample,channel,entries,tau_cm,tau_cm_err,tau_ps,tau_ps_err,chi2_ndf,ndf,fit_status\n";
    for (const auto &fit : genFits) {
        csv << fit.sample << ',' << fit.channel << ',' << fit.entries << ','
            << fit.tauCm << ',' << fit.tauCmErr << ','
            << TauPs(fit) << ',' << TauPsErr(fit) << ','
            << fit.chi2Ndf << ','
            << fit.ndf << ',' << fit.status << '\n';
    }
    for (const auto &fit : pureGenTwoFineFits) {
        csv << fit.sample << ',' << fit.channel << ',' << fit.entries << ','
            << fit.tauCm << ',' << fit.tauCmErr << ','
            << TauPs(fit) << ',' << TauPsErr(fit) << ','
            << fit.chi2Ndf << ','
            << fit.ndf << ',' << fit.status << '\n';
    }
    for (const auto &fit : denFits) {
        csv << fit.sample << ',' << fit.channel << ',' << fit.entries << ','
            << fit.tauCm << ',' << fit.tauCmErr << ','
            << TauPs(fit) << ',' << TauPsErr(fit) << ','
            << fit.chi2Ndf << ','
            << fit.ndf << ',' << fit.status << '\n';
    }
    for (const auto &fit : sigFits) {
        csv << fit.sample << ',' << fit.channel << ',' << fit.entries << ','
            << fit.tauCm << ',' << fit.tauCmErr << ','
            << TauPs(fit) << ',' << TauPsErr(fit) << ','
            << fit.chi2Ndf << ','
            << fit.ndf << ',' << fit.status << '\n';
    }
    csv.close();

    DrawSet(std::string(outputDir) + "/generated_fGenCt_lifetime_fit.pdf",
            "Generated fGenCt lifetime fit",
            {hGenAll.get(), hGenTwo.get(), hGenThree.get()},
            genFits,
            fitMin,
            fitMax);
    DrawSet(std::string(outputDir) + "/twobody_pure_generated_fGenCt_fine_lifetime_fit.pdf",
            "Two-body pure generated fGenCt lifetime fit",
            {hPureGenTwoFine.get()},
            pureGenTwoFineFits,
            fitMin,
            fitMax);
    DrawSet(std::string(outputDir) + "/denominator_fGenCt_lifetime_fit.pdf",
            "Denominator-selected fGenCt lifetime fit",
            {hDenAll.get(), hDenTwo.get(), hDenThree.get()},
            denFits,
            fitMin,
            fitMax);
    DrawSet(std::string(outputDir) + "/signal_fGenCt_lifetime_fit.pdf",
            "Signal-selected fGenCt lifetime fit",
            {hSigAll.get(), hSigTwo.get(), hSigThree.get()},
            sigFits,
            fitMin,
            fitMax);

    std::cout << "[GeneratedLifetimeTwoThreeBodyCheck] Output dir: " << outputDir << std::endl;
    for (const auto &fit : genFits) {
        std::cout << "[Generated] " << fit.channel << ": tau=" << TauPs(fit)
                  << " +/- " << TauPsErr(fit) << " ps, chi2/ndf=" << fit.chi2Ndf << std::endl;
    }
    for (const auto &fit : pureGenTwoFineFits) {
        std::cout << "[PureGeneratedFine] " << fit.channel << ": tau=" << TauPs(fit)
                  << " +/- " << TauPsErr(fit) << " ps, chi2/ndf=" << fit.chi2Ndf << std::endl;
    }
    for (const auto &fit : denFits) {
        std::cout << "[Denominator] " << fit.channel << ": tau=" << TauPs(fit)
                  << " +/- " << TauPsErr(fit) << " ps, chi2/ndf=" << fit.chi2Ndf << std::endl;
    }
    for (const auto &fit : sigFits) {
        std::cout << "[Signal] " << fit.channel << ": tau=" << TauPs(fit)
                  << " +/- " << TauPsErr(fit) << " ps, chi2/ndf=" << fit.chi2Ndf << std::endl;
    }
    return 0;
}
