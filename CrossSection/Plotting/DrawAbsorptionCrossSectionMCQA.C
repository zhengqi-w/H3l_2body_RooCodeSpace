// DrawAbsorptionCrossSectionMCQA.C
// QA plots for the hypertriton material-absorption information stored in MCHypCands.
// Usage:
//   root -l -b -q 'ROOTWorkFlow/CodeSpace/CrossSection/Plotting/DrawAbsorptionCrossSectionMCQA.C()'

#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TRandom3.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../include/include.h"

using namespace Physics;

namespace {

constexpr double kH3LMass = 2.99131; // GeV/c^2
constexpr double kHe3Mass = 2.809230089; // GeV/c^2
constexpr double kOriginalTauCt = 253.0 * c_cm_per_ps;

std::vector<double> BuildCtBins()
{
    std::vector<double> bins;
    for (double x = 0.0; x < 10.0 - 1e-9; x += 0.5) bins.push_back(x);
    for (double x = 10.0; x < 20.0 - 1e-9; x += 1.0) bins.push_back(x);
    for (double x = 20.0; x < 40.0 - 1e-9; x += 2.0) bins.push_back(x);
    bins.push_back(40.0);
    return bins;
}

const std::vector<double> kPtBins{0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 8.0};
const std::vector<double> kCtBins = BuildCtBins();

std::string MacroDir()
{
    std::string macroPath = __FILE__;
    const size_t pos = macroPath.find_last_of("/\\");
    return (pos == std::string::npos) ? "." : macroPath.substr(0, pos);
}

std::string ResolveOutputPath(const std::string& path)
{
    if (path.empty() || gSystem->IsAbsoluteFileName(path.c_str())) {
        return path;
    }
    return MacroDir() + "/" + path;
}

void EnsureDir(const std::string& dir)
{
    gSystem->mkdir(dir.c_str(), true);
}

std::string AddSuffixBeforeExtension(const std::string& fileName, const std::string& suffix)
{
    const size_t slash = fileName.find_last_of("/\\");
    const size_t dot = fileName.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return fileName + suffix;
    }
    return fileName.substr(0, dot) + suffix + fileName.substr(dot);
}

int StatusCategory(int status)
{
    if (status == 0) return 1;
    if (status == 4) return 2;
    if (status == 13) return 3;
    if (status == 20) return 4;
    if (status == 23) return 5;
    if (status == 37) return 6;
    return 7;
}

const char* StatusLabel(int cat)
{
    switch (cat) {
    case 1: return "0 primary/reco";
    case 2: return "4 decay";
    case 3: return "13 hadronic";
    case 4: return "20 H elastic";
    case 5: return "23 H inelastic";
    case 6: return "37 Cerenkov";
    default: return "other";
    }
}

bool IsAbsorptionConservative(int status)
{
    return status == 23;
}

bool IsAbsorptionInclusive(int status)
{
    return status == 13 || status == 20 || status == 23;
}

int FindPtBin(double pt)
{
    for (size_t i = 0; i + 1 < kPtBins.size(); ++i) {
        if (pt >= kPtBins[i] && pt < kPtBins[i + 1]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::unique_ptr<TH1D> MakeHist1D(const std::string& name,
                                 const std::string& title,
                                 const std::vector<double>& bins)
{
    auto h = std::make_unique<TH1D>(name.c_str(), title.c_str(), static_cast<int>(bins.size() - 1), bins.data());
    h->Sumw2();
    h->SetDirectory(nullptr);
    return h;
}

std::unique_ptr<TH1D> MakeFixedHist1D(const std::string& name,
                                      const std::string& title,
                                      int nBins,
                                      double min,
                                      double max)
{
    auto h = std::make_unique<TH1D>(name.c_str(), title.c_str(), nBins, min, max);
    h->Sumw2();
    h->SetDirectory(nullptr);
    return h;
}

std::unique_ptr<TH1D> MakeStatusHist(const std::string& name, const std::string& title)
{
    auto h = std::make_unique<TH1D>(name.c_str(), title.c_str(), 7, 0.5, 7.5);
    h->Sumw2();
    h->SetDirectory(nullptr);
    for (int i = 1; i <= 7; ++i) {
        h->GetXaxis()->SetBinLabel(i, StatusLabel(i));
    }
    return h;
}

void StyleHist(TH1* h, Color_t color, Style_t marker)
{
    h->SetStats(0);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(0.9);
    h->SetLineWidth(2);
    h->GetXaxis()->SetLabelFont(42);
    h->GetYaxis()->SetLabelFont(42);
    h->GetXaxis()->SetTitleFont(42);
    h->GetYaxis()->SetTitleFont(42);
    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetXaxis()->SetLabelSize(0.035);
    h->GetYaxis()->SetLabelSize(0.04);
    h->GetYaxis()->SetTitleOffset(1.25);
}

std::unique_ptr<TH1D> MakeDensityClone(const TH1D* h, const std::string& name)
{
    auto out = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(h->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    for (int ib = 1; ib <= out->GetNbinsX(); ++ib) {
        const double width = out->GetXaxis()->GetBinWidth(ib);
        if (width <= 0.0) continue;
        out->SetBinContent(ib, out->GetBinContent(ib) / width);
        out->SetBinError(ib, out->GetBinError(ib) / width);
    }
    return out;
}

struct ExpoFitResult {
    std::unique_ptr<TF1> func;
    double tauCt = 0.0;
    double tauCtErr = 0.0;
    double tauPs = 0.0;
    double tauPsErr = 0.0;
    double chi2 = 0.0;
    int ndf = 0;
};

ExpoFitResult FitCtDensity(TH1D* h, const std::string& name, Color_t color)
{
    ExpoFitResult out;
    if (!h || h->Integral("width") <= 0.0) {
        return out;
    }

    out.func = std::make_unique<TF1>(name.c_str(), "[0]*exp(-x/[1])", 0.5, 40.0);
    out.func->SetParameters(std::max(1e-12, h->GetMaximum()), 253.0 * c_cm_per_ps);
    out.func->SetParLimits(0, 0.0, 10.0 * std::max(1e-12, h->GetMaximum()));
    out.func->SetParLimits(1, 1.0, 30.0);
    out.func->SetLineColor(color);
    out.func->SetLineWidth(3);
    out.func->SetNpx(600);
    h->Fit(out.func.get(), "RQ0");

    out.tauCt = out.func->GetParameter(1);
    out.tauCtErr = out.func->GetParError(1);
    out.tauPs = out.tauCt / c_cm_per_ps;
    out.tauPsErr = out.tauCtErr / c_cm_per_ps;
    out.chi2 = out.func->GetChisquare();
    out.ndf = out.func->GetNDF();
    return out;
}

struct SampleQA {
    std::string name;
    std::string title;
    Color_t color = kBlack;

    long long nAll = 0;
    long long nSignal = 0;
    long long nTwoBody = 0;
    long long nReco = 0;
    long long nAbs23 = 0;
    long long nAbsInclusive = 0;
    long long nNonTwoDecay = 0;
    std::map<int, long long> statusCounts;

    std::unique_ptr<TH1D> hStatus;
    std::unique_ptr<TH1D> hPtSurvived;
    std::unique_ptr<TH1D> hPtAbs23;
    std::unique_ptr<TH1D> hPtAbsInclusive;
    std::unique_ptr<TH1D> hPtDen23;
    std::unique_ptr<TH1D> hPtDenInclusive;
    std::unique_ptr<TH1D> hCtSurvived;
    std::vector<std::unique_ptr<TH1D>> hCtAbs23Pt;
    std::vector<std::unique_ptr<TH1D>> hCtAbsInclusivePt;
    std::vector<std::unique_ptr<TH1D>> hCtDen23Pt;
    std::unique_ptr<TH1D> hAbsVertexLogR;
    std::unique_ptr<TH1D> hAbsVertexRxy;
    std::unique_ptr<TH1D> hAbsVertexLength;
    std::unique_ptr<TH1D> hAbsVertexX;
    std::unique_ptr<TH1D> hAbsVertexY;
    std::unique_ptr<TH2D> hAbsVertexXY;
    std::unique_ptr<TH1D> hAllVertexX;
    std::unique_ptr<TH1D> hAllVertexY;
    std::unique_ptr<TH2D> hAllVertexXY;
    std::unique_ptr<TH1D> hRecoVertexX;
    std::unique_ptr<TH1D> hRecoVertexY;
    std::unique_ptr<TH2D> hRecoVertexXY;
    std::unique_ptr<TH2D> hStatusVsPt;
};

void InitSample(SampleQA& s)
{
    s.hStatus = MakeStatusHist("h_status_" + s.name, ";TMCProcess category;Entries");
    s.hPtSurvived = MakeHist1D("h_pt_survived_" + s.name, ";p_{T} (GeV/c);Entries", kPtBins);
    s.hPtAbs23 = MakeHist1D("h_pt_abs23_" + s.name, ";p_{T} (GeV/c);Entries", kPtBins);
    s.hPtAbsInclusive = MakeHist1D("h_pt_absinclusive_" + s.name, ";p_{T} (GeV/c);Entries", kPtBins);
    s.hPtDen23 = MakeHist1D("h_pt_den23_" + s.name, ";p_{T} (GeV/c);Entries", kPtBins);
    s.hPtDenInclusive = MakeHist1D("h_pt_deninclusive_" + s.name, ";p_{T} (GeV/c);Entries", kPtBins);
    s.hCtSurvived = MakeHist1D("h_ct_survived_" + s.name, ";c#tau from two-body decay (cm);Entries / cm", kCtBins);
    s.hCtAbs23Pt.clear();
    s.hCtAbsInclusivePt.clear();
    s.hCtDen23Pt.clear();
    for (size_t ipt = 0; ipt + 1 < kPtBins.size(); ++ipt) {
        const std::string tag = Form("pt_%g_%g", kPtBins[ipt], kPtBins[ipt + 1]);
        s.hCtAbs23Pt.push_back(MakeHist1D("h_ct_abs23_" + s.name + "_" + tag, ";stored c#tau (cm);Entries", kCtBins));
        s.hCtAbsInclusivePt.push_back(MakeHist1D("h_ct_absinclusive_" + s.name + "_" + tag, ";stored c#tau (cm);Entries", kCtBins));
        s.hCtDen23Pt.push_back(MakeHist1D("h_ct_den23_" + s.name + "_" + tag, ";stored c#tau (cm);Entries", kCtBins));
    }
    s.hAbsVertexLogR = std::make_unique<TH1D>(("h_abs_vertex_logr_" + s.name).c_str(), ";log_{10}(|r_{decay}| / cm);Entries", 120, -8.0, 20.0);
    s.hAbsVertexLogR->Sumw2();
    s.hAbsVertexLogR->SetDirectory(nullptr);
    s.hAbsVertexRxy = MakeFixedHist1D("h_abs_vertex_rxy_" + s.name, ";absolute R_{xy} = #sqrt{x^{2}+y^{2}} (cm);Entries", 160, 0.0, 400.0);
    s.hAbsVertexLength = MakeFixedHist1D("h_abs_vertex_length_" + s.name, ";absolute |r_{decay}| (cm);Entries", 200, 0.0, 1000.0);
    s.hAbsVertexX = MakeFixedHist1D("h_abs_vertex_x_" + s.name, ";absolute x_{decay} (cm);Entries", 160, -400.0, 400.0);
    s.hAbsVertexY = MakeFixedHist1D("h_abs_vertex_y_" + s.name, ";absolute y_{decay} (cm);Entries", 160, -400.0, 400.0);
    s.hAbsVertexXY = std::make_unique<TH2D>(("h_abs_vertex_xy_" + s.name).c_str(), ";absolute x_{decay} (cm);absolute y_{decay} (cm)",
                                            120, -400.0, 400.0, 120, -400.0, 400.0);
    s.hAbsVertexXY->SetDirectory(nullptr);
    s.hAllVertexX = MakeFixedHist1D("h_all_vertex_x_" + s.name, ";absolute x_{decay} (cm);Entries", 160, -400.0, 400.0);
    s.hAllVertexY = MakeFixedHist1D("h_all_vertex_y_" + s.name, ";absolute y_{decay} (cm);Entries", 160, -400.0, 400.0);
    s.hAllVertexXY = std::make_unique<TH2D>(("h_all_vertex_xy_" + s.name).c_str(), ";absolute x_{decay} (cm);absolute y_{decay} (cm)",
                                            120, -400.0, 400.0, 120, -400.0, 400.0);
    s.hAllVertexXY->SetDirectory(nullptr);
    s.hRecoVertexX = MakeFixedHist1D("h_reco_vertex_x_" + s.name, ";absolute x_{decay} (cm);Entries", 160, -40.0, 40.0);
    s.hRecoVertexY = MakeFixedHist1D("h_reco_vertex_y_" + s.name, ";absolute y_{decay} (cm);Entries", 160, -40.0, 40.0);
    s.hRecoVertexXY = std::make_unique<TH2D>(("h_reco_vertex_xy_" + s.name).c_str(), ";absolute x_{decay} (cm);absolute y_{decay} (cm)",
                                             160, -40.0, 40.0, 160, -40.0, 40.0);
    s.hRecoVertexXY->SetDirectory(nullptr);
    s.hStatusVsPt = std::make_unique<TH2D>(("h_status_vs_pt_" + s.name).c_str(), ";p_{T} (GeV/c);TMCProcess category",
                                           static_cast<int>(kPtBins.size() - 1), kPtBins.data(), 7, 0.5, 7.5);
    s.hStatusVsPt->SetDirectory(nullptr);
    for (int i = 1; i <= 7; ++i) {
        s.hStatusVsPt->GetYaxis()->SetBinLabel(i, StatusLabel(i));
    }
}

void AppendSampleFile(const std::string& filePath, SampleQA& s)
{
    TFile input(filePath.c_str(), "READ");
    if (input.IsZombie()) {
        std::cerr << "[Error] Cannot open " << filePath << "\n";
        return;
    }

    TIter nextKey(input.GetListOfKeys());
    TKey* key = nullptr;
    while ((key = dynamic_cast<TKey*>(nextKey()))) {
        if (std::string(key->GetClassName()) != "TDirectoryFile") continue;
        auto* dir = dynamic_cast<TDirectory*>(key->ReadObj());
        if (!dir) continue;
        auto* tree = dynamic_cast<TTree*>(dir->Get("O2mchypcands"));
        if (!tree) continue;

        Float_t genPt = 0.f;
        Float_t genEta = 0.f;
        Float_t genX = 0.f;
        Float_t genY = 0.f;
        Float_t genZ = 0.f;
        Float_t primX = 0.f;
        Float_t primY = 0.f;
        Float_t primZ = 0.f;
        Bool_t isSignal = false;
        Bool_t isTwoBody = false;
        Bool_t isReco = false;
        Int_t status = 0;

        tree->SetBranchAddress("fGenPt", &genPt);
        tree->SetBranchAddress("fGenEta", &genEta);
        tree->SetBranchAddress("fGenXDecVtx", &genX);
        tree->SetBranchAddress("fGenYDecVtx", &genY);
        tree->SetBranchAddress("fGenZDecVtx", &genZ);
        tree->SetBranchAddress("fXPrimVtx", &primX);
        tree->SetBranchAddress("fYPrimVtx", &primY);
        tree->SetBranchAddress("fZPrimVtx", &primZ);
        tree->SetBranchAddress("fIsSignal", &isSignal);
        tree->SetBranchAddress("fIsTwoBodyDecay", &isTwoBody);
        tree->SetBranchAddress("fIsReco", &isReco);
        tree->SetBranchAddress("fStatusCode", &status);

        const Long64_t entries = tree->GetEntries();
        for (Long64_t i = 0; i < entries; ++i) {
            tree->GetEntry(i);
            ++s.nAll;
            if (isSignal) ++s.nSignal;
            if (isTwoBody) ++s.nTwoBody;
            if (isReco) ++s.nReco;
            s.statusCounts[status]++;

            const int cat = StatusCategory(status);
            const double pt = std::abs(static_cast<double>(genPt));
            const int ptBin = FindPtBin(pt);
            s.hStatus->Fill(cat);
            s.hStatusVsPt->Fill(pt, cat);

            const double p = pt * std::cosh(static_cast<double>(genEta));
            const double l = std::sqrt(genX * genX + genY * genY + genZ * genZ);
            const double ct = (p > 0.0) ? l * kH3LMass / p : -1.0;
            const double absoluteGenX = genX + primX;
            const double absoluteGenY = genY + primY;
            const double absoluteGenZ = genZ + primZ;
            const bool hasValidStoredCt = ct >= kCtBins.front() && ct < kCtBins.back() && std::isfinite(ct);
            if (std::isfinite(absoluteGenX) && std::isfinite(absoluteGenY)) {
                s.hAllVertexX->Fill(absoluteGenX);
                s.hAllVertexY->Fill(absoluteGenY);
                s.hAllVertexXY->Fill(absoluteGenX, absoluteGenY);
                if (isReco) {
                    s.hRecoVertexX->Fill(absoluteGenX);
                    s.hRecoVertexY->Fill(absoluteGenY);
                    s.hRecoVertexXY->Fill(absoluteGenX, absoluteGenY);
                }
            }

            const bool abs23 = IsAbsorptionConservative(status);
            const bool absInclusive = IsAbsorptionInclusive(status);
            if (abs23) ++s.nAbs23;
            if (absInclusive) ++s.nAbsInclusive;
            if (!isTwoBody && status == 4) ++s.nNonTwoDecay;

            s.hPtDen23->Fill(pt);
            s.hPtDenInclusive->Fill(pt);
            if (hasValidStoredCt && ptBin >= 0) {
                s.hCtDen23Pt[static_cast<size_t>(ptBin)]->Fill(ct);
            }

            if (isTwoBody) {
                s.hPtSurvived->Fill(pt);

                if (hasValidStoredCt) {
                    s.hCtSurvived->Fill(ct);
                }
            }

            if (abs23) {
                s.hPtAbs23->Fill(pt);
                if (hasValidStoredCt && ptBin >= 0) {
                    s.hCtAbs23Pt[static_cast<size_t>(ptBin)]->Fill(ct);
                }
                const double rxy = std::sqrt(absoluteGenX * absoluteGenX + absoluteGenY * absoluteGenY);
                const double length = std::sqrt(absoluteGenX * absoluteGenX +
                                                absoluteGenY * absoluteGenY +
                                                absoluteGenZ * absoluteGenZ);
                if (std::isfinite(length) && length > 0.0) {
                    s.hAbsVertexLogR->Fill(std::log10(length));
                    s.hAbsVertexLength->Fill(length);
                }
                if (std::isfinite(rxy) && rxy >= 0.0) {
                    s.hAbsVertexRxy->Fill(rxy);
                }
                if (std::isfinite(absoluteGenX) && std::isfinite(absoluteGenY)) {
                    s.hAbsVertexX->Fill(absoluteGenX);
                    s.hAbsVertexY->Fill(absoluteGenY);
                    s.hAbsVertexXY->Fill(absoluteGenX, absoluteGenY);
                }
            }

            if (absInclusive) {
                s.hPtAbsInclusive->Fill(pt);
                if (hasValidStoredCt && ptBin >= 0) {
                    s.hCtAbsInclusivePt[static_cast<size_t>(ptBin)]->Fill(ct);
                }
            }
        }
    }
}

void ReadSample(const std::string& filePath, SampleQA& s)
{
    InitSample(s);
    AppendSampleFile(filePath, s);
}

void ReadSamples(const std::vector<std::string>& filePaths, SampleQA& s)
{
    InitSample(s);
    for (const auto& filePath : filePaths) {
        std::cout << "  " << filePath << "\n";
        AppendSampleFile(filePath, s);
    }
}

std::unique_ptr<TH1D> MakeFraction(const TH1D* num, const TH1D* den, const std::string& name, const std::string& title)
{
    auto out = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(num->Clone(name.c_str())));
    out->SetDirectory(nullptr);
    out->SetTitle(title.c_str());
    out->Divide(num, den, 1.0, 1.0, "B");
    return out;
}

void DrawStatusComparison(const SampleQA& noAbs, const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_status_comparison", "TMCProcess status comparison", 1050, 760);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.28);
    c->SetTopMargin(0.07);
    c->SetLogy();
    c->SetTicks();

    auto hNoAbs = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(noAbs.hStatus->Clone("h_status_noabs_draw")));
    auto hAbs = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hStatus->Clone("h_status_abs_draw")));
    StyleHist(hNoAbs.get(), kAzure + 2, 20);
    StyleHist(hAbs.get(), kOrange + 7, 24);
    hNoAbs->SetFillStyle(0);
    hAbs->SetFillStyle(0);
    hNoAbs->SetLineWidth(3);
    hAbs->SetLineWidth(3);
    hNoAbs->SetMarkerSize(1.15);
    hAbs->SetMarkerSize(1.15);
    hNoAbs->SetTitle("TMCProcess content in MCHypCands");
    hNoAbs->GetYaxis()->SetTitle("Entries");
    hNoAbs->GetXaxis()->SetTitle("TMCProcess category");
    hNoAbs->GetXaxis()->SetTitleSize(0.038);
    hNoAbs->GetXaxis()->SetTitleOffset(2.55);
    hNoAbs->GetXaxis()->SetLabelSize(0.030);
    hNoAbs->GetXaxis()->SetLabelOffset(0.018);
    hNoAbs->SetMinimum(0.5);
    hNoAbs->SetMaximum(2.5 * std::max(hNoAbs->GetMaximum(), hAbs->GetMaximum()));
    hNoAbs->LabelsOption("v", "X");
    hNoAbs->Draw("HIST");
    hAbs->Draw("HIST SAME");
    hNoAbs->Draw("P SAME");
    hAbs->Draw("P SAME");

    auto leg = std::make_unique<TLegend>(0.58, 0.72, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.035);
    leg->AddEntry(hNoAbs.get(), noAbs.title.c_str(), "lp");
    leg->AddEntry(hAbs.get(), abs.title.c_str(), "lp");
    leg->Draw();

    c->SaveAs(Form("%s/status_code_comparison.pdf", outDir.c_str()));
}

void DrawPtAbsorption(const SampleQA& noAbs, const SampleQA& abs, const std::string& outDir)
{
    auto fracAbs23 = MakeFraction(abs.hPtAbs23.get(), abs.hPtDen23.get(), "h_frac_abs23", ";p_{T} (GeV/c);Material-loss fraction");
    auto fracAbsIncl = MakeFraction(abs.hPtAbsInclusive.get(), abs.hPtDenInclusive.get(), "h_frac_absinclusive", ";p_{T} (GeV/c);Material-loss fraction");
    auto fracNoAbs23 = MakeFraction(noAbs.hPtAbs23.get(), noAbs.hPtDen23.get(), "h_frac_noabs23", ";p_{T} (GeV/c);Material-loss fraction");

    auto c = std::make_unique<TCanvas>("c_pt_absorption_fraction", "pT absorption fraction", 950, 720);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetGridy();
    c->SetTicks();

    StyleHist(fracAbs23.get(), kRed + 1, 20);
    StyleHist(fracAbsIncl.get(), kOrange + 7, 24);
    StyleHist(fracNoAbs23.get(), kAzure + 2, 25);
    fracAbs23->SetTitle("TMCProcess material-loss proxy");
    fracAbs23->SetMinimum(0.0);
    fracAbs23->SetMaximum(0.08);
    fracAbs23->Draw("E");
    fracAbsIncl->Draw("E SAME");
    fracNoAbs23->Draw("E SAME");

    auto leg = std::make_unique<TLegend>(0.48, 0.66, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.033);
    leg->AddEntry(fracAbs23.get(), "G4 absorption: status 23", "lep");
    leg->AddEntry(fracAbsIncl.get(), "G4 absorption: status 13/20/23", "lep");
    leg->AddEntry(fracNoAbs23.get(), "No absorption reference: status 23", "lep");
    leg->Draw();

    auto note = std::make_unique<TPaveText>(0.16, 0.70, 0.46, 0.86, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextFont(42);
    note->SetTextAlign(12);
    note->SetTextSize(0.030);
    note->AddText("Numerator: selected TMCProcess status");
    note->AddText("Denominator: all generated hypertritons");
    note->Draw();

    c->SaveAs(Form("%s/absorption_fraction_vs_pt.pdf", outDir.c_str()));
}

void DrawPtSpectra(const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_pt_spectra", "survived and absorbed pt spectra", 950, 720);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetLogy();
    c->SetTicks();

    auto hSurv = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hPtSurvived->Clone("h_pt_survived_draw")));
    auto hAbs23 = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hPtAbs23->Clone("h_pt_abs23_draw")));
    auto hAbsIncl = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hPtAbsInclusive->Clone("h_pt_absincl_draw")));
    StyleHist(hSurv.get(), kAzure + 2, 20);
    StyleHist(hAbs23.get(), kRed + 1, 24);
    StyleHist(hAbsIncl.get(), kOrange + 7, 25);
    hSurv->SetTitle("Generated two-body vs material-loss candidates");
    hSurv->GetYaxis()->SetTitle("Entries");
    hSurv->SetMinimum(0.5);
    hSurv->SetMaximum(5.0 * std::max({hSurv->GetMaximum(), hAbs23->GetMaximum(), hAbsIncl->GetMaximum()}));
    hSurv->Draw("E");
    hAbs23->Draw("E SAME");
    hAbsIncl->Draw("E SAME");

    auto leg = std::make_unique<TLegend>(0.54, 0.68, 0.89, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.034);
    leg->AddEntry(hSurv.get(), "Survived two-body decay", "lep");
    leg->AddEntry(hAbs23.get(), "Material loss: status 23", "lep");
    leg->AddEntry(hAbsIncl.get(), "Material loss: status 13/20/23", "lep");
    leg->Draw();

    c->SaveAs(Form("%s/material_loss_pt_spectra.pdf", outDir.c_str()));
}

void DrawCtSurvived(const SampleQA& noAbs, const SampleQA& abs, const std::string& outDir)
{
    auto hNoAbs = MakeDensityClone(noAbs.hCtSurvived.get(), "h_ct_noabs_density");
    auto hAbs = MakeDensityClone(abs.hCtSurvived.get(), "h_ct_abs_density");
    if (hNoAbs->Integral("width") > 0.0) hNoAbs->Scale(1.0 / hNoAbs->Integral("width"));
    if (hAbs->Integral("width") > 0.0) hAbs->Scale(1.0 / hAbs->Integral("width"));
    auto fitNoAbs = FitCtDensity(hNoAbs.get(), "fit_ct_no_absorption", kAzure + 3);
    auto fitAbs = FitCtDensity(hAbs.get(), "fit_ct_g4_absorption", kOrange + 8);

    auto c = std::make_unique<TCanvas>("c_ct_survived", "survived two-body ct", 950, 720);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetLogy();
    c->SetTicks();

    StyleHist(hNoAbs.get(), kAzure + 2, 20);
    StyleHist(hAbs.get(), kOrange + 7, 24);
    hNoAbs->SetTitle("c#tau sanity check for survived two-body decays");
    hNoAbs->GetYaxis()->SetTitle("Normalized entries / cm");
    hNoAbs->GetXaxis()->SetTitle("c#tau (cm)");
    hNoAbs->SetMinimum(3e-6);
    hNoAbs->SetMaximum(3.0 * std::max(hNoAbs->GetMaximum(), hAbs->GetMaximum()));
    hNoAbs->Draw("E");
    hAbs->Draw("E SAME");
    if (fitNoAbs.func) fitNoAbs.func->Draw("SAME");
    if (fitAbs.func) fitAbs.func->Draw("SAME");

    auto leg = std::make_unique<TLegend>(0.50, 0.64, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.031);
    leg->AddEntry(hNoAbs.get(), noAbs.title.c_str(), "lep");
    leg->AddEntry(hAbs.get(), abs.title.c_str(), "lep");
    if (fitNoAbs.func) {
        leg->AddEntry(fitNoAbs.func.get(), Form("No absorption fit: #tau = %.1f #pm %.1f ps", fitNoAbs.tauPs, fitNoAbs.tauPsErr), "l");
    }
    if (fitAbs.func) {
        leg->AddEntry(fitAbs.func.get(), Form("G4 absorption fit: #tau = %.1f #pm %.1f ps", fitAbs.tauPs, fitAbs.tauPsErr), "l");
    }
    leg->Draw();

    auto note = std::make_unique<TPaveText>(0.16, 0.17, 0.48, 0.31, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextAlign(12);
    note->SetTextFont(42);
    note->SetTextSize(0.030);
    note->AddText("Fit model: A exp(-c#tau / c#tau_{0})");
    note->AddText("Fit range: 0.5 < c#tau < 40 cm");
    if (fitNoAbs.ndf > 0) note->AddText(Form("No absorption #chi^{2}/ndf = %.1f/%d", fitNoAbs.chi2, fitNoAbs.ndf));
    if (fitAbs.ndf > 0) note->AddText(Form("G4 absorption #chi^{2}/ndf = %.1f/%d", fitAbs.chi2, fitAbs.ndf));
    note->Draw();

    c->SaveAs(Form("%s/survived_twobody_ct_comparison.pdf", outDir.c_str()));
}

void DrawAbsorptionFractionVsCtPtBins(const SampleQA& abs,
                                      const std::vector<std::unique_ptr<TH1D>>& numerators,
                                      const std::string& numeratorLabel,
                                      const std::string& plotTitle,
                                      const std::string& outputName,
                                      const std::string& outDir)
{
    const std::vector<Color_t> colors{kAzure + 2, kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2, kBlack};
    const std::vector<Style_t> markers{20, 21, 22, 23, 24, 25, 26};
    std::vector<std::unique_ptr<TH1D>> fractions;
    fractions.reserve(numerators.size());

    double yMax = 0.015;
    for (size_t ipt = 0; ipt < numerators.size(); ++ipt) {
        auto frac = MakeFraction(numerators[ipt].get(), abs.hCtDen23Pt[ipt].get(),
                                 Form("h_abs_fraction_vs_ct_pt_%zu", ipt),
                                 ";stored c#tau (cm);Material-loss fraction");
        StyleHist(frac.get(), colors[ipt % colors.size()], markers[ipt % markers.size()]);
        frac->SetLineWidth(2);
        frac->SetMarkerSize(0.8);
        yMax = std::max(yMax, 1.35 * frac->GetMaximum());
        fractions.push_back(std::move(frac));
    }

    auto c = std::make_unique<TCanvas>(Form("c_%s", outputName.c_str()), "absorption fraction vs ct in pt bins", 1000, 760);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetGridy();
    c->SetTicks();

    auto frame = std::make_unique<TH1D>("h_abs_fraction_vs_ct_frame", ";stored c#tau (cm);Material-loss fraction", 1, kCtBins.front(), kCtBins.back());
    frame->SetStats(0);
    frame->SetMinimum(0.0);
    frame->SetMaximum(yMax);
    frame->GetXaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleOffset(1.25);
    frame->SetTitle(plotTitle.c_str());
    frame->Draw();

    auto leg = std::make_unique<TLegend>(0.56, 0.56, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.030);

    for (size_t ipt = 0; ipt < fractions.size(); ++ipt) {
        fractions[ipt]->Draw("E SAME");
        leg->AddEntry(fractions[ipt].get(), Form("%.1f #leq p_{T} < %.1f GeV/c", kPtBins[ipt], kPtBins[ipt + 1]), "lep");
    }
    leg->Draw();

    auto note = std::make_unique<TPaveText>(0.16, 0.76, 0.52, 0.88, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextAlign(12);
    note->SetTextFont(42);
    note->SetTextSize(0.029);
    note->AddText(Form("Numerator: %s", numeratorLabel.c_str()));
    note->AddText("Denominator: all generated hypertritons");
    note->Draw();

    c->SaveAs(Form("%s/%s.pdf", outDir.c_str(), outputName.c_str()));
}

struct ProxyAbsorptionQA {
    std::vector<std::unique_ptr<TH1D>> hCtAbsPt;
    std::vector<std::unique_ptr<TH1D>> hCtDenPt;
    std::unique_ptr<TH1D> hAbsRxy;
    std::unique_ptr<TH1D> hAbsLength;
    std::unique_ptr<TH1D> hAbsX;
    std::unique_ptr<TH1D> hAbsY;
    std::unique_ptr<TH2D> hAbsXY;
};

ProxyAbsorptionQA ReadHe3ProxyAbsorptionTree(const std::string& filePath,
                                             const std::string& treeName = "he3candidates")
{
    ProxyAbsorptionQA out;
    out.hAbsRxy = MakeFixedHist1D("h_proxy_abs_rxy", ";R_{xy} = #sqrt{absoX^{2}+absoY^{2}} (cm);Entries", 160, 0.0, 400.0);
    out.hAbsLength = MakeFixedHist1D("h_proxy_abs_length", ";length = #sqrt{absoX^{2}+absoY^{2}+absoZ^{2}} (cm);Entries", 200, 0.0, 1000.0);
    out.hAbsX = MakeFixedHist1D("h_proxy_abs_x", ";absoX (cm);Entries", 160, -400.0, 400.0);
    out.hAbsY = MakeFixedHist1D("h_proxy_abs_y", ";absoY (cm);Entries", 160, -400.0, 400.0);
    out.hAbsXY = std::make_unique<TH2D>("h_proxy_abs_xy", ";absoX (cm);absoY (cm)", 120, -400.0, 400.0, 120, -400.0, 400.0);
    out.hAbsXY->SetDirectory(nullptr);
    for (size_t ipt = 0; ipt + 1 < kPtBins.size(); ++ipt) {
        const std::string tag = Form("pt_%g_%g", kPtBins[ipt], kPtBins[ipt + 1]);
        out.hCtAbsPt.push_back(MakeHist1D("h_proxy_ct_abs_" + tag, ";absorption c#tau (cm);Entries", kCtBins));
        out.hCtDenPt.push_back(MakeHist1D("h_proxy_ct_den_" + tag, ";absorption c#tau (cm);Entries", kCtBins));
    }

    TFile input(filePath.c_str(), "READ");
    if (input.IsZombie()) {
        std::cerr << "[Error] Cannot open proxy absorption file " << filePath << "\n";
        return out;
    }
    auto* tree = dynamic_cast<TTree*>(input.Get(treeName.c_str()));
    if (!tree) {
        std::cerr << "[Error] Cannot find tree " << treeName << " in " << filePath << "\n";
        return out;
    }

    Float_t pt = 0.f;
    Float_t eta = 0.f;
    Float_t phi = 0.f;
    Float_t absoX = 0.f;
    Float_t absoY = 0.f;
    Float_t absoZ = 0.f;
    Int_t pdg = 0;
    tree->SetBranchAddress("pt", &pt);
    tree->SetBranchAddress("eta", &eta);
    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("absoX", &absoX);
    tree->SetBranchAddress("absoY", &absoY);
    tree->SetBranchAddress("absoZ", &absoZ);
    tree->SetBranchAddress("pdg", &pdg);
    std::cout << "[ReadHe3ProxyAbsorptionTree] Sampling He3 proxy entries with P(survive to absoCt) = exp(-absoCt/(253 ps*c)), using m_H3L.\n";

    TRandom3 rng(0);
    const Long64_t entries = tree->GetEntries();
    for (Long64_t i = 0; i < entries; ++i) {
        tree->GetEntry(i);
        const int ptBin = FindPtBin(pt);
        if (ptBin < 0) continue;

        const double p = static_cast<double>(pt) * std::cosh(static_cast<double>(eta));
        const double absoL = std::sqrt(absoX * absoX + absoY * absoY + absoZ * absoZ);
        const double absoCt = (p > 0.0) ? absoL * kH3LMass / p : 1e30;
        if (!std::isfinite(absoCt) || absoCt < 0.0) continue;

        const double absoRxy = std::sqrt(absoX * absoX + absoY * absoY);
        const bool inCtRange = absoCt >= kCtBins.front() && absoCt < kCtBins.back();
        if (inCtRange) out.hCtDenPt[static_cast<size_t>(ptBin)]->Fill(absoCt);

        const double survivalProbability = std::exp(-absoCt / kOriginalTauCt);
        if (rng.Uniform() >= survivalProbability) continue;

        if (std::isfinite(absoRxy) && absoRxy >= 0.0) out.hAbsRxy->Fill(absoRxy);
        if (std::isfinite(absoL) && absoL >= 0.0) out.hAbsLength->Fill(absoL);
        if (std::isfinite(static_cast<double>(absoX)) && std::isfinite(static_cast<double>(absoY))) {
            out.hAbsX->Fill(absoX);
            out.hAbsY->Fill(absoY);
            out.hAbsXY->Fill(absoX, absoY);
        }
        if (inCtRange) out.hCtAbsPt[static_cast<size_t>(ptBin)]->Fill(absoCt);
    }
    return out;
}

void DrawAbsorbedRadiusDistributions(const SampleQA& abs,
                                     const ProxyAbsorptionQA* proxy,
                                     const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_absorbed_radius_distributions", "absorbed radius distributions", 1100, 520);
    c->Divide(2, 1);

    auto hMcRxy = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAbsVertexRxy->Clone("h_mc_abs_rxy_draw")));
    auto hMcLength = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAbsVertexLength->Clone("h_mc_abs_length_draw")));
    std::unique_ptr<TH1D> hProxyRxy;
    std::unique_ptr<TH1D> hProxyLength;
    if (proxy && proxy->hAbsRxy && proxy->hAbsLength) {
        hProxyRxy = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsRxy->Clone("h_proxy_abs_rxy_draw")));
        hProxyLength = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsLength->Clone("h_proxy_abs_length_draw")));
    }

    auto normalize = [](TH1D* h) {
        if (h && h->Integral() > 0.0) h->Scale(1.0 / h->Integral());
    };
    normalize(hMcRxy.get());
    normalize(hMcLength.get());
    normalize(hProxyRxy.get());
    normalize(hProxyLength.get());

    c->cd(1);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.04);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogy();
    StyleHist(hMcRxy.get(), kRed + 1, 20);
    hMcRxy->SetTitle("Absorbed-candidate transverse radius");
    hMcRxy->GetYaxis()->SetTitle("Normalized entries");
    hMcRxy->SetMinimum(1e-5);
    hMcRxy->SetMaximum(2.0 * std::max(hMcRxy->GetMaximum(), hProxyRxy ? hProxyRxy->GetMaximum() : 0.0));
    hMcRxy->Draw("HIST");
    if (hProxyRxy) {
        StyleHist(hProxyRxy.get(), kAzure + 2, 24);
        hProxyRxy->SetLineWidth(3);
        hProxyRxy->Draw("HIST SAME");
    }
    auto legRxy = std::make_unique<TLegend>(0.46, 0.72, 0.90, 0.88);
    legRxy->SetBorderSize(0);
    legRxy->SetFillStyle(0);
    legRxy->SetTextFont(42);
    legRxy->SetTextSize(0.033);
    legRxy->AddEntry(hMcRxy.get(), "MC AO2D status 23: fGenX/YDecVtx", "l");
    if (hProxyRxy) legRxy->AddEntry(hProxyRxy.get(), "He3 proxy all entries: absoX/Y", "l");
    legRxy->Draw();

    c->cd(2);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.04);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogy();
    StyleHist(hMcLength.get(), kRed + 1, 20);
    hMcLength->SetTitle("Absorbed-candidate 3D path length");
    hMcLength->GetYaxis()->SetTitle("Normalized entries");
    hMcLength->SetMinimum(1e-5);
    hMcLength->SetMaximum(2.0 * std::max(hMcLength->GetMaximum(), hProxyLength ? hProxyLength->GetMaximum() : 0.0));
    hMcLength->Draw("HIST");
    if (hProxyLength) {
        StyleHist(hProxyLength.get(), kAzure + 2, 24);
        hProxyLength->SetLineWidth(3);
        hProxyLength->Draw("HIST SAME");
    }
    auto legLength = std::make_unique<TLegend>(0.44, 0.72, 0.90, 0.88);
    legLength->SetBorderSize(0);
    legLength->SetFillStyle(0);
    legLength->SetTextFont(42);
    legLength->SetTextSize(0.033);
    legLength->AddEntry(hMcLength.get(), "MC AO2D status 23: fGenX/Y/ZDecVtx", "l");
    if (hProxyLength) legLength->AddEntry(hProxyLength.get(), "He3 proxy all entries: absoX/Y/Z", "l");
    legLength->Draw();

    c->SaveAs(Form("%s/absorbed_radius_length_distributions.pdf", outDir.c_str()));
}

void DrawAbsorbedXYDistributions(const SampleQA& abs,
                                 const ProxyAbsorptionQA* proxy,
                                 const std::string& outDir,
                                 const std::string& outputName = "absorbed_xy_distributions.pdf",
                                 bool zoomMcXY = true)
{
    auto c = std::make_unique<TCanvas>("c_absorbed_xy_distributions", "absorbed xy distributions", 1100, 900);
    c->Divide(2, 2);

    auto hMcX = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAbsVertexX->Clone("h_mc_abs_x_draw")));
    auto hMcY = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAbsVertexY->Clone("h_mc_abs_y_draw")));
    auto hMcXY = std::unique_ptr<TH2D>(dynamic_cast<TH2D*>(abs.hAbsVertexXY->Clone("h_mc_abs_xy_draw")));
    std::unique_ptr<TH1D> hProxyX;
    std::unique_ptr<TH1D> hProxyY;
    std::unique_ptr<TH2D> hProxyXY;
    if (proxy && proxy->hAbsX && proxy->hAbsY && proxy->hAbsXY) {
        hProxyX = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsX->Clone("h_proxy_abs_x_draw")));
        hProxyY = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsY->Clone("h_proxy_abs_y_draw")));
        hProxyXY = std::unique_ptr<TH2D>(dynamic_cast<TH2D*>(proxy->hAbsXY->Clone("h_proxy_abs_xy_draw")));
    }

    auto normalize = [](TH1D* h) {
        if (h && h->Integral() > 0.0) h->Scale(1.0 / h->Integral());
    };
    normalize(hMcX.get());
    normalize(hMcY.get());
    normalize(hProxyX.get());
    normalize(hProxyY.get());

    auto drawOverlay = [](TH1D* mc, TH1D* proxyHist, const char* title, const char* mcLabel, const char* proxyLabel) {
        gPad->SetLeftMargin(0.13);
        gPad->SetRightMargin(0.04);
        gPad->SetBottomMargin(0.13);
        gPad->SetTopMargin(0.08);
        gPad->SetTicks();
        gPad->SetLogy();
        StyleHist(mc, kRed + 1, 20);
        mc->SetTitle(title);
        mc->GetYaxis()->SetTitle("Normalized entries");
        mc->SetMinimum(1e-5);
        mc->SetMaximum(2.0 * std::max(mc->GetMaximum(), proxyHist ? proxyHist->GetMaximum() : 0.0));
        mc->Draw("HIST");
        if (proxyHist) {
            StyleHist(proxyHist, kAzure + 2, 24);
            proxyHist->SetLineWidth(3);
            proxyHist->Draw("HIST SAME");
        }
        auto* leg = new TLegend(0.48, 0.72, 0.91, 0.88);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextFont(42);
        leg->SetTextSize(0.031);
        leg->AddEntry(mc, mcLabel, "l");
        if (proxyHist) leg->AddEntry(proxyHist, proxyLabel, "l");
        leg->Draw();

        auto* stats = new TPaveText(0.16, 0.70, 0.45, 0.86, "NDC");
        stats->SetBorderSize(0);
        stats->SetFillStyle(0);
        stats->SetTextFont(42);
        stats->SetTextAlign(12);
        stats->SetTextSize(0.030);
        stats->AddText(Form("MC entries: %.0f", mc->GetEntries()));
        if (proxyHist) {
            stats->AddText(Form("He3 entries: %.0f", proxyHist->GetEntries()));
        }
        stats->Draw();
    };

    c->cd(1);
    drawOverlay(hMcX.get(), hProxyX.get(), "Absorbed-candidate x distribution",
                "MC AO2D status 23: x_{gen} + x_{PV}", "He3 proxy absorbed: #tau=253 ps, m_{H3L}");

    c->cd(2);
    drawOverlay(hMcY.get(), hProxyY.get(), "Absorbed-candidate y distribution",
                "MC AO2D status 23: y_{gen} + y_{PV}", "He3 proxy absorbed: #tau=253 ps, m_{H3L}");

    c->cd(3);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogz();
    hMcXY->SetStats(0);
    hMcXY->SetTitle("MC AO2D status 23: absolute generated decay vertex");
    if (zoomMcXY) {
        hMcXY->GetXaxis()->SetRangeUser(-40.0, 40.0);
        hMcXY->GetYaxis()->SetRangeUser(-40.0, 40.0);
    }
    hMcXY->GetZaxis()->SetTitle("Entries");
    hMcXY->SetMinimum(1.0);
    hMcXY->Draw("COLZ");

    c->cd(4);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogz();
    if (hProxyXY) {
        hProxyXY->SetStats(0);
        hProxyXY->SetTitle("He3 proxy absorbed: sampled c#tau with #tau=253 ps, m_{H3L}");
        hProxyXY->GetZaxis()->SetTitle("Entries");
        hProxyXY->SetMinimum(1.0);
        hProxyXY->Draw("COLZ");
    } else {
        auto frame = std::make_unique<TH2D>("h_abs_xy_empty_proxy_frame", ";absoX (cm);absoY (cm)", 1, -400.0, 400.0, 1, -400.0, 400.0);
        frame->SetStats(0);
        frame->SetTitle("He3 proxy all entries: absoX vs absoY");
        frame->Draw();
    }

    c->SaveAs(Form("%s/%s", outDir.c_str(), outputName.c_str()));
}

void DrawReconstructedXYDistribution(const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_reconstructed_xy_distribution", "reconstructed xy distribution", 1000, 840);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.16);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetTicks();
    c->SetLogz();

    auto hRecoXY = std::unique_ptr<TH2D>(dynamic_cast<TH2D*>(abs.hRecoVertexXY->Clone("h_reco_vertex_xy_draw")));
    hRecoXY->SetStats(0);
    hRecoXY->SetTitle("G4 absorption sample: reconstructed absolute generated decay vertex");
    hRecoXY->GetXaxis()->SetTitle("absolute x_{decay} (cm)");
    hRecoXY->GetYaxis()->SetTitle("absolute y_{decay} (cm)");
    hRecoXY->GetZaxis()->SetTitle("Entries");
    hRecoXY->GetXaxis()->SetTitleSize(0.043);
    hRecoXY->GetYaxis()->SetTitleSize(0.043);
    hRecoXY->GetZaxis()->SetTitleSize(0.043);
    hRecoXY->GetYaxis()->SetTitleOffset(1.25);
    hRecoXY->SetMinimum(1.0);
    hRecoXY->Draw("COLZ");

    auto note = std::make_unique<TPaveText>(0.15, 0.78, 0.50, 0.89, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextFont(42);
    note->SetTextAlign(12);
    note->SetTextSize(0.032);
    note->AddText("fIsReco == true");
    note->AddText(Form("Entries: %.0f", hRecoXY->GetEntries()));
    note->Draw();

    c->SaveAs(Form("%s/reconstructed_xy_distribution.pdf", outDir.c_str()));
}

void DrawAllMCVsHe3ProxyXYDistributions(const SampleQA& abs,
                                        const ProxyAbsorptionQA* proxy,
                                        const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_all_mc_vs_he3_proxy_xy", "all MC vs He3 proxy xy", 1100, 900);
    c->Divide(2, 2);

    auto hMcAllX = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAllVertexX->Clone("h_mc_all_x_draw")));
    auto hMcAllY = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(abs.hAllVertexY->Clone("h_mc_all_y_draw")));
    auto hMcAllXY = std::unique_ptr<TH2D>(dynamic_cast<TH2D*>(abs.hAllVertexXY->Clone("h_mc_all_xy_draw")));
    std::unique_ptr<TH1D> hProxyX;
    std::unique_ptr<TH1D> hProxyY;
    std::unique_ptr<TH2D> hProxyXY;
    if (proxy && proxy->hAbsX && proxy->hAbsY && proxy->hAbsXY) {
        hProxyX = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsX->Clone("h_proxy_all_x_draw")));
        hProxyY = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(proxy->hAbsY->Clone("h_proxy_all_y_draw")));
        hProxyXY = std::unique_ptr<TH2D>(dynamic_cast<TH2D*>(proxy->hAbsXY->Clone("h_proxy_all_xy_draw")));
    }

    auto normalize = [](TH1D* h) {
        if (h && h->Integral() > 0.0) h->Scale(1.0 / h->Integral());
    };
    normalize(hMcAllX.get());
    normalize(hMcAllY.get());
    normalize(hProxyX.get());
    normalize(hProxyY.get());

    auto drawOverlay = [](TH1D* mc, TH1D* proxyHist, const char* title, const char* mcLabel, const char* proxyLabel) {
        gPad->SetLeftMargin(0.13);
        gPad->SetRightMargin(0.04);
        gPad->SetBottomMargin(0.13);
        gPad->SetTopMargin(0.08);
        gPad->SetTicks();
        gPad->SetLogy();
        StyleHist(mc, kRed + 1, 20);
        mc->SetTitle(title);
        mc->GetYaxis()->SetTitle("Normalized entries");
        mc->SetMinimum(1e-5);
        mc->SetMaximum(2.0 * std::max(mc->GetMaximum(), proxyHist ? proxyHist->GetMaximum() : 0.0));
        mc->Draw("HIST");
        if (proxyHist) {
            StyleHist(proxyHist, kAzure + 2, 24);
            proxyHist->SetLineWidth(3);
            proxyHist->Draw("HIST SAME");
        }
        auto* leg = new TLegend(0.48, 0.72, 0.91, 0.88);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextFont(42);
        leg->SetTextSize(0.031);
        leg->AddEntry(mc, mcLabel, "l");
        if (proxyHist) leg->AddEntry(proxyHist, proxyLabel, "l");
        leg->Draw();

        auto* stats = new TPaveText(0.16, 0.70, 0.45, 0.86, "NDC");
        stats->SetBorderSize(0);
        stats->SetFillStyle(0);
        stats->SetTextFont(42);
        stats->SetTextAlign(12);
        stats->SetTextSize(0.030);
        stats->AddText(Form("MC entries: %.0f", mc->GetEntries()));
        if (proxyHist) {
            stats->AddText(Form("He3 entries: %.0f", proxyHist->GetEntries()));
        }
        stats->Draw();
    };

    c->cd(1);
    drawOverlay(hMcAllX.get(), hProxyX.get(), "All-candidate x distribution",
                "MC AO2D all entries: x_{gen} + x_{PV}", "He3 proxy all entries: absoX");

    c->cd(2);
    drawOverlay(hMcAllY.get(), hProxyY.get(), "All-candidate y distribution",
                "MC AO2D all entries: y_{gen} + y_{PV}", "He3 proxy all entries: absoY");

    c->cd(3);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogz();
    hMcAllXY->SetStats(0);
    hMcAllXY->SetTitle("MC AO2D all entries: absolute generated decay vertex");
    hMcAllXY->GetZaxis()->SetTitle("Entries");
    hMcAllXY->SetMinimum(1.0);
    hMcAllXY->Draw("COLZ");

    c->cd(4);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.08);
    gPad->SetTicks();
    gPad->SetLogz();
    if (hProxyXY) {
        hProxyXY->SetStats(0);
        hProxyXY->SetTitle("He3 proxy all entries: absoX vs absoY");
        hProxyXY->GetZaxis()->SetTitle("Entries");
        hProxyXY->SetMinimum(1.0);
        hProxyXY->Draw("COLZ");
    } else {
        auto frame = std::make_unique<TH2D>("h_all_mc_vs_proxy_empty_frame", ";absoX (cm);absoY (cm)", 1, -400.0, 400.0, 1, -400.0, 400.0);
        frame->SetStats(0);
        frame->SetTitle("He3 proxy all entries: absoX vs absoY");
        frame->Draw();
    }

    c->SaveAs(Form("%s/all_mc_vs_he3_proxy_xy_distributions.pdf", outDir.c_str()));
}

void DrawHe3ProxyAbsorptionFractionVsCtPtBins(const ProxyAbsorptionQA& proxy, const std::string& outDir)
{
    const std::vector<Color_t> colors{kAzure + 2, kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2, kBlack};
    const std::vector<Style_t> markers{20, 21, 22, 23, 24, 25, 26};
    std::vector<std::unique_ptr<TH1D>> fractions;
    fractions.reserve(proxy.hCtAbsPt.size());

    double yMax = 0.015;
    for (size_t ipt = 0; ipt < proxy.hCtAbsPt.size(); ++ipt) {
        auto frac = MakeFraction(proxy.hCtAbsPt[ipt].get(), proxy.hCtDenPt[ipt].get(),
                                 Form("h_proxy_abs_fraction_vs_ct_pt_%zu", ipt),
                                 ";generated c#tau (cm);Absorbed fraction");
        StyleHist(frac.get(), colors[ipt % colors.size()], markers[ipt % markers.size()]);
        frac->SetLineWidth(2);
        frac->SetMarkerSize(0.8);
        yMax = std::max(yMax, 1.35 * frac->GetMaximum());
        fractions.push_back(std::move(frac));
    }

    auto c = std::make_unique<TCanvas>("c_he3_proxy_absorption_fraction_vs_ct_ptbins", "He3 proxy absorption fraction vs ct", 1000, 760);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetGridy();
    c->SetTicks();

    auto frame = std::make_unique<TH1D>("h_he3_proxy_abs_fraction_vs_ct_frame", ";generated c#tau (cm);Absorbed fraction", 1, kCtBins.front(), kCtBins.back());
    frame->SetStats(0);
    frame->SetMinimum(0.0);
    frame->SetMaximum(yMax);
    frame->GetXaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleOffset(1.25);
    frame->SetTitle("He3 proxy absorption tree: absorbed fraction vs generated c#tau");
    frame->Draw();

    auto leg = std::make_unique<TLegend>(0.56, 0.56, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.030);
    for (size_t ipt = 0; ipt < fractions.size(); ++ipt) {
        fractions[ipt]->Draw("E SAME");
        leg->AddEntry(fractions[ipt].get(), Form("%.1f #leq p_{T} < %.1f GeV/c", kPtBins[ipt], kPtBins[ipt + 1]), "lep");
    }
    leg->Draw();

    auto note = std::make_unique<TPaveText>(0.16, 0.76, 0.52, 0.88, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextAlign(12);
    note->SetTextFont(42);
    note->SetTextSize(0.029);
    note->AddText("Numerator: abso c#tau #leq generated c#tau");
    note->AddText("Denominator: all generated He3 proxies");
    note->Draw();

    c->SaveAs(Form("%s/he3_proxy_absorption_fraction_vs_ct_ptbins_x3.pdf", outDir.c_str()));
}

void DrawVertexSanity(const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_abs_vertex_sanity", "absorbed vertex sanity", 950, 720);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetLogy();
    c->SetTicks();

    StyleHist(abs.hAbsVertexLogR.get(), kRed + 1, 20);
    abs.hAbsVertexLogR->SetTitle("Stored decay vector for status 23 candidates");
    abs.hAbsVertexLogR->Draw("HIST");

    auto note = std::make_unique<TPaveText>(0.16, 0.66, 0.58, 0.86, "NDC");
    note->SetBorderSize(0);
    note->SetFillStyle(0);
    note->SetTextFont(42);
    note->SetTextAlign(12);
    note->SetTextSize(0.032);
    note->AddText("Status 23 entries do not have a He3 decay daughter.");
    note->AddText("Stored fGen*DecVtx is therefore not a valid absorption point.");
    note->AddText("Use TMCProcess for material-loss QA; need material path for #sigma.");
    note->Draw();

    c->SaveAs(Form("%s/absorbed_vertex_sanity_status23.pdf", outDir.c_str()));
}

void DrawStatusVsPt(const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_status_vs_pt", "status vs pt", 980, 760);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.17);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.07);
    c->SetTicks();
    gStyle->SetPaintTextFormat(".0f");

    abs.hStatusVsPt->SetStats(0);
    abs.hStatusVsPt->SetTitle("G4 absorption sample: TMCProcess vs generated p_{T}");
    abs.hStatusVsPt->GetZaxis()->SetTitle("Entries");
    abs.hStatusVsPt->Draw("COLZ TEXT");
    c->SaveAs(Form("%s/status_vs_pt_heatmap.pdf", outDir.c_str()));
}

void DrawSummary(const SampleQA& noAbs, const SampleQA& abs, const std::string& outDir)
{
    auto c = std::make_unique<TCanvas>("c_absorption_mc_summary", "summary", 1000, 720);
    c->SetLeftMargin(0.08);
    c->SetRightMargin(0.04);
    c->SetTopMargin(0.05);
    c->SetBottomMargin(0.06);

    auto text = std::make_unique<TPaveText>(0.08, 0.08, 0.94, 0.94, "NDC");
    text->SetBorderSize(0);
    text->SetFillStyle(0);
    text->SetTextFont(42);
    text->SetTextAlign(12);
    text->SetTextSize(0.030);
    text->AddText("Hypertriton absorption QA from MCHypCands");
    text->AddText("");
    text->AddText(Form("No-absorption reference: entries=%lld, two-body=%lld, reco/status0=%lld, status23=%lld",
                       noAbs.nAll, noAbs.nTwoBody, noAbs.nReco, noAbs.statusCounts.count(23) ? noAbs.statusCounts.at(23) : 0));
    text->AddText(Form("G4 absorption sample: entries=%lld, two-body=%lld, reco/status0=%lld", abs.nAll, abs.nTwoBody, abs.nReco));
    text->AddText(Form("G4 material-loss candidates: status23=%lld, status13/20/23=%lld", abs.nAbs23, abs.nAbsInclusive));
    text->AddText(Form("G4 non-two-body decay-like entries: status4=%lld", abs.nNonTwoDecay));
    text->AddText("");
    text->AddText("Definition used in QA:");
    text->AddText("  material-loss fraction numerator = selected TMCProcess status");
    text->AddText("  material-loss fraction denominator = all generated hypertritons in the same bin");
    text->AddText("  conservative selected status = fStatusCode == 23 (kPHInhelastic)");
    text->AddText("  inclusive selected status = fStatusCode in {13,20,23}");
    text->AddText("  fIsTwoBodyDecay is used only as a decay-channel sanity check");
    text->AddText("");
    text->AddText("Important limitation:");
    text->AddText("  For status23 entries processMC did not find a He3 decay daughter; fGen*DecVtx is not a valid absorption point.");
    text->AddText("  The current AO2D supports a material-loss probability / survival correction from TMCProcess.");
    text->AddText("  Absolute #sigma_{abs} still needs material path length or #int n dl information.");
    text->Draw();

    c->SaveAs(Form("%s/absorption_mc_qa_summary.pdf", outDir.c_str()));
}

void WriteResults(const SampleQA& noAbs, const SampleQA& abs, const std::string& outDir)
{
    TFile out(Form("%s/absorption_mc_qa_results.root", outDir.c_str()), "RECREATE");
    noAbs.hStatus->Write();
    noAbs.hPtSurvived->Write();
    noAbs.hPtAbs23->Write();
    noAbs.hPtAbsInclusive->Write();
    noAbs.hPtDen23->Write();
    noAbs.hPtDenInclusive->Write();
    noAbs.hCtSurvived->Write();
    for (auto& h : noAbs.hCtAbs23Pt) h->Write();
    for (auto& h : noAbs.hCtAbsInclusivePt) h->Write();
    for (auto& h : noAbs.hCtDen23Pt) h->Write();
    noAbs.hAbsVertexLogR->Write();
    noAbs.hAbsVertexRxy->Write();
    noAbs.hAbsVertexLength->Write();
    noAbs.hAbsVertexX->Write();
    noAbs.hAbsVertexY->Write();
    noAbs.hAbsVertexXY->Write();
    noAbs.hAllVertexX->Write();
    noAbs.hAllVertexY->Write();
    noAbs.hAllVertexXY->Write();
    noAbs.hRecoVertexX->Write();
    noAbs.hRecoVertexY->Write();
    noAbs.hRecoVertexXY->Write();
    noAbs.hStatusVsPt->Write();
    abs.hStatus->Write();
    abs.hPtSurvived->Write();
    abs.hPtAbs23->Write();
    abs.hPtAbsInclusive->Write();
    abs.hPtDen23->Write();
    abs.hPtDenInclusive->Write();
    abs.hCtSurvived->Write();
    for (auto& h : abs.hCtAbs23Pt) h->Write();
    for (auto& h : abs.hCtAbsInclusivePt) h->Write();
    for (auto& h : abs.hCtDen23Pt) h->Write();
    abs.hAbsVertexLogR->Write();
    abs.hAbsVertexRxy->Write();
    abs.hAbsVertexLength->Write();
    abs.hAbsVertexX->Write();
    abs.hAbsVertexY->Write();
    abs.hAbsVertexXY->Write();
    abs.hAllVertexX->Write();
    abs.hAllVertexY->Write();
    abs.hAllVertexXY->Write();
    abs.hRecoVertexX->Write();
    abs.hRecoVertexY->Write();
    abs.hRecoVertexXY->Write();
    abs.hStatusVsPt->Write();

    auto fracAbs23 = MakeFraction(abs.hPtAbs23.get(), abs.hPtDen23.get(), "h_absorption_fraction_status23", ";p_{T} (GeV/c);Material-loss fraction");
    auto fracAbsIncl = MakeFraction(abs.hPtAbsInclusive.get(), abs.hPtDenInclusive.get(), "h_absorption_fraction_status13_20_23", ";p_{T} (GeV/c);Material-loss fraction");
    fracAbs23->Write();
    fracAbsIncl->Write();
    out.Close();
}

} // namespace

void DrawAbsorptionCrossSectionMCQA(
    const char* absorptionFile = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
    const char* noAbsorptionFile = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11/NCrossedRows/AO2D_CustomV0s.root",
    const char* outputDir = "../../../Outputs/CrossSection/Plotting/AbsorptionMCQA",
    const char* he3ProxyAbsorptionFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x3.root")
{
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    gStyle->SetNumberContours(255);
    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetLegendFont(42);

    const std::string outDir = ResolveOutputPath(outputDir ? outputDir : "");
    EnsureDir(outDir);

    SampleQA noAbs;
    noAbs.name = "no_absorption";
    noAbs.title = "No hypertriton absorption";
    noAbs.color = kAzure + 2;

    SampleQA abs;
    abs.name = "g4_absorption";
    abs.title = "G4 material absorption";
    abs.color = kOrange + 7;

    std::cout << "Reading no-absorption sample:\n  " << noAbsorptionFile << "\n";
    ReadSample(noAbsorptionFile, noAbs);
    std::cout << "Reading absorption sample:\n  " << absorptionFile << "\n";
    ReadSample(absorptionFile, abs);

    DrawSummary(noAbs, abs, outDir);
    DrawStatusComparison(noAbs, abs, outDir);
    DrawPtAbsorption(noAbs, abs, outDir);
    DrawPtSpectra(abs, outDir);
    DrawCtSurvived(noAbs, abs, outDir);
    DrawAbsorptionFractionVsCtPtBins(abs,
                                     abs.hCtAbs23Pt,
                                     "status 23",
                                     "Status 23 material-loss fraction vs stored c#tau",
                                     "absorption_fraction_vs_ct_ptbins",
                                     outDir);
    DrawAbsorptionFractionVsCtPtBins(abs,
                                     abs.hCtAbsInclusivePt,
                                     "status 13 + 20 + 23",
                                     "Status 13 + 20 + 23 material-loss fraction vs stored c#tau",
                                     "absorption_fraction_vs_ct_ptbins_status13_20_23",
                                     outDir);
    std::unique_ptr<ProxyAbsorptionQA> proxyAbsorption;
    if (he3ProxyAbsorptionFile && std::string(he3ProxyAbsorptionFile).size() > 0) {
        std::cout << "Reading He3 proxy absorption tree:\n  " << he3ProxyAbsorptionFile << "\n";
        proxyAbsorption = std::make_unique<ProxyAbsorptionQA>(ReadHe3ProxyAbsorptionTree(he3ProxyAbsorptionFile));
        DrawHe3ProxyAbsorptionFractionVsCtPtBins(*proxyAbsorption, outDir);
    }
    DrawAbsorbedRadiusDistributions(abs, proxyAbsorption.get(), outDir);
    DrawAbsorbedXYDistributions(abs, proxyAbsorption.get(), outDir);
    DrawReconstructedXYDistribution(abs, outDir);
    DrawAllMCVsHe3ProxyXYDistributions(abs, proxyAbsorption.get(), outDir);
    DrawVertexSanity(abs, outDir);
    DrawStatusVsPt(abs, outDir);
    WriteResults(noAbs, abs, outDir);

    std::cout << "QA plots written to: " << outDir << "\n";
}

void DrawAbsorbedXYDistributionsReweightedUpdatedMCG4list(
    const char* outputDir = "../../../Outputs/CrossSection/Plotting/AbsorptionMCQA",
    const char* he3ProxyAbsorptionFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x3.root",
    const char* outputName = "absorbed_xy_distributions_full_proxy.pdf",
    const char* rootOutputName = "absorbed_xy_distributions_full_proxy.root")
{
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    gStyle->SetNumberContours(255);

    const std::string outDir = ResolveOutputPath(outputDir ? outputDir : "");
    EnsureDir(outDir);

    SampleQA abs;
    abs.name = "updated_g4_absorption_reweighted";
    abs.title = "Updated G4list MC";
    abs.color = kOrange + 7;

    const std::vector<std::pair<std::string, std::string>> updatedG4listFiles = {
        {"LHC25g11", "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC25g11_G4list.root"},
        {"LHC26e5", "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC26e5_G4list.root"},
        {"LHC26e6", "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC26e6_G4list.root"}
    };
    std::vector<std::string> combinedPaths;
    combinedPaths.reserve(updatedG4listFiles.size());
    for (const auto& period : updatedG4listFiles) combinedPaths.push_back(period.second);

    std::cout << "Reading updated G4list MC samples:\n";
    ReadSamples(combinedPaths, abs);

    std::unique_ptr<ProxyAbsorptionQA> proxyAbsorption;
    if (he3ProxyAbsorptionFile && std::string(he3ProxyAbsorptionFile).size() > 0) {
        std::cout << "Reading full He3 proxy absorption tree:\n  " << he3ProxyAbsorptionFile << "\n";
        proxyAbsorption = std::make_unique<ProxyAbsorptionQA>(ReadHe3ProxyAbsorptionTree(he3ProxyAbsorptionFile));
    }

    DrawAbsorbedXYDistributions(abs,
                                proxyAbsorption.get(),
                                outDir,
                                outputName,
                                false);

    TFile fout(Form("%s/%s", outDir.c_str(), rootOutputName), "RECREATE");
    abs.hAbsVertexX->Write("h_abs_vertex_x_combined");
    abs.hAbsVertexY->Write("h_abs_vertex_y_combined");
    abs.hAbsVertexXY->Write("h_abs_vertex_xy_combined");
    if (proxyAbsorption) {
        proxyAbsorption->hAbsX->Write("h_proxy_abs_x_full_proxy");
        proxyAbsorption->hAbsY->Write("h_proxy_abs_y_full_proxy");
        proxyAbsorption->hAbsXY->Write("h_proxy_abs_xy_full_proxy");
    }
    for (const auto& period : updatedG4listFiles) {
        SampleQA periodAbs;
        periodAbs.name = "updated_g4_absorption_" + period.first;
        periodAbs.title = "Updated G4list MC " + period.first;
        periodAbs.color = kOrange + 7;
        std::cout << "Reading updated G4list MC sample for period " << period.first << ":\n"
                  << "  " << period.second << "\n";
        ReadSample(period.second, periodAbs);
        const std::string periodOutputName = AddSuffixBeforeExtension(outputName, "_" + period.first);
        DrawAbsorbedXYDistributions(periodAbs,
                                    proxyAbsorption.get(),
                                    outDir,
                                    periodOutputName,
                                    false);
        fout.cd();
        periodAbs.hAbsVertexX->Write(("h_abs_vertex_x_" + period.first).c_str());
        periodAbs.hAbsVertexY->Write(("h_abs_vertex_y_" + period.first).c_str());
        periodAbs.hAbsVertexXY->Write(("h_abs_vertex_xy_" + period.first).c_str());
    }
    fout.Close();

    std::cout << "Updated-G4list absorbed XY reweighted plot written to: "
              << outDir << "/" << outputName << "\n";
    std::cout << "Period-split absorbed XY plots written with suffixes next to the combined plot\n";
}

void DrawHe3ProxyAbsoCtQA(
    const char* he3ProxyAbsorptionFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x3.root",
    const char* outputDir = "../../../Outputs/CrossSection/Plotting/AbsorptionMCQA",
    const char* outputName = "he3_proxy_absoct_distribution_full_proxy.pdf")
{
    gStyle->SetOptStat(0);
    const std::string outDir = ResolveOutputPath(outputDir ? outputDir : "");
    EnsureDir(outDir);

    TFile input(he3ProxyAbsorptionFile, "READ");
    auto* tree = dynamic_cast<TTree*>(input.Get("he3candidates"));
    if (!tree) {
        std::cerr << "[DrawHe3ProxyAbsoCtQA] Cannot find he3candidates in " << he3ProxyAbsorptionFile << "\n";
        return;
    }

    Float_t pt = 0.f;
    Float_t eta = 0.f;
    Float_t absoX = 0.f;
    Float_t absoY = 0.f;
    Float_t absoZ = 0.f;
    tree->SetBranchAddress("pt", &pt);
    tree->SetBranchAddress("eta", &eta);
    tree->SetBranchAddress("absoX", &absoX);
    tree->SetBranchAddress("absoY", &absoY);
    tree->SetBranchAddress("absoZ", &absoZ);

    auto hRaw = std::make_unique<TH1D>("h_he3_proxy_absoct_raw", ";absorption c#tau = |r_{abs}| m_{H3L}/p (cm);Entries / cm", 140, 0.0, 700.0);
    auto hWeighted = std::make_unique<TH1D>("h_he3_proxy_absoct_tau253_weighted", ";absorption c#tau = |r_{abs}| m_{H3L}/p (cm);Expected entries / cm", 140, 0.0, 700.0);
    hRaw->Sumw2();
    hWeighted->Sumw2();

    std::vector<double> cts;
    cts.reserve(tree->GetEntries());
    double expectedKept = 0.0;
    long long selectedPt = 0;
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        if (FindPtBin(pt) < 0) continue;
        const double p = static_cast<double>(pt) * std::cosh(static_cast<double>(eta));
        const double absoL = std::sqrt(absoX * absoX + absoY * absoY + absoZ * absoZ);
        const double absoCt = (p > 0.0) ? absoL * kH3LMass / p : -1.0;
        if (!std::isfinite(absoCt) || absoCt < 0.0) continue;
        const double survival = std::exp(-absoCt / kOriginalTauCt);
        hRaw->Fill(absoCt);
        hWeighted->Fill(absoCt, survival);
        cts.push_back(absoCt);
        expectedKept += survival;
        ++selectedPt;
    }

    auto hRawDensity = MakeDensityClone(hRaw.get(), "h_he3_proxy_absoct_raw_density");
    auto hWeightedDensity = MakeDensityClone(hWeighted.get(), "h_he3_proxy_absoct_weighted_density");
    auto hSurvival = std::unique_ptr<TH1D>(dynamic_cast<TH1D*>(hWeighted.get()->Clone("h_he3_proxy_absoct_survival_probability")));
    hSurvival->SetDirectory(nullptr);
    hSurvival->Divide(hWeighted.get(), hRaw.get(), 1.0, 1.0, "B");
    hSurvival->GetYaxis()->SetTitle("Expected survival probability");

    TCanvas c("c_he3_proxy_absoct_distribution", "He3 proxy absoCt distribution", 950, 900);
    c.Divide(1, 2);
    c.cd(1);
    gPad->SetLogy();
    gPad->SetTicks();
    StyleHist(hRawDensity.get(), kBlack, 20);
    StyleHist(hWeightedDensity.get(), kRed + 1, 24);
    hRawDensity->SetTitle("He3 proxy absorption c#tau before/after #tau=253 ps survival sampling");
    hRawDensity->SetMinimum(1e-4);
    hRawDensity->SetMaximum(std::max(1.0, 4.0 * hRawDensity->GetMaximum()));
    hRawDensity->Draw("HIST");
    hWeightedDensity->Draw("HIST SAME");
    TLegend leg(0.50, 0.72, 0.88, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextFont(42);
    leg.SetTextSize(0.032);
    leg.AddEntry(hRawDensity.get(), Form("Raw proxy, N = %lld", selectedPt), "l");
    leg.AddEntry(hWeightedDensity.get(), Form("Expected kept, N = %.0f", expectedKept), "l");
    leg.Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextFont(42);
    text.SetTextSize(0.030);
    text.DrawLatex(0.16, 0.84, Form("c#tau_{0} = 253 ps #times c = %.2f cm", kOriginalTauCt));

    c.cd(2);
    gPad->SetTicks();
    StyleHist(hSurvival.get(), kBlue + 1, 20);
    hSurvival->SetTitle(";absorption c#tau = |r_{abs}| m_{H3L}/p (cm);#LT exp(-c#tau_{abs}/c#tau_{0}) #GT");
    hSurvival->SetMinimum(0.0);
    hSurvival->SetMaximum(1.05);
    hSurvival->Draw("E1");
    c.SaveAs(Form("%s/%s", outDir.c_str(), outputName));

    std::sort(cts.begin(), cts.end());
    auto quantile = [&](double q) {
        if (cts.empty()) return -1.0;
        return cts[std::min(static_cast<size_t>((cts.size() - 1) * q), cts.size() - 1)];
    };
    std::cout << "[DrawHe3ProxyAbsoCtQA] N=" << selectedPt
              << ", expected kept=" << expectedKept
              << ", fraction=" << (selectedPt > 0 ? expectedKept / selectedPt : 0.0)
              << ", absoCt q10/q25/median/q75/q90/q99="
              << quantile(0.10) << "/" << quantile(0.25) << "/" << quantile(0.50) << "/"
              << quantile(0.75) << "/" << quantile(0.90) << "/" << quantile(0.99)
              << " cm\n";
}
