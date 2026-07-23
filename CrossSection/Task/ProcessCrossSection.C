// ProcessCrossSection.C
// Usage: root -l -b -q 'ProcessCrossSection.C("../../configs/crosssection_config.json")'
#include <RooFit.h>
#include <RooRealVar.h>
#include <RooExponential.h>
#include <RooSimultaneous.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooBinning.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooAbsPdf.h>
#include <RooPlot.h>
#include <RooFormulaVar.h>
#include <RooExtendPdf.h>
#include <RooAbsReal.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <regex>
#include <cmath>
#include <map>
#include <memory>
#include <TMath.h>

#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TSystem.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// helper for reading config json
#include <nlohmann/json.hpp>
#include "../../include/include.h"
#include "../../include/GlobalChi2Roo.h"
#include "../../Tools/GeneralHelper.hpp"
#include "../../Tools/AbsorptionHelper.h"

using namespace RooFit;
using std::string;
using std::vector;
using json = nlohmann::json;
using namespace Physics;
using namespace Absorption;
using namespace GeneralHelper;

inline std::string ResolvePathFromMacroDir(const std::string& path) {
    if (path.empty() || gSystem->IsAbsoluteFileName(path.c_str()) || !gSystem->AccessPathName(path.c_str())) {
        return path;
    }
    std::string macroPath = __FILE__;
    std::string macroDir = ".";
    size_t pos = macroPath.find_last_of("/\\");
    if (pos != std::string::npos) {
        macroDir = macroPath.substr(0, pos);
    }
    std::string candidate = macroDir + "/" + path;
    return gSystem->AccessPathName(candidate.c_str()) ? path : candidate;
}

inline std::string ResolvePathFromFileDir(const std::string& baseFile, const std::string& path) {
    if (path.empty() || gSystem->IsAbsoluteFileName(path.c_str()) || !gSystem->AccessPathName(path.c_str())) {
        return path;
    }
    std::string baseDir = ".";
    size_t pos = baseFile.find_last_of("/\\");
    if (pos != std::string::npos) {
        baseDir = baseFile.substr(0, pos);
    }
    return baseDir + "/" + path;
}

struct FitResult {
    double tau;
    double tauErr;
    double chi2;
    int ndf;
    double globalfitprobility;
    std::vector<double> chi2PerChannel;
    std::vector<int> ndfPerChannel;
    std::vector<TF1*> fitfuncs;
    std::vector<double> fitprobilitys;
};

inline bool UseBinIntegralFitMode(const std::string& fitBinMode) {
    return fitBinMode == "bin_integral_counts" || fitBinMode == "integral" || fitBinMode == "counts";
}

inline std::string FormatNumberForFilename(double value) {
    std::string text = Form("%g", value);
    for (char& c : text) {
        if (c == '.') c = 'p';
        if (c == '-') c = 'm';
    }
    return text;
}

inline std::unique_ptr<TH1> MakeDensityClone(const TH1* hist, const std::string& name) {
    if (!hist) return nullptr;
    auto* clone = dynamic_cast<TH1*>(hist->Clone(name.c_str()));
    if (!clone) return nullptr;
    clone->SetDirectory(nullptr);
    for (int b = 1; b <= clone->GetNbinsX(); ++b) {
        const double width = clone->GetXaxis()->GetBinWidth(b);
        if (width <= 0.0) continue;
        clone->SetBinContent(b, clone->GetBinContent(b) / width);
        clone->SetBinError(b, clone->GetBinError(b) / width);
    }
    return std::unique_ptr<TH1>(clone);
}

inline FitResult ProcessSimultaneousExpoFitHists(const vector<TH1*>& hists,
                                                bool useFixedTau = false,
                                                double fixedTauPs = 253.0,
                                                const std::string& fitBinMode = "bin_center_value") {
    FitResult result;
    result.tau = 0.0;
    result.tauErr = 0.0;
    result.chi2 = 0.0;
    result.ndf = 0;
    result.globalfitprobility = 0.0;
    result.chi2PerChannel = std::vector<double>();
    result.ndfPerChannel = std::vector<int>();
    result.fitfuncs = std::vector<TF1*>();
    result.fitprobilitys = std::vector<double>();

    if (hists.empty()) {
        std::cerr << "[Error] No histograms provided for fitting.\n";
        return result;
    }

    int nHists = hists.size();
    std::cout << "Fitting " << nHists << " histograms simultaneously with shared tau parameter.\n";
    std::cout << "Using custom GlobalChi2Roo and Minuit2 minimizer.\n";

    int npar = 1 + nHists; // tau + amplitudes

    const bool useBinIntegral = UseBinIntegralFitMode(fitBinMode);
    RooRealVar tau("tau", "decay constant", 7.6, 1.0, 30.0);
    if (useFixedTau) {
        double fixedTauCt = fixedTauPs * c_cm_per_ps;
        tau.setVal(fixedTauCt);
        tau.setRange(fixedTauCt, fixedTauCt);
        tau.setConstant(true);
        std::cout << "Tau fixed to " << fixedTauPs << " ps (" << fixedTauCt
                  << " cm) for chi2 minimization.\n";
    }
    vector<RooRealVar*> A;
    for (int i = 0; i < nHists; ++i) {
        const double xmin = hists[i]->GetXaxis()->GetXmin();
        const double xmax = hists[i]->GetXaxis()->GetXmax();
        double Ainit = std::max(1.0, hists[i]->GetMaximum());
        if (useBinIntegral) {
            const double normIntegral = tau.getVal() * (std::exp(-xmin / tau.getVal()) - std::exp(-xmax / tau.getVal()));
            const double integral = hists[i]->Integral();
            Ainit = (normIntegral > 0.0) ? integral / normIntegral : Ainit;
        }
        if (!std::isfinite(Ainit) || Ainit <= 0.0) Ainit = std::max(1.0, hists[i]->GetMaximum());

        A.push_back(new RooRealVar(Form("A%d", i+1), Form("A%d", i+1),
                                   Ainit, 0.0, std::max(10.0 * Ainit, 10.0)));
    }

    // Construct χ² object
    GlobalChi2Roo chi2("chi2", hists, tau, A, useBinIntegral);
    if (useFixedTau) {
        chi2.SetTauValue(tau.getVal());
        chi2.SetTauConstant(true);
    }

    // RooMinimizer
    RooMinimizer minim(chi2);
    minim.setPrintLevel(0);
    minim.optimizeConst(false);
    minim.setStrategy(2); // more robust (but slower) minimization
    minim.setPrintLevel(0);
    // Increase budget for Minuit2: allow more function calls and iterations
    minim.setMaxFunctionCalls(1000000);
    minim.setMaxIterations(1000000);
    // Run a generous MIGRAD pass before the final minimize call to help convergence
    // int statusMigrad = minim.migrad(500000, 1e-6);
    int statusMigrad = minim.minimize("Minuit2", "Migrad");
    int statusHesse = minim.hesse();
    int statusMinos = 0;
    cout << "Minimization status: Migrad=" << statusMigrad
         << ", Hesse=" << statusHesse << std::endl;
    // collect fit results
    double tauVal = tau.getVal();
    double tauErr = useFixedTau ? 0.0 : tau.getError();
    double realTauVal = tauVal / c_cm_per_ps;
    double realTauErr = tauErr / c_cm_per_ps;

    // compute per-histogram chi2 and ndf using the same bin interpretation
    // as the minimizer. h_corrected_counts stores corrected bin values, so the
    // default mode compares content at the bin center; an integral-counts mode
    // is kept for closure tests with raw count histograms.
    std::vector<double> chi2PerChannel(nHists, 0.0);
    std::vector<int> ndfPerChannel(nHists, 0);
    std::vector<TF1*> funcs;

    const double tauValSafe = (tauVal == 0.0) ? 1e-12 : tauVal;
    int totalValidBins = 0;
    double totalChi2 = 0.0;
    for (int i = 0; i < nHists; ++i) {
        TH1* h = hists[i];
        if (!h) continue;

        int nbins = h->GetNbinsX();
        int validBins = 0;
        double chi2_i = 0.0;
        double Ai = A[i]->getVal();

        TF1* f = new TF1(Form("fitfunc_%d", i), "[0]*exp(-x/[1])", h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
        f->SetParameters(Ai, tauValSafe);
        funcs.push_back(f);
        
        for (int b = 1; b <= nbins; ++b) {
            // double xlo = h->GetXaxis()->GetBinLowEdge(b);
            // double xhi = h->GetXaxis()->GetBinUpEdge(b);
            // double binwidth = xhi - xlo;
            // double obs = h->GetBinContent(b) * binwidth;
            // double err = h->GetBinError(b);
            // if (err <= 0.0) continue; // skip bins without a valid uncertainty
            // double expected = Ai * tauValSafe * (std::exp(-xlo / tauValSafe) - std::exp(-xhi / tauValSafe));

            // double delta = obs - expected;
            // chi2_i += (delta * delta) / (err * err * binwidth); //Poisson errors
            double obs = h->GetBinContent(b);
            double err = h->GetBinError(b);
            if (err <= 0.0) continue; // skip bins without a valid uncertainty
            double expected = 0.0;
            if (useBinIntegral) {
                const double xlo = h->GetXaxis()->GetBinLowEdge(b);
                const double xhi = h->GetXaxis()->GetBinUpEdge(b);
                expected = Ai * tauValSafe * (std::exp(-xlo / tauValSafe) - std::exp(-xhi / tauValSafe));
            } else {
                expected = Ai * std::exp(-h->GetXaxis()->GetBinCenter(b) / tauValSafe);
            }
            double delta = obs - expected;
            chi2_i += (delta * delta) / (err * err);
            ++validBins;
        }

        chi2PerChannel[i] = chi2_i;
        // assign ndf per channel as (valid bins - 1) because each channel has its own amplitude A_i
        ndfPerChannel[i] = std::max(0, validBins - 1);
        totalValidBins += validBins;
        totalChi2 += chi2_i;
    }

    // total number of fitted parameters: tau (if free) + nHists (each amplitude)
    int totalPars = (useFixedTau ? 0 : 1) + nHists;
    int totalNdf = std::max(0, totalValidBins - totalPars);

    // compute per-channel fit probabilities (p-values) and global fit probability
    std::vector<double> fitProbs(nHists, 0.0);
    for (int i = 0; i < nHists; ++i) {
        int nd = ndfPerChannel[i];
        double chi = chi2PerChannel[i];
        if (nd > 0) {
            // TMath::Prob returns the upper tail probability for the chi2 distribution
            fitProbs[i] = TMath::Prob(chi, nd);
        } else {
            fitProbs[i] = 0.0; // undefined/insufficient dof -> set 0.0
        }
    }

    double globalProb = 0.0;
    if (totalNdf > 0) {
        globalProb = TMath::Prob(totalChi2, totalNdf);
    }

    // prepare FitResult
    result.tau = realTauVal;
    result.tauErr = realTauErr;
    result.chi2 = totalChi2;
    result.ndf = totalNdf;
    result.chi2PerChannel = std::move(chi2PerChannel);
    result.ndfPerChannel = std::move(ndfPerChannel);
    result.fitfuncs = std::move(funcs);
    result.fitprobilitys = std::move(fitProbs);
    result.globalfitprobility = globalProb;

    // no RooPlot frames produced here; leave empty so caller can create plots if desired
    return result;
}

inline void FillFitDiagnostics(const vector<TH1*>& hists,
                               const std::vector<double>& amplitudes,
                               double tauCt,
                               bool useFixedTau,
                               FitResult& result,
                               const std::string& fitBinMode = "bin_center_value") {
    const int nHists = static_cast<int>(hists.size());
    std::vector<double> chi2PerChannel(nHists, 0.0);
    std::vector<int> ndfPerChannel(nHists, 0);
    std::vector<TF1*> funcs;

    const double tauValSafe = (tauCt == 0.0) ? 1e-12 : tauCt;
    const bool useBinIntegral = UseBinIntegralFitMode(fitBinMode);
    int totalValidBins = 0;
    double totalChi2 = 0.0;
    for (int i = 0; i < nHists; ++i) {
        TH1* h = hists[i];
        if (!h) continue;

        const double amp = (i < static_cast<int>(amplitudes.size())) ? amplitudes[i] : h->GetMaximum();
        TF1* f = new TF1(Form("fitfunc_%d", i), "[0]*exp(-x/[1])", h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
        f->SetParameters(amp, tauValSafe);
        funcs.push_back(f);

        int validBins = 0;
        double chi2_i = 0.0;
        for (int b = 1; b <= h->GetNbinsX(); ++b) {
            double obs = h->GetBinContent(b);
            double err = h->GetBinError(b);
            if (err <= 0.0) continue;
            double expected = 0.0;
            if (useBinIntegral) {
                const double xlo = h->GetXaxis()->GetBinLowEdge(b);
                const double xhi = h->GetXaxis()->GetBinUpEdge(b);
                expected = amp * tauValSafe * (std::exp(-xlo / tauValSafe) - std::exp(-xhi / tauValSafe));
            } else {
                expected = amp * std::exp(-h->GetXaxis()->GetBinCenter(b) / tauValSafe);
            }
            double delta = obs - expected;
            chi2_i += (delta * delta) / (err * err);
            ++validBins;
        }

        chi2PerChannel[i] = chi2_i;
        ndfPerChannel[i] = std::max(0, validBins - 1);
        totalValidBins += validBins;
        totalChi2 += chi2_i;
    }

    int totalPars = (useFixedTau ? 0 : 1) + nHists;
    int totalNdf = std::max(0, totalValidBins - totalPars);
    std::vector<double> fitProbs(nHists, 0.0);
    for (int i = 0; i < nHists; ++i) {
        if (ndfPerChannel[i] > 0) {
            fitProbs[i] = TMath::Prob(chi2PerChannel[i], ndfPerChannel[i]);
        }
    }

    result.chi2 = totalChi2;
    result.ndf = totalNdf;
    result.globalfitprobility = (totalNdf > 0) ? TMath::Prob(totalChi2, totalNdf) : 0.0;
    result.chi2PerChannel = std::move(chi2PerChannel);
    result.ndfPerChannel = std::move(ndfPerChannel);
    result.fitfuncs = std::move(funcs);
    result.fitprobilitys = std::move(fitProbs);
}

inline FitResult ProcessRooFitSimultaneousExpoFitHists(const vector<TH1*>& hists,
                                                       bool useFixedTau = false,
                                                       double fixedTauPs = 253.0,
                                                       const std::string& fitBinMode = "bin_center_value") {
    FitResult result;
    result.tau = 0.0;
    result.tauErr = 0.0;
    result.chi2 = 0.0;
    result.ndf = 0;
    result.globalfitprobility = 0.0;

    if (hists.empty()) {
        std::cerr << "[Error] No histograms provided for RooFit fitting.\n";
        return result;
    }

    std::cout << "Fitting " << hists.size()
              << " histograms with RooSimultaneous binned chi2 and a shared tau.\n";
    const bool useBinIntegral = UseBinIntegralFitMode(fitBinMode);
    if (!useBinIntegral) {
        std::cout << "RooFit bin-center mode: converting N_corr values to width-weighted pseudo-counts.\n";
    }

    double xmin = hists.front()->GetXaxis()->GetXmin();
    double xmax = hists.front()->GetXaxis()->GetXmax();
    for (TH1* h : hists) {
        if (!h) continue;
        xmin = std::min(xmin, h->GetXaxis()->GetXmin());
        xmax = std::max(xmax, h->GetXaxis()->GetXmax());
    }

    std::vector<double> mergedEdges;
    for (TH1* h : hists) {
        if (!h) continue;
        for (int b = 1; b <= h->GetNbinsX(); ++b) {
            mergedEdges.push_back(h->GetXaxis()->GetBinLowEdge(b));
        }
        mergedEdges.push_back(h->GetXaxis()->GetBinUpEdge(h->GetNbinsX()));
    }
    std::sort(mergedEdges.begin(), mergedEdges.end());
    mergedEdges.erase(std::unique(mergedEdges.begin(), mergedEdges.end(), [](double a, double b) {
        return std::fabs(a - b) < 1e-9;
    }), mergedEdges.end());
    if (mergedEdges.size() < 2) {
        mergedEdges = {xmin, xmax};
    }

    RooRealVar ct("ct", "c#tau", xmin, xmax, "cm");
    RooBinning ctBinning(static_cast<int>(mergedEdges.size() - 1), mergedEdges.data(), "ctBinning");
    ct.setBinning(ctBinning);
    RooCategory sample("sample", "sample");

    RooRealVar tau("tau", "lifetime", 7.6, 1.0, 30.0);
    if (useFixedTau) {
        const double fixedTauCt = fixedTauPs * c_cm_per_ps;
        tau.setVal(fixedTauCt);
        tau.setRange(fixedTauCt, fixedTauCt);
        tau.setConstant(true);
        std::cout << "Tau fixed to " << fixedTauPs << " ps (" << fixedTauCt
                  << " cm) for RooFit chi2 minimization.\n";
    }
    RooFormulaVar slope("slope", "-1.0/@0", RooArgList(tau));

    RooSimultaneous simPdf("simPdf", "simultaneous exponential PDF", sample);
    std::vector<std::unique_ptr<RooExponential>> pdfs;
    std::vector<std::unique_ptr<RooRealVar>> yields;
    std::vector<std::unique_ptr<RooExtendPdf>> extendedPdfs;

    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        const std::string label = Form("ptbin_%zu", i);
        sample.defineType(label.c_str());

        pdfs.emplace_back(std::make_unique<RooExponential>(Form("expo_%zu", i), Form("expo_%zu", i), ct, slope));
        double integral = 0.0;
        for (int b = 1; b <= h->GetNbinsX(); ++b) {
            const double width = h->GetXaxis()->GetBinWidth(b);
            integral += h->GetBinContent(b) * (useBinIntegral ? 1.0 : width);
        }
        integral = std::max(1.0, integral);
        yields.emplace_back(std::make_unique<RooRealVar>(Form("yield_%zu", i), Form("yield_%zu", i), integral, 0.0, 10.0 * integral));
        extendedPdfs.emplace_back(std::make_unique<RooExtendPdf>(Form("ext_expo_%zu", i), Form("ext_expo_%zu", i),
                                                                 *pdfs.back(), *yields.back()));
        simPdf.addPdf(*extendedPdfs.back(), label.c_str());
    }

    RooDataHist data("data", "combined binned data", RooArgSet(ct, sample));
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        if (!h) continue;
        const std::string label = Form("ptbin_%zu", i);
        sample.setLabel(label.c_str());
        for (int b = 1; b <= h->GetNbinsX(); ++b) {
            const double width = h->GetXaxis()->GetBinWidth(b);
            const double scale = useBinIntegral ? 1.0 : width;
            const double content = h->GetBinContent(b) * scale;
            if (content <= 0.0) continue;
            ct.setVal(h->GetXaxis()->GetBinCenter(b));
            data.add(RooArgSet(ct, sample), content, h->GetBinError(b) * scale);
        }
    }
    std::unique_ptr<RooAbsReal> chi2(simPdf.createChi2(data, Extended(true), DataError(RooAbsData::SumW2)));
    RooMinimizer minim(*chi2);
    minim.setPrintLevel(0);
    minim.optimizeConst(false);
    minim.setStrategy(2);
    minim.setMaxFunctionCalls(1000000);
    minim.setMaxIterations(1000000);
    int statusMigrad = minim.minimize("Minuit2", "Migrad");
    int statusHesse = minim.hesse();
    std::cout << "RooFit minimization status: Migrad=" << statusMigrad
              << ", Hesse=" << statusHesse << std::endl;

    const double tauVal = tau.getVal();
    const double tauErr = useFixedTau ? 0.0 : tau.getError();
    result.tau = tauVal / c_cm_per_ps;
    result.tauErr = tauErr / c_cm_per_ps;

    std::vector<double> amplitudes;
    amplitudes.reserve(hists.size());
    for (size_t i = 0; i < hists.size(); ++i) {
        TH1* h = hists[i];
        const double normIntegral = tauVal * (std::exp(-h->GetXaxis()->GetXmin() / tauVal)
                                  - std::exp(-h->GetXaxis()->GetXmax() / tauVal));
        const double amp = (normIntegral > 0.0 && i < yields.size()) ? yields[i]->getVal() / normIntegral
                                                                     : h->GetMaximum();
        amplitudes.push_back(amp);
    }
    FillFitDiagnostics(hists, amplitudes, tauVal, useFixedTau, result, fitBinMode);
    return result;
}

inline FitResult ProcessExpoFitHists(const vector<TH1*>& hists,
                                     const std::string& fitBackend,
                                     bool useFixedTau = false,
                                     double fixedTauPs = 253.0,
                                     const std::string& fitBinMode = "bin_center_value") {
    if (fitBackend == "roofit" || fitBackend == "RooFit") {
        return ProcessRooFitSimultaneousExpoFitHists(hists, useFixedTau, fixedTauPs, fitBinMode);
    }
    return ProcessSimultaneousExpoFitHists(hists, useFixedTau, fixedTauPs, fitBinMode);
}

inline TH1* FindPtCtHistogram(TFile* inputFile,
                              const std::string& ptDirName,
                              const std::string& histName,
                              double ptmin,
                              double ptmax) {
    if (!inputFile) return nullptr;

    const std::string oldHistName = Form("%s_pt_%g_%g", histName.c_str(), ptmin, ptmax);
    const std::vector<std::string> candidates = {
        ptDirName + "/std/" + histName,
        ptDirName + "/" + histName,
        "std/" + ptDirName + "/" + oldHistName,
        "std/" + ptDirName + "/" + histName,
        ptDirName + "/std/" + oldHistName,
        ptDirName + "/" + oldHistName
    };

    for (const auto& path : candidates) {
        if (auto* h = dynamic_cast<TH1*>(inputFile->Get(path.c_str()))) {
            std::cout << "Loaded input histogram: " << path << "\n";
            return h;
        }
    }
    std::cerr << "Histogram not found for " << ptDirName << ". Tried:\n";
    for (const auto& path : candidates) {
        std::cerr << "  " << path << "\n";
    }
    return nullptr;
}

inline int CheckBinsHist(std::vector<double> ptbins, std::vector<std::vector<double>> ctbins, TFile* inputFile, std::vector<TH1*> & outHists, std::string histName)
{
    if (!((ptbins.size() - 1) == ctbins.size())) 
    {
        std::cerr << "Mismatch: ptbins.size() = " << ptbins.size()
                  << ", ctbins.size() = " << ctbins.size() << "\n";

        // print ptbins
        std::cerr << "ptbins: [";
        for (size_t i = 0; i < ptbins.size(); ++i) {
            std::cerr << ptbins[i];
            if (i + 1 < ptbins.size()) std::cerr << ", ";
        }
        std::cerr << "]\n";

        // print ctbins
        std::cerr << "ctbins (per element show size and edges):\n";
        for (size_t i = 0; i < ctbins.size(); ++i) {
            std::cerr << "  ctbins[" << i << "] size=" << ctbins[i].size() << " edges=[";
            for (size_t j = 0; j < ctbins[i].size(); ++j) {
                std::cerr << ctbins[i][j];
                if (j + 1 < ctbins[i].size()) std::cerr << ", ";
            }
            std::cerr << "]\n";
        }
        return 0;
    }

    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Invalid input TFile pointer\n";
        return 0;
    }

    if (ptbins.size() < 2) {
        std::cerr << "ptbins must contain bin edges (at least two values)\n";
        return 0;
    }

    // expect ptbins to be edges: number of pt bins = edges - 1
    size_t nPtBins = ptbins.size() - 1;
    if (ctbins.size() != nPtBins) {
        std::cerr << "ctbins size (" << ctbins.size() << ") does not match number of pt-bins (" << nPtBins << ")\n";
        return 0;
    }

    const double eps = 1e-6;
    char bufDir[128];
    char bufHist[256];

    for (size_t i = 0; i < nPtBins; ++i) {
        double ptmin = ptbins[i];
        double ptmax = ptbins[i+1];
        snprintf(bufDir, sizeof(bufDir), "pt_%g_%g", ptmin, ptmax);
        std::string dirName(bufDir);

        TH1 *h = FindPtCtHistogram(inputFile, dirName, histName, ptmin, ptmax);
        if (!h) {
            return 0;
        }
        outHists.push_back(h);

        // check histogram binning against provided ctbins[i]
        const auto &ctedges = ctbins[i];
        if (ctedges.size() < 2) {
            std::cerr << "ctbins[" << i << "] must contain at least two edges\n";
            return 0;
        }

        int nbins = h->GetNbinsX();
        size_t expectBins = ctedges.size() - 1;
        if (nbins != static_cast<int>(expectBins)) {
            std::cerr << "Binning mismatch for " << h->GetName() << ": histogram bins = " << nbins
                      << " expected = " << expectBins << "\n";
            return 0;
        }

        // check each bin edge (use low edges for bins 1..nbins and final upper edge)
        for (size_t b = 0; b < ctedges.size(); ++b) {
            double hEdge;
            if (b < ctedges.size() - 1) {
                // low edge of bin b+1
                hEdge = h->GetXaxis()->GetBinLowEdge(static_cast<int>(b + 1));
            } else {
                // upper edge: low edge of last bin + width
                int last = nbins;
                hEdge = h->GetXaxis()->GetBinLowEdge(last) + h->GetXaxis()->GetBinWidth(last);
            }
            double expected = static_cast<double>(ctedges[b]);
            if (std::fabs(hEdge - expected) > eps) {
                std::cerr << "Edge mismatch for " << h->GetName() << " at edge index " << b
                          << ": hist = " << hEdge << " expected = " << expected << "\n";
                return 0;
            }
        }
    }
    return 1;
}



// helper: extract multiplier from filename like "absorption_treex1.5.root"
inline double ExtractMultiplierFromFilename(const std::string& fname) {
    if (fname.empty()) return 1.0;

    // strip path, keep base name
    std::string base = fname;
    size_t p = base.find_last_of("/\\");
    if (p != std::string::npos) base = base.substr(p + 1);

    // search for last occurrence of 'x' or 'X' followed by a number (e.g. x1.5)
    std::regex re(R"([xX]([+-]?[0-9]+(?:\.[0-9]+)?))");
    std::smatch m;
    std::string::const_iterator start = base.cbegin();
    double value = 1.0;
    bool found = false;
    while (std::regex_search(start, base.cend(), m, re)) {
        try {
            value = std::stod(m[1].str()); // keep last match
            found = true;
        } catch (...) {
            // ignore parse errors, continue
        }
        start = m.suffix().first;
    }

    // fallback: try to parse a number right before ".root" if no 'x' pattern found
    if (!found) {
        std::regex re2(R"(([+-]?[0-9]+(?:\.[0-9]+)?)\.root$)", std::regex::icase);
        if (std::regex_search(base, m, re2)) {
            try { value = std::stod(m[1].str()); found = true; } catch(...) {}
        }
    }

    if (!found) {
        std::cerr << "Warning: cannot extract multiplier from filename '" << fname << "'. Using 1.0\n";
    }
    return value;
}

inline double ExtractMultiplierFromTFile(TFile* f) {
    if (!f) return 1.0;
    const char* name = f->GetName();
    return ExtractMultiplierFromFilename(name ? std::string(name) : std::string());
}

void ProcessCrossSection(const char* config_path = "../../configs/crosssection_config.json") {
    // read config.json (path passed as config_path)
    std::string cfgpath = ResolvePathFromMacroDir(config_path ? std::string(config_path) : std::string("../../configs/crosssection_config.json"));
    std::ifstream ifs(cfgpath);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open config file: " << cfgpath << "\n";
        return;
    }
    json cfg;
    ifs >> cfg;
    std::vector<std::string> AbsorptionFilePath = cfg["absorptionpath"].get<std::vector<std::string>>();
    for (auto& path : AbsorptionFilePath) {
        path = ResolvePathFromFileDir(cfgpath, path);
    }
    std::string inputFilePath = ResolvePathFromFileDir(cfgpath, cfg["input"].get<std::string>());
    std::string outputPath = ResolvePathFromFileDir(cfgpath, cfg["outputpath"].get<std::string>());
    std::vector<double> ptBins = cfg["ptbins"].get<std::vector<double>>();
    std::vector<std::vector<double>> ctBins = cfg["ctbins"].get<std::vector<std::vector<double>>>();
    std::string isMatter = cfg["ismatter"].get<std::string>();
    std::string absTreeName = cfg["abstreename"].get<std::string>();
    std::string histName  = cfg["histName"].get<std::string>();
    bool useFixedTau = cfg.value("use_fixed_tau", false);
    double fixedTauPs = cfg.value("fixed_tau_ps", 253.0);
    double fixedTauCt = fixedTauPs * c_cm_per_ps;
    std::string fitBackend = cfg.value("fit_backend", std::string("custom_chi2"));
    std::string fitBinMode = cfg.value("fit_bin_mode", std::string("bin_center_value"));
    if (cfg.value("use_roofit", false)) {
        fitBackend = "roofit";
    }
    const bool useBinIntegralFit = UseBinIntegralFitMode(fitBinMode);
    // load all the file needed
    std::vector<TFile*> AbsorptionFiles;
    for (const auto& path : AbsorptionFilePath) {
        TFile* f = TFile::Open(path.c_str(), "READ");
        if (!f || f->IsZombie()) {
            std::cerr << "Cannot open absorption file: " << path << "\n";
            continue;
        }
        AbsorptionFiles.push_back(f);
    }
    TFile* inputFile = TFile::Open(inputFilePath.c_str(), "READ");
    std::vector<TH1*> inputHists;
    if (CheckBinsHist(ptBins, ctBins, inputFile, inputHists, histName) != 1) {
        std::cerr << "ptBins and ctBins size mismatch\n";
        return;
    }
    else {
        std::cout << "All input " << inputHists.size() << "histograms loaded and verified.\n";
    }
    // main process
    std::map<std::string, std::vector<TH1F*>> fHCounts;
    std::map<std::string, std::vector<TH1F*>> fHCountsAbsorb;
    std::map<std::string, std::vector<TH1F*>> fHRatio;
    std::vector<TDirectory*> subDirs;
    // ensure output directory exists before creating the ROOT file
    std::string outDir = outputPath + "/crosssection_extraction/" + isMatter + "/";
    EnsureDir(outDir);
    std::string outFileName = outDir + "crosssection_extraction_results.root";
    TFile* outfile = TFile::Open(outFileName.c_str(), "RECREATE");
    if (!outfile || outfile->IsZombie()) {
        std::cerr << "Failed to create output file: " << outFileName << "\n";
        return;
    }
    std::vector<double> chi2s;
    std::vector<int> ndfs;
    std::vector<double> taus;
    std::vector<double> tauErrs;
    std::vector<double> multipliers;
    for (size_t i = 0; i < AbsorptionFiles.size(); ++i) {
        double multiplier = ExtractMultiplierFromTFile(AbsorptionFiles[i]);
        multipliers.push_back(multiplier);
        TTree* absTree = dynamic_cast<TTree*>(AbsorptionFiles[i]->Get(absTreeName.c_str()));
        ROOT::RDataFrame inputRDF(*absTree);
        std::string histNameSuffix = Form("_x%g", multiplier);
        double seedTauCt = useFixedTau ? fixedTauCt : 7.6;
        PtAbsorptionCalculator absCalculator(&inputRDF, ptBins, ctBins, seedTauCt, histNameSuffix);
        absCalculator.Calculate();
        cout << "Processing absorption file with multiplier " << multiplier << "...\n";
        fHCounts = absCalculator.HistCounts();
        fHCountsAbsorb = absCalculator.HistCountsAbsorb();
        fHRatio = absCalculator.HistRatio();
        TDirectory* dirMulti = outfile->mkdir(Form("He3CrosssectionMulti_%g", multiplier));
        subDirs.push_back(dirMulti);
        dirMulti->cd();
        std::vector<TH1*> copyedHistos = CopyTH1Vector(inputHists, Form("cloned_%g", multiplier));
        for (size_t j = 0; j < ptBins.size() - 1; ++j) {
                double ptmin = ptBins[j];
                double ptmax = ptBins[j + 1];
                // fHCounts[isMatter][j]->Write();
                // fHCountsAbsorb[isMatter][j]->Write();
                TH1* ratio = fHRatio[isMatter][j];
                ratio->Write();
                TH1* ratioNoErr = nullptr;
                if (ratio) {
                    ratioNoErr = dynamic_cast<TH1*>(ratio->Clone(Form("%s_noerr_%zu", ratio->GetName(), j)));
                    if (ratioNoErr) {
                        ratioNoErr->SetDirectory(nullptr);
                        int nb = ratioNoErr->GetNbinsX();
                        for (int b = 1; b <= nb; ++b) ratioNoErr->SetBinError(b, 0.0);
                    }
                }
                if (ratioNoErr) {
                    copyedHistos[j]->Divide(inputHists[j], ratioNoErr, 1.0, 1.0, "");
                    delete ratioNoErr;
                } else {
                    std::cerr << "[Warning] ratio histogram missing for bin " << j << ". Skipping division.\n";
                }
                copyedHistos[j]->SetName(Form("corrected_spectrum_pt_%g_%g_multi_%g", ptmin, ptmax, multiplier));
                copyedHistos[j]->Write();
                std::cout << "Done. Canvas contains fits and chi2 per pt-bin.\n";
        }
        FitResult fitRes = ProcessExpoFitHists(copyedHistos, fitBackend, useFixedTau, fixedTauPs, fitBinMode);
        chi2s.push_back(fitRes.chi2);
        ndfs.push_back(fitRes.ndf);
        taus.push_back(fitRes.tau);
        tauErrs.push_back(fitRes.tauErr);
        // cout fit results
        std::cout << "Fitted tau (multiplier " << multiplier << ") = " << fitRes.tau
              << " ± " << fitRes.tauErr << std::endl;
        cout << "Chi2 / ndf = " << fitRes.chi2 << " / " << fitRes.ndf << std::endl;
        cout << "Global fit probability = " << fitRes.globalfitprobility << std::endl;
        cout << fitRes.fitfuncs.size() << " funcs generated.\n";
        // 保存每个 pt-bin 的拟合图到子目录
        TDirectory* dirFits = dirMulti->mkdir("FitFrames");
        dirFits->cd();
        for (size_t k = 0; k < fitRes.fitfuncs.size(); ++k) {
            TCanvas* cFit = new TCanvas(Form("cFit_ptbin_%zu_multi_%g", k, multiplier), Form("Fit for pt-bin %zu (multi %g)", k, multiplier), 800, 600);
            TH1* h = copyedHistos[k];
            if (!h) continue;
            auto hDensity = useBinIntegralFit ? MakeDensityClone(h, Form("%s_density_draw", h->GetName())) : nullptr;
            TH1* hDraw = hDensity ? hDensity.get() : h;
            // Beautify plot, draw fit and annotate full fit information
            gStyle->SetOptStat(0);
            hDraw->SetStats(0);
            hDraw->SetLineWidth(2);
            hDraw->SetMarkerStyle(20);
            hDraw->SetMarkerSize(1.0);
            hDraw->GetXaxis()->SetTitle("ct (cm)");
            hDraw->GetYaxis()->SetTitle(useBinIntegralFit ? "Corrected counts / cm" : "N_{corr}");
            hDraw->GetXaxis()->SetTitleSize(0.05);
            hDraw->GetYaxis()->SetTitleSize(0.05);
            hDraw->GetXaxis()->SetLabelSize(0.04);
            hDraw->GetYaxis()->SetLabelSize(0.04);
            hDraw->GetXaxis()->SetTitleFont(42);
            hDraw->GetYaxis()->SetTitleFont(42);
            hDraw->GetXaxis()->SetLabelFont(42);
            hDraw->GetYaxis()->SetLabelFont(42);

            // recover pt-bin edges for title (safe check)
            double fptmin = 0.0, fptmax = 0.0;
            if (k < (int)ptBins.size() - 1) { fptmin = ptBins[k]; fptmax = ptBins[k + 1]; }

            hDraw->SetTitle(Form("Corrected spectrum pt [%.3g, %.3g] (mult %.3g)", fptmin, fptmax, multiplier));
            hDraw->Draw("E");

            // In bin-center mode the function is drawn against N_corr at each bin center.
            // In integral-counts mode the canvas shows density for variable-width bins.
            TF1* f = fitRes.fitfuncs[k];
            if (f) {
                f->SetLineWidth(2);
                f->SetLineColor(kRed);
                f->SetNpx(500);
                f->Draw("Same");
            }

            // Add legend
            TLegend* leg = new TLegend(0.15, 0.15, 0.45, 0.30);
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->SetTextFont(42);
            leg->SetTextSize(0.035);
            leg->AddEntry(hDraw, "Data", "lep");
            if (f) leg->AddEntry(f, "Exponential fit", "l");
            leg->Draw();

            // Annotate detailed fit information in a transparent box
            TPaveText* info = new TPaveText(0.45, 0.5, 0.95, 0.85, "NDC");
            info->SetFillColor(0);
            info->SetFillStyle(0);
            info->SetBorderSize(0);
            info->SetTextAlign(12);
            info->SetTextFont(42);
            info->SetTextSize(0.033);
            info->AddText(Form("Fit model: A #times exp(-x/c#tau) %g x He3(#sigma_{abso}) (%s)", multiplier, isMatter.c_str()));
            info->AddText(Form("Bin mode: %s", fitBinMode.c_str()));

            // global tau (already converted to real units) and error
            info->AddText(Form("#tau = %.4g #pm %.4g ps", fitRes.tau, fitRes.tauErr));
            info->AddText(Form("N_%d = %g", (int)k + 1, fitRes.fitfuncs[k]->GetParameter(0)));
            

            // global chi2/ndf
            info->AddText(Form("Global #chi^{2}/ndf = %.2f / %d", fitRes.chi2, fitRes.ndf));
            info->AddText(Form("Global fit probability = %.4g", fitRes.globalfitprobility));

            // per-channel chi2/ndf if available
            if (k < (int)fitRes.chi2PerChannel.size() && k < (int)fitRes.ndfPerChannel.size()) {
                info->AddText(Form("Channel #chi^{2}/ndf = %.2f / %d",
                                   fitRes.chi2PerChannel[k],
                                   fitRes.ndfPerChannel[k]));          
            }
            if (k < (int)fitRes.fitprobilitys.size()) {
                info->AddText(Form("Channel fit probability = %.4g",
                                   fitRes.fitprobilitys[k]));
            }

            // if more per-channel info exists, show neighbouring bins summary (optional compact)
            if (!fitRes.chi2PerChannel.empty()) {
                // show sum of other channels as context
                double sumChi2Others = 0;
                int sumNdfOthers = 0;
                for (size_t ii = 0; ii < fitRes.chi2PerChannel.size(); ++ii) {
                    if ((int)ii == k) continue;
                    sumChi2Others += fitRes.chi2PerChannel[ii];
                    sumNdfOthers += fitRes.ndfPerChannel[ii];
                }
                info->AddText(Form("Other channels #chi^{2}/ndf = %.2f / %d", sumChi2Others, sumNdfOthers));
            }

            info->Draw();

            // Improve canvas appearance
            cFit->SetGridx(0);
            cFit->SetGridy(0);
            cFit->SetLogy(1);
            cFit->SetTickx();
            cFit->SetTicky();
            cFit->Modified();
            cFit->Update();
            cFit->Write();
            const std::string ptMinName = FormatNumberForFilename(fptmin);
            const std::string ptMaxName = FormatNumberForFilename(fptmax);
            cFit->SaveAs(Form("%scrosssection_extraction_fit_pt_%s_%s_multi_%g.pdf",
                              outDir.c_str(), ptMinName.c_str(), ptMaxName.c_str(), multiplier));
            delete cFit;
            fitRes.fitfuncs[k]->Write();
        }
    }
    // create tau and chi2 distibutions
    TGraphErrors* grTau = new TGraphErrors((int)multipliers.size());
    grTau->SetName("grTauvsMultiplier");
    grTau->SetTitle("Fitted #tau vs Absorption Multiplier");
    grTau->GetXaxis()->SetTitle("n #times #sigma_{abso}(^{3}He)");
    grTau->GetYaxis()->SetTitle("Fitted #tau (ps)");
    grTau->SetMarkerStyle(21);
    grTau->SetMarkerSize(1.0);
    grTau->SetMarkerColor(kBlue + 1);
    TGraphErrors* grChi2 = new TGraphErrors((int)multipliers.size());
    grChi2->SetName("grChi2ndfvsMultiplier");
    grChi2->SetTitle("Fitted #chi^{2}/ndf vs Absorption Multiplier");
    grChi2->GetXaxis()->SetTitle("n #times #sigma_{abso}(^{3}He)");
    //grChi2->GetYaxis()->SetTitle("Fitted #chi^{2}/ndf");
    grChi2->GetYaxis()->SetTitle("Fitted #chi^{2}");
    grChi2->SetMarkerStyle(22);
    grChi2->SetMarkerSize(1.0);
    grChi2->SetMarkerColor(kRed + 1);
    TGraphErrors* grFitProbility = new TGraphErrors((int)multipliers.size());
    grFitProbility->SetName("grFitProbilityvsMultiplier");
    grFitProbility->SetTitle("Fitted Global Fit Probility vs Absorption Multiplier");
    grFitProbility->GetXaxis()->SetTitle("n #times #sigma_{abso}(^{3}He)");
    grFitProbility->GetYaxis()->SetTitle("Fitted Global Fit Probility");
    grFitProbility->SetMarkerStyle(23);
    grFitProbility->SetMarkerSize(1.0);
    grFitProbility->SetMarkerColor(kGreen + 1);
    for (int i = 0; i < (int)multipliers.size(); ++i) {
        //double chi2ndf = (ndfs[i] != 0) ? (chi2s[i] / ndfs[i]) : 0.0;
        double chi2 = chi2s[i];
        grTau->SetPoint(i, multipliers[i], taus[i]);
        grTau->SetPointError(i, 0.0, tauErrs[i]);
        //grChi2->SetPoint(i, multipliers[i], chi2ndf);
        grChi2->SetPoint(i, multipliers[i], chi2);
        grChi2->SetPointError(i, 0.0, 0.0);
        grFitProbility->SetPoint(i, multipliers[i], (ndfs[i] != 0) ? TMath::Prob(chi2s[i], ndfs[i]) : 0.0);
        grFitProbility->SetPointError(i, 0.0, 0.0);
    }
    auto drawAndSaveGraph = [&](TGraphErrors *graph,
                                 const char *canvasName,
                                 const char *legendLabel,
                                 const std::string &pdfPath,
                                 const char *yTitle,
                                 double yMin = -1,
                                 double yMax = -1) {
        if (!graph) {
            return;
        }
        auto canvas = std::make_unique<TCanvas>(canvasName, canvasName, 900, 650);
        canvas->SetLeftMargin(0.12);
        canvas->SetRightMargin(0.05);
        canvas->SetTopMargin(0.08);
        canvas->SetBottomMargin(0.13);
        canvas->SetGridx();
        canvas->SetGridy();
        canvas->SetTicks();
        graph->SetLineWidth(2);
        graph->SetMarkerSize(1.2);
        graph->SetTitle("");
        if (auto axis = graph->GetXaxis()) {
            axis->SetLabelSize(0.04);
            axis->SetTitleSize(0.045);
            axis->SetTitleOffset(1.0);
            axis->SetTitle("n #times #sigma_{abso}(^{3}He)");
        }
        if (auto axisY = graph->GetYaxis()) {
            axisY->SetLabelSize(0.04);
            axisY->SetTitleSize(0.045);
            axisY->SetTitleOffset(1.1);
            axisY->SetTitle(yTitle ? yTitle : "");
            if (yMin < yMax) {
                axisY->SetRangeUser(yMin, yMax);
            }
        }
        graph->Draw("APL");

        auto legend = std::make_unique<TLegend>(0.55, 0.75, 0.88, 0.9);
        legend->SetFillStyle(0);
        legend->SetBorderSize(0);
        legend->SetTextSize(0.042);
        legend->AddEntry(graph, legendLabel, "lep");
        legend->Draw();

        if (!pdfPath.empty()) {
            canvas->SaveAs(pdfPath.c_str());
        }
        canvas->Write();
    };

    const std::string tauPdf = outDir + "tau_vs_absorption_multiplier.pdf";
    const std::string chi2Pdf = outDir + "chi2_vs_absorption_multiplier.pdf";
    const std::string probPdf = outDir + "fitprob_vs_absorption_multiplier.pdf";

    drawAndSaveGraph(grTau, "c_grTau", "Fitted #tau", tauPdf, "Fitted #tau (ps)");
    drawAndSaveGraph(grChi2, "c_grChi2", "Global #chi^{2}", chi2Pdf, "Global #chi^{2}");
    drawAndSaveGraph(grFitProbility, "c_grFitProb", "Global fit probability", probPdf, "Fit probability", 0.0, 1.05);

    outfile->cd();
    grTau->Write();
    grChi2->Write();
    grFitProbility->Write();
    // write original histograms to output file
    TDirectory* dirOriginSpectra = outfile->mkdir("OriginalSpectra");
    dirOriginSpectra->cd();
    for (size_t j = 0; j < inputHists.size(); ++j) {
        inputHists[j]->Write();
    }
    outfile->Close();
}

void ProcessCrossSecction(const char* config_path = "../../configs/crosssection_config.json") {
    ProcessCrossSection(config_path);
}

// provide a simple main when compiled as standalone executable (no effect when used as ROOT macro)
#if !defined(__CLING__) && !defined(__CINT__)
int main(int argc, char** argv) {
    const char* cfg = (argc > 1) ? argv[1] : "config.json";
    ProcessCrossSection(cfg);
    return 0;
}
#endif
