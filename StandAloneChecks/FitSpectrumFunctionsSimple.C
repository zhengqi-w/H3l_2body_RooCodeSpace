#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TNtuple.h>
#include <TVirtualPad.h>
#include <TString.h>

#include <Math/MinimizerOptions.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../include/AliPWGFunc.cxx"

namespace {

using ParLimit = std::pair<double, double>;

struct FunctionConfig {
    std::vector<double> initial;
    std::vector<ParLimit> limits;
};

struct FitResult {
    std::string funcName;
    std::string centrality;
    int fitStatus;
    double chi2;
    int ndf;
    double mass;
    std::vector<double> initialValues;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
};

int GetFixedMassParamIndex(const std::string &funcName) {
    if (funcName == "fBGBW") return 0;
    if (funcName == "fLevi") return 3;
    if (funcName == "fTsallisBW") return 0;
    return -1;
}

int GetCanonicalParameterIndex(const std::string &funcName, int iCanonical) {
    if (funcName == "fBGBW") {
        // canonical: beta, T, n, norm; AliPWGFunc: mass, beta, T, n, norm
        return iCanonical + 1;
    }
    if (funcName == "fLevi") {
        // canonical: T, n, norm; AliPWGFunc: norm, n, T, mass
        static const int map[] = {2, 1, 0};
        return (iCanonical >= 0 && iCanonical < 3) ? map[iCanonical] : -1;
    }
    if (funcName == "fBoltzmann" || funcName == "fPtExp" ||
        funcName == "fMTExp" || funcName == "fBoseEinstein" || funcName == "fFermiDirac") {
        // canonical: T, norm; AliPWGFunc: norm, T
        static const int map[] = {1, 0};
        return (iCanonical >= 0 && iCanonical < 2) ? map[iCanonical] : -1;
    }
    if (funcName == "fPowerLaw") {
        // canonical: pt0, n, norm; AliPWGFunc: norm, pt0, n
        static const int map[] = {1, 2, 0};
        return (iCanonical >= 0 && iCanonical < 3) ? map[iCanonical] : -1;
    }
    if (funcName == "fTsallisBW") {
        // canonical: beta, T, q, norm, ymax; AliPWGFunc: mass, beta, T, q, norm, ymax
        return (iCanonical >= 0 && iCanonical < 5) ? iCanonical + 1 : -1;
    }
    return iCanonical;
}

int GetCanonicalNormIndex(const std::string &funcName) {
    if (funcName == "fBGBW") return 3;
    if (funcName == "fLevi") return 2;
    if (funcName == "fBoltzmann" || funcName == "fPtExp" ||
        funcName == "fMTExp" || funcName == "fBoseEinstein" || funcName == "fFermiDirac") return 1;
    if (funcName == "fPowerLaw") return 2;
    if (funcName == "fTsallisBW") return 3;
    return -1;
}

void ApplyFunctionConfig(TF1 *f, const std::string &funcName, const FunctionConfig &cfg) {
    if (!f) return;
    for (size_t i = 0; i < cfg.initial.size(); ++i) {
        const int ip = GetCanonicalParameterIndex(funcName, static_cast<int>(i));
        if (ip >= 0 && ip < f->GetNpar()) f->SetParameter(ip, cfg.initial[i]);
    }
    for (size_t i = 0; i < cfg.limits.size(); ++i) {
        const int ip = GetCanonicalParameterIndex(funcName, static_cast<int>(i));
        if (ip < 0 || ip >= f->GetNpar()) continue;
        const auto &lim = cfg.limits[i];
        if (lim.second > lim.first) f->SetParLimits(ip, lim.first, lim.second);
    }
}

void ConfigureFitDefaults() {
    static const bool configured = []() {
        ROOT::Math::MinimizerOptions::SetDefaultMaxIterations(20000);
        ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);
        ROOT::Math::MinimizerOptions::SetDefaultStrategy(2);
        return true;
    }();
    (void)configured;
}

std::unique_ptr<TF1> BuildFunction(const std::string &funcName,
                                   const std::string &nameTag,
                                   const std::vector<double> &userInit,
                                   const std::vector<ParLimit> &userLimits,
                                   double mass = 2.991) {
    // Keep the AliPWGFunc owner alive. In this ROOT setup a cloned BGBW TF1
    // does not move during fitting, while the original TF1 fits correctly.
    static std::vector<AliPWGFunc *> sAliPwgOwners;
    auto *helper = new AliPWGFunc();
    sAliPwgOwners.push_back(helper);
    helper->SetVarType(AliPWGFunc::kdNdpt);

    TF1 *raw = nullptr;
    if (funcName == "fBGBW") {
        raw = helper->GetBGBW(mass, 0.6, 0.2, 1.0, 500.0, ("fBGBW_" + nameTag).c_str());
    } else if (funcName == "fLevi") {
        raw = helper->GetLevi(mass, 0.45, 20.0, 1.0, ("fLevi_" + nameTag).c_str());
    } else if (funcName == "fBoltzmann") {
        raw = helper->GetBoltzmann(mass, 0.30, 1.0, ("fBoltzmann_" + nameTag).c_str());
    } else if (funcName == "fPtExp") {
        raw = helper->GetPTExp(0.35, 1.0, ("fPtExp_" + nameTag).c_str());
    } else if (funcName == "fMTExp") {
        raw = helper->GetMTExp(mass, 0.25, 1.0, ("fMTExp_" + nameTag).c_str());
    } else if (funcName == "fPowerLaw") {
        raw = helper->GetPowerLaw(1.0, 8.0, 1.0, ("fPowerLaw_" + nameTag).c_str());
    } else if (funcName == "fTsallisBW") {
        raw = helper->GetTsallisBW(mass, 0.76, 0.16, 1.003, 1.0, 0.03, ("fTsallisBW_" + nameTag).c_str());
    } else if (funcName == "fBoseEinstein") {
        raw = helper->GetBoseEinstein(mass, 0.25, 1.0, ("fBoseEinstein_" + nameTag).c_str());
    } else if (funcName == "fFermiDirac") {
        raw = helper->GetFermiDirac(mass, 0.25, 1.0, ("fFermiDirac_" + nameTag).c_str());
    }

    if (!raw) return nullptr;
    const int iMass = GetFixedMassParamIndex(funcName);
    if (iMass >= 0 && iMass < raw->GetNpar()) raw->FixParameter(iMass, mass);
    ApplyFunctionConfig(raw, funcName, FunctionConfig{userInit, userLimits});
    raw->SetRange(0.0, 10.0);
    raw->SetNpx(3000);
    raw->SetLineWidth(2);
    return std::unique_ptr<TF1>(raw);
}

std::vector<std::string> CollectCentralityDirs(TFile &f) {
    std::vector<std::string> out;
    TIter nextKey(f.GetListOfKeys());
    TKey *key = nullptr;
    while ((key = dynamic_cast<TKey *>(nextKey()))) {
        const std::string name = key->GetName();
        if (name.rfind("cen_", 0) == 0) out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string GetFreeParamDisplayName(const std::string &funcName, int iFree) {
    if (funcName == "fBGBW") {
        if (iFree == 0) return "beta";
        if (iFree == 1) return "T";
        if (iFree == 2) return "n";
        if (iFree == 3) return "norm";
    } else if (funcName == "fLevi") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "n";
        if (iFree == 2) return "norm";
    } else if (funcName == "fBoltzmann") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "norm";
    } else if (funcName == "fPtExp") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "norm";
    } else if (funcName == "fMTExp") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "norm";
    } else if (funcName == "fPowerLaw") {
        if (iFree == 0) return "pt0";
        if (iFree == 1) return "n";
        if (iFree == 2) return "norm";
    } else if (funcName == "fTsallisBW") {
        if (iFree == 0) return "beta";
        if (iFree == 1) return "T";
        if (iFree == 2) return "q";
        if (iFree == 3) return "norm";
        if (iFree == 4) return "ymax";
    } else if (funcName == "fBoseEinstein") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "norm";
    } else if (funcName == "fFermiDirac") {
        if (iFree == 0) return "T";
        if (iFree == 1) return "norm";
    }
    return Form("p%d", iFree);
}

int GetCanonicalParameterCount(const std::string &funcName) {
    if (funcName == "fBGBW") return 4;
    if (funcName == "fLevi") return 3;
    if (funcName == "fBoltzmann") return 2;
    if (funcName == "fPtExp") return 2;
    if (funcName == "fMTExp") return 2;
    if (funcName == "fPowerLaw") return 3;
    if (funcName == "fTsallisBW") return 5;
    if (funcName == "fBoseEinstein") return 2;
    if (funcName == "fFermiDirac") return 2;
    return 0;
}

double ClampToLimit(double value, const ParLimit &lim) {
    if (!(lim.second > lim.first)) return value;
    return std::min(lim.second, std::max(lim.first, value));
}

void SeedNormFromHistogram(TH1D *h, TF1 *f, const std::string &funcName,
                           const std::vector<ParLimit> &limitPars,
                           double fitMin, double fitMax) {
    if (!h || !f) return;
    const int iCanonNorm = GetCanonicalNormIndex(funcName);
    const int iParNorm = GetCanonicalParameterIndex(funcName, iCanonNorm);
    if (iCanonNorm < 0 || iParNorm < 0 || iParNorm >= f->GetNpar()) return;

    const double oldNorm = f->GetParameter(iParNorm);
    f->SetParameter(iParNorm, 1.0);
    double num = 0.0;
    double den = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double x = h->GetXaxis()->GetBinCenter(ib);
        if (x < fitMin || x > fitMax) continue;
        const double y = h->GetBinContent(ib);
        const double e = h->GetBinError(ib);
        if (!std::isfinite(y) || y <= 0.0 || !std::isfinite(e) || e <= 0.0) continue;
        const double g = f->Eval(x);
        if (!std::isfinite(g) || g <= 0.0) continue;
        const double w = 1.0 / (e * e);
        num += y * g * w;
        den += g * g * w;
    }

    double seededNorm = (den > 0.0) ? (num / den) : oldNorm;
    if (!std::isfinite(seededNorm) || seededNorm <= 0.0) seededNorm = oldNorm;
    if (static_cast<size_t>(iCanonNorm) < limitPars.size()) {
        seededNorm = ClampToLimit(seededNorm, limitPars[static_cast<size_t>(iCanonNorm)]);
    }
    f->SetParameter(iParNorm, seededNorm);
}

int CountPositiveFitPoints(TH1D *h, double fitMin, double fitMax) {
    if (!h) return 0;
    int n = 0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double x = h->GetXaxis()->GetBinCenter(ib);
        if (x < fitMin || x > fitMax) continue;
        const double y = h->GetBinContent(ib);
        const double e = h->GetBinError(ib);
        if (std::isfinite(y) && y > 0.0 && std::isfinite(e) && e > 0.0) ++n;
    }
    return n;
}

void FixCanonicalParameter(TF1 *f, const std::string &funcName, int iCanonical) {
    if (!f) return;
    const int ip = GetCanonicalParameterIndex(funcName, iCanonical);
    if (ip >= 0 && ip < f->GetNpar()) f->FixParameter(ip, f->GetParameter(ip));
}

void StabilizeSparseFit(TH1D *h, TF1 *f, const std::string &funcName,
                        double fitMin, double fitMax) {
    const int nPoints = CountPositiveFitPoints(h, fitMin, fitMax);
    if (!f || nPoints <= 0) return;
    const int maxFree = std::max(1, nPoints - 1);
    int nFree = GetCanonicalParameterCount(funcName);

    auto fixAndCount = [&](int iCanonical) {
        if (nFree <= maxFree) return;
        FixCanonicalParameter(f, funcName, iCanonical);
        --nFree;
    };

    if (funcName == "fBGBW") {
        fixAndCount(2); // n
        fixAndCount(0); // beta
        fixAndCount(1); // T
    } else if (funcName == "fLevi") {
        fixAndCount(1); // n
        fixAndCount(0); // T
    } else if (funcName == "fBoltzmann" || funcName == "fPtExp" ||
               funcName == "fMTExp" || funcName == "fBoseEinstein" ||
               funcName == "fFermiDirac") {
        fixAndCount(0); // T
    } else if (funcName == "fPowerLaw") {
        fixAndCount(1); // n
        fixAndCount(0); // pt0
    } else if (funcName == "fTsallisBW") {
        fixAndCount(4); // ymax
        fixAndCount(2); // q
        fixAndCount(0); // beta
        fixAndCount(1); // T
    }
}

std::vector<std::vector<double>> BuildSeedGrid(const std::string &funcName,
                                               const std::vector<double> &baseInit) {
    std::vector<std::vector<double>> seeds;
    if (!baseInit.empty()) seeds.push_back(baseInit);

    if (funcName == "fBGBW") {
        for (double beta : {0.45, 0.60, 0.72})
            for (double temp : {0.10, 0.16, 0.24, 0.34})
                for (double n : {0.5, 1.0, 2.5})
                    seeds.push_back({beta, temp, n, 1.0});
    } else if (funcName == "fLevi") {
        for (double temp : {0.06, 0.09, 0.12, 0.18, 0.28, 0.45, 0.70, 1.00})
            for (double n : {2.1, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0, 150.0, 300.0, 500.0})
                seeds.push_back({temp, n, 1.0});
    } else if (funcName == "fBoltzmann" || funcName == "fMTExp" ||
               funcName == "fBoseEinstein" || funcName == "fFermiDirac") {
        for (double temp : {0.06, 0.08, 0.10, 0.14, 0.18, 0.23, 0.30, 0.40, 0.60, 0.90, 1.20})
            seeds.push_back({temp, 1.0});
    } else if (funcName == "fPtExp") {
        for (double temp : {0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.35, 0.60, 0.90, 1.20, 1.80, 2.50})
            seeds.push_back({temp, 1.0});
    } else if (funcName == "fPowerLaw") {
        for (double pt0 : {0.6, 1.0, 1.5, 2.5, 4.0})
            for (double n : {3.0, 5.0, 8.0, 12.0, 20.0})
                seeds.push_back({pt0, n, 1.0});
    } else if (funcName == "fTsallisBW") {
        for (double beta : {0.35, 0.55, 0.70, 0.76, 0.82})
            for (double temp : {0.04, 0.08, 0.12, 0.16, 0.20, 0.28})
                for (double q : {1.0005, 1.002, 1.005, 1.010, 1.030, 1.080})
                    for (double ymax : {0.01, 0.03, 0.10, 0.50})
                        seeds.push_back({beta, temp, q, 1.0, ymax});
    }

    return seeds;
}

std::unique_ptr<TF1> FitBestSeed(TH1D *h,
                                 const std::string &funcName,
                                 const std::string &tag,
                                 const std::vector<double> &initPars,
                                 const std::vector<ParLimit> &limitPars,
                                 double mass,
                                 double fitMin,
                                 double fitMax,
                                 FitResult &result) {
    std::unique_ptr<TF1> bestFunc;
    double bestScore = std::numeric_limits<double>::infinity();
    int iSeed = 0;
    for (const auto &seed : BuildSeedGrid(funcName, initPars)) {
        auto f = BuildFunction(funcName,
                               tag + "_" + funcName + Form("_seed%d", iSeed),
                               seed,
                               limitPars,
                               mass);
        ++iSeed;
        if (!f) continue;
        SeedNormFromHistogram(h, f.get(), funcName, limitPars, fitMin, fitMax);
        StabilizeSparseFit(h, f.get(), funcName, fitMin, fitMax);
        const std::vector<double> start = [&]() {
            std::vector<double> out;
            for (int ic = 0; ic < GetCanonicalParameterCount(funcName); ++ic) {
                const int ip = GetCanonicalParameterIndex(funcName, ic);
                out.push_back((ip >= 0 && ip < f->GetNpar()) ? f->GetParameter(ip) : 0.0);
            }
            return out;
        }();
        const int fitStatus = static_cast<int>(h->Fit(f.get(), "Q0RSN", "", fitMin, fitMax));
        const double chi2 = f->GetChisquare();
        const int ndf = f->GetNDF();
        const double score = (ndf > 0 && std::isfinite(chi2)) ? (chi2 / static_cast<double>(ndf)) :
                             (std::isfinite(chi2) ? chi2 : std::numeric_limits<double>::infinity());
        const bool converged = (fitStatus == 0);
        const double rankedScore = score * (converged ? 1.0 : 1.05);
        if (rankedScore < bestScore) {
            bestScore = rankedScore;
            result.fitStatus = fitStatus;
            result.chi2 = chi2;
            result.ndf = ndf;
            result.initialValues = start;
            bestFunc = std::move(f);
        }
    }
    return bestFunc;
}

FitResult ExtractFitResult(TH1D *hSrc,
                           const std::string &funcName,
                           const std::string &tag,
                           int color,
                           const std::vector<double> &initPars,
                           const std::vector<ParLimit> &limitPars,
                           double mass,
                           bool drawHistogram = true) {
    FitResult result;
    result.funcName = funcName;
    result.centrality = tag;
    result.mass = mass;
    result.fitStatus = -1;
    result.chi2 = 0.0;
    result.ndf = 0;
    
    if (!hSrc) {
        return result;
    }

    auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(hSrc->Clone(("h_" + tag + "_" + funcName).c_str())));
    h->SetDirectory(nullptr);
    h->SetStats(false);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(0.8);
    h->SetMarkerColor(kBlack);
    h->SetLineColor(kBlack);

    const double fitMin = h->GetBinLowEdge(1);
    const double fitMax = h->GetBinLowEdge(h->GetNbinsX() + 1);
    ConfigureFitDefaults();
    auto f = FitBestSeed(h.get(), funcName, tag, initPars, limitPars, mass, fitMin, fitMax, result);
    if (!f) {
        return result;
    }
    
    // Extract parameters in the same canonical order used by the config.
    const int iMass = GetFixedMassParamIndex(funcName);
    if (iMass >= 0 && iMass < f->GetNpar()) {
        result.paramNames.push_back("mass");
        result.paramValues.push_back(f->GetParameter(iMass));
    }
    
    for (int ic = 0; ic < GetCanonicalParameterCount(funcName); ++ic) {
        const int ip = GetCanonicalParameterIndex(funcName, ic);
        if (ip < 0 || ip >= f->GetNpar()) continue;
        const std::string pname = GetFreeParamDisplayName(funcName, ic);
        result.paramNames.push_back(pname);
        result.paramValues.push_back(f->GetParameter(ip));
    }

    TH1D *hDraw = nullptr;
    if (drawHistogram) {
        hDraw = dynamic_cast<TH1D *>(h->DrawCopy("E1"));
    } else if (gPad) {
        // In function-only mode, derive y-range directly from fitted function.
        const double drawXMin = 0.0;
        const double drawXMax = std::max(fitMax, drawXMin + 1e-6);
        double yMin = 0.0;
        double yMax = 0.0;
        bool hasValue = false;
        constexpr int nSample = 800;
        for (int i = 0; i < nSample; ++i) {
            const double x = drawXMin + (drawXMax - drawXMin) * static_cast<double>(i) / static_cast<double>(nSample - 1);
            const double y = f->Eval(x);
            if (!std::isfinite(y)) continue;
            if (!hasValue) {
                yMin = y;
                yMax = y;
                hasValue = true;
            } else {
                yMin = std::min(yMin, y);
                yMax = std::max(yMax, y);
            }
        }

        if (!hasValue) {
            yMin = 1e-8;
            yMax = 1.0;
        }

        if (gPad->GetLogy()) {
            if (yMax <= 0.0) {
                yMin = 1e-8;
                yMax = 1.0;
            } else {
                const double yPosMin = (yMin > 0.0) ? yMin : (yMax * 1e-4);
                yMin = std::max(1e-12, yPosMin / 1.8);
                yMax = yMax * 2.5;
            }
        } else {
            if (std::abs(yMax - yMin) < 1e-12) {
                const double s = std::max(1.0, std::abs(yMax));
                yMin -= 0.5 * s;
                yMax += 0.5 * s;
            } else {
                const double pad = 0.15 * (yMax - yMin);
                yMin -= pad;
                yMax += pad;
            }
        }

        auto *frame = gPad->DrawFrame(drawXMin, yMin, drawXMax, yMax);
        if (frame) {
            frame->SetTitle(h->GetTitle());
            frame->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
            frame->GetYaxis()->SetTitle(h->GetYaxis()->GetTitle());
        }
    }

    f->SetLineColor(color);
    auto *fDraw = dynamic_cast<TF1 *>(f->DrawCopy("SAME"));

    TLegend leg(0.58, 0.76, 0.90, 0.90);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    if (drawHistogram && hDraw) leg.AddEntry(hDraw, "h_corrected_counts", "lep");
    if (fDraw) leg.AddEntry(fDraw, (funcName + " fit").c_str(), "l");
    leg.Draw();

    TLatex tl;
    tl.SetNDC();
    tl.SetTextSize(0.045);
    tl.DrawLatex(0.15, 0.90, funcName.c_str());
    tl.DrawLatex(0.15, 0.84, Form("fit status = %d", result.fitStatus));

    double y = 0.72;
    if (result.fitStatus == 0) {
        const int ndf = f->GetNDF();
        const double chi2ndf = (ndf > 0) ? (f->GetChisquare() / static_cast<double>(ndf)) : 0.0;
        tl.DrawLatex(0.15, 0.78, Form("#chi^{2}/ndf = %.3f", chi2ndf));
    } else {
        tl.DrawLatex(0.15, 0.78, "fit not converged; showing last parameters");
    }

    const int iMass2 = GetFixedMassParamIndex(funcName);
    if (iMass2 >= 0 && iMass2 < f->GetNpar()) {
        tl.DrawLatex(0.15, y, Form("mass (fixed) = %.4g", f->GetParameter(iMass2)));
        y -= 0.055;
    }

    for (int ic = 0; ic < GetCanonicalParameterCount(funcName); ++ic) {
        const int ip = GetCanonicalParameterIndex(funcName, ic);
        if (ip < 0 || ip >= f->GetNpar()) continue;
        const std::string pname = GetFreeParamDisplayName(funcName, ic);
        tl.DrawLatex(0.15, y, Form("%s = %.4g", pname.c_str(), f->GetParameter(ip)));
        y -= 0.055;
        if (y < 0.12) break;
    }
    
    return result;
}

} // namespace

int FitSpectrumFunctionsSimple(
    const char *inputRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/FitFunctionScan",
    double mass = 2.991,
    const std::vector<double> &initBGBW = {0.6, 0.2, 1.0, 500.0},
    const std::vector<double> &initLevi = {0.45, 20.0, 1.0},
    const std::vector<double> &initBoltzmann = {0.30, 1.0},
    const std::vector<double> &initPtExp = {1.2, 1.0},
    const std::vector<double> &initTsallisBW = {0.76, 0.16, 1.003, 1.0, 0.03},
    const std::vector<ParLimit> &limitBGBW = {{0.2, 0.9}, {0.1, 0.6}, {0.01, 5.0}, {1e-10, 1e9}},
    const std::vector<ParLimit> &limitLevi = {{0.03, 2.0}, {2.0, 1e3}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitBoltzmann = {{0.03, 2.0}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitPtExp = {{0.03, 2.0}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitTsallisBW = {{0.0, 0.95}, {0.005, 0.6}, {1.000001, 1.3}, {1e-10, 1e12}, {0.001, 2.0}}) {

    std::filesystem::create_directories(outputDir);

    std::unique_ptr<TFile> fin(TFile::Open(inputRoot, "READ"));
    if (!fin || fin->IsZombie()) {
        std::cerr << "[Error] Cannot open input file: " << inputRoot << std::endl;
        return 1;
    }

    const std::vector<std::string> cenDirs = CollectCentralityDirs(*fin);
    if (cenDirs.empty()) {
        std::cerr << "[Error] No cen_* directories found in: " << inputRoot << std::endl;
        return 1;
    }

    const std::vector<std::string> funcs = {"fBGBW", "fLevi", "fBoltzmann", "fPtExp", "fMTExp", "fPowerLaw", "fTsallisBW"};
    const std::vector<int> colors = {kRed + 1, kBlue + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2, kViolet + 1};
    const std::vector<std::vector<double>> initParsByFunc = {
        initBGBW,
        initLevi,
        initBoltzmann,
        initPtExp,
        {0.30, 1.0},
        {1.2, 8.0, 1.0},
        initTsallisBW};
    const std::vector<std::vector<ParLimit>> limitParsByFunc = {
        limitBGBW,
        limitLevi,
        limitBoltzmann,
        limitPtExp,
        {{0.05, 2.5}, {1e-10, 1e12}},
        {{0.03, 10.0}, {2.01, 120.0}, {1e-10, 1e12}},
        limitTsallisBW};

    // Collect all fit results
    std::vector<FitResult> allResults;

    for (const auto &cen : cenDirs) {
        const std::string hPath = cen + "/std/h_corrected_counts";
        auto *hIn = dynamic_cast<TH1D *>(fin->Get(hPath.c_str()));
        if (!hIn) {
            std::cout << "[Warn] Missing histogram: " << hPath << std::endl;
            continue;
        }

        auto c = std::make_unique<TCanvas>(("c_fit_funcs_" + cen).c_str(), ("Fit scan " + cen).c_str(), 1300, 900);
        c->Divide(3, 3);
        auto cOnly = std::make_unique<TCanvas>(("c_fit_funcs_only_" + cen).c_str(), ("Fit functions only " + cen).c_str(), 1300, 900);
        cOnly->Divide(3, 3);

        for (size_t i = 0; i < funcs.size(); ++i) {
            c->cd(i + 1);
            gPad->SetLogy();
            gPad->SetLeftMargin(0.12);
            gPad->SetBottomMargin(0.12);
            FitResult res1 = ExtractFitResult(hIn,
                                             funcs[i],
                                             cen,
                                             colors[i],
                                             initParsByFunc[i],
                                             limitParsByFunc[i],
                                             mass,
                                             true);
            allResults.push_back(res1);

            cOnly->cd(i + 1);
            gPad->SetLogy();
            gPad->SetLeftMargin(0.12);
            gPad->SetBottomMargin(0.12);
            ExtractFitResult(hIn,
                             funcs[i],
                             cen,
                             colors[i],
                             initParsByFunc[i],
                             limitParsByFunc[i],
                             mass,
                             false);
        }

        c->cd();
        c->SaveAs((std::string(outputDir) + "/fit_function_scan_" + cen + ".pdf").c_str());
        std::cout << "[Info] Saved: " << (std::string(outputDir) + "/fit_function_scan_" + cen + ".pdf") << std::endl;
        cOnly->cd();
        cOnly->SaveAs((std::string(outputDir) + "/fit_function_only_scan_" + cen + ".pdf").c_str());
        std::cout << "[Info] Saved: " << (std::string(outputDir) + "/fit_function_only_scan_" + cen + ".pdf") << std::endl;
    }

    // Output results to text file
    std::ofstream txtOut((std::string(outputDir) + "/fit_results.txt").c_str());
    if (txtOut.is_open()) {
        txtOut << "=== Fit Results Summary ===" << std::endl;
        txtOut << "Mass (fixed): " << mass << " GeV" << std::endl << std::endl;
        
        for (const auto &result : allResults) {
            txtOut << "Centrality: " << result.centrality << std::endl;
            txtOut << "Function: " << result.funcName << std::endl;
            txtOut << "Fit Status: " << result.fitStatus << std::endl;
            if (result.ndf > 0) {
                txtOut << "Chi2/ndf: " << (result.chi2 / result.ndf) << std::endl;
            }
            txtOut << "Parameters:" << std::endl;
            txtOut << "Selected initial values:" << std::endl;
            for (size_t i = 0; i < result.initialValues.size(); ++i) {
                txtOut << "  " << GetFreeParamDisplayName(result.funcName, static_cast<int>(i))
                       << "_init = " << std::setprecision(6) << std::scientific
                       << result.initialValues[i] << std::endl;
            }
            for (size_t i = 0; i < result.paramNames.size(); ++i) {
                txtOut << "  " << result.paramNames[i] << " = " << std::setprecision(6) 
                       << std::scientific << result.paramValues[i] << std::endl;
            }
            txtOut << std::endl;
        }
        txtOut.close();
        std::cout << "[Info] Saved: " << (std::string(outputDir) + "/fit_results.txt") << std::endl;
    }

    // Output results to ROOT file with hierarchical structure
    std::unique_ptr<TFile> fOut(TFile::Open((std::string(outputDir) + "/fit_results.root").c_str(), "RECREATE"));
    if (fOut && !fOut->IsZombie()) {
        // Create directory structure: cen_X_Y/std/funcName
        std::map<std::string, std::map<std::string, std::vector<FitResult>>> resultsByHierarchy;
        for (const auto &res : allResults) {
            resultsByHierarchy[res.centrality]["std"].push_back(res);
        }
        
        for (const auto &[cenKey, cenMap] : resultsByHierarchy) {
            TDirectory *cenDir = fOut->mkdir(cenKey.c_str());
            if (!cenDir) continue;
            for (const auto &[subKey, resultList] : cenMap) {
                TDirectory *subDir = cenDir->mkdir(subKey.c_str());
                if (!subDir) continue;
                subDir->cd();
                
                for (const auto &res : resultList) {
                    std::string funcLower = res.funcName;
                    std::transform(funcLower.begin(), funcLower.end(), funcLower.begin(), ::tolower);
                    std::string treeName = funcLower + "_" + cenKey;
                    
                    // Write TNtuple with fit results
                    double mass_val = res.mass;
                    double fitStatus_val = res.fitStatus;
                    double chi2_val = res.chi2;
                    double ndf_val = res.ndf;
                    
                    TNtuple *nt = new TNtuple(treeName.c_str(), "Fit Results", 
                                             "mass:fitStatus:chi2:ndf");
                    nt->Fill(mass_val, fitStatus_val, chi2_val, ndf_val);
                    nt->Write();
                    delete nt;
                }
            }
        }
        fOut->Close();
        std::cout << "[Info] Saved: " << (std::string(outputDir) + "/fit_results.root") << std::endl;
    }

    return 0;
}
