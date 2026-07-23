#include <TCanvas.h>
#include <TDirectory.h>
#include <TError.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
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

struct TsallisBWShapeMetrics {
    double maxAbsLogCurvature = 0.0;
    double rmsLogCurvature = 0.0;
    double upturnStrength = 0.0;
    double slopeChangeCount = 0.0;
};

struct TsallisBWRangeScanCase {
    std::string name;
    std::vector<double> init;
    std::vector<ParLimit> limits;
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
        raw = helper->GetTsallisBW(mass, 0.68, 0.14, 1.04, 1.0, 0.50, ("fTsallisBW_" + nameTag).c_str());
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
    raw->SetNpx(funcName == "fTsallisBW" ? 8000 : 3000);
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

bool IsFixedParameter(const TF1 *f, int ip) {
    if (!f || ip < 0 || ip >= f->GetNpar()) return true;
    double low = 0.0;
    double high = 0.0;
    f->GetParLimits(ip, low, high);
    return (high > low && std::abs(high - low) < 1e-15);
}

double BoundaryPenalty(TF1 *f,
                       const std::string &funcName,
                       const std::vector<ParLimit> &limitPars) {
    if (!f) return 1.0;
    double penalty = 1.0;
    for (size_t ic = 0; ic < limitPars.size(); ++ic) {
        const auto &lim = limitPars[ic];
        if (!(lim.second > lim.first)) continue;
        const int ip = GetCanonicalParameterIndex(funcName, static_cast<int>(ic));
        if (ip < 0 || ip >= f->GetNpar()) continue;
        if (IsFixedParameter(f, ip)) continue;
        const double v = f->GetParameter(ip);
        if (!std::isfinite(v)) return 1e6;
        const double width = lim.second - lim.first;
        if (width <= 0.0) continue;
        const double fracLow = (v - lim.first) / width;
        const double fracHigh = (lim.second - v) / width;
        const double edgeFrac = std::min(fracLow, fracHigh);
        if (edgeFrac < 0.010) penalty *= 8.0;
        else if (edgeFrac < 0.025) penalty *= 3.0;
        else if (edgeFrac < 0.050) penalty *= 1.5;
    }
    if (funcName == "fTsallisBW") {
        const int iq = GetCanonicalParameterIndex(funcName, 2);
        if (iq >= 0 && iq < f->GetNpar() && !IsFixedParameter(f, iq)) {
            const double q = f->GetParameter(iq);
            if (q < 1.015) penalty *= 20.0;
            else if (q < 1.025) penalty *= 4.0;
        }
    }
    return penalty;
}

double LowPtSmoothnessPenalty(TF1 *f, double xMin = 0.05, double xMax = 2.0) {
    if (!f || !(xMax > xMin)) return 1.0;
    constexpr int n = 80;
    std::vector<double> logY;
    logY.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double x = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(n - 1);
        const double y = f->Eval(x);
        if (!std::isfinite(y) || y <= 0.0) return 1e6;
        logY.push_back(std::log(y));
    }

    double maxCurvature = 0.0;
    double sumCurvature2 = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        const double c = logY[i + 1] - 2.0 * logY[i] + logY[i - 1];
        maxCurvature = std::max(maxCurvature, std::abs(c));
        sumCurvature2 += c * c;
    }
    const double rmsCurvature = std::sqrt(sumCurvature2 / static_cast<double>(n - 2));

    double penalty = 1.0;
    if (maxCurvature > 0.030) penalty *= 1.0 + 80.0 * (maxCurvature - 0.030);
    if (rmsCurvature > 0.012) penalty *= 1.0 + 120.0 * (rmsCurvature - 0.012);
    return penalty;
}

TsallisBWShapeMetrics MeasureTsallisBWLowPtShape(TF1 *f, double xMin = 0.05, double xMax = 2.0) {
    TsallisBWShapeMetrics metrics;
    if (!f || !(xMax > xMin)) {
        metrics.maxAbsLogCurvature = std::numeric_limits<double>::infinity();
        metrics.rmsLogCurvature = std::numeric_limits<double>::infinity();
        metrics.upturnStrength = std::numeric_limits<double>::infinity();
        metrics.slopeChangeCount = std::numeric_limits<double>::infinity();
        return metrics;
    }

    constexpr int n = 120;
    std::vector<double> logY;
    logY.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double x = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(n - 1);
        const double y = f->Eval(x);
        if (!std::isfinite(y) || y <= 0.0) {
            metrics.maxAbsLogCurvature = std::numeric_limits<double>::infinity();
            metrics.rmsLogCurvature = std::numeric_limits<double>::infinity();
            metrics.upturnStrength = std::numeric_limits<double>::infinity();
            metrics.slopeChangeCount = std::numeric_limits<double>::infinity();
            return metrics;
        }
        logY.push_back(std::log(y));
    }

    std::vector<double> slope;
    slope.reserve(n - 1);
    for (int i = 0; i < n - 1; ++i) slope.push_back(logY[i + 1] - logY[i]);

    double sumCurv2 = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        const double c = logY[i + 1] - 2.0 * logY[i] + logY[i - 1];
        metrics.maxAbsLogCurvature = std::max(metrics.maxAbsLogCurvature, std::abs(c));
        sumCurv2 += c * c;
    }
    metrics.rmsLogCurvature = std::sqrt(sumCurv2 / static_cast<double>(n - 2));

    double minSlope = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < slope.size(); ++i) {
        minSlope = std::min(minSlope, slope[i]);
        metrics.upturnStrength = std::max(metrics.upturnStrength, slope[i] - minSlope);
        if (i > 0 && slope[i] * slope[i - 1] < 0.0) metrics.slopeChangeCount += 1.0;
    }
    return metrics;
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
        for (double beta : {0.50, 0.68, 0.82})
            for (double temp : {0.07, 0.12, 0.18, 0.28})
                for (double q : {1.020, 1.040, 1.070, 1.110})
                    for (double ymax : {0.30, 0.60, 1.00})
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
                                 FitResult &result,
                                 bool penalizeLowPtShape = false,
                                 bool useSeedGrid = true) {
    std::unique_ptr<TF1> bestFunc;
    double bestScore = std::numeric_limits<double>::infinity();
    int iSeed = 0;
    const std::vector<std::vector<double>> seeds = useSeedGrid ? BuildSeedGrid(funcName, initPars)
                                                               : std::vector<std::vector<double>>{initPars};
    for (const auto &seed : seeds) {
        std::vector<double> seedClamped = seed;
        for (size_t i = 0; i < seedClamped.size() && i < limitPars.size(); ++i) {
            seedClamped[i] = ClampToLimit(seedClamped[i], limitPars[i]);
        }
        auto f = BuildFunction(funcName,
                               tag + "_" + funcName + Form("_seed%d", iSeed),
                               seedClamped,
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
        double rankedScore = score * (converged ? 1.0 : 1.05) * BoundaryPenalty(f.get(), funcName, limitPars);
        if (penalizeLowPtShape && funcName == "fTsallisBW") {
            rankedScore *= LowPtSmoothnessPenalty(f.get(), 0.05, 2.0);
        }
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
    const std::vector<double> &initTsallisBW = {0.68, 0.14, 1.04, 1.0, 0.50},
    const std::vector<ParLimit> &limitBGBW = {{0.2, 0.9}, {0.1, 0.6}, {0.01, 5.0}, {1e-10, 1e9}},
    const std::vector<ParLimit> &limitLevi = {{0.03, 2.0}, {2.0, 1e3}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitBoltzmann = {{0.03, 2.0}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitPtExp = {{0.03, 2.0}, {1e-9, 0.1}},
    const std::vector<ParLimit> &limitTsallisBW = {{0.15, 0.90}, {0.025, 0.45}, {1.010, 1.180}, {1e-10, 1e12}, {0.10, 1.50}}) {

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

int FitTsallisBWLowPtMerged(
    const char *inputRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/FitTsallisBWLowPtMerged",
    double mass = 2.991,
    const std::vector<double> &initTsallisBW = {0.60, 0.11, 1.04, 1.0, 0.60},
    const std::vector<ParLimit> &limitTsallisBW = {{0.30, 0.75}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}) {
    std::filesystem::create_directories(outputDir);

    std::unique_ptr<TFile> fin(TFile::Open(inputRoot, "READ"));
    if (!fin || fin->IsZombie()) {
        std::cerr << "[Error] Cannot open input file: " << inputRoot << std::endl;
        return 1;
    }

    auto cenDirs = CollectCentralityDirs(*fin);
    if (cenDirs.empty()) {
        std::cerr << "[Error] No cen_* directories found in: " << inputRoot << std::endl;
        return 1;
    }

    ConfigureFitDefaults();
    std::vector<FitResult> results;
    std::ofstream txtOut((std::string(outputDir) + "/tsallisbw_lowpt_parameters.txt").c_str());
    txtOut << "=== TsallisBW low-pT tuned fit ===\n";
    txtOut << "Input: " << inputRoot << "\n";
    txtOut << "Low-pT smoothness penalty applied in 0.05-2.0 GeV/c\n";
    txtOut << "Limits: beta [" << limitTsallisBW[0].first << ", " << limitTsallisBW[0].second
           << "], T [" << limitTsallisBW[1].first << ", " << limitTsallisBW[1].second
           << "], q [" << limitTsallisBW[2].first << ", " << limitTsallisBW[2].second
           << "], ymax [" << limitTsallisBW[4].first << ", " << limitTsallisBW[4].second << "]\n\n";

    for (const auto &cen : cenDirs) {
        const std::string hPath = cen + "/std/h_corrected_counts";
        auto *hIn = dynamic_cast<TH1D *>(fin->Get(hPath.c_str()));
        if (!hIn) {
            std::cerr << "[Warn] Missing histogram: " << hPath << std::endl;
            continue;
        }

        auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(hIn->Clone(("h_lowpt_" + cen).c_str())));
        h->SetDirectory(nullptr);
        h->SetStats(false);
        h->SetMarkerStyle(kFullCircle);
        h->SetMarkerSize(0.95);
        h->SetMarkerColor(kBlack);
        h->SetLineColor(kBlack);

        const double fitMin = h->GetBinLowEdge(1);
        const double fitMax = h->GetBinLowEdge(h->GetNbinsX() + 1);
        FitResult result;
        result.funcName = "fTsallisBW";
        result.centrality = cen;
        result.mass = mass;
        result.fitStatus = -1;
        result.chi2 = 0.0;
        result.ndf = 0;
        auto f = FitBestSeed(h.get(), "fTsallisBW", cen + "_lowpt", initTsallisBW, limitTsallisBW, mass, fitMin, fitMax, result, true);
        if (!f) continue;

        const int iMass = GetFixedMassParamIndex("fTsallisBW");
        if (iMass >= 0 && iMass < f->GetNpar()) {
            result.paramNames.push_back("mass");
            result.paramValues.push_back(f->GetParameter(iMass));
        }
        for (int ic = 0; ic < GetCanonicalParameterCount("fTsallisBW"); ++ic) {
            const int ip = GetCanonicalParameterIndex("fTsallisBW", ic);
            if (ip < 0 || ip >= f->GetNpar()) continue;
            result.paramNames.push_back(GetFreeParamDisplayName("fTsallisBW", ic));
            result.paramValues.push_back(f->GetParameter(ip));
        }
        results.push_back(result);

        f->SetLineColor(kViolet + 1);
        f->SetLineWidth(3);
        f->SetNpx(10000);

        auto c = std::make_unique<TCanvas>(("c_tsallisbw_lowpt_" + cen).c_str(), ("TsallisBW low-pT " + cen).c_str(), 1200, 650);
        c->Divide(2, 1, 0.01, 0.0);

        c->cd(1);
        gPad->SetLogy();
        gPad->SetLeftMargin(0.14);
        gPad->SetBottomMargin(0.12);
        h->Draw("E1");
        h->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
        h->GetYaxis()->SetTitle("corrected counts");
        h->GetYaxis()->SetTitleOffset(1.35);
        f->Draw("SAME");
        TLatex text;
        text.SetNDC();
        text.SetTextFont(42);
        text.SetTextSize(0.038);
        text.DrawLatex(0.18, 0.88, cen.c_str());
        text.SetTextSize(0.032);
        text.DrawLatex(0.18, 0.82, Form("#chi^{2}/NDF = %.2f/%d", f->GetChisquare(), f->GetNDF()));

        c->cd(2);
        gPad->SetLeftMargin(0.14);
        gPad->SetBottomMargin(0.12);
        double yMaxLow = 0.0;
        double yMinLow = 1e30;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetBinCenter(ib);
            if (x > 2.2) continue;
            const double y = h->GetBinContent(ib);
            const double e = h->GetBinError(ib);
            if (y > 0.0) {
                yMinLow = std::min(yMinLow, std::max(0.0, y - e));
                yMaxLow = std::max(yMaxLow, y + e);
            }
        }
        for (int i = 0; i < 250; ++i) {
            const double x = 0.0 + 2.2 * static_cast<double>(i) / 249.0;
            const double y = f->Eval(x);
            if (std::isfinite(y) && y > 0.0) {
                yMinLow = std::min(yMinLow, y);
                yMaxLow = std::max(yMaxLow, y);
            }
        }
        if (!(yMaxLow > 0.0)) yMaxLow = 1.0;
        if (!(yMinLow < yMaxLow)) yMinLow = 0.0;
        auto *frame = gPad->DrawFrame(0.0, std::max(0.0, yMinLow * 0.85), 2.2, yMaxLow * 1.20,
                                      ";#it{p}_{T} (GeV/#it{c});corrected counts");
        frame->GetYaxis()->SetTitleOffset(1.35);
        h->Draw("E1 SAME");
        f->Draw("SAME");
        auto *lowLine = new TLine(2.0, std::max(0.0, yMinLow * 0.85), 2.0, yMaxLow * 1.20);
        lowLine->SetLineStyle(2);
        lowLine->SetLineColor(kGray + 2);
        lowLine->Draw("SAME");
        text.SetTextSize(0.034);
        text.DrawLatex(0.18, 0.88, "low-#it{p}_{T} shape check");
        text.DrawLatex(0.18, 0.82, Form("q = %.4f", f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 2))));

        const std::string outPdf = std::string(outputDir) + "/tsallisbw_lowpt_" + cen + ".pdf";
        c->SaveAs(outPdf.c_str());
        std::cout << "[Info] Saved: " << outPdf << std::endl;

        txtOut << "Centrality: " << cen << "\n";
        txtOut << "Fit Status: " << result.fitStatus << "\n";
        txtOut << "Chi2/ndf: " << (result.ndf > 0 ? result.chi2 / static_cast<double>(result.ndf) : 0.0) << "\n";
        txtOut << "Selected initial values:\n";
        for (size_t i = 0; i < result.initialValues.size(); ++i) {
            txtOut << "  " << GetFreeParamDisplayName("fTsallisBW", static_cast<int>(i))
                   << "_init = " << std::setprecision(6) << std::scientific << result.initialValues[i] << "\n";
        }
        txtOut << "Parameters:\n";
        for (size_t i = 0; i < result.paramNames.size(); ++i) {
            txtOut << "  " << result.paramNames[i] << " = " << std::setprecision(6) << std::scientific << result.paramValues[i] << "\n";
        }
        txtOut << "\n";
    }

    txtOut.close();
    std::cout << "[Info] Saved: " << (std::string(outputDir) + "/tsallisbw_lowpt_parameters.txt") << std::endl;
    return 0;
}

int FitTsallisBWLowPtRangeScanMerged(
    const char *inputRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/FitTsallisBWLowPtRangeScan",
    double mass = 2.991) {
    std::filesystem::create_directories(outputDir);

    std::unique_ptr<TFile> fin(TFile::Open(inputRoot, "READ"));
    if (!fin || fin->IsZombie()) {
        std::cerr << "[Error] Cannot open input file: " << inputRoot << std::endl;
        return 1;
    }

    const auto cenDirs = CollectCentralityDirs(*fin);
    if (cenDirs.empty()) {
        std::cerr << "[Error] No cen_* directories found in: " << inputRoot << std::endl;
        return 1;
    }

    const std::vector<TsallisBWRangeScanCase> scanCases = {
        {"wide_current",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"q_tight",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.025, 1.080}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"q_very_tight",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.030, 1.065}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"beta_floor_045",
         {0.62, 0.11, 1.04, 1.0, 0.60},
         {{0.45, 0.82}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"beta_cap_075",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.75}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"beta_tight",
         {0.62, 0.11, 1.04, 1.0, 0.60},
         {{0.45, 0.75}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"T_tight",
         {0.60, 0.12, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.070, 0.18}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"ymax_tight",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.35, 0.85}}},
        {"ymax_floor_035",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.35, 1.20}}},
        {"q_floor_025",
         {0.60, 0.11, 1.04, 1.0, 0.60},
         {{0.30, 0.82}, {0.045, 0.22}, {1.025, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"beta_floor_q_floor",
         {0.62, 0.11, 1.04, 1.0, 0.60},
         {{0.45, 0.82}, {0.045, 0.22}, {1.025, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"beta_T_tight",
         {0.62, 0.12, 1.04, 1.0, 0.60},
         {{0.45, 0.75}, {0.070, 0.18}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}}},
        {"recommended_smooth",
         {0.62, 0.12, 1.04, 1.0, 0.60},
         {{0.45, 0.75}, {0.070, 0.18}, {1.025, 1.080}, {1e-10, 1e12}, {0.35, 0.85}}}
    };

    ConfigureFitDefaults();
    std::ofstream csv((std::string(outputDir) + "/tsallisbw_lowpt_range_scan.csv").c_str());
    csv << "case,centrality,fitStatus,chi2ndf,beta,T,q,norm,ymax,maxAbsLogCurvature,rmsLogCurvature,upturnStrength,slopeChangeCount\n";

    std::ofstream summary((std::string(outputDir) + "/tsallisbw_lowpt_range_scan_summary.txt").c_str());
    summary << "=== TsallisBW low-pT range scan ===\n";
    summary << "Input: " << inputRoot << "\n";
    summary << "Metrics are measured in 0.05-2.0 GeV/c from log(f(pT)). Smaller curvature/upturn is smoother.\n\n";

    struct CaseSummary {
        std::string name;
        double meanChi2Ndf = 0.0;
        double meanMaxCurv = 0.0;
        double meanRmsCurv = 0.0;
        double meanUpturn = 0.0;
        double meanSlopeChanges = 0.0;
        double meanQ = 0.0;
        int n = 0;
        int nFitNonzero = 0;
    };
    std::vector<CaseSummary> caseSummaries;

    for (const auto &scanCase : scanCases) {
        CaseSummary caseSummary;
        caseSummary.name = scanCase.name;
        summary << "Case: " << scanCase.name << "\n";
        summary << "  Limits: beta [" << scanCase.limits[0].first << ", " << scanCase.limits[0].second
                << "], T [" << scanCase.limits[1].first << ", " << scanCase.limits[1].second
                << "], q [" << scanCase.limits[2].first << ", " << scanCase.limits[2].second
                << "], ymax [" << scanCase.limits[4].first << ", " << scanCase.limits[4].second << "]\n";

        for (const auto &cen : cenDirs) {
            const std::string hPath = cen + "/std/h_corrected_counts";
            auto *hIn = dynamic_cast<TH1D *>(fin->Get(hPath.c_str()));
            if (!hIn) {
                std::cerr << "[Warn] Missing histogram: " << hPath << std::endl;
                continue;
            }

            auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(hIn->Clone(("h_scan_" + scanCase.name + "_" + cen).c_str())));
            h->SetDirectory(nullptr);
            const double fitMin = h->GetBinLowEdge(1);
            const double fitMax = h->GetBinLowEdge(h->GetNbinsX() + 1);

            FitResult result;
            result.funcName = "fTsallisBW";
            result.centrality = cen;
            result.mass = mass;
            result.fitStatus = -1;
            result.chi2 = 0.0;
            result.ndf = 0;

            const Int_t oldErrorIgnoreLevel = gErrorIgnoreLevel;
            gErrorIgnoreLevel = kFatal;
            auto f = FitBestSeed(h.get(), "fTsallisBW", scanCase.name + "_" + cen,
                                 scanCase.init, scanCase.limits, mass, fitMin, fitMax, result, true, false);
            gErrorIgnoreLevel = oldErrorIgnoreLevel;
            if (!f) continue;
            const auto metrics = MeasureTsallisBWLowPtShape(f.get(), 0.05, 2.0);
            const double chi2ndf = (result.ndf > 0) ? result.chi2 / static_cast<double>(result.ndf) : 0.0;
            const double beta = f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 0));
            const double temp = f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 1));
            const double q = f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 2));
            const double norm = f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 3));
            const double ymax = f->GetParameter(GetCanonicalParameterIndex("fTsallisBW", 4));

            csv << scanCase.name << "," << cen << "," << result.fitStatus << ","
                << chi2ndf << "," << beta << "," << temp << "," << q << ","
                << norm << "," << ymax << "," << metrics.maxAbsLogCurvature << ","
                << metrics.rmsLogCurvature << "," << metrics.upturnStrength << ","
                << metrics.slopeChangeCount << "\n";

            summary << "  " << cen
                    << " chi2/ndf=" << std::setprecision(4) << chi2ndf
                    << " q=" << q
                    << " beta=" << beta
                    << " T=" << temp
                    << " ymax=" << ymax
                    << " maxCurv=" << metrics.maxAbsLogCurvature
                    << " upturn=" << metrics.upturnStrength
                    << "\n";

            caseSummary.meanChi2Ndf += chi2ndf;
            caseSummary.meanMaxCurv += metrics.maxAbsLogCurvature;
            caseSummary.meanRmsCurv += metrics.rmsLogCurvature;
            caseSummary.meanUpturn += metrics.upturnStrength;
            caseSummary.meanSlopeChanges += metrics.slopeChangeCount;
            caseSummary.meanQ += q;
            caseSummary.n += 1;
            if (result.fitStatus != 0) caseSummary.nFitNonzero += 1;
        }
        if (caseSummary.n > 0) {
            const double invN = 1.0 / static_cast<double>(caseSummary.n);
            caseSummary.meanChi2Ndf *= invN;
            caseSummary.meanMaxCurv *= invN;
            caseSummary.meanRmsCurv *= invN;
            caseSummary.meanUpturn *= invN;
            caseSummary.meanSlopeChanges *= invN;
            caseSummary.meanQ *= invN;
        }
        caseSummaries.push_back(caseSummary);
        summary << "\n";
    }

    std::sort(caseSummaries.begin(), caseSummaries.end(), [](const CaseSummary &a, const CaseSummary &b) {
        const double scoreA = a.meanUpturn * 5.0 + a.meanMaxCurv + 0.02 * a.meanChi2Ndf;
        const double scoreB = b.meanUpturn * 5.0 + b.meanMaxCurv + 0.02 * b.meanChi2Ndf;
        return scoreA < scoreB;
    });

    summary << "=== Ranked summary ===\n";
    summary << "case mean_chi2ndf mean_q mean_maxCurv mean_rmsCurv mean_upturn mean_slopeChanges nonzeroFitStatus/n\n";
    for (const auto &s : caseSummaries) {
        summary << s.name << " "
                << std::setprecision(6) << s.meanChi2Ndf << " "
                << s.meanQ << " "
                << s.meanMaxCurv << " "
                << s.meanRmsCurv << " "
                << s.meanUpturn << " "
                << s.meanSlopeChanges << " "
                << s.nFitNonzero << "/" << s.n << "\n";
    }

    csv.close();
    summary.close();
    std::cout << "[Info] Saved: " << (std::string(outputDir) + "/tsallisbw_lowpt_range_scan.csv") << std::endl;
    std::cout << "[Info] Saved: " << (std::string(outputDir) + "/tsallisbw_lowpt_range_scan_summary.txt") << std::endl;
    return 0;
}

int DrawTsallisBWNoKinkComparison(
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/FitTsallisBWLowPtRangeScan",
    double mass = 2.991) {
    std::filesystem::create_directories(outputDir);

    struct ShapeCase {
        std::string label;
        double beta;
        double temp;
        double q;
        double ymax;
        Color_t color;
        Style_t lineStyle;
        Width_t lineWidth;
    };

    const std::vector<ShapeCase> cases = {
        {"with visible low-#it{p}_{T} turn", 0.34, 0.045, 1.019, 1.19, kGray + 2, 2, 3},
        {"smooth: #beta=0.60, T=0.12, q=1.040", 0.60, 0.120, 1.040, 0.60, kAzure + 2, 1, 4},
        {"smooth: lower #beta", 0.50, 0.120, 1.040, 0.60, kGreen + 2, 1, 3},
        {"smooth: higher T", 0.60, 0.160, 1.040, 0.60, kOrange + 7, 1, 3},
        {"smooth: slightly larger q", 0.60, 0.120, 1.060, 0.60, kRed + 1, 1, 3}
    };

    static std::vector<AliPWGFunc *> sAliPwgOwners;
    std::vector<TF1 *> funcs;
    funcs.reserve(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        auto *helper = new AliPWGFunc();
        sAliPwgOwners.push_back(helper);
        helper->SetVarType(AliPWGFunc::kdNdpt);
        auto *f = helper->GetTsallisBW(mass, cases[i].beta, cases[i].temp, cases[i].q, 1.0, cases[i].ymax,
                                       Form("fTsallisBW_shape_%zu", i));
        if (!f) continue;
        f->SetRange(0.0, 5.0);
        f->SetNpx(10000);
        f->SetLineColor(cases[i].color);
        f->SetLineStyle(cases[i].lineStyle);
        f->SetLineWidth(cases[i].lineWidth);

        const double ref = f->Eval(2.0);
        if (std::isfinite(ref) && ref > 0.0) {
            f->SetParameter(GetCanonicalParameterIndex("fTsallisBW", 3), 1.0 / ref);
        }
        funcs.push_back(f);
    }

    auto c = std::make_unique<TCanvas>("c_tsallisbw_no_kink_comparison", "TsallisBW no-kink comparison", 1200, 560);
    c->Divide(2, 1, 0.01, 0.0);

    auto drawPanel = [&](int padId, double xMax, bool logy) {
        c->cd(padId);
        gPad->SetLeftMargin(0.13);
        gPad->SetBottomMargin(0.13);
        gPad->SetRightMargin(0.04);
        if (logy) gPad->SetLogy();

        double yMin = logy ? 1e-3 : 0.0;
        double yMax = 0.0;
        for (auto *f : funcs) {
            for (int i = 0; i < 500; ++i) {
                const double x = xMax * static_cast<double>(i) / 499.0;
                const double y = f->Eval(x);
                if (!std::isfinite(y) || y <= 0.0) continue;
                if (logy) yMin = std::min(yMin, y);
                yMax = std::max(yMax, y);
            }
        }
        if (!(yMax > 0.0)) yMax = 2.0;
        if (logy) yMin = std::max(1e-4, yMin * 0.5);

        auto *frame = gPad->DrawFrame(0.0, yMin, xMax, logy ? yMax * 5.0 : yMax * 1.18,
                                      ";#it{p}_{T} (GeV/#it{c});TsallisBW shape, normalized at 2 GeV/#it{c}");
        frame->GetYaxis()->SetTitleOffset(1.25);
        for (auto *f : funcs) f->Draw("SAME");

        auto *line = new TLine(2.0, yMin, 2.0, logy ? yMax * 5.0 : yMax * 1.18);
        line->SetLineColor(kGray + 1);
        line->SetLineStyle(3);
        line->Draw("SAME");

        TLatex text;
        text.SetNDC();
        text.SetTextFont(42);
        text.SetTextSize(0.040);
        text.DrawLatex(0.18, 0.88, padId == 1 ? "low-#it{p}_{T} shape" : "full range");
    };

    drawPanel(1, 2.2, false);
    drawPanel(2, 5.0, true);

    c->cd(1);
    auto leg = std::make_unique<TLegend>(0.43, 0.17, 0.92, 0.43);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.030);
    for (size_t i = 0; i < funcs.size(); ++i) leg->AddEntry(funcs[i], cases[i].label.c_str(), "l");
    leg->Draw();

    const std::string outPdf = std::string(outputDir) + "/tsallisbw_no_kink_shape_comparison.pdf";
    c->SaveAs(outPdf.c_str());
    std::cout << "[Info] Saved: " << outPdf << std::endl;
    return 0;
}

int FitStableExtrapolationAlternativesMerged(
    const char *inputRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/StableExtrapolationAlternatives",
    double mass = 2.991) {
    std::filesystem::create_directories(outputDir);

    std::unique_ptr<TFile> fin(TFile::Open(inputRoot, "READ"));
    if (!fin || fin->IsZombie()) {
        std::cerr << "[Error] Cannot open input file: " << inputRoot << std::endl;
        return 1;
    }

    const auto cenDirs = CollectCentralityDirs(*fin);
    if (cenDirs.empty()) {
        std::cerr << "[Error] No cen_* directories found in: " << inputRoot << std::endl;
        return 1;
    }

    ConfigureFitDefaults();

    auto seedNorm = [](TH1D *h, TF1 *f, int normPar, double fitMin, double fitMax) {
        if (!h || !f || normPar < 0 || normPar >= f->GetNpar()) return;
        const double oldNorm = f->GetParameter(normPar);
        f->SetParameter(normPar, 1.0);
        double num = 0.0;
        double den = 0.0;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetBinCenter(ib);
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
        double norm = (den > 0.0) ? num / den : oldNorm;
        if (!std::isfinite(norm) || norm <= 0.0) norm = oldNorm;
        f->SetParameter(normPar, norm);
    };

    auto seedLogA0 = [](TH1D *h, TF1 *f, double massValue, bool useMt, double fitMin, double fitMax) {
        if (!h || !f) return;
        double sum = 0.0;
        int n = 0;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetBinCenter(ib);
            if (x < fitMin || x > fitMax || x <= 0.0) continue;
            const double y = h->GetBinContent(ib);
            if (!std::isfinite(y) || y <= 0.0) continue;
            const double mt = std::sqrt(x * x + massValue * massValue);
            const double variable = useMt ? (mt - massValue) : x;
            (void)variable;
            sum += std::log(y / x);
            ++n;
        }
        if (n > 0) f->SetParameter(0, sum / static_cast<double>(n));
    };

    auto fitCustomSeeds = [&](TH1D *h,
                              const std::string &caseName,
                              const std::vector<std::vector<double>> &seeds,
                              const std::vector<ParLimit> &limits,
                              double fitMin,
                              double fitMax) {
        std::unique_ptr<TF1> best;
        double bestScore = std::numeric_limits<double>::infinity();
        int bestStatus = -1;
        int iSeed = 0;

        for (auto seed : seeds) {
            auto build = [&]() -> std::unique_ptr<TF1> {
                TF1 *f = nullptr;
                if (caseName == "FlowExp") {
                    f = new TF1(Form("f_%s_seed%d", caseName.c_str(), iSeed),
                                [mass](double *x, double *p) {
                                    const double pt = x[0];
                                    const double mt = std::sqrt(pt * pt + mass * mass);
                                    const double beta = p[0];
                                    const double temp = p[1];
                                    const double norm = p[2];
                                    if (pt <= 0.0 || temp <= 0.0) return 0.0;
                                    return norm * pt * mt * std::exp(-(mt - beta * pt) / temp);
                                }, 0.0, 10.0, 3);
                    f->SetParNames("beta", "T", "norm");
                    f->SetParameters(seed[0], seed[1], seed[2]);
                    seedNorm(h, f, 2, fitMin, fitMax);
                } else if (caseName == "LogPolyPt") {
                    f = new TF1(Form("f_%s_seed%d", caseName.c_str(), iSeed),
                                [](double *x, double *p) {
                                    const double pt = x[0];
                                    if (pt <= 0.0) return 0.0;
                                    return pt * std::exp(p[0] + p[1] * pt + p[2] * pt * pt);
                                }, 0.0, 10.0, 3);
                    f->SetParNames("a0", "a1", "a2");
                    f->SetParameters(seed[0], seed[1], seed[2]);
                    seedLogA0(h, f, mass, false, fitMin, fitMax);
                } else if (caseName == "LogPolyMt") {
                    f = new TF1(Form("f_%s_seed%d", caseName.c_str(), iSeed),
                                [mass](double *x, double *p) {
                                    const double pt = x[0];
                                    if (pt <= 0.0) return 0.0;
                                    const double mtMinusM = std::sqrt(pt * pt + mass * mass) - mass;
                                    return pt * std::exp(p[0] + p[1] * mtMinusM + p[2] * mtMinusM * mtMinusM);
                                }, 0.0, 10.0, 3);
                    f->SetParNames("a0", "a1", "a2");
                    f->SetParameters(seed[0], seed[1], seed[2]);
                    seedLogA0(h, f, mass, true, fitMin, fitMax);
                }
                if (!f) return nullptr;
                for (int ip = 0; ip < f->GetNpar() && ip < static_cast<int>(limits.size()); ++ip) {
                    if (limits[ip].second > limits[ip].first) f->SetParLimits(ip, limits[ip].first, limits[ip].second);
                }
                f->SetNpx(10000);
                f->SetLineWidth(3);
                return std::unique_ptr<TF1>(f);
            };

            auto f = build();
            ++iSeed;
            if (!f) continue;
            const int status = static_cast<int>(h->Fit(f.get(), "Q0RSN", "", fitMin, fitMax));
            const double chi2 = f->GetChisquare();
            const int ndf = f->GetNDF();
            const double chi2ndf = (ndf > 0 && std::isfinite(chi2)) ? chi2 / static_cast<double>(ndf) : 1e30;
            const double shapePenalty = LowPtSmoothnessPenalty(f.get(), 0.05, std::min(2.0, fitMax));
            const double score = chi2ndf * shapePenalty * (status == 0 ? 1.0 : 1.05);
            if (score < bestScore) {
                bestScore = score;
                bestStatus = status;
                best = std::move(f);
            }
        }
        return std::make_pair(std::move(best), bestStatus);
    };

    struct AlternativeResult {
        std::string centrality;
        std::string name;
        int status = -1;
        double chi2ndf = 0.0;
        double integral = 0.0;
        double extrapLow = 0.0;
        TsallisBWShapeMetrics shape;
        std::vector<double> params;
    };

    std::vector<AlternativeResult> allResults;
    std::ofstream csv((std::string(outputDir) + "/stable_extrapolation_alternatives.csv").c_str());
    csv << "centrality,function,status,chi2ndf,integral_0_fitmax,integral_0_firstbinlow,maxAbsLogCurvature,rmsLogCurvature,upturnStrength,slopeChangeCount,params\n";

    const std::vector<std::string> caseNames = {"ConstrainedTsallisBW", "ConstrainedLevi", "FlowExp", "LogPolyPt", "LogPolyMt"};
    const std::map<std::string, Color_t> colors = {
        {"ConstrainedTsallisBW", kViolet + 1},
        {"ConstrainedLevi", kAzure + 2},
        {"FlowExp", kGreen + 2},
        {"LogPolyPt", kOrange + 7},
        {"LogPolyMt", kRed + 1}
    };

    for (const auto &cen : cenDirs) {
        const std::string hPath = cen + "/std/h_corrected_counts";
        auto *hIn = dynamic_cast<TH1D *>(fin->Get(hPath.c_str()));
        if (!hIn) {
            std::cerr << "[Warn] Missing histogram: " << hPath << std::endl;
            continue;
        }

        auto h = std::unique_ptr<TH1D>(static_cast<TH1D *>(hIn->Clone(("h_stable_alt_" + cen).c_str())));
        h->SetDirectory(nullptr);
        h->SetStats(false);
        h->SetMarkerStyle(kFullCircle);
        h->SetMarkerSize(0.95);
        h->SetMarkerColor(kBlack);
        h->SetLineColor(kBlack);
        h->SetTitle("");

        const double fitMin = h->GetBinLowEdge(1);
        const double fitMax = h->GetBinLowEdge(h->GetNbinsX() + 1);
        std::vector<std::pair<std::string, std::unique_ptr<TF1>>> funcsToDraw;

        for (const auto &caseName : caseNames) {
            std::unique_ptr<TF1> f;
            int status = -1;
            if (caseName == "ConstrainedTsallisBW") {
                FitResult result;
                f = FitBestSeed(h.get(), "fTsallisBW", cen + "_stable_alt",
                                {0.60, 0.11, 1.04, 1.0, 0.60},
                                {{0.30, 0.75}, {0.045, 0.22}, {1.018, 1.120}, {1e-10, 1e12}, {0.20, 1.20}},
                                mass, fitMin, fitMax, result, true, true);
                status = result.fitStatus;
            } else if (caseName == "ConstrainedLevi") {
                FitResult result;
                f = FitBestSeed(h.get(), "fLevi", cen + "_stable_alt",
                                {0.22, 8.0, 1.0},
                                {{0.06, 0.70}, {2.0, 80.0}, {1e-12, 1e12}},
                                mass, fitMin, fitMax, result, true, true);
                status = result.fitStatus;
            } else if (caseName == "FlowExp") {
                auto fitted = fitCustomSeeds(h.get(), caseName,
                                             {{0.40, 0.10, 1.0}, {0.55, 0.13, 1.0}, {0.70, 0.16, 1.0}},
                                             {{0.0, 0.85}, {0.04, 0.50}, {1e-14, 1e14}},
                                             fitMin, fitMax);
                f = std::move(fitted.first);
                status = fitted.second;
            } else if (caseName == "LogPolyPt") {
                auto fitted = fitCustomSeeds(h.get(), caseName,
                                             {{0.0, -1.0, -0.05}, {0.0, -0.5, -0.10}, {0.0, -1.5, 0.0}},
                                             {{-100.0, 100.0}, {-20.0, 5.0}, {-10.0, 1.0}},
                                             fitMin, fitMax);
                f = std::move(fitted.first);
                status = fitted.second;
            } else if (caseName == "LogPolyMt") {
                auto fitted = fitCustomSeeds(h.get(), caseName,
                                             {{0.0, -3.0, -0.10}, {0.0, -2.0, -0.30}, {0.0, -5.0, 0.0}},
                                             {{-100.0, 100.0}, {-50.0, 5.0}, {-50.0, 5.0}},
                                             fitMin, fitMax);
                f = std::move(fitted.first);
                status = fitted.second;
            }

            if (!f) continue;
            f->SetLineColor(colors.at(caseName));
            f->SetLineWidth(3);
            f->SetLineStyle(caseName == "ConstrainedLevi" ? 7 : 1);

            AlternativeResult res;
            res.centrality = cen;
            res.name = caseName;
            res.status = status;
            res.chi2ndf = (f->GetNDF() > 0) ? f->GetChisquare() / static_cast<double>(f->GetNDF()) : 0.0;
            res.integral = f->Integral(0.0, fitMax);
            res.extrapLow = f->Integral(0.0, fitMin);
            res.shape = MeasureTsallisBWLowPtShape(f.get(), 0.05, std::min(2.0, fitMax));
            for (int ip = 0; ip < f->GetNpar(); ++ip) res.params.push_back(f->GetParameter(ip));
            allResults.push_back(res);

            csv << cen << "," << caseName << "," << status << ","
                << res.chi2ndf << "," << res.integral << "," << res.extrapLow << ","
                << res.shape.maxAbsLogCurvature << "," << res.shape.rmsLogCurvature << ","
                << res.shape.upturnStrength << "," << res.shape.slopeChangeCount << ",\"";
            for (size_t ip = 0; ip < res.params.size(); ++ip) {
                if (ip) csv << ";";
                csv << res.params[ip];
            }
            csv << "\"\n";

            funcsToDraw.emplace_back(caseName, std::move(f));
        }

        auto c = std::make_unique<TCanvas>(("c_stable_alt_" + cen).c_str(), ("Stable alternatives " + cen).c_str(), 1200, 650);
        c->Divide(2, 1, 0.01, 0.0);

        c->cd(1);
        gPad->SetLogy();
        gPad->SetLeftMargin(0.14);
        gPad->SetBottomMargin(0.12);
        h->Draw("E1");
        h->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
        h->GetYaxis()->SetTitle("corrected counts");
        h->GetYaxis()->SetTitleOffset(1.35);
        for (auto &item : funcsToDraw) item.second->Draw("SAME");
        auto leg = std::make_unique<TLegend>(0.44, 0.58, 0.88, 0.88);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextFont(42);
        leg->SetTextSize(0.030);
        leg->AddEntry(h.get(), "data", "pe");
        for (auto &item : funcsToDraw) {
            leg->AddEntry(item.second.get(), item.first.c_str(), "l");
        }
        leg->Draw();
        TLatex text;
        text.SetNDC();
        text.SetTextFont(42);
        text.SetTextSize(0.040);
        text.DrawLatex(0.18, 0.88, cen.c_str());

        c->cd(2);
        gPad->SetLeftMargin(0.14);
        gPad->SetBottomMargin(0.12);
        double yMinLow = 1e30;
        double yMaxLow = 0.0;
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetBinCenter(ib);
            if (x > 2.2) continue;
            const double y = h->GetBinContent(ib);
            const double e = h->GetBinError(ib);
            if (y > 0.0) {
                yMinLow = std::min(yMinLow, std::max(0.0, y - e));
                yMaxLow = std::max(yMaxLow, y + e);
            }
        }
        for (auto &item : funcsToDraw) {
            for (int i = 0; i < 300; ++i) {
                const double x = 2.2 * static_cast<double>(i) / 299.0;
                const double y = item.second->Eval(x);
                if (std::isfinite(y) && y > 0.0) {
                    yMinLow = std::min(yMinLow, y);
                    yMaxLow = std::max(yMaxLow, y);
                }
            }
        }
        if (!(yMaxLow > 0.0)) yMaxLow = 1.0;
        if (!(yMinLow < yMaxLow)) yMinLow = 0.0;
        auto *frame = gPad->DrawFrame(0.0, std::max(0.0, yMinLow * 0.85), 2.2, yMaxLow * 1.20,
                                      ";#it{p}_{T} (GeV/#it{c});corrected counts");
        frame->GetYaxis()->SetTitleOffset(1.35);
        h->Draw("E1 SAME");
        for (auto &item : funcsToDraw) item.second->Draw("SAME");
        auto *line = new TLine(fitMin, std::max(0.0, yMinLow * 0.85), fitMin, yMaxLow * 1.20);
        line->SetLineColor(kGray + 2);
        line->SetLineStyle(3);
        line->Draw("SAME");
        text.SetTextSize(0.036);
        text.DrawLatex(0.18, 0.88, "low-#it{p}_{T} extrapolation");

        const std::string outPdf = std::string(outputDir) + "/stable_extrapolation_alternatives_" + cen + ".pdf";
        c->SaveAs(outPdf.c_str());
        std::cout << "[Info] Saved: " << outPdf << std::endl;
    }

    csv.close();

    std::ofstream summary((std::string(outputDir) + "/stable_extrapolation_alternatives_summary.txt").c_str());
    summary << "=== Stable extrapolation alternatives ===\n";
    summary << "Input: " << inputRoot << "\n";
    summary << "Functions: ConstrainedTsallisBW, ConstrainedLevi, FlowExp, LogPolyPt, LogPolyMt\n";
    summary << "FlowExp = norm * pT * mT * exp(-(mT - beta*pT)/T)\n";
    summary << "LogPolyPt = pT * exp(a0 + a1*pT + a2*pT^2)\n";
    summary << "LogPolyMt = pT * exp(a0 + a1*(mT-m) + a2*(mT-m)^2)\n\n";
    summary << "centrality function status chi2ndf integral_0_fitmax integral_0_firstbinlow upturn maxCurv params\n";
    for (const auto &res : allResults) {
        summary << res.centrality << " " << res.name << " " << res.status << " "
                << std::setprecision(6) << res.chi2ndf << " "
                << res.integral << " " << res.extrapLow << " "
                << res.shape.upturnStrength << " " << res.shape.maxAbsLogCurvature << " ";
        for (size_t ip = 0; ip < res.params.size(); ++ip) {
            if (ip) summary << ";";
            summary << res.params[ip];
        }
        summary << "\n";
    }
    summary.close();

    std::cout << "[Info] Saved: " << (std::string(outputDir) + "/stable_extrapolation_alternatives.csv") << std::endl;
    std::cout << "[Info] Saved: " << (std::string(outputDir) + "/stable_extrapolation_alternatives_summary.txt") << std::endl;
    return 0;
}
