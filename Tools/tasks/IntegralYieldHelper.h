#ifndef INTEGRAL_YIELD_HELPER_H
#define INTEGRAL_YIELD_HELPER_H

#include <TCanvas.h>
#include <TDecompChol.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TGraph.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TBox.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TRandom3.h>
#include <TVectorD.h>

#include <ROOT/RDataFrame.hxx>

#include "../../include/AliPWGFunc.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace UnifiedAnalysis {

using SpectrumParLimit = std::pair<double, double>;

struct SpectrumFunctionConfig {
    std::vector<double> initial;
    std::vector<SpectrumParLimit> limits;
};

struct IntegralYieldConfig {
    std::string nominalFitFunc{"fBGBW"};
    std::vector<std::string> fitFuncCandidates;
    std::map<std::string, SpectrumFunctionConfig> fitFuncParameters;
    bool doSystematics{true};
    int nTrailsForIntegralSyst{1000};
    int nCombinationsForExtrapolation{500};
    double branchingRatioFractionalUncertainty{0.08};
    double integrateMin{0.0};
    double integrateMax{10.0};
    bool rejectFitFuncByChi2{false};
    double fitFuncMaxChi2Ndf{5.0};
    double fitFuncFallbackFraction{0.20};
    double gaussFitMaxChi2Ndf{3.0};
    double extrapToyMaxChi2Ndf{-1.0};
    double fitRangeMin{0.0};
    double fitRangeMax{10.0};
    double lowPtMaxFactor{10.0};
    bool useMinosErrors{false};
    double absorptionLength{0.5};
    bool usePerformanceLabel{false};
    std::string performanceLabel;
    std::string collisionSystem;
    std::string collisionEnergy;
    std::string period;
    std::string periodMark;
    std::string isMatter;
};

struct IntegralYieldInput {
    std::string groupTag;
    double cenMin{0.0};
    double cenMax{0.0};
    TH1D *hCorrected{nullptr};
    TH1D *hCorrectedSyst{nullptr};
    std::vector<TH1D*> absorptionVariants;
    std::vector<std::string> absorptionVariantLabels;
    std::vector<std::vector<double>> trailCorrectedValuesByBin;
    std::vector<std::vector<double>> trailCorrectedValuesByTrialBin;
    std::vector<double> measuredPtEdges;
};

struct IntegralFitParameterRow {
    std::string functionName;
    std::string parameterName;
    int parameterIndex{-1};
    double value{0.0};
    double error{0.0};
    double limitMin{0.0};
    double limitMax{0.0};
    bool hasLimits{false};
    double chi2{0.0};
    int ndf{0};
    double chi2Ndf{0.0};
    bool isNominal{false};
    bool rejectedByChi2{false};
    bool rejectedByLowPt{false};
};

struct IntegralYieldResult {
    bool ok{false};
    double cenMin{0.0};
    double cenMax{0.0};
    double value{0.0};
    double statErr{0.0};

    double systExtrapolation{0.0};
    double systFitFunction{0.0};
    double systAbsorption{0.0};
    double systTrails{0.0};
    double systBranchingRatio{0.0};
    double systTotal{0.0};

    std::unique_ptr<TF1> fNominal;
    std::vector<std::unique_ptr<TF1>> fFitCandidates;
    std::vector<IntegralFitParameterRow> fitParameterRows;

    std::unique_ptr<TH1D> hExtrapRatioDist;
    std::unique_ptr<TH1D> hIntegralTrailDist;
    std::unique_ptr<TH1D> hAbsorptionYieldScan;
    std::unique_ptr<TH1D> hSystSources;
    std::unique_ptr<TH1D> hSystSourceFractions;
    std::unique_ptr<TH1D> hIntegralYieldOneBin;

    std::unique_ptr<TCanvas> cNominalAndFunctions;
    std::unique_ptr<TCanvas> cFitFunctionParameters;
    std::unique_ptr<TCanvas> cExtrapolation;
    std::unique_ptr<TCanvas> cAbsorption;
    std::unique_ptr<TCanvas> cAbsorptionYieldScan;
    std::unique_ptr<TCanvas> cTrails;
    std::unique_ptr<TCanvas> cSources;
    std::unique_ptr<TCanvas> cIntegralYieldOneBin;
};

inline double IntegrateFunction(TF1 *f, double xMin, double xMax);

inline double ComputeHistogramIntegral(TH1D *h) {
    if (!h) return 0.0;
    double sum = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        sum += h->GetBinContent(ib) * h->GetXaxis()->GetBinWidth(ib);
    }
    return sum;
}

inline bool GetMeasuredRange(TH1D *h,
                             const std::vector<double> *measuredEdges,
                             double &measuredMin,
                             double &measuredMax) {
    if (measuredEdges && measuredEdges->size() >= 2) {
        measuredMin = measuredEdges->front();
        measuredMax = measuredEdges->back();
        return std::isfinite(measuredMin) && std::isfinite(measuredMax) && measuredMax > measuredMin;
    }
    if (!h || h->GetNbinsX() <= 0) return false;
    measuredMin = h->GetXaxis()->GetBinLowEdge(1);
    measuredMax = h->GetXaxis()->GetBinUpEdge(h->GetNbinsX());
    return std::isfinite(measuredMin) && std::isfinite(measuredMax) && measuredMax > measuredMin;
}

inline std::vector<std::pair<double, double>> BuildExtrapolationRanges(TH1D *h,
                                                                       const IntegralYieldConfig &cfg,
                                                                       const std::vector<double> *measuredEdges = nullptr) {
    std::vector<std::pair<double, double>> ranges;
    double measuredMin = 0.0;
    double measuredMax = 0.0;
    if (!GetMeasuredRange(h, measuredEdges, measuredMin, measuredMax)) return ranges;
    if (measuredMin > cfg.integrateMin) {
        ranges.emplace_back(cfg.integrateMin, measuredMin);
    }
    if (cfg.integrateMax > measuredMax) {
        ranges.emplace_back(measuredMax, cfg.integrateMax);
    }
    return ranges;
}

inline double ComputeHybridIntegral(TH1D *h,
                                    TF1 *f,
                                    const IntegralYieldConfig &cfg,
                                    const std::vector<double> *measuredEdges = nullptr) {
    double sum = ComputeHistogramIntegral(h);
    if (!f) return sum;
    for (const auto &range : BuildExtrapolationRanges(h, cfg, measuredEdges)) {
        if (range.second > range.first) {
            sum += IntegrateFunction(f, range.first, range.second);
        }
    }
    return sum;
}

inline std::string BuildIntegralDecayString(const std::string &isMatter) {
    if (isMatter == "matter") {
        return "{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}";
    }
    if (isMatter == "antimatter") {
        return "{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}";
    }
    if (isMatter == "both") {
        return "{}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}He + #pi^{-} (^{3}#bar{He} + #pi^{+})";
    }
    return std::string();
}

inline double ComputeHistogramIntegralError(TH1D *h) {
    if (!h) return 0.0;
    double err2 = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double e = h->GetBinError(ib) * h->GetXaxis()->GetBinWidth(ib);
        err2 += e * e;
    }
    return std::sqrt(std::max(0.0, err2));
}

inline double ComputeChi2Ndf(TF1 *f) {
    if (!f || f->GetNDF() <= 0) return std::numeric_limits<double>::infinity();
    return f->GetChisquare() / static_cast<double>(f->GetNDF());
}

inline std::pair<double, double> ComputeRobustDisplayRange(const std::vector<double> &values,
                                                           double fallbackCenter,
                                                           double fallbackWidth,
                                                           double lowQuantile = 0.01,
                                                           double highQuantile = 0.99) {
    std::vector<double> finite;
    finite.reserve(values.size());
    for (double v : values) {
        if (std::isfinite(v)) finite.push_back(v);
    }
    if (finite.empty()) {
        const double width = std::max(std::abs(fallbackWidth), 1e-12);
        return {fallbackCenter - width, fallbackCenter + width};
    }
    std::sort(finite.begin(), finite.end());
    auto pickQuantile = [&](double q) {
        q = std::clamp(q, 0.0, 1.0);
        const double pos = q * static_cast<double>(finite.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(finite.size() - 1, lo + 1);
        const double frac = pos - static_cast<double>(lo);
        return finite[lo] * (1.0 - frac) + finite[hi] * frac;
    };
    double rMin = pickQuantile(lowQuantile);
    double rMax = pickQuantile(highQuantile);
    if (!(rMax > rMin)) {
        rMin = finite.front();
        rMax = finite.back();
    }
    if (!(rMax > rMin)) {
        const double width = std::max({std::abs(fallbackWidth), 0.02 * std::abs(fallbackCenter), 1e-12});
        rMin = fallbackCenter - width;
        rMax = fallbackCenter + width;
    } else {
        const double pad = 0.10 * (rMax - rMin);
        rMin -= pad;
        rMax += pad;
    }
    return {rMin, rMax};
}

inline std::pair<double, double> ComputeMeanRms(const std::vector<double> &values) {
    double sum = 0.0;
    int n = 0;
    for (double v : values) {
        if (!std::isfinite(v)) continue;
        sum += v;
        ++n;
    }
    if (n <= 0) return {0.0, 0.0};
    const double mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double v : values) {
        if (!std::isfinite(v)) continue;
        const double d = v - mean;
        var += d * d;
    }
    return {mean, std::sqrt(std::max(0.0, var / static_cast<double>(n)))};
}

inline bool PassLowPtExtrapolationGuard(TH1D *h,
                                        TF1 *f,
                                        const IntegralYieldConfig &cfg,
                                        double *maxLowPtOut = nullptr,
                                        double *refOut = nullptr) {
    if (maxLowPtOut) *maxLowPtOut = std::numeric_limits<double>::quiet_NaN();
    if (refOut) *refOut = std::numeric_limits<double>::quiet_NaN();
    if (!h || !f || !(cfg.lowPtMaxFactor > 0.0)) return true;
    if (h->GetNbinsX() <= 0) return true;

    const double measuredMin = h->GetXaxis()->GetBinLowEdge(1);
    if (!(measuredMin > cfg.integrateMin)) return true;

    double ref = std::numeric_limits<double>::quiet_NaN();
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double y = h->GetBinContent(ib);
        const double e = h->GetBinError(ib);
        if (std::isfinite(y) && y > 0.0) {
            ref = y + (std::isfinite(e) && e > 0.0 ? 3.0 * e : 0.0);
            break;
        }
    }
    if (!(std::isfinite(ref) && ref > 0.0)) return true;

    double maxLowPt = 0.0;
    constexpr int nProbe = 80;
    for (int i = 0; i <= nProbe; ++i) {
        const double x = cfg.integrateMin + (measuredMin - cfg.integrateMin) *
                                             static_cast<double>(i) / static_cast<double>(nProbe);
        const double y = f->Eval(x);
        if (!std::isfinite(y) || y < 0.0) {
            if (maxLowPtOut) *maxLowPtOut = y;
            if (refOut) *refOut = ref;
            return false;
        }
        maxLowPt = std::max(maxLowPt, y);
    }
    if (maxLowPtOut) *maxLowPtOut = maxLowPt;
    if (refOut) *refOut = ref;
    return maxLowPt <= cfg.lowPtMaxFactor * ref;
}

inline double ComputeSpectrumChi2Ndf(TH1D *h, TF1 *f, double fitMin, double fitMax) {
    if (!h || !f) return std::numeric_limits<double>::infinity();
    double chi2 = 0.0;
    int nPoints = 0;
    int nFree = 0;
    for (int ip = 0; ip < f->GetNpar(); ++ip) {
        double lo = 0.0;
        double hi = 0.0;
        f->GetParLimits(ip, lo, hi);
        const bool fixed = (lo == hi && lo != 0.0);
        if (!fixed) ++nFree;
    }
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double x = h->GetXaxis()->GetBinCenter(ib);
        if (x < fitMin || x > fitMax) continue;
        const double y = h->GetBinContent(ib);
        const double e = h->GetBinError(ib);
        if (!std::isfinite(y) || !std::isfinite(e) || e <= 0.0) continue;
        const double yf = f->Eval(x);
        if (!std::isfinite(yf)) return std::numeric_limits<double>::infinity();
        const double pull = (y - yf) / e;
        chi2 += pull * pull;
        ++nPoints;
    }
    const int ndf = nPoints - nFree;
    if (ndf <= 0) return std::isfinite(chi2) ? chi2 : std::numeric_limits<double>::infinity();
    return chi2 / static_cast<double>(ndf);
}

inline bool AcceptGaussianFit(const TFitResultPtr &fitRes, TF1 *f, double maxChi2Ndf) {
    const int fitStatus = static_cast<int>(fitRes);
    const double chi2Ndf = ComputeChi2Ndf(f);
    const double sigma = f ? std::abs(f->GetParameter(2)) : 0.0;
    return fitStatus == 0 &&
           std::isfinite(chi2Ndf) &&
           chi2Ndf <= maxChi2Ndf &&
           std::isfinite(sigma) &&
           sigma > 0.0;
}

inline double IntegrateFunction(TF1 *f, double xMin, double xMax);

inline double IntegrateFunctionRanges(TF1 *f,
                                      const std::vector<std::pair<double, double>> &ranges) {
    if (!f) return 0.0;
    double sum = 0.0;
    for (const auto &range : ranges) {
        if (range.second > range.first) sum += IntegrateFunction(f, range.first, range.second);
    }
    return sum;
}

inline double ComputeFitIntegralErrorFromCovariance(TF1 *f,
                                                    const TFitResultPtr &fitResult,
                                                    const std::vector<std::pair<double, double>> &ranges) {
    if (!f || !fitResult.Get()) return 0.0;
    if (static_cast<int>(fitResult) != 0) return 0.0;
    if (ranges.empty()) return 0.0;
    const int npar = f->GetNpar();
    if (npar <= 0) return 0.0;

    std::vector<double> original(static_cast<size_t>(npar), 0.0);
    std::vector<double> gradient(static_cast<size_t>(npar), 0.0);
    for (int ip = 0; ip < npar; ++ip) original[static_cast<size_t>(ip)] = f->GetParameter(ip);

    for (int ip = 0; ip < npar; ++ip) {
        const double err = f->GetParError(ip);
        const double step = std::max(std::abs(err), std::max(1e-6 * std::abs(original[static_cast<size_t>(ip)]), 1e-12));
        if (!std::isfinite(step) || step <= 0.0) continue;
        double lo = 0.0;
        double hi = 0.0;
        f->GetParLimits(ip, lo, hi);
        const bool hasLimits = hi > lo;
        const double center = original[static_cast<size_t>(ip)];
        double plusValue = center + step;
        double minusValue = center - step;
        if (hasLimits) {
            plusValue = std::min(hi, plusValue);
            minusValue = std::max(lo, minusValue);
        }

        if (!(plusValue > center) && !(minusValue < center)) continue;

        f->SetParameter(ip, plusValue);
        const double plus = IntegrateFunctionRanges(f, ranges);
        f->SetParameter(ip, minusValue);
        const double minus = IntegrateFunctionRanges(f, ranges);
        f->SetParameter(ip, center);
        if (std::isfinite(plus) && std::isfinite(minus) && plusValue > minusValue) {
            gradient[static_cast<size_t>(ip)] = (plus - minus) / (plusValue - minusValue);
        }
    }

    double err2 = 0.0;
    for (int ip = 0; ip < npar; ++ip) {
        for (int jp = 0; jp < npar; ++jp) {
            const double cov = fitResult->CovMatrix(ip, jp);
            if (!std::isfinite(cov)) continue;
            err2 += gradient[static_cast<size_t>(ip)] * cov * gradient[static_cast<size_t>(jp)];
        }
    }
    for (int ip = 0; ip < npar; ++ip) f->SetParameter(ip, original[static_cast<size_t>(ip)]);
    return std::sqrt(std::max(0.0, err2));
}

inline SpectrumFunctionConfig DefaultSpectrumFunctionConfig(const std::string &funcName) {
    if (funcName == "fBGBW") {
        return SpectrumFunctionConfig{{0.6, 0.2, 1.0, 500.0},
                                      {{0.2, 0.9}, {0.1, 0.6}, {0.01, 5.0}, {1e-10, 1e9}}};
    }
    if (funcName == "fLevi") {
        return SpectrumFunctionConfig{{0.45, 20.0, 1.0},
                                      {{0.03, 2.0}, {2.0, 1e3}, {1e-9, 0.1}}};
    }
    if (funcName == "fBoltzmann") {
        return SpectrumFunctionConfig{{0.30, 1.0},
                                      {{0.03, 2.0}, {1e-9, 0.1}}};
    }
    if (funcName == "fPtExp") {
        return SpectrumFunctionConfig{{1.2, 1.0},
                                      {{0.03, 2.0}, {1e-9, 0.1}}};
    }
    if (funcName == "fTsallisBW") {
        return SpectrumFunctionConfig{{0.76, 0.16, 1.003, 1.0, 0.03},
                                      {{0.0, 0.95}, {0.005, 0.6}, {1.000001, 1.3}, {1e-10, 1e12}, {0.001, 2.0}}};
    }
    return SpectrumFunctionConfig{};
}

inline SpectrumFunctionConfig ResolveSpectrumFunctionConfig(
    const std::string &funcName,
    const std::map<std::string, SpectrumFunctionConfig> &configured) {
    SpectrumFunctionConfig out = DefaultSpectrumFunctionConfig(funcName);
    auto it = configured.find(funcName);
    if (it == configured.end()) return out;
    if (!it->second.initial.empty()) out.initial = it->second.initial;
    if (!it->second.limits.empty()) out.limits = it->second.limits;
    return out;
}

inline int GetCanonicalParameterIndex(const std::string &funcName, int iCanonical) {
    if (funcName == "fBGBW") {
        // canonical: beta, T, n, norm; AliPWGFunc: mass, beta, T, n, norm
        return iCanonical + 1;
    }
    if (funcName == "fLevi") {
        // canonical: T, n, norm; AliPWGFunc: norm, n, T, mass
        static const int map[] = {2, 1, 0};
        return (iCanonical >= 0 && iCanonical < 3) ? map[iCanonical] : -1;
    }
    if (funcName == "fBoltzmann" || funcName == "fPtExp") {
        // canonical: T, norm; AliPWGFunc: norm, T
        static const int map[] = {1, 0};
        return (iCanonical >= 0 && iCanonical < 2) ? map[iCanonical] : -1;
    }
    if (funcName == "fTsallisBW") {
        // canonical: beta, T, q, norm, ymax; AliPWGFunc: mass, beta, T, q, norm, ymax
        return (iCanonical >= 0 && iCanonical < 5) ? iCanonical + 1 : -1;
    }
    return iCanonical;
}

inline std::string GetCanonicalParameterName(const std::string &funcName, int iCanonical) {
    if (funcName == "fBGBW") {
        static const char *names[] = {"#beta", "T", "n", "Norm"};
        return (iCanonical >= 0 && iCanonical < 4) ? names[iCanonical] : Form("p%d", iCanonical);
    }
    if (funcName == "fLevi") {
        static const char *names[] = {"T", "n", "Norm"};
        return (iCanonical >= 0 && iCanonical < 3) ? names[iCanonical] : Form("p%d", iCanonical);
    }
    if (funcName == "fBoltzmann" || funcName == "fPtExp") {
        static const char *names[] = {"T", "Norm"};
        return (iCanonical >= 0 && iCanonical < 2) ? names[iCanonical] : Form("p%d", iCanonical);
    }
    if (funcName == "fTsallisBW") {
        static const char *names[] = {"#beta", "T", "q", "Norm", "ymax"};
        return (iCanonical >= 0 && iCanonical < 5) ? names[iCanonical] : Form("p%d", iCanonical);
    }
    return Form("p%d", iCanonical);
}

inline std::vector<std::string> BuildSpectrumParameterTextLines(
    const std::string &funcName,
    TF1 *f,
    const std::map<std::string, SpectrumFunctionConfig> &configuredPars) {
    std::vector<std::string> lines;
    if (!f) return lines;
    const SpectrumFunctionConfig parCfg = ResolveSpectrumFunctionConfig(funcName, configuredPars);
    const size_t nCanon = std::max(parCfg.initial.size(), parCfg.limits.size());
    lines.push_back(funcName);
    for (size_t i = 0; i < nCanon; ++i) {
        const int ip = GetCanonicalParameterIndex(funcName, static_cast<int>(i));
        if (ip < 0 || ip >= f->GetNpar()) continue;
        const double value = f->GetParameter(ip);
        std::string range = "free";
        if (i < parCfg.limits.size() && parCfg.limits[i].second > parCfg.limits[i].first) {
            range = Form("[%.3g, %.3g]", parCfg.limits[i].first, parCfg.limits[i].second);
        }
        lines.push_back(Form("  %s = %.4g, range %s",
                             GetCanonicalParameterName(funcName, static_cast<int>(i)).c_str(),
                             value,
                             range.c_str()));
    }
    return lines;
}

inline double ParameterFallbackSigma(TF1 *f, int ip);

inline std::vector<IntegralFitParameterRow> BuildIntegralFitParameterRows(
    const std::string &funcName,
    TF1 *f,
    const std::map<std::string, SpectrumFunctionConfig> &configuredPars,
    const TFitResultPtr *fitResult,
    bool isNominal,
    bool rejectedByChi2,
    bool rejectedByLowPt) {
    std::vector<IntegralFitParameterRow> rows;
    if (!f) return rows;

    const SpectrumFunctionConfig parCfg = ResolveSpectrumFunctionConfig(funcName, configuredPars);
    const size_t nCanon = std::max(parCfg.initial.size(), parCfg.limits.size());
    const double chi2 = f->GetChisquare();
    const int ndf = f->GetNDF();
    const double chi2Ndf = ndf > 0 ? chi2 / static_cast<double>(ndf) : std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < nCanon; ++i) {
        const int ip = GetCanonicalParameterIndex(funcName, static_cast<int>(i));
        if (ip < 0 || ip >= f->GetNpar()) continue;
        IntegralFitParameterRow row;
        row.functionName = funcName;
        row.parameterName = GetCanonicalParameterName(funcName, static_cast<int>(i));
        row.parameterIndex = ip;
        row.value = f->GetParameter(ip);
        row.error = f->GetParError(ip);
        if (!(std::isfinite(row.error) && row.error > 0.0) && fitResult && fitResult->Get()) {
            const double fitErr = (*fitResult)->ParError(ip);
            if (std::isfinite(fitErr) && fitErr > 0.0) row.error = fitErr;
            if (!(std::isfinite(row.error) && row.error > 0.0)) {
                const TMatrixDSym cov = (*fitResult)->GetCovarianceMatrix();
                if (ip < cov.GetNrows() && cov(ip, ip) > 0.0) {
                    const double covErr = std::sqrt(cov(ip, ip));
                    if (std::isfinite(covErr) && covErr > 0.0) row.error = covErr;
                }
            }
        }
        if (!(std::isfinite(row.error) && row.error > 0.0)) {
            row.error = ParameterFallbackSigma(f, ip);
        }
        if (i < parCfg.limits.size() && parCfg.limits[i].second > parCfg.limits[i].first) {
            row.hasLimits = true;
            row.limitMin = parCfg.limits[i].first;
            row.limitMax = parCfg.limits[i].second;
        }
        row.chi2 = chi2;
        row.ndf = ndf;
        row.chi2Ndf = chi2Ndf;
        row.isNominal = isNominal;
        row.rejectedByChi2 = rejectedByChi2;
        row.rejectedByLowPt = rejectedByLowPt;
        rows.push_back(row);
    }
    return rows;
}

inline void ApplySpectrumFunctionConfig(TF1 *f,
                                        const std::string &funcName,
                                        const SpectrumFunctionConfig &cfg) {
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

inline int GetCanonicalNormIndex(const std::string &funcName) {
    if (funcName == "fBGBW") return 3;
    if (funcName == "fLevi") return 2;
    if (funcName == "fBoltzmann" || funcName == "fPtExp") return 1;
    if (funcName == "fTsallisBW") return 3;
    return -1;
}

inline double ClampToSpectrumLimit(double value, const SpectrumParLimit &lim) {
    if (!(lim.second > lim.first)) return value;
    return std::min(lim.second, std::max(lim.first, value));
}

inline void SeedSpectrumNormFromHistogram(TH1D *h,
                                          TF1 *f,
                                          const std::string &funcName,
                                          const SpectrumFunctionConfig &cfg,
                                          double fitMin,
                                          double fitMax) {
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
    if (static_cast<size_t>(iCanonNorm) < cfg.limits.size()) {
        seededNorm = ClampToSpectrumLimit(seededNorm, cfg.limits[static_cast<size_t>(iCanonNorm)]);
    }
    f->SetParameter(iParNorm, seededNorm);
}

inline std::vector<std::vector<double>> BuildSpectrumSeedGrid(const std::string &funcName,
                                                              const std::vector<double> &baseInit) {
    std::vector<std::vector<double>> seeds;
    if (!baseInit.empty()) seeds.push_back(baseInit);

    if (funcName == "fBGBW") {
        for (double beta : {0.45, 0.60, 0.72})
            for (double temp : {0.10, 0.16, 0.24, 0.34})
                for (double n : {0.5, 1.0, 2.5})
                    seeds.push_back({beta, temp, n, 1.0});
    } else if (funcName == "fLevi") {
        for (double temp : {0.06, 0.08, 0.09, 0.12, 0.18, 0.28, 0.45, 0.70, 1.00})
            for (double n : {2.1, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0, 120.0, 150.0, 200.0, 300.0, 500.0})
                seeds.push_back({temp, n, 1.0});
    } else if (funcName == "fBoltzmann") {
        for (double temp : {0.06, 0.08, 0.10, 0.12, 0.18, 0.25, 0.35, 0.50, 0.80, 1.20})
            seeds.push_back({temp, 1.0});
    } else if (funcName == "fPtExp") {
        for (double temp : {0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.35, 0.60, 0.90, 1.20, 1.80, 2.50})
            seeds.push_back({temp, 1.0});
    } else if (funcName == "fTsallisBW") {
        for (double beta : {0.35, 0.55, 0.70, 0.76, 0.82})
            for (double temp : {0.04, 0.08, 0.12, 0.16, 0.20, 0.28})
                for (double q : {1.0005, 1.002, 1.005, 1.010, 1.030, 1.080})
                    for (double ymax : {0.01, 0.03, 0.10, 0.50})
                        seeds.push_back({beta, temp, q, 1.0, ymax});
    }
    return seeds;
}

inline int CountPositiveSpectrumPoints(TH1D *h, double fitMin, double fitMax) {
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

inline void FixCanonicalSpectrumParameter(TF1 *f, const std::string &funcName, int iCanonical) {
    if (!f) return;
    const int ip = GetCanonicalParameterIndex(funcName, iCanonical);
    if (ip >= 0 && ip < f->GetNpar()) f->FixParameter(ip, f->GetParameter(ip));
}

inline void StabilizeSparseSpectrumFit(TH1D *h,
                                       TF1 *f,
                                       const std::string &funcName,
                                       double fitMin,
                                       double fitMax) {
    const int nPoints = CountPositiveSpectrumPoints(h, fitMin, fitMax);
    if (nPoints <= 0 || !f) return;
    const int maxFree = std::max(1, nPoints - 1);
    int nFree = static_cast<int>(ResolveSpectrumFunctionConfig(funcName, {}).initial.size());
    if (nFree <= 0) nFree = f->GetNpar();

    auto fixAndCount = [&](int iCanonical) {
        if (nFree <= maxFree) return;
        FixCanonicalSpectrumParameter(f, funcName, iCanonical);
        --nFree;
    };

    if (funcName == "fBGBW") {
        // For very sparse peripheral spectra, keep the stable shape anchors and fit T/norm.
        fixAndCount(2); // n
        fixAndCount(0); // beta
        fixAndCount(1); // T, only if a single free parameter is required
    } else if (funcName == "fLevi") {
        fixAndCount(1); // n
        fixAndCount(0); // T, only if a single free parameter is required
    } else if (funcName == "fTsallisBW") {
        fixAndCount(4); // ymax
        fixAndCount(2); // q
        fixAndCount(0); // beta
        fixAndCount(1); // T, only if a single free parameter is required
    }
}

inline std::unique_ptr<TF1> BuildSpectrumFunction(const std::string &funcName,
                                                  const std::string &nameTag,
                                                  const std::map<std::string, SpectrumFunctionConfig> &configuredPars = {},
                                                  double mass = 2.991) {
    // BGBW from AliPWGFunc does not remain fit-equivalent after TF1::Clone()
    // in this ROOT setup, so keep the AliPWGFunc owner alive and fit the
    // original TF1 directly.
    static std::vector<AliPWGFunc *> sAliPwgOwners;
    auto *helper = new AliPWGFunc();
    sAliPwgOwners.push_back(helper);
    helper->SetVarType(AliPWGFunc::kdNdpt);

    const SpectrumFunctionConfig parCfg = ResolveSpectrumFunctionConfig(funcName, configuredPars);
    TF1 *raw = nullptr;
    if (funcName == "fBGBW") {
        raw = helper->GetBGBW(mass, 0.6, 0.2, 1.0, 500.0, ("fBGBW_" + nameTag).c_str());
    } else if (funcName == "fLevi") {
        raw = helper->GetLevi(mass, 0.45, 20.0, 1.0, ("fLevi_" + nameTag).c_str());
    } else if (funcName == "fBoltzmann") {
        raw = helper->GetBoltzmann(mass, 0.30, 1.0, ("fBoltzmann_" + nameTag).c_str());
    } else if (funcName == "fPtExp") {
        raw = helper->GetPTExp(0.35, 1.0, ("fPtExp_" + nameTag).c_str());
    } else if (funcName == "fTsallisBW") {
        raw = helper->GetTsallisBW(mass, 0.76, 0.16, 1.003, 1.0, 0.03,
                                   ("fTsallisBW_" + nameTag).c_str());
    }

    if (!raw) {
        return nullptr;
    }
    if (funcName == "fTsallisBW" && raw->GetNpar() > 0) {
        raw->FixParameter(0, mass);
    }
    ApplySpectrumFunctionConfig(raw, funcName, parCfg);
    raw->SetRange(0.0, 10.0);
    raw->SetNpx(1200);
    raw->SetLineWidth(2);
    return std::unique_ptr<TF1>(raw);
}

inline bool FitHistogramWithFunction(TH1D *h,
                                     TF1 *f,
                                     double fitMin,
                                     double fitMax,
                                     TFitResultPtr *fitResultOut = nullptr,
                                     bool estimateParameterErrors = false) {
    if (!h || !f) return false;
    if (!(fitMax > fitMin)) return false;
    static std::mutex sFitMutex;
    const bool imtWasEnabled = ROOT::IsImplicitMTEnabled();
    const unsigned int imtThreads = ROOT::GetThreadPoolSize();
    if (imtWasEnabled) ROOT::DisableImplicitMT();

    int fitStatus = -1;
    TFitResultPtr fitRes;
    {
        std::lock_guard<std::mutex> lock(sFitMutex);
        fitRes = h->Fit(f, estimateParameterErrors ? "Q0RNSE" : "Q0RNS", "", fitMin, fitMax);
        fitStatus = static_cast<int>(fitRes);
    }

    if (imtWasEnabled) ROOT::EnableImplicitMT(imtThreads > 0 ? imtThreads : 1);
    if (fitResultOut) {
        *fitResultOut = fitRes;
    }
    if (fitStatus == 0) return true;

    const double probeX = 0.5 * (fitMin + fitMax);
    const double probeY = f->Eval(probeX);
    const double chi2 = f->GetChisquare();
    const int ndf = f->GetNDF();
    const double chi2Ndf = (ndf > 0 && std::isfinite(chi2)) ? chi2 / static_cast<double>(ndf)
                                                            : std::numeric_limits<double>::infinity();
    const bool hasUsableShape = ndf > 0 &&
                                std::isfinite(chi2) &&
                                std::isfinite(probeY) &&
                                probeY > 0.0;
    if (hasUsableShape) {
        const std::string fname = f->GetName();
        bool printWarning = fname.find("_trail_") == std::string::npos &&
                            fname.find("_seed") == std::string::npos;
        if (printWarning) {
            static std::mutex sWarnMutex;
            static std::set<std::string> sWarnedFunctions;
            std::lock_guard<std::mutex> lock(sWarnMutex);
            printWarning = sWarnedFunctions.insert(fname).second;
        }
        if (printWarning) {
            std::cout << "[Warn] FitHistogramWithFunction: accepting non-zero ROOT fit status "
                      << fitStatus << " for " << f->GetName()
                      << " because the fitted shape is finite (chi2/ndf="
                      << chi2Ndf << ", chi2=" << chi2 << ", ndf=" << ndf << ")" << std::endl;
        }
        return true;
    }
    return false;
}

inline std::unique_ptr<TF1> FitSpectrumFunctionBestSeed(
    TH1D *h,
    const std::string &funcName,
    const std::string &tag,
    const std::map<std::string, SpectrumFunctionConfig> &configuredPars,
    double fitMin,
    double fitMax,
    TFitResultPtr *fitResultOut = nullptr,
    bool estimateParameterErrors = false,
    const IntegralYieldConfig *lowPtGuardCfg = nullptr) {
    const SpectrumFunctionConfig baseCfg = ResolveSpectrumFunctionConfig(funcName, configuredPars);
    std::unique_ptr<TF1> bestFunc;
    TFitResultPtr bestFitRes;
    double bestScore = std::numeric_limits<double>::infinity();
    bool bestPassesLowPtGuard = false;
    int iSeed = 0;

    for (const auto &seed : BuildSpectrumSeedGrid(funcName, baseCfg.initial)) {
        SpectrumFunctionConfig seedCfg = baseCfg;
        seedCfg.initial = seed;
        std::map<std::string, SpectrumFunctionConfig> seedMap;
        seedMap[funcName] = seedCfg;
        auto f = BuildSpectrumFunction(funcName, tag + Form("_seed%d", iSeed), seedMap);
        ++iSeed;
        if (!f) continue;
        SeedSpectrumNormFromHistogram(h, f.get(), funcName, seedCfg, fitMin, fitMax);
        StabilizeSparseSpectrumFit(h, f.get(), funcName, fitMin, fitMax);

        TFitResultPtr fitRes;
        if (!FitHistogramWithFunction(h, f.get(), fitMin, fitMax, &fitRes, estimateParameterErrors)) continue;
        const double score = ComputeChi2Ndf(f.get());
        const int status = static_cast<int>(fitRes);
        const bool passesLowPtGuard = !lowPtGuardCfg || PassLowPtExtrapolationGuard(h, f.get(), *lowPtGuardCfg);
        const double rankedScore = score * (status == 0 ? 1.0 : 1.05);
        const bool improvesGuardClass = passesLowPtGuard && !bestPassesLowPtGuard;
        const bool sameGuardClassBetter = passesLowPtGuard == bestPassesLowPtGuard && rankedScore < bestScore;
        if (std::isfinite(rankedScore) && (improvesGuardClass || sameGuardClassBetter)) {
            bestScore = rankedScore;
            bestPassesLowPtGuard = passesLowPtGuard;
            bestFitRes = fitRes;
            bestFunc = std::move(f);
        }
    }

    if (fitResultOut) *fitResultOut = bestFitRes;
    return bestFunc;
}

inline double IntegrateFunction(TF1 *f, double xMin, double xMax) {
    if (!f || !(xMax > xMin)) return 0.0;
    constexpr int nSteps = 2000;
    const double h = (xMax - xMin) / static_cast<double>(nSteps);
    double sum = 0.0;
    for (int i = 0; i <= nSteps; ++i) {
        const double x = xMin + h * static_cast<double>(i);
        const double y = f->Eval(x);
        if (!std::isfinite(y)) return std::numeric_limits<double>::quiet_NaN();
        const double w = (i == 0 || i == nSteps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        sum += w * y;
    }
    return sum * h / 3.0;
}

inline double ComputeRmsAroundReference(const std::vector<double> &vals, double ref) {
    if (vals.empty()) return 0.0;
    double sum2 = 0.0;
    for (double v : vals) {
        const double d = v - ref;
        sum2 += d * d;
    }
    return std::sqrt(sum2 / static_cast<double>(vals.size()));
}

inline double ExtractAbsorptionMultiplier(const std::string &label) {
    const size_t xpos = label.find('x');
    const size_t start = (xpos == std::string::npos) ? 0 : xpos + 1;
    for (size_t i = start; i < label.size(); ++i) {
        if (!(std::isdigit(static_cast<unsigned char>(label[i])) || label[i] == '.' || label[i] == '+' || label[i] == '-')) continue;
        size_t j = i + 1;
        while (j < label.size() &&
               (std::isdigit(static_cast<unsigned char>(label[j])) || label[j] == '.' || label[j] == 'e' || label[j] == 'E' ||
                label[j] == '+' || label[j] == '-')) {
            ++j;
        }
        try {
            return std::stod(label.substr(i, j - i));
        } catch (...) {
            return std::numeric_limits<double>::infinity();
        }
    }
    return std::numeric_limits<double>::infinity();
}

inline void SeedFitParametersFromReference(TF1 *target, TF1 *reference) {
    if (!target || !reference) return;
    const int n = std::min(target->GetNpar(), reference->GetNpar());
    for (int ip = 0; ip < n; ++ip) {
        target->SetParameter(ip, reference->GetParameter(ip));
    }
}

inline bool IsWithinActiveParLimits(TF1 *f, int ip, double value) {
    if (!f || ip < 0 || ip >= f->GetNpar()) return false;
    double lo = 0.0;
    double hi = 0.0;
    f->GetParLimits(ip, lo, hi);
    if (!(hi > lo)) return true;
    return value >= lo && value <= hi;
}

inline bool IsNearActiveParLimit(TF1 *f, int ip) {
    if (!f || ip < 0 || ip >= f->GetNpar()) return false;
    double lo = 0.0;
    double hi = 0.0;
    f->GetParLimits(ip, lo, hi);
    if (!(hi > lo)) return false;
    const double value = f->GetParameter(ip);
    const double tol = std::max(1e-9, 1e-3 * (hi - lo));
    return std::abs(value - lo) < tol || std::abs(value - hi) < tol;
}

inline double ParameterFallbackSigma(TF1 *f, int ip) {
    if (!f || ip < 0 || ip >= f->GetNpar()) return 0.0;
    if (IsNearActiveParLimit(f, ip)) return 0.0;
    double lo = 0.0;
    double hi = 0.0;
    f->GetParLimits(ip, lo, hi);
    const double value = std::abs(f->GetParameter(ip));
    double sigma = (value > 0.0) ? 0.05 * value : 0.0;
    if (hi > lo) {
        const double spanSigma = 0.005 * (hi - lo);
        if (!(sigma > 0.0)) sigma = spanSigma;
        sigma = std::min(sigma, 0.02 * (hi - lo));
    }
    return (std::isfinite(sigma) && sigma > 0.0) ? sigma : 0.0;
}

inline std::unique_ptr<TPaveText> MakeIntegralLabelBox(const IntegralYieldInput &input,
                                                       const IntegralYieldConfig &cfg,
                                                       double x1,
                                                       double y1,
                                                       double x2,
                                                       double y2) {
    auto text = std::make_unique<TPaveText>(x1, y1, x2, y2, "NDC");
    text->SetBorderSize(0);
    text->SetFillStyle(0);
    text->SetTextAlign(12);
    text->SetTextFont(42);
    text->SetTextSize(0.035);
    if (!cfg.performanceLabel.empty() || cfg.usePerformanceLabel) {
        text->AddText((cfg.usePerformanceLabel ? cfg.performanceLabel : "ALICE").c_str());
    }
    if (!cfg.collisionSystem.empty() || !cfg.collisionEnergy.empty()) {
        text->AddText((cfg.collisionSystem + " " + cfg.collisionEnergy).c_str());
    }
    if (!cfg.period.empty() || !cfg.periodMark.empty()) {
        text->AddText((cfg.period + " " + cfg.periodMark).c_str());
    }
    const std::string decay = BuildIntegralDecayString(cfg.isMatter);
    if (!decay.empty()) {
        text->AddText(decay.c_str());
    }
    text->AddText(Form("Centrality %.0f-%.0f%%", input.cenMin, input.cenMax));
    return text;
}

inline IntegralYieldResult ComputeIntegralYield(const IntegralYieldInput &input,
                                                const IntegralYieldConfig &cfg,
                                                int randomSeed = 42) {
    IntegralYieldResult out;
    out.cenMin = input.cenMin;
    out.cenMax = input.cenMax;
    if (!input.hCorrected) {
        std::cout << "[Warn] IntegralYield: null corrected histogram for group "
                  << input.groupTag << std::endl;
        return out;
    }

    TH1D *h = input.hCorrected;
    const double histMin = h->GetXaxis()->GetXmin();
    const double histMax = h->GetXaxis()->GetXmax();
    const double fitMin = (std::isfinite(cfg.fitRangeMin) && std::isfinite(cfg.fitRangeMax) && cfg.fitRangeMax > cfg.fitRangeMin)
                              ? cfg.fitRangeMin
                              : histMin;
    const double fitMax = (std::isfinite(cfg.fitRangeMin) && std::isfinite(cfg.fitRangeMax) && cfg.fitRangeMax > cfg.fitRangeMin)
                              ? cfg.fitRangeMax
                              : histMax;

    const auto *measuredEdges = input.measuredPtEdges.empty() ? nullptr : &input.measuredPtEdges;
    double measuredMin = histMin;
    double measuredMax = histMax;
    GetMeasuredRange(h, measuredEdges, measuredMin, measuredMax);

    std::cout << "[Info] IntegralYield: start group " << input.groupTag
              << " (cen " << input.cenMin << "-" << input.cenMax << "%)"
              << ", nominal=" << cfg.nominalFitFunc
              << ", doSystematics=" << (cfg.doSystematics ? "true" : "false")
              << ", nTrails=" << cfg.nTrailsForIntegralSyst
              << ", fitRange=[" << fitMin << "," << fitMax << "]"
              << ", integratedYieldRange=[" << cfg.integrateMin << "," << cfg.integrateMax << "]"
              << ", measuredRange=[" << measuredMin << "," << measuredMax << "]"
              << std::endl;

    std::vector<std::string> nominalTryList;
    nominalTryList.push_back(cfg.nominalFitFunc);
    for (const auto &name : cfg.fitFuncCandidates) {
        if (std::find(nominalTryList.begin(), nominalTryList.end(), name) == nominalTryList.end()) {
            nominalTryList.push_back(name);
        }
    }
    const std::vector<std::string> defaults = {"fBGBW", "fLevi", "fBoltzmann", "fPtExp", "fTsallisBW"};
    for (const auto &name : defaults) {
        if (std::find(nominalTryList.begin(), nominalTryList.end(), name) == nominalTryList.end()) {
            nominalTryList.push_back(name);
        }
    }

    std::unique_ptr<TF1> fNom;
    std::string usedNominalName;
    TFitResultPtr nominalFitResult;
    for (const auto &tryName : nominalTryList) {
        TFitResultPtr fitRes;
        auto fTry = FitSpectrumFunctionBestSeed(h,
                                                tryName,
                                                input.groupTag + "_nominal_" + tryName,
                                                cfg.fitFuncParameters,
                                                fitMin,
                                                fitMax,
                                                &fitRes,
                                                cfg.useMinosErrors,
                                                &cfg);
        if (!fTry) {
            std::cout << "[Warn] IntegralYield: nominal fit failed with " << tryName
                      << " for group " << input.groupTag << std::endl;
            continue;
        }
        usedNominalName = tryName;
        nominalFitResult = fitRes;
        fNom = std::move(fTry);
        break;
    }
    if (!fNom) {
        std::cout << "[Error] IntegralYield: all nominal fit candidates failed for group "
                  << input.groupTag << std::endl;
        return out;
    }

    const double yNom = ComputeHybridIntegral(h, fNom.get(), cfg, measuredEdges);
    const double yStatHist = ComputeHistogramIntegralError(h);
    std::vector<std::pair<double, double>> statFitRanges = BuildExtrapolationRanges(h, cfg, measuredEdges);
    const double yStatFit = ComputeFitIntegralErrorFromCovariance(fNom.get(),
                                                                  nominalFitResult,
                                                                  statFitRanges);
    const double yStat = std::hypot(yStatHist, yStatFit);
    std::cout << "[Info] IntegralYield: stat error for " << input.groupTag
              << " hist=" << yStatHist
              << ", fitCov=" << yStatFit
              << ", used=" << yStat
              << " (measured bins summed directly; fit covariance propagated only outside measured pT range)" << std::endl;

    out.value = yNom;
    out.statErr = yStat;
    out.fNominal = std::move(fNom);
    {
        auto rows = BuildIntegralFitParameterRows(usedNominalName,
                                                  out.fNominal.get(),
                                                  cfg.fitFuncParameters,
                                                  &nominalFitResult,
                                                  true,
                                                  false,
                                                  false);
        out.fitParameterRows.insert(out.fitParameterRows.end(),
                                    std::make_move_iterator(rows.begin()),
                                    std::make_move_iterator(rows.end()));
    }

    // Fit-function systematic and fit comparison canvas.
    std::vector<double> fitFuncIntegrals;
    std::vector<std::string> fitNames = cfg.doSystematics ? cfg.fitFuncCandidates
                                                          : std::vector<std::string>{usedNominalName};
    if (fitNames.empty()) fitNames.push_back(usedNominalName);

    out.cNominalAndFunctions = std::make_unique<TCanvas>(("c_integral_funcs_" + input.groupTag).c_str(), "", 900, 700);
    out.cNominalAndFunctions->cd();
    out.cNominalAndFunctions->SetLogy(true);
    out.cNominalAndFunctions->SetLeftMargin(0.14);
    out.cNominalAndFunctions->SetBottomMargin(0.12);
    out.cNominalAndFunctions->SetRightMargin(0.04);
    out.cNominalAndFunctions->SetTopMargin(0.06);
    out.cNominalAndFunctions->SetTicks(1, 1);
    h->SetStats(false);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(1.0);
    h->SetMarkerColor(kBlack);
    h->SetLineColor(kBlack);
    h->SetLineWidth(2);
    h->SetTitle("");
    const double xPlotMin = 0.0;
    const double xPlotMax = 10.0;
    double yPlotMin = std::numeric_limits<double>::infinity();
    double yPlotMax = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double x = h->GetXaxis()->GetBinCenter(ib);
        if (x < xPlotMin || x > xPlotMax) continue;
        const double y = h->GetBinContent(ib);
        const double eStat = h->GetBinError(ib);
        const double eSys = input.hCorrectedSyst ? input.hCorrectedSyst->GetBinContent(ib) : 0.0;
        if (std::isfinite(y) && y > 0.0) {
            yPlotMin = std::min(yPlotMin, std::max(1e-20, y - eStat - eSys));
            yPlotMax = std::max(yPlotMax, y + eStat + eSys);
        }
    }
    if (!std::isfinite(yPlotMin) || yPlotMin <= 0.0) yPlotMin = std::max(1e-12, h->GetMinimum(1));
    if (!std::isfinite(yPlotMax) || yPlotMax <= yPlotMin) yPlotMax = std::max(yPlotMin * 10.0, h->GetMaximum());
    yPlotMin *= 0.55;
    yPlotMax *= 1.55;
    if (yPlotMax / yPlotMin > 1e5) yPlotMin = yPlotMax / 1e5;
    auto *fitFrame = out.cNominalAndFunctions->DrawFrame(xPlotMin,
                                                         yPlotMin,
                                                         xPlotMax,
                                                         yPlotMax);
    fitFrame->SetTitle("");
    fitFrame->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    fitFrame->GetYaxis()->SetTitle("#frac{1}{N_{ev}} #frac{d^{2}N}{d#it{p}_{T}d#it{y}} ((GeV/#it{c})^{-1})");
    fitFrame->GetYaxis()->SetTitleOffset(1.35);

    if (input.hCorrectedSyst) {
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetXaxis()->GetBinCenter(ib);
            const double y = h->GetBinContent(ib);
            const double ey = input.hCorrectedSyst->GetBinContent(ib);
            if (!std::isfinite(y) || !std::isfinite(ey) || ey <= 0.0) continue;
            const double halfWidth = 0.5 * h->GetXaxis()->GetBinWidth(ib);
            TBox box(x - halfWidth, std::max(1e-20, y - ey), x + halfWidth, y + ey);
            box.SetFillStyle(0);
            box.SetLineColor(kBlue + 2);
            box.SetLineWidth(2);
            box.SetLineStyle(kDashed);
            box.DrawClone("l");
        }
    }
    h->Draw("E1 X0 SAME");

    TLegend leg(0.52, 0.62, 0.90, 0.90);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextFont(42);
    leg.SetTextSize(0.032);
    leg.AddEntry(h, "Corrected spectrum (stat.)", "lep");
    if (input.hCorrectedSyst) {
        auto *sysLine = new TLine();
        sysLine->SetLineColor(kBlue + 2);
        sysLine->SetLineStyle(kDashed);
        sysLine->SetLineWidth(2);
        leg.AddEntry(sysLine, "Point-to-point syst.", "l");
    }

    const std::map<std::string, int> funcColors = {
        {"fBGBW", kRed + 1},
        {"fLevi", kAzure + 2},
        {"fBoltzmann", kGreen + 2},
        {"fPtExp", kMagenta + 2},
        {"fTsallisBW", kViolet + 1}};
    const double nominalFitFuncChi2Ndf = ComputeSpectrumChi2Ndf(h, out.fNominal.get(), fitMin, fitMax);
    const double fitFuncMaxChi2Ndf =
        (std::isfinite(cfg.fitFuncMaxChi2Ndf) && cfg.fitFuncMaxChi2Ndf > 0.0)
            ? cfg.fitFuncMaxChi2Ndf
            : std::max(5.0, 2.0 * nominalFitFuncChi2Ndf + 1.0);
    std::vector<std::string> chi2Lines;
    std::vector<std::unique_ptr<TGraph>> curveGraphs;
    std::vector<std::string> curveLabels;
    std::vector<std::vector<std::string>> parameterTextBlocks;
    for (const auto &name : fitNames) {
        TFitResultPtr fitRes;
        auto f = FitSpectrumFunctionBestSeed(h,
                                             name,
                                             input.groupTag + "_" + name,
                                             cfg.fitFuncParameters,
                                             fitMin,
                                             fitMax,
                                             &fitRes,
                                             false,
                                             name == usedNominalName ? nullptr : &cfg);
        if (!f) {
            std::cout << "[Warn] IntegralYield: skip fit-func source, fit failed for "
                      << name << " in group " << input.groupTag << std::endl;
            continue;
        }
        const double chi2Ndf = ComputeChi2Ndf(f.get());
        double lowPtMax = std::numeric_limits<double>::quiet_NaN();
        double lowPtRef = std::numeric_limits<double>::quiet_NaN();
        const bool rejectedByLowPt = name != usedNominalName &&
                                     !PassLowPtExtrapolationGuard(h, f.get(), cfg, &lowPtMax, &lowPtRef);
        const bool rejectedByChi2 = cfg.rejectFitFuncByChi2 &&
                                    name != usedNominalName &&
                                    std::isfinite(chi2Ndf) &&
                                    chi2Ndf > fitFuncMaxChi2Ndf;
        chi2Lines.push_back(Form("%s: #chi^{2}/ndf = %.2f%s%s",
                                 name.c_str(),
                                 chi2Ndf,
                                 rejectedByChi2 ? " rejected" : "",
                                 rejectedByLowPt ? " low-p_{T}" : ""));
        if (rejectedByChi2) {
            std::cout << "[Warn] IntegralYield: reject fit function " << name
                      << " in " << input.groupTag << " by chi2/ndf=" << chi2Ndf
                      << " > " << fitFuncMaxChi2Ndf << std::endl;
        }
        if (rejectedByLowPt) {
            std::cout << "[Warn] IntegralYield: reject fit function " << name
                      << " in " << input.groupTag
                      << " by low-pT guard, maxLowPt=" << lowPtMax
                      << " > " << cfg.lowPtMaxFactor << " * firstMeasuredRef=" << lowPtRef
                      << std::endl;
        }
        const auto itColor = funcColors.find(name);
        const int lineColor = itColor != funcColors.end() ? itColor->second : kGray + 2;
        auto gCurve = std::make_unique<TGraph>();
        gCurve->SetName((input.groupTag + "_" + name + "_curve").c_str());
        gCurve->SetLineColor(lineColor);
        gCurve->SetLineWidth(name == usedNominalName ? 3 : 2);
        gCurve->SetLineStyle(name == usedNominalName ? kSolid : kDashed);
        const int nCurvePoints = 400;
        const double xCurveMin = xPlotMin;
        const double xCurveMax = xPlotMax;
        for (int ipt = 0; ipt < nCurvePoints; ++ipt) {
            const double x = xCurveMin + (xCurveMax - xCurveMin) * static_cast<double>(ipt) /
                                           static_cast<double>(nCurvePoints - 1);
            const double y = f->Eval(x);
            if (std::isfinite(y) && y > 0.0) gCurve->SetPoint(gCurve->GetN(), x, y);
        }
        gCurve->DrawClone("L SAME");
        curveGraphs.push_back(std::move(gCurve));
        std::string legLabel = (name == usedNominalName) ? (name + " (std.)") : name;
        if (rejectedByChi2 || rejectedByLowPt) legLabel += " (rejected)";
        leg.AddEntry(curveGraphs.back().get(), legLabel.c_str(), "l");
        curveLabels.push_back(legLabel);
        parameterTextBlocks.push_back(BuildSpectrumParameterTextLines(name, f.get(), cfg.fitFuncParameters));
        if (name != usedNominalName) {
            auto rows = BuildIntegralFitParameterRows(name,
                                                      f.get(),
                                                      cfg.fitFuncParameters,
                                                      &fitRes,
                                                      false,
                                                      rejectedByChi2,
                                                      rejectedByLowPt);
            out.fitParameterRows.insert(out.fitParameterRows.end(),
                                        std::make_move_iterator(rows.begin()),
                                        std::make_move_iterator(rows.end()));
        }
        const double yi = ComputeHybridIntegral(h, f.get(), cfg, measuredEdges);
        if (name != usedNominalName && !rejectedByChi2 && !rejectedByLowPt) fitFuncIntegrals.push_back(yi);
        out.fFitCandidates.push_back(std::move(f));
    }
    leg.DrawClone();
    auto fitText = MakeIntegralLabelBox(input, cfg, 0.16, 0.68, 0.50, 0.90);
    fitText->DrawClone();
    out.systFitFunction = ComputeRmsAroundReference(fitFuncIntegrals, yNom);
    if (cfg.doSystematics && fitFuncIntegrals.empty() && fitNames.size() > 1) {
        out.systFitFunction = std::abs(yNom) * std::max(0.0, cfg.fitFuncFallbackFraction);
        chi2Lines.push_back(Form("Fit-func fallback syst = %.1f%%", 100.0 * cfg.fitFuncFallbackFraction));
        std::cout << "[Warn] IntegralYield: no accepted non-nominal fit functions in "
                  << input.groupTag << ", use fallback fit-function systematic "
                  << 100.0 * cfg.fitFuncFallbackFraction << "%" << std::endl;
    }
    auto chi2Text = std::make_unique<TPaveText>(0.16, 0.46, 0.50, 0.66, "NDC");
    chi2Text->SetBorderSize(0);
    chi2Text->SetFillStyle(0);
    chi2Text->SetTextAlign(12);
    chi2Text->SetTextFont(42);
    chi2Text->SetTextSize(0.030);
    const std::string chi2CutLine = cfg.rejectFitFuncByChi2
                                        ? (cfg.fitFuncMaxChi2Ndf > 0.0
                                               ? Form("Fit #chi^{2}/ndf cut: < %.1f", fitFuncMaxChi2Ndf)
                                               : Form("Fit #chi^{2}/ndf dyn cut: < %.1f", fitFuncMaxChi2Ndf))
                                        : "Fit #chi^{2}/ndf cut: off";
    chi2Text->AddText(chi2CutLine.c_str());
    for (const auto &line : chi2Lines) chi2Text->AddText(line.c_str());
    chi2Text->DrawClone();

    out.cFitFunctionParameters = std::make_unique<TCanvas>(("c_integral_func_params_" + input.groupTag).c_str(), "", 900, 700);
    out.cFitFunctionParameters->cd();
    out.cFitFunctionParameters->SetLogy(true);
    out.cFitFunctionParameters->SetLeftMargin(0.14);
    out.cFitFunctionParameters->SetBottomMargin(0.12);
    out.cFitFunctionParameters->SetRightMargin(0.04);
    out.cFitFunctionParameters->SetTopMargin(0.06);
    out.cFitFunctionParameters->SetTicks(1, 1);
    auto *paramFrame = out.cFitFunctionParameters->DrawFrame(xPlotMin,
                                                             yPlotMin,
                                                             xPlotMax,
                                                             yPlotMax);
    paramFrame->SetTitle("");
    paramFrame->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    paramFrame->GetYaxis()->SetTitle("#frac{1}{N_{ev}} #frac{d^{2}N}{d#it{p}_{T}d#it{y}} ((GeV/#it{c})^{-1})");
    paramFrame->GetYaxis()->SetTitleOffset(1.35);
    if (input.hCorrectedSyst) {
        for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
            const double x = h->GetXaxis()->GetBinCenter(ib);
            const double y = h->GetBinContent(ib);
            const double ey = input.hCorrectedSyst->GetBinContent(ib);
            if (!std::isfinite(y) || !std::isfinite(ey) || ey <= 0.0) continue;
            const double halfWidth = 0.5 * h->GetXaxis()->GetBinWidth(ib);
            TBox box(x - halfWidth, std::max(1e-20, y - ey), x + halfWidth, y + ey);
            box.SetFillStyle(0);
            box.SetLineColor(kBlue + 2);
            box.SetLineWidth(2);
            box.SetLineStyle(kDashed);
            box.DrawClone("l");
        }
    }
    h->Draw("E1 X0 SAME");
    TLegend paramLeg(0.58, 0.66, 0.90, 0.90);
    paramLeg.SetBorderSize(0);
    paramLeg.SetFillStyle(0);
    paramLeg.SetTextFont(42);
    paramLeg.SetTextSize(0.032);
    for (size_t i = 0; i < curveGraphs.size(); ++i) {
        curveGraphs[i]->DrawClone("L SAME");
        const std::string label = (i < curveLabels.size()) ? curveLabels[i] : curveGraphs[i]->GetName();
        paramLeg.AddEntry(curveGraphs[i].get(), label.c_str(), "l");
    }
    paramLeg.DrawClone();
    auto paramText = std::make_unique<TPaveText>(0.16, 0.34, 0.56, 0.90, "NDC");
    paramText->SetBorderSize(0);
    paramText->SetFillStyle(0);
    paramText->SetTextAlign(12);
    paramText->SetTextFont(42);
    paramText->SetTextSize(0.022);
    for (const auto &block : parameterTextBlocks) {
        for (const auto &line : block) paramText->AddText(line.c_str());
    }
    paramText->DrawClone();

    // Extrapolation systematic from correlated Gaussian toys of fit params.
    // Only the unmeasured pT ranges are varied; measured bins are summed directly.
    const auto extrapRanges = BuildExtrapolationRanges(h, cfg, measuredEdges);
    const double extrapNom = IntegrateFunctionRanges(out.fNominal.get(), extrapRanges);

    std::vector<double> extrapYields;
    if (cfg.doSystematics && nominalFitResult.Get() && cfg.nCombinationsForExtrapolation > 0) {
        const bool trustFitCovariance = (static_cast<int>(nominalFitResult) == 0);
        if (!trustFitCovariance) {
            std::cout << "[Warn] IntegralYield: nominal fit status="
                      << static_cast<int>(nominalFitResult)
                      << " in " << input.groupTag
                      << ", use conservative diagonal fallback for extrap toys" << std::endl;
        }
        const int nPar = out.fNominal ? out.fNominal->GetNpar() : 0;
        if (nPar > 0) {
            TMatrixDSym cov = nominalFitResult->GetCovarianceMatrix();
            const SpectrumFunctionConfig nominalParCfg =
                ResolveSpectrumFunctionConfig(usedNominalName, cfg.fitFuncParameters);
            const int normParIndex =
                GetCanonicalParameterIndex(usedNominalName, GetCanonicalNormIndex(usedNominalName));
            std::vector<int> freePars;
            for (int ip = 0; ip < nPar; ++ip) {
                if (ip == normParIndex) continue;
                double lo = 0.0;
                double hi = 0.0;
                out.fNominal->GetParLimits(ip, lo, hi);
                const bool fixedPar = lo == hi && lo != 0.0;
                const double tf1Err = out.fNominal->GetParError(ip);
                const double fitErr = nominalFitResult->ParError(ip);
                const double covErr = (ip < cov.GetNrows() && cov(ip, ip) > 0.0) ? std::sqrt(cov(ip, ip)) : 0.0;
                const double fallbackErr = ParameterFallbackSigma(out.fNominal.get(), ip);
                const bool hasUncertainty = trustFitCovariance
                                                ? ((std::isfinite(tf1Err) && tf1Err > 0.0) ||
                                                   (std::isfinite(fitErr) && fitErr > 0.0) ||
                                                   (std::isfinite(covErr) && covErr > 0.0) ||
                                                   (std::isfinite(fallbackErr) && fallbackErr > 0.0))
                                                : (std::isfinite(fallbackErr) && fallbackErr > 0.0);
                if (!fixedPar && hasUncertainty) {
                    freePars.push_back(ip);
                }
            }

            std::vector<double> thetaFull(static_cast<size_t>(nPar), 0.0);
            for (int ip = 0; ip < nPar; ++ip) thetaFull[static_cast<size_t>(ip)] = out.fNominal->GetParameter(ip);

            if (freePars.empty()) {
                std::cout << "[Warn] IntegralYield: no free parameters with finite errors for extrap toys in "
                          << input.groupTag << std::endl;
            } else {
                if (normParIndex >= 0 && normParIndex < nPar) {
                    std::cout << "[Info] IntegralYield: extrap toys in " << input.groupTag
                              << " sample shape parameters only; norm parameter p" << normParIndex
                              << " is refitted to measured spectrum for each toy" << std::endl;
                }
                TMatrixDSym covFree(static_cast<int>(freePars.size()));
                TVectorD thetaHat(static_cast<int>(freePars.size()));
                for (size_t i = 0; i < freePars.size(); ++i) {
                    thetaHat[static_cast<int>(i)] = out.fNominal->GetParameter(freePars[i]);
                    for (size_t j = 0; j < freePars.size(); ++j) {
                        const int ip = freePars[i];
                        const int jp = freePars[j];
                        covFree(static_cast<int>(i), static_cast<int>(j)) =
                            (ip < cov.GetNrows() && jp < cov.GetNcols()) ? cov(ip, jp) : 0.0;
                    }
                }

                TMatrixDSymEigen eigen(covFree);
                const TVectorD eigenValues = eigen.GetEigenValues();
                const TMatrixD eigenVectors = eigen.GetEigenVectors();
                bool validEigenBasis = trustFitCovariance;
                bool hasPositiveEigenValue = false;
                for (int i = 0; i < eigenValues.GetNrows(); ++i) {
                    if (!std::isfinite(eigenValues[i]) || eigenValues[i] < -1e-18) {
                        validEigenBasis = false;
                        break;
                    }
                    if (eigenValues[i] > 0.0) hasPositiveEigenValue = true;
                }
                if (!hasPositiveEigenValue) validEigenBasis = false;

                TRandom3 rand(static_cast<UInt_t>(randomSeed));
                const int barLenEx = 20;
                if (!validEigenBasis) {
                    std::cout << "[Warn] IntegralYield: invalid covariance eigenbasis for extrap toys in "
                              << input.groupTag << ", fallback to diagonal parameter errors" << std::endl;
                }
                const double nominalSpectrumChi2Ndf = ComputeSpectrumChi2Ndf(h, out.fNominal.get(), fitMin, fitMax);
                const double extrapToyMaxChi2Ndf =
                    (std::isfinite(cfg.extrapToyMaxChi2Ndf) && cfg.extrapToyMaxChi2Ndf > 0.0)
                        ? cfg.extrapToyMaxChi2Ndf
                        : std::max(5.0, 2.0 * nominalSpectrumChi2Ndf + 1.0);
                int rejectedByMeasuredShape = 0;
                for (int k = 0; k < cfg.nCombinationsForExtrapolation; ++k) {
                    bool acceptedToy = false;
                    if (validEigenBasis) {
                        for (int attempt = 0; attempt < 200 && !acceptedToy; ++attempt) {
                            TVectorD independentShift(static_cast<int>(freePars.size()));
                            for (int i = 0; i < static_cast<int>(freePars.size()); ++i) {
                                const double sigma = std::sqrt(std::max(0.0, eigenValues[i]));
                                independentShift[i] = rand.Gaus(0.0, sigma);
                            }

                            TVectorD theta = thetaHat + eigenVectors * independentShift;
                            bool inRange = true;
                            for (size_t i = 0; i < freePars.size(); ++i) {
                                const double value = theta[static_cast<int>(i)];
                                if (!std::isfinite(value) ||
                                    !IsWithinActiveParLimits(out.fNominal.get(), freePars[i], value)) {
                                    inRange = false;
                                    break;
                                }
                            }
                            if (!inRange) continue;
                            for (size_t i = 0; i < freePars.size(); ++i) {
                                out.fNominal->SetParameter(freePars[i], theta[static_cast<int>(i)]);
                            }
                            SeedSpectrumNormFromHistogram(h,
                                                          out.fNominal.get(),
                                                          usedNominalName,
                                                          nominalParCfg,
                                                          fitMin,
                                                          fitMax);
                            acceptedToy = true;
                        }
                    } else {
                        for (int attempt = 0; attempt < 200 && !acceptedToy; ++attempt) {
                            std::vector<double> thetaCandidate;
                            thetaCandidate.reserve(freePars.size());
                            bool inRange = true;
                            for (int ip : freePars) {
                                const double tf1Err = out.fNominal->GetParError(ip);
                                const double fitErr = nominalFitResult->ParError(ip);
                                const double covErr = (ip < cov.GetNrows() && cov(ip, ip) > 0.0) ? std::sqrt(cov(ip, ip)) : 0.0;
                                double sigma = ParameterFallbackSigma(out.fNominal.get(), ip);
                                if (trustFitCovariance) {
                                    sigma = (std::isfinite(fitErr) && fitErr > 0.0) ? fitErr : tf1Err;
                                    if (!(std::isfinite(sigma) && sigma > 0.0)) sigma = covErr;
                                    if (!(std::isfinite(sigma) && sigma > 0.0)) sigma = ParameterFallbackSigma(out.fNominal.get(), ip);
                                }
                                const double value = rand.Gaus(thetaFull[static_cast<size_t>(ip)], sigma);
                                thetaCandidate.push_back(value);
                                if (!std::isfinite(value) ||
                                    !IsWithinActiveParLimits(out.fNominal.get(), ip, value)) {
                                    inRange = false;
                                    break;
                                }
                            }
                            if (!inRange) continue;
                            for (size_t i = 0; i < freePars.size(); ++i) {
                                out.fNominal->SetParameter(freePars[i], thetaCandidate[i]);
                            }
                            SeedSpectrumNormFromHistogram(h,
                                                          out.fNominal.get(),
                                                          usedNominalName,
                                                          nominalParCfg,
                                                          fitMin,
                                                          fitMax);
                            acceptedToy = true;
                        }
                    }
                    if (!acceptedToy) continue;

                    const double yExtrapToy = IntegrateFunctionRanges(out.fNominal.get(), extrapRanges);
                    const double toyChi2Ndf = ComputeSpectrumChi2Ndf(h, out.fNominal.get(), fitMin, fitMax);
                    if (std::isfinite(yExtrapToy) &&
                        yExtrapToy >= 0.0 &&
                        std::isfinite(toyChi2Ndf) &&
                        toyChi2Ndf <= extrapToyMaxChi2Ndf) {
                        extrapYields.push_back(yExtrapToy);
                    } else {
                        ++rejectedByMeasuredShape;
                    }

                    const int filledEx = static_cast<int>(std::round(static_cast<double>(k + 1) /
                                                                     static_cast<double>(cfg.nCombinationsForExtrapolation) * barLenEx));
                    std::string barEx(static_cast<size_t>(std::max(0, filledEx)), '<');
                    std::cout << "\r[Info] Integral " << input.groupTag
                              << " extrap toys " << (k + 1) << "/" << cfg.nCombinationsForExtrapolation
                              << " " << barEx << std::flush;
                }
                std::cout << std::endl;
                if (rejectedByMeasuredShape > 0) {
                    std::cout << "[Info] IntegralYield: rejected " << rejectedByMeasuredShape
                              << " extrap toys in " << input.groupTag
                              << " by measured-spectrum compatibility (max chi2/ndf="
                              << extrapToyMaxChi2Ndf << ")" << std::endl;
                }

                for (int ip = 0; ip < nPar; ++ip) out.fNominal->SetParameter(ip, thetaFull[static_cast<size_t>(ip)]);
            }
        }
    }

    if (cfg.doSystematics) {
        out.cExtrapolation = std::make_unique<TCanvas>(("c_integral_extrap_" + input.groupTag).c_str(), "", 900, 700);
        out.cExtrapolation->cd();
    }
    if (cfg.doSystematics && !extrapYields.empty()) {
        double rMin = *std::min_element(extrapYields.begin(), extrapYields.end());
        double rMax = *std::max_element(extrapYields.begin(), extrapYields.end());
        if (!(rMax > rMin)) {
            rMin -= 1e-9;
            rMax += 1e-9;
        }
        out.hExtrapRatioDist = std::make_unique<TH1D>(("h_integral_extrap_unmeasured_" + input.groupTag).c_str(),
                                                      ";Y_{unmeasured}^{toy};Entries",
                                                      60,
                                                      rMin,
                                                      rMax);
        out.hExtrapRatioDist->SetDirectory(nullptr);
        out.hExtrapRatioDist->SetStats(false);
        out.hExtrapRatioDist->SetLineColor(kBlack);
        out.hExtrapRatioDist->SetLineWidth(2);
        for (double yv : extrapYields) out.hExtrapRatioDist->Fill(yv);

        TBox *bandGauss = nullptr;
        TLine *lineStd = nullptr;
        std::unique_ptr<TF1> fgaus;
        double gausMean = std::numeric_limits<double>::quiet_NaN();
        double gausSigma = std::numeric_limits<double>::quiet_NaN();
        double gausChi2Ndf = std::numeric_limits<double>::infinity();
        bool gausAccepted = false;

        const double yTop = std::max(1.0, out.hExtrapRatioDist->GetMaximum()) * 1.02;
        lineStd = new TLine(extrapNom, 0.0, extrapNom, yTop);
        lineStd->SetLineColor(kRed + 2);
        lineStd->SetLineWidth(2);

        const double initMean = out.hExtrapRatioDist->GetMean();
        const double initSigma = std::max(out.hExtrapRatioDist->GetRMS(), 1e-9);
        fgaus = std::make_unique<TF1>((input.groupTag + "_f_extrap_lowpt_gaus").c_str(),
                                      "gaus",
                                      out.hExtrapRatioDist->GetXaxis()->GetXmin(),
                                      out.hExtrapRatioDist->GetXaxis()->GetXmax());
        fgaus->SetParameters(out.hExtrapRatioDist->GetMaximum(), initMean, initSigma);
        auto fitRes = out.hExtrapRatioDist->Fit(fgaus.get(), "QSN0");
        gausChi2Ndf = ComputeChi2Ndf(fgaus.get());
        gausAccepted = AcceptGaussianFit(fitRes, fgaus.get(), cfg.gaussFitMaxChi2Ndf);
        if (static_cast<int>(fitRes) == 0 && fgaus) {
            fgaus->SetLineColor(kGreen + 3);
            fgaus->SetLineWidth(2);
            fgaus->SetLineStyle(gausAccepted ? kSolid : kDashed);
            gausMean = fgaus->GetParameter(1);
            gausSigma = std::abs(fgaus->GetParameter(2));
            if (std::isfinite(gausMean) && std::isfinite(gausSigma) && gausSigma > 0.0) {
                bandGauss = new TBox(gausMean - gausSigma, 0.0, gausMean + gausSigma, yTop);
                bandGauss->SetFillStyle(3005);
                bandGauss->SetFillColor(kGreen + 1);
                bandGauss->SetLineColor(kGreen + 3);
            }
        }

        out.hExtrapRatioDist->Draw("HIST");
        if (bandGauss) bandGauss->Draw("same");
        if (fgaus && static_cast<int>(fitRes) == 0) fgaus->DrawCopy("same");
        if (lineStd) lineStd->Draw("same");
        out.hExtrapRatioDist->Draw("HIST SAME");

        auto legEx = std::make_unique<TLegend>(0.56, 0.70, 0.90, 0.90);
        legEx->SetBorderSize(0);
        legEx->SetFillStyle(0);
        legEx->SetTextFont(42);
        legEx->AddEntry(out.hExtrapRatioDist.get(), "Unmeasured yield toys", "l");
        if (lineStd) legEx->AddEntry(lineStd, "Std unmeasured yield", "l");
        if (bandGauss) legEx->AddEntry(bandGauss, "Gauss #pm1#sigma", "f");
        if (fgaus && static_cast<int>(fitRes) == 0) {
            legEx->AddEntry(fgaus.get(), gausAccepted ? "Gauss fit" : "Gauss fit rejected", "l");
        }
        legEx->DrawClone();

        auto paveEx = std::make_unique<TPaveText>(0.14, 0.64, 0.48, 0.90, "NDC");
        paveEx->SetBorderSize(0);
        paveEx->SetFillStyle(0);
        paveEx->SetTextAlign(12);
        paveEx->SetTextFont(42);
        paveEx->AddText(Form("Cen: %.0f-%.0f%%", input.cenMin, input.cenMax));
        for (const auto &range : extrapRanges) {
            paveEx->AddText(Form("Unmeasured: [%.2f, %.2f]", range.first, range.second));
        }
        paveEx->AddText(Form("Std Y_{unmeas} = %.4e", extrapNom));
        if (std::isfinite(gausMean) && std::isfinite(gausSigma)) {
            paveEx->AddText(Form("Gauss #mu = %.4e", gausMean));
            paveEx->AddText(Form("Gauss #sigma = %.4e", gausSigma));
        }
        if (std::isfinite(gausChi2Ndf)) {
            paveEx->AddText(Form("Gauss #chi^{2}/ndf = %.2f (%s)",
                                 gausChi2Ndf,
                                 gausAccepted ? "used" : "RMS used"));
        }
        paveEx->AddText(Form("Hist RMS = %.4e", out.hExtrapRatioDist->GetRMS()));
        paveEx->DrawClone();

        out.systExtrapolation = (gausAccepted && std::isfinite(gausSigma) && gausSigma > 0.0)
                                     ? gausSigma
                                     : out.hExtrapRatioDist->GetRMS();
    } else if (cfg.doSystematics) {
        out.systExtrapolation = 0.0;
        std::cout << "[Warn] IntegralYield: empty extrapolation yield distribution for "
                  << input.groupTag << std::endl;
    } else {
        out.systExtrapolation = 0.0;
        std::cout << "[Info] IntegralYield: extrapolation systematic skipped for "
                  << input.groupTag << " because execution.do_systematics=false" << std::endl;
    }

    // Absorption systematic: each variant rescales all bins coherently, then refit.
    struct AbsorptionScanPoint {
        double multiplier;
        std::string label;
        double integral;
    };
    std::vector<AbsorptionScanPoint> absorptionScanPoints;
    if (cfg.doSystematics) {
        out.cAbsorption = std::make_unique<TCanvas>(("c_integral_absorption_" + input.groupTag).c_str(), "", 900, 700);
        out.cAbsorption->cd();
        out.cAbsorption->SetLogy(true);
        h->SetLineColor(kBlack);
        h->SetMarkerColor(kBlack);
        h->SetMarkerStyle(20);
        h->Draw("E1");
        TLegend legAb(0.46, 0.60, 0.90, 0.90);
        legAb.SetBorderSize(0);
        legAb.SetFillStyle(0);
        legAb.AddEntry(h, "Std corrected counts", "lep");

        int abColor = kMagenta + 1;
        for (size_t i = 0; i < input.absorptionVariants.size(); ++i) {
            TH1D *hv = input.absorptionVariants[i];
            if (!hv) continue;
            auto f = BuildSpectrumFunction(usedNominalName, input.groupTag + "_abso_" + std::to_string(i), cfg.fitFuncParameters);
            if (!f) continue;
            if (!FitHistogramWithFunction(hv, f.get(), fitMin, fitMax)) continue;
            hv->SetMarkerColor(abColor);
            hv->SetLineColor(abColor);
            hv->SetMarkerStyle(24 + static_cast<int>(i % 4));
            hv->SetLineWidth(2);
            f->SetLineColor(abColor);
            f->SetLineStyle(static_cast<int>(i % 3) + 1);
            f->SetLineWidth(2);
            hv->DrawCopy("E1 SAME");
            f->DrawCopy("SAME");
            const std::string label = (i < input.absorptionVariantLabels.size() && !input.absorptionVariantLabels[i].empty())
                                          ? input.absorptionVariantLabels[i]
                                          : ("var_" + std::to_string(i));
            legAb.AddEntry(f.get(), (label + " " + usedNominalName).c_str(), "l");
            absorptionScanPoints.push_back(AbsorptionScanPoint{
                ExtractAbsorptionMultiplier(label),
                label,
                ComputeHybridIntegral(hv, f.get(), cfg, measuredEdges)});
            abColor += 2;
        }
        legAb.Draw();
        {
            TLatex lab;
            lab.SetNDC();
            lab.SetTextSize(0.035);
            lab.DrawLatex(0.15, 0.88, Form("%s, cen %.0f-%.0f%%", input.groupTag.c_str(), input.cenMin, input.cenMax));
        }
    } else {
        out.systAbsorption = 0.0;
    }

    if (cfg.doSystematics && !absorptionScanPoints.empty()) {
        std::sort(absorptionScanPoints.begin(), absorptionScanPoints.end(), [](const auto &a, const auto &b) {
            if (a.multiplier == b.multiplier) return a.label < b.label;
            return a.multiplier < b.multiplier;
        });
        std::vector<double> absorptionIntegrals;
        absorptionIntegrals.reserve(absorptionScanPoints.size());
        for (const auto &p : absorptionScanPoints) absorptionIntegrals.push_back(p.integral);
        const auto mm = std::minmax_element(absorptionIntegrals.begin(), absorptionIntegrals.end());
        out.systAbsorption = std::max(0.0, cfg.absorptionLength) * (*mm.second - *mm.first);

        out.cAbsorptionYieldScan = std::make_unique<TCanvas>(("c_integral_absorption_scan_" + input.groupTag).c_str(), "", 900, 700);
        out.cAbsorptionYieldScan->cd();
        out.hAbsorptionYieldScan = std::make_unique<TH1D>(("h_integral_absorption_scan_" + input.groupTag).c_str(),
                                                          ";n #times #sigma_{He3};Integrated yield",
                                                          static_cast<int>(absorptionScanPoints.size()),
                                                          0.5,
                                                          static_cast<double>(absorptionScanPoints.size()) + 0.5);
        out.hAbsorptionYieldScan->SetDirectory(nullptr);
        out.hAbsorptionYieldScan->SetStats(false);
        out.hAbsorptionYieldScan->SetFillColor(kAzure - 9);
        out.hAbsorptionYieldScan->SetLineColor(kAzure + 2);
        out.hAbsorptionYieldScan->SetLineWidth(2);
        for (size_t i = 0; i < absorptionScanPoints.size(); ++i) {
            out.hAbsorptionYieldScan->SetBinContent(static_cast<int>(i + 1), absorptionScanPoints[i].integral);
            out.hAbsorptionYieldScan->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), absorptionScanPoints[i].label.c_str());
        }
        out.hAbsorptionYieldScan->Draw("HIST");
        TLatex lab;
        lab.SetNDC();
        lab.SetTextSize(0.035);
        lab.DrawLatex(0.15, 0.88, Form("Absorption syst (max-min) = %.3e", out.systAbsorption));
    }

    // Corrected-count trail systematic: independent random draw in each measured pT bin.
    std::vector<double> trailIntegrals;
    if (cfg.doSystematics && !input.trailCorrectedValuesByBin.empty() && cfg.nTrailsForIntegralSyst > 0) {
        std::mt19937 rng(static_cast<unsigned>(randomSeed));
        const int barLen = 20;
        for (int it = 0; it < cfg.nTrailsForIntegralSyst; ++it) {
            auto hTrial = std::unique_ptr<TH1D>(static_cast<TH1D *>(h->Clone(("h_trial_" + input.groupTag).c_str())));
            hTrial->SetDirectory(nullptr);
            for (int ib = 1; ib <= hTrial->GetNbinsX(); ++ib) {
                const size_t idx = static_cast<size_t>(ib - 1);
                if (idx < input.trailCorrectedValuesByBin.size()) {
                    const auto &vals = input.trailCorrectedValuesByBin[idx];
                    if (!vals.empty()) {
                        std::uniform_int_distribution<size_t> pick(0, vals.size() - 1);
                        hTrial->SetBinContent(ib, vals[pick(rng)]);
                    }
                }
            }
            auto f = BuildSpectrumFunction(usedNominalName, input.groupTag + "_trail_" + std::to_string(it), cfg.fitFuncParameters);
            if (!f) continue;
            SeedFitParametersFromReference(f.get(), out.fNominal.get());
            if (!FitHistogramWithFunction(hTrial.get(), f.get(), fitMin, fitMax)) continue;
            trailIntegrals.push_back(ComputeHybridIntegral(hTrial.get(), f.get(), cfg, measuredEdges));

            const int filled = static_cast<int>(std::round(static_cast<double>(it + 1) /
                                                           static_cast<double>(cfg.nTrailsForIntegralSyst) * barLen));
            std::string bar(static_cast<size_t>(std::max(0, filled)), '<');
            std::cout << "\r[Info] Integral " << input.groupTag
                      << " trails " << (it + 1) << "/" << cfg.nTrailsForIntegralSyst
                      << " " << bar << std::flush;
        }
        std::cout << std::endl;
    } else {
        std::cout << "[Info] Integral " << input.groupTag
                  << " trails skipped (systematics disabled, empty inputs, or nTrails<=0)" << std::endl;
    }

    if (!trailIntegrals.empty()) {
        const auto [trailRawMean, trailRawRms] = ComputeMeanRms(trailIntegrals);
        auto [vMin, vMax] = ComputeRobustDisplayRange(trailIntegrals,
                                                       yNom,
                                                       std::max(1e-12, yStat),
                                                       0.01,
                                                       0.99);
        out.hIntegralTrailDist = std::make_unique<TH1D>(("h_integral_trails_" + input.groupTag).c_str(),
                                                        ";Integrated yield;Entries",
                                                        60,
                                                        vMin,
                                                        vMax);
        out.hIntegralTrailDist->SetDirectory(nullptr);
        out.hIntegralTrailDist->SetStats(false);
        for (double v : trailIntegrals) out.hIntegralTrailDist->Fill(v);

        out.cTrails = std::make_unique<TCanvas>(("c_integral_trails_" + input.groupTag).c_str(), "", 900, 700);
        out.cTrails->cd();
        out.hIntegralTrailDist->SetLineColor(kBlack);
        out.hIntegralTrailDist->SetLineWidth(2);
        out.hIntegralTrailDist->Draw("HIST");

        const double yTopTrail = std::max(1.0, out.hIntegralTrailDist->GetMaximum()) * 1.02;
        TLine *lineStdTrail = new TLine(yNom, 0.0, yNom, yTopTrail);
        lineStdTrail->SetLineColor(kRed + 2);
        lineStdTrail->SetLineWidth(2);
        TBox *bandStdTrail = nullptr;
        if (yStat > 0.0) {
            bandStdTrail = new TBox(yNom - yStat, 0.0, yNom + yStat, yTopTrail);
            bandStdTrail->SetFillStyle(3004);
            bandStdTrail->SetFillColor(kOrange - 2);
            bandStdTrail->SetLineColor(kRed + 2);
        }

        std::unique_ptr<TF1> fgausTrail;
        TBox *bandGaussTrail = nullptr;
        double gausMeanTrail = std::numeric_limits<double>::quiet_NaN();
        double gausSigmaTrail = std::numeric_limits<double>::quiet_NaN();
        double gausChi2NdfTrail = std::numeric_limits<double>::infinity();
        bool gausAcceptedTrail = false;
        fgausTrail = std::make_unique<TF1>((input.groupTag + "_f_trails_gaus").c_str(),
                                           "gaus",
                                           out.hIntegralTrailDist->GetXaxis()->GetXmin(),
                                           out.hIntegralTrailDist->GetXaxis()->GetXmax());
        fgausTrail->SetParameters(out.hIntegralTrailDist->GetMaximum(),
                                  out.hIntegralTrailDist->GetMean(),
                                  std::max(out.hIntegralTrailDist->GetRMS(), 1e-9));
        auto fitResTrail = out.hIntegralTrailDist->Fit(fgausTrail.get(), "QSN0");
        gausChi2NdfTrail = ComputeChi2Ndf(fgausTrail.get());
        gausAcceptedTrail = AcceptGaussianFit(fitResTrail, fgausTrail.get(), cfg.gaussFitMaxChi2Ndf);
        if (static_cast<int>(fitResTrail) == 0 && fgausTrail) {
            fgausTrail->SetLineColor(kGreen + 3);
            fgausTrail->SetLineWidth(2);
            fgausTrail->SetLineStyle(gausAcceptedTrail ? kSolid : kDashed);
            gausMeanTrail = fgausTrail->GetParameter(1);
            gausSigmaTrail = std::abs(fgausTrail->GetParameter(2));
            if (std::isfinite(gausMeanTrail) && std::isfinite(gausSigmaTrail) && gausSigmaTrail > 0.0) {
                bandGaussTrail = new TBox(gausMeanTrail - gausSigmaTrail, 0.0, gausMeanTrail + gausSigmaTrail, yTopTrail);
                bandGaussTrail->SetFillStyle(3005);
                bandGaussTrail->SetFillColor(kGreen + 1);
                bandGaussTrail->SetLineColor(kGreen + 3);
            }
        }

        if (bandStdTrail) bandStdTrail->Draw("same");
        if (bandGaussTrail) bandGaussTrail->Draw("same");
        if (fgausTrail && static_cast<int>(fitResTrail) == 0) fgausTrail->DrawCopy("same");
        if (lineStdTrail) lineStdTrail->Draw("same");
        out.hIntegralTrailDist->Draw("HIST SAME");

        auto legTrail = std::make_unique<TLegend>(0.56, 0.70, 0.90, 0.90);
        legTrail->SetBorderSize(0);
        legTrail->SetFillStyle(0);
        legTrail->SetTextFont(42);
        legTrail->AddEntry(out.hIntegralTrailDist.get(), "Trails distribution", "l");
        if (lineStdTrail) legTrail->AddEntry(lineStdTrail, "Std integral", "l");
        if (bandStdTrail) legTrail->AddEntry(bandStdTrail, "Std stat band", "f");
        if (bandGaussTrail) legTrail->AddEntry(bandGaussTrail, "Gauss #pm1#sigma", "f");
        if (fgausTrail && static_cast<int>(fitResTrail) == 0) {
            legTrail->AddEntry(fgausTrail.get(), gausAcceptedTrail ? "Gauss fit" : "Gauss fit rejected", "l");
        }
        legTrail->DrawClone();

        auto paveTrail = std::make_unique<TPaveText>(0.14, 0.64, 0.48, 0.90, "NDC");
        paveTrail->SetBorderSize(0);
        paveTrail->SetFillStyle(0);
        paveTrail->SetTextAlign(12);
        paveTrail->SetTextFont(42);
        paveTrail->AddText(Form("Cen: %.0f-%.0f%%", input.cenMin, input.cenMax));
        if (std::isfinite(gausMeanTrail) && std::isfinite(gausSigmaTrail)) {
            paveTrail->AddText(Form("Gauss #mu = %.3e", gausMeanTrail));
            paveTrail->AddText(Form("Gauss #sigma = %.3e", gausSigmaTrail));
        }
        if (std::isfinite(gausChi2NdfTrail)) {
            paveTrail->AddText(Form("Gauss #chi^{2}/ndf = %.2f (%s)",
                                    gausChi2NdfTrail,
                                    gausAcceptedTrail ? "used" : "RMS used"));
        }
        paveTrail->AddText(Form("Shown Mean = %.3e", out.hIntegralTrailDist->GetMean()));
        paveTrail->AddText(Form("Raw Mean = %.3e", trailRawMean));
        paveTrail->AddText(Form("Shown RMS = %.3e", out.hIntegralTrailDist->GetRMS()));
        paveTrail->AddText(Form("Raw RMS = %.3e", trailRawRms));
        paveTrail->DrawClone();

        if (gausAcceptedTrail && std::isfinite(gausSigmaTrail) && gausSigmaTrail > 0.0) {
            out.systTrails = gausSigmaTrail;
        } else {
            out.systTrails = trailRawRms;
        }
    }

    out.systBranchingRatio = cfg.doSystematics
                                  ? std::abs(yNom) * std::max(0.0, cfg.branchingRatioFractionalUncertainty)
                                  : 0.0;
    out.systTotal = std::sqrt(out.systExtrapolation * out.systExtrapolation +
                              out.systFitFunction * out.systFitFunction +
                              out.systAbsorption * out.systAbsorption +
                              out.systTrails * out.systTrails +
                              out.systBranchingRatio * out.systBranchingRatio);

    out.hSystSources = std::make_unique<TH1D>(("h_integral_syst_sources_" + input.groupTag).c_str(),
                                              ";Source;Absolute uncertainty",
                                              5,
                                              0.5,
                                              5.5);
    out.hSystSources->SetDirectory(nullptr);
    out.hSystSources->SetStats(false);
    out.hSystSources->GetXaxis()->SetBinLabel(1, "Extrap");
    out.hSystSources->GetXaxis()->SetBinLabel(2, "FitFunc");
    out.hSystSources->GetXaxis()->SetBinLabel(3, "Absorption");
    out.hSystSources->GetXaxis()->SetBinLabel(4, "CorrTrails");
    out.hSystSources->GetXaxis()->SetBinLabel(5, "Branching");
    out.hSystSources->SetBinContent(1, out.systExtrapolation);
    out.hSystSources->SetBinContent(2, out.systFitFunction);
    out.hSystSources->SetBinContent(3, out.systAbsorption);
    out.hSystSources->SetBinContent(4, out.systTrails);
    out.hSystSources->SetBinContent(5, out.systBranchingRatio);

    out.hSystSourceFractions = std::make_unique<TH1D>(("h_integral_syst_source_fractions_" + input.groupTag).c_str(),
                                                      ";Source;Uncertainty / integrated yield (%)",
                                                      5,
                                                      0.5,
                                                      5.5);
    out.hSystSourceFractions->SetDirectory(nullptr);
    out.hSystSourceFractions->SetStats(false);
    out.hSystSourceFractions->GetXaxis()->SetBinLabel(1, "Extrap");
    out.hSystSourceFractions->GetXaxis()->SetBinLabel(2, "FitFunc");
    out.hSystSourceFractions->GetXaxis()->SetBinLabel(3, "Absorption");
    out.hSystSourceFractions->GetXaxis()->SetBinLabel(4, "CorrTrails");
    out.hSystSourceFractions->GetXaxis()->SetBinLabel(5, "Branching");
    const double fracDen = std::abs(out.value);
    if (fracDen > 0.0) {
        out.hSystSourceFractions->SetBinContent(1, 100.0 * out.systExtrapolation / fracDen);
        out.hSystSourceFractions->SetBinContent(2, 100.0 * out.systFitFunction / fracDen);
        out.hSystSourceFractions->SetBinContent(3, 100.0 * out.systAbsorption / fracDen);
        out.hSystSourceFractions->SetBinContent(4, 100.0 * out.systTrails / fracDen);
        out.hSystSourceFractions->SetBinContent(5, 100.0 * out.systBranchingRatio / fracDen);
    }

    out.cSources = std::make_unique<TCanvas>(("c_integral_syst_sources_" + input.groupTag).c_str(), "", 900, 700);
    out.cSources->cd();
    out.cSources->SetLeftMargin(0.12);
    out.cSources->SetBottomMargin(0.14);
    out.hSystSourceFractions->SetFillColor(kOrange - 3);
    out.hSystSourceFractions->SetLineColor(kBlack);
    out.hSystSourceFractions->Draw("HIST");

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.035);
    latex.DrawLatex(0.15, 0.88, Form("Integrated yield %.0f-%.0f%% = %.4e #pm %.4e (stat) #pm %.4e (sys)",
                                     input.cenMin,
                                     input.cenMax,
                                     out.value,
                                     out.statErr,
                                     out.systTotal));

    out.hIntegralYieldOneBin = std::make_unique<TH1D>(("h_integral_yield_onebin_" + input.groupTag).c_str(),
                                                      ";Centrality;Integrated yield",
                                                      1,
                                                      0.0,
                                                      1.0);
    out.hIntegralYieldOneBin->SetDirectory(nullptr);
    out.hIntegralYieldOneBin->SetStats(false);
    out.hIntegralYieldOneBin->GetXaxis()->SetBinLabel(1, Form("%.0f-%.0f%%", input.cenMin, input.cenMax));
    out.hIntegralYieldOneBin->SetBinContent(1, out.value);
    out.hIntegralYieldOneBin->SetBinError(1, out.statErr);
    out.hIntegralYieldOneBin->SetMarkerStyle(20);
    out.hIntegralYieldOneBin->SetMarkerColor(kBlack);
    out.hIntegralYieldOneBin->SetLineColor(kBlack);

    out.cIntegralYieldOneBin = std::make_unique<TCanvas>(("c_integral_yield_onebin_" + input.groupTag).c_str(), "", 900, 700);
    out.cIntegralYieldOneBin->cd();
    out.cIntegralYieldOneBin->SetLeftMargin(0.14);
    out.cIntegralYieldOneBin->SetBottomMargin(0.12);
    const double yLow = std::max(0.0, out.value - 1.8 * (out.statErr + out.systTotal));
    const double yHigh = out.value + 1.8 * (out.statErr + out.systTotal);
    out.hIntegralYieldOneBin->GetYaxis()->SetRangeUser(yLow, yHigh > yLow ? yHigh : out.value * 1.2 + 1e-12);
    out.hIntegralYieldOneBin->Draw("E1");
    TBox sysBox(0.25, out.value - out.systTotal, 0.75, out.value + out.systTotal);
    sysBox.SetFillColorAlpha(kAzure + 1, 0.25);
    sysBox.SetLineColor(kAzure + 2);
    sysBox.DrawClone("same");
    out.hIntegralYieldOneBin->Draw("E1 SAME");
    TLegend yieldLeg(0.54, 0.75, 0.90, 0.90);
    yieldLeg.SetBorderSize(0);
    yieldLeg.SetFillStyle(0);
    yieldLeg.SetTextSize(0.035);
    yieldLeg.AddEntry(out.hIntegralYieldOneBin.get(), "Statistical uncertainty", "lep");
    yieldLeg.AddEntry(&sysBox, "Total systematic uncertainty", "f");
    yieldLeg.DrawClone();
    TPaveText yieldText(0.15, 0.62, 0.56, 0.90, "NDC");
    yieldText.SetBorderSize(0);
    yieldText.SetFillStyle(0);
    yieldText.SetTextAlign(12);
    yieldText.SetTextFont(42);
    yieldText.SetTextSize(0.035);
    if (!cfg.performanceLabel.empty() || cfg.usePerformanceLabel) {
        yieldText.AddText((cfg.usePerformanceLabel ? cfg.performanceLabel : "ALICE").c_str());
    }
    if (!cfg.collisionSystem.empty() || !cfg.collisionEnergy.empty()) {
        yieldText.AddText((cfg.collisionSystem + " " + cfg.collisionEnergy).c_str());
    }
    if (!cfg.period.empty() || !cfg.periodMark.empty()) {
        yieldText.AddText((cfg.period + " " + cfg.periodMark).c_str());
    }
    const std::string decay = BuildIntegralDecayString(cfg.isMatter);
    if (!decay.empty()) {
        yieldText.AddText(decay.c_str());
    }
    yieldText.AddText(Form("Centrality %.0f-%.0f%%", input.cenMin, input.cenMax));
    yieldText.AddText(Form("Y = %.4e", out.value));
    yieldText.AddText(Form("stat = %.4e, syst = %.4e", out.statErr, out.systTotal));
    yieldText.DrawClone();

    std::cout << "[Info] IntegralYield: done group " << input.groupTag
              << ", nominalUsed=" << usedNominalName
              << ", value=" << out.value
              << ", stat=" << out.statErr
              << ", sys=" << out.systTotal
              << std::endl;

    out.ok = true;
    return out;
}

} // namespace UnifiedAnalysis

#endif // INTEGRAL_YIELD_HELPER_H
