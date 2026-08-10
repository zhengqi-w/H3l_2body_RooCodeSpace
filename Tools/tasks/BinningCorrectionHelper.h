#ifndef BINNING_CORRECTION_HELPER_H
#define BINNING_CORRECTION_HELPER_H

#include "../AcceptanceHelper.h"
#include "../AbsorptionHelper.h"
#include "../EventSignalLossHelper.h"
#include "../GeneralHelper.hpp"
#include "../binning/BinPlan.h"

#include <ROOT/RDataFrame.hxx>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace UnifiedAnalysis {

struct BinningCorrectionConfig {
    std::string mode;
    std::string isMatter;
    std::string eventSignalLossMethod{"multiplicity"};
    std::string mcTree;
    std::string absorptionTree;
    std::string mcFileForAcceptance;
    std::vector<std::string> mcFilesForAcceptance;
    std::vector<double> mcFileWeights;
    std::vector<std::vector<double>> mcFileWeightsByCentrality;
    std::string mcFileForAbsorption;
    std::string mcEfficiencySelection;
    bool mcAcceptanceRequireTwoBody{true};
    bool mcAcceptanceConstrainDenominatorOuterBin{true};
    double originalCtaoAbsorption{7.6};
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCentrality;
    std::vector<std::vector<std::string>> topologySelectionsByCentrality;
    struct SpectrumMcCentralityOverride {
        double targetMin{0.0};
        double targetMax{0.0};
        double sourceMin{0.0};
        double sourceMax{0.0};
    };
    bool useSpectrumMcCentralityOverrides{false};
    std::vector<SpectrumMcCentralityOverride> spectrumMcCentralityOverrides;
    std::vector<double> ptBins;
    std::vector<std::vector<double>> ctBinsByPt;
    std::vector<double> radBins;
    std::vector<std::vector<double>> ctBinsByRad;
};

struct PtCtAcceptanceCache {
    bool ready{false};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> valuesByPt;
    std::vector<std::vector<double>> genCountsByPt;
};

struct RadCtAcceptanceCache {
    bool ready{false};
    std::vector<double> radBins;
    std::vector<std::vector<double>> valuesByRad;
    std::vector<std::vector<double>> genCountsByRad;
};

struct PtCtAbsorptionCache {
    bool ready{false};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> valuesByPt;
};

struct RadCtAbsorptionCache {
    bool ready{false};
    std::vector<double> radBins;
    std::vector<std::vector<double>> valuesByRad;
};

struct BinValueWithError {
    double value{1.0};
    double error{0.0};
};

struct SpectrumAcceptanceCache {
    bool ready{false};
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCentrality;
    std::vector<std::vector<BinValueWithError>> valuesByCentrality;
    std::vector<std::vector<double>> genCountsByCentrality;
};

struct SpectrumEventSignalLossCache {
    bool ready{false};
    std::vector<double> cenBins;
    std::vector<std::vector<double>> ptBinsByCentrality;
    std::vector<bool> applyCorrectionByCentrality;
    std::vector<std::vector<BinValueWithError>> eventLossByCentrality;
    std::vector<std::vector<BinValueWithError>> eventSplittingByCentrality;
    std::vector<std::vector<BinValueWithError>> signalLossByCentrality;
};

inline std::unique_ptr<TChain> MakeChainFromFileForCorrection(const std::string &file, const std::string &tree) {
    auto chain = std::make_unique<TChain>(tree.c_str());
    TFile f(file.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Failed to open " + file);
    }
    TTree *t = dynamic_cast<TTree *>(f.Get(tree.c_str()));
    if (t) {
        chain->Add(file.c_str());
    } else {
        GeneralHelper::fillChainFromAO2D(*chain, &f);
    }
    if (chain->GetEntries() == 0) {
        throw std::runtime_error("No entries found for tree " + tree + " in " + file);
    }
    return chain;
}

inline int FindPtIndex(const std::vector<double> &ptBins, double ptMin, double ptMax) {
    for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
        if (std::abs(ptBins[ipt] - ptMin) < 1e-6 && std::abs(ptBins[ipt + 1] - ptMax) < 1e-6) {
            return static_cast<int>(ipt);
        }
    }
    return -1;
}

inline bool SameBinEdges(const std::vector<double> &lhs,
                         const std::vector<double> &rhs,
                         double tolerance = 1e-6) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) >= tolerance) return false;
    }
    return true;
}

inline int FindCentralityIndex(const std::vector<double> &cenBins, double cenMin, double cenMax) {
    for (size_t ic = 0; ic + 1 < cenBins.size(); ++ic) {
        if (std::abs(cenBins[ic] - cenMin) < 1e-6 && std::abs(cenBins[ic + 1] - cenMax) < 1e-6) {
            return static_cast<int>(ic);
        }
    }
    return -1;
}

inline int FindPtIndexInPerCentBinning(const std::vector<std::vector<double>> &ptBinsByCentrality,
                                       int cenIdx,
                                       double ptMin,
                                       double ptMax) {
    if (cenIdx < 0 || static_cast<size_t>(cenIdx) >= ptBinsByCentrality.size()) return -1;
    return FindPtIndex(ptBinsByCentrality[static_cast<size_t>(cenIdx)], ptMin, ptMax);
}

inline const BinningCorrectionConfig::SpectrumMcCentralityOverride*
FindSpectrumMcCentralityOverride(const BinningCorrectionConfig &cfg, double cenMin, double cenMax) {
    if (!cfg.useSpectrumMcCentralityOverrides) return nullptr;
    for (const auto &ov : cfg.spectrumMcCentralityOverrides) {
        if (std::abs(ov.targetMin - cenMin) < 1e-6 &&
            std::abs(ov.targetMax - cenMax) < 1e-6 &&
            ov.sourceMax > ov.sourceMin) {
            return &ov;
        }
    }
    return nullptr;
}

inline SpectrumEventSignalLossCache BuildSpectrumEventSignalLossCache(const BinningCorrectionConfig &cfg,
                                                                      const std::string &eventSignalLossFile,
                                                                      const std::string &mcFileForSignalLoss,
                                                                      const std::vector<bool> &addEventSignalLossByCent) {
    SpectrumEventSignalLossCache cache;
    if (!(cfg.mode == "bdt_spectrum" || cfg.mode == "topology_spectrum")) return cache;
    if (cfg.cenBins.size() < 2) return cache;
    if (cfg.ptBinsByCentrality.size() != cfg.cenBins.size() - 1) return cache;

    const int nCent = static_cast<int>(cfg.cenBins.size()) - 1;
    cache.cenBins = cfg.cenBins;
    cache.ptBinsByCentrality = cfg.ptBinsByCentrality;
    cache.applyCorrectionByCentrality.assign(static_cast<size_t>(nCent), false);
    for (int ic = 0; ic < nCent; ++ic) {
        cache.applyCorrectionByCentrality[static_cast<size_t>(ic)] =
            (static_cast<size_t>(ic) < addEventSignalLossByCent.size()) ? addEventSignalLossByCent[static_cast<size_t>(ic)] : false;
    }

    cache.eventLossByCentrality.assign(static_cast<size_t>(nCent), std::vector<BinValueWithError>{});
    cache.eventSplittingByCentrality.assign(static_cast<size_t>(nCent), std::vector<BinValueWithError>{});
    cache.signalLossByCentrality.assign(static_cast<size_t>(nCent), std::vector<BinValueWithError>{});
    for (int ic = 0; ic < nCent; ++ic) {
        const auto &ptEdges = cfg.ptBinsByCentrality[static_cast<size_t>(ic)];
        if (ptEdges.size() < 2) continue;
        const size_t nPt = ptEdges.size() - 1;
        cache.eventLossByCentrality[static_cast<size_t>(ic)].assign(nPt, BinValueWithError{1.0, 0.0});
        cache.eventSplittingByCentrality[static_cast<size_t>(ic)].assign(nPt, BinValueWithError{1.0, 0.0});
        cache.signalLossByCentrality[static_cast<size_t>(ic)].assign(nPt, BinValueWithError{1.0, 0.0});
    }

    const bool useImpactParameter = (cfg.eventSignalLossMethod == "impactparameter" ||
                                     cfg.eventSignalLossMethod == "impact_parameter" ||
                                     cfg.eventSignalLossMethod == "impact");

    // Event loss is centrality-dependent only; choose the configured method.
    if (!eventSignalLossFile.empty()) {
        auto evLossRes = EventSignalLossHelper::ComputeEventLoss(eventSignalLossFile, cfg.cenBins);
        for (int ic = 0; ic < nCent; ++ic) {
            const auto &values = useImpactParameter ? evLossRes.impactValue : evLossRes.multiplicityValue;
            const auto &errors = useImpactParameter ? evLossRes.impactError : evLossRes.multiplicityError;
            const double v = (static_cast<size_t>(ic) < values.size() && values[static_cast<size_t>(ic)] > 0.0)
                                 ? values[static_cast<size_t>(ic)]
                                 : 1.0;
            const double e = (static_cast<size_t>(ic) < errors.size())
                                 ? errors[static_cast<size_t>(ic)]
                                 : 0.0;
            const double split = (static_cast<size_t>(ic) < evLossRes.eventSplittingValue.size() &&
                                  evLossRes.eventSplittingValue[static_cast<size_t>(ic)] > 0.0)
                                     ? evLossRes.eventSplittingValue[static_cast<size_t>(ic)]
                                     : 1.0;
            const double splitErr = (static_cast<size_t>(ic) < evLossRes.eventSplittingError.size())
                                        ? evLossRes.eventSplittingError[static_cast<size_t>(ic)]
                                        : 0.0;
            const auto correctedEventLoss = EventSignalLossHelper::RatioWithError(v, e, split, splitErr);
            for (auto &cell : cache.eventLossByCentrality[static_cast<size_t>(ic)]) {
                cell.value = correctedEventLoss.first > 0.0 ? correctedEventLoss.first : 1.0;
                cell.error = correctedEventLoss.first > 0.0 ? correctedEventLoss.second : 0.0;
            }
            for (auto &cell : cache.eventSplittingByCentrality[static_cast<size_t>(ic)]) {
                cell.value = split;
                cell.error = splitErr;
            }
        }
        evLossRes.Clear();
    }

    // Signal loss is centrality + pt dependent and is computed from EventLoss QA histograms.
    (void)mcFileForSignalLoss;
    if (!eventSignalLossFile.empty()) {
        auto sigLossRes = EventSignalLossHelper::ComputeSignalLossCenPt(eventSignalLossFile,
                                                                  cfg.cenBins,
                                                                  cfg.ptBinsByCentrality);

        const auto &hVec = useImpactParameter ? sigLossRes.impact_pt_per_cent
                          : (cfg.isMatter == "matter") ? sigLossRes.signal_loss_pt_per_cent_matter
                          : (cfg.isMatter == "antimatter") ? sigLossRes.signal_loss_pt_per_cent_antimatter
                          : sigLossRes.signal_loss_pt_per_cent;
        for (int ic = 0; ic < nCent; ++ic) {
            if (static_cast<size_t>(ic) >= hVec.size() || !hVec[static_cast<size_t>(ic)]) continue;
            TH1D *h = hVec[static_cast<size_t>(ic)];
            for (size_t ip = 0; ip < cache.signalLossByCentrality[static_cast<size_t>(ic)].size(); ++ip) {
                const int bin = static_cast<int>(ip + 1);
                double v = h->GetBinContent(bin);
                if (v <= 0.0) v = 1.0;
                const double e = h->GetBinError(bin);
                cache.signalLossByCentrality[static_cast<size_t>(ic)][ip] = BinValueWithError{v, e};
            }
        }
        sigLossRes.Clear();
    }

    cache.ready = true;
    return cache;
}

inline BinValueWithError GetSpectrumEventLossForBin(const SpectrumEventSignalLossCache &cache,
                                                    const BinPlanItem &item) {
    if (!cache.ready || !item.hasCen || !item.hasPt) return BinValueWithError{1.0, 0.0};
    const int cenIdx = FindCentralityIndex(cache.cenBins, item.cenMin, item.cenMax);
    if (cenIdx < 0 || static_cast<size_t>(cenIdx) >= cache.eventLossByCentrality.size()) return BinValueWithError{1.0, 0.0};
    if (!cache.applyCorrectionByCentrality[static_cast<size_t>(cenIdx)]) return BinValueWithError{1.0, 0.0};
    const int ptIdx = FindPtIndexInPerCentBinning(cache.ptBinsByCentrality, cenIdx, item.ptMin, item.ptMax);
    if (ptIdx < 0 || static_cast<size_t>(ptIdx) >= cache.eventLossByCentrality[static_cast<size_t>(cenIdx)].size()) return BinValueWithError{1.0, 0.0};
    return cache.eventLossByCentrality[static_cast<size_t>(cenIdx)][static_cast<size_t>(ptIdx)];
}

inline BinValueWithError GetSpectrumEventSplittingForBin(const SpectrumEventSignalLossCache &cache,
                                                         const BinPlanItem &item) {
    if (!cache.ready || !item.hasCen || !item.hasPt) return BinValueWithError{1.0, 0.0};
    const int cenIdx = FindCentralityIndex(cache.cenBins, item.cenMin, item.cenMax);
    if (cenIdx < 0 || static_cast<size_t>(cenIdx) >= cache.eventSplittingByCentrality.size()) return BinValueWithError{1.0, 0.0};
    if (!cache.applyCorrectionByCentrality[static_cast<size_t>(cenIdx)]) return BinValueWithError{1.0, 0.0};
    const int ptIdx = FindPtIndexInPerCentBinning(cache.ptBinsByCentrality, cenIdx, item.ptMin, item.ptMax);
    if (ptIdx < 0 || static_cast<size_t>(ptIdx) >= cache.eventSplittingByCentrality[static_cast<size_t>(cenIdx)].size()) return BinValueWithError{1.0, 0.0};
    return cache.eventSplittingByCentrality[static_cast<size_t>(cenIdx)][static_cast<size_t>(ptIdx)];
}

inline BinValueWithError GetSpectrumSignalLossForBin(const SpectrumEventSignalLossCache &cache,
                                                     const BinPlanItem &item) {
    if (!cache.ready || !item.hasCen || !item.hasPt) return BinValueWithError{1.0, 0.0};
    const int cenIdx = FindCentralityIndex(cache.cenBins, item.cenMin, item.cenMax);
    if (cenIdx < 0 || static_cast<size_t>(cenIdx) >= cache.signalLossByCentrality.size()) return BinValueWithError{1.0, 0.0};
    if (!cache.applyCorrectionByCentrality[static_cast<size_t>(cenIdx)]) return BinValueWithError{1.0, 0.0};
    const int ptIdx = FindPtIndexInPerCentBinning(cache.ptBinsByCentrality, cenIdx, item.ptMin, item.ptMax);
    if (ptIdx < 0 || static_cast<size_t>(ptIdx) >= cache.signalLossByCentrality[static_cast<size_t>(cenIdx)].size()) return BinValueWithError{1.0, 0.0};
    return cache.signalLossByCentrality[static_cast<size_t>(cenIdx)][static_cast<size_t>(ptIdx)];
}

inline PtCtAcceptanceCache BuildPtCtAcceptanceCache(const BinningCorrectionConfig &cfg,
                                                    const std::string &mcFileForAcceptance) {
    PtCtAcceptanceCache cache;
    if (cfg.ptBins.size() < 2 || cfg.ctBinsByPt.size() != cfg.ptBins.size() - 1 || mcFileForAcceptance.empty()) {
        return cache;
    }

    auto mcChain = MakeChainFromFileForCorrection(mcFileForAcceptance, cfg.mcTree);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode mcNode(mcReady);

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        cfg.ptBins,
        std::vector<double>{},
        cfg.ctBinsByPt,
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.mcEfficiencySelection,
        std::vector<std::string>{},
        std::vector<std::vector<std::string>>{},
        cfg.mcAcceptanceRequireTwoBody);

    const auto &hVec = (cfg.isMatter == "matter") ? accRes.acc_ct_per_pt_matter
                      : (cfg.isMatter == "antimatter") ? accRes.acc_ct_per_pt_antimatter
                      : accRes.acc_ct_per_pt;
    const auto &hGenVec = (cfg.isMatter == "matter") ? accRes.evsel_ct_per_pt_matter
                         : (cfg.isMatter == "antimatter") ? accRes.evsel_ct_per_pt_antimatter
                         : accRes.evsel_ct_per_pt;

    cache.ptBins = cfg.ptBins;
    cache.valuesByPt.assign(cfg.ctBinsByPt.size(), std::vector<double>{});
    cache.genCountsByPt.assign(cfg.ctBinsByPt.size(), std::vector<double>{});
    for (size_t ipt = 0; ipt < cfg.ctBinsByPt.size(); ++ipt) {
        const auto &ctEdges = cfg.ctBinsByPt[ipt];
        if (ctEdges.size() < 2) continue;
        cache.valuesByPt[ipt].assign(ctEdges.size() - 1, 1.0);
        cache.genCountsByPt[ipt].assign(ctEdges.size() - 1, 0.0);
        if (ipt < hVec.size() && hVec[ipt]) {
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                double v = hVec[ipt]->GetBinContent(static_cast<int>(ict + 1));
                if (v <= 0.0) v = 1.0;
                cache.valuesByPt[ipt][ict] = v;
            }
        }
        if (ipt < hGenVec.size() && hGenVec[ipt]) {
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                cache.genCountsByPt[ipt][ict] = hGenVec[ipt]->GetBinContent(static_cast<int>(ict + 1));
            }
        }
    }
    cache.ready = true;
    accRes.Clear();
    return cache;
}

inline std::vector<double> NormalizedWeights(std::vector<double> weights, size_t n) {
    if (n == 0) return {};
    if (weights.size() != n) weights.assign(n, 1.0);
    double sum = 0.0;
    for (double &w : weights) {
        if (!std::isfinite(w) || w < 0.0) w = 0.0;
        sum += w;
    }
    if (sum <= 0.0) {
        weights.assign(n, 1.0 / static_cast<double>(n));
        return weights;
    }
    for (double &w : weights) w /= sum;
    return weights;
}

inline std::vector<double> NormalizedSingleBinCountWeights(const std::vector<double> &counts, size_t n) {
    std::vector<double> weights(n, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < n && i < counts.size(); ++i) {
        const double c = counts[i];
        if (std::isfinite(c) && c > 0.0) {
            weights[i] = c;
            sum += c;
        }
    }
    if (sum <= 0.0) {
        weights.assign(n, n > 0 ? 1.0 / static_cast<double>(n) : 0.0);
        return weights;
    }
    for (double &w : weights) w /= sum;
    return weights;
}

inline PtCtAcceptanceCache BuildWeightedPtCtAcceptanceCache(const BinningCorrectionConfig &cfg) {
    PtCtAcceptanceCache out;
    if (cfg.mcFilesForAcceptance.empty()) return out;

    std::vector<PtCtAcceptanceCache> periodCaches;
    periodCaches.reserve(cfg.mcFilesForAcceptance.size());
    bool initialized = false;
    for (size_t imc = 0; imc < cfg.mcFilesForAcceptance.size(); ++imc) {
        auto one = BuildPtCtAcceptanceCache(cfg, cfg.mcFilesForAcceptance[imc]);
        if (!one.ready) continue;
        if (!initialized) {
            out.ptBins = one.ptBins;
            out.valuesByPt.assign(one.valuesByPt.size(), std::vector<double>{});
            out.genCountsByPt.assign(one.valuesByPt.size(), std::vector<double>{});
            for (size_t ip = 0; ip < one.valuesByPt.size(); ++ip) {
                out.valuesByPt[ip].assign(one.valuesByPt[ip].size(), 0.0);
                out.genCountsByPt[ip].assign(one.valuesByPt[ip].size(), 0.0);
            }
            initialized = true;
        }
        periodCaches.push_back(std::move(one));
    }
    if (!initialized) return out;

    for (size_t ip = 0; ip < out.valuesByPt.size(); ++ip) {
        for (size_t ib = 0; ib < out.valuesByPt[ip].size(); ++ib) {
            std::vector<double> counts(periodCaches.size(), 0.0);
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ip < periodCaches[imc].genCountsByPt.size() &&
                    ib < periodCaches[imc].genCountsByPt[ip].size()) {
                    counts[imc] = periodCaches[imc].genCountsByPt[ip][ib];
                }
            }
            const auto weights = NormalizedSingleBinCountWeights(counts, periodCaches.size());
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ip >= periodCaches[imc].valuesByPt.size() ||
                    ib >= periodCaches[imc].valuesByPt[ip].size()) {
                    continue;
                }
                const double v = periodCaches[imc].valuesByPt[ip][ib] > 0.0 ? periodCaches[imc].valuesByPt[ip][ib] : 1.0;
                out.valuesByPt[ip][ib] += weights[imc] * v;
                out.genCountsByPt[ip][ib] += counts[imc];
            }
            if (out.valuesByPt[ip][ib] <= 0.0) out.valuesByPt[ip][ib] = 1.0;
        }
    }
    out.ready = true;
    return out;
}

inline RadCtAcceptanceCache BuildRadCtAcceptanceCache(const BinningCorrectionConfig &cfg,
                                                      const std::string &mcFileForAcceptance) {
    RadCtAcceptanceCache cache;
    if (cfg.radBins.size() < 2 || cfg.ctBinsByRad.size() != cfg.radBins.size() - 1 || mcFileForAcceptance.empty()) {
        return cache;
    }

    auto mcChain = MakeChainFromFileForCorrection(mcFileForAcceptance, cfg.mcTree);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode mcNode(mcReady);

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        cfg.radBins,
        std::vector<double>{},
        cfg.ctBinsByRad,
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.mcEfficiencySelection,
        std::vector<std::string>{},
        std::vector<std::vector<std::string>>{},
        cfg.mcAcceptanceRequireTwoBody,
        "fCentralityFT0C",
        "fIsSurvEvSel",
        "fIsReco",
        "fGenDecRad",
        "fGenCt",
        "fGenPt",
        cfg.mcAcceptanceConstrainDenominatorOuterBin);

    const auto &hVec = (cfg.isMatter == "matter") ? accRes.acc_ct_per_pt_matter
                      : (cfg.isMatter == "antimatter") ? accRes.acc_ct_per_pt_antimatter
                      : accRes.acc_ct_per_pt;
    const auto &hGenVec = (cfg.isMatter == "matter") ? accRes.evsel_ct_per_pt_matter
                         : (cfg.isMatter == "antimatter") ? accRes.evsel_ct_per_pt_antimatter
                         : accRes.evsel_ct_per_pt;

    cache.radBins = cfg.radBins;
    cache.valuesByRad.assign(cfg.ctBinsByRad.size(), std::vector<double>{});
    cache.genCountsByRad.assign(cfg.ctBinsByRad.size(), std::vector<double>{});
    for (size_t ir = 0; ir < cfg.ctBinsByRad.size(); ++ir) {
        const auto &ctEdges = cfg.ctBinsByRad[ir];
        if (ctEdges.size() < 2) continue;
        cache.valuesByRad[ir].assign(ctEdges.size() - 1, 1.0);
        cache.genCountsByRad[ir].assign(ctEdges.size() - 1, 0.0);
        if (ir < hVec.size() && hVec[ir]) {
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                double v = hVec[ir]->GetBinContent(static_cast<int>(ict + 1));
                if (v <= 0.0) v = 1.0;
                cache.valuesByRad[ir][ict] = v;
            }
        }
        if (ir < hGenVec.size() && hGenVec[ir]) {
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                cache.genCountsByRad[ir][ict] = hGenVec[ir]->GetBinContent(static_cast<int>(ict + 1));
            }
        }
    }
    cache.ready = true;
    accRes.Clear();
    return cache;
}

inline RadCtAcceptanceCache BuildWeightedRadCtAcceptanceCache(const BinningCorrectionConfig &cfg) {
    RadCtAcceptanceCache out;
    if (cfg.mcFilesForAcceptance.empty()) return out;

    std::vector<RadCtAcceptanceCache> periodCaches;
    periodCaches.reserve(cfg.mcFilesForAcceptance.size());
    bool initialized = false;
    for (size_t imc = 0; imc < cfg.mcFilesForAcceptance.size(); ++imc) {
        auto one = BuildRadCtAcceptanceCache(cfg, cfg.mcFilesForAcceptance[imc]);
        if (!one.ready) continue;
        if (!initialized) {
            out.radBins = one.radBins;
            out.valuesByRad.assign(one.valuesByRad.size(), std::vector<double>{});
            out.genCountsByRad.assign(one.valuesByRad.size(), std::vector<double>{});
            for (size_t ir = 0; ir < one.valuesByRad.size(); ++ir) {
                out.valuesByRad[ir].assign(one.valuesByRad[ir].size(), 0.0);
                out.genCountsByRad[ir].assign(one.valuesByRad[ir].size(), 0.0);
            }
            initialized = true;
        }
        periodCaches.push_back(std::move(one));
    }
    if (!initialized) return out;

    for (size_t ir = 0; ir < out.valuesByRad.size(); ++ir) {
        for (size_t ib = 0; ib < out.valuesByRad[ir].size(); ++ib) {
            std::vector<double> counts(periodCaches.size(), 0.0);
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ir < periodCaches[imc].genCountsByRad.size() &&
                    ib < periodCaches[imc].genCountsByRad[ir].size()) {
                    counts[imc] = periodCaches[imc].genCountsByRad[ir][ib];
                }
            }
            const auto weights = NormalizedSingleBinCountWeights(counts, periodCaches.size());
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ir >= periodCaches[imc].valuesByRad.size() ||
                    ib >= periodCaches[imc].valuesByRad[ir].size()) {
                    continue;
                }
                const double v = periodCaches[imc].valuesByRad[ir][ib] > 0.0 ? periodCaches[imc].valuesByRad[ir][ib] : 1.0;
                out.valuesByRad[ir][ib] += weights[imc] * v;
                out.genCountsByRad[ir][ib] += counts[imc];
            }
            if (out.valuesByRad[ir][ib] <= 0.0) out.valuesByRad[ir][ib] = 1.0;
        }
    }
    out.ready = true;
    return out;
}

inline PtCtAbsorptionCache BuildPtCtAbsorptionCache(const BinningCorrectionConfig &cfg,
                                                    const std::string &absorptionFile) {
    PtCtAbsorptionCache cache;
    if (cfg.ptBins.size() < 2 || cfg.ctBinsByPt.size() != cfg.ptBins.size() - 1 || absorptionFile.empty()) {
        return cache;
    }

    auto chain = MakeChainFromFileForCorrection(absorptionFile, cfg.absorptionTree);
    ROOT::RDataFrame rdfBase(*chain);
    auto calc = Absorption::PtAbsorptionCalculator(&rdfBase, cfg.ptBins, cfg.ctBinsByPt, cfg.originalCtaoAbsorption);
    calc.Calculate();

    const std::string key = (cfg.isMatter == "matter") ? "matter"
                          : (cfg.isMatter == "antimatter") ? "antimatter"
                          : "both";
    const auto eff = calc.GetAbsorptionEfficiency(key);

    cache.ptBins = cfg.ptBins;
    cache.valuesByPt = eff.first;
    for (auto &row : cache.valuesByPt) {
        for (double &v : row) {
            if (v <= 0.0) v = 1.0;
        }
    }
    cache.ready = true;
    return cache;
}

inline RadCtAbsorptionCache BuildRadCtAbsorptionCache(const BinningCorrectionConfig &cfg,
                                                      const std::string &absorptionFile) {
    RadCtAbsorptionCache cache;
    if (cfg.radBins.size() < 2 || cfg.ctBinsByRad.size() != cfg.radBins.size() - 1 || absorptionFile.empty()) {
        return cache;
    }

    auto chain = MakeChainFromFileForCorrection(absorptionFile, cfg.absorptionTree);
    ROOT::RDataFrame rdfBase(*chain);
    auto calc = Absorption::RadCtAbsorptionCalculator(&rdfBase, cfg.radBins, cfg.ctBinsByRad, cfg.originalCtaoAbsorption);
    calc.Calculate();

    const std::string key = (cfg.isMatter == "matter") ? "matter"
                          : (cfg.isMatter == "antimatter") ? "antimatter"
                          : "both";
    const auto eff = calc.GetAbsorptionEfficiency(key);

    cache.radBins = cfg.radBins;
    cache.valuesByRad = eff.first;
    for (auto &row : cache.valuesByRad) {
        for (double &v : row) {
            if (v <= 0.0) v = 1.0;
        }
    }
    cache.ready = true;
    return cache;
}

inline SpectrumAcceptanceCache BuildSpectrumAcceptanceCache(const BinningCorrectionConfig &cfg,
                                                            const std::string &mcFileForAcceptance) {
    SpectrumAcceptanceCache cache;
    if (!(cfg.mode == "bdt_spectrum" || cfg.mode == "topology_spectrum")) return cache;
    if (mcFileForAcceptance.empty()) return cache;
    if (cfg.cenBins.size() < 2) return cache;
    if (cfg.ptBinsByCentrality.size() != cfg.cenBins.size() - 1) return cache;

    auto mcChain = MakeChainFromFileForCorrection(mcFileForAcceptance, cfg.mcTree);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode mcNode(mcReady);

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        std::vector<double>{},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.cenBins,
        cfg.ptBinsByCentrality,
        cfg.mcEfficiencySelection,
        std::vector<std::string>{},
        cfg.mode == "topology_spectrum" ? cfg.topologySelectionsByCentrality
                                        : std::vector<std::vector<std::string>>{},
        cfg.mcAcceptanceRequireTwoBody);

    const auto &hVec = (cfg.isMatter == "matter") ? accRes.acc_pt_per_cent_matter
                      : (cfg.isMatter == "antimatter") ? accRes.acc_pt_per_cent_antimatter
                      : accRes.acc_pt_per_cent;
    const auto &hGenVec = (cfg.isMatter == "matter") ? accRes.evsel_pt_per_cent_matter
                         : (cfg.isMatter == "antimatter") ? accRes.evsel_pt_per_cent_antimatter
                         : accRes.evsel_pt_per_cent;

    cache.cenBins = cfg.cenBins;
    cache.ptBinsByCentrality = cfg.ptBinsByCentrality;
    cache.valuesByCentrality.assign(cfg.ptBinsByCentrality.size(), std::vector<BinValueWithError>{});
    cache.genCountsByCentrality.assign(cfg.ptBinsByCentrality.size(), std::vector<double>{});

    for (size_t ic = 0; ic < cfg.ptBinsByCentrality.size(); ++ic) {
        const auto &ptEdges = cfg.ptBinsByCentrality[ic];
        if (ptEdges.size() < 2) continue;
        cache.valuesByCentrality[ic].assign(ptEdges.size() - 1, BinValueWithError{});
        cache.genCountsByCentrality[ic].assign(ptEdges.size() - 1, 0.0);
        if (ic < hVec.size() && hVec[ic]) {
            for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
                const int bin = static_cast<int>(ip + 1);
                double v = hVec[ic]->GetBinContent(bin);
                double e = hVec[ic]->GetBinError(bin);
                if (v <= 0.0) v = 1.0;
                cache.valuesByCentrality[ic][ip] = BinValueWithError{v, e};
            }
        }
        if (ic < hGenVec.size() && hGenVec[ic]) {
            for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
                cache.genCountsByCentrality[ic][ip] = hGenVec[ic]->GetBinContent(static_cast<int>(ip + 1));
            }
        }
    }

    if (cfg.useSpectrumMcCentralityOverrides) {
        for (size_t ic = 0; ic < cfg.ptBinsByCentrality.size() && ic + 1 < cfg.cenBins.size(); ++ic) {
            const auto *ov = FindSpectrumMcCentralityOverride(cfg, cfg.cenBins[ic], cfg.cenBins[ic + 1]);
            if (!ov) continue;

            const auto &ptEdges = cfg.ptBinsByCentrality[ic];
            if (ptEdges.size() < 2) continue;

            std::vector<std::vector<std::string>> topoForTarget;
            if (cfg.mode == "topology_spectrum") {
                topoForTarget.resize(1);
                if (ic < cfg.topologySelectionsByCentrality.size()) {
                    topoForTarget.front() = cfg.topologySelectionsByCentrality[ic];
                }
                if (topoForTarget.front().size() != ptEdges.size() - 1) {
                    topoForTarget.front().assign(ptEdges.size() - 1, std::string{});
                }
            }

            auto overrideAcc = AcceptanceHelper::ComputeAcceptanceFlexible(
                mcNode,
                std::vector<double>{},
                std::vector<double>{},
                std::vector<std::vector<double>>{},
                std::vector<double>{ov->sourceMin, ov->sourceMax},
                std::vector<std::vector<double>>{ptEdges},
                cfg.mcEfficiencySelection,
                std::vector<std::string>{},
                cfg.mode == "topology_spectrum" ? topoForTarget
                                                : std::vector<std::vector<std::string>>{},
                cfg.mcAcceptanceRequireTwoBody);

            TH1D *src = (cfg.isMatter == "matter")
                            ? (!overrideAcc.acc_pt_per_cent_matter.empty() ? overrideAcc.acc_pt_per_cent_matter.front() : nullptr)
                        : (cfg.isMatter == "antimatter")
                            ? (!overrideAcc.acc_pt_per_cent_antimatter.empty() ? overrideAcc.acc_pt_per_cent_antimatter.front() : nullptr)
                            : (!overrideAcc.acc_pt_per_cent.empty() ? overrideAcc.acc_pt_per_cent.front() : nullptr);
            TH1D *srcGen = (cfg.isMatter == "matter")
                               ? (!overrideAcc.evsel_pt_per_cent_matter.empty() ? overrideAcc.evsel_pt_per_cent_matter.front() : nullptr)
                           : (cfg.isMatter == "antimatter")
                               ? (!overrideAcc.evsel_pt_per_cent_antimatter.empty() ? overrideAcc.evsel_pt_per_cent_antimatter.front() : nullptr)
                               : (!overrideAcc.evsel_pt_per_cent.empty() ? overrideAcc.evsel_pt_per_cent.front() : nullptr);
            if (src) {
                cache.valuesByCentrality[ic].assign(ptEdges.size() - 1, BinValueWithError{});
                cache.genCountsByCentrality[ic].assign(ptEdges.size() - 1, 0.0);
                for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
                    const int bin = static_cast<int>(ip + 1);
                    double v = src->GetBinContent(bin);
                    double e = src->GetBinError(bin);
                    if (v <= 0.0) v = 1.0;
                    cache.valuesByCentrality[ic][ip] = BinValueWithError{v, e};
                    if (srcGen) {
                        cache.genCountsByCentrality[ic][ip] = srcGen->GetBinContent(bin);
                    }
                }
                Info("BinningCorrectionHelper",
                     "Spectrum MC acceptance for target %.0f-%.0f%% uses MC centrality %.0f-%.0f%%",
                     cfg.cenBins[ic], cfg.cenBins[ic + 1], ov->sourceMin, ov->sourceMax);
            }
            overrideAcc.Clear();
        }
    }

    cache.ready = true;
    accRes.Clear();
    return cache;
}

inline SpectrumAcceptanceCache BuildWeightedSpectrumAcceptanceCache(const BinningCorrectionConfig &cfg) {
    SpectrumAcceptanceCache out;
    if (cfg.mcFilesForAcceptance.empty()) return out;

    std::vector<SpectrumAcceptanceCache> periodCaches;
    periodCaches.reserve(cfg.mcFilesForAcceptance.size());
    bool initialized = false;
    for (size_t imc = 0; imc < cfg.mcFilesForAcceptance.size(); ++imc) {
        auto one = BuildSpectrumAcceptanceCache(cfg, cfg.mcFilesForAcceptance[imc]);
        if (!one.ready) continue;
        if (!initialized) {
            out.cenBins = one.cenBins;
            out.ptBinsByCentrality = one.ptBinsByCentrality;
            out.valuesByCentrality.assign(one.valuesByCentrality.size(), std::vector<BinValueWithError>{});
            out.genCountsByCentrality.assign(one.valuesByCentrality.size(), std::vector<double>{});
            for (size_t ic = 0; ic < one.valuesByCentrality.size(); ++ic) {
                out.valuesByCentrality[ic].assign(one.valuesByCentrality[ic].size(), BinValueWithError{0.0, 0.0});
                out.genCountsByCentrality[ic].assign(one.valuesByCentrality[ic].size(), 0.0);
            }
            initialized = true;
        }
        periodCaches.push_back(std::move(one));
    }
    if (!initialized) return out;

    for (size_t ic = 0; ic < out.valuesByCentrality.size(); ++ic) {
        for (size_t ip = 0; ip < out.valuesByCentrality[ic].size(); ++ip) {
            std::vector<double> counts(periodCaches.size(), 0.0);
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ic < periodCaches[imc].genCountsByCentrality.size() &&
                    ip < periodCaches[imc].genCountsByCentrality[ic].size()) {
                    counts[imc] = periodCaches[imc].genCountsByCentrality[ic][ip];
                }
            }
            const auto weights = NormalizedSingleBinCountWeights(counts, periodCaches.size());
            for (size_t imc = 0; imc < periodCaches.size(); ++imc) {
                if (ic >= periodCaches[imc].valuesByCentrality.size() ||
                    ip >= periodCaches[imc].valuesByCentrality[ic].size()) {
                    continue;
                }
                const double v = periodCaches[imc].valuesByCentrality[ic][ip].value > 0.0
                                     ? periodCaches[imc].valuesByCentrality[ic][ip].value
                                     : 1.0;
                const double e = periodCaches[imc].valuesByCentrality[ic][ip].error;
                out.valuesByCentrality[ic][ip].value += weights[imc] * v;
                out.valuesByCentrality[ic][ip].error =
                    std::hypot(out.valuesByCentrality[ic][ip].error, weights[imc] * e);
                out.genCountsByCentrality[ic][ip] += counts[imc];
            }
            if (out.valuesByCentrality[ic][ip].value <= 0.0) {
                out.valuesByCentrality[ic][ip].value = 1.0;
            }
        }
    }
    out.ready = true;
    return out;
}

inline std::vector<double> ComputeGeneratedCountsPerBinForAcceptance(const BinningCorrectionConfig &cfg,
                                                                     const std::vector<BinPlanItem> &items,
                                                                     const std::vector<double> &edges,
                                                                     const std::string &mcFileForAcceptance) {
    std::vector<double> out(items.size(), 0.0);
    if (edges.size() < 2 || items.empty() || mcFileForAcceptance.empty()) return out;

    auto mcChain = MakeChainFromFileForCorrection(mcFileForAcceptance, cfg.mcTree);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode mcNode(mcReady);

    auto fillFromHist = [&](TH1D *src) {
        if (!src) return;
        for (size_t i = 0; i < items.size(); ++i) {
            out[i] = src->GetBinContent(static_cast<int>(i + 1));
        }
    };

    if (cfg.mode == "bdt_spectrum") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody);
        TH1D *src = (cfg.isMatter == "matter") ? accRes.evsel_pt_matter
                  : (cfg.isMatter == "antimatter") ? accRes.evsel_pt_antimatter
                  : accRes.evsel_pt_both;
        fillFromHist(src);
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "topology_spectrum") {
        std::vector<std::string> topoSel;
        topoSel.reserve(items.size());
        for (const auto &item : items) topoSel.push_back(item.topologySelection);

        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            std::vector<double>{items.front().cenMin, items.front().cenMax},
            std::vector<std::vector<double>>{edges},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{topoSel},
            cfg.mcAcceptanceRequireTwoBody);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.evsel_pt_per_cent_matter.empty() ? accRes.evsel_pt_per_cent_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.evsel_pt_per_cent_antimatter.empty() ? accRes.evsel_pt_per_cent_antimatter.front() : nullptr)
                        : (!accRes.evsel_pt_per_cent.empty() ? accRes.evsel_pt_per_cent.front() : nullptr);
        fillFromHist(src);
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "pt_ct") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{items.front().ptMin, items.front().ptMax},
            std::vector<double>{},
            std::vector<std::vector<double>>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody,
            "fCentralityFT0C",
            "fIsSurvEvSel",
            "fIsReco",
            "fAbsGenPt",
            "fGenCt",
            "fGenPt",
            cfg.mcAcceptanceConstrainDenominatorOuterBin);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.evsel_ct_per_pt_matter.empty() ? accRes.evsel_ct_per_pt_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.evsel_ct_per_pt_antimatter.empty() ? accRes.evsel_ct_per_pt_antimatter.front() : nullptr)
                        : (!accRes.evsel_ct_per_pt.empty() ? accRes.evsel_ct_per_pt.front() : nullptr);
        fillFromHist(src);
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "rad_ct") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{items.front().radMin, items.front().radMax},
            std::vector<double>{},
            std::vector<std::vector<double>>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody,
            "fCentralityFT0C",
            "fIsSurvEvSel",
            "fIsReco",
            "fGenDecRad",
            "fGenCt",
            "fGenPt",
            cfg.mcAcceptanceConstrainDenominatorOuterBin);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.evsel_ct_per_pt_matter.empty() ? accRes.evsel_ct_per_pt_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.evsel_ct_per_pt_antimatter.empty() ? accRes.evsel_ct_per_pt_antimatter.front() : nullptr)
                        : (!accRes.evsel_ct_per_pt.empty() ? accRes.evsel_ct_per_pt.front() : nullptr);
        fillFromHist(src);
        accRes.Clear();
        return out;
    }

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        std::vector<double>{},
        std::vector<double>{edges},
        std::vector<std::vector<double>>{},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.mcEfficiencySelection,
        std::vector<std::string>{},
        std::vector<std::vector<std::string>>{},
        cfg.mcAcceptanceRequireTwoBody);
    TH1D *src = (cfg.isMatter == "matter") ? accRes.evsel_ct_matter
              : (cfg.isMatter == "antimatter") ? accRes.evsel_ct_antimatter
              : accRes.evsel_ct_both;
    fillFromHist(src);
    accRes.Clear();
    return out;
}

inline std::vector<BinValueWithError> ComputeAcceptancePerBinWithErrors(const BinningCorrectionConfig &cfg,
                                                                        const std::vector<BinPlanItem> &items,
                                                                        const std::vector<double> &edges,
                                                                        const std::string &mcFileForAcceptance,
                                                                        const PtCtAcceptanceCache *ptCtCache = nullptr,
                                                                        const RadCtAcceptanceCache *radCtCache = nullptr,
                                                                        const SpectrumAcceptanceCache *spectrumCache = nullptr) {
    std::vector<BinValueWithError> out(items.size(), BinValueWithError{});
    if (edges.size() < 2 || items.empty()) return out;

    if (cfg.mode == "pt_ct" && ptCtCache && ptCtCache->ready && ptCtCache->ptBins.size() >= 2) {
        const int ptIdx = FindPtIndex(ptCtCache->ptBins, items.front().ptMin, items.front().ptMax);
        const bool matchingCtEdges =
            ptIdx >= 0 &&
            static_cast<size_t>(ptIdx) < cfg.ctBinsByPt.size() &&
            SameBinEdges(cfg.ctBinsByPt[static_cast<size_t>(ptIdx)], edges);
        if (matchingCtEdges && static_cast<size_t>(ptIdx) < ptCtCache->valuesByPt.size()) {
            const auto &vals = ptCtCache->valuesByPt[static_cast<size_t>(ptIdx)];
            if (vals.size() == items.size()) {
                for (size_t i = 0; i < items.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }
    }

    if (cfg.mode == "rad_ct" && radCtCache && radCtCache->ready && radCtCache->radBins.size() >= 2) {
        const int radIdx = FindPtIndex(radCtCache->radBins, items.front().radMin, items.front().radMax);
        const bool matchingCtEdges =
            radIdx >= 0 &&
            static_cast<size_t>(radIdx) < cfg.ctBinsByRad.size() &&
            SameBinEdges(cfg.ctBinsByRad[static_cast<size_t>(radIdx)], edges);
        if (matchingCtEdges && static_cast<size_t>(radIdx) < radCtCache->valuesByRad.size()) {
            const auto &vals = radCtCache->valuesByRad[static_cast<size_t>(radIdx)];
            if (vals.size() == items.size()) {
                for (size_t i = 0; i < items.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }
    }

    if ((cfg.mode == "bdt_spectrum" || cfg.mode == "topology_spectrum") &&
        spectrumCache && spectrumCache->ready && !items.empty()) {
        const int cenIdx = FindCentralityIndex(spectrumCache->cenBins, items.front().cenMin, items.front().cenMax);
        if (cenIdx >= 0 && static_cast<size_t>(cenIdx) < spectrumCache->ptBinsByCentrality.size() &&
            static_cast<size_t>(cenIdx) < spectrumCache->valuesByCentrality.size()) {
            const auto &ptEdges = spectrumCache->ptBinsByCentrality[static_cast<size_t>(cenIdx)];
            const auto &vals = spectrumCache->valuesByCentrality[static_cast<size_t>(cenIdx)];
            for (size_t i = 0; i < items.size(); ++i) {
                const int ptIdx = FindPtIndex(ptEdges, items[i].ptMin, items[i].ptMax);
                if (ptIdx >= 0 && static_cast<size_t>(ptIdx) < vals.size()) {
                    out[i] = vals[static_cast<size_t>(ptIdx)];
                    if (out[i].value <= 0.0) out[i].value = 1.0;
                }
            }
            return out;
        }
    }

    if (cfg.mcFilesForAcceptance.size() > 1) {
        for (auto &v : out) v = BinValueWithError{0.0, 0.0};
        BinningCorrectionConfig singleCfg = cfg;
        singleCfg.mcFilesForAcceptance.clear();
        singleCfg.mcFileWeights.clear();
        singleCfg.mcFileWeightsByCentrality.clear();
        std::vector<std::vector<BinValueWithError>> valuesByPeriod;
        std::vector<std::vector<double>> genCountsByPeriod;
        valuesByPeriod.reserve(cfg.mcFilesForAcceptance.size());
        genCountsByPeriod.reserve(cfg.mcFilesForAcceptance.size());
        for (size_t imc = 0; imc < cfg.mcFilesForAcceptance.size(); ++imc) {
            valuesByPeriod.push_back(ComputeAcceptancePerBinWithErrors(singleCfg,
                                                                       items,
                                                                       edges,
                                                                       cfg.mcFilesForAcceptance[imc],
                                                                       nullptr,
                                                                       nullptr,
                                                                       nullptr));
            genCountsByPeriod.push_back(ComputeGeneratedCountsPerBinForAcceptance(singleCfg,
                                                                                  items,
                                                                                  edges,
                                                                                  cfg.mcFilesForAcceptance[imc]));
        }
        for (size_t ib = 0; ib < out.size(); ++ib) {
            std::vector<double> counts(valuesByPeriod.size(), 0.0);
            for (size_t imc = 0; imc < genCountsByPeriod.size(); ++imc) {
                if (ib < genCountsByPeriod[imc].size()) counts[imc] = genCountsByPeriod[imc][ib];
            }
            const auto weights = NormalizedSingleBinCountWeights(counts, valuesByPeriod.size());
            for (size_t imc = 0; imc < valuesByPeriod.size(); ++imc) {
                if (ib >= valuesByPeriod[imc].size()) continue;
                const double v = valuesByPeriod[imc][ib].value > 0.0 ? valuesByPeriod[imc][ib].value : 1.0;
                out[ib].value += weights[imc] * v;
                out[ib].error = std::hypot(out[ib].error, weights[imc] * valuesByPeriod[imc][ib].error);
            }
        }
        for (auto &v : out) {
            if (v.value <= 0.0) v.value = 1.0;
        }
        return out;
    }

    if (mcFileForAcceptance.empty()) return out;

    auto mcChain = MakeChainFromFileForCorrection(mcFileForAcceptance, cfg.mcTree);
    ROOT::RDataFrame rdf(*mcChain);
    auto mcReady = GeneralHelper::CorrectAndConvertRDF(rdf, false, true, false);
    ROOT::RDF::RNode mcNode(mcReady);

    if (cfg.mode == "bdt_spectrum") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody);
        TH1D *src = (cfg.isMatter == "matter") ? accRes.acc_pt_matter
                  : (cfg.isMatter == "antimatter") ? accRes.acc_pt_antimatter
                  : accRes.acc_pt_both;
        if (src) {
            for (size_t i = 0; i < items.size(); ++i) {
                out[i].value = src->GetBinContent(static_cast<int>(i + 1));
                out[i].error = src->GetBinError(static_cast<int>(i + 1));
                if (out[i].value <= 0.0) out[i].value = 1.0;
            }
        }
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "topology_spectrum") {
        std::vector<std::string> topoSel;
        topoSel.reserve(items.size());
        for (const auto &item : items) topoSel.push_back(item.topologySelection);

        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            std::vector<double>{items.front().cenMin, items.front().cenMax},
            std::vector<std::vector<double>>{edges},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{topoSel},
            cfg.mcAcceptanceRequireTwoBody);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.acc_pt_per_cent_matter.empty() ? accRes.acc_pt_per_cent_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.acc_pt_per_cent_antimatter.empty() ? accRes.acc_pt_per_cent_antimatter.front() : nullptr)
                        : (!accRes.acc_pt_per_cent.empty() ? accRes.acc_pt_per_cent.front() : nullptr);
        if (src) {
            for (size_t i = 0; i < items.size(); ++i) {
                out[i].value = src->GetBinContent(static_cast<int>(i + 1));
                out[i].error = src->GetBinError(static_cast<int>(i + 1));
                if (out[i].value <= 0.0) out[i].value = 1.0;
            }
        }
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "pt_ct") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{items.front().ptMin, items.front().ptMax},
            std::vector<double>{},
            std::vector<std::vector<double>>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody,
            "fCentralityFT0C",
            "fIsSurvEvSel",
            "fIsReco",
            "fAbsGenPt",
            "fGenCt",
            "fGenPt",
            cfg.mcAcceptanceConstrainDenominatorOuterBin);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.acc_ct_per_pt_matter.empty() ? accRes.acc_ct_per_pt_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.acc_ct_per_pt_antimatter.empty() ? accRes.acc_ct_per_pt_antimatter.front() : nullptr)
                        : (!accRes.acc_ct_per_pt.empty() ? accRes.acc_ct_per_pt.front() : nullptr);
        if (src) {
            for (size_t i = 0; i < items.size(); ++i) {
                out[i].value = src->GetBinContent(static_cast<int>(i + 1));
                out[i].error = src->GetBinError(static_cast<int>(i + 1));
                if (out[i].value <= 0.0) out[i].value = 1.0;
            }
        }
        accRes.Clear();
        return out;
    }

    if (cfg.mode == "rad_ct") {
        auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
            mcNode,
            std::vector<double>{items.front().radMin, items.front().radMax},
            std::vector<double>{},
            std::vector<std::vector<double>>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            std::vector<std::string>{},
            std::vector<std::vector<std::string>>{},
            cfg.mcAcceptanceRequireTwoBody,
            "fCentralityFT0C",
            "fIsSurvEvSel",
            "fIsReco",
            "fGenDecRad",
            "fGenCt",
            "fGenPt",
            cfg.mcAcceptanceConstrainDenominatorOuterBin);
        TH1D *src = (cfg.isMatter == "matter")
                        ? (!accRes.acc_ct_per_pt_matter.empty() ? accRes.acc_ct_per_pt_matter.front() : nullptr)
                    : (cfg.isMatter == "antimatter")
                        ? (!accRes.acc_ct_per_pt_antimatter.empty() ? accRes.acc_ct_per_pt_antimatter.front() : nullptr)
                        : (!accRes.acc_ct_per_pt.empty() ? accRes.acc_ct_per_pt.front() : nullptr);
        if (src) {
            for (size_t i = 0; i < items.size(); ++i) {
                out[i].value = src->GetBinContent(static_cast<int>(i + 1));
                out[i].error = src->GetBinError(static_cast<int>(i + 1));
                if (out[i].value <= 0.0) out[i].value = 1.0;
            }
        }
        accRes.Clear();
        return out;
    }

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        std::vector<double>{},
        std::vector<double>{edges},
        std::vector<std::vector<double>>{},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.mcEfficiencySelection,
        std::vector<std::string>{},
        std::vector<std::vector<std::string>>{},
        cfg.mcAcceptanceRequireTwoBody);
    TH1D *src = (cfg.isMatter == "matter") ? accRes.acc_ct_matter
              : (cfg.isMatter == "antimatter") ? accRes.acc_ct_antimatter
              : accRes.acc_ct_both;
    if (src) {
        for (size_t i = 0; i < items.size(); ++i) {
            out[i].value = src->GetBinContent(static_cast<int>(i + 1));
            out[i].error = src->GetBinError(static_cast<int>(i + 1));
            if (out[i].value <= 0.0) out[i].value = 1.0;
        }
    }
    accRes.Clear();
    return out;
}

inline std::vector<double> ComputeAcceptancePerBin(const BinningCorrectionConfig &cfg,
                                                   const std::vector<BinPlanItem> &items,
                                                   const std::vector<double> &edges,
                                                   const std::string &mcFileForAcceptance,
                                                   const PtCtAcceptanceCache *ptCtCache = nullptr,
                                                   const RadCtAcceptanceCache *radCtCache = nullptr,
                                                   const SpectrumAcceptanceCache *spectrumCache = nullptr) {
    const auto withErr = ComputeAcceptancePerBinWithErrors(cfg, items, edges, mcFileForAcceptance, ptCtCache, radCtCache, spectrumCache);
    std::vector<double> out;
    out.reserve(withErr.size());
    for (const auto &v : withErr) out.push_back(v.value);
    return out;
}

inline std::vector<BinValueWithError> ComputeAbsorptionPerBinWithErrors(const BinningCorrectionConfig &cfg,
                                                                        const std::vector<BinPlanItem> &items,
                                                                        const std::vector<double> &edges,
                                                                        const std::string &absorptionFileOverride = "",
                                                                        const PtCtAbsorptionCache *ptCtCache = nullptr,
                                                                        const RadCtAbsorptionCache *radCtCache = nullptr) {
    std::vector<BinValueWithError> out(items.size(), BinValueWithError{});
    if (edges.size() < 2 || items.empty()) return out;

    const std::string absoFile = absorptionFileOverride.empty() ? cfg.mcFileForAbsorption : absorptionFileOverride;
    if (absoFile.empty()) return out;

    if (cfg.mode == "pt_ct") {
        if (ptCtCache && ptCtCache->ready && ptCtCache->ptBins.size() >= 2) {
            const int ptIdx = FindPtIndex(ptCtCache->ptBins, items.front().ptMin, items.front().ptMax);
            if (ptIdx >= 0 && static_cast<size_t>(ptIdx) < ptCtCache->valuesByPt.size()) {
                const auto &vals = ptCtCache->valuesByPt[static_cast<size_t>(ptIdx)];
                for (size_t i = 0; i < items.size() && i < vals.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }

        auto cache = BuildPtCtAbsorptionCache(cfg, absoFile);
        if (cache.ready) {
            const int ptIdx = FindPtIndex(cache.ptBins, items.front().ptMin, items.front().ptMax);
            if (ptIdx >= 0 && static_cast<size_t>(ptIdx) < cache.valuesByPt.size()) {
                const auto &vals = cache.valuesByPt[static_cast<size_t>(ptIdx)];
                for (size_t i = 0; i < items.size() && i < vals.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }
        return out;
    }

    if (cfg.mode == "rad_ct") {
        if (radCtCache && radCtCache->ready && radCtCache->radBins.size() >= 2) {
            const int radIdx = FindPtIndex(radCtCache->radBins, items.front().radMin, items.front().radMax);
            if (radIdx >= 0 && static_cast<size_t>(radIdx) < radCtCache->valuesByRad.size()) {
                const auto &vals = radCtCache->valuesByRad[static_cast<size_t>(radIdx)];
                for (size_t i = 0; i < items.size() && i < vals.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }

        auto cache = BuildRadCtAbsorptionCache(cfg, absoFile);
        if (cache.ready) {
            const int radIdx = FindPtIndex(cache.radBins, items.front().radMin, items.front().radMax);
            if (radIdx >= 0 && static_cast<size_t>(radIdx) < cache.valuesByRad.size()) {
                const auto &vals = cache.valuesByRad[static_cast<size_t>(radIdx)];
                for (size_t i = 0; i < items.size() && i < vals.size(); ++i) {
                    out[i].value = vals[i] > 0.0 ? vals[i] : 1.0;
                    out[i].error = 0.0;
                }
                return out;
            }
        }
        return out;
    }

    auto chain = MakeChainFromFileForCorrection(absoFile, cfg.absorptionTree);
    ROOT::RDataFrame rdf(*chain);
    ROOT::RDF::RNode node(rdf);

    if (cfg.mode == "bdt_spectrum" || cfg.mode == "topology_spectrum") {
        Absorption::SpectrumAbsorptionCalculator calc(node, edges, cfg.originalCtaoAbsorption);
        calc.Calculate();
        std::string key = cfg.isMatter;
        if (key != "both" && key != "matter" && key != "antimatter") key = "both";
        const auto &ratioMap = calc.Ratio();
        auto it = ratioMap.find(key);
        if (it != ratioMap.end()) {
            const TH1F &src = it->second;
            for (size_t i = 0; i < items.size(); ++i) {
                out[i].value = src.GetBinContent(static_cast<int>(i + 1));
                out[i].error = src.GetBinError(static_cast<int>(i + 1));
                if (out[i].value <= 0.0) out[i].value = 1.0;
            }
        }
        return out;
    }

    for (size_t i = 0; i < items.size(); ++i) {
        Absorption::CtAbsorptionCalculator calc(node, items[i].ctMin, items[i].ctMax, cfg.originalCtaoAbsorption);
        calc.Calculate();
        const auto &res = calc.Result();
        if (cfg.isMatter == "matter") {
            out[i].value = res.effMatter;
            out[i].error = res.errMatter;
        } else if (cfg.isMatter == "antimatter") {
            out[i].value = res.effAnti;
            out[i].error = res.errAnti;
        } else {
            out[i].value = res.effBoth;
            out[i].error = res.errBoth;
        }
        if (out[i].value <= 0.0) out[i].value = 1.0;
    }
    return out;
}

inline std::vector<double> ComputeAbsorptionPerBin(const BinningCorrectionConfig &cfg,
                                                   const std::vector<BinPlanItem> &items,
                                                   const std::vector<double> &edges,
                                                   const std::string &absorptionFileOverride = "",
                                                   const PtCtAbsorptionCache *ptCtCache = nullptr,
                                                   const RadCtAbsorptionCache *radCtCache = nullptr) {
    const auto withErr = ComputeAbsorptionPerBinWithErrors(cfg, items, edges, absorptionFileOverride, ptCtCache, radCtCache);
    std::vector<double> out;
    out.reserve(withErr.size());
    for (const auto &v : withErr) out.push_back(v.value);
    return out;
}

} // namespace UnifiedAnalysis

#endif // BINNING_CORRECTION_HELPER_H
