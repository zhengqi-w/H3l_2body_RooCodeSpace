#ifndef BINNING_CORRECTION_HELPER_H
#define BINNING_CORRECTION_HELPER_H

#include "../AcceptanceHelper.h"
#include "../AbsorptionHelper.h"
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
    std::string mcTree;
    std::string absorptionTree;
    std::string mcFileForAcceptance;
    std::string mcFileForAbsorption;
    std::string mcEfficiencySelection;
    double originalCtaoAbsorption{7.6};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> ctBinsByPt;
};

struct PtCtAcceptanceCache {
    bool ready{false};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> valuesByPt;
};

struct PtCtAbsorptionCache {
    bool ready{false};
    std::vector<double> ptBins;
    std::vector<std::vector<double>> valuesByPt;
};

struct BinValueWithError {
    double value{1.0};
    double error{0.0};
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
        cfg.mcEfficiencySelection);

    const auto &hVec = (cfg.isMatter == "matter") ? accRes.acc_ct_per_pt_matter
                      : (cfg.isMatter == "antimatter") ? accRes.acc_ct_per_pt_antimatter
                      : accRes.acc_ct_per_pt;

    cache.ptBins = cfg.ptBins;
    cache.valuesByPt.assign(cfg.ctBinsByPt.size(), std::vector<double>{});
    for (size_t ipt = 0; ipt < cfg.ctBinsByPt.size(); ++ipt) {
        const auto &ctEdges = cfg.ctBinsByPt[ipt];
        if (ctEdges.size() < 2) continue;
        cache.valuesByPt[ipt].assign(ctEdges.size() - 1, 1.0);
        if (ipt < hVec.size() && hVec[ipt]) {
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                double v = hVec[ipt]->GetBinContent(static_cast<int>(ict + 1));
                if (v <= 0.0) v = 1.0;
                cache.valuesByPt[ipt][ict] = v;
            }
        }
    }
    cache.ready = true;
    accRes.Clear();
    return cache;
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

inline std::vector<BinValueWithError> ComputeAcceptancePerBinWithErrors(const BinningCorrectionConfig &cfg,
                                                                        const std::vector<BinPlanItem> &items,
                                                                        const std::vector<double> &edges,
                                                                        const std::string &mcFileForAcceptance,
                                                                        const PtCtAcceptanceCache *ptCtCache = nullptr) {
    std::vector<BinValueWithError> out(items.size(), BinValueWithError{});
    if (edges.size() < 2 || items.empty() || mcFileForAcceptance.empty()) return out;

    if (cfg.mode == "pt_ct" && ptCtCache && ptCtCache->ready && ptCtCache->ptBins.size() >= 2) {
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
            cfg.mcEfficiencySelection);
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
            std::vector<double>{edges},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            std::vector<double>{},
            std::vector<std::vector<double>>{},
            cfg.mcEfficiencySelection,
            topoSel);
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

    auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
        mcNode,
        std::vector<double>{},
        std::vector<double>{edges},
        std::vector<std::vector<double>>{},
        std::vector<double>{},
        std::vector<std::vector<double>>{},
        cfg.mcEfficiencySelection);
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
                                                   const PtCtAcceptanceCache *ptCtCache = nullptr) {
    const auto withErr = ComputeAcceptancePerBinWithErrors(cfg, items, edges, mcFileForAcceptance, ptCtCache);
    std::vector<double> out;
    out.reserve(withErr.size());
    for (const auto &v : withErr) out.push_back(v.value);
    return out;
}

inline std::vector<BinValueWithError> ComputeAbsorptionPerBinWithErrors(const BinningCorrectionConfig &cfg,
                                                                        const std::vector<BinPlanItem> &items,
                                                                        const std::vector<double> &edges,
                                                                        const std::string &absorptionFileOverride = "",
                                                                        const PtCtAbsorptionCache *ptCtCache = nullptr) {
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
                                                   const PtCtAbsorptionCache *ptCtCache = nullptr) {
    const auto withErr = ComputeAbsorptionPerBinWithErrors(cfg, items, edges, absorptionFileOverride, ptCtCache);
    std::vector<double> out;
    out.reserve(withErr.size());
    for (const auto &v : withErr) out.push_back(v.value);
    return out;
}

} // namespace UnifiedAnalysis

#endif // BINNING_CORRECTION_HELPER_H
