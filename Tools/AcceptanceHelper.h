#ifndef ACCEPTANCE_HELPER_H
#define ACCEPTANCE_HELPER_H

// Helper to compute acceptance histograms from an MC RDataFrame.
// Header-only, optimized to book all histograms first and trigger the
// event loop only once at the end, minimizing memory and CPU overhead.

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TH1D.h>
#include <TError.h>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <utility>

namespace AcceptanceHelper {

// Result container supporting four scenarios and matter/antimatter splits:
// 1) acc vs pt (1D pt bins)
// 2) acc vs ct (1D ct bins)
// 3) acc vs ct per-pt-bin (1D pt bins + per-pt ct bins)
// 4) acc vs pt per-centrality-bin (1D centrality bins + per-cent pt bins)
struct AcceptanceResult {
    // scenario 1 (both + matter + antimatter)
    TH1D* evsel_pt_both = nullptr;
    TH1D* reco_pt_both  = nullptr;
    TH1D* acc_pt_both   = nullptr;

    TH1D* evsel_pt_matter = nullptr;
    TH1D* reco_pt_matter  = nullptr;
    TH1D* acc_pt_matter   = nullptr;

    TH1D* evsel_pt_antimatter = nullptr;
    TH1D* reco_pt_antimatter  = nullptr;
    TH1D* acc_pt_antimatter   = nullptr;

    // scenario 2 (both + matter + antimatter)
    TH1D* evsel_ct_both = nullptr;
    TH1D* reco_ct_both  = nullptr;
    TH1D* acc_ct_both   = nullptr;

    TH1D* evsel_ct_matter = nullptr;
    TH1D* reco_ct_matter  = nullptr;
    TH1D* acc_ct_matter   = nullptr;

    TH1D* evsel_ct_antimatter = nullptr;
    TH1D* reco_ct_antimatter  = nullptr;
    TH1D* acc_ct_antimatter   = nullptr;

    // scenario 3 (per-pt: both + matter + antimatter)
    std::vector<TH1D*> evsel_ct_per_pt;           // size = nPtBins
    std::vector<TH1D*> reco_ct_per_pt;            // size = nPtBins
    std::vector<TH1D*> acc_ct_per_pt;             // size = nPtBins
    std::vector<TH1D*> evsel_ct_per_pt_matter;    // size = nPtBins
    std::vector<TH1D*> reco_ct_per_pt_matter;     // size = nPtBins
    std::vector<TH1D*> acc_ct_per_pt_matter;      // size = nPtBins
    std::vector<TH1D*> evsel_ct_per_pt_antimatter;// size = nPtBins
    std::vector<TH1D*> reco_ct_per_pt_antimatter; // size = nPtBins
    std::vector<TH1D*> acc_ct_per_pt_antimatter;  // size = nPtBins

    // scenario 4 (per-cent: both + matter + antimatter)
    std::vector<TH1D*> evsel_pt_per_cent;           // size = nCentBins
    std::vector<TH1D*> reco_pt_per_cent;            // size = nCentBins
    std::vector<TH1D*> acc_pt_per_cent;             // size = nCentBins
    std::vector<TH1D*> evsel_pt_per_cent_matter;    // size = nCentBins
    std::vector<TH1D*> reco_pt_per_cent_matter;     // size = nCentBins
    std::vector<TH1D*> acc_pt_per_cent_matter;      // size = nCentBins
    std::vector<TH1D*> evsel_pt_per_cent_antimatter;// size = nCentBins
    std::vector<TH1D*> reco_pt_per_cent_antimatter; // size = nCentBins
    std::vector<TH1D*> acc_pt_per_cent_antimatter;  // size = nCentBins

    void Clear() {
        auto clearHist = [](TH1D *&h) {
            if (h) {
                delete h;
                h = nullptr;
            }
        };
        auto clearVec = [&](std::vector<TH1D*> &vec) {
            for (auto &ptr : vec) {
                clearHist(ptr);
            }
            vec.clear();
        };

        clearHist(evsel_pt_both);
        clearHist(reco_pt_both);
        clearHist(acc_pt_both);
        clearHist(evsel_pt_matter);
        clearHist(reco_pt_matter);
        clearHist(acc_pt_matter);
        clearHist(evsel_pt_antimatter);
        clearHist(reco_pt_antimatter);
        clearHist(acc_pt_antimatter);

        clearHist(evsel_ct_both);
        clearHist(reco_ct_both);
        clearHist(acc_ct_both);
        clearHist(evsel_ct_matter);
        clearHist(reco_ct_matter);
        clearHist(acc_ct_matter);
        clearHist(evsel_ct_antimatter);
        clearHist(reco_ct_antimatter);
        clearHist(acc_ct_antimatter);

        clearVec(evsel_ct_per_pt);
        clearVec(reco_ct_per_pt);
        clearVec(acc_ct_per_pt);
        clearVec(evsel_ct_per_pt_matter);
        clearVec(reco_ct_per_pt_matter);
        clearVec(acc_ct_per_pt_matter);
        clearVec(evsel_ct_per_pt_antimatter);
        clearVec(reco_ct_per_pt_antimatter);
        clearVec(acc_ct_per_pt_antimatter);

        clearVec(evsel_pt_per_cent);
        clearVec(reco_pt_per_cent);
        clearVec(acc_pt_per_cent);
        clearVec(evsel_pt_per_cent_matter);
        clearVec(reco_pt_per_cent_matter);
        clearVec(acc_pt_per_cent_matter);
        clearVec(evsel_pt_per_cent_antimatter);
        clearVec(reco_pt_per_cent_antimatter);
        clearVec(acc_pt_per_cent_antimatter);
    }
};

// Utility: check if RDF has a column (non-const RNode to avoid const member issues in some ROOT versions)
inline bool HasColumn(ROOT::RDF::RNode df, const std::string &col) {
    auto cols = df.GetColumnNames();
    return std::find(cols.begin(), cols.end(), col) != cols.end();
}

inline void EnsureImplicitMT() {
    // ForeachSlot benefits from multiple slots, but enabling IMT after an
    // RDataFrame has been instantiated is not supported. Leave the decision to
    // the caller and simply warn when IMT is disabled.
    if (!ROOT::IsImplicitMTEnabled()) {
        Warning("AcceptanceHelper", "Implicit MT is disabled. Call ROOT::EnableImplicitMT() before creating the RDataFrame for multi-threaded execution.");
    }
}

inline std::unique_ptr<TH1D> MakeHist(const std::string &name,
                                      const std::string &title,
                                      const std::vector<double> &edges) {
    auto h = std::make_unique<TH1D>(name.c_str(), title.c_str(), static_cast<int>(edges.size()) - 1, edges.data());
    h->Sumw2();
    h->SetDirectory(nullptr);
    return h;
}

inline TH1D* BuildAcceptanceHistogram(TH1D* reco, TH1D* evsel, const std::string &name) {
    if (!reco || !evsel)
        return nullptr;
    auto acc = static_cast<TH1D*>(reco->Clone(name.c_str()));
    acc->SetDirectory(nullptr);
    acc->Divide(reco, evsel, 1.0, 1.0, "B");
    return acc;
}
inline int FindBin(const std::vector<double> &edges, double value) {
    if (edges.size() < 2)
        return -1;
    if (value < edges.front() || value >= edges.back())
        return -1;
    auto upper = std::upper_bound(edges.begin(), edges.end(), value);
    int idx = static_cast<int>(upper - edges.begin()) - 1;
    if (idx < 0 || idx >= static_cast<int>(edges.size()) - 1)
        return -1;
    return idx;
}

template <typename SlotType, typename Accessor>
TH1D* MergeSlotHists(const std::vector<std::unique_ptr<SlotType>> &slots,
                     Accessor accessor,
                     const std::string &name) {
    std::unique_ptr<TH1D> merged;
    for (const auto &slot : slots) {
        if (!slot)
            continue;
        TH1D *h = accessor(*slot);
        if (!h)
            continue;
        if (!merged)
            merged.reset(static_cast<TH1D*>(h->Clone(name.c_str())));
        else
            merged->Add(h);
    }
    if (!merged)
        return nullptr;
    merged->SetDirectory(nullptr);
    return merged.release();
}

// Core flexible API. Depending on which binning arguments are provided, it performs:
// 1) ptBins1D only                        -> acc vs pt
// 2) ctBins1D only                        -> acc vs ct
// 3) ptBins1D + ctBinsPerPt (size match)  -> acc vs ct in each pt bin
// 4) centBins1D + ptBinsPerCent (size match) -> acc vs pt in each centrality bin
//
// Columns:
//   genPtCol:     generator-level |pt|
//   genCtCol:     generator-level ct
//   evselCol:     event selection (1 passes). If missing, treated as all pass
//   recoCol:      reconstruction flag (1 passes). If missing, treated as all pass
//   centVar:      centrality variable for scenario 4 (default fCentralityFT0C)
//   genMatterCol: generator-level signed pt (default fGenPt) used to split matter/antimatter
inline AcceptanceResult ComputeAcceptanceFlexible(
    ROOT::RDF::RNode rdf,
    const std::vector<double> &ptBins1D,
    const std::vector<double> &ctBins1D,
    const std::vector<std::vector<double>> &ctBinsPerPt,
    const std::vector<double> &centBins1D,
    const std::vector<std::vector<double>> &ptBinsPerCent,
    const std::string &centVar    = "fCentralityFT0C",
    const std::string &evselCol   = "fIsSurvEvSel",
    const std::string &recoCol    = "fIsReco",
    const std::string &genPtCol   = "fAbsGenPt",
    const std::string &genCtCol   = "fGenCt",
    const std::string &genMatterCol = "fGenPt")
{
    EnsureImplicitMT();
    AcceptanceResult res;

    if (!HasColumn(rdf, genPtCol))
        throw std::runtime_error("ComputeAcceptanceFlexible: missing column '" + genPtCol + "'");
    if (!HasColumn(rdf, genCtCol))
        throw std::runtime_error("ComputeAcceptanceFlexible: missing column '" + genCtCol + "'");
    if (!HasColumn(rdf, genMatterCol))
        throw std::runtime_error("ComputeAcceptanceFlexible: missing column '" + genMatterCol + "'");

    const bool haveEvsel   = HasColumn(rdf, evselCol);
    const bool haveReco    = HasColumn(rdf, recoCol);

    if (!centBins1D.empty() && !ptBinsPerCent.empty() && !HasColumn(rdf, centVar))
        throw std::runtime_error("ComputeAcceptanceFlexible: missing centrality column '" + centVar + "'");

    auto ensure_flag_column = [](ROOT::RDF::RNode node,
                                 bool hasColumn,
                                 const std::string &existingName,
                                 const std::string &dummyName) {
        if (hasColumn)
            return std::make_pair(node, existingName);
        auto next = node.Define(dummyName, []() -> int { return 1; });
        return std::make_pair(next, dummyName);
    };

    auto [df_evsel_ready, evselFlagCol] = ensure_flag_column(rdf, haveEvsel, evselCol, "__acc_evsel_flag");
    auto [df_reco_ready, recoFlagCol] = ensure_flag_column(df_evsel_ready, haveReco, recoCol, "__acc_reco_flag");
    auto df_ready_flags = df_reco_ready;

    auto ensure_int_flag_column = [](ROOT::RDF::RNode node,
                                     const std::string &colName,
                                     const std::string &alias) {
        auto type = node.GetColumnType(colName);
        if (type == "int" || type == "Int_t")
            return std::make_pair(node, colName);
        const std::string expr = "(" + colName + ") ? 1 : 0";
        auto next = node.Define(alias, expr);
        return std::make_pair(next, alias);
    };

    auto [df_flags_evsel_int, evselFlagColInt] = ensure_int_flag_column(df_ready_flags, evselFlagCol, "__acc_evsel_flag_int");
    auto [df_ready_flags_int, recoFlagColInt] = ensure_int_flag_column(df_flags_evsel_int, recoFlagCol, "__acc_reco_flag_int");

    auto ensure_double_column = [](ROOT::RDF::RNode node,
                                   const std::string &colName,
                                   const std::string &alias) {
        auto type = node.GetColumnType(colName);
        if (type == "double" || type == "Double_t")
            return std::make_pair(node, colName);
        const std::string expr = "static_cast<double>(" + colName + ")";
        auto next = node.Define(alias, expr);
        return std::make_pair(next, alias);
    };

    auto [df_with_genPt, genPtColUsed] = ensure_double_column(df_ready_flags_int, genPtCol, "__acc_genPt_double");
    auto [df_with_genCt, genCtColUsed] = ensure_double_column(df_with_genPt, genCtCol, "__acc_genCt_double");
    auto [df_with_genMatter, genMatterColUsed] = ensure_double_column(df_with_genCt, genMatterCol, "__acc_genMatter_double");
    auto df_ready = df_with_genMatter;
    std::string centColUsed = centVar;
    if (!centBins1D.empty() && !ptBinsPerCent.empty()) {
        auto pairCent = ensure_double_column(df_ready, centVar, "__acc_cent_double");
        df_ready = pairCent.first;
        centColUsed = pairCent.second;
    }

    // Scenario 1: acc vs pt (ptBins1D only)
    if (!ptBins1D.empty() && ctBins1D.empty() && ctBinsPerPt.empty() && centBins1D.empty() && ptBinsPerCent.empty()) {
        struct SlotHistPt {
            std::unique_ptr<TH1D> evsel_pt_both;
            std::unique_ptr<TH1D> reco_pt_both;
            std::unique_ptr<TH1D> evsel_pt_matter;
            std::unique_ptr<TH1D> reco_pt_matter;
            std::unique_ptr<TH1D> evsel_pt_antimatter;
            std::unique_ptr<TH1D> reco_pt_antimatter;
        };

        std::vector<std::unique_ptr<SlotHistPt>> slotHists;
        std::mutex slotMutex;

        auto make_slot = [&](unsigned slot) {
            auto sh = std::make_unique<SlotHistPt>();
            const std::string suffix = "_slot" + std::to_string(slot);
            sh->evsel_pt_both = MakeHist("h_evsel_pt_both" + suffix, "evsel pt;pt;counts", ptBins1D);
            sh->reco_pt_both  = MakeHist("h_reco_pt_both"  + suffix, "reco pt;pt;counts", ptBins1D);
            sh->evsel_pt_matter = MakeHist("h_evsel_pt_matter" + suffix, "evsel pt matter;pt;counts", ptBins1D);
            sh->reco_pt_matter  = MakeHist("h_reco_pt_matter"  + suffix, "reco pt matter;pt;counts", ptBins1D);
            sh->evsel_pt_antimatter = MakeHist("h_evsel_pt_antimatter" + suffix, "evsel pt antimatter;pt;counts", ptBins1D);
            sh->reco_pt_antimatter  = MakeHist("h_reco_pt_antimatter"  + suffix, "reco pt antimatter;pt;counts", ptBins1D);
            return sh;
        };

        auto acquire_slot = [&](unsigned slot) -> SlotHistPt& {
            if (slot < slotHists.size() && slotHists[slot])
                return *slotHists[slot];
            std::lock_guard<std::mutex> guard(slotMutex);
            if (slot >= slotHists.size())
                slotHists.resize(slot + 1);
            if (!slotHists[slot])
                slotHists[slot] = make_slot(slot);
            return *slotHists[slot];
        };

        df_ready.ForeachSlot(
            [&](unsigned slot, double genPt, int evselFlag, int recoFlag, double genMatter) {
                auto &slotHist = acquire_slot(slot);
                const bool passEvsel = evselFlag != 0;
                const bool passReco = passEvsel && (recoFlag != 0);
                const bool isMatter = genMatter > 0.0;
                if (passEvsel) {
                    slotHist.evsel_pt_both->Fill(genPt);
                    if (isMatter)
                        slotHist.evsel_pt_matter->Fill(genPt);
                    else
                        slotHist.evsel_pt_antimatter->Fill(genPt);
                }
                if (passReco) {
                    slotHist.reco_pt_both->Fill(genPt);
                    if (isMatter)
                        slotHist.reco_pt_matter->Fill(genPt);
                    else
                        slotHist.reco_pt_antimatter->Fill(genPt);
                }                
            },
            {genPtColUsed, evselFlagColInt, recoFlagColInt, genMatterColUsed});

        res.evsel_pt_both = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.evsel_pt_both.get(); }, "h_evsel_pt_both");
        res.reco_pt_both  = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.reco_pt_both.get(); },  "h_reco_pt_both");
        res.acc_pt_both   = BuildAcceptanceHistogram(res.reco_pt_both, res.evsel_pt_both, "h_acc_pt_both");
        res.evsel_pt_matter = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.evsel_pt_matter.get(); }, "h_evsel_pt_matter");
        res.reco_pt_matter  = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.reco_pt_matter.get(); },  "h_reco_pt_matter");
        res.acc_pt_matter   = BuildAcceptanceHistogram(res.reco_pt_matter, res.evsel_pt_matter, "h_acc_pt_matter");
        res.evsel_pt_antimatter = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.evsel_pt_antimatter.get(); }, "h_evsel_pt_antimatter");
        res.reco_pt_antimatter  = MergeSlotHists(slotHists, [](const SlotHistPt &slot) { return slot.reco_pt_antimatter.get(); },  "h_reco_pt_antimatter");
        res.acc_pt_antimatter   = BuildAcceptanceHistogram(res.reco_pt_antimatter, res.evsel_pt_antimatter, "h_acc_pt_antimatter");
        return res;
    }

    // Scenario 2: acc vs ct (ctBins1D only)
    if (ptBins1D.empty() && !ctBins1D.empty() && ctBinsPerPt.empty() && centBins1D.empty() && ptBinsPerCent.empty()) {
        struct SlotHistCt {
            std::unique_ptr<TH1D> evsel_ct_both;
            std::unique_ptr<TH1D> reco_ct_both;
            std::unique_ptr<TH1D> evsel_ct_matter;
            std::unique_ptr<TH1D> reco_ct_matter;
            std::unique_ptr<TH1D> evsel_ct_antimatter;
            std::unique_ptr<TH1D> reco_ct_antimatter;
        };

        std::vector<std::unique_ptr<SlotHistCt>> slotHists;
        std::mutex slotMutex;

        auto make_slot = [&](unsigned slot) {
            auto sh = std::make_unique<SlotHistCt>();
            const std::string suffix = "_slot" + std::to_string(slot);
            sh->evsel_ct_both = MakeHist("h_evsel_ct_both" + suffix, "evsel ct;ct;counts", ctBins1D);
            sh->reco_ct_both  = MakeHist("h_reco_ct_both"  + suffix, "reco ct;ct;counts", ctBins1D);
            sh->evsel_ct_matter = MakeHist("h_evsel_ct_matter" + suffix, "evsel ct matter;ct;counts", ctBins1D);
            sh->reco_ct_matter  = MakeHist("h_reco_ct_matter"  + suffix, "reco ct matter;ct;counts", ctBins1D);
            sh->evsel_ct_antimatter = MakeHist("h_evsel_ct_antimatter" + suffix, "evsel ct antimatter;ct;counts", ctBins1D);
            sh->reco_ct_antimatter  = MakeHist("h_reco_ct_antimatter"  + suffix, "reco ct antimatter;ct;counts", ctBins1D);
            return sh;
        };

        auto acquire_slot = [&](unsigned slot) -> SlotHistCt& {
            if (slot < slotHists.size() && slotHists[slot])
                return *slotHists[slot];
            std::lock_guard<std::mutex> guard(slotMutex);
            if (slot >= slotHists.size())
                slotHists.resize(slot + 1);
            if (!slotHists[slot])
                slotHists[slot] = make_slot(slot);
            return *slotHists[slot];
        };

        df_ready.ForeachSlot(
            [&](unsigned slot, double genCt, int evselFlag, int recoFlag, double genMatter) {
                auto &slotHist = acquire_slot(slot);
                const bool passEvsel = evselFlag != 0;
                const bool passReco = passEvsel && (recoFlag != 0);
                const bool isMatter = genMatter > 0.0;
                if (passEvsel) {
                    slotHist.evsel_ct_both->Fill(genCt);
                    if (isMatter)
                        slotHist.evsel_ct_matter->Fill(genCt);
                    else
                        slotHist.evsel_ct_antimatter->Fill(genCt);
                }
                if (passReco) {
                    slotHist.reco_ct_both->Fill(genCt);
                    if (isMatter)
                        slotHist.reco_ct_matter->Fill(genCt);
                    else
                        slotHist.reco_ct_antimatter->Fill(genCt);
                }
            },
            {genCtColUsed, evselFlagColInt, recoFlagColInt, genMatterColUsed});

        res.evsel_ct_both = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.evsel_ct_both.get(); }, "h_evsel_ct_both");
        res.reco_ct_both  = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.reco_ct_both.get(); },  "h_reco_ct_both");
        res.acc_ct_both   = BuildAcceptanceHistogram(res.reco_ct_both, res.evsel_ct_both, "h_acc_ct_both");
        res.evsel_ct_matter = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.evsel_ct_matter.get(); }, "h_evsel_ct_matter");
        res.reco_ct_matter  = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.reco_ct_matter.get(); },  "h_reco_ct_matter");
        res.acc_ct_matter   = BuildAcceptanceHistogram(res.reco_ct_matter, res.evsel_ct_matter, "h_acc_ct_matter");
        res.evsel_ct_antimatter = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.evsel_ct_antimatter.get(); }, "h_evsel_ct_antimatter");
        res.reco_ct_antimatter  = MergeSlotHists(slotHists, [](const SlotHistCt &slot) { return slot.reco_ct_antimatter.get(); },  "h_reco_ct_antimatter");
        res.acc_ct_antimatter   = BuildAcceptanceHistogram(res.reco_ct_antimatter, res.evsel_ct_antimatter, "h_acc_ct_antimatter");
        return res;
    }

    // Scenario 3: acc vs ct per pt-bin (ptBins1D + ctBinsPerPt)
    if (!ptBins1D.empty() && !ctBinsPerPt.empty() && ctBinsPerPt.size() == ptBins1D.size() - 1 && ctBins1D.empty() && centBins1D.empty() && ptBinsPerCent.empty()) {
        const int nPt = static_cast<int>(ptBins1D.size()) - 1;
        res.evsel_ct_per_pt.assign(nPt, nullptr);
        res.reco_ct_per_pt.assign(nPt, nullptr);
        res.acc_ct_per_pt.assign(nPt, nullptr);
        res.evsel_ct_per_pt_matter.assign(nPt, nullptr);
        res.reco_ct_per_pt_matter.assign(nPt, nullptr);
        res.acc_ct_per_pt_matter.assign(nPt, nullptr);
        res.evsel_ct_per_pt_antimatter.assign(nPt, nullptr);
        res.reco_ct_per_pt_antimatter.assign(nPt, nullptr);
        res.acc_ct_per_pt_antimatter.assign(nPt, nullptr);

        for (size_t i = 0; i < ctBinsPerPt.size(); ++i)
            if (ctBinsPerPt[i].size() < 2)
                throw std::runtime_error("ComputeAcceptanceFlexible: ctBinsPerPt[" + std::to_string(i) + "] has < 2 edges");

        struct SlotHistCtPerPt {
            std::vector<std::unique_ptr<TH1D>> evsel_ct_both;
            std::vector<std::unique_ptr<TH1D>> reco_ct_both;
            std::vector<std::unique_ptr<TH1D>> evsel_ct_matter;
            std::vector<std::unique_ptr<TH1D>> reco_ct_matter;
            std::vector<std::unique_ptr<TH1D>> evsel_ct_antimatter;
            std::vector<std::unique_ptr<TH1D>> reco_ct_antimatter;
        };

        std::vector<std::unique_ptr<SlotHistCtPerPt>> slotHists;
        std::mutex slotMutex;

        auto make_slot = [&](unsigned slot) {
            auto sh = std::make_unique<SlotHistCtPerPt>();
            sh->evsel_ct_both.resize(nPt);
            sh->reco_ct_both.resize(nPt);
            sh->evsel_ct_matter.resize(nPt);
            sh->reco_ct_matter.resize(nPt);
            sh->evsel_ct_antimatter.resize(nPt);
            sh->reco_ct_antimatter.resize(nPt);
            for (int i = 0; i < nPt; ++i) {
                const auto &edges = ctBinsPerPt[i];
                const std::string suffix = "_ptbin_" + std::to_string(i) + "_slot" + std::to_string(slot);
                sh->evsel_ct_both[i] = MakeHist("h_evsel_ct" + suffix, "evsel ct;ct;counts", edges);
                sh->reco_ct_both[i]  = MakeHist("h_reco_ct"  + suffix, "reco ct;ct;counts", edges);
                sh->evsel_ct_matter[i]     = MakeHist("h_evsel_ct_matter" + suffix, "evsel ct matter;ct;counts", edges);
                sh->reco_ct_matter[i]      = MakeHist("h_reco_ct_matter"  + suffix, "reco ct matter;ct;counts", edges);
                sh->evsel_ct_antimatter[i] = MakeHist("h_evsel_ct_antimatter" + suffix, "evsel ct antimatter;ct;counts", edges);
                sh->reco_ct_antimatter[i]  = MakeHist("h_reco_ct_antimatter"  + suffix, "reco ct antimatter;ct;counts", edges);
            }
            return sh;
        };

        auto acquire_slot = [&](unsigned slot) -> SlotHistCtPerPt& {
            if (slot < slotHists.size() && slotHists[slot])
                return *slotHists[slot];
            std::lock_guard<std::mutex> guard(slotMutex);
            if (slot >= slotHists.size())
                slotHists.resize(slot + 1);
            if (!slotHists[slot])
                slotHists[slot] = make_slot(slot);
            return *slotHists[slot];
        };

        df_ready.ForeachSlot(
            [&](unsigned slot, double genPt, double genCt, int evselFlag, int recoFlag, double genMatter) {
                const int ptIdx = FindBin(ptBins1D, genPt);
                if (ptIdx < 0)
                    return;
                auto &slotHist = acquire_slot(slot);
                const bool passEvsel = evselFlag != 0;
                const bool passReco = recoFlag != 0;
                const bool isMatter = genMatter > 0.0;
                if (passEvsel) {
                    slotHist.evsel_ct_both[ptIdx]->Fill(genCt);
                    if (isMatter)
                        slotHist.evsel_ct_matter[ptIdx]->Fill(genCt);
                    else
                        slotHist.evsel_ct_antimatter[ptIdx]->Fill(genCt);
                }
                if (passReco) {
                    slotHist.reco_ct_both[ptIdx]->Fill(genCt);
                    if (isMatter)
                        slotHist.reco_ct_matter[ptIdx]->Fill(genCt);
                    else
                        slotHist.reco_ct_antimatter[ptIdx]->Fill(genCt);
                }
            },
            {genPtColUsed, genCtColUsed, evselFlagColInt, recoFlagColInt, genMatterColUsed});

        auto merge_per_pt = [&](auto accessor, const std::string &namePrefix, std::vector<TH1D*> &target) {
            for (int i = 0; i < nPt; ++i) {
                std::unique_ptr<TH1D> merged;
                for (const auto &slot : slotHists) {
                    if (!slot)
                        continue;
                    const auto &vec = accessor(*slot);
                    if (i >= static_cast<int>(vec.size()))
                        continue;
                    TH1D *h = vec[i].get();
                    if (!h)
                        continue;
                    if (!merged)
                        merged.reset(static_cast<TH1D*>(h->Clone((namePrefix + std::to_string(i)).c_str())));
                    else
                        merged->Add(h);
                }
                if (merged) {
                    merged->SetDirectory(nullptr);
                    target[i] = merged.release();
                }
            }
        };

        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.evsel_ct_both; }, "h_evsel_ct_ptbin_", res.evsel_ct_per_pt);
        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.reco_ct_both; }, "h_reco_ct_ptbin_", res.reco_ct_per_pt);
        for (int i = 0; i < nPt; ++i)
            res.acc_ct_per_pt[i] = BuildAcceptanceHistogram(res.reco_ct_per_pt[i], res.evsel_ct_per_pt[i], "h_acc_ct_ptbin_" + std::to_string(i));
        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>> & { return slot.evsel_ct_matter; }, "h_evsel_ct_ptbin_matter_", res.evsel_ct_per_pt_matter);
        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>> & { return slot.reco_ct_matter; },  "h_reco_ct_ptbin_matter_",  res.reco_ct_per_pt_matter);
        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>> & { return slot.evsel_ct_antimatter; }, "h_evsel_ct_ptbin_antimatter_", res.evsel_ct_per_pt_antimatter);
        merge_per_pt([](const SlotHistCtPerPt &slot) -> const std::vector<std::unique_ptr<TH1D>> & { return slot.reco_ct_antimatter; },  "h_reco_ct_ptbin_antimatter_",  res.reco_ct_per_pt_antimatter);
        for (int i = 0; i < nPt; ++i) {
            res.acc_ct_per_pt_matter[i] = BuildAcceptanceHistogram(res.reco_ct_per_pt_matter[i], res.evsel_ct_per_pt_matter[i], "h_acc_ct_ptbin_matter_" + std::to_string(i));
            res.acc_ct_per_pt_antimatter[i] = BuildAcceptanceHistogram(res.reco_ct_per_pt_antimatter[i], res.evsel_ct_per_pt_antimatter[i], "h_acc_ct_ptbin_antimatter_" + std::to_string(i));
        }
        return res;
    }

    // Scenario 4: acc vs pt per centrality bin
    if (!centBins1D.empty() && !ptBinsPerCent.empty() && ptBinsPerCent.size() == centBins1D.size() - 1 && ptBins1D.empty() && ctBins1D.empty() && ctBinsPerPt.empty()) {
        const int nCent = static_cast<int>(centBins1D.size()) - 1;
        res.evsel_pt_per_cent.assign(nCent, nullptr);
        res.reco_pt_per_cent.assign(nCent, nullptr);
        res.acc_pt_per_cent.assign(nCent, nullptr);
        res.evsel_pt_per_cent_matter.assign(nCent, nullptr);
        res.reco_pt_per_cent_matter.assign(nCent, nullptr);
        res.acc_pt_per_cent_matter.assign(nCent, nullptr);
        res.evsel_pt_per_cent_antimatter.assign(nCent, nullptr);
        res.reco_pt_per_cent_antimatter.assign(nCent, nullptr);
        res.acc_pt_per_cent_antimatter.assign(nCent, nullptr);

        for (size_t i = 0; i < ptBinsPerCent.size(); ++i)
            if (ptBinsPerCent[i].size() < 2)
                throw std::runtime_error("ComputeAcceptanceFlexible: ptBinsPerCent[" + std::to_string(i) + "] has < 2 edges");

        struct SlotHistPtPerCent {
            std::vector<std::unique_ptr<TH1D>> evsel_pt_both;
            std::vector<std::unique_ptr<TH1D>> reco_pt_both;
            std::vector<std::unique_ptr<TH1D>> evsel_pt_matter;
            std::vector<std::unique_ptr<TH1D>> reco_pt_matter;
            std::vector<std::unique_ptr<TH1D>> evsel_pt_antimatter;
            std::vector<std::unique_ptr<TH1D>> reco_pt_antimatter;
        };

        std::vector<std::unique_ptr<SlotHistPtPerCent>> slotHists;
        std::mutex slotMutex;

        auto make_slot = [&](unsigned slot) {
            auto sh = std::make_unique<SlotHistPtPerCent>();
            sh->evsel_pt_both.resize(nCent);
            sh->reco_pt_both.resize(nCent);
            sh->evsel_pt_matter.resize(nCent);
            sh->reco_pt_matter.resize(nCent);
            sh->evsel_pt_antimatter.resize(nCent);
            sh->reco_pt_antimatter.resize(nCent);
            for (int i = 0; i < nCent; ++i) {
                const auto &edges = ptBinsPerCent[i];
                const std::string suffix = "_centbin_" + std::to_string(i) + "_slot" + std::to_string(slot);
                sh->evsel_pt_both[i] = MakeHist("h_evsel_pt" + suffix, "evsel pt;pt;counts", edges);
                sh->reco_pt_both[i]  = MakeHist("h_reco_pt"  + suffix, "reco pt;pt;counts", edges);
                sh->evsel_pt_matter[i]     = MakeHist("h_evsel_pt_matter" + suffix, "evsel pt matter;pt;counts", edges);
                sh->reco_pt_matter[i]      = MakeHist("h_reco_pt_matter"  + suffix, "reco pt matter;pt;counts", edges);
                sh->evsel_pt_antimatter[i] = MakeHist("h_evsel_pt_antimatter" + suffix, "evsel pt antimatter;pt;counts", edges);
                sh->reco_pt_antimatter[i]  = MakeHist("h_reco_pt_antimatter"  + suffix, "reco pt antimatter;pt;counts", edges);
            }
            return sh;
        };

        auto acquire_slot = [&](unsigned slot) -> SlotHistPtPerCent& {
            if (slot < slotHists.size() && slotHists[slot])
                return *slotHists[slot];
            std::lock_guard<std::mutex> guard(slotMutex);
            if (slot >= slotHists.size())
                slotHists.resize(slot + 1);
            if (!slotHists[slot])
                slotHists[slot] = make_slot(slot);
            return *slotHists[slot];
        };

        df_ready.ForeachSlot(
            [&](unsigned slot, double genPt, double cent, int evselFlag, int recoFlag, double genMatter) {
                const int centIdx = FindBin(centBins1D, cent);
                if (centIdx < 0)
                    return;
                auto &slotHist = acquire_slot(slot);
                const bool passEvsel = evselFlag != 0;
                const bool passReco = recoFlag != 0;
                const bool isMatter = genMatter > 0.0;
                if (passEvsel) {
                    slotHist.evsel_pt_both[centIdx]->Fill(genPt);
                    if (isMatter)
                        slotHist.evsel_pt_matter[centIdx]->Fill(genPt);
                    else
                        slotHist.evsel_pt_antimatter[centIdx]->Fill(genPt);
                }
                if (passReco) {
                    slotHist.reco_pt_both[centIdx]->Fill(genPt);
                    if (isMatter)
                        slotHist.reco_pt_matter[centIdx]->Fill(genPt);
                    else
                        slotHist.reco_pt_antimatter[centIdx]->Fill(genPt);
                }
            },
            {genPtColUsed, centColUsed, evselFlagColInt, recoFlagColInt, genMatterColUsed});

        auto merge_per_cent = [&](auto accessor, const std::string &namePrefix, std::vector<TH1D*> &target) {
            for (int i = 0; i < nCent; ++i) {
                std::unique_ptr<TH1D> merged;
                for (const auto &slot : slotHists) {
                    if (!slot)
                        continue;
                    const auto &vec = accessor(*slot);
                    if (i >= static_cast<int>(vec.size()))
                        continue;
                    TH1D *h = vec[i].get();
                    if (!h)
                        continue;
                    if (!merged)
                        merged.reset(static_cast<TH1D*>(h->Clone((namePrefix + std::to_string(i)).c_str())));
                    else
                        merged->Add(h);
                }
                if (merged) {
                    merged->SetDirectory(nullptr);
                    target[i] = merged.release();
                }
            }
        };

        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.evsel_pt_both; }, "h_evsel_pt_centbin_", res.evsel_pt_per_cent);
        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.reco_pt_both; }, "h_reco_pt_centbin_", res.reco_pt_per_cent);
        for (int i = 0; i < nCent; ++i)
            res.acc_pt_per_cent[i] = BuildAcceptanceHistogram(res.reco_pt_per_cent[i], res.evsel_pt_per_cent[i], "h_acc_pt_centbin_" + std::to_string(i));
        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.evsel_pt_matter; }, "h_evsel_pt_centbin_matter_", res.evsel_pt_per_cent_matter);
        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.reco_pt_matter; },  "h_reco_pt_centbin_matter_",  res.reco_pt_per_cent_matter);
        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.evsel_pt_antimatter; }, "h_evsel_pt_centbin_antimatter_", res.evsel_pt_per_cent_antimatter);
        merge_per_cent([](const SlotHistPtPerCent &slot) -> const std::vector<std::unique_ptr<TH1D>>& { return slot.reco_pt_antimatter; },  "h_reco_pt_centbin_antimatter_",  res.reco_pt_per_cent_antimatter);
        for (int i = 0; i < nCent; ++i) {
            res.acc_pt_per_cent_matter[i] = BuildAcceptanceHistogram(res.reco_pt_per_cent_matter[i], res.evsel_pt_per_cent_matter[i], "h_acc_pt_centbin_matter_" + std::to_string(i));
            res.acc_pt_per_cent_antimatter[i] = BuildAcceptanceHistogram(res.reco_pt_per_cent_antimatter[i], res.evsel_pt_per_cent_antimatter[i], "h_acc_pt_centbin_antimatter_" + std::to_string(i));
        }
        return res;
    }

    throw std::runtime_error("ComputeAcceptanceFlexible: unsupported combination of binning arguments");
}

// Backward-compatible minimal API: when only 1D pt or 1D ct is requested.
inline AcceptanceResult ComputeAcceptance(ROOT::RDataFrame &rdf,
                                          const std::vector<double> &ptBins,
                                          const std::vector<double> &ctBins)
{
    // delegate to flexible version
    return ComputeAcceptanceFlexible(rdf, ptBins, ctBins, {}, {}, {});
}

} // namespace AcceptanceHelper

#endif // ACCEPTANCE_HELPER_H
