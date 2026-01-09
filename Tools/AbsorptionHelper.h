#ifndef ABSORPTION_HELPER_H
#define ABSORPTION_HELPER_H

// Simple C++ helpers to compute absorption quantities from a ROOT TTree.
// This header is intended for use inside ROOT/cling or compiled code.
// Example usage:
//   TFile *f = TFile::Open("file.root");
//   TTree *t = (TTree*)f->Get("absoTree");
//   std::vector<double> ptbins = {2.0,3.0,4.0,5.5,8.0};
//   PtAbsorptionCalculator calc(t, ptbins);
//   calc.Calculate();
//   auto [counts, errs] = calc.GetCountsList("both");

#include <ROOT/RDataFrame.hxx>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1F.h>
#include <TLorentzVector.h>
#include <TRandom.h>
#include <TMath.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

namespace Absorption {

constexpr double HE3_MASS = 2.809230089; // GeV/c^2

// Utility: create TH1F with variable bin edges
inline TH1F* MakeTH1(const std::string &name, const std::string &title, const std::vector<double> &bins) {
    int nb = static_cast<int>(bins.size()) - 1;
    std::unique_ptr<double[]> edges(new double[bins.size()]);
    for (size_t i=0;i<bins.size();++i) edges[i] = bins[i];
    TH1F *h = new TH1F(name.c_str(), title.c_str(), nb, edges.get());
    return h;
}

// PtAbsorptionCalculator: operate on a TTree, fill histograms per pt-bin
class PtAbsorptionCalculator {
public:
    PtAbsorptionCalculator(ROOT::RDataFrame* rdf, const std::vector<double> &ptBins, const std::vector<std::vector<double>> &ctBins, double org_ctao=7.6, std::string histNameSuffix="")
    : fRdfPtr(rdf), fPtBins(ptBins), fCtBins(ctBins), fOrgCtao(org_ctao), fHistNameSuffix(histNameSuffix) {}

    ~PtAbsorptionCalculator() {
        // histos created with new; ownership remains with this object
        fHCounts.clear();
        fHCountsAbsorb.clear();
        fHRatio.clear();
    }

    void Calculate() {
        if (!fRdfPtr) return;
        // init histograms if needed
        if (fHCounts.empty()) initHistos();
        // reset
        for (auto &p : fHCounts) 
            for (auto &h : p.second) h->Reset();
        for (auto &p : fHCountsAbsorb) 
            for (auto &h : p.second) h->Reset();
        for (auto &p : fHRatio) 
            for (auto &h : p.second) h->Reset();

        // RNG for Foreach must be copyable; we create local RNG here
        std::mt19937 rng(static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        std::exponential_distribution<float> expo(1.0 / fOrgCtao);
        fRdfPtr->Foreach([&](float pt, float eta, float phi, float ax, float ay, float az, int pdg){
            std::string mat = (pdg>0) ? "matter" : "antimatter";
            TLorentzVector lv; lv.SetPtEtaPhiM(pt, eta, phi, HE3_MASS);
            float he3p = lv.P();
            float absoL = std::sqrt(ax*ax + ay*ay + az*az);
            float absoCt = (he3p!=0) ? absoL * HE3_MASS / he3p : 1e9;
            float decCt = expo(rng);
            // find pt bin
            for (size_t i=0;i<fPtBins.size()-1;++i) {
                if (pt >= fPtBins[i] && pt < fPtBins[i+1]) {
                    fHCounts[keyBoth][i]->Fill(decCt);
                    fHCounts[mat][i]->Fill(decCt);
                    if (absoCt > decCt) {
                        fHCountsAbsorb[keyBoth][i]->Fill(decCt);
                        fHCountsAbsorb[mat][i]->Fill(decCt);
                    }
                    break;
                }
            }
        }, {"pt","eta","phi","absoX","absoY","absoZ","pdg"});
        // compute ratios
        for (size_t i=0;i<fPtBins.size()-1;++i) {
            for (const auto &key : {keyBoth, keyMat, keyAnti}) {
                TH1F *hCounts = fHCounts[key][i];
                TH1F *hCountsAbsorb = fHCountsAbsorb[key][i];
                TH1F *hRatio = fHRatio[key][i];
                hRatio->Divide(hCountsAbsorb, hCounts, 1.0, 1.0, "B");
            }
        }
    }

    // Accessors: return copies of bin contents as vectors
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> GetCountsList(const std::string &which="both") const {
        std::vector<std::vector<double>> vals, errs;
        if (fHCounts.empty()) return {vals, errs};
        auto it = fHCounts.find(which);
        if (it == fHCounts.end()) return {vals, errs};
        for (size_t i=0;i<fPtBins.size()-1;++i) {
            TH1F *h = it->second[i];
            int nb = h->GetNbinsX();
            std::vector<double> vbin(nb), ebin(nb);
            for (int b=0;b<nb;++b) {
                vbin[b] = h->GetBinContent(b+1);
                ebin[b] = h->GetBinError(b+1);
            }
            vals.push_back(vbin);
            errs.push_back(ebin);
        }
        return {vals, errs};
    }

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> GetAbsorbCountsList(const std::string &which="both") const {
        std::vector<std::vector<double>> vals, errs;
        if (fHCountsAbsorb.empty()) return {vals, errs};
        auto it = fHCountsAbsorb.find(which);
        if (it == fHCountsAbsorb.end()) return {vals, errs};
        for (size_t i=0;i<fPtBins.size()-1;++i) {
            TH1F *h = it->second[i];
            int nb = h->GetNbinsX();
            std::vector<double> vbin(nb), ebin(nb);
            for (int b=0;b<nb;++b) {
                vbin[b] = h->GetBinContent(b+1);
                ebin[b] = h->GetBinError(b+1);
            }
            vals.push_back(vbin);
            errs.push_back(ebin);
        }
        return {vals, errs};
    }

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> GetAbsorptionEfficiency(const std::string &which="both") const {
        std::vector<std::vector<double>> vals, errs;
        if (fHRatio.empty()) return {vals, errs};
        auto it = fHRatio.find(which);
        if (it == fHRatio.end()) return {vals, errs};
        for (size_t i=0;i<fPtBins.size()-1;++i) {
            TH1F *h = it->second[i];
            int nb = h->GetNbinsX();
            std::vector<double> vbin(nb), ebin(nb);
            for (int b=0;b<nb;++b) {
                vbin[b] = h->GetBinContent(b+1);
                ebin[b] = h->GetBinError(b+1);
            }
            vals.push_back(vbin);
            errs.push_back(ebin);
        }
        return {vals, errs};
    }

    // expose histogram pointers (for saving/inspection)
    const std::map<std::string, std::vector<TH1F*>>& HistCounts() const { return fHCounts; }
    const std::map<std::string, std::vector<TH1F*>>& HistCountsAbsorb() const { return fHCountsAbsorb; }
    const std::map<std::string, std::vector<TH1F*>>& HistRatio() const { return fHRatio; }
    // Keys configuration interface
    void SetKeyNames(const std::string &both, const std::string &matter, const std::string &antimatter) {
        keyBoth = both;
        keyMat = matter;
        keyAnti = antimatter;
    }

    void SetKeyBoth(const std::string &k) { keyBoth = k; }
    void SetKeyMatter(const std::string &k) { keyMat = k; }
    void SetKeyAntimatter(const std::string &k) { keyAnti = k; }

    const std::string& GetKeyBoth() const { return keyBoth; }
    const std::string& GetKeyMatter() const { return keyMat; }
    const std::string& GetKeyAntimatter() const { return keyAnti; }

private:
    void initHistos() {
        // create one histogram per pt-bin for 'both','matter','antimatter'
        for (size_t i=0;i<fPtBins.size()-1;++i) {
            std::string name = Form("h_he_counts_pt%zu", i);
            std::string name_abs = Form("h_he_counts_absorb_pt%zu", i);
            std::string name_ratio = Form("h_he_ratio_absorb_pt%zu", i);
            // use decCt axis: sensible ct range 0..50 default; binning 100 if not provided
            std::vector<double> bins;
            if (!fCtBins.empty()) {
                bins = fCtBins[i];
            }
            else {
                bins.reserve(101);
                double xmin = 0.0, xmax = 50.0; int nb=100;
                for (int b=0;b<=nb;++b) bins.push_back(xmin + (xmax-xmin)/nb * b);
            }
            // create three histos per bin keyed by "both","matter","antimatter"
            fBinNames.push_back(Form("pt_%g_%g", fPtBins[i], fPtBins[i+1]));
            TH1F* fHCountsBinBoth = MakeTH1((name+"_both"+fHistNameSuffix).c_str(), Form("Both c#tau distribution before absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHCountsBinMatter = MakeTH1((name+"_matter"+fHistNameSuffix).c_str(), Form("Matter c#tau distribution before absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHCountsBinAnti = MakeTH1((name+"_antimat"+fHistNameSuffix).c_str(), Form("Antimatter c#tau distribution before absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHCountsAbsorbBinBoth = MakeTH1((name_abs+"_both"+fHistNameSuffix).c_str(), Form("Both c#tau distribution after absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHCountsAbsorbBinMatter = MakeTH1((name_abs+"_matter"+fHistNameSuffix).c_str(), Form("Matter c#tau distribution after absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHCountsAbsorbBinAnti = MakeTH1((name_abs+"_antimat"+fHistNameSuffix).c_str(), Form("Antimatter c#tau distribution after absorption pt: %g - %g ;c#tau;Counts", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHRatioBinBoth = MakeTH1((name_ratio+"_both"+fHistNameSuffix).c_str(), Form("Both abso efficiency pt: %g - %g ;c#tau;Efficiency", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHRatioBinMatter = MakeTH1((name_ratio+"_matter"+fHistNameSuffix).c_str(), Form("Matter abso efficiency pt: %g - %g ;c#tau;Efficiency", fPtBins[i], fPtBins[i+1]), bins);
            TH1F* fHRatioBinAnti = MakeTH1((name_ratio+"_antimat"+fHistNameSuffix).c_str(), Form("Antimatter abso efficiency pt: %g - %g ;c#tau;Efficiency", fPtBins[i], fPtBins[i+1]), bins);
            fHCounts[keyBoth].push_back(fHCountsBinBoth);
            fHCounts[keyMat].push_back(fHCountsBinMatter);
            fHCounts[keyAnti].push_back(fHCountsBinAnti);
            fHCountsAbsorb[keyBoth].push_back(fHCountsAbsorbBinBoth);
            fHCountsAbsorb[keyMat].push_back(fHCountsAbsorbBinMatter);
            fHCountsAbsorb[keyAnti].push_back(fHCountsAbsorbBinAnti);
            fHRatio[keyBoth].push_back(fHRatioBinBoth);
            fHRatio[keyMat].push_back(fHRatioBinMatter);
            fHRatio[keyAnti].push_back(fHRatioBinAnti);
        }
    }

    ROOT::RDataFrame* fRdfPtr;
    std::vector<double> fPtBins;
    std::vector<std::vector<double>> fCtBins;
    double fOrgCtao{7.6};
    std::string fHistNameSuffix{""};
    std::vector<std::string> fBinNames;
    std::map<std::string, std::vector<TH1F*>> fHCounts;
    std::map<std::string, std::vector<TH1F*>> fHCountsAbsorb;
    std::map<std::string, std::vector<TH1F*>> fHRatio;
    std::string keyBoth = "both";
    std::string keyMat = "matter";
    std::string keyAnti = "antimatter";
};

// SpectrumAbsorptionCalculator: fills pt-binned absorption efficiencies (both/matter/antimatter)
// using a simple survival test per candidate. Histograms are 1D in pt with the provided binning.
class SpectrumAbsorptionCalculator {
public:
    SpectrumAbsorptionCalculator(ROOT::RDF::RNode rdf, const std::vector<double> &ptBins, double org_ctao = 7.6)
        : fRdf(std::move(rdf)), fPtBins(ptBins), fOrgCtao(org_ctao) {
        initHistos();
    }

    void Calculate() {
        resetHistos();
        // slot-local histograms to keep thread safety
        std::vector<SlotHists> slotHists(1);
        slotHists.front().init(fPtBins);

        // RNG per slot
        std::vector<std::mt19937> rngs(1);
        auto seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        for (unsigned int i = 0; i < rngs.size(); ++i) rngs[i].seed(seed + i * 101);
        std::exponential_distribution<float> expo(1.0f / static_cast<float>(fOrgCtao));

        fRdf.ForeachSlot([&](unsigned int slot, float pt, float eta, float phi, float ax, float ay, float az, int pdg) {
            if (slot >= slotHists.size()) {
                slotHists.resize(slot + 1);
                slotHists[slot].init(fPtBins);
                rngs.resize(slot + 1);
                rngs[slot].seed(seed + slot * 101);
            }
            auto &sh = slotHists[slot];
            std::string key = (pdg > 0) ? keyMat : keyAnti;
            TLorentzVector lv; lv.SetPtEtaPhiM(pt, eta, phi, HE3_MASS);
            float he3p = lv.P();
            float absoL = std::sqrt(ax * ax + ay * ay + az * az);
            float absoCt = (he3p != 0) ? absoL * HE3_MASS / he3p : 1e9;
            float decCt = expo(rngs[slot]);
            sh.hCounts[keyBoth]->Fill(pt);
            sh.hCounts[key]->Fill(pt);
            if (absoCt > decCt) {
                sh.hCountsAbsorb[keyBoth]->Fill(pt);
                sh.hCountsAbsorb[key]->Fill(pt);
            }
        }, {"pt", "eta", "phi", "absoX", "absoY", "absoZ", "pdg"});

        // merge slot histograms
        for (const auto &sh : slotHists) {
            mergeInto(fHCounts, sh.hCounts);
            mergeInto(fHCountsAbsorb, sh.hCountsAbsorb);
        }

        // compute ratios
        for (const auto &key : {keyBoth, keyMat, keyAnti}) {
            fHRatio[key].Divide(&fHCountsAbsorb[key], &fHCounts[key], 1.0, 1.0, "B");
        }
    }

    const std::map<std::string, TH1F> &Counts() const { return fHCounts; }
    const std::map<std::string, TH1F> &CountsAbsorb() const { return fHCountsAbsorb; }
    const std::map<std::string, TH1F> &Ratio() const { return fHRatio; }

private:
    struct SlotHists {
        std::map<std::string, std::unique_ptr<TH1F>> hCounts;
        std::map<std::string, std::unique_ptr<TH1F>> hCountsAbsorb;

        void init(const std::vector<double> &edges) {
            if (!hCounts.empty()) return;
            auto make = [&](const std::string &name) {
                auto h = std::make_unique<TH1F>(name.c_str(), "" , static_cast<int>(edges.size()) - 1, edges.data());
                h->SetDirectory(nullptr);
                h->Sumw2();
                return h;
            };
            hCounts[keyBoth] = make("slot_counts_both");
            hCounts[keyMat] = make("slot_counts_matter");
            hCounts[keyAnti] = make("slot_counts_antimatter");
            hCountsAbsorb[keyBoth] = make("slot_abs_both");
            hCountsAbsorb[keyMat] = make("slot_abs_matter");
            hCountsAbsorb[keyAnti] = make("slot_abs_antimatter");
        }
    };

    void initHistos() {
        auto make = [&](const std::string &name) {
            TH1F h(name.c_str(), "" , static_cast<int>(fPtBins.size()) - 1, fPtBins.data());
            h.SetDirectory(nullptr);
            h.Sumw2();
            return h;
        };
        fHCounts[keyBoth] = make("counts_both");
        fHCounts[keyMat] = make("counts_matter");
        fHCounts[keyAnti] = make("counts_antimatter");
        fHCountsAbsorb[keyBoth] = make("counts_abs_both");
        fHCountsAbsorb[keyMat] = make("counts_abs_matter");
        fHCountsAbsorb[keyAnti] = make("counts_abs_antimatter");
        fHRatio[keyBoth] = make("ratio_both");
        fHRatio[keyMat] = make("ratio_matter");
        fHRatio[keyAnti] = make("ratio_antimatter");
    }

    void resetHistos() {
        for (auto &kv : fHCounts) kv.second.Reset();
        for (auto &kv : fHCountsAbsorb) kv.second.Reset();
        for (auto &kv : fHRatio) kv.second.Reset();
    }

    void mergeInto(std::map<std::string, TH1F> &dest,
                   const std::map<std::string, std::unique_ptr<TH1F>> &src) {
        for (const auto &kv : src) {
            if (kv.second) dest[kv.first].Add(kv.second.get());
        }
    }

    ROOT::RDF::RNode fRdf;
    std::vector<double> fPtBins;
    double fOrgCtao{7.6};
    std::map<std::string, TH1F> fHCounts;
    std::map<std::string, TH1F> fHCountsAbsorb;
    std::map<std::string, TH1F> fHRatio;
    inline static const std::string keyBoth = "both";
    inline static const std::string keyMat = "matter";
    inline static const std::string keyAnti = "antimatter";
};

} // namespace Absorption

#endif // ABSORPTION_HELPER_H
