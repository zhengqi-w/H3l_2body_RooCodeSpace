#include <iostream>
#include <vector>
#include <cmath>

#include "TH1.h"
#include "TFile.h"
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "RooArgSet.h"
#include "RooRealProxy.h"

using std::vector;


// ----------------- YOUR CHI2 AS ROOABSREAL ------------------
// GlobalChi2Roo: compute global chi2 (expo func) over multiple histograms with shared tau
class GlobalChi2Roo : public RooAbsReal {
public:
    // constructor takes references to external RooRealVar objects and TH1* vector
    GlobalChi2Roo(const char* name,
                  const vector<TH1*>& h_,
                  RooRealVar& tau_,
                  vector<RooRealVar*>& A_)
        : RooAbsReal(name, name),
          hists(h_),
          tauProxy("tauProxy", "tauProxy", this, tau_),
          tauVar(&tau_)
    {
        // store pointers to original RooRealVar objects
        Avars.reserve(A_.size());
        for (auto p : A_) Avars.push_back(p);
        // initialize proxies for A_i (allocate on heap)
        Aprox.reserve(Avars.size());
        for (size_t i = 0; i < Avars.size(); ++i) {
            std::string nm = std::string("A") + std::to_string(i);
            Aprox.push_back(new RooRealProxy(nm.c_str(), nm.c_str(), this, *Avars[i]));
        }
    }

    // copy ctor: build proxies referring to the same underlying RooRealVar objects
    GlobalChi2Roo(const GlobalChi2Roo& other, const char* name)
        : RooAbsReal(other, name),
          hists(other.hists),
          tauProxy("tauProxy", "tauProxy", this, *other.tauVar),
          tauVar(other.tauVar)
    {
        // copy pointer list and construct proxies bound to same RooRealVar objects
        Avars = other.Avars;
        Aprox.reserve(Avars.size());
        for (size_t i = 0; i < Avars.size(); ++i) {
            std::string nm = std::string("A") + std::to_string(i);
            Aprox.push_back(new RooRealProxy(nm.c_str(), nm.c_str(), this, *Avars[i]));
        }
    }

    virtual ~GlobalChi2Roo() {
        for (auto p : Aprox) {
            delete p;
        }
        Aprox.clear();
    }

    virtual TObject* clone(const char* newname) const override {
        return new GlobalChi2Roo(*this, newname);
    }

    void SetTauValue(double value) {
        if (tauVar) {
            tauVar->setVal(value);
        }
    }

    void SetTauConstant(bool isConstant) {
        if (tauVar) {
            tauVar->setConstant(isConstant);
        }
    }
    
protected:
    double evaluate() const override {
        double chi2 = 0.0;

        double tau_v = double(tauProxy);
        if (tau_v <= 0) return 1e20;  // forbid negative tau

        for (size_t j = 0; j < hists.size(); ++j) {
            TH1* h = hists[j];
            double Aj = 0.0;
            if (j < Aprox.size() && Aprox[j]) Aj = double(*Aprox[j]);

            int nb = h->GetNbinsX();

            for (int ib = 1; ib <= nb; ++ib) {
                double xlow = h->GetXaxis()->GetBinLowEdge(ib);
                double xhigh = h->GetXaxis()->GetBinUpEdge(ib);

                double expect = Aj * tau_v *
                    (std::exp(-xlow/tau_v) - std::exp(-xhigh/tau_v));
                double binwidth = xhigh - xlow;
                double obs = h->GetBinContent(ib) * binwidth;
                double err = h->GetBinError(ib);
                if (err <= 0) {
                    err = sqrt(std::max(1.0, obs));
                }

                double d = obs - expect;
                chi2 += d*d / (err*err*binwidth);
            }
        }

        return chi2;
    }

private:
    vector<TH1*> hists;                // TH1 pointers (not owned)
    RooRealProxy tauProxy;             // proxy for shared tau
    std::vector<RooRealProxy*> Aprox;  // proxies for per-channel amplitudes (heap-allocated pointers)

    // store the original RooRealVar pointers so copy ctor can bind proxies correctly
    RooRealVar* tauVar = nullptr;
    std::vector<RooRealVar*> Avars;
};
