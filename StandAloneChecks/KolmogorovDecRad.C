#include <TFile.h>
#include <TH1.h>
#include <TROOT.h>
#include <iostream>

void KolmogorovDecRad(const char *checksPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/Checks/checks.root",
                      const char *hyperPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/ct_single/both/Checks_hypertriton/checks_hypertriton.root") {
    gROOT->SetBatch(kTRUE);

    TFile fChecks(checksPath, "READ");
    if (fChecks.IsZombie()) {
        std::cerr << "[ERROR] Cannot open checks ROOT: " << checksPath << std::endl;
        return;
    }
    TH1 *hReco = dynamic_cast<TH1 *>(fChecks.Get("mc_checks/h1_fDecRad_reco"));
    if (!hReco) {
        std::cerr << "[ERROR] Missing hist mc_checks/h1_fDecRad_reco in " << checksPath << std::endl;
        return;
    }

    TFile fHyper(hyperPath, "READ");
    if (fHyper.IsZombie()) {
        std::cerr << "[ERROR] Cannot open hyper checks ROOT: " << hyperPath << std::endl;
        return;
    }
    TH1 *hAll = dynamic_cast<TH1 *>(fHyper.Get("all_candidates/h1_fDecRad"));
    if (!hAll) {
        std::cerr << "[ERROR] Missing hist all_candidates/h1_fDecRad in " << hyperPath << std::endl;
        return;
    }

    // Ensure same binning; KolmogorovTest will rebin internally only if identical bin edges.
    if (hReco->GetNbinsX() != hAll->GetNbinsX() ||
        hReco->GetXaxis()->GetXmin() != hAll->GetXaxis()->GetXmin() ||
        hReco->GetXaxis()->GetXmax() != hAll->GetXaxis()->GetXmax()) {
        std::cerr << "[WARN] Histogram binning differs; KolmogorovTest may be invalid." << std::endl;
    }

    const double pVal = hReco->KolmogorovTest(hAll, "M"); // "M" uses combined normalization
    std::cout << "KolmogorovTest(mc_checks/h1_fDecRad_reco vs all_candidates/h1_fDecRad) p-value = "
              << pVal << std::endl;
}
