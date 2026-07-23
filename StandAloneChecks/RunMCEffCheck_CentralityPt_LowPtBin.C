#include "MCEffCheck.C"

void RunMCEffCheck_CentralityPt_LowPtBin() {
    const std::string mc =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root";

    const std::vector<std::string> labels = {
        "LHC25g11 G4list reweighted"
    };
    const std::vector<bool> twoBody = {true};
    const std::vector<std::string> selection = {
        "fDecRad > 0.8"
    };

    const std::vector<double> cenbins = {0, 5, 10, 20, 30, 40, 50, 60, 70, 90};
    const std::vector<std::vector<double>> ptbinsforcent = {
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8},
        {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7},
        {1.5, 2, 2.5, 3, 3.5, 4.5, 6.5},
        {1.5, 2, 2.5, 3.5, 6}
    };
    const std::vector<double> ptbins = {1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8};
    const std::vector<double> ctbins = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40};
    const std::vector<double> ptbinsforct = {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8};
    const std::vector<std::vector<double>> ctbinsforpt = {
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35}
    };

    const std::string out =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/MCEfficiency_CentralityPt_LowPtBin";

    MCEffCheck(mc, {}, twoBody, labels, cenbins, ptbinsforcent,
               ptbins, ctbins, ptbinsforct, ctbinsforpt,
               selection, out, "both",
               "centrality-pT bins include 1.5-2 GeV/c",
               "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json");
}
