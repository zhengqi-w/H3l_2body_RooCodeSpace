#include "MCEffCheck.C"

void RunMCEffCheck_PeriodMCCompare() {
    const std::string lhc23 =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root";
    const std::vector<std::string> compare = {
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5_G4list/reweighted/AO2D_combined_reweighted.root",
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6_G4list/reweighted/AO2D_combined_reweighted.root"
    };

    const std::vector<std::string> labels = {
        "LHC23 pass5 MC",
        "LHC24ar pass3 MC",
        "LHC25 pass1 MC"
    };
    const std::vector<bool> twoBody = {true, true, true};
    const std::vector<std::string> selection = {
        "fDecRad > 0.8",
        "fDecRad > 0.8",
        "fDecRad > 0.8"
    };

    const std::vector<double> cenbins = {0, 5, 10, 20, 30, 40, 50, 60, 70, 90};
    const std::vector<std::vector<double>> ptbinsforcent = {
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 7},
        {2, 2.5, 3, 3.5, 4, 6},
        {2, 3, 4, 6}
    };
    const std::vector<double> ptbins = {2, 3, 3.5, 4, 5.5, 6.5, 8};
    const std::vector<double> ctbins = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40};
    const std::vector<double> ptbinsforct = {2, 3, 3.5, 4, 5.5, 6.5, 8};
    const std::vector<std::vector<double>> ctbinsforpt = {
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.2, 3.5, 5.2, 7.5, 11, 18, 35},
        {1, 2.6, 4.6, 7.5, 12, 20, 35},
        {1, 2.6, 4.6, 8, 15, 35},
        {1, 3, 6, 12, 35}
    };

    const std::string out =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/MCEfficiency_PeriodMCCompare";

    MCEffCheck(lhc23, compare, twoBody, labels, cenbins, ptbinsforcent,
               ptbins, ctbins, ptbinsforct, ctbinsforpt,
               selection, out, "both",
               "period MC comparison, LHC24ar/LHC25 over LHC23",
               "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json");
}
