#include "MCEffCheck.C"

void CompareLHC26e6ReweightedMCEfficiency() {
    const std::string oldMc =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root";
    const std::string newMc =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6_G4list/reweighted/AO2D_combined_reweighted.root";

    const std::vector<std::string> compare = {newMc};
    const std::vector<std::string> labels = {
        "LHC25g11 G4list reweighted",
        "LHC26e6 G4list reweighted"
    };
    const std::vector<bool> twoBody = {true, true};
    const std::vector<std::string> selection = {
        "fDecRad > 0.8 && fCentralityFT0C >= 0 && fCentralityFT0C < 90",
        "fDecRad > 0.8 && fCentralityFT0C >= 0 && fCentralityFT0C < 90"
    };

    const std::vector<double> cenbins = {0, 10, 30, 50, 80};
    const std::vector<std::vector<double>> ptbinsforcent = {
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8}
    };
    const std::vector<double> ptbins = {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8};
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
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/MCEfficiency_LHC26e6_vs_LHC25g11_Reweighted";

    MCEffCheck(oldMc, compare, twoBody, labels, cenbins, ptbinsforcent,
               ptbins, ctbins, ptbinsforct, ctbinsforpt,
               selection, out, "both",
               "0-80% FT0C, LHC26e6 reweighted",
               "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config_merged.json");
}
