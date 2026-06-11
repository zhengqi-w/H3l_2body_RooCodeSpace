#include "MCEffCheck.C"

void RunMCEffCheck_FT0C_0_90() {
    const std::string base = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc";
    const std::string stdmc = base + "/LHC25g11/AO2D_V0s_full.root";
    const std::vector<std::string> compare = {
        base + "/LHC25g11/AO2D_CustomV0s.root",
        base + "/LHC25g11_G4list/AO2D_V0s_full.root",
        base + "/LHC25g11_G4list/AO2D_CustomV0s.root"
    };
    const std::vector<std::string> labels = {
        "LHC25g11 V0s full",
        "LHC25g11 Custom V0s",
        "LHC25g11 G4list V0s full",
        "LHC25g11 G4list Custom V0s"
    };
    const std::vector<bool> twoBody = {true, true, true, true};
    const std::vector<std::string> selection(labels.size(), "fCentralityFT0C >= 0 && fCentralityFT0C < 90");
    const std::vector<double> cenbins = {0, 90};
    const std::vector<std::vector<double>> ptbinsforcent = {
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8}
    };
    const std::string out = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/MCEfficiency_FT0C_0_90_Compare4";

    MCEffCheck(stdmc, compare, twoBody, labels, cenbins, ptbinsforcent,
               {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8},
               {1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 27, 33},
               {2, 3, 4, 5.5, 8},
               {{1, 3, 6, 9, 12, 18, 30},
                {1, 3, 6, 9, 12, 18, 25},
                {1, 3, 6, 9, 15, 25},
                {1, 3, 6, 10, 23}},
               selection, out, "both", "Centrality: 0-90% FT0C");
}
