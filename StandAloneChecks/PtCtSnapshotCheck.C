#include <ROOT/RDataFrame.hxx>
#include <TROOT.h>
#include <TSystem.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
long long countEntries(const std::string &file, const std::string &tree) {
    if (!std::filesystem::exists(file)) return -1; // missing file
    try {
        ROOT::RDataFrame df(tree, file);
        return *df.Count();
    } catch (...) {
        return -2; // failed to read
    }
}
}

void PtCtSnapshotCheck(const std::string &snapshotDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID_Test",
                       const std::string &treeName = "O2hypcands") {
    ROOT::EnableImplicitMT();

    const std::vector<double> ptBins = {2.0, 3.0, 4.0, 5.5, 8.0};
    const std::vector<double> ctBins = {1.0, 3.0, 6.0, 10.0, 23.0};

    if (!std::filesystem::exists(snapshotDir)) {
        std::cerr << "[ERROR] Snapshot dir missing: " << snapshotDir << std::endl;
        return;
    }

    std::cout << "Snapshot dir: " << snapshotDir << "\n";
    std::cout << "Tree name  : " << treeName << "\n";

    std::cout << std::left << std::setw(14) << "ct bin" << std::setw(18) << "data_ct count" << std::setw(18)
              << "sum pt_ct" << "diff (pt_ct - ct)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (size_t ict = 0; ict + 1 < ctBins.size(); ++ict) {
        const int ctLo = static_cast<int>(ctBins[ict]);
        const int ctHi = static_cast<int>(ctBins[ict + 1]);
        const std::string ctFile = snapshotDir + "/data_ct_" + std::to_string(ctLo) + "_" + std::to_string(ctHi) + ".root";

        const long long ctCount = countEntries(ctFile, treeName);

        long long sumPtCt = 0;
        bool hasError = false;
        std::vector<long long> ptCounts;
        ptCounts.reserve(ptBins.size() - 1);

        for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
            const double ptLo = ptBins[ipt];
            const double ptHi = ptBins[ipt + 1];
            std::ostringstream fname;
            fname << snapshotDir << "/data_pt_" << ptLo << "_" << ptHi << "_ct_" << ctLo << "_" << ctHi << ".root";
            const long long cnt = countEntries(fname.str(), treeName);
            ptCounts.push_back(cnt);
            if (cnt >= 0) sumPtCt += cnt; else hasError = true;
        }

        const long long diff = (ctCount >= 0 && sumPtCt >= 0) ? (sumPtCt - ctCount) : -9;
        std::cout << std::left << std::setw(14) << (Form("[%d,%d]", ctLo, ctHi))
                  << std::setw(18) << ctCount << std::setw(18) << sumPtCt << diff << "\n";

        std::cout << "  data_ct file : " << ctFile << (ctCount < 0 ? " (missing/error)" : "") << "\n";
        for (size_t ipt = 0; ipt + 1 < ptBins.size(); ++ipt) {
            const double ptLo = ptBins[ipt];
            const double ptHi = ptBins[ipt + 1];
            const auto cnt = ptCounts[ipt];
            std::ostringstream fname;
            fname << "data_pt_" << ptLo << "_" << ptHi << "_ct_" << ctLo << "_" << ctHi << ".root";
            std::cout << "    " << std::setw(32) << fname.str() << " -> " << cnt << "\n";
        }
        if (hasError) std::cout << "  [WARN] At least one pt-ct file missing or failed to read." << "\n";
        std::cout << std::string(70, '-') << "\n";
    }

    std::cout << "Done." << std::endl;
}
