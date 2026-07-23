#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TChain.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

using namespace GeneralHelper;

namespace {

int FindBin(const std::vector<double> &edges, double value)
{
    if (edges.size() < 2 || value < edges.front() || value >= edges.back()) return -1;
    auto upper = std::upper_bound(edges.begin(), edges.end(), value);
    const int idx = static_cast<int>(upper - edges.begin()) - 1;
    return (idx >= 0 && idx < static_cast<int>(edges.size()) - 1) ? idx : -1;
}

struct Counts {
    double oldDen = 0.0;
    double newDen = 0.0;
    double reco = 0.0;
};

double Ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

bool ReadAcceptanceTwoBodySwitch(const std::string &configPath, bool fallback = true)
{
    if (configPath.empty()) return fallback;
    try {
        const auto cfg = GeneralHelper::LoadJsonFile(configPath);
        const auto common = cfg.value("common", GeneralHelper::Json::object());
        const auto selection = common.value("selection", GeneralHelper::Json::object());
        return selection.value("mc_acceptance_require_two_body",
                               selection.value("is_two_body_selected", fallback));
    } catch (const std::exception &e) {
        std::cerr << "[AcceptanceRecoMCCollisionCompare] Failed to read two-body switch from config: "
                  << e.what() << ". Use fallback=" << fallback << std::endl;
        return fallback;
    }
}

} // namespace

void AcceptanceRecoMCCollisionCompare(
    const std::string &mcPath = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
    const std::string &outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/StandAloneChecks/AcceptanceRecoMCCollisionCompare",
    const std::vector<double> &centBins = {0, 5, 10, 20, 30, 40, 50, 60, 70, 90},
    const std::vector<std::vector<double>> &ptBinsByCent = {
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 7},
        {2, 2.5, 3, 3.5, 4, 4.5, 5, 7},
        {2, 2.5, 3, 3.5, 4, 6},
        {2, 2.5, 3.5, 6}},
    const std::string &basicSelection = "fDecRad > 0.8",
    const std::string &matterOpt = "both",
    const std::string &configPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/general_config.json")
{
    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT(std::clamp(std::thread::hardware_concurrency(), 2u, 12u));
    }

    TChain chain("O2mchypcands");
    auto file = std::unique_ptr<TFile>(TFile::Open(mcPath.c_str(), "READ"));
    if (!file || file->IsZombie()) {
        throw std::runtime_error("Cannot open MC file: " + mcPath);
    }
    fillChainFromAO2D(chain, file.get());

    ROOT::RDataFrame rdf(chain);
    auto ready = CorrectAndConvertRDF(rdf, false, true, false)
        .Define("__acc_basic", ("(" + basicSelection + ") ? 1 : 0").c_str())
        .Define("__acc_evsel", "(fIsSurvEvSel) ? 1 : 0")
        .Define("__acc_reco", "(fIsReco) ? 1 : 0")
        .Define("__acc_reco_mc_collision", "(fIsRecoMCCollision) ? 1 : 0")
        .Define("__acc_two_body", "(fIsTwoBodyDecay > 0) ? 1 : 0");
    const bool requireTwoBody = ReadAcceptanceTwoBodySwitch(configPath, true);

    const int nCent = static_cast<int>(centBins.size()) - 1;
    std::vector<std::vector<Counts>> totals(nCent);
    for (int ic = 0; ic < nCent; ++ic) {
        totals[ic].assign(static_cast<int>(ptBinsByCent[ic].size()) - 1, Counts{});
    }

    std::vector<std::vector<std::vector<Counts>>> slotTotals;
    std::mutex slotMutex;
    auto acquireSlot = [&](unsigned slot) -> std::vector<std::vector<Counts>>& {
        if (slot < slotTotals.size()) return slotTotals[slot];
        std::lock_guard<std::mutex> guard(slotMutex);
        if (slot >= slotTotals.size()) {
            slotTotals.resize(slot + 1);
            for (auto &perCent : slotTotals) {
                if (!perCent.empty()) continue;
                perCent.resize(nCent);
                for (int ic = 0; ic < nCent; ++ic) {
                    perCent[ic].assign(static_cast<int>(ptBinsByCent[ic].size()) - 1, Counts{});
                }
            }
        }
        return slotTotals[slot];
    };

    ready.ForeachSlot(
        [&](unsigned slot, float genPt, float cent, float genMatter,
            int evselFlag, int recoFlag, int recoMCCollisionFlag, int basicFlag, int twoBodyFlag) {
            if (matterOpt == "matter" && genMatter <= 0.0) return;
            if (matterOpt == "antimatter" && genMatter > 0.0) return;
            const int centIdx = FindBin(centBins, cent);
            if (centIdx < 0) return;
            const int ptIdx = FindBin(ptBinsByCent[centIdx], genPt);
            if (ptIdx < 0) return;

            const bool passFundamental = (!requireTwoBody) || (twoBodyFlag != 0);
            const bool passOldDen = passFundamental && (evselFlag != 0);
            const bool passNewDen = passOldDen && (recoMCCollisionFlag != 0);
            const bool passReco = passFundamental && (basicFlag != 0) && (recoFlag != 0);

            auto &counts = acquireSlot(slot)[centIdx][ptIdx];
            if (passOldDen) counts.oldDen += 1.0;
            if (passNewDen) counts.newDen += 1.0;
            if (passReco) counts.reco += 1.0;
        },
        {"fAbsGenPt", "fCentralityFT0C", "fGenPt", "__acc_evsel", "__acc_reco",
         "__acc_reco_mc_collision", "__acc_basic", "__acc_two_body"});

    for (const auto &slot : slotTotals) {
        for (int ic = 0; ic < nCent; ++ic) {
            for (size_t ip = 0; ip < totals[ic].size(); ++ip) {
                totals[ic][ip].oldDen += slot[ic][ip].oldDen;
                totals[ic][ip].newDen += slot[ic][ip].newDen;
                totals[ic][ip].reco += slot[ic][ip].reco;
            }
        }
    }

    std::filesystem::create_directories(outputDir);
    const std::string csvPath = outputDir + "/acceptance_reco_mc_collision_compare.csv";
    std::ofstream csv(csvPath);
    csv << "centrality,pt_low,pt_high,old_den,new_den,reco,old_acc,new_acc,new_over_old,relative_change_percent\n";
    csv << std::setprecision(10);

    std::cout << "\nAcceptance denominator comparison: old=fIsSurvEvSel, new=fIsSurvEvSel && fIsRecoMCCollision\n";
    std::cout << "matterOpt=" << matterOpt
              << ", mc_acceptance_require_two_body=" << requireTwoBody << "\n";
    std::cout << std::setw(12) << "centrality"
              << std::setw(14) << "pt"
              << std::setw(14) << "old_acc"
              << std::setw(14) << "new_acc"
              << std::setw(14) << "new/old"
              << std::setw(14) << "%change"
              << "\n";

    for (int ic = 0; ic < nCent; ++ic) {
        for (size_t ip = 0; ip < totals[ic].size(); ++ip) {
            const auto &c = totals[ic][ip];
            const double oldAcc = Ratio(c.reco, c.oldDen);
            const double newAcc = Ratio(c.reco, c.newDen);
            const double newOverOld = oldAcc > 0.0 ? newAcc / oldAcc : 0.0;
            const double pct = oldAcc > 0.0 ? (newAcc - oldAcc) / oldAcc * 100.0 : 0.0;
            csv << centBins[ic] << "-" << centBins[ic + 1] << ","
                << ptBinsByCent[ic][ip] << ","
                << ptBinsByCent[ic][ip + 1] << ","
                << c.oldDen << ","
                << c.newDen << ","
                << c.reco << ","
                << oldAcc << ","
                << newAcc << ","
                << newOverOld << ","
                << pct << "\n";
            std::cout << std::setw(5) << centBins[ic] << "-" << std::left << std::setw(6) << centBins[ic + 1] << std::right
                      << std::setw(6) << ptBinsByCent[ic][ip] << "-" << std::left << std::setw(6) << ptBinsByCent[ic][ip + 1] << std::right
                      << std::setw(14) << oldAcc
                      << std::setw(14) << newAcc
                      << std::setw(14) << newOverOld
                      << std::setw(14) << pct
                      << "\n";
        }
    }
    std::cout << "Saved CSV: " << csvPath << "\n";
}
