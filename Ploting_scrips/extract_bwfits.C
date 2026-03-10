#include "TFile.h"
#include "TF1.h"
#include "TSystem.h"
#include <iostream>
#include <vector>
#include <utility>

void extract_bwfits() {
  std::vector<std::pair<std::string, std::string>> files = {
    {"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_AllCent/cen0-80/pt_analysis_pbpb.root", "BlastWave_H3L_0_80"},
    {"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen0-10/pt_analysis_pbpb.root", "BlastWave_H3L_0_10"},
    {"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen10-30/pt_analysis_pbpb.root", "BlastWave_H3L_10_30"},
    {"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen30-50/pt_analysis_pbpb.root", "BlastWave_H3L_30_50"},
    {"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen50-80/pt_analysis_pbpb.root", "BlastWave_H3L_50_80"}
  };

  TFile out("/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/H3L_BWFit_Run3_23.root", "RECREATE");
  for (const auto &entry : files) {
    const std::string &inPath = entry.first;
    const std::string &outName = entry.second;
    TFile inFile(inPath.c_str(), "READ");
    if (inFile.IsZombie()) {
      std::cout << "[Warn] Cannot open " << inPath << std::endl;
      continue;
    }
    auto f = dynamic_cast<TF1*>(inFile.Get("std/fBGBW_h_corrected_counts"));
    if (!f) {
      std::cout << "[Warn] Missing TF1 std/fBGBW_h_corrected_counts in " << inPath << std::endl;
      continue;
    }
    TF1 *clone = static_cast<TF1*>(f->Clone(outName.c_str()));
    out.cd();
    clone->Write(outName.c_str(), TObject::kOverwrite);
    std::cout << "[Info] Wrote " << outName << " from " << inPath << std::endl;
  }
  out.Close();
}
