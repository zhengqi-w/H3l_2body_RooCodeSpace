#include "../Tools/GeneralHelper.hpp"

#include <ROOT/RDataFrame.hxx>

#include <TCanvas.h>
#include <TChain.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLegend.h>
#include <TObject.h>
#include <TRandom.h>
#include <TTree.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct CenWeightBin {
  std::string suffix;
  std::string funcName;
  std::unique_ptr<TF1> func;
};

std::vector<std::string> GetBranchNames(TTree *tree) {
  std::vector<std::string> names;
  if (!tree) return names;
  auto *branches = tree->GetListOfBranches();
  if (!branches) return names;
  names.reserve(branches->GetEntries());
  for (int i = 0; i < branches->GetEntries(); ++i) {
    auto *obj = branches->At(i);
    if (!obj) continue;
    names.emplace_back(obj->GetName());
  }
  return names;
}

TF1 *LoadTF1(TFile &f, const std::string &name) {
  // Support both direct key names and legacy directory-style paths.
  if (auto *obj = f.Get(name.c_str())) {
    if (auto *fn = dynamic_cast<TF1 *>(obj)) return fn;
  }

  const std::string inLegacyDir = "BlastWave_H3l_0-80/" + name;
  if (auto *obj = f.Get(inLegacyDir.c_str())) {
    if (auto *fn = dynamic_cast<TF1 *>(obj)) return fn;
  }

  return nullptr;
}

std::vector<CenWeightBin> BuildWeightBins(TFile &bwFile) {
  std::vector<CenWeightBin> bins;
  bins.push_back({"0_10", "BlastWave_H3L_0_10", nullptr});
  bins.push_back({"10_30", "BlastWave_H3L_10_30", nullptr});
  bins.push_back({"30_50", "BlastWave_H3L_30_50", nullptr});
  bins.push_back({"50_80", "BlastWave_H3L_50_80", nullptr});
  bins.push_back({"0_80", "BlastWave_H3L_0_80", nullptr});

  TF1 *fallback = LoadTF1(bwFile, "BlastWave_H3L_0_80");
  if (!fallback) {
    fallback = LoadTF1(bwFile, "0_80");
  }

  for (auto &b : bins) {
    TF1 *found = LoadTF1(bwFile, b.funcName);
    if (!found) {
      const std::string shortName = b.suffix;
      found = LoadTF1(bwFile, shortName);
    }
    if (!found) {
      found = fallback;
      std::cout << "[Warn] Missing TF1 " << b.funcName << ", fallback to BlastWave_H3L_0_80" << std::endl;
    }
    if (!found) {
      throw std::runtime_error("No valid reweight TF1 found in BW file");
    }
    b.func.reset(static_cast<TF1 *>(found->Clone((b.funcName + "_clone").c_str())));
  }
  return bins;
}

std::string TempFileName(const std::string &baseDir, const std::string &dfName, int idx) {
  return baseDir + "/tmp_reweight_" + dfName + "_" + std::to_string(idx) + ".root";
}

void DrawQaAndSave(TH1D &hBefore, TH1D &hAfter, const std::string &outDir, const std::string &tag = "") {
  auto hBeforeDraw = std::unique_ptr<TH1D>(static_cast<TH1D *>(hBefore.Clone("hAbsGenPtBefore_draw")));
  auto hAfterDraw = std::unique_ptr<TH1D>(static_cast<TH1D *>(hAfter.Clone("hAbsGenPtAfter_draw")));

  hBeforeDraw->SetStats(0);
  hAfterDraw->SetStats(0);
  hBeforeDraw->SetLineColor(kBlack);
  hBeforeDraw->SetLineWidth(2);
  hAfterDraw->SetLineColor(kRed + 1);
  hAfterDraw->SetLineWidth(2);

  TCanvas c("cAbsGenPt", "fAbsGenPt QA", 900, 700);
  c.SetTicks(1, 1);
  c.SetLogy();

  hBeforeDraw->SetTitle("fAbsGenPt QA;fAbsGenPt (GeV/c);Counts");
  hBeforeDraw->Draw("hist");
  hAfterDraw->Draw("hist same");

  TLegend leg(0.55, 0.72, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.AddEntry(hBeforeDraw.get(), "Before reweight", "l");
  leg.AddEntry(hAfterDraw.get(), "After reweight", "l");
  leg.Draw();

  const std::string pdf = tag.empty() ?
      (outDir + "/QA_fAbsGenPt_before_after.pdf") :
      (outDir + "/QA_fAbsGenPt_before_after_" + tag + ".pdf");
  c.SaveAs(pdf.c_str());
}

}  // namespace

void ReweightMCAO2D(
    const std::string &inputAo2d =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root",
    const std::string &bwFilePath =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/H3L_BWFit_Run3_23.root",
    const std::string &outputDir =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/reweighted",
    const std::string &treeName = "O2mchypcands") {
  if (ROOT::IsImplicitMTEnabled()) {
    ROOT::DisableImplicitMT();
  }

  std::filesystem::create_directories(outputDir);

  TFile inputFile(inputAo2d.c_str(), "READ");
  if (inputFile.IsZombie()) {
    throw std::runtime_error("Failed to open input AO2D: " + inputAo2d);
  }

  TFile bwFile(bwFilePath.c_str(), "READ");
  if (bwFile.IsZombie()) {
    throw std::runtime_error("Failed to open BW file: " + bwFilePath);
  }

  auto weightBins = BuildWeightBins(bwFile);

  const auto inPath = std::filesystem::path(inputAo2d);
  for (const auto &wb : weightBins) {
    const std::string outFilePath = outputDir + "/" + inPath.stem().string() + "_" + wb.suffix + "_reweighted" + inPath.extension().string();
    TFile outFile(outFilePath.c_str(), "RECREATE");
    if (outFile.IsZombie()) {
      throw std::runtime_error("Failed to create output AO2D: " + outFilePath);
    }

    TH1D hBeforeAll("hAbsGenPtBefore", ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0);
    TH1D hAfterAll("hAbsGenPtAfter", ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0);
    hBeforeAll.Sumw2();
    hAfterAll.Sumw2();

    TIter nextKey(inputFile.GetListOfKeys());
    while (auto *key = static_cast<TKey *>(nextKey())) {
      TObject *obj = key->ReadObj();
      if (!obj) continue;

      const std::string keyName = key->GetName();
      if (obj->InheritsFrom(TDirectory::Class()) && keyName.rfind("DF_", 0) == 0) {
        auto *inDf = dynamic_cast<TDirectory *>(obj);
        auto *outDf = outFile.mkdir(keyName.c_str());
        if (!inDf || !outDf) continue;

        TIter nextSubKey(inDf->GetListOfKeys());
        while (auto *subKey = static_cast<TKey *>(nextSubKey())) {
          TObject *subObj = subKey->ReadObj();
          if (!subObj) continue;

          const std::string subName = subKey->GetName();
          if (subObj->InheritsFrom(TTree::Class()) && subName == treeName) {
            auto *inTree = dynamic_cast<TTree *>(subObj);
            if (!inTree) continue;

            auto originalColumns = GetBranchNames(inTree);
            ROOT::RDataFrame rdfInput(*inTree);
            auto rdfReady = GeneralHelper::CorrectAndConvertRDF(rdfInput, false, true, false);

            auto hBefore = rdfReady.Histo1D(
                {"hBefore_tmp", ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0}, "fAbsGenPt");
            auto weighted = GeneralHelper::ReWeightSpectrum(rdfReady, wb.func.get(), "fAbsGenPt");
            auto hAfter = weighted.Histo1D(
                {"hAfter_tmp", ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0}, "fAbsGenPt");

            const auto &hBeforeVal = hBefore.GetValue();
            const auto &hAfterVal = hAfter.GetValue();
            hBeforeAll.Add(&hBeforeVal);
            hAfterAll.Add(&hAfterVal);

            const std::string tmp = TempFileName(outputDir, keyName + "_" + wb.suffix, 0);
            auto snap = weighted.Snapshot(treeName.c_str(), tmp.c_str(), originalColumns);
            snap.GetValue();

            TChain mergedChain(treeName.c_str());
            mergedChain.Add(tmp.c_str());
            outDf->cd();
            TTree *mergedTree = mergedChain.CloneTree(-1, "fast");
            if (mergedTree) {
              mergedTree->SetName(treeName.c_str());
              mergedTree->Write("", TObject::kOverwrite);
              delete mergedTree;
            }

            std::error_code ec;
            std::filesystem::remove(tmp, ec);
          } else {
            outDf->cd();
            subObj->Write(subName.c_str(), TObject::kOverwrite);
          }
        }
      } else {
        outFile.cd();
        obj->Write(keyName.c_str(), TObject::kOverwrite);
      }
    }

    outFile.cd();
    hBeforeAll.Write("hAbsGenPtBefore");
    hAfterAll.Write("hAbsGenPtAfter");
    DrawQaAndSave(hBeforeAll, hAfterAll, outputDir, wb.suffix);
    std::cout << "[Done] Reweighted AO2D saved to: " << outFilePath << std::endl;
  }

  std::cout << "[Done] QA plots saved to: " << outputDir << std::endl;
}
