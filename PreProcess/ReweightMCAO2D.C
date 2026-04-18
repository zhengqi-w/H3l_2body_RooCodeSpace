#include <ROOT/RDataFrame.hxx>

#include <RooFitResult.h>

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
#include <stdexcept>
#include <string>
#include <vector>

#include "../Tools/GeneralHelper.hpp"

namespace {

struct CenWeightBin {
  std::string suffix;
  std::string funcName;
  std::unique_ptr<TF1> func;
};

struct CombinedCenBin {
  std::string label;
  float minCen;
  float maxCen;
  std::string funcName;
  std::unique_ptr<TF1> func;
  int color;
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

std::unique_ptr<TF1> LoadWeightFunctionOrThrow(TFile &bwFile, const std::string &funcName, const std::string &shortName, const std::string &fallbackName = "") {
  TF1 *found = LoadTF1(bwFile, funcName);
  if (!found && !shortName.empty()) {
    found = LoadTF1(bwFile, shortName);
  }
  if (!found && !fallbackName.empty()) {
    found = LoadTF1(bwFile, fallbackName);
    if (found) {
      std::cout << "[Warn] Missing TF1 " << funcName << ", fallback to " << fallbackName << std::endl;
    }
  }
  if (!found) {
    throw std::runtime_error("Missing reweight TF1: " + funcName);
  }
  return std::unique_ptr<TF1>(static_cast<TF1 *>(found->Clone((funcName + "_clone").c_str())));
}

std::vector<CombinedCenBin> BuildCombinedBins(TFile &bwFile) {
  std::vector<CombinedCenBin> bins;
  bins.push_back({"0-10", 0.f, 10.f, "BlastWave_0_10", nullptr, kRed + 1});
  bins.push_back({"10-30", 10.f, 30.f, "BlastWave_10_30", nullptr, kBlue + 1});
  bins.push_back({"30-50", 30.f, 50.f, "BlastWave_30_50", nullptr, kGreen + 2});
  bins.push_back({">=50 (use 50-90)", 50.f, 90.f, "BlastWave_50_90", nullptr, kOrange + 7});

  for (auto &b : bins) {
    std::string shortName = b.label;
    std::replace(shortName.begin(), shortName.end(), '-', '_');
    b.func = LoadWeightFunctionOrThrow(bwFile, b.funcName, shortName);
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

void DrawCombinedQaAndSave(
    TH1D &hCent,
    const std::vector<TH1D> &hBeforeByCen,
    const std::vector<TH1D> &hAfterByCen,
    const std::vector<CombinedCenBin> &bins,
    const std::string &outDir,
    const std::string &tag = "combined") {
  if (hBeforeByCen.size() != bins.size() || hAfterByCen.size() != bins.size()) {
    throw std::runtime_error("Combined QA histogram/bin size mismatch");
  }

  TCanvas c("cAbsGenPtCombined", "fAbsGenPt Combined QA", 1600, 700);
  c.Divide(2, 1);

  c.cd(1);
  gPad->SetTicks(1, 1);
  auto hCentBase = std::unique_ptr<TH1D>(static_cast<TH1D *>(hCent.Clone("hCentBase")));
  hCentBase->SetStats(0);
  hCentBase->SetLineColor(kBlack);
  hCentBase->SetLineWidth(2);
  hCentBase->SetTitle("Centrality regions;fCentralityFT0C;Counts");
  hCentBase->Draw("hist");

  std::vector<std::unique_ptr<TH1D>> hCentRegions;
  hCentRegions.reserve(bins.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    hCentRegions.emplace_back(static_cast<TH1D *>(hCent.Clone(("hCentRegion_" + std::to_string(i)).c_str())));
    auto &hReg = hCentRegions.back();
    for (int b = 1; b <= hReg->GetNbinsX(); ++b) {
      const double cval = hReg->GetXaxis()->GetBinCenter(b);
      const bool inRange = (i + 1 == bins.size()) ? (cval >= bins[i].minCen) : (cval >= bins[i].minCen && cval < bins[i].maxCen);
      if (!inRange) {
        hReg->SetBinContent(b, 0.0);
        hReg->SetBinError(b, 0.0);
      }
    }
    hReg->SetLineColor(bins[i].color);
    hReg->SetFillColorAlpha(bins[i].color, 0.35);
    hReg->SetLineWidth(2);
    hReg->Draw("hist same");
  }

  TLegend legCen(0.52, 0.62, 0.88, 0.88);
  legCen.SetBorderSize(0);
  legCen.SetFillStyle(0);
  for (size_t i = 0; i < bins.size(); ++i) {
    if (i + 1 == bins.size()) {
      legCen.AddEntry(hCentRegions[i].get(), (bins[i].label + " (He3: 50-90)").c_str(), "lf");
    } else {
      legCen.AddEntry(hCentRegions[i].get(), (bins[i].label + " (H3L: " + bins[i].label + ")").c_str(), "lf");
    }
  }
  legCen.Draw();

  c.cd(2);
  gPad->SetTicks(1, 1);
  gPad->SetLogy();

  double maxY = 0.0;
  for (size_t i = 0; i < bins.size(); ++i) {
    maxY = std::max(maxY, hBeforeByCen[i].GetMaximum());
    maxY = std::max(maxY, hAfterByCen[i].GetMaximum());
  }

  auto frame = std::unique_ptr<TH1D>(static_cast<TH1D *>(hBeforeByCen.front().Clone("hPtFrame")));
  frame->Reset("ICES");
  frame->SetStats(0);
  frame->SetTitle("fAbsGenPt by centrality;fAbsGenPt (GeV/c);Counts");
  frame->SetMaximum(maxY > 0 ? maxY * 2.0 : 1.0);
  frame->SetMinimum(0.5);
  frame->GetYaxis()->SetTitleOffset(1.25);
  frame->GetXaxis()->SetTitleOffset(1.05);
  frame->Draw("hist");

  std::vector<std::unique_ptr<TH1D>> hBeforeDraw;
  std::vector<std::unique_ptr<TH1D>> hAfterDraw;
  hBeforeDraw.reserve(bins.size());
  hAfterDraw.reserve(bins.size());

  for (size_t i = 0; i < bins.size(); ++i) {
    hBeforeDraw.emplace_back(static_cast<TH1D *>(hBeforeByCen[i].Clone(("hBeforeDraw_" + std::to_string(i)).c_str())));
    hAfterDraw.emplace_back(static_cast<TH1D *>(hAfterByCen[i].Clone(("hAfterDraw_" + std::to_string(i)).c_str())));
    hBeforeDraw.back()->SetDirectory(nullptr);
    hAfterDraw.back()->SetDirectory(nullptr);

    hBeforeDraw.back()->SetStats(0);
    hAfterDraw.back()->SetStats(0);
    hBeforeDraw.back()->SetLineColor(bins[i].color);
    hAfterDraw.back()->SetLineColor(bins[i].color);
    hBeforeDraw.back()->SetLineStyle(2);
    hAfterDraw.back()->SetLineStyle(1);
    hBeforeDraw.back()->SetLineWidth(2);
    hAfterDraw.back()->SetLineWidth(3);

    hBeforeDraw.back()->Draw("hist same");
    hAfterDraw.back()->Draw("hist same");
  }

  TLegend legCenPt(0.50, 0.56, 0.78, 0.88);
  legCenPt.SetBorderSize(0);
  legCenPt.SetFillStyle(0);
  for (size_t i = 0; i < bins.size(); ++i) {
    std::string label = bins[i].label + " ( H3L: " + bins[i].label + " )";
    if (i + 1 == bins.size()) {
      label = bins[i].label + " ( He3: 50-90 )";
    }
    legCenPt.AddEntry(hAfterDraw[i].get(), label.c_str(), "l");
  }
  legCenPt.Draw();

  auto hStyleBefore = std::unique_ptr<TH1D>(static_cast<TH1D *>(frame->Clone("hStyleBefore")));
  auto hStyleAfter = std::unique_ptr<TH1D>(static_cast<TH1D *>(frame->Clone("hStyleAfter")));
  hStyleBefore->SetLineColor(kBlack);
  hStyleAfter->SetLineColor(kBlack);
  hStyleBefore->SetLineStyle(2);
  hStyleAfter->SetLineStyle(1);
  hStyleBefore->SetLineWidth(2);
  hStyleAfter->SetLineWidth(3);

  TLegend legStyle(0.80, 0.74, 0.95, 0.88);
  legStyle.SetBorderSize(0);
  legStyle.SetFillStyle(0);
  legStyle.AddEntry(hStyleBefore.get(), "Before", "l");
  legStyle.AddEntry(hStyleAfter.get(), "After", "l");
  legStyle.Draw();

  c.cd();
  const std::string pdf = outDir + "/QA_fAbsGenPt_before_after_" + tag + ".pdf";
  c.SaveAs(pdf.c_str());
}

}  // namespace

void ReweightMCAO2D(
    const std::string &inputAo2d =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/NCrossedRows/AO2D_CustomV0s.root",
    const std::string &bwFilePath =
    "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/ReweightFunc.root",
    const std::string &outputDir =
        "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/NCrossedRows/reweighted",
    const std::string &treeName = "O2mchypcands",
    bool onlyTwobody = false) {
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

  auto combinedBins = BuildCombinedBins(bwFile);

  const auto inPath = std::filesystem::path(inputAo2d);
  {
    const std::string outFilePath = outputDir + "/" + inPath.stem().string() + "_combined_reweighted" + inPath.extension().string();
    TFile outFile(outFilePath.c_str(), "RECREATE");
    if (outFile.IsZombie()) {
      throw std::runtime_error("Failed to create output AO2D: " + outFilePath);
    }

    TH1D hCentAll("hCentralityBefore", ";fCentralityFT0C;counts", 120, 0.0, 120.0);
    hCentAll.Sumw2();

    std::vector<TH1D> hBeforeByCen;
    std::vector<TH1D> hAfterByCen;
    hBeforeByCen.reserve(combinedBins.size());
    hAfterByCen.reserve(combinedBins.size());
    for (size_t i = 0; i < combinedBins.size(); ++i) {
      std::string label = combinedBins[i].label;
      std::replace(label.begin(), label.end(), '-', '_');
      hBeforeByCen.emplace_back(("hAbsGenPtBefore_" + label).c_str(), ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0);
      hAfterByCen.emplace_back(("hAbsGenPtAfter_" + label).c_str(), ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0);
      hBeforeByCen.back().Sumw2();
      hAfterByCen.back().Sumw2();
    }

    TF1 *f0 = combinedBins[0].func.get();
    TF1 *f1 = combinedBins[1].func.get();
    TF1 *f2 = combinedBins[2].func.get();
    TF1 *f3 = combinedBins[3].func.get();
    const double max0 = f0->GetMaximum();
    const double max1 = f1->GetMaximum();
    const double max2 = f2->GetMaximum();
    const double max3 = f3->GetMaximum();
    if (max0 <= 0 || max1 <= 0 || max2 <= 0 || max3 <= 0) {
      throw std::runtime_error("Combined reweight TF1 maximum <= 0");
    }

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
            ROOT::RDF::RNode rdfSelected = rdfReady;
            if (onlyTwobody) {
              const auto cols = rdfReady.GetColumnNames();
              if (std::find(cols.begin(), cols.end(), "fIsTwoBodyDecay") == cols.end()) {
                throw std::runtime_error("Missing required column 'fIsTwoBodyDecay' for onlyTwobody=true");
              }
              rdfSelected = rdfReady.Filter([](bool isTwoBody) { return isTwoBody == true; }, {"fIsTwoBodyDecay"});
            }

            auto hCent = rdfSelected.Histo1D(
                {"hCent_tmp", ";fCentralityFT0C;counts", 120, 0.0, 120.0}, "fCentralityFT0C");
            hCentAll.Add(&hCent.GetValue());

            for (size_t i = 0; i < combinedBins.size(); ++i) {
              const float cmin = combinedBins[i].minCen;
              const float cmax = combinedBins[i].maxCen;
              const bool isLast = (i + 1 == combinedBins.size());
              auto byCen = rdfSelected.Filter(
                  [cmin, cmax, isLast](float cen) {
                    if (isLast) return cen >= cmin;
                    return cen >= cmin && cen < cmax;
                  },
                  {"fCentralityFT0C"});

              auto hBefore = byCen.Histo1D(
                  {("hBefore_tmp_" + std::to_string(i)).c_str(), ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0}, "fAbsGenPt");
              auto weightedByCen = GeneralHelper::ReWeightSpectrum(byCen, combinedBins[i].func.get(), "fAbsGenPt");
              auto hAfter = weightedByCen.Histo1D(
                  {("hAfter_tmp_" + std::to_string(i)).c_str(), ";fAbsGenPt (GeV/c);counts", 120, 0.0, 12.0}, "fAbsGenPt");

              hBeforeByCen[i].Add(&hBefore.GetValue());
              hAfterByCen[i].Add(&hAfter.GetValue());
            }

            auto weighted = rdfSelected
                                .Define(
                                    "rej",
                                    [f0, f1, f2, f3, max0, max1, max2, max3](float pt, float cen) {
                                      TF1 *func = f3;
                                      double maxVal = max3;
                                      if (cen >= 0.f && cen < 10.f) {
                                        func = f0;
                                        maxVal = max0;
                                      } else if (cen >= 10.f && cen < 30.f) {
                                        func = f1;
                                        maxVal = max1;
                                      } else if (cen >= 30.f && cen < 50.f) {
                                        func = f2;
                                        maxVal = max2;
                                      }
                                      return (gRandom->Uniform() > func->Eval(pt) / maxVal) ? -1 : 1;
                                    },
                                    {"fAbsGenPt", "fCentralityFT0C"})
                                .Filter([](int rej) { return rej >= 0; }, {"rej"});

            const std::string tmp = TempFileName(outputDir, keyName + "_combined", 0);
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
    hCentAll.Write("hCentralityBefore");
    for (size_t i = 0; i < hBeforeByCen.size(); ++i) {
      std::string label = combinedBins[i].label;
      std::replace(label.begin(), label.end(), '-', '_');
      hBeforeByCen[i].Write(("hAbsGenPtBefore_" + label).c_str());
      hAfterByCen[i].Write(("hAbsGenPtAfter_" + label).c_str());
    }
    DrawCombinedQaAndSave(hCentAll, hBeforeByCen, hAfterByCen, combinedBins, outputDir, "combined");
    std::cout << "[Done] Reweighted AO2D saved to: " << outFilePath << std::endl;
  }

  std::cout << "[Done] QA plots saved to: " << outputDir << std::endl;
}
