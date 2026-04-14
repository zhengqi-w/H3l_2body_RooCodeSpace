#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TKey.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h"
#include "TObject.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TSystem.h"

#include "../include/AliPWGFunc.h"
#include "../include/AliPWGFunc.cxx"

namespace {
TF1 *FindTF1ByPattern(TFile *f, const std::string &pattern) {
  if (!f) return nullptr;

  // Try direct common naming first.
  const std::vector<std::string> directNames = {
      "BlastWave_H3L_" + pattern,
      "BlastWave_H3l_" + pattern,
      "fBGBW_" + pattern,
      pattern};

  for (const auto &name : directNames) {
    auto *obj = f->Get(name.c_str());
    if (obj && obj->InheritsFrom(TF1::Class())) {
      return static_cast<TF1 *>(obj);
    }
  }

  // Fall back to scanning all top-level keys.
  TIter next(f->GetListOfKeys());
  TKey *key = nullptr;
  while ((key = static_cast<TKey *>(next()))) {
    const char *className = key->GetClassName();
    if (!className) continue;
    TClass *cl = TClass::GetClass(className);
    if (!cl || !cl->InheritsFrom(TF1::Class())) continue;

    std::string keyName = key->GetName();
    if (keyName.find(pattern) != std::string::npos) {
      auto *obj = key->ReadObj();
      if (obj && obj->InheritsFrom(TF1::Class())) {
        return static_cast<TF1 *>(obj);
      }
    }
  }

  return nullptr;
}
}  // namespace

void BuildReweightFunc(
    const char *he3File = "/Users/zhengqingwang/alice/data/h3l_spec_run2/He3_50_90.root",
    const char *he3HistPath = "Hist1D_y1",
  const char *he3GraphPath = "Graph1D_y1",
    const char *h3lBwFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/H3L_BWFit_Run3_23.root",
    const char *outFile = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/ReweightFunc.root") {

  TFile *fHe3 = TFile::Open(he3File, "READ");
  if (!fHe3 || fHe3->IsZombie()) {
    Error("BuildReweightFunc", "Cannot open He3 file: %s", he3File);
    return;
  }

  TH1 *hHe3Raw = nullptr;
  fHe3->GetObject(he3HistPath, hHe3Raw);
  if (!hHe3Raw) {
    Error("BuildReweightFunc", "Cannot find histogram %s in %s", he3HistPath, he3File);
    fHe3->Close();
    return;
  }

  TGraphAsymmErrors *gHe3Raw = nullptr;
  fHe3->GetObject(he3GraphPath, gHe3Raw);

  TH1 *hHe3 = static_cast<TH1 *>(hHe3Raw->Clone("hHe3_forFit"));
  hHe3->SetDirectory(nullptr);

  TGraphAsymmErrors *gHe3 = nullptr;
  if (gHe3Raw) {
    gHe3 = static_cast<TGraphAsymmErrors *>(gHe3Raw->Clone("gHe3_forFit"));
  }

  AliPWGFunc bwHelper;
  bwHelper.SetVarType(AliPWGFunc::kdNdpt);

  // He3 mass in GeV/c^2.
  TF1 *fHe3BW = bwHelper.GetBGBW(2.80839, 0.60, 0.12, 1.0, 1.0, "BlastWave_50_90");
  fHe3BW->SetParameter(1, 0.60);
  fHe3BW->SetParameter(2, 0.12);
  fHe3BW->SetParameter(3, 1.0);
  fHe3BW->SetParameter(4, 1.0);

  fHe3BW->SetRange(0.0, 10.0);

  int fitStatus = 0;
  if (gHe3) {
    fitStatus = gHe3->Fit(fHe3BW, "RQ");
  } else {
    fitStatus = hHe3->Fit(fHe3BW, "RQ0");
    std::cout << "[Warn] Graph " << he3GraphPath << " not found, fallback to histogram fit." << std::endl;
  }
  std::cout << "[Info] He3 fit status = " << fitStatus << std::endl;
  std::cout << "[Info] He3 BW parameters: beta=" << fHe3BW->GetParameter(1)
            << " T=" << fHe3BW->GetParameter(2)
            << " n=" << fHe3BW->GetParameter(3)
            << " norm=" << fHe3BW->GetParameter(4) << std::endl;

  // Save a quick QA plot for He fit.
  gStyle->SetOptStat(0);
  TCanvas cHe("cHeBWFit", "He BW fit", 900, 700);
  if (gHe3) {
    gHe3->SetMarkerStyle(20);
    gHe3->SetMarkerSize(1.0);
    gHe3->SetLineColor(kBlack);
    gHe3->SetTitle(";p_{T} (GeV/c);dN/dp_{T}");
    gHe3->Draw("AP");
  } else {
    hHe3->SetMarkerStyle(20);
    hHe3->SetMarkerSize(1.0);
    hHe3->SetLineColor(kBlack);
    hHe3->Draw("E");
  }
  fHe3BW->SetLineColor(kRed + 1);
  fHe3BW->SetLineWidth(2);
  fHe3BW->Draw("SAME");
  TLegend leg(0.55, 0.72, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  if (gHe3) {
    leg.AddEntry(gHe3, "He3 Graph1D_y1", "lep");
  } else {
    leg.AddEntry(hHe3, "He3 Hist1D_y1", "lep");
  }
  leg.AddEntry(fHe3BW, "BlastWave fit", "l");
  leg.Draw();
  TString qaPdf = TString::Format("%s/He3_50_90_BlastWave_QA.pdf", gSystem->DirName(outFile));
  cHe.SaveAs(qaPdf.Data());

  TFile *fH3L = TFile::Open(h3lBwFile, "READ");
  if (!fH3L || fH3L->IsZombie()) {
    Error("BuildReweightFunc", "Cannot open H3L BW file: %s", h3lBwFile);
    fHe3->Close();
    return;
  }

  TFile out(outFile, "RECREATE");
  if (out.IsZombie()) {
    Error("BuildReweightFunc", "Cannot create output file: %s", outFile);
    fH3L->Close();
    fHe3->Close();
    return;
  }

  out.cd();
  auto *he3Clone = static_cast<TF1 *>(fHe3BW->Clone("BlastWave_50_90"));
  he3Clone->SetRange(0.0, 10.0);
  he3Clone->Write("BlastWave_50_90", TObject::kOverwrite);

  const std::vector<std::string> centPatterns = {"0_10", "10_30", "30_50"};
  for (const auto &pat : centPatterns) {
    TF1 *src = FindTF1ByPattern(fH3L, pat);
    if (!src) {
      std::cout << "[Warn] No TF1 matched pattern " << pat << " in " << h3lBwFile << std::endl;
      continue;
    }
    const std::string outName = "BlastWave_" + pat;
    // Rebuild BGBW from source parameters instead of direct cloning.
    // This avoids a flat-looking segment at low pT when extending range to [0, 10].
    TF1 *rebuilt = bwHelper.GetBGBW(2.991, src->GetParameter(1), src->GetParameter(2),
                                    src->GetParameter(3), src->GetParameter(4), outName.c_str());
    rebuilt->SetParameter(1, src->GetParameter(1));
    rebuilt->SetParameter(2, src->GetParameter(2));
    rebuilt->SetParameter(3, src->GetParameter(3));
    rebuilt->SetParameter(4, src->GetParameter(4));
    rebuilt->SetRange(0.0, 10.0);
    rebuilt->SetNpx(1000);
    rebuilt->Write(outName.c_str(), TObject::kOverwrite);
    std::cout << "[Info] Wrote " << outName << " rebuilt from " << src->GetName() << std::endl;
  }

  out.Close();
  fH3L->Close();
  fHe3->Close();

  std::cout << "[Done] Output: " << outFile << std::endl;
}
