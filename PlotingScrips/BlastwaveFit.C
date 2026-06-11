// Simple macro to fit h_corrected_counts with a Blast-Wave (BGBW) function
// and store the TF1 back into the input ROOT file. Also produces a PDF
// overlay of the data and fit.

#include <memory>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TF1.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH1.h"
#include "TSystem.h"

// Include implementation so the class is available without an external library.
#include "../include/AliPWGFunc.h"
#include "../include/AliPWGFunc.cxx"

namespace {
std::string MakePdfName(const char *filePath) {
  std::string base = gSystem->BaseName(filePath ? filePath : "output.root");
  const auto dotPos = base.rfind('.');
  if (dotPos != std::string::npos) {
    base = base.substr(0, dotPos);
  }
  std::string dir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/BlastwaveFit";
  return dir + "/" + base + "_BGBW_fit.pdf";
}

std::pair<std::string, std::string> SplitPath(const std::string &full) {
  const auto pos = full.rfind('/');
  if (pos == std::string::npos) {
    return {"", full};
  }
  return {full.substr(0, pos), full.substr(pos + 1)};
}
} // namespace

void BlastwaveFit(
    const std::vector<std::string> &filePaths = {
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen0-10/pt_analysis_pbpb.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen10-30/pt_analysis_pbpb.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen30-50/pt_analysis_pbpb.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_LHC25g11_G4list/cen50-80/pt_analysis_pbpb.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_backup/BdtSpectrum_AllCent/cen0-80/pt_analysis_pbpb.root"
    },
    const char *histPath = "std/h_corrected_counts",
    double mass = 2.991, double beta = 0.6,
    double temp = 0.12, double n = 1.0,
    double norm = 1.0, const char *pdfOut = "") {
  if (!histPath) {
    Error("FitBGBW", "Invalid histogram path");
    return;
  }

  for (const auto &filePathStr : filePaths) {
    const char *filePath = filePathStr.c_str();
    const std::string pdfName = (pdfOut && std::string(pdfOut).size()) ? pdfOut : MakePdfName(filePath);
    gSystem->mkdir(gSystem->DirName(pdfName.c_str()), true);
    const std::string histPathStr(histPath);
    const auto [dirPath, objName] = SplitPath(histPathStr);

    std::unique_ptr<TFile> f(TFile::Open(filePath, "UPDATE"));
    if (!f || f->IsZombie()) {
      Error("FitBGBW", "Cannot open %s", filePath);
      continue;
    }

    TH1 *hRaw = nullptr;
    f->GetObject(histPath, hRaw);
    if (!hRaw) {
      Error("FitBGBW", "Histogram %s not found in %s", histPath, filePath);
      continue;
    }
    std::unique_ptr<TH1> h(static_cast<TH1 *>(hRaw->Clone((objName + "_clone").c_str())));
    h->SetDirectory(nullptr);

    AliPWGFunc bwHelper;
    bwHelper.SetVarType(AliPWGFunc::kdNdpt);
    TF1 *fBW = bwHelper.GetBGBW(mass, beta, temp, n, norm, ("fBGBW_" + objName).c_str());
    // Start from user-provided seeds; mass is fixed in GetBGBW
    fBW->SetParameter(1, beta);
    fBW->SetParameter(2, temp);
    fBW->SetParameter(3, n);
    fBW->SetParameter(4, norm);

    // const int first = h->FindFirstBinAbove(0.0);
    // const int last = h->FindLastBinAbove(0.0);
    // if (first > 0 && last >= first) {
    //   const double xmin = h->GetXaxis()->GetBinLowEdge(first);
    //   const double xmax = h->GetXaxis()->GetBinUpEdge(last);
    //   fBW->SetRange(xmin, xmax);
    // }
    fBW->SetRange(0, 10);

    h->Fit(fBW, "RQ");

    TDirectory *targetDir = dirPath.empty() ? f.get() : f->GetDirectory(dirPath.c_str());
    if (!targetDir) {
      targetDir = f.get();
    }
    targetDir->cd();
    fBW->Write(fBW->GetName(), TObject::kOverwrite);

    gStyle->SetOptStat(0);
    auto c = std::make_unique<TCanvas>("c_bw_fit", "BGBW fit", 900, 700);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(1.0);
    h->SetLineColor(kBlack);
    h->Draw("E");
    fBW->SetLineColor(kRed + 1);
    fBW->SetLineWidth(2);
    fBW->Draw("SAME");

    auto leg = std::make_unique<TLegend>(0.55, 0.68, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->AddEntry(h.get(), "corrected counts", "lep");
    leg->AddEntry(fBW, "BGBW fit", "l");
    leg->Draw();

    TLatex lat;
    lat.SetNDC(true);
    lat.SetTextSize(0.035);
    lat.DrawLatex(0.55, 0.62, Form("#beta = %.3f #pm %.3f", fBW->GetParameter(1), fBW->GetParError(1)));
    lat.DrawLatex(0.55, 0.58, Form("T = %.3f #pm %.3f", fBW->GetParameter(2), fBW->GetParError(2)));
    lat.DrawLatex(0.55, 0.54, Form("n = %.3f #pm %.3f", fBW->GetParameter(3), fBW->GetParError(3)));
    lat.DrawLatex(0.55, 0.50, Form("norm = %.3g #pm %.3g", fBW->GetParameter(4), fBW->GetParError(4)));

    c->SaveAs(pdfName.c_str());

    targetDir->cd();
    c->Write("c_bw_fit", TObject::kOverwrite);

    f->Write();
    f->Close();
  }
}
