#include <RooFit.h>
#include <RooRealVar.h>
#include <RooExponential.h>
#include <RooSimultaneous.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooArgSet.h>
#include <RooAbsPdf.h>
#include <RooPlot.h>
#include <RooFormulaVar.h>
#include <RooExtendPdf.h>
#include <RooChi2Var.h>
#include <RooMinimizer.h>
#include <regex>
#include <cmath>
#include <TMath.h>

#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <xgboost/c_api.h>
#include "../Tools/GeneralHelper.hpp"

using namespace GeneralHelper;
static BoosterHandle g_booster = nullptr;
// Generic ComputeBDT that accepts a feature vector of arbitrary length.
static float ComputeBDT(const std::vector<float> &feats){
  if(!g_booster) return 0.0f;
  if(feats.empty()) return 0.0f;
  // copy to float buffer
  std::vector<float> data;
  data.reserve(feats.size());
  for(float v: feats) data.push_back(static_cast<float>(v));

  DMatrixHandle dmat = nullptr;
  // rows=1, cols=feats.size(), missing=-1
  if(XGDMatrixCreateFromMat(data.data(), 1, static_cast<unsigned long>(data.size()), -1.0f, &dmat) != 0){
    return 0.0f;
  }
  bst_ulong out_len = 0;
  const float* out_result = nullptr;
  if(XGBoosterPredict(g_booster, dmat, 0, 0, 0, &out_len, &out_result) != 0){
    XGDMatrixFree(dmat);
    return 0.0f;
  }
  float outv = (out_result && out_len>0) ? out_result[0] : 0.0f;
  XGDMatrixFree(dmat);
  return outv;
}

void Test() {
    std::vector<std::string> training_variables = {"fDcaV0Daug", "fDcaHe", "fDcaPi", "fCosPA", "fNSigmaHe"};
    std::string model_json = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/MLProcess/TrainedModels/Model_BDT_2_3.5_ct_1_3.json";
    if(XGBoosterCreate(NULL, 0, &g_booster) != 0){
      printf("Failed to create XGBoost booster\n");
    } else {
      if(XGBoosterLoadModel(g_booster, model_json.c_str()) != 0){
        printf("Failed to load XGBoost model: %s\n", model_json.c_str());
        XGBoosterFree(g_booster);
        g_booster = nullptr;
      } else {
        printf("Loaded XGBoost model: %s\n", model_json.c_str());
      }
    }
    TFile* f = TFile::Open("/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s.root", "READ");
    TChain chain("O2hypcands");
    fillChainFromAO2D(chain, f);
    ROOT::RDataFrame rdf(chain);
    auto rdf_corrected = CorrectAndConvertRDF(rdf);
    // Build column list from training_variables and use a vector-based ComputeBDT.
    std::vector<std::string> cols;
    cols.reserve(training_variables.size());
    for (const auto &v : training_variables) cols.push_back(v);

    // Define a new column 'bdt_score' by passing the requested columns as a vector.
    // The functor receives a std::vector<float> containing the requested column values in order.
    auto rdf_bdt = rdf_corrected.Define("bdt_score",
      [](const std::vector<float> &feats){ return ComputeBDT(feats); },
      cols);
    rdf_selcted = rdf_bdt.Filter("bdt_score > 0.5");
    TH1F* h_mass = rdf_selcted.Histo1D({"h_mass", "Mass after BDT cut;Mass (GeV/c^{2});Counts", 30, 2.96, 3.04}, "fMassH3L").GetValue();
    TCanvas* c1 = new TCanvas("c1", "Mass after BDT cut", 800, 600);
    h_mass->Draw();
    c1->SaveAs("mass_after_bdt_cut.pdf");
}
