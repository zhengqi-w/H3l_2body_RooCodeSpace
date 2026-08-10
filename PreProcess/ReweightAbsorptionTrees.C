#include <ROOT/RDataFrame.hxx>

#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TRandom3.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> GetBranchNames(TTree *tree) {
  std::vector<std::string> names;
  if (!tree || !tree->GetListOfBranches()) return names;
  names.reserve(tree->GetListOfBranches()->GetEntries());
  for (int i = 0; i < tree->GetListOfBranches()->GetEntries(); ++i) {
    auto *obj = tree->GetListOfBranches()->At(i);
    if (obj) names.emplace_back(obj->GetName());
  }
  return names;
}

std::unique_ptr<TF1> LoadFunction(const std::string &path, const std::string &name) {
  TFile file(path.c_str(), "READ");
  if (file.IsZombie()) {
    throw std::runtime_error("Cannot open reweight function file: " + path);
  }
  auto *func = dynamic_cast<TF1 *>(file.Get(name.c_str()));
  if (!func) {
    throw std::runtime_error("Cannot find TF1 " + name + " in " + path);
  }
  return std::unique_ptr<TF1>(static_cast<TF1 *>(func->Clone((name + "_clone").c_str())));
}

void DrawQa(const TH1D &before,
            const TH1D &after,
            const std::string &outPdf,
            const std::string &title) {
  auto hBefore = std::unique_ptr<TH1D>(static_cast<TH1D *>(before.Clone("h_pt_before_draw")));
  auto hAfter = std::unique_ptr<TH1D>(static_cast<TH1D *>(after.Clone("h_pt_after_draw")));
  hBefore->SetDirectory(nullptr);
  hAfter->SetDirectory(nullptr);
  hBefore->SetStats(0);
  hAfter->SetStats(0);
  hBefore->SetLineColor(kBlack);
  hBefore->SetLineWidth(2);
  hAfter->SetLineColor(kRed + 1);
  hAfter->SetLineWidth(2);

  TCanvas c("c_absorption_pt_reweight_qa", "", 900, 700);
  c.SetTicks(1, 1);
  c.SetLogy();
  hBefore->SetTitle((title + ";p_{T} (GeV/#it{c});Counts").c_str());
  const double maxY = std::max(hBefore->GetMaximum(), hAfter->GetMaximum());
  hBefore->SetMaximum(maxY > 0 ? maxY * 4.0 : 1.0);
  hBefore->Draw("hist");
  hAfter->Draw("hist same");

  TLegend leg(0.55, 0.72, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.AddEntry(hBefore.get(), "Before reweight", "l");
  leg.AddEntry(hAfter.get(), "After BlastWave 0-80", "l");
  leg.Draw();
  c.SaveAs(outPdf.c_str());
}

int FindPtBin(double pt, const std::vector<double> &ptBins) {
  for (size_t i = 0; i + 1 < ptBins.size(); ++i) {
    const bool isLast = (i + 2 == ptBins.size());
    if (pt >= ptBins[i] && (pt < ptBins[i + 1] || (isLast && pt <= ptBins[i + 1]))) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void ReweightOneFile(const std::filesystem::path &inputPath,
                     const std::filesystem::path &outputPath,
                     TF1 *func,
                     const std::vector<double> &ptBins,
                     double funcMax,
                     const std::string &treeName,
                     unsigned int seed) {
  TFile input(inputPath.c_str(), "READ");
  if (input.IsZombie()) {
    throw std::runtime_error("Cannot open input absorption file: " + inputPath.string());
  }
  auto *tree = dynamic_cast<TTree *>(input.Get(treeName.c_str()));
  if (!tree) {
    throw std::runtime_error("Cannot find tree " + treeName + " in " + inputPath.string());
  }
  const auto columns = GetBranchNames(tree);
  if (std::find(columns.begin(), columns.end(), "pt") == columns.end()) {
    throw std::runtime_error("Missing pt branch in " + inputPath.string());
  }

  float pt = 0.0f;
  float eta = 0.0f;
  float phi = 0.0f;
  float absoX = 0.0f;
  float absoY = 0.0f;
  float absoZ = 0.0f;
  int process = 0;
  int pdg = 0;
  tree->SetBranchAddress("pt", &pt);
  tree->SetBranchAddress("eta", &eta);
  tree->SetBranchAddress("phi", &phi);
  tree->SetBranchAddress("absoX", &absoX);
  tree->SetBranchAddress("absoY", &absoY);
  tree->SetBranchAddress("absoZ", &absoZ);
  tree->SetBranchAddress("process", &process);
  tree->SetBranchAddress("pdg", &pdg);

  TFile out(outputPath.c_str(), "RECREATE");
  if (out.IsZombie()) {
    throw std::runtime_error("Cannot create output absorption file: " + outputPath.string());
  }
  auto *outTree = tree->CloneTree(0);
  if (!outTree) {
    throw std::runtime_error("Failed to clone tree structure for " + inputPath.string());
  }

  TH1D hBefore("h_pt_before_reweight", ";p_{T};Counts", 120, 0.0, 12.0);
  TH1D hAfter("h_pt_after_reweight", ";p_{T};Counts", 120, 0.0, 12.0);
  hBefore.Sumw2();
  hAfter.Sumw2();

  TRandom3 rng(seed);
  const Long64_t nEntries = tree->GetEntries();
  Long64_t nInReweightBins = 0;
  Long64_t nAcceptedInBins = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    hBefore.Fill(pt);
    const int ib = FindPtBin(pt, ptBins);
    bool keep = true;
    if (ib >= 0) {
      ++nInReweightBins;
      const double y = func ? func->Eval(pt) : 0.0;
      const double prob = (std::isfinite(y) && y > 0.0 && funcMax > 0.0)
                              ? std::clamp(y / funcMax, 0.0, 1.0)
                              : 0.0;
      keep = rng.Uniform() < prob;
      if (keep) ++nAcceptedInBins;
    }
    if (keep) {
      hAfter.Fill(pt);
      outTree->Fill();
    }
  }

  out.cd();
  outTree->Write(treeName.c_str(), TObject::kOverwrite);
  hBefore.Write("h_pt_before_reweight", TObject::kOverwrite);
  hAfter.Write("h_pt_after_reweight", TObject::kOverwrite);
  auto *funcOut = static_cast<TF1 *>(func->Clone("BlastWave_0_80"));
  if (funcOut) funcOut->Write("BlastWave_0_80", TObject::kOverwrite);
  const Long64_t nOutEntries = outTree->GetEntries();
  out.Close();

  DrawQa(hBefore, hAfter,
         outputPath.parent_path().string() + "/QA_" + outputPath.stem().string() + ".pdf",
         inputPath.filename().string());

  std::cout << "[Done] " << inputPath.filename().string()
            << ": " << nEntries << " -> " << nOutEntries
            << " entries, accepted-in-bins=" << nAcceptedInBins << "/" << nInReweightBins
            << " entries, output=" << outputPath << '\n';
}

} // namespace

void ReweightAbsorptionTrees(
    const std::string &inputDir =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees",
    const std::string &reweightFuncPath =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/configs/ReweightFunc.root",
    const std::string &funcName = "BlastWave_0_80",
    const std::string &treeName = "he3candidates",
    const std::string &outputDir = "",
    const std::vector<double> &ptBins = {0.0, 10.0},
    unsigned int seed = 12345) {
  if (ROOT::IsImplicitMTEnabled()) {
    ROOT::DisableImplicitMT();
  }

  const std::filesystem::path inDir(inputDir);
  const std::filesystem::path outDir = outputDir.empty()
                                           ? (inDir / "Reweighted")
                                           : std::filesystem::path(outputDir);
  std::filesystem::create_directories(outDir);

  auto func = LoadFunction(reweightFuncPath, funcName);
  if (ptBins.size() < 2) {
    throw std::runtime_error("Need at least two pT bin edges");
  }
  const double ptMin = ptBins.front();
  const double ptMax = ptBins.back();
  const double funcMax = func->GetMaximum(ptMin, ptMax);
  if (!(funcMax > 0.0) || !std::isfinite(funcMax)) {
    throw std::runtime_error("Invalid global maximum for " + funcName);
  }

  std::vector<std::filesystem::path> inputs;
  for (const auto &entry : std::filesystem::directory_iterator(inDir)) {
    if (!entry.is_regular_file()) continue;
    const auto path = entry.path();
    if (path.extension() != ".root") continue;
    inputs.push_back(path);
  }
  std::sort(inputs.begin(), inputs.end());

  std::cout << "[Info] Reweight " << inputs.size()
            << " absorption tree files with " << funcName
            << " using global accept-reject in pT range";
  for (double edge : ptBins) std::cout << ' ' << edge;
  std::cout << " (global max=" << funcMax << ")"
            << " into " << outDir << '\n';
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &path = inputs[i];
    const auto outPath = outDir / (path.stem().string() + "_BlastWave_0_80_reweighted.root");
    ReweightOneFile(path, outPath, func.get(), ptBins, funcMax, treeName, seed + static_cast<unsigned int>(i * 1009));
  }
}
