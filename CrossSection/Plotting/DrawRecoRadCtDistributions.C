#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TKey.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TH1D.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Sample {
  const char* label;
  const char* path;
  int color;
};

int AddDfTrees(TChain& chain, const char* path, const char* treeName)
{
  TFile file(path, "READ");
  if (file.IsZombie()) {
    std::cerr << "cannot open " << path << std::endl;
    return 0;
  }

  int nDirs = 0;
  for (auto* obj : *file.GetListOfKeys()) {
    auto* key = dynamic_cast<TKey*>(obj);
    if (!key) {
      continue;
    }
    std::string dirName = key->GetName();
    if (dirName.rfind("DF_", 0) != 0) {
      continue;
    }

    const std::string treePath = dirName + "/" + treeName;
    if (file.Get(treePath.c_str())) {
      chain.Add((std::string(path) + "/" + treePath).c_str());
      ++nDirs;
    }
  }
  return nDirs;
}

ROOT::RDF::RNode DefineRecoKinematics(ROOT::RDF::RNode node)
{
  return node.Define("reco_px", "fPtHe3*cos(fPhiHe3) + fPtPi*cos(fPhiPi)")
      .Define("reco_py", "fPtHe3*sin(fPhiHe3) + fPtPi*sin(fPhiPi)")
      .Define("reco_pz", "fPtHe3*sinh(fEtaHe3) + fPtPi*sinh(fEtaPi)")
      .Define("reco_p", "sqrt(reco_px*reco_px + reco_py*reco_py + reco_pz*reco_pz)")
      .Define("reco_dec_rad", "sqrt((fXDecVtx+fXPrimVtx)*(fXDecVtx+fXPrimVtx) + "
                                  "(fYDecVtx+fYPrimVtx)*(fYDecVtx+fYPrimVtx))")
      .Define("reco_dec_len", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)")
      .Define("reco_ct", "reco_dec_len * 2.99131 / reco_p");
}

std::string RadTag(double radMin, double radMax)
{
  return Form("rad_%g_%g", radMin, radMax);
}

TH1D* MakeCtHist(const std::string& name)
{
  auto* hist = new TH1D(name.c_str(), ";reconstructed ct (cm);Candidates", 140, 0., 40.);
  hist->SetCanExtend(TH1::kXaxis);
  hist->SetDirectory(nullptr);
  return hist;
}

void ApplyAutoXRange(TH1D* hist)
{
  const int firstBin = hist->FindFirstBinAbove(0., 1);
  const int lastBin = hist->FindLastBinAbove(0., 1);
  if (firstBin <= 0 || lastBin < firstBin) {
    return;
  }

  const int firstShown = std::max(1, firstBin - 1);
  const int lastShown = std::min(hist->GetNbinsX(), lastBin + 1);
  hist->GetXaxis()->SetRange(firstShown, lastShown);
}

void DrawOneSet(const std::vector<Sample>& samples,
                const char* treeName,
                bool isMc,
                double radMin,
                double radMax,
                const std::string& outputStem)
{
  std::vector<TH1D*> periodHists;
  std::vector<std::string> labels;
  TH1D* mergedHist = nullptr;
  Long64_t mergedCount = 0;
  Long64_t mergedCtGe40 = 0;
  double mergedMeanNumerator = 0.;
  double mergedCtMin = std::numeric_limits<double>::infinity();
  double mergedCtMax = -std::numeric_limits<double>::infinity();

  std::cout << outputStem << ": "
            << radMin << " < reconstructed DecayRad < " << radMax << '\n';

  mergedHist = MakeCtHist(Form("h_merged_%s", outputStem.c_str()));
  for (const auto& sample : samples) {
    TChain chain(treeName);
    const int nDirs = AddDfTrees(chain, sample.path, treeName);
    ROOT::RDataFrame df(chain);
    ROOT::RDF::RNode base = DefineRecoKinematics(ROOT::RDF::RNode(df));
    if (isMc) {
      base = base.Filter("fIsReco == true");
    }

    auto selected = base.Filter(
        Form("reco_dec_rad > %.17g && reco_dec_rad < %.17g", radMin, radMax));
    auto* cloned = MakeCtHist(Form("hc_%s_%s", outputStem.c_str(), sample.label));
    Long64_t n = 0;
    Long64_t nCtGe40 = 0;
    double ctSum = 0.;
    double ctMin = std::numeric_limits<double>::infinity();
    double ctMax = -std::numeric_limits<double>::infinity();

    selected.Foreach(
        [&](double ct) {
          ++n;
          ctSum += ct;
          ctMin = std::min(ctMin, ct);
          ctMax = std::max(ctMax, ct);
          if (ct >= 40.) {
            ++nCtGe40;
          }
          cloned->Fill(ct);
          mergedHist->Fill(ct);
        },
        {"reco_ct"});

    const double ctMean = n ? ctSum / n : 0.;

    std::cout << sample.label << " dfDirs=" << nDirs
              << " rad=" << n
              << " ct_min=" << (n ? ctMin : 0.)
              << " ct_max=" << (n ? ctMax : 0.)
              << " ct_mean=" << ctMean
              << " ct_ge40=" << nCtGe40 << '\n';

    cloned->SetLineColor(sample.color);
    cloned->SetLineWidth(3);
    periodHists.push_back(cloned);
    labels.emplace_back(sample.label);

    if (n) {
      mergedCtMin = std::min(mergedCtMin, ctMin);
      mergedCtMax = std::max(mergedCtMax, ctMax);
    }
    mergedCount += n;
    mergedCtGe40 += nCtGe40;
    mergedMeanNumerator += ctSum;
  }

  std::cout << "MERGED rad=" << mergedCount
            << " ct_mean=" << (mergedCount ? mergedMeanNumerator / mergedCount : 0.)
            << " ct_ge40=" << mergedCtGe40 << '\n';

  TCanvas canvas(Form("c_%s", outputStem.c_str()), "", 1050, 780);
  canvas.SetLeftMargin(0.12);
  canvas.SetRightMargin(0.04);
  canvas.SetBottomMargin(0.11);
  canvas.SetLogy();

  mergedHist->SetTitle(isMc ? "No-G4list reweighted MC: reconstructed ct"
                            : "Data: reconstructed ct");
  mergedHist->SetLineColor(kBlack);
  mergedHist->SetLineWidth(4);
  mergedHist->SetMinimum(0.7);
  ApplyAutoXRange(mergedHist);
  mergedHist->SetMaximum(mergedHist->GetMaximum() * 5.);
  mergedHist->Draw("hist");
  for (auto* hist : periodHists) {
    hist->Draw("hist same");
  }
  mergedHist->Draw("hist same");

  TLegend legend(0.55, 0.61, 0.91, 0.86);
  legend.SetBorderSize(0);
  legend.SetFillStyle(0);
  legend.SetTextSize(0.031);
  legend.AddEntry(mergedHist, Form("Merged: %lld", mergedCount), "l");
  for (size_t i = 0; i < periodHists.size(); ++i) {
    legend.AddEntry(periodHists[i], Form("%s: %.0f", labels[i].c_str(), periodHists[i]->Integral()), "l");
  }
  legend.Draw();

  TLatex text;
  text.SetNDC();
  text.SetTextSize(0.033);
  text.DrawLatex(0.16, 0.86, isMc ? "No-G4list reweighted MC, fIsReco == true"
                                  : "Original AO2D Data, no SnapShotsData");
  text.DrawLatex(0.16, 0.81,
                 Form("%.0f < reconstructed DecayRad < %.0f, reconstructed ct", radMin, radMax));
  text.DrawLatex(0.16, 0.76,
                 Form("Merged: %lld, mean ct: %.2f cm",
                      mergedCount,
                      mergedCount ? mergedMeanNumerator / mergedCount : 0.));
  text.DrawLatex(0.16, 0.71,
                 Form("ct min/max: %.3g / %.3g cm",
                      mergedCount ? mergedCtMin : 0.,
                      mergedCount ? mergedCtMax : 0.));

  const std::string outDir = "ROOTWorkFlow/Outputs/StandAloneChecks/MergedRawDataDecayRadCtCheck";
  canvas.SaveAs((outDir + "/" + outputStem + ".pdf").c_str());
}

struct BinSummary {
  std::vector<TH1D*> periodHists;
  std::vector<std::string> labels;
  TH1D* mergedHist = nullptr;
  Long64_t count = 0;
  Long64_t ctGe40 = 0;
  double meanNumerator = 0.;
  double ctMin = std::numeric_limits<double>::infinity();
  double ctMax = -std::numeric_limits<double>::infinity();
};

void SaveBinCanvas(BinSummary& summary,
                   bool isMc,
                   double radMin,
                   double radMax,
                   const std::string& outputStem)
{
  std::cout << "MERGED " << outputStem
            << " rad=" << summary.count
            << " ct_min=" << (summary.count ? summary.ctMin : 0.)
            << " ct_max=" << (summary.count ? summary.ctMax : 0.)
            << " ct_mean=" << (summary.count ? summary.meanNumerator / summary.count : 0.)
            << " ct_ge40=" << summary.ctGe40 << '\n';

  TCanvas canvas(Form("c_%s", outputStem.c_str()), "", 1050, 780);
  canvas.SetLeftMargin(0.12);
  canvas.SetRightMargin(0.04);
  canvas.SetBottomMargin(0.11);
  canvas.SetLogy();

  summary.mergedHist->SetTitle(isMc ? "No-G4list reweighted MC: reconstructed ct"
                                    : "Data: reconstructed ct");
  summary.mergedHist->SetLineColor(kBlack);
  summary.mergedHist->SetLineWidth(4);
  summary.mergedHist->SetMinimum(0.7);
  ApplyAutoXRange(summary.mergedHist);
  summary.mergedHist->SetMaximum(summary.mergedHist->GetMaximum() * 5.);
  summary.mergedHist->Draw("hist");
  for (auto* hist : summary.periodHists) {
    hist->Draw("hist same");
  }
  summary.mergedHist->Draw("hist same");

  TLegend legend(0.55, 0.61, 0.91, 0.86);
  legend.SetBorderSize(0);
  legend.SetFillStyle(0);
  legend.SetTextSize(0.031);
  legend.AddEntry(summary.mergedHist, Form("Merged: %lld", summary.count), "l");
  for (size_t i = 0; i < summary.periodHists.size(); ++i) {
    legend.AddEntry(summary.periodHists[i],
                    Form("%s: %.0f", summary.labels[i].c_str(), summary.periodHists[i]->Integral()),
                    "l");
  }
  legend.Draw();

  TLatex text;
  text.SetNDC();
  text.SetTextSize(0.033);
  text.DrawLatex(0.16, 0.86, isMc ? "No-G4list reweighted MC, fIsReco == true"
                                  : "Original AO2D Data, no SnapShotsData");
  text.DrawLatex(0.16, 0.81,
                 Form("%g < reconstructed DecayRad < %g, reconstructed ct", radMin, radMax));
  text.DrawLatex(0.16, 0.76,
                 Form("Merged: %lld, mean ct: %.2f cm",
                      summary.count,
                      summary.count ? summary.meanNumerator / summary.count : 0.));
  text.DrawLatex(0.16, 0.71,
                 Form("ct min/max: %.3g / %.3g cm",
                      summary.count ? summary.ctMin : 0.,
                      summary.count ? summary.ctMax : 0.));

  const std::string outDir = "ROOTWorkFlow/Outputs/StandAloneChecks/MergedRawDataDecayRadCtCheck";
  canvas.SaveAs((outDir + "/" + outputStem + ".pdf").c_str());
}

void DrawMultipleBinsOneSet(const std::vector<Sample>& samples,
                            const char* treeName,
                            bool isMc,
                            const std::vector<double>& radBinEdges,
                            const std::string& outputPrefix)
{
  const size_t nBins = radBinEdges.size() - 1;
  std::vector<BinSummary> summaries(nBins);
  for (size_t iBin = 0; iBin < nBins; ++iBin) {
    summaries[iBin].mergedHist = MakeCtHist(Form("h_merged_%s_%zu", outputPrefix.c_str(), iBin));
  }

  for (const auto& sample : samples) {
    TChain chain(treeName);
    const int nDirs = AddDfTrees(chain, sample.path, treeName);
    ROOT::RDataFrame df(chain);
    ROOT::RDF::RNode base = DefineRecoKinematics(ROOT::RDF::RNode(df));
    if (isMc) {
      base = base.Filter("fIsReco == true");
    }

    std::vector<TH1D*> hists;
    std::vector<Long64_t> counts(nBins, 0);
    std::vector<Long64_t> ctGe40(nBins, 0);
    std::vector<double> ctSums(nBins, 0.);
    std::vector<double> ctMins(nBins, std::numeric_limits<double>::infinity());
    std::vector<double> ctMaxs(nBins, -std::numeric_limits<double>::infinity());
    hists.reserve(nBins);
    for (size_t iBin = 0; iBin < nBins; ++iBin) {
      auto* hist = MakeCtHist(Form("h_%s_%s_%zu", outputPrefix.c_str(), sample.label, iBin));
      hists.push_back(hist);
    }

    base.Foreach(
        [&](float decRad, double ct) {
          for (size_t iBin = 0; iBin < nBins; ++iBin) {
            if (decRad > radBinEdges[iBin] && decRad < radBinEdges[iBin + 1]) {
              ++counts[iBin];
              ctSums[iBin] += ct;
              ctMins[iBin] = std::min(ctMins[iBin], ct);
              ctMaxs[iBin] = std::max(ctMaxs[iBin], ct);
              if (ct >= 40.) {
                ++ctGe40[iBin];
              }
              hists[iBin]->Fill(ct);
              summaries[iBin].mergedHist->Fill(ct);
              break;
            }
          }
        },
        {"reco_dec_rad", "reco_ct"});

    for (size_t iBin = 0; iBin < nBins; ++iBin) {
      const Long64_t count = counts[iBin];
      const double mean = count ? ctSums[iBin] / count : 0.;
      std::cout << sample.label << " dfDirs=" << nDirs
                << " " << radBinEdges[iBin] << "<rad<" << radBinEdges[iBin + 1]
                << " rad=" << count
                << " ct_min=" << (count ? ctMins[iBin] : 0.)
                << " ct_max=" << (count ? ctMaxs[iBin] : 0.)
                << " ct_mean=" << mean
                << " ct_ge40=" << ctGe40[iBin] << '\n';

      auto& summary = summaries[iBin];
      hists[iBin]->SetLineColor(sample.color);
      hists[iBin]->SetLineWidth(3);
      summary.periodHists.push_back(hists[iBin]);
      summary.labels.emplace_back(sample.label);
      summary.count += count;
      summary.ctGe40 += ctGe40[iBin];
      summary.meanNumerator += ctSums[iBin];
      if (count) {
        summary.ctMin = std::min(summary.ctMin, ctMins[iBin]);
        summary.ctMax = std::max(summary.ctMax, ctMaxs[iBin]);
      }
    }
  }

  for (size_t iBin = 0; iBin < nBins; ++iBin) {
    const std::string tag = RadTag(radBinEdges[iBin], radBinEdges[iBin + 1]);
    SaveBinCanvas(summaries[iBin], isMc, radBinEdges[iBin], radBinEdges[iBin + 1],
                  outputPrefix + "_" + tag + "_logy");
  }
}

} // namespace

void DrawRecoRadCtDistributions(double radMin = 25., double radMax = 30.)
{
  const std::string outDir = "ROOTWorkFlow/Outputs/StandAloneChecks/MergedRawDataDecayRadCtCheck";
  std::filesystem::create_directories(outDir);
  gStyle->SetOptStat(0);

  const std::vector<Sample> dataSamples = {
      {"LHC23 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/data/NCrossedRows/AO2D_CustomV0s_HadronPID.root",
       kAzure + 1},
      {"LHC24 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/data/AO2D_CustomV0s_HadronPID.root",
       kOrange + 7},
      {"LHC25 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/data/AO2D_CustomV0s_HadronPID.root",
       kGreen + 2},
  };

  const std::vector<Sample> mcSamples = {
      {"LHC23 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
       kAzure + 1},
      {"LHC24 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5/reweighted/AO2D_combined_reweighted.root",
       kOrange + 7},
      {"LHC25 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6/reweighted/AO2D_combined_reweighted.root",
       kGreen + 2},
  };

  const std::string tag = RadTag(radMin, radMax);
  DrawOneSet(dataSamples, "O2hypcands", false, radMin, radMax, "data_ct_recodecayrad_" + tag + "_logy");
  DrawOneSet(mcSamples, "O2mchypcands", true, radMin, radMax, "mc_nog4_reco_ct_recodecayrad_" + tag + "_logy");
}

void DrawRecoRadCtDistributions(const std::vector<double>& radBinEdges)
{
  if (radBinEdges.size() < 2) {
    std::cerr << "Need at least two reconstructed DecayRad bin edges." << std::endl;
    return;
  }

  for (size_t iBin = 0; iBin + 1 < radBinEdges.size(); ++iBin) {
    if (radBinEdges[iBin + 1] <= radBinEdges[iBin]) {
      std::cerr << "Skip non-increasing DecayRad interval: "
                << radBinEdges[iBin] << " to " << radBinEdges[iBin + 1] << std::endl;
      continue;
    }
  }

  const std::vector<Sample> dataSamples = {
      {"LHC23 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/data/NCrossedRows/AO2D_CustomV0s_HadronPID.root",
       kAzure + 1},
      {"LHC24 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/data/AO2D_CustomV0s_HadronPID.root",
       kOrange + 7},
      {"LHC25 Data",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/data/AO2D_CustomV0s_HadronPID.root",
       kGreen + 2},
  };
  const std::vector<Sample> mcSamples = {
      {"LHC23 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
       kAzure + 1},
      {"LHC24 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5/reweighted/AO2D_combined_reweighted.root",
       kOrange + 7},
      {"LHC25 MC",
       "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6/reweighted/AO2D_combined_reweighted.root",
       kGreen + 2},
  };

  DrawMultipleBinsOneSet(dataSamples, "O2hypcands", false, radBinEdges, "data_ct_recodecayrad");
  DrawMultipleBinsOneSet(mcSamples, "O2mchypcands", true, radBinEdges, "mc_nog4_reco_ct_recodecayrad");
}
