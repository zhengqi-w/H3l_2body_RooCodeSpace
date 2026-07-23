#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TH1.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include "../include/AliPWGFunc.h"
#include "../include/AliPWGFunc.cxx"

namespace {

constexpr double kHe3Mass = 2.8083916;

struct SpectrumItem {
  std::string species;
  std::string cent;
  std::string fileName;
  int color;
  int marker;
};

struct FitResult {
  std::unique_ptr<TGraphAsymmErrors> graph;
  std::unique_ptr<TH1> hist;
  std::unique_ptr<TF1> func;
  std::string species;
  std::string cent;
  int color = kBlack;
  int marker = 20;
  int status = 999;
  double chi2 = 0.0;
  int ndf = 0;
  double chi2ndf = 0.0;
};

std::vector<AliPWGFunc *> gAliPwgOwners;

std::string Sanitize(const std::string &s)
{
  std::string out = s;
  std::replace(out.begin(), out.end(), '-', '_');
  std::replace(out.begin(), out.end(), '%', '_');
  std::replace(out.begin(), out.end(), ' ', '_');
  return out;
}

std::string DisplayCent(const std::string &s)
{
  std::string out = s;
  std::replace(out.begin(), out.end(), '_', '-');
  return out;
}

TDirectory *FindFirstDirectory(TFile &file)
{
  TIter next(file.GetListOfKeys());
  while (auto *obj = next()) {
    auto *key = dynamic_cast<TKey *>(obj);
    if (!key) continue;
    TObject *readObj = key->ReadObj();
    auto *dir = dynamic_cast<TDirectory *>(readObj);
    if (dir) return dir;
    delete readObj;
  }
  return nullptr;
}

std::unique_ptr<TGraphAsymmErrors> LoadGraph(TDirectory *dir, const std::string &name)
{
  if (!dir) return nullptr;
  auto *raw = dynamic_cast<TGraphAsymmErrors *>(dir->Get("Graph1D_y1"));
  if (!raw) return nullptr;
  auto out = std::unique_ptr<TGraphAsymmErrors>(static_cast<TGraphAsymmErrors *>(raw->Clone(name.c_str())));
  return out;
}

std::unique_ptr<TH1> LoadHist(TDirectory *dir, const std::string &name)
{
  if (!dir) return nullptr;
  auto *raw = dynamic_cast<TH1 *>(dir->Get("Hist1D_y1"));
  if (!raw) return nullptr;
  auto out = std::unique_ptr<TH1>(static_cast<TH1 *>(raw->Clone(name.c_str())));
  out->SetDirectory(nullptr);
  return out;
}

std::pair<double, double> GetXRange(const TGraphAsymmErrors *g, const TH1 *h)
{
  double xmin = 1e30;
  double xmax = -1e30;
  if (g && g->GetN() > 0) {
    for (int i = 0; i < g->GetN(); ++i) {
      double x = 0.0, y = 0.0;
      g->GetPoint(i, x, y);
      if (y <= 0.0) continue;
      xmin = std::min(xmin, x - g->GetErrorXlow(i));
      xmax = std::max(xmax, x + g->GetErrorXhigh(i));
    }
  } else if (h) {
    for (int i = 1; i <= h->GetNbinsX(); ++i) {
      if (h->GetBinContent(i) <= 0.0) continue;
      xmin = std::min(xmin, h->GetBinLowEdge(i));
      xmax = std::max(xmax, h->GetBinLowEdge(i) + h->GetBinWidth(i));
    }
  }
  if (!(xmin < xmax)) return {0.0, 10.0};
  return {std::max(0.0, xmin), xmax};
}

std::pair<double, double> GetYRange(const TGraphAsymmErrors *g, const TH1 *h)
{
  double ymin = 1e30;
  double ymax = -1e30;
  if (g && g->GetN() > 0) {
    for (int i = 0; i < g->GetN(); ++i) {
      double x = 0.0, y = 0.0;
      g->GetPoint(i, x, y);
      if (y <= 0.0) continue;
      ymin = std::min(ymin, y - g->GetErrorYlow(i));
      ymax = std::max(ymax, y + g->GetErrorYhigh(i));
    }
  } else if (h) {
    for (int i = 1; i <= h->GetNbinsX(); ++i) {
      const double y = h->GetBinContent(i);
      if (y <= 0.0) continue;
      ymin = std::min(ymin, y - h->GetBinError(i));
      ymax = std::max(ymax, y + h->GetBinError(i));
    }
  }
  if (!(ymin < ymax)) return {1e-8, 1e-3};
  return {std::max(1e-12, ymin * 0.45), ymax * 2.5};
}

TF1 *MakeBGBW(const std::string &name, double beta, double temp, double n, double norm, double xmin, double xmax)
{
  auto *helper = new AliPWGFunc();
  gAliPwgOwners.push_back(helper);
  helper->SetVarType(AliPWGFunc::kdNdpt);
  TF1 *f = helper->GetBGBW(kHe3Mass, beta, temp, n, norm, name.c_str());
  f->SetRange(xmin, xmax);
  f->SetParName(1, "#LT#beta#GT");
  f->SetParName(2, "T_{kin}");
  f->SetParName(3, "n");
  f->SetParName(4, "Norm");
  f->SetParameter(1, beta);
  f->SetParameter(2, temp);
  f->SetParameter(3, n);
  f->SetParameter(4, norm);
  f->SetParLimits(1, 0.15, 0.95);
  f->SetParLimits(2, 0.03, 0.60);
  f->SetParLimits(3, 0.01, 5.0);
  f->SetParLimits(4, 1e-12, 1e9);
  f->SetNpx(600);
  return f;
}

double EstimateNorm(const TGraphAsymmErrors *g, const TH1 *h)
{
  if (g && g->GetN() > 0) {
    double best = 0.0;
    for (int i = 0; i < g->GetN(); ++i) {
      double x = 0.0, y = 0.0;
      g->GetPoint(i, x, y);
      if (y > best) best = y;
    }
    return std::max(best, 1e-8);
  }
  if (h) return std::max(h->GetMaximum(), 1e-8);
  return 1e-5;
}

std::unique_ptr<TF1> FitBGBW(TGraphAsymmErrors *g, TH1 *h, const std::string &funcName, int &bestStatus)
{
  auto [xmin, xmax] = GetXRange(g, h);
  const double norm = EstimateNorm(g, h);
  const std::vector<std::array<double, 4>> seeds = {
      {0.65, 0.10, 0.80, norm},
      {0.55, 0.13, 1.00, norm},
      {0.75, 0.08, 0.60, norm},
      {0.45, 0.18, 1.50, norm},
      {0.82, 0.06, 0.50, norm}};

  double bestMetric = 1e99;
  std::unique_ptr<TF1> best;
  bestStatus = 999;

  for (size_t i = 0; i < seeds.size(); ++i) {
    TF1 *trial = MakeBGBW(Form("%s_seed%zu", funcName.c_str(), i), seeds[i][0], seeds[i][1], seeds[i][2], seeds[i][3], xmin, xmax);
    int status = 999;
    if (g) {
      status = g->Fit(trial, "RQ0");
    } else if (h) {
      status = h->Fit(trial, "RQ0");
    }
    const int ndf = trial->GetNDF();
    const double chi2ndf = ndf > 0 ? trial->GetChisquare() / ndf : trial->GetChisquare();
    const bool finite = std::isfinite(chi2ndf) && trial->IsValid();
    const double penalty = (status == 0 ? 0.0 : 1e6 + std::abs(status) * 1e3);
    const double metric = finite ? chi2ndf + penalty : 1e99;
    if (metric < bestMetric) {
      bestMetric = metric;
      bestStatus = status;
      best.reset(static_cast<TF1 *>(trial->Clone(funcName.c_str())));
    }
    delete trial;
  }
  if (best) {
    best->SetName(funcName.c_str());
    best->SetLineWidth(3);
    best->SetNpx(800);
  }
  return best;
}

void DrawFitCanvas(const FitResult &res, const std::string &outPdf)
{
  auto c = std::make_unique<TCanvas>(Form("c_%s_%s", res.species.c_str(), Sanitize(res.cent).c_str()),
                                    "He3 BGBW QA", 900, 760);
  c->SetTicks(1, 1);
  c->SetLogy();
  c->SetLeftMargin(0.13);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.06);
  c->SetBottomMargin(0.12);

  auto [xmin, xmax] = GetXRange(res.graph.get(), res.hist.get());
  auto [ymin, ymax] = GetYRange(res.graph.get(), res.hist.get());
  TH1D frame("frame", ";#it{p}_{T} (GeV/#it{c});dN/d#it{p}_{T}", 100, std::max(0.0, xmin - 0.2), xmax + 0.4);
  frame.SetMinimum(ymin);
  frame.SetMaximum(ymax);
  frame.GetXaxis()->SetTitleSize(0.045);
  frame.GetYaxis()->SetTitleSize(0.045);
  frame.GetXaxis()->SetLabelSize(0.04);
  frame.GetYaxis()->SetLabelSize(0.04);
  frame.Draw("AXIS");

  TObject *dataObj = nullptr;
  if (res.graph) {
    res.graph->SetMarkerStyle(res.species == "He3" ? 20 : 24);
    res.graph->SetMarkerSize(1.0);
    res.graph->SetLineColor(kBlack);
    res.graph->SetMarkerColor(kBlack);
    res.graph->Draw("P SAME");
    dataObj = res.graph.get();
  } else if (res.hist) {
    res.hist->SetDirectory(nullptr);
    res.hist->SetMarkerStyle(res.species == "He3" ? 20 : 24);
    res.hist->SetMarkerSize(1.0);
    res.hist->SetLineColor(kBlack);
    res.hist->SetMarkerColor(kBlack);
    res.hist->Draw("E SAME");
    dataObj = res.hist.get();
  }

  if (res.func) {
    res.func->SetLineColor(res.species == "He3" ? kRed + 1 : kBlue + 1);
    res.func->Draw("SAME");
  }

  TLegend leg(0.17, 0.18, 0.50, 0.31);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.SetTextSize(0.035);
  if (dataObj) leg.AddEntry(dataObj, Form("%s %s%%", res.species.c_str(), DisplayCent(res.cent).c_str()), "lep");
  if (res.func) leg.AddEntry(res.func.get(), "BGBW fit", "l");
  leg.Draw();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextFont(42);
  latex.SetTextSize(0.037);
  latex.DrawLatex(0.17, 0.88, "ALICE Run 2 #it{^{3}He} spectrum");
  latex.DrawLatex(0.17, 0.83, Form("%s, centrality %s%%", res.species.c_str(), DisplayCent(res.cent).c_str()));
  if (res.func) {
    latex.DrawLatex(0.56, 0.88, Form("status = %d, #chi^{2}/NDF = %.2f/%d = %.2f",
                                     res.status, res.chi2, res.ndf, res.chi2ndf));
    latex.DrawLatex(0.56, 0.83, Form("#LT#beta#GT = %.3f #pm %.3f",
                                     res.func->GetParameter(1), res.func->GetParError(1)));
    latex.DrawLatex(0.56, 0.78, Form("T_{kin} = %.3f #pm %.3f GeV",
                                     res.func->GetParameter(2), res.func->GetParError(2)));
    latex.DrawLatex(0.56, 0.73, Form("n = %.3f #pm %.3f",
                                     res.func->GetParameter(3), res.func->GetParError(3)));
    latex.DrawLatex(0.56, 0.68, Form("Norm = %.3e #pm %.1e",
                                     res.func->GetParameter(4), res.func->GetParError(4)));
  }

  c->SaveAs(outPdf.c_str());
}

void DrawSummaryCanvas(const std::vector<FitResult> &results, const std::string &species, const std::string &outPdf)
{
  auto c = std::make_unique<TCanvas>(Form("c_summary_%s", species.c_str()), "He3 summary", 980, 780);
  c->SetTicks(1, 1);
  c->SetLogy();
  c->SetLeftMargin(0.13);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.06);
  c->SetBottomMargin(0.12);

  TH1D frame("frame_summary", ";#it{p}_{T} (GeV/#it{c});dN/d#it{p}_{T}", 100, 0.0, 10.0);
  frame.SetMinimum(1e-8);
  frame.SetMaximum(2e-3);
  frame.GetXaxis()->SetTitleSize(0.045);
  frame.GetYaxis()->SetTitleSize(0.045);
  frame.GetXaxis()->SetLabelSize(0.04);
  frame.GetYaxis()->SetLabelSize(0.04);
  frame.Draw("AXIS");

  TLegend leg(0.16, 0.16, 0.48, 0.38);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.SetTextSize(0.032);
  leg.SetNColumns(2);

  for (const auto &res : results) {
    if (res.species != species) continue;
    if (res.graph) {
      res.graph->SetMarkerStyle(res.marker);
      res.graph->SetMarkerColor(res.color);
      res.graph->SetLineColor(res.color);
      res.graph->Draw("P SAME");
      leg.AddEntry(res.graph.get(), Form("%s%%", DisplayCent(res.cent).c_str()), "lep");
    } else if (res.hist) {
      res.hist->SetMarkerStyle(res.marker);
      res.hist->SetMarkerColor(res.color);
      res.hist->SetLineColor(res.color);
      res.hist->Draw("E SAME");
      leg.AddEntry(res.hist.get(), Form("%s%%", DisplayCent(res.cent).c_str()), "lep");
    }
    if (res.func) res.func->Draw("SAME");
  }
  leg.Draw();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextFont(42);
  latex.SetTextSize(0.038);
  latex.DrawLatex(0.16, 0.88, Form("ALICE Run 2 %s BGBW fits", species.c_str()));
  c->SaveAs(outPdf.c_str());
}

void DrawAllSummaryCanvas(const std::vector<FitResult> &results, const std::string &outPdf)
{
  auto c = std::make_unique<TCanvas>("c_summary_all_he3_antihe3", "He3 and antiHe3 summary", 1050, 820);
  c->SetTicks(1, 1);
  c->SetLogy();
  c->SetLeftMargin(0.13);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.06);
  c->SetBottomMargin(0.12);

  TH1D frame("frame_summary_all", ";#it{p}_{T} (GeV/#it{c});dN/d#it{p}_{T}", 100, 0.0, 10.0);
  frame.SetMinimum(1e-8);
  frame.SetMaximum(2e-3);
  frame.GetXaxis()->SetTitleSize(0.045);
  frame.GetYaxis()->SetTitleSize(0.045);
  frame.GetXaxis()->SetLabelSize(0.04);
  frame.GetYaxis()->SetLabelSize(0.04);
  frame.Draw("AXIS");

  TLegend leg(0.16, 0.14, 0.56, 0.38);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.SetTextSize(0.030);
  leg.SetNColumns(2);

  for (const auto &res : results) {
    if (res.graph) {
      res.graph->SetMarkerStyle(res.marker);
      res.graph->SetMarkerColor(res.color);
      res.graph->SetLineColor(res.color);
      res.graph->Draw("P SAME");
      leg.AddEntry(res.graph.get(), Form("%s %s%%", res.species.c_str(), DisplayCent(res.cent).c_str()), "lep");
    } else if (res.hist) {
      res.hist->SetMarkerStyle(res.marker);
      res.hist->SetMarkerColor(res.color);
      res.hist->SetLineColor(res.color);
      res.hist->Draw("E SAME");
      leg.AddEntry(res.hist.get(), Form("%s %s%%", res.species.c_str(), DisplayCent(res.cent).c_str()), "lep");
    }
    if (res.func) res.func->Draw("SAME");
  }
  leg.Draw();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextFont(42);
  latex.SetTextSize(0.038);
  latex.DrawLatex(0.16, 0.88, "ALICE Run 2 #it{^{3}He} and anti-#it{^{3}He} BGBW fits");
  latex.SetTextSize(0.030);
  latex.DrawLatex(0.16, 0.83, "Solid lines: #it{^{3}He}; dashed lines: anti-#it{^{3}He}");
  c->SaveAs(outPdf.c_str());
}

} // namespace

void FitHe3SpectrumBGBW(
    const char *inputDir = "/Users/zhengqingwang/alice/data/h3l_spec_run2",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/He3SpectrumBGBW")
{
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  std::filesystem::create_directories(outputDir);
  const std::string qaDir = std::string(outputDir) + "/QA";
  std::filesystem::create_directories(qaDir);

  const std::vector<std::string> cents = {"0_5", "5_10", "10_30", "30_50", "50_90"};
  const std::vector<int> colors = {kRed + 1, kOrange + 7, kGreen + 2, kAzure + 1, kViolet + 1};
  std::vector<SpectrumItem> items;
  for (size_t i = 0; i < cents.size(); ++i) {
    items.push_back({"He3", cents[i], Form("%s/He3_%s.root", inputDir, cents[i].c_str()), colors[i], 20});
    items.push_back({"AntiHe3", cents[i], Form("%s/AntiHe3_%s.root", inputDir, cents[i].c_str()), colors[i], 24});
  }

  std::vector<FitResult> results;
  TFile outFile((std::string(outputDir) + "/He3_spectrum.root").c_str(), "RECREATE");
  TDirectory *funcDir = outFile.mkdir("BGBW");
  TDirectory *dataDir = outFile.mkdir("data");

  for (const auto &item : items) {
    std::unique_ptr<TFile> in(TFile::Open(item.fileName.c_str(), "READ"));
    if (!in || in->IsZombie()) {
      std::cerr << "[Warn] Cannot open " << item.fileName << std::endl;
      continue;
    }
    TDirectory *dir = FindFirstDirectory(*in);
    if (!dir) {
      std::cerr << "[Warn] No directory found in " << item.fileName << std::endl;
      continue;
    }

    const std::string tag = item.species + "_" + item.cent;
    auto graph = LoadGraph(dir, "g_" + tag);
    auto hist = LoadHist(dir, "h_" + tag);
    if (!graph && !hist) {
      std::cerr << "[Warn] Missing Graph1D_y1 and Hist1D_y1 in " << item.fileName << std::endl;
      continue;
    }

    if (graph) {
      graph->SetName(("g_" + tag).c_str());
      graph->SetMarkerStyle(item.marker);
      graph->SetMarkerColor(item.color);
      graph->SetLineColor(item.color);
    }
    if (hist) {
      hist->SetName(("h_" + tag).c_str());
      hist->SetMarkerStyle(item.marker);
      hist->SetMarkerColor(item.color);
      hist->SetLineColor(item.color);
    }

    int fitStatus = 999;
    auto fit = FitBGBW(graph.get(), hist.get(), "BlastWave_" + tag, fitStatus);
    if (!fit) {
      std::cerr << "[Warn] Fit failed completely for " << tag << std::endl;
      continue;
    }
    fit->SetLineColor(item.color);
    fit->SetLineStyle(item.species == "He3" ? 1 : 2);

    FitResult res;
    res.species = item.species;
    res.cent = item.cent;
    res.color = item.color;
    res.marker = item.marker;
    res.status = fitStatus;
    res.chi2 = fit->GetChisquare();
    res.ndf = fit->GetNDF();
    res.chi2ndf = res.ndf > 0 ? res.chi2 / res.ndf : res.chi2;
    res.graph = std::move(graph);
    res.hist = std::move(hist);
    res.func = std::move(fit);

    funcDir->cd();
    res.func->Write(res.func->GetName(), TObject::kOverwrite);
    dataDir->cd();
    if (res.graph) res.graph->Write(res.graph->GetName(), TObject::kOverwrite);
    if (res.hist) res.hist->Write(res.hist->GetName(), TObject::kOverwrite);
    outFile.cd();

    DrawFitCanvas(res, qaDir + "/" + tag + "_BGBW_QA.pdf");
    DrawFitCanvas(res, qaDir + "/" + tag + "_BGBW_QA.png");
    std::cout << "[Info] " << tag << " status=" << res.status
              << " chi2/ndf=" << res.chi2ndf
              << " beta=" << res.func->GetParameter(1)
              << " T=" << res.func->GetParameter(2)
              << " n=" << res.func->GetParameter(3)
              << " norm=" << res.func->GetParameter(4) << std::endl;

    results.push_back(std::move(res));
  }

  DrawSummaryCanvas(results, "He3", qaDir + "/Summary_He3_BGBW_QA.pdf");
  DrawSummaryCanvas(results, "He3", qaDir + "/Summary_He3_BGBW_QA.png");
  DrawSummaryCanvas(results, "AntiHe3", qaDir + "/Summary_AntiHe3_BGBW_QA.pdf");
  DrawSummaryCanvas(results, "AntiHe3", qaDir + "/Summary_AntiHe3_BGBW_QA.png");
  DrawAllSummaryCanvas(results, qaDir + "/Summary_All_BGBW_QA.pdf");
  DrawAllSummaryCanvas(results, qaDir + "/Summary_All_BGBW_QA.png");

  outFile.Write();
  outFile.Close();
  std::cout << "[Done] Wrote " << outputDir << "/He3_spectrum.root" << std::endl;
  std::cout << "[Done] QA canvases in " << qaDir << std::endl;
}
