// CtSingleSpectrum.C
// Extract corrected ct spectrum for ct-single bins:
//  - auto-detect data snapshots data_ct_<lo>_<hi>.root in SnapShotsData folder
//  - load matching mc_ct_<lo>_<hi>.root for shape (DSCB tails fixed from MC fit)
//  - fit data with DSCB+Chebychev(2) to get raw yields
//  - correct by BDT eff (from WorkingPoint file), MC acceptance (from AO2D MC via AcceptanceHelper), and bin width
//  - store frames, histograms, and final exponential fit of corrected spectrum.
// Run: root -l -b -q 'CtSingleSpectrum.C()'

#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TSystem.h>
#include <TError.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TMath.h>
#include <TString.h>

#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooPlot.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooAddPdf.h>
#include <RooFitResult.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "../Tools/AcceptanceHelper.h"
#include "../Tools/GeneralHelper.hpp"

using namespace std;
constexpr double kSpeedOfLightCmPerPs = 0.0299792458; // c * 1 ps

struct WPBin { double ctMin{}, ctMax{}, score{}, eff{}; };

static map<pair<double,double>, WPBin> LoadWP(const string &wpPath){
  map<pair<double,double>, WPBin> m;
  ifstream ifs(wpPath);
  if(!ifs){ ::Error("LoadWP","Cannot open %s", wpPath.c_str()); return m; }
  string line; while(getline(ifs,line)){
    if(line.empty() || line[0]=='#') continue;
    stringstream ss(line); vector<double> v; double x; while(ss>>x) v.push_back(x);
    if(v.size()>=6){
      double ctMin=v[2], ctMax=v[3], score=v[4], eff=v[5];
      m[{ctMin,ctMax}] = WPBin{ctMin,ctMax,score,eff};
    } else if(v.size()>=4){
      double ctMin=v[0], ctMax=v[1], score=v[2], eff=v[3];
      m[{ctMin,ctMax}] = WPBin{ctMin,ctMax,score,eff};
    }
  }
  return m;
}

static bool ParseCtEdges(const string &fname, double &lo, double &hi){
  // expects .../data_ct_<lo>_<hi>.root
  auto base = std::filesystem::path(fname).filename().string();
  auto pos = base.find("data_ct_");
  if(pos==string::npos) return false;
  auto rest = base.substr(pos + strlen("data_ct_"));
  if(rest.rfind(".root")!=string::npos) rest = rest.substr(0, rest.size()-5);
  auto uscore = rest.find('_');
  if(uscore==string::npos) return false;
  lo = atof(rest.substr(0, uscore).c_str());
  hi = atof(rest.substr(uscore+1).c_str());
  return true;
}

static std::unique_ptr<RooFitResult> FitMCShape(const vector<double> &masses, RooRealVar &mass,
                                                double &alphaL, double &nL, double &alphaR, double &nR,
                                                double &meanMC, double &sigmaMC, RooPlot *&frameOut){
  RooArgSet vars(mass);
  RooDataSet mcset("mc","mc", vars);
  for(double v: masses){ if(v<mass.getMin() || v>mass.getMax()) continue; mass.setVal(v); mcset.add(vars); }
  if(mcset.numEntries()<30) return nullptr;
  RooRealVar mean("meanMC","meanMC", 2.991, mass.getMin(), mass.getMax());
  RooRealVar sigma("sigmaMC","sigmaMC", 1.6e-3, 1.1e-3, 2.2e-3);
  RooRealVar aL("aL","aL", 1.5, 0.5, 5.0);
  RooRealVar nLr("nL","nL", 3.0, 0.5, 20.0);
  RooRealVar aR("aR","aR", 1.5, 0.5, 5.0);
  RooRealVar nRr("nR","nR", 3.0, 0.5, 20.0);
  RooCrystalBall cb("cb","cb", mass, mean, sigma, aL, nLr, aR, nRr);
  auto res = std::unique_ptr<RooFitResult>(cb.fitTo(mcset, RooFit::Save(true), RooFit::PrintLevel(-1)));
  if(res){
    alphaL=aL.getVal(); nL=nLr.getVal(); alphaR=aR.getVal(); nR=nRr.getVal(); meanMC=mean.getVal(); sigmaMC=sigma.getVal();
    frameOut = mass.frame();
    mcset.plotOn(frameOut, RooFit::Name("mc"));
    cb.plotOn(frameOut, RooFit::LineColor(kRed+1), RooFit::Name("sigMC"));
    auto pave = new TPaveText(0.58,0.55,0.89,0.88,"NDC");
    pave->SetFillStyle(0); pave->SetBorderSize(0); pave->SetTextAlign(12);
    pave->AddText(TString::Format("#mu = %.2f MeV", meanMC*1e3));
    pave->AddText(TString::Format("#sigma = %.2f MeV", sigmaMC*1e3));
    pave->AddText(TString::Format("#alpha_{L} = %.2f", alphaL));
    pave->AddText(TString::Format("#alpha_{R} = %.2f", alphaR));
    pave->AddText(TString::Format("n_{L} = %.2f", nL));
    pave->AddText(TString::Format("n_{R} = %.2f", nR));
    const int nFloat = 6;
    const double chi2 = frameOut->chiSquare("sigMC","mc", nFloat);
    pave->AddText(TString::Format("#chi^{2}/ndf = %.2f", chi2));
    frameOut->addObject(pave);
  }
  return res;
}

static std::unique_ptr<RooFitResult> FitData(const vector<double> &masses, RooRealVar &mass,
                                             double alphaL, double nL, double alphaR, double nR,
                                             double meanMC, double sigmaMC,
                                             double &nsigOut, double &nsigErrOut,
                                             double &meanOut, double &sigmaOut,
                                             double &raw3sOut, double &raw3sErrOut,
                                             RooPlot *&frameOut){
  RooArgSet vars(mass);
  RooDataSet data("data","data", vars);
  for(double v: masses){ if(v<mass.getMin() || v>mass.getMax()) continue; mass.setVal(v); data.add(vars); }
  if(data.numEntries()<30) return nullptr;
  RooRealVar mean("mean","mean", meanMC, mass.getMin(), mass.getMax());
  RooRealVar sigma("sigma","sigma", sigmaMC, 0.8*sigmaMC, 1.5*sigmaMC);
  RooRealVar aL("aL","aL", alphaL, 0.5, 5.0); aL.setConstant(true);
  RooRealVar nLr("nL","nL", nL, 0.5, 20.0); nLr.setConstant(true);
  RooRealVar aR("aR","aR", alphaR, 0.5, 5.0); aR.setConstant(true);
  RooRealVar nRr("nR","nR", nR, 0.5, 20.0); nRr.setConstant(true);
  RooCrystalBall sig("sig","sig", mass, mean, sigma, aL, nLr, aR, nRr);
  RooRealVar c0("c0","c0", 0.0, -10, 10);
  RooRealVar c1("c1","c1", 0.0, -10, 10);
  RooChebychev bkg("bkg","bkg", mass, RooArgList(c0,c1));
  RooRealVar nsig("nsig","nsig", data.numEntries()*0.5, 0.0, data.numEntries()*5.0);
  RooRealVar nbkg("nbkg","nbkg", data.numEntries()*0.5+1.0, 0.0, data.numEntries()*5.0+10.0);
  RooAddPdf model("model","model", RooArgList(sig,bkg), RooArgList(nsig, nbkg));
  auto res = std::unique_ptr<RooFitResult>(model.fitTo(data, RooFit::Save(true), RooFit::Extended(true), RooFit::PrintLevel(-1)));
  frameOut = mass.frame(50);
  data.plotOn(frameOut, RooFit::Name("data"));
  model.plotOn(frameOut, RooFit::LineColor(kAzure+1), RooFit::Name("model"));
  model.plotOn(frameOut, RooFit::Components(bkg.GetName()), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed+1), RooFit::Name("bkg"));
  model.plotOn(frameOut, RooFit::Components(sig.GetName()), RooFit::LineStyle(kDotted), RooFit::LineColor(kGreen+2), RooFit::Name("sig"));
  nsigOut = nsig.getVal(); nsigErrOut = nsig.getError();
  meanOut = mean.getVal(); sigmaOut = sigma.getVal();

  const double winMin = std::max(mass.getMin(), mean.getVal() - 3.0*sigma.getVal());
  const double winMax = std::min(mass.getMax(), mean.getVal() + 3.0*sigma.getVal());
  mass.setRange("sigWindow", winMin, winMax);
  std::unique_ptr<RooAbsReal> sigInt(sig.createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
  const double sigFrac = sigInt ? sigInt->getVal() : 0.0;
  raw3sOut = nsigOut * sigFrac;
  raw3sErrOut = nsigErrOut * sigFrac;

  const int nFloat = 6; // mean, sigma, c0, c1, nsig, nbkg
  const double chi2 = frameOut->chiSquare("model","data", nFloat);
  auto pave = new TPaveText(0.58,0.55,0.89,0.88,"NDC");
  pave->SetFillStyle(0); pave->SetBorderSize(0); pave->SetTextAlign(12);
  pave->AddText(TString::Format("N_{sig} = %.1f #pm %.1f", nsigOut, nsigErrOut));
  pave->AddText(TString::Format("N_{3#sigma} = %.1f #pm %.1f", raw3sOut, raw3sErrOut));
  pave->AddText(TString::Format("#mu = %.2f MeV", meanOut*1e3));
  pave->AddText(TString::Format("#sigma = %.2f MeV", sigmaOut*1e3));
  pave->AddText(TString::Format("#chi^{2}/ndf = %.2f", chi2));
  frameOut->addObject(pave);
  return res;
}

void CtSingleSpectrum(){
  const string snapshotDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID";
  const string wpPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass5_CustomV0s_HadronPID/WorkingPoint/WorkingPoint_CtSingle.txt";
  const string mcAo2d = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_CustomV0s.root";
  const string dataTree = "O2hypcands";
  const string mcTree   = "O2mchypcands";
  const string isMatter = "both"; // "matter", "antimatter", "both"
  const string basicSelectionDataForMCEff = "fTPCsignalPi<1000 && fCosPA>0.99 && fAvgClusterSizeHe > 5 && fDecRad < 2.1";
  const string outFileDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/Checks/LifeTime_BeamPipe";
  const string outFileName = outFileDir + "/ct_single_spectrum.root";
  const double massLo=2.96, massHi=3.04;

  // discover ct bins from snapshot filenames
  vector<string> dataFiles;
  vector<double> ctEdges;
  for(const auto &entry : std::filesystem::directory_iterator(snapshotDir)){
    if(!entry.is_regular_file()) continue;
    auto name = entry.path().filename().string();
    if(name.rfind("data_ct_",0)!=0) continue;
    double lo=0,hi=0; if(!ParseCtEdges(name,lo,hi)) continue;
    dataFiles.push_back(entry.path().string());
    ctEdges.push_back(lo); ctEdges.push_back(hi);
  }
  if(dataFiles.empty()){ ::Error("CtSingleSpectrum","No data_ct_* files found under %s", snapshotDir.c_str()); return; }
  // unique sort edges
  sort(ctEdges.begin(), ctEdges.end()); ctEdges.erase(unique(ctEdges.begin(), ctEdges.end()), ctEdges.end());
  sort(dataFiles.begin(), dataFiles.end());
  cout << "Found " << dataFiles.size() << " ct bins" << endl;

  // load WP map
  auto wpMap = LoadWP(wpPath);
  if(wpMap.empty()){ ::Error("CtSingleSpectrum","WP map empty"); return; }

  // acceptance from AO2D MC
  if(gSystem->AccessPathName(mcAo2d.c_str())){ ::Error("CtSingleSpectrum","MC AO2D missing: %s", mcAo2d.c_str()); return; }
  ROOT::EnableImplicitMT();
  TFile mcFile(mcAo2d.c_str(), "READ");
  if(mcFile.IsZombie()){ ::Error("CtSingleSpectrum","Cannot open MC AO2D: %s", mcAo2d.c_str()); return; }
  TChain mcChain(mcTree.c_str());
  GeneralHelper::fillChainFromAO2D(mcChain, &mcFile);
  if(mcChain.GetEntries()<=0){ ::Error("CtSingleSpectrum","MC chain empty (tree %s)", mcTree.c_str()); return; }
  ROOT::RDataFrame mcAo2df(mcChain);
  auto mcReady = GeneralHelper::CorrectAndConvertRDF(mcAo2df, false, true);
  auto accRes = AcceptanceHelper::ComputeAcceptanceFlexible(
    mcReady,
    std::vector<double>{},
    ctEdges,
    std::vector<std::vector<double>>{},
    std::vector<double>{},
    std::vector<std::vector<double>>{},
    basicSelectionDataForMCEff
  );
  TH1D *accHist = nullptr;
  if (isMatter == "matter") {
      accHist = accRes.acc_ct_matter;
  } else if (isMatter == "antimatter") {
      accHist = accRes.acc_ct_antimatter;
  } else {
      accHist = accRes.acc_ct_both;
  }
  if(!accHist){ ::Error("CtSingleSpectrum","Acceptance histogram null"); return; }

  // output file
  std::filesystem::create_directories(outFileDir);
  TFile fout(Form("%s", outFileName.c_str()),"RECREATE");
  TDirectory *dFits = fout.mkdir("Fits");

  TH1D hRaw("h_raw_counts", ";ct (cm);Raw counts (|#it{m}-#mu|<3#sigma)", static_cast<int>(ctEdges.size())-1, ctEdges.data());
  TH1D hAcc("h_acc", ";ct (cm);Acceptance", static_cast<int>(ctEdges.size())-1, ctEdges.data());
  TH1D hBdt("h_bdt_eff", ";ct (cm);BDT efficiency", static_cast<int>(ctEdges.size())-1, ctEdges.data());
  TH1D hCorr("h_corrected", ";ct (cm);1/(Acc #times #epsilon_{BDT} #times #Delta ct) N_{3#sigma}", static_cast<int>(ctEdges.size())-1, ctEdges.data());

  for(const auto &dfPath : dataFiles){
    double lo=0,hi=0; if(!ParseCtEdges(dfPath,lo,hi)) continue;
    auto wpIt = wpMap.find({lo,hi});
    if(wpIt==wpMap.end()){ ::Warning("CtSingleSpectrum","No WP for %.2f-%.2f, skip", lo, hi); continue; }
    const WPBin &wp = wpIt->second;

    // load data masses
    if(gSystem->AccessPathName(dfPath.c_str())){ ::Warning("CtSingleSpectrum","Data file missing %s", dfPath.c_str()); continue; }
    ROOT::RDataFrame ddf(dataTree, dfPath);
    auto filtered = ddf.Filter([&](double ct, float sc, bool matt){
        bool passMatter = true;
      if(isMatter == "matter") passMatter = matt;
      else if(isMatter == "antimatter") passMatter = !matt;
        return passMatter && ct>=lo && ct<hi && sc>wp.score;
      }, {"fCt","model_output","fIsMatter"});
    auto massesVec = filtered.Take<double>("fMassH3L");
    vector<double> massesData(massesVec->begin(), massesVec->end());

    // mc snapshot for shape
    auto mcSnap = snapshotDir + "/mc_ct_" + TString::Format("%g_%g", lo, hi).Data() + ".root";
    if(gSystem->AccessPathName(mcSnap.c_str())){ ::Warning("CtSingleSpectrum","MC snapshot missing %s", mcSnap.c_str()); continue; }
    ROOT::RDataFrame mcsnap(mcTree, mcSnap);
    auto mcMassVec = mcsnap.Take<double>("fMassH3L");
    vector<double> massesMC(mcMassVec->begin(), mcMassVec->end());

    RooRealVar mass("mass","mass", massLo, massHi);
    double aL=1.5,nL=3.0,aR=1.5,nR=3.0,meanMC=2.991,sigmaMC=1.6e-3; RooPlot *frMC=nullptr;
    auto resMC = FitMCShape(massesMC, mass, aL,nL,aR,nR, meanMC, sigmaMC, frMC);
    if(!resMC){ ::Warning("CtSingleSpectrum","MC fit failed for bin %.2f-%.2f", lo, hi); continue; }

    double nsig=0, nsigErr=0, meanD=0, sigmaD=0, raw3s=0, raw3sErr=0; RooPlot *frData=nullptr;
    auto resData = FitData(massesData, mass, aL,nL,aR,nR, meanMC, sigmaMC, nsig, nsigErr, meanD, sigmaD, raw3s, raw3sErr, frData);
    if(!resData){ ::Warning("CtSingleSpectrum","Data fit failed for bin %.2f-%.2f", lo, hi); continue; }

    double acc = accHist->GetBinContent(accHist->FindBin((lo+hi)*0.5));
    double accErr = accHist->GetBinError(accHist->FindBin((lo+hi)*0.5));
    if(acc<=0){ ::Warning("CtSingleSpectrum","Zero acc for bin %.2f-%.2f", lo, hi); continue; }
    double width = hi-lo;
    double corrected = raw3s / (wp.eff * acc * width);
    double rel2 = 0.0;
    if(raw3s>0 && raw3sErr>0) rel2 += (raw3sErr*raw3sErr)/(raw3s*raw3s);
    if(acc>0 && accErr>0) rel2 += (accErr*accErr)/(acc*acc);
    double corrErr = corrected * sqrt(rel2);

    dFits->cd();
    if(frMC){ frMC->SetName(TString::Format("frame_mc_ct_%g_%g", lo, hi)); frMC->Write(); }
    frData->SetName(TString::Format("frame_ct_%g_%g", lo, hi));
    frData->Write();

    TCanvas cFit(TString::Format("c_fit_%g_%g", lo, hi), "data fit", 900, 700);
    frData->Draw();
    auto leg = new TLegend(0.60,0.65,0.90,0.90);
    leg->SetBorderSize(0); leg->SetFillStyle(0); leg->SetTextSize(0.04);
    leg->AddEntry((TObject*)nullptr, TString::Format("ct %.2f-%.2f", lo, hi), "");
    leg->AddEntry("model","Total","l");
    leg->AddEntry("bkg","Bkg","l");
    leg->AddEntry("sig","Signal","l");
    leg->Draw();
    cFit.Write();

    const int binIdx = hRaw.FindBin((lo+hi)*0.5);
    hRaw.SetBinContent(binIdx, raw3s); hRaw.SetBinError(binIdx, raw3sErr);
    hAcc.SetBinContent(binIdx, acc); hAcc.SetBinError(binIdx, accErr);
    hBdt.SetBinContent(binIdx, wp.eff); hBdt.SetBinError(binIdx, 0.0);
    hCorr.SetBinContent(binIdx, corrected); hCorr.SetBinError(binIdx, corrErr);
  }

  // fit corrected spectrum histogram directly
  fout.cd();
  if(hCorr.GetEntries()>1){
    TF1 fexp("fexp","[0]*exp(-x/[1])", hCorr.GetXaxis()->GetXmin(), hCorr.GetXaxis()->GetXmax());
    fexp.SetParName(0,"A"); fexp.SetParName(1,"ct");
    fexp.SetParameters(std::max(1.0, hCorr.GetMaximum()), 7.6);
    fexp.SetParLimits(1, 0.1, 10.0);
    TCanvas c("c_ct_fit","ct exponential",900,650);
    c.SetLeftMargin(0.14); c.SetBottomMargin(0.12); c.SetRightMargin(0.05); c.SetTopMargin(0.05);
    c.SetGridx(); c.SetGridy();
    c.SetLogy();
    hCorr.SetStats(false);
    hCorr.SetMinimum(std::max(1e-3, hCorr.GetMinimum(1) * 0.5));
    hCorr.SetLineColor(kAzure+2);
    hCorr.SetMarkerColor(kAzure+2);
    hCorr.SetMarkerStyle(20);
    hCorr.SetMarkerSize(1.1);
    hCorr.Draw("E1");
    hCorr.Fit(&fexp, "QIS");
    fexp.SetLineColor(kRed+1); fexp.SetLineWidth(3); fexp.Draw("SAME");

    auto leg = new TLegend(0.60,0.70,0.90,0.88);
    leg->SetBorderSize(0); leg->SetFillStyle(0); leg->SetTextSize(0.045);
    leg->AddEntry(&hCorr, "Corrected spectrum", "lep");
    leg->AddEntry(&fexp, "Exp fit", "l");
    leg->Draw();

    const double tauCm = fexp.GetParameter(1);
    const double tauCmErr = fexp.GetParError(1);
    const double tauPs = tauCm / kSpeedOfLightCmPerPs;
    const double tauPsErr = tauCmErr / kSpeedOfLightCmPerPs;
    const double chi2 = fexp.GetChisquare();
    const int ndf = fexp.GetNDF();
    const double fitProb = (ndf > 0) ? TMath::Prob(chi2, ndf) : 0.0;

    auto pave = new TPaveText(0.18,0.70,0.55,0.90,"NDC");
    pave->SetFillStyle(0); pave->SetBorderSize(0); pave->SetTextAlign(12); pave->SetTextSize(0.045);
    pave->AddText(TString::Format("#tau = %.2f #pm %.2f ps", tauPs, tauPsErr));
    pave->AddText(TString::Format("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
    pave->AddText(TString::Format("Fit prob. = %.3f", fitProb));
    pave->Draw();

    c.Write("c_ct_fit");
    fexp.Write("exp_fit");
  }

  fout.cd();
  hRaw.Write();
  hAcc.Write();
  hBdt.Write();
  hCorr.Write();

  fout.Close();
  cout << "CtSingleSpectrum done. Output: ct_single_spectrum.root" << endl;
}
