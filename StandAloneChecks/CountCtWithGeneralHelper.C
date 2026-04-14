// CountCtWithGeneralHelper.C
#include <ROOT/RDataFrame.hxx>
#include "../Tools/GeneralHelper.hpp"
#include <TFile.h>
#include <TChain.h>
#include <TSystem.h>
#include <iostream>

using namespace std;

void CountCtWithGeneralHelper(){
  const char *path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s_HadronPID.root";
  const string treeName = "O2hypcands";
  int bins[] = {1,2,4,6,8,10,14,18,23,35};

  if(gSystem->AccessPathName(path)){ cerr<<"AO2D file missing: "<<path<<"\n"; return; }
  TFile f(path, "READ");
  if(f.IsZombie()){ cerr<<"Cannot open AO2D file\n"; return; }

  TChain chain(treeName.c_str());
  GeneralHelper::fillChainFromAO2D(chain, &f);
  if(chain.GetEntries()<=0){ cerr<<"No entries in chain\n"; return; }

  ROOT::EnableImplicitMT();
  ROOT::RDataFrame rdf(chain);
  auto conv = GeneralHelper::CorrectAndConvertRDF(rdf, false, false);

  cout<<"Counting entries per fCt bin (using GeneralHelper::CorrectAndConvertRDF)"<<"\n";
  const int nbins = sizeof(bins)/sizeof(bins[0]) - 1;
  double edges[sizeof(bins)/sizeof(bins[0])];
  for(size_t i=0;i<sizeof(bins)/sizeof(bins[0]); ++i) edges[i] = bins[i];

  // basic selection
  const string basicSel = "fTPCsignalPi<1000 && fCosPA>0.99 && fAvgClusterSizeHe > 5 && fCentralityFT0C > 0 && fCentralityFT0C < 80";
  auto convSel = conv.Filter(basicSel);

  // histogram counts after selection
  auto hptr = convSel.Histo1D({"h_ct_sel",";ct (cm);counts", nbins, edges}, "fCt");
  auto hobj = hptr.GetValue();
  // collect AO2D counts per bin
  std::vector<double> ao2dCounts; ao2dCounts.reserve(nbins);
  for(int ib=1; ib<=nbins; ++ib){
    double lo = hobj.GetXaxis()->GetBinLowEdge(ib);
    double hi = hobj.GetXaxis()->GetBinUpEdge(ib);
    double cnt = hobj.GetBinContent(ib);
    ao2dCounts.push_back(cnt);
    cout<<"(sel) bin "<<(ib-1)<<" ["<<lo<<","<<hi<<"]: "<<cnt<<"\n";
  }

  // snapshot files existence and count
  const string snapshotDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID";
  size_t foundSnapCount = 0;
  if(!std::filesystem::exists(snapshotDir)){
    cout<<"Snapshot dir missing: "<<snapshotDir<<"\n";
  } else {
    for(const auto &e: std::filesystem::directory_iterator(snapshotDir)){
      if(!e.is_regular_file()) continue;
      auto name = e.path().filename().string();
      if(name.rfind("data_ct_",0)==0) ++foundSnapCount;
    }
    cout<<"Snapshot data_ct files found: "<<foundSnapCount<<"\n";
    // check expected filenames
    for(int i=0;i<nbins;++i){
      int lo = bins[i]; int hi = bins[i+1];
      string fname = Form("data_ct_%d_%d.root", lo, hi);
      string full = snapshotDir + "/" + fname;
      bool exists = std::filesystem::exists(full);
      cout<<"expected: "<<fname<<" -> "<<(exists ? "PRESENT" : "MISSING")<<"\n";
      // if present, open and count entries passing basic selection
      long long snapCount = -1;
      if(exists){
        try{
          ROOT::RDataFrame sdf("O2hypcands", full);
          // count raw candidates in snapshot tree without applying any selection
          auto cnt = sdf.Count();
          snapCount = cnt.GetValue();
        } catch(...) {
          cout<<"  Error reading/processing "<<full<<"\n";
          snapCount = -1;
        }
      }
      double ao2d = (i < (int)ao2dCounts.size()) ? ao2dCounts[i] : -1;
      cout<<"  AO2D (conv+sel): "<<ao2d<<"  snapshot(sel): "<<snapCount;
      if(snapCount>=0 && ao2d>=0){
        double diff = snapCount - ao2d;
        double pct = (ao2d>0) ? (diff/ao2d*100.0) : 0.0;
        cout<<"  diff="<<diff<<" ("<<pct<<"%)";
      }
      cout<<"\n";
    }
  }
}
