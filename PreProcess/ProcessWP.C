// ProcessWP.C (rewrite per config_WP.json)
// Usage: root -l -b -q 'ProcessWP.C("config_WP.json")'

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TString.h>
#include <Rtypes.h>
#include <TDirectory.h>
#include <TMath.h>
#include <TGraph.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TStyle.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <cmath>

#include <nlohmann/json.hpp>

#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooArgList.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooFit.h"
#include "RooAbsReal.h"
#include "RooCrystalBall.h"
#include "RooPlot.h"
#include "RooMsgService.h"
#include "../Tools/GeneralHelper.hpp"

using json = nlohmann::json;
using namespace GeneralHelper;

// tiny helper
static std::string read_file_to_string(const std::string &path) {
  std::ifstream ifs(path);
  if(!ifs) return std::string();
  std::stringstream ss; ss << ifs.rdbuf();
  return ss.str();
}

static void read_score_eff_file(const std::string &path, std::vector<double> &scores, std::vector<double> &effs){
  scores.clear(); effs.clear();
  std::ifstream ifs(path);
  if(!ifs){ return; }
  std::string line;
  while(std::getline(ifs, line)){
    if(line.empty() || line[0]=='#') continue;
    std::stringstream ss(line);
    double s,e; ss >> s >> e; if(ss.fail()) continue;
    scores.push_back(s); effs.push_back(e);
  }
}

// format helper: avoid trailing decimals in filenames
static std::string fmt_edge(double x){
  char buf[64];
  if (std::floor(x)==x) snprintf(buf, sizeof(buf), "%g", x); else snprintf(buf, sizeof(buf), "%g", x);
  return std::string(buf);
}

static std::string make_label(bool hasCen, bool hasPt, bool hasCt,
                              double cenmin, double cenmax, double ptmin, double ptmax, double ctmin, double ctmax){
  std::vector<std::string> parts;
  if(hasCen) parts.push_back(std::string("cen_") + fmt_edge(cenmin) + "_" + fmt_edge(cenmax));
  if(hasPt)  parts.push_back(std::string("pt_")  + fmt_edge(ptmin)  + "_" + fmt_edge(ptmax));
  if(hasCt)  parts.push_back(std::string("ct_")  + fmt_edge(ctmin)  + "_" + fmt_edge(ctmax));
  if(parts.empty()) return std::string("all");
  std::string out = parts[0];
  for(size_t i=1;i<parts.size();++i){ out += "_" + parts[i]; }
  return out;
}

static std::string make_desc(bool hasCen, bool hasPt, bool hasCt,
                             double cenmin, double cenmax, double ptmin, double ptmax, double ctmin, double ctmax){
  std::vector<std::string> parts;
  if(hasCen) parts.push_back(Form("centrality %.0f-%.0f", cenmin, cenmax));
  if(hasPt)  parts.push_back(Form("pT %.2f-%.2f GeV/c", ptmin, ptmax));
  if(hasCt)  parts.push_back(Form("ct %.2f-%.2f cm", ctmin, ctmax));
  if(parts.empty()) return std::string("full phase space");
  std::string out = parts[0];
  for(size_t i=1;i<parts.size();++i){ out += std::string(", ") + parts[i]; }
  return out;
}

void ProcessWP(const char *config_path = "../configs/config_WP.json"){
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL); // suppress RooFit messages
  gStyle->SetOptStat(0); // disable stats boxes on plots
  // read and parse config
  std::string cfg_text = read_file_to_string(config_path);
  if(cfg_text.empty()){ printf("Failed to read config: %s\n", config_path); return; }
  json cfg; try{ cfg = json::parse(cfg_text);} catch(...){ printf("Invalid JSON config.\n"); return; }

  // required fields
  std::string trained_data_dir = cfg.value("trained_data_dir", std::string(""));
  std::string tree_name       = cfg.value("tree_name", std::string("O2hypcands"));
  std::string tree_name_mc    = cfg.value("tree_name_mc", std::string("O2mchypcands"));
  std::string score_eff_dir   = cfg.value("score_eff_dir", std::string(""));
  std::string out_dir         = cfg.value("out_dir", std::string("WP_output"));
  std::string name_suffix         = cfg.value("name_suffix", std::string("Crosssection_Customvertex"));
  std::string mix_mode_cfg        = cfg.value("Mix_mode", cfg.value("Mix_Mode", std::string("pt-ct")));
  // compatibility with legacy boolean MIXMode
  if(cfg.contains("MIXMode")){
    bool legacy = cfg.value("MIXMode", true);
    mix_mode_cfg = legacy ? std::string("pt-ct") : std::string("cen-pt");
  }
  std::string mix_mode = mix_mode_cfg;
  std::transform(mix_mode.begin(), mix_mode.end(), mix_mode.begin(), ::tolower);
  std::vector<double> pt_bins = cfg.value("pt_bins", std::vector<double>{});
  std::vector<std::vector<double>> ct_bins;
  if(cfg.contains("ct_bins")){
    try {
      ct_bins = cfg.at("ct_bins").get<std::vector<std::vector<double>>>();
    } catch(...){
      try {
        auto flat_ct = cfg.at("ct_bins").get<std::vector<double>>();
        if(!flat_ct.empty()) ct_bins.push_back(flat_ct);
      } catch(...){
        ct_bins.clear();
      }
    }
  }
  std::vector<double> ct_bins_single = cfg.value("ct_bins_single", std::vector<double>{});
  std::vector<double> cen_bins = cfg.value("cen_bins", std::vector<double>{});
  std::vector<std::vector<double>> pt_bins_by_centrality = cfg.value("pt_bins_by_centrality", std::vector<std::vector<double>>{});
  std::vector<double> pt_bins_single = cfg.value("pt_bins_single", std::vector<double>{});
  std::vector<double> mass_range = cfg.value("mass_range", std::vector<double>{2.96, 3.04});
  int mass_nbins = cfg.value("mass_nbins", 50);
  std::vector<double> side_low  = cfg.value("sideband_low", std::vector<double>{2.96, 2.98});
  std::vector<double> side_high = cfg.value("sideband_high", std::vector<double>{3.005, 3.04});
  double signal_sigma_mult = cfg.value("signal_window_sigma", 3.0);
  int min_entries_for_fit   = cfg.value("min_entries_for_fit", 50);
  double fixed_signal_yield = cfg.value("fixed_signal_yield", 100.0);
  double max_chi2_ndf       = cfg.value("max_chi2_ndf", 5.0);
  double max_sideband_rel_diff = cfg.value("max_sideband_rel_diff", 0.5);
  bool aliceperformance    = cfg.value("performance", false);
  std::string period_text = cfg.value("period_text", std::string("LHC23PbPb apass5"));
  std::string additional_text = cfg.value("additional_pave_text", std::string(""));
  std::vector<double> target_pt_range = cfg.value("target_pt_range", std::vector<double>{});
  std::vector<double> target_ct_range = cfg.value("target_ct_range", std::vector<double>{});
  std::vector<double> target_cen_range = cfg.value("target_cen_range", std::vector<double>{});
  std::vector<double> yield_eff_range = cfg.value("yield_eff_range", std::vector<double>{0.5, 0.9});
  bool enable_mt = cfg.value("enable_implicit_mt", false);

  if(enable_mt) EnableImplicitMTWithPreferredThreads();


  gSystem->mkdir(out_dir.c_str(), true);
  
  // working point summary file (always rewritten)
  std::string wp_txt = out_dir + "/WorkingPoint_" + name_suffix + ".txt";
  std::vector<std::string> wp_lines; // start empty; will be rebuilt for current run

  enum class WPFormat { Full, CenPt, PtCt, PtOnly, CtOnly };
  auto format_from_mix = [](const std::string &mode){
    if(mode == "cen-pt") return WPFormat::CenPt;
    if(mode == "pt-ct" || mode == "pt-ct-single") return WPFormat::PtCt;
    if(mode == "pt-single") return WPFormat::PtOnly;
    if(mode == "ct-single") return WPFormat::CtOnly;
    return WPFormat::Full; // fallback keeps legacy 6-column boundaries
  };
  WPFormat wp_format = format_from_mix(mix_mode);

  auto header_for_format = [](WPFormat fmt){
    switch(fmt){
      case WPFormat::CenPt: return std::string("# cenmin cenmax ptmin ptmax best_score best_eff max_significance");
      case WPFormat::PtCt:  return std::string("# ptmin ptmax ctmin ctmax best_score best_eff max_significance");
      case WPFormat::PtOnly:return std::string("# ptmin ptmax best_score best_eff max_significance");
      case WPFormat::CtOnly:return std::string("# ctmin ctmax best_score best_eff max_significance");
      default:              return std::string("# cenmin cenmax ptmin ptmax ctmin ctmax best_score best_eff max_significance");
    }
  };

  auto detect_format_from_lines = [&](WPFormat current){
    for(const auto &ln : wp_lines){
      if(ln.rfind("#",0)!=0) continue;
      if(ln.find("ptmin ptmax ctmin ctmax") != std::string::npos) return WPFormat::PtCt;
      if(ln.find("cenmin cenmax ptmin ptmax") != std::string::npos) return WPFormat::CenPt;
      if(ln.find("ptmin ptmax best_score") != std::string::npos) return WPFormat::PtOnly;
      if(ln.find("ctmin ctmax best_score") != std::string::npos) return WPFormat::CtOnly;
      if(ln.find("ctmin ctmax") != std::string::npos) return WPFormat::Full;
    }
    return current;
  };

  auto make_key_string = [&](double cenmin, double cenmax, double ptmin, double ptmax, double ctmin, double ctmax){
    switch(wp_format){
      case WPFormat::CenPt: return std::string(Form("%g %g %g %g", cenmin, cenmax, ptmin, ptmax));
      case WPFormat::PtCt:  return std::string(Form("%g %g %g %g", ptmin, ptmax, ctmin, ctmax));
      case WPFormat::PtOnly:return std::string(Form("%g %g", ptmin, ptmax));
      case WPFormat::CtOnly:return std::string(Form("%g %g", ctmin, ctmax));
      default:              return std::string(Form("%g %g %g %g %g %g", cenmin, cenmax, ptmin, ptmax, ctmin, ctmax));
    }
  };

  auto parse_line_key = [&](const std::string &line){
    if(line.empty() || line[0]=='#') return std::string();
    std::stringstream ss(line);
    std::vector<double> vals; double v;
    while(ss>>v) vals.push_back(v);
    if(wp_format == WPFormat::Full && vals.size() >= 6){
      return make_key_string(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    }
    if(wp_format == WPFormat::CenPt && vals.size() >= 4){
      return make_key_string(vals[0], vals[1], vals[2], vals[3], -1.0, -1.0);
    }
    if(wp_format == WPFormat::PtCt && vals.size() >= 4){
      return make_key_string(-1.0, -1.0, vals[0], vals[1], vals[2], vals[3]);
    }
    if(wp_format == WPFormat::PtOnly && vals.size() >= 2){
      return make_key_string(-1.0, -1.0, vals[0], vals[1], -1.0, -1.0);
    }
    if(wp_format == WPFormat::CtOnly && vals.size() >= 2){
      return make_key_string(-1.0, -1.0, -1.0, -1.0, vals[0], vals[1]);
    }
    return std::string();
  };

  wp_format = detect_format_from_lines(wp_format);
  auto upsert_wp_line = [&](bool hasCen, bool hasPt, bool hasCt,
                            double cenmin, double cenmax, double ptmin, double ptmax, double ctmin, double ctmax,
                            double bestScore, double bestEff, double bestSig){
    double cenmin_w = hasCen ? cenmin : -1.0;
    double cenmax_w = hasCen ? cenmax : -1.0;
    double ptmin_w  = hasPt  ? ptmin  : -1.0;
    double ptmax_w  = hasPt  ? ptmax  : -1.0;
    double ctmin_w  = hasCt  ? ctmin  : -1.0;
    double ctmax_w  = hasCt  ? ctmax  : -1.0;
    std::string key = make_key_string(cenmin_w, cenmax_w, ptmin_w, ptmax_w, ctmin_w, ctmax_w);
    std::string newline;
    if(wp_format == WPFormat::PtOnly || wp_format == WPFormat::CtOnly ||
       wp_format == WPFormat::CenPt || wp_format == WPFormat::PtCt || wp_format == WPFormat::Full){
      newline = Form("%s %g %g %g", key.c_str(), bestScore, bestEff, bestSig);
    }
    bool replaced=false;
    for(size_t i=0;i<wp_lines.size();++i){
      std::string existing_key = parse_line_key(wp_lines[i]);
      if(existing_key.empty()) continue;
      if(key == existing_key) {
        if(!replaced){
          wp_lines[i] = newline; // replace first occurrence
          replaced = true;
        } else {
          wp_lines[i].clear();   // drop duplicate occurrences
        }
      }
    }
    if(!replaced){
      // ensure header exists once (only when header string is non-empty)
      std::string hdr = header_for_format(wp_format);
      if(!hdr.empty() && (wp_lines.empty() || wp_lines[0].rfind("#",0)!=0)){
        wp_lines.insert(wp_lines.begin(), hdr);
      }
      wp_lines.push_back(newline);
    }
  };

  struct BinContext {
    bool hasCen{false};
    bool hasPt{true};
    bool hasCt{true};
    double cenmin{0}, cenmax{0};
    double ptmin{0}, ptmax{0};
    double ctmin{0}, ctmax{0};
    int mode{0}; // optional log code
    std::string label;
    std::string desc;
  };

  auto process_one_bin = [&](const BinContext &ctx){
      printf("[WP] mode %d | %s\n", ctx.mode, ctx.desc.c_str());

      std::string label = ctx.label.empty() ? make_label(ctx.hasCen, ctx.hasPt, ctx.hasCt,
                                                         ctx.cenmin, ctx.cenmax, ctx.ptmin, ctx.ptmax, ctx.ctmin, ctx.ctmax)
                                            : ctx.label;
      // snapshot and score-eff file paths aligned with BDTPreProcess labels
      std::string snap_path   = trained_data_dir + "/data_" + label + ".root";
      std::string snap_path_mc   = trained_data_dir + "/mc_" + label + ".root";
      std::string score_path  = score_eff_dir   + "/score_efficiency_array_" + label + ".txt";

      // open data
      if (gSystem->AccessPathName(snap_path.c_str())){ printf("  missing snapshot: %s\n", snap_path.c_str()); return -1; }
      // read only required columns to reduce IO on large snapshots
      ROOT::RDataFrame df(tree_name.c_str(), snap_path.c_str(), {"fMassH3L", "model_output"});
      bool has_mc_snapshot = (gSystem->AccessPathName(snap_path_mc.c_str()) == 0);

      // read score-eff
      std::vector<double> scores, effs; read_score_eff_file(score_path, scores, effs);
      if(scores.empty()) { printf("  missing score-eff: %s\n", score_path.c_str()); return -1; }

      // output ROOT file per-bin
      std::string out_root = out_dir + "/WP_" + label + ".root";
      TFile fout(out_root.c_str(), "RECREATE");
      if(fout.IsZombie()){ printf("  cannot create %s\n", out_root.c_str()); return -1; }
      TDirectory *dFits = fout.mkdir("Fits");
      TDirectory *dSigs = fout.mkdir("Graphs");

      // mass variable and ranges
      double mmin = mass_range.size()>0 ? mass_range[0] : 2.96;
      double mmax = mass_range.size()>1 ? mass_range[1] : 3.04;
      RooRealVar m_global("m","mass", mmin, mmax);

      double init_mean = 2.991;
      double init_sigma = 1.7e-3;
      double tail_alphaL = 1.5;
      double tail_nL = 2.0;
      double tail_alphaR = 1.5;
      double tail_nR = 2.0;

      if(has_mc_snapshot){
        ROOT::RDataFrame df_mc(tree_name_mc.c_str(), snap_path_mc.c_str(), {"fMassH3L"});
        auto mcMasses = df_mc.Take<double>("fMassH3L");
        RooArgSet mc_vars(m_global);
        RooDataSet mcData("mcData","mcData", mc_vars);
        for(double mv : *mcMasses){
          if(mv < mmin || mv > mmax) continue;
          m_global.setVal(mv);
          mcData.add(mc_vars);
        }
        const size_t min_entries_mc = std::max<int>(20, min_entries_for_fit / 2);
        if(mcData.numEntries() >= min_entries_mc){
          RooRealVar mc_mean("mc_mean","mc_mean", init_mean, 2.985, 3.005);
          RooRealVar mc_sigma("mc_sigma","mc_sigma", init_sigma, 1.4e-3, 2.0e-3);
          RooRealVar mc_a1("mc_a1","mc_a1", tail_alphaL, 0.7, 5.0);
          RooRealVar mc_n1("mc_n1","mc_n1", tail_nL, 0.9, 15.0);
          RooRealVar mc_a2("mc_a2","mc_a2", tail_alphaR, 0.7, 5.0);
          RooRealVar mc_n2("mc_n2","mc_n2", tail_nR, 0.9, 15.0);
          RooCrystalBall mc_signal("mc_signal","mc_signal", m_global, mc_mean, mc_sigma, mc_a1, mc_n1, mc_a2, mc_n2);
          std::unique_ptr<RooFitResult> mcRes(mc_signal.fitTo(mcData, RooFit::Save(true), RooFit::PrintLevel(-1)));
          if(mcRes){
            tail_alphaL = mc_a1.getVal();
            tail_nL = mc_n1.getVal();
            tail_alphaR = mc_a2.getVal();
            tail_nR = mc_n2.getVal();
            init_mean = mc_mean.getVal();
            init_sigma = mc_sigma.getVal();
            RooPlot *mcFrame = m_global.frame(mass_nbins);
            mcData.plotOn(mcFrame, RooFit::Name("mc_data"));
            mc_signal.plotOn(mcFrame, RooFit::Name("mc_pdf"));
            TPaveText *ptMC = new TPaveText(0.14, 0.65, 0.45, 0.9, "NDC");
            ptMC->SetBorderSize(0); ptMC->SetFillStyle(0); ptMC->SetTextFont(42); ptMC->SetTextAlign(11);
            ptMC->AddText("MC DSCB fit");
            ptMC->AddText(Form("#alpha_{L}=%.2f n_{L}=%.2f", tail_alphaL, tail_nL));
            ptMC->AddText(Form("#alpha_{R}=%.2f n_{R}=%.2f", tail_alphaR, tail_nR));
            ptMC->AddText(Form("#mu=%.4f #sigma=%.4f", init_mean, init_sigma));
            mcFrame->addObject(ptMC);
            dFits->cd();
            mcFrame->SetName("frame_mc_signal");
            mcFrame->Write();
            delete mcFrame;
            fout.cd();
          } else {
            printf("  MC fit failed for %s, using defaults\n", label.c_str());
          }
        } else {
          printf("  MC stats too low for %s, using defaults\n", label.c_str());
        }
      } else {
        printf("  missing MC snapshot: %s (using default tails)\n", snap_path_mc.c_str());
      }

      // results holder (3σ baseline + 2σ/4σ band + chi2/ndf)
      std::vector<double> sig_vals(scores.size(), 0.0);          // 3σ window S/sqrt(S+B3)
      std::vector<double> sig_vals_2sigma(scores.size(), 0.0);   // 2σ window upper
      std::vector<double> sig_vals_4sigma(scores.size(), 0.0);   // 4σ window lower
      std::vector<double> chi2_ndf_vals(scores.size(), 999.0);   // fit quality
      std::vector<double> sideband_diff_vals(scores.size(), 999.0); // fitted signal yield
      std::vector<double> s3_over_eff_vals(scores.size(), 0.0);   // S(3σ) / ε(BDT)
      std::vector<double> s3_over_eff_vals_err(scores.size(), 0.0); // propagated error

      // 预取该 bin 全部事件的质量与分数，避免每个 score 阈值重复扫描树
      // 使用 ForeachSlot 收集每个 slot 的局部向量，减少锁开销，再合并并按分数降序一次排序
      const auto nSlots = df.GetNSlots();
      std::vector<std::vector<std::pair<float,double>>> slotPairs(nSlots);
      df.ForeachSlot([&](unsigned int slot, double mass, float score){
        slotPairs[slot].emplace_back(score, mass);
      }, {"fMassH3L", "model_output"});

      std::vector<std::pair<float,double>> events;
      size_t totalEv = 0; for (const auto &v : slotPairs) totalEv += v.size();
      events.reserve(totalEv);
      for (auto &v : slotPairs) {
        events.insert(events.end(), v.begin(), v.end());
      }
      std::sort(events.begin(), events.end(), [](const auto &a, const auto &b){ return a.first > b.first; });
      // 为真正增量：单一 RooDataSet，按降序遍历 score 阈值时只追加新事件（events 已降序）
      // 创建数据集（初始为空）
      RooArgSet global_vars(m_global);
      RooDataSet dataSetIncremental("dataSetIncremental","dataSetIncremental", global_vars);
      // 累积直方图（用于侧带逐点残差评估）
      TH1D hCum("hCum","hCum", mass_nbins, mmin, mmax);
      // 记录当前已添加到数据集的事件数（指向 events 前缀）
      size_t ptr_added = 0;
      // 为输出保持原始顺序，需要一个按分数降序的 index 数组
      std::vector<int> idx_scores(scores.size());
      for(size_t i=0;i<scores.size();++i) idx_scores[i]=static_cast<int>(i);
      std::sort(idx_scores.begin(), idx_scores.end(), [&](int a, int b){ return scores[a] > scores[b]; });
      // 拟合参数缓存（用于下一次拟合初值加速）
      bool have_prev_fit = false;
      double prev_c0 = 0.0;
      double prev_c1 = 0.0;
      double prev_nsig = 0.0;
      double prev_nbkg = 0.0;
      double prev_mean = init_mean;
      double prev_sigma = init_sigma;

      for(size_t ord=0; ord<idx_scores.size(); ++ord){
        int original_index = idx_scores[ord];
        double sc = scores[original_index];
        // 追加新事件（score >= sc 且尚未添加）
        while(ptr_added < events.size() && events[ptr_added].first >= sc){
          double mv = events[ptr_added].second;
          if(mv >= mmin && mv <= mmax){
            m_global.setVal(mv);
            dataSetIncremental.add(global_vars);
            hCum.Fill(mv);
          }
          ++ptr_added;
        }
        size_t nPass = dataSetIncremental.numEntries();
        if(nPass < (size_t)min_entries_for_fit){ sig_vals[original_index]=0; chi2_ndf_vals[original_index]=999.0;  continue; }
        // 构建此次拟合所需 PDF 变量（复用缓存初值）
        RooRealVar c0("c0","c0", have_prev_fit?prev_c0:0.0, -10.0, 10.0);
        RooRealVar c1("c1","c1", have_prev_fit?prev_c1:0.0, -10.0, 10.0);
        RooArgList coeffs(c0, c1);
        RooChebychev bkg("bkg","bkg", m_global, coeffs);
        double lo1 = side_low.size()>0 ? side_low[0] : 2.96;
        double lo2 = side_low.size()>1 ? side_low[1] : 2.98;
        double hi1 = side_high.size()>0 ? side_high[0] : 3.005;
        double hi2 = side_high.size()>1 ? side_high[1] : 3.04;
        m_global.setRange("side_lo", lo1, lo2);
        m_global.setRange("side_hi", hi1, hi2);
        bkg.fitTo(dataSetIncremental, RooFit::Range("side_lo"), RooFit::PrintLevel(-1));
        bkg.fitTo(dataSetIncremental, RooFit::Range("side_hi"), RooFit::PrintLevel(-1));
        RooRealVar mean("mean","mean", have_prev_fit?prev_mean:init_mean, 2.985, 3.005);
        RooRealVar sigma("sigma","sigma", have_prev_fit?prev_sigma:init_sigma, 1.4e-3, 2.0e-3);
        RooRealVar a1("a1","a1", tail_alphaL);
        a1.setConstant(true);
        RooRealVar n1("n1","n1", tail_nL);
        n1.setConstant(true);
        RooRealVar a2("a2","a2", tail_alphaR);
        a2.setConstant(true);
        RooRealVar n2("n2","n2", tail_nR);
        n2.setConstant(true);
        RooCrystalBall signal("signal","dscb", m_global, mean, sigma, a1, n1, a2, n2);
        RooRealVar nsig("nsig","nsig", have_prev_fit?prev_nsig:0.5 * nPass, 0.0, 5*nPass);
        RooRealVar nbkg("nbkg","nbkg", have_prev_fit?prev_nbkg:0.5 * nPass, 0.0, 5*nPass + 10.0);
        RooAddPdf totalPdf("totalPdf","signal+bkg", RooArgList(signal, bkg), RooArgList(nsig, nbkg));
        RooFitResult *res = totalPdf.fitTo(dataSetIncremental, RooFit::Extended(true), RooFit::Save(true), RooFit::PrintLevel(-1));
        if(!res){ sig_vals[original_index]=0; chi2_ndf_vals[original_index]=999.0;  continue; }
        // 缓存当前拟合参数用于下一次初值
        prev_c0 = c0.getVal();
        prev_c1 = c1.getVal();
        prev_mean = mean.getVal();
        prev_sigma = sigma.getVal();
        prev_nsig = nsig.getVal();
        prev_nbkg = nbkg.getVal();
        have_prev_fit = true;
        RooPlot *fitFrame = m_global.frame(mass_nbins);
        dataSetIncremental.plotOn(fitFrame, RooFit::Name("data"));
        totalPdf.plotOn(fitFrame, RooFit::Name("pdf"));
        double chi2ndf = fitFrame->chiSquare("pdf","data");
        totalPdf.plotOn(fitFrame,
                          RooFit::Components("bkg"),
                          RooFit::LineStyle(kDashed),
                          RooFit::LineColor(kRed+1),
                          RooFit::Name("pdf_bkg"));
        totalPdf.plotOn(fitFrame,
                          RooFit::Components("signal"),
                          RooFit::LineStyle(kDotted),
                          RooFit::LineColor(kGreen+1),
                          RooFit::Name("pdf_sig"));
        chi2_ndf_vals[original_index] = chi2ndf;
        double s_lo3 = mean.getVal() - 3.0 * sigma.getVal();
        double s_hi3 = mean.getVal() + 3.0 * sigma.getVal();
        m_global.setRange("sigwin3", s_lo3, s_hi3);
        std::unique_ptr<RooAbsReal> intBkg3(bkg.createIntegral(RooArgSet(m_global), RooArgSet(m_global), "sigwin3"));
        std::unique_ptr<RooAbsReal> intSig3(signal.createIntegral(RooArgSet(m_global), RooArgSet(m_global), "sigwin3"));
        double B3 = intBkg3 ? nbkg.getVal() * intBkg3->getVal() : 0.0;
        double Sfixed = fixed_signal_yield;
        double S3 = intSig3 ? nsig.getVal() * intSig3->getVal() : 0.0;
        double S3_err = intSig3 ? nsig.getError() * intSig3->getVal() : 0.0;
        double base_signif3 = (Sfixed+B3>0) ? Sfixed/std::sqrt(Sfixed+B3) : 0.0;
        double eff_here = (original_index < (int)effs.size() ? effs[original_index] : 1.0);
        double signif3_val = eff_here * base_signif3;
        double signifi_org = (S3+B3>0) ? S3/std::sqrt(S3+B3) : 0.0;
        sig_vals[original_index] = signif3_val;
        s3_over_eff_vals[original_index] = (eff_here > 0.0) ? (S3 / eff_here) : 0.0;
        s3_over_eff_vals_err[original_index] = (eff_here > 0.0) ? (S3_err / eff_here) : 0.0;
        double s_lo2 = mean.getVal() - 2.0 * sigma.getVal();
        double s_hi2 = mean.getVal() + 2.0 * sigma.getVal();
        m_global.setRange("sigwin2", s_lo2, s_hi2);
        std::unique_ptr<RooAbsReal> intBkg2(bkg.createIntegral(RooArgSet(m_global), RooArgSet(m_global), "sigwin2"));
        double B2 = intBkg2 ? nbkg.getVal() * intBkg2->getVal() : 0.0;
        double base_signif2 = (Sfixed+B2>0) ? Sfixed/std::sqrt(Sfixed+B2) : 0.0;
        sig_vals_2sigma[original_index] = eff_here * base_signif2;
        double s_lo4 = mean.getVal() - 4.0 * sigma.getVal();
        double s_hi4 = mean.getVal() + 4.0 * sigma.getVal();
        m_global.setRange("sigwin4", s_lo4, s_hi4);
        std::unique_ptr<RooAbsReal> intBkg4(bkg.createIntegral(RooArgSet(m_global), RooArgSet(m_global), "sigwin4"));
        double B4 = intBkg4 ? nbkg.getVal() * intBkg4->getVal() : 0.0;
        double base_signif4 = (Sfixed+B4>0) ? Sfixed/std::sqrt(Sfixed+B4) : 0.0;
        sig_vals_4sigma[original_index] = eff_here * base_signif4;
        // 侧带逐点（逐 bin）绝对残差均值：|data_bin - pred_bin| 在 sidebands 上的均值
        int nSideBins = 0; double sumAbsDiff = 0.0; double sumData = 0.0;
        for(int ib=1; ib<=hCum.GetNbinsX(); ++ib){
          double binLo = hCum.GetXaxis()->GetBinLowEdge(ib);
          double binHi = hCum.GetXaxis()->GetBinUpEdge(ib);
          bool inLo = (binLo>= (side_low.size()>0?side_low[0]:2.96) && binHi<= (side_low.size()>1?side_low[1]:2.98));
          bool inHi = (binLo>= (side_high.size()>0?side_high[0]:3.005) && binHi<= (side_high.size()>1?side_high[1]:3.04));
          if(!(inLo || inHi)) continue;
          m_global.setRange("binRange", binLo, binHi);
          std::unique_ptr<RooAbsReal> intB(bkg.createIntegral(RooArgSet(m_global), RooArgSet(m_global), "binRange"));
          double pred = intB ? nbkg.getVal() * intB->getVal() : 0.0;
          double data = hCum.GetBinContent(ib);
          sumAbsDiff += (data - pred);
          sumData += data;
          ++nSideBins;
        }
        double sideband_rel_diff = (nSideBins>0) ? std::fabs(sumAbsDiff / sumData) : 999.0;
        sideband_diff_vals[original_index] = sideband_rel_diff;
        TPaveText *ptInfo = new TPaveText(0.14, 0.6, 0.42, 0.9, "NDC");
        ptInfo->SetBorderSize(0); ptInfo->SetFillStyle(0); ptInfo->SetTextFont(42); ptInfo->SetTextAlign(11);
        if (aliceperformance) ptInfo->AddText("ALICE Performance");
        else ptInfo->AddText(period_text.c_str()); 
        if(!additional_text.empty()) ptInfo->AddText(additional_text.c_str());
        ptInfo->AddText(Form("Fixed S = %.0f", Sfixed));
        ptInfo->AddText(Form("S(3#sigma)=%.1f", S3));
        ptInfo->AddText(Form("B(3#sigma)=%.1f", B3));
        ptInfo->AddText(Form("S/#sqrt{(S+B)} = %.2f", signifi_org));
        ptInfo->AddText(Form("#chi^{2}/NDF=%.2f", chi2ndf));
        ptInfo->AddText(Form("Side #Delta_{abs}^{avg}=%.3f", sideband_rel_diff));
        ptInfo->AddText(Form("N_{s}/#sqrt{(N_{s}+N_{B})} #times #epsilon(#it{BDT}): %.2f", signif3_val));
        ptInfo->AddText((chi2ndf <= max_chi2_ndf && sideband_rel_diff <= max_sideband_rel_diff) ? "Fit PASS" : "Fit FAIL(excluded)" );
        fitFrame->addObject(ptInfo);
        // 另起一个 Text：拟合参数与 BDT 信息
        TPaveText *ptPars = new TPaveText(0.632, 0.5, 0.932, 0.85, "NDC");
        ptPars->SetBorderSize(0); ptPars->SetFillStyle(0); ptPars->SetTextFont(42); ptPars->SetTextAlign(11);
        ptPars->AddText(Form("BDT score>%.3f #epsilon(#it{BDT})=%.3f", sc, eff_here));
        ptPars->AddText(Form("#mu=%.4f #sigma=%.4f", mean.getVal(), sigma.getVal()));
        ptPars->AddText(Form("a1=%.2f n1=%.2f a2=%.2f n2=%.2f", a1.getVal(), n1.getVal(), a2.getVal(), n2.getVal()));
        ptPars->AddText(Form("c0=%.3f c1=%.3f", c0.getVal(), c1.getVal()));
        ptPars->AddText(Form("nsig_{fac}=%.1f nbkg_{fac}=%.1f", nsig.getVal(), nbkg.getVal()));
        fitFrame->addObject(ptPars);
        dFits->cd();
        fitFrame->SetName(Form("frame_score_%0.3f", sc));
        fitFrame->Write();
        if(chi2ndf > max_chi2_ndf){ sig_vals[original_index] = -1.0; }
      }
      // 原循环已替换为增量式构建与拟合

      // significance vs score（仅包含通过 chi2/NDF 的点）+ band (2σ-4σ) + best point annotation
      std::vector<double> passScores; passScores.reserve(scores.size());
      std::vector<double> passEffs;   passEffs.reserve(scores.size());
      std::vector<double> passSig3;   passSig3.reserve(scores.size());
      std::vector<double> passSig2;   passSig2.reserve(scores.size());
      std::vector<double> passSig4;   passSig4.reserve(scores.size());
      std::vector<double> passS3OverEff;  passS3OverEff.reserve(scores.size());
      std::vector<double> passSig3Err; passSig3Err.reserve(scores.size());
      for(size_t i=0;i<scores.size();++i){
        if(sig_vals[i] >= 0.0 && chi2_ndf_vals[i] <= max_chi2_ndf && sideband_diff_vals[i] <= max_sideband_rel_diff){
          passScores.push_back(scores[i]);
          if(i < effs.size()) passEffs.push_back(effs[i]); else passEffs.push_back(0.0);
          passSig3.push_back(sig_vals[i]);
          passSig2.push_back(sig_vals_2sigma[i]);
          passSig4.push_back(sig_vals_4sigma[i]);
          passS3OverEff.push_back(s3_over_eff_vals[i]);
          passSig3Err.push_back(s3_over_eff_vals_err[i]);
        }
      }

      // find best WP (max significance)
      int bestIdx = -1; double bestSig = -1.0; double bestScore = 0.0; double bestEff = 0.0;
      for(int i=0;i<(int)passSig3.size();++i){
        if(passSig3[i] >= 0.0 && passSig3[i] > bestSig){ bestSig = passSig3[i]; bestIdx = i; }
      }
      double bestS3OverEff = 0.0;
      if(bestIdx >= 0){
        bestScore = passScores[bestIdx];
        if((size_t)bestIdx < passEffs.size()) bestEff = passEffs[bestIdx];
        if((size_t)bestIdx < passS3OverEff.size()) bestS3OverEff = passS3OverEff[bestIdx];
      }

      dSigs->cd();
      if(!passScores.empty()){
        // 3σ curve (× eff)
        TGraph grPass((int)passScores.size());
        for(int i=0;i<(int)passScores.size();++i) grPass.SetPoint(i, passScores[i], passSig3[i]);
        grPass.SetName("gr_significance_vs_score_3sigma");
        grPass.SetTitle((ctx.desc + ";BDT score;Expected significance (3#sigma) #times eff").c_str());
        grPass.SetLineWidth(2);
        grPass.SetLineColor(kBlack);
        grPass.Write();

        // band polygon（±1σ band：上边界用2σ窗口，下边界用4σ窗口）
        TGraph band;
        int npts_band = (int)passScores.size();
        for(int i=0;i<npts_band;++i) band.SetPoint(i, passScores[i], passSig2[i]);
        for(int i=0;i<npts_band;++i) band.SetPoint(npts_band + i, passScores[npts_band-1-i], passSig4[npts_band-1-i]);
        band.SetName("gr_significance_band_pm1sigma");
        band.SetFillColorAlpha(kAzure+1, 0.30);
        band.SetLineColor(kAzure+2);
        band.Write();

        // best point（本身就来源于通过点集合）
        TGraph grBest(1);
        grBest.SetPoint(0, bestScore, bestSig);
        grBest.SetName("gr_best_point");
        grBest.SetTitle(Form("Best point: score=%.3f, eff=%.3f, sig=%.2f", bestScore, bestEff, bestSig));
        grBest.SetMarkerStyle(29);
        grBest.SetMarkerSize(2.0);
        grBest.SetMarkerColor(kRed+1);
        grBest.Write();
        TGraph grSigEff((int)passEffs.size());
        for(int i=0;i<(int)passEffs.size();++i){ grSigEff.SetPoint(i, passEffs[i], passSig3[i]); }
        grSigEff.SetName("gr_significance_vs_efficiency");
        grSigEff.SetTitle((ctx.desc + ";BDT efficiency;Expected significance (3#sigma) #times eff").c_str());
        grSigEff.SetLineWidth(2);
        grSigEff.SetLineColor(kBlue+1);
        grSigEff.Write();
        // ±1σ band for significance vs efficiency (upper: 2σ window, lower: 4σ window)
        TGraph bandEff;
        int npts_eff = (int)passEffs.size();
        for(int i=0;i<npts_eff;++i) bandEff.SetPoint(i, passEffs[i], passSig2[i]);
        for(int i=0;i<npts_eff;++i) bandEff.SetPoint(npts_eff + i, passEffs[npts_eff-1-i], passSig4[npts_eff-1-i]);
        bandEff.SetName("gr_significance_eff_band_pm1sigma");
        bandEff.SetTitle((ctx.desc + ";BDT efficiency;Expected significance (3#sigma) #times eff").c_str());
        bandEff.SetFillColorAlpha(kAzure+1, 0.30);
        bandEff.SetLineColor(kAzure+2);
        bandEff.Write();
        int nYieldBins = std::max<int>(1, (int)passEffs.size());
        double yieldXmin = (yield_eff_range.size()==2) ? yield_eff_range[0] : 0.0;
        double yieldXmax = (yield_eff_range.size()==2) ? yield_eff_range[1] : 1.0;
        TH1F hYieldEff("h_s3overeff_vs_efficiency", (ctx.desc + ";BDT efficiency;S(3#sigma) / #epsilon(#it{BDT})").c_str(), nYieldBins, yieldXmin, yieldXmax);
        hYieldEff.Sumw2();
        for(int i=0;i<(int)passEffs.size();++i){
          int bin = hYieldEff.FindBin(passEffs[i]);
          hYieldEff.SetBinContent(bin, passS3OverEff[i]);
          hYieldEff.SetBinError(bin, passSig3Err[i]);
        }
        hYieldEff.SetLineWidth(2);
        hYieldEff.SetLineColor(kMagenta+1);
        hYieldEff.SetMarkerStyle(20);
        hYieldEff.SetMarkerColor(kMagenta+1);
        hYieldEff.Write();
        fout.Close();

        // PDF rendering 仅在有通过点时绘制
        TCanvas c("c_sig","c_sig",900,650);
        c.SetLeftMargin(0.12); c.SetRightMargin(0.04); c.SetBottomMargin(0.12); c.SetTopMargin(0.08);
        c.SetGridx(); c.SetGridy();
        band.SetTitle((ctx.desc + ";BDT score;Expected significance (3#sigma) #times eff").c_str());
        band.Draw("AF");
        grPass.Draw("L");
        TGraph grBestDraw(1); grBestDraw.SetPoint(0, bestScore, bestSig);
        grBestDraw.SetMarkerStyle(29); grBestDraw.SetMarkerSize(2.0); grBestDraw.SetMarkerColor(kRed+1);
        grBestDraw.Draw("P");
        // Legend
        TLegend leg(0.25,0.72,0.55,0.92);
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextFont(42);
        leg.AddEntry(&grPass, "3#sigma curve", "l");
        leg.AddEntry(&band,  "#pm1#sigma band", "f");
        leg.AddEntry(&grBestDraw, "Best WP", "p");
        leg.Draw("SAME");
        // WP box
        auto ptWP = std::make_unique<TPaveText>(0.55, 0.68, 0.9, 0.92, "NDC");
        ptWP->SetFillStyle(0);
        ptWP->SetBorderSize(0);
        ptWP->SetTextFont(42);
        ptWP->SetTextAlign(11);
        ptWP->AddText(Form("WP score = %.3f", bestScore));
        ptWP->AddText(Form("#epsilon(#it{BDT}) = %.3f", bestEff));
        ptWP->AddText(Form("N_{s}/#sqrt{(N_{s}+N_{B})} #times #epsilon(#it{BDT}) = %.2f", bestSig));
        ptWP->Draw();
        c.Update();
        c.SaveAs((out_dir+"/sig_vs_score_"+label+".pdf").c_str());
        TCanvas cEff("c_sig_eff","c_sig_eff",900,650);
        cEff.SetLeftMargin(0.12); cEff.SetRightMargin(0.04); cEff.SetBottomMargin(0.12); cEff.SetTopMargin(0.08);
        cEff.SetGridx(); cEff.SetGridy();
        bandEff.Draw("AF");
        grSigEff.Draw("L");
        TGraph grBestEff(1); grBestEff.SetPoint(0, bestEff, bestSig);
        grBestEff.SetMarkerStyle(29); grBestEff.SetMarkerSize(2.0); grBestEff.SetMarkerColor(kRed+1);
        grBestEff.Draw("P");
        TLegend legEff(0.25,0.72,0.55,0.92);
        legEff.SetBorderSize(0); legEff.SetFillStyle(0); legEff.SetTextFont(42);
        legEff.AddEntry(&grSigEff, "3#sigma curve", "l");
        legEff.AddEntry(&bandEff,  "#pm1#sigma band", "f");
        legEff.AddEntry(&grBestEff, "Best WP", "p");
        legEff.Draw("SAME");
        auto ptEff = std::make_unique<TPaveText>(0.6, 0.68, 0.9, 0.92, "NDC");
        ptEff->SetFillStyle(0);
        ptEff->SetBorderSize(0);
        ptEff->SetTextFont(42);
        ptEff->SetTextAlign(11);
        ptEff->AddText(Form("#epsilon(#it{BDT}) = %.3f", bestEff));
        ptEff->AddText(Form("N_{s}/#sqrt{(N_{s}+N_{B})} #times #epsilon(#it{BDT}) = %.2f", bestSig));
        ptEff->Draw();
        cEff.Update();
        cEff.SaveAs((out_dir+"/sig_vs_eff_"+label+".pdf").c_str());
        TCanvas cYield("c_s3eff","c_s3eff",900,650);
        cYield.SetLeftMargin(0.12); cYield.SetRightMargin(0.04); cYield.SetBottomMargin(0.12); cYield.SetTopMargin(0.08);
        cYield.SetGridx(); cYield.SetGridy();
        hYieldEff.Draw("PE");
        TGraph grBestYield(1); grBestYield.SetPoint(0, bestEff, bestS3OverEff);
        grBestYield.SetMarkerStyle(29); grBestYield.SetMarkerSize(2.0); grBestYield.SetMarkerColor(kRed+1);
        grBestYield.Draw("P");
        TLegend legYield(0.25, 0.15, 0.45, 0.3);
        legYield.SetBorderSize(0); legYield.SetFillStyle(0); legYield.SetTextFont(42);
        legYield.AddEntry(&hYieldEff, "S(3#sigma) / #epsilon", "lep");
        legYield.AddEntry(&grBestYield, "Best WP", "p");
        legYield.Draw("SAME");
        auto ptYield = std::make_unique<TPaveText>(0.45, 0.15, 0.9, 0.3, "NDC");
        ptYield->SetFillStyle(0);
        ptYield->SetBorderSize(0);
        ptYield->SetTextFont(42);
        ptYield->SetTextAlign(11);
        ptYield->AddText(Form("#epsilon(#it{BDT}) = %.3f", bestEff));
        ptYield->AddText(Form("S(3#sigma) / #epsilon = %.1f", bestS3OverEff));
        ptYield->Draw();
        cYield.Update();
        cYield.SaveAs((out_dir+"/s3overeff_vs_eff_"+label+".pdf").c_str());
      }

      // upsert working point line to summary vector (preserve other bins)
      upsert_wp_line(ctx.hasCen, ctx.hasPt, ctx.hasCt, ctx.cenmin, ctx.cenmax, ctx.ptmin, ctx.ptmax, ctx.ctmin, ctx.ctmax, bestScore, bestEff, bestSig);

      // 保存最佳工作点对应的拟合图
      if(bestIdx >= 0){
        TFile fbest(out_root.c_str(), "READ");
        TDirectory *dirFits = fbest.GetDirectory("Fits");
        if(dirFits){
          TString frName = Form("frame_score_%0.3f", bestScore);
          RooPlot *frBest = (RooPlot*)dirFits->Get(frName);
          if(frBest){
            TCanvas cbest("c_bestfit","c_bestfit",900,650);
            cbest.SetLeftMargin(0.12); cbest.SetRightMargin(0.04); cbest.SetBottomMargin(0.12); cbest.SetTopMargin(0.08);
            frBest->Draw();
            cbest.SaveAs((out_dir+"/best_fit_"+label+".pdf").c_str());
          }
        }
      }

      return 1;
  };

  // iterate bins per mix_mode 语义
  bool did_any = false;
  if(mix_mode == "pt-ct"){
    for(size_t i_pt=0; i_pt+1<pt_bins.size(); ++i_pt){
      double ptmin = pt_bins[i_pt];
      double ptmax = pt_bins[i_pt+1];
      if(target_pt_range.size()==2){
        if( !(fabs(ptmin - target_pt_range[0])<1e-6 && fabs(ptmax - target_pt_range[1])<1e-6) ) continue;
      }
      if(i_pt >= ct_bins.size()) { printf("ct_bins missing for pt index %zu\n", i_pt); break; }
      const auto &ct_edges = ct_bins[i_pt];
      for(size_t i_ct=0; i_ct+1<ct_edges.size(); ++i_ct){
        double ctmin = ct_edges[i_ct];
        double ctmax = ct_edges[i_ct+1];
        if(target_ct_range.size()==2){
          if( !(fabs(ctmin - target_ct_range[0])<1e-6 && fabs(ctmax - target_ct_range[1])<1e-6) ) continue;
        }
        BinContext ctx; ctx.hasPt=true; ctx.hasCt=true; ctx.mode=0;
        ctx.ptmin=ptmin; ctx.ptmax=ptmax; ctx.ctmin=ctmin; ctx.ctmax=ctmax;
        ctx.label = make_label(false,true,true,0,0,ptmin,ptmax,ctmin,ctmax);
        ctx.desc  = make_desc(false,true,true,0,0,ptmin,ptmax,ctmin,ctmax);
        int status = process_one_bin(ctx); did_any = true;
        if (status < 1){
          printf("  failed processing pt %g-%g ct %g-%g with error type: %d \n", ptmin, ptmax, ctmin, ctmax, status);}
      }
    }
  } else if(mix_mode == "cen-pt"){
    if(cen_bins.size()<2){ printf("cen-pt mode requires cen_bins.\n"); }
    const auto &pt_by_cen = (!pt_bins_by_centrality.empty()) ? pt_bins_by_centrality : std::vector<std::vector<double>>{};
    for(size_t i_c=0; i_c+1<cen_bins.size(); ++i_c){
      double cenmin = cen_bins[i_c];
      double cenmax = cen_bins[i_c+1];
      if(target_cen_range.size()==2){
        if( !(fabs(cenmin - target_cen_range[0])<1e-6 && fabs(cenmax - target_cen_range[1])<1e-6) ) continue;
      }
      std::vector<double> pt_edges;
      if(!pt_by_cen.empty()){
        if(i_c >= pt_by_cen.size()) { printf("pt_bins_by_centrality missing for cen index %zu\n", i_c); break; }
        pt_edges = pt_by_cen[i_c];
      } else {
        pt_edges = pt_bins;
      }
      if(pt_edges.size()<2){ printf("pt bins missing for cen index %zu\n", i_c); continue; }
      for(size_t i_pt=0; i_pt+1<pt_edges.size(); ++i_pt){
        double ptmin = pt_edges[i_pt];
        double ptmax = pt_edges[i_pt+1];
        BinContext ctx; ctx.hasCen=true; ctx.hasPt=true; ctx.hasCt=false; ctx.mode=3;
        ctx.cenmin=cenmin; ctx.cenmax=cenmax; ctx.ptmin=ptmin; ctx.ptmax=ptmax; ctx.ctmin=0; ctx.ctmax=0;
        ctx.label = make_label(true,true,false,cenmin,cenmax,ptmin,ptmax,0,0);
        ctx.desc  = make_desc(true,true,false,cenmin,cenmax,ptmin,ptmax,0,0);
        int status = process_one_bin(ctx); did_any = true;
        if (status < 1){
          printf("  failed processing cen %g-%g pt %g-%g with error type: %d \n", cenmin, cenmax, ptmin, ptmax, status);
        }
      }
    }
  } else if(mix_mode == "pt-ct-single"){
    if(target_pt_range.size()!=2 || target_ct_range.size()!=2){
      printf("pt-ct-single mode requires target_pt_range and target_ct_range (len=2).\n");
    } else {
      BinContext ctx; ctx.hasPt=true; ctx.hasCt=true; ctx.mode=4;
      ctx.ptmin=target_pt_range[0]; ctx.ptmax=target_pt_range[1];
      ctx.ctmin=target_ct_range[0]; ctx.ctmax=target_ct_range[1];
      ctx.label = make_label(false,true,true,0,0,ctx.ptmin,ctx.ptmax,ctx.ctmin,ctx.ctmax);
      ctx.desc  = make_desc(false,true,true,0,0,ctx.ptmin,ctx.ptmax,ctx.ctmin,ctx.ctmax);
      int status = process_one_bin(ctx); did_any = true;
      if (status < 1){ printf("  failed processing single pt-ct bin with error type: %d \n", status); }
    }
  } else if(mix_mode == "pt-single"){
    if(target_pt_range.size()==2){
      BinContext ctx; ctx.hasPt=true; ctx.hasCt=false; ctx.mode=1; ctx.ptmin=target_pt_range[0]; ctx.ptmax=target_pt_range[1];
      ctx.label = make_label(false,true,false,0,0,ctx.ptmin,ctx.ptmax,0,0);
      ctx.desc  = make_desc(false,true,false,0,0,ctx.ptmin,ctx.ptmax,0,0);
      int status = process_one_bin(ctx); did_any = true;
      if (status < 1){ printf("  failed processing pt %g-%g (pt-only) with error type: %d \n", ctx.ptmin, ctx.ptmax, status); }
    } else {
      const std::vector<double> &pt_edges = (pt_bins_single.size()>1) ? pt_bins_single : pt_bins;
      for(size_t i_pt=0; i_pt+1<pt_edges.size(); ++i_pt){
        double ptmin = pt_edges[i_pt];
        double ptmax = pt_edges[i_pt+1];
        BinContext ctx; ctx.hasPt=true; ctx.hasCt=false; ctx.mode=1; ctx.ptmin=ptmin; ctx.ptmax=ptmax;
        ctx.label = make_label(false,true,false,0,0,ptmin,ptmax,0,0);
        ctx.desc  = make_desc(false,true,false,0,0,ptmin,ptmax,0,0);
        int status = process_one_bin(ctx); did_any = true;
        if (status < 1){ printf("  failed processing pt %g-%g (pt-only) with error type: %d \n", ptmin, ptmax, status); }
      }
    }
  } else if(mix_mode == "ct-single"){
    // pt filter optional via target_pt_range (keeps label consistent if provided)
    auto run_ct_bin = [&](double ctmin, double ctmax){
      BinContext ctx; ctx.hasCt=true; ctx.hasPt = (target_pt_range.size()==2); ctx.mode=2;
      ctx.ctmin=ctmin; ctx.ctmax=ctmax;
      if(ctx.hasPt){ ctx.ptmin=target_pt_range[0]; ctx.ptmax=target_pt_range[1]; }
      ctx.label = make_label(false, ctx.hasPt, true, 0,0, ctx.ptmin, ctx.ptmax, ctx.ctmin, ctx.ctmax);
      ctx.desc  = make_desc(false, ctx.hasPt, true, 0,0, ctx.ptmin, ctx.ptmax, ctx.ctmin, ctx.ctmax);
      int status = process_one_bin(ctx); did_any = true;
      if (status < 1){ printf("  failed processing ct %g-%g (ct-only) with error type: %d \n", ctx.ctmin, ctx.ctmax, status); }
    };

    const bool has_single_ct = (target_ct_range.size()==2);
    const bool has_list_ct = (ct_bins_single.size()>1);
    if(has_single_ct){
      run_ct_bin(target_ct_range[0], target_ct_range[1]);
    } else if(has_list_ct){
      for(size_t i_ct=0; i_ct+1<ct_bins_single.size(); ++i_ct){
        run_ct_bin(ct_bins_single[i_ct], ct_bins_single[i_ct+1]);
      }
    } else {
      printf("ct-single mode requires target_ct_range (len=2) or ct_bins_single edges.\n");
    }
  } else {
    printf("Unsupported Mix_mode: %s\n", mix_mode.c_str());
  }

  if(!did_any){
    printf("No bins to process for current mix_mode.\n");
  }

  // write back summary file
  {
    std::vector<std::string> cleaned;
    cleaned.reserve(wp_lines.size());
    for(const auto &ln : wp_lines){
      if(ln.empty()) continue;
      cleaned.push_back(ln);
    }
    std::string hdr = header_for_format(wp_format);
    if(!hdr.empty() && (cleaned.empty() || cleaned[0].rfind("#",0)!=0)){
      cleaned.insert(cleaned.begin(), hdr);
    }
    std::ofstream ofs_wp_out(wp_txt.c_str());
    for(const auto &ln : cleaned) ofs_wp_out << ln << "\n";
  }
  printf("ProcessWP finished. Outputs in %s\n", out_dir.c_str());
}
