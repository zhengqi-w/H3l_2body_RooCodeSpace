#ifndef GENERALHELPER_HPP
#define GENERALHELPER_HPP

// GeneralHelper.hpp
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <cmath>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <unordered_map>
#include <TRandom.h>
#include <ROOT/RDataFrame.hxx>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "TCanvas.h"
#include "TStyle.h"
#include "TH1.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TColor.h"
#include "TROOT.h"
#include "TSystem.h"
// I/O / tree helpers
#include "TFile.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TTree.h"

#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooFit.h>
#include <RooGaussian.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include "TPaveText.h"

namespace GeneralHelper {
using Json = nlohmann::json;

inline Json LoadJsonFile(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    Json j;
    ifs >> j;
    return j;
}

inline void SaveJsonFile(const std::string &path, const Json &j, int indent = 2) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot write JSON file: " + path);
    }
    ofs << j.dump(indent) << "\n";
}

inline void MergeJson(Json &base, const Json &override) {
    if (!override.is_object()) {
        base = override;
        return;
    }
    if (!base.is_object()) {
        base = Json::object();
    }
    for (auto it = override.begin(); it != override.end(); ++it) {
        const auto &key = it.key();
        const auto &val = it.value();
        if (base.contains(key) && base[key].is_object() && val.is_object()) {
            MergeJson(base[key], val);
        } else {
            base[key] = val;
        }
    }
}

// open EnableImplicitMT with preferred number of threads
inline void EnableImplicitMTWithPreferredThreads() {
  unsigned int preferred = std::thread::hardware_concurrency();
      if (preferred == 0) {
          preferred = 4;
      }
      const unsigned int nThreads = std::clamp(preferred, 2u, 12u);
      if (!ROOT::IsImplicitMTEnabled()) {
          ROOT::EnableImplicitMT(nThreads);
          std::cout << "[Info] Enabled ROOT implicit MT with " << nThreads << " threads\n";
  }
}
//func to fill TChain from AO2D files
inline void fillChainFromAO2D(TChain &chain, TFile* file)
{
    if (!file || file->IsZombie()) {
        std::cerr << "Invalid TFile pointer!" << std::endl;
        return;
    }
    TString fileName = file->GetName();
    TIter nextKey(file->GetListOfKeys());
    TKey* key = nullptr;
    while ((key = (TKey*)nextKey())) {
        TString keyName = key->GetName();
        if (keyName.BeginsWith("DF_")) {
            TString fullPath = fileName + "/" + keyName + "/" + chain.GetName();
            TObject* obj = file->Get((keyName + "/" + chain.GetName()));
            if (!obj) {
                std::cerr << "Warning: tree " << chain.GetName()
                          << " not found in " << keyName << std::endl;
                continue;
            }
            chain.Add(fullPath);
        }
    }
}

// Convert and add common derived columns to an input RDataFrame.
// Mirrors the logic in the Python utils.correct_and_convert_df RDF branch.
// Returns the modified RDataFrame (RDataFrame is cheap to copy).
// ITS helpers (inline at namespace scope). Do NOT define these inside another function -
// nested function definitions are not allowed in standard C++.
inline unsigned int CountITSHits(unsigned long long packed){
    unsigned int n = 0;
    for(int i=0;i<7;i++){
        unsigned int val = (unsigned int)((packed >> (4*i)) & 0xFULL);
        if(val > 0) ++n;
    }
    return n;
}

inline double AvgITSClusterSize(unsigned long long packed){
    unsigned int n = 0;
    unsigned int sum = 0;
    for(int i=0;i<7;i++){
        unsigned int val = (unsigned int)((packed >> (4*i)) & 0xFULL);
        if(val > 0){ sum += val; ++n; }
    }
    if(n == 0) return 0.0;
    return static_cast<double>(sum)/static_cast<double>(n);
}

template <typename RDFType>
inline auto CorrectAndConvertRDF(RDFType rdf, bool calibrate_he3_pt = false, bool isMC = false, bool isH4l = false)
{
    // We'll create sequential temporaries to avoid assigning different RInterface types
    auto out0 = rdf;
    auto cols0 = out0.GetColumnNames();
    auto has0 = [&](const std::string &n){ return std::find(cols0.begin(), cols0.end(), n) != cols0.end(); };

    // fFlags
    // Define fHePIDHypo / fPiPIDHypo unconditionally (assumes fFlags exists in input)
    auto out1 = out0.Define("fHePIDHypo", "(int)(fFlags >> 4)")
                    .Define("fPiPIDHypo", "(int)(fFlags & 0xF)");
    auto cols1 = out1.GetColumnNames();

    // calibrate he3 pt
    // Define fPtHe3 unconditionally: if calibrate_he3_pt is false we just define it as the original column
    std::string fPtExpr = "fPtHe3";
    if (calibrate_he3_pt) {
        fPtExpr = R"RAW(((fHePIDHypo==6) ? (fPtHe3 + (-0.1286 - 0.1269 * fPtHe3 + 0.06 * fPtHe3*fPtHe3)) : (fPtHe3 + 2.98019e-02 + 7.66100e-01 * exp(-1.31641e+00 * fPtHe3)))) )RAW";
    }
    auto out2 = out1.Redefine("fPtHe3", fPtExpr);
    auto cols2 = out2.GetColumnNames();

    // 3He momentum & energies
    auto out3 = out2.Define("fPxHe3", "fPtHe3 * cos(fPhiHe3)")
                    .Define("fPyHe3", "fPtHe3 * sin(fPhiHe3)")
                    .Define("fPzHe3", "fPtHe3 * sinh(fEtaHe3)")
                    .Define("fPHe3",  "fPtHe3 * cosh(fEtaHe3)")
                    .Define("fEnHe3", "sqrt(fPHe3*fPHe3 + 2.8083916*2.8083916)")
                    .Define("fEnHe4", "sqrt(fPHe3*fPHe3 + 3.7273794*3.7273794)");

    // pion momentum & energy
    auto out4 = out3.Define("fPxPi", "fPtPi * cos(fPhiPi)")
                    .Define("fPyPi", "fPtPi * sin(fPhiPi)")
                    .Define("fPzPi", "fPtPi * sinh(fEtaPi)")
                    .Define("fPPi",  "fPtPi * cosh(fEtaPi)")
                    .Define("fEnPi", "sqrt(fPPi*fPPi + 0.139570*0.139570)");

    // hypertriton kinematics
    auto out5 = out4.Define("fPx", "fPxHe3 + fPxPi")
                    .Define("fPy", "fPyHe3 + fPyPi")
                    .Define("fPz", "fPzHe3 + fPzPi")
                    .Define("fP",  "sqrt(fPx*fPx + fPy*fPy + fPz*fPz)")
                    .Define("fEn", "fEnHe3 + fEnPi")
                    .Define("fEn4", "fEnHe4 + fEnPi");

    // derived momentum variables
    auto out6 = out5.Define("fPt", "sqrt(fPx*fPx + fPy*fPy)")
                    .Define("fEta", "acosh(fP / fPt)")
                    .Define("fCosLambda", "fPt / fP")
                    .Define("fCosLambdaHe", "fPtHe3 / fPHe3");

    // decay lengths, ct
    decltype(out6) out7 = out6;
    if (!isH4l) {
        out7 = out6.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)")
                   .Define("fCt", "fDecLen * 2.99131 / fP");
    } else {
        out7 = out6.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)")
                   .Define("fCt", "fDecLen * 3.922 / fP");
    }

    auto out8 = out7.Define("fDecRad", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx)")
                    .Define("fCosPA", "(fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)")
                    .Define("fMassH3L", "sqrt(fEn*fEn - fP*fP)")
                    .Define("fMassH4L", "sqrt(fEn4*fEn4 - fP*fP)");

    // simple signed momenta
    auto out9 = out8.Define("fTPCSignMomHe3", "fTPCmomHe * (-1 + 2*fIsMatter)")
                    .Define("fGloSignMomHe3", "fPHe3 / 2. * (-1 + 2*fIsMatter)");

    // if MC add generator-level derived vars
    // Choose expressions at runtime so we can define columns unconditionally and avoid type-assign issues
    std::string genDecRadExpr = "0";
    std::string genDecLenExpr = "0";
    std::string genPzExpr = "0";
    std::string genPExpr = "0";
    std::string absGenPtExpr = "0";
    std::string genCtExpr = "0";
    if (isMC) {
        genDecRadExpr = "sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx)";
        genDecLenExpr = "sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx + fGenZDecVtx*fGenZDecVtx)";
        genPzExpr = "fGenPt * sinh(fGenEta)";
        genPExpr = "sqrt(fGenPt*fGenPt + fGenPz*fGenPz)";
        absGenPtExpr = "abs(fGenPt)";
        double factor = (!isH4l) ? 2.99131 : 3.922;
        genCtExpr = std::string("fGenDecLen * ") + std::to_string(factor) + " / fGenP";
    }
    auto out10 = out9.Define("fGenDecRad", genDecRadExpr)
                    .Define("fGenDecLen", genDecLenExpr)
                    .Define("fGenPz", genPzExpr)
                    .Define("fGenP", genPExpr)
                    .Define("fAbsGenPt", absGenPtExpr)
                    .Define("fGenCt", genCtExpr);

    // ITS cluster columns: pick expressions depending on whether packed ITS columns exist
    auto cols10 = out10.GetColumnNames();
    bool hasITS = (std::find(cols10.begin(), cols10.end(), std::string("fITSclusterSizesHe")) != cols10.end() &&
                   std::find(cols10.begin(), cols10.end(), std::string("fITSclusterSizesPi")) != cols10.end());
    std::string avgHeExpr = hasITS ? std::string("GeneralHelper::AvgITSClusterSize(fITSclusterSizesHe)") : std::string("0");
    std::string nHeExpr = hasITS ? std::string("GeneralHelper::CountITSHits(fITSclusterSizesHe)") : std::string("0");
    std::string avgPiExpr = hasITS ? std::string("GeneralHelper::AvgITSClusterSize(fITSclusterSizesPi)") : std::string("0");
    std::string nPiExpr = hasITS ? std::string("GeneralHelper::CountITSHits(fITSclusterSizesPi)") : std::string("0");
    std::string avgCosExpr = hasITS ? std::string("fAvgClusterSizeHe * fCosLambdaHe") : std::string("0");

    auto out11 = out10.Define("fAvgClusterSizeHe", avgHeExpr)
                     .Define("nITSHitsHe", nHeExpr)
                     .Define("fAvgClusterSizePi", avgPiExpr)
                     .Define("nITSHitsPi", nPiExpr)
                     .Define("fAvgClSizeCosLambda", avgCosExpr);

    bool hasPsi = (std::find(cols10.begin(), cols10.end(), std::string("fPsiFT0C")) != cols10.end());
    std::string phiExpr = hasPsi ? std::string("atan2(fPy, fPx)") : std::string("0");
    std::string v2Expr = hasPsi ? std::string("cos(2*(fPhi - fPsiFT0C))") : std::string("0");
    auto out12 = out11.Define("fPhi", phiExpr)
                      .Define("fV2", v2Expr);

    return out12;
}

template <typename RDFType>
inline auto ReWeightSpectrum(RDFType rdf, TF1* distribution, const std::string& varName, TRandom* randGen = nullptr)
{
    if (!distribution) {
        throw std::runtime_error("Distribution TF1 pointer is null");
    }
    if (!randGen) {
        randGen = gRandom;
    }
    float max_bw = distribution->GetMaximum();
    if (max_bw <= 0) {
        throw std::runtime_error("Distribution maximum <= 0");
    }
    return rdf.Define("rej", [distribution, randGen, max_bw](float x) {
        return (randGen->Uniform() > distribution->Eval(x)/max_bw) ? -1 : 1;
    }, {varName}).Filter([](int rej) { return rej >= 0; }, {"rej"});
}

inline std::vector<TH1*> CopyTH1Vector(const std::vector<TH1*>& src, const std::string& suffix = "_copy") {
    std::vector<TH1*> out;
    out.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        TH1* h = src[i];
        if (!h) { out.push_back(nullptr); continue; }
        const char* name = h->GetName() ? h->GetName() : Form("hist_%zu", i);
        std::string newName = std::string(name) + suffix;
        // try cloning to TH1 directly
        TH1* c = dynamic_cast<TH1*>(h->Clone(newName.c_str()));
        if (c) {
            c->SetDirectory(nullptr);
            out.push_back(c);
            continue;
        }
        // fallback: create TH1 with same binning and copy contents
        else {
            cout << "Warning: histogram " << name << " is not TH1, performing manual copy to TH1D.\n";
            int nb = h->GetNbinsX();
            double xmin = h->GetXaxis()->GetXmin();
            double xmax = h->GetXaxis()->GetXmax();
            TH1D* nf = new TH1D(newName.c_str(), h->GetTitle() ? h->GetTitle() : newName.c_str(), nb, xmin, xmax);
            nf->SetDirectory(nullptr);
            for (int b = 1; b <= nb; ++b) {
                nf->SetBinContent(b, h->GetBinContent(b));
                nf->SetBinError(b, h->GetBinError(b));
            }
            out.push_back(nf);
        } 
    }
    return out;
}

inline void EnsureDir(const std::string& path) {
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(p, ec)) {
            if (ec) {
                std::cerr << "Error creating directory " << p 
                          << ": " << ec.message() << std::endl;
            }
        } else {
            std::cout << "Created directory: " << p << std::endl;
        }
    }
}


// 设置一个通用的绘图风格（会修改全局 gStyle）
inline void SetDefaultStyle(bool grid = true) {
    gStyle->Reset("Plain");
    gStyle->SetCanvasColor(0);
    gStyle->SetPadColor(0);
    gStyle->SetFrameFillColor(0);
    gStyle->SetStatColor(0);
    gStyle->SetOptStat(0);        // 默认不显示统计框
    gStyle->SetOptTitle(0);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(0);
    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadTopMargin(0.08);
    gStyle->SetPadBottomMargin(0.12);
    gStyle->SetTitleSize(0.05, "xyz");
    gStyle->SetLabelSize(0.04, "xyz");
    gStyle->SetNdivisions(510, "x");
    gStyle->SetNdivisions(510, "y");
    gStyle->SetLineWidth(2);
    gStyle->SetHistLineWidth(2);
    gStyle->SetGridStyle(3);
    gStyle->SetGridColor(kGray+1);
    gStyle->SetGridWidth(1);
    gStyle->SetPadGridX(grid);
    gStyle->SetPadGridY(grid);
}

// 创建并返回一个 TCanvas，默认 800x600
inline TCanvas* CreateCanvas(const std::string& name = "c",
                             const std::string& title = "Canvas",
                             int width = 800, int height = 600,
                             bool logx = false, bool logy = false, bool logz = false) {
    TCanvas* c = new TCanvas(name.c_str(), title.c_str(), width, height);
    c->cd();
    c->SetTicks(1,1);
    c->SetRightMargin(0.05);
    c->SetLeftMargin(0.12);
    if (logx) c->SetLogx();
    if (logy) c->SetLogy();
    if (logz) c->SetLogz();
    return c;
}

// 在当前画布上添加文本（使用 NDC 坐标）
// x,y 为 NDC 坐标（0-1），默认左上角原点方向
inline TLatex* AddText(double x, double y, const std::string& text,
                       double size = 0.04, int color = kBlack, int align = 11, int font = 42) {
    TLatex* tl = new TLatex(x, y, text.c_str());
    tl->SetTextFont(font);
    tl->SetTextSize(size);
    tl->SetTextColor(color);
    tl->SetNDC();
    tl->SetTextAlign(align);
    tl->Draw();
    return tl;
}

// 在当前画布上添加多行文本（每行向下偏移 lineSpacing）
// lines: 每一行为一个字符串
inline std::vector<TLatex*> AddTextBlock(double x, double y, const std::vector<std::string>& lines,
                                         double size = 0.04, double lineSpacing = 1.1,
                                         int color = kBlack, int align = 11, int font = 42) {
    std::vector<TLatex*> out;
    double curY = y;
    for (const auto& l : lines) {
        TLatex* t = AddText(x, curY, l, size, color, align, font);
        out.push_back(t);
        curY -= size * lineSpacing;
    }
    return out;
}

// 简单绘制 TH1（会将画布切换到 c，如果 c==nullptr 则使用当前画布）
// option 如 "hist", "E", "hist same", 等。
// 若 clearStats 为 true，则临时关闭统计框。
inline void DrawHistogram(TH1* h, const std::string& option = "hist",
                          TCanvas* c = nullptr, int lineColor = kBlack, int fillColor = 0,
                          bool clearStats = true) {
    if (!h) return;
    int prevStat = gStyle->GetOptStat();
    if (clearStats) gStyle->SetOptStat(0);
    if (c) c->cd();
    h->SetLineColor(lineColor);
    if (fillColor != 0) {
        h->SetFillColor(fillColor);
    }
    h->Draw(option.c_str());
    if (clearStats) gStyle->SetOptStat(prevStat);
}

// 创建并返回一个简单的图例
// entries: pair<objPointer, label>
inline TLegend* CreateLegend(double x1 = 0.65, double y1 = 0.65, double x2 = 0.88, double y2 = 0.88,
                             double textSize = 0.03) {
    TLegend* leg = new TLegend(x1, y1, x2, y2);
    leg->SetTextSize(textSize);
    leg->SetFillColor(0);
    leg->SetBorderSize(0);
    return leg;
}

inline bool SaveCanvas(TCanvas* c, const std::string& filename) {
    if (!c) return false;
    std::string path = filename;
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos);
        if (gSystem && !gSystem->AccessPathName(dir.c_str())) {
            // exists
        } else if (gSystem) {
            gSystem->mkdir(dir.c_str(), true);
        }
    }
    c->SaveAs(filename.c_str());
    return true;
}

struct MassFitConfig {
    double massMin{2.96};
    double massMax{3.04};
    std::vector<double> sigmaRangeMcToData{1.0, 1.5};
    int nBinsMcFrame{80};
    int nBinsDataFrame{40};
};

struct MassFitResult {
    double signal{0.0};
    double signalErr{0.0};
    double significance{0.0};
    double significanceErr{0.0};
    double meanData{0.0};
    double meanDataErr{0.0};
    double sigmaData{0.0};
    double sigmaDataErr{0.0};
    double sigmaMc{0.0};
    double sigmaMcErr{0.0};
    double chi2Data{0.0};
    double chi2Mc{0.0};
    int ndfData{0};
    int ndfMc{0};
    std::unique_ptr<RooPlot> frame;
    std::unique_ptr<RooPlot> frameMc;
    std::shared_ptr<RooRealVar> massAxis;
};


inline MassFitResult FitMassSpectrum(const std::vector<double> &dataMass,
                                     const std::vector<double> &mcMass,
                                     const MassFitConfig &cfg,
                                     const std::string &bkgFuncRaw,
                                     const std::string &sigFuncRaw) {
    const std::string sigFunc = sigFuncRaw;
    const std::string bkgFunc = bkgFuncRaw;

    const double sigmaScaleMin = cfg.sigmaRangeMcToData.empty() ? 1.0 : cfg.sigmaRangeMcToData.front();
    const double sigmaScaleMax = cfg.sigmaRangeMcToData.size() < 2 ? 1.5 : cfg.sigmaRangeMcToData[1];
    const int nBinsMcFrame = std::max(1, cfg.nBinsMcFrame);
    const int nBinsDataFrame = std::max(1, cfg.nBinsDataFrame);

    RooRealVar mass("m", "Mass(H3l)", cfg.massMin, cfg.massMax, "GeV/c^{2}");
    RooDataSet data("data", "data", RooArgSet(mass));
    int dataCounts = 0;
    for (double v : dataMass) {
        if (v < cfg.massMin || v > cfg.massMax) continue;
        mass.setVal(v);
        data.add(RooArgSet(mass));
        ++dataCounts;
    }

    RooDataSet mc("mc", "mc", RooArgSet(mass));
    for (double v : mcMass) {
        if (v < cfg.massMin || v > cfg.massMax) continue;
        mass.setVal(v);
        mc.add(RooArgSet(mass));
    }

    RooRealVar muMc("muMc", "muMc", 2.991, 2.97, 3.01);
    RooRealVar sigmaMcVar("sigmaMc", "sigmaMc", 1.5e-3, 1.1e-3, 2.1e-3);
    RooRealVar a1McVar("a1Mc", "a1Mc", 1.5, 0.1, 10.0);
    RooRealVar a2McVar("a2Mc", "a2Mc", 1.5, 0.1, 10.0);
    RooRealVar n1McVar("n1Mc", "n1Mc", 5.0, 0.5, 30.0);
    RooRealVar n2McVar("n2Mc", "n2Mc", 5.0, 0.5, 30.0);
    RooAbsPdf *signalPdfMc = nullptr;
    if (sigFunc == "gauss") {
        signalPdfMc = new RooGaussian("sigMc", "sigMc", mass, muMc, sigmaMcVar);
    } else {
        signalPdfMc = new RooCrystalBall("sigMc", "sigMc", mass, muMc, sigmaMcVar, a1McVar, n1McVar, a2McVar, n2McVar);
    }

    signalPdfMc->fitTo(mc, RooFit::Range(2.97, 3.01), RooFit::Save(true), RooFit::PrintLevel(-1));
    if (sigFunc != "gauss") {
        a1McVar.setConstant(); a2McVar.setConstant(); n1McVar.setConstant(); n2McVar.setConstant();
    }
    const double sigmaMc = sigmaMcVar.getVal();
    const double sigmaErrMc = sigmaMcVar.getError();
    const double muMcVal = muMc.getVal();
    const double muErrMc = muMc.getError();
    double a1Mc = 0.0, a1ErrMc = 0.0, n1Mc = 0.0, n1ErrMc = 0.0, a2Mc = 0.0, a2ErrMc = 0.0, n2Mc = 0.0, n2ErrMc = 0.0;
    if (sigFunc != "gaus") {
        a1Mc = a1McVar.getVal();
        a1ErrMc = a1McVar.getError();
        n1Mc = n1McVar.getVal();
        n1ErrMc = n1McVar.getError();
        a2Mc = a2McVar.getVal();
        a2ErrMc = a2McVar.getError();
        n2Mc = n2McVar.getVal();
        n2ErrMc = n2McVar.getError();
    }
    const int nMcFloatParams = ((sigFunc == "gaus") ? 2 : 6);
    const int ndfMc = std::max(1, nBinsMcFrame - nMcFloatParams);
    double chi2OverNdfMc = 0.0;

    RooRealVar mu("mu", "mu", 2.991, 2.985, 2.992);
    RooRealVar sigma("sigma", "sigma", sigmaMc, 1.1e-3, 3e-3);
    RooAbsPdf *signalPdf = nullptr;
    if (sigFunc == "gaus") {
        signalPdf = new RooGaussian("sig", "sig", mass, mu, sigma);
    } else {
        signalPdf = new RooCrystalBall("sig", "sig", mass, mu, sigma, a1McVar, n1McVar, a2McVar, n2McVar);
    }
    sigma.setRange(sigmaScaleMin * sigmaMc, sigmaScaleMax * sigmaMc);

    RooAbsPdf *bkg = nullptr;
    RooRealVar c0("c0", "c0", 0.0, -0.8, 0.8);
    RooRealVar c1("c1", "c1", 0.0, -0.8, 0.8);
    RooRealVar c2("c2", "c2", 0.0, -0.8, 0.8);
    RooRealVar c3("c3", "c3", 0.0, -0.8, 0.8);
    const std::unordered_map<std::string, int> bkgOrders = {
        {"expo", 0},
        {"pol1", 1},
        {"pol2", 2},
        {"pol3", 3},
        {"pol4", 4}
    };
    const int bkgOrder = [&]() {
        auto it = bkgOrders.find(bkgFunc);
        if (it != bkgOrders.end()) {
            return it->second;
        }
        std::cerr << "[FitMassSpectrum] Unknown bkg function '" << bkgFunc
                  << "', fallback to pol2\n";
        return 2;
    }();
    if (bkgOrder == 0) {
        bkg = new RooExponential("bkg", "bkg", mass, c0);
    } else {
        RooArgList coeffs;
        coeffs.add(c0);
        if (bkgOrder >= 2) coeffs.add(c1);
        if (bkgOrder >= 3) coeffs.add(c2);
        if (bkgOrder >= 4) coeffs.add(c3);
        bkg = new RooChebychev("bkg", "bkg", mass, coeffs);
    }

    const double nSigInit = std::max(1.0, 0.7 * static_cast<double>(dataCounts));
    const double nSigMax = std::max(150.0, 3.0 * static_cast<double>(dataCounts));
    const double nBkgInit = std::max(1.0, 0.3 * static_cast<double>(dataCounts));
    const double nBkgMax = std::max(50.0, 1.0 * static_cast<double>(dataCounts));
    RooRealVar nSig("nSig", "nSig", nSigInit, 0.0, nSigMax);
    RooRealVar nBkg("nBkg", "nBkg", nBkgInit, 0.0, nBkgMax);
    RooAddPdf model("model", "total_pdf", RooArgList(*signalPdf, *bkg), RooArgList(nSig, nBkg));
    model.fitTo(data, RooFit::Extended(true), RooFit::Save(true), RooFit::PrintLevel(-1));

    const double muData = mu.getVal();
    const double muErrData = mu.getError();
    const double sigmaData = sigma.getVal();
    const double sigmaErrData = sigma.getError();

    const double windowMin = muData - 3.0 * sigmaData;
    const double windowMax = muData + 3.0 * sigmaData;
    mass.setRange("sigWindow", windowMin, windowMax);
    std::unique_ptr<RooAbsReal> sigIntegral(signalPdf->createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
    std::unique_ptr<RooAbsReal> bkgIntegral(bkg->createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
    const double sigFrac = sigIntegral ? sigIntegral->getVal() : 0.0;
    const double bkgFrac = bkgIntegral ? bkgIntegral->getVal() : 0.0;
    const double signalValue = nSig.getVal();
    const double signalValueErr = nSig.getError();
    const double bkgValue = nBkg.getVal();
    const double bkgValueErr = nBkg.getError();
    const double signalCounts3s = signalValue * sigFrac;
    const double signalCounts3sErr = signalValueErr * sigFrac;
    const double bkgCounts3s = bkgValue * bkgFrac;
    const double bkgCounts3sErr = bkgValueErr * bkgFrac;

    double significance = 0.0;
    double significanceErr = 0.0;
    bool validSignificance = bkgCounts3s + signalCounts3s > 0.0;
    if (validSignificance) {
        significance = signalCounts3s / std::sqrt(signalCounts3s + bkgCounts3s);
        const double dSdSig = std::sqrt(signalCounts3s + bkgCounts3s) - (signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
        const double dBdSig = -(signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
        significanceErr = std::sqrt(std::pow(dSdSig * signalCounts3sErr, 2) + std::pow(dBdSig * bkgCounts3sErr, 2));
    }

    const int nBkgFloatParams = (bkgOrder == 0) ? 1 : bkgOrder;
    const int nDataFloatParams = nBkgFloatParams + ((sigFunc == "gaus") ? 2 : 6);
    const int ndfData = std::max(1, nBinsDataFrame - nDataFloatParams);
    double chi2OverNdfData = 0.0;

    std::unique_ptr<RooPlot> frame;
    std::unique_ptr<RooPlot> frameMc;
    std::shared_ptr<RooRealVar> massHolder = std::make_shared<RooRealVar>(mass);

    frameMc.reset(massHolder->frame(nBinsMcFrame));
    mc.plotOn(frameMc.get(), RooFit::Name("mc"));
    signalPdfMc->plotOn(frameMc.get(), RooFit::LineColor(kRed), RooFit::LineStyle(kDashed), RooFit::Name("sig_fit_mc"));
    chi2OverNdfMc = frameMc->chiSquare("sig_fit_mc", "mc", nMcFloatParams);
    auto textMC = std::make_unique<TPaveText>(0.6, 0.43, 0.9, 0.85, "NDC");
    textMC->SetBorderSize(0);
    textMC->SetFillStyle(0);
    textMC->SetTextAlign(12);
    textMC->AddText(Form("MC Fit Parameters:"));
    textMC->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muMcVal * 1e3, muErrMc * 1e3));
    textMC->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaMc * 1e3, sigmaErrMc * 1e3));
    if (sigFunc != "gaus") {
        textMC->AddText(Form(" #alpha_{l} = %.3f #pm %.3f", a1Mc, a1ErrMc));
        textMC->AddText(Form(" n_{l} = %.3f #pm %.3f", n1Mc, n1ErrMc));
        textMC->AddText(Form(" #alpha_{r} = %.3f #pm %.3f", a2Mc, a2ErrMc));
        textMC->AddText(Form(" n_{r} = %.3f #pm %.3f", n2Mc, n2ErrMc));
    }
    textMC->AddText(Form(" #chi^{2}/NDF = %.2f(NDF:%d)", chi2OverNdfMc , ndfMc));
    frameMc->addObject(textMC.release());

    frame.reset(massHolder->frame(nBinsDataFrame));
    data.plotOn(frame.get(), RooFit::Name("data"));
    model.plotOn(frame.get(), RooFit::Name("total"));
    model.plotOn(frame.get(), RooFit::Components(*bkg), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed + 1), RooFit::Name("bkg"));
    model.plotOn(frame.get(), RooFit::Components(*signalPdf), RooFit::LineStyle(kDotted), RooFit::LineColor(kGreen + 2), RooFit::Name("sig"));
    chi2OverNdfData = frame->chiSquare("total", "data", nDataFloatParams);
    auto textData = std::make_unique<TPaveText>(0.58,0.36,0.88,0.88, "NDC");
    textData->SetBorderSize(0);
    textData->SetFillStyle(0);
    textData->SetTextAlign(12);
    textData->AddText(Form("Data Fit Parameters:"));
    textData->AddText(Form(" S = %.1f #pm %.1f", signalValue, signalValueErr));
    textData->AddText(Form(" B = %.1f #pm %.1f", bkgValue, bkgValueErr));
    if (validSignificance) {
        textData->AddText(Form(" S/#sqrt{S+B} (3#sigma) = %.2f #pm %.2f", significance, significanceErr));
    } else {
        textData->AddText(" Significance = N/A");
    }
    textData->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muData * 1e3, muErrData * 1e3));
    textData->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaData * 1e3, sigmaErrData * 1e3));
    textData->AddText(Form(" #chi^{2}/NDF = %.2f(NDF:%d)", chi2OverNdfData , ndfData));
    frame->addObject(textData.release());

    MassFitResult out;
    out.signal = signalValue;
    out.signalErr = signalValueErr;
    out.significance = significance;
    out.significanceErr = significanceErr;
    out.meanData = muData;
    out.meanDataErr = muErrData;
    out.sigmaData = sigmaData;
    out.sigmaDataErr = sigmaErrData;
    out.sigmaMc = sigmaMc;
    out.sigmaMcErr = sigmaErrMc;
    out.chi2Data = chi2OverNdfData;
    out.chi2Mc = chi2OverNdfMc;
    out.ndfData = ndfData;
    out.ndfMc = ndfMc;
    out.frame = std::move(frame);
    out.frameMc = std::move(frameMc);
    out.massAxis = std::move(massHolder);

    delete bkg;
    delete signalPdf;
    delete signalPdfMc;
    return out;
}

struct WorkingPointResult {
    double score = 0.0;
    double eff = 0.0;
    double significance = 0.0;
    bool found = false;
};

inline bool IsWpBinSet(double minv, double maxv, double eps = 1e-6) {
    return (std::abs(minv - (-1.0)) >= eps) && (std::abs(maxv - (-1.0)) >= eps);
}

inline bool MatchEdge(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

// Simple parser: turn a whitespace separated numeric line into a vector<double>
inline std::vector<double> ParseNumbers(const std::string &line) {
    std::istringstream iss(line);
    std::vector<double> vals;
    double v;
    while (iss >> v) {
        vals.push_back(v);
    }
    return vals;
}

// Unified WP lookup.
// Supports layouts:
//  (1) cenMin cenMax ptMin ptMax ctMin ctMax score eff sig        (>=9)
//  (2) cenMin cenMax ptMin ptMax score eff sig                     (>=7)
//  (3) ptMin ptMax ctMin ctMax score eff sig                       (>=7)
//  (4) ptMin ptMax ctMin ctMax score eff                           (>=6)
//  (5) ptMin ptMax score eff [sig]                                 (>=4)
//  (6) ctMin ctMax score eff [sig]                                 (>=4)
// Sentinel -1 -1 is treated as wildcard for centrality in file/query.
inline WorkingPointResult GetWp(const std::string &wpFile,
                                double cenMin = -1.0, double cenMax = -1.0,
                                double ptMin = -1.0, double ptMax = -1.0,
                                double ctMin = -1.0, double ctMax = -1.0) {
    std::ifstream in(wpFile);
    if (!in.is_open()) {
        std::cerr << "[GetWp] Cannot open WP file: " << wpFile << "\n";
        return {};
    }

    WorkingPointResult res;
    const double eps = 1e-6;
    const bool needCen = IsWpBinSet(cenMin, cenMax, eps);
    const bool needPt = IsWpBinSet(ptMin, ptMax, eps);
    const bool needCt = IsWpBinSet(ctMin, ctMax, eps);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto vals = ParseNumbers(line);
        if (vals.size() < 4) continue;

        auto fillResult = [&](int iScore, int iEff, int iSig) {
            res.score = (iScore >= 0 && iScore < static_cast<int>(vals.size())) ? vals[iScore] : 0.0;
            res.eff = (iEff >= 0 && iEff < static_cast<int>(vals.size())) ? vals[iEff] : 0.0;
            res.significance = (iSig >= 0 && iSig < static_cast<int>(vals.size())) ? vals[iSig] : 0.0;
            res.found = true;
        };

        // (1) full: cen+pt+ct + score/eff/sig
        if (vals.size() >= 9) {
            const double cCenMin = vals[0], cCenMax = vals[1];
            const double cPtMin  = vals[2], cPtMax  = vals[3];
            const double cCtMin  = vals[4], cCtMax  = vals[5];

            const bool cenMatch = !needCen ||
                                  ((MatchEdge(cCenMin, -1.0, eps) && MatchEdge(cCenMax, -1.0, eps)) ||
                                   (MatchEdge(cenMin, -1.0, eps) && MatchEdge(cenMax, -1.0, eps)) ||
                                   (MatchEdge(cCenMin, cenMin, eps) && MatchEdge(cCenMax, cenMax, eps)));
            const bool ptMatch = !needPt || (MatchEdge(cPtMin, ptMin, eps) && MatchEdge(cPtMax, ptMax, eps));
            const bool ctMatch = !needCt || (MatchEdge(cCtMin, ctMin, eps) && MatchEdge(cCtMax, ctMax, eps));
            if (cenMatch && ptMatch && ctMatch) {
                fillResult(6, 7, 8);
                break;
            }
            continue;
        }

        // (2)/(3): 4 edges + score/eff/sig
        if (vals.size() >= 7) {
            const double a0 = vals[0], a1 = vals[1], a2 = vals[2], a3 = vals[3];

            // prefer cen+pt if query asks for cen+pt only
            if (needCen && needPt && !needCt) {
                if (MatchEdge(a0, cenMin, eps) && MatchEdge(a1, cenMax, eps) &&
                    MatchEdge(a2, ptMin, eps)  && MatchEdge(a3, ptMax, eps)) {
                    fillResult(4, 5, 6);
                    break;
                }
            }

            // prefer pt+ct if query asks for pt+ct
            if (needPt && needCt) {
                if (MatchEdge(a0, ptMin, eps) && MatchEdge(a1, ptMax, eps) &&
                    MatchEdge(a2, ctMin, eps) && MatchEdge(a3, ctMax, eps)) {
                    fillResult(4, 5, 6);
                    break;
                }
            }

            // fallback for pt single / ct single:
            // - pt single may come from [pt ct ...] (a0,a1) or [cen pt ...] (a2,a3)
            // - ct single may come from [pt ct ...] (a2,a3) or [ct ...] (a0,a1)
            if (needPt && !needCt && !needCen) {
                if ((MatchEdge(a0, ptMin, eps) && MatchEdge(a1, ptMax, eps)) ||
                    (MatchEdge(a2, ptMin, eps) && MatchEdge(a3, ptMax, eps))) {
                    fillResult(4, 5, 6);
                    break;
                }
            }
            if (needCt && !needPt && !needCen) {
                if ((MatchEdge(a2, ctMin, eps) && MatchEdge(a3, ctMax, eps)) ||
                    (MatchEdge(a0, ctMin, eps) && MatchEdge(a1, ctMax, eps))) {
                    fillResult(4, 5, 6);
                    break;
                }
            }
            continue;
        }

        // (4): 4 edges + score/eff (no significance)
        if (vals.size() >= 6) {
            const double a0 = vals[0], a1 = vals[1], a2 = vals[2], a3 = vals[3];
            if (needPt && needCt && MatchEdge(a0, ptMin, eps) && MatchEdge(a1, ptMax, eps) &&
                MatchEdge(a2, ctMin, eps) && MatchEdge(a3, ctMax, eps)) {
                fillResult(4, 5, -1);
                break;
            }
            if (needCen && needPt && !needCt &&
                MatchEdge(a0, cenMin, eps) && MatchEdge(a1, cenMax, eps) &&
                MatchEdge(a2, ptMin, eps) && MatchEdge(a3, ptMax, eps)) {
                fillResult(4, 5, -1);
                break;
            }
            continue;
        }

        // (5)/(6): single-dimension bins
        if (vals.size() >= 4) {
            const double a0 = vals[0], a1 = vals[1];
            const int iSig = (vals.size() >= 5) ? 4 : -1;
            if (needPt && !needCt && !needCen && MatchEdge(a0, ptMin, eps) && MatchEdge(a1, ptMax, eps)) {
                fillResult(2, 3, iSig);
                break;
            }
            if (needCt && !needPt && !needCen && MatchEdge(a0, ctMin, eps) && MatchEdge(a1, ctMax, eps)) {
                fillResult(2, 3, iSig);
                break;
            }
        }
    }
    return res;
}

// Look up WP by centrality and pT bin in a file shaped like WorkingPoint_Spectrum*.txt
// Columns: cenMin cenMax ptMin ptMax best_score best_eff max_significance
inline WorkingPointResult GetWpForCenPt(const std::string &wpFile,
                                        double cenMin, double cenMax,
                                        double ptMin, double ptMax) {
    return GetWp(wpFile, cenMin, cenMax, ptMin, ptMax, -1.0, -1.0);
}

// Look up WP by pT/ct bin (optionally centrality) in a file like WorkingPoint_Crosssection*.txt
// Supports two layouts:
//  (a) ptMin ptMax ctMin ctMax score eff sig          (7 columns)
//  (b) cenMin cenMax ptMin ptMax ctMin ctMax score eff sig (9 columns, cen may be -1 -1 for wildcard)
inline WorkingPointResult GetWpForPtCt(const std::string &wpFile,
                                       double ptMin, double ptMax,
                                       double ctMin, double ctMax,
                                       double cenMin = -1.0, double cenMax = -1.0) {
    return GetWp(wpFile, cenMin, cenMax, ptMin, ptMax, ctMin, ctMax);
}

// Look up WP for ct-only bins (ctMin ctMax ...)
inline WorkingPointResult GetWpForCtSingle(const std::string &wpFile,
                                           double ctMin, double ctMax) {
    return GetWp(wpFile, -1.0, -1.0, -1.0, -1.0, ctMin, ctMax);
}

// Look up WP for pt-only bins (ptMin ptMax ...)
inline WorkingPointResult GetWpForPtSingle(const std::string &wpFile,
                                           double ptMin, double ptMax) {
    return GetWp(wpFile, -1.0, -1.0, ptMin, ptMax, -1.0, -1.0);
}


} // namespace GeneralHelper

#endif // GENERALHELPER_HPP