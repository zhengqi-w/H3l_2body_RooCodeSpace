#ifndef GENERALHELPER_HPP
#define GENERALHELPER_HPP

// GeneralHelper.hpp
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <TRandom.h>
#include <ROOT/RDataFrame.hxx>
#include <algorithm>

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

namespace GeneralHelper {
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
    std::string genDecLenExpr = "0";
    std::string genPzExpr = "0";
    std::string genPExpr = "0";
    std::string absGenPtExpr = "0";
    std::string genCtExpr = "0";
    if (isMC) {
        genDecLenExpr = "sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx + fGenZDecVtx*fGenZDecVtx)";
        genPzExpr = "fGenPt * sinh(fGenEta)";
        genPExpr = "sqrt(fGenPt*fGenPt + fGenPz*fGenPz)";
        absGenPtExpr = "abs(fGenPt)";
        double factor = (!isH4l) ? 2.99131 : 3.922;
        genCtExpr = std::string("fGenDecLen * ") + std::to_string(factor) + " / fGenP";
    }
    auto out10 = out9.Define("fGenDecLen", genDecLenExpr)
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

} // namespace GeneralHelper

#endif // GENERALHELPER_HPP