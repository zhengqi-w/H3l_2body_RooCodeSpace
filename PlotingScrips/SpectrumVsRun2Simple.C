#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraph.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TKey.h>
#include <TLegend.h>
#include <TObject.h>
#include <TPad.h>
#include <TSystem.h>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

R__LOAD_LIBRARY(libGpad)

struct Run2Ref {
    const char *centTag;
    const char *run2File;
    const char *bwName;
    bool hasRun2;
};

namespace {

TObject *FindObjectRecursive(TDirectory *dir, const std::string &name) {
    if (!dir) return nullptr;

    if (auto *obj = dir->Get(name.c_str())) return obj;

    TIter nextKey(dir->GetListOfKeys());
    TKey *key = nullptr;
    while ((key = dynamic_cast<TKey *>(nextKey()))) {
        TObject *child = key->ReadObj();
        auto *sub = dynamic_cast<TDirectory *>(child);
        if (!sub) continue;
        if (auto *found = FindObjectRecursive(sub, name)) return found;
    }
    return nullptr;
}

TObject *FindFirstGraphRecursive(TDirectory *dir) {
    if (!dir) return nullptr;

    TIter nextKey(dir->GetListOfKeys());
    TKey *key = nullptr;
    while ((key = dynamic_cast<TKey *>(nextKey()))) {
        TObject *obj = key->ReadObj();
        if (!obj) continue;

        if (obj->InheritsFrom(TGraphAsymmErrors::Class()) ||
            obj->InheritsFrom(TGraphErrors::Class()) ||
            obj->InheritsFrom(TGraph::Class())) {
            return obj;
        }

        auto *sub = dynamic_cast<TDirectory *>(obj);
        if (!sub) continue;
        if (auto *found = FindFirstGraphRecursive(sub)) return found;
    }
    return nullptr;
}

TCanvas *LoadCanvasByCentrality(TFile *f, const std::string &centTag, const std::string &canvasName) {
    if (!f || f->IsZombie()) return nullptr;

    const std::string p1 = centTag + "/" + canvasName;
    if (auto *obj = f->Get(p1.c_str())) return dynamic_cast<TCanvas *>(obj);

    const std::string p2 = centTag + "/std/" + canvasName;
    if (auto *obj = f->Get(p2.c_str())) return dynamic_cast<TCanvas *>(obj);

    if (auto *centDirObj = f->Get(centTag.c_str())) {
        if (auto *centDir = dynamic_cast<TDirectory *>(centDirObj)) {
            if (auto *obj = FindObjectRecursive(centDir, canvasName)) {
                return dynamic_cast<TCanvas *>(obj);
            }
        }
    }

    // Fallback: search in whole file (for old flat layouts)
    return dynamic_cast<TCanvas *>(FindObjectRecursive(f, canvasName));
}

TGraphAsymmErrors *LoadGraphAsymm(TFile *f, const std::string &name, const std::string &subdir = "") {
    if (!f || f->IsZombie()) return nullptr;

    TObject *obj = f->Get(name.c_str());
    if (!obj && !subdir.empty()) {
        obj = f->Get((subdir + "/" + name).c_str());
    }
    if (!obj) obj = FindFirstGraphRecursive(f);
    if (!obj) return nullptr;

    if (auto *ga = dynamic_cast<TGraphAsymmErrors *>(obj)) {
        return dynamic_cast<TGraphAsymmErrors *>(ga->Clone(Form("%s_clone", name.c_str())));
    }

    if (auto *ge = dynamic_cast<TGraphErrors *>(obj)) {
        auto *out = new TGraphAsymmErrors(ge->GetN());
        out->SetName(Form("%s_clone", name.c_str()));
        for (int i = 0; i < ge->GetN(); ++i) {
            double x = 0.0, y = 0.0;
            ge->GetPoint(i, x, y);
            const double ex = ge->GetErrorX(i);
            const double ey = ge->GetErrorY(i);
            out->SetPoint(i, x, y);
            out->SetPointError(i, ex, ex, ey, ey);
        }
        return out;
    }

    if (auto *g = dynamic_cast<TGraph *>(obj)) {
        auto *out = new TGraphAsymmErrors(g->GetN());
        out->SetName(Form("%s_clone", name.c_str()));
        for (int i = 0; i < g->GetN(); ++i) {
            double x = 0.0, y = 0.0;
            g->GetPoint(i, x, y);
            out->SetPoint(i, x, y);
            out->SetPointError(i, 0.0, 0.0, 0.0, 0.0);
        }
        return out;
    }

    return nullptr;
}

TF1 *LoadTF1(TFile *f, const std::string &name) {
    if (!f || f->IsZombie()) return nullptr;
    TObject *obj = f->Get(name.c_str());
    if (!obj) return nullptr;

    auto *func = dynamic_cast<TF1 *>(obj);
    return func ? dynamic_cast<TF1 *>(func->Clone(Form("%s_clone", name.c_str()))) : nullptr;
}

TObject *FindPrimaryRun3Drawable(TPad *pad) {
    if (!pad) return nullptr;
    TIter next(pad->GetListOfPrimitives());
    TObject *obj = nullptr;
    while ((obj = next())) {
        if (obj->InheritsFrom(TGraphAsymmErrors::Class())) return obj;
        if (obj->InheritsFrom(TGraph::Class())) return obj;
        if (obj->InheritsFrom(TH1::Class())) return obj;
    }
    return nullptr;
}

bool DeterminePadXRange(TPad *pad, double &xmin, double &xmax) {
    if (!pad) return false;
    xmin = 0.0;
    xmax = 0.0;

    TIter next(pad->GetListOfPrimitives());
    TObject *obj = nullptr;
    while ((obj = next())) {
        if (auto *h = dynamic_cast<TH1 *>(obj)) {
            if (h->GetXaxis()) {
                xmin = h->GetXaxis()->GetXmin();
                xmax = h->GetXaxis()->GetXmax();
                return (xmax > xmin);
            }
        }
        if (auto *g = dynamic_cast<TGraph *>(obj)) {
            const int n = g->GetN();
            if (n <= 0) continue;
            double x = 0.0, y = 0.0;
            double xMinLoc = 1e20;
            double xMaxLoc = -1e20;
            for (int i = 0; i < n; ++i) {
                g->GetPoint(i, x, y);
                xMinLoc = std::min(xMinLoc, x);
                xMaxLoc = std::max(xMaxLoc, x);
            }
            if (xMaxLoc > xMinLoc) {
                xmin = xMinLoc;
                xmax = xMaxLoc;
                return true;
            }
        }
    }
    return false;
}

TLegend *FindLegend(TPad *pad) {
    if (!pad) return nullptr;
    TIter next(pad->GetListOfPrimitives());
    TObject *obj = nullptr;
    while ((obj = next())) {
        auto *leg = dynamic_cast<TLegend *>(obj);
        if (leg) return leg;
    }
    return nullptr;
}

} // namespace

void SpectrumVsRun2Simple(
    const char *run3SpectrumRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
    const char *run2BaseDir = "/Users/zhengqingwang/alice/data/h3l_spec_run2",
    const char *run2GraphName = "Graph1D_y1",
    const char *run2GraphSubdir = "",
    const char *run2BWRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root",
    const char *run2BWName = "BlastWave_H3L_10_30",
    double run2BWXMin = -1.0,
    double run2BWXMax = -1.0,
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/SpectrumvsRun2",
    const char *outputRootName = "Spectrum_vs_run2_simple.root") {

    const std::array<Run2Ref, 4> refs = {{
        {"cen_0_10", "h3l_0_10.root", "BlastWave_H3L_0_10", true},
        {"cen_10_30", "h3l_10_30.root", "BlastWave_H3L_10_30", true},
        {"cen_30_50", "h3l_30_50.root", "BlastWave_H3L_30_50", true},
        {"cen_50_80", "", "", false}
    }};

    std::unique_ptr<TFile> fRun3(TFile::Open(run3SpectrumRoot, "READ"));
    if (!fRun3 || fRun3->IsZombie()) {
        std::cerr << "[SpectrumVsRun2Simple] Cannot open Run3 file: " << run3SpectrumRoot << std::endl;
        return;
    }

    gSystem->mkdir(outputDir, true);
    const std::string outPath = std::string(outputDir) + "/" + std::string(outputRootName);

    TFile fOut(outPath.c_str(), "RECREATE");
    if (fOut.IsZombie()) {
        std::cerr << "[SpectrumVsRun2Simple] Cannot create output file: " << outPath << std::endl;
        return;
    }

    std::unique_ptr<TFile> fBW(TFile::Open(run2BWRoot, "READ"));
    std::vector<TCanvas *> keptCanvases;
    keptCanvases.reserve(refs.size());

    for (const auto &ref : refs) {
        TCanvas *cIn = LoadCanvasByCentrality(fRun3.get(), ref.centTag, "c_final_spectrum");
        if (!cIn) {
            std::cerr << "[SpectrumVsRun2Simple] Cannot find c_final_spectrum under " << ref.centTag << std::endl;
            continue;
        }

        const std::string outCanvasName = std::string("c_final_spectrum_run2_") + ref.centTag;
        auto *cOut = dynamic_cast<TCanvas *>(cIn->Clone(outCanvasName.c_str()));
        if (!cOut) {
            std::cerr << "[SpectrumVsRun2Simple] Failed to clone canvas for " << ref.centTag << std::endl;
            continue;
        }

        TPad *pad = dynamic_cast<TPad *>(cOut);
        if (!pad) {
            std::cerr << "[SpectrumVsRun2Simple] Invalid pad for " << ref.centTag << std::endl;
            delete cOut;
            continue;
        }

        std::unique_ptr<TGraphAsymmErrors> gRun2Stat;
        std::unique_ptr<TF1> fRun2BW;

        if (ref.hasRun2) {
            const std::string run2Path = std::string(run2BaseDir) + "/" + ref.run2File;
            std::unique_ptr<TFile> fRun2(TFile::Open(run2Path.c_str(), "READ"));
            if (!fRun2 || fRun2->IsZombie()) {
                std::cerr << "[SpectrumVsRun2Simple] Cannot open Run2 file: " << run2Path << std::endl;
            } else {
                gRun2Stat.reset(LoadGraphAsymm(fRun2.get(), run2GraphName, run2GraphSubdir));
                if (!gRun2Stat) {
                    std::cerr << "[SpectrumVsRun2Simple] Missing Run2 graph in " << run2Path
                              << ": " << run2GraphName << std::endl;
                }
            }

            if (fBW && !fBW->IsZombie() && std::string(ref.bwName).size() > 0) {
                fRun2BW.reset(LoadTF1(fBW.get(), ref.bwName));
            }

            if (gRun2Stat) {
                gRun2Stat->SetName((std::string("g_run2_stat_") + ref.centTag).c_str());
                gRun2Stat->SetMarkerStyle(33);
                gRun2Stat->SetMarkerSize(1.5);
                gRun2Stat->SetMarkerColor(kAzure + 2);
                gRun2Stat->SetLineColor(kAzure + 2);
                gRun2Stat->SetLineWidth(3);
                gRun2Stat->SetFillStyle(0);

                // Remove horizontal error bars for Run2 points.
                for (int ip = 0; ip < gRun2Stat->GetN(); ++ip) {
                    gRun2Stat->SetPointEXlow(ip, 0.0);
                    gRun2Stat->SetPointEXhigh(ip, 0.0);
                }
            }

            if (fRun2BW) {
                fRun2BW->SetName((std::string("f_run2_bw_") + ref.centTag).c_str());
                fRun2BW->SetLineColor(kAzure + 2);
                fRun2BW->SetLineStyle(7);
                fRun2BW->SetLineWidth(3);

                // Force Run2 BW curve to start from x = 0.
                const double bwMin = 0.0;
                double bwMax = run2BWXMax;
                if (!(bwMax > bwMin)) {
                    double autoMin = 0.0;
                    double autoMax = 0.0;
                    if (DeterminePadXRange(pad, autoMin, autoMax) && autoMax > autoMin) {
                        bwMax = autoMax;
                    } else {
                        bwMax = 8.0;
                    }
                }
                fRun2BW->SetRange(bwMin, bwMax);
            }

            if (gRun2Stat) {
                pad->cd();
                TObject *drawnRun2 = gRun2Stat->DrawClone("P SAME");
                TObject *drawnBW = nullptr;
                if (fRun2BW) drawnBW = fRun2BW->DrawClone("SAME");

                TObject *run3Obj = FindPrimaryRun3Drawable(pad);
                TLegend *leg = FindLegend(pad);
                if (!leg) {
                    leg = new TLegend(0.56, 0.62, 0.90, 0.88);
                    leg->SetName((std::string("leg_run3_run2_") + ref.centTag).c_str());
                    leg->SetFillStyle(0);
                    leg->SetBorderSize(0);
                    leg->SetTextFont(42);
                    leg->SetTextSize(0.032);
                    if (run3Obj) leg->AddEntry(run3Obj, "Run 3", "lep");
                }

                leg->AddEntry(drawnRun2 ? drawnRun2 : static_cast<TObject *>(gRun2Stat.get()), "Run 2 (5.02 TeV)", "lep");
                if (fRun2BW) leg->AddEntry(drawnBW ? drawnBW : static_cast<TObject *>(fRun2BW.get()), "Run 2 Blast-Wave fit", "l");
                leg->Draw();
            }
        }

        cOut->Modified();
        cOut->Update();

        TDirectory *outDirObj = fOut.mkdir(ref.centTag);
        if (!outDirObj) outDirObj = fOut.GetDirectory(ref.centTag);
        if (outDirObj) {
            outDirObj->cd();
            cOut->Write("c_final_spectrum", TObject::kOverwrite);
            if (gRun2Stat) gRun2Stat->Write("g_run2_stat", TObject::kOverwrite);
            if (fRun2BW) fRun2BW->Write("f_run2_bw", TObject::kOverwrite);
        }

        keptCanvases.push_back(cOut);
    }

    fOut.Close();

    std::cout << "[SpectrumVsRun2Simple] Saved comparison ROOT file: " << outPath << std::endl;
}

void ExportSpectrumVsRun2SimpleCanvases(const char *rootPath,
                                        const char *periodTag,
                                        const char *outputDir) {
    const std::array<const char *, 4> cents = {{"cen_0_10", "cen_10_30", "cen_30_50", "cen_50_80"}};

    std::unique_ptr<TFile> f(TFile::Open(rootPath, "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[ExportSpectrumVsRun2SimpleCanvases] Cannot open: " << rootPath << std::endl;
        return;
    }

    gSystem->mkdir(outputDir, true);
    for (const auto *cent : cents) {
        auto *c = dynamic_cast<TCanvas *>(f->Get((std::string(cent) + "/c_final_spectrum").c_str()));
        if (!c) {
            std::cerr << "[ExportSpectrumVsRun2SimpleCanvases] Missing canvas for " << cent
                      << " in " << rootPath << std::endl;
            continue;
        }
        const std::string outPdf = std::string(outputDir) + "/Run3_vs_Run2_" + periodTag + "_" + cent + ".pdf";
        c->SaveAs(outPdf.c_str());
        std::cout << "[Info] Saved: " << outPdf << std::endl;
    }
}

void SpectrumVsRun2SimplePeriodCompare(
    const char *run2BaseDir = "/Users/zhengqingwang/alice/data/h3l_spec_run2",
    const char *run2BWRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root",
    const char *outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/SpectrumvsRun2") {

    struct PeriodSpec {
        const char *tag;
        const char *path;
        const char *rootName;
    };

    const std::array<PeriodSpec, 3> periods = {{
        {"LHC23_PbPb_pass5",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
         "Spectrum_vs_run2_simple_LHC23_PbPb_pass5.root"},
        {"LHC24ar_pass3",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC24ar_pass3_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
         "Spectrum_vs_run2_simple_LHC24ar_pass3.root"},
        {"LHC25_PbPb_pass1",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC25_PbPb_pass1_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
         "Spectrum_vs_run2_simple_LHC25_PbPb_pass1.root"}
    }};

    for (const auto &period : periods) {
        SpectrumVsRun2Simple(period.path,
                             run2BaseDir,
                             "Graph1D_y1",
                             "",
                             run2BWRoot,
                             "BlastWave_H3L_10_30",
                             -1.0,
                             -1.0,
                             outputDir,
                             period.rootName);

        const std::string rootPath = std::string(outputDir) + "/" + period.rootName;
        ExportSpectrumVsRun2SimpleCanvases(rootPath.c_str(), period.tag, outputDir);
    }
}
