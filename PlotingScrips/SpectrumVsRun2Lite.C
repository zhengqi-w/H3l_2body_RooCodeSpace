// Lightweight ROOT macro to overlay Run3 spectra with Run2 references (no ratio pads)
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>
#include <TMath.h>
#include <TFile.h>
#include <TF1.h>
#include <TLatex.h>
#include <TSystem.h>
#include <TString.h>

#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <algorithm>

struct LiteSpec {
    std::string label;
    std::vector<std::string> run3File;
    std::string run2File;   // leave empty to skip Run2 (e.g. 50-80)
    std::string run2Graph;
    std::string run3Hist;
    std::string run3Subdir;
    std::string run2Subdir;
    std::string bwName;     // TF1 name in BW file
    std::vector<std::string> legendNames; 
};

const std::string kBWFitPathLite = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root";

static TH1* GetHistSimple(const std::string& path, const std::string& histName, const std::string& subdir = "") {
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(histName.c_str());
    if (!obj && !subdir.empty()) {
        TDirectory* dir = file;
        std::string token;
        std::istringstream ss(subdir);
        while (std::getline(ss, token, '/')) {
            if (!dir) break;
            dir = dynamic_cast<TDirectory*>(dir->Get(token.c_str()));
        }
        if (dir) obj = dir->Get(histName.c_str());
    }
    if (!obj && file->Get("std")) obj = file->Get(Form("std/%s", histName.c_str()));
    if (!obj) {
        std::cerr << "Histogram not found: " << histName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    TH1* hist = dynamic_cast<TH1*>(obj);
    if (!hist) {
        std::cerr << histName << " is not a TH1" << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    TH1* clone = dynamic_cast<TH1*>(hist->Clone(Form("%s_clone", histName.c_str())));
    if (clone) clone->SetDirectory(nullptr);
    file->Close();
    delete file;
    return clone;
}

static TGraphAsymmErrors* GetGraphSimple(const std::string& path, const std::string& graphName, const std::string& subdir = "") {
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(graphName.c_str());
    if (!obj && !subdir.empty()) {
        TDirectory* dir = file;
        std::string token;
        std::istringstream ss(subdir);
        while (std::getline(ss, token, '/')) {
            if (!dir) break;
            dir = dynamic_cast<TDirectory*>(dir->Get(token.c_str()));
        }
        if (dir) obj = dir->Get(graphName.c_str());
    }
    if (!obj) {
        std::cerr << "Graph not found: " << graphName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    auto* g = dynamic_cast<TGraphAsymmErrors*>(obj);
    auto* clone = g ? dynamic_cast<TGraphAsymmErrors*>(g->Clone(Form("%s_clone", graphName.c_str()))) : nullptr;
    file->Close();
    delete file;
    return clone;
}

static TF1* GetTF1Simple(const std::string& path, const std::string& funcName) {
    if (funcName.empty()) return nullptr;
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(funcName.c_str());
    if (!obj) {
        std::cerr << "TF1 not found: " << funcName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    auto* f = dynamic_cast<TF1*>(obj);
    auto* clone = f ? dynamic_cast<TF1*>(f->Clone(Form("%s_clone", funcName.c_str()))) : nullptr;
    file->Close();
    delete file;
    return clone;
}

static TGraphErrors* HistToGraph(const TH1* hist, const std::string& name) {
    if (!hist) return nullptr;
    auto* g = new TGraphErrors(hist->GetNbinsX());
    g->SetName(name.c_str());
    for (int i = 1; i <= hist->GetNbinsX(); ++i) {
        g->SetPoint(i - 1, hist->GetBinCenter(i), hist->GetBinContent(i));
        g->SetPointError(i - 1, 0.0, hist->GetBinError(i));
    }
    return g;
}

static void ShiftGraphX(TGraph* g, double dx) {
    if (!g) return;
    double x, y;
    for (int i = 0; i < g->GetN(); ++i) {
        g->GetPoint(i, x, y);
        g->SetPoint(i, x + dx, y);
    }
}

static void StyleGraph(TGraphErrors* g, Color_t color, int marker) {
    if (!g) return;
    g->SetMarkerColor(color);
    g->SetLineColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.0);
    g->SetLineWidth(2);
}

static void StyleGraph(TGraphAsymmErrors* g, Color_t color, int marker) {
    if (!g) return;
    g->SetMarkerColor(color);
    g->SetLineColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.0);
    g->SetLineWidth(2);
}

static void UpdateRange(TGraph* g, double& xmin, double& xmax, double& ymin, double& ymax) {
    if (!g) return;
    double x, y;
    for (int i = 0; i < g->GetN(); ++i) {
        g->GetPoint(i, x, y);
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
        double ylow = y, yhigh = y;
        if (auto* asym = dynamic_cast<TGraphAsymmErrors*>(g)) {
            ylow = y - asym->GetErrorYlow(i);
            yhigh = y + asym->GetErrorYhigh(i);
        } else if (auto* ge = dynamic_cast<TGraphErrors*>(g)) {
            ylow = y - ge->GetErrorY(i);
            yhigh = y + ge->GetErrorY(i);
        }
        if (yhigh > 0) ymax = std::max(ymax, yhigh);
        if (ylow > 0) ymin = std::min(ymin, ylow);
    }
}

static std::string FormatCentralityRange(const std::string& label) {
    if (label.empty()) return "";
    auto range = label;
    std::replace(range.begin(), range.end(), '_', '-');
    return range;
}

// Build the centrality-aware directory (new layout: spectrum.root/cen_x_y/<subdir>/hist)
static std::string BuildCentralitySubdir(const std::string& centLabel, const std::string& baseSubdir) {
    if (centLabel.empty()) return baseSubdir;
    std::string dir = "cen_" + centLabel;
    if (!baseSubdir.empty()) dir += "/" + baseSubdir;
    return dir;
}

// Fetch histogram trying the new centrality layout first, then falling back to the old one
static TH1* GetHistWithCentrality(const std::string& path, const std::string& histName, const std::string& centLabel, const std::string& baseSubdir) {
    const std::string centDir = BuildCentralitySubdir(centLabel, baseSubdir);
    if (auto* h = GetHistSimple(path, histName, centDir)) return h;
    if (auto* h = GetHistSimple(path, histName, baseSubdir)) return h;
    return GetHistSimple(path, histName, "");
}

void SpectrumVsRun2Lite() {
    gStyle->SetOptStat(0);
    std::string outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/SpectrumvsRun2";
    gSystem->mkdir(outputDir.c_str(), true);
    const std::string mcTag = "";
    const std::string saveSuffixName = "NCrossedRowsCompare";
    std::vector<LiteSpec> specs = {
        {
          "0_10",
          {
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
            "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum_no_itscut/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/bdt_spectrum/both/spectrum.root"
          },
          "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_0_10.root",
          "Graph1D_y1",
          "h_corrected_counts",
          "std",
          "(Anti)hypertriton spectrum in 0-10% V0M centrality class",
          "BlastWave_H3L_0_10",
          {
            //"23_pass4_CustomV0s_HadronPID",
            "23_pass5_CustomV0s_HadronPID",
            //"23_pass5_CustomV0s_HadronPID_TPCNcls",
            //"23_pass5_V0s_HadronPID_TPCNCls",
          } 
        },
        {
          "10_30",
          {
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
            "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum_no_itscut/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/bdt_spectrum/both/spectrum.root"
          },         
          "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_10_30.root",
          "Graph1D_y1",
          "h_corrected_counts",
          "std",
          "(Anti)hypertriton spectrum in 10-30% V0M centrality class",
          "BlastWave_H3L_10_30",
          {
            //"23_pass4_CustomV0s_HadronPID",
            "23_pass5_CustomV0s_HadronPID",
            //"23_pass5_CustomV0s_HadronPID_TPCNcls",
            //"23_pass5_V0s_HadronPID_TPCNcls"
          }
        },
        {
          "30_50",
          {
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
            "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum_no_itscut/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/bdt_spectrum/both/spectrum.root"
         },
          "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_30_50.root",
          "Graph1D_y1",
          "h_corrected_counts",
          "std",
          "(Anti)hypertriton spectrum in 30-50% V0M centrality class",
          "BlastWave_H3L_30_50",
          {
            //"23_pass4_CustomV0s_HadronPID",
            "23_pass5_CustomV0s_HadronPID",
            //"23_pass5_CustomV0s_HadronPID_TPCNcls",
            //"23_pass5_V0s_HadronPID_TPCNcls"
          }
        },
        {
          "50_80",
          {
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum/both/spectrum.root",
            "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/bdt_spectrum_no_itscut/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/bdt_spectrum/both/spectrum.root",
            //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/bdt_spectrum/both/spectrum.root"
          },
          "", // skip Run2 spectrum here
          "",
          "h_corrected_counts",
          "std",
          "(Anti)hypertriton spectrum in 50-80% V0M centrality class",
          "",
          {
            //"23_pass4_CustomV0s_HadronPID",
            "23_pass5_CustomV0s_HadronPID",
            //"23_pass5_CustomV0s_HadronPID_TPCNcls",
            //"23_pass5_V0s_HadronPID_TPCNcls"
          }
        }
    };

    // Color/marker palettes for multiple Run 3 inputs
    const std::vector<Color_t> run3Colors = {
        static_cast<Color_t>(kMagenta + 2),
        static_cast<Color_t>(kOrange + 7),
        static_cast<Color_t>(kTeal + 4),
        static_cast<Color_t>(kViolet + 7),
        static_cast<Color_t>(kGreen + 2),
        static_cast<Color_t>(kPink + 6)
    };
    const std::vector<int> run3Markers = {20, 21, 22, 23, 33, 34};

    for (const auto& spec : specs) {
        // Load Run 2 references once per centrality
        std::unique_ptr<TGraphAsymmErrors> g2Base;
        std::unique_ptr<TF1> f2Base;
        if (!spec.run2File.empty()) {
            g2Base.reset(GetGraphSimple(spec.run2File, spec.run2Graph, spec.run2Subdir));
            f2Base.reset(GetTF1Simple(kBWFitPathLite, spec.bwName));
            StyleGraph(g2Base.get(), kAzure + 2, 21);
            if (f2Base) {
                f2Base->SetLineColor(kAzure + 2);
                f2Base->SetLineStyle(2);
                f2Base->SetLineWidth(2);
                f2Base->SetRange(2.0, 8.0);
            }
        }

        // 1) Per-run3-file comparison vs Run2
        for (size_t idx = 0; idx < spec.run3File.size(); ++idx) {
            std::unique_ptr<TH1> h3(GetHistWithCentrality(spec.run3File[idx], spec.run3Hist, spec.label, spec.run3Subdir));
            if (!h3) continue;
            auto g3 = std::unique_ptr<TGraphErrors>(HistToGraph(h3.get(), Form("run3_%s_%zu", spec.label.c_str(), idx)));
            double shift = -0.05;  // keep a small offset to avoid overlap with Run2
            ShiftGraphX(g3.get(), shift);
            Color_t cRun3 = run3Colors[idx % run3Colors.size()];
            int mRun3 = run3Markers[idx % run3Markers.size()];
            StyleGraph(g3.get(), cRun3, mRun3);

            std::unique_ptr<TGraphAsymmErrors> g2;
            std::unique_ptr<TF1> f2;
            if (g2Base) {
                g2.reset(dynamic_cast<TGraphAsymmErrors*>(g2Base->Clone(Form("run2_%s_%zu", spec.label.c_str(), idx))));
                if (g2) StyleGraph(g2.get(), kAzure + 2, 21);
            }
            if (f2Base) {
                f2.reset(dynamic_cast<TF1*>(f2Base->Clone(Form("bw_%s_%zu", spec.label.c_str(), idx))));
                if (f2) {
                    f2->SetLineColor(kAzure + 2);
                    f2->SetLineStyle(2);
                    f2->SetLineWidth(2);
                    f2->SetRange(2.0, 8.0);
                }
            }

            double xmin = std::numeric_limits<double>::infinity();
            double xmax = 0.0;
            double ymin = std::numeric_limits<double>::infinity();
            double ymax = 0.0;
            UpdateRange(g3.get(), xmin, xmax, ymin, ymax);
            UpdateRange(g2.get(), xmin, xmax, ymin, ymax);
            if (!std::isfinite(ymin) || ymin <= 0) ymin = 1e-8;
            if (ymax <= 0) ymax = 1.0;
            double yMinDraw = std::max(1e-8, ymin * 0.4);
            double yMaxDraw = ymax * 1.2;

            TCanvas c(Form("c_lite_%s_%zu", spec.label.c_str(), idx), "Run3 vs Run2", 900, 700);
            c.SetLeftMargin(0.12);
            c.SetRightMargin(0.05);
            c.SetTopMargin(0.08);
            c.SetBottomMargin(0.12);
            c.SetLogy();
            c.SetGrid();

            auto* frame = gPad->DrawFrame(2.0, yMinDraw, 8.0, yMaxDraw);
            frame->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
            frame->GetYaxis()->SetTitle("#frac{1}{N_{ev}} dN/dy dp_{T}");
            frame->GetXaxis()->SetTitleFont(42);
            frame->GetYaxis()->SetTitleFont(42);
            frame->GetXaxis()->SetLabelFont(42);
            frame->GetYaxis()->SetLabelFont(42);
            frame->GetXaxis()->SetTitleSize(0.045);
            frame->GetYaxis()->SetTitleSize(0.045);
            frame->GetXaxis()->SetLabelSize(0.04);
            frame->GetYaxis()->SetLabelSize(0.04);
            frame->GetXaxis()->SetTitleOffset(1.1);
            frame->GetYaxis()->SetTitleOffset(1.2);
            std::string centLabel = FormatCentralityRange(spec.label);
            frame->SetTitle(Form("(Anti)^{3}_{#Lambda}H Spectrum Run 3 vs Run 2 Centrality: %s%%", centLabel.c_str()));
            frame->SetTitleFont(62);
            frame->SetTitleSize(0.052);
            g3->GetXaxis()->SetLimits(2.0, 8.0);
            g3->Draw("P SAME");
            if (g2) g2->Draw("P SAME");
            if (f2) f2->Draw("SAME");

            const bool isPeripheral = (spec.label == "50_80");
            double legX1 = isPeripheral ? 0.60 : 0.16;
            double legY1 = isPeripheral ? 0.70 : 0.25;
            double legX2 = isPeripheral ? 0.92 : 0.46;
            double legY2 = isPeripheral ? 0.85 : 0.55;

            auto legend = std::make_unique<TLegend>(legX1, legY1, legX2, legY2);
            legend->SetFillStyle(0);
            legend->SetBorderSize(0);
            legend->SetTextFont(42);
            std::string legEntry = spec.legendNames.size() > idx ? spec.legendNames[idx] : Form("Run3_%zu", idx);
            legEntry += mcTag;
            legend->AddEntry(g3.get(), legEntry.c_str(), "lep");
            if (g2) legend->AddEntry(g2.get(), "Run 2 (5.02 TeV)", "lep");
            if (f2) legend->AddEntry(f2.get(), "Run 2 Blast-Wave fit", "l");
            legend->Draw();

            TLatex header;
            header.SetTextFont(42);
            header.SetTextSize(0.042);
            header.SetNDC();
            double headerX = isPeripheral ? 0.60 : legX1;
            double headerY = isPeripheral ? 0.87 : std::min(legY2 + 0.025, 0.95);
            header.DrawLatex(headerX, headerY, "LHC23_PbPb_pass5");
            std::string subTitle = spec.legendNames.size() > idx ? spec.legendNames[idx] : Form("Run3_%zu", idx);
            std::string out = outputDir + "/Spectrum_vs_run2_lite_" + spec.label + "_" + saveSuffixName + "_" + subTitle + ".pdf";
            c.SaveAs(out.c_str());
        }

        // 2) Combined plot for all Run3 files (only if more than one)
        if (spec.run3File.size() < 2) continue;

        std::vector<std::unique_ptr<TGraphErrors>> g3All;
        g3All.reserve(spec.run3File.size());

        double xmin = std::numeric_limits<double>::infinity();
        double xmax = 0.0;
        double ymin = std::numeric_limits<double>::infinity();
        double ymax = 0.0;

        for (size_t idx = 0; idx < spec.run3File.size(); ++idx) {
            std::unique_ptr<TH1> h3(GetHistWithCentrality(spec.run3File[idx], spec.run3Hist, spec.label, spec.run3Subdir));
            if (!h3) continue;
            auto g3 = std::unique_ptr<TGraphErrors>(HistToGraph(h3.get(), Form("run3_all_%s_%zu", spec.label.c_str(), idx)));
            double dx = 0.02 * (static_cast<double>(idx) - (static_cast<double>(spec.run3File.size()) - 1.0) / 2.0);
            ShiftGraphX(g3.get(), dx);
            Color_t cRun3 = run3Colors[idx % run3Colors.size()];
            int mRun3 = run3Markers[idx % run3Markers.size()];
            StyleGraph(g3.get(), cRun3, mRun3);
            UpdateRange(g3.get(), xmin, xmax, ymin, ymax);
            g3All.push_back(std::move(g3));
        }

        std::unique_ptr<TGraphAsymmErrors> g2All;
        std::unique_ptr<TF1> f2All;
        if (g2Base) {
            g2All.reset(dynamic_cast<TGraphAsymmErrors*>(g2Base->Clone(Form("run2_all_%s", spec.label.c_str()))));
            if (g2All) {
                StyleGraph(g2All.get(), kAzure + 2, 21);
                UpdateRange(g2All.get(), xmin, xmax, ymin, ymax);
            }
        }
        if (f2Base) {
            f2All.reset(dynamic_cast<TF1*>(f2Base->Clone(Form("bw_all_%s", spec.label.c_str()))));
            if (f2All) {
                f2All->SetLineColor(kAzure + 2);
                f2All->SetLineStyle(2);
                f2All->SetLineWidth(2);
                f2All->SetRange(2.0, 8.0);
            }
        }

        if (g3All.empty()) continue;

        if (!std::isfinite(ymin) || ymin <= 0) ymin = 1e-8;
        if (ymax <= 0) ymax = 1.0;
        double yMinDraw = std::max(1e-8, ymin * 0.4);
        double yMaxDraw = ymax * 1.2;

        TCanvas cAll(Form("c_lite_%s_all", spec.label.c_str()), "Run3 vs Run2 Combined", 900, 700);
        cAll.SetLeftMargin(0.12);
        cAll.SetRightMargin(0.05);
        cAll.SetTopMargin(0.08);
        cAll.SetBottomMargin(0.12);
        cAll.SetLogy();
        cAll.SetGrid();

        auto* frameAll = gPad->DrawFrame(2.0, yMinDraw, 8.0, yMaxDraw);
        frameAll->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
        frameAll->GetYaxis()->SetTitle("#frac{1}{N_{ev}} dN/dy dp_{T}");
        frameAll->GetXaxis()->SetTitleFont(42);
        frameAll->GetYaxis()->SetTitleFont(42);
        frameAll->GetXaxis()->SetLabelFont(42);
        frameAll->GetYaxis()->SetLabelFont(42);
        frameAll->GetXaxis()->SetTitleSize(0.045);
        frameAll->GetYaxis()->SetTitleSize(0.045);
        frameAll->GetXaxis()->SetLabelSize(0.04);
        frameAll->GetYaxis()->SetLabelSize(0.04);
        frameAll->GetXaxis()->SetTitleOffset(1.1);
        frameAll->GetYaxis()->SetTitleOffset(1.2);
        std::string centLabelAll = FormatCentralityRange(spec.label);
        frameAll->SetTitle(Form("(Anti)^{3}_{#Lambda}H Spectrum Run 3 vs Run 2 Centrality: %s%%", centLabelAll.c_str()));
        frameAll->SetTitleFont(62);
        frameAll->SetTitleSize(0.052);

        for (auto& g3 : g3All) {
            g3->GetXaxis()->SetLimits(2.0, 8.0);
            g3->Draw("P SAME");
        }
        if (g2All) g2All->Draw("P SAME");
        if (f2All) f2All->Draw("SAME");

        const bool isPeripheral = (spec.label == "50_80");
        double legX1 = isPeripheral ? 0.45 : 0.13;
        double legY1 = isPeripheral ? 0.62 : 0.12;
        double legX2 = isPeripheral ? 0.78 : 0.43;
        double legY2 = isPeripheral ? 0.87 : 0.47;

        auto legendAll = std::make_unique<TLegend>(legX1, legY1, legX2, legY2);
        legendAll->SetFillStyle(0);
        legendAll->SetBorderSize(0);
        legendAll->SetTextFont(62);
        legendAll->SetTextSize(0.025);
        for (size_t idx = 0; idx < g3All.size(); ++idx) {
            std::string legEntryAll = spec.legendNames.size() > idx ? spec.legendNames[idx] : Form("Run3_%zu", idx);
            legEntryAll += mcTag;
            legendAll->AddEntry(g3All[idx].get(), legEntryAll.c_str(), "lep");
        }
        if (g2All) legendAll->AddEntry(g2All.get(), "Run 2 (5.02 TeV)", "lep");
        if (f2All) legendAll->AddEntry(f2All.get(), "Run 2 Blast-Wave fit", "l");
        legendAll->Draw();

        TLatex headerAll;
        headerAll.SetTextFont(42);
        headerAll.SetTextSize(0.042);
        headerAll.SetNDC();
        double headerXAll = isPeripheral ? 0.45 : legX1;
        double headerYAll = isPeripheral ? 0.87 : std::min(legY2 + 0.025, 0.95);
        headerAll.DrawLatex(headerXAll, headerYAll, "ALICE Run 3 Pb--Pb @ #sqrt{#it{s}_{NN}} = 5.36 TeV");

        std::string outAll = outputDir + "/Spectrum_vs_run2_lite_" + spec.label + "_" + saveSuffixName + ".pdf";
        cAll.SaveAs(outAll.c_str());
    }
}
