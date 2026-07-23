#include <TBox.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Point {
    std::string label;
    std::string group;
    double x{0.0};
    double ex{0.0};
    double y{0.0};
    double stat{0.0};
    double syst{0.0};
    int color{kBlack};
    int marker{20};
    double markerSize{1.15};
};

struct YieldRow {
    double cenMin{0.0};
    double cenMax{0.0};
    double y{0.0};
    double stat{0.0};
    double syst{0.0};
};

std::vector<std::string> SplitCsvLine(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

std::map<std::string, int> HeaderIndex(const std::vector<std::string> &header) {
    std::map<std::string, int> out;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) out[header[i]] = i;
    return out;
}

double GetField(const std::vector<std::string> &row, const std::map<std::string, int> &idx, const std::string &key, double fallback = 0.0) {
    auto it = idx.find(key);
    if (it == idx.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) return fallback;
    try {
        return std::stod(row[it->second]);
    } catch (...) {
        return fallback;
    }
}

std::vector<YieldRow> ReadYieldSummary(const std::string &path) {
    std::vector<YieldRow> rows;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "[Warn] Cannot open " << path << std::endl;
        return rows;
    }
    std::string line;
    if (!std::getline(in, line)) return rows;
    const auto idx = HeaderIndex(SplitCsvLine(line));
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cols = SplitCsvLine(line);
        YieldRow r;
        r.cenMin = GetField(cols, idx, "centrality_min");
        r.cenMax = GetField(cols, idx, "centrality_max");
        r.y = GetField(cols, idx, "integrated_yield");
        r.stat = GetField(cols, idx, "stat_err");
        r.syst = GetField(cols, idx, "syst_total_abs");
        rows.push_back(r);
    }
    return rows;
}

std::string CentKey(double min, double max) {
    return Form("%.0f_%.0f", min, max);
}

int RainbowColor(size_t i, size_t n) {
    const double frac = n <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    return TColor::GetColorPalette(static_cast<int>((1.0 - frac) * (gStyle->GetNumberOfColors() - 1)));
}

void StyleGraph(TGraphAsymmErrors *g, int color, int marker, double markerSize = 1.15) {
    if (!g) return;
    g->SetLineColor(color);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(markerSize);
    g->SetLineWidth(2);
}

std::unique_ptr<TGraphAsymmErrors> MakeGraph(const std::vector<Point> &pts,
                                             const std::string &name,
                                             bool useSyst,
                                             int color,
                                             int marker,
                                             double markerSize = 1.15) {
    auto g = std::make_unique<TGraphAsymmErrors>(static_cast<int>(pts.size()));
    g->SetName(name.c_str());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const auto &p = pts[i];
        g->SetPoint(i, p.x, p.y);
        const double ey = useSyst ? p.syst : p.stat;
        const double ex = useSyst ? p.ex : 0.0;
        g->SetPointError(i, ex, ex, ey, ey);
    }
    StyleGraph(g.get(), color, marker, markerSize);
    return g;
}

void DrawSystBoxes(const std::vector<Point> &pts, double xScale = 1.0) {
    for (const auto &p : pts) {
        if (p.syst <= 0.0) continue;
        const double halfWidth = std::max(p.ex, p.x * xScale);
        auto *box = new TBox(p.x - halfWidth, std::max(1e-12, p.y - p.syst), p.x + halfWidth, p.y + p.syst);
        box->SetFillStyle(0);
        box->SetLineColor(p.color);
        box->SetLineWidth(2);
        box->Draw("SAME");
    }
}

std::vector<Point> ReferencePoints() {
    return {
        {"pp MB", "pp", 6.9, 0.9, 2.1e-8, 0.6e-8, 0.4e-8, kCyan + 2, 20, 1.15},
        {"pp HM", "pp", 30.8, 3.7, 2.4e-7, 0.5e-7, 0.3e-7, kCyan + 2, 22, 1.15},
        {"p--Pb", "pPb", 29.4, 3.0, 6.83e-7, 1.8e-7, 1.2e-7, kAzure - 4, 23, 1.25},
        {"O--O", "OO", 43.9, 2.5, 7.39e-7, 0.59e-7, 1.69e-7, kBlue + 2, 33, 1.45},
        {"Pb--Pb 0--10%", "PbPbRun2", 1764.0, 51.6, 4.83e-5, 0.23e-5, 0.57e-5, kBlue + 1, 21, 1.15},
        {"Pb--Pb 10--30%", "PbPbRun2", 983.0, 36.9, 2.62e-5, 0.25e-5, 0.40e-5, kBlue + 1, 21, 1.15},
        {"Pb--Pb 30--50%", "PbPbRun2", 415.0, 19.2, 1.27e-5, 0.10e-5, 0.14e-5, kBlue + 1, 21, 1.15},
    };
}

std::vector<Point> LoadRun3FromRoot(const std::string &bothRoot) {
    std::vector<Point> pts;
    std::unique_ptr<TFile> file(TFile::Open(bothRoot.c_str(), "READ"));
    if (!file || file->IsZombie()) return pts;
    auto *gStat = dynamic_cast<TGraphAsymmErrors *>(file->Get("summary/integral_yield/g_integral_yield_vs_multiplicity_stat"));
    auto *gSys = dynamic_cast<TGraphAsymmErrors *>(file->Get("summary/integral_yield/g_integral_yield_vs_multiplicity_sys"));
    if (!gStat) return pts;
    for (int i = 0; i < gStat->GetN(); ++i) {
        double x = 0.0, y = 0.0;
        gStat->GetPoint(i, x, y);
        Point p;
        p.label = Form("Run 3 point %d", i);
        p.group = "PbPbRun3";
        p.x = x;
        p.ex = std::max(gStat->GetErrorXlow(i), gStat->GetErrorXhigh(i));
        p.y = y;
        p.stat = std::max(gStat->GetErrorYlow(i), gStat->GetErrorYhigh(i));
        if (gSys && i < gSys->GetN()) {
            p.ex = std::max(gSys->GetErrorXlow(i), gSys->GetErrorXhigh(i));
            p.syst = std::max(gSys->GetErrorYlow(i), gSys->GetErrorYhigh(i));
        }
        pts.push_back(p);
    }
    return pts;
}

std::vector<Point> LoadRun3FromMatterAntiAverage(const std::string &baseDir) {
    const auto matter = ReadYieldSummary(baseDir + "/matter/integrated_yield_summary.csv");
    const auto antimatter = ReadYieldSummary(baseDir + "/antimatter/integrated_yield_summary.csv");
    std::map<std::string, YieldRow> antiByCent;
    for (const auto &r : antimatter) antiByCent[CentKey(r.cenMin, r.cenMax)] = r;

    const std::vector<double> mult = {2047, 1668, 1253, 848, 559, 351, 205, 110, 38.1};
    const std::vector<double> multErr = {54, 42, 33, 25, 19, 14, 11, 8, 2.9};

    std::vector<Point> pts;
    for (size_t i = 0; i < matter.size() && i < mult.size(); ++i) {
        const auto it = antiByCent.find(CentKey(matter[i].cenMin, matter[i].cenMax));
        if (it == antiByCent.end()) continue;
        const auto &anti = it->second;
        Point p;
        p.label = Form("%.0f--%.0f%%", matter[i].cenMin, matter[i].cenMax);
        p.group = "PbPbRun3";
        p.x = mult[i];
        p.ex = multErr[i];
        p.y = 0.5 * (matter[i].y + anti.y);
        p.stat = 0.5 * std::hypot(matter[i].stat, anti.stat);
        p.syst = 0.5 * std::hypot(matter[i].syst, anti.syst);
        pts.push_back(p);
    }
    return pts;
}

std::vector<Point> LoadRun3Points(const std::string &bothRoot, const std::string &baseDir) {
    auto pts = LoadRun3FromRoot(bothRoot);
    if (!pts.empty()) {
        std::cout << "[Info] Loaded Run 3 Pb--Pb points from " << bothRoot << std::endl;
    } else {
        std::cout << "[Warn] Cannot read Run 3 integrated-yield graph from " << bothRoot
                  << "; using (matter + antimatter)/2 integrated_yield_summary.csv fallback." << std::endl;
        pts = LoadRun3FromMatterAntiAverage(baseDir);
    }
    std::sort(pts.begin(), pts.end(), [](const Point &a, const Point &b) { return a.x < b.x; });
    for (size_t i = 0; i < pts.size(); ++i) {
        pts[i].color = kRed + 1;
        pts[i].marker = 29;
        pts[i].markerSize = 1.35;
    }
    return pts;
}

std::unique_ptr<TGraphErrors> MakeFitGraph(const std::vector<Point> &pts) {
    auto g = std::make_unique<TGraphErrors>(static_cast<int>(pts.size()));
    g->SetName("g_power_fit_points");
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        g->SetPoint(i, pts[i].x, pts[i].y);
        g->SetPointError(i, 0.0, 0.0);
    }
    return g;
}

void DrawHeader(double x, double y, bool compact = false) {
    TLatex t;
    t.SetNDC();
    t.SetTextFont(42);
    t.SetTextSize(0.044);
    t.DrawLatex(x, y, "ALICE Work In Progress");
    t.SetTextSize(0.034);
    t.DrawLatex(x, y - 0.055, "{}^{3}_{#Lambda}H #rightarrow {}^{3}He + #pi");
    if (!compact) t.DrawLatex(x, y - 0.103, "Pb--Pb #sqrt{#it{s}_{NN}} = 5.36 TeV, Run 3 merged");
}

TLegend *MakeLegend(double x1, double y1, double x2, double y2) {
    auto *leg = new TLegend(x1, y1, x2, y2);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.029);
    return leg;
}

void SaveCanvas(TCanvas &c, const std::string &outDir, const std::string &name) {
    c.SaveAs((outDir + "/" + name + ".pdf").c_str());
}

} // namespace

void DrawHypDndchPbPb536(
    const char *bothRoot = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root",
    const char *baseDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/HypDndchPbPb536")
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPalette(kRainBow);
    gStyle->SetEndErrorSize(0);
    gSystem->mkdir(outDir, true);

    auto refs = ReferencePoints();
    auto run3 = LoadRun3Points(bothRoot, baseDir);
    if (run3.empty()) {
        std::cerr << "[Error] No Run 3 points were loaded." << std::endl;
        return;
    }

    std::vector<Point> pp, pPb, oo, pbpbRun2;
    for (const auto &p : refs) {
        if (p.group == "pp") pp.push_back(p);
        else if (p.group == "pPb") pPb.push_back(p);
        else if (p.group == "OO") oo.push_back(p);
        else if (p.group == "PbPbRun2") pbpbRun2.push_back(p);
    }

    auto grPp = MakeGraph(pp, "g_pp", false, kCyan + 2, 20);
    grPp->SetMarkerStyle(20);
    auto grPPb = MakeGraph(pPb, "g_pPb", false, kAzure - 4, 23, 1.25);
    auto grOO = MakeGraph(oo, "g_oo", false, kBlue + 2, 33, 1.55);
    auto grPbPbRun2 = MakeGraph(pbpbRun2, "g_pbpb_run2", false, kBlue + 1, 21);
    auto grRun3Legend = MakeGraph(std::vector<Point>{run3.back()}, "g_pbpb_run3_legend", false, kRed + 1, 29, 1.35);

    std::vector<std::unique_ptr<TGraphAsymmErrors>> grRun3;
    grRun3.reserve(run3.size());
    for (size_t i = 0; i < run3.size(); ++i) {
        grRun3.push_back(MakeGraph(std::vector<Point>{run3[i]}, Form("g_pbpb_run3_%zu", i), false,
                                   run3[i].color, run3[i].marker, run3[i].markerSize));
    }

    std::vector<Point> fitPoints;
    fitPoints.insert(fitPoints.end(), refs.begin(), refs.end());
    fitPoints.insert(fitPoints.end(), run3.begin(), run3.end());
    auto grFit = MakeFitGraph(fitPoints);
    TF1 fPow("f_hyp_dndch_power", "[0]*TMath::Power(x,[1])", 4.0, 2300.0);
    fPow.SetParameters(3e-8, 1.1);
    fPow.SetLineColor(kGray + 2);
    fPow.SetLineStyle(2);
    fPow.SetLineWidth(3);
    grFit->Fit(&fPow, "Q0R");

    TCanvas c("c_hyp_dndch_pbpb536", "Hypertriton yield vs charged-particle multiplicity", 1350, 900);
    c.SetTicks(1, 1);
    c.SetLogx();
    c.SetLogy();
    c.SetLeftMargin(0.12);
    c.SetRightMargin(0.035);
    c.SetBottomMargin(0.12);
    c.SetTopMargin(0.05);

    auto *frame = c.DrawFrame(4.0, 5e-9, 2300.0, 1.5e-4,
                              ";#LTd#it{N}_{ch}/d#it{#eta}#GT_{|#it{#eta}|<0.5};#LTd#it{N}/d#it{y}#GT");
    frame->GetXaxis()->SetTitleSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.045);
    frame->GetXaxis()->SetLabelSize(0.037);
    frame->GetYaxis()->SetLabelSize(0.037);
    frame->GetYaxis()->SetTitleOffset(1.25);

    DrawSystBoxes(refs, 0.0);
    DrawSystBoxes(run3, 0.012);
    grPp->Draw("P E1 SAME");
    grPPb->Draw("P E1 SAME");
    grOO->Draw("P E1 SAME");
    grPbPbRun2->Draw("P E1 SAME");
    for (auto &g : grRun3) g->Draw("P E1 SAME");
    fPow.Draw("SAME");
    DrawHeader(0.16, 0.88);

    auto *leg = MakeLegend(0.56, 0.16, 0.94, 0.40);
    leg->AddEntry(grPp.get(), "pp #sqrt{#it{s}} = 13 TeV", "pe");
    leg->AddEntry(grPPb.get(), "p--Pb #sqrt{#it{s}_{NN}} = 5.02 TeV", "pe");
    leg->AddEntry(grOO.get(), "O--O #sqrt{#it{s}_{NN}} = 5.36 TeV", "pe");
    leg->AddEntry(grPbPbRun2.get(), "Pb--Pb #sqrt{#it{s}_{NN}} = 5.02 TeV", "pe");
    leg->AddEntry(grRun3Legend.get(), "Pb--Pb #sqrt{#it{s}_{NN}} = 5.36 TeV, this work", "pe");
    leg->AddEntry(&fPow, Form("Power fit: a x^{N}, N = %.2f #pm %.2f", fPow.GetParameter(1), fPow.GetParError(1)), "l");
    leg->Draw();
    SaveCanvas(c, outDir, "hyp_dndch_pbpb536_full");

    TCanvas cZoom("c_hyp_dndch_pbpb536_zoom", "Hypertriton yield vs multiplicity zoom", 1050, 850);
    cZoom.SetTicks(1, 1);
    cZoom.SetLogx();
    cZoom.SetLogy();
    cZoom.SetLeftMargin(0.13);
    cZoom.SetRightMargin(0.04);
    cZoom.SetBottomMargin(0.13);
    cZoom.SetTopMargin(0.055);
    auto *frameZoom = cZoom.DrawFrame(4.0, 5e-9, 70.0, 2.0e-6,
                                      ";#LTd#it{N}_{ch}/d#it{#eta}#GT_{|#it{#eta}|<0.5};#LTd#it{N}/d#it{y}#GT");
    frameZoom->GetXaxis()->SetTitleSize(0.047);
    frameZoom->GetYaxis()->SetTitleSize(0.047);
    frameZoom->GetXaxis()->SetLabelSize(0.038);
    frameZoom->GetYaxis()->SetLabelSize(0.038);
    frameZoom->GetYaxis()->SetTitleOffset(1.25);
    DrawSystBoxes(refs, 0.0);
    DrawSystBoxes(run3, 0.012);
    grPp->Draw("P E1 SAME");
    grPPb->Draw("P E1 SAME");
    grOO->Draw("P E1 SAME");
    for (auto &g : grRun3) g->Draw("P E1 SAME");
    fPow.Draw("SAME");
    DrawHeader(0.17, 0.88, true);
    auto *legZoom = MakeLegend(0.52, 0.18, 0.90, 0.38);
    legZoom->AddEntry(grPp.get(), "pp", "pe");
    legZoom->AddEntry(grPPb.get(), "p--Pb", "pe");
    legZoom->AddEntry(grOO.get(), "O--O", "pe");
    legZoom->AddEntry(grRun3Legend.get(), "Pb--Pb 5.36 TeV peripheral", "pe");
    legZoom->AddEntry(&fPow, "Power fit from full range", "l");
    legZoom->Draw();
    SaveCanvas(cZoom, outDir, "hyp_dndch_pbpb536_zoom");

    TFile fout((std::string(outDir) + "/hyp_dndch_pbpb536.root").c_str(), "RECREATE");
    c.Write();
    cZoom.Write();
    grPp->Write();
    grPPb->Write();
    grOO->Write();
    grPbPbRun2->Write();
    for (auto &g : grRun3) g->Write();
    fPow.Write();
    fout.Close();

    std::cout << "[DrawHypDndchPbPb536] Saved plots under " << outDir << std::endl;
}
