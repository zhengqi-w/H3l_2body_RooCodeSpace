#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TPad.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Point {
    std::string label;
    double ptMin{0.0};
    double ptMax{0.0};
    double x{0.0};
    double ex{0.0};
    double y{0.0};
    double ey{0.0};
};

std::vector<std::string> Split(const std::string &s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

double ToDouble(const std::string &s) {
    try {
        return std::stod(s);
    } catch (...) {
        return 0.0;
    }
}

std::string JoinAsDecimal(const std::vector<std::string> &parts, int first, int last) {
    std::string out;
    for (int i = first; i < last; ++i) {
        if (i > first) out += ".";
        out += parts[i];
    }
    return out;
}

bool ParsePtRange(const std::string &label, double binWidth, double &ptMin, double &ptMax) {
    const std::string key = "_pt_";
    const auto pos = label.find(key);
    if (pos == std::string::npos) return false;

    const auto parts = Split(label.substr(pos + key.size()), '_');
    if (parts.size() < 2) return false;

    bool found = false;
    double bestMin = 0.0;
    double bestMax = 0.0;
    double bestScore = 1e30;
    for (int split = 1; split < static_cast<int>(parts.size()); ++split) {
        const double low = ToDouble(JoinAsDecimal(parts, 0, split));
        const double high = ToDouble(JoinAsDecimal(parts, split, parts.size()));
        if (!(high > low)) continue;
        const double score = std::abs((high - low) - binWidth);
        if (score < bestScore) {
            bestScore = score;
            bestMin = low;
            bestMax = high;
            found = true;
        }
    }

    if (!found) return false;
    ptMin = bestMin;
    ptMax = bestMax;
    return true;
}

std::map<std::string, std::vector<Point>> LoadCsv(const std::string &path) {
    std::map<std::string, std::vector<Point>> out;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        return out;
    }

    std::string line;
    if (!std::getline(in, line)) return out;
    const auto header = Split(line, ',');
    std::map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) col[header[i]] = i;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto fields = Split(line, ',');
        if (fields.size() < header.size()) continue;

        Point p;
        const std::string group = fields[col["group"]];
        p.label = fields[col["label"]];
        const double binWidth = ToDouble(fields[col["bin_width"]]);
        if (!ParsePtRange(p.label, binWidth, p.ptMin, p.ptMax)) continue;
        p.x = 0.5 * (p.ptMin + p.ptMax);
        p.ex = 0.5 * (p.ptMax - p.ptMin);
        p.y = ToDouble(fields[col["corrected"]]);
        p.ey = ToDouble(fields[col["corrected_err"]]);
        out[group].push_back(p);
    }

    for (auto &kv : out) {
        std::sort(kv.second.begin(), kv.second.end(), [](const Point &a, const Point &b) {
            return a.ptMin < b.ptMin;
        });
    }
    return out;
}

TGraphErrors *MakeGraph(const std::vector<Point> &points, const char *name, Color_t color, Style_t marker) {
    auto *g = new TGraphErrors(points.size());
    g->SetName(name);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        g->SetPoint(i, points[i].x, points[i].y);
        g->SetPointError(i, points[i].ex, points[i].ey);
    }
    g->SetLineColor(color);
    g->SetMarkerColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.05);
    g->SetLineWidth(2);
    return g;
}

TGraphErrors *MakeRatioGraph(const std::vector<Point> &topology, const std::vector<Point> &bdt) {
    std::vector<Point> ratios;
    for (const auto &topo : topology) {
        const auto it = std::find_if(bdt.begin(), bdt.end(), [&topo](const Point &ref) {
            return std::abs(ref.ptMin - topo.ptMin) < 1e-6 && std::abs(ref.ptMax - topo.ptMax) < 1e-6;
        });
        if (it == bdt.end() || it->y <= 0.0 || topo.y <= 0.0) continue;
        Point r;
        r.x = topo.x;
        r.ex = topo.ex;
        r.y = topo.y / it->y;
        r.ey = r.y * std::sqrt(std::pow(topo.ey / topo.y, 2) + std::pow(it->ey / it->y, 2));
        ratios.push_back(r);
    }
    return MakeGraph(ratios, "g_ratio_topology_bdt", kRed + 1, kFullSquare);
}

std::string CentText(const std::string &cent) {
    if (cent == "cen_0_10") return "0--10%";
    if (cent == "cen_10_30") return "10--30%";
    if (cent == "cen_30_50") return "30--50%";
    if (cent == "cen_50_80") return "50--80%";
    return cent;
}

std::string FileTag(const std::string &cent) {
    std::string out = cent;
    std::replace(out.begin(), out.end(), '_', '-');
    return out;
}

std::pair<double, double> SpectrumRange(const std::vector<Point> &a, const std::vector<Point> &b) {
    double ymin = 1e30;
    double ymax = 0.0;
    for (const auto *vec : {&a, &b}) {
        for (const auto &p : *vec) {
            if (p.y <= 0.0) continue;
            ymin = std::min(ymin, std::max(1e-30, p.y - p.ey));
            ymax = std::max(ymax, p.y + p.ey);
        }
    }
    if (ymax <= 0.0 || ymin >= 1e29) return {1e-8, 1e-4};
    return {ymin * 0.45, ymax * 4.5};
}

std::pair<double, double> RatioRange(const TGraphErrors *g) {
    double ymin = 1.0;
    double ymax = 1.0;
    bool seen = false;
    for (int i = 0; g && i < g->GetN(); ++i) {
        double x = 0.0;
        double y = 0.0;
        g->GetPoint(i, x, y);
        const double ey = g->GetErrorY(i);
        if (!std::isfinite(y)) continue;
        ymin = seen ? std::min(ymin, y - ey) : y - ey;
        ymax = seen ? std::max(ymax, y + ey) : y + ey;
        seen = true;
    }
    if (!seen) return {0.4, 1.6};
    const double span = std::max(0.4, ymax - ymin);
    ymin = std::max(0.0, ymin - 0.25 * span);
    ymax += 0.25 * span;
    ymin = std::min(ymin, 0.85);
    ymax = std::max(ymax, 1.15);
    return {ymin, ymax};
}

void DrawOne(const std::string &cent, const std::vector<Point> &topology, const std::vector<Point> &bdt,
             const std::string &outDir) {
    if (topology.empty() || bdt.empty()) return;

    auto *c = new TCanvas(("c_topology_bdt_" + cent).c_str(), "", 760, 760);
    c->SetMargin(0, 0, 0, 0);

    auto *padTop = new TPad("padTop", "", 0.0, 0.31, 1.0, 1.0);
    auto *padBot = new TPad("padBot", "", 0.0, 0.0, 1.0, 0.31);
    padTop->SetBottomMargin(0.025);
    padTop->SetLeftMargin(0.14);
    padTop->SetRightMargin(0.04);
    padTop->SetTopMargin(0.055);
    padTop->SetLogy();
    padBot->SetTopMargin(0.025);
    padBot->SetBottomMargin(0.31);
    padBot->SetLeftMargin(0.14);
    padBot->SetRightMargin(0.04);
    padTop->Draw();
    padBot->Draw();

    const double xmin = std::min(topology.front().ptMin, bdt.front().ptMin);
    const double xmax = std::max(topology.back().ptMax, bdt.back().ptMax);
    const auto yr = SpectrumRange(topology, bdt);

    padTop->cd();
    TH1F *frame = padTop->DrawFrame(xmin, yr.first, xmax, yr.second);
    frame->GetYaxis()->SetTitle("#frac{1}{N_{ev}} #frac{d^{2}N}{d#it{y}d#it{p}_{T}} (#it{c}/GeV)");
    frame->GetYaxis()->SetTitleSize(0.053);
    frame->GetYaxis()->SetTitleOffset(1.16);
    frame->GetYaxis()->SetLabelSize(0.046);
    frame->GetXaxis()->SetLabelSize(0);

    auto *gBdt = MakeGraph(bdt, "g_bdt", kAzure + 2, kFullCircle);
    auto *gTopo = MakeGraph(topology, "g_topology", kRed + 1, kFullSquare);
    gBdt->Draw("P SAME");
    gTopo->Draw("P SAME");

    TLatex lat;
    lat.SetNDC();
    lat.SetTextFont(42);
    lat.SetTextSize(0.044);
    lat.DrawLatex(0.18, 0.89, "ALICE Run 3 internal");
    lat.SetTextSize(0.037);
    lat.DrawLatex(0.18, 0.835, "Pb--Pb #sqrt{#it{s}_{NN}} = 5.36 TeV, LHC23 pass5");
    lat.DrawLatex(0.18, 0.785, (CentText(cent) + ", matter + antimatter").c_str());
    lat.DrawLatex(0.18, 0.735, "Stat. uncertainties only");

    auto *leg = new TLegend(0.61, 0.73, 0.93, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.038);
    leg->AddEntry(gBdt, "BDT spectrum", "lep");
    leg->AddEntry(gTopo, "Topology spectrum", "lep");
    leg->Draw();

    padBot->cd();
    auto *gRatio = MakeRatioGraph(topology, bdt);
    const auto rr = RatioRange(gRatio);
    TH1F *rFrame = padBot->DrawFrame(xmin, rr.first, xmax, rr.second);
    rFrame->GetYaxis()->SetTitle("Topology / BDT");
    rFrame->GetYaxis()->SetTitleSize(0.105);
    rFrame->GetYaxis()->SetTitleOffset(0.58);
    rFrame->GetYaxis()->SetLabelSize(0.085);
    rFrame->GetYaxis()->SetNdivisions(505);
    rFrame->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
    rFrame->GetXaxis()->SetTitleSize(0.105);
    rFrame->GetXaxis()->SetTitleOffset(1.05);
    rFrame->GetXaxis()->SetLabelSize(0.085);

    auto *line = new TLine(xmin, 1.0, xmax, 1.0);
    line->SetLineColor(kGray + 2);
    line->SetLineStyle(2);
    line->SetLineWidth(2);
    line->Draw();
    gRatio->Draw("P SAME");

    c->SaveAs((outDir + "/topology_bdt_spectrum_" + cent + ".pdf").c_str());
    c->SaveAs((outDir + "/topology_bdt_spectrum_" + cent + ".png").c_str());
    delete c;
}
} // namespace

void CompareTopologyBDTSpectrumQA() {
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.5);
    gStyle->SetTitleFont(42, "xyz");
    gStyle->SetLabelFont(42, "xyz");

    const std::string base = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID";
    const std::string topologyCsv = base + "/topology_spectrum/both/corrections_all.csv";
    const std::string bdtCsv = base + "/bdt_spectrum/both/corrections_all.csv";
    const std::string outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/TopologyBDTQA";
    gSystem->mkdir(outDir.c_str(), true);

    const auto topology = LoadCsv(topologyCsv);
    const auto bdt = LoadCsv(bdtCsv);
    const std::vector<std::string> centralities = {"cen_0_10", "cen_10_30", "cen_30_50", "cen_50_80"};

    for (const auto &cent : centralities) {
        if (!topology.count(cent) || !bdt.count(cent)) {
            std::cerr << "[Warn] missing centrality " << cent << std::endl;
            continue;
        }
        DrawOne(cent, topology.at(cent), bdt.at(cent), outDir);
    }
    std::cout << "Saved topology-BDT QA plots to " << outDir << std::endl;
}
