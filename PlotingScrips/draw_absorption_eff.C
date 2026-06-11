#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLatex.h>
#include "../Tools/GeneralHelper.hpp"
#include "../Tools/AbsorptionHelper.h"
#include <TStyle.h>
#include <TROOT.h>
#include <TSystem.h>
using namespace GeneralHelper;
using namespace Absorption;



// helper: extract multiplier from filename like "absorption_treex1.5.root"
inline double ExtractMultiplierFromFilename(const std::string& fname) {
    if (fname.empty()) return 1.0;

    // strip path, keep base name
    std::string base = fname;
    size_t p = base.find_last_of("/\\");
    if (p != std::string::npos) base = base.substr(p + 1);

    // search for last occurrence of 'x' or 'X' followed by a number (e.g. x1.5)
    std::regex re(R"([xX]([+-]?[0-9]+(?:\.[0-9]+)?))");
    std::smatch m;
    std::string::const_iterator start = base.cbegin();
    double value = 1.0;
    bool found = false;
    while (std::regex_search(start, base.cend(), m, re)) {
        try {
            value = std::stod(m[1].str()); // keep last match
            found = true;
        } catch (...) {
            // ignore parse errors, continue
        }
        start = m.suffix().first;
    }

    // fallback: try to parse a number right before ".root" if no 'x' pattern found
    if (!found) {
        std::regex re2(R"(([+-]?[0-9]+(?:\.[0-9]+)?)\.root$)", std::regex::icase);
        if (std::regex_search(base, m, re2)) {
            try { value = std::stod(m[1].str()); found = true; } catch(...) {}
        }
    }

    if (!found) {
        std::cerr << "Warning: cannot extract multiplier from filename '" << fname << "'. Using 1.0\n";
    }
    return value;
}

inline double ExtractMultiplierFromTFile(TFile* f) {
    if (!f) return 1.0;
    const char* name = f->GetName();
    return ExtractMultiplierFromFilename(name ? std::string(name) : std::string());
}

void DrawAbsorptionEff() {
    // Ensure style options are set when running as a compiled macro or executable
    if (gStyle) gStyle->SetOptStat(0);
    if (gStyle) {
        gStyle->SetTitleSize(0.045, "XYZ");
        gStyle->SetLabelSize(0.04, "XYZ");
        gStyle->SetPadTickX(1);
        gStyle->SetPadTickY(1);
    }
    std::vector<std::string> AbsorptionFilePath = {
        //"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/results/absorption/absorption_tree_x1.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x1.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x1.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x2.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x2.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x3.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x3.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x4.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x4.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x5.root"
    };
    std::vector<double> pt_bins = {2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8};
    std::vector<TFile*> AbsorptionFiles;
    for (const auto& path : AbsorptionFilePath) {
        TFile* f = TFile::Open(path.c_str(), "READ");
        if (!f || f->IsZombie()) {
            std::cerr << "Cannot open absorption file: " << path << "\n";
            continue;
        }
        AbsorptionFiles.push_back(f);

    }
    std::vector<TH1*> alleff;
    std::vector<double> multipliers;
    for (size_t i = 0; i < AbsorptionFiles.size(); ++i) {
        double multiplier = ExtractMultiplierFromTFile(AbsorptionFiles[i]);
        multipliers.push_back(multiplier);
        TTree* absTree = dynamic_cast<TTree*>(AbsorptionFiles[i]->Get("he3candidates"));
        ROOT::RDataFrame inputRDF(*absTree);
        ROOT::RDF::RNode inputNode(inputRDF);
        SpectrumAbsorptionCalculator absCalculator(inputNode, pt_bins, 7.6);
        absCalculator.Calculate();
        std::cout << "Processing absorption file with multiplier " << multiplier << "...\n";
        const auto &eff_all = absCalculator.Ratio();
        TH1 *hcopyall = dynamic_cast<TH1*>(eff_all.at("both").Clone(Form("eff_all_multi_%g", multiplier)));
        hcopyall->SetDirectory(nullptr);
        alleff.push_back(hcopyall);
    }
    const std::string outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/DrawAbsorptionEff";
    gSystem->mkdir(outDir.c_str(), true);

    const std::vector<int> colorPalette = {kAzure + 1, kBlack, kGreen + 2, kOrange + 7, kMagenta + 2,
                                           kTeal + 1, kViolet + 1, kPink + 9, kCyan + 2};
    const std::vector<int> markerPalette = {20, 21, 22, 23, 33, 34, 29, 47, 45};

    double yMin = 1.0;
    for (size_t j = 0; j < alleff.size(); ++j) {
        if (std::abs(multipliers[j] - 5.0) > 1e-6) continue;
        TH1 *h = alleff[j];
        for (int b = 1; b <= h->GetNbinsX(); ++b) {
            const double y = h->GetBinContent(b);
            if (y > 0.0) yMin = std::min(yMin, y);
        }
    }
    yMin = std::max(0.0, yMin - 0.025);

    auto styleHistogram = [&](TH1 *h, size_t j) {
        const bool isNominal = std::abs(multipliers[j] - 1.5) < 1e-6;
        const int color = isNominal ? kBlack : colorPalette[j % colorPalette.size()];
        h->SetLineColor(color);
        h->SetMarkerColor(color);
        h->SetMarkerStyle(isNominal ? 21 : markerPalette[j % markerPalette.size()]);
        h->SetMarkerSize(isNominal ? 1.2 : 0.95);
        h->SetLineWidth(isNominal ? 3 : 2);
        h->SetLineStyle(isNominal ? 2 : 1);
        h->GetYaxis()->SetRangeUser(yMin, 1.05);
        h->GetXaxis()->SetTitle("p_{T} (GeV/c)");
        h->GetYaxis()->SetTitle("Survival probability 1 - f_{abs}");
    };

    auto drawLabel = [&]() {
        TLatex latex;
        latex.SetNDC();
        latex.SetTextAlign(22);
        latex.SetTextSize(0.032);
        latex.DrawLatex(0.55, 0.86, "ALICE Run 3 absorption study");
        latex.DrawLatex(0.55, 0.81, "Matter + antimatter, 10--20% centrality p_{T} binning");
        latex.DrawLatex(0.55, 0.76, "1--5 #times ^{3}He cross section scan");
    };

    TCanvas* c1 = new TCanvas("c_absorption_eff_ptbins", "c_absorption_eff_ptbins", 950, 700);
    c1->SetLeftMargin(0.13);
    c1->SetRightMargin(0.04);
    c1->SetTopMargin(0.06);
    c1->SetBottomMargin(0.12);
    TLegend* leg = new TLegend(0.16, 0.16, 0.50, 0.39);
    leg->SetNColumns(2);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.033);
    for (size_t j = 0; j < alleff.size(); ++j) {
        TH1* h = alleff[j];
        styleHistogram(h, j);
        h->Draw(j == 0 ? "E" : "E SAME");
        leg->AddEntry(h, Form("%g #times ^{3}He%s", multipliers[j], std::abs(multipliers[j] - 1.5) < 1e-6 ? " (nominal)" : ""), "lep");
    }
    drawLabel();
    leg->Draw();
    c1->SaveAs(Form("%s/absorption_eff_ptbins_10_20.pdf", outDir.c_str()));
    delete c1;
}

void draw_absorption_eff() {
    DrawAbsorptionEff();
}
