#include <TFile.h>
#include <TH1.h>
#include <TCanvas.h>
#include "../Tools/GeneralHelper.hpp"
#include "../Tools/AbsorptionHelper.h"
#include <TStyle.h>
#include <TROOT.h>
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

void draw_absorption_eff() {
    // Ensure style options are set when running as a compiled macro or executable
    if (gStyle) gStyle->SetOptStat(0);
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
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x5.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x6.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x6.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x7.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x7.5.root",
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x8.root"
    };
    std::vector<double> pt_bins = {2, 3, 4, 5.5, 8};
    std::vector<std::vector<double>> ct_bins = { {1, 3, 6, 9, 12, 18, 30},
                {1, 3, 6, 9, 12, 18, 25},
                {1, 3, 6, 9, 15, 25},
                {1, 3, 6, 10, 23} };
    std::vector<TFile*> AbsorptionFiles;
    for (const auto& path : AbsorptionFilePath) {
        TFile* f = TFile::Open(path.c_str(), "READ");
        if (!f || f->IsZombie()) {
            std::cerr << "Cannot open absorption file: " << path << "\n";
            continue;
        }
        AbsorptionFiles.push_back(f);

    }
    std::vector<std::vector<TH1*>> alleff;
    std::vector<std::vector<TH1*>> mattereff;
    std::vector<std::vector<TH1*>> antimattereff;
    alleff.reserve(pt_bins.size() - 1);
    antimattereff.reserve(pt_bins.size() - 1);
    mattereff.reserve(pt_bins.size() - 1);
    std::vector<double> multipliers;
    for (size_t i = 0; i < AbsorptionFiles.size(); ++i) {
        double multiplier = ExtractMultiplierFromTFile(AbsorptionFiles[i]);
        multipliers.push_back(multiplier);
        TTree* absTree = dynamic_cast<TTree*>(AbsorptionFiles[i]->Get("he3candidates"));
        ROOT::RDataFrame inputRDF(*absTree);
        std::string name_suffix = Form("_x%g", multiplier);
        PtAbsorptionCalculator absCalculator(&inputRDF, pt_bins, ct_bins, 7.6, name_suffix);
        absCalculator.Calculate();
        std::cout << "Processing absorption file with multiplier " << multiplier << "...\n";
        auto eff_all = absCalculator.HistRatio();
        for (size_t j = 0 ; j < pt_bins.size() - 1; ++j) {
            TH1 * hcopyall = dynamic_cast<TH1*>(eff_all["both"][j]->Clone());
            hcopyall->SetName(Form("eff_all_multi_%g_ptbin_%zu", multiplier, j));
            hcopyall->SetDirectory(nullptr);
            alleff[j].push_back(hcopyall);
            TH1 * hcopymat = dynamic_cast<TH1*>(eff_all["matter"][j]->Clone());
            hcopymat->SetName(Form("eff_matter_multi_%g_ptbin_%zu", multiplier, j));
            hcopymat->SetDirectory(nullptr);
            mattereff[j].push_back(hcopymat);
            TH1 * hcopyant = dynamic_cast<TH1*>(eff_all["antimatter"][j]->Clone());
            hcopyant->SetName(Form("eff_antimatter_multi_%g_ptbin_%zu", multiplier, j));
            hcopyant->SetDirectory(nullptr);
            antimattereff[j].push_back(hcopyant);
        }
    }
    const std::vector<int> colorPalette = {
        kRed + 1,     kBlue + 1,    kGreen + 2,   kMagenta + 2, kOrange + 7,
        kTeal + 1,    kViolet + 1,  kPink + 9,    kCyan + 2,    kGray + 3,
        kAzure + 2,   kSpring + 5,  kYellow + 2,  kOrange + 3,  kGreen + 3,
        kBlue + 3,    kRed + 3,     kMagenta + 4, kTeal + 3,    kViolet + 9
    };

    for (size_t i = 0; i < pt_bins.size() - 1; ++i) {
        TCanvas* c1 = new TCanvas(Form("c_eff_all_ptbin_%zu", i), Form("c_eff_all_ptbin_%zu", i), 800, 600);
        TLegend* leg = new TLegend(0.15, 0.1, 0.45, 0.4);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        for (size_t j = 0; j < alleff[i].size(); ++j) {
            TH1* h = alleff[i][j];
            h->GetYaxis()->SetRangeUser(0.5,1);
            const int color = colorPalette[j % colorPalette.size()];
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(20 + static_cast<int>(j));
            if (j == 0) {
                h->Draw("E");
            } else {
                h->Draw("E SAME");
            }
            leg->AddEntry(h, Form("x%g", multipliers[j]), "lep");
        }
        leg->Draw();
        c1->SaveAs(Form("../Outputs/absorption_eff_all_ptbin_%zu.pdf", i));
        delete c1;
    }
    for (size_t i = 0; i < pt_bins.size() - 1; ++i) {
        TCanvas* c1 = new TCanvas(Form("c_eff_matter_ptbin_%zu", i), Form("c_eff_matter_ptbin_%zu", i), 800, 600);
        TLegend* leg = new TLegend(0.15, 0.1, 0.45, 0.4);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        for (size_t j = 0; j < mattereff[i].size(); ++j) {
            TH1* h = mattereff[i][j];
            h->GetYaxis()->SetRangeUser(0.5,1);
            const int color = colorPalette[j % colorPalette.size()];
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(20 + static_cast<int>(j));
            if (j == 0) {
                h->Draw("E");
            } else {
                h->Draw("E SAME");
            }
            leg->AddEntry(h, Form("x%g", multipliers[j]), "lep");
        }
        leg->Draw();
        c1->SaveAs(Form("../Outputs/absorption_eff_matter_ptbin_%zu.pdf", i));
        delete c1;
    }
    for (size_t i = 0; i < pt_bins.size() - 1; ++i) {
        TCanvas* c1 = new TCanvas(Form("c_eff_antimatter_ptbin_%zu", i), Form("c_eff_antimatter_ptbin_%zu", i), 800, 600);
        TLegend* leg = new TLegend(0.15, 0.1, 0.45, 0.4);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        for (size_t j = 0; j < antimattereff[i].size(); ++j) {
            TH1* h = antimattereff[i][j];
            h->GetYaxis()->SetRangeUser(0.5,1);
            const int color = colorPalette[j % colorPalette.size()];
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(20 + static_cast<int>(j));
            if (j == 0) {
                h->Draw("E");
            } else {
                h->Draw("E SAME");
            }
            leg->AddEntry(h, Form("x%g", multipliers[j]), "lep");
        }
        leg->Draw();
        c1->SaveAs(Form("../Outputs/absorption_eff_antimatter_ptbin_%zu.pdf", i));
        delete c1;
    }
}