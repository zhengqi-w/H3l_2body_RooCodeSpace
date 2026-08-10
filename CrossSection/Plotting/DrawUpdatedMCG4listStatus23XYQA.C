#include "../../Tools/GeneralHelper.hpp"

#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2D.h>
#include <TKey.h>
#include <TLatex.h>
#include <TPad.h>
#include <TStyle.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct SamplePair {
    std::string tag;
    std::string oldPath;
    std::string newPath;
};

struct HistSet {
    std::string tag;
    std::string selectionLabel;
    std::unique_ptr<TH2D> oldHist;
    std::unique_ptr<TH2D> newHist;
    double oldSelected{0.0};
    double newSelected{0.0};
};

std::unique_ptr<TChain> MakeAO2DChain(const std::string &path, const std::string &treeName)
{
    auto chain = std::make_unique<TChain>(treeName.c_str());
    TFile f(path.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Cannot open " + path);
    }
    if (auto *direct = dynamic_cast<TTree *>(f.Get(treeName.c_str()))) {
        (void)direct;
        chain->Add(path.c_str());
    } else {
        GeneralHelper::fillChainFromAO2D(*chain, &f);
    }
    if (chain->GetEntries() <= 0) {
        throw std::runtime_error("No entries found for " + treeName + " in " + path);
    }
    return chain;
}

std::unique_ptr<TH2D> CloneHist2D(const TH2D &src, const std::string &name)
{
    auto out = std::unique_ptr<TH2D>(static_cast<TH2D *>(src.Clone(name.c_str())));
    out->SetDirectory(nullptr);
    return out;
}

std::unique_ptr<TH2D> BuildXYHist(const std::string &path,
                                  const std::string &treeName,
                                  const std::string &histName,
                                  const std::string &selection,
                                  int nBins,
                                  double minXY,
                                  double maxXY,
                                  double &selected)
{
    auto chain = MakeAO2DChain(path, treeName);
    ROOT::RDataFrame rdf(*chain);
    auto filtered = rdf.Filter(selection, selection);
    auto h = filtered.Histo2D({histName.c_str(),
                               ";fGenXDecVtx (cm);fGenYDecVtx (cm);Entries",
                               nBins, minXY, maxXY, nBins, minXY, maxXY},
                              "fGenXDecVtx", "fGenYDecVtx");
    auto count = filtered.Count();
    selected = static_cast<double>(*count);
    auto out = CloneHist2D(*h, histName);
    return out;
}

void Style2D(TH2D *h)
{
    if (!h) return;
    h->SetStats(false);
    h->GetXaxis()->SetTitleOffset(1.15);
    h->GetYaxis()->SetTitleOffset(1.25);
    h->GetZaxis()->SetTitleOffset(1.2);
}

void DrawLabel(double x, double y, const std::string &text, double size = 0.035)
{
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(size);
    latex.DrawLatex(x, y, text.c_str());
}

void DrawComparisonCanvas(const HistSet &set, const std::string &outPdf, const std::string &rangeLabel)
{
    TCanvas c(("c_" + set.tag + "_" + rangeLabel).c_str(), "", 1300, 720);
    c.Divide(2, 1, 0.002, 0.002);

    c.cd(1);
    gPad->SetLogz();
    gPad->SetTicks(1, 1);
    gPad->SetRightMargin(0.16);
    gPad->SetLeftMargin(0.12);
    gPad->SetBottomMargin(0.12);
    set.oldHist->Draw("COLZ");
    DrawLabel(0.14, 0.93, set.tag + " old unreweighted G4list");
    DrawLabel(0.14, 0.88, Form("selected = %.0f", set.oldSelected), 0.032);
    DrawLabel(0.14, 0.83, set.selectionLabel, 0.03);

    c.cd(2);
    gPad->SetLogz();
    gPad->SetTicks(1, 1);
    gPad->SetRightMargin(0.16);
    gPad->SetLeftMargin(0.12);
    gPad->SetBottomMargin(0.12);
    set.newHist->Draw("COLZ");
    DrawLabel(0.14, 0.93, set.tag + " updatedMC_G4list");
    DrawLabel(0.14, 0.88, Form("selected = %.0f", set.newSelected), 0.032);
    DrawLabel(0.14, 0.83, set.selectionLabel, 0.03);

    c.SaveAs(outPdf.c_str());
}

HistSet BuildSet(const SamplePair &sample,
                 const std::string &treeName,
                 const std::string &selectionName,
                 const std::string &selection,
                 const std::string &selectionLabel,
                 int nBins,
                 double minXY,
                 double maxXY,
                 const std::string &suffix)
{
    HistSet out;
    out.tag = sample.tag;
    out.selectionLabel = selectionLabel;
    out.oldHist = BuildXYHist(sample.oldPath, treeName, "h_old_" + sample.tag + "_" + selectionName + "_" + suffix,
                              selection,
                              nBins, minXY, maxXY, out.oldSelected);
    out.newHist = BuildXYHist(sample.newPath, treeName, "h_new_" + sample.tag + "_" + selectionName + "_" + suffix,
                              selection,
                              nBins, minXY, maxXY, out.newSelected);
    Style2D(out.oldHist.get());
    Style2D(out.newHist.get());
    return out;
}

void AddTo(TH2D *dst, const TH2D *src)
{
    if (dst && src) dst->Add(src);
}

HistSet CloneForCombined(const HistSet &src, const std::string &suffix)
{
    HistSet out;
    out.tag = "combined";
    out.selectionLabel = src.selectionLabel;
    out.oldSelected = src.oldSelected;
    out.newSelected = src.newSelected;
    out.oldHist = CloneHist2D(*src.oldHist, "h_old_combined_" + suffix);
    out.newHist = CloneHist2D(*src.newHist, "h_new_combined_" + suffix);
    Style2D(out.oldHist.get());
    Style2D(out.newHist.get());
    return out;
}

} // namespace

void DrawUpdatedMCG4listStatus23XYQA(
    const std::string &outputDir =
        "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/CrossSection/Plotting/UpdatedMCG4listStatus23QA",
    const std::string &treeName = "O2mchypcands",
    int nBinsFull = 240,
    double fullMin = -400.0,
    double fullMax = 400.0,
    int nBinsZoom = 200,
    double zoomMin = -40.0,
    double zoomMax = 40.0)
{
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    std::filesystem::create_directories(outputDir);

    const std::vector<SamplePair> samples = {
        {
            "LHC25g11",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_pass5/mc/LHC25g11_G4list/NCrossedRows/AO2D_CustomV0s.root",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC25g11_G4list.root"
        },
        {
            "LHC26e5",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC24ar_pass3/mc/LHC26e5_G4list/AO2D.root",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC26e5_G4list.root"
        },
        {
            "LHC26e6",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC25_PbPb_pass1/mc/LHC26e6_G4list/AO2D.root",
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/updatedMC_G4list/AO2D_LHC26e6_G4list.root"
        }
    };

    TFile fout((outputDir + "/gen_xy_status23_not_twobody_old_vs_updated.root").c_str(), "RECREATE");

    auto drawSelection = [&](const std::string &selectionName,
                             const std::string &selection,
                             const std::string &selectionLabel) {
        std::vector<HistSet> fullSets;
        std::vector<HistSet> zoomSets;
        fullSets.reserve(samples.size());
        zoomSets.reserve(samples.size());

        for (const auto &sample : samples) {
            std::cout << "[Info] Build full-range XY QA for " << sample.tag
                      << " (" << selectionName << ")" << std::endl;
            fullSets.push_back(BuildSet(sample, treeName, selectionName, selection, selectionLabel,
                                        nBinsFull, fullMin, fullMax, "full"));
            DrawComparisonCanvas(fullSets.back(),
                                 outputDir + "/gen_xy_status23_" + selectionName + "_" + sample.tag + "_full.pdf",
                                 "full");

            std::cout << "[Info] Build zoom XY QA for " << sample.tag
                      << " (" << selectionName << ")" << std::endl;
            zoomSets.push_back(BuildSet(sample, treeName, selectionName, selection, selectionLabel,
                                        nBinsZoom, zoomMin, zoomMax, "zoom"));
            DrawComparisonCanvas(zoomSets.back(),
                                 outputDir + "/gen_xy_status23_" + selectionName + "_" + sample.tag + "_zoom.pdf",
                                 "zoom");
        }

        auto combinedFull = CloneForCombined(fullSets.front(), selectionName + "_full");
        for (size_t i = 1; i < fullSets.size(); ++i) {
            AddTo(combinedFull.oldHist.get(), fullSets[i].oldHist.get());
            AddTo(combinedFull.newHist.get(), fullSets[i].newHist.get());
            combinedFull.oldSelected += fullSets[i].oldSelected;
            combinedFull.newSelected += fullSets[i].newSelected;
        }
        DrawComparisonCanvas(combinedFull,
                             outputDir + "/gen_xy_status23_" + selectionName + "_combined_full.pdf",
                             "full");

        auto combinedZoom = CloneForCombined(zoomSets.front(), selectionName + "_zoom");
        for (size_t i = 1; i < zoomSets.size(); ++i) {
            AddTo(combinedZoom.oldHist.get(), zoomSets[i].oldHist.get());
            AddTo(combinedZoom.newHist.get(), zoomSets[i].newHist.get());
            combinedZoom.oldSelected += zoomSets[i].oldSelected;
            combinedZoom.newSelected += zoomSets[i].newSelected;
        }
        DrawComparisonCanvas(combinedZoom,
                             outputDir + "/gen_xy_status23_" + selectionName + "_combined_zoom.pdf",
                             "zoom");

        fout.cd();
        for (const auto &set : fullSets) {
            set.oldHist->Write();
            set.newHist->Write();
        }
        for (const auto &set : zoomSets) {
            set.oldHist->Write();
            set.newHist->Write();
        }
        combinedFull.oldHist->Write(("h_old_combined_" + selectionName + "_full").c_str());
        combinedFull.newHist->Write(("h_new_combined_" + selectionName + "_full").c_str());
        combinedZoom.oldHist->Write(("h_old_combined_" + selectionName + "_zoom").c_str());
        combinedZoom.newHist->Write(("h_new_combined_" + selectionName + "_zoom").c_str());
    };

    drawSelection("not_twobody",
                  "fStatusCode == 23 && !fIsTwoBodyDecay",
                  "fStatusCode == 23 && !fIsTwoBodyDecay");
    drawSelection("twobody",
                  "fStatusCode == 23 && fIsTwoBodyDecay",
                  "fStatusCode == 23 && fIsTwoBodyDecay");
    drawSelection("all_decay",
                  "fStatusCode == 23",
                  "fStatusCode == 23");

    fout.Close();

    std::cout << "[Done] Updated MC G4list XY QA written to " << outputDir << std::endl;
}
