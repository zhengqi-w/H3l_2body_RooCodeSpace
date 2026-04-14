#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TString.h>
#include <iostream>
#include <memory>

namespace {
std::unique_ptr<TH1> CloneHist(TFile *f, const char *path, const char *newName) {
    if (!f) return nullptr;
    auto *h = dynamic_cast<TH1 *>(f->Get(path));
    if (!h) return nullptr;
    auto out = std::unique_ptr<TH1>(dynamic_cast<TH1 *>(h->Clone(newName)));
    if (out) out->SetDirectory(nullptr);
    return out;
}

void NormalizeHist(TH1 *h) {
    if (!h) return;
    const double sum = h->Integral(0, h->GetNbinsX() + 1);
    if (sum > 0.0) h->Scale(1.0 / sum);
}

void StyleHist(TH1 *h, Color_t color, Style_t marker) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetMarkerSize(1.1);
    h->SetLineWidth(2);
}

double GetMaxInRange(const TH1 *h, double xmin, double xmax) {
    if (!h) return 0.0;
    const int bmin = h->GetXaxis()->FindBin(xmin);
    const int bmax = h->GetXaxis()->FindBin(xmax);
    double maxv = 0.0;
    for (int ib = bmin; ib <= bmax; ++ib) {
        maxv = std::max(maxv, h->GetBinContent(ib));
    }
    return maxv;
}
} // namespace

void TempComparePass4Pass5(
    const char *pass4File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
    const char *pass5File = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MCEfficiency_NCrossedRows") {

    gSystem->mkdir(outDir, true);
    gStyle->SetOptStat(0);

    std::unique_ptr<TFile> f4(TFile::Open(pass4File, "READ"));
    std::unique_ptr<TFile> f5(TFile::Open(pass5File, "READ"));

    if (!f4 || f4->IsZombie()) {
        std::cerr << "ERROR: cannot open pass4 file: " << pass4File << std::endl;
        return;
    }
    if (!f5 || f5->IsZombie()) {
        std::cerr << "ERROR: cannot open pass5 file: " << pass5File << std::endl;
        return;
    }

    auto hZvtx4 = CloneHist(f4.get(), "hyper-reco-task/hZvtx", "hZvtx_pass4");
    auto hZvtx5 = CloneHist(f5.get(), "hyper-reco-task/hZvtx", "hZvtx_pass5");
    auto hCent4 = CloneHist(f4.get(), "hyper-reco-task/hCentFT0C", "hCent_pass4");
    auto hCent5 = CloneHist(f5.get(), "hyper-reco-task/hCentFT0C", "hCent_pass5");

    if (!hZvtx4 || !hZvtx5) {
        std::cerr << "ERROR: missing histogram hyper-reco-task/hZvtx in one or both files." << std::endl;
        return;
    }
    if (!hCent4 || !hCent5) {
        std::cerr << "ERROR: missing histogram hyper-reco-task/hCentFT0C in one or both files." << std::endl;
        return;
    }

    const char *label4 = "pass4";
    const char *label5 = "pass5";

    const double nEvt4 = hZvtx4->Integral(0, hZvtx4->GetNbinsX() + 1);
    const double nEvt5 = hZvtx5->Integral(0, hZvtx5->GetNbinsX() + 1);

    NormalizeHist(hZvtx4.get());
    NormalizeHist(hZvtx5.get());
    NormalizeHist(hCent4.get());
    NormalizeHist(hCent5.get());

    StyleHist(hZvtx4.get(), kRed + 1, 20);
    StyleHist(hZvtx5.get(), kAzure + 2, 24);
    StyleHist(hCent4.get(), kRed + 1, 20);
    StyleHist(hCent5.get(), kAzure + 2, 24);

    {
        TCanvas c("c_hZvtx_compare", "hZvtx normalized comparison", 900, 700);
        hZvtx4->SetTitle("hyper-reco-task/hZvtx (Normalized);z_{vtx} (cm);Normalized counts");
        hZvtx4->SetMinimum(0.0);
        hZvtx4->Draw("E1");
        hZvtx5->Draw("E1 SAME");

        TLegend leg(0.55, 0.72, 0.88, 0.88);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.SetTextSize(0.04);
        leg.AddEntry(hZvtx4.get(), label4, "lep");
        leg.AddEntry(hZvtx5.get(), label5, "lep");
        leg.Draw();

        TLatex latex;
        latex.SetNDC();
        latex.SetTextSize(0.037);
        latex.DrawLatex(0.16, 0.86, Form("NEvents: %s = %.3e", label4, nEvt4));
        latex.DrawLatex(0.16, 0.80, Form("NEvents: %s = %.3e", label5, nEvt5));

        c.SaveAs(Form("%s/hZvtx_compare_norm.pdf", outDir));
    }

    {
        TCanvas c("c_hCent_compare", "hCentFT0C normalized comparison", 900, 700);
        hCent4->SetTitle("hyper-reco-task/hCentFT0C (Normalized, 0-90);Centrality FT0C (%%);Normalized counts");
        hCent4->SetMinimum(0.0);
        hCent4->GetXaxis()->SetRangeUser(0.0, 90.0);
        const double maxCent = std::max(GetMaxInRange(hCent4.get(), 0.0, 90.0), GetMaxInRange(hCent5.get(), 0.0, 90.0));
        hCent4->SetMaximum((maxCent > 0.0) ? 1.25 * maxCent : 1.0);
        hCent4->Draw("E1");

        hCent5->GetXaxis()->SetRangeUser(0.0, 90.0);
        hCent5->Draw("E1 SAME");

        TLegend leg(0.55, 0.76, 0.88, 0.90);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.SetTextSize(0.04);
        leg.AddEntry(hCent4.get(), label4, "lep");
        leg.AddEntry(hCent5.get(), label5, "lep");
        leg.Draw();

        c.SaveAs(Form("%s/hCentFT0C_compare_norm_0_90.pdf", outDir));
    }

    std::cout << "Saved: " << outDir << "/hZvtx_compare_norm.pdf" << std::endl;
    std::cout << "Saved: " << outDir << "/hCentFT0C_compare_norm_0_90.pdf" << std::endl;
}
