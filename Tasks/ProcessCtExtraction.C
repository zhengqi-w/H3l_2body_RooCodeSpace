// ProcessCtExtraction.C
// Usage from ROOT prompt:
//   root -l -b -q 'Tasks/ProcessCtExtraction.C("configs/ct_extraction.json", true)'
// The macro loads the JSON config, enables implicit MT (optional), and runs CtExtraction.

#include <TROOT.h>
#include <TSystem.h>

#include <exception>
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <TROOT.h>
#include <TSystem.h>
#include <TDirectory.h>

#include "../Tools/CtExtraction.h"
#include "../Tools/CtExtraction.cxx"
#include "../Tools/GeneralHelper.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void ProcessCtExtraction(const char *configPath = "../configs/ct_extraction.json",
                         bool enableImplicitMT = true) {
#include <TCanvas.h>
#include <TH1.h>
#include <RooPlot.h>
    if (!configPath || std::string(configPath).empty()) {
        throw std::runtime_error("ProcessCtExtraction: config path is empty");
    }

    if (enableImplicitMT) {
        GeneralHelper::EnableImplicitMTWithPreferredThreads();
    }

    try {
        std::ifstream ifs(configPath);
        if (!ifs) {
            throw std::runtime_error(std::string("Cannot open config: ") + configPath);
        }
        json cfgJson = json::parse(ifs, nullptr, true, true);
        auto get_string = [&](const char *key, const std::string &fallback = std::string()) {
            return cfgJson.value(key, fallback);
        };
        // direct sigma mc->data ranges from config (per pt bin)
        std::vector<std::vector<double>> sigmaRangesCfg = cfgJson.value("sigma_mc_to_data_range", std::vector<std::vector<double>>{});

        CtExtraction extractor(configPath);
        // *************************sigma range setting block start*************************
        // Use sigma_mc_to_data_range directly from config (per-pt-bin array of [min,max])
        if (!sigmaRangesCfg.empty()) {
            extractor.SetSigmaRangeMcToData(sigmaRangesCfg);
            std::cout << "[ProcessCtExtraction] Using sigma_mc_to_data_range from config for "
                      << sigmaRangesCfg.size() << " pt bins." << std::endl;
        }
        // *************************sigma range setting block end*************************

        extractor.Run();
        std::cout << "[ProcessCtExtraction] Completed successfully using config: "
                  << configPath << std::endl;

        // -------- Post-processing: export key histograms/canvases to PDF --------
        const std::string matter = cfgJson.value("is_matter", std::string("both"));
        const std::string outDirBase = cfgJson.value("output_dir", std::string("results/ct_extraction"));
        const std::string outFileBase = cfgJson.value("output_file", std::string("ct_analysis"));
        const std::string trialSuffix = cfgJson.value("trial_suffix", std::string(""));
        const std::string stdDirName = trialSuffix.empty() ? "std" : ("std_" + trialSuffix);
        const std::filesystem::path outDir = std::filesystem::path(outDirBase) / matter;
        const std::filesystem::path rootPath = outDir / (outFileBase + ".root");
        std::filesystem::create_directories(outDir);
        if (!std::filesystem::exists(rootPath)) {
            std::cerr << "[ProcessCtExtraction] Output ROOT file not found: " << rootPath << std::endl;
            return;
        }

        auto styleAndSaveHist = [&](TH1 *h, const std::string &pdfName, Color_t color, const std::string &title, const char *yTitle = nullptr) {
            if (!h) return;
            h->SetStats(false); // hide statistics box on exported plots
            h->SetTitle(title.c_str());
            if (yTitle) h->GetYaxis()->SetTitle(yTitle);
            h->SetLineColor(color);
            h->SetMarkerColor(color);
            h->SetMarkerStyle(20);
            h->SetLineStyle(2); // dashed line
            h->SetLineWidth(2);
            TCanvas c("c_tmp", title.c_str(), 900, 650);
            c.SetLeftMargin(0.14);
            c.SetBottomMargin(0.12);
            c.SetTopMargin(0.08);
            c.SetRightMargin(0.05);
            c.SetGridx();
            c.SetGridy();
            h->Draw("EP L");
            c.SaveAs(pdfName.c_str());
        };

        auto saveCanvas = [&](TCanvas *c, const std::string &pdfName, const std::string &title) {
            if (!c) return;
            c->SetTitle(title.c_str());
            c->SaveAs(pdfName.c_str());
        };

        auto saveRooPlot = [&](RooPlot *fr, const std::string &pdfName, const std::string &title) {
            if (!fr) return;
            TCanvas c("c_frame", title.c_str(), 900, 650);
            fr->SetTitle(title.c_str());
            fr->Draw();
            c.SaveAs(pdfName.c_str());
        };

        std::unique_ptr<TFile> tf(TFile::Open(rootPath.string().c_str(), "READ"));
        if (!tf || tf->IsZombie()) {
            std::cerr << "[ProcessCtExtraction] Failed to open output file: " << rootPath << std::endl;
            return;
        }
        TDirectory *stdDir = tf->GetDirectory(stdDirName.c_str());
        if (!stdDir) {
            std::cerr << "[ProcessCtExtraction] std directory not found: " << stdDirName << std::endl;
            return;
        }

        // Save tau vs pt histogram
        if (auto hTau = dynamic_cast<TH1*>(stdDir->Get("tau_per_ptbin"))) {
            std::string pdf = (outDir / "tau_per_ptbin.pdf").string();
            styleAndSaveHist(hTau, pdf, kBlack, Form("#tau vs p_{T} (%s)", matter.c_str()));
        }

        // Loop pt directories
        TIter nextPtDir(stdDir->GetListOfKeys());
        while (TObject *obj = nextPtDir()) {
            TKey *key = dynamic_cast<TKey*>(obj);
            if (!key) continue;
            if (std::string(key->GetClassName()) != "TDirectoryFile") continue;
            TDirectory *ptDir = dynamic_cast<TDirectory*>(key->ReadObj());
            if (!ptDir) continue;
            const std::string ptDirName = ptDir->GetName();

            auto exportHistIfPresent = [&](const char *hname, Color_t col, const char *tag, const char *yTitle) {
                if (auto h = dynamic_cast<TH1*>(ptDir->Get(hname))) {
                    const std::string pdf = (outDir / Form("%s_%s.pdf", tag, ptDirName.c_str())).string();
                    const std::string ttl = Form("%s (%s, %s)", tag, ptDirName.c_str(), matter.c_str());
                    styleAndSaveHist(h, pdf, col, ttl, yTitle);
                }
            };

            exportHistIfPresent(Form("h_acc_eff_%s", ptDirName.c_str()), kBlue + 1, "Acceptance", "Efficiency #times Acceptance");
            exportHistIfPresent(Form("h_bdt_eff_%s", ptDirName.c_str()), kGreen + 2, "BDT efficiency", "BDT efficiency");

            if (auto cFit = dynamic_cast<TCanvas*>(ptDir->Get(Form("c_ct_fit_%s", ptDirName.c_str())))) {
                const std::string pdf = (outDir / Form("ct_fit_%s.pdf", ptDirName.c_str())).string();
                const std::string ttl = Form("CT fit %s (%s)", ptDirName.c_str(), matter.c_str());
                saveCanvas(cFit, pdf, ttl);
            }

            // Export any RooPlot in the pt directory (e.g., mc_massfit_*, data_massfit_*)
            TIter nextObj(ptDir->GetListOfKeys());
            while (TObject *o = nextObj()) {
                TKey *k2 = dynamic_cast<TKey*>(o);
                if (!k2) continue;
                TObject *obj2 = k2->ReadObj();
                if (auto fr = dynamic_cast<RooPlot*>(obj2)) {
                    const std::string name = fr->GetName();
                    const std::string pdf = (outDir / Form("%s.pdf", name.c_str())).string();
                    const std::string ttl = Form("%s (%s)", name.c_str(), matter.c_str());
                    saveRooPlot(fr, pdf, ttl);
                }
            }
        }
    } catch (const std::exception &ex) {
        std::cerr << "[ProcessCtExtraction] Error: " << ex.what() << std::endl;
        throw;
    }
}
