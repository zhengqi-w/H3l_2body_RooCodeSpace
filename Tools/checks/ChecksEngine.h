#ifndef CHECKS_ENGINE_H
#define CHECKS_ENGINE_H

#include "ChecksConfig.h"
#include "../binning/BinPlan.h"
#include "../GeneralHelper.hpp"

#include <ROOT/RDataFrame.hxx>

#include <TDirectory.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TTree.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>

namespace UnifiedAnalysis {

class ChecksEngine {
public:
    void Run(const ChecksConfig &cfg, const BinPlan &plan) const {
        if (!cfg.enabled) return;
        std::filesystem::create_directories(std::filesystem::path(cfg.outputRootFile).parent_path());
        TFile fout(cfg.outputRootFile.c_str(), "UPDATE");
        if (fout.IsZombie()) {
            std::cerr << "[ChecksEngine] failed to open output: " << cfg.outputRootFile << std::endl;
            return;
        }

        const auto baseDir = std::filesystem::path(cfg.outputRootFile).parent_path();

        // Clean stale outputs when a block is disabled, so old results do not appear as fresh outputs.
        if (!cfg.mcChecks.enable) {
            if (fout.GetDirectory("mc_checks")) {
                fout.Delete("mc_checks;*");
            }
            std::error_code ec;
            std::filesystem::remove_all(baseDir / "mc_checks", ec);
        }
        if (!cfg.dataAllChecks.enable) {
            if (fout.GetDirectory("data_all_checks")) {
                fout.Delete("data_all_checks;*");
            }
            std::error_code ec;
            std::filesystem::remove_all(baseDir / "data_all_checks", ec);
        }

        ProcessBlock(fout, "mc_checks", cfg.mcChecks, cfg.axisPool, true, "", true);
        if (cfg.mcChecks.enable) {
            CheckBlockConfig recoBlock = cfg.mcChecks;
            recoBlock.selection = MergeSelection(recoBlock.selection, "fIsReco > 0");
            ProcessBlock(fout, "mc_checks", recoBlock, cfg.axisPool, true, "_reco", false);
        }
        ProcessBlock(fout, "data_all_checks", cfg.dataAllChecks, cfg.axisPool, false, "", true);

        if (cfg.savePdf) {
            if (cfg.mcChecks.enable) {
                ExportDirectoryToPdf(fout, "mc_checks", baseDir / "mc_checks");
            }
            if (cfg.dataAllChecks.enable) {
                ExportDirectoryToPdf(fout, "data_all_checks", baseDir / "data_all_checks");
            }
        }

        std::cout << "[ChecksEngine] mode=" << plan.mode << ", bins=" << plan.items.size() << std::endl;
        fout.Close();
    }

private:
    static std::string MergeSelection(const std::string &base, const std::string &extra) {
        if (extra.empty()) return base;
        if (base.empty()) return extra;
        return "(" + base + ") && (" + extra + ")";
    }

    static std::unique_ptr<TChain> MakeChainForChecks(const std::string &file, const std::string &tree) {
        auto chain = std::make_unique<TChain>(tree.c_str());
        TFile f(file.c_str(), "READ");
        if (f.IsZombie()) {
            throw std::runtime_error("Failed to open " + file);
        }
        TTree *t = dynamic_cast<TTree *>(f.Get(tree.c_str()));
        if (t) {
            chain->Add(file.c_str());
        } else {
            GeneralHelper::fillChainFromAO2D(*chain, &f);
        }
        if (chain->GetEntries() == 0) {
            throw std::runtime_error("No entries found for tree " + tree + " in " + file);
        }
        return chain;
    }

    static AxisSpec GetAxis(const std::string &var,
                            const std::unordered_map<std::string, AxisSpec> &axisPool) {
        auto it = axisPool.find(var);
        if (it != axisPool.end()) return it->second;
        AxisSpec ax;
        ax.title = var;
        if (var == "fPt" || var == "fAbsGenPt") {
            ax = AxisSpec{80, 0.0, 12.0, var};
        } else if (var == "fCt") {
            ax = AxisSpec{80, 0.0, 40.0, var};
        } else if (var == "fCosPA") {
            ax = AxisSpec{100, 0.95, 1.0, var};
        } else if (var == "fNSigmaHe") {
            ax = AxisSpec{100, -10.0, 10.0, var};
        }
        return ax;
    }

    static void ProcessBlock(TFile &fout,
                             const std::string &dirName,
                             const CheckBlockConfig &block,
                             const std::unordered_map<std::string, AxisSpec> &axisPool,
                             bool isMC,
                             const std::string &nameSuffix,
                             bool resetDir) {
        if (!block.enable || block.file.empty() || block.tree.empty()) return;
        if (!std::filesystem::exists(block.file)) {
            std::cerr << "[ChecksEngine] skip " << dirName << ": file not found " << block.file << std::endl;
            return;
        }

        auto chain = MakeChainForChecks(block.file, block.tree);
        ROOT::RDataFrame rdf(*chain);
        auto converted = GeneralHelper::CorrectAndConvertRDF(rdf, false, isMC, false);
        ROOT::RDF::RNode node(converted);
        if (!block.selection.empty()) {
            node = node.Filter(block.selection);
        }

        std::vector<ROOT::RDF::RResultPtr<TH1D>> h1Ptrs;
        h1Ptrs.reserve(block.variables.size());
        for (const auto &var : block.variables) {
            if (var.empty()) continue;
            const auto ax = GetAxis(var, axisPool);
            const std::string xTitle = ax.title.empty() ? var : ax.title;
            const std::string hTitle = ";" + xTitle + ";Counts";
            const std::string hname = "h1_" + var + nameSuffix;
            h1Ptrs.emplace_back(node.Histo1D({hname.c_str(), hTitle.c_str(), ax.nBins, ax.min, ax.max}, var));
        }

        std::vector<ROOT::RDF::RResultPtr<TH2D>> h2Ptrs;
        h2Ptrs.reserve(block.hist2dPairs.size());
        for (const auto &p : block.hist2dPairs) {
            if (p.x.empty() || p.y.empty()) continue;
            const auto ax = GetAxis(p.x, axisPool);
            const auto ay = GetAxis(p.y, axisPool);
            const std::string name = "h2_" + p.x + "_vs_" + p.y + nameSuffix;
            const std::string xTitle = ax.title.empty() ? p.x : ax.title;
            const std::string yTitle = ay.title.empty() ? p.y : ay.title;
            const std::string hTitle = ";" + xTitle + ";" + yTitle + ";Counts";
            h2Ptrs.emplace_back(node.Histo2D({name.c_str(), hTitle.c_str(), ax.nBins, ax.min, ax.max, ay.nBins, ay.min, ay.max}, p.x, p.y));
        }

        if (resetDir && fout.GetDirectory(dirName.c_str())) {
            fout.Delete((dirName + ";*").c_str());
        }
        TDirectory *dir = fout.GetDirectory(dirName.c_str());
        if (!dir) dir = fout.mkdir(dirName.c_str());
        if (!dir) return;
        dir->cd();

        for (auto &h : h1Ptrs) {
            if (h) h->Write(h->GetName(), TObject::kOverwrite);
        }
        for (auto &h2 : h2Ptrs) {
            if (h2) h2->Write(h2->GetName(), TObject::kOverwrite);
        }
    }

    static void ExportDirectoryToPdf(TFile &fout,
                                     const std::string &dirName,
                                     const std::filesystem::path &outDir) {
        TDirectory *dir = fout.GetDirectory(dirName.c_str());
        if (!dir) return;
        std::filesystem::create_directories(outDir);

        TIter next(dir->GetListOfKeys());
        while (auto *key = next()) {
            TObject *obj = dir->Get(key->GetName());
            if (!obj) continue;

            if (auto *h1 = dynamic_cast<TH1 *>(obj)) {
                TCanvas c((std::string("c_pdf_") + h1->GetName()).c_str(), h1->GetTitle(), 900, 700);
                h1->Draw("HIST");
                c.SaveAs((outDir / (std::string(h1->GetName()) + ".pdf")).string().c_str());
            } else if (auto *h2 = dynamic_cast<TH2 *>(obj)) {
                TCanvas c((std::string("c_pdf_") + h2->GetName()).c_str(), h2->GetTitle(), 900, 700);
                h2->Draw("COLZ");
                c.SaveAs((outDir / (std::string(h2->GetName()) + ".pdf")).string().c_str());
            }
        }
    }
};

} // namespace UnifiedAnalysis

#endif // CHECKS_ENGINE_H
