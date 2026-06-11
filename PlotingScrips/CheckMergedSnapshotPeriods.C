#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct SnapshotRow {
    std::string label;
    std::string path;
    double entries{0.0};
    double periodCounts[3]{0.0, 0.0, 0.0};
    double modelCounts[3]{0.0, 0.0, 0.0};
    double bdtPassCounts[3]{0.0, 0.0, 0.0};
    double bestScore{0.0};
    bool hasPeriodIndex{false};
    bool hasModelOutput{false};
    bool hasBestScore{false};
};

std::string BaseNameNoExt(const std::string &path) {
    std::string base = gSystem->BaseName(path.c_str());
    if (base.rfind("data_", 0) == 0) base = base.substr(5);
    if (base.size() > 5 && base.substr(base.size() - 5) == ".root") base = base.substr(0, base.size() - 5);
    return base;
}

std::vector<std::string> ListDataSnapshots(const char *snapshotDir) {
    std::vector<std::string> out;
    TString cmd = Form("find '%s' -maxdepth 1 -type f -name 'data_*.root' | sort", snapshotDir);
    TString listing = gSystem->GetFromPipe(cmd.Data());
    std::unique_ptr<TObjArray> lines(listing.Tokenize("\n"));
    for (int i = 0; i < lines->GetEntriesFast(); ++i) {
        auto *obj = dynamic_cast<TObjString *>(lines->At(i));
        if (!obj) continue;
        TString s = obj->GetString();
        s = s.Strip(TString::kBoth);
        if (!s.IsNull()) out.emplace_back(s.Data());
    }
    return out;
}

void StyleHist(TH1 *h, Color_t color, Style_t marker) {
    if (!h) return;
    h->SetStats(false);
    h->SetLineColor(color);
    h->SetMarkerColor(color);
    h->SetMarkerStyle(marker);
    h->SetLineWidth(2);
}

bool LoadBestScore(const std::string &wpDir, const std::string &label, double &score) {
    double cenMin = 0.0, cenMax = 0.0, ptMin = 0.0, ptMax = 0.0;
    if (std::sscanf(label.c_str(), "cen_%lf_%lf_pt_%lf_%lf", &cenMin, &cenMax, &ptMin, &ptMax) != 4) {
        return false;
    }

    const std::string wpPath = wpDir + "/WorkingPoint_SpectrumTest.txt";
    std::ifstream in(wpPath);
    if (!in.is_open()) return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double c0 = 0.0, c1 = 0.0, p0 = 0.0, p1 = 0.0, s = 0.0, eff = 0.0, sig = 0.0;
        if (!(ss >> c0 >> c1 >> p0 >> p1 >> s >> eff >> sig)) continue;
        if (std::fabs(c0 - cenMin) < 1e-6 && std::fabs(c1 - cenMax) < 1e-6 &&
            std::fabs(p0 - ptMin) < 1e-6 && std::fabs(p1 - ptMax) < 1e-6) {
            score = s;
            return std::isfinite(score);
        }
    }
    return false;
}

bool ParseCenPtLabel(const std::string &label, double &cenMin, double &cenMax, double &ptMin, double &ptMax) {
    return std::sscanf(label.c_str(), "cen_%lf_%lf_pt_%lf_%lf", &cenMin, &cenMax, &ptMin, &ptMax) == 4;
}

bool MatchesCentrality(const std::string &path, double targetCenMin, double targetCenMax) {
    double cenMin = 0.0, cenMax = 0.0, ptMin = 0.0, ptMax = 0.0;
    if (!ParseCenPtLabel(BaseNameNoExt(path), cenMin, cenMax, ptMin, ptMax)) return false;
    return std::fabs(cenMin - targetCenMin) < 1e-6 && std::fabs(cenMax - targetCenMax) < 1e-6;
}

SnapshotRow InspectSnapshot(const std::string &path, const char *treeName, const std::string &wpDir) {
    SnapshotRow row;
    row.path = path;
    row.label = BaseNameNoExt(path);
    row.hasBestScore = LoadBestScore(wpDir, row.label, row.bestScore);
    if (!row.hasBestScore) row.bestScore = 0.0; // Same fallback as UnifiedTaskRunner when wp.found is false.

    std::unique_ptr<TFile> f(TFile::Open(path.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "[Warn] Cannot open " << path << std::endl;
        return row;
    }
    auto *tree = dynamic_cast<TTree *>(f->Get(treeName));
    if (!tree) {
        std::cerr << "[Warn] Missing tree " << treeName << " in " << path << std::endl;
        return row;
    }

    row.entries = static_cast<double>(tree->GetEntries());
    row.hasPeriodIndex = (tree->GetBranch("fPeriodIndex") != nullptr);
    row.hasModelOutput = (tree->GetBranch("model_output") != nullptr);
    if (!row.hasPeriodIndex) return row;

    TH1D hPeriod(Form("hPeriod_tmp_%p", tree), ";period;entries", 3, -0.5, 2.5);
    hPeriod.SetDirectory(gDirectory);
    tree->Project(hPeriod.GetName(), "fPeriodIndex", "");
    for (int ip = 0; ip < 3; ++ip) row.periodCounts[ip] = hPeriod.GetBinContent(ip + 1);

    if (row.hasModelOutput) {
        TH1D hModel(Form("hModel_tmp_%p", tree), ";period;entries with model output", 3, -0.5, 2.5);
        hModel.SetDirectory(gDirectory);
        tree->Project(hModel.GetName(), "fPeriodIndex", "model_output==model_output");
        for (int ip = 0; ip < 3; ++ip) row.modelCounts[ip] = hModel.GetBinContent(ip + 1);

        TH1D hPass(Form("hPass_tmp_%p", tree), ";period;entries passing BDT", 3, -0.5, 2.5);
        hPass.SetDirectory(gDirectory);
        tree->Project(hPass.GetName(), "fPeriodIndex", Form("model_output==model_output && model_output>%g", row.bestScore));
        for (int ip = 0; ip < 3; ++ip) row.bdtPassCounts[ip] = hPass.GetBinContent(ip + 1);
    }
    return row;
}

void SaveCsv(const std::vector<SnapshotRow> &rows, const std::string &csvPath) {
    std::ofstream out(csvPath);
    out << "label,path,total_entries,has_fPeriodIndex,has_model_output,"
        << "has_best_score,best_score,"
        << "period0_entries,period1_entries,period2_entries,"
        << "period0_fraction,period1_fraction,period2_fraction,"
        << "period0_model_entries,period1_model_entries,period2_model_entries,"
        << "period0_model_fraction,period1_model_fraction,period2_model_fraction,"
        << "period0_bdt_pass_entries,period1_bdt_pass_entries,period2_bdt_pass_entries,"
        << "period0_bdt_pass_fraction,period1_bdt_pass_fraction,period2_bdt_pass_fraction\n";
    for (const auto &r : rows) {
        double totalPeriod = r.periodCounts[0] + r.periodCounts[1] + r.periodCounts[2];
        out << r.label << ',' << r.path << ',' << r.entries << ','
            << static_cast<int>(r.hasPeriodIndex) << ',' << static_cast<int>(r.hasModelOutput) << ','
            << static_cast<int>(r.hasBestScore) << ',' << r.bestScore;
        for (double v : r.periodCounts) out << ',' << v;
        for (double v : r.periodCounts) out << ',' << ((totalPeriod > 0.0) ? v / totalPeriod : 0.0);
        for (double v : r.modelCounts) out << ',' << v;
        for (int ip = 0; ip < 3; ++ip) {
            out << ',' << ((r.periodCounts[ip] > 0.0) ? r.modelCounts[ip] / r.periodCounts[ip] : 0.0);
        }
        double passSum = r.bdtPassCounts[0] + r.bdtPassCounts[1] + r.bdtPassCounts[2];
        for (double v : r.bdtPassCounts) out << ',' << v;
        for (double v : r.bdtPassCounts) out << ',' << ((passSum > 0.0) ? v / passSum : 0.0);
        out << '\n';
    }
}
} // namespace

void CheckMergedSnapshotPeriods(
    const char *snapshotDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1",
    const char *outDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/PlotingScrips/PeriodMergedQA/SnapshotPeriodQA_70_90",
    const char *treeName = "O2hypcands",
    const char *wpDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/WorkingPoint",
    double targetCenMin = 70.0,
    double targetCenMax = 90.0,
    bool requireBestScore = true) {
    gSystem->mkdir(outDir, true);
    gStyle->SetOptStat(0);
    gStyle->SetPaintTextFormat(".2f");

    auto files = ListDataSnapshots(snapshotDir);
    if (files.empty()) {
        std::cerr << "ERROR: no data_*.root snapshots found in " << snapshotDir << std::endl;
        return;
    }
    std::vector<std::string> filteredFiles;
    filteredFiles.reserve(files.size());
    for (const auto &path : files) {
        if (MatchesCentrality(path, targetCenMin, targetCenMax)) filteredFiles.push_back(path);
    }
    files.swap(filteredFiles);
    if (files.empty()) {
        std::cerr << "ERROR: no snapshots found for centrality " << targetCenMin << "-" << targetCenMax
                  << " in " << snapshotDir << std::endl;
        return;
    }

    std::vector<SnapshotRow> rows;
    rows.reserve(files.size());
    for (const auto &path : files) {
        std::cout << "[Info] Inspect " << path << std::endl;
        auto row = InspectSnapshot(path, treeName, wpDir);
        if (requireBestScore && !row.hasBestScore) {
            std::cout << "[Info] Skip " << row.label << " because it has no best_score in WorkingPoint_SpectrumTest.txt" << std::endl;
            continue;
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        std::cerr << "ERROR: no snapshots left after requiring best_score from WorkingPoint_SpectrumTest.txt" << std::endl;
        return;
    }

    const std::string csvPath = std::string(outDir) + "/merged_snapshot_period_model_QA.csv";
    SaveCsv(rows, csvPath);

    double total[3]{0.0, 0.0, 0.0};
    double totalModel[3]{0.0, 0.0, 0.0};
    double totalPass[3]{0.0, 0.0, 0.0};
    for (const auto &r : rows) {
        for (int ip = 0; ip < 3; ++ip) {
            total[ip] += r.periodCounts[ip];
            totalModel[ip] += r.modelCounts[ip];
            totalPass[ip] += r.bdtPassCounts[ip];
        }
    }

    TH1D hTotal("h_total_period_entries", ";period;entries in merged data snapshots", 3, -0.5, 2.5);
    TH1D hModelFrac("h_total_model_fraction", ";period;entries with finite model_output / entries", 3, -0.5, 2.5);
    TH1D hPass("h_total_bdt_pass_entries", ";period;entries passing final BDT cut", 3, -0.5, 2.5);
    const char *periodLabels[3] = {"LHC23 pass5", "LHC24ar pass3", "LHC25 pass1"};
    for (int ip = 0; ip < 3; ++ip) {
        hTotal.GetXaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
        hModelFrac.GetXaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
        hPass.GetXaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
        hTotal.SetBinContent(ip + 1, total[ip]);
        hModelFrac.SetBinContent(ip + 1, (total[ip] > 0.0) ? totalModel[ip] / total[ip] : 0.0);
        hPass.SetBinContent(ip + 1, totalPass[ip]);
    }
    hTotal.SetFillColor(kAzure - 9);
    hModelFrac.SetFillColor(kGreen + 1);
    hPass.SetFillColor(kOrange + 1);

    TCanvas cTotal("c_total_period_entries", "total entries per period", 900, 700);
    cTotal.SetLeftMargin(0.13);
    hTotal.SetStats(false);
    hTotal.SetMaximum(std::max({total[0], total[1], total[2]}) * 1.35);
    hTotal.Draw("BAR TEXT0");
    cTotal.SaveAs((std::string(outDir) + "/total_entries_per_period.pdf").c_str());

    TCanvas cModel("c_total_model_fraction", "model output fraction per period", 900, 700);
    cModel.SetLeftMargin(0.13);
    hModelFrac.SetStats(false);
    hModelFrac.SetMinimum(0.0);
    hModelFrac.SetMaximum(1.08);
    hModelFrac.Draw("BAR TEXT0");
    cModel.SaveAs((std::string(outDir) + "/model_output_fraction_per_period.pdf").c_str());

    TCanvas cPass("c_total_bdt_pass_entries", "BDT-selected entries per period", 900, 700);
    cPass.SetLeftMargin(0.13);
    hPass.SetStats(false);
    hPass.SetMaximum(std::max({totalPass[0], totalPass[1], totalPass[2]}) * 1.35);
    hPass.Draw("BAR TEXT0");
    cPass.SaveAs((std::string(outDir) + "/bdt_selected_entries_per_period.pdf").c_str());

    TH2D hFrac("h_snapshot_period_fraction", ";snapshot bin;period;fraction in snapshot", rows.size(), 0.5, rows.size() + 0.5, 3, -0.5, 2.5);
    TH2D hRecognized("h_snapshot_model_fraction", ";snapshot bin;period;model_output finite fraction", rows.size(), 0.5, rows.size() + 0.5, 3, -0.5, 2.5);
    TH2D hPassFrac("h_snapshot_bdt_pass_fraction", ";snapshot bin;period;fraction after BDT cut", rows.size(), 0.5, rows.size() + 0.5, 3, -0.5, 2.5);
    hFrac.SetStats(false);
    hRecognized.SetStats(false);
    hPassFrac.SetStats(false);
    for (int ip = 0; ip < 3; ++ip) {
        hFrac.GetYaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
        hRecognized.GetYaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
        hPassFrac.GetYaxis()->SetBinLabel(ip + 1, periodLabels[ip]);
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &r = rows[i];
        const double sum = r.periodCounts[0] + r.periodCounts[1] + r.periodCounts[2];
        hFrac.GetXaxis()->SetBinLabel(static_cast<int>(i + 1), r.label.c_str());
        hRecognized.GetXaxis()->SetBinLabel(static_cast<int>(i + 1), r.label.c_str());
        hPassFrac.GetXaxis()->SetBinLabel(static_cast<int>(i + 1), r.label.c_str());
        const double passSum = r.bdtPassCounts[0] + r.bdtPassCounts[1] + r.bdtPassCounts[2];
        for (int ip = 0; ip < 3; ++ip) {
            hFrac.SetBinContent(static_cast<int>(i + 1), ip + 1, (sum > 0.0) ? r.periodCounts[ip] / sum : 0.0);
            hRecognized.SetBinContent(static_cast<int>(i + 1), ip + 1,
                                      (r.periodCounts[ip] > 0.0) ? r.modelCounts[ip] / r.periodCounts[ip] : 0.0);
            hPassFrac.SetBinContent(static_cast<int>(i + 1), ip + 1, (passSum > 0.0) ? r.bdtPassCounts[ip] / passSum : 0.0);
        }
    }

    TCanvas cFrac("c_snapshot_period_fraction", "period fraction by snapshot", 1800, 600);
    cFrac.SetLeftMargin(0.12);
    cFrac.SetBottomMargin(0.35);
    hFrac.GetXaxis()->LabelsOption("v");
    hFrac.GetZaxis()->SetRangeUser(0.0, 1.0);
    hFrac.Draw("COLZ TEXT");
    cFrac.SaveAs((std::string(outDir) + "/snapshot_period_fraction_heatmap.pdf").c_str());

    TCanvas cRecognized("c_snapshot_model_fraction", "model output fraction by snapshot", 1800, 600);
    cRecognized.SetLeftMargin(0.12);
    cRecognized.SetBottomMargin(0.35);
    hRecognized.GetXaxis()->LabelsOption("v");
    hRecognized.GetZaxis()->SetRangeUser(0.0, 1.0);
    hRecognized.Draw("COLZ TEXT");
    cRecognized.SaveAs((std::string(outDir) + "/snapshot_model_output_fraction_heatmap.pdf").c_str());

    TCanvas cPassFrac("c_snapshot_bdt_pass_fraction", "period fraction after BDT cut by snapshot", 1800, 600);
    cPassFrac.SetLeftMargin(0.12);
    cPassFrac.SetBottomMargin(0.35);
    hPassFrac.GetXaxis()->LabelsOption("v");
    hPassFrac.GetZaxis()->SetRangeUser(0.0, 1.0);
    hPassFrac.Draw("COLZ TEXT");
    cPassFrac.SaveAs((std::string(outDir) + "/snapshot_bdt_selected_period_fraction_heatmap.pdf").c_str());

    int missingWpBins = 0;
    for (const auto &r : rows) {
        if (!r.hasBestScore) ++missingWpBins;
    }

    std::cout << "\nSummary across " << rows.size() << " data snapshots for centrality "
              << targetCenMin << "-" << targetCenMax << ":" << std::endl;
    std::cout << "  missing WP bins=" << missingWpBins << " / " << rows.size() << std::endl;
    for (int ip = 0; ip < 3; ++ip) {
        std::cout << "  " << periodLabels[ip] << ": entries=" << total[ip]
                  << ", model_output finite fraction=" << ((total[ip] > 0.0) ? totalModel[ip] / total[ip] : 0.0)
                  << ", BDT-pass entries=" << totalPass[ip]
                  << std::endl;
    }
    std::cout << "Saved CSV: " << csvPath << std::endl;
    std::cout << "Saved plots in: " << outDir << std::endl;
}
