#include <TCanvas.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TMath.h>

#include <Math/MinimizerOptions.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../Tools/GeneralHelper.hpp"
#include "../include/AliPWGFunc.cxx"

namespace {

constexpr double kMass = 2.991;
constexpr double kIntegralMin = 0.0;
constexpr double kIntegralMaxForInfinity = 100.0;

struct Variant {
    std::string name;
    double betaMin;
    double betaMax;
    double tempMin;
    double tempMax;
    double nMin;
    double nMax;
    double fixedBeta;
    double fixedTemp;
    double fixedN;
    bool fixBeta;
    bool fixTemp;
    bool fixN;
    bool useSavedNominal;
    bool onlyMergedPeripheral;
};

struct Row {
    std::string variant;
    std::string cen;
    double cenMin;
    double cenMax;
    double measured;
    double low;
    double high;
    double total;
    double chi2ndf;
    double beta;
    double temp;
    double n;
    double norm;
};

std::vector<AliPWGFunc *> gOwners;

std::vector<double> ReadDoubleArray(const GeneralHelper::Json &j)
{
    std::vector<double> out;
    if (!j.is_array()) return out;
    out.reserve(j.size());
    for (const auto &v : j) {
        if (v.is_number()) out.push_back(v.get<double>());
    }
    return out;
}

struct MultiplicityPoint {
    double cenMin{0.0};
    double cenMax{0.0};
    double value{0.0};
    double error{0.0};
};

std::vector<MultiplicityPoint> LoadMultiplicityPoints(const char *configPath)
{
    std::vector<MultiplicityPoint> out;
    try {
        const auto cfg = GeneralHelper::LoadJsonFile(configPath);
        const auto common = cfg.value("common", GeneralHelper::Json::object());
        const auto binning = common.value("binning", GeneralHelper::Json::object());
        const auto cenBins = ReadDoubleArray(binning.value("cen_bins", GeneralHelper::Json::array()));
        const auto centers = ReadDoubleArray(binning.value("related_multiplicity_center", GeneralHelper::Json::array()));
        const auto errors = ReadDoubleArray(binning.value("related_multiplicity_uncertainty", GeneralHelper::Json::array()));
        const size_t n = std::min(cenBins.size() > 0 ? cenBins.size() - 1 : 0, centers.size());
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const double err = (i < errors.size()) ? errors[i] : 0.0;
            out.push_back(MultiplicityPoint{cenBins[i], cenBins[i + 1], centers[i], err});
        }
    } catch (const std::exception &e) {
        std::cerr << "LoadMultiplicityPoints failed: " << e.what() << std::endl;
    }
    return out;
}

const MultiplicityPoint *FindMultiplicityPoint(const std::vector<MultiplicityPoint> &points,
                                               double cenMin,
                                               double cenMax)
{
    for (const auto &p : points) {
        if (std::abs(p.cenMin - cenMin) < 1e-6 && std::abs(p.cenMax - cenMax) < 1e-6) {
            return &p;
        }
    }
    return nullptr;
}

struct He3BGBWConstraint {
    bool valid{false};
    double beta{0.0};
    double temp{0.0};
};

He3BGBWConstraint LoadHe3BGBW50To90(const char *he3SpectrumPath)
{
    He3BGBWConstraint out;
    if (!he3SpectrumPath || std::string(he3SpectrumPath).empty()) return out;

    TFile fHe3(he3SpectrumPath, "READ");
    if (fHe3.IsZombie()) {
        std::cerr << "Warning: cannot open He3 BGBW file " << he3SpectrumPath << std::endl;
        return out;
    }

    auto *func = dynamic_cast<TF1 *>(fHe3.Get("BGBW/BlastWave_He3_50_90"));
    if (!func) {
        std::cerr << "Warning: missing BGBW/BlastWave_He3_50_90 in " << he3SpectrumPath << std::endl;
        return out;
    }

    out.beta = func->GetParameter(1);
    out.temp = func->GetParameter(2);
    out.valid = std::isfinite(out.beta) && std::isfinite(out.temp) && out.beta > 0.0 && out.temp > 0.0;
    if (out.valid) {
        std::cout << "Loaded He3 50-90 BGBW constraint: beta=" << out.beta
                  << ", T=" << out.temp << std::endl;
    }
    return out;
}

bool IsMergedPeriodSpectrum(const char *spectrumPath)
{
    if (!spectrumPath) return false;
    const std::string path = spectrumPath;
    return path.find("LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1") != std::string::npos;
}

bool IsMergedPeripheralBin(double cenLo, double cenHi)
{
    return (std::abs(cenLo - 60.0) < 1e-6 && std::abs(cenHi - 70.0) < 1e-6) ||
           (std::abs(cenLo - 70.0) < 1e-6 && std::abs(cenHi - 90.0) < 1e-6);
}

std::unique_ptr<TF1> MakeBGBW(const std::string &name, const Variant &v, double beta, double temp, double n)
{
    auto *helper = new AliPWGFunc();
    gOwners.push_back(helper);
    helper->SetVarType(AliPWGFunc::kdNdpt);
    TF1 *f = helper->GetBGBW(kMass, beta, temp, n, 500.0, name.c_str());
    if (!f) return nullptr;
    f->FixParameter(0, kMass);
    if (v.fixBeta) {
        f->FixParameter(1, v.fixedBeta);
    } else {
        f->SetParLimits(1, v.betaMin, v.betaMax);
    }
    if (v.fixTemp) {
        f->FixParameter(2, v.fixedTemp);
    } else {
        f->SetParLimits(2, v.tempMin, v.tempMax);
    }
    if (v.fixN) {
        f->FixParameter(3, v.fixedN);
    } else {
        f->SetParLimits(3, v.nMin, v.nMax);
    }
    f->SetParLimits(4, 1e-8, 1e10);
    f->SetRange(0.0, kIntegralMaxForInfinity);
    f->SetNpx(5000);
    return std::unique_ptr<TF1>(f);
}

void SeedNorm(TH1D *h, TF1 *f)
{
    if (!h || !f) return;
    const double oldNorm = f->GetParameter(4);
    f->SetParameter(4, 1.0);
    double num = 0.0;
    double den = 0.0;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        const double x = h->GetBinCenter(ib);
        const double y = h->GetBinContent(ib);
        double e = h->GetBinError(ib);
        if (e <= 0.0 || !std::isfinite(e)) e = std::max(1e-12, 0.1 * std::abs(y));
        const double g = f->Eval(x);
        if (!std::isfinite(g) || g <= 0.0) continue;
        const double w = 1.0 / (e * e);
        num += y * g * w;
        den += g * g * w;
    }
    double norm = (den > 0.0) ? (num / den) : oldNorm;
    if (!std::isfinite(norm) || norm <= 0.0) norm = oldNorm;
    f->SetParameter(4, std::clamp(norm, 1e-8, 1e10));
}

double HistIntegral(TH1D *h)
{
    double sum = 0.0;
    if (!h) return sum;
    for (int ib = 1; ib <= h->GetNbinsX(); ++ib) {
        sum += h->GetBinContent(ib) * h->GetXaxis()->GetBinWidth(ib);
    }
    return sum;
}

std::unique_ptr<TF1> FitBest(TH1D *h, const Variant &v, const std::string &tag)
{
    std::vector<double> betaSeeds{0.60, 0.72, 0.82};
    std::vector<double> tempSeeds{0.10, 0.14, 0.18};
    std::vector<double> nSeeds{0.45, 0.55, 0.65};
    if (v.fixBeta) betaSeeds = {v.fixedBeta};
    if (v.fixTemp) tempSeeds = {v.fixedTemp};
    if (v.fixN) nSeeds = {v.fixedN};

    std::unique_ptr<TF1> best;
    double bestScore = std::numeric_limits<double>::infinity();
    int iseed = 0;
    for (double beta : betaSeeds) {
        if (beta < v.betaMin || beta > v.betaMax) continue;
        for (double temp : tempSeeds) {
            if (temp < v.tempMin || temp > v.tempMax) continue;
            for (double n : nSeeds) {
                if (!v.fixN && (n < v.nMin || n > v.nMax)) continue;
                auto f = MakeBGBW(Form("f_%s_%s_seed%d", v.name.c_str(), tag.c_str(), iseed++), v, beta, temp, n);
                if (!f) continue;
                if (h->GetNbinsX() <= 3) {
                    f->FixParameter(1, beta);
                    if (!v.fixN) f->FixParameter(3, std::clamp(n, v.nMin, v.nMax));
                }
                SeedNorm(h, f.get());
                auto res = h->Fit(f.get(), "Q0RNS", "", h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
                const int status = static_cast<int>(res);
                const int ndf = f->GetNDF();
                const double chi2 = f->GetChisquare();
                const double chi2ndf = (ndf > 0 && std::isfinite(chi2)) ? chi2 / ndf : std::numeric_limits<double>::infinity();
                const double low = f->Integral(0.0, h->GetXaxis()->GetXmin());
                const double firstBin = h->GetBinContent(1) * h->GetXaxis()->GetBinWidth(1);
                const double guardPenalty = (firstBin > 0.0 && low > 2.0 * firstBin) ? 10.0 * (low / firstBin - 2.0) : 0.0;
                const double score = chi2ndf + guardPenalty + (status == 0 ? 0.0 : 0.2);
                if (std::isfinite(score) && score < bestScore) {
                    bestScore = score;
                    best = std::move(f);
                }
            }
        }
    }
    return best;
}

bool ParseCen(const std::string &name, double &lo, double &hi)
{
    if (name.rfind("cen_", 0) != 0) return false;
    std::string rest = name.substr(4);
    const auto pos = rest.find('_');
    if (pos == std::string::npos) return false;
    lo = std::stod(rest.substr(0, pos));
    hi = std::stod(rest.substr(pos + 1));
    return true;
}

} // namespace

void IntegratedYieldBWExtrapCheck(const char *spectrumPath =
                                      "ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_LHC24ar_pass3_LHC25_PbPb_pass1/bdt_spectrum/both/spectrum.root",
                                  const char *configPath =
                                      "ROOTWorkFlow/CodeSpace/configs/general_config.json",
                                  const char *he3SpectrumPath =
                                      "ROOTWorkFlow/Outputs/PlotingScrips/He3SpectrumBGBW/He3_spectrum.root")
{
    ROOT::Math::MinimizerOptions::SetDefaultMaxIterations(20000);
    ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);

    const auto he3BGBW = LoadHe3BGBW50To90(he3SpectrumPath);
    const bool isMergedPeriod = IsMergedPeriodSpectrum(spectrumPath);

    std::vector<Variant> variants = {
        {"saved_nominal", 0.2, 0.9, 0.10, 0.60, 0.01, 5.0, 0.0, 0.0, 0.0, false, false, false, true, false},
        {"n_0p45", 0.2, 0.9, 0.10, 0.60, 0.01, 5.0, 0.0, 0.0, 0.45, false, false, true, false, false},
        {"n_0p55", 0.2, 0.9, 0.10, 0.60, 0.01, 5.0, 0.0, 0.0, 0.55, false, false, true, false, false},
        {"central_tight", 0.55, 0.86, 0.10, 0.18, 0.35, 0.65, 0.0, 0.0, 0.0, false, false, false, false, false}
    };
    if (he3BGBW.valid) {
        variants.push_back({"he3_50_90_fixed_T_beta",
                            he3BGBW.beta, he3BGBW.beta,
                            he3BGBW.temp, he3BGBW.temp,
                            0.01, 5.0,
                            he3BGBW.beta, he3BGBW.temp, 0.0,
                            true, true, false, false, true});
    }

    TFile f(spectrumPath, "READ");
    if (f.IsZombie()) {
        std::cerr << "Cannot open " << spectrumPath << std::endl;
        return;
    }
    const auto multiplicityPoints = LoadMultiplicityPoints(configPath);
    if (multiplicityPoints.empty()) {
        std::cerr << "Warning: no multiplicity points loaded from " << configPath << std::endl;
    } else {
        std::cout << "Loaded multiplicity points from " << configPath << std::endl;
        for (const auto &p : multiplicityPoints) {
            std::cout << "  cen " << p.cenMin << "-" << p.cenMax
                      << ": " << p.value << " +/- " << p.error << std::endl;
        }
    }

    std::filesystem::path outDir = "ROOTWorkFlow/Outputs/StandAloneChecks/IntegratedYieldBWExtrapCheck";
    std::filesystem::path qaDir = outDir / "BlastWaveFitQA";
    std::filesystem::create_directories(outDir);
    std::filesystem::create_directories(qaDir);
    std::ofstream csv(outDir / "bw_extrap_scan.csv");
    csv << "variant,centrality_min,centrality_max,measured,low_extrap_0_2,high_extrap_8_inf,total,chi2ndf,beta,T,n,norm\n";

    std::vector<Row> rows;
    TIter next(f.GetListOfKeys());
    TKey *key = nullptr;
    while ((key = dynamic_cast<TKey *>(next()))) {
        const std::string cenName = key->GetName();
        double cenLo = 0.0;
        double cenHi = 0.0;
        if (!ParseCen(cenName, cenLo, cenHi)) continue;
        const bool useMergedPeripheralConstraint = isMergedPeriod && IsMergedPeripheralBin(cenLo, cenHi);
        auto *dir = dynamic_cast<TDirectory *>(f.Get(cenName.c_str()));
        if (!dir) continue;
        auto *h = dynamic_cast<TH1D *>(dir->Get("h_final_spectrum_stat"));
        if (!h) continue;
        auto *intDir = dynamic_cast<TDirectory *>(dir->Get("integral_yield"));
        const double measured = HistIntegral(h);
        const double xMin = h->GetXaxis()->GetXmin();
        const double xMax = h->GetXaxis()->GetXmax();
        std::vector<std::unique_ptr<TF1>> ownedFitsForQa;
        std::vector<std::pair<std::string, TF1 *>> fitsForQa;

        for (const auto &v : variants) {
            if (v.onlyMergedPeripheral && !useMergedPeripheralConstraint) continue;
            TF1 *savedFit = nullptr;
            std::unique_ptr<TF1> ownedFit;
            if (v.useSavedNominal) {
                savedFit = intDir ? dynamic_cast<TF1 *>(intDir->Get("f_integral_nominal")) : nullptr;
                if (!savedFit) continue;
            } else {
                ownedFit = FitBest(h, v, cenName);
                if (!ownedFit) continue;
                savedFit = ownedFit.get();
                ownedFitsForQa.push_back(std::move(ownedFit));
                savedFit = ownedFitsForQa.back().get();
            }
            const double low = savedFit->Integral(kIntegralMin, xMin);
            const double high = savedFit->Integral(xMax, kIntegralMaxForInfinity);
            const double total = measured + low + high;
            const double chi2ndf = savedFit->GetNDF() > 0 ? savedFit->GetChisquare() / savedFit->GetNDF() : 0.0;
            Row row{v.name, cenName, cenLo, cenHi, measured, low, high, total, chi2ndf,
                    savedFit->GetParameter(1), savedFit->GetParameter(2), savedFit->GetParameter(3), savedFit->GetParameter(4)};
            rows.push_back(row);
            csv << row.variant << "," << row.cenMin << "," << row.cenMax << ","
                << std::setprecision(12) << row.measured << "," << row.low << ","
                << row.high << "," << row.total << "," << row.chi2ndf << ","
                << row.beta << "," << row.temp << "," << row.n << "," << row.norm << "\n";
            fitsForQa.push_back({v.name, savedFit});
        }

        if (!fitsForQa.empty()) {
            TCanvas cqa(("c_bw_fit_qa_" + cenName).c_str(), "", 900, 700);
            cqa.SetTicks(1, 1);
            cqa.SetLogy();
            auto *hDraw = dynamic_cast<TH1D *>(h->Clone(("h_qa_" + cenName).c_str()));
            hDraw->SetDirectory(nullptr);
            hDraw->SetMarkerStyle(20);
            hDraw->SetMarkerSize(1.0);
            hDraw->SetLineColor(kBlack);
            hDraw->SetMarkerColor(kBlack);
            hDraw->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
            hDraw->GetYaxis()->SetTitle("d^{2}N/(dyd#it{p}_{T})");
            double yMin = std::numeric_limits<double>::infinity();
            double yMax = 0.0;
            for (int ib = 1; ib <= hDraw->GetNbinsX(); ++ib) {
                const double y = hDraw->GetBinContent(ib);
                const double e = hDraw->GetBinError(ib);
                if (y > 0.0) yMin = std::min(yMin, std::max(1e-16, y - e));
                yMax = std::max(yMax, y + e);
            }
            if (!std::isfinite(yMin)) yMin = 1e-10;
            hDraw->SetMinimum(yMin * 0.35);
            hDraw->SetMaximum(yMax * 5.0);
            hDraw->Draw("E");

            TLegend leg(0.52, 0.56, 0.88, 0.88);
            leg.SetBorderSize(0);
            leg.SetFillStyle(0);
            leg.AddEntry(hDraw, "Corrected spectrum", "lep");
            const std::vector<int> fitColors = {kRed + 1, kAzure + 1, kGreen + 2, kOrange + 7, kMagenta + 2,
                                                kViolet + 1, kCyan + 2, kBlack};
            int style = 1;
            int ifit = 0;
            for (auto &entry : fitsForQa) {
                TF1 *fit = entry.second;
                if (!fit) continue;
                fit->SetRange(0.0, kIntegralMaxForInfinity);
                const int color = fitColors[ifit % fitColors.size()];
                fit->SetLineColor(color);
                fit->SetLineStyle(style);
                fit->SetLineWidth(2);
                fit->Draw("SAME");
                const double chi2ndf = fit->GetNDF() > 0 ? fit->GetChisquare() / fit->GetNDF() : 0.0;
                leg.AddEntry(fit, Form("%s (#chi^{2}/ndf=%.2f)", entry.first.c_str(), chi2ndf), "l");
                ++style;
                ++ifit;
            }
            leg.Draw();
            TLatex latex;
            latex.SetNDC();
            latex.SetTextSize(0.038);
            latex.DrawLatex(0.16, 0.91, Form("Centrality %.0f-%.0f%%", cenLo, cenHi));
            latex.SetTextSize(0.030);
            latex.DrawLatex(0.16, 0.86, "BGBW extrapolation QA, high-pT integral to 100 GeV/#it{c}");
            cqa.SaveAs((qaDir / ("bw_fit_qa_" + cenName + ".pdf")).c_str());
            delete hDraw;
        }
    }
    csv.close();

    TCanvas c("c_bw_total_vs_multiplicity_loglog", "", 900, 700);
    c.SetLogx();
    c.SetLogy();
    TLegend leg(0.15, 0.68, 0.48, 0.88);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    bool first = true;
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    double yPlotMin = std::numeric_limits<double>::infinity();
    double yPlotMax = 0.0;
    for (const auto &r : rows) {
        if (r.total <= 0.0) continue;
        if (!FindMultiplicityPoint(multiplicityPoints, r.cenMin, r.cenMax)) continue;
        yPlotMin = std::min(yPlotMin, r.total);
        yPlotMax = std::max(yPlotMax, r.total);
    }
    if (!std::isfinite(yPlotMin) || yPlotMin <= 0.0) yPlotMin = 1e-8;
    if (!(yPlotMax > yPlotMin)) yPlotMax = yPlotMin * 10.0;
    const std::vector<int> graphColors = {kBlack, kRed + 1, kAzure + 1, kGreen + 2, kMagenta + 2,
                                          kOrange + 7, kViolet + 1, kCyan + 2};
    int ivar = 0;
    for (const auto &v : variants) {
        auto g = std::make_unique<TGraphErrors>();
        for (const auto &r : rows) {
            if (r.variant != v.name) continue;
            const auto *mp = FindMultiplicityPoint(multiplicityPoints, r.cenMin, r.cenMax);
            if (mp && mp->value > 0.0) {
                const int ip = g->GetN();
                g->SetPoint(ip, mp->value, r.total);
                g->SetPointError(ip, mp->error, 0.0);
            }
        }
        const int color = graphColors[ivar % graphColors.size()];
        g->SetLineColor(color);
        g->SetMarkerColor(color);
        g->SetMarkerStyle(20 + (ivar % 10));
        g->SetLineWidth(2);
        if (first) {
            g->SetTitle(";#LTd#it{N}_{ch}/d#eta#GT;Integrated yield");
            g->SetMinimum(yPlotMin * 0.35);
            g->SetMaximum(yPlotMax * 2.5);
            g->Draw("AP");
            first = false;
        } else {
            g->Draw("P SAME");
        }
        leg.AddEntry(g.get(), v.name.c_str(), "p");
        graphs.push_back(std::move(g));
        ++ivar;
    }
    leg.Draw();
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.035);
    latex.DrawLatex(0.15, 0.91, "BGBW integrated-yield extrapolation scan");
    c.SaveAs((outDir / "bw_total_vs_multiplicity_loglog.pdf").c_str());
    c.SaveAs((outDir / "bw_total_vs_multiplicity.pdf").c_str());

    std::cout << "Wrote " << (outDir / "bw_extrap_scan.csv") << std::endl;
    std::cout << "Wrote " << (outDir / "bw_total_vs_multiplicity_loglog.pdf") << std::endl;
    std::cout << "Wrote fit QA PDFs under " << qaDir << std::endl;
}
