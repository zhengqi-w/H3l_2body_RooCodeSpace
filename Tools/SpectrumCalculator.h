#pragma once

#include "BdtSpectrumHelper.h"
#include "GeneralHelper.hpp"

#include <ROOT/RDataFrame.hxx>
#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TString.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <cmath>
#include <cctype>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using FitResult = GeneralHelper::MassFitResult;

struct BinInput {
    double ptMin{0.0};
    double ptMax{0.0};
    std::shared_ptr<ROOT::RDataFrame> dfData;
    std::shared_ptr<ROOT::RDataFrame> dfMc;
    double acceptance{1.0};
    double absorption{1.0};
    WorkingPoint wp;
    std::string label;
};

struct SpectrumResult {
    std::unique_ptr<TH1D> hRaw;
    std::unique_ptr<TH1D> hCorr;
    std::unique_ptr<TH1D> hAcc;
    std::unique_ptr<TH1D> hAbso;
    std::unique_ptr<TH1D> hBdtEff;
    std::vector<std::unique_ptr<RooPlot>> frames;
    std::vector<std::unique_ptr<RooPlot>> framesMc;
    std::vector<std::unique_ptr<TCanvas>> canvases;
    std::vector<std::unique_ptr<TCanvas>> canvasesMc;
    std::vector<std::shared_ptr<RooRealVar>> massAxes;
    struct CorrectionRow {
        double ptMin{0.0};
        double ptMax{0.0};
        double raw{0.0};
        double rawErr{0.0};
        double acc{1.0};
        double abso{1.0};
        double bdtEff{1.0};
        double binWidth{0.0};
        double nEvents{0.0};
        double branchingRatio{1.0};
        double deltaRap{1.0};
        double matterRatio{1.0};
        double corrected{0.0};
        double correctedErr{0.0};
        std::string label;
        std::string tag;
    };
    std::vector<CorrectionRow> corrections;
};

class SpectrumCalculator {
public:
    explicit SpectrumCalculator(Config cfg) : cfg_(std::move(cfg)) {}

    SpectrumResult Calculate(const std::vector<BinInput> &bins,
                             double nEvents,
                             const std::string &bkgFunc,
                             const std::string &sigFunc,
                             const std::string &isMatter,
                             bool saveCanvas,
                             const std::string &frameSuffix = std::string()) const {
        if (bins.empty()) {
            throw std::runtime_error("No bins provided to SpectrumCalculator");
        }
        std::vector<double> edges;
        edges.reserve(bins.size() + 1);
        edges.push_back(bins.front().ptMin);
        for (const auto &b : bins) edges.push_back(b.ptMax);

        auto hRaw = std::make_unique<TH1D>("h_raw_counts", ";p_{T};N_{raw}", static_cast<int>(bins.size()), edges.data());
        auto hCorr = std::make_unique<TH1D>("h_corrected_counts", ";p_{T};#frac{1}{N_{ev}} dN/dy dp_{T}", static_cast<int>(bins.size()), edges.data());
        auto hAcc = std::make_unique<TH1D>("h_acceptance", ";p_{T};A\times#epsilon_{geo}", static_cast<int>(bins.size()), edges.data());
        auto hAbso = std::make_unique<TH1D>("h_absorption", ";p_{T};#epsilon_{abso}", static_cast<int>(bins.size()), edges.data());
        auto hBdt = std::make_unique<TH1D>("h_bdt_efficiency", ";p_{T};#epsilon_{BDT}", static_cast<int>(bins.size()), edges.data());
        hRaw->SetDirectory(nullptr);
        hCorr->SetDirectory(nullptr);
        hAcc->SetDirectory(nullptr);
        hAbso->SetDirectory(nullptr);
        hBdt->SetDirectory(nullptr);
        hRaw->SetStats(false);
        hCorr->SetStats(false);
        hAcc->SetStats(false);
        hAbso->SetStats(false);
        hBdt->SetStats(false);

        SpectrumResult out;
        std::vector<std::unique_ptr<RooPlot>> frames;
        std::vector<std::unique_ptr<RooPlot>> framesMc;
        frames.reserve(bins.size());
        framesMc.reserve(bins.size());
        out.massAxes.reserve(bins.size() * 2);

        auto makeSafe = [](std::string s) {
            for (char &c : s) {
                if (!std::isalnum(static_cast<unsigned char>(c))) {
                    c = '_';
                }
            }
            return s;
        };
        const std::string safeSuffix = makeSafe(frameSuffix);
        const double matterRatio = (isMatter == "both") ? 2.0 : 1.0;
        std::vector<typename SpectrumResult::CorrectionRow> corrRows;
        corrRows.reserve(bins.size());

        for (size_t i = 0; i < bins.size(); ++i) {
            const auto &bin = bins[i];
            std::string cut = Form("model_output > %f", bin.wp.score);
            const double bw = bin.ptMax - bin.ptMin;
            const double acc = (bin.acceptance > 0) ? bin.acceptance : 1.0;
            const double abso = (bin.absorption > 0) ? bin.absorption : 1.0;
            double corr = 0.0;
            double corrErr = 0.0;
            double rawVal = 0.0;
            double rawErr = 0.0;

            auto dataMass = bin.dfData->Filter(cut).Take<double>("fMassH3L");
            auto mcMass = bin.dfMc->Filter("fMassH3L>2.95 && fMassH3L<3.02").Take<double>("fMassH3L");
            if (dataMass->empty() || mcMass->empty()) {
                hRaw->SetBinContent(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinContent(static_cast<int>(i + 1), 0.0);
                SpectrumResult::CorrectionRow row;
                row.ptMin = bin.ptMin;
                row.ptMax = bin.ptMax;
                row.raw = 0.0;
                row.rawErr = 0.0;
                row.acc = acc;
                row.abso = abso;
                row.bdtEff = bin.wp.efficiency;
                row.binWidth = bw;
                row.nEvents = nEvents;
                row.branchingRatio = cfg_.branchingRatio;
                row.deltaRap = cfg_.deltaRap;
                row.matterRatio = matterRatio;
                row.corrected = 0.0;
                row.correctedErr = 0.0;
                row.label = bin.label;
                row.tag = safeSuffix;
                corrRows.push_back(std::move(row));
                continue;
            }

            FitResult fit = FitMassPublic(*dataMass, *mcMass, bkgFunc, sigFunc);
            if (!std::isfinite(fit.signal) || !std::isfinite(fit.signalErr) || fit.signal < 0) {
                hRaw->SetBinContent(static_cast<int>(i + 1), 0.0);
                hRaw->SetBinError(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinContent(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinError(static_cast<int>(i + 1), 0.0);
                hAcc->SetBinContent(static_cast<int>(i + 1), bin.acceptance);
                hAbso->SetBinContent(static_cast<int>(i + 1), bin.absorption);
                hBdt->SetBinContent(static_cast<int>(i + 1), bin.wp.efficiency);
                SpectrumResult::CorrectionRow row;
                row.ptMin = bin.ptMin;
                row.ptMax = bin.ptMax;
                row.raw = 0.0;
                row.rawErr = 0.0;
                row.acc = acc;
                row.abso = abso;
                row.bdtEff = bin.wp.efficiency;
                row.binWidth = bw;
                row.nEvents = nEvents;
                row.branchingRatio = cfg_.branchingRatio;
                row.deltaRap = cfg_.deltaRap;
                row.matterRatio = matterRatio;
                row.corrected = 0.0;
                row.correctedErr = 0.0;
                row.label = bin.label;
                row.tag = safeSuffix;
                corrRows.push_back(std::move(row));
                continue;
            }

            rawVal = fit.signal;
            rawErr = fit.signalErr;
            if (isMatter == "both") {
                corr = fit.signal / acc / abso / bin.wp.efficiency / bw / nEvents / cfg_.branchingRatio / cfg_.deltaRap / matterRatio;
                corrErr = fit.signalErr / acc / abso / bin.wp.efficiency / bw / nEvents / cfg_.branchingRatio / cfg_.deltaRap / matterRatio;
            } else {
                corr = fit.signal / acc / abso / bin.wp.efficiency / bw / nEvents / cfg_.branchingRatio / cfg_.deltaRap;
                corrErr = fit.signalErr / acc / abso / bin.wp.efficiency / bw / nEvents / cfg_.branchingRatio / cfg_.deltaRap;
            }

            hRaw->SetBinContent(static_cast<int>(i + 1), fit.signal);
            hRaw->SetBinError(static_cast<int>(i + 1), fit.signalErr);
            hCorr->SetBinContent(static_cast<int>(i + 1), corr);
            hCorr->SetBinError(static_cast<int>(i + 1), corrErr);
            hAcc->SetBinContent(static_cast<int>(i + 1), bin.acceptance);
            hAbso->SetBinContent(static_cast<int>(i + 1), bin.absorption);
            hBdt->SetBinContent(static_cast<int>(i + 1), bin.wp.efficiency);

            SpectrumResult::CorrectionRow row;
            row.ptMin = bin.ptMin;
            row.ptMax = bin.ptMax;
            row.raw = rawVal;
            row.rawErr = rawErr;
            row.acc = acc;
            row.abso = abso;
            row.bdtEff = bin.wp.efficiency;
            row.binWidth = bw;
            row.nEvents = nEvents;
            row.branchingRatio = cfg_.branchingRatio;
            row.deltaRap = cfg_.deltaRap;
            row.matterRatio = matterRatio;
            row.corrected = corr;
            row.correctedErr = corrErr;
            row.label = bin.label;
            row.tag = safeSuffix;
            corrRows.push_back(std::move(row));

            const std::string safeLabel = makeSafe(bin.label);
            if (fit.frame) {
                const std::string name = Form("data_frame_%s_%s", safeLabel.c_str(), safeSuffix.c_str());
                const std::string title = Form("Data Fit (%s) %s", bin.label.c_str(), frameSuffix.c_str());
                fit.frame->SetName(name.c_str());
                fit.frame->SetTitle(title.c_str());
                if (saveCanvas) {
                    auto canvas = MakeFrameCanvas(Form("data_canvas_%s", name.c_str()), fit.frame.get(), false);
                    out.canvases.push_back(std::move(canvas));
                }
                frames.push_back(std::move(fit.frame));
                if (fit.massAxis) out.massAxes.push_back(fit.massAxis);
            }
            if (fit.frameMc) {
                const std::string nameMc = Form("mc_frame_%s_%s", safeLabel.c_str(), safeSuffix.c_str());
                fit.frameMc->SetName(nameMc.c_str());
                fit.frameMc->SetTitle(Form("MC Fit (%s) %s", bin.label.c_str(), frameSuffix.c_str()));
                if (saveCanvas) {
                    auto canvasMc = MakeFrameCanvas(Form("mc_canvas_%s", nameMc.c_str()), fit.frameMc.get(), true);
                    out.canvasesMc.push_back(std::move(canvasMc));
                }
                framesMc.push_back(std::move(fit.frameMc));
                if (fit.massAxis) out.massAxes.push_back(fit.massAxis);
            }
        }

        out.hRaw = std::move(hRaw);
        out.hCorr = std::move(hCorr);
        out.hAcc = std::move(hAcc);
        out.hAbso = std::move(hAbso);
        out.hBdtEff = std::move(hBdt);
        out.frames = std::move(frames);
        out.framesMc = std::move(framesMc);
        out.corrections = std::move(corrRows);
        cout << "SpectrumCalculator::Calculate completed." << endl; // DEBUG
        return out;
    }

    FitResult FitMassPublic(const std::vector<double> &dataMass,
                            const std::vector<double> &mcMass,
                            const std::string &bkgFunc,
                            const std::string &sigFunc) const {
        GeneralHelper::MassFitConfig fitCfg;
        fitCfg.massMin = cfg_.massMin;
        fitCfg.massMax = cfg_.massMax;
        fitCfg.sigmaRangeMcToData = cfg_.sigmaRangeMcToData;
        return GeneralHelper::FitMassSpectrum(dataMass, mcMass, fitCfg, bkgFunc, sigFunc);
    }

    void RedrawFrameCanvas(TCanvas *canvas, RooPlot *frame, bool isMc) const {
        DrawFrameCanvas(canvas, frame, isMc);
    }

private:
    std::unique_ptr<TCanvas> MakeFrameCanvas(const std::string &canvasName, RooPlot *frame, bool isMc) const {
        if (!frame) return nullptr;
        auto canvas = std::make_unique<TCanvas>(canvasName.c_str(), canvasName.c_str(), 800, 600);
        DrawFrameCanvas(canvas.get(), frame, isMc);
        return canvas;
    }

    void DrawFrameCanvas(TCanvas *canvas, RooPlot *frame, bool isMc) const {
        if (!canvas || !frame) return;
        canvas->cd();
        canvas->Clear();
        frame->Draw();
        DrawLegend(frame, isMc);
        canvas->Modified();
        canvas->Update();
    }

    void DrawLegend(RooPlot *frame, bool isMc) const {
        if (!frame) return;
        const auto entries = isMc ? std::vector<std::tuple<const char *, const char *, const char *>>{
                                        {"mc", "MC", "l"},
                                        {"sig_fit_mc", "Signal (MC)", "l"}}
                                  : std::vector<std::tuple<const char *, const char *, const char *>>{
                                        {"data", "Data", "lep"},
                                        {"total", "Total fit", "l"},
                                        {"bkg", "Background", "l"},
                                        {"sig", "Signal", "l"}};
        TLegend *legend = new TLegend(0.14,0.50,0.50,0.70);
        legend->SetBorderSize(0);
        legend->SetFillStyle(0);
        legend->SetTextFont(42);
        bool added = false;
        for (const auto &[name, label, option] : entries) {
            if (auto *obj = FindPlotObject(frame, name)) {
                legend->AddEntry(obj, label, option);
                added = true;
            }
        }
        if (added) {
            legend->Draw();
        } else {
            delete legend;
        }
    }

    static TObject *FindPlotObject(RooPlot *frame, const char *name) {
        if (!frame || !name) return nullptr;
        return frame->findObject(name);
    }

    Config cfg_;
};
