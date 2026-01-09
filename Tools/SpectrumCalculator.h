#pragma once

#include "BdtSpectrumHelper.h"

#include <ROOT/RDataFrame.hxx>
#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TString.h>
#include <TPaveText.h>

#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <cmath>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct FitResult {
    double signal{0.0};
    double signalErr{0.0};
    std::unique_ptr<RooPlot> frame;
    std::unique_ptr<RooPlot> frameMc;
    std::shared_ptr<RooRealVar> massAxis;
};

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
};

class SpectrumCalculator {
public:
    explicit SpectrumCalculator(Config cfg) : cfg_(std::move(cfg)) {}

    SpectrumResult Calculate(const std::vector<BinInput> &bins,
                             double nEvents,
                             const std::string &bkgFunc,
                             const std::string &sigFunc,
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

        for (size_t i = 0; i < bins.size(); ++i) {
            const auto &bin = bins[i];
            std::string cut = Form("model_output > %f", bin.wp.score);
            auto dataMass = bin.dfData->Filter(cut).Take<double>("fMassH3L");
            auto mcMass = bin.dfMc->Filter("fMassH3L>2.95 && fMassH3L<3.02").Take<double>("fMassH3L");
            if (dataMass->empty() || mcMass->empty()) {
                hRaw->SetBinContent(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinContent(static_cast<int>(i + 1), 0.0);
                continue;
            }

            FitResult fit = FitMass(*dataMass, *mcMass, bkgFunc, sigFunc);
            if (!std::isfinite(fit.signal) || !std::isfinite(fit.signalErr) || fit.signal < 0) {
                hRaw->SetBinContent(static_cast<int>(i + 1), 0.0);
                hRaw->SetBinError(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinContent(static_cast<int>(i + 1), 0.0);
                hCorr->SetBinError(static_cast<int>(i + 1), 0.0);
                hAcc->SetBinContent(static_cast<int>(i + 1), bin.acceptance);
                hAbso->SetBinContent(static_cast<int>(i + 1), bin.absorption);
                hBdt->SetBinContent(static_cast<int>(i + 1), bin.wp.efficiency);
                continue;
            }

            double bw = bin.ptMax - bin.ptMin;
            double acc = (bin.acceptance > 0) ? bin.acceptance : 1.0;
            double abso = (bin.absorption > 0) ? bin.absorption : 1.0;
            double corr = fit.signal / acc / abso / bin.wp.efficiency / bw / nEvents / cfg_.branchingRatio / cfg_.deltaRap;
            double relErr2 = 0.0;
            if (fit.signal > 0) relErr2 += (fit.signalErr / fit.signal) * (fit.signalErr / fit.signal);

            hRaw->SetBinContent(static_cast<int>(i + 1), fit.signal);
            hRaw->SetBinError(static_cast<int>(i + 1), fit.signalErr);
            hCorr->SetBinContent(static_cast<int>(i + 1), corr);
            hCorr->SetBinError(static_cast<int>(i + 1), corr * std::sqrt(relErr2));
            hAcc->SetBinContent(static_cast<int>(i + 1), bin.acceptance);
            hAbso->SetBinContent(static_cast<int>(i + 1), bin.absorption);
            hBdt->SetBinContent(static_cast<int>(i + 1), bin.wp.efficiency);

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
        cout << "SpectrumCalculator::Calculate completed." << endl; // DEBUG
        return out;
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
    FitResult FitMass(const std::vector<double> &dataMass, const std::vector<double> &mcMass,
                      const std::string &bkgFunc, const std::string &sigFunc) const {
        RooRealVar mass("m", "Mass(H3l)", cfg_.massMin, cfg_.massMax, "GeV/c^{2}");
        RooDataSet data("data", "data", RooArgSet(mass));
        int dataCounts = 0;
        for (double v : dataMass) {
            if (v < cfg_.massMin || v > cfg_.massMax) continue; // drop under/overflow to avoid pileup in first bin
            mass.setVal(v);
            data.add(RooArgSet(mass));
            ++dataCounts;
        }

        RooDataSet mc("mc", "mc", RooArgSet(mass));
        for (double v : mcMass) {
            if (v < cfg_.massMin || v > cfg_.massMax) continue;
            mass.setVal(v);
            mc.add(RooArgSet(mass));
        }

        RooRealVar muMc("muMc", "muMc", 2.991, 2.97, 3.01);
        RooRealVar sigmaMcVar("sigmaMc", "sigmaMc", 1.5e-3, 1.1e-3, 2.1e-3);
        RooRealVar a1McVar("a1Mc", "a1Mc", 1.5, 0.1, 10.0);
        RooRealVar a2McVar("a2Mc", "a2Mc", 1.5, 0.1, 10.0);
        RooRealVar n1McVar("n1Mc", "n1Mc", 5.0, 0.5, 30.0);
        RooRealVar n2McVar("n2Mc", "n2Mc", 5.0, 0.5, 30.0);
        RooAbsPdf *signalPdfMc = nullptr;
        if (sigFunc == "gauss") {
            signalPdfMc = new RooGaussian("sigMc", "sigMc", mass, muMc, sigmaMcVar);
        } else {
            signalPdfMc = new RooCrystalBall("sigMc", "sigMc", mass, muMc, sigmaMcVar, a1McVar, n1McVar, a2McVar, n2McVar);
        }

        signalPdfMc->fitTo(mc, RooFit::Range(2.97, 3.01), RooFit::Save(true), RooFit::PrintLevel(-1));
        if (sigFunc != "gauss") {
            a1McVar.setConstant(); a2McVar.setConstant(); n1McVar.setConstant(); n2McVar.setConstant();
        }
        const double sigmaMc = sigmaMcVar.getVal();
        const double sigmaErrMc = sigmaMcVar.getError();
        const double muMcVal = muMc.getVal();
        const double muErrMc = muMc.getError();
        double a1Mc = 0.0, a1ErrMc = 0.0, n1Mc = 0.0, n1ErrMc = 0.0, a2Mc = 0.0, a2ErrMc = 0.0, n2Mc = 0.0, n2ErrMc = 0.0;
        if (sigFunc != "gauss") {
            a1Mc = a1McVar.getVal();
            a1ErrMc = a1McVar.getError();
            n1Mc = n1McVar.getVal();
            n1ErrMc = n1McVar.getError();
            a2Mc = a2McVar.getVal();
            a2ErrMc = a2McVar.getError();
            n2Mc = n2McVar.getVal();
            n2ErrMc = n2McVar.getError();
        }
        const int nMcFloatParams = ((sigFunc == "gauss") ? 2 : 6);
        const int ndfMc = std::max(1, 80 - nMcFloatParams); // 80 bins used in frame below
        double chi2OverNdfMc = 0.0;

        // build independent signal pdf for data so MC frame is not altered by data fit
        RooRealVar mu("mu", "mu", 2.991, 2.985, 2.992);
        RooRealVar sigma("sigma", "sigma", sigmaMc, 1.1e-3, 3e-3);
        RooAbsPdf *signalPdf = nullptr;
        if (sigFunc == "gauss") {
            signalPdf = new RooGaussian("sig", "sig", mass, mu, sigma);
        } else {
            signalPdf = new RooCrystalBall("sig", "sig", mass, mu, sigma, a1McVar, n1McVar, a2McVar, n2McVar);
        }
        sigma.setRange(cfg_.sigmaRangeMcToData[0] * sigmaMc, cfg_.sigmaRangeMcToData[1] * sigmaMc);

        RooAbsPdf *bkg = nullptr;
        RooRealVar c0("c0", "c0", 0.0, -1.5, 1.5);
        RooRealVar c1("c1", "c1", 0.0, -1.5, 1.5);
        RooRealVar c2("c2", "c2", 0.0, -1.5, 1.5);
        if (bkgFunc == "pol1") {
            bkg = new RooChebychev("bkg", "bkg", mass, RooArgList(c0, c1));
        } else if (bkgFunc == "expo") {
            bkg = new RooExponential("bkg", "bkg", mass, c0);
        } else {
            bkg = new RooChebychev("bkg", "bkg", mass, RooArgList(c0, c1, c2));
        }

        const double nSigInit = std::max(1.0, 0.5 * static_cast<double>(dataCounts));
        const double nSigMax = std::max(500.0, 10.0 * static_cast<double>(dataCounts));
        const double nBkgInit = std::max(1.0, 0.5 * static_cast<double>(dataCounts));
        const double nBkgMax = std::max(500.0, 30.0 * static_cast<double>(dataCounts));
        RooRealVar nSig("nSig", "nSig", nSigInit, 0.0, nSigMax);
        RooRealVar nBkg("nBkg", "nBkg", nBkgInit, 0.0, nBkgMax);
        RooAddPdf model("model", "total_pdf", RooArgList(*signalPdf, *bkg), RooArgList(nSig, nBkg));
        model.fitTo(data, RooFit::Extended(true), RooFit::Save(true), RooFit::PrintLevel(-1));

        const double muData = mu.getVal();
        const double muErrData = mu.getError();
        const double sigmaData = sigma.getVal();
        const double sigmaErrData = sigma.getError();

        const double windowMin = muData - 3.0 * sigmaData;
        const double windowMax = muData + 3.0 * sigmaData;
        mass.setRange("sigWindow", windowMin, windowMax);
        std::unique_ptr<RooAbsReal> sigIntegral(signalPdf->createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
        std::unique_ptr<RooAbsReal> bkgIntegral(bkg->createIntegral(mass, RooFit::NormSet(mass), RooFit::Range("sigWindow")));
        const double sigFrac = sigIntegral ? sigIntegral->getVal() : 0.0;
        const double bkgFrac = bkgIntegral ? bkgIntegral->getVal() : 0.0;
        const double signalValue = nSig.getVal();
        const double signalValueErr = nSig.getError();
        const double signalCounts3s = signalValue * sigFrac;
        const double signalCounts3sErr = signalValueErr * sigFrac;
        const double bkgCounts3s = nBkg.getVal() * bkgFrac;
        const double bkgCounts3sErr = nBkg.getError() * bkgFrac;
        double significance = 0.0;
        double significanceErr = 0.0;
        bool vaildSignificance = bkgCounts3s + signalCounts3s > 0.0;
        if (vaildSignificance) {
            significance = signalCounts3s / std::sqrt(signalCounts3s + bkgCounts3s);
            const double dSdSig = std::sqrt(signalCounts3s + bkgCounts3s) - (signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
            const double dBdSig = -(signalCounts3s / (2.0 * std::sqrt(signalCounts3s + bkgCounts3s)));
            significanceErr = std::sqrt(std::pow(dSdSig * signalCounts3sErr, 2) + std::pow(dBdSig * bkgCounts3sErr, 2));
        }
        const int nDataFloatParams = ((bkgFunc == "pol1") ? 2 : (bkgFunc == "expo") ? 2 : 3) + ((sigFunc == "gauss") ? 2 : 6);
        const int ndfData = std::max(1, 40 - nDataFloatParams); // 40 bins used in frame below
        double chi2OverNdfData = 0.0;
        

        std::unique_ptr<RooPlot> frame;
        std::unique_ptr<RooPlot> frameMc;
        std::shared_ptr<RooRealVar> massHolder = std::make_shared<RooRealVar>(mass); // keep axis alive with frames
        //MC
        frameMc.reset(massHolder->frame(80));
        mc.plotOn(frameMc.get(), RooFit::Name("mc"));
        signalPdfMc->plotOn(frameMc.get(), RooFit::LineColor(kRed), RooFit::LineStyle(kDashed), RooFit::Name("sig_fit_mc"));
        chi2OverNdfMc = frameMc->chiSquare("sig_fit_mc", "mc", nMcFloatParams);
        auto textMC = std::make_unique<TPaveText>(0.6, 0.43, 0.9, 0.85, "NDC");
        textMC->SetBorderSize(0);
        textMC->SetFillStyle(0);
        textMC->SetTextAlign(12);
        textMC->AddText(Form("MC Fit Parameters:"));
        textMC->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muMcVal * 1e3, muErrMc * 1e3));
        textMC->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaMc * 1e3, sigmaErrMc * 1e3));
        if (sigFunc != "gauss") {
            textMC->AddText(Form(" #alpha_{l} = %.3f #pm %.3f", a1Mc, a1ErrMc));
            textMC->AddText(Form(" n_{l} = %.3f #pm %.3f", n1Mc, n1ErrMc));
            textMC->AddText(Form(" #alpha_{r} = %.3f #pm %.3f", a2Mc, a2ErrMc));
            textMC->AddText(Form(" n_{r} = %.3f #pm %.3f", n2Mc, n2ErrMc));
        }
        textMC->AddText(Form(" #chi^{2}/NDF = %.2f / %d", chi2OverNdfMc , ndfMc));
        frameMc->addObject(textMC.release());
        //Data
        frame.reset(massHolder->frame(40));
        data.plotOn(frame.get(), RooFit::Name("data"));
        model.plotOn(frame.get(), RooFit::Name("total"));
        model.plotOn(frame.get(), RooFit::Components(*bkg), RooFit::LineStyle(kDashed), RooFit::LineColor(kRed + 1), RooFit::Name("bkg"));
        model.plotOn(frame.get(), RooFit::Components(*signalPdf), RooFit::LineStyle(kDotted), RooFit::LineColor(kGreen + 2), RooFit::Name("sig"));
        chi2OverNdfData = frame->chiSquare("total", "data", nDataFloatParams);
        auto textData = std::make_unique<TPaveText>(0.58,0.36,0.88,0.88, "NDC");
        textData->SetBorderSize(0);
        textData->SetFillStyle(0);
        textData->SetTextAlign(12);
        textData->AddText(Form("Data Fit Parameters:"));
        textData->AddText(Form(" S (3#sigma) = %.1f #pm %.1f", signalCounts3s, signalCounts3sErr));
        textData->AddText(Form(" B (3#sigma) = %.1f #pm %.1f", bkgCounts3s, bkgCounts3sErr));
        if (vaildSignificance) {
            textData->AddText(Form(" S/#sqrt{S+B} (3#sigma) = %.2f #pm %.2f", significance, significanceErr));
        } else {
            textData->AddText(" Significance = N/A");
        }
        textData->AddText(Form(" #mu = %.3f #pm %.3f MeV/c^{2}", muData * 1e3, muErrData * 1e3));
        textData->AddText(Form(" #sigma = %.3f #pm %.3f MeV/c^{2}", sigmaData * 1e3, sigmaErrData * 1e3));
        textData->AddText(Form(" #chi^{2}/NDF = %.2f / %d", chi2OverNdfData , ndfData));
        frame->addObject(textData.release());

        FitResult out;
        out.signal = nSig.getVal();
        out.signalErr = nSig.getError();
        out.frame = std::move(frame);
        out.frameMc = std::move(frameMc);
        out.massAxis = std::move(massHolder);
        delete bkg;
        delete signalPdf;
        delete signalPdfMc;
        return out;
    }

    Config cfg_;
};
