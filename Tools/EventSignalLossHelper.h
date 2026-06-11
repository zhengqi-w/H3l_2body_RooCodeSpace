#ifndef EVENTSIGNALLOSS_HELPER_H
#define EVENTSIGNALLOSS_HELPER_H

#include <TError.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2.h>
#include <TH2D.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace EventSignalLossHelper {

struct EventLossResult {
	std::vector<double> centLow;
	std::vector<double> centHigh;
	std::vector<double> bLow;
	std::vector<double> bHigh;

	std::vector<double> impactValue;
	std::vector<double> impactError;

	std::vector<double> multiplicityValue;
	std::vector<double> multiplicityError;

	std::vector<double> eventSplittingValue;
	std::vector<double> eventSplittingError;

	TH1D *hImpact = nullptr;
	TH1D *hMultiplicity = nullptr;
	TH1D *hEventSplitting = nullptr;

	void Clear() {
		if (hImpact) {
			delete hImpact;
			hImpact = nullptr;
		}
		if (hMultiplicity) {
			delete hMultiplicity;
			hMultiplicity = nullptr;
		}
		if (hEventSplitting) {
			delete hEventSplitting;
			hEventSplitting = nullptr;
		}
		centLow.clear();
		centHigh.clear();
		bLow.clear();
		bHigh.clear();
		impactValue.clear();
		impactError.clear();
		multiplicityValue.clear();
		multiplicityError.clear();
		eventSplittingValue.clear();
		eventSplittingError.clear();
	}
};

struct SignalLossResult {
	std::vector<double> centLow;
	std::vector<double> centHigh;
	std::vector<double> bLow;
	std::vector<double> bHigh;

	std::vector<TH1D*> impact_pt_per_cent;
	std::vector<TH1D*> multiplicity_pt_per_cent;

	// Backward-compatible aliases to the default method used in the workflow.
	std::vector<TH1D*> signal_loss_pt_per_cent;
	std::vector<TH1D*> signal_loss_pt_per_cent_matter;
	std::vector<TH1D*> signal_loss_pt_per_cent_antimatter;

	void Clear() {
		auto clearOwnedVec = [](std::vector<TH1D*> &vec) {
			for (auto &ptr : vec) {
				if (ptr) {
					delete ptr;
					ptr = nullptr;
				}
			}
			vec.clear();
		};
		clearOwnedVec(impact_pt_per_cent);
		clearOwnedVec(multiplicity_pt_per_cent);
		signal_loss_pt_per_cent.clear();
		signal_loss_pt_per_cent_matter.clear();
		signal_loss_pt_per_cent_antimatter.clear();
		centLow.clear();
		centHigh.clear();
		bLow.clear();
		bHigh.clear();
	}
};

inline TH1 *RequireTH1(TFile &f, const std::string &path) {
	auto *obj = f.Get(path.c_str());
	auto *h = dynamic_cast<TH1 *>(obj);
	if (!h) {
		throw std::runtime_error("EventSignalLossHelper: missing TH1 at path: " + path);
	}
	return h;
}

inline TH2 *RequireTH2(TFile &f, const std::string &path) {
	auto *obj = f.Get(path.c_str());
	auto *h = dynamic_cast<TH2 *>(obj);
	if (!h) {
		throw std::runtime_error("EventSignalLossHelper: missing TH2 at path: " + path);
	}
	return h;
}

inline std::vector<double> BuildImpactParameterEdgesFromCentrality(
	TH1 *hImpactParamGen,
	const std::vector<double> &centBins,
	double bMaxPhys = 14.2) {
	if (!hImpactParamGen) {
		throw std::runtime_error("BuildImpactParameterEdgesFromCentrality: null hImpactParamGen");
	}
	if (centBins.size() < 2) {
		throw std::runtime_error("BuildImpactParameterEdgesFromCentrality: centBins needs at least 2 edges");
	}

	std::unique_ptr<TH1> hTrunc(static_cast<TH1 *>(hImpactParamGen->Clone("hImpactParamGen_trunc_tmp")));
	hTrunc->SetDirectory(nullptr);

	const int binMaxPhys = hTrunc->GetXaxis()->FindBin(bMaxPhys - 1e-6);
	for (int i = binMaxPhys + 1; i <= hTrunc->GetNbinsX(); ++i) {
		hTrunc->SetBinContent(i, 0.0);
		hTrunc->SetBinError(i, 0.0);
	}

	auto hCDF = std::unique_ptr<TH1>(hTrunc->GetCumulative());
	hCDF->SetDirectory(nullptr);
	const double cdfNorm = hCDF->GetBinContent(hCDF->GetNbinsX());
	if (cdfNorm <= 0.0) {
		throw std::runtime_error("BuildImpactParameterEdgesFromCentrality: cumulative integral is zero");
	}
	hCDF->Scale(1.0 / cdfNorm);

	std::vector<double> bEdges;
	bEdges.reserve(centBins.size());
	for (double cent : centBins) {
		const double frac = cent / 100.0;
		double b = hCDF->GetXaxis()->GetBinLowEdge(1);
		if (frac >= 1.0) {
			b = hCDF->GetXaxis()->GetBinUpEdge(binMaxPhys);
		} else if (frac > 0.0) {
			const int binIdx = hCDF->FindFirstBinAbove(frac);
			if (binIdx > 0) {
				b = hCDF->GetXaxis()->GetBinLowEdge(binIdx);
			} else {
				b = hCDF->GetXaxis()->GetBinUpEdge(binMaxPhys);
			}
		}
		bEdges.push_back(b);
	}
	return bEdges;
}

inline std::pair<double, double> RatioWithError(double numerator, double numeratorErr,
											   double denominator, double denominatorErr) {
	if (denominator <= 0.0) return {0.0, 0.0};
	const double ratio = numerator / denominator;
	const double relNum2 = (numerator > 0.0) ? (numeratorErr * numeratorErr) / (numerator * numerator) : 0.0;
	const double relDen2 = (denominatorErr * denominatorErr) / (denominator * denominator);
	return {ratio, ratio * std::sqrt(relNum2 + relDen2)};
}

inline std::pair<double, double> WeightedIntegralOverY(TH2 *h,
													  int xBinMin,
													  int xBinMax,
													  TH1 *weights) {
	if (!h || !weights) return {0.0, 0.0};
	const int nYMatch = std::min(h->GetNbinsY(), weights->GetNbinsX());
	double sum = 0.0;
	double err2 = 0.0;
	for (int iy = 1; iy <= nYMatch; ++iy) {
		const double w = weights->GetBinContent(iy);
		if (w == 0.0) continue;
		for (int ix = xBinMin; ix <= xBinMax; ++ix) {
			const double v = h->GetBinContent(ix, iy);
			const double e = h->GetBinError(ix, iy);
			sum += v * w;
			err2 += (e * w) * (e * w);
		}
	}
	return {sum, std::sqrt(std::max(0.0, err2))};
}

inline std::pair<double, double> Integral2DWithError(TH2 *h,
													 int xBinMin,
													 int xBinMax,
													 int yBinMin,
													 int yBinMax) {
	if (!h) return {0.0, 0.0};
	double sum = 0.0;
	double err2 = 0.0;
	for (int ix = xBinMin; ix <= xBinMax; ++ix) {
		for (int iy = yBinMin; iy <= yBinMax; ++iy) {
			const double v = h->GetBinContent(ix, iy);
			const double e = h->GetBinError(ix, iy);
			sum += v;
			err2 += e * e;
		}
	}
	return {sum, std::sqrt(std::max(0.0, err2))};
}

inline EventLossResult ComputeEventLoss(
	const std::string &rootFilePath,
	const std::vector<double> &centBins,
	double bMaxPhys = 14.2,
	const std::string &impactParamHistPath = "hyper-reco-task/QAEvent/McColAll/hImpactParamGen",
	const std::string &impactRecoHistPath = "hyper-reco-task/QAEvent/McColPassedEvSel/hImpactParamGenOneReco",
	const std::string &centVsMultPath = "hyper-reco-task/QAEvent/McColPassedEvSel/hGenCentralityColvsMultiplicityGenEta08",
	const std::string &genRecoEventsPath = "hyper-reco-task/QAEvent/hGenEventsNchEta08",
	const std::string &genOneRecoCentralityPath = "hyper-reco-task/QAEvent/McColPassedEvSel/hGenOneRecoCentrality",
	const std::string &recoCentralityPath = "hyper-reco-task/QAEvent/McColAll/hRecoCentrality") {
	if (centBins.size() < 2) {
		throw std::runtime_error("ComputeEventLoss: centBins needs at least 2 edges");
	}
	for (size_t i = 1; i < centBins.size(); ++i) {
		if (centBins[i] <= centBins[i - 1]) {
			throw std::runtime_error("ComputeEventLoss: centBins must be strictly increasing");
		}
	}

	std::unique_ptr<TFile> f(TFile::Open(rootFilePath.c_str(), "READ"));
	if (!f || f->IsZombie()) {
		throw std::runtime_error("ComputeEventLoss: cannot open file: " + rootFilePath);
	}

	auto *hImpactParamGen = RequireTH1(*f, impactParamHistPath);
	auto *hImpactParamReco = RequireTH1(*f, impactRecoHistPath);
	auto *hCentVsMult = RequireTH2(*f, centVsMultPath);
	auto *hGenRecoEvents = RequireTH2(*f, genRecoEventsPath);
	auto *hGenOneRecoCent = RequireTH1(*f, genOneRecoCentralityPath);
	auto *hRecoCent = RequireTH1(*f, recoCentralityPath);

	const auto bEdges = BuildImpactParameterEdgesFromCentrality(hImpactParamGen, centBins, bMaxPhys);

	EventLossResult res;
	const int nCent = static_cast<int>(centBins.size()) - 1;
	res.centLow.reserve(nCent);
	res.centHigh.reserve(nCent);
	res.bLow.reserve(nCent);
	res.bHigh.reserve(nCent);
	res.impactValue.reserve(nCent);
	res.impactError.reserve(nCent);
	res.multiplicityValue.reserve(nCent);
	res.multiplicityError.reserve(nCent);
	res.eventSplittingValue.reserve(nCent);
	res.eventSplittingError.reserve(nCent);

	res.hImpact = new TH1D("hEventLossImpact", "Event loss (ImpactParameter);Centrality (%);Event loss", nCent, centBins.data());
	res.hImpact->SetDirectory(nullptr);
	res.hImpact->Sumw2();

	res.hMultiplicity = new TH1D("hEventLossMultiplicity", "Event loss (Multiplicity);Centrality (%);Event loss", nCent, centBins.data());
	res.hMultiplicity->SetDirectory(nullptr);
	res.hMultiplicity->Sumw2();

	res.hEventSplitting = new TH1D("hEventSplitting", "Event splitting;Centrality (%);Event splitting", nCent, centBins.data());
	res.hEventSplitting->SetDirectory(nullptr);
	res.hEventSplitting->Sumw2();

	for (int iCent = 0; iCent < nCent; ++iCent) {
		const double cLow = centBins[iCent];
		const double cHigh = centBins[iCent + 1];
		const double bLow = bEdges[iCent];
		const double bHigh = bEdges[iCent + 1];

		res.centLow.push_back(cLow);
		res.centHigh.push_back(cHigh);
		res.bLow.push_back(bLow);
		res.bHigh.push_back(bHigh);

		const int cBinMin = hGenOneRecoCent->GetXaxis()->FindBin(cLow + 1e-6);
		const int cBinMax = hGenOneRecoCent->GetXaxis()->FindBin(cHigh - 1e-6);
		const int cRecoBinMin = hRecoCent->GetXaxis()->FindBin(cLow + 1e-6);
		const int cRecoBinMax = hRecoCent->GetXaxis()->FindBin(cHigh - 1e-6);
		double nGenOneRecoErr = 0.0;
		double nRecoCentErr = 0.0;
		const double nGenOneReco = hGenOneRecoCent->IntegralAndError(cBinMin, cBinMax, nGenOneRecoErr);
		const double nRecoCent = hRecoCent->IntegralAndError(cRecoBinMin, cRecoBinMax, nRecoCentErr);
		const auto eventSplitRatio = RatioWithError(nGenOneReco, nGenOneRecoErr, nRecoCent, nRecoCentErr);
		const double eventSplitting = eventSplitRatio.first > 0.0 ? eventSplitRatio.first : 1.0;
		const double eventSplittingErr = eventSplitRatio.first > 0.0 ? eventSplitRatio.second : 0.0;
		res.eventSplittingValue.push_back(eventSplitting);
		res.eventSplittingError.push_back(eventSplittingErr);
		res.hEventSplitting->SetBinContent(iCent + 1, eventSplitting);
		res.hEventSplitting->SetBinError(iCent + 1, eventSplittingErr);

		// Method 1: impact parameter, using the event-level reco/gen ratio.
		const int bBinMin = hImpactParamGen->GetXaxis()->FindBin(bLow + 1e-6);
		const int bBinMax = hImpactParamGen->GetXaxis()->FindBin(bHigh - 1e-6);
		double nBeforeErr = 0.0;
		double nAfterErr = 0.0;
		const double nBefore = hImpactParamGen->IntegralAndError(bBinMin, bBinMax, nBeforeErr);
		const double nAfter = hImpactParamReco->IntegralAndError(bBinMin, bBinMax, nAfterErr);
		const auto impactRatio = RatioWithError(nAfter, nAfterErr, nBefore, nBeforeErr);
		const double effImpact = impactRatio.first;
		const double errImpact = impactRatio.second;

		res.impactValue.push_back(effImpact);
		res.impactError.push_back(errImpact);
		res.hImpact->SetBinContent(iCent + 1, effImpact);
		res.hImpact->SetBinError(iCent + 1, errImpact);

		// Method 2: multiplicity-weighted method
		const int xBinMinCent = hCentVsMult->GetXaxis()->FindBin(cLow + 1e-6);
		const int xBinMaxCent = hCentVsMult->GetXaxis()->FindBin(cHigh - 1e-6);

		auto hMult = std::unique_ptr<TH1D>(hCentVsMult->ProjectionY(Form("hMult_tmp_%d", iCent), xBinMinCent, xBinMaxCent));
		hMult->SetDirectory(nullptr);

		const int nXMatch = std::min(hMult->GetNbinsX(), hGenRecoEvents->GetNbinsX());
		if (hMult->GetNbinsX() != hGenRecoEvents->GetNbinsX()) {
			Warning("EventSignalLossHelper", "Multiplicity and hGenEventsNchEta08 x-bin counts differ (%d vs %d), using min=%d",
					hMult->GetNbinsX(), hGenRecoEvents->GetNbinsX(), nXMatch);
		}

		const int nY = hGenRecoEvents->GetNbinsY();
		std::vector<double> sumY(static_cast<size_t>(nY) + 1, 0.0);
		std::vector<double> err2Y(static_cast<size_t>(nY) + 1, 0.0);
		for (int ix = 1; ix <= nXMatch; ++ix) {
			const double w = hMult->GetBinContent(ix);
			for (int iy = 1; iy <= nY; ++iy) {
				const double v = hGenRecoEvents->GetBinContent(ix, iy);
				const double e = hGenRecoEvents->GetBinError(ix, iy);
				sumY[static_cast<size_t>(iy)] += v * w;
				err2Y[static_cast<size_t>(iy)] += (e * w) * (e * w);
			}
		}

		const double nGen = (nY >= 1) ? sumY[1] : 0.0;
		const double eGen = (nY >= 1) ? std::sqrt(err2Y[1]) : 0.0;
		const double nReco = (nY >= 2) ? sumY[2] : 0.0;
		const double eReco = (nY >= 2) ? std::sqrt(err2Y[2]) : 0.0;

		double effMult = 0.0;
		double errMult = 0.0;
		if (nGen > 0.0) {
			effMult = nReco / nGen;
			const double relReco2 = (nReco > 0.0) ? (eReco * eReco) / (nReco * nReco) : 0.0;
			const double relGen2 = (eGen * eGen) / (nGen * nGen);
			errMult = effMult * std::sqrt(relReco2 + relGen2);
		}

		res.multiplicityValue.push_back(effMult);
		res.multiplicityError.push_back(errMult);
		res.hMultiplicity->SetBinContent(iCent + 1, effMult);
		res.hMultiplicity->SetBinError(iCent + 1, errMult);
	}

	return res;
}

inline SignalLossResult ComputeSignalLossCenPt(
	const std::string &rootFilePath,
	const std::vector<double> &centBins,
	const std::vector<std::vector<double>> &ptBinsPerCent,
	double bMaxPhys = 14.2,
	const std::string &impactParamHistPath = "hyper-reco-task/QAEvent/McColAll/hImpactParamGen",
	const std::string &centVsMultPath = "hyper-reco-task/QAEvent/McColPassedEvSel/hGenCentralityColvsMultiplicityGenEta08",
	const std::string &multBeforePath = "hyper-reco-task/QAEvent/McCol3HL/hGen3HLvsMultiplicityGenEta08BeforeEvtSel",
	const std::string &multAfterPath = "hyper-reco-task/QAEvent/McCol3HL/hGen3HLvsMultiplicityGenEta08AfterSel",
	const std::string &impactBeforePath = "hyper-reco-task/QAEvent/McCol3HL/hGen3HLvsImpactParameterBeforeEvtSel",
	const std::string &impactAfterPath = "hyper-reco-task/QAEvent/McCol3HL/hGen3HLvsImpactParameterAfterSel") {
	if (centBins.size() < 2) {
		throw std::runtime_error("ComputeSignalLossCenPt: centBins needs at least 2 edges");
	}
	if (ptBinsPerCent.size() != centBins.size() - 1) {
		throw std::runtime_error("ComputeSignalLossCenPt: ptBinsPerCent size must match nCentBins");
	}
	for (size_t i = 1; i < centBins.size(); ++i) {
		if (centBins[i] <= centBins[i - 1]) {
			throw std::runtime_error("ComputeSignalLossCenPt: centBins must be strictly increasing");
		}
	}
	for (size_t i = 0; i < ptBinsPerCent.size(); ++i) {
		if (ptBinsPerCent[i].size() < 2) {
			throw std::runtime_error("ComputeSignalLossCenPt: ptBinsPerCent[" + std::to_string(i) + "] has < 2 edges");
		}
	}

	std::unique_ptr<TFile> f(TFile::Open(rootFilePath.c_str(), "READ"));
	if (!f || f->IsZombie()) {
		throw std::runtime_error("ComputeSignalLossCenPt: cannot open file: " + rootFilePath);
	}

	auto *hImpactParamGen = RequireTH1(*f, impactParamHistPath);
	auto *hCentVsMult = RequireTH2(*f, centVsMultPath);
	auto *hMultBefore = RequireTH2(*f, multBeforePath);
	auto *hMultAfter = RequireTH2(*f, multAfterPath);
	auto *hImpactBefore = RequireTH2(*f, impactBeforePath);
	auto *hImpactAfter = RequireTH2(*f, impactAfterPath);

	const auto bEdges = BuildImpactParameterEdgesFromCentrality(hImpactParamGen, centBins, bMaxPhys);

	SignalLossResult res;
	const int nCent = static_cast<int>(centBins.size()) - 1;
	res.centLow.reserve(nCent);
	res.centHigh.reserve(nCent);
	res.bLow.reserve(nCent);
	res.bHigh.reserve(nCent);
	res.impact_pt_per_cent.assign(nCent, nullptr);
	res.multiplicity_pt_per_cent.assign(nCent, nullptr);
	res.signal_loss_pt_per_cent.assign(nCent, nullptr);
	res.signal_loss_pt_per_cent_matter.assign(nCent, nullptr);
	res.signal_loss_pt_per_cent_antimatter.assign(nCent, nullptr);

	for (int iCent = 0; iCent < nCent; ++iCent) {
		const double cLow = centBins[iCent];
		const double cHigh = centBins[iCent + 1];
		const double bLow = bEdges[iCent];
		const double bHigh = bEdges[iCent + 1];
		const auto &ptEdges = ptBinsPerCent[static_cast<size_t>(iCent)];

		res.centLow.push_back(cLow);
		res.centHigh.push_back(cHigh);
		res.bLow.push_back(bLow);
		res.bHigh.push_back(bHigh);

		auto hImpact = std::make_unique<TH1D>(Form("h_signal_loss_impact_pt_centbin_%d", iCent),
											  Form("Signal loss impact %.0f-%.0f%%;#it{p}_{T} (GeV/#it{c});Signal loss", cLow, cHigh),
											  static_cast<int>(ptEdges.size()) - 1,
											  ptEdges.data());
		auto hMult = std::make_unique<TH1D>(Form("h_signal_loss_multiplicity_pt_centbin_%d", iCent),
											Form("Signal loss multiplicity %.0f-%.0f%%;#it{p}_{T} (GeV/#it{c});Signal loss", cLow, cHigh),
											static_cast<int>(ptEdges.size()) - 1,
											ptEdges.data());
		hImpact->SetDirectory(nullptr);
		hMult->SetDirectory(nullptr);
		hImpact->Sumw2();
		hMult->Sumw2();

		const int xBinMinCent = hCentVsMult->GetXaxis()->FindBin(cLow + 1e-6);
		const int xBinMaxCent = hCentVsMult->GetXaxis()->FindBin(cHigh - 1e-6);
		auto hMultWeight = std::unique_ptr<TH1D>(hCentVsMult->ProjectionY(Form("hSignalMultWeight_tmp_%d", iCent),
																		   xBinMinCent,
																		   xBinMaxCent));
		hMultWeight->SetDirectory(nullptr);

		const int yBinMinB = hImpactBefore->GetYaxis()->FindBin(bLow + 1e-6);
		const int yBinMaxB = hImpactBefore->GetYaxis()->FindBin(bHigh - 1e-6);

		for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
			const double ptLow = ptEdges[ip];
			const double ptHigh = ptEdges[ip + 1];
			const int hBin = static_cast<int>(ip + 1);

			const int xBinMinMult = hMultBefore->GetXaxis()->FindBin(ptLow + 1e-6);
			const int xBinMaxMult = hMultBefore->GetXaxis()->FindBin(ptHigh - 1e-6);
			const auto multBefore = WeightedIntegralOverY(hMultBefore, xBinMinMult, xBinMaxMult, hMultWeight.get());
			const auto multAfter = WeightedIntegralOverY(hMultAfter, xBinMinMult, xBinMaxMult, hMultWeight.get());
			const auto multRatio = RatioWithError(multAfter.first, multAfter.second, multBefore.first, multBefore.second);
			hMult->SetBinContent(hBin, multRatio.first);
			hMult->SetBinError(hBin, multRatio.second);

			const int xBinMinImpact = hImpactBefore->GetXaxis()->FindBin(ptLow + 1e-6);
			const int xBinMaxImpact = hImpactBefore->GetXaxis()->FindBin(ptHigh - 1e-6);
			const auto impactBefore = Integral2DWithError(hImpactBefore, xBinMinImpact, xBinMaxImpact, yBinMinB, yBinMaxB);
			const auto impactAfter = Integral2DWithError(hImpactAfter, xBinMinImpact, xBinMaxImpact, yBinMinB, yBinMaxB);
			const auto impactRatio = RatioWithError(impactAfter.first, impactAfter.second, impactBefore.first, impactBefore.second);
			hImpact->SetBinContent(hBin, impactRatio.first);
			hImpact->SetBinError(hBin, impactRatio.second);
		}

		res.impact_pt_per_cent[static_cast<size_t>(iCent)] = hImpact.release();
		res.multiplicity_pt_per_cent[static_cast<size_t>(iCent)] = hMult.release();
		res.signal_loss_pt_per_cent[static_cast<size_t>(iCent)] = res.multiplicity_pt_per_cent[static_cast<size_t>(iCent)];
		res.signal_loss_pt_per_cent_matter[static_cast<size_t>(iCent)] = res.signal_loss_pt_per_cent[static_cast<size_t>(iCent)];
		res.signal_loss_pt_per_cent_antimatter[static_cast<size_t>(iCent)] = res.signal_loss_pt_per_cent[static_cast<size_t>(iCent)];
	}

	return res;
}

} // namespace EventSignalLossHelper

#endif // EVENTSIGNALLOSS_HELPER_H
