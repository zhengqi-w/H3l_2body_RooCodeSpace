// Plot fAvgClusterSizeHe distributions and correlations for data vs MC
#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TLegend.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TStyle.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "../Tools/GeneralHelper.hpp"

using namespace GeneralHelper;

void AverClusterSize() {
	const std::string dataFile = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s_HadronPID.root";
	const std::string mcFile   = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_CustomV0s.root";
	const std::string outDir  = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/Checks/AverClusterSize";

	SetDefaultStyle();
	EnableImplicitMTWithPreferredThreads();
	std::filesystem::create_directories(outDir);

	auto openFile = [](const std::string &path) {
		std::unique_ptr<TFile> f(TFile::Open(path.c_str(), "READ"));
		if (!f || f->IsZombie()) {
			std::cerr << "[Error] Cannot open file: " << path << "\n";
			return std::unique_ptr<TFile>(nullptr);
		}
		return f;
	};

	auto dataPtr = openFile(dataFile);
	auto mcPtr   = openFile(mcFile);
	if (!dataPtr || !mcPtr) {
		return;
	}

	TChain dataChain("O2hypcands");
	fillChainFromAO2D(dataChain, dataPtr.get());
	ROOT::RDataFrame dataRdf(dataChain);
	auto dataReady = CorrectAndConvertRDF(dataRdf, false, false, false);

	TChain mcChain("O2mchypcands");
	fillChainFromAO2D(mcChain, mcPtr.get());
	ROOT::RDataFrame mcRdf(mcChain);
	auto mcReady = CorrectAndConvertRDF(mcRdf, false, true, false).Filter("fIsReco > 0", "MC reco only");

	const int nBinsCl = 15;
	const double clMin = 0.0;
	const double clMax = 15.0;

	auto hDataClPtr = dataReady.Histo1D({"hDataCl", ";fAvgClusterSizeHe;Entries", nBinsCl, clMin, clMax}, "fAvgClusterSizeHe");
	auto hMcClPtr   = mcReady.Histo1D({"hMcCl", ";fAvgClusterSizeHe;Entries", nBinsCl, clMin, clMax}, "fAvgClusterSizeHe");
	std::unique_ptr<TH1D> hDataCl(static_cast<TH1D*>(hDataClPtr->Clone("hDataCl_clone")));
	std::unique_ptr<TH1D> hMcCl(static_cast<TH1D*>(hMcClPtr->Clone("hMcCl_clone")));
	hDataCl->SetDirectory(nullptr);
	hMcCl->SetDirectory(nullptr);
	std::unique_ptr<TH1D> hDataClSolo(static_cast<TH1D*>(hDataCl->Clone("hDataCl_solo")));
	std::unique_ptr<TH1D> hMcClSolo(static_cast<TH1D*>(hMcCl->Clone("hMcCl_solo")));

	hDataCl->SetLineColor(kBlack);
	hDataCl->SetMarkerColor(kBlack);
	hDataCl->SetMarkerStyle(20);
	hMcCl->SetLineColor(kRed + 1);
	hMcCl->SetMarkerColor(kRed + 1);
	hMcCl->SetMarkerStyle(24);
	const double dataYield = hDataCl->Integral();
	const double mcYield = hMcCl->Integral();
	if (dataYield > 0.0 && mcYield > 0.0) {
		hMcCl->Scale(dataYield / mcYield);
	}

	// Separate plots
	{
		TCanvas cData("cData", "fAvgClusterSizeHe Data", 800, 600);
		hDataClSolo->SetLineColor(kBlack);
		hDataClSolo->SetMarkerColor(kBlack);
		hDataClSolo->SetMarkerStyle(20);
		hDataClSolo->Draw("HIST");
		SaveCanvas(&cData, outDir + "/fAvgClusterSizeHe_Data.pdf");
	}
	{
		TCanvas cMc("cMc", "fAvgClusterSizeHe MC", 800, 600);
		hMcClSolo->SetLineColor(kRed + 1);
		hMcClSolo->SetMarkerColor(kRed + 1);
		hMcClSolo->SetMarkerStyle(24);
		hMcClSolo->Draw("HIST");
		SaveCanvas(&cMc, outDir + "/fAvgClusterSizeHe_MC.pdf");
	}

	TCanvas cCl("cCl", "fAvgClusterSizeHe Data vs MC", 800, 600);
	hDataCl->Draw("PE");
	hMcCl->Draw("HIST SAME");
	TLegend leg(0.58, 0.68, 0.88, 0.88);
	leg.SetBorderSize(0);
	leg.AddEntry(hDataCl.get(), "Data", "lep");
	leg.AddEntry(hMcCl.get(), "MC (fIsReco>0)", "l");
	leg.Draw();
	SaveCanvas(&cCl, outDir + "/fAvgClusterSizeHe_DataVsMC.pdf");

	auto make2D = [](ROOT::RDF::RNode rdf, const std::string &name, const std::string &title,
					 const std::pair<int, std::pair<double, double>> &xbins,
					 const std::pair<int, std::pair<double, double>> &ybins,
					 const std::string &xvar, const std::string &yvar) {
		auto hPtr = rdf.Histo2D({name.c_str(), title.c_str(), xbins.first, xbins.second.first, xbins.second.second,
													ybins.first, ybins.second.first, ybins.second.second}, xvar.c_str(), yvar.c_str());
		std::unique_ptr<TH2D> h(static_cast<TH2D*>(hPtr->Clone((name + "_clone").c_str())));
		h->SetDirectory(nullptr);
		return h;
	};

	auto draw2D = [&](TH2D *h, const std::string &title, const std::string &fname) {
		if (!h) return;
		TCanvas c("c2d", title.c_str(), 820, 700);
		c.SetRightMargin(0.12);
		h->SetStats(false);
		h->Draw("COLZ");
		SaveCanvas(&c, outDir + "/" + fname);
	};

	auto dataDecLen = make2D(dataReady, "hData_DecLen", ";fDecLen (cm);fAvgClusterSizeHe", {60, {0.0, 30.0}}, {60, {clMin, clMax}}, "fDecLen", "fAvgClusterSizeHe");
	auto mcDecLen   = make2D(mcReady, "hMc_DecLen", ";fDecLen (cm);fAvgClusterSizeHe", {60, {0.0, 30.0}}, {60, {clMin, clMax}}, "fDecLen", "fAvgClusterSizeHe");

	auto dataMom = make2D(dataReady, "hData_P", ";fP (GeV/c);fAvgClusterSizeHe", {60, {0.0, 10.0}}, {60, {clMin, clMax}}, "fP", "fAvgClusterSizeHe");
	auto mcMom   = make2D(mcReady, "hMc_P", ";fP (GeV/c);fAvgClusterSizeHe", {60, {0.0, 10.0}}, {60, {clMin, clMax}}, "fP", "fAvgClusterSizeHe");

	auto dataNSigma = make2D(dataReady, "hData_NSigmaHe", ";fNSigmaHe;fAvgClusterSizeHe", {60, {-8.0, 8.0}}, {60, {clMin, clMax}}, "fNSigmaHe", "fAvgClusterSizeHe");
	auto mcNSigma   = make2D(mcReady, "hMc_NSigmaHe", ";fNSigmaHe;fAvgClusterSizeHe", {60, {-8.0, 8.0}}, {60, {clMin, clMax}}, "fNSigmaHe", "fAvgClusterSizeHe");

	draw2D(dataDecLen.get(), "Data: fAvgClusterSizeHe vs fDecLen", "fAvgClusterSizeHe_vs_fDecLen_Data.pdf");
	draw2D(mcDecLen.get(),   "MC: fAvgClusterSizeHe vs fDecLen",   "fAvgClusterSizeHe_vs_fDecLen_MC.pdf");

	draw2D(dataMom.get(), "Data: fAvgClusterSizeHe vs fP", "fAvgClusterSizeHe_vs_fP_Data.pdf");
	draw2D(mcMom.get(),   "MC: fAvgClusterSizeHe vs fP",   "fAvgClusterSizeHe_vs_fP_MC.pdf");

	draw2D(dataNSigma.get(), "Data: fAvgClusterSizeHe vs fNSigmaHe", "fAvgClusterSizeHe_vs_fNSigmaHe_Data.pdf");
	draw2D(mcNSigma.get(),   "MC: fAvgClusterSizeHe vs fNSigmaHe",   "fAvgClusterSizeHe_vs_fNSigmaHe_MC.pdf");
}
