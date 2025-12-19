#ifndef CT_EXTRACTION_H
#define CT_EXTRACTION_H

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

class TH1D;
class TFile;
class TCanvas;
class RooDataSet;
class RooArgSet;
class RooPlot;

namespace AcceptanceHelper {
struct AcceptanceResult;
}

/**
 * CtExtraction encapsulates the lifetime extraction pipeline using ROOT C++ tools.
 *
 * Features:
 *  - loads per-bin snapshots and corresponding XGBoost JSON models
 *  - applies BDT selections (with optional overrides) using TMVA::Experimental::RBDT
 *  - builds per-bin RooDataSets via RDataFrame::ForeachSlot and fits with DSCB + Chebyshev(2)
 *  - corrects raw yields with AcceptanceHelper results and stores QA outputs in a ROOT file.
 */
class CtExtraction {
public:
    explicit CtExtraction(const std::string &configPath);
    ~CtExtraction();

    CtExtraction(const CtExtraction&) = delete;
    CtExtraction& operator=(const CtExtraction&) = delete;

    /// Execute the full analysis chain. Throws std::runtime_error on fatal errors.
    void Run();

    /// Override the default BDT score threshold for a specific (pt, ct) bin.
    void SetBDTScoreOverride(double ptMin, double ptMax,
                             double ctMin, double ctMax,
                             double score);

    /// Clear all user-provided BDT overrides.
    void ClearBDTOverrides();

    // Setter function to provide external access to sigma_range_mc_to_data
    void SetSigmaRangeMcToData(const std::vector<std::vector<double>> &ranges) {
        fCfg.sigmaRangeMcToData = ranges;
    }

private:
    struct BinKey {
        double ptMin{};
        double ptMax{};
        double ctMin{};
        double ctMax{};

        std::string ToString() const;
        bool operator<(const BinKey &other) const;
    };

    struct WorkingPoint {
        double score{0.0};
        double efficiency{0.0};
        double significance{0.0};
    };

    struct Config {
        std::string dataSnapshotDir;
        std::string snapshotTreeName;
        std::string mcFile;
        std::string mcTreeName;
        std::string mcSnapshotDir;
        std::string mcSnapshotTreeName;
        std::string mcSnapshotPattern{"mc_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root"};
        std::string mcReweightFile{"/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root"};
        std::string mcReweightFunc{"BlastWave_H3L_10_30"};
        std::string workingPointFile;
        std::string outputDir;
        std::string outputFile;
        std::string trialSuffix;
        std::string isMatter{"both"};
        std::string massColumn{"fMassH3L"};
        std::string bdtScoreColumn{"model_output"};
        std::string snapshotPattern{"data_pt_%PTMIN%_%PTMAX%_ct_%CTMIN%_%CTMAX%.root"};
        std::vector<double> ptBins;
        std::vector<std::vector<double>> ctBins; // same length as ptBins-1
        std::vector<double> massRange{2.95, 3.05};
        int massBins{40};
        int mcMassBins{80};
        double minScoreShift{0.0};
        double minEntriesForFit{60.0};
        double chebRangeBuffer{0.0};
        std::string runPeriodLabel{"Run 3"};
        std::string collidingSystem{"Pb-Pb"};
        std::string sqrtsLabel{"#sqrt{s_{NN}}"};
        double collisionEnergyTeV{5.36};
        std::string dataSetLabel{"LHC23_PbPb_pass5"};
        bool alicePerformance{false};
        std::vector<std::vector<double>> sigmaRangeMcToData; // Stores sigma range for MC to data
    };

    struct BinComputationResult {
        BinKey key;
        double rawYield{0.0};
        double rawYieldErr{0.0};
        double acceptance{0.0};
        double acceptanceErr{0.0};
        double bdtEfficiency{0.0};
        double correctedYield{0.0};
        double correctedYieldErr{0.0};
        double bdtScore{0.0};
        double fittedMean{0.0};
        double fittedSigma{0.0};
        double fittedSigmaErr{0.0};
        double fittedSigmaMC{0.0};
        double fittedSigmaMCErr{0.0};
        double fittedChi2{0.0};
        int entriesAfterBDT{0};
        int entriesBeforeBDT{0};
        std::unique_ptr<RooPlot> mcMassFrame;
        std::unique_ptr<RooPlot> dataMassFrame;
        std::shared_ptr<RooRealVar> massAxis;
    };

    // Configuration and inputs
    Config fCfg;
    nlohmann::json fCfgJson;
    std::map<BinKey, WorkingPoint> fWorkingPoints;
    std::map<BinKey, double> fUserOverrides;

    // Acceptance cache per pt-bin (ownership via unique_ptr)
    std::vector<std::unique_ptr<TH1D>> fAcceptancePerPt;

    // Output file handle (created in Run)
    std::unique_ptr<TFile> fOutputFile;
    TFile *fInputMcFile{nullptr};

    // helpers
    void LoadConfig(const std::string &path);
    void ValidateConfig() const;
    void LoadWorkingPoints();
    void PrepareOutputFile();
    void BuildAcceptance();

    WorkingPoint GetWorkingPoint(const BinKey &key) const;
    double ResolveBDTScore(const BinKey &key) const;

    BinComputationResult ProcessOneBin(size_t ptIndex, size_t ctIndex);

    std::vector<double> CollectMassValues(const BinKey &key,
                                          double bdtScore,
                                          int &entriesBefore,
                                          int &entriesAfter) const;
    std::vector<double> CollectMCMasses(const BinKey &key) const;

    BinComputationResult FitSpectrum(const BinKey &key,
                                     const WorkingPoint &wp,
                                     const std::vector<double> &massValues,
                                     const std::vector<double> &mcMassValues,
                                     int entriesBefore,
                                     int entriesAfter,
                                     std::vector<double> sigmaRange) const;

    double LookupAcceptance(const BinKey &key, double &err) const;

    static std::string FormatEdge(double value);
    std::string ExpandPattern(const std::string &pattern, const BinKey &key) const;
    std::string BuildPath(const std::string &dir, const std::string &pattern, const BinKey &key) const;

};

#endif // CT_EXTRACTION_H
