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

#include "../Tools/CtExtraction.h"
#include "../Tools/CtExtraction.cxx"
#include "../Tools/GeneralHelper.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void ProcessCtExtraction(const char *configPath = "../configs/ct_extraction.json",
                         bool enableImplicitMT = true) {
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
        std::string sigmaMcToDataRangeStr = get_string("sigma_range_path", "");
        bool easyMode = cfgJson.value("sigma_range_easy_mode", false);
        std::string rangeFile;
        if (!sigmaMcToDataRangeStr.empty()) {
            if (easyMode) {
                rangeFile = sigmaMcToDataRangeStr + "/combined/sigma_summary.txt";
            } else {
                rangeFile = sigmaMcToDataRangeStr + "/per_bin/pt_ratio_summary.txt";
            }
        }

        CtExtraction extractor(configPath);
        // *************************sigma range setting block start*************************
        // If a range file was provided and exists, parse it and set sigma ranges
        if (!rangeFile.empty() && std::filesystem::exists(rangeFile)) {
            std::ifstream rf(rangeFile);
            if (rf) {
                std::string line;
                // Read header to determine column indices
                std::vector<std::string> headerTokens;
                std::vector<std::tuple<double,double,double>> perBin; // ptmin, ptmax, ratio
                double combinedRatio = 1.0;
                while (std::getline(rf, line)) {
                    if (line.empty()) continue;
                    if (line.rfind("#", 0) == 0) {
                        // header or comment
                        // capture last header line that contains tokens
                        std::string hdr = line.substr(1);
                        std::istringstream hss(hdr);
                        headerTokens.clear();
                        std::string tok;
                        while (hss >> tok) headerTokens.push_back(tok);
                        continue;
                    }
                    // data line
                    std::istringstream ss(line);
                    std::vector<std::string> toks;
                    std::string t;
                    while (ss >> t) toks.push_back(t);
                    if (toks.empty()) continue;
                    if (easyMode) {
                        // need to find column 'ratio' in headerTokens
                        int idx = -1;
                        for (size_t i = 0; i < headerTokens.size(); ++i) {
                            if (headerTokens[i] == "ratio") { idx = static_cast<int>(i); break; }
                        }
                        if (idx >= 0 && idx < static_cast<int>(toks.size())) {
                            combinedRatio = std::stod(toks[idx]);
                        } else if (toks.size() >= 9) {
                            // fallback: ratio is commonly the 9th column in combined summary
                            combinedRatio = std::stod(toks[8]);
                        }
                        // only one combined line expected; break after reading
                        break;
                    } else {
                        // per-bin file: columns pt_min pt_max constant_ratio ...
                        int idxPtMin = -1, idxPtMax = -1, idxRatio = -1;
                        for (size_t i = 0; i < headerTokens.size(); ++i) {
                            if (headerTokens[i] == "pt_min") idxPtMin = static_cast<int>(i);
                            if (headerTokens[i] == "pt_max") idxPtMax = static_cast<int>(i);
                            if (headerTokens[i] == "constant_ratio") idxRatio = static_cast<int>(i);
                        }
                        double ptmin=0, ptmax=0, ratio=1.0;
                        if (idxPtMin >=0 && idxPtMin < static_cast<int>(toks.size())) ptmin = std::stod(toks[idxPtMin]);
                        else ptmin = std::stod(toks[0]);
                        if (idxPtMax >=0 && idxPtMax < static_cast<int>(toks.size())) ptmax = std::stod(toks[idxPtMax]);
                        else ptmax = std::stod(toks[1]);
                        if (idxRatio >=0 && idxRatio < static_cast<int>(toks.size())) ratio = std::stod(toks[idxRatio]);
                        else if (toks.size() >= 3) ratio = std::stod(toks[2]);
                        perBin.emplace_back(ptmin, ptmax, ratio);
                    }
                }

                // Build sigma ranges matching pt bins from config
                std::vector<double> ptBins = cfgJson.value("pt_bins", std::vector<double>{});
                if (ptBins.size() < 2) {
                    std::cerr << "[ProcessCtExtraction] Warning: pt_bins not found or too short in config; skipping sigma range set." << std::endl;
                } else {
                    size_t nPtBins = ptBins.size() - 1;
                    std::vector<std::vector<double>> ranges;
                    ranges.reserve(nPtBins);
                    for (size_t ib = 0; ib < nPtBins; ++ib) {
                        double ptmin = ptBins[ib];
                        double ptmax = ptBins[ib+1];
                        double useRatio = combinedRatio;
                        if (!easyMode) {
                            // find matching perBin entry
                            bool found = false;
                            for (const auto &tpl : perBin) {
                                double pmin, pmax, r;
                                std::tie(pmin, pmax, r) = tpl;
                                if (std::abs(pmin - ptmin) < 1e-6 && std::abs(pmax - ptmax) < 1e-6) {
                                    useRatio = r; found = true; break;
                                }
                            }
                            if (!found) {
                                std::cerr << "[ProcessCtExtraction] Warning: no ratio found for pt bin " << ptmin << "-" << ptmax << "; using 1.0" << std::endl;
                                useRatio = 1.0;
                            }
                        }
                        ranges.push_back(std::vector<double>{1.0, useRatio});
                    }
                    extractor.SetSigmaRangeMcToData(ranges);
                    std::cout << "[ProcessCtExtraction] Set sigma range mc->data for " << ranges.size() << " pt bins." << std::endl;
                    std::cout << "[ProcessCtExtraction] Set sigma range mc->data for " << ranges.size() << " pt bins." << std::endl;
                    std::cout << "[ProcessCtExtraction] Sigma ranges detail:" << std::endl;
                    for (size_t ib = 0; ib < ranges.size(); ++ib) {
                        std::cout << "  pt bin [" << ptBins[ib] << ", " << ptBins[ib + 1] << "] -> "
                                  << ranges[ib][0] << " to " << ranges[ib][1] << std::endl;
                    }
                }
            }
        }
        // *************************sigma range setting block end*************************

        extractor.Run();
        std::cout << "[ProcessCtExtraction] Completed successfully using config: "
                  << configPath << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "[ProcessCtExtraction] Error: " << ex.what() << std::endl;
        throw;
    }
}
