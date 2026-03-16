#ifndef PROCESS_ONE_BIN_H
#define PROCESS_ONE_BIN_H

#include "../GeneralHelper.hpp"
#include "../binning/BinPlan.h"

#include <ROOT/RDataFrame.hxx>

#include <cctype>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace UnifiedAnalysis {

struct ProcessOneBinOptions {
    std::string dataTreeName{"O2hypcands"};
    std::string mcTreeName{"O2mchypcands"};

    std::string massColumn{"fMassH3L"};
    std::string bdtScoreColumn{"model_output"};

    std::string dataSelection;
    std::string mcSelection;
    std::string mcMassSelection{"fMassH3L>2.95 && fMassH3L<3.02"};

    bool useBdtCut{false};
    double bdtCut{0.0};
    std::string isMatter{"both"};
    bool enableQACapture{false};
    std::vector<std::string> qaColumns{"fDecRad", "fCt", "fAvgClusterSizeHe", "fMassH3L"};
    bool throwOnError{false};
};

struct ProcessOneBinResult {
    bool success{false};
    std::string label;
    std::string error;

    std::string snapshotDataPath;
    std::string snapshotMcPath;

    double bdtCutUsed{0.0};
    std::size_t nDataSelected{0};
    std::size_t nMcSelected{0};
    struct H3lQAVars {
        std::vector<double> values;
    };
    std::vector<std::string> qaColumns;
    std::vector<H3lQAVars> qaCandidates;

    GeneralHelper::MassFitResult massFit;
};

namespace detail {

inline std::string BuildFilterExpr(const std::vector<std::string> &parts) {
    std::string out;
    for (const auto &part : parts) {
        if (part.empty()) {
            continue;
        }
        if (!out.empty()) {
            out += " && ";
        }
        out += "(" + part + ")";
    }
    return out;
}

inline std::string FormatDoubleLiteral(double value) {
    std::ostringstream os;
    os << std::setprecision(17) << value;
    return os.str();
}

inline std::string BuildIsMatterFilter(const std::string &isMatter) {
    if (isMatter == "matter") return "fIsMatter > 0";
    if (isMatter == "antimatter") return "fIsMatter <= 0";
    return std::string();
}

inline std::string NormalizeTypeName(std::string typeName) {
    std::string out;
    out.reserve(typeName.size());
    for (char c : typeName) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

template <typename T>
inline std::vector<double> TakeColumnAsDoubleTyped(ROOT::RDF::RNode &node, const std::string &col) {
    auto values = node.Take<T>(col);
    std::vector<double> out;
    out.reserve(values->size());
    for (const auto &v : *values) {
        out.push_back(static_cast<double>(v));
    }
    return out;
}

inline std::vector<double> TakeNumericColumnAsDouble(ROOT::RDF::RNode &node, const std::string &col) {
    const std::string typeRaw = node.GetColumnType(col);
    const std::string type = NormalizeTypeName(typeRaw);

    if (type == "double" || type == "double_t") {
        return TakeColumnAsDoubleTyped<double>(node, col);
    }
    if (type == "float" || type == "float_t") {
        return TakeColumnAsDoubleTyped<float>(node, col);
    }
    if (type == "int" || type == "int_t") {
        return TakeColumnAsDoubleTyped<int>(node, col);
    }
    if (type == "unsignedint" || type == "uint_t") {
        return TakeColumnAsDoubleTyped<unsigned int>(node, col);
    }
    if (type == "short" || type == "short_t") {
        return TakeColumnAsDoubleTyped<short>(node, col);
    }
    if (type == "unsignedshort" || type == "ushort_t") {
        return TakeColumnAsDoubleTyped<unsigned short>(node, col);
    }
    if (type == "char" || type == "char_t") {
        return TakeColumnAsDoubleTyped<char>(node, col);
    }
    if (type == "unsignedchar" || type == "uchar_t") {
        return TakeColumnAsDoubleTyped<unsigned char>(node, col);
    }
    if (type == "long" || type == "long_t" || type == "long64_t") {
        return TakeColumnAsDoubleTyped<long>(node, col);
    }
    if (type == "unsignedlong" || type == "ulong_t" || type == "ulong64_t") {
        return TakeColumnAsDoubleTyped<unsigned long>(node, col);
    }
    if (type == "longlong" || type == "longlong_t") {
        return TakeColumnAsDoubleTyped<long long>(node, col);
    }
    if (type == "unsignedlonglong" || type == "ulonglong_t") {
        return TakeColumnAsDoubleTyped<unsigned long long>(node, col);
    }
    if (type == "bool") {
        return TakeColumnAsDoubleTyped<bool>(node, col);
    }

    throw std::runtime_error("Unsupported QA column type for '" + col + "': " + typeRaw);
}

inline std::shared_ptr<ROOT::RDataFrame> GetOrLoadSnapshotRdf(
    const std::string &path,
    const std::string &treeName,
    std::unordered_map<std::string, std::shared_ptr<ROOT::RDataFrame>> *rdfCache = nullptr) {

    if (path.empty()) {
        throw std::runtime_error("Snapshot path is empty");
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Snapshot file not found: " + path);
    }

    const std::string cacheKey = treeName + "@" + path;
    if (rdfCache) {
        auto it = rdfCache->find(cacheKey);
        if (it != rdfCache->end() && it->second) {
            return it->second;
        }
    }

    auto rdf = std::make_shared<ROOT::RDataFrame>(treeName, path);
    if (rdfCache) {
        (*rdfCache)[cacheKey] = rdf;
    }
    return rdf;
}

} // namespace detail

inline ProcessOneBinResult ProcessOneBin(
    const BinPlanItem &item,
    const ProcessOneBinOptions &opt,
    const GeneralHelper::MassFitConfig &fitCfg,
    const std::string &bkgFuncRaw,
    const std::string &sigFuncRaw,
    std::unordered_map<std::string, std::shared_ptr<ROOT::RDataFrame>> *rdfCache = nullptr) {

    ProcessOneBinResult out;
    out.label = item.label;
    out.snapshotDataPath = item.snapshotDataPath;
    out.snapshotMcPath = item.snapshotMcPath;
    out.bdtCutUsed = opt.bdtCut;

    try {
        auto dfData = detail::GetOrLoadSnapshotRdf(item.snapshotDataPath, opt.dataTreeName, rdfCache);
        auto dfMc = detail::GetOrLoadSnapshotRdf(item.snapshotMcPath, opt.mcTreeName, rdfCache);

        std::vector<std::string> dataSelections;
        if (!opt.dataSelection.empty()) {
            dataSelections.push_back(opt.dataSelection);
        }
        const std::string matterExpr = detail::BuildIsMatterFilter(opt.isMatter);
        if (!matterExpr.empty()) {
            dataSelections.push_back(matterExpr);
        }
        if (opt.useBdtCut) {
            dataSelections.push_back(opt.bdtScoreColumn + " > " + detail::FormatDoubleLiteral(opt.bdtCut));
        }
        const std::string dataFilter = detail::BuildFilterExpr(dataSelections);

        std::vector<std::string> mcSelections;
        if (!opt.mcSelection.empty()) {
            mcSelections.push_back(opt.mcSelection);
        }
        if (!opt.mcMassSelection.empty()) {
            mcSelections.push_back(opt.mcMassSelection);
        }
        if (!matterExpr.empty()) {
            mcSelections.push_back(matterExpr);
        }
        const std::string mcFilter = detail::BuildFilterExpr(mcSelections);

        ROOT::RDF::RNode dataNode(*dfData);
        if (!dataFilter.empty()) {
            dataNode = dataNode.Filter(dataFilter);
        }
        ROOT::RDF::RNode mcNode(*dfMc);
        if (!mcFilter.empty()) {
            mcNode = mcNode.Filter(mcFilter);
        }

        auto dataMass = dataNode.Take<double>(opt.massColumn);
        auto mcMass = mcNode.Take<double>(opt.massColumn);

        out.nDataSelected = dataMass->size();
        out.nMcSelected = mcMass->size();

        if (out.nDataSelected == 0 || out.nMcSelected == 0) {
            out.error = "Insufficient entries after selection (data=" + std::to_string(out.nDataSelected) +
                        ", mc=" + std::to_string(out.nMcSelected) + ")";
            return out;
        }

        out.massFit = GeneralHelper::FitMassSpectrum(*dataMass, *mcMass, fitCfg, bkgFuncRaw, sigFuncRaw);
        if (opt.enableQACapture) {
            std::vector<std::string> validCols;
            std::vector<std::vector<double>> colValues;
            validCols.reserve(opt.qaColumns.size());
            colValues.reserve(opt.qaColumns.size());
            for (const auto &col : opt.qaColumns) {
                if (col.empty()) continue;
                try {
                    colValues.emplace_back(detail::TakeNumericColumnAsDouble(dataNode, col));
                    validCols.push_back(col);
                } catch (const std::exception &) {
                    // Ignore missing or incompatible QA columns to keep analysis running.
                }
            }

            if (!validCols.empty()) {
                size_t nRows = colValues.front().size();
                for (size_t ic = 1; ic < colValues.size(); ++ic) {
                    nRows = std::min(nRows, colValues[ic].size());
                }
                out.qaColumns = std::move(validCols);
                out.qaCandidates.reserve(nRows);
                for (size_t ir = 0; ir < nRows; ++ir) {
                    ProcessOneBinResult::H3lQAVars row;
                    row.values.reserve(colValues.size());
                    for (auto &vals : colValues) {
                        row.values.push_back(vals[ir]);
                    }
                    out.qaCandidates.push_back(std::move(row));
                }
            }
        }
        out.success = true;
        return out;
    } catch (const std::exception &ex) {
        out.error = ex.what();
        if (opt.throwOnError) {
            throw;
        }
        return out;
    }
}

} // namespace UnifiedAnalysis

#endif // PROCESS_ONE_BIN_H
