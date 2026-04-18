#include "BinPlanBuilder.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace UnifiedAnalysis {

namespace {

std::vector<double> ReadDoubleArray(const GeneralHelper::Json &j) {
    if (!j.is_array()) return {};
    std::vector<double> out;
    out.reserve(j.size());
    for (const auto &v : j) {
        if (!v.is_number()) continue;
        out.push_back(v.get<double>());
    }
    return out;
}

void AddEdge(std::vector<double> &edges, double value) {
    constexpr double eps = 1e-9;
    for (double x : edges) {
        if (std::abs(x - value) < eps) return;
    }
    edges.push_back(value);
}

} // namespace

BinPlan BinPlanBuilder::Build(const GeneralHelper::Json &cfg, const ModePolicy &policy) const {
    const auto analysis = cfg.value("analysis", GeneralHelper::Json::object());
    const auto modeProfiles = analysis.value("mode_profiles", GeneralHelper::Json::object());
    if (!modeProfiles.contains(policy.profileKey)) {
        throw std::runtime_error("analysis.mode_profiles missing key: " + policy.profileKey);
    }

    const auto profile = modeProfiles.at(policy.profileKey);
    const auto common = cfg.value("common", GeneralHelper::Json::object());
    const auto commonBinning = common.value("binning", GeneralHelper::Json::object());
    const auto commonPath = common.value("path", GeneralHelper::Json::object());
    const std::string snapshotDir = commonPath.value("snapshot_dir", std::string());

    BinPlan out;
    out.mode = policy.mode;

    if (policy.useCentrality && policy.usePt) {
        const auto cenEdges = ReadDoubleArray(commonBinning.value("cen_bins", GeneralHelper::Json::array()));
        const auto ptByCen = commonBinning.value("pt_bins_by_centrality", GeneralHelper::Json::array());
        if (cenEdges.size() < 2) {
            throw std::runtime_error("Invalid cen_bins for mode: " + policy.mode);
        }
        if (!ptByCen.is_array() || ptByCen.size() != cenEdges.size() - 1) {
            throw std::runtime_error("pt_bins_by_centrality size must equal cen_bins.size()-1");
        }

        const auto topoSelections = profile.value("data_selection_topology", GeneralHelper::Json::array());
        for (size_t ic = 0; ic + 1 < cenEdges.size(); ++ic) {
            const auto ptEdges = ReadDoubleArray(ptByCen.at(ic));
            if (ptEdges.size() < 2) {
                throw std::runtime_error("Invalid pt_bins_by_centrality row");
            }
            for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
                BinPlanItem item;
                item.mode = policy.mode;
                item.hasCen = true;
                item.hasPt = true;
                item.cenMin = cenEdges[ic];
                item.cenMax = cenEdges[ic + 1];
                item.ptMin = ptEdges[ip];
                item.ptMax = ptEdges[ip + 1];

                if (policy.useTopologyArray && topoSelections.is_array() && ic < topoSelections.size()) {
                    const auto &row = topoSelections.at(ic);
                    if (row.is_array() && ip < row.size() && row.at(ip).is_string()) {
                        item.topologySelection = row.at(ip).get<std::string>();
                    }
                }

                item.label = MakeLabel(item);
                item.snapshotDataPath = BuildDataSnapshotPath(snapshotDir, item);
                item.snapshotMcPath = BuildMcSnapshotPath(snapshotDir, item);
                out.items.push_back(item);

                AddEdge(out.cenEdges, item.cenMin);
                AddEdge(out.cenEdges, item.cenMax);
                AddEdge(out.ptEdges, item.ptMin);
                AddEdge(out.ptEdges, item.ptMax);
            }
        }
    } else if (policy.usePt && policy.useCt) {
        const auto ptEdges = ReadDoubleArray(commonBinning.value("pt_bins", GeneralHelper::Json::array()));
        const auto ctByPt = commonBinning.value("ct_bins_by_pt", GeneralHelper::Json::array());
        if (ptEdges.size() < 2) {
            throw std::runtime_error("Invalid pt_bins for mode: " + policy.mode);
        }
        if (!ctByPt.is_array() || ctByPt.size() != ptEdges.size() - 1) {
            throw std::runtime_error("ct_bins_by_pt size must equal pt_bins.size()-1");
        }

        for (size_t ip = 0; ip + 1 < ptEdges.size(); ++ip) {
            const auto ctEdges = ReadDoubleArray(ctByPt.at(ip));
            if (ctEdges.size() < 2) {
                throw std::runtime_error("Invalid ct_bins_by_pt row");
            }
            for (size_t ict = 0; ict + 1 < ctEdges.size(); ++ict) {
                BinPlanItem item;
                item.mode = policy.mode;
                item.hasPt = true;
                item.hasCt = true;
                item.ptMin = ptEdges[ip];
                item.ptMax = ptEdges[ip + 1];
                item.ctMin = ctEdges[ict];
                item.ctMax = ctEdges[ict + 1];
                item.label = MakeLabel(item);
                item.snapshotDataPath = BuildDataSnapshotPath(snapshotDir, item);
                item.snapshotMcPath = BuildMcSnapshotPath(snapshotDir, item);
                out.items.push_back(item);

                AddEdge(out.ptEdges, item.ptMin);
                AddEdge(out.ptEdges, item.ptMax);
                AddEdge(out.ctEdges, item.ctMin);
                AddEdge(out.ctEdges, item.ctMax);
            }
        }
    } else if (policy.useCt && !policy.usePt) {
        const auto ctEdges = ReadDoubleArray(commonBinning.value("ct_bins_single", GeneralHelper::Json::array()));
        if (ctEdges.size() < 2) {
            throw std::runtime_error("Invalid ct_bins_single for mode: " + policy.mode);
        }
        for (size_t i = 0; i + 1 < ctEdges.size(); ++i) {
            BinPlanItem item;
            item.mode = policy.mode;
            item.hasCt = true;
            item.ctMin = ctEdges[i];
            item.ctMax = ctEdges[i + 1];
            item.label = MakeLabel(item);
            item.snapshotDataPath = BuildDataSnapshotPath(snapshotDir, item);
            item.snapshotMcPath = BuildMcSnapshotPath(snapshotDir, item);
            out.items.push_back(item);

            AddEdge(out.ctEdges, item.ctMin);
            AddEdge(out.ctEdges, item.ctMax);
        }
    } else {
        throw std::runtime_error("Mode policy has unsupported axis combination for: " + policy.mode);
    }

    std::sort(out.cenEdges.begin(), out.cenEdges.end());
    std::sort(out.ptEdges.begin(), out.ptEdges.end());
    std::sort(out.ctEdges.begin(), out.ctEdges.end());
    return out;
}

std::string BinPlanBuilder::FormatEdge(double value) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << value;
    std::string s = os.str();
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    if (s.empty()) s = "0";
    return s;
}

std::string BinPlanBuilder::MakeLabel(const BinPlanItem &item) {
    std::string label;
    if (item.hasCen) {
        label += "cen_" + FormatEdge(item.cenMin) + "_" + FormatEdge(item.cenMax) + "_";
    }
    if (item.hasPt) {
        label += "pt_" + FormatEdge(item.ptMin) + "_" + FormatEdge(item.ptMax) + "_";
    }
    if (item.hasCt) {
        label += "ct_" + FormatEdge(item.ctMin) + "_" + FormatEdge(item.ctMax) + "_";
    }
    if (!label.empty() && label.back() == '_') label.pop_back();
    return label.empty() ? "all" : label;
}

std::string BinPlanBuilder::BuildDataSnapshotPath(const std::string &dir, const BinPlanItem &item) {
    if (dir.empty()) return std::string();
    if (item.hasCen && item.hasPt) {
        return dir + "/data_cen_" + FormatEdge(item.cenMin) + "_" + FormatEdge(item.cenMax) +
               "_pt_" + FormatEdge(item.ptMin) + "_" + FormatEdge(item.ptMax) + ".root";
    }
    if (item.hasPt && item.hasCt) {
        return dir + "/data_pt_" + FormatEdge(item.ptMin) + "_" + FormatEdge(item.ptMax) +
               "_ct_" + FormatEdge(item.ctMin) + "_" + FormatEdge(item.ctMax) + ".root";
    }
    if (item.hasCt) {
        return dir + "/data_ct_" + FormatEdge(item.ctMin) + "_" + FormatEdge(item.ctMax) + ".root";
    }
    return std::string();
}

std::string BinPlanBuilder::BuildMcSnapshotPath(const std::string &dir, const BinPlanItem &item) {
    if (dir.empty()) return std::string();
    if (item.hasCen && item.hasPt) {
        return dir + "/mc_cen_" + FormatEdge(item.cenMin) + "_" + FormatEdge(item.cenMax) +
               "_pt_" + FormatEdge(item.ptMin) + "_" + FormatEdge(item.ptMax) + ".root";
    }
    if (item.hasPt && item.hasCt) {
        return dir + "/mc_pt_" + FormatEdge(item.ptMin) + "_" + FormatEdge(item.ptMax) +
               "_ct_" + FormatEdge(item.ctMin) + "_" + FormatEdge(item.ctMax) + ".root";
    }
    if (item.hasCt) {
        return dir + "/mc_ct_" + FormatEdge(item.ctMin) + "_" + FormatEdge(item.ctMax) + ".root";
    }
    return std::string();
}

std::string BinPlanBuilder::JoinSelection(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return "(" + a + ") && (" + b + ")";
}

} // namespace UnifiedAnalysis
