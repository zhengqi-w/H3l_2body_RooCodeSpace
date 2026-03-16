#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include <iostream>
#include <string>

namespace UnifiedAnalysis {

class OutputWriter {
public:
    explicit OutputWriter(std::string outDir) : outDir_(std::move(outDir)) {}

    void WriteRunSummary(const std::string &mode, size_t nBins) const {
        std::cout << "[OutputWriter] mode=" << mode << ", bins=" << nBins
                  << ", outDir=" << outDir_ << std::endl;
    }

private:
    std::string outDir_;
};

} // namespace UnifiedAnalysis

#endif // OUTPUT_WRITER_H
