// Helper functions to unpack ITS cluster sizes stored as 4-bit nibbles per layer.
// Each layer occupies 4 bits in the packed integer. We assume up to 7 layers.
#include <cstdint>
extern "C" {
unsigned int CountITSHits(unsigned long long packed) {
    unsigned int n = 0;
    for (int i = 0; i < 7; i++) {
        unsigned int val = (unsigned int)((packed >> (4 * i)) & 0xFULL);
        if (val > 0) ++n;
    }
    return n;
}

double AvgITSClusterSize(unsigned long long packed) {
    unsigned int n = 0;
    unsigned int sum = 0;
    for (int i = 0; i < 7; i++) {
        unsigned int val = (unsigned int)((packed >> (4 * i)) & 0xFULL);
        if (val > 0) {
            sum += val;
            ++n;
        }
    }
    if (n == 0) return 0.0;
    return static_cast<double>(sum) / static_cast<double>(n);
}
} // extern "C"
