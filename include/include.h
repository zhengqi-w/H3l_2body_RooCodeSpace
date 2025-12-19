#ifndef ALICE_INCLUDE_H
#define ALICE_INCLUDE_H

// include.h
// 常用物理常量与基本函数头文件
// GitHub Copilot

#include <cmath>
#include <array>

namespace Physics {

// 光速 cm/ps
constexpr double c_cm_per_ps = 0.0299792458; // exact

// 基本常量（SI / 精确值或CODATA近似值）
constexpr double pi        = 3.14159265358979323846;
constexpr double c_light   = 299792458.0;                 // m/s (exact)
constexpr double h_planck  = 6.62607015e-34;              // J s (exact)
constexpr double hbar      = h_planck / (2.0 * pi);       // J s
constexpr double k_B      = 1.380649e-23;                 // J/K (exact)
constexpr double e_charge = 1.602176634e-19;              // C (exact)

// 单位换算
constexpr double eV_to_J   = e_charge;                    // 1 eV = 1.602...e-19 J
constexpr double MeV_to_J  = 1.0e6 * eV_to_J;             // 1 MeV in J
constexpr double J_to_MeV  = 1.0 / MeV_to_J;
constexpr double u_kg      = 1.66053906660e-27;           // atomic mass unit in kg (CODATA)
constexpr double MeV_c2_to_kg = MeV_to_J / (c_light * c_light); // 1 MeV/c^2 in kg (~1.78266e-30 kg)
constexpr double kg_to_MeV_c2 = 1.0 / MeV_c2_to_kg;

// 常用粒子质量（以公斤为单位；原子质量/近似值）
// 注意：某些值为原子质量（包含电子），若需要核质量请根据电子数做修正。
constexpr double mass_electron_kg = 9.1093837015e-31;      // kg
constexpr double mass_proton_kg   = 1.67262192369e-27;    // kg
constexpr double mass_neutron_kg  = 1.67492749804e-27;    // kg

// 常见轻核（原子质量单位近似值）
constexpr double amu_triton        = 3.01604928199;       // 3H 原子质量 (u)
constexpr double amu_helium3       = 3.01602932265;       // 3He 原子质量 (u)
constexpr double amu_deuteron      = 2.013553212745;      // 2H 原子质量 (u), 近似
constexpr double amu_alpha         = 4.00260325413;       // 4He 原子质量 (u)

// 将上面的原子质量转换为 kg（包含电子）
constexpr double mass_triton_kg  = amu_triton  * u_kg;
constexpr double mass_helium3_kg = amu_helium3 * u_kg;
constexpr double mass_deuteron_kg= amu_deuteron * u_kg;
constexpr double mass_alpha_kg   = amu_alpha    * u_kg;

// 以能量单位 (MeV/c^2) 表示
constexpr double mass_electron_MeV = mass_electron_kg * kg_to_MeV_c2;
constexpr double mass_proton_MeV   = mass_proton_kg   * kg_to_MeV_c2;
constexpr double mass_neutron_MeV  = mass_neutron_kg  * kg_to_MeV_c2;
constexpr double mass_triton_MeV   = mass_triton_kg   * kg_to_MeV_c2;
constexpr double mass_helium3_MeV  = mass_helium3_kg  * kg_to_MeV_c2;
constexpr double mass_deuteron_MeV = mass_deuteron_kg * kg_to_MeV_c2;
constexpr double mass_alpha_MeV    = mass_alpha_kg    * kg_to_MeV_c2;

// 简单的四矢量结构（自然单位可选）
struct LorentzVector {
    double t; // energy (or E/c)
    double x;
    double y;
    double z;
    LorentzVector() : t(0), x(0), y(0), z(0) {}
    LorentzVector(double Et, double px, double py, double pz) : t(Et), x(px), y(py), z(pz) {}
};

// 常用函数（内联实现，便于直接包含使用）

// 单位换算简便函数
inline constexpr double MeV_to_Joule(double MeV) { return MeV * MeV_to_J; }
inline constexpr double Joule_to_MeV(double J)   { return J * J_to_MeV; }
inline constexpr double u_to_kg(double u)         { return u * u_kg; }
inline constexpr double kg_to_u(double kg)        { return kg / u_kg; }
inline constexpr double MeV_c2_to_kg_f(double MeV_c2) { return MeV_c2 * MeV_c2_to_kg; }

// 经典与相对论能量/动量转换（有单位时请注意 E 为 J or MeV 的一致）
inline double kinetic_energy_nonrel(double mass_kg, double p_mag) {
    // p_mag in kg*m/s, returns kinetic energy in J (non-relativistic)
    return p_mag * p_mag / (2.0 * mass_kg);
}
inline double momentum_from_ke_nonrel(double mass_kg, double KE_J) {
    return std::sqrt(2.0 * mass_kg * KE_J);
}

// 相对论性能量与动量（质量以 kg, momentum 以 kg*m/s, energy 以 J）
inline double energy_rel_from_p(double mass_kg, double p_mag) {
    return std::sqrt((mass_kg * c_light * c_light) * (mass_kg * c_light * c_light) + p_mag * p_mag * c_light * c_light);
}
inline double momentum_mag_from_E(double mass_kg, double E_J) {
    double m2c4 = (mass_kg * c_light * c_light) * (mass_kg * c_light * c_light);
    double pc = std::sqrt(std::max(0.0, E_J * E_J - m2c4));
    return pc / c_light;
}

// 速度参数 beta, gamma given momentum or energy
inline double beta_from_p(double mass_kg, double p_mag) {
    double E = energy_rel_from_p(mass_kg, p_mag);
    return (p_mag * c_light) / E;
}
inline double gamma_from_beta(double beta) {
    return 1.0 / std::sqrt(1.0 - beta * beta);
}

// 四矢量不变量（使用 E in J, p in kg*m/s）
// 计算不变量质量（返回质量 in kg）
inline double invariant_mass_kg(const LorentzVector& a, const LorentzVector& b) {
    // t = E (J), x,y,z = px,py,pz (kg*m/s)
    double E = a.t + b.t;
    double px = a.x + b.x;
    double py = a.y + b.y;
    double pz = a.z + b.z;
    double p2 = px*px + py*py + pz*pz;
    double m2c4 = E*E - (p2 * c_light * c_light);
    if (m2c4 <= 0.0) return 0.0;
    return std::sqrt(m2c4) / (c_light * c_light);
}

// 快速计算转动质量（transverse mass）给 E, px, py (J, kg*m/s)
inline double transverse_mass_kg(double E, double px, double py) {
    double pt2 = px*px + py*py;
    double m2c4 = E*E - pt2 * c_light * c_light;
    if (m2c4 <= 0.0) return 0.0;
    return std::sqrt(m2c4) / (c_light * c_light);
}

} // namespace Physics

#endif // ALICE_INCLUDE_H