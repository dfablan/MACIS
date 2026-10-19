#include <complex>
#include <macis/impurity_solver.hpp>
#include <macis/util/fock_matrices.hpp>

#include "ut_common.hpp"

// Spin average of the impurity Green's function when NALPHA != NBETA
// (evaluate_GF, GF.SPIN_AVERAGE; symmetry-sector-solve.md, sec. 3.9).

namespace {

constexpr size_t NB = 64;
using params_t = macis::impurity_params<NB>;
using gf_t = std::vector<std::vector<std::complex<double>>>;

// Single-orbital Anderson impurity (orbital 0, U = 4, eps_d = -2) with three
// bath levels. `zeeman` adds -/+ zeeman to the spin-up/down impurity level
// (spin_dep), which breaks spin-flip symmetry.
params_t make_anderson(size_t na, size_t nb, double zeeman = 0.0) {
  const size_t n = 4;
  params_t p{};
  p.norb = p.n_active = n;
  p.n_inactive = 0;
  p.n_imp = 1;
  p.nbands = 1;
  p.nalpha = na;
  p.nbeta = nb;
  p.E_core = p.E_inactive = 0.0;
  p.just_singles = false;
  p.spin_dep = zeeman != 0.0;
  p.ci_exp = CIExpansion::CAS;
  p.T.assign(n * n, 0.0);
  p.V.assign(n * n * n * n, 0.0);
  const double eb[3] = {-1.5, -0.5, 1.0};
  p.T[0] = -2.0;
  for(size_t k = 1; k < n; ++k) {
    p.T[k + k * n] = eb[k - 1];
    p.T[k] = p.T[k * n] = 0.5;
  }
  p.V[0] = 4.0;
  p.Td = p.T;
  p.T[0] -= zeeman;
  p.Td[0] += zeeman;

  p.mcscf_settings.ci_max_subspace = 100;
  p.mcscf_settings.ci_res_tol = 1e-12;
  p.asci_settings.nrots = 0;
  p.occs.assign(n, 0.0);
  p.orb_rot.assign(n * n, 0.0);
  for(size_t i = 0; i < n; ++i) p.orb_rot[i * n + i] = 1.0;
  p.T_active.resize(n * n);
  p.Td_active.resize(n * n);
  p.V_active.resize(n * n * n * n);
  p.F_inactive.resize(n * n);
  p.Fd_inactive.resize(n * n);
  p.compute_asci_E0 = true;
  macis::active_hamiltonian(NumOrbital(n), NumActive(n), NumInactive(0),
                            p.T.data(), n, p.V.data(), n, p.F_inactive.data(),
                            n, p.T_active.data(), n, p.V_active.data(), n);
  if(p.spin_dep)
    macis::active_hamiltonian(NumOrbital(n), NumActive(n), NumInactive(0),
                              p.Td.data(), n, p.V.data(), n,
                              p.Fd_inactive.data(), n, p.Td_active.data(), n,
                              p.V_active.data(), n);
  return p;
}

// GF of the impurity orbital for the (orbital, spin) list comp/up
gf_t impurity_gf(params_t& p, double E0, std::vector<int> comp,
                 std::vector<bool> up, bool spin_average) {
  const size_t n = p.n_active;
  macis::SDBuildHamiltonianGenerator<NB> H(
      macis::matrix_span<double>(p.T_active.data(), n, n),
      macis::rank4_span<double>(p.V_active.data(), n, n, n, n));
  if(p.spin_dep)
    H.ReadTdo(macis::matrix_span<double>(p.Td_active.data(), n, n));
  macis::GFSettings s;
  s.GF_orbs_basis = comp;
  s.is_up_basis = up;
  s.GF_orbs_comp = comp;
  s.is_up_comp = up;
  s.imag_freq = false;
  s.wmin = -3.0;
  s.wmax = 3.0;
  s.nws = 7;
  s.eta = 0.2;
  s.nLanIts = 200;
  s.trunc_size = 100;
  s.tot_SD = 4;
  s.GFseedThres = 0.0;
  s.asThres = 0.0;
  s.spin_average = spin_average;
  return macis::evaluate_GF<NB>(E0, p, H, s);
}

double max_diff(const gf_t& a, size_t ia, const gf_t& b, size_t ib) {
  double d = 0.;
  for(size_t iw = 0; iw < a.size(); ++iw)
    d = std::max(d, std::abs(a[iw][ia] - b[iw][ib]));
  return d;
}

}  // namespace

TEST_CASE("GF spin average for NALPHA != NBETA") {
  // (2,1) and its spin flip (1,2): exactly degenerate partners
  auto pu = make_anderson(2, 1);
  const double Eu = macis::SolveImpurityED<NB>(pu);
  auto pd = make_anderson(1, 2);
  const double Ed = macis::SolveImpurityED<NB>(pd);
  REQUIRE(Eu == Approx(Ed).margin(1e-10));

  // Unaveraged G_up of each state: they differ (the spin bias)
  const auto Gu_up = impurity_gf(pu, Eu, {0}, {true}, false);
  const auto Gd_up = impurity_gf(pd, Ed, {0}, {true}, false);
  const auto Gu_dn = impurity_gf(pu, Eu, {0}, {false}, false);
  CHECK(max_diff(Gu_up, 0, Gd_up, 0) > 1e-3);
  // Spin flip: G_up of (1,2) is G_down of (2,1)
  CHECK(max_diff(Gd_up, 0, Gu_dn, 0) < 1e-8);

  SECTION("one channel requested: the opposite one is computed and averaged") {
    const auto G = impurity_gf(pu, Eu, {0}, {true}, true);
    // The ensemble over the doublet: (G_up[m=+1/2] + G_up[m=-1/2]) / 2
    for(size_t iw = 0; iw < G.size(); ++iw)
      CHECK(std::abs(G[iw][0] - 0.5 * (Gu_up[iw][0] + Gd_up[iw][0])) < 1e-8);
    // The same whichever member was solved, and whichever channel requested
    CHECK(max_diff(G, 0, impurity_gf(pd, Ed, {0}, {true}, true), 0) < 1e-8);
    CHECK(max_diff(G, 0, impurity_gf(pu, Eu, {0}, {false}, true), 0) < 1e-8);
  }

  SECTION("both channels requested: averaged by permutation, no second run") {
    // n = 2 entries: (0 up) at 0, (0 down) at 1. Compared with the raw GF
    // of the same two-entry run, not with separate one-spin runs: the band
    // Lanczos does not deflate, so on this tiny model (16 particle
    // determinants per spin) the two-seed run is not exact (~5e-3 at the
    // band edges, against an exact Lehmann sum) while one-seed runs are.
    const auto G2 = impurity_gf(pu, Eu, {0, 0}, {true, false}, true);
    const auto R2 = impurity_gf(pu, Eu, {0, 0}, {true, false}, false);
    for(size_t iw = 0; iw < G2.size(); ++iw) {
      const auto avg = 0.5 * (R2[iw][0] + R2[iw][3]);
      CHECK(std::abs(G2[iw][0] - avg) < 1e-12);
      CHECK(std::abs(G2[iw][3] - avg) < 1e-12);
    }
    for(size_t iw = 0; iw < G2.size(); ++iw) {
      CHECK(std::abs(G2[iw][1]) < 1e-12);  // no spin-flip element
      CHECK(std::abs(G2[iw][2]) < 1e-12);
    }
  }

  SECTION("even N with NALPHA == NBETA: unchanged") {
    auto p = make_anderson(2, 2);
    const double E = macis::SolveImpurityED<NB>(p);
    CHECK(max_diff(impurity_gf(p, E, {0}, {true}, true), 0,
                   impurity_gf(p, E, {0}, {true}, false), 0) == 0.0);
  }

  SECTION("spin-dependent one-body terms: no average") {
    auto p = make_anderson(2, 1, 0.3);
    const double E = macis::SolveImpurityED<NB>(p);
    CHECK(max_diff(impurity_gf(p, E, {0}, {true}, true), 0,
                   impurity_gf(p, E, {0}, {true}, false), 0) == 0.0);
  }
}
