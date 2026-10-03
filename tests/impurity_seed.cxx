#include <macis/impurity_solver.hpp>
#include <macis/util/fock_matrices.hpp>

#include "ut_common.hpp"

namespace {

constexpr size_t NB = 64;
using params_t = macis::impurity_params<NB>;

// Two-band Kanamori impurity (orbitals 0, 1) with a band-diagonal bath of four
// levels, listed in the FCIDUMP order a bath fit might emit them -- not by
// energy:
//
//   orbital   2      3      4      5
//   eps     +1.9   -2.0   -0.7   +0.6
//   band      1      0      1      0
//
// Without cross-band hybridization H conserves the parity of each band's
// electron count, so an ASCI expansion never leaves the parity sector of its
// seed. At (NALPHA, NBETA) = (3, 2) the unpaired alpha electron goes to
// orbital 2 (band 1, eps = +1.9) under the raw-index fill and to orbital 3
// (band 0, eps = -2.0) under the energy fill: different sectors.
// Exact sector minima (parity-sector-toy-ed.py 6 0.8 -3 3 2, same model up to
// the orbital order): (N_0, N_1) = (even, odd) -8.700141181,
// (odd, even) -8.865079189 = the ground state.
params_t make_permuted_model(size_t na, size_t nb) {
  const size_t n_imp = 2, n = 6;
  const double U = 6.0, J = 0.8, eps_d = -3.0;
  params_t p{};
  p.norb = p.n_active = n;
  p.n_inactive = 0;
  p.n_imp = n_imp;
  p.nbands = 2;
  p.nalpha = na;
  p.nbeta = nb;
  p.E_core = 0.0;
  p.E_inactive = 0.0;
  p.just_singles = false;
  p.spin_dep = false;
  p.ci_exp = CIExpansion::ASCI;
  p.T.assign(n * n, 0.0);
  p.Td.assign(n * n, 0.0);
  p.V.assign(n * n * n * n, 0.0);

  const double eps[4] = {1.9, -2.0, -0.7, 0.6};
  const size_t band[4] = {1, 0, 1, 0};
  for(size_t k = 0; k < 4; ++k) {
    const size_t b = n_imp + k;
    p.T[b + b * n] = eps[k];
    p.T[band[k] + b * n] = p.T[b + band[k] * n] = 0.5;
  }
  for(size_t m = 0; m < n_imp; ++m) p.T[m + m * n] = eps_d;
  auto V = [&](size_t a, size_t b, size_t c, size_t d) -> double& {
    return p.V[a + b * n + c * n * n + d * n * n * n];
  };
  for(size_t m = 0; m < n_imp; ++m) {
    V(m, m, m, m) = U;
    for(size_t mp = 0; mp < n_imp; ++mp) {
      if(mp == m) continue;
      V(m, m, mp, mp) = U - 2 * J;
      V(m, mp, mp, m) = J;
      V(m, mp, m, mp) = J;
    }
  }

  p.mcscf_settings.ci_max_subspace = 200;
  p.mcscf_settings.ci_res_tol = 1e-10;
  p.asci_settings.ntdets_max = 1000;  // each parity sector has 150 dets
  p.asci_settings.ntdets_min = 10;
  p.asci_settings.ncdets_max = 400;
  p.asci_settings.max_refine_iter = 6;
  p.asci_settings.refine_energy_tol = 1e-10;
  p.asci_settings.nrots = 0;
  p.asci_settings.just_singles = false;

  p.occs.assign(n, 0.0);
  p.orb_rot.assign(n * n, 0.0);
  for(size_t i = 0; i < n; ++i) p.orb_rot[i * n + i] = 1.0;
  p.T_active.resize(n * n);
  p.Td_active.resize(n * n);
  p.V_active.resize(n * n * n * n);
  p.F_inactive.resize(n * n);
  p.Fd_inactive.resize(n * n);
  macis::active_hamiltonian(
      NumOrbital(n), NumActive(n), NumInactive(0), p.T.data(), n, p.V.data(),
      n, p.F_inactive.data(), n, p.T_active.data(), n, p.V_active.data(), n);
  p.asci_wfn_fname = "";
  p.compute_asci_E0 = true;
  p.asci_E0 = 0.0;
  return p;
}

// Electrons in band 0 (orbitals 0, 3, 5), spins summed
size_t band0_count(const macis::wfn_t<NB>& d) {
  size_t c = 0;
  for(size_t q : {0, 3, 5}) c += d[q] + d[q + NB / 2];
  return c;
}

}  // namespace

TEST_CASE("ASCI impurity seed ordering") {
  const double E_odd_even = -8.865079189;   // ground state
  const double E_even_odd = -8.700141181;   // the other parity sector

  SECTION("energy-ordered seed (default) reaches the ground state") {
    auto p = make_permuted_model(3, 2);
    REQUIRE(p.asci_settings.hf_by_energy);
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    REQUIRE(E == Approx(E_odd_even).margin(1e-7));
    for(const auto& d : p.dets) REQUIRE(band0_count(d) % 2 == 1);
  }

  SECTION("energy-ordered seed, NROTS = 0 solver without rotations") {
    auto p = make_permuted_model(3, 2);
    const double E = macis::SolveImpurityASCI<NB>(p);
    REQUIRE(E == Approx(E_odd_even).margin(1e-7));
  }

  SECTION("raw-index seed stays in the parity sector of orbital 2") {
    // Documents why the seed order matters: the raw fill puts the unpaired
    // electron in band 1 and the expansion never leaves that sector, so it
    // converges to the other sector's minimum, 0.165 Ha above the ground
    // state. If this starts reaching E_odd_even, the solver has learned to
    // cross parity sectors and this section should be revisited.
    auto p = make_permuted_model(3, 2);
    p.asci_settings.hf_by_energy = false;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    REQUIRE(E == Approx(E_even_odd).margin(1e-7));
    for(const auto& d : p.dets) REQUIRE(band0_count(d) % 2 == 0);
  }

  SECTION("both rules give identical runs when they pick the same orbitals") {
    // At (2, 2) both rules occupy orbitals 0 and 1 (eps_d = -3 is the lowest
    // level and they are also the first two indices), so the seed and hence
    // the whole run must be identical, bit for bit.
    auto p1 = make_permuted_model(2, 2);
    auto p2 = make_permuted_model(2, 2);
    p2.asci_settings.hf_by_energy = false;
    const double E1 = macis::SolveImpurityASCI_rot<NB>(p1);
    const double E2 = macis::SolveImpurityASCI_rot<NB>(p2);
    REQUIRE(E1 == E2);
    REQUIRE(p1.dets == p2.dets);
  }
}
