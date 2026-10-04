#include <complex>
#include <filesystem>
#include <macis/impurity_solver.hpp>
#include <macis/parity_sectors.hpp>
#include <macis/util/fock_matrices.hpp>
#include <sstream>

#include "ut_common.hpp"

// Band-orbit average of the impurity Green's function (evaluate_GF,
// GF.BAND_AVERAGE; symmetry-sector-solve.md, sec. 3.9): at odd N with two
// degenerate bands the parity sectors (o,e) and (e,o) are exchanged by the
// band swap, the solved state is one of the two partners, and its GF is
// band-polarized.

namespace {

constexpr size_t NB = 64;
// Separate solves of the two partner sectors are exact mirror images (they
// agree to 1e-12 here); the permutation identities inside one run as well
constexpr double PARTNER_TOL = 1e-10;
using params_t = macis::impurity_params<NB>;
using gf_t = std::vector<std::vector<std::complex<double>>>;

// Two-band Kanamori impurity (orbitals 0, 1) with bath levels 2..5 coupled to
// band k % 2, both bands identical (the degenerate model of
// tests/parity_sectors.cxx), so the band swap 0<->1, 2<->3, 4<->5 is exact.
params_t make_model(size_t na, size_t nb) {
  const size_t n_imp = 2, n = 6;
  const double U = 6.0, J = 0.8, eps_d = -3.0;
  params_t p{};
  p.norb = p.n_active = n;
  p.n_inactive = 0;
  p.n_imp = n_imp;
  p.nbands = 2;
  p.nalpha = na;
  p.nbeta = nb;
  p.E_core = p.E_inactive = 0.0;
  p.just_singles = false;
  p.spin_dep = false;
  p.ci_exp = CIExpansion::ASCI;
  p.T.assign(n * n, 0.0);
  p.Td.assign(n * n, 0.0);
  p.V.assign(n * n * n * n, 0.0);
  const double eps_b[4] = {-2.0, -2.0, 0.6, 0.6};
  for(size_t k = 0; k < 4; ++k) {
    const size_t b = n_imp + k, band = k % 2;
    p.T[b + b * n] = eps_b[k];
    p.T[band + b * n] = p.T[b + band * n] = 0.5;
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
      V(m, mp, m, mp) = J;  // pair hopping: band parities, not counts
    }
  }

  p.mcscf_settings.ci_max_subspace = 200;
  p.mcscf_settings.ci_res_tol = 1e-10;
  // Every parity sector has <= 200 determinants: the solves are exact
  p.asci_settings.ntdets_max = 1000;
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
  p.asci_wfn_fname = "";
  p.compute_asci_E0 = true;
  p.asci_E0 = 0.0;
  return p;
}

struct Opts {
  bool parity = true;  // ASCI.PARITY_SOLVE
  bool sym = true;     // ASCI.SYMMETRIZE_DETS with the band swap
  std::vector<int> only;
};

params_t make(size_t na, size_t nb, const Opts& o) {
  auto p = make_model(na, nb);
  if(o.sym) {
    p.asci_settings.symmetrize_dets = true;
    p.asci_settings.sym_group =
        std::make_shared<std::vector<std::vector<uint32_t>>>(
            std::vector<std::vector<uint32_t>>{{1, 0, 3, 2, 5, 4}});
  }
  if(o.parity) macis::setup_parity_sectors<NB>(p, 1e-10);
  p.parity_only = o.only;
  const size_t n = p.norb;
  macis::active_hamiltonian(NumOrbital(n), NumActive(n), NumInactive(0),
                            p.T.data(), n, p.V.data(), n, p.F_inactive.data(),
                            n, p.T_active.data(), n, p.V_active.data(), n);
  return p;
}

// Impurity GF for the spin-up (orbital) list comp
gf_t impurity_gf(params_t& p, double E0, std::vector<int> comp,
                 bool band_average, bool spin_average = false) {
  const size_t n = p.n_active;
  macis::SDBuildHamiltonianGenerator<NB> H(
      macis::matrix_span<double>(p.T_active.data(), n, n),
      macis::rank4_span<double>(p.V_active.data(), n, n, n, n));
  macis::GFSettings s;
  const std::vector<bool> up(comp.size(), true);
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
  s.band_average = band_average;
  return macis::evaluate_GF<NB>(E0, p, H, s);
}

double max_diff(const gf_t& a, size_t ia, const gf_t& b, size_t ib) {
  double d = 0.;
  for(size_t iw = 0; iw < a.size(); ++iw)
    d = std::max(d, std::abs(a[iw][ia] - b[iw][ib]));
  return d;
}

bool has(const std::string& out, const char* s) {
  return out.find(s) != std::string::npos;
}

struct CoutCapture {
  std::ostringstream buf;
  std::streambuf* old;
  CoutCapture() : old(std::cout.rdbuf(buf.rdbuf())) {}
  ~CoutCapture() { std::cout.rdbuf(old); }
  std::string str() const { return buf.str(); }
};

void barrier() { MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);) }

// The solvers write wfn/rdm files; keep them out of the build tree
struct ScratchDir {
  std::filesystem::path old = std::filesystem::current_path();
  ScratchDir() {
    const auto dir =
        std::filesystem::temp_directory_path() / "macis_gf_band_average";
    int rank = 0;
    MACIS_MPI_CODE(MPI_Comm_rank(MPI_COMM_WORLD, &rank);)
    if(rank == 0) std::filesystem::create_directories(dir);
    barrier();
    std::filesystem::current_path(dir);
  }
  ~ScratchDir() {
    barrier();
    std::filesystem::current_path(old);
  }
};

}  // namespace

TEST_CASE("GF band average over the parity-sector orbit") {
  ScratchDir scratch;
  // (3,2): odd N, the winner is (o,e) or (e,o), exact partners
  auto p = make(3, 2, {});
  double E;
  {
    CoutCapture cap;
    E = macis::SolveImpurityASCI<NB>(p);
    REQUIRE(has(cap.str(), "PARITY_TIE"));
  }
  REQUIRE(p.parity_tied_keys.size() == 1);

  // Each partner sector alone, raw GF of both impurity orbitals
  auto pa = make(3, 2, Opts{true, true, {1, 0}});
  auto pb = make(3, 2, Opts{true, true, {0, 1}});
  const double Ea = macis::SolveImpurityASCI<NB>(pa);
  const double Eb = macis::SolveImpurityASCI<NB>(pb);
  REQUIRE(Ea == Approx(E).margin(1e-8));
  REQUIRE(Eb == Approx(E).margin(1e-8));
  gf_t Ra, Rb;
  {
    CoutCapture cap;
    Ra = impurity_gf(pa, Ea, {0, 1}, false);
    Rb = impurity_gf(pb, Eb, {0, 1}, false);
    // A list the swap does not map onto itself (bath orbital 2 -> 3) is
    // reported and left as is
    impurity_gf(pa, Ea, {0, 2}, true);
    CHECK(has(cap.str(), "GF_BAND_AVERAGE none: the band permutation maps"));
  }
  // Each partner is band-polarized, and the swap exchanges them
  CHECK(max_diff(Ra, 0, Ra, 3) > 1e-3);
  CHECK(max_diff(Ra, 0, Rb, 3) < PARTNER_TOL);
  CHECK(max_diff(Ra, 3, Rb, 0) < PARTNER_TOL);

  SECTION("averaged: (G_A + G_B) / 2, the same for either partner") {
    CoutCapture cap;
    const auto G = impurity_gf(p, E, {0, 1}, true);
    const auto R = impurity_gf(p, E, {0, 1}, false);
    CHECK(has(cap.str(), "GF_BAND_AVERAGE parity sector"));
    CHECK(has(cap.str(), "m = 2 degenerate partners"));
    CHECK(has(cap.str(), "WARNING: GF_BAND_AVERAGE off"));
    // Exact permutation average of the raw GF of the same run
    for(size_t iw = 0; iw < G.size(); ++iw) {
      const auto avg = 0.5 * (R[iw][0] + R[iw][3]);
      CHECK(std::abs(G[iw][0] - avg) < 1e-12);
      CHECK(std::abs(G[iw][3] - avg) < 1e-12);
      CHECK(std::abs(G[iw][1]) < 1e-12);  // no band-mixing element
      CHECK(std::abs(G[iw][2]) < 1e-12);
    }
    // ... which is the ensemble of the two partner sectors
    for(size_t iw = 0; iw < G.size(); ++iw)
      for(size_t k : {0, 3})
        CHECK(std::abs(G[iw][k] - 0.5 * (Ra[iw][k] + Rb[iw][k])) <
              PARTNER_TOL);
    // Either partner gives the same average
    const auto Ga = impurity_gf(pa, Ea, {0, 1}, true);
    const auto Gb = impurity_gf(pb, Eb, {0, 1}, true);
    CHECK(max_diff(Ga, 0, Gb, 0) < 1e-8);
    CHECK(max_diff(Ga, 3, Gb, 3) < 1e-8);
  }

  SECTION("with the spin average as well: still band-symmetric") {
    const auto G = impurity_gf(p, E, {0, 1}, true, true);
    for(size_t iw = 0; iw < G.size(); ++iw)
      CHECK(std::abs(G[iw][0] - G[iw][3]) < 1e-12);
  }

  SECTION("no symmetry group: the tie is warned, the GF left as is") {
    auto q = make(3, 2, Opts{true, false, {}});
    const double Eq = macis::SolveImpurityASCI<NB>(q);
    CoutCapture cap;
    const auto G = impurity_gf(q, Eq, {0, 1}, true);
    CHECK(has(cap.str(), "WARNING: GF_BAND_AVERAGE none: parity sector"));
    CHECK(has(cap.str(), "no symmetry group is available"));
    CHECK(max_diff(G, 0, impurity_gf(q, Eq, {0, 1}, false), 0) == 0.0);
  }

  SECTION("no parity labels: the broken band swap is warned") {
    auto q = make(3, 2, Opts{false, true, {}});
    const double Eq = macis::SolveImpurityASCI<NB>(q);
    CoutCapture cap;
    const auto G = impurity_gf(q, Eq, {0, 1}, true);
    CHECK(has(cap.str(), "is not invariant under the SYMMETRIZE_DETS group"));
    CHECK(max_diff(G, 0, impurity_gf(q, Eq, {0, 1}, false), 0) == 0.0);
  }
}

TEST_CASE("GF band average: invariant sectors are untouched") {
  ScratchDir scratch;
  SECTION("even N: the (o,o) winner is band-swap invariant") {
    auto p = make(3, 3, {});
    const double E = macis::SolveImpurityASCI<NB>(p);
    CoutCapture cap;
    const auto G = impurity_gf(p, E, {0, 1}, true);
    CHECK(!has(cap.str(), "GF_BAND_AVERAGE"));
    CHECK(max_diff(G, 0, impurity_gf(p, E, {0, 1}, false), 0) == 0.0);
    CHECK(max_diff(G, 0, G, 3) < 1e-8);  // symmetric on its own
  }

  SECTION("even N without parity labels: invariant, nothing reported") {
    auto p = make(3, 3, Opts{false, true, {}});
    const double E = macis::SolveImpurityASCI<NB>(p);
    CoutCapture cap;
    impurity_gf(p, E, {0, 1}, true);
    CHECK(!has(cap.str(), "GF_BAND_AVERAGE"));
  }
}
